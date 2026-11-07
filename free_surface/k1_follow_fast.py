"""The V0 follow map rebuilt per cut as one pipeline (11-06), the same map as FollowMapGPU2 built from k1_follow_warp's rows
(the explicit run's path before), without its round trips:
  one wall list per cut   the walls by cell (stable argsort, counts by scatter_add: no host sync), shared by the column
                          estimate, the fits and the rows; per entry the neighbour, the shared area and A / L
  column estimate         k1_column_warp's sort estimate on that list: every coefficient emitted (bounds-checked), radix
                          sorted on a fixed capacity padded with a sentinel key (no host count), the runs summed
  rows                    level 1 the tiny cells, level 2 the cells over the target not slaved at level 1: k_fit on the
                          level's mask, the rows (k1_follow_warp.k_rows' rule) written per particle, level 1 into scratch
                          slots, level 2 into the final slots; the level-1 rows composed onto the final masters in place
  pack                    CSR by prefix sums (ranks, offsets), CSC by counts and atomic cursors
  one host sync           ns, nnz, the column's count and the overflow flags read together
  S and its factor        S = E D^-1 E^T + diag(1 / m_s) (k_gm_Sasm), Cholesky without the error check (the info read at
                          the next sync)
  the solve               M_r^-1 P^T r1 and M_r^-1 P^T r2 with one triangular solve on six columns (accel2): the explicit
                          step's projection of v and its acceleration together, the projection's energy change still known
Graphs (P, P^T, the solve's two halves) captured once; the slaved count read from an array. Checked against FollowMapGPU2
by merged_drop.py --replay-mode fast_check."""
import torch
import warp as wp

from free_surface.k1_column_warp import k_count
from free_surface.k1_follow import FM_NB, k_fm_Et, k_fm_Pt, k_gm_E, k_gm_P, k_gm_S, k_gm_Sasm
from free_surface.k1_follow_warp import k_fit

f64, i32 = torch.float64, torch.int32
SENT = 2**31-1


@wp.kernel(enable_backward=False)
def k_rows_n(x: wp.array(dtype=wp.vec3d), off: wp.array(dtype=wp.int32), nbr: wp.array(dtype=wp.int32), na: wp.array(dtype=wp.float64),
             nw: wp.array(dtype=wp.float64), sl: wp.array(dtype=wp.int32), Mi: wp.array(dtype=wp.mat33d), ok: wp.array(dtype=wp.int32),
             flag: wp.array(dtype=wp.int32), cap: int, wmax: wp.float64,
             rk: wp.array2d(dtype=wp.int32), rw: wp.array2d(dtype=wp.float64), rn: wp.array(dtype=wp.int32), over: wp.array(dtype=wp.int32)):
    """k1_follow_warp.k_rows for the particles flagged, one thread per particle, the row into its own slots"""
    s = wp.tid()
    rn[s] = 0
    if flag[s] == 0:
        return
    a1 = wp.float64(-1.0)
    a2 = wp.float64(-1.0)
    a3 = wp.float64(-1.0)
    k1 = int(-1)
    k2 = int(-1)
    k3 = int(-1)
    for e in range(off[s], off[s+1]):
        k = nbr[e]
        if sl[k] != 0:
            continue
        a = na[e]
        if a > a1 or (a == a1 and k > k1):
            a3 = a2
            k3 = k2
            a2 = a1
            k2 = k1
            a1 = a
            k1 = k
        elif a > a2 or (a == a2 and k > k2):
            a3 = a2
            k3 = k2
            a2 = a
            k2 = k
        elif a > a3 or (a == a3 and k > k3):
            a3 = a
            k3 = k
    best = int(-1)
    bestm = wp.float64(1e300)
    for ci in range(3):
        k = k1
        if ci == 1:
            k = k2
        elif ci == 2:
            k = k3
        if k < 0:
            continue
        if ok[k] == 0:
            continue
        b = Mi[k]@(x[s]-x[k])
        csum = wp.float64(0.0)
        mx = wp.float64(0.0)
        for e in range(off[k], off[k+1]):
            o = nbr[e]
            if sl[o] == 0:
                c = nw[e]*wp.dot(x[o]-x[k], b)
                csum = csum+c
                mx = wp.max(mx, wp.abs(c))
        mx = wp.max(mx, wp.abs(wp.float64(1.0)-csum))
        if mx < bestm:
            bestm = mx
            best = k
        if mx <= wmax:
            break
    if best < 0:
        return
    k = best
    b = Mi[k]@(x[s]-x[k])
    csum = wp.float64(0.0)
    m = int(1)
    for e in range(off[k], off[k+1]):
        o = nbr[e]
        if sl[o] == 0:
            c = nw[e]*wp.dot(x[o]-x[k], b)
            csum = csum+c
            if m < cap:
                rk[s, m] = o
                rw[s, m] = c
                m += 1
            else:
                over[0] = 1
    rk[s, 0] = k
    rw[s, 0] = wp.float64(1.0)-csum
    rn[s] = m


@wp.func
def mrg(Fk: wp.array2d(dtype=wp.int32), Fw: wp.array2d(dtype=wp.float64), p: int, c: int, k: int, w: wp.float64, cap: int,
        over: wp.array(dtype=wp.int32)):
    for q in range(c):
        if Fk[p, q] == k:
            Fw[p, q] = Fw[p, q]+w
            return c
    if c < cap:
        Fk[p, c] = k
        Fw[p, c] = w
        return c+1
    over[1] = 1
    return c


@wp.kernel(enable_backward=False)
def k_compose2(T1k: wp.array2d(dtype=wp.int32), T1w: wp.array2d(dtype=wp.float64), T1n: wp.array(dtype=wp.int32), lev1: wp.array(dtype=wp.int32),
               s2: wp.array(dtype=wp.int32), cap: int, Fk: wp.array2d(dtype=wp.int32), Fw: wp.array2d(dtype=wp.float64), Fn: wp.array(dtype=wp.int32),
               over: wp.array(dtype=wp.int32)):
    """a level-1 row onto the final masters: an entry on a level-2 particle replaced by its row times w, equal masters merged"""
    p = wp.tid()
    if lev1[p] == 0:
        return
    c = int(0)
    for e in range(T1n[p]):
        k = T1k[p, e]
        w = T1w[p, e]
        if s2[k] != 0:
            for f in range(Fn[k]):
                c = mrg(Fk, Fw, p, c, Fk[k, f], w*Fw[k, f], cap, over)
        else:
            c = mrg(Fk, Fw, p, c, k, w, cap, over)
    Fn[p] = c


@wp.kernel(enable_backward=False)
def k_pack_n(slv: wp.array(dtype=wp.int32), rank: wp.array(dtype=wp.int32), rowoff: wp.array(dtype=wp.int32), Fk: wp.array2d(dtype=wp.int32),
             Fw: wp.array2d(dtype=wp.float64), Fn: wp.array(dtype=wp.int32), sid: wp.array(dtype=int), ptr: wp.array(dtype=int),
             col: wp.array(dtype=int), val: wp.array(dtype=wp.float64)):
    p = wp.tid()
    if slv[p] == 0:
        return
    r = rank[p]-1
    o = rowoff[p]
    sid[r] = p
    ptr[r] = o
    for q in range(Fn[p]):
        col[o+q] = Fk[p, q]
        val[o+q] = Fw[p, q]


@wp.kernel(enable_backward=False)
def k_tail(ptr: wp.array(dtype=int), nsd: wp.array(dtype=wp.int32), nnzd: wp.array(dtype=wp.int32), nsa: wp.array(dtype=int)):
    ptr[nsd[0]] = nnzd[0]
    nsa[0] = nsd[0]


@wp.kernel(enable_backward=False)
def k_ccount(slv: wp.array(dtype=wp.int32), Fk: wp.array2d(dtype=wp.int32), Fn: wp.array(dtype=wp.int32), ccnt: wp.array(dtype=wp.int32)):
    p = wp.tid()
    if slv[p] == 0:
        return
    for q in range(Fn[p]):
        wp.atomic_add(ccnt, Fk[p, q], 1)


@wp.kernel(enable_backward=False)
def k_cfill(slv: wp.array(dtype=wp.int32), rank: wp.array(dtype=wp.int32), Fk: wp.array2d(dtype=wp.int32), Fw: wp.array2d(dtype=wp.float64),
            Fn: wp.array(dtype=wp.int32), cptr: wp.array(dtype=int), ccur: wp.array(dtype=wp.int32), crow: wp.array(dtype=int),
            cval: wp.array(dtype=wp.float64)):
    p = wp.tid()
    if slv[p] == 0:
        return
    r = rank[p]-1
    for q in range(Fn[p]):
        k = Fk[p, q]
        pos = wp.atomic_add(ccur, k, 1)
        crow[cptr[k]+pos] = r
        cval[cptr[k]+pos] = Fw[p, q]


@wp.kernel(enable_backward=False)
def k_sdiag(sid: wp.array(dtype=int), nsa: wp.array(dtype=int), m: wp.array(dtype=wp.float64), S: wp.array2d(dtype=wp.float64)):
    r = wp.tid()
    if r < nsa[0]:
        S[r, r] = S[r, r]+wp.float64(1.0)/m[sid[r]]


# ---- the column estimate's emission, bounds-checked (k1_column_warp.k_emit otherwise)
@wp.func
def emit2(keys: wp.array(dtype=wp.int32), idx: wp.array(dtype=wp.int32), cv: wp.array(dtype=wp.vec3d), p: int, s: int, n: int, k: int, c: wp.vec3d,
          C: int, over: wp.array(dtype=wp.int32)):
    if p < C:
        keys[p] = s*n+k
        idx[p] = p
        cv[p] = c
    else:
        over[2] = 1
    return p+1


@wp.kernel(enable_backward=False)
def k_emit2(wi: wp.array(dtype=wp.int32), wj: wp.array(dtype=wp.int32), wa: wp.array(dtype=wp.float64), wn: wp.array(dtype=wp.vec3d),
            wt: wp.array(dtype=wp.float64), wtau: wp.array(dtype=wp.vec3d), offW: wp.array(dtype=wp.int32), lstW: wp.array(dtype=wp.int32),
            offG: wp.array(dtype=wp.int32), gO: wp.array(dtype=wp.int32), gR: wp.array(dtype=wp.vec3d),
            offA: wp.array(dtype=wp.int32), aO: wp.array(dtype=wp.int32), aW: wp.array(dtype=wp.float64), used: wp.array(dtype=wp.int32),
            a_: wp.array(dtype=wp.vec3d), Mc: wp.array(dtype=wp.mat33d), capc: wp.array(dtype=wp.int32), start: wp.array(dtype=wp.int32), n: int, C: int,
            keys: wp.array(dtype=wp.int32), idx: wp.array(dtype=wp.int32), cv: wp.array(dtype=wp.vec3d), over: wp.array(dtype=wp.int32)):
    s = wp.tid()
    one = wp.float64(1.0)
    half = wp.float64(0.5)
    p = start[s]
    for e in range(offW[s], offW[s+1]):
        w = lstW[e]
        i = wi[w]
        j = wj[w]
        sg = one
        if s != i:
            sg = -one
        A = wa[w]
        n_ = wn[w]
        p = emit2(keys, idx, cv, p, s, n, i, (sg*A*(one-wt[w]))*n_, C, over)
        p = emit2(keys, idx, cv, p, s, n, j, (sg*A*wt[w])*n_, C, over)
        base = sg*A*half
        tau = wtau[w]
        for side in range(2):
            c = i
            if side == 1:
                c = j
            if used[c] == 0:
                for f in range(offG[c], offG[c+1]):
                    co = base*wp.dot(gR[f], tau)
                    p = emit2(keys, idx, cv, p, s, n, gO[f], co*n_, C, over)
                    p = emit2(keys, idx, cv, p, s, n, c, -co*n_, C, over)
            else:
                for b in range(offA[c], offA[c+1]):
                    o = aO[b]
                    ww = aW[b]
                    for f in range(offG[o], offG[o+1]):
                        co = base*ww*wp.dot(gR[f], tau)
                        p = emit2(keys, idx, cv, p, s, n, gO[f], co*n_, C, over)
                        p = emit2(keys, idx, cv, p, s, n, o, -co*n_, C, over)
    p = emit2(keys, idx, cv, p, s, n, s, a_[s], C, over)
    if capc[s] != 0:
        M = wp.transpose(Mc[s])
        if used[s] == 0:
            for f in range(offG[s], offG[s+1]):
                p = emit2(keys, idx, cv, p, s, n, gO[f], M@gR[f], C, over)
                p = emit2(keys, idx, cv, p, s, n, s, -(M@gR[f]), C, over)
        else:
            for b in range(offA[s], offA[s+1]):
                o = aO[b]
                ww = aW[b]
                for f in range(offG[o], offG[o+1]):
                    p = emit2(keys, idx, cv, p, s, n, gO[f], ww*(M@gR[f]), C, over)
                    p = emit2(keys, idx, cv, p, s, n, o, -ww*(M@gR[f]), C, over)


@wp.kernel(enable_backward=False)
def k_sent(keys: wp.array(dtype=wp.int32), idx: wp.array(dtype=wp.int32), totd: wp.array(dtype=wp.int32), sent: int):
    q = wp.tid()
    if q >= totd[0]:
        keys[q] = sent
        idx[q] = 0


@wp.kernel(enable_backward=False)
def k_runs2(keys: wp.array(dtype=wp.int32), idx: wp.array(dtype=wp.int32), cv: wp.array(dtype=wp.vec3d), C: int, n: int, sent: int,
            Vi: wp.array(dtype=wp.float64), l2m: wp.float64, K: wp.array(dtype=wp.float64)):
    q = wp.tid()
    key = keys[q]
    if key == sent:
        return
    if q > 0:
        if keys[q-1] == key:
            return
    c = wp.vec3d()
    r = q
    while r < C:
        if keys[r] != key:
            break
        c = c+cv[idx[r]]
        r += 1
    s = key/n
    k = key-s*n
    vi = Vi[s]
    cc = c*vi
    wp.atomic_add(K, k, l2m/vi*wp.dot(cc, cc))


class FollowFast:
    def __init__(self, n, dev='cuda', nscap=2048, cap1=64, capF=256):
        self.n, self.dev, self.nscap, self.cap1, self.capF = n, dev, nscap, cap1, capF
        Z = lambda *s_, dt=i32: torch.zeros(*s_, dtype=dt, device=dev)
        W_ = lambda t_, d_: wp.from_torch(t_, dtype=d_)
        self.t = dict(T1k=Z(n, cap1), T1w=Z(n, cap1, dt=f64), T1n=Z(n), Fk=Z(n, capF), Fw=Z(n, capF, dt=f64), Fn=Z(n), over=Z(4), ok=Z(n),
                      ccnt=Z(n), ccur=Z(n), cptr=Z(n+1), ptr=Z(n+1), sid=Z(n), col=Z(n*capF), val=Z(n*capF, dt=f64), crow=Z(n*capF), cval=Z(n*capF, dt=f64),
                      Dinv=Z(n, dt=f64), one=Z(n, dt=f64), nsa=Z(1), Si=Z(nscap, nscap, dt=f64), K=Z(n, dt=f64),
                      v=Z(n, 3, dt=f64), r=Z(n, 3, dt=f64), y=Z(n, 3, dt=f64), x=Z(n, 3, dt=f64), q=Z(n, 3, dt=f64), tt=Z(nscap, 3, dt=f64), z=Z(nscap, 3, dt=f64))
        self.Mi = wp.zeros(n, dtype=wp.mat33d, device=dev)
        t = self.t
        self.w = {k_: W_(v_, wp.vec3d if v_.dim() == 2 and v_.shape[1] == 3 and v_.dtype == f64 and k_ in ('v', 'r', 'y', 'x', 'q', 'tt', 'z') else
                         (wp.float64 if v_.dtype == f64 else wp.int32)) for k_, v_ in t.items()}
        self.C = 0
        self.ns = 0
        self._graphs()

    def _graphs(self):
        n, cs, w = self.n, self.nscap, self.w
        I = lambda k_: wp.from_torch(self.t[k_], dtype=wp.int32)

        def runP():
            wp.launch(k_gm_P, cs, inputs=[w['v'], I('ptr'), I('col'), w['val'], I('sid'), I('nsa')])

        def runPt():
            wp.launch(k_fm_Pt, n, inputs=[w['r'], I('cptr'), I('crow'), w['cval'], I('sid'), w['one']], outputs=[w['q']])

        def pre():
            wp.launch(k_fm_Pt, n, inputs=[w['r'], I('cptr'), I('crow'), w['cval'], I('sid'), w['Dinv']], outputs=[w['y']])
            wp.launch(k_gm_E, cs, inputs=[w['y'], I('ptr'), I('col'), w['val'], I('nsa')], outputs=[w['tt']])

        def post():
            wp.launch(k_fm_Et, n, inputs=[w['y'], w['z'], I('cptr'), I('crow'), w['cval'], w['Dinv']], outputs=[w['x']])
        self.g = {}
        for name, fn in (('P', runP), ('Pt', runPt), ('pre', pre), ('post', post)):
            fn(); wp.capture_begin(); fn(); self.g[name] = wp.capture_end()

    def rebuild(self, W, x, mass, target=9600., wmax=4., ratio=1e-3):
        n, dev, t, w = self.n, self.dev, self.t, self.w
        tul, sp = W.tul, W.tsp
        m, ml = W._w[5], W._ml
        I32 = lambda a_: wp.from_torch(a_.to(i32) if a_.dtype != i32 else a_, dtype=wp.int32)
        F64 = lambda a_: wp.from_torch(a_, dtype=wp.float64)
        V3 = lambda a_: wp.from_torch(a_, dtype=wp.vec3d)
        # one wall list per cut
        wi, wj = tul['wi'][:m], tul['wj'][:m]; wa = tul['wa'][:m]
        keyW = torch.cat([wi, wj]).long()
        o = torch.argsort(keyW, stable=True); cellE = keyW[o]
        lstW = torch.cat([torch.arange(m, device=dev)]*2)[o]
        cnt = torch.zeros(n, dtype=torch.long, device=dev).scatter_add_(0, keyW, torch.ones_like(keyW))
        offW = torch.zeros(n+1, dtype=torch.long, device=dev); offW[1:] = torch.cumsum(cnt, 0)
        other = wi.long()[lstW]+wj.long()[lstW]-cellE
        areaE = wa[lstW]; nwE = areaE/(x[other]-x[cellE]).norm(dim=1)
        offWw, lstWw, othw, areaw, nww = I32(offW), I32(lstW), I32(other), F64(areaE.contiguous()), F64(nwE.contiguous())
        # the column estimate
        S0, O0, R0 = tul['S'][:2*ml].long(), tul['O'][:2*ml], tul['R'][:2*ml]
        oG = torch.argsort(S0, stable=True)
        cG = torch.zeros(n, dtype=torch.long, device=dev).scatter_add_(0, S0, torch.ones_like(S0))
        offG = torch.zeros(n+1, dtype=torch.long, device=dev); offG[1:] = torch.cumsum(cG, 0)
        k3 = W._k3
        As, Ao, Aw = tul['As'][:k3].long(), tul['Ao'][:k3], tul['Aw'][:k3]
        oA = torch.argsort(As, stable=True)
        cA = torch.zeros(n, dtype=torch.long, device=dev).scatter_add_(0, As, torch.ones_like(As))
        offA = torch.zeros(n+1, dtype=torch.long, device=dev); offA[1:] = torch.cumsum(cA, 0)
        capc = (tul['Mc'].abs().sum((1, 2)) > 0).to(i32)
        wiw, wjw, offGw, offAw, aOw, usedw, capw = I32(wi), I32(wj), I32(offG), I32(offA), I32(Ao[oA].contiguous()), I32(tul['used']), I32(capc)
        ccol = wp.zeros(n, dtype=wp.int32, device=dev)
        wp.launch(k_count, n, inputs=[wiw, wjw, offWw, lstWw, offGw, offAw, aOw, usedw, capw], outputs=[ccol], device=dev)
        cc_ = wp.to_torch(ccol).long(); cum = torch.cumsum(cc_, 0); start = (cum-cc_).to(i32); totd = cum[-1:].to(i32)
        if self.C == 0:                                                            # (the first cut: the count read once, the capacity set)
            self._alloc_col(int(totd[0]))
        C = self.C
        over = w['over']; t['over'].zero_()
        wp.launch(k_emit2, n, inputs=[wiw, wjw, F64(wa), V3(tul['wn'][:m]), F64(tul['wt'][:m]), V3(tul['wtau'][:m]), offWw, lstWw, offGw, I32(O0[oG].contiguous()),
                                      V3(R0[oG].contiguous()), offAw, aOw, F64(Aw[oA].contiguous()), usedw, V3(tul['a']), wp.from_torch(tul['Mc'], dtype=wp.mat33d), capw,
                                      I32(start), n, C], outputs=[self.keys, self.idx, self.cv, over], device=dev)
        wp.launch(k_sent, C, inputs=[self.keys, self.idx, I32(totd), SENT], device=dev)
        wp.utils.radix_sort_pairs(self.keys, self.idx, C)
        t['K'].zero_()
        wp.launch(k_runs2, C, inputs=[self.keys, self.idx, self.cv, C, n, SENT, F64(tul['Vi']), float(W.kap+4*W.mu/3)], outputs=[w['K']], device=dev)
        skk = sp['sk'][:W._m_s]
        Kt = t['K'].index_add(0, W.tlo, skk).index_add(0, W.thi, skk)
        om = torch.sqrt(Kt/W.V0)
        # level 1: the tiny cells
        if not hasattr(self, 'tiny'):
            self.tiny = (W.V0 < W._ul_agg*W.V0.median()).to(i32)
        xw = V3(x.contiguous()); Mi = self.Mi; okw = w['ok']
        wp.launch(k_fit, n, inputs=[xw, offWw, othw, nww, I32(self.tiny), float(ratio)], outputs=[Mi, okw], device=dev)
        wp.launch(k_rows_n, n, inputs=[xw, offWw, othw, areaw, nww, I32(self.tiny), Mi, okw, I32(self.tiny), self.cap1, float(wmax)],
                  outputs=[w['T1k'], w['T1w'], w['T1n'], over], device=dev)
        lev1 = ((t['T1n'] > 0) & (self.tiny > 0)).to(i32)
        # level 2: over the target, not slaved at level 1
        new2 = ((om > target) & (lev1 == 0)).to(i32)
        sl2 = (lev1 | new2).to(i32)
        wp.launch(k_fit, n, inputs=[xw, offWw, othw, nww, I32(sl2), float(ratio)], outputs=[Mi, okw], device=dev)
        wp.launch(k_rows_n, n, inputs=[xw, offWw, othw, areaw, nww, I32(sl2), Mi, okw, I32(new2), self.cap1, float(wmax)],
                  outputs=[w['Fk'], w['Fw'], w['Fn'], over], device=dev)
        s2 = ((t['Fn'] > 0) & (new2 > 0)).to(i32)
        self.last = dict(om=om, lev1=lev1, new2=new2, s2=s2)                         # (logging only: the column estimate, the levels' flags)
        wp.launch(k_compose2, n, inputs=[w['T1k'], w['T1w'], w['T1n'], I32(lev1), I32(s2), self.capF], outputs=[w['Fk'], w['Fw'], w['Fn'], over], device=dev)
        slv = (lev1 | s2).to(i32); self._slv = slv
        # pack
        rank = torch.cumsum(slv, 0).to(i32)
        fn = (t['Fn']*slv).long(); cf = torch.cumsum(fn, 0); rowoff = (cf-fn).to(i32)
        nsd, nnzd = rank[-1:].contiguous(), cf[-1:].to(i32)
        I = lambda k_: wp.from_torch(t[k_], dtype=wp.int32)
        wp.launch(k_pack_n, n, inputs=[I32(slv), I32(rank), I32(rowoff), w['Fk'], w['Fw'], w['Fn']], outputs=[I('sid'), I('ptr'), I('col'), w['val']], device=dev)
        wp.launch(k_tail, 1, inputs=[I('ptr'), I32(nsd), I32(nnzd)], outputs=[I('nsa')], device=dev)
        t['ccnt'].zero_()
        wp.launch(k_ccount, n, inputs=[I32(slv), w['Fk'], w['Fn']], outputs=[w['ccnt']], device=dev)
        t['cptr'][0] = 0; t['cptr'][1:] = torch.cumsum(t['ccnt'], 0)
        t['ccur'].zero_()
        wp.launch(k_cfill, n, inputs=[I32(slv), I32(rank), w['Fk'], w['Fw'], w['Fn'], I('cptr'), w['ccur']], outputs=[I('crow'), w['cval']], device=dev)
        master = slv == 0
        t['Dinv'].copy_(torch.where(master, 1/mass, torch.zeros_like(mass))); t['one'].copy_(master.to(f64))
        # the one host sync: counts, flags, the last factor's info
        info = getattr(self, 'info', None)
        st = torch.cat([nsd.long(), nnzd.long(), totd.long(), t['over'].long(), (info.reshape(1).long() if info is not None else torch.zeros(1, dtype=torch.long, device=dev))]).cpu().tolist()
        ns, nnz, tot = st[0], st[1], st[2]; ov = st[3:7]; self.ns, self.nnz = ns, nnz
        if st[7] != 0:
            raise RuntimeError('FollowFast: the last S was not positive definite (info %d)' % st[7])
        if ov[2]:                                                                  # (the column's capacity: grown, the cut done again)
            self._alloc_col(tot)
            return self.rebuild(W, x, mass, target, wmax, ratio)
        if ov[0] or ov[1] or ns > self.nscap:
            raise RuntimeError('FollowFast: a row over its slots (%s) or ns %d over %d' % (ov[:2], ns, self.nscap))
        # S and its factor
        if ns:
            S = t['Si'][:ns, :ns]; S.zero_()
            wp.launch(k_gm_Sasm, n, inputs=[I('cptr'), I('crow'), w['cval'], w['Dinv']], outputs=[w['Si']], device=dev)
            wp.launch(k_sdiag, ns, inputs=[I('sid'), I('nsa'), F64(mass)], outputs=[w['Si']], device=dev)
            self.L, self.info = torch.linalg.cholesky_ex(S.clone())
        return om

    def _alloc_col(self, tot):
        C = int(1.4*tot)+4096; dev = self.dev
        self.keys, self.idx = wp.zeros(2*C, dtype=wp.int32, device=dev), wp.zeros(2*C, dtype=wp.int32, device=dev)
        self.cv = wp.zeros(C, dtype=wp.vec3d, device=dev); self.C = C

    @property
    def master(self):
        return self._slv == 0

    @property
    def slave(self):
        return self._slv != 0

    def P(self, y):
        self.t['v'].copy_(y*self.master[:, None])
        if self.ns:
            wp.capture_launch(self.g['P'])
        return self.t['v'].clone()

    def Pt(self, r):
        self.t['r'].copy_(r); wp.capture_launch(self.g['Pt'])
        return self.t['q'].clone()

    def _half(self, r):
        self.t['r'].copy_(r); wp.capture_launch(self.g['pre'])
        return self.t['y'].clone(), self.t['tt'][:self.ns].clone()

    def _back(self, y, z):
        self.t['y'].copy_(y)
        if self.ns:
            self.t['z'][:self.ns] = z
        wp.capture_launch(self.g['post'])
        return self.t['x'].clone()

    def force_to_accel(self, r):
        """M_r^-1 P^T r (masters)"""
        y, tt = self._half(r)
        z = torch.cholesky_solve(tt, self.L) if self.ns else tt
        return self._back(y, z)

    def accel2(self, r1, r2):
        """M_r^-1 P^T r1 and M_r^-1 P^T r2, one triangular solve on six columns"""
        y1, t1 = self._half(r1); y2, t2 = self._half(r2)
        if self.ns:
            z = torch.cholesky_solve(torch.cat([t1, t2], 1), self.L); z1, z2 = z[:, :3], z[:, 3:]
        else:
            z1, z2 = t1, t2
        return self._back(y1, z1), self._back(y2, z2)
