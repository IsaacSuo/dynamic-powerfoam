"""MergedUL's cut assembly in Warp with one host read (11-06, stage 3 of the explicit step's restructuring; the same
outputs as MergedUL._assemble_torch, to rounding, in the same orders):
  walls       the cut's pairs (geo_pipe / GeometryWarp: key order i n + j, i < j); a wall when its area is positive
  springs     the old list (sorted by key) and the walls matched by binary search: a wall with no spring whose cells
              have both yielded is born; a spring stays unless both its cells have yielded and its wall is gone; the
              new list merged in key order without sorting (a stayer's place: its rank among the stayers + the born
              walls before its key; a born wall's place likewise); per spring the offset p = d - dl and k (both cells
              yielded and a wall: kshare 2 mu A / L, else kept)
  connected   a wall with a spring (born, or found in the old list); compacted in order: the walls' buffers, the fit's
              entries (walls, then the springs without a wall at their kept weight), each cell's M, a, M_w by atomics
  fit         M^-1 per cell (cofactor; pinv by the eigenvalues when singular), the rows R
  aggregation the connected walls with one tiny end, those with a tiny first end first (as before), weights V0 / sum
  one read    the counts (springs stayed and born, connected walls, ghosts, aggregation entries, valid walls), the
              capacity flags, the floored count
AsmWarp(W).run(W, x, r, G) writes W's buffers and attributes as _assemble_torch does; check(W, x, r, G) runs both on
the same input and prints the largest differences (the Warp result kept)."""
import torch
import warp as wp

f64, i32, i64 = torch.float64, torch.int32, torch.int64


@wp.func
def lbound(a: wp.array(dtype=wp.int64), n: int, key: wp.int64):
    lo = int(0)
    hi = int(n)
    while lo < hi:
        mid = (lo+hi)//2
        if a[mid] < key:
            lo = mid+1
        else:
            hi = mid
    return lo


@wp.kernel(enable_backward=False)
def k_wall_flags(pk: wp.array(dtype=wp.int64), A: wp.array(dtype=wp.float64), pi: wp.array(dtype=wp.int64), pj: wp.array(dtype=wp.int64),
                 yld: wp.array(dtype=wp.int32), tkey: wp.array(dtype=wp.int64), nold: int,
                 valid: wp.array(dtype=wp.int32), fold: wp.array(dtype=wp.int32), born: wp.array(dtype=wp.int32)):
    q = wp.tid()
    v = int(0)
    f = int(-1)
    b_ = int(0)
    if A[q] > wp.float64(0.0):
        v = 1
        b = lbound(tkey, nold, pk[q])
        if b < nold:
            if tkey[b] == pk[q]:
                f = b
        if f < 0 and yld[int(pi[q])] != 0 and yld[int(pj[q])] != 0:
            b_ = 1
    valid[q] = v
    fold[q] = f
    born[q] = b_


@wp.kernel(enable_backward=False)
def k_spring_flags(tkey: wp.array(dtype=wp.int64), tlo: wp.array(dtype=wp.int64), thi: wp.array(dtype=wp.int64), yld: wp.array(dtype=wp.int32),
                   pk: wp.array(dtype=wp.int64), valid: wp.array(dtype=wp.int32), npr: int,
                   stay: wp.array(dtype=wp.int32), swall: wp.array(dtype=wp.int32)):
    s = wp.tid()
    b = lbound(pk, npr, tkey[s])
    q = int(-1)
    if b < npr:
        if pk[b] == tkey[s] and valid[b] == 1:
            q = b
    swall[s] = q
    st = int(1)
    if yld[int(tlo[s])] != 0 and yld[int(thi[s])] != 0 and q < 0:
        st = 0
    stay[s] = st


@wp.kernel(enable_backward=False)
def k_merge_old(tkey: wp.array(dtype=wp.int64), tlo: wp.array(dtype=wp.int64), thi: wp.array(dtype=wp.int64), tdl: wp.array(dtype=wp.vec3d),
                tk0: wp.array(dtype=wp.float64), stay: wp.array(dtype=wp.int32), sinc: wp.array(dtype=wp.int32), swall: wp.array(dtype=wp.int32),
                pk: wp.array(dtype=wp.int64), npr: int, binc: wp.array(dtype=wp.int32), cap: int,
                nlo: wp.array(dtype=wp.int64), nhi: wp.array(dtype=wp.int64), nkey: wp.array(dtype=wp.int64), ndl: wp.array(dtype=wp.vec3d),
                nk0: wp.array(dtype=wp.float64), nwall: wp.array(dtype=wp.int32), over: wp.array(dtype=wp.int32)):
    s = wp.tid()
    if stay[s] == 0:
        return
    b = lbound(pk, npr, tkey[s])
    nb = int(0)
    if b > 0:
        nb = binc[b-1]
    pos = sinc[s]-1+nb
    if pos >= cap:
        wp.atomic_max(over, 0, 1)
        return
    nlo[pos] = tlo[s]
    nhi[pos] = thi[s]
    nkey[pos] = tkey[s]
    ndl[pos] = tdl[s]
    nk0[pos] = tk0[s]
    nwall[pos] = swall[s]


@wp.kernel(enable_backward=False)
def k_merge_born(pk: wp.array(dtype=wp.int64), pi: wp.array(dtype=wp.int64), pj: wp.array(dtype=wp.int64), born: wp.array(dtype=wp.int32),
                 binc: wp.array(dtype=wp.int32), tkey: wp.array(dtype=wp.int64), nold: int, sinc: wp.array(dtype=wp.int32), cap: int,
                 nlo: wp.array(dtype=wp.int64), nhi: wp.array(dtype=wp.int64), nkey: wp.array(dtype=wp.int64), ndl: wp.array(dtype=wp.vec3d),
                 nk0: wp.array(dtype=wp.float64), nwall: wp.array(dtype=wp.int32), over: wp.array(dtype=wp.int32)):
    q = wp.tid()
    if born[q] == 0:
        return
    b = lbound(tkey, nold, pk[q])
    ns = int(0)
    if b > 0:
        ns = sinc[b-1]
    pos = binc[q]-1+ns
    if pos >= cap:
        wp.atomic_max(over, 0, 1)
        return
    nlo[pos] = pi[q]
    nhi[pos] = pj[q]
    nkey[pos] = pk[q]
    ndl[pos] = wp.vec3d()
    nk0[pos] = wp.float64(0.0)
    nwall[pos] = q


@wp.kernel(enable_backward=False)
def k_springs(x: wp.array(dtype=wp.vec3d), yld: wp.array(dtype=wp.int32), A: wp.array(dtype=wp.float64), cnt: wp.array(dtype=wp.int32), ks2mu: wp.float64,
              nlo: wp.array(dtype=wp.int64), nhi: wp.array(dtype=wp.int64), ndl: wp.array(dtype=wp.vec3d), nk0: wp.array(dtype=wp.float64), nwall: wp.array(dtype=wp.int32),
              nP: wp.array(dtype=wp.vec3d), nk: wp.array(dtype=wp.float64), gh: wp.array(dtype=wp.int32),
              si: wp.array(dtype=wp.int32), sj: wp.array(dtype=wp.int32), sP: wp.array(dtype=wp.vec3d), sdn: wp.array(dtype=wp.vec3d), sk: wp.array(dtype=wp.float64),
              sb: wp.array(dtype=wp.float64)):
    """over the capacity: the live springs (cnt 0 + cnt 1), zeros after"""
    t = wp.tid()
    sb[t] = wp.float64(1.0)
    if t >= cnt[0]+cnt[1]:
        si[t] = 0
        sj[t] = 0
        sP[t] = wp.vec3d()
        sdn[t] = wp.vec3d()
        sk[t] = wp.float64(0.0)
        gh[t] = 0
        return
    lo = int(nlo[t])
    hi = int(nhi[t])
    ds = x[hi]-x[lo]
    dl = ndl[t]
    p = ds-dl
    k = nk0[t]
    q = nwall[t]
    if q >= 0 and yld[lo] != 0 and yld[hi] != 0:
        k = ks2mu*A[q]/wp.length(ds)
    nP[t] = p
    nk[t] = k
    si[t] = lo
    sj[t] = hi
    sP[t] = p
    sdn[t] = dl
    sk[t] = k
    g = int(0)
    if q < 0:
        g = 1
    gh[t] = g


@wp.kernel(enable_backward=False)
def k_walls(x: wp.array(dtype=wp.vec3d), pi: wp.array(dtype=wp.int64), pj: wp.array(dtype=wp.int64), A: wp.array(dtype=wp.float64), C: wp.array(dtype=wp.vec3d),
            valid: wp.array(dtype=wp.int32), fold: wp.array(dtype=wp.int32), born: wp.array(dtype=wp.int32), cinc: wp.array(dtype=wp.int32),
            tiny: wp.array(dtype=wp.int32), kap24: wp.float64, Kw: int,
            wi: wp.array(dtype=wp.int32), wj: wp.array(dtype=wp.int32), wa: wp.array(dtype=wp.float64), wn: wp.array(dtype=wp.vec3d), wt: wp.array(dtype=wp.float64),
            wtau: wp.array(dtype=wp.vec3d), wv: wp.array(dtype=wp.float64), Lwb: wp.array(dtype=wp.float64),
            fS: wp.array(dtype=wp.int32), fO: wp.array(dtype=wp.int32), fd: wp.array(dtype=wp.vec3d), fw: wp.array(dtype=wp.float64),
            M: wp.array(dtype=wp.mat33d), acc: wp.array(dtype=wp.vec3d), Mw: wp.array(dtype=wp.mat33d), openc: wp.array(dtype=wp.int32),
            ag1: wp.array(dtype=wp.int32), ag2: wp.array(dtype=wp.int32), over: wp.array(dtype=wp.int32)):
    q = wp.tid()
    ag1[q] = 0
    ag2[q] = 0
    if valid[q] == 0:
        return
    i = int(pi[q])
    j = int(pj[q])
    if fold[q] < 0 and born[q] == 0:                                             # (a wall without a spring: open to both cells)
        openc[i] = 1
        openc[j] = 1
        return
    e = cinc[q]-1
    if e >= Kw:
        wp.atomic_max(over, 1, 1)
        return
    d = x[j]-x[i]
    L = wp.length(d)
    nn = d/L
    a = A[q]
    c = C[q]
    tw = wp.dot(c-x[i], nn)/L
    wi[e] = i
    wj[e] = j
    wa[e] = a
    wn[e] = nn
    wt[e] = tw
    wtau[e] = c-(x[i]+tw*d)
    wv[e] = kap24*a*L
    Lwb[e] = L
    fS[e] = i
    fO[e] = j
    fd[e] = d
    w = a/L
    fw[e] = w
    q3 = w*wp.outer(d, d)
    wp.atomic_add(M, i, q3)
    wp.atomic_add(M, j, q3)
    wp.atomic_add(acc, i, -a*nn)
    wp.atomic_add(acc, j, a*nn)
    wp.atomic_add(Mw, i, a*wp.outer(c-x[i], nn))
    wp.atomic_add(Mw, j, -a*wp.outer(c-x[j], nn))
    if tiny[i] != 0 and tiny[j] == 0:
        ag1[q] = 1
    if tiny[j] != 0 and tiny[i] == 0:
        ag2[q] = 1


@wp.kernel(enable_backward=False)
def k_ghosts(x: wp.array(dtype=wp.vec3d), cnt: wp.array(dtype=wp.int32), gh: wp.array(dtype=wp.int32), ginc: wp.array(dtype=wp.int32),
             nlo: wp.array(dtype=wp.int64), nhi: wp.array(dtype=wp.int64), nk: wp.array(dtype=wp.float64), inv_ks2mu: wp.float64, Kw: int,
             fS: wp.array(dtype=wp.int32), fO: wp.array(dtype=wp.int32), fd: wp.array(dtype=wp.vec3d), fw: wp.array(dtype=wp.float64),
             M: wp.array(dtype=wp.mat33d), over: wp.array(dtype=wp.int32)):
    """the springs without a wall, after the connected walls (cnt 2): the fit at their kept weight"""
    t = wp.tid()
    if gh[t] == 0:
        return
    e = cnt[2]+ginc[t]-1
    if e >= Kw:
        wp.atomic_max(over, 1, 1)
        return
    lo = int(nlo[t])
    hi = int(nhi[t])
    d = x[hi]-x[lo]
    w = nk[t]*inv_ks2mu
    fS[e] = lo
    fO[e] = hi
    fd[e] = d
    fw[e] = w
    q3 = w*wp.outer(d, d)
    wp.atomic_add(M, lo, q3)
    wp.atomic_add(M, hi, q3)


@wp.kernel(enable_backward=False)
def k_fit_inv(M: wp.array(dtype=wp.mat33d), Mi: wp.array(dtype=wp.mat33d)):
    """M^-1 by the cofactors (as before); singular (|det| <= 1e-10 |M|_F^3): the pseudo-inverse by the eigenvalues (M is
    symmetric; torch's pinv drops the singular values at or below 3 eps of the largest)"""
    k = wp.tid()
    m = M[k]
    a0 = wp.vec3d(m[0, 0], m[1, 0], m[2, 0])
    a1 = wp.vec3d(m[0, 1], m[1, 1], m[2, 1])
    a2 = wp.vec3d(m[0, 2], m[1, 2], m[2, 2])
    c0 = wp.cross(a1, a2)
    c1 = wp.cross(a2, a0)
    c2 = wp.cross(a0, a1)
    det = wp.dot(a0, c0)
    f2 = wp.float64(0.0)
    for r_ in range(3):
        for c_ in range(3):
            f2 = f2+m[r_, c_]*m[r_, c_]
    if wp.abs(det) <= wp.float64(1e-10)*f2*wp.sqrt(f2):
        Q, ev = wp.eig3(m)
        emax = wp.max(wp.abs(ev[0]), wp.max(wp.abs(ev[1]), wp.abs(ev[2])))
        tol = wp.float64(6.661338147750939e-16)*emax
        P = wp.mat33d()
        for e in range(3):
            if wp.abs(ev[e]) > tol:
                qv = wp.vec3d(Q[0, e], Q[1, e], Q[2, e])
                P = P+wp.outer(qv, qv)/ev[e]
        Mi[k] = P
    else:
        Mi[k] = wp.mat33d(c0[0], c0[1], c0[2], c1[0], c1[1], c1[2], c2[0], c2[1], c2[2])/det


@wp.kernel(enable_backward=False)
def k_fit_rows(cnt: wp.array(dtype=wp.int32), fS: wp.array(dtype=wp.int32), fO: wp.array(dtype=wp.int32), fd: wp.array(dtype=wp.vec3d), fw: wp.array(dtype=wp.float64),
               Mi: wp.array(dtype=wp.mat33d), S: wp.array(dtype=wp.int32), O: wp.array(dtype=wp.int32), R: wp.array(dtype=wp.vec3d)):
    """entry e < ml (= cnt 2 + cnt 3): its i side at e, its j side at ml + e"""
    e = wp.tid()
    ml = cnt[2]+cnt[3]
    if e >= ml:
        return
    li = fS[e]
    lj = fO[e]
    vd = fw[e]*fd[e]
    S[e] = li
    O[e] = lj
    R[e] = vd*Mi[li]
    S[ml+e] = lj
    O[ml+e] = li
    R[ml+e] = -vd*Mi[lj]


@wp.kernel(enable_backward=False)
def k_agg(pi: wp.array(dtype=wp.int64), pj: wp.array(dtype=wp.int64), ag1: wp.array(dtype=wp.int32), ag2: wp.array(dtype=wp.int32),
          a1inc: wp.array(dtype=wp.int32), a2inc: wp.array(dtype=wp.int32), cnt: wp.array(dtype=wp.int32), V0: wp.array(dtype=wp.float64), Kw: int,
          As: wp.array(dtype=wp.int32), Ao: wp.array(dtype=wp.int32), va: wp.array(dtype=wp.float64), den: wp.array(dtype=wp.float64), over: wp.array(dtype=wp.int32)):
    """the connected walls with one tiny end: those with a tiny first end first (cnt 4 of them), then the others"""
    q = wp.tid()
    e = int(-1)
    s = int(0)
    o = int(0)
    if ag1[q] != 0:
        e = a1inc[q]-1
        s = int(pi[q])
        o = int(pj[q])
    elif ag2[q] != 0:
        e = cnt[4]+a2inc[q]-1
        s = int(pj[q])
        o = int(pi[q])
    if e < 0:
        return
    if e >= Kw:
        wp.atomic_max(over, 2, 1)
        return
    As[e] = s
    Ao[e] = o
    va[e] = V0[o]
    wp.atomic_add(den, s, V0[o])


@wp.kernel(enable_backward=False)
def k_agg_w(cnt: wp.array(dtype=wp.int32), As: wp.array(dtype=wp.int32), va: wp.array(dtype=wp.float64), den: wp.array(dtype=wp.float64), Aw: wp.array(dtype=wp.float64)):
    e = wp.tid()
    if e < cnt[4]+cnt[5]:
        Aw[e] = va[e]/den[As[e]]


@wp.kernel(enable_backward=False)
def k_cells(V: wp.array(dtype=wp.float64), pA: wp.array(dtype=wp.float64), openc: wp.array(dtype=wp.int32), acc: wp.array(dtype=wp.vec3d), Mw: wp.array(dtype=wp.mat33d),
            a: wp.array(dtype=wp.vec3d), Mc: wp.array(dtype=wp.mat33d)):
    k = wp.tid()
    if pA[k] > wp.float64(0.0) or openc[k] != 0:
        a[k] = acc[k]
        Mc[k] = V[k]*wp.identity(n=3, dtype=wp.float64)-Mw[k]
    else:
        a[k] = wp.vec3d()
        Mc[k] = wp.mat33d()


class AsmWarp:
    def __init__(self, W):
        n, dev, K, Kw = W.n, W.dev, W.sp['K'], W.ul['Kw']
        self.n, self.dev, self.K, self.Kw = n, dev, K, Kw
        Z = lambda *s_, dt=f64: torch.zeros(*s_, dtype=dt, device=dev)
        self.sets = [dict(lo=Z(K, dt=i64), hi=Z(K, dt=i64), key=Z(K, dt=i64), dl=Z(K, 3), k0=Z(K), P=Z(K, 3), k=Z(K), wall=Z(K, dt=i32)) for _ in range(2)]
        self.cur = 0
        self.gh, self.ginc = Z(K, dt=i32), Z(K, dt=i32)
        self.fS, self.fO, self.fd, self.fw = Z(Kw, dt=i32), Z(Kw, dt=i32), Z(Kw, 3), Z(Kw)
        self.Lw = Z(Kw); self.va = Z(Kw)
        self.M, self.Mi, self.Mw = Z(n, 3, 3), Z(n, 3, 3), Z(n, 3, 3)
        self.acc, self.openc, self.den = Z(n, 3), Z(n, dt=i32), Z(n)
        self.cnt = Z(8, dt=i32); self.over = Z(4, dt=i32)
        self.tiny = (W.V0 < W._ul_agg*W.V0.median()).to(i32)

    def run(self, W, x, r, G):
        n, dev, K, Kw = self.n, self.dev, self.K, self.Kw
        t, s, yld = W.tul, W.tsp, W._yld
        Wa = lambda t_, d_: wp.from_torch(t_, dtype=d_)
        F_, I_, L_, V3, M3 = (lambda t_: Wa(t_, wp.float64)), (lambda t_: Wa(t_, wp.int32)), (lambda t_: Wa(t_, wp.int64)), (lambda t_: Wa(t_, wp.vec3d)), (lambda t_: Wa(t_, wp.mat33d))
        x = x.contiguous(); xw = V3(x)
        pi, pj = G.i.contiguous(), G.j.contiguous(); npr = len(pi)
        pk = (pi*n+pj).contiguous()
        A = G.wall_A.detach().contiguous(); C = G.wall_c.detach().contiguous()
        yl = yld.to(i32)
        nold = len(W.tkey)
        tkey, tlo, thi = W.tkey.contiguous(), W.tlo.contiguous(), W.thi.contiguous()
        tdl, tk0 = W.tdl.contiguous(), W.tk0.contiguous()
        Z = lambda m_, dt=i32: torch.zeros(max(m_, 1), dtype=dt, device=dev)
        valid, fold, born = Z(npr), Z(npr), Z(npr)
        self.cnt.zero_(); self.over.zero_()
        if npr:
            wp.launch(k_wall_flags, npr, inputs=[L_(pk), F_(A), L_(pi), L_(pj), I_(yl), L_(tkey), nold], outputs=[I_(valid), I_(fold), I_(born)], device=dev)
        stay, swall = Z(nold), Z(nold)
        if nold:
            wp.launch(k_spring_flags, nold, inputs=[L_(tkey), L_(tlo), L_(thi), I_(yl), L_(pk), I_(valid), npr], outputs=[I_(stay), I_(swall)], device=dev)
        sinc = torch.cumsum(stay, 0, dtype=i32); binc = torch.cumsum(born, 0, dtype=i32)
        self.cnt[0] = sinc[-1]; self.cnt[1] = binc[-1]
        N = self.sets[1-self.cur]
        if nold:
            wp.launch(k_merge_old, nold, inputs=[L_(tkey), L_(tlo), L_(thi), V3(tdl), F_(tk0), I_(stay), I_(sinc), I_(swall), L_(pk), npr, I_(binc), K],
                      outputs=[L_(N['lo']), L_(N['hi']), L_(N['key']), V3(N['dl']), F_(N['k0']), I_(N['wall']), I_(self.over)], device=dev)
        if npr:
            wp.launch(k_merge_born, npr, inputs=[L_(pk), L_(pi), L_(pj), I_(born), I_(binc), L_(tkey), nold, I_(sinc), K],
                      outputs=[L_(N['lo']), L_(N['hi']), L_(N['key']), V3(N['dl']), F_(N['k0']), I_(N['wall']), I_(self.over)], device=dev)
        ks2mu = W.kshare*2*W.mu
        wp.launch(k_springs, K, inputs=[xw, I_(yl), F_(A), I_(self.cnt), float(ks2mu), L_(N['lo']), L_(N['hi']), V3(N['dl']), F_(N['k0']), I_(N['wall'])],
                  outputs=[V3(N['P']), F_(N['k']), I_(self.gh), I_(s['si']), I_(s['sj']), V3(s['sP']), V3(s['sdn']), F_(s['sk']), F_(s['sb'])], device=dev)
        from free_surface.merged_ul import _mk
        _mk(W)
        # the connected walls
        conn = ((valid != 0) & ((fold >= 0) | (born != 0))).to(i32)
        cinc = torch.cumsum(conn, 0, dtype=i32); self.cnt[2] = cinc[-1]
        for k2 in ('wi', 'wj', 'wa', 'wn', 'wt', 'wtau', 'wv'):
            t[k2].zero_()
        self.M.zero_(); self.acc.zero_(); self.Mw.zero_(); self.openc.zero_()
        ag1, ag2 = Z(npr), Z(npr)
        if npr:
            wp.launch(k_walls, npr, inputs=[xw, L_(pi), L_(pj), F_(A), V3(C), I_(valid), I_(fold), I_(born), I_(cinc), I_(self.tiny), float(W.kap/24), Kw],
                      outputs=[I_(t['wi']), I_(t['wj']), F_(t['wa']), V3(t['wn']), F_(t['wt']), V3(t['wtau']), F_(t['wv']), F_(self.Lw),
                               I_(self.fS), I_(self.fO), V3(self.fd), F_(self.fw), M3(self.M), V3(self.acc), M3(self.Mw), I_(self.openc), I_(ag1), I_(ag2), I_(self.over)], device=dev)
        _mk(W)
        # the fit: the springs without a wall after the walls
        torch.cumsum(self.gh, 0, dtype=i32, out=self.ginc); self.cnt[3] = self.ginc[-1]
        wp.launch(k_ghosts, K, inputs=[xw, I_(self.cnt), I_(self.gh), I_(self.ginc), L_(N['lo']), L_(N['hi']), F_(N['k']), float(1/ks2mu), Kw],
                  outputs=[I_(self.fS), I_(self.fO), V3(self.fd), F_(self.fw), M3(self.M), I_(self.over)], device=dev)
        wp.launch(k_fit_inv, n, inputs=[M3(self.M)], outputs=[M3(self.Mi)], device=dev)
        t['S'].zero_(); t['O'].zero_(); t['R'].zero_()
        wp.launch(k_fit_rows, Kw, inputs=[I_(self.cnt), I_(self.fS), I_(self.fO), V3(self.fd), F_(self.fw), M3(self.Mi)], outputs=[I_(t['S']), I_(t['O']), V3(t['R'])], device=dev)
        _mk(W)
        # the aggregation
        a1inc, a2inc = torch.cumsum(ag1, 0, dtype=i32), torch.cumsum(ag2, 0, dtype=i32)
        self.cnt[4] = a1inc[-1]; self.cnt[5] = a2inc[-1]
        t['As'].zero_(); t['Ao'].zero_(); t['Aw'].zero_(); self.den.zero_()
        if npr:
            wp.launch(k_agg, npr, inputs=[L_(pi), L_(pj), I_(ag1), I_(ag2), I_(a1inc), I_(a2inc), I_(self.cnt), F_(W.V0), Kw],
                      outputs=[I_(t['As']), I_(t['Ao']), F_(self.va), F_(self.den), I_(self.over)], device=dev)
        wp.launch(k_agg_w, Kw, inputs=[I_(self.cnt), I_(t['As']), F_(self.va), F_(self.den)], outputs=[F_(t['Aw'])], device=dev)
        t['used'].copy_(((self.tiny != 0) & (self.den > 0)).to(i32))
        _mk(W)
        # the cells
        V = G.V.detach()
        wp.launch(k_cells, n, inputs=[F_(V.contiguous()), F_(G.patch_A.detach().contiguous()), I_(self.openc), V3(self.acc), M3(self.Mw)], outputs=[V3(t['a']), M3(t['Mc'])], device=dev)
        t['Vi'].copy_(1/V.clamp(min=1e-12*V.median()))
        t['xn'].copy_(x)
        if W.ball is not None:
            W.tbR.copy_(torch.where(G.patch_A.detach() > 0, r, -torch.ones_like(r)))
        self.cnt[6] = valid.sum()
        # the one read
        c = torch.cat([self.cnt, self.over]).tolist()
        n_stay, n_born, m, n_gh, n1, n2, n_valid = c[:7]; ov = c[8:12]
        m_s = n_stay+n_born; ml = m+n_gh; k3 = n1+n2
        if ov[0] or m_s > K:
            raise RuntimeError('update_ul: %d springs over the capacity %d' % (m_s, K))
        if ov[1] or m > Kw or ml > Kw:
            raise RuntimeError('update_ul: %d walls or %d fit entries over the capacity %d' % (m, ml, Kw))
        if ov[2] or k3 > Kw:
            raise RuntimeError('update_ul: %d aggregation entries over the capacity %d' % (k3, Kw))
        self.cur = 1-self.cur
        W.tlo, W.thi, W.tkey = N['lo'][:m_s], N['hi'][:m_s], N['key'][:m_s]
        W.tP, W.tdl, W.tk0 = N['P'][:m_s], N['dl'][:m_s].clone(), N['k'][:m_s].clone()
        W._m_s, W._ml, W._k3, W._V = m_s, ml, k3, V
        st = W.ul_stats
        st['closed'], st['born'], st['walls'], st['open_walls'], st['rows'], st['ghost'] = nold-n_stay, n_born, m, n_valid-m, m_s, n_gh
        W._w = (t['wi'][:m].long(), t['wj'][:m].long(), self.Lw[:m].clone(), pk, None, m)
        _mk(W)

    def check(self, W, x, r, G):
        """both on the same input: the torch assembly's outputs kept, the spring state put back, the Warp one run, the
        largest differences printed (the Warp result stays)"""
        W.asm_check_left = getattr(W, 'asm_check_left', 1)-1
        if W.asm_check_left <= 0:
            W.asm = 'warp'
        keep = {k_: getattr(W, k_, None) for k_ in ('tlo', 'thi', 'tkey', 'tdl', 'tk0', 'tP')}
        W._assemble_torch(x, r, G)
        names_t = ('S', 'O', 'R', 'As', 'Ao', 'Aw', 'used', 'wi', 'wj', 'wa', 'wn', 'wt', 'wtau', 'wv', 'a', 'Mc', 'Vi', 'xn')
        names_s = ('si', 'sj', 'sP', 'sdn', 'sk', 'sb')
        ref = dict(t={k_: W.tul[k_].clone() for k_ in names_t}, s={k_: W.tsp[k_].clone() for k_ in names_s},
                   a={k_: getattr(W, k_).clone() for k_ in ('tlo', 'thi', 'tkey', 'tdl', 'tk0', 'tP')},
                   h=(W._m_s, W._ml, W._k3, W._w[5], dict(W.ul_stats)), w=(W._w[0].clone(), W._w[1].clone(), W._w[2].clone()))
        for k_, v_ in keep.items():
            setattr(W, k_, v_)
        self.run(W, x, r, G)
        out = []
        rel = lambda a_, b_: float((a_.double()-b_.double()).abs().max()/b_.double().abs().max().clamp(min=1e-300)) if a_.numel() else 0.
        for k_ in names_t:
            out.append('%s %.1e' % (k_, rel(W.tul[k_], ref['t'][k_])))
        for k_ in names_s:
            out.append('%s %.1e' % (k_, rel(W.tsp[k_], ref['s'][k_])))
        for k_ in ('tlo', 'thi', 'tkey', 'tdl', 'tk0', 'tP'):
            a_, b_ = getattr(W, k_), ref['a'][k_]
            out.append('%s %s' % (k_, ('%.1e' % rel(a_, b_)) if a_.shape == b_.shape else 'shape %s / %s' % (tuple(a_.shape), tuple(b_.shape))))
        h_ = (W._m_s, W._ml, W._k3, W._w[5])
        st_ = {k_: W.ul_stats[k_] for k_ in ('closed', 'born', 'walls', 'open_walls', 'rows', 'ghost')}
        print('  ASMCHECK counts %s (torch %s); stats %s; %s' % (h_, ref['h'][:4], 'same' if all(ref['h'][4][k_] == v_ for k_, v_ in st_.items()) else 'DIFFER %s / %s' % (
            st_, {k_: ref['h'][4][k_] for k_ in st_}), ', '.join(out)), flush=True)
