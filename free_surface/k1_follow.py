"""The follow map for k1's stiff and tiny particles (11-06).

Each slaved particle s moves with an anchor's affine motion, u_s = u_k + G_k (x_s - x_k) (k1_slave.anchor_map: the anchor a
non-slaved wall neighbour, G_k its least-squares fit over its own non-slaved neighbours), so u = P q with q the masters'
motion. Particles are added in nested levels: a level keeps every earlier slaved particle with its earlier map and adds new
ones whose anchors and fits use only the current masters, so each level's admissible motions are a subspace of the last's.

The step's three operations, after build():
  P       v_s = sum_k E_sk v_k on the slaved particles (E: the levels composed onto the final masters), the masters kept
  P^T     r_k + sum_s E_sk r_s on the masters
  solve   M_r^-1 f for M_r = P^T M P = (D + E^T diag(m_s) E) (x) I_3 (the same on x, y, z), exactly by Woodbury:
          M_r^-1 = D^-1 - D^-1 E^T S^-1 E D^-1, S = diag(1 / m_s) + E D^-1 E^T (n_s x n_s, SPD), S^-1 kept dense
Each is a Warp kernel over rows (P: the slaved particles; P^T and the solve's E^T: the masters, by E's columns), no atomics
there; the dense S^-1 product splits each row over 32 threads (atomics into z). P is one captured graph, P^T + solve another. Torch versions (level by level, and the CG solve) are kept for
the checks only."""
import numpy as np
import scipy.sparse as sp
import torch
import warp as wp

from free_surface.k1_follow_warp import anchor_map_warp
from free_surface.k1_slave import anchor_map  # noqa: F401 (the reference)

f64, dev = torch.float64, 'cuda'


@wp.kernel(enable_backward=False)
def k_fm_P(v: wp.array(dtype=wp.vec3d), ptr: wp.array(dtype=int), col: wp.array(dtype=int), val: wp.array(dtype=wp.float64), sid: wp.array(dtype=int)):
    i = wp.tid()
    a = wp.vec3d()
    for e in range(ptr[i], ptr[i+1]):
        a = a+val[e]*v[col[e]]
    v[sid[i]] = a


@wp.kernel(enable_backward=False)
def k_fm_Pt(r: wp.array(dtype=wp.vec3d), cptr: wp.array(dtype=int), crow: wp.array(dtype=int), cval: wp.array(dtype=wp.float64),
            sid: wp.array(dtype=int), scale: wp.array(dtype=wp.float64), out: wp.array(dtype=wp.vec3d)):
    """out_k = scale_k (r_k + sum_s E_sk r_s); scale 0 on the slaved particles (1: P^T, D^-1: the solve's first step)"""
    k = wp.tid()
    a = r[k]
    for e in range(cptr[k], cptr[k+1]):
        a = a+cval[e]*r[sid[crow[e]]]
    out[k] = scale[k]*a


@wp.kernel(enable_backward=False)
def k_fm_E(y: wp.array(dtype=wp.vec3d), ptr: wp.array(dtype=int), col: wp.array(dtype=int), val: wp.array(dtype=wp.float64), t: wp.array(dtype=wp.vec3d)):
    i = wp.tid()
    a = wp.vec3d()
    for e in range(ptr[i], ptr[i+1]):
        a = a+val[e]*y[col[e]]
    t[i] = a


FM_NB = 32                                                                           # (the dense product's row split)


@wp.kernel(enable_backward=False)
def k_fm_S(Si: wp.array2d(dtype=wp.float64), t: wp.array(dtype=wp.vec3d), ns: int, z: wp.array(dtype=wp.vec3d)):
    """z_i += sum over j = b, b + NB, ... of S^-1_ij t_j (z zeroed before; S^-1 symmetric, read by column so that
    neighbouring i read neighbouring entries); the row's sum split over NB threads"""
    i, b = wp.tid()
    a = wp.vec3d()
    j = b
    while j < ns:
        a = a+Si[j, i]*t[j]
        j += FM_NB
    wp.atomic_add(z, i, a)


@wp.kernel(enable_backward=False)
def k_fm_Et(y: wp.array(dtype=wp.vec3d), z: wp.array(dtype=wp.vec3d), cptr: wp.array(dtype=int), crow: wp.array(dtype=int),
            cval: wp.array(dtype=wp.float64), Dinv: wp.array(dtype=wp.float64), x: wp.array(dtype=wp.vec3d)):
    k = wp.tid()
    a = wp.vec3d()
    for e in range(cptr[k], cptr[k+1]):
        a = a+cval[e]*z[crow[e]]
    x[k] = y[k]-Dinv[k]*a


class FollowMap:
    def __init__(self, n):
        self.n = n
        self.levels = []                                                             # (s, k, w) per level, as anchor_map gave them
        self.slave = torch.zeros(n, dtype=torch.bool, device=dev)
        self.ns = 0

    @property
    def master(self):
        return ~self.slave

    def add(self, X, wi, wj, wa, new):
        """slave the particles in new that are not yet slaved; returns (slaved now, left free: no anchor, max |w|)"""
        new = new & ~self.slave
        if not bool(new.any()):
            return 0, 0, 0.
        s_, k_, w_, free, _ = anchor_map_warp(X, wi, wj, wa, self.slave | new)    # (11-06: the Warp copy, identical rows, 3 ms against 110-226 ms)
        keep = new[s_]
        if not bool(keep.any()):                                                     # (none of them found an anchor)
            return 0, int(new.sum()), 0.
        got = torch.zeros(self.n, dtype=torch.bool, device=dev); got[s_[keep]] = True
        self.levels.append((s_[keep], k_[keep], w_[keep]))
        self.slave |= got
        return int(got.sum()), int((new & ~got).sum()), float(w_[keep].abs().max())

    # ---- the torch references (checks only)
    def P_ref(self, y):
        v = y*self.master[:, None]
        for s_, k_, w_ in reversed(self.levels):
            v = v.index_add(0, s_, w_[:, None]*v[k_])
        return v

    def Pt_ref(self, r):
        r = r.clone()
        for s_, k_, w_ in self.levels:
            r = r.index_add(0, k_, w_[:, None]*r[s_])
            r[s_] = 0.
        return r*self.master[:, None]

    def M_ref(self, m, y):
        return self.Pt_ref(m[:, None]*self.P_ref(y))

    # ---- the step's operations
    def build(self, m):
        """E expanded onto the final masters, the Woodbury factor, the Warp arrays and the two graphs"""
        n = self.n; master = self.master.cpu().numpy(); mm = m.cpu().numpy()
        rows = {}
        for s_, k_, w_ in reversed(self.levels):                                     # (the last level points at final masters only)
            for sv, kk, ww in zip(s_.cpu().numpy(), k_.cpu().numpy(), w_.cpu().numpy()):
                row = rows.setdefault(int(sv), {})
                if master[kk]:
                    row[int(kk)] = row.get(int(kk), 0.)+float(ww)
                else:
                    for k2, w2 in rows[int(kk)].items():
                        row[k2] = row.get(k2, 0.)+float(ww)*w2
        sl = np.array(sorted(rows), dtype=np.int64); self.ns = ns = len(sl)
        ri = [r_ for r_, sv in enumerate(sl) for _ in rows[sv]]
        ci = [k2 for sv in sl for k2 in rows[sv]]; vv = [w2 for sv in sl for w2 in rows[sv].values()]
        E = sp.csr_matrix((vv, (ri, ci)), shape=(ns, n)); Ec = E.tocsc()
        Di = np.where(master, 1/mm, 0.); one = master.astype(np.float64)
        if ns:
            S = (E@sp.diags(Di)@E.T).toarray()+np.diag(1/mm[sl])
            Sinv = torch.cholesky_inverse(torch.linalg.cholesky(torch.as_tensor(S, dtype=f64, device=dev)))
        else:
            Sinv = torch.zeros(1, 1, dtype=f64, device=dev)
        I = lambda a_: wp.array(np.asarray(a_, dtype=np.int32), dtype=int, device=dev)
        F = lambda a_: wp.array(np.asarray(a_, dtype=np.float64), dtype=wp.float64, device=dev)
        self.wa = dict(ptr=I(E.indptr), col=I(E.indices), val=F(E.data), sid=I(sl if ns else [0]),
                       cptr=I(Ec.indptr), crow=I(Ec.indices), cval=F(Ec.data), Dinv=F(Di), one=F(one),
                       Si=wp.from_torch(Sinv.contiguous(), dtype=wp.float64))
        z3 = lambda k_: wp.zeros(max(k_, 1), dtype=wp.vec3d, device=dev)
        self.wv = dict(v=z3(n), r=z3(n), y=z3(n), x=z3(n), q=z3(n), t=z3(ns), z=z3(ns))
        self.tv = {k_: wp.to_torch(a_, requires_grad=False) for k_, a_ in self.wv.items()}
        a, v = self.wa, self.wv

        def runP():
            if ns:
                wp.launch(k_fm_P, ns, inputs=[v['v'], a['ptr'], a['col'], a['val'], a['sid']])

        def runPt():
            wp.launch(k_fm_Pt, n, inputs=[v['r'], a['cptr'], a['crow'], a['cval'], a['sid'], a['one']], outputs=[v['q']])

        def runSolve():
            wp.launch(k_fm_Pt, n, inputs=[v['r'], a['cptr'], a['crow'], a['cval'], a['sid'], a['Dinv']], outputs=[v['y']])
            if ns:
                wp.launch(k_fm_E, ns, inputs=[v['y'], a['ptr'], a['col'], a['val']], outputs=[v['t']])
                v['z'].zero_()
                wp.launch(k_fm_S, (ns, FM_NB), inputs=[a['Si'], v['t'], ns], outputs=[v['z']])
            wp.launch(k_fm_Et, n, inputs=[v['y'], v['z'], a['cptr'], a['crow'], a['cval'], a['Dinv']], outputs=[v['x']])
        self.graphs = {}
        for name, fn in (('P', runP), ('Pt', runPt), ('force', runSolve)):
            if name == 'P' and not ns:
                continue
            fn()                                                                     # (compile outside the capture)
            wp.capture_begin(); fn(); self.graphs[name] = wp.capture_end()

    def P(self, y):
        """all particles' motion from the masters' (torch in and out)"""
        self.tv['v'].copy_(y*self.master[:, None])
        if 'P' in self.graphs:
            wp.capture_launch(self.graphs['P'])
        return self.tv['v'].clone()

    def Pt(self, r):
        self.tv['r'].copy_(r); wp.capture_launch(self.graphs['Pt'])
        return self.tv['q'].clone()

    def force_to_accel(self, r):
        """M_r^-1 P^T r: the masters' accelerations from all particles' forces"""
        self.tv['r'].copy_(r); wp.capture_launch(self.graphs['force'])
        return self.tv['x'].clone()


# ---- the follow rows, chosen one particle at a time (11-06, after the expert's third reply)
def neighbour_lists(X, wi, wj, wa):
    """per particle: [(neighbour, shared wall area, A / L)] from the cut's walls (numpy)"""
    n = len(X); nb = [[] for _ in range(n)]
    for p_, q_, a_ in zip(wi, wj, wa):
        L = float(np.linalg.norm(X[q_]-X[p_])); nb[p_].append((int(q_), float(a_), float(a_)/L)); nb[q_].append((int(p_), float(a_), float(a_)/L))
    return nb


def _fit(X, nb, k, avail, ring2):
    """k's least-squares gradient over its available neighbours (the 2-ring when ring2 and the 1-ring does not span):
    (points, rows w d^T M^-1, eigenvalues of M, ring) or (None, eigenvalues, count, ring) when it does not span"""
    for ring in ((1, 2) if ring2 else (1,)):
        pts = {}
        for o, _, w in nb[k]:
            if avail[o]:
                pts[o] = pts.get(o, 0.)+w
        if ring == 2:
            for o, _, _ in nb[k]:
                for o2, _, w2 in nb[o]:
                    if o2 != k and avail[o2] and o2 not in pts:
                        pts[o2] = .5*w2
        ev = np.zeros(3)
        if len(pts) >= 3:
            o_ = np.array(list(pts)); w_ = np.array([pts[o] for o in o_]); d = X[o_]-X[k]
            M = (w_[:, None, None]*d[:, :, None]*d[:, None, :]).sum(0); ev = np.linalg.eigvalsh(M)
            if ev[0] > 1e-3*ev[-1]:
                return o_, (w_[:, None]*d)@np.linalg.inv(M), ev, ring
        last = (None, ev, len(pts), ring)
    return last


def anchor_row(X, nb, s, avail, allcand=True, ring2=False, wmax=4.):
    """u_s = u_k + G_k (x_s - x_k) for the best anchor k among s's available wall neighbours (all, or the 3 of the largest
    shared walls); returns (points, weights) or None, and the per-candidate record for the diagnosis"""
    cands = sorted([(a_, k) for k, a_, _ in nb[s] if avail[k]], reverse=True)
    if not allcand:
        cands = cands[:3]
    rec, best = [], None
    for a_, k in cands:
        f = _fit(X, nb, k, avail, ring2)
        if f[0] is None:
            rec.append((k, a_, 'fit does not span (%d points, ring %d, eigenvalues %s)' % (f[2], f[3], np.array2string(f[1], precision=2))))
            continue
        o_, rows, ev, ring = f; c = rows@(X[s]-X[k])
        ks = np.concatenate([[k], o_]); ws = np.concatenate([[1-c.sum()], c]); mw = np.abs(ws).max()
        rec.append((k, a_, 'ok: ring %d, %d points, eigenvalue ratio %.1e, max |w| %.2f' % (ring, len(o_), ev[0]/ev[-1], mw)))
        if best is None or mw < best[2]:
            best = (ks, ws, mw)
        if mw <= wmax:
            break
    return (best[:2] if best else None), rec


def direct_row(X, nb, s, avail, ring2=True, cond_max=1e8):
    """u_s = sum_k w_k u_k over available points near s, the minimum-norm weights with sum w = 1 and sum w x = x_s (affine
    exact), each point weighted by 1 / |x_k - x_s|^2; the 1-ring, else the 2-ring"""
    for ring in ((1, 2) if ring2 else (1,)):
        pts = set(k for k, _, _ in nb[s] if avail[k])
        if ring == 2:
            pts |= set(o2 for k, _, _ in nb[s] for o2, _, _ in nb[k] if avail[o2] and o2 != s)
        if len(pts) < 4:
            continue
        ks = np.array(sorted(pts)); d = X[ks]-X[s]; om = 1/np.maximum((d*d).sum(1), 1e-30)
        A = np.vstack([np.ones(len(ks)), d.T])                                       # (4, N): sum w = 1, sum w d = 0
        G = (A*om)@A.T
        if np.linalg.cond(G) > cond_max:
            continue
        w = om*(A.T@np.linalg.solve(G, np.array([1., 0., 0., 0.])))
        return (ks, w), [('direct', 0., 'ring %d, %d points' % (ring, len(ks)))]
    return None, [('direct', 0., 'no well-conditioned point set')]


def direct_row_scaled(X, nb, s, avail, ratio_min=1e-3, wmax=4.):
    """direct_row with the expert's fourth-reply fixes: the offsets d made dimensionless by the stencil's rms length (the
    weights do not change, the acceptance no longer depends on the unit), the minimum-norm weights by a QR of the weighted
    constraint matrix (no normal matrix), and the anchor fit's quality rules: the stencil must span, the weighted centred
    second moment's eigenvalue ratio above 1e-3 (the anchor fit's threshold), and |w| <= 4 preferred: the 2-ring tried when the
    1-ring does not span or exceeds it, the smaller max |w| kept"""
    best, rec = None, []
    for ring in (1, 2):
        pts = set(k for k, _, _ in nb[s] if avail[k])
        if ring == 2:
            pts |= set(o2 for k, _, _ in nb[s] for o2, _, _ in nb[k] if avail[o2] and o2 != s)
        if len(pts) < 4:
            rec.append(('direct', 0., 'ring %d: %d points' % (ring, len(pts)))); continue
        ks = np.array(sorted(pts)); d = X[ks]-X[s]; h = float(np.sqrt((d*d).sum(1).mean())); dt = d/h
        om = 1/np.maximum((dt*dt).sum(1), 1e-30)
        c = (om[:, None]*dt).sum(0)/om.sum(); e_ = dt-c
        ev = np.linalg.eigvalsh((om[:, None, None]*e_[:, :, None]*e_[:, None, :]).sum(0)/om.sum())
        if ev[0] < ratio_min*ev[-1]:
            rec.append(('direct', 0., 'ring %d, %d points: does not span (ratio %.1e)' % (ring, len(ks), ev[0]/ev[-1]))); continue
        so = np.sqrt(om); B = np.vstack([np.ones(len(ks)), dt.T])*so                 # (4, N)
        Qm, R = np.linalg.qr(B.T)
        w = so*(Qm@np.linalg.solve(R.T, np.array([1., 0., 0., 0.])))
        err = float(np.abs(np.vstack([np.ones(len(ks)), dt.T])@w-np.array([1., 0., 0., 0.])).max())
        mw = float(np.abs(w).max())
        rec.append(('direct', 0., 'ring %d, %d points, ratio %.1e, max |w| %.2f, constraint error %.0e' % (ring, len(ks), ev[0]/ev[-1], mw, err)))
        if best is None or mw < best[2]:
            best = (ks, w, mw)
        if mw <= wmax:
            break
    return (best[:2] if best else None), rec


def add_sequential(FM, X, nb, cand, order, mode='anchor', allcand=True, ring2=False, watch=(), log=None):
    """slave the particles of cand one at a time in the given order, each as its own level: its row uses only the particles
    that are masters at that moment (it may use particles of cand still to come; when one of those is slaved later, the
    later level's row is substituted, so the levels stay nested and acyclic). Returns (slaved, left free)."""
    n = FM.n; Xn = X.cpu().numpy() if torch.is_tensor(X) else X
    avail = (~FM.slave).cpu().numpy().copy(); got = free = 0
    for s in order:
        if not cand[s] or not avail[s]:
            continue
        avail[s] = False
        if mode == 'anchor':
            row, rec = anchor_row(Xn, nb, s, avail, allcand, ring2)
        elif mode == 'direct_scaled':
            row, rec = direct_row_scaled(Xn, nb, s, avail)
        else:
            row, rec = direct_row(Xn, nb, s, avail)
        if s in watch and log is not None:
            log('    WATCH %d (%s): %s' % (s, mode, ' | '.join('%s area %.2e: %s' % (k, a_, r_) for k, a_, r_ in rec) if rec else 'no available wall neighbour'))
        if row is None:
            avail[s] = True; free += 1; continue
        ks, ws = row
        T = lambda a, dt=torch.long: torch.as_tensor(np.asarray(a), dtype=dt, device=dev)
        FM.levels.append((T([s]*len(ks)), T(ks), T(ws, f64)))
        FM.slave[s] = True; got += 1
    return got, free


def row_stats(FM, m):
    """per slaved particle, on its row expanded onto the final masters: max |w|, sum |w|, m_s sum w^2 / m_k"""
    master = FM.master.cpu().numpy(); mm = m.cpu().numpy(); rows = {}
    for s_, k_, w_ in reversed(FM.levels):
        for sv, kk, ww in zip(s_.cpu().numpy(), k_.cpu().numpy(), w_.cpu().numpy()):
            row = rows.setdefault(int(sv), {})
            if master[kk]:
                row[int(kk)] = row.get(int(kk), 0.)+float(ww)
            else:
                for k2, w2 in rows[int(kk)].items():
                    row[k2] = row.get(k2, 0.)+float(ww)*w2
    if not rows:
        return dict(n=0)
    wmax = np.array([max(abs(v) for v in r.values()) for r in rows.values()])
    wsum = np.array([sum(abs(v) for v in r.values()) for r in rows.values()])
    mamp = np.array([mm[s]*sum(v*v/mm[k] for k, v in r.items()) for s, r in rows.items()])
    q = lambda a: (float(np.percentile(a, 99)), float(a.max()))
    return dict(n=len(rows), wmax=q(wmax), wsum=q(wsum), mamp=q(mamp))


def add_batch(FM, X, nb, cand, allcand=True, ring2=False, watch=(), log=None):
    """the earlier batch rule with a choice of candidates: every particle of cand at once, its row using only the particles
    neither slaved nor in cand (so no row refers to another of the batch), one level; returns (slaved, left free)"""
    Xn = X.cpu().numpy() if torch.is_tensor(X) else X
    avail = (~FM.slave).cpu().numpy() & ~cand
    S_, K_, W_ = [], [], []; got = free = 0
    for s in np.nonzero(cand & (~FM.slave).cpu().numpy())[0]:
        row, rec = anchor_row(Xn, nb, int(s), avail, allcand, ring2)
        if int(s) in watch and log is not None:
            log('    WATCH %d (batch): %s' % (s, ' | '.join('%s area %.2e: %s' % (k, a_, r_) for k, a_, r_ in rec) if rec else 'no available wall neighbour'))
        if row is None:
            free += 1; continue
        ks, ws = row; S_ += [int(s)]*len(ks); K_ += ks.tolist(); W_ += ws.tolist(); got += 1
    if got:
        T = lambda a, dt=torch.long: torch.as_tensor(np.asarray(a), dtype=dt, device=dev)
        FM.levels.append((T(S_), T(K_), T(W_, f64)))
        FM.slave[torch.as_tensor(sorted(set(S_)), device=dev)] = True
    return got, free


# ---- the rebuild on the GPU (11-06: FollowMap.build was 17-24 ms per cut: Python expansion of the nested rows, S on the CPU,
# three graphs captured again each time). Buffers by capacity, the slaved count read from an array, the graphs captured once
# per capacity; the rows composed by vectorised substitution, S = E D^-1 E^T + diag(1 / m_s) summed per master column in Warp.
@wp.kernel(enable_backward=False)
def k_gm_P(v: wp.array(dtype=wp.vec3d), ptr: wp.array(dtype=int), col: wp.array(dtype=int), val: wp.array(dtype=wp.float64), sid: wp.array(dtype=int),
           nsa: wp.array(dtype=int)):
    i = wp.tid()
    if i >= nsa[0]:
        return
    a = wp.vec3d()
    for e in range(ptr[i], ptr[i+1]):
        a = a+val[e]*v[col[e]]
    v[sid[i]] = a


@wp.kernel(enable_backward=False)
def k_gm_E(y: wp.array(dtype=wp.vec3d), ptr: wp.array(dtype=int), col: wp.array(dtype=int), val: wp.array(dtype=wp.float64), nsa: wp.array(dtype=int),
           t: wp.array(dtype=wp.vec3d)):
    i = wp.tid()
    if i >= nsa[0]:
        return
    a = wp.vec3d()
    for e in range(ptr[i], ptr[i+1]):
        a = a+val[e]*y[col[e]]
    t[i] = a


@wp.kernel(enable_backward=False)
def k_gm_S(Si: wp.array2d(dtype=wp.float64), t: wp.array(dtype=wp.vec3d), nsa: wp.array(dtype=int), z: wp.array(dtype=wp.vec3d)):
    i, b = wp.tid()
    ns = nsa[0]
    if i >= ns:
        return
    a = wp.vec3d()
    j = b
    while j < ns:
        a = a+Si[j, i]*t[j]
        j += FM_NB
    wp.atomic_add(z, i, a)


@wp.kernel(enable_backward=False)
def k_gm_Sasm(cptr: wp.array(dtype=int), crow: wp.array(dtype=int), cval: wp.array(dtype=wp.float64), Dinv: wp.array(dtype=wp.float64),
              S: wp.array2d(dtype=wp.float64)):
    """S_ab += E_ak E_bk / m_k over master k's column"""
    k = wp.tid()
    d = Dinv[k]
    for e in range(cptr[k], cptr[k+1]):
        for f in range(cptr[k], cptr[k+1]):
            wp.atomic_add(S, crow[e], crow[f], cval[e]*cval[f]*d)


class FollowMapGPU(FollowMap):
    """FollowMap with build() on the GPU; the same P, P^T, M_r^-1 (checked against FollowMap's references)"""

    def __init__(self, n):
        super().__init__(n)
        self._cap = (0, 0)

    def _compose(self):
        """the levels' rows on the final masters: (s, k, w) coalesced"""
        n, dev = self.n, self.slave.device
        has = torch.zeros(n, dtype=torch.bool, device=dev)
        Fs = torch.zeros(0, dtype=torch.long, device=dev); Fk = Fs.clone(); Fw = torch.zeros(0, dtype=f64, device=dev)
        for s_, k_, w_ in reversed(self.levels):
            sub = has[k_]
            ds, dk, dw = s_[~sub], k_[~sub], w_[~sub]
            if bool(sub.any()):
                o = torch.argsort(Fs, stable=True); Ss, Sk, Sw = Fs[o], Fk[o], Fw[o]
                off = torch.zeros(n+1, dtype=torch.long, device=dev); off[1:] = torch.cumsum(torch.bincount(Ss, minlength=n), 0)
                qs, qk, qw = s_[sub], k_[sub], w_[sub]
                cnt = off[qk+1]-off[qk]; src = torch.repeat_interleave(torch.arange(len(qk), device=dev), cnt)
                pos = off[qk][src]+(torch.arange(int(cnt.sum()), device=dev)-torch.repeat_interleave(torch.cumsum(cnt, 0)-cnt, cnt))
                ds, dk, dw = torch.cat([ds, qs[src]]), torch.cat([dk, Sk[pos]]), torch.cat([dw, qw[src]*Sw[pos]])
            key, inv = torch.unique(ds*n+dk, return_inverse=True)
            ww = torch.zeros(len(key), dtype=f64, device=dev).index_add(0, inv, dw)
            Fs, Fk, Fw = torch.cat([Fs, key//n]), torch.cat([Fk, key % n]), torch.cat([Fw, ww]); has[key//n] = True
        return Fs, Fk, Fw

    def _alloc(self, ns, nnz):
        n = self.n; dev = str(self.slave.device)
        cs, cz = max(int(1.5*ns)+16, 64), max(int(1.5*nnz)+64, 256)
        I = lambda k_: wp.zeros(k_, dtype=int, device=dev); Fz = lambda k_: wp.zeros(k_, dtype=wp.float64, device=dev)
        self.wa = dict(ptr=I(cs+1), col=I(cz), val=Fz(cz), sid=I(cs), cptr=I(n+1), crow=I(cz), cval=Fz(cz), Dinv=Fz(n), one=Fz(n), nsa=I(1),
                       Si=wp.zeros((cs, cs), dtype=wp.float64, device=dev))
        z3 = lambda k_: wp.zeros(k_, dtype=wp.vec3d, device=dev)
        self.wv = dict(v=z3(n), r=z3(n), y=z3(n), x=z3(n), q=z3(n), t=z3(cs), z=z3(cs))
        self.tv = {k_: wp.to_torch(a_, requires_grad=False) for k_, a_ in self.wv.items()}
        self.ta = {k_: wp.to_torch(a_, requires_grad=False) for k_, a_ in self.wa.items()}
        a, v = self.wa, self.wv

        def runP():
            wp.launch(k_gm_P, cs, inputs=[v['v'], a['ptr'], a['col'], a['val'], a['sid'], a['nsa']])

        def runPt():
            wp.launch(k_fm_Pt, n, inputs=[v['r'], a['cptr'], a['crow'], a['cval'], a['sid'], a['one']], outputs=[v['q']])

        def runSolve():
            wp.launch(k_fm_Pt, n, inputs=[v['r'], a['cptr'], a['crow'], a['cval'], a['sid'], a['Dinv']], outputs=[v['y']])
            wp.launch(k_gm_E, cs, inputs=[v['y'], a['ptr'], a['col'], a['val'], a['nsa']], outputs=[v['t']])
            v['z'].zero_()
            wp.launch(k_gm_S, (cs, FM_NB), inputs=[a['Si'], v['t'], a['nsa']], outputs=[v['z']])
            wp.launch(k_fm_Et, n, inputs=[v['y'], v['z'], a['cptr'], a['crow'], a['cval'], a['Dinv']], outputs=[v['x']])
        self.graphs = {}
        for name, fn in (('P', runP), ('Pt', runPt), ('force', runSolve)):
            fn()
            wp.capture_begin(); fn(); self.graphs[name] = wp.capture_end()
        self._cap = (cs, cz)

    def build(self, m):
        n, dev = self.n, self.slave.device
        T_ = getattr(self, 'timing', None); sy = torch.cuda.synchronize; pc = __import__('time').perf_counter
        if T_ is not None:
            sy(); T_['t0'] = pc()
        Fs, Fk, Fw = self._compose()
        if T_ is not None:
            sy(); T_['compose'] = pc()
        sl, rinv = torch.unique(Fs, return_inverse=True); ns = len(sl); nnz = len(Fs); self.ns = ns
        if ns > self._cap[0] or nnz > self._cap[1]:
            self._alloc(ns, nnz)
        ta = self.ta
        o = torch.argsort(rinv*n+Fk); r_, k_, w_ = rinv[o], Fk[o], Fw[o]
        ta['ptr'].zero_(); ta['ptr'][1:ns+1] = torch.cumsum(torch.bincount(r_, minlength=ns), 0)
        ta['col'][:nnz] = k_; ta['val'][:nnz] = w_; ta['sid'][:ns] = sl
        oc = torch.argsort(k_, stable=True)
        ta['cptr'][0] = 0; ta['cptr'][1:] = torch.cumsum(torch.bincount(k_, minlength=n), 0)
        ta['crow'][:nnz] = r_[oc]; ta['cval'][:nnz] = w_[oc]
        master = self.master
        ta['Dinv'].copy_(torch.where(master, 1/m, torch.zeros_like(m))); ta['one'].copy_(master.to(f64)); ta['nsa'][0] = ns
        if T_ is not None:
            sy(); T_['fill'] = pc()
        if ns:
            S = ta['Si'][:ns, :ns]; S.zero_()
            wp.launch(k_gm_Sasm, n, inputs=[self.wa['cptr'], self.wa['crow'], self.wa['cval'], self.wa['Dinv']], outputs=[self.wa['Si']])
            S += torch.diag(1/m[sl])
            if T_ is not None:
                sy(); T_['assemble'] = pc()
            S.copy_(torch.cholesky_inverse(torch.linalg.cholesky(S.clone())))
            if T_ is not None:
                sy(); T_['factor'] = pc()

    def P(self, y):
        self.tv['v'].copy_(y*self.master[:, None])
        if self.ns:
            wp.capture_launch(self.graphs['P'])
        return self.tv['v'].clone()


# ---- the rows' composition in Warp and the solve by the Cholesky factor (11-06: compose 9.9 ms by torch.unique chains;
# the explicit inverse 7.5 ms of a 10 ms factorisation, the factor alone 2.8 ms and its two triangular solves 0.3 ms a step)
RCAP = 256


@wp.kernel(enable_backward=False)
def k_gm_compose(ls: wp.array(dtype=int), lptr: wp.array(dtype=int), lk: wp.array(dtype=int), lw: wp.array(dtype=wp.float64),
                 fin: wp.array(dtype=int), cap: int, rk: wp.array2d(dtype=int), rw: wp.array2d(dtype=wp.float64), rn: wp.array(dtype=int),
                 over: wp.array(dtype=int)):
    """one level, one thread per slaved particle p = ls[i] with its row lk, lw in [lptr[i], lptr[i+1]): an entry on a
    particle with a final row (fin >= 0: slaved at a later level, already composed) is replaced by that row times w; the
    rest kept; equal masters merged. The final row into p's slots"""
    i = wp.tid()
    p = ls[i]
    c = int(0)
    for e in range(lptr[i], lptr[i+1]):
        k = lk[e]
        w = lw[e]
        if fin[k] >= 0:
            for f in range(rn[k]):
                k2 = rk[k, f]
                w2 = w*rw[k, f]
                hit = int(0)
                for q in range(c):
                    if rk[p, q] == k2:
                        rw[p, q] = rw[p, q]+w2
                        hit = 1
                        break
                if hit == 0:
                    if c < cap:
                        rk[p, c] = k2
                        rw[p, c] = w2
                        c += 1
                    else:
                        over[0] = 1
        else:
            hit = int(0)
            for q in range(c):
                if rk[p, q] == k:
                    rw[p, q] = rw[p, q]+w
                    hit = 1
                    break
            if hit == 0:
                if c < cap:
                    rk[p, c] = k
                    rw[p, c] = w
                    c += 1
                else:
                    over[0] = 1
    rn[p] = c
    fin[p] = 1


@wp.kernel(enable_backward=False)
def k_gm_pack(sl: wp.array(dtype=int), rk: wp.array2d(dtype=int), rw: wp.array2d(dtype=wp.float64), ptr: wp.array(dtype=int),
              col: wp.array(dtype=int), val: wp.array(dtype=wp.float64)):
    r = wp.tid()
    p = sl[r]
    o = ptr[r]
    for q in range(ptr[r+1]-ptr[r]):
        col[o+q] = rk[p, q]
        val[o+q] = rw[p, q]


class FollowMapGPU2(FollowMapGPU):
    """FollowMapGPU with the rows composed in Warp and, with factor='cholesky', M_r^-1 through S's Cholesky factor (two
    triangular solves a step) instead of the dense S^-1"""

    def __init__(self, n, factor='cholesky'):
        super().__init__(n)
        self.factor = factor
        dev = str(self.slave.device)
        self.rk = wp.zeros((n, RCAP), dtype=int, device=dev); self.rw = wp.zeros((n, RCAP), dtype=wp.float64, device=dev)
        self.rn = wp.zeros(n, dtype=int, device=dev); self.fin = wp.zeros(n, dtype=int, device=dev); self.over = wp.zeros(1, dtype=int, device=dev)

    def reset(self):
        """a new map on the same buffers and graphs (the next cut's): no levels, nothing slaved"""
        self.levels = []; self.slave.zero_(); self.ns = 0

    def add(self, X, wi, wj, wa, new, nb=None):
        """FollowMap.add with the walls' neighbour lists given once per cut (nb: k1_follow_warp.neighbours)"""
        new = new & ~self.slave
        s_, k_, w_, free, _ = anchor_map_warp(X, wi, wj, wa, self.slave | new, nb=nb)
        keep = new[s_]
        self.levels.append((s_[keep], k_[keep], w_[keep]))
        self.slave[s_[keep]] = True
        return None

    def _compose_warp(self):
        dev = self.slave.device; T = lambda a_: wp.to_torch(a_, requires_grad=False)
        T(self.fin).fill_(-1); T(self.rn).zero_(); T(self.over).zero_()
        for s_, k_, w_ in reversed(self.levels):
            o = torch.argsort(s_, stable=True); s_, k_, w_ = s_[o], k_[o], w_[o]
            ls, cnt = torch.unique_consecutive(s_, return_counts=True)
            lptr = torch.zeros(len(ls)+1, dtype=torch.long, device=dev); lptr[1:] = torch.cumsum(cnt, 0)
            I = lambda x_: wp.from_torch(x_.to(torch.int32).contiguous(), dtype=wp.int32)
            wp.launch(k_gm_compose, len(ls), inputs=[I(ls), I(lptr), I(k_), wp.from_torch(w_.contiguous(), dtype=wp.float64), self.fin, RCAP],
                      outputs=[self.rk, self.rw, self.rn, self.over], device=str(dev))
        return T(self.fin) >= 0

    def build(self, m):
        n, dev = self.n, self.slave.device
        T_ = getattr(self, 'timing', None); sy = torch.cuda.synchronize; pc = __import__('time').perf_counter
        if T_ is not None:
            sy(); T_['t0'] = pc()
        isl = self._compose_warp()
        T = lambda a_: wp.to_torch(a_, requires_grad=False)
        sl = torch.nonzero(isl).flatten(); ns = len(sl); self.ns = ns
        rn = T(self.rn)[sl].long(); nnz = int(rn.sum())
        if T_ is not None:
            sy(); T_['compose'] = pc()
        if int(self.over.numpy()[0]):
            raise RuntimeError('FollowMapGPU2: a composed row over %d entries' % RCAP)
        if ns > self._cap[0] or nnz > self._cap[1]:
            self._alloc(ns, nnz)
        ta, wa = self.ta, self.wa
        ta['ptr'].zero_(); ta['ptr'][1:ns+1] = torch.cumsum(rn, 0); ta['sid'][:ns] = sl
        if ns:
            wp.launch(k_gm_pack, ns, inputs=[wa['sid'], self.rk, self.rw, wa['ptr']], outputs=[wa['col'], wa['val']], device=str(dev))
        r_ = torch.repeat_interleave(torch.arange(ns, device=dev), rn); k_ = ta['col'][:nnz].long(); w_ = ta['val'][:nnz]
        oc = torch.argsort(k_, stable=True)
        ta['cptr'][0] = 0; ta['cptr'][1:] = torch.cumsum(torch.bincount(k_, minlength=n), 0)
        ta['crow'][:nnz] = r_[oc]; ta['cval'][:nnz] = w_[oc]
        master = self.master
        ta['Dinv'].copy_(torch.where(master, 1/m, torch.zeros_like(m))); ta['one'].copy_(master.to(f64)); ta['nsa'][0] = ns
        if T_ is not None:
            sy(); T_['fill'] = pc()
        if ns:
            S = ta['Si'][:ns, :ns]; S.zero_()
            wp.launch(k_gm_Sasm, n, inputs=[wa['cptr'], wa['crow'], wa['cval'], wa['Dinv']], outputs=[wa['Si']], device=str(dev))
            S += torch.diag(1/m[sl])
            if T_ is not None:
                sy(); T_['assemble'] = pc()
            if self.factor == 'cholesky':
                self.L = torch.linalg.cholesky(S.clone())
            else:
                S.copy_(torch.cholesky_inverse(torch.linalg.cholesky(S.clone())))
            if T_ is not None:
                sy(); T_['factor'] = pc()
        if self.factor == 'cholesky' and not hasattr(self, 'g_pre'):
            a, v = wa, self.wv; cs = self._cap[0]
            def pre():
                wp.launch(k_fm_Pt, n, inputs=[v['r'], a['cptr'], a['crow'], a['cval'], a['sid'], a['Dinv']], outputs=[v['y']])
                wp.launch(k_gm_E, cs, inputs=[v['y'], a['ptr'], a['col'], a['val'], a['nsa']], outputs=[v['t']])
            def post():
                wp.launch(k_fm_Et, n, inputs=[v['y'], v['z'], a['cptr'], a['crow'], a['cval'], a['Dinv']], outputs=[v['x']])
            pre(); wp.capture_begin(); pre(); self.g_pre = wp.capture_end()
            post(); wp.capture_begin(); post(); self.g_post = wp.capture_end()

    def _alloc(self, ns, nnz):
        super()._alloc(ns, nnz)
        for k_ in ('g_pre', 'g_post'):
            if hasattr(self, k_):
                delattr(self, k_)

    def force_to_accel(self, r):
        if self.factor != 'cholesky':
            return super().force_to_accel(r)
        self.tv['r'].copy_(r); wp.capture_launch(self.g_pre)
        ns = self.ns
        if ns:
            self.tv['z'][:ns] = torch.cholesky_solve(self.tv['t'][:ns], self.L)
        wp.capture_launch(self.g_post)
        return self.tv['x'].clone()
