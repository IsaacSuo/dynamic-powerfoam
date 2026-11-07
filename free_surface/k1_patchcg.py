"""The stalled-CG fallback (11-06, the expert's full-step test of §23.7–23.8): when a Newton iteration's TrustCG hits its cap, the
same subproblem is solved again at the same point with
  the group    the ncell sites where the CG's residual concentrates (from its residual vector, no extra cost)
  the columns  their full Hessian columns H_:,S by central differences (hc per coordinate; 6 evaluations per site)
  the joint    the group's 3m x 3m block from the columns (symmetrised, eigenvalues floored at the group's smallest 4 m / dt^2),
               the rest the blocks Newton applies (P: ul_blocks through k_mid_pre, the balls' term); non-overlapping
  the product  H v = H_:,S v_S + H_FD(v with S zeroed), H_FD the production one-sided product (h = CG.h)
  the CG       Steihaug-Toint in the M-norm of that preconditioner (the region as TrustCG's), stop |r| <= eta_patch |g| (recursive),
               the model decrease from the accumulated H p
Columns, joint block and group belong to this Newton point: built when triggered, fixed through the one CG, discarded after; the
next Newton point runs the standard CG first. Checks (not in the solve's cost): the split product's symmetry along the first two
search directions, the independent residual at the exit (columns + central FD on the rest; central FD on the whole p). Logging only
besides the step p it hands back (in CG.a['p']); nothing in the model changes."""
import math
import time

import torch
import warp as wp

f64 = torch.float64


def solve_py(ST, g_arr, delta, cap, log, ncell=6, hc=1e-8, eta=.005, tag='', S=None):
    W, CG, A = ST.W, ST.CG, ST.a
    wpt = lambda a_: wp.to_torch(a_, requires_grad=False)
    dev, n = W.dev, W.n
    I3 = torch.eye(3, dtype=f64, device=dev)
    nev, nprod = [0], [0]
    T = dict(select=0., columns=0., factor=0., iterate=0., checks=0.)
    sync = torch.cuda.synchronize
    nrm = lambda t_: float(t_.norm()); dot = lambda a_, b_: float((a_*b_).sum())

    z = CG.t['z'].clone(); gz = wpt(g_arr).clone(); g0 = CG.t['g0'].clone(); cm = CG.t['cm']
    xw, gw = wpt(W.xw), wpt(W.xw.grad)
    gn = nrm(gz)

    def gE(x):
        sync(); xw.copy_(x); W._run(); wp.synchronize(); nev[0] += 1
        return gw.clone()

    # the group from the stalled CG's residual
    sync(); t0 = time.perf_counter()
    rk = CG.t['r'].norm(dim=1)
    if S is None:
        S = torch.argsort(rk, descending=True)[:ncell].tolist()
    gi = torch.as_tensor(S, device=dev)
    share = float((rk[gi]**2).sum()/(rk**2).sum())
    sync(); T['select'] = time.perf_counter()-t0

    # the columns
    t0 = time.perf_counter()
    cols = {}
    for k in S:
        Hk = torch.zeros(n, 3, 3, dtype=f64, device=dev)
        for a in range(3):
            e = torch.zeros(n, 3, dtype=f64, device=dev); e[k, a] = 1.
            Hk[:, :, a] = (gE(z+hc*e)-gE(z-hc*e))/(2*hc)
        Hk[k] += cm[k]*I3; cols[k] = Hk
    sync(); T['columns'] = time.perf_counter()-t0; ev_cols = nev[0]

    # the preconditioner: P as applied, the joint block on the group
    t0 = time.perf_counter()
    P = wpt(A['P']).clone(); Pinv = wpt(A['Pinv']).clone()
    m = len(S); J = torch.zeros(m, 3, m, 3, dtype=f64, device=dev)
    for b, kb in enumerate(S):
        for a, ka in enumerate(S):
            J[a, :, b, :] = cols[kb][ka]
    J = J.reshape(3*m, 3*m); asym = float((J-J.T).abs().max()/J.abs().max()); J = .5*(J+J.T)
    ev, U = torch.linalg.eigh(J); floor = float(cm[gi].min()); nfl = int((ev < floor).sum())
    J = U@torch.diag(torch.maximum(ev, torch.full_like(ev, floor)))@U.T; Ji = torch.linalg.inv(J)
    Pinv_o = Pinv.clone(); Pinv_o[gi] = 0.; P_o = P.clone(); P_o[gi] = 0.

    def Minv(r_):
        y_ = torch.einsum('kab,kb->ka', Pinv_o, r_); y_[gi] = (Ji@r_[gi].reshape(-1)).reshape(m, 3); return y_

    def M(v_):
        y_ = torch.einsum('kab,kb->ka', P_o, v_); y_[gi] = (J@v_[gi].reshape(-1)).reshape(m, 3); return y_
    sync(); T['factor'] = time.perf_counter()-t0

    Hcols = lambda v_: sum(torch.einsum('lab,b->la', cols[k], v_[k]) for k in S)

    def Hfd(v, h=CG.h, central=False):
        s = float(v.abs().max())
        if s == 0.:
            return torch.zeros_like(v)
        if central:
            return cm[:, None]*v+(gE(z+(h/s)*v)-gE(z-(h/s)*v))*(s/(2*h))
        return cm[:, None]*v+(gE(z+(h/s)*v)-g0)*(s/h)

    def Hsplit(v, h=CG.h, central=False):
        vr = v.clone(); vr[gi] = 0.
        nprod[0] += 1
        return Hcols(v)+Hfd(vr, h=h, central=central)

    # Steihaug-Toint PCG in the M-norm
    t0 = time.perf_counter()
    p = torch.zeros_like(gz); Hp = torch.zeros_like(gz); Mp = torch.zeros_like(gz)
    r = -gz.clone(); y = Minv(r); d = y.clone(); Md = r.clone(); rz = dot(r, y)
    if not delta > 0.:
        delta = math.sqrt(rz)                                                     # (as k_tr_start: the preconditioned gradient's length)
    pmp = 0.; k = 0; exit_ = 'cap'; edge = neg = False; dirs = []
    while k < cap:
        hd = Hsplit(d); k += 1
        if len(dirs) < 1:
            dirs.append((d.clone(), hd.clone()))                                  # (d1 and H d1; d2 is H-conjugate to d1, so the check uses y2 = M^-1 r1 instead)
        elif len(dirs) == 1 and k == 2:
            dirs.append((y.clone(), None))
        dhd = dot(d, hd); dmd = dot(d, Md); pmd = dot(p, Md)
        if dhd <= 0.:
            tau = (-pmd+math.sqrt(max(pmd*pmd+dmd*(delta*delta-pmp), 0.)))/dmd
            p += tau*d; Hp += tau*hd; Mp += tau*Md; pmp = delta*delta; exit_ = 'negative'; edge = neg = True; break
        al = rz/dhd
        npmp = pmp+2*al*pmd+al*al*dmd
        if npmp >= delta*delta:
            tau = (-pmd+math.sqrt(max(pmd*pmd+dmd*(delta*delta-pmp), 0.)))/dmd
            p += tau*d; Hp += tau*hd; Mp += tau*Md; pmp = delta*delta; exit_ = 'edge'; edge = True; break
        p += al*d; Hp += al*hd; Mp += al*Md; pmp = npmp; r -= al*hd; y = Minv(r); rz1 = dot(r, y)
        if nrm(r) <= eta*gn:
            exit_ = 'tolerance'; break
        be = rz1/rz; d = y+be*d; Md = r+be*Md; rz = rz1
    sync(); T['iterate'] = time.perf_counter()-t0; ev_cg = nev[0]-ev_cols
    rr = nrm(r)

    # checks (their evaluations and time apart)
    t0 = time.perf_counter(); e0 = nev[0]
    sym = float('nan')
    if len(dirs) == 2:
        (d1, h1), (y2, _) = dirs
        h2 = Hsplit(y2); nprod[0] -= 1                                            # (one product for the check: u = d1, v = y2, not conjugate)
        sym = abs(dot(d1, h2)-dot(y2, h1))/(abs(dot(d1, h2))+abs(dot(y2, h1))+1e-300)
    vr = p.clone(); vr[gi] = 0.
    res_split = nrm(gz+Hcols(p)+Hfd(vr, h=hc, central=True))/gn
    res_cent = nrm(gz+Hfd(p, h=hc, central=True))/gn
    sync(); T['checks'] = time.perf_counter()-t0; ev_chk = nev[0]-e0
    gE(z)                                                                           # (W's buffers back at z)
    wpt(CG.a['p']).copy_(p); wpt(CG.a['Hp']).copy_(Hp)
    gp, php = dot(gz, p), dot(p, Hp)
    log('  PATCHCG-py%s: group %s (%.0f %% of the stalled residual; cells V0/med %s); columns %d evaluations; joint block asymmetry %.1e, eigenvalues %.3g..%.3g (floored %d); '
        'CG exit %s after %d products (%d evaluations), recursive |r| / |g| %.3g (eta %.3g), delta %.3g, |p|_M %.3g; checks: split symmetry u = d1, v = M^-1 r1 %.1e, independent residual columns + central rest %.3g, central on p %.3g (%d evaluations); '
        'time select %.3f, columns %.3f, factor %.3f, iterate %.3f (checks %.3f) s' % (
            tag, S, 100*share, ['%.2f' % float(W.V0[c_]/W.V0.median()) for c_ in S], ev_cols, asym, float(ev[0]), float(ev[-1]), nfl, exit_, k, ev_cg, rr/gn, eta, delta, math.sqrt(pmp),
            sym, res_split, res_cent, ev_chk, T['select'], T['columns'], T['factor'], T['iterate'], T['checks']))
    return dict(K=k, EDGE=edge, NEG=neg, PMP=pmp, DELTA=delta, GP=gp, PHP=php, RR=rr*rr, evals=ev_cols+ev_cg, evals_checks=ev_chk, time=T['select']+T['columns']+T['factor']+T['iterate'],
                time_checks=T['checks'], res_split=res_split, res_cent=res_cent, group=S)


# ---- the same fallback on the device (11-06 "搬吧"): TrustCG's kernels with two changes inside one captured graph of `per` iterations:
#   the product   H d = cm d_r + FD(d_r) + sum_k H_:,k d_k   (d_r = d with the group zeroed; k_sp_mask, k_sp_hd in place of k_tr_hd)
#   the apply     y = Pinv_m r (Pinv with the group's blocks zeroed, k_tr_update) then y_S = J^-1 r_S (k_sp_joint)
# TrustCG's region, alpha, beta, stop and bookkeeping (tr slots) unchanged; the group's size in a device int (mm) so the graph is captured once.
from free_surface.closure_warp import (k_tr_init, k_tr_start, k_tr_absmax, k_tr_perturb, k_tr_alpha, k_tr_update, k_tr_beta, k_tr_dir, k_tr_pred,
                                       TR_RZ, TR_DHD, TR_RR, TR_RZN, TR_PMAX, TR_DONE, TR_DONE0, TR_EDGE, TR_NEG, TR_DELTA, TR_STOP, TR_K, TR_CAP, TR_GP, TR_PHP, TR_PMP)

MCAP = 8


@wp.kernel(enable_backward=False)
def k_sp_mask(d: wp.array(dtype=wp.vec3d), inS: wp.array(dtype=int), d_r: wp.array(dtype=wp.vec3d)):
    i = wp.tid()
    if inS[i] > 0:
        d_r[i] = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
    else:
        d_r[i] = d[i]


@wp.kernel(enable_backward=False)
def k_sp_hd(gE: wp.array(dtype=wp.vec3d), g0: wp.array(dtype=wp.vec3d), d: wp.array(dtype=wp.vec3d), d_r: wp.array(dtype=wp.vec3d), cm: wp.array(dtype=wp.float64),
            cols: wp.array2d(dtype=wp.mat33d), Sidx: wp.array(dtype=int), mm: wp.array(dtype=int), tr: wp.array(dtype=wp.float64), h: wp.float64, Hd: wp.array(dtype=wp.vec3d)):
    """H d = cm d_r + (dE/dx(z + h d_r / |d_r|_max) - dE/dx(z)) |d_r|_max / h + sum_k H_:,k d_k (the columns hold cm on their own diagonal block); d . H d into tr"""
    l = wp.tid()
    hd = cm[l]*d_r[l]+(gE[l]-g0[l])*(wp.max(tr[TR_PMAX], wp.float64(1e-300))/h)
    for k in range(mm[0]):
        hd = hd+cols[k, l]@d[Sidx[k]]
    Hd[l] = hd
    wp.atomic_add(tr, TR_DHD, wp.dot(d[l], hd))


@wp.kernel(enable_backward=False)
def k_sp_joint(r: wp.array(dtype=wp.vec3d), Sidx: wp.array(dtype=int), mm: wp.array(dtype=int), Ji: wp.array(dtype=wp.float64), init: int, tr: wp.array(dtype=wp.float64),
               y: wp.array(dtype=wp.vec3d), d: wp.array(dtype=wp.vec3d)):
    """y_S = J^-1 r_S (J^-1 dense 3m x 3m, row-major in Ji); init: at the CG's start (r . y into TR_RZ, d = y), else after k_tr_update (into TR_RZN) unless the CG ended"""
    a = wp.tid()
    m = mm[0]
    if a >= m:
        return
    if init == 0 and (tr[TR_DONE0] > wp.float64(0.0) or tr[TR_DONE] > wp.float64(0.0)):
        return
    n3 = 3*m
    ya = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
    for b in range(m):
        rb = r[Sidx[b]]
        for i in range(3):
            s = wp.float64(0.0)
            for j in range(3):
                s = s+Ji[(3*a+i)*n3+3*b+j]*rb[j]
            ya[i] = ya[i]+s
    ia = Sidx[a]
    y[ia] = ya
    if init > 0:
        d[ia] = ya
        wp.atomic_add(tr, TR_RZ, wp.dot(r[ia], ya))
    else:
        wp.atomic_add(tr, TR_RZN, wp.dot(r[ia], ya))


class SplitCG:
    """the fallback's CG on the device: TrustCG's buffers (z, g0, cm, p, Hp, r, y, d, Hd, tr) and W's gradient pass, one graph of `per` iterations"""

    def __init__(self, CG, per=2):
        W = CG.W; n = W.n; dev = 'cuda'
        self.CG, self.W, self.per = CG, W, per
        z = lambda dt_, k=n: wp.zeros(k, dtype=dt_, device=dev)
        self.b = dict(d_r=z(wp.vec3d), Pinv_m=z(wp.mat33d), inS=z(int), Sidx=z(int, MCAP), mm=z(int, 1), Ji=z(wp.float64, 9*MCAP*MCAP),
                      cols=wp.zeros((MCAP, n), dtype=wp.mat33d, device=dev))
        self.t = {k_: wp.to_torch(a_, requires_grad=False) for k_, a_ in self.b.items()}
        A, B, h = CG.a, self.b, CG.h

        def iteration():
            wp.launch(k_sp_mask, n, inputs=[A['d'], B['inS']], outputs=[B['d_r']])
            wp.launch(k_tr_absmax, n, inputs=[B['d_r']], outputs=[A['tr']])
            wp.launch(k_tr_perturb, n, inputs=[A['z'], B['d_r'], A['tr'], h], outputs=[W.xw])
            W._run()
            wp.launch(k_sp_hd, n, inputs=[W.xw.grad, A['g0'], A['d'], B['d_r'], A['cm'], B['cols'], B['Sidx'], B['mm'], A['tr'], h], outputs=[A['Hd']])
            wp.launch(k_tr_alpha, 1, inputs=[A['tr']])
            wp.launch(k_tr_update, n, inputs=[A['d'], A['Hd'], B['Pinv_m'], A['tr']], outputs=[A['p'], A['Hp'], A['r'], A['y']])
            wp.launch(k_sp_joint, MCAP, inputs=[A['r'], B['Sidx'], B['mm'], B['Ji'], 0, A['tr']], outputs=[A['y'], A['d']])
            wp.launch(k_tr_beta, 1, inputs=[A['tr']])
            wp.launch(k_tr_dir, n, inputs=[A['y'], A['tr']], outputs=[A['d']])
        self.t['mm'].fill_(0)
        CG.t['tr'].zero_(); CG.t['tr'][TR_DONE] = 1.                              # (a finished CG: the warm-up moves nothing)
        iteration()
        wp.capture_begin()
        try:
            iteration()
            for _ in range(per-1):
                iteration()
        finally:
            self.graph = wp.capture_end()

    def solve(self, g_arr, S, cols, Ji, delta, stop, cap):
        """as TrustCG.solve, with the group S (list), its full columns {k: n x 3 x 3}, the joint block's inverse (3m x 3m), the region, the stop (absolute |r|), the cap"""
        CG, A, B, T = self.CG, self.CG.a, self.b, self.t; n = self.W.n; m = len(S)
        assert m <= MCAP
        tr = CG.t['tr']; tr.zero_(); tr[TR_DELTA] = float(delta); tr[TR_STOP] = float(stop); tr[TR_CAP] = float(cap)
        gi = torch.as_tensor(S, device='cuda')
        T['inS'].zero_(); T['inS'][gi] = 1; T['Sidx'].zero_(); T['Sidx'][:m] = gi.to(torch.int32); T['mm'].fill_(m)
        T['Ji'].zero_(); T['Ji'][:9*m*m] = Ji.reshape(-1)
        for a, k in enumerate(S):
            T['cols'][a].copy_(cols[k])
        T['Pinv_m'].copy_(CG.t['Pinv']); T['Pinv_m'][gi] = 0.
        wp.launch(k_tr_init, n, inputs=[g_arr, B['Pinv_m']], outputs=[A['r'], A['y'], A['d'], A['p'], A['Hp'], A['tr']])
        wp.launch(k_sp_joint, MCAP, inputs=[A['r'], B['Sidx'], B['mm'], B['Ji'], 1, A['tr']], outputs=[A['y'], A['d']])
        wp.launch(k_tr_start, 1, inputs=[A['tr']])
        it = 0
        while it < cap+self.per:
            wp.capture_launch(self.graph)
            it += self.per
            if tr[TR_DONE].item() > 0:
                break
        wp.launch(k_tr_pred, n, inputs=[g_arr, A['p'], A['Hp']], outputs=[A['tr']])
        v = tr.tolist()
        return dict(K=int(v[TR_K]), EDGE=v[TR_EDGE] > 0, NEG=v[TR_NEG] > 0, PMP=v[TR_PMP], DELTA=v[TR_DELTA], GP=v[TR_GP], PHP=v[TR_PHP], RR=v[TR_RR])


def solve(ST, g_arr, delta, cap, log, ncell=6, hc=1e-8, eta=.005, tag='', check_py=False):
    """the fallback: the group from the stalled CG's residual, its columns (torch, 6 evaluations per site), the joint block, then SplitCG; the checks
    (symmetry along d1 / M^-1 r1 and the independent residual) by python products, apart. check_py: solve_py on the same point too, compared"""
    W, CG, A = ST.W, ST.CG, ST.a
    wpt = lambda a_: wp.to_torch(a_, requires_grad=False)
    dev, n = W.dev, W.n
    I3 = torch.eye(3, dtype=f64, device=dev)
    sync = torch.cuda.synchronize
    nrm = lambda t_: float(t_.norm()); dot = lambda a_, b_: float((a_*b_).sum())
    T = {}
    if getattr(ST, '_splitcg', None) is None:
        sync(); t0 = time.perf_counter(); ST._splitcg = SplitCG(CG); sync(); log('  PATCHCG: the device CG captured (%.2f s, once)' % (time.perf_counter()-t0))
    SC = ST._splitcg
    z = CG.t['z'].clone(); gz = wpt(g_arr).clone(); g0 = CG.t['g0'].clone(); cm = CG.t['cm']
    xw, gw = wpt(W.xw), wpt(W.xw.grad); gn = nrm(gz)
    nev = [0]

    def gE(x):
        sync(); xw.copy_(x); W._run(); wp.synchronize(); nev[0] += 1
        return gw.clone()
    sync(); t0 = time.perf_counter()
    rk = CG.t['r'].norm(dim=1); S = torch.argsort(rk, descending=True)[:ncell].tolist(); gi = torch.as_tensor(S, device=dev)
    share = float((rk[gi]**2).sum()/(rk**2).sum()); sync(); T['select'] = time.perf_counter()-t0
    t0 = time.perf_counter(); cols = {}
    for k in S:
        Hk = torch.zeros(n, 3, 3, dtype=f64, device=dev)
        for a in range(3):
            e = torch.zeros(n, 3, dtype=f64, device=dev); e[k, a] = 1.
            Hk[:, :, a] = (gE(z+hc*e)-gE(z-hc*e))/(2*hc)
        Hk[k] += cm[k]*I3; cols[k] = Hk
    sync(); T['columns'] = time.perf_counter()-t0; ev_cols = nev[0]
    t0 = time.perf_counter(); m = len(S); J = torch.zeros(m, 3, m, 3, dtype=f64, device=dev)
    for b, kb in enumerate(S):
        for a, ka in enumerate(S):
            J[a, :, b, :] = cols[kb][ka]
    J = J.reshape(3*m, 3*m); asym = float((J-J.T).abs().max()/J.abs().max()); J = .5*(J+J.T)
    ev, U = torch.linalg.eigh(J); floor = float(cm[gi].min()); nfl = int((ev < floor).sum())
    J = U@torch.diag(torch.maximum(ev, torch.full_like(ev, floor)))@U.T; Ji = torch.linalg.inv(J); sync(); T['factor'] = time.perf_counter()-t0
    gE(z)                                                                           # (W's buffers at z: the graph's products start there)
    t0 = time.perf_counter(); r = SC.solve(g_arr, S, cols, Ji, delta, eta*gn, cap); sync(); T['iterate'] = time.perf_counter()-t0
    p = CG.t['p'].clone(); Hp = CG.t['Hp'].clone(); k = r['K']; ev_cg = k
    exit_ = 'negative' if r['NEG'] else ('edge' if r['EDGE'] else ('cap' if k >= cap else 'tolerance'))
    # checks apart: the split product's symmetry along d1 = y0, v = M^-1 r1 (python products), the independent residual
    t0 = time.perf_counter(); e0 = nev[0]
    Pinv_o = CG.t['Pinv'].clone(); Pinv_o[gi] = 0.

    def Minv(r_):
        y_ = torch.einsum('kab,kb->ka', Pinv_o, r_); y_[gi] = (Ji@r_[gi].reshape(-1)).reshape(m, 3); return y_
    Hcols = lambda v_: sum(torch.einsum('lab,b->la', cols[k_], v_[k_]) for k_ in S)

    def Hsplit(v, h=CG.h, central=False):
        vr = v.clone(); vr[gi] = 0.; s_ = float(vr.abs().max())
        if s_ == 0.:
            return Hcols(v)
        if central:
            return Hcols(v)+cm[:, None]*vr+(gE(z+(h/s_)*vr)-gE(z-(h/s_)*vr))*(s_/(2*h))
        return Hcols(v)+cm[:, None]*vr+(gE(z+(h/s_)*vr)-g0)*(s_/h)
    d1 = Minv(-gz); h1 = Hsplit(d1); r1 = -gz-(dot(-gz, d1)/max(dot(d1, h1), 1e-300))*h1; y2 = Minv(r1); h2 = Hsplit(y2)
    sym = abs(dot(d1, h2)-dot(y2, h1))/(abs(dot(d1, h2))+abs(dot(y2, h1))+1e-300)
    res_split = nrm(gz+Hsplit(p, h=hc, central=True))/gn
    vr = p.clone(); s_ = float(p.abs().max()); res_cent = nrm(gz+cm[:, None]*p+(gE(z+(hc/s_)*p)-gE(z-(hc/s_)*p))*(s_/(2*hc)))/gn
    sync(); T['checks'] = time.perf_counter()-t0; ev_chk = nev[0]-e0
    cmp_ = ''
    if check_py:
        rp = solve_py(ST, g_arr, delta, cap, log, ncell=ncell, hc=hc, eta=eta, tag=tag+' (python, the same point)', S=S)
        pp = CG.t['p'].clone(); cmp_ = '; against the python solve: K %d vs %d, |p_warp - p_py| / |p| %.2e, pred %.4g vs %.4g' % (k, rp['K'], nrm(p-pp)/max(nrm(pp), 1e-300), -(r['GP']+.5*r['PHP']), -(rp['GP']+.5*rp['PHP']))
    gE(z); wpt(CG.a['p']).copy_(p); wpt(CG.a['Hp']).copy_(Hp)
    log('  PATCHCG%s: group %s (%.0f %% of the stalled residual; cells V0/med %s); columns %d evaluations; joint block asymmetry %.1e, eigenvalues %.3g..%.3g (floored %d); '
        'CG exit %s after %d products, recursive |r| / |g| %.3g (eta %.3g), delta %.3g, |p|_M %.3g; checks: split symmetry u = d1, v = M^-1 r1 %.1e, independent residual columns + central rest %.3g, central on p %.3g (%d evaluations); '
        'time select %.3f, columns %.3f, factor %.3f, iterate %.3f (checks %.3f) s%s' % (
            tag, S, 100*share, ['%.2f' % float(W.V0[c_]/W.V0.median()) for c_ in S], ev_cols, asym, float(ev[0]), float(ev[-1]), nfl, exit_, k, math.sqrt(max(r['RR'], 0.))/gn, eta, r['DELTA'], math.sqrt(max(r['PMP'], 0.)),
            sym, res_split, res_cent, ev_chk, T['select'], T['columns'], T['factor'], T['iterate'], T['checks'], cmp_))
    return dict(r, evals=ev_cols+ev_cg, evals_checks=ev_chk, time=T['select']+T['columns']+T['factor']+T['iterate'], time_checks=T['checks'], res_split=res_split, res_cent=res_cent, group=S)
