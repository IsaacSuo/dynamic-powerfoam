"""A coarse level for the skin's preconditioner, a prototype that only counts products (2026-10-24). The skin alone at a
saved state is slow (split_pass_check.py: 12.3 products per e-fold, with its group blocks exact still ~14), so the slow
error should be one spread over many groups, which the group blocks (additive Schwarz, one group a body's skin points)
see only as each group moving whole. Two-level additive Schwarz: P2 r = P r + Phi A_c^-1 Phi^T r, A_c = Phi^T K Phi with
K = the skin's Hessian probed at the start (SkinSolver.probe_K: elastic, viscous, the held contacts' penalty), Phi the
groups' moves with a partition of unity (a point shared by c groups weighs 1/c): C1 each group's three translations
(dense Cholesky); A4 / A8 the translations of aggregates of up to 4 / 8 neighbouring groups (A_c from C1's); C1R C1
and each group's three rotations about its centroid (sparse LU on the CPU). At the skin-alone start of the step's
first pass (the bodies held at the joint solve's x*) the solve each way (torch products throughout): products, products
per e-fold, |y - y*| / |y* - y_s|; and one coarse correction's time. (10-24, first run: C2, each group's translation
along its outward normal only, did nothing: 113 products against 114.)
  python free_surface/coarse_proto_check.py --out output/round01_pruned_exact --dt 2.0833333e-4 --tols 1e-4 --beta 0.01 --integrator midpoint --state output/round01_pruned_exact/skin_drop_state_step900.npz"""
import math
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from free_surface.joint_step import MidpointIntegrator
from free_surface.skin_contact_check import f64, dev
from free_surface.skin_dynamics_check import build_model, parser


def main():
    a = parser()
    a.add_argument('--state', type=Path, required=True)
    a.add_argument('--gravity', type=float, default=9.8)
    args = a.parse_args()
    b = build_model(args, contact=dict(normal=(0., 1., 0.), c0=0., shift=True))
    st, W, S, prec, mass = b['st'], b['W'], b['S'], b['prec'], b['mass']
    z0 = np.load(args.state)
    T = lambda k: torch.as_tensor(z0[k], device=dev)
    fg = -args.gravity*mass[:, None]*torch.tensor([0., 1., 0.], dtype=f64, device=dev)
    ref = float(fg.norm()); tol, dt = args.tols[0], args.dt
    I = MidpointIntegrator(b, dt, T('x'), T('v'), T('ym1'), fg, tol=tol, ref=ref, contact=True)
    I.x, I.v, I.a, I.xm1, I.ym1, I.xm2, I.ym2 = (T(k) for k in ('x', 'v', 'a', 'xm1', 'ym1', 'xm2', 'ym2'))
    def restore():
        W.tlam.copy_(T('lam')); W.tact.copy_(T('act')); W.tgoff.copy_(T('goff')); W.trho.copy_(T('rho'))
        for k_, r_ in W.tref.items():
            r_.copy_(T('ref_'+k_))
        W.total_mode()
    restore()
    J, n, npt = I.J, W.n, W.npt
    x, v = I.x, I.v
    xs = x+.5*dt*v+.25*dt*dt*I.a
    ys = I.predict_skin(S, xs, I.xm1, I.xm2, I.ym1, I.ym2)
    J.set_step(x, x+.5*dt*v, fg)
    z, s_ = J.solve(None, torch.cat((xs, ys)), prec, ref, tol=tol, maxit=2000)
    xa, ya = z[:n].clone(), z[n:].clone()
    print('joint first pass: %d products; the skin alone from its start with the bodies at x*' % s_['products'], flush=True)

    # the skin's Hessian at the skin-alone start, and the coarse spaces
    restore()
    t0 = time.perf_counter()
    K = S.probe_K(xa, ys)
    print('K probed at the start (contact term on, the held set): %d nonzeros, %.1f s' % (K.nnz, time.perf_counter()-t0), flush=True)
    g = S.bodies
    lists, owner = g['lists'], g['owner'].cpu().numpy()
    G = len(lists)
    rows = np.concatenate([np.asarray(l) for l in lists]); grp = np.concatenate([np.full(len(l), k) for k, l in enumerate(lists)])
    cnt = np.bincount(rows, minlength=npt).astype(float)
    print('groups %d, skin points in a group %d to %d; points shared by groups: %d of %d (up to %d groups)' % (
        G, min(len(l) for l in lists), max(len(l) for l in lists), int((cnt > 1).sum()), npt, int(cnt.max())), flush=True)
    w = 1/cnt[rows]
    Y_, X_ = ys.cpu().numpy(), xa.cpu().numpy()
    nrm = np.zeros((G, 3)); np.add.at(nrm, grp, Y_[rows]-X_[owner[grp]])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    csr = lambda M_: torch.sparse_csr_tensor(torch.as_tensor(M_.indptr, device=dev), torch.as_tensor(M_.indices, device=dev),
                                             torch.as_tensor(M_.data, dtype=f64, device=dev), size=M_.shape)
    P1 = sp.coo_matrix((np.repeat(w, 3), ((3*rows[:, None]+np.arange(3)).ravel(), (3*grp[:, None]+np.arange(3)).ravel())), shape=(3*npt, 3*G)).tocsr()
    cen = np.zeros((G, 3)); np.add.at(cen, grp, Y_[rows]); cen /= np.bincount(grp, minlength=G)[:, None]
    dpos = Y_[rows]-cen[grp]                                                       # rotations about each group's centroid
    E3 = np.eye(3)
    PR = sp.coo_matrix((np.concatenate([(np.cross(E3[k][None, :], dpos)*w[:, None]).ravel() for k in range(3)]),
                        (np.concatenate([(3*rows[:, None]+np.arange(3)).ravel()]*3), np.concatenate([np.repeat(3*grp+k, 3) for k in range(3)]))),
                       shape=(3*npt, 3*G)).tocsr()
    Bg = sp.coo_matrix((np.ones(len(rows)), (grp, rows)), shape=(G, npt)).tocsr()
    Ag = ((Bg@Bg.T) > 0).tocsr()                                                   # groups sharing skin points
    def aggregate(target):
        """greedy: each unassigned group in turn seeds an aggregate grown breadth-first over unassigned neighbours"""
        lab = -np.ones(G, int); na = 0
        for s0 in range(G):
            if lab[s0] >= 0:
                continue
            lab[s0] = na; q = [s0]; qi = 0; size = 1
            while qi < len(q) and size < target:
                u = q[qi]; qi += 1
                for v_ in Ag.indices[Ag.indptr[u]:Ag.indptr[u+1]]:
                    if lab[v_] < 0 and size < target:
                        lab[v_] = na; q.append(v_); size += 1
            na += 1
        return lab, na
    H = S.hvp(xa, ys)
    def check(P_, Amul):
        Pt_, PtT_ = csr(P_), csr(P_.T.tocsr())
        cv = torch.randn(P_.shape[1], dtype=f64, device=dev)                       # A_c against products along Phi c
        Hv = H((Pt_@cv)[None])[0]
        Ac_cv = Amul(cv)
        return float(((PtT_@Hv)-Ac_cv).norm()/Ac_cv.norm())
    coarse = {}
    def dense(name, P_, Ac, t1, chk=None):
        A_ = torch.as_tensor(.5*(Ac+Ac.T), dtype=f64, device=dev)
        L, info = torch.linalg.cholesky_ex(A_)
        coarse[name] = (csr(P_), csr(P_.T.tocsr()), lambda rc, L=L: torch.cholesky_solve(rc, L))
        print('%s: %d coarse unknowns, dense Cholesky %s%s; %.1f s' % (name, P_.shape[1], 'ok' if int(info) == 0 else 'failed (%d)' % int(info),
              '' if chk is None else ', A_c c against Phi^T H Phi c: relative %.1e' % chk, time.perf_counter()-t1), flush=True)
    t1 = time.perf_counter()
    A1 = (P1.T@(K@P1)).toarray()
    dense('C1', P1, A1, t1, check(P1, lambda c: torch.as_tensor(A1, device=dev)@c))
    for target in (4, 8):
        t1 = time.perf_counter()
        lab, na = aggregate(target)
        sz = np.bincount(lab)
        Sg = sp.coo_matrix((np.ones(3*G), ((3*np.arange(G)[:, None]+np.arange(3)).ravel(), (3*lab[:, None]+np.arange(3)).ravel())), shape=(3*G, 3*na)).tocsr()
        print('aggregates of up to %d groups: %d (sizes %d to %d, mean %.1f)' % (target, na, sz.min(), sz.max(), sz.mean()), flush=True)
        dense('A%d' % target, (P1@Sg).tocsr(), np.asarray(Sg.T@(Sg.T@A1).T), t1)
    t1 = time.perf_counter()
    PB = sp.hstack((P1, PR)).tocsr()
    AB = (PB.T@(K@PB)).tocsc(); AB = (.5*(AB+AB.T)).tocsc()
    import scipy.sparse.linalg as spl
    lu = spl.splu(AB)
    def lu_solve(rc):
        return torch.as_tensor(lu.solve(rc.cpu().numpy()), dtype=f64, device=dev)
    chk = check(PB, lambda c: torch.as_tensor(AB@c.cpu().numpy(), device=dev))
    coarse['C1R'] = (csr(PB), csr(PB.T.tocsr()), lu_solve)
    print('C1R: %d coarse unknowns (translations and rotations about each group\'s centroid), sparse LU on the CPU (%d nonzeros, factor %d), '
          'A_c c against Phi^T H Phi c: relative %.1e; %.1f s' % (PB.shape[1], AB.nnz, lu.L.nnz+lu.U.nnz, chk, time.perf_counter()-t1), flush=True)
    for name, (Pt, PtT, sol) in coarse.items():                                     # one coarse correction's time
        r = torch.randn(1, 3*npt, dtype=f64, device=dev)
        for _ in range(3):
            (Pt@sol(PtT@r.T)).T
        torch.cuda.synchronize(); t1 = time.perf_counter()
        for _ in range(20):
            (Pt@sol(PtT@r.T)).T
        torch.cuda.synchronize()
        print('   %s: one coarse correction %.3f ms' % (name, 1e3*(time.perf_counter()-t1)/20), flush=True)

    # the skin alone, the preconditioners (torch products throughout: the same counts as the graph's)
    def run(name):
        restore()
        cc = coarse.get(name)
        def skin_prec_at(mp, x_, y_, aux):
            f = mp(aux[1])
            f.wprec.contact_update(S.contact_weights(aux[4]), W.trho*float(W.tcpar[1]*W.tcpar[0]**2)*(W.tact > 0))
            if cc is None:
                return lambda r: f(r)
            Pt, PtT, sol = cc
            def f2(r):
                rc = (PtT@r.T)                                                     # (coarse, k)
                return f(r)+(Pt@sol(rc)).T
            return f2
        S.prec_at = skin_prec_at
        r0 = float(S.eg(xa, ys)[1].norm())/ref
        torch.cuda.synchronize(); t1 = time.perf_counter()
        y1, s2 = S.solve(xa, ys.clone(), prec, ref, tol=tol, maxit=2000)
        torch.cuda.synchronize()
        ef = math.log(r0/max(s2['residual'], 1e-300))
        print('%-10s: start residual %.3e, %4d products, %2d outer, %d rejected, %d negative; residual %.2e; %.1f products per e-fold; |y - y*| / |y* - y_s| %.1e; %.0f ms' % (
            name, r0, s2['products'], s2['outer'], s2['rejected'], s2['negative'], s2['residual'], s2['products']/ef,
            float((y1-ya).norm()/(ya-ys).norm()), 1e3*(time.perf_counter()-t1)), flush=True)
    for name in ('groups', 'C1', 'A4', 'A8', 'C1R'):
        run(name)


if __name__ == '__main__':
    main()
