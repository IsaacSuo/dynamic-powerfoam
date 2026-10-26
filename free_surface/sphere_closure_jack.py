"""How hard is one implicit-midpoint step of the skinless sphere model (sphere_closure.py) on the jack (8k foam,
output/round01_pruned_exact, copied from talk-pages; 2026-10-26)? Linear analysis at rest, matrix-free (a dense K would
be 22k x 22k):
  K v      exact at rest: forces and stresses vanish there, so K v = [I; G]^T H [I; G] v with H the Hessian of the
           energy in (x, F_B) (one forward-over-reverse product) and G = J^-1 dS_B / dx the closure's linearisation
  contact  the ground as in a landing: normal from the rest frame to the state at step 240 of the plasticine landing
           (landing_contact240.npz, Kabsch), the cells whose exposed sphere's lowest point (rest) lies within --band of the
           lowest one; a penalty rho (grad g)(grad g)^T on g = n . (x_B + F_B d_B), d_B the lowest point's lever, rho = kappa /
           (a^T K_bb^-1 a) (kappa 100 as the augmented Lagrangian, a the gap's direction on the body's own position); 'glue': the same
           on the two tangential directions (the friction glue fully stuck)
  P        the bodies' 3 x 3 diagonal blocks of A = 4 M / h^2 + K (+ contact), probed with a distance-4 colouring of the
           bond graph (a few far couplings leak into the blocks, as in a real probed preconditioner)
  PCG      from the inertial predictor, the right-hand side -K x~ of a smooth field (quadratic, random coefficients) and of a
           rough one (random per body): iterations to 1e-3 / 1e-6, and P^-1 A's extreme eigenvalues from the CG's Lanczos
           coefficients
Skin model on the jack for comparison, from the logs (PLAN): at rest 3.7 iterations per e-fold, the compression phase
13-15 per e-fold, 150-900 products per contact step.
  PYTHONPATH=<main repo>:<worktree> python free_surface/sphere_closure_jack.py"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from experiments.pacnerf_dem_forward import build
from free_surface.sphere_closure import ClosureTerms

f64, dev = torch.float64, 'cuda'


def pcg(Av, Pinv, b, tol=1e-6, maxit=3000):
    """PCG from 0 (b the initial residual); iterations to 1e-3 and to tol, and the Lanczos estimates of P^-1 A's spectrum"""
    z = torch.zeros_like(b); r = b.clone(); r0 = float(r.norm())
    d = Pinv(r); rz = float((r*d).sum()); p_ = d.clone()
    al_, be_ = [], []
    it3 = None
    for it in range(1, maxit+1):
        Ap = Av(p_); al = rz/float((p_*Ap).sum())
        z = z+al*p_; r = r-al*Ap
        al_.append(al)
        rel = float(r.norm())/r0
        if it3 is None and rel < 1e-3:
            it3 = it
        if rel < tol:
            break
        d = Pinv(r); rz_new = float((r*d).sum()); be = rz_new/rz; be_.append(be)
        p_ = d+be*p_; rz = rz_new
    k = len(al_)
    T = np.zeros((k, k))
    for i in range(k):
        T[i, i] = 1/al_[i]+(be_[i-1]/al_[i-1] if i > 0 else 0.)
        if i < k-1:
            T[i, i+1] = T[i+1, i] = math.sqrt(be_[i])/al_[i]
    ev = np.linalg.eigvalsh(T)
    return it3, it, float(ev[0]), float(ev[-1])


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument('--out', type=Path, default=Path('output/round01_pruned_exact'))
    a_.add_argument('--h', type=float, nargs='*', default=[1/1200])
    a_.add_argument('--band', type=float, nargs='*', default=[.002, .006, .015], help='contact: lowest points within this of the lowest (m)')
    a_.add_argument('--kappa', type=float, default=100.)
    a_.add_argument('--series', action=argparse.BooleanOptionalAction, default=True,
                    help='the contact cell\'s cap may leave its free-surface state under its own contact load (dq, energy '
                         '1/2 dq^T V_cap diag(lam + 2 mu, mu, mu) dq, moving the lowest point only): in series with rho, '
                         'k_n = (lam + 2 mu) A_cap / 3 r, k_t = 4 mu A_cap / 3 r')
    a_.add_argument('--c0', type=float, default=-.05)
    args = a_.parse_args()
    torch.manual_seed(0); rng = np.random.default_rng(0)
    cells = np.load(args.out/'dem_exact_single'/'cells.npz')
    g = dict(np.load(args.out/'dem_exact_single'/'geometry.npz'))
    p, r = cells['points'], cells['radii']
    m = build(g, 23008.38, .3008, dev, 'noslip', .25, .3, f64, kinematics='exact', nodalvol=True, nonlinear=False, ogden_alpha=2.)
    t0 = time.time()
    new = ClosureTerms(m, g, p, r, dict(np.load(args.out/'boundary_skin.npz')))
    n = new.n; n3 = 3*n
    th = torch.zeros(n, 9, dtype=f64, device=dev)
    X = new.X; I3 = torch.eye(3, dtype=f64, device=dev)
    mass = m.mass.to(f64)
    print('jack: %d bodies (%d with an exposed sphere), %d walls; built in %.0f s; body volume min / median %.1e' % (
        n, int(new.cap.sum()), len(new.A), time.time()-t0, float(new.V.min()/new.V.median())), flush=True)

    # the exact K at rest, matrix-free
    S0 = new.walls(X)
    J0 = new.jacobian(I3.expand(n, 3, 3).contiguous(), S0, new.kap0)
    JT = J0.transpose(1, 2).contiguous()
    print('closure Jacobians at rest: condition number median %.1f, max %.1f' % (float(torch.linalg.cond(J0).median()), float(torch.linalg.cond(J0).max())), flush=True)
    Gv = lambda v: torch.linalg.solve(J0, new.walls(v.view(n, 3)).reshape(n, 9, 1)).view(n, 3, 3)

    def GTw(w):
        t = torch.linalg.solve(JT, w.reshape(n, 9, 1)).view(n, 3, 3)
        xv = torch.zeros(n, 3, dtype=f64, device=dev, requires_grad=True)
        return torch.autograd.grad(new.walls(xv), xv, grad_outputs=t)[0]
    e = lambda z: sum(new.energy_F(th, X+z[0], I3+z[1]).values())
    ge = torch.func.grad(e)
    z0 = (torch.zeros(n, 3, dtype=f64, device=dev), torch.zeros(n, 3, 3, dtype=f64, device=dev))

    def Kv(v):
        v = v.view(n, 3)
        hx, hF = torch.func.jvp(ge, (z0,), ((v, Gv(v)),))[1]
        return hx+GTw(hF)
    t0 = time.time()
    u, w_ = torch.randn(n, 3, dtype=f64, device=dev), torch.randn(n, 3, dtype=f64, device=dev)
    Ku, Kw = Kv(u), Kv(w_)
    tK = time.time()-t0

    def grad_full(x):
        x = x.detach().clone().requires_grad_(True)
        return torch.autograd.grad(new.energy_x(th, x), x)[0]
    hh = 1e-7
    gd = (grad_full(X+hh*u)-grad_full(X-hh*u))/(2*hh)
    print('K v: %.2f s per product; symmetry |u.Kw - w.Ku| / |u.Kw| %.1e; against central differences of the gradient %.1e' % (
        tK/2, abs(float((u*Kw).sum()-(w_*Ku).sum()))/abs(float((u*Kw).sum())), float((gd-Ku).norm()/Ku.norm())), flush=True)

    # colouring: distance 4 in the bond graph
    i_, j_ = new.i.cpu().numpy(), new.j.cpu().numpy()
    Adj = sp.coo_matrix((np.ones(len(i_)), (i_, j_)), shape=(n, n)).tocsr(); Adj = ((Adj+Adj.T+sp.eye(n)) > 0).astype(np.float64)
    D = Adj.copy()
    for _ in range(3):
        D = ((D@Adj) > 0).astype(np.float64)
    D = D.tocsr()
    color = -np.ones(n, int)
    for b in np.argsort(-np.diff(D.indptr)):
        used = set(color[D.indices[D.indptr[b]:D.indptr[b+1]]].tolist())
        c = 0
        while c in used:
            c += 1
        color[b] = c
    nc = color.max()+1
    print('colouring (distance 4): %d colours, %d products for the blocks' % (nc, 3*nc), flush=True)

    def probe(Op):
        B = torch.zeros(n, 3, 3, dtype=f64, device=dev)
        for c in range(nc):
            msk = torch.as_tensor(color == c, device=dev)
            for k in range(3):
                v = torch.zeros(n, 3, dtype=f64, device=dev); v[msk, k] = 1.
                B[msk, :, k] = Op(v)[msk]
        return .5*(B+B.transpose(1, 2))
    t0 = time.time()
    Kbb = probe(Kv)
    print('blocks probed in %.0f s' % (time.time()-t0), flush=True)

    # contact geometry in the rest frame
    st240 = np.load(args.out/'landing_contact240.npz')
    Xc, Lc = X.cpu().numpy(), st240['x']
    U_, _, Vt = np.linalg.svd((Xc-Xc.mean(0)).T@(Lc-Lc.mean(0)))
    Q = (U_@Vt).T
    if np.linalg.det(Q) < 0:
        U_[:, -1] *= -1; Q = (U_@Vt).T
    up = np.array([0., 1., 0.])
    nr = Q.T@up                                                                     # the ground normal in the rest frame
    pt, rt = torch.as_tensor(p, dtype=f64, device=dev), torch.as_tensor(r, dtype=f64, device=dev)
    nT = torch.as_tensor(nr, dtype=f64, device=dev)
    boc = torch.as_tensor(g['body_of_cell'], device=dev)
    bot = pt-rt[:, None]*nT                                                         # each sphere's lowest point
    pw = torch.cdist(bot, pt)**2-rt**2; pw[torch.arange(n, device=dev), torch.arange(n, device=dev)] = float('inf')
    exposed = (pw.min(1).values >= 0) & new.cap[boc]
    Yr = new.Y; lowY = torch.full((n,), float('inf'), dtype=f64, device=dev)          # else the lowest of the cap's skin points
    hY = (Yr*nT).sum(1)
    lowY = lowY.scatter_reduce(0, new.e_body, hY[new.e_pid], 'amin')
    argY = torch.zeros(n, 3, dtype=f64, device=dev)
    sel = hY[new.e_pid] == lowY[new.e_body]
    argY[new.e_body[sel]] = Yr[new.e_pid[sel]]
    low = torch.where(exposed[:, None], bot, argY)
    hl = (low*nT).sum(1); hl = torch.where(new.cap, hl, torch.full_like(hl, float('inf')))
    print('ground normal in the rest frame %s (rotation from rest to step 240: %.1f deg); cells with an exposed lowest sphere point %d' % (
        np.round(nr, 3).tolist(), math.degrees(math.acos(max(-1., min(1., (np.trace(Q)-1)/2)))), int(exposed.sum())), flush=True)
    Md = mass[:, None].expand(n, 3)
    t1 = torch.linalg.cross(nT, torch.as_tensor([1., 0., 0.], dtype=f64, device=dev)); t1 = t1/t1.norm(); t2 = torch.linalg.cross(nT, t1)

    def contact_op(ids, dirs, rho, dB):
        """sum over the contact cells and directions of rho (grad g)(grad g)^T v"""
        def Op(v):
            v = v.view(n, 3); Fv = Gv(v)
            out = torch.zeros(n, 3, dtype=f64, device=dev); W = torch.zeros(n, 3, 3, dtype=f64, device=dev)
            for k, dvec in enumerate(dirs):
                a = (v[ids]*dvec).sum(1)+torch.einsum('i,cij,cj->c', dvec, Fv[ids], dB)
                out.index_add_(0, ids, (rho[:, k]*a)[:, None]*dvec)
                W.index_add_(0, ids, (rho[:, k]*a)[:, None, None]*dvec[:, None]*dB[:, None, :])
            return out+GTw(W)
        return Op
    smooth = (X-X.mean(0))
    Cq = torch.as_tensor(rng.normal(size=(3, 3, 3)), dtype=f64, device=dev)
    xs = torch.einsum('aij,bi,bj->ba', Cq, smooth, smooth)                          # a quadratic field
    xr = torch.randn(n, 3, dtype=f64, device=dev)
    cases = [('no contact', None)]
    lam_, mu_ = float(m.k_v), new.mu
    Acap = torch.zeros(n, dtype=f64, device=dev).index_add(0, new.c_body, new.c_A)
    kn, kt = (lam_+2*mu_)*Acap/(3*rt), 4*mu_*Acap/(3*rt)
    for band in args.band:
        ids = torch.nonzero(hl <= hl.min()+band).flatten()
        cases += [('contact %d cells (band %.0f mm), normal' % (len(ids), 1e3*band), (ids, 1)),
                  ('contact %d cells (band %.0f mm), normal + glue' % (len(ids), 1e3*band), (ids, 3))]
    for h in args.h:
        Pb0 = Kbb+(4*mass/h**2)[:, None, None]*I3
        for name, cs in cases:
            if cs is None:
                Av = lambda v: Kv(v)+4*Md/h**2*v.view(n, 3); Pb = Pb0
            else:
                ids, nd = cs
                dB = low[ids]-X[ids]
                dirs = [nT, t1, t2][:nd]
                # rho = kappa / (a^T K_bb^-1 a), a the gap's gradient on the body's own position (the direction)
                Ki = torch.linalg.inv(Kbb[ids])
                rho = torch.stack([args.kappa/torch.einsum('i,cij,j->c', dvec, Ki, dvec) for dvec in dirs], 1)
                if args.series:                      # the cap's own compliance in series: rho k / (rho + k)
                    k = torch.stack([kn, kt, kt][:nd], 1)[ids]
                    print('   %s: cap stiffness / rho median %.3f (normal), the series stiffness / rho median %.3f' % (
                        name, float((k[:, 0]/rho[:, 0]).median()), float((k/(rho+k)).median())), flush=True)
                    rho = rho*k/(rho+k)
                Cop = contact_op(ids, dirs, rho, dB)
                Av = lambda v, Cop=Cop: Kv(v)+Cop(v)+4*Md/h**2*v.view(n, 3)
                Pb = Pb0+probe(Cop)
            Pi = torch.linalg.inv(Pb)
            Pinv = lambda R, Pi=Pi: torch.einsum('bij,bj->bi', Pi, R)
            res = []
            for rhs in (xs, xr):
                b = -(Av(rhs)-4*Md/h**2*rhs)                                          # -(K + contact) x~
                res.append(pcg(Av, Pinv, b))
            print('h 1/%.0f, %-40s smooth: %s / %s iterations to 1e-3 / 1e-6 (%.1f per e-fold), P^-1 A in [%.3f, %.2f]; rough: %s / %s (%.1f per e-fold), [%.3f, %.2f]' % (
                1/h, name, res[0][0], res[0][1], res[0][1]/math.log(1e6), res[0][2], res[0][3],
                res[1][0], res[1][1], res[1][1]/math.log(1e6), res[1][2], res[1][3]), flush=True)


if __name__ == '__main__':
    main()
