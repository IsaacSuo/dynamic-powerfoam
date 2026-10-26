"""Is the skin's equilibrium well posed under deformation? (2026-10-23, new foam, single bodies, skin C: shepard0, lever c;
skin_step_cost.py failed to converge from a cold start at a 3 cm, 1 rad state.) Part 1: deformation only, from rest.
  path       the bodies at X + s u, u the 6-mode combination of skin_step_cost.py at t = 0 (the failing state without its
             rotation), s = 0 .. 1 in --n increments; at each the skin from the previous ones (linear extrapolation), Newton +
             PCG (region blocks of K_rest turned by the bodies' rotations) to |dE/dy| <= 1e-8 |dE/dy at the cells' affine skin|
  curvature  in every CG iteration p^T K p of the search direction (relative to |p|^2 and the rest's smallest eigenvalue);
             a value <= 0 is a direction along which the energy curves down (the skin's energy not convex there)
  spectrum   at s = 0, 1/4, 1/2, 3/4, 1 the smallest eigenvalues of K_yy (LOBPCG, 4 vectors, preconditioned), relative to
             rest; if one is negative, the energy terms' share of its curvature (second directional derivative per term)
Part 2 (--part 2): rotation only. The rest state turned rigidly by 1 rad (the failing state's rotation): |dE/dy| there (0 for an
objective energy) against the load of part 1's end, and the smallest eigenvalues against rest; then one skin perturbation solved
at rest and, turned, at the rotated state: the same products and residuals if the solver (preconditioner turned by R_B) is
objective too."""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from experiments.dem_fem_mode_match import interpolate
from experiments.foam_surface_completion import polar_newton
from experiments.pacnerf_dem_forward import build as dem_build
from free_surface.boundary_skin import BoundaryTerms

f64, dev = torch.float64, 'cuda'


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--out', type=Path, default=Path('output/round01_exact'))
    a.add_argument('--skin', default='boundary_skin.npz')
    a.add_argument('--amp', type=float, default=.03)
    a.add_argument('--n', type=int, default=30)
    a.add_argument('--part', type=int, choices=(1, 2), default=1)
    args = a.parse_args()
    t0 = time.time()
    E, nu = 23008.38, .3008
    cells = np.load(args.out/'dem_exact_single'/'cells.npz'); p, r = cells['points'], cells['radii']
    g = dict(np.load(args.out/'dem_exact_single'/'geometry.npz')); bsk = dict(np.load(args.out/args.skin))
    m = dem_build(g, E, nu, dev, 'noslip', .25, .3, f64, kinematics='exact', nodalvol=True, nonlinear=False, ogden_alpha=2.)
    st = BoundaryTerms(m, g, p, r, bsk, lever='c')
    n, npt = st.n, st.npt; v3 = 3*npt
    th = torch.zeros(n, 9, dtype=f64, device=dev)
    X, Y = st.X, st.Y
    gy = lambda x, y: torch.func.grad(lambda yy: st.energy_xy(th, x, yy.view(-1, 3)))(y.reshape(-1))
    def hvp(x, y):
        f = lambda yy: torch.func.grad(lambda z: st.energy_xy(th, x, z.view(-1, 3)))(yy)
        y0 = y.reshape(-1)
        return torch.func.vmap(lambda v: torch.func.jvp(f, (y0,), (v,))[1])
    # region-block preconditioner from the saved K_rest (skin_step_cost.py)
    z = np.load(args.out/'skin_Kyy_rest.npz'); P_, Q_, vals = z['P'], z['Q'], z['vals']
    rows = (3*P_[:, None, None]+np.arange(3)[None, :, None]).repeat(3, 2)
    cols = (3*Q_[:, None, None]+np.arange(3)[None, None, :]).repeat(3, 1)
    K = sp.coo_matrix((vals.ravel(), (rows.ravel(), cols.ravel())), shape=(v3, v3)).tocsr(); K = (.5*(K+K.T)).tocsr()
    boc = np.asarray(g['body_of_cell'])
    off_ = bsk['part_off']; P = len(off_)-1
    loc = [bsk['ent_pid'][off_[k]:off_[k+1]] for k in range(P)]
    Smax = max(len(l) for l in loc)
    L = -np.ones((P, Smax), int); blocks = np.zeros((P, 3*Smax, 3*Smax))
    for k, l in enumerate(loc):
        L[k, :len(l)] = l
        i3 = (3*l[:, None]+np.arange(3)).ravel()
        blocks[k, :len(i3), :len(i3)] = K[i3][:, i3].toarray(); blocks[k, len(i3):, len(i3):] = np.eye(3*Smax-len(i3))
    Bt = torch.as_tensor(blocks, dtype=f64, device=dev); Binv = torch.linalg.inv(.5*(Bt+Bt.transpose(1, 2)))
    Lt = torch.as_tensor(L, device=dev); Lc = Lt.clamp(min=0); Lm = (Lt >= 0).to(f64)
    reg_body = torch.as_tensor(boc[bsk['part_site']], device=dev)
    def prec_region(Rb):
        Rr = Rb[reg_body]
        def f(rv):
            B = rv.shape[0]
            rl = rv.view(B, npt, 3)[:, Lc]*Lm[None, :, :, None]
            rl = torch.einsum('pji,bpsj->bpsi', Rr, rl).reshape(B, P, -1)
            zl = torch.einsum('pij,bpj->bpi', Binv, rl).view(B, P, Smax, 3)
            zl = torch.einsum('pij,bpsj->bpsi', Rr, zl)*Lm[None, :, :, None]
            return torch.zeros(B, npt, 3, dtype=f64, device=dev).index_add(1, Lc.reshape(-1), zl.reshape(B, -1, 3)).reshape(B, -1)
        return f
    # the failing state's deformation (skin_step_cost.py, t = 0, without the rotation)
    cache = np.load(args.out/'fem_modes_h0.008.npz')
    fem = dict(nodes=cache['nodes'], node_grid=cache['node_grid'], lo=cache['lo'], h=float(cache['h']), masses=cache['masses'])
    nm = 6
    Ux, _ = interpolate(fem, cache['V'][:, :nm], X.cpu().numpy())
    U = torch.as_tensor(Ux.transpose(2, 0, 1), dtype=f64, device=dev)
    rng = np.random.default_rng(1)
    amp = torch.as_tensor(rng.standard_normal(nm), dtype=f64, device=dev); phase = torch.as_tensor(rng.uniform(0, 2*np.pi, nm), dtype=f64, device=dev)
    om = torch.as_tensor(2*np.pi*cache['f'][:nm], dtype=f64, device=dev)
    disp = lambda t: torch.einsum('k,kni->ni', amp*torch.sin(om*t+phase), U)
    amp = amp*args.amp/float(max(disp(t_).norm(dim=1).max() for t_ in np.linspace(0, .5, 51)))
    u = disp(0.)
    # largest strain of the path's end: the bodies' F (bonds and skin at the affine positions)
    _, first = np.unique(bsk['ent_pid'], return_index=True)
    cell_of_pt = torch.as_tensor(bsk['ent_site'][first], device=dev)
    def affine_skin(x, y):
        F = st.F(x, y)
        return x[cell_of_pt]+torch.einsum('qij,qj->qi', F[cell_of_pt], Y-X[cell_of_pt])
    print('bodies %d, skin points %d, %d regions (up to %d points); path end: largest body displacement %.4f m (setup %.0f s)' % (
        n, npt, P, Smax, float(u.norm(dim=1).max()), time.time()-t0), flush=True)
    def lobpcg(x, y, k=4, maxiter=150):
        H = hvp(x, y); prec = prec_region(polar_newton(st.F(x, y)))
        Aop = spla.LinearOperator((v3, v3), matvec=lambda v: H(torch.as_tensor(v, dtype=f64, device=dev).reshape(1, -1))[0].cpu().numpy(),
                                  matmat=lambda V: H(torch.as_tensor(V.T.copy(), dtype=f64, device=dev)).cpu().numpy().T, dtype=np.float64)
        Mop = spla.LinearOperator((v3, v3), matvec=lambda v: prec(torch.as_tensor(v, dtype=f64, device=dev).reshape(1, -1))[0].cpu().numpy(),
                                  matmat=lambda V: prec(torch.as_tensor(V.T.copy(), dtype=f64, device=dev)).cpu().numpy().T, dtype=np.float64)
        X0 = np.random.default_rng(5).standard_normal((v3, k))
        w, V = spla.lobpcg(Aop, X0, M=Mop, largest=False, tol=1e-6, maxiter=maxiter)
        return w, V
    t1 = time.time()
    w0, _ = lobpcg(X, Y)
    lam0 = float(w0.min())
    print('rest: smallest eigenvalues of K_yy %s (%.0f s)' % (' '.join('%.3e' % w_ for w_ in w0), time.time()-t1), flush=True)
    def solve(x, y, ref, tol=1e-8, maxit=1500):
        """Newton + PCG; returns y, products, residual, and the smallest p^T K p / |p|^2 met (relative to rest)"""
        nh = 0; curv = float('inf')
        gr = gy(x, y); res = float(gr.norm())/ref
        while res > tol and nh < maxit:
            H = hvp(x, y); prec = prec_region(polar_newton(st.F(x, y)))
            b = -gr[None]; zv = torch.zeros_like(b); rr = b.clone(); zz = prec(rr); pv = zz.clone(); rz = (rr*zz).sum()
            neg = False
            while nh < maxit:
                Ap = H(pv); nh += 1
                pAp = (pv*Ap).sum()
                c_ = float(pAp/(pv*pv).sum())/lam0; curv = min(curv, c_)
                if c_ <= 0:
                    neg = True
                    break
                al = rz/pAp; zv = zv+al*pv; rr = rr-al*Ap
                if float(rr.norm())/ref <= max(tol, .01*res):
                    break
                zz = prec(rr); rz1 = (rr*zz).sum(); pv = zz+(rz1/rz)*pv; rz = rz1
            y = y+zv.view(-1, 3)
            gr = gy(x, y); res = float(gr.norm())/ref
            if neg:
                break
        return y, nh, res, curv
    if args.part == 2:
        k_ = torch.tensor([.3, 1., -.2], dtype=f64, device=dev); k_ = k_/k_.norm()
        Kx = torch.tensor([[0, -k_[2], k_[1]], [k_[2], 0, -k_[0]], [-k_[1], k_[0], 0]], dtype=f64, device=dev)
        Q = torch.eye(3, dtype=f64, device=dev)+math.sin(1.)*Kx+(1-math.cos(1.))*Kx@Kx
        cen = X.mean(0)
        turn = lambda Z: (Z-cen)@Q.T+cen
        xr, yr = turn(X), turn(Y)
        load = float(gy(X+u, affine_skin(X+u, Y)).norm())
        print('rotation only, 1 rad: |dE/dy| at rest %.2e, turned %.2e (load of part 1\'s end %.2e)' % (float(gy(X, Y).norm()), float(gy(xr, yr).norm()), load), flush=True)
        t1 = time.time()
        wr, _ = lobpcg(xr, yr)
        print('turned: smallest eigenvalues of K_yy %s; / rest %s (%.0f s)' % (' '.join('%.3e' % w_ for w_ in wr), np.round(np.sort(wr)/np.sort(w0), 6).tolist(), time.time()-t1), flush=True)
        dy = torch.randn(Y.shape, generator=torch.Generator(device=dev).manual_seed(7), dtype=f64, device=dev)*1e-3*float(u.norm(dim=1).max())
        for nm_, xx, y0_ in (('rest', X, Y+dy), ('turned', xr, yr+dy@Q.T)):
            t1 = time.time()
            ref = float(gy(xx, y0_).norm())
            y_, nh, res, curv = solve(xx, y0_, ref)
            print('perturbed skin (0.1%% of part 1\'s largest displacement), %s: |dE/dy| %.3e at the start, %d products to %.1e, smallest p^T K p / |p|^2 met %.4e of rest (%.0f s)' % (
                nm_, ref, nh, res, curv, time.time()-t1), flush=True)
        print('total %.0f s' % (time.time()-t0), flush=True)
        return
    ys = [Y]; table = []
    for k in range(1, args.n+1):
        s = k/args.n
        x = X+s*u
        yi = 2*ys[-1]-ys[-2] if len(ys) > 1 else ys[-1]
        ref = float(gy(x, affine_skin(x, yi)).norm())
        t1 = time.time()
        y, nh, res, curv = solve(x, yi, ref)
        ys.append(y)
        table.append((s, nh, res, curv))
        print('s %.3f (largest displacement %.2f mm): %d products, residual %.1e, smallest p^T K p / |p|^2 met %.2e of the rest\'s smallest eigenvalue%s (%.0f s)' % (
            s, 1e3*s*float(u.norm(dim=1).max()), nh, res, curv, '  <- curves down' if curv <= 0 else '', time.time()-t1), flush=True)
        if res > 1e-4:
            print('   not converged; stopping the path here', flush=True)
            break
    for q in (.25, .5, .75, 1.):
        k = int(round(q*args.n))
        if k >= len(ys):
            break
        t1 = time.time()
        w, V = lobpcg(X+k/args.n*u, ys[k])
        line = 's %.2f: smallest eigenvalues of K_yy / rest\'s smallest %s' % (k/args.n, np.round(w/lam0, 4).tolist())
        if w.min() <= 0:
            v = torch.as_tensor(V[:, int(np.argmin(w))], dtype=f64, device=dev)
            x = X+k/args.n*u; y0 = ys[k].reshape(-1)
            parts = {}
            for name in st.energy_terms(th, x, ys[k]):
                fe = lambda yy: st.energy_terms(th, x, yy.view(-1, 3))[name]
                d1 = lambda yy: torch.func.jvp(fe, (yy,), (v,))[1]
                parts[name] = float(torch.func.jvp(d1, (y0,), (v,))[1])
            line += '; curvature of the lowest by term %s' % ' '.join('%s %.2e' % kv for kv in parts.items())
        print(line+' (%.0f s)' % (time.time()-t1), flush=True)
    print('total %.0f s' % (time.time()-t0), flush=True)


if __name__ == '__main__':
    main()
