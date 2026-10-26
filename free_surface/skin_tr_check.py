"""Trust-region skin solve (skin_solve.py) on the new foam, small runs (2026-10-23):
  cold start   at the state where plain Newton-CG failed (skin_step_cost.py t = 0: the 6-mode deformation, largest 18.3 mm,
               turned by 1 rad), from the cells' affine skin; preconditioners (skin_solve.py): K_rest blocks per region and
               per body ('probed'), the per-body quadratic model and the skin's own squares per region (both assembled first, then the
               groups' blocks taken: built from the structure alone); products to
               1e-2 / 1e-4 / 1e-6, negative-curvature exits, rejected steps
  warm steps   from there 5 time steps of 1 / 4800 s of the same motion (deformation + 3 rad/s spin), each from the linear
               extrapolation of the last two skins, products to 1e-4 / 1e-6 for both preconditioners
  blocks       how far the skin's own squares (per region) and the model (per body) are from the probed blocks (relative
               Frobenius difference per group)"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from experiments.dem_fem_mode_match import interpolate
from experiments.pacnerf_dem_forward import build as dem_build
from free_surface.boundary_skin import BoundaryTerms
from free_surface.skin_solve import SkinSolver

f64, dev = torch.float64, 'cuda'


def rot(axis, ang):
    k = axis/axis.norm()
    K = torch.tensor([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], dtype=f64, device=dev)
    return torch.eye(3, dtype=f64, device=dev)+math.sin(ang)*K+(1-math.cos(ang))*K@K


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--out', type=Path, default=Path('output/round01_exact'))
    a.add_argument('--skin', default='boundary_skin.npz')
    a.add_argument('--steps', type=int, default=5)
    a.add_argument('--cap', type=int, default=600)
    args = a.parse_args()
    t0 = time.time()
    cells = np.load(args.out/'dem_exact_single'/'cells.npz'); p, r = cells['points'], cells['radii']
    g = dict(np.load(args.out/'dem_exact_single'/'geometry.npz')); bsk = dict(np.load(args.out/args.skin))
    m = dem_build(g, 23008.38, .3008, dev, 'noslip', .25, .3, f64, kinematics='exact', nodalvol=True, nonlinear=False, ogden_alpha=2.)
    st = BoundaryTerms(m, g, p, r, bsk, lever='c')
    S = SkinSolver(st, bsk, g['body_of_cell'])
    X, Y, npt = st.X, st.Y, st.npt; v3 = 3*npt
    z = np.load(args.out/'skin_Kyy_rest.npz'); P_, Q_, vals = z['P'], z['Q'], z['vals']
    rows = (3*P_[:, None, None]+np.arange(3)[None, :, None]).repeat(3, 2)
    cols = (3*Q_[:, None, None]+np.arange(3)[None, None, :]).repeat(3, 1)
    K = sp.coo_matrix((vals.ravel(), (rows.ravel(), cols.ravel())), shape=(v3, v3)).tocsr(); K = (.5*(K+K.T)).tocsr()
    def rel(B1, B0, g):
        real = g['Lm'].repeat_interleave(3, 1); msk = real[:, :, None]*real[:, None, :]
        d = ((B1-B0)*msk).flatten(1).norm(dim=1)/(B0*msk).flatten(1).norm(dim=1)
        return 'median %.2f, 90%% %.2f, max %.2f' % (float(d.median()), float(d.quantile(.9)), float(d.max()))
    Bpr, Bpb = S.blocks_probed(K, S.regions), S.blocks_probed(K, S.bodies)
    t1 = time.time()
    Bm = S.blocks_probed(S.assemble(S.blocks_model(), S.bodies), S.bodies)
    tm = time.time()-t1
    Bs = S.blocks_probed(S.assemble(S.blocks_skin(), S.regions), S.regions)
    print('regions %d (up to %d points), bodies with skin %d (up to %d points); |block - probed| / |probed|: skin squares assembled, per region %s; '
          'model assembled, per body %s (built in %.1f s; setup %.0f s)' % (S.regions['G'], S.regions['S'], S.bodies['G'], S.bodies['S'], rel(Bs, Bpr, S.regions),
                                                                            rel(Bm, Bpb, S.bodies), tm, time.time()-t0), flush=True)
    precs = {'probed, bodies': S.preconditioner(Bpb, S.bodies), 'model, bodies': S.preconditioner(Bm, S.bodies), 'skin squares, regions': S.preconditioner(Bs, S.regions)}
    # the failing state and its motion (skin_step_cost.py)
    cache = np.load(args.out/'fem_modes_h0.008.npz')
    fem = dict(nodes=cache['nodes'], node_grid=cache['node_grid'], lo=cache['lo'], h=float(cache['h']), masses=cache['masses'])
    nm = 6
    Ux, _ = interpolate(fem, cache['V'][:, :nm], X.cpu().numpy())
    U = torch.as_tensor(Ux.transpose(2, 0, 1), dtype=f64, device=dev)
    rng = np.random.default_rng(1)
    amp = torch.as_tensor(rng.standard_normal(nm), dtype=f64, device=dev); phase = torch.as_tensor(rng.uniform(0, 2*np.pi, nm), dtype=f64, device=dev)
    om = torch.as_tensor(2*np.pi*cache['f'][:nm], dtype=f64, device=dev)
    disp = lambda t: torch.einsum('k,kni->ni', amp*torch.sin(om*t+phase), U)
    amp = amp*.03/float(max(disp(t_).norm(dim=1).max() for t_ in np.linspace(0, .5, 51)))
    cen = X.mean(0); axis = torch.tensor([.3, 1., -.2], dtype=f64, device=dev); spin_axis = torch.tensor([1., 0, .4], dtype=f64, device=dev)
    Qt = lambda t: rot(spin_axis, 3.*t)@rot(axis, 1.)
    xt = lambda t: (X+disp(t)-cen)@Qt(t).T+cen
    dt = 1/4800
    x0 = xt(0.)
    y0 = (Y-cen)@Qt(0.).T+cen
    for _ in range(5):
        y0 = S.affine_skin(x0, y0)
    ref0 = float(S.grad(x0, y0).norm())
    print('cold start at the failing state: largest displacement %.4f m, turned 1 rad; |dE/dy| at the cells\' affine skin %.3e' % (
        float(disp(0.).norm(dim=1).max()), ref0), flush=True)
    sol = {}
    for nm_, mk in precs.items():
        t1 = time.time()
        y_, s_ = S.solve(x0, y0, mk, ref0, tol=1e-6, maxit=args.cap, log=lambda s: print('   %s: %s' % (nm_, s), flush=True))
        sol[nm_] = (y_, s_)
        print('cold start, %-21s: products to 1e-2 / 1e-4 / 1e-6 %s / %s / %s; residual %.1e after %d products; %d outer steps, %d rejected, %d negative-curvature exits (%.0f s)' % (
            nm_, s_['hits'][1e-2], s_['hits'][1e-4], s_['hits'][1e-6], s_['residual'], s_['products'], s_['outer'], s_['rejected'], s_['negative'], time.time()-t1), flush=True)
    best = min(sol, key=lambda k: (sol[k][1]['residual'] > 1e-6, sol[k][1]['products']))
    if sol[best][1]['residual'] > 1e-6:
        print('no preconditioner reached 1e-6 from the cold start; no warm steps', flush=True)
        return
    yb, _ = S.solve(x0, sol[best][0], precs[best], ref0, tol=1e-9, maxit=args.cap)
    yp = yc = yb
    for s in range(1, args.steps+1):
        x = xt(s*dt)
        yi = 2*yc-yp
        rf = float(S.grad(x, S.affine_skin(x, yi)).norm())
        line = 'step %d: |dE/dy| at the extrapolated skin %.1e of the affine one;' % (s, float(S.grad(x, yi).norm())/rf)
        ynext = None
        for nm_, mk in precs.items():
            t1 = time.time()
            y_, s_ = S.solve(x, yi, mk, rf, tol=1e-6, maxit=200)
            line += ' %s: to 1e-4 / 1e-6 %s / %s products (%.0f s);' % (nm_, s_['hits'][1e-4], s_['hits'][1e-6], time.time()-t1)
            if ynext is None or nm_ == best:
                ynext = y_
        yn, _ = S.solve(x, ynext, precs[best], rf, tol=1e-9, maxit=300)
        yp, yc = yc, yn
        print(line, flush=True)
    print('total %.0f s' % (time.time()-t0), flush=True)


if __name__ == '__main__':
    main()
