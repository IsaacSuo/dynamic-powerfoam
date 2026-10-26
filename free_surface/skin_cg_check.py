"""The CG iteration as one CUDA graph (WarpCG) against the same iteration in torch ops (2026-10-23): a cold start from
the cells' affine skin at the dynamics start, and N dynamics steps from it; products, solutions, energies and time per
step should agree to rounding.
  python free_surface/skin_cg_check.py --out output/round01_pruned_exact --fem output/round01_exact/fem_modes_h0.008.npz --dt 2.0833333e-4 --steps 40 --tols 1e-4"""
import time

import numpy as np
import torch

from free_surface.skin_dynamics_check import build, integrate, parser
from free_surface.skin_warp import warp_solver


def main():
    args = parser().parse_args()
    b = build(args)
    st, W, Sg, x0, y0 = b['st'], b['W'], b['S'], b['x0'], b['y0']
    g = dict(np.load(b['out']/'dem_exact_single'/'geometry.npz'))
    St = warp_solver(st, dict(np.load(b['out']/'boundary_skin.npz')), g['body_of_cell'], W, 1e-7, 'forward', cg_graph=False)
    ya = Sg.affine_skin(x0, y0)
    ref = float(Sg.grad(x0, ya).norm())
    res = {}
    for nm, S in (('graph', Sg), ('torch', St)):
        t0 = time.perf_counter()
        y_, s_ = S.solve(x0, ya, b['prec'], ref, tol=1e-6, maxit=600)
        torch.cuda.synchronize()
        res[nm] = y_
        print('cold start, %-5s: products %d (to 1e-2 / 1e-4 / 1e-6: %s), outer %d, rejected %d, negative %d, residual %.2e, %.2f s' % (
            nm, s_['products'], s_['hits'], s_['outer'], s_['rejected'], s_['negative'], s_['residual'], time.perf_counter()-t0), flush=True)
    print('cold start: |y_graph - y_torch| max %.2e m (|y - y_affine| max %.2e m)' % (
        float((res['graph']-res['torch']).abs().max()), float((res['torch']-ya).abs().max())), flush=True)
    tol = args.tols[0]
    out = {}
    for nm, S in (('graph', Sg), ('torch', St)):
        bb = dict(b, S=S)
        integrate(bb, tol, 3, args.dt)                                                     # warm-up
        torch.cuda.synchronize(); t0 = time.perf_counter()
        H, prods = integrate(bb, tol, args.steps, args.dt)
        torch.cuda.synchronize()
        out[nm] = (H, prods)
        print('%d steps at tol %.0e, %-5s: %.1f ms per step, products %s, H / H_0 - 1 last %.3e' % (
            args.steps, tol, nm, 1e3*(time.perf_counter()-t0)/args.steps, prods, H[-1]/H[0]-1), flush=True)
    Hg, Ht = out['graph'][0], out['torch'][0]
    print('energies every 20 steps: max |H_graph / H_torch - 1| %.1e; products equal: %s' % (
        float(np.abs(Hg/Ht-1).max()), out['graph'][1] == out['torch'][1]), flush=True)


if __name__ == '__main__':
    main()
