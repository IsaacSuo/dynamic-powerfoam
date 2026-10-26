"""ClosureWarp (closure_warp.py) against the torch ClosureTerms on the part (2026-10-26): energy terms and dE/dx at the
rest shape moved rigidly, stretched and roughened (random per body), the closure's last Newton step, and the time per
gradient (CUDA graph).
  PYTHONPATH=<main repo>:<worktree> python free_surface/closure_warp_check.py"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

from experiments.pacnerf_dem_forward import build
from free_surface.closure_warp import ClosureWarp
from free_surface.sphere_closure import ClosureTerms
from free_surface.sphere_closure_check import rotation

f64, dev = torch.float64, 'cuda'


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument('--part', type=Path, default=Path('output/test_part'))
    args = a_.parse_args()
    torch.manual_seed(0); rng = np.random.default_rng(0)
    stt = torch.load(args.part/'run'/'model.pt', map_location='cpu', weights_only=True)
    p = stt['points'].double().numpy(); r = torch.nn.functional.softplus(stt['radii'].double(), beta=100).numpy()
    g = dict(np.load(args.part/'dem_exact_single'/'geometry.npz'))
    m = build(g, 23008.38, .3008, dev, 'noslip', .25, .3, f64, kinematics='exact', nodalvol=True, nonlinear=False, ogden_alpha=2.)
    st = ClosureTerms(m, g, p, r, dict(np.load(args.part/'boundary_skin.npz')))
    W = ClosureWarp(st)
    W.prepare()
    n = st.n; X = st.X
    th = torch.zeros(n, 9, dtype=f64, device=dev)
    c = X.mean(0)
    Q = torch.as_tensor(rotation(rng.normal(size=3), .7), dtype=f64, device=dev)
    T = torch.eye(3, dtype=f64, device=dev)+torch.as_tensor(rng.normal(size=(3, 3))*.05, dtype=f64, device=dev)
    for name, x in (('rest', X.clone()), ('rigid', c+(X-c)@Q.T+.01),
                    ('stretched', c+(X-c)@(Q@T).T), ('rough', c+(X-c)@(Q@T).T+torch.randn(n, 3, dtype=f64, device=dev)*5e-4)):
        W.cold()                                                                     # a cold start (rest), then warm
        for k in range(4):
            E_w, g_w = W.grad(x)
        xt = x.detach().clone().requires_grad_(True)
        terms = st.energy_terms(th, xt)
        E_t = sum(terms.values())
        g_t = torch.autograd.grad(E_t, xt)[0]
        ts = W.tsum.cpu().numpy()
        sc = max(abs(float(E_t)), 1e-30)
        print('%-9s energy torch %.6e warp %.6e (rel %.1e; bonds %.1e, volume %.1e, o1 %.1e); |dE/dx| rel diff %.1e; closure last step %.1e' % (
            name, float(E_t), float(E_w), abs(float(E_w)-float(E_t))/sc,
            abs(ts[0]-float(terms['bonds']))/sc, abs(ts[1]-float(terms['volume']))/sc, abs(ts[2]-float(terms['outer_stretch']))/sc,
            float((g_w-g_t).norm()/g_t.norm().clamp(min=1e-300)), float(W.tcstep.max())), flush=True)
    torch.cuda.synchronize(); t0 = time.time()
    for k in range(50):
        W.grad(x)
    torch.cuda.synchronize()
    print('one gradient (CUDA graph): %.2f ms' % ((time.time()-t0)/50*1e3), flush=True)


if __name__ == '__main__':
    main()
