"""Selective mass scaling for the tet sub-mesh: raise the explicit time step by adding mass only
to the few (bound) vertex clusters that carry the highest-frequency modes (sliver tets).

Loop: power iteration for the top mode of M^-1 K (cluster space, reference state); clusters with
more than `share` of that mode's kinetic energy get their mass multiplied so that the mode's
frequency drops to the target (lambda ~ 1/m for a mode concentrated on them); until
omega_max <= target. Reports the added mass fraction. Saves the cluster masses.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.tet_dynamics import TetSolid


def top_mode(m, p, iters, y=None):
    n = m.ncl; sm = m.cmass.sqrt()[:, None]
    if y is None:
        y = torch.randn(n, 3, dtype=m.dtype, device=m.device)
    lam = 0.
    for _ in range(iters):
        y = y/y.norm()
        _, Ky = torch.func.jvp(lambda z: torch.zeros(n, 3, dtype=z.dtype, device=z.device).index_add_(0, m.cl, m.forces(m.X+z[m.cl], p)),
                               (torch.zeros_like(y),), (y/sm,))
        z = -Ky/sm; lam = float((y*z).sum()); y = z
    return max(lam, 0)**.5, y/y.norm()


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--geometry', default='output/pacnerf_vem/elastic_0_f0/tet_cells.npz')
    a.add_argument('--tol', type=float, default=5e-4)
    a.add_argument('--target', type=float, default=4500.)
    a.add_argument('--share', type=float, default=.02)
    a.add_argument('--max-rounds', type=int, default=400)
    a.add_argument('--max-added', type=float, default=.005, help='stop if added mass exceeds this fraction')
    a.add_argument('--out', type=Path, default=Path('output/pacnerf_vem/elastic_0_f0/tet_mass_scaled.npz'))
    args = a.parse_args()
    gt = json.loads(Path('data/pacnerf/gt/elastic.json').read_text())['0']
    m = TetSolid(args.geometry, 'cuda', torch.float64, cluster_tol=args.tol)
    p = tuple(torch.tensor(t, dtype=torch.float64, device='cuda') for t in (gt['E'], gt['nu'], 1.))
    m0 = m.cmass.clone(); total = float(m0.sum())
    t0 = time.perf_counter(); y = None
    for r in range(args.max_rounds):
        w, y = top_mode(m, p, 60 if r else 200, y)
        added = float((m.cmass-m0).sum())/total
        if r % 10 == 0 or w <= args.target:
            print(f'round {r}: omega_max {w:.0f}, added mass {added*100:.4f}%, clusters scaled {int((m.cmass > m0*1.0001).sum())}, {time.perf_counter()-t0:.0f}s', flush=True)
        if w <= args.target or added > args.max_added:
            break
        ke = (y**2).sum(1)                                   # mass-normalised mode: kinetic share per cluster
        ke = ke/ke.sum()
        sel = ke > args.share
        if not sel.any():
            sel = ke >= ke.max()*.5
        m.cmass[sel] *= (w/args.target)**2
        y = y+1e-3*torch.randn_like(y)                       # restart near the old mode
    w_final, _ = top_mode(m, p, 300)
    added = float((m.cmass-m0).sum())/total
    print(json.dumps(dict(omega_final=w_final, added_mass_fraction=added, clusters_scaled=int((m.cmass > m0*1.0001).sum()),
                          max_mass_factor=float((m.cmass/m0).max()), seconds=time.perf_counter()-t0)), flush=True)
    np.savez_compressed(args.out, cmass=m.cmass.cpu().numpy(), cl=m.cl.cpu().numpy(), tol=args.tol, target=args.target)


if __name__ == '__main__':
    main()
