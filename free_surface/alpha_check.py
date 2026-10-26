"""alpha_complex.AlphaComplex against qhull's regular triangulation (2026-10-24), on the jack at rest and at a saved
step's bodies (radii kept): the walls (pairs with area > 0) both ways, each wall seen from its two cells alike, the
exact geometry (gpu_geometry.Geometry: volumes, wall areas, exposed areas) on either structure, the per-cell status,
the times (BVH + candidates, the kernel, packing, the values).
  python free_surface/alpha_check.py --out output/round01_pruned_exact --state output/round01_pruned_exact/skin_drop_state_step900.npz"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import warp as wp

from free_surface.alpha_complex import AlphaComplex
from free_surface.gpu_geometry import Geometry, structure_triangulation

f64 = torch.float64


def timed(f):
    torch.cuda.synchronize(); wp.synchronize(); t0 = time.perf_counter()
    r = f()
    torch.cuda.synchronize(); wp.synchronize()
    return r, time.perf_counter()-t0


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--out', type=Path, required=True)
    a.add_argument('--state', type=Path, required=True)
    args = a.parse_args()
    wp.init()
    c = np.load(args.out/'dem_exact_single'/'cells.npz')
    r = torch.as_tensor(c['radii'], dtype=f64, device='cuda')
    cases = (('rest', torch.as_tensor(c['points'], dtype=f64, device='cuda')),
             (args.state.stem, torch.as_tensor(np.load(args.state)['xm1'], dtype=f64, device='cuda')))
    n = len(r)
    A = AlphaComplex(n)
    for name, p in cases:
        for _ in range(2):                                                         # (warm: kernels built)
            out = A(p, r)
        (coff_j), tc = timed(lambda: A.candidates(p, r))
        out, ta = timed(lambda: A(p, r))
        tk = ta-tc
        S_a, tpk = timed(lambda: A.structure(out, n))
        st = out['status'].cpu().numpy()
        print('%s: %d candidate planes (%.1f per cell); status %s; BVH + candidates %.1f ms, kernel %.1f ms, packing %.0f ms' % (
            name, out['ncand'], out['ncand']/n, {int(k): int((st == k).sum()) for k in np.unique(st)}, 1e3*tc, 1e3*tk, 1e3*tpk), flush=True)
        # each wall from its two cells
        W, nw = out['walls'].cpu().numpy(), out['nwall'].cpu().numpy()
        Lk, nl = out['links'].cpu().numpy(), out['nlink'].cpu().numpy()
        side = {}
        for i in range(n):
            for s in range(nw[i]):
                side[(i, int(W[i, s]))] = frozenset(Lk[i, s, :nl[i, s]].tolist())
        one = [(i, j) for (i, j) in side if (j, i) not in side]
        diff = [(i, j) for (i, j) in side if i < j and (j, i) in side and side[(i, j)] != side[(j, i)]]
        print('   walls seen from one side only: %d; walls whose links differ between the sides: %d' % (len(one), len(diff)), flush=True)
        # against qhull
        (S_q), tq = timed(lambda: structure_triangulation(p.cpu().numpy(), r.cpu().numpy()))
        G_q, tgq = timed(lambda: Geometry(p, r, S_q))
        G_a, tga = timed(lambda: Geometry(p, r, S_a))
        wq = {(int(x), int(y)): float(A_) for (x, y), A_ in zip(G_q.P.cpu().numpy(), G_q.wall_A.cpu().numpy()) if A_ > 0}
        wa = {(int(x), int(y)): float(A_) for (x, y), A_ in zip(G_a.P.cpu().numpy(), G_a.wall_A.cpu().numpy()) if A_ > 0}
        miss, extra = set(wq)-set(wa), set(wa)-set(wq)
        common = sorted(set(wq) & set(wa))
        dA = max(abs(wa[k]/wq[k]-1) for k in common)
        Aq = np.array(list(wq.values()))
        print('   qhull walls %d, alpha walls %d; missing %d (largest area %.1e of the median), extra %d (largest %.1e); common walls area max relative difference %.1e' % (
            len(wq), len(wa), len(miss), max((wq[k] for k in miss), default=0.)/np.median(Aq), len(extra), max((wa[k] for k in extra), default=0.)/np.median(Aq), dA), flush=True)
        rel = lambda x, y: float(((x-y).abs()/y.abs().clamp(min=1e-300)).max())
        print('   volumes max relative difference %.1e (cells with volume); exposed areas max abs difference %.1e of the median patch; hidden qhull %d alpha %d' % (
            rel(G_a.V[G_q.V > 0], G_q.V[G_q.V > 0]), float((G_a.patch_A-G_q.patch_A).abs().max()/G_q.patch_A[G_q.patch_A > 0].median()),
            int(S_q['hidden'].sum()), int(S_a['hidden'].sum())), flush=True)
        print('   times: qhull structure %.2f s, values on it %.3f s; values on the alpha structure %.3f s (its link width %d against qhull %d)' % (
            tq, tgq, tga, S_a['links'].shape[1], S_q['links'].shape[1]), flush=True)


if __name__ == '__main__':
    main()
