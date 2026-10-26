"""Exact geometry, DEM, boundary skin and voxel FEM modes for the new foam (round01_8k, 2026-10-22), offline: material
sites only (the DEM's source_index; the others hold 1.6e-13 m^3 of cells), one body per cell, in the model.pt frame.
  dem_exact_single/  cells.npz, geometry.npz as the DEM builders write them: walls with area > 0 (gpu_geometry, exact),
                     their centroids, cell volumes as masses, body radius (3 V / 4 pi)^(1/3)
  sphere_skin.npz    patch_area, patch_nn (exact) for the completion springs of the current model (part_modes.PatchSprings)
  boundary_skin.npz  boundary_skin.build (Pi + transfinite Shepard, k_sub 4, nq 20)
  fem_modes_h*.npz   voxel FEM of the union of the material balls (dem_surface_completion_modal.fem_frequencies, E / 1000
                     and unit density as for the part)"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

from experiments.dem_surface_completion_modal import fem_frequencies
from free_surface.boundary_skin import build as build_skin
from free_surface.gpu_geometry import Geometry, structure_triangulation

F = torch.nn.functional


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--run', type=Path, default=Path('/home/fangsuo/py/dynamic-powerfoam/.claude/worktrees/thin-surface/output/pacnerf_elastic0_f0_round01_8k'))
    a.add_argument('--dem', type=Path, default=Path('/home/fangsuo/py/dynamic-powerfoam/.claude/worktrees/thin-surface/output/pacnerf_dem_round01_8k/elastic_0_zup'))
    a.add_argument('--out', type=Path, default=Path('output/round01_exact'))
    a.add_argument('--fem-h', type=float, nargs='*', default=[.008, .006])
    a.add_argument('--modes', type=int, default=20)
    a.add_argument('--interp', default='shepard0', help="boundary_skin.build interpolation: 'shepard0' (positive weights; default since the "
                   "projection's signed weights left a skin point with negative stiffness here), 'shepard' (with Pi), 'mvc'")
    args = a.parse_args()
    t0 = time.time()
    st = torch.load(args.run/'model.pt', map_location='cpu', weights_only=True)
    src = np.load(args.dem/'cells.npz')['source_index']
    p = st['points'].double()[src].cuda(); r = F.softplus(st['radii'].double(), beta=100)[src].cuda()
    pn, rn = p.cpu().numpy(), r.cpu().numpy(); n = len(pn)
    args.out.mkdir(parents=True, exist_ok=True)
    if not (args.out/'boundary_skin.npz').exists():
        S = structure_triangulation(pn, rn); G = Geometry(p, r, S)
        keep = (G.wall_A > 0).cpu().numpy()
        pairs = G.P.cpu().numpy()[keep]; area = G.wall_A.cpu().numpy()[keep]; cen = G.wall_c.cpu().numpy()[keep]; V = G.V.cpu().numpy()
        print('material sites %d (hidden %d); walls %d; volume %.6f m^3; exposed area %.4f m^2 on %d cells (%.0f s)' % (
            n, int(S['hidden'].sum()), len(pairs), V.sum(), float(G.patch_A.sum()), int((G.patch_A > 0).sum()), time.time()-t0), flush=True)
        cells = dict(points=pn, radii=rn, pairs=pairs, areas=area, contacts=cen, masses=V, fixed=np.zeros(n, bool), density=np.ones(n), source_index=src)
        single = dict(points=pn, radii=(3*V/(4*math.pi))**(1/3), pairs=pairs, areas=area, contacts=cen, masses=V, fixed=np.zeros(n, bool), body_of_cell=np.arange(n))
        (args.out/'dem_exact_single').mkdir(exist_ok=True)
        np.savez_compressed(args.out/'dem_exact_single'/'cells.npz', **cells)
        np.savez_compressed(args.out/'dem_exact_single'/'geometry.npz', **single)
        np.savez_compressed(args.out/'sphere_skin.npz', patch_area=G.patch_A.cpu().numpy(), patch_nn=G.patch_nn.cpu().numpy(), patch_n=G.patch_n.cpu().numpy())
        sk, d = build_skin(p, r, k_sub=4, nq=20, G=G, S=S, interp=args.interp)
        d.pop('per_patch')
        np.savez_compressed(args.out/'boundary_skin.npz', **sk)
        print('boundary skin: %d skin points (%d corners), %d regions; %s (%.0f s)' % (len(sk['Y']), sk['n_corners'], len(sk['part_site']),
              ', '.join('%s %.3g' % (k_, d[k_]) for k_ in ('sites_with_islands', 'affine_reproduction_max', 'partition_of_unity_max', 'sumA_minus_A_of_sphere_max',
                                                           'mass_matrix_min_eig_ratio', 'seconds')), time.time()-t0), flush=True)
    E, nu = 23008.38, .3008
    for h in args.fem_h:
        cache = args.out/('fem_modes_h%g.npz' % h)
        if cache.exists():
            continue
        t1 = time.time()
        f, Vm, F_ = fem_frequencies(pn, rn, np.ones(n, bool), np.zeros(3), E/1000., nu, h, args.modes+6, return_modes=True)
        np.savez(cache, f=f, V=Vm, **{k_: np.asarray(v_) for k_, v_ in F_.items()})
        print('FEM h %.1f mm: %d nodes, volume %.6f m^3, lowest %s Hz (%.0f s)' % (1e3*h, len(F_['nodes']), float(F_['masses'].sum()) if 'masses' in F_ else float('nan'),
              np.round(f[:8], 3).tolist(), time.time()-t1), flush=True)


if __name__ == '__main__':
    main()
