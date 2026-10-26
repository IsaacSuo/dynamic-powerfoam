"""A small synthetic part for the sphere surface (2026-10-22): an L bracket with a through hole in its base plate, filled
with irregular power cells and NO air sites: every cell is material and the outer surface is the cells' own spheres.
Sites: greedy maximal Poisson-disk sampling (minimum distance s) of the bracket shrunk by s; radii s (1 + rvar u) (default rvar 0.6), u
uniform in [0, 1], so every point of the sampled region is inside its power cell's ball (pi_i(x) <= pi_j*(x) <= s^2 -
r_min^2 <= 0 for the nearest site j*) and the outermost balls reach about s beyond the outermost sites. Hidden sites
(empty cells) are dropped. Writes <out>/run/model.pt (points, radii as softplus beta 100 raw values, density),
<out>/dem (bodies by cluster_cells, min_fraction 0.1), <out>/dem_single (one body per cell), geometry.json, and a
picture of the surface (exposed sphere points, coloured by cell) from four directions."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import qmc

from experiments.bread_material_geometry import clipped_faces, neighbors, volume_from_faces
from experiments.pacnerf_foam_geometry import cluster_cells
from experiments.powersim_bread_demo import regular_candidates


def box_sdf(x, lo, hi):
    c, h = (lo+hi)/2, (hi-lo)/2
    q = np.abs(x-c)-h
    return np.linalg.norm(np.maximum(q, 0), axis=1)+np.minimum(q.max(1), 0)


def bracket_sdf(x):
    base = box_sdf(x, np.array([0, 0, 0.]), np.array([.09, .075, .024]))
    upright = box_sdf(x, np.array([0, 0, 0.]), np.array([.09, .024, .06]))
    hole = np.linalg.norm(x[:, :2]-np.array([.055, .048]), axis=1)-.011
    return np.maximum(np.minimum(base, upright), -hole)


def poisson(sdf, lo, hi, s, depth, rng, per_cell=60):
    n = int(per_cell*np.prod(hi-lo)/s**3)
    C = lo+rng.random((n, 3))*(hi-lo)
    C = C[sdf(C) < -depth]
    grid, out = {}, []
    for x in C:
        k = tuple((x//s).astype(int))
        ok = True
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for y in grid.get((k[0]+dx, k[1]+dy, k[2]+dz), ()):
                        if ((x-y)**2).sum() < s*s:
                            ok = False; break
                    if not ok: break
                if not ok: break
            if not ok: break
        if ok:
            grid.setdefault(k, []).append(x); out.append(x)
    return np.array(out)


def cells_of(p, r):
    all_pairs, possible, active_ids = regular_candidates(p, r)
    active = np.zeros(len(p), bool); active[active_ids] = True
    offs, ids = neighbors(len(p), all_pairs)
    cand = all_pairs[possible]
    area, cen = clipped_faces(p, r, cand, offs, ids, 64)
    keep = area > 0
    pairs, area, cen = cand[keep], area[keep], cen[keep]
    volume, cone, _ = volume_from_faces(p, r, pairs, area, cen, offs, ids, active, 12)
    return pairs, area, cen, volume, cone


def exposed_points(p, r, rays=256):
    smp = qmc.Sobol(2, scramble=True, seed=3).random(rays)
    z, ph = 1-2*smp[:, 0], 2*np.pi*smp[:, 1]
    u = np.column_stack((np.sqrt(1-z*z)*np.cos(ph), np.sqrt(1-z*z)*np.sin(ph), z))
    X = (p[:, None, :]+r[:, None, None]*u[None]).reshape(-1, 3); own = np.repeat(np.arange(len(p)), rays)
    pt, rt = torch.as_tensor(p, device='cuda'), torch.as_tensor(r, device='cuda')
    keep = np.zeros(len(X), bool)
    for s0 in range(0, len(X), 8192):
        y = torch.as_tensor(X[s0:s0+8192], device='cuda')
        pw = torch.cdist(y, pt)**2-rt[None]**2
        pw[torch.arange(len(y)), torch.as_tensor(own[s0:s0+8192], device='cuda')] = np.inf
        keep[s0:s0+8192] = (pw.min(1).values >= -1e-12).cpu().numpy()      # inside no other ball
    return X[keep], own[keep]


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--out', type=Path, default=Path('output/test_part'))
    a.add_argument('--s', type=float, default=.005)
    a.add_argument('--seed', type=int, default=0)
    a.add_argument('--rvar', type=float, default=.6, help='radii s (1 + rvar u)')
    args = a.parse_args()
    rng = np.random.default_rng(args.seed)
    lo, hi = np.array([0, 0, 0.]), np.array([.09, .075, .06])
    s = args.s
    p = poisson(bracket_sdf, lo, hi, s, s, rng)
    r = s*(1+args.rvar*rng.random(len(p)))
    pairs, area, cen, volume, cone = cells_of(p, r)
    keep = volume > 0
    p, r = p[keep], r[keep]
    pairs, area, cen, volume, cone = cells_of(p, r)
    assert (volume > 0).all()
    n = len(p)
    # coverage: the sampled region (sdf < -s) must be material everywhere (no inner gaps)
    g = .001
    axes = [np.arange(lo[k]+g/2, hi[k], g) for k in range(3)]
    G = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
    G = G[bracket_sdf(G) < -s]
    pt, rt = torch.as_tensor(p, device='cuda'), torch.as_tensor(r, device='cuda')
    gap = 0
    for s0 in range(0, len(G), 8192):
        y = torch.as_tensor(G[s0:s0+8192], device='cuda')
        pw = torch.cdist(y, pt)**2-rt[None]**2
        gap += int((pw.min(1).values > 0).sum())
    # hull side: is every site's own power cell around it? (site inside its cell)
    d = p[pairs[:, 1]]-p[pairs[:, 0]]; L = np.linalg.norm(d, axis=1); nrm = d/L[:, None]
    hi_ = ((cen-p[pairs[:, 0]])*nrm).sum(1); hj_ = L-hi_
    outside = np.zeros(n, bool); outside[pairs[hi_ <= 0, 0]] = True; outside[pairs[hj_ <= 0, 1]] = True
    geometry = dict(points=p, radii=r, pairs=pairs, areas=area, contacts=cen, masses=volume, fixed=np.zeros(n, bool), density=np.ones(n), source_index=np.arange(n))
    raw = lambda v: np.log(np.expm1(100*v))/100
    (args.out/'run').mkdir(parents=True, exist_ok=True)
    torch.save(dict(points=torch.as_tensor(p), radii=torch.as_tensor(raw(r)), density=torch.as_tensor(raw(np.ones(n)))), args.out/'run'/'model.pt')
    for tag, bodies in (('dem', cluster_cells(geometry, min_fraction=.1)),
                        ('dem_single', dict(points=p, radii=(3*volume/(4*np.pi))**(1/3), pairs=pairs, areas=area, contacts=cen, masses=volume,
                                            fixed=np.zeros(n, bool), body_of_cell=np.arange(n)))):
        (args.out/tag).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.out/tag/'cells.npz', **geometry)
        np.savez_compressed(args.out/tag/'geometry.npz', **bodies)
    nb = len(cluster_cells(geometry, min_fraction=.1)['masses'])
    deg = np.bincount(pairs.ravel(), minlength=n)
    sph = cone > 0
    rep = dict(rvar=args.rvar, cells=n, bodies_merged=nb, bonds=len(pairs), volume=float(volume.sum()), exposed_sphere_volume=float(cone.sum()),
               surface_cells=int(sph.sum()), inner_gap_points_1mm=gap, sampled_region_points=len(G),
               sites_outside_own_cell=int(outside.sum()), surface_sites_outside_own_cell=int((outside & sph).sum()),
               neighbours_min=int(deg.min()), neighbours_median=float(np.median(deg)), volume_median_mm3=float(1e9*np.median(volume)),
               volume_min_mm3=float(1e9*volume.min()), s=s, seed=args.seed)
    (args.out/'geometry.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep), flush=True)
    X, own = exposed_points(p, r)
    col = np.random.default_rng(1).random((n, 3))*.7+.15
    fig = plt.figure(figsize=(16, 12))
    for k, (el, az) in enumerate([(25, -60), (25, 30), (25, 150), (70, -100)]):
        ax = fig.add_subplot(2, 2, k+1, projection='3d')
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=1.2, c=col[own], depthshade=True)
        ax.set_xlim(-.01, .1); ax.set_ylim(-.01, .085); ax.set_zlim(-.01, .07); ax.set_box_aspect((11, 9.5, 8))
        ax.view_init(el, az); ax.set_title('elev %d azim %d' % (el, az))
    plt.tight_layout(); plt.savefig(args.out/'surface.png', dpi=70)


if __name__ == '__main__':
    main()
