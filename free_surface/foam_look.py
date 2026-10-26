"""First look at a foam trained without air cells (2026-10-22): its material (the DEM's cells) and surface. The exposed
sphere area per material cell (Sobol rays, a ray point exposed when its power-nearest site is the cell's own), the flat
faces against non-material sites (clipped areas, as the DEM build), where the non-material sites sit and their density,
merged bodies; a picture of the exposed sphere points coloured by body (world, y up)."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import qmc

from experiments.bread_material_geometry import clipped_faces, neighbors
from experiments.powersim_bread_demo import regular_candidates


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--run', type=Path, required=True)
    a.add_argument('--dem', type=Path, required=True, help='world-frame DEM directory (cells.npz, geometry.npz)')
    a.add_argument('--out', type=Path, required=True)
    args = a.parse_args()
    st = torch.load(args.run/'model.pt', map_location='cpu', weights_only=True)
    p = st['points'].double().numpy(); r = torch.nn.functional.softplus(st['radii'].double(), beta=100).numpy()
    dens = torch.nn.functional.softplus(st['density'].double(), beta=100).numpy()
    c = np.load(args.dem/'cells.npz'); g = np.load(args.dem/'geometry.npz')
    src, boc = c['source_index'], g['body_of_cell']
    n = len(p); mat = np.zeros(n, bool); mat[src] = True
    body = -np.ones(n, int); body[src] = boc
    # exposed sphere area
    smp = qmc.Sobol(2, scramble=True, seed=3).random_base2(10)
    z, ph = 1-2*smp[:, 0], 2*np.pi*smp[:, 1]
    U = torch.as_tensor(np.column_stack((np.sqrt(1-z*z)*np.cos(ph), np.sqrt(1-z*z)*np.sin(ph), z)), device='cuda')
    pt, rt = torch.as_tensor(p, device='cuda'), torch.as_tensor(r, device='cuda')
    area = np.zeros(n); pts_, own_ = [], []
    for i in src:
        X = pt[i]+rt[i]*U
        keep = ((torch.cdist(X, pt)**2-rt[None]**2).argmin(1) == i)
        area[i] = 4*np.pi*r[i]**2*float(keep.double().mean())
        if keep.any():
            pts_.append(X[keep][::4].cpu().numpy()); own_.append(np.full(len(pts_[-1]), body[i]))
    # flat faces against non-material sites
    all_pairs, possible, _ = regular_candidates(p, r)
    offs, ids = neighbors(n, all_pairs)
    cand = all_pairs[possible]
    cand = cand[mat[cand].sum(1) == 1]
    fa, _ = clipped_faces(p, r, cand, offs, ids, 64)
    keepf = fa > 0
    cand, fa = cand[keepf], fa[keepf]
    other = np.where(mat[cand[:, 0]], cand[:, 1], cand[:, 0])
    print('sites %d, material %d, non-material %d | bodies %d, merged bodies %d holding %d cells (largest %d cells)' % (
        n, mat.sum(), (~mat).sum(), boc.max()+1, int((np.bincount(boc) > 1).sum()), int((np.bincount(boc)[boc] > 1).sum()), np.bincount(boc).max()), flush=True)
    print('surface: exposed sphere %.4f m^2 on %d cells; flat faces against non-material sites %.4f m^2 (%d faces, %d material cells, %d non-material sites)' % (
        area.sum(), int((area > 0).sum()), fa.sum(), len(fa), len(np.unique(np.where(mat[cand[:, 0]], cand[:, 0], cand[:, 1]))), len(np.unique(other))), flush=True)
    nm = ~mat
    dmat = np.median(dens[mat])
    print('non-material sites: density / material median: quantiles 10/50/90 %% %s; distance to the nearest material site / its radius median %.2f' % (
        np.round(np.quantile(dens[nm]/dmat, [.1, .5, .9]), 3).tolist(),
        float(np.median(torch.cdist(pt[torch.as_tensor(nm, device='cuda')], pt[torch.as_tensor(mat, device='cuda')]).min(1).values.cpu().numpy()/r[nm]))), flush=True)
    X = np.concatenate(pts_); own = np.concatenate(own_)
    col = np.random.default_rng(1).random((boc.max()+1, 3))*.7+.15
    fig = plt.figure(figsize=(16, 12))
    for k, (el, az) in enumerate([(20, -60), (20, 30), (20, 150), (75, -90)]):
        ax = fig.add_subplot(2, 2, k+1, projection='3d')
        ax.scatter(X[:, 0], X[:, 2], X[:, 1], s=.4, c=col[own], depthshade=True)
        lo, hi = X.min(0), X.max(0); mid = (lo+hi)/2; R = (hi-lo).max()/2
        ax.set_xlim(mid[0]-R, mid[0]+R); ax.set_ylim(mid[2]-R, mid[2]+R); ax.set_zlim(mid[1]-R, mid[1]+R)
        ax.view_init(el, az); ax.set_title('elev %d azim %d' % (el, az))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(args.out, dpi=60)


if __name__ == '__main__':
    main()
