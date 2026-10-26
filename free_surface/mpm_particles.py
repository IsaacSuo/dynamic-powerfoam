"""MPM particles filling a foam_exact.py geometry (2026-10-23): the material region is the union of the material
spheres (material sites only: a point inside the union lies in the power cell of a sphere that holds it), uniform
random points at the ground truth's density (8 per (1 cm)^3: p_vol = (dx/2)^3 at dx = 1 cm), moved from the model.pt frame
to the PAC-NeRF world (minus the scene's offset). Prints the sampled volume against the cells' volumes, and the MPM
--height that puts the shape's lowest point (the lowest sphere bottom) --height above the ground.
  python free_surface/mpm_particles.py --out output/round01_pruned_exact"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--out', type=Path, default=Path('output/round01_pruned_exact'))
    a.add_argument('--scene-dir', type=Path, default=Path('/home/fangsuo/py/dynamic-powerfoam/data/pacnerf_powerfoam/elastic_0_f0'))
    a.add_argument('--density', type=float, default=8e6)
    a.add_argument('--height', type=float, default=.02)
    a.add_argument('--name', default='mpm_particles.npz')
    args = a.parse_args()
    c = np.load(args.out/'dem_exact_single'/'cells.npz')
    p, r = c['points'], c['radii']
    vol = float(c['masses'].sum()) if 'masses' in c.files else float('nan')
    offset = np.array(json.loads((args.scene_dir/'conversion.json').read_text())['offset'])
    lo, hi = (p-r[:, None]).min(0), (p+r[:, None]).max(0)
    rng = np.random.default_rng(0)
    n_box = int(round(args.density*np.prod(hi-lo)))
    tree = cKDTree(p)
    keep = []
    for k0 in range(0, n_box, 1_000_000):
        x = lo+(hi-lo)*rng.random((min(1_000_000, n_box-k0), 3))
        _, near = tree.query(x, k=32)
        pw = ((x[:, None]-p[near])**2).sum(-1)-r[near]**2
        keep.append(x[pw.min(1) < 0])
    x = np.concatenate(keep)
    low_surface = float((p[:, 1]-r).min())
    low_particle = float(x[:, 1].min())
    xw = x-offset
    np.savez_compressed(args.out/args.name, x=xw)
    print('%d particles; sampled volume %.6f m^3 (cells %.6f, ratio %.4f); world x %.3f..%.3f, y %.3f..%.3f, z %.3f..%.3f' % (
        len(x), len(x)/args.density, vol, len(x)/args.density/vol, xw[:, 0].min(), xw[:, 0].max(), xw[:, 1].min(), xw[:, 1].max(), xw[:, 2].min(), xw[:, 2].max()))
    print('lowest particle %.5f m above the lowest surface point: MPM --height %.5f puts the surface %.3f m above the ground' % (
        low_particle-low_surface, args.height+low_particle-low_surface, args.height))


if __name__ == '__main__':
    main()
