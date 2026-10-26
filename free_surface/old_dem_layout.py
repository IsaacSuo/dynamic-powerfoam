"""Our jack (output/round01_pruned_exact/dem_exact_single) in the layout the old pipeline's surface completion reads
(2026-10-24, for old_dem_drop.py), CPU only: run/model.pt with the material sites alone (points; radii stored as the
foam stores them, softplus^-1 at beta 100), dem/cells.npz with source_index the identity, dem/geometry.npz our geometry
with the masses replaced by the old routine's cell volumes (foam_surface_completion: regular_candidates, clipped_faces
64, volume_from_faces 12; it asserts the DEM masses equal them: interior cells agree to rounding, the exposed boundary
cells differ by the routine's sphere-cap approximation). Checks: bodies 7478, positions identical to ours.
  python free_surface/old_dem_layout.py"""
import argparse
from pathlib import Path

import numpy as np
import torch

from experiments.bread_material_geometry import clipped_faces, neighbors, volume_from_faces
from experiments.powersim_bread_demo import regular_candidates


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--src', type=Path, default=Path('output/round01_pruned_exact/dem_exact_single'))
    a.add_argument('--out', type=Path, default=Path('output/round01_pruned_exact/old_dem_layout'))
    args = a.parse_args()
    c, g = dict(np.load(args.src/'cells.npz')), dict(np.load(args.src/'geometry.npz'))
    p, r = np.asarray(c['points'], float), np.asarray(c['radii'], float)
    N = len(p)
    all_pairs, possible, active_ids = regular_candidates(p, r)
    active = np.zeros(N, bool); active[active_ids] = True
    offs, ids = neighbors(N, all_pairs)
    cand = all_pairs[possible]
    area, cen = clipped_faces(p, r, cand, offs, ids, 64)
    keep = area > 0
    volume, _, _ = volume_from_faces(p, r, cand[keep], area[keep], cen[keep], offs, ids, active, 12)
    raw = np.log(np.expm1(100*r))/100                                              # softplus(raw, beta 100) = r
    assert np.allclose(torch.nn.functional.softplus(torch.as_tensor(raw), beta=100).numpy(), r, rtol=1e-12)
    (args.out/'run').mkdir(parents=True, exist_ok=True); (args.out/'dem').mkdir(exist_ok=True)
    torch.save(dict(points=torch.as_tensor(p, dtype=torch.float32), radii=torch.as_tensor(raw, dtype=torch.float32)), args.out/'run'/'model.pt')
    # (float32 as the foam stores them: the completion compares positions at 1e-6 relative and recomputes the volumes)
    st = torch.load(args.out/'run'/'model.pt', weights_only=True)
    p32 = st['points'].double().numpy(); r32 = torch.nn.functional.softplus(st['radii'].double(), beta=100).numpy()
    all_pairs, possible, active_ids = regular_candidates(p32, r32)
    active = np.zeros(N, bool); active[active_ids] = True
    offs, ids = neighbors(N, all_pairs)
    cand = all_pairs[possible]
    area, cen = clipped_faces(p32, r32, cand, offs, ids, 64)
    keep = area > 0
    V32, _, _ = volume_from_faces(p32, r32, cand[keep], area[keep], cen[keep], offs, ids, active, 12)
    c2 = dict(c); c2['source_index'] = np.arange(N)
    g2 = dict(g); g2['masses'] = V32
    np.savez_compressed(args.out/'dem'/'cells.npz', **c2)
    np.savez_compressed(args.out/'dem'/'geometry.npz', **g2)
    rel = V32/g['masses']-1
    print('bodies %d (ours %d); positions max |difference| %.1e m (float32 run %.1e m); masses: old routine on the float32 sites, total %.6f (ours %.6f), '
          'per cell relative difference median %.1e, p99 %.1e, max %.1e (%d cells beyond 1e-6); walls ours %d (the routine finds %d)' % (
              len(g2['points']), len(g['points']), float(np.abs(g2['points']-g['points']).max()), float(np.abs(p32-p).max()), V32.sum(), g['masses'].sum(),
              np.median(np.abs(rel)), np.percentile(np.abs(rel), 99), np.abs(rel).max(), int((np.abs(rel) > 1e-6).sum()), len(g['pairs']), int(keep.sum())))


if __name__ == '__main__':
    main()
