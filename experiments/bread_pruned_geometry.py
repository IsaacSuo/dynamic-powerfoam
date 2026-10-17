"""Rebuild bounded power geometry after the saved small-cell deletion.

Keep global checkpoint IDs; recompute regular neighbors, clipped faces and
support volumes using the same polygon/quadrature settings as the baseline.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from experiments.powersim_bread_demo import regular_candidates
from experiments.bread_material_geometry import neighbors, clipped_faces, volume_from_faces


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, default=Path(
        'output/powersim_bread/small_cell_prune_1e-4/selection.npz'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    source = Path('output/powersim_bread/material_geometry.npz')
    old = np.load(source)
    keep = np.load(args.selection)['keep']
    assert np.all(~keep | old['keep'])
    original_ids = np.flatnonzero(keep)
    p, r = old['points'][keep], old['radii'][keep]
    print(f'Rebuilding regular triangulation for {len(p)} sites', flush=True)
    all_pairs, possible, active_ids = regular_candidates(p, r)
    offsets, ids = neighbors(len(p), all_pairs)
    candidate = all_pairs[possible]
    area, centers = clipped_faces(p, r, candidate, offsets, ids, 64)
    positive = area > 0
    pairs, area, centers = candidate[positive], area[positive], centers[positive]
    active = np.zeros(len(p), dtype=bool); active[active_ids] = True
    print(f'Rebuilt {len(pairs)} positive faces; computing volumes', flush=True)
    volume, _, _ = volume_from_faces(p, r, pairs, area, centers, offsets, ids, active, 12)
    assert np.isfinite(volume).all() and (volume > 0).all()
    volumes = np.zeros_like(old['volumes']); volumes[keep] = volume
    global_pairs = original_ids[pairs]
    np.savez_compressed(args.out/'material_geometry.npz', points=old['points'], radii=old['radii'],
                        volumes=volumes, pairs=global_pairs, areas=area, contacts=centers, keep=keep)
    size = len(old['keep'])
    old_pair_ids = old['pairs'].astype(np.int64)
    old_keys = old_pair_ids[:, 0]*size+old_pair_ids[:, 1]
    new_keys = global_pairs[:, 0]*size+global_pairs[:, 1]
    common, old_idx, new_idx = np.intersect1d(old_keys, new_keys, return_indices=True)
    graph = sparse.coo_matrix((np.ones(2*len(pairs)),
        (np.r_[pairs[:, 0], pairs[:, 1]], np.r_[pairs[:, 1], pairs[:, 0]])), shape=(len(p), len(p))).tocsr()
    count, labels = connected_components(graph)
    component_mass = np.bincount(labels, weights=volume)
    report = dict(before_cells=int(old['keep'].sum()), after_cells=int(keep.sum()),
        removed_cells=int(old['keep'].sum()-keep.sum()), before_bonds=len(old['pairs']),
        after_bonds=len(pairs), new_bonds=int(len(new_keys)-len(common)),
        removed_bonds=int(len(old_keys)-len(common)), before_volume=float(old['volumes'].sum()),
        after_volume=float(volume.sum()), total_volume_relative_change=float(volume.sum()/old['volumes'].sum()-1),
        retained_volume_relative_L1=float(np.abs(volume-old['volumes'][keep]).sum()/volume.sum()),
        common_face_area_relative_L1=float(np.abs(area[new_idx]-old['areas'][old_idx]).sum()/area[new_idx].sum()),
        components=int(count), main_component_mass_fraction=float(component_mass.max()/volume.sum()),
        min_volume_before=float(old['volumes'][old['keep']].min()), min_volume_after=float(volume.min()),
        elapsed_seconds=time.perf_counter()-start, face_polygon_segments=64, sphere_samples=4096,
        source_geometry_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        selection_sha256=hashlib.sha256(args.selection.read_bytes()).hexdigest(),
        source_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                      for name in ('bread_pruned_geometry.py', 'bread_material_geometry.py', 'powersim_bread_demo.py')})
    (args.out/'material_geometry.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
