"""DEM geometry of a Power Foam run whose model already carries air ghost sites (training with hull_clip: model.pt has
num_sites real sites, then ghost sites of density 0). As experiments/pacnerf_foam_geometry.py (PAC-NeRF world, no
registration; bodies by cluster_cells), except that the ghost sites can never be material, even when they fall inside
the visual hull. Writes <out>/{cells,geometry}.npz + geometry.json and the z-up copy <out>_zup (as hull_ghosts --zup).
The visual-hull FEM modes of the same material (dem_surface_completion_modal.fem_frequencies, h 8 mm) go to <out>/fem_modes_h0.008.npz."""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from experiments.bread_material_geometry import clipped_faces, neighbors, volume_from_faces
from experiments.pacnerf_foam_geometry import cluster_cells, visual_hull
from experiments.powersim_bread_demo import regular_candidates

RZ = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--run', type=Path, required=True)
    a.add_argument('--scene-dir', type=Path, default=Path('/home/fangsuo/py/dynamic-powerfoam/data/pacnerf_powerfoam/elastic_0_f0'))
    a.add_argument('--out', type=Path, required=True)
    a.add_argument('--min-fraction', type=float, default=.1)
    a.add_argument('--fem-h', type=float, default=.008)
    a.add_argument('--fem-modes', type=int, default=20)
    args = a.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    state = torch.load(args.run/'model.pt', map_location='cpu', weights_only=True)
    pts = state['points'].double().numpy()
    radii = torch.nn.functional.softplus(state['radii'].double(), beta=100).numpy()
    density = torch.nn.functional.softplus(state['density'].double(), beta=100).numpy()
    n_real = int(state['num_sites']) if 'num_sites' in state else len(pts)
    offset = np.array(json.loads((args.scene_dir/'conversion.json').read_text())['offset'])
    to_world = lambda x: x-offset
    all_pairs, possible, active_ids = regular_candidates(pts, radii)
    active = np.zeros(len(pts), bool); active[active_ids] = True
    offsets, ids = neighbors(len(pts), all_pairs)
    candidates = all_pairs[possible]
    area, centres = clipped_faces(pts, radii, candidates, offsets, ids, 64)
    keep = area > 0
    pairs, area, centres = candidates[keep], area[keep], centres[keep]
    volume, surface, _ = volume_from_faces(pts, radii, pairs, area, centres, offsets, ids, active, 12)
    hull = visual_hull(pts, args.scene_dir)
    ghosts_in_hull = int(hull[n_real:].sum())
    hull[n_real:] = False
    material = hull & (volume > 1e-12)
    bond = material[pairs].all(1)
    idx = np.flatnonzero(material)
    remap = -np.ones(len(pts), int); remap[idx] = np.arange(len(idx))
    mp = remap[pairs[bond]]
    graph = sparse.coo_matrix((np.ones(len(mp)), (mp[:, 0], mp[:, 1])), shape=(len(idx),)*2)
    _, labels = connected_components(graph, directed=False)
    cells = idx[labels == np.bincount(labels).argmax()]
    remap = -np.ones(len(pts), int); remap[cells] = np.arange(len(cells))
    in_main = (remap[pairs] >= 0).all(1)
    world = to_world(pts)
    geometry = dict(points=world[cells], radii=radii[cells], pairs=remap[pairs[in_main]], areas=area[in_main], contacts=to_world(centres[in_main]),
                    masses=volume[cells], fixed=np.zeros(len(cells), bool), density=density[cells], source_index=cells)
    np.savez_compressed(args.out/'cells.npz', **geometry)
    bodies = cluster_cells(geometry, min_fraction=args.min_fraction)
    np.savez_compressed(args.out/'geometry.npz', **bodies)
    deg = np.bincount(bodies['pairs'].ravel(), minlength=len(bodies['masses']))
    report = dict(run=str(args.run), real_sites=n_real, ghost_sites=len(pts)-n_real, ghosts_in_hull_excluded=ghosts_in_hull,
                  material_cells=int(material.sum()), kept_cells=len(cells), components=int(labels.max()+1),
                  bodies=len(bodies['masses']), body_bonds=len(bodies['pairs']), body_neighbours_min=int(deg.min()),
                  cells_per_body_max=int(np.bincount(bodies['body_of_cell']).max()),
                  material_volume=float(volume[cells].sum()), exposed_sphere_volume=float(surface[cells].sum()),
                  body_size_mm_median=float(1e3*np.median(bodies['masses'])**(1/3)),
                  bbox_min=world[cells].min(0).tolist(), bbox_max=world[cells].max(0).tolist(),
                  centroid=((world[cells]*volume[cells, None]).sum(0)/volume[cells].sum()).tolist())
    (args.out/'geometry.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(report), flush=True)
    zup = Path(str(args.out)+'_zup'); zup.mkdir(parents=True, exist_ok=True)
    for fn in ('geometry.npz', 'cells.npz'):
        d = dict(np.load(args.out/fn))
        for k in ('points', 'contacts'):
            d[k] = d[k]@RZ.T
        np.savez_compressed(zup/fn, **d)
    shutil.copy(args.out/'geometry.json', zup/'geometry.json')
    # FEM modes of the same material (voxel occupancy of the bounded material cells, ghosts clip them)
    from experiments.dem_surface_completion_modal import fem_frequencies
    mat_all = np.zeros(len(pts), bool); mat_all[cells] = True
    f, V, fem = fem_frequencies(pts, radii, mat_all, offset, 23008.38/1000., .3008, args.fem_h, args.fem_modes+6, return_modes=True)   # as dem_fem_mode_match (scene 0 truth, per unit density)
    np.savez(args.out/('fem_modes_h%g.npz' % args.fem_h), f=f, V=V, **{k_: np.asarray(v_) for k_, v_ in fem.items()})
    print('FEM h %g: %d modes, lowest %s Hz' % (args.fem_h, len(f), np.round(f[:4], 3).tolist()), flush=True)


if __name__ == '__main__':
    main()
