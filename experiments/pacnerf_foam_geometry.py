"""Power Foam reconstruction of a PAC-NeRF frame -> EnergyFoam geometry (material + air cells).

All trained cells (material and air) define the bounded power diagram: regular-triangulation
candidates, faces clipped by every neighbour and by the support spheres, cell volumes by the
divergence theorem with sampled exposed sphere patches (bread_material_geometry). Material
cells are those whose centre projects inside the object mask in every view (visual hull):
optical density is unreliable inside an opaque object, which no camera sees. Material cells
keep their bounded power volumes and faces (so faces toward air cells or toward empty space
close them, option (a)); bonds are faces between material cells; the largest connected
material component is kept (cells.npz). Tiny / poorly connected cells are merged into rigid
bodies (cluster_cells) for a usable explicit time step (geometry.npz, body_of_cell maps cells
to bodies). Output coordinates are PAC-NeRF world (scene - offset).
--registration (Spring-Gaus real capture): the scene is a COLMAP one (springgaus_to_powerfoam),
the visual hull uses its RGBA masks, and outputs are mapped by the similarity foam -> COLMAP ->
dynamic world (metres; springgaus_dynamic register): points/contacts transformed, radii x k,
areas x k^2, volumes x k^3.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from experiments.bread_material_geometry import clipped_faces, neighbors, volume_from_faces
from experiments.powersim_bread_demo import regular_candidates


def visual_hull(points, scene_dir):
    meta = json.loads((scene_dir/'transforms_all.json').read_text())
    W, H = meta['w'], meta['h']
    focal = .5*W/np.tan(.5*meta['camera_angle_x'])
    inside = np.ones(len(points), bool)
    for frame in meta['frames']:
        mask = np.asarray(Image.open(scene_dir/f"{frame['file_path']}.png"))[..., 3] > 127
        c2w = np.array(frame['transform_matrix'])
        cam = (np.c_[points, np.ones(len(points))]@np.linalg.inv(c2w).T)[:, :3]
        z = -cam[:, 2]
        u = focal*cam[:, 0]/z+W/2; v = -focal*cam[:, 1]/z+H/2
        ok = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        hit = np.zeros(len(points), bool)
        hit[ok] = mask[v[ok].astype(int), u[ok].astype(int)]
        inside &= hit
    return inside


def visual_hull_colmap(points, scene_dir):
    import pycolmap
    recon = pycolmap.Reconstruction(str(scene_dir/'sparse'/'0'))
    img_dir = next(scene_dir.glob('images_*'))
    inside = np.ones(len(points), bool)
    for im in recon.images.values():
        cam = recon.cameras[im.camera_id]
        mask = np.asarray(Image.open(img_dir/im.name))[..., 3] > 127
        fx, fy, cx, cy = np.array(cam.params)*mask.shape[1]/cam.width
        Rt = im.cam_from_world().matrix()
        x = points@Rt[:, :3].T+Rt[:, 3]
        z = x[:, 2]
        u = fx*x[:, 0]/z+cx; v = fy*x[:, 1]/z+cy
        ok = (z > 0) & (u >= 0) & (u < mask.shape[1]) & (v >= 0) & (v < mask.shape[0])
        hit = np.zeros(len(points), bool)
        hit[ok] = mask[v[ok].astype(int), u[ok].astype(int)]
        inside &= hit
    return inside


def foam_to_dynamic(scene_dir, registration):
    """x_dyn = A x_foam + b (uniform scale k = det(A)^(1/3))."""
    conv = json.loads((scene_dir/'conversion.json').read_text())
    reg = json.loads(Path(registration).read_text())
    R, s = np.array(reg['R']), reg['s']
    k = s/conv['scale']
    A = k*R
    b = R@(s*(np.array(conv['centre'])-.5/conv['scale']-np.array(reg['colmap_mean'])))+np.array(reg['t'])
    return A, b, k


def cluster_cells(geometry, min_fraction=.1, min_neighbours=4, min_distance=.3):
    """Merge tiny or poorly connected cells into rigid bodies (union of cells).

    Repeatedly, the smallest cluster below min_fraction x median cell volume, or with fewer
    than min_neighbours cluster neighbours, joins the neighbour cluster sharing the largest
    face area; two clusters whose centres are closer than min_distance x the smaller cluster's
    size (volume^(1/3)) merge (near-duplicate sites from densification make very stiff bonds). A body's centre is the volume-weighted mean of its cells' centres, its mass the
    summed volume, its isotropic inertia sum m_k (0.4 r_k^2 + 2/3 |c_k - C|^2) with r_k the
    equivalent-sphere radius (stored as radii with 0.4 m r^2 = I, as LinearBondsGPU expects).
    Bonds between bodies merge faces: summed area, area-weighted contact point.
    """
    p, vol = geometry['points'], geometry['masses']
    pairs, area, contact = geometry['pairs'], geometry['areas'], geometry['contacts']
    n = len(p)
    parent = np.arange(n)
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    threshold = min_fraction*np.median(vol)
    while True:
        root = np.array([find(a) for a in range(n)])
        cvol = np.bincount(root, weights=vol, minlength=n)
        ra, rb = root[pairs[:, 0]], root[pairs[:, 1]]
        cross = ra != rb
        key = {}
        for a, b, s in zip(ra[cross], rb[cross], area[cross]):
            key[(a, b)] = key.get((a, b), 0.)+s; key[(b, a)] = key.get((b, a), 0.)+s
        nbrs = {}
        for (a, b), s in key.items():
            nbrs.setdefault(a, []).append((s, b))
        roots = np.unique(root)
        centre = np.stack([np.bincount(root, weights=vol*p[:, k], minlength=n) for k in range(3)], 1)
        centre /= np.maximum(cvol, 1e-300)[:, None]
        size = cvol**(1/3)
        close = {}
        for (a, b) in key:
            if np.linalg.norm(centre[a]-centre[b]) < min_distance*min(size[a], size[b]) and cvol[a] <= cvol[b]:
                close.setdefault(a, b)
        bad = [a for a in roots if cvol[a] < threshold or len(nbrs.get(a, [])) < min_neighbours or a in close]
        bad = [a for a in bad if a in nbrs]
        if not bad:
            break
        merged = set()
        for a in sorted(bad, key=lambda a: cvol[a]):
            if a in merged:
                continue
            b = close.get(a, max(nbrs[a])[1])
            if b in merged:
                continue
            parent[a] = b; merged.update((a, b))
    root = np.array([find(a) for a in range(n)])
    roots, body = np.unique(root, return_inverse=True)
    m = np.bincount(body, weights=vol)
    centre = np.stack([np.bincount(body, weights=vol*p[:, k]) for k in range(3)], 1)/m[:, None]
    r_eq = (3*vol/(4*np.pi))**(1/3)
    inertia = np.bincount(body, weights=vol*(.4*r_eq**2+2/3*((p-centre[body])**2).sum(1)))
    ba, bb = body[pairs[:, 0]], body[pairs[:, 1]]
    keep = ba != bb
    lo, hi = np.minimum(ba, bb)[keep], np.maximum(ba, bb)[keep]
    edge, inv = np.unique(np.stack((lo, hi), 1), axis=0, return_inverse=True)
    inv = inv.ravel()
    e_area = np.bincount(inv, weights=area[keep])
    e_contact = np.stack([np.bincount(inv, weights=area[keep]*contact[keep, k]) for k in range(3)], 1)/e_area[:, None]
    return dict(points=centre, radii=np.sqrt(inertia/(.4*m)), pairs=edge, areas=e_area, contacts=e_contact,
                masses=m, fixed=np.zeros(len(m), bool), body_of_cell=body)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, default=Path('output/pacnerf_elastic0_f0_v1'))
    p.add_argument('--scene-dir', type=Path, default=Path('data/pacnerf_powerfoam/elastic_0_f0'))
    p.add_argument('--out', type=Path, default=Path('output/pacnerf_dem/elastic_0_f0'))
    p.add_argument('--min-fraction', type=float, default=.3,
                   help='merge bodies below this fraction of the median cell volume (0.3: ~3x larger stable dt than 0.1)')
    p.add_argument('--registration', type=Path, default=None, help='springgaus_dynamic register json (COLMAP scene)')
    p.add_argument('--extra-material', type=Path, default=None,
                   help='npy of cell indices that are material regardless of the visual hull (fill_bottom.py)')
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    state = torch.load(args.run/'model.pt', map_location='cpu', weights_only=True)
    pts = state['points'].double().numpy()
    radii = torch.nn.functional.softplus(state['radii'].double(), beta=100).numpy()
    density = torch.nn.functional.softplus(state['density'].double(), beta=100).numpy()
    conversion = json.loads((args.scene_dir/'conversion.json').read_text())
    if args.registration is None:
        offset = np.array(conversion['offset'])
        to_world, k = (lambda x: x-offset), 1.
    else:
        A, b, k = foam_to_dynamic(args.scene_dir, args.registration)
        to_world = lambda x: x@A.T+b
    all_pairs, possible, active_ids = regular_candidates(pts, radii)
    active = np.zeros(len(pts), bool); active[active_ids] = True
    offsets, ids = neighbors(len(pts), all_pairs)
    candidates = all_pairs[possible]
    area, centres = clipped_faces(pts, radii, candidates, offsets, ids, 64)
    keep = area > 0
    pairs, area, centres = candidates[keep], area[keep], centres[keep]
    volume, surface, _ = volume_from_faces(pts, radii, pairs, area, centres, offsets, ids, active, 12)
    hull = visual_hull(pts, args.scene_dir) if args.registration is None else visual_hull_colmap(pts, args.scene_dir)
    if args.extra_material is not None:
        hull[np.load(args.extra_material)] = True
    material = hull & (volume > 1e-12)
    bond = material[pairs].all(1)
    # Largest connected material component.
    idx = np.flatnonzero(material)
    remap = -np.ones(len(pts), int); remap[idx] = np.arange(len(idx))
    mp = remap[pairs[bond]]
    graph = sparse.coo_matrix((np.ones(len(mp)), (mp[:, 0], mp[:, 1])), shape=(len(idx),)*2)
    _, labels = connected_components(graph, directed=False)
    main_label = np.bincount(labels).argmax()
    cells = idx[labels == main_label]
    remap = -np.ones(len(pts), int); remap[cells] = np.arange(len(cells))
    in_main = (remap[pairs] >= 0).all(1)
    out_pairs = remap[pairs[in_main]]
    world = to_world(pts)
    volume, surface = volume*k**3, surface*k**3
    geometry = dict(points=world[cells], radii=k*radii[cells], pairs=out_pairs, areas=k**2*area[in_main],
                    contacts=to_world(centres[in_main]), masses=volume[cells], fixed=np.zeros(len(cells), bool),
                    density=density[cells], source_index=cells)
    np.savez_compressed(args.out/'cells.npz', **geometry)
    bodies = cluster_cells(geometry, min_fraction=args.min_fraction)
    np.savez_compressed(args.out/'geometry.npz', **bodies)
    # Air side of each material cell: area of its faces to non-material cells + exposed sphere part.
    to_air = np.zeros(len(pts))
    air_face = material[pairs].sum(1) == 1
    for e in (0, 1):
        np.add.at(to_air, pairs[air_face, e], k**2*area[air_face])
    deg = np.bincount(bodies['pairs'].ravel(), minlength=len(bodies['masses']))
    report_bodies = dict(bodies=len(bodies['masses']), body_bonds=len(bodies['pairs']),
                         body_volume_quantiles=np.quantile(bodies['masses'], [0, .01, .5, .99, 1]).tolist(),
                         body_neighbours_min=int(deg.min()), cells_per_body_max=int(np.bincount(bodies['body_of_cell']).max()))
    report = dict(**report_bodies, total_cells=len(pts), material_cells=int(material.sum()), kept_cells=len(cells),
                  components=int(labels.max()+1), bonds=int(len(out_pairs)),
                  material_volume=float(volume[cells].sum()), exposed_sphere_volume=float(surface[cells].sum()),
                  material_cells_touching_air_faces=int((to_air[cells] > 0).sum()),
                  bbox_min=world[cells].min(0).tolist(), bbox_max=world[cells].max(0).tolist(),
                  centroid=((world[cells]*volume[cells, None]).sum(0)/volume[cells].sum()).tolist(),
                  bonds_per_cell=float(2*len(out_pairs)/len(cells)),
                  density_quantiles_material=np.quantile(density[cells], [.05, .5, .95]).tolist(),
                  density_quantiles_air=np.quantile(density[~material], [.05, .5, .95]).tolist())
    (args.out/'geometry.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
