"""The material without the hull ghost (mirror) sites (2026-10-22, toward a surface made of the cells' own spheres): the
foam's real sites only, material = real sites inside the visual hull with bounded volume (as build_geometry.py), every
material cell = its power cell (real sites) clipped by its ball. Against the version with the ghosts:
  volume (bounded cells, exact faces) and its exposed-sphere part; surface area split into flat faces (material against
  a non-material real site) and exposed sphere; by grid sampling (power-nearest site, inside its ball): material only
  without the ghosts, its thickness (distance to the material with the ghosts) and where it is (world y up)."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.stats import qmc

from experiments.bread_material_geometry import clipped_faces, neighbors, volume_from_faces
from experiments.pacnerf_foam_geometry import visual_hull
from experiments.powersim_bread_demo import regular_candidates


def geometry(pts, radii, hull_mask):
    all_pairs, possible, active_ids = regular_candidates(pts, radii)
    active = np.zeros(len(pts), bool); active[active_ids] = True
    offs, ids = neighbors(len(pts), all_pairs)
    cand = all_pairs[possible]
    area, cen = clipped_faces(pts, radii, cand, offs, ids, 64)
    keep = area > 0
    pairs, area, cen = cand[keep], area[keep], cen[keep]
    volume, cone, _ = volume_from_faces(pts, radii, pairs, area, cen, offs, ids, active, 12)
    material = hull_mask & (volume > 1e-12)
    flat = material[pairs].sum(1) == 1
    return material, volume, cone, pairs, area, flat


def sphere_area(pts, radii, cells, exponent=12):
    """exposed sphere area of the given cells: Sobol rays on each sphere, a point is exposed when its power-nearest site
    (lifted k-d tree) is the cell's own site (the DEM's cone volume is measured from the face-centroid point, not the
    sphere centre, so 3 cone / r is not this area)"""
    smp = qmc.Sobol(2, scramble=True, seed=7).random_base2(exponent)
    z, ph = 1-2*smp[:, 0], 2*np.pi*smp[:, 1]
    U = np.column_stack((np.sqrt(1-z*z)*np.cos(ph), np.sqrt(1-z*z)*np.sin(ph), z))
    rmax = radii.max()
    lift = cKDTree(np.column_stack((pts, np.sqrt(rmax**2-radii**2))))
    total = 0.
    for i in cells:
        X = pts[i]+radii[i]*U
        _, c = lift.query(np.column_stack((X, np.zeros(len(X)))))
        total += 4*np.pi*radii[i]**2*(c == i).mean()
    return total


def sample(pts, radii, material, lo, hi, spacing):
    rmax = radii.max()
    lift = cKDTree(np.column_stack((pts, np.sqrt(rmax**2-radii**2))))
    axes = [np.arange(lo[k]+spacing/2, hi[k], spacing) for k in range(3)]
    G = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
    out = np.zeros(len(G), bool)
    for s in range(0, len(G), 500000):
        P = G[s:s+500000]
        _, c = lift.query(np.column_stack((P, np.zeros(len(P)))))
        out[s:s+500000] = material[c] & (((P-pts[c])**2).sum(1) <= radii[c]**2)
    return G, out


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--run', type=Path, default=Path('output/pacnerf_elastic0_f0_transparent_cutft'))
    a.add_argument('--scene-dir', type=Path, default=Path('/home/fangsuo/py/dynamic-powerfoam/data/pacnerf_powerfoam/elastic_0_f0'))
    a.add_argument('--spacing', type=float, default=.003)
    args = a.parse_args()
    st = torch.load(args.run/'model.pt', map_location='cpu', weights_only=True)
    pts = st['points'].double().numpy(); radii = torch.nn.functional.softplus(st['radii'].double(), beta=100).numpy()
    n_real = int(st['num_sites'])
    offset = np.array(json.loads((args.scene_dir/'conversion.json').read_text())['offset'])
    hull = visual_hull(pts, args.scene_dir); hull[n_real:] = False
    res = {}
    for tag, n in (('with ghosts', len(pts)), ('without ghosts', n_real)):
        material, volume, cone, pairs, area, flat = geometry(pts[:n], radii[:n], hull[:n])
        sph_area = sphere_area(pts[:n], radii[:n], np.flatnonzero(material & (cone > 0)))
        res[tag] = (material, volume)
        print('[%s] material cells %d, volume %.5f m^3 (truth 0.0316), exposed-sphere cone (DEM, from the face-centroid point) %.2e m^3; surface: flat faces %.4f m^2, exposed sphere %.4f m^2; '
              'sphere-bounded cells %d' % (tag, material.sum(), volume[material].sum(), cone[material].sum(), area[flat].sum(), sph_area,
                                           int(((cone > 0) & material).sum())), flush=True)
    m1, _ = res['with ghosts']; m0, _ = res['without ghosts']
    print('material cells only without ghosts %d, only with ghosts %d' % (int((m0 & ~m1[:n_real]).sum()), int((m1[:n_real] & ~m0).sum())), flush=True)
    sel = np.flatnonzero(m0)
    lo = (pts[sel]-radii[sel, None]).min(0); hi = (pts[sel]+radii[sel, None]).max(0)
    G, in0 = sample(pts[:n_real], radii[:n_real], m0, lo, hi, args.spacing)
    m1f = np.zeros(len(pts), bool); m1f[:len(m1)] = m1
    _, in1 = sample(pts, radii, m1f, lo, hi, args.spacing)
    dv = args.spacing**3
    extra = in0 & ~in1
    print('grid %.1f mm: volume with ghosts %.5f, without %.5f; only without ghosts %.5f m^3 (%.1f %% of the volume with ghosts), only with ghosts %.2e' % (
        1e3*args.spacing, in1.sum()*dv, in0.sum()*dv, extra.sum()*dv, 100*extra.sum()/in1.sum(), (in1 & ~in0).sum()*dv), flush=True)
    if extra.any():
        d, _ = cKDTree(G[in1]).query(G[extra])
        W = G[extra]-offset
        Wm = G[in1]-offset
        print('extra material: thickness (distance to the material with ghosts) median %.1f mm, 90 %% %.1f, max %.1f mm' % (
            1e3*np.median(d), 1e3*np.quantile(d, .9), 1e3*d.max()), flush=True)
        y0, y1 = Wm[:, 1].min(), Wm[:, 1].max()
        for name, mk in (('below the object (world y < its lowest point)', W[:, 1] < y0), ('within its height', (W[:, 1] >= y0) & (W[:, 1] <= y1)),
                         ('above it', W[:, 1] > y1)):
            print('   %-46s %.5f m^3, thickness max %.1f mm' % (name, mk.sum()*dv, 1e3*d[mk].max() if mk.any() else 0), flush=True)


if __name__ == '__main__':
    main()
