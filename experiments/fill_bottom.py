"""Fill the unseen bottom of a Spring-Gaus static reconstruction from the dynamic free-flight visual hull.

The static capture never sees the underside (the object stands on a table), so the material visual hull ends in a flat
cut; the drop video sees the bottom in flight; the interior is never seen, and the reconstruction is a dome shell over an open
cavity with voids inside. springgaus_train.py carve gives the flight visual hull on a rest-frame grid (3 cameras x the
pre-impact frames). Here, per vertical column of that grid (along gravity), hull points below the topmost point already
covered by material and not covered themselves are missing material (the object is solid); columns without material are left alone, and the hull is
bounded by its axisymmetric profile (median radius over azimuth per depth), since 3 cameras leave it too wide where none
looks edge-on. The gap is filled with new Power Foam cells on a regular grid
at the typical material cell size (air cells whose centre falls in the gap are removed); the new cells copy orientation
from the nearest material cell, get the median material density and one flat colour (--colour; the underside is never seen in
the static scan). Writes a new run directory
(model.pt with a rebuilt adjacency, config.yaml) and extra_material.npy (indices of the new cells, forced material in
pacnerf_foam_geometry --extra-material).
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from experiments.foam_dynamic_render import load
from experiments.pacnerf_foam_geometry import foam_to_dynamic

NAMES = ('points', 'density', 'radii', 'quaternions', 'texel_sites', 'texel_sv_axis', 'texel_sv_rgb', 'texel_height')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, default=Path('output/springgaus_burger_spacing8k'))
    p.add_argument('--scene-dir', type=Path, default=Path('data/springgaus_powerfoam/burger'))
    p.add_argument('--registration', type=Path,
                   default=Path('data/springgaus_powerfoam/burger_dynamic/register_calib/initial_state_upright_nospin.json'))
    p.add_argument('--dem', type=Path, default=Path('output/springgaus_dem/burger_spacing8k_mf0.1'))
    p.add_argument('--carve', type=Path, default=Path('output/springgaus_dem/burger_spacing8k_mf0.1/carve/carve.npz'))
    p.add_argument('--out-run', type=Path, default=Path('output/springgaus_burger_spacing8k_filled'))
    p.add_argument('--colour', default='#B26A14', help='flat colour of the new cells (the underside is never seen)')
    args = p.parse_args()
    A, b, k = foam_to_dynamic(args.scene_dir, args.registration)
    foam, _ = load(args.run)
    sp = lambda t: torch.nn.functional.softplus(t.detach().double(), beta=100).cpu().numpy()
    pf = foam.points.detach().double().cpu().numpy()
    pw, rw = pf@A.T+b, k*sp(foam.radii)
    cells = np.load(args.dem/'cells.npz')['source_index']
    material = np.zeros(len(pf), bool); material[cells] = True
    cv = np.load(args.carve)
    X, hull, step, Bm, c0 = cv['points'], cv['hull'], float(cv['step']), cv['frame'], cv['centre']
    # which grid points are already material: owner = min power distance among the 32 nearest sites, inside its sphere
    tree = cKDTree(pw)
    _, nn = tree.query(X, 32)
    pd = ((X[:, None]-pw[nn])**2).sum(-1)-rw[nn]**2
    own = nn[np.arange(len(X)), pd.argmin(1)]
    covered = material[own] & (np.linalg.norm(X-pw[own], axis=1) <= rw[own])
    # per gravity column: hull points below the topmost covered point (the static reconstruction is a dome shell over an
    # open cavity, with voids inside: no camera saw the interior or the underside, so no cells were placed there)
    G = (X-c0)@Bm
    key = np.round(G[:, :2]/step).astype(np.int64)
    kid = (key[:, 0]-key[:, 0].min())*(key[:, 1].max()-key[:, 1].min()+1)+(key[:, 1]-key[:, 1].min())
    low = np.full(kid.max()+1, np.inf)
    np.minimum.at(low, kid[covered], G[covered, 2])
    # with 3 cameras the hull overshoots where no camera looks edge-on; bound it by its axisymmetric profile: per depth
    # slice, the median over 36 azimuth sectors (about the vertical axis through the centroid) of the hull's radius
    rho = np.hypot(G[:, 0], G[:, 1])
    sector = (((np.arctan2(G[:, 1], G[:, 0])+np.pi)/(2*np.pi)*36).astype(int)) % 36
    zi = np.round(G[:, 2]/step).astype(int); zi -= zi.min()
    Rh = np.zeros((zi.max()+1, 36)); np.maximum.at(Rh, (zi[hull], sector[hull]), rho[hull])
    r_sym = np.median(Rh, 1)
    gap = hull & (rho <= r_sym[zi]) & ~covered & (G[:, 2] > low[kid]) & np.isfinite(low[kid])
    gap_vol = gap.sum()*step**3
    # new cells: the typical material cell size and support radius
    h = float(np.median(np.load(args.dem/'cells.npz')['masses'])**(1/3))     # typical material cell size
    r_new = float(np.median(rw[cells]))
    # new sites: regular grid of spacing h over the gap (grid points snapped to the nearest gap voxel)
    gp = X[gap]
    Gg = (gp-c0)@Bm
    cell_key = np.floor(Gg/h).astype(np.int64)
    _, first, inv = np.unique(cell_key, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    centre = np.stack([np.bincount(inv, weights=gp[:, d]) for d in range(3)], 1)/np.bincount(inv)[:, None]
    count = np.bincount(inv)
    new = centre[count*step**3 >= .25*h**3]                 # drop slivers of the gap (< 1/4 of a cell)
    new = new+np.random.default_rng(0).uniform(-.15*h, .15*h, new.shape)   # break the lattice (degenerate power / Delaunay)
    # air cells inside the gap are removed
    gap_tree = cKDTree(gp)
    d_air, _ = gap_tree.query(pw)
    drop = ~material & (d_air <= step)
    _, src = cKDTree(pw[cells]).query(new)
    src = cells[src]
    keep = np.flatnonzero(~drop)
    flat_rgb = torch.tensor([int(args.colour.lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4)])-.5   # colour = stored + 0.5
    data = {}
    dens_raw = float(foam.density.detach().double()[torch.as_tensor(cells)].median())
    r_raw = float(np.log(np.expm1(100*r_new/k))/100)
    for name in NAMES:
        t = getattr(foam, name).detach()
        if name == 'points':
            add = torch.tensor((new-b)@np.linalg.inv(A).T, dtype=t.dtype, device=t.device)
        elif name == 'density':
            add = torch.full((len(new),), dens_raw, dtype=t.dtype, device=t.device)
        elif name == 'radii':
            add = torch.full((len(new),), r_raw, dtype=t.dtype, device=t.device)
        elif name == 'texel_sv_rgb':                          # one flat, view-independent colour (--colour)
            add = flat_rgb.to(t).repeat(t[0].numel()//3).view(t.shape[1:]).expand(len(new), *t.shape[1:]).clone()   # (all lobes equal)
        else:
            add = t[torch.as_tensor(src, device=t.device)].clone()
        data[name] = torch.cat((t[torch.as_tensor(keep, device=t.device)], add))
    for name in NAMES:
        setattr(foam, name, torch.nn.Parameter(data[name].clone()))
    foam.rebuild_adjacency()
    args.out_run.mkdir(parents=True, exist_ok=True)
    state = {name: getattr(foam, name).detach().cpu() for name in NAMES}
    state['adjacency'] = foam.adjacency.cpu(); state['adjacency_offsets'] = foam.adjacency_offsets.cpu()
    torch.save(state, args.out_run/'model.pt')
    shutil.copy(args.run/'config.yaml', args.out_run/'config.yaml')
    extra = np.arange(len(keep), len(keep)+len(new))
    np.save(args.out_run/'extra_material.npy', extra)
    report = dict(grid_points=int(len(X)), hull_points=int(hull.sum()), covered_by_material=int(covered.sum()),
                  gap_points=int(gap.sum()), gap_volume_cm3=1e6*gap_vol, material_volume_cm3=1e6*float(np.load(args.dem/'geometry.npz')['masses'].sum()),
                  new_cell_size_mm=1e3*h, new_radius_mm=1e3*r_new, new_cells=int(len(new)), removed_air_cells=int(drop.sum()),
                  cells_before=int(len(pf)), cells_after=int(len(keep)+len(new)), flat_colour=(0.5+flat_rgb).clamp(0, 1).tolist())
    (args.out_run/'fill_report.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
