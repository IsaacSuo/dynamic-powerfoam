"""Continuum ground truth for a Vid2Sim object: MLS-MPM (the torch port of PAC-NeRF's simulator, verified bitwise
against its data) on the DEM's own material region, same material (E, nu from gt_phys_params.yaml), same drop
(floor z = -0.7, gravity -z, 24 frames at 0.05 s, initial velocity g dt/2 like the DEM), frictionless floor.
Particles: lattice of spacing dx/2 inside the material power cells of the static foam (owner by power distance).
MPM frame (y up, floor y = 0): (x, y, z)_mpm = (x, z + 0.7, -y)_world. Output: per-frame body targets (mean of the
particles owned by each DEM body at frame 0) in data/vid2sim_gt_mpm/<obj>/targets.npz, plus a particle subset."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from experiments.springgaus_mpm_drop import simulate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--obj', default='sofa')
    p.add_argument('--dx', type=float, default=.02)
    p.add_argument('--dt', type=float, default=2.5e-4)
    p.add_argument('--frames', type=int, default=23)
    p.add_argument('--bc', default='separate', choices=('separate', 'sticky'))
    p.add_argument('--out', type=Path, default=Path('data/vid2sim_gt_mpm'))
    a = p.parse_args()
    import yaml
    gt = yaml.safe_load(open(f'data/vid2sim/ext/{a.obj}/gt_phys_params.yaml')); E, nu = float(gt['yms']), float(gt['prs'])
    dem = Path(f'output/vid2sim_dem/{a.obj}')
    g = dict(np.load(dem/'geometry.npz')); cells = dict(np.load(dem/'cells.npz'))
    offset = np.array(json.loads(Path(f'data/vid2sim_powerfoam/{a.obj}_f0/conversion.json').read_text())['offset'])
    st = torch.load(f'output/vid2sim_{a.obj}_spacing8k/model.pt', map_location='cpu', weights_only=True)
    sites = st['points'].double().numpy()-offset
    radii = torch.nn.functional.softplus(st['radii'].double(), beta=100).numpy()
    body_of_site = -np.ones(len(sites), int); body_of_site[cells['source_index']] = g['body_of_cell']
    # particles: lattice inside the material cells
    h = a.dx/2
    lo, hi = g['points'].min(0)-.05, g['points'].max(0)+.05
    axes = [np.arange(lo[k]+h/2, hi[k], h) for k in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
    _, nn = cKDTree(sites).query(grid, k=8)
    power = ((grid[:, None]-sites[nn])**2).sum(-1)-radii[nn]**2
    best = power.argmin(1)
    owner = nn[np.arange(len(grid)), best]
    inside = (power[np.arange(len(grid)), best] <= 0) & (body_of_site[owner] >= 0)
    X = grid[inside]; body = body_of_site[owner[inside]]
    n = len(g['points'])
    count = np.bincount(body, minlength=n)
    print('object %s: E %.0f nu %.3f | particles %d (spacing %.0f mm), bodies with particles %d / %d, median particles per body %d' % (
        a.obj, E, nu, len(X), 1000*h, int((count > 0).sum()), n, int(np.median(count[count > 0]))), flush=True)
    # MPM frame: y up, floor at y = 0
    to_mpm = lambda q: np.stack((q[..., 0], q[..., 2]+.7, -q[..., 1]), -1)
    from_mpm = lambda q: np.stack((q[..., 0], -q[..., 2], q[..., 1]-.7), -1)
    x0 = to_mpm(X)
    v0 = np.array([0., -9.8*.05/2, 0.])
    sample_every = int(round(.05/a.dt))
    steps = a.frames*sample_every
    positions = []
    t0 = time.perf_counter()
    rows = simulate(x0, v0, E, nu, a.dx, a.dt, steps, sample_every, a.bc, positions=positions, margin=.4)
    print('MPM %d steps in %.0f s; min J %.3f; lowest particle per frame: %s' % (steps, time.perf_counter()-t0, min(r['min_J'] for r in rows),
                                                                               np.round([from_mpm(pp)[:, 2].min() for pp in positions], 3).tolist()), flush=True)
    traj = np.array([from_mpm(pp) for pp in positions])                      # (frames+1, P, 3)
    targets = np.zeros((len(traj), n, 3))
    for t in range(len(traj)):
        s = np.zeros((n, 3)); np.add.at(s, body, traj[t]); targets[t] = s/np.maximum(count, 1)[:, None]
    # bodies without particles: nearest body with particles (rest distance), same motion
    empty = count == 0
    if empty.any():
        has = np.flatnonzero(~empty); _, j = cKDTree(g['points'][has]).query(g['points'][empty]); targets[:, empty] = targets[:, has[j]]
    out = a.out/a.obj; out.mkdir(parents=True, exist_ok=True)
    sub = np.random.default_rng(0).choice(len(X), min(20000, len(X)), replace=False)
    np.savez_compressed(out/'targets.npz', targets=targets.astype(np.float32), count=count, particles=traj[:, sub].astype(np.float32),
                        body_of_particle_subset=body[sub], E=E, nu=nu, dx=a.dx, dt=a.dt, bc=a.bc)
    (out/'drop.json').write_text(json.dumps(dict(rows=rows, E=E, nu=nu, particles=int(len(X)), dx=a.dx, dt=a.dt, bc=a.bc), default=float))
    print('targets: body motion range %.3f m, body-centre offset from particle mean at frame 0: %.1f mm' % (
        float(np.abs(targets[-1]-targets[0]).max()), 1000*float(np.linalg.norm(targets[0][~empty]-g['points'][~empty], axis=1).mean())), flush=True)


if __name__ == '__main__':
    main()
