"""Same-model-class baseline: PAC-NeRF's MLS-MPM re-implemented in torch, run from the GT particles.

Line-by-line port of references/pac_nerf_code/lib/engine/mpm_simulator.py for the elastic case
(substep = F update with the previous C, p2g with the Kirchhoff Neo-Hookean stress
tau = mu F F^T + (lam log J - mu) I, grid update with gravity and the sticky ground at nodes
y <= 0, g2p with APIC C; quadratic B-splines; p_vol = (dx/2)^3, p_mass = rho p_vol). The GT
generator's dx and dt are not published: dx = 1 cm matches the particle count (8 per cell)
and the object's mass; dt is tried at PAC-NeRF's two defaults.

Scored exactly like the DEM: particles -> bodies of output/pacnerf_dem/<scene> (same owner
assignment), normalized RMSE of body means over frames 1-10 (train), 11-15 (held out), 1-15.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from experiments.pacnerf_dem_forward import read_ply

RHO, G = 1000., (0., -9.8, 0.)
OFFSETS = torch.tensor([(i, j, k) for i in range(3) for j in range(3) for k in range(3)])


def simulate(x, v0, E, nu, dx, dt, frames, frame_dt=1/24, device='cuda', dtype=torch.float32):
    mu, lam = E/(2*(1+nu)), E*nu/((1+nu)*(1-2*nu))
    x = torch.as_tensor(x, dtype=dtype, device=device)
    n = len(x)
    v = torch.as_tensor(v0, dtype=dtype, device=device).expand(n, 3).clone()
    C = torch.zeros(n, 3, 3, dtype=dtype, device=device)
    F = torch.eye(3, dtype=dtype, device=device).expand(n, 3, 3).clone()
    p_vol = (dx*.5)**3
    p_mass = RHO*p_vol
    inv_dx = 1/dx
    lo = torch.tensor([-80, -10, -60], device=device)          # grid index box (nodes at I dx)
    shape = torch.tensor([160, 110, 120], device=device)
    strides = torch.tensor([shape[1]*shape[2], shape[2], 1], device=device)
    offsets = OFFSETS.to(device)
    gravity = torch.tensor(G, dtype=dtype, device=device)
    eye = torch.eye(3, dtype=dtype, device=device)
    substeps = int(round(frame_dt/dt))
    out = [x.clone()]
    for f in range(frames):
        for _ in range(substeps):
            F = (eye+dt*C)@F                                        # compute_F_tmp + project_F (elastic)
            J = torch.linalg.det(F)
            tau = mu*(F@F.transpose(1, 2))+(lam*torch.log(J)-mu)[:, None, None]*eye
            affine = (-dt*p_vol*4*inv_dx**2)*tau+p_mass*C
            base = torch.floor(x*inv_dx-.5).long()
            fx = x*inv_dx-base.to(dtype)
            w = torch.stack((.5*(1.5-fx)**2, .75-(fx-1)**2, .5*(fx-.5)**2), 1)   # (n, 3 nodes, 3 dims)
            idx = base[:, None, :]+offsets[None]                                      # (n, 27, 3)
            assert bool(((idx-lo >= 0) & (idx-lo < shape)).all()), 'particle left the grid box'
            flat = ((idx-lo)*strides).sum(-1)                                         # (n, 27)
            weight = w[:, offsets[:, 0], 0]*w[:, offsets[:, 1], 1]*w[:, offsets[:, 2], 2]   # (n, 27)
            dpos = (offsets[None].to(dtype)-fx[:, None, :])*dx                         # (n, 27, 3)
            momentum = weight[..., None]*(p_mass*v[:, None, :]+(affine[:, None]@dpos[..., None])[..., 0])
            cells = int(shape.prod())
            grid_m = torch.zeros(cells, dtype=dtype, device=device).index_add_(0, flat.reshape(-1), (weight*p_mass).reshape(-1))
            grid_v = torch.zeros(cells, 3, dtype=dtype, device=device).index_add_(0, flat.reshape(-1), momentum.reshape(-1, 3))
            active = grid_m > 1e-8*dx**3
            grid_v = torch.where(active[:, None], grid_v/grid_m.clamp(min=1e-30)[:, None]+dt*gravity, torch.zeros_like(grid_v))
            node_y = (torch.arange(cells, device=device)//strides[1]) % shape[1]+lo[1]
            grid_v = torch.where((node_y.to(dtype)*dx <= 1e-6)[:, None], torch.zeros_like(grid_v), grid_v)   # sticky ground
            g_v = grid_v[flat]                                                         # (n, 27, 3)
            v = (weight[..., None]*g_v).sum(1)
            C = 4*inv_dx*(weight[..., None, None]*g_v[..., :, None]*(dpos/dx)[..., None, :]).sum(1)
            x = x+dt*v
        out.append(x.clone())
    return torch.stack(out).cpu().numpy(), substeps


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--scene', default='elastic/0')
    a.add_argument('--run', type=Path, default=Path('output/pacnerf_dem/elastic_0_f0'))
    a.add_argument('--frames', type=int, default=15)
    a.add_argument('--train-frames', type=int, default=10)
    a.add_argument('--dx', type=float, default=.01)
    a.add_argument('--dt', type=float, nargs='+', default=[1/9600, 1/4800])
    a.add_argument('--params', default='gt', help="'gt' or 'E,nu'")
    args = a.parse_args()
    gt = json.loads(Path('data/pacnerf/gt/elastic.json').read_text())[args.scene.split('/')[1]]
    E, nu = (gt['E'], gt['nu']) if args.params == 'gt' else map(float, args.params.split(','))
    sim_dir = Path('data/pacnerf/simulation_data')/args.scene
    x0 = read_ply(sim_dir/'0.ply')
    truth = np.stack([read_ply(sim_dir/f'{k}.ply') for k in range(args.frames+1)])
    cells = dict(np.load(args.run/'cells.npz')); bodies = dict(np.load(args.run/'geometry.npz'))
    _, near = cKDTree(cells['points']).query(x0, k=16)
    power = ((x0[:, None]-cells['points'][near])**2).sum(-1)-cells['radii'][near]**2
    owner = bodies['body_of_cell'][near[np.arange(len(x0)), power.argmin(1)]]
    count = np.bincount(owner, minlength=len(bodies['masses'])); has = count > 0

    def body_means(P):
        out = np.zeros((len(P), len(count), 3))
        for k in range(len(P)):
            for d in range(3):
                out[k, :, d] = np.bincount(owner, weights=P[k, :, d], minlength=len(count))
        return out/np.maximum(count, 1)[None, :, None]

    target = body_means(truth)

    def nrmse(pos, lo, hi):
        p, t = pos[lo:hi+1, has], target[lo:hi+1, has]
        return float(np.sqrt(np.mean((p-t)**2)/np.mean((t-target[:1, has])**2)))
    results = {}
    for dt in args.dt:
        start = time.perf_counter()
        P, substeps = simulate(x0, gt['v'], E, nu, args.dx, dt, args.frames)
        pos = body_means(P)
        particle = float(np.sqrt(np.mean((P[1:]-truth[1:])**2)/np.mean((truth[1:]-truth[:1])**2)))
        results[f'dt=1/{round(1/dt)}'] = dict(train=nrmse(pos, 1, args.train_frames), test=nrmse(pos, args.train_frames+1, args.frames),
                                              all=nrmse(pos, 1, args.frames), particle_all=particle, substeps=substeps,
                                              seconds=time.perf_counter()-start, centroid_y=pos[:, has, 1].mean(1).tolist())
        print(args.params, f'dt=1/{round(1/dt)}', json.dumps({k: v for k, v in results[f'dt=1/{round(1/dt)}'].items() if k != 'centroid_y'}), flush=True)
    out = args.run/'mpm_baseline'
    out.mkdir(exist_ok=True)
    (out/f"mpm_{args.params.replace(',', '_')}.json").write_text(json.dumps(dict(E=E, nu=nu, dx=args.dx, results=results,
                                                                               gt_centroid_y=target[:, has, 1].mean(1).tolist()), indent=1))


if __name__ == '__main__':
    main()
