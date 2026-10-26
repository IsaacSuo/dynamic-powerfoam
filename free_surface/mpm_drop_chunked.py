"""experiments/mpm_drop_energy.py (PAC-NeRF's MPM, the torch port) with the particles processed in chunks (2026-10-23):
the same algorithm and records, the particle-to-grid and grid-to-particle transfers done --chunk particles at a time so
that fine grids fit in memory (at dx 5 mm, 2.3 M particles: the per-particle (27, 3) temporaries of the whole set fill
12 GB); prints each frame's progress. Particles from --particles (PAC-NeRF world), the lowest at --height above the
ground y = 0, at rest; ground-truth E and nu; ground 'slip' (frictionless) or 'sticky'.
  python free_surface/mpm_drop_chunked.py --dx 0.005 --particles output/round01_pruned_exact/mpm_particles_dx005.npz --height 0.02025 --out output/round01_pruned_exact/mpm_drop_slip_dx005"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.pacnerf_mpm_baseline import OFFSETS, RHO, G


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--root', type=Path, default=Path('/home/fangsuo/py/dynamic-powerfoam'))
    a.add_argument('--scene', default='elastic/0')
    a.add_argument('--particles', type=Path, required=True)
    a.add_argument('--height', type=float, default=.02)
    a.add_argument('--frames', type=int, default=12)
    a.add_argument('--dx', type=float, default=.01)
    a.add_argument('--dt', type=float, default=1/4800)
    a.add_argument('--every', type=int, default=20)
    a.add_argument('--ground', choices=('sticky', 'slip', 'none'), default='slip')
    a.add_argument('--gravity', type=float, default=9.8)
    a.add_argument('--squeeze', type=float, default=0., help='initial velocity v_y = -squeeze (y - centre of mass y), 1/s (free vibration)')
    a.add_argument('--chunk', type=int, default=400_000)
    a.add_argument('--out', type=Path, required=True)
    args = a.parse_args()
    gt = json.loads((args.root/'data/pacnerf/gt/elastic.json').read_text())[args.scene.split('/')[1]]
    E, nu = gt['E'], gt['nu']
    mu, lam = E/(2*(1+nu)), E*nu/((1+nu)*(1-2*nu))
    dev, dtype = 'cuda', torch.float32
    x0 = np.load(args.particles)['x']
    x0 = x0-np.array([0., x0[:, 1].min()-args.height, 0.])
    dx, dt = args.dx, args.dt
    x = torch.as_tensor(x0, dtype=dtype, device=dev)
    n = len(x)
    v = torch.zeros(n, 3, dtype=dtype, device=dev)
    v[:, 1] = -args.squeeze*(x[:, 1]-x[:, 1].mean())
    C = torch.zeros(n, 3, 3, dtype=dtype, device=dev)
    F = torch.eye(3, dtype=dtype, device=dev).expand(n, 3, 3).clone()
    p_vol = (dx*.5)**3
    p_mass = RHO*p_vol
    M = p_mass*n
    inv_dx = 1/dx
    lo = torch.tensor([-80, -10, -60], device=dev)
    shape = torch.tensor([160, 110, 120], device=dev)
    strides = torch.tensor([shape[1]*shape[2], shape[2], 1], device=dev)
    offsets = OFFSETS.to(dev)
    gravity = torch.tensor(G, dtype=dtype, device=dev)*(args.gravity/9.8)
    eye = torch.eye(3, dtype=dtype, device=dev)
    cells = int(shape.prod())
    node_y = (torch.arange(cells, device=dev)//strides[1]) % shape[1]+lo[1]
    sticky = (node_y.to(dtype)*dx <= 1e-6)[:, None]
    substeps = int(round(1/24/dt))
    chunks = [slice(k, min(k+args.chunk, n)) for k in range(0, n, args.chunk)]
    rows = []
    lost = [0.]                                                                     # kinetic energy removed at the ground

    def stencil(xs):
        base = torch.floor(xs*inv_dx-.5).long()
        fx = xs*inv_dx-base.to(dtype)
        w = torch.stack((.5*(1.5-fx)**2, .75-(fx-1)**2, .5*(fx-.5)**2), 1)
        idx = base[:, None, :]+offsets[None]
        assert bool(((idx-lo >= 0) & (idx-lo < shape)).all()), 'particle left the grid box'
        flat = ((idx-lo)*strides).sum(-1)
        weight = w[:, offsets[:, 0], 0]*w[:, offsets[:, 1], 1]*w[:, offsets[:, 2], 2]
        dpos = (offsets[None].to(dtype)-fx[:, None, :])*dx
        return flat, weight, dpos

    def record(t):
        el, ke, ys = 0., 0., []
        vc = v.double().mean(0)
        for sl in chunks:
            Fd = F[sl].double(); J = torch.linalg.det(Fd)
            el += float((mu/2*((Fd*Fd).sum((1, 2))-3)-mu*torch.log(J)+lam/2*torch.log(J)**2).sum())
            ke += float((.5*(v[sl].double()**2).sum(1)).sum())
        y = x[:, 1].double()
        q = torch.quantile(y[::7], torch.tensor([.01, .99], dtype=torch.float64, device=dev))
        rows.append(dict(t=t, com_y=float(y.mean()), vy_com=float(vc[1]), ke_total=ke/n, ke_com=float(.5*(vc@vc)), elastic=p_vol*el/M,
                         gravity=float(args.gravity*y.mean()), extent=float(q[1]-q[0]), lowest=float(y.min()), ground_loss=lost[0]/M))
    start = time.perf_counter()
    record(0.)
    grid_m = torch.zeros(cells, dtype=dtype, device=dev)
    grid_v = torch.zeros(cells, 3, dtype=dtype, device=dev)
    for f in range(args.frames):
        for s in range(substeps):
            grid_m.zero_(); grid_v.zero_()
            for sl in chunks:                                                        # particles to grid
                F[sl] = (eye+dt*C[sl])@F[sl]
                J = torch.linalg.det(F[sl])
                tau = mu*(F[sl]@F[sl].transpose(1, 2))+(lam*torch.log(J)-mu)[:, None, None]*eye
                affine = (-dt*p_vol*4*inv_dx**2)*tau+p_mass*C[sl]
                flat, weight, dpos = stencil(x[sl])
                momentum = weight[..., None]*(p_mass*v[sl][:, None, :]+(affine[:, None]@dpos[..., None])[..., 0])
                grid_m.index_add_(0, flat.reshape(-1), (weight*p_mass).reshape(-1))
                grid_v.index_add_(0, flat.reshape(-1), momentum.reshape(-1, 3))
            active = grid_m > 1e-8*dx**3
            gv = torch.where(active[:, None], grid_v/grid_m.clamp(min=1e-30)[:, None]+dt*gravity, torch.zeros_like(grid_v))
            if args.ground == 'sticky':
                lost[0] += float((.5*grid_m*(gv*gv).sum(1)*sticky[:, 0]).sum())
                gv = torch.where(sticky, torch.zeros_like(gv), gv)
            elif args.ground == 'slip':                                              # the grid kinetic energy the clamp takes
                lost[0] += float((.5*grid_m*gv[:, 1].clamp(max=0)**2*sticky[:, 0]).sum())
                gv = torch.where(sticky, torch.stack((gv[:, 0], gv[:, 1].clamp(min=0), gv[:, 2]), 1), gv)
            for sl in chunks:                                                        # grid to particles
                flat, weight, dpos = stencil(x[sl])
                g_v = gv[flat]
                v[sl] = (weight[..., None]*g_v).sum(1)
                C[sl] = 4*inv_dx*(weight[..., None, None]*g_v[..., :, None]*(dpos/dx)[..., None, :]).sum(1)
                x[sl] = x[sl]+dt*v[sl]
            if (s+1) % args.every == 0:
                record(f/24+(s+1)*dt)
        print('frame %d / %d: %.0f s, centre of mass y %.4f m, peak memory %.1f GB' % (
            f+1, args.frames, time.perf_counter()-start, rows[-1]['com_y'], torch.cuda.max_memory_allocated()/2**30), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    (Path(str(args.out)+'.json')).write_text(json.dumps(dict(E=E, nu=nu, dx=dx, dt=dt, height=args.height, v0=[0., 0., 0.], particles=n,
                                                              seconds=time.perf_counter()-start, rows=rows), indent=1))
    print('mpm drop, dx %g, %d particles: %d substeps/frame, %.0f s' % (dx, n, substeps, time.perf_counter()-start), flush=True)


if __name__ == '__main__':
    main()
