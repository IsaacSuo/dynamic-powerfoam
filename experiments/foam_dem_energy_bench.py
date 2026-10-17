"""Training-speed benchmark of EnergyFoam rollouts (closed-form forces, checkpointed autograd).

1. Gradient check through a drop with sticky contact (bar, 400 steps, CPU): adjoint vs central
   finite difference along a random direction.
2. Forward + backward wall time, CPU bar (16 cells, 400 steps).
3. GPU lattice of m^3 unit cubes (all free, gravity, sticky ground): forward + backward time and
   peak memory for a PAC-NeRF-sized rollout (13 frames x 200 substeps).
"""
import json
import time

import numpy as np
import torch

from experiments.foam_dem_energy import EnergyFoam


def lattice(m):
    idx = np.stack(np.meshgrid(*(np.arange(m),)*3, indexing='ij'), -1).reshape(-1, 3)
    points = idx.astype(float)
    key = {tuple(p): k for k, p in enumerate(idx)}
    pairs = [(k, key[tuple(p+e)]) for k, p in enumerate(idx) for e in np.eye(3, dtype=int) if tuple(p+e) in key]
    pairs = np.array(pairs)
    return dict(points=points, radii=np.full(len(points), np.sqrt(1/2.4)), pairs=pairs,
                areas=np.ones(len(pairs)), contacts=.5*(points[pairs[:, 0]]+points[pairs[:, 1]]),
                masses=np.ones(len(points)), fixed=np.zeros(len(points), bool))


def drop_model(geometry, device, ground):
    model = EnergyFoam(geometry, device=device)
    model.fixed[:] = False
    model.set_environment(gravity=[0, -9.8, 0], ground_point=[0, ground, 0], ground_normal=[0, 1, 0],
                          contact_radius=np.full(model.n, .5))
    return model


def loss_fn(model, theta, dt, substeps, frames, target, checkpoint=True):
    x0, q0, _, _ = model.initial_state()
    v0 = torch.tensor([.5, -2., 0.], dtype=torch.float64, device=model.device).expand(model.n, 3).clone()
    positions, _ = model.rollout(theta, (x0, q0, v0, torch.zeros_like(x0)), dt, substeps, frames, checkpoint)
    return ((positions[1:]-target)**2).mean()


def main():
    out = {}
    g = dict(np.load('output/foam_fem/cube_bar_reference_q018/patch.npz'))
    model = drop_model(g, 'cpu', -1.2)
    rng = np.random.default_rng(3)
    target = model.X+torch.tensor([.2, -.5, 0.], dtype=torch.float64)
    theta = torch.tensor(rng.uniform(-.5, .5, (model.n, 5)), requires_grad=True)
    start = time.perf_counter()
    L = loss_fn(model, theta, 1e-3, 100, 4, target)
    L.backward()
    out['cpu_bar_400_steps_forward_backward_s'] = time.perf_counter()-start
    d = torch.tensor(rng.normal(size=(model.n, 5)))
    with torch.no_grad():
        fd = (loss_fn(model, theta+1e-5*d, 1e-3, 100, 4, target)-loss_fn(model, theta-1e-5*d, 1e-3, 100, 4, target))/2e-5
    out['gradient_relative_error'] = float(abs(fd-(theta.grad*d).sum())/abs(fd))
    if torch.cuda.is_available():
        for m in (8, 16):
            geo = lattice(m)
            model = drop_model(geo, 'cuda', -1.)
            theta = torch.zeros(model.n, 5, dtype=torch.float64, device='cuda', requires_grad=True)
            target = model.X+.1
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            with torch.no_grad():
                loss_fn(model, theta.detach(), 1e-4, 200, 13, target)
            torch.cuda.synchronize(); forward = time.perf_counter()-start
            start = time.perf_counter()
            L = loss_fn(model, theta, 1e-4, 200, 13, target)
            L.backward()
            torch.cuda.synchronize()
            out[f'gpu_lattice{m}^3_{model.n}cells_2600steps'] = dict(
                forward_only_s=forward, forward_backward_s=time.perf_counter()-start,
                peak_memory_GB=torch.cuda.max_memory_allocated()/1e9, finite_gradient=bool(torch.isfinite(theta.grad).all()))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
