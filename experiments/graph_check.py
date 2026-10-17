"""CUDA-graph frame replay (WarpFoam graphs=True) vs plain launches: identical results, speed."""
import time
import numpy as np
import torch
import warp as wp
from experiments.foam_dem_warp import WarpFoam, rollout
from experiments.pacnerf_dem_forward import build


def main():
    wp.init()
    g = dict(np.load('output/vid2sim_dem/backpack/geometry.npz'))
    m = build(g, 46906., .482, 'cuda', 'noslip', .25, .3, torch.float32, kinematics='exact', residual=True)
    m.set_environment(gravity=[0, 0, -9.8], ground_point=[0, 0, -.7], ground_normal=[0, 0, 1],
                      contact_radius=m.contact_radius.cpu().numpy(), ground_mode='noslip')
    th = torch.zeros(m.n, 9, device='cuda'); th[:, 6] = float(np.log(68.89))
    x0, q0, v0, w0 = m.initial_state()
    v0 = (v0+torch.tensor([0, 0, -9.8*.025], device='cuda')).contiguous()
    frames, fdt = 10, .05
    sub = int(np.ceil(fdt/m.stable_dt(th[:, :7], include_damping=False)))
    out = {}
    for graphs in (False, True):
        wf = WarpFoam(m, contact='smooth', layer_dx=.01, implicit_damping=True, graphs=graphs)
        with torch.no_grad():
            rollout(wf, th, (x0, q0, v0, w0), fdt/sub, sub, 1)            # warm-up / compile
        torch.cuda.synchronize(); t = time.perf_counter()
        with torch.no_grad():
            pos, _ = rollout(wf, th, (x0, q0, v0, w0), fdt/sub, sub, frames)
        torch.cuda.synchronize(); tf = time.perf_counter()-t
        p = th[0].clone().requires_grad_(True)
        torch.cuda.synchronize(); t = time.perf_counter()
        P, _ = rollout(wf, p.expand(m.n, 9), (x0, q0, v0, w0), fdt/sub, sub, frames, 200)
        loss = (P[1:]**2).sum(); loss.backward()
        torch.cuda.synchronize(); tg = time.perf_counter()-t
        out[graphs] = (pos, p.grad.clone())
        print('graphs=%s: %d substeps/frame, no-grad forward %.1f s, forward+backward %.1f s' % (graphs, sub, tf, tg), flush=True)
    print('positions identical: %s (max diff %.1e); gradient rel diff %.1e' % (
        bool(torch.equal(out[False][0], out[True][0])), float((out[False][0]-out[True][0]).abs().max()),
        float((out[False][1]-out[True][1]).norm()/out[False][1].norm())))


if __name__ == '__main__':
    main()
