"""Implicit pairwise damping (WarpFoam implicit_damping=True) vs explicit dashpots, backpack DEM at the
parameters the Vid2Sim fit reached (E 46906, nu .482, damping x68.9): stability at the elastic-only
time step, trajectory difference, gradient agreement, and cost."""
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
    res = {}
    for name, imp in (('explicit', False), ('implicit', True)):
        wf = WarpFoam(m, contact='smooth', layer_dx=.01, implicit_damping=imp)
        sub = int(np.ceil(fdt/m.stable_dt(th[:, :7], include_damping=not imp)))
        torch.cuda.synchronize(); t = time.perf_counter()
        with torch.no_grad():
            pos, quat = rollout(wf, th, (x0, q0, v0, w0), fdt/sub, sub, frames)
        torch.cuda.synchronize(); el = time.perf_counter()-t
        # gradient of a smooth trajectory loss w.r.t. the shared damping / stiffness logs
        p = th[0].clone().requires_grad_(True)
        P, _ = rollout(wf, p.expand(m.n, 9), (x0, q0, v0, w0), fdt/sub, sub, frames, 200)
        loss = (P[1:, :, 2]*torch.linspace(0, 1, frames, device='cuda')[:, None]).sum()+(P[1:]**2).sum()
        loss.backward()
        res[name] = (pos, p.grad.clone())
        print('%-8s substeps/frame %6d (%d colours)  forward %.1f s  finite %s  colours' % (
            name, sub, getattr(wf, 'n_colors', 0), el, bool(torch.isfinite(pos).all())), flush=True)
    pe, ge = res['explicit']; pi, gi = res['implicit']
    h = float(np.median(np.cbrt(g['masses'])))
    com = lambda P: P.mean(1)
    print('trajectory: max |x_imp - x_exp| / body size per frame %s; COM diff / body size %s' % (
        np.round(((pi-pe).norm(dim=-1).amax(-1)/h).cpu().numpy(), 2).tolist(), np.round(((com(pi)-com(pe)).norm(dim=-1)/h).cpu().numpy(), 3).tolist()))
    print('gradient cosine (implicit vs explicit) %.3f; components exp %s imp %s' % (
        float(torch.nn.functional.cosine_similarity(gi, ge, 0)), np.round(ge.cpu().numpy(), 3).tolist(), np.round(gi.cpu().numpy(), 3).tolist()))


if __name__ == '__main__':
    main()
