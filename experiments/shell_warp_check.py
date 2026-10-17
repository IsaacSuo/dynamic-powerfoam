"""Warp shell path vs the torch EnergyFoam (BOND_BENDING.md: bond-direction residual, enhanced strain, thin closure)
on the sofa DEM: elastic forces / torques of one step at a perturbed state (float32 vs float64), one frame of
rollout, the gradient of a short rollout, and the time per step."""
import time
import numpy as np
import torch
import warp as wp

from experiments.foam_dem_warp import WarpFoam, rollout as warp_rollout
from experiments.pacnerf_dem_forward import build


def main():
    wp.init()
    g = dict(np.load('output/vid2sim_dem/sofa/geometry.npz'))
    E, nu = 18615., .4135
    dev = 'cuda'
    gravity, ground_p, ground_n = [0., 0., -9.8], [0., 0., -.7], [0., 0., 1.]
    for shell in (False, True):
        m64 = build(g, E, nu, dev, 'noslip', .25, .3, torch.float64, inertia_scale=1., kinematics='exact', residual=True, shell=shell)
        m64.set_environment(gravity=gravity, ground_point=ground_p, ground_normal=ground_n, contact_radius=m64.contact_radius.cpu().numpy(), ground_mode='noslip')
        m32 = build(g, E, nu, dev, 'noslip', .25, .3, torch.float32, inertia_scale=1., kinematics='exact', residual=True, shell=shell)
        m32.set_environment(gravity=gravity, ground_point=ground_p, ground_normal=ground_n, contact_radius=m32.contact_radius.cpu().numpy(), ground_mode='noslip')
        wf = WarpFoam(m32, contact='smooth', layer_dx=.01, implicit_damping=True, graphs=False)
        print('shell %s: thin %d, enhanced %d, residual %s' % (shell, int(m64.thin.sum()) if shell else 0, int(m64.eas_on.sum()) if shell else 0, getattr(m64, 'residual_mode', '-')))
        x0, q0, v0, w0 = m64.initial_state()
        torch.manual_seed(0)
        x = x0+.003*torch.randn_like(x0); q = q0+.03*torch.randn_like(q0); q = q/q.norm(dim=-1, keepdim=True)
        th = torch.zeros(m64.n, 9, dtype=torch.float64, device=dev)
        f64, t64 = m64.forces(th[:, :7], x, q)
        # one Warp step with zero velocity: buf['f'], buf['tau'] are the elastic forces
        dtypes = (wp.vec3, wp.vec4, wp.vec3, wp.vec3)
        s = tuple(wp.from_torch(t.float().contiguous().clone(), dtype=d) for t, d in zip((x, q, torch.zeros_like(x), torch.zeros_like(x)), dtypes))
        other = tuple(wp.zeros(m64.n, dtype=d, device='cuda') for d in dtypes)
        buf = wf._buffers(False)
        for b_ in buf.values():
            b_.zero_()
        wf.step(wp.from_torch(th.float().contiguous()), s, other, buf, 1e-5)
        wp.synchronize()
        f32 = wp.to_torch(buf['f']).double(); t32 = wp.to_torch(buf['tau']).double()
        print('   forces: rel err %.2e, torques: rel err %.2e' % (float((f32-f64).norm()/f64.norm()), float((t32-t64).norm()/t64.norm())))
        # time per step
        t0 = time.perf_counter()
        for _ in range(200):
            for b_ in buf.values():
                b_.zero_()
            wf.step(wp.from_torch(th.float().contiguous()), s, other, buf, 1e-5)
        wp.synchronize()
        print('   %.2f ms per step' % (1000*(time.perf_counter()-t0)/200))
        sub = int(np.ceil(.05/m64.stable_dt(th[:, :7], include_damping=False)))
        print('   substeps per frame %d' % sub)
        # rollout gradient (2 frames) w.r.t. theta through the autograd function
        thg = torch.zeros(m32.n, 9, dtype=torch.float32, device=dev, requires_grad=True)
        v0f = (v0+torch.tensor(gravity, dtype=torch.float32, device=dev)*.05/2).float().contiguous()
        pos, quat = warp_rollout(wf, thg, (x0.float().contiguous(), q0.float().contiguous(), v0f, w0.float().contiguous()), .05/sub, sub, 2, 100)
        loss = (pos[-1]**2).sum()
        loss.backward()
        print('   rollout 2 frames ok, grad finite %s, |grad| %.3g' % (bool(torch.isfinite(thg.grad).all()), float(thg.grad.norm())))


if __name__ == '__main__':
    main()
