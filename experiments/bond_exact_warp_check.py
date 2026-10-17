"""Warp bond_kernel_exact_det vs the torch reference (kinematics='exact'), float32 on CUDA:
(1) total force / torque at random large-rotation states (one step's buffers);
(2) a short rollout from the burger initial state onto the table (hard no-slip ground in both).
"""
import json
from pathlib import Path

import numpy as np
import torch
import warp as wp

from experiments.bond_exact_check import random_quat, rel
from experiments.foam_dem_warp import WarpFoam, rollout as warp_rollout
from experiments.pacnerf_dem_forward import build


def main():
    wp.init()
    g = dict(np.load('output/springgaus_dem/burger_spacing8k_mf0.1/geometry.npz'))
    dev, f32 = 'cuda', torch.float32
    gen = torch.Generator().manual_seed(1)
    for kin in ('exact', 'small'):
        m = build(g, 2e4, .3, dev, 'noslip', .25, 0., f32, kinematics=kin)
        m.set_environment(gravity=[0, 0, 0], ground_point=[0, 0, 1e3], ground_normal=[0, 0, -1],
                          contact_radius=m.contact_radius.cpu().numpy(), ground_mode='noslip')   # ground out of reach
        wf = WarpFoam(m)
        theta = (.2*torch.randn(m.n, 7, generator=gen)).to(dev)
        h = float(np.median(np.cbrt(g['masses'])))
        for max_deg in (10., 90., 170.):
            q = random_quat(m.n, np.radians(max_deg), gen).float().to(dev)
            x = (m.X+.3*h*torch.randn(m.X.shape, generator=gen).to(dev)).contiguous()
            f_t, tau_t = m.forces(theta, x, q)
            dtypes = (wp.vec3, wp.vec4, wp.vec3, wp.vec3)
            z = torch.zeros_like(x)
            s = tuple(wp.from_torch(t.contiguous(), dtype=d) for t, d in zip((x, q, z, z), dtypes))
            s1 = tuple(wp.zeros(m.n, dtype=d, device='cuda') for d in dtypes)
            buf = wf._buffers(False)
            wf.step(wp.from_torch(theta.contiguous()), s, s1, buf, 0.)
            f_w, tau_w = wp.to_torch(buf['f']), wp.to_torch(buf['tau'])
            print('[%s] rotations up to %3.0f deg: warp vs torch force rel err %.1e, torque rel err %.1e' % (
                kin, max_deg, rel(f_w, f_t), rel(tau_w, tau_t)), flush=True)
    # (2) rollout agreement, both kinematics, hard ground, no damping (contact amplifies float32 noise)
    dyn = Path('data/springgaus_powerfoam/burger_dynamic')
    meta = json.loads((dyn/'cameras.json').read_text())
    reg = json.loads((dyn/'register_calib'/'initial_state_upright_nospin.json').read_text())
    gz = json.loads((dyn/'rest'/'ground_rim.json').read_text())['plane_z_at_centre']
    for kin in ('exact', 'small'):
        m = build(g, 2e4, .3, dev, 'noslip', .25, 0., f32, inertia_scale=1., kinematics=kin)
        m.set_environment(gravity=meta['gravity'], ground_point=[0, 0, gz], ground_normal=meta['ground_normal'],
                          contact_radius=m.contact_radius.cpu().numpy(), ground_mode='noslip')
        wf = WarpFoam(m)
        theta = torch.zeros(m.n, 7, device=dev)
        x0, q0, _, w0 = m.initial_state()
        v0 = torch.tensor(reg['v0'], dtype=f32, device=dev).expand_as(x0).contiguous()
        sub = int(np.ceil((1/120)/m.stable_dt(theta)))
        frames = 9
        with torch.no_grad():
            pw, qw = warp_rollout(wf, theta, (x0, q0, v0, w0), 1/120/sub, sub, frames)
            pt, qt, _ = m.rollout(theta, (x0, q0, v0, w0), 1/120/sub, sub, frames, checkpoint=False, quaternions=True)
        h = float(np.median(np.cbrt(g['masses'])))
        err = (pw-pt).norm(dim=-1).amax(-1)/h
        com = ((pw-pt).mean(1)).norm(dim=-1)/h
        print('[rollout, %s, %d substeps/frame] per frame, max |x_warp - x_torch| / body size: %s | centre-of-mass diff / body size: %s' % (
            kin, sub, np.round(err.cpu().numpy(), 3).tolist(), np.round(com.cpu().numpy(), 4).tolist()), flush=True)


if __name__ == '__main__':
    main()
