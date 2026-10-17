"""Independent continuum check of the oblique-bounce nu signal (IDENTIFIABILITY.md section 11).

MLS-MPM (the torch port of PAC-NeRF's simulator, springgaus_mpm_drop.simulate: same Neo-Hookean law, APIC transfer) on a
solid ball, with a Coulomb-friction floor instead of the sticky / frictionless ones: at grid nodes y <= 0 moving into the
floor, the normal velocity is removed and the tangential speed is reduced by mu_f times the removed normal speed
(sticking when that exceeds it) - the standard MPM friction boundary. No material damping (so the contact mechanics are
not blurred by a damping model). Runs at fixed E* = E / (1 - nu^2) for several nu and friction coefficients; reports the
free-flight state after the bounce: centroid velocity, spin (from the particle angular momentum), bounce height, and the
contact duration.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

RHO = 1000.
OFFSETS = torch.tensor([(i, j, k) for i in range(3) for j in range(3) for k in range(3)])


def ball_particles(R, h):
    ax = np.arange(-R+h/2, R, h)
    g = np.stack(np.meshgrid(ax, ax, ax, indexing='ij'), -1).reshape(-1, 3)
    return g[np.linalg.norm(g, axis=1) <= R]


def simulate(x, v0, E, nu, mu_f, dx, dt, steps, sample_every, device='cuda', dtype=torch.float32, gravity_mag=9.8, margin=.25):
    mu, lam = E/(2*(1+nu)), E*nu/((1+nu)*(1-2*nu))
    x = torch.as_tensor(x, dtype=dtype, device=device)
    n = len(x)
    v = torch.as_tensor(v0, dtype=dtype, device=device).expand(n, 3).clone()
    C = torch.zeros(n, 3, 3, dtype=dtype, device=device)
    F = torch.eye(3, dtype=dtype, device=device).expand(n, 3, 3).clone()
    p_vol = (dx*.5)**3
    p_mass = RHO*p_vol
    inv_dx = 1/dx
    lo_w = x.min(0).values-torch.tensor([margin, 0., margin], device=device, dtype=dtype)
    hi_w = x.max(0).values+torch.tensor([margin, .2, margin], device=device, dtype=dtype)
    lo = torch.floor(lo_w*inv_dx).long(); lo[1] = -4
    shape = torch.ceil(hi_w*inv_dx).long()-lo+4
    strides = torch.tensor([shape[1]*shape[2], shape[2], 1], device=device)
    offsets = OFFSETS.to(device)
    gravity = torch.tensor((0., -gravity_mag, 0.), dtype=dtype, device=device)
    eye = torch.eye(3, dtype=dtype, device=device)
    nodes = torch.stack(torch.meshgrid(*[torch.arange(int(shape[d]), device=device) for d in range(3)], indexing='ij'), -1).reshape(-1, 3)+lo
    floor = (nodes[:, 1].to(dtype)*dx <= 1e-6)
    samples = []

    def measure(t):
        c = x.mean(0); vc = v.mean(0)
        r = x-c
        L = (torch.linalg.cross(r, v-vc)).sum(0)                     # angular momentum / p_mass
        I = (r*r).sum()*eye-r.T@r
        w = torch.linalg.solve(I, L)
        low = x[:, 1] <= float(x[:, 1].min())+dx                       # bottom layer of particles (contact patch)
        samples.append(dict(t=t, c=c.tolist(), v=vc.tolist(), w=w.tolist(), L=(p_mass*L).tolist(), M=p_mass*n, ymin=float(x[:, 1].min()),
                            patch_x=float(x[low, 0].mean()), height=float(x[:, 1].max()-x[:, 1].min()), minJ=float(torch.det(F).min()),
                            width=float(max(x[:, 0].max()-x[:, 0].min(), x[:, 2].max()-x[:, 2].min()))))
    measure(0.)
    for s in range(steps):
        F = (eye+dt*C)@F
        J = torch.det(F)
        tau = mu*(F@F.transpose(1, 2))+((lam*torch.log(J)-mu)[:, None, None])*eye
        affine = (-dt*p_vol*4*inv_dx**2)*tau+p_mass*C
        base = torch.floor(x*inv_dx-.5).long()
        fx = x*inv_dx-base.to(dtype)
        W = torch.stack([.5*(1.5-fx)**2, .75-(fx-1)**2, .5*(fx-.5)**2], 1)
        weight = W[:, offsets[:, 0], 0]*W[:, offsets[:, 1], 1]*W[:, offsets[:, 2], 2]
        idx = base[:, None, :]+offsets[None]
        if not bool(((idx-lo >= 0) & (idx-lo < shape)).all()):
            raise RuntimeError('particle left the grid box at step %d' % s)
        flat = ((idx-lo)*strides).sum(-1)
        dpos = (offsets[None].to(dtype)-fx[:, None, :])*dx
        mom = weight[..., None]*(p_mass*v[:, None, :]+(affine[:, None]@dpos[..., None])[..., 0])
        N = int(shape.prod())
        grid_v = torch.zeros(N, 3, dtype=dtype, device=device).index_add_(0, flat.reshape(-1), mom.reshape(-1, 3))
        grid_m = torch.zeros(N, dtype=dtype, device=device).index_add_(0, flat.reshape(-1), (weight*p_mass).reshape(-1))
        active = grid_m > 1e-8*dx**3
        grid_v = torch.where(active[:, None], grid_v/grid_m.clamp(min=1e-30)[:, None]+dt*gravity, torch.zeros_like(grid_v))
        # Coulomb floor: remove the velocity into the floor; tangential speed reduced by mu_f |v_n| (stick if exceeded)
        vn = grid_v[:, 1]
        into = floor & (vn < 0)
        vt = torch.stack((grid_v[:, 0], torch.zeros_like(vn), grid_v[:, 2]), -1)
        st = vt.norm(dim=-1)
        scale = torch.clamp(1+mu_f*vn/st.clamp(min=1e-12), min=0.)              # vn < 0
        bc_v = vt*scale[:, None]
        grid_v = torch.where(into[:, None], bc_v, grid_v)
        g_v = grid_v[flat]
        v = (weight[..., None]*g_v).sum(1)
        C = 4*inv_dx**2*(weight[..., None, None]*g_v[..., :, None]*dpos[..., None, :]).sum(1)
        x = x+dt*v
        if (s+1) % sample_every == 0:
            measure((s+1)*dt)
    return samples


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--R', type=float, default=.035)
    p.add_argument('--dx', type=float, default=.0025)
    p.add_argument('--Estar', type=float, default=5e4)
    p.add_argument('--nus', type=float, nargs='+', default=[.3, .4, .45, .49])
    p.add_argument('--mus', type=float, nargs='+', default=[1., .3])
    p.add_argument('--v', type=float, nargs=2, default=[1.0, -1.5], help='horizontal, vertical impact velocity (m/s)')
    p.add_argument('--T', type=float, default=.25)
    p.add_argument('--cfl', type=float, default=.3)
    p.add_argument('--out', type=Path, default=Path('output/mpm_oblique'))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    x0 = ball_particles(args.R, args.dx/2)
    x0[:, 1] += args.R+args.dx                                         # lowest particles just above the floor
    print('ball R %.3f m, %d particles, dx %.4f m' % (args.R, len(x0), args.dx), flush=True)
    results = []
    for mu_f in args.mus:
        for nu in args.nus:
            E = args.Estar*(1-nu*nu)
            M = E*(1-nu)/((1+nu)*(1-2*nu))
            cp = np.sqrt(M/RHO)
            dt = args.cfl*args.dx/(cp+3.)
            steps = int(np.ceil(args.T/dt)); every = max(1, int(round(1e-3/dt)))
            s = simulate(x0, [args.v[0], args.v[1], 0.], E, nu, mu_f, args.dx, dt, steps, every)
            t = np.array([q['t'] for q in s]); y = np.array([q['ymin'] for q in s])
            touch = y <= args.dx*.75
            t_on = t[touch].min() if touch.any() else np.nan; t_off = t[touch].max() if touch.any() else np.nan
            after = t > t_off+.01
            sel = np.flatnonzero(after)
            v_after = np.array([s[k]['v'] for k in sel]).mean(0); w_after = np.array([s[k]['w'] for k in sel]).mean(0)
            peak = max(q['c'][1] for q in s[sel[0]:]) if len(sel) else np.nan
            r = dict(nu=nu, mu_f=mu_f, E=E, dt=dt, contact_ms=1e3*(t_off-t_on), vx_after=float(v_after[0]), vy_after_mean=float(v_after[1]),
                     spin_z=float(w_after[2]), rolling_spin=float(-v_after[0]/args.R), peak_height_mm=1e3*float(peak), minJ=min(q['minJ'] for q in s))
            hs = np.array([q['height'] for q in s]); ws = np.array([q['width'] for q in s])
            k = int(np.argmin(hs))
            r.update(width0_mm=1e3*ws[0], height0_mm=1e3*hs[0], width_max_mm=1e3*float(ws.max()), width_at_min_height_mm=1e3*float(ws[k]),
                     min_height_mm=1e3*float(hs[k]), t_min_height_ms=1e3*s[k]['t'],
                     width_series_mm=(1e3*ws).tolist(), height_series_mm=(1e3*hs).tolist(), t_series_ms=[1e3*q['t'] for q in s])
            print('   widths: rest %.1f mm, max %.1f mm, at min height %.1f mm (min height %.1f of %.1f mm, t %.1f ms)' % (
                r['width0_mm'], r['width_max_mm'], r['width_at_min_height_mm'], r['min_height_mm'], r['height0_mm'], r['t_min_height_ms']), flush=True)
            results.append(r)
            print('mu_f %.1f nu %.2f: contact %.1f ms | after bounce vx %.4f m/s, spin %.2f rad/s (rolling would be %.2f), peak centroid %.1f mm, min J %.3f' % (
                mu_f, nu, r['contact_ms'], r['vx_after'], r['spin_z'], r['rolling_spin'], r['peak_height_mm'], r['minJ']), flush=True)
            (args.out/'results.json').write_text(json.dumps(dict(config={k: str(v) for k, v in vars(args).items()}, results=results), indent=1))


if __name__ == '__main__':
    main()
