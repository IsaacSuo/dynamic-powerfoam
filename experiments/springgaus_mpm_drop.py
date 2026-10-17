"""Continuum reference for the burger drop: MLS-MPM (the torch port of PAC-NeRF's simulator, same
Neo-Hookean law) on the same shape, material, initial velocity and table, measured like
springgaus_drop_test / springgaus_energy_budget.

Shape: the strict static visual hull (the region the DEM material cells were selected from),
sampled on a lattice of spacing dx/2 (8 particles per cell, p_vol = (dx/2)^3, rho = 1000), mapped
to the dynamic world with the registration used for the DEM. MPM frame: y up, table at y = 0:
(x, y, z)_mpm = (x, ground_z - z, y)_world (proper rotation). Ground boundary at grid nodes y <= 0:
'sticky' (PAC-NeRF: all velocity zeroed) or 'separate' (frictionless, separable: only the
velocity into the table removed). Energies per sample: KE_com, KE_internal (particle velocities
relative to the COM; APIC affine velocity not included), elastic sum p_vol psi(F), gravity.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.pacnerf_foam_geometry import foam_to_dynamic, visual_hull_colmap

OUT = Path('data/springgaus_powerfoam')
RHO = 1000.
OFFSETS = torch.tensor([(i, j, k) for i in range(3) for j in range(3) for k in range(3)])


def burger_particles(obj, registration, dx):
    sd = OUT/obj
    A, b, k = foam_to_dynamic(sd, OUT/f'{obj}_dynamic'/'register_calib'/registration)
    h = dx/2/k                                                    # lattice spacing in foam units
    ax = np.arange(.05, .95, h)
    grid = np.stack(np.meshgrid(ax, ax, ax, indexing='ij'), -1).reshape(-1, 3)
    inside = np.zeros(len(grid), bool)
    for s in range(0, len(grid), 2_000_000):
        inside[s:s+2_000_000] = visual_hull_colmap(grid[s:s+2_000_000], sd)
    return grid[inside]@A.T+b


def simulate(x, v0, E, nu, dx, dt, steps, sample_every, bc, device='cuda', dtype=torch.float32, positions=None, gravity_mag=9.8, margin=.12):
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
    hi_w = x.max(0).values+torch.tensor([margin, .05, margin], device=device, dtype=dtype)
    lo = torch.floor(lo_w*inv_dx).long(); lo[1] = -4
    shape = torch.ceil(hi_w*inv_dx).long()-lo+4
    strides = torch.tensor([shape[1]*shape[2], shape[2], 1], device=device)
    offsets = OFFSETS.to(device)
    gravity = torch.tensor((0., -gravity_mag, 0.), dtype=dtype, device=device)
    eye = torch.eye(3, dtype=dtype, device=device)
    nodes = torch.stack(torch.meshgrid(*[torch.arange(int(shape[d]), device=device) for d in range(3)], indexing='ij'), -1).reshape(-1, 3)+lo
    floor = (nodes[:, 1].to(dtype)*dx <= 1e-6)
    rows = []

    def measure(t):
        if positions is not None:
            positions.append(x.detach().cpu().numpy().copy())
        vcom = v.mean(0); com = x.mean(0)
        M = p_mass*n
        J = torch.det(F)
        psi = .5*mu*((F*F).sum((1, 2))-3)-mu*torch.log(J)+.5*lam*torch.log(J)**2
        y = x[:, 1]
        rows.append(dict(t=t, KE_com=float(.5*M*(vcom*vcom).sum()), KE_internal=float(.5*p_mass*((v-vcom)**2).sum()),
                         E_elastic=float(p_vol*psi.sum()), E_grav=float(-M*(gravity@com)), vy_com=float(vcom[1]),
                         height=float(y.max()-y.min()+dx/2), width=float(max(np.ptp(x[:, 0].cpu().numpy()), np.ptp(x[:, 2].cpu().numpy()))+dx/2),
                         touching=bool((y.min() <= dx).item()), min_J=float(J.min())))
    measure(0.)
    for s in range(steps):
        F = (eye+dt*C)@F
        J = torch.det(F)
        Ft = F.transpose(1, 2)
        tau = mu*(F@Ft)+((lam*torch.log(J)-mu)[:, None, None])*eye
        affine = (-dt*p_vol*4*inv_dx**2)*tau+p_mass*C
        base = torch.floor(x*inv_dx-.5).long()
        fx = x*inv_dx-base.to(dtype)
        w = [.5*(1.5-fx)**2, .75-(fx-1)**2, .5*(fx-.5)**2]
        W = torch.stack(w, 1)                                                  # (n, 3, 3)
        weight = W[:, offsets[:, 0], 0]*W[:, offsets[:, 1], 1]*W[:, offsets[:, 2], 2]
        idx = base[:, None, :]+offsets[None]
        if not bool(((idx-lo >= 0) & (idx-lo < shape)).all()):
            raise RuntimeError('particle left the grid box at step %d (t %.4f s): x range %s..%s, max |v| %.2f m/s, min J %.3f, max J %.3f' % (
                s, s*dt, x.min(0).values.tolist(), x.max(0).values.tolist(), float(v.norm(dim=-1).max()),
                float(torch.det(F).min()), float(torch.det(F).max())))
        flat = ((idx-lo)*strides).sum(-1)
        dpos = (offsets[None].to(dtype)-fx[:, None, :])*dx
        mom = weight[..., None]*(p_mass*v[:, None, :]+(affine[:, None]@dpos[..., None])[..., 0])
        N = int(shape.prod())
        grid_v = torch.zeros(N, 3, dtype=dtype, device=device).index_add_(0, flat.reshape(-1), mom.reshape(-1, 3))
        grid_m = torch.zeros(N, dtype=dtype, device=device).index_add_(0, flat.reshape(-1), (weight*p_mass).reshape(-1))
        active = grid_m > 1e-8*dx**3
        grid_v = torch.where(active[:, None], grid_v/grid_m.clamp(min=1e-30)[:, None]+dt*gravity, torch.zeros_like(grid_v))
        if bc == 'sticky':
            grid_v = torch.where(floor[:, None], torch.zeros_like(grid_v), grid_v)
        else:
            vy = grid_v[:, 1].clamp(min=0.)
            grid_v = torch.where(floor[:, None], torch.stack((grid_v[:, 0], vy, grid_v[:, 2]), -1), grid_v)
        g_v = grid_v[flat]
        v = (weight[..., None]*g_v).sum(1)
        C = 4*inv_dx**2*(weight[..., None, None]*g_v[..., :, None]*dpos[..., None, :]).sum(1)
        x = x+dt*v
        if (s+1) % sample_every == 0:
            measure((s+1)*dt)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--obj', default='burger')
    p.add_argument('--registration', default='initial_state_upright_nospin.json')
    p.add_argument('--E', type=float, default=2e4)
    p.add_argument('--nu', type=float, default=.3)
    p.add_argument('--factors', type=float, nargs='+', default=[1., 5.])
    p.add_argument('--bcs', nargs='+', default=['separate', 'sticky'])
    p.add_argument('--dx', type=float, default=.0025)
    p.add_argument('--dt', type=float, default=2e-5)
    p.add_argument('--duration', type=float, default=.2)
    p.add_argument('--sample', type=float, default=1/1200)
    p.add_argument('--tag', default='')
    p.add_argument('--out', type=Path, default=Path('output/springgaus_dem/burger_spacing8k_mf0.1/mpm_drop'))
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    dyn = OUT/f'{a.obj}_dynamic'
    reg = json.loads((dyn/'register_calib'/a.registration).read_text())
    ground_z = json.loads((dyn/'rest'/'ground_rim.json').read_text())['plane_z_at_centre']
    xw = burger_particles(a.obj, a.registration, a.dx)
    to_mpm = lambda p_: np.stack((p_[..., 0], ground_z-p_[..., 2], p_[..., 1]), -1)
    x0 = to_mpm(xw); v0 = to_mpm(np.array(reg['v0'])+np.array([0., 0., ground_z]))   # velocity: no offset
    v0 = np.array([reg['v0'][0], -reg['v0'][2], reg['v0'][1]])
    print('particles %d, volume %.1f cm^3, height above table %.1f mm (bottom) .. %.1f mm (top)' % (
        len(x0), len(x0)*(a.dx/2)**3*1e6, 1000*x0[:, 1].min(), 1000*x0[:, 1].max()), flush=True)
    sample_every = int(round(a.sample/a.dt))
    out = {}
    for bc in a.bcs:
        for fac in a.factors:
            rows = simulate(x0, v0, a.E*fac, a.nu, a.dx, a.dt, int(round(a.duration/a.dt)), sample_every, bc)
            R = {k_: np.array([r[k_] for r in rows]) for k_ in rows[0]}
            hit = int(np.argmax(R['touching']))
            imp = max(hit-1, 0)
            v_in = float(-R['vy_com'][:hit+1].min())
            v_up = float(R['vy_com'][hit:].max())
            off = hit+int(np.argmin(R['touching'][hit:])) if not R['touching'][-1] else len(rows)-1
            ke = R['KE_com'][imp]
            tot = R['KE_com']+R['KE_internal']+R['E_elastic']+R['E_grav']
            s = dict(bc=bc, factor=fac, restitution=v_up/v_in, v_in=v_in, v_up=v_up, squash=float(R['height'].min()/R['height'][0]),
                     spread=float(R['width'].max()/R['width'][0]), contact_ms=float(R['touching'].sum()*a.sample*1000),
                     energy_lost_in_contact=float((tot[imp]-tot[off])/ke),
                     at_liftoff=dict(com=float(R['KE_com'][off]/ke), internal=float(R['KE_internal'][off]/ke), elastic=float(R['E_elastic'][off]/ke)),
                     height_end=float(R['height'][-1]/R['height'][0]), min_J=float(R['min_J'].min()))
            out[f'{bc}_x{fac:g}'] = dict(summary=s, series={k_: v_.tolist() for k_, v_ in R.items()})
            print('MPM %-8s E x%-3g restitution %.2f (in %.2f up %.2f m/s) squash %.2f spread %.2f contact %.1f ms | lost in contact %.2f | at lift-off com %.2f internal %.2f elastic %.2f | end height %.2f | min J %.2f' % (
                bc, fac, s['restitution'], v_in, v_up, s['squash'], s['spread'], s['contact_ms'], s['energy_lost_in_contact'],
                s['at_liftoff']['com'], s['at_liftoff']['internal'], s['at_liftoff']['elastic'], s['height_end'], s['min_J']), flush=True)
    (a.out/f'mpm_drop{a.tag}.json').write_text(json.dumps(out))


if __name__ == '__main__':
    main()
