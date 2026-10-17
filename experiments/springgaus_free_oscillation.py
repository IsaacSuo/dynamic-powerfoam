"""Root cause, step 1: the burger DEM vs a continuum (MLS-MPM) in free flight -- no ground, no gravity.

Same object on both sides: MPM particles are sampled on a lattice (spacing dx/2) and kept where the
owner power cell (foam sites/radii, power distance <= 0) is one of the DEM material cells, then mapped
to the dynamic world with the DEM's registration. Initial condition: a vertical squeeze
v_z = -a (z - z_c) (a ~ impact-like at --a-large, linear at --a-small). Per sample:
  KE_affine    kinetic energy of the mass-weighted least-squares affine velocity field v_c + L (x - c)
  KE_nonaffine the rest of the translational kinetic energy (scattered motion)
  KE_rot       DEM body spins (0 for MPM)
  E_elastic    DEM: EnergyFoam.energy; MPM: sum p_vol psi_NH(F)
  height       extent along the squeeze axis (oscillation period -> effective stiffness)
No damping on the MPM side (its transfers are mildly dissipative); the DEM is run undamped and with
its default dashpots.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import warp as wp
from scipy.spatial import cKDTree

from experiments.foam_dem_warp import WarpFoam
from experiments.pacnerf_dem_forward import build
from experiments.pacnerf_foam_geometry import foam_to_dynamic

OUT = Path('data/springgaus_powerfoam')
RHO = 1000.
OFFSETS = torch.tensor([(i, j, k) for i in range(3) for j in range(3) for k in range(3)])


def material_particles(run, dem, scene_dir, registration, dx):
    st = torch.load(Path(run)/'model.pt', map_location='cpu', weights_only=True)
    p = st['points'].double().numpy()
    r = torch.nn.functional.softplus(st['radii'].double(), beta=100).numpy()
    cells = dict(np.load(Path(dem)/'cells.npz'))
    is_mat = np.zeros(len(p), bool); is_mat[cells['source_index']] = True
    A, b, k = foam_to_dynamic(scene_dir, registration)
    q = p[is_mat]
    lo, hi = q.min(0)-.02, q.max(0)+.02
    h = dx/2/k
    axes = [np.arange(lo[d]+h/2, hi[d], h) for d in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
    _, nn = cKDTree(p).query(grid, k=8)
    power = ((grid[:, None]-p[nn])**2).sum(-1)-r[nn]**2
    best = power.argmin(1)
    owner = nn[np.arange(len(grid)), best]
    keep = is_mat[owner] & (power[np.arange(len(grid)), best] <= 0)
    return grid[keep]@A.T+b


def affine_split(x, v, m):
    """Kinetic energy of the best affine velocity field and of the remainder (mass weighted)."""
    M = m.sum()
    c = (m[:, None]*x).sum(0)/M; vc = (m[:, None]*v).sum(0)/M
    dx_, dv = x-c, v-vc
    S = (m[:, None, None]*dx_[:, :, None]*dx_[:, None, :]).sum(0)
    B = (m[:, None, None]*dv[:, :, None]*dx_[:, None, :]).sum(0)
    L = B@torch.linalg.inv(S)
    va = vc+dx_@L.T
    ke_aff = .5*(m*(va*va).sum(-1)).sum()
    ke_tot = .5*(m*(v*v).sum(-1)).sum()
    return float(ke_aff), float(ke_tot-ke_aff)


def run_mpm(x0, axis, a, E, nu, dx, dt, duration, sample, device='cuda'):
    f32 = torch.float32
    mu, lam = E/(2*(1+nu)), E*nu/((1+nu)*(1-2*nu))
    x = torch.tensor(x0, dtype=f32, device=device)
    n = len(x)
    c = x.mean(0)
    v = torch.zeros_like(x); v[:, axis] = -a*(x[:, axis]-c[axis])
    C = torch.zeros(n, 3, 3, dtype=f32, device=device)
    F = torch.eye(3, dtype=f32, device=device).expand(n, 3, 3).clone()
    p_vol = (dx*.5)**3; p_mass = RHO*p_vol; inv_dx = 1/dx
    lo = torch.floor((x.min(0).values-.1)*inv_dx).long()
    shape = torch.ceil((x.max(0).values+.1)*inv_dx).long()-lo+4
    strides = torch.tensor([shape[1]*shape[2], shape[2], 1], device=device)
    offsets = OFFSETS.to(device)
    eye = torch.eye(3, dtype=f32, device=device)
    m = torch.full((n,), p_mass, dtype=f32, device=device)
    rows = []
    every = int(round(sample/dt))

    def measure(t):
        J = torch.det(F)
        psi = .5*mu*((F*F).sum((1, 2))-3)-mu*torch.log(J)+.5*lam*torch.log(J)**2
        ka, kn = affine_split(x, v, m)
        rows.append(dict(t=t, KE_affine=ka, KE_nonaffine=kn, KE_rot=0., E_elastic=float(p_vol*psi.sum()),
                         height=float(x[:, axis].max()-x[:, axis].min()), min_J=float(J.min())))
    measure(0.)
    for s in range(int(round(duration/dt))):
        F = (eye+dt*C)@F
        J = torch.det(F)
        tau = mu*(F@F.transpose(1, 2))+((lam*torch.log(J)-mu)[:, None, None])*eye
        affine = (-dt*p_vol*4*inv_dx**2)*tau+p_mass*C
        base = torch.floor(x*inv_dx-.5).long()
        fx = x*inv_dx-base.to(f32)
        W = torch.stack([.5*(1.5-fx)**2, .75-(fx-1)**2, .5*(fx-.5)**2], 1)
        weight = W[:, offsets[:, 0], 0]*W[:, offsets[:, 1], 1]*W[:, offsets[:, 2], 2]
        idx = base[:, None, :]+offsets[None]
        if not bool(((idx-lo >= 0) & (idx-lo < shape)).all()):
            rows.append(dict(t=s*dt, failed='left the grid'))
            break
        flat = ((idx-lo)*strides).sum(-1)
        dpos = (offsets[None].to(f32)-fx[:, None, :])*dx
        mom = weight[..., None]*(p_mass*v[:, None, :]+(affine[:, None]@dpos[..., None])[..., 0])
        N = int(shape.prod())
        gv = torch.zeros(N, 3, dtype=f32, device=device).index_add_(0, flat.reshape(-1), mom.reshape(-1, 3))
        gm = torch.zeros(N, dtype=f32, device=device).index_add_(0, flat.reshape(-1), (weight*p_mass).reshape(-1))
        gv = torch.where((gm > 1e-8*dx**3)[:, None], gv/gm.clamp(min=1e-30)[:, None], torch.zeros_like(gv))
        g_v = gv[flat]
        v = (weight[..., None]*g_v).sum(1)
        C = 4*inv_dx**2*(weight[..., None, None]*g_v[..., :, None]*dpos[..., None, :]).sum(1)
        x = x+dt*v
        if (s+1) % every == 0:
            measure((s+1)*dt)
    return rows


def run_dem(g, axis, a, E, nu, damping, kinematics, duration, sample, device='cuda'):
    f32 = torch.float32
    m = build(g, E, nu, device, 'noslip', .25, .3, f32, inertia_scale=4., kinematics=kinematics)
    m.set_environment(gravity=[0, 0, 0], ground_point=[0, 0, 1e3], ground_normal=[0, 0, -1],
                      contact_radius=m.contact_radius.cpu().numpy(), ground_mode='noslip')
    wf = WarpFoam(m)
    theta = torch.zeros(m.n, 7, dtype=f32, device=device)
    theta[:, 6] = 0. if damping else -30.
    sub = int(np.ceil(sample/m.stable_dt(theta)))
    dt = sample/sub
    x0, q0, _, w0 = m.initial_state()
    c = (m.mass[:, None]*x0).sum(0)/m.mass.sum()
    v0 = torch.zeros_like(x0); v0[:, axis] = -a*(x0[:, axis]-c[axis])
    dtypes = (wp.vec3, wp.vec4, wp.vec3, wp.vec3)
    s = tuple(wp.from_torch(t_.contiguous().clone(), dtype=d) for t_, d in zip((x0, q0, v0, w0), dtypes))
    other = tuple(wp.zeros(m.n, dtype=d, device='cuda') for d in dtypes)
    buf = wf._buffers(False)
    th = wp.from_torch(theta.contiguous())
    rows = []

    def measure(t):
        x, q, v, w = (wp.to_torch(arr).clone() for arr in s)
        ka, kn = affine_split(x, v, m.mass)
        kr = float(.5*(m.inertia*(w*w).sum(-1)).sum())
        F = m.deformation_gradient(x)
        rows.append(dict(t=t, KE_affine=ka, KE_nonaffine=kn, KE_rot=kr, E_elastic=float(m.energy(theta, x, q)),
                         height=float(x[:, axis].max()-x[:, axis].min()), min_J=float(torch.det(F).min()),
                         finite=bool(torch.isfinite(x).all())))
    with torch.no_grad():
        measure(0.)
        for k_ in range(int(round(duration/sample))):
            for _ in range(sub):
                for b_ in buf.values():
                    b_.zero_()
                wf.step(th, s, other, buf, dt)
                s, other = other, s
            measure((k_+1)*sample)
            if not rows[-1]['finite']:
                break
    return rows, sub


def summarize(rows, sample):
    R = {k_: np.array([r.get(k_, np.nan) for r in rows], float) for k_ in ('t', 'KE_affine', 'KE_nonaffine', 'KE_rot', 'E_elastic', 'height', 'min_J')}
    E0 = R['KE_affine'][0]+R['KE_nonaffine'][0]+R['KE_rot'][0]+R['E_elastic'][0]
    tot = (R['KE_affine']+R['KE_nonaffine']+R['KE_rot']+R['E_elastic'])/E0
    h = R['height']-np.nanmean(R['height'])
    zc = np.where(np.diff(np.sign(h)) != 0)[0]
    period = float(2*np.mean(np.diff(zc))*sample) if len(zc) > 2 else float('nan')
    idx = lambda t: min(int(round(t/sample)), len(rows)-1)
    at = lambda t: dict(affine=float(R['KE_affine'][idx(t)]/E0), nonaffine=float(R['KE_nonaffine'][idx(t)]/E0),
                        rot=float(R['KE_rot'][idx(t)]/E0), elastic=float(R['E_elastic'][idx(t)]/E0), total=float(tot[idx(t)]))
    # coherent energy kept: peak affine KE in the last quarter-window vs the initial affine KE
    late = R['KE_affine'][len(rows)*3//4:]
    return dict(period_ms=1000*period, min_height_ratio=float(np.nanmin(R['height'])/R['height'][0]),
                min_J=float(np.nanmin(R['min_J'])), total_energy_end=float(tot[-1]), samples=len(rows),
                affine_peak_late_over_initial=float(np.nanmax(late)/R['KE_affine'][0]), at_end=at(R['t'][-1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', default='output/springgaus_burger_spacing8k')
    p.add_argument('--dem', default='output/springgaus_dem/burger_spacing8k_mf0.1')
    p.add_argument('--registration', default='initial_state_upright_nospin.json')
    p.add_argument('--E', type=float, default=2e4)
    p.add_argument('--nu', type=float, default=.3)
    p.add_argument('--amplitudes', type=float, nargs='+', default=[4., 80.])
    p.add_argument('--dx', type=float, default=.0025)
    p.add_argument('--mpm-dt', type=float, default=2e-5)
    p.add_argument('--duration', type=float, default=.1)
    p.add_argument('--sample', type=float, default=1/2400)
    p.add_argument('--out', type=Path, default=Path('output/springgaus_dem/burger_spacing8k_mf0.1/free_oscillation'))
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    wp.init()
    reg = OUT/'burger_dynamic'/'register_calib'/a.registration
    g = dict(np.load(Path(a.dem)/'geometry.npz'))
    axis = 2                                                    # gravity axis of the dynamic world
    xm = material_particles(a.run, a.dem, OUT/'burger', reg, a.dx)
    print('same object: MPM %d particles, volume %.1f cm^3 | DEM %d bodies, volume %.1f cm^3' % (
        len(xm), len(xm)*(a.dx/2)**3*1e6, len(g['masses']), g['masses'].sum()*1e6), flush=True)
    out = {}
    for amp in a.amplitudes:
        runs = {}
        rows = run_mpm(xm, axis, amp, a.E, a.nu, a.dx, a.mpm_dt, a.duration, a.sample)
        runs['MPM'] = (rows, None)
        for name, damp, kin in (('DEM undamped', False, 'small'), ('DEM damped', True, 'small'), ('DEM undamped exact', False, 'exact')):
            rows, sub = run_dem(g, axis, amp, a.E, a.nu, damp, kin, a.duration, a.sample)
            runs[name] = (rows, sub)
        for name, (rows, sub) in runs.items():
            s = summarize(rows, a.sample)
            out[f'a{amp:g}_{name}'] = dict(summary=s, series=rows)
            e = s['at_end']
            print('a=%-4g %-19s period %5.1f ms | min height %.2f, min J %.2f | at end (fraction of initial energy): affine %.2f nonaffine %.2f rot %.2f elastic %.2f, total %.2f | late peak affine / initial %.2f%s' % (
                amp, name, s['period_ms'], s['min_height_ratio'], s['min_J'], e['affine'], e['nonaffine'], e['rot'], e['elastic'],
                e['total'], s['affine_peak_late_over_initial'], '' if s['samples'] > a.duration/a.sample else ' (STOPPED: diverged/failed at t=%.1f ms)' % (1000*rows[-1]['t'])), flush=True)
    (a.out/'free_oscillation.json').write_text(json.dumps(out))


if __name__ == '__main__':
    main()
