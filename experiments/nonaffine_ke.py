"""How much kinetic energy sits in mesh-scale (locally non-affine) motion after impact: VEM vs MPM.

For sampled points i, a mass-weighted least-squares affine fit v ~ a + B (x - x_i) of the
velocities of all points within radius r of x_i; the residual at i, r_i = v_i - a, is the
non-affine (sub-r) part. Reported: sum m |r|^2 / 2 over the samples relative to the kinetic
energy in the centre-of-mass frame of the same samples. Same definition for VEM vertices (lumped
masses) and MPM particles (equal masses). Runs: SFVEM NH (no damping, until it fails), pyrlin
beta=8 (stable), MPM port at the GT settings.
"""
import json
import sys
import numpy as np
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, '.')
from experiments.pacnerf_dem_forward import read_ply
from experiments.pacnerf_mpm_baseline import OFFSETS
from experiments.sfvem_dynamics import SFVEMSolid
from experiments.vem_dynamics import VEMSolid

gt = json.loads(open('data/pacnerf/gt/elastic.json').read())['0']
TIMES = [4.5, 4.75, 5.0, 5.25, 5.5, 5.75, 6.0, 6.5, 7.0, 8.0]
RADII = [.01, .02]


def measure(x, v, m, label, nsample=20000, seed=0):
    x = x.double(); v = v.double(); m = m.double()
    rng = np.random.default_rng(seed)
    idx = torch.as_tensor(rng.choice(len(x), min(nsample, len(x)), replace=False), device=x.device)
    tree = cKDTree(x.cpu().numpy())
    out = {}
    for r in RADII:
        k = 96
        dist, nb = tree.query(x[idx].cpu().numpy(), k=k)
        nb = torch.as_tensor(nb, device=x.device); dist = torch.as_tensor(dist, device=x.device)
        w = m[nb]*(dist < r)
        dx = x[nb]-x[idx][:, None]
        A = torch.cat((torch.ones_like(dx[..., :1]), dx), -1)              # (s, k, 4)
        AtW = A.transpose(1, 2)*w[:, None, :]
        coef = torch.linalg.solve(AtW@A+1e-12*torch.eye(4, device=x.device, dtype=x.dtype), AtW@v[nb])   # (s, 4, 3)
        res = v[idx]-coef[:, 0]
        ms = m[idx]
        vcm = (ms[:, None]*v[idx]).sum(0)/ms.sum()
        ke = .5*(ms*((v[idx]-vcm)**2).sum(1)).sum()
        na = .5*(ms*(res**2).sum(1)).sum()
        out[f'r{int(r*100)}cm'] = float(na/ke)
        out[f'rms_residual_speed_r{int(r*100)}cm'] = float(((ms*(res**2).sum(1)).sum()/ms.sum()).sqrt())
    out['vmax'] = float(v.norm(dim=1).max())
    print(f'   {label}: non-affine KE fraction r=1cm {out["r1cm"]:.3f}  r=2cm {out["r2cm"]:.3f} | rms residual speed '
          f'{out["rms_residual_speed_r1cm"]:.3f} / {out["rms_residual_speed_r2cm"]:.3f} m/s | vmax {out["vmax"]:.2f}', flush=True)
    return out


def run_vem(m, p, name, factor=1.):
    w = m.max_frequency(tuple(torch.as_tensor(t, device='cuda') for t in p))
    sub = int(np.ceil(factor*w/24)); dt = torch.tensor(1/24/sub, device='cuda')
    x, v = m.X.clone(), torch.tensor(gt['v'], device='cuda').expand(m.nv, 3).clone()
    step = torch.compile(m.step, dynamic=False)
    p = tuple(torch.as_tensor(t, device='cuda') for t in p)
    print(f'== {name} ({sub} substeps/frame)', flush=True)
    res = {}
    marks = {int(round(t*sub)): t for t in TIMES}
    with torch.no_grad():
        for n in range(1, int(TIMES[-1]*sub)+1):
            x, v = step(x, v, p, dt)
            if not torch.isfinite(x).all():
                print(f'   NaN at frame {n/sub:.2f}', flush=True); break
            if n in marks:
                res[marks[n]] = measure(x, v, m.mass, f'frame {marks[n]:.2f}')
    return res


def run_mpm():
    dev, dtype = 'cuda', torch.float32
    E, nu = gt['E'], gt['nu']; mu, lam = E/(2*(1+nu)), E*nu/((1+nu)*(1-2*nu))
    dx, dt = .02, 1/4800
    x = torch.as_tensor(read_ply('data/pacnerf/simulation_data/elastic/0/0.ply'), dtype=dtype, device=dev); n = len(x)
    v = torch.as_tensor(gt['v'], dtype=dtype, device=dev).expand(n, 3).clone()
    C = torch.zeros(n, 3, 3, dtype=dtype, device=dev); F = torch.eye(3, dtype=dtype, device=dev).expand(n, 3, 3).clone()
    p_vol = (dx*.5)**3; p_mass = 1000*p_vol; inv_dx = 1/dx
    lo = torch.tensor([-80, -10, -60], device=dev); shape = torch.tensor([160, 110, 120], device=dev)
    strides = torch.tensor([shape[1]*shape[2], shape[2], 1], device=dev); offsets = OFFSETS.to(dev)
    gravity = torch.tensor([0., -9.8, 0.], dtype=dtype, device=dev); eye = torch.eye(3, dtype=dtype, device=dev)
    marks = {int(round(t*200)): t for t in TIMES}
    print('== MPM (GT settings, 200 substeps/frame)', flush=True)
    res = {}
    for s in range(1, int(TIMES[-1]*200)+1):
        F = (eye+dt*C)@F
        J = torch.linalg.det(F)
        tau = mu*(F@F.transpose(1, 2))+(lam*torch.log(J)-mu)[:, None, None]*eye
        affine = (-dt*p_vol*4*inv_dx**2)*tau+p_mass*C
        base = torch.floor(x*inv_dx-.5).long(); fx = x*inv_dx-base.to(dtype)
        w = torch.stack((.5*(1.5-fx)**2, .75-(fx-1)**2, .5*(fx-.5)**2), 1)
        idx = base[:, None, :]+offsets[None]; flat = ((idx-lo)*strides).sum(-1)
        weight = w[:, offsets[:, 0], 0]*w[:, offsets[:, 1], 1]*w[:, offsets[:, 2], 2]
        dpos = (offsets[None].to(dtype)-fx[:, None, :])*dx
        momentum = weight[..., None]*(p_mass*v[:, None, :]+(affine[:, None]@dpos[..., None])[..., 0])
        cells = int(shape.prod())
        grid_m = torch.zeros(cells, dtype=dtype, device=dev).index_add_(0, flat.reshape(-1), (weight*p_mass).reshape(-1))
        grid_v = torch.zeros(cells, 3, dtype=dtype, device=dev).index_add_(0, flat.reshape(-1), momentum.reshape(-1, 3))
        active = grid_m > 1e-8*dx**3
        grid_v = torch.where(active[:, None], grid_v/grid_m.clamp(min=1e-30)[:, None]+dt*gravity, torch.zeros_like(grid_v))
        node_y = (torch.arange(cells, device=dev)//strides[1]) % shape[1]+lo[1]
        grid_v = torch.where((node_y.to(dtype)*dx <= 1e-6)[:, None], torch.zeros_like(grid_v), grid_v)
        g_v = grid_v[flat]
        v = (weight[..., None]*g_v).sum(1)
        C = 4*inv_dx*(weight[..., None, None]*g_v[..., :, None]*(dpos/dx)[..., None, :]).sum(1)
        x = x+dt*v
        if s in marks:
            res[marks[s]] = measure(x, v, torch.ones(n, device=dev), f'frame {marks[s]:.2f}')
    return res


if __name__ == '__main__':
    results = {}
    results['mpm'] = run_mpm()
    m = VEMSolid('output/pacnerf_vem/elastic_0_f0/vem_cap1_bodies_pyr.npz', 'cuda', torch.float32, stab='pyrlin'); m.ground_mode = 'noslip'
    results['pyrlin_b8'] = run_vem(m, (gt['E'], gt['nu'], 8.), 'pyrlin beta=8')
    del m
    m = SFVEMSolid('output/pacnerf_vem/elastic_0_f0/sfvem_cells.npz', 'cuda', torch.float32)
    results['sfvem_nh'] = run_vem(m, (gt['E'], gt['nu'], 1.), 'SFVEM NH, no damping')
    json.dump({k: {str(t): r for t, r in v.items()} for k, v in results.items()}, open('output/pacnerf_vem/elastic_0_f0/nonaffine_ke_v1.json', 'w'), indent=1)
