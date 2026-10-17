"""SFVEM blow-up diagnostics: energy budget, velocity, point J and within-element J spread per quarter frame.

Variants (GT E, nu): A NH + ground; B NH + Kelvin-Voigt damping tau; C stable NH + ground;
D NH, no ground/gravity, a localised upward 'impact' velocity at the bottom 3 cm (no contact).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.pacnerf_dem_forward import cell_targets
from experiments.sfvem_dynamics import SFVEMSolid


def run(name, law, tau, contact, frames, geo, gt, bodies, cells, T, has, factor=1., ground='noslip', omega_c=2000., friction=1.):
    m = SFVEMSolid(geo, 'cuda', torch.float32, cells=(bodies['body_of_cell'], cells['masses']), law=law)
    m.ground_mode, m.contact_omega, m.friction = ground, omega_c, friction
    p = [torch.tensor(t, device='cuda') for t in (gt['E'], gt['nu'], 1.)]
    w = m.max_frequency(tuple(p))
    zeta = tau*w/2
    sub = int(np.ceil(factor*w/(24*(np.sqrt(1+zeta**2)-zeta))))
    if tau > 0:
        p.append(torch.tensor(tau, device='cuda'))
    p = tuple(p); dt = 1/24/sub
    x = m.X.clone()
    if contact:
        v = torch.tensor(gt['v'], device='cuda').expand(m.nv, 3).clone()
    else:
        m.ground = False; m.gravity = torch.zeros(3, device='cuda')
        y = m.X[:, 1]; v = torch.zeros_like(m.X)
        v[:, 1] = 3.5*torch.exp(-((y-y.min())/.03)**2)
    step = torch.compile(m.step, dynamic=False)
    dtt = torch.tensor(dt, device='cuda')
    rows, probes = [], [m.probe(x)]
    t0 = time.perf_counter()
    print(f'== {name}: law {law}, tau {tau:g}, contact {contact} ({ground}, omega_c {omega_c:g}, friction {friction:g}), omega {w:.0f}, {sub} substeps/frame', flush=True)
    with torch.no_grad():
        for n in range(frames*sub):
            x, v = step(x, v, p, dtt)
            if (n+1) % sub == 0 and torch.isfinite(x).all():
                probes.append(m.probe(x))
            if (n+1) % (sub//4) == 0 or not torch.isfinite(x).all():
                f = (n+1)/sub
                if not torch.isfinite(x).all():
                    print(f'   NaN by frame {f:.2f}', flush=True); break
                E = float(m.energy(x, p[:3])); K = float(.5*(m.mass*(v*v).sum(1)).sum()); G = float((m.mass*-m.gravity[1]*x[:, 1]).sum())
                Jmin, spread = 9., 0.
                for b in m.buckets:
                    Fq, _ = m._bucket_F(x, b)
                    J = torch.linalg.det(Fq.reshape(-1, 3, 3)).view(-1, b['Qp'])
                    Wb = b['W']; ok = Wb > 0
                    Fm = (Fq*Wb[..., None, None]).sum(1)/Wb.sum(1)[:, None, None]
                    Jm = torch.linalg.det(Fm)
                    Jq = torch.where(ok, J, torch.full_like(J, 9.))
                    Jmin = min(Jmin, float(Jq.min())); spread = max(spread, float((Jm-Jq.min(1).values).max()))
                rows.append(dict(frame=f, total=E+K+G, strain=E, kinetic=K, vmax=float(v.norm(dim=1).max()), Jmin=Jmin, Jspread=spread))
                if f >= 4.5 or (n+1) % sub == 0:
                    r = rows[-1]
                    print(f'   f{f:5.2f} total {r["total"]:8.3f} strain {r["strain"]:7.3f} kin {r["kinetic"]:7.3f} vmax {r["vmax"]:6.2f} '
                          f'Jmin {r["Jmin"]:6.2f} (elem mean - point min, max over elems {r["Jspread"]:.2f})', flush=True)
    out = dict(name=name, substeps=sub, seconds=time.perf_counter()-t0, rows=rows)
    if contact and len(probes) == frames+1:
        S = torch.stack(probes).double().cpu().numpy(); S = S-S[:1]+T[:1]; S = S[:, has]; Tt = T[:, has]
        den = np.mean((Tt[1:]-Tt[:1])**2)
        nr = lambda lo, hi: float(np.sqrt(np.mean((S[lo:hi+1]-Tt[lo:hi+1])**2)/den))
        out.update(nrmse_1_10=nr(1, 10), nrmse_11_13=nr(11, 13), nrmse_1_13=nr(1, 13), nrmse_11_15=nr(11, 15))
        print('   nRMSE 1-10 %.3f 11-13 %.3f 1-13 %.3f 11-15 %.3f' % (out['nrmse_1_10'], out['nrmse_11_13'], out['nrmse_1_13'], out['nrmse_11_15']), flush=True)
    return out


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--geometry', default='output/pacnerf_vem/elastic_0_f0/sfvem_cells.npz')
    a.add_argument('--variants', nargs='+', default=['A', 'B1', 'B3', 'C', 'D'])
    a.add_argument('--out', type=Path, default=Path('output/pacnerf_vem/elastic_0_f0/sfvem_diag_v1.json'))
    args = a.parse_args()
    gt = json.loads(Path('data/pacnerf/gt/elastic.json').read_text())['0']
    dem = Path('output/pacnerf_dem/elastic_0_f0')
    bodies = dict(np.load(dem/'geometry.npz')); cells = dict(np.load(dem/'cells.npz'))
    T, has, _ = cell_targets(cells, bodies['body_of_cell'], bodies['points'], Path('data/pacnerf/simulation_data/elastic/0'), 15)
    spec = dict(A=('nh', 0., True, 15), B1=('nh', 1e-4, True, 15), B3=('nh', 3e-4, True, 15), C=('snh', 0., True, 15), D=('nh', 0., False, 6),
                P2=('nh', 0., True, 15), P1=('nh', 0., True, 15))
    extra = dict(P2=dict(ground='penalty', omega_c=2000.), P1=dict(ground='penalty', omega_c=1000.))
    results = []
    for v in args.variants:
        results.append(run(v, *spec[v], args.geometry, gt, bodies, cells, T, has, **extra.get(v, {})))
        args.out.write_text(json.dumps(results, indent=1))


if __name__ == '__main__':
    main()
