"""Tet sub-mesh Neo-Hookean, GT E/nu, PAC-NeRF elastic/0 forward with diagnostics.

Per quarter frame: total energy (strain + kinetic + gravity), vmax, minimum tet J, and the
mesh-scale kinetic energy (rms residual speed of a mass-weighted local affine fit of vertex
velocities within 1 cm, as nonaffine_ke.py; MPM: 0.003-0.006 m/s). Per frame: DEM-body probes
-> nRMSE (frames 1-10 / 11-13 / 1-13 / 11-15) against the GT particle means.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.nonaffine_ke import measure
from experiments.pacnerf_dem_forward import cell_targets
from experiments.tet_dynamics import TetSolid


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--geometry', default='output/pacnerf_vem/elastic_0_f0/tet_cells.npz')
    a.add_argument('--mass-file', default='output/pacnerf_vem/elastic_0_f0/tet_mass_scaled.npz')
    a.add_argument('--tol', type=float, default=5e-4)
    a.add_argument('--factor', type=float, nargs='+', default=[1.], help='substeps x factor (dt = 1/(factor omega_max))')
    a.add_argument('--frames', type=int, default=15)
    a.add_argument('--law', default='nh')
    a.add_argument('--bulk-visc', type=float, default=0., help='scale of the artificial bulk viscosity (1 = LS-DYNA defaults)')
    a.add_argument('--out', type=Path, default=Path('output/pacnerf_vem/elastic_0_f0/tet_forward_v1.json'))
    args = a.parse_args()
    gt = json.loads(Path('data/pacnerf/gt/elastic.json').read_text())['0']
    dem = Path('output/pacnerf_dem/elastic_0_f0')
    bodies = dict(np.load(dem/'geometry.npz')); cells = dict(np.load(dem/'cells.npz'))
    T, has, _ = cell_targets(cells, bodies['body_of_cell'], bodies['points'], Path('data/pacnerf/simulation_data/elastic/0'), args.frames)
    Tt = T[:, has]; den = np.mean((Tt[1:]-Tt[:1])**2)
    m = TetSolid(args.geometry, 'cuda', torch.float32, cluster_tol=args.tol, cells=(bodies['body_of_cell'], cells['masses']),
                 law=args.law, mass_file=args.mass_file or None)
    if not m.ncl:                                          # no binding: identity clusters so the diagnostics below work
        m.cl = torch.arange(m.nv, device='cuda'); m.cmass = m.mass; m.ncl = 0
    p = tuple(torch.tensor(t, device='cuda') for t in (gt['E'], gt['nu'], 1.))
    w = m.max_frequency(p, iterations=300)
    if args.bulk_visc > 0:
        p = p+(torch.tensor(args.bulk_visc, device='cuda'),)
    print(f'added mass {m.added_mass_fraction*100:.4f}%, omega_max {w:.0f}', flush=True)
    vmass = m.cmass[m.cl]/torch.bincount(m.cl, minlength=len(m.cmass))[m.cl] if m.ncl else m.mass
    step = torch.compile(m.step, dynamic=False)
    results = []
    for fac in args.factor:
        sub = int(np.ceil(fac*w/24)); dt = torch.tensor(1/24/sub, device='cuda')
        x, v = m.X.clone(), torch.tensor(gt['v'], device='cuda').expand(m.nv, 3).clone()
        probes = [m.probe(x)]; rows = []
        t0 = time.perf_counter()
        print(f'== dt = 1/({fac:g} omega_max): {sub} substeps/frame', flush=True)
        ok = True
        with torch.no_grad():
            for n in range(1, args.frames*sub+1):
                x, v = step(x, v, p, dt)
                if n % sub == 0:
                    if not torch.isfinite(x).all():
                        print(f'   NaN by frame {n/sub:.0f}', flush=True); ok = False; break
                    probes.append(m.probe(x))
                if n % (sub//4) == 0 and n/sub >= 4.5 and n/sub <= 9.01 and torch.isfinite(x).all():
                    E = float(m.energy(x, p[:3])); K = float(.5*(vmass*(v*v).sum(1)).sum()); G = float((vmass*9.8*x[:, 1]).sum())
                    J = m.tet_J(x)
                    na = measure(x, v, vmass, f'frame {n/sub:.2f}', nsample=8000)
                    rows.append(dict(frame=n/sub, total=E+K+G, strain=E, kinetic=K, Jmin=float(J.min()), J_below_0p2=int((J < .2).sum()), **na))
                    print(f'   f{n/sub:5.2f} total {E+K+G:8.3f} strain {E:7.3f} kin {K:7.3f} Jmin {float(J.min()):6.3f} (n<0.2: {int((J < .2).sum())})', flush=True)
        r = dict(factor=fac, substeps=sub, seconds=time.perf_counter()-t0, finite=ok, rows=rows)
        if ok:
            S = torch.stack(probes).double().cpu().numpy(); S = S-S[:1]+T[:1]; S = S[:, has]
            nr = lambda lo, hi: float(np.sqrt(np.mean((S[lo:hi+1]-Tt[lo:hi+1])**2)/den))
            cy = S[:, :, 1].mean(1)
            r.update(nrmse_1_10=nr(1, 10), nrmse_11_13=nr(11, 13), nrmse_1_13=nr(1, 13), nrmse_11_15=nr(11, 15), nrmse_1_15=nr(1, 15),
                     rebound_peak=float(cy[8:].max()), rebound_peak_gt=float(Tt[8:, :, 1].mean(1).max()))
            print('   nRMSE 1-10 %.3f 11-13 %.3f 1-13 %.3f 11-15 %.3f | rebound %.3f (GT %.3f) | %.0fs' % (
                r['nrmse_1_10'], r['nrmse_11_13'], r['nrmse_1_13'], r['nrmse_11_15'], r['rebound_peak'], r['rebound_peak_gt'], r['seconds']), flush=True)
        results.append(r)
        args.out.write_text(json.dumps(dict(added_mass=m.added_mass_fraction, omega_max=w, runs=results), indent=1))


if __name__ == '__main__':
    main()
