"""Fit VEM (E, nu, beta) to the GT 3D trajectories of frames 1-10 (pure motion, no rendering).

Same targets and score as the DEM (pacnerf_dem_train.py): GT particles -> DEM body, target =
particle mean; prediction = the body's reference centre carried by its element's affine map;
motion relative to frame 0. Loss = squared nRMSE over the training frames. Adam on
z = (log E/E0, nu - nu0, log beta/beta0), best iterate by training loss, substeps re-estimated
(dt = 1/omega_max) every 5 iterations. Reports nRMSE over frames 1-10 / 11-13 / 1-13 (silhouette
protocol) and 11-15 / 1-15 (the DEM trajectory runs' split).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.pacnerf_dem_forward import cell_targets
from experiments.vem_dynamics import VEMSolid


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--scene', default='elastic/0')
    a.add_argument('--dem', type=Path, default=Path('output/pacnerf_dem/elastic_0_f0'))
    a.add_argument('--geometry', type=Path, default=Path('output/pacnerf_vem/elastic_0_f0/vem_cap1_bodies_pyr.npz'))
    a.add_argument('--frames', type=int, default=15)
    a.add_argument('--train-frames', type=int, default=10)
    a.add_argument('--iterations', type=int, default=40)
    a.add_argument('--lr', type=float, default=.05)
    a.add_argument('--E0', type=float, default=None)
    a.add_argument('--nu0', type=float, default=None)
    a.add_argument('--beta0', type=float, default=4.)
    a.add_argument('--tau0', type=float, default=0., help='Kelvin-Voigt damping time (s), learned in log; 0: no damping')
    a.add_argument('--model', default='pyrlin', choices=('pyrlin', 'pyrtan', 'tet'), help="'tet': linear tet sub-mesh, NH, (E, nu) only")
    a.add_argument('--tet-geometry', default='output/pacnerf_vem/elastic_0_f0/tet_cells.npz')
    a.add_argument('--mass-file', default='output/pacnerf_vem/elastic_0_f0/tet_mass_scaled.npz')
    a.add_argument('--out', type=Path, required=True)
    args = a.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    gt = json.loads(Path('data/pacnerf/gt/elastic.json').read_text())[args.scene.split('/')[1]]
    bodies = dict(np.load(args.dem/'geometry.npz')); cells = dict(np.load(args.dem/'cells.npz'))
    targets, has, _ = cell_targets(cells, bodies['body_of_cell'], bodies['points'],
                                   Path('data/pacnerf/simulation_data')/args.scene, args.frames)
    if args.model == 'tet':
        from experiments.tet_dynamics import TetSolid
        m = TetSolid(args.tet_geometry, 'cuda', torch.float32, cluster_tol=5e-4, cells=(bodies['body_of_cell'], cells['masses']),
                     mass_file=args.mass_file or None)
    else:
        m = VEMSolid(args.geometry, 'cuda', torch.float32, stab=args.model)
    m.ground_mode = 'noslip'
    P = torch.as_tensor(bodies['points'], dtype=torch.float32, device='cuda'); elem = torch.arange(len(P), device='cuda')
    v0 = torch.tensor(gt['v'], dtype=torch.float32, device='cuda').expand(m.nv, 3).clone()
    T = torch.as_tensor(targets, dtype=torch.float32, device='cuda'); H = torch.as_tensor(has, device='cuda')
    E0 = args.E0 or gt['E']; nu0 = args.nu0 if args.nu0 is not None else gt['nu']; b0 = args.beta0
    nz = 4 if args.tau0 > 0 else (2 if args.model in ('tet', 'pyrtan') else 3)   # pyrtan: kappa = beta0 fixed
    z = torch.zeros(nz, dtype=torch.float32, device='cuda', requires_grad=True)
    opt = torch.optim.Adam([z], lr=args.lr)

    def params(zz):
        b = b0*torch.exp(zz[2]) if nz >= 3 else torch.tensor(b0, device=zz.device)
        p = (E0*torch.exp(zz[0]), (nu0+zz[1]).clamp(0., .45), b)
        return p+(args.tau0*torch.exp(zz[3]),) if nz == 4 else p

    def readable(zz):
        p = [float(t) for t in params(zz.detach())]
        r = dict(E=p[0], nu=p[1], beta=p[2], E_over_gt=p[0]/gt['E'], nu_minus_gt=p[1]-gt['nu'])
        if nz == 4:
            r['tau'] = p[3]
        return r

    def substeps_for(zz):
        p = tuple(t.detach() for t in params(zz))
        w = m.max_frequency(p[:3])
        zeta = float(p[3])*w/2 if nz == 4 else 0.
        return int(np.ceil(w/(24*(np.sqrt(1+zeta**2)-zeta))))   # half the damped explicit limit

    def positions(frames, sub, grad=True):
        p = params(z if grad else z.detach())
        pos, _ = m.rollout(p, m.X, v0, 1/24/sub, sub, frames, P=P, elem=elem, compile=True)
        return pos-pos[:1]+T[:1]

    def nrmse_t(pos, lo, hi):
        pp, tt = pos[lo:hi+1][:, H], T[lo:hi+1][:, H]
        return (((pp-tt)**2).mean()/((tt-T[:1][:, H])**2).mean()).sqrt()

    def evaluate(sub):
        with torch.no_grad():
            pos = positions(args.frames, sub, grad=False)
        r = {f'nrmse_{lo}_{hi}': float(nrmse_t(pos, lo, hi)) for lo, hi in
             ((1, args.train_frames), (args.train_frames+1, 13), (1, 13), (args.train_frames+1, args.frames), (1, args.frames))}
        r.update(parameters=readable(z), substeps=sub,
                 centroid_y=pos[:, H, 1].mean(1).tolist(), centroid_y_gt=T[:, H, 1].mean(1).tolist())
        return r

    sub = substeps_for(z)
    report = dict(config={k: str(v) for k, v in vars(args).items()}, initial=evaluate(sub))
    print('initial', json.dumps({k: v for k, v in report['initial'].items() if not k.startswith('centroid')}), flush=True)
    history, best = [], (np.inf, z.detach().clone())
    start = time.perf_counter()
    for it in range(args.iterations):
        if it and it % 5 == 0:
            sub = substeps_for(z)
        opt.zero_grad()
        pos = positions(args.train_frames, sub)
        loss = nrmse_t(pos, 1, args.train_frames)**2
        value = float(loss.detach())
        loss.backward()
        if not np.isfinite(value) or not torch.isfinite(z.grad).all():
            print('non-finite at iteration', it, flush=True); break
        if value < best[0]:
            best = (value, z.detach().clone())
        opt.step()
        history.append(dict(iteration=it, train_nrmse=value**.5, grad=z.grad.tolist(), parameters=readable(z),
                            substeps=sub, seconds=time.perf_counter()-start))
        print(json.dumps(history[-1]), flush=True)
        (args.out/'history.json').write_text(json.dumps(history, indent=1))
    with torch.no_grad():
        z.copy_(best[1])
    report['best'] = evaluate(substeps_for(z))
    report['train_seconds'] = time.perf_counter()-start
    (args.out/'report.json').write_text(json.dumps(report, indent=1))
    print('best', json.dumps({k: v for k, v in report['best'].items() if not k.startswith('centroid')}), flush=True)


if __name__ == '__main__':
    main()
