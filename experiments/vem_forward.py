"""VEM on the reconstructed PAC-NeRF elastic/0 object with the GT material, no training.

Same targets and score as the DEM (pacnerf_dem_forward.py): GT particles -> power cell -> DEM
body (one VEM element per body), target = particle mean; prediction = the body's reference
centre carried by its element's affine map; normalized RMSE of motion relative to frame 0,
reported over frames 1-10 (training), 11-13 (held out) and 1-13, as in the silhouette runs.
Initial state: reference vertices, uniform GT velocity. dt = safety x 2 / omega_max (reference).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.pacnerf_dem_forward import cell_targets
from experiments.vem_dynamics import VEMSolid


def nrmse(P, targets, has, lo, hi):
    pp, tt = P[lo:hi+1, has], targets[lo:hi+1, has]
    return float(np.sqrt(np.mean((pp-tt)**2)/np.mean((tt-targets[:1, has])**2)))


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--scene', default='elastic/0')
    a.add_argument('--geometry', type=Path, default=Path('output/pacnerf_vem/elastic_0_f0/vem_cap1_bodies.npz'))
    a.add_argument('--dem', type=Path, default=Path('output/pacnerf_dem/elastic_0_f0'))
    a.add_argument('--gamma', '--stab-param', type=float, nargs='+', default=None,
                   help="gamma (vertex penalty) or beta (pyramid geometry); default .1 .3 1 3 / .2 .4 .6 1")
    a.add_argument('--frames', type=int, default=13)
    a.add_argument('--safety', type=float, default=.5)
    a.add_argument('--dtype', default='float32', choices=('float32', 'float64'))
    a.add_argument('--compile', action='store_true')
    a.add_argument('--ground', default='noslip', choices=('layer', 'noslip'))
    a.add_argument('--stab', default=None, choices=(None, 'gamma', 'pyramid', 'pyrlin', 'pyrtan'))
    a.add_argument('--out', type=Path, default=Path('output/pacnerf_vem/elastic_0_f0/forward'))
    args = a.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    gt = json.loads(Path('data/pacnerf/gt/elastic.json').read_text())[args.scene.split('/')[1]]
    bodies = dict(np.load(args.dem/'geometry.npz')); cells = dict(np.load(args.dem/'cells.npz'))
    targets, has, _ = cell_targets(cells, bodies['body_of_cell'], bodies['points'],
                                   Path('data/pacnerf/simulation_data')/args.scene, args.frames)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = dict(float32=torch.float32, float64=torch.float64)[args.dtype]
    m = VEMSolid(args.geometry, dev, dtype, stab=args.stab)
    m.ground_mode = args.ground
    P = torch.as_tensor(bodies['points'], dtype=dtype, device=dev)
    elem = torch.arange(len(P), device=dev)          # element id = DEM body id
    v0 = torch.as_tensor(gt['v'], dtype=dtype, device=dev).expand(m.nv, 3).clone()
    rows = []
    if args.gamma is None:
        args.gamma = {'pyramid': [.2, .4, .6, 1.], 'pyrlin': [.5, 1., 2., 4.]}.get(m.stab, [.1, .3, 1., 3.])
    tag = f'{m.stab}_{args.ground}_{args.geometry.stem}_{args.dtype}'
    for gamma in args.gamma:
        params = (gt['E'], gt['nu'], gamma)
        omega = m.max_frequency(tuple(torch.tensor(p, dtype=dtype, device=dev) for p in params))
        substeps = int(np.ceil(1/24/(args.safety*2/omega)))
        torch.cuda.synchronize() if dev == 'cuda' else None
        t0 = time.perf_counter()
        with torch.no_grad():
            pos, _ = m.rollout(params, m.X, v0, 1/24/substeps, substeps, args.frames, P=P, elem=elem, compile=args.compile)
        torch.cuda.synchronize() if dev == 'cuda' else None
        seconds = time.perf_counter()-t0
        pos = pos.double().cpu().numpy()
        finite = bool(np.isfinite(pos).all())
        S = pos-pos[:1]+targets[:1]
        r = dict(gamma=gamma, omega_max=omega, substeps=substeps, seconds=seconds, finite=finite,
                 nrmse_1_10=nrmse(S, targets, has, 1, 10), nrmse_11_13=nrmse(S, targets, has, 11, args.frames),
                 nrmse_1_13=nrmse(S, targets, has, 1, args.frames),
                 per_frame_rms_m=np.sqrt(((S-targets)[:, has]**2).sum(-1).mean(-1)).tolist(),
                 centroid_y_pred=S[:, has, 1].mean(1).tolist(), centroid_y_gt=targets[:, has, 1].mean(1).tolist(),
                 min_y_pred=S[:, has, 1].min(1).tolist(), min_y_gt=targets[:, has, 1].min(1).tolist())
        print(json.dumps({k: r[k] for k in ('gamma', 'omega_max', 'substeps', 'seconds', 'finite',
                                            'nrmse_1_10', 'nrmse_11_13', 'nrmse_1_13')}), flush=True)
        rows.append(r)
        np.savez_compressed(args.out/f'{tag}_s{gamma:g}.npz', positions=pos, targets=targets, has=has)
        (args.out/f'forward_{tag}.json').write_text(json.dumps(rows, indent=1))


if __name__ == '__main__':
    main()
