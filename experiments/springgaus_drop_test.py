"""Drop test: where does the burger DEM lose the energy that makes it rebound?

Forward-only Warp rollouts of the burger DEM from the fitted initial state (registration
initial_state_upright_nospin: v0, w0 = 0), measured table (rest/ground_rim.json), smooth contact,
sampled every 1/1200 s for 0.2 s. Per run:
  restitution  = max upward COM speed after impact / max downward COM speed before it,
  squash       = min over time of the object's height (body centres +/- contact radius, along
                 gravity) / its initial height; spread = max horizontal extent / initial,
  contact time = duration with any body within its contact radius (+ layer) of the table.
Grid: stiffness factor (all 6 stiffness multipliers, around E = --E) x mechanism switches
(baseline = the trained shared damping / friction / restitution and rotational inertia x4;
no damping; no friction; inertia x1; all off). Real targets from the captured masks (bounding-box
height / width at the most squashed frame relative to frame 0, per camera) and the triangulated
centroid speeds.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from experiments.foam_dem_warp import WarpFoam, rollout as warp_rollout
from experiments.pacnerf_dem_forward import build
from experiments.springgaus_dynamic import CAMS

OUT = Path('data/springgaus_powerfoam')


def real_targets(obj):
    dyn = OUT/f'{obj}_dynamic'
    out = {}
    for c in CAMS:
        hw = []
        for f in range(20):
            m = np.asarray(Image.open(dyn/'masks'/c/f'{f:02d}.png')) > 127
            ys, xs = np.nonzero(m)
            hw.append((np.ptp(ys)+1, np.ptp(xs)+1))
        hw = np.array(hw, float)
        k = int(np.argmin(hw[:, 0]/hw[:, 1]))
        out[c] = dict(most_squashed_frame=k, height_ratio=float(hw[k, 0]/hw[0, 0]), width_ratio=float(hw[k, 1]/hw[0, 1]))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--obj', default='burger')
    p.add_argument('--dem', type=Path, default=Path('output/springgaus_dem/burger_spacing8k_mf0.1'))
    p.add_argument('--trained', type=Path, default=Path('output/springgaus_dem/burger_spacing8k_mf0.1/material_shared_smooth/report.json'))
    p.add_argument('--registration', default='initial_state_upright_nospin.json')
    p.add_argument('--E', type=float, default=2e4)
    p.add_argument('--alpha', type=float, default=.25, help='share of mu carried by the bonds (build alpha)')
    p.add_argument('--settings', nargs='*', default=None, help='subset of the mechanism grid')
    p.add_argument('--kinematics', default='small', choices=('small', 'exact'), help='bond strain measures (BOND_EXACT.md)')
    p.add_argument('--factors', type=float, nargs='+', default=[.5, 1, 2, 5, 10])
    p.add_argument('--duration', type=float, default=.2)
    p.add_argument('--sample', type=float, default=1/1200)
    p.add_argument('--restitutions', type=float, nargs='*', default=[],
                   help='instead of the mechanism grid: contact restitution values x {trained friction, no friction}')
    p.add_argument('--tag', default='')
    p.add_argument('--out', type=Path, default=None)
    a = p.parse_args()
    a.out = a.out or Path('output/springgaus_dem/burger_spacing8k_mf0.1')/(Path(__file__).stem.replace('springgaus_', '')+('' if a.kinematics == 'small' else '_exact')+('' if a.alpha == .25 else f'_alpha{a.alpha:g}'))
    a.out.mkdir(parents=True, exist_ok=True)
    import warp as wp
    wp.init()
    dev, f32 = 'cuda', torch.float32
    dyn = OUT/f'{a.obj}_dynamic'
    meta = json.loads((dyn/'cameras.json').read_text())
    reg = json.loads((dyn/'register_calib'/a.registration).read_text())
    ground_z = json.loads((dyn/'rest'/'ground_rim.json').read_text())['plane_z_at_centre']
    g = dict(np.load(a.dem/'geometry.npz'))
    trained = json.loads(a.trained.read_text())['best']['parameters']
    gdir = np.array(meta['gravity'])/np.linalg.norm(meta['gravity'])
    targets = real_targets(a.obj)
    print('real:', json.dumps(targets), flush=True)

    settings = {
        'baseline': dict(damping=trained[6], friction=trained[7], restitution=trained[8], inertia=4.),
        'no_damping': dict(damping=-30., friction=trained[7], restitution=trained[8], inertia=4.),
        'no_friction': dict(damping=trained[6], friction=-30., restitution=trained[8], inertia=4.),
        'inertia_x1': dict(damping=trained[6], friction=trained[7], restitution=trained[8], inertia=1.),
        'all_off': dict(damping=-30., friction=-30., restitution=trained[8], inertia=1.),
    }
    if a.restitutions:
        logit = lambda q: float(np.log(q/(1-q)))
        settings = {}
        for e in a.restitutions:
            r8 = logit(e)-logit(.3)                      # WarpFoam restitution0 = 0.3
            settings[f'e{e:g}'] = dict(damping=trained[6], friction=trained[7], restitution=r8, inertia=4.)
            settings[f'e{e:g}_nofric'] = dict(damping=trained[6], friction=-30., restitution=r8, inertia=4.)
    if a.settings:
        settings = {k_: v_ for k_, v_ in settings.items() if k_ in a.settings}
    rows = []
    models = {}
    for name, s in settings.items():
        if s['inertia'] not in models:
            m = build(g, a.E, .3, dev, 'noslip', a.alpha, .3, f32, inertia_scale=s['inertia'], kinematics=a.kinematics)
            m.set_environment(gravity=meta['gravity'], ground_point=[0, 0, ground_z], ground_normal=meta['ground_normal'],
                              contact_radius=m.contact_radius.cpu().numpy(), ground_mode='noslip')
            models[s['inertia']] = (m, WarpFoam(m, contact='smooth', layer_dx=.002))
        m, wf = models[s['inertia']]
        x0, q0, _, w0 = m.initial_state()
        v0 = torch.tensor(reg['v0'], dtype=f32, device=dev).expand_as(x0).contiguous()
        mass = m.mass.to(f32)
        cr = m.contact_radius.cpu().numpy()
        for fac in a.factors:
            theta = torch.zeros(m.n, 9, dtype=f32, device=dev)
            theta[:, :6] = float(np.log(fac))
            theta[:, 6] = s['damping']; theta[:, 7] = s['friction']; theta[:, 8] = s['restitution']
            sub = int(np.ceil(a.sample/m.stable_dt(theta[:, :7])))
            n = int(round(a.duration/a.sample))
            with torch.no_grad():
                pos, _ = warp_rollout(wf, theta, (x0, q0, v0, w0), a.sample/sub, sub, n)
            with torch.no_grad():
                vol = torch.stack([(m.mass*torch.det(m.deformation_gradient(pos[k].to(m.X.dtype)))).sum()/m.mass.sum()
                                   for k in range(len(pos))]).cpu().numpy()
            P = pos.cpu().numpy()
            com = (P*mass.cpu().numpy()[None, :, None]).sum(1)/mass.sum().item()
            vz = np.gradient(com@gdir, a.sample)                     # + = towards the table
            h = P@gdir
            height = (h+cr).max(1)-(h-cr).min(1)
            horiz = P-h[..., None]*gdir
            width = np.array([np.ptp(hh, 0).max() for hh in horiz])
            touching = ((h+cr+.002) >= ground_z).any(1)
            penetration = float(((h+cr)-ground_z).max())          # > 0: contact sphere below the table
            centre_below = float((h-ground_z).max())              # > 0: a body centre below the table
            i_hit = int(np.argmax(touching)) if touching.any() else n
            v_in = float(vz[:i_hit+1].max())
            v_up = float(-vz[i_hit:].min()) if i_hit < n else 0.
            row = dict(setting=name, factor=fac, E_eff=a.E*fac, substeps_per_sample=sub,
                       restitution=v_up/max(v_in, 1e-9), v_in=v_in, v_up=v_up,
                       squash=float(height.min()/height[0]), spread=float(width.max()/width[0]),
                       contact_ms=float(touching.sum()*a.sample*1000), still_touching_at_end=bool(touching[-1]),
                       penetration_mm=1000*penetration, centre_below_table_mm=1000*centre_below,
                       min_mean_detF=float(np.nanmin(vol)), detF_at_max_squash=float(vol[int(np.argmin(height))]))
            rows.append(row)
            print('%-14s E x%-4g restitution %.2f (in %.2f up %.2f m/s) squash %.2f spread %.2f contact %.1f ms | penetration %.1f mm, centre below %.1f mm, mean detF min %.2f%s' % (
                name, fac, row['restitution'], v_in, v_up, row['squash'], row['spread'], row['contact_ms'],
                row['penetration_mm'], row['centre_below_table_mm'], row['min_mean_detF'],
                ' (still touching)' if row['still_touching_at_end'] else ''), flush=True)
            np.save(a.out/f'com_{name}_x{fac:g}.npy', com)
    (a.out/f'drop_test{a.tag}.json').write_text(json.dumps(dict(real=targets, rows=rows, ground_z=ground_z, E=a.E,
                                                        trained=trained, settings=settings), indent=1))


if __name__ == '__main__':
    main()
