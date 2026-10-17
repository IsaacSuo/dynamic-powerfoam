"""Energy budget of the burger drop (companion of springgaus_drop_test): where does the impact
kinetic energy go?

Steps the Warp DEM directly (WarpFoam.step, ping-pong buffers, no tape) from the fitted initial
state onto the measured table with smooth contact, and reads the exact body velocities every
--sample seconds. Per sample (all per unit density, J/kg-like units; masses are volumes):
  KE_com      1/2 M |v_com|^2                        (coherent motion)
  KE_internal sum 1/2 m |v - v_com|^2                 (bodies moving relative to the COM)
  KE_rot      sum 1/2 I |w|^2                         (body spins)
  E_elastic   EnergyFoam.energy(theta, x, q)          (bonds + Neo-Hookean bodies)
  E_grav      -M g . x_com                            (relative to the start)
With damping and friction off the total should be conserved up to the contact layer and the
integrator. Also the object's height (along gravity) relative to the start, to see whether the
shape comes back after the bounce.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import warp as wp

from experiments.foam_dem_warp import WarpFoam
from experiments.pacnerf_dem_forward import build

OUT = Path('data/springgaus_powerfoam')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--obj', default='burger')
    p.add_argument('--dem', type=Path, default=Path('output/springgaus_dem/burger_spacing8k_mf0.1'))
    p.add_argument('--trained', type=Path, default=Path('output/springgaus_dem/burger_spacing8k_mf0.1/material_shared_smooth/report.json'))
    p.add_argument('--registration', default='initial_state_upright_nospin.json')
    p.add_argument('--E', type=float, default=2e4)
    p.add_argument('--kinematics', default='small', choices=('small', 'exact'), help='bond strain measures (BOND_EXACT.md)')
    p.add_argument('--duration', type=float, default=.2)
    p.add_argument('--sample', type=float, default=1/1200)
    p.add_argument('--out', type=Path, default=None)
    a = p.parse_args()
    a.out = a.out or Path('output/springgaus_dem/burger_spacing8k_mf0.1')/(Path(__file__).stem.replace('springgaus_', '')+('' if a.kinematics == 'small' else '_exact'))
    a.out.mkdir(parents=True, exist_ok=True)
    wp.init()
    dev, f32 = 'cuda', torch.float32
    dyn = OUT/f'{a.obj}_dynamic'
    meta = json.loads((dyn/'cameras.json').read_text())
    reg = json.loads((dyn/'register_calib'/a.registration).read_text())
    ground_z = json.loads((dyn/'rest'/'ground_rim.json').read_text())['plane_z_at_centre']
    g = dict(np.load(a.dem/'geometry.npz'))
    tr = json.loads(a.trained.read_text())['best']['parameters']
    grav = torch.tensor(meta['gravity'], dtype=f32, device=dev)
    gdir = grav/grav.norm()
    runs = [('all_off', 1., -30., -30., tr[8], 1.), ('all_off', 5., -30., -30., tr[8], 1.),
            ('baseline', 1., tr[6], tr[7], tr[8], 4.)]
    results = {}
    for name, fac, damp, fric, rest, inertia in runs:
        m = build(g, a.E, .3, dev, 'noslip', .25, .3, f32, inertia_scale=inertia, kinematics=a.kinematics)
        m.set_environment(gravity=meta['gravity'], ground_point=[0, 0, ground_z], ground_normal=meta['ground_normal'],
                          contact_radius=m.contact_radius.cpu().numpy(), ground_mode='noslip')
        wf = WarpFoam(m, contact='smooth', layer_dx=.002)
        theta = torch.zeros(m.n, 9, dtype=f32, device=dev)
        theta[:, :6] = float(np.log(fac)); theta[:, 6] = damp; theta[:, 7] = fric; theta[:, 8] = rest
        sub = int(np.ceil(a.sample/m.stable_dt(theta[:, :7])))
        dt = a.sample/sub
        x0, q0, _, w0 = m.initial_state()
        v0 = torch.tensor(reg['v0'], dtype=f32, device=dev).expand_as(x0).contiguous()
        dtypes = (wp.vec3, wp.vec4, wp.vec3, wp.vec3)
        s = tuple(wp.from_torch(t.detach().float().contiguous().clone(), dtype=d) for t, d in zip((x0, q0, v0, w0), dtypes))
        other = tuple(wp.zeros(m.n, dtype=d, device='cuda') for d in dtypes)
        buf = wf._buffers(False)
        th = wp.from_torch(theta.contiguous())
        M = m.mass.sum()
        cr = m.contact_radius
        rows = []

        def measure(t):
            x, q, v, w = (wp.to_torch(arr).clone() for arr in s)
            com = (m.mass[:, None]*x).sum(0)/M
            vcom = (m.mass[:, None]*v).sum(0)/M
            ke_com = .5*M*(vcom*vcom).sum()
            ke_int = .5*(m.mass*((v-vcom)**2).sum(-1)).sum()
            ke_rot = .5*(m.inertia*(w*w).sum(-1)).sum()
            e_el = m.energy(theta[:, :7], x.to(m.X.dtype), q.to(m.X.dtype))
            e_gr = -M*(grav@com)
            h = x@gdir
            height = float(((h+cr).max()-(h-cr).min()))
            rows.append(dict(t=t, KE_com=float(ke_com), KE_internal=float(ke_int), KE_rot=float(ke_rot), E_elastic=float(e_el),
                             E_grav=float(e_gr), vz_com=float(vcom@gdir), height=height,
                             touching=bool(((h+cr+.002) >= ground_z).any())))
        with torch.no_grad():
            measure(0.)
            n = int(round(a.duration/a.sample))
            for k in range(n):
                for _ in range(sub):
                    for b_ in buf.values():
                        b_.zero_()
                    wf.step(th, s, other, buf, dt)
                    s, other = other, s
                measure((k+1)*a.sample)
        R = {key: np.array([r[key] for r in rows]) for key in rows[0]}
        E0 = R['KE_com'][0]+R['KE_internal'][0]+R['KE_rot'][0]+R['E_elastic'][0]
        total = R['KE_com']+R['KE_internal']+R['KE_rot']+R['E_elastic']+R['E_grav']-R['E_grav'][0]
        hit = int(np.argmax(R['touching']))
        last = len(rows)-1
        # lift-off: last frame touching, then first not touching
        off = hit+int(np.argmin(R['touching'][hit:])) if not R['touching'][-1] else last
        imp = hit-1 if hit > 0 else 0
        shares = lambda i: {key: float(R[key][i]/(R['KE_com'][imp]+1e-12)) for key in ('KE_com', 'KE_internal', 'KE_rot', 'E_elastic')}
        summary = dict(setting=name, factor=fac, substeps_per_sample=sub, impact_index=imp, liftoff_index=off,
                       impact_KE_com=float(R['KE_com'][imp]),
                       at_liftoff_fraction_of_impact_KE=shares(off), at_end_fraction_of_impact_KE=shares(last),
                       total_energy_drift_fraction=float((total.max()-total.min())/(R['KE_com'][imp]+1e-12)),
                       height_at_liftoff_over_start=float(R['height'][off]/R['height'][0]),
                       height_at_end_over_start=float(R['height'][last]/R['height'][0]),
                       min_height_over_start=float(R['height'].min()/R['height'][0]))
        results[f'{name}_x{fac:g}'] = dict(summary=summary, series={k: v.tolist() for k, v in R.items()})
        f = lambda d: ' '.join('%s %.2f' % (k.replace('KE_', '').replace('E_', ''), v) for k, v in d.items())
        print('%-9s x%-3g | at lift-off (fraction of impact KE): %s | at end: %s | total-energy drift %.2f | height: min %.2f, lift-off %.2f, end %.2f' % (
            name, fac, f(summary['at_liftoff_fraction_of_impact_KE']), f(summary['at_end_fraction_of_impact_KE']),
            summary['total_energy_drift_fraction'], summary['min_height_over_start'], summary['height_at_liftoff_over_start'],
            summary['height_at_end_over_start']), flush=True)
    (a.out/'energy_budget.json').write_text(json.dumps(results))


if __name__ == '__main__':
    main()
