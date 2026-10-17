"""Why does the soft burger DEM lock elastic energy after the bounce? (follow-up of
springgaus_energy_budget). Same drop (all dissipation off), and at a few instants:
  - relative rotation angle between bonded bodies (geodesic angle of R_i^T R_j; 0 at rest),
    fraction above 45 / 90 / 135 deg -- the bond twist term uses phi = vee(R_i^T R_j - R_j^T R_i)/2,
    |phi| = sin(angle), so its energy has a second minimum at 180 deg;
  - where the elastic energy sits: bond normal / shear / bend / twist and body Neo-Hookean terms;
  - body orientation vs the local continuum rotation (polar factor of the least-squares F).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import warp as wp

from experiments.foam_dem_warp import WarpFoam
from experiments.foam_deform_render import polar_rotation
from experiments.foam_dynamic_render import quat_mul  # noqa: F401  (kept for parity with other tools)
from experiments.pacnerf_dem_forward import build
from experiments.foam_dem_energy import quat_matrix

OUT = Path('data/springgaus_powerfoam')


def angle(R):
    return torch.rad2deg(torch.arccos(((R.diagonal(dim1=-2, dim2=-1).sum(-1)-1)/2).clamp(-1, 1)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dem', type=Path, default=Path('output/springgaus_dem/burger_spacing8k_mf0.1'))
    p.add_argument('--trained', type=Path, default=Path('output/springgaus_dem/burger_spacing8k_mf0.1/material_shared_smooth/report.json'))
    p.add_argument('--factors', type=float, nargs='+', default=[1., 5.])
    p.add_argument('--E', type=float, default=2e4)
    p.add_argument('--kinematics', default='small', choices=('small', 'exact'), help='bond strain measures (BOND_EXACT.md)')
    p.add_argument('--sample', type=float, default=1/1200)
    p.add_argument('--checkpoints', type=int, nargs='+', default=[0, 71, 80, 87, 100, 140, 240])
    p.add_argument('--out', type=Path, default=None)
    a = p.parse_args()
    a.out = a.out or Path('output/springgaus_dem/burger_spacing8k_mf0.1')/(Path(__file__).stem.replace('springgaus_', '')+('' if a.kinematics == 'small' else '_exact'))
    a.out.mkdir(parents=True, exist_ok=True)
    wp.init()
    dev, f32 = 'cuda', torch.float32
    dyn = OUT/'burger_dynamic'
    meta = json.loads((dyn/'cameras.json').read_text())
    reg = json.loads((dyn/'register_calib'/'initial_state_upright_nospin.json').read_text())
    ground_z = json.loads((dyn/'rest'/'ground_rim.json').read_text())['plane_z_at_centre']
    g = dict(np.load(a.dem/'geometry.npz'))
    tr = json.loads(a.trained.read_text())['best']['parameters']
    out = {}
    for fac in a.factors:
        m = build(g, a.E, .3, dev, 'noslip', .25, .3, f32, inertia_scale=1., kinematics=a.kinematics)
        m.set_environment(gravity=meta['gravity'], ground_point=[0, 0, ground_z], ground_normal=meta['ground_normal'],
                          contact_radius=m.contact_radius.cpu().numpy(), ground_mode='noslip')
        wf = WarpFoam(m, contact='smooth', layer_dx=.002)
        theta = torch.zeros(m.n, 9, dtype=f32, device=dev)
        theta[:, :6] = float(np.log(fac)); theta[:, 6] = -30.; theta[:, 7] = -30.; theta[:, 8] = tr[8]
        sub = int(np.ceil(a.sample/m.stable_dt(theta[:, :7])))
        x0, q0, _, w0 = m.initial_state()
        v0 = torch.tensor(reg['v0'], dtype=f32, device=dev).expand_as(x0).contiguous()
        dtypes = (wp.vec3, wp.vec4, wp.vec3, wp.vec3)
        s = tuple(wp.from_torch(t.detach().float().contiguous().clone(), dtype=d) for t, d in zip((x0, q0, v0, w0), dtypes))
        other = tuple(wp.zeros(m.n, dtype=d, device='cuda') for d in dtypes)
        buf = wf._buffers(False)
        th = wp.from_torch(theta.contiguous())
        rows = []
        with torch.no_grad():
            for k in range(max(a.checkpoints)+1):
                if k in a.checkpoints:
                    x = wp.to_torch(s[0]).clone().to(m.X.dtype); q = wp.to_torch(s[1]).clone().to(m.X.dtype)
                    R = quat_matrix(q)
                    rel = angle(R[m.i].transpose(1, 2)@R[m.j])
                    absr = angle(R)
                    Rc = polar_rotation(m.deformation_gradient(x))
                    mis = angle(Rc.transpose(1, 2)@R)                      # body vs local continuum rotation
                    _, _, _, _, _, _, kk, dn, dt_, pn, pt = m._bond_kinematics(theta[:, :7], x, R)
                    terms = dict(normal=.5*(kk[:, 0]*dn**2).sum(), shear=.5*(kk[:, 1]*(dt_*dt_).sum(-1)).sum(),
                                 bend=.5*(kk[:, 2]*(pt*pt).sum(-1)).sum(), twist=.5*(kk[:, 3]*pn**2).sum(),
                                 body=m._body_energy_and_stress(theta[:, :7], x)[0].sum())
                    row = dict(sample=k, t_ms=1000*k*a.sample,
                               rel_angle_pct=dict(p50=float(rel.median()), p99=float(torch.quantile(rel.float(), .99)), max=float(rel.max())),
                               rel_over_45=float((rel > 45).float().mean()), rel_over_90=float((rel > 90).float().mean()),
                               rel_over_135=float((rel > 135).float().mean()),
                               body_abs_rot_median=float(absr.median()), body_vs_continuum_median=float(mis.median()),
                               body_vs_continuum_over_90=float((mis > 90).float().mean()),
                               energy={k_: float(v) for k_, v in terms.items()})
                    rows.append(row)
                    e = row['energy']
                    print('x%-3g t=%5.1f ms | bond rel. angle median %.1f p99 %.1f max %.1f deg, >45: %.3f >90: %.3f >135: %.3f | body vs continuum rot: median %.1f, >90: %.3f | energy normal %.3g shear %.3g bend %.3g twist %.3g body %.3g' % (
                        fac, row['t_ms'], row['rel_angle_pct']['p50'], row['rel_angle_pct']['p99'], row['rel_angle_pct']['max'],
                        row['rel_over_45'], row['rel_over_90'], row['rel_over_135'], row['body_vs_continuum_median'],
                        row['body_vs_continuum_over_90'], e['normal'], e['shear'], e['bend'], e['twist'], e['body']), flush=True)
                if k == max(a.checkpoints):
                    break
                for _ in range(sub):
                    for b_ in buf.values():
                        b_.zero_()
                    wf.step(th, s, other, buf, a.sample/sub)
                    s, other = other, s
        out[f'x{fac:g}'] = rows
    (a.out/'rotation_check.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
