"""The oblique-bounce check of mpm_oblique.py repeated with our model (default: the bonded model) and the Coulomb ground
(WarpFoam contact='coulomb'): same ball (cubic lattice, R 3.5 cm), same impact (1.0 m/s sideways, 1.5 m/s down), fixed
E* = E / (1 - nu^2), no damping (theta[:, 6] = -20, as the MPM has none). Reports the free-flight state after the bounce
(centroid velocity, spin of the best-fit rotation, peak centroid height) and the contact time, next to the MPM results
(output/mpm_oblique/results.json) when present."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import warp as wp

from experiments.foam_dem_warp import WarpFoam, rollout as warp_rollout
from experiments.nu_signal_sphere import lattice_ball
from experiments.pacnerf_dem_forward import build


def rigid(p, X):
    c = p.mean(0); a = X-X.mean(0); b = p-c
    U, S, Vt = np.linalg.svd(b.T@a); D = np.diag([1, 1, np.sign(np.linalg.det(U@Vt))])
    return c, U@D@Vt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', default='old', choices=('old', 'cosserat', 'clean', 'dilatation', 'truevol'))
    ap.add_argument('--R', type=float, default=.035)
    ap.add_argument('--cell', type=float, default=.0035)
    ap.add_argument('--Estar', type=float, default=5e4)
    ap.add_argument('--nus', type=float, nargs='+', default=[.3, .4, .45, .49])
    ap.add_argument('--mus', type=float, nargs='+', default=[1., .3])
    ap.add_argument('--v', type=float, nargs=2, default=[1.0, -1.5])
    ap.add_argument('--layer-dx', type=float, default=.002)
    ap.add_argument('--fps', type=float, default=1000.)
    ap.add_argument('--frames', type=int, default=250)
    ap.add_argument('--out', type=Path, default=Path('output/model_oblique'))
    args = ap.parse_args()
    wp.init()
    args.out.mkdir(parents=True, exist_ok=True)
    dev = 'cuda'
    g = lattice_ball(args.R, args.cell, prune=True); X = g['points']        # no singular neighbourhoods
    kw = dict(kinematics='exact', shell=True, cosserat=True) if args.model == 'cosserat' else dict(shell=True, clean=True) if args.model == 'clean' else dict(dilatation=True) if args.model == 'dilatation' else dict(nodalvol=True) if args.model == 'truevol' else {}
    mpm = {}
    mp = Path('output/mpm_oblique/results.json')
    if mp.exists():
        for r in json.loads(mp.read_text())['results']:
            mpm[(round(r['mu_f'], 2), round(r['nu'], 2))] = r
    rows = []
    for mu_f in args.mus:
        for nu in args.nus:
            E = args.Estar*(1-nu*nu)
            m32 = build(g, E, nu, dev, 'noslip', .25, .3, torch.float32, **kw)
            m64 = build(g, E, nu, dev, 'noslip', .25, .3, torch.float64, **kw)
            for m in (m32, m64):
                m.set_environment(gravity=[0, 0, -9.8], ground_point=[0, 0, 0], ground_normal=[0, 0, 1],
                                  contact_radius=m.contact_radius.cpu().numpy(), ground_mode='noslip')
            th = torch.zeros(m32.n, 9, dtype=torch.float32, device=dev); th[:, 6] = -20.     # no damping
            sub = int(np.ceil(1/args.fps/m64.stable_dt(th[:, :7].double(), safety=.4, include_damping=False)))
            wf = WarpFoam(m32, contact='coulomb', layer_dx=args.layer_dx, friction0=mu_f, implicit_damping=True, graphs=True)
            x0, q0, _, w0 = m32.initial_state()
            r_c = float(m32.contact_radius.max())
            x0 = x0+torch.tensor([0., 0., args.R+r_c+.0015], device=dev)
            v = torch.tensor([args.v[0], 0., args.v[1]], device=dev).expand_as(x0).contiguous()
            with torch.no_grad():
                pos, quat = warp_rollout(wf, th, (x0.contiguous(), q0.contiguous(), v, w0.contiguous()), 1/args.fps/sub, sub, args.frames)
            P = pos.double().cpu().numpy()
            zmin = P[:, :, 2].min(1)
            touch = np.flatnonzero(zmin <= r_c+1.5*args.layer_dx)            # contact weight W > 0
            if not len(touch):
                print('no contact: closest approach %.2f mm above the contact radius, finite %s' % (1e3*(zmin.min()-r_c), bool(np.isfinite(P).all())), flush=True)
                continue
            t_on = touch.min()
            gaps = np.flatnonzero(np.diff(touch) > 1)                    # end of the first contact (a later landing may follow)
            t_off = touch[gaps[0]] if len(gaps) else touch.max()
            # free flight after the bounce: centroid velocity and spin from the angular momentum (w = I^-1 L, averaged
            # over the flight); a best-fit rotation between two distant frames aliases once the angle exceeds pi
            mb = m32.mass.double().cpu().numpy(); Mt = mb.sum()
            V = np.gradient(P, axis=0)*args.fps
            nxt = touch[touch > t_off+1]
            sel = np.arange(t_off+10, (nxt.min()-2) if len(nxt) else len(P)-1)
            ws, vs = [], []
            for k in sel:
                cc = (mb[:, None]*P[k]).sum(0)/Mt; vc = (mb[:, None]*V[k]).sum(0)/Mt; rr = P[k]-cc
                Lk = (mb[:, None]*np.cross(rr, V[k]-vc)).sum(0)
                Ik = (mb*(rr*rr).sum(1)).sum()*np.eye(3)-(mb[:, None]*rr).T@rr
                ws.append(np.linalg.solve(Ik, Lk)); vs.append(vc)
            w = np.mean(ws, 0); vel = np.mean(vs, 0)
            peak = max(rigid(P[k], X)[0][2] for k in range(t_off, sel[-1]+1, 5))
            r = dict(mu_f=mu_f, nu=nu, substeps_per_ms=sub, closest_mm=1e3*float(zmin.min()-r_c), contact_ms=float(t_off-t_on+1)*1e3/args.fps, vx_after=float(vel[0]),
                     spin_y=float(w[1]), peak_mm=1e3*float(peak), finite=bool(np.isfinite(P).all()))
            ref = mpm.get((round(mu_f, 2), round(nu, 2)))
            rows.append(dict(**r, mpm=ref))
            print('mu_f %.1f nu %.2f: closest %.2f mm, contact %.0f ms | vx after %.4f m/s (MPM %s), spin_y %.2f rad/s (MPM spin_z %s), peak centroid %.1f mm (MPM %s)' % (
                mu_f, nu, r['closest_mm'], r['contact_ms'], r['vx_after'], '%.4f' % ref['vx_after'] if ref else '-', r['spin_y'],
                '%.2f' % ref['spin_z'] if ref else '-', r['peak_mm'], '%.1f' % ref['peak_height_mm'] if ref else '-'), flush=True)
            (args.out/('results_%s.json' % args.model)).write_text(json.dumps(rows, indent=1))


if __name__ == '__main__':
    main()
