"""How much does nu show in a drop video? (IDENTIFIABILITY.md)

A cubic-lattice ball (radius R, cell a) dropped on the ground with the current DEM (bond-level Cosserat, Warp). Runs at a
fixed contact modulus E* = E / (1 - nu^2) for several nu, plus a control with nu fixed and E* changed by a few percent.
Compares, against the reference nu, three observations in a side camera (orthographic along y, 1 px = 2R / px_diameter):
  silhouette  - union of the bodies' equivalent spheres rasterised on a fine grid; mean boundary shift = XOR area / perimeter;
  texture     - visible surface bodies (y > 0 hemisphere) projected to the image: per-point 2D displacement (perfect tracking);
  3d          - all bodies, 3D position difference (in px units).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import warp as wp

from experiments.foam_dem_warp import WarpFoam, rollout as warp_rollout
from experiments.pacnerf_dem_forward import build


def lattice_ball(R, a, prune=False):
    """prune=True: iteratively drop surface voxels whose lattice neighbours do not span 3D (singular neighbour moment
    matrix; the averaged-F body law of the bonded model needs an invertible one)."""
    n = int(np.ceil(R/a))
    g = np.arange(-n, n+1)
    idx = np.array([(i, j, k) for i in g for j in g for k in g])
    pts = idx*a
    keep = np.linalg.norm(pts, axis=1) <= R-.5*a
    idx, pts = idx[keep], pts[keep]
    while prune:
        present = {tuple(t) for t in idx}
        axes = np.array([[any((t[0]+s*d[0], t[1]+s*d[1], t[2]+s*d[2]) in present for s in (1, -1)) for d in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
                         for t in idx])
        ok = axes.all(1)
        if ok.all():
            break
        idx, pts = idx[ok], pts[ok]
    lookup = {tuple(t): m for m, t in enumerate(idx)}
    pairs, contacts = [], []
    for m, t in enumerate(idx):
        for d in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            o = lookup.get((t[0]+d[0], t[1]+d[1], t[2]+d[2]))
            if o is not None:
                pairs.append((m, o)); contacts.append(.5*(pts[m]+pts[o]))
    return dict(points=pts, radii=np.full(len(pts), a*np.sqrt(1/2.4)), pairs=np.array(pairs), areas=np.full(len(pairs), a*a),
                contacts=np.array(contacts), masses=np.full(len(pts), a**3), fixed=np.zeros(len(pts), bool))


def silhouette(xz, r, lo, px, res):
    """Union of discs (centre xz, radius r) on a grid of `res` px per image px; returns a bool image."""
    H = W = int(np.ceil(res*(lo[2]/px)))
    img = np.zeros((H, W), bool)
    s = px/res
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cz = lo[0]+(xx+.5)*s, lo[1]+(yy+.5)*s
    for (x, z), rr in zip(xz, r):
        i0, i1 = int(max(0, (z-rr-lo[1])/s)), int(min(H, (z+rr-lo[1])/s+2))
        j0, j1 = int(max(0, (x-rr-lo[0])/s)), int(min(W, (x+rr-lo[0])/s+2))
        img[i0:i1, j0:j1] |= (cx[i0:i1, j0:j1]-x)**2+(cz[i0:i1, j0:j1]-z)**2 <= rr*rr
    return img


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--R', type=float, default=.035)
    p.add_argument('--cell', type=float, default=.0035)
    p.add_argument('--Estar', type=float, default=5e4, help='contact modulus E/(1-nu^2), Pa at 1000 kg/m^3')
    p.add_argument('--v0', type=float, default=1.5, help='impact speed (m/s)')
    p.add_argument('--nus', type=float, nargs='+', default=[.3, .4, .45, .49])
    p.add_argument('--nu-ref', type=float, default=.45)
    p.add_argument('--estar-control', type=float, default=1.05, help='control run: nu = nu_ref, E* times this')
    p.add_argument('--fps', type=float, default=120.)
    p.add_argument('--frames', type=int, default=30)
    p.add_argument('--px-diameter', type=float, default=70., help='ball diameter in image px')
    p.add_argument('--model', default='cosserat', choices=('cosserat', 'truevol'),
                   help='cosserat: bond-level Cosserat (IDENTIFIABILITY.md sections 8-10); truevol: springs + nodal true volume')
    p.add_argument('--out', type=Path, default=Path('output/nu_signal_sphere'))
    args = p.parse_args()
    wp.init()
    args.out.mkdir(parents=True, exist_ok=True)
    dev = 'cuda'
    g = lattice_ball(args.R, args.cell)
    px = 2*args.R/args.px_diameter
    print('ball R %.3f m, cell %.4f m, %d bodies; 1 px = %.2f mm' % (args.R, args.cell, len(g['points']), 1e3*px), flush=True)
    runs = [('nu %.2f' % nu, nu, args.Estar) for nu in args.nus]+[('control: nu %.2f, E* x%.2f' % (args.nu_ref, args.estar_control), args.nu_ref, args.Estar*args.estar_control)]
    traj = {}
    for name, nu, Estar in runs:
        E = Estar*(1-nu*nu)
        kw = dict(kinematics='exact', shell=True, cosserat=True) if args.model == 'cosserat' else dict(nodalvol=True)
        m32 = build(g, E, nu, dev, 'noslip', .25, .3, torch.float32, **kw)
        m64 = build(g, E, nu, dev, 'noslip', .25, .3, torch.float64, **kw)
        for m in (m32, m64):
            m.set_environment(gravity=[0, 0, -9.8], ground_point=[0, 0, 0], ground_normal=[0, 0, 1],
                              contact_radius=m.contact_radius.cpu().numpy(), ground_mode='noslip')
        th = torch.zeros(m32.n, 9, dtype=torch.float32, device=dev)
        sub = int(np.ceil(1/args.fps/m64.stable_dt(th[:, :7].double(), safety=.4, include_damping=False)))
        wf = WarpFoam(m32, contact='smooth', layer_dx=.002, implicit_damping=True, graphs=True)
        x0, q0, _, w0 = m32.initial_state()
        gap = float(m32.contact_radius.max())+.003
        x0 = x0+torch.tensor([0., 0., args.R+gap], device=dev)                      # just above the ground
        v = torch.zeros_like(x0); v[:, 2] = -args.v0
        with torch.no_grad():
            pos, quat = warp_rollout(wf, th, (x0.contiguous(), q0.contiguous(), v.contiguous(), w0.contiguous()), 1/args.fps/sub, sub, args.frames)
        pos = pos.double().cpu().numpy()
        h = pos[:, :, 2].max(1)-pos[:, :, 2].min(1)
        print('%-28s E %.0f, substeps %d, finite %s, height min %.3f of rest' % (name, E, sub, bool(np.isfinite(pos).all()), h.min()/h[0]), flush=True)
        traj[name] = pos
    ref = 'nu %.2f' % args.nu_ref
    P = traj[ref]
    X0 = g['points']
    surf = (np.linalg.norm(X0, axis=1) > args.R-1.6*args.cell) & (X0[:, 1] > 0)    # visible surface bodies (front hemisphere)
    r_body = (3*g['masses']/(4*np.pi))**(1/3)
    allxz = np.concatenate([t[:, :, [0, 2]].reshape(-1, 2) for t in traj.values()])
    lo = (allxz[:, 0].min()-.01, allxz[:, 1].min()-.01, max(np.ptp(allxz[:, 0]), np.ptp(allxz[:, 1]))+.02)
    res = 8
    rows = []
    sil_ref = [silhouette(P[f][:, [0, 2]], r_body, lo, px, res) for f in range(len(P))]
    for name, Q in traj.items():
        if name == ref:
            continue
        sil, tex, d3, dw, dh = [], [], [], [], []
        for f in range(len(P)):
            s = silhouette(Q[f][:, [0, 2]], r_body, lo, px, res)
            # raw width / height of the silhouette (px), this run minus the reference
            def wh(m_):
                ys_, xs_ = np.nonzero(m_)
                return (np.ptp(xs_)+1)/res, (np.ptp(ys_)+1)/res
            (w_q, h_q), (w_p, h_p) = wh(s), wh(sil_ref[f])
            dw.append(w_q-w_p); dh.append(h_q-h_p)
            xor = (s ^ sil_ref[f]).sum()/res**2                                    # px^2
            per = max(1., np.pi*np.sqrt(4*sil_ref[f].sum()/res**2/np.pi))           # perimeter of the equal-area disc, px
            sil.append(xor/per)
            dt_ = np.linalg.norm((Q[f]-P[f])[surf][:, [0, 2]], axis=1)/px
            tex.append((np.sqrt((dt_**2).mean()), dt_.max()))
            d3_ = np.linalg.norm(Q[f]-P[f], axis=1)/px
            d3.append((np.sqrt((d3_**2).mean()), d3_.max()))
        sil, tex, d3 = np.array(sil), np.array(tex), np.array(d3)
        fmin = int(np.argmin([np.ptp(P[f][:, 2]) for f in range(len(P))]))       # reference's most compressed frame
        r = dict(run=name, width_diff_px_maxabs=float(np.abs(dw).max()), width_diff_px_at_max_compression=float(dw[fmin]),
                 height_diff_px_maxabs=float(np.abs(dh).max()),
                 silhouette_shift_px_max=float(sil.max()), silhouette_shift_px_mean=float(sil.mean()),
                 texture_rms_px_max=float(tex[:, 0].max()), texture_rms_px_mean=float(tex[:, 0].mean()), texture_max_px=float(tex[:, 1].max()),
                 internal3d_rms_px_max=float(d3[:, 0].max()), internal3d_rms_px_mean=float(d3[:, 0].mean()),
                 per_frame=dict(width_diff=list(map(float, dw)), height_diff=list(map(float, dh)), silhouette=sil.tolist(), texture_rms=tex[:, 0].tolist(), internal3d_rms=d3[:, 0].tolist()))
        rows.append(r)
        print('vs %s | %-28s RAW width diff: max |.| %.2f px, at max compression %+.2f px | height diff max |.| %.2f px' % (
            ref, name, r['width_diff_px_maxabs'], r['width_diff_px_at_max_compression'], r['height_diff_px_maxabs']), flush=True)
        print('vs %s | %-28s silhouette shift: max %.2f mean %.2f px | texture rms: max %.2f mean %.2f px (single point max %.2f) | 3D rms: max %.2f mean %.2f px' % (
            ref, name, r['silhouette_shift_px_max'], r['silhouette_shift_px_mean'], r['texture_rms_px_max'], r['texture_rms_px_mean'], r['texture_max_px'],
            r['internal3d_rms_px_max'], r['internal3d_rms_px_mean']), flush=True)
    (args.out/'nu_signal.json').write_text(json.dumps(dict(config={k: str(v) for k, v in vars(args).items()}, px_m=px, rows=rows), indent=1))
    np.savez_compressed(args.out/'trajectories.npz', **{k.replace(' ', '_').replace(':', '').replace(',', ''): v for k, v in traj.items()})


if __name__ == '__main__':
    main()
