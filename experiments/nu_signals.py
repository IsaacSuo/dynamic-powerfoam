"""Which observations of a dropped ball carry information on nu? (IDENTIFIABILITY.md sections 9-10)

Model: the current DEM (bond-level Cosserat, EnergyFoam / WarpFoam) on a cubic-lattice ball (nu_signal_sphere.lattice_ball).
Parameters: log E* (contact modulus E / (1 - nu^2)), nu, log damping multiplier, friction and restitution logits.
The dashpots are built from a reference material (E*, nu_ref) so that changing nu does not change the damping.

  modes   lowest vibration modes of the free ball for several nu at fixed E*: sparse stiffness assembled by graph-coloured
          forward-mode derivatives of the closed-form forces (two bodies interact only within 3 lattice steps, so colours
          (i, j, k) mod 4 separate the columns), shift-invert eigensolve of M^-1/2 K M^-1/2; each mode is classified by its
          radial fraction (spheroidal vs torsional), its volume change and its angular order. Colours (i, j, k) mod 7: two
          bodies of one colour are >= 7 steps apart, so no body has two same-coloured partners within 3 steps.
  fisher  drops (several scenarios); each parameter perturbed by central differences; observations at 1000 fps and
          decimated to 125 fps: silhouette (support function of the side view, 90 directions, sigma 0.5 px), texture
          (visible surface bodies projected to the side view, sigma 0.2 px), 3D (all bodies, sigma 0.2 px = 0.2 mm).
          Fisher matrix F = J^T J / sigma^2 + weak priors; reported: posterior std of each parameter (all others unknown),
          the std of nu if all others were known, and the correlations of nu.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
import warp as wp

from experiments.foam_dem_energy import quat_mul, quat_small
from experiments.foam_dem_warp import WarpFoam, rollout as warp_rollout
from experiments.nu_signal_sphere import lattice_ball
from scipy.spatial.transform import Rotation
from experiments.pacnerf_dem_forward import build

PARAMS = ('logEstar', 'nu', 'logdamp', 'friction', 'restitution')
STEP = dict(logEstar=.02, nu=.01, logdamp=.1, friction=.2, restitution=.2)
PRIOR = dict(logEstar=1., nu=.2, logdamp=1., friction=2., restitution=2.)   # weak priors (std) keep F invertible
KW = dict(kinematics='exact', shell=True, cosserat=True)


def make_model(g, Estar, nu, dtype, dev, ref=None, old=False):
    # old=True: the bonded model the burger was first trained with (springs + averaged-F Neo-Hookean, alpha 0.25,
    # small-strain bond kinematics, rotational inertia x4): springgaus_train without --cosserat
    m = build(g, Estar*(1-nu*nu), nu, dev, 'noslip', .25, .3, dtype, **({} if old is True else dict(shell=True, clean=True) if old == 'clean' else KW))
    m.set_environment(gravity=[0, 0, -9.8], ground_point=[0, 0, 0], ground_normal=[0, 0, 1],
                      contact_radius=m.contact_radius.cpu().numpy(), ground_mode='noslip')
    if ref is not None:                       # dashpots from the reference material: damping independent of nu
        m.c_n, m.c_t, m.c_r = (ref.c_n.to(dtype), ref.c_t.to(dtype), ref.c_r.to(dtype))
    return m


# ----------------------------------------------------------------------------------------------------------- modes
def stiffness(model, idx):
    n = model.n
    x0, q0, _, _ = model.initial_state()
    th = torch.zeros(n, 7, dtype=torch.float64, device=x0.device)

    def force(z):
        f, tau = model.forces(th, x0+z[:, :3], quat_mul(quat_small(z[:, 3:]), q0))
        return torch.cat((f, tau), 1)
    lookup = {tuple(t): k for k, t in enumerate(idx)}
    offs = [(a, b, c) for a in range(-3, 4) for b in range(-3, 4) for c in range(-3, 4) if abs(a)+abs(b)+abs(c) <= 3]
    color = (idx[:, 0] % 7)*49+(idx[:, 1] % 7)*7+(idx[:, 2] % 7)
    partner = -np.ones((n, 343), np.int64)               # the body of colour c within 3 steps of each body (unique)
    for r, t in enumerate(idx):
        for o in offs:
            s = lookup.get((t[0]+o[0], t[1]+o[1], t[2]+o[2]))
            if s is not None:
                assert partner[r, color[s]] < 0
                partner[r, color[s]] = s
    rows, cols, vals = [], [], []
    zero = torch.zeros(n, 6, dtype=torch.float64, device=x0.device)
    for c in range(343):
        sel = torch.as_tensor(color == c, device=x0.device)
        if not bool(sel.any()):
            continue
        r_ = np.flatnonzero(partner[:, c] >= 0); s_ = partner[r_, c]
        for d in range(6):
            v = zero.clone(); v[sel, d] = 1.
            _, Kv = torch.func.jvp(force, (zero,), (v,))
            Kv = (-Kv).cpu().numpy()                          # K = -d(force)/dz
            for e in range(6):
                rows.append(6*r_+e); cols.append(6*s_+d); vals.append(Kv[r_, e])
    K = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(6*n, 6*n)).tocsr()
    asym = abs(K-K.T).max()/abs(K).max()
    return .5*(K+K.T), float(asym)


def modes(args, g, idx):
    dev = 'cuda'
    X = g['points']
    rhat = X/np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    out = []
    for nu in args.nus:
        m = make_model(g, args.Estar, nu, torch.float64, dev, old=True if args.model == 'old' else 'clean' if args.model == 'clean' else False)
        K, asym = stiffness(m, idx)
        mvec = np.concatenate((np.repeat(m.mass.cpu().numpy()[:, None], 3, 1), np.repeat(m.inertia.cpu().numpy()[:, None], 3, 1)), 1).reshape(-1)
        if args.model == 'clean':                        # translations only (no rotational degrees of freedom)
            keep = np.flatnonzero((np.arange(6*m.n) % 6) < 3)
            K = K[keep][:, keep]; mvec = mvec[keep]
        Dm = sp.diags(1/np.sqrt(mvec))
        A = (Dm@K@Dm).tocsc()
        w2, V = spla.eigsh(A, k=args.n_modes+6, sigma=-1., which='LM')
        order = np.argsort(w2); w2, V = w2[order], V[:, order]
        rows = []
        dof = 3 if args.model == 'clean' else 6
        for k in range(6, len(w2)):
            u = (V[:, k]/np.sqrt(mvec)).reshape(-1, dof)[:, :3]
            ur = (u*rhat).sum(1)
            radial = float((ur**2).sum()/max((u**2).sum(), 1e-300))
            # volume change: divergence proxy sum(u . X) / (|u| |X|) (uniform breathing = 1)
            vol = float(abs((u*X).sum())/np.sqrt((u**2).sum()*(X**2).sum()))
            # angular order of the radial pattern: best fit among l = 0, 1, 2, 3 real harmonics of the direction
            th, ph = np.arccos(np.clip(rhat[:, 2], -1, 1)), np.arctan2(rhat[:, 1], rhat[:, 0])
            from scipy.special import sph_harm_y
            fit = {}
            for l in range(4):
                B = np.stack([np.real(sph_harm_y(l, mm, th, ph)) for mm in range(-l, l+1)]+[np.imag(sph_harm_y(l, mm, th, ph)) for mm in range(1, l+1)], 1)
                coef = np.linalg.lstsq(B, ur, rcond=None)[0]
                fit[l] = float(((B@coef)**2).sum()/max((ur**2).sum(), 1e-300))
            rows.append(dict(f_hz=float(np.sqrt(max(w2[k], 0))/(2*np.pi)), radial=radial, volume=vol, l_fit=fit))
        out.append(dict(nu=nu, E=args.Estar*(1-nu*nu), mu=args.Estar*(1-nu)/2, K_asym=asym, modes=rows))
        print('nu %.2f (E %.0f, mu %.0f; K asymmetry %.1e):' % (nu, args.Estar*(1-nu*nu), args.Estar*(1-nu)/2, asym), flush=True)
        for r in rows:
            lbest = max(r['l_fit'], key=r['l_fit'].get)
            print('   %7.2f Hz  radial %.2f  volume %.2f  best l %d (%.2f)' % (r['f_hz'], r['radial'], r['volume'], lbest, r['l_fit'][lbest]), flush=True)
    (args.out/('modes%s.json' % ('' if args.model == 'cosserat' else '_'+args.model))).write_text(json.dumps(out, indent=1))


def lattice_shape(shape, R, a, axes=(1., .8, .6), tilt=(30., 0.)):
    """Cubic lattice inside a ball, an ellipsoid (semi-axes R * axes) or a box (half sides R * axes), rotated by
    tilt = (about y, about x) degrees; 'surface' = bodies with fewer than 6 lattice neighbours."""
    if shape == 'ball':
        g = lattice_ball(R, a)
    else:
        ax = np.asarray(axes)*R
        n = int(np.ceil(ax.max()/a)); gg = np.arange(-n, n+1)
        idx = np.array([(i, j, k) for i in gg for j in gg for k in gg])
        pts = idx*a
        keep = ((pts/(ax-.5*a))**2).sum(1) <= 1 if shape == 'ellipsoid' else (np.abs(pts) <= ax-.5*a+1e-9).all(1)
        idx, pts = idx[keep], pts[keep]
        lookup = {tuple(t): m for m, t in enumerate(idx)}
        pairs, contacts = [], []
        for m, t in enumerate(idx):
            for d in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                o = lookup.get((t[0]+d[0], t[1]+d[1], t[2]+d[2]))
                if o is not None:
                    pairs.append((m, o)); contacts.append(.5*(pts[m]+pts[o]))
        g = dict(points=pts, radii=np.full(len(pts), a*np.sqrt(1/2.4)), pairs=np.array(pairs), areas=np.full(len(pairs), a*a),
                 contacts=np.array(contacts), masses=np.full(len(pts), a**3), fixed=np.zeros(len(pts), bool))
    deg = np.bincount(g['pairs'].reshape(-1), minlength=len(g['points']))
    g['surface'] = deg < 6
    if shape != 'ball':
        Q = Rotation.from_euler('yx', tilt, degrees=True).as_matrix()
        g['points'] = g['points']@Q.T; g['contacts'] = g['contacts']@Q.T
    g['points'] = g['points']-g['points'].mean(0)
    return g


# ---------------------------------------------------------------------------------------------------------- fisher
def simulate(g, p, scen, ref, sub, args):
    dev = 'cuda'
    nu, Estar = p['nu'], np.exp(p['logEstar'])
    m = make_model(g, Estar, nu, torch.float32, dev, ref=ref)
    th = torch.zeros(m.n, 9, dtype=torch.float32, device=dev)
    th[:, 6], th[:, 7], th[:, 8] = p['logdamp'], p['friction'], p['restitution']
    wf = WarpFoam(m, contact='smooth', layer_dx=.002, implicit_damping=True, graphs=True)
    x0, q0, _, w0 = m.initial_state()
    x0 = x0+torch.tensor([0., 0., float(-x0[:, 2].min())+float(m.contact_radius.max())+.002], device=dev)
    v = torch.tensor(scen['v'], dtype=torch.float32, device=dev).expand_as(x0).contiguous()
    with torch.no_grad():
        pos, _ = warp_rollout(wf, th, (x0.contiguous(), q0.contiguous(), v, w0.contiguous()), 1/args.fps/sub, sub, args.frames)
    return pos.double().cpu().numpy()


def features(pos, g, args):
    X = g['points']
    px = 2*args.R/args.px_diameter
    surf = g['surface'] & (X[:, 1] > 0)                                            # visible: surface bodies facing the camera (+y)
    r_body = (3*g['masses']/(4*np.pi))**(1/3)
    ang = np.linspace(0, 2*np.pi, 90, endpoint=False)
    U = np.stack((np.cos(ang), np.sin(ang)), 1)                                   # directions in the (x, z) image plane
    xz = pos[:, :, [0, 2]]
    sil = (xz@U.T+r_body[None, :, None]).max(1)/px                               # support function (frames, 90), px
    tex = xz[:, surf].reshape(len(pos), -1)/px
    d3 = pos.reshape(len(pos), -1)/px
    return dict(silhouette=sil[1:], texture=tex[1:], d3=d3[1:])


def fisher(args, g):
    dev = 'cuda'
    scenarios = dict(vertical=dict(v=[0., 0., -args.v0]), vertical_fast=dict(v=[0., 0., -2*args.v0]),
                     oblique=dict(v=[args.v0*.67, 0., -args.v0]))
    sigma = dict(silhouette=.5, texture=.2, d3=.2)
    report = []
    for scen_name in args.scenarios:
        scen = scenarios[scen_name]
        for nu_b in (args.fisher_nus if scen_name == 'vertical' else [args.nu_ref]):
            base = dict(logEstar=float(np.log(args.Estar)), nu=nu_b, logdamp=0., friction=0., restitution=0.)
            ref = make_model(g, args.Estar, args.nu_dashpot, torch.float64, dev)
            stiff = make_model(g, args.Estar*np.exp(STEP['logEstar']), nu_b+STEP['nu'], torch.float64, dev)
            sub = int(np.ceil(1.1/args.fps/stiff.stable_dt(torch.zeros(stiff.n, 7, dtype=torch.float64, device=dev), safety=.4, include_damping=False)))
            feats = {}
            for k in PARAMS:
                for s in (1, -1):
                    p = dict(base); p[k] += s*STEP[k]
                    feats[k, s] = features(simulate(g, p, scen, ref, sub, args), g, args)
            f0 = features(simulate(g, base, scen, ref, sub, args), g, args)
            np.savez_compressed(args.out/('features_%s%s_nu%.2f.npz' % ('' if args.shape == 'ball' else args.shape+'_', scen_name, nu_b)),
                                **{'%s_%s_%+d' % (obs, k, s): v[obs] for (k, s), v in feats.items() for obs in v}, **{'%s_base' % o: f0[o] for o in f0})
            h0 = f0['silhouette']
            print('%s (%s), nu %.2f: substeps %d per ms; height (px) min %.1f of %.1f' % (scen_name, args.shape, nu_b, sub, (h0[:, 22]+h0[:, 67]).min(), h0[0, 22]+h0[0, 67]), flush=True)
            for obs in ('silhouette', 'texture', 'd3'):
                for rate, dec in (('1000fps', 1), ('125fps', 8)):
                    J = np.stack([((feats[k, 1][obs]-feats[k, -1][obs])/(2*STEP[k]))[dec-1::dec].reshape(-1) for k in PARAMS], 1)
                    F = J.T@J/sigma[obs]**2
                    Fp = F+np.diag([1/PRIOR[k]**2 for k in PARAMS])
                    C = np.linalg.inv(Fp)
                    std = np.sqrt(np.diag(C))
                    corr = C[1]/np.sqrt(C[1, 1]*np.diag(C))
                    r = dict(shape=args.shape, scenario=scen_name, nu=nu_b, obs=obs, rate=rate, std=dict(zip(PARAMS, std.tolist())),
                             nu_std_others_known=float(1/np.sqrt(F[1, 1]+1e-300)), nu_corr=dict(zip(PARAMS, corr.tolist())),
                             fisher_eig=np.linalg.eigvalsh(F).tolist())
                    report.append(r)
                    print('   %-10s %-7s std: nu %.3f (others known %.4f) | logE* %.4f logdamp %.3f fric %.2f rest %.3f | corr(nu, .): E* %+.2f damp %+.2f rest %+.2f' % (
                        obs, rate, std[1], r['nu_std_others_known'], std[0], std[2], std[3], std[4], corr[0], corr[2], corr[4]), flush=True)
            (args.out/('fisher%s.json' % ('' if args.shape == 'ball' else '_'+args.shape))).write_text(json.dumps(report, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('modes', 'fisher'))
    p.add_argument('--R', type=float, default=.035)
    p.add_argument('--cell', type=float, default=.0035)
    p.add_argument('--Estar', type=float, default=5e4)
    p.add_argument('--nus', type=float, nargs='+', default=[.3, .4, .45, .49])
    p.add_argument('--n-modes', type=int, default=24)
    p.add_argument('--v0', type=float, default=1.5)
    p.add_argument('--scenarios', nargs='+', default=['vertical', 'vertical_fast', 'oblique'])
    p.add_argument('--fisher-nus', type=float, nargs='+', default=[.3, .45])
    p.add_argument('--nu-ref', type=float, default=.45)
    p.add_argument('--nu-dashpot', type=float, default=.3, help='reference nu of the dashpot scale')
    p.add_argument('--fps', type=float, default=1000.)
    p.add_argument('--frames', type=int, default=250)
    p.add_argument('--px-diameter', type=float, default=70.)
    p.add_argument('--shape', default='ball', choices=('ball', 'ellipsoid', 'box'))
    p.add_argument('--model', default='cosserat', choices=('cosserat', 'old', 'clean'))
    p.add_argument('--axes', type=float, nargs=3, default=[1., .8, .6], help='ellipsoid semi-axes / box half sides, in units of R')
    p.add_argument('--tilt', type=float, nargs=2, default=[30., 0.], help='rotation about y, then x (degrees)')
    p.add_argument('--out', type=Path, default=Path('output/nu_signals'))
    args = p.parse_args()
    wp.init()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stage == 'fisher':
        fisher(args, lattice_shape(args.shape, args.R, args.cell, args.axes, args.tilt))
        return
    g = lattice_ball(args.R, args.cell)
    n = int(np.ceil(args.R/args.cell)); gg = np.arange(-n, n+1)
    idx = np.array([(i, j, k) for i in gg for j in gg for k in gg])
    idx = idx[np.linalg.norm(idx*args.cell, axis=1) <= args.R-.5*args.cell]       # same order as lattice_ball
    assert np.allclose(idx*args.cell, g['points'])
    if args.stage == 'modes':
        modes(args, g, idx)
    else:
        fisher(args, g)


if __name__ == '__main__':
    main()
