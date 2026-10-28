"""The jack's drop with the updated-Lagrangian Power cells (ul_model.ULModel, 2026-10-27), set up as hertz_drop.py: the 8k
foam at rest with its lowest ball point --height above y = 0, gravity -y, frictionless Hertz ground on the balls of the
cells with an exposed cap (radius this step's), backward Euler dt 1/1200, tol 1e-3 of |m g|, 0.5 s, no viscosity.
Records every 1/240 s in mpm_compare.py's columns (t, com_y, com_vy, K, elastic, contact/weight) and more: the total
energy's change (kinetic + elastic + ground + gravity), walls born / died and the elastic energy's jump at the recuts
since the last record, the solve's and the volume fit's work.
  PYTHONPATH=<main repo>:<worktree> python free_surface/ul_drop.py --foam <dir with dem_exact_single> --out <dir>"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from free_surface.ul_model import ULModel, cell_moments_grid

f64, dev = torch.float64, 'cuda'


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument('--foam', type=Path, default=Path('output/round01_pruned_exact'))
    a_.add_argument('--out', type=Path, default=Path('output/round01_pruned_exact'))
    a_.add_argument('--height', type=float, default=.02)
    a_.add_argument('--dt', type=float, default=1/1200)
    a_.add_argument('--T', type=float, default=.5)
    a_.add_argument('--tol', type=float, default=1e-3)
    a_.add_argument('--E', type=float, default=23008.38)
    a_.add_argument('--nu', type=float, default=.3008)
    a_.add_argument('--grid', type=float, default=.002, help="the cells' centroids and second moments: sample spacing (m)")
    a_.add_argument('--contact-radius', choices=('current', 'rest'), default='current',
                    help="current: the volume fit's radii; rest: the rest radii (10-27 test of the contact / volume-fit feedback)")
    a_.add_argument('--squeeze', type=float, default=0., help='> 0: no ground, no gravity; the foam starts at rest in place with the '
                    'velocity v_y = -a (y - y_c) (a smooth compression; 10-28 test against the drop\'s contact)')
    a_.add_argument('--f-mode', default='total', help="ul_model f_mode: 'increment' or 'total' (UPDATED_LAGRANGE.md section 20)")
    a_.add_argument('--new-ref', default='rest', help="f_mode 'total': a newly met pair's reference, 'contact' or 'rest'")
    a_.add_argument('--fit-w', default='rest', help="f_mode 'total': the fits' pairs and weights, 'wall' (current cut), 'rest' (rest neighbours and weights) or 'lens'")
    a_.add_argument('--tag', default='ul_drop')
    args = a_.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    g = np.load(args.foam/'dem_exact_single'/'geometry.npz'); cl = np.load(args.foam/'dem_exact_single'/'cells.npz')
    X, r, V = g['points'], cl['radii'], g['masses']; n = len(X)
    cf = args.out/('ul_moments_%g.npz' % args.grid)
    if cf.exists():
        c = np.load(cf); dlt, J0 = c['dlt'], c['J0']
    else:
        dlt, J0, cnt = cell_moments_grid(X, r, V, args.grid); np.savez(cf, dlt=dlt, J0=J0, cnt=cnt)
    E, nu = args.E/1000, args.nu
    mdl = ULModel(X, r, V, dlt, J0, E, nu, f_mode=args.f_mode, new_ref=args.new_ref, fit_w=args.fit_w)
    up = torch.tensor([0., 1., 0.], dtype=f64, device=dev)
    I = torch.eye(3, dtype=f64, device=dev).expand(n, 3, 3).contiguous()
    low = float((mdl.X0[:, 1]-mdl.r0).min())
    st = dict(x=mdl.X0+(args.height-low)*up, v=torch.zeros(n, 3, dtype=f64, device=dev), F=I.clone(), r=mdl.r0.clone())
    ground = ((0., 1., 0.), 0., E/(1-nu**2))
    M = float(mdl.m.sum()); gacc = 9.8; ref = gacc*M
    if args.squeeze > 0:
        yc = float((mdl.m*st['x'][:, 1]).sum())/M
        st['v'][:, 1] = -args.squeeze*(st['x'][:, 1]-yc); ground = None; gacc = 0.
    hgt = lambda x: x[:, 1]-float((x[:, 1]-mdl.r0).min())                            # a site's height above the lowest rest-radius ball point
    dt = args.dt; nst = int(round(args.T/dt)); every = max(1, round(1/(240*dt)))
    print('jack: %d cells, height %.3f m; E %.1f, nu %.4f, E* %.3f (per unit density); dt 1/%.0f, tol %.0e of the weight, %d steps' % (
        n, args.height, args.E, nu, E/(1-nu**2), 1/dt, args.tol, nst), flush=True)
    cols = ['t', 'com_y', 'com_vy', 'K', 'elastic', 'contact/weight', 'touching', 'overlap/R', 'min J', 'H change', 'born', 'died', 'recut jump',
            'products', 'Newton', 'radii it', 'step s', 'V/target lo', 'V/target hi', 'vol err', 'V/target p1', 'V/target p99', 'off share',
            'off height']
    rec = []
    com = lambda x: float((mdl.m*x[:, 1]).sum())/M
    H0 = gacc*float((mdl.m*st['x'][:, 1]).sum()); U_prev = None; keys_prev = None; acc = dict(born=0, died=0, jump=0.)
    t0 = time.time()
    for s in range(1, nst+1):
        ts = time.time()
        try:
            st, rep = mdl.step(st, dt, grav=(0., -gacc, 0.), ref=ref, tol=args.tol, ground=ground,
                               ground_r=mdl.r0 if args.contact_radius == 'rest' else None)
            if ground is None:
                rep['ground'] = dict(force=0., touching=0, overlap=0., energy=0.)
        except RuntimeError as e_:
            print('step %d (t %.4f s): %s; stopping' % (s, s*dt, e_), flush=True)
            break
        q_ = rep['radii_start']; acc['qlo'] = min(acc.get('qlo', 1.), q_[0]); acc['qhi'] = max(acc.get('qhi', 1.), q_[1])
        acc['p1'] = min(acc.get('p1', 1.), q_[2]); acc['p99'] = max(acc.get('p99', 1.), q_[3]); acc['off'] = max(acc.get('off', 0.), q_[4])
        if len(q_[5]):
            acc['offh'] = float(hgt(st['x'])[q_[5]].median())/float(np.median(V))**(1/3)
        acc['verr'] = max(acc.get('verr', 0.), rep['vol_err'])
        i, j, _ = rep['walls']; kk = torch.minimum(i, j)*n+torch.maximum(i, j)
        if keys_prev is not None:
            acc['born'] += int((~torch.isin(kk, keys_prev)).sum()); acc['died'] += int((~torch.isin(keys_prev, kk)).sum())
        keys_prev = kk
        U0, U1 = sum(rep['e_start'].values()), sum(rep['e_end'].values())
        if U_prev is not None:
            acc['jump'] += U0-U_prev
        U_prev = U1
        K = float(.5*(mdl.m[:, None]*st['v']**2).sum())
        H = K+U1+rep['ground']['energy']+gacc*float((mdl.m*st['x'][:, 1]).sum())
        if s == 1 and args.squeeze > 0:
            H0 = float(.5*(mdl.m*(args.squeeze*(mdl.X0[:, 1]-float((mdl.m*mdl.X0[:, 1]).sum())/M))**2).sum())
        if s % every == 0 or s == nst:
            rec.append((s*dt, com(st['x']), float((mdl.m*st['v'][:, 1]).sum())/M, K, U1, rep['ground']['force']/ref, rep['ground']['touching'],
                        rep['ground']['overlap'], rep['minJ'], (H-H0)/M, acc['born'], acc['died'], acc['jump']/M, rep['solve']['products'],
                        rep['solve']['outer'], rep['radii_it'], time.time()-ts, acc['qlo'], acc['qhi'], acc['verr'], acc['p1'], acc['p99'], acc['off'],
                        acc.get('offh', np.nan)))
            acc = dict(born=0, died=0, jump=0.)
            q = rec[-1]
            print('t %.4f s (%.0f s): com y %.5f m, vy %+.4f m/s; ground / weight %.3f on %d balls, overlap max %.3f R; min J %.3f; '
                  'H change %+.4f J/kg; walls +%d -%d, recut jump %+.1e J/kg; products %d, Newton %d, radii %d (V / target before %.2f..%.2f, '
                  '1-99 %% %.2f..%.2f, share outside [0.5, 2] %.1e at median height %.1f h; after %.0e), %.2f s / step' % (
                      q[0], time.time()-t0, q[1], q[2], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[17], q[18], q[20],
                      q[21], q[22], q[23], q[19], q[16]), flush=True)
        if rep['solve']['residual'] > args.tol:
            print('step %d: the solve stopped at %.1e of the weight (%s)' % (s, rep['solve']['residual'], rep['solve']), flush=True)
    R = np.array(rec)
    np.savez(args.out/(args.tag+'.npz'), cols=np.array(cols), rec=R)
    c = {k: R[:, i] for i, k in enumerate(cols)}
    on = np.flatnonzero(c['contact/weight'] > .01); t_on = c['t'][on[0]] if len(on) else np.nan
    imin = int(np.argmin(c['com_y'])); after = c['t'] > t_on
    print('total %.0f s for %d steps (recorded to t %.4f s); centre of mass lowest %.4f m at %.3f s; largest upward velocity %.3f m/s; at the end %.4f m, %+.3f m/s; '
          'peak ground force / weight %.2f at %.3f s; min J %.3f; products per record median %d, max %d' % (
              time.time()-t0, nst, c['t'][-1], c['com_y'][imin], c['t'][imin], c['com_vy'][after].max() if after.any() else np.nan, c['com_y'][-1], c['com_vy'][-1],
              c['contact/weight'].max(), c['t'][int(np.argmax(c['contact/weight']))], c['min J'].min(), np.median(c['products']), c['products'].max()), flush=True)


if __name__ == '__main__':
    main()
