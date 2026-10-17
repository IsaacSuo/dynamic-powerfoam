"""Identifiability of the burger parameters from silhouette features (springgaus_train.py featbatch output).

Central-difference Jacobian of the features (feat6 per frame and camera: cx, cy, sx, sy, sqrt area, aspect) with
respect to each parameter, at the base (truth) parameters. Features are normalised as in the training loss (each by
its own std over frames 1..19 of the base series, per camera, with floors); parameters are expressed in 'natural
units' (E, damping, friction, alpha: 10 %; nu: 0.01; velocity: 0.01 m/s; spin: 0.1 rad/s). Then, with
F = J^T J (Gauss-Newton / Fisher information for unit noise on the normalised features):
  - per-parameter sensitivity (column norm) and which feature group dominates it;
  - singular values / vectors of J: the weakest directions, i.e. parameter combinations the silhouettes barely see;
  - correlation matrix from F^-1 (pairs that compensate each other).
Natural units also: tilt 0.01 rad, ground height 1 mm, contact-field coefficients 1 mm. --phys adds the physical observables (log period, decay, squash,
stiffness ratio of the aspect series, springgaus_train.phys_obs) as rows normalised by 5 %.
"""
import argparse
import json
from pathlib import Path

import numpy as np

PARAMS = [('E', .1, 'log'), ('nu', None, 'nu'), ('damping', .1, 'log'), ('friction', .1, 'log'), ('alpha', .1, 'log'),
          ('vx', .01, 'v'), ('vy', .01, 'v'), ('vz', .01, 'v'), ('wx', .1, 'w'), ('wy', .1, 'w'), ('wz', .1, 'w'),
          ('tiltA', .01, 'v'), ('tiltB', .01, 'v'), ('ground', .001, 'v'),
          ('c0', 1., 'v'), ('c1', 1., 'v'), ('c2', 1., 'v'), ('c3', 1., 'v'), ('c4', 1., 'v'), ('c5', 1., 'v')]
FEATS = ['cx', 'cy', 'sx', 'sy', 'sqrtA', 'aspect']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dir', type=Path, default=Path('output/springgaus_dem/burger_filled_mf0.1/synth/ident'))
    p.add_argument('--cams', type=int, nargs='+', default=[1, 2], help='camera indices used (training cameras 1, 2)')
    p.add_argument('--frames', type=int, nargs=2, default=[1, 19])
    p.add_argument('--phys', action='store_true')
    p.add_argument('--pos-scale', type=float, default=0., help='normalise the centroid by this many px (as springgaus_train --pos-scale)')
    p.add_argument('--drop', nargs='*', default=[], help='parameters left out (fixed)')
    args = p.parse_args()
    fb = json.loads((args.dir/'featbatch.json').read_text())
    batch = {r['name']: r for r in json.loads((args.dir/'batch.json').read_text())}
    runs = {k: np.array(v) for k, v in fb['runs'].items()}               # (frames, cams, 6)
    f0, f1 = args.frames
    sel = lambda a: a[f0:f1+1][:, args.cams]
    base = sel(runs['base'])
    floor = np.array([.5, .5, .5, .5, .5, .01])
    scale = np.maximum(base.std(0), floor)                               # (cams, 6)
    if args.pos_scale > 0:
        scale[:, :2] = args.pos_scale
    cols, names = [], []
    for name, unit, kind in PARAMS:
        if name+'+' not in runs or name in args.drop:
            continue
        h = batch[name+'+']['step']
        d = (sel(runs[name+'+'])-sel(runs[name+'-']))/(2*h)              # per raw parameter unit
        if kind == 'nu':                                                  # p5 -> nu: dnu/dp5 = 0.5 s (1 - s)
            p5 = batch['base']['parameters'][5]; s_ = 1/(1+np.exp(-(np.log(1.5)+p5)))
            d = d/(.5*s_*(1-s_))*.01                                      # per 0.01 of nu
        elif kind == 'log':
            d = d*unit                                                    # per 10 %
        else:
            d = d*(unit/1.)                                               # per 0.01 m/s or 0.1 rad/s (step in the same units)
        col = (d/scale).reshape(-1)
        if args.phys:
            import torch
            from experiments.springgaus_train import phys_obs_np
            real = np.array(fb['real'])
            rows = []
            for ci, c in enumerate(args.cams):
                k_imp = int(np.argmin(real[1:20, c, 5]))
                lp = np.log(phys_obs_np(np.array(runs[name+'+'])[1:20, c], k_imp))
                lm = np.log(phys_obs_np(np.array(runs[name+'-'])[1:20, c], k_imp))
                rows.append((lp-lm)/(2*h)/.05)
            dp = np.concatenate(rows)
            if kind == 'nu':
                dp = dp/(.5*s_*(1-s_))*.01
            elif kind == 'log':
                dp = dp*unit
            else:
                dp = dp*unit
            col = np.concatenate((col, dp))
        cols.append(col); names.append(name)
    J = np.stack(cols, 1)
    n_feat_per_group = {}
    print('per-parameter sensitivity (rms over frames/cameras of the normalised feature change per natural unit), and by feature:')
    for k, n in enumerate(names):
        nf_ = (f1-f0+1)*len(args.cams)*6
        col = J[:nf_, k].reshape(f1-f0+1, len(args.cams), 6)
        by = np.sqrt((col**2).mean((0, 1)))
        extra = (' | phys %.3f' % np.sqrt((J[nf_:, k]**2).mean())) if J.shape[0] > nf_ else ''
        print('  %-9s total %.3f | %s%s' % (n, np.sqrt((J[:, k]**2).mean()), ' '.join('%s %.3f' % (FEATS[i], by[i]) for i in range(6)), extra))
    U, S, Vt = np.linalg.svd(J, full_matrices=False)
    print('\nsingular values (strongest -> weakest) and the parameter combination of each direction:')
    for i in range(len(S)):
        v = Vt[i]
        top = np.argsort(-np.abs(v))[:4]
        print('  s %.3f | %s' % (S[i], '  '.join('%+.2f %s' % (v[j], names[j]) for j in top)))
    F = J.T@J
    C = np.linalg.inv(F)
    sd = np.sqrt(np.diag(C))
    R = C/np.outer(sd, sd)
    print('\nposterior std per parameter (natural units, unit noise per normalised feature) and strongest correlations:')
    for k, n in enumerate(names):
        r = R[k].copy(); r[k] = 0
        j = np.argsort(-np.abs(r))[:2]
        print('  %-9s std %.3g | corr %s %+.2f, %s %+.2f' % (n, sd[k], names[j[0]], r[j[0]], names[j[1]], r[j[1]]))
    print('\ncondition number %.3g' % (S[0]/S[-1]))


if __name__ == '__main__':
    main()
