"""A leg lying curled on the frictionless ground: the part beyond each joint of the leg turned rigidly about the vertical through the joint, the way that lengthens the leg's end to end. Per term the torque (+ straightens) and the stiffness, from the run's own frame.
  python free_surface/ring_moment_coil.py --foam <dir> --frames <frames.npz> [--k 48 72]"""
import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from free_surface.ring_moment import RingMoment

f64, dev = torch.float64, 'cuda'
a_ = argparse.ArgumentParser()
a_.add_argument('--foam', type=Path, required=True)
a_.add_argument('--frames', type=Path, required=True)
a_.add_argument('--k', type=int, nargs='+', default=[48, 72])
args = a_.parse_args()
E, nu, C0 = 10648.614517785627/1000, .3912951704721983, -.2
g = np.load(args.foam/'dem_exact_single'/'geometry.npz'); cl = np.load(args.foam/'dem_exact_single'/'cells.npz')
X0, r0, V0, P = g['points'], cl['radii'], g['masses'], g['pairs']; n = len(X0)
Z = np.load(args.frames)
zr = X0[:, 2]-X0[:, 2].min(); low = zr < .18
idx = -np.ones(n, int); idx[low] = np.arange(low.sum())
m = low[P[:, 0]] & low[P[:, 1]]; p = idx[P[m]]
_, lab = connected_components(coo_matrix((np.ones(len(p)), (p[:, 0], p[:, 1])), shape=(low.sum(), low.sum())), directed=False)
ids = np.nonzero(low)[0]
legs = sorted([ids[lab == k] for k in np.argsort(-np.bincount(lab))[:4]], key=lambda a: X0[a, 0].mean())
ed = np.linspace(0., .17, 8)
TERMS = {0: 'tensor', 1: 'volume', 2: 'var', 4: 'remainder', 5: 'outer'}
W = RingMoment(X0, r0, V0, P, g['areas'], g['contacts'], E, nu, 1e9, theta=.5, kappa=5., eps_cut=-1.)
W.set_ball((0., 0., 1.), C0, 0.)
W.start(); W.prepare(); W.update_ul(W.X)
up = torch.tensor([0., 0., 1.], dtype=f64, device=dev)


def terms(x):
    W.grad(x)
    t = W.tsum.clone(); t = t.sum(1) if t.dim() > 1 else t
    return {TERMS[k]: float(t[k]) for k in TERMS} | {'ground': float(W.ground_store(x))}


for k in args.k:
    x = torch.as_tensor(Z['x'][k], device=dev, dtype=f64)
    W._Ft = torch.as_tensor(Z['F'][k], device=dev, dtype=f64); W._Fe = W._Ft.clone()
    W.tul['Ln'].copy_(torch.as_tensor(Z['L'][k], device=dev, dtype=f64))
    W.update_ul(x); W.grad(x); W.reread(); W.grad(x)
    e0 = terms(x)
    vel = (Z['x'][min(k+1, len(Z['t'])-1)]-Z['x'][k-1])/(Z['t'][min(k+1, len(Z['t'])-1)]-Z['t'][k-1])
    print('t %.3f; ' % float(Z['t'][k]), flush=True)
    for li, a in enumerate(legs[:2]):
        bins = [a[(zr[a] >= ed[i]) & (zr[a] < ed[i+1])] for i in range(7)]
        print('  leg %d: speed of the paw %.4f, of the leg median %.4f' % (li, float(np.linalg.norm(vel[bins[0]], axis=1).mean()), float(np.median(np.linalg.norm(vel[a], axis=1)))), flush=True)
        for j in (2, 3, 4, 5):
            joint = x[torch.as_tensor(bins[j], device=dev)].mean(0)
            below = np.concatenate(bins[:j]); bt = torch.as_tensor(below, device=dev)
            rot = torch.linalg.cross(up.expand(len(below), 3), x[bt]-joint)
            pawc = x[torch.as_tensor(bins[0], device=dev)].mean(0); topc = x[torch.as_tensor(bins[-1], device=dev)].mean(0)
            pm = rot[torch.as_tensor(np.isin(below, bins[0]), device=dev)].mean(0)
            if float((pm*(pawc-topc)).sum()) < 0:
                rot = -rot
            eps = 1e-3
            xp, xm = x.clone(), x.clone(); xp[bt] += eps*rot; xm[bt] -= eps*rot
            ep, em = terms(xp), terms(xm)
            Q = {t_: -(ep[t_]-em[t_])/(2*eps) for t_ in ep}; S = {t_: (ep[t_]+em[t_]-2*e0[t_])/eps**2 for t_ in ep}
            print('    joint at slice %d, %d cells beyond, lever %.3f: torque, + straightens: total %+.2e; %s; stiffness total %.2e; %s' % (j, len(below), float((pawc-joint).norm()), sum(Q.values()), {t_: float('%+.1e' % Q[t_]) for t_ in Q}, sum(S.values()), {t_: float('%.1e' % S[t_]) for t_ in S}), flush=True)
