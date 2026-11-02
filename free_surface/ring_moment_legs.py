"""Are the front legs joined by the cut? The lion's own run rebuilt at a frame, the cut at its x: the walls between the two front legs (pieces of the rest foam below a height, no rest wall between them), whether each has a spring.
  python free_surface/ring_moment_legs.py --foam <dir> --frames <frames.npz>"""
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
a_.add_argument('--k', type=int, nargs='+', default=[16, 18, 20, 24, 30, 36, 48, 60, 72])
args = a_.parse_args()
E, nu, C0 = 10648.614517785627/1000, .3912951704721983, -.2
g = np.load(args.foam/'dem_exact_single'/'geometry.npz'); cl = np.load(args.foam/'dem_exact_single'/'cells.npz')
X0, r0, V0, P = g['points'], cl['radii'], g['masses'], g['pairs']; n = len(X0)
h = float(np.median(V0)**(1/3))
Z = np.load(args.frames)
zr = X0[:, 2]-X0[:, 2].min()
sets = {}
for H in (.14, .18, .22):
    low = zr < H
    idx = -np.ones(n, int); idx[low] = np.arange(low.sum())
    m = low[P[:, 0]] & low[P[:, 1]]; p = idx[P[m]]
    _, lab = connected_components(coo_matrix((np.ones(len(p)), (p[:, 0], p[:, 1])), shape=(low.sum(), low.sum())), directed=False)
    ids = np.nonzero(low)[0]
    pcs = sorted([ids[lab == k] for k in np.argsort(-np.bincount(lab))[:4]], key=lambda a: X0[a, 0].mean())
    a, b = pcs[0], pcs[1]
    if abs(X0[a, 1].mean()-X0[b, 1].mean()) < .05:                                         # (one piece: the front legs have joined at this height)
        print('below %.2f the front legs are one piece' % H, flush=True)
        continue
    sets[H] = (a, b)
    print('below %.2f: legs of %d and %d cells, rest gap of the centres %.3f, rest walls between them %d' % (H, len(a), len(b), abs(X0[a, 1].mean()-X0[b, 1].mean()), int((np.isin(P[:, 0], a) & np.isin(P[:, 1], b)).sum()+(np.isin(P[:, 1], a) & np.isin(P[:, 0], b)).sum())), flush=True)
W = RingMoment(X0, r0, V0, P, g['areas'], g['contacts'], E, nu, 1e9, theta=.5, kappa=5., eps_cut=-1.)
W.set_ball((0., 0., 1.), C0, 0.)
W.start(); W.prepare()
W.update_ul(W.X)
for k in args.k:
    x = torch.as_tensor(Z['x'][k], device=dev, dtype=f64)
    W._Ft = torch.as_tensor(Z['F'][k], device=dev, dtype=f64); W._Fe = W._Ft.clone()
    W.tul['Ln'].copy_(torch.as_tensor(Z['L'][k], device=dev, dtype=f64))
    W.update_ul(x)
    G = W._ulG
    kp = G.wall_A.detach() > 0
    i, j, A = G.i[kp].long(), G.j[kp].long(), G.wall_A.detach()[kp]
    key = torch.minimum(i, j)*n+torch.maximum(i, j)
    has = torch.isin(key, W.tkey)
    out = []
    for H, (a, b) in sets.items():
        ma, mb = torch.zeros(n, dtype=torch.bool, device=dev), torch.zeros(n, dtype=torch.bool, device=dev)
        ma[torch.as_tensor(a, device=dev)] = True; mb[torch.as_tensor(b, device=dev)] = True
        ab = (ma[i] & mb[j]) | (mb[i] & ma[j])
        xa, xb = x[ma], x[mb]
        dm = torch.cdist(xa, xb).min()
        out.append('H %.2f: walls %d (area / h^2 %.2f), with a spring %d; nearest centres %.2f h' % (H, int(ab.sum()), float(A[ab].sum())/h**2, int((ab & has).sum()), float(dm)/h))
    print('t %.3f: %s' % (float(Z['t'][k]), '; '.join(out)), flush=True)
