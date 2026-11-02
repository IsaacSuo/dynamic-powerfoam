"""The limb's bending stiffness in the model against the beam's: at rest, a leg (a piece of the rest foam below a height) given a small pure bending about its root, plane sections plane, the body fixed: energy per term over 1/2 E I kappa^2 integrated, I from the cells' own volumes.
  python free_surface/ring_moment_bend.py --foam <dir> [--kappa 1.] [--walls]"""
import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from free_surface.ring_moment import RingMoment
from free_surface.ring_yield import RingYield

f64, dev = torch.float64, 'cuda'
a_ = argparse.ArgumentParser()
a_.add_argument('--foam', type=Path, required=True)
a_.add_argument('--kappa', type=float, default=1.)
a_.add_argument('--walls', action='store_true')
a_.add_argument('--height', type=float, default=.17)
args = a_.parse_args()
E, nu, C0 = 10648.614517785627/1000, .3912951704721983, -5.
g = np.load(args.foam/'dem_exact_single'/'geometry.npz'); cl = np.load(args.foam/'dem_exact_single'/'cells.npz')
X0, r0, V0, P = g['points'], cl['radii'], g['masses'], g['pairs']; n = len(X0)
W = (RingYield if args.walls else RingMoment)(X0, r0, V0, P, g['areas'], g['contacts'], E, nu, 1e9, theta=.5, kappa=5., eps_cut=-1.)
W.set_ball((0., 0., 1.), C0, 0.)
W.start(); W.prepare()
W.update_ul(W.X)
zr = X0[:, 2]-X0[:, 2].min(); low = zr < args.height
idx = -np.ones(n, int); idx[low] = np.arange(low.sum())
m = low[P[:, 0]] & low[P[:, 1]]; p = idx[P[m]]
_, lab = connected_components(coo_matrix((np.ones(len(p)), (p[:, 0], p[:, 1])), shape=(low.sum(), low.sum())), directed=False)
ids = np.nonzero(low)[0]
legs = sorted([ids[lab == k] for k in np.argsort(-np.bincount(lab))[:4]], key=lambda a: X0[a, 0].mean())
TERMS = {0: 'tensor', 1: 'volume', 2: 'var', 4: 'remainder', 5: 'outer'}
X = W.X.clone()


def energy(x):
    W.update_ul(x)
    W.grad(x); W.reread(); W.grad(x)
    t = W.tsum.clone()
    return t.sum(1) if t.dim() > 1 else t


e0 = energy(X)
print('E / rho %.3f; rest energy %s' % (E, {TERMS[k]: float('%.1e' % float(e0[k])) for k in TERMS}), flush=True)
for li, a in enumerate(legs):
    z = X0[a, 2]; top = z.max()
    # the shank: from the top of the piece down to above the paw
    keep = a
    s = top-X0[keep, 2]; L = float(s.max())
    nb = 8; edges = np.linspace(0., L, nb+1)
    for ax_name, lat_i in (('x', 0), ('y', 1)):
        # the beam's I per slice from the volumes, each cell a ball of its volume
        Iint = 0.; Aint = 0.
        for b in range(nb):
            k = keep[(s >= edges[b]) & (s <= edges[b+1])]
            if len(k) == 0:
                continue
            ds = edges[b+1]-edges[b]
            A = V0[k].sum()/ds
            cm = (V0[k]*X0[k, lat_i]).sum()/V0[k].sum()
            rb = (3*V0[k]/(4*np.pi))**(1/3)
            I = ((V0[k]*((X0[k, lat_i]-cm)**2+rb**2/5)).sum())/ds
            Iint += I*ds; Aint += A*ds
        cen = np.zeros(len(keep))
        for b in range(nb):
            mk = (s >= edges[b]) & (s <= edges[b+1])
            if mk.any():
                cen[mk] = (V0[keep[mk]]*X0[keep[mk], lat_i]).sum()/V0[keep[mk]].sum()
        out = []
        for kap in (args.kappa, 2*args.kappa):
            x = X.clone()
            kt = torch.as_tensor(keep, device=dev)
            s_t = torch.as_tensor(s, device=dev, dtype=f64); y_t = torch.as_tensor(X0[keep, lat_i]-cen, device=dev, dtype=f64)
            x[kt, lat_i] = x[kt, lat_i]+.5*kap*s_t*s_t
            x[kt, 2] = x[kt, 2]+kap*s_t*y_t
            e = energy(x)-e0
            out.append(e)
        th = .5*args.kappa**2*E*Iint
        print('leg %d (%d cells, length %.3f, mean area %.2e, I %.2e), bent in %s: energy / beam: total %.2f (at 2 kappa %.2f); %s; tip moved %.2f h' % (li, len(keep), L, Aint/L, Iint/L, ax_name, float(sum(out[0][k] for k in TERMS))/th, float(sum(out[1][k] for k in TERMS))/(4*th), {TERMS[k]: round(float(out[0][k])/th, 2) for k in TERMS}, .5*args.kappa*L*L/float(np.median(V0)**(1/3))), flush=True)
