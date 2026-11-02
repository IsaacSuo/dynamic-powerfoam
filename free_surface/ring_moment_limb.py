"""Who holds a limb? The lion's elastic state rebuilt at a frame; each leg (the pieces of the rest foam below a height) turned rigidly by a small angle about its root: per energy term the generalised force and the stiffness. And the model's energy at another run's final shape.
  python free_surface/ring_moment_limb.py --foam <dir> --frames <frames.npz> --other <frames.npz> --k 48 72"""
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
a_.add_argument('--other', type=Path, default=None)
a_.add_argument('--k', type=int, nargs='+', default=[48, 72])
a_.add_argument('--height', type=float, default=.14)
args = a_.parse_args()
E, nu, C0 = 10648.614517785627/1000, .3912951704721983, -.2
g = np.load(args.foam/'dem_exact_single'/'geometry.npz'); cl = np.load(args.foam/'dem_exact_single'/'cells.npz')
X0, r0, V0, P = g['points'], cl['radii'], g['masses'], g['pairs']; n = len(X0)
Z = np.load(args.frames)
zr = X0[:, 2]-X0[:, 2].min(); low = zr < args.height
idx = -np.ones(n, int); idx[low] = np.arange(low.sum())
m = low[P[:, 0]] & low[P[:, 1]]; p = idx[P[m]]
_, lab = connected_components(coo_matrix((np.ones(len(p)), (p[:, 0], p[:, 1])), shape=(low.sum(), low.sum())), directed=False)
ids = np.nonzero(low)[0]
legs = [ids[lab == k] for k in np.argsort(-np.bincount(lab))[:4]]
legs = sorted(legs, key=lambda a: X0[a, 0].mean())
names = ['front A', 'front B', 'hind A', 'hind B']
print('legs (rest centre, cells):', [(names[i], np.round(X0[a].mean(0), 3).tolist(), len(a)) for i, a in enumerate(legs)], flush=True)
TERMS = {0: 'tensor', 1: 'volume', 2: 'var', 4: 'remainder', 5: 'outer', 6: 'ground'}
W = None


def build(x, F, L):
    global W
    W = RingMoment(X0, r0, V0, P, g['areas'], g['contacts'], E, nu, 1e9, theta=.5, kappa=5., eps_cut=-1.)
    W.set_ball((0., 0., 1.), C0, 0.)
    W.start(); W.prepare()
    W.update_ul(W.X)
    W._Ft = F.clone(); W._Fe = F.clone()
    W.tul['Ln'].copy_(L)
    W.update_ul(x)
    W.grad(x); W.reread(); W.grad(x)


def terms(x):
    W.grad(x)
    t = W.tsum.clone()
    return t.sum(1) if t.dim() > 1 else t


for k in args.k:
    x = torch.as_tensor(Z['x'][k], device=dev, dtype=f64)
    build(x, torch.as_tensor(Z['F'][k], device=dev, dtype=f64), torch.as_tensor(Z['L'][k], device=dev, dtype=f64))
    print('t %.3f; centre of the body %s' % (float(Z['t'][k]), np.round(x.mean(0).cpu().numpy(), 3).tolist()), flush=True)
    com = x.mean(0)
    e0 = terms(x)
    grav = 9.8*torch.as_tensor(V0, device=dev, dtype=f64)
    for nm, a in zip(names, legs):
        a_t = torch.as_tensor(a, device=dev)
        top = a[np.argsort(-X0[a, 2])[:max(len(a)//6, 3)]]
        root = x[torch.as_tensor(top, device=dev)].mean(0)
        paw = x[a_t].mean(0)
        out = paw-root
        # opening: the leg's own end away from the body's mid plane: a turn about the body's long axis, signed so the leg's centre goes away from the mid plane in y
        for ax_n, ax in (('about x', [1., 0., 0.]), ('about y', [0., 1., 0.])):
            axis = torch.tensor(ax, dtype=f64, device=dev)
            rot = torch.linalg.cross(axis.expand(len(a), 3), x[a_t]-root)
            if ax_n == 'about x' and float((rot.mean(0)[1])*(paw[1]-com[1])) < 0:
                rot = -rot
            eps = 1e-3
            xp, xm = x.clone(), x.clone()
            xp[a_t] += eps*rot; xm[a_t] -= eps*rot
            ep, em = terms(xp), terms(xm)
            Q = -(ep-em)/(2*eps); S = (ep+em-2*e0)/eps**2
            qg = float((-(grav[a_t])*rot[:, 2]).sum())
            tot = sum(float(Q[k_]) for k_ in TERMS)
            print('  %s %s (leg length %.3f, centre off the mid plane %+.3f): torque, + opens: %s; total %+.3e; gravity %+.3e; stiffness %s' % (nm, ax_n, float(out.norm()), float(paw[1]-com[1]), {TERMS[k_]: float('%.2e' % float(Q[k_])) for k_ in TERMS if abs(float(Q[k_])) > 0}, tot, qg, {TERMS[k_]: float('%.2e' % float(S[k_])) for k_ in TERMS if abs(float(S[k_])) > 0}), flush=True)
    stored = {TERMS[k_]: float('%.3e' % float(e0[k_])) for k_ in TERMS}
    gp = float((grav*x[:, 2]).sum())
    print('  energy here: %s; gravity %.4e; total %.5e' % (stored, gp, gp+sum(float(e0[k_]) for k_ in TERMS)), flush=True)
if args.other is not None:
    O = np.load(args.other)
    x = torch.as_tensor(O['x'][-1], device=dev, dtype=f64); F = torch.as_tensor(O['F'][-1], device=dev, dtype=f64)
    L = torch.log(torch.linalg.det(F).clamp(min=.05))
    build(x, F, L)
    e0 = terms(x)
    gp = float((9.8*torch.as_tensor(V0, device=dev, dtype=f64)*x[:, 2]).sum())
    print('this model at the other run\'s final shape (t %.3f): %s; gravity %.4e; total %.5e' % (float(O['t'][-1]), {TERMS[k_]: float('%.3e' % float(e0[k_])) for k_ in TERMS}, gp, gp+sum(float(e0[k_]) for k_ in TERMS)), flush=True)
