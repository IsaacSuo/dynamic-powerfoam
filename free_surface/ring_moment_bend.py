"""A leg bent by hand from rest, an arc with plane sections, the body fixed, nothing yields: the state goes along the path as in a run (update_ul, the reading, end_ul: L, F_g, q all the model's own). At each angle the model's own force against the curvature: Q = g . dx/dkappa, + restores; per term by the energy's slope. A linear beam: Q = E I_int kappa. Out and back: the loop's work.
  python free_surface/ring_moment_bend.py --foam <dir> [--leg 0] [--lat 1] [--max 150] [--step 2.5]"""
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
a_.add_argument('--leg', type=int, default=0)
a_.add_argument('--lat', type=int, default=1)
a_.add_argument('--top', type=float, default=.15)
a_.add_argument('--max', type=float, default=150.)
a_.add_argument('--step', type=float, default=2.5)
a_.add_argument('--every', type=int, default=6)
args = a_.parse_args()
E, nu, C0 = 10648.614517785627/1000, .3912951704721983, -5.
g = np.load(args.foam/'dem_exact_single'/'geometry.npz'); cl = np.load(args.foam/'dem_exact_single'/'cells.npz')
X0, r0, V0, P = g['points'], cl['radii'], g['masses'], g['pairs']; n = len(X0)
zr = X0[:, 2]-X0[:, 2].min(); low = zr < .18
idx = -np.ones(n, int); idx[low] = np.arange(low.sum())
m = low[P[:, 0]] & low[P[:, 1]]; p = idx[P[m]]
_, lab = connected_components(coo_matrix((np.ones(len(p)), (p[:, 0], p[:, 1])), shape=(low.sum(), low.sum())), directed=False)
ids = np.nonzero(low)[0]
legs = sorted([ids[lab == k] for k in np.argsort(-np.bincount(lab))[:4]], key=lambda a: X0[a, 0].mean())
a = legs[args.leg]; a = a[zr[a] < args.top]
at = torch.as_tensor(a, device=dev); N = len(a)
s_ = zr[a].max()-zr[a]; Lb = float(s_.max()); ztop = float(X0[a, 2].max())
ed = np.linspace(0., Lb, 9); cs = .5*(ed[1:]+ed[:-1])
lat = args.lat
cl_ = np.array([X0[a[(s_ >= ed[i]) & (s_ <= ed[i+1])], lat].mean() for i in range(8)])
c_s = np.interp(s_, cs, cl_)
Iint = 0.
for i in range(8):
    mk = (s_ >= ed[i]) & (s_ <= ed[i+1]); k = a[mk]; v = V0[k]; rb = (3*v/(4*np.pi))**(1/3)
    Iint += (v*((X0[k, lat]-(v*X0[k, lat]).sum()/v.sum())**2+rb**2/5)).sum()
print('leg %d: %d cells, length %.3f, bent in coordinate %d; integral of I ds %.3e; beam Q = %.3e kappa, kappa = angle / %.3f' % (args.leg, N, Lb, lat, Iint, E*Iint, Lb), flush=True)
W = RingMoment(X0, r0, V0, P, g['areas'], g['contacts'], E, nu, 1e9, theta=.5, kappa=5., eps_cut=-1.)
W.set_ball((0., 0., 1.), C0, 0.)
W.start(); W.prepare(); W.update_ul(W.X)
X = W.X.clone()
NAMES = {0: 'tensor', 1: 'volume', 2: 'var', 4: 'remainder', 5: 'outer'}
s_t = torch.as_tensor(s_, device=dev, dtype=f64); c_t = torch.as_tensor(c_s, device=dev, dtype=f64)
y_t = X[at, lat]-c_t


def bent(kap):
    x = X.clone()
    ph = kap*s_t
    fc = (1-torch.cos(ph))/kap if kap > 1e-9 else 0.*s_t
    fs = torch.sin(ph)/kap if kap > 1e-9 else s_t
    x[at, lat] = c_t+fc+y_t*torch.cos(ph)
    x[at, 2] = ztop-fs+y_t*torch.sin(ph)
    return x


def terms(y):
    W.grad(y)
    t = W.tsum.clone(); t = t.sum(1) if t.dim() > 1 else t
    return np.array([float(t[k]) for k in NAMES])


def advance(x0, x1):
    W.update_ul(x0)
    z = .5*(x0+x1)
    W.grad(z); W.reread(); W.grad(z)
    W.end_ul(.5)


def read(kap, deg, tag):
    x = bent(kap)
    W.update_ul(x)
    W.grad(x); W.reread(); _, gg = W.grad(x); gg = gg.clone()
    dk = np.radians(.2)/Lb
    dx = (bent(kap+dk)-bent(max(kap-dk, 1e-8)))/(dk+min(dk, kap-1e-8))
    Q = float((gg*dx).sum())
    ep, em = terms(x+dk*dx), terms(x-dk*dx)
    W.grad(x)
    q_ = (ep-em)/(2*dk)
    e0_ = terms(x); W.grad(x)
    sv = torch.linalg.svdvals(W.ty['Ff'][at].cpu())
    L_ = torch.exp(W.tul['Ln'][at])
    gl = gg[at]
    print('%s %5.1f deg: Q %+.3e (beam %.3e, ratio %+.2f); by term %s; stored %s; F sv %.2f / %.2f, smallest %.3f; e^L %.2f..%.2f; off the leg |g| %.2e' % (tag, deg, Q, E*Iint*kap, Q/(E*Iint*kap), {k: float('%+.1e' % q_[i]) for i, k in enumerate(NAMES.values())}, {k: float('%.1e' % e0_[i]) for i, k in enumerate(NAMES.values())}, float(sv[:, 0].median()), float(sv[:, 2].median()), float(sv[:, 2].min()), float(L_.min()), float(L_.max()), float(gg.norm())), flush=True)
    return Q


d = np.radians(args.step)/Lb
kmax = int(round(args.max/args.step))
x = X.clone(); work = 0.; Qlast = 0.
W.grad(X); W.reread(); W.grad(X)
for k in range(1, kmax+1):
    kap = k*d
    x1 = bent(kap)
    advance(x, x1)
    x = x1
    if k % args.every == 0 or k == kmax:
        Qlast = read(kap, k*args.step, 'out ')
Q0 = Qlast
for k in range(kmax-1, 0, -1):
    kap = k*d
    x1 = bent(kap)
    advance(x, x1)
    x = x1
    if k % args.every == 0:
        read(kap, k*args.step, 'back')
x1 = X.clone(); advance(x, x1)
W.update_ul(X); W.grad(X); W.reread(); _, gg = W.grad(X)
t_ = terms(X)
print('back at rest: |g| %.3e; stored %s; max |F - I| on the leg %.3f; e^L %.3f..%.3f' % (float(gg.norm()), {k: float('%.2e' % t_[i]) for i, k in enumerate(NAMES.values())}, float((W.ty['Ff'][at]-W.I3).flatten(1).norm(dim=1).max()), float(torch.exp(W.tul['Ln'][at]).min()), float(torch.exp(W.tul['Ln'][at]).max())), flush=True)
