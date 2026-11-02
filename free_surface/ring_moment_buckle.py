"""One leg's own modes in the model, its root held and everything else fixed: the dense stiffness of the leg's cells by finite differences of the force, generalised eigenproblem with the masses. At rest against the Euler-Bernoulli beam of the same volumes; at a loaded frame with R re-read at each product (the true tangent, geometric stiffness in it) and frozen (what one inner solve sees).
  python free_surface/ring_moment_buckle.py --foam <dir> --frames <frames.npz> [--leg 0] [--k 18 19]"""
import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import eigh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from free_surface.ring_moment import RingMoment

f64, dev = torch.float64, 'cuda'
a_ = argparse.ArgumentParser()
a_.add_argument('--foam', type=Path, required=True)
a_.add_argument('--frames', type=Path, required=True)
a_.add_argument('--leg', type=int, default=0)
a_.add_argument('--k', type=int, nargs='+', default=[18, 19])
a_.add_argument('--top', type=float, default=.15)
args = a_.parse_args()
E, nu, C0 = 10648.614517785627/1000, .3912951704721983, -5.
g = np.load(args.foam/'dem_exact_single'/'geometry.npz'); cl = np.load(args.foam/'dem_exact_single'/'cells.npz')
X0, r0, V0, P = g['points'], cl['radii'], g['masses'], g['pairs']; n = len(X0)
Z = np.load(args.frames); h_ = float(np.median(V0)**(1/3))
zr = X0[:, 2]-X0[:, 2].min(); low = zr < .18
idx = -np.ones(n, int); idx[low] = np.arange(low.sum())
m = low[P[:, 0]] & low[P[:, 1]]; p = idx[P[m]]
_, lab = connected_components(coo_matrix((np.ones(len(p)), (p[:, 0], p[:, 1])), shape=(low.sum(), low.sum())), directed=False)
ids = np.nonzero(low)[0]
legs = sorted([ids[lab == k] for k in np.argsort(-np.bincount(lab))[:4]], key=lambda a: X0[a, 0].mean())
a = legs[args.leg]; a = a[zr[a] < args.top]
at = torch.as_tensor(a, device=dev); N = len(a)
s_ = zr[a].max()-zr[a]; Lb = float(s_.max())
nb = 8; ed = np.linspace(0., Lb, nb+1); Ax, Ay, Ar = [], [], []
for b in range(nb):
    k = a[(s_ >= ed[b]) & (s_ <= ed[b+1])]
    ds = ed[b+1]-ed[b]; v = V0[k]; rb = (3*v/(4*np.pi))**(1/3)
    Ar.append(v.sum()/ds)
    Ax.append((v*((X0[k, 0]-(v*X0[k, 0]).sum()/v.sum())**2+rb**2/5)).sum()/ds)
    Ay.append((v*((X0[k, 1]-(v*X0[k, 1]).sum()/v.sum())**2+rb**2/5)).sum()/ds)
A_, Ix, Iy = float(np.mean(Ar)), float(np.mean(Ax)), float(np.mean(Ay))
print('leg %d: %d free cells, length %.3f, mean area %.2e (width as a square %.3f), I about bending in x %.2e, in y %.2e; h %.4f' % (args.leg, N, Lb, A_, A_**.5, Ix, Iy, h_), flush=True)
beam = lambda I: 1.875**4*E*I/(A_*Lb**4)
print('beam, clamped-free: omega^2 bending in x %.3e, in y %.3e; axial %.3e, omega^2 of a uniform column; twist not counted' % (beam(Ix), beam(Iy), (np.pi/2)**2*E/Lb**2), flush=True)
W = RingMoment(X0, r0, V0, P, g['areas'], g['contacts'], E, nu, 1e9, theta=.5, kappa=5., eps_cut=-1.)
W.set_ball((0., 0., 1.), C0, 0.)
W.start(); W.prepare(); W.update_ul(W.X)
M = np.repeat(V0[a], 3)


def stiff(x, true):
    def force(y):
        W.grad(y)
        if true:
            W.reread_light()
            W.grad(y)
        return W.tgx[at].flatten().clone()
    f0 = force(x)
    K = torch.zeros(3*N, 3*N, dtype=f64)
    he = 1e-7
    for j in range(3*N):
        y = x.clone(); y[at[j//3], j % 3] += he
        K[:, j] = ((force(y)-f0)/he).cpu()
    asym = float((K-K.T).norm()/K.norm())
    return K.numpy(), asym


def report(x, tag):
    for true in (False, True):
        K, asym = stiff(x, true)
        w, Q = np.linalg.eig(K/M[:, None])                                                  # (not symmetrised: a nearly triangular operator's symmetric part has negative values its eigenvalues do not)
        o = np.argsort(w.real); w, Q = w[o], Q[:, o]
        kind = 'true tangent' if true else 'frozen R'
        print('%s, %s: asymmetry %.1e; real part negative: %d, the most %.0f; largest imaginary part %.0f; lowest real omega^2 %s' % (tag, kind, asym, int((w.real < -1e-3*abs(w.real).max()*0-1.).sum()), float(w.real[0]), float(abs(w.imag).max()), np.round(w.real[:8], 0).tolist()), flush=True)
        w = w.real; Q = Q.real
        tip = np.argsort(s_)[-max(N//8, 3):]
        for i in range(4):
            u = Q[:, i].reshape(N, 3); u = u/np.sqrt((M.reshape(N, 3)*u*u).sum())
            d = u[tip].mean(0)
            en = (M.reshape(N, 3)*u*u).sum(1); en = en/en.sum()
            print('    mode %d, omega^2 %.3e: tip moves %s (x, y, z share); lives on %.0f cells; share in the lowest third %.2f, in the top third %.2f' % (i, w[i], np.round(d*d/(d*d).sum(), 2).tolist(), 1/float((en*en).sum()), float(en[s_ > 2*Lb/3].sum()), float(en[s_ < Lb/3].sum())), flush=True)


W.grad(W.X); W.reread(); W.grad(W.X)
report(W.X.clone(), 'rest')
for k in args.k:
    x = torch.as_tensor(Z['x'][k], device=dev, dtype=f64)
    W._Ft = torch.as_tensor(Z['F'][k], device=dev, dtype=f64); W._Fe = W._Ft.clone()
    W.tul['Ln'].copy_(torch.as_tensor(Z['L'][k], device=dev, dtype=f64))
    W.update_ul(x); W.grad(x); W.reread(); W.grad(x)
    report(x, 't %.3f' % float(Z['t'][k]))
