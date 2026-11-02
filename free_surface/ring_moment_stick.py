"""Does a wall with a weightless spring hold the limbs? The lion's elastic state rebuilt at a saved frame (q, w, outer layer at rest; x, F, L from the frame), the force on one cut with those walls connected and with them free surfaces.
  python free_surface/ring_moment_stick.py --foam <dir> --frames <frames.npz> --k 48"""
import argparse
from pathlib import Path

import numpy as np
import torch

from free_surface.ring_moment import RingMoment

f64, dev = torch.float64, 'cuda'
a_ = argparse.ArgumentParser()
a_.add_argument('--foam', type=Path, required=True)
a_.add_argument('--frames', type=Path, required=True)
a_.add_argument('--k', type=int, nargs='+', default=[30, 48, 72])
args = a_.parse_args()
E, nu, C0 = 10648.614517785627/1000, .3912951704721983, -.2
g = np.load(args.foam/'dem_exact_single'/'geometry.npz'); cl = np.load(args.foam/'dem_exact_single'/'cells.npz')
X0, r0, V0 = g['points'], cl['radii'], g['masses']; n = len(X0)
Z = np.load(args.frames)
h = float(np.median(V0)**(1/3))


def state(k, births):
    W = RingMoment(X0, r0, V0, g['pairs'], g['areas'], g['contacts'], E, nu, 1e9, theta=.5, kappa=5., eps_cut=-1.)
    W.set_ball((0., 0., 1.), C0, 0.)
    W.start(); W.prepare()
    W.update_ul(W.X)
    W.elastic_births = births
    F = torch.as_tensor(Z['F'][k], device=dev, dtype=f64); x = torch.as_tensor(Z['x'][k], device=dev, dtype=f64)
    W._Ft = F.clone(); W._Fe = F.clone()
    W.tul['Ln'].copy_(torch.as_tensor(Z['L'][k], device=dev, dtype=f64))
    W.update_ul(x)
    W.grad(x); W.reread(); _, f = W.grad(x)
    return W, x, f.clone()


for k in args.k:
    WA, x, fA = state(k, True)
    ms = WA._m_s; lo, hi = WA.tlo, WA.thi
    wt = WA._wi+WA._wj
    zero = wt == 0
    i, j = lo[zero], hi[zero]
    rest = torch.as_tensor(X0, device=dev, dtype=f64)
    keyA = WA.tkey.clone(); LA = WA.tul['Ln'].clone()
    WB, _, fB = state(k, False)
    df = fA-fB                                                                             # (dE/dx: the force is its negative)
    weight = 9.8*torch.as_tensor(V0, device=dev, dtype=f64)
    d = x[j]-x[i]; dn = d/d.norm(dim=1, keepdim=True).clamp(min=1e-12)
    fi, fj = -df[i], -df[j]
    attract = ((fi*dn).sum(1)-(fj*dn).sum(1))
    rm = (rest[j]-rest[i]).norm(dim=1)/h
    print('t %.3f: weightless springs with a wall %d; rest distance / h median %.1f, 90 %% %.1f; open walls without them %d' % (float(Z['t'][k]), int(zero.sum()), float(rm.median()), float(torch.quantile(rm, .9)), WB.ul_stats.get('open_walls', 0)), flush=True)
    print('  added force |df| / own weight: median %.2f, 90 %% %.2f, 99 %% %.2f, max %.1f; total |f| / own weight, median %.2f; cells it touches %d' % (float((df.norm(dim=1)/weight).median()), float(torch.quantile(df.norm(dim=1)/weight, .9)), float(torch.quantile(df.norm(dim=1)/weight, .99)), float((df.norm(dim=1)/weight).max()), float((fA.norm(dim=1)/weight).median()), int((df.norm(dim=1) > 1e-9*float(weight.median())).sum())), flush=True)
    print('  along those pairs, + pulls them together: sum %+.3e; share that pull %.2f; of the pull, by the pairs |.| / weight median %.2f max %.1f; L of those cells median %.3f, of all %.3f' % (float(attract.sum()), float((attract > 0).double().mean()), float((attract.abs()/weight[i]).median()), float((attract.abs()/weight[i]).max()), float(torch.cat([LA[i], LA[j]]).median()), float(LA.median())), flush=True)
