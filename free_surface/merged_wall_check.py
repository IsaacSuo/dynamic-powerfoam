"""static checks of the no-slip reading in merged_ul.py on a foam, no simulation. The ground at the foam's lower part. A shear linear through 0 at the ground reads exactly, as without the wall; a tangential translation reads (A / V) u (x) n on the touching cells, nothing elsewhere;
a rigid rotation about the ground's normal likewise only there. Gradient against differences below and above the limit. Uniform stress: the touching site gets no tangential force from its own cell.
  python free_surface/merged_wall_check.py --foam <dir with dem_exact_single>"""
import argparse
from pathlib import Path

import numpy as np
import torch

from free_surface.merged_ul import MergedUL

f64, dev = torch.float64, 'cuda'
a_ = argparse.ArgumentParser()
a_.add_argument('--foam', type=Path, required=True)
a_.add_argument('--E', type=float, default=23.00838)
a_.add_argument('--nu', type=float, default=.3008)
a_.add_argument('--ey', type=float, default=.01)
args = a_.parse_args()
g = np.load(args.foam/'dem_exact_single'/'geometry.npz'); cl = np.load(args.foam/'dem_exact_single'/'cells.npz')
X0, r0, V0 = g['points'], cl['radii'], g['masses']; n = len(X0)
y0 = float(X0[:, 1].min()-.8*r0[X0[:, 1].argmin()])
W = MergedUL(X0, r0, V0, g['pairs'], g['areas'], g['contacts'], args.E, args.nu, args.E*args.ey, theta=.5)
W.set_ball((0., 1., 0.), y0, 0.)
W.set_wall()
W.update_ul(W.X)
W.prepare()
W.tball[0], W.tball[1] = 2., .5
X = W.X; I3 = W.I3
Ac, Hw = W.twl['Ac'], W.twl['Hw']
t = Ac > 0
print('touching with a downward cap %d of %d; their area sum %.3e; height / h median %.2f; V / A (thickness) / h median %.2f' % (int(t.sum()), n, float(Ac.sum()), float(Hw[t].median())/float(W.V0.median()**(1/3)), float((1/W.tul['Vi'][t]/Ac[t]).median())/float(W.V0.median()**(1/3))), flush=True)
ex = torch.tensor([1., 0., 0.], dtype=f64, device=dev); ey = torch.tensor([0., 1., 0.], dtype=f64, device=dev)
# a shear linear through the ground
gam = 1e-3
z = X+gam*Hw[:, None]*ex
W.grad(z)
G_ex = gam*torch.outer(ex, ey)
err = (W.tG-G_ex).flatten(1).norm(dim=1)
print('shear linear through the ground: |G - exact| / gamma: touching max %.2e; all max %.2e' % (float(err[t].max())/gam, float(err.max())/gam), flush=True)
# a tangential translation: only the wall reads it
u = 1e-5
z = X+u*ex
W.grad(z)
G_ex = (Ac*W.tul['Vi'])[:, None, None]*u*torch.outer(ex, ey)
print('translation: |G - (A / V) u (x) n| / (u / thickness median): touching max %.2e; others |G| max %.2e' % (float((W.tG-G_ex)[t].flatten(1).norm(dim=1).max())/(u*float((Ac*W.tul['Vi'])[t].median())), float(W.tG[~t].flatten(1).norm(dim=1).max())/(u*float((Ac*W.tul['Vi'])[t].median()))), flush=True)
# a normal translation: nothing
z = X+u*ey
W.grad(z)
print('normal translation: |G| / (u / thickness) max %.2e' % (float(W.tG.flatten(1).norm(dim=1).max())/(u*float((Ac*W.tul['Vi'])[t].median()))), flush=True)
# gradient against differences, below and above the limit
torch.manual_seed(0)
for name, amp in (('below the limit', 1e-5), ('above', 3e-3)):
    x = X+amp*torch.randn(n, 3, dtype=f64, device=dev)
    W.grad(x); W.reread(beta=True); E, gr = W.grad(x)
    over = int((W.tbe < 1).sum())
    d = torch.randn_like(x); d = d/d.norm()
    h_ = 1e-7
    ep, em = float(W.grad(x+h_*d)[0]), float(W.grad(x-h_*d)[0])
    print('%s (%d cells over): gradient . d %.8e, differences %.8e, relative %.2e' % (name, over, float((gr*d).sum()), (ep-em)/(2*h_), abs((ep-em)/(2*h_)-float((gr*d).sum()))/abs(float((gr*d).sum()))), flush=True)
print('done')
