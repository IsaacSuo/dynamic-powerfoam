"""static checks of the ground's anchor in merged_ul.py on a foam, no simulation. A shear linear through 0 at the ground: alpha reads it, so the glue's force is 0 on sites with a layer above; a translation: (1 - alpha) of the pinned site's. Gradient against
differences, glue holding and slipping. The glue's data as k_mid_ball would write them, the ground at the foam's lower part.
  python free_surface/merged_anchor_check.py --foam <dir with dem_exact_single>"""
import argparse
from pathlib import Path

import numpy as np
import torch
import warp as wp

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
W.set_ball((0., 1., 0.), y0, 1.)
W.set_anchor()
W.update_ul(W.X)
W.prepare()
W.tball[0], W.tball[1] = 2., .5
X = W.X
tt = lambda a__: wp.to_torch(a__, requires_grad=False)
bl = W.ball; xn, kt, lamt = tt(bl['xn']), tt(bl['kt']), tt(bl['lamt'])
xn.copy_(X)
R_ = W.tbR
dn = (R_-(X[:, 1]-y0)).clamp(min=0.)
touch = dn > 0
sq = torch.sqrt(R_.clamp(min=0.)*dn)
kt.copy_(8*W.Gs*sq); W.tbbound.copy_(torch.where(touch, torch.full_like(R_, 1e3), torch.zeros_like(R_)))
W.read_anchor(X)
ai, aj, aw = W.tan['ai'], W.tan['aj'], W.tan['aw']
k3 = int((aw > 0).sum())
al = torch.zeros(n, dtype=f64, device=dev).index_add(0, ai[:k3].long(), aw[:k3])
has = al > 0
print('touching %d of %d; with a layer above %d, alpha median %.3f, 10 %% %.3f, 90 %% %.3f, max %.3f; entries %d of %d; height of the touching sites / h median %.2f' % (
    int(touch.sum()), n, int((touch & has).sum()), float(al[touch & has].median()), float(torch.quantile(al[touch & has], .1)), float(torch.quantile(al[touch & has], .9)), float(al.max()), k3, W.an['K'], float((X[touch, 1]-y0).median())/float(W.V0.median()**(1/3))), flush=True)
# a shear linear through 0 at the ground: x_e - x_n = gamma H e_x, so z = x_n + th (x_e - x_n)
gam = 1e-3
ex = torch.tensor([1., 0., 0.], dtype=f64, device=dev)
z = X+.5*gam*(X[:, 1]-y0)[:, None]*ex
W.grad(z)
ff = tt(W.B['bff'])
pin = touch & ~has; hold = touch & has
print('shear linear through the ground, gamma %.0e: the force / k gamma H: with a layer above max %.2e; pinned (none above, alpha 0) median %.3f' % (gam, float((ff[hold].norm(dim=1)/(kt[hold]*gam*(X[hold, 1]-y0))).max()), float((ff[pin].norm(dim=1)/(kt[pin]*gam*(X[pin, 1]-y0))).median()) if bool(pin.any()) else 0.), flush=True)
# a translation: (1 - alpha) k u
u = 1e-5
z = X+.5*u*ex
W.grad(z)
print('translation: the force / (1 - alpha) k u: with a layer above min %.6f, max %.6f' % (float((ff[hold].norm(dim=1)/((1-al[hold])*kt[hold]*u)).min()), float((ff[hold].norm(dim=1)/((1-al[hold])*kt[hold]*u)).max())), flush=True)
# gradient against differences, holding and slipping
torch.manual_seed(0)
for name, bound_ in (('holding', 1e3), ('slipping', 1e-9)):
    W.tbbound.copy_(torch.where(touch, torch.full_like(R_, bound_), torch.zeros_like(R_)))
    x = X+1e-4*torch.randn(n, 3, dtype=f64, device=dev)
    E, gr = W.grad(x)
    d = torch.randn_like(x); d = d/d.norm()
    h_ = 1e-7
    ep, em = float(W.grad(x+h_*d)[0]), float(W.grad(x-h_*d)[0])
    print('%s: gradient . d %.8e, differences %.8e, relative %.2e; glue term %.3e' % (name, float((gr*d).sum()), (ep-em)/(2*h_), abs((ep-em)/(2*h_)-float((gr*d).sum()))/abs(float((gr*d).sum())), float(W.tsum[8])), flush=True)
print('done')
