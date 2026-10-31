"""static checks of spring_plastic.py on a foam, none of them a simulation. Rest: E and dE/dx 0. Rigid motion: E and dE/dx ~ 0. Uniform deviatoric strain below the limit: E against mu |e|^2 V (the identity
sum A L n n + sum r N = V I). Above the limit: dE/d|e| against s_y V. Gradient against central differences, below and above the limit, on a smooth random field. The return: after end_ul at a state above the limit,
every yielded cell's store is at the limit at the same x, with the same rows and with the rows read from the cut; the force jump of that re-read.
  python free_surface/spring_plastic_check.py --foam <dir with dem_exact_single>"""
import argparse
import math
from pathlib import Path

import numpy as np
import torch

from free_surface.spring_plastic import SpringPlastic

f64, dev = torch.float64, 'cuda'
a_ = argparse.ArgumentParser()
a_.add_argument('--foam', type=Path, required=True)
a_.add_argument('--E', type=float, default=23.00838)
a_.add_argument('--nu', type=float, default=.3008)
a_.add_argument('--ey', type=float, default=.01)
args = a_.parse_args()
g = np.load(args.foam/'dem_exact_single'/'geometry.npz'); cl = np.load(args.foam/'dem_exact_single'/'cells.npz')
X0, r0, V0 = g['points'], cl['radii'], g['masses']; n = len(X0)
W = SpringPlastic(X0, r0, V0, args.E, args.nu, args.E*args.ey)
W.set_ball((0., 1., 0.), -100., 0.)
W.update_ul(W.X)
W.prepare()
X = W.X; Vt = float(W.V0.sum()); xc = X.mean(0); mu, sy, ey = W.mu, W.sy, W.ey
print('foam %d cells, %d rest halves; mu %.4f, s_y %.4f, e_y %.5f, kap %.4f; volumes: G.V / masses %.6f .. %.6f' % (n, len(W._rest[0]), mu, sy, ey, W.kap, float((W._ulG.V.detach()/W.V0).min()), float((W._ulG.V.detach()/W.V0).max())))
E, gr = W.grad(X)
print('rest: E %.3e, max |g| %.3e' % (float(E), float(gr.norm(dim=1).max())))
Rq = torch.linalg.qr(torch.as_tensor(np.random.default_rng(1).normal(size=(3, 3)), device=dev))[0]
Rq = Rq*torch.sign(torch.linalg.det(Rq))
x = (X-xc)@Rq.T+xc+.01
W.set_rotation(x)
E, gr = W.grad(x)
print('rigid motion (Q read at x): E %.3e against the store at e_y %.3e; max |g| %.3e against one wall at the limit %.3e' % (float(E), mu*ey*ey*Vt, float(gr.norm(dim=1).max()), 2*mu*ey*float(W._ulG.wall_A.detach().max())))
W.tul['Q'].copy_(W.I3.expand(n, 3, 3))
e0 = {}
for name, v in dict(uniaxial=(2, -1, -1), shear=(1, -1, 0)).items():
    v = torch.tensor(v, dtype=f64, device=dev); v = v/v.norm(); e0[name] = Rq@torch.diag(v)@Rq.T


def strained(e, rot=None):
    Fx = torch.matrix_exp(e)
    y = (X-xc)@Fx.T+xc
    return y if rot is None else (y-xc)@rot.T+xc


for name, e_ in e0.items():
    for a in (.1, .5):
        x = strained(a*ey*e_)
        E, _ = W.grad(x)
        el = mu*(a*ey)**2*Vt
        print('%s, |e| = %.1f e_y: E / mu |e|^2 V %.5f; cells over the limit %d; e_i / |e| 1%% %.3f, median %.3f, 99%% %.3f' % (name, a, float(W.tsum[0])/el, int((W.tbe < 1).sum()), *[float(torch.quantile(W.tei/(a*ey), q_)) for q_ in (.01, .5, .99)]))
    for a in (3, 10, 30):
        h_ = .01*a*ey
        Ep = (W.grad(strained((a*ey+h_)*e_))[0], float(W.tsum[0]))[1]
        Em = (W.grad(strained((a*ey-h_)*e_))[0], float(W.tsum[0]))[1]
        print('%s, |e| = %2d e_y: d(E_dev)/d|e| / (s_y V) %.4f' % (name, a, (Ep-Em)/(2*h_)/(sy*Vt)))
torch.manual_seed(0)
f = torch.rand(6, 3, dtype=f64, device=dev)*2-1; kk = torch.rand(6, 3, dtype=f64, device=dev)*12


def smooth(y):
    return sum(f[k_][None, :]*torch.sin((y*kk[k_]).sum(1)+k_)[:, None] for k_ in range(6))


for name, amp in (('below the limit', .1*ey), ('some cells over', 1.*ey), ('most cells over', 10*ey)):
    x = X+amp*smooth(X)*.05+.0
    W.tul['Q'].copy_(W.I3.expand(n, 3, 3))
    E, gr = W.grad(x)
    over = int((W.tbe < 1).sum())
    d = torch.randn_like(x); d = d/d.norm()
    for h_ in (1e-6, 1e-7):
        ep, em = float(W.grad(x+h_*d)[0]), float(W.grad(x-h_*d)[0])
        print('%s (%d over): gradient . d %.8e, differences (h %.0e) %.8e, relative %.2e' % (name, over, float((gr*d).sum()), h_, (ep-em)/(2*h_), abs((ep-em)/(2*h_)-float((gr*d).sum()))/abs(float((gr*d).sum()))))
# the return: strain over the limit, the step ended, the same x read again from the cut; one model each, from rest
for a in (1.5, 3, 10):
    W = SpringPlastic(X0, r0, V0, args.E, args.nu, args.E*args.ey)
    W.set_ball((0., 1., 0.), -100., 0.)
    W.update_ul(W.X)
    W.prepare()
    x = strained(a*ey*e0['uniaxial'])
    E0, g0 = W.grad(x)
    g0 = g0.clone()
    W.end_ul(1.)
    yl = W.ul_stats['yielding']
    W.update_ul(x)
    E1, g1 = W.grad(x)
    sm = float(W.sy*(W._ulG.wall_A.detach()).sum()*2/n)                             # (a cell's sum of |limit force|, about)
    print('|e| = %.1f e_y: yielding %d of %d; re-read from the cut: rows %d (rest %d), born %d, closed %d; e_i / e_y 1%% %.4f, median %.4f, 99%% %.4f; dev stored / limit %.4f' % (a, yl, n, W.ul_stats['rows'], len(W._rest[0]), W.ul_stats['born'], W.ul_stats['closed'],
          *[float(torch.quantile(W.tei/ey, q_)) for q_ in (.01, .5, .99)], float(W.tsum[0])/(mu*ey*ey*Vt)))
    print('   force jump at the re-read |g1 - g0| / (a cell\'s sum of limit forces): median %.4f, 99%% %.4f, max %.4f; |g0| the same way: median %.4f, 99%% %.4f' % (*[float(torch.quantile((g1-g0).norm(dim=1), q_))/sm for q_ in (.5, .99)], float((g1-g0).norm(dim=1).max())/sm, float(g0.norm(dim=1).median())/sm, float(torch.quantile(g0.norm(dim=1), .99))/sm))
print('done')
