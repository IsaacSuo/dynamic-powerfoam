"""static checks of merged_ul.py on a foam, no simulation. Rest and rigid motion: E, dE/dx ~ 0. Uniform deviatoric strain below the limit: E against mu |e|^2 V; above it: dE/d|e| against theta s_y V (the first step from rest).
Gradient against central differences, below and above the limit, first step and with memories. A closed path at no limit: x goes out along one strain path and back along another in many steps: F and the remainder must come back. The return: after
end_ul above the limit every yielded cell's store at the limit at the step's end, from the cut; the force jump between the end of a step and the next start at the same x (theta 1).
  python free_surface/merged_check.py --foam <dir with dem_exact_single>"""
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
a_.add_argument('--theta', type=float, default=.5)
args = a_.parse_args()
g = np.load(args.foam/'dem_exact_single'/'geometry.npz'); cl = np.load(args.foam/'dem_exact_single'/'cells.npz')
X0, r0, V0 = g['points'], cl['radii'], g['masses']; n = len(X0); th = args.theta


def model(sigma_y):
    W = MergedUL(X0, r0, V0, g['pairs'], g['areas'], g['contacts'], args.E, args.nu, sigma_y, theta=th)
    W.set_ball((0., 1., 0.), -100., 0.)
    W.update_ul(W.X)
    W.prepare()
    return W


W = model(args.E*args.ey)
X = W.X; Vt = float(W.V0.sum()); xc = X.mean(0); mu, sy, ey = W.mu, W.sy, W.ey
print('foam %d cells, %d springs; mu %.4f, s_y %.4f, e_y %.5f, kap %.4f' % (n, len(W.tlo), mu, sy, ey, W.kap))
E, gr = W.grad(X)
print('rest: E %.3e, max |g| %.3e' % (float(E), float(gr.norm(dim=1).max())))
Rq = torch.linalg.qr(torch.as_tensor(np.random.default_rng(1).normal(size=(3, 3)), device=dev))[0]
Rq = Rq*torch.sign(torch.linalg.det(Rq))
x = (X-xc)@Rq.T+xc+.01
W.grad(x); W.reread()
E, gr = W.grad(x)
print('rigid motion: E %.3e against the store at e_y %.3e; max |g| %.3e against one wall at the limit %.3e' % (float(E), mu*ey*ey*Vt, float(gr.norm(dim=1).max()), 2*mu*ey*float(W._ulG.wall_A.detach().max())))
e0 = {}
for name, v in dict(uniaxial=(2, -1, -1), shear=(1, -1, 0)).items():
    v = torch.tensor(v, dtype=f64, device=dev); v = v/v.norm(); e0[name] = Rq@torch.diag(v)@Rq.T


def strained(e):
    return (X-xc)@torch.matrix_exp(e).T+xc


def fresh(x_):
    W.tul['Q'].copy_(W.I3.expand(n, 3, 3)); W.tsp['Rh'].copy_(W.I3.expand(n, 3, 3)); W.tsp['sb'].fill_(1.)
    W.grad(x_); W.reread()
    W.tsp['sb'].fill_(1.)
    return W.grad(x_)


for name, e_ in e0.items():
    for a in (.1, .4):
        x = strained(a*ey*e_)
        fresh(x)
        el = mu*(a*ey)**2*Vt
        print('%s, |e| = %.1f e_y: shape energy + the start\'s part / mu |e|^2 V %.5f; spring energy / it %.2e; cells over the limit %d' % (name, a, float(W.tsum[0]+W.tsum[3])/el, float(W.tsum[5])/el, int((W.tbe < 1).sum())))
    for a in (3, 10):
        h_ = .01*a*ey
        fresh(strained((a*ey+h_)*e_)); Ep = float(W.tsum[0])
        fresh(strained((a*ey-h_)*e_)); Em = float(W.tsum[0])
        print('%s, |e| = %2d e_y: d(E_shape)/d|e| / (theta s_y V) %.4f' % (name, a, (Ep-Em)/(2*h_)/(th*sy*Vt)))
torch.manual_seed(0)
f = torch.rand(6, 3, dtype=f64, device=dev)*2-1; kk = torch.rand(6, 3, dtype=f64, device=dev)*12


def smooth(y):
    return sum(f[k_][None, :]*torch.sin((y*kk[k_]).sum(1)+k_)[:, None] for k_ in range(6))


def gradcheck(x, tag):
    E, gr = W.grad(x)
    W.reread(beta=True)
    E, gr = W.grad(x)
    over = int((W.tbe < 1).sum()); nb = int((W.tsp['sb'][:W._m_s] < 1).sum())
    d = torch.randn_like(x); d = d/d.norm()
    h_ = 1e-7
    ep, em = float(W.grad(x+h_*d)[0]), float(W.grad(x-h_*d)[0])
    print('%s (%d cells over, %d springs below 1): gradient . d %.8e, differences %.8e, relative %.2e' % (tag, over, nb, float((gr*d).sum()), (ep-em)/(2*h_), abs((ep-em)/(2*h_)-float((gr*d).sum()))/abs(float((gr*d).sum()))))


for name, amp in (('first step, below the limit', .1*ey), ('first step, some over', 1.*ey), ('first step, most over', 10*ey)):
    W.tul['Q'].copy_(W.I3.expand(n, 3, 3)); W.tsp['Rh'].copy_(W.I3.expand(n, 3, 3))
    gradcheck(X+amp*smooth(X)*.05, name)
# a closed path at no limit
W = model(float('inf'))
print('closed path, no limit (%s):' % ('midpoint' if th == .5 else 'backward Euler'))
for N in (20, 40, 80):
    W = model(float('inf')); s_ = 0.
    x_ = X.clone()
    A1, A2 = .25*e0['uniaxial'], .25*e0['shear']+.2*torch.tensor([[0, 1, 0], [-1, 0, 0], [0, 0, 0]], dtype=f64, device=dev)
    def at(u):                                                                      # (a loop in strain space, with rotation in it, and a smooth part)
        return (X-xc)@torch.matrix_exp(np.sin(np.pi*u)*A1+np.sin(2*np.pi*u)*A2).T+xc+np.sin(np.pi*u)*.02*smooth(X)
    for k in range(N):
        x1 = at((k+1)/N)
        W.update_ul(x_)
        W.grad(x_+(x1-x_)*th); W.reread(); W.grad(x_+(x1-x_)*th)
        W.end_ul(th)
        x_ = x1
    Fe = W._Fe; dF = (Fe-W.I3).flatten(1).norm(dim=1)
    W.update_ul(x_)
    E, gr = W.grad(x_)
    print('  %d steps: |F - I| median %.2e, 99%% %.2e, max %.2e; ln J max %.2e; remainder / L max %.2e; E %.3e, against the peak store mu |e|^2 V %.3e; walls %d, springs without a wall %d, new %d' % (
        N, float(dF.median()), float(torch.quantile(dF, .99)), float(dF.max()), float(W.tul['Ln'].abs().max()), float((W.tsp['sdn'][:W._m_s].norm(dim=1)/(W.tT.norm(dim=1))).max()), float(E), mu*.25**2*Vt, W.ul_stats['walls'], W.ul_stats['ghost'], W.ul_stats['born']))
# the return, and a step with memories
for a in (1.5, 3, 10):
    W = model(args.E*args.ey)
    x = strained(a*ey*e0['uniaxial'])
    z = X+(x-X)*th
    W.grad(z); W.reread(); W.grad(z)
    W.end_ul(th)
    yl = W.ul_stats['yielding']
    W.update_ul(x)
    W.grad(x); W.reread(); W.grad(x)
    print('|e| = %.1f e_y: yielding %d of %d, D_min %.2e; the end read from the cut: springs %d, born %d, closed %d; e_i / e_y 1%% %.4f, median %.4f, 99%% %.4f; spring energy / limit store %.2e' % (a, yl, n, W.ul_stats['D_min'], W._m_s, W.ul_stats['born'], W.ul_stats['closed'],
          *[float(torch.quantile(W.tei/ey, q_)) for q_ in (.01, .5, .99)], float(W.tsum[5])/(mu*ey*ey*Vt)))
    gradcheck(x+.01*a*ey*smooth(x)*.05, '  a step with memories')
    gradcheck(x+a*ey*smooth(x)*.05, '  a step with memories, larger')
print('done')
