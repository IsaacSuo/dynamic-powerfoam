"""ring_ul.py against what ring_checks.py established, on a small foam, in seconds: the gradient by differences, elastic and with the viscous part; a rigid motion reads nothing; the energy against the CPU form D; the free flight: energy, angular momentum.
  python free_surface/ring_ul_check.py --foam <dir with dem_exact_single and fem_modes_h0.0015.npz> [--periods 4]"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import warp as wp

from free_surface.closure_warp import MidpointStep, TrustCG
from free_surface.ring_ul import RingUL

sys.path.insert(0, '/home/fangsuo/py/dynamic-powerfoam')
from experiments.dem_fem_mode_match import interpolate

f64, dev = torch.float64, 'cuda'
a_ = argparse.ArgumentParser()
a_.add_argument('--foam', type=Path, required=True)
a_.add_argument('--periods', type=float, default=4.)
args = a_.parse_args()
E, nu = 23.00838, .3008
g = np.load(args.foam/'dem_exact_single'/'geometry.npz'); cl = np.load(args.foam/'dem_exact_single'/'cells.npz')
X0, r0, V0 = g['points'], cl['radii'], g['masses']; n = len(X0)
c_ = np.load(args.foam/'fem_modes_h0.0015.npz'); f_fem = c_['f']
F_ = dict(nodes=c_['nodes'], node_grid=c_['node_grid'], lo=c_['lo'], h=float(c_['h']), masses=c_['masses'])
Ux, _ = interpolate(F_, c_['V'][:, :3], X0)
phi = Ux[:, :, 0]+Ux[:, :, 2]; phi = phi/np.abs(phi).max()
w1 = (2*np.pi*f_fem[0])**2; T1 = 2*np.pi/np.sqrt(w1); h = float(np.median(V0))**(1/3)
dt = T1/60


def model(beta, ground=False):
    W = RingUL(X0, r0, V0, g['pairs'], g['areas'], g['contacts'], E, nu, theta=.5)
    W.set_ball((0., 0., 1.), -10., 0.)                                                # (far below: the step needs the arrays, nothing touches)
    W.set_viscosity(beta, dt)
    W.rest_cut()
    W.tbR.fill_(-1.)
    W.prepare()
    return W


W = model(0.)
x0 = torch.as_tensor(X0+.03*phi, dtype=f64, device=dev)                                # (a large deformation, 3 cm)
rng = np.random.default_rng(0); u = torch.as_tensor(rng.normal(size=X0.shape), dtype=f64, device=dev); e = 1e-6*h
print('%s, %d cells; springs %d; h %.2f mm; T1 %.4f s' % (args.foam.name, n, W._m, 1e3*h, T1), flush=True)
L0, g0 = W.grad(W.X)
print('at rest: E %.2e, |g| %.2e' % (float(L0), float(g0.abs().max())), flush=True)
Q = torch.tensor(np.linalg.qr(rng.normal(size=(3, 3)))[0], dtype=f64, device=dev); Q = Q*torch.sign(torch.linalg.det(Q))
xr = W.X@Q.T+torch.tensor([.3, -.2, .1], dtype=f64, device=dev)
Lr, gr = W.grad(xr)
L_, g_ = W.grad(x0); Lq, gq = W.grad(x0@Q.T)
print('a rigid motion of the rest shape: E %.2e, |g| %.2e; of the deformed: the variance and pressure read F_g from the rest cut, E %.8e against %.8e, |g| rotated off by %.2e of %.2e' % (float(Lr), float(gr.abs().max()), float(Lq), float(L_), float((gq-g_@Q.T).abs().max()), float(g_.abs().max())), flush=True)
fd = (float(W.grad(x0+e*u)[0])-float(W.grad(x0-e*u)[0]))/(2*e)
print('gradient, elastic: along a random direction %.8e, by differences %.8e, gap %.2e; net force %.1e, net moment %.1e' % (float((g_*u).sum()), fd, abs(float((g_*u).sum())-fd)/abs(fd), float(g_.sum(0).norm()), float(torch.linalg.cross(x0-x0.mean(0), g_).sum(0).norm())), flush=True)
print('energy terms at the deformed state: springs %.5e, volume %.5e, variance %.5e, outer %.5e; total %.5e (CPU form D at 3 cm: not this start)' % (float(W.tsum[0]), float(W.tsum[1]), float(W.tsum[2]), float(W.tsum[5]), float(L_)), flush=True)
del W
Wv = model(.002)
Wv.grad(x0*0+Wv.X+.01*torch.as_tensor(phi, dtype=f64, device=dev))
T, t = Wv.tr, Wv.tul
T['min'].copy_(T['mi']); T['mjn'].copy_(T['mj']); T['Hn'].copy_(T['Hr']); t['Ln'].copy_(Wv.tL)                     # (a reference state, a step away)
L_, g_ = Wv.grad(x0)
fd = (float(Wv.grad(x0+e*u)[0])-float(Wv.grad(x0-e*u)[0]))/(2*e)
print('gradient, with the viscous part (beta .002, its energy %.3e of %.3e): %.8e, by differences %.8e, gap %.2e; net moment %.1e' % (float(Wv.tsum[7]), float(L_), float((g_*u).sum()), fd, abs(float((g_*u).sum())-fd)/abs(fd), float(torch.linalg.cross(x0-x0.mean(0), g_).sum(0).norm())), flush=True)
del Wv
# the free flight of ring_checks.py --test 10, form D
d_ = phi[g['pairs'][:, 1].astype(int)]-phi[g['pairs'][:, 0].astype(int)]
W = model(0.)
mm = np.maximum(V0, .1*np.median(V0))
mass = torch.as_tensor(mm, dtype=f64, device=dev); M3 = mass[:, None]
Om = 2*np.pi/(5*T1)*np.array([.3, .5, .8])/np.linalg.norm([.3, .5, .8])
cm0 = (mm[:, None]*X0).sum(0)/mm.sum()
phi0 = phi-(mm[:, None]*phi).sum(0)/mm.sum()
amp = .5*1.356e-2                                                                      # (test 10's: strain 99 % about 5 %)
v0 = np.cross(Om, X0-cm0)+amp*np.sqrt(w1)*phi0
fg = torch.zeros(n, 3, dtype=f64, device=dev)
CG = TrustCG(W, mass, dt, per=2, product_iters=1)
ST = MidpointStep(W, CG, mass, W.ul_blocks(), fg, dt, 0.)
ST.pre_ul = True; ST.pre_every = 4; ST.smooth_pred = ST.carry_delta = True; ST.newton_max = 100
ST.set_theta(.5, mass)
ST.set_state(W.X.clone(), torch.as_tensor(v0, dtype=f64, device=dev), fg)
W.store = dict(springs=0., vol=0., var=0., outer=0.)
tol = 1e-8*float(torch.as_tensor(E*h*h))
L0_ = (mass[:, None]*torch.linalg.cross(ST.tx-ST.tx.mean(0)*0, ST.tv)).sum(0)
cm = (M3*ST.tx).sum(0)/mass.sum(); vm = (M3*ST.tv).sum(0)/mass.sum()
Lc0 = (M3*torch.linalg.cross(ST.tx-cm, ST.tv-vm)).sum(0)
E0 = .5*float((M3*ST.tv**2).sum())
print('free flight: start energy %.5e (CPU form D 9.997e-06), tol %.1e; dt T1 / 60' % (E0, tol), flush=True)
N = int(args.periods*60); t0 = time.time(); nit = npr = 0
for s in range(1, N+1):
    e_, its, g2 = ST.step(tol, lambda g__: .5*tol)
    ST.end(); nit += its; npr += e_
    if s % max(1, N//12) == 0 or s == N:
        ek = .5*float((M3*ST.tv**2).sum()); ed, ev = W.stored()
        cm = (M3*ST.tx).sum(0)/mass.sum(); vm = (M3*ST.tv).sum(0)/mass.sum()
        Lc = (M3*torch.linalg.cross(ST.tx-cm, ST.tv-vm)).sum(0)
        print('  t / T1 %.2f: energy / start %.5f (potential %.3e, kinetic %.3e); angular momentum off by %.2e, along it %.5f; momentum %.1e; |g| %.1e; Newton per step %.1f, products per step %.0f, rejected %d; %.1f s' % (s*dt/T1, (ek+ed+ev)/E0, ed+ev, ek, float((Lc-Lc0).norm()/Lc0.norm()), float((Lc*Lc0).sum()/(Lc0*Lc0).sum()), float((M3*ST.tv).sum(0).norm()), math.sqrt(g2), nit/s, npr/s, ST.stats['rejected'], time.time()-t0), flush=True)
print('done', flush=True)
