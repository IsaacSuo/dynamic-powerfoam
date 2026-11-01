"""ring_plastic.py on a small foam, in a minute: at no strain nothing reads; a uniform motion of the sites: the read, the limit; the force against E by differences, elastic, flowing, far past the limit; the free flight of ring_ul_check.py with the cut each step and a yield limit: energy + plastic work, angular momentum, Newton.
  python free_surface/ring_plastic_check.py --foam <dir with dem_exact_single and fem_modes_h0.0015.npz> [--periods 1.5] [--ey 0.0095] [--rest-walls] [--stop-bad]"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import warp as wp

from free_surface.closure_warp import TrustCG
from free_surface.ring_plastic import RingPlastic, RingStep

sys.path.insert(0, '/home/fangsuo/py/dynamic-powerfoam')
from experiments.dem_fem_mode_match import interpolate

f64, dev = torch.float64, 'cuda'
a_ = argparse.ArgumentParser()
a_.add_argument('--foam', type=Path, required=True)
a_.add_argument('--periods', type=float, default=1.5)
a_.add_argument('--ey', type=float, default=.0095)
a_.add_argument('--rest-walls', action='store_true', help='keep every spring at its rest wall: no spring list changes, for comparison')
a_.add_argument('--stop-bad', action='store_true', help='stop at the first step whose solve does not converge')
a_.add_argument('--kappa', type=float, default=2.)
args = a_.parse_args()
E, nu = 23.00838, .3008
mu = E/(2*(1+nu))
sy = 2*mu*args.ey; sigma_y = sy/math.sqrt(2/3)
g = np.load(args.foam/'dem_exact_single'/'geometry.npz'); cl = np.load(args.foam/'dem_exact_single'/'cells.npz')
X0, r0, V0 = g['points'], cl['radii'], g['masses']; n = len(X0)
c_ = np.load(args.foam/'fem_modes_h0.0015.npz'); f_fem = c_['f']
F_ = dict(nodes=c_['nodes'], node_grid=c_['node_grid'], lo=c_['lo'], h=float(c_['h']), masses=c_['masses'])
Ux, _ = interpolate(F_, c_['V'][:, :3], X0)
phi = Ux[:, :, 0]+Ux[:, :, 2]; phi = phi/np.abs(phi).max()
w1 = (2*np.pi*f_fem[0])**2; T1 = 2*np.pi/np.sqrt(w1); h = float(np.median(V0))**(1/3)
dt = T1/60
W = RingPlastic(X0, r0, V0, g['pairs'], g['areas'], g['contacts'], E, nu, sigma_y, theta=.5, kappa=args.kappa)
W.set_ball((0., 0., 1.), -10., 0.)
W.start()
W.tbR.fill_(-1.)
W.prepare()
print('%s, %d cells; springs %d; h %.2f mm; T1 %.4f s; yield strain %.2f %%' % (args.foam.name, n, W._m_s, 1e3*h, T1, 100*W.ey), flush=True)
L0, g0 = W.grad(W.X)
print('at rest: E %.2e, |g| %.2e; read %.2e, beta min %.3f' % (float(L0), float(g0.abs().max()), float(W.tr['rho'].max()), float(W.tbe.min())), flush=True)
T = W.tr
closed = W.tul['Mc'].abs().sum((1, 2)) == 0
for gam in (.5*W.ey*2**.5, 2*W.ey*2**.5, .05):
    Sh = torch.tensor([[1., gam, 0.], [0., 1., 0.], [0., 0., 1.]], dtype=f64, device=dev)
    Xc = W.X-W.X.mean(0)
    _, g_ = W.grad(Xc@Sh.T+W.X.mean(0))
    r_ = T['rho'][closed]/(gam/2**.5)
    print('uniform shear %.4f: rho / |e| on the closed cells: 1 %% %.3f, median %.3f, 99 %% %.3f; rho / e_y median %.3f; beta %.4f..%.4f; |g| on the closed cells / (s_y A median) median %.2e, max %.2e' % (gam, float(torch.quantile(r_, .01)), float(r_.median()), float(torch.quantile(r_, .99)), float(T['rho'].median())/W.ey, float(W.tbe.min()), float(W.tbe.max()), float(g_[closed].norm(dim=1).median())/(sy*float(np.median(g['areas']))), float(g_[closed].norm(dim=1).max())/(sy*float(np.median(g['areas'])))), flush=True)
rng = np.random.default_rng(0); u = torch.as_tensor(rng.normal(size=X0.shape), dtype=f64, device=dev); e = 1e-6*h
for name, amp_ in (('below and past the limit', .002), ('far past it', .03)):
    xx = W.X+amp_*torch.as_tensor(phi, dtype=f64, device=dev)
    L_, g_ = W.grad(xx)
    bm = float(W.tbe.min()); fl = int((W.tbe < 1).sum())
    fd = (float(W.grad(xx+e*u)[0])-float(W.grad(xx-e*u)[0]))/(2*e)
    print('the force against E by differences, %s (%d cells flowing, beta min %.3f): %.8e, %.8e, gap %.2e; net moment %.1e' % (name, fl, bm, float((g_*u).sum()), fd, abs(float((g_*u).sum())-fd)/abs(fd), float(torch.linalg.cross(xx-xx.mean(0), g_).sum(0).norm())), flush=True)
    v_ = torch.as_tensor(rng.normal(size=X0.shape), dtype=f64, device=dev); e2 = 1e-6*h
    Hu = (W.grad(xx+e2*u)[1]-W.grad(xx-e2*u)[1])/(2*e2); Hv = (W.grad(xx+e2*v_)[1]-W.grad(xx-e2*v_)[1])/(2*e2)
    print('  the Hessian: v.H u %.6e, u.H v %.6e; u.H u / |u|^2 %.3e' % (float((v_*Hu).sum()), float((u*Hv).sum()), float((u*Hu).sum()/(u*u).sum())), flush=True)
mm = np.maximum(V0, .1*np.median(V0))
mass = torch.as_tensor(mm, dtype=f64, device=dev); M3 = mass[:, None]
Om = 2*np.pi/(5*T1)*np.array([.3, .5, .8])/np.linalg.norm([.3, .5, .8])
cm0 = (mm[:, None]*X0).sum(0)/mm.sum()
phi0 = phi-(mm[:, None]*phi).sum(0)/mm.sum()
amp = .5*1.356e-2
v0 = np.cross(Om, X0-cm0)+amp*np.sqrt(w1)*phi0
fg = torch.zeros(n, 3, dtype=f64, device=dev)
CG = TrustCG(W, mass, dt, per=2, product_iters=1)
W.update_ul(W.X); W.tbR.fill_(-1.)
W.grad(W.X)
ST = RingStep(W, CG, mass, W.ul_blocks(), fg, dt, 0.)
ST.pre_ul = True; ST.pre_every = 4; ST.smooth_pred = ST.carry_delta = True; ST.newton_max = 100
ST.set_theta(.5, mass)
ST.set_state(W.X.clone(), torch.as_tensor(v0, dtype=f64, device=dev), fg)
tol = 1e-6*E*h*h
cm = (M3*ST.tx).sum(0)/mass.sum(); vm = (M3*ST.tv).sum(0)/mass.sum()
Lc0 = (M3*torch.linalg.cross(ST.tx-cm, ST.tv-vm)).sum(0)
E0 = .5*float((M3*ST.tv**2).sum())
print('free flight: start energy %.5e (CPU 9.997e-06), tol %.1e; dt T1 / 60; the CPU\'s rest-wall flight with the fit\'s limit: kinetic at 1.5 T1 8.2404e-06, plastic work 1.6390e-06' % (E0, tol), flush=True)
N = int(args.periods*60); t0 = time.time(); nit = npr = 0
for s in range(1, N+1):
    if args.rest_walls:
        W._yld.fill_(False)
    e_, its, g2 = ST.step(tol, lambda g__: .5*tol)
    if math.sqrt(g2) > tol:
        print('  step %d t / T1 %.3f: not converged, |g| %.2e against %.1e after %d Newton iterations; rejected %d, negative %d; |g| %s; CG %s; ratio %s' % (s, s*dt/T1, math.sqrt(g2), tol, its, ST.stats['rejected'], ST.stats['negative'], ['%.2e' % a__[4] for a__ in ST.stats['trace']][:14], [a__[0] for a__ in ST.stats['trace']][:14], ['%.2g' % a__[5] for a__ in ST.stats['trace']][:14]), flush=True)
        if args.stop_bad:
            break
    W.tbR.fill_(-1.)
    ST.end(); nit += its; npr += e_
    if s % max(1, N//18) == 0 or s == N:
        ek = .5*float((M3*ST.tv**2).sum()); ed, ev = W.stored()
        cm = (M3*ST.tx).sum(0)/mass.sum(); vm = (M3*ST.tv).sum(0)/mass.sum()
        Lc = (M3*torch.linalg.cross(ST.tx-cm, ST.tv-vm)).sum(0)
        print('  t / T1 %.2f: kinetic %.4e, stored %.3e, plastic work %.4e; balance / start %.5f; flowing %d, yielded %d; walls %d, springs %d; angular momentum off by %.2e, along it %.5f; |g| %.1e; Newton per step %.1f, products per step %.0f, rejected %d, negative %d; radii %d (fails %d), flips %d; %.1f s' % (s*dt/T1, ek, ed+ev, W.Wp, (ek+ed+ev+W.Wp)/E0, W.ul_stats['yielding'], W.ul_stats['yielded'], W.ul_stats['walls'], W._m_s, float((Lc-Lc0).norm()/Lc0.norm()), float((Lc*Lc0).sum()/(Lc0*Lc0).sum()), math.sqrt(g2), nit/s, npr/s, ST.stats['rejected'], ST.stats['negative'], W.ul_stats['radii_it'], W.ul_stats['radii_fail'], W.ul_stats['flips'], time.time()-t0), flush=True)
print('done', flush=True)
