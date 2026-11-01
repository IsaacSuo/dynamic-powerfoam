"""RingPlastic on a small foam, in a minute: at no strain nothing reads; a uniform motion of the sites: the stress held at the limit, the read against the limit; the free flight of ring_ul_check.py with the cut each step and a yield limit: energy + plastic work, angular momentum, Newton.
  python free_surface/ring_plastic_check.py --foam <dir with dem_exact_single and fem_modes_h0.0015.npz> [--periods 1.5] [--ey 0.0095]"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import warp as wp

from free_surface.closure_warp import TrustCG
from free_surface.ring_ul import RingPlastic, RingStep

sys.path.insert(0, '/home/fangsuo/py/dynamic-powerfoam')
from experiments.dem_fem_mode_match import interpolate

f64, dev = torch.float64, 'cuda'
a_ = argparse.ArgumentParser()
a_.add_argument('--foam', type=Path, required=True)
a_.add_argument('--periods', type=float, default=1.5)
a_.add_argument('--ey', type=float, default=.0095)
a_.add_argument('--rest-walls', action='store_true', help='keep every spring at its rest wall: no spring list changes, as ring_checks.py --test 12, for comparison')
a_.add_argument('--stop-bad', action='store_true', help='stop at the first step whose solve does not converge')
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
W = RingPlastic(X0, r0, V0, g['pairs'], g['areas'], g['contacts'], E, nu, sigma_y, theta=.5)
W.set_ball((0., 0., 1.), -10., 0.)
W.start()
W.tbR.fill_(-1.)
W.prepare()
print('%s, %d cells; springs %d; h %.2f mm; T1 %.4f s; yield strain %.2f %%, kappa %.1f' % (args.foam.name, n, W._m_s, 1e3*h, T1, 100*W.ey, W.kappa), flush=True)
L0, g0 = W.grad(W.X)
print('at rest: E %.2e, |g| %.2e; read %.2e, beta min %.3f' % (float(L0), float(g0.abs().max()), float(W.tr['rho'].max()), float(W.tbe.min())), flush=True)
# a uniform shear of the sites at z, from the rest state: the read and the return, no solve
for gam in (.5*W.ey*2**.5, 2*W.ey*2**.5, .05):
    Sh = torch.tensor([[1., gam, 0.], [0., 1., 0.], [0., 0., 1.]], dtype=f64, device=dev)
    Xc = W.X-W.X.mean(0)
    _, g_ = W.grad(Xc@Sh.T+W.X.mean(0))
    closed = W.tul['Mc'].abs().sum((1, 2)) == 0
    H = W.tr['Rc'].transpose(1, 2)@W.tFc
    print('uniform shear %.4f: rho / e_y median %.4f, max %.4f; beta %.4f..%.4f; springs after the return, |dev U_e| %.4f; E after %.3e, plastic potential %.3e; |g| on the closed cells / (s_y A median) %.1e' % (gam, float(W.tr['rho'].median())/W.ey, float(W.tr['rho'].max())/W.ey, float(W.tbe.min()), float(W.tbe.max()), float((W.tr['Ur']*W.tbe[:, None, None]-W.I3).abs().median()), float(W.tsum[0]+W.tsum[5]), float(W.tsum[3]), float(g_[closed].norm(dim=1).max())/(sy*float(torch.as_tensor(g['areas']).median()))), flush=True)
L_, g_ = W.grad(W.X+.002*torch.as_tensor(phi, dtype=f64, device=dev))
x0 = W.X+.03*torch.as_tensor(phi, dtype=f64, device=dev); rng = np.random.default_rng(0); u = torch.as_tensor(rng.normal(size=X0.shape), dtype=f64, device=dev); e = 1e-6*h
for name, xx in (('elastic', W.X+.002*torch.as_tensor(phi, dtype=f64, device=dev)), ('flowing', x0)):
    L_, g_ = W.grad(xx)
    bm = float(W.tbe.min()); fl = int((W.tbe < 1).sum())
    fd = (float(W.grad(xx+e*u)[0])-float(W.grad(xx-e*u)[0]))/(2*e)
    print('the plastic potential\'s gradient, %s (%d cells flowing, beta min %.3f): force along a random direction %.8e, E + W_p by differences %.8e, gap %.2e; net moment %.1e' % (name, fl, bm, float((g_*u).sum()), fd, abs(float((g_*u).sum())-fd)/abs(fd), float(torch.linalg.cross(xx-xx.mean(0), g_).sum(0).norm())), flush=True)
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
print('free flight: start energy %.5e (CPU 9.997e-06), tol %.1e; dt T1 / 60; the CPU\'s rest-wall flight: kinetic at 1.5 T1 8.2404e-06, plastic work 1.6390e-06' % (E0, tol), flush=True)
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
    if s in (6, 12, 24):
        # the potential's gradient where the flow really is: a step's move from the state just reached, returned from it
        W.update_ul(ST.tx); W.tbR.fill_(-1.)
        W.tul['Q'].copy_(W.I3.expand(n, 3, 3))
        xx = ST.tx+.5*dt*ST.tv
        for name, u_ in (('random', torch.as_tensor(rng.normal(size=X0.shape), dtype=f64, device=dev)), ('the velocity', ST.tv)):
            L_, g_ = W.grad(xx)
            fl = int((W.tbe < 1).sum()); bm = float(W.tbe.min())
            e_fd = 1e-5*h/float(u_.abs().max())
            fd = (float(W.grad(xx+e_fd*u_)[0])-float(W.grad(xx-e_fd*u_)[0]))/(2*e_fd)
            gu = float((g_*u_).sum())
            print('  step %d, half a step on (%d cells flowing, beta min %.3f, median of the flowing %.3f): along %s: force %.6e, E + W_p by differences %.6e, gap %.2e' % (s, fl, bm, float(W.tbe[W.tbe < 1].median()) if fl else 1., name, gu, fd, abs(gu-fd)/abs(fd)), flush=True)
    if s % max(1, N//18) == 0 or s == N:
        ek = .5*float((M3*ST.tv**2).sum()); ed, ev = W.stored()
        cm = (M3*ST.tx).sum(0)/mass.sum(); vm = (M3*ST.tv).sum(0)/mass.sum()
        Lc = (M3*torch.linalg.cross(ST.tx-cm, ST.tv-vm)).sum(0)
        print('  t / T1 %.2f: kinetic %.4e, stored %.3e, plastic work %.4e; balance / start %.5f; flowing %d, yielded %d; walls %d, springs %d; angular momentum off by %.2e, along it %.5f; |g| %.1e; Newton per step %.1f, products per step %.0f, rejected %d; radii %d (fails %d), flips %d; %.1f s' % (s*dt/T1, ek, ed+ev, W.Wp, (ek+ed+ev+W.Wp)/E0, W.ul_stats['yielding'], W.ul_stats['yielded'], W.ul_stats['walls'], W._m_s, float((Lc-Lc0).norm()/Lc0.norm()), float((Lc*Lc0).sum()/(Lc0*Lc0).sum()), math.sqrt(g2), nit/s, npr/s, ST.stats['rejected'], W.ul_stats['radii_it'], W.ul_stats['radii_fail'], W.ul_stats['flips'], time.time()-t0), flush=True)
print('done', flush=True)
