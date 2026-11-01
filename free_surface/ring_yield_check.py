"""ring_yield.py on a small foam, in order of what rests on what. No limit: the force at a frozen read is RingUL's. Frozen: the force against E by differences, the Hessian's symmetry, below and past the limit. A uniform motion past the limit: the reading, the force on the deep sites. The net force and moment at the fixed point. The free flight with a yield limit: energy + plastic work, angular momentum, the outer iterations.
  python free_surface/ring_yield_check.py --foam <dir with dem_exact_single and fem_modes_h0.0015.npz> [--periods 1.5] [--ey 0.0095] [--kappa 5] [--no-merge]"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

from free_surface.closure_warp import TrustCG
from free_surface.ring_ul import RingUL
from free_surface.ring_yield import RingYield, YieldStep

sys.path.insert(0, '/home/fangsuo/py/dynamic-powerfoam')
from experiments.dem_fem_mode_match import interpolate

f64, dev = torch.float64, 'cuda'
a_ = argparse.ArgumentParser()
a_.add_argument('--foam', type=Path, required=True)
a_.add_argument('--periods', type=float, default=1.5)
a_.add_argument('--ey', type=float, default=.0095)
a_.add_argument('--kappa', type=float, default=5.)
a_.add_argument('--eps-cut', type=float, default=0.)
a_.add_argument('--no-merge', action='store_true')
a_.add_argument('--stop-bad', action='store_true')
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
agg = 0. if args.no_merge else .1


def model(sig=sigma_y, agg_=agg):
    W = RingYield(X0, r0, V0, g['pairs'], g['areas'], g['contacts'], E, nu, sig, theta=.5, kappa=args.kappa, agg_merge=agg_, eps_cut=args.eps_cut)
    W.set_ball((0., 0., 1.), -10., 0.)
    W.start()
    W.tbR.fill_(-1.)
    W.prepare()
    return W


Xt = torch.as_tensor(X0, dtype=f64, device=dev)
ph = torch.as_tensor(phi, dtype=f64, device=dev)
W0 = model(1e30, 0.)
R0 = RingUL(X0, r0, V0, g['pairs'], g['areas'], g['contacts'], E, nu, theta=.5)
R0.set_ball((0., 0., 1.), -10., 0.); R0.rest_cut(); R0.tbR.fill_(-1.); R0.prepare()
th_ = .7; Qz = torch.tensor([[np.cos(th_), -np.sin(th_), 0.], [np.sin(th_), np.cos(th_), 0.], [0., 0., 1.]], dtype=f64, device=dev)
for name, am in (('small', .002), ('large', .05)):
    xx = (Xt+am*ph+am*.05*torch.as_tensor(np.random.default_rng(1).normal(size=X0.shape), dtype=f64, device=dev))@Qz.T
    W0.grad(xx); W0.reread()
    e0, g0_ = W0.grad(xx); e1, g1_ = R0.grad(xx)
    print('no limit, no merging, read at x, against RingUL, %s motion: E %.8e, %.8e, gap %.1e; |g difference| / |g| %.1e' % (name, float(e0), float(e1), abs(float(e0-e1))/abs(float(e1)), float((g0_-g1_).norm()/g1_.norm())), flush=True)
del W0, R0
W = model()
print('%s, %d cells, %d in groups; springs %d; h %.2f mm; T1 %.4f s; yield strain %.2f %%, kappa %g' % (args.foam.name, n, int((W._gid != torch.arange(n, device=dev)).sum()), W._m_s, 1e3*h, T1, 100*W.ey, W.kappa), flush=True)
L0, g0 = W.grad(W.X)
print('at rest: E %.2e, |g| %.2e' % (float(L0), float(g0.abs().max())), flush=True)
rng = np.random.default_rng(0)
u = torch.as_tensor(rng.normal(size=X0.shape), dtype=f64, device=dev); e = 1e-6*h
for name, amp_ in (('below and past the limit', .002), ('far past it', .03)):
    xx = W.X+amp_*ph
    W.grad(xx); W.reread()
    L_, g_ = W.grad(xx)
    fd = (float(W.grad(xx+e*u)[0])-float(W.grad(xx-e*u)[0]))/(2*e)
    Y = W.ty
    print('frozen, %s (%d cells past the limit, beta min %.3f, beta_r min %.3f, reading / e_y max %.1f): the force %.8e, E by differences %.8e, gap %.2e; net force %.1e, net moment %.1e' % (name, int((Y['be'] < 1).sum()), float(Y['be'].min()), float(Y['br'].min()), float(Y['ei'].max())/W.ey, float((g_*u).sum()), fd, abs(float((g_*u).sum())-fd)/abs(fd), float(g_.sum(0).norm()), float(torch.linalg.cross(xx-xx.mean(0), g_).sum(0).norm())), flush=True)
    v_ = torch.as_tensor(rng.normal(size=X0.shape), dtype=f64, device=dev); e2 = 1e-6*h
    Hu = (W.grad(xx+e2*u)[1]-W.grad(xx-e2*u)[1])/(2*e2); Hv = (W.grad(xx+e2*v_)[1]-W.grad(xx-e2*v_)[1])/(2*e2)
    print('  the Hessian: v.H u %.6e, u.H v %.6e; u.H u / |u|^2 %.3e (positive: a convex frozen problem)' % (float((v_*Hu).sum()), float((u*Hv).sum()), float((u*Hu).sum()/(u*u).sum())), flush=True)
Wc = model()
Wc._yld[:] = True; Wc.tr['flag'][:] = 1; Wc._eps[:] = 1e-3
Wc._Ft = Wc.I3.expand(n, 3, 3).clone()
xc = Wc.X+.01*ph
Wc.update_ul(xc)
for name, amp_ in (('below and past the limit', .002), ('far past it', .03)):
    xx = xc+amp_*ph
    Wc.grad(xx); Wc.reread()
    L_, g_ = Wc.grad(xx)
    fd = (float(Wc.grad(xx+e*u)[0])-float(Wc.grad(xx-e*u)[0]))/(2*e)
    Y = Wc.ty
    print('on the cut, frozen, %s (%d past the limit, beta min %.3f; 1 + v %.4f..%.4f; fit gap at the step\'s start %.1e): the force %.8e, E by differences %.8e, gap %.2e; net force %.1e, net moment %.1e' % (name, int((Y['be'] < 1).sum()), float(Y['be'].min()), float(Y['vm'][Wc._gid == torch.arange(n, device=dev)].min()), float(Y['vm'].max()), float((Y['Fn'][Wc._gid == torch.arange(n, device=dev)]-Wc._Ft[Wc._gid == torch.arange(n, device=dev)]).abs().max()), float((g_*u).sum()), fd, abs(float((g_*u).sum())-fd)/abs(fd), float(g_.sum(0).norm()), float(torch.linalg.cross(xx-xx.mean(0), g_).sum(0).norm())), flush=True)
    v_ = torch.as_tensor(rng.normal(size=X0.shape), dtype=f64, device=dev); e2 = 1e-6*h
    Hu = (Wc.grad(xx+e2*u)[1]-Wc.grad(xx-e2*u)[1])/(2*e2); Hv = (Wc.grad(xx+e2*v_)[1]-Wc.grad(xx-e2*v_)[1])/(2*e2)
    print('  the Hessian: v.H u %.6e, u.H v %.6e; u.H u / |u|^2 %.3e' % (float((v_*Hu).sum()), float((u*Hv).sum()), float((u*Hu).sum()/(u*u).sum())), flush=True)
del Wc
closed = W.tul['Mc'].abs().sum((1, 2)) == 0
Amed = float(np.median(g['areas']))
i_, j_ = W.tlo, W.thi
depth = torch.where(closed, torch.full((n,), -1, device=dev), torch.zeros(n, dtype=torch.long, device=dev))
for k in range(1, 8):
    fr = depth == k-1
    if not bool(fr.any()):
        break
    nb = torch.zeros(n, dtype=torch.bool, device=dev); nb[j_[fr[i_]]] = True; nb[i_[fr[j_]]] = True
    depth[nb & (depth < 0)] = k
Xc = W.X-W.X.mean(0)
for lab, gam in (('below the limit', .5*W.ey*2**.5), ('at 2 limits', 2*W.ey*2**.5), ('at 8 limits', 8*W.ey*2**.5)):
    Sh = torch.tensor([[1., gam, 0.], [0., 1., 0.], [0., 0., 1.]], dtype=f64, device=dev)
    xx = Xc@Sh.T+W.X.mean(0)
    W.grad(xx); W.reread()
    _, g_ = W.grad(xx)
    Y = W.ty
    lead = W._gid == torch.arange(n, device=dev)
    print('uniform shear %s: reading / e_y on the closed cells 1 %% %.3f, median %.3f, 99 %% %.3f; beta median %.4f; |g| / (s_y A median) at depth 0, 1, 2+: %s' % (lab, float(torch.quantile(Y['ei'][lead & closed], .01))/W.ey, float(Y['ei'][lead & closed].median())/W.ey, float(torch.quantile(Y['ei'][lead & closed], .99))/W.ey, float(Y['be'][lead].median()), ['%.2e' % float(g_[(depth == k) if k < 2 else (depth >= 2)].norm(dim=1).max()/(sy*Amed)) for k in range(3)]), flush=True)
del W
# the free flight
mm = np.maximum(V0, .1*np.median(V0))
mass = torch.as_tensor(mm, dtype=f64, device=dev); M3 = mass[:, None]
Om = 2*np.pi/(5*T1)*np.array([.3, .5, .8])/np.linalg.norm([.3, .5, .8])
cm0 = (mm[:, None]*X0).sum(0)/mm.sum()
phi0 = phi-(mm[:, None]*phi).sum(0)/mm.sum()
amp = .5*1.356e-2
v0 = np.cross(Om, X0-cm0)+amp*np.sqrt(w1)*phi0
fg = torch.zeros(n, 3, dtype=f64, device=dev)
W = model()
CG = TrustCG(W, mass, dt, per=2, product_iters=1)
W.update_ul(W.X); W.tbR.fill_(-1.)
W.grad(W.X); W.reread(); W.grad(W.X)
ST = YieldStep(W, CG, mass, W.ul_blocks(), fg, dt, 0.)
ST.pre_ul = True; ST.pre_every = 4; ST.smooth_pred = ST.carry_delta = True
ST.set_theta(.5, mass)
ST.set_state(W.X.clone(), torch.as_tensor(v0, dtype=f64, device=dev), fg)
tol = 1e-6*E*h*h
cm = (M3*ST.tx).sum(0)/mass.sum(); vm = (M3*ST.tv).sum(0)/mass.sum()
Lc0 = (M3*torch.linalg.cross(ST.tx-cm, ST.tv-vm)).sum(0)
E0 = .5*float((M3*ST.tv**2).sum())
print('free flight: start energy %.5e, tol %.1e; dt T1 / 60; ring_flow.py on the same flight: kinetic at 1.5 T1 8.35e-06, plastic work 1.54e-06' % (E0, tol), flush=True)
N = int(args.periods*60); t0 = time.time(); nit = npr = nou = 0
Epr, Wpr, Wcr, Whr = E0, 0., 0., 0.
for s in range(1, N+1):
    e_, its, g2 = ST.step(tol, lambda g__: .5*tol)
    nou += ST.stats['outer']
    if math.sqrt(g2) > tol:
        print('  step %d t / T1 %.3f: not converged, |g| %.2e against %.1e, %d outer, %d inner; rejected %d, negative %d; trace |g| %s' % (s, s*dt/T1, math.sqrt(g2), tol, ST.stats['outer'], its, ST.stats['rejected'], ST.stats['negative'], ['%.2e' % a__[4] for a__ in ST.stats['trace']][:14]), flush=True)
        if args.stop_bad:
            break
    W.tbR.fill_(-1.)
    ST.end(); nit += its; npr += e_
    if s <= 12:
        ek_ = .5*float((M3*ST.tv**2).sum()); a1_, a2_ = W.stored(); En_ = ek_+a1_+a2_
        lead_ = W._gid == torch.arange(n, device=dev)
        Y_ = W.ty
        be_ = Y_['be'][lead_]
        print('  step %d: energy fell %.3e; plastic work cells %.3e, remainder %.3e; ratio %.3f; past the limit %d, beta min %.3f, median of those %.3f, beta_r min %.3f; reading / e_y max %.1f; outer %d' % (s, Epr-En_, W.Wp_cell-Wcr, W.Wp_half-Whr, (W.Wp-Wpr)/max(Epr-En_, 1e-300), int((be_ < 1).sum()), float(be_.min()), float(be_[be_ < 1].median()) if bool((be_ < 1).any()) else 1., float(Y_['br'][lead_].min()), float(Y_['ei'][lead_].max())/W.ey, ST.stats['outer']), flush=True)
        Epr, Wpr, Wcr, Whr = En_, W.Wp, W.Wp_cell, W.Wp_half
    if s % max(1, N//12) == 0 or s == N:
        ek = .5*float((M3*ST.tv**2).sum()); ed, ev = W.stored()
        cm = (M3*ST.tx).sum(0)/mass.sum(); vm = (M3*ST.tv).sum(0)/mass.sum()
        Lc = (M3*torch.linalg.cross(ST.tx-cm, ST.tv-vm)).sum(0)
        print('  t / T1 %.2f: kinetic %.4e, stored %.3e, plastic work %.4e (remainder %.2e); balance / start %.5f; past the limit %d, yielded %d; angular momentum off by %.2e, along it %.5f; |g| %.1e; per step outer %.1f, inner %.1f, products %.0f; rejected %d, negative %d; %.1f s' % (s*dt/T1, ek, ed+ev, W.Wp, W.Wp_half, (ek+ed+ev+W.Wp)/E0, W.ul_stats['yielding'], W.ul_stats['yielded'], float((Lc-Lc0).norm()/Lc0.norm()), float((Lc*Lc0).sum()/(Lc0*Lc0).sum()), math.sqrt(g2), nou/s, nit/s, npr/s, ST.stats['rejected'], ST.stats['negative'], time.time()-t0), flush=True)
print('done', flush=True)
