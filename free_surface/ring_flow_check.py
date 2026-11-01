"""ring_flow.py on a small foam, in order of what rests on what: at rest nothing reads; a uniform motion of the sites, all cells elastic and all flowing, against the limit: the force on the deep sites; the force against E by differences on a mix of the two forms, below and past the limit, and the Hessian's symmetry; the free flight with a yield limit: energy + plastic work, angular momentum, Newton.
  python free_surface/ring_flow_check.py --foam <dir with dem_exact_single and fem_modes_h0.0015.npz> [--periods 1.5] [--ey 0.0095] [--kappa 5]"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import warp as wp

from free_surface.closure_warp import TrustCG
from free_surface.ring_flow import FlowStep, RingFlow
from free_surface.ring_ul import RingUL

sys.path.insert(0, '/home/fangsuo/py/dynamic-powerfoam')
from experiments.dem_fem_mode_match import interpolate

f64, dev = torch.float64, 'cuda'
a_ = argparse.ArgumentParser()
a_.add_argument('--foam', type=Path, required=True)
a_.add_argument('--periods', type=float, default=1.5)
a_.add_argument('--ey', type=float, default=.0095)
a_.add_argument('--kappa', type=float, default=5.)
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


def model():
    W = RingFlow(X0, r0, V0, g['pairs'], g['areas'], g['contacts'], E, nu, sigma_y, theta=.5, kappa=args.kappa)
    W.set_ball((0., 0., 1.), -10., 0.)
    W.start()
    W.tbR.fill_(-1.)
    W.prepare()
    return W


# nothing flagged: RingUL itself
W0 = RingFlow(X0, r0, V0, g['pairs'], g['areas'], g['contacts'], E, nu, 1e30, theta=.5)
W0.set_ball((0., 0., 1.), -10., 0.); W0.start(); W0.tbR.fill_(-1.); W0.prepare()
R0 = RingUL(X0, r0, V0, g['pairs'], g['areas'], g['contacts'], E, nu, theta=.5)
R0.set_ball((0., 0., 1.), -10., 0.); R0.rest_cut(); R0.tbR.fill_(-1.); R0.prepare()
Xt = torch.as_tensor(X0, dtype=f64, device=dev)
th_ = .7; Qz = torch.tensor([[np.cos(th_), -np.sin(th_), 0.], [np.sin(th_), np.cos(th_), 0.], [0., 0., 1.]], dtype=f64, device=dev)
for name, am in (('small', .002), ('large', .05)):
    xx = (Xt+am*torch.as_tensor(phi, dtype=f64, device=dev)+am*.05*torch.as_tensor(np.random.default_rng(1).normal(size=X0.shape), dtype=f64, device=dev))@Qz.T
    e0, g0_ = W0.grad(xx); e1, g1_ = R0.grad(xx)
    print('no limit against RingUL, %s motion: E %.8e, %.8e, gap %.1e; |g difference| / |g| %.1e' % (name, float(e0), float(e1), abs(float(e0-e1))/abs(float(e1)), float((g0_-g1_).norm()/g1_.norm())), flush=True)
del W0, R0
W = model()
print('%s, %d cells; springs %d; h %.2f mm; T1 %.4f s; yield strain %.2f %%, kappa %g' % (args.foam.name, n, W._m_s, 1e3*h, T1, 100*W.ey, W.kappa), flush=True)
L0, g0 = W.grad(W.X)
print('at rest, all elastic: E %.2e, |g| %.2e; reading %.2e' % (float(L0), float(g0.abs().max()), float(W.tei.max())), flush=True)
W.flag_at = 1e30
closed = W.tul['Mc'].abs().sum((1, 2)) == 0
# depth from the cap cells, by the walls
i_, j_ = W.tlo, W.thi
depth = torch.where(closed, torch.full((n,), -1, device=dev), torch.zeros(n, dtype=torch.long, device=dev))
for k in range(1, 8):
    fr = depth == k-1
    if not bool(fr.any()):
        break
    nb = torch.zeros(n, dtype=torch.bool, device=dev); nb[j_[fr[i_]]] = True; nb[i_[fr[j_]]] = True
    depth[nb & (depth < 0)] = k
print('depth from the cap cells: %s' % [int((depth == k).sum()) for k in range(6)], flush=True)
Amed = float(np.median(g['areas']))
Xc = W.X-W.X.mean(0)
for lab, gam in (('below the limit', .5*W.ey*2**.5), ('past it', 4*W.ey*2**.5)):
    Sh = torch.tensor([[1., gam, 0.], [0., 1., 0.], [0., 0., 1.]], dtype=f64, device=dev)
    xx = Xc@Sh.T+W.X.mean(0)
    _, g_ = W.grad(xx)
    print('uniform shear %s, %.4f, all elastic: reading / e_y median %.3f, range %.3f..%.3f; |g| / (s_y A median) at depth 0, 1, 2+: %s' % (lab, gam, float(W.tei.median())/W.ey, float(W.tei.min())/W.ey, float(W.tei.max())/W.ey, ['%.2e' % float(g_[(depth == k) if k < 2 else (depth >= 2)].norm(dim=1).max()/(sy*Amed)) for k in range(3)]), flush=True)
W.flag_new(torch.ones(n, dtype=torch.bool, device=dev))
for lab, gam in (('below the limit', .5*W.ey*2**.5), ('past it', 4*W.ey*2**.5)):
    Sh = torch.tensor([[1., gam, 0.], [0., 1., 0.], [0., 0., 1.]], dtype=f64, device=dev)
    xx = Xc@Sh.T+W.X.mean(0)
    W.grad(xx); W.reread(beta=True)
    _, g_ = W.grad(xx)
    print('uniform shear %s, all flowing: beta %.4f..%.4f; |g| / (s_y A median) at depth 0, 1, 2+: %s' % (lab, float(W.tbe.min()), float(W.tbe.max()), ['%.2e' % float(g_[(depth == k) if k < 2 else (depth >= 2)].norm(dim=1).max()/(sy*Amed)) for k in range(3)]), flush=True)
del W
W = model()
W.flag_at = 1e30
rng = np.random.default_rng(0)
W.grad(W.X)
W.flag_new(torch.as_tensor(rng.random(n) < .3, device=dev))
print('the mix: %d of %d cells flowing, springs with both ends elastic %d, one %d, none %d' % (int(W._yld.sum()), n, int((~W._yld[W.tlo] & ~W._yld[W.thi]).sum()), int((W._yld[W.tlo] ^ W._yld[W.thi]).sum()), int((W._yld[W.tlo] & W._yld[W.thi]).sum())), flush=True)
u = torch.as_tensor(rng.normal(size=X0.shape), dtype=f64, device=dev); e = 1e-6*h
for name, amp_ in (('below and past the limit', .002), ('far past it', .03)):
    xx = W.X+amp_*torch.as_tensor(phi, dtype=f64, device=dev)
    W.grad(xx); W.reread(beta=True)
    L_, g_ = W.grad(xx)
    fl = int(((W.tbe < 1) & W._yld).sum())
    fd = (float(W.grad(xx+e*u)[0])-float(W.grad(xx-e*u)[0]))/(2*e)
    print('the force against E by differences, %s (%d flowing cells past the limit, beta min %.3f, reading / e_y max %.1f): %.8e, %.8e, gap %.2e; net force %.1e, net moment %.1e' % (name, fl, float(W.tbe.min()), float(W.tei.max())/W.ey, float((g_*u).sum()), fd, abs(float((g_*u).sum())-fd)/abs(fd), float(g_.sum(0).norm()), float(torch.linalg.cross(xx-xx.mean(0), g_).sum(0).norm())), flush=True)
    v_ = torch.as_tensor(rng.normal(size=X0.shape), dtype=f64, device=dev); e2 = 1e-6*h
    Hu = (W.grad(xx+e2*u)[1]-W.grad(xx-e2*u)[1])/(2*e2); Hv = (W.grad(xx+e2*v_)[1]-W.grad(xx-e2*v_)[1])/(2*e2)
    print('  the Hessian: v.H u %.6e, u.H v %.6e; u.H u / |u|^2 %.3e' % (float((v_*Hu).sum()), float((u*Hv).sum()), float((u*Hu).sum()/(u*u).sum())), flush=True)
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
W.grad(W.X)
ST = FlowStep(W, CG, mass, W.ul_blocks(), fg, dt, 0.)
ST.pre_ul = True; ST.pre_every = 4; ST.smooth_pred = ST.carry_delta = True; ST.newton_max = 100
ST.set_theta(.5, mass)
ST.set_state(W.X.clone(), torch.as_tensor(v0, dtype=f64, device=dev), fg)
tol = 1e-6*E*h*h
cm = (M3*ST.tx).sum(0)/mass.sum(); vm = (M3*ST.tv).sum(0)/mass.sum()
Lc0 = (M3*torch.linalg.cross(ST.tx-cm, ST.tv-vm)).sum(0)
E0 = .5*float((M3*ST.tv**2).sum())
print('free flight: start energy %.5e, tol %.1e; dt T1 / 60' % (E0, tol), flush=True)
N = int(args.periods*60); t0 = time.time(); nit = npr = redo = cells = 0
for s in range(1, N+1):
    e_, its, g2 = ST.step(tol, lambda g__: .5*tol)
    redo += ST.stats['redo']; cells += ST.stats['redo_cells']
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
        print('  t / T1 %.2f: kinetic %.4e, stored %.3e, plastic work %.4e (halves %.2e); balance / start %.5f; flagged %d, past the limit %d; walls %d, springs %d; angular momentum off by %.2e, along it %.5f; |g| %.1e; Newton per step %.1f, products per step %.0f, repeats %d (%d cells), rejected %d, negative %d; radii %d (fails %d), flips %d; %.1f s' % (s*dt/T1, ek, ed+ev, W.Wp, W.Wp_half, (ek+ed+ev+W.Wp)/E0, int(W._yld.sum()), W.ul_stats['yielding'], W.ul_stats['walls'], W._m_s, float((Lc-Lc0).norm()/Lc0.norm()), float((Lc*Lc0).sum()/(Lc0*Lc0).sum()), math.sqrt(g2), nit/s, npr/s, redo, cells, ST.stats['rejected'], ST.stats['negative'], W.ul_stats['radii_it'], W.ul_stats['radii_fail'], W.ul_stats['flips'], time.time()-t0), flush=True)
print('done', flush=True)
