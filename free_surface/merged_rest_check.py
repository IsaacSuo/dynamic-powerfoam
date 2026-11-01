"""checks of merged_rest.py's kernels on a part, no lion. Rest, rigid motion and uniform strain against their exact energies; the Hessian at rest against tl_check.py's sparse form (no cap term); the gradient against differences on a large deformation, viscous on; the blocks against probed ones; energy over a free oscillation at the midpoint, no viscosity, from a bend; F and r read at the end against the state carried
  python free_surface/merged_rest_check.py --foam <part foam dir>"""
import argparse
import math
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import warp as wp

from free_surface.closure_warp import TrustCG
from free_surface.merged_rest import MergedRest
from free_surface.merged_ul import MergedStep
from free_surface.tl_check import build, kap as kap_tl, mu as mu_tl, stiff_parts

f64, dev = torch.float64, 'cuda'
a_ = argparse.ArgumentParser()
a_.add_argument('--foam', type=Path, required=True)
a_.add_argument('--beta2', type=float, default=3.)
a_.add_argument('--bend', type=float, default=20.)
a_.add_argument('--beta', type=float, default=0.)
args = a_.parse_args()
E, nu = 23.00838, .3008
g = np.load(args.foam/'dem_exact_single'/'geometry.npz'); cl = np.load(args.foam/'dem_exact_single'/'cells.npz')
X0, r0, V0 = g['points'], cl['radii'], g['masses']; n = len(X0); dt = 1/1200


def model(beta=0.):
    W = MergedRest(X0, r0, V0, g['pairs'], g['areas'], g['contacts'], E, nu, beta2=args.beta2)
    W.set_ball((0., 1., 0.), -100., 0.)
    if beta > 0.:
        W.set_viscosity(beta, dt)
    W.rest_cut(); W.set_rest_state(); W.update_ul(W.X); W.prepare()
    return W


W = model()
X = W.X; xc = X.mean(0); h = float(np.median(V0))**(1/3)
print('%d cells, %d walls; smallest height from the centroid %.2e h, not positive on %d' % (n, W.ul_stats['walls'], W.h_stats[0], W.h_stats[1]), flush=True)
Ee, ge = W.grad(X)
print('rest: E %.2e, max |g| %.2e' % (float(Ee), float(ge.norm(dim=1).max())), flush=True)
rg = np.random.default_rng(1)
Rq = torch.linalg.qr(torch.as_tensor(rg.normal(size=(3, 3)), device=dev))[0]; Rq = Rq*torch.sign(torch.linalg.det(Rq))


def at(F_):
    """the state at x = F_ (X - xc) + xc: F_n = F_, L_n = ln det F_, remainders 0; z there, R and C read there"""
    x = (X-xc)@F_.T+xc
    W.set_state_at(x)
    W.update_ul(x)
    W.grad(x); W.reread()
    return (x,)+W.grad(x)


x, Ee, ge = at(Rq)
print('rigid: E %.2e, max |g| %.2e, r max %.2e' % (float(Ee), float(ge.norm(dim=1).max()), float(W.th_['rI'].norm(dim=1).max())), flush=True)
for name, F_ in (('stretch', torch.diag(torch.tensor([1.3, .9, .8], dtype=f64, device=dev))), ('rotated squash', Rq@torch.diag(torch.tensor([1., 1., .5], dtype=f64, device=dev)))):
    x, Ee, ge = at(F_)
    s = torch.linalg.svdvals(F_); dv = s-s.mean()
    ex = float(W.V0.sum())*(W.mu*float((dv*dv).sum())+.5*W.kap*float(torch.log(torch.det(F_)))**2)
    fr = float(W.tsum[0]+W.tsum[3]+W.tsum[1])
    print('%s: E %.6e, exact %.6e, ratio %.8f; remainders\' energy / E %.1e' % (name, fr, ex, fr/ex, float(W.tsum[5])/fr), flush=True)
W._Fe = W.I3.expand(n, 3, 3).clone(); W.tul['Ln'].zero_(); W.update_ul(X); W.grad(X); W.reread()
# the Hessian at rest against the sparse form (same moduli)
D = build(args.foam)
Kr, Kcap = stiff_parts(D)
Psym = np.zeros((9, 9)); I9 = np.eye(3).ravel()
for a in range(3):
    for b in range(3):
        Psym[3*a+b, 3*a+b] += .5; Psym[3*a+b, 3*b+a] += .5
Q = 2*mu_tl*(Psym-np.outer(I9, I9)/3)+kap_tl*np.outer(I9, I9)
Kc = (D['Bg'].T@sp.block_diag([V0[s]*Q for s in range(n)], format='csr')@D['Bg']).tocsr()
Kall = (Kc+args.beta2/3*Kr).tocsr()
gz = W.grad(X)[1].clone()
worst = 0.
for k_ in range(4):
    v = torch.as_tensor(rg.normal(size=(n, 3)), dtype=f64, device=dev)
    e = 1e-6*h/float(v.abs().max())
    Hv = (W.grad(X+e*v)[1]-W.grad(X-e*v)[1])/(2*e)
    Hs = torch.as_tensor(Kall@v.cpu().numpy().ravel(), device=dev).reshape(n, 3)
    worst = max(worst, float((Hv-Hs).norm()/Hs.norm()))
print('Hessian at rest against the sparse form (no cap term): relative difference %.2e' % worst, flush=True)
# the gradient, viscous on, on a large deformation
W = model(.05)
X = W.X
torch.manual_seed(0)


def smooth(y):
    f_ = torch.as_tensor(np.random.default_rng(3).uniform(-1, 1, (6, 3)), device=dev); kk = torch.as_tensor(np.random.default_rng(4).uniform(0, 12, (6, 3)), device=dev)
    return sum(f_[k_][None, :]*torch.sin((y*kk[k_]).sum(1)+k_)[:, None] for k_ in range(6))


W.set_rest_state()
xn = X+.04*smooth(X)
W.set_state_at(xn)
W.update_ul(xn)
z = xn+.002*smooth(X*1.3)
W.grad(z); W.reread()
Ee, gr = W.grad(z)
print('a state: F singular %s; remainders\' energy / shape %.3f; viscous term %.2e' % (np.array2string(torch.linalg.svdvals(W.tFc)[:, [0, 2]].cpu().numpy().max(0), precision=2), float(W.tsum[5])/float(W.tsum[0]+1e-300), float(W.tsum[7])), flush=True)
for k_ in range(3):
    d = torch.randn_like(z); d = d/d.norm()
    h_ = 1e-7
    ep, em = float(W.grad(z+h_*d)[0]), float(W.grad(z-h_*d)[0])
    print('  gradient . d %.8e, differences %.8e, relative %.2e' % (float((gr*d).sum()), (ep-em)/(2*h_), abs((ep-em)/(2*h_)-float((gr*d).sum()))/abs(float((gr*d).sum()))), flush=True)
# the blocks against probed ones, at this state
W.grad(z)
K = W.ul_blocks()
c1 = 1+W.cv
Gz = W.grad(z)[1].clone()
tr_p, tr_t = [], []
for k in np.random.default_rng(0).choice(n, 100, replace=False):
    t_ = 0.
    for c_ in range(3):
        v = torch.zeros(n, 3, dtype=f64, device=dev); v[int(k), c_] = 1.
        e = 1e-6*h
        t_ += float(((W.grad(z+e*v)[1]-W.grad(z-e*v)[1])/(2*e))[int(k), c_])
    tr_p.append(c1*float(K[int(k)].diagonal().sum())); tr_t.append(t_)
ra = np.array(tr_p)/np.array(tr_t)
print('blocks (times 1 + cv): the trace against the probed one, 100 cells: 1%% %.3f, median %.3f, 99%% %.3f, min %.3f, max %.3f' % (np.quantile(ra, .01), np.median(ra), np.quantile(ra, .99), ra.min(), ra.max()), flush=True)
# a free oscillation at the midpoint, no gravity: from a bend at rest
W = model(args.beta)
X = W.X
mass = W.V0; M3 = mass[:, None]
fg = torch.zeros_like(X); wref = float((9.8*M3).norm()); tol = 1e-3*wref
CG = TrustCG(W, mass, dt, per=2, product_iters=1)
ST = MergedStep(W, CG, mass, W.ul_blocks(), fg, dt, 0.)
ST.pre_ul = True; ST.pre_every = 4; ST.smooth_pred = ST.carry_delta = True; ST.newton_max = 300
ST.set_theta(.5, mass)
Xc = X-(M3*X).sum(0)/mass.sum()
ax = int(torch.argmax(Xc.max(0).values-Xc.min(0).values)); o1 = [k_ for k_ in range(3) if k_ != ax][0]; S = float(Xc[:, ax].abs().max())
kp = math.radians(args.bend)/S
x0 = Xc.clone(); th = kp*Xc[:, ax]; rr = 1/kp-Xc[:, o1]
x0[:, ax] = rr*torch.sin(th); x0[:, o1] = 1/kp-rr*torch.cos(th)
x0 = x0+(M3*X).sum(0)/mass.sum()
W.set_state_at(x0)
ST.set_state(x0, torch.zeros_like(x0), fg)
ed, ev = W.stored()
W.update_ul(x0); W.grad(x0); W.reread(); W.grad(x0)
Ei = float(W.tsum[5])
H0 = ed+ev+Ei
print('free oscillation from a %g deg bend: E shape %.4e, volume %.4e, remainders %.4e; F singular min %.3f' % (args.bend, ed, ev, Ei, float(torch.linalg.svdvals(W._Fe)[:, 2].min())), flush=True)
pr, nw = [], []
for s in range(1, 601):
    e_, its, g2 = ST.step(tol, lambda g_: .5*tol)
    if not math.isfinite(g2) or math.sqrt(g2) > tol:
        print('step %d did not converge: |g| / weight %s after %d Newton' % (s, math.sqrt(g2)/wref if math.isfinite(g2) else g2, its), flush=True)
        break
    ST.end(); pr.append(e_); nw.append(its)
    if s % 100 == 0:
        ek = .5*float((M3*ST.tv**2).sum()); ed, ev = W.stored()
        Fs = W._Fe.clone(); rI = W.th_['rIn'].clone()
        W.grad(ST.tx)
        ef = float((W.tFc-Fs).abs().max()); er = float((W.th_['rI']-rI).abs().max())
        J = torch.linalg.det(Fs)
        print('  step %d: KE %.4e shape %.4e vol %.4e remainders %.4e W_v %.4e, energy off %+.4f of the start; F state vs read %.1e, r %.1e; J %.3f..%.3f, F singular %.3f..%.3f; products median %d, Newton median %d' % (s, ek, ed, ev, W.spring_store, W.Wv, (ek+ed+ev+W.spring_store+W.Wv)/H0-1, ef, er, float(J.min()), float(J.max()), float(torch.linalg.svdvals(Fs)[:, 2].min()), float(torch.linalg.svdvals(Fs)[:, 0].max()), np.median(pr[-100:]), np.median(nw[-100:])), flush=True)
print('done', flush=True)
