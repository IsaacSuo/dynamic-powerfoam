"""ring_yield.py, the springs rewritten on the newest walls every step: is an elastic path still elastic? A prescribed motion out and back, no solve, no contact, the limit out of reach (beta = 1). All sites x = X + s u(X), s up to 1 and back to 0, --loops times:
  uniform   u a stretch of log --amp at constant volume
  volume    u a stretch of log --amp at every axis
  wave      u = (amp / k) (sin k y, sin k z, sin k x), strain up to amp, wavelength --wave of the body's size
  mixed     a uniform stretch and a volume out of phase: a loop that encloses an area, s its phase
Read at each return to X, where an elastic body has nothing: F_e - I, the springs' q - d, the force over the peak's. Read over a loop: the internal work against the peak's stored energy, 0 if the path is a potential.
  python free_surface/ring_yield_elastic.py --foam <dir with dem_exact_single> [--mode wave] [--amp .1] [--eps-cut -1] [--steps 40] [--loops 3]"""
import argparse
import math
from pathlib import Path

import numpy as np
import torch

from free_surface.ring_yield import RingYield

f64, dev = torch.float64, 'cuda'
a_ = argparse.ArgumentParser()
a_.add_argument('--foam', type=Path, required=True)
a_.add_argument('--mode', default='wave')
a_.add_argument('--amp', type=float, default=.1)
a_.add_argument('--wave', type=float, default=.5)
a_.add_argument('--eps-cut', type=float, default=-1.)
a_.add_argument('--steps', type=int, default=40)
a_.add_argument('--loops', type=int, default=3)
a_.add_argument('--E', type=float, default=23.00838)
a_.add_argument('--nu', type=float, default=.3008)
a_.add_argument('--beta', type=float, default=0.)
a_.add_argument('--dt', type=float, default=1/1200)
args = a_.parse_args()
g = np.load(args.foam/'dem_exact_single'/'geometry.npz'); cl = np.load(args.foam/'dem_exact_single'/'cells.npz')
X0, r0, V0, C0 = g['points'], cl['radii'], g['masses'], g['contacts']; n = len(X0)
W = RingYield(X0, r0, V0, g['pairs'], g['areas'], C0, args.E, args.nu, 1e6, theta=.5, kappa=5., eps_cut=args.eps_cut, ledger=True)
W.set_ball((0., 1., 0.), -100., 0.)
if args.beta > 0:
    W.set_viscosity(args.beta, args.dt)
W.start()
W.prepare()
X = W.X.clone()
Xc = (W.V0[:, None]*X).sum(0)/W.V0.sum()
size = float((X.max(0)[0]-X.min(0)[0]).max())
cap = W.tul['Mc'].abs().sum((1, 2)) > 0
lo, hi = W.tlo, W.thi
near = cap.clone()
for _ in range(2):
    nx = near.clone()
    nx[lo[near[hi]]] = True; nx[hi[near[lo]]] = True
    near = nx
deep = ~near
print('%d cells, %d deep; size %.3f; mode %s, amp %.3f, eps_cut %g, %d steps a way' % (n, int(deep.sum()), size, args.mode, args.amp, args.eps_cut, args.steps), flush=True)
k_ = 2*math.pi/(args.wave*size)


def motion(s):
    Y = X-Xc
    if args.mode == 'uniform':
        l = math.exp(s*args.amp)
        return Xc+Y*torch.tensor([l, l**-.5, l**-.5], dtype=f64, device=dev)
    if args.mode == 'volume':
        return Xc+Y*math.exp(s*args.amp)
    if args.mode == 'spin':                                                                # (a stretch, s up to .5, then held and turned: a rigid spin of a strained body)
        l = math.exp(min(2*s, 1.)*args.amp)
        a = max(2*s-1, 0.)*math.pi/2
        F = torch.tensor([[l*math.cos(a), -l**-.5*math.sin(a), 0.], [l*math.sin(a), l**-.5*math.cos(a), 0.], [0., 0., l**-.5]], dtype=f64, device=dev)
        return Xc+Y@F.T
    if args.mode == 'mixed':                                                               # (s the phase of a loop that encloses an area in shear and volume)
        ph = 2*math.pi*s
        l, v = math.exp(args.amp*.5*(1-math.cos(ph))), math.exp(args.amp*math.sin(ph))
        return Xc+Y*torch.tensor([v*l, v*l**-.5, v*l**-.5], dtype=f64, device=dev)
    a = s*args.amp/k_
    return X+a*torch.stack([torch.sin(k_*Y[:, 1]), torch.sin(k_*Y[:, 2]), torch.sin(k_*Y[:, 0])], 1)


def step(xn, x):
    W.update_ul(xn)
    z = .5*(xn+x)
    W.grad(z); W.reread(); e0, gz = W.grad(z)
    gz = gz.clone()
    w = float((gz*(x-xn)).sum())
    if args.beta > 0 and not hasattr(W, '_fd'):                                            # (the frozen energy against its force, by differences, once)
        u = torch.as_tensor(np.random.default_rng(1).normal(size=(n, 3)), device=dev)*1e-7
        W._fd = (float(W.grad(z+u)[0])-float(W.grad(z-u)[0]))/2, float((gz*u).sum())
        print('the frozen energy with the viscosity: by differences %.8e, the force %.8e' % W._fd, flush=True)
        W.grad(z)
    W.end_ul(.5)
    return w


def stored(x):
    W.update_ul(x)
    e, b = W.stored(x)
    return e+b


def force(x):
    W.update_ul(x)
    W.grad(x); W.reread()
    return W.grad(x)[1].clone()


path = np.concatenate([np.linspace(0, 1, args.steps+1)[1:], np.linspace(1, 0, args.steps+1)[1:]])
if args.mode in ('mixed', 'spin'):
    path = np.linspace(0, 1, 2*args.steps+1)[1:]
if args.beta > 0 and args.mode == 'uniform':
    print('the viscous work of a loop, closed form 2 mu beta |de|^2 / dt over the steps: %.4e' % (2*W.mu*args.beta*1.5*(args.amp/args.steps)**2/args.dt*2*args.steps*float(W.V0.sum())), flush=True)
if args.mode == 'uniform':
    l_ = math.exp(args.amp)
    print('closed form mu V |U - I|^2: %.4e' % (W.mu*float(W.V0.sum())*((l_-1)**2+2*(l_**-.5-1)**2)), flush=True)
x = motion(0.)
W.update_ul(x)
glead = W._gid == torch.arange(n, device=dev)
for lp in range(args.loops):
    work, peak, fpeak = 0., None, None
    for k, s in enumerate(path):
        xn = x
        x = motion(float(s))
        work += step(xn, x)
        if k == args.steps-1:
            if args.beta > 0:
                print('  the viscous work up to the half of the path %.4e' % W.Wv, flush=True)
            peak = stored(x)
            gp = force(x)
            fpeak = gp.norm(dim=1)
    if args.beta > 0:
        print('  the viscous work at the end of the path %.4e' % W.Wv, flush=True)
    x = motion(0.)
    g0 = force(x)
    Fe = W._Ft
    dF = (Fe-W.I3).flatten(1).norm(dim=1)
    ms = W._m_s
    d = x[W.thi]-x[W.tlo]
    dq =torch.maximum((W._qi[:ms]-d).norm(dim=1), (W._qj[:ms]-d).norm(dim=1))/d.norm(dim=1)
    # (q in a cell's frame; through T a uniform state is untouched)
    Ee = stored(x)
    print('loop %d: at X |F_e - I| deep median %.2e, max %.2e; all median %.2e, max %.2e; |q - d| / |d| median %.2e, 99 %% %.2e; |g| over the peak\'s, deep median %.2e; all %.2e; stored at X / at the peak %.2e; the loop\'s work / the peak %.2e; peak %.4e' % (
        lp, float(dF[deep].median()), float(dF[deep].max()), float(dF.median()), float(dF.max()), float(dq.median()), float(torch.quantile(dq, .99)), float(g0.norm(dim=1)[deep].median()/fpeak[deep].median().clamp(min=1e-300)), float(g0.norm(dim=1).median()/fpeak.median().clamp(min=1e-300)), Ee/peak, work/peak, peak), flush=True)
print('done', flush=True)
