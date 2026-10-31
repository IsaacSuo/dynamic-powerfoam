"""ul_plastic.py without dynamics (2026-10-31). On a foam (--foam, its dem_exact_single):
  1. the gradient against central differences of the energy, at a state with cells and walls both below and above their limits and balls touching
  2. affine increments: |G - H| (gg and fit)
  3. a uniform stress: the forces at the interior sites over |sigma| A (A the cell size squared), gg against fit (fit: 0.6 expected, gg: 0)
  4. a rigid rotation of the whole foam read by the step's rotation: the energy, force and H stay 0
  5. the energy's convexity: second differences along random directions at a yielding state
  6. the return: the stress after end_ul on or inside the yield surface
  python free_surface/ul_plastic_check.py --foam <dir>"""
import argparse
from pathlib import Path

import numpy as np
import torch

from free_surface.ul_plastic import PlasticUL

f64, dev = torch.float64, 'cuda'
a_ = argparse.ArgumentParser()
a_.add_argument('--foam', type=Path, required=True)
a_.add_argument('--E', type=float, default=174.2839)
a_.add_argument('--nu', type=float, default=.3008)
a_.add_argument('--ys', type=float, default=1.5539)
args = a_.parse_args()
g = np.load(args.foam/'dem_exact_single'/'geometry.npz'); cl = np.load(args.foam/'dem_exact_single'/'cells.npz')
X, r, V0 = g['points'], cl['radii'], g['masses']; n = len(X)
h = float(np.median(V0))**(1/3)
print('foam %s: %d cells, h %.2f mm; E %g, nu %g, sigma_y %g per unit density; yield strain %.4f' % (args.foam, n, 1e3*h, args.E, args.nu, args.ys, args.ys/args.E), flush=True)
torch.manual_seed(0)


def model(kind):
    W = PlasticUL(X, r, V0, args.E, args.nu, args.ys, grad=kind)
    W.set_ball((0., 1., 0.), float((W.X[:, 1]-W._rr0).min())+.5*h, 0.)
    W.update_ul(W.X)
    W.prepare()
    return W


W = model('gg')
print('rest: %d walls of capacity %d, cut volume / V0 - 1 max %.2e, radii iterations %d; caps %d' % (W.ul_stats['walls'], W.ul['Kw'], float(((W._V-W.V0)/W.V0).abs().max()), W.ul_stats['radii_it'], int((W.tbR > 0).sum())), flush=True)
X_ = W.X
# 1. the gradient
x1 = X_+.03*h*torch.randn(n, 3, dtype=f64, device=dev)
t = W.tul
t['Ln'].copy_(.02*torch.randn(n, dtype=f64, device=dev))
e_ = .01*torch.randn(n, 3, 3, dtype=f64, device=dev); e_ = .5*(e_+e_.transpose(1, 2)); e_ = e_-(e_.diagonal(dim1=1, dim2=2).sum(1)/3)[:, None, None]*W.I3
t['en'].copy_(e_)
t['wdn'][:W._w[5]] = .02*h*torch.randn(W._w[5], 3, dtype=f64, device=dev)
th = .03; Qr = torch.tensor([[np.cos(th), -np.sin(th), 0.], [np.sin(th), np.cos(th), 0.], [0., 0., 1.]], dtype=f64, device=dev)
W.set_rotation(X_+(X_@(Qr.T-W.I3)))
x1 = x1+(X_@(Qr.T-W.I3))
E0, g0 = W.grad(x1)
print('state for the gradient: E terms dev %.3e vol %.3e var %.3e jump %.3e ball %.3e; cells over the limit %d of %d, walls with beta < 1 %d of %d (beta median %.3f), balls touching %d' % (
    *[float(W.tsum[k]) for k in (0, 1, 2, 3, 6)], int((2*W.mu*W.tenrm > W.sy).sum()), n, int((W.tul['wb'][:W._w[5]] < 1).sum()), W._w[5], float(W.tul['wb'][:W._w[5]].median()), int((W.B['bdelta'].numpy() > 0).sum())), flush=True)
wrong = []
for k in range(6):
    d = torch.randn(n, 3, dtype=f64, device=dev)
    if k >= 3:                                                                      # (a few cells: the local terms' signs)
        d = d*(torch.randperm(n, device=dev)[:, None] < 30).to(f64)
    d = d/d.norm(dim=1).max()
    for eps in (1e-5*h, 1e-6*h):
        Ep, _ = W.grad(x1+eps*d); Em, _ = W.grad(x1-eps*d)
        fd = float(Ep-Em)/(2*eps); an = float((g0*d).sum())
        print('  direction %d, eps %.1e h: analytic %.8e, differences %.8e, relative %.2e' % (k, eps/h, an, fd, abs(an-fd)/max(abs(fd), 1e-300)), flush=True)
    wrong.append(abs(an-fd)/max(abs(fd), 1e-300))
print('check 1: largest relative difference %.2e' % max(wrong), flush=True)
# 5. convexity, same state
neg = 0; lo = np.inf
for k in range(40):
    d = torch.randn(n, 3, dtype=f64, device=dev)
    if k % 2:
        d = d*(torch.randperm(n, device=dev)[:, None] < 20).to(f64)
    d = d/d.norm(dim=1).max()
    eps = 1e-4*h
    Ep, _ = W.grad(x1+eps*d); Em, _ = W.grad(x1-eps*d); Ec, _ = W.grad(x1)
    c2 = float(Ep+Em-2*Ec)/eps**2
    Eg = float(Ec)
    neg += c2 < -1e-9*abs(Eg)/eps**2; lo = min(lo, c2)
print('check 5: second difference along 40 random directions: smallest %.3e, below the energy\'s rounding %d (the ground\'s plane contact is convex too)' % (lo, neg), flush=True)
# 2. affine, 3. uniform stress, 4. rigid
Ha = torch.tensor([[.013, .004, -.002], [.001, -.007, .003], [.005, .002, .004]], dtype=f64, device=dev)
for kind in ('gg', 'fit'):
    W = model(kind)
    t = W.tul
    cap = W.tbR > 0
    W.grad(X_+X_@Ha.T)
    print('check 2 (%s): affine increment: |G - H| median %.2e, max %.2e; tiny cells on the neighbours\' mean %d' % (kind, float((W.tG-Ha).flatten(1).norm(dim=1).median()), float((W.tG-Ha).flatten(1).norm(dim=1).max()), int(t['used'].sum())), flush=True)
    s0 = torch.diag(torch.tensor([.004, -.001, -.003], dtype=f64, device=dev))
    t['en'].copy_(s0.expand(n, 3, 3)); t['Ln'].fill_(.05)
    _, gs = W.grad(X_)
    sig = float((2*W.mu*s0+W.kap*.05*W.I3).norm())
    i_, j_ = W._w[0], W._w[1]
    near = cap.clone(); near[j_[cap[i_]]] = True; near[i_[cap[j_]]] = True
    ring2 = near.clone(); ring2[j_[near[i_]]] = True; ring2[i_[near[j_]]] = True
    Af = V0**(2/3)
    q = gs.norm(dim=1)/(sig*torch.as_tensor(Af, device=dev))
    deep = ~ring2
    r1 = near & ~cap; r2 = ring2 & ~near                                            # (their forces come through the caps' fits: the boundary traction's share)
    q_ = lambda m_: (float(q[m_].median()), float(q[m_].max())) if bool(m_.any()) else (float('nan'), float('nan'))
    print('check 3 (%s): uniform stress |sigma| %.3g: |force| / |sigma| A, median and max: beyond two rings of the caps (%d of %d) %.2e %.2e; second ring (%d) %.2e %.2e; first ring (%d) %.2e %.2e; surface sites (%d) %.2e %.2e' % (
        kind, sig, int(deep.sum()), n, *q_(deep), int(r2.sum()), *q_(r2), int(r1.sum()), *q_(r1), int(cap.sum()), *q_(cap)), flush=True)
    t['en'].zero_(); t['Ln'].zero_()
    th2 = .2; Qr2 = torch.tensor([[np.cos(th2), 0., np.sin(th2)], [0., 1., 0.], [-np.sin(th2), 0., np.cos(th2)]], dtype=f64, device=dev)
    xr = X_@Qr2.T+.01
    t['Q'].copy_(W.I3.expand(n, 3, 3))
    W.tbR.fill_(-1.)
    W.set_rotation(xr)
    E_, gr = W.grad(xr)
    print('check 4 (%s): rigid rotation %.2f rad: |Q - Q_true| max %.2e, E %.3e, |dE/dx| max %.3e (a 1 %% strain\'s force ~ %.3e)' % (kind, th2, float((t['Q']-Qr2).abs().max()), float(E_), float(gr.norm(dim=1).max()), W.mu*.01*float(np.median(V0)**(2/3))), flush=True)
# 6. the return
W = model('gg')
x2 = X_+.6*h*torch.randn(n, 3, dtype=f64, device=dev)*(args.ys/args.E*20)
W.set_rotation(x2)
W.grad(x2)
over0 = int((2*W.mu*W.tenrm > W.sy).sum())
m_ = W._w[5]; o_ = torch.argsort(W._w[3]); dl0, wb0 = W.twdl[:m_].clone(), W.tul['wb'][:m_].clone()
W.end_ul(1.)
nr = W.tul['en'].flatten(1).norm(dim=1)
print('check 6: %d cells over the limit before; after the return |s| / s_y max %.6f; traceless %.1e; symmetric %.1e; walls with beta < 1 %d; memory / (beta |delta_tr|) %.4f..%.4f (<= 1: the mean of two rotations); plastic work %.3e (cells %.3e)' % (
    over0, float((2*W.mu*nr/W.sy).max()), float(W.tul['en'].diagonal(dim1=1, dim2=2).sum(1).abs().max()), float((W.tul['en']-W.tul['en'].transpose(1, 2)).abs().max()), W.ul_stats['wall_yielding'],
    float((W._mem_d.norm(dim=1)/(wb0[o_]*dl0[o_].norm(dim=1))).min()), float((W._mem_d.norm(dim=1)/(wb0[o_]*dl0[o_].norm(dim=1))).max()), W.Wp, W.Wp_cell), flush=True)
# 8. the footprint: two equal discs against the union's exact area, half each; a third, disjoint disc of its own; 9. the glue in the gradient, sticking and sliding
import math
W = model('gg')
yp = float(W.tball[2]); Ra, Da = .02, .03
xs = W.X[:, :0].new_zeros((n, 3)); xs[:, 1] = yp+10.
xs[0] = torch.tensor([0., yp, 0.], dtype=f64); xs[1] = torch.tensor([Da, yp, 0.], dtype=f64); xs[2] = torch.tensor([.2, yp, 0.], dtype=f64)
rs = torch.full((n,), 1e-3, dtype=f64, device=dev); rs[:3] = Ra
lens = 2*Ra*Ra*math.acos(Da/(2*Ra))-.5*Da*math.sqrt(4*Ra*Ra-Da*Da)
ex = (math.pi*Ra*Ra-.5*lens, math.pi*Ra*Ra)
for gs in (h/4, h/16):
    Af = W.footprint(xs, rs, g=gs)
    print('check 8: the footprint, raster %.2f mm: discs 1, 2 of the overlapping pair %.4e, %.4e against %.4e; the third %.4e against %.4e; others %.1e' % (1e3*gs, float(Af[0]), float(Af[1]), ex[0], float(Af[2]), ex[1], float(Af[3:].sum())), flush=True)
W.set_stick()
x9 = X_+.02*h*torch.randn(n, 3, dtype=f64, device=dev)
tb = {k_: __import__('warp').to_torch(W.ball[k_], requires_grad=False) for k_ in ('xn', 'kt', 'bound', 'lamt')}
W.tball[4] = 1.
W.tball[2] = float((x9[:, 1]-W.tbR.clamp(min=0)).min())+.8*h
d9 = W.tbR-(x9[:, 1]-float(W.tball[2]))
tb['xn'].copy_(x9-.01*h*torch.randn(n, 3, dtype=f64, device=dev)); tb['kt'].copy_(8*W.Gs*torch.sqrt((W.tbR*d9).clamp(min=0)))
Af9 = W.footprint(x9, W._ulr)
tb['bound'].copy_(torch.where(W.tbR > 0, W.tau_y*Af9, torch.zeros_like(Af9)))
tb['lamt'].copy_(.3*tb['bound'][:, None]*torch.randn(n, 3, dtype=f64, device=dev)*torch.tensor([1., 0., 1.], dtype=f64, device=dev))
W.tul['wb'][:W._w[5]] = 1.
E0, g9 = W.grad(x9)
on = tb['bound'] > 0
s9 = (x9-tb['xn']); s9[:, 1] = 0.; q9 = tb['lamt']-tb['kt'][:, None]*s9
slide = int(((q9.norm(dim=1) > tb['bound']) & on).sum())
print('state for the glue: %d cells with a limit, %d of them over it (sliding), %d sticking; friction energy %.3e, limits sum %.3e' % (int(on.sum()), slide, int(on.sum())-slide, float(W.tsum[8]), float(tb['bound'].sum())), flush=True)
worst = 0.
for k in range(4):
    d = torch.randn(n, 3, dtype=f64, device=dev)*(on[:, None].to(f64) if k % 2 else 1.)
    d = d/d.norm(dim=1).max(); eps = 1e-6*h
    Ep, _ = W.grad(x9+eps*d); Em, _ = W.grad(x9-eps*d)
    fd = float(Ep-Em)/(2*eps); an = float((g9*d).sum()); worst = max(worst, abs(an-fd)/max(abs(fd), 1e-300))
    print('  direction %d: analytic %.8e, differences %.8e' % (k, an, fd), flush=True)
print('check 9: the glue in the gradient: largest relative difference %.2e' % worst, flush=True)
# 7. an elastic cell's wall keeps its memory: no yield, no motion: the energy is 1/2 k |delta|^2 and the force k delta, in the sum
W = PlasticUL(X, r, V0, args.E, args.nu, float('inf'), grad='gg')
W.update_ul(W.X); W.prepare()
m_ = W._w[5]
W.tul['wdn'][:m_] = .01*h*torch.randn(m_, 3, dtype=f64, device=dev)
W.set_rotation(W.X)
E7, _ = W.grad(W.X)
print('check 7: sigma_y inf: beta min %.6f; jump energy %.6e against 1/2 k |delta|^2 %.6e' % (float(W.tul['wb'][:m_].min()), float(W.tsum[3]), float((.5*W.tul['wk'][:m_]*W.tul['wdn'][:m_].norm(dim=1)**2).sum())), flush=True)
