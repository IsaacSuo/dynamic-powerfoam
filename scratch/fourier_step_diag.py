"""why a forward's step does not converge (11-07: fourier_elastic0_30k, step 352, t 0.293 s, |g| 5.2 weights after 50
Newton iterations, 38 after 200): the same forward (lion_inverse.forward) with a hook keeping W and the step object; at
the failure: the step's Newton trace (per iteration CG products, trust-region edge, negative curvature, |g| / weight,
ratio, accepted), then at the last accepted iterate z: the residual per site (MidpointStep's gz), the sites carrying
it with their cell's det F_B, singular values, the ground ball's state and how far the site moved within the step
(z - x_n), and the cells over the whole body that are inverted or near the polar rotation's singular line (s2 ~ s3).
Saves the state before the step (x, v, a) to out_inverse/fourier_forward/<scene>/diag_step<s-1>.pt.
  python -u scratch/fourier_step_diag.py <scene> [K]"""
import math
import sys
from pathlib import Path

import numpy as np
import torch
import warp as wp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import free_surface.lion_inverse as li

scene = sys.argv[1]; li.set_scene(scene)
K = int(sys.argv[2]) if len(sys.argv) > 2 else li.NFRAMES-1
th = np.array([math.log(li.GT['E']), li.GT['nu'], math.log(li.BETA_MIN)])
OUT = ROOT/'out_inverse'/'fourier_forward'/scene; OUT.mkdir(parents=True, exist_ok=True)
keep = {}


def hook(s, W, ST, cbody, Rb):
    keep.update(W=W, ST=ST, cbody=cbody, Rb=Rb, s=s, x=ST.tx.clone(), v=ST.tv.clone(), a=ST.ta.clone())


try:
    li.forward(th, K, 1e-3, save_frames=False, verbose=True, on_step=hook)
    print('the run finished: nothing to diagnose'); sys.exit(0)
except RuntimeError as e:
    print('FAILED as expected: %s' % e, flush=True)
W, ST, cbody, s0 = keep['W'], keep['ST'], keep['cbody'], keep['s']
torch.save(dict(x=keep['x'], v=keep['v'], a=keep['a'], step=s0), str(OUT/('diag_step%d.pt' % s0)))
n = W.n; dev = 'cuda'
mass = torch.as_tensor(W.cmass['rowsum'], dtype=torch.float64, device=dev) if W.cmass is not None else None
ref = 9.8*float(wp.to_torch(ST.a['m'])[:n].norm())                                 # (the failure message's scale: |lumped weight vector|)
tr = ST.stats['trace']
print('\n== the failed step %d: %d Newton iterations, rejected %d, trust-region edge %d, negative curvature %d' % (
    s0+1, len(tr), ST.stats['rejected'], ST.stats['edge'], ST.stats['negative']))
print('   iteration: CG products, edge, negative, |g| / weight, ratio, accepted')
rows = list(enumerate(tr))
for i, (k_, e_, ng_, stop, g_, rho, ok) in (rows[:12]+[(None, None)]+rows[-12:] if len(rows) > 24 else rows):
    if i is None:
        print('   ...'); continue
    print('   %3d: %4d %d %d  %.3e  %+.3f  %s' % (i+1, k_, e_, ng_, g_/ref, rho, 'yes' if ok else 'no'))
# the last accepted iterate
z = ST.tz.clone(); xn = keep['x']
W.grad(z)
gz = wp.to_torch(ST.a['gz']).clone()[:n]
gn = gz.norm(dim=1)
F = W.tF.clone(); J = torch.linalg.det(F); S = torch.linalg.svdvals(F)
mv = (z[:n]-xn[:n]).norm(dim=1)
ball = torch.zeros(n, dtype=torch.bool, device=dev); ball[cbody[W.tbdelta > 0]] = True
pen = torch.zeros(n, dtype=torch.float64, device=dev); pen.index_reduce_(0, cbody, W.tbdelta.to(torch.float64), 'amax', include_self=False)
iscap = torch.zeros(n, dtype=torch.bool, device=dev)
try:
    off = W._pos['off']; iscap[torch.nonzero(off[1:] > off[:-1])[:, 0]] = True
except Exception:
    pass
print('\n== at the last accepted iterate: |g| / weight %.3e; the 10 sites carrying most of it (share of |g|^2 %.1f %%)' % (
    float(gz.norm())/ref, 100*float((gn.sort(descending=True).values[:10]**2).sum()/(gn**2).sum())))
print('   site   |g|/wt     det F   s1     s2     s3     moved in step (mm)  ground ball  penetration (mm)  cap cell  height (mm)')
for i in torch.argsort(gn, descending=True)[:10].tolist():
    print('   %5d  %.3e  %+.3f  %.3f  %.3f  %.3f  %8.2f            %-5s        %7.3f           %-5s     %7.1f' % (
        i, float(gn[i])/ref, float(J[i]), *[float(u) for u in S[i]], 1e3*float(mv[i]), bool(ball[i]), 1e3*float(pen[i]), bool(iscap[i]),
        1e3*float(z[i, li.UP]-li.C0)))
# neighbourhood of the worst site
w = int(torch.argmax(gn))
d = (z[:n]-z[w]).norm(dim=1); near = torch.argsort(d)[:30]
print('\n== around site %d (the 30 nearest): det F min %.3f, |g| share %.1f %%, moved max %.2f mm, touching the ground %d' % (
    w, float(J[near].min()), 100*float((gn[near]**2).sum()/(gn**2).sum()), 1e3*float(mv[near].max()), int(ball[near].sum())))
# the whole body
sl = S[:, 1]/S[:, 0].clamp(min=1e-300); s23 = (S[:, 1]-S[:, 2])/S[:, 0].clamp(min=1e-300)
print('== the whole body: det F < 0: %d cells, < 0.2: %d, < 0.4: %d; min det %.3f (cell %d); (s2 - s3) / s1 < 0.02 with det < 0.5: %d; sites moved > 5 mm in the step: %d (max %.1f mm, site %d)' % (
    int((J < 0).sum()), int((J < .2).sum()), int((J < .4).sum()), float(J.min()), int(torch.argmin(J)), int(((s23 < .02) & (J < .5)).sum()),
    int((mv > 5e-3).sum()), 1e3*float(mv.max()), int(torch.argmax(mv))))
# the step's start for comparison
W.grad(xn)
J0 = torch.linalg.det(W.tF)
print('== at the step start (x_n): det F < 0: %d, < 0.2: %d, min %.3f (cell %d)' % (int((J0 < 0).sum()), int((J0 < .2).sum()), float(J0.min()), int(torch.argmin(J0))))
print('   the worst |g| sites\' det F at the start: %s' % ', '.join('%d: %.3f' % (i, float(J0[i])) for i in torch.argsort(gn, descending=True)[:10].tolist()))
