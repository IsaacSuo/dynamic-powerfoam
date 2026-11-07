"""k1's explicit step limit on its own operator (MergedUL at rest on PAC-NeRF plasticine scene 0's foam, as merged_drop.py):
K = d2E/dx2 by central differences of k1's gradient pass (at theta 1/2 the step's Hessian in z is the material stiffness:
the cell energy th^2 W(e(z) / th) and the volume and springs read at z all give W''(de/dx)^2), lumped masses V0 (per unit
density). Lanczos with full reorthogonalisation on M^-1/2 K M^-1/2 gives the highest eigenvalues: omega_max, the explicit
(leapfrog) limit 2 / omega, how many modes lie above the MPM substep's 2 / (1/4800 s) and 1/2400's, and on which cells each of
the highest modes sits (volume over the median, shortest current wall pair distance over the cell size, exposed or not).
Also k1's own diagonal blocks (ul_blocks, the preconditioner's estimate) per cell: omega_i^2 = lambda_max(K_ii) / m_i.
Run (k1's environment): PYTHONPATH=$PWD python -u free_surface/k1_explicit_dt.py [--foam foams/fourier8k_clean] [--lanczos 160]"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import warp as wp

from free_surface.merged_ul import MergedUL

f64, dev = torch.float64, 'cuda'
a_ = argparse.ArgumentParser()
a_.add_argument('--foam', type=Path, default=Path('foams/fourier8k_clean'))
a_.add_argument('--lanczos', type=int, default=160)
a_.add_argument('--top', type=int, default=12)
args = a_.parse_args()
gt = json.loads((args.foam/'plasticine_scenes.json').read_text())['scenes']['0']
E, nu, ys = gt['E']/1000, gt['nu'], gt['ys']/1000
g = np.load(args.foam/'dem_exact_single'/'geometry.npz'); cl = np.load(args.foam/'dem_exact_single'/'cells.npz')
X0, r0, V0 = g['points'], cl['radii'], g['masses']; n = len(X0)
W = MergedUL(X0, r0, V0, g['pairs'], g['areas'], g['contacts'], E, nu, ys, theta=.5, kshare=1/6, grad='gg')
W.set_ball((0., 1., 0.), -.05, 10.)
W.update_ul(W.X)
W.prepare()
X = W.X.clone(); m = torch.as_tensor(V0, dtype=f64, device=dev); Ms = torch.sqrt(m)[:, None]


def grad(x):
    W.tx.copy_(x); wp.capture_launch(W.graph)
    return W.tgx.clone()


hfd = 1e-7
Kv = lambda v: (grad(X+hfd*v)-grad(X-hfd*v))/(2*hfd)
Av = lambda v: Kv(v/Ms)/Ms                                                           # (M^-1/2 K M^-1/2)
# Lanczos, full reorthogonalisation
k = args.lanczos
Q = torch.zeros(k+1, n, 3, dtype=f64, device=dev)
q = torch.randn(n, 3, dtype=f64, device=dev, generator=torch.Generator(device=dev).manual_seed(0)); Q[0] = q/q.norm()
al, be = [], []
for it in range(k):
    w = Av(Q[it])
    a = float((w*Q[it]).sum()); al.append(a)
    w = w-a*Q[it]-(be[-1]*Q[it-1] if it > 0 else 0.)
    c = torch.einsum('kab,ab->k', Q[:it+1], w); w = w-torch.einsum('k,kab->ab', c, Q[:it+1])       # (twice is enough)
    c = torch.einsum('kab,ab->k', Q[:it+1], w); w = w-torch.einsum('k,kab->ab', c, Q[:it+1])
    b = float(w.norm()); be.append(b); Q[it+1] = w/b
T = np.diag(al)+np.diag(be[:-1], 1)+np.diag(be[:-1], -1)
lam, S = np.linalg.eigh(T)
om = np.sqrt(np.clip(lam, 0, None))
V = m; Vmed = float(V.median()); h = V**(1/3)
G = W._ulG; kp = G.wall_A.detach() > 0
wi, wj = G.i[kp].long(), G.j[kp].long(); Lw = (X[wj]-X[wi]).norm(dim=1)
Lmin = torch.full((n,), float('inf'), dtype=f64, device=dev).scatter_reduce(0, wi, Lw, 'amin').scatter_reduce(0, wj, Lw, 'amin')
cap = (G.patch_A.detach() > 0)
print(f"k1 at rest, plasticine scene 0 (E {gt['E']:.4g} Pa, nu {nu:.3f}): {n} cells, median h {1e3*float(h.median()):.2f} mm; "
      f"continuum estimate 2 c_p / h = {2*math.sqrt(E*(1-nu)/((1+nu)*(1-2*nu)))/float(h.median()):.3e} rad/s", flush=True)
print(f"Lanczos {k}: omega_max {om[-1]:.3e} rad/s -> explicit limit 2 / omega_max = {2/om[-1]:.3e} s (MPM substep {1/4800:.3e}, ratio {2/om[-1]*4800:.3g})", flush=True)
for name, dt in (('MPM substep 1/4800', 1/4800), ('1/2400', 1/2400)):
    print(f"  converged modes above 2 / ({name} s) = {2/dt:.3e} rad/s: {int((om > 2/dt).sum())} (of the top {k}; the lowest of the top is {om[-min(k, 40)]:.3e})", flush=True)
for r_ in range(1, args.top+1):
    y = torch.as_tensor(S[:, -r_], dtype=f64, device=dev)
    v = torch.einsum('k,kab->ab', y, Q[:k]); wv = (v*v).sum(1); top = torch.argsort(wv, descending=True)[:3]
    print(f"  mode {r_}: omega {om[-r_]:.3e} (dt {2/om[-r_]:.2e} s); cells " + "; ".join(
        f"{c} {100*float(wv[c]):.0f} % (V/med {float(V[c])/Vmed:.2f}, Lmin/h {float(Lmin[c]/h[c]):.3f}, {'surface' if bool(cap[c]) else 'inside'})" for c in top.tolist()), flush=True)
K = W.ul_blocks()
lk = torch.linalg.eigvalsh(.5*(K+K.transpose(1, 2)))[:, -1]
omi = torch.sqrt(lk.clamp(min=0)/m)
qn = lambda p: float(torch.quantile(omi, p))
print(f"ul_blocks' local omega_i: median {qn(.5):.3e}, 99 % {qn(.99):.3e}, 99.9 % {qn(.999):.3e}, max {float(omi.max()):.3e}; "
      f"cells above the MPM substep's 2/dt: {int((omi > 9600).sum())}", flush=True)
