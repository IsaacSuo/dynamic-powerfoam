"""free vibration on our old DEM (lion_inverse's build of PAC-NeRF elastic scene 0 at GT E, nu; no gravity, the ground far
away): freevib_prep.py's v0 at the sites with the rigid part taken off in the model's own mass (consistent or lumped);
records the motion's V-weighted least-squares projection on the FEM's modes 0-11, q (samples, 12).
args: beta, T, rate, out, [lumped]"""
import math
import sys
import time
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
beta, T, rate, out = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
lumped = len(sys.argv) > 5 and sys.argv[5] == 'lumped'
import free_surface.lion_inverse as li
from experiments.pacnerf_dem_forward import build
from free_surface.closure_drop import colouring, probe_blocks
from free_surface.closure_warp import ClosureWarp, MidpointStep, TrustCG
from free_surface.sphere_closure import ClosureTerms
li.set_scene('pacnerf_e0')
f64, dev = torch.float64, 'cuda'
cells, g, site_np = li.load_foam()
m = build(g, li.GT['E'], li.GT['nu'], dev, 'noslip', .25, .3, f64, kinematics='exact', nodalvol=True, nonlinear=False, ogden_alpha=2.)
st = ClosureTerms(m, g, cells['points'], cells['radii'])
W = ClosureWarp(st)
W.set_viscosity(beta*W.mu, beta*(W.lam+2*W.mu/3), li.DT); W.visc_weight = 2.; W.total_mode()
site = torch.as_tensor(site_np, device=dev).long()
W.set_ball(site, torch.as_tensor(cells['radii'], dtype=f64, device=dev)[site], (0., 1., 0.), -10., 0.)
W.set_cap_positions(); W.set_cap_floor(0.)
W.cm_cache = str(li.FOAM/'dem_exact_single'/'cell_moments.npz')
if lumped:
    W.mass_mode = 'lumped'
W.prepare()
n = W.n; X = st.X; mass = m.mass.to(f64)
ref = float((9.8*mass).norm())
Kbb = probe_blocks(W, X, colouring(st.i.cpu().numpy(), st.j.cpu().numpy(), n))
W.visc_ref(X)
W.tball[0], W.tball[1] = 2., .5
fg = torch.zeros_like(X)
CG = TrustCG(W, mass, li.DT, per=2, product_iters=1); ST = MidpointStep(W, CG, mass, Kbb, fg, li.DT, 0.)
ST.smooth_pred = ST.carry_delta = True
pr = np.load(ROOT/'out_inverse'/'freevib'/'prep.npz')
Xn = X.cpu().numpy(); v0 = pr['v0S'].copy(); M = CG.M
xc = Xn-Xn.mean(0)
Rb = np.zeros((n, 3, 6))
for k in range(3):
    Rb[:, k, k] = 1.; Rb[:, :, 3+k] = np.cross(np.eye(3)[k], xc)
MR = np.stack([M@Rb[:, :, j] for j in range(6)], 2)
coef = np.linalg.solve(np.einsum('naj,nak->jk', Rb, MR), np.einsum('naj,na->j', MR, v0))
v0 = v0-Rb@coef
print('mass %s; rigid part taken off: |translation| %.2e m/s, |rotation| %.2e rad/s' % (W.mass_mode, np.linalg.norm(coef[:3]), np.linalg.norm(coef[3:])), flush=True)
ST.set_state(X, torch.as_tensor(v0, dtype=f64, device=dev), torch.zeros_like(X))
phi, Vw = pr['phiS'], g['masses']
Gi = np.linalg.inv(np.einsum('n,nak,naj->kj', Vw, phi, phi))
spr = round(1/(rate*li.DT)); ns = round(T*rate); tl = 1e-3*ref
q = np.zeros((ns+1, phi.shape[2])); t0 = time.time(); prods = 0
for s in range(1, ns*spr+1):
    p_, its, g2 = ST.step(tl, lambda g2_: .5*tl)
    if not (math.isfinite(g2) and math.sqrt(g2) <= tl):
        raise SystemExit('step %d: not converged (%s after %d Newton iterations)' % (s, g2, its))
    ST.end(); prods += p_
    if s % spr == 0:
        u = ST.tx[:n].cpu().numpy()-Xn
        q[s//spr] = Gi@np.einsum('n,nak,na->k', Vw, phi, u)
        if (s//spr) % round(rate/4) == 0:
            print('t %.2f s (%.0f s, products %d): q0..2 %s' % (s*li.DT, time.time()-t0, prods, np.round(q[s//spr, :3], 6)), flush=True)
np.savez(out, q=q, rate=rate, beta=beta, mass=W.mass_mode)
print('saved', out)
