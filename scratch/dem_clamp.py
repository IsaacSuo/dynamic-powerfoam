"""the clamped test on our old DEM (lion_inverse.forward's build of PAC-NeRF elastic scene 0 at GT E, nu, beta given; the
ground far away): the clamp cells of clamp_prep.py held by a mass MF (argument 8, default 1e6) times theirs, no gravity on them, starting at rest.
Loads as mpm_clamp.py: gravity 1 (from rest) or kick 1 (gravity off, v0's horizontal part on the free cells). Records the
groups' volume-weighted mean positions (all, tips 0-3, core) at rate Hz for T s.
args: sets npz, gravity, kick, beta, T, rate, out"""
import math
import sys
import time
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sets, gravity, kick, beta, T, rate, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6]), sys.argv[7]
MF = float(sys.argv[8]) if len(sys.argv) > 8 else 1e6
KS = float(sys.argv[9]) if len(sys.argv) > 9 else 1.
import free_surface.lion_inverse as li
from experiments.pacnerf_dem_forward import build
from free_surface.closure_drop import colouring, probe_blocks
from free_surface.closure_warp import ClosureWarp, MidpointStep, TrustCG
from free_surface.sphere_closure import ClosureTerms
li.set_scene('pacnerf_e0')
f64, dev = torch.float64, 'cuda'
E, nu = li.GT['E'], li.GT['nu']
Sz = np.load(sets); clamp = torch.as_tensor(Sz['clamp_d'], device=dev); groups = Sz['groups_d']
cells, g, site_np = li.load_foam()
m = build(g, E, nu, dev, 'noslip', .25, .3, f64, kinematics='exact', nodalvol=True, nonlinear=False, ogden_alpha=2.)
st = ClosureTerms(m, g, cells['points'], cells['radii'])
W = ClosureWarp(st)
W.set_viscosity(beta*W.mu, beta*(W.lam+2*W.mu/3), li.DT)
W.visc_weight = 2.; W.total_mode()
boc = torch.as_tensor(np.asarray(g['body_of_cell']), device=dev).long()
site = torch.as_tensor(site_np, device=dev).long()
cbody = boc[site]; Rb = torch.as_tensor(cells['radii'], dtype=f64, device=dev)[site]
upt = tuple(1. if k == li.UP else 0. for k in range(3))
W.set_ball(cbody, Rb, upt, -10., 0.)                                                  # (the ground far away)
W.set_cap_positions(); W.set_cap_floor(0.)
W.cm_cache = str(li.FOAM/'dem_exact_single'/'cell_moments.npz')
if len(sys.argv) > 10 and sys.argv[10] == 'lumped':
    W.mass_mode = 'lumped'
W.prepare()
n = W.n; X = st.X; mass = m.mass.to(f64); M3 = mass[:, None]
up = torch.tensor(upt, dtype=f64, device=dev); fg = -9.8*M3*up; ref = float(fg.norm())
if W.cmass is not None:
    fg = -9.8*torch.as_tensor(W.cmass['rowsum'], dtype=f64, device=dev)[:, None]*up
fg = fg*gravity; fg[clamp] = 0.
mass_c = mass.clone(); mass_c[clamp] *= MF
Kbb = probe_blocks(W, X, colouring(st.i.cpu().numpy(), st.j.cpu().numpy(), n))
W.visc_ref(X)
W.tball[0], W.tball[1] = 2., .5
CG = TrustCG(W, mass_c, li.DT, per=2, product_iters=1); ST = MidpointStep(W, CG, mass_c, Kbb, fg, li.DT, 0.)
ST.smooth_pred = ST.carry_delta = True
v = torch.zeros_like(X)
if kick:
    v[:] = KS*torch.tensor([li.V0[0], 0., li.V0[2]], dtype=f64, device=dev)
v[clamp] = 0.
ST.set_state(X, v, fg/mass_c[:, None])
Wg = groups.astype(np.float64)*g['masses'][None]; Wg /= Wg.sum(1, keepdims=True)
spr = round(1/(rate*li.DT)); ns = round(T*rate); tl = 1e-3*ref
print('mass %s; cells %d, clamped %d, beta %g, gravity %d kick %d, %d steps per sample' % (W.mass_mode, n, int(clamp.sum()), beta, gravity, kick, spr), flush=True)
rec = np.zeros((ns+1, len(Wg), 3)); rec[0] = Wg@X.cpu().numpy()
t0 = time.time()
for s in range(1, ns*spr+1):
    p_, its, g2 = ST.step(tl, lambda g2_: .5*tl)
    if not (math.isfinite(g2) and math.sqrt(g2) <= tl):
        raise SystemExit('step %d: not converged (%s after %d Newton iterations)' % (s, g2, its))
    ST.end()
    if s % spr == 0:
        rec[s//spr] = Wg@ST.tx[:n].cpu().numpy()
        if (s//spr) % round(rate/8) == 0:
            print('t %.3f s (%.1f s): com %s' % (s*li.DT, time.time()-t0, np.round(rec[s//spr, 0], 4)), flush=True)
print('clamped cells: largest displacement %.3g m' % float((ST.tx[:n][clamp]-X[clamp]).norm(dim=1).max()))
np.savez(out, rec=rec, rate=rate, beta=beta, gravity=gravity, kick=kick, hc=float(Sz['hc']))
print('saved', out)
