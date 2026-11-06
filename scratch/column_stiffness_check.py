"""how much the model charges a cap's normal column moved alone: at rest (PAC-NeRF elastic scene 0's foam, GT E, nu, no
viscosity), one cap cell at a time, its completion's column moved by Delta (the closure's traction hook: cap_traction adds
nh t_n / (lam + 2 mu) + t_t / mu to the column, so t = (lam + 2 mu) Delta_n nh or mu Delta_t t1), every site at rest; the
model's energy curvature k = (E(+D) + E(-D) - 2 E(0)) / D^2 against the continuum's for that cap layer's material alone,
V_layer (lam + 2 mu) along the normal and V_layer mu along the surface (V_layer = r A / 3, the cap's cone, its share in
Vdet). Ratios over a sample of cap cells; argv: sample size (default 300), Delta (default 1e-3)"""
import sys
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import free_surface.lion_inverse as li
from experiments.pacnerf_dem_forward import build
from free_surface.closure_warp import ClosureWarp
from free_surface.sphere_closure import ClosureTerms
K = int(sys.argv[1]) if len(sys.argv) > 1 else 300
D = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-3
f64, dev = torch.float64, 'cuda'
li.set_scene('pacnerf_e0')
cells, g, site_np = li.load_foam()
m = build(g, li.GT['E'], li.GT['nu'], dev, 'noslip', .25, .3, f64, kinematics='exact', nodalvol=True, nonlinear=False, ogden_alpha=2.)
st = ClosureTerms(m, g, cells['points'], cells['radii'])
W = ClosureWarp(st)
W.set_viscosity(0., 0., li.DT); W.visc_weight = 2.; W.total_mode()
site = torch.as_tensor(site_np, device=dev).long()
W.set_ball(site, torch.as_tensor(cells['radii'], dtype=f64, device=dev)[site], (0., 1., 0.), -10., 0.)
W.set_cap_positions(); W.set_cap_floor(0.)
W.mass_mode = 'lumped'
W.prepare()
mu, lam = W.mu, W.lam
X = W.X_ext
E0 = float(W.grad(X)[0])
Vl = torch.zeros(st.n, dtype=f64, device=dev).index_add(0, st.c_body, st.c_r*st.c_A/3)
capcells = torch.nonzero(st.cap & (Vl > 0))[:, 0]
gen = torch.Generator(device='cpu').manual_seed(0)
pick = capcells[torch.randperm(len(capcells), generator=gen)[:K].to(dev)]
mA, t1 = st.mA, st.t1
print('rest energy %.3e; %d of %d cap cells; Delta %g; lam %.4g, mu %.4g (per unit density)' % (E0, len(pick), len(capcells), D, lam, mu), flush=True)
rows = []
for b in pick.tolist():
    k_ = {}
    for nm, t in (('n', (lam+2*mu)*mA[b]), ('t', mu*t1[b])):
        e = []
        for sgn in (1., -1.):
            W.ttrac.zero_(); W.ttrac[b] = sgn*D*t
            e.append(float(W.grad(X)[0]))
        k_[nm] = (e[0]+e[1]-2*E0)/D**2
    W.ttrac.zero_()
    v = float(Vl[b]); share = v/float(st.V[b])
    rows.append((k_['n']/(v*(lam+2*mu)), k_['t']/(v*mu), share, abs(float(mA[b, 1]))))
R = np.array(rows)
q = lambda a: '%.3f [%.3f, %.3f]' % (np.median(a), np.percentile(a, 10), np.percentile(a, 90))
print('model / continuum (median [10 %, 90 %]):')
print('  normal column (compression):   %s' % q(R[:, 0]))
print('  tangential column (shear):     %s' % q(R[:, 1]))
print('  (the cap layer\'s share of its cell volume: %s)' % q(R[:, 2]))
for lo_, hi_ in ((0., .5), (.5, .85), (.85, 1.01)):
    s_ = (R[:, 3] >= lo_) & (R[:, 3] < hi_)
    if s_.any():
        print('  caps with |m_y| %.2f-%.2f (%d): normal %s, shear %s' % (lo_, min(hi_, 1.), s_.sum(), q(R[s_, 0]), q(R[s_, 1])))
