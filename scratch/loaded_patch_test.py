"""the patch test with surface traction: is the one-way cap's closure exact on a loaded surface? The jack (PAC-NeRF
elastic scene 0's foam, GT E, nu) under a uniform uniaxial-stress compression along y, x = c + F0 (X - c),
F0 = diag(lt, la, lt) with lateral stress zero for the model's own law psi = mu |U - I|^2 + lam / 2 (ln J)^2
(2 mu lt (lt - 1) = -lam ln J). Exact: F_B = F0 in every cell, energy = V psi(F0). Caps whose normal m is horizontal
carry no traction (plane stress about m: the completion's premise holds); caps with a vertical component carry
sigma . m = sigma_yy m_y y (the completion takes them as traction-free). Reports |F_B - F0| / |F0| by |m_y| bins (and the
interior cells), the vertical stretch F_B y . y there, and the total energy against V psi(F0). argv: la (default 0.9)"""
import math
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
la = float(sys.argv[1]) if len(sys.argv) > 1 else .9
with_t = len(sys.argv) > 2 and sys.argv[2] == 'traction'
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
lt = 1.
for _ in range(100):                                                                # (2 mu lt (lt - 1) + lam ln(la lt^2) = 0, Newton)
    fval = 2*mu*lt*(lt-1)+lam*math.log(la*lt*lt); dfv = 2*mu*(2*lt-1)+2*lam/lt; lt -= fval/dfv
F0 = torch.diag(torch.tensor([lt, la, lt], dtype=f64, device=dev))
J = la*lt*lt
psi = mu*((la-1)**2+2*(lt-1)**2)+.5*lam*math.log(J)**2
X = W.X_ext; c = X.mean(0)
if with_t:                                                                          # (the exact traction on the caps: linear sigma(F0) . m)
    eps_ = F0-torch.eye(3, dtype=f64, device=dev); sig = lam*torch.trace(eps_)*torch.eye(3, dtype=f64, device=dev)+2*mu*eps_
    W.ttrac.zero_(); W.ttrac[st.cap] = st.mA[st.cap]@sig.T
    print('caps loaded with sigma(F0) . m, sigma_yy %.4f (per unit density)' % float(sig[1, 1]))
x = c+(X-c)@F0.T
E, _ = W.grad(x); E = float(E)
ts = W.tsum.cpu().numpy().ravel()
print('energy rows: bonds %.6g, volume %.6g, outer layer o1 %.6g' % (ts[0], ts[1], ts[2]))
FB = W.tF.clone()
err = ((FB-F0).norm(dim=(1, 2))/F0.norm()).cpu().numpy()
vert = FB[:, :, 1][:, 1].cpu().numpy()
cap = st.cap.cpu().numpy(); my = np.abs(st.mA[:, 1].cpu().numpy())
print('F0 = diag(%.4f, %.4f, %.4f), J %.4f; energy: DEM %.6g against V psi(F0) %.6g -> ratio %.4f' % (lt, la, lt, J, E, float(st.V.sum())*psi, E/(float(st.V.sum())*psi)))
print('cells | count | |F_B - F0| / |F0| median, 90 %% | vertical stretch F_B_yy median (exact %.4f)' % la)
print('interior      | %5d | %.2e, %.2e | %.4f' % ((~cap).sum(), np.median(err[~cap]), np.percentile(err[~cap], 90), np.median(vert[~cap])))
for lo_, hi_ in ((0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01)):
    q = cap & (my >= lo_) & (my < hi_)
    print('cap |m_y| %.1f-%.1f | %5d | %.2e, %.2e | %.4f' % (lo_, min(hi_, 1.), q.sum(), np.median(err[q]), np.percentile(err[q], 90), np.median(vert[q])))
