"""F1 (ClosureWarp.set_cap_columns: the caps' normal columns c as unknowns), static checks on PAC-NeRF elastic scene 0's foam
(GT E, nu, no viscosity, the consistent mass with the columns):
 1 rest: E = 0, dE/dq = 0; dE/dq against central differences at a perturbed state (all unknowns, the columns alone)
 2 rigid motion: q -> (R x + t, R c): E unchanged, the gradient rotated (rest and the perturbed state)
 3 the mass pulled back: 1/2 qdot^T M qdot against 1/2 sum_C int rho |u_C + L_C r|^2 with L_C = F_B(qdot) from the kernel
   (the map is linear in q) and the cells' moments; L_C = W for an affine qdot (u = W X + b, cdot = W m); a rigid
   rotation's 1/2 w^T I w
 4 uniform F on the sites: E with c = F m against V psi(F) (the model's law mu |U - I|^2 + lam / 2 ln^2 J); with c = the
   old completion's plane-stress column (equal to the old model's E, a cross-check of the kernels); with c relaxed
   (L-BFGS over the columns, the sites held): how far the free columns lower it"""
import math
import sys
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import free_surface.lion_inverse as li
from experiments.pacnerf_dem_forward import build
from free_surface.closure_warp import ClosureWarp
from free_surface.sphere_closure import ClosureTerms
f64, dev = torch.float64, 'cuda'
li.set_scene('pacnerf_e0')
cells, g, site_np = li.load_foam()
m = build(g, li.GT['E'], li.GT['nu'], dev, 'noslip', .25, .3, f64, kinematics='exact', nodalvol=True, nonlinear=False, ogden_alpha=2.)


def make(cols):
    st = ClosureTerms(m, g, cells['points'], cells['radii'])
    W = ClosureWarp(st)
    W.set_viscosity(0., 0., li.DT); W.visc_weight = 2.; W.total_mode()
    site = torch.as_tensor(site_np, device=dev).long()
    W.set_ball(site, torch.as_tensor(cells['radii'], dtype=f64, device=dev)[site], (0., 1., 0.), -10., 0.)
    W.set_cap_positions(); W.set_cap_floor(0.)
    if cols:
        W.set_cap_columns()
    W.cm_cache = str(li.FOAM/'dem_exact_single'/'cell_moments.npz')
    W.prepare()
    return W, st


W, st = make(True)
n, N = W.n, W.ndof; X = W.X_ext.clone(); mu, lam = W.mu, W.lam
print('F1: %d sites, %d cap columns, ndof %d; mu %.4g lam %.4g (per unit density)' % (n, W.ncap, N, mu, lam), flush=True)
E = lambda q: float(W.grad(q)[0])
gen = torch.Generator(device='cpu').manual_seed(0)
rnd = lambda *s_: torch.randn(*s_, generator=gen, dtype=f64).to(dev)
# ---------------------------------------------------------------- 1
e0, g0 = W.grad(X)
print('\n1. rest: E %.2e, max |dE/dx| %.2e, max |dE/dc| %.2e' % (float(e0), float(g0[:n].abs().max()), float(g0[n:].abs().max())), flush=True)
q0 = X.clone(); q0[:n] += 1e-3*rnd(n, 3); q0[n:] += 2e-2*rnd(W.ncap, 3)
e1, g1 = W.grad(q0); g1 = g1.clone()
for nm, sl in (('all unknowns', slice(0, N)), ('columns alone', slice(n, N))):
    for k in range(2):
        v = torch.zeros_like(q0); v[sl] = rnd(*v[sl].shape); v /= v.abs().max()
        for h in (1e-6, 1e-7):
            fd = (E(q0+h*v)-E(q0-h*v))/(2*h); an = float((g1*v).sum())
            print('   %-13s dir %d h %.0e: difference %+.10e, gradient %+.10e, relative error %.2e' % (nm, k, h, fd, an, abs(fd-an)/abs(an)), flush=True)
# ---------------------------------------------------------------- 2
th = .7; ax = torch.tensor([.3, -.5, .8], dtype=f64, device=dev); ax /= ax.norm()
K_ = torch.tensor([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]], dtype=f64, device=dev)
R = torch.eye(3, dtype=f64, device=dev)+math.sin(th)*K_+(1-math.cos(th))*K_@K_
t = torch.tensor([.1, .2, -.05], dtype=f64, device=dev)
rig = lambda q: torch.cat([q[:n]@R.T+t, q[n:]@R.T], 0)
eR, gR = W.grad(rig(X)); eR = float(eR)
e2, g2 = W.grad(rig(q0)); e2 = float(e2)
print('\n2. rigid motion (rotation %.2f rad, translation): rest E %.2e; perturbed E %.10e -> %.10e (relative %.1e); gradient rotated, max error / max |g| %.1e' % (
    th, eR, float(e1), e2, abs(e2-float(e1))/abs(float(e1)), float((g2-g1@R.T).abs().max()/g1.abs().max())), flush=True)
# ---------------------------------------------------------------- 3
M = W.cmass['M']; dm = torch.as_tensor(W.cmass['d'], device=dev); Jm = torch.as_tensor(W.cmass['J'], device=dev); Vm = st.V
TM = lambda qd: .5*sum(float(qd[:, k].cpu().numpy()@(M@qd[:, k].cpu().numpy())) for k in range(3))


def Tdirect(qd):
    W.grad(qd); L = W.tF.clone(); u = qd[:n]
    Ld = torch.einsum('nij,nj->ni', L, dm)
    return .5*float((Vm*(u*u).sum(1)).sum()+2*(Vm*(u*Ld).sum(1)).sum()+torch.einsum('nij,njk,nik->', L, Jm, L))


for k in range(2):
    qd = torch.cat([rnd(n, 3), rnd(W.ncap, 3)], 0)
    a_, b_ = TM(qd), Tdirect(qd)
    print('\n3. mass pulled back, random qdot %d: 1/2 qdot^T M qdot %.12e, direct %.12e, relative %.1e' % (k, a_, b_, abs(a_-b_)/b_) if k == 0 else
          '   random qdot %d: 1/2 qdot^T M qdot %.12e, direct %.12e, relative %.1e' % (k, a_, b_, abs(a_-b_)/b_), flush=True)
Wa = rnd(3, 3); ba = rnd(3)
qa = torch.cat([X[:n]@Wa.T+ba, X[n:]@Wa.T], 0)
W.grad(qa); print('   affine qdot: max |L_C - W| %.2e over the cells' % float((W.tF-Wa).abs().max()), flush=True)
w_ = torch.tensor([.4, -1.1, .7], dtype=f64, device=dev)
Om = torch.tensor([[0, -w_[2], w_[1]], [w_[2], 0, -w_[0]], [-w_[1], w_[0], 0]], dtype=f64, device=dev)
qr = torch.cat([X[:n]@Om.T, X[n:]@Om.T], 0)
Xc = X[:n]; I3 = torch.eye(3, dtype=f64, device=dev); trJ = torch.einsum('nii->n', Jm)   # (about the origin: int rho (|x|^2 I - x x^T), x = X_C + r)
Iab = (Vm[:, None, None]*(((Xc*Xc).sum(1))[:, None, None]*I3-Xc[:, :, None]*Xc[:, None, :])).sum(0)
Iab = Iab+(2*(Vm*(Xc*dm).sum(1)).sum()*torch.eye(3, dtype=f64, device=dev)-(Vm[:, None, None]*(Xc[:, :, None]*dm[:, None, :]+dm[:, :, None]*Xc[:, None, :])).sum(0))
Iab = Iab+(trJ.sum()*torch.eye(3, dtype=f64, device=dev)-Jm.sum(0))
print('   rigid rotation: 1/2 qdot^T M qdot %.10e, 1/2 w^T I w %.10e (I from the moments about the origin)' % (TM(qr), .5*float(w_@Iab@w_)), flush=True)
# ---------------------------------------------------------------- 4
Wo, sto = make(False)
kap = lam/(lam+2*mu); t1, t2, mA = st.t1, st.t2, st.mA; cc = W.cap_cells


def psi(F):
    U = torch.linalg.svdvals(F)
    return mu*float(((U-1)**2).sum())+.5*lam*math.log(float(torch.linalg.det(F)))**2


def col_ps(F):                                                                      # (comp_f's column at G = F, eps = 0)
    c1, c2 = t1[cc]@F.T, t2[cc]@F.T
    nt = torch.linalg.cross(c1, c2, dim=1); a = nt.norm(dim=1)
    s_ = 1-kap*(torch.sqrt((c1*c1).sum(1)+(c2*c2).sum(1)+2*a)-2)
    return nt/a[:, None]*s_[:, None]


def relax(qF):
    qx = qF.clone()
    def fg(cv):
        qx[n:] = torch.as_tensor(cv.reshape(-1, 3), device=dev)
        e_, g_ = W.grad(qx)
        return float(e_), g_[n:].cpu().numpy().ravel().copy()
    r_ = minimize(fg, qF[n:].cpu().numpy().ravel(), jac=True, method='L-BFGS-B', options=dict(maxiter=3000, gtol=1e-14, ftol=1e-16))
    return r_.fun, r_.nit, float(np.abs(r_.jac).max())


la = .9; lt = 1.
for _ in range(100):
    fv = 2*mu*lt*(lt-1)+lam*math.log(la*lt*lt); lt -= fv/(2*mu*(2*lt-1)+2*lam/lt)
Fs_ = {'uniaxial stress (y 0.90)': torch.diag(torch.tensor([lt, la, lt], dtype=f64, device=dev)),
       'simple shear 0.10': torch.tensor([[1, .1, 0], [0, 1, 0], [0, 0, 1]], dtype=f64, device=dev),
       'dilation 1.05': 1.05*torch.eye(3, dtype=f64, device=dev),
       'general (stretch x rotation)': R@torch.tensor([[1.04, .03, -.02], [.01, .93, .05], [-.03, .02, 1.02]], dtype=f64, device=dev)}
Vt = float(st.V.sum()); c0 = X[:n].mean(0)
print('\n4. uniform F on the sites (energy / V psi(F))', flush=True)
for nm, F in Fs_.items():
    ex = Vt*psi(F)
    qF = torch.cat([c0+(X[:n]-c0)@F.T, X[n:]@F.T], 0)
    eF = E(qF)
    qp = qF.clone(); qp[n:] = col_ps(F); ep = E(qp)
    eo = float(Wo.grad(qF[:n])[0])
    er, nit, gmax = relax(qF)
    print('   %-29s c = F m %.6f | c = plane stress %.6f (old model %.6f, |diff| %.1e) | c relaxed %.6f (%d L-BFGS its, max |dE/dc| %.1e)' % (
        nm, eF/ex, ep/ex, eo/ex, abs(ep-eo), er/ex, nit, gmax), flush=True)
