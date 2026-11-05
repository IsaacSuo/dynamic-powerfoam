"""the old DEM (lion_inverse's build of PAC-NeRF elastic scene 0, GT E nu, elastic part only, no contact) against the voxel
FEM of the same material region (talk-pages round01_exact fem_modes_h0.008.npz: E 23008.38, nu 0.3008, unit density, 26
elastic modes, M-normalised): each FEM mode interpolated to our sites (trilinear, dem_fem_mode_match.interpolate) gives
  Rayleigh: sqrt(phi^T K phi / sum V |phi|^2) / (2 pi f_FEM)   (K by central differences of our gradient, lumped mass V)
and Rayleigh-Ritz in the span of the 26: K_r = Phi^T K Phi, M_r = Phi^T diag(V) Phi, our frequencies in that subspace"""
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
f64, dev = torch.float64, 'cuda'
li.set_scene('pacnerf_e0')
cells, g, site_np = li.load_foam()
E, nu = 23008.38, .3008
m = build(g, E, nu, dev, 'noslip', .25, .3, f64, kinematics='exact', nodalvol=True, nonlinear=False, ogden_alpha=2.)
st = ClosureTerms(m, g, cells['points'], cells['radii'])
W = ClosureWarp(st)
W.set_viscosity(1e-4*W.mu, 1e-4*(W.lam+2*W.mu/3), li.DT); W.visc_weight = 2.; W.total_mode()
site = torch.as_tensor(site_np, device=dev).long()
W.set_ball(site, torch.as_tensor(cells['radii'], dtype=f64, device=dev)[site], (0., 1., 0.), -10., 0.)   # (the ground far away)
W.set_cap_positions(); W.set_cap_floor(0.)
W.prepare()
X = W.X_ext; n = W.n; V = g['masses']
W.visc_ref(X); W.elastic_mode()
fem = dict(np.load('/home/fangsuo/py/dynamic-powerfoam/.claude/worktrees/talk-pages/output/round01_exact/fem_modes_h0.008.npz'))


def interpolate(fem, vecs, Xq):
    lo, h, grid = fem['lo'], float(fem['h']), fem['node_grid']
    key = lambda g_: (g_[:, 0]*100003+g_[:, 1])*100003+g_[:, 2]
    order = np.argsort(key(grid)); keys = key(grid)[order]

    def node_of(g_):
        k_ = key(g_); i_ = np.clip(np.searchsorted(keys, k_), 0, len(keys)-1)
        return np.where(keys[i_] == k_, order[i_], -1)
    t = (Xq-lo)/h; b = np.floor(t).astype(int); fr = t-b
    U = vecs.reshape(len(grid), 3, -1)
    out = np.zeros((len(Xq), 3, vecs.shape[1])); ok = np.ones(len(Xq), bool)
    for c in np.array(np.meshgrid([0, 1], [0, 1], [0, 1], indexing='ij')).reshape(3, -1).T:
        n_ = node_of(b+c); ok &= n_ >= 0
        w = np.prod(np.where(c[None] == 1, fr, 1-fr), axis=1)
        out += np.where((n_ >= 0)[:, None, None], w[:, None, None]*U[np.maximum(n_, 0)], 0.)
    near = np.argmin(((Xq[~ok][:, None]-fem['nodes'][None])**2).sum(-1), 1) if (~ok).any() else []
    out[~ok] = U[near]
    return out, ok


Xs = X[:n].cpu().numpy()
Phi, ok = interpolate(fem, fem['V'], Xs)                                             # (n, 3, 26)
print('%d sites, %d inside the voxel mesh (trilinear), %d by the nearest node; FEM f (Hz) %s' % (n, ok.sum(), (~ok).sum(), np.round(fem['f'][:12], 3).tolist()), flush=True)
nm = Phi.shape[2]; HPhi = np.zeros_like(Phi); eps = 1e-6
for j in range(nm):
    d = torch.zeros_like(X); d[:n] = torch.as_tensor(Phi[:, :, j], device=dev)
    s = eps/float(d.abs().max())
    HPhi[:, :, j] = ((W.grad(X+s*d)[1]-W.grad(X-s*d)[1])/(2*s))[:n].cpu().numpy()
Kr = np.einsum('nai,naj->ij', Phi, HPhi); Kr = .5*(Kr+Kr.T)
Mr = np.einsum('n,nai,naj->ij', V, Phi, Phi)
ray = np.sqrt(np.maximum(np.diag(Kr), 0)/np.diag(Mr))/(2*math.pi)
import scipy.linalg as sl
lam = sl.eigh(Kr, Mr, eigvals_only=True); fr = np.sqrt(np.maximum(lam, 0))/(2*math.pi)
print('mode | FEM f (Hz) | ours, Rayleigh on the FEM shape: f, ratio | ours, Rayleigh-Ritz in the 26 modes: f, ratio')
for j in range(nm):
    print('%4d | %6.3f | %6.3f %.3f | %6.3f %.3f' % (j, fem['f'][j], ray[j], ray[j]/fem['f'][j], fr[j], fr[j]/fem['f'][j]), flush=True)
print('lumped mass against the FEM mass on the modes (FEM M-normalised: 1): median %.3f' % np.median(np.diag(Mr)))
