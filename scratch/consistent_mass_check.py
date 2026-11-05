"""offline checks of a constant consistent mass for the old DEM (lion_inverse's build of PAC-NeRF elastic scene 0, the
one-way cap), before any solver change:
  cell velocity field  v(X) = u_C + L_C(u) (X - X_C),  L_C(u) = (S_C(u) + G_C(u) Mcap_C) Mt_C^-1
    S_C = sum_walls 1/2 A (u_j - u_i) (x) n (k_S_bonds on u), G_C the 2-ring fit sum_e (u_k - u_C) (x) dX_e M2_C^-1 (k_closP
    without comp_f) on the cap cells; L_C = sum_k u_k a_Ck^T exactly linear, exact for any affine u (checked below)
  T = 1/2 sum_C int_C rho |v|^2 = 1/2 u^T (Mh (x) I3) u,  Mh_kl = sum_C [m_C d_kC d_lC + m_C (d_kC a_Cl.d_C + d_lC a_Ck.d_C) + a_Ck^T J_C a_Cl]
    d_C = centroid - X_C, J_C = int rho r r^T (r = X - X_C) of the cell (its power cell inside its sphere), sampled on a grid
    and scaled to the cell's volume V (unit density, as the model's mass)
checks: (1) affine exactness of L; (2) total mass, rigid-rotation inertia against the sampled material; (3) Mh positive
definite and the spectrum of D^-1/2 Mh D^-1/2 (D its diagonal: the block-Jacobi preconditioner's view); (4) the modal
mass along the voxel FEM's modes and the Rayleigh-Ritz frequencies with K of our model (as pacnerf_modes.py), lumped
against consistent. Saves the moments and Mh to out_inverse/consistent_mass/."""
import math
import sys
import time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.linalg as sl
import torch
from scipy.spatial import cKDTree
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'scratch'))
import free_surface.lion_inverse as li
from experiments.pacnerf_dem_forward import build
from free_surface.closure_warp import ClosureWarp
from free_surface.sphere_closure import ClosureTerms
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
    if (~ok).any():
        _, near = cKDTree(fem['nodes']).query(Xq[~ok]); out[~ok] = U[near]
    return out, ok


OUT = ROOT/'out_inverse'/'consistent_mass'; OUT.mkdir(parents=True, exist_ok=True)
SP = float(sys.argv[1]) if len(sys.argv) > 1 else 1.5e-3
f64, dev = torch.float64, 'cuda'
t0 = time.time()
li.set_scene('pacnerf_e0')
cells, g, site_np = li.load_foam()
E, nu = li.GT['E'], li.GT['nu']
m = build(g, E, nu, dev, 'noslip', .25, .3, f64, kinematics='exact', nodalvol=True, nonlinear=False, ogden_alpha=2.)
st = ClosureTerms(m, g, cells['points'], cells['radii'])
W = ClosureWarp(st)
W.set_viscosity(1e-4*W.mu, 1e-4*(W.lam+2*W.mu/3), li.DT); W.visc_weight = 2.; W.total_mode()
site = torch.as_tensor(site_np, device=dev).long()
W.set_ball(site, torch.as_tensor(cells['radii'], dtype=f64, device=dev)[site], (0., 1., 0.), -10., 0.)
W.set_cap_positions(); W.set_cap_floor(0.)
W.prepare()
X = W.X_ext; n = W.n
W.visc_ref(X); W.elastic_mode()
c = lambda t: t.detach().cpu().numpy()
Xn, mass, V = c(X[:n]), c(m.mass.to(f64)), np.asarray(g['masses'], float)
print('built in %.0f s: %d cells, %d bonds, %d cap cells; mass vs V: max rel %.2e' % (time.time()-t0, n, len(st.i), int(st.cap.sum()), np.abs(mass/V-1).max()), flush=True)

# ---- the cells' moments about their sites (power cell inside its sphere), sampled
p, r = cells['points'].astype(np.float64), cells['radii'].astype(np.float64)
Rm = r.max(); tree = cKDTree(np.c_[p, np.sqrt(Rm*Rm-r*r)])
lo, hi = (p-r[:, None]).min(0), (p+r[:, None]).max(0)
ax = [np.arange(lo[k]+SP/2, hi[k], SP) for k in range(3)]
cnt = np.zeros(n); S1 = np.zeros((n, 3)); S2 = np.zeros((n, 3, 3))
t1 = time.time()
for ia in range(0, len(ax[0]), 8):
    G = np.stack(np.meshgrid(ax[0][ia:ia+8], ax[1], ax[2], indexing='ij'), -1).reshape(-1, 3)
    _, own = tree.query(np.c_[G, np.zeros(len(G))], workers=-1)
    q = ((G-p[own])**2).sum(1) <= r[own]**2
    G, own = G[q], own[q]; rr = G-p[own]
    cnt += np.bincount(own, minlength=n)
    for k in range(3):
        S1[:, k] += np.bincount(own, rr[:, k], n)
        for l in range(k, 3):
            S2[:, k, l] += np.bincount(own, rr[:, k]*rr[:, l], n)
for k in range(3):
    for l in range(k):
        S2[:, k, l] = S2[:, l, k]
vs = cnt*SP**3
print('sampled at %.1f mm in %.0f s: volume %.5f vs sum V %.5f; per cell vs V: median %.4f, 1-99 %% %.3f-%.3f; cells with < 20 samples %d (none %d)' % (
    1e3*SP, time.time()-t1, vs.sum(), V.sum(), np.median(vs/V), *np.percentile(vs/V, [1, 99]), (cnt < 20).sum(), (cnt == 0).sum()), flush=True)
cs = np.maximum(cnt, 1)
d = S1/cs[:, None]                                                     # centroid - site
J = S2/cs[:, None, None]*mass[:, None, None]                           # second moment about the site, scaled to the mass
d[cnt == 0] = 0.; J[cnt == 0] = 0.
off_n = np.linalg.norm(d, axis=1)
print('site-centroid offset (mm): median %.1f, 90 %% %.1f, max %.1f; cap cells median %.1f, interior %.1f' % (
    1e3*np.median(off_n), 1e3*np.percentile(off_n, 90), 1e3*off_n.max(), 1e3*np.median(off_n[c(st.cap)]), 1e3*np.median(off_n[~c(st.cap)])), flush=True)

# ---- L's coefficients a_Ck (L_C = sum_k u_k a_Ck^T)
bi, bj, A, nrm = c(st.i).astype(int), c(st.j).astype(int), c(st.A), c(st.nrm)
w = .5*A[:, None]*nrm
rows_C = [bi, bi, bj, bj]; rows_k = [bj, bi, bj, bi]; vecs = [w, -w, w, -w]
P = W._pos; off, pk, dX, M2i = c(P['off']), c(P['pk']).astype(int), c(P['dX']), c(P['M2i'])
Mc, Mti = c(st.Mcap), c(st.Mt_inv)
pc = np.repeat(np.arange(n), np.diff(off))
cv = np.einsum('eij,ej->ei', Mc[pc], np.einsum('eij,ej->ei', M2i[pc], dX))
rows_C += [pc, pc]; rows_k += [pk, pc]; vecs += [cv, -cv]
Cc, Kk, Vv = np.concatenate(rows_C), np.concatenate(rows_k), np.concatenate(vecs)
key = Cc.astype(np.int64)*n+Kk
uk, inv = np.unique(key, return_inverse=True)
a = np.zeros((len(uk), 3))
for k in range(3):
    a[:, k] = np.bincount(inv, Vv[:, k], len(uk))
aC, aK = uk//n, uk % n
a = np.einsum('eij,ej->ei', Mti[aC], a)                               # a_Ck = Mt_C^-1 (sum of terms), Mt symmetric
sumA = np.zeros((n, 3)); sumXA = np.zeros((n, 3, 3))
for k in range(3):
    sumA[:, k] = np.bincount(aC, a[:, k], n)
    for l in range(3):
        sumXA[:, k, l] = np.bincount(aC, Xn[aK, k]*a[:, l], n)
print('(1) affine exactness: |sum_k a_Ck| max %.1e; |sum_k X_k a_Ck^T - I| max %.1e; stencil size median %d, max %d' % (
    np.abs(sumA).max(), np.abs(sumXA-np.eye(3)).max(), int(np.median(np.bincount(aC))), np.bincount(aC).max()), flush=True)

# ---- Mh: per cell the block over its stencil
t2 = time.time()
starts = np.searchsorted(aC, np.arange(n+1))
ri, ci, vv = [], [], []
for C in range(n):
    s0, s1 = starts[C], starts[C+1]
    ks, Ac = aK[s0:s1], a[s0:s1]
    e = (ks == C).astype(float)
    Ad = Ac@d[C]
    blk = Ac@J[C]@Ac.T+mass[C]*(np.outer(e, Ad)+np.outer(Ad, e)+np.outer(e, e))
    ri.append(np.repeat(ks, len(ks))); ci.append(np.tile(ks, len(ks))); vv.append(blk.ravel())
Mh = sp.coo_matrix((np.concatenate(vv), (np.concatenate(ri), np.concatenate(ci))), shape=(n, n)).tocsr()
Mh = (.5*(Mh+Mh.T)).tocsr()
sp.save_npz(OUT/'Mh.npz', Mh); np.savez(OUT/'moments.npz', d=d, J=J, cnt=cnt, spacing=SP, a=a, aC=aC, aK=aK)
one = np.ones(n)
print('Mh assembled in %.0f s: nnz %d (%.0f per row); total 1^T M 1 = %.5f vs sum m %.5f; row sums - m: max |.| / m %.3f' % (
    time.time()-t2, Mh.nnz, Mh.nnz/n, one@Mh@one, mass.sum(), np.abs(Mh@one-mass).max()/np.median(mass)), flush=True)

# (2) rigid rotation about the centre of mass: 1/2 u^T M u = 1/2 w^T I w, against the sampled material's inertia
com = (mass[:, None]*(Xn+d)).sum(0)/mass.sum()
Ims = np.zeros((3, 3))
for C in range(n):                                                     # material inertia about com from the cells' moments
    rc = Xn[C]-com
    Jc = J[C]+mass[C]*(np.outer(rc, d[C])+np.outer(d[C], rc)+np.outer(rc, rc))
    Ims += np.trace(Jc)*np.eye(3)-Jc
Il = np.zeros((3, 3))
for C in range(n):
    rc = Xn[C]-com; Il += mass[C]*(rc@rc*np.eye(3)-np.outer(rc, rc))
Ic = np.zeros((3, 3))
for k_ in range(3):
    for l_ in range(3):
        wk, wl = np.eye(3)[k_], np.eye(3)[l_]
        uk_, ul_ = np.cross(wk, Xn-com), np.cross(wl, Xn-com)
        Ic[k_, l_] = sum(uk_[:, q]@(Mh@ul_[:, q]) for q in range(3))
ev = lambda M_: np.linalg.eigvalsh(M_)
print('(2) principal moments about the com: material %s | consistent %s | lumped %s' % (np.round(ev(Ims), 6), np.round(ev(Ic), 6), np.round(ev(Il), 6)), flush=True)

# (3) definiteness and D^-1/2 M D^-1/2
Dh = 1/np.sqrt(Mh.diagonal())
Md = Mh.toarray()*Dh[:, None]*Dh[None, :]
w_ = np.linalg.eigvalsh(Md)
print('(3) D^-1/2 M D^-1/2: eigenvalues min %.4f, max %.4f (condition %.1f); diagonal / m median %.3f' % (w_[0], w_[-1], w_[-1]/w_[0], np.median(Mh.diagonal()/mass)), flush=True)
del Md

# (4) along the FEM modes: modal masses, Rayleigh-Ritz with our K
fem = dict(np.load('/home/fangsuo/py/dynamic-powerfoam/.claude/worktrees/talk-pages/output/round01_exact/fem_modes_h0.008.npz'))
nm = 26
Phi, _ = interpolate(fem, fem['V'][:, :nm], Xn)                        # (n, 3, nm)
HPhi = np.zeros_like(Phi); eps = 1e-6
for j in range(nm):
    dz = torch.zeros_like(X); dz[:n] = torch.as_tensor(Phi[:, :, j], device=dev)
    s_ = eps/float(dz.abs().max())
    HPhi[:, :, j] = c((W.grad(X+s_*dz)[1]-W.grad(X-s_*dz)[1])/(2*s_))[:n]
Kr = np.einsum('nai,naj->ij', Phi, HPhi); Kr = .5*(Kr+Kr.T)
Ml = np.einsum('n,nai,naj->ij', mass, Phi, Phi)
Mc_ = sum(Phi[:, q, :].T@(Mh@Phi[:, q, :]) for q in range(3)); Mc_ = .5*(Mc_+Mc_.T)
fl = np.sqrt(np.maximum(sl.eigh(Kr, Ml, eigvals_only=True), 0))/(2*math.pi)
fc = np.sqrt(np.maximum(sl.eigh(Kr, Mc_, eigvals_only=True), 0))/(2*math.pi)
ff = fem['f'][:nm]
print('(4) mode | FEM f | modal mass lumped, consistent | Rayleigh-Ritz f / f_FEM: lumped, consistent')
for j in range(nm):
    print('   %2d | %6.3f | %.3f %.3f | %.3f %.3f' % (j, ff[j], Ml[j, j], Mc_[j, j], fl[j]/ff[j], fc[j]/ff[j]))
print('   median f / f_FEM over modes 0-8: lumped %.3f, consistent %.3f; 0-25: %.3f, %.3f' % (
    np.median(fl[:9]/ff[:9]), np.median(fc[:9]/ff[:9]), np.median(fl/ff), np.median(fc/ff)), flush=True)
