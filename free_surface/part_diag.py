"""Diagnostics of the part's FEM comparison (part_modes.py, 2026-10-22), offline, the FEM modes from the cache:
  1. B again with the skin points outside the voxel mesh extrapolated (trilinear from the nearest complete voxel, local
     coordinates outside [0, 1]) instead of the nearest node;
  2. where the free skin (C) leaves the FEM's surface motion: with the bodies on the FEM mode, E(y_B) - E(y_C) =
     1/2 dy^T K_yy dy (C minimises over the skin), dy = y_B - y_C; its split over corners / arc points / part points
     (dy_t^T K_yy dy), and |dy| against |y_B| per kind;
  3. consistent mass (KB_DERIVATION.md 21.3: the cell's own affine motion u_B + G_B (X - X_B), G_B the model's gradient
     of the displacement; d_B, J_B by 0.5 mm grid sampling of each cell, scaled to its exact volume): mass ratio along
     the FEM modes for A and C, and each model's own modes with that mass (paired by MAC)."""
import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from experiments.dem_fem_mode_match import interpolate
from experiments.pacnerf_dem_forward import build
from free_surface.grad_weights import MidpointTerms
from free_surface.part_modes import PatchSprings, dense_hessian, mac_pairs
from free_surface.boundary_skin import BoundaryTerms
from free_surface.sphere_model import SphereTerms

f64, dev = torch.float64, 'cuda'


def interpolate_extrapolated(fem, vecs, Xq):
    """trilinear in the voxel holding the point when its 8 nodes exist, else trilinear extrapolation from the nearest
    voxel with all 8 nodes"""
    out, nout = interpolate(fem, vecs, Xq)
    lo, h, grid = fem['lo'], fem['h'], fem['node_grid']
    key = lambda g: (g[:, 0]*100003+g[:, 1])*100003+g[:, 2]
    keys = set(key(grid).tolist())
    order = {k_: i for i, k_ in enumerate(key(grid).tolist())}
    corners = np.array(np.meshgrid([0, 1], [0, 1], [0, 1], indexing='ij')).reshape(3, -1).T
    cand = np.unique(grid[:, None, :]-corners[None], axis=0).reshape(-1, 3)
    full = np.array([all(k_ in keys for k_ in key(c_+corners).tolist()) for c_ in cand])
    vox = cand[full]
    t = (Xq-lo)/h; b = np.floor(t).astype(int)
    inside = np.array([all(k_ in keys for k_ in key(b_+corners).tolist()) for b_ in b])
    U = vecs.reshape(len(grid), 3, -1)
    _, nv = cKDTree(vox+.5).query(t[~inside])
    for q, (row, v_) in enumerate(zip(np.flatnonzero(~inside), vox[nv])):
        f = t[row]-v_
        acc = 0.
        for c_ in corners:
            w = np.prod(np.where(c_ == 1, f, 1-f))
            acc = acc+w*U[order[key((v_+c_)[None])[0]]]
        out[row] = acc
    return out, ~inside


def cell_moments(p, r, X, V, spacing=.0005):
    """per cell (one body per cell): d = int (X - X_B) dV and J = int (X - X_B)(X - X_B)^T dV, grid-sampled, scaled to V"""
    lo, hi = (p-r[:, None]).min(0), (p+r[:, None]).max(0)
    pt, rt = torch.as_tensor(p, device=dev), torch.as_tensor(r, device=dev)
    n = len(p); cnt = np.zeros(n); d = np.zeros((n, 3)); J = np.zeros((n, 3, 3))
    axes = [np.arange(lo[k]+spacing/2, hi[k], spacing) for k in range(3)]
    for x0 in axes[0]:
        G = np.stack(np.meshgrid([x0], axes[1], axes[2], indexing='ij'), -1).reshape(-1, 3)
        y = torch.as_tensor(G, device=dev)
        pw = torch.cdist(y, pt)**2-rt[None]**2
        v, a = pw.min(1)
        ok = (v <= 0).cpu().numpy(); a = a.cpu().numpy()[ok]; R = G[ok]-X[a]
        np.add.at(cnt, a, 1.); np.add.at(d, a, R); np.add.at(J, a, R[:, :, None]*R[:, None, :])
    s = np.where(cnt > 0, V/np.maximum(cnt, 1)/spacing**3, 0.)       # cells without a sample (tiny): a point mass at the site
    return d*spacing**3*s[:, None], J*spacing**3*s[:, None, None], cnt


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--part', type=Path, default=Path('output/test_part'))
    a.add_argument('--fem-h', type=float, default=.0015)
    a.add_argument('--skin', choices=('fan', 'boundary'), default='fan', help='fan: sphere_points.npz; boundary: --skin-file')
    a.add_argument('--skin-file', default='boundary_skin.npz')
    a.add_argument('--lever', choices=('point', 'c'), default='point')
    a.add_argument('--gamma', choices=('bonds', 'sym_skin'), default='bonds')
    args = a.parse_args()
    stt = torch.load(args.part/'run'/'model.pt', map_location='cpu', weights_only=True)
    p = stt['points'].double().numpy(); r = torch.nn.functional.softplus(stt['radii'].double(), beta=100).numpy()
    c_ = np.load(args.part/('fem_modes_h%g.npz' % args.fem_h))
    f_fem, V_fem = c_['f'][:20], c_['V'][:, :20]
    F_ = dict(nodes=c_['nodes'], node_grid=c_['node_grid'], lo=c_['lo'], h=float(c_['h']), masses=c_['masses'])
    E, nu = 23008.38, .3008
    g = dict(np.load(args.part/'dem_exact_single'/'geometry.npz'))
    skin = dict(np.load(args.part/'sphere_skin.npz')); pts = dict(np.load(args.part/'sphere_points.npz'))
    m = build(g, E, nu, dev, 'noslip', .25, .3, f64, kinematics='exact', nodalvol=True, nonlinear=False, ogden_alpha=2.)
    bsk = dict(np.load(args.part/args.skin_file)) if args.skin == 'boundary' else None
    st = SphereTerms(m, g, p, r, skin, pts) if bsk is None else BoundaryTerms(m, g, p, r, bsk, lever=args.lever, gamma=args.gamma)
    n, npt = st.n, st.npt; n3, v3 = 3*n, 3*npt
    th = torch.zeros(n, 9, dtype=f64, device=dev)
    X, Y = st.X, st.Y
    mid = MidpointTerms(m, PatchSprings(g, r, skin, float(m.mu_ref), st.V), g)
    KA = dense_hessian(lambda z: mid.energy(th, X+z.view(-1, 3), smooth_polar=True)[0], n3)
    H = dense_hessian(lambda z: st.energy_xy(th, X+z[:n3].view(-1, 3), Y+z[n3:].view(-1, 3)), n3+v3)
    Kxx, Kxy, Kyy = H[:n3, :n3], H[:n3, n3:], H[n3:, n3:]
    Gy = torch.linalg.solve(Kyy, Kxy.T)
    KC = Kxx-Kxy@Gy; KC = .5*(KC+KC.T)
    Ux, _ = interpolate(F_, V_fem, X.cpu().numpy())
    Uy_old, n_old = interpolate(F_, V_fem, Y.cpu().numpy())
    Uy, outside = interpolate_extrapolated(F_, V_fem, Y.cpu().numpy())
    chg = np.linalg.norm(Uy-Uy_old, axis=1)[outside]/np.maximum(np.linalg.norm(Uy, axis=1)[outside], 1e-30)
    print('skin points outside the voxel mesh: %d (nearest node before) / %d (extrapolated now); their displacement changed by a median %.2f, 90 %% %.2f of |u|' % (
        n_old, int(outside.sum()), np.median(chg), np.quantile(chg, .9)), flush=True)
    if bsk is None:
        nc = int(pts['n_corners']); na = int(pts['arc_off'][-1])
        kind = np.zeros(npt, int); kind[nc:nc+na] = 1; kind[nc+na:] = 2
    else:
        kind = bsk['kind']                                               # corners, arc midpoints; no part points
    names = ('corners', 'arc points', 'part points')
    mass = m.mass.to(f64)
    # 3. consistent mass
    d, J, cnt = cell_moments(p, r, X.cpu().numpy(), st.V.cpu().numpy())
    dT, JT = torch.as_tensor(d, device=dev), torch.as_tensor(J, device=dev)
    I = torch.eye(3, dtype=f64, device=dev)
    def Q(u, Gr):
        return float((mass*(u*u).sum(1)).sum()+2*(u*torch.einsum('bij,bj->bi', Gr, dT)).sum()+torch.einsum('bij,bjk,bik->', Gr, JT, Gr))
    def mass_matrix(gradfn):
        def T(v):
            u = v.view(-1, 3); Gr = gradfn(u)
            return .5*((mass*(u*u).sum(1)).sum()+2*(u*torch.einsum('bij,bj->bi', Gr, dT)).sum()+torch.einsum('bij,bjk,bik->', Gr, JT, Gr))
        return dense_hessian(T, n3)
    gradA = lambda u: mid.F(X+u)-I
    gradC = lambda u: st.F(X+u, Y-(Gy@u.reshape(-1)).view(-1, 3))-I
    print('cell sampling: points per cell min %d median %d' % (cnt.min(), np.median(cnt)), flush=True)
    print('mode | stiffness: B old interp, B extrapolated, C | energy B - C split corners / arcs / parts | |dy| / |y_B| median per kind | mass ratio: lumped, consistent A, consistent C')
    rows = []
    for k in range(len(f_fem)):
        om2 = (2*np.pi*f_fem[k])**2
        px = torch.as_tensor(Ux[:, :, k].ravel(), dtype=f64, device=dev)
        pyo = torch.as_tensor(Uy_old[:, :, k].ravel(), dtype=f64, device=dev); py = torch.as_tensor(Uy[:, :, k].ravel(), dtype=f64, device=dev)
        sBo = float(torch.cat((px, pyo))@H@torch.cat((px, pyo)))/om2; sB = float(torch.cat((px, py))@H@torch.cat((px, py)))/om2
        sC = float(px@KC@px)/om2
        yC = -(Gy@px); dy = py-yC
        Kd = Kyy@dy; tot = float(dy@Kd)
        split = [float((dy.view(-1, 3)[torch.as_tensor(kind == t, device=dev)]*Kd.view(-1, 3)[torch.as_tensor(kind == t, device=dev)]).sum())/tot for t in range(3)]
        rel = [float(np.median(np.linalg.norm(dy.view(-1, 3).cpu().numpy()[kind == t], axis=1)/np.maximum(np.linalg.norm(Uy[:, :, k][kind == t], axis=1), 1e-30))) if (kind == t).any() else 0. for t in range(3)]
        u = px.view(-1, 3)
        mr = float((mass*(u*u).sum(1)).sum()); mA = Q(u, gradA(u)); mC = Q(u, gradC(u))
        rows.append((sBo, sB, sC, split, rel, mr, mA, mC))
        print('%2d | %.3f %.3f %.3f | %.2f %.2f %.2f | %.2f %.2f %.2f | %.3f %.3f %.3f' % (k, sBo, sB, sC, *split, *rel, mr, mA, mC), flush=True)
    R = np.array([[x[0], x[1], x[2], *x[3], *x[4], x[5], x[6], x[7]] for x in rows])
    print('median: stiffness B old %.4f, B extrapolated %.4f, C %.4f | energy B - C split %.2f / %.2f / %.2f | |dy|/|y_B| %.2f / %.2f / %.2f | mass lumped %.4f, consistent A %.4f, C %.4f' % tuple(np.median(R, 0)), flush=True)
    W = mass.cpu().numpy()
    for tag, K, gf in (('A', KA, gradA), ('C', KC, gradC)):
        Mc = mass_matrix(gf); Mc = .5*(Mc+Mc.T)
        Lc = torch.linalg.cholesky(Mc)
        Li = torch.linalg.inv(Lc)
        w, Vv = torch.linalg.eigh(Li@K@Li.T)
        Z = (Li.T@Vv).cpu().numpy(); w = w.cpu().numpy()
        fz = np.sqrt(np.clip(w[6:6+26], 0, None))/(2*np.pi); Z = Z[:, 6:6+26].reshape(-1, 3, 26)
        pair = mac_pairs(Z, Ux, W)
        good = [(k, fz[pair[k][0]]/f_fem[k], pair[k][1]) for k in range(len(f_fem)) if pair[k][1] >= .8]
        print('own modes %s with consistent mass: rigid %s | paired (MAC >= 0.8) %d of %d, frequency / FEM median %.4f (%.4f..%.4f); per mode %s' % (
            tag, np.array2string(np.abs(w[:6])/w[-1], precision=1), len(good), len(f_fem), np.median([x[1] for x in good]), min(x[1] for x in good), max(x[1] for x in good),
            ' '.join('%d:%.3f(%.2f)' % (k, fz[pair[k][0]]/f_fem[k], pair[k][1]) for k in range(len(f_fem)))), flush=True)


if __name__ == '__main__':
    main()
