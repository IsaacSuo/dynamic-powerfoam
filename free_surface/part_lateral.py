"""Is the free skin too soft because the outer layer has no lateral stiffness? (2026-10-22, part_diag.py follow-up)
The skin deviation delta (skin minus the cell's affine field) enters the model only as (mu / r) int |delta|^2 dA per
patch: fibres from the site, each on its own. A deviation that varies over the patch shears neighbouring fibres against
each other, which no term sees. Estimate: in the cone from the site to the patch, the deviation grown linearly along the
radius, u = (rho / r) delta(s); its energy mu int |grad u|^2 dV = (mu / 3r) int |delta|^2 dA (along the radius) +
(mu r / 3) int |grad_s delta|^2 dA (across: the missing lateral part). With the bodies on each FEM mode, for the skin on
the FEM surface (B, extrapolated) and relaxed (C): the model's term, the lateral estimate, against the energy B - C.
All as fractions of the FEM mode energy omega^2 / 2. The linear growth is an assumed shape (an upper bound on the cone's
energy for that surface deviation, walls ignored)."""
import argparse
from pathlib import Path

import numpy as np
import torch

from experiments.dem_fem_mode_match import interpolate
from experiments.pacnerf_dem_forward import build
from free_surface.part_diag import interpolate_extrapolated
from free_surface.part_modes import dense_hessian
from free_surface.sphere_model import SphereTerms

f64, dev = torch.float64, 'cuda'


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--part', type=Path, default=Path('output/test_part'))
    a.add_argument('--fem-h', type=float, default=.0015)
    args = a.parse_args()
    stt = torch.load(args.part/'run'/'model.pt', map_location='cpu', weights_only=True)
    p = stt['points'].double().numpy(); r = torch.nn.functional.softplus(stt['radii'].double(), beta=100).numpy()
    c_ = np.load(args.part/('fem_modes_h%g.npz' % args.fem_h))
    f_fem, V_fem = c_['f'][:20], c_['V'][:, :20]
    F_ = dict(nodes=c_['nodes'], node_grid=c_['node_grid'], lo=c_['lo'], h=float(c_['h']), masses=c_['masses'])
    g = dict(np.load(args.part/'dem_exact_single'/'geometry.npz'))
    skin = dict(np.load(args.part/'sphere_skin.npz')); pts = dict(np.load(args.part/'sphere_points.npz'))
    m = build(g, 23008.38, .3008, dev, 'noslip', .25, .3, f64, kinematics='exact', nodalvol=True, nonlinear=False, ogden_alpha=2.)
    st = SphereTerms(m, g, p, r, skin, pts)
    n, npt = st.n, st.npt; n3, v3 = 3*n, 3*npt
    th = torch.zeros(n, 9, dtype=f64, device=dev)
    X, Y = st.X, st.Y
    H = dense_hessian(lambda z: st.energy_xy(th, X+z[:n3].view(-1, 3), Y+z[n3:].view(-1, 3)), n3+v3)
    Gy = torch.linalg.solve(H[n3:, n3:], H[n3:, :n3])
    Ux, _ = interpolate(F_, V_fem, X.cpu().numpy())
    Uy, _ = interpolate_extrapolated(F_, V_fem, Y.cpu().numpy())
    mu = st.mu
    def parts_energy(x, y):
        F = st.F(x, y)
        d = y[st.e_pid]-x[st.e_body]-torch.einsum('eij,ej->ei', F[st.e_body], st.e_d)
        da, db = d[st.pr_a], d[st.pr_b]; r_p = st.p_r[st.pr_part]; dot = (da*db).sum(1)
        return float((mu/r_p*st.pr_m*dot).sum()), float((mu*r_p/3*st.pr_k*dot).sum())
    print('mode | stiffness B, C, gap B - C | model term (mu / r) int |delta|^2: B, C | lateral estimate (mu r / 3) int |grad delta|^2: B, C | all / (omega^2 / 2)')
    rows = []
    for k in range(len(f_fem)):
        e2 = .5*(2*np.pi*f_fem[k])**2
        px = torch.as_tensor(Ux[:, :, k].ravel(), dtype=f64, device=dev)
        pyB = torch.as_tensor(Uy[:, :, k].ravel(), dtype=f64, device=dev); pyC = -(Gy@px)
        sB = float(torch.cat((px, pyB))@H@torch.cat((px, pyB)))/(2*e2); sC = float(torch.cat((px, pyC))@H@torch.cat((px, pyC)))/(2*e2)
        x = X+px.view(-1, 3)
        ddB, latB = parts_energy(x, Y+pyB.view(-1, 3)); ddC, latC = parts_energy(x, Y+pyC.view(-1, 3))
        rows.append((sB, sC, sB-sC, ddB/e2, ddC/e2, latB/e2, latC/e2))
        print('%2d | %.3f %.3f %.3f | %.4f %.4f | %.4f %.4f' % (k, *rows[-1]), flush=True)
    R = np.array(rows)
    print('median: stiffness B %.4f C %.4f gap %.4f | model term B %.4f C %.4f | lateral estimate B %.4f C %.4f' % tuple(np.median(R, 0)), flush=True)


if __name__ == '__main__':
    main()
