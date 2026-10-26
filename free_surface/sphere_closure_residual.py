"""Residual forces of the skinless sphere model (sphere_closure.py) on the part's FEM modes (2026-10-26), the test the
earlier skinless version A failed (lattice beam, exact bending: residual 15-34 %, growing with refinement; skin model
3-12 %). A model that reproduces the continuum has K u = omega^2 M u for an FEM mode u (interpolated at the bodies), so
per mode f = K u - omega^2 M u against the elastic forces K u, as ||f|| / ||K u|| over the bodies with an exposed sphere and
over the inner bodies; consistent mass (part_diag's affine cells) and the masses at the bodies. Both models, exact
stiffness at rest: new K = [I; G]^T H [I; G] (sphere_closure_modes.closure_stiffness); skin model (BoundaryTerms, lever c,
Gamma sym_skin) condensed over its relaxed skin, K_xx - K_xy K_yy^-1 K_yx, its cells' gradient with the skin following.
  PYTHONPATH=<main repo>:<worktree> python free_surface/sphere_closure_residual.py"""
import argparse
from pathlib import Path

import numpy as np
import torch

from experiments.dem_fem_mode_match import interpolate
from experiments.pacnerf_dem_forward import build
from free_surface.boundary_skin import BoundaryTerms
from free_surface.part_diag import cell_moments
from free_surface.part_modes import dense_hessian
from free_surface.sphere_closure import ClosureTerms
from free_surface.sphere_closure_modes import closure_stiffness, consistent_mass

f64, dev = torch.float64, 'cuda'


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument('--part', type=Path, default=Path('output/test_part'))
    a_.add_argument('--fem-h', type=float, default=.0015)
    args = a_.parse_args()
    stt = torch.load(args.part/'run'/'model.pt', map_location='cpu', weights_only=True)
    p = stt['points'].double().numpy(); r = torch.nn.functional.softplus(stt['radii'].double(), beta=100).numpy()
    c_ = np.load(args.part/('fem_modes_h%g.npz' % args.fem_h))
    f_fem, V_fem = c_['f'][:20], c_['V'][:, :20]
    F_ = dict(nodes=c_['nodes'], node_grid=c_['node_grid'], lo=c_['lo'], h=float(c_['h']), masses=c_['masses'])
    g = dict(np.load(args.part/'dem_exact_single'/'geometry.npz'))
    m = build(g, 23008.38, .3008, dev, 'noslip', .25, .3, f64, kinematics='exact', nodalvol=True, nonlinear=False, ogden_alpha=2.)
    bsk = dict(np.load(args.part/'boundary_skin.npz'))
    new = ClosureTerms(m, g, p, r, bsk)
    old = BoundaryTerms(m, g, p, r, bsk, lever='c', gamma='sym_skin')
    n = new.n; n3 = 3*n
    th = torch.zeros(n, 9, dtype=f64, device=dev)
    X, Y = new.X, old.Y
    I3 = torch.eye(3, dtype=f64, device=dev)
    mass = m.mass.to(f64)
    d, Jc, _ = cell_moments(p, r, X.cpu().numpy(), new.V.cpu().numpy())
    dT, JT = torch.as_tensor(d, device=dev), torch.as_tensor(Jc, device=dev)
    Ml = torch.diag(mass.repeat_interleave(3))
    Ux = torch.as_tensor(interpolate(F_, V_fem, X.cpu().numpy())[0], dtype=f64, device=dev)       # (n, 3, 20)
    cap = new.cap

    Kn, G = closure_stiffness(new, th)
    Mn = consistent_mass(lambda v: (G@v).view(n, 3, 3), mass, dT, JT)
    del G; torch.cuda.empty_cache()
    H = dense_hessian(lambda z: old.energy_xy(th, X+z[:n3].view(-1, 3), Y+z[n3:].view(-1, 3)), n3+3*old.npt)
    Kxx, Kxy, Kyy = H[:n3, :n3], H[:n3, n3:], H[n3:, n3:]
    Gy = torch.linalg.solve(Kyy, Kxy.T)
    Ko = Kxx-Kxy@Gy; Ko = .5*(Ko+Ko.T)
    del H, Kxx, Kxy, Kyy; torch.cuda.empty_cache()
    Mo = consistent_mass(lambda v: old.F(X+v.view(-1, 3), Y-(Gy@v).view(-1, 3))-I3, mass, dT, JT)
    print('bodies with an exposed sphere %d, inner %d; ||f|| / ||K u|| per mode (exposed / inner), median over the 20 modes and range' % (
        int(cap.sum()), int((~cap).sum())), flush=True)
    for tag, K, M in (('new, consistent mass', Kn, Mn), ('skin model, consistent mass', Ko, Mo),
                      ('new, masses at the bodies', Kn, Ml), ('skin model, masses at the bodies', Ko, Ml)):
        rows = []
        for k in range(len(f_fem)):
            u = Ux[:, :, k].reshape(-1)
            Ku = (K@u).view(n, 3); f = Ku-(2*np.pi*f_fem[k])**2*(M@u).view(n, 3)
            rows.append([float(f[msk].norm()/Ku[msk].norm()) for msk in (cap, ~cap)])
        R = np.array(rows)
        print('%-33s exposed %.3f (%.3f..%.3f), inner %.3f (%.3f..%.3f); per mode %s' % (
            tag, np.median(R[:, 0]), R[:, 0].min(), R[:, 0].max(), np.median(R[:, 1]), R[:, 1].min(), R[:, 1].max(),
            ' '.join('%d:%.2f/%.2f' % (k, a, b) for k, (a, b) in enumerate(R))), flush=True)


if __name__ == '__main__':
    main()
