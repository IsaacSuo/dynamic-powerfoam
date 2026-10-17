"""Cosserat body law (COSSERAT_FOAM.md, EnergyFoam(cosserat=True)): zero-energy modes of a free block, and the
cantilever plates of plate_bending_check against the Hex8 FEM. Stiffness by autograd Hessian of the energy at rest
(positions and world rotation vectors), so no closed-form forces are needed for these static checks."""
import argparse
import numpy as np
import scipy.sparse.linalg as spla
import torch

from experiments.foam_dem_energy import EnergyFoam, quat_mul, quat_small
from experiments.plate_bending_check import lattice_plate, fem_tip_deflection, RHO


def make(geometry, E, nu, cosserat=True, residual='normal', enhanced=True, thin_ratio=.1, bond_level=False, clean=False):
    lam, mu = E*nu/((1+nu)*(1-2*nu)), E/(2*(1+nu))
    if clean:                    # CLEAN_FOAM.md: translations only, bond-level two-end Neo-Hookean
        m = EnergyFoam(geometry, k_v=lam/RHO, volume=True, device='cpu', young=mu/RHO, shear=mu/RHO, body_law='neohookean', k_mu=mu/RHO,
                       dtype=torch.float64, kinematics='exact', residual=False, enhanced=False, thin_ratio=thin_ratio, cosserat=True,
                       bond_level=True, micropolar=False)
        m.bond_level_mid = True
        return m
    if cosserat:
        return EnergyFoam(geometry, k_v=lam/RHO, volume=True, device='cpu', young=mu/RHO, shear=mu/RHO, body_law='neohookean', k_mu=mu/RHO,
                          dtype=torch.float64, kinematics='exact', residual=False if bond_level else residual, enhanced=False if bond_level else enhanced,
                          thin_ratio=thin_ratio, cosserat=True, bond_level=bond_level)
    return EnergyFoam(geometry, k_v=lam/RHO, volume=True, device='cpu', young=.5*mu/RHO, shear=.5*mu/RHO, body_law='neohookean', k_mu=.75*mu/RHO,
                      dtype=torch.float64, kinematics='exact', residual=residual, enhanced=enhanced, thin_ratio=thin_ratio, thin_bend_factor=1/(.25*(1-nu)))


def hessian(model):
    x0, q0, _, _ = model.initial_state()
    th = torch.zeros(model.n, 7, dtype=torch.float64)
    n = model.n

    dof = 3 if not getattr(model, 'micropolar', True) else 6

    def energy(z):
        z = z.reshape(n, dof)
        return model.energy(th, x0+z[:, :3], quat_mul(quat_small(z[:, 3:]), q0) if dof == 6 else q0)
    H = torch.autograd.functional.hessian(energy, torch.zeros(dof*n, dtype=torch.float64), vectorize=True)
    assert torch.isfinite(H).all(), 'non-finite Hessian entries: %d' % int((~torch.isfinite(H)).sum())
    return .5*(H+H.T)


def zero_modes(nx=4, ny=4, nz=4, nu=.414, E=24107.):
    g, idx = lattice_plate(nx, ny, nz, .02); g['fixed'][:] = False
    for name, kw in (('cosserat, res normal', dict(residual='normal')), ('cosserat, res off', dict(residual=False)), ('cosserat, res off, no EAS', dict(residual=False, enhanced=False)),
                     ('cosserat, bond-level F', dict(bond_level=True))):
        m = make(g, E, nu, **kw)
        H = hessian(m).numpy()
        w = np.linalg.eigvalsh(H); w = w/w.max()
        print('%d x %d x %d free block, %-28s: zero modes beyond the 6 rigid: %d | eigenvalues 6..12: %s' % (nx, ny, nz, name, int((w < 1e-9).sum())-6, np.array2string(w[6:12], precision=2)))


def plates(layers, nu=.414, E=24107., nx=16, ny=6, a=.02, **kw):
    for nz in layers:
        g, idx = lattice_plate(nx, ny, nz, a)
        fem = fem_tip_deflection(nx, ny, nz, a, 3, E, nu)
        m = make(g, E, nu, **kw)
        H = hessian(m).numpy(); n = m.n
        dof = H.shape[0]//n
        free = np.repeat(~g['fixed'], dof); tip = idx[:, 0] == idx[:, 0].max()
        f = np.zeros(dof*n); f[dof*np.flatnonzero(tip)+2] = 1./tip.sum()
        u = np.zeros(dof*n); u[free] = np.linalg.solve(H[free][:, free], f[free])
        d = float(u[dof*np.flatnonzero(tip)+2].mean())
        print('plate %d layer(s), nu %.3f, %s: FEM %.4g | Cosserat %.4g (%.2f x FEM) | thin bodies %d' % (nz, nu, kw, fem, d, d/fem, int(m.thin.sum())))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--what', default='all', choices=('all', 'zero', 'plates'))
    p.add_argument('--nu', type=float, default=.414)
    p.add_argument('--layers', type=int, nargs='+', default=[1, 2, 3, 4])
    args = p.parse_args()
    if args.what in ('all', 'zero'):
        zero_modes(nu=args.nu)
    if args.what in ('all', 'plates'):
        plates(args.layers, nu=args.nu)
