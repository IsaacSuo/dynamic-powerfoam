"""Where does the DEM's softness come from? Impose a small homogeneous strain on the burger DEM and
let parts of it relax (static energy minimisation, float64):
  (a) affine: every body follows the strain, no rotation           -> continuum value (calibration)
  (b) + body rotations free (positions affine)                     -> rotation relaxation
  (c) + interior positions free too (boundary bodies held affine)  -> full relaxation
Effective stiffness = relaxed energy / continuum energy. Minimisation by L-BFGS on the free DOFs
(rotation vectors as left perturbations of the identity; the energy is exact, so gradients are
autograd of EnergyFoam.energy).
"""
import numpy as np
import torch

from experiments.foam_dem_energy import quat_mul, quat_small
from experiments.pacnerf_dem_forward import build, RHO


def main():
    g = dict(np.load('output/springgaus_dem/burger_spacing8k_mf0.1/geometry.npz'))
    E, nu = 2e4, .3
    lam, mu = E*nu/((1+nu)*(1-2*nu)), E/(2*(1+nu))
    deg = np.bincount(g['pairs'].ravel(), minlength=len(g['masses']))
    boundary = torch.tensor(deg < np.median(deg)-2)       # crude: cells with fewer material neighbours (surface)
    for kin in ('small', 'exact'):
        m = build(g, E, nu, 'cpu', 'noslip', .25, 0., torch.float64, kinematics=kin)
        X, V = m.X, float(m.cell_volume.sum())
        th = torch.zeros(m.n, 7, dtype=torch.float64)
        q0 = torch.zeros(m.n, 4, dtype=torch.float64); q0[:, 0] = 1
        c = X.mean(0)
        eps = 1e-3
        for name, H in (('uniaxial strain zz', torch.diag(torch.tensor([0., 0., -eps], dtype=torch.float64))),
                        ('pure shear xz', torch.tensor([[0, 0, eps], [0, 0, 0], [eps, 0, 0]], dtype=torch.float64)/2)):
            S = .5*(H+H.T)
            ref = .5*(lam*torch.trace(S)**2+2*mu*(S*S).sum())*V/RHO
            xa = X+(X-c)@H.T
            Ea = float(m.energy(th, xa, q0))
            # (b) rotations free
            r = torch.zeros(m.n, 3, dtype=torch.float64, requires_grad=True)
            opt = torch.optim.LBFGS([r], max_iter=400, line_search_fn='strong_wolfe', tolerance_grad=1e-20, tolerance_change=1e-30)
            def cl_b():
                opt.zero_grad(); e = m.energy(th, xa, quat_mul(quat_small(r), q0)); e.backward(); return e
            opt.step(cl_b)
            Eb = float(m.energy(th, xa, quat_mul(quat_small(r.detach()), q0)))
            # (c) rotations + interior positions free
            r2 = r.detach().clone().requires_grad_(True)
            du = torch.zeros(m.n, 3, dtype=torch.float64, requires_grad=True)
            free = (~boundary).double()[:, None]
            opt = torch.optim.LBFGS([r2, du], max_iter=1500, line_search_fn='strong_wolfe', tolerance_grad=1e-20, tolerance_change=1e-30)
            def cl_c():
                opt.zero_grad(); e = m.energy(th, xa+free*du, quat_mul(quat_small(r2), q0)); e.backward(); return e
            for _ in range(3):
                opt.step(cl_c)
            Ec = float(m.energy(th, xa+free*du.detach(), quat_mul(quat_small(r2.detach()), q0)))
            print('[%s] %-18s energy / continuum: affine %.3f | + rotations relax %.3f | + interior positions relax %.3f   (boundary bodies held: %d of %d)' % (
                kin, name, Ea/ref, Eb/ref, Ec/ref, int(boundary.sum()), m.n), flush=True)


if __name__ == '__main__':
    main()
