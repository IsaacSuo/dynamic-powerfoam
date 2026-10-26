"""Springback of a plastically bent part (2026-10-26, plastic.Plasticity, the bonds' plastic rotation thp): the
compatible isochoric bending field of plastic_bend_check.py (eps_p = kappa y diag(1, -1/2, -1/2), u = kappa (x y, -y^2
/ 4 - x^2 / 2 + z^2 / 4, -y z / 2)) set as every cell's plastic strain, the body started at X + u(X) and let go: the
static equilibrium (JointSolver's solve with a weak proximal inertia about the last iterate, repeated until it no
longer moves; no gravity, no contact), then the bending left, kappa' from the least-squares fit of x - X to kappa' u_1
+ a rigid motion. kappa' / kappa = 1: no springback. On the plastic references with thp, without, and the elastic
ones (all back: kappa' ~ 0)"""
import math

import numpy as np
import torch

from free_surface.joint_step import JointSolver
from free_surface.plastic import Plasticity
from free_surface.skin_contact_check import f64, dev
from free_surface.skin_dynamics_check import build, parser


def main():
    args = parser().parse_args(['--case', 'part', '--integrator', 'midpoint'])
    b = build(args)
    st, W, S, mass = b['st'], b['W'], b['S'], b['mass']
    X, Y = st.X, st.Y
    n = W.n
    xc = X.mean(0)
    pl = Plasticity(W, st, math.inf)
    D = torch.diag(torch.tensor([1., -.5, -.5], dtype=f64, device=dev))

    def field(P, kap):
        x_, y_, z_ = (P-xc).unbind(1)
        return torch.stack((kap*x_*y_, -kap*y_**2/4-kap*x_**2/2+kap*z_**2/4, -kap*y_*z_/2), 1)
    ymax = float((X[:, 1]-xc[1]).abs().max())
    kap = .02/ymax
    eps = kap*(X[:, 1]-xc[1])[:, None, None]*D
    lowest = 2*math.pi*6.61                                                         # (the part's lowest mode, rad/s)
    dt = 4/lowest                                                                   # c_in = 4 / dt^2 = lowest^2 / 4: each
    J = JointSolver(S, W, mass, b['kblocks'], b['vblocks'], dt, 4/dt**2, 2.)        # proximal pass cuts the slowest mode's
    zero = torch.zeros_like(X)                                                      # error to ~1/5
    U1 = field(X, 1.)
    c = X-xc
    A = torch.cat((U1.reshape(-1, 1), torch.cat([torch.cross(torch.eye(3, dtype=f64, device=dev)[k].expand_as(c), c, dim=1).reshape(-1, 1) for k in range(3)], 1),
                   torch.eye(3, dtype=f64, device=dev).repeat(n, 1)), 1)              # [u_1, omega x c, t]
    pl.apply(torch.zeros_like(eps))
    x0, y0 = X+field(X, kap), Y+field(Y, kap)
    for _ in range(3):
        y0 = S.affine_skin(x0, y0)
    _, gx, gy = W.grad(x0, y0)
    ref = math.sqrt(float((gx**2).sum())+float((gy**2).sum()))                    # (the bending's force scale)
    for name, rot, grd, eps_ in (('elastic references', True, True, torch.zeros_like(eps)), ('plastic, first order only', False, False, eps),
                                 ('plastic, thp', True, False, eps), ('plastic, thp + branch gradient', True, True, eps)):
        pl.rotation, pl.gradient = rot, grd
        pl.apply(eps_)
        z = torch.cat((x0, y0))
        prods = 0
        for it in range(1, 200):
            x = z[:n]
            J.set_step(x, x, zero)
            z1, s_ = J.solve(None, z, b['prec'], ref, tol=1e-10, maxit=3000)
            prods += s_['products']
            move = float((z1[:n]-z[:n]).norm(dim=1).max())
            z = z1
            if move <= 1e-10:
                break
        x, y = z[:n], z[n:]
        sol = torch.linalg.lstsq(A, (x-X).reshape(-1, 1)).solution[:, 0]
        res = float(((A@sol).view(n, 3)-(x-X)).norm(dim=1).max())
        e = pl.elastic(x, y)
        _, gx, gy = W.grad(x, y)
        print('%s: %d proximal passes, %d products, last move %.1e m, |grad| / ref %.1e; kappa\' / kappa %.4f (fit residual max %.2e m, '
              'u max %.2e m); energy left %.3e' % (name, it, prods, move, math.sqrt(float((gx**2).sum())+float((gy**2).sum()))/ref,
                                                    float(sol[0])/kap, res, float(field(X, kap).norm(dim=1).max()), e), flush=True)


if __name__ == '__main__':
    main()
