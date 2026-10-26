"""Which time stepping for the skinless sphere model (sphere_closure.py)? Linear analysis on the part (2026-10-26): the
model's exact stiffness at rest K (sphere_closure_modes.closure_stiffness), the masses at the bodies M, so every scheme
is a linear map of the state (x, v) and its accuracy and cost come out without running any dynamics.
  each step minimises Pi(z) = c / (2 h^2) |z - x~|_M^2 + 1/2 z^T K z
    backward Euler  c = 1, x~ = x + h v;      x' = z, v' = (z - x) / h
    midpoint        c = 4, x~ = x + h v / 2;  x' = 2 z - x, v' = 4 (z - x) / h - v   (the current integrator)
  solved exactly, or from z = x~ by s sweeps preconditioned with the bodies' own 3 x 3 blocks of A = c M / h^2 + K
  (P = c m / h^2 + K_bb, the 'local implicit' update): damped Jacobi (omega = 2 / (lam_min + lam_max) of P^-1 A) or
  Chebyshev on [lam_min, lam_max]; one gradient per sweep.
Per scheme: the growth per step (power iteration, 2000 steps: above 1 = unstable); for the model's own 20 lowest modes
(eigen of K with M) the step restricted to the mode, frequency / exact and the amplitude kept per period; the cost in
gradients per simulated second. The exact solves' cost: block-Jacobi PCG iterations per step at the current h = 1/1200
to 1e-3 and 1e-6, for this model and for the skin model (BoundaryTerms, lever c, Gamma sym_skin: massless skin, a 3 x 3
block per body and per skin point), with the condition numbers of P^-1 A.
  PYTHONPATH=<main repo>:<worktree> python free_surface/sphere_closure_steps.py"""
import argparse
import math
from pathlib import Path

import numpy as np
import torch

from experiments.pacnerf_dem_forward import build
from free_surface.boundary_skin import BoundaryTerms
from free_surface.part_modes import dense_hessian
from free_surface.sphere_closure import ClosureTerms
from free_surface.sphere_closure_modes import closure_stiffness

f64, dev = torch.float64, 'cuda'


def blocks(K, nb):
    """the 3 x 3 diagonal blocks of K (3 nb, 3 nb)"""
    a = torch.arange(nb, device=K.device)
    return K.view(nb, 3, nb, 3)[a, :, a, :]


def bounds(A, Pb):
    """lam_min, lam_max of P^-1 A, P the block diagonal Pb (nb, 3, 3)"""
    w, U = torch.linalg.eigh(Pb)
    Ph = U@torch.diag_embed(w**-.5)@U.transpose(1, 2)                              # P^-1/2 per block
    nb = len(Pb)
    Pf = torch.zeros(3*nb, 3*nb, dtype=A.dtype, device=A.device)
    a = torch.arange(nb, device=A.device)
    Pf.view(nb, 3, nb, 3)[a, :, a, :] = Ph
    lam = torch.linalg.eigvalsh(Pf@A@Pf)
    return float(lam[0]), float(lam[-1])


def pcg(A, Pinv, b, z, tol, maxit=5000):
    r = b-A@z; r0 = float(r.norm()); d = Pinv(r); rz = float(r@d); p_ = d.clone()
    for it in range(1, maxit+1):
        Ap = A@p_; al = rz/float(p_@Ap)
        z = z+al*p_; r = r-al*Ap
        if float(r.norm()) < tol*r0:
            return it
        d = Pinv(r); rz_new = float(r@d); p_ = d+rz_new/rz*p_; rz = rz_new
    return maxit


class Scheme:
    cache = {}

    def __init__(self, K, Md, base, h, method, s):
        self.K, self.Md, self.h, self.method, self.s = K, Md, h, method, s
        self.c = 1. if base == 'BE' else 4.
        self.base = base
        if (base, h) not in Scheme.cache:                                           # A, P^-1, the bounds, A's LU per (base, h)
            nb = len(Md)//3
            A = self.c*torch.diag(Md)/h**2+K
            Pb = blocks(K, nb)+(self.c*Md.view(nb, 3)/h**2)[:, :, None]*torch.eye(3, dtype=f64, device=dev)
            Scheme.cache[(base, h)] = (A, torch.linalg.inv(Pb), *bounds(A, Pb), torch.linalg.lu_factor(A))
        self.A, self.Pi, self.lmin, self.lmax, self.LU = Scheme.cache[(base, h)]

    def P(self, R):
        nb = len(self.Pi)
        return torch.einsum('bij,bjc->bic', self.Pi, R.view(nb, 3, -1)).reshape(R.shape)

    def step(self, X, V):
        """X, V (3n, k): k states at once"""
        h, A = self.h, self.A
        xt = X+(h if self.base == 'BE' else h/2)*V
        B = self.c*self.Md[:, None]/h**2*xt
        if self.method == 'exact':
            Z = torch.linalg.lu_solve(*self.LU, B)
        else:
            Z = xt.clone(); R = B-A@Z
            if self.method == 'jacobi':
                om = 2/(self.lmin+self.lmax)
                for k in range(self.s):
                    Z = Z+om*self.P(R)
                    if k < self.s-1:
                        R = B-A@Z
            else:                                                                   # Chebyshev (Saad, Alg. 12.1)
                th = .5*(self.lmax+self.lmin); de = .5*(self.lmax-self.lmin); s1 = th/de
                D = self.P(R)/th; rho = 1/s1
                for k in range(self.s):
                    Z = Z+D
                    if k == self.s-1:
                        break
                    R = R-A@D; rn = 1/(2*s1-rho)
                    D = rn*rho*D+2*rn/de*self.P(R); rho = rn
        if self.base == 'BE':
            return Z, (Z-X)/h
        return 2*Z-X, 4*(Z-X)/h-V


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument('--part', type=Path, default=Path('output/test_part'))
    a_.add_argument('--hs', type=float, nargs='*', default=[4e-4, 2e-4, 1e-4, 5e-5])
    a_.add_argument('--sweeps', type=int, nargs='*', default=[1, 2, 3, 5])
    a_.add_argument('--skin-model', action=argparse.BooleanOptionalAction, default=True)
    args = a_.parse_args()
    torch.manual_seed(0)
    stt = torch.load(args.part/'run'/'model.pt', map_location='cpu', weights_only=True)
    p = stt['points'].double().numpy(); r = torch.nn.functional.softplus(stt['radii'].double(), beta=100).numpy()
    g = dict(np.load(args.part/'dem_exact_single'/'geometry.npz'))
    m = build(g, 23008.38, .3008, dev, 'noslip', .25, .3, f64, kinematics='exact', nodalvol=True, nonlinear=False, ogden_alpha=2.)
    bsk = dict(np.load(args.part/'boundary_skin.npz'))
    new = ClosureTerms(m, g, p, r, bsk)
    n = new.n; n3 = 3*n
    th = torch.zeros(n, 9, dtype=f64, device=dev)
    K, _ = closure_stiffness(new, th)
    mass = m.mass.to(f64); Md = mass.repeat_interleave(3)
    Mi = Md**-.5
    w, U = torch.linalg.eigh(Mi[:, None]*K*Mi[None, :])
    om = torch.sqrt(w[6:26].clamp(min=0)); Phi = Mi[:, None]*U[:, 6:26]                 # M-orthonormal
    print('model (masses at the bodies): omega of the 20 lowest modes %.1f..%.1f rad/s, highest %.3e rad/s (explicit limit 2 / omega %.2e s); '
          'the bodies\' own stiffness / mass: largest sqrt %.3e rad/s' % (
              float(om[0]), float(om[-1]), float(w[-1].sqrt()), 2/float(w[-1].sqrt()),
              float((torch.linalg.eigvalsh(blocks(K, n))[:, -1]/mass).max().sqrt())), flush=True)

    # exact solves at the current step: PCG with the bodies' blocks
    hc = 1/1200
    sch = Scheme(K, Md, 'mid', hc, 'exact', 0)
    a = torch.randn(20, dtype=f64, device=dev); b_ = torch.randn(20, dtype=f64, device=dev)
    x0, v0 = Phi@a*1e-3, Phi@(om*b_)*1e-3
    xt = x0+hc/2*v0; B = sch.c*Md/hc**2*xt
    its = [pcg(sch.A, sch.P, B, xt.clone(), tol) for tol in (1e-3, 1e-6)]
    print('exact midpoint at h = 1/1200, new model: P^-1 A in [%.3e, %.3f] (condition %.1f); block-Jacobi PCG from x~: %d / %d iterations to 1e-3 / 1e-6' % (
        sch.lmin, sch.lmax, sch.lmax/sch.lmin, *its), flush=True)
    if args.skin_model:
        old = BoundaryTerms(m, g, p, r, bsk, lever='c', gamma='sym_skin')
        X, Y = new.X, old.Y; npt = old.npt
        H = dense_hessian(lambda z: old.energy_xy(th, X+z[:n3].view(-1, 3), Y+z[n3:].view(-1, 3)), n3+3*npt)
        Mo = torch.cat((Md, torch.zeros(3*npt, dtype=f64, device=dev)))
        Ao = 4*torch.diag(Mo)/hc**2+H
        Pb = blocks(Ao, n+npt)
        Pi = torch.linalg.inv(Pb)
        Po = lambda R: torch.einsum('bij,bj->bi', Pi, R.view(-1, 3)).reshape(-1)
        lo_, hi_ = bounds(Ao, Pb)
        yx = -torch.linalg.solve(H[n3:, n3:], H[n3:, :n3]@x0)                             # the skin relaxed on x0
        zt = torch.cat((xt, yx)); Bo = Mo*4/hc**2*zt
        its = [pcg(Ao, Po, Bo, zt.clone(), tol) for tol in (1e-3, 1e-6)]
        print('exact midpoint at h = 1/1200, skin model: P^-1 A in [%.3e, %.3f] (condition %.1f); block-Jacobi PCG from x~ and the relaxed skin: '
              '%d / %d iterations to 1e-3 / 1e-6' % (lo_, hi_, hi_/lo_, *its), flush=True)
        del H, Ao, Pb, Pi; torch.cuda.empty_cache()

    # schemes
    Ms = Md[:, None]
    for base in ('mid', 'BE'):
        for h in args.hs:
            for method, ss in (('exact', [0]), ('jacobi', args.sweeps), ('chebyshev', args.sweeps)):
                for s in ss:
                    sch = Scheme(K, Md, base, h, method, s)
                    v = torch.randn(n3, 2, dtype=f64, device=dev); v = v/v.norm(dim=0)
                    xs, vs = v[:, :1], v[:, 1:]
                    lg = 0.
                    for k in range(2000):
                        xs, vs = sch.step(xs, vs)
                        nr = float(torch.cat((xs, h*vs)).norm())
                        lg += math.log(nr); xs, vs = xs/nr, vs/nr
                    grow = math.exp(lg/2000)
                    # each mode: the step on (phi, 0) and (0, omega phi), projected back (M inner product)
                    Z0 = torch.zeros(n3, 20, dtype=f64, device=dev)
                    x1, v1 = sch.step(torch.cat((Phi, Z0), 1), torch.cat((Z0, Phi*om), 1))
                    pr = lambda Q: (Phi*Ms).T@Q                                              # (20, 40)
                    a11, a12 = torch.diagonal(pr(x1)[:, :20]), torch.diagonal(pr(x1)[:, 20:])
                    a21, a22 = torch.diagonal(pr(v1)[:, :20])/om, torch.diagonal(pr(v1)[:, 20:])/om
                    tr, det = a11+a22, a11*a22-a12*a21
                    disc = (tr**2-4*det).to(torch.complex128)
                    mu = (tr+torch.sqrt(disc))/2
                    theta = torch.angle(mu).abs(); rho = mu.abs()
                    fr = (theta/h/om).cpu().numpy(); keep = (rho**(2*math.pi/theta.clamp(min=1e-300))).cpu().numpy()
                    cost = (1 if method == 'exact' else s)/h
                    print('%-3s h %.0e %-9s %s: growth per step %.6f%s; 20 modes: frequency / exact %.4f..%.4f, amplitude kept per period %.4f..%.4f; '
                          '%s gradients per simulated second' % (
                              base, h, method, ('s %d' % s) if s else '   ', grow, ' UNSTABLE' if grow > 1+1e-9 else '',
                              fr.min(), fr.max(), keep.min(), keep.max(), '%.0f' % cost if method != 'exact' else '(solve) %.0f steps' % cost), flush=True)


if __name__ == '__main__':
    main()
