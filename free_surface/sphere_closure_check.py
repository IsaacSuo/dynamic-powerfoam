"""Check 1 of the skinless sphere model (sphere_closure.py) on the part (2026-10-26), against the current skin model
(BoundaryTerms, lever c, Gamma sym_skin, as part_diag_gamma_sym_skin.log):
  0. geometry: the exposed spheres' moment r N against the skin's lever moment; the caps' share of the cells' moment;
     how far the caps' normals spread (the plane stress is about each cap's mean normal)
  1. rigid motions: F_B = Q, energy 0, forces 0, the skin on the moved spheres' meeting points
  2. uniform plane-stress states about each cap's normal (with and without a rotation): F_B exact per cell
  3. uniform strains (not free of traction on a curved surface, so the surface relaxes in both models): energy against
     the continuum, the skin model with its skin affine (exact) and relaxed (one Newton step on the skin, K_yy dense at
     rest); forces: zero away from the surface, the new model against the relaxed skin model; the geometric skin against
     the affine and the relaxed skin
  PYTHONPATH=<worktree>:<main repo> python free_surface/sphere_closure_check.py"""
import argparse
import math
from pathlib import Path

import numpy as np
import torch

from experiments.pacnerf_dem_forward import build
from free_surface.boundary_skin import BoundaryTerms
from free_surface.part_modes import dense_hessian
from free_surface.sphere_closure import ClosureTerms, SphereSkin

f64, dev = torch.float64, 'cuda'


def rotation(axis, ang):
    a = axis/np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3)+math.sin(ang)*K+(1-math.cos(ang))*K@K


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument('--part', type=Path, default=Path('output/test_part'))
    a_.add_argument('--strain', type=float, default=1e-4)
    args = a_.parse_args()
    torch.manual_seed(0); rng = np.random.default_rng(0)
    stt = torch.load(args.part/'run'/'model.pt', map_location='cpu', weights_only=True)
    p = stt['points'].double().numpy(); r = torch.nn.functional.softplus(stt['radii'].double(), beta=100).numpy()
    E, nu = 23008.38, .3008
    g = dict(np.load(args.part/'dem_exact_single'/'geometry.npz'))
    m = build(g, E, nu, dev, 'noslip', .25, .3, f64, kinematics='exact', nodalvol=True, nonlinear=False, ogden_alpha=2.)
    bsk = dict(np.load(args.part/'boundary_skin.npz'))
    old = BoundaryTerms(m, g, p, r, bsk, lever='c', gamma='sym_skin')
    new = ClosureTerms(m, g, p, r, bsk)
    pt, rt = torch.as_tensor(p, dtype=f64, device=dev), torch.as_tensor(r, dtype=f64, device=dev)
    sk = SphereSkin(pt, rt)
    n, npt = new.n, old.npt
    th = torch.zeros(n, 9, dtype=f64, device=dev)
    X, Y = new.X, old.Y
    mu, lam = new.mu, float(m.k_v)
    I3 = torch.eye(3, dtype=f64, device=dev)
    ctr = X.mean(0)
    rmin = rt.min()

    # 0. geometry
    capb = new.cap
    dM = float(((new.Ms-new.Mcap).flatten(1).norm(dim=1)/new.Mcap.flatten(1).norm(dim=1).clamp(min=1e-300))[capb].max())
    L = torch.linalg.cholesky(new.Mb+new.Mcap); Li = torch.linalg.inv(L)
    share = torch.linalg.eigvalsh(Li@new.Mcap@Li.transpose(1, 2))[:, -1][capb].cpu().numpy()
    ev = torch.linalg.eigvalsh(new.c_N/new.c_A[:, None, None]).cpu().numpy()              # per patch, ascending
    spread = ev[:, 1]/ev[:, 2]
    wev = torch.linalg.eigvalsh(new.Mb).cpu().numpy(); wr = wev[:, 1]/wev[:, 2]
    print('part: %d bodies (%d with an exposed sphere), %d walls, %d skin points; the recomputed skin points against boundary_skin.npz %.1e m; '
          'r N against the skin\'s lever moment: largest relative difference %.1e' % (
              n, int(capb.sum()), len(new.A), npt, float((sk.Y-Y).abs().max()), dM), flush=True)
    q = lambda v: '%.3f / %.3f / %.3f' % tuple(np.quantile(v, (.5, .9, 1.)))
    print('caps\' share of the cell\'s moment (largest generalised eigenvalue of r N against M_B + r N), median / 90 %% / max: %s; '
          'the caps\' normal spread (second / first eigenvalue of N / A): %s; the walls\' coverage (second / first eigenvalue of M_B), all bodies: %s, '
          'below 0.05: %d' % (q(share), q(spread), q(wr), int((wr < .05).sum())), flush=True)

    def energy_new(x):
        return new.energy_x(th, x)

    def grad_new(x):
        x = x.detach().clone().requires_grad_(True)
        E_ = energy_new(x)
        return float(E_.detach()), torch.autograd.grad(E_, x)[0]

    def grad_old(x, y):
        x = x.detach().clone().requires_grad_(True); y = y.detach().clone().requires_grad_(True)
        E_ = old.energy_xy(th, x, y)
        gx, gy = torch.autograd.grad(E_, (x, y))
        return float(E_.detach()), gx, gy

    def strain_state(e):
        T = I3+torch.as_tensor(e, dtype=f64, device=dev)
        return ctr+(X-ctr)@T.T, ctr+(Y-ctr)@T.T, T

    # scales: uniaxial strain along x
    e1 = np.zeros((3, 3)); e1[0, 0] = args.strain
    xs, ys, _ = strain_state(e1)
    E_ref = float(new.V.sum())*(mu*args.strain**2+.5*lam*args.strain**2)
    _, gref = grad_new(xs)
    f_ref = float(gref.norm(dim=1).max())
    print('scales: continuum energy at a uniaxial strain %.0e: %.3e; the new model\'s largest body force there %.3e' % (args.strain, E_ref, f_ref), flush=True)

    # 1. rigid motions
    for ang in (.3, 1.5, 3.):
        Q = torch.as_tensor(rotation(rng.normal(size=3), ang), dtype=f64, device=dev)
        t = torch.as_tensor(rng.normal(size=3)*.05, dtype=f64, device=dev)
        x = ctr+(X-ctr)@Q.T+t
        F, ch = new.closure(new.walls(x))
        En, gn = grad_new(x)
        Eo = float(old.energy_xy(th, x, ctr+(Y-ctr)@Q.T+t))
        y, rc, rs, rp = sk.points(x, F)
        ys_ = ctr+(sk.Y-ctr)@Q.T+t
        print('1. rigid, angle %.1f rad: |F_B - Q| max %.1e (Newton last step %.1e); energy / reference: new %.1e, skin model (skin moved along) %.1e; '
              'largest force / reference %.1e; skin points off the moved rest points %.1e of the smallest radius (residuals: corners %.1e, '
              'midpoints %.1e, planes %.1e)' % (ang, float((F-Q).abs().max()), ch, En/E_ref, Eo/E_ref, float(gn.norm(dim=1).max())/f_ref,
                                               float((y-ys_).norm(dim=1).max()/rmin), rc, rs, rp), flush=True)

    # 2. uniform plane-stress states about each cap's normal
    kap = new.kap0
    for amp in (1e-2, .1):
        for turn in (False, True):
            A = torch.as_tensor(rng.normal(size=(n, 3, 3))*amp, dtype=f64, device=dev)
            et = new.P@(.5*(A+A.transpose(1, 2)))@new.P
            e = et-kap*et.diagonal(dim1=1, dim2=2).sum(1)[:, None, None]*new.mm
            Fb = I3+e
            if turn:
                Fb = torch.as_tensor(np.stack([rotation(rng.normal(size=3), rng.uniform(0, 3)) for _ in range(n)]), dtype=f64, device=dev)@Fb
            F, ch = new.closure(Fb@new.Mb)
            err = (F-Fb).flatten(1).abs().max(1).values
            print('2. plane stress about each cap\'s normal, strain amplitude %.2f%s: |F_B - F| max over the caps\' bodies %.1e (others %.1e); '
                  'Newton last step %.1e' % (amp, ', each cell turned' if turn else '', float(err[capb].max()), float(err[~capb].max()), ch), flush=True)

    # 3. uniform strains
    print('3. skin model K_yy at rest (dense, %d unknowns) ...' % (3*npt), flush=True)
    Kyy = dense_hessian(lambda z: old.energy_xy(th, X, Y+z.view(-1, 3)), 3*npt)
    LU = torch.linalg.lu_factor(Kyy)
    nb = torch.zeros(n, dtype=torch.bool, device=dev); nb[new.i[capb[new.j]]] = True; nb[new.j[capb[new.i]]] = True
    deep = ~capb & ~nb
    print('   bodies away from the surface (no exposed sphere, no neighbour with one): %d' % int(deep.sum()), flush=True)
    states = [('xx', (0, 0)), ('yy', (1, 1)), ('zz', (2, 2)), ('xy', (0, 1)), ('xz', (0, 2)), ('yz', (1, 2)), ('hydrostatic', None)]
    for name, ij in states:
        e = np.zeros((3, 3))
        if ij is None:
            e = np.eye(3)*args.strain
        else:
            e[ij] = args.strain; e[ij[::-1]] = args.strain
        x, ya, T = strain_state(e)
        Ec = float(new.V.sum())*(mu*(e*e).sum()+.5*lam*np.trace(e)**2)
        Ea, _, gy = grad_old(x, ya)
        yr = ya-torch.linalg.lu_solve(*LU, gy.reshape(-1, 1)).view(-1, 3)
        Er, gxo, gy2 = grad_old(x, yr)
        En, gxn = grad_new(x)
        F, ch = new.closure(new.walls(x))
        yg, rc, rs, rp = sk.points(x, F)
        du = (yr-Y).norm(dim=1)
        rel = lambda a, b: float(torch.median((a-b).norm(dim=1))/torch.median(du))
        fd = float((gxn-gxo).norm()/gxo.norm())
        fdc = float((gxn-gxo)[capb].norm()/gxo[capb].norm())
        print('   %-11s energy / continuum: skin model affine skin %.4f, relaxed %.4f; new %.4f | forces: away from the surface max %.1e of the reference; '
              'new against relaxed skin model |df| / |f| %.3f (bodies with a cap %.3f); skin residual after the step %.1e of before | '
              'skin points (median, of the relaxed skin\'s displacement): geometric - affine %.3f, geometric - relaxed %.3f, affine - relaxed %.3f; '
              'closure Newton last step %.0e, Newton residuals %.0e %.0e %.0e' % (
                  name, Ea/Ec, Er/Ec, En/Ec, float(gxn[deep].norm(dim=1).max())/f_ref if bool(deep.any()) else float('nan'), fd, fdc,
                  float(gy2.norm()/gy.norm()), rel(yg, ya), rel(yg, yr), rel(ya, yr), ch, rc, rs, rp), flush=True)


if __name__ == '__main__':
    main()
