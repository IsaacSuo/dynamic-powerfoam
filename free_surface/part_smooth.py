"""Smooth fields with exact answers on the synthetic part (2026-10-22): what does each term count when the bodies (and
the skin) follow a known non-affine field? Fields (X~ = X - centroid, L the part's size): pure bending in the planes xz,
yz, xy (u_a = X~_a X~_b / L, u_b = -X~_a^2 / 2L: eps_aa = X~_b / L, no shear), torsion about z (u = X~_z (-X~_y, X~_x, 0)
/ L), and four random quadratic fields. Continuum: the small-strain energy int mu eps:eps + lam/2 (tr eps)^2 dV over the
material (the union of the balls), grid-sampled at two spacings. Models (linear bonds, quadratic in the amplitude):
  A        current (bonds, walls, volume, completion springs of the exposed spheres; part_modes.PatchSprings)
  skin     sphere model (sphere_model.SphereTerms) with the skin on the exact field
  relaxed  the same with the skin free (condensed: y = -K_yy^-1 K_yx x, dense)
each split into terms; the bonds into normal, shear, bending (kb) and torsion (kt)."""
import argparse
from pathlib import Path

import numpy as np
import torch

from experiments.foam_surface_completion import matrix_quat, polar_newton
from experiments.pacnerf_dem_forward import build
from free_surface.grad_weights import MidpointTerms
from free_surface.part_modes import PatchSprings, dense_hessian
from free_surface.boundary_skin import BoundaryTerms
from free_surface.sphere_model import SphereTerms

f64, dev = torch.float64, 'cuda'


def fields(c, L, seed=0):
    """name -> (u(X), grad u(X)) for arrays X (n, 3), numpy"""
    out = {}
    for a, b, nm in ((0, 2, 'bend xz'), (1, 2, 'bend yz'), (0, 1, 'bend xy')):
        def u(X, a=a, b=b):
            Z = X-c; U = np.zeros_like(X)
            U[:, a] = Z[:, a]*Z[:, b]/L; U[:, b] = -Z[:, a]**2/(2*L)
            return U
        def gu(X, a=a, b=b):
            Z = X-c; G = np.zeros((len(X), 3, 3))
            G[:, a, a] = Z[:, b]/L; G[:, a, b] = Z[:, a]/L; G[:, b, a] = -Z[:, a]/L
            return G
        out[nm] = (u, gu)
    def ut(X):
        Z = X-c
        return np.column_stack((-Z[:, 2]*Z[:, 1], Z[:, 2]*Z[:, 0], np.zeros(len(X))))/L
    def gt(X):
        Z = X-c; G = np.zeros((len(X), 3, 3))
        G[:, 0, 1] = -Z[:, 2]; G[:, 0, 2] = -Z[:, 1]; G[:, 1, 0] = Z[:, 2]; G[:, 1, 2] = Z[:, 0]
        return G/L
    out['torsion z'] = (ut, gt)
    rng = np.random.default_rng(seed)
    for k in range(4):
        Aq = rng.standard_normal((3, 3, 3)); Aq = .5*(Aq+Aq.transpose(0, 2, 1))/L
        out['random %d' % k] = (lambda X, Aq=Aq: .5*np.einsum('ijk,nj,nk->ni', Aq, X-c, X-c),
                                lambda X, Aq=Aq: np.einsum('ijk,nk->nij', Aq, X-c))
    return out


def continuum(p, r, gu, mu, lam, spacing):
    lo, hi = (p-r[:, None]).min(0), (p+r[:, None]).max(0)
    pt, rt = torch.as_tensor(p, device=dev), torch.as_tensor(r, device=dev)
    axes = [np.arange(lo[k]+spacing/2, hi[k], spacing) for k in range(3)]
    E = 0.
    for x0 in axes[0]:
        G = np.stack(np.meshgrid([x0], axes[1], axes[2], indexing='ij'), -1).reshape(-1, 3)
        ins = ((torch.cdist(torch.as_tensor(G, device=dev), pt)**2-rt[None]**2).min(1).values <= 0).cpu().numpy()
        if not ins.any():
            continue
        Gr = gu(G[ins]); eps = .5*(Gr+Gr.transpose(0, 2, 1))
        E += float((mu*(eps*eps).sum((1, 2))+.5*lam*np.trace(eps, axis1=1, axis2=2)**2).sum())*spacing**3
    return E


def bond_parts(m, theta, x, q):
    *_, k, dn, dt, pn, pt = m._bond_exact(theta, x, q)
    return dict(b_normal=float((.5*k[:, 0]*dn**2).sum()), b_shear=float((.5*k[:, 1]*(dt*dt).sum(-1)).sum()),
                b_bend=float((.5*k[:, 2]*(pt*pt).sum(-1)).sum()), b_twist=float((.5*k[:, 3]*pn**2).sum()))


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--part', type=Path, default=Path('output/test_part'))
    a.add_argument('--skin', choices=('fan', 'boundary'), default='fan')
    a.add_argument('--skin-file', default='boundary_skin.npz')
    a.add_argument('--lever', choices=('point', 'c'), default='point')
    args = a.parse_args()
    stt = torch.load(args.part/'run'/'model.pt', map_location='cpu', weights_only=True)
    p = stt['points'].double().numpy(); r = torch.nn.functional.softplus(stt['radii'].double(), beta=100).numpy()
    g = dict(np.load(args.part/'dem_exact_single'/'geometry.npz'))
    skin = dict(np.load(args.part/'sphere_skin.npz')); pts = dict(np.load(args.part/'sphere_points.npz'))
    m = build(g, 23008.38, .3008, dev, 'noslip', .25, .3, f64, kinematics='exact', nodalvol=True, nonlinear=False, ogden_alpha=2.)
    st = SphereTerms(m, g, p, r, skin, pts) if args.skin == 'fan' else BoundaryTerms(m, g, p, r, dict(np.load(args.part/args.skin_file)), lever=args.lever)
    mid = MidpointTerms(m, PatchSprings(g, r, skin, float(m.mu_ref), st.V), g)
    n, npt = st.n, st.npt; n3, v3 = 3*n, 3*npt
    th = torch.zeros(n, 9, dtype=f64, device=dev)
    X, Y = st.X, st.Y
    mu, lam = float(m.mu_ref), float(m.k_v)
    Xn, Yn = X.cpu().numpy(), Y.cpu().numpy()
    c = Xn.mean(0); L = float(np.ptp(Xn, axis=0).max())
    H = dense_hessian(lambda z: st.energy_xy(th, X+z[:n3].view(-1, 3), Y+z[n3:].view(-1, 3)), n3+v3)
    Gy = torch.linalg.solve(H[n3:, n3:], H[n3:, :n3])
    print('terms as fractions of the continuum energy; bonds split into normal / shear / bending kb / torsion kt', flush=True)
    for name, (u, gu) in fields(c, L).items():
        Ec = continuum(p, r, gu, mu, lam, .0005); Ec2 = continuum(p, r, gu, mu, lam, .00035)
        ux = torch.as_tensor(u(Xn), device=dev); uy = torch.as_tensor(u(Yn), device=dev)
        eps = 1e-6/float(ux.abs().max())
        s2 = 1/(eps*eps)
        # A
        x = X+eps*ux
        F = mid.F(x); R = polar_newton(F); q = matrix_quat(R)
        bA = {k_: v_*s2 for k_, v_ in bond_parts(m, th, x, q).items()}
        vA = float(mid.volume_energy(x, F, R, m.k_v))*s2; cA = float(mid.completion_energy(F, R, th))*s2
        EA = sum(bA.values())+vA+cA
        rows = {'A': dict(**bA, volume=vA, outer=cA)}
        for tag, yy in (('skin', Y+eps*uy), ('relaxed', Y-eps*(Gy@ux.reshape(-1)).view(-1, 3))):
            T = {k_: float(v_)*s2 for k_, v_ in st.energy_terms(th, x, yy).items()}
            F = st.F(x, yy); q = matrix_quat(polar_newton(F))
            bp = {k_: v_*s2 for k_, v_ in bond_parts(m, th, x, q).items()}
            rows[tag] = dict(**bp, volume=T['volume'], outer=T['outer_stretch']+T['outer_cross'], skin_dev=T['outer_skin'], lateral=T.get('outer_lateral', 0.))
            rows[tag]['_check'] = T['bonds']-sum(bp.values())
        print('\n%s: continuum %.6e (grid 0.5 / 0.35 mm: %.2e rel)' % (name, Ec2, Ec/Ec2-1), flush=True)
        for tag, rw in rows.items():
            chk = rw.pop('_check', 0.)
            tot = sum(rw.values())
            print('   %-8s total / continuum %.4f | %s%s' % (tag, tot/Ec2, ' '.join('%s %.4f' % (k_, v_/Ec2) for k_, v_ in rw.items()),
                                                         '' if tag == 'A' else ' (bond split check %.1e)' % (chk/Ec2)), flush=True)


if __name__ == '__main__':
    main()
