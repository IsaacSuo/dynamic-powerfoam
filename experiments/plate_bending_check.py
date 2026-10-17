"""Bending of thin plates in the bonded DEM: the beam-formula kb/kt (bond share E_b = 2 alpha mu, circular
I = A^2/4pi) against the continuum-derived values of BOND_BENDING.md (full E, true face second moment:
kb = E I_face / ((1 - nu^2) L), kt = E h^2 A / (12 (1 + nu) L)), on cubic-lattice cantilever plates 1..4 cells
thick. Reference: Hex8 FEM of the same slab (--fem-div elements per cell edge) and, for the thin cases, plate /
beam theory. Static tip deflection under a uniformly distributed vertical tip load, DEM linearised at rest
(pacnerf_modal_check.dem_stiffness), clamped first column. Per unit density (E / rho), float64, CPU.
"""
import argparse

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from experiments.foam_dem_energy import EnergyFoam
from experiments.pacnerf_modal_check import dem_stiffness
from experiments.springgaus_modal_compare import hex_fem

RHO = 1000.


def lattice_plate(nx, ny, nz, a):
    idx = np.array([(i, j, k) for i in range(nx) for j in range(ny) for k in range(nz)])
    pts = (idx+.5)*a
    lookup = {tuple(t): n for n, t in enumerate(idx)}
    pairs, contacts = [], []
    for n, t in enumerate(idx):
        for d in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            m = lookup.get((t[0]+d[0], t[1]+d[1], t[2]+d[2]))
            if m is not None:
                pairs.append((n, m)); contacts.append(.5*(pts[n]+pts[m]))
    pairs = np.array(pairs); contacts = np.array(contacts)
    m1 = np.zeros((len(pts), 3))                          # first moment of the neighbour set: free-surface normal = -m1 direction
    np.add.at(m1, pairs[:, 0], pts[pairs[:, 1]]-pts[pairs[:, 0]]); np.add.at(m1, pairs[:, 1], pts[pairs[:, 0]]-pts[pairs[:, 1]])
    nrm = np.linalg.norm(m1, axis=1, keepdims=True)
    air_normal = np.where(nrm > 1e-9, -m1/np.maximum(nrm, 1e-12), 0.)
    return dict(air_normal=air_normal, points=pts, radii=np.full(len(pts), a*np.sqrt(1/2.4)), pairs=pairs, areas=np.full(len(pairs), a*a),
                contacts=contacts, masses=np.full(len(pts), a**3), fixed=idx[:, 0] == 0), idx


def dem_tip_deflection(geometry, idx, E, nu, alpha, kb_mode, volume, residual, a, enhanced=False, thin_ratio=0., thin_bend=False):
    lam, mu = E*nu/((1+nu)*(1-2*nu)), E/(2*(1+nu))
    model = EnergyFoam(geometry, k_v=lam/RHO, volume=volume, device='cpu', young=2*alpha*mu/RHO, shear=2*alpha*mu/RHO,
                       body_law='neohookean', k_mu=(1-alpha)*mu/RHO, dtype=torch.float64, kinematics='small', residual=residual,
                       enhanced=enhanced, thin_ratio=thin_ratio, thin_bend_factor=1./(alpha*(1-nu)) if thin_bend else 1.)
    if kb_mode == 'derived':
        # square faces a x a: I_face = a^4/12 about either in-plane axis, h = a, A = a^2, L = a
        L = torch.linalg.vector_norm(model.X[model.j]-model.X[model.i], dim=1)
        A = torch.as_tensor(geometry['areas'], dtype=torch.float64)
        h = torch.sqrt(A)                                  # square faces
        I_face = h**3*torch.sqrt(A)/12
        model.base[:, 2] = (E/RHO)*I_face/((1-nu*nu)*L)
        model.base[:, 3] = (E/RHO)*h*h*A/(12*(1+nu)*L)
    n = model.n
    K = dem_stiffness(model, torch.zeros(n, 7, dtype=torch.float64)).tocsr()
    free = np.repeat(~geometry['fixed'], 6)
    tip = idx[:, 0] == idx[:, 0].max()
    f = np.zeros(6*n); f[6*np.flatnonzero(tip)+2] = 1./tip.sum()
    Kff = K[free][:, free]
    u = np.zeros(6*n); u[free] = spla.spsolve(Kff.tocsc(), f[free])
    return float(u[6*np.flatnonzero(tip)+2].mean()), model


def fem_tip_deflection(nx, ny, nz, a, div, E, nu):
    h = a/div
    vox = np.array([(i, j, k) for i in range(nx*div) for j in range(ny*div) for k in range(nz*div)])
    K_lam, K_mu, M, node_id, _, N = hex_fem(vox, np.zeros(3), h)
    lam, mu = E*nu/((1+nu)*(1-2*nu))/RHO, E/(2*(1+nu))/RHO
    K = (lam*K_lam+mu*K_mu).tocsr()
    nodes = np.array(list(node_id.keys())); order = np.array(list(node_id.values()))
    xyz = np.zeros((len(nodes), 3)); xyz[order] = nodes*h
    clamped = xyz[:, 0] < 1e-9
    tipn = xyz[:, 0] > nx*a-1e-9
    free = np.repeat(~clamped, 3)
    f = np.zeros(N); f[3*np.flatnonzero(tipn)+2] = 1./tipn.sum()
    u = np.zeros(N); u[free] = spla.spsolve(K[free][:, free].tocsc(), f[free])
    return float(u[3*np.flatnonzero(tipn)+2].mean())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--E', type=float, default=24107.)
    p.add_argument('--nu', type=float, default=.414)
    p.add_argument('--a', type=float, default=.02, help='cell size (m)')
    p.add_argument('--nx', type=int, default=16)
    p.add_argument('--ny', type=int, default=6)
    p.add_argument('--layers', type=int, nargs='+', default=[1, 2, 3, 4])
    p.add_argument('--fem-div', type=int, default=3)
    args = p.parse_args()
    E, nu, a = args.E, args.nu, args.a
    Lx = args.nx*a; W = args.ny*a
    print('cantilever plate %d x %d cells of %.0f mm, tip load 1 (per unit density), E %.0f nu %.3f' % (args.nx, args.ny, 1000*a, E, nu))
    for nz in args.layers:
        geometry, idx = lattice_plate(args.nx, args.ny, nz, a)
        fem = fem_tip_deflection(args.nx, args.ny, nz, a, args.fem_div, E, nu)
        hthk = nz*a
        theory = RHO*Lx**3/(3*E*W*hthk**3/12)               # Euler-Bernoulli narrow beam, E (no 1-nu^2), per unit density
        one_layer = nz == 1                                  # body term degenerate (coplanar neighbours): bonds only, alpha = 1
        variants = [('bonds only a=1', dict(alpha=1., volume=False, residual=False)),
                    ('full+thin+EAS', dict(alpha=.25, volume=True, residual='normal', enhanced=True, thin_ratio=.25)),
                    ('full+thin+EAS+kb', dict(alpha=.25, volume=True, residual='normal', enhanced=True, thin_ratio=.25, thin_bend=True))] if one_layer else [
            ('full (res full)', dict(alpha=.25, volume=True, residual='full')), ('full (res normal)', dict(alpha=.25, volume=True, residual='normal')),
            ('res normal + EAS', dict(alpha=.25, volume=True, residual='normal', enhanced=True)),
            ('full, res off', dict(alpha=.25, volume=True, residual=False)),
            ('bonds only a=1', dict(alpha=1., volume=False, residual=False))]
        print('%-6d FEM %10.4g   theory(narrow beam) %10.4g   plate E/(1-nu^2) %10.4g' % (nz, fem, theory, theory*(1-nu*nu)))
        for name, kw in variants:
            d_beam, _ = dem_tip_deflection(geometry, idx, E, nu, kb_mode='beam', a=a, **kw)
            d_der, _ = dem_tip_deflection(geometry, idx, E, nu, kb_mode='derived', a=a, **kw)
            print('        %-16s beam-kb %10.4g (%.2f x FEM)   derived-kb %10.4g (%.2f x FEM)' % (name, d_beam, d_beam/fem, d_der, d_der/fem))


if __name__ == '__main__':
    main()


def decompose(nz, nu, kb_mode, residual, E=24107., a=.02, nx=16, ny=6, alpha=.25):
    """Energy split of the solved cantilever state: which term carries the bending work."""
    geometry, idx = lattice_plate(nx, ny, nz, a)
    d, model = dem_tip_deflection(geometry, idx, E, nu, alpha=alpha, kb_mode=kb_mode, volume=True, residual=residual, a=a)
    n = model.n
    K = dem_stiffness(model, torch.zeros(n, 7, dtype=torch.float64)).tocsr()
    free = np.repeat(~geometry['fixed'], 6)
    tip = idx[:, 0] == idx[:, 0].max()
    f = np.zeros(6*n); f[6*np.flatnonzero(tip)+2] = 1./tip.sum()
    u = np.zeros(6*n); u[free] = spla.spsolve(K[free][:, free].tocsc(), f[free])
    total = .5*u@(K@u)
    parts = {}
    base = model.base.clone()
    for name, cols in (('kn', [0]), ('ks', [1]), ('kb', [2]), ('kt', [3])):
        model.base[:] = 0; model.base[:, cols] = base[:, cols]
        model.volume = False
        Kp = dem_stiffness(model, torch.zeros(n, 7, dtype=torch.float64)).tocsr(); parts[name] = .5*u@(Kp@u)
    model.base[:] = 0; model.volume = True; model.residual = False
    Kp = dem_stiffness(model, torch.zeros(n, 7, dtype=torch.float64)).tocsr(); parts['NH'] = .5*u@(Kp@u)
    if residual:
        model.residual = True
        Kp = dem_stiffness(model, torch.zeros(n, 7, dtype=torch.float64)).tocsr(); parts['residual'] = .5*u@(Kp@u)-parts['NH']
    model.base[:] = base
    print('layers %d nu %.3f %s-kb res %s: tip %.4g | energy share ' % (nz, nu, kb_mode, residual, d)+'  '.join('%s %.0f%%' % (k, 100*v/total) for k, v in parts.items()))


def zero_modes(nz=4, nu=.414, E=24107., a=.02, nx=4, ny=4, tol=1e-6):
    """Free block: eigenvalues of the linearised stiffness; count the near-zero ones beyond the 6 rigid modes."""
    geometry, idx = lattice_plate(nx, ny, nz, a); geometry['fixed'][:] = False
    lam, mu = E*nu/((1+nu)*(1-2*nu)), E/(2*(1+nu))
    for name, res in (('res off', False), ('res full', 'full'), ('res normal', 'normal')):
        model = EnergyFoam(geometry, k_v=lam/RHO, volume=True, device='cpu', young=.5*mu/RHO, shear=.5*mu/RHO, body_law='neohookean',
                           k_mu=.75*mu/RHO, dtype=torch.float64, kinematics='small', residual=res)
        K = dem_stiffness(model, torch.zeros(model.n, 7, dtype=torch.float64)).toarray()
        w = np.linalg.eigvalsh(K); w = w/w.max()
        print('%-5dx%d x%d block, %-11s: zero modes beyond the 6 rigid ones: %d (eigenvalues 6..12: %s)' % (
            nx, ny, nz, name, int((w < tol).sum())-6, np.array2string(w[6:12], precision=2)))


if __name__ == '__main__' and False:
    pass
