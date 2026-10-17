"""Experiment 1 of the discretization-error budget: free-free natural frequencies, DEM vs FEM.

No ground, gravity or damping: only the stiffness and mass of the object. DEM: the EnergyFoam of
pacnerf_dem_forward.build (bonds share alpha of mu, Neo-Hookean bodies, inertia scaled x4) is
linearised at rest. Its sparse stiffness is assembled exactly: bond Hessians (12 x 12, torch.func)
plus body terms G^T (V C) G, where G maps body centre displacements to the least-squares F and
C = d2 psi/dF2 at F = I (linear elasticity with the Neo-Hookean lam, mu). Checks: rigid modes
are null, and K v equals the forward-mode derivative of the closed-form forces.
FEM reference: the object of the GT particles (frame 0) voxelised at h (occupied voxels, holes
filled), trilinear hexahedra (scikit-fem), consistent mass, same E, nu, unit density.
Both: the 6 rigid modes are skipped; frequencies f = sqrt(eig)/(2 pi) in Hz (rho = 1000 scaling
cancels: stiffness per unit density).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import torch
from scipy import ndimage

from experiments.foam_dem_energy import quat_mul, quat_small
from experiments.pacnerf_dem_forward import build, read_ply


def dem_stiffness(model, theta):
    n = model.n
    i, j = model.i, model.j
    X = model.X

    def bond_energy(z, ri, rj, n0, k, dX):
        # z: (12,) = u_i, rot_i, u_j, rot_j for one bond at rest positions (relative, frame of the bond)
        from experiments.foam_dem_energy import quat_matrix
        q0 = torch.tensor([1., 0, 0, 0], dtype=z.dtype)
        Ri = quat_matrix(quat_mul(quat_small(z[3:6]), q0))
        Rj = quat_matrix(quat_mul(quat_small(z[9:12]), q0))
        D = dX+(z[6:9]+Rj@rj)-(z[0:3]+Ri@ri)       # X_j - X_i + r_j - r_i = 0 at rest
        d = .5*(Ri.T@D+Rj.T@D)
        Psi = Ri.T@Rj
        A = Psi-Psi.T
        phi = .5*torch.stack((A[2, 1], A[0, 2], A[1, 0]))
        dn, pn = d@n0, phi@n0
        dt, pt = d-dn*n0, phi-pn*n0
        return .5*(k[0]*dn**2+k[1]*dt@dt+k[2]*pt@pt+k[3]*pn**2)
    # D at rest is (X_j + r_j) - (X_i + r_i) = 0 since both levers reach the same contact point.
    k = model.base*torch.exp(.5*(theta[i, :4]+theta[j, :4]))
    H = torch.func.vmap(torch.func.hessian(bond_energy))(torch.zeros(len(i), 12, dtype=torch.float64),
                                                         model.ri, model.rj, model.n0, k, X[j]-X[i])
    dofs = torch.cat((6*i[:, None]+torch.arange(6), 6*j[:, None]+torch.arange(6)), 1)
    rows = dofs[:, :, None].expand(-1, 12, 12)
    cols = dofs[:, None, :].expand(-1, 12, 12)
    K = sp.coo_matrix((H.reshape(-1).numpy(), (rows.reshape(-1).numpy(), cols.reshape(-1).numpy())), shape=(6*n, 6*n))
    if not getattr(model, 'volume', True):                 # bonds only (EnergyFoam(volume=False))
        K = K.tocsr()
        return .5*(K+K.T)
    # Body (Neo-Hookean) part: F_i = sum_j w (x_j - x_i) (dX M^-1)_row -> G (9n x 3n), translations only.
    a = model.ls_a.numpy(); b = model.ls_b.numpy()
    coef = (model.ls_w[:, None]*(model.ls_dX[:, None, :]@model.moment_inv[model.ls_a])[:, 0]).numpy()   # (pairs, 3): d F_pq / d x_b,p = coef_q
    r_, c_, v_ = [], [], []
    for p in range(3):
        for q in range(3):
            r_ += [9*a+3*p+q, 9*a+3*p+q]; c_ += [6*b+p, 6*a+p]; v_ += [coef[:, q], -coef[:, q]]
    G = sp.coo_matrix((np.concatenate(v_), (np.concatenate(r_), np.concatenate(c_))), shape=(9*n, 6*n)).tocsr()
    if getattr(model, 'thin', None) is not None and bool(model.thin.any()):
        # thin bodies: F = F_t + normalize(F_t t1 x F_t t2) e^T, linearised at F_t = I - e e^T (tangent identity)
        def closure(ft9, e, t1, t2):
            Ft = ft9.reshape(3, 3)
            nrm = torch.linalg.cross(Ft@t1, Ft@t2); nrm = nrm/torch.linalg.vector_norm(nrm)
            return (Ft+nrm[:, None]*e[None, :]).reshape(9)
        thin = model.thin.numpy()
        rest = (torch.eye(3, dtype=torch.float64)-model.thin_e[:, :, None]*model.thin_e[:, None, :]).reshape(-1, 9)
        T = torch.func.vmap(torch.func.jacrev(closure))(rest[model.thin], model.thin_e[model.thin], model.thin_t1[model.thin], model.thin_t2[model.thin]).numpy()
        Tfull = sp.identity(9*n, format='lil')
        for k_, b_ in enumerate(np.flatnonzero(thin)):
            Tfull[9*b_:9*b_+9, 9*b_:9*b_+9] = T[k_]
        G = (Tfull.tocsr()@G).tocsr()
    lam = model.k_v*torch.exp(theta[:, 4])*model.cell_volume
    mu = model.k_mu*torch.exp(theta[:, 5])*model.cell_volume
    e = np.eye(3)
    C = np.einsum('ij,kl->ijkl', e, e)
    S = np.einsum('ik,jl->ijkl', e, e)+np.einsum('il,jk->ijkl', e, e)
    blocks = lam.numpy()[:, None, None]*C.reshape(9, 9)+mu.numpy()[:, None, None]*S.reshape(9, 9)
    if getattr(model, 'enhanced', False):               # static condensation of the enhanced strain along e: C - (Cp)(Cp)^T / p^T C p
        ev = model.eas_dir.numpy(); on = model.eas_on.numpy()
        for b_ in np.flatnonzero(on):
            p9 = np.outer(ev[b_], ev[b_]).reshape(9)
            Cp = blocks[b_]@p9
            blocks[b_] = blocks[b_]-np.outer(Cp, Cp)/(p9@Cp)
    Cb = sp.block_diag(list(blocks), format='csr')
    K = (K+G.T@Cb@G).tocsr()
    if getattr(model, 'residual', False):
        # r = (u_b - u_a) - F_a dX (linear in translations): K_res = 2 R^T diag(c) R
        P = len(a)
        dX = model.ls_dX.numpy()
        rr, cc, vv = [], [], []
        for p_ in range(3):
            rr += [3*np.arange(P)+p_]*2; cc += [6*b+p_, 6*a+p_]; vv += [np.ones(P), -np.ones(P)]
        Dm = sp.coo_matrix((np.concatenate(vv), (np.concatenate(rr), np.concatenate(cc))), shape=(3*P, 6*n)).tocsr()
        rr, cc, vv = [], [], []
        for p_ in range(3):
            for q_ in range(3):
                rr.append(3*np.arange(P)+p_); cc.append(9*a+3*p_+q_); vv.append(dX[:, q_])
        Sel = sp.coo_matrix((np.concatenate(vv), (np.concatenate(rr), np.concatenate(cc))), shape=(3*P, 9*n)).tocsr()
        R = Dm-Sel@G
        c, _ = model._residual(theta, model.X)
        if getattr(model, 'residual_mode', 'full') == 'normal':      # (r . n)^2 with n the rest bond direction
            nhat = (dX/np.linalg.norm(dX, axis=1, keepdims=True))
            W = sp.block_diag([c_*np.outer(n_, n_) for c_, n_ in zip(c.numpy(), nhat)], format='csr')
        else:
            W = sp.diags(np.repeat(c.numpy(), 3))
        K = (K+2*R.T@W@R).tocsr()
    return .5*(K+K.T)


def dem_modes(model, theta, k=20):
    K = dem_stiffness(model, theta)
    m = np.concatenate((np.repeat(model.mass.numpy()[:, None], 3, 1), np.repeat(model.inertia.numpy()[:, None], 3, 1)), 1).ravel()
    # checks: rigid motions are null; K v = - d(forces)/dz v (forward mode) for a random v
    x0, q0, _, _ = model.initial_state()
    rng = np.random.default_rng(0)
    v = torch.tensor(rng.normal(size=(model.n, 6)))
    def force(zv):
        f, tau = model.forces(theta, x0+zv[:, :3], quat_mul(quat_small(zv[:, 3:]), q0))
        return torch.cat((f, tau), 1)
    _, Kv = torch.func.jvp(force, (torch.zeros(model.n, 6, dtype=torch.float64),), (v,))
    check_kv = float(np.abs(K@v.numpy().ravel()+Kv.numpy().ravel()).max()/np.abs(Kv.numpy()).max())
    w = torch.tensor([.3, -.8, .5], dtype=torch.float64)
    rigid = torch.cat((torch.linalg.cross(w.expand(model.n, 3), x0-x0.mean(0))+torch.tensor([.1, .2, -.3]), w.expand(model.n, 3)), 1)
    check_rigid = float(np.abs(K@rigid.numpy().ravel()).max()/np.abs(K).max())
    Mi = sp.diags(1/np.sqrt(m))
    A = (Mi@K@Mi).tocsc()
    vals = sla.eigsh(A, k=k+6, sigma=-1e-3, which='LM', return_eigenvectors=False)
    vals = np.sort(vals)
    return vals, dict(kv_vs_forces_relative=check_kv, rigid_residual_relative=check_rigid, dofs=6*model.n)


def fem_modes(points, E, nu, h, k=20):
    from skfem import Basis, ElementHex1, ElementVector, MeshHex, BilinearForm, asm
    from skfem.helpers import dot
    from skfem.models.elasticity import linear_elasticity
    lo = points.min(0)-2*h
    idx = np.floor((points-lo)/h).astype(int)
    shape = idx.max(0)+3
    occ = np.zeros(shape, bool); occ[tuple(idx.T)] = True
    occ = ndimage.binary_closing(occ, iterations=1)
    occ = ndimage.binary_fill_holes(occ)
    vox = np.argwhere(occ)
    corners = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
    # skfem MeshHex local node order: (0,0,0),(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(1,1,1)
    order = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
    nodes_all = (vox[:, None, :]+order[None]).reshape(-1, 3)
    uniq, inv = np.unique(nodes_all, axis=0, return_inverse=True)
    t = inv.reshape(-1, 8).T
    mesh = MeshHex(np.ascontiguousarray((uniq*h+lo).T), np.ascontiguousarray(t))
    basis = Basis(mesh, ElementVector(ElementHex1()))
    lam, mu = E*nu/((1+nu)*(1-2*nu)), E/(2*(1+nu))
    K = asm(linear_elasticity(lam, mu), basis)
    @BilinearForm
    def mass(u, v, w):
        return dot(u, v)
    M = asm(mass, basis)
    from experiments.foam_fem_reference import cholesky
    sigma = -1e-3*E
    factor = cholesky((K-sigma*M).tocsc())                     # SPD: CHOLMOD instead of SuperLU
    op = sla.LinearOperator(K.shape, matvec=factor.solve, dtype=float)
    vals = sla.eigsh(K, k=k+6, M=M, sigma=sigma, which='LM', OPinv=op, return_eigenvectors=False)
    return np.sort(vals), dict(voxels=int(len(vox)), volume=float(len(vox)*h**3), dofs=int(basis.N))


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--dem', type=Path, default=Path('output/pacnerf_dem/elastic_0_f0'))
    a.add_argument('--h', type=float, nargs='+', default=[.008, .006])
    a.add_argument('--modes', type=int, default=16)
    args = a.parse_args()
    gt = json.loads(Path('data/pacnerf/gt/elastic.json').read_text())['0']
    rho = 1000.
    g = dict(np.load(args.dem/'geometry.npz'))
    model = build(g, gt['E'], gt['nu'], 'cpu', 'noslip', .25, .3, torch.float64)
    to_hz = lambda vals: np.sqrt(np.clip(vals[6:], 0, None))/(2*np.pi)
    out = dict(E=gt['E'], nu=gt['nu'])
    runs = dict(theta0=torch.zeros(model.n, 7, dtype=torch.float64),
                trained_shared=torch.tensor(json.loads((args.dem/'silhouette_v1/report.json').read_text())['best']['parameters'], dtype=torch.float64).expand(model.n, 7).clone(),
                trained_perbody=torch.tensor(np.load(args.dem/'perbody_smooth1e-2/theta_best.npz')['theta'], dtype=torch.float64))
    for name, theta in runs.items():
        vals, checks = dem_modes(model, theta, args.modes)
        out[f'dem_{name}'] = dict(rigid_eigs=vals[:6].tolist(), hz=to_hz(vals).tolist(), checks=checks)
        print('DEM', name, checks, 'rigid', np.round(vals[:6], 4), 'Hz', np.round(to_hz(vals), 2), flush=True)
    pts = read_ply('data/pacnerf/simulation_data/elastic/0/0.ply')
    for h in args.h:
        vals, info = fem_modes(pts, gt['E']/rho, gt['nu'], h, args.modes)
        out[f'fem_h{h}'] = dict(rigid_eigs=vals[:6].tolist(), hz=to_hz(vals).tolist(), **info)
        print('FEM h', h, info, 'rigid', np.round(vals[:6], 4), 'Hz', np.round(to_hz(vals), 2), flush=True)
    out['dem_volume'] = float(model.cell_volume.sum())
    dst = args.dem/'error_budget'; dst.mkdir(exist_ok=True)
    (dst/'modal_check.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
