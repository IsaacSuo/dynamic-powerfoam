"""Linearized implicit-midpoint trainer for the FEM-target cell fit.

Small-strain linearization of the finite four-bond (+ optional nodal tetra
volume) model about the stress-free reference: M q'' + K(theta) q = f, with the
same implicit-midpoint scheme as LinearBondsGPU.step and the same substeps,
loss, bounds, regularization and L-BFGS-B budgets as foam_fem_volume_inverse.

K(theta) = sum_b sum_m kbase_bm exp((theta_im+theta_jm)/2) G_bm
         + sum_i c k_v W_i exp(theta_i4) B_i^T B_i,
where G_bm are fixed 12x12 unit bond matrices and B_i = grad J_i at reference.
Per objective: one sparse factorization per clip, forward/adjoint back-solves,
and per-bond contractions for the parameter gradient. No dense Jacobians.

Valid only for small displacements about the reference; frozen parameters are
re-evaluated with the finite model when --finite-eval is given.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import torch
from scipy.optimize import minimize

from experiments.bonded_dem_linear_gpu import LinearBondsGPU
from experiments.foam_dem_moduli import write
from experiments.foam_fem_inverse import TRAIN, TEST
from experiments.foam_tetra_volume import NodalTetraVolume, grid_tetrahedra


def skew(r):
    s = np.zeros(r.shape[:-1]+(3, 3))
    s[..., 0, 1], s[..., 0, 2] = -r[..., 2], r[..., 1]
    s[..., 1, 0], s[..., 1, 2] = r[..., 2], -r[..., 0]
    s[..., 2, 0], s[..., 2, 1] = -r[..., 1], r[..., 0]
    return s


def _power_face_coefficients(geometry, clamp_face=False):
    """Yields (j, i, a) with d V_i = sum a . u_j over the cell's faces (see power_volume_rows).

    clamp_face: faces shared with a fixed cell carry the clamp value u_f = 0 (Dirichlet face,
    as in a finite-volume clamp) instead of the two-cell average, and fixed cells get no rows.
    """
    points = np.asarray(geometry['points'], float)
    fixed = np.asarray(geometry['fixed'], bool)
    axes = [np.unique(points[:, k]) for k in range(3)]
    index = {tuple(int(np.searchsorted(axes[k], p[k])) for k in range(3)): i for i, p in enumerate(points)}
    face = {}
    for (i, j), area, contact in zip(geometry['pairs'], geometry['areas'], geometry['contacts']):
        face[(int(i), int(j))] = face[(int(j), int(i))] = (float(area), np.asarray(contact, float))
    for key, i in index.items():
        if clamp_face and fixed[i]:
            continue
        for axis in range(3):
            for side in (-1, 1):
                nb = list(key); nb[axis] += side
                opp = list(key); opp[axis] -= side
                j = index.get(tuple(nb))
                if j is not None and clamp_face and fixed[j]:
                    continue  # u_f = 0 on the clamp face: no contribution
                if j is not None:
                    area, c = face[(i, j)]
                    d = np.linalg.norm(points[j]-points[i])
                    yield j, i, area/d*(points[j]-c)
                    yield i, i, area/d*(c-points[i])
                    continue
                o = index.get(tuple(opp))
                if o is None:
                    raise ValueError('Ghost extrapolation needs an opposite neighbour.')
                area, c_opp = face[(i, o)]
                x_g, c_g = 2*points[i]-points[o], 2*points[i]-c_opp
                d = np.linalg.norm(x_g-points[i])
                g = area/d*(x_g-c_g)          # coefficient on the ghost displacement
                yield i, i, area/d*(c_g-points[i])+2*g
                yield o, i, -g


def power_volume_rows(geometry, clamp_face=False):
    """Linearized power-cell volume change, one row per cell over all 6n DOFs.

    For a Voronoi/equal-weight power cell, d V_i / d x_j = (A_ij/d_ij)(x_j - c_ij) for a
    neighbour j and d V_i / d x_i = sum_j (A_ij/d_ij)(c_ij - x_i), with A the shared face
    area and c its centroid. Faces toward air (no material neighbour along a lattice
    axis) use a ghost cell mirrored through the cell centre whose displacement is the
    linear extrapolation 2 u_i - u_opp, so uniform strains give J_i - 1 = tr(eps) exactly
    for every cell. Cartesian lattices only (the ghost needs the opposite neighbour).
    Returns (rows as a sparse n x 6n matrix, cell volumes).
    """
    n = len(geometry['points'])
    rows, cols, vals = [], [], []
    for j, i, a in _power_face_coefficients(geometry, clamp_face):
        for k in range(3):
            rows.append(i); cols.append(6*j+k); vals.append(a[k])
    B = sp.coo_matrix((vals, (rows, cols)), shape=(n, 6*n)).tocsr()
    return B, np.asarray(geometry['masses'], float)  # unit density: masses are cell volumes


def power_gradient_rows(geometry, clamp_face=False):
    """Linearized cell displacement gradient from the same faces and air ghosts.

    G_i = (1/V_i) sum_j u_j (x) a_ij, row 9 i + 3 p + q holding G_i[p, q] = du_p/dx_q.
    On the cube lattice a_ij = (A/2) n, i.e. Green-Gauss with face-averaged displacement;
    tr G_i equals the power volume row / V_i, and affine fields give G_i = H exactly.
    """
    n = len(geometry['points'])
    volume = np.asarray(geometry['masses'], float)
    rows, cols, vals = [], [], []
    for j, i, a in _power_face_coefficients(geometry, clamp_face):
        for p_ in range(3):
            for q in range(3):
                rows.append(9*i+3*p_+q); cols.append(6*j+p_); vals.append(a[q]/volume[i])
    return sp.coo_matrix((vals, (rows, cols)), shape=(9*n, 6*n)).tocsr()


class LinearizedFoam:
    def __init__(self, geometry, volume=False, k_v=400., deviatoric=False, k_d=400., volume_mode='tet',
                 shape_mode='tet', clamp_face=False):
        bonds = LinearBondsGPU(geometry['points'], geometry['radii'], geometry['pairs'], geometry['areas'],
                               geometry['contacts'], geometry['masses'], geometry['fixed'], device='cpu')
        numpy = lambda t: t.detach().cpu().numpy()
        self.points = numpy(bonds.p)
        self.n = len(self.points)
        self.i, self.j = numpy(bonds.i), numpy(bonds.j)
        # Same order as STIFFNESS_NAMES in bonded_dem_differentiable: kn, ks, kb, kt.
        self.base = np.stack([numpy(getattr(bonds, k)) for k in ('kn', 'ks', 'kb', 'kt')], axis=1)
        normal, ri, rj = numpy(bonds.n), numpy(bonds.ri), numpy(bonds.rj)
        eye = np.broadcast_to(np.eye(3), (len(normal), 3, 3))
        zero = np.zeros_like(eye)
        # du = D q_local, dw = E q_local with q_local = (u_i, w_i, u_j, w_j).
        D = np.concatenate((-eye, skew(ri), eye, -skew(rj)), axis=2)
        E = np.concatenate((zero, -eye, zero, eye), axis=2)
        nn = normal[:, :, None]*normal[:, None, :]
        tt = eye-nn
        dT = D.transpose(0, 2, 1)
        eT = E.transpose(0, 2, 1)
        self.unit = np.stack((dT@nn@D, dT@tt@D, eT@tt@E, eT@nn@E), axis=1)  # (B, 4, 12, 12)
        self.dofs = np.concatenate((6*self.i[:, None]+np.arange(6), 6*self.j[:, None]+np.arange(6)), axis=1)
        self.mass = numpy(bonds.mass).ravel()
        self.fixed = np.asarray(geometry['fixed'], bool)
        self.free = np.flatnonzero(~np.repeat(self.fixed, 6))
        rows = np.repeat(self.dofs[:, :, None], 12, axis=2)
        self._rows, self._cols = rows.ravel(), rows.transpose(0, 2, 1).ravel()
        self.parameter_count = 4+int(volume)+int(deviatoric)
        self.volume_B = None
        self.shape_D = None
        if volume or deviatoric:
            term = NodalTetraVolume(geometry['points'], grid_tetrahedra(geometry['points'], geometry['radii']),
                                    float(np.sum(geometry['masses'])), k_v)
            self.volume_term = term
            _, gradients = term.volumes_and_gradients(term.reference)
            tets, g = numpy(term.tetrahedra), numpy(gradients)
        if deviatoric and shape_mode == 'power':
            # Exact power-cell strain: eps_i = sym G_i from power_gradient_rows (air ghosts included).
            # Orthonormal symmetric coordinates s_m = E_m : G_i, deviatoric projection P;
            # E_shape = sum_i k_d exp(theta_i) V_i |dev eps_i|^2 -> affine (4/3, -2/3, 1) k_d.
            basis = np.zeros((6, 3, 3))
            for m in range(3):
                basis[m, m, m] = 1
            for m, (a, b) in enumerate(((1, 2), (0, 2), (0, 1)), start=3):
                basis[m, a, b] = basis[m, b, a] = 2**-.5
            identity = np.array([1., 1, 1, 0, 0, 0])/np.sqrt(3)
            P = (np.eye(6)-np.outer(identity, identity))@basis.reshape(6, 9)
            self.shape_D = sp.kron(sp.identity(self.n), P).tocsr()@power_gradient_rows(geometry, clamp_face)
            self.shape_base = 2*k_d*np.asarray(geometry['masses'], float)  # Hessian weight
            self.shape_column = self.parameter_count-1
        elif deviatoric:
            # Nodal small strain eps_i = sum_{T contains i} sym(sum_a u_a (x) g_Ta) / S_i, with
            # g_Ta = dV_T/dx_a (so V_T * grad N_a), i.e. the volume-weighted average of linear-tet
            # strains, the same averaging as the volume term (whose row is tr eps_i).
            # Orthonormal symmetric basis E_m; D = P_dev B maps all DOFs to deviatoric coordinates.
            # E_shape = c * sum_i k_d exp(theta_i) W_i |dev eps_i|^2 -> affine shear modulus k_d.
            basis = np.zeros((6, 3, 3))
            for m in range(3):
                basis[m, m, m] = 1
            for m, (a, b) in enumerate(((1, 2), (0, 2), (0, 1)), start=3):
                basis[m, a, b] = basis[m, b, a] = 2**-.5
            coeff = np.einsum('mpq,tvq->tvmp', basis, g)                      # (T, vertex, m, k)
            owner = np.broadcast_to(tets[:, :, None, None, None], (len(tets), 4, 4, 6, 3))
            row = 6*owner+np.arange(6)[None, None, None, :, None]
            col = 6*tets[:, None, :, None, None]+np.arange(3)[None, None, None, None, :]
            col = np.broadcast_to(col, row.shape)
            val = np.broadcast_to(coeff[:, None], row.shape)
            B = sp.coo_matrix((val.ravel(), (row.ravel(), col.ravel())), shape=(6*self.n, 6*self.n)).tocsr()
            B = sp.diags(np.repeat(1/numpy(term.nodal_reference_sum), 6))@B
            identity = np.array([1., 1, 1, 0, 0, 0])/np.sqrt(3)
            P = np.eye(6)-np.outer(identity, identity)
            self.shape_D = sp.kron(sp.identity(self.n), P).tocsr()@B
            self.shape_base = 2*float(term.normalization)*k_d*numpy(term.weights)  # Hessian weight
            self.shape_column = self.parameter_count-1
        if volume and volume_mode == 'power':
            B, cell_volume = power_volume_rows(geometry, clamp_face)
            # E = 1/2 k_v m_i V_i (J_i - 1)^2 with J_i - 1 = (row_i . u) / V_i; cells tile the material.
            self.volume_B = sp.diags(1/cell_volume)@B
            self.volume_base = k_v*cell_volume
        elif volume:
            r = np.repeat(tets[:, :, None, None], 4, axis=2).repeat(3, axis=3)       # owner
            c = 6*tets[:, None, :, None]+np.arange(3)                                # vertex position dof
            c = np.broadcast_to(c, r.shape)
            v = np.broadcast_to(g[:, None, :, :], r.shape)
            B = sp.coo_matrix((v.ravel(), (r.ravel(), c.ravel())), shape=(self.n, 6*self.n)).tocsr()
            self.volume_B = sp.diags(1/numpy(term.nodal_reference_sum))@B
            self.volume_base = float(term.normalization)*k_v*numpy(term.weights)

    def bond_scale(self, theta):
        return self.base*np.exp(.5*(theta[self.i, :4]+theta[self.j, :4]))

    def stiffness(self, theta):
        """Full 6n x 6n reference tangent (sparse CSR)."""
        blocks = np.einsum('bm,bmij->bij', self.bond_scale(theta), self.unit)
        K = sp.coo_matrix((blocks.ravel(), (self._rows, self._cols)), shape=(6*self.n,)*2).tocsr()
        if self.volume_B is not None:
            weight = self.volume_base*np.exp(theta[:, 4])
            K = K+self.volume_B.T@sp.diags(weight)@self.volume_B
        if self.shape_D is not None:
            weight = np.repeat(self.shape_base*np.exp(theta[:, self.shape_column]), 6)
            K = K+self.shape_D.T@sp.diags(weight)@self.shape_D
        return K

    def loads(self, data, substeps):
        count = len(data['target'])-1
        dt = data['dt']/substeps
        times = (np.arange(count*substeps)+.5)*dt
        pulse = np.where(times < data['pulse'], np.sin(np.pi*times/data['pulse'])**2, 0.)
        return pulse, np.asarray(data['pattern']).ravel()[self.free], dt

    def simulate(self, theta, data, substeps, gradient=False):
        """Returns normalized MSE loss, frame positions, and optionally d loss / d theta."""
        K = self.stiffness(theta)
        Kf = K[self.free][:, self.free].tocsc()
        loss, positions, adjoint = self.dynamics(Kf, self.mass[self.free], data, substeps, gradient)
        if not gradient:
            return loss, positions, None
        steps, h = len(adjoint['mu']), adjoint['h']
        # dL/dK = -h sum_k mu_k w_k^T; contract with each parameter direction.
        mu_full = np.zeros((steps, 6*self.n)); mu_full[:, self.free] = adjoint['mu']
        w_full = np.zeros((steps, 6*self.n)); w_full[:, self.free] = adjoint['w']
        C = np.einsum('tbi,tbj->bij', mu_full[:, self.dofs], w_full[:, self.dofs])
        bond = -h*np.einsum('bmij,bij->bm', self.unit, C)*self.bond_scale(theta)
        grad = np.zeros_like(theta)
        np.add.at(grad[:, :4], self.i, .5*bond)
        np.add.at(grad[:, :4], self.j, .5*bond)
        if self.volume_B is not None:
            weight = self.volume_base*np.exp(theta[:, 4])
            bm = (self.volume_B@mu_full.T)
            bw = (self.volume_B@w_full.T)
            grad[:, 4] = -h*weight*np.sum(bm*bw, axis=1)
        if self.shape_D is not None:
            weight = self.shape_base*np.exp(theta[:, self.shape_column])
            dm = (self.shape_D@mu_full.T).reshape(self.n, 6, -1)
            dw = (self.shape_D@w_full.T).reshape(self.n, 6, -1)
            grad[:, self.shape_column] = -h*weight*np.sum(dm*dw, axis=(1, 2))
        return loss, positions, grad

    def dynamics(self, Kf, M, data, substeps, gradient=False):
        """Implicit-midpoint rollout for any free-dof stiffness Kf and diagonal mass M.

        With gradient=True also returns the adjoint record: dL/dK = -h sum_k mu_k w_k^T
        and dL/dM (diagonal).
        """
        free = self.free
        pulse, pattern, dt = self.loads(data, substeps)
        steps = len(pulse)
        lu = sla.splu((sp.diags(M)+.25*dt*dt*Kf).tocsc())
        nf = len(free)
        U = np.zeros((steps+1, nf))
        Y = np.zeros((steps, nf))
        V = np.zeros((steps, nf))
        u, v = np.zeros(nf), np.zeros(nf)
        h = .5*dt*dt
        for k in range(steps):
            V[k] = v
            y = lu.solve(dt*M*v-h*(Kf@u)+h*pulse[k]*pattern)
            Y[k] = y
            u = u+y
            v = 2*y/dt-v
            U[k+1] = u
        # Frame positions of all cells (fixed cells stay at reference).
        full = np.zeros((len(range(0, steps+1, substeps)), 6*self.n))
        full[:, free] = U[::substeps]
        positions = self.points[None]+full.reshape(len(full), self.n, 6)[:, :, :3]
        target = np.asarray(data['target'])
        cells = ~self.fixed
        scale2 = max(np.mean((target[1:, cells]-self.points[cells])**2), 1e-30)
        residual = positions[1:, cells]-target[1:, cells]
        loss = float(np.mean(residual**2)/scale2)
        if not gradient:
            return loss, positions, None
        # d loss / d positions at observed frames, mapped to free dofs.
        dpos = np.zeros((len(positions), self.n, 6))
        dpos[1:, cells, :3] = 2*residual/(residual.size*scale2)
        a = dpos.reshape(len(positions), -1)[:, free]
        ubar = np.zeros(nf); vbar = np.zeros(nf)
        Mu = np.zeros((steps, nf))
        if steps % substeps:
            raise ValueError('Steps must be a multiple of substeps.')
        ubar = a[-1].copy()
        for k in range(steps-1, -1, -1):
            mu = lu.solve(ubar+(2/dt)*vbar)  # A is symmetric
            Mu[k] = mu
            ubar = ubar-h*(Kf@mu)
            vbar = -vbar+dt*M*mu
            if k % substeps == 0 and k:
                ubar = ubar+a[k//substeps]
        # A y = dt M v - h K u + h f, A = M + h/2 K: dL/dM = sum_k mu_k (dt v_k - y_k).
        mass_grad = np.sum(Mu*(dt*V-Y), axis=0)
        return loss, positions, dict(mu=Mu, w=U[:-1]+.5*Y, h=h, mass=mass_grad)


def load_case(reference, name):
    with np.load(reference/f'{name}_target.npz') as a:
        return dict(pattern=a['force_pattern'], target=a['positions'],
                    pulse=float(a['pulse_duration']), dt=float(a['dt']))


def finite_positions(model, theta, data, substeps):
    """Forward-only rollout of the finite model (DifferentiableBonds) on the same loads."""
    tensor = model.reference.tensor
    count = len(data['target'])-1
    dt = data['dt']/substeps
    times = (np.arange(count*substeps)+.5)*dt
    pulse = np.where(times < data['pulse'], np.sin(np.pi*times/data['pulse'])**2, 0.)
    loads = tensor(pulse)[:, None, None]*tensor(data['pattern'])[None]
    with torch.no_grad():
        positions, _ = model.rollout(tensor(theta), loads, dt)
    return positions[::substeps].numpy()


def normalized_rmse(prediction, target, initial, cells):
    scale2 = max(np.mean((target[1:, cells]-initial[cells])**2), 1e-30)
    return float(np.sqrt(np.mean((prediction[1:, cells]-target[1:, cells])**2)/scale2))


def projected_gradient(x, gradient, bound):
    return x-np.clip(x-gradient, -bound, bound)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, default=Path('output/foam_fem/reference_edge007'))
    parser.add_argument('--old-inverse', type=Path, default=Path('output/foam_fem/inverse_exploratory_v1'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max-iterations', type=int, default=120)
    parser.add_argument('--max-evaluations', type=int, default=240)
    parser.add_argument('--multiplier-bound', type=float, default=20.)
    parser.add_argument('--gradient-tolerance', type=float, default=1e-6)
    parser.add_argument('--k-v', type=float, default=400.)
    parser.add_argument('--finite-eval', action='store_true',
                        help='Re-evaluate frozen parameters with the finite nonlinear model (slow).')
    args = parser.parse_args()
    torch.set_num_threads(1)
    reference = json.loads((args.reference/'report.json').read_text())
    assert reference['complete'], 'Incomplete FEM data.'
    old_config = json.loads((args.old_inverse/'config.json').read_text())
    with np.load(args.reference/'patch.npz') as a:
        geometry = dict(a)
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    def status(stage, **details):
        row = dict(stage=stage, wall_elapsed_seconds=time.perf_counter()-started, updated_unix=time.time(), **details)
        write(args.out/'status.json', row)
        print(json.dumps(row), flush=True)
    models = {name: LinearizedFoam(geometry, volume=name == 'five', k_v=args.k_v) for name in ('four', 'five')}
    n = len(geometry['points'])
    datasets = {name: load_case(args.reference, name) for name in TRAIN}
    steps = {name: int(old_config['internal_substeps'][name]) for name in TRAIN}
    bound = float(np.log(args.multiplier_bound))
    regularization = 1e-6
    config = dict(reference=str(args.reference), model='linearized about stress-free reference',
        training_cases=list(TRAIN), held_out_cases=list(TEST), k_v=args.k_v, internal_substeps=steps,
        multiplier_bounds=[1/args.multiplier_bound, args.multiplier_bound],
        initialization='All log multipliers zero.', optimizer='SciPy L-BFGS-B, direct log multipliers',
        max_iterations=args.max_iterations, max_evaluations=args.max_evaluations,
        gradient_tolerance=args.gradient_tolerance,
        regularization=f'{regularization} * sum(theta**2) / {4*n}',
        protocol='Matches foam_fem_volume_inverse except the dynamics are linearized. '
                 'Held-out targets are read only after both fits freeze.')
    write(args.out/'config.json', config)

    try:
        time_checks = []
        for kind, model in models.items():
            theta = np.zeros((n, model.parameter_count))
            for name in TRAIN:
                _, coarse, _ = model.simulate(theta, datasets[name], steps[name])
                _, fine, _ = model.simulate(theta, datasets[name], 2*steps[name])
                time_checks.append(dict(model=kind, case=name, substeps=steps[name],
                    normalized_difference=normalized_rmse(coarse, fine, model.points, ~model.fixed)))
        write(args.out/'initial_time_checks.json', time_checks)

        fits = {}
        for kind, model in models.items():
            folder = args.out/kind
            folder.mkdir()
            shape = (n, model.parameter_count)
            evaluations, iterations, cache = [], [], {}
            fit_start = time.perf_counter()
            def value_gradient(vector):
                theta = vector.reshape(shape)
                losses, gradient = [], np.zeros(shape)
                for name in TRAIN:
                    loss, _, grad = model.simulate(theta, datasets[name], steps[name], gradient=True)
                    losses.append(loss)
                    gradient += grad/len(TRAIN)
                value = float(np.mean(losses))+regularization*float(np.sum(theta**2))/(4*n)
                gradient = (gradient+2*regularization*theta/(4*n)).ravel()
                row = dict(evaluation=len(evaluations), elapsed_seconds=time.perf_counter()-fit_start,
                    objective=value, train_rmse=float(np.sqrt(np.mean(losses))),
                    per_case_rmse=dict(zip(TRAIN, np.sqrt(losses).tolist())),
                    projected_gradient_max=float(np.max(np.abs(projected_gradient(vector, gradient, bound)))))
                evaluations.append(row)
                with (folder/'training.jsonl').open('a') as handle:
                    handle.write(json.dumps(row)+'\n')
                cache.update(x=vector.copy(), gradient=gradient.copy(), row=row)
                return value, gradient

            def checkpoint(vector):
                if not np.array_equal(cache['x'], vector):
                    value_gradient(vector)
                iterations.append(dict(iteration=len(iterations)+1, **cache['row']))

            fitted = minimize(value_gradient, np.zeros(np.prod(shape)), jac=True, method='L-BFGS-B',
                bounds=[(-bound, bound)]*int(np.prod(shape)), callback=checkpoint,
                options=dict(maxiter=args.max_iterations, maxfun=args.max_evaluations,
                             gtol=args.gradient_tolerance, ftol=0., maxls=20, maxcor=10))
            if not np.array_equal(cache['x'], fitted.x):
                value_gradient(fitted.x)
            pg = float(np.max(np.abs(projected_gradient(fitted.x, cache['gradient'], bound))))
            result = dict(converged=pg <= args.gradient_tolerance, stop_message=str(fitted.message),
                iterations=int(fitted.nit), evaluations=len(evaluations), final_objective=cache['row'],
                projected_gradient_max=pg, multiplier_range=np.exp([fitted.x.min(), fitted.x.max()]).tolist(),
                fraction_near_bound=float(np.mean(np.abs(fitted.x) > .99*bound)),
                elapsed_seconds=time.perf_counter()-fit_start)
            np.savez_compressed(folder/'parameters.npz', theta=fitted.x.reshape(shape))
            write(folder/'checkpoints.json', iterations)
            write(folder/'fit.json', result)
            fits[kind] = result
            status('fit_frozen', model=kind, converged=result['converged'], iterations=result['iterations'],
                   train_rmse=cache['row']['train_rmse'], seconds=result['elapsed_seconds'])

        # Held-out targets are opened only after both fits are frozen.
        for name in TEST:
            datasets[name] = load_case(args.reference, name)
        finite_models = {}
        if args.finite_eval:
            from experiments.foam_tetra_dynamics_probe import make_model
            finite_models = {kind: make_model(geometry, volume=kind == 'five', k_v=args.k_v) for kind in models}
        for kind, model in models.items():
            folder = args.out/kind
            theta = np.load(folder/'parameters.npz')['theta']
            metrics = {}
            for name in TRAIN+TEST:
                count = steps[name] if name in TRAIN else max(steps.values())
                data = datasets[name]
                _, linear, _ = model.simulate(theta, data, count)
                _, finer, _ = model.simulate(theta, data, 2*count)
                cells = ~model.fixed
                metric = dict(linear_rmse=normalized_rmse(linear, data['target'], model.points, cells),
                    linear_time_refinement=normalized_rmse(linear, finer, model.points, cells))
                saved = dict(linear=linear, target=data['target'])
                if kind in finite_models:
                    status('finite_evaluation', model=kind, case=name)
                    finite = finite_positions(finite_models[kind], theta, data, count)
                    metric.update(finite_rmse=normalized_rmse(finite, data['target'], model.points, cells),
                                  linear_vs_finite=normalized_rmse(linear, finite, model.points, cells))
                    saved['finite'] = finite
                metrics[name] = metric
                np.savez_compressed(folder/f'{name}.npz', **saved)
                write(folder/'metrics.json', metrics)
            fits[kind]['metrics'] = metrics
            write(folder/'report.json', fits[kind])
        write(args.out/'report.json', dict(complete=True, config=config, time_checks=time_checks, models=fits))
        status('complete')
    except BaseException as error:
        status('failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
