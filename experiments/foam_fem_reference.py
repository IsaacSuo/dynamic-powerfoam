"""Independent P2 linear-elastic FEM targets on the shared power-foam solid.

scikit-fem assembles continuum K and consistent M. This file implements the
implicit midpoint time integrator and a conservative load transfer to DEM.
No DEM force law or stiffness matrix enters the reference trajectory.

Linear systems (K and M + dt^2/4 K, both SPD on the free DOFs) use CHOLMOD sparse
Cholesky from scikit-sparse (installed in output/fem_dependencies). For speed, run
with an optimized BLAS, e.g. LD_PRELOAD=$CONDA/lib/libmkl_rt.so.2 MKL_THREADING_LAYER=GNU.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.sparse.linalg import LinearOperator, eigsh
from sksparse.cholmod import CholeskyFactor
from skfem import MeshTet2, Basis, FacetBasis, ElementVector, ElementTetP2, BilinearForm, LinearForm, asm
from skfem.helpers import dot
from skfem.models.elasticity import linear_elasticity, lame_parameters

CASES = ('train_down', 'train_pull', 'train_twist', 'test_side', 'test_local')
# Fresh held-out loads fixed on 2026-10-08 before any model was evaluated on them.
# end: outer surface with x > 3h; mid_top / mid_side: outer surface of the x = 2h
# layer facing +z / -y. Tractions are per unit area before amplitude calibration.
FRESH = ('fresh_oblique', 'fresh_local_side', 'fresh_end_bend_y', 'fresh_end_bend_z',
         'fresh_mid_top_down', 'fresh_mid_side')
CASE_SETS = dict(original=CASES, fresh_v1=FRESH, all_v1=CASES+FRESH)
REGION = dict(train_down='end', train_pull='end', train_twist='end', test_side='end', test_local='end',
              fresh_oblique='end', fresh_local_side='end', fresh_end_bend_y='end', fresh_end_bend_z='end',
              fresh_mid_top_down='mid_top', fresh_mid_side='mid_side')


def traction_field(name, xyz):
    traction = np.zeros_like(xyz)
    local = ((xyz[1] > 0) & (xyz[2] > 0)).astype(float)
    if name == 'train_down': traction[2] = -1
    if name == 'train_pull': traction[0] = 1
    if name == 'train_twist': traction[1], traction[2] = -xyz[2], xyz[1]
    if name == 'test_side': traction[1] = 1
    if name == 'test_local': traction[2] = -local
    if name == 'fresh_oblique': traction[1], traction[2] = 2**-.5, -2**-.5
    if name == 'fresh_local_side': traction[1] = local
    if name == 'fresh_end_bend_y': traction[0] = xyz[2]   # zero-net-force couple about y
    if name == 'fresh_end_bend_z': traction[0] = xyz[1]   # zero-net-force couple about z
    if name == 'fresh_mid_top_down': traction[2] = -1
    if name == 'fresh_mid_side': traction[1] = 1
    if not np.any(traction):
        raise ValueError(f'Unknown or empty load case {name}')
    return traction


def material_probe(basis, points):
    """Locate each material point in a curved element, then use its P2 basis.

    The library's batched finder also inverts far-away candidate elements and
    can fail there. Here candidates are inverted individually and rejected if
    outside; this is still interpolation at the exact point, not a nearest node.
    """
    centroids = basis.mesh.p[:, basis.mesh.t].mean(axis=1).T
    tree = cKDTree(centroids)
    rows, cols, values = [], [], []
    for index, point in enumerate(points):
        _, candidates = tree.query(point, k=min(128, len(centroids)))
        for cell in np.atleast_1d(candidates):
            cells = np.array([cell])
            try:
                local = basis.mapping.invF(point[:, None, None], tind=cells)
            except Exception as error:
                if 'Newton iteration' not in str(error):
                    raise
                continue
            if np.min(local) < -1e-8 or local.sum() > 1+1e-8:
                continue
            assert np.max(abs(basis.mapping.F(local, tind=cells).ravel()-point)) < 1e-9
            for j in range(basis.Nbfun):
                vector = np.asarray(basis.elem.gbasis(basis.mapping, local, j, tind=cells)[0]).ravel()
                for component, value in enumerate(vector):
                    rows.append(component*len(points)+index)
                    cols.append(basis.element_dofs[j, cell])
                    values.append(value)
            break
        else:
            raise RuntimeError(f'No curved FEM element contains material point {index}: {point}')
    return sparse.coo_matrix((values, (rows, cols)), shape=(3*len(points), basis.N)).tocsr()


def cholesky(matrix):
    factor = CholeskyFactor(matrix)
    factor.factorize(matrix)
    return factor


def normalized_difference(a, b, points, fixed):
    return np.sqrt(np.mean((a[1:, ~fixed]-b[1:, ~fixed])**2)/
                   np.mean((b[1:, ~fixed]-points[~fixed])**2))


class Reference:
    def __init__(self, geometry, cases=CASES):
        self.cases = tuple(cases)
        self.g = dict(np.load(geometry))
        self.points = self.g['points']
        self.mesh = MeshTet2(self.g['vertices'].T, self.g['tetrahedra'].T)
        self.basis = Basis(self.mesh, ElementVector(ElementTetP2()), intorder=5)
        b = self.basis
        self.free = np.flatnonzero(b.doflocs[0] > .5+1e-9)
        self.probe = material_probe(b, self.points)
        k = asm(linear_elasticity(*lame_parameters(1200., .25)), b).tocsc()

        @BilinearForm
        def mass(u, v, w):
            return dot(u, v)
        m = asm(mass, b).tocsc()
        self.K, self.M = k[self.free][:, self.free].tocsc(), m[self.free][:, self.free].tocsc()
        # Check vector DOF/probe ordering and rigid-motion null response.
        translation = np.tile([.3, -.2, .1], b.N//3)
        self.n = len(self.points)
        expected = np.tile([.3, -.2, .1], (self.n, 1))
        assert np.max(abs(self.center_values(translation)-expected)) < 1e-10
        rotation = np.zeros(b.N)
        rotation[0::3] = -b.doflocs[1, 0::3]
        rotation[1::3] = b.doflocs[0, 1::3]
        rigid_error = np.linalg.norm(k@rotation)/(np.linalg.norm(k.data)*np.linalg.norm(rotation))
        assert rigid_error < 1e-12
        strain_diagonal = np.array([.01, -.005, .002])
        affine = np.zeros(b.N)
        for component in range(3):
            affine[component::3] = strain_diagonal[component]*b.doflocs[component, component::3]
        lam, mu = lame_parameters(1200., .25)
        expected_energy = b.dx.sum()*(.5*lam*strain_diagonal.sum()**2+mu*np.sum(strain_diagonal**2))
        affine_error = abs(float(.5*affine@(k@affine))/expected_energy-1)
        assert affine_error < 1e-10, affine_error
        assert np.max(abs(self.center_values(affine)-self.points*strain_diagonal)) < 1e-10
        expected_mass_form = b.dx.sum()*np.dot([.3, -.2, .1], [.3, -.2, .1])
        mass_error = abs(float(translation@(m@translation))/expected_mass_form-1)
        assert mass_error < 1e-10, mass_error
        facets = self.mesh.boundary_facets()
        mean = self.mesh.p[:, self.mesh.facets[:, facets]].mean(axis=1)
        middle = (mean[0] > 1.5) & (mean[0] < 2.5)
        # Outer surface beyond the outermost centre rows (bar: |y|,|z| > 0.5; 4x4 block: > 1.5).
        top, side = self.points[:, 2].max(), self.points[:, 1].min()
        selections = dict(end=mean[0] > 3.+1e-9, mid_top=middle & (mean[2] > top), mid_side=middle & (mean[1] < side))
        regions = {}
        for region in sorted({REGION[name] for name in self.cases}):
            fb = FacetBasis(self.mesh, b.elem, facets=facets[selections[region]], intorder=5)
            xyz = np.asarray(fb.global_coordinates())
            owner = np.argmin(np.sum((xyz[:, :, :, None]-self.points.T[:, None, None, :])**2, axis=0), axis=-1)
            regions[region] = (fb, xyz, owner)
        loads, dem, transfer_checks = [], [], {}
        for name in self.cases:
            fb, xyz, owner = regions[REGION[name]]
            traction = traction_field(name, xyz)
            @LinearForm
            def load(v, w):
                return dot(traction, v)
            assembled = asm(load, fb)
            nodal = assembled.reshape(-1, 3)
            total_force = nodal.sum(axis=0)
            total_torque = np.cross(b.doflocs[:, ::3].T, nodal).sum(axis=0)
            cell_load = np.zeros((self.n, 6))
            f = (traction*fb.dx[None]).reshape(3, -1).T
            application = xyz.reshape(3, -1).T
            ids = owner.ravel()
            np.add.at(cell_load[:, :3], ids, f)
            np.add.at(cell_load[:, 3:], ids, np.cross(application-self.points[ids], f))
            dem_force = cell_load[:, :3].sum(axis=0)
            dem_torque = (cell_load[:, 3:]+np.cross(self.points, cell_load[:, :3])).sum(axis=0)
            mismatch = max(np.max(abs(dem_force-total_force)), np.max(abs(dem_torque-total_torque)))
            assert mismatch < 1e-10, (name, mismatch)
            transfer_checks[name] = dict(force=total_force.tolist(), torque=total_torque.tolist(),
                                          max_absolute_mismatch=float(mismatch), region=REGION[name],
                                          loaded_cells=np.flatnonzero(np.abs(cell_load).sum(1) > 0).tolist())
            loads.append(assembled[self.free]); dem.append(cell_load)
        self.loads = np.stack(loads, axis=1)
        self.dem_loads = np.stack(dem)
        self.checks = dict(rigid_rotation_residual=float(rigid_error), load_transfer=transfer_checks,
            affine_strain_energy_relative_error=float(affine_error), constant_velocity_mass_relative_error=float(mass_error),
            dofs=int(b.N), free_dofs=len(self.free), elements=int(self.mesh.nelements),
            volume_relative_error=float(abs(b.dx.sum()/self.g['masses'].sum()-1)))
        print('Assembled', json.dumps(self.checks), flush=True)

    def center_values(self, full):
        value = self.probe@full
        if value.ndim == 1:
            return value.reshape(3, self.n).T
        return value.reshape(3, self.n, -1).transpose(2, 1, 0)

    def full(self, reduced):
        value = np.zeros((self.basis.N,)+reduced.shape[1:])
        value[self.free] = reduced
        return value

    def calibrate(self):
        stiffness = cholesky(self.K)
        inverse = LinearOperator(self.K.shape, matvec=stiffness.solve, dtype=float)
        # Shift-invert at sigma=0 needs only K^-1, supplied by the Cholesky factor.
        eigen = eigsh(self.K, M=self.M, k=1, sigma=0., which='LM', OPinv=inverse, return_eigenvectors=False)[0]
        period = float(2*np.pi/np.sqrt(eigen))
        static = stiffness.solve(self.loads)
        centers = self.center_values(self.full(static))
        amplitudes = .005*4.8/np.linalg.norm(centers, axis=-1).max(axis=1)
        strain_peaks = []
        for col in range(len(self.cases)):
            gradient = self.basis.interpolate(self.full(static[:, col]*amplitudes[col])).grad
            strain = .5*(gradient+gradient.swapaxes(0, 1))
            strain_peaks.append(float(np.sqrt(np.sum(strain*strain, axis=(0, 1))).max()))
        # Conservative small-strain guard, including the doubled-down test.
        maximum = max(2*strain_peaks[0], *strain_peaks[1:])
        common_factor = min(1., .01/maximum)
        amplitudes *= common_factor
        return dict(period=period, pulse_duration=period/2, duration=1.5*period,
            dt=1.5*period/24, frames=24, amplitudes=amplitudes.tolist(),
            initial_static_max_strains=strain_peaks, common_amplitude_reduction=common_factor,
            calibration='Only continuum static displacement/strain used, before any DEM fitting.',
            load_cases=list(self.cases), material=dict(E=1200., nu=.25, rho=1.), model_length=4.8)

    def rollout(self, config, substeps):
        start = time.perf_counter()
        dt = config['dt']/substeps
        system = self.M+.25*dt*dt*self.K
        factor = cholesky(system)
        force = self.loads*np.array(config['amplitudes'])[None]
        q = np.zeros_like(force); v = q.copy()
        count = len(self.cases)
        positions = [np.broadcast_to(self.points, (count, self.n, 3)).copy()]
        work = np.zeros(count)
        strain_peak = np.zeros(count)
        for step in range(config['frames']*substeps):
            t = (step+.5)*dt
            pulse = np.sin(np.pi*t/config['pulse_duration'])**2 if t < config['pulse_duration'] else 0.
            external = pulse*force
            rhs = dt*(self.M@v)+.5*dt*dt*(external-self.K@q)
            dq = factor.solve(rhs)
            if step == 0:
                defect = np.linalg.norm(system@dq-rhs)/np.linalg.norm(rhs)
                assert defect < 1e-9, defect
            work += np.sum(external*dq, axis=0)
            v = 2*dq/dt-v
            q += dq
            if (step+1) % substeps == 0:
                full = self.full(q)
                positions.append(self.points+self.center_values(full))
                for col in range(len(self.cases)):
                    grad = self.basis.interpolate(full[:, col]).grad
                    eps = .5*(grad+grad.swapaxes(0, 1))
                    strain_peak[col] = max(strain_peak[col], float(np.sqrt(np.sum(eps*eps, axis=(0, 1))).max()))
                if (step+1)//substeps % 8 == 0:
                    print('FEM progress', substeps, (step+1)//substeps, '/24', flush=True)
        energy = .5*np.sum(q*(self.K@q)+v*(self.M@v), axis=0)
        energy_error = np.max(abs(energy-work)/np.maximum(abs(work), 1e-30))
        assert energy_error < 1e-8, energy_error
        array = np.stack(positions, axis=1)
        assert np.max(abs(array[:, :, self.g['fixed']]-self.points[self.g['fixed']])) < 1e-12
        assert np.isfinite(array).all()
        metrics = dict(substeps=substeps, elapsed_seconds=time.perf_counter()-start,
            linear_solver='CHOLMOD sparse Cholesky', first_step_linear_relative_residual=float(defect),
            energy_work_max_relative_error=float(energy_error), peak_sampled_strains=strain_peak.tolist(),
            doubled_down_peak_strain=float(2*strain_peak[0]))
        print('Rollout', json.dumps(metrics), flush=True)
        return {name: array[i] for i, name in enumerate(self.cases)}, metrics

    def amplitudes_for(self, config):
        """Amplitudes for cases not in config, with the shared calibration rule and reduction.

        Same rule as calibrate(): static max center displacement 0.5% of the model
        length, times the stored common reduction. Timing is never changed.
        """
        known = dict(zip(config['load_cases'], config['amplitudes']))
        missing = [i for i, name in enumerate(self.cases) if name not in known]
        amplitudes = np.array([known.get(name, np.nan) for name in self.cases])
        strains = {}
        if missing:
            static = cholesky(self.K).solve(self.loads[:, missing])
            centers = self.center_values(self.full(static))
            scale = .005*config['model_length']/np.linalg.norm(centers, axis=-1).max(axis=1)
            amplitudes[missing] = scale*config['common_amplitude_reduction']
            for k, col in enumerate(missing):
                gradient = self.basis.interpolate(self.full(static[:, k]*amplitudes[col])).grad
                strain = .5*(gradient+gradient.swapaxes(0, 1))
                strains[self.cases[col]] = float(np.sqrt(np.sum(strain*strain, axis=(0, 1))).max())
        return dict(config, load_cases=list(self.cases), amplitudes=amplitudes.tolist(),
                    calibrated_here=[self.cases[i] for i in missing], static_peak_strains_here=strains)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--geometry', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--config', type=Path)
    p.add_argument('--compare', type=Path)
    p.add_argument('--min-substeps', type=int, default=4)
    p.add_argument('--case-set', choices=sorted(CASE_SETS), default='original')
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    reference = Reference(args.geometry, CASE_SETS[args.case_set])
    config = reference.amplitudes_for(json.loads(args.config.read_text())) if args.config else reference.calibrate()
    (args.out/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    print('Config', json.dumps(config), flush=True)
    levels, prior, accepted = [], None, None
    substeps = args.min_substeps
    while substeps <= 64:
        trajectories, info = reference.rollout(config, substeps)
        if prior is not None:
            info['time_refinement_errors'] = {name: float(normalized_difference(prior[name], trajectories[name],
                reference.points, reference.g['fixed'])) for name in reference.cases}
            print('Time differences', json.dumps(info['time_refinement_errors']), flush=True)
        levels.append(info)
        np.savez_compressed(args.out/f'trajectories_substeps_{substeps}.npz', **trajectories)
        if prior is not None and max(info['time_refinement_errors'].values()) < .005:
            accepted = trajectories
            break
        prior = trajectories
        substeps *= 2
    assert accepted is not None, 'Time refinement failed; do not use these targets for training.'
    mesh_errors = None
    if args.compare:
        with np.load(args.compare/'trajectories.npz') as old:
            mesh_errors = {name: float(normalized_difference(old[name], accepted[name], reference.points,
                reference.g['fixed'])) for name in reference.cases if name in old.files}
    if 'train_down' in accepted:
        accepted['test_strong_down'] = reference.points+2*(accepted['train_down']-reference.points)
    np.savez_compressed(args.out/'trajectories.npz', **accepted)
    patterns = reference.dem_loads*np.array(config['amplitudes'])[:, None, None]
    for i, name in enumerate(reference.cases):
        np.savez_compressed(args.out/f'{name}_target.npz', positions=accepted[name],
            force_pattern=patterns[i], dt=config['dt'], pulse_duration=config['pulse_duration'])
    if 'train_down' in accepted:
        np.savez_compressed(args.out/'test_strong_down_target.npz', positions=accepted['test_strong_down'],
            force_pattern=2*patterns[reference.cases.index('train_down')], dt=config['dt'],
            pulse_duration=config['pulse_duration'])
    np.savez_compressed(args.out/'patch.npz', **{k:reference.g[k] for k in
        ('points', 'radii', 'pairs', 'areas', 'contacts', 'masses', 'fixed')})
    ready = bool(mesh_errors) and max(mesh_errors.values()) < .005
    ready = ready and reference.checks['volume_relative_error'] < .005
    ready = ready and max(levels[-1]['peak_sampled_strains']+[levels[-1]['doubled_down_peak_strain']]) < .05
    report = dict(complete=True, ready_for_training=ready, checks=reference.checks,
        time_refinement=levels, mesh_refinement_errors=mesh_errors, geometry=str(args.geometry),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        geometry_sha256=hashlib.sha256(args.geometry.read_bytes()).hexdigest(),
        target_source='Independent continuum P2 linear elasticity; no DEM force/stiffness assembly.',
        scope='Small-strain synthetic continuum target; not real data or nonlinear material validation.')
    (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print('Reference result', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
