"""Serial tetrahedron-volume patch tests and apparent tangent scans.

Reuses completed pair tangents and actual clipped-cell volumes. No time
integration, training, FEM remeshing, or alteration of old experiment outputs.
"""
import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import cho_factor, cho_solve

from experiments.foam_dem_moduli import measures, strain_fields, write
from experiments.foam_tetra_volume import NodalTetraVolume, grid_tetrahedra


def norm(x):
    return float(torch.linalg.vector_norm(x).detach())


def scalar(x):
    return float(x.detach())


def mechanics_checks(model):
    X = model.reference
    n = len(X)
    theta = torch.zeros(n, dtype=X.dtype)
    t = X.new_tensor([.31, -.23, .17])
    angle = .71
    rotation = X.new_tensor([[np.cos(angle), -np.sin(angle), 0],
                            [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    transformations = [('identity', torch.eye(3, dtype=X.dtype)),
        ('rigid', rotation),
        ('isochoric_shear', X.new_tensor([[1, .7, 0], [0, 1, 0], [0, 0, 1]])),
        ('isochoric_stretch', torch.diag(X.new_tensor([2., .5, 1.]))),
        ('general_affine', X.new_tensor([[1.2, .2, .1], [.1, .9, .3], [0, .1, 1.1]]))]
    transformations += [(f'dilation_{e:g}', (1+e)*torch.eye(3, dtype=X.dtype))
                        for e in (-.1, 1e-6, 1e-4, 1e-3, .1)]
    rows = []
    for name, F in transformations:
        x = X@F.T+t
        model.validate_configuration(x)
        E, force, J = model.evaluate(x, theta)
        det = float(torch.linalg.det(F))
        pressure_scale = abs(det-1)*max(norm(F)**2/3, 1e-30)
        internal = norm(force[~model.boundary])
        J_error = float((J-det).abs().max())
        assert J_error < 1e-12
        assert internal < 2e-11
        if abs(det-1) < 1e-14:
            assert abs(float(E)) < 1e-24 and norm(force) < 2e-11
        rows.append(dict(name=name, determinant=det, energy=float(E), J_max_error=J_error,
            interior_force_norm=internal if (~model.boundary).any() else None,
            interior_force_over_pressure=internal/pressure_scale
                if pressure_scale > 1e-14 and (~model.boundary).any() else None,
            force_norm=norm(force), net_force=norm(force.sum(0)),
            net_moment=norm(torch.linalg.cross(x, force).sum(0))))

    # Independent force/energy, tangent and parameter derivative checks in a
    # nonaffine state with heterogeneous multipliers, not just affine patch tests.
    rng = np.random.default_rng(705)
    x = (X+X.new_tensor(.03*rng.normal(size=X.shape))).requires_grad_()
    theta = X.new_tensor(.3*rng.normal(size=n)).requires_grad_()
    model.validate_configuration(x)
    E, force, _ = model.evaluate(x, theta)
    energy_gradient = torch.autograd.grad(E, x, retain_graph=True)[0]
    force_error = norm(force+energy_gradient)/norm(force)
    direction = X.new_tensor(rng.normal(size=X.shape))
    step = 1e-6
    fd = (model.evaluate(x+step*direction, theta)[0]-
          model.evaluate(x-step*direction, theta)[0])/(2*step)
    work = -(force*direction).sum()
    fd_error = abs(scalar(fd-work))/max(abs(scalar(work)), 1e-30)
    K = model.reference_tangent(theta)
    jac = torch.autograd.functional.jacobian(lambda p: model.evaluate(p, theta)[1], X,
                                              vectorize=True).reshape(3*n, 3*n)
    tangent_error = norm(K+jac)/norm(K)
    parameter_direction = X.new_tensor(rng.normal(size=n))
    force_probe = (force*direction).sum()
    analytic = (torch.autograd.grad(force_probe, theta)[0]*parameter_direction).sum()
    changed = [model.evaluate(x, theta+s*step*parameter_direction)[1] for s in (-1, 1)]
    fd_parameter = ((changed[1]-changed[0])*direction).sum()/(2*step)
    parameter_error = abs(scalar(analytic-fd_parameter))/max(abs(scalar(analytic)), 1e-30)
    rotated = model.evaluate(x@rotation.T+t, theta)
    objectivity_error = norm(rotated[1]-force@rotation.T)/norm(force)
    energy_objectivity_error = abs(scalar(rotated[0]-E))/scalar(E)
    assert max(force_error, tangent_error, objectivity_error, energy_objectivity_error) < 1e-10
    assert max(fd_error, parameter_error) < 1e-7
    assert norm(force.sum(0)) < 1e-11
    assert norm(torch.linalg.cross(x, force).sum(0)) < 1e-11
    inverted = X.clone(); inverted[:, 0] *= -1
    try:
        model.validate_configuration(inverted)
    except ValueError:
        pass
    else:
        raise AssertionError('Inversion was not rejected.')

    total_gradient = torch.autograd.functional.jacobian(
        lambda p: model.volumes_and_gradients(p)[0].sum(), X)
    assert norm(total_gradient[~model.boundary]) < 1e-12
    return dict(topology=model.topology, normalization=float(model.normalization),
        boundary_vertex_indices=torch.nonzero(model.boundary).flatten().tolist(),
        interior_vertex_indices=torch.nonzero(~model.boundary).flatten().tolist(),
        interior_total_volume_gradient_norm=norm(total_gradient[~model.boundary]),
        affine=rows, nonaffine_checks=dict(force_energy_gradient_relative_error=force_error,
            force_energy_finite_difference_relative_error=fd_error,
            reference_tangent_relative_error=tangent_error,
            force_parameter_derivative_relative_error=parameter_error,
            force_objectivity_relative_error=objectivity_error,
            energy_objectivity_relative_error=energy_objectivity_error,
            net_force=norm(force.sum(0)), net_moment=norm(torch.linalg.cross(x, force).sum(0)),
            inversion_rejected=True))


def strain_rotation(R):
    fields = np.zeros((6, 3, 3))
    for a in range(3):
        fields[a, a, a] = 1
    for k, (a, b) in enumerate(((1, 2), (0, 2), (0, 1)), 3):
        fields[k, a, b] = fields[k, b, a] = .5
    rotated = R@fields@R.T
    return np.array([rotated[:, 0, 0], rotated[:, 1, 1], rotated[:, 2, 2],
                     2*rotated[:, 1, 2], 2*rotated[:, 0, 2], 2*rotated[:, 0, 1]])


def cubic_summary(C):
    errors = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1, 1), repeat=3):
            R = np.eye(3)[list(permutation)]*np.array(signs)[:, None]
            if np.linalg.det(R) > 0:
                T = strain_rotation(R)
                errors.append(np.linalg.norm(T.T@C@T-C)/np.linalg.norm(C))
    return dict(C11=float(np.diag(C)[:3].mean()),
                C12=float((C[:3, :3].sum()-np.trace(C[:3, :3]))/6),
                C44=float(np.diag(C)[3:].mean()),
                max_relative_cubic_rotation_defect=float(max(errors)))


def homogenize(K, Q0, boundary, volume):
    prescribed = np.zeros((len(boundary), 6), bool)
    prescribed[boundary, :3] = True
    free = np.flatnonzero(~prescribed.ravel())
    Q = Q0.copy()
    Q[free] -= cho_solve(cho_factor(K[np.ix_(free, free)], lower=True), (K@Q)[free])
    residual = np.linalg.norm((K@Q)[free])/np.linalg.norm(K@Q)
    assert residual < 1e-10
    affine, relaxed = Q0.T@K@Q0/volume, Q.T@K@Q/volume
    assert np.linalg.eigvalsh((affine-relaxed+affine.T-relaxed.T)/2).min() > -1e-8
    hydro_a = Q0[:, :3].sum(1)
    hydro_r = Q[:, :3].sum(1)
    return dict(C_affine=affine.tolist(), C_relaxed=relaxed.tolist(),
        affine_cubic_projection=cubic_summary(affine),
        relaxed_cubic_projection=cubic_summary(relaxed), relaxed=measures(relaxed),
        equilibrium_relative_residual=float(residual),
        hydro_relative_energy_release=float(1-(hydro_r@K@hydro_r)/(hydro_a@K@hydro_a)),
        hydro_relaxation_norm_over_affine_norm=float(np.linalg.norm(hydro_r-hydro_a)/
                                                    np.linalg.norm(hydro_a))), Q


def scan(geometry, cached_fields, cached_report, out, base_modulus):
    points = geometry['points']; n = len(points)
    # The geometry archive uses unit density, so masses equal actual cell volumes.
    volume = float(geometry['masses'].sum())
    assert abs(volume-cached_report['volume']) < 1e-12*volume
    assert np.allclose(points, cached_fields['points'], rtol=0, atol=1e-14)
    Kpair = cached_fields['K']
    Q0 = strain_fields(points)
    models = {}
    unit_tangents = {}
    checks = {}
    translation_dofs = (6*np.arange(n)[:, None]+np.arange(3)).ravel()
    for name, reflected in (('A', False), ('B', True)):
        model = NodalTetraVolume(points, grid_tetrahedra(points, geometry['radii'], reflected), volume)
        models[name] = model
        print('  patch checks', n, name, flush=True)
        checks[name] = mechanics_checks(model)
        K3 = model.reference_tangent(torch.zeros(n, dtype=torch.float64)).numpy()
        unit = np.zeros_like(Kpair)
        unit[np.ix_(translation_dofs, translation_dofs)] = K3
        unit_tangents[name] = unit
        expected = np.zeros((6, 6)); expected[:3, :3] = 1
        assert np.max(abs(Q0.T@unit@Q0/volume-expected)) < 1e-12
        expected_hull = float(np.prod(np.ptp(points, axis=0)))
        assert abs(model.topology['reference_volume']-expected_hull) < 1e-12
    assert torch.equal(models['A'].boundary, models['B'].boundary)
    boundary = models['A'].boundary.numpy()
    assert np.array_equal(boundary, cached_fields['boundary'])
    unit_tangents['average_energy'] = .5*(unit_tangents['A']+unit_tangents['B'])
    baseline, _ = homogenize(Kpair, Q0, boundary, volume)
    assert np.allclose(baseline['C_relaxed'], cached_report['C_relaxed'], rtol=1e-10, atol=1e-9)
    records = []
    for name, Kunit in unit_tangents.items():
        for multiplier in (0., .25, .5, 1., 2.):
            modulus = base_modulus*multiplier
            record, Q = homogenize(Kpair+modulus*Kunit, Q0, boundary, volume)
            increment = np.array(record['C_relaxed'])-np.array(baseline['C_relaxed'])
            assert np.linalg.eigvalsh((increment+increment.T)/2).min() > -1e-8
            record.update(triangulation=name, multiplier=multiplier, k_v=base_modulus,
                          dC_d_uniform_multiplier=(base_modulus*Q.T@Kunit@Q/volume).tolist())
            records.append(record)
    out.mkdir()
    np.savez_compressed(out/'matrices.npz', points=points, boundary=boundary, K_pair=Kpair,
        K_volume_A=unit_tangents['A'], K_volume_B=unit_tangents['B'],
        tetrahedra_A=models['A'].tetrahedra.numpy(), tetrahedra_B=models['B'].tetrahedra.numpy())
    report = dict(cells=n, material_volume=volume, geometry_density=1., checks=checks,
        baseline=baseline, scan=records,
        boundary_condition='Topological surface-center translations prescribed affinely; all rotations and interior translations relaxed.',
        cubic_projection_caution='A cubic projection is descriptive only. The 16-cell bar and learned heterogeneous multipliers need not have cubic symmetry.',
        average_definition='Mean of energies for two fixed triangulations, not energy of an averaged J.',
        inversion_policy='Reject nonpositive signed tetrahedron volumes; no inversion barrier in this candidate energy.')
    write(out/'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--moduli', type=Path, default=Path('output/foam_fem/moduli_v1'))
    parser.add_argument('--reference', type=Path, default=Path('output/foam_fem/reference_edge007'))
    parser.add_argument('--k-v', type=float, default=400.)
    args = parser.parse_args()
    if args.k_v <= 0:
        parser.error('--k-v must be positive')
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    start = time.time()
    def status(stage, **details):
        write(args.out/'status.json', dict(stage=stage, elapsed_seconds=time.time()-start, **details))
        print(stage, details, flush=True)
    try:
        results = {}
        for name, geometry_path, old in (
                ('regular64_initial', args.moduli/'patch64.npz', 'regular64_initial'),
                ('per_cell16', args.reference/'patch.npz', 'per_cell')):
            status('static_checks_and_scan', specimen=name)
            with np.load(geometry_path) as a:
                geometry = {key: a[key] for key in ('points', 'radii', 'masses')}
            with np.load(args.moduli/old/'fields.npz') as a:
                fields = {key: a[key] for key in ('points', 'K', 'boundary')}
            old_report = json.loads((args.moduli/old/'report.json').read_text())
            results[name] = scan(geometry, fields, old_report, args.out/name, args.k_v)
        write(args.out/'report.json', dict(complete=True, elapsed_seconds=time.time()-start,
            models=results, fem=json.loads((args.moduli/'fem.json').read_text()),
            scope='Fixed-grid auxiliary volume energy and reference-state static tests only. No augmented trajectory, time-adjoint integration or training yet.',
            adoption='Undecided; requires matched four-mode/five-mode training and held-out loading comparison.'))
        status('complete', report=str(args.out/'report.json'))
    except BaseException as error:
        status('failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
