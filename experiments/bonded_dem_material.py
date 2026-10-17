"""Stage 2: synthetic 3D bonded specimens, static response and actual dynamics.

Run: python -m experiments.bonded_dem_material (conda om, BLAS threads=1).
One micro-parameter set across all loads/resolutions. No fitted target material,
no Power Foam geometry, no paper-experiment reproduction, no RGB optimization.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from experiments.bonded_dem_core import ParallelBonds
from experiments.bonded_dem_dynamics import (
    BondedDynamics, mass_vector, reference_dt_bound, reference_stiffness)


def specimen(resolution=3, length=1., jitter=0.):
    """Box with boundary nodes, assigned trapezoidal masses, overlap graph.

    These assigned volumes are NOT computed Power Foam cell volumes. Equal
    radii are interaction proxies; mass is not overlapping-sphere mass.
    """
    h = 1/resolution
    counts = [int(round(length*resolution))+1, resolution+1, resolution+1]
    grid = np.stack(np.meshgrid(*[np.arange(n) for n in counts], indexing='ij'), axis=-1).reshape(-1, 3)
    x = grid*h
    weights = np.ones(len(x))
    boundary_axis = np.zeros_like(x, dtype=bool)
    for axis, count in enumerate(counts):
        boundary_axis[:, axis] = (grid[:, axis] == 0) | (grid[:, axis] == count-1)
        weights[boundary_axis[:, axis]] *= .5
    rng = np.random.default_rng(3401)
    x += rng.uniform(-jitter, jitter, x.shape)*h*(~boundary_axis)
    radii = np.full(len(x), .775*h)
    pairs = cKDTree(x).query_pairs(1.55*h, output_type='ndarray')
    lengths = np.linalg.norm(x[pairs[:, 1]]-x[pairs[:, 0]], axis=1)
    bonds = ParallelBonds(x, pairs, radii, .25*h, 1200./lengths, 480./lengths)
    mass = h**3*weights  # density=1, exact total assigned mass=box volume
    inertia = .4*mass*radii**2  # explicitly chosen isotropic proxy
    return dict(bonds=bonds, x=x, mass=mass, inertia=inertia,
                boundary=boundary_axis.any(1), h=h, volume=length,
                resolution=resolution, length=length, jitter=jitter)


def solve_prescribed(k, prescribed, values, load=None):
    """Linear reference equilibrium with exact elimination of fixed DOFs."""
    q = np.asarray(values, dtype=float).copy().ravel()
    mask = np.asarray(prescribed, dtype=bool).ravel()
    free, fixed = np.flatnonzero(~mask), np.flatnonzero(mask)
    rhs = np.zeros(len(q)) if load is None else np.asarray(load).ravel()
    q[free] = spsolve(k[free][:, free], rhs[free]-k[free][:, fixed]@q[fixed])
    reaction = k@q-rhs
    scale = max(np.linalg.norm(rhs), np.linalg.norm(reaction[fixed]), 1e-30)
    residual = np.linalg.norm(reaction[free])/scale
    if not np.isfinite(q).all() or residual > 1e-8:
        raise AssertionError(f'Unresolved static equilibrium, relative residual {residual}')
    return q, float(residual)


def tangent_and_modes():
    report, spectra = {}, {}
    for label, jitter in [('regular', 0.), ('jittered', .08)]:
        s = specimen(2, jitter=jitter)
        b, x = s['bonds'], s['x']
        k = reference_stiffness(b)
        rng = np.random.default_rng(201)
        errors = []
        for _ in range(5):
            direction = rng.normal(size=(len(x), 6))
            direction[:, :3] *= s['h']
            eps = 1e-7
            force = []
            for sign in (1., -1.):
                candidate = copy.deepcopy(b)
                delta = sign*eps*direction
                candidate.advance(x+delta[:, :3], delta[:, :3], delta[:, 3:], 1.)
                f, t = candidate.loads()
                force.append(np.column_stack((f, t)).ravel())
            numeric = -(force[0]-force[1])/(2*eps)
            expected = k@direction.ravel()
            errors.append(float(np.linalg.norm(numeric-expected)/np.linalg.norm(expected)))
        assert max(errors) < 1e-8
        rigid = []
        for axis in np.eye(3):
            q = np.zeros((len(x), 6)); q[:, :3] = axis
            rigid.append(q.ravel())
            q = np.zeros_like(q); q[:, :3] = np.cross(axis, x); q[:, 3:] = axis
            rigid.append(q.ravel())
        rigid = np.array(rigid).T
        rigid_error = np.linalg.norm(k@rigid)/np.linalg.norm(k.toarray())/np.linalg.norm(rigid)
        m = mass_vector(s['mass'], s['inertia'])
        eigenvalues = eigh(k.toarray(), np.diag(m), eigvals_only=True)
        tol = eigenvalues[-1]*1e-9
        zero = int(np.count_nonzero(np.abs(eigenvalues) < tol))
        negative = int(np.count_nonzero(eigenvalues < -tol))
        assert zero == 6 and negative == 0 and rigid_error < 1e-12
        spectra[label] = eigenvalues
        report[label] = dict(particles=len(x), bonds=len(b.pairs), zero_modes=zero,
                              negative_modes=negative, rigid_mode_residual=float(rigid_error),
                              tangent_relative_errors=errors,
                              first_deformational_frequency=float(np.sqrt(eigenvalues[6])),
                              eigenvalue_zero_tolerance=float(tol))
    return report, spectra


def affine_moduli(s, k):
    """Finite-box apparent stiffness under affine boundary displacement.

    Interior translations and ALL particle rotations relax. This is a
    displacement-boundary estimate, not a periodic or infinite-body modulus.
    Engineering strain ordering: xx, yy, zz, yz, xz, xy.
    """
    x = s['x']
    mask = np.zeros((len(x), 6), dtype=bool)
    mask[s['boundary'], :3] = True
    strains = []
    for a in range(3):
        strain = np.zeros((3, 3)); strain[a, a] = 1.; strains.append(strain)
    for a, b in ((1, 2), (0, 2), (0, 1)):
        strain = np.zeros((3, 3)); strain[a, b] = strain[b, a] = .5; strains.append(strain)
    fields, residuals = [], []
    for strain in strains:
        q = np.zeros((len(x), 6)); q[:, :3] = x@strain.T
        q, residual = solve_prescribed(k, mask, q)
        fields.append(q); residuals.append(residual)
    fields = np.array(fields).T
    c = fields.T@(k@fields)/s['volume']
    compliance = np.linalg.inv(c)
    young = 1/np.diag(compliance)[:3]
    shear = 1/np.diag(compliance)[3:]
    poisson = -compliance[1, 0]/compliance[0, 0]
    # A diagnostic only; cubic/irregular networks need not be isotropic.
    isotropic_shear = young[0]/(2*(1+poisson))
    return dict(C=c.tolist(), young_xyz=young.tolist(), shear_yz_xz_xy=shear.tolist(),
                poisson_xy=float(poisson), isotropic_shear_from_Ex_nuxy=float(isotropic_shear),
                shear_xy_vs_isotropic_relative_difference=float(shear[2]/isotropic_shear-1),
                max_static_residual=max(residuals))


def uniaxial(s, k, strain=1e-4):
    """Axial displacement grips, transverse sides free; minimal rigid pins."""
    x = s['x']; left = np.isclose(x[:, 0], 0); right = np.isclose(x[:, 0], s['length'])
    mask = np.zeros((len(x), 6), dtype=bool)
    mask[left | right, 0] = True
    origin = np.argmin(np.linalg.norm(x, axis=1))
    ycorner = np.argmin(np.linalg.norm(x-[0, 1, 0], axis=1))
    mask[origin, 1:3] = True; mask[ycorner, 2] = True
    q = np.zeros((len(x), 6)); q[right, 0] = strain*s['length']
    q, residual = solve_prescribed(k, mask, q)
    reaction = (k@q).reshape(-1, 6)
    q = q.reshape(-1, 6)
    force = reaction[right, 0].sum()  # cross-section area=1
    nus = []
    for axis in (1, 2):
        side0, side1 = np.isclose(x[:, axis], 0), np.isclose(x[:, axis], 1)
        lateral = q[side1, axis].mean()-q[side0, axis].mean()
        nus.append(float(-lateral/strain))
    assert force > 0
    return dict(apparent_young=float(force/strain), poisson_xy_xz=nus, residual=residual), q


def beam_case(resolution, cube_moduli, jitter=0.):
    s = specimen(resolution, length=3., jitter=jitter)
    x, b = s['x'], s['bonds']; k = reference_stiffness(b)
    fixed = np.isclose(x[:, 0], 0); tip = np.isclose(x[:, 0], 3.)
    mask = np.repeat(fixed[:, None], 6, axis=1)
    load = np.zeros((len(x), 6))
    total_force = .001
    # Trapezoidal surface traction, normalized to the same total load.
    weights = s['mass'][tip]/s['mass'][tip].sum()
    load[tip, 2] = -total_force*weights
    q, residual = solve_prescribed(k, mask, np.zeros_like(load), load)
    q = q.reshape(-1, 6)
    deflection = float(-weights@q[tip, 2])
    ex, gxz = cube_moduli['young_xyz'][0], cube_moduli['shear_yz_xz_xy'][1]
    estimate = total_force*(3.**3/(3*ex*(1/12))+3./((5/6)*gxz))
    return dict(particles=len(x), bonds=len(b.pairs), deflection=deflection,
                timoshenko_estimate=float(estimate), relative_difference=float(deflection/estimate-1),
                residual=residual,
                note='Short discrete beam, 6-DOF clamp; continuum formula is only an approximate comparator.'), (x, q)


def material_cases():
    report, beams = [], {}
    for jitter in (0., .08):
        for resolution in (2, 3, 4):
            s = specimen(resolution, jitter=jitter); k = reference_stiffness(s['bonds'])
            moduli = affine_moduli(s, k)
            tension, _ = uniaxial(s, k)
            bending, beam = beam_case(resolution, moduli, jitter)
            result = dict(resolution=resolution, jitter=jitter, particles=len(s['x']),
                          bonds=len(s['bonds'].pairs), affine_boundary=moduli,
                          free_lateral_tension=tension, bending=bending)
            report.append(result)
            beams[f'r{resolution}_j{jitter}'] = beam
            print(f'Material r={resolution}, jitter={jitter}: E={tension["apparent_young"]:.4f}, '
                  f'nu={tension["poisson_xy_xz"][0]:.4f}, beam={bending["deflection"]:.6g}', flush=True)
    return report, beams


def preload(bonds, q, steps=32):
    """Load actual bond history along a small-displacement straight path."""
    x = bonds.x.copy()
    for step in range(steps):
        bonds.advance(x+q[:, :3]*(step+1)/steps, q[:, :3], q[:, 3:], 1/steps)


def actual_static_force_checks():
    """Verify linear fixtures against forces from actual loaded bond histories.

    We do not claim to have solved full nonlinear static equilibrium. The
    discrepancy must shrink with loading amplitude in this small-strain test.
    """
    s = specimen(3, jitter=.08); k = reference_stiffness(s['bonds'])
    _, tension = uniaxial(s, k)
    mask = np.zeros((len(s['x']), 6), dtype=bool); mask[s['boundary'], :3] = True
    shear = np.zeros_like(mask, dtype=float)
    shear[:, 0] = .5e-4*s['x'][:, 1]; shear[:, 1] = .5e-4*s['x'][:, 0]
    shear, _ = solve_prescribed(k, mask, shear)
    beam = specimen(3, length=3., jitter=.08); kb = reference_stiffness(beam['bonds'])
    _, (_, bending) = beam_case(3, affine_moduli(s, k), .08)
    report = {}
    for name, model, stiffness, q in [('tension', s['bonds'], k, tension),
                                     ('shear', s['bonds'], k, shear.reshape(-1, 6)),
                                     ('bending', beam['bonds'], kb, bending)]:
        errors, energy_errors = [], []
        for scale in (1., .5):
            loaded = copy.deepcopy(model); displacement = q*scale
            preload(loaded, displacement)
            f, t = loaded.loads()
            actual = np.column_stack((f, t)).ravel()
            predicted = -stiffness@displacement.ravel()
            errors.append(float(np.linalg.norm(actual-predicted)/np.linalg.norm(predicted)))
            elastic = -.5*np.dot(displacement.ravel(), predicted)
            energy_errors.append(float(abs(loaded.elastic_energy_per_bond().sum()/elastic-1)))
        assert max(errors) < 2e-3 and max(energy_errors) < 2e-3, (name, errors, energy_errors)
        assert errors[0]/errors[1] > 1.7, (name, errors)
        report[name] = dict(load_scales=[1., .5], relative_force_errors=errors,
                            relative_energy_errors=energy_errors)
    return report


def modal_release():
    s = specimen(2, length=2., jitter=.08)
    x, k = s['x'], reference_stiffness(s['bonds'])
    fixed = np.isclose(x[:, 0], 0)
    free = np.flatnonzero(~np.repeat(fixed, 6))
    mass = mass_vector(s['mass'], s['inertia'])
    eig, vectors = eigh(k[free][:, free].toarray(), np.diag(mass[free]), subset_by_index=[0, 0])
    frequency = np.sqrt(eig[0])
    mode = np.zeros(6*len(x)); mode[free] = vectors[:, 0]
    mode = mode.reshape(-1, 6)
    amplitude = 2e-6/np.max(np.linalg.norm(mode[:, :3], axis=1))
    q0 = amplitude*mode
    dt_bound = reference_dt_bound(k, s['mass'], s['inertia'], fixed)
    duration = 2*2*np.pi/frequency
    coarse_steps = int(np.ceil(duration/(.35*dt_bound)))
    runs, curves = [], {}
    for refinement in (1, 2, 4):
        bonds = copy.deepcopy(s['bonds'])
        preload(bonds, q0)
        dynamic = BondedDynamics(bonds, s['mass'], s['inertia'], fixed)
        dynamic.rotation = Rotation.from_rotvec(q0[:, 3:]).as_matrix()
        e0 = dynamic.energy()
        steps = coarse_steps*refinement; dt = duration/steps
        errors, energies, samples = [], [], []
        for step in range(steps+1):
            # A mass-weighted projection onto the independently computed mode.
            q = np.column_stack((bonds.x-x, Rotation.from_matrix(dynamic.rotation).as_rotvec()))
            projection = np.dot(mode.ravel()*mass, q.ravel())/amplitude
            analytic = np.cos(frequency*step*dt)
            relative_energy = dynamic.energy()/e0-1
            errors.append(abs(projection-analytic)); energies.append(abs(relative_energy))
            if step % max(1, steps//300) == 0 or step == steps:
                samples.append([step*dt, projection, analytic, relative_energy])
            if step < steps:
                dynamic.step(dt)
        assert np.max(np.abs(bonds.x[fixed]-x[fixed])) == 0
        assert np.max(np.abs(dynamic.rotation[fixed]-np.eye(3))) == 0
        orthogonality = np.linalg.norm(dynamic.rotation@dynamic.rotation.transpose(0, 2, 1)-np.eye(3))
        assert orthogonality < 1e-10
        result = dict(steps=steps, dt=dt, max_modal_error=max(errors), max_relative_energy_error=max(energies),
                      orientation_orthogonality_error=float(orthogonality))
        runs.append(result); curves[str(refinement)] = np.array(samples)
        print(f'Modal release: {result}', flush=True)
    assert runs[-1]['max_modal_error'] < 1e-3
    assert runs[-1]['max_relative_energy_error'] < 1e-3
    for key in ('max_modal_error', 'max_relative_energy_error'):
        assert runs[0][key]/runs[1][key] > 2.5 and runs[1][key]/runs[2][key] > 2.5, (key, runs)
    return dict(particles=len(x), frequency=float(frequency), reference_dt_bound=dt_bound,
                max_initial_translation=float(np.max(np.linalg.norm(q0[:, :3], axis=1))), runs=runs), curves


def free_momentum():
    s = specimen(2, jitter=.08)
    b = s['bonds']; rng = np.random.default_rng(934)
    q = rng.normal(0, 1e-5, (len(b.x), 6)); preload(b, q)
    d = BondedDynamics(b, s['mass'], s['inertia'])
    d.rotation = Rotation.from_rotvec(q[:, 3:]).as_matrix()
    d.velocity = rng.normal(0, 1e-3, b.x.shape)
    d.omega = rng.normal(0, 1e-3, b.x.shape)
    p0, l0 = d.momentum(); e0 = d.energy()
    maxp, maxl = 0., 0.
    for _ in range(250):
        d.step(1e-4)
        p, l = d.momentum()
        maxp = max(maxp, float(np.linalg.norm(p-p0)))
        maxl = max(maxl, float(np.linalg.norm(l-l0)))
    assert maxp < 1e-12 and maxl < 1e-12
    return dict(max_linear_momentum_drift=maxp, max_angular_momentum_drift=maxl,
                final_relative_energy_change=float(d.energy()/e0-1))


def external_load_checks():
    s = specimen(2, jitter=.08)
    x = s['x'].copy()
    acceleration = np.array([.3, -.2, .1])
    d = BondedDynamics(s['bonds'], s['mass'], s['inertia'])
    for _ in range(60):
        d.step(.001, force_external=s['mass'][:, None]*acceleration)
    error = float(np.max(np.abs(d.bonds.x-x-.5*d.time**2*acceleration)))
    velocity_error = float(np.max(np.abs(d.velocity-d.time*acceleration)))
    assert error < 1e-12 and velocity_error < 1e-12
    s = specimen(2)
    d = BondedDynamics(s['bonds'], s['mass'], s['inertia'], np.ones(len(s['x']), dtype=bool))
    d.step(.001, force_external=np.ones_like(s['x']), torque_external=np.ones_like(s['x']))
    assert np.array_equal(d.bonds.x, s['x']) and np.all(d.velocity == 0) and np.all(d.omega == 0)
    # Rejected input must not advance body state or bond history.
    previous_time = d.time
    try:
        d.step(-1.)
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid timestep accepted')
    assert d.time == previous_time
    return dict(uniform_acceleration_position_error=error, velocity_error=velocity_error,
                fixed_bodies_resist_external_force_and_torque=True, negative_dt_rejected=True)


def plot(out, report, spectra, beams, releases):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    ax = axes[0, 0]
    for label, eigenvalues in spectra.items():
        ax.semilogy(np.arange(7, 27), eigenvalues[6:26], 'o-', label=label)
    ax.set(title='Exactly 6 rigid modes; no additional zero modes', xlabel='mode index', ylabel='squared frequency')
    ax.legend(); ax.grid(alpha=.25)
    for jitter, label in ((0., 'regular'), (.08, 'jittered')):
        cases = [r for r in report['materials'] if r['jitter'] == jitter]
        axes[0, 1].plot([r['resolution'] for r in cases], [r['free_lateral_tension']['apparent_young'] for r in cases], 'o-', label=label)
        axes[0, 2].plot([r['resolution'] for r in cases], [r['affine_boundary']['shear_xy_vs_isotropic_relative_difference']*100 for r in cases], 'o-', label=label)
    axes[0, 1].set(title='Measured axial stiffness (same micro parameters)', xlabel='cells per unit length', ylabel='apparent Young modulus')
    axes[0, 2].set(title='Shear differs from isotropic E, nu prediction', xlabel='cells per unit length', ylabel='relative difference (%)')
    for ax in axes[0, 1:]: ax.legend(); ax.grid(alpha=.25)
    ax = axes[1, 0]
    x, q = beams['r4_j0.0']
    ids = np.isclose(x[:, 1], .5) & np.isclose(x[:, 2], .5)
    ax.plot(x[ids, 0], q[ids, 2], 'o-')
    ax.set(title='Cantilever centerline under end load', xlabel='x', ylabel='z displacement (unscaled)')
    ax.grid(alpha=.25)
    ax = axes[1, 1]
    values = releases['4']
    ax.plot(values[:, 0], values[:, 2], color='black', label='linear eigenmode prediction')
    ax.plot(values[::5, 0], values[::5, 1], '.', label='actual bond dynamics')
    ax.set(title='45-particle release, no damping', xlabel='time', ylabel='normalized mode amplitude'); ax.legend()
    ax = axes[1, 2]
    runs = report['modal_release']['runs']
    for key, label in [('max_modal_error', 'modal error'), ('max_relative_energy_error', 'energy error')]:
        ax.loglog([r['dt'] for r in runs], [r[key] for r in runs], 'o-', label=label)
    ax.set(title='Time-step convergence', xlabel='dt', ylabel='relative error'); ax.legend(); ax.grid(alpha=.25)
    fig.suptitle('Stage 2: synthetic bonded material — no Power Foam coupling, no target-material fit')
    fig.savefig(out/'material.png', dpi=160); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=Path('output/bonded_dem_stage2'))
    args = parser.parse_args()
    report = dict(scope='Synthetic small-strain bonded specimens, no collisions or Power Foam coupling.',
                  micro_parameters=dict(E_b=1200., G_b=480., bond_radius_over_h=.25,
                                        particle_radius_over_h=.775, density=1.),
                  no_per_load_parameter_fit=True)
    report['modes'], spectra = tangent_and_modes()
    print('Reference tangent and six rigid modes: passed', flush=True)
    report['materials'], beams = material_cases()
    report['actual_static_force_checks'] = actual_static_force_checks()
    print(f'Actual bond forces versus linear statics: {report["actual_static_force_checks"]}', flush=True)
    report['momentum'] = free_momentum()
    report['external_load_checks'] = external_load_checks()
    print(f'Free-body momentum: {report["momentum"]}', flush=True)
    report['modal_release'], releases = modal_release()
    report['source_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (Path(__file__), Path(__file__).with_name('bonded_dem_dynamics.py'),
                                        Path(__file__).with_name('bonded_dem_core.py'))}
    args.out.mkdir(parents=True, exist_ok=True)
    plot(args.out, report, spectra, beams, releases)
    (args.out/'results.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    np.savez_compressed(args.out/'curves.npz', **{f'release_{k}': v for k, v in releases.items()},
                        **{f'spectrum_{k}': v for k, v in spectra.items()},
                        **{f'beam_{k}_positions': v[0] for k, v in beams.items()},
                        **{f'beam_{k}_displacements': v[1] for k, v in beams.items()})
    print(f'All stage-2 checks passed. Artifacts: {args.out}', flush=True)


if __name__ == '__main__':
    main()
