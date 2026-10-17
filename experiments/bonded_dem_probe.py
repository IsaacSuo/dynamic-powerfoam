"""Stage 1: analytical mechanics checks of our parallel-bond implementation.

Run from the repository root in conda om:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.bonded_dem_probe

These are our verification fixtures, not the papers' experimental specimens.
No rendering, image optimization, bulk material calibration or Foam coupling.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy
from scipy.spatial.transform import Rotation

from experiments.bonded_dem_core import ParallelBonds


X = np.array([[-.5, 0., 0.], [.5, 0., 0.]])
RB, E_OVER_L, G_OVER_L = .12, 1200., 480.
AREA = np.pi*RB**2
KN, KS = E_OVER_L*AREA, G_OVER_L*AREA
KB, KT = E_OVER_L*np.pi*RB**4/4, G_OVER_L*np.pi*RB**4/2


def pair(**kwargs):
    params = dict(reference=X, pairs=np.array([[0, 1]]), radii=.8,
                  bond_radius=RB, kbar_n=E_OVER_L, kbar_s=G_OVER_L)
    params.update(kwargs)
    return ParallelBonds(**params)


def prescribed(model, mode, q0, q1, steps=128):
    """Analytic trajectory and midpoint velocities over unit time."""
    rate, dt = q1-q0, 1/steps
    for step in range(steps):
        qm = q0+rate*(step+.5)/steps
        qe = q0+rate*(step+1)/steps
        mid, end = X.copy(), X.copy()
        v, w = np.zeros_like(X), np.zeros_like(X)
        if mode in ('normal', 'shear'):
            axis = 0 if mode == 'normal' else 1
            mid[1, axis] += qm
            end[1, axis] += qe
            v[1, axis] = rate
        else:
            axis = 2 if mode == 'bend' else 0
            w[0, axis], w[1, axis] = -.5*rate, .5*rate
        model.advance(end, v, w, dt, x_mid=mid)


def expected(mode, q):
    """Independent closed-form integration of the published differential law.

    For transverse center displacement y=q, L0=1 and zero body spins:
    dn/dq=q/sqrt(1+q^2), ds/dq=1/sqrt(1+q^2) in the moving tangent.
    Thus normal extension is sqrt(1+q^2)-1, shear extension is asinh(q).
    This path is not exactly pure shear at finite displacement.
    """
    fn, fs, mt, mb = 0., np.zeros(3), 0., np.zeros(3)
    if mode == 'normal':
        fn = KN*q
    elif mode == 'shear':
        length = np.sqrt(1+q*q)
        fn = KN*(length-1)
        fs = KS*np.arcsinh(q)*np.array([-q, 1., 0.])/length
    elif mode == 'bend':
        mb[2] = KB*q
    elif mode == 'twist':
        mt = KT*q
    return np.r_[fn, fs, mt, mb]


def bond_values(model):
    return np.r_[model.fn[0], model.fs[0], model.mt[0], model.mb[0]]


def check_balance(model):
    f, t = model.loads()
    force = np.linalg.norm(f.sum(0))
    torque = np.linalg.norm((np.cross(model.x, f)+t).sum(0))
    assert force < 1e-10 and torque < 1e-10, (force, torque)
    return dict(net_force=float(force), net_torque=float(torque))


def analytic_checks():
    report, curves = {}, {}
    for mode, amplitude in [('normal', .03), ('shear', .08), ('bend', .12), ('twist', .12)]:
        model = pair()
        prescribed(model, mode, 0., amplitude)
        want, got = expected(mode, amplitude), bond_values(model)
        error = np.linalg.norm(got-want)/np.linalg.norm(want)
        assert error < 2e-7, (mode, error)
        balance = check_balance(model)
        initial_energy = float(model.elastic_energy_per_bond().sum())
        prescribed(model, mode, amplitude, 0.)
        residual = np.linalg.norm(bond_values(model))
        assert residual < 1e-10, (mode, residual)
        assert abs(model.work) < 1e-10
        report[mode] = dict(relative_error=float(error), unload_residual=float(residual),
                            peak_elastic_energy=initial_energy, **balance)
        model = pair()
        samples, last = [], 0.
        for q in np.linspace(0., amplitude, 25):
            prescribed(model, mode, last, q, steps=8)
            got, want = bond_values(model), expected(mode, q)
            index = {'normal': 0, 'shear': 2, 'bend': 7, 'twist': 4}[mode]
            samples.append([q, got[index], want[index]])
            last = q
        curves[mode] = np.array(samples)
    # Compression uses the same normal spring, with the opposite sign.
    model = pair()
    prescribed(model, 'normal', 0., -.03)
    assert np.isclose(model.fn[0], -KN*.03, atol=1e-12)
    report['compression_force'] = float(model.fn[0])
    return report, curves


def rigid_motion(model, angle, steps, axis=None):
    axis = np.array([.3, -.5, .7]) if axis is None else np.asarray(axis, dtype=float)
    axis = axis/np.linalg.norm(axis)
    spin = axis*angle
    initial = model.x.copy()
    drift = np.array([.17, -.23, .11])
    dt = 1/steps
    for step in range(steps):
        tm, te = (step+.5)*dt, (step+1)*dt
        rm = Rotation.from_rotvec(spin*tm).as_matrix()
        re = Rotation.from_rotvec(spin*te).as_matrix()
        centered = initial@rm.T
        mid, end = centered+tm*drift, initial@re.T+te*drift
        v = np.cross(spin, centered)+drift
        w = np.broadcast_to(spin, initial.shape)
        model.advance(end, v, w, dt, x_mid=mid)
    return Rotation.from_rotvec(spin).as_matrix()


def rotation_checks():
    report = {}
    virgin = pair(radii=np.array([.9, .65]))
    rigid_motion(virgin, 2.3, 100)
    residual = float(np.linalg.norm(bond_values(virgin)))
    assert residual < 1e-11
    report['unloaded_rigid_motion_force_moment_norm'] = residual

    # Rotation ABOUT the contact normal must rotate existing shear history.
    # There is no normal change for this test; normal-only transport fails it.
    axial = pair()
    prescribed(axial, 'shear', 0., .08)
    before = axial.fs.copy()
    energy0 = float(axial.elastic_energy_per_bond().sum())
    axis = axial.normal[0].copy()
    rot = rigid_motion(axial, 1.7, 80, axis)
    err = np.linalg.norm(axial.fs-before@rot.T)/np.linalg.norm(before)
    assert err < 1e-11
    report['axial_spin_history_relative_error'] = float(err)
    report['axial_spin_omitted_history_relative_error'] = float(
        np.linalg.norm(before-before@rot.T)/np.linalg.norm(before))
    assert report['axial_spin_omitted_history_relative_error'] > 1.
    assert abs(axial.elastic_energy_per_bond().sum()-energy0) < 1e-11

    # For arbitrary finite rigid rotation, the split frame integrator has a
    # truncation error. Report its dt convergence instead of claiming exactness.
    errors = []
    for steps in (20, 40, 80):
        model = pair()
        prescribed(model, 'shear', 0., .08)
        f0, t0 = model.loads()
        e0 = model.elastic_energy_per_bond().sum()
        rot = rigid_motion(model, 2.3, steps)
        f, t = model.loads()
        error = np.linalg.norm(np.r_[f-f0@rot.T, t-t0@rot.T])/np.linalg.norm(np.r_[f0, t0])
        errors.append(float(error))
        assert abs(model.elastic_energy_per_bond().sum()-e0) < 1e-11
        check_balance(model)
    assert errors[-1] < 1e-4 and errors[0]/errors[1] > 3.5 and errors[1]/errors[2] > 3.5, errors
    report['loaded_rigid_rotation'] = dict(steps=[20, 40, 80], relative_errors=errors)
    return report


def network_checks():
    # Irregular centers, unequal radii, coupled translation/spin, and an
    # arbitrary change of observer axes. Not a bulk material calibration.
    reference = np.array([[0., 0., 0.], [1., .1, 0.], [.1, .8, .2], [.2, .1, 1.1]])
    pairs = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
    radii = np.array([.8, .7, .9, .75])
    rot = Rotation.from_euler('xyz', [31., -27., 43.], degrees=True).as_matrix()
    shift = np.array([.8, -.3, .4])
    common = dict(pairs=pairs, radii=radii, bond_radius=.09, kbar_n=1200., kbar_s=480.)
    a = ParallelBonds(reference, **common)
    b = ParallelBonds(reference@rot.T+shift, **common)
    rng = np.random.default_rng(813)
    for _ in range(60):
        v, w = rng.normal(0, .1, reference.shape), rng.normal(0, .2, reference.shape)
        end = a.x+v*.01
        b.advance(end@rot.T+shift, v@rot.T, w@rot.T, .01)
        a.advance(end, v, w, .01)
    fa, ta = a.loads()
    fb, tb = b.loads()
    error = np.linalg.norm(np.r_[fb-fa@rot.T, tb-ta@rot.T])/np.linalg.norm(np.r_[fa, ta])
    assert error < 1e-11
    energy = float(a.elastic_energy_per_bond().sum())
    assert abs(energy-a.work) < 1e-11
    return dict(observer_rotation_relative_error=float(error), **check_balance(a),
                constitutive_work_minus_energy=float(a.work-energy),
                note='Work identity is bookkeeping, not an independent energy-conservation proof.')


def failure_checks():
    results = {}
    # The same critical stress=1 is exceeded separately by each loading mode.
    limits = dict(normal=1/E_OVER_L, shear=1/G_OVER_L,
                  bend=1/(E_OVER_L*RB), twist=1/(G_OVER_L*RB))
    for mode, critical in limits.items():
        model = pair(tensile_strength=1. if mode in ('normal', 'bend') else np.inf,
                     cohesion=1. if mode in ('shear', 'twist') else np.inf)
        prescribed(model, mode, 0., .8*critical)
        assert model.intact[0]
        prescribed(model, mode, .8*critical, 1.2*critical)
        expected_failure = 1 if mode in ('normal', 'bend') else 2
        assert model.failure[0] == expected_failure
        removed = float(model.break_energy.sum())
        assert removed > 0 and np.linalg.norm(bond_values(model)) == 0
        assert abs(model.work-removed) < 1e-11
        # No healing on closure/reloading, no second energy removal.
        prescribed(model, mode, 1.2*critical, 0.)
        prescribed(model, mode, 0., 1.4*critical)
        assert model.failure[0] == expected_failure and np.linalg.norm(bond_values(model)) == 0
        assert model.break_energy.sum() == removed and abs(model.work-removed) < 1e-11
        results[mode] = dict(failure_code=int(model.failure[0]), removed_energy=removed,
                             remained_broken_on_reloading=True)
    # Compression must not be misclassified as tensile failure.
    model = pair(tensile_strength=.1)
    prescribed(model, 'normal', 0., -.02)
    assert model.intact[0]
    results['compression_does_not_break_in_tension'] = True
    # Pressure-dependent shear strength: compression raises, tension lowers it.
    for sign, should_break in [(-1., False), (1., True)]:
        model = pair(tensile_strength=10., cohesion=1., friction_angle=np.pi/4)
        prescribed(model, 'normal', 0., sign*.5/E_OVER_L)
        # Spin both endpoints together about z -> shear, fixed normal, no bend.
        w = np.array([[0., 0., -1.], [0., 0., -1.]])/G_OVER_L
        v = np.zeros_like(X)
        for _ in range(100):
            model.advance(model.x, v, w, .01)
        assert bool(~model.intact[0]) == should_break
    results['pressure_dependent_shear_strength'] = True
    return results


def oscillator(mode, steps, cycles=4):
    """Free two-body normal/twist mode, independently solved as a harmonic oscillator.

    Kick-drift-kick, no damping/external force. Isotropic scalar inertia only.
    This is a dynamics test fixture, not a general rigid-body integrator.
    """
    model = pair()
    amplitude = .02 if mode == 'normal' else .08
    stiffness = KN if mode == 'normal' else KT
    mass_or_inertia = 1. if mode == 'normal' else .1
    frequency = np.sqrt(2*stiffness/mass_or_inertia)
    duration = cycles*2*np.pi/frequency
    dt = duration/steps
    prescribed(model, mode, 0., amplitude)
    velocity, omega = np.zeros_like(X), np.zeros_like(X)
    q = amplitude
    initial = float(model.elastic_energy_per_bond().sum())
    f, t = model.loads()
    samples, max_energy_error, max_q_error = [], 0., 0.
    for step in range(steps+1):
        exact = amplitude*np.cos(frequency*dt*step)
        kinetic = .5*mass_or_inertia*np.sum((velocity if mode == 'normal' else omega)**2)
        energy = kinetic+model.elastic_energy_per_bond().sum()
        max_q_error = max(max_q_error, abs(q-exact)/amplitude)
        max_energy_error = max(max_energy_error, abs(energy/initial-1))
        samples.append([step*dt, q, exact, energy/initial-1])
        if step == steps:
            break
        if mode == 'normal':
            vh = velocity+.5*dt*f/mass_or_inertia
            model.advance(model.x+dt*vh, vh, omega, dt)
            f, t = model.loads()
            velocity = vh+.5*dt*f/mass_or_inertia
            q = model.x[1, 0]-model.x[0, 0]-1.
        else:
            wh = omega+.5*dt*t/mass_or_inertia
            q += dt*(wh[1, 0]-wh[0, 0])
            model.advance(model.x, velocity, wh, dt)
            f, t = model.loads()
            omega = wh+.5*dt*t/mass_or_inertia
    return dict(steps=steps, dt=float(dt), max_relative_trajectory_error=float(max_q_error),
                max_relative_energy_error=float(max_energy_error)), np.array(samples)


def dynamics_checks():
    report, curves = {}, {}
    for mode in ('normal', 'twist'):
        runs = []
        for steps in (400, 800, 1600):
            metrics, curve = oscillator(mode, steps)
            runs.append(metrics)
        for key in ('max_relative_trajectory_error', 'max_relative_energy_error'):
            values = [r[key] for r in runs]
            assert values[-1] < 1e-3 and values[0]/values[1] > 3.5 and values[1]/values[2] > 3.5, values
        report[mode], curves[mode] = runs, curve
    return report, curves


def input_checks():
    failed = 0
    for params in (dict(pairs=np.array([[0, 0]])), dict(pairs=np.array([[0, 1], [1, 0]])),
                   dict(radii=0.), dict(kbar_n=-1.), dict(reference=np.zeros_like(X))):
        try:
            pair(**params)
        except ValueError:
            failed += 1
    assert failed == 5
    # Deep initial sphere overlap produces no force in the bond component.
    model = pair(radii=4.)
    assert np.linalg.norm(model.loads()[0]) == 0
    before = bond_values(model).copy()
    try:
        model.advance(model.x, np.zeros_like(X), np.ones_like(X), 1.)
    except ValueError:
        pass
    else:
        raise AssertionError('Oversized rotation was not rejected')
    assert np.array_equal(bond_values(model), before) and model.time == 0
    return dict(invalid_initializations_rejected=failed, deep_initial_overlap_force=0.,
                oversized_rotation_rejected_without_mutation=True)


def plot_results(out, modes, dynamics, report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), constrained_layout=True)
    for ax, (mode, values) in zip(axes.flat, modes.items()):
        ax.plot(values[:, 0], values[:, 2], color='black', label='analytic')
        ax.plot(values[:, 0], values[:, 1], 'o', ms=3, label='implemented')
        ax.set(title=mode, xlabel='relative displacement / angle', ylabel='force / moment')
        ax.grid(alpha=.25)
        ax.legend()
    ax = axes.flat[4]
    values = dynamics['normal']
    ax.plot(values[:, 0], values[:, 2], color='black', label='analytic')
    ax.plot(values[::20, 0], values[::20, 1], '.', label='implemented')
    ax.set(title='Undamped normal oscillation', xlabel='time', ylabel='extension')
    ax.legend()
    ax = axes.flat[5]
    for mode, runs in report['dynamics'].items():
        ax.loglog([r['dt'] for r in runs], [r['max_relative_energy_error'] for r in runs], 'o-', label=mode)
    ax.set(title='Energy error under time-step refinement', xlabel='dt', ylabel='max relative error')
    ax.legend()
    ax.grid(alpha=.25)
    fig.suptitle('Parallel-bond mechanics verification — synthetic fixtures, no Power Foam rendering')
    fig.savefig(out/'mechanics.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=Path('output/bonded_dem_stage1'))
    args = parser.parse_args()
    report = dict(scope='Stage 1: our parallel-bond component and analytic fixtures; no collision or Foam coupling.',
                  numpy=np.__version__, scipy=scipy.__version__)
    report['inputs'] = input_checks()
    print('Input validation and stress-free overlap: passed', flush=True)
    report['analytic'], curves = analytic_checks()
    print('Four loading modes and reversible unloading: passed', flush=True)
    report['rotation'] = rotation_checks()
    report['network'] = network_checks()
    print('Rigid motion, rotating history, network balance: passed', flush=True)
    report['failure'] = failure_checks()
    print('Brittle thresholds, pressure dependence and no healing: passed', flush=True)
    report['dynamics'], dynamic_curves = dynamics_checks()
    print('Translational and rotational oscillation / dt refinement: passed', flush=True)
    report['source_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (Path(__file__), Path(__file__).with_name('bonded_dem_core.py'))}
    args.out.mkdir(parents=True, exist_ok=True)
    plot_results(args.out, curves, dynamic_curves, report)
    (args.out/'results.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    np.savez_compressed(args.out/'curves.npz', **curves,
                        **{f'oscillation_{k}': v for k, v in dynamic_curves.items()})
    print(json.dumps(report, indent=2, allow_nan=False))
    print(f'Artifacts: {args.out}', flush=True)


if __name__ == '__main__':
    main()
