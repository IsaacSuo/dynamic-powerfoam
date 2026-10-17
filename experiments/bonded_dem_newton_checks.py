"""Consistent Jacobian checks and matched-equation solver comparisons on CPU.

Small synthetic fixtures deliberately include offset contacts, pre-stress,
finite accumulated rotation, and masses spanning fourteen orders. These are
solver verification cases, not bread material calibration or full simulations.
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from experiments.bonded_dem_finite_gpu import FiniteBondsGPU, exp_rotation
from experiments.bonded_dem_newton import NewtonBonds, fgmres


def fixture(cls=NewtonBonds, prestress=True, minimum_mass=1e-14):
    rng = np.random.default_rng(906)
    p = np.array(np.meshgrid(np.arange(3), np.arange(2), np.arange(2))).reshape(3, -1).T*.4
    p += rng.normal(0, .008, p.shape)
    pairs = np.array([(i, j) for i in range(len(p)) for j in range(i+1, len(p))
                      if np.linalg.norm(p[i]-p[j]) < .65])
    radii = rng.uniform(.3, .4, len(p))
    area = rng.uniform(.008, .016, len(pairs))
    contacts = .5*(p[pairs[:, 0]]+p[pairs[:, 1]])+rng.normal(0, .025, (len(pairs), 3))
    mass = np.geomspace(minimum_mass, 1., len(p)); rng.shuffle(mass)
    s = cls(p, radii, pairs, area, mass, p[:, 0] < .1, contacts=contacts, device='cpu')
    if prestress:
        # Same committed nonzero histories for every solver, then rotate the
        # whole stressed fixture by 1.2 radians. This is not motion driving
        # during the subsequent freely solved step.
        y = torch.zeros_like(s.mass)
        y[:, 1] = .025*s.p[:, 0]
        y[:, 3] = .02*s.p[:, 2]
        s.commit(s.trial(y)); s.rotation = exp_rotation(y[:, 3:])@s.rotation
        axis = np.array([.3, -.4, .6]); axis /= np.linalg.norm(axis)
        initial = s.x.cpu().numpy().copy()
        for step in range(12):
            rm = Rotation.from_rotvec(axis*1.2*(step+.5)/12).as_matrix()
            re = Rotation.from_rotvec(axis*1.2*(step+1)/12).as_matrix()
            mid, end = initial@rm.T, initial@re.T
            y = s.tensor(np.c_[np.cross(axis*.1, mid), np.tile(axis*.1, (len(p), 1))])
            s.commit(s.trial(y, mid_positions=s.tensor(mid), end_positions=s.tensor(end)))
            s.rotation = exp_rotation(y[:, 3:])@s.rotation
    return s


def physical_state(s):
    return {k: getattr(s, k).clone() for k in ('x', 'v', 'rotation', 'normal', 'offset', 'fn', 'fs', 'mt', 'mb')}


def main():
    torch.set_num_threads(1)
    out = Path('output/bonded_dem_newton'); out.mkdir(parents=True, exist_ok=True)
    report = {}
    # Nonsymmetric system and variable preconditioner, independent dense answer.
    rng = np.random.default_rng(381)
    a = torch.tensor(rng.normal(size=(36, 36)), dtype=torch.float64)+8*torch.eye(36)
    b = torch.tensor(rng.normal(size=36), dtype=torch.float64)
    calls = [0]
    def variable_precondition(v):
        calls[0] += 1
        return v/(torch.diag(a)*(1+.08*np.sin(calls[0])))
    x, info = fgmres(lambda z: a@z, b, variable_precondition, tolerance=1e-10, restart=24)
    error = float((x-torch.linalg.solve(a, b)).norm()/torch.linalg.solve(a, b).norm())
    assert error < 1e-9
    report['fgmres_vs_dense'] = dict(relative_error=error, **info)

    s = fixture(); dt = .02
    external = s.tensor(rng.normal(0, .01, s.mass.shape)); external[s.fixed] = 0
    y = s.tensor(rng.normal(0, .006, s.mass.shape)); y[s.fixed] = 0
    direction = s.tensor(rng.normal(size=s.mass.shape)); direction[s.fixed] = 0
    snapshot = physical_state(s)
    force_error = float((s.midpoint_force(y)-s.trial(y)['force']).norm()/s.trial(y)['force'].norm())
    assert force_error < 1e-13
    report['midpoint_force_matches_original_trial'] = force_error
    s.prepare_scaling(s.trial(y), dt)
    direction *= s.units
    derivative = s.jacobian_vector(y, direction, dt, external)
    scaled_derivative = s.scale_residual(derivative)
    errors = []
    for h in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
        difference = (s.residual(y+h*direction, dt, external)-s.residual(y-h*direction, dt, external))/(2*h)
        errors.append(float(s.scale_residual(difference-derivative).norm()/scaled_derivative.norm()))
    assert min(errors) < 1e-8 and errors[0]/errors[1] > 80, errors
    assert all(torch.equal(getattr(s, k), v) for k, v in snapshot.items())
    report['full_jvp_centered_difference'] = dict(h=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6], relative_errors=errors,
                                                committed_state_unchanged=True)
    approximate = s._operator(direction, dt*dt/4)
    approximate[s.fixed] = direction[s.fixed]
    report['old_frozen_tangent_relative_difference'] = float(s.scale_residual(approximate-derivative).norm()/scaled_derivative.norm())
    print('Jacobian checks:', report, flush=True)

    runs = []
    for prestress in (False, True):
        reference = fixture(prestress=prestress)
        load = torch.zeros_like(reference.mass)
        load[:, 2] = -.03*reference.mass[:, 0]; load[reference.fixed] = 0
        solved = {}
        for name, backend, pc in [('dense_newton', 'dense', 'elastic'),
                                  ('newton_block', 'fgmres', 'block'),
                                  ('newton_elastic', 'fgmres', 'elastic'),
                                  ('previous_solver', None, None)]:
            solver = fixture(cls=FiniteBondsGPU if backend is None else NewtonBonds, prestress=prestress)
            energy0 = solver.energy()
            start = time.perf_counter()
            try:
                if backend is None:
                    work, stats = solver.step(dt, load, tolerance=2e-7)
                else:
                    work, stats = solver.step(dt, load, tolerance=2e-7, backend=backend,
                                              preconditioning=pc, linear_tolerance=1e-7)
                elapsed = time.perf_counter()-start
                difference = abs(solver.energy()-energy0-work)/max(energy0, abs(work), 1e-30)
                assert difference < 1e-5, difference
                result = dict(solver=name, prestress=prestress, success=True, elapsed_seconds=elapsed,
                              energy_balance_relative_error=difference, **stats)
                solved[name] = physical_state(solver)
            except RuntimeError as error:
                assert all(torch.equal(getattr(solver, k), v) for k, v in physical_state(reference).items())
                result = dict(solver=name, prestress=prestress, success=False,
                              elapsed_seconds=time.perf_counter()-start, reason=str(error))
            runs.append(result); print(json.dumps(result), flush=True)
        assert 'dense_newton' in solved and 'newton_elastic' in solved
        for name in ('newton_block', 'newton_elastic', 'previous_solver'):
            if name not in solved: continue
            position_error = float((solved[name]['x']-solved['dense_newton']['x']).abs().max())
            rotation_error = float((solved[name]['rotation']-solved['dense_newton']['rotation']).abs().max())
            assert position_error < 1e-7 and rotation_error < 1e-6, (name, position_error, rotation_error)
            for result in runs:
                if result['solver'] == name and result['prestress'] == prestress:
                    result['vs_dense_max_position_error'] = position_error
                    result['vs_dense_max_rotation_matrix_error'] = rotation_error
    report['same_step_solver_comparisons'] = runs
    report['source_sha256'] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ('bonded_dem_newton.py', 'bonded_dem_newton_checks.py', 'bonded_dem_finite_gpu.py')}
    (out/'checks.json').write_text(json.dumps(report, indent=2)+'\n')
    print(out/'checks.json', flush=True)


if __name__ == '__main__':
    main()
