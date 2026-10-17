"""Short full-trajectory adjoint/forward checks for pair plus tetra volume.

Synthetic target trajectories here are implementation tests, not a FEM fit.
The independent forward root uses numerical Jacobians and the energy gradient
for volume force; it does not call the custom implicit backward or solve().
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import root

from experiments.bonded_dem_differentiable import DifferentiableBonds, STATE_NAMES
from experiments.bonded_dem_finite_gpu import FiniteBondsGPU, exp_rotation
from experiments.bread_inverse_pilot import objective
from experiments.foam_dem_moduli import write
from experiments.foam_tetra_volume import NodalTetraVolume, grid_tetrahedra


def make_model(geometry, volume=True, k_v=400.):
    term = None
    if volume:
        term = NodalTetraVolume(geometry['points'],
            grid_tetrahedra(geometry['points'], geometry['radii']),
            float(np.sum(geometry['masses'])), k_v)
    return DifferentiableBonds(**geometry, volume_term=term, device='cpu')


def independent_rollout(model, theta, loads, dt):
    state = model.initial_state()
    positions = [state[0]]
    residuals = []
    for load in loads:
        bound = model.bind(theta, state)
        def trial(y):
            result = FiniteBondsGPU.trial(bound, y)
            mid = (state[0]+.5*y[:, :3]).detach().requires_grad_()
            model.volume_term.validate_configuration(mid)
            model.volume_term.validate_configuration(result['x'])
            energy = model.volume_term.evaluate(mid, theta[:, 4])[0]
            force = -torch.autograd.grad(energy, mid)[0].detach()
            result['force'] += torch.cat((force, torch.zeros_like(force)), dim=1)
            return result
        def residual(z):
            y = model.increment(model.reference.tensor(z))
            value = model.reference.mass*(y-dt*state[1])-.5*dt*dt*(trial(y)['force']+load)
            scale = (model.reference.mass+.25*dt*dt*model.diagonal)*model.units
            return (value/scale).flatten()[model.free].detach().numpy()
        solved = root(residual, np.zeros(len(model.free)), method='hybr', options={'xtol': 1e-10})
        defect = float(np.max(np.abs(residual(solved.x))))
        assert defect < 1e-11, (solved.message, defect)
        residuals.append(defect)
        y = model.increment(model.reference.tensor(solved.x))
        result = trial(y)
        result.update(v=2*y/dt-state[1], rotation=exp_rotation(y[:, 3:])@state[2])
        state = tuple(result[name].detach() for name in STATE_NAMES)
        positions.append(state[0])
    return torch.stack(positions), state, residuals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, default=Path('output/foam_fem/reference_edge007'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    start = time.perf_counter()
    def stage(name):
        print(name, flush=True)
        write(args.out/'status.json', dict(stage=name, elapsed_seconds=time.perf_counter()-start))
    try:
        with np.load(args.reference/'patch.npz') as a:
            geometry = dict(a)
        model = make_model(geometry)
        tensor = model.reference.tensor
        rng = np.random.default_rng(4108)
        theta = tensor(.15*rng.normal(size=(len(geometry['points']), 5)))
        truth = theta+tensor(.2*rng.normal(size=theta.shape))
        patterns = []
        for name in ('train_down', 'train_pull', 'train_twist'):
            with np.load(args.reference/f'{name}_target.npz') as a:
                patterns.append(tensor(a['force_pattern']))
        frames, duration = 8, .12
        dt = duration/frames
        def loads_for(count):
            t = (torch.arange(count, dtype=torch.float64)+.5)/count
            pulse = torch.sin(np.pi*t)**2
            return pulse[:, None, None]*(patterns[0]+.7*patterns[1]+.6*patterns[2])[None]
        loads = loads_for(frames)
        stage('independent_forward')
        with torch.no_grad():
            predicted, state = model.rollout(theta, loads, dt)
        direct, direct_state, residuals = independent_rollout(model, theta, loads, dt)
        position_error = float((predicted-direct).abs().max())
        state_errors = {name: float((a-b).norm()/b.norm().clamp_min(1e-15))
                        for name, a, b in zip(STATE_NAMES, state, direct_state)}
        assert position_error < 1e-9 and max(state_errors.values()) < 1e-6
        assert float((predicted[:, model.reference.fixed]-model.reference.p[model.reference.fixed]).abs().max()) == 0

        stage('pair_regression_and_zero_volume_limit')
        pair = make_model(geometry, volume=False)
        zero_volume = theta.clone(); zero_volume[:, 4] = -700
        with torch.no_grad():
            original, _ = pair.rollout(theta[:, :4], loads, dt)
            limit, _ = model.rollout(zero_volume, loads, dt)
        limit_error = float((original-limit).abs().max())
        assert limit_error < 1e-9
        # Independent old solver checks the four-mode path was not altered.
        from experiments.bread_inverse_pilot import original_rollout
        old, _, _ = original_rollout(dict(geometry, device='cpu'), theta[:, :4], loads, dt)
        old_error = float((original-old).abs().max())
        assert old_error < 1e-8

        stage('full_trajectory_gradient')
        with torch.no_grad():
            target, _ = model.rollout(truth, loads, dt)
        active = theta.clone().requires_grad_()
        trajectory, _ = model.rollout(active, loads, dt)
        loss = objective(trajectory, target, model.reference.p, ~model.reference.fixed)
        gradient, = torch.autograd.grad(loss, active)
        checks = []
        for mode in range(5):
            direction = torch.zeros_like(theta)
            direction[:, mode] = tensor(rng.normal(size=len(theta)))
            direction /= direction.norm()
            analytical = float((gradient*direction).sum())
            differences = []
            for epsilon in (1e-3, 3e-4):
                values = []
                with torch.no_grad():
                    for sign in (-1, 1):
                        moved, _ = model.rollout(theta+sign*epsilon*direction, loads, dt)
                        values.append(float(objective(moved, target, model.reference.p, ~model.reference.fixed)))
                numerical = (values[1]-values[0])/(2*epsilon)
                error = abs(numerical-analytical)/max(abs(numerical), abs(analytical), 1e-12)
                differences.append(dict(epsilon=epsilon, numerical=numerical, relative_error=error))
            assert min(d['relative_error'] for d in differences) < 1e-4, (mode, differences)
            checks.append(dict(mode=('normal', 'shear', 'bend', 'twist', 'volume')[mode],
                               analytical=analytical, finite_differences=differences))
            print('gradient', checks[-1], flush=True)

        stage('time_refinement')
        trajectories = [predicted]
        with torch.no_grad():
            for count in (16, 32):
                trajectory, _ = model.rollout(theta, loads_for(count), duration/count)
                trajectories.append(trajectory[::count//frames])
        refinement = [float(objective(a, b, model.reference.p, ~model.reference.fixed).sqrt())
                      for a, b in zip(trajectories[:-1], trajectories[1:])]
        assert refinement[1] < refinement[0]/2, refinement
        record = dict(complete=True, elapsed_seconds=time.perf_counter()-start,
            frames=frames, duration=duration, k_v=400., tetrahedra=18,
            forward_max_position_difference=position_error, forward_state_relative_errors=state_errors,
            independent_solver_max_residual=max(residuals), old_pair_position_difference=old_error,
            zero_volume_pair_position_difference=limit_error, gradients=checks,
            refinement_normalized_trajectory_differences=refinement,
            gradient_norm_by_family=gradient.norm(dim=0).tolist(),
            max_newton_residual=max(row['scaled_residual'] for row in model.stats),
            scope='Synthetic short-clip implementation checks with FEM load patterns, not FEM fitting or held-out evaluation.')
        write(args.out/'report.json', record)
        np.savez_compressed(args.out/'trajectories.npz', implicit=predicted.numpy(),
                            independent=direct.numpy(), target=target.numpy(), theta=theta.numpy())
        stage('complete')
    except BaseException as error:
        write(args.out/'status.json', dict(stage='failed', error=repr(error)))
        raise


if __name__ == '__main__':
    main()
