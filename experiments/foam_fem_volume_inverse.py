"""Matched four/five-mode FEM fitting, serial, with direct bounded log parameters.

Both fits finish before held-out observations are opened. Fresh optimizer runs
share initial pair parameters, regularization per parameter, bounds and budgets.
No tanh coordinate or raw-gradient convergence shortcut is used.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize

from experiments.bread_inverse_pilot import objective
from experiments.foam_dem_moduli import write
from experiments.foam_fem_inverse import TRAIN, TEST
from experiments.foam_tetra_dynamics_probe import make_model


def projected_gradient(x, gradient, bound):
    return x-np.clip(x-gradient, -bound, bound)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, default=Path('output/foam_fem/reference_edge007'))
    parser.add_argument('--old-inverse', type=Path, default=Path('output/foam_fem/inverse_exploratory_v1'))
    parser.add_argument('--dynamics-check', type=Path, default=Path('output/foam_fem/tetra_dynamics_v1/report.json'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max-iterations', type=int, default=120)
    parser.add_argument('--max-evaluations', type=int, default=240)
    parser.add_argument('--multiplier-bound', type=float, default=20.)
    parser.add_argument('--gradient-tolerance', type=float, default=1e-6)
    parser.add_argument('--k-v', type=float, default=400.)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if (min(args.max_iterations, args.max_evaluations) <= 0 or args.multiplier_bound <= 1
            or args.k_v <= 0 or args.gradient_tolerance <= 0):
        parser.error('Positive budgets and k_v; multiplier bound > 1 required.')
    torch.set_num_threads(1)
    assert json.loads(args.dynamics_check.read_text())['complete'], 'Complete dynamics/gradient checks first.'
    reference = json.loads((args.reference/'report.json').read_text())
    assert reference['complete'], 'Incomplete FEM data.'
    old_config = json.loads((args.old_inverse/'config.json').read_text())
    assert Path(old_config['reference']).resolve() == args.reference.resolve()
    with np.load(args.reference/'patch.npz') as a:
        geometry = dict(a)
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    def status(stage, **details):
        row = dict(stage=stage, wall_elapsed_seconds=time.perf_counter()-started, updated_unix=time.time(), **details)
        write(args.out/'status.json', row)
        print(json.dumps(row), flush=True)
    models = {name: make_model(geometry, volume=name == 'five', k_v=args.k_v)
              for name in ('four', 'five')}
    n = len(geometry['points'])
    tensor = models['four'].reference.tensor
    free = ~models['four'].reference.fixed
    datasets = {}
    def load_case(name):
        with np.load(args.reference/f'{name}_target.npz') as a:
            datasets[name] = dict(pattern=tensor(a['force_pattern']), target=tensor(a['positions']),
                pulse=float(a['pulse_duration']), dt=float(a['dt']))
    for name in TRAIN:
        load_case(name)
    steps = {name: int(old_config['internal_substeps'][name]) for name in TRAIN}
    bound = float(np.log(args.multiplier_bound))
    regularization = 1e-6
    config = dict(reference=str(args.reference), training_cases=list(TRAIN), held_out_cases=list(TEST),
        modes=dict(four=4, five=5), parameters=dict(four=4*n, five=5*n),
        k_v=args.k_v, triangulation='A: fixed Cartesian Freudenthal subdivision',
        multiplier_bounds=[1/args.multiplier_bound, args.multiplier_bound],
        initialization='All four shared pair log multipliers zero in both fits; fifth log multiplier zero.',
        restarts_per_model=1, optimizer='SciPy L-BFGS-B, direct log multipliers',
        max_iterations=args.max_iterations, max_evaluations=args.max_evaluations,
        convergence=dict(projected_gradient_infinity_tolerance=args.gradient_tolerance,
                         ftol=0., loss_stagnation='Logged, not treated as convergence without the gradient criterion.'),
        regularization=f'{regularization} * sum(theta**2) / {4*n}; same coefficient per parameter in both models',
        time_refinement_threshold=.005, reference_checks=reference,
        protocol='Same budgets are upper bounds; a converged fit may stop early. Both fits freeze before held-out data are read.',
        source_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                       for name in ('foam_fem_volume_inverse.py', 'bonded_dem_differentiable.py',
                                    'foam_tetra_volume.py', 'bonded_dem_finite_gpu.py')})
    write(args.out/'config.json', config)

    def rollout(model, theta, name, substeps):
        data = datasets[name]
        count = len(data['target'])-1
        dt = data['dt']/substeps
        times = (np.arange(count*substeps)+.5)*dt
        pulse = np.where(times < data['pulse'], np.sin(np.pi*times/data['pulse'])**2, 0.)
        loads = tensor(pulse)[:, None, None]*data['pattern'][None]
        model.stats.clear()
        positions, _ = model.rollout(theta, loads, dt)
        return positions[::substeps]

    try:
        # Start from already verified old step counts, not a repeated 1,2,4,...
        # ladder. Require both initial models to pass at the same chosen counts.
        time_checks = []
        for name in TRAIN:
            for attempt in range(3):
                passed = True
                for kind, model in models.items():
                    status('initial_time_check', model=kind, case=name, substeps=steps[name])
                    theta = tensor(np.zeros((n, model.parameter_count)))
                    with torch.no_grad():
                        coarse = rollout(model, theta, name, steps[name])
                        fine = rollout(model, theta, name, 2*steps[name])
                    difference = float(objective(coarse, fine, model.reference.p, free).sqrt())
                    row = dict(model=kind, case=name, substeps=steps[name], normalized_difference=difference)
                    time_checks.append(row)
                    write(args.out/'initial_time_checks.json', time_checks)
                    status('initial_time_check_finished', **row)
                    passed &= difference < .005
                if passed:
                    break
                steps[name] *= 2
            else:
                raise RuntimeError('Common time step failed validation; no training started.')
        config['internal_substeps'] = steps
        write(args.out/'config.json', config)
        if args.prepare_only:
            status('prepared')
            return

        fits = {}
        for kind, model in models.items():
            folder = args.out/kind
            folder.mkdir()
            shape = (n, model.parameter_count)
            evaluations, iterations, cache = [], [], {}
            fit_start = time.perf_counter()
            def value_gradient(vector):
                theta = tensor(vector.reshape(shape)).requires_grad_()
                losses, clip_seconds = [], {}
                for name in TRAIN:
                    status('training', model=kind, evaluation=len(evaluations),
                           accepted_iterations=len(iterations), case=name)
                    tick = time.perf_counter()
                    prediction = rollout(model, theta, name, steps[name])
                    loss = objective(prediction, datasets[name]['target'], model.reference.p, free)
                    (loss/len(TRAIN)).backward()
                    losses.append(float(loss.detach()))
                    clip_seconds[name] = time.perf_counter()-tick
                    status('training_clip_finished', model=kind, evaluation=len(evaluations),
                           case=name, rmse=float(loss.detach().sqrt()), seconds=clip_seconds[name])
                penalty = regularization*theta.square().sum()/(4*n)
                penalty.backward()
                gradient = theta.grad.detach().numpy().ravel().copy()
                value = float(np.mean(losses))+float(penalty.detach())
                if not np.isfinite(value) or not np.isfinite(gradient).all():
                    raise RuntimeError('Nonfinite objective/gradient.')
                row = dict(evaluation=len(evaluations), elapsed_seconds=time.perf_counter()-fit_start,
                    objective=value, train_rmse=float(np.sqrt(np.mean(losses))),
                    per_case_rmse=dict(zip(TRAIN, np.sqrt(losses).tolist())),
                    projected_gradient_max=float(np.max(np.abs(projected_gradient(vector, gradient, bound)))),
                    gradient_max=float(np.max(np.abs(gradient))), clip_seconds=clip_seconds)
                evaluations.append(row)
                with (folder/'training.jsonl').open('a') as handle:
                    handle.write(json.dumps(row)+'\n')
                cache.update(x=vector.copy(), value=value, gradient=gradient.copy(), row=row)
                status('objective_finished', model=kind, **row)
                return value, gradient

            def checkpoint(vector):
                if not np.array_equal(cache['x'], vector):
                    value_gradient(vector)
                row = dict(iteration=len(iterations)+1, **cache['row'])
                iterations.append(row)
                temporary = folder/'parameters.tmp'
                with temporary.open('wb') as handle:
                    np.savez_compressed(handle, theta=vector.reshape(shape))
                temporary.replace(folder/'parameters.npz')
                write(folder/'checkpoints.json', iterations)
                status('checkpoint', model=kind, **row)

            fitted = minimize(value_gradient, np.zeros(np.prod(shape)), jac=True, method='L-BFGS-B',
                bounds=[(-bound, bound)]*int(np.prod(shape)), callback=checkpoint,
                options=dict(maxiter=args.max_iterations, maxfun=args.max_evaluations,
                             gtol=args.gradient_tolerance, ftol=0., maxls=20, maxcor=10))
            if not np.array_equal(cache['x'], fitted.x):
                value_gradient(fitted.x)
            pg = float(np.max(np.abs(projected_gradient(fitted.x, cache['gradient'], bound))))
            converged = pg <= args.gradient_tolerance
            result = dict(converged=converged, stop_message=str(fitted.message), optimizer_success=bool(fitted.success),
                iterations=int(fitted.nit), evaluations=len(evaluations), final_objective=cache['row'],
                projected_gradient_max=pg, multiplier_range=np.exp([fitted.x.min(), fitted.x.max()]).tolist(),
                fraction_near_bound=float(np.mean(np.abs(fitted.x) > .99*bound)),
                recent_accepted_loss_change=(abs(iterations[-1]['objective']-iterations[-5]['objective'])
                                             if len(iterations) >= 5 else None),
                elapsed_seconds=time.perf_counter()-fit_start)
            np.savez_compressed(folder/'parameters.npz', theta=fitted.x.reshape(shape))
            write(folder/'fit.json', result)
            fits[kind] = result
            status('fit_frozen', model=kind, converged=converged, stop_message=result['stop_message'])

        # Selection, stopping and bounds above never inspect held-out targets.
        for name in TEST:
            load_case(name)
        for kind, model in models.items():
            folder = args.out/kind
            theta = tensor(np.load(folder/'parameters.npz')['theta'])
            metrics = {}
            for name in TRAIN+TEST:
                count = steps[name] if name in TRAIN else max(steps.values())
                status('frozen_evaluation', model=kind, case=name, substeps=count)
                with torch.no_grad():
                    prediction = rollout(model, theta, name, count)
                    finer = rollout(model, theta, name, 2*count)
                metric = dict(normalized_rmse=float(objective(prediction, datasets[name]['target'], model.reference.p, free).sqrt()),
                    refined_normalized_rmse=float(objective(finer, datasets[name]['target'], model.reference.p, free).sqrt()),
                    time_refinement_difference=float(objective(prediction, finer, model.reference.p, free).sqrt()))
                metrics[name] = metric
                np.savez_compressed(folder/f'{name}.npz', predicted=prediction.numpy(), refined=finer.numpy(),
                    target=datasets[name]['target'].numpy(), dt=datasets[name]['dt'])
                write(folder/'metrics.json', metrics)
            fits[kind].update(metrics=metrics, time_refinement_passed=all(
                m['time_refinement_difference'] < .005 for m in metrics.values()))
            write(folder/'report.json', fits[kind])
        write(args.out/'report.json', dict(complete=True, config=config, models=fits,
            limitations='One matched initialization per model. No capacity-limit conclusion if unconverged. These held-out cases become validation cases if used to adopt the volume model.'))
        status('complete')
    except BaseException as error:
        status('failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
