"""Fit unchanged finite bonded DEM to checked independent FEM trajectories.

Two fits run serially. Only train_down/train_pull/train_twist enter optimization.
Each trajectory has its own verified internal time step; observation times and
force pulses are identical to FEM. Test trajectories are read after fitting.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.bonded_dem_differentiable import DifferentiableBonds
from experiments.bread_inverse_pilot import objective

TRAIN = ('train_down', 'train_pull', 'train_twist')
TEST = ('test_side', 'test_local', 'test_strong_down')


def write(path, data):
    path.write_text(json.dumps(data, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max-iterations', type=int, default=400)
    parser.add_argument('--checkpoint-every', type=int, default=5)
    parser.add_argument('--exploratory-reference', action='store_true',
                        help='Use a completed reference without requiring the historical 0.5%% mesh gate; record its checks unchanged.')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Resume this run from saved time checks and optimizer states.')
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = json.loads((args.reference/'report.json').read_text())
    assert report['complete'], 'FEM reference export is incomplete.'
    assert report['ready_for_training'] or args.exploratory_reference, 'Use --exploratory-reference to explicitly accept the recorded reference uncertainty.'
    assert args.max_iterations > 0 and args.checkpoint_every > 0
    args.out.mkdir(parents=True, exist_ok=args.resume)
    if args.resume and (args.out/'config.json').exists():
        old_config = json.loads((args.out/'config.json').read_text())
        assert old_config['reference'] == str(args.reference), 'Cannot resume with different targets.'
    with np.load(args.reference/'patch.npz') as a:
        kwargs = dict(a)
    model = DifferentiableBonds(**kwargs, device='cpu')
    tensor = model.reference.tensor
    zero = tensor(np.zeros((16, 4)))
    free = ~model.reference.fixed
    patterns, observations, pulses = {}, {}, {}
    dt = None
    for name in TRAIN:
        with np.load(args.reference/f'{name}_target.npz') as a:
            patterns[name] = tensor(a['force_pattern'])
            observations[name] = tensor(a['positions'])
            pulses[name] = float(a['pulse_duration'])
            if dt is None: dt = float(a['dt'])
            assert dt == float(a['dt'])

    def rollout(theta, name, substeps):
        times = (np.arange(24*substeps)+.5)*dt/substeps
        fraction = np.where(times < pulses[name], np.sin(np.pi*times/pulses[name])**2, 0.)
        loads = tensor(fraction)[:, None, None]*patterns[name][None]
        positions, _ = model.rollout(theta, loads, dt/substeps)
        return positions[::substeps]

    def choose_steps(theta, name):
        history = []
        with torch.no_grad():
            print('Initial DEM time check', name, 'substeps=1', flush=True)
            previous = rollout(theta, name, 1)
            for fine in (2, 4, 8, 16, 32, 64):
                started = time.perf_counter()
                print('Initial DEM time check', name, f'substeps={fine}', flush=True)
                prediction = rollout(theta, name, fine)
                error = float(objective(previous, prediction, model.reference.p, free).sqrt())
                history.append(dict(coarse=fine//2, fine=fine, normalized_difference=error,
                                    elapsed_seconds=time.perf_counter()-started))
                print('Time check', name, history[-1], flush=True)
                if error < .0025:
                    # Use the finer result for extra margin as parameters evolve.
                    return fine, history
                previous = prediction
        raise RuntimeError(f'Time discretization did not converge for {name}')

    steps, time_checks = {}, {}
    write(args.out/'reference_checks.json', report)
    print('Reference accepted', str(args.reference),
          'exploratory=', args.exploratory_reference,
          'mesh_differences=', report['mesh_refinement_errors'], flush=True)
    if args.resume and (args.out/'initial_time_checks.json').exists():
        time_checks = json.loads((args.out/'initial_time_checks.json').read_text())
    for name in TRAIN:
        if name in time_checks:
            assert time_checks[name][-1]['normalized_difference'] < .0025
            steps[name] = time_checks[name][-1]['fine']
        else:
            steps[name], time_checks[name] = choose_steps(zero, name)
            write(args.out/'initial_time_checks.json', time_checks)
    config = dict(reference=str(args.reference), train_cases=list(TRAIN), test_cases=list(TEST),
        parameters=dict(shared=4, per_cell=64), internal_substeps=steps,
        initialization='All stiffness multipliers 1', theta='1.5*tanh(raw)', regularization=1e-6,
        max_iterations=args.max_iterations, dt_observation=dt, frames=24,
        checkpoint_every=args.checkpoint_every,
        reference_policy='exploratory' if args.exploratory_reference else 'historical_0.5_percent_gate',
        reference_mesh_differences=report['mesh_refinement_errors'],
        force_transfer='Reference surface quadrature supplies each cell force and moment.',
        target='Independent continuum FEM. No constitutive parameters used as training labels.',
        stopped_on_test=False, final_time_refinement_threshold=.005)
    write(args.out/'config.json', config)
    write(args.out/'initial_time_checks.json', time_checks)
    if args.prepare_only:
        return
    summaries = {}
    for kind, shape in [('shared', (4,)), ('per_cell', (16, 4))]:
        folder = args.out/kind
        folder.mkdir(exist_ok=args.resume)
        if args.resume and (folder/'report.json').exists():
            previous_report = json.loads((folder/'report.json').read_text())
            if previous_report['iterations'] >= args.max_iterations or previous_report['stop_reason'] == 'optimizer_stopping_condition':
                summaries[kind] = previous_report
                continue
        raw = torch.zeros(shape, dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.LBFGS([raw], lr=1., max_iter=40, max_eval=80, history_size=10,
            line_search_fn='strong_wolfe', tolerance_grad=1e-8, tolerance_change=1e-12)
        log, checkpoints = [], []
        elapsed_before = 0.
        if args.resume and (folder/'optimizer.pt').exists():
            saved = torch.load(folder/'optimizer.pt', map_location='cpu', weights_only=True)
            assert saved['steps'] == steps
            with torch.no_grad(): raw.copy_(saved['raw'])
            optimizer.load_state_dict(saved['optimizer'])
            if (folder/'training.jsonl').exists():
                log = [json.loads(line) for line in (folder/'training.jsonl').read_text().splitlines()]
                elapsed_before = log[-1]['elapsed_seconds']
            if (folder/'checkpoints.json').exists():
                checkpoints = json.loads((folder/'checkpoints.json').read_text())
            print('Resumed', kind, optimizer.state[raw]['n_iter'], flush=True)
        start = time.perf_counter()

        def parameters():
            theta = 1.5*raw.tanh()
            return theta.expand(16, 4) if kind == 'shared' else theta

        def closure():
            optimizer.zero_grad()
            losses = []
            # Backpropagate clips one at a time to release their state graphs.
            for name in TRAIN:
                clip_started = time.perf_counter()
                print(kind, 'Evaluation', len(log), name, 'forward/backward started', flush=True)
                theta = parameters()
                prediction = rollout(theta, name, steps[name])
                loss = objective(prediction, observations[name], model.reference.p, free)
                (loss/len(TRAIN)).backward()
                losses.append(float(loss.detach()))
                print(kind, 'Evaluation', len(log), name,
                      'rmse=', float(loss.detach().sqrt()),
                      'seconds=', time.perf_counter()-clip_started, flush=True)
            penalty = 1e-6*parameters().square().mean()
            penalty.backward()
            assert torch.isfinite(raw.grad).all()
            value = float(np.mean(losses))+float(penalty.detach())
            row = dict(evaluation=len(log), elapsed_seconds=elapsed_before+time.perf_counter()-start,
                train_rmse=float(np.sqrt(np.mean(losses))), per_case_rmse=np.sqrt(losses).tolist(),
                total_loss=value, gradient_max=float(raw.grad.abs().max()), gradient_norm=float(raw.grad.norm()))
            log.append(row)
            with (folder/'training.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
            print(kind, 'Training', json.dumps(row), flush=True)
            return tensor(value)

        stop_reason = 'iteration_cap'
        while optimizer.state[raw].get('n_iter', 0) < args.max_iterations:
            before = optimizer.state[raw].get('n_iter', 0)
            requested = min(args.checkpoint_every, args.max_iterations-before)
            optimizer.param_groups[0]['max_iter'] = requested
            optimizer.param_groups[0]['max_eval'] = 2*requested
            optimizer.step(closure)
            count = optimizer.state[raw]['n_iter']
            theta = parameters().detach()
            torch.save(dict(raw=raw.detach(), optimizer=optimizer.state_dict(), steps=steps), folder/'optimizer.pt')
            np.savez_compressed(folder/'parameters.npz', theta=theta.numpy())
            checkpoint = dict(iterations=count, objective_evaluations=len(log), last_evaluation=log[-1])
            checkpoints.append(checkpoint)
            write(folder/'checkpoints.json', checkpoints)
            print(kind, 'Checkpoint', json.dumps(checkpoint), flush=True)
            if count-before < requested:
                stop_reason = 'optimizer_stopping_condition'
                break
        closure()
        theta = parameters().detach()
        # Load held-out observations only after the parameter fit is frozen.
        for name in TEST:
            with np.load(args.reference/f'{name}_target.npz') as a:
                patterns[name] = tensor(a['force_pattern'])
                observations[name] = tensor(a['positions'])
                pulses[name] = float(a['pulse_duration'])
        metrics, refinement = {}, {}
        with torch.no_grad():
            for name in TRAIN+TEST:
                print(kind, 'Frozen evaluation', name, flush=True)
                n = steps[name] if name in TRAIN else max(steps.values())
                prediction = rollout(theta, name, n)
                finer = rollout(theta, name, 2*n)
                refinement[name] = float(objective(prediction, finer, model.reference.p, free).sqrt())
                metrics[name] = dict(normalized_rmse=float(objective(prediction, observations[name], model.reference.p, free).sqrt()),
                    refined_normalized_rmse=float(objective(finer, observations[name], model.reference.p, free).sqrt()))
                np.savez_compressed(folder/f'{name}.npz', target=observations[name].numpy(),
                    predicted=prediction.numpy(), refined=finer.numpy(), dt=dt)
        final = dict(iterations=optimizer.state[raw]['n_iter'], stop_reason=stop_reason,
            final_objective=log[-1], metrics=metrics, time_refinement=refinement,
            time_refinement_passed=max(refinement.values()) < .005,
            multiplier_range=[float(theta.exp().min()), float(theta.exp().max())],
            fraction_near_parameter_bound=float((theta.abs() > 1.485).double().mean()),
            elapsed_seconds=elapsed_before+time.perf_counter()-start)
        summaries[kind] = final
        write(folder/'report.json', final)
    write(args.out/'report.json', dict(complete=True, config=config, models=summaries,
        caution='Iteration caps and failed time checks preclude a capacity-limit conclusion.'))
    print('Complete', args.out, flush=True)


if __name__ == '__main__':
    main()
