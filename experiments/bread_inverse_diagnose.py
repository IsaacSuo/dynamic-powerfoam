"""Serial optimization/initialization diagnostic using unchanged training data.

The known generating parameters are used only for a forward sanity check.
Held-out loads are monitored at fixed checkpoints, never used for optimization,
stopping, or choosing a model. An iteration cap is not a convergence claim.
"""
import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.bonded_dem_differentiable import DifferentiableBonds
from experiments.bread_inverse_pilot import objective


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--pilot', type=Path, default=Path('output/bonded_dem_inverse/pilot_16cells'))
    parser.add_argument('--capacity', type=Path, default=Path('output/bonded_dem_inverse/capacity_16cells'))
    parser.add_argument('--total-iterations', type=int, default=160)
    args = parser.parse_args()
    assert args.total_iterations > 40 and args.total_iterations % 40 == 0
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=False)
    pilot_config = json.loads((args.pilot/'config.json').read_text())
    for name, digest in pilot_config['source_sha256'].items():
        assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() == digest, name
    with np.load(args.pilot/'patch.npz') as a:
        kwargs = {k: a[k] for k in ('points', 'radii', 'pairs', 'areas', 'masses', 'fixed', 'contacts')}
        original_ids = a['original_ids']
    model = DifferentiableBonds(**kwargs, device='cpu')
    tensor = model.reference.tensor
    free_cells = ~model.reference.fixed
    dt = pilot_config['timing']['dt']
    cases, observations = {}, {}
    for name in ('train_down', 'train_pull', 'test_side', 'test_twist', 'test_local', 'test_strong_down'):
        source = args.pilot if name in ('train_down', 'train_pull', 'test_side') else args.capacity
        with np.load(source/f'{name}_target.npz') as a:
            assert float(a['dt']) == dt
            cases[name] = tensor(a['loads'])
            observations[name] = tensor(a['positions'])
            if name == 'train_down':
                known_theta = tensor(a['true_parameters'])
    with np.load(args.pilot/'learned_parameters.npz') as a:
        old_theta = tensor(a['log_multipliers'])
    config = dict(pilot=str(args.pilot), capacity=str(args.capacity),
        training_cases=['train_down', 'train_pull'], total_iteration_cap=args.total_iterations,
        checkpoint_every=40, regularization=1e-6, parameterization='theta=1.5*tanh(raw)',
        starts=[dict(name='continued', previous_iterations=40, initialization='frozen pilot parameters'),
                dict(name='seed_1729', previous_iterations=0, initialization='raw uniform(-0.2,0.2)', seed=1729),
                dict(name='seed_3141', previous_iterations=0, initialization='raw uniform(-0.2,0.2)', seed=3141)],
        optimizer=dict(name='LBFGS', history_size=10, line_search='strong_wolfe',
            tolerance_grad=1e-8, tolerance_change=1e-12,
            note='Optimizer history persists between new 40-iteration blocks. '
                 'Old pilot optimizer state was not saved; continuation restarts LBFGS history.'),
        evaluation='All six loads at fixed checkpoints. Tests never affect training or stopping; no best-test selection.',
        scope='Same DEM synthetic targets; finite optimization and three starts do not prove identifiability.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_json(args.out/'config.json', config)

    def evaluate(theta):
        metrics, positions = {}, {}
        with torch.no_grad():
            for name, loads in cases.items():
                predicted, _ = model.rollout(theta, loads, dt)
                assert torch.isfinite(predicted).all()
                assert torch.equal(predicted[:, ~free_cells], observations[name][:, ~free_cells])
                metrics[name] = float(objective(predicted, observations[name], model.reference.p, free_cells).sqrt())
                positions[name] = predicted.numpy()
        metrics['train_combined'] = float(np.sqrt(np.mean([metrics[n]**2 for n in config['training_cases']])))
        return metrics, positions

    # A new held-out local load must also be reproduced with the known parameters.
    with torch.no_grad():
        reference, _ = model.rollout(known_theta, cases['test_local'], dt)
        sanity_error = float(objective(reference, observations['test_local'], model.reference.p, free_cells).sqrt())
    assert sanity_error < 1e-6, sanity_error
    write_json(args.out/'checks.json', dict(known_parameters_local_normalized_rmse=sanity_error,
        source_hashes_match=True, known_parameters_used_in_training=False))
    baseline, _ = evaluate(old_theta)
    write_json(args.out/'baseline.json', baseline)
    print('Baseline', json.dumps(baseline), flush=True)
    finals, predictions, parameters = {}, {}, {}
    for specification in config['starts']:
        name = specification['name']
        run = args.out/name
        run.mkdir()
        prior = specification['previous_iterations']
        if name == 'continued':
            initial = torch.atanh(old_theta/1.5)
        else:
            initial = tensor(np.random.default_rng(specification['seed']).uniform(-.2, .2, old_theta.shape))
        raw = initial.clone().requires_grad_(True)
        optimizer = torch.optim.LBFGS([raw], lr=1., max_iter=40, max_eval=80, history_size=10,
            line_search_fn='strong_wolfe', tolerance_grad=1e-8, tolerance_change=1e-12)
        history, checkpoints = [], []
        start = time.perf_counter()

        def closure():
            optimizer.zero_grad()
            theta = 1.5*raw.tanh()
            errors = []
            for case in config['training_cases']:
                predicted, _ = model.rollout(theta, cases[case], dt)
                errors.append(objective(predicted, observations[case], model.reference.p, free_cells))
            motion_loss = torch.stack(errors).mean()
            penalty = 1e-6*theta.square().mean()
            total = motion_loss+penalty
            total.backward()
            assert torch.isfinite(raw.grad).all()
            row = dict(evaluation=len(history), elapsed_seconds=time.perf_counter()-start,
                motion_loss=float(motion_loss.detach()), train_rmse=float(motion_loss.detach().sqrt()),
                total_loss=float(total.detach()), gradient_norm=float(raw.grad.norm()),
                gradient_max=float(raw.grad.abs().max()))
            history.append(row)
            with (run/'training.jsonl').open('a') as f:
                f.write(json.dumps(row)+'\n')
            print(name, 'Training', json.dumps(row), flush=True)
            return total

        print('Starting', name, flush=True)
        stop_reason = 'iteration_cap'
        while prior+optimizer.state[raw].get('n_iter', 0) < args.total_iterations:
            before = optimizer.state[raw].get('n_iter', 0)
            optimizer.step(closure)
            accepted = optimizer.state[raw]['n_iter']
            total_iterations = prior+accepted
            theta = (1.5*raw.tanh()).detach()
            metrics, positions = evaluate(theta)
            checkpoint = dict(total_iterations=total_iterations, new_iterations=accepted,
                objective_evaluations=len(history), metrics=metrics,
                elapsed_seconds=time.perf_counter()-start)
            checkpoints.append(checkpoint)
            write_json(run/'checkpoints.json', checkpoints)
            np.savez_compressed(run/f'iteration_{total_iterations:03d}.npz', theta=theta.numpy(),
                                **positions, original_ids=original_ids)
            # Save the actual optimizer state to allow genuine future continuation.
            torch.save(dict(raw=raw.detach(), optimizer=optimizer.state_dict(),
                            total_iterations=total_iterations), run/'optimizer.pt')
            print(name, 'Checkpoint', json.dumps(checkpoint), flush=True)
            if accepted-before < 40:
                stop_reason = 'optimizer_stopping_condition'
                break
        closure()  # Explicit final objective and gradient at accepted parameters.
        final = dict(**checkpoints[-1], stop_reason=stop_reason,
            objective_evaluations_including_final_check=len(history), final_objective=history[-1],
            optimizer_previous_step_max_update=float(optimizer.state[raw]['d'].abs().max()*optimizer.state[raw]['t']),
            initial_raw_range=[float(initial.min()), float(initial.max())])
        write_json(run/'report.json', final)
        finals[name], predictions[name], parameters[name] = final, positions, theta.numpy()

    differences = {}
    for left, right in itertools.combinations(finals, 2):
        row = dict(log_parameter_rmse=float(np.sqrt(np.mean((parameters[left]-parameters[right])**2))))
        for case in cases:
            a, b = tensor(predictions[left][case]), tensor(predictions[right][case])
            denominator = ((observations[case][1:, free_cells]-model.reference.p[free_cells])**2).mean()
            row[case] = float((((a-b)[1:, free_cells]**2).mean()/denominator).sqrt())
        differences[f'{left}__{right}'] = row
    report = dict(complete=True, baseline=baseline, runs=finals, pairwise_differences=differences,
        known_parameters_local_normalized_rmse=sanity_error,
        note='Pairwise trajectory differences use the same target-motion normalization as prediction errors. '
             'Different parameters alone are not evidence of different physical predictions.')
    write_json(args.out/'report.json', report)
    make_plot(args.out, report)
    print('Complete', args.out, flush=True)


def make_plot(out, report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for name in report['runs']:
        records = json.loads((out/name/'checkpoints.json').read_text())
        steps = [r['total_iterations'] for r in records]
        train = [100*r['metrics']['train_combined'] for r in records]
        local = [100*r['metrics']['test_local'] for r in records]
        if name == 'continued':
            steps.insert(0, 40)
            train.insert(0, 100*report['baseline']['train_combined'])
            local.insert(0, 100*report['baseline']['test_local'])
        axes[0].plot(steps, train, 'o-', label=name)
        axes[1].plot(steps, local, 'o-', label=name)
    for ax, title in zip(axes, ('Original two training motions', 'Held-out single-cell force')):
        ax.set_title(title)
        ax.set_xlabel('Total accepted LBFGS iterations')
        ax.set_ylabel('Position RMSE / target motion RMS (%)')
        ax.grid(alpha=.25)
        ax.legend()
    fig.suptitle('Unchanged training data; three serial fits; synthetic DEM targets')
    fig.tight_layout()
    fig.savefig(out/'comparison.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
