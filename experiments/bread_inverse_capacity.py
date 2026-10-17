"""Shared versus per-cell stiffness on the existing synthetic inverse pilot.

Reuse the completed per-cell fit, train the shared fit with the same budget,
then evaluate predeclared held-out loads. No test data enter optimization.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.bonded_dem_differentiable import DifferentiableBonds
from experiments.bread_inverse_pilot import objective, original_rollout


def extra_loads(model, timing, down_loads):
    s = model.reference
    p = s.p.numpy()
    fixed = s.fixed.numpy()
    right = np.argsort(p[:, 0])[-4:]
    free = model.free.numpy()
    columns = []
    for index in free:
        direction = torch.zeros_like(s.mass).flatten()
        direction[index] = 1
        columns.append(s.stiffness(direction.reshape_as(s.mass)).flatten()[model.free].numpy())
    stiffness = np.stack(columns, axis=1)
    times = (np.arange(timing['frames'])+.5)*timing['dt']
    pulse = timing['pulse_duration']
    fraction = np.where(times < pulse, np.sin(np.pi*times/pulse)**2, 0.)
    patterns = {}
    twist = np.zeros_like(s.mass.numpy())
    arm = p[right]-p[right].mean(axis=0)
    twist[right, 1] = -arm[:, 2]
    twist[right, 2] = arm[:, 1]
    assert not fixed[right].any()
    assert np.linalg.norm(twist[:, :3].sum(axis=0)) < 1e-14
    assert np.linalg.norm(twist) > 0
    patterns['test_twist'] = (twist, dict(loaded_ids=right.tolist(),
        description='Force couple around x on the four rightmost cells; zero resultant force.'))
    local = np.zeros_like(twist)
    local_id = int(right[np.argmax(p[right, 2])])
    local[local_id, 2] = -1
    patterns['test_local'] = (local, dict(loaded_ids=[local_id],
        description='Downward force on the highest of the four rightmost cells.'))
    result = {}
    for name, (force, info) in patterns.items():
        equilibrium = np.zeros(s.mass.numel())
        equilibrium[free] = np.linalg.solve(stiffness, force.flatten()[free])
        max_motion = np.linalg.norm(equilibrium.reshape(-1, 6)[:, :3], axis=1).max()
        scale = .03*timing['reference_width']/max_motion
        loads = fraction[:, None, None]*scale*force[None]
        result[name] = (s.tensor(loads), dict(**info,
            reference_static_max_displacement_width_fraction=.03, force_scale=float(scale)))
    result['test_strong_down'] = (2*down_loads, dict(
        description='Exactly twice the training down force; identical timing and loaded cells.',
        force_multiplier=2.))
    return result


def plot_result(out, metrics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = list(metrics)
    labels = ['Down\n(train)', 'Pull\n(train)', 'Side\n(test)', 'Twist\n(test)',
              'Local\n(test)', '2x down\n(test)']
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(names))
    for shift, key, title in [(-.19, 'shared', '4 shared parameters'),
                               (.19, 'per_cell', '64 per-cell parameters')]:
        values = [100*metrics[n][key]['normalized_rmse'] for n in names]
        bars = ax.bar(x+shift, values, width=.36, label=title)
        ax.bar_label(bars, labels=[f'{v:.2f}%' for v in values], padding=3, fontsize=9)
    ax.set_xticks(x, labels)
    ax.set_ylabel('Position RMSE / target displacement RMS (%)')
    ax.set_title('Same DEM synthetic targets; only down and pull used for training')
    ax.legend()
    ax.set_ylim(0, ax.get_ylim()[1]*1.23)
    fig.tight_layout()
    fig.savefig(out/'comparison.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', type=Path, default=Path('output/bonded_dem_inverse/pilot_16cells'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=False)
    pilot_config = json.loads((args.pilot/'config.json').read_text())
    pilot_report = json.loads((args.pilot/'report.json').read_text())
    assert pilot_report['complete'] and pilot_report['accepted_lbfgs_iterations'] == 40
    for name, digest in pilot_config['source_sha256'].items():
        assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() == digest, name
    with np.load(args.pilot/'patch.npz') as a:
        kwargs = {key: a[key] for key in ('points', 'radii', 'pairs', 'areas', 'masses', 'fixed', 'contacts')}
        original_ids = a['original_ids']
    kwargs['device'] = 'cpu'
    model = DifferentiableBonds(**kwargs)
    tensor = model.reference.tensor
    free_cells = ~model.reference.fixed
    timing = pilot_config['timing']
    dt = timing['dt']
    cases, observations = {}, {}
    for name in ('train_down', 'train_pull', 'test_side'):
        with np.load(args.pilot/f'{name}_target.npz') as a:
            assert float(a['dt']) == dt
            cases[name] = tensor(a['loads'])
            observations[name] = tensor(a['positions'])
            true_parameters = tensor(a['true_parameters'])
    extras = extra_loads(model, timing, cases['train_down'])
    cases.update({name: values[0] for name, values in extras.items()})
    config = dict(pilot=str(args.pilot), timing=timing, training_cases=['train_down', 'train_pull'],
        test_cases=[n for n in cases if n.startswith('test_')],
        new_load_definitions={n: values[1] for n, values in extras.items()},
        parameters=dict(shared=4, per_cell=64),
        optimizer=dict(method='LBFGS', max_iterations=40, max_evaluations=80,
            history_size=10, initialization='zero raw', theta='1.5*tanh(raw)', log_parameter_l2=1e-6),
        per_cell_fit='Reuse frozen pilot fit; do not retrain or select using test cases.',
        scope='Same-model heterogeneous synthetic targets, all free center positions observed. '
              'Not independent FEM, real material, image supervision, or unique parameter recovery.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        pilot_parameters_sha256=hashlib.sha256((args.pilot/'learned_parameters.npz').read_bytes()).hexdigest())
    (args.out/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    for name, (loads, info) in extras.items():
        np.savez_compressed(args.out/f'{name}_load.npz', loads=loads.numpy(), dt=dt)

    raw = torch.zeros(4, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([raw], lr=1., max_iter=40, max_eval=80, history_size=10,
        line_search_fn='strong_wolfe', tolerance_grad=1e-8, tolerance_change=1e-12)
    history = []
    start = time.perf_counter()

    def closure():
        optimizer.zero_grad()
        theta = (1.5*raw.tanh()).expand(len(original_ids), -1)
        errors = []
        for name in config['training_cases']:
            predicted, _ = model.rollout(theta, cases[name], dt)
            errors.append(objective(predicted, observations[name], model.reference.p, free_cells))
        motion_loss = torch.stack(errors).mean()
        loss = motion_loss+1e-6*theta.square().mean()
        loss.backward()
        assert torch.isfinite(raw.grad).all()
        row = dict(evaluation=len(history), elapsed_seconds=time.perf_counter()-start,
            motion_loss=float(motion_loss.detach()), normalized_rmse=float(motion_loss.detach().sqrt()),
            gradient_norm=float(raw.grad.norm()))
        history.append(row)
        with (args.out/'shared_training.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
        print('Shared training', json.dumps(row), flush=True)
        return loss

    optimizer.step(closure)
    training_seconds = time.perf_counter()-start
    shared = (1.5*raw.tanh()).detach().expand(len(original_ids), -1).clone()
    with np.load(args.pilot/'learned_parameters.npz') as a:
        per_cell = tensor(a['log_multipliers'])
    np.savez_compressed(args.out/'parameters.npz', shared=shared.numpy(), per_cell=per_cell.numpy(),
                        original_ids=original_ids)
    for name, (loads, info) in extras.items():
        positions, _, stats = original_rollout(kwargs, true_parameters, loads, dt)
        observations[name] = positions
        np.savez_compressed(args.out/f'{name}_target.npz', positions=positions.numpy(),
                            loads=loads.numpy(), dt=dt)
        print('Generated target', name, flush=True)

    metrics = {}
    with torch.no_grad():
        for name, loads in cases.items():
            target = observations[name]
            positions = {}
            metrics[name] = dict(target_max_displacement_width_fraction=float(
                torch.linalg.vector_norm(target-model.reference.p, dim=-1).max()/timing['reference_width']))
            for kind, theta in [('shared', shared), ('per_cell', per_cell)]:
                predicted, _ = model.rollout(theta, loads, dt)
                positions[kind] = predicted.numpy()
                metrics[name][kind] = dict(normalized_rmse=float(objective(
                    predicted, target, model.reference.p, free_cells).sqrt()),
                    max_center_error_width_fraction=float(torch.linalg.vector_norm(
                        predicted-target, dim=-1).max()/timing['reference_width']))
                assert torch.isfinite(predicted).all()
                assert torch.equal(predicted[:, ~free_cells], target[:, ~free_cells])
                if kind == 'per_cell' and name in pilot_report['final_normalized_rmse']:
                    assert abs(metrics[name][kind]['normalized_rmse']-
                               pilot_report['final_normalized_rmse'][name]) < 1e-10
            np.savez_compressed(args.out/f'{name}_comparison.npz', target=target.numpy(), **positions, dt=dt)
            print('Evaluation', name, json.dumps(metrics[name]), flush=True)
    report = dict(complete=True, config=config, metrics=metrics,
        shared_training=dict(seconds=training_seconds, accepted_iterations=int(optimizer.state[raw]['n_iter']),
            objective_evaluations=len(history), multipliers=shared[0].exp().tolist()),
        per_cell_training=dict(seconds=pilot_report['training_seconds'],
            accepted_iterations=pilot_report['accepted_lbfgs_iterations'],
            objective_evaluations=pilot_report['objective_evaluations']),
        checks='Unchanged pilot solver hashes; reproduced frozen pilot metrics; finite trajectories and exact fixed cells.')
    (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    plot_result(args.out, metrics)
    print('Complete', args.out, flush=True)


if __name__ == '__main__':
    main()
