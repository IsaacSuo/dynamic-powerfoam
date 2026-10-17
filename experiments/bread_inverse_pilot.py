"""Small, explicitly synthetic inverse-dynamics check on real Power Foam cells.

Targets come from the existing finite DEM solver with hidden stiffness fields.
This checks differentiable fitting and held-out loading within that model; it
does not establish agreement with real materials or an independent continuum.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from scipy.linalg import eigh

from experiments.bonded_dem_differentiable import DifferentiableBonds, STATE_NAMES, STIFFNESS_NAMES
from experiments.bonded_dem_newton import NewtonBonds


def patch(geometry, count=16):
    p, volume, pairs = geometry['points'], geometry['volumes'], geometry['pairs']
    keep = geometry['keep']; median = np.median(volume[keep])
    eligible = keep & (volume > .5*median) & (volume < 5*median)
    graph = sparse.coo_matrix((np.ones(2*len(pairs)),
        (np.r_[pairs[:, 0], pairs[:, 1]], np.r_[pairs[:, 1], pairs[:, 0]])),
        shape=(len(p), len(p))).tocsr()
    center = np.average(p[keep], axis=0, weights=volume[keep])
    seeds = np.flatnonzero(eligible)
    seeds = seeds[np.argsort(np.linalg.norm(p[seeds]-center, axis=1))]
    for seed in seeds:
        nodes = {int(seed)}
        while len(nodes) < count:
            candidates = {int(j) for i in nodes for j in graph[i].indices if eligible[j] and j not in nodes}
            if not candidates:
                break
            nodes.add(min(candidates, key=lambda j: (np.linalg.norm(p[j]-p[seed]), j)))
        if len(nodes) == count:
            break
    else:
        raise RuntimeError('No sufficiently large connected robust-volume patch')
    original_ids = np.array(sorted(nodes))
    selected = np.zeros(len(p), dtype=bool); selected[original_ids] = True
    edge_ids = np.flatnonzero(selected[pairs].all(axis=1))
    mapping = np.full(len(p), -1, dtype=int); mapping[original_ids] = np.arange(count)
    fixed = np.zeros(count, dtype=bool)
    fixed[np.argsort(p[original_ids, 0])[:max(3, count//4)]] = True
    kwargs = dict(points=p[original_ids], radii=geometry['radii'][original_ids],
        pairs=mapping[pairs[edge_ids]], areas=geometry['areas'][edge_ids],
        masses=volume[original_ids], fixed=fixed, contacts=geometry['contacts'][edge_ids], device='cpu')
    return kwargs, original_ids, edge_ids


def loading(model, frames):
    s = model.reference
    free = model.free.numpy()
    size = s.mass.numel()
    columns = []
    for index in free:
        direction = torch.zeros_like(s.mass).flatten(); direction[index] = 1
        columns.append(s.stiffness(direction.reshape_as(s.mass)).flatten()[model.free].numpy())
    stiffness = np.stack(columns, axis=1)
    assert np.max(abs(stiffness-stiffness.T)) < 1e-9
    eig = eigh(stiffness, np.diag(s.mass.flatten().numpy()[free]), eigvals_only=True)
    assert eig[0] > 0
    period = float(2*np.pi/np.sqrt(eig[0]))
    pulse = .5*period
    dt = 3*pulse/frames
    p, mass = s.p.numpy(), s.mass[:, 0].numpy()
    width = float(np.ptp(p[:, 0]))
    right = np.argsort(p[:, 0])[-4:]
    load_patterns = []
    for name, axis, ids, amplitude in [('train_down', 2, right, -.03),
                                       ('train_pull', 0, right, .03),
                                       ('test_side', 1, np.flatnonzero(~s.fixed.numpy()), .02)]:
        force = np.zeros((len(p), 6))
        force[ids, axis] = mass[ids]/mass[ids].sum()
        force[s.fixed.numpy()] = 0
        force[:, axis] /= force[:, axis].sum()
        equilibrium = np.zeros(size)
        equilibrium[free] = np.linalg.solve(stiffness, force.flatten()[free])
        mean = np.sum(force[:, axis]*equilibrium.reshape(-1, 6)[:, axis])
        peak_force = amplitude*width/mean
        times = (np.arange(frames)+.5)*dt
        fraction = np.where(times < pulse, np.sin(np.pi*times/pulse)**2, 0.)
        loads = s.tensor(fraction[:, None, None]*peak_force*force[None])
        load_patterns.append((name, loads, dict(axis=axis, loaded_ids=ids.tolist(), peak_force=float(peak_force))))
    return load_patterns, dt, dict(period=period, pulse_duration=pulse, dt=dt,
                                   duration=frames*dt, frames=frames, reference_width=width,
                                   lowest_mode_frequency=float(np.sqrt(eig[0])))


def original_rollout(kwargs, parameters, loads, dt):
    s = NewtonBonds(**kwargs)
    factor = torch.exp(.5*(parameters[s.i]+parameters[s.j]))
    for col, key in enumerate(STIFFNESS_NAMES):
        setattr(s, key, getattr(s, key)*factor[:, col])
    positions = [s.x.clone()]
    stats = []
    for force in loads:
        _, info = s.step(dt, force, tolerance=2e-10, pose_tolerance=2e-11,
                         linear_tolerance=1e-10, max_iterations=16)
        positions.append(s.x.clone()); stats.append(info)
    return torch.stack(positions), tuple(getattr(s, key).clone() for key in STATE_NAMES), stats


def objective(prediction, observation, initial, free_cells):
    scale2 = ((observation[1:, free_cells]-initial[free_cells])**2).mean().clamp_min(1e-30)
    return ((prediction[1:, free_cells]-observation[1:, free_cells])**2).mean()/scale2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=24)
    parser.add_argument('--iterations', type=int, default=40)
    parser.add_argument('--checks-only', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    source = Path('output/powersim_bread/prune_speed_1e-4/geometry/material_geometry.npz')
    with np.load(source) as a: geometry = dict(a)
    kwargs, original_ids, edge_ids = patch(geometry)
    model = DifferentiableBonds(**kwargs)
    cases, dt, timing = loading(model, args.frames)
    coordinates = kwargs['points']
    normalized = (coordinates-coordinates.min(0))/np.ptp(coordinates, axis=0)
    target_parameters = model.reference.tensor(np.column_stack((
        .45+.35*np.sin(np.pi*normalized[:, 0]),
        -.4+.25*np.cos(np.pi*normalized[:, 1]),
        .2+.3*np.sin(np.pi*normalized[:, 2]),
        -.25+.35*np.sin(np.pi*(normalized[:, 0]+normalized[:, 2])))))
    config = dict(cells=len(original_ids), bonds=len(edge_ids), free_cells=int((~kwargs['fixed']).sum()),
        learnable_parameters=int(target_parameters.numel()), parameter_order=list(STIFFNESS_NAMES),
        parameterization='Per-cell log stiffness multipliers; each bond uses exp((theta_i+theta_j)/2). '
                         'Training theta = 1.5*tanh(raw), initialized at zero.',
        timing=timing, original_ids=original_ids.tolist(), geometry=str(source),
        geometry_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                      for name in ('bread_inverse_pilot.py', 'bonded_dem_differentiable.py',
                                   'bonded_dem_finite_gpu.py', 'bonded_dem_newton.py', 'bonded_dem_linear_gpu.py')},
        targets='Synthetic motion from the existing NewtonBonds solver, hidden heterogeneous cell stiffnesses.',
        observation='All free cell center positions at 24 consecutive time steps; no RGB or rotation labels.',
        patch_boundary='Induced connected cutout of actual bread cells; retain source masses/faces, omit outside bonds, '
                       'clamp the four leftmost generators. This is a test specimen, not the full bread boundary problem.',
        training_cases=[c[0] for c in cases[:2]], test_case=cases[2][0],
        scope='Implementation and within-model inverse-dynamics experiment only; no real-material accuracy claim. '
              'Known external loads, fixed masses, fixed cell geometry and topology. No damping, plasticity or damage.',
        optimization=dict(method='LBFGS', max_iterations=args.iterations, max_evaluations=2*args.iterations,
                          log_parameter_l2=1e-6, held_out_used_for_training=False))
    (args.out/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    np.savez_compressed(args.out/'patch.npz', **{k:v for k,v in kwargs.items() if k!='device'},
                        original_ids=original_ids, original_edge_ids=edge_ids)
    print(json.dumps(config), flush=True)
    observations, reference_states, targets = {}, {}, {}
    for name, loads, info in cases:
        start = time.perf_counter()
        position, state, stats = original_rollout(kwargs, target_parameters, loads, dt)
        observations[name], reference_states[name] = position, state
        targets[name] = dict(**info, generation_seconds=time.perf_counter()-start,
                             max_displacement_width_fraction=float(torch.linalg.vector_norm(position-model.reference.p,dim=-1).max()/timing['reference_width']),
                             max_local_defect=max(r['local_pose_defect'] for r in stats))
        np.savez_compressed(args.out/f'{name}_target.npz', positions=position.numpy(), loads=loads.numpy(),
                            dt=dt, true_parameters=target_parameters.numpy())
        print('Target', name, targets[name], flush=True)

    free_cells = ~model.reference.fixed
    with torch.no_grad():
        predicted, state = model.rollout(target_parameters, cases[0][1], dt)
    position_error = float(abs(predicted-observations[cases[0][0]]).max()/timing['reference_width'])
    state_errors = {name:float((a-b).norm()/b.norm().clamp_min(1e-20))
                    for name,a,b in zip(STATE_NAMES,state,reference_states[cases[0][0]])}
    assert position_error < 1e-7, position_error
    assert max(state_errors.values()) < 1e-5, state_errors
    theta = torch.zeros_like(target_parameters, requires_grad=True)
    prediction, _ = model.rollout(theta, cases[0][1], dt)
    loss = objective(prediction, observations[cases[0][0]], model.reference.p, free_cells)
    gradient, = torch.autograd.grad(loss, theta)
    rng = np.random.default_rng(8402)
    direction = model.reference.tensor(rng.normal(size=theta.shape)); direction /= direction.norm()
    analytic = float((gradient*direction).sum())
    finite_differences = []
    for epsilon in (1e-3, 1e-4, 1e-5):
        values = []
        with torch.no_grad():
            for sign in (-1, 1):
                prediction, _ = model.rollout(sign*epsilon*direction, cases[0][1], dt)
                values.append(float(objective(prediction, observations[cases[0][0]], model.reference.p, free_cells)))
        numeric = (values[1]-values[0])/(2*epsilon)
        finite_differences.append(dict(epsilon=epsilon, numerical=numeric, analytical=analytic,
            relative_error=abs(numeric-analytic)/max(abs(numeric),abs(analytic),1e-12)))
    checks = dict(forward_matches_existing_solver_max_position_over_width=position_error,
        final_state_relative_errors=state_errors, full_trajectory_parameter_gradient=finite_differences,
        gradient_finite=bool(torch.isfinite(gradient).all()), gradient_norm=float(gradient.norm()),
        history_backpropagation='x, velocity, rotations, normals, offsets, and all four bond histories retained across steps')
    assert checks['gradient_finite'] and min(d['relative_error'] for d in finite_differences) < 1e-4, checks
    (args.out/'checks.json').write_text(json.dumps(checks, indent=2)+'\n')
    print('Checks', json.dumps(checks), flush=True)
    if args.checks_only:
        return

    raw = torch.zeros_like(target_parameters, requires_grad=True)
    history = []
    optimizer = torch.optim.LBFGS([raw], lr=1., max_iter=args.iterations, max_eval=2*args.iterations,
                                 history_size=10, line_search_fn='strong_wolfe',
                                 tolerance_grad=1e-8, tolerance_change=1e-12)
    def evaluate(parameters):
        metrics, positions = {}, {}
        with torch.no_grad():
            for name, loads, _ in cases:
                positions[name], _ = model.rollout(parameters, loads, dt)
                metrics[name] = float(torch.sqrt(objective(positions[name], observations[name],model.reference.p,free_cells)))
        return metrics, positions
    initial_metrics, initial_positions = evaluate(torch.zeros_like(raw))
    start = time.perf_counter()
    def closure():
        optimizer.zero_grad()
        parameters = 1.5*torch.tanh(raw)
        errors = []
        for name, loads, _ in cases[:2]:
            prediction, _ = model.rollout(parameters, loads, dt)
            errors.append(objective(prediction, observations[name], model.reference.p, free_cells))
        motion_loss = torch.stack(errors).mean()
        total = motion_loss+1e-6*parameters.square().mean()
        total.backward()
        assert raw.grad is not None and torch.isfinite(raw.grad).all()
        row = dict(evaluation=len(history), elapsed_seconds=time.perf_counter()-start,
                   motion_loss=float(motion_loss.detach()), normalized_rmse=float(motion_loss.detach().sqrt()),
                   gradient_norm=float(raw.grad.norm()), total_loss=float(total.detach()))
        history.append(row)
        with (args.out/'training.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
        print('Training', json.dumps(row), flush=True)
        return total
    optimizer.step(closure)
    train_seconds = time.perf_counter()-start
    learned = (1.5*torch.tanh(raw)).detach()
    final_metrics, final_positions = evaluate(learned)
    np.savez_compressed(args.out/'learned_parameters.npz', log_multipliers=learned.numpy(),
                        multipliers=learned.exp().numpy(), original_ids=original_ids,
                        initial_log_multipliers=np.zeros_like(learned.numpy()))
    for name, loads, _ in cases:
        np.savez_compressed(args.out/f'{name}_comparison.npz', target=observations[name].numpy(),
            before=initial_positions[name].numpy(), after=final_positions[name].numpy(), dt=dt)
    report = dict(config=config, targets=targets, checks=checks, initial_normalized_rmse=initial_metrics,
        final_normalized_rmse=final_metrics,
        improvement_factor={name:initial_metrics[name]/max(final_metrics[name],1e-30) for name in final_metrics},
        training_seconds=train_seconds, objective_evaluations=len(history),
        accepted_lbfgs_iterations=int(optimizer.state[raw]['n_iter']),
        learned_multiplier_range=[float(learned.exp().min()),float(learned.exp().max())],
        parameter_recovery_note='No unique per-cell parameter recovery is claimed; optimize motion, not parameter labels.',
        complete=True)
    (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print('Result', json.dumps({k:v for k,v in report.items() if k not in ('config','targets','checks')}), flush=True)


if __name__ == '__main__':
    main()
