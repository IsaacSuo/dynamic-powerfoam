"""Matched physical intervals or a full trajectory with either finite solver.

Same material, initial state, load, tolerances and adaptive controller. Timings
include failed attempts and accepted substeps, but exclude setup/checkpoint I/O.
Each run owns a new output directory; completed old experiments are untouched.
"""
import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from experiments.bonded_dem_adaptive import AdaptiveIntervals
from experiments.bonded_dem_finite_gpu import FiniteBondsGPU
from experiments.bonded_dem_newton import NewtonBonds
from experiments.bonded_dem_tangent_warp import WarpTangent
from experiments.bread_linear_sim import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--solver', choices=('newton', 'previous'), default='newton')
    parser.add_argument('--case', choices=('initial', 'early', 'release'), default='initial')
    parser.add_argument('--intervals', type=int, default=192)
    parser.add_argument('--dt-factor', type=float, default=1.)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--checkpoint-every', type=int, default=8)
    parser.add_argument('--resume', type=Path, help='Saved prefix; write the continuation into a new output directory')
    args = parser.parse_args()
    if args.intervals < 1 or args.dt_factor <= 0 or args.checkpoint_every < 1:
        parser.error('positive intervals, dt-factor and checkpoint-every required')
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out/'config.json').exists():
        parser.error('output already contains a run; select a new directory')
    torch.set_num_threads(1)
    names = ('bread_solver_run.py', 'bonded_dem_finite_gpu.py', 'bonded_dem_newton.py',
             'bonded_dem_linear_gpu.py', 'bonded_dem_tangent_warp.py', 'bonded_dem_adaptive.py',
             'bread_linear_sim.py')
    hashes = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in names}
    previous = None
    if args.resume:
        previous = json.loads((args.resume/'mechanics.json').read_text())
        assert previous['solver'] == args.solver and previous['case'] == args.case == 'initial'
        assert previous['checks_pass'] and previous['failure'] is None
        assert previous['completed_intervals'] < args.intervals
        for name, digest in previous['source_sha256'].items():
            if name != 'bread_solver_run.py':
                assert hashes.get(name) == digest, f'Resume requires unchanged mechanics: {name}'
    with np.load('output/powersim_bread/material_geometry.npz') as a: geo = dict(a)
    baseline = json.loads(Path('output/powersim_bread/linear_dem_10x/mechanics.json').read_text())
    linear, force, period, setup = prepare(geo)
    pairs, fixed = linear.edges.cpu().numpy(), linear.fixed.cpu().numpy()
    keep = geo['keep']; p = geo['points'][keep]
    cls = NewtonBonds if args.solver == 'newton' else FiniteBondsGPU
    s = cls(p, geo['radii'][keep], pairs, geo['areas'], geo['volumes'][keep], fixed,
            contacts=geo['contacts'])
    assert torch.equal(s.mass, linear.mass)
    for name in ('kn', 'ks', 'kb', 'kt', 'n', 'ri', 'rj'):
        assert torch.allclose(getattr(s, name), getattr(linear, name), rtol=1e-12, atol=1e-14)
    del linear; gc.collect(); torch.cuda.empty_cache()
    source = None
    if args.case != 'initial':
        tag = 'diagnostic' if args.case == 'early' else 'adaptive_coarse'
        root = Path('output/powersim_bread/finite_dem')/tag
        prior = json.loads((root/'mechanics.json').read_text())
        source = root/'final_state.npz'
        with np.load(source) as a:
            for key in ('x', 'rotation', 'fn', 'fs', 'mt', 'mb', 'normal', 'offset'):
                setattr(s, key, s.tensor(a[key]))
            s.v = s.tensor(a['velocity'])
        s.time = prior['steps']*prior['dt']
    if args.resume:
        with np.load(args.resume/'final_state.npz') as a:
            for key in ('x', 'rotation', 'fn', 'fs', 'mt', 'mb', 'normal', 'offset'):
                setattr(s, key, s.tensor(a[key]))
            s.v = s.tensor(a['velocity']); s.time = float(a['time'])
        assert abs(s.time-previous['actual_time']) < 1e-14
    assert torch.equal(s.x[s.fixed], s.p[s.fixed]) and not bool(s.v[s.fixed].any())
    s.stiffness = WarpTangent(s)
    s.stiffness(torch.zeros_like(s.mass))
    applied = s.tensor(force*baseline['total_peak_force'])
    def load(t):
        return applied*(np.sin(np.pi*t/period)**2 if t < period else 0.)
    dt = 3*period/192*args.dt_factor
    if previous:
        assert np.isclose(dt, previous['dt'], rtol=1e-13, atol=0)
        assert baseline['total_peak_force'] == previous['total_peak_force']
        dt = previous['dt']
    config = dict(solver=args.solver, case=args.case, intervals=args.intervals, dt=dt,
                  start_time=previous['start_time'] if previous else s.time,
                  end_time=(previous['start_time'] if previous else s.time)+args.intervals*dt, setup=setup,
                  total_peak_force=baseline['total_peak_force'], source_sha256=hashes,
                  source_state=str(source) if source else None,
                  source_state_sha256=hashlib.sha256(source.read_bytes()).hexdigest() if source else None,
                  scope='Same finite bond model; only solver differs. Adaptivity controls nonlinear convergence, not temporal accuracy.')
    if previous:
        config.update(resumed_from=str(args.resume), resume_source_sha256=previous['source_sha256'],
                      resume_state_sha256=hashlib.sha256((args.resume/'final_state.npz').read_bytes()).hexdigest())
    (args.out/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    print(json.dumps(config), flush=True)
    controller = AdaptiveIntervals(s)
    attempts, durations = [], []
    original_step = s.step
    def tracked_step(dt, external):
        torch.cuda.synchronize(); tick = time.perf_counter()
        try:
            work, info = original_step(dt, external)
            attempts.append(dict(dt=dt, success=True, **info))
            return work, info
        except RuntimeError as error:
            attempts.append(dict(dt=dt, success=False, reason=str(error)))
            raise
        finally:
            torch.cuda.synchronize(); durations.append(time.perf_counter()-tick)
    s.step = tracked_step
    initial_energy = s.energy() if not previous else 0.
    work, worst_balance, peak_energy, elapsed = 0., 0., initial_energy, 0.
    qs, samples, records = [], [], []
    first_interval = 0
    if previous:
        first_interval = previous['completed_intervals']
        records = [json.loads(line) for line in (args.resume/'progress.jsonl').read_text().splitlines()]
        records = [r for r in records if r['step'] <= first_interval]
        assert len(records) == first_interval and records[-1]['step'] == first_interval
        for line in (args.resume/'attempts.jsonl').read_text().splitlines():
            row = json.loads(line)
            if row['output_interval'] <= first_interval:
                durations.append(row['elapsed_seconds'])
                attempts.append(row)
        assert len(attempts) == previous['attempts']
        assert sum(a['success'] for a in attempts) == previous['actual_substeps']
        # Reconstruct controller memory from accepted attempts, not aggregate counts.
        by_interval = {}
        for a in attempts:
            if a['success']: by_interval.setdefault(a['output_interval'], []).append(a)
        for row in records:
            controller.depth = row['max_subdivision_depth']
            easy = max(a['nonlinear_iterations'] for a in by_interval[row['step']]) <= 3
            controller.easy_intervals = controller.easy_intervals+1 if easy else 0
            if controller.easy_intervals >= 8 and controller.depth:
                controller.depth -= 1; controller.easy_intervals = 0
        with np.load(args.resume/'trajectory.npz') as a:
            assert np.array_equal(a['original_ids'], np.flatnonzero(keep))
            qs, samples = list(a['q']), list(a['samples'])
        assert abs(samples[-1][0]-s.time) < 1e-12, 'Resume needs a saved trajectory sample at the checkpoint'
        work = records[-1]['work']
        elapsed = previous['elapsed_seconds']
        peak_energy = max([initial_energy, *(r['energy'] for r in records)])
        worst_balance = max(abs(r['energy']-initial_energy-r['work']) for r in records)
        assert abs(s.energy()-records[-1]['energy']) < 1e-12
        (args.out/'progress.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
        (args.out/'attempts.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in attempts))
        print(f'Resuming after interval {first_interval}; adaptive depth={controller.depth}', flush=True)
    stride = 4 if args.case == 'initial' and args.intervals >= 192 else 1
    def sample():
        q = np.c_[s.x.cpu().numpy()-p, Rotation.from_matrix(s.rotation.cpu().numpy()).as_rotvec()]
        q[fixed] = 0
        qs.append(q)
        samples.append([s.time, float(-(force*q).sum()), s.energy(), work])
    if not previous: sample()
    def save(completed, failure=None):
        accepted = [a for a in attempts if a['success']]
        balance = worst_balance/max(peak_energy, abs(work), 1e-30)
        orthogonality = float(abs(s.rotation.transpose(1, 2)@s.rotation-torch.eye(3, device=s.device)).max())
        checks = balance < .002 and orthogonality < 1e-10
        report = dict(**config, complete=completed == args.intervals and failure is None,
            completed_intervals=completed, actual_time=s.time, elapsed_seconds=elapsed,
            steps=completed, actual_substeps=len(accepted), attempts=len(attempts),
            failed_attempts=len(attempts)-len(accepted), solve_seconds=sum(durations),
            max_subdivision_depth=max((r['max_subdivision_depth'] for r in records), default=0),
            max_local_pose_defect=max((a['local_pose_defect'] for a in accepted), default=0),
            max_nonlinear_relative_residual=max((a['nonlinear_relative_residual'] for a in accepted), default=0),
            work_energy_error_over_peak=balance, orientation_orthogonality_max_error=orthogonality,
            fixed_positions_unchanged=bool(torch.equal(s.x[s.fixed], s.p[s.fixed])),
            fixed_velocities_zero=not bool(s.v[s.fixed].any()), checks_pass=checks, failure=failure)
        for name, digest in hashes.items():
            assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() == digest, name
        np.savez_compressed(args.out/'final_state.tmp.npz',
            **{key: getattr(s, key).cpu().numpy() for key in ('x', 'rotation', 'fn', 'fs', 'mt', 'mb', 'normal', 'offset')},
            velocity=s.v.cpu().numpy(), time=s.time)
        (args.out/'final_state.tmp.npz').replace(args.out/'final_state.npz')
        np.savez_compressed(args.out/'trajectory.tmp.npz', q=np.asarray(qs), samples=np.asarray(samples),
                            period=period, original_ids=np.flatnonzero(keep))
        (args.out/'trajectory.tmp.npz').replace(args.out/'trajectory.npz')
        (args.out/'mechanics.tmp.json').write_text(json.dumps(report, indent=2)+'\n')
        (args.out/'mechanics.tmp.json').replace(args.out/'mechanics.json')
        return report
    completed = first_interval
    try:
        for step in range(first_interval, args.intervals):
            before = len(attempts)
            torch.cuda.synchronize(); tick = time.perf_counter()
            delta_work, info = controller.advance(dt, load)
            torch.cuda.synchronize(); elapsed += time.perf_counter()-tick
            work += delta_work; completed = step+1
            energy = s.energy(); peak_energy = max(peak_energy, energy)
            worst_balance = max(worst_balance, abs(energy-initial_energy-work))
            assert torch.equal(s.x[s.fixed], s.p[s.fixed]) and not bool(s.v[s.fixed].any())
            row = dict(step=completed, time=s.time, elapsed_seconds=elapsed, energy=energy, work=work,
                substeps=len(info), failed_attempts=sum(not a['success'] for a in attempts[before:]),
                max_subdivision_depth=max(a['subdivision_depth'] for a in info),
                nonlinear_iterations=sum(a['nonlinear_iterations'] for a in info),
                cg_iterations=sum(a['cg_iterations'] for a in info),
                krylov_iterations=sum(a.get('krylov_iterations', 0) for a in info),
                max_local_pose_defect=max(a['local_pose_defect'] for a in info))
            records.append(row)
            with (args.out/'progress.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
            with (args.out/'attempts.jsonl').open('a') as f:
                for a, seconds in zip(attempts[before:], durations[before:]):
                    f.write(json.dumps(dict(output_interval=completed, elapsed_seconds=seconds, **a))+'\n')
            print(json.dumps(row), flush=True)
            if completed % stride == 0 or completed == args.intervals: sample()
            if completed % args.checkpoint_every == 0 or completed == args.intervals:
                report = save(completed)
                assert report['checks_pass'], report
    except Exception as error:
        save(completed, repr(error))
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
