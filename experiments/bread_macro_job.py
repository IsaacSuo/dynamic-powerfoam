"""Serial force calibration and load-release tests targeting macro displacement.

The target changes only the peak applied force between trials. It never drives
positions, edits material parameters or scales the rendered displacement.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--targets', type=float, nargs='+', default=[.10, .20])
    parser.add_argument('--relative-tolerance', type=float, default=.05)
    parser.add_argument('--max-trials', type=int, default=4)
    args = parser.parse_args()
    if any(t <= 0 or not np.isfinite(t) for t in args.targets) or not 0 < args.relative_tolerance < 1 or args.max_trials < 1:
        parser.error('positive targets/trial count and relative tolerance in (0,1) required')
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out/'job.json').exists(): parser.error('choose a fresh output directory')
    baseline = Path('output/powersim_bread/finite_dem/newton_fast_after_resume')
    with np.load(baseline/'trajectory.npz') as a: baseline_samples = a['samples']
    with np.load('output/powersim_bread/material_geometry.npz') as a:
        width = float(np.ptp(a['points'][a['keep'], 0]))
    baseline_peak = -float(baseline_samples[:, 1].min())/width
    baseline_force = json.loads((baseline/'mechanics.json').read_text())['total_peak_force']
    status = dict(pid=os.getpid(), state='running', stage='calibration', started_at=time.time(),
                  targets=args.targets, relative_tolerance=args.relative_tolerance,
                  reference_width=width, baseline_peak_fraction=baseline_peak,
                  baseline_peak_force=baseline_force, accepted=[], trials=[],
                  metric='Max over 3T of downward reference-mass-weighted mean z displacement of original loaded cells / initial x width',
                  control='Peak force only; same sin-squared pulse over T, followed by 2T free motion',
                  source_sha256={n: hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
                    for n in ('bread_macro_job.py', 'bread_macro_run.py', 'bread_macro_report.py')})
    def write():
        (args.out/'job.tmp.json').write_text(json.dumps(status, indent=2)+'\n')
        (args.out/'job.tmp.json').replace(args.out/'job.json')
    def command(stage, module, options, logfile):
        status.update(stage=stage, stage_started_at=time.time()); write()
        print(stage, options, flush=True)
        with logfile.open('x') as log:
            child = subprocess.Popen([sys.executable, '-u', '-m', module, *options], stdout=log, stderr=subprocess.STDOUT)
            status['child_pid'] = child.pid; write()
            code = child.wait()
        if code: raise RuntimeError(f'{stage} failed ({code}); see {logfile}')
        status.pop('child_pid', None); write()
    def run(target, multiplier, intervals, out, resume=None):
        options = ['--solver', 'fast', '--case', 'initial', '--intervals', str(intervals),
                   '--load-multiplier', str(multiplier), '--target-fraction', str(target),
                   '--sample-stride', '4', '--out', str(out)]
        if resume: options += ['--resume', str(resume)]
        command('calibration' if intervals < 192 else 'full_simulation', 'experiments.bread_macro_run',
                options, out.with_suffix('.log'))
        report = json.loads((out/'mechanics.json').read_text())
        if not report['complete'] or not report['checks_pass']:
            raise RuntimeError(f'Incomplete or failed mechanics checks: {out}')
        return report
    write()
    try:
        last_multiplier, last_peak = 1., baseline_peak
        for target in args.targets:
            tag = f'target_{100*target:g}pct'
            directory = args.out/tag; directory.mkdir()
            multiplier = last_multiplier*target/last_peak
            observations = []
            calibration_intervals = 96  # First downward peak is near 1.2T in the baseline.
            accepted = None
            for trial in range(1, args.max_trials+1):
                status.update(target=target, trial=trial, load_multiplier=multiplier); write()
                out = directory/f'trial_{trial:02d}_pilot'
                report = run(target, multiplier, calibration_intervals, out)
                # Extend if the maximum is too close to the end of the pilot.
                for end in (128, 192):
                    if report['completed_intervals'] >= end or report['peak_interval'] <= report['completed_intervals']-8:
                        continue
                    extension = directory/f'trial_{trial:02d}_extended_{end}'
                    report = run(target, multiplier, end, extension, out); out = extension
                if report['completed_intervals'] == 192 and calibration_intervals < 192:
                    calibration_intervals = 192; observations = []
                peak = report['peak_downward_fraction']
                hit = abs(peak/target-1) <= args.relative_tolerance
                if hit and report['completed_intervals'] < 192:
                    full = directory/f'trial_{trial:02d}_full'
                    report = run(target, multiplier, 192, full, out); out = full
                    peak = report['peak_downward_fraction']
                    hit = abs(peak/target-1) <= args.relative_tolerance
                    if not hit:
                        # Later oscillations changed the global peak. Subsequent
                        # force trials must use the full time window as well.
                        calibration_intervals = 192; observations = []
                row = dict(target=target, trial=trial, load_multiplier=multiplier,
                           total_peak_force=report['total_peak_force'], achieved_fraction=peak,
                           intervals=report['completed_intervals'], path=str(out),
                           relative_error=peak/target-1, target_met=hit,
                           complete_3T=report['completed_intervals'] == 192)
                status['trials'].append(row); write()
                print(json.dumps(row), flush=True)
                if hit and row['complete_3T']:
                    accepted = row; break
                if peak <= 0 or not np.isfinite(peak): raise RuntimeError('No valid downward response for force calibration')
                observations.append((multiplier, peak))
                low = [m for m, a in observations if a < target]
                high = [m for m, a in observations if a > target]
                if low and high and max(low) < min(high):
                    multiplier = float(np.sqrt(max(low)*min(high)))
                else:
                    multiplier *= float(np.clip(target/peak, .6, 1.6))
            if accepted is None:
                raise RuntimeError(f'Target {target:g} not reached within {args.max_trials} trials; results preserved')
            status['accepted'].append(accepted); write()
            out = Path(accepted['path'])
            command('render', 'experiments.bread_finite_sim', ['--out', str(out), '--render-only'],
                    directory/'render.log')
            command('report', 'experiments.bread_macro_report', ['--root', str(args.out)],
                    directory/'report.log')
            last_multiplier, last_peak = accepted['load_multiplier'], accepted['achieved_fraction']
        status.update(state='complete', stage='complete', ended_at=time.time()); write()
    except Exception as error:
        status.update(state='failed', reason=str(error), ended_at=time.time()); write()
        raise


if __name__ == '__main__':
    main()
