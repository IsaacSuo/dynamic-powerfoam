"""Queue a full accelerated run after the baseline, then compare and render."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=Path('output/powersim_bread/finite_dem/newton_full'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out/'job.json').exists() or (args.out/'config.json').exists():
        parser.error('choose a new output directory')
    status = dict(pid=os.getpid(), started_at=time.time(), state='queued', stage='waiting_for_baseline',
                  baseline=str(args.baseline))
    def write():
        (args.out/'job.tmp.json').write_text(json.dumps(status, indent=2)+'\n')
        (args.out/'job.tmp.json').replace(args.out/'job.json')
    def fail(message):
        status.update(state='failed', reason=message, ended_at=time.time()); write()
        raise RuntimeError(message)
    write()
    print('Queued after baseline:', args.baseline, flush=True)
    while True:
        base = json.loads((args.baseline/'job.json').read_text())
        if base['state'] == 'complete':
            break
        if base['state'] == 'failed':
            fail('Baseline job failed; full comparison cannot proceed')
        try:
            os.kill(base['pid'], 0)
        except ProcessLookupError:
            fail('Baseline process exited without completing its report')
        time.sleep(30)
    report = json.loads((args.baseline/'mechanics.json').read_text())
    if not report['complete'] or not report['checks_pass']:
        fail('Baseline mechanics did not complete and pass checks')
    commands = [
        ('simulation', ['experiments.bread_fast_solver_run', '--solver', 'fast', '--case', 'initial',
                        '--intervals', '192', '--out', str(args.out)]),
        ('comparison', ['experiments.bread_newton_trajectory_report', '--new', str(args.out),
                        '--previous', str(args.baseline)]),
        ('render', ['experiments.bread_finite_sim', '--out', str(args.out), '--render-only'])]
    for stage, command in commands:
        status.update(state='running', stage=stage, stage_started_at=time.time()); write()
        print('Starting:', stage, flush=True)
        child = subprocess.Popen([sys.executable, '-m', *command])
        status['child_pid'] = child.pid; write()
        result = child.wait()
        if result:
            fail(f'{stage} exited with code {result}')
    status.update(state='complete', ended_at=time.time()); write()


if __name__ == '__main__':
    main()
