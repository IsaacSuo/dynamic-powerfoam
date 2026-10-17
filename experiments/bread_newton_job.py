"""Run the full Newton trajectory, compare it, then render saved motion."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out/'job.json').exists() or (args.out/'config.json').exists():
        parser.error('choose a new output directory')
    status = dict(pid=os.getpid(), started_at=time.time(), state='running')
    def write():
        (args.out/'job.tmp.json').write_text(json.dumps(status, indent=2)+'\n')
        (args.out/'job.tmp.json').replace(args.out/'job.json')
    commands = [
        ('simulation', ['experiments.bread_solver_run', '--solver', 'newton', '--case', 'initial',
                        '--intervals', '192', '--out', str(args.out)]),
        ('comparison', ['experiments.bread_newton_trajectory_report', '--new', str(args.out)]),
        ('render', ['experiments.bread_finite_sim', '--out', str(args.out), '--render-only'])]
    if args.resume:
        commands[0][1].extend(['--resume', str(args.resume)])
        status['resumed_from'] = str(args.resume)
    for stage, command in commands:
        status.update(stage=stage, stage_started_at=time.time()); write()
        child = subprocess.Popen([sys.executable, '-m', *command])
        status['child_pid'] = child.pid; write()
        result = child.wait()
        if result:
            status.update(state='failed', returncode=result, ended_at=time.time()); write()
            raise SystemExit(result)
    status.update(state='complete', ended_at=time.time()); write()


if __name__ == '__main__':
    main()
