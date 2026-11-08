"""Serial GPU checks after the independently running material validation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--wait-pid',type=int,default=0)
    ap.add_argument('--prefix',default='hyperelastic_releasing_impact')
    ap.add_argument('--until',type=float,default=.06)
    ap.add_argument('--start',type=Path,default=Path('output/ale/hyperelastic_acoustic_impact50ms.pt'))
    ap.add_argument('--continue-rebound',action='store_true');args=ap.parse_args()
    while args.wait_pid:
        try:os.kill(args.wait_pid,0)
        except ProcessLookupError:break
        time.sleep(5)
    root=Path('output/ale');rows=[]
    def run(tag,command,report):
        print(json.dumps({'start':tag,'command':command}),flush=True);start=time.perf_counter()
        with (root/(tag+'_validation.log')).open('w') as log:
            status=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        record=json.loads(report.read_text()) if report.exists() else {}
        okay=status.returncode==0 and record.get('passed',record.get('completed',False))
        row={'tag':tag,'returncode':status.returncode,'passed':okay,'wall_seconds':time.perf_counter()-start,
             'time_reached':record.get('time_reached'),'diagnostic_stop':record.get('diagnostic_stop')}
        rows.append(row);print(json.dumps(row),flush=True)
        (root/(args.prefix+'_validation_batch.json')).write_text(json.dumps(rows,indent=2)+'\n')
        if not okay:raise RuntimeError('boundary validation failed: '+tag)
        return record
    short=args.prefix+'_short_cuda'
    run(short,[sys.executable,'-m','free_surface.fv_ale_releasing_step_check','--device','cuda','--tag',short,'--snapshot',str(args.start)],
        root/(short+'.json'))
    source=json.loads(args.start.with_suffix('.json').read_text())
    command=[sys.executable,'-m','free_surface.fv_ale_hyperelastic_check']+source['command_line'][1:]
    if '--upgrade-release-boundary' in command:command.remove('--upgrade-release-boundary')
    tag=args.prefix+str(round(args.until*1000))+'ms'
    for flag,value in [('--until',repr(args.until)),('--steps','100000'),('--output',tag+'.json'),('--frame-interval','.001'),('--record-interval','.001')]:
        if flag in command:command[command.index(flag)+1]=value
        else:command.extend([flag,value])
    if '--resume' in command:command[command.index('--resume')+1]=str(args.start)
    else:command+=['--resume',str(args.start)]
    if '--releasing-planes' not in command:command.append('--releasing-planes')
    if not source.get('releasing_planes',False):command.append('--upgrade-release-boundary')
    run(tag,command,root/(tag+'.json'))
    print('Boundary short-step and requested continuation passed; longer rebound still requires validation.',flush=True)
    if args.continue_rebound:
        subprocess.run([sys.executable,'-m','free_surface.fv_ale_rebound_batch','--source',tag,'--prefix',args.prefix],check=True)


if __name__=='__main__':main()
