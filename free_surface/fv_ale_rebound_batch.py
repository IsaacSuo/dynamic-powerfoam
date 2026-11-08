"""Continue the same audited impact until separation, in saved 20 ms chunks."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--wait-pid',type=int,default=0)
    ap.add_argument('--source',default='hyperelastic_releasing_impact60ms')
    ap.add_argument('--prefix',default='hyperelastic_releasing_impact');args=ap.parse_args()
    while args.wait_pid:
        try:os.kill(args.wait_pid,0)
        except ProcessLookupError:break
        time.sleep(5)
    root=Path('output/ale')
    entry=json.loads((root/'hyperelastic_releasing_entry/summary.json').read_text())
    assert entry['passed']
    previous=args.source;records=[]
    for end in [.08,.10,.12,.14,.16,.18]:
        prior=json.loads((root/(previous+'.json')).read_text())
        assert prior['completed'],prior.get('diagnostic_stop')
        if prior['time_reached']>=end-1e-14:continue
        command=[sys.executable,'-m','free_surface.fv_ale_hyperelastic_check']+prior['command_line'][1:]
        if '--upgrade-release-boundary' in command:command.remove('--upgrade-release-boundary')
        tag=args.prefix+str(round(end*1000))+'ms'
        for flag,value in [('--until',repr(end)),('--steps','100000'),('--output',tag+'.json'),
                           ('--resume',str(root/(previous+'.pt'))),('--frame-interval','.001'),('--record-interval','.001')]:
            if flag in command:command[command.index(flag)+1]=value
            else:command.extend([flag,value])
        print(json.dumps({'start':tag,'command':command}),flush=True);tick=time.perf_counter()
        with (root/(tag+'_validation.log')).open('w') as log:
            status=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        result=json.loads((root/(tag+'.json')).read_text())
        last=result['history'][-1]
        recent=[r for r in result['history'] if r['time']>=end-.005-1e-12]
        separated=len(recent)>=5 and all(r['contact_area_m2']<1e-10 and r['com_velocity_m_s'][1]>0 for r in recent)
        row={'tag':tag,'returncode':status.returncode,'completed':result['completed'],
             'time_reached':result['time_reached'],'seconds_this_chunk':time.perf_counter()-tick,
             'contact_area_m2':last['contact_area_m2'],'vertical_com_velocity_m_s':last['com_velocity_m_s'][1],
             'five_ms_sustained_separation':separated,'diagnostic_stop':result.get('diagnostic_stop')}
        records.append(row);print(json.dumps(row),flush=True)
        (root/(args.prefix+'_batch.json')).write_text(json.dumps(records,indent=2)+'\n')
        if status.returncode or not result['completed']:raise RuntimeError('rebound continuation failed: '+tag)
        previous=tag
        if separated:
            print('Sustained upward separation observed; examine energy, geometry and render before acceptance.',flush=True)
            return
    raise RuntimeError('No sustained complete separation by 180 ms; do not label this completed rebound')


if __name__=='__main__':main()
