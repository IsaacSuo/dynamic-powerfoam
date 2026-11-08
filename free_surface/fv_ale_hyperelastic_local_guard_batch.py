"""Serial full-foam validation of the local energy accuracy guard.

Every run starts from its original input; no old trajectory is re-labeled as
having satisfied this stronger acceptance policy.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--wait-pid',type=int,default=0);args=parser.parse_args()
    while args.wait_pid:
        try:os.kill(args.wait_pid,0)
        except ProcessLookupError:break
        time.sleep(5)
    root=Path('output/ale');rows=[]
    common=[sys.executable,'-m','free_surface.fv_ale_hyperelastic_check','--case','release',
            '--mechanical-reconstruction','compatible','--affine-stencil','smooth','--affine-cap-motion','balanced',
            '--transport-reconstruction','entropy','--transport-dissipation','.1','--cap-dissipation','0',
            '--energy-ledger','--fast-energy-audit','--work-retry-fraction','1e-6','--local-energy-tolerance','1e-6',
            '--precision-geometry','--steps','50000','--frames','--frame-interval','.001','--record-interval','.001']
    jobs=[('hyperelastic_nu049_localguard50ms',['--poisson','.49','--merge-dt','2e-6','--interface-dissipation','.1','--until','.05']),
          ('hyperelastic_alpha001_localguard50ms',['--merge-dt','1e-5','--interface-dissipation','.01','--until','.05']),
          ('hyperelastic_stretch2_localguard10ms',['--stretch','2','--merge-dt','1e-5','--interface-dissipation','.1','--until','.01'])]
    for tag,extra in jobs:
        command=common+extra+['--output',tag+'.json']
        print(json.dumps({'start':tag,'command':command}),flush=True);tick=time.perf_counter()
        with (root/(tag+'_batch.log')).open('w') as log:
            status=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        report=json.loads((root/(tag+'.json')).read_text())
        row={'tag':tag,'returncode':status.returncode,'seconds':time.perf_counter()-tick,
             'completed':report['completed'],'time_reached':report['time_reached'],
             'diagnostic_stop':report['diagnostic_stop'],'step_audit':report['step_audit']}
        rows.append(row);print(json.dumps(row),flush=True)
        (root/'hyperelastic_local_guard_batch.json').write_text(json.dumps(rows,indent=2)+'\n')
        if status.returncode or not report['completed']:raise RuntimeError('local-guard full run failed: '+tag)


if __name__=='__main__':main()
