"""Capture the first resolved local energy deficit, then check gravity entry."""
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
    command=[sys.executable,'-m','free_surface.fv_ale_hyperelastic_check','--case','release',
             '--mechanical-reconstruction','compatible','--affine-stencil','smooth','--affine-cap-motion','balanced',
             '--transport-reconstruction','entropy','--transport-dissipation','.1','--interface-dissipation','.1',
             '--cap-dissipation','0','--energy-ledger','--fast-energy-audit','--work-retry-fraction','1e-6',
             '--precision-geometry','--poisson','.49','--merge-dt','2e-6','--until','.003','--steps','10000',
             '--local-energy-snapshot-threshold','.001','--record-interval','.0001',
             '--output','hyperelastic_nu049_local_capture.json']
    jobs=[('hyperelastic_nu049_local_capture',command),('hyperelastic_gravity_entry_check',[
        sys.executable,'-m','free_surface.fv_ale_hyperelastic_gravity_entry_check'])]
    for tag,command in jobs:
        print(json.dumps({'start':tag,'command':command}),flush=True);tick=time.perf_counter()
        with (root/(tag+'_batch.log')).open('w') as log:
            status=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        row={'tag':tag,'returncode':status.returncode,'seconds':time.perf_counter()-tick}
        path=root/(tag+'.json')
        if path.exists():
            data=json.loads(path.read_text());row.update({key:data.get(key) for key in ['completed','time_reached','diagnostic_stop']})
        rows.append(row);print(json.dumps(row),flush=True)
        (root/'hyperelastic_local_energy_batch.json').write_text(json.dumps(rows,indent=2)+'\n')
        if status.returncode:raise RuntimeError('diagnostic execution failed: '+tag)


if __name__=='__main__':main()
