"""Serial opt-in wall validation after the current material comparison batch."""
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
    common=['--case','impact','--wall-mode','compatible_slip','--mechanical-reconstruction','compatible',
            '--affine-stencil','smooth','--affine-cap-motion','balanced','--transport-reconstruction','entropy',
            '--transport-dissipation','.1','--interface-dissipation','.1','--cap-dissipation','0',
            '--energy-ledger','--fast-energy-audit','--work-retry-fraction','1e-6','--precision-geometry',
            '--steps','20000','--frames','--frame-interval','.001','--record-interval','.0002',
            '--merge-dt','1e-5','--impact-gap','.0002','--impact-speed','1']
    jobs=[('hyperelastic_wall_gpu_check','fv_ale_compatible_wall_check',['--device','cuda']),
          ('hyperelastic_wall_default_regression','fv_ale_hyperelastic_check',[
              '--case','regression','--output','hyperelastic_wall_default_regression.json']),
          ('hyperelastic_compatible_impact5ms','fv_ale_hyperelastic_check',common+[
              '--until','.005','--output','hyperelastic_compatible_impact5ms.json'])]
    for tag,module,extra in jobs:
        command=[sys.executable,'-m','free_surface.'+module,*extra]
        print(json.dumps({'start':tag,'command':command}),flush=True);tick=time.perf_counter()
        with (root/(tag+'_batch.log')).open('w') as log:
            status=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        row={'tag':tag,'returncode':status.returncode,'seconds':time.perf_counter()-tick}
        output=root/(tag+'.json')
        if output.exists():
            data=json.loads(output.read_text());row.update({key:data.get(key) for key in ['completed','time_reached','diagnostic_stop']})
        rows.append(row);print(json.dumps(row),flush=True)
        (root/'hyperelastic_wall_batch.json').write_text(json.dumps(rows,indent=2)+'\n')
        if status.returncode:raise RuntimeError('wall validation failed: '+tag)


if __name__=='__main__':main()
