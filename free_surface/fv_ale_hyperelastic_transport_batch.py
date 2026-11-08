"""One-variable lower transport-dissipation trial, serialized behind wall checks."""
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
    common=['--case','release','--mechanical-reconstruction','compatible','--affine-stencil','smooth',
            '--affine-cap-motion','balanced','--transport-reconstruction','entropy','--transport-dissipation','.03',
            '--interface-dissipation','.01','--cap-dissipation','0','--energy-ledger','--fast-energy-audit',
            '--work-retry-fraction','1e-6','--precision-geometry','--steps','30000','--frames',
            '--frame-interval','.002','--record-interval','.001','--merge-dt','1e-5']
    for until,tag in [('.01','hyperelastic_transport003_guarded10ms'),('.05','hyperelastic_transport003_guarded50ms')]:
        command=[sys.executable,'-m','free_surface.fv_ale_hyperelastic_check',*common,'--until',until,'--output',tag+'.json']
        print(json.dumps({'start':tag,'command':command}),flush=True);tick=time.perf_counter()
        with (root/(tag+'_batch.log')).open('w') as log:
            status=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        data=json.loads((root/(tag+'.json')).read_text()) if (root/(tag+'.json')).exists() else {}
        row={'tag':tag,'returncode':status.returncode,'seconds':time.perf_counter()-tick,
             **{key:data.get(key) for key in ['completed','time_reached','diagnostic_stop','final']}}
        rows.append(row);print(json.dumps(row),flush=True)
        (root/'hyperelastic_transport_batch.json').write_text(json.dumps(rows,indent=2)+'\n')
        if status.returncode or not data.get('completed'):break


if __name__=='__main__':main()
