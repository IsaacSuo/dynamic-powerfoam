"""Full main-entry start/resume from zero stored energy in gravity."""
import json
import argparse
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
    root=Path('output/ale/hyperelastic_gravity_entry');root.mkdir(parents=True,exist_ok=True)
    common=[sys.executable,'-m','free_surface.merged_drop','--foam','foams/fourier8k_clean',
            '--scene','0','--save',str(root),'--frames','1','--fv-ale','--fv-ground-cut',
            '--fv-constitutive','neo_hookean','--fv-hyperelastic-config','configs/hyperelastic_frictionless_drop.json',
            '--mu','0','--fv-agglomerate','--fv-agglomerate-dt','1e-5','--fv-steps','1000','--fv-budget','600']
    rows=[];tolerance=None
    for tag,extra in [('rest',['--fv-until','.0001']),('resume',['--fv-until','.0002','--fv-resume',str(root/'fv_state_rest.pt')])]:
        with (root/(tag+'.log')).open('w') as log:
            status=subprocess.run(common+['--tag',tag]+extra,stdout=log,stderr=subprocess.STDOUT)
        report=json.loads((root/('fv_solid_'+tag+'.json')).read_text())
        assert status.returncode==0 and report['status']=='completed',report.get('failure')
        reference=report['work_retry_reference'];assert reference['initial_total_energy_joule']==0
        assert reference['scale_joule']==reference['initial_available_vertical_gravity_joule']>0
        audit=report['hyperelastic_step_audit']
        if tolerance is None:tolerance=audit['tolerance_joule']
        else:assert audit['tolerance_joule']==tolerance
        rows.append({'tag':tag,'status':report['status'],'time_reached':report['time_reached'],
                     'work_retry_reference':reference,'audit':audit})
    (root/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
