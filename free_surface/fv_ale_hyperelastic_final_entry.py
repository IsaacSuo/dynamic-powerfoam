"""Fresh full production-entry release, resume, and old-default regression."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import torch


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--wait-pid',type=int,default=0);args=ap.parse_args()
    while args.wait_pid:
        try:os.kill(args.wait_pid,0)
        except ProcessLookupError:break
        time.sleep(5)
    rebound=Path('output/ale/hyperelastic_releasing_floorfix_batch.json')
    if not rebound.exists() or not json.loads(rebound.read_text())[-1]['five_ms_sustained_separation']:
        raise RuntimeError('Contact batch did not establish separation; leave the GPU available for diagnosis')
    root=Path('output/ale/hyperelastic_final_entry');root.mkdir(parents=True,exist_ok=True)
    common=[sys.executable,'-m','free_surface.merged_drop','--foam','foams/fourier8k_clean','--scene','0',
        '--save',str(root),'--frames','2','--fv-ale','--fv-ground-cut','--fv-agglomerate',
        '--fv-agglomerate-dt','1e-5','--fv-steps','30000','--fv-budget','3600']
    hyper=['--fv-constitutive','neo_hookean','--fv-hyperelastic-config','configs/hyperelastic_free_release_low_dissipation.json',
           '--mu','0','--fv-cfl','.5']
    jobs=[('free50ms',hyper+['--fv-until','.05']),
          ('resume',hyper+['--fv-until','.0501','--fv-resume',str(root/'fv_state_free50ms.pt')]),
          ('old_default',['--fv-until','.0001','--fv-no-freeflight'])]
    rows=[]
    for tag,extra in jobs:
        command=common+['--tag',tag]+extra;print(json.dumps({'start':tag,'command':command}),flush=True)
        tick=time.perf_counter()
        with (root/(tag+'.log')).open('w') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        reportpath=root/('fv_solid_'+tag+'.json')
        report=json.loads(reportpath.read_text()) if reportpath.exists() else {}
        row={'tag':tag,'returncode':result.returncode,'status':report.get('status'),
            'time_reached':report.get('time_reached'),'physical_steps':report.get('physical_steps'),
            'wall_seconds':time.perf_counter()-tick,'failure':report.get('failure'),
            'step_audit':report.get('hyperelastic_step_audit')}
        rows.append(row);(root/'progress.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(row),flush=True)
        assert result.returncode==0 and report['status']=='completed',row
        if tag!='old_default':
            assert report['hyperelastic_step_audit']['max_negative_remainder_over_local_elastic_plus_mu_over_rho0']<=1e-6
            state=torch.load(root/('fv_state_'+tag+'.pt'),weights_only=False,map_location='cpu')
            assert bool((state['q'][:,14:16]==0).all()),'pure hyperelastic branch accumulated plastic state'
            if tag=='free50ms':assert report['analytical_freeflight_time']==0.
        else:
            assert report['constitutive']=='wilkins' and report['cfl']==2. and report['physical_steps']>0
    summary={'passed':True,'fresh_production_release_and_resume':True,'old_default_exercised_physical_step':True,'runs':rows}
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
