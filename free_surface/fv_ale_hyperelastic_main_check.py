"""Exercise the original scene entry point, snapshot, resume and old default.

Runs the full program, not a replacement time integrator. GPU jobs are serial.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--wait-pid',type=int,default=0)
    args=parser.parse_args()
    while args.wait_pid:
        try:os.kill(args.wait_pid,0)
        except ProcessLookupError:break
        time.sleep(5)
    root=Path('output/ale/hyperelastic_main_check');root.mkdir(parents=True,exist_ok=True)
    common=[sys.executable,'-m','free_surface.merged_drop','--foam','foams/fourier8k_clean',
            '--save',str(root),'--fv-ale','--fv-ground-cut','--frames','1',
            '--fv-steps','1000','--fv-budget','600','--fv-agglomerate','--fv-agglomerate-dt','1e-5']
    hyper=['--fv-constitutive','neo_hookean','--fv-hyperelastic-config','configs/hyperelastic_free_release.json']
    jobs=[('main_free',hyper+['--fv-until','.0001']),
          ('main_resume',hyper+['--fv-until','.0002','--fv-resume',str(root/'fv_state_main_free.pt')]),
          ('old_default',['--fv-until','.0001'])]
    reports=[]
    for tag,extra in jobs:
        command=common+['--tag',tag]+extra
        with (root/(tag+'.log')).open('w') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        path=root/('fv_solid_'+tag+'.json')
        report=json.loads(path.read_text()) if path.exists() else {}
        row={'tag':tag,'returncode':result.returncode,'status':report.get('status'),
             'time_reached':report.get('time_reached'),'physical_steps':report.get('physical_steps'),
             'wall_seconds':report.get('wall_seconds'),'failure':report.get('failure'),
             'energy_audit':report.get('hyperelastic_step_audit')}
        reports.append(row)
        (root/'summary.json').write_text(json.dumps(reports,indent=2)+'\n')
        print(json.dumps(row),flush=True)
        if result.returncode or report.get('status')!='completed':
            raise RuntimeError('full entry-point check failed: '+tag+'; see '+str(root/(tag+'.log')))
        if tag=='main_free':
            assert report['analytical_freeflight_time']==0.,'prestrained body must not skip elastic dynamics'
            assert report['hyperelastic_step_audit']['scope']=='every accepted physical step in this invocation'
        if tag=='main_resume':
            assert abs(report['start_time']-.0001)<1e-14
        if tag=='old_default':
            assert report['constitutive']=='wilkins' and report['cfl']==2.
    # A changed constitutive configuration must be rejected before advancing.
    wrong=json.loads(Path('configs/hyperelastic_free_release.json').read_text())
    wrong['solver']['interface_dissipation']=.2
    wrong_path=root/'wrong_resume_config.json';wrong_path.write_text(json.dumps(wrong))
    command=common+['--tag','wrong_resume','--fv-constitutive','neo_hookean',
        '--fv-hyperelastic-config',str(wrong_path),'--fv-until','.0002',
        '--fv-resume',str(root/'fv_state_main_free.pt')]
    with (root/'wrong_resume.log').open('w') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    assert result.returncode!=0
    assert 'configuration mismatch' in (root/'wrong_resume.log').read_text()
    reports.append({'tag':'changed_configuration_resume','correctly_rejected':True})
    (root/'summary.json').write_text(json.dumps(reports,indent=2)+'\n')
    print(json.dumps(reports),flush=True)


if __name__=='__main__':main()
