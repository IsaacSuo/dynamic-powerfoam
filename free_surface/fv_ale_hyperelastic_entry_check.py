"""Validate actual scene entry/resume with saved material and local tolerance."""
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
    root=Path('output/ale/hyperelastic_local_guard_entry');root.mkdir(parents=True,exist_ok=True)
    config=json.loads(Path('configs/hyperelastic_free_release.json').read_text())
    config['material']={'E':200000.,'nu':.49}
    path=root/'config.json';path.write_text(json.dumps(config,indent=2)+'\n')
    from free_surface.fv_ale_hyperelastic_config import load_config
    normalized=load_config(path)
    assert normalized['material']==config['material'] and normalized['local_energy_tolerance']==1e-6
    common=[sys.executable,'-m','free_surface.merged_drop','--foam','foams/fourier8k_clean',
            '--scene','0','--save',str(root),'--frames','1','--fv-ale','--fv-ground-cut',
            '--fv-constitutive','neo_hookean','--fv-hyperelastic-config',str(path),
            '--fv-agglomerate','--fv-agglomerate-dt','2e-6','--fv-steps','1000','--fv-budget','600']
    rows=[];tolerance=None
    for tag,extra in [('start',['--fv-until','.0001']),('resume',['--fv-until','.0002','--fv-resume',str(root/'fv_state_start.pt')])]:
        with (root/(tag+'.log')).open('w') as log:
            status=subprocess.run(common+['--tag',tag]+extra,stdout=log,stderr=subprocess.STDOUT)
        report=json.loads((root/('fv_solid_'+tag+'.json')).read_text())
        assert status.returncode==0 and report['status']=='completed',report.get('failure')
        assert report['material']['E']==200000. and report['material']['nu']==.49
        assert report['original_scene_material']['nu']!=.49
        audit=report['hyperelastic_step_audit']
        assert audit['local_energy_tolerance']==1e-6
        assert audit['max_negative_remainder_over_local_elastic_plus_mu_over_rho0']<=1e-6
        if tolerance is None:tolerance=audit['tolerance_joule']
        else:assert tolerance==audit['tolerance_joule']
        rows.append({'tag':tag,'status':report['status'],'time_reached':report['time_reached'],
                     'physical_steps':report['physical_steps'],'material':report['material'],'audit':audit})
    changed=dict(config,material={'E':210000.,'nu':.49})
    changed_path=root/'changed_config.json';changed_path.write_text(json.dumps(changed,indent=2)+'\n')
    changed_command=list(common);changed_command[changed_command.index(str(path))]=str(changed_path)
    log_path=root/'changed_rejected.log'
    with log_path.open('w') as log:
        status=subprocess.run(changed_command+['--tag','changed','--fv-until','.0002','--fv-resume',str(root/'fv_state_start.pt')],stdout=log,stderr=subprocess.STDOUT)
    assert status.returncode!=0 and 'configuration mismatch' in log_path.read_text()
    result={'passed':True,'start_resume':rows,'changed_material_resume_rejected':True}
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
