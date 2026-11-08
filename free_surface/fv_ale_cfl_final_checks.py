"""Finalize numerical checks, controller restart and legacy-step regression."""
import json,subprocess,sys
from pathlib import Path
import torch

def main():
    root=Path('output/ale/cfl_continuity');rows=[]
    common=[sys.executable,'-m','free_surface.merged_drop','--foam','foams/fourier8k_clean','--scene','0',
            '--save',str(root),'--frames','2','--fv-ale','--fv-ground-cut','--fv-agglomerate','--fv-steps','1000','--fv-budget','120']
    jobs=[('free_checks',[sys.executable,'-m','free_surface.fv_ale_cap_regularization_check']),
          ('contact_checks',[sys.executable,'-m','free_surface.fv_ale_cap_contact_check']),
          ('resume',common+['--tag','memory_resume','--fv-agglomerate-dt','4e-5','--fv-cfl','2',
             '--fv-constitutive','neo_hookean','--fv-hyperelastic-config','configs/hyperelastic_free_release_adaptive.json',
             '--mu','0','--fv-resume',str(root/'fv_state_memory5ms.pt'),'--fv-until','.00505']),
          ('legacy',common+['--tag','legacy_check','--fv-agglomerate-dt','1e-5','--fv-no-freeflight','--fv-until','.0001'])]
    for tag,cmd in jobs:
        with (root/('final_'+tag+'.log')).open('w') as f:proc=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
        row={'tag':tag,'returncode':proc.returncode};rows.append(row)
        (root/'final_checks.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(row),flush=True)
        assert proc.returncode==0,row
    a=torch.load(root/'fv_state_memory5ms.pt',map_location='cpu',weights_only=False)
    b=torch.load(root/'fv_state_memory_resume.pt',map_location='cpu',weights_only=False)
    history=json.loads((root/'fv_steps_memory_resume.json').read_text())
    assert history[0]['proposed_dt']<=a['step_control_state']['next_dt']*(1+1e-12)
    assert b['step_control_state']['next_dt']>0 and b['time']==.00505
    old=json.loads((root/'fv_solid_legacy_check.json').read_text())
    assert old['status']=='completed' and old['constitutive']=='wilkins' and old['cfl']==2.
    print('Production controller resume and legacy defaults passed',flush=True)

if __name__=='__main__':main()
