"""Full production ablation of continuous cap motion and remembered dt."""
import json,subprocess,sys,time
from pathlib import Path

def main():
    root=Path('output/ale/cfl_continuity');root.mkdir(parents=True,exist_ok=True)
    base=json.loads(Path('configs/hyperelastic_free_release_low_dissipation.json').read_text())
    rows=[]
    for tag,eps,remember in [('original',0.,False),('memory',0.,True),('cap',.01,False),('both',.01,True)]:
        config=json.loads(json.dumps(base))
        config['solver']['cap_motion_regularization']=eps
        config['step_control']='remember' if remember else 'original'
        path=root/(tag+'.json');path.write_text(json.dumps(config,indent=2)+'\n')
        command=[sys.executable,'-m','free_surface.merged_drop','--foam','foams/fourier8k_clean','--scene','0',
            '--save',str(root),'--frames','2','--fv-ale','--fv-ground-cut','--fv-agglomerate',
            '--fv-agglomerate-dt','4e-5','--fv-steps','30000','--fv-budget','600',
            '--fv-constitutive','neo_hookean','--fv-hyperelastic-config',str(path),
            '--mu','0','--fv-cfl','2','--fv-until','.001','--tag',tag]
        print(json.dumps({'start':tag,'command':command}),flush=True);tick=time.perf_counter()
        with (root/(tag+'.log')).open('w') as log:proc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        p=root/('fv_solid_'+tag+'.json');d=json.loads(p.read_text()) if p.exists() else {}
        row={'tag':tag,'returncode':proc.returncode,'process_seconds':time.perf_counter()-tick,
             **{k:d.get(k) for k in ['status','time_reached','physical_steps','wall_seconds','rejected_attempts','rejection_reasons',
                'hyperelastic_final_energy','hyperelastic_step_audit','control_volume_count']}}
        rows.append(row);(root/'ablation.json').write_text(json.dumps(rows,indent=2)+'\n')
        print(json.dumps(row),flush=True)
        if proc.returncode!=0 or d.get('status')!='completed':break

if __name__=='__main__':main()
