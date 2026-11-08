"""Exercise production scene resume with an actual separated contact boundary.

The seed is a documented format conversion of the independently audited GPU
release replay, not a fresh landing. Its p, r, q, groups and planes are unchanged.
Compare an uninterrupted continuation with a serialized split continuation.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import torch


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--wait-pid',type=int,default=0)
    ap.add_argument('--seed',type=Path,default=Path('output/ale/hyperelastic_releasing_step_cuda.pt'))
    ap.add_argument('--output',type=Path,default=Path('output/ale/hyperelastic_releasing_entry'));args=ap.parse_args()
    while args.wait_pid:
        try:os.kill(args.wait_pid,0)
        except ProcessLookupError:break
        time.sleep(5)
    root=args.output;root.mkdir(parents=True,exist_ok=True)
    source=args.seed
    d=torch.load(source,weights_only=False,map_location='cpu')
    baseline=torch.load('output/ale/hyperelastic_main_check/fv_state_main_free.pt',weights_only=False,map_location='cpu')
    audit=json.loads(source.with_suffix('.json').read_text())
    config=json.loads(Path('configs/hyperelastic_releasing_drop.json').read_text())
    config['solver']['ground']=d['ground'];config['solver']['gravity']=[0.,0.,0.]
    config['initial_velocity']=[0.,-1.,0.]
    configpath=root/'config.json';configpath.write_text(json.dumps(config,indent=2)+'\n')
    from free_surface.fv_ale_hyperelastic_config import load_config
    normalized=load_config(configpath)
    seed=dict(baseline)
    seed.update(d,hyperelastic_config=normalized,gravity=[0.,0.,0.],friction=0.,
        steps=len(audit['rows']),frames_file=None,reason='audited diagnostic-to-main format conversion',
        work_retry_tolerance_joule=audit['audit']['tolerance_joule'],
        work_retry_reference={'scale_joule':audit['audit']['tolerance_joule']/1e-6},
        diagnostic_conversion_source=str(source.resolve()),boundary_insertions=audit['activations'],
        boundary_prunings=audit['prunings'])
    assert len(d['moving_planes']['owner'])>0
    seedpath=root/'seed.pt';torch.save(seed,seedpath)
    common=[sys.executable,'-m','free_surface.merged_drop','--foam','foams/fourier8k_clean',
        '--scene','0','--save',str(root),'--frames','2','--fv-ale','--fv-ground-cut','--mu','0',
        '--fv-constitutive','neo_hookean','--fv-hyperelastic-config',str(configpath),
        '--fv-cfl','.5','--dt','1e-6','--fv-steps','1000','--fv-budget','600']
    records=[];t0=float(d['time'])
    jobs=[('split1',seedpath,t0+1e-5),('split2',root/'fv_state_split1.pt',t0+2e-5),
          ('direct',seedpath,t0+2e-5)]
    for tag,start,end in jobs:
        command=common+['--tag',tag,'--fv-resume',str(start),'--fv-until',repr(end)]
        with (root/(tag+'.log')).open('w') as log:
            status=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        reportpath=root/('fv_solid_'+tag+'.json')
        report=json.loads(reportpath.read_text()) if reportpath.exists() else {}
        row={'tag':tag,'returncode':status.returncode,'status':report.get('status'),
             'time_reached':report.get('time_reached'),'failure':report.get('failure')}
        records.append(row);print(json.dumps(row),flush=True)
        (root/'progress.json').write_text(json.dumps(records,indent=2)+'\n')
        assert status.returncode==0 and report['status']=='completed',row
    split=torch.load(root/'fv_state_split2.pt',weights_only=False,map_location='cpu')
    direct=torch.load(root/'fv_state_direct.pt',weights_only=False,map_location='cpu')
    errors={key:float((split[key]-direct[key]).norm()/direct[key].norm().clamp(min=1e-300))
            for key in ['p','r','q']}
    assert torch.equal(split['moving_planes']['owner'],direct['moving_planes']['owner'])
    for key in ['d','a','b']:
        a=split['moving_planes'][key];b=direct['moving_planes'][key]
        errors['planes_'+key]=float((a-b).norm()/b.norm().clamp(min=1e-300))
    assert max(errors.values())<1e-7,errors
    result={'passed':True,'seed_source':str(source),'unchanged_seed_fields':['p','r','q','groups','moving_planes'],
            'continuations':records,'direct_split_relative_errors':errors,
            'scope':'production entry and serialized boundary continuation, not a fresh full landing'}
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
