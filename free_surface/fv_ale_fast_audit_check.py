"""Compare full-program trajectories with independent versus cached audits."""
import json
from pathlib import Path
import subprocess
import sys
import torch


def main():
    root=Path('output/ale')
    common=[sys.executable,'-m','free_surface.fv_ale_hyperelastic_check',
        '--case','release','--until','.001','--steps','2000','--merge-dt','1e-5',
        '--mechanical-reconstruction','compatible','--affine-stencil','smooth',
        '--affine-cap-motion','balanced','--transport-reconstruction','entropy',
        '--transport-dissipation','.1','--interface-dissipation','.1','--cap-dissipation','0',
        '--energy-ledger','--work-retry-fraction','1e-6']
    reports=[];states=[]
    for name,fast in [('independent_v2',False),('independent_repeat',False),('cached_every_step',True)]:
        tag='hyperelastic_audit_'+name+'_1ms'
        with (root/(tag+'.log')).open('w') as log:
            result=subprocess.run(common+(['--fast-energy-audit','--audit-crosscheck-every','1'] if fast else [])+['--output',tag+'.json'],
                                  stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError('audit comparison run failed: '+tag)
        report=json.loads((root/(tag+'.json')).read_text())
        assert report['completed'] and report['diagnostic_stop'] is None,tag
        reports.append(report);states.append(torch.load(root/(tag+'.pt'),map_location='cpu',weights_only=False))
    differences={key:float((states[0][key]-states[2][key]).abs().max()/states[0][key].abs().max().clamp(min=1.))
                 for key in ['p','r','q']}
    repeat={key:float((states[0][key]-states[1][key]).abs().max()/states[0][key].abs().max().clamp(min=1.))
            for key in ['p','r','q']}
    # Atomic GPU reductions are not bitwise reproducible. Require changes to
    # stay within the measured same-code repeat spread as well as a fixed
    # accuracy ceiling; do not infer equivalence from timings alone.
    assert all(differences[key]<max(5*repeat[key],1e-8) for key in differences),(differences,repeat)
    assert max(differences.values())<1e-5 and max(repeat.values())<1e-5,(differences,repeat)
    assert len(set(r['steps'] for r in reports))==1
    assert all(torch.equal(states[0]['groups'],s['groups']) for s in states[1:])
    ledger={key:abs(value-reports[2]['energy_ledger'][key])/max(abs(value),1.)
            for key,value in reports[0]['energy_ledger'].items()}
    assert max(ledger.values())<1e-6,ledger
    report={'passed':True,'state_relative_max_errors':differences,'same_code_repeat_errors':repeat,'ledger_relative_errors':ledger,
            'physical_steps':reports[0]['steps'],'independent_seconds':reports[0]['seconds'],
            'cached_seconds_with_every_step_crosscheck':reports[2]['seconds'],'cached_checks':reports[2]['fast_audit_crosscheck']}
    (root/'hyperelastic_fast_audit_check.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
