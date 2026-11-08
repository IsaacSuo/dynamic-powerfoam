"""CPU checks of opt-in configuration and nonmutating merge fallback."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import torch
from free_surface.fv_ale_agglomerate import merge_cell
from free_surface.fv_ale_hyperelastic_config import load_config


def main():
    config=load_config('configs/hyperelastic_free_release.json')
    assert config['initial_stretch']==1.5 and config['solver']['gravity']==[0.,0.,0.]
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/'invalid.json'
        for bad in [dict(config,typo=1),dict(config,initial_stretch=-1),
                    dict(config,solver={'gravity':[0.,float('nan'),0.]}),dict(config,solver=[]),
                    dict(config,work_retry_fraction=-1)]:
            path.write_text(json.dumps(bad))
            try:load_config(path)
            except ValueError:pass
            else:raise AssertionError('invalid configuration accepted')
    labels=torch.tensor([0,0,2,2]);volume=torch.ones(4,dtype=torch.float64)
    solver=SimpleNamespace(groups=labels,group_sum=lambda x:torch.zeros_like(x).index_add(0,labels,x))
    state=SimpleNamespace(r=volume.clone(),q=torch.ones((4,16),dtype=torch.float64),
                          mesh=SimpleNamespace(volume=volume,i=torch.tensor([1]),j=torch.tensor([2]),area=torch.ones(1,dtype=volume.dtype)))
    before=state.q.clone()
    try:merge_cell(solver,state,0,max_members=3)
    except RuntimeError as exc:assert 'no adjacent admissible FV group' in str(exc)
    else:raise AssertionError('old default unexpectedly changed')
    result,info=merge_cell(solver,state,0,max_members=3,allow_unavailable=True)
    assert result is state and solver.groups is labels and torch.equal(state.q,before)
    assert info['skipped'] and info['candidate_groups']==[2] and info['candidate_member_counts']==[2]
    report={'invalid_configurations_rejected':5,'legacy_merge_error_preserved':True,
            'fallback_changes_no_state':True,'diagnostic':info}
    Path('output/ale/hyperelastic_config_check.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
