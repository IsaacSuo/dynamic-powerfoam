"""Independent acoustic-matrix checks of the all-direction closed maximum."""
import json
from pathlib import Path
import torch
from free_surface.fv_ale_hyperelastic_material import acoustic_tensor,maximum_acoustic_eigenvalue


def main():
    torch.set_num_threads(1);torch.manual_seed(781);dtype=torch.float64
    logs=torch.randn((4096,3),dtype=dtype)*3
    logs-=logs.mean(1)[:,None];logs=logs.sort(1).values
    logs[:8]=0.;values=logs.exp();B=torch.diag_embed(values)
    J=torch.exp(torch.linspace(-3.,3.,len(values),dtype=dtype))
    results=[]
    for nu in [-.5,0.,.3,.49,.499]:
        E=2e5;mu=E/(2*(1+nu));bulk=E/(3*(1-2*nu))
        bound,beta=maximum_acoustic_eigenvalue(values,J,mu,bulk,return_beta=True)
        a,b=values[:,0],values[:,-1]
        t=torch.where(b>a,(beta-a)/(b-a).clamp(min=1e-300),0.).clamp(0,1)
        n=torch.stack(((1-t).sqrt(),torch.zeros_like(t),t.sqrt()),1)
        attained=torch.linalg.eigvalsh(acoustic_tensor(B,J,n,mu,bulk))[:,-1]
        exact_error=float(((bound-attained)/bound).abs().max())
        assert exact_error<2e-12,(nu,exact_error)
        sampled_ratio=0.
        for _ in range(32):
            n=torch.randn((len(values),3),dtype=dtype);n/=n.norm(dim=1)[:,None]
            observed=torch.linalg.eigvalsh(acoustic_tensor(B,J,n,mu,bulk))[:,-1]
            sampled_ratio=max(sampled_ratio,float((observed/bound).max()))
        assert sampled_ratio<=1+1e-12,(nu,sampled_ratio)
        old=.5*bulk*(J+1/J)+(mu/J)*(7*values[:,-1]/3+5*values.sum(1)/9)
        assert bool((bound<=old*(1+1e-12)).all())
        reference=maximum_acoustic_eigenvalue(torch.ones((1,3),dtype=dtype),torch.ones(1,dtype=dtype),mu,bulk)
        reference_error=float((reference/(bulk+4*mu/3)-1).abs().max())
        assert reference_error<1e-12
        results.append({'nu':nu,'attaining_direction_relative_error':exact_error,
            'max_sampled_eigenvalue_over_bound':sampled_ratio,'isotropic_reference_error':reference_error,
            'old_over_tight_speed_median':float((old/bound).sqrt().median()),
            'old_over_tight_speed_max':float((old/bound).sqrt().max())})
    report={'tested_states':len(values),'J_range':[float(J.min()),float(J.max())],
            'metric_eigenvalue_range':[float(values.min()),float(values.max())],'results':results,
            'production_default_changed':False}
    Path('output/ale/hyperelastic_acoustic_bound_check.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
