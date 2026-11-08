"""Mass-weighted local deformation distributions for matched completed runs.

Different cell/particle resolutions: these are distributions, not pointwise
error norms or proof that either discretization is locally converged.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import warp as wp
from free_surface.fv_ale_compatible_mesh_audit import restored
from free_surface.fv_ale_hyperelastic_material import symmetric_deviator


def distribution(value,mass):
    x=value.numpy();w=mass.numpy();keep=w>0;x=x[keep];w=w[keep];w=w/w.sum()
    order=np.argsort(x);cdf=np.cumsum(w[order]);levels=np.array([0.,.001,.01,.1,.5,.9,.99,.999,1.])
    mean=float(np.dot(x,w))
    return {'mass_quantile_levels':levels.tolist(),'quantiles':np.interp(levels,cdf,x[order]).tolist(),
            'mass_weighted_mean':mean,'mass_weighted_standard_deviation':float(np.sqrt(np.dot((x-mean)**2,w)))}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--fv',required=True);parser.add_argument('--mpm',required=True)
    args=parser.parse_args();torch.set_num_threads(1);wp.init();root=Path('output/ale')
    fr=json.loads((root/(args.fv+'.json')).read_text());mr=json.loads((root/(args.mpm+'.json')).read_text())
    assert fr['completed'] and mr['status']=='completed'
    assert abs(fr['time_reached']-mr['time_reached'])<1e-12
    assert all(np.isclose(fr['material'][key],mr[key],rtol=1e-12,atol=0) for key in ['E','nu'])
    assert fr['stretch']==mr['stretch']
    data=torch.load(root/(args.fv+'.pt'),map_location='cpu',weights_only=False)
    sol,state,_,_=restored(data);q=state.q;total=sol.group_sum(q)[sol.groups]
    volume=sol.group_sum(state.mesh.volume)[sol.groups]
    J=sol.rho0*volume/total[:,0];H=total[:,5:14].reshape(-1,3,3)/total[:,0,None,None]
    eigen=torch.linalg.eigvalsh(symmetric_deviator(H))
    stretches=torch.exp(.5*eigen)*J[:,None]**(1/3)
    fv={'J':distribution(J,q[:,0]),'largest_principal_stretch':distribution(stretches[:,-1],q[:,0]),
        'smallest_principal_stretch':distribution(stretches[:,0],q[:,0])}
    data=torch.load(root/(args.mpm+'_end.pt'),map_location='cpu',weights_only=False)
    F=data['F'].double();mass=data['mass'].double();J=torch.linalg.det(F)
    assert bool((J>0).all())
    stretches=torch.linalg.svdvals(F)
    mpm={'J':distribution(J,mass),'largest_principal_stretch':distribution(stretches[:,0],mass),
         'smallest_principal_stretch':distribution(stretches[:,-1],mass)}
    result={'fv':args.fv,'mpm':args.mpm,'time_seconds':fr['time_reached'],
            'FV':fv,'MPM':mpm,'scope':'Local mass distributions at different spatial resolutions; not pointwise accuracy or convergence.'}
    path=root/(args.fv+'_material_distributions.json');path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
