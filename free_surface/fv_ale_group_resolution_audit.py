"""Read-only audit of persistent FV grouping after a full elastic release.

The individual-cell acoustic estimate omits relative transport and therefore
is only a candidate screen, not an admissible time-step certificate.
No state or grouping is changed by this diagnostic.
"""
import argparse
import json
from pathlib import Path
import torch
import warp as wp
from free_surface.fv_ale_precise_geometry_check import cpu_mesh
from free_surface.fv_ale_hyperelastic_material import response


def main():
    parser=argparse.ArgumentParser();parser.add_argument('tag');args=parser.parse_args()
    torch.set_num_threads(1);wp.init();root=Path('output/ale')
    data=torch.load(root/(args.tag+'.pt'),map_location='cpu',weights_only=False)
    report=json.loads((root/(args.tag+'.json')).read_text())
    q,g=data['q'],data['groups'];n=len(g);m=data['material']
    mesh=cpu_mesh(data['p'],data['r']);volume=mesh.volume.clamp(min=0)
    summed=lambda x:torch.zeros_like(x).index_add(0,g,x)
    total=summed(q)[g];groupV=summed(volume)[g];rho=total[:,0]/groupV
    H=total[:,5:14].reshape(-1,3,3)/total[:,0,None,None]
    mu=m['E']/(2*(1+m['nu']));bulk=m['E']/(3*(1-2*m['nu']))
    out=response(H,rho,mu,bulk,1000.,data.get('acoustic_bound','conservative'))
    area=mesh.cap_area.clamp(min=0)+torch.zeros_like(volume).index_add(0,mesh.i,mesh.area).index_add(0,mesh.j,mesh.area)
    dt=report['cfl']*volume/(area*out[4].sqrt()).clamp(min=1e-300)
    counts=torch.bincount(g,minlength=n);roots=torch.unique(g);rows=[]
    for factor in [1.,2.,4.]:
        eligible=dt>factor*report['merge_dt_target']
        failures=summed((~eligible).double())
        groups=roots[(counts[roots]>1)&(failures[roots]==0)]
        mask=torch.isin(g,groups)
        rows.append({'acoustic_target_factor':factor,'fully_eligible_groups':len(groups),
            'member_count':int(mask.sum()),'possible_extra_degrees_of_freedom':int(mask.sum())-len(groups),
            'mass_fraction':float(q[mask,0].sum()/q[:,0].sum())})
    result={'tag':args.tag,'cells':n,'control_volume_groups':len(roots),
        'group_size_quantiles':torch.quantile(counts[roots].double(),torch.tensor([.5,.9,.99,1.],dtype=torch.float64)).tolist(),
        'candidate_screens':rows,'scope':'No mutation. Acoustic-only estimate; relative flux and complete-step stability still required.'}
    (root/(args.tag+'_group_audit.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
