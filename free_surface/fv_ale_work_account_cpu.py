"""Read a saved complete step and separate its work error without using GPU."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from free_surface.fv_ale_hyperelastic_work_replay import restored_state,volume_rate,group_energy,group_power
from free_surface.fv_ale_reconstruction import group_sum,group_centroids
from free_surface.fv_ale_compatible_mechanics import affine_map
from free_surface.fv_ale_affine_stencil import build,gradient,negative_adjoint,compact


def main():
    torch.set_num_threads(1)
    parser=argparse.ArgumentParser();parser.add_argument('snapshot')
    parser.add_argument('--stencil',choices=['extended','smooth'],default=None);args=parser.parse_args()
    path=Path(args.snapshot);data=torch.load(path,map_location='cpu',weights_only=False)
    mat=data['material'];g=data['groups'];dt=data['dt']
    sol=SimpleNamespace(groups=g,mu=mat['E']/(2*(1+mat['nu'])),bulk=mat['E']/(3*(1-2*mat['nu'])),rho0=1000.)
    sol.group_sum=lambda x:group_sum(x,g)
    states=[restored_state(data[k],'cpu') for k in ['before','stage','after']]
    s0,s1,s2=states;f0,f1=data['first_rhs'],data['stage_rhs']
    Vrk=s0.mesh.volume+.5*dt*(volume_rate(s0,f0)+volume_rate(s1,f1))
    roots,e0=group_energy(sol,s0.q,s0.mesh.volume);_,e2=group_energy(sol,s2.q,s2.mesh.volume)
    _,erk=group_energy(sol,s2.q,Vrk)
    work=.5*dt*(group_power(sol,s0,f0)+group_power(sol,s1,f1))
    errors=e2-e0-work;geo=e2-erk
    result={'time':data['time'],'dt':dt,'saved_error_J':data['error'],
            'independent_CPU_error_J':float(errors.sum()),'geometry_endpoint_J':float(geo.sum()),
            'RK_chain_J':float((erk-e0-work).sum()),'configuration':data['args'],'top_groups':[]}
    diagnostics=[];stencil=args.stencil or data['args'].get('affine_stencil','extended')
    result['compact_storage_checks']=[]
    result['gradient_diagnostic_stencil']=stencil
    for s in states:
        m=s.mesh
        b={'q':s.q,'p':s.p,'r':s.r,'V':m.volume,'groups':g,'i':m.i,'j':m.j,'A':m.area,
           'n':m.normal,'c':m.centroid,'N':m.cap_normal_integral,'floor':m.ground_area,
           'floor_c':m.ground_centroid,'floor_n':m.ground_normal}
        extra,summary=build(b,transition_ratio=.01 if stencil=='smooth' else None);b.update(extra)
        total=group_sum(s.q,g);V=group_sum(m.volume,g);mass=total[:,0]
        v=total[:,1:4]/mass.clamp(min=1e-300)[:,None]
        G=gradient(b,v[g]);_,inverse,_,_=affine_map(b,v[g],group_centroids(b))
        edges,capacity,count=compact(b);cb=dict(b,**edges)
        generator=torch.Generator().manual_seed(192)
        conjugate=torch.randn((len(g),3,3),dtype=G.dtype,generator=generator)
        cg=gradient(cb,v[g]);f=negative_adjoint(b,conjugate);cf=negative_adjoint(cb,conjugate)
        check={'gradient_relative_error':float((cg-G).norm()/G.norm().clamp(min=1.)),
               'transpose_relative_error':float((cf-f).norm()/f.norm().clamp(min=1.)),
               'directed_supports':count,'capacity':capacity}
        assert max(check['gradient_relative_error'],check['transpose_relative_error'])<1e-12,check
        result['compact_storage_checks'].append(check)
        eigen=torch.linalg.eigvalsh(inverse)
        b['one_ring_ratio']=eigen[:,0]/eigen[:,-1].clamp(min=1e-300)
        diagnostics.append((total,V,v,G,b,summary))
    for k in errors.abs().topk(8).indices:
        root=int(roots[k]);row={'group':root,'work_error_J':float(errors[k]),'geometry_J':float(geo[k]),'stages':[]}
        for total,V,v,G,b,summary in diagnostics:
            ids=b['affine_neighbors'][root];keep=b['affine_weights'][root]>0
            row['stages'].append({'mass_kg':float(total[root,0]),'volume_m3':float(V[root]),
                'J':float(sol.rho0*V[root]/total[root,0]),'velocity_m_s':v[root].tolist(),
                'gradient_norm_per_s':float(G[root].norm()),'neighbors':ids[keep].tolist()})
            row['stages'][-1]['one_ring_normal_eigenvalue_ratio']=float(b['one_ring_ratio'][root])
        result['top_groups'].append(row)
    result['stencil_summaries']=[x[-1] for x in diagnostics]
    suffix='_'+args.stencil if args.stencil else ''
    output=path.with_name(path.stem+'_cpu_account'+suffix+'.json');output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['configuration','top_groups']},indent=2))
    print(json.dumps(result['top_groups'][:2],indent=2))


if __name__=='__main__':main()
