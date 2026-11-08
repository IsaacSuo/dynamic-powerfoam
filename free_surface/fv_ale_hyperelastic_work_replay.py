"""Separate cut-volume error from RK2 error on a saved complete bad step."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from free_surface.fv_ale_geometry import PowerControlVolumes
from free_surface.fv_ale_solid import State
from free_surface.fv_ale_hyperelastic import HyperelasticALE
from free_surface.fv_ale_hyperelastic_material import response,symmetric_deviator
from free_surface.fv_ale_energy_ledger import rate,mechanical


def restored_state(record,device):
    p,r,q=(record[k].to(device) for k in ['p','r','q'])
    m={k:v.to(device) for k,v in record['mesh'].items()}
    mesh=PowerControlVolumes(p,r,m['volume'],m['i'],m['j'],m['area'],m['normal'],m['centroid'],m['cap_area'],m['cap_normal_integral'])
    for k,v in m.items():setattr(mesh,k,v)
    mesh.ignored_cap_area=0.;mesh.ground_geometry_info={'method':'saved_exact_step_geometry'}
    return State(p,r,q,mesh)


def volume_rate(s,rhs):
    wall,cap=s.mesh.mesh_flux(rhs[0],rhs[1])
    return s.mesh.incidence(wall)+cap


def group_energy(sol,q,V):
    g=sol.groups;roots=torch.unique(g)
    total=sol.group_sum(q)[roots];volume=sol.group_sum(V)[roots]
    mass=total[:,0];v=total[:,1:4]/mass[:,None]
    H=total[:,5:14].reshape(-1,3,3)/mass[:,None,None]
    e=response(H,mass/volume,sol.mu,sol.bulk,sol.rho0)[3]
    return roots,mass*(e+.5*v.square().sum(1))


def group_power(sol,s,rhs):
    roots=torch.unique(sol.groups);total=sol.group_sum(s.q)[roots]
    vol=sol.group_sum(s.mesh.volume)[roots];mass=total[:,0];v=total[:,1:4]/mass[:,None]
    H=total[:,5:14].reshape(-1,3,3)/mass[:,None,None]
    p,_,_,e,_,B,_=response(H,mass/vol,sol.mu,sol.bulk,sol.rho0)
    T=sol.mu/(2*sol.rho0)*symmetric_deviator(B);chi=torch.zeros_like(total)
    chi[:,0]=e+p*vol/mass-(T*H).sum((1,2))-.5*v.square().sum(1)
    chi[:,1:4]=v;chi[:,5:14]=T.reshape(-1,9)
    return (chi*sol.group_sum(rhs[2])[roots]).sum(1)-p*sol.group_sum(volume_rate(s,rhs))[roots]


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('snapshot')
    parser.add_argument('--stencil',choices=['one_ring','extended','smooth'],default=None)
    parser.add_argument('--cap-motion',choices=['isotropic','volume','balanced'],default=None);args=parser.parse_args()
    path=Path(args.snapshot);data=torch.load(path,map_location='cpu',weights_only=False)
    cfg=data['args'];mat=data['material'];before=data['before']
    stencil=args.stencil or cfg.get('affine_stencil','one_ring')
    cap_motion=args.cap_motion or cfg.get('affine_cap_motion','isotropic')
    sol=HyperelasticALE(before['p'],before['r'],mat['E'],mat['nu'],mat['ys'],groups=data['groups'],
        contact_model='plane',ground=-10.,gravity=(0.,0.,0.),
        mechanical_reconstruction=cfg['mechanical_reconstruction'],transport_reconstruction=cfg['transport_reconstruction'],
        interface_dissipation=cfg['interface_dissipation'],cap_dissipation=cfg['cap_dissipation'],
        transport_dissipation=cfg.get('transport_dissipation',.1),
        affine_stencil=stencil,affine_cap_motion=cap_motion,acoustic_bound=cfg.get('acoustic_bound','conservative'),
        precision_geometry=cfg.get('precision_geometry',False))
    states=[restored_state(data[k],'cuda') for k in ['before','stage','after']]
    s0,s1,s2=states;dt=data['dt']
    rhs0=[x.cuda() if isinstance(x,torch.Tensor) else x for x in data['first_rhs']]
    rhs1=[x.cuda() if isinstance(x,torch.Tensor) else x for x in data['stage_rhs']]
    Vdot0=volume_rate(s0,rhs0);Vdot1=volume_rate(s1,rhs1)
    Vrk=s0.mesh.volume+.5*dt*(Vdot0+Vdot1)
    roots,e0=group_energy(sol,s0.q,s0.mesh.volume)
    _,e2=group_energy(sol,s2.q,s2.mesh.volume);_,erk=group_energy(sol,s2.q,Vrk)
    geometry=e2-erk
    qraw=s0.q+.5*dt*(rhs0[2]+rhs1[2]);_,eraw=group_energy(sol,qraw,s2.mesh.volume)
    after_cut=sol.cut(s2.p,s2.r)
    direct=sol.ground_cutter.recut.pipe.direct_sphere
    sol.ground_cutter.recut.pipe.direct_sphere=True
    alternative_cut=sol.cut(s2.p,s2.r)
    sol.ground_cutter.recut.pipe.direct_sphere=direct
    difference=sol.group_sum(s2.mesh.volume-Vrk)[roots]
    ids=geometry.abs().topk(min(15,len(roots))).indices
    result={'time':data['time'],'step':data['step'],'dt':dt,
        'saved_work_error_J':data['error'],'geometry_endpoint_contribution_J':float(geometry.sum()),
        'RK2_chain_contribution_J':float((erk-e0).sum())-data['expected_mechanical_change'],
        'post_update_projection_change_J':float((e2-eraw).sum()),
        'minimum_RK_volume_group_m3':float(sol.group_sum(Vrk)[roots].min()),
        'repeat_cut_max_group_volume_error_m3':float(sol.group_sum(after_cut.volume-s2.mesh.volume)[roots].abs().max()),
        'alternative_formula_max_group_volume_error_m3':float(sol.group_sum(alternative_cut.volume-s2.mesh.volume)[roots].abs().max()),
        'top_geometry_groups':[{'group':int(roots[k]),'energy_J':float(geometry[k]),'volume_error_m3':float(difference[k]),
            'saved_volume_m3':float(sol.group_sum(s2.mesh.volume)[roots[k]]),
            'repeat_volume_m3':float(sol.group_sum(after_cut.volume)[roots[k]]),
            'alternative_volume_m3':float(sol.group_sum(alternative_cut.volume)[roots[k]])} for k in ids]}
    errors=e2-e0-.5*dt*(group_power(sol,s0,rhs0)+group_power(sol,s1,rhs1))
    largest=errors.abs().topk(10).indices
    from free_surface.fv_ale_reconstruction import group_centroids
    from free_surface.fv_ale_compatible_mechanics import affine_map
    mesh=s0.mesh
    b={'q':s0.q,'p':s0.p,'r':s0.r,'V':mesh.volume,'groups':sol.groups,'i':mesh.i,'j':mesh.j,
        'A':mesh.area,'n':mesh.normal,'c':mesh.centroid,'N':mesh.cap_normal_integral,
        'floor':mesh.ground_area,'floor_c':mesh.ground_centroid,'floor_n':mesh.ground_normal}
    centers=group_centroids(b);total=sol.group_sum(s0.q)[sol.groups]
    v=total[:,1:4]/total[:,0,None];_,inv,delta,weights=affine_map(b,v,centers)
    ev=torch.linalg.eigvalsh(inv)
    result['top_work_error_groups']=[]
    for k in largest:
        root=int(roots[k]);members=sol.groups==root
        edge=(sol.groups[mesh.i]==root)|(sol.groups[mesh.j]==root)
        neighbors=torch.unique(torch.cat((sol.groups[mesh.i[edge]],sol.groups[mesh.j[edge]])))
        neighbors=neighbors[neighbors!=root]
        row={'group':root,'error_J':float(errors[k]),'geometry_J':float(geometry[k]),
            'mass_before_kg':float(s0.q[members,0].sum()),'mass_after_kg':float(s2.q[members,0].sum()),
            'member_count':int(members.sum()),'neighbor_count':len(neighbors),
            'centroid_nearest_neighbor_m':float((centers[neighbors]-centers[root]).norm(dim=1).min()),
            'LS_normal_eigenvalue_ratio':float(ev[root,0]/ev[root,-1].clamp(min=1e-300)),
            'velocity_before':v[root].tolist(),
            'velocity_after':(s2.q[members,1:4].sum(0)/s2.q[members,0].sum()).tolist(),
            'acceleration_like_m_s2':((sol.group_sum(rhs0[2])[root,1:4]-v[root]*sol.group_sum(rhs0[2])[root,0])/total[root,0]).tolist()}
        result['top_work_error_groups'].append(row)
    # Independently integrate the exact face-area volume derivative along the
    # straight geometric segment. This is separate from the RK2 time path.
    dp=s2.p-s0.p;dr=s2.r-s0.r;line=[]
    for order in [5,10]:
        nodes,weights=np.polynomial.legendre.leggauss(order);integral=torch.zeros_like(s0.r)
        for node,weight in zip(.5*(nodes+1),.5*weights):
            mesh=sol.cut(s0.p+float(node)*dp,s0.r+float(node)*dr)
            fw,fc=mesh.mesh_flux(dp,dr);integral+=float(weight)*(mesh.incidence(fw)+fc)
        defect=sol.group_sum(s2.mesh.volume-s0.mesh.volume-integral)[roots]
        line.append({'quadrature_order':order,'max_group_volume_defect_m3':float(defect.abs().max()),
                     'sum_abs_group_volume_defect_m3':float(defect.abs().sum()),
                     'worst_group':int(roots[defect.abs().argmax()])})
    result['straight_geometry_path']=line
    # Keep the accepted state and first RHS exactly fixed; only change dt.
    native=sol.rhs;captured={}
    def record_stage(s):
        rhs=native(s);captured['state']=s;captured['rhs']=rhs;return rhs
    sol.rhs=record_stage
    changed=stencil!=cfg.get('affine_stencil','one_ring') or cap_motion!=cfg.get('affine_cap_motion','isotropic')
    replay_first=native(s0) if changed else rhs0
    result['replay_stencil']=stencil;result['replay_cap_motion']=cap_motion
    result['saved_trajectory_configuration']=cfg
    if stencil!='one_ring' and changed:result['stencil_summary']=sol.device_ops().stencil_summary
    first_rate=rate(sol,s0,replay_first)['mechanical_rate_chain_W'];replays=[]
    for fraction in [1.,.5,.25,.125,.0625,.03125,.015625]:
        try:
            out,_=sol.advance(s0,dt*fraction,replay_first,cfg['cfl'])
            stage_rate=rate(sol,captured['state'],captured['rhs'])['mechanical_rate_chain_W']
            expected=.5*dt*fraction*(first_rate+stage_rate)
            error=float(mechanical(sol,out.q,out.mesh.volume)-mechanical(sol,s0.q,s0.mesh.volume))-expected
            replays.append({'dt_fraction':fraction,'work_error_J':error,'audit':sol.energy_audit(out),
                'repeat_saved_position_max_error_m':float((out.p-s2.p).abs().max()) if fraction==1 else None})
        except RuntimeError as exc:replays.append({'dt_fraction':fraction,'error':str(exc)})
    result['replays']=replays
    suffix=('_'+stencil if stencil!='one_ring' else '')+('_'+cap_motion if cap_motion!='isotropic' else '')+'_analysis.json'
    out=path.with_name(path.stem+suffix);out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
