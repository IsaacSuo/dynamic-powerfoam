"""Exact per-group energy accounting from a complete saved RK2 step (CPU).

No solver update or geometry repair. Separates the nonlinear RK2 chain-rule
defect, actual re-cut endpoint volume, and the stored total-energy transport.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from free_surface.fv_ale_hyperelastic_work_replay import restored_state,volume_rate,group_energy,group_power
from free_surface.fv_ale_hyperelastic_material import response


def main():
    parser=argparse.ArgumentParser();parser.add_argument('snapshot');args=parser.parse_args()
    torch.set_num_threads(1);path=Path(args.snapshot);data=torch.load(path,map_location='cpu',weights_only=False)
    g=data['groups'];mat=data['material'];dt=data['dt']
    sol=SimpleNamespace(groups=g,mu=mat['E']/(2*(1+mat['nu'])),bulk=mat['E']/(3*(1-2*mat['nu'])),rho0=1000.)
    sol.group_sum=lambda value:torch.zeros_like(value).index_add(0,g,value)
    s0,s1,s2=[restored_state(data[name],'cpu') for name in ['before','stage','after']]
    rhs0,rhs1=data['first_rhs'],data['stage_rhs'];roots=torch.unique(g)
    e0=group_energy(sol,s0.q,s0.mesh.volume)[1];e2=group_energy(sol,s2.q,s2.mesh.volume)[1]
    raw=s0.q+.5*dt*(rhs0[2]+rhs1[2])
    Vrk=s0.mesh.volume+.5*dt*(volume_rate(s0,rhs0)+volume_rate(s1,rhs1))
    assert bool((sol.group_sum(Vrk)[roots]>0).all()),'RK-integrated group volume is nonpositive'
    erk=group_energy(sol,raw,Vrk)[1];egeom=group_energy(sol,raw,s2.mesh.volume)[1]
    p0,p1=group_power(sol,s0,rhs0),group_power(sol,s1,rhs1)
    total0=sol.group_sum(s0.q)[roots];total1=sol.group_sum(s1.q)[roots];total2=sol.group_sum(s2.q)[roots]
    r0=total0[:,4]-e0;r2=total2[:,4]-e2
    rdot0=sol.group_sum(rhs0[2])[roots,4]-p0;rdot1=sol.group_sum(rhs1[2])[roots,4]-p1
    predicted=r0+.5*dt*(rdot0+rdot1)
    geometry=egeom-erk;time=erk-e0-.5*dt*(p0+p1);projection=e2-egeom
    identity=r2-(predicted-geometry-time-projection)
    assert float(identity.abs().max())<1e-10,identity.abs().max()
    v=total2[:,1:4]/total2[:,0,None];elastic=e2-.5*total2[:,0]*v.square().sum(1)
    ratio=(-r2).clamp(min=0)/(elastic+total2[:,0]*sol.mu/sol.rho0)
    ids=ratio.topk(min(12,len(roots))).indices
    rows=[]
    for k in ids.tolist():
        root=int(roots[k]);members=g==root
        def state_values(s,total):
            volume=float(sol.group_sum(s.mesh.volume)[root]);mass=float(total[k,0])
            energy=float(group_energy(sol,s.q,s.mesh.volume)[1][k])
            return {'mass_kg':mass,'volume_m3':volume,'J':sol.rho0*volume/mass,
                    'mechanical_joule':energy,'remainder_joule':float(total[k,4])-energy,
                    'velocity_m_s':(total[k,1:4]/total[k,0]).tolist()}
        rows.append({'group':root,'members':torch.nonzero(members).flatten().tolist(),
            'negative_over_elastic_plus_mu_mass_over_rho0':float(ratio[k]),
            'before':state_values(s0,total0),'stage':state_values(s1,total1),'after':state_values(s2,total2),
            'predicted_remainder_joule':float(predicted[k]),
            'remainder_rate_before_W':float(rdot0[k]),'remainder_rate_stage_W':float(rdot1[k]),
            'mechanical_geometry_endpoint_defect_joule':float(geometry[k]),
            'mechanical_RK2_chain_defect_joule':float(time[k]),
            'post_update_projection_defect_joule':float(projection[k]),
            'actual_minus_RK_volume_m3':float(sol.group_sum(s2.mesh.volume-Vrk)[root])})
    result={'snapshot':str(path),'time':data['time'],'dt':dt,'step':data['step'],
            'identity_max_absolute_error_J':float(identity.abs().max()),'top_local_deficit_groups':rows,
            'scope':'Exact bookkeeping of one saved step, not a causal counterfactual or a convergence proof.'}
    output=path.with_name(path.stem+'_local_account.json');output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
