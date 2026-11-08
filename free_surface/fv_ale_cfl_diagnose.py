"""Frozen full-state CFL diagnostics; no trial is committed to a trajectory.

The large CFL passed to advance only permits inspecting otherwise rejected
stage/end states. Actual .5/2 acceptance predicates are reported separately.
"""
import gc
import json
from pathlib import Path
import torch
import warp as wp
from free_surface.fv_ale_hyperelastic import HyperelasticALE
from free_surface.fv_ale_releasing import ReleasingHyperelasticALE,ReleaseRequired
from free_surface.fv_ale_planar_boundary import Planes
from free_surface.fv_ale_solid import State
from free_surface.fv_ale_hyperelastic_material import response,symmetric_deviator
from free_surface.fv_ale_hyperelastic_work_replay import group_energy


def load(path,alpha,cap_regularization=0.):
    d=torch.load(path,map_location='cpu',weights_only=False);m=d['material']
    main='hyperelastic_config' in d
    src=json.loads((path.parent/('fv_solid_'+path.stem.removeprefix('fv_state_')+'.json') if main else path.with_suffix('.json')).read_text())
    planar=d.get('moving_planes') is not None
    sol=(ReleasingHyperelasticALE if planar else HyperelasticALE)(d['p'],d['r'],m['E'],m['nu'],m['ys'],groups=d['groups'],
        contact_model='plane',ground=d['ground'],gravity=(0.,0.,0.),friction=0.,
        wall_mode='compatible_slip_acoustic' if planar else 'unsupported',
        mechanical_reconstruction='compatible',affine_stencil='smooth',affine_cap_motion='balanced',cap_motion_regularization=cap_regularization,
        cap_dissipation=0.,interface_dissipation=alpha,transport_reconstruction='entropy',transport_dissipation=.1,precision_geometry=True)
    p,r,q=(d[k].cuda() for k in ['p','r','q'])
    planes=Planes(**d['moving_planes']).to('cuda') if planar else None
    s=State(p,r,q,sol.cut(p,r,planes) if planar else sol.cut(p,r))
    tol=src['hyperelastic_step_audit']['tolerance_joule'] if main else src['work_retry_tolerance_joule']
    return sol,s,tol


def volumes(s,rhs):
    kwargs={}
    if hasattr(s.mesh,'planes'):
        rates=rhs[3]['plane_rates'];kwargs['plane_rates']=(s.q.new_tensor(rates['a']),s.q.new_tensor(rates['b']))
    wall,cap=s.mesh.mesh_flux(rhs[0],rhs[1],**kwargs)
    return s.mesh.incidence(wall)+cap


def details(sol,s,rhs):
    g=sol.groups;roots=torch.unique(g);T=sol.group_sum(s.q)[roots];V=sol.group_sum(s.mesh.volume)[roots]
    m=T[:,0];v=T[:,1:4]/m[:,None];H=T[:,5:14].reshape(-1,3,3)/m[:,None,None]
    p,_,_,e,c2,B,J=response(H,m/V,sol.mu,sol.bulk,sol.rho0)
    stressgrad=sol.mu/(2*sol.rho0)*symmetric_deviator(B);chi=torch.zeros_like(T)
    chi[:,0]=e+p*V/m-(stressgrad*H).sum((1,2))-.5*v.square().sum(1)
    chi[:,1:4]=v;chi[:,5:14]=stressgrad.reshape(-1,9)
    Vdot=sol.group_sum(volumes(s,rhs))[roots] if rhs is not None else torch.zeros_like(V)
    power=(chi*sol.group_sum(rhs[2])[roots]).sum(1)-p*Vdot if rhs is not None else torch.zeros_like(V)
    energy=m*(e+.5*v.square().sum(1));remainder=T[:,4]-energy
    return dict(roots=roots,V=V,m=m,J=J,energy=energy,power=power,remainder=remainder,
                localratio=(-remainder).clamp(min=0)/(m*(e+sol.mu/sol.rho0)),Vdot=Vdot,c=c2.sqrt())


@torch.no_grad()
def main():
    torch.set_num_threads(1);wp.init();rows=[]
    paths=[Path('output/ale/cfl2_entry/fv_state_cfl05_free5ms.pt'),Path('output/ale/hyperelastic_releasing_floorfix100ms.pt')]
    out=Path('output/ale/cfl_frozen_diagnosis.json')
    for path in paths:
      for alpha in [.01,.1]:
        sol,s,tol=load(path,alpha)
        for _ in range(32):
            try:first=sol.rhs(s);break
            except ReleaseRequired as ex:s,_=sol.activate(s,ex.cells)
        # Pre-activate the same required vacuum faces for all frozen trials.
        # Insertion is geometrically checked by the production activate method.
        if hasattr(sol,'activate'):
            for _ in range(32):
                try:sol.advance(s,2*first[3]['dt_scale'],first,cfl=1e6);break
                except ReleaseRequired as ex:s,_=sol.activate(s,ex.cells);first=sol.rhs(s)
                except RuntimeError:break
        start=details(sol,s,first);base=first[3]['dt_scale']
        for factor in ([0.,1e-5,.001,.01,.1,.25,.5,1.,2.] if 'cfl05' in path.stem else [.25,.5,1.,2.]):
            h=base*factor;stages=[];original=sol.rhs
            def capture(state):
                result=original(state);stages.append((state,result));return result
            sol.rhs=capture
            row={'snapshot':str(path),'alpha':alpha,'factor':factor,'dt_us':h*1e6,'start_dt_scale_us':base*1e6,
                 'start_wave_dt_us':first[3]['wave_dt_scale']*1e6,'work_tolerance_J':tol,
                 'start_elastic_J':sol.energy_audit(s)['elastic_energy_joule'],
                 'start_remainder_J':float(start['remainder'].sum()),'groups':len(start['roots'])}
            try:
                # Diagnostic only: bypass CFL gates, NEVER submit the candidate.
                end,stats=sol.advance(s,h,first,cfl=1e6)
                stage,second=stages[-1];mid=details(sol,stage,second)
                finish=details(sol,end,None)
                error=finish['energy']-start['energy']-.5*h*(start['power']+mid['power'])
                vrk=s.mesh.volume+.5*h*(volumes(s,first)+volumes(stage,second))
                if bool((sol.group_sum(vrk)[start['roots']]>0).all()):
                    _,erk=group_energy(sol,end.q,vrk)
                    row['geometry_work_error_J']=float((finish['energy']-erk).sum())
                    row['RK_chain_error_J']=float((erk-start['energy']-.5*h*(start['power']+mid['power'])).sum())
                err=float(error.sum());neg=float((-finish['remainder']).clamp(min=0).sum());loc=float(finish['localratio'].max())
                top=error.abs().topk(6).indices
                row.update(work_error_J=err,negative_remainder_J=neg,max_local_ratio=loc,
                    work_ratio=abs(err)/tol,energy_pass=abs(err)<=tol and neg<=tol and loc<=1e-6,
                    top_work_groups=[{'group':int(start['roots'][k]),'error_J':float(error[k]),'initial_J':float(start['J'][k]),
                        'volume_ratio_to_median':float(start['V'][k]/start['V'].median()),
                        'stage_volume_ratio':float(mid['V'][k]/start['V'][k]),'end_volume_ratio':float(finish['V'][k]/start['V'][k])} for k in top])
            except Exception as ex:row['error']=str(ex)
            finally:sol.rhs=original
            if stages:
                stage,second=stages[0];d=second[3];k=d['wave_cell'];root=int(sol.groups[k]);mask=start['roots']==root
                row.update(stage_dt_scale_us=d['dt_scale']*1e6,stage_wave_dt_us=d['wave_dt_scale']*1e6,
                    stage_q_relative_change=float((stage.q-s.q).norm()/s.q.norm()),
                    stage_wave_cell=k,stage_limiting_group=root,
                    stage_limiting_group_volume_ratio=float(sol.group_sum(stage.mesh.volume)[root]/sol.group_sum(s.mesh.volume)[root]),
                    stage_limiting_group_start_relative_volume=float(start['V'][mask][0]/start['V'].median()),
                    stage_CFL_pass_at_own_factor=h<=factor*d['dt_scale']*1.25,
                    stage_CFL_pass_at_2=h<=2*d['dt_scale']*1.25,
                    stage_CFL_pass_at_half=h<=.5*d['dt_scale']*1.25,
                    radius_dt_us=float((stage.r/second[1].abs().clamp(min=1e-300)).min())*1e6)
            rows.append(row);out.write_text(json.dumps(rows,indent=2)+'\n')
            print(json.dumps({k:v for k,v in row.items() if k!='top_work_groups'}),flush=True)
        del sol,s;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
