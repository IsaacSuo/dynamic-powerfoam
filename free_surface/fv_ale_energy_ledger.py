"""Mechanical-energy rate identity for the current first-order hyperelastic FV.

Independent diagnostic: does not alter fluxes or repair energies. It separates
acoustic interface dissipation, free-cap projection, donor mixing, wall work
and gravity, and checks their sum against the energy chain rule and finite
differences. Grouping is handled by repeated group entropy variables.
"""
import json
from pathlib import Path
import torch
from free_surface.fv_ale_hyperelastic_material import response,symmetric_deviator
from free_surface.fv_ale_hyperelastic import HyperelasticALE
from free_surface.fv_ale_solid import State


def mechanical(solver,q,V):
    total=solver.group_sum(q)[solver.groups]
    volume=solver.group_sum(V)[solver.groups]
    mass=total[:,0];v=total[:,1:4]/mass[:,None]
    H=total[:,5:14].reshape(-1,3,3)/mass[:,None,None]
    energy=response(H,mass/volume,solver.mu,solver.bulk,solver.rho0)[3]
    return (q[:,0]*(energy+.5*v.square().sum(1))).sum()


@torch.no_grad()
def rate(solver,state,first,fd=False):
    mesh=state.mesh;q=state.q;g=solver.groups
    total=solver.group_sum(q)[g];Vg=solver.group_sum(mesh.volume)[g]
    M=total[:,0];v=total[:,1:4]/M[:,None];rho=M/Vg
    H=total[:,5:14].reshape(-1,3,3)/M[:,None,None]
    pressure,shear,sigma,elastic,c2,B,J=response(H,rho,solver.mu,solver.bulk,solver.rho0,getattr(solver,'acoustic_bound','conservative'))
    T=solver.mu/(2*solver.rho0)*symmetric_deviator(B)
    chi=torch.zeros_like(q)
    chi[:,0]=elastic+pressure/rho-(T*H).sum((1,2))-.5*v.square().sum(1)
    chi[:,1:4]=v;chi[:,5:14]=T.reshape(-1,9)
    w,dr,dq,_=first
    i,j,A,n,c=mesh.i,mesh.j,mesh.area,mesh.normal,mesh.centroid
    Z=rho*c2.sqrt()
    ti=(sigma[i]@n[:,:,None])[:,:,0];tj=(sigma[j]@n[:,:,None])[:,:,0]
    b={'q':q,'V':mesh.volume,'groups':g,'p':state.p,'r':state.r,
       'i':i,'j':j,'A':A,'n':n,'c':c,'N':mesh.cap_normal_integral,'Q':mesh.cap_Q,
       'cap':mesh.cap_area,'surface':mesh.surface,
       'floor':mesh.ground_area,'floor_c':mesh.ground_centroid,'floor_n':mesh.ground_normal}
    if solver.mechanical_reconstruction=='compatible' and solver.affine_stencil!='one_ring':
        b.update(solver.device_ops().extra_rhs_values(mesh))
    kin=solver.device_ops().mechanical_kinematics(b,v,sigma,Z)
    cap_grad,us,ts=kin[3],kin[5],kin[6]
    plane_rates=None
    if hasattr(mesh,'planes'):
        saved=first[3]['plane_rates']
        plane_rates=(q.new_tensor(saved['a']),q.new_tensor(saved['b']))
    mw,mc=mesh.mesh_flux(w,dr,plane_rates=plane_rates)
    Vdot=mesh.incidence(mw)+mc
    relative=A*(us*n).sum(1)-mw
    donor=solver.device_ops().advective_donor(b,total,Vg,relative)
    internal=A*((ti*(us-v[i])).sum(1)-(tj*(us-v[j])).sum(1)+(ts*(v[i]-v[j])).sum(1))
    internal_prediction=-solver.interface_dissipation*A*((tj-ti).square().sum(1)+Z[i]*Z[j]*(v[j]-v[i]).square().sum(1))/(Z[i]+Z[j])
    work=solver.device_ops().interface_energy_flux(us,ts,v,ti,tj,i,j)
    heat_i=A*(work-(v[i]*ts).sum(1)-(ti*(us-v[i])).sum(1))
    heat_j=A*(-work+(v[j]*ts).sum(1)+(tj*(us-v[j])).sum(1))
    donor_rate=-relative*(((chi[i]-chi[j])*donor).sum(1)-(pressure[i]-pressure[j]))
    delta=w-v;N=mesh.cap_normal_integral;Q=mesh.cap_Q;cap=mesh.cap_area
    cap_rate=(sigma*cap_grad).sum((1,2))
    cap_prediction=-Z/max(solver.cap_dissipation,1e-300)*(cap*delta.square().sum(1)+2*dr*(delta*N).sum(1)+dr.square()*cap)
    compatible=solver.mechanical_reconstruction=='compatible'
    if compatible:
        cache=solver.device_ops()._compatible
        internal=-cache['stress_heat']-cache['velocity_heat']
        cap_rate=-cache['cap_heat']
        if cache['planar'] is not None:cap_rate-=cache['planar']['heat']
        internal_prediction=internal;cap_prediction=cap_rate
        heat_i=heat_j=.5*(-internal)
    floor=mesh.ground_area
    if compatible:floor=cache['floor']
    free=v+sigma[:,:,1]/Z[:,None];P=(-Z*free[:,1]).clamp(min=0)*(floor>0)
    tangent=free[:,[0,2]];speed=tangent.norm(dim=1);amount=torch.minimum(solver.friction*P,Z*speed)
    traction=torch.stack((-amount*tangent[:,0]/speed.clamp(min=1e-300),P,-amount*tangent[:,1]/speed.clamp(min=1e-300)),1)
    uv=free+traction/Z[:,None];uv[:,1]=0
    wall_acoustic=-floor*Z*(uv-v).square().sum(1)
    wall_friction=floor*(traction*uv).sum(1)
    if compatible and getattr(solver,'wall_mode','unsupported')=='compatible_slip':
        # A stationary frictionless plane does no work. Its reaction is in
        # the compatible force; there is no extra acoustic wall dissipation.
        wall_acoustic=torch.zeros_like(floor)
        wall_friction=torch.zeros_like(floor)
    if compatible and getattr(solver,'wall_mode','unsupported')=='compatible_slip_acoustic':
        wall_acoustic=-floor*Z*cache['wall_normal_velocity'].square()
        wall_friction=torch.zeros_like(floor)
    gravity=(q[:,1:4]*solver.g).sum()
    rhs_derivative=(chi*dq).sum()-(pressure*Vdot).sum()
    summed=internal.sum()+donor_rate.sum()+cap_rate.sum()+wall_acoustic.sum()+wall_friction.sum()+gravity
    error=rhs_derivative-summed
    scale=max(float(rhs_derivative.abs()),float(summed.abs()),1.)
    output={'internal_acoustic_W':float(internal.sum()),'donor_mixing_W':float(donor_rate.sum()),
            'free_cap_W':float(cap_rate.sum()),'wall_acoustic_W':float(wall_acoustic.sum()),
            'wall_friction_W':float(wall_friction.sum()),'gravity_W':float(gravity),
            'mechanical_rate_chain_W':float(rhs_derivative),'mechanical_rate_split_W':float(summed),
            'split_relative_error':float(error)/scale,
            'cap_prediction_relative_error':float((cap_rate-cap_prediction).norm()/cap_rate.norm().clamp(min=1.)) if solver.mechanical_reconstruction=='constant' or compatible else None,
            'internal_prediction_relative_error':float((internal-internal_prediction).norm()/internal.norm().clamp(min=1.)) if solver.mechanical_reconstruction=='constant' or compatible else None,
            'negative_interface_heating_W':float(heat_i.clamp(max=0).sum()+heat_j.clamp(max=0).sum()),
            'positive_donor_rate_sum_W':float(donor_rate.clamp(min=0).sum()),
            'mechanical_energy_joule':float(mechanical(solver,q,mesh.volume))}
    if fd:
        values=[]
        for eps in [1e-8,1e-9,1e-10]:
            finite=(mechanical(solver,q+eps*dq,mesh.volume+eps*Vdot)-mechanical(solver,q-eps*dq,mesh.volume-eps*Vdot))/(2*eps)
            values.append({'epsilon_seconds':eps,'mechanical_rate_W':float(finite),'relative_error':float(finite-rhs_derivative)/scale})
        output['finite_differences']=values
        assert min(abs(r['relative_error']) for r in values)<1e-5,output
    assert abs(output['split_relative_error'])<1e-8,output
    return output


@torch.no_grad()
def main():
    root=Path('output/ale');results={}
    for name in ['initial','hyperelastic_release_grouped','hyperelastic_release_roundoff']:
        path=root/('fv_state_dt8us.pt' if name=='initial' else name+'.pt')
        data=torch.load(path,map_location='cpu',weights_only=False);m=data['material']
        solver=HyperelasticALE(data['p'],data['r'],m['E'],m['nu'],m['ys'],groups=data['groups'],
                               contact_model='plane',ground=-10.,gravity=(0.,0.,0.))
        state=solver.initial
        if name=='initial':state=solver.initialize_strain(state,torch.diag(torch.tensor([1.5,1.5**-.5,1.5**-.5],device='cuda',dtype=torch.float64)))
        else:state=State(state.p,state.r,data['q'].to(state.p.device),state.mesh)
        results[name]=rate(solver,state,solver.rhs(state),fd=True)
        print(name,json.dumps(results[name]),flush=True)
    (root/'hyperelastic_energy_frozen_ledger.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':main()
