"""Full-impact frozen work/transpose audit of the new planar boundary path."""
import json
from pathlib import Path
import time
import torch
import warp as wp
from free_surface.fv_ale_planar_boundary import Planes,geometry_cpu,values,geometric_rate
from free_surface.fv_ale_planar_mechanics import motion
from free_surface.fv_ale_compatible_mesh_audit import restored
from free_surface.fv_ale_energy_ledger import mechanical
from free_surface.fv_ale_hyperelastic_material import response,symmetric_deviator


def main():
    torch.set_num_threads(1);wp.init();root=Path('output/ale');tick=time.perf_counter()
    data=torch.load(root/'hyperelastic_acoustic_impact50ms.pt',map_location='cpu',weights_only=False)
    sol,state,ops,_=restored(data,ground=data['ground']);sol.ground=data['ground']
    p,r=state.p,state.r;planes=Planes.floor(p,r,[2919,6333,2979],sol.ground)
    output={'snapshot_time':data['time'],'cells':len(r),'cases':[]}
    for case in ['floor','lift','tilt']:
        pl=planes.clone()
        if case!='floor':pl.b-=r.new_tensor([.0001,.00017,.00008])
        if case=='tilt':
            rotation=r.new_tensor([[.015,0.,-.01],[-.012,0.,.005],[.009,0.,.013]])
            pl.a+=rotation;pl.b+=(rotation*p[pl.owner]).sum(1)
        mesh=geometry_cpu(p,r,pl,sol.ground);state.mesh=mesh
        b={'q':state.q,'V':mesh.volume,'groups':sol.groups,'p':p,'r':r,'i':mesh.i,'j':mesh.j,
           'A':mesh.area,'n':mesh.normal,'c':mesh.centroid,'N':mesh.cap_normal_integral,'cap':mesh.cap_area,
           'floor':mesh.ground_area,'floor_c':mesh.ground_centroid,'floor_n':mesh.ground_normal,**values(mesh)}
        mesh.cap_Q,mesh.surface,mesh.surface_area,ignored=ops.moments_gpu(b)
        b.update(Q=mesh.cap_Q,surface=mesh.surface);b.update(ops.extra_rhs_values(mesh))
        with torch.no_grad():
            w,rd,dq,packed=ops.rhs_gpu(b)
            diagnostics=dict(zip(ops.rhs_keys,packed.tolist()));cache=ops._compatible
            total,V,v,rho,pressure,stress,sigma,sound,valid=ops.primitive(state.q,mesh.volume,sol.groups)
            G=cache['gradient'].clone();centers=b['affine_centers']
        # Check the analytic transpose against ordinary autodiff of the scalar
        # kinematic power. v and G are independent here; the production code
        # subsequently applies the existing affine reconstruction transpose.
        vv=v.detach().requires_grad_(True);GG=G.detach().requires_grad_(True)
        boundary=motion(b,vv,GG,sigma,rho*sound,centers,sol.ground)
        dv,dG=torch.autograd.grad(boundary['kinematic_power'],(vv,GG))
        direct_error=float((dv+boundary['direct']).norm()/dv.norm().clamp(min=1.))
        conjugate_error=float((dG-boundary['conjugate']).norm()/dG.norm().clamp(min=1.))
        dissipation_error=float(abs(boundary['kinematic_power']-boundary['actual_power']-boundary['heat'].sum())/boundary['heat'].sum().abs().clamp(min=1.))
        assert max(direct_error,conjugate_error,dissipation_error)<1e-10,(case,direct_error,conjugate_error,dissipation_error)
        with torch.no_grad():
            boundary=cache['planar'];vdot=mesh.volume_rate(w,rd,plane_rates=(boundary['adot'],boundary['bdot']))
            M=total[:,0];H=total[:,5:14].reshape(-1,3,3)/M[:,None,None]
            mat=response(H,rho,sol.mu,sol.bulk,sol.rho0);elastic,B=mat[3],mat[5]
            T=sol.mu/(2*sol.rho0)*symmetric_deviator(B);chi=torch.zeros_like(total)
            chi[:,0]=elastic+pressure/rho-(T*H).sum((1,2))-.5*v.square().sum(1)
            chi[:,1:4]=v;chi[:,5:14]=T.reshape(-1,9)
            actual=(chi*dq).sum()-(pressure*vdot).sum()
            predicted=-(cache['heat'].sum()+cache['mixing_heat'].sum())
            rate_error=float(abs(actual-predicted)/predicted.abs().clamp(min=1.))
            total_error=float(dq[:,4].sum().abs()/predicted.abs().clamp(min=1.))
            row={'case':case,'direct_transpose_error':direct_error,'affine_transpose_error':conjugate_error,
                 'boundary_dissipation_error':dissipation_error,'mechanical_power_relative_error':rate_error,
                 'total_energy_power_relative_error':total_error,'mechanical_power_W':float(actual),
                 'planar_heat_W':float(boundary['heat'].sum()),'active_planes':boundary['active'].tolist(),
                 'tangential_force_N':dq[:,[1,3]].sum(0).tolist()}
            assert max(rate_error,total_error)<1e-9,row
            if case=='tilt':
                # Rigid change of coordinates, including a translated origin.
                # Plane b is an offset, so its rate also transforms with adot.
                axis=r.new_tensor([.2,.5,-.3]);axis=axis/axis.norm()
                K=r.new_tensor([[0.,-axis[2],axis[1]],[axis[2],0.,-axis[0]],[-axis[1],axis[0],0.]])
                R=torch.eye(3,dtype=r.dtype)+torch.sin(r.new_tensor(.7))*K+(1-torch.cos(r.new_tensor(.7)))*(K@K)
                shift=r.new_tensor([1.2,-.4,.7]);rotated=dict(b)
                for key in ['p','c','plane_c','floor_c']:rotated[key]=b[key]@R.T+shift
                for key in ['n','N','plane_n','floor_n','plane_a']:rotated[key]=b[key]@R.T
                rotated['Q']=R@b['Q']@R.T
                rotated['plane_b']=b['plane_b']+(rotated['plane_a']*shift).sum(1)
                transformed=motion(rotated,v@R.T,R@G@R.T,R@sigma@R.T,rho*sound,centers@R.T+shift,-10.)
                original=motion(b,v,G,sigma,rho*sound,centers,-10.)
                objectivity=max(float((transformed['direct']-original['direct']@R.T).norm()/original['direct'].norm().clamp(min=1.)),
                    float((transformed['conjugate']-R@original['conjugate']@R.T).norm()/original['conjugate'].norm().clamp(min=1.)),
                    float((transformed['bdot']-original['bdot']-(transformed['adot']*shift).sum(1)).abs().max()))
                row['rotated_translated_coordinates_error']=objectivity
                assert objectivity<1e-10,row
                errors=[]
                for h in [1e-7,1e-8]:
                    pp=pl.clone();pm=pl.clone();pp.a+=h*boundary['adot'];pm.a-=h*boundary['adot']
                    pp.b+=h*boundary['bdot'];pm.b-=h*boundary['bdot']
                    gp=geometry_cpu(p+h*w,r+h*rd,pp,sol.ground)
                    gm=geometry_cpu(p-h*w,r-h*rd,pm,sol.ground)
                    diff=(mechanical(sol,state.q+h*dq,gp.volume)-mechanical(sol,state.q-h*dq,gm.volume))/(2*h)
                    errors.append(float(abs(diff-actual)/actual.abs().clamp(min=1.)))
                row['actual_recut_mechanical_rate_errors']=errors
                assert min(errors)<2e-5,row
            output['cases'].append(row);print(json.dumps(row),flush=True)
    output.update(passed=True,seconds=time.perf_counter()-tick)
    (root/'hyperelastic_planar_mechanics_check.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
