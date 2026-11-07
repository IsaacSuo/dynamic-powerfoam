"""Independent balance checks for coupled sphere motion/contact on a full foam."""
import argparse,json,time
from pathlib import Path
import torch,warp as wp
from free_surface.fv_ale_solid import SolidALE
from free_surface.fv_ale_contact import surface_quadrature,reference_surface_quadrature
from free_surface.fv_ale_boundary import solve_boundary,surface_traction


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('snapshot',type=Path);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    wp.init();d=torch.load(args.snapshot,weights_only=False,map_location='cpu');g=d['material']
    sol=SolidALE(d['p'],d['r'],g['E'],g['nu'],g['ys'],surface_motion='coupled',groups=d.get('groups'))
    if 'reference_volume' in d:sol.contact_length=d['reference_volume']**(1/3);sol.contact_k=sol.bulk/sol.contact_length
    s=sol.initial;s.q=d['q'].to(s.q.device);mesh=s.mesh
    m,v,rho,_,_,sigma,c=sol.primitive(s);Z=rho*c
    timings={};quads={}
    for name,fn in [('reference',reference_surface_quadrature),('batched',surface_quadrature)]:
        fn(mesh.geometry,sol.ground,8)
        torch.cuda.synchronize();tick=time.perf_counter()
        for _ in range(3):quad=fn(mesh.geometry,sol.ground,8)
        torch.cuda.synchronize();timings[name]=(time.perf_counter()-tick)/3;quads[name]=quad
    qa,qb=quads['reference'],quads['batched']
    same_owner=bool(torch.equal(qa[0],qb[0]))
    quadrature_difference=[float((x-y).abs().max()) if x.numel() else 0. for x,y in zip(qa[1:],qb[1:])] if same_owner else None
    reports=[];forces=[]
    for order in [8,12,16]:
        mesh.contact_quad=surface_quadrature(mesh.geometry,sol.ground,order)
        w,dr,grad,rate,b,info=solve_boundary(mesh,v,sigma,Z,sol.contact_k,sol.friction,sol.contact_damping)
        owner,n,weight,delta=mesh.contact_quad
        actual=w[owner]+dr[owner,None]*n
        t,P,drag=surface_traction(actual,Z[owner],delta,sol.contact_k,sol.friction,sol.contact_damping)
        F=torch.zeros_like(v).index_add(0,owner,weight[:,None]*t)
        Tn=torch.zeros_like(s.r).index_add(0,owner,weight*(t*n).sum(1))
        A=mesh.cap_area;N=mesh.cap_normal_integral;Q=mesh.cap_Q
        Rw=Z[:,None]*(A[:,None]*(w-v)+N*dr[:,None])+(sigma@N[:,:,None])[:,:,0]-F
        Rr=Z*((N*(w-v)).sum(1)+A*dr)+(sigma*Q).sum((1,2))-Tn
        var=1-(N/A.clamp(min=1e-300)[:,None]).square().sum(1)
        active=mesh.surface;radial=active&(var>1e-3)
        scale=(Z*A*(v.norm(dim=1)+1e-3)+F.norm(dim=1)+(sigma@N[:,:,None])[:,:,0].norm(dim=1)).clamp(min=1e-100)
        force_res=float((Rw.norm(dim=1)[active]/scale[active]).max())
        radial_res=float((Rr.abs()[radial]/scale[radial]).max()) if bool(radial.any()) else 0.
        boundary_power=float((weight*(t*actual).sum(1)).sum())
        heat=float((-weight*(t[:,[0,2]]*actual[:,[0,2]]).sum(1)).sum())
        work_error=abs(float(b[1].sum())-boundary_power-heat)
        point_equality=float((actual-info['point_velocity']).abs().max()) if len(owner) else 0.
        reports.append({'order':order,'iterations':info['iterations'],'solver_residual':info['relative_residual'],
          'independent_force_residual':force_res,'independent_radial_residual':radial_res,
          'traction_work_plus_heat_error':work_error,'point_velocity_difference':point_equality,
          'force':F.sum(0).cpu().tolist(),'normal_work_rate':float(b[1].sum()),'friction_heat_rate':heat})
        forces.append(F)
    mesh.contact_quad=surface_quadrature(mesh.geometry,sol.ground,8)
    rhs=sol.rhs(s);dt=.1*rhs[3]['dt_scale'];end,stats=sol.advance(s,dt,rhs)
    total_energy_error=abs(float((end.q[:,4]-s.q[:,4]).sum())-stats['source_work'])
    force_errors=[float((f-forces[-1]).norm()/forces[-1].norm().clamp(min=1e-100)) for f in forces[:-1]]
    result={'snapshot':str(args.snapshot),'time':d['time'],'orders':reports,'force_error_to_order16':force_errors,
      'quadrature_seconds':timings,'batched_same_point_owners':same_owner,'batched_max_difference_normal_weight_gap':quadrature_difference,
      'step_dt':dt,'step_mass_error':float((end.q[:,0].sum()-s.q[:,0].sum())/s.q[:,0].sum()),
      'step_energy_source_error_joule':total_energy_error,'boundary_motion_work_defect':stats['boundary_work_defect'],
      'regularization':'Coulomb friction regularized with viscous impedance; not exact static sticking',
      'checks':{'batched_quadrature_equivalence':same_owner and max(quadrature_difference)<1e-12,
                'local_equations':max(max(x['independent_force_residual'],x['independent_radial_residual']) for x in reports)<1e-7,
                'same_boundary_velocity':all(x['point_velocity_difference']<1e-14 for x in reports),
                'work_heat_balance':all(x['traction_work_plus_heat_error']<1e-8 for x in reports),
                'quadrature':force_errors[0]<2e-3,
                'accepted_step_balance':total_energy_error<1e-8 and abs(float(end.q[:,0].sum()-s.q[:,0].sum()))<1e-10}}
    result['passed']=all(result['checks'].values());args.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    if not result['passed']:raise SystemExit(2)

if __name__=='__main__':main()
