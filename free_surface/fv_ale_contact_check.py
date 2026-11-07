"""Contact checks on stored full-foam states plus analytic surface integrals."""
import argparse,json,math,time
from pathlib import Path
from types import SimpleNamespace
import torch,warp as wp
from free_surface.fv_ale_solid import SolidALE
from free_surface.fv_ale_contact import surface_quadrature,contact_flux


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('snapshot',type=Path);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();wp.init();d=torch.load(args.snapshot,weights_only=False,map_location='cpu');g=d['material']
    solver=SolidALE(d['p'],d['r'],g['E'],g['nu'],g['ys'],contact_model='surface',contact_order=8,surface_motion='least-squares')
    state=solver.initial;state.q=d['q'].to(state.q.device)
    m,v,rho,_,_,sigma,c=solver.primitive(state);Z=rho*c
    report={'snapshot':str(args.snapshot),'time':d['time'],'checks':{}}
    # Independent exact single-sphere cap: area 2*pi*r*penetration and
    # integral delta dA = pi*r*penetration^2 (lower hemisphere).
    r=.2;penetration=.03
    mock=SimpleNamespace(p=state.p.new_tensor([[0,r-penetration,0]]),r=state.r.new_tensor([r]),
        patch_A=state.r.new_tensor([4*math.pi*r*r]),wall_A=state.r.new_tensor([1.]),
        i=torch.empty(0,dtype=torch.long,device=state.p.device),
        site_ent=torch.empty(0,dtype=torch.long,device=state.p.device),
        site_off=torch.zeros(2,dtype=torch.long,device=state.p.device),
        nbrs=torch.empty((1,0),dtype=torch.long,device=state.p.device))
    _,_,w,delta=surface_quadrature(mock,0,12)
    report['analytic_cap_area_error']=abs(float(w.sum())/(2*math.pi*r*penetration)-1)
    report['analytic_gap_integral_error']=abs(float((w*delta).sum())/(math.pi*r*penetration**2)-1)
    report['checks']['analytic_integrals']=max(report['analytic_cap_area_error'],report['analytic_gap_integral_error'])<1e-10
    orders=[];forces=[];areas=[]
    for order in [6,8,12,16]:
        tick=time.perf_counter();quad=surface_quadrature(state.mesh.geometry,solver.ground,order)
        owner,n,w,delta=quad
        t,us,fp=contact_flux(*quad,v,sigma,Z,solver.contact_k,solver.friction,solver.contact_damping)
        force=torch.zeros_like(v).index_add(0,owner,w[:,None]*t)
        area=torch.zeros_like(state.r).index_add(0,owner,w)
        torch.cuda.synchronize()
        orders.append({'order':order,'points':len(w),'area':float(w.sum()),'force':force.sum(0).cpu().tolist(),'seconds':time.perf_counter()-tick})
        forces.append(force);areas.append(area)
    denom=forces[-1].norm().clamp(min=1e-30)
    report['quadrature']=orders
    report['force_error_to_order16']=[float((x-forces[-1]).norm()/denom) for x in forces[:-1]]
    report['area_error_to_order16']=[float((x-areas[-1]).norm()/areas[-1].norm().clamp(min=1e-30)) for x in areas[:-1]]
    report['contact_area_over_full_cap_max']=float((areas[-1]/state.mesh.cap_area.clamp(min=1e-300)).max())
    # Verify all quadrature points against ALL balls, not only the stencil used
    # to construct the exposed arc intervals.
    xyz=state.p[owner]+state.r[owner,None]*n
    worst=math.inf
    for start in range(0,len(xyz),256):
        y=xyz[start:start+256]
        pw=((y[:,None,:]-state.p[None,:,:])**2).sum(2)-state.r[None,:]**2
        pw.scatter_(1,owner[start:start+256,None],math.inf)
        worst=min(worst,float(pw.min()))
    report['minimum_power_to_other_balls']=worst if math.isfinite(worst) else None
    report['checks']['actual_exposed_surface']=worst>=-1e-12 and report['contact_area_over_full_cap_max']<1+1e-7
    report['checks']['quadrature_convergence']=report['force_error_to_order16'][1]<1e-3 and report['area_error_to_order16'][1]<1e-4
    report['normal_traction_min']=float(t[:,1].min()) if len(t) else 0.
    report['coulomb_violation']=float((t[:,[0,2]].norm(dim=1)-solver.friction*t[:,1]).clamp(min=0).max()) if len(t) else 0.
    report['friction_power_max']=float(fp.max()) if len(fp) else 0.
    report['checks']['unilateral_and_friction']=report['normal_traction_min']>=0 and report['coulomb_violation']<1e-8 and report['friction_power_max']<1e-8
    # Same-state comparison with the failed inherited whole-ball rule.
    solver.contact_model='legacy-ball';legacy,_,_=solver.contact(state);solver.contact_model='surface'
    ids=d.get('ids',torch.arange(len(m)))
    candidates=torch.nonzero(ids==3502).flatten()
    if len(candidates):
        k=int(candidates[0]);report['original_cell_3502']={'legacy_force':legacy[k].cpu().tolist(),'surface_force':forces[-1][k].cpu().tolist(),
            'exposed_below_ground_area':float(areas[-1][k]),'entire_exposed_area':float(state.mesh.cap_area[k]),'mass':float(m[k])}
    solver.surface_motion='radial'
    _,_,_,radial_diagnostics=solver.rhs(state)
    solver.surface_motion='least-squares'
    meshv,dr,dq,diagnostics=solver.rhs(state)
    force8=forces[1]
    report['force_balance_error']=float((dq[:,1:4].sum(0)-(m[:,None]*solver.g).sum(0)-force8.sum(0)).norm())
    report['power_balance_error']=abs(float(dq[:,4].sum())-diagnostics['source_power'])
    report['boundary_integrated_volume_flux_defect']=diagnostics['cap_integrated_flux_defect']
    report['boundary_motion_power_defect']=diagnostics['contact_boundary_motion_power_defect']
    report['radial_only_boundary_motion_power_defect']=radial_diagnostics['contact_boundary_motion_power_defect']
    report['normal_contact_power']=diagnostics['normal_contact_power']
    report['mesh_material_speed_difference_max']=diagnostics['mesh_material_speed_difference_max']
    A=state.mesh.cap_area;N=state.mesh.cap_normal_integral
    wanted=(-(sigma@N[:,:,None])[:,:,0]+force8)/Z[:,None]
    vector_residual=A[:,None]*(meshv-v)+N*dr[:,None]-wanted
    selected=state.mesh.surface & (1-(N/A.clamp(min=1e-300)[:,None]).square().sum(1)>1e-3)
    report['cap_mean_velocity_fit_residual']=float(vector_residual[selected].norm()/wanted[selected].norm().clamp(min=1e-30))
    report['nearly_flat_caps_using_tangential_fit']=diagnostics['nearly_flat_cap_fits']
    report['checks']['cap_velocity_fit']=report['cap_mean_velocity_fit_residual']<1e-7 and diagnostics['cap_integrated_flux_defect']<1e-12
    report['checks']['assembled_balances']=report['force_balance_error']<1e-7 and report['power_balance_error']<1e-7 and report['boundary_integrated_volume_flux_defect']<1e-12
    report['passed']=all(report['checks'].values());args.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    if not report['passed']:raise SystemExit(2)

if __name__=='__main__':main()
