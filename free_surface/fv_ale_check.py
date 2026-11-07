"""Full-foam ALE geometry and conservative remap trial, not a landing solver.
Invoked through merged_drop.py --fv-ale-remap-check, reusing its foam, scene
transform and material parameters. Pure remeshing has zero physical motion.
The tests deliberately change neighbours while keeping the outer union fixed.
"""
import json
import math
import time
import numpy as np
import torch
import warp as wp
from free_surface.recut import Recut
from free_surface.geo_pipe import GeoPipe
from free_surface.fv_ale_geometry import PowerControlVolumes


def quantiles(a):
    return {str(q): float(torch.quantile(a.double(),q)) for q in (0.,.5,.99,1.)}


@torch.no_grad()
def run(positions,radii,args):
    wp.init()
    p=torch.as_tensor(positions,dtype=torch.float64,device='cuda').clone()
    r=torch.as_tensor(radii,dtype=torch.float64,device='cuda').clone()
    n=len(p); recut=Recut(n); recut.pipe=GeoPipe(n)
    def cut(x,rr):
        g,_=recut.geometry(x,rr)
        return PowerControlVolumes.from_geometry(g)
    t0=time.perf_counter(); mesh=cut(p,r)
    if not bool(torch.isfinite(mesh.volume).all() and (mesh.volume>0).all()):
        raise RuntimeError('nonpositive initial power cell')
    v0=mesh.volume.clone(); h=float(v0.median())**(1/3)
    report={'scope':'geometry + pure remap only; solid dynamics NOT implemented',
            'foam':str(args.foam),'scene':args.scene,'cells':n,
            'wall_count':len(mesh.i),'median_cell_length_m':h}
    print('FV-ALE full foam: %d cells, %d walls, h %.6g m'%(n,len(mesh.i),h),flush=True)
    area=mesh.incidence(mesh.area[:,None]*mesh.normal)+mesh.cap_normal_integral
    report['surface_closure_relative_max']=float(area.abs().max()/mesh.area.max())
    # Generic mesh translation + radial change, checked against rebuilt geometry.
    phase=(p-p.mean(0))/h
    dv=.02*torch.stack((torch.sin(phase[:,1]),torch.cos(phase[:,2]),torch.sin(phase[:,0])),1)
    dr=.01*r*torch.sin(phase[:,0]+phase[:,1])
    expected=mesh.volume_rate(dv,dr)
    report['dV_finite_difference']=[]
    for eps in (1e-4,3e-5,1e-5):
        plus=cut(p+eps*dv,r+eps*dr); minus=cut(p-eps*dv,r-eps*dr)
        observed=(plus.volume-minus.volume)/(2*eps)
        report['dV_finite_difference'].append({'epsilon':eps,
            'relative_l2':float((observed-expected).norm()/expected.norm()),
            'max_error_per_cell_volume':float(((observed-expected)/v0).abs().max())})
    # Translation is an exact rigid-motion case for finite changes as well.
    shift=torch.tensor([.023,-.017,.011],dtype=p.dtype,device=p.device)
    translated=cut(p+shift,r)
    w,c=mesh.mesh_flux(shift.expand_as(p),torch.zeros_like(r))
    physical=mesh.area*(mesh.normal*shift).sum(1)
    report['translation_volume_relative_max']=float(((translated.volume-v0)/v0).abs().max())
    report['translation_relative_wall_flux_max']=float((w-physical).abs().max())
    # Independent cap second moment from the existing geometry implementation.
    from free_surface.gpu_geometry import Geometry
    g,S=recut.geometry(p,r)
    full=Geometry(p,r,S)
    H=torch.tensor([[.2,.4,-.1],[.1,-.3,.15],[.05,-.2,.1]],dtype=p.dtype,device=p.device)
    b=torch.tensor([.12,-.05,.03],dtype=p.dtype,device=p.device)
    fw,fc=mesh.affine_material_flux(H,b)
    direct=((b+p@H.T)*full.patch_n).sum(1)+r*(full.patch_nn*H[None]).sum((1,2))
    report['cap_affine_flux_independent_relative_l2']=float((fc-direct).norm()/direct.norm())
    report['affine_divergence_relative_max']=float(((mesh.incidence(fw)+fc-v0*torch.trace(H))/v0).abs().max())
    del full,g,S
    # A moving free exterior is NOT silently sent through the closed-domain remap.
    try:
        mesh.remap_rate(v0[:,None],torch.zeros_like(p),.01*r)
    except ValueError as exc:
        report['moving_exterior_rejected']=True
        report['moving_exterior_message']=str(exc)
    else:
        report['moving_exterior_rejected']=False
    # Isotropic material dilation is exactly representable by spheres.
    w_iso,c_iso=mesh.mesh_flux(p-p.mean(0),r)
    f_iso,b_iso=mesh.affine_material_flux(torch.eye(3,dtype=p.dtype,device=p.device),-p.mean(0))
    report['isotropic_dilation_relative_flux_max']=float(torch.cat((w_iso-f_iso,c_iso-b_iso)).abs().max())
    # A remap moves mesh only: the material and the free outer boundary stay put.
    # Move only deep interior generators; validate zero exposed swept flux at ALL cuts.
    surf=mesh.cap_area>mesh.cap_area.max()*1e-10
    surface_points=p[surf]
    nearest=torch.cat([torch.cdist(q,surface_points).min(1).values for q in p.split(256)])
    deep=(nearest>r+r.max()+.001) & ~surf
    print("rezone candidates",int(deep.sum()),"surface",int(surf.sum()),"radii",float(r.min()),float(r.max()),"depth",float(nearest.max()),flush=True)
    if int(deep.sum())<10: raise RuntimeError('not enough buried generators for a fixed-boundary rezone')
    displacement=.04*h*torch.stack((torch.sin(phase[:,1]*.7),torch.cos(phase[:,2]*.6),torch.sin(phase[:,0]*.8)),1)
    displacement*=deep[:,None]
    zero=torch.zeros_like(r)
    for shorten in range(12):
        middle=cut(p+.5*displacement,r); target=cut(p+displacement,r)
        if bool((middle.volume>0).all() and (target.volume>0).all()): break
        displacement*=.5
    else: raise RuntimeError('could not find positive-cell rezone')
    report['rezone_max_site_displacement_m']=float(displacement.norm(dim=1).max())
    report['rezone_amplitude_halvings']=shorten
    oldwalls=set(zip(mesh.i.cpu().tolist(),mesh.j.cpu().tolist()))
    newwalls=set(zip(target.i.cpu().tolist(),target.j.cpu().tolist()))
    report['remap_moving_sites']=int(deep.sum())
    report['remap_walls_removed']=len(oldwalls-newwalls)
    report['remap_walls_added']=len(newwalls-oldwalls)
    report['remap_domain_volume_change_relative']=float((target.volume.sum()-v0.sum())/v0.sum())
    report['without_remap_density_error']=quantiles((v0/target.volume-1).abs())
    # density, momentum, total energy, mass-weighted bounded passive tracer.
    tracer=.5+.2*torch.sin(9*p[:,0])*torch.cos(7*p[:,2])
    density=torch.stack((torch.ones_like(r),torch.full_like(r,.2),torch.full_like(r,-.1),
                         torch.full_like(r,.3),torch.full_like(r,2.),tracer),1)*1000
    q0=v0[:,None]*density
    totals=q0.sum(0); report['remap_refinement']=[]
    cfl=min(m.remap_cfl(displacement,zero) for m in (mesh,cut(p+.5*displacement,r),target))
    minimum=max(8,math.ceil(1/(.35*cfl)))
    for count in (minimum,2*minimum,4*minimum):
        dt=1/count; q=q0.clone(); current=mesh
        begin=time.perf_counter(); capmax=0.
        for k in range(count):
            if dt>.8*current.remap_cfl(displacement,zero):
                raise RuntimeError('remap CFL exceeded; refuse unsafe step')
            d0=current.remap_rate(q,displacement,zero)
            end=cut(p+(k+1)*dt*displacement,r)
            if dt>.8*end.remap_cfl(displacement,zero):
                raise RuntimeError('end-stage remap CFL exceeded')
            _,cf=end.mesh_flux(displacement,zero)
            capmax=max(capmax,float(cf.abs().sum()))
            # SSP RK2 on extensive quantities. Geometry is always the exact recut,
            # not an independently accumulated volume or a post-normalized mass.
            trial=q+dt*d0
            d1=end.remap_rate(trial,displacement,zero)
            q=.5*q+.5*(trial+dt*d1)
            if not bool(torch.isfinite(q).all() and (q[:,0]>0).all()):
                raise RuntimeError('invalid transported mass')
            current=end
        rel=(q.sum(0)-totals).abs()/totals.abs().clamp(min=1)
        constant=(q[:,0]/current.volume/1000-1).abs()
        tr=q[:,5]/q[:,0]
        result={'substeps':count,'seconds':time.perf_counter()-begin,
                'conservation_relative_by_channel':rel.cpu().tolist(),
                'uniform_density_error_max':float(constant.max()),
                'uniform_density_error_l2':float(constant.norm()/math.sqrt(n)),
                'tracer_range':[float(tr.min()),float(tr.max())],
                'tracer_initial_range':[float(tracer.min()),float(tracer.max())],
                'exposed_cap_swept_flux_l1_max':capmax}
        report['remap_refinement'].append(result)
        print('REMAP '+json.dumps(result),flush=True)
    report['seconds']=time.perf_counter()-t0
    checks={'moving_exterior_guard':report['moving_exterior_rejected'],
            'closure':report['surface_closure_relative_max']<1e-10,
            'rigid_translation':report['translation_volume_relative_max']<1e-7,
            'geometric_derivative':report['dV_finite_difference'][-1]['relative_l2']<1e-4,
            'cap_moment':report['cap_affine_flux_independent_relative_l2']<1e-8,
            'changed_neighbours':report['remap_walls_added']>0 and report['remap_walls_removed']>0,
            'fixed_domain':abs(report['remap_domain_volume_change_relative'])<1e-10,
            'conservation':all(max(t['conservation_relative_by_channel'])<1e-10 for t in report['remap_refinement']),
            'uniform_density_converges':report['remap_refinement'][-1]['uniform_density_error_max']<report['remap_refinement'][0]['uniform_density_error_max']/5,
            'positive_bounded_tracer':all(t['tracer_range'][0]>=float(tracer.min())-1e-8 and t['tracer_range'][1]<=float(tracer.max())+1e-8 for t in report['remap_refinement'])}
    report['checks']=checks; report['passed']=all(checks.values())
    path=args.save/('fv_ale_'+args.tag+'.json'); path.write_text(json.dumps(report,indent=2)+'\n')
    print('FV-ALE '+('PASS' if report['passed'] else 'FAIL')+' '+str(path)+' '+json.dumps(checks),flush=True)
    return report
