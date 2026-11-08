"""Frozen full-foam audit of interior-only ALE motion with compatible forces.

Calls production tensor operators on CPU or through a captured GPU graph.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import torch
import warp as wp
from free_surface.fv_ale_hyperelastic import HyperelasticDeviceOps,HyperelasticALE
from free_surface.fv_ale_precise_geometry_check import cpu_mesh
from free_surface.fv_ale_solid import State
from free_surface.fv_ale_energy_ledger import rate


def restored(data,device='cpu',ground=None):
    mat=data['material'];g=data['groups']
    if data.get('moving_planes') is not None:
        from free_surface.fv_ale_planar_boundary import Planes
        from free_surface.fv_ale_planar_floor_reuse import geometry_cpu
        if ground is None:ground=data['ground']
        mesh=geometry_cpu(data['p'],data['r'],Planes(**data['moving_planes']),ground)
    else:mesh=cpu_mesh(data['p'],data['r'],ground)
    sol=SimpleNamespace(groups=g,has_groups=True,mu=mat['E']/(2*(1+mat['nu'])),
        bulk=mat['E']/(3*(1-2*mat['nu'])),rho0=1000.,friction=0.,g=torch.zeros(3,dtype=torch.float64),
        acoustic_bound=data.get('acoustic_bound','conservative'),
        interface_dissipation=data['interface_dissipation'],cap_dissipation=data['cap_dissipation'],
        mechanical_reconstruction='compatible',affine_stencil='smooth',affine_cap_motion='balanced',
        transport_reconstruction='entropy',transport_dissipation=data['transport_dissipation'],
        mesh_fit_iterations=0,mesh_fit_radius=False,mesh_fit_regularization=.1)
    sol.wall_mode=data.get('wall_mode','unsupported')
    sol.ground=ground
    sol.group_sum=lambda x:torch.zeros_like(x).index_add(0,g,x)
    HyperelasticALE.normalize_volume_roundoff(sol,mesh)
    assert bool((mesh.volume>=0).all())
    if ground is None:
        mesh.ground_centroid=mesh.p.clone();mesh.ground_normal=torch.zeros_like(mesh.p)
    if device=='cuda':
        for key,value in vars(mesh).items():
            if isinstance(value,torch.Tensor):setattr(mesh,key,value.cuda())
        if hasattr(mesh,'planes'):mesh.planes=mesh.planes.to('cuda')
        g=g.cuda();sol.groups=g;sol.g=sol.g.cuda()
    state=State(mesh.p,mesh.r,data['q'].to(device),mesh);ops=HyperelasticDeviceOps(sol)
    sol._device_ops=ops;sol.device_ops=lambda:sol._device_ops
    b={'q':state.q,'V':mesh.volume,'groups':g,'p':state.p,'r':state.r,'i':mesh.i,'j':mesh.j,
       'A':mesh.area,'n':mesh.normal,'c':mesh.centroid,'N':mesh.cap_normal_integral,'cap':mesh.cap_area,
       'floor':mesh.ground_area,'floor_c':mesh.ground_centroid,'floor_n':mesh.ground_normal}
    if hasattr(mesh,'planes'):
        from free_surface.fv_ale_planar_boundary import values as plane_values
        b.update(plane_values(mesh))
    mesh.cap_Q,mesh.surface,mesh.surface_area,ignored=ops.moments_gpu(b)
    b.update(Q=mesh.cap_Q,surface=mesh.surface);b.update(ops.extra_rhs_values(mesh))
    return sol,state,ops,b


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('tag')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu');args=parser.parse_args()
    torch.set_num_threads(1);wp.init();root=Path('output/ale')
    data=torch.load(root/(args.tag+'.pt'),map_location='cpu',weights_only=False)
    sol,state,ops,b=restored(data,args.device);mesh=state.mesh
    rows=[];base=None
    for iterations,radius in [(0,False),(12,False),(24,False),(24,True)]:
        sol.mesh_fit_iterations=iterations;sol.mesh_fit_radius=radius
        ops=HyperelasticDeviceOps(sol)
        sol._device_ops=ops
        if args.device=='cuda':
            w,dr,dq,packed=ops.call('rhs',ops.rhs_gpu,b,('i','j','A','n','c'))
        else:w,dr,dq,packed=ops.rhs_gpu(b)
        diagnostics=dict(zip(ops.rhs_keys,packed.tolist()));rhs=(w,dr,dq,diagnostics)
        force=ops._compatible['force'].clone();gradient=ops._compatible['gradient'].clone()
        if base is None:base=(w.clone(),dr.clone(),force.clone(),gradient.clone())
        boundary=mesh.surface|(mesh.ground_area>0)
        change=max(float((w-base[0])[boundary].abs().max()),float((dr-base[1])[boundary].abs().max()))
        force_error=float((force-base[2]).norm()/base[2].norm().clamp(min=1.))
        gradient_error=float((gradient-base[3]).norm()/base[3].norm().clamp(min=1.))
        account=rate(sol,state,rhs,fd=True)
        assert change<1e-12 and force_error<1e-12 and gradient_error<1e-12
        rows.append({'iterations':iterations,'allow_radius':radius,'boundary_velocity_change':change,
            'elastic_force_relative_change':force_error,'reconstruction_relative_change':gradient_error,
            'energy_rate_sum_W':diagnostics['energy_rate_sum'],'ledger':account})
    result={'tag':args.tag,'device':args.device,'rows':rows,'scope':'Frozen full-state identity audit only; no dynamics or cost claim.'}
    (root/(args.tag+'_mesh_audit'+('_cuda' if args.device=='cuda' else '')+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
