"""Replay the full stopped impact through release, using production RK2.

CPU mode exercises the same tensor/geometry kernels with a complete power
hull instead of the GPU recutter. CUDA mode uses the actual integrated solver
and captured operations. Neither substitutes a small scene or a toy model.
"""
import argparse
import json
from pathlib import Path
import time
import torch
import warp as wp
from free_surface.fv_ale_releasing import ReleasingHyperelasticALE,ReleasingDeviceOps,ReleaseRequired
from free_surface.fv_ale_solid import State
from free_surface.fv_ale_hyperelastic_config import StepAudit


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    ap.add_argument('--tag',default=None)
    ap.add_argument('--snapshot',type=Path,default=Path('output/ale/hyperelastic_acoustic_impact50ms.pt'))
    ap.add_argument('--duration',type=float,default=.0001);args=ap.parse_args()
    torch.set_num_threads(1);wp.init();root=Path('output/ale');tick=time.perf_counter()
    data=torch.load(args.snapshot,map_location='cpu',weights_only=False)
    source=json.loads(args.snapshot.with_suffix('.json').read_text());mat=data['material']
    if args.device=='cuda':
        solver=ReleasingHyperelasticALE(data['p'],data['r'],mat['E'],mat['nu'],mat['ys'],groups=data['groups'],
            device='cuda',contact_model='plane',ground=data['ground'],gravity=(0.,0.,0.),friction=0.,
            wall_mode='compatible_slip_acoustic',mechanical_reconstruction='compatible',affine_stencil='smooth',
            affine_cap_motion='balanced',cap_dissipation=0.,interface_dissipation=data['interface_dissipation'],
            transport_reconstruction='entropy',transport_dissipation=data['transport_dissipation'],precision_geometry=True)
        state=State(solver.initial.p,solver.initial.r,data['q'].cuda(),solver.initial.mesh)
        if data.get('moving_planes') is not None:
            from free_surface.fv_ale_planar_boundary import Planes
            state.mesh=solver.cut(state.p,state.r,Planes(**data['moving_planes']).to('cuda'))
    else:
        from free_surface.fv_ale_compatible_mesh_audit import restored
        from free_surface.fv_ale_precise_geometry_check import cpu_mesh
        from free_surface.fv_ale_planar_floor_reuse import geometry_cpu
        original,state,_,_=restored(data,ground=data['ground'])
        solver=object.__new__(ReleasingHyperelasticALE)
        solver.__dict__.update({k:v for k,v in vars(original).items() if k not in ['device_ops','_device_ops']})
        solver.ground=data['ground'];solver.device_step=True;solver.contact_model='plane';solver.geometry_calls=0
        solver.boundary_insertions=[];solver.boundary_prunings=[];ops=ReleasingDeviceOps(solver);solver._device_ops=ops
        solver._precision_geometry_records=[]
        ops.call=lambda name,fn,values,face_keys=():fn(values)
        def cpu_cut(p,r,planes=None):
            mesh=cpu_mesh(p,r,solver.ground) if planes is None else geometry_cpu(p,r,planes,solver.ground)
            solver.normalize_volume_roundoff(mesh)
            if not bool(torch.isfinite(mesh.volume).all()&(mesh.volume>=0).all()):raise RuntimeError('invalid CPU replay volume')
            ops.moments(mesh);return mesh
        solver.cut=cpu_cut;state.mesh=cpu_cut(state.p,state.r,getattr(state.mesh,'planes',None))
    audit=StepAudit(solver,source['work_retry_tolerance_joule'],source['local_energy_tolerance'])
    def ready(s):
        s,_=solver.compact_boundaries(s)
        for _ in range(32):
            try:return s,solver.rhs(s)
            except ReleaseRequired as request:
                s,record=solver.activate(s,request.cells);print(json.dumps({'activation':record}),flush=True)
        raise RuntimeError('activation did not settle')
    rows=[];t=0.;initial=solver.energy_audit(state);initial_mass=float(state.q[:,0].sum())
    while t<args.duration-1e-15:
        state,first=ready(state);dt=min(.5*first[3]['dt_scale'],args.duration-t)
        audit.begin(state,first);activations=0;retries=0
        while True:
            try:
                end,stats=solver.advance(state,dt,first,.5);checked=audit.check(end,dt);break
            except ReleaseRequired as request:
                activations+=1
                if activations>32:raise
                state,record=solver.activate(state,request.cells);print(json.dumps({'activation':record}),flush=True)
                state,first=ready(state);audit.begin(state,first);dt=min(dt,.5*first[3]['dt_scale'])
            except RuntimeError:
                retries+=1
                if retries>8:raise
                dt*=.5
        state=end;t+=dt
        row={'time_since_snapshot':t,'dt':dt,'planes':len(state.mesh.planes.owner) if hasattr(state.mesh,'planes') else 0,
             'contact_area':stats['contact_area'],'audit':checked,'retries':retries}
        rows.append(row);print(json.dumps(row),flush=True)
    # Re-cut a serialized boundary state. Physics and boundary controls must
    # survive replay together; omitting them would incorrectly refill the gap.
    from free_surface.fv_ale_planar_boundary import Planes
    packed=state.mesh.planes.packed();restored_planes=Planes(**packed).to(state.p.device)
    remesh=solver.cut(state.p,state.r,restored_planes)
    volume_error=float((remesh.volume-state.mesh.volume).norm()/state.mesh.volume.norm())
    assert volume_error<1e-10 and (len(solver.boundary_insertions)>0 or data.get('moving_planes') is not None)
    final=solver.energy_audit(state)
    result={'device':args.device,'full_cell_count':len(state.r),'initial_time':data['time'],'duration':t,
            'passed':True,'rows':rows,'initial':initial,'final':final,'audit':audit.report(),
            'mass_relative_error':float(state.q[:,0].sum())/initial_mass-1,
            'serialized_boundary_recut_error':volume_error,'activations':solver.boundary_insertions,
            'prunings':solver.boundary_prunings,
            'seconds':time.perf_counter()-tick}
    tag=args.tag or 'hyperelastic_releasing_step_'+args.device
    (root/(tag+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    torch.save({'p':state.p.cpu(),'r':state.r.cpu(),'q':state.q.cpu(),'groups':solver.groups.cpu(),
                'moving_planes':packed,'time':data['time']+t,'material':mat,'ground':solver.ground},
                root/(tag+'.pt'))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
