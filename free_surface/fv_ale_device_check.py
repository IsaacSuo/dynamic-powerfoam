"""Real-state cache, retry ownership, diagnostics and rejection checks."""
import argparse,copy,json
from pathlib import Path
import torch,warp as wp
from free_surface.fv_ale_profile import load,physical_difference
from free_surface.fv_ale_solid import State
from free_surface.fv_ale_agglomerate import merge_cell


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('snapshot',type=Path);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();wp.init();sol,reset,data=load(args.snapshot);s=reset();report={'snapshot':str(args.snapshot),'checks':{}}
    def compare(state):
        sol.device_step=False;a=sol.rhs(state)
        sol.device_step=True;b=sol.rhs(state)
        err=[float((x-y).norm()/x.norm().clamp(min=1e-100)) for x,y in zip(a[:3],b[:3])]
        diag={}
        for key,value in a[3].items():
            if key=='ground_geometry':continue
            if isinstance(value,(int,float)):
                diag[key]=abs(value-b[3][key])/max(1.,abs(value))
            elif isinstance(value,list):
                diag[key]=max(abs(x-y)/max(1.,abs(x)) for x,y in zip(value,b[3][key]))
        if max(err)>1e-8 or max(diag.values())>1e-8:raise AssertionError((err,diag))
        return {'rhs_relative_errors':err,'max_scaled_diagnostic_error':max(diag.values()),'faces':len(state.mesh.i)}
    report['initial']=compare(s)
    # Exceed a face-buffer bucket with zero-area faces. These carry no flux;
    # using a real face's distinct endpoints keeps the reference arithmetic valid.
    mesh=copy.copy(s.mesh)
    for field in ['i','j','area','normal','centroid']:
        x=getattr(mesh,field);extra=x[:1].expand((1024,)+x.shape[1:]).clone()
        if field=='area':extra.zero_()
        setattr(mesh,field,torch.cat((x,extra)))
    padded=State(s.p,s.r,s.q,mesh)
    report['grown_capacity']=compare(padded);report['shrunk_capacity']=compare(s)
    # Existing graphs must consume new group labels after an actual FV merge.
    counts=torch.bincount(sol.groups,minlength=len(s.r));candidates=(counts[sol.groups]==1)&(s.mesh.volume>0)
    k=int(torch.where(candidates,s.mesh.volume,torch.full_like(s.r,float('inf'))).argmin())
    merged,info=merge_cell(sol,s,k);report['merge']=info;report['after_merge']=compare(merged)
    sol.device_step=True;first=sol.rhs(merged);saved=[x.clone() for x in first[:3]]
    dt=.1*first[3]['dt_scale'];sol.advance(merged,dt,first)
    after,_=sol.advance(merged,dt/2,first)
    if not all(torch.equal(a,b) for a,b in zip(saved,first[:3])):raise AssertionError('saved retry RHS overwritten')
    sol.device_step=False;reference,rs=sol.advance(merged,dt/2,sol.rhs(merged))
    report['retry_comparison']=physical_difference(sol,reference,after)
    report['checks']['retry_owned_buffers']=report['retry_comparison']['max_velocity_difference_m_s']<1e-9
    tests={}
    bad=merged.q.clone();bad[:,0]=-bad[:,0].abs();tests['negative_mass']=State(merged.p,merged.r,bad,merged.mesh)
    bad=merged.q.clone();bad[0,4]=float('nan');tests['nonfinite_state']=State(merged.p,merged.r,bad,merged.mesh)
    up=sol.set_velocity(State(s.p,s.r,sol.initial.q.clone(),s.mesh),[0.,1.,0.]);tests['tensile_contact']=up
    errors={}
    for name,state in tests.items():
        errors[name]=[]
        for mode in [False,True]:
            sol.device_step=mode
            try:sol.rhs(state)
            except RuntimeError as e:errors[name].append(str(e).split(':')[0])
            else:raise AssertionError(name+' accepted invalid state')
        if errors[name][0]!=errors[name][1]:raise AssertionError(errors[name])
    report['rejections']=errors;report['checks']['same_rejections']=True
    report['checks']['cache_and_group_changes']=True
    report['graph_builds']=sol.device_ops().builds
    report['graph_build_seconds']=sum(x['seconds'] for x in report['graph_builds'])
    report['passed']=all(report['checks'].values());args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if not report['passed']:raise SystemExit(2)


if __name__=='__main__':main()
