"""Full contact continuation and affine volume cost of regularized cap motion."""
import gc,json,time
from pathlib import Path
import torch,warp as wp
from free_surface.fv_ale_cfl_diagnose import load
from free_surface.fv_ale_hyperelastic_speed import step
from free_surface.fv_ale_energy_ledger import rate
from free_surface.fv_ale_releasing import ReleaseRequired

def transpose_check(sol,s,eps):
    from free_surface.fv_ale_planar_mechanics import motion
    from free_surface.fv_ale_planar_boundary import values
    m=s.mesh;ops=sol.device_ops()
    b={'q':s.q,'V':m.volume,'groups':sol.groups,'p':s.p,'r':s.r,'i':m.i,'j':m.j,
       'A':m.area,'n':m.normal,'c':m.centroid,'N':m.cap_normal_integral,'Q':m.cap_Q,
       'cap':m.cap_area,'surface':m.surface,'floor':m.ground_area,'floor_c':m.ground_centroid,'floor_n':m.ground_normal}
    b.update(ops.extra_rhs_values(m));b.update(values(m))
    _,_,v,rho,_,_,sigma,sound,_=ops.primitive(s.q,m.volume,sol.groups)
    ops.mechanical_kinematics(b,v,sigma,rho*sound);G=ops._compatible['gradient'].clone()
    with torch.enable_grad():
        vv=v.detach().requires_grad_(True);GG=G.detach().requires_grad_(True)
        part=motion(b,vv,GG,sigma,rho*sound,b['affine_centers'],sol.ground,regularization=eps)
        dv,dG=torch.autograd.grad(part['kinematic_power'],(vv,GG))
        a=float((dv+part['direct']).norm()/dv.norm().clamp(min=1.))
        c=float((dG-part['conjugate']).norm()/dG.norm().clamp(min=1.))
    assert max(a,c)<1e-10,(a,c)
    return {'direct_error':a,'gradient_error':c}

def affine_cost(sol,s,eps):
    m=s.mesh;cap=m.cap_area;N=m.cap_normal_integral;Q=m.cap_Q;i,j,A=m.i,m.j,m.area
    area=cap.clamp(min=0)+m.ground_area+torch.zeros_like(cap).index_add(0,i,A).index_add(0,j,A)
    if hasattr(m,'planes'):area.index_add_(0,m.plane_i,m.plane_area)
    weight=(N.square().sum(1)+cap.square())/(N.square().sum(1)+cap.square()+(eps*area).square()).clamp(min=1e-300)
    Q0=cap[:,None,None]*torch.eye(3,dtype=cap.dtype,device=cap.device)/3
    delta=m.r[:,None,None]*(1-weight[:,None,None])*(Q0-Q)
    roots=torch.unique(sol.groups);V=sol.group_sum(m.volume)[roots]
    relative=(sol.group_sum(delta)[roots]/V[:,None,None]).norm(dim=(1,2))
    mass=sol.group_sum(s.q[:,0])[roots]
    return {'definition':'Operator norm bound for divergence error under unit Frobenius affine velocity gradient, aggregated per actual control volume.',
       'max':float(relative.max()),'p99':float(torch.quantile(relative,.99)),
       'mass_weighted_rms':float(((relative.square()*mass).sum()/mass.sum()).sqrt()),
       'worst_group':int(roots[relative.argmax()])}

@torch.no_grad()
def main():
    torch.set_num_threads(1);wp.init();rows=[];states=[];root=Path('output/ale/cfl_continuity')
    for eps in [0.,.01]:
        sol,s,tol=load(Path('output/ale/hyperelastic_releasing_floorfix100ms.pt'),.1,eps)
        for _ in range(32):
            try:first=sol.rhs(s);break
            except ReleaseRequired as e:s,_=sol.activate(s,e.cells)
        ledger=rate(sol,s,first,fd=True);cost=affine_cost(sol,s,eps);transpose=transpose_check(sol,s,eps)
        source={'work_retry_tolerance_joule':tol,'local_energy_tolerance':1e-6}
        # Warm kernels on a disposable step, never replace the initial state.
        step(sol,s,source,cfl=2.)
        t=0.;history=[];torch.cuda.synchronize();tick=time.perf_counter()
        while t<.001-1e-14:
            s,row=step(sol,s,source,cfl=2.,dt_max=.001-t);history.append(row);t+=row['dt']
        torch.cuda.synchronize();elapsed=time.perf_counter()-tick
        row={'epsilon':eps,'seconds':elapsed,'steps':len(history),'time':t,'retries':sum(x['retry'] for x in history),
             'ledger':ledger,'transpose':transpose,'affine_cost':cost,'energy':sol.energy_audit(s),'history':history}
        states.append({k:getattr(s,k).cpu().clone() for k in ['p','r','q']});rows.append(row)
        print(json.dumps({k:v for k,v in row.items() if k!='history'}),flush=True)
        del sol,s;gc.collect();torch.cuda.empty_cache()
    d=states[1]['p']-states[0]['p']
    result={'runs':rows,'position_rms_m':float(d.square().sum(1).mean().sqrt()),'position_max_m':float(d.norm(dim=1).max())}
    for eps in [.003,.01,.03]:
        sol,s,_=load(Path('output/ale/cfl2_entry/fv_state_cfl05_free5ms.pt'),.01,eps)
        result.setdefault('free_affine_cost',{})[str(eps)]=affine_cost(sol,s,eps)
        del sol,s;gc.collect();torch.cuda.empty_cache()
    (root/'contact_check.json').write_text(json.dumps(result,indent=2)+'\n')
    torch.save(states,root/'contact_check.pt');print(json.dumps({k:v for k,v in result.items() if k!='runs'}),flush=True)

if __name__=='__main__':main()
