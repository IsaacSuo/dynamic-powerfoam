"""Power/adjoint, continuity and actual frozen-step checks of cap regularization."""
import gc,json
from pathlib import Path
import torch,warp as wp
from free_surface.fv_ale_cfl_diagnose import load,details
from free_surface.fv_ale_energy_ledger import rate
from free_surface.fv_ale_solid import State
from free_surface.fv_ale_reconstruction import group_sum
from free_surface.fv_ale_step_control import RememberedStep


def main():
    torch.set_num_threads(1);wp.init();rows=[]
    for eps in [0.,.003,.01,.03]:
      with torch.no_grad():
        sol,s,tol=load(Path('output/ale/cfl2_entry/fv_state_cfl05_free5ms.pt'),.01,eps)
        first=sol.rhs(s);start=details(sol,s,first);ledger=rate(sol,s,first,fd=True)
        m=s.mesh;ops=sol.device_ops();g=sol.groups
        b={'q':s.q,'V':m.volume,'groups':g,'p':s.p,'r':s.r,'i':m.i,'j':m.j,'A':m.area,'n':m.normal,'c':m.centroid,
           'Q':m.cap_Q,'N':m.cap_normal_integral,'cap':m.cap_area,'surface':m.surface,'floor':m.ground_area}
        b.update(ops.extra_rhs_values(m))
        _,_,v,rho,_,_,sigma,sound,_=ops.primitive(s.q,m.volume,g);Z=rho*sound
      rootv=v.detach().clone().requires_grad_(True);vv=rootv[g]
      kin=ops.mechanical_kinematics(b,vv,sigma,Z);cg=kin[3];us=kin[5]
      i,j,A,n=m.i,m.j,m.area,m.normal
      grad=torch.zeros_like(sigma).index_add(0,i,A[:,None,None]*(us-vv[i])[:,:,None]*n[:,None,:])
      grad.index_add_(0,j,-A[:,None,None]*(us-vv[j])[:,:,None]*n[:,None,:])
      # Independent scalar mechanical power differentiated through the primal
      # map; the manually assembled negative-adjoint force is not used here.
      power=(sigma*(grad+cg)).sum()
      damping=.5*ops._compatible['velocity_heat'].sum()
      measured=group_sum(ops._compatible['force'].detach(),g)
      reference=-torch.autograd.grad(power+damping,rootv)[0]
      adjoint=float((measured-reference).norm()/reference.norm())
      assert adjoint<1e-10,adjoint
      row={'epsilon':eps,'adjoint_relative_error':adjoint,'ledger':ledger,'trials':[]}
      with torch.no_grad():
        for h in [1e-7,2e-7,9.66181928397495e-6,38.6472771358998e-6]:
          stages=[];original=sol.rhs
          def capture(x):
            r=original(x);stages.append((x,r));return r
          sol.rhs=capture;trial={'dt_us':h*1e6}
          try:
            end,stats=sol.advance(s,h,first,cfl=1e6);mid,r=stages[-1]
            second=details(sol,mid,r);finish=details(sol,end,None)
            error=finish['energy']-start['energy']-.5*h*(start['power']+second['power'])
            trial.update(work_error_J=float(error.sum()),work_ratio=abs(float(error.sum()))/tol,
                         group7191_work_error_J=float(error[start['roots']==7191][0]),
                         stage_scale_us=r[3]['dt_scale']*1e6,dr4987=float(r[1][4987]),
                         max_local_ratio=float(finish['localratio'].max()))
          except Exception as e:trial['error']=str(e)
          finally:sol.rhs=original
          row['trials'].append(trial)
      rows.append(row);print(json.dumps(row),flush=True)
      Path('output/ale/cap_regularization_check.json').write_text(json.dumps(rows,indent=2)+'\n')
      del sol,s,ops;gc.collect();torch.cuda.empty_cache()
    control=RememberedStep();assert control.propose(4e-5)==4e-5
    control.accept(1e-5,2);assert control.propose(4e-5)==1e-5
    control.accept(1e-5,0);assert control.propose(4e-5)==1.25e-5
    resumed=RememberedStep(control.packed());assert resumed.propose(4e-5)==control.propose(4e-5)
    resumed.accept(1e-8,0,True);assert resumed.packed()==control.packed()
    print('controller reject memory, growth, resume and output-boundary checks passed',flush=True)

if __name__=='__main__':main()
