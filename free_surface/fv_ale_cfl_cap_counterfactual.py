"""Diagnostic only: suppress one cap branch, never used as a model fix."""
from pathlib import Path
import gc,json
import torch,warp as wp
from free_surface.fv_ale_cfl_diagnose import load,details

@torch.no_grad()
def main():
    torch.set_num_threads(1);wp.init();rows=[]
    for suppress in [False,True]:
      sol,s,tol=load(Path('output/ale/cfl2_entry/fv_state_cfl05_free5ms.pt'),.01)
      if suppress:
        ops=sol.device_ops();old=ops.mechanical_kinematics;keep=torch.arange(len(s.r),device=s.r.device)!=4987
        def kin(b,*args):
            b=dict(b);b['surface']=b['surface']&keep
            return old(b,*args)
        ops.mechanical_kinematics=kin
      first=sol.rhs(s);start=details(sol,s,first)
      for h in [2e-7,9.66181928397495e-6,38.6472771358998e-6]:
        stages=[];original=sol.rhs
        def capture(x):
            r=original(x);stages.append((x,r));return r
        sol.rhs=capture;row={'suppress_4987_cap_branch':suppress,'dt_us':h*1e6}
        try:
            end,stats=sol.advance(s,h,first,cfl=1e6);mid,r=stages[-1]
            second=details(sol,mid,r);finish=details(sol,end,None)
            error=finish['energy']-start['energy']-.5*h*(start['power']+second['power'])
            row.update(work_error_J=float(error.sum()),group7191_error_J=float(error[start['roots']==7191][0]),
                negative_remainder_J=float((-finish['remainder']).clamp(min=0).sum()),
                max_local_ratio=float(finish['localratio'].max()),stage_scale_us=r[3]['dt_scale']*1e6,
                stage_site_speed=float(r[0][4987].norm()),stage_radius_speed=float(r[1][4987]))
        except Exception as ex:row['error']=str(ex)
        finally:sol.rhs=original
        rows.append(row);print(json.dumps(row),flush=True)
      del sol,s;gc.collect();torch.cuda.empty_cache()
    Path('output/ale/cfl_cap_counterfactual.json').write_text(json.dumps(rows,indent=2)+'\n')

if __name__=='__main__':main()
