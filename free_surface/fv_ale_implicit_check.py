"""Full-scene implicit trials, same saved states and same material/geometry."""
import argparse,json,time
from pathlib import Path
import torch
from free_surface.fv_ale_profile import load,physical_difference
from free_surface.fv_ale_implicit import ImplicitStep

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',default='dt8us');ap.add_argument('--dt',type=float,nargs='+',default=[80e-6,200e-6,800e-6])
    args=ap.parse_args();out=Path('output/ale');sol,reset,data=load(out/f'fv_state_{args.tag}.pt')
    sol.ground_cutter.recut.pipe.direct_sphere=False
    records=[]
    for dt in args.dt:
        s=reset();torch.cuda.synchronize();tick=time.perf_counter();calls=sol.geometry_calls
        record={'snapshot':args.tag,'dt':dt,'method':'BE with lagged faces and coupled group volume, geometry defect checked'}
        try:
            end,info=ImplicitStep(sol,s,dt).solve()
            torch.cuda.synchronize();info['total_seconds']=time.perf_counter()-tick
            info['recuts']=sol.geometry_calls-calls;record['implicit']=info
            record['status']='passed' if info['geometry_accepted'] else 'geometry_defect_rejected'
        except Exception as exc:
            record.update(status='solver_failed',error=repr(exc),total_seconds=time.perf_counter()-tick)
            records.append(record);print(json.dumps(record),flush=True);continue
        ref=reset();time_left=dt;steps=0;rejects=0;torch.cuda.synchronize();tick=time.perf_counter()
        try:
            while time_left>1e-14:
                first=sol.rhs(ref);h=min(time_left,2*first[3]['dt_scale'])
                for retry in range(9):
                    try:ref,stats=sol.advance(ref,h,first,2.);break
                    except RuntimeError:
                        rejects+=1;h*=.5
                        if retry==8:raise
                steps+=1;time_left-=h
            torch.cuda.synchronize()
            record['explicit']={'seconds':time.perf_counter()-tick,'steps':steps,'rejects':rejects}
            record['difference']=physical_difference(sol,ref,end)
            mass=sol.primitive(ref)[0];vref=sol.primitive(ref)[1];vend=sol.primitive(end)[1]
            record['difference']['velocity_mass_rms']=float(((mass*((vend-vref)**2).sum(1)).sum()/mass.sum()).sqrt())
        except Exception as exc:record['explicit_error']=repr(exc)
        records.append(record);print(json.dumps(record),flush=True)
    (out/f'implicit_trial_{args.tag}.json').write_text(json.dumps(records,indent=2)+'\n')

if __name__=='__main__':main()
