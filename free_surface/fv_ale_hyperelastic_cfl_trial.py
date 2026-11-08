"""Equal physical duration CFL comparison using complete saved production states.

Groups remain fixed to isolate time stepping; this is not a new merge policy.
Material, moving boundary, geometry and energy acceptance are unchanged.
"""
import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import time
import torch
import warp as wp
from free_surface.fv_ale_hyperelastic_speed import load, step


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('snapshot',type=Path)
    ap.add_argument('--duration',type=float,default=.002)
    ap.add_argument('--cfl',type=float,nargs='+',default=[.5,2.])
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();torch.set_num_threads(1);wp.init()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    runs=[];states=[]
    result={'snapshot':str(args.snapshot),'duration_s':args.duration,
            'scope':'Same saved complete state, fixed groups, production RK2/cuts and unchanged strict energy guards. Warmup excluded; no new agglomeration.',
            'runs':runs}
    for cfl in args.cfl:
        sol,reset,source=load(args.snapshot)
        # Same warmup for both; only caches persist, never trial state.
        step(sol,reset(),source)
        s=reset();initial=sol.energy_audit(s)
        rows=[];t=0.;failure=None;last=0.
        torch.cuda.synchronize();start=time.perf_counter()
        try:
            while t<args.duration-1e-14:
                s,row=step(sol,s,source,cfl=cfl,dt_max=args.duration-t)
                rows.append(row);t+=row['dt']
                if time.perf_counter()-last>20:
                    print(json.dumps({'cfl':cfl,'steps':len(rows),'advanced_ms':t*1000,
                                      'seconds':time.perf_counter()-start}),flush=True)
                    last=time.perf_counter()
        except Exception as e:
            failure=repr(e)
        torch.cuda.synchronize();seconds=time.perf_counter()-start
        energy=sol.energy_audit(s)
        # Hidden zero-mass geometric sites carry zero momentum, not a velocity.
        mass=s.q[:,0];velocity=s.q[:,1:4]/mass.clamp(min=1e-100)[:,None]
        center=(mass[:,None]*s.p).sum(0)/mass.sum()
        dims=((mass[:,None]*(s.p-center)**2).sum(0)/mass.sum()).sqrt()
        reasons=Counter(x.split(':')[0] for row in rows for x in row['retry_reasons'])
        run={'cfl':cfl,'completed':failure is None,'failure':failure,'physical_time_s':t,
             'seconds':seconds,'steps':len(rows),'average_dt_us':t/max(len(rows),1)*1e6,
             'min_dt_us':min((r['dt'] for r in rows),default=0)*1e6,
             'max_dt_us':max((r['dt'] for r in rows),default=0)*1e6,
             'rejected_attempts':sum(r['retry'] for r in rows),'reasons':dict(reasons),
             'initial_energy':initial,'final_energy':energy,'rms_dimensions':dims.tolist(),
             'max_displacement_m':float((s.p-reset().p).norm(dim=1).max()),
             'max_local_energy_deficit_ratio':max((r['audit']['max_negative_remainder_over_local_elastic_plus_mu_over_rho0'] for r in rows),default=0),
             'sum_work_defect_joule':sum(r['audit']['cumulative_work_error_joule'] for r in rows),
             'rows':rows}
        state={k:getattr(s,k).cpu().clone() for k in ['p','r','q']}
        state['velocity']=velocity.cpu();state['group_volume']=sol.group_sum(s.mesh.volume)[sol.groups].cpu()
        if hasattr(s.mesh,'planes'):state['planes']=s.mesh.planes.packed()
        states.append(state);runs.append(run)
        if len(runs)>1:
            base=states[0];current=states[-1]
            run['comparison_to_first']={
                'speedup':runs[0]['seconds']/seconds,
                'position_rms_difference_m':float((base['p']-current['p']).square().sum(1).mean().sqrt()),
                'position_max_difference_m':float((base['p']-current['p']).norm(dim=1).max()),
                'velocity_relative_l2_difference':float((base['velocity']-current['velocity']).norm()/base['velocity'].norm()),
                'rms_dimensions_relative_difference':[(b-a)/a for a,b in zip(runs[0]['rms_dimensions'],run['rms_dimensions'])]}
        torch.save({'states':states,'groups':sol.groups.cpu(),'result':result},args.out.with_suffix('.pt'))
        args.out.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:v for k,v in run.items() if k!='rows'}),flush=True)
        del sol,reset,s;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
