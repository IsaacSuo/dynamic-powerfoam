"""Compare local ground updates to the retained full-field implementation.

Uses real saved production states, alternating full-step replays. Reset cuts
are excluded from timings. No material law, CFL or grouping rule is changed.
"""
import argparse,json,time
from pathlib import Path
import torch,warp as wp
from free_surface.fv_ale_profile import load,one_step,summary,physical_difference


def geometry_difference(a,b):
    result={}
    for key in ['i','j','hidden']:
        if not torch.equal(getattr(a,key),getattr(b,key)):
            raise AssertionError('geometry topology differs: '+key)
    for key in ['volume','area','normal','centroid','cap_area','cap_normal_integral','cap_Q','ground_area','ground_centroid']:
        x=getattr(a,key);y=getattr(b,key)
        result[key]={'max_abs':float((x-y).abs().max()) if x.numel() else 0.,
            'relative_l2':float((x-y).norm()/x.norm().clamp(min=1e-100))}
    return result


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('snapshots',type=Path,nargs='+')
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--repeat',type=int,default=8)
    ap.add_argument('--sequence',type=int,default=0,help='also compare this many successive steps from each snapshot')
    ap.add_argument('--device-step',action='store_true',help='compare reference tensor step to captured GPU numerical segments; both use local ground clipping')
    args=ap.parse_args();wp.init();report={'snapshots':[],'repeat':args.repeat,'passed':True,'comparison':'whole_step_cuda_segments' if args.device_step else 'local_ground_geometry'}
    for path in args.snapshots:
        sol,reset,data=load(path);rv=data['reference_volume'];timings={False:[],True:[]};differences=[]
        def set_mode(value):
            if args.device_step:sol.device_step=value
            else:sol.ground_cutter.local=value
        for mode in [False,True,True,False]:
            set_mode(mode);one_step(sol,reset(),rv,True)
        meshes={};outputs={};rhs={};dt={}
        for k in range(args.repeat):
            for mode in ([False,True] if k%2==0 else [True,False]):
                set_mode(mode);s=reset();meshes[mode]=s.mesh
                torch.cuda.synchronize();tick=time.perf_counter()
                outputs[mode],stats=one_step(sol,s,rv,True)
                torch.cuda.synchronize();timings[mode].append(1000*(time.perf_counter()-tick))
            differences.append(physical_difference(sol,outputs[False],outputs[True]))
        for mode in [False,True]:
            set_mode(mode);rr=sol.rhs(reset());rhs[mode]=rr;dt[mode]=rr[3]['dt_scale']
        geom=geometry_difference(meshes[False],meshes[True])
        rates={str(j):float((rhs[False][j]-rhs[True][j]).norm()/rhs[False][j].norm().clamp(min=1e-100)) for j in range(3)}
        passed=all(v['relative_l2']<1e-10 for v in geom.values()) and max(rates.values())<1e-8
        passed=passed and abs(dt[False]/dt[True]-1)<1e-10
        passed=passed and all(d['max_position_difference_m']<1e-12 and d['max_velocity_difference_m_s']<1e-9 and d['max_pressure_difference_Pa']<1e-5 for d in differences)
        record={'snapshot':str(path),'time':data['time'],'full_step_ms':summary(timings[False]),'local_step_ms':summary(timings[True]),
            'reference_step_ms':summary(timings[False]),'optimized_step_ms':summary(timings[True]),
            'paired_saved_ms':summary([x-y for x,y in zip(timings[False],timings[True])]),
            'local_faster_pairs':sum(x>y for x,y in zip(timings[False],timings[True])),
            'geometry_differences':geom,'rhs_relative_differences':rates,'dt_scale':dt[True],
            'physical_differences':differences,'local_geometry':meshes[True].ground_geometry_info,'passed':passed}
        if args.sequence:
            trajectories={};history={}
            for mode in [False,True]:
                set_mode(mode);s=reset();times=[]
                for _ in range(args.sequence):
                    first=sol.rhs(s);step=min(1/1200,.2*first[3]['dt_scale'])
                    s,stats=sol.advance(s,step,first,.2);times.append(step)
                trajectories[mode]=s;history[mode]=times
            diff=physical_difference(sol,trajectories[False],trajectories[True])
            error=max(abs(x-y) for x,y in zip(history[False],history[True]))
            seqpass=diff['max_position_difference_m']<1e-10 and diff['max_velocity_difference_m_s']<1e-7 and diff['max_pressure_difference_Pa']<1e-3 and error<1e-12
            record['sequence']={'steps':args.sequence,'dt_max_difference':error,'physical_differences':diff,'passed':seqpass}
            record['passed']=record['passed'] and seqpass
        report['passed']=report['passed'] and record['passed'];report['snapshots'].append(record)
        args.out.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:record[k] for k in ['snapshot','full_step_ms','local_step_ms','local_geometry','passed']},indent=2),flush=True)
    if not report['passed']:raise SystemExit(2)


if __name__=='__main__':main()
