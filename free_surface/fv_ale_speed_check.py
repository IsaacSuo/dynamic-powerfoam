"""Compare implementation choices on the same complete saved FV trajectory."""
import argparse
import gc
import json
from pathlib import Path
import time
import importlib.util
import torch
import warp as wp
from free_surface.fv_ale_hyperelastic_speed import load,step


def reference_stencil():
    path=Path('output/ale/backups/hyperelastic_speed_20261108/fv_ale_affine_stencil.py')
    spec=importlib.util.spec_from_file_location('saved_stencil',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.build


def report(args,states,runs):
    diffs=[{k:(float((states[0][k]-state[k]).double().norm()/states[0][k].double().norm().clamp(min=1e-100))
               if states[0][k].shape==state[k].shape else None) for k in states[0]} for state in states[1:]]
    shapes=[{k:[list(states[0][k].shape),list(state[k].shape)] for k in states[0]
             if states[0][k].shape!=state[k].shape} for state in states[1:]]
    diff=diffs[-1]
    dt=max(abs(a['dt']-b['dt']) for a,b in zip(runs[0]['rows'],runs[-1]['rows']))
    same_attempts=[(r['retry'],r['activation']) for r in runs[0]['rows']]==[(r['retry'],r['activation']) for r in runs[-1]['rows']]
    passed=all(v is not None and v<1e-8 for v in diff.values()) and dt<1e-12 and same_attempts
    result={'passed':passed,'strict_state_test':'All stored arrays same shape and relative difference < 1e-8; not relaxed for divergent boundary topology.',
            'snapshot':str(args.snapshot),'steps':args.steps,'relative_differences':diff,
            'all_comparisons_to_first':diffs,'shape_differences':shapes,'same_retries_and_activations':same_attempts,
            'max_dt_difference_s':dt,'runs':runs,'speedup':runs[0]['seconds']/runs[-1]['seconds']}
    if len(runs)==3:
        result['optimized_material_differences_below_observed_original_repeat']=all(
            diffs[-1][k]<=diffs[0][k] for k in ['p','r','q','group_volume'])
        result['scope_note']='One original/original control, not a statistical repeatability bound. Raw plane arrays are not comparable when topology counts differ.'
    args.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='runs'},indent=2))
    if not args.repeat_baseline:assert passed,result


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('snapshot',type=Path)
    ap.add_argument('--steps',type=int,default=20);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--repeat-baseline',action='store_true',help='measure original/original as well as original/optimized')
    ap.add_argument('--report-saved',action='store_true')
    args=ap.parse_args();torch.set_num_threads(1)
    if args.report_saved:
        saved=torch.load(args.out.with_suffix('.pt'),weights_only=False,map_location='cpu')
        report(args,saved['states'],saved['runs']);return
    wp.init();runs=[];states=[]
    import free_surface.fv_ale_affine_stencil as stencil
    optimized=stencil.build;reference=reference_stencil()
    for padded in ([False,False,True] if args.repeat_baseline else [False,True]):
        stencil.build=optimized if padded else reference
        sol,reset,source=load(args.snapshot);sol.device_ops().pad_planar_graph=padded
        sol.fast_planar_geometry=padded
        s=reset();rows=[];torch.cuda.synchronize();tick=time.perf_counter()
        for k in range(args.steps):
            s,d=step(sol,s,source);rows.append(d)
        torch.cuda.synchronize();seconds=time.perf_counter()-tick
        state={k:getattr(s,k).cpu().clone() for k in ['p','r','q']}
        if hasattr(s.mesh,'planes'):
            state.update({'plane_'+k:v for k,v in s.mesh.planes.packed().items()})
            state['exposed_plane_area']=torch.zeros_like(s.mesh.planes.b).index_add(0,s.mesh.plane_j,s.mesh.plane_area).cpu()
        state['group_volume']=sol.group_sum(s.mesh.volume)[sol.groups].cpu()
        states.append(state)
        runs.append({'padded':padded,'seconds':seconds,'rows':rows,'energy':sol.energy_audit(s),
                     'graph_builds':len(sol.device_ops().builds),
                     'peak_allocated_MB':torch.cuda.max_memory_allocated()/1e6})
        torch.save({'states':states,'runs':runs,'groups':sol.groups.cpu()},args.out.with_suffix('.pt'))
        del sol,reset,s;gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    stencil.build=optimized
    report(args,states,runs)


if __name__=='__main__':main()
