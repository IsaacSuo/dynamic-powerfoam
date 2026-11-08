"""Paired full/local time-integral comparisons; dense reference read after timing."""
import argparse,json,time,math,gc
from pathlib import Path
import numpy as np
import torch
from free_surface.fv_ale_profile import load
from free_surface.fv_ale_swept_check import PathAudit,endpoint
from free_surface.fv_ale_swept_local import LocalPath,indicator,errors


def evaluate(a,kind,panels,tol):
    a.sample(0);a.sample(1)
    history=[]
    if kind=='full_serial':return a.integrate(panels),{'full_cut_seconds':a.cut_seconds}
    full=LocalPath(a,np.arange(a.n),4)
    if kind=='full_batch':
        result=full.integrate(panels)
        return result,{'full_cut_seconds':full.cut_seconds,'batch_calls':full.batched_cut_calls}
    coarse=full.integrate(2);result=full.integrate(4)
    # These are whole-scene samples, reusable by every subsequent local region.
    a.cache.update(full.cache)
    estimate,defect=indicator(a,result,coarse)
    selected=(estimate>tol*.25)|(defect>tol*.25)
    if not selected.any():return result,{'geometry_seconds':full.cut_seconds,'coverage_seconds':0.,'levels':[]}
    target_only=kind=='local_target'
    local=LocalPath(a,np.flatnonzero(selected[a.g]),4,target_only)
    cut_total=full.cut_seconds;coverage_total=local.coverage_seconds
    for power in range(3,int(math.log2(panels))+1):
        count=2**power;before=local.cut_seconds
        refined=local.integrate(count);combined=local.combine(result,refined)
        cut_total+=local.cut_seconds-before
        estimate,defect=indicator(a,combined,result)
        history.append({'nodes':count+1,'selected_groups':int(selected.sum()),'target_cells':len(local.targets),
            'patch_cells':len(local.ids),'max_group_volume_defect':float(defect.max()),
            'new_geometry_seconds':local.cut_seconds-before})
        result=combined
        if kind in ['local_adaptive','local_target'] and count<panels:
            new_selected=((estimate>tol*.25)&selected)|(defect>tol*.25)
            if not new_selected.any():break
            next_local=LocalPath(a,np.flatnonzero(new_selected[a.g]),4,target_only)
            coverage_total+=next_local.coverage_seconds
            if np.all(~new_selected|selected):
                # An old target owns its complete face star. A subset therefore
                # reuses those samples exactly; newly added targets may not.
                next_local.cache={t:next_local.restrict(s) for t,s in local.cache.items()}
            local=next_local;selected=new_selected
    return result,{'geometry_seconds':cut_total,'coverage_seconds':coverage_total,'levels':history}


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repeat',type=int,default=2)
    ap.add_argument('--panels',type=int,default=64);ap.add_argument('--tolerance',type=float,default=1e-5)
    ap.add_argument('--tags',nargs='+',default=['dt8us','cfl20_long'])
    ap.add_argument('--modes',nargs='+',default=['full_serial','full_batch','local_fixed','local_adaptive'])
    ap.add_argument('--output',default='swept_local_bench.json')
    args=ap.parse_args();out=Path('output/ale');records=[]
    modes=args.modes
    for tag in args.tags:
        sol,reset,data=load(out/f'fv_state_{tag}.pt');start=reset();end,reference_cost=endpoint(sol,start,800e-6)
        for rep in range(args.repeat):
            current={}
            for kind in modes if rep%2==0 else list(reversed(modes)):
                gc.collect();a=PathAudit(sol,start,end,800e-6,data['reference_volume'])
                # Start and end cuts exist already; exclude identical extraction.
                a.sample(0);a.sample(1);torch.cuda.synchronize();tick=time.perf_counter()
                result,stats=evaluate(a,kind,args.panels,args.tolerance)
                torch.cuda.synchronize();elapsed=time.perf_counter()-tick
                current[kind]=result
                dense=dict(np.load(out/f'swept_{tag}_800us.npz'))
                record={'snapshot':tag,'repeat':rep,'mode':kind,'seconds':elapsed,
                    'panels':args.panels,'tolerance':args.tolerance,**stats,'errors':errors(a,result,dense)}
                records.append(record);print(json.dumps(record),flush=True)
            a=PathAudit(sol,start,end,800e-6,data['reference_volume'])
            pair={'snapshot':tag,'repeat':rep,'mode':'same_nodes_validation'}
            if 'full_serial' in current:
                for name in modes:
                    if name!='full_serial':pair[name+'_vs_full']=errors(a,current[name],current['full_serial'])
                if 'full_batch' in current:assert pair['full_batch_vs_full']['flux_max_face_error_scaled']<1e-8,pair
            if 'local_target' in current and 'local_adaptive' in current:
                pair['target_vs_previous']=errors(a,current['local_target'],current['local_adaptive'])
                assert pair['target_vs_previous']['flux_max_face_error_scaled']<1e-8,pair
            records.append(pair)
            (out/args.output).write_text(json.dumps(records,indent=2)+'\n')

if __name__=='__main__':main()
