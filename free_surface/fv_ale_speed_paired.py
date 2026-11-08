"""Alternating old/new complete frozen steps; compile/reset excluded equally."""
import argparse
import json
import statistics
from pathlib import Path
import time
import torch
import warp as wp
from free_surface.fv_ale_hyperelastic_speed import load,step
from free_surface.fv_ale_speed_check import reference_stencil


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('snapshot',type=Path)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--repeat',type=int,default=8)
    args=ap.parse_args();torch.set_num_threads(1);wp.init()
    import free_surface.fv_ale_affine_stencil as stencil
    methods=[reference_stencil(),stencil.build];solvers=[]
    for fast in [False,True]:
        sol,reset,source=load(args.snapshot);sol.fast_planar_geometry=fast;sol.device_ops().pad_planar_graph=fast
        solvers.append((sol,reset,source));stencil.build=methods[int(fast)]
        for _ in range(2):step(sol,reset(),source)
    times=[[],[]];errors=[]
    for repeat in range(args.repeat):
        outputs={}
        for k in ([0,1] if repeat%2==0 else [1,0]):
            stencil.build=methods[k];sol,reset,source=solvers[k];s=reset()
            torch.cuda.synchronize();tick=time.perf_counter();end,row=step(sol,s,source)
            torch.cuda.synchronize();times[k].append(1000*(time.perf_counter()-tick));outputs[k]=end
        errors.append({key:float((getattr(outputs[0],key)-getattr(outputs[1],key)).norm()/getattr(outputs[0],key).norm()) for key in ['p','r','q']})
    stencil.build=methods[1]
    result={'snapshot':str(args.snapshot),'baseline_ms':times[0],'optimized_ms':times[1],
            'baseline_median_ms':statistics.median(times[0]),'optimized_median_ms':statistics.median(times[1]),
            'speedup':statistics.median(times[0])/statistics.median(times[1]),'relative_state_errors':errors}
    result['passed']=all(torch.isfinite(torch.tensor(list(x.values()))).all() and max(x.values())<1e-8 for x in errors)
    args.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    assert result['passed'],errors


if __name__=='__main__':main()
