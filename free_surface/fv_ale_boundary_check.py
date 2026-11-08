"""Real full-step replay for the optional local clipping-boundary experiment."""
import json,time,statistics
from pathlib import Path
import torch,warp as wp
from free_surface.fv_ale_profile import load,physical_difference
from free_surface.fv_ale_agglomerate import merge_cell


@torch.no_grad()
def main():
    out=Path('output/ale');records=[]
    for tag in ['dt8us','cfl20_long']:
        sol,reset,data=load(out/f'fv_state_{tag}.pt');labels=sol.groups.clone()
        def fresh(mode):
            sol.groups=labels.clone();sol.has_groups=True
            sol.ground_cutter.recut.pipe.local_boundary=mode
            return reset()
        def step(s):
            while True:
                gv=sol.group_sum(s.mesh.volume)[sol.groups]
                small=gv<.001*data['reference_volume']
                first=None
                if bool(small.any()):
                    k=int(torch.where(small,gv,torch.full_like(gv,float('inf'))).argmin())
                else:
                    first=sol.rhs(s)
                    k=first[3]['wave_cell'] if 2*first[3]['wave_dt_scale']<80e-6 else None
                if k is None:break
                s,_=merge_cell(sol,s,k)
            return sol.advance(s,min(1/1200,2*first[3]['dt_scale']),first,2.)[0]
        for mode in [False,True]*2:step(fresh(mode))
        times={False:[],True:[]};errors=[]
        for rep in range(8):
            states={};raw={}
            for mode in ([False,True] if rep%2==0 else [True,False]):
                s=fresh(mode);torch.cuda.synchronize();tick=time.perf_counter();states[mode]=step(s);torch.cuda.synchronize()
                times[mode].append(1000*(time.perf_counter()-tick))
                a=sol.ground_cutter.recut.A
                nw=wp.to_torch(a.nwall).clone()
                walls=wp.to_torch(a.walls).clone()
                raw[mode]=(nw,torch.where(torch.arange(walls.shape[1],device=walls.device)[None]<nw[:,None],walls,-1))
            assert all(torch.equal(a,b) for a,b in zip(raw[False],raw[True]))
            d=physical_difference(sol,states[False],states[True]);errors.append(d)
            assert d['max_position_difference_m']<1e-10 and d['max_velocity_difference_m_s']<1e-7 and d['max_pressure_difference_Pa']<1e-3,d
        record={'snapshot':tag,'reference_median_ms':statistics.median(times[False]),'local_boundary_median_ms':statistics.median(times[True]),
            'paired_saved_median_ms':statistics.median([a-b for a,b in zip(times[False],times[True])]),'faster_pairs':sum(a>b for a,b in zip(times[False],times[True])),
            'physical_differences':errors,'timings':times,'passed':True}
        records.append(record);print(json.dumps(record),flush=True)
    (out/'recut_local_boundary_step.json').write_text(json.dumps(records,indent=2)+'\n')


if __name__=='__main__':main()
