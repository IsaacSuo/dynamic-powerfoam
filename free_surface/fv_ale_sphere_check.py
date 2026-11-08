"""Direct spherical area versus reference on complete saved FV states."""
import json,time,statistics
from pathlib import Path
import torch
from free_surface.fv_ale_profile import load
from free_surface.fv_ale_ground_speed_check import geometry_difference

@torch.no_grad()
def main():
    out=Path('output/ale');records=[]
    for tag in ['plane_direct_b','dt8us','cfl20_long']:
        sol,reset,data=load(out/f'fv_state_{tag}.pt')
        timings={False:[],True:[]};meshes={}
        for direct in [False,True]*2:
            sol.ground_cutter.recut.pipe.direct_sphere=direct;reset()
        for rep in range(12):
            for direct in ([False,True] if rep%2==0 else [True,False]):
                sol.ground_cutter.recut.pipe.direct_sphere=direct;torch.cuda.synchronize();tick=time.perf_counter()
                meshes[direct]=reset().mesh;torch.cuda.synchronize()
                timings[direct].append(1000*(time.perf_counter()-tick))
        errors=geometry_difference(meshes[False],meshes[True])
        active=meshes[False].volume>0
        dv=((meshes[False].volume[active]-meshes[True].volume[active])/meshes[False].volume[active]).abs()
        assert torch.isfinite(meshes[True].volume).all()
        record={'snapshot':tag,'geometry_errors':errors,'max_relative_volume_error':float(dv.max()),
            'median_ms':{str(k):statistics.median(v) for k,v in timings.items()},'times_ms':timings}
        print(json.dumps(record),flush=True);records.append(record)
        assert float(dv.max())<1e-7,record
        assert errors['cap_area']['max_abs']<1e-10,record
    (out/'sphere_direct_geometry.json').write_text(json.dumps(records,indent=2)+'\n')

if __name__=='__main__':main()
