"""Analytic motion checks on the same full ground-clipped production foams."""
import json
from pathlib import Path
import torch
import numpy as np
from free_surface.fv_ale_profile import load


@torch.no_grad()
def main():
    out=Path('output/ale');records=[]
    for tag in ['dt8us','cfl20_long']:
        sol,reset,data=load(out/f'fv_state_{tag}.pt');s=reset()
        for kind in ['horizontal_translation','homothety_about_floor']:
            if kind=='horizontal_translation':
                dp=torch.zeros_like(s.p);dp[:,0]=.001;dr=torch.zeros_like(s.r);scale=1.
            else:
                center=torch.tensor([0.,sol.ground,0.],device=s.p.device,dtype=s.p.dtype)
                scale=1.002;dp=(scale-1)*(s.p-center);dr=(scale-1)*s.r
            integrated=torch.zeros_like(s.r);cut_final=None
            for tau,weight in [(0.,1/6),(.5,4/6),(1.,1/6)]:
                mesh=s.mesh if tau==0 else sol.cut(s.p+tau*dp,s.r+tau*dr)
                rate=mesh.volume_rate(dp,dr);integrated+=weight*rate
                if tau==1:cut_final=mesh
            expected=(scale**3-1)*s.mesh.volume
            gv=sol.group_sum(s.mesh.volume)[sol.groups]
            integral_error=sol.group_sum(integrated-expected)[sol.groups]/gv
            cut_error=sol.group_sum(cut_final.volume-scale**3*s.mesh.volume)[sol.groups]/gv
            row={'snapshot':tag,'motion':kind,
                'integrated_volume_max_group_relative_error':float(integral_error.abs().max()),
                'recut_volume_max_group_relative_error':float(cut_error.abs().max())}
            assert row['integrated_volume_max_group_relative_error']<1e-8,row
            assert row['recut_volume_max_group_relative_error']<1e-8,row
            records.append(row);print(json.dumps(row),flush=True)
    (out/'swept_geometry_controls.json').write_text(json.dumps(records,indent=2)+'\n')

if __name__=='__main__':main()
