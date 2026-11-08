"""Geometry edge cases for the target-only pipeline, alongside full-state tests."""
import json
from pathlib import Path
import numpy as np
import torch
from free_surface.fv_ale_target_cut import TargetCut
from free_surface.fv_ale_ground_geometry import GroundClip

@torch.no_grad()
def main():
    cases=[('isolated',[[0,2,0]],[1],[0]),
        ('floor_cap',[[0,.2,0]],[1],[0]),
        ('below_floor',[[0,-2,0]],[1],[0]),
        ('hidden',[[0,2,0],[.1,2,0]],[2,.2],[1]),
        ('larger_target_id',[[-.5,.3,0],[.5,.3,0]],[1,1],[1]),
        ('both_targets',[[-.5,.3,0],[.5,.3,0]],[1,1],[0,1]),
        ('third_ball_constraint',[[-.5,.3,0],[.5,.3,0],[0,.5,.4]],[1,1,.8],[1])]
    report=[]
    for name,points,radii,targets in cases:
        p=torch.tensor(points,dtype=torch.float64,device='cuda');r=torch.tensor(radii,dtype=p.dtype,device=p.device)
        rows=[np.array([j for j in range(len(p)) if j!=i],dtype=np.int64) for i in targets]
        ref=GroundClip(0.,'cuda').geometry(p,r);got=TargetCut(len(p),targets,rows,0.,'cuda').geometry(p,r)
        record={'case':name};ix=torch.tensor(targets,device=p.device)
        for key in ['volume','cap_area','cap_normal_integral','ground_area']:
            diff=float((getattr(ref,key)[ix]-getattr(got,key)[ix]).abs().max())
            record[key+'_max_abs']=diff;assert diff<1e-11,record
        targetmask=torch.zeros(len(p),dtype=torch.bool,device=p.device);targetmask[ix]=True
        keep=targetmask[ref.i]|targetmask[ref.j]
        rk=ref.i[keep]*len(p)+ref.j[keep];gk=got.i*len(p)+got.j
        assert torch.equal(rk,gk),(name,rk,gk)
        for key in ['area','normal','centroid']:
            delta=getattr(ref,key)[keep]-getattr(got,key)
            diff=float(delta.abs().max()) if delta.numel() else 0.
            record[key+'_max_abs']=diff;assert diff<1e-11,record
        report.append(record)
    Path('output/ale/target_cut_edge_checks.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
