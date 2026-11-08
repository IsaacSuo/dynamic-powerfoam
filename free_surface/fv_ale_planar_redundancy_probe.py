"""Read-only diagnosis of accumulated boundary generators on a full replay."""
import json
from pathlib import Path
import time
import torch
import warp as wp
from free_surface.fv_ale_planar_boundary import Planes,geometry_cpu


def main():
    torch.set_num_threads(1);wp.init();root=Path('output/ale');tick=time.perf_counter()
    data=torch.load(root/'hyperelastic_releasing_step_cpu_1ms.pt',map_location='cpu',weights_only=False)
    p,r=data['p'],data['r'];pl=Planes(**data['moving_planes']);ground=data['ground']
    mesh=geometry_cpu(p,r,pl,ground);areas=torch.zeros_like(pl.b).index_add(0,mesh.plane_j,mesh.plane_area)
    heights=torch.full_like(pl.b,float('inf')).scatter_reduce_(0,mesh.plane_j,mesh.plane_centroid[:,1]-ground,reduce='amin')
    records=[{'plane':j,'owner':int(pl.owner[j]),'area_m2':float(areas[j]),'minimum_face_centroid_gap_m':float(heights[j]),
              'a':pl.a[j].tolist(),'b':float(pl.b[j])} for j in range(len(pl.owner))]
    group=data['groups'];sumg=lambda x:torch.zeros_like(x).index_add(0,group,x)
    original=sumg(mesh.volume);checks=[]
    masks={'strictly_nonzero_area':areas>0}
    latest=torch.zeros_like(areas,dtype=torch.bool)
    for owner in torch.unique(pl.owner):latest[torch.nonzero(pl.owner==owner).flatten()[-1]]=True
    masks['only_latest_per_owner']=latest
    for name,keep in masks.items():
        q=Planes(*(getattr(pl,key)[keep] for key in ['owner','d','a','b']))
        cut=geometry_cpu(p,r,q,ground)
        checks.append({'selection':name,'remaining':int(keep.sum()),
            'volume_relative_l2_change':float((cut.volume-mesh.volume).norm()/mesh.volume.norm()),
            'maximum_group_volume_relative_change':float(((sumg(cut.volume)-original).abs()/original.clamp(min=1e-300))[torch.unique(group)].max()),
            'maximum_cell_absolute_volume_change':float((cut.volume-mesh.volume).abs().max()),
            'ground_area_absolute_change':float((cut.ground_area-mesh.ground_area).abs().sum())})
    result={'plane_count':len(pl.owner),'strictly_exposed_count':int((areas>0).sum()),'planes':records,'removal_probes':checks,
            'scope':'Read only; no boundary was removed from an advancing state.','seconds':time.perf_counter()-tick}
    (root/'hyperelastic_planar_redundancy_probe.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
