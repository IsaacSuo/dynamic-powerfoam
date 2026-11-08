"""Inspect a saved rejected actual planar cut; never changes the trajectory."""
import json
from pathlib import Path
import torch
import warp as wp
from free_surface.fv_ale_planar_boundary import Planes
from free_surface.fv_ale_planar_floor_reuse import geometry_cpu


def main():
    torch.set_num_threads(1);wp.init();root=Path('output/ale')
    data=torch.load(root/'hyperelastic_releasing_floorfix80ms.pt',weights_only=False,map_location='cpu')
    bad=data.get('rejected_geometry',{})
    if not bad:
        print('No rejected geometry was saved.');return
    p,r=bad['p'],bad['r'];pl=Planes(**bad['moving_planes']);mesh=geometry_cpu(p,r,pl,data['ground'])
    groups=data['groups'];summed=lambda a:torch.zeros_like(a).index_add(0,groups,a)[groups]
    members=summed(torch.ones_like(r));vol=summed(mesh.volume.clamp(min=0))
    bound=256*torch.finfo(r.dtype).eps*(4*torch.pi/3)*r**3
    ids=torch.nonzero(mesh.volume<0).flatten();rows=[]
    for k in ids.tolist():
        real=(mesh.i==k)|(mesh.j==k);planar=mesh.plane_i==k
        rows.append({'cell':k,'volume':float(mesh.volume[k]),'roundoff_bound':float(bound[k]),
            'negative_over_bound':float(-mesh.volume[k]/bound[k]),'group_members':int(members[k]),
            'group_volume':float(vol[k]),'floor_area':float(mesh.ground_area[k]),
            'real_faces':[{'neighbor':int(j if i==k else i),'area':float(A)} for i,j,A in zip(mesh.i[real],mesh.j[real],mesh.area[real])],
            'plane_faces':[{'plane':int(j),'area':float(A)} for j,A in zip(mesh.plane_j[planar],mesh.plane_area[planar])],
            'cap_area':float(mesh.cap_area[k]),'radius':float(r[k])})
    from types import SimpleNamespace
    from free_surface.fv_ale_precise_geometry import reevaluate
    solver=SimpleNamespace(groups=groups,ground=data['ground'],_precision_geometry_records=[])
    solver.group_sum=lambda x:torch.zeros_like(x).index_add(0,groups,x)
    original=mesh.volume.clone();ok=reevaluate(solver,mesh)
    result={'saved_reason':bad['reason'],'negative_cells_cpu':rows,'saved_state_time':data['time'],
        'positive_integral_reevaluation_succeeded':ok,'records':solver._precision_geometry_records,
        'maximum_group_relative_volume_correction':float((summed((mesh.volume-original).abs())/vol).max())}
    assert ok and bool((mesh.volume>=0).all()),result
    (root/'hyperelastic_planar_negative_probe.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
