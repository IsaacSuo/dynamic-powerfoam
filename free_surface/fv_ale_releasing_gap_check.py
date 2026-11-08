"""Read-only gap check by lifting the diagnostic floor and independently recutting.

No trajectory is advanced with a changed floor. Unchanged material volumes and
zero floor area establish a resolved geometric gap, rather than inferring it
from a centre-of-mass velocity or an unclipped sphere bottom.
"""
import argparse
import json
from pathlib import Path
import torch
import warp as wp
from free_surface.fv_ale_planar_boundary import Planes
from free_surface.fv_ale_planar_floor_reuse import geometry_cpu


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);args=ap.parse_args()
    torch.set_num_threads(1);wp.init();root=Path('output/ale')
    d=torch.load(root/(args.tag+'.pt'),weights_only=False,map_location='cpu')
    pl=Planes(**d['moving_planes']);p,r=d['p'],d['r'];groups=d['groups']
    baseline=geometry_cpu(p,r,pl,d['ground'])
    summed=lambda v:torch.zeros_like(v).index_add(0,groups,v)[groups]
    V=summed(baseline.volume);rows=[];lower=0.
    for gap in [1e-5,1e-4,.001,.002,.005,.01,.02]:
        try:
            mesh=geometry_cpu(p,r,pl,d['ground']+gap)
            change=float(((summed(mesh.volume)-V).abs()/V).max())
            area=float(mesh.ground_area.sum());unchanged=change<1e-10 and area<1e-12
            rows.append({'raised_floor_m':gap,'maximum_group_volume_relative_change':change,
                'raised_floor_contact_area_m2':area,'resolved_empty_gap':unchanged})
            if unchanged:lower=gap
            else:break
        except RuntimeError as error:
            rows.append({'raised_floor_m':gap,'error':str(error),'resolved_empty_gap':False});break
    result={'snapshot_time':d['time'],'original_floor_contact_area_m2':float(baseline.ground_area.sum()),
            'resolved_gap_lower_bound_m':lower,'passed':lower>=1e-5,'rows':rows,
            'scope':'Read-only geometric resolution check; no changed-floor simulation and no sphere-bottom proxy'}
    (root/(args.tag+'_gap_check.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if not result['passed']:raise RuntimeError('No resolved 10 micrometre gap established')


if __name__=='__main__':main()
