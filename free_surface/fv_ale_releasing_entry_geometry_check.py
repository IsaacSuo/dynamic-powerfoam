"""Physical geometry comparison for two differently stepped continuations."""
import json
from pathlib import Path
import torch
import warp as wp
from free_surface.fv_ale_planar_boundary import Planes
from free_surface.fv_ale_planar_floor_reuse import geometry_cpu


def compare(root):
    torch.set_num_threads(1);wp.init()
    a=torch.load(root/'fv_state_split2.pt',weights_only=False,map_location='cpu')
    b=torch.load(root/'fv_state_direct.pt',weights_only=False,map_location='cpu')
    pa,pb=Planes(**a['moving_planes']),Planes(**b['moving_planes'])
    ga=geometry_cpu(a['p'],a['r'],pa,a['ground']);gb=geometry_cpu(b['p'],b['r'],pb,b['ground'])
    groups=a['groups'];summed=lambda x:torch.zeros_like(x).index_add(0,groups,x)[groups]
    Va,Vb=summed(ga.volume),summed(gb.volume)
    da=pa.a-pb.a;db=pa.b-pb.b
    bound=((da*b['p'][pb.owner]).sum(1)-db).abs()+b['r'][pb.owner]*da.norm(dim=1)
    result={'maximum_group_volume_relative_difference':float(((Va-Vb).abs()/Vb).max()),
        'cap_area_relative_difference':float((ga.cap_area-gb.cap_area).norm()/gb.cap_area.norm()),
        'floor_area_relative_difference':float((ga.ground_area-gb.ground_area).norm()/gb.ground_area.norm()),
        'max_plane_equation_distance_bound_over_owner_ball_m':float((bound/pb.a.norm(dim=1)).max()),
        'max_plane_normal_difference':float((pa.a/pa.a.norm(dim=1)[:,None]-pb.a/pb.a.norm(dim=1)[:,None]).norm(dim=1).max())}
    counts={tag:json.loads((root/('fv_solid_'+tag+'.json')).read_text())['physical_steps'] for tag in ['split1','split2','direct']}
    result['physical_step_counts']=counts
    result['identical_step_sequence']=counts['split1']+counts['split2']==counts['direct']
    seed=torch.load(root/'seed.pt',weights_only=False,map_location='cpu')
    source=torch.load(seed['diagnostic_conversion_source'],weights_only=False,map_location='cpu')
    result['seed_material_and_plane_fields_preserved_exactly']=all(torch.equal(seed[k],source[k]) for k in ['p','r','q','groups']) and all(
        torch.equal(seed['moving_planes'][k],source['moving_planes'][k]) for k in ['owner','d','a','b'])
    return result


if __name__=='__main__':
    root=Path('output/ale/hyperelastic_releasing_floor_entry');result=compare(root)
    (root/'geometry_comparison.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
