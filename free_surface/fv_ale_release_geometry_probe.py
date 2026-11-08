"""Inspect the actual release state and an unchanged-domain boundary insertion.

This is a full saved impact state, not a replacement scene. A weighted vacuum
generator is geometric only; it receives no mass or constitutive state. This
probe does not yet advance that generator or claim a working release method.
"""
import argparse
import json
import math
from pathlib import Path
import mpmath as mp
import torch
import warp as wp
from free_surface.fv_ale_compatible_mesh_audit import restored
from free_surface.fv_ale_precise_geometry_check import cpu_mesh
from free_surface.fv_ale_ghost_plane_geometry_check import structure_only


def floor_generator(p,r,owner,ground):
    """Use exact input doubles to certify that the new power plane is >= floor."""
    ghost=p[owner].clone();ghost[1]-=1.
    with mp.workdps(60):
        py=mp.mpf(float(p[owner,1]));gy=mp.mpf(float(ghost[1]));radius=mp.mpf(float(r[owner]))
        distance=py-gy;target=radius**2+distance**2-2*distance*(py-mp.mpf(ground))
        assert target>0
        rad=math.sqrt(float(target))
        while mp.mpf(rad)**2<target:rad=math.nextafter(rad,math.inf)
        actual_height=py+(mp.mpf(rad)**2-radius**2-distance**2)/(2*distance)
        assert actual_height>=mp.mpf(ground)
        displacement=float(actual_height-mp.mpf(ground))
    return ghost,r.new_tensor(rad),displacement


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',default='hyperelastic_acoustic_impact50ms');args=parser.parse_args()
    torch.set_num_threads(1);wp.init();root=Path('output/ale')
    data=torch.load(root/(args.tag+'.pt'),map_location='cpu',weights_only=False)
    floor=data['ground'];sol,state,ops,b=restored(data,ground=floor)
    with torch.no_grad():
        w,dr,dq,packed=ops.rhs_gpu(b)
        _,_,v,rho,_,_,sigma,c,_=ops.primitive(state.q,state.mesh.volume,sol.groups)
    vn=ops._compatible['wall_normal_velocity'];pressure=-sigma[:,1,1]-rho*c*vn
    active=state.mesh.ground_area>0
    ids=torch.argsort(torch.where(active,pressure,torch.full_like(pressure,math.inf)))[:5]
    rows=[]
    for k in ids.tolist():
        rows.append({'cell':k,'group':int(sol.groups[k]),'area_m2':float(state.mesh.ground_area[k]),
                     'material_normal_stress_Pa':float(sigma[k,1,1]),'riemann_pressure_Pa':float(pressure[k]),
                     'normal_velocity_reconstructed_m_s':float(vn[k]),
                     'traction_free_normal_velocity_m_s':float(vn[k]+sigma[k,1,1]/(rho[k]*c[k])),
                     'ball_bottom_rate_m_s':float(w[k,1]-dr[k]),'mass_kg':float(state.q[k,0])})
    owner=int(ids[0]);p,r=state.p,state.r;n=len(r)
    ghost,rad,rounding=floor_generator(p,r,owner,floor)
    pp,rr=torch.cat((p,ghost[None])),torch.cat((r,rad[None]))
    redundant=torch.zeros(n+1,dtype=torch.bool);redundant[owner]=True
    mesh=cpu_mesh(pp,rr,floor,structure_only(pp,rr),redundant,n)
    free=(mesh.i<n)&(mesh.j==n)
    # This includes all existing internal/free-sphere geometry, not merely a
    # single-cell volume formula. All material variables remain untouched.
    volume_difference=mesh.volume[:n]-state.mesh.volume
    ground_plus_free=mesh.ground_area[:n].clone().index_add(0,mesh.i[free],mesh.area[free])
    area_difference=ground_plus_free-state.mesh.ground_area
    centroid_height=mesh.centroid[free,1]-floor
    # Give the inserted boundary a finite gap and verify the complete volume
    # derivative under motion of its weight, including all cross-pair faces.
    distance=float(p[owner,1]-ghost[1]);lift=.0001
    lifted_weight=rad.square()+2*distance*lift
    lifted_r=rr.clone();lifted_r[-1]=lifted_weight.sqrt()
    lifted=cpu_mesh(pp,lifted_r,floor,structure_only(pp,lifted_r),redundant,n)
    speed=torch.zeros_like(lifted_r);speed[-1]=distance/lifted_r[-1]
    predicted=lifted.volume_rate(torch.zeros_like(pp),speed)[:n]
    checks=[]
    for h in [1e-7,1e-8]:
        plus_r=lifted_r.clone();minus_r=lifted_r.clone()
        plus_r[-1]=(lifted_weight+2*distance*h).sqrt();minus_r[-1]=(lifted_weight-2*distance*h).sqrt()
        plus=cpu_mesh(pp,plus_r,floor,structure_only(pp,plus_r),redundant,n)
        minus=cpu_mesh(pp,minus_r,floor,structure_only(pp,minus_r),redundant,n)
        measured=(plus.volume[:n]-minus.volume[:n])/(2*h)
        checks.append({'h_seconds':h,'relative_volume_rate_error':float((measured-predicted).norm()/predicted.norm())})
    result={'snapshot':args.tag,'time':data['time'],'release_candidates':rows,'inserted_owner':owner,
            'new_plane_upward_rounding_m':rounding,
            'volume_max_absolute_difference_m3':float(volume_difference.abs().max()),
            'volume_relative_l2_difference':float(volume_difference.norm()/state.mesh.volume.norm()),
            'group_volume_max_relative_difference':float((sol.group_sum(volume_difference).abs()/sol.group_sum(state.mesh.volume).clamp(min=1e-300))[torch.unique(sol.groups)].max()),
            'ground_plus_planar_free_area_max_difference_m2':float(area_difference.abs().max()),
            'new_boundary_centroid_height_range_m':[float(centroid_height.min()),float(centroid_height.max())],
            'precise_disks':getattr(mesh,'precise_disks',[]),
            'lifted_boundary_gap_m':lift,'lifted_volume_derivative_checks':checks,
            'new_planar_faces':[{'real_cell':int(mesh.i[k]),'area_m2':float(mesh.area[k]),
                                'centroid_m':mesh.centroid[k].tolist(),'normal':mesh.normal[k].tolist()}
                               for k in torch.nonzero(free).flatten().tolist()],
            'scope':'Read-only full-state geometry insertion; no mass, energy, velocity or elastic-state change; dynamics not implemented.'}
    result['passed']=result['volume_relative_l2_difference']<1e-10 and result['group_volume_max_relative_difference']<1e-10 and result['ground_plus_planar_free_area_max_difference_m2']<1e-10 and bool((centroid_height>=-1e-14).all() and (centroid_height<=rounding+1e-14).all()) and min(row['relative_volume_rate_error'] for row in checks)<1e-5
    (root/(args.tag+'_release_geometry_probe.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
