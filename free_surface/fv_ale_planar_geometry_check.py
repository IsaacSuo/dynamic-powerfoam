"""Validate moving boundary topology against a complete augmented power hull.

Uses the actual 7,690-cell saved impact, including its ground contact. No
constitutive state, momentum, mass or energy is altered during these checks.
"""
import json
from pathlib import Path
import time
import torch
import warp as wp
from free_surface.fv_ale_planar_boundary import Planes,geometry_cpu,values,geometric_rate
from free_surface.fv_ale_ghost_plane_geometry_check import structure_only


def face_map(mesh):
    return {(int(i),int(j)): (float(a),c,n) for i,j,a,c,n in
        zip(mesh.plane_i,mesh.plane_j,mesh.plane_area,mesh.plane_centroid,mesh.plane_normal)}


def main():
    torch.set_num_threads(1);wp.init();root=Path('output/ale');tick=time.perf_counter()
    data=torch.load(root/'hyperelastic_acoustic_impact50ms.pt',map_location='cpu',weights_only=False)
    p,r=data['p'],data['r'];ground=data['ground'];base=structure_only(p,r)
    # Both the released owner and neighboring contacts. Nonuniform heights and
    # tilted planes exercise cross-pair faces, not only own-cell floor disks.
    owners=[2919,6333,2979];planes=Planes.floor(p,r,owners,ground)
    result={'snapshot_time':data['time'],'cells':len(r),'cases':[]}
    for case in ['insert','nonuniform_lift','tilted']:
        q=planes.clone()
        if case!='insert':q.b-=r.new_tensor([.0001,.00017,.00008])
        if case=='tilted':
            rotation=r.new_tensor([[.015,0.,-.01],[-.012,0.,.005],[.009,0.,.013]])
            q.a+=rotation;q.b+=(rotation*p[q.owner]).sum(1)
        fast=geometry_cpu(p,r,q,ground,base);reference=geometry_cpu(p,r,q,ground,full_hull=True)
        ev=float((fast.volume-reference.volume).norm()/reference.volume.norm())
        ea=float((fast.cap_area-reference.cap_area).norm()/reference.cap_area.norm())
        ef=float((fast.ground_area-reference.ground_area).norm()/reference.ground_area.norm())
        fa,fb=face_map(fast),face_map(reference);total=max(float(reference.plane_area.sum()),1e-300)
        common=fa.keys()&fb.keys()
        area_error=sum(abs(fa.get(k,(0,))[0]-fb.get(k,(0,))[0]) for k in fa.keys()|fb.keys())/total
        moment_error=sum(float((fa[k][0]*fa[k][1]-fb[k][0]*fb[k][1]).norm()) for k in common)/total
        row={'case':case,'volume_relative_error':ev,'cap_relative_error':ea,'floor_relative_error':ef,
             'planar_area_relative_error':area_error,'planar_moment_error_m':moment_error,
             'planar_faces':len(fast.plane_i),'cross_faces':int((fast.plane_i!=q.owner[fast.plane_j]).sum())}
        assert max(ev,ea,ef,area_error,moment_error)<1e-7,row
        if case=='tilted':
            gen=torch.Generator().manual_seed(88)
            w=torch.randn(p.shape,generator=gen,dtype=p.dtype)*.03
            rd=torch.randn(r.shape,generator=gen,dtype=r.dtype)*.01
            ad=torch.randn(q.a.shape,generator=gen,dtype=p.dtype)*.02
            bd=torch.randn(q.b.shape,generator=gen,dtype=p.dtype)*.01
            b={'p':p,'r':r,**values(fast)}
            fi=fast.plane_i;normal=geometric_rate(b,w,rd,ad,bd)
            predicted=fast.volume_rate(w,rd,plane_rates=(ad,bd))
            errors=[]
            for h in [1e-7,1e-8]:
                plus=q.clone();minus=q.clone();plus.a+=h*ad;minus.a-=h*ad;plus.b+=h*bd;minus.b-=h*bd
                gp=geometry_cpu(p+h*w,r+h*rd,plus,ground)
                gm=geometry_cpu(p-h*w,r-h*rd,minus,ground)
                measured=(gp.volume-gm.volume)/(2*h)
                errors.append(float((measured-predicted).norm()/predicted.norm()))
            row['volume_rate_errors']=errors
            assert min(errors)<2e-5,row
        result['cases'].append(row);print(json.dumps(row),flush=True)
    result.update(passed=True,seconds=time.perf_counter()-tick)
    (root/'hyperelastic_planar_geometry_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
