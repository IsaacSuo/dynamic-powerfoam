"""Read-only test of a possible boundary representation on the full foam.

No ghost is added to the mechanics. The test asks only whether existing
geometry kernels can reproduce a fixed ground by weighted vacuum generators.
It is not an impact, release algorithm or replacement material model.
"""
import json
from pathlib import Path
import time
import numpy as np
import torch
import warp as wp
from free_surface.fv_ale_precise_geometry_check import cpu_mesh


def structure_only(p,r):
    # The existing triangulation also solves tetrahedral circumcentres,
    # which are singular for exact paired coplanar generators and unused by
    # the disk/sphere kernels. Extract just the hull incidence, without
    # perturbing positions or inventing inverse matrices.
    from scipy.spatial import ConvexHull
    from free_surface.gpu_geometry import _pack
    p=p.numpy();r=r.numpy();n=len(r);size=np.ptp(p,axis=0).max();center=p.mean(0)
    box=np.array([[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-1,1]])
    pp=np.concatenate((p,center+10*size*(box+.1*np.random.default_rng(0).standard_normal(box.shape))))
    rr=np.concatenate((r,np.zeros(8)));z=(pp-center)/size
    hull=ConvexHull(np.column_stack((z,(z*z).sum(1)-(rr/size)**2)))
    T=hull.simplices[hull.equations[:,-2]<-1e-12]
    faces=np.unique(np.sort(np.concatenate([T[:,[0,1,2]],T[:,[0,1,3]],T[:,[0,2,3]],T[:,[1,2,3]]]),axis=1),axis=0)
    faces=faces[(faces<n).all(1)]
    edges=np.sort(np.concatenate([T[:,[a,b]] for a in range(4) for b in range(a+1,4)]),axis=1)
    edges=np.unique(edges[(edges<n).all(1)],axis=0)
    hidden=np.ones(n,bool);hidden[T[T<n]]=False
    return _pack(edges,np.concatenate([faces[:,[0,1,2]],faces[:,[0,2,1]],faces[:,[1,2,0]]]),n,hidden,'cpu')


def main():
    torch.set_num_threads(1);wp.init();root=Path('output/ale')
    data=torch.load(root/'fv_state_dt8us.pt',map_location='cpu',weights_only=False)
    p,r=data['p'],data['r'];n=len(r);ground=float((p[:,1]-r).min())+.02
    d=max(1.,2*float((p[:,1]-ground).max())+.1)
    ghost=p.clone();ghost[:,1]-=d
    weight=r.square()+d*d-2*d*(p[:,1]-ground)
    assert bool((weight>0).all())
    reference=cpu_mesh(p,r,ground);tick=time.perf_counter()
    result={'cells':n,'ghost_count':n,'ground':ground,'offset':d,
            'scope':'Read-only geometric equivalence, no dynamics or contact-release claim.'}
    try:
        pp,rr=torch.cat((p,ghost)),torch.cat((r,weight.sqrt()))
        mesh=cpu_mesh(pp,rr,structure=structure_only(pp,rr))
        finite=bool(torch.isfinite(mesh.volume[:n]).all())
        crossing=(mesh.i<n)&(mesh.j>=n)
        area_threshold=256*torch.finfo(r.dtype).eps*torch.pi*r[mesh.i[crossing]].square()
        meaningful=mesh.area[crossing]>area_threshold
        centroid_error=(mesh.centroid[crossing,1]-ground).abs()
        wall=torch.zeros_like(r).index_add(0,mesh.i[crossing],mesh.area[crossing])
        result.update(finite=finite,
            volume_relative_error=float((mesh.volume[:n]-reference.volume).norm()/reference.volume.norm()),
            cap_area_relative_error=float((mesh.cap_area[:n]-reference.cap_area).norm()/reference.cap_area.norm()),
            ground_area_relative_error=float((wall-reference.ground_area).norm()/reference.ground_area.norm()),
            total_volume_difference_m3=float(mesh.volume[:n].sum()-reference.volume.sum()),
            wall_faces=int(crossing.sum()),
            roundoff_area_faces=int((~meaningful).sum()),
            roundoff_faces_total_area_m2=float(mesh.area[crossing][~meaningful].sum()),
            maximum_floor_centroid_error_m=float(centroid_error.max()),
            maximum_resolved_floor_centroid_error_m=float(centroid_error[meaningful].max()),
            area_weighted_floor_centroid_error_m=float((centroid_error*mesh.area[crossing]).sum()/mesh.area[crossing].sum()))
        result['passed']=finite and max(result[key] for key in ['volume_relative_error','cap_area_relative_error','ground_area_relative_error'])<1e-8 and result['maximum_resolved_floor_centroid_error_m']<1e-8
        # A nonconstant height field introduces genuine planar free faces,
        # including cross-pair faces. Check geometry's complete derivative
        # before proposing ANY mechanical update for these extra coordinates.
        center=reference.ground_centroid[reference.ground_area.argmax()]
        radial=(p[:,[0,2]]-center[[0,2]]).square().sum(1)
        profile=torch.exp(-radial/.04**2)
        height=ground+.002*profile
        variable_weight=r.square()+d*d-2*d*(p[:,1]-height)
        rr=torch.cat((r,variable_weight.sqrt()))
        variable=cpu_mesh(pp,rr,structure=structure_only(pp,rr))
        rate=torch.cat((torch.zeros_like(r),d*profile/variable_weight.sqrt()))
        predicted=variable.volume_rate(torch.zeros_like(pp),rate)[:n]
        crossing=(variable.i<n)&(variable.j>=n)
        area_threshold=256*torch.finfo(r.dtype).eps*torch.pi*r[variable.i[crossing]].square()
        meaningful=variable.area[crossing]>area_threshold
        height_above_floor=variable.centroid[crossing,1]-ground
        derivative=[]
        for h in [1e-6,1e-7]:
            rp=torch.cat((r,(variable_weight+2*d*h*profile).sqrt()))
            rm=torch.cat((r,(variable_weight-2*d*h*profile).sqrt()))
            plus=cpu_mesh(pp,rp,structure=structure_only(pp,rp))
            minus=cpu_mesh(pp,rm,structure=structure_only(pp,rm))
            measured=(plus.volume[:n]-minus.volume[:n])/(2*h)
            derivative.append({'height_increment_scale_m':h,
                'volume_derivative_relative_error':float((measured-predicted).norm()/predicted.norm())})
        result['variable_boundary_geometry']={
            'maximum_height_above_floor_m':float((height-ground).max()),
            'boundary_faces':int(crossing.sum()),
            'cross_pair_faces':int(((variable.j[crossing]-n)!=variable.i[crossing]).sum()),
            'minimum_boundary_centroid_height_above_floor_m':float(height_above_floor.min()),
            'minimum_resolved_boundary_centroid_height_above_floor_m':float(height_above_floor[meaningful].min()),
            'roundoff_area_faces':int((~meaningful).sum()),
            'roundoff_faces_total_area_m2':float(variable.area[crossing][~meaningful].sum()),
            'below_floor_centroid_faces':[
                {'real_cell':int(variable.i[k]),'ghost_owner':int(variable.j[k]-n),
                 'area_m2':float(variable.area[k]),'centroid_y_m':float(variable.centroid[k,1]),
                 'height_error_m':float(variable.centroid[k,1]-ground)}
                for k in torch.nonzero(crossing&(variable.centroid[:,1]<ground-1e-10)).flatten().tolist()],
            'volume_derivative_checks':derivative,
            'no_mechanical_state_or_contact_rule_implemented':True}
        result['passed']=result['passed'] and min(row['volume_derivative_relative_error'] for row in derivative)<1e-5 and bool((height_above_floor[meaningful]>=-1e-10).all())
    except Exception as error:
        result.update(passed=False,error=repr(error))
    result['seconds']=time.perf_counter()-tick
    (root/'hyperelastic_ghost_plane_geometry_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
