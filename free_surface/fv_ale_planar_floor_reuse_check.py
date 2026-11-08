"""Full-state equivalence of exact floor relabelling and explicit generators."""
import json
from pathlib import Path
import time
import torch
import warp as wp
from free_surface.fv_ale_compatible_mesh_audit import restored
from free_surface.fv_ale_planar_boundary import Planes,geometry_cpu as explicit,values
from free_surface.fv_ale_planar_floor_reuse import geometry_cpu as reused


def main():
    torch.set_num_threads(1);wp.init();root=Path('output/ale');rows=[]
    for tag in ['hyperelastic_acoustic_impact50ms','hyperelastic_releasing_impact80ms']:
        data=torch.load(root/(tag+'.pt'),weights_only=False,map_location='cpu')
        sol,state,ops,_=restored(data,ground=data['ground'])
        planes=(state.mesh.planes if hasattr(state.mesh,'planes') else
            Planes.floor(state.p,state.r,[2919,6333,2979],sol.ground))
        def evaluate(mesh):
            b={'q':state.q,'V':mesh.volume,'groups':sol.groups,'p':state.p,'r':state.r,
                'i':mesh.i,'j':mesh.j,'A':mesh.area,'n':mesh.normal,'c':mesh.centroid,
                'N':mesh.cap_normal_integral,'cap':mesh.cap_area,'floor':mesh.ground_area,
                'floor_c':mesh.ground_centroid,'floor_n':mesh.ground_normal,**values(mesh)}
            mesh.cap_Q,mesh.surface,mesh.surface_area,_=ops.moments_gpu(b)
            b.update(Q=mesh.cap_Q,surface=mesh.surface);b.update(ops.extra_rhs_values(mesh))
            w,dr,dq,pack=ops.rhs_gpu(b);boundary=ops._compatible['planar']
            return [w.clone(),dr.clone(),dq.clone(),boundary['adot'].clone(),boundary['bdot'].clone()]
        start=time.perf_counter();a=explicit(state.p,state.r,planes,sol.ground);ta=time.perf_counter()-start
        start=time.perf_counter();b=reused(state.p,state.r,planes,sol.ground);tb=time.perf_counter()-start
        fa=evaluate(a);fb=evaluate(b)
        errors={key:float((getattr(a,key)-getattr(b,key)).norm()/getattr(a,key).norm().clamp(min=1e-300))
                for key in ['volume','cap_area','cap_normal_integral']}
        for key,x,y in zip(['site_velocity','radius_velocity','conserved_rhs','plane_adot','plane_bdot'],fa,fb):
            errors[key]=float((x-y).norm()/x.norm().clamp(min=1.))
        row={'source':tag,'planes':len(planes.owner),'explicit_cut_cpu_seconds':ta,
             'reused_cut_cpu_seconds':tb,'errors':errors,'reused_floor_faces':b.ground_geometry_info['exactly_reused_floor_faces']}
        rows.append(row);assert max(errors.values())<1e-8,row
    result={'passed':True,'rows':rows,'scope':'Full-state CPU geometry and mechanics; no whole-run speed claim'}
    (root/'hyperelastic_planar_floor_reuse_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
