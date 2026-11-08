"""Read-only diagnosis of a repeated floor activation at the saved full state."""
import json
from pathlib import Path
import torch
import warp as wp
from free_surface.fv_ale_compatible_mesh_audit import restored


def main():
    torch.set_num_threads(1);wp.init();root=Path('output/ale')
    data=torch.load(root/'hyperelastic_releasing_impact80ms.pt',weights_only=False,map_location='cpu')
    sol,state,ops,b=restored(data,ground=data['ground'])
    with torch.no_grad():
        w,dr,dq,packed=ops.rhs_gpu(b)
        diagnostics=dict(zip(ops.rhs_keys,packed.tolist()))
        total,V,v,rho,pressure,stress,sigma,sound,valid=ops.primitive(state.q,state.mesh.volume,sol.groups)
        cache=ops._compatible;pl=cache['planar'];o=b['plane_owner'];n=b['plane_n'];i=b['plane_i'];j=b['plane_j']
        rows=[]
        for index in torch.nonzero(o==99).flatten().tolist():
            face=j==index
            rows.append({'plane_index':index,'a':b['plane_a'][index].tolist(),'b':float(b['plane_b'][index]),
                'floor':sol.ground,'active':bool(pl['active'][index]),'bdot':float(pl['bdot'][index]),
                'generalized_stress_force':float(pl['stress_force'][index]),'denominator':float(pl['denominator'][index]),
                'faces':[{'i':int(k),'area':float(A),'normal':nn.tolist(),'centroid':c.tolist()}
                    for k,A,nn,c in zip(i[face],b['plane_A'][face],n[face],b['plane_c'][face])]})
        cell=99;center=b['affine_centers'][cell];G=cache['gradient'][cell]
        wall_c=pl['floor_c'][cell];facev=v[cell]+G@(wall_c-center)
        result={'time':data['time'],'tensile_count':diagnostics['wall_tensile_prediction_cells'],
            'cell':cell,'original_floor_area':float(b['floor'][cell]),'effective_floor_area':float(pl['floor'][cell]),
            'normal_stress':float(sigma[cell,1,1]),'reconstructed_floor_vy':float(facev[1]),
            'floor_pstar':float(-sigma[cell,1,1]-rho[cell]*sound[cell]*facev[1]),'planes':rows}
        from free_surface.fv_ale_ghost_plane_geometry_check import structure_only
        structure=structure_only(state.p,state.r)
        hidden_faces=structure['hidden'][state.mesh.plane_i]
        result['hidden_real_cell_boundary_faces']=int(hidden_faces.sum())
        result['hidden_real_cell_boundary_area']=float(state.mesh.plane_area[hidden_faces].sum())
        result['passed_hidden_face_regression']=not bool(hidden_faces.any())
        assert result['passed_hidden_face_regression'],result
        for row in rows:
            for face in row['faces']:
                x=state.p.new_tensor(face['centroid']);power=(state.p-x).square().sum(1)-state.r.square()
                k=face['i'];neighbors=structure['nbrs'][k];neighbors=neighbors[neighbors>=0]
                face.update(power_gap_to_owner=float(power[k]-power[99]),
                    hidden_in_real_diagram=bool(structure['hidden'][k]),
                    owner_is_real_neighbor=bool((neighbors==99).any()),
                    minimum_power_owner=int(power.argmin()),
                    maximum_neighbor_inequality_violation=float((power[k]-power[neighbors]).max()) if len(neighbors) else None)
        from free_surface.fv_ale_planar_boundary import geometry_cpu
        try:
            full=geometry_cpu(state.p,state.r,state.mesh.planes,sol.ground,full_hull=True)
            idx=torch.nonzero(full.planes.owner==99).flatten()[0]
            face=full.plane_j==idx
            result['independent_full_hull']={'volume_relative_difference':float((full.volume-state.mesh.volume).norm()/full.volume.norm()),
                'faces':[{'i':int(k),'area':float(A),'centroid':c.tolist()} for k,A,c in
                zip(full.plane_i[face],full.plane_area[face],full.plane_centroid[face])]}
            def fields(mesh):
                index=mesh.plane_i*len(mesh.planes.owner)+mesh.plane_j
                value=torch.cat((mesh.plane_area[:,None],mesh.plane_area[:,None]*mesh.plane_normal,
                                 mesh.plane_area[:,None]*mesh.plane_centroid),1)
                return value.new_zeros((len(mesh.r)*len(mesh.planes.owner),7)).index_add_(0,index,value)
            a=fields(state.mesh);bfull=fields(full)
            error=float((a-bfull).norm()/bfull.norm())
            result['independent_full_hull']['boundary_area_normal_moment_relative_error']=error
            assert error<1e-10,error
        except RuntimeError as error:result['independent_full_hull']={'error':str(error)}
    (root/'hyperelastic_releasing_duplicate_probe.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
