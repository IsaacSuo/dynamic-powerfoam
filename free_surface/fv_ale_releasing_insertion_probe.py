"""Read-only recut/continuity audit at a supposedly redundant floor insertion."""
import json
from pathlib import Path
import torch
import warp as wp
from free_surface.fv_ale_compatible_mesh_audit import restored
from free_surface.fv_ale_planar_boundary import Planes,geometry_cpu
from free_surface.fv_ale_planar_floor_reuse import geometry_cpu as reuse
from free_surface.fv_ale_energy_ledger import mechanical


def main():
    torch.set_num_threads(1);wp.init();root=Path('output/ale');tag='hyperelastic_releasing_hiddenfix80ms'
    data=torch.load(root/(tag+'.pt'),weights_only=False,map_location='cpu')
    report=json.loads((root/(tag+'.json')).read_text())
    sol,state,ops,b=restored(data,ground=data['ground'])
    with torch.no_grad():
        ops.rhs_gpu(b)
        total,V,v,rho,pressure,stress,sigma,sound,valid=ops.primitive(state.q,state.mesh.volume,sol.groups)
        cells=torch.nonzero(ops.wall_state(b,v,sigma,rho*sound)[5]).flatten().tolist()
        assert cells
        old=state.mesh.planes;new=Planes.floor(state.p,state.r,cells,sol.ground)
        planes=Planes(*(torch.cat((getattr(old,key),getattr(new,key))) for key in ['owner','d','a','b']))
        before=sol.group_sum(state.mesh.volume)[sol.groups];rows=[]
        for name,cut in [('explicit',lambda pl:geometry_cpu(state.p,state.r,pl,sol.ground)),
                         ('independent_hull',lambda pl:geometry_cpu(state.p,state.r,pl,sol.ground,full_hull=True)),
                         ('reuse_floor',lambda pl:reuse(state.p,state.r,pl,sol.ground))]:
            for height in [0.,1e-14,1e-13,1e-12,1e-11,1e-10,1e-9,1e-8,1e-7]:
                pl=planes.clone();pl.b[len(old.owner):]-=height
                try:
                    mesh=cut(pl);after=sol.group_sum(mesh.volume)[sol.groups]
                    relative=(after-before)/before.clamp(min=1e-300);cell=int(relative.abs().argmax())
                    rows.append({'method':name,'lift_m':height,'maximum_group_relative_volume_change':float(relative.abs().max()),
                        'worst_cell':cell,'worst_signed_change':float(relative[cell]),
                        'mechanical_energy_change_J':float(mechanical(sol,state.q,mesh.volume)-mechanical(sol,state.q,state.mesh.volume))})
                except RuntimeError as e:rows.append({'method':name,'lift_m':height,'error':str(e)})
        recorded=report['final']['elastic_energy_joule']+report['final']['kinetic_energy_joule']
        result={'time':data['time'],'requested_owners':cells,'prior_planes':len(old.owner),
            'recut_existing_mechanical_minus_recorded_J':float(mechanical(sol,state.q,state.mesh.volume))-recorded,'rows':rows}
    (root/'hyperelastic_releasing_insertion_probe.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
