"""Independent group energy and geometric-rate audit at a stopped full state."""
import argparse
import json
from pathlib import Path
import torch
import warp as wp
from free_surface.fv_ale_compatible_mesh_audit import restored
from free_surface.fv_ale_hyperelastic_work_replay import group_energy,group_power,volume_rate
from free_surface.fv_ale_precise_geometry_check import cpu_mesh
from free_surface.fv_ale_hyperelastic import HyperelasticALE


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('tag');args=parser.parse_args()
    torch.set_num_threads(1);wp.init();root=Path('output/ale')
    data=torch.load(root/(args.tag+'.pt'),map_location='cpu',weights_only=False)
    sol,state,ops,b=restored(data)
    sol.mesh_fit_iterations=data['mesh_fit']['iterations'];sol.mesh_fit_radius=data['mesh_fit']['radius']
    sol.mesh_fit_regularization=data['mesh_fit']['regularization']
    w,dr,dq,packed=ops.rhs_gpu(b);rhs=(w,dr,dq,dict(zip(ops.rhs_keys,packed.tolist())))
    roots,energy=group_energy(sol,state.q,state.mesh.volume)
    total=sol.group_sum(state.q)[roots];thermal=total[:,4]-energy
    power=group_power(sol,state,rhs);thermal_rate=sol.group_sum(dq)[roots,4]-power
    bad=thermal<0;Vdot=volume_rate(state,rhs);groupVdot=sol.group_sum(Vdot)[roots]
    rows=[]
    for h in [1e-7,1e-8,1e-9]:
        plus=cpu_mesh(state.p+h*w,state.r+h*dr)
        minus=cpu_mesh(state.p-h*w,state.r-h*dr)
        HyperelasticALE.normalize_volume_roundoff(sol,plus);HyperelasticALE.normalize_volume_roundoff(sol,minus)
        trueVdot=sol.group_sum((plus.volume-minus.volume)/(2*h))[roots]
        _,Ep=group_energy(sol,state.q+h*dq,plus.volume)
        _,Em=group_energy(sol,state.q-h*dq,minus.volume)
        measured=(Ep-Em)/(2*h);measured_thermal_rate=sol.group_sum(dq)[roots,4]-measured
        rows.append({'epsilon_seconds':h,'group_volume_rate_relative_error':float((trueVdot-groupVdot).norm()/groupVdot.norm()),
            'group_mechanical_rate_relative_error':float((measured-power).norm()/power.norm()),
            'negative_groups_measured_thermal_rate_W':float(measured_thermal_rate[bad].sum()),
            'max_group_power_difference_W':float((measured-power).abs().max())})
    order=torch.argsort(thermal);top=[]
    for k in order[:8]:
        top.append({'group':int(roots[k]),'thermal_energy_J':float(thermal[k]),'thermal_rate_W':float(thermal_rate[k]),
                    'mass_kg':float(total[k,0]),'volume_m3':float(sol.group_sum(state.mesh.volume)[roots[k]])})
    result={'tag':args.tag,'time':data['time'],'negative_remainder_J':float(-thermal[bad].sum()),
            'negative_groups_thermal_rate_W':float(thermal_rate[bad].sum()),
            'motion_diagnostics':{key:value for key,value in rhs[3].items() if key.startswith('mesh_fit_')},
            'maximum_site_speed_m_s':float(w.norm(dim=1).max()),'maximum_radius_speed_m_s':float(dr.abs().max()),
            'negative_interface_mixing_heat_W':float(ops._compatible['mixing_heat'].clamp(max=0).sum()),
            'geometric_derivative_checks':rows,'lowest_remainders':top}
    (root/(args.tag+'_failure_audit.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
