"""Frozen full-foam comparison of ALE mesh motion, with energy identities."""
import json,time
from pathlib import Path
import torch
from free_surface.fv_ale_hyperelastic import HyperelasticALE,HyperelasticDeviceOps
from free_surface.fv_ale_solid import State
from free_surface.fv_ale_mesh_motion import fit_motion
from free_surface.fv_ale_energy_ledger import rate


@torch.no_grad()
def main():
    root=Path('output/ale');data=torch.load(root/'hyperelastic_release_blend001_compatible.pt',weights_only=False)
    m=data['material'];sol=HyperelasticALE(data['p'],data['r'],m['E'],m['nu'],m['ys'],groups=data['groups'],
          ground=-10.,gravity=(0.,0.,0.),contact_model='plane',interface_dissipation=.01,cap_dissipation=.01)
    state=State(sol.initial.p,sol.initial.r,data['q'].cuda(),sol.initial.mesh)
    mesh=state.mesh;i,j,n=mesh.i,mesh.j,mesh.normal
    primitive=sol.device_ops().primitive(state.q,mesh.volume,sol.groups)
    v,rho,sigma,c=primitive[2],primitive[3],primitive[6],primitive[7]
    ti=(sigma[i]@n[:,:,None])[:,:,0];tj=(sigma[j]@n[:,:,None])[:,:,0]
    us,_=sol.device_ops().acoustic_interface(v,rho*c,ti,tj,i,j)
    base=sol.rhs(state)
    b={'p':state.p,'r':state.r,'i':i,'j':j,'A':mesh.area,'n':n,'c':mesh.centroid,
       'groups':sol.groups,'surface':mesh.surface,'floor':mesh.ground_area,'V':mesh.volume}
    results=[]
    for iterations,radius in [(0,False),(12,False),(24,False),(24,True),(48,True)]:
        sol.mesh_fit_iterations=iterations;sol.mesh_fit_radius=radius;sol._device_ops=HyperelasticDeviceOps(sol)
        rhs=sol.rhs(state);ledger=rate(sol,state,rhs,fd=True)
        torch.cuda.synchronize();begin=time.perf_counter()
        for _ in range(5):sol.rhs(state)
        torch.cuda.synchronize();elapsed=(time.perf_counter()-begin)/5
        mw,_=mesh.mesh_flux(rhs[0],rhs[1]);slip=mesh.area*(us*n).sum(1)-mw
        ext=sol.groups[i]!=sol.groups[j]
        rms=float((slip[ext].square()/mesh.area[ext]).sum().div(mesh.area[ext].sum()).sqrt())
        boundary=mesh.surface|(mesh.ground_area>0)
        change=max(float((rhs[0]-base[0])[boundary].abs().max()),float((rhs[1]-base[1])[boundary].abs().max()))
        assert change<1e-11,change
        entry={'iterations':iterations,'radius':radius,'rhs_seconds':elapsed,'normal_slip_rms_m_s':rms,
               'boundary_velocity_change':change,'ledger':ledger}
        if iterations:
            _,_,diag=fit_motion(b,base[0],base[1],us,iterations,.1,radius)
            entry['fit_diagnostics']=diag.tolist()
        results.append(entry);print(json.dumps(entry),flush=True)
    (root/'hyperelastic_mesh_motion_check.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':main()
