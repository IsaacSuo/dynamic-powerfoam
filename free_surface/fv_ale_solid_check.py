"""Mechanics checks on the complete real foam before a contact trial."""
import argparse,json,math
from pathlib import Path
import numpy as np
import torch
import warp as wp
from free_surface.fv_ale_solid import SolidALE,State,acoustic_face
from free_surface.fv_ale_derefine import dissolve


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--foam',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--surface-motion',choices=['radial','least-squares','coupled'],default='least-squares');a=p.parse_args()
    wp.init();g=np.load(a.foam/'dem_exact_single/geometry.npz');cells=np.load(a.foam/'dem_exact_single/cells.npz')
    material=json.loads((a.foam/'plasticine_scenes.json').read_text())['scenes']['0']
    solver=SolidALE(g['points'],cells['radii'],material['E'],material['nu'],material['ys'],gravity=(0,0,0),ground=None,friction=0,surface_motion=a.surface_motion)
    state=solver.set_velocity(solver.initial,[.2,-.3,.1]); report={'foam':str(a.foam),'cells':len(state.p),'checks':{}}
    # Uniform translation is an exact material motion independent of mesh shape.
    vp,vr,dq,diag=solver.rhs(state)
    report['translation_rhs_max']=float(dq.abs().max());report['translation_radius_rate_max']=float(vr.abs().max())
    dt=.15*diag['dt_scale'];m0=state.q[:,0].sum();en0=state.q[:,4].sum()
    moved,stats=solver.advance(state,dt,(vp,vr,dq,diag))
    report['translation_velocity_error']=float((moved.q[:,1:4]/moved.q[:,:1]-state.q[:,1:4]/state.q[:,:1]).abs().max())
    report['translation_density_error']=float((moved.q[:,0]/moved.mesh.volume/1000-1).abs().max())
    report['checks']['uniform_translation']=report['translation_velocity_error']<1e-8 and report['translation_density_error']<1e-7
    # Nonuniform cell stresses and velocities exercise pressure, shear and ALE
    # transfer. A free exterior contributes zero resultant force and zero work.
    phase=(state.p-state.p.mean(0))/.03
    velocity=.03*torch.stack((torch.sin(phase[:,1]),torch.cos(phase[:,2]),torch.sin(phase[:,0])),1)
    q=state.q.clone();q[:,1:4]=q[:,:1]*velocity;q[:,4]=q[:,0]*(solver.e0+.5*(velocity*velocity).sum(1))
    s=torch.zeros((len(q),3,3),dtype=q.dtype,device=q.device)
    s[:,0,1]=s[:,1,0]=.2*solver.ys*torch.sin(phase[:,0]);s[:,0,0]=.1*solver.ys;s[:,1,1]=-.1*solver.ys
    q[:,5:14]=(q[:,:1,None]*s).reshape(-1,9)
    nonuniform=State(state.p,state.r,q,state.mesh)
    vp,vr,dq,diag=solver.rhs(nonuniform)
    report['closed_domain_mass_rate']=float(dq[:,0].sum())
    report['internal_force_sum']=dq[:,1:4].sum(0).cpu().tolist()
    report['internal_energy_rate']=float(dq[:,4].sum())
    report['cap_integrated_flux_defect']=diag['cap_integrated_flux_defect']
    report['checks']['pairwise_conservation']=abs(report['closed_domain_mass_rate'])<1e-10 and max(abs(x) for x in report['internal_force_sum'])<1e-8 and abs(report['internal_energy_rate'])<1e-8
    # Rotating the ENTIRE geometry/state must rotate forces and preserve radial
    # rates, including the spherical-cap closure. This tests actual assembled RHS.
    axis=torch.tensor([.3,.4,.5],dtype=q.dtype,device=q.device);axis/=axis.norm();angle=.47
    K=torch.zeros((3,3),dtype=q.dtype,device=q.device)
    K[0,1]=-axis[2];K[0,2]=axis[1];K[1,0]=axis[2];K[1,2]=-axis[0];K[2,0]=-axis[1];K[2,1]=axis[0]
    R=torch.eye(3,dtype=q.dtype,device=q.device)+math.sin(angle)*K+(1-math.cos(angle))*(K@K)
    centre=state.p.mean(0);pr=(state.p-centre)@R.T+centre;qr=q.clone()
    qr[:,1:4]=q[:,1:4]@R.T;qr[:,5:14]=(R@q[:,5:14].reshape(-1,3,3)@R.T).reshape(-1,9)
    rotated=State(pr,state.r,qr,solver.cut(pr,state.r));vpr,vrr,dqr,dr=solver.rhs(rotated)
    report['rotation_force_relative_error']=float((dqr[:,1:4]-dq[:,1:4]@R.T).norm()/dq[:,1:4].norm())
    report['rotation_stress_rate_relative_error']=float((dqr[:,5:14].reshape(-1,3,3)-R@dq[:,5:14].reshape(-1,3,3)@R.T).norm()/dq[:,5:14].norm())
    report['rotation_radius_rate_relative_error']=float((vrr-vr).norm()/vr.norm())
    report['checks']['rotation_covariance']=max(report[k] for k in ('rotation_force_relative_error','rotation_stress_rate_relative_error','rotation_radius_rate_relative_error'))<1e-7
    # Large trial shear must return to exactly the yield surface, preserving
    # total energy, with nonnegative plastic work and traceless symmetric stress.
    large=q.clone();large[:,5:14]*=100;returned=solver.plastic_return(large,state.mesh)
    sr=returned[:,5:14].reshape(-1,3,3)/returned[:,:1,None]
    vm=torch.sqrt(1.5*(sr*sr).sum((1,2)))
    report['returned_von_mises_max']=float(vm.max());report['yield_stress']=solver.ys
    report['plastic_work_min']=float(returned[:,15].min())
    report['plastic_energy_channel_change']=float((returned[:,4]-large[:,4]).abs().max())
    report['checks']['plastic_return']=float(vm.max())<=solver.ys*(1+1e-12) and report['plastic_work_min']>=0 and report['plastic_energy_channel_change']==0
    # Advance an actual nonuniform elastic pulse, recutting every stage.
    mass0=float(nonuniform.q[:,0].sum());energy0=float(nonuniform.q[:,4].sum());start=nonuniform
    dtmin=math.inf;dtmax=0
    for _ in range(20):
        rhs=solver.rhs(start);dt=.1*rhs[3]['dt_scale'];dtmin=min(dtmin,dt);dtmax=max(dtmax,dt)
        start,_=solver.advance(start,dt,rhs)
    report['pulse_mass_error']=(float(start.q[:,0].sum())-mass0)/mass0
    report['pulse_total_energy_error']=(float(start.q[:,4].sum())-energy0)/energy0
    report['pulse_dt_range']=[dtmin,dtmax]
    report['pulse_density_range']=[float((start.q[:,0]/start.mesh.volume).min()),float((start.q[:,0]/start.mesh.volume).max())]
    report['pulse_cap_mismatch']=solver.cap_velocity_mismatch(start)
    report['checks']['twenty_step_pulse']=abs(report['pulse_mass_error'])<1e-10 and abs(report['pulse_total_energy_error'])<1e-10
    # Derefinement on this same full, nonuniform, physically evolved foam.
    # Reject exposed removal, then verify buried deletion's geometric remap
    # and continue actual mechanics with a changed number of cells.
    try:
        dissolve(solver,start,int(torch.nonzero(start.mesh.surface)[0]))
        exposed_rejected=False
    except RuntimeError as exc:
        exposed_rejected='exposed-cell removal' in str(exc)
    k=int(torch.where(~start.mesh.surface,start.mesh.volume,torch.full_like(start.r,float('inf'))).argmin())
    dissolved,keep,info=dissolve(solver,start,k)
    rhs=solver.rhs(dissolved)
    continued,_=solver.advance(dissolved,.1*rhs[3]['dt_scale'],rhs)
    info['continued_mass_error']=(float(continued.q[:,0].sum())-mass0)/mass0
    info['continued_energy_error']=(float(continued.q[:,4].sum())-energy0)/energy0
    report['derefine_check']=info
    report['checks']['conservative_derefinement']=exposed_rejected and info['conservation_relative_max']<1e-12 and info['uniform_density_error_max']<1e-7 and abs(info['continued_mass_error'])<1e-12 and abs(info['continued_energy_error'])<1e-12
    report['passed']=all(report['checks'].values());a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    if not report['passed']:raise SystemExit(2)

if __name__=='__main__':main()
