"""Exact floor geometry, geometric flux, and real-foam planar boundary checks."""
import argparse,json,math
from pathlib import Path
import numpy as np
import torch,warp as wp
from free_surface.fv_ale_ground import GroundClip,plane_boundary
from free_surface.fv_ale_solid import SolidALE,State


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--foam',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--penetration',type=float,default=.001)
    ap.add_argument('--deep-geometry',action='store_true',help='clip the full foam at several additional heights; geometry only, no initial mass deletion')
    ap.add_argument('--regression-snapshot',type=Path);a=ap.parse_args()
    wp.init();dev='cuda';dt=torch.float64;single=[]
    for h in [.3,0.,-.2,1.2,-1.2]:
        p=torch.tensor([[0.,h,0.]],device=dev,dtype=dt);r=torch.ones(1,device=dev,dtype=dt)
        mesh=GroundClip(0.,dev).geometry(p,r)
        hh=max(-1.,min(1.,h));volume=math.pi*(2/3+hh-hh**3/3)
        area=math.pi*(1-h*h) if abs(h)<1 else 0.
        cap=2*math.pi*(1+hh)
        single.append({'centre_height':h,'volume_error':abs(float(mesh.volume[0])-volume),
            'ground_area_error':abs(float(mesh.ground_area[0])-area),
            'cap_area_error':abs(float(mesh.cap_area[0])-cap)})
    geometry=np.load(a.foam/'dem_exact_single/geometry.npz');cells=np.load(a.foam/'dem_exact_single/cells.npz')
    material=json.loads((a.foam/'plasticine_scenes.json').read_text())['scenes']['0']
    p=torch.tensor(geometry['points'],device=dev,dtype=dt);r=torch.tensor(cells['radii'],device=dev,dtype=dt)
    # Real full foam at a shallow floor intersection, not a new simulation scene.
    p[:,1]-=(p[:,1]-r).min()+a.penetration
    sol=SolidALE(p,r,material['E'],material['nu'],material['ys'],ground=0.,gravity=(0,0,0),contact_model='plane')
    s=sol.set_velocity(sol.initial,[.2,-.1,.05]);mesh=s.mesh
    closure=torch.zeros_like(p).index_add(0,mesh.i,mesh.area[:,None]*mesh.normal).index_add(0,mesh.j,-mesh.area[:,None]*mesh.normal)
    closure+=mesh.cap_normal_integral+mesh.ground_area[:,None]*mesh.ground_normal
    closure_error=float(closure.norm(dim=1).max())
    phase=p/.03;vp=.01*torch.sin(phase);vr=.001*torch.cos(phase[:,0])
    analytical=mesh.volume_rate(vp,vr)
    eps=1e-5
    plus=sol.cut(p+eps*vp,r+eps*vr);minus=sol.cut(p-eps*vp,r-eps*vr)
    fd=(plus.volume-minus.volume)/(2*eps)
    volume_rate_error=float((fd-analytical).norm()/analytical.norm())
    # Reset the original geometry; production geometry buffers are reusable.
    s=State(p,r,s.q,sol.cut(p,r));rhs=sol.rhs(s)
    _,vel,rho,_,_,sigma,c=sol.primitive(s)
    F,ub,heat,info=plane_boundary(s.mesh,vel,sigma,rho*c,sol.friction)
    Fs,us,hs,slip_info=plane_boundary(s.mesh,vel,sigma,rho*c,.01)
    active=s.mesh.ground_area>0
    stick_error=float(ub[active][:,[0,2]].norm(dim=1).max())
    coulomb_error=float((Fs[:,[0,2]].norm(dim=1)-.01*Fs[:,1]).abs().max())
    end,stats=sol.advance(s,.1*rhs[3]['dt_scale'],rhs)
    force_error=float((rhs[2][:,1:4].sum(0)-F.sum(0)).norm())
    energy_error=float((end.q[:,4]-s.q[:,4]).sum())-stats['source_work']
    mass_error=float((end.q[:,0].sum()-s.q[:,0].sum())/s.q[:,0].sum())
    # Exact tangent translation while touching a stationary plane.
    tangent=sol.set_velocity(State(p,r,s.q.clone(),sol.cut(p,r)),[.2,0.,.05])
    trhs=sol.rhs(tangent);te,_=sol.advance(tangent,.1*trhs[3]['dt_scale'],trhs)
    trho=sol.primitive(te)[2];translation_error=float((trho/sol.rho0-1).abs().max())
    report={'foam':str(a.foam),'test_penetration':a.penetration,'single_sphere_analytic_checks':single,
        'closure_error_m2':closure_error,'volume_rate_relative_error':volume_rate_error,
        'wall_force_balance_error_N':force_error,'step_energy_error_J':energy_error,'step_mass_error':mass_error,
        'tangent_translation_density_error':translation_error,'plane_boundary':info,
        'geometry':s.mesh.ground_geometry_info,
        'friction_checks':{'stick_velocity_error_m_s':stick_error,'sliding_coulomb_force_error_N':coulomb_error,'sliding_cells':slip_info['wall_sliding_cells'],'minimum_friction_heat_rate_W':float(hs.min())},
        'checks':{'single_sphere':max(max(x[k] for k in ['volume_error','ground_area_error','cap_area_error']) for x in single)<1e-10,
            'closure':closure_error<1e-10,'geometric_flux':volume_rate_error<1e-5,
            'force_balance':force_error<1e-8,'energy_balance':abs(energy_error)<1e-8,
            'mass_balance':abs(mass_error)<1e-12,'tangent_translation':translation_error<1e-7,
            'impermeable_ground':info['wall_normal_velocity_max']==0.,'compressive_contact':info['wall_tensile_prediction_cells']==0,
            'friction_stick_slip':stick_error<1e-12 and coulomb_error<1e-10 and slip_info['wall_sliding_cells']>0 and float(hs.min())>=-1e-12}}
    if a.deep_geometry:
        samples=[]
        for height in [.03,.08,.15,.30]:
            clip=GroundClip(height,dev);g=clip.geometry(p,r)
            cl=torch.zeros_like(p).index_add(0,g.i,g.area[:,None]*g.normal).index_add(0,g.j,-g.area[:,None]*g.normal)
            cl+=g.cap_normal_integral+g.ground_area[:,None]*g.ground_normal
            rate=g.volume_rate(vp,vr);eps=1e-5
            gp=clip.geometry(p+eps*vp,r+eps*vr);gm=clip.geometry(p-eps*vp,r-eps*vr)
            error=float(((gp.volume-gm.volume)/(2*eps)-rate).norm()/rate.norm())
            samples.append({'height':height,'floor_cells':int((g.ground_area>0).sum()),
                'hidden_cells':int(g.hidden.sum()),'closure_m2':float(cl.norm(dim=1).max()),
                'volume_rate_relative_error':error,'min_volume':float(g.volume.min())})
        report['deep_geometry']=samples
        report['checks']['deep_clipping']=all(x['closure_m2']<1e-10 and x['volume_rate_relative_error']<1e-5 and x['min_volume']>=-1e-14 for x in samples)
    if a.regression_snapshot:
        data=torch.load(a.regression_snapshot,map_location='cpu',weights_only=False)
        pp=data['p'].to(dev);rr=data['r'].to(dev);H=data['ground']
        clipped=GroundClip(H,dev);good=clipped.geometry(pp,rr)
        from free_surface.recut import Recut
        from free_surface.geo_pipe import GeoPipe
        raw=Recut(len(rr),dev);raw.pipe=GeoPipe(len(rr),dev);raw.pipe.stable_area=True
        uncut,_=raw.geometry(pp,rr)
        volume_increase=float((good.volume-uncut.V).max())
        closure=torch.zeros_like(pp).index_add(0,good.i,good.area[:,None]*good.normal).index_add(0,good.j,-good.area[:,None]*good.normal)
        closure+=good.cap_normal_integral+good.ground_area[:,None]*good.ground_normal
        k=6623
        velocity=.01*torch.sin(pp/.03);radial=.001*torch.cos(pp[:,0]/.03)
        rate=good.volume_rate(velocity,radial);eps=1e-5
        plus=clipped.geometry(pp+eps*velocity,rr+eps*radial)
        minus=clipped.geometry(pp-eps*velocity,rr-eps*radial)
        rate_error=float(((plus.volume-minus.volume)/(2*eps)-rate).norm()/rate.norm())
        report['degenerate_arc_regression']={'snapshot':str(a.regression_snapshot),
            'previously_corrupted_cell':k,'direct_clipped_volume':float(good.volume[k]),
            'uncut_volume':float(uncut.V[k]),'max_clipping_volume_increase_m3':volume_increase,
            'closure_error_m2':float(closure.norm(dim=1).max()),'volume_rate_relative_error':rate_error}
        report['checks']['clipping_cannot_increase_volume']=volume_increase<1e-14
        report['checks']['regression_closure']=float(closure.norm(dim=1).max())<1e-10
        report['checks']['regression_volume_rate']=rate_error<1e-5
    report['passed']=all(report['checks'].values());a.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    if not report['passed']:raise SystemExit(2)


if __name__=='__main__':main()
