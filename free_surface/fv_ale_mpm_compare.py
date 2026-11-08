"""Matched release using PowerSim's actual PhysGaussian MPM implementation.

Only its stress/return kernel is replaced in this process, to match the FV
neo-Hookean energy. P2G, grid update, G2P, F update and APIC are upstream code.
No changes are written to PowerSim or PhysGaussian. Free release: no collider,
gravity, plasticity, velocity damping or energy normalization during stepping.
"""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import warp as wp

POWERSIM=Path('/home/fangsuo/py/PowerSim')
ENGINE=POWERSIM/'third_party/PhysGaussian/mpm_solver_warp'
sys.path.insert(0,str(POWERSIM))
sys.path.insert(0,str(ENGINE))
from powersim.core.warp_compat import patch_warp_array_owner_kwarg
patch_warp_array_owner_kwarg()
import mpm_solver_warp as upstream
from warp_utils import MPMStateStruct, MPMModelStruct


@wp.kernel
def matched_stress(state: MPMStateStruct, model: MPMModelStruct, dt: float):
    p=wp.tid()
    if state.particle_selection[p]==0:
        F=state.particle_F_trial[p]
        state.particle_F[p]=F
        J=wp.determinant(F)
        B=F*wp.transpose(F)*wp.pow(J,-2.0/3.0)
        I=wp.identity(n=3,dtype=float)
        mu=model.mu[p]
        bulk=model.lam[p]+2.0*mu/3.0
        state.particle_stress[p]=mu*(B-wp.trace(B)/3.0*I)+0.5*bulk*(J*J-1.0)*I


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sample_union(p,r,spacing):
    """Exact ball-membership test at Cartesian quadrature points.

    Union of (power cell intersect ball) equals union of the balls. Boundary
    quadrature is approximate; raw volume discrepancy is reported explicitly.
    """
    origin=np.floor((p-r[:,None]).min(0)/spacing)*spacing
    shape=np.ceil(((p+r[:,None]).max(0)-origin)/spacing).astype(int)+1
    inside=np.zeros(tuple(shape),dtype=bool)
    for c,radius in zip(p,r):
        lo=np.maximum(np.ceil((c-radius-origin)/spacing-.5).astype(int),0)
        hi=np.minimum(np.floor((c+radius-origin)/spacing-.5).astype(int)+1,shape)
        axes=[origin[d]+(np.arange(lo[d],hi[d])+.5)*spacing-c[d] for d in range(3)]
        dist2=axes[0][:,None,None]**2+axes[1][None,:,None]**2+axes[2][None,None,:]**2
        inside[tuple(slice(a,b) for a,b in zip(lo,hi))] |= dist2<=radius*radius
    return origin+(np.argwhere(inside)+.5)*spacing


def material_check(solver,mu,bulk,rho0):
    from free_surface.fv_ale_hyperelastic_material import response
    Fs=torch.tensor([[[1.,0,0],[0,1,0],[0,0,1]],
                     [[1.5,0,0],[0,1.5**-.5,0],[0,0,1.5**-.5]],
                     [[.7,.4,0],[0,1.1,.2],[0,0,1.3]],
                     [[0.,-1.,0],[2.,0.,0],[0.,0.,.5]]],dtype=torch.float64)
    # Test exactly the float32 F and moduli used by the external solver.
    Fs=Fs.float().double();J=torch.linalg.det(Fs);B=Fs@Fs.transpose(1,2)/J[:,None,None]**(2/3)
    eig,R=torch.linalg.eigh(B);H=(R*eig.log()[:,None,:])@R.transpose(1,2)
    expected=response(H,rho0/J,mu,bulk,rho0)[2]*J[:,None,None]
    wp.to_torch(solver.mpm_state.particle_F_trial)[:len(Fs)].copy_(Fs.cuda().float())
    wp.synchronize()
    wp.launch(matched_stress,len(Fs),inputs=[solver.mpm_state,solver.mpm_model,0.],device='cuda:0')
    wp.synchronize()
    actual=wp.to_torch(solver.mpm_state.particle_stress)[:len(Fs)].cpu().double()
    err=float((actual-expected).norm()/expected.norm())
    # Near incompressibility amplifies FP32 determinant rounding into pressure.
    # Check shear independently and infer the determinant used by the pressure
    # term; a fixed relative stress tolerance would conflate the two effects.
    eye=torch.eye(3,dtype=actual.dtype)
    trace_actual=actual.diagonal(dim1=1,dim2=2).sum(1)
    trace_expected=expected.diagonal(dim1=1,dim2=2).sum(1)
    dev_actual=actual-trace_actual[:,None,None]*eye/3
    dev_expected=expected-trace_expected[:,None,None]*eye/3
    shear_error=float((dev_actual-dev_expected).norm()/dev_expected.norm())
    inferred_J=torch.sqrt(1+2*trace_actual/(3*bulk))
    determinant_error=float(((inferred_J-J)/J).abs().max())
    assert shear_error<2e-6 and determinant_error<8*torch.finfo(torch.float32).eps,(err,shear_error,determinant_error)
    print(json.dumps({'stress_check_relative_error':err,'shear_relative_error':shear_error,
                      'pressure_inferred_determinant_relative_error':determinant_error}),flush=True)
    return err


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dx',type=float,default=.02)
    parser.add_argument('--dt',type=float,default=2e-5)
    parser.add_argument('--until',type=float,default=.01)
    parser.add_argument('--ppc-axis',type=int,default=2)
    parser.add_argument('--stretch',type=float,default=1.5)
    parser.add_argument('--young',type=float,default=None);parser.add_argument('--poisson',type=float,default=None)
    parser.add_argument('--translation-check',action='store_true')
    parser.add_argument('--rotation-check',action='store_true')
    parser.add_argument('--tag',required=True)
    args=parser.parse_args()
    out=Path('output/ale');out.mkdir(exist_ok=True,parents=True)
    source=out/'fv_state_dt8us.pt';fv_end=out/'hyperelastic_release_roundoff.pt'
    saved=torch.load(source,map_location='cpu',weights_only=False)
    mass_exact=float(torch.load(fv_end,map_location='cpu',weights_only=False)['q'][:,0].sum())
    rho0=1000.;volume=mass_exact/rho0;m=saved['material'];E,nu=m['E'],m['nu']
    if args.young is not None:E=args.young
    if args.poisson is not None:nu=args.poisson
    if E<=0 or not -1<nu<.5:raise ValueError('invalid compressible elastic material parameters')
    mu=E/(2*(1+nu));bulk=E/(3*(1-2*nu))
    source_hash=digest(source);p=saved['p'].numpy();r=saved['r'].numpy()
    dx=args.dx;spacing=dx/args.ppc_axis
    cache=out/f'mpm_particles_{spacing:.8f}.npz'
    if cache.exists() and str(np.load(cache)['source_hash'])==source_hash:
        x_world=np.load(cache)['x']
    else:
        x_world=sample_union(p,r,spacing)
        np.savez(cache,x=x_world,source_hash=source_hash)
    n=len(x_world);raw_volume=n*spacing**3
    origin=np.floor(((p-r[:,None]).min(0)-.20)/dx)*dx
    grid_size=int(math.ceil((((p+r[:,None]).max(0)+.20-origin)/dx).max()))+1
    wp.init()
    hashes={str(ENGINE/f):digest(ENGINE/f) for f in ['mpm_solver_warp.py','mpm_utils.py','warp_utils.py']}
    solver=upstream.MPM_Simulator_WARP(n,n_grid=grid_size,grid_lim=grid_size*dx)
    x=torch.as_tensor(x_world-origin,dtype=torch.float32,device='cuda').contiguous()
    solver.load_initial_data_from_torch(x,torch.full((n,),volume/n,dtype=torch.float32),
                                        n_grid=grid_size,grid_lim=grid_size*dx)
    solver.set_parameters_dict({'material':'jelly','E':E,'nu':nu,'density':rho0,'g':[0.,0.,0.],
                                'rpic_damping':0.,'grid_v_damping_scale':1.})
    solver.finalize_mu_lam()
    upstream.compute_stress_from_F_trial=matched_stress
    # The backend computes Lamé constants in FP32; nu close to 1/2 amplifies
    # its parameter rounding. Test the implementation with those exact
    # constants, and report their difference from the requested material.
    mu_used=float(wp.to_torch(solver.mpm_model.mu)[0]);lam_used=float(wp.to_torch(solver.mpm_model.lam)[0]);K_used=lam_used+2*mu_used/3
    stress_error=material_check(solver,mu_used,K_used,rho0)
    if args.translation_check and args.rotation_check:raise ValueError('choose only one rigid-motion check')
    stretch=1. if args.translation_check or args.rotation_check else args.stretch
    F0=torch.diag(torch.tensor([stretch,stretch**-.5,stretch**-.5],device='cuda',dtype=torch.float32))
    wp.to_torch(solver.mpm_state.particle_F_trial).copy_(F0.expand(n,3,3))
    wp.to_torch(solver.mpm_state.particle_F).copy_(F0.expand(n,3,3))
    wp.to_torch(solver.mpm_state.particle_C).zero_()
    initial_v=torch.tensor([.7,.2,-.1] if args.translation_check else [0.,0.,0.],device='cuda')
    wp.to_torch(solver.mpm_state.particle_v).copy_(initial_v.expand(n,3))
    if args.rotation_check:
        omega=torch.tensor([[0.,-1.,.3],[1.,0.,-.2],[-.3,.2,0.]],device='cuda',dtype=torch.float32)
        wp.to_torch(solver.mpm_state.particle_v).copy_((x-x.mean(0))@omega.T)
        # APIC's affine state represents the same initial velocity field.
        wp.to_torch(solver.mpm_state.particle_C).copy_(omega.expand(n,3,3))
    wp.synchronize()
    # Use backend's rounded material constants in its independent energy audit.
    mu_used=float(wp.to_torch(solver.mpm_model.mu)[0]);lam_used=float(wp.to_torch(solver.mpm_model.lam)[0]);K_used=lam_used+2*mu_used/3
    rows=[];frames=[];frame_times=[];failed=None
    def record(t,step):
        wp.synchronize()
        s=solver.mpm_state
        F=wp.to_torch(s.particle_F_trial).double() # G2P's updated F, not stale F
        pos=wp.to_torch(s.particle_x).double();vel=wp.to_torch(s.particle_v).double()
        C=wp.to_torch(s.particle_C).double();mass=wp.to_torch(s.particle_mass).double();V0=wp.to_torch(s.particle_vol).double()
        J=torch.linalg.det(F)
        if not bool(torch.isfinite(F).all() and (J>0).all()):raise RuntimeError('MPM F invalid/inverted; no clamping')
        if not bool(((pos>2*dx)&(pos<(grid_size-3)*dx)).all()):raise RuntimeError('MPM particle approaching finite grid boundary')
        logJ=J.log();trB=F.square().sum((1,2))/J**(2/3)
        W=.5*mu_used*(trB-3)+.25*K_used*(torch.expm1(2*logJ)-2*logJ)
        elastic=float((V0*W).sum());kinetic=float((.5*mass*vel.square().sum(1)).sum())
        affine=float((mass*dx*dx/8*C.square().sum((1,2))).sum())
        center=(mass[:,None]*pos).sum(0)/mass.sum()
        row={'step':step,'time':t,'elastic_energy_joule':elastic,'kinetic_energy_joule':kinetic,
             'apic_affine_kinetic_joule':affine,'mechanical_energy_joule':elastic+kinetic,
             'mechanical_plus_affine_joule':elastic+kinetic+affine,
             'J_min':float(J.min()),'J_max':float(J.max()),'mass_kg':float(mass.sum()),
             'momentum_kg_m_s':(mass[:,None]*vel).sum(0).tolist(),'com_world_m':(center+torch.tensor(origin,device='cuda')).tolist(),
             'rms_lengths_m':((mass[:,None]*(pos-center).square()).sum(0)/mass.sum()).sqrt().tolist(),
             'max_speed_m_s':float(vel.norm(dim=1).max())}
        rows.append(row);print(json.dumps(row),flush=True)
        if step==0 or abs(t/.002-round(t/.002))<1e-6 or t>=args.until-1e-12:
            frames.append((pos.cpu().numpy()+origin)[::max(1,n//60000)].astype(np.float32));frame_times.append(t)
    record(0.,0)
    # Timing includes first-use upstream compilation; this is not a speed benchmark.
    wp.synchronize();begin=time.perf_counter();t=0.;step=0
    try:
        while t<args.until-1e-14:
            dt=min(args.dt,args.until-t)
            solver.p2g2p(step,dt)
            t+=dt;step+=1
            if step%max(1,round(.001/args.dt))==0 or t>=args.until-1e-14:record(t,step)
    except Exception as exc:
        failed=repr(exc)
    wp.synchronize();elapsed=time.perf_counter()-begin
    initial=rows[0]['mechanical_energy_joule']
    for row in rows:
        row['mechanical_loss_fraction']=1-row['mechanical_energy_joule']/initial
        row['mechanical_plus_affine_loss_fraction']=1-row['mechanical_plus_affine_joule']/initial
    extra={}
    if args.translation_check:
        pos=wp.to_torch(solver.mpm_state.particle_x).double()
        exact=torch.tensor(x_world-origin,device='cuda')+t*initial_v.double()
        extra['translation_max_position_error_m']=float((pos-exact).abs().max())
        extra['translation_max_velocity_error_m_s']=float((wp.to_torch(solver.mpm_state.particle_v)-initial_v).abs().max())
    report={'status':'failed' if failed else 'completed','failure':failed,'backend':'PowerSim vendored PhysGaussian MPM',
            'stress_replacement':'isotropic decoupled neo-Hookean; no plastic return','dtype':'float32 backend, float64 energy audit',
            'upstream_source_sha256':hashes,'upstream_unchanged':all(digest(f)==h for f,h in hashes.items()),
            'source_snapshot':str(source),'source_sha256':source_hash,'dx':dx,'dt':args.dt,'until':args.until,
            'particles':n,'grid_nodes_per_axis':grid_size,'ppc_axis':args.ppc_axis,'particle_spacing_m':spacing,
            'raw_sampling_volume_relative_error':raw_volume/volume-1,'quadrature_volume_normalization':volume/raw_volume,
            'target_mass_kg':mass_exact,'mass_relative_error':rows[0]['mass_kg']/mass_exact-1,
            'E':E,'nu':nu,'mu_backend':mu_used,'K_backend':K_used,'stretch':stretch,
            'rotation_check':args.rotation_check,'translation_check':args.translation_check,
            'mu_rounding_relative_error':mu_used/mu-1,'K_rounding_relative_error':K_used/bulk-1,
            'initial_stress_relative_error':stress_error,'steps':step,'time_reached':t,'seconds_including_first_kernel_compilation':elapsed,
            'gravity':[0.,0.,0.],'colliders':0,'grid_damping':1.,'rpic_damping':0.,'rows':rows,**extra}
    (out/f'{args.tag}.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(out/f'{args.tag}_frames.npz',t=frame_times,x=frames)
    torch.save({'x':wp.to_torch(solver.mpm_state.particle_x).cpu()+torch.tensor(origin,dtype=torch.float32),
                'v':wp.to_torch(solver.mpm_state.particle_v).cpu(),'F':wp.to_torch(solver.mpm_state.particle_F_trial).cpu(),
                'C':wp.to_torch(solver.mpm_state.particle_C).cpu(),'mass':wp.to_torch(solver.mpm_state.particle_mass).cpu()},out/f'{args.tag}_end.pt')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)
    if failed:raise SystemExit(2)


if __name__=='__main__':main()
