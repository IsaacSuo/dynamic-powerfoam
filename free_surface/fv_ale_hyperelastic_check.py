"""Constitutive verification plus real full-foam explicit integration checks.

Material identities are checked independently of the FV implementation. The
dynamic checks then use the actual 7690-cell foam, original cuts and GPU path.
No claim that a homogeneous material test alone validates the full method.
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
import math
import time
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import torch
import numpy as np

from free_surface.fv_ale_hyperelastic_material import response, log_metric_rate, acoustic_tensor, symmetric_deviator
from free_surface.fv_ale_hyperelastic import HyperelasticALE
from free_surface.fv_ale_profile import load
from free_surface.fv_ale_solid import State
from free_surface.fv_ale_device import Replay, PlaneDeviceOps
from free_surface.fv_ale_agglomerate import merge_cell


def from_F(F, rho0=1000.):
    J = torch.linalg.det(F)
    B = F @ F.transpose(-1, -2) / J[..., None, None]**(2/3)
    vals, R = torch.linalg.eigh(B)
    H = symmetric_deviator((R * vals.log()[..., None, :]) @ R.transpose(-1, -2))
    return H, rho0/J


def material_checks():
    torch.manual_seed(91)
    dtype = torch.float64; I = torch.eye(3, dtype=dtype)
    mu, bulk, rho0 = 2e5, 2e6, 1000.
    Fs = torch.stack([I, torch.diag(torch.tensor([2., 2**-.5, 2**-.5], dtype=dtype)),
                      torch.diag(torch.tensor([.5, 1., 1.], dtype=dtype)),
                      I + torch.tensor([[0., 1., 0.], [0., 0., .2], [0., 0., 0.]], dtype=dtype)])
    H, rho = from_F(Fs)
    p, s, sigma, energy, c2, B, J = response(H, rho, mu, bulk, rho0)
    L = torch.randn_like(Fs) * .2; h = 1e-6
    Fp = torch.matrix_exp(h*L) @ Fs; Fm = torch.matrix_exp(-h*L) @ Fs
    Hp, rhop = from_F(Fp); Hm, rhom = from_F(Fm)
    ep = response(Hp, rhop, mu, bulk, rho0)[3]; em = response(Hm, rhom, mu, bulk, rho0)[3]
    numerical_power = (ep-em)/(2*h)
    exact_power = (sigma * L).sum((-1,-2))/rho
    work_error = float((numerical_power-exact_power).norm()/exact_power.norm())
    source_error = float(((Hp-Hm)/(2*h)-log_metric_rate(H,L)).norm()/log_metric_rate(H,L).norm())
    assert work_error < 1e-7, work_error
    assert source_error < 1e-7, source_error
    # Frame indifference: rotate the deformed material, not just the stress.
    R = torch.matrix_exp(torch.tensor([[0., -.9, .3], [.9, 0., -.2], [-.3, .2, 0.]], dtype=dtype))
    Hr, rr = from_F(R @ Fs); rotated = response(Hr, rr, mu, bulk, rho0)
    rotation_error = float((rotated[2]-R@sigma@R.T).norm()/sigma.norm())
    assert rotation_error < 1e-12, rotation_error
    assert torch.allclose(rotated[3], energy, atol=1e-9, rtol=1e-12)
    # Linearized traction derivative and the declared all-direction speed bound.
    n = torch.randn((len(Fs),3),dtype=dtype); n /= n.norm(dim=1)[:,None]
    a = torch.randn_like(n); plane = a[:,:,None]*n[:,None,:]
    hp,rp=from_F((I+h*plane)@Fs); hm,rm=from_F((I-h*plane)@Fs)
    ds=(response(hp,rp,mu,bulk,rho0)[2]-response(hm,rm,mu,bulk,rho0)[2])/(2*h)
    Q=acoustic_tensor(B,J,n,mu,bulk)
    expected=(Q@a[:,:,None])[:,:,0]; measured=(ds@n[:,:,None])[:,:,0]
    acoustic_error=float((expected-measured).norm()/expected.norm())
    assert acoustic_error<1e-7,acoustic_error
    max_ratio=0.;min_eigen=math.inf
    for _ in range(64):
        normal=torch.randn_like(n);normal/=normal.norm(dim=1)[:,None]
        eig=torch.linalg.eigvalsh(acoustic_tensor(B,J,normal,mu,bulk))/rho[:,None]
        min_eigen=min(min_eigen,float(eig.min()));max_ratio=max(max_ratio,float((eig[:,-1]/c2).max()))
    assert min_eigen>0 and max_ratio<=1+1e-12,(min_eigen,max_ratio)
    # Large elastic simple shear: gamma=sin(pi*t), load to gamma=1 then unload.
    cycles=[]
    for steps in [32,64,128]:
        state=torch.zeros((1,3,3),dtype=dtype);dt=1/steps;peak=0.;work=0.
        for k in range(steps):
            t=k*dt
            l0=torch.zeros_like(state);l0[:,0,1]=math.pi*math.cos(math.pi*t)
            l1=torch.zeros_like(state);l1[:,0,1]=math.pi*math.cos(math.pi*(t+dt))
            rate=log_metric_rate(state,l0);trial=state+dt*rate
            old_sigma=response(state,torch.tensor([rho0],dtype=dtype),mu,bulk,rho0)[2]
            trial_sigma=response(trial,torch.tensor([rho0],dtype=dtype),mu,bulk,rho0)[2]
            work+=float(.5*dt*((old_sigma*l0).sum()+(trial_sigma*l1).sum())/rho0)
            state=state+.5*dt*(rate+log_metric_rate(trial,l1))
            F=I.clone();F[0,1]=math.sin(math.pi*(t+dt));exactH,_=from_F(F[None])
            peak=max(peak,float((state-exactH).norm()))
        e=float(response(state,torch.tensor([rho0],dtype=dtype),mu,bulk,rho0)[3][0])
        cycles.append({'steps':steps,'max_log_shape_error':peak,'unloaded_log_shape_norm':float(state.norm()),
                       'unloaded_elastic_energy_J_per_kg':e,'integrated_work_J_per_kg':work})
    assert cycles[-1]['max_log_shape_error']<cycles[0]['max_log_shape_error']/10,cycles
    assert cycles[-1]['unloaded_elastic_energy_J_per_kg']<1e-5,cycles
    return {'energy_power_relative_error':work_error,'log_kinematics_relative_error':source_error,
            'rotation_stress_relative_error':rotation_error,'acoustic_tensor_relative_error':acoustic_error,
            'sampled_min_acoustic_eigenvalue_m2_s2':min_eigen,'max_sampled_wave_speed_squared_over_bound':max_ratio,
            'large_shear_load_unload':cycles}


def gpu_checks():
    from free_surface.fv_ale_symmetric_eigen import eigh3
    torch.manual_seed(42)
    A=symmetric_deviator(torch.randn((4096,3,3),dtype=torch.float64))
    A[:32]=0;A[32:64]*=1e-12;A[64:96]*=10
    eig,R=eigh3(A.cuda());eig,R=eig.cpu(),R.cpu()
    relative=float((R@torch.diag_embed(eig)@R.transpose(-1,-2)-A).norm()/A.norm())
    orth=float((R.transpose(-1,-2)@R-torch.eye(3,dtype=A.dtype)).abs().max())
    assert relative<1e-13 and orth<1e-13,(relative,orth)
    mu,bulk,rho0=2e5,2e6,1000.
    L=torch.randn_like(A)*.2
    def fn(b):
        result=response(b['H'],b['rho'],mu,bulk,rho0)
        tight=response(b['H'],b['rho'],mu,bulk,rho0,'tight')[4]
        return result[2],result[3],log_metric_rate(b['H'],b['L']),tight
    inputs={'H':A.cuda(),'L':L.cuda(),'rho':torch.full((len(A),),rho0,dtype=A.dtype,device='cuda')}
    replay=Replay(fn,inputs);errors=[]
    for H in [A*.1,-A*.2,torch.zeros_like(A)]:
        inputs['H']=H.cuda();actual=[v.clone().cpu() for v in replay(inputs)]
        expected=fn({'H':H,'L':L,'rho':inputs['rho'].cpu()})
        error=max(float((x-y).norm()/y.norm().clamp(min=1.)) for x,y in zip(actual,expected))
        assert error<1e-11,error
        errors.append(error)
    return {'jacobi_reconstruction_relative_error':relative,'jacobi_orthogonality_max_error':orth,
            'changing_state_graph_replay_errors':errors,'graph_build_seconds':replay.build_seconds}


def default_regression():
    oldfile=Path('output/ale/hyperelastic_baseline_20261108/fv_ale_device.py')
    spec=importlib.util.spec_from_file_location('old_fv_device',oldfile)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    sol,reset,data=load(Path('output/ale/fv_state_dt8us.pt'))
    results=[]
    for cls in [module.PlaneDeviceOps,PlaneDeviceOps]:
        sol._device_ops=cls(sol);s=reset();first=sol.rhs(s)
        dt=min(20e-6,.5*first[3]['dt_scale'])
        end,stats=sol.advance(s,dt,first,.5);results.append(end)
    error={key:float((getattr(results[0],key)-getattr(results[1],key)).norm()/getattr(results[0],key).norm().clamp(min=1.)) for key in ['p','r','q']}
    assert max(error.values())<1e-11,error
    return {'old_wilkins_one_step_relative_difference':error}


def roundoff_checks():
    # Exercise the policy without a cut: only a grouped roundoff negative is
    # zeroed; true negative volumes and single-cell negatives remain invalid.
    sol=object.__new__(HyperelasticALE)
    sol.groups=torch.tensor([0,0,2]);sol.has_groups=True
    r=torch.full((3,),.01,dtype=torch.float64)
    bound=256*torch.finfo(r.dtype).eps*(4*torch.pi/3)*r[0]**3
    mesh=SimpleNamespace(r=r,volume=torch.tensor([-.5*bound,1e-6,-.5*bound],dtype=r.dtype))
    sol.normalize_volume_roundoff(mesh)
    assert mesh.volume[0]==0 and mesh.volume[1]==1e-6 and mesh.volume[2]<0
    mesh.volume[0]=-2*bound
    sol.normalize_volume_roundoff(mesh)
    assert mesh.volume[0]<0
    return {'guard_volume_m3':float(bound),'roundoff_grouped_zeroed':True,
            'nonroundoff_negative_not_accepted':True,'singleton_negative_not_accepted':True}


@torch.no_grad()
def full_foam(args):
    out=Path('output/ale')
    input_paths=([Path(args.foam)/'dem_exact_single'/'geometry.npz',
                  Path(args.foam)/'dem_exact_single'/'cells.npz',Path(args.foam)/'plasticine_scenes.json']
                 if args.foam else [out/'fv_state_dt8us.pt'])
    input_hashes={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in input_paths}
    if args.foam:
        foam=Path(args.foam)
        geometry=np.load(foam/'dem_exact_single'/'geometry.npz')
        cells=np.load(foam/'dem_exact_single'/'cells.npz')
        m=json.loads((foam/'plasticine_scenes.json').read_text())['scenes']['0']
        source_p,source_r=geometry['points'],cells['radii'];initial_groups=None
        initial_source={'foam':str(foam.resolve()),'scene':0}
    else:
        old,reset,data=load(out/'fv_state_dt8us.pt');source=reset();m=data['material']
        source_p,source_r=source.p,source.r;initial_groups=old.groups.clone()
        initial_source={'snapshot':str((out/'fv_state_dt8us.pt').resolve())}
    if input_hashes!={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in input_paths}:
        raise RuntimeError('initial geometry/configuration changed while loading')
    initial_source['sha256']=input_hashes
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    source_archive=out/(Path(args.output or args.case).stem+'_sources_'+stamp+'.zip')
    source_hashes={}
    with zipfile.ZipFile(source_archive,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(Path('free_surface').glob('*.py')):
            contents=path.read_bytes();source_hashes[str(path)]=hashlib.sha256(contents).hexdigest()
            archive.writestr(str(path),contents)
    m=dict(m)
    if args.young is not None:m['E']=args.young
    if args.poisson is not None:m['nu']=args.poisson
    ground=float((torch.as_tensor(source_p)[:,1]-torch.as_tensor(source_r)).min())-args.impact_gap if args.case=='impact' else -10.
    solver_class=HyperelasticALE
    from free_surface.fv_ale_releasing import ReleasingHyperelasticALE,ReleaseRequired
    if args.releasing_planes:solver_class=ReleasingHyperelasticALE
    sol=solver_class(source_p,source_r,m['E'],m['nu'],m['ys'],groups=initial_groups,
                       contact_model='plane',surface_motion='coupled',ground=ground,gravity=(0.,0.,0.),
                       friction=0.,wall_mode=args.wall_mode,
                       interface_dissipation=args.interface_dissipation,cap_dissipation=args.cap_dissipation,
                       acoustic_bound=args.acoustic_bound,
                       precision_geometry=args.precision_geometry,
                       transport_reconstruction=args.transport_reconstruction,
                       transport_dissipation=args.transport_dissipation,
                       mechanical_reconstruction=args.mechanical_reconstruction,
                       affine_stencil=args.affine_stencil,
                       affine_cap_motion=args.affine_cap_motion,
                       mesh_fit_iterations=args.mesh_fit_iterations,mesh_fit_regularization=args.mesh_fit_regularization,
                       mesh_fit_radius=args.mesh_fit_radius)
    state=sol.initial
    if args.case=='translation':
        state=sol.set_velocity(state,[.7,.2,-.1])
    elif args.case=='impact':
        state=sol.set_velocity(state,[0.,-args.impact_speed,0.])
    elif args.case=='rotation':
        from free_surface.fv_ale_hyperelastic_flux_checks import centroids
        V=state.mesh.volume;c=centroids(state.mesh)
        c=sol.group_sum(V[:,None]*c)[sol.groups]/sol.group_sum(V)[sol.groups,None]
        origin=(state.q[:,:1]*c).sum(0)/state.q[:,0].sum()
        omega=torch.tensor([[0.,-1.,.3],[1.,0.,-.2],[-.3,.2,0.]],dtype=c.dtype,device=c.device)
        state=sol.set_velocity(state,(c-origin)@omega.T)
    else:
        F=torch.diag(torch.tensor([args.stretch,args.stretch**-.5,args.stretch**-.5],dtype=state.p.dtype,device=state.p.device))
        state=sol.initialize_strain(state,F)
    p0=state.p.clone();r0=state.r.clone();q0=state.q.clone()
    initial_reference_volume=float(state.mesh.volume.median())
    audit0=sol.energy_audit(state);initial_mass=float(state.q[:,0].sum());initial_energy=float(state.q[:,4].sum())
    records=[];t=0.;rejects=[];count=0;merges=[];merge_skips=[];stop=None
    min_accepted_dt=math.inf;max_accepted_dt=0.
    frame_times=[0.];frame_p=[state.p.cpu().numpy()];frame_r=[state.r.cpu().numpy()]
    frame_planes=[None]
    next_frame=args.frame_interval;next_record=args.record_interval
    def spatial_audit(s):
        from free_surface.fv_ale_reconstruction import group_centroids
        mesh=s.mesh
        b={'p':s.p,'r':s.r,'V':mesh.volume,'groups':sol.groups,'i':mesh.i,'j':mesh.j,
           'A':mesh.area,'n':mesh.normal,'c':mesh.centroid,'N':mesh.cap_normal_integral,
           'floor':mesh.ground_area,'floor_c':mesh.ground_centroid,'floor_n':mesh.ground_normal}
        if hasattr(mesh,'planes'):
            from free_surface.fv_ale_planar_boundary import values as plane_values
            b.update(plane_values(mesh))
        x=group_centroids(b);mass=s.q[:,0];center=(mass[:,None]*x).sum(0)/mass.sum()
        return {'com_world_m':center.tolist(),
                'com_velocity_m_s':(s.q[:,1:4].sum(0)/mass.sum()).tolist(),
                'contact_area_m2':float(mesh.ground_area.sum()),
                'unclipped_ball_bottom_m':float((s.p[:,1]-s.r).min()),
                'angular_momentum_about_com_kg_m2_s':torch.cross(x-center,s.q[:,1:4],dim=1).sum(0).tolist(),
                'rms_lengths_m':((mass[:,None]*(x-center).square()).sum(0)/mass.sum()).sqrt().tolist(),
                'max_speed_m_s':float(sol.primitive(s)[1].norm(dim=1).max())}
    spatial0=spatial_audit(state)
    prior=None;elapsed_prior=0.
    if args.resume:
        resume_path=Path(args.resume)
        if not args.output or Path(args.output).stem==resume_path.stem:
            raise ValueError('resume requires a distinct output tag; preserve the source run')
        saved=torch.load(resume_path,map_location='cpu',weights_only=False)
        prior=json.loads(resume_path.with_suffix('.json').read_text())
        if saved.get('state_schema')!=sol.state_schema or saved.get('constitutive')!='neo_hookean':
            raise ValueError('diagnostic resume state schema mismatch')
        if prior.get('releasing_planes',False)!=args.releasing_planes and not args.upgrade_release_boundary:
            raise ValueError('boundary mode differs; an explicit --upgrade-release-boundary is required')
        expected={'case':args.case,'cfl':args.cfl,'stretch':args.stretch if args.case=='release' else 1.,
                  'ground':ground,'wall_mode':args.wall_mode,'merge_dt_target':args.merge_dt,
                  'transport_reconstruction':args.transport_reconstruction,'transport_dissipation':args.transport_dissipation,
                  'mechanical_reconstruction':args.mechanical_reconstruction,'affine_stencil':args.affine_stencil,
                  'affine_cap_motion':args.affine_cap_motion,'interface_dissipation':args.interface_dissipation,
                  'cap_dissipation':args.cap_dissipation,'acoustic_bound':args.acoustic_bound,
                  'precision_geometry':args.precision_geometry,'local_energy_tolerance':args.local_energy_tolerance,
                  'mesh_fit':{'iterations':args.mesh_fit_iterations,'regularization':args.mesh_fit_regularization,'radius':args.mesh_fit_radius}}
        for key,value in expected.items():
            if prior.get(key)!=value:raise ValueError('diagnostic resume configuration mismatch: '+key)
        if prior['material']!=m or prior['initial_source']!=initial_source:
            raise ValueError('diagnostic resume material or original input mismatch')
        if not np.allclose(spatial0['com_velocity_m_s'],prior['initial_spatial']['com_velocity_m_s'],rtol=1e-12,atol=1e-12):
            raise ValueError('diagnostic resume initial velocity mismatch')
        if bool(prior['energy_ledger'])!=args.energy_ledger:
            raise ValueError('diagnostic resume must retain its energy ledger policy')
        if saved['time']!=prior['time_reached'] or saved['time']>=args.until:
            raise ValueError('diagnostic resume time mismatch or no continuation requested')
        sol.groups=saved['groups'].to(state.p.device);sol.has_groups=len(torch.unique(sol.groups))<len(sol.groups)
        pp,rr,qq=[saved[key].to(state.p.device) for key in ['p','r','q']]
        planes=None
        if saved.get('moving_planes') is not None:
            from free_surface.fv_ale_planar_boundary import Planes
            planes=Planes(**saved['moving_planes']).to(pp.device)
        state=State(pp,rr,qq,sol.cut(pp,rr,planes) if planes is not None else sol.cut(pp,rr));sol.primitive(state)
        if args.releasing_planes:
            sol.boundary_insertions=list(prior.get('boundary_insertions',[]))
            sol.boundary_prunings=list(prior.get('boundary_prunings',[]))
        restored_audit=sol.energy_audit(state)
        for key in ['elastic_energy_joule','kinetic_energy_joule','nonelastic_remainder_joule']:
            if not math.isclose(restored_audit[key],prior['final'][key],rel_tol=1e-9,abs_tol=1e-9):
                raise ValueError('diagnostic snapshot and companion report disagree: '+key)
        t=float(saved['time']);count=int(prior['steps']);elapsed_prior=float(prior['seconds'])
        records=list(prior['history']);rejects=list(prior['rejected_attempts'])
        merges=list(prior['merges']);merge_skips=list(prior.get('merge_skips',[]))
        min_accepted_dt=prior['min_accepted_dt'];max_accepted_dt=prior['max_accepted_dt']
        audit0=prior['initial'];spatial0=prior['initial_spatial']
        initial_energy=prior.get('initial_total_energy_joule',sum(audit0[key] for key in ['elastic_energy_joule','kinetic_energy_joule','nonelastic_remainder_joule']))
        initial_mass=prior.get('initial_mass_kg',initial_mass)
        initial_reference_volume=prior.get('initial_reference_volume_m3',initial_reference_volume)
        if args.frames:
            previous_frames=np.load(resume_path.with_name(resume_path.stem+'_frames.npz'))
            frame_times=previous_frames['t'].tolist();frame_p=list(previous_frames['p']);frame_r=list(previous_frames['r'])
            from free_surface.fv_ale_planar_boundary import unpack_frames
            frame_planes=unpack_frames(previous_frames)
        next_frame=t+args.frame_interval;next_record=t+args.record_interval
    ledger={};capture_ledger=False;stage_ledger=None;stage_state=None;stage_rhs=None
    step_audit={'scope':'every accepted step' if args.energy_ledger else 'output records only',
                'max_local_negative_remainder_joule':audit0['negative_remainder_deficit_joule'],
                'max_negative_specific_remainder_joule_per_kg':audit0['negative_specific_remainder_max_joule_per_kg'],
                'max_negative_remainder_over_local_elastic_plus_mu_over_rho0':audit0['negative_remainder_over_local_elastic_plus_mu_over_rho0'],
                'max_local_negative_remainder_time':0.,'max_abs_work_error_joule':0.,
                'max_abs_work_error_time':0.}
    if prior:step_audit=dict(prior['step_audit'])
    if (args.work_snapshot_threshold>0 or args.local_energy_snapshot_threshold>0) and not args.energy_ledger:
        raise ValueError('work-error capture requires --energy-ledger')
    if args.work_retry_fraction<0 or (args.work_retry_fraction>0 and (not args.energy_ledger or initial_energy<=0)):
        raise ValueError('work retry requires a nonnegative fraction, positive initial energy and --energy-ledger')
    work_retry_tolerance=args.work_retry_fraction*initial_energy
    if prior:
        if not math.isclose(work_retry_tolerance,prior['work_retry_tolerance_joule'],rel_tol=1e-12,abs_tol=1e-15):
            raise ValueError('diagnostic resume work tolerance mismatch')
        work_retry_tolerance=prior['work_retry_tolerance_joule']
    if not math.isfinite(args.local_energy_tolerance) or args.local_energy_tolerance<0 or (args.local_energy_tolerance>0 and work_retry_tolerance<=0):
        raise ValueError('local energy tolerance must be nonnegative and requires work retry')
    if args.fast_energy_audit and (not args.energy_ledger or args.mechanical_reconstruction!='compatible'):
        raise ValueError('fast energy audit requires the compatible scheme and --energy-ledger')
    fast_audit_crosscheck={'calls':0,'independent_checks':0,'max_relative_difference':0.,
                           'guard_calls':0,'guard_checks':0,'guard_max_absolute_error_joule':0.}
    if args.energy_ledger:
        from free_surface.fv_ale_energy_ledger import rate,mechanical
        def ledger_rate(s,result):
            if not args.fast_energy_audit:return rate(sol,s,result)
            keys=['mechanical_rate_chain_W','mechanical_energy_joule','internal_acoustic_W','donor_mixing_W',
                  'free_cap_W','wall_acoustic_W','wall_friction_W','gravity_W']
            measured={key:result[3][key] for key in keys}
            if fast_audit_crosscheck['calls']<3 or fast_audit_crosscheck['calls']%args.audit_crosscheck_every==0:
                reference=rate(sol,s,result)
                for key in keys:
                    error=abs(reference[key]-measured[key])/max(abs(reference[key]),1.)
                    fast_audit_crosscheck['max_relative_difference']=max(fast_audit_crosscheck['max_relative_difference'],error)
                    if error>1e-8:raise AssertionError('cached GPU energy account differs from independent full account: '+key)
                fast_audit_crosscheck['independent_checks']+=1
            fast_audit_crosscheck['calls']+=1
            return measured
        original_rhs=sol.rhs
        def audited_rhs(s):
            nonlocal stage_ledger,stage_state,stage_rhs
            result=original_rhs(s)
            if capture_ledger:
                stage_ledger=ledger_rate(s,result);stage_state=s;stage_rhs=result
            return result
        sol.rhs=audited_rhs
        ledger={'merge_change_joule':0.,'step_time_geometry_error_joule':0.}
        if prior:ledger=dict(prior['energy_ledger'])
    rejected_geometry={}
    original_cut=sol.cut
    def audited_cut(p,r,*extra):
        try:return original_cut(p,r,*extra)
        except RuntimeError as exc:
            rejected_geometry.update(p=p.clone().cpu(),r=r.clone().cpu(),reason=str(exc))
            rejected_geometry['moving_planes']=extra[0].packed() if extra and extra[0] is not None else None
            raise
    sol.cut=audited_cut
    reference_volume=initial_reference_volume
    def ready(s):
        if args.releasing_planes:
            s,pruned=sol.compact_boundaries(s)
            if pruned is not None:pruned['time']=t
        for activation in range(64):
            try:return s,sol.rhs(s)
            except ReleaseRequired as request:
                s,info=sol.activate(s,request.cells);info['time']=t
                print(json.dumps({'boundary_activation':info}),flush=True)
        raise RuntimeError('too many boundary activations at one physical state')
    torch.cuda.synchronize();begin=time.perf_counter()
    for k in range(count,args.steps):
        if t>=args.until-1e-14:break
        try:state,first=ready(state)
        except RuntimeError as exc:
            stop={'reason':'accepted-state RHS failed','error':str(exc),'time':t};break
        if args.merge_dt>0:
            while True:
                gv=sol.group_sum(state.mesh.volume)[sol.groups]
                small=gv<.001*reference_volume
                cell=int(torch.where(small,gv,torch.full_like(gv,float('inf'))).argmin()) if bool(small.any()) else None
                if cell is None and args.cfl*first[3]['wave_dt_scale']<args.merge_dt:cell=first[3]['wave_cell']
                if cell is None:break
                if args.energy_ledger:before_merge=float(mechanical(sol,state.q,state.mesh.volume))
                try:state,info=merge_cell(sol,state,cell,allow_unavailable=True)
                except RuntimeError as exc:
                    stop={'reason':'group merge failed; last state retained','error':str(exc),'time':t,'cell':cell};break
                info['time']=t
                if info.get('skipped'):
                    # Merging only targets throughput. A missing admissible
                    # neighbour does not justify changing the state or cap.
                    # Retain the smaller physical CFL step below.
                    merge_skips.append(info)
                    if len(merge_skips)==1:print(json.dumps({'merge_target_not_reached':info}),flush=True)
                    break
                merges.append(info)
                if args.energy_ledger:
                    info['mechanical_change_joule']=float(mechanical(sol,state.q,state.mesh.volume))-before_merge
                    ledger['merge_change_joule']+=info['mechanical_change_joule']
                try:state,first=ready(state)
                except RuntimeError as exc:
                    stop={'reason':'post-merge RHS failed; merged state retained','time':t,'error':str(exc)};break
        if stop is not None:break
        dt=min(args.until-t,args.cfl*first[3]['dt_scale'])
        if dt<1e-10:
            stop={'reason':'diagnostic small timestep stop','rhs':first[3]};break
        if args.energy_ledger:
            first_ledger=ledger_rate(state,first)
            before_step=first_ledger['mechanical_energy_joule']
            capture_ledger=True
        retry=0;activation_attempts=0
        while retry<9:
            try:
                next_state,stats=sol.advance(state,dt,first,args.cfl)
                trial_audit=None;guard_mechanical=None
                if work_retry_tolerance>0:
                    predicted=.5*dt*(first_ledger['mechanical_rate_chain_W']+stage_ledger['mechanical_rate_chain_W'])
                    if args.fast_energy_audit:
                        guard_mechanical,guard_negative,guard_specific,guard_relative=sol.device_ops().energy_guard_full(next_state)
                        trial_audit={'negative_remainder_deficit_joule':guard_negative,
                            'negative_specific_remainder_max_joule_per_kg':guard_specific,
                            'negative_remainder_over_local_elastic_plus_mu_over_rho0':guard_relative}
                        if fast_audit_crosscheck['guard_calls']<3 or fast_audit_crosscheck['guard_calls']%args.audit_crosscheck_every==0:
                            independent=sol.energy_audit(next_state)
                            independent_mechanical=independent['elastic_energy_joule']+independent['kinetic_energy_joule']
                            error=max(abs(guard_mechanical-independent_mechanical),
                                      abs(guard_negative-independent['negative_remainder_deficit_joule']))
                            for key in ['negative_specific_remainder_max_joule_per_kg','negative_remainder_over_local_elastic_plus_mu_over_rho0']:
                                if abs(trial_audit[key]-independent[key])>1e-11*max(abs(independent[key]),1.):
                                    raise AssertionError('GPU local energy audit differs from independent calculation: '+key)
                            if error>1e-11*max(abs(independent_mechanical),1.):
                                raise AssertionError('GPU energy guard differs from independent full audit')
                            fast_audit_crosscheck['guard_max_absolute_error_joule']=max(fast_audit_crosscheck['guard_max_absolute_error_joule'],error)
                            fast_audit_crosscheck['guard_checks']+=1
                        fast_audit_crosscheck['guard_calls']+=1
                    observed=(guard_mechanical if guard_mechanical is not None else float(mechanical(sol,next_state.q,next_state.mesh.volume)))-before_step
                    if abs(observed-predicted)>work_retry_tolerance:
                        raise RuntimeError('mechanical work defect %.9g J exceeds retry tolerance %.9g J; unchanged state retried'
                                           %(observed-predicted,work_retry_tolerance))
                    if trial_audit is None:trial_audit=sol.energy_audit(next_state)
                    if trial_audit['negative_remainder_deficit_joule']>work_retry_tolerance:
                        raise RuntimeError('local mechanical energy deficit %.9g J exceeds retry tolerance %.9g J; unchanged state retried'
                                           %(trial_audit['negative_remainder_deficit_joule'],work_retry_tolerance))
                    relative=trial_audit['negative_remainder_over_local_elastic_plus_mu_over_rho0']
                    if args.local_energy_tolerance>0 and relative>args.local_energy_tolerance:
                        raise RuntimeError('per-group energy deficit / (elastic + mass*mu/rho0) %.9g exceeds tolerance %.9g; unchanged state retried'
                                           %(relative,args.local_energy_tolerance))
                break
            except ReleaseRequired as request:
                activation_attempts+=1
                if activation_attempts>64:
                    stop={'reason':'too many stage release activations','time':t};break
                try:
                    state,info=sol.activate(state,request.cells);info['time']=t
                    print(json.dumps({'boundary_activation':info}),flush=True)
                    state,first=ready(state);dt=min(dt,args.cfl*first[3]['dt_scale'])
                except RuntimeError as exc:
                    stop={'reason':'boundary activation failed; accepted state retained','time':t,'error':str(exc)};break
                if args.energy_ledger:
                    first_ledger=ledger_rate(state,first);before_step=first_ledger['mechanical_energy_joule']
                continue
            except RuntimeError as exc:
                rejects.append({'time':t,'dt':dt,'reason':str(exc)});dt*=.5
                if retry==8:
                    stop={'reason':'all step retries rejected','error':str(exc),'time':t}
                retry+=1
        capture_ledger=False
        if stop is not None:break
        if args.energy_ledger:
            expected=.5*dt*(first_ledger['mechanical_rate_chain_W']+stage_ledger['mechanical_rate_chain_W'])
            actual=(guard_mechanical if guard_mechanical is not None else float(mechanical(sol,next_state.q,next_state.mesh.volume)))-before_step
            if abs(actual-expected)>step_audit['max_abs_work_error_joule']:
                step_audit.update(max_abs_work_error_joule=abs(actual-expected),max_abs_work_error_time=t+dt)
            if args.local_energy_snapshot_threshold>0 and trial_audit is None:
                trial_audit=sol.energy_audit(next_state)
            local_capture=args.local_energy_snapshot_threshold>0 and trial_audit['negative_remainder_over_local_elastic_plus_mu_over_rho0']>args.local_energy_snapshot_threshold
            if (args.work_snapshot_threshold>0 and abs(actual-expected)>args.work_snapshot_threshold) or local_capture:
                def pack_state(s):
                    names=['volume','i','j','area','normal','centroid','cap_area','cap_normal_integral','cap_Q',
                           'ground_area','ground_centroid','ground_normal','surface']
                    return {'p':s.p.cpu(),'r':s.r.cpu(),'q':s.q.cpu(),
                            'moving_planes':s.mesh.planes.packed() if hasattr(s.mesh,'planes') else None,
                            'mesh':{name:getattr(s.mesh,name).cpu() for name in names}}
                def pack_rhs(rhs):return [rhs[0].cpu(),rhs[1].cpu(),rhs[2].cpu(),rhs[3]]
                kind='local_energy' if local_capture else 'work_error'
                path=out/(Path(args.output).stem+f'_{kind}_step{count+1}.pt')
                torch.save({'before':pack_state(state),'stage':pack_state(stage_state),'after':pack_state(next_state),
                    'first_rhs':pack_rhs(first),'stage_rhs':pack_rhs(stage_rhs),'groups':sol.groups.cpu(),
                    'material':m,'time':t,'dt':dt,'step':count+1,'expected_mechanical_change':expected,
                    'actual_mechanical_change':actual,'error':actual-expected,'args':vars(args),
                    'trial_energy_audit':trial_audit,
                    'first_ledger':first_ledger,'stage_ledger':stage_ledger,'source_archive':str(source_archive)},path)
                stop={'reason':kind+' diagnostic snapshot; trial state not submitted','snapshot':str(path),
                      'time':t,'dt':dt,'error_joule':actual-expected}
                print(json.dumps(stop),flush=True)
                break
            for key in ['internal_acoustic_W','donor_mixing_W','free_cap_W','wall_acoustic_W','wall_friction_W','gravity_W']:
                joule_key=key[:-2]+'_joule'
                ledger[joule_key]=ledger.get(joule_key,0.)+.5*dt*(first_ledger[key]+stage_ledger[key])
            ledger['step_time_geometry_error_joule']+=actual-expected
        state=next_state;t+=dt;count+=1
        min_accepted_dt=min(min_accepted_dt,dt);max_accepted_dt=max(max_accepted_dt,dt)
        audit=trial_audit if trial_audit is not None else (sol.energy_audit(state) if args.energy_ledger else None)
        if audit is not None:
            for target,source in [('max_negative_specific_remainder_joule_per_kg','negative_specific_remainder_max_joule_per_kg'),
                                  ('max_negative_remainder_over_local_elastic_plus_mu_over_rho0','negative_remainder_over_local_elastic_plus_mu_over_rho0')]:
                step_audit[target]=max(step_audit[target],audit[source])
        if audit is not None and audit['negative_remainder_deficit_joule']>step_audit['max_local_negative_remainder_joule']:
            step_audit.update(max_local_negative_remainder_joule=audit['negative_remainder_deficit_joule'],
                              max_local_negative_remainder_time=t)
        if args.frames and (t>=next_frame or t>=args.until-1e-14):
            frame_times.append(t);frame_p.append(state.p.cpu().numpy());frame_r.append(state.r.cpu().numpy())
            frame_planes.append(state.mesh.planes.packed() if hasattr(state.mesh,'planes') else None)
            next_frame=t+args.frame_interval
        if k==0 or t>=next_record or t>=args.until-1e-14:
            if audit is None or 'elastic_energy_joule' not in audit:audit=sol.energy_audit(state)
            if audit['negative_remainder_deficit_joule']>step_audit['max_local_negative_remainder_joule']:
                step_audit.update(max_local_negative_remainder_joule=audit['negative_remainder_deficit_joule'],
                                  max_local_negative_remainder_time=t)
            row={'step':k+1,'time':t,'dt':dt,'wall_seconds':elapsed_prior+time.perf_counter()-begin,**audit,**spatial_audit(state),
                 'mass_relative_error':(float(state.q[:,0].sum())-initial_mass)/initial_mass,
                 'total_energy_error_joule':float(state.q[:,4].sum())-initial_energy,
                 'wave_dt':args.cfl*first[3]['wave_dt_scale'],'wave_cell':first[3]['wave_cell'],
                 'merges_so_far':len(merges)}
            if args.energy_ledger:row['energy_ledger']=ledger.copy()
            if args.affine_stencil!='one_ring':row['stencil_statistics']=getattr(sol.device_ops(),'stencil_statistics',{}).copy()
            if args.mesh_fit_iterations:row['mesh_motion_diagnostics']={key:value for key,value in first[3].items() if key.startswith('mesh_fit_')}
            records.append(row);print(json.dumps(row),flush=True)
            next_record=t+args.record_interval
    if args.frames and t>frame_times[-1]+1e-14:
        frame_times.append(t);frame_p.append(state.p.cpu().numpy());frame_r.append(state.r.cpu().numpy())
        frame_planes.append(state.mesh.planes.packed() if hasattr(state.mesh,'planes') else None)
    torch.cuda.synchronize();elapsed=time.perf_counter()-begin
    record={'case':args.case,'stretch':args.stretch if args.case=='release' else 1.,'cells':len(state.r),'groups':int(torch.unique(sol.groups).numel()),
            'ground':ground,'wall_mode':args.wall_mode,
            'releasing_planes':args.releasing_planes,'upgraded_release_boundary_on_resume':args.upgrade_release_boundary,
            'boundary_insertions':getattr(sol,'boundary_insertions',[]),
            'boundary_prunings':getattr(sol,'boundary_prunings',[]),
            'time_reached':t,'completed':t>=args.until-1e-14,'steps':count,'seconds':elapsed_prior+elapsed,'invocation_seconds':elapsed,'cfl':args.cfl,
            'resumed_from':str(Path(args.resume).resolve()) if args.resume else None,
            'prior_source_archive':prior['source_archive'] if prior else None,
            'initial_total_energy_joule':initial_energy,'initial_mass_kg':initial_mass,'initial_reference_volume_m3':initial_reference_volume,
            'min_accepted_dt':min_accepted_dt if count else None,'max_accepted_dt':max_accepted_dt,
            'initial':audit0,'final':sol.energy_audit(state),'history':records,'rejected_attempts':rejects,
            'initial_spatial':spatial0,'material':m,
            'initial_source':initial_source,
            'source_archive':str(source_archive),'source_sha256':source_hashes,'command_line':sys.argv,
            'merge_dt_target':args.merge_dt,'merges':merges,'merge_skips':merge_skips,'diagnostic_stop':stop,
            'energy_ledger':ledger,
            'step_audit':step_audit,
            'work_retry_tolerance_joule':work_retry_tolerance,
            'local_energy_tolerance':args.local_energy_tolerance,
            'fast_energy_audit':args.fast_energy_audit,'fast_audit_crosscheck':fast_audit_crosscheck,
            'transport_reconstruction':args.transport_reconstruction,
            'transport_dissipation':args.transport_dissipation,
            'mechanical_reconstruction':args.mechanical_reconstruction,
            'affine_stencil':args.affine_stencil,
            'affine_cap_motion':args.affine_cap_motion,
            'stencil_statistics':getattr(sol.device_ops(),'stencil_statistics',{}),
            'mesh_fit':{'iterations':args.mesh_fit_iterations,'regularization':args.mesh_fit_regularization,'radius':args.mesh_fit_radius},
            'interface_dissipation':args.interface_dissipation,'cap_dissipation':args.cap_dissipation,
            'acoustic_bound':args.acoustic_bound,
            'precision_geometry':args.precision_geometry,
            'precision_geometry_records':sol._precision_geometry_records,
            'mass_relative_error':(float(state.q[:,0].sum())-initial_mass)/initial_mass,
            'total_energy_error_joule':float(state.q[:,4].sum())-initial_energy,
            'total_momentum_change_kg_m_s':(state.q[:,1:4].sum(0)-q0[:,1:4].sum(0)).tolist(),
            'graph_builds':sol.device_ops().builds}
    if args.case=='translation':
        v=torch.tensor([.7,.2,-.1],dtype=state.p.dtype,device=state.p.device)
        record['translation_max_position_error_m']=float((state.p-p0-t*v).abs().max())
        record['translation_max_radius_error_m']=float((state.r-r0).abs().max())
        assert record['translation_max_position_error_m']<1e-7,record
        assert abs(record['mass_relative_error'])<1e-12,record
    torch.save({'p':state.p.cpu(),'r':state.r.cpu(),'q':state.q.cpu(),'groups':sol.groups.cpu(),
                'moving_planes':state.mesh.planes.packed() if hasattr(state.mesh,'planes') else None,
                'releasing_planes':args.releasing_planes,
                'time':t,'constitutive':'neo_hookean','state_schema':sol.state_schema,'material':m,
                'test_case':args.case,'ground':ground,'gravity':[0.,0.,0.],'wall_mode':args.wall_mode,
                'transport_reconstruction':args.transport_reconstruction,
                'transport_dissipation':args.transport_dissipation,
                'mechanical_reconstruction':args.mechanical_reconstruction,
                'affine_stencil':args.affine_stencil,
                'affine_cap_motion':args.affine_cap_motion,
                'mesh_fit':{'iterations':args.mesh_fit_iterations,'regularization':args.mesh_fit_regularization,'radius':args.mesh_fit_radius},
                'interface_dissipation':args.interface_dissipation,'cap_dissipation':args.cap_dissipation,
                'acoustic_bound':args.acoustic_bound,
                'precision_geometry':args.precision_geometry,
                'rejected_geometry':rejected_geometry},out/(Path(args.output).stem+'.pt' if args.output else f'hyperelastic_{args.case}.pt'))
    if args.frames:
        from free_surface.fv_ale_planar_boundary import pack_frames
        np.savez_compressed(out/(Path(args.output).stem+'_frames.npz'),t=frame_times,p=frame_p,r=frame_r,**pack_frames(frame_planes))
    return record


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--case',choices=['material','gpu','regression','roundoff','translation','rotation','release','impact'],default='material')
    ap.add_argument('--wall-mode',choices=['unsupported','compatible_slip','compatible_slip_acoustic'],default='unsupported')
    ap.add_argument('--resume',type=Path,default=None,help='Continue a complete diagnostic snapshot with identical material, mechanics and accuracy settings; use a new output tag')
    ap.add_argument('--releasing-planes',action='store_true',help='Opt-in moving material/vacuum power faces for frictionless release')
    ap.add_argument('--upgrade-release-boundary',action='store_true',help='Explicitly change an old fixed-wall resume into the experimental releasing-boundary mode; recorded as a model change')
    ap.add_argument('--impact-gap',type=float,default=.0002)
    ap.add_argument('--impact-speed',type=float,default=1.)
    ap.add_argument('--stretch',type=float,default=1.5);ap.add_argument('--steps',type=int,default=200)
    ap.add_argument('--until',type=float,default=.002);ap.add_argument('--cfl',type=float,default=.5)
    ap.add_argument('--merge-dt',type=float,default=0.,help='Existing production P0 grouping; 0 disables dynamic merges')
    ap.add_argument('--energy-ledger',action='store_true',help='Diagnostic mechanical energy account; slower, no model change')
    ap.add_argument('--fast-energy-audit',action='store_true',help='Use the same energy account already computed inside the GPU RHS; independently cross-check every 500 rate evaluations')
    ap.add_argument('--audit-crosscheck-every',type=int,default=500,help='Independent cached-audit check interval; first three are always checked')
    ap.add_argument('--work-snapshot-threshold',type=float,default=0.,help='Stop and save full RK2 stages if one-step work error exceeds this many joules; 0 disables')
    ap.add_argument('--local-energy-snapshot-threshold',type=float,default=0.,help='Stop before submitting a trial whose largest local energy deficit / (elastic energy + mu/rho0) exceeds this value; save full RK2 stages')
    ap.add_argument('--local-energy-tolerance',type=float,default=0.,help='Retry from unchanged state when any group has negative remainder exceeding this fraction of elastic energy + mass*mu/rho0; 0 retains the legacy global guard')
    ap.add_argument('--work-retry-fraction',type=float,default=0.,help='Diagnostic adaptive retry: halve a rejected step if its work error or summed negative mechanical-energy remainder exceeds this fraction of initial energy; requires ledger, 0 disables')
    ap.add_argument('--interface-dissipation',type=float,default=1.)
    ap.add_argument('--acoustic-bound',choices=['conservative','tight'],default='conservative')
    ap.add_argument('--precision-geometry',action='store_true',help='Reevaluate eligible tiny two-plane wedges with a positive high-precision integral; no volume floor')
    ap.add_argument('--cap-dissipation',type=float,default=1.)
    ap.add_argument('--transport-reconstruction',choices=['constant','linear','entropy'],default='constant')
    ap.add_argument('--transport-dissipation',type=float,default=.1)
    ap.add_argument('--mechanical-reconstruction',choices=['constant','affine','compatible'],default='constant')
    ap.add_argument('--affine-stencil',choices=['one_ring','extended','smooth'],default='one_ring')
    ap.add_argument('--affine-cap-motion',choices=['isotropic','volume','balanced'],default='isotropic')
    ap.add_argument('--mesh-fit-iterations',type=int,default=0)
    ap.add_argument('--mesh-fit-regularization',type=float,default=.1)
    ap.add_argument('--mesh-fit-radius',action='store_true')
    ap.add_argument('--young',type=float,default=None);ap.add_argument('--poisson',type=float,default=None)
    ap.add_argument('--foam',default=None,help='Use this real production foam and scene-0 material instead of the saved 7690-cell geometry')
    ap.add_argument('--record-interval',type=float,default=.0001)
    ap.add_argument('--frames',action='store_true');ap.add_argument('--frame-interval',type=float,default=.002)
    ap.add_argument('--output',default=None);args=ap.parse_args()
    if args.audit_crosscheck_every<1:ap.error('--audit-crosscheck-every must be positive')
    if args.upgrade_release_boundary and (not args.releasing_planes or args.resume is None):
        ap.error('--upgrade-release-boundary requires --resume and --releasing-planes')
    if args.case=='impact' and (args.impact_gap<0 or args.impact_speed<=0):ap.error('impact gap must be nonnegative and speed positive')
    checks={'material':material_checks,'gpu':gpu_checks,'regression':default_regression,'roundoff':roundoff_checks}
    record=checks[args.case]() if args.case in checks else full_foam(args)
    path=Path('output/ale')/(args.output or f'hyperelastic_{args.case}_check.json')
    path.write_text(json.dumps(record,indent=2)+'\n')
    brief={k:v for k,v in record.items() if k not in ['history','merges','merge_skips','rejected_attempts','source_sha256','graph_builds']}
    print(json.dumps(brief),flush=True)


if __name__=='__main__':main()
