"""Integrated real-scene trial driver for fv_ale_solid; no toy scene substitution."""
import json
import math
import time
import traceback
import numpy as np
import torch
import warp as wp
from free_surface.fv_ale_solid import SolidALE,State
from free_surface.fv_ale_derefine import dissolve
from free_surface.fv_ale_agglomerate import merge_cell


@torch.no_grad()
def run(args,X0,r0,gt):
    constitutive=getattr(args,'fv_constitutive','wilkins')
    from free_surface.fv_ale_hyperelastic_config import load_config
    hyper_config=load_config(getattr(args,'fv_hyperelastic_config',None))
    if hyper_config is not None and constitutive!='neo_hookean':
        raise ValueError('hyperelastic run configuration cannot be applied to the old material')
    scene_material=gt
    if hyper_config and hyper_config.get('material'):
        # Copy rather than mutate the loaded scene shared with the caller.
        # The effective material is saved and compared on every resume.
        gt=dict(gt,**hyper_config['material'])
    hyper_options=dict(hyper_config['solver']) if hyper_config else {}
    boundary_mode=hyper_options.pop('boundary_mode','fixed')
    from free_surface.fv_ale_releasing import ReleasingHyperelasticALE,ReleaseRequired
    solver_class=SolidALE
    if constitutive=='neo_hookean':
        from free_surface.fv_ale_hyperelastic import HyperelasticALE
        solver_class=HyperelasticALE
        if boundary_mode=='releasing_planes':solver_class=ReleasingHyperelasticALE
    if boundary_mode=='releasing_planes' and args.fv_derefine:
        raise ValueError('moving-plane owner indices require retained geometric sites; use agglomeration, not derefinement')
    target_time=args.frames/24
    if getattr(args,'fv_until',None) is not None:
        if not math.isfinite(args.fv_until) or args.fv_until<=0:
            raise ValueError('--fv-until must be finite and positive')
        target_time=min(target_time,args.fv_until)
    wp.init(); torch.cuda.synchronize(); begin=time.perf_counter()
    resume=None
    if args.fv_resume is not None:
        resume=torch.load(args.fv_resume,weights_only=False,map_location='cpu')
        if resume.get('constitutive','wilkins')!=constitutive:
            raise ValueError('FV resume constitutive/state representation mismatch; stress is not log shape')
        if constitutive=='neo_hookean' and resume.get('state_schema')!='fv_neo_hookean_log_bbar_v1':
            raise ValueError('unrecognized hyperelastic snapshot schema')
        if resume.get('hyperelastic_config')!=hyper_config:
            raise ValueError('FV resume hyperelastic mechanics/initialization configuration mismatch')
        if resume['material']!=gt or resume.get('original_cell_count',len(resume['p']))!=len(X0):
            raise ValueError('FV resume material/original-cell-count mismatch')
        for key,wanted in [('friction',args.mu),('ground',hyper_options.get('ground',-.05)),('rho0',1000.)]:
            if key in resume and resume[key]!=wanted: raise ValueError('FV resume '+key+' mismatch')
    solver=solver_class(resume['p'] if resume else X0,resume['r'] if resume else r0,
                    gt['E'],gt['nu'],gt['ys'],friction=args.mu,interior_radii=args.fv_interior_radii,
                    contact_model='plane' if args.fv_ground_cut else ('legacy-ball' if args.fv_legacy_ball_contact else 'surface'),
                    contact_order=args.fv_contact_order,contact_scale=args.fv_contact_scale,contact_damping=args.fv_contact_damping,surface_motion=args.fv_surface_motion,groups=resume.get('groups') if resume else None,device_step=not args.fv_reference_step,**hyper_options)
    initial_assigned_velocity=hyper_config.get('initial_velocity') if hyper_config else None
    initial_assigned_velocity=gt['v'] if initial_assigned_velocity is None else initial_assigned_velocity
    state=solver.set_velocity(solver.initial,initial_assigned_velocity)
    if hyper_config and not resume and hyper_config['initial_stretch']!=1.:
        stretch=hyper_config['initial_stretch']
        state=solver.initialize_strain(state,torch.diag(torch.tensor([stretch,stretch**-.5,stretch**-.5],dtype=state.p.dtype,device=state.p.device)))
    if resume:
        if args.fv_ground_cut and resume.get('contact_model')!='plane' and bool((state.mesh.ground_area>0).any()):
            raise ValueError('do not reinterpret an already penetrating curved-contact state as a planar-contact trajectory')
        if resume.get('contact_model')=='plane' and not args.fv_ground_cut:
            raise ValueError('a ground-clipped snapshot requires --fv-ground-cut')
        state.q=resume['q'].to(state.q.device)
        if 'groups' in resume:
            solver.groups=resume['groups'].to(state.p.device)
            solver.has_groups=bool(torch.unique(solver.groups).numel()<len(solver.groups))
        if resume.get('moving_planes') is not None:
            from free_surface.fv_ale_planar_boundary import Planes
            state.mesh=solver.cut(state.p,state.r,Planes(**resume['moving_planes']).to(state.p.device))
        if boundary_mode=='releasing_planes':
            solver.boundary_insertions=list(resume.get('boundary_insertions',[]))
            solver.boundary_prunings=list(resume.get('boundary_prunings',[]))
    ids=(resume.get('ids',torch.arange(len(state.r))) if resume else torch.arange(len(state.r))).to(state.p.device)
    reference_volume=float(resume.get('reference_volume',state.mesh.volume.median())) if resume else float(state.mesh.volume.median())
    solver.contact_length=reference_volume**(1/3)
    solver.contact_k=args.fv_contact_scale*solver.bulk/solver.contact_length
    removals=[];merges=[];merge_skips=[]
    controller=None
    if hyper_config and hyper_config.get('step_control')=='remember':
        from free_surface.fv_ale_step_control import RememberedStep
        if resume and resume.get('step_control_state') is None:
            raise ValueError('remembered-step resume lacks its controller state')
        controller=RememberedStep(resume.get('step_control_state') if resume else None)
    report={'status':'running','kind':'experimental direct ALE solid FV adaptation, not original paper reproduction',
      'constitutive':constitutive,'state_schema':getattr(solver,'state_schema','fv_wilkins_stress_v1'),
      'hyperelastic_config':hyper_config,
      'foam':str(args.foam),'scene':args.scene,'cell_count':len(X0),'material':gt,
      'surface_motion':args.fv_surface_motion,'contact_model':solver.contact_model,'contact_order':args.fv_contact_order,
      'contact_scale':args.fv_contact_scale,'contact_damping':args.fv_contact_damping,
      'contact_stiffness_pa_per_m':solver.contact_k,
      'contact_limitations':('Sphere motion IS the boundary velocity; acoustic traction matching is weak over each cap, not pointwise. Viscous Coulomb regularization permits slip; compliant contact permits penetration. Full convergence/thermodynamic validation remains open.' if args.fv_surface_motion=='coupled' else 'Diagnostic projection: actual spherical motion can differ pointwise from acoustic boundary velocity; work defect is recorded.'),
      'physics_changes':['Wilkins/Jaumann deviatoric stress, perfect von Mises plasticity',
          'stiffened-gas EOS gamma=2 matching initial bulk modulus (different from k1 log-volume energy)',
          'integrated traction-free spherical-cap boundary; pointwise kinematic mismatch is measured',
          'Surface penalty/dashpot traction; coupled mode uses viscously regularized Coulomb friction, with frictional heat retained in the body. Legacy ball rule only if explicitly selected',
          'explicit SSPRK2 with reported physical substeps; no F/L parallel volume variables'],
      'agglomerate':args.fv_agglomerate,'agglomerate_dt':args.fv_agglomerate_dt,
      'derefine':args.fv_derefine,'min_volume_ratio':args.fv_min_volume_ratio,
      'interior_radii':args.fv_interior_radii,'cfl':args.fv_cfl,'max_physical_dt':args.dt,'target_time':target_time,
      'budget_wall_seconds':args.fv_budget,'step_limit':args.fv_steps,'tests_do_not_prove':'accuracy, robustness or speed for complete elastoplastic landing'}
    if args.fv_ground_cut:
        report['contact_limitations']='Real geometry is clipped at fixed ground. Compressive acoustic wall reaction and Coulomb friction; tensile/release predictions stop the experiment rather than apply adhesion. Global material and time convergence remain unvalidated.'
        report['physics_changes'][3]='Stationary ground face, zero normal mass flux/work, compressive acoustic reaction, exact local Coulomb stick/slip; no penalty stiffness or surface quadrature.'
    if constitutive=='neo_hookean':
        from free_surface.fv_ale_hyperelastic_config import archive_run
        report.update(archive_run(args,hyper_config))
        report['original_scene_material']=scene_material
        report['physics_changes'][0]='Pure compressible neo-Hookean; mass-weighted log isochoric metric; no yield return or plasticity'
        report['physics_changes'][1]='Unified elastic shape/volume energy; J=rho0/rho from actual cut volume; no stiffened-gas pressure'
        report['physics_changes'][4]='Explicit SSPRK2; log-metric transport; state-dependent scalar upper wave-speed bound'
        report['hyperelastic_limitations']='Experimental athermal model. Total minus kinetic minus elastic energy is audited, not reset. Log-state mixing is numerical dissipation. Large-strain material compatibility, free-surface adaptation and contact release remain to be validated.'
        if getattr(solver,'wall_mode','unsupported')=='compatible_slip':
            report['contact_limitations']='Opt-in frictionless compatible fixed-plane wall. The reaction is included once through the force/gradient transpose. Stationary-wall work is zero. Tensile reaction still stops the run; full release is not established.'
            report['physics_changes'][3]='Fixed ground face, reconstructed tangential velocity and zero normal velocity; compatible force transpose, no extra acoustic wall force or friction.'
        if getattr(solver,'wall_mode','unsupported')=='compatible_slip_acoustic':
            report['contact_limitations']='Opt-in frictionless fixed-plane wall with reconstructed normal acoustic impedance. Reaction and dissipation use the same reconstruction transpose. Tensile Riemann reaction stops the run; full separation is not established.'
            report['physics_changes'][3]='Fixed ground face, zero normal wall velocity, elastic reaction plus acoustic impedance correction; numerical wall heat retained, no friction or external wall work.'
        if boundary_mode=='releasing_planes':
            report['contact_limitations']='Experimental frictionless moving material/vacuum power faces. Fixed ground remains in the cut. Release inserts a redundant geometric boundary, with compatible acoustic normal motion and force transpose. Full rebound and repeated recontact validation are still required.'
            report['physics_changes'][3]='Ground-clipped unilateral contact plus explicitly advected planar free surfaces; no ghost material mass, no adhesion, no state or energy reset.'
    ts=[]; xs=[]; rs=[]; masses=[]; vs=[]; stresses=[]; Ls=[]; eps=[]; works=[]; hist=[];plane_frames=[]
    m0=float(state.q[:,0].sum()); e0=float(state.q[:,4].sum())
    step_audit=None;work_reference=None
    if hyper_config and hyper_config['work_retry_fraction']>0:
        from free_surface.fv_ale_hyperelastic_config import StepAudit,initial_work_reference
        work_reference=resume.get('work_retry_reference') if resume else initial_work_reference(solver,state)
        tolerance=resume.get('work_retry_tolerance_joule') if resume else hyper_config['work_retry_fraction']*work_reference['scale_joule']
        if tolerance is None or not math.isfinite(tolerance) or tolerance<=0:
            raise ValueError('work retry requires a positive saved or initial energy scale')
        step_audit=StepAudit(solver,tolerance,hyper_config.get('local_energy_tolerance',0.))
        report['work_retry_reference']=work_reference
    initial_velocity=solver.primitive(state)[1]
    energy_scale=max(float((.5*state.q[:,:1]*initial_velocity**2).sum()),m0*9.8*.5,1.)
    physical_time=float(resume['time']) if resume else 0.
    if target_time<physical_time:
        raise ValueError('FV end time precedes the saved state')
    report['start_time']=physical_time
    report['resume_from']=str(args.fv_resume) if resume else None
    count=0; retries=0; rejection_reasons={}; rejection_examples=[]; source_work=0.; friction_heat=0.; boundary_work_defect=0.; boundary_abs_work_defect=0.; next_frame=(math.floor(physical_time*24+1e-9)+1)/24
    min_dt=math.inf; max_dt=0.; analytic_time=0.
    def save_frame(s,t):
        m,v,rho,p,stress,_,_=solver.primitive(s)
        def padded(a,fill=float('nan')):
            out=torch.full((len(X0),)+tuple(a.shape[1:]),fill,dtype=a.dtype,device=a.device)
            out[ids]=a
            return out.cpu().numpy().copy()
        # Fixed original slots; inactive cells have zero radius/mass and NaN
        # fields. Consumers MUST use mass>0 as the active mask.
        ts.append(t);xs.append(padded(s.p));rs.append(padded(s.r,0.))
        masses.append(padded(m,0.));vs.append(padded(v));stresses.append(padded(stress))
        Ls.append(padded(torch.log(solver.rho0/rho)))
        eps.append(padded(s.q[:,14]/m));works.append(padded(s.q[:,15],0.))
        plane_frames.append(s.mesh.planes.packed() if hasattr(s.mesh,'planes') else None)
    def save():
        from free_surface.fv_ale_planar_boundary import pack_frames
        np.savez(args.save/('frames_'+args.tag+'.npz'),t=np.array(ts),x=np.array(xs),r=np.array(rs),
            mass=np.array(masses),v=np.array(vs),stress=np.array(stresses),L=np.array(Ls),ep=np.array(eps),plastic_work=np.array(works),
            ground_clipped=np.array(args.fv_ground_cut),ground_height=np.array(solver.ground),**pack_frames(plane_frames))
        (args.save/('fv_solid_'+args.tag+'.json')).write_text(json.dumps(report,indent=2)+'\n')
        (args.save/('fv_steps_'+args.tag+'.json')).write_text(json.dumps(hist,indent=2)+'\n')
    def snapshot(reason):
        torch.save({'p':state.p.cpu(),'r':state.r.cpu(),'q':state.q.cpu(),'time':physical_time,
           'moving_planes':state.mesh.planes.packed() if hasattr(state.mesh,'planes') else None,
           'boundary_insertions':getattr(solver,'boundary_insertions',[]),
           'boundary_prunings':getattr(solver,'boundary_prunings',[]),
           'constitutive':constitutive,'state_schema':getattr(solver,'state_schema','fv_wilkins_stress_v1'),
           'hyperelastic_config':hyper_config,'gravity':solver.g.cpu().tolist(),
           'work_retry_tolerance_joule':step_audit.tolerance if step_audit else None,
           'work_retry_reference':work_reference,
           'step_control_state':controller.packed() if controller else None,
           'steps':count+(int(resume['steps']) if resume else 0),'material':gt,'reason':reason,
           'groups':solver.groups.cpu(),'ids':ids.cpu(),'original_cell_count':len(X0),'reference_volume':reference_volume,
           'surface_motion':args.fv_surface_motion,'contact_model':solver.contact_model,'contact_order':args.fv_contact_order,
           'contact_scale':args.fv_contact_scale,'contact_damping':args.fv_contact_damping,
           'interior_radii':args.fv_interior_radii,'friction':args.mu,'ground':solver.ground,'rho0':solver.rho0,'frames_file':str((args.save/('frames_'+args.tag+'.npz')).resolve())},args.save/('fv_state_'+args.tag+'.pt'))
    if resume and resume.get('frames_file'):
        old=np.load(resume['frames_file'])
        for name,target in [('t',ts),('x',xs),('r',rs),('mass',masses),('v',vs),('stress',stresses),('L',Ls),('ep',eps),('plastic_work',works)]:
            target.extend(list(old[name]))
        from free_surface.fv_ale_planar_boundary import unpack_frames
        plane_frames.extend(unpack_frames(old))
    else: save_frame(state,physical_time)
    print('FV CONTACT: '+solver.contact_model+'; surface-boundary kinematics remain an approximation.',flush=True)
    print('FV SOLID: %d cells; new constitutive law and approximate spherical boundary; full landing is an experiment'%len(X0),flush=True)
    # Exact uniform free flight, only for this uniform unstressed initial scene.
    # Geometry is still recut at saved states; no elastic/contact dynamics skipped.
    if not args.fv_no_freeflight and resume is None and not (hyper_config and hyper_config['initial_stretch']!=1.):
        gap=float((state.p[:,1]-state.r-solver.ground)[state.mesh.surface].min())
        vy=float(initial_assigned_velocity[1]); gy=float(solver.g[1]); hit=0.
        if gap>0 and gy<0:
            hit=(-vy-math.sqrt(vy*vy-2*gy*gap))/gy
        hit=min(hit,target_time)
        p_initial=state.p.clone(); q_initial=state.q.clone(); v_initial=torch.tensor(initial_assigned_velocity,dtype=state.p.dtype,device=state.p.device)
        for t in sorted(set([k/24 for k in range(1,args.frames+1) if k/24<hit]+([hit] if hit>0 else []))):
            p=p_initial+t*v_initial+.5*t*t*solver.g
            q=q_initial.clone(); v=v_initial+t*solver.g
            q[:,1:4]=q[:,:1]*v
            q[:,4]=q[:,0]*(solver.e0+.5*(v*v).sum())
            state=State(p,state.r,q,solver.cut(p,state.r))
            physical_time=t;save_frame(state,t)
        analytic_time=physical_time
        source_work=float(state.q[:,4].sum())-e0
        next_frame=(math.floor(physical_time*24+1e-9)+1)/24
        print('Exact rigid free flight to %.9f s; next motion includes contact'%physical_time,flush=True)
    def ready(s):
        if boundary_mode=='releasing_planes':
            s,pruned=solver.compact_boundaries(s)
            if pruned is not None:pruned['time']=physical_time
        for activation in range(64):
            try:return s,solver.rhs(s)
            except ReleaseRequired as request:
                s,info=solver.activate(s,request.cells);info['time']=physical_time
                print('FV BOUNDARY ACTIVATION '+json.dumps(info),flush=True)
        raise RuntimeError('too many boundary activations at one state')
    try:
        state,first=ready(state)
        report['initial_after_flight_rhs']=first[3]
        report['initial_after_flight_cap_mismatch']=solver.cap_velocity_mismatch(state)
        print('First stable-step estimate %.3g s (macro %.3g s), limiting cell %d'%(args.fv_cfl*first[3]['dt_scale'],args.dt,first[3]['wave_cell']),flush=True)
        while physical_time<target_time-1e-13:
            if count>=args.fv_steps or time.perf_counter()-begin>=args.fv_budget:
                report['status']='budget_stopped_not_complete';break
            first=None
            if args.fv_agglomerate:
                while True:
                    gv=solver.group_sum(state.mesh.volume)[solver.groups]
                    small=gv<args.fv_min_volume_ratio*reference_volume
                    k=int(torch.where(small,gv,torch.full_like(gv,float('inf'))).argmin()) if bool(small.any()) else None
                    if k is None and args.fv_agglomerate_dt>0:
                        state,first=ready(state);probe=first[3]
                        if args.fv_cfl*probe['wave_dt_scale']<args.fv_agglomerate_dt:k=probe['wave_cell']
                    if k is None:break
                    tick=time.perf_counter();state,info=merge_cell(solver,state,k,allow_unavailable=constitutive=='neo_hookean')
                    if info.get('skipped'):
                        info.update(time=physical_time,seconds=time.perf_counter()-tick)
                        merge_skips.append(info)
                        if len(merge_skips)==1:print('FV MERGE TARGET NOT REACHED '+json.dumps(info),flush=True)
                        break
                    first=None
                    info.update(time=physical_time,seconds=time.perf_counter()-tick,
                        original_ids=ids[info['members']].cpu().tolist())
                    merges.append(info);print('FV MERGE '+json.dumps(info),flush=True)
            if args.fv_derefine:
                # Process sequentially: removing one competitor changes the
                # candidate neighbours' volumes, as in AREPO derefinement.
                while True:
                    small=(state.mesh.volume<args.fv_min_volume_ratio*reference_volume)&~state.mesh.surface
                    if not bool(small.any()):break
                    k=int(torch.where(small,state.mesh.volume,torch.full_like(state.r,float('inf'))).argmin())
                    old_id=int(ids[k]); tick=time.perf_counter()
                    state,keep,removal=dissolve(solver,state,k);ids=ids[keep]
                    removal.update(original_id=old_id,time=physical_time,seconds=time.perf_counter()-tick)
                    removals.append(removal)
                    print('FV DEREFINE '+json.dumps(removal),flush=True)
            if first is None:state,first=ready(state)
            proposal=min(args.dt,args.fv_cfl*first[3]['dt_scale'])
            if controller:proposal=controller.propose(proposal)
            dt=min(proposal,target_time-physical_time,next_frame-physical_time)
            boundary_limited=dt<proposal
            proposed_dt=dt
            if args.fv_stop_dt>0 and dt<args.fv_stop_dt:
                report['status']='diagnostic_stop_small_dt';break
            if dt<1e-12:raise RuntimeError('physical dt below 1e-12; not advanced')
            if step_audit:step_audit.begin(state,first)
            attempt=0;activations=0
            while attempt<9:
                if dt<1e-12: raise RuntimeError('retry dt below 1e-12; state not advanced')
                try:
                    candidate,stats=solver.advance(state,dt,first,args.fv_cfl)
                    if step_audit:stats.update(step_audit.check(candidate,dt))
                    break
                except ReleaseRequired as request:
                    activations+=1
                    if activations>64:raise RuntimeError('too many stage boundary activations')
                    state,info=solver.activate(state,request.cells);info['time']=physical_time
                    print('FV BOUNDARY ACTIVATION '+json.dumps(info),flush=True)
                    state,first=ready(state);dt=min(dt,args.fv_cfl*first[3]['dt_scale'])
                    if step_audit:step_audit.begin(state,first)
                    continue
                except RuntimeError as exc:
                    retries+=1
                    reason=str(exc).split(':')[0];rejection_reasons[reason]=rejection_reasons.get(reason,0)+1
                    if len(rejection_examples)<10:rejection_examples.append({'time':physical_time,'dt':dt,'reason':str(exc)})
                    if attempt==8:raise
                    dt*=.5;attempt+=1
            if controller:controller.accept(dt,attempt,boundary_limited)
            state=candidate; physical_time+=dt;count+=1
            min_dt=min(min_dt,dt);max_dt=max(max_dt,dt)
            source_work+=stats['source_work'];friction_heat+=stats['friction_heat']
            boundary_work_defect+=stats['boundary_work_defect']
            boundary_abs_work_defect+=stats['boundary_abs_work_defect']
            if 'total_mass' in stats:total_mass,total_energy=stats['total_mass'],stats['total_energy']
            else:total_mass,total_energy=torch.stack((state.q[:,0].sum(),state.q[:,4].sum())).tolist()
            mass_error=(total_mass-m0)/m0
            energy_error=(total_energy-e0-source_work)/energy_scale
            stats.update(step=count,time=physical_time,mass_error=mass_error,energy_source_balance_error=energy_error,wall_seconds=time.perf_counter()-begin)
            stats.update(proposed_dt=proposed_dt,retries=attempt)
            hist.append(stats)
            if count==1 or count%20==0:
                if constitutive=='neo_hookean': stats['hyperelastic_energy']=solver.energy_audit(state)
                print('FV step %d t %.9f dt %.3g rho %.3f..%.3f contact %d VM %.3g Pa energy residual %.2g wall %.1fs'%(
                  count,physical_time,dt,stats['rho_min'],stats['rho_max'],stats['contact_cells'],stats['stress_vm_max'],energy_error,time.perf_counter()-begin),flush=True)
            if physical_time>=next_frame-1e-13:
                save_frame(state,physical_time);next_frame+=1/24;save()
        else: report['status']='completed'
    except KeyboardInterrupt:
        report['status']='operator_stopped';print('FV interrupted; saving last accepted state',flush=True)
    except Exception as exc:
        report['status']='failed';report['failure']=repr(exc);traceback.print_exc()
    if not ts or abs(ts[-1]-physical_time)>1e-13:save_frame(state,physical_time)
    try:report['final_cap_mismatch']=solver.cap_velocity_mismatch(state)
    except Exception as exc:report['final_cap_mismatch_error']=repr(exc)
    report.update(time_reached=physical_time,analytical_freeflight_time=analytic_time,
      physical_steps=count,rejected_attempts=retries,rejection_reasons=rejection_reasons,rejection_examples=rejection_examples,geometry_calls=solver.geometry_calls,
      geometric_site_count=len(state.r),active_cell_count=int((state.mesh.volume>0).sum()),derefinements=removals,agglomerations=merges,skipped_agglomerations=merge_skips,
      control_volume_count=int(torch.unique(solver.groups).numel()),
      dt_min=min_dt if math.isfinite(min_dt) else None,dt_max=max_dt,
      wall_seconds=time.perf_counter()-begin,mass_relative_error=(float(state.q[:,0].sum())-m0)/m0,
      energy_source_balance_error=(float(state.q[:,4].sum())-e0-source_work)/energy_scale,
      boundary_motion_work_defect_joule=boundary_work_defect,boundary_motion_abs_work_defect_joule=boundary_abs_work_defect,friction_heat=friction_heat,plastic_work_sum=float(state.q[:,15].sum()),
      source_work=source_work,energy_error_scale_joule=energy_scale)
    report['device_step']=solver.device_step and solver.contact_model=='plane'
    report['boundary_mode']=boundary_mode
    report['boundary_insertions']=getattr(solver,'boundary_insertions',[])
    report['boundary_prunings']=getattr(solver,'boundary_prunings',[])
    if constitutive=='neo_hookean': report['hyperelastic_final_energy']=solver.energy_audit(state)
    if step_audit:report['hyperelastic_step_audit']=step_audit.report()
    if controller:report['step_control_state']=controller.packed()
    report['device_graph_builds']=solver._device_ops.builds if solver._device_ops is not None else []
    snapshot(report['status']);save()
    print('FV RESULT '+json.dumps({k:report[k] for k in ('status','time_reached','physical_steps','dt_min','dt_max','wall_seconds','mass_relative_error','energy_source_balance_error')}),flush=True)
    return report
