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


@torch.no_grad()
def run(args,X0,r0,gt):
    wp.init(); torch.cuda.synchronize(); begin=time.perf_counter()
    resume=None
    if args.fv_resume is not None:
        resume=torch.load(args.fv_resume,weights_only=False,map_location='cpu')
        if resume['material']!=gt or resume.get('original_cell_count',len(resume['p']))!=len(X0):
            raise ValueError('FV resume material/original-cell-count mismatch')
        for key,wanted in [('friction',args.mu),('ground',-.05),('rho0',1000.)]:
            if key in resume and resume[key]!=wanted: raise ValueError('FV resume '+key+' mismatch')
    solver=SolidALE(resume['p'] if resume else X0,resume['r'] if resume else r0,
                    gt['E'],gt['nu'],gt['ys'],friction=args.mu,interior_radii=args.fv_interior_radii)
    state=solver.set_velocity(solver.initial,gt['v'])
    if resume: state.q=resume['q'].to(state.q.device)
    ids=(resume.get('ids',torch.arange(len(state.r))) if resume else torch.arange(len(state.r))).to(state.p.device)
    reference_volume=float(resume.get('reference_volume',state.mesh.volume.median())) if resume else float(state.mesh.volume.median())
    removals=[]
    report={'status':'running','kind':'experimental direct ALE solid FV adaptation, not original paper reproduction',
      'foam':str(args.foam),'scene':args.scene,'cell_count':len(X0),'material':gt,
      'known_contact_defect':'Legacy whole-ball contact can load a buried bottom point and does not vanish with actual exposed contact area; diagnostic only',
      'physics_changes':['Wilkins/Jaumann deviatoric stress, perfect von Mises plasticity',
          'stiffened-gas EOS gamma=2 matching initial bulk modulus (different from k1 log-volume energy)',
          'integrated traction-free spherical-cap boundary; pointwise kinematic mismatch is measured',
          'Hertz ground normal force plus split Coulomb sliding impulse (no Mindlin stick history)',
          'explicit SSPRK2 with reported physical substeps; no F/L parallel volume variables'],
      'derefine':args.fv_derefine,'min_volume_ratio':args.fv_min_volume_ratio,
      'interior_radii':args.fv_interior_radii,'cfl':args.fv_cfl,'max_physical_dt':args.dt,'target_time':args.frames/24,
      'budget_wall_seconds':args.fv_budget,'step_limit':args.fv_steps,'tests_do_not_prove':'accuracy, robustness or speed for complete elastoplastic landing'}
    ts=[]; xs=[]; rs=[]; masses=[]; vs=[]; stresses=[]; Ls=[]; eps=[]; works=[]; hist=[]
    m0=float(state.q[:,0].sum()); e0=float(state.q[:,4].sum())
    energy_scale=max(float((.5*state.q[:,1:4]**2/state.q[:,:1]).sum()),m0*9.8*.5,1.)
    physical_time=float(resume['time']) if resume else 0.
    report['start_time']=physical_time
    report['resume_from']=str(args.fv_resume) if resume else None
    count=0; retries=0; source_work=0.; friction_heat=0.; next_frame=(math.floor(physical_time*24+1e-9)+1)/24
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
    def save():
        np.savez(args.save/('frames_'+args.tag+'.npz'),t=np.array(ts),x=np.array(xs),r=np.array(rs),
            mass=np.array(masses),v=np.array(vs),stress=np.array(stresses),L=np.array(Ls),ep=np.array(eps),plastic_work=np.array(works))
        (args.save/('fv_solid_'+args.tag+'.json')).write_text(json.dumps(report,indent=2)+'\n')
        (args.save/('fv_steps_'+args.tag+'.json')).write_text(json.dumps(hist,indent=2)+'\n')
    def snapshot(reason):
        torch.save({'p':state.p.cpu(),'r':state.r.cpu(),'q':state.q.cpu(),'time':physical_time,
           'steps':count+(int(resume['steps']) if resume else 0),'material':gt,'reason':reason,
           'ids':ids.cpu(),'original_cell_count':len(X0),'reference_volume':reference_volume,
           'interior_radii':args.fv_interior_radii,'friction':args.mu,'ground':solver.ground,'rho0':solver.rho0,'frames_file':str((args.save/('frames_'+args.tag+'.npz')).resolve())},args.save/('fv_state_'+args.tag+'.pt'))
    if resume and resume.get('frames_file'):
        old=np.load(resume['frames_file'])
        for name,target in [('t',ts),('x',xs),('r',rs),('mass',masses),('v',vs),('stress',stresses),('L',Ls),('ep',eps),('plastic_work',works)]:
            target.extend(list(old[name]))
    else: save_frame(state,physical_time)
    print('FV CONTACT IS KNOWN INVALID FOR VANISHING CELLS: this run is diagnostic reproduction only.',flush=True)
    print('FV SOLID: %d cells; new constitutive law and approximate spherical boundary; full landing is an experiment'%len(X0),flush=True)
    # Exact uniform free flight, only for this uniform unstressed initial scene.
    # Geometry is still recut at saved states; no elastic/contact dynamics skipped.
    if not args.fv_no_freeflight and resume is None:
        gap=float((state.p[:,1]-state.r-solver.ground)[state.mesh.surface].min())
        vy=float(gt['v'][1]); gy=float(solver.g[1]); hit=0.
        if gap>0 and gy<0:
            hit=(-vy-math.sqrt(vy*vy-2*gy*gap))/gy
        hit=min(hit,args.frames/24)
        p_initial=state.p.clone(); q_initial=state.q.clone(); v_initial=torch.tensor(gt['v'],dtype=state.p.dtype,device=state.p.device)
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
    try:
        first=solver.rhs(state)
        report['initial_after_flight_rhs']=first[3]
        report['initial_after_flight_cap_mismatch']=solver.cap_velocity_mismatch(state)
        print('First stable-step estimate %.3g s (macro %.3g s), limiting cell %d'%(args.fv_cfl*first[3]['dt_scale'],args.dt,first[3]['wave_cell']),flush=True)
        while physical_time<args.frames/24-1e-13:
            if count>=args.fv_steps or time.perf_counter()-begin>=args.fv_budget:
                report['status']='budget_stopped_not_complete';break
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
            first=solver.rhs(state)
            dt=min(args.dt,args.fv_cfl*first[3]['dt_scale'],args.frames/24-physical_time,next_frame-physical_time)
            if dt<1e-12:raise RuntimeError('physical dt below 1e-12; not advanced')
            for attempt in range(9):
                if dt<1e-12: raise RuntimeError('retry dt below 1e-12; state not advanced')
                try:
                    candidate,stats=solver.advance(state,dt,first,args.fv_cfl)
                    break
                except RuntimeError:
                    retries+=1
                    if attempt==8:raise
                    dt*=.5
            state=candidate; physical_time+=dt;count+=1
            min_dt=min(min_dt,dt);max_dt=max(max_dt,dt)
            source_work+=stats['source_work'];friction_heat+=stats['friction_heat']
            mass_error=(float(state.q[:,0].sum())-m0)/m0
            energy_error=(float(state.q[:,4].sum())-e0-source_work)/energy_scale
            stats.update(step=count,time=physical_time,mass_error=mass_error,energy_source_balance_error=energy_error,wall_seconds=time.perf_counter()-begin)
            hist.append(stats)
            if count==1 or count%20==0:
                print('FV step %d t %.9f dt %.3g rho %.3f..%.3f contact %d VM %.3g Pa energy residual %.2g wall %.1fs'%(
                  count,physical_time,dt,stats['rho_min'],stats['rho_max'],stats['contact_cells'],stats['stress_vm_max'],energy_error,time.perf_counter()-begin),flush=True)
            if physical_time>=next_frame-1e-13:
                save_frame(state,physical_time);next_frame+=1/24;save()
        else: report['status']='completed'
    except Exception as exc:
        report['status']='failed';report['failure']=repr(exc);traceback.print_exc()
    if not ts or abs(ts[-1]-physical_time)>1e-13:save_frame(state,physical_time)
    try:report['final_cap_mismatch']=solver.cap_velocity_mismatch(state)
    except Exception as exc:report['final_cap_mismatch_error']=repr(exc)
    report.update(time_reached=physical_time,analytical_freeflight_time=analytic_time,
      physical_steps=count,rejected_attempts=retries,geometry_calls=solver.geometry_calls,
      active_cell_count=len(state.r),derefinements=removals,
      dt_min=min_dt if math.isfinite(min_dt) else None,dt_max=max_dt,
      wall_seconds=time.perf_counter()-begin,mass_relative_error=(float(state.q[:,0].sum())-m0)/m0,
      energy_source_balance_error=(float(state.q[:,4].sum())-e0-source_work)/energy_scale,
      friction_heat=friction_heat,plastic_work_sum=float(state.q[:,15].sum()),
      source_work=source_work,energy_error_scale_joule=energy_scale)
    snapshot(report['status']);save()
    print('FV RESULT '+json.dumps({k:report[k] for k in ('status','time_reached','physical_steps','dt_min','dt_max','wall_seconds','mass_relative_error','energy_source_balance_error')}),flush=True)
    return report
