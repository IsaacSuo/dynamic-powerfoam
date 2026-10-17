"""Nonlinear load continuation on the original irregular TLSPH beam.

Actual force equilibria, not geometrically amplified trajectories. The original
reference masses/material/neighbors are preserved. Fitted radii do not feed back
into stress. Saved `times` contain load magnitudes for helper compatibility;
these snapshots are static equilibria, not an animation of physical time.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from scipy.linalg import lu_factor,lu_solve
from experiments.tlsph_core import TLSPH,beam
from experiments.tlsph_probe import tangent
from experiments.power_cell_volume import RadialVolumes,directions,solve_radii
from experiments.tlsph_radius_fit import errors
from experiments.tlsph_material_probe import run as material_run,mapping_audit,cell_markers,mls_map


def equilibrium(model,fixed,tip,start,load,max_iterations=24):
    x=start.copy();external=np.zeros_like(x);external[tip,1]=-load/tip.sum()
    free=np.flatnonzero(np.repeat(~fixed,3));history=[]
    for iteration in range(max_iterations+1):
        internal,data=model.evaluate(x)
        residual=(internal+external).ravel()[free]
        norm=float(np.linalg.norm(residual));history.append(norm/load)
        if norm/load<1e-7:
            U,s,Vt=np.linalg.svd(data['F']);R=U@Vt
            angle=np.arccos(np.clip((np.trace(R,axis1=1,axis2=2)-1)/2,-1,1))
            reaction=-internal[fixed].sum(0)
            return x,dict(load=load,iterations=iteration,relative_residual=norm/load,history=history,
                mean_tip_y_displacement=float(np.mean(x[tip,1]-model.X[tip,1])),
                detF_range=[float(data['J'].min()),float(data['J'].max())],
                principal_stretch_range=[float(s.min()),float(s.max())],
                max_rotation_degrees=float(np.rad2deg(angle.max())),
                reaction_plus_external_norm_over_load=float(np.linalg.norm(reaction+external.sum(0))/load))
        if iteration==max_iterations: break
        K,_=tangent(model,fixed,x)
        increment=lu_solve(lu_factor(K),residual)
        for fraction in 2.**-np.arange(12):
            trial=x.copy();trial.ravel()[free]+=fraction*increment
            try: new_residual=(model.evaluate(trial)[0]+external).ravel()[free]
            except ValueError:continue
            if np.linalg.norm(new_residual)<norm:
                x=trial;break
        else:raise RuntimeError(f'Newton stalled at load {load}, residual/load {norm/load}')
    raise RuntimeError(f'Newton exceeded iterations at load {load}: {history}')


def advance(model,fixed,tip,start,old_load,new_load,log,depth=0):
    try:
        state,info=equilibrium(model,fixed,tip,start,new_load)
        log.append(info);print('Equilibrium:',json.dumps(info),flush=True)
        return state,info
    except (RuntimeError,ValueError,np.linalg.LinAlgError) as e:
        log.append(dict(load=new_load,failed_attempt=str(e),depth=depth))
        print('Refining load increment:',str(e),flush=True)
        if depth>=6:raise
        middle=.5*(old_load+new_load)
        x,_=advance(model,fixed,tip,start,old_load,middle,log,depth+1)
        return advance(model,fixed,tip,x,middle,new_load,log,depth+1)


def mechanics(source,out,resume):
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    if resume and (out/'mechanics.json').exists():
        results=json.loads((out/'mechanics.json').read_text())
        assert results['source_sha256']==digest
        return np.load(out/'trajectory.npz'),results
    original=np.load(source);X=original['reference'];V=original['volumes'];fixed=original['fixed'];tip=beam(4)[3]
    model=TLSPH(X,V,.15,hourglass=10.)
    loads=np.array([.16,.4,.8,1.6,3.2])
    x=original['positions'][0].copy();previous=.16;states=[];Fs=[]
    result=dict(scope=__doc__,source_sha256=digest,material=dict(young=1000.,poisson=.3,density=1.,h=.15,hourglass=10.),
                states=[],continuation=[])
    for load in loads:
        x,info=advance(model,fixed,tip,x,previous,float(load),result['continuation'])
        result['states'].append(info);states.append(x.copy());Fs.append(model.deformation(x));previous=float(load)
        np.savez_compressed(out/'trajectory.npz',reference=X,volumes=V,radii0=original['radii0'],fixed=fixed,
                            positions=np.asarray(states),deformation=np.asarray(Fs),times=loads[:len(states)],loads=loads[:len(states)])
        (out/'mechanics_partial.json').write_text(json.dumps(result,indent=2)+'\n')
    # Check the largest-load solution using smaller increments from the 1.6 state.
    check=states[-2].copy();old=loads[-2];check_log=[]
    for load in [2.,2.4,2.8,3.2]:
        check,_=advance(model,fixed,tip,check,float(old),load,check_log);old=load
    result['load_increment_refinement']=dict(
        max_position_difference=float(np.max(np.linalg.norm(check-states[-1],axis=1))),
        difference_over_reference_spacing=float(np.max(np.linalg.norm(check-states[-1],axis=1))/.125),
        steps=check_log)
    (out/'mechanics.json').write_text(json.dumps(result,indent=2)+'\n')
    return np.load(out/'trajectory.npz'),result


def geometry(data,out,resume):
    if resume and (out/'geometry.json').exists():
        return np.load(out/'radii.npz'),json.loads((out/'geometry.json').read_text())
    X=data['reference'];r0=data['radii0'];J=np.linalg.det(data['deformation'])
    refinedV=RadialVolumes(X,directions(18,2901)).evaluate(r0)
    sequences=dict(fixed=np.tile(r0,(len(J),1)),scaled=r0*J**(1/3),joint=[],joint_refined=[])
    result=dict(target='Saved mechanical V0 times J; refined target control changes only radius targets, not mechanics.',frames=[])
    for k,p in enumerate(data['positions']):
        ev=RadialVolumes(p,directions(14,19));validation=RadialVolumes(p,directions(17,901))
        frame=dict(load=float(data['loads'][k]),fits={},methods={})
        for name,ref in [('joint',data['volumes']),('joint_refined',refinedV)]:
            initial=sequences['scaled'][k] if k==0 else sequences[name][-1]
            r,info=solve_radii(ev,ref*J[k],initial,r0)
            sequences[name].append(r);frame['fits'][name]=info
        for name,sequence in sequences.items():
            volume=validation.evaluate(sequence[k])
            frame['methods'][name]=dict(against_mechanical_target=errors(volume,data['volumes']*J[k]),
                                       against_refined_geometry_target=errors(volume,refinedV*J[k]))
        result['frames'].append(frame)
        print('Geometry:',json.dumps(frame),flush=True)
        (out/'geometry_partial.json').write_text(json.dumps(result,indent=2)+'\n')
    sequences={k:np.asarray(v) for k,v in sequences.items()}
    result['radius_range_over_rest']={k:[float((v/r0).min()),float((v/r0).max())] for k,v in sequences.items()}
    np.savez_compressed(out/'radii.npz',**sequences,refined_reference_volumes=refinedV)
    (out/'geometry.json').write_text(json.dumps(result,indent=2)+'\n')
    return np.load(out/'radii.npz'),result


def make_plots(out,data,mechanical,geometric,material):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,3,figsize=(15,4))
    loads=data['loads']
    ax[0].plot(loads,[-s['mean_tip_y_displacement'] for s in mechanical['states']],'-o')
    ax[0].set(xlabel='applied tip force',ylabel='downward tip displacement',title='Actual nonlinear equilibria')
    for name in ['fixed','scaled','joint_refined']:
        color={'fixed':'tab:blue','scaled':'tab:orange','joint_refined':'tab:green'}[name]
        ax[1].plot(loads,[100*f['methods'][name]['against_refined_geometry_target']['relative_l1'] for f in geometric['frames']],'-o',label=name,color=color)
        for mapping,style in [('local_affine','-'),('continuous_mls','--')]:
            rows=[r for r in material['rows'] if r['method']==name and r['mapping']==mapping]
            short='local' if mapping=='local_affine' else 'MLS'
            ax[2].plot(loads,[100*r['identity_loss_fraction'] for r in rows],style+'o',label=f'{name} ({short})',color=color)
    ax[1].set(xlabel='applied tip force',ylabel='volume error (%)',yscale='log',title='Independent cell-volume validation')
    ax[2].set(xlabel='applied tip force',ylabel='mass leaving original cell (%)',title='Material identity under two extensions')
    ax[1].legend(fontsize=8);ax[2].legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'load_comparison.png',dpi=160);plt.close(fig)


def additional_checks(data,radii,out):
    """Independent largest-load volume refinement and interior MLS spot checks."""
    k=len(data['loads'])-1;J=np.linalg.det(data['deformation'][k])
    ev=RadialVolumes(data['positions'][k],directions(18,1901))
    result=dict(volume_refinement=dict(exponent=18,seed=1901,methods={},
        note='joint uses saved mechanical V0; other methods use refined geometric V0 for this check.'))
    for name in ['fixed','scaled','joint','joint_refined']:
        ref=data['volumes'] if name=='joint' else radii['refined_reference_volumes']
        result['volume_refinement']['methods'][name]=errors(ev.evaluate(radii[name][k]),ref*J)
    X=data['reference'];V=data['volumes'];h=.15
    points=np.concatenate([cell_markers(X,data['radii0'],i,4,7301)[0] for i in range(len(X))])
    eps=h*1e-4;axes=[]
    for a in range(3):
        d=np.eye(3)[a]*eps
        axes.append((mls_map(points+d,X,V,h,data['positions'][-1:])-mls_map(points-d,X,V,h,data['positions'][-1:]))/(2*eps))
    det=np.linalg.det(np.stack(axes,axis=-1))
    result['mls_interior_gradient_spot_check']=dict(points=len(points),det_range=[float(det.min()),float(det.max())],
        nonpositive_count=int(np.sum(det<=0)),note='Interior samples at largest load only, not a proof of a globally injective motion.')
    (out/'additional_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('output/tlsph_load_sweep'))
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--no-render',action='store_true')
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True);start=time.time()
    data,mechanical=mechanics(Path('output/tlsph_irregular/trajectory.npz'),args.out,args.resume)
    radii,geometric=geometry(data,args.out,args.resume)
    path=args.out/'additional_checks.json'
    extra=json.loads(path.read_text()) if args.resume and path.exists() else additional_checks(data,radii,args.out)
    frames=np.arange(len(data['loads']));names=['fixed','scaled','joint','joint_refined']
    runs={}
    for tag,power,seed,which in [('coarse',11,7101,frames),('fine',13,7101,frames),('independent',13,9101,frames[-1:])]:
        path=args.out/f'{tag}.json'
        runs[tag]=json.loads(path.read_text()) if args.resume and path.exists() else material_run(data,radii,which,power,seed,.15,args.out,tag,method_names=names)
    audit=mapping_audit(data,.15,frames)
    result=dict(mechanics=mechanical,geometry=geometric,material=runs,mapping_audit=audit,additional_checks=extra,
                elapsed_seconds_before_render=time.time()-start)
    (args.out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    make_plots(args.out,data,mechanical,geometric,runs['fine'])
    if not args.no_render:
        from experiments.tlsph_render import render_trajectory
        from PIL import Image
        images=render_trajectory(data,data['radii0'],args.out/'render',radii['joint_refined'],
            label='force equilibria; joint volume radii',frame_label='load',auto_frame=True,frame_duration_ms=600)
        sheet=Image.new('RGB',(512*len(images),320),'white')
        for k,im in enumerate(images):sheet.paste(im,(512*k,0))
        sheet.save(args.out/'load_frames.png')
    print('Load sweep complete.',flush=True)


if __name__=='__main__':main()
