"""Fit bounded power-cell radii on a frozen TLSPH trajectory.

Targets are exactly the saved mechanical reference volumes times det(F).
Centers, material identities, masses, constitutive model, and F are unchanged.
Training: 2^14 angular Sobol rays, seed 19, exact radial boundary distances.
Validation: 2^17 rays, independent seed 901; loaded frame refined separately.
No rendered-image loss and no feedback of fitted radii into the mechanics.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from experiments.power_cell_volume import RadialVolumes,directions,solve_radii,checks


def errors(volume,target):
    rel=volume/target-1
    return dict(relative_l1=float(np.sum(abs(volume-target))/sum(target)),
        max_cell_relative_error=float(np.max(abs(rel))),
        p95_cell_relative_error=float(np.percentile(abs(rel),95)),
        total_volume_signed_error=float((sum(volume)-sum(target))/sum(target)),
        minimum_cell_volume=float(volume.min()),empty_cells=int(np.sum(volume<=0)))


def reference_sensitivity(data):
    """Separate a geometry-target sensitivity check from the fixed-mass run."""
    r=data['radii0']; p=data['positions'][0]
    J=np.linalg.det(data['deformation'][0])
    reference=RadialVolumes(data['reference'],directions(18,2901)).evaluate(r)
    target=reference*J
    fit,info=solve_radii(RadialVolumes(p,directions(14,19)),target,r*J**(1/3),r)
    ev=RadialVolumes(p,directions(18,1901))
    result=dict(scope='Loaded-frame sensitivity diagnostic only: replace noisy saved rest-volume targets by refined geometric rest-volume targets, hold the same center/F trajectory. Does not rerun or modify the mechanical simulation.',
        reference_exponent=18,reference_seed=2901,validation_exponent=18,validation_seed=1901,
        saved_reference_error=errors(reference,data['volumes']),fit=info,methods={})
    for name,rr in [('fixed',r),('scaled',r*J**(1/3)),('joint',fit)]:
        result['methods'][name]=errors(ev.evaluate(rr),target)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trajectory',type=Path,default=Path('output/tlsph_irregular/trajectory.npz'))
    parser.add_argument('--out',type=Path,default=Path('output/tlsph_radius_fit'))
    parser.add_argument('--resume',action='store_true',help='Reuse the complete saved radius trajectory with matching source hash')
    parser.add_argument('--no-render',action='store_true')
    args=parser.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    started=time.time(); data=np.load(args.trajectory)
    digest=hashlib.sha256(args.trajectory.read_bytes()).hexdigest()
    r0=data['radii0']; J=np.linalg.det(data['deformation']); target=data['volumes'][None,:]*J
    assert np.all(J>0)
    result=dict(scope=__doc__,source=str(args.trajectory),source_sha256=digest,
        checks=checks(),training=dict(exponent=14,seed=19,tolerance=1e-9),
        validation=dict(exponent=17,seed=901),fits=[],frames=[])
    sequences=dict(fixed=np.tile(r0,(len(J),1)),scaled=r0*J**(1/3))
    rays=directions(14,19)
    if args.resume:
        saved=np.load(args.out/'radii.npz')
        assert str(saved['source_sha256'])==digest
        sequences['joint']=saved['joint']
        result['fits']=json.loads((args.out/'results.json').read_text())['fits']
    else:
        fitted=[]; previous=sequences['scaled'][0]
        for k,p in enumerate(data['positions']):
            ev=RadialVolumes(p,rays)
            r,info=solve_radii(ev,target[k],previous,r0)
            info.update(frame=k,time=float(data['times'][k]))
            result['fits'].append(info); fitted.append(r); previous=r
            if k%8==0: print(f'Fit {k+1}/{len(J)}: {info}',flush=True)
        sequences['joint']=np.asarray(fitted)
        np.savez_compressed(args.out/'radii.npz',**sequences,target_volumes=target,
                            times=data['times'],source_sha256=digest)
        (args.out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    result['radius_statistics']={}
    for name,r in sequences.items():
        increments=np.diff(r,axis=0)/r0
        result['radius_statistics'][name]=dict(
            min_ratio_to_rest=float(np.min(r/r0)),max_ratio_to_rest=float(np.max(r/r0)),
            max_adjacent_frame_change_over_rest=float(np.max(abs(increments))),
            max_second_difference_over_rest=float(np.max(abs(np.diff(increments,axis=0)))),
            bound_hits=int(np.sum((r/r0<=.500001)|(r/r0>=1.999999))),
            max_change_from_scaled_over_rest=float(np.max(abs(r-sequences['scaled'])/r0)))
    validation_rays=directions(17,901)
    rest=RadialVolumes(data['reference'],validation_rays).evaluate(r0)
    result['saved_reference_volume_vs_independent_geometry']=errors(rest,data['volumes'])
    # The saved rest volumes already have numerical integration error. Do not
    # silently replace them: they also set mass in the frozen mechanics run.
    for k in np.linspace(0,len(J)-1,7).astype(int):
        ev=RadialVolumes(data['positions'][k],validation_rays)
        frame=dict(frame=int(k),time=float(data['times'][k]),methods={})
        for name,r in sequences.items():
            frame['methods'][name]=errors(ev.evaluate(r[k]),target[k])
        result['frames'].append(frame)
        print('Independent validation:',json.dumps(frame),flush=True)
        (args.out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    high=RadialVolumes(data['positions'][0],directions(18,1901))
    result['loaded_frame_refinement']=dict(exponent=18,seed=1901,methods={})
    for name,r in sequences.items():
        result['loaded_frame_refinement']['methods'][name]=errors(high.evaluate(r[0]),target[0])
    result['summary']={name:dict(
        mean_relative_l1=float(np.mean([f['methods'][name]['relative_l1'] for f in result['frames']])),
        worst_relative_l1=float(max(f['methods'][name]['relative_l1'] for f in result['frames'])),
        worst_cell_relative_error=float(max(f['methods'][name]['max_cell_relative_error'] for f in result['frames'])))
        for name in sequences}
    result['elapsed_seconds_before_render']=time.time()-started
    (args.out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    sensitivity=reference_sensitivity(data)
    (args.out/'reference_sensitivity.json').write_text(json.dumps(sensitivity,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,3,figsize=(14,4))
    for name,r in sequences.items():
        times=[f['time'] for f in result['frames']]
        ax[0].semilogy(times,[100*f['methods'][name]['relative_l1'] for f in result['frames']],'-o',label=name)
        ax[1].plot(data['times'],100*np.max(abs(r/r0-1),axis=1),label=name)
        ax[2].plot(data['times'][1:],100*np.max(abs(np.diff(r,axis=0)/r0),axis=1),label=name)
    ax[0].set(xlabel='time',ylabel='sum absolute volume error / target volume (%)',title='Independent geometry validation')
    ax[1].set(xlabel='time',ylabel='max radius change from rest (%)',title='Radius magnitude')
    ax[2].set(xlabel='time',ylabel='max adjacent-frame radius change (%)',title='Temporal variation (no smoothing)')
    for a in ax: a.legend()
    fig.tight_layout(); fig.savefig(args.out/'comparison.png',dpi=150); plt.close(fig)
    print('FINAL',json.dumps(dict(summary=result['summary'],radius_statistics=result['radius_statistics'],
        loaded_frame_refinement=result['loaded_frame_refinement'])),flush=True)
    if not args.no_render:
        from experiments.tlsph_render import render_trajectory
        from PIL import Image
        rendered={name:render_trajectory(data,r0,args.out/name,r,label=name+' radii') for name,r in sequences.items()}
        images=[]
        for k in range(len(J)):
            im=Image.new('RGB',(512*3,320),'white')
            for col,name in enumerate(sequences): im.paste(rendered[name][k],(512*col,0))
            images.append(im)
        images[0].save(args.out/'comparison.gif',save_all=True,append_images=images[1:],duration=50,loop=0)
        images[0].save(args.out/'loaded_comparison.png')
        print('Saved original-renderer three-method comparison.',flush=True)


if __name__=='__main__': main()
