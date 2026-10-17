"""Material-label diagnostics on the existing, frozen TLSPH beam trajectory.

TLSPH here only defines center positions and F, not a unique interior flow.
Compare owner-local affine maps and continuous first-order MLS displacement
extensions. Neither is asserted to be the exact continuum solution. Markers are
passive integration points: no new physical particles, P2G, or image fitting.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation
from scipy.stats import qmc


def cell_markers(X,r,i,power,seed):
    u=qmc.Sobol(3,scramble=True,seed=seed+104729*i).random_base2(power)
    z=1-2*u[:,0]; phi=2*np.pi*u[:,1]
    direction=np.column_stack([np.sqrt(1-z*z)*np.cos(phi),np.sqrt(1-z*z)*np.sin(phi),z])
    delta=X-X[i]; dist=np.linalg.norm(delta,axis=1)
    ids=np.flatnonzero((dist>0)&(dist<r[i]+r))
    normals=delta[ids]/dist[ids,None]
    b=(dist[ids]**2+r[i]**2-r[ids]**2)/(2*dist[ids])
    if np.any(b<=0): raise ValueError('Generator outside its initial cell')
    projection=direction@normals.T
    lengths=np.full_like(projection,np.inf)
    np.divide(b,projection,out=lengths,where=projection>0)
    rho=np.minimum(r[i],lengths.min(axis=1)) if len(ids) else np.full(len(u),r[i])
    points=X[i]+direction*(rho*u[:,2]**(1/3))[:,None]
    # Angular importance quadrature: rho^3 is essential, plain point counts
    # would oversample short directions. Normalize to each saved material mass.
    weight=rho**3; weight/=weight.sum()
    return points,weight


def mls_weights(q,X,V,h):
    delta=(X[None,:,:]-q[:,None,:])/h
    distance=np.linalg.norm(delta,axis=2)
    kernel=np.where(distance<1,1-1.5*distance**2+.75*distance**3,
                    .25*np.maximum(2-distance,0)**3)
    weight=kernel*V[None,:]
    basis=np.concatenate([np.ones((*distance.shape,1)),delta],axis=2)
    moment=np.einsum('qni,qn,qnj->qij',basis,weight,basis,optimize=True)
    rhs=np.broadcast_to(np.array([1.,0,0,0]),(len(q),4))
    beta=np.linalg.solve(moment,rhs[...,None])[...,0]
    N=weight*np.einsum('qni,qi->qn',basis,beta)
    if not np.isfinite(N).all(): raise ValueError('Invalid MLS neighborhood')
    return N


def mls_map(q,X,V,h,positions):
    parts=[]
    for start in range(0,len(q),256):
        points=q[start:start+256]
        N=mls_weights(points,X,V,h)
        parts.append(points[None,:,:]+np.einsum('qn,tnc->tqc',N,positions-X))
    return np.concatenate(parts,axis=1)


def classify(points,centers,radii,owner,h):
    """Exact hard ownership, safely culled by a ball enclosing this point set.

    The violation diagnostic uses the candidate planes and own sphere; it is a
    lower bound on distance to the original cell, not an exact closest distance.
    """
    extent=np.max(np.linalg.norm(points-centers[owner],axis=1))
    ids=np.flatnonzero(np.linalg.norm(centers-centers[owner],axis=1)<=extent+radii.max(axis=0)+1e-12)
    squared=cdist(points,centers[ids],'sqeuclidean')
    outputs=[]
    owner_col=int(np.flatnonzero(ids==owner)[0])
    center_distance=np.linalg.norm(centers[ids]-centers[owner],axis=1)
    center_distance[owner_col]=np.inf
    for r in radii:
        power=squared-r[ids]**2
        selected=np.argmin(power,axis=1)
        minimum=power[np.arange(len(points)),selected]
        labels=ids[selected].copy(); labels[minimum>0]=-1
        status=np.where(labels==owner,0,np.where(labels<0,2,1))
        plane_violation=np.max((power[:,owner_col,None]-power)/(2*center_distance),axis=1)
        sphere_violation=np.sqrt(squared[:,owner_col])-r[owner]
        violation=np.maximum(0,np.maximum(plane_violation,sphere_violation))/h
        outputs.append((status,violation))
    return outputs


def checks(data,h):
    X=data['reference']; V=data['volumes']; r=data['radii0']
    chosen=[0,len(X)//2,len(X)-1]
    rotation=Rotation.from_euler('xyz',[19,-28,37],degrees=True).as_matrix()
    A=np.array([[1.03,.07,0],[0,.98,.02],[0,0,1.01]])
    mls_error=0.; marker_mass_error=0.
    for i in chosen:
        points,weight=cell_markers(X,r,i,9,121)
        N=mls_weights(points,X,V,h)
        mls_error=max(mls_error,float(np.max(abs(points+N@(X@A.T-X)-points@A.T))))
        marker_mass_error=max(marker_mass_error,abs(weight.sum()-1))
        for matrix,scale in [(np.eye(3),1.),(rotation,1.),(1.2*rotation,1.2)]:
            moved=points@matrix.T+[.4,-.2,.1]
            centers=X@matrix.T+[.4,-.2,.1]
            status,_=classify(moved,centers,(r*scale)[None,:],i,h)[0]
            assert not status.any()
        # Compare culling against all centers, not just the owner's neighbors.
        moved=points@A.T
        center=X@A.T
        status,_=classify(moved,center,r[None,:],i,h)[0]
        power=cdist(moved,center,'sqeuclidean')-r*r
        labels=np.argmin(power,axis=1); labels[power.min(axis=1)>0]=-1
        exact=np.where(labels==i,0,np.where(labels<0,2,1))
        assert np.array_equal(status,exact)
    assert mls_error<1e-11
    return dict(mls_affine_reproduction_max_error=mls_error,
                marker_weight_normalization_error=marker_mass_error,
                rest_rotation_uniform_scale_ownership='zero loss',
                ownership_culling='matches exhaustive classification')


def mapping_audit(data,h,frames):
    X=data['reference'];V=data['volumes'];positions=data['positions'][frames]
    mapped=mls_map(X,X,V,h,positions)
    center_error=np.linalg.norm(mapped-positions,axis=2)
    # Check the extension for folding and compare its gradient at generators
    # with the distinct TLSPH deformation estimator. Neither is silently equated.
    eps=h*1e-4
    def gradient(step):
        axes=[]
        for axis in range(3):
            offset=np.eye(3)[axis]*step
            axes.append((mls_map(X+offset,X,V,h,positions)-mls_map(X-offset,X,V,h,positions))/(2*step))
        return np.stack(axes,axis=-1)
    G=gradient(eps); refined=gradient(eps/2)
    return dict(center_position_max_error=float(center_error.max()),
        center_position_max_error_over_h=float(center_error.max()/h),
        center_position_mean_error_over_h=float(center_error.mean()/h),
        det_gradient_at_centers_range=[float(np.linalg.det(G).min()),float(np.linalg.det(G).max())],
        gradient_vs_tlsph_max_frobenius=float(np.linalg.norm(G-data['deformation'][frames],axis=(2,3)).max()),
        gradient_fd_refinement_max_difference=float(np.max(abs(G-refined))),
        note='MLS is affine-exact but not nodally interpolating. Positive determinant was checked at centers only, not proved over the whole domain.')


def run(data,radii,frames,power,seed,h,out,tag,save_view=False,method_names=None):
    X=data['reference'];r0=data['radii0'];V=data['volumes'];tree=cKDTree(X)
    names=['fixed','scaled','joint'] if method_names is None else method_names
    maps=['local_affine','continuous_mls']
    totals=np.zeros((2,len(frames),len(names),5))
    per_cell=np.zeros((2,len(frames),len(names),len(X),3))
    view={f'{a}_{b}':[] for a in maps for b in names}
    for i in range(len(X)):
        points,weight=cell_markers(X,r0,i,power,seed)
        weight*=V[i]
        # Exact MLS support candidates for every point inside the owner's ball.
        ids=np.asarray(tree.query_ball_point(X[i],r0[i]+2*h+1e-12))
        N=mls_weights(points,X[ids],V[ids],h)
        continuous=points[None,:,:]+np.einsum('qn,tnc->tqc',N,data['positions'][frames][:,ids]-X[ids])
        for t,k in enumerate(frames):
            local=data['positions'][k,i]+(points-X[i])@data['deformation'][k,i].T
            for a,moved in enumerate([local,continuous[t]]):
                results=classify(moved,data['positions'][k],np.stack([radii[name][k] for name in names]),i,h)
                for b,(status,violation) in enumerate(results):
                    masses=np.bincount(status,weights=weight,minlength=3)
                    totals[a,t,b,:3]+=masses
                    totals[a,t,b,3]+=weight@violation
                    totals[a,t,b,4]=max(totals[a,t,b,4],violation.max())
                    per_cell[a,t,b,i]=masses/V[i]
                    if save_view and t==0:
                        mask=abs(points[:,2])<.015
                        view[f'{maps[a]}_{names[b]}'].append(np.column_stack([moved[mask,:2],status[mask]]))
        if i%64==0: print(f'{tag}: {i+1}/{len(X)} cells',flush=True)
    rows=[]
    for a,mapping in enumerate(maps):
        for t,k in enumerate(frames):
            for b,name in enumerate(names):
                raw=totals[a,t,b]; mass=V.sum()
                assert abs(raw[:3].sum()/mass-1)<1e-12
                rows.append(dict(mapping=mapping,frame=int(k),time=float(data['times'][k]),method=name,
                    retained_mass_fraction=float(raw[0]/mass),switched_mass_fraction=float(raw[1]/mass),
                    outside_mass_fraction=float(raw[2]/mass),identity_loss_fraction=float((raw[1]+raw[2])/mass),
                    mean_constraint_violation_over_h=float(raw[3]/mass),max_constraint_violation_over_h=float(raw[4]),
                    max_cell_identity_loss_fraction=float(per_cell[a,t,b,:,1:].sum(axis=1).max())))
    result=dict(power=power,seed=seed,markers_per_cell=2**power,total_markers=len(X)*2**power,rows=rows)
    (out/f'{tag}.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(out/f'{tag}_per_cell.npz',fractions=per_cell,frames=frames)
    if save_view: np.savez_compressed(out/'loaded_markers.npz',**{k:np.concatenate(v) for k,v in view.items()})
    return result


def refined_reference_control(data,radii,frames,h,out):
    from experiments.power_cell_volume import RadialVolumes,directions,solve_radii
    r0=data['radii0']
    V=RadialVolumes(data['reference'],directions(18,2901)).evaluate(r0)
    joint=radii['joint'].copy();fits=[]
    for k in frames:
        target=V*np.linalg.det(data['deformation'][k])
        joint[k],info=solve_radii(RadialVolumes(data['positions'][k],directions(14,19)),target,joint[k],r0)
        fits.append(info)
    np.savez_compressed(out/'refined_reference_radii.npz',radii=joint[frames],reference_volumes=V,frames=frames)
    result=run(data,{'joint_refined_reference':joint},frames,13,7001,h,out,'refined_reference',method_names=['joint_refined_reference'])
    result['scope']='Sensitivity only: radius targets use refined geometric reference volumes; the original center/F trajectory and marker masses are unchanged.'
    result['fits']=fits
    (out/'refined_reference.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def plot(out,primary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    for a,mapping in zip(ax,['local_affine','continuous_mls']):
        for name in sorted(set(r['method'] for r in primary['rows'])):
            rows=[r for r in primary['rows'] if r['mapping']==mapping and r['method']==name]
            a.plot([r['time'] for r in rows],[100*r['identity_loss_fraction'] for r in rows],'-o',label=name)
        a.set(xlabel='time',ylabel='material mass leaving original cell (%)',title=mapping)
        a.legend()
    fig.tight_layout();fig.savefig(out/'identity_loss.png',dpi=160);plt.close(fig)
    view=np.load(out/'loaded_markers.npz')
    fig,axes=plt.subplots(2,3,figsize=(15,5),sharex=True,sharey=True)
    for a,mapping in enumerate(['local_affine','continuous_mls']):
        for b,name in enumerate(['fixed','scaled','joint']):
            points=view[f'{mapping}_{name}'];ax=axes[a,b]
            for status,color,label in [(0,'#aaaaaa','retained'),(1,'#e57b18','other cell'),(2,'#92278f','outside foam')]:
                selected=points[points[:,2]==status]
                if status==0:selected=selected[::10]
                ax.scatter(selected[:,0],selected[:,1],s=1 if status==0 else 4,c=color,label=label,rasterized=True)
            ax.set(title=f'{mapping}: {name}',aspect='equal',xlabel='x',ylabel='y')
    axes[0,0].legend(markerscale=3,fontsize=8)
    fig.suptitle('Passive markers, loaded frame; thin reference z slice (not a rendered image)\nGray points thinned 10x; highlighted points shown in full. Use weighted metrics, not visual counts.')
    fig.tight_layout();fig.savefig(out/'loaded_identity.png',dpi=180);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('output/tlsph_material'))
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    source=Path('output/tlsph_irregular/trajectory.npz'); radius_path=Path('output/tlsph_radius_fit/radii.npz')
    data=np.load(source);radii=np.load(radius_path)
    assert str(radii['source_sha256'])==hashlib.sha256(source.read_bytes()).hexdigest()
    frames=np.linspace(0,len(data['times'])-1,7).astype(int);h=.15
    result=dict(scope=__doc__,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                radius_sha256=hashlib.sha256(radius_path.read_bytes()).hexdigest(),h=h,
                checks=checks(data,h),mapping_audit=mapping_audit(data,h,frames))
    if args.resume and (args.out/'results.json').exists():
        previous=json.loads((args.out/'results.json').read_text())
        for key in ['source_sha256','radius_sha256']:
            if previous[key]!=result[key]:raise ValueError('Cannot resume diagnostics with changed input: '+key)
    print('Checks and mapping audit:',json.dumps(result),flush=True)
    started=time.time();runs={}
    for tag,power,seed,which in [('coarse',11,7001,frames),('fine',13,7001,frames),('independent',13,9001,frames[:1])]:
        path=args.out/f'{tag}.json'
        runs[tag]=json.loads(path.read_text()) if args.resume and path.exists() else run(data,radii,which,power,seed,h,args.out,tag,tag=='fine')
    path=args.out/'refined_reference.json'
    refined=json.loads(path.read_text()) if args.resume and path.exists() else refined_reference_control(data,radii,frames,h,args.out)
    result['refined_reference_control']=refined
    path=args.out/'refined_independent.json'
    if args.resume and path.exists():
        refined_repeat=json.loads(path.read_text())
    else:
        saved=np.load(args.out/'refined_reference_radii.npz')
        refined_radii=radii['joint'].copy();refined_radii[saved['frames']]=saved['radii']
        refined_repeat=run(data,{'joint_refined_reference':refined_radii},frames[:1],13,9001,h,args.out,'refined_independent',method_names=['joint_refined_reference'])
    result['refined_reference_independent_seed']=refined_repeat
    result['runs']=runs;result['elapsed_seconds']=time.time()-started
    result['summary']={}
    for mapping in ['local_affine','continuous_mls']:
        result['summary'][mapping]={}
        for name in ['fixed','scaled','joint']:
            rows=[r for r in runs['fine']['rows'] if r['mapping']==mapping and r['method']==name]
            coarse=[r for r in runs['coarse']['rows'] if r['mapping']==mapping and r['method']==name]
            repeat=[r for r in runs['independent']['rows'] if r['mapping']==mapping and r['method']==name][0]
            result['summary'][mapping][name]=dict(
                mean_identity_loss_fraction=float(np.mean([r['identity_loss_fraction'] for r in rows])),
                mean_switched_mass_fraction=float(np.mean([r['switched_mass_fraction'] for r in rows])),
                mean_outside_mass_fraction=float(np.mean([r['outside_mass_fraction'] for r in rows])),
                worst_identity_loss_fraction=max(r['identity_loss_fraction'] for r in rows),
                max_coarse_fine_difference=max(abs(a['identity_loss_fraction']-b['identity_loss_fraction']) for a,b in zip(rows,coarse)),
                loaded_independent_seed_difference=abs(rows[0]['identity_loss_fraction']-repeat['identity_loss_fraction']))
        rows=[r for r in refined['rows'] if r['mapping']==mapping]
        repeat=[r for r in refined_repeat['rows'] if r['mapping']==mapping][0]
        result['summary'][mapping]['joint_refined_reference']=dict(
            mean_identity_loss_fraction=float(np.mean([r['identity_loss_fraction'] for r in rows])),
            mean_switched_mass_fraction=float(np.mean([r['switched_mass_fraction'] for r in rows])),
            mean_outside_mass_fraction=float(np.mean([r['outside_mass_fraction'] for r in rows])),
            worst_identity_loss_fraction=max(r['identity_loss_fraction'] for r in rows),
            loaded_independent_seed_difference=abs(rows[0]['identity_loss_fraction']-repeat['identity_loss_fraction']))
    (args.out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    plot(args.out,dict(rows=runs['fine']['rows']+refined['rows']))
    print('SUMMARY',json.dumps(result['summary']),flush=True)


if __name__=='__main__':main()
