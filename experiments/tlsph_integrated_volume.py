"""Audit volume consistency on frozen load states, without rerunning mechanics.

Compare the TLSPH determinant estimator with the integral of det(grad(phi))
for the SAME continuous MLS extension used by tlsph_material_probe. MLS is a
diagnostic extension, not a unique physical motion or a nodal interpolant.
The integral is signed; sampled positive determinants do not prove injectivity.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from experiments.tlsph_material_probe import cell_markers, mls_map
from experiments.power_cell_volume import RadialVolumes, directions
from experiments.tlsph_radius_fit import errors


def mls_gradient(q, X, V, h, positions):
    """Analytic gradient of q + sum_j N_j(q) * (x_j-X_j), shape (t,q,3,3).

    Let p=[1,(X_j-q)/h], M=sum W pp^T, A=M^-1 sum W p u^T,
    beta=M^-1 e0. Differentiating the normal equations gives
      grad(phi) = I + A_linear^T/h
                    + sum_j (p_j^T beta) (u_j-p_j^T A) outer grad(W_j).
    Terms involving sum W*(u-pA) vanish by the constant normal equation.
    This includes derivatives of the MLS fit; freezing its weights is wrong.
    """
    delta=(X[None]-q[:,None])/h
    s=np.linalg.norm(delta,axis=-1)
    kernel=np.where(s<1,1-1.5*s*s+.75*s**3,.25*np.maximum(2-s,0)**3)
    derivative=np.where(s<1,-3*s+2.25*s*s,-.75*np.maximum(2-s,0)**2)
    W=kernel*V[None]
    radial=np.divide(derivative,s,out=np.zeros_like(s),where=s>0)
    dW=-radial[...,None]*delta*V[None,:,None]/h
    p=np.concatenate([np.ones((*s.shape,1)),delta],axis=-1)
    M=np.einsum('qni,qn,qnj->qij',p,W,p,optimize=True)
    u=positions-X
    B=np.einsum('qni,qn,tnc->qitc',p,W,u,optimize=True)
    e0=np.zeros((len(q),4,1)); e0[:,0,0]=1
    solution=np.linalg.solve(M,np.concatenate([B.reshape(len(q),4,-1),e0],axis=-1))
    A=solution[...,:-1].reshape(B.shape); beta=solution[...,-1]
    residual=u[None].transpose(0,2,1,3)-np.einsum('qni,qitc->qntc',p,A,optimize=True)
    factor=np.einsum('qni,qi->qn',p,beta)
    G=np.einsum('qn,qntc,qna->tqca',factor,residual,dW,optimize=True)
    G+=A[:,1:].transpose(2,0,3,1)/h+np.eye(3)
    if not np.isfinite(G).all():
        raise ValueError('Nonfinite MLS gradient')
    return G


def checks(data,h):
    X=data['reference']; V=data['volumes']; r=data['radii0']
    q=np.concatenate([cell_markers(X,r,i,5,211)[0] for i in [0,17,135,254,271]])
    positions=data['positions'][[-1]]
    G=mls_gradient(q,X,V,h,positions)
    fd=[]
    for eps in [h*1e-4,h*1e-5]:
        numerical=np.stack([(mls_map(q+np.eye(3)[a]*eps,X,V,h,positions)-
                             mls_map(q-np.eye(3)[a]*eps,X,V,h,positions))/(2*eps)
                            for a in range(3)],axis=-1)
        fd.append(float(np.max(abs(G-numerical))))
    A=np.array([[1.03,.07,.01],[-.02,.98,.02],[.01,-.01,1.01]])
    affine=mls_gradient(q,X,V,h,(X@A.T+[.3,-.2,.1])[None])
    affine_error=float(np.max(abs(affine-A)))
    rest_error=float(np.max(abs(mls_gradient(q,X,V,h,X[None])-np.eye(3))))
    assert fd[-1]<1e-7 and affine_error<1e-10 and rest_error<1e-14
    # Candidate culling must retain every basis function with nonzero support.
    i=271; points,_=cell_markers(X,r,i,5,211)
    ids=np.flatnonzero(np.linalg.norm(X-X[i],axis=1)<=r[i]+2*h+1e-12)
    culled=mls_gradient(points,X[ids],V[ids],h,positions[:,ids])
    culling_error=float(np.max(abs(culled-mls_gradient(points,X,V,h,positions))))
    assert culling_error<1e-10
    return dict(fd_max_errors=fd,fd_steps=[h*1e-4,h*1e-5],
                affine_gradient_max_error=affine_error,rest_gradient_max_error=rest_error,
                support_culling_max_error=culling_error)


def integrate(data,reference_volumes,h,power,seed,frames,out,tag):
    X=data['reference']; V=data['volumes']; r=data['radii0']
    tree=cKDTree(X); nt=len(frames); n=len(X)
    mean_J=np.zeros((nt,n)); mean_G=np.zeros((nt,n,3,3))
    minimum=np.full((nt,n),np.inf); maximum=np.full((nt,n),-np.inf)
    nonpositive=np.zeros((nt,n)); started=time.time()
    for i in range(n):
        points,weights=cell_markers(X,r,i,power,seed)
        ids=np.asarray(tree.query_ball_point(X[i],r[i]+2*h+1e-12))
        for start in range(0,len(points),256):
            q=points[start:start+256]; w=weights[start:start+256]
            G=mls_gradient(q,X[ids],V[ids],h,data['positions'][frames][:,ids])
            J=np.linalg.det(G)
            mean_J[:,i]+=J@w
            mean_G[:,i]+=np.einsum('tqca,q->tca',G,w)
            minimum[:,i]=np.minimum(minimum[:,i],J.min(axis=1))
            maximum[:,i]=np.maximum(maximum[:,i],J.max(axis=1))
            nonpositive[:,i]+=(J<=0)@w
        if i%64==0:
            print(f'{tag}: {i+1}/{n} cells, {time.time()-started:.1f}s',flush=True)
    volume=mean_J*reference_volumes[None]
    np.savez_compressed(out/f'{tag}.npz',volumes=volume,mean_J=mean_J,mean_G=mean_G,
                        minimum_J=minimum,maximum_J=maximum,nonpositive_fraction=nonpositive,
                        frames=frames,power=power,seed=seed)
    return dict(volumes=volume,mean_J=mean_J,mean_G=mean_G,minimum_J=minimum,
                maximum_J=maximum,nonpositive_fraction=nonpositive,frames=frames)


def plot(out,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    loads=[r['load'] for r in rows]
    for key,label in [('mls_vs_refined_tlsph','MLS signed integral vs TLSPH target'),
                      ('geometry_vs_mls','Rebuilt cell vs MLS signed integral'),
                      ('geometry_vs_refined_tlsph','Rebuilt cell vs TLSPH target')]:
        ax[0].plot(loads,[100*r[key]['relative_l1'] for r in rows],'-o',label=label)
    ax[0].set(xlabel='tip load',ylabel='sum |cell volume difference| / total target (%)')
    ax[0].legend(fontsize=8)
    for key,label in [('total_mls','MLS signed integral'),('total_refined_tlsph','TLSPH target'),
                      ('total_geometry','Rebuilt cells')]:
        ax[1].plot(loads,[r[key] for r in rows],'-o',label=label)
    ax[1].set(xlabel='tip load',ylabel='total / signed volume');ax[1].legend()
    fig.suptitle('Diagnostic MLS extension: local inversion detected at loads 0.8, 1.6, 3.2',fontsize=11)
    fig.tight_layout();fig.savefig(out/'volume_comparison.png',dpi=170);plt.close(fig)


def extreme_gradient_audit(data,fine,h,power,seed):
    """Validate the largest-load minimum Jacobians, including any folding."""
    X=data['reference']; V=data['volumes']; r=data['radii0']
    positions=data['positions'][[-1]]; rows=[]
    for i in np.argsort(fine['minimum_J'][-1])[:3]:
        q,_=cell_markers(X,r,int(i),power,seed)
        ids=np.flatnonzero(np.linalg.norm(X-X[i],axis=1)<=r[i]+2*h+1e-12)
        G=np.concatenate([mls_gradient(q[s:s+256],X[ids],V[ids],h,positions[:,ids])
                          for s in range(0,len(q),256)],axis=1)
        J=np.linalg.det(G)[0]; j=int(np.argmin(J)); point=q[j:j+1]
        numerical=[]; fd_errors=[]
        for eps in [h*1e-4,h*1e-5]:
            F=np.stack([(mls_map(point+np.eye(3)[a]*eps,X,V,h,positions)-
                         mls_map(point-np.eye(3)[a]*eps,X,V,h,positions))/(2*eps)
                        for a in range(3)],axis=-1)
            numerical.append(float(np.linalg.det(F)[0,0]))
            fd_errors.append(float(np.max(abs(F-G[:,j:j+1]))))
        assert fd_errors[-1]<1e-5
        powers=np.sum((X-point)**2,axis=1)-r*r
        assert int(np.argmin(powers))==i and powers[i]<0
        s=np.linalg.norm((X-point)/h,axis=1)
        rows.append(dict(cell=int(i),reference_point=point[0].tolist(),
                         det_gradient=float(J[j]),fd_determinants=numerical,
                         fd_max_errors=fd_errors,supporting_nodes=int(np.sum(s<2)),
                         point_confirmed_in_original_cell=True))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('output/tlsph_load_sweep'))
    parser.add_argument('--out',type=Path,default=Path('output/tlsph_integrated_volume'))
    parser.add_argument('--coarse',type=int,default=11)
    parser.add_argument('--fine',type=int,default=13)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    data=np.load(args.source/'trajectory.npz'); radii=np.load(args.source/'radii.npz')
    ref=radii['refined_reference_volumes']; h=.15
    report=dict(scope=__doc__,checks=checks(data,h),
                source_sha256={name:hashlib.sha256((args.source/name).read_bytes()).hexdigest()
                               for name in ['trajectory.npz','radii.npz']},
                integration='Per-cell Sobol weighted mean of det(grad(phi)), multiplied by refined geometric rest volume. MLS weights retain saved mechanical volumes. No absolute value of determinant.',
                coarse_power=args.coarse,fine_power=args.fine,
                reference_volume_noise=errors(data['volumes'],ref))
    provenance_path=args.out/'sources.json'
    if args.resume and any(args.out.glob('*.npz')):
        if provenance_path.exists():
            previous=json.loads(provenance_path.read_text())
        else:
            previous=json.loads((args.out/'results.json').read_text())['source_sha256']
        assert previous==report['source_sha256'], 'Cached input hashes differ'
    provenance_path.write_text(json.dumps(report['source_sha256'],indent=2)+'\n')
    print('Checks:',json.dumps(report['checks']),flush=True)
    frames=np.arange(len(data['positions']))
    results={}
    for tag,power,seed,ff in [('coarse',args.coarse,6101,frames),('fine',args.fine,6101,frames),
                             ('independent',args.fine,9101,frames[-1:])]:
        path=args.out/f'{tag}.npz'
        if args.resume and path.exists():
            saved=np.load(path)
            assert int(saved['power'])==power and int(saved['seed'])==seed
            results[tag]=dict(saved)
        else:
            results[tag]=integrate(data,ref,h,power,seed,ff,args.out,tag)
    fine=results['fine']; coarse=results['coarse']; independent=results['independent']
    report['extreme_gradient_audit']=extreme_gradient_audit(data,fine,h,args.fine,6101)
    geo_path=args.out/'geometry.npz'
    if args.resume and geo_path.exists():
        geo=np.load(geo_path)['volumes']
    else:
        geo=[]
        for k,p in enumerate(data['positions']):
            geo.append(RadialVolumes(p,directions(17,901)).evaluate(radii['joint_refined'][k]))
            print(f'Geometry: state {k+1}/{len(frames)}',flush=True)
        geo=np.asarray(geo);np.savez_compressed(geo_path,volumes=geo)
    report['rows']=[]
    for k in frames:
        J=np.linalg.det(data['deformation'][k]); mls=fine['volumes'][k]; target=ref*J
        row=dict(load=float(data['loads'][k]),mls_vs_refined_tlsph=errors(mls,target),
                 mls_vs_saved_tlsph=errors(mls,data['volumes']*J),
                 geometry_vs_refined_tlsph=errors(geo[k],target),geometry_vs_mls=errors(geo[k],mls),
                 quadrature_coarse_vs_fine=errors(coarse['volumes'][k],mls),
                 mean_det_vs_det_mean=errors(mls,ref*np.linalg.det(fine['mean_G'][k])),
                 det_mean_gradient_vs_tlsph=errors(ref*np.linalg.det(fine['mean_G'][k]),target),
                 sampled_J_range=[float(fine['minimum_J'][k].min()),float(fine['maximum_J'][k].max())],
                 sampled_nonpositive_reference_volume_fraction=float(ref@fine['nonpositive_fraction'][k]/ref.sum()),
                 total_mls=float(mls.sum()),total_refined_tlsph=float(target.sum()),total_geometry=float(geo[k].sum()))
        report['rows'].append(row)
    report['independent_seed_largest_load']=errors(independent['volumes'][0],fine['volumes'][-1])
    report['rest_total_volume']=float(ref.sum())
    report['limitations']='MLS is not nodally interpolating and is not the TLSPH gradient estimator. No unique physical interior flow is given by the particles. Positive sampled Jacobians do not prove global injectivity. Comparison uses frozen states; no radius correction or force feedback.'
    (args.out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    plot(args.out,report['rows'])
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
