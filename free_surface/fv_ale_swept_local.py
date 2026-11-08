"""Local temporal geometry refinement on full production saved states.

Select targets from 3-versus-5-point quadrature and the endpoint volume defect,
never from the dense reference. A target is clipped together with every ball
that may overlap it over the entire linear path. Other patch cells are ghosts:
only target caps and faces incident to a target are used. Faces are shared once.
This is an experiment, not a topology-certified production implicit integrator.
"""
import argparse,json,time,math
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from free_surface.fv_ale_ground_geometry import GroundClip
from free_surface.fv_ale_profile import load
from free_surface.fv_ale_solid import acoustic_face
from free_surface.fv_ale_swept_check import PathAudit,endpoint,cpu

FACE_FIELDS=['flux','positive','negative','area','area_n']
CELL_FIELDS=['cap','cap_area','cap_n','cap_Q','floor']


def aligned(a,b,name):
    keys=np.union1d(a['keys'],b['keys']);shape=(len(keys),)+a[name].shape[1:]
    delta=np.zeros(shape);delta[np.searchsorted(keys,a['keys'])]+=a[name]
    delta[np.searchsorted(keys,b['keys'])]-=b[name]
    return keys,delta


def indicator(audit,fine,coarse):
    n=audit.n;g=audit.g;score=np.zeros(audit.ng)
    for name in ['flux','positive','negative']:
        keys,diff=aligned(fine,coarse,name);i=keys//n;j=keys%n;ext=g[i]!=g[j]
        score+=np.bincount(g[i[ext]],weights=np.abs(diff[ext]),minlength=audit.ng)
        score+=np.bincount(g[j[ext]],weights=np.abs(diff[ext]),minlength=audit.ng)
    score+=np.abs(audit.group(fine['cap']-coarse['cap']))
    # Include boundary-moment changes, not only cancellation in summed volume.
    motion=np.linalg.norm(cpu(audit.dp),axis=1)+np.abs(cpu(audit.dr))
    for name in ['cap_area','cap_n','cap_Q','floor']:
        diff=fine[name]-coarse[name]
        if diff.ndim>1:diff=np.linalg.norm(diff.reshape(n,-1),axis=1)
        score+=audit.group(np.abs(diff)*motion)
    keys,diff=aligned(fine,coarse,'area_n');i=keys//n;j=keys%n;ext=g[i]!=g[j]
    value=np.linalg.norm(diff,axis=1)*np.maximum(motion[i],motion[j])
    score+=np.bincount(g[i[ext]],weights=value[ext],minlength=audit.ng)
    score+=np.bincount(g[j[ext]],weights=value[ext],minlength=audit.ng)
    keys=fine['keys'];i=keys//n;j=keys%n
    rate=np.bincount(i,weights=fine['flux'],minlength=n)-np.bincount(j,weights=fine['flux'],minlength=n)+fine['cap']
    defect=np.abs(audit.group(rate-(audit.v1-audit.v0)))
    return score/audit.gv0,defect/audit.gv0


class LocalPath:
    def __init__(self,audit,targets,batch=1,target_only=False):
        tick=time.perf_counter();self.a=audit;self.targets=np.asarray(targets,dtype=np.int64)
        self.mask=np.zeros(audit.n,dtype=bool);self.mask[self.targets]=True
        p0=cpu(audit.start.p);p1=cpu(audit.end.p);r0=cpu(audit.start.r);r1=cpu(audit.end.r)
        center=(p0+p1)*.5;radius=np.maximum(r0,r1)+np.linalg.norm(p1-p0,axis=1)*.5
        neighbours=[] if len(self.targets)==audit.n else cKDTree(center).query_ball_point(center[self.targets],(radius[self.targets]+radius.max())*(1+1e-12))
        covered=[self.targets];self.candidate_pairs=0
        for i,ns in zip(self.targets,neighbours):
            j=np.asarray(ns,dtype=np.int64)
            j=j[np.linalg.norm(center[j]-center[i],axis=1)<=radius[j]+radius[i]+1e-12*radius[i]]
            d0=p0[j]-p0[i];dd=(p1[j]-p0[j])-(p1[i]-p0[i]);sr=r0[j]+r0[i];ds=(r1[j]-r0[j])+(r1[i]-r0[i])
            A=(dd*dd).sum(1)-ds*ds;B=2*((d0*dd).sum(1)-sr*ds);C=(d0*d0).sum(1)-sr*sr
            tau=np.clip(-B/(2*np.where(A>0,A,1.)),0,1)
            minimum=np.minimum(C,A+B+C);minimum=np.where(A>0,np.minimum(minimum,A*tau*tau+B*tau+C),minimum)
            j=j[minimum<=1e-12*np.maximum(sr*sr,1e-300)]
            covered.append(j);self.candidate_pairs+=len(j)
        self.ids=np.unique(np.concatenate(covered));self.device_ids=torch.as_tensor(self.ids,device=audit.start.p.device)
        self.local_targets=np.flatnonzero(self.mask[self.ids])
        self.cutter=GroundClip(audit.sol.ground,str(audit.start.p.device))
        if target_only and len(self.targets)<audit.n:
            from free_surface.fv_ale_target_cut import TargetCut
            local_rows=[np.searchsorted(self.ids,row[row!=owner]) for owner,row in zip(self.targets,covered[1:])]
            self.cutter=TargetCut(len(self.ids),self.local_targets,local_rows,audit.sol.ground,str(audit.start.p.device))
        self.batch=batch;self.batched_cut_calls=0
        # Disjoint translations parallel to the floor let the existing GPU
        # geometry pipeline process several independent times in one call.
        lo=np.minimum(p0[:,0]-r0,p1[:,0]-r1).min()
        hi=np.maximum(p0[:,0]+r0,p1[:,0]+r1).max()
        self.stride=float(hi-lo+.1)
        self.cache={};self.cut_seconds=0.;self.evaluation_seconds=0.;self.cuts=0
        self.coverage_seconds=time.perf_counter()-tick

    def restrict(self,full):
        keys=full['keys'];owned=self.mask[keys//self.a.n]|self.mask[keys%self.a.n]
        record={'keys':keys[owned]}
        for name in FACE_FIELDS:record[name]=full[name][owned]
        for name in CELL_FIELDS:
            record[name]=np.zeros_like(full[name]);record[name][self.targets]=full[name][self.targets]
        return record

    def sample_batch(self,taus):
        missing=[t for t in taus if t not in self.cache and t not in self.a.cache]
        for first in range(0,len(missing),self.batch):
            times=missing[first:first+self.batch]
            if self.batch==1:
                for tau in times:self.sample(tau)
                continue
            a=self.a;ids=self.device_ids;k=len(ids);nb=len(times)
            tick=time.perf_counter()
            tau=torch.tensor(times,dtype=a.start.p.dtype,device=ids.device)
            offset=torch.zeros((nb,1,3),dtype=tau.dtype,device=ids.device)
            offset[:,0,0]=torch.arange(nb,device=ids.device)*self.stride
            p=(a.start.p[ids][None]+tau[:,None,None]*a.dp[ids][None]+offset).reshape(-1,3)
            r=(a.start.r[ids][None]+tau[:,None]*a.dr[ids][None]).reshape(-1)
            torch.cuda.synchronize();begin=time.perf_counter()
            mesh=self.cutter.geometry(p,r);a.sol.device_ops().moments(mesh)
            torch.cuda.synchronize();self.cut_seconds+=time.perf_counter()-begin
            self.cuts+=nb;self.batched_cut_calls+=1
            i,j=mesh.i,mesh.j;block=i//k
            if not bool((block==j//k).all()):raise RuntimeError('batched geometry copies overlap')
            gi=ids[i%k];gj=ids[j%k];own=torch.as_tensor(self.mask,device=ids.device)
            owned=own[gi]|own[gj]
            flux,cap=mesh.mesh_flux(a.dp[ids].repeat(nb,1),a.dr[ids].repeat(nb))
            n=mesh.normal;ti=(a.sigma[gi]@n[:,:,None])[:,:,0];tj=(a.sigma[gj]@n[:,:,None])[:,:,0]
            us,_=acoustic_face(a.v[gi],a.v[gj],ti,tj,a.Z[gi],a.Z[gj])
            relative=a.dt*mesh.area*(us*n).sum(1)-flux
            fields={'flux':cpu(flux),'positive':cpu(relative.clamp(min=0)),'negative':cpu(relative.clamp(max=0)),
                'area':cpu(mesh.area),'area_n':cpu(mesh.area[:,None]*n)}
            keys=cpu(gi*a.n+gj);blocks=cpu(block);keep=cpu(owned)
            cells={name:cpu(value).reshape((nb,k)+tuple(value.shape[1:])) for name,value in
                [('cap',cap),('cap_area',mesh.cap_area),('cap_n',mesh.cap_normal_integral),('cap_Q',mesh.cap_Q),('floor',mesh.ground_area)]}
            for b,t in enumerate(times):
                selected=keep&(blocks==b);record={'keys':keys[selected]}
                for name in FACE_FIELDS:record[name]=fields[name][selected]
                for name in CELL_FIELDS:
                    array=cells[name][b];record[name]=np.zeros((a.n,)+array.shape[1:])
                    record[name][self.targets]=array[self.local_targets]
                self.cache[t]=record
            self.evaluation_seconds+=time.perf_counter()-tick

    def sample(self,tau):
        if tau in self.cache:return self.cache[tau]
        if tau in self.a.cache:
            record=self.restrict(self.a.cache[tau]);self.cache[tau]=record;return record
        tick=time.perf_counter();a=self.a;ids=self.device_ids
        p=(a.start.p+tau*a.dp)[ids];r=(a.start.r+tau*a.dr)[ids]
        torch.cuda.synchronize();start=time.perf_counter()
        mesh=self.cutter.geometry(p,r);a.sol.device_ops().moments(mesh)
        torch.cuda.synchronize();self.cut_seconds+=time.perf_counter()-start;self.cuts+=1
        i,j=mesh.i,mesh.j;gi=ids[i];gj=ids[j]
        own=torch.as_tensor(self.mask,device=p.device);keep=own[gi]|own[gj]
        flux,cap=mesh.mesh_flux(a.dp[ids],a.dr[ids]);normal=mesh.normal
        ti=(a.sigma[gi]@normal[:,:,None])[:,:,0];tj=(a.sigma[gj]@normal[:,:,None])[:,:,0]
        us,_=acoustic_face(a.v[gi],a.v[gj],ti,tj,a.Z[gi],a.Z[gj])
        relative=a.dt*mesh.area*(us*normal).sum(1)-flux
        record={'keys':cpu((gi*a.n+gj)[keep]),'flux':cpu(flux[keep]),
            'positive':cpu(relative[keep].clamp(min=0)),'negative':cpu(relative[keep].clamp(max=0)),
            'area':cpu(mesh.area[keep]),'area_n':cpu((mesh.area[:,None]*normal)[keep])}
        for name,value in [('cap',cap),('cap_area',mesh.cap_area),('cap_n',mesh.cap_normal_integral),
                           ('cap_Q',mesh.cap_Q),('floor',mesh.ground_area)]:
            array=cpu(value);record[name]=np.zeros((a.n,)+array.shape[1:])
            record[name][self.targets]=array[self.local_targets]
        self.evaluation_seconds+=time.perf_counter()-tick;self.cache[tau]=record
        return record

    def integrate(self,panels):
        weights=np.ones(panels+1);weights[1:-1:2]=4;weights[2:-1:2]=2;weights/=3*panels
        self.sample_batch([k/panels for k in range(panels+1)])
        samples=[self.sample(k/panels) for k in range(panels+1)]
        keys=np.unique(np.concatenate([s['keys'] for s in samples]));result={'keys':keys}
        for name in FACE_FIELDS:result[name]=np.zeros((len(keys),)+samples[0][name].shape[1:])
        for name in CELL_FIELDS:result[name]=np.zeros_like(samples[0][name])
        for weight,sample in zip(weights,samples):
            ix=np.searchsorted(keys,sample['keys'])
            for name in FACE_FIELDS:result[name][ix]+=weight*sample[name]
            for name in CELL_FIELDS:result[name]+=weight*sample[name]
        return result

    def combine(self,base,local):
        keys=np.union1d(base['keys'],local['keys']);result={'keys':keys}
        owned=self.mask[keys//self.a.n]|self.mask[keys%self.a.n]
        for name in FACE_FIELDS:
            value=np.zeros((len(keys),)+base[name].shape[1:]);value[np.searchsorted(keys,base['keys'])]=base[name]
            value[owned]=0;value[np.searchsorted(keys,local['keys'])]=local[name];result[name]=value
        for name in CELL_FIELDS:
            result[name]=base[name].copy();result[name][self.targets]=local[name][self.targets]
        return result


def errors(audit,result,reference):
    n=audit.n;keys=result['keys'];i=keys//n;j=keys%n
    rate=np.bincount(i,weights=result['flux'],minlength=n)-np.bincount(j,weights=result['flux'],minlength=n)+result['cap']
    defect=np.abs(audit.group(rate-(audit.v1-audit.v0)))/audit.gv0
    record={'max_group_volume_defect':float(defect.max()),'volume_defect_rms':float(np.sqrt(np.mean(defect**2)))}
    for name in ['flux','positive','negative']:
        union,delta=aligned(result,reference,name);ii=union//n;jj=union%n;ext=audit.g[ii]!=audit.g[jj]
        scale=np.minimum(audit.gv0[audit.g[ii]],audit.gv0[audit.g[jj]])
        record[name+'_max_face_error_scaled']=float(np.max(np.abs(delta[ext])/scale[ext],initial=0))
        record[name+'_l1_error_m3']=float(np.abs(delta).sum())
    for name in ['cap_area','cap_n','cap_Q','floor']:
        record[name+'_relative_l2']=float(np.linalg.norm(result[name]-reference[name])/max(np.linalg.norm(reference[name]),1e-300))
    missing=np.setdiff1d(reference['keys'],result['keys'])
    record['reference_faces_missing']=len(missing)
    record['missing_reference_abs_flux_m3']=float(np.abs(reference['flux'][np.isin(reference['keys'],missing)]).sum())
    return record


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tags',nargs='+',default=['dt8us','cfl20_long'])
    ap.add_argument('--tolerance',type=float,default=1e-6);ap.add_argument('--max-panels',type=int,default=32)
    ap.add_argument('--batch',type=int,default=1)
    ap.add_argument('--target-only',action='store_true')
    ap.add_argument('--output',default='swept_local_trial.json');args=ap.parse_args()
    out=Path('output/ale');records=[]
    for tag in args.tags:
        sol,reset,data=load(out/f'fv_state_{tag}.pt');start=reset();end,refcost=endpoint(sol,start,800e-6)
        audit=PathAudit(sol,start,end,800e-6,data['reference_volume']);torch.cuda.synchronize();begin=time.perf_counter()
        coarse=audit.integrate(2);base=audit.integrate(4)
        estimate,defect=indicator(audit,base,coarse)
        chosen=(estimate>args.tolerance*.25)|(defect>args.tolerance*.25)
        targets=np.flatnonzero(chosen[audit.g]);local=LocalPath(audit,targets,args.batch,args.target_only)
        setup=time.perf_counter()-begin
        # Selection is complete before loading the dense reference.
        reference=dict(np.load(out/f'swept_{tag}_800us.npz'))
        record={'snapshot':tag,'tolerance':args.tolerance,'batch':args.batch,'target_only':args.target_only,'selected_groups':int(chosen.sum()),'group_count':audit.ng,
            'target_cells':len(targets),'patch_cells_with_ghosts':len(local.ids),'total_cells':audit.n,
            'coverage_seconds':local.coverage_seconds,'coarse_selection_setup_seconds':setup,
            'coarse_full_intermediate_cut_seconds':audit.cut_seconds,'endpoint_generation':refcost,
            'coarse_errors':errors(audit,base,reference),'levels':[]}
        print(json.dumps({'phase':'selection',**record}),flush=True)
        last=base
        for power in range(3,int(math.log2(args.max_panels))+1):
            panels=2**power;fine=local.integrate(panels);combined=local.combine(base,fine)
            row={'nodes':panels+1,'local_cuts':local.cuts,'local_cut_seconds':local.cut_seconds,
                'batched_cut_calls':local.batched_cut_calls,
                'local_evaluation_seconds':local.evaluation_seconds,'elapsed_seconds':time.perf_counter()-begin,
                **errors(audit,combined,reference)}
            _,new_defect=indicator(audit,combined,last)
            row['unselected_groups_above_volume_tolerance']=int(((new_defect>args.tolerance)&~chosen).sum())
            record['levels'].append(row);print(json.dumps({'snapshot':tag,**row}),flush=True);last=combined
        # Independent full cuts at new sample times verify exactness of patch extraction.
        checks=[]
        for tau in [.125,.375,.875]:
            patch=local.sample(tau);whole=local.restrict(audit.sample(tau))
            check={'tau':tau,'keys_equal':bool(np.array_equal(patch['keys'],whole['keys']))}
            for name in FACE_FIELDS:
                union,delta=aligned(patch,whole,name)
                check[name+'_max_abs']=float(np.max(np.abs(delta),initial=0))
            for name in CELL_FIELDS:check[name+'_max_abs']=float(np.max(np.abs(patch[name]-whole[name]),initial=0))
            assert check['flux_max_abs']<1e-12 and check['cap_max_abs']<1e-12,check
            assert all(v<1e-10 for k,v in check.items() if k.endswith('_max_abs')),check
            checks.append(check)
        record['patch_vs_full_checks']=checks;record['topology_certified']=False
        records.append(record);(out/args.output).write_text(json.dumps(records,indent=2)+'\n')

if __name__=='__main__':main()
