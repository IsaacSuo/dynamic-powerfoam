"""Whole-scene space-time geometry audit, not a new time integrator.

Endpoints come from accepted production SSPRK2 steps. For each proposed large
step, p and r interpolate linearly between those endpoints. Nested temporal
quadrature integrates the production exact instantaneous mesh-volume flux.
Only geometry is subdivided: no material updates occur at quadrature nodes.
No production state or default is changed. Sampling is not topology certification.
"""
import argparse,json,time,math
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from free_surface.fv_ale_profile import load
from free_surface.fv_ale_solid import acoustic_face


def cpu(t):return t.detach().cpu().numpy().copy()


def swept_pair_superset(p0,r0,p1,r1):
    """Conservative ball-overlap candidates, NOT a certificate of visible faces."""
    center=(p0+p1)*.5;radius=np.maximum(r0,r1)+np.linalg.norm(p1-p0,axis=1)*.5
    neighbours=cKDTree(center).query_ball_point(center,radius+radius.max())
    ii=[];jj=[]
    for i,ns in enumerate(neighbours):
        ids=np.asarray([j for j in ns if j>i],dtype=np.int64)
        ids=ids[np.linalg.norm(center[ids]-center[i],axis=1)<=radius[ids]+radius[i]]
        ii.extend([i]*len(ids));jj.extend(ids.tolist())
    i=np.asarray(ii,dtype=np.int64);j=np.asarray(jj,dtype=np.int64)
    d0=p0[j]-p0[i];dd=(p1[j]-p0[j])-(p1[i]-p0[i])
    sr=r0[i]+r0[j];ds=(r1[i]-r0[i])+(r1[j]-r0[j])
    a=(dd*dd).sum(1)-ds*ds;b=2*((d0*dd).sum(1)-sr*ds);c=(d0*d0).sum(1)-sr*sr
    tau=np.clip(-b/(2*np.where(a>0,a,1.)),0,1)
    minimum=np.minimum(c,a+b+c);minimum=np.where(a>0,np.minimum(minimum,a*tau*tau+b*tau+c),minimum)
    keep=minimum<=1e-13*np.maximum(sr*sr,1e-300)
    return i[keep]*len(p0)+j[keep]


class PathAudit:
    def __init__(self,sol,start,end,dt,reference_volume):
        self.sol=sol;self.start=start;self.end=end;self.dt=dt;self.rv=reference_volume
        self.dp=end.p-start.p;self.dr=end.r-start.r;self.n=len(start.r)
        self.labels,self.g=np.unique(cpu(sol.groups),return_inverse=True);self.ng=len(self.labels)
        self.v0=cpu(start.mesh.volume);self.v1=cpu(end.mesh.volume)
        self.gv0=self.group(self.v0);self.gv1=self.group(self.v1)
        _,self.v,self.rho,_,_,self.sigma,sound=sol.primitive(end);self.Z=self.rho*sound
        self.cache={};self.cut_seconds=0.;self.cuts=0;self.analysis_seconds=0.

    def group(self,a):return np.bincount(self.g,weights=a,minlength=self.ng)

    def sample(self,tau):
        if tau in self.cache:return self.cache[tau]
        torch.cuda.synchronize();start=time.perf_counter()
        if tau==0:mesh=self.start.mesh
        elif tau==1:mesh=self.end.mesh
        else:
            mesh=self.sol.cut(self.start.p+tau*self.dp,self.start.r+tau*self.dr)
            torch.cuda.synchronize();self.cut_seconds+=time.perf_counter()-start;self.cuts+=1
        tick=time.perf_counter();i,j,n=mesh.i,mesh.j,mesh.normal
        if not bool((i<j).all()):raise RuntimeError('face orientation is not canonical')
        swept,cap=mesh.mesh_flux(self.dp,self.dr)
        ti=(self.sigma[i]@n[:,:,None])[:,:,0];tj=(self.sigma[j]@n[:,:,None])[:,:,0]
        us,_=acoustic_face(self.v[i],self.v[j],ti,tj,self.Z[i],self.Z[j])
        relative=self.dt*mesh.area*(us*n).sum(1)-swept
        rate=mesh.incidence(swept)+cap
        record={'keys':cpu(i*self.n+j),'flux':cpu(swept),'positive':cpu(relative.clamp(min=0)),
            'negative':cpu(relative.clamp(max=0)),'area':cpu(mesh.area),'area_n':cpu(mesh.area[:,None]*n),
            'cap':cpu(cap),'rate':cpu(rate),'cap_area':cpu(mesh.cap_area),
            'cap_n':cpu(mesh.cap_normal_integral),'cap_Q':cpu(mesh.cap_Q),'floor':cpu(mesh.ground_area)}
        self.analysis_seconds+=time.perf_counter()-tick;self.cache[tau]=record
        return record

    def integrate(self,panels):
        if panels==1:weights=np.array([.5,.5])
        else:
            weights=np.ones(panels+1);weights[1:-1:2]=4;weights[2:-1:2]=2;weights/=3*panels
        samples=[self.sample(k/panels) for k in range(panels+1)]
        keys=np.unique(np.concatenate([x['keys'] for x in samples]))
        result={'keys':keys}
        for name in ['flux','positive','negative','area']:
            result[name]=np.zeros(len(keys))
        result['area_n']=np.zeros((len(keys),3))
        for name in ['cap','rate','cap_area','cap_n','cap_Q','floor']:
            result[name]=np.zeros_like(samples[0][name])
        for weight,sample in zip(weights,samples):
            ids=np.searchsorted(keys,sample['keys'])
            for name in ['flux','positive','negative','area','area_n']:
                result[name][ids]+=weight*sample[name]
            for name in ['cap','rate','cap_area','cap_n','cap_Q','floor']:
                result[name]+=weight*sample[name]
        return result

    def report(self,result,previous,panels):
        keys=result['keys'];i=keys//self.n;j=keys%self.n
        incidence=np.bincount(i,weights=result['flux'],minlength=self.n)-np.bincount(j,weights=result['flux'],minlength=self.n)
        integrated=incidence+result['cap'];error=integrated-(self.v1-self.v0)
        grouped=self.group(error);relative=grouped/self.gv0
        scale=np.maximum.reduce([self.v0,self.v1,np.full(self.n,self.rv*1e-6)])
        row={'panels':panels,'temporal_nodes':panels+1,'new_cuts_excluding_cached_endpoints':self.cuts,
            'cut_seconds_cumulative':self.cut_seconds,'analysis_seconds_cumulative':self.analysis_seconds,
            'group_volume_error_max_relative_initial':float(np.max(np.abs(relative))),
            'group_volume_error_rms_relative_initial':float(np.sqrt(np.mean(relative**2))),
            'group_volume_error_l1_relative_body':float(np.abs(grouped).sum()/self.gv0.sum()),
            'cell_volume_error_max_scaled':float(np.max(np.abs(error)/scale)),
            'incidence_accumulation_roundoff_m3':float(np.max(np.abs(integrated-result['rate']))),
            'worst_group_label':int(self.labels[np.argmax(np.abs(relative))]),'observed_face_pairs':len(keys)}
        endpoints=np.union1d(self.sample(0)['keys'],self.sample(1)['keys'])
        mid_only=~np.isin(keys,endpoints)
        row['faces_absent_at_both_endpoints']=int(mid_only.sum())
        row['mid_only_abs_swept_volume_m3']=float(np.abs(result['flux'][mid_only]).sum())
        row['mid_only_max_time_average_area_m2']=float(result['area'][mid_only].max(initial=0))
        external=self.g[i]!=self.g[j]
        local_scale=np.minimum(self.gv0[self.g[i]],self.gv0[self.g[j]])
        reversed_=(result['positive']>local_scale*1e-12)&(result['negative']<-local_scale*1e-12)&external
        rho=cpu(self.rho);net=result['positive']+result['negative']
        split=result['positive']*rho[i]+result['negative']*rho[j]
        net_only=net*np.where(net>=0,rho[i],rho[j])
        row['external_faces_with_relative_flux_reversal']=int(reversed_.sum())
        row['net_donor_vs_split_abs_mass_difference_kg']=float(np.abs(split[external]-net_only[external]).sum())
        if previous is not None:
            union=np.union1d(keys,previous['keys']);iu=union//self.n;ju=union%self.n
            local=np.minimum(self.gv0[self.g[iu]],self.gv0[self.g[ju]])
            ext=self.g[iu]!=self.g[ju]
            for name in ['flux','positive','negative']:
                delta=np.zeros(len(union));delta[np.searchsorted(union,keys)]+=result[name]
                delta[np.searchsorted(union,previous['keys'])]-=previous[name]
                row[name+'_refinement_max_scaled']=float(np.max(np.abs(delta[ext])/local[ext],initial=0))
                row[name+'_refinement_l1_m3']=float(np.abs(delta).sum())
            for name in ['area_n','cap_area','cap_n','cap_Q','floor']:
                if name=='area_n':
                    delta=np.zeros((len(union),3));delta[np.searchsorted(union,keys)]+=result[name]
                    delta[np.searchsorted(union,previous['keys'])]-=previous[name]
                else:delta=result[name]-previous[name]
                row[name+'_refinement_relative_l2']=float(np.linalg.norm(delta)/max(np.linalg.norm(result[name]),1e-300))
        return row


def endpoint(sol,start,dt):
    s=start;left=dt;steps=0;rejects=0;tick=time.perf_counter();cuts=sol.geometry_calls
    while left>1e-14:
        first=sol.rhs(s);h=min(left,2*first[3]['dt_scale'])
        for trial in range(9):
            try:s,_=sol.advance(s,h,first,2.);break
            except RuntimeError:
                rejects+=1;h*=.5
                if trial==8:raise
        left-=h;steps+=1
    torch.cuda.synchronize()
    return s,{'steps':steps,'rejects':rejects,'seconds':time.perf_counter()-tick,'recuts':sol.geometry_calls-cuts}


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tags',nargs='+',default=['dt8us','cfl20_long'])
    ap.add_argument('--dt',type=float,nargs='+',default=[80e-6,200e-6,800e-6]);ap.add_argument('--max-panels',type=int,default=64)
    ap.add_argument('--output',default='swept_geometry_trial.json')
    args=ap.parse_args();out=Path('output/ale');records=[]
    if args.max_panels<2 or args.max_panels&(args.max_panels-1):raise ValueError('max-panels must be a power of two >=2')
    for tag in args.tags:
        sol,reset,data=load(out/f'fv_state_{tag}.pt')
        sol.ground_cutter.recut.pipe.direct_sphere=False
        for dt in args.dt:
            record={'snapshot':tag,'dt':dt,'path':'linear p,r between production explicit endpoints','levels':[]}
            try:
                start=reset();end,ref=endpoint(sol,start,dt);record['endpoint_generation']=ref
                audit=PathAudit(sol,start,end,dt,data['reference_volume'])
                previous=None
                for power in range(int(math.log2(args.max_panels))+1):
                    panels=2**power;result=audit.integrate(panels)
                    row=audit.report(result,previous,panels);record['levels'].append(row);previous=result
                    print(json.dumps({'snapshot':tag,'dt':dt,**row}),flush=True)
                possible=swept_pair_superset(cpu(start.p),cpu(start.r),cpu(end.p),cpu(end.r))
                missed=np.setdiff1d(result['keys'],possible)
                assert not len(missed),'observed face escaped conservative pair coverage'
                record['swept_ball_candidate_pairs']=len(possible)
                record['candidate_pairs_never_observed_as_faces']=len(np.setdiff1d(possible,result['keys']))
                record['topology_certified']=False
                record['status']='measured'
                np.savez_compressed(out/f'swept_{tag}_{round(dt*1e6)}us.npz',**result)
            except Exception as exc:record.update(status='failed',error=repr(exc));print(json.dumps(record),flush=True)
            records.append(record)
            (out/args.output).write_text(json.dumps(records,indent=2)+'\n')

if __name__=='__main__':main()
