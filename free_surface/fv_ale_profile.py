"""Frozen-state FV timing; does not change the production stepping algorithm.

Replay one real step from each snapshot. Warm caches, exclude snapshot loading
and the reset cut, alternate production/reused-probe diagnostic controls, then
measure nested synchronized stages and a separate CPU/CUDA profiler trace.
Never commit a replayed state. See JSON for timer overhead and numerical checks.
"""
import argparse,collections,contextlib,json,statistics,time
from pathlib import Path
import torch,warp as wp
import free_surface.fv_ale_solid as core


def summary(values):
    s=sorted(values)
    return {'median':statistics.median(s),'min':s[0],'max':s[-1],'samples':s}


class Stages:
    def __init__(self,sol,trace=False):
        self.sol=sol;self.trace=trace;self.saved=[];self.events=[];self.stack=[]

    def wrap(self,obj,key,name):
        original=getattr(obj,key);self.saved.append((obj,key,original))
        def call(*args,**kwargs):
            if self.trace:
                with torch.profiler.record_function('FV::'+name):
                    return original(*args,**kwargs)
            torch.cuda.synchronize();start=time.perf_counter();frame=[name,0.]
            self.stack.append(frame)
            try:return original(*args,**kwargs)
            finally:
                torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
                self.stack.pop();self.events.append({'name':name,'inclusive_ms':elapsed,'exclusive_ms':elapsed-frame[1]})
                if self.stack:self.stack[-1][1]+=elapsed
        setattr(obj,key,call)

    def __enter__(self):
        for key in ['rhs','cut','plastic_return','advance']:
            self.wrap(self.sol,key,key)
        self.wrap(self.sol.recut,'geometry','geometry')
        self.wrap(core,'surface_quadrature','contact_quadrature')
        self.wrap(core,'solve_boundary','contact_boundary_solve')
        return self

    def __exit__(self,*args):
        for obj,key,original in reversed(self.saved):setattr(obj,key,original)


def load(path):
    d=torch.load(path,map_location='cpu',weights_only=False);m=d['material']
    sol=core.SolidALE(d['p'],d['r'],m['E'],m['nu'],m['ys'],
        groups=d.get('groups'),surface_motion=d['surface_motion'],contact_model=d['contact_model'],
        contact_order=d['contact_order'],contact_scale=d['contact_scale'],
        contact_damping=d['contact_damping'],friction=d['friction'],interior_radii=d['interior_radii'])
    sol.contact_length=d['reference_volume']**(1/3);sol.contact_k=sol.contact_scale*sol.bulk/sol.contact_length
    p=d['p'].to(sol.initial.p.device);r=d['r'].to(p.device);q=d['q'].to(p.device)
    def reset():return core.State(p,r,q,sol.cut(p,r))
    return sol,reset,d


def one_step(sol,s,reference_volume,reuse_probe=False):
    first=None
    # Production selection on this fixed state. Refuse to turn a merge into a
    # timing control: compare the exact same already-admissible state instead.
    if sol.has_groups:
        gv=sol.group_sum(s.mesh.volume)[sol.groups]
        if bool((gv<.001*reference_volume).any()):raise RuntimeError('snapshot would trigger a volume merge')
        first=sol.rhs(s)
        if .2*first[3]['wave_dt_scale']<1e-6:raise RuntimeError('snapshot would trigger a wave-step merge')
        if not reuse_probe:first=None
    else:
        if bool(((s.mesh.volume<.001*reference_volume)&~s.mesh.surface).any()):raise RuntimeError('snapshot would trigger buried deletion')
    if first is None:first=sol.rhs(s)
    dt=min(1/1200,.2*first[3]['dt_scale'])
    end,stats=sol.advance(s,dt,first,.2)
    # The production driver also reads these reductions at each accepted step.
    stats['mass_error']=float((end.q[:,0].sum()-s.q[:,0].sum())/s.q[:,0].sum())
    stats['energy_error']=float((end.q[:,4]-s.q[:,4]).sum())-stats['source_work']
    return end,stats


def numerical_difference(a,b):
    return {key:float((getattr(a,key)-getattr(b,key)).norm()/getattr(a,key).norm().clamp(min=1e-100)) for key in ['p','r','q']}


def physical_difference(sol,a,b):
    pa=sol.primitive(a);pb=sol.primitive(b)
    return {'max_position_difference_m':float((a.p-b.p).abs().max()),
        'max_radius_difference_m':float((a.r-b.r).abs().max()),
        'max_velocity_difference_m_s':float((pa[1]-pb[1]).abs().max()),
        'max_density_relative_difference':float(((pa[2]-pb[2])/pa[2]).abs().max()),
        'max_pressure_difference_Pa':float((pa[3]-pb[3]).abs().max()),
        'max_deviatoric_stress_difference_Pa':float((pa[4]-pb[4]).abs().max())}


@contextlib.contextmanager
def contact_formula_control():
    """Diagnostic only: expand B=[I,n] algebra, retaining quadrature/iterations.

    B^T diag(d) B = [[diag(d), d*n], [(d*n)^T, sum(d*n*n)]].
    B^T t = [t, dot(n,t)]. No solver source or tolerance is changed.
    """
    original=torch.einsum
    def einsum(equation,*values,**kwargs):
        if equation=='pai,pa,paj->pij':
            B,d,_=values;n=B[:,:,3];dn=d*n
            out=torch.zeros((len(B),4,4),dtype=B.dtype,device=B.device)
            out[:,0,0]=d[:,0];out[:,1,1]=d[:,1];out[:,2,2]=d[:,2]
            out[:,:3,3]=dn;out[:,3,:3]=dn;out[:,3,3]=(dn*n).sum(1)
            return out
        if equation=='pai,pa->pi':
            B,t=values
            return torch.cat((t,(B[:,:,3]*t).sum(1,keepdim=True)),1)
        return original(equation,*values,**kwargs)
    torch.einsum=einsum
    try:yield
    finally:torch.einsum=original


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('snapshots',type=Path,nargs='+')
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--repeat',type=int,default=6)
    ap.add_argument('--trace',action='store_true')
    ap.add_argument('--formula-control',action='store_true');a=ap.parse_args();wp.init()
    report={'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),
        'method':'Frozen-state warm replay; reset cut excluded. Nested wall timing synchronizes stage boundaries; overhead measured against ordinary replay. Trace is separate from wall timing.',
        'production_algorithm_modified':False,'snapshots':[]}
    for path in a.snapshots:
        sol,reset,d=load(path);rv=d['reference_volume']
        for _ in range(2):one_step(sol,reset(),rv)
        ordinary=[];reuse=[];differences=[]
        for k in range(a.repeat):
            outputs={}
            modes=[False,True] if k%2==0 else [True,False]
            for mode in modes:
                s=reset();torch.cuda.synchronize();t=time.perf_counter()
                out,stats=one_step(sol,s,rv,mode)
                torch.cuda.synchronize();elapsed=(time.perf_counter()-t)*1000
                (reuse if mode else ordinary).append(elapsed);outputs[mode]=out
            differences.append(numerical_difference(outputs[False],outputs[True]))
        timed=[];per_stage=collections.defaultdict(list);counts={}
        for _ in range(a.repeat):
            s=reset();torch.cuda.synchronize();t=time.perf_counter()
            with Stages(sol) as timer:out,stats=one_step(sol,s,rv)
            torch.cuda.synchronize();timed.append((time.perf_counter()-t)*1000)
            grouped=collections.defaultdict(lambda:[0.,0.,0])
            for e in timer.events:
                row=grouped[e['name']];row[0]+=e['inclusive_ms'];row[1]+=e['exclusive_ms'];row[2]+=1
            for name,vals in grouped.items():
                per_stage[name+'.inclusive_ms'].append(vals[0]);per_stage[name+'.exclusive_ms'].append(vals[1]);counts[name]=vals[2]
        s=reset();rr=sol.rhs(s);quad=s.mesh.contact_quad
        record={'snapshot':str(path),'time':d['time'],'cells':len(s.r),'grouped':sol.has_groups,
            'wall_faces':len(s.mesh.i),'contact_cells':rr[3]['contact_cells'],
            'quadrature_points':len(quad[0]),'boundary_iterations':rr[3]['boundary_local_iterations'],
            'dt':.2*rr[3]['dt_scale'],'ordinary_step_ms':summary(ordinary),
            'reuse_probe_step_ms':summary(reuse),'reuse_probe_relative_state_errors':differences,
            'synchronized_step_ms':summary(timed),'stage_sums_ms':{k:summary(v) for k,v in per_stage.items()},
            'calls_per_step':counts}
        if a.formula_control:
            controls=collections.defaultdict(list);control_errors=[];physical_errors=[]
            for k in range(a.repeat):
                results={}
                modes=[(False,False),(True,False),(False,True),(True,True)]
                modes=modes[k%4:]+modes[:k%4]
                for algebra,reuse_one in modes:
                    s=reset();torch.cuda.synchronize();tick=time.perf_counter()
                    with contact_formula_control() if algebra else contextlib.nullcontext():
                        result,ss=one_step(sol,s,rv,reuse_one)
                    torch.cuda.synchronize();elapsed=(time.perf_counter()-tick)*1000
                    name=('formula' if algebra else 'original')+('_reuse' if reuse_one else '')
                    controls[name].append(elapsed);results[name]=result
                control_errors.append({name:numerical_difference(results['original'],result) for name,result in results.items() if name!='original'})
                physical_errors.append({name:physical_difference(sol,results['original'],result) for name,result in results.items() if name!='original'})
            s=reset()
            with contact_formula_control(),Stages(sol) as timer:
                one_step(sol,s,rv,True)
            sums=collections.defaultdict(float)
            for e in timer.events:sums[e['name']+'.exclusive_ms']+=e['exclusive_ms']
            record['formula_control']={'timings_ms':{k:summary(v) for k,v in controls.items()},
                'relative_state_errors':control_errors,'physical_errors':physical_errors,'one_reused_step_stage_exclusive_ms':dict(sums),
                'scope':'Temporary diagnostic einsum substitution; same points, formula, iterations, tolerances, physical dt. Production implementation untouched.'}
        if a.trace:
            s=reset()
            activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]
            with torch.profiler.profile(activities=activities,record_shapes=True,profile_memory=False) as prof:
                with Stages(sol,trace=True):one_step(sol,s,rv)
                torch.cuda.synchronize()
            trace=a.out.with_name(a.out.stem+'_'+path.stem+'_trace.json');prof.export_chrome_trace(str(trace))
            rows=[]
            for e in prof.key_averages():
                rows.append({'name':e.key,'calls':e.count,'self_cpu_ms':e.self_cpu_time_total/1000,
                    'self_device_ms':getattr(e,'self_device_time_total',0)/1000,
                    'total_device_ms':getattr(e,'device_time_total',0)/1000})
            record['trace']=str(trace)
            record['top_self_device']=sorted(rows,key=lambda x:x['self_device_ms'],reverse=True)[:30]
            record['top_self_cpu']=sorted(rows,key=lambda x:x['self_cpu_ms'],reverse=True)[:30]
            record['fv_trace_scopes']=[r for r in rows if r['name'].startswith('FV::')]
        report['snapshots'].append(record)
        a.out.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:record[k] for k in ['snapshot','quadrature_points','boundary_iterations','ordinary_step_ms','reuse_probe_step_ms','calls_per_step']},indent=2),flush=True)
    print('Saved '+str(a.out),flush=True)


if __name__=='__main__':main()
