"""Full-state frozen-step timing including unchanged physical acceptance checks."""
import argparse
import collections
import contextlib
import json
from pathlib import Path
import statistics
import time
import torch
import warp as wp
from free_surface.fv_ale_hyperelastic import HyperelasticALE
from free_surface.fv_ale_releasing import ReleasingHyperelasticALE, ReleaseRequired
from free_surface.fv_ale_planar_boundary import Planes
from free_surface.fv_ale_solid import State
from free_surface.fv_ale_hyperelastic_config import StepAudit


def load(path):
    d = torch.load(path, map_location='cpu', weights_only=False)
    m = d['material']
    source = json.loads(path.with_suffix('.json').read_text())
    planar = d.get('moving_planes') is not None
    cls = ReleasingHyperelasticALE if planar else HyperelasticALE
    sol = cls(d['p'], d['r'], m['E'], m['nu'], m['ys'], groups=d['groups'],
        contact_model='plane', ground=d['ground'], gravity=(0., 0., 0.), friction=0.,
        wall_mode='compatible_slip_acoustic' if planar else 'unsupported',
        mechanical_reconstruction='compatible', affine_stencil='smooth', affine_cap_motion='balanced',
        cap_dissipation=0., interface_dissipation=source['interface_dissipation'],
        transport_reconstruction='entropy', transport_dissipation=source['transport_dissipation'],
        precision_geometry=True)
    p,r,q=(d[k].cuda() for k in ['p','r','q'])
    planes=Planes(**d['moving_planes']).to('cuda') if planar else None
    def reset():
        mesh=sol.cut(p,r,planes.clone()) if planar else sol.cut(p,r)
        return State(p,r,q,mesh)
    return sol,reset,source


class Timers:
    def __init__(self):self.saved=[];self.events=[];self.stack=[]
    def wrap(self,obj,key,name):
        old=getattr(obj,key);self.saved.append((obj,key,old))
        def call(*a,**kw):
            torch.cuda.synchronize();tick=time.perf_counter();frame=[name,0.];self.stack.append(frame)
            try:return old(*a,**kw)
            finally:
                torch.cuda.synchronize();elapsed=1000*(time.perf_counter()-tick);self.stack.pop()
                self.events.append((name,elapsed,elapsed-frame[1]))
                if self.stack:self.stack[-1][1]+=elapsed
        setattr(obj,key,call)
    def __enter__(self):return self
    def __exit__(self,*a):
        for obj,key,old in reversed(self.saved):setattr(obj,key,old)


def step(sol,state,source,cfl=.5,dt_max=float('inf')):
    def ready(s):
        if hasattr(sol,'compact_boundaries'):s,_=sol.compact_boundaries(s)
        for _ in range(32):
            try:return s,sol.rhs(s)
            except ReleaseRequired as e:s,_=sol.activate(s,e.cells)
        raise RuntimeError('activation limit')
    state,first=ready(state)
    saved_rhs=sol.rhs
    audit=StepAudit(sol,source['work_retry_tolerance_joule'],source['local_energy_tolerance'])
    # The cross-check is one-off in production, not repeated every physical step.
    audit.checked_rate=True;audit.checked_guard=True
    dt=min(cfl*first[3]['dt_scale'],dt_max);retry=0;activation=0;reasons=[]
    try:
        if dt<1e-12:
            raise RuntimeError('physical dt below 1e-12; not advanced')
        audit.begin(state,first)
        while True:
            try:
                end,stats=sol.advance(state,dt,first,cfl);audit.check(end,dt);break
            except ReleaseRequired as e:
                activation+=1
                if activation>32:raise
                state,_=sol.activate(state,e.cells);state,first=ready(state)
                dt=min(dt,cfl*first[3]['dt_scale']);audit.begin(state,first)
            except RuntimeError as e:
                reasons.append(str(e))
                retry+=1
                if retry>8:raise
                dt*=.5
        return end,{'dt':dt,'retry':retry,'activation':activation,'retry_reasons':reasons,'audit':audit.report()}
    finally:
        sol.rhs=saved_rhs


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('snapshot',type=Path)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--repeat',type=int,default=5)
    ap.add_argument('--unpad-planar',action='store_true')
    ap.add_argument('--reference-geometry',action='store_true')
    args=ap.parse_args();torch.set_num_threads(1);wp.init()
    sol,reset,source=load(args.snapshot)
    sol.device_ops().pad_planar_graph=not args.unpad_planar
    sol.fast_planar_geometry=not args.reference_geometry
    for _ in range(2):step(sol,reset(),source)
    elapsed=[];rows=[]
    for _ in range(args.repeat):
        s=reset();torch.cuda.synchronize();tick=time.perf_counter();end,row=step(sol,s,source)
        torch.cuda.synchronize();elapsed.append(1000*(time.perf_counter()-tick));rows.append(row)
    import free_surface.fv_ale_affine_stencil as stencil
    import free_surface.fv_ale_planar_cutter as planar
    stages=collections.defaultdict(list)
    for _ in range(3):
        s=reset()
        with Timers() as timers:
            for key in ['cut','rhs','advance','plastic_return']:timers.wrap(sol,key,key)
            timers.wrap(sol.recut,'geometry','real_geometry')
            timers.wrap(sol.recut,'structure','real_topology')
            if sol.ground_cutter:timers.wrap(sol.ground_cutter,'geometry','ground_geometry')
            timers.wrap(sol.device_ops(),'moments','cap_moments')
            timers.wrap(stencil,'build','affine_stencil');timers.wrap(stencil,'compact','affine_compact')
            timers.wrap(stencil,'matrices','affine_matrices');timers.wrap(stencil,'rows','affine_rows')
            timers.wrap(stencil,'eigh3','affine_eigen')
            timers.wrap(planar,'clipped_values','augmented_geometry')
            timers.wrap(planar,'augmented_structure','augmented_topology')
            timers.wrap(sol.device_ops(),'call','device_graph')
            end,row=step(sol,s,source)
        sums=collections.defaultdict(lambda:[0.,0.,0])
        for name,inc,exc in timers.events:
            sums[name][0]+=inc;sums[name][1]+=exc;sums[name][2]+=1
        for name,value in sums.items():stages[name].append(value)
    result={'snapshot':str(args.snapshot),'ordinary_ms':elapsed,'median_ms':statistics.median(elapsed),
        'stages':{k:{'inclusive_ms':statistics.median(v[0] for v in vals),
                      'exclusive_ms':statistics.median(v[1] for v in vals),'calls':vals[-1][2]} for k,vals in stages.items()},
        'rows':rows,'memory_allocated_MB':torch.cuda.memory_allocated()/1e6,
        'memory_reserved_MB':torch.cuda.memory_reserved()/1e6,'graph_builds':sol.device_ops().builds,
        'scope':'Warm full RK2 step incl. two cuts, boundary readiness and unchanged energy guard; reset cut excluded; synchronized stages measured separately.'}
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','graph_builds']},indent=2))


if __name__=='__main__':main()
