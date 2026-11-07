"""Repeat an identical production k1 Newton problem, without time advancement.

Old pre-step snapshots omit recut/geometry objects. Reconstruct the previous cut
once, call the real begin() once, and capture every numeric solver/energy buffer
at the first _newton entry. Restore that exact snapshot before each repetition.
This isolates numerical solve variability from changes in cut/group/frozen C.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
import warp as wp
from free_surface.k1_snapshot import take, restore, to_host, inventory
from free_surface.k1_patchcg import SplitCG


def digest(snap):
    h=hashlib.sha256()
    def feed(v):
        if isinstance(v,torch.Tensor):
            a=v.detach().cpu().contiguous().numpy();h.update(str((a.dtype,a.shape)).encode());h.update(a.tobytes())
        elif isinstance(v,np.ndarray):
            h.update(str((v.dtype,v.shape)).encode());h.update(v.tobytes())
        elif isinstance(v,(list,tuple)):
            h.update(type(v).__name__.encode())
            for item in v:feed(item)
        else:h.update(repr(v).encode())
    for p,v in sorted(snap['values'].items(),key=lambda q:repr(q[0])):
        h.update(repr(p).encode());feed(v)
    return h.hexdigest()


def selected_snapshot(current,reference):
    # Ignore attributes created solely by diagnostics, but check every leaf of
    # the frozen reference, including duplicated tensor views and scalar state.
    return dict(values={p:current['values'][p] for p in reference['values']})


def run(W,ST,CG,path,tol,wref,out,tag,count=5):
    if count<2:raise ValueError('repeat-count must be >= 2')
    out=Path(out);out.mkdir(exist_ok=True,parents=True)
    source=torch.load(path,weights_only=False)
    roots=dict(W=W,ST=ST,CG=CG)
    # Allocate/capture fallback scratch before freezing; this object is skipped
    # by the old snapshot walker unless explicitly supplied as another root.
    ST._splitcg=SplitCG(CG);roots['PATCH']=ST._splitcg
    restore(roots,source)
    assert hasattr(W,'integration'), 'use the same integration flag to allocate captured buffers'
    assert ST.newton_max==300 and ST.c_two_pass and ST.pre_every==1 and ST.c2_refresh
    assert float(W.integration_threshold)==.5 and int(W.integration_max_cells)==8
    restored=take(roots)
    input_hash=digest(selected_snapshot(restored,source))
    source_hash=digest(source)
    assert input_hash==source_hash, ('pre-step snapshot did not restore bitwise',input_hash,source_hash)
    print('REPEATCHECK restored all saved numeric leaves bitwise: '+source_hash,flush=True)

    # Rebuild omitted previous geometry once, so radii prediction is based on
    # the restored previous cut, not the newly constructed model's initial cut.
    G,_=W._recut.geometry(W.tul['xn'].clone(),W._ulr.clone())
    G.gen=W._ul_gen;W._ulG=G
    ST.begin()
    geometry=dict(walls=int(W._w[-1]),springs=int(W._m_s),integration=dict(W.integration_stats))
    original_newton=ST._newton;captured={}
    class Capture(Exception):pass
    def capture(*args):
        assert args[-1]==1
        captured['arguments']=args
        captured['snapshot']=take(roots)
        raise Capture()
    ST._newton=capture
    try:
        ST.solve(tol,lambda g:.5*tol)
        raise RuntimeError('Newton capture was not reached')
    except Capture:pass
    finally:ST._newton=original_newton
    frozen=captured['snapshot'];arguments=captured['arguments']
    frozen_hash=digest(frozen)
    torch.save(dict(to_host(frozen),step=source['step'],repeatcheck_stage='first Newton entry'),out/(tag+'_frozen.pt'))
    print('REPEATCHECK frozen first-pass problem: %s; initial |g|/fg %.9g; %s' % (frozen_hash,math.sqrt(arguments[0])/wref,geometry),flush=True)

    # Repeated evaluations at EXACTLY the same z and frozen coefficients.
    evals=[];gradient_ref=None
    for k in range(6):
        restore(roots,frozen)
        values=ST._eval(0.)
        grad=wp.to_torch(ST.a['gz']).clone()
        if gradient_ref is None:gradient_ref=grad.clone()
        evals.append(dict(index=k,gradient_norm=float(grad.norm()),
            gradient_relative_difference=float((grad-gradient_ref).norm()/gradient_ref.norm()),
            gradient_max_abs_difference=float((grad-gradient_ref).abs().max()),scalars=values))
    print('REPEATCHECK identical-point gradient relative differences: %s' % [q['gradient_relative_difference'] for q in evals],flush=True)

    report=dict(original_snapshot=str(path),step=source['step'],prestep_hash=source_hash,
        frozen_hash=frozen_hash,frozen_geometry=geometry,dt=float(ST.dt),tol_over_fg=tol/wref,
        old_snapshot_skipped=source.get('skipped'),frozen_skipped=frozen.get('skipped'),
        scope='One reconstructed cut, identical first-Newton-entry numeric state and frozen C/R/beta. No recut or grouping inside repeats. No end-state commit.',
        coefficient_evaluations=evals,repeats=[],complete=False)
    refs={}
    for k in range(count):
        restore(roots,frozen)
        actual=digest(selected_snapshot(take(roots),frozen))
        assert actual==frozen_hash, ('frozen replay not bitwise equal',k,actual)
        st=copy.deepcopy(arguments[4]);args=list(arguments);args[4]=st
        torch.cuda.synchronize();start=time.perf_counter()
        ne,ni,g2,delta=original_newton(*args)
        pass1=dict(iterations=ni,evaluations=ne,residual_over_fg=math.sqrt(max(g2,0))/wref,converged=math.isfinite(g2) and g2<=tol*tol)
        st['its1'],st['g1']=ni,math.sqrt(max(g2,0))
        if pass1['converged']:
            ne,ni,g2,delta=ST.pass2(tol,lambda g:.5*tol,st,ne,ni,delta,ST.c2_refresh)
        ST.stats=st
        torch.cuda.synchronize();sec=time.perf_counter()-start
        z=ST.tz.clone();trace=st['trace']
        if k==0:refs['z']=z.clone();refs['trace']=copy.deepcopy(trace)
        first_branch=None
        for j,(a,b) in enumerate(zip(refs['trace'],trace)):
            # Discrete CG length, boundary/negative-curvature exit, acceptance.
            if (a[0],a[1],a[2],a[6])!=(b[0],b[1],b[2],b[6]):first_branch=j+1;break
        record=dict(repeat=k+1,start_hash=actual,pass1=pass1,iterations=ni,evaluations=ne,
            residual_over_fg=math.sqrt(max(g2,0))/wref,converged=math.isfinite(g2) and g2<=tol*tol,
            seconds=sec,first_discrete_trace_difference_vs_first=first_branch,
            max_z_difference_vs_first=float((z-refs['z']).abs().max()),
            rms_z_difference_vs_first=float((z-refs['z']).square().mean().sqrt()),
            rejected=st['rejected'],cg_cap_count=sum(int(q[0]>=ST.cg_cap) for q in trace),
            trace=trace,two_pass=st.get('two_pass'))
        # At the final accepted z; logging only, after solve timing and result.
        fresh=[]
        for j in range(3):
            ST._eval(0.);fresh.append(float(wp.to_torch(ST.a['gz']).norm())/wref)
        record['fresh_final_residuals_over_fg']=fresh
        record['min_final_trial_increment_det']=float(torch.linalg.det(W.I3+W.tG/ST.th).min())
        torch.save(dict(z=z.cpu(),g=wp.to_torch(ST.a['gz']).cpu(),G=W.tG.cpu(),C=W.tul['Q'].cpu()),out/(tag+'_repeat%d.pt'%(k+1)))
        report['repeats'].append(record)
        (out/(tag+'.json')).write_text(json.dumps(report,indent=2)+'\n')
        print('REPEATCHECK RESULT '+json.dumps({a:b for a,b in record.items() if a not in ('trace','two_pass')}),flush=True)
    report['complete']=True
    (out/(tag+'.json')).write_text(json.dumps(report,indent=2)+'\n')
    print('REPEATCHECK COMPLETE %d / %d converged' % (sum(q['converged'] for q in report['repeats']),count),flush=True)
