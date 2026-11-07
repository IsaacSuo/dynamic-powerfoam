"""GPU time against wall time per part of the explicit k1 step (11-06, the structural review; timing only).
torch.profiler (CUPTI) over a few steps; every part labelled (record_function: the runner's phases, the cut's methods,
the assembly's blocks through merged_ul._mk); each GPU kernel / memcpy / memset is charged to the innermost label around
its launch (the launch's correlation id), and the host syncs (cuda*Synchronize, cu*Synchronize) are counted per label.
install(W) before the run, the runner starts and stops the profiler (k1_explicit_run.run(tprof=...)), report(path)."""
import collections
import json

import torch
from torch.profiler import record_function

import free_surface.merged_ul as MU

ASM = ('asm.springs', 'asm.walls', 'asm.fit', 'asm.aggregation', 'asm.wall data', 'asm.caps, polar')


def _wrap(obj, name, label):
    f = getattr(obj, name)

    def g(*a, **k):
        with record_function(label):
            return f(*a, **k)
    setattr(obj, name, g)


def install(W):
    R = W._recut
    _wrap(W, 'update_ul', 'cut'); _wrap(R, 'predict', 'predict'); _wrap(R, 'solve_radii', 'radii')
    _wrap(R, 'geometry', 'geom'); _wrap(R, 'cg', 'cg'); _wrap(R, 'jacobian', 'jacobian')
    _wrap(R.A, 'candidates', 'geom.candidates'); _wrap(R.A, 'structure_gpu', 'geom.structure')
    R.geom_cls = (lambda f_: (lambda *a, **k: _rf_call(f_, 'geom.values', *a, **k)))(R.geom_cls)
    W._marks = None
    W._rf_asm = []                                                                  # (merged_ul._mk: the assembly's blocks as labels)
    mk0 = MU._mk

    def mk(W_):
        st = getattr(W_, '_rf_asm', None)
        if st is None:
            return mk0(W_)
        k = getattr(W_, '_rf_k', 0)
        if st:
            st.pop().__exit__(None, None, None)
        if k < len(ASM):
            rf = record_function(ASM[k]); rf.__enter__(); st.append(rf)
        W_._rf_k = k+1
    MU._mk = mk
    f_up = W.update_ul

    def upd(*a, **k):
        W._rf_k = 0
        try:
            return f_up(*a, **k)
        finally:
            while W._rf_asm:
                W._rf_asm.pop().__exit__(None, None, None)
    W.update_ul = upd


def _rf_call(f, label, *a, **k):
    with record_function(label):
        return f(*a, **k)


SYNC = ('cudaStreamSynchronize', 'cudaDeviceSynchronize', 'cudaEventSynchronize', 'cuStreamSynchronize', 'cuCtxSynchronize', 'cuEventSynchronize')


def report(path, nsteps, log=print):
    ev = json.load(open(path))['traceEvents']
    X = [e for e in ev if e.get('ph') == 'X']
    ann = [e for e in X if e.get('cat') == 'user_annotation']
    api = [e for e in X if e.get('cat') in ('cuda_runtime', 'cuda_driver')]
    gpu = [e for e in X if e.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset')]
    # the innermost label at a host time: labels sorted by start, nested on one thread
    by_tid = collections.defaultdict(list)
    for a in ann:
        by_tid[a['tid']].append((a['ts'], a['ts']+a['dur'], a['name']))
    for k_ in by_tid:
        by_tid[k_].sort()

    def labels_at(tid, t):
        out = []
        for s_, e_, nm in by_tid.get(tid, ()):
            if s_ <= t <= e_:
                out.append((e_-s_, nm))
        out.sort()
        return [nm for _, nm in out]                                                 # (innermost first)
    corr = {}
    launches = collections.Counter(); syncs = collections.Counter(); syncs_in = collections.Counter()
    for a in api:
        labs = labels_at(a['tid'], a['ts'])
        inner = labs[0] if labs else '(none)'
        c = a.get('args', {}).get('correlation')
        if c is not None:
            corr[c] = labs
        if a['name'] in SYNC:
            syncs[inner] += 1
            for l_ in labs:
                syncs_in[l_] += 1
        elif 'Launch' in a['name'] or 'Memcpy' in a['name'] or 'Memset' in a['name']:
            launches[inner] += 1
    excl = collections.Counter(); incl = collections.Counter(); kcount = collections.Counter(); kinds = collections.defaultdict(collections.Counter)
    for g in gpu:
        labs = corr.get(g.get('args', {}).get('correlation'), [])
        inner = labs[0] if labs else '(none)'
        excl[inner] += g['dur']; kcount[inner] += 1
        kinds[inner][g['name'][:60]] += g['dur']
        for l_ in set(labs):
            incl[l_] += g['dur']
    wall = collections.Counter(); calls = collections.Counter()
    for a in ann:
        wall[a['name']] += a['dur']; calls[a['name']] += 1
    total_gpu = sum(g['dur'] for g in gpu)
    steps_wall = wall.get('step', 0)
    log('TPROF %d steps: GPU busy %.1f ms/step (all kernels, copies, sets) of the steps\' wall %.1f ms/step' % (nsteps, total_gpu/1e3/nsteps, steps_wall/1e3/nsteps))
    log('TPROF   label              wall ms/step  GPU incl ms/step  GPU excl ms/step  launches/step (excl)  syncs/step (incl)  calls/step')
    order = ['step', 'cut', 'predict', 'jacobian', 'cg', 'radii', 'geom', 'geom.candidates', 'geom.structure', 'geom.values'] + list(ASM) + ['follow', 'force', 'kick', 'end', 'out']
    for nm in order+sorted(set(wall)-set(order)):
        if nm not in wall:
            continue
        log('TPROF   %-18s %9.2f %16.2f %16.2f %14.1f %20.1f %12.2f' % (nm, wall[nm]/1e3/nsteps, incl[nm]/1e3/nsteps, excl[nm]/1e3/nsteps, launches[nm]/nsteps, syncs_in[nm]/nsteps, calls[nm]/nsteps))
    log('TPROF   (none: launched outside every label) GPU %.2f ms/step, launches %.1f/step, syncs %.1f/step' % (excl['(none)']/1e3/nsteps, launches['(none)']/nsteps, syncs['(none)']/nsteps))
    for nm in ('geom', 'geom.candidates', 'geom.structure', 'geom.values', 'cut', 'asm.springs', 'follow', 'end', 'force', 'kick', 'predict', 'cg', 'radii'):
        top = kinds[nm].most_common(5)
        if top:
            log('TPROF   top GPU in %-16s %s' % (nm, '; '.join('%s %.2f' % (k_, v_/1e3/nsteps) for k_, v_ in top)))
