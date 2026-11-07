"""Timing of the actual k1 run, phase by phase (11-06). merged_drop.py --profile: nothing in the model changes; each phase is
wrapped in a CUDA-synchronised timer (the syncs add a little to the total, which is reported alongside). Inclusive times:
  step begin        W.update_ul: the cut at the step's start
    radii solve       Recut.solve_radii (inside: geometry calls, the radii CG, repairs)
      geometry          Recut.geometry (inside: the BVH overlap search on fresh calls, the GPU structure)
    (assembly)        update_ul minus the radii solve: springs, walls, fit, cap data
  Newton solve      MergedStep.solve: predictor, the contact's set-up, reread, the gradient passes, CG
    gradient pass     the captured graph (one energy + gradient)
  step end          MergedStep.end: W.end_ul (polar, the yielding cells' eigh on the CPU, F, springs) and the rest
  (outside)         the driver's own per-step work: logs, frames, saves
Every `follow_every` steps, after the cut: the V0 follow map's rebuild at this state (the explicit line's engineering
baseline): the column estimate, the rows (k1_slave.anchor_map through FollowMap.add, tiny then stiff), the Woodbury build."""
import time
from collections import defaultdict

import numpy as np
import torch

import free_surface.merged_ul as MU


class Prof:
    def __init__(self):
        self.t = defaultdict(float); self.n = defaultdict(int); self.steps = []; self.cur = None; self.follow = []

    def timed(self, f, label):
        def g(*a, **k):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            try:
                return f(*a, **k)
            finally:
                torch.cuda.synchronize(); dt_ = time.perf_counter()-t0
                self.t[label] += dt_; self.n[label] += 1
                if self.cur is not None:
                    self.cur[label] = self.cur.get(label, 0.)+dt_
        return g

    def wrap(self, obj, name, label):
        setattr(obj, name, self.timed(getattr(obj, name), label))


def install(W, ST, target=9600., follow_every=50, log=print):
    P = Prof()
    R = W._recut
    P.wrap(R, 'solve_radii', 'radii solve'); P.wrap(R, 'geometry', 'geometry'); P.wrap(R, 'cg', 'radii CG'); P.wrap(R, 'repair', 'radii repair')
    P.wrap(R.A, 'overlaps', 'BVH overlaps'); P.wrap(R.A, 'structure_gpu', 'structure')
    P.wrap(W, 'update_ul', 'update_ul'); P.wrap(W, 'reread', 'reread'); P.wrap(W, 'end_ul', 'end_ul'); P.wrap(W, 'stored', 'stored (logs)')
    MU.polar_batch = P.timed(MU.polar_batch, 'end_ul: polar')
    eigh0 = torch.linalg.eigh
    torch.linalg.eigh = P.timed(eigh0, 'end_ul: eigh (CPU)')
    P.wrap(ST, '_eval', 'gradient pass'); P.wrap(ST.CG, 'solve', 'CG')
    P.wrap(ST, 'begin', 'step begin'); P.wrap(ST, 'solve', 'Newton solve'); P.wrap(ST, 'end', 'step end')
    step0 = ST.step
    state = dict(k=0, t_last=None)

    def step(*a, **k):
        now = time.perf_counter()
        if P.cur is not None and state['t_last'] is not None:
            P.cur['wall'] = now-state['t_last']; P.steps.append(P.cur)
        state['t_last'] = now; state['k'] += 1
        P.cur = {}
        out = step0(*a, **k)
        if state['k'] % follow_every == 0:
            follow_cost(W, ST, P, state['k'], target, log)
        return out
    ST.step = step
    P.finish = lambda: (P.steps.append(dict(P.cur, wall=time.perf_counter()-state['t_last'])) if P.cur is not None else None)
    return P


def follow_cost(W, ST, P, k, target, log):
    """the V0 follow map at this cut, as the explicit run would rebuild it (not used by this run)"""
    from free_surface.k1_follow import FollowMap
    from free_surface.k1_slave import column_omega
    t = W.tul; mt = W._w[5]; wi, wj, wa = t['wi'][:mt].long(), t['wj'][:mt].long(), t['wa'][:mt]
    X = ST.tx.clone(); m = W.V0
    s_ = torch.cuda.synchronize; pc = time.perf_counter
    s_(); t0 = pc(); om = column_omega(W); s_(); t1 = pc()
    FM = FollowMap(W.n)
    FM.add(X, wi, wj, wa, m < W._ul_agg*m.median()); FM.add(X, wi, wj, wa, om > target); s_(); t2 = pc()
    FM.build(m); s_(); t3 = pc()
    P.follow.append((k, t1-t0, t2-t1, t3-t2, int(FM.slave.sum())))
    log('  PROFILE step %d: the V0 follow map at this cut: column estimate %.0f ms, rows %.0f ms, Woodbury build %.0f ms; slaved %d' % (
        k, 1e3*(t1-t0), 1e3*(t2-t1), 1e3*(t3-t2), int(FM.slave.sum())), flush=True)


def report(P, total, log=print, windows=((1, 150), (151, 300), (301, 450), (451, 600), (601, 750))):
    P.finish()
    S = P.steps; ns = len(S)
    keys = ['step begin', 'update_ul', 'radii solve', 'geometry', 'BVH overlaps', 'structure', 'radii CG', 'radii repair', 'Newton solve', 'gradient pass', 'CG', 'reread',
            'step end', 'end_ul', 'end_ul: polar', 'end_ul: eigh (CPU)', 'stored (logs)']
    wall = sum(s_.get('wall', 0.) for s_ in S)
    log('PROFILE: %d steps, run wall %.1f s (the steps\' own %.1f s); per phase (inclusive): total s, ms per step, calls per step, ms per call' % (ns, total, wall))
    for k_ in keys:
        if P.n[k_]:
            log('PROFILE   %-20s %7.2f s  %7.2f ms/step  %6.2f calls/step  %7.3f ms/call  (%4.1f %% of the steps)' % (
                k_, P.t[k_], 1e3*P.t[k_]/ns, P.n[k_]/ns, 1e3*P.t[k_]/P.n[k_], 100*P.t[k_]/wall))
    top = P.t['step begin']+P.t['Newton solve']+P.t['step end']
    log('PROFILE   %-20s %7.2f s  %7.2f ms/step  (the driver\'s own per-step work: logs, frames, saves, the follow timing)' % ('(outside)', wall-top, 1e3*(wall-top)/ns))
    log('PROFILE   assembly in update_ul (update_ul - radii solve) %.2f ms/step; radii solve minus geometry and CG %.2f ms/step; geometry minus BVH and structure %.2f ms/step' % (
        1e3*(P.t['update_ul']-P.t['radii solve'])/ns, 1e3*(P.t['radii solve']-P.t['geometry']-P.t['radii CG']-P.t['radii repair'])/ns,
        1e3*(P.t['geometry']-P.t['BVH overlaps']-P.t['structure'])/ns))
    for a, b in windows:
        W_ = [s_ for i, s_ in enumerate(S, 1) if a <= i <= b]
        if not W_:
            continue
        g = lambda k_: 1e3*np.mean([s_.get(k_, 0.) for s_ in W_])
        log('PROFILE   steps %d-%d: wall %.1f ms/step: begin %.1f (radii %.1f, geometry %.1f), Newton %.1f (gradient %.1f), end %.1f (end_ul %.1f, eigh %.1f)' % (
            a, b, g('wall'), g('step begin'), g('radii solve'), g('geometry'), g('Newton solve'), g('gradient pass'), g('step end'), g('end_ul'), g('end_ul: eigh (CPU)')))
    if P.follow:
        f = np.array([r[1:4] for r in P.follow])
        log('PROFILE   the V0 follow map per cut (%d samples): column estimate %.0f ms, rows %.0f ms, build %.0f ms (means; max %.0f / %.0f / %.0f)' % (
            len(f), *(1e3*f.mean(0)), *(1e3*f.max(0))))


def install_cut(W, skip=0):
    """the explicit run's cut (W.update_ul) split by CUDA-synchronised timers (11-06): the radii solve and inside it the
    prediction, the CG, the geometry calls; inside a geometry call the BVH overlaps (fresh calls), the candidate planes
    (two argsorts), the structure, the values (geom_cls); the alpha kernel and the checks are the rest of a call. The
    first skip cuts are not counted"""
    P = Prof(); R = W._recut
    P.wrap(W, 'update_ul', 'update_ul'); P.wrap(R, 'solve_radii', 'radii solve')
    cg0 = R.cg; pr0 = R.predict

    def pr(*a, **k):                                                                # (the prediction's own CG timed as 'predict CG')
        R.cg = P.timed(cg0, 'predict CG')
        try:
            return pr0(*a, **k)
        finally:
            R.cg = cg_t
    R.predict = P.timed(pr, 'predict')
    cg_t = P.timed(cg0, 'radii CG'); R.cg = cg_t
    P.wrap(R, 'geometry', 'geometry'); P.wrap(R, 'repair', 'radii repair')
    P.wrap(R.A, 'overlaps', 'BVH overlaps'); P.wrap(R.A, 'candidates', 'candidates'); P.wrap(R.A, 'structure_gpu', 'structure')
    R.geom_cls = P.timed(R.geom_cls, 'values')
    if getattr(R, 'pipe', None) is not None:                                        # (geo_pipe: its phases by synchronised marks)
        last = [None]

        def mark(lb):
            torch.cuda.synchronize(); t_ = time.perf_counter()
            if lb != 'start' and last[0] is not None:
                P.t['pipe: '+lb] += t_-last[0]; P.n['pipe: '+lb] += 1
            last[0] = t_
        R.pipe.timer = mark
    ASM = ('springs', 'walls', 'fit', 'aggregation', 'wall data', 'caps, polar')
    f1 = W.update_ul

    def upd_marks(*a, **k):                                                         # (the assembly's blocks: W._marks set by merged_ul._mk)
        W._marks = []
        try:
            return f1(*a, **k)
        finally:
            mk = W._marks; W._marks = None
            for nm_, a_, b_ in zip(ASM, mk[:-1], mk[1:]):
                P.t['asm: '+nm_] += b_-a_; P.n['asm: '+nm_] += 1
    W.update_ul = upd_marks
    if skip:
        f0, cnt = W.update_ul, [0]

        def upd(*a, **k):
            if cnt[0] == skip:
                P.t.clear(); P.n.clear()
            cnt[0] += 1
            return f0(*a, **k)
        W.update_ul = upd
    return P


def install_struct_count(W):
    """logging only (11-06): how much of the cut's structure (alpha_complex.structure_gpu: the pairs with a wall in the
    balls, each wall's links, the hidden sites) changes between geometry calls: within a radii solve (its first call
    against its result) and from one cut's result to the next cut's first call and result. Keys: pair i n + j (i < j),
    link (i n + j) n + k; links counted on the pairs both structures have"""
    R = W._recut; n = W.n
    rec = dict(prev=None, rows=[], calls=None)

    def keys(S):
        P_ = S['pairs'].long(); pk = P_[:, 0]*n+P_[:, 1]
        L_ = S['links'].long(); m_ = L_ >= 0
        return pk, (pk[:, None].expand_as(L_)*n+L_)[m_], S['hidden'].clone()

    def diff(a, b):
        pa, la, ha = a; pb, lb, hb = b
        common = pa[torch.isin(pa, pb)]
        la_, lb_ = la[torch.isin(la//n, common)], lb[torch.isin(lb//n, common)]
        return (int((~torch.isin(pb, pa)).sum()), int((~torch.isin(pa, pb)).sum()), int((~torch.isin(lb_, la_)).sum()), int((~torch.isin(la_, lb_)).sum()),
                int((ha != hb).sum()))
    g0, s0 = R.geometry, R.solve_radii

    def geometry(*a, **k):
        G, S = g0(*a, **k)
        if rec['calls'] is not None:
            rec['calls'].append(keys(S))
        return G, S

    def solve_radii(*a, **k):
        rec['calls'] = []
        out = s0(*a, **k)
        fin = keys(out[2]); first = rec['calls'][0] if rec['calls'] else fin
        row = dict(calls=len(rec['calls']), within=diff(first, fin), pairs=len(fin[0]), links=len(fin[1]))
        if rec['prev'] is not None:
            row['to_first'], row['step'] = diff(rec['prev'], first), diff(rec['prev'], fin)
        rec['prev'] = fin; rec['rows'].append(row); rec['calls'] = None
        return out
    R.geometry, R.solve_radii = geometry, solve_radii
    return rec


def report_struct(rec, log=print, skip=1):
    rows = rec['rows'][skip:]
    if not rows:
        return
    log('STRUCT %d cuts: pairs (walls in the balls) median %d, links median %d' % (len(rows), int(np.median([r_['pairs'] for r_ in rows])), int(np.median([r_['links'] for r_ in rows]))))
    names = ('pairs added', 'pairs removed', 'links added', 'links removed', 'hidden flips')
    for key, what in (('within', 'within a cut (its first call -> its result)'), ('to_first', 'last cut\'s result -> this cut\'s first call'), ('step', 'last cut\'s result -> this cut\'s result')):
        D = np.array([r_[key] for r_ in rows if key in r_])
        if not len(D):
            continue
        anyc = (D.sum(1) > 0)
        log('STRUCT   %s: cuts with any change %d of %d; %s' % (what, int(anyc.sum()), len(D), ', '.join('%s mean %.1f max %d' % (nm_, D[:, q].mean(), D[:, q].max()) for q, nm_ in enumerate(names))))


def report_cut(P, ns, log=print):
    keys = ['update_ul', 'radii solve', 'predict', 'predict CG', 'radii CG', 'radii repair', 'geometry', 'BVH overlaps', 'candidates', 'structure', 'values',
            'pipe: overlaps', 'pipe: candidates', 'pipe: clip', 'pipe: structure', 'pipe: read', 'pipe: values',
            'asm: springs', 'asm: walls', 'asm: fit', 'asm: aggregation', 'asm: wall data', 'asm: caps, polar']
    log('PROFILE cut over %d steps: per part (inclusive) ms per step, calls per step, ms per call' % ns)
    for k_ in keys:
        if P.n[k_]:
            log('PROFILE   %-14s %7.2f ms/step  %5.2f calls/step  %7.3f ms/call' % (k_, 1e3*P.t[k_]/ns, P.n[k_]/ns, 1e3*P.t[k_]/P.n[k_]))
    t = lambda k_: 1e3*P.t[k_]/ns
    log('PROFILE   assembly (update_ul - radii solve - predict) %.2f ms/step; radii solve minus geometry, its CG, repairs: %.2f ms/step; '
        'predict minus its CG %.2f ms/step; geometry minus BVH, candidates, structure, values (the alpha kernel, the status check): %.2f ms/step' % (
            t('update_ul')-t('radii solve')-t('predict'), t('radii solve')-t('geometry')-t('radii CG')-t('radii repair'), t('predict')-t('predict CG'),
            t('geometry')-t('BVH overlaps')-t('candidates')-t('structure')-t('values')))
