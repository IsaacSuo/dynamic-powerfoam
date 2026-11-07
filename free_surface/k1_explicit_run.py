"""k1 stepped explicitly, with the follow map (11-06: the follow version's actual run). merged_drop.py --explicit --theta 1.

The model is k1's (MergedUL), read at theta = 1: its energy at a point z of the step is the end state's, so its gradient
at the step's start x_n (G = 0) is the force of the state there: the cells' shape stress from F_n (rotation and return
factor read at x_n), the pressure from L_n (C read at x_n: G = 0 gives C = I, the volume reading linear in the step and
its work equal to its store), the springs' kept remainders, the ground's Hertz force and the glue force carried in lamt.
One step:
  cut       W.update_ul(x_n): radii (the start predicted from the last cut, --radii-start predict), walls, fit, springs (every step, as k1)
  follow    (--follow v0) the V0 map rebuilt on this cut (k1_follow_fast.FollowFast: tiny cells, then the cells whose
            column estimate exceeds the target; one host sync)
  force     the balls' step-start data (k_mid_ball), the readings at x_n (reread), f = m g - dE/dx at x_n
  kick      v <- P M_r^-1 P^T M v + dt P M_r^-1 P^T f (follow: the projection onto the new map, mass-orthogonal, and the
            acceleration in one triangular solve; the projection's kinetic energy change logged) or v + dt f / m; x_n+1 = x_n + dt v
  end       the graph at x_n+1 on cut n, then W.end_ul(1): F_n+1 = (I + G) F_n with the return, L_n+1 = L_n + tr G,
            the springs, the plastic work; the glue's force kept (k_mid_lamt)
Timing per phase CUDA-synchronised (the syncs add a little). FRAME lines as the implicit driver's."""
import math
import time

import torch
import warp as wp
from torch.profiler import record_function as _rf

from free_surface.closure_warp import k_mid_ball, k_mid_lamt

f64 = torch.float64


def run(W, ST, mass, fg, dt, t_end, row, log, follow='v0', target=9600., log_every=200, frame_dt=1/24, follow_every=1, time_from=1, trace_start=False, tprof=None, cut_every=1, cut_eps=0., omega_at=(), strain_log=0., strain_sample=False, diag_from=0):
    assert abs(W.th-1.) < 1e-15, 'the explicit run reads k1 at theta = 1 (--theta 1)'
    n = W.n; dev = ST.tx.device; M3 = mass[:, None]; bl = W.ball; A = ST.a
    x = ST.tx.clone(); v = ST.tv.clone()
    T = dict(cut=0., follow=0., force=0., kick=0., end=0., out=0.)
    sy = torch.cuda.synchronize; pc = time.perf_counter
    RI = []; RT = []
    FM = None; dKE = 0.; dKE_t = torch.zeros((), dtype=f64, device=dev); nsteps = int(math.ceil(t_end/dt-1e-9)); kf = 1
    E0 = None; t_wall = pc()

    def gr(z):
        W.tx.copy_(z); wp.capture_launch(W.graph)
        return W.tgx

    def ke(v_):
        return .5*float((M3*v_*v_).sum())
    log('EXPLICIT dt %.3e s, %d steps to t %.3f, follow %s (rebuilt every %d steps)' % (dt, nsteps, t_end, follow, follow_every))
    lagged = cut_every > 1 or cut_eps > 0                                           # (k1_lag: the cut kept over steps, the state rebased on it)
    if lagged:
        from free_surface.k1_lag import Lag, omega_max, strain_max, strain_report
        LG = Lag(W); since = 0; cuts = []; Fri = None
        log('EXPLICIT the cut kept: a new cut every %s steps%s (k1_lag)' % (cut_every if cut_every > 1 else '-', ', or when a cell\'s strain since the cut passes %.3g' % cut_eps if cut_eps > 0 else ''))
    W._recut.trace0 = trace_start                                                   # (the start's worst cells, logging only)
    W._recut.log_start = False                                                      # (the start's V / target ratios: the implicit driver's logs only)
    PR = None                                                                       # (tprof: dict(start, n, path), k1_tprof: a torch.profiler trace of n steps)
    win = dict(cuts=0, steps=0, t=pc())                                             # (the log window's cuts and wall time)
    dg = dict(ke=None, omega=False)                                                 # (diag_from: per step from there, Lanczos once when KE grows 5 % a step)
    for s in range(1, nsteps+1):
        if s == time_from and s > 1:
            T = dict.fromkeys(T, 0.)                                                 # (the steps before time_from not timed: warm-up, a restored state's first cut)
        if tprof is not None and s == tprof['start']:
            PR = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]); PR.start()
        rf_step = _rf('step'); rf_step.__enter__()
        sy(); t0 = pc()
        cut = True
        if lagged and s > 1:
            em = strain_max(W)                                                      # (at x_n on the kept cut: the last step's end evaluated it)
            cut = (cut_every > 1 and since >= cut_every) or (cut_eps > 0 and em > cut_eps)
            if cut:
                cuts.append((s, since, em))
                if (strain_log > 0 and em > strain_log) or (strain_sample and win.get('sample', True)):   # (logging only: the strain's spread and the most
                    strain_report(W, FM, x, log=log, tag='step %d t %.4f, %d steps since the cut' % (s, s*dt, since))   # strained cells; above the
                    win['sample'] = False                                           # threshold, or the window's first cut)
        win['steps'] += 1; win['cuts'] += int(cut)
        if cut:
            g0 = W._recut.stats['geometry']; c0_ = W._recut.stats['cg']
            W.update_ul(x)
            RI.append((W.ul_stats.get('radii_it', -1), W._recut.stats['geometry']-g0, W._recut.stats.get('err0', float('nan')), W._recut.stats['cg']-c0_))
            if getattr(W._recut, 'trace0', False):
                RT.append((s, W._recut.stats.get('top0', []), W._ulr.clone()))
            if lagged:
                since = 1; Fri = None
        else:
            Fri = LG.rebase(W, x); since += 1
        sy(); t1 = pc(); T['cut'] += t1-t0
        if follow == 'v0' and cut and ((s-1) % follow_every == 0 or FM is None or lagged):
            from free_surface.k1_follow_fast import FollowFast
            if FM is None:
                FM = FollowFast(n)                                                   # (one map: its buffers and graphs kept over the cuts)
            with _rf('follow'):
                FM.rebuild(W, x, mass, target)
        sy(); t2 = pc(); T['follow'] += t2-t1
        rf_ = _rf('force'); rf_.__enter__()
        ST.tx.copy_(x)
        wp.launch(k_mid_ball, bl['k'], inputs=[A['x'], bl['body'], bl['R'], bl['ng'], ST.c0, ST.g8, ST.cb], outputs=[bl['xn'], bl['kt'], bl['bound']])
        if getattr(W, 'stick', False):
            W.tbbound.copy_(torch.minimum(W.tbbound, W.tau_y*W.footprint(x, W._ulr)))
        gr(x); W.reread(beta=True, keep_C=not cut)
        f = fg-gr(x)
        if lagged and s in omega_at and FM is not None:                             # (timing off: Lanczos on this state's follow pencil)
            omega_max(W, FM, mass, x, gr, log=log, tag='step %d (%s, %d steps since the cut)' % (s, 'a cut' if cut else 'kept', since))
        rf_.__exit__(None, None, None)
        sy(); t3 = pc(); T['force'] += t3-t2
        rf_ = _rf('kick'); rf_.__enter__()
        if FM is not None and not cut:                                               # (the map kept: v is in its range, the kick alone)
            v = v+dt*FM.P(FM.force_to_accel(f))
        elif FM is not None:                                                         # (v projected onto this cut's map and kicked, one solve:
            q1, q2 = FM.accel2(M3*v, f)                                              # v <- P M_r^-1 P^T M v + dt P M_r^-1 P^T f; the projection's
            vp = FM.P(q1); dKE_t += .5*((M3*vp*vp).sum()-(M3*v*v).sum()); v = vp+dt*FM.P(q2)   # kinetic energy change kept apart, on the device)
        else:
            v = v+dt*f/M3
        x1 = x+dt*v
        rf_.__exit__(None, None, None)
        sy(); t4 = pc(); T['kick'] += t4-t3
        rf_ = _rf('end'); rf_.__enter__()
        gr(x1); W.end_ul(1., Fri=Fri if lagged else None)
        wp.launch(k_mid_lamt, bl['k'], inputs=[W.B['bff'], W.B['bdelta']], outputs=[bl['lamt']])
        rf_.__exit__(None, None, None)
        x = x1
        sy(); t5 = pc(); T['end'] += t5-t4
        if diag_from and s >= diag_from:                                            # (logging only, a sync a step)
            from free_surface.k1_lag import describe, omega_max as om_
            ke_ = .5*float((M3*v*v).sum()); vm = v.norm(dim=1); iv = int(vm.argmax())
            Gs_ = .5*(W.tG+W.tG.transpose(1, 2)); es = (Gs_*Gs_).sum((1, 2)).sqrt(); ie = int(es.argmax())
            ri_ = RI[-1] if cut and RI else (-1, 0, float('nan'), 0)
            rs_ = W._recut.stats
            log('DIAG step %d t %.5f: KE %.5e (x %.4f), |v| max %.3e at %s, the step\'s strain max %.3e at %s, 99.9 %% %.2e; slaved %d; cut %s: radii it %s, geometry %s, err0 %.2e, CG %s; overflow %d, stalled %d, radii fails %d' % (
                s, s*dt, ke_, ke_/dg['ke'] if dg['ke'] else 1., float(vm[iv]), describe(W, FM, x1, iv), float(es[ie]), describe(W, FM, x1, ie), float(torch.quantile(es, .999)),
                int(FM.slave.sum()) if FM is not None else 0, cut, ri_[0], ri_[1], ri_[2], ri_[3], rs_.get('overflow', 0), rs_.get('stalled', 0), W.ul_stats.get('radii_fail', 0)))
            if FM is not None and getattr(FM, 'last', None) is not None:          # (the slaved set's changes at this cut: who left, who came, why)
                sl_now = FM.slave.clone(); L_ = FM.last
                if dg.get('slv') is not None:
                    gone = torch.nonzero(dg['slv'] & ~sl_now).flatten().tolist(); came = torch.nonzero(~dg['slv'] & sl_now).flatten().tolist()
                    why = lambda c_: '%d (column estimate %.0f, tiny %d, flagged %d, row %d)' % (c_, float(L_['om'][c_]), int(L_['lev1'][c_]), int(L_['new2'][c_]), int(FM.t['Fn'][c_] > 0))
                    if gone or came:
                        log('DIAG   slaved set at step %d: left %s; came %s' % (s, '; '.join(why(c_) for c_ in gone[:8]) or '-', '; '.join(why(c_) for c_ in came[:8]) or '-'))
                dg['slv'] = sl_now
            if dg['ke'] and ke_ > 1.05*dg['ke'] and not dg['omega'] and FM is not None:
                dg['omega'] = True
                om_(W, FM, mass, x1, gr, log=log, tag='step %d end (KE x %.3f a step)' % (s, ke_/dg['ke']), where=3)
            dg['ke'] = ke_
        rf_ = _rf('out'); rf_.__enter__()
        ST.tx.copy_(x); ST.tv.copy_(v)
        if s % log_every == 0 or s == nsteps:                                       # (the checks and the energies read here only: no sync a step)
            if getattr(W, 'end_mode', 'torch') != 'torch':                         # (the device's plastic work and counts read here)
                W.end_warp.flush(W)
            bad = not bool(torch.isfinite(x).all()); dKE = float(dKE_t)
            Es, Ev = W.stored(); k_ = ke(v); g_ = float((-fg*x).sum())
            Etot = k_+Es+Ev+g_+W.Wp
            if E0 is None:
                E0 = Etot
            log('  EXPLICIT step %d t %.4f: KE %.3e, stored shape %.3e volume %.3e, plastic work %.3e, total %.4e (start %.4e); projection dKE so far %.2e; slaved %d; %.1f s; the last %d steps: %d cuts, %.1f ms/step' % (
                s, s*dt, k_, Es, Ev, W.Wp, Etot, E0, dKE, int(FM.slave.sum()) if FM is not None else 0, pc()-t_wall, win['steps'], win['cuts'], 1e3*(pc()-win['t'])/max(win['steps'], 1)))
            win = dict(cuts=0, steps=0, t=pc())
            if bad or k_ > 1e3*max(abs(E0), 1e-12):
                log('EXPLICIT stopped at step %d: %s' % (s, 'not finite' if bad else 'the kinetic energy blew up'))
                break
        while kf*frame_dt <= s*dt+1e-12:
            row(kf); kf += 1
        rf_.__exit__(None, None, None)
        sy(); T['out'] += pc()-t5
        rf_step.__exit__(None, None, None)
        if PR is not None and s == tprof['start']+tprof['n']-1:
            PR.stop(); PR.export_chrome_trace(str(tprof['path'])); PR = None
            log('TPROF trace of steps %d-%d: %s' % (tprof['start'], s, tprof['path']))
    if lagged and cuts:
        import numpy as np
        ks = np.array([c_[1] for c_ in cuts]); es = np.array([c_[2] for c_ in cuts])
        log('EXPLICIT the cut kept: %d cuts after the first in %d steps; steps between cuts median %d (min %d, max %d); the largest cell strain since the cut at the cut: median %.2e, max %.2e' % (
            len(cuts), s, int(np.median(ks)), ks.min(), ks.max(), np.median(es), es.max()))
    tot = sum(T.values()); ns = s-time_from+1 if s >= time_from else s
    log('EXPLICIT timing over %d steps, %.1f s: %s' % (ns, tot, ', '.join('%s %.2f ms/step (%.0f %%)' % (k_, 1e3*v_/ns, 100*v_/tot) for k_, v_ in T.items())))
    if RT:                                                                          # (the start's worst cells: which, and how their radius moves)
        import numpy as np
        V0m = float(W.V0.median()); tiny = W.V0 < W._ul_agg*W.V0.median(); sl = FM.slave if FM is not None else torch.zeros(n, dtype=torch.bool, device=dev)
        from collections import Counter
        cnt_ = Counter(c_ for _, top_, _ in RT for c_ in top_[:1])
        log('EXPLICIT the start\'s worst cell per step: %s' % ', '.join('%d x%d (V0/med %.3f, %s, %s)' % (c_, k_, float(W.V0[c_])/V0m, 'tiny' if bool(tiny[c_]) else 'not tiny',
            'slaved' if bool(sl[c_]) else 'master') for c_, k_ in cnt_.most_common(6)))
        for c_, _ in cnt_.most_common(3):
            rr = np.array([float(r_[c_]) for _, _, r_ in RT]); d1 = np.diff(rr); d2 = np.diff(rr, 2)
            log('EXPLICIT cell %d: radius / h at the start %.4f; its converged change per step: median |dr/r| %.2e, max %.2e; the change of the change (what a linear extrapolation misses): median %.2e, max %.2e' % (
                c_, rr[0]/V0m**(1/3), float(np.median(np.abs(d1/rr[1:]))), float(np.abs(d1/rr[1:]).max()), float(np.median(np.abs(d2/rr[2:]))), float(np.abs(d2/rr[2:]).max())))
    if RI:
        import numpy as np
        it_ = np.array([r_[0] for r_ in RI]); gc_ = np.array([r_[1] for r_ in RI]); e0_ = np.array([r_[2] for r_ in RI])
        cg_ = np.array([r_[3] for r_ in RI])
        log('EXPLICIT the radii solve per cut: Newton iterations %s; geometry calls mean %.2f (max %d); the start error median %.2e, 90 %% %.2e, max %.2e; CG iterations per cut (the prediction\'s and the Newton\'s) median %d, max %d' % (
            ', '.join('%d: %d steps' % (k_, int((it_ == k_).sum())) for k_ in sorted(set(it_.tolist()))), gc_.mean(), gc_.max(), np.median(e0_), np.percentile(e0_, 90), e0_.max(), int(np.median(cg_)), int(cg_.max())))
    return x, v
