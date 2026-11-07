"""Screening the follow map level by level on fixed states (11-06): k1_follow.FollowMap's nested levels at a step's start,
each level reporting
  speed     the consistent reduced omega_max of the pencil (P^T K P, P^T M P) by Lanczos in the M_r inner product, and where
            its fastest converged modes sit
  fidelity  the response over one implicit step of the full and of the reduced system to the same load, A = K + c M,
            c = 4 / dt^2 (dt 1/1200, k1's step): e = |u_f - P u_r|_A / |u_f|_A (the reduced solution is the A-projection of the
            full one, so e^2 is the share of the response's energy the constraints remove); loads: gravity m g, and at a loaded
            state its own forces -dE/dx
  cost      per explicit step one gradient pass, the graph P and the graph P^T + M_r^-1; the explicit step 0.9 x 2 / omega_max;
            the time for the landing's 0.625 s from these (the cut's cost not in it); the factorisation once per cut
  checks    the graphs against the torch references (P, P^T level by level, M_r^-1 by CG on P^T M P), P^T against P
Levels: 0 none; 1 the tiny cells (V0 < 0.1 median, k1's aggregation set); 2 + the cells whose column estimate exceeds the
target; 3.. + the cells holding >= 20 % of a converged reduced mode above the target. Called from merged_drop.py's probe."""
import math
import time

import numpy as np
import torch
import warp as wp

from free_surface.k1_follow import FollowMap, neighbour_lists, add_batch, add_sequential, row_stats
from free_surface.k1_screen_diag import expand_rows, mode_energy, mode_stats, row_history
from free_surface.k1_slave import column_omega

f64, dev = torch.float64, 'cuda'


def cg(A, b, Pinv, tol=1e-10, maxit=3000):
    x = torch.zeros_like(b); r = b.clone(); z = Pinv(r); p = z.clone(); rz = float((r*z).sum()); b0 = float(b.norm())
    for it in range(maxit):
        Ap = A(p); al = rz/float((p*Ap).sum()); x = x+al*p; r = r-al*Ap
        if float(r.norm()) <= tol*b0:
            return x, it+1
        z = Pinv(r); rz1 = float((r*z).sum()); p = z+(rz1/rz)*p; rz = rz1
    return x, maxit


def gen_lanczos(Kf, Mf, Minv, n, k, mask, seed=0):
    """Lanczos on the pencil (K, M): an M-orthonormal basis, M^-1 by the caller; returns lambda, the Ritz vectors' coefficients,
    the basis and the Ritz residuals"""
    q = torch.randn(n, 3, dtype=f64, device=dev, generator=torch.Generator(device=dev).manual_seed(seed))*mask[:, None]
    q = q/math.sqrt(float((q*Mf(q)).sum()))
    Q = [q]; MQ = [Mf(q)]; al, be = [], []
    for it in range(k):
        w = Kf(Q[it]); a = float((w*Q[it]).sum()); al.append(a)
        z = Minv(w)-a*Q[it]-(be[-1]*Q[it-1] if it > 0 else 0.)
        for _ in range(2):
            for qi, mqi in zip(Q, MQ):
                z = z-float((mqi*z).sum())*qi
        mz = Mf(z); b = math.sqrt(max(float((z*mz).sum()), 0.)); be.append(b)
        Q.append(z/b); MQ.append(mz/b)
    lam, S = np.linalg.eigh(np.diag(al)+np.diag(be[:-1], 1)+np.diag(be[:-1], -1))
    return lam, S, Q[:k], be[-1]*np.abs(S[-1, :])


def timed(fn, reps=200):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter()-t0)/reps


VARIANTS = {'V0': 'the earlier rule (the 3 largest walls, the batch excluded)', 'V1': 'all 1-ring candidates, the batch excluded',
            'V2': 'all 1-ring candidates, one at a time', 'V3': 'V2, the anchor fit on the 2-ring where the 1-ring does not span',
            'V4': 'the direct affine reconstruction from the masters around (1-ring, else 2-ring), one at a time',
            'V2r': 'V2, the hot set also by the masters\' signed products q_k . (M_r q)_k >= 20 % of q^T M_r q (a heuristic: with the coupled mass these are no shares, they can be negative or above 100 %)',
            'V4b': 'V4 made dimensionless, QR weights, the anchor fit\'s quality rules (spanning ratio 1e-3, |w| <= 4 preferred, 2-ring tried)'}
VARIANTS['V2p'] = ('V2, then the patch study (k1_patch_study.py): the whole gradient stencils of a mode\'s 3 most energetic cells on one affine '
                   'motion of 4 control points (the largest tetrahedron of the stencils\' points), masses and energies kept; relations replaced, not nested')
_E0_DONE = set()


def screen(W, gr, X, step, log, k=100, target=9600., dt=1/1200., h_abs=1e-8, rounds=6, T_land=.625, variant='V0', watch=(3817,)):
    n = W.n; t = W.tul; m = W.V0; capc = t['Mc'].abs().sum((1, 2)) > 0; Vm = float(m.median())
    mt = W._w[5]; wi, wj, wa = t['wi'][:mt].long(), t['wj'][:mt].long(), t['wa'][:mt]
    c = 4/dt**2
    gen = torch.Generator(device=dev).manual_seed(3)
    rnd = lambda: torch.randn(n, 3, dtype=f64, device=dev, generator=gen)

    def Kv(v):
        s_ = h_abs/max(float(v.abs().max()), 1e-300)
        return (gr(X+s_*v)-gr(X-s_*v))/(2*s_)
    t_g = timed(lambda: gr(X), 20)
    Kbb = W.ul_blocks()
    loads = {'gravity': m[:, None]*torch.tensor([0., -9.8, 0.], dtype=f64, device=dev)}
    f_state = -gr(X)
    if float(f_state.norm()) > 1e-6*float(loads['gravity'].norm()):
        loads['state forces'] = f_state
    I3 = torch.eye(3, dtype=f64, device=dev)
    Afull = lambda v: Kv(v)+c*m[:, None]*v
    Bf = torch.linalg.inv(Kbb+c*m[:, None, None]*I3)
    ufull = {}
    for name, f in loads.items():
        u_, _ = cg(Afull, f, lambda r_: torch.einsum('nab,nb->na', Bf, r_), tol=1e-9)
        ufull[name] = (u_, float((u_*Afull(u_)).sum()))
    FM = FollowMap(n)
    om_est = column_omega(W)
    plan = [('none', None), ('tiny', m < W._ul_agg*m.median()), ('+ stiff by estimate', om_est > target)]
    nb = neighbour_lists(X.cpu().numpy(), wi.cpu().numpy(), wj.cpu().numpy(), wa.cpu().numpy())
    order = torch.argsort(om_est, descending=True).cpu().numpy()                      # (one at a time: the stiffest first)
    log('  SCREEN step %d variant %s: %s' % (step, variant, VARIANTS[variant]))
    hot = None; lev = 0; last = None
    while True:
        if lev < len(plan):
            label, sel = plan[lev]
        elif lev >= len(plan)+rounds:
            log('  SCREEN step %d STOP at level %d: the %d rounds used up' % (step, lev, rounds)); break
        elif hot is None or not bool((hot & ~FM.slave).any()):
            log('  SCREEN step %d STOP at level %d: no new candidate (%s)' % (step, lev, last['why'] if last else '-')); break
        else:
            label, sel = '+ hot of the reduced modes', hot
        got = free = 0; wmax = 0.; t_sel = 0.
        if sel is not None:
            t0_ = time.perf_counter()
            sel_np = (sel & ~FM.slave).cpu().numpy()
            if variant == 'V0':
                got, free, wmax = FM.add(X, wi, wj, wa, sel)
            elif variant == 'V1':
                got, free = add_batch(FM, X, nb, sel_np, allcand=True, watch=watch, log=log)
            else:
                mode = {'V4': 'direct', 'V4b': 'direct_scaled'}.get(variant, 'anchor')
                got, free = add_sequential(FM, X, nb, sel_np, order, mode=mode, allcand=True, ring2=(variant == 'V3'), watch=watch, log=log)
            t_sel = time.perf_counter()-t0_
            if got == 0 and lev >= len(plan):                                         # (the hot cells found no anchor: nothing changes)
                log('  SCREEN step %d STOP at level %d: the %d hot cells left all found no row' % (step, lev, free))
                break
        torch.cuda.synchronize(); tf0 = time.perf_counter()
        FM.build(m)
        torch.cuda.synchronize(); t_build = time.perf_counter()-tf0
        mk = FM.master[:, None].to(f64)
        # checks: the graphs against the torch references
        y1, r1, b1 = rnd()*mk, rnd(), rnd()*mk
        eP = float((FM.P(y1)-FM.P_ref(y1)).norm()/FM.P_ref(y1).norm())
        ePt = float((FM.Pt(r1)-FM.Pt_ref(r1)).norm()/FM.Pt_ref(r1).norm())
        eT = abs(float((FM.P(y1)*r1).sum())-float((y1*FM.Pt(r1)).sum()))/abs(float((FM.P(y1)*r1).sum()))
        Mf = lambda y_: FM.Pt(m[:, None]*FM.P(y_))
        dM = torch.maximum(Mf(torch.ones(n, 3, dtype=f64, device=dev)*mk), .5*m[:, None])   # (a Jacobi guess: the row sums, floored)
        xs = FM.force_to_accel(b1)
        xc, itc = cg(lambda y_: FM.M_ref(m, y_), b1, lambda r_: r_/dM*mk, tol=1e-13, maxit=3000)
        eS = float((xs-xc).norm()/xc.norm()); rS = float((FM.M_ref(m, xs)-b1).norm()/b1.norm())
        # speed
        Kf = lambda y_: FM.Pt(Kv(FM.P(y_)))
        lam, S, Q, res = gen_lanczos(Kf, Mf, FM.force_to_accel, n, k, mk[:, 0])
        om = np.sqrt(np.clip(lam, 0, None)); conv = res <= 1e-6*np.abs(lam).max()
        hot = torch.zeros(n, dtype=torch.bool, device=dev); where = []
        hot_u = torch.zeros(n, dtype=torch.bool, device=dev); hot_r = torch.zeros(n, dtype=torch.bool, device=dev)
        cnt = dict(modes=0, free_hot=0, only_slaved=0, none=0, red_hot=0, red_only=0); slsh = []; top_mode = None
        for r_ in range(1, k+1):
            if om[-r_] <= target:
                break
            if not conv[-r_]:
                continue
            qv = sum(float(S[i, -r_])*Q[i] for i in range(len(Q)))
            ms_ = mode_stats(FM, m, qv, Mf(qv)); pv = ms_['u']; wv = (pv*pv).sum(1); wm = ms_['wm']
            hot_u |= ms_['hot_u']; hot_r |= ms_['hot_r']
            cnt['modes'] += 1; slsh.append(ms_['slaved_share'])
            fh, rh_ = bool(ms_['hot_u'].any()), bool(ms_['hot_r'].any())
            cnt['free_hot'] += fh; cnt['only_slaved'] += (not fh) and ms_['any_u']; cnt['none'] += not ms_['any_u']
            cnt['red_hot'] += rh_; cnt['red_only'] += rh_ and not fh
            if top_mode is None:
                top_mode = ms_
            if len(where) < 2:
                tc = int(torch.argmax(wm))
                where.append('%d (%s, V/med %.2f; m|u|^2 %.0f %%, |u|^2 %.0f %%)' % (tc, 'surface' if bool(capc[tc]) else 'inside', float(m[tc])/Vm,
                                                                                    100*float(wm[tc]), 100*float(wv[tc]/wv.sum())))
        hot = (hot_u | hot_r) if variant == 'V2r' else hot_u                         # (V2r: also the masters that let a slaved particle's mode exist)
        # fidelity
        Ar = lambda y_: FM.Pt(Afull(FM.P(y_)))
        Br = torch.linalg.inv(Kbb+c*dM[:, :, None]*I3)
        fid = []
        for name, f in loads.items():
            uf, Ef = ufull[name]
            if not FM.levels:
                fid.append('%s 0' % name); continue
            ur, _ = cg(Ar, FM.Pt(f), lambda r_: torch.einsum('nab,nb->na', Br, r_)*mk, tol=1e-9)
            d = uf-FM.P(ur)
            fid.append('%s %.2e' % (name, math.sqrt(max(float((d*Afull(d)).sum()), 0.)/Ef)))
        # cost: the graphs alone, and with the torch copies around them
        t_Pg = timed(lambda: wp.capture_launch(FM.graphs['P'])) if 'P' in FM.graphs else 0.
        t_Fg = timed(lambda: wp.capture_launch(FM.graphs['force']))
        t_P = timed(lambda: FM.P(y1)); t_F = timed(lambda: FM.force_to_accel(r1))
        dt_ex = .9*2/om[-1]; t_step = t_g+t_Pg+t_Fg; T_run = T_land/dt_ex*t_step
        rs = row_stats(FM, m)
        rows_txt = ('rows (expanded onto the final masters, 99 %% / max): max |w| %.2f / %.2f, sum |w| %.2f / %.2f, m_s sum w^2 / m_k %.2f / %.2f' % (
            *rs['wmax'], *rs['wsum'], *rs['mamp'])) if rs['n'] else 'rows: none'
        log('  SCREEN step %d level %d (%s): slaved %d (+%d, %d left free, max |w| %.2f) | omega_max %.4e (%s), converged above the target %d of %d; fastest at %s '
            '| fidelity over one step: %s | cost: dt %.2e s (%.2f of the MPM substep); graphs P %.3f ms, P^T + M_r^-1 %.3f ms (with the torch copies %.3f / %.3f), gradient %.3f ms '
            '-> %.3f ms per step, an estimate %.1f s for %.3f s of landing (this state\'s step cost x steps; cuts not included); choosing the rows %.0f ms, build %.0f ms, per cut | %s '
            '| checks: P %.1e, P^T %.1e, transpose %.1e, M_r^-1 vs CG (%d its) %.1e, residual %.1e' % (
                step, lev, label, int(FM.slave.sum()), got, free, rs.get('wmax', (0., 0.))[1] if rs['n'] else 0., om[-1], 'converged' if conv[-1] else 'NOT converged', int(((om > target) & conv).sum()), k,
                '; '.join(where) if where else '-', ', '.join(fid), dt_ex, dt_ex*4800, 1e3*t_Pg, 1e3*t_Fg, 1e3*t_P, 1e3*t_F, 1e3*t_g, 1e3*t_step, T_run, T_land,
                1e3*t_sel, 1e3*t_build, rows_txt, eP, ePt, eT, itc, eS, rS))
        # which particles carry the modes above the target, and what lets them move
        why = ('%d converged modes above the target: %d with a free particle >= 20 %% of m|u|^2, %d whose >= 20 %% particles are all slaved, %d with no particle >= 20 %%; '
               'by the masters\' signed q_k.(M_r q)_k >= 20 %%: %d modes (%d of them with no free particle >= 20 %% of m|u|^2); the slaved particles\' share of m|u|^2 per mode: median %.0f %%, max %.0f %%' % (
                   cnt['modes'], cnt['free_hot'], cnt['only_slaved'], cnt['none'], cnt['red_hot'], cnt['red_only'],
                   100*float(np.median(slsh)) if slsh else 0., 100*float(np.max(slsh)) if slsh else 0.))
        log('  SCREEN step %d level %d MODES: %s; new candidates by m|u|^2 %d, by q.(M_r q) %d' % (step, lev, why, int(hot_u.sum()), int(hot_r.sum())))
        if top_mode is not None:
            rows, depth = expand_rows(FM) if FM.levels else ({}, {})
            wm_, red_ = top_mode['wm'], top_mode['red']; txt = []
            for p_ in torch.argsort(wm_, descending=True)[:3].tolist():
                if p_ in rows:
                    ms_ = sorted(rows[p_].items(), key=lambda q_: -abs(q_[1]))[:8]
                    txt.append('%d slaved (m|u|^2 %.0f %%, depth %d) <- masters %s' % (p_, 100*float(wm_[p_]), depth[p_], ', '.join(
                        '%d w %+.2f (m|u|^2 %.0f %%, q.Mq %.0f %%)' % (j, w, 100*float(wm_[j]), 100*float(red_[j])) for j, w in ms_)))
                else:
                    txt.append('%d master (m|u|^2 %.0f %%, q.Mq %.0f %%)' % (p_, 100*float(wm_[p_]), 100*float(red_[p_])))
            rtop = torch.argsort(red_, descending=True)[:4].tolist()
            log('  SCREEN step %d level %d TOP MODE: %s | the largest signed q_k.(M_r q)_k: %s' % (step, lev, ' | '.join(txt), ', '.join('%d %.0f %%' % (j, 100*float(red_[j])) for j in rtop)))
            for w_ in watch:
                h_ = row_history(FM, w_) if w_ in rows else None
                if h_:
                    log('  SCREEN step %d level %d WATCH %d: slaved at FM level %d with %d points (max |w| %.2f, sum |w| %.2f); then %d substitutions: %s; final %d masters, max |w| %.2f, sum |w| %.2f, depth %d' % (
                        step, lev, w_, h_['level'], h_['hist'][0][2], h_['hist'][0][3], h_['hist'][0][4], len(h_['hist'])-1,
                        ', '.join('level %d replaced %d -> %d points, max |w| %.2f, sum |w| %.2f' % (li, p_, n_, a_, b_) for li, p_, n_, a_, b_ in h_['hist'][1:]) or '-',
                        len(h_['final']), max(abs(v) for v in h_['final'].values()), sum(abs(v) for v in h_['final'].values()), depth[w_]))
            if lev == 0 and step not in _E0_DONE:
                _E0_DONE.add(step)
                mode_energy(W, gr, X, top_mode['u'], log, 'step %d level 0 (nothing slaved) top mode' % step, FM=FM, red=top_mode['red'])
        last = dict(why=why, top=top_mode, lev=lev, om=om[-1])
        lev += 1
        if om[-1] <= target:
            log('  SCREEN step %d STOP after level %d: omega_max at or below the target' % (step, lev-1)); break
    if last is not None and last['top'] is not None and last['lev'] > 0:
        res = mode_energy(W, gr, X, last['top']['u'], log, 'step %d variant %s level %d (final) top mode, omega %.4e' % (step, variant, last['lev'], last['om']), FM=FM, red=last['top']['red'])
        if variant == 'V2p':
            from free_surface.k1_patch_study import study
            study(W, gr, X, FM, res, step, log, dict(m=m, Kv=Kv, Kbb=Kbb, loads=loads, k=k, target=target, capc=capc, Vm=Vm, cg=cg, gen_lanczos=gen_lanczos))
