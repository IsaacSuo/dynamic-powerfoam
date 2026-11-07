"""The patch study (11-06, after the expert's sixth reply). Logging only, on a frozen state, after the V2 screen.

A patch: the cells holding the most of a mode's energy, and every point whose motion enters their G (their gradient
stencils, term_triplets); all of them on one affine motion u = b + H x of 4 control points (the largest tetrahedron of the
points; x the cut's positions). Building it REPLACES relations: the rows of the patch's points are removed, the controls
become masters, the others follow the controls; rows outside that refer to patch points then expand differently. So a
patched space is not nested in V2's and the Rayleigh bound against V2 does not apply; the study counts the outside rows
that changed.

  affine check  by the gradient's own coefficients (V_s G_s = sum_k u_k (x) c_sk), no differences: E_s = V_s^-1 sum_k x_k c_sk^T - I
                and t_s = V_s^-1 sum_k c_sk on every cell, and the fit's own sum_k (x_O - x_S) R^T - I; for the patch cells |b (x) t + H E| / |H|
  configs       at rest: V2, A (the V2 top mode's cluster), A + B, A + C, A + D (B, C, D: the clusters of A's three top modes, each
                added alone), A + B + C + D; overlapping point sets merged and logged; at the impact state V2 and A
  spectrum      omega_max, the converged modes above the target, the three top modes (where, whether on a patch, their top cells)
  responses     one step of A_c = K + c M at c = 4 / dt^2 for dt = 1/1200 (k1's step) and dt = 1/100 (nearly static: inertia
                below 200 rad/s); loads: gravity, the state's forces (when loaded), and per patch a smooth dent (a Gaussian
                normal load of the patch's rms radius) and a bending couple (the same Gaussian times the in-plane offset);
                e = |u_f - u_c|_A / |u_f|_A, d = |u_c - u_V2|_A / |u_f|_A, and on the loaded patch's points the displacement
                error and the cells' G error (sum V0 |dG|^2) against the full response
  impact mode   (impact state, config A's top mode) the top cells' volume change dL = tr(C^T dG) split by stencil point over the
                points' motion relative to the cell's own (exact: sum_k c_sk = 0), and by term; 168's mass, motion, coefficients;
                the contact's direct share and which top cells touch"""
import math

import numpy as np
import torch

from free_surface.k1_diag2 import term_triplets
from free_surface.k1_screen_diag import barycentric, expand_rows, max_volume_tet, mode_energy, mode_stats

f64, dev = torch.float64, 'cuda'


def coeff_rows(W):
    """the gradient's coefficients summed over the terms: (s, k, c) with V_s G_s = sum u_k (x) c, and by term"""
    T = term_triplets(W)
    s = torch.cat([v[0] for v in T.values()]); k = torch.cat([v[1] for v in T.values()]); c = torch.cat([v[2] for v in T.values()])
    return s, k, c, T


def affine_check(W, cells, log, tag):
    n = W.n; t = W.tul; xn = t['xn']; Vi = t['Vi']; I3 = torch.eye(3, dtype=f64, device=dev)
    s, k, c, T = coeff_rows(W)
    E = torch.zeros(n, 3, 3, dtype=f64, device=dev).index_add(0, s, xn[k][:, :, None]*c[:, None, :])*Vi[:, None, None]-I3
    tv = torch.zeros(n, 3, dtype=f64, device=dev).index_add(0, s, c)*Vi[:, None]
    hs = W.V0**(1/3)
    ml = W._ml; S_, O_, R_ = t['S'][:2*ml].long(), t['O'][:2*ml].long(), t['R'][:2*ml]
    Ef = torch.zeros(n, 3, 3, dtype=f64, device=dev).index_add(0, S_, (xn[O_]-xn[S_])[:, :, None]*R_[:, None, :])
    fitted = torch.zeros(n, dtype=torch.bool, device=dev); fitted[S_] = True
    Ef = Ef-I3*fitted[:, None, None]
    En = E.flatten(1).norm(dim=1); iso = (E.diagonal(dim1=1, dim2=2).sum(1)/3).abs()
    log('  AFFINE %s: by the coefficients, every cell: |V^-1 sum x c^T - I| median %.1e, max %.1e (cell %d; its isotropic part %.1e); |V^-1 sum c| h max %.1e; the fits\' |sum (x_O - x_S) R^T - I| max %.1e; '
        'the patch cells: %s' % (
            tag, float(En.median()), float(En.max()), int(En.argmax()), float(iso[int(En.argmax())]), float((tv.norm(dim=1)*hs).max()), float(Ef[fitted].flatten(1).norm(dim=1).max()),
            ', '.join('%d %.1e' % (c_, float(En[c_])) for c_ in cells)))
    return s, k, c, E, tv


def cell_G(s, k, c, Vi, u, n):
    return torch.zeros(n, 3, 3, dtype=f64, device=dev).index_add(0, s, u[k][:, :, None]*c[:, None, :])*Vi[:, None, None]


def stencils(s, k, c, cells):
    cs = torch.as_tensor(list(cells), device=dev); sel = torch.isin(s, cs) & (c.norm(dim=1) > 0)
    return set(k[sel].tolist()) | set(cells)


def make_patch(W, cells, s, k, c, name):
    xn = W.tul['xn'].cpu().numpy(); S = stencils(s, k, c, cells); Sl = sorted(S)
    ci, vol = max_volume_tet(xn[Sl]); ctrl = [Sl[i] for i in ci]; lam = barycentric(xn[ctrl], xn[Sl])
    return dict(name=name, cells=list(cells), S=S, Sl=Sl, ctrl=ctrl, lam=lam, vol=vol)


def merge(W, patches, s, k, c):
    """patches whose point sets overlap become one (logged); returns the final list and the merge record"""
    out, rec = [], []
    for p in patches:
        hit = [q for q in out if q['S'] & p['S']]
        if not hit:
            out.append(p); continue
        cells = list(dict.fromkeys(p['cells']+[c_ for q in hit for c_ in q['cells']]))
        rec.append('%s overlaps %s (%s shared points) -> merged' % (p['name'], '+'.join(q['name'] for q in hit), '+'.join(str(len(q['S'] & p['S'])) for q in hit)))
        out = [q for q in out if q not in hit]
        out.append(make_patch(W, cells, s, k, c, '+'.join([q['name'] for q in hit]+[p['name']])))
    return out, rec


def patched_map_multi(FM, patches):
    from free_surface.k1_follow import FollowMap
    allS = set().union(*[p['S'] for p in patches]); St = torch.as_tensor(sorted(allS), device=dev)
    FM2 = FollowMap(FM.n)
    for s_, k_, w_ in FM.levels:
        keep = ~torch.isin(s_, St)
        if bool(keep.any()):
            FM2.levels.append((s_[keep], k_[keep], w_[keep]))
    FM2.slave = FM.slave.clone(); FM2.slave[St] = False
    T = lambda a, dt=torch.long: torch.as_tensor(np.asarray(a), dtype=dt, device=dev)
    ss, kk, ww, sl = [], [], [], []
    for p in patches:
        pos = {q: i for i, q in enumerate(p['Sl'])}
        for q in p['Sl']:
            if q in p['ctrl']:
                continue
            sl.append(q); ss += [q]*4; kk += p['ctrl']; ww += [float(p['lam'][pos[q], j]) for j in range(4)]
    FM2.levels.append((T(ss), T(kk), T(ww, f64))); FM2.slave[T(sl)] = True
    return FM2


def patch_loads(W, p, capc):
    """a smooth dent and a bending couple on the patch: Gaussian of the patch's rms radius, along the cap normal"""
    xn = W.tul['xn']; m = W.V0; Sl = torch.as_tensor(p['Sl'], device=dev)
    xb = xn[Sl].mean(0); R = float(((xn[Sl]-xb)**2).sum(1).mean().sqrt())
    cc = [c_ for c_ in p['cells'] if bool(capc[c_])]
    nv = W.tul['a'][torch.as_tensor(cc, device=dev)].sum(0) if cc else xb-(m[:, None]*xn).sum(0)/m.sum()
    nh = nv/nv.norm()
    e = torch.tensor([1., 0., 0.], dtype=f64, device=dev)
    if float(abs(nh@e)) > .9:
        e = torch.tensor([0., 1., 0.], dtype=f64, device=dev)
    th = torch.linalg.cross(nh, e); th = th/th.norm()
    g = torch.exp(-((xn-xb)**2).sum(1)/(2*R*R))
    dent = -(m*g)[:, None]*nh
    bend = (m*g*((xn-xb)@th)/R)[:, None]*nh
    return {'dent ' + p['name']: dent, 'bend ' + p['name']: bend}, R


def study(W, gr, X, FM, res, step, log, ctx):
    """ctx: the screen's m, Kv, Kbb, loads, k, target, capc, Vm, cg, gen_lanczos"""
    n = W.n; m = ctx['m']; Kv = ctx['Kv']; Kbb = ctx['Kbb']; capc = ctx['capc']; Vm = ctx['Vm']; cg = ctx['cg']; gen_lanczos = ctx['gen_lanczos']
    k_l, target = ctx['k'], ctx['target']; I3 = torch.eye(3, dtype=f64, device=dev); Vi = W.tul['Vi']
    cA = [int(c_) for c_ in res['top'][:3]]
    s, k, c, E, tv = affine_check(W, cA, log, 'step %d' % step)
    pA = make_patch(W, cA, s, k, c, 'A')
    rows0, _ = expand_rows(FM)

    def describe(FMx, patches):
        allS = set().union(*[p['S'] for p in patches]) if patches else set()
        rows, _ = expand_rows(FMx) if FMx.levels else ({}, {})
        changed = sum(1 for q, r in rows.items() if q not in allS and (set(r) != set(rows0.get(q, {})) or any(abs(r[j]-rows0[q].get(j, 0.)) > 1e-12 for j in r)))
        xn = W.tul['xn']; hmed = float(Vm**(1/3))
        return ('; '.join('%s: cells %s, %d points (%d slaved in V2), controls %s, max |lambda| %.2f, sum |lambda| max %.2f, diameter %.1f median cell sizes' % (
            p['name'], p['cells'], len(p['S']), int(FM.slave[torch.as_tensor(p['Sl'], device=dev)].sum()), p['ctrl'], float(np.abs(p['lam']).max()),
            float(np.abs(p['lam']).sum(1).max()), float(torch.cdist(xn[torch.as_tensor(p['Sl'], device=dev)], xn[torch.as_tensor(p['Sl'], device=dev)]).max())/hmed) for p in patches)
            + ' | rows outside the patches whose expansion changed: %d of %d; masters %d' % (changed, len(rows), int(FMx.master.sum())))

    def spectrum(FMx, label, patches):
        mk = FMx.master[:, None].to(f64)
        Mf = lambda y_: FMx.Pt(m[:, None]*FMx.P(y_)); Kf = lambda y_: FMx.Pt(Kv(FMx.P(y_)))
        lam_, S_, Q_, res_ = gen_lanczos(Kf, Mf, FMx.force_to_accel, n, k_l, mk[:, 0])
        om = np.sqrt(np.clip(lam_, 0, None)); conv = res_ <= 1e-6*np.abs(lam_).max()
        allS = set().union(*[p['S'] for p in patches]) if patches else set()
        tops = []
        for r_ in range(1, 4):
            qv = sum(float(S_[i, -r_])*Q_[i] for i in range(len(Q_))); ms_ = mode_stats(FMx, m, qv, Mf(qv))
            en = mode_energy(W, gr, X, ms_['u'], log, '', FM=FMx, verbose=False)
            onp = float(ms_['wm'][torch.as_tensor(sorted(allS), device=dev)].sum()) if allS else 0.
            tops.append(dict(om=om[-r_], conv=bool(conv[-r_]), u=ms_['u'], cells=en['top'][:3], share=[float(en['cellE'][c_]/en['uKu']) for c_ in en['top'][:3]],
                             onp=onp, rest=en['rest']/en['uKu']))
        log('  STUDY step %d %s: omega_max %.4e (%s), converged above the target %d of %d; top modes: %s' % (
            step, label, om[-1], 'converged' if conv[-1] else 'NOT converged', int(((om > target) & conv).sum()), k_l,
            '; '.join('omega %.4e%s, energy in %s, m|u|^2 on the patches %.0f %%' % (
                t_['om'], '' if t_['conv'] else ' (not converged)', ', '.join('%d (%s%s, V/med %.2f) %.0f %%' % (
                    c_, 'slaved' if bool(FMx.slave[c_]) else 'master', ', surface' if bool(capc[c_]) else '', float(m[c_])/Vm, 100*sh)
                    for c_, sh in zip(t_['cells'], t_['share'])), 100*t_['onp']) for t_ in tops)))
        return tops

    # loads and the full responses at the two steps
    loads = dict(ctx['loads'])
    dts = (('k1 step', 1/1200.), ('slow', 1/100.))
    fullsol = {}

    def Afun(cv):
        return lambda v: Kv(v)+cv*m[:, None]*v

    def solve_full(name, f, cv):
        B = torch.linalg.inv(Kbb+cv*m[:, None, None]*I3)
        u_, it = cg(Afun(cv), f, lambda r_: torch.einsum('nab,nb->na', B, r_), tol=1e-9, maxit=20000)
        return u_, float((u_*Afun(cv)(u_)).sum()), it

    def solve_red(FMx, f, cv):
        mk = FMx.master[:, None].to(f64)
        Mf = lambda y_: FMx.Pt(m[:, None]*FMx.P(y_))
        dM = torch.maximum(Mf(torch.ones(n, 3, dtype=f64, device=dev)*mk), .5*m[:, None])
        B = torch.linalg.inv(Kbb+cv*dM[:, :, None]*I3)
        Ar = lambda y_: FMx.Pt(Afun(cv)(FMx.P(y_)))
        ur, it = cg(Ar, FMx.Pt(f), lambda r_: torch.einsum('nab,nb->na', B, r_)*mk, tol=1e-9, maxit=10000)
        return FMx.P(ur), it

    def responses(FMx, label, uV2, patches_loaded):
        out = {}
        for dname, dt_ in dts:
            cv = 4/dt_**2; txt = []
            for name, f in loads.items():
                uf, Ef, itf = fullsol[(name, dname)]
                u_, it = solve_red(FMx, f, cv); out[(name, dname)] = u_
                dd = uf-u_; e = math.sqrt(max(float((dd*Afun(cv)(dd)).sum()), 0.)/Ef)
                s_ = '%s: e %.4f' % (name, e)
                if uV2 is not None:
                    d2 = u_-uV2[(name, dname)]; s_ += ', d %.4f' % math.sqrt(max(float((d2*Afun(cv)(d2)).sum()), 0.)/Ef)
                p = patches_loaded.get(name)
                if p is not None:
                    Sl = torch.as_tensor(p['Sl'], device=dev); mw = m[Sl][:, None]
                    du = float(((mw*(u_[Sl]-uf[Sl])**2).sum()/(mw*uf[Sl]**2).sum()).sqrt())
                    Gf = cell_G(s, k, c, Vi, uf, n)[Sl]; Gc = cell_G(s, k, c, Vi, u_, n)[Sl]
                    dg = float(((W.V0[Sl][:, None, None]*(Gc-Gf)**2).sum()/(W.V0[Sl][:, None, None]*Gf**2).sum()).sqrt())
                    s_ += ', on its %d points: displacement %.4f, G %.4f' % (len(p['Sl']), du, dg)
                if it >= 10000:
                    s_ += ' (CG NOT converged)'
                txt.append(s_)
            log('  STUDY step %d %s responses (%s, dt %s): %s' % (step, label, dname, '1/%d' % round(1/dt_), ' | '.join(txt)))
        return out

    def run_config(label, patches, uV2):
        if patches:
            FMx = patched_map_multi(FM, patches); FMx.build(m)
            y = torch.randn(n, 3, dtype=f64, device=dev, generator=torch.Generator(device=dev).manual_seed(7))*FMx.master[:, None]
            u = FMx.P(y); eP = float((u-FMx.P_ref(y)).norm()/u.norm()); xn = W.tul['xn']; ea = 0.
            for p in patches:
                Xc, Uc = xn[p['ctrl']], u[p['ctrl']]
                H = torch.linalg.solve(Xc[:3]-Xc[3], Uc[:3]-Uc[3]).T; b = Uc[3]-H@Xc[3]; Sl = torch.as_tensor(p['Sl'], device=dev)
                ea = max(ea, float((u[Sl]-(b+xn[Sl]@H.T)).abs().max()/u[Sl].abs().max()))
            log('  STUDY step %d %s: %s | checks: P %.1e, u on each patch against its b + H x %.1e' % (step, label, describe(FMx, patches), eP, ea))
        else:
            FMx = FM
        tops = spectrum(FMx, label, patches)
        return FMx, tops

    # V2 and A
    FMA, topsA = None, None
    _, topsV2 = run_config('V2', [], None)
    pl = {}
    lA, RA = patch_loads(W, pA, capc); loads.update(lA); pl.update({kk: pA for kk in lA})
    FMA, topsA = run_config('A', [pA], None)
    others = []
    if step <= 1:                                                                    # (at rest: the clusters that take over after A)
        for i_, t_ in enumerate(topsA):
            others.append(make_patch(W, t_['cells'], s, k, c, 'BCD'[i_]))
        for p in others:
            lp, _ = patch_loads(W, p, capc); loads.update(lp); pl.update({kk: p for kk in lp})
    for name, f in loads.items():
        for dname, dt_ in dts:
            fullsol[(name, dname)] = solve_full(name, f, 4/dt_**2)
    log('  STUDY step %d: loads %s; full solves: CG iterations %s' % (step, ', '.join(loads), ', '.join('%s/%s %d' % (a_, b_, v_[2]) for (a_, b_), v_ in fullsol.items())))
    uV2 = responses(FM, 'V2', None, pl)
    responses(FMA, 'A', uV2, pl)
    if step <= 1:
        for p in others:
            pp, rec = merge(W, [pA, p], s, k, c)
            if rec:
                log('  STUDY step %d A + %s: %s' % (step, p['name'], '; '.join(rec)))
            FMx, _ = run_config('A + %s' % p['name'], pp, None)
            responses(FMx, 'A + %s' % p['name'], uV2, pl)
        pp, rec = merge(W, [pA]+others, s, k, c)
        if rec:
            log('  STUDY step %d A + B + C + D: %s' % (step, '; '.join(rec)))
        FMx, _ = run_config('A + B + C + D', pp, None)
        responses(FMx, 'A + B + C + D', uV2, pl)
    else:
        impact_mode(W, gr, X, FMA, topsA[0], s, k, c, step, log, ctx)


def impact_mode(W, gr, X, FMx, top, s, k, c, step, log, ctx, watch=168):
    n = W.n; m = ctx['m']; Vm = ctx['Vm']; t = W.tul; Vi = t['Vi']; Q = t['Q']
    u = top['u']; un = u.norm(dim=1); umax = float(un.max())
    en = mode_energy(W, gr, X, u, log, 'step %d config A top mode, omega %.4e' % (step, top['om']), FM=FMx, watch=(watch,))
    bd = torch.as_tensor(W.B['bdelta'].numpy(), device=dev)
    touch = bd > 0 if len(bd) == n else None
    log('  IMPACT step %d: the contact\'s direct share of u^T K u (the remainder of the exact split) %.2f %%; touching among the top cells: %s' % (
        step, 100*en['rest']/en['uKu'], ', '.join('%d %s' % (c_, ('yes' if bool(touch[c_]) else 'no') if touch is not None else '?') for c_ in en['top'][:4])))
    T = term_triplets(W)
    wm = m*(u*u).sum(1); wm = wm/wm.sum(); wv = (u*u).sum(1); wv = wv/wv.sum()
    mt = W._w[5]; wi, wj = t['wi'][:mt].long(), t['wj'][:mt].long()
    for cell in en['top'][:3]:
        dL = 0.; per, byterm = {}, {}
        for tn, (s0, k0, c0) in T.items():
            sel = s0 == cell
            kk = k0[sel]; cv = c0[sel]
            contr = Vi[cell]*((u[kk]-u[cell])*(cv@Q[cell].T)).sum(1)                # (u_k - u_s) . (C c): the relative motion's
            byterm[tn] = float(contr.sum())
            for q_, v_ in zip(kk.tolist(), contr.tolist()):
                per[q_] = per.get(q_, 0.)+v_
        dL = sum(byterm.values()); dLg = float(en['dL'][cell])
        nb = set(wj[wi == cell].tolist()) | set(wi[wj == cell].tolist())
        coef = {}
        for tn, (s0, k0, c0) in T.items():
            sel = s0 == cell
            for q_, cv in zip(k0[sel].tolist(), (c0[sel]@Q[cell].T).norm(dim=1).tolist()):
                coef[q_] = coef.get(q_, 0.)+cv*float(Vi[cell])
        cmax = max(coef.values())
        order = sorted(per, key=lambda q_: -abs(per[q_]))[:6]
        log('  IMPACT step %d: cell %d (V/med %.2f, %s): dL %.3e (the graph\'s %.3e) = %s by term (relative to the cell\'s own motion); by stencil point (of %d; signed share, relative motion |u_k - u_s| / max |u|, coefficient |C c| / V of the stencil\'s largest): %s' % (
            step, cell, float(m[cell])/Vm, 'slaved' if bool(FMx.slave[cell]) else 'master', dL, dLg, ', '.join('%s %+.0f %%' % (a_, 100*b_/dL) for a_, b_ in byterm.items()),
            len(per), '; '.join('%d (%s, %s, V/med %.2f) %+.0f %%, %.2f, %.2f' % (
                q_, 'itself' if q_ == cell else ('wall neighbour' if q_ in nb else 'through a fit'), 'slaved' if bool(FMx.slave[q_]) else 'master', float(m[q_])/Vm,
                100*per[q_]/dL, float((u[q_]-u[cell]).norm())/umax, coef.get(q_, 0.)/cmax) for q_ in order)))
        if watch in per:
            log('  IMPACT step %d: %d in cell %d: share %+.0f %%, coefficient %.2f of the largest (rank %d of %d)' % (
                step, watch, cell, 100*per[watch]/dL, coef[watch]/cmax, 1+sorted(coef.values(), reverse=True).index(coef[watch]), len(coef)))
    rows, depth = expand_rows(FMx)
    r168 = rows.get(watch)
    log('  IMPACT step %d: %d: V/med %.2f, %s%s; in the mode m|u|^2 %.1f %%, |u|^2 %.1f %%, |u| %.2f of the max' % (
        step, watch, float(m[watch])/Vm, 'slaved' if bool(FMx.slave[watch]) else 'master',
        (' (row %s, depth %d)' % (', '.join('%d %+.2f' % (j, w) for j, w in sorted(r168.items(), key=lambda p_: -abs(p_[1]))[:6]), depth[watch])) if r168 else '',
        100*float(wm[watch]), 100*float(wv[watch]), float(un[watch])/umax))
