"""The cut kept over several steps (11-06): the last cut x_m the reference of the steps after it, the same energy each step.
With H_m the cut's operator (G = H_m(x - x_m), the run's graph with xn left at x_m) and F_rel(x) = I + G, at each step's start
x_n that is not a cut the state is put on the reference (rebase):
  F0 = K_n = F_rel(x_n)^-1 F_n^e        (F = F_rel(x) K_n: F_n^e at x_n, linear in x through the step)
  C  = C_n = F_rel(x_n)^-T              (L = L_n + C_n : (G(x) - G(x_n)) = L_n + tr(F_rel^-1 dH): the log volume's increment)
  Ln = L_n - C_n : G(x_n)               (the stored constant so that L(x_n) = L_n)
  spring p = (I + Gbar_n)^-1 (x_j - x_i - dl_n), Gbar = (G_i + G_j) / 2   (d(x_n) = dl_n; the offset carried by the cells' F_rel)
  Rh = polar(F_n^e)                     (as update_ul sets it before the step's first evaluation: the yield factors and the
                                         springs' beta read at x_n with the step's own rotation; the patch test found it)
so F, L and the springs' remainders at x_n are the state's, the step's energy the same form (the same kernels); at a cut
(update_ul) G(x_n) = 0 and these reduce to the cut's own (C = I, F0 = F_e, p = d - dl). The step's increment for the passive
records (ep's rotation, H) is (I + G(x_n+1)) F_rel(x_n)^-1 (end_warp's Fri).
  rebase(W, x)        the rebase kernels; returns F_rel(x_n)^-1 per cell (for end_warp)
  strain_max(W)       the largest cell strain since the cut, |1/2 (F_rel^T F_rel - I)|_F over the cells, at the graph's
                      last point (x_n after the last step's end)
  omega_max(...)      Lanczos on the follow map's pencil (P^T K P, P^T M P), K by central differences of the gradient
  patch_test(...)     the same affine motion with a cut every step and with one cut: F, L, the springs' remainders"""
import math

import numpy as np
import torch
import warp as wp

f64 = torch.float64


@wp.func
def ddot(A: wp.mat33d, B: wp.mat33d):
    s = wp.float64(0.0)
    for r in range(3):
        for c in range(3):
            s = s+A[r, c]*B[r, c]
    return s


@wp.kernel(enable_backward=False)
def k_rebase_cell(G: wp.array(dtype=wp.mat33d), Fe: wp.array(dtype=wp.mat33d), F0: wp.array(dtype=wp.mat33d), C: wp.array(dtype=wp.mat33d),
                  Ln: wp.array(dtype=wp.float64), Fri: wp.array(dtype=wp.mat33d)):
    k = wp.tid()
    Ri = wp.inverse(wp.identity(n=3, dtype=wp.float64)+G[k])
    Fri[k] = Ri
    F0[k] = Ri@Fe[k]
    Ck = wp.transpose(Ri)
    C[k] = Ck
    Ln[k] = Ln[k]-ddot(Ck, G[k])


@wp.kernel(enable_backward=False)
def k_rebase_spring(x: wp.array(dtype=wp.vec3d), G: wp.array(dtype=wp.mat33d), si: wp.array(dtype=int), sj: wp.array(dtype=int), dl: wp.array(dtype=wp.vec3d),
                    sP: wp.array(dtype=wp.vec3d), sdn: wp.array(dtype=wp.vec3d)):
    e = wp.tid()
    i = si[e]
    j = sj[e]
    if i == j:
        return
    Gm = (G[i]+G[j])*wp.float64(0.5)
    sP[e] = wp.inverse(wp.identity(n=3, dtype=wp.float64)+Gm)@(x[j]-x[i]-dl[e])
    sdn[e] = dl[e]


class Lag:
    def __init__(self, W):
        self.Fri = torch.zeros(W.n, 3, 3, dtype=f64, device=W.dev)

    def rebase(self, W, x):
        t, s, m = W.tul, W.tsp, W._m_s
        M3 = lambda t_: wp.from_torch(t_.contiguous(), dtype=wp.mat33d)
        wp.launch(k_rebase_cell, W.n, inputs=[M3(W.tG), M3(W._Fe)], outputs=[W.sp['F0'], wp.from_torch(t['Q'], dtype=wp.mat33d), wp.from_torch(t['Ln'], dtype=wp.float64),
                                                                        wp.from_torch(self.Fri, dtype=wp.mat33d)], device=W.dev)
        from free_surface.closure_warp import k_F8
        wp.launch(k_F8, W.n, inputs=[M3(W._Fe)], outputs=[W.B['Fp'], W.B['Rp'], W.B['qp']], device=W.dev)
        s['Rh'].copy_(W.tRp)
        if m:
            wp.launch(k_rebase_spring, m, inputs=[wp.from_torch(x.contiguous(), dtype=wp.vec3d), M3(W.tG), W.sp['si'], W.sp['sj'], wp.from_torch(W.tdl.contiguous(), dtype=wp.vec3d)],
                      outputs=[W.sp['sP'], W.sp['sdn']], device=W.dev)
        return self.Fri


def strain_max(W):
    F = torch.eye(3, dtype=f64, device=W.tG.device)+W.tG
    E = .5*(F.transpose(1, 2)@F)-.5*torch.eye(3, dtype=f64, device=F.device)
    return float((E*E).sum((1, 2)).sqrt().max())


def strain_report(W, FM, x, log=print, tag='', top=8):
    """logging only, at a cut (before update_ul): the cells' strain since the last cut, its spread and the most strained cells
    (V0 / median, tiny, slaved, a cap at the last cut, the height of the site above the ground over the cell size)"""
    F = torch.eye(3, dtype=f64, device=W.tG.device)+W.tG
    E = .5*(F.transpose(1, 2)@F)-.5*torch.eye(3, dtype=f64, device=F.device)
    e = (E*E).sum((1, 2)).sqrt()
    q = torch.quantile(e, torch.tensor([.5, .9, .99, .999], dtype=f64, device=e.device)).tolist()
    V0m = W.V0.median(); h = float(V0m)**(1/3)
    tiny = W.V0 < W._ul_agg*V0m; sl = FM.slave if FM is not None else torch.zeros_like(tiny)
    cap = W._ulG.patch_A > 0
    ng = torch.tensor([W.ball['ng'][k_] for k_ in range(3)], dtype=f64, device=x.device); hgt = (x@ng-float(W.tball[2]))/h
    o = torch.argsort(e, descending=True)[:top]
    # the neighbours over the kept cut's connected walls: their area-weighted mean strain, and the cell's own over it
    t = W.tul; m = W._w[5]; wi, wj, wa = t['wi'][:m].long(), t['wj'][:m].long(), t['wa'][:m]
    num = torch.zeros_like(e).index_add(0, wi, wa*e[wj]).index_add(0, wj, wa*e[wi]); den = torch.zeros_like(e).index_add(0, wi, wa).index_add(0, wj, wa)
    en_ = num/den.clamp(min=1e-300)
    big = e > 1e-1
    log('LAG strain %s: the cells above 1e-1: their neighbours\' mean strain median %.2e (the cells\' own median %.2e); own / neighbours median %.1f, 10 %% %.1f' % (
        tag, float(en_[big].median()) if bool(big.any()) else 0., float(e[big].median()) if bool(big.any()) else 0.,
        float((e[big]/en_[big].clamp(min=1e-12)).median()) if bool(big.any()) else 0., float(torch.quantile(e[big]/en_[big].clamp(min=1e-12), .1)) if int(big.sum()) > 1 else 0.))
    log('LAG strain %s: median %.2e, 90 %% %.2e, 99 %% %.2e, 99.9 %% %.2e, max %.2e; cells above 1e-2 %d, 5e-2 %d, 1e-1 %d (tiny %d, slaved %d, caps %d among those above 1e-2)' % (
        tag, *q, float(e.max()), int((e > 1e-2).sum()), int((e > 5e-2).sum()), int((e > 1e-1).sum()), int((tiny & (e > 1e-2)).sum()), int((sl & (e > 1e-2)).sum()), int((cap & (e > 1e-2)).sum())))
    log('LAG   the most strained: %s' % '; '.join('%d: %.3f, neighbours %.3f (V0/med %.3f%s%s%s, height %.2f h)' % (int(c_), float(e[c_]), float(en_[c_]), float(W.V0[c_]/V0m), ', tiny' if bool(tiny[c_]) else '',
        ', slaved' if bool(sl[c_]) else '', ', cap' if bool(cap[c_]) else '', float(hgt[c_])) for c_ in o))


def describe(W, FM, x, c):
    """one cell for the logs: V0 / median, tiny, slaved, a cap at the cut, the site's height above the ground over the cell size"""
    V0m = W.V0.median(); hh = float(V0m)**(1/3)
    ng = torch.tensor([W.ball['ng'][k_] for k_ in range(3)], dtype=f64, device=x.device)
    tiny = bool(W.V0[c] < W._ul_agg*V0m); sl = bool(FM.slave[c]) if FM is not None else False
    cap = bool(W._ulG.patch_A[c] > 0) if W._ulG is not None else False
    return '%d (V0/med %.3f%s%s%s, height %.2f h)' % (c, float(W.V0[c]/V0m), ', tiny' if tiny else '', ', slaved' if sl else '', ', cap' if cap else '', float((x[c]@ng-float(W.tball[2]))/hh))


def omega_max(W, FM, mass, x, gr, k=100, h=1e-8, log=print, tag='', where=0):
    """the explicit step's limit on this state: the largest omega of the follow map's pencil (masters only)"""
    from free_surface.k1_follow_screen import gen_lanczos
    n = W.n; mk = FM.master[:, None].to(f64)

    def Kv(y):
        g1 = gr(x+h*y).clone(); g2 = gr(x-h*y).clone()
        return (g1-g2)/(2*h)
    Kf = lambda y_: FM.Pt(Kv(FM.P(y_)))
    Mf = lambda y_: FM.Pt(mass[:, None]*FM.P(y_))
    lam, S, Q, res = gen_lanczos(Kf, Mf, FM.force_to_accel, n, k, mk[:, 0])
    if where:                                                                       # (the top modes' sites: |P q|^2 m, the largest shares)
        for r_ in range(1, min(where, k)+1):
            y = sum(float(S[i, -r_])*Q[i] for i in range(len(Q))); u = FM.P(y); wv = mass*(u*u).sum(1); wv = wv/wv.sum()
            o = torch.argsort(wv, descending=True)[:5]
            log('LAG   mode %d omega %.4e: %s' % (r_, math.sqrt(max(float(lam[-r_]), 0.)), '; '.join('%.0f %% %s' % (100*float(wv[c_]), describe(W, FM, x, int(c_))) for c_ in o)))
    gr(x)                                                                           # (the graph's buffers back at x)
    om = np.sqrt(np.clip(lam, 0, None)); conv = res <= 1e-6*np.abs(lam).max()
    top = [float(om[-r]) for r in range(1, 6)]
    log('LAG omega %s: omega_max %.4e (converged %s), the next %s; the step limit 2 / omega %.3e s' % (tag, top[0], bool(conv[-1]), ', '.join('%.3e' % v_ for v_ in top[1:]), 2/top[0]))
    return top[0]


def patch_test(W, ST, roots, D, nsteps, log=print, trace=False):
    """the same affine motion x <- c + (I + D)(x - c) from the present state, (a) a cut every step, (b) one cut at the
    start and the rebase after; the state compared at the end (the run's state put back after)"""
    from free_surface.k1_snapshot import restore, take
    snap = take(roots)
    gr_ = lambda z: (W.tx.copy_(z), wp.capture_launch(W.graph))
    x0 = ST.tx.clone(); c = x0.mean(0, keepdim=True); Dm = torch.as_tensor(D, dtype=f64, device=x0.device)
    out = {}; tr = {}
    for mode in ('every', 'lagged'):
        restore(roots, snap)
        lag = Lag(W); x = x0.clone(); Fri = None; tr[mode] = []
        for k in range(nsteps):
            if mode == 'every' or k == 0:
                W.update_ul(x); Fri = None
            else:
                Fri = lag.rebase(W, x)
            gr_(x); W.reread(beta=True, keep_C=(Fri is not None))
            x1 = c+(x-c)@(torch.eye(3, dtype=f64, device=x.device)+Dm).T
            gr_(x1)
            if trace:                                                               # (per step: the springs by key, their p, beta, remainder at the end point, the cells' G)
                m_ = W._m_s
                tr[mode].append(dict(key=W.tkey.clone(), p=W.tsp['sP'][:m_].clone(), sb=W.tsp['sb'][:m_].clone(), sr=W.tsp['sr'][:m_].clone(), dl0=W.tdl.clone() if len(W.tdl) == m_ else None,
                                     lo=W.tlo.clone(), hi=W.thi.clone(), G=W.tG.clone(), x=x.clone(), x1=x1.clone()))
            W.end_ul(1., Fri=Fri)
            x = x1
        out[mode] = dict(F=W._Fe.clone(), L=W.tul['Ln'].clone(), dl=W.tdl.clone(), key=W.tkey.clone(), cap=(W._ulG.patch_A > 0).clone(), ep=W.ep.clone())
    restore(roots, snap)
    if trace:
        for k in range(nsteps):
            ta, tb = tr['every'][k], tr['lagged'][k]
            ca, cb = torch.isin(ta['key'], tb['key']), torch.isin(tb['key'], ta['key'])
            dsr = (ta['sr'][ca]-tb['sr'][cb]).norm(dim=1); dsb = (ta['sb'][ca]-tb['sb'][cb]).abs()
            w = int(dsr.argmax()); ia = torch.nonzero(ca).flatten()[w]; ib = torch.nonzero(cb).flatten()[w]
            lo, hi = int(ta['lo'][ia]), int(ta['hi'][ia])
            log('LAG trace step %d: matched springs %d; the end remainder differs by %.2e at most (spring %d-%d: every %s, lagged %s), beta by %.2e at most; '
                'that spring: beta %.4f / %.4f, p every %s lagged %s; G_lo-G_hi every %.2e lagged %.2e' % (
                    k, int(ca.sum()), float(dsr.max()), lo, hi, [round(v_, 9) for v_ in ta['sr'][ia].tolist()], [round(v_, 9) for v_ in tb['sr'][ib].tolist()], float(dsb.max()),
                    float(ta['sb'][ia]), float(tb['sb'][ib]), [round(v_, 7) for v_ in ta['p'][ia].tolist()], [round(v_, 7) for v_ in tb['p'][ib].tolist()],
                    float((ta['G'][lo]-ta['G'][hi]).abs().max()), float((tb['G'][lo]-tb['G'][hi]).abs().max())))
    a, b = out['every'], out['lagged']
    cap = a['cap']
    dF = (a['F']-b['F']).flatten(1).abs().max(1).values
    F_rel = float(((a['F']-torch.eye(3, dtype=f64, device=dF.device)).flatten(1).abs().max()))
    acc = np.linalg.norm(np.linalg.matrix_power(np.eye(3)+np.asarray(D), nsteps)-np.eye(3))
    common = torch.isin(a['key'], b['key']); cb = torch.isin(b['key'], a['key'])
    ddl = float((a['dl'][common]-b['dl'][cb]).abs().max()) if bool(common.any()) else 0.
    log('LAG patch test, %d steps of x <- c + (I + D)(x - c) (|(I + D)^%d - I| %.3e): F_e max |difference| %.2e (interior %.2e, cap cells %.2e; |F_e - I| max %.2e), '
        'L %.2e, ep %.2e; springs: %d / %d (every / lagged), %d in both, their remainders differ by %.2e (largest remainder %.2e); only with a cut each step %d (largest remainder %.2e), only lagged %d' % (
            nsteps, nsteps, acc, float(dF.max()), float(dF[~cap].max()), float(dF[cap].max()) if bool(cap.any()) else 0., F_rel, float((a['L']-b['L']).abs().max()),
            float((a['ep']-b['ep']).abs().max()), len(a['key']), len(b['key']), int(common.sum()), ddl, float(a['dl'].abs().max()), int((~common).sum()),
            float(a['dl'][~common].abs().max()) if bool((~common).any()) else 0., int((~cb).sum())))
    return out
