"""The Newton operator at a stalled point and the preconditioner's blocks against it (11-06, the expert's three stages;
logging only, nothing in the model or the solver changes). merged_drop.py --replay <snapshot> --replay-mode opcheck.

The stalled point: the step's first pass run from the snapshot for `at` Newton iterations (CG exits at the cap by then),
then everything frozen there (z, C, R, beta, the balls' step data, the cut). The product as TrustCG forms it:
    H v = cm v + (dE/dx(z + h v / |v|_max) - dE/dx(z)) |v|_max / h          (one-sided, h = CG.h)
Stage 1, the product's reliability: repeat, linearity H(u + v) vs H u + H v, symmetry u.Hv vs v.Hu, h from 1e-7 to 1e-11
one-sided and central, and whether a perturbation crosses a branch (a cell's return be < 1, a ball's contact, its slip).
Stage 2, three block preconditioners on the same point: the blocks in use (built at the step's start), ul_blocks rebuilt
at the point, and H's exact diagonal blocks (central differences on a distance coloring, checked by single sites), each
through the same k_mid_pre path where it applies; per block the same-point CG (plain PCG on the product, no region; and
TrustCG as Newton calls it), residual |r|, preconditioned r.M^-1 r and the model q(p) by products; where |r| sits.
Single-site columns also give the coupling across cells: sum_l != k |B_l^-1/2 H_lk B_k^-1/2| against |B_k^-1/2 H_kk B_k^-1/2|.
Stage 3, the spectrum of B^-1/2 H B^-1/2 (Lanczos, full reorthogonalisation) with each B, its extreme vectors' cells."""
import math

import numpy as np
import torch
import warp as wp

f64 = torch.float64


def run(W, ST, CG, tol, wref, log, at=40, h_lanczos=1e-8, k_lanczos=80, cg_max=600, seed=0, pcg_h=(1e-8, 1e-9)):
    from free_surface.closure_warp import k_mid_pre, k_mid_pre_ball, k_mid_inv
    wpt = lambda a_: wp.to_torch(a_, requires_grad=False)
    dev, n = W.dev, W.n
    A, bl = ST.a, W.ball
    gen = torch.Generator(device=dev).manual_seed(seed)

    # the stalled point
    ST.c_two_pass, ST.newton_log, ST.pre_force = False, True, False
    nmax0 = ST.newton_max; ST.newton_max = at
    e_, its, g2 = ST.step(tol, lambda g_: .5*tol)
    ST.newton_max = nmax0
    nl = ST.stats['nlog']
    log('OP stalled point: the first pass stopped after %d Newton iterations, %d products, |g| / weight %.3g (tol %.3g); the last 10: %s' % (
        its, e_, math.sqrt(g2)/wref, tol/wref, ' '.join('%d%s(%.2f)' % (q['K'], q['exit'], q['res_true']/q['g']) for q in nl[-10:])))
    z = ST.tz.clone(); gz = wpt(A['gz']).clone(); g0s = CG.t['g0'].clone(); cm = CG.t['cm'].clone()
    xw, gw = wpt(W.xw), wpt(W.xw.grad)
    bound = wpt(bl['bound']) if bl is not None else None
    body = wpt(bl['body']).long() if bl is not None else None

    def gE(x):
        torch.cuda.synchronize(); xw.copy_(x); W._run(); wp.synchronize()
        return gw.clone()

    def flags():
        be = W.tbe.clone()
        if bl is None:
            return be < 1-1e-12, None, None
        bd = wpt(W.B['bdelta']).clone(); ff = wpt(W.B['bff']).norm(dim=1); bo = bound
        return be < 1-1e-12, bd > 0, (bo > 0) & (ff >= bo*(1-1e-12))

    g0 = gE(z); f0 = flags()
    log('OP base gradient again vs the one Newton holds: max |diff| / max |g0| %.2e; yielding cells %d, balls touching %d, slipping %d' % (
        float((g0-g0s).abs().max()/g0s.abs().max()), int(f0[0].sum()), int(f0[1].sum()) if f0[1] is not None else -1, int(f0[2].sum()) if f0[2] is not None else -1))

    def Hv(v, h=CG.h, central=False, fl=None):
        s = float(v.abs().max())
        if central:
            gp = gE(z+(h/s)*v); fp = flags(); gm = gE(z-(h/s)*v); fm = flags()
            d = (gp-gm)*(s/(2*h))
            if fl is not None:
                fl.append(tuple(int((a_ != b_).sum()) if a_ is not None else 0 for a_, b_ in zip(fp, f0))+tuple(int((a_ != b_).sum()) if a_ is not None else 0 for a_, b_ in zip(fm, f0)))
        else:
            gp = gE(z+(h/s)*v)
            if fl is not None:
                fp = flags(); fl.append(tuple(int((a_ != b_).sum()) if a_ is not None else 0 for a_, b_ in zip(fp, f0)))
            d = (gp-g0)*(s/h)
        return cm[:, None]*v+d

    nrm = lambda t_: float(t_.norm())
    dot = lambda a_, b_: float((a_*b_).sum())

    # directions: random, the last CG step p, the preconditioned gradient
    P_use = wpt(A['P']).clone()
    Pi_use = torch.linalg.inv(P_use)
    rnd = [torch.randn(n, 3, dtype=f64, device=dev, generator=gen) for _ in range(3)]
    dirs = dict(rand1=rnd[0], rand2=rnd[1], rand3=rnd[2], cg_p=CG.t['p'].clone(), Mg=-torch.einsum('kab,kb->ka', Pi_use, gz))
    for k_, v_ in dirs.items():
        dirs[k_] = v_/v_.norm()

    # stage 1
    log('OP stage 1: the product as CG forms it (one-sided, h %.0e)' % CG.h)
    Hd = {}
    for k_, v_ in dirs.items():
        fl = []; a1 = Hv(v_, fl=fl); a2 = Hv(v_)
        Hd[k_] = a1
        log('  %s: |Hv| %.4e, v.Hv %.4e; repeat |diff| / |Hv| %.2e; branch changes at the perturbation (yield, contact, slip) %s' % (k_, nrm(a1), dot(v_, a1), nrm(a1-a2)/nrm(a1), fl[0]))
    for a_, b_ in (('rand1', 'rand2'), ('cg_p', 'rand1'), ('cg_p', 'Mg'), ('Mg', 'rand2')):
        u, v = dirs[a_], dirs[b_]
        Huv = Hv(u+v)
        log('  %s, %s: linearity |H(u+v) - Hu - Hv| / (|Hu| + |Hv|) %.2e; symmetry |u.Hv - v.Hu| / (|u.Hv| + |v.Hu|) %.2e (u.Hv %.4e, v.Hu %.4e)' % (
            a_, b_, nrm(Huv-Hd[a_]-Hd[b_])/(nrm(Hd[a_])+nrm(Hd[b_])), abs(dot(u, Hd[b_])-dot(v, Hd[a_]))/(abs(dot(u, Hd[b_]))+abs(dot(v, Hd[a_]))+1e-300), dot(u, Hd[b_]), dot(v, Hd[a_])))
    hs = (1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11)
    for k_ in ('rand1', 'cg_p', 'Mg'):
        v_ = dirs[k_]; rows = {}
        for h in hs:
            fl1, fl2 = [], []
            rows[h] = (Hv(v_, h=h, fl=fl1), Hv(v_, h=h, central=True, fl=fl2), fl1[0], fl2[0])
        ref = rows[1e-8][1]
        log('  %s against central 1e-8: %s' % (k_, '; '.join('h %.0e: one-sided %.2e, central %.2e, v.Hv %.5e / %.5e, branches %s / %s' % (
            h, nrm(r_[0]-ref)/nrm(ref), nrm(r_[1]-ref)/nrm(ref), dot(v_, r_[0]), dot(v_, r_[1]), r_[2], r_[3]) for h, r_ in rows.items())))
    for k_ in ('rand1', 'rand2'):
        u, v = dirs[k_], dirs['cg_p']
        a_, b_ = Hv(u, h=1e-8, central=True), Hv(v, h=1e-8, central=True)
        log('  central 1e-8, %s and cg_p: symmetry %.2e (u.Hv %.5e, v.Hu %.5e)' % (k_, abs(dot(u, b_)-dot(v, a_))/(abs(dot(u, b_))+abs(dot(v, a_))), dot(u, b_), dot(v, a_)))

    # stage 2: the blocks
    tK = ST.tK; K_use = tK.clone()

    def P_of(K):
        tK.copy_(K)
        gE(z)                                                                       # (R, beta at z for k_mid_pre)
        wp.launch(k_mid_pre, n, inputs=[W.B['R'], A['K'], ST.c1, A['cm']], outputs=[A['P']])
        wp.launch(k_mid_pre_ball, bl['k'], inputs=[bl['body'], bl['R'], W.B['bdelta'], bl['ng'], ST.e4, bl['kt'], bl['bound'], ST.muf], outputs=[A['P']])
        wp.synchronize()
        return wpt(A['P']).clone()
    P_cur = P_of(K_use)
    log('OP stage 2: the blocks in use rebuilt through k_mid_pre at z vs those Newton last applied: max |diff| / max |P| %.2e' % float((P_cur-P_use).abs().max()/P_use.abs().max()))
    gE(z); K_now = W.ul_blocks(); P_now = P_of(K_now)
    tK.copy_(K_use)

    # the exact diagonal blocks: the stencil from single sites, then a coloring
    i_, j_ = W._w[0].long(), W._w[1].long()
    ad = [torch.stack([i_, j_]), torch.stack([j_, i_])]
    ms = W._m_s
    if ms > 0:
        lo, hi = W.tlo[:ms].long(), W.thi[:ms].long(); ad += [torch.stack([lo, hi]), torch.stack([hi, lo])]
    S_, O_ = W.tul['S'][:2*W._ml].long(), W.tul['O'][:2*W._ml].long()
    ad += [torch.stack([S_, O_]), torch.stack([O_, S_])]
    E = torch.cat(ad, 1).cpu().numpy()
    import scipy.sparse as sp
    A1 = sp.csr_matrix((np.ones(E.shape[1]), (E[0], E[1])), shape=(n, n)); A1.data[:] = 1.; A1 = A1+sp.eye(n, format='csr'); A1.data[:] = 1.

    def hops(k):
        d = np.full(n, -1); d[k] = 0; fr = [k]; h = 0
        while fr and h < 8:
            h += 1; nb = np.unique(A1[fr].indices); nb = nb[d[nb] < 0]; d[nb] = h; fr = list(nb)
        return d
    gnorm = gz.norm(dim=1)
    probe = list(dict.fromkeys(torch.argsort(gnorm, descending=True)[:6].tolist()+torch.randint(0, n, (6,), device=dev, generator=gen).tolist()))
    hc = 1e-8
    cols = {}
    reach = 0
    for k in probe:
        Hk = torch.zeros(n, 3, 3, dtype=f64, device=dev)
        for a in range(3):
            e = torch.zeros(n, 3, dtype=f64, device=dev); e[k, a] = 1.
            Hk[:, :, a] = (gE(z+hc*e)-gE(z-hc*e))/(2*hc)
        Hk[k] += cm[k]*torch.eye(3, dtype=f64, device=dev)
        cols[k] = Hk
        nz = (Hk.flatten(1).abs().max(1).values > 1e-6*float(Hk[k].abs().max())).cpu().numpy()   # (1e-6: the difference's roundoff noise reaches everywhere below ~1e-8)
        d = hops(k); far = int((nz & (d < 0)).sum()); reach = max(reach, int(d[nz & (d >= 0)].max()))
        if far:
            log('OP site %d: %d entries above 1e-6 relative beyond 8 hops (noise, or a long-range term): the coloring ignores them' % (k, far))
    reach = min(reach, 6)
    log('OP the stencil from %d single sites: H_lk nonzero up to %d hops (walls, springs, fit rows; capped at 6)' % (len(probe), reach))
    R = max(reach, 1)
    AR = A1.copy()
    for _ in range(R-1):
        AR = AR@A1; AR.data[:] = 1.
    AR = AR.tocsr(); color = -np.ones(n, dtype=int)
    for k in np.argsort(-np.diff(AR.indptr)):
        used = set(color[AR.indices[AR.indptr[k]:AR.indptr[k+1]]].tolist()); c = 0
        while c in used:
            c += 1
        color[k] = c
    nc = int(color.max())+1
    col_t = torch.as_tensor(color, device=dev)
    Hdiag = torch.zeros(n, 3, 3, dtype=f64, device=dev)
    for c in range(nc):
        m = (col_t == c)
        for a in range(3):
            e = torch.zeros(n, 3, dtype=f64, device=dev); e[m, a] = 1.
            gd = (gE(z+hc*e)-gE(z-hc*e))/(2*hc)
            Hdiag[m, :, a] = gd[m]
    Hdiag += cm[:, None, None]*torch.eye(3, dtype=f64, device=dev)
    chk = max(float((Hdiag[k]-cols[k][k]).abs().max()/cols[k][k].abs().max()) for k in probe)
    Hs = .5*(Hdiag+Hdiag.transpose(1, 2)); ev, U = torch.linalg.eigh(Hs)
    neg = int((ev[:, 0] <= 0).sum())
    floor = cm[:, None]*1.
    P_ex = U@torch.diag_embed(torch.maximum(ev, floor))@U.transpose(1, 2)
    log('OP exact diagonal blocks: %d colors at distance %d (%d gradient pairs); against single sites max rel diff %.2e; asymmetry max %.2e; blocks with an eigenvalue <= 0: %d (floored at cm)' % (
        nc, R, 3*nc, chk, float(((Hdiag-Hdiag.transpose(1, 2)).flatten(1).abs().max(1).values/Hdiag.flatten(1).abs().max(1).values).max()), neg))

    def bmh(P):
        Ps = .5*(P+P.transpose(1, 2)); e_, V_ = torch.linalg.eigh(Ps)
        return V_@torch.diag_embed(e_.clamp(min=1e-300).rsqrt())@V_.transpose(1, 2), int((e_[:, 0] <= 0).sum())
    blocks = dict(in_use=P_cur, rebuilt_here=P_now, exact_diag=P_ex)
    for nm_, P in blocks.items():
        Bm, bad = bmh(P)
        for k in probe[:6]:
            Hk = cols[k]
            T = torch.einsum('lab,lbc,cd->lad', Bm, Hk, Bm[k])
            off = T.norm(dim=(1, 2)); dg = float(off[k]); off[k] = 0.
            if k == probe[0]:
                pass
        # coupling over the probe sites, summarised
        rat = []
        for k in probe:
            T = torch.einsum('lab,lbc,cd->lad', Bm, cols[k], Bm[k]); off = T.norm(dim=(1, 2)); dg = float(off[k]); off[k] = 0.
            rat.append((k, float(off.sum())/dg, dg, int((off > .1*dg).sum())))
        log('OP %s: blocks not positive %d; ratio |P_k^-1/2 H_kk P_k^-1/2| per probe site: %s; coupling sum_l |P_l^-1/2 H_lk P_k^-1/2| / diagonal: %s' % (
            nm_, bad, ' '.join('%d:%.3g' % (k, d_) for k, _, d_, _ in rat), ' '.join('%d:%.2f(%d>0.1)' % (k, r_, c_) for k, r_, _, c_ in rat)))

    # same-point CG: plain PCG on the product (no region), one right-hand side, one stop (recursive Euclidean |r| <= eta |g|, as Newton's CG), the
    # true residual by the checked central product (h 1e-8) at checkpoints and at the stop; gradient evaluations and time without the checks
    gn = nrm(gz)
    Ht = lambda v_: Hv(v_, h=1e-8, central=True)
    import time as _t
    for hp_ in pcg_h:
        for nm_, P in blocks.items():
            Pinv = torch.linalg.inv(P)
            Mi = lambda r_: torch.einsum('kab,kb->ka', Pinv, r_)
            p = torch.zeros_like(gz); r = -gz.clone(); y = Mi(r); d = y.clone(); rz = dot(r, y); rz0 = rz; Hp = torch.zeros_like(gz)
            hist, k, why, tcg, tchk = [], 0, 'cap %d' % cg_max, 0., 0.
            torch.cuda.synchronize(); t0_ = _t.perf_counter()
            while k < cg_max:
                hd = Hv(d, h=hp_); k += 1
                dhd = dot(d, hd)
                if dhd <= 0:
                    why = 'negative curvature'; break
                al = rz/dhd; p += al*d; Hp += al*hd; r -= al*hd; y = Mi(r); rz1 = dot(r, y)
                done = nrm(r) <= ST.eta*gn
                if k in (30, 100, 200, 300, 600) or done:
                    torch.cuda.synchronize(); t1_ = _t.perf_counter()
                    rt = gz+Ht(p)
                    hist.append('%d: |r| %.3g (true %.3g), sqrt(r.M^-1r / g.M^-1g) %.3g, q %.5g (true %.5g)' % (k, nrm(r)/gn, nrm(rt)/gn, math.sqrt(max(rz1, 0.)/rz0), dot(gz, p)+.5*dot(p, Hp), dot(gz, p)+.5*dot(p, Ht(p))))
                    torch.cuda.synchronize(); tchk += _t.perf_counter()-t1_
                if done:
                    why = 'stop (|r| <= %.2g |g|)' % ST.eta; break
                d = y+(rz1/rz)*d; rz = rz1
            torch.cuda.synchronize(); tcg = _t.perf_counter()-t0_-tchk
            rk = r.norm(dim=1); top = torch.argsort(rk, descending=True)[:6]; sh = float((rk[top]**2).sum()/(rk**2).sum())
            Vm = W.V0.median()
            log('OP PCG h %.0e one-sided, %s blocks: %s after %d products (%d gradient evaluations, %.2f s without the checks); %s; |r|^2 in its 6 largest sites %.0f %%: %s' % (
                hp_, nm_, why, k, k, tcg, ' | '.join(hist), 100*sh, ', '.join('%d (V0/med %.2f, %s%s)' % (c_, float(W.V0[c_]/Vm), 'yield' if bool(f0[0][c_]) else 'elastic',
                                                                                             ', touching' if f0[1] is not None and bool((f0[1] & (body == c_)).any()) else '') for c_ in top.tolist())))

    # the same point through TrustCG as Newton calls it
    delta = ST.delta_start if ST.delta_start else -1.
    stop = max(.5*tol, ST.eta*gn)
    for nm_, P in blocks.items():
        wpt(A['P']).copy_(P); wp.launch(k_mid_inv, n, inputs=[A['P']], outputs=[A['Pinv']]); wp.synchronize()
        gE(z); torch.cuda.synchronize()
        r_ = CG.solve(A['gz'], delta, stop, ST.cg_cap)
        rt = gz+Hv(CG.t['p'].clone())
        log('OP TrustCG (its region p.Bp <= delta^2 changes with B: not a preconditioner comparison) with %s blocks at delta %.3g: %d products, exit %s, |r| / |g| %.3g (true %.3g), model decrease %.4g' % (
            nm_, delta, r_['K'], 'negative' if r_['NEG'] else ('edge' if r_['EDGE'] else ('cap' if r_['K'] >= ST.cg_cap else 'tolerance')), math.sqrt(max(r_['RR'], 0.))/gn, nrm(rt)/gn, -(r_['GP']+.5*r_['PHP'])))

    # stage 3: the preconditioned spectrum (central products at h_lanczos)
    for nm_, P in blocks.items():
        Bm, _ = bmh(P)
        op = lambda y_: torch.einsum('kab,kb->ka', Bm, Hv(torch.einsum('kab,kb->ka', Bm, y_), h=h_lanczos, central=True))
        Q = torch.zeros(k_lanczos+1, n, 3, dtype=f64, device=dev)
        q = torch.randn(n, 3, dtype=f64, device=dev, generator=gen); Q[0] = q/q.norm(); al, be = [], []
        for it in range(k_lanczos):
            w = op(Q[it]); a = dot(w, Q[it]); al.append(a)
            for _ in range(2):
                c_ = torch.einsum('kab,ab->k', Q[:it+1], w); w = w-torch.einsum('k,kab->ab', c_, Q[:it+1])
            b = nrm(w); be.append(b)
            if b < 1e-14:
                break
            Q[it+1] = w/b
        m = len(al); T = np.diag(al)+np.diag(be[:m-1], 1)+np.diag(be[:m-1], -1); lam, S = np.linalg.eigh(T)
        loc = []
        for idx in (0, 1, m-2, m-1):
            y_ = torch.einsum('k,kab->ab', torch.as_tensor(S[:, idx], dtype=f64, device=dev), Q[:m]); x_ = torch.einsum('kab,kb->ka', Bm, y_)
            wv = (x_*x_).sum(1); t3 = torch.argsort(wv, descending=True)[:3]
            ry = op(y_)-lam[idx]*y_                                                  # (the Ritz pair's residual, by one more product; the estimate beta_m |s_m| beside)
            loc.append('%.4g (Ritz residual |op y - lam y| %.2e, estimate %.2e) on %s (%.0f %%)' % (lam[idx], nrm(ry), be[m-1]*abs(S[m-1, idx]), t3.tolist(), 100*float(wv[t3].sum()/wv.sum())))
        log('OP spectrum of B^-1/2 H B^-1/2 with %s blocks (%d Lanczos steps, central h %.0e): smallest %s, largest %s; ratio %.3g; negative Ritz values %d; extreme vectors: %s' % (
            nm_, m, h_lanczos, np.round(lam[:4], 5).tolist(), np.round(lam[-4:], 3).tolist(), lam[-1]/max(lam[0], 1e-300) if lam[0] > 0 else float('inf'), int((lam < 0).sum()), '; '.join(loc)))
    tK.copy_(K_use)


def patch(W, ST, CG, tol, wref, log, at=40, cg_max=600, seed=0, ncell=6, hc=1e-8, extra_cap=18):
    """11-06, the expert's A / B / C at one frozen stalled point (solver help only: no material, no motion constraint):
      A  ul_blocks rebuilt at the point (the cheap blocks)
      B  A with the `ncell` cells where A's PCG residual concentrates replaced by their exact 3x3 diagonal blocks (single-site central columns)
      C  those cells as one joint 3n x 3n block with the measured cross terms (symmetrised, checked), the rest A: non-overlapping block Jacobi
      D  (the follow-up the expert named if C gains little) C's group extended by the neighbours with the strongest measured coupling (their own
         columns measured too), capped at extra_cap more cells
    One right-hand side -g, one stop (recursive |r| <= eta |g|), the production product (one-sided h = CG.h); the true residual by central
    differences at hc, independently. Per arm: gradient evaluations (construction + CG), build / factor time, iteration time."""
    import time as _t
    from free_surface.closure_warp import k_mid_pre, k_mid_pre_ball
    wpt = lambda a_: wp.to_torch(a_, requires_grad=False)
    dev, n = W.dev, W.n
    A, bl = ST.a, W.ball
    I3 = torch.eye(3, dtype=f64, device=dev)

    ST.c_two_pass, ST.newton_log, ST.pre_force = False, True, False
    nmax0 = ST.newton_max; ST.newton_max = at
    e_, its, g2 = ST.step(tol, lambda g_: .5*tol)
    ST.newton_max = nmax0
    nl = ST.stats['nlog']
    log('PATCH stalled point: the first pass stopped after %d Newton iterations, %d products, |g| / weight %.3g (tol %.3g); the last 6: %s' % (
        its, e_, math.sqrt(g2)/wref, tol/wref, ' '.join('%d%s(%.2f)' % (q['K'], q['exit'], q['res_true']/q['g']) for q in nl[-6:])))
    z = ST.tz.clone(); gz = wpt(A['gz']).clone(); cm = CG.t['cm'].clone()
    xw, gw = wpt(W.xw), wpt(W.xw.grad)
    body = wpt(bl['body']).long() if bl is not None else None
    nev = [0]

    def gE(x):
        torch.cuda.synchronize(); xw.copy_(x); W._run(); wp.synchronize(); nev[0] += 1
        return gw.clone()
    g0 = gE(z); yld = W.tbe.clone() < 1-1e-12
    touch = torch.zeros(n, dtype=torch.bool, device=dev)
    if bl is not None:
        touch[body[wpt(W.B['bdelta']) > 0]] = True

    def Hv(v, h=CG.h, central=False):
        s = float(v.abs().max())
        if central:
            return cm[:, None]*v+(gE(z+(h/s)*v)-gE(z-(h/s)*v))*(s/(2*h))
        return cm[:, None]*v+(gE(z+(h/s)*v)-g0)*(s/h)
    Ht = lambda v_: Hv(v_, h=hc, central=True)
    nrm = lambda t_: float(t_.norm()); dot = lambda a_, b_: float((a_*b_).sum())
    gn = nrm(gz); Vm = W.V0.median()
    desc = lambda c_: '%d (V0/med %.2f%s%s)' % (c_, float(W.V0[c_]/Vm), ', yield' if bool(yld[c_]) else '', ', touching' if bool(touch[c_]) else '')

    # A: the cheap blocks at the point, through k_mid_pre (as Newton applies them)
    torch.cuda.synchronize(); t0 = _t.perf_counter()
    gE(z); ST.tK.copy_(W.ul_blocks())
    wp.launch(k_mid_pre, n, inputs=[W.B['R'], A['K'], ST.c1, A['cm']], outputs=[A['P']])
    wp.launch(k_mid_pre_ball, bl['k'], inputs=[bl['body'], bl['R'], W.B['bdelta'], bl['ng'], ST.e4, bl['kt'], bl['bound'], ST.muf], outputs=[A['P']])
    wp.synchronize(); PA = wpt(A['P']).clone(); PAi = torch.linalg.inv(PA); torch.cuda.synchronize(); tA = _t.perf_counter()-t0
    ev0 = nev[0]

    def pcg(Minv, label, build_evals, t_build):
        p = torch.zeros_like(gz); r = -gz.clone(); y = Minv(r); d = y.clone(); rz = dot(r, y); Hp = torch.zeros_like(gz)
        hist, k, why, tchk = [], 0, 'cap %d' % cg_max, 0.
        torch.cuda.synchronize(); t0_ = _t.perf_counter(); e0_ = nev[0]
        while k < cg_max:
            hd = Hv(d); k += 1
            dhd = dot(d, hd)
            if dhd <= 0:
                why = 'negative curvature'; break
            al = rz/dhd; p += al*d; Hp += al*hd; r -= al*hd; y = Minv(r); rz1 = dot(r, y)
            done = nrm(r) <= ST.eta*gn
            if k in (100, 200, 300, 450) or done:
                torch.cuda.synchronize(); t1_ = _t.perf_counter(); e1_ = nev[0]
                rt = gz+Ht(p); hist.append('%d: |r| %.3g (true %.3g), q %.5g' % (k, nrm(r)/gn, nrm(rt)/gn, dot(gz, p)+.5*dot(p, Hp)))
                torch.cuda.synchronize(); tchk += _t.perf_counter()-t1_; nev[0] = e1_                   # (the checks' evaluations and time not counted)
            if done:
                why = 'stop |r| <= %.2g |g|' % ST.eta; break
            d = y+(rz1/rz)*d; rz = rz1
        torch.cuda.synchronize(); tit = _t.perf_counter()-t0_-tchk; cg_evals = nev[0]-e0_
        rt = gz+Ht(p); nev[0] -= 2
        rk = r.norm(dim=1); top = torch.argsort(rk, descending=True)[:ncell]; sh = float((rk[top]**2).sum()/(rk**2).sum())
        log('PATCH %s: %s after %d products; true residual at the end %.4g |g|; gradient evaluations: construction %d + CG %d = %d; time: build + factor %.3f s, iterations %.2f s, total %.2f s; %s; |r|^2 in its %d largest sites %.0f %%: %s' % (
            label, why, k, nrm(rt)/gn, build_evals, cg_evals, build_evals+cg_evals, t_build, tit, t_build+tit, ' | '.join(hist), ncell, 100*sh, ', '.join(desc(c_) for c_ in top.tolist())))
        return top.tolist(), rk
    MA = lambda r_: torch.einsum('kab,kb->ka', PAi, r_)
    S, rkA = pcg(MA, 'A (cheap blocks rebuilt at the point)', 0, tA)

    # the single-site columns of the group (central hc): H_lk for all l, the inertia on the diagonal
    cols = {}

    def column(k):
        Hk = torch.zeros(n, 3, 3, dtype=f64, device=dev)
        for a in range(3):
            e = torch.zeros(n, 3, dtype=f64, device=dev); e[k, a] = 1.
            Hk[:, :, a] = (gE(z+hc*e)-gE(z-hc*e))/(2*hc)
        Hk[k] += cm[k]*I3
        cols[k] = Hk
    torch.cuda.synchronize(); t0 = _t.perf_counter(); e0 = nev[0]
    for k in S:
        column(k)
    torch.cuda.synchronize(); tcol = _t.perf_counter()-t0; ecol = nev[0]-e0

    def joint(group):
        m = len(group); J = torch.zeros(m, 3, m, 3, dtype=f64, device=dev)
        for b, kb in enumerate(group):
            for a, ka in enumerate(group):
                J[a, :, b, :] = cols[kb][ka]
        J = J.reshape(3*m, 3*m); asym = float((J-J.T).abs().max()/J.abs().max()); J = .5*(J+J.T)
        ev, U = torch.linalg.eigh(J); floor = float(cm[group].min())
        Jf = U@torch.diag(torch.maximum(ev, torch.full_like(ev, floor)))@U.T
        return Jf, asym, ev
    # B: A with the group's exact diagonal blocks
    torch.cuda.synchronize(); t0 = _t.perf_counter()
    PB = PA.clone(); bad = 0
    for k in S:
        D = .5*(cols[k][k]+cols[k][k].T); ev, U = torch.linalg.eigh(D); bad += int(ev[0] <= 0)
        PB[k] = U@torch.diag(torch.maximum(ev, torch.full_like(ev, float(cm[k]))))@U.T
    PBi = torch.linalg.inv(PB); torch.cuda.synchronize(); tB = tcol+_t.perf_counter()-t0
    log('PATCH group %s: columns %d evaluations, %.2f s; exact diagonal blocks not positive: %d; A vs exact per cell |P_A^-1/2 H_kk P_A^-1/2| (ideal 1.73): %s' % (
        S, ecol, tcol, bad, ' '.join('%d:%.2f' % (k, float(torch.linalg.norm(torch.linalg.inv(torch.linalg.cholesky(PA[k]))@cols[k][k]@torch.linalg.inv(torch.linalg.cholesky(PA[k])).T))) for k in S)))
    pcg(lambda r_: torch.einsum('kab,kb->ka', PBi, r_), 'B (A, the group\'s exact 3x3 blocks)', ecol, tB)

    def run_joint(group, label, extra_evals, t_extra):
        torch.cuda.synchronize(); t0_ = _t.perf_counter()
        Jf, asym, ev = joint(group); Ji = torch.linalg.inv(Jf); gi = torch.as_tensor(group, device=dev)
        PCi = PAi.clone(); PCi[gi] = 0.
        torch.cuda.synchronize(); tC = t_extra+_t.perf_counter()-t0_
        # the coupling inside the group vs its diagonal, in the exact blocks' own scaling
        Dh = torch.block_diag(*[torch.linalg.inv(torch.linalg.cholesky(.5*(cols[k][k]+cols[k][k].T))) for k in group])
        T = Dh@Jf@Dh.T; m = len(group); off = T.clone()
        for a in range(m):
            off[3*a:3*a+3, 3*a:3*a+3] = 0.
        log('PATCH %s joint block %d x %d: asymmetry %.2e, eigenvalues %.3g .. %.3g (floored below %.3g: %d), |off-diagonal| / |diagonal| in the exact blocks\' scaling %.3f' % (
            label, 3*m, 3*m, asym, float(ev[0]), float(ev[-1]), float(cm[group].min()), int((ev < cm[group].min()).sum()), float(off.norm()/(T-off).norm())))

        def MC(r_):
            y_ = torch.einsum('kab,kb->ka', PCi, r_); y_[gi] = (Ji@r_[gi].reshape(-1)).reshape(m, 3); return y_
        pcg(MC, label, ecol+extra_evals, tC)
    run_joint(S, 'C (the group as one joint block, the rest A)', 0, tcol)

    # D: the group extended by the neighbours with the strongest measured coupling
    cand = {}
    for k in S:
        Lk = torch.linalg.inv(torch.linalg.cholesky(.5*(cols[k][k]+cols[k][k].T)))
        Hk = cols[k]; PAh = torch.linalg.inv(torch.linalg.cholesky(PA)).transpose(1, 2)                       # (P_A^-1/2 per site, upper)
        T = torch.einsum('lab,lbc,cd->lad', PAh.transpose(1, 2), Hk, Lk.T); w = T.norm(dim=(1, 2)); w[k] = 0.
        for l in torch.argsort(w, descending=True)[:8].tolist():
            if l not in S and float(w[l]) > .1*float(torch.linalg.norm(Lk@Hk[k]@Lk.T)):
                cand[l] = max(cand.get(l, 0.), float(w[l]))
    ext = sorted(cand, key=lambda l_: -cand[l_])[:extra_cap]
    log('PATCH D: %d neighbours with coupling > 0.1 of the diagonal found from the measured columns, taking %d: %s' % (len(cand), len(ext), ', '.join('%s (%.2f)' % (desc(l_), cand[l_]) for l_ in ext)))
    if ext:
        torch.cuda.synchronize(); t0 = _t.perf_counter(); e0 = nev[0]
        for l in ext:
            column(l)
        torch.cuda.synchronize(); run_joint(S+ext, 'D (C\'s group + %d coupled neighbours)' % len(ext), nev[0]-e0, tcol+_t.perf_counter()-t0)


def patch_split(W, ST, CG, tol, wref, log, at=40, cg_max=600, ncell=6, hc=1e-8):
    """11-06, the expert's check of the product on C's group, at the same frozen point: (1) the perturbation actually written, fl(z + h v / |v|_inf) - z,
    on the group's components for C's late search directions and its final p; (2) the split product H v ~ H_:,S v_S (the measured full columns) +
    H_FD(v_S-bar), the columns checked for step consistency (hc / 10, hc x 10); (3) C's same-point PCG again (same rhs, preconditioner, stop) with the
    split product; residuals by both independent schemes (central FD on the whole p; the split one). A is run only to select the cells as before."""
    import time as _t
    from free_surface.closure_warp import k_mid_pre, k_mid_pre_ball
    wpt = lambda a_: wp.to_torch(a_, requires_grad=False)
    dev, n = W.dev, W.n
    A, bl = ST.a, W.ball
    I3 = torch.eye(3, dtype=f64, device=dev)
    ST.c_two_pass, ST.newton_log, ST.pre_force = False, True, False
    nmax0 = ST.newton_max; ST.newton_max = at
    e_, its, g2 = ST.step(tol, lambda g_: .5*tol)
    ST.newton_max = nmax0
    log('SPLIT stalled point: the first pass stopped after %d Newton iterations, %d products, |g| / weight %.3g' % (its, e_, math.sqrt(g2)/wref))
    z = ST.tz.clone(); gz = wpt(A['gz']).clone(); cm = CG.t['cm'].clone()
    xw, gw = wpt(W.xw), wpt(W.xw.grad)
    nev = [0]

    def gE(x):
        torch.cuda.synchronize(); xw.copy_(x); W._run(); wp.synchronize(); nev[0] += 1
        return gw.clone()
    g0 = gE(z)
    nrm = lambda t_: float(t_.norm()); dot = lambda a_, b_: float((a_*b_).sum()); gn = nrm(gz)

    def Hfd(v, h=CG.h, central=False):
        s = float(v.abs().max())
        if s == 0.:
            return torch.zeros_like(v)
        if central:
            return cm[:, None]*v+(gE(z+(h/s)*v)-gE(z-(h/s)*v))*(s/(2*h))
        return cm[:, None]*v+(gE(z+(h/s)*v)-g0)*(s/h)
    gE(z); ST.tK.copy_(W.ul_blocks())
    wp.launch(k_mid_pre, n, inputs=[W.B['R'], A['K'], ST.c1, A['cm']], outputs=[A['P']])
    wp.launch(k_mid_pre_ball, bl['k'], inputs=[bl['body'], bl['R'], W.B['bdelta'], bl['ng'], ST.e4, bl['kt'], bl['bound'], ST.muf], outputs=[A['P']])
    wp.synchronize(); PA = wpt(A['P']).clone(); PAi = torch.linalg.inv(PA)

    def pcg(Minv, Hv, label, keep_last=0, truths=()):
        p = torch.zeros_like(gz); r = -gz.clone(); y = Minv(r); d = y.clone(); rz = dot(r, y); Hp = torch.zeros_like(gz)
        k, why, hist, last = 0, 'cap %d' % cg_max, [], []
        torch.cuda.synchronize(); t0_ = _t.perf_counter(); e0_ = nev[0]
        while k < cg_max:
            hd = Hv(d); k += 1
            dhd = dot(d, hd)
            if dhd <= 0:
                why = 'negative curvature'; break
            if keep_last:
                last.append(d.clone()); last = last[-keep_last:]
            al = rz/dhd; p += al*d; Hp += al*hd; r -= al*hd; y = Minv(r); rz1 = dot(r, y)
            done = nrm(r) <= ST.eta*gn
            if k in (50, 100, 200, 300, 450) or done:
                e1_ = nev[0]; torch.cuda.synchronize(); t1_ = _t.perf_counter()
                hist.append('%d: |r| %.3g, %s' % (k, nrm(r)/gn, ', '.join('%s %.3g' % (nm_, nrm(gz+T_(p))/gn) for nm_, T_ in truths)))
                nev[0] = e1_; torch.cuda.synchronize(); t0_ += _t.perf_counter()-t1_
            if done:
                why = 'stop |r| <= %.2g |g|' % ST.eta; break
            d = y+(rz1/rz)*d; rz = rz1
        torch.cuda.synchronize(); tit = _t.perf_counter()-t0_
        rk = r.norm(dim=1); top = torch.argsort(rk, descending=True)[:ncell]
        log('SPLIT %s: %s after %d products (%d gradient evaluations, %.2f s); %s; |r|^2 in its %d largest sites %.0f %%: %s' % (
            label, why, k, nev[0]-e0_, tit, ' | '.join(hist), ncell, 100*float((rk[top]**2).sum()/(rk**2).sum()), top.tolist()))
        return p, last, top.tolist(), rk
    MA = lambda r_: torch.einsum('kab,kb->ka', PAi, r_)
    _, _, S, rkA = pcg(MA, Hfd, 'A (cells selected from its residual)')
    gi = torch.as_tensor(S, device=dev)

    # the group's full columns at three steps
    def columns(h):
        C = {}
        for k in S:
            Hk = torch.zeros(n, 3, 3, dtype=f64, device=dev)
            for a in range(3):
                e = torch.zeros(n, 3, dtype=f64, device=dev); e[k, a] = 1.
                Hk[:, :, a] = (gE(z+h*e)-gE(z-h*e))/(2*h)
            Hk[k] += cm[k]*I3; C[k] = Hk
        return C
    cols = columns(hc); c_lo = columns(hc/10); c_hi = columns(hc*10)
    log('SPLIT columns of %s at h %.0e: against h / 10 max rel diff of the full column %s, of its diagonal block %s; against h x 10: %s / %s' % (
        S, hc, ['%.1e' % float((cols[k]-c_lo[k]).norm()/cols[k].norm()) for k in S], ['%.1e' % float((cols[k][k]-c_lo[k][k]).norm()/cols[k][k].norm()) for k in S],
        ['%.1e' % float((cols[k]-c_hi[k]).norm()/cols[k].norm()) for k in S], ['%.1e' % float((cols[k][k]-c_hi[k][k]).norm()/cols[k][k].norm()) for k in S]))
    Hcols = lambda v_: sum(torch.einsum('lab,b->la', cols[k], v_[k]) for k in S)

    def Hsplit(v, h=CG.h, central=False):
        vr = v.clone(); vr[gi] = 0.
        return Hcols(v)+Hfd(vr, h=h, central=central)
    Ht_old = lambda p_: Hfd(p_, h=hc, central=True)
    Ht_split = lambda p_: Hsplit(p_, h=hc, central=True)

    # C's preconditioner as before
    m = len(S); J = torch.zeros(m, 3, m, 3, dtype=f64, device=dev)
    for b, kb in enumerate(S):
        for a, ka in enumerate(S):
            J[a, :, b, :] = cols[kb][ka]
    J = J.reshape(3*m, 3*m); J = .5*(J+J.T); ev, U = torch.linalg.eigh(J); J = U@torch.diag(torch.maximum(ev, torch.full_like(ev, float(cm[gi].min()))))@U.T; Ji = torch.linalg.inv(J)
    PCi = PAi.clone(); PCi[gi] = 0.

    def MC(r_):
        y_ = torch.einsum('kab,kb->ka', PCi, r_); y_[gi] = (Ji@r_[gi].reshape(-1)).reshape(m, 3); return y_
    log('SPLIT C joint block eigenvalues %.3g .. %.3g' % (float(ev[0]), float(ev[-1])))
    truths = (('independent: central FD on p', Ht_old), ('independent: columns on p_S + central FD on the rest', Ht_split))
    p1, last, top1, _ = pcg(MC, Hfd, 'C with the production product (one-sided h %.0e)' % CG.h, keep_last=4, truths=truths)

    # (1) the perturbation actually written, on the group's components
    def written(v, name):
        s = float(v.abs().max()); dt_ = (CG.h/s)*v; da = (z+dt_)-z
        on = dt_[gi]; wr = da[gi]
        zero = int(((wr == 0) & (on != 0)).sum()); rel = float((wr-on).norm()/on.norm()) if float(on.norm()) > 0 else float('nan')
        relc = [float((wr[i_]-on[i_]).norm()/on[i_].norm()) if float(on[i_].norm()) > 0 else float('nan') for i_ in range(m)]
        allrel = float((da-dt_).norm()/dt_.norm())
        log('SPLIT written perturbation for %s: |v_S| / |v| %.2e, target |delta_S|_max %.2e m; group components unchanged %d of %d, rel error on the group %.2e (per cell %s), on the whole vector %.2e' % (
            name, float(v[gi].norm()/v.norm()), float(on.abs().max()), zero, 3*m, rel, ' '.join('%d:%.1e' % (S[i_], relc[i_]) for i_ in range(m)), allrel))
    for i_, d_ in enumerate(last):
        written(d_, 'search direction %d before C\'s stop' % (len(last)-i_))
    written(p1, 'C\'s final p')
    r1 = gz+Ht_split(p1); rk1 = r1.norm(dim=1)
    log('SPLIT C (production product) final p: residual by columns + central rest %.4g |g|, by central FD on p %.4g |g|; of the former, on the group %.0f %%, on the 6 largest sites %s' % (
        nrm(r1)/gn, nrm(gz+Ht_old(p1))/gn, 100*float((rk1[gi]**2).sum()/(rk1**2).sum()), torch.argsort(rk1, descending=True)[:6].tolist()))
    # the product itself along those directions: production vs split vs central
    for i_, d_ in enumerate(last[-2:]+[p1]):
        name = 'd%d' % (i_+1) if i_ < 2 else 'p'
        a_ = Hfd(d_); b_ = Hsplit(d_); c_ = Hsplit(d_, h=hc, central=True)
        log('SPLIT product along %s: |H_prod - H_split| / |H_split| %.2e (on the group\'s rows %.2e), |H_split one-sided - H_split central| / |.| %.2e' % (
            name, nrm(a_-b_)/nrm(b_), float((a_-b_)[gi].norm()/b_[gi].norm().clamp(min=1e-300)), nrm(b_-c_)/nrm(c_)))
    # (3) C again with the split product
    p2, _, top2, _ = pcg(MC, Hsplit, 'C with the split product (columns on S, one-sided h %.0e on the rest)' % CG.h, truths=truths)
    r2 = gz+Ht_split(p2); rk2 = r2.norm(dim=1)
    log('SPLIT C (split product) final p: residual by columns + central rest %.4g |g|, by central FD on p %.4g |g|; on the group %.0f %%, 6 largest sites %s' % (
        nrm(r2)/gn, nrm(gz+Ht_old(p2))/gn, 100*float((rk2[gi]**2).sum()/(rk2**2).sum()), torch.argsort(rk2, descending=True)[:6].tolist()))
