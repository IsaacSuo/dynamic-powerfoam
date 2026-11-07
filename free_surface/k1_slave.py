"""Stiff particles following their neighbours' fit (11-06, the user: the particles keep moving, where the stiffness is too
large their motion is fitted, and they spring back): a check on k1's operator at a step's start, no change to the run.
  select   the cells whose column-stiffness estimate (k1_stiff_cells.py) exceeds the target omega = 2 / dt
  slave    u_s = sum_k w_sk u_k, the affine least-squares fit of its non-slaved wall neighbours (area weights) evaluated at x_s
           (the 2-ring when the 1-ring does not span): w_sk = om_k / sum om + om_k (x_k - xbar)^T M^-1 (x_s - xbar), sum_k w_sk = 1,
           so translations and (to first order) rotations go through; each slaved particle keeps its mass and state
  reduced  K_r = P^T K P (P: masters -> all), masses lumped by the row sums m_k + sum_s w_sk m_s; Lanczos on M_r^-1/2 K_r M_r^-1/2;
           the cells holding >= 20 % of a mode above the target are slaved next, in rounds
lanczos(Av, n, k) works on (n, 3) vectors; Av applies the scaled operator."""
import numpy as np
import torch

f64, dev = torch.float64, 'cuda'


def lanczos(Av, n, k, mask=None, seed=0):
    Q = torch.zeros(k+1, n, 3, dtype=f64, device=dev)
    q = torch.randn(n, 3, dtype=f64, device=dev, generator=torch.Generator(device=dev).manual_seed(seed))
    if mask is not None:
        q = q*mask[:, None]
    Q[0] = q/q.norm()
    al, be = [], []
    for it in range(k):
        w = Av(Q[it]); a = float((w*Q[it]).sum()); al.append(a)
        w = w-a*Q[it]-(be[-1]*Q[it-1] if it > 0 else 0.)
        for _ in range(2):
            c = torch.einsum('kab,ab->k', Q[:it+1], w); w = w-torch.einsum('k,kab->ab', c, Q[:it+1])
        b = float(w.norm()); be.append(b)
        if b < 1e-14:
            break
        Q[it+1] = w/b
    m = len(al)
    lam, S = np.linalg.eigh(np.diag(al)+np.diag(be[:m-1], 1)+np.diag(be[:m-1], -1))
    return np.sqrt(np.clip(lam, 0, None)), S, Q[:m]


def mode_cells(S, Q, om, target, share=.2):
    """cells holding >= share of each Ritz vector above the target; and the top mode's 3 cells with their shares"""
    hot = torch.zeros(Q.shape[1], dtype=torch.bool, device=dev); top = None
    for r_ in range(1, len(om)+1):
        y = torch.as_tensor(S[:, -r_], dtype=f64, device=dev); v = torch.einsum('k,kab->ab', y, Q); w = (v*v).sum(1)
        if r_ == 1:
            o = torch.argsort(w, descending=True)[:3]; top = (o.tolist(), float(w[o].sum()/w.sum()))
        if om[-r_] <= target:
            break
        hot |= w >= share*w.sum()
    return hot, top


def column_omega(W):
    """k1_stiff_cells.py's estimate per cell: omega_k^2 ~ [(lam + 2 mu) sum_s V_s |c_sk|^2 + springs] / m_k"""
    t, n = W.tul, W.n; ml, m = W._ml, W._w[5]
    S, O, R = t['S'][:2*ml].long(), t['O'][:2*ml].long(), t['R'][:2*ml]
    As, Ao, Aw = t['As'].long(), t['Ao'].long(), t['Aw']; used = t['used'].bool()
    wi, wj, wa, wn, wt, wtau = t['wi'][:m].long(), t['wj'][:m].long(), t['wa'][:m], t['wn'][:m], t['wt'][:m], t['wtau'][:m]
    a_, Mc, Vi = t['a'], t['Mc'], t['Vi']; V = 1/Vi; l2m = W.kap+4*W.mu/3
    s0, k0, g0 = torch.cat([S, S]), torch.cat([O, S]), torch.cat([R, -R])
    keep = ~used[s0]
    sa, ka, ga = [s0[keep]], [k0[keep]], [g0[keep]]
    ok = Aw != 0
    As_, Ao_, Aw_ = As[ok], Ao[ok], Aw[ok]
    order = torch.argsort(s0); s0o, k0o, g0o = s0[order], k0[order], g0[order]
    cnt = torch.bincount(s0o, minlength=n); off = torch.cat([torch.zeros(1, dtype=torch.long, device=dev), torch.cumsum(cnt, 0)])
    reps = cnt[Ao_]; src = torch.repeat_interleave(torch.arange(len(Ao_), device=dev), reps)
    pos = off[Ao_][src]+(torch.arange(int(reps.sum()), device=dev)-torch.repeat_interleave(torch.cumsum(reps, 0)-reps, reps))
    sa.append(As_[src]); ka.append(k0o[pos]); ga.append(Aw_[src][:, None]*g0o[pos])
    se, ke, ge = torch.cat(sa), torch.cat(ka), torch.cat(ga)
    o2 = torch.argsort(se); se, ke, ge = se[o2], ke[o2], ge[o2]
    cntE = torch.bincount(se, minlength=n); offE = torch.cat([torch.zeros(1, dtype=torch.long, device=dev), torch.cumsum(cntE, 0)])

    def expand(cells):
        r_ = cntE[cells]; idx = torch.repeat_interleave(torch.arange(len(cells), device=dev), r_)
        p_ = offE[cells][idx]+(torch.arange(int(r_.sum()), device=dev)-torch.repeat_interleave(torch.cumsum(r_, 0)-r_, r_))
        return idx, ke[p_], ge[p_]
    ss, ks = [torch.cat([wi, wi, wj, wj])], [torch.cat([wi, wj, wi, wj])]
    cs = [torch.cat([(wa*(1-wt))[:, None]*wn, (wa*wt)[:, None]*wn, -(wa*(1-wt))[:, None]*wn, -(wa*wt)[:, None]*wn])]
    ids = torch.arange(n, device=dev); ss.append(ids); ks.append(ids); cs.append(a_)
    for side in (wi, wj):
        idx, kk, gg = expand(side); coef = (gg*wtau[idx]).sum(1)*wa[idx]/2
        for cell, sg in ((wi, 1.), (wj, -1.)):
            ss.append(cell[idx]); ks.append(kk); cs.append((sg*coef)[:, None]*wn[idx])
    cc = torch.nonzero(Mc.abs().sum((1, 2)) > 0).flatten(); idx, kk, gg = expand(cc)
    ss.append(cc[idx]); ks.append(kk); cs.append(torch.einsum('fba,fb->fa', Mc[cc[idx]], gg))
    s_, k_, c_ = torch.cat(ss), torch.cat(ks), torch.cat(cs)
    u, inv = torch.unique(s_*n+k_, return_inverse=True)
    c = torch.zeros(len(u), 3, dtype=f64, device=dev).index_add(0, inv, c_); su, ku = u//n, u % n; c = c*Vi[su][:, None]
    K = torch.zeros(n, dtype=f64, device=dev).index_add(0, ku, l2m*V[su]*(c*c).sum(1))
    sk = W.tsp['sk'][:W._m_s]
    K = K.index_add(0, W.tlo, sk).index_add(0, W.thi, sk)
    return torch.sqrt(K/W.V0)


def slave_map(x, wi, wj, wa, slave):
    """for each slaved cell its masters and weights (the affine fit of its non-slaved wall neighbours, the 2-ring when the 1-ring
    does not span); returns (s, k, w) triplets and the slaved cells left free (no spanning masters)"""
    n = len(x)
    a_i, a_j = torch.cat([wi, wj]), torch.cat([wj, wi]); a_w = torch.cat([wa, wa])
    nb = [[] for _ in range(n)]
    ai, aj, aw = a_i.cpu().numpy(), a_j.cpu().numpy(), a_w.cpu().numpy()
    for p_, q_, w_ in zip(ai, aj, aw):
        nb[p_].append((q_, w_))
    X = x.cpu().numpy(); sl = slave.cpu().numpy()
    S_, K_, W_ = [], [], []; free = []
    for s in np.nonzero(sl)[0]:
        for ring in (1, 2):
            cand = {}
            for q_, w_ in nb[s]:
                if not sl[q_]:
                    cand[q_] = cand.get(q_, 0.)+w_
            if ring == 2:
                for q_, _ in nb[s]:
                    for r_, w2 in nb[q_]:
                        if r_ != s and not sl[r_]:
                            cand[r_] = cand.get(r_, 0.)+.5*w2
            if len(cand) < 4:
                continue
            ks = np.array(list(cand)); om = np.array([cand[k] for k in ks]); om = om/om.sum()
            xb = (om[:, None]*X[ks]).sum(0); d = X[ks]-xb
            M = (om[:, None, None]*d[:, :, None]*d[:, None, :]).sum(0)
            ev = np.linalg.eigvalsh(M)
            if ev[0] < 1e-3*ev[-1]:
                continue
            w = om+om*(d@np.linalg.solve(M, X[s]-xb))
            S_ += [s]*len(ks); K_ += ks.tolist(); W_ += w.tolist()
            break
        else:
            free.append(s)
    T = lambda a, dt=torch.long: torch.as_tensor(np.array(a), dtype=dt, device=dev)
    return T(S_), T(K_), T(W_, f64), free


def reduced_omega(W, gradfn, X, sel, k=120, hfd=1e-7, target=9600., rounds=4):
    """slave sel (and, in rounds, the reduced operator's hot cells), Lanczos on the reduced operator; returns a list of round records"""
    n = W.n; m_ = W._w[5]; t = W.tul
    wi, wj, wa = t['wi'][:m_].long(), t['wj'][:m_].long(), t['wa'][:m_]
    slave = sel.clone(); out = []
    for rnd in range(rounds):
        s_, k_, w_, free = slave_map(X, wi, wj, wa, slave)
        sl = slave.clone()
        if free:
            sl[torch.as_tensor(free, device=dev)] = False                           # (no spanning masters: left free)
        master = ~sl
        mr_raw = W.V0.clone().index_add(0, k_, w_*W.V0[s_])
        mr = torch.where(master, torch.maximum(mr_raw, .5*W.V0), torch.ones_like(mr_raw))   # (negative fit weights could take mass off)
        Ms = torch.sqrt(mr)[:, None]

        def P(y):
            return y*master[:, None]+torch.zeros_like(y).index_add(0, s_, w_[:, None]*y[k_])

        def Pt(r):
            return r*master[:, None]+torch.zeros_like(r).index_add(0, k_, w_[:, None]*r[s_])

        def Av(y):
            v = P(y/Ms)
            Kv = (gradfn(X+hfd*v)-gradfn(X-hfd*v))/(2*hfd)
            return Pt(Kv)/Ms*master[:, None]
        om, S, Q = lanczos(Av, n, k, mask=master.to(f64))
        hot, top = mode_cells(S, Q, om, target)
        out.append(dict(round=rnd+1, slaved=int(sl.sum()), free=len(free), om=float(om[-1]), above=int((om > target).sum()), top=top,
                        mr_min=float((mr_raw[master]/W.V0[master]).min())))
        new = hot & master
        if om[-1] <= target or not bool(new.any()):
            break
        slave |= new
    return out


def floor_fit(W, eps):
    """k1's least-squares fit G0 rebuilt with a floor (11-06; as the ring route's fit floor of 11-03, 9a4c3ca): per cell
    M = sum w d d^T (update_ul's entries: the connected walls, A / L, and the springs without a wall at their rest weight),
    its eigenvalues under eps of its largest raised to it, E the raise, G0_s = S_s (M + E)^-1 + Gbar_s E (M + E)^-1, Gbar_s the
    volume-weighted unfloored fits of its wall neighbours whose own M needs no floor (exact under uniform strain: H M Mf^-1 +
    H E Mf^-1 = H); a floored cell with no such neighbour keeps the floor alone. Returns the fit arrays (S, O, R) in
    k_Gul's form G0_s = sum (u_o - u_s) (x) R and stats (floored cells, with no good neighbour, uniform-strain error)"""
    n = W.n; t = W.tul; x = t['xn']; G = W._ulG
    kp = G.wall_A.detach() > 0
    i, j, A = G.i[kp].long(), G.j[kp].long(), G.wall_A.detach()[kp]
    lo, hi = torch.minimum(i, j), torch.maximum(i, j); key = lo*n+hi
    conn = torch.isin(key, W.tkey); has_wall = torch.isin(W.tkey, key)
    ic, jc, Ac = i[conn], j[conn], A[conn]
    d = x[jc]-x[ic]; w = Ac/d.norm(dim=1)
    gh = ~has_wall
    ds = x[W.thi]-x[W.tlo]
    li, lj = torch.cat([ic, W.tlo[gh]]), torch.cat([jc, W.thi[gh]])
    dd, ww = torch.cat([d, ds[gh]]), torch.cat([w, W.tk0[gh]/(W.kshare*2*W.mu)])
    q = ww[:, None, None]*dd[:, :, None]*dd[:, None, :]
    M = torch.zeros(n, 3, 3, dtype=f64, device=dev).index_add(0, li, q).index_add(0, lj, q)
    lam, U = torch.linalg.eigh(M)
    fl = eps*lam[:, -1:].clamp(min=1e-300)
    raise_ = (fl-lam).clamp(min=0.)
    floored = (raise_ > 0).any(1)
    Mfi = U@torch.diag_embed(1/(lam+raise_).clamp(min=1e-300))@U.transpose(1, 2)
    E = U@torch.diag_embed(raise_)@U.transpose(1, 2)
    B = E@Mfi                                                                        # (Gbar E Mf^-1 = Gbar B)
    vd = ww[:, None]*dd
    Ri = torch.einsum('fa,fab->fb', vd, Mfi[li]); Rj = torch.einsum('fa,fab->fb', -vd, Mfi[lj])
    S0, O0, R0 = torch.cat([li, lj]), torch.cat([lj, li]), torch.cat([Ri, Rj])
    # the good neighbours' plain fits (their M needs no floor, so Mfi = M^-1 there)
    good = ~floored
    a_s, a_o = torch.cat([ic, jc]), torch.cat([jc, ic])
    sel = floored[a_s] & good[a_o]
    a_s, a_o = a_s[sel], a_o[sel]
    va = W.V0[a_o]; den = torch.zeros(n, dtype=f64, device=dev).index_add(0, a_s, va); aw = va/den[a_s]
    order = torch.argsort(S0); S0o, O0o, R0o = S0[order], O0[order], R0[order]
    cnt = torch.bincount(S0o, minlength=n); off = torch.cat([torch.zeros(1, dtype=torch.long, device=dev), torch.cumsum(cnt, 0)])
    reps = cnt[a_o]; src = torch.repeat_interleave(torch.arange(len(a_o), device=dev), reps)
    pos = off[a_o][src]+(torch.arange(int(reps.sum()), device=dev)-torch.repeat_interleave(torch.cumsum(reps, 0)-reps, reps))
    s_f, o_f, Rp = a_s[src], O0o[pos], R0o[pos]
    BtR = aw[src][:, None]*torch.einsum('fba,fb->fa', B[s_f], Rp)                # (u_o' - u_o) (x) R' B = u (x) B^T R'
    sumR = torch.zeros(len(a_o), 3, dtype=f64, device=dev).index_add(0, src, BtR)
    S_ = torch.cat([S0, s_f, a_s]); O_ = torch.cat([O0, o_f, a_o]); R_ = torch.cat([R0, BtR, -sumR])
    # uniform strain: sum (x_o - x_s) (x) R should be I per cell
    T = torch.zeros(n, 3, 3, dtype=f64, device=dev).index_add(0, S_, (x[O_]-x[S_])[:, :, None]*R_[:, None, :])
    I3 = torch.eye(3, dtype=f64, device=dev)
    err = (T-I3).flatten(1).abs().max(1).values
    nogood = floored & (den == 0)
    return S_, O_, R_, dict(floored=int(floored.sum()), nogood=int(nogood.sum()), err_good=float(err[~nogood].max()),
                            err_nogood=float(err[nogood].max()) if bool(nogood.any()) else 0., entries=len(S_), Rh_max_before=None)


def anchor_map(x, wi, wj, wa, slave, wmax=4.):
    """the follow rule (11-06, second form): each slaved particle s takes an anchor k, a non-slaved wall neighbour (largest
    shared wall first), and moves with k's affine motion, u_s = u_k + G_k (x_s - x_k), G_k k's least-squares fit over its own
    non-slaved neighbours (weights A / L): weights 1 - sum c_o on k and c_o = w_o d_o^T M_k^-1 (x_s - x_k) on each fit neighbour o.
    Sum 1, affine fields exact, the lever one spacing (no extrapolation from a one-sided set). Of the first three anchors the
    first with max |w| <= wmax, else the one with the smallest max |w|. Returns (s, k, w) triplets, the slaved left free (no
    anchor with a spanning fit) and the max |w| per slaved particle"""
    n = len(x); X = x.cpu().numpy(); sl = slave.cpu().numpy()
    I, J, Aw = wi.cpu().numpy(), wj.cpu().numpy(), wa.cpu().numpy()
    nb = [[] for _ in range(n)]
    for p_, q_, a_ in zip(I, J, Aw):
        L = np.linalg.norm(X[q_]-X[p_]); nb[p_].append((q_, a_, a_/L)); nb[q_].append((p_, a_, a_/L))
    fitc = {}

    def fit(k):
        """k's fit over its non-slaved neighbours: (neighbours, rows w_o d_o^T M^-1) or None"""
        if k in fitc:
            return fitc[k]
        os_ = [(o, w) for o, _, w in nb[k] if not sl[o]]
        res = None
        if len(os_) >= 3:
            o_ = np.array([o for o, _ in os_]); w_ = np.array([w for _, w in os_]); d = X[o_]-X[k]
            M = (w_[:, None, None]*d[:, :, None]*d[:, None, :]).sum(0); ev = np.linalg.eigvalsh(M)
            if ev[0] > 1e-3*ev[-1]:
                res = (o_, (w_[:, None]*d)@np.linalg.inv(M))
        fitc[k] = res
        return res
    S_, K_, W_, free, wm = [], [], [], [], []
    for s in np.nonzero(sl)[0]:
        anchors = sorted([(a_, k) for k, a_, _ in nb[s] if not sl[k]], reverse=True)[:3]
        best = None
        for _, k in anchors:
            f = fit(k)
            if f is None:
                continue
            o_, rows = f; c = rows@(X[s]-X[k])
            ks = np.concatenate([[k], o_]); ws = np.concatenate([[1-c.sum()], c]); m_ = np.abs(ws).max()
            if best is None or m_ < best[2]:
                best = (ks, ws, m_)
            if m_ <= wmax:
                break
        if best is None:
            free.append(s); continue
        S_ += [s]*len(best[0]); K_ += best[0].tolist(); W_ += best[1].tolist(); wm.append(best[2])
    T = lambda a, dt=torch.long: torch.as_tensor(np.array(a), dtype=dt, device=dev)
    return T(S_), T(K_), T(W_, f64), free, np.array(wm)


def follow_omega(W, gradfn, X, sel, k=120, hfd=1e-7, target=9600., rounds=4, check_plumbing=False, log=print):
    """the slaved particles following their anchors (anchor_map), the reduced operator's omega by Lanczos, in rounds (the
    reduced operator's hot cells slaved next); before each Lanczos the map's checks: sum of weights, an affine field reproduced,
    max |w|; check_plumbing: the reduced operator with no particle slaved against the full one first. Masses lumped:
    m_k + sum_s m_s |w_sk| / sum |w_s.| (positive, the total kept)"""
    n = W.n; m_ = W._w[5]; t = W.tul
    wi, wj, wa = t['wi'][:m_].long(), t['wj'][:m_].long(), t['wa'][:m_]
    tiny = W.V0 < W._ul_agg*W.V0.median()

    def run(slave):
        s_, k_, w_, free, wm = anchor_map(X, wi, wj, wa, slave)
        sl = slave.clone()
        if free:
            sl[torch.as_tensor(free, device=dev)] = False
        master = ~sl
        aw = w_.abs(); den = torch.zeros(n, dtype=f64, device=dev).index_add(0, s_, aw)
        mr = W.V0.clone().index_add(0, k_, W.V0[s_]*aw/den[s_]); mr = torch.where(master, mr, torch.ones_like(mr))
        Ms = torch.sqrt(mr)[:, None]
        P = lambda y: y*master[:, None]+torch.zeros_like(y).index_add(0, s_, w_[:, None]*y[k_])
        Pt = lambda r: r*master[:, None]+torch.zeros_like(r).index_add(0, k_, w_[:, None]*r[s_])
        # the map's checks
        g = torch.Generator(device=dev).manual_seed(1)
        a0, H = torch.randn(3, dtype=f64, device=dev, generator=g), torch.randn(3, 3, dtype=f64, device=dev, generator=g)
        u = a0+X@H.T
        up = P(u*master[:, None])
        aff = float((up-u)[sl].abs().max()/u.abs().max()) if bool(sl.any()) else 0.
        pu = float((torch.zeros(n, dtype=f64, device=dev).index_add(0, s_, w_)-1)[sl].abs().max()) if bool(sl.any()) else 0.

        def Av(y):
            v = P(y/Ms)
            return Pt((gradfn(X+hfd*v)-gradfn(X-hfd*v))/(2*hfd))/Ms*master[:, None]
        om, S, Q = lanczos(Av, n, k, mask=master.to(f64))
        hot, top = mode_cells(S, Q, om, target)
        return om, hot, top, dict(slaved=int(sl.sum()), tiny=int((sl & tiny).sum()), free=len(free), pu=pu, aff=aff,
                                  wmax=float(wm.max()) if len(wm) else 0., w99=float(np.percentile(wm, 99)) if len(wm) else 0.)
    if check_plumbing:
        om0, _, _, _ = run(torch.zeros(n, dtype=torch.bool, device=dev))
        log('    FOLLOW plumbing: no particle slaved, reduced omega_max %.4e (the full one above)' % om0[-1])
    slave = sel | tiny; out = []
    for rnd in range(rounds):
        om, hot, top, info = run(slave)
        Vm = float(W.V0.median()); capc = W.tul['Mc'].abs().sum((1, 2)) > 0
        log('    FOLLOW round %d: %d particles follow an anchor (%d of them tiny; %d left free), map checks: weights sum to 1 within %.1e, an affine field '
            'within %.1e, max |w| %.2f (99 %% %.2f) | reduced omega_max %.3e -> %.2f of the MPM substep; modes above the target %d of %d; fastest on %s (%.0f %%; %s)' % (
                rnd+1, info['slaved'], info['tiny'], info['free'], info['pu'], info['aff'], info['wmax'], info['w99'], om[-1], 2/om[-1]*4800,
                int((om > target).sum()), len(om), top[0], 100*top[1],
                ', '.join('%s V/med %.2f' % ('surface' if bool(capc[c_]) else 'inside', float(W.V0[c_])/Vm) for c_ in top[0])))
        out.append((om[-1], info))
        new = hot & ~slave
        if om[-1] <= target or not bool(new.any()):
            break
        slave |= new
    return out
