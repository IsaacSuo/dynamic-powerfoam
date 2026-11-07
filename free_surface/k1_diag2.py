"""The expert's three diagnostics on k1's operator at a step's start (11-06), plus the two checks of the measurement itself.
Nothing here changes the run: called from merged_drop.py's probe (--diag2), which evaluates the step's point again afterwards.
  A  differences: K v by central differences with the perturbation scaled to a maximum site displacement h_abs (1e-8 m), against
     the earlier fixed step h = 1e-7 on M^-1/2 y (whose largest site displacement is reported for the top mode)
  B  Lanczos convergence: each Ritz pair's residual beta_m |s_mi|; the modes counted above the target only where converged
  C  the integration constraint: r_k = sum_s V_s c_sk (V_s G_s = sum_k u_k (x) c_sk), by term and by class of site (surface;
     interior inside some cap's stencil; interior outside every cap's stencil); with the energy's weights V0_s / V_s as well; and
     k1's cap data a_s, M_s against the geometry's own cap integrals (patch_n, r patch_nn)
  D  (at rest) the top mode's stiffness u^T K u split into the terms' cross contributions E_ab over (wall, open, tau, cap,
     spring direct), the cells' 1/2 V0 (2 mu dev sym : dev sym + kap tr tr), the variance kap A L / 24 (tr G_i - tr G_j)^2 and
     the springs' 1/2 k beta |d|^2 (d = u_j - u_i - 1/2 (G_i + G_j) p) all split by the same terms; their sum against the
     differences' u^T K u
  E  the follow map with the consistent reduced mass: K_r = P^T K P, M_r = P^T M P, Lanczos on the pencil (K_r, M_r) in the
     M_r inner product (M_r^-1 by CG); checks: no particle slaved gives the full omega_max, and slaved omega_max <= full"""
import math

import numpy as np
import torch

from free_surface.k1_slave import anchor_map

f64, dev = torch.float64, 'cuda'


def term_triplets(W):
    """V_s G_s = sum_k u_k (x) c_sk per term: dict term -> (s, k, c) (k1_slave.column_omega's construction, terms kept apart)"""
    t, n = W.tul, W.n; ml, m = W._ml, W._w[5]
    S, O, R = t['S'][:2*ml].long(), t['O'][:2*ml].long(), t['R'][:2*ml]
    As, Ao, Aw = t['As'].long(), t['Ao'].long(), t['Aw']; used = t['used'].bool()
    wi, wj, wa, wn, wt, wtau = t['wi'][:m].long(), t['wj'][:m].long(), t['wa'][:m], t['wn'][:m], t['wt'][:m], t['wtau'][:m]
    a_, Mc = t['a'], t['Mc']
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
    T = {}
    T['wall'] = (torch.cat([wi, wi, wj, wj]), torch.cat([wi, wj, wi, wj]),
                 torch.cat([(wa*(1-wt))[:, None]*wn, (wa*wt)[:, None]*wn, -(wa*(1-wt))[:, None]*wn, -(wa*wt)[:, None]*wn]))
    ids = torch.arange(n, device=dev); T['open'] = (ids, ids, a_)
    ss, ks, cs = [], [], []
    for side in (wi, wj):
        idx, kk, gg = expand(side); coef = (gg*wtau[idx]).sum(1)*wa[idx]/2
        for cell, sg in ((wi, 1.), (wj, -1.)):
            ss.append(cell[idx]); ks.append(kk); cs.append((sg*coef)[:, None]*wn[idx])
    T['tau'] = (torch.cat(ss), torch.cat(ks), torch.cat(cs))
    cc = torch.nonzero(Mc.abs().sum((1, 2)) > 0).flatten(); idx, kk, gg = expand(cc)
    T['cap'] = (cc[idx], kk, torch.einsum('fba,fb->fa', Mc[cc[idx]], gg))
    return T


def G_of(trip, u, n):
    """V_s G_s(u) for one term (n, 3, 3)"""
    s_, k_, c_ = trip
    return torch.zeros(n, 3, 3, dtype=f64, device=dev).index_add(0, s_, u[k_][:, :, None]*c_[:, None, :])


def lanczos(Av, n, k, mask=None, seed=0):
    Q = torch.zeros(k+1, n, 3, dtype=f64, device=dev)
    q = torch.randn(n, 3, dtype=f64, device=dev, generator=torch.Generator(device=dev).manual_seed(seed))
    if mask is not None:
        q = q*mask[:, None]
    Q[0] = q/q.norm(); al, be = [], []
    for it in range(k):
        w = Av(Q[it]); a = float((w*Q[it]).sum()); al.append(a)
        w = w-a*Q[it]-(be[-1]*Q[it-1] if it > 0 else 0.)
        for _ in range(2):
            c = torch.einsum('kab,ab->k', Q[:it+1], w); w = w-torch.einsum('k,kab->ab', c, Q[:it+1])
        b = float(w.norm()); be.append(b); Q[it+1] = w/b
    T = np.diag(al)+np.diag(be[:-1], 1)+np.diag(be[:-1], -1)
    lam, S = np.linalg.eigh(T)
    res = be[-1]*np.abs(S[-1, :])                                                    # (Ritz residuals |A Q y - theta Q y|)
    return lam, S, Q[:k], res


def gen_lanczos(Kf, Mf, Minv, n, k, mask, seed=0):
    """Lanczos on the pencil (K, M): M-orthonormal basis, M^-1 by the caller's solve"""
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
    T = np.diag(al)+np.diag(be[:-1], 1)+np.diag(be[:-1], -1)
    lam, S = np.linalg.eigh(T)
    return lam, be[-1]*np.abs(S[-1, :])


def run(W, gr, X, step, log, k=120, target=9600., h_abs=1e-8, h_old=1e-7):
    n = W.n; t = W.tul; V0 = W.V0; m = V0; Ms = torch.sqrt(m)[:, None]
    capc = t['Mc'].abs().sum((1, 2)) > 0; Vm = float(V0.median()); hcell = V0**(1/3)

    def Kv(v):
        s_ = h_abs/float(v.abs().max())
        return (gr(X+s_*v)-gr(X-s_*v))/(2*s_)
    Av = lambda y: Kv(y/Ms)/Ms
    # B (and A): Lanczos with the scaled differences, Ritz residuals
    lam, S, Q, res = lanczos(Av, n, k)
    om = np.sqrt(np.clip(lam, 0, None)); conv = res <= 1e-6*np.abs(lam).max()
    above = om > target
    y = torch.einsum('k,kab->ab', torch.as_tensor(S[:, -1], dtype=f64, device=dev), Q); u = y/Ms
    wv = (y*y).sum(1); top = torch.argsort(wv, descending=True)[:3].tolist()
    log('  DIAG2 step %d: B  omega_max %.4e (residual / lambda_max %.1e); Ritz values above the target %d, of them converged %d; converged overall %d of %d' % (
        step, om[-1], res[-1]/lam[-1], int(above.sum()), int((above & conv).sum()), int(conv.sum()), k))
    # A: the earlier fixed step on the same top mode
    vold = y/Ms
    Kold = (gr(X+h_old*vold)-gr(X-h_old*vold))/(2*h_old)
    lam_old = float((y*(Kold/Ms)).sum()); lam_new = float((y*Av(y)).sum())
    dmax = h_old*vold.norm(dim=1); c_ = int(torch.argmax(dmax))
    log('  DIAG2 step %d: A  top mode Rayleigh: scaled step (max site displacement %.0e m) omega %.4e, earlier fixed step omega %.4e; the earlier step moved a site up to %.2e m '
        '(cell %d: %.2f %% of its size %.2f mm, V/med %.2f); top mode on %s (%.0f %%; %s)' % (
            step, h_abs, math.sqrt(max(lam_new, 0)), math.sqrt(max(lam_old, 0)), float(dmax[c_]), c_, 100*float(dmax[c_]/hcell[c_]), 1e3*float(hcell[c_]),
            float(V0[c_])/Vm, top, 100*float(wv[top].sum()), ', '.join('%s V/med %.2f' % ('surface' if bool(capc[q_]) else 'inside', float(V0[q_])/Vm) for q_ in top)))
    # C: the integration constraint
    T = term_triplets(W)
    V = 1/t['Vi']
    rk = {a: torch.zeros(n, 3, dtype=f64, device=dev).index_add(0, T[a][1], T[a][2]) for a in T}
    rk0 = {a: torch.zeros(n, 3, dtype=f64, device=dev).index_add(0, T[a][1], (V0/V)[T[a][0]][:, None]*T[a][2]) for a in T}
    scale = torch.zeros(n, dtype=f64, device=dev)
    for a in T:
        scale.index_add_(0, T[a][1], T[a][2].norm(dim=1))
    stencil_cap = torch.zeros(n, dtype=torch.bool, device=dev)
    for a in ('open', 'cap'):
        s_, k_, c_ = T[a]; nz = c_.norm(dim=1) > 0
        stencil_cap[k_[nz & capc[s_]]] = True
    cls = {'surface': capc, 'interior in a cap stencil': ~capc & stencil_cap, 'interior outside every cap stencil': ~capc & ~stencil_cap}
    rtot = sum(rk.values()); rtot0 = sum(rk0.values())
    for name, msk in cls.items():
        if not bool(msk.any()):
            continue
        rel = (rtot.norm(dim=1)/scale.clamp(min=1e-300))[msk]; rel0 = (rtot0.norm(dim=1)/scale.clamp(min=1e-300))[msk]
        parts = ', '.join('%s %.1e' % (a, float((rk[a].norm(dim=1)/scale.clamp(min=1e-300))[msk].max())) for a in T)
        log('  DIAG2 step %d: C  %s (%d sites): |sum_s V_s c_sk| / sum |c| median %.1e, max %.1e (with V0: max %.1e); max by term: %s' % (
            step, name, int(msk.sum()), float(rel.median()), float(rel.max()), float(rel0.max()), parts))
    rw = (rk['wall']+rk['tau']).norm(dim=1)/scale.clamp(min=1e-300)
    log('  DIAG2 step %d: C  walls + tau summed over the cells, all sites: max %.1e (shared walls cancel); with V0 weights: max %.1e' % (
        step, float(rw.max()), float(((rk0['wall']+rk0['tau']).norm(dim=1)/scale.clamp(min=1e-300)).max())))
    Gg = W._ulG
    pn, pnn = Gg.patch_n.detach(), Gg.patch_nn.detach(); rr = W._ulr
    gcap = Gg.patch_A.detach() > 0                                                   # (the geometry's exposed caps; k1 also opens cells at walls it does not count)
    ea = float(((t['a']-pn)[gcap]).norm(dim=1).max()/pn[gcap].norm(dim=1).max())
    eM = float(((t['Mc']-rr[:, None, None]*pnn)[gcap]).flatten(1).norm(dim=1).max()/(rr[:, None, None]*pnn)[gcap].flatten(1).norm(dim=1).max())
    log('  DIAG2 step %d: C  k1\'s cap data against the geometry\'s cap integrals: a_s vs int n dA max rel %.1e; M_s vs r int n n^T dA max rel %.1e' % (step, ea, eM))
    # D: cross contributions of the top mode (at rest only: the tangent below is the elastic one at F = I)
    if step <= 1:
        G = {a: G_of(T[a], u, n)/V[:, None, None] for a in T}
        I3 = torch.eye(3, dtype=f64, device=dev)
        dsym = lambda A_: .5*(A_+A_.transpose(1, 2))-(A_.diagonal(dim1=1, dim2=2).sum(1)/3)[:, None, None]*I3
        tr = lambda A_: A_.diagonal(dim1=1, dim2=2).sum(1)
        mt = W._w[5]; wi, wj = t['wi'][:mt].long(), t['wj'][:mt].long(); wv_ = t['wv'][:mt]
        ms = W._m_s; si, sj = W.tsp['si'][:ms].long(), W.tsp['sj'][:ms].long(); sp, sk, sb = W.tsp['sP'][:ms], W.tsp['sk'][:ms], W.tsp['sb'][:ms]
        D = {a: -.5*torch.einsum('eab,eb->ea', G[a][si]+G[a][sj], sp) for a in T}
        D['sdirect'] = u[sj]-u[si]
        keys = list(T)+['sdirect']
        E = np.zeros((len(keys), len(keys)))
        for ia, a in enumerate(keys):
            for ib, b in enumerate(keys):
                e_ = float((.5*sk*sb*(D[a]*D[b]).sum(1)).sum())
                if a in T and b in T:
                    e_ += float((.5*V0*(2*W.mu*(dsym(G[a])*dsym(G[b])).sum((1, 2))+W.kap*tr(G[a])*tr(G[b]))).sum())
                    e_ += float((wv_*(tr(G[a])[wi]-tr(G[a])[wj])*(tr(G[b])[wi]-tr(G[b])[wj])).sum())
                E[ia, ib] = e_
        Etot = E.sum(); uKu = float((u*Kv(u)).sum())
        log('  DIAG2 step %d: D  top mode: 2 x sum of the split %.6e vs differences u^T K u %.6e (rel %.1e)' % (step, 2*Etot, uKu, abs(2*Etot-uKu)/abs(uKu)))
        rows = []
        for ia, a in enumerate(keys):
            rows.append('%s: ' % a+' '.join('%+.2f' % ((E[ia, ib] if ia == ib else 2*E[ia, ib])/Etot) for ib in range(len(keys)) if ib >= ia))
        log('  DIAG2 step %d: D  absolute energies of the top mode (u M-normalised x 1; E_total %.4e): diagonal %s; largest off-diagonal pairs (x 2) %s' % (
            step, Etot, ', '.join('%s %.4e' % (a, E[ia, ia]) for ia, a in enumerate(keys)),
            ', '.join('%s-%s %.4e' % (keys[i_], keys[j_], 2*E[i_, j_]) for i_, j_ in sorted([(i_, j_) for i_ in range(len(keys)) for j_ in range(i_+1, len(keys))], key=lambda p_: -abs(E[p_]))[:4])))
        log('  DIAG2 step %d: D  shares of the top mode\'s energy (upper triangle, off-diagonal pairs doubled; order %s): %s' % (step, ' '.join(keys), ' | '.join(rows)))
    # E: the follow map with the consistent reduced mass
    from free_surface.k1_slave import column_omega
    mt = W._w[5]; wi, wj, wa = t['wi'][:mt].long(), t['wj'][:mt].long(), t['wa'][:mt]
    tiny = V0 < W._ul_agg*V0.median()
    for label, sel in (('none', torch.zeros(n, dtype=torch.bool, device=dev)), ('stiff + tiny', (column_omega(W) > target) | tiny)):
        if bool(sel.any()):
            s_, k_, w_, free, wm = anchor_map(X, wi, wj, wa, sel)
            sl = sel.clone()
            if free:
                sl[torch.as_tensor(free, device=dev)] = False
        else:
            s_ = k_ = torch.zeros(0, dtype=torch.long, device=dev); w_ = torch.zeros(0, dtype=f64, device=dev); sl = sel; wm = np.zeros(1)
        master = ~sl; mk = master[:, None].to(f64)
        P = lambda y_: y_*mk+torch.zeros_like(y_).index_add(0, s_, w_[:, None]*y_[k_])
        Pt = lambda r_: r_*mk+torch.zeros_like(r_).index_add(0, k_, w_[:, None]*r_[s_])
        Mf = lambda y_: Pt(m[:, None]*P(y_))*mk
        dM = (m+torch.zeros(n, dtype=f64, device=dev).index_add(0, k_, w_*w_*m[s_]))[:, None]

        def Minv(b_):
            x_ = torch.zeros_like(b_); r_ = b_*mk; z_ = r_/dM; p_ = z_.clone(); rz = float((r_*z_).sum()); b0 = float(r_.norm())
            for _ in range(500):
                Ap = Mf(p_); al_ = rz/float((p_*Ap).sum()); x_ = x_+al_*p_; r_ = r_-al_*Ap
                if float(r_.norm()) <= 1e-13*b0:
                    break
                z_ = r_/dM; rz1 = float((r_*z_).sum()); p_ = z_+(rz1/rz)*p_; rz = rz1
            return x_*mk
        Kf = lambda y_: Pt(Kv(P(y_)))*mk
        lam_r, res_r = gen_lanczos(Kf, Mf, Minv, n, k, master.to(f64))
        omr = math.sqrt(max(lam_r[-1], 0.))
        log('  DIAG2 step %d: E  follow (%s): %d slaved, max |w| %.2f; consistent reduced omega_max %.4e (residual / lambda %.1e) vs full %.4e: %s; -> %.2f of the MPM substep' % (
            step, label, int(sl.sum()), float(np.max(wm)), omr, res_r[-1]/lam_r[-1], om[-1], 'not above (as the Rayleigh bound requires)' if omr <= om[-1]*(1+1e-6) else 'ABOVE the full one: check',
            2/omr*4800))
