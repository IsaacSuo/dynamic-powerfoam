"""Diagnostics for the follow screen (11-06, after the expert's fourth reply). Logging only.

  expand_rows   each slaved particle's row composed onto the final masters, and its chain depth
  row_history   one slaved particle's row from its own level on, substitution by substitution (which later level replaced
                which of its points, and how max |w|, sum |w| grew)
  mode_energy   a mode u (all particles' motion, u = P q) split into k1's quadratic parts at the step's point X:
                  the cell's shape  2 V0 mu |de|^2 (yielded: 2 V0 mu beta (|de|^2 - (rhat : de)^2)), de = dev sym(Rh^T dG F0)
                  the cell's volume V0 kap dL^2, dL = tr(C^T dG)
                  the variance     2 wv (dL_i - dL_j)^2 per wall
                  the springs      k beta |dd|^2, dd = u_j - u_i - 1/2 (dG_i + dG_j) p
                dG = G(X + s u) - G(X - s u) / 2 s from the graph itself (G is linear in x within the step, so this is exact and
                every gradient term's cross contribution stays inside the cell); their sum against u^T K u by the gradient's
                differences, the remainder being the ground contact. Per cell: its own shape + volume, half of each of its walls'
                variance and springs. For the cell holding the most: which masters' motion makes its dG (its stencil's points
                from term_triplets, the slaved ones expanded onto their masters), each master's exact share of the cell's
                energy (de^(j) : de; they sum to the cell's)."""
import numpy as np
import torch

from free_surface.k1_diag2 import term_triplets

f64, dev = torch.float64, 'cuda'


def level_rows(FM):
    """per level: {slaved particle: {point: w}} as chosen (the points masters at that moment)"""
    out = []
    for s_, k_, w_ in FM.levels:
        d = {}
        for sv, kk, ww in zip(s_.cpu().numpy(), k_.cpu().numpy(), w_.cpu().numpy()):
            r = d.setdefault(int(sv), {}); r[int(kk)] = r.get(int(kk), 0.)+float(ww)
        out.append(d)
    return out


def expand_rows(FM):
    """{slaved: {final master: w}}, {slaved: chain depth} (1: its row holds masters only)"""
    master = FM.master.cpu().numpy(); rows, depth = {}, {}
    for d in reversed(level_rows(FM)):
        for sv, r in d.items():
            row = {}; dp = 1
            for kk, ww in r.items():
                if master[kk]:
                    row[kk] = row.get(kk, 0.)+ww
                else:
                    dp = max(dp, 1+depth[kk])
                    for k2, w2 in rows[kk].items():
                        row[k2] = row.get(k2, 0.)+ww*w2
            rows[sv] = row; depth[sv] = dp
    return rows, depth


def row_history(FM, s):
    """s's level, its row then, and each later substitution: [(level, replaced point, its row's size, max |w|, sum |w|)]"""
    L = level_rows(FM); ls = next((i for i, d in enumerate(L) if s in d), None)
    if ls is None:
        return None
    row = dict(L[ls][s]); hist = [(ls, None, len(row), max(abs(v) for v in row.values()), sum(abs(v) for v in row.values()))]
    for li in range(ls+1, len(L)):
        for p, rp in L[li].items():
            if p in row:
                wp_ = row.pop(p)
                for k2, w2 in rp.items():
                    row[k2] = row.get(k2, 0.)+wp_*w2
                hist.append((li, p, len(row), max(abs(v) for v in row.values()), sum(abs(v) for v in row.values())))
    return dict(level=ls, first=L[ls][s], hist=hist, final=row)


def _dev_sym(A, I3):
    S = .5*(A+A.transpose(-1, -2))
    return S-(S.diagonal(dim1=-2, dim2=-1).sum(-1)/3)[..., None, None]*I3


def mode_energy(W, gr, X, u, log, tag, FM=None, red=None, h_abs=1e-8, ntop=6, watch=(3817,), verbose=True):
    """u: (n, 3) all particles' motion; red: the masters' signed q_k.(M_r q)_k / q^T M_r q (for the masters' lines)"""
    n = W.n; t, sp = W.tul, W.tsp; V0 = W.V0; I3 = W.I3; Vm = float(V0.median()); capc = t['Mc'].abs().sum((1, 2)) > 0
    s_ = h_abs/max(float(u.abs().max()), 1e-300)
    gp = gr(X+s_*u); Gp = W.tG.clone()
    gm = gr(X-s_*u); Gm = W.tG.clone()
    gr(X)                                                                            # (back at X: the return factors there)
    uKu = float((u*(gp-gm)).sum())/(2*s_)
    dG = (Gp-Gm)/(2*s_)
    Rh, F0 = sp['Rh'], sp['F0']
    de = _dev_sym(Rh.transpose(1, 2)@dG@F0, I3)
    be = W.tbe.clone(); er = W.ter.clone(); rn = er.flatten(1).norm(dim=1)
    rh = er/rn.clamp(min=1e-300)[:, None, None]; yl = be < 1
    pr = (rh*de).sum((1, 2)); dd = (de*de).sum((1, 2))
    q_shape = 2*V0*W.mu*torch.where(yl, be*(dd-pr*pr), dd)
    dL = (t['Q']*dG).sum((1, 2)); q_vol = V0*W.kap*dL*dL
    mt = W._w[5]; wi, wj, wv = t['wi'][:mt].long(), t['wj'][:mt].long(), t['wv'][:mt]
    q_var = 2*wv*(dL[wi]-dL[wj])**2*(wi != wj)
    ms = W._m_s; si, sj = sp['si'][:ms].long(), sp['sj'][:ms].long(); sP, sk, sb = sp['sP'][:ms], sp['sk'][:ms], sp['sb'][:ms]
    ddv = u[sj]-u[si]-.5*torch.einsum('eab,eb->ea', dG[si]+dG[sj], sP)
    q_spr = sk*sb*(ddv*ddv).sum(1)*(si != sj)
    parts = dict(shape=float(q_shape.sum()), volume=float(q_vol.sum()), variance=float(q_var.sum()), springs=float(q_spr.sum()))
    rest = uKu-sum(parts.values())
    own = q_shape+q_vol
    cellE = own+torch.zeros(n, dtype=f64, device=dev).index_add(0, wi, .5*q_var).index_add(0, wj, .5*q_var) \
        .index_add(0, si, .5*q_spr).index_add(0, sj, .5*q_spr)
    slave = FM.slave if FM is not None else torch.zeros(n, dtype=torch.bool, device=dev)
    top = torch.argsort(cellE, descending=True)[:ntop].tolist()
    if not verbose:
        return dict(uKu=uKu, top=top, cellE=cellE, rest=rest, parts=parts, own=own, dL=dL)
    nb1 = lambda c: set(wj[wi == c].tolist()) | set(wi[wj == c].tolist())
    lines = []
    for c in top:
        lines.append('%d (%s %s, V/med %.2f%s; %.1f %%: shape %.1f, volume %.1f, its walls\' variance %.1f, springs %.1f)' % (
            c, 'slaved' if bool(slave[c]) else 'master', 'surface' if bool(capc[c]) else 'inside', float(V0[c])/Vm, ', yielded' if bool(yl[c]) else '',
            100*float(cellE[c])/uKu, 100*float(q_shape[c])/uKu, 100*float(q_vol[c])/uKu,
            100*.5*float(q_var[(wi == c) | (wj == c)].sum())/uKu, 100*.5*float(q_spr[(si == c) | (sj == c)].sum())/uKu))
    cum = torch.sort(cellE, descending=True)[0].cumsum(0)/uKu
    log('  ENERGY %s: u^T K u by the differences %.4e; the parts %.4e (shape %.1f %%, volume %.1f %%, variance %.1f %%, springs %.1f %%), remainder (contact) %.1f %%; '
        'the cells holding half of it %d, 90 %%: %d; top cells: %s' % (
            tag, uKu, uKu-rest, 100*parts['shape']/uKu, 100*parts['volume']/uKu, 100*parts['variance']/uKu, 100*parts['springs']/uKu, 100*rest/uKu,
            int((cum < .5).sum())+1, int((cum < .9).sum())+1, '; '.join(lines)))
    for c in list(dict.fromkeys([top[0]]+[w_ for w_ in watch if w_ < n])):
        ring = nb1(c)
        log('  ENERGY %s: cell %d: own %.1f %% (shape %.1f, volume %.1f), with its walls and springs %.1f %%; its 1-ring\'s cells (%d) %.1f %%' % (
            tag, c, 100*float(own[c])/uKu, 100*float(q_shape[c])/uKu, 100*float(q_vol[c])/uKu, 100*float(cellE[c])/uKu, len(ring),
            100*float(cellE[list(ring)].sum())/uKu if ring else 0.))
    # the top cell's dG by the masters whose motion makes it
    c = top[0]
    T = term_triplets(W); Cc = {}
    for s0, k0, c0 in T.values():
        sel = s0 == c
        for kk, cv in zip(k0[sel].tolist(), c0[sel].cpu().numpy()):
            Cc[kk] = Cc.get(kk, 0.)+cv
    Vi = float(t['Vi'][c]); un = u.cpu().numpy()
    dGc = sum(np.outer(un[kk], cv) for kk, cv in Cc.items())*Vi
    chk = float(np.linalg.norm(dGc-dG[c].cpu().numpy())/max(np.linalg.norm(dG[c].cpu().numpy()), 1e-300))
    rows, depth = expand_rows(FM) if FM is not None and FM.levels else ({}, {})
    Bm = {}
    for kk, cv in Cc.items():
        for j, w in (rows[kk].items() if kk in rows else [(kk, 1.)]):
            Bm[j] = Bm.get(j, 0.)+w*cv
    Rc, F0c, rhc = Rh[c].cpu().numpy(), F0[c].cpu().numpy(), rh[c].cpu().numpy(); Qc = t['Q'][c].cpu().numpy()
    dec = de[c].cpu().numpy(); dLc = float(dL[c]); mu, kap, V0c = W.mu, W.kap, float(V0[c]); bec = float(be[c]); ylc = bool(yl[c])

    def share(dGj):
        dej = _dev_sym(torch.as_tensor(Rc.T@dGj@F0c), torch.eye(3, dtype=f64)).numpy(); dLj = float((Qc*dGj).sum())
        if ylc:
            sh = 2*V0c*mu*bec*((dej*dec).sum()-(rhc*dej).sum()*(rhc*dec).sum())
        else:
            sh = 2*V0c*mu*(dej*dec).sum()
        return sh+V0c*kap*dLj*dLc
    sh = {j: share(np.outer(un[j], b)*Vi) for j, b in Bm.items()}
    tot = sum(sh.values())
    redn = red.cpu().numpy() if red is not None else None
    order = sorted(sh, key=lambda j: -abs(sh[j]))[:8]
    log('  ENERGY %s: cell %d\'s dG from its stencil (%d points, %d of them slaved; check against the graph %.1e) = from %d masters; their shares of the cell\'s own %.4e '
        '(sum %.4e): %s' % (
            tag, c, len(Cc), sum(1 for kk in Cc if kk in rows), chk, len(Bm), float(own[c]), tot,
            ', '.join('%d %+.0f %%%s (|u| %.2f of the max%s)' % (
                j, 100*sh[j]/float(own[c]), ' [in the stencil]' if j in Cc else '', float(np.linalg.norm(un[j]))/float(np.linalg.norm(un, axis=1).max()),
                ('; q.Mq %.0f %%' % (100*redn[j])) if redn is not None else '') for j in order)))
    if c in rows:
        log('  ENERGY %s: cell %d is slaved (chain depth %d): its row on the masters %s' % (
            tag, c, depth[c], ', '.join('%d %+.2f' % (j, w) for j, w in sorted(rows[c].items(), key=lambda p: -abs(p[1]))[:10])))
    return dict(uKu=uKu, top=top, cellE=cellE, rest=rest, parts=parts, own=own, dL=dL)


def mode_stats(FM, m, qv, Mq, thr=.2):
    """one reduced mode: all particles' m |u|^2 shares (non-negative) and the masters' signed products q_k . (M_r q)_k / q^T M_r q
    (they sum to 1, but with the coupled mass M_r they are no shares: single ones can be negative or above 1)"""
    u = FM.P(qv); wm = m*(u*u).sum(1); wm = wm/wm.sum()
    red = (qv*Mq).sum(1); red = red/red.sum()
    sl = FM.slave
    return dict(u=u, wm=wm, red=red, slaved_share=float(wm[sl].sum()), max_free=float(wm[~sl].max()) if bool((~sl).any()) else 0.,
                hot_u=(wm >= thr) & ~sl, any_u=bool((wm >= thr).any()), hot_r=(red >= thr) & ~sl)


# ---- the patch experiment (11-06, the expert's fifth reply): a few hot cells' whole gradient stencils on one affine motion
def stencil_points(W, cells):
    """every point whose motion enters the cells' G (term_triplets: walls, open face, tau through the neighbours' fits, cap
    through the cell's fit), as {cell: set of points}"""
    T = term_triplets(W); out = {c: set() for c in cells}
    cs = torch.as_tensor(list(cells), device=dev)
    for s0, k0, c0 in T.values():
        sel = torch.isin(s0, cs) & (c0.norm(dim=1) > 0)
        for c, k in zip(s0[sel].tolist(), k0[sel].tolist()):
            out[c].add(k)
    return out


def max_volume_tet(Xs):
    """indices of 4 points of Xs spanning the largest tetrahedron (greedy start, then single swaps while the volume grows):
    at a maximum every point's barycentric coordinates are within [-1, 1], since replacing vertex i by p scales the volume by |lambda_i(p)|"""
    N = len(Xs); c0 = Xs.mean(0)
    i1 = int(np.argmax(((Xs-c0)**2).sum(1))); i2 = int(np.argmax(((Xs-Xs[i1])**2).sum(1)))
    e = (Xs[i2]-Xs[i1])/np.linalg.norm(Xs[i2]-Xs[i1]); r = Xs-Xs[i1]; r = r-(r@e)[:, None]*e
    i3 = int(np.argmax((r*r).sum(1)))
    nrm = np.cross(Xs[i2]-Xs[i1], Xs[i3]-Xs[i1]); i4 = int(np.argmax(np.abs((Xs-Xs[i1])@nrm)))
    ctrl = [i1, i2, i3, i4]
    vol = lambda ids: abs(np.linalg.det(np.stack([Xs[ids[0]]-Xs[ids[3]], Xs[ids[1]]-Xs[ids[3]], Xs[ids[2]]-Xs[ids[3]]])))/6
    cur = vol(ctrl)
    for _ in range(100):
        better = False
        for sl in range(4):
            for p in range(N):
                if p in ctrl:
                    continue
                tr_ = ctrl.copy(); tr_[sl] = p; v = vol(tr_)
                if v > cur*(1+1e-9):
                    ctrl, cur, better = tr_, v, True
        if not better:
            break
    return ctrl, cur


def barycentric(Xc, Xp):
    """(N, 4) barycentric coordinates of Xp in the tetrahedron Xc (4, 3): x = sum lambda_c x_c, sum lambda = 1"""
    A = np.stack([Xc[0]-Xc[3], Xc[1]-Xc[3], Xc[2]-Xc[3]], 1)
    mu = np.linalg.solve(A, (Xp-Xc[3]).T).T
    return np.concatenate([mu, 1-mu.sum(1, keepdims=True)], 1)


def patched_map(FM, S, ctrl, lam):
    """a FollowMap: FM's levels with every row of a point of S removed (the controls become masters), then one last level
    putting every other point of S on the controls by its barycentric weights, so u_k = b + H x_k on all of S. Earlier rows
    that refer to a point of S get its patch row substituted, as the levels are nested"""
    from free_surface.k1_follow import FollowMap
    FM2 = FollowMap(FM.n); St = torch.as_tensor(sorted(S), device=dev)
    for s_, k_, w_ in FM.levels:
        keep = ~torch.isin(s_, St)
        if bool(keep.any()):
            FM2.levels.append((s_[keep], k_[keep], w_[keep]))
    FM2.slave = FM.slave.clone(); FM2.slave[St] = False
    sl = [p for p in sorted(S) if p not in ctrl]
    T = lambda a, dt=torch.long: torch.as_tensor(np.asarray(a), dtype=dt, device=dev)
    pos = {p: i for i, p in enumerate(sorted(S))}
    ss = [p for p in sl for _ in range(4)]; kk = [c for _ in sl for c in ctrl]; ww = [float(lam[pos[p], j]) for p in sl for j in range(4)]
    FM2.levels.append((T(ss), T(kk), T(ww, f64)))
    FM2.slave[T(sl)] = True
    return FM2
