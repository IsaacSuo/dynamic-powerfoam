"""Two cures for k1's stiff surface cells, checked on the operator only (no dynamics; k1_stiff_cells.py found the cause: the
cap term Ge M_c / V and the walls' tau term go through each cell's own least-squares fit G0, ill-conditioned on thin
surface cells, |R| h up to 7, |M_c| / V up to 3):
  agg    k1's own aggregation (update_ul: tiny cells' Ge = the volume-weighted G0 of their non-tiny wall neighbours) extended to
         the cells whose omega estimate exceeds the target, in rounds until none does (the arrays As, Ao, Aw, used rewritten
         after update_ul; the model code untouched)
  merge  those cells' sites merged into the wall neighbour of the largest wall area that is not itself selected (groups by
         union-find; the group's site the volume-weighted mean, its volume the sum, its radius start (sum r^3)^1/3), the foam
         re-cut once by Recut.solve_radii for the target volumes, a new MergedUL on it; in rounds
Each: Lanczos omega_max (k1_explicit_dt.py's), the column-stiffness estimate per cell (k1_stiff_cells.py's), against the target
omega = 2 / dt for dt the MPM substep 1/4800 s; merge also: cells, exposed area and extent before / after.
Run: PYTHONPATH=$PWD python -u free_surface/k1_cure_check.py [agg|merge|both]"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import warp as wp

from free_surface.merged_ul import MergedUL
from free_surface.recut import Recut

f64, dev = torch.float64, 'cuda'
foam = Path('foams/fourier8k_clean')
gt = json.loads((foam/'plasticine_scenes.json').read_text())['scenes']['0']
E, nu, ys = gt['E']/1000, gt['nu'], gt['ys']/1000
OM_T = 2*4800.                                                                       # (the target: leapfrog at the MPM substep)
which = sys.argv[1] if len(sys.argv) > 1 else 'both'


def build(X0, r0, V0, pairs, A, c):
    W = MergedUL(X0, r0, V0, pairs, A, c, E, nu, ys, theta=.5, kshare=1/6, grad='gg')
    W.set_ball((0., 1., 0.), -.05, 10.)
    W.update_ul(W.X)
    W.prepare()
    return W


def grad(W, x):
    W.tx.copy_(x); wp.capture_launch(W.graph)
    return W.tgx.clone()


def lanczos(W, k=120, hfd=1e-7):
    X, n = W.X, W.n
    Ms = torch.sqrt(W.V0)[:, None]
    Av = lambda v: ((grad(W, X+hfd*v/Ms)-grad(W, X-hfd*v/Ms))/(2*hfd))/Ms
    Q = torch.zeros(k+1, n, 3, dtype=f64, device=dev)
    q = torch.randn(n, 3, dtype=f64, device=dev, generator=torch.Generator(device=dev).manual_seed(0)); Q[0] = q/q.norm()
    al, be = [], []
    for it in range(k):
        w = Av(Q[it]); a = float((w*Q[it]).sum()); al.append(a)
        w = w-a*Q[it]-(be[-1]*Q[it-1] if it > 0 else 0.)
        for _ in range(2):
            c = torch.einsum('kab,ab->k', Q[:it+1], w); w = w-torch.einsum('k,kab->ab', c, Q[:it+1])
        b = float(w.norm()); be.append(b); Q[it+1] = w/b
    lam, S = np.linalg.eigh(np.diag(al)+np.diag(be[:-1], 1)+np.diag(be[:-1], -1))
    om = np.sqrt(np.clip(lam, 0, None))
    y = torch.as_tensor(S[:, -1], dtype=f64, device=dev); v = torch.einsum('k,kab->ab', y, Q[:k]); wv = (v*v).sum(1)
    hot = torch.zeros(n, dtype=torch.bool, device=dev)                               # (cells holding >= 20 % of a mode above the target)
    for r_ in range(1, k+1):
        if om[-r_] <= OM_T:
            break
        y_ = torch.as_tensor(S[:, -r_], dtype=f64, device=dev); v_ = torch.einsum('k,kab->ab', y_, Q[:k]); w_ = (v_*v_).sum(1)
        hot |= w_ >= .2*w_.sum()
    lanczos.hot = hot
    return om, torch.argsort(wv, descending=True)[:3].tolist(), wv


def column_omega(W):
    """k1_stiff_cells.py's estimate: omega_k^2 ~ [(lam + 2 mu) sum_s V_s |c_sk|^2 + springs] / m_k"""
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
    ss, ks, cs = [torch.cat([wi, wi, wj, wj])], [torch.cat([wi, wj, wi, wj])], \
        [torch.cat([(wa*(1-wt))[:, None]*wn, (wa*wt)[:, None]*wn, -(wa*(1-wt))[:, None]*wn, -(wa*wt)[:, None]*wn])]
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


def report(tag, W, om_c, extra=''):
    om, top, wv = lanczos(W)
    cap = W.tul['Mc'].abs().sum((1, 2)) > 0
    q = lambda p: float(torch.quantile(om_c, p))
    print(f"[{tag}] {W.n} cells{extra}: Lanczos omega_max {om[-1]:.3e} -> explicit limit {2/om[-1]:.3e} s ({2/om[-1]*4800:.2f} of the MPM substep), "
          f"modes above the target {OM_T:.0f}: {int((om > OM_T).sum())} of the top {len(om)}; estimate median {q(.5):.3e}, 99 % {q(.99):.3e}, max {float(om_c.max()):.3e}, "
          f"cells above the target {int((om_c > OM_T).sum())} (surface {int(((om_c > OM_T) & cap).sum())}); fastest mode on {top} "
          f"({100*float(wv[top].sum()):.0f} %)", flush=True)
    return om


g = np.load(foam/'dem_exact_single'/'geometry.npz'); cl = np.load(foam/'dem_exact_single'/'cells.npz')
X0, r0, V0 = g['points'], cl['radii'], g['masses']
W0 = build(X0, r0, V0, g['pairs'], g['areas'], g['contacts'])
om0c = column_omega(W0)
report('baseline', W0, om0c)
G0g = W0._ulG; ext0 = (W0.X.max(0).values-W0.X.min(0).values).cpu().numpy(); A0 = float(G0g.patch_A.detach().sum())

if which in ('agg', 'both'):
    W = W0; t = W.tul; n = W.n
    tiny = W.V0 < W._ul_agg*W.V0.median()
    sel = torch.zeros(n, dtype=torch.bool, device=dev)
    m = W._w[5]; wi, wj = t['wi'][:m].long(), t['wj'][:m].long()
    for rnd in range(8):
        om_c = column_omega(W)
        new = (om_c > OM_T) & ~sel
        if not bool(new.any()):
            break
        sel |= new
        agg = tiny | sel
        m1, m2 = agg[wi] & ~agg[wj], agg[wj] & ~agg[wi]
        as_, ao_ = torch.cat([wi[m1], wj[m2]]), torch.cat([wj[m1], wi[m2]])
        va = W.V0[ao_]; den = torch.zeros(n, dtype=f64, device=dev).index_add(0, as_, va)
        t['As'].zero_(); t['Ao'].zero_(); t['Aw'].zero_(); k3 = len(as_)
        t['As'][:k3] = as_.to(torch.int32); t['Ao'][:k3] = ao_.to(torch.int32); t['Aw'][:k3] = va/den[as_]
        t['used'].copy_((agg & (den > 0)).to(torch.int32))
        print(f"  agg round {rnd+1}: {int(new.sum())} more cells aggregated ({int(sel.sum())} beyond the tiny {int(tiny.sum())}; "
              f"without a neighbour to borrow from {int((agg & (den == 0)).sum())})", flush=True)
    report('agg', W, column_omega(W), f", {int(sel.sum())} cells aggregated beyond the tiny")

if which in ('merge', 'both'):
    X, R0, V = torch.as_tensor(X0, dtype=f64, device=dev), torch.as_tensor(r0, dtype=f64, device=dev), torch.as_tensor(V0, dtype=f64, device=dev)
    W, om_c = W0, om0c
    hot = None
    for rnd in range(8):
        n = len(X); sel = om_c > OM_T
        if hot is not None:
            sel |= hot                                                               # (and the cells of the true modes above the target)
        if not bool(sel.any()):
            break
        Gg = W._ulG; kp = Gg.wall_A.detach() > 0
        i, j, A = Gg.i[kp].long(), Gg.j[kp].long(), Gg.wall_A.detach()[kp]
        parent = torch.arange(n, device=dev)
        # each selected cell to the non-selected wall neighbour of the largest area
        cand_s, cand_t, cand_a = torch.cat([i, j]), torch.cat([j, i]), torch.cat([A, A])
        okc = sel[cand_s] & ~sel[cand_t]
        cs_, ct_, ca_ = cand_s[okc], cand_t[okc], cand_a[okc]
        best = torch.full((n,), -1., dtype=f64, device=dev).scatter_reduce(0, cs_, ca_, 'amax')
        win = ca_ == best[cs_]
        parent[cs_[win]] = ct_[win]
        lone = sel & (parent == torch.arange(n, device=dev))
        groups = parent                                                              # (one level: a selected cell joins a non-selected one)
        uq, gid = torch.unique(groups, return_inverse=True); ng = len(uq)
        Vg = torch.zeros(ng, dtype=f64, device=dev).index_add(0, gid, V)
        Xg = torch.zeros(ng, 3, dtype=f64, device=dev).index_add(0, gid, V[:, None]*X)/Vg[:, None]
        rg = torch.zeros(ng, dtype=f64, device=dev).index_add(0, gid, R0**3)**(1/3)
        rc = Recut(ng, dev); rc.A.f32 = True
        r_, Gn, S_, it, nrep = rc.solve_radii(Xg, rg, Vg, tol=1e-9)
        kp2 = Gn.wall_A.detach() > 0
        pairs = torch.stack([Gn.i[kp2], Gn.j[kp2]], 1).cpu().numpy(); Aw_ = Gn.wall_A.detach()[kp2].cpu().numpy(); cw = Gn.wall_c.detach()[kp2].cpu().numpy()
        verr = float(((Gn.V.detach()-Vg).abs()/Vg).max())
        X, R0, V = Xg, r_.detach(), Vg
        W = build(X.cpu().numpy(), R0.cpu().numpy(), V.cpu().numpy(), pairs, Aw_, cw)
        om_c = column_omega(W)
        ext = (W.X.max(0).values-W.X.min(0).values).cpu().numpy(); Ae = float(W._ulG.patch_A.detach().sum())
        hot = None
        omr = report(f'merge round {rnd+1}', W, om_c, f" ({int(sel.sum())} selected, {int(lone.sum())} without a partner; radii solve {it} iterations, "
               f"{nrep} repairs, volume error {verr:.1e}; exposed area {Ae/A0:.4f} of the rest's, site extent {np.round(ext/ext0, 4).tolist()})")
        hot = lanczos.hot.clone()
        out = foam.parent/(foam.name+'_merged')/'dem_exact_single'; out.mkdir(parents=True, exist_ok=True)
        Gw = W._ulG; kw = Gw.wall_A.detach() > 0
        np.savez(out/'geometry.npz', points=X.cpu().numpy(), masses=V.cpu().numpy(), pairs=torch.stack([Gw.i[kw], Gw.j[kw]], 1).cpu().numpy(),
                 areas=Gw.wall_A.detach()[kw].cpu().numpy(), contacts=Gw.wall_c.detach()[kw].cpu().numpy())
        np.savez(out/'cells.npz', points=X.cpu().numpy(), radii=R0.cpu().numpy(), masses=V.cpu().numpy())
        print(f"  saved {out} ({len(X)} cells, round {rnd+1}); {int(hot.sum())} cells carry the modes above the target", flush=True)
        if omr[-1] < OM_T:
            break
