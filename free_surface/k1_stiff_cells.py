"""Why k1's surface cells are stiff: its gradient written out as G_s = sum_k u_k (x) c_sk (u = x - x_n, linear within a step),
term by term from the arrays update_ul builds (ul_plastic.py k_Gul / k_Aul / k_Ge / k_gg_wall / k_gg_cell):
  G0_s = sum_e (u_o - u_s) (x) R_e                      the least-squares fit (tiny cells: Ge = sum Aw G0 of non-tiny neighbours)
  V_s G_s = sum_w (+-) A [(1 - t) u_i + t u_j] (x) n     'wall'  (the wall's motion interpolated along the site segment)
          + sum_w (+-) A [1/2 (Ge_i + Ge_j) tau] (x) n   'tau'   (its in-plane offset carried by the fits)
          + u_s (x) a_s                                  'open'  (a_s = -sum A n_out: the exposed cap moves with the site)
          + Ge_s Mc_s                                    'cap'   (Mc = V I - sum A (c - x_s) (x) n_out: the cap's moment, through the fit)
A site k's stiffness from the cells' energy V (2 mu |dev sym G|^2 / 2 + kap / 2 (tr G)^2) is about (lam + 2 mu) sum_s V_s |c_sk|^2
(the column of k), plus the springs' k = mu A / 3 L on their two sites ('spring'), so omega_k^2 ~ that / m_k; each term's share is
its part alone over the sum of the parts alone (cross terms left out).
Checked against Lanczos's top cells (k1_explicit_dt.py: 3744, 2680, 329, 3485, 3502, 611, ...).
Run: PYTHONPATH=$PWD python -u free_surface/k1_stiff_cells.py"""
import json
import math
from pathlib import Path

import numpy as np
import torch
import warp as wp

from free_surface.merged_ul import MergedUL

f64, dev = torch.float64, 'cuda'
foam = Path('foams/fourier8k_clean')
gt = json.loads((foam/'plasticine_scenes.json').read_text())['scenes']['0']
E, nu, ys = gt['E']/1000, gt['nu'], gt['ys']/1000
g = np.load(foam/'dem_exact_single'/'geometry.npz'); cl = np.load(foam/'dem_exact_single'/'cells.npz')
X0, r0, V0 = g['points'], cl['radii'], g['masses']; n = len(X0)
W = MergedUL(X0, r0, V0, g['pairs'], g['areas'], g['contacts'], E, nu, ys, theta=.5, kshare=1/6, grad='gg')
W.set_ball((0., 1., 0.), -.05, 10.)
W.update_ul(W.X)
t = W.tul; ml, m = W._ml, W._w[5]
T = lambda k_: wp.to_torch(k_, requires_grad=False) if not torch.is_tensor(k_) else k_
S, O, R = t['S'][:2*ml].long(), t['O'][:2*ml].long(), t['R'][:2*ml]
As, Ao, Aw = t['As'].long(), t['Ao'].long(), t['Aw']; used = t['used'].bool()
wi, wj, wa, wn, wt, wtau = t['wi'][:m].long(), t['wj'][:m].long(), t['wa'][:m], t['wn'][:m], t['wt'][:m], t['wtau'][:m]
a_, Mc, Vi = t['a'], t['Mc'], t['Vi']
V = 1/Vi; lam_2mu = W.kap+4*W.mu/3
key = lambda s_, k_: s_*n+k_


def fit_entries():
    """G0_s = sum (u_k) (x) g: triplets (s, k, g)"""
    s_ = torch.cat([S, S]); k_ = torch.cat([O, S]); v_ = torch.cat([R, -R])
    return s_, k_, v_


s0, k0, g0 = fit_entries()
# Ge: the fit, or for a used (tiny) cell the weighted fits of its partners
keep = ~used[s0]
sa, ka, ga = [s0[keep]], [k0[keep]], [g0[keep]]
ok = Aw[:len(As)] != 0
As_, Ao_, Aw_ = As[ok], Ao[ok], Aw[:len(As)][ok]
# expand: for each aggregation entry (s <- o, w), every fit triplet of o
order = torch.argsort(s0); s0o, k0o, g0o = s0[order], k0[order], g0[order]
cnt = torch.bincount(s0o, minlength=n); off = torch.cat([torch.zeros(1, dtype=torch.long, device=dev), torch.cumsum(cnt, 0)])
reps = cnt[Ao_]
src = torch.repeat_interleave(torch.arange(len(Ao_), device=dev), reps)
pos = off[Ao_][src]+(torch.arange(int(reps.sum()), device=dev)-torch.repeat_interleave(torch.cumsum(reps, 0)-reps, reps))
sa.append(As_[src]); ka.append(k0o[pos]); ga.append(Aw_[src][:, None]*g0o[pos])
se, ke, ge = torch.cat(sa), torch.cat(ka), torch.cat(ga)                            # Ge_s = sum u_k (x) ge
# per cell lists of Ge's triplets (for the tau and cap terms)
o2 = torch.argsort(se); se, ke, ge = se[o2], ke[o2], ge[o2]
cntE = torch.bincount(se, minlength=n); offE = torch.cat([torch.zeros(1, dtype=torch.long, device=dev), torch.cumsum(cntE, 0)])


def expand(cells):
    """all Ge triplets of the given cells: (row index into cells, k, ge)"""
    r_ = cntE[cells]
    idx = torch.repeat_interleave(torch.arange(len(cells), device=dev), r_)
    p_ = offE[cells][idx]+(torch.arange(int(r_.sum()), device=dev)-torch.repeat_interleave(torch.cumsum(r_, 0)-r_, r_))
    return idx, ke[p_], ge[p_]


terms = {}
# 'wall': V_s G_s += +A [(1-t) u_i + t u_j] (x) n for s = i, -... for s = j
sw = torch.cat([wi, wi, wj, wj]); kw = torch.cat([wi, wj, wi, wj])
cw = torch.cat([(wa*(1-wt))[:, None]*wn, (wa*wt)[:, None]*wn, -(wa*(1-wt))[:, None]*wn, -(wa*wt)[:, None]*wn])
terms['wall'] = (sw, kw, cw)
# 'open': u_s (x) a_s
ids = torch.arange(n, device=dev)
terms['open'] = (ids, ids, a_)
# 'tau': +-A/2 (Ge_i + Ge_j) tau (x) n: (u_k (x) ge) tau = u_k (ge . tau) -> c = (ge . tau) A/2 n
st_, kt_, ct_ = [], [], []
for side, sgn in ((wi, 1.), (wj, 1.)):
    idx, kk, gg = expand(side)
    coef = (gg*wtau[idx]).sum(1)*wa[idx]/2
    for cell, sg2 in ((wi, 1.), (wj, -1.)):
        st_.append(cell[idx]); kt_.append(kk); ct_.append((sg2*coef)[:, None]*wn[idx])
terms['tau'] = (torch.cat(st_), torch.cat(kt_), torch.cat(ct_))
# 'cap': Ge_s Mc_s = sum u_k (x) (Mc^T ge)
cells_cap = torch.nonzero(Mc.abs().sum((1, 2)) > 0).flatten()
idx, kk, gg = expand(cells_cap)
terms['cap'] = (cells_cap[idx], kk, torch.einsum('fba,fb->fa', Mc[cells_cap[idx]], gg))


def column_stiffness(trip):
    """per site k: (lam + 2 mu) sum_s V_s |c_sk|^2 with c_sk = (sum of the triplets for (s, k)) / V_s"""
    s_, k_, c_ = trip
    u, inv = torch.unique(key(s_, k_), return_inverse=True)
    c = torch.zeros(len(u), 3, dtype=f64, device=dev).index_add(0, inv, c_)
    ss, kk = u//n, u % n
    c = c*Vi[ss][:, None]
    return torch.zeros(n, dtype=f64, device=dev).index_add(0, kk, lam_2mu*V[ss]*(c*c).sum(1))


allt = tuple(torch.cat([terms[k_][i_] for k_ in terms]) for i_ in range(3))
mass = torch.as_tensor(V0, dtype=f64, device=dev)
Ktot = column_stiffness(allt); omega = torch.sqrt(Ktot/mass)
parts = {k_: column_stiffness(terms[k_]) for k_ in terms}
ms_ = W._m_s; sk = W.tsp['sk'][:ms_]                                                 # the springs' direct stiffness on their two sites
parts['spring'] = torch.zeros(n, dtype=f64, device=dev).index_add(0, W.tlo, sk).index_add(0, W.thi, sk)
Ktot = Ktot+parts['spring']; omega = torch.sqrt(Ktot/mass)
h = mass**(1/3); Vmed = float(mass.median())
cap = Mc.abs().sum((1, 2)) > 0
q = lambda x_, p: float(torch.quantile(x_, p))
print(f"k1's gradient written out ({len(allt[0])} triplets); column-stiffness estimate omega_k: median {q(omega, .5):.3e}, 99 % {q(omega, .99):.3e}, "
      f"99.9 % {q(omega, .999):.3e}, max {float(omega.max()):.3e} (Lanczos max 4.78e4, ul_blocks median 2.33e3, max 5.50e4)", flush=True)
for grp, msk in (('inside', ~cap), ('surface', cap)):
    print(f"  {grp} ({int(msk.sum())} cells): omega median {q(omega[msk], .5):.3e}, 99 % {q(omega[msk], .99):.3e}; share of the stiffness by term "
          "(each term alone over the sum of the terms alone, median / at the 1 % stiffest): " +
          ", ".join(f"{k_} {q((parts[k_]/sum(parts.values()))[msk], .5):.2f} / "
                    f"{float((parts[k_]/sum(parts.values()))[msk & (omega >= q(omega[msk], .99))].median()):.2f}" for k_ in parts), flush=True)
top = [3744, 2680, 329, 3485, 3502, 611, 5051, 219, 3817, 3541, 5046, 4565]
print("Lanczos's top cells: omega estimate; each term alone as omega; V / median; |a| / V (1/m); |Mc| / V; fit weights max |R| h", flush=True)
Rmax = torch.zeros(n, dtype=f64, device=dev).scatter_reduce(0, S, R.norm(dim=1), 'amax')
for c_ in top:
    print(f"  {c_}: {float(omega[c_]):.2e}; " + ", ".join(f"{k_} {math.sqrt(float(parts[k_][c_]/mass[c_])):.2e}" for k_ in parts) +
          f"; V {float(mass[c_])/Vmed:.2f}; |a|/V {float(a_[c_].norm()*Vi[c_]):.1f}; |Mc|/V {float(Mc[c_].norm()*Vi[c_]):.2f}; "
          f"|R| h {float(Rmax[c_]*h[c_]):.2f}; used (tiny) {bool(used[c_])}", flush=True)
rank = torch.argsort(omega, descending=True)[:40].tolist()
print(f"overlap of the 40 highest estimates with Lanczos's top cells: {len(set(rank) & set(top))} of {len(top)}", flush=True)
