"""prescribed uniform shear past yield with the real cut each step: springs at the current walls once both cells have yielded, gone with their wall, born at r = 0. Is the stress capped? sigma / 2 mu = (1 / V) sum_walls (A / L) 1/2 (m_i + m_j) (x) d. No solve: the sites are prescribed; radii for the rest volumes (a shear keeps J = 1)"""
import sys
import time

import numpy as np
import torch
import warp as wp

import free_surface.recut as rc

RULE = sys.argv[1]
wp.init()
f64, dev = torch.float64, 'cuda'
g = np.load('output/fourier8k_clean/dem_exact_single/geometry.npz'); cl = np.load('output/fourier8k_clean/dem_exact_single/cells.npz')
X, V0, r0, pairs0, A0 = g['points'], g['masses'], cl['radii'], g['pairs'].astype(np.int64), g['areas']; n = len(X)
Ey = .0095
Xm = X.mean(0); Xc = X-Xm
Vt = torch.as_tensor(V0, dtype=f64, device=dev); rr0 = torch.as_tensor(r0, dtype=f64, device=dev).contiguous()
R_c = rc.Recut(n, dev); R_c.A.f32 = True
lo, hi = np.minimum(pairs0[:, 0], pairs0[:, 1]), np.maximum(pairs0[:, 0], pairs0[:, 1])
o = np.argsort(lo*n+hi); lo, hi, Aw = lo[o], hi[o], A0[o]
D = X[hi]-X[lo]; w = Aw/np.linalg.norm(D, axis=1)
qi, qj = D.copy(), D.copy()
keys = lo*n+hi
yl = np.zeros(n, dtype=bool)
Fe = np.tile(np.eye(3), (n, 1, 1))
r = rr0.clone()
N = 200; gam_max = 2.
t0 = time.time()
want = {2, 4, 10, 30, 60, 100, 150, 200}
print('steps %d, shear to %.1f; columns: shear; plastic integral (median); |dev sigma| / (2 mu e_y); springs, born, gone in this step; cells yielded; r max / L; radii iterations, repairs' % (N, gam_max), flush=True)
ep = np.zeros(n)
for k in range(1, N+1):
    gam = gam_max*k/N
    F = np.array([[1, gam, 0], [0, 1, 0], [0, 0, 1.]])
    x = Xc@F.T+Xm
    p = torch.as_tensor(x, dtype=f64, device=dev).contiguous()
    r, G, S, it, nrep = R_c.solve_radii(p, r, Vt, tol=1e-5, r_base=rr0)
    kp = (G.wall_A.detach() > 0).cpu().numpy()
    ic, jc, Ac = G.i.cpu().numpy()[kp], G.j.cpu().numpy()[kp], G.wall_A.detach().cpu().numpy()[kp]
    lc, hc = np.minimum(ic, jc), np.maximum(ic, jc); kc = lc*n+hc
    oc = np.argsort(kc); kc, lc, hc, Ac = kc[oc], lc[oc], hc[oc], Ac[oc]
    # the list: both yielded and no wall: gone; a new wall with both yielded: born at r = 0
    both = yl[lo] & yl[hi]
    inc = np.isin(keys, kc)
    stay = ~both | inc
    gone = int((~stay).sum())
    lo, hi, w, qi, qj, keys = lo[stay], hi[stay], w[stay], qi[stay], qj[stay], keys[stay]
    newm = ~np.isin(kc, keys) & yl[lc] & yl[hc]
    if newm.any():
        ln, hn = lc[newm], hc[newm]; dn = x[hn]-x[ln]
        qin = np.einsum('fab,fb->fa', np.linalg.inv(Fe[ln]), dn); qjn = np.einsum('fab,fb->fa', np.linalg.inv(Fe[hn]), dn)
        lo = np.r_[lo, ln]; hi = np.r_[hi, hn]; keys = np.r_[keys, kc[newm]]; w = np.r_[w, np.zeros(len(ln))]; qi = np.r_[qi, qin]; qj = np.r_[qj, qjn]
        o = np.argsort(keys); lo, hi, keys, w, qi, qj = lo[o], hi[o], keys[o], w[o], qi[o], qj[o]
    born = int(newm.sum())
    both = yl[lo] & yl[hi]
    pos = np.searchsorted(kc, keys).clip(max=len(kc)-1); has = kc[pos] == keys
    d = x[hi]-x[lo]; Lc = np.linalg.norm(d, axis=1)
    w = np.where(both & has, Ac[pos]/Lc, w)
    # the read
    Am = np.zeros((n, 3, 3)); M = np.zeros((n, 3, 3))
    np.add.at(Am, lo, w[:, None, None]*d[:, :, None]*qi[:, None, :]); np.add.at(Am, hi, w[:, None, None]*d[:, :, None]*qj[:, None, :])
    np.add.at(M, lo, w[:, None, None]*qi[:, :, None]*qi[:, None, :]); np.add.at(M, hi, w[:, None, None]*qj[:, :, None]*qj[:, None, :])
    Fr = Am@np.linalg.pinv(M)
    Us, s, Vh = np.linalg.svd(Fr); sg = np.sign(np.linalg.det(Us@Vh)); s = s.copy(); s[:, 2] *= sg; Us = Us.copy(); Us[:, :, 2] *= sg[:, None]
    Rr = Us@Vh
    ls = np.log(np.maximum(s, 1e-12)); lm = ls.mean(1, keepdims=True); dv = ls-lm; nr = np.linalg.norm(dv, axis=1); a = np.exp(lm[:, 0])
    ri = d-np.einsum('fab,fb->fa', Fr[lo], qi); rj = d-np.einsum('fab,fb->fa', Fr[hi], qj)
    Wr = np.zeros(n); np.add.at(Wr, lo, w*(ri*ri).sum(1)); np.add.at(Wr, hi, w*(rj*rj).sum(1))
    rho2 = (a*nr)**2+Wr/(2*V0)
    if RULE == 'C':
        be = np.minimum(1., Ey/np.maximum(np.sqrt(rho2), 1e-300)); br = be
    else:
        be = np.minimum(1., Ey/np.maximum(a*nr, 1e-300)); br = be if RULE == 'A' else be*be
    ep += (1-be)*nr
    P = (Vh.transpose(0, 2, 1)*np.exp((1-be)[:, None]*dv)[:, None, :])@Vh
    qi = np.einsum('fab,fb->fa', P[lo], qi)+(1-br[lo])[:, None]*np.einsum('fba,fb->fa', Rr[lo], ri)
    qj = np.einsum('fab,fb->fa', P[hi], qj)+(1-br[hi])[:, None]*np.einsum('fba,fb->fa', Rr[hi], rj)
    yl = yl | (be < 1)
    Ue = (Vh.transpose(0, 2, 1)*np.exp(lm+be[:, None]*dv)[:, None, :])@Vh
    Fe = Rr@Ue
    if k in want:
        mi = d-np.einsum('fab,fb->fa', Rr[lo], qi); mj = d-np.einsum('fab,fb->fa', Rr[hi], qj)
        sig = np.einsum('f,fa,fb->ab', w, .5*(mi+mj), d)/V0.sum()
        Ss = .5*(sig+sig.T); dv_ = Ss-np.trace(Ss)/3*np.eye(3)
        f = np.zeros((n, 3))
        np.add.at(f, hi, w[:, None]*(mi+mj)); np.add.at(f, lo, -w[:, None]*(mi+mj))
        cap = G.patch_A.detach().cpu().numpy() > 0
        print('%.2f; %.3f; %.3f; %d, %d, %d; yielded %d; r / L max %.1e, remainder share of the energy median inside %.3f, 90%% %.3f; radii %d, repairs %d; force on the inside cells / (yield stress x median wall) median %.1e, 99%% %.1e, max %.1e; %.0f s' % (gam, np.median(ep), np.linalg.norm(dv_)/Ey, len(keys), born, gone, int(yl.sum()), float((np.linalg.norm(ri, axis=1)/np.linalg.norm(d, axis=1)).max()), np.median((Wr/(2*V0)/rho2)[~cap]), np.quantile((Wr/(2*V0)/rho2)[~cap], .9), it, nrep, *(np.array([np.median(np.linalg.norm(f[~cap], axis=1)), np.quantile(np.linalg.norm(f[~cap], axis=1), .99), np.linalg.norm(f[~cap], axis=1).max()])/(2*Ey*np.median(Ac))), time.time()-t0), flush=True)
print('done', flush=True)
