"""read-only, CPU: the ring's F on the elastic lion's real frames (r1), q = rest, the surface cells above all. F read from F = A M^-1 blew up on the older walls-only form (10-29); R from polar(A) needs no inverse. Per frame, by cap and inside: the smallest singular value and det of the ring's F, the angle between R(F), R(A) and gg's R (r1's own F), the strain read; the springs' spread"""
import sys

import numpy as np

foam, frames = sys.argv[1], sys.argv[2]
g = np.load(foam+'/dem_exact_single/geometry.npz')
X, V, A, pairs = g['points'], g['masses'], g['areas'], g['pairs'].astype(np.int64); n = len(X)
i, j = pairs[:, 0], pairs[:, 1]; m = len(i)
D = X[j]-X[i]; L = np.linalg.norm(D, axis=1); nh = D/L[:, None]; w = A/L
a_ = np.zeros((n, 3)); np.add.at(a_, i, -A[:, None]*nh); np.add.at(a_, j, A[:, None]*nh)
closed = np.linalg.norm(a_, axis=1) < 1e-9*np.median(A)
M = np.zeros((n, 3, 3)); q2 = w[:, None, None]*D[:, :, None]*D[:, None, :]
np.add.at(M, i, q2); np.add.at(M, j, q2)
ev = np.linalg.eigvalsh(M)/V[:, None]
nw = np.bincount(np.r_[i, j], minlength=n)
sing = ev[:, 0] < 1e-8*ev[:, 2]
print('walls per cell: min %d, median %d; cells with at most 2 walls %d, 3 walls %d; M singular (rank below 3) %d, of them walls %s; volume of those / median %s' % (nw.min(), np.median(nw), int((nw <= 2).sum()), int((nw == 3).sum()), int(sing.sum()), np.bincount(nw[sing]).tolist(), np.array2string((V[sing]/np.median(V))[:10], precision=3)), flush=True)
print('M / V, smallest eigenvalue, of 2 (a full ring reads 2): quantiles 0.1, 1, 5, 50 %% %s; cap only %s' % (np.array2string(np.quantile(ev[:, 0], [.001, .01, .05, .5]), precision=3), np.array2string(np.quantile(ev[~closed, 0], [.001, .01, .05, .5]), precision=3)), flush=True)
Mi = np.linalg.pinv(M)
ev = np.maximum(ev, 1e-12*ev[:, 2:])
print('%s: %d cells, %d closed; M / V condition: inside median %.2f, 99 %% %.1f, max %.1f; cap median %.2f, 99 %% %.1f, max %.1f' % (foam, n, closed.sum(), np.median((ev[:, 2]/ev[:, 0])[closed]), np.quantile((ev[:, 2]/ev[:, 0])[closed], .99), (ev[:, 2]/ev[:, 0])[closed].max(), np.median((ev[:, 2]/ev[:, 0])[~closed]), np.quantile((ev[:, 2]/ev[:, 0])[~closed], .99), (ev[:, 2]/ev[:, 0])[~closed].max()), flush=True)
f = np.load(frames)
xs, Fs, ts = f['x'], f['F'], f['t']
print('frames %d, t %.3f .. %.3f' % (len(ts), ts[0], ts[-1]), flush=True)


def polar(F):
    Us, s, Vt = np.linalg.svd(F)
    sg = np.sign(np.linalg.det(Us@Vt)); s = s.copy(); s[:, 2] *= sg; Us = Us.copy(); Us[:, :, 2] *= sg[:, None]
    return Us@Vt, s


def ang(R1, R2):
    return np.degrees(np.arccos(np.clip((np.trace(R1.transpose(0, 2, 1)@R2, axis1=1, axis2=2)-1)/2, -1, 1)))


step = max(1, len(ts)//24)
for k in range(0, len(ts), step):
    x = xs[k]
    d = x[j]-x[i]
    Am = np.zeros((n, 3, 3)); q_ = w[:, None, None]*d[:, :, None]*D[:, None, :]
    np.add.at(Am, i, q_); np.add.at(Am, j, q_)
    F = Am@Mi
    Rf, sf = polar(F); Ra, _ = polar(Am); Rg, sg_ = polar(Fs[k])
    dt = np.linalg.det(F)
    lg = np.log(np.maximum(np.abs(sf), 1e-12)); dv = lg-lg.mean(1, keepdims=True); ef = np.linalg.norm(dv, axis=1)
    lg2 = np.log(np.maximum(np.abs(sg_), 1e-12)); eg = np.linalg.norm(lg2-lg2.mean(1, keepdims=True), axis=1)
    c = ~closed
    print('t %.3f: inside: det F min %.3f, sigma_min min %.3f, sigma_max max %.2f, R(F) vs gg 50/99/max %.1f/%.1f/%.1f deg, R(A) vs R(F) 50/99/max %.1f/%.1f/%.1f, strain ring / gg median %.3f / %.3f; cap: det F min %.3f, inverted %d, sigma_min min %.3f, sigma_max max %.2f, R(F) vs gg 50/99/max %.1f/%.1f/%.1f, R(A) vs gg 50/99/max %.1f/%.1f/%.1f, R(A) vs R(F) 50/99/max %.1f/%.1f/%.1f; strain ring / gg median %.3f / %.3f' % (
        ts[k], dt[closed].min(), sf[closed].min(), sf[closed].max(), *np.quantile(ang(Rf, Rg)[closed], [.5, .99, 1]), *np.quantile(ang(Ra, Rf)[closed], [.5, .99, 1]), np.median(ef[closed]), np.median(eg[closed]),
        dt[c].min(), int((dt[c] <= 0).sum()), sf[c].min(), sf[c].max(), *np.quantile(ang(Rf, Rg)[c], [.5, .99, 1]), *np.quantile(ang(Ra, Rg)[c], [.5, .99, 1]), *np.quantile(ang(Ra, Rf)[c], [.5, .99, 1]), np.median(ef[c]), np.median(eg[c])), flush=True)
