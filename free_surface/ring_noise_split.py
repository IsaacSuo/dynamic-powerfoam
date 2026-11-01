"""read-only, CPU: where does a false yield come from, inside or on the cap cells, by the reading: |e| of the ring's F; rho^2 = |e|^2 + sum w |r|^2 / (2 V) (the cell's whole distortion energy, the law derived 11-01); and the same with |e|^2 read through M's poorly seen directions weighted down (weight min(1, m / m0), m the eigenvalue of M / 2 V, m0 0.25: uniform strain still reads exactly where the ring sees all round).
Site noise at no strain; then the accepted landing's frames (k1), among the cells k1 has not yielded"""
import numpy as np

g = np.load('output/fourier8k_clean/dem_exact_single/geometry.npz')
X, V, A, pairs = g['points'], g['masses'], g['areas'], g['pairs'].astype(np.int64); n = len(X)
i, j = pairs[:, 0], pairs[:, 1]
D = X[j]-X[i]; L = np.linalg.norm(D, axis=1); nh = D/L[:, None]; w = A/L
Ey = .0095; h = float(np.median(V))**(1/3)
a_ = np.zeros((n, 3)); np.add.at(a_, i, -A[:, None]*nh); np.add.at(a_, j, A[:, None]*nh)
closed = np.linalg.norm(a_, axis=1) < 1e-9*np.median(A)
M = np.zeros((n, 3, 3)); q2 = w[:, None, None]*D[:, :, None]*D[:, None, :]
np.add.at(M, i, q2); np.add.at(M, j, q2)
Mi = np.linalg.pinv(M)
ev, ve = np.linalg.eigh(M/(2*V[:, None, None]))
print('M / 2V eigenvalues: inside smallest 1 / 5 / 50 %% %s; cap smallest %s (1 = a full ring)' % (np.array2string(np.quantile(ev[closed, 0], [.01, .05, .5]), precision=3), np.array2string(np.quantile(ev[~closed, 0], [.01, .05, .5]), precision=3)), flush=True)
wt = np.minimum(1., np.maximum(ev, 0)/.25)
print('directions weighted down: inside cells with any %d of %d; cap %d of %d' % (int((wt[closed].min(1) < 1).sum()), closed.sum(), int((wt[~closed].min(1) < 1).sum()), (~closed).sum()), flush=True)
Wh = (ve*wt[:, None, :])@ve.transpose(0, 2, 1)                                         # (sqrt weights on the reading's material side: e W e)


def read(x):
    d = x[j]-x[i]
    Am = np.zeros((n, 3, 3)); q_ = w[:, None, None]*d[:, :, None]*D[:, None, :]
    np.add.at(Am, i, q_); np.add.at(Am, j, q_)
    F = Am@Mi
    Us, s, Vt = np.linalg.svd(F)
    sg = np.sign(np.linalg.det(Us@Vt)); s = s.copy(); s[:, 2] *= sg; Us = Us.copy(); Us[:, :, 2] *= sg[:, None]
    ls = np.log(np.maximum(s, 1e-12)); lm = ls.mean(1, keepdims=True); dv = ls-lm
    a = np.exp(lm[:, 0])
    e2 = (a*np.linalg.norm(dv, axis=1))**2
    # e in material axes: V dv V^T
    E_ = (Vt.transpose(0, 2, 1)*dv[:, None, :])@Vt
    e2w = (a*a)*np.einsum('nab,nbc,nac->n', E_, Wh, E_)
    ri = d-np.einsum('fab,fb->fa', F[i], D); rj = d-np.einsum('fab,fb->fa', F[j], D)
    Wr = np.zeros(n); np.add.at(Wr, i, w*(ri*ri).sum(1)); np.add.at(Wr, j, w*(rj*rj).sum(1))
    return np.sqrt(e2), np.sqrt(e2+Wr/(2*V)), np.sqrt(e2w+Wr/(2*V)), np.sqrt(Wr/(2*V))


rng = np.random.default_rng(0)
print('site noise, no strain, yield strain %.2f %%: cells over the limit (median of the reading in %%), inside of %d; cap of %d' % (100*Ey, closed.sum(), (~closed).sum()), flush=True)
for an in (.0005, .001, .002, .005):
    x = X+an*h*rng.normal(size=X.shape)
    e, r1_, r2_, rr = read(x)
    print('  %.4f h: |e|: inside %d (%.3f), cap %d (%.3f); rho: inside %d (%.3f), cap %d (%.3f); rho weighted: inside %d (%.3f), cap %d (%.3f); remainder alone median inside %.3f, cap %.3f' % (an, int((e[closed] > Ey).sum()), 100*np.median(e[closed]), int((e[~closed] > Ey).sum()), 100*np.median(e[~closed]), int((r1_[closed] > Ey).sum()), 100*np.median(r1_[closed]), int((r1_[~closed] > Ey).sum()), 100*np.median(r1_[~closed]), int((r2_[closed] > Ey).sum()), 100*np.median(r2_[closed]), int((r2_[~closed] > Ey).sum()), 100*np.median(r2_[~closed]), 100*np.median(rr[closed]), 100*np.median(rr[~closed])), flush=True)
f = np.load('output/spring/frames_k1.npz'); xs, ts, yl = f['x'], f['t'], f['yld']
print('k1 frames, among the cells k1 has not yielded: over the limit by each reading, inside / cap', flush=True)
for t_ in (.15, .1833, .2, .2083, .2167, .225, .2333, .25, .2667):
    k = int(np.argmin(np.abs(ts-t_)))
    e, r1_, r2_, rr = read(xs[k])
    no = ~yl[k].astype(bool)
    y_ = ~no
    print('  t %.4f: k1 has yielded %d; not yet: inside %d, cap %d; |e| over %d / %d; rho over %d / %d; rho weighted over %d / %d; among the yielded, rho weighted over %.1f %%; remainder alone over %d' % (ts[k], int(y_.sum()), int((no & closed).sum()), int((no & ~closed).sum()), int((e[no & closed] > Ey).sum()), int((e[no & ~closed] > Ey).sum()), int((r1_[no & closed] > Ey).sum()), int((r1_[no & ~closed] > Ey).sum()), int((r2_[no & closed] > Ey).sum()), int((r2_[no & ~closed] > Ey).sum()), 100*(r2_[y_] > Ey).mean() if y_.any() else 0., int((rr[no] > Ey).sum())), flush=True)
