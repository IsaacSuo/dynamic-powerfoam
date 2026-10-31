"""does the DEM's spring energy split into a cell part and a remainder? Small strain, real foam, fields vanishing near the surface (so every cell that moves is closed).
Springs: E_s = sum 1/2 k |a|^2, k = 2 mu A / L, a = du - Wbar d (Wbar the two cells' mean spin). Gradient G_i from the divergence theorem, V_i G_i = sum A u_w (x) n_out, u_w at the wall's centroid on the site line.
Cell part: E_c = sum V_i mu |sym G_i|^2 (per unit volume over the deep cells; the springs over the deep walls). Remainder: E_r = sum 1/2 k |r|^2, r = du - Gbar d, which needs no rotation of its own. Cross term: E_x = sum k (ebar d) . r. E_s = (sum 1/2 k |ebar d|^2) + E_x + E_r."""
import numpy as np
g = np.load('plasticine_foam_geometry.npz')
X, V, A, c = g['points'], g['masses'], g['areas'], g['contacts']
i, j = g['pairs'][:, 0].astype(np.int64), g['pairs'][:, 1].astype(np.int64)
n = len(X); mu = 1.
d = X[j]-X[i]; L = np.linalg.norm(d, axis=1); nh = d/L[:, None]
hi = ((c-X[i])*nh).sum(1); t = hi/L; k = 2*mu*A/L
front = np.zeros(n); np.add.at(front, i, A*hi/3); np.add.at(front, j, A*(L-hi)/3)
surf = (V-front)/V > 1e-3
ring = np.where(surf, 0, 10**6)
for _ in range(6):
    r2 = ring.copy(); np.minimum.at(r2, j, ring[i]+1); np.minimum.at(r2, i, ring[j]+1); ring = r2
free = ring >= 2
print('cells %d, surface %d, moving cells (2 rings in) %d, walls among any %d' % (n, surf.sum(), free.sum(), len(i)))


w = A/L
q = w[:, None, None]*d[:, :, None]*d[:, None, :]
M = np.zeros((n, 3, 3)); np.add.at(M, i, q); np.add.at(M, j, q)
Mi = np.linalg.pinv(M)
Ri = np.einsum('fa,fab->fb', w[:, None]*d, Mi[i]); Rj = np.einsum('fa,fab->fb', -w[:, None]*d, Mi[j])
tau = c-(X[i]+hi[:, None]*nh)
deepw = (ring[i] >= 2) & (ring[j] >= 2)
deepc = ring >= 3
Vw, Vc = (A*L/3)[deepw].sum(), V[deepc].sum()
print('deep walls %d, their volume A L / 3 %.5f; deep cells %d, their volume %.5f' % (deepw.sum(), Vw, deepc.sum(), Vc))


def split(u):
    du = u[j]-u[i]
    G0 = np.zeros((n, 3, 3)); np.add.at(G0, i, du[:, :, None]*Ri[:, None, :]); np.add.at(G0, j, (-du)[:, :, None]*Rj[:, None, :])
    uw = (1-t)[:, None]*u[i]+t[:, None]*u[j]+.5*np.einsum('fab,fb->fa', G0[i]+G0[j], tau)
    S = A[:, None, None]*uw[:, :, None]*nh[:, None, :]
    G = np.zeros((n, 3, 3)); np.add.at(G, i, S); np.add.at(G, j, -S); G /= V[:, None, None]
    e = .5*(G+G.transpose(0, 2, 1)); W = .5*(G-G.transpose(0, 2, 1))
    ebar, Wbar = .5*(e[i]+e[j]), .5*(W[i]+W[j])
    a = du-np.einsum('fab,fb->fa', Wbar, d)
    ed = np.einsum('fab,fb->fa', ebar, d)
    r = a-ed
    f = deepw
    Es = (.5*k*(a*a).sum(1))[f].sum()/Vw
    Ee = (.5*k*(ed*ed).sum(1))[f].sum()/Vw
    Ex = (k*(ed*r).sum(1))[f].sum()/Vw
    Er = (.5*k*(r*r).sum(1))[f].sum()/Vw
    Ec = (V*mu*(e*e).sum((1, 2)))[deepc].sum()/Vc
    rel = (np.linalg.norm(r, axis=1)/L)[f]; ae = (np.linalg.norm(ed, axis=1)/L)[f]
    return Es, Ec, Ee, Ex, Er, np.median(rel)/max(np.median(ae), 1e-30)


rng = np.random.default_rng(0)
xc = X.mean(0)
Q = np.linalg.qr(rng.normal(size=(3, 3)))[0]; e0 = Q@np.diag([2, -1, -1])@Q.T/6**.5
h = np.median(V)**(1/3)
cases = {}
cases['uniform strain'] = (X-xc)@e0.T
for lam in (40, 16, 8, 4):
    kk = rng.normal(size=(4, 3)); kk = 2*np.pi*kk/np.linalg.norm(kk, axis=1, keepdims=True)/(lam*h); aa = rng.normal(size=(4, 3))
    cases['smooth waves, wavelength %d cell sizes' % lam] = sum(aa[m]*np.sin((X*kk[m]).sum(1))[:, None] for m in range(4))*.01*lam*h
cases['uniform strain + site noise, 1 %% of h for strain 1 %%'] = (X-xc)@e0.T*.01+rng.normal(size=X.shape)*.01*h
cases['uniform strain + site noise, 10 %% of h for strain 1 %%'] = (X-xc)@e0.T*.01+rng.normal(size=X.shape)*.1*h
cases['site noise alone'] = rng.normal(size=X.shape)*h
print('per unit volume, E_c + E_r against E_s and its parts; the remainder against the shape part of the reading, median')
for name, u in cases.items():
    Es, Ec, Ee, Ex, Er, rr = split(u)
    print('%-52s (E_c + E_r) / E_s %.4f; E_c / E_s %.4f, E_r / E_s %.4f, cross / E_s %+.4f, springs on the shape part / E_c %.4f; r / shape %.3f' % (name, (Ec+Er)/Es, Ec/Es, Er/Es, Ex/Es, Ee/max(Ec, 1e-300), rr))
