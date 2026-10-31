"""the merged form at no limit, rest, on the part, against the voxel FEM (CPU, sparse, no GPU). Energy, quadratic at rest: sum V (2 mu |dev sym G|^2 / 2 + kap / 2 tr(G)^2) + variance kap A L / 24 (tr G_i - tr G_j)^2
+ sum 1/2 c k |r|^2, k = 2 mu A / L, r = du - 1/2 (G_i + G_j) d. G: the divergence-theorem gradient as ul_plastic.py, tiny cells' fit from their neighbours. c the springs' share: 1 the DEM's stiffness, 1/6 PlasticUL's.
Against FEM: the first modes, the Rayleigh quotient along each FEM mode, MAC pairs, spurious modes below half the first. usage: merged_modes.py <part foam dir>"""
import sys, time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from scipy.sparse.linalg import eigsh, splu, LinearOperator
sys.path.insert(0, '/home/fangsuo/py/dynamic-powerfoam')
out = Path(sys.argv[1])
E, NU = 23.00838, .3008
mu = E/(2*(1+NU)); lam = E*NU/((1+NU)*(1-2*NU)); kap = lam+2*mu/3
g = np.load(out/'dem_exact_single'/'geometry.npz')
X, V, A, c = g['points'], g['masses'], g['areas'], g['contacts']; n = len(X)
i, j = g['pairs'][:, 0].astype(np.int64), g['pairs'][:, 1].astype(np.int64); m = len(i)
d = X[j]-X[i]; L = np.linalg.norm(d, axis=1); nh = d/L[:, None]
hi = ((c-X[i])*nh).sum(1); t = hi/L; tau = c-(X[i]+hi[:, None]*nh)
print('%s: %d cells, %d walls, h median %.2f mm' % (out.name, n, m, 1e3*np.median(L)), flush=True)


def coo(rows, cols, vals, shape):
    return sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=shape).tocsr()


# the least-squares fit G0: x -> 9 n
w = A/L
q = w[:, None, None]*d[:, :, None]*d[:, None, :]
M = np.zeros((n, 3, 3)); np.add.at(M, i, q); np.add.at(M, j, q)
Mi = np.linalg.pinv(M)
Ri = np.einsum('fa,fab->fb', w[:, None]*d, Mi[i]); Rj = np.einsum('fa,fab->fb', -w[:, None]*d, Mi[j])
S_, O_, R_ = np.concatenate([i, j]), np.concatenate([j, i]), np.concatenate([Ri, Rj])
rw, cl, vl = [], [], []
for a in range(3):
    for b in range(3):
        ga = 9*S_+3*a+b
        rw += [ga, ga]; cl += [3*O_+a, 3*S_+a]; vl += [R_[:, b], -R_[:, b]]
B0 = coo(rw, cl, vl, (9*n, 3*n))
# tiny cells: the volume-weighted mean of their non-tiny neighbours' fit
tiny = V < .1*np.median(V)
m1, m2 = tiny[i] & ~tiny[j], tiny[j] & ~tiny[i]
as_, ao_ = np.concatenate([i[m1], j[m2]]), np.concatenate([j[m1], i[m2]])
va = V[ao_]; den = np.zeros(n); np.add.at(den, as_, va)
used = tiny & (den > 0); keep = np.where(~used)[0]
rw = [9*np.repeat(keep, 9)+np.tile(np.arange(9), len(keep)), 9*np.repeat(as_, 9)+np.tile(np.arange(9), len(as_))]
cl = [9*np.repeat(keep, 9)+np.tile(np.arange(9), len(keep)), 9*np.repeat(ao_, 9)+np.tile(np.arange(9), len(ao_))]
vl = [np.ones(9*len(keep)), np.repeat(va/den[as_], 9)]
Age = coo(rw, cl, vl, (9*n, 9*n))
# the gradient: V G = sum A u_w (x) n_out + u (x) a + Ge Mc
a_ = np.zeros((n, 3)); np.add.at(a_, i, -A[:, None]*nh); np.add.at(a_, j, A[:, None]*nh)
Mw = np.zeros((n, 3, 3))
np.add.at(Mw, i, A[:, None, None]*(c-X[i])[:, :, None]*nh[:, None, :]); np.add.at(Mw, j, -A[:, None, None]*(c-X[j])[:, :, None]*nh[:, None, :])
Mc = V[:, None, None]*np.eye(3)-Mw
closed = np.linalg.norm(a_, axis=1) < 1e-9*np.median(A)
Mc[closed] = 0.; a_[closed] = 0.
rw, cl, vl = [], [], []
for a in range(3):
    for b in range(3):
        for s_, sg in ((i, 1.), (j, -1.)):
            gs = 9*s_+3*a+b
            rw += [gs, gs]; cl += [3*i+a, 3*j+a]; vl += [sg*A*(1-t)*nh[:, b], sg*A*t*nh[:, b]]
        gs = 9*np.arange(n)+3*a+b
        rw.append(gs); cl.append(3*np.arange(n)+a); vl.append(a_[:, b])
Bx = coo(rw, cl, vl, (9*n, 3*n))
rw, cl, vl = [], [], []
for a in range(3):
    for b in range(3):
        for k_ in range(3):
            for s_, sg in ((i, 1.), (j, -1.)):
                gs = 9*s_+3*a+b
                rw += [gs, gs]; cl += [9*i+3*a+k_, 9*j+3*a+k_]; vl += [.5*sg*A*tau[:, k_]*nh[:, b], .5*sg*A*tau[:, k_]*nh[:, b]]
            gs = 9*np.arange(n)+3*a+b
            rw.append(gs); cl.append(9*np.arange(n)+3*a+k_); vl.append(Mc[:, k_, b])
Bg_ = coo(rw, cl, vl, (9*n, 9*n))
Vi = sp.diags(np.repeat(1/V, 9))
Bg = Vi@(Bx+Bg_@Age@B0)
print('gradient on an affine field: largest error %.1e' % np.abs(Bg@((X@np.array([[.3, .1, -.2], [.05, -.4, .7], [.2, .6, .1]]).T).ravel())-np.tile(np.array([.3, .1, -.2, .05, -.4, .7, .2, .6, .1]), n)).max(), flush=True)
# the cell's energy
I9 = np.eye(3).ravel()
Psym = np.zeros((9, 9))
for a in range(3):
    for b in range(3):
        Psym[3*a+b, 3*a+b] += .5; Psym[3*a+b, 3*b+a] += .5
Q = 2*mu*(Psym-np.outer(I9, I9)/3)+kap*np.outer(I9, I9)
Dc = sp.block_diag([V[s]*Q for s in range(n)], format='csr')
Kc = (Bg.T@Dc@Bg).tocsr()
# the variance
tr = coo([np.arange(n)]*3, [9*np.arange(n), 9*np.arange(n)+4, 9*np.arange(n)+8], [np.ones(n)]*3, (n, 9*n))
dT = sp.coo_matrix((np.concatenate([np.ones(m), -np.ones(m)]), (np.r_[np.arange(m), np.arange(m)], np.r_[i, j])), shape=(m, n)).tocsr()
Kv = ((dT@tr@Bg).T@sp.diags(2*kap*A*L/24)@(dT@tr@Bg)).tocsr()
# the remainder r = du - 1/2 (G_i + G_j) d
e_ = np.arange(m)
rw, cl, vl = [], [], []
for a in range(3):
    rw += [3*e_+a, 3*e_+a]; cl += [3*j+a, 3*i+a]; vl += [np.ones(m), -np.ones(m)]
D1 = coo(rw, cl, vl, (3*m, 3*n))
rw, cl, vl = [], [], []
for a in range(3):
    for b in range(3):
        rw += [3*e_+a, 3*e_+a]; cl += [9*i+3*a+b, 9*j+3*a+b]; vl += [.5*d[:, b], .5*d[:, b]]
D2 = coo(rw, cl, vl, (3*m, 9*n))
C = (D1-D2@Bg).tocsr()
k0 = 2*mu*A/L
Ks1 = (C.T@sp.diags(np.repeat(k0, 3))@C).tocsr()
print('remainder on an affine field: largest %.1e' % np.abs(C@((X@np.array([[.3, .1, -.2], [.05, -.4, .7], [.2, .6, .1]]).T).ravel())).max(), flush=True)
# FEM
from experiments.dem_fem_mode_match import interpolate
c_ = np.load(out/'fem_modes_h0.0015.npz'); nm = min(26, c_['V'].shape[1]); f_fem, V_fem = c_['f'][:nm], c_['V'][:, :nm]
F_ = dict(nodes=c_['nodes'], node_grid=c_['node_grid'], lo=c_['lo'], h=float(c_['h']), masses=c_['masses'])
Ux, _ = interpolate(F_, V_fem, X); UF = Ux.reshape(-1, nm)
print('FEM first modes (Hz): %s' % np.array2string(f_fem[:6], precision=2), flush=True)
Md = sp.diags(np.repeat(V, 3))
w1 = (2*np.pi*f_fem[0])**2


def modes(K, kk=70):
    sigma = -1e-3*w1
    lu = splu((K-sigma*Md).tocsc())
    OP = LinearOperator(K.shape, matvec=lu.solve, dtype=np.float64)
    lam_, vec = eigsh(K, k=kk, M=Md, sigma=sigma, OPinv=OP, which='LM', tol=1e-8)
    o = np.argsort(lam_)
    return lam_[o], vec[:, o]


def report(name, K):
    t0 = time.time()
    lam_, vec = modes(K)
    nz = int((np.abs(lam_) < 1e-6*w1).sum())
    fr = np.sqrt(np.clip(lam_, 0, None))/(2*np.pi)
    Z = vec[:, nz:].reshape(n, 3, -1)
    num = np.einsum('n,nkd,nkf->df', V, Z, Ux)**2
    den = np.outer(np.einsum('n,nkd,nkd->d', V, Z, Z), np.einsum('n,nkf,nkf->f', V, Ux, Ux))
    Mac = num/den
    ri, ci = linear_sum_assignment(-Mac)
    good = [fr[nz+a_]/f_fem[b_] for a_, b_ in zip(ri, ci) if Mac[a_, b_] >= .8]
    sf = np.array([UF[:, k]@(K@UF[:, k])/(UF[:, k]@(Md@UF[:, k]))/(2*np.pi*f_fem[k])**2 for k in range(nm)])
    print('%-26s zeros %d, negative %d; first modes / FEM %s; along the FEM modes, frequency^2 / FEM median %.3f (%.3f..%.3f); paired %d of %d, frequency / FEM median %s; below half the FEM first %d; (%.0f s)' % (
        name, nz, int((lam_ < -1e-6*w1).sum()), np.array2string(fr[nz:nz+6]/f_fem[:6], precision=3), np.median(sf), sf.min(), sf.max(), len(good), nm, '%.3f' % np.median(good) if good else '-', int(((lam_ > 1e-6*w1) & (fr < .5*f_fem[0])).sum()), time.time()-t0), flush=True)


base = (Kc+Kv).tocsr()
report('cells only', base)
for c__, nm_ in ((1/6, 'springs 1/6 (PlasticUL)'), (1/3, 'springs 1/3'), (1/2, 'springs 1/2'), (1., 'springs 1 (the DEM)')):
    report(nm_, (base+c__*Ks1).tocsr())
print('done', flush=True)
