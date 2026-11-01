"""read-only, CPU, seconds per state: linear buckling of the elastic end under a uniform compression, the surface sites held. K = K_ref + G; K_ref the stiffness without any geometric part (springs at rest, lam c (x) c, the variance), positive; G what the compressed state adds. G z = theta K_ref z: K has a negative mode when theta < -1. Separately: the geometric part of the springs, of the pressure. Where is the lowest mode, on what kind of cell.
python ring_buckle.py <foam>"""
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, eigsh, splu

sys.path.insert(0, '/home/fangsuo/py/dynamic-powerfoam')
from free_surface.tl_check import build

group = {}

foam = Path(sys.argv[1])
tc_ = float(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] != 'none' else None
agg = float(sys.argv[3]) if len(sys.argv) > 3 else 0.
Dg = build(foam, tc=tc_); Bg = Dg['Bg']
X, V, A, L, i, j, n, m, closed = Dg['X'], Dg['V'], Dg['A'], Dg['L'], Dg['i'], Dg['j'], Dg['n'], Dg['m'], Dg['closed']
# the pressure's cells: a very small cell joined with its neighbour across its largest wall until the volume is agg of the median: V_s F_s = sum V_i F_i, so a uniform pressure's force still cancels across every wall; the wall inside a joined cell drops out
par = np.arange(n)


def root(a):
    while par[a] != a:
        par[a] = par[par[a]]; a = par[a]
    return a


Vs_ = V.copy(); med = np.median(V)
wall_of = [[] for _ in range(n)]
for w_, (a_i, a_j) in enumerate(zip(i, j)):
    wall_of[a_i].append(w_); wall_of[a_j].append(w_)
for s in np.argsort(V):
    if agg <= 0:
        break
    while True:
        r_ = root(s)
        if Vs_[r_] >= agg*med:
            break
        members = group.get(r_, [r_])
        best, bA = -1, -1.
        for a in members:
            for w_ in wall_of[a]:
                o_ = j[w_] if i[w_] == a else i[w_]
                if root(o_) != r_ and A[w_] > bA:
                    bA, best = A[w_], o_
        if best < 0:
            break
        r2 = root(best)
        par[r_] = r2; Vs_[r2] += Vs_[r_]
        group[r2] = group.get(r2, [r2])+members; group.pop(r_, None)
sup_ = np.array([root(a) for a in range(n)])
uq, sid = np.unique(sup_, return_inverse=True); ns = len(uq)
Vsup = np.bincount(sid, weights=V, minlength=ns)
ro = (9*sid[:, None]+np.arange(9)).ravel(); co = (9*np.arange(n)[:, None]+np.arange(9)).ravel(); va = np.repeat(V/Vsup[sid], 9)
Sg = sp.coo_matrix((va, (ro, co)), shape=(9*ns, 9*n)).tocsr()
Bs = (Sg@Bg).tocsr()
ks_ = sid[i] != sid[j]
print('t limit %s; joined: %d cells in %d pressure cells, smallest %.3f of the median' % (tc_, n, ns, Vsup.min()/med), flush=True)
D = X[j]-X[i]
E_m = 23.00838
Ad = sp.coo_matrix((np.ones(2*m, dtype=bool), (np.r_[i, j], np.r_[j, i])), shape=(n, n)).tocsr()
depth = np.full(n, -1); depth[~closed] = 0; front = depth == 0; k_ = 0
while (depth < 0).any() and front.any():
    k_ += 1
    nx_ = ((Ad@front.astype(np.int64)) > 0) & (depth < 0)
    depth[nx_] = k_; front = nx_
free = np.where(depth >= 1)[0]; fd = (3*free[:, None]+np.arange(3)).ravel()
nw = np.bincount(np.r_[i, j], minlength=n)
hm = float(np.median(V))**(1/3)
AL = np.zeros(n); np.add.at(AL, i, A*L); np.add.at(AL, j, A*L)
Bn = np.sqrt(np.asarray(Bs.multiply(Bs).sum(1)).ravel()).reshape(ns, 9).max(1)[sid]*hm
print('%s: %d cells, free %d; V / median 0.1, 1, 50 %% %s; below 0.1 median %d (free %d); sum A L / V median %.1f, 99 %% %.1f, max %.1f; gg row x h median %.2f, 99 %% %.2f, max %.2f' % (foam.name, n, len(free), np.array2string(np.quantile(V/np.median(V), [.001, .01, .5]), precision=3), int((V < .1*np.median(V)).sum()), int((V[free] < .1*np.median(V)).sum()), np.median(AL/V), np.quantile(AL/V, .99), (AL/V).max(), np.median(Bn), np.quantile(Bn, .99), Bn.max()), flush=True)
wm0 = A/L
e_ = np.arange(m)
rw, cl_, vl = [], [], []
for a in range(3):
    rw += [3*e_+a, 3*e_+a]; cl_ += [3*j+a, 3*i+a]; vl += [np.ones(m), -np.ones(m)]
C1 = sp.coo_matrix((np.concatenate(vl), (np.concatenate(rw), np.concatenate(cl_))), shape=(3*m, 3*n)).tocsr()
Dx = np.zeros((m, 3, 3)); Dx[:, 0, 1], Dx[:, 0, 2], Dx[:, 1, 0], Dx[:, 1, 2], Dx[:, 2, 0], Dx[:, 2, 1] = -D[:, 2], D[:, 1], D[:, 2], -D[:, 0], -D[:, 1], D[:, 0]
Mr = np.zeros((n, 3, 3)); q2 = wm0[:, None, None]*D[:, :, None]*D[:, None, :]
np.add.at(Mr, i, q2); np.add.at(Mr, j, q2)
Mri = np.linalg.pinv(Mr)
H3 = np.zeros((n, 3, 3)); q3 = wm0[:, None, None]*(Dx.transpose(0, 2, 1)@Dx)
np.add.at(H3, i, q3); np.add.at(H3, j, q3)
rows, cols, vals = [], [], []; rows2, cols2, vals2 = [], [], []
for s_ in (i, j):
    v_ = np.einsum('fab,fb->fa', Mri[s_], D)
    Vx = np.zeros((m, 3, 3)); Vx[:, 0, 1], Vx[:, 0, 2], Vx[:, 1, 0], Vx[:, 1, 2], Vx[:, 2, 0], Vx[:, 2, 1] = -v_[:, 2], v_[:, 1], v_[:, 2], -v_[:, 0], -v_[:, 1], v_[:, 0]
    Tb = .5*wm0[:, None, None]*Vx
    Cc = wm0[:, None, None]*Dx
    for a in range(3):
        for b in range(3):
            rows += [3*s_+a, 3*s_+a]; cols += [3*j+b, 3*i+b]; vals += [Tb[:, a, b], -Tb[:, a, b]]
            rows2 += [3*s_+a, 3*s_+a]; cols2 += [3*j+b, 3*i+b]; vals2 += [Cc[:, a, b], -Cc[:, a, b]]
TF = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(3*n, 3*n)).tocsr()
Cr0 = sp.coo_matrix((np.concatenate(vals2), (np.concatenate(rows2), np.concatenate(cols2))), shape=(3*n, 3*n)).tocsr()
Ks0 = (C1.T@sp.diags(np.repeat(2*wm0, 3))@C1).tocsr()
Hi = sp.block_diag([np.linalg.pinv(H3[s_]) for s_ in range(n)], format='csr')
Kr0 = (Ks0-Cr0.T@Hi@Cr0).tocsr()                                                       # (each cell's rotation relaxed, at rest)
dT = sp.coo_matrix((np.r_[np.ones(m), -np.ones(m)], (np.r_[e_, e_], np.r_[sid[i], sid[j]])), shape=(m, ns)).tocsr()
Tr9 = np.zeros((9, 9))
trm = sp.coo_matrix((np.ones(3*ns), (np.repeat(np.arange(ns), 3), (9*np.arange(ns)[:, None]+np.array([0, 4, 8])).ravel())), shape=(ns, 9*ns)).tocsr()


def lowest(G, lu, Kref, k=6):
    """G z = theta K_ref z, the lowest theta"""
    G_ = .5*(G+G.T).tocsr()
    Op = LinearOperator(G_.shape, matvec=lambda z: lu.solve(G_@z), dtype=np.float64)
    # K_ref-self-adjoint: Lanczos in K_ref's inner product
    th, z = eigsh(G_, k=k, M=Kref, Minv=LinearOperator(G_.shape, matvec=lu.solve, dtype=np.float64), which='SA', tol=1e-5, maxiter=2000)
    o = np.argsort(th)
    return th[o], z[:, o]


for nu in (.3008, .3913):
    mu_m, lam_m = E_m/(2*(1+nu)), E_m*nu/((1+nu)*(1-2*nu))
    K_s0 = mu_m*Kr0
    Sk = mu_m*Ks0
    print('nu %.4f, lam / mu %.2f; theta < -1: a negative mode. Springs with rotation relaxed: (K_0 - (1 - s) L) / s; pressure on gg\'s F' % (nu, lam_m/mu_m), flush=True)
    for label, s in (('iso 0.95', .95), ('iso 0.9', .9), ('iso 0.85', .85), ('iso 0.8', .8), ('iso 0.75', .75), ('iso 0.7', .7), ('iso 0.6', .6)):
        lnJ = 3*np.log(s)
        c = np.array([1/s, 0, 0, 0, 1/s, 0, 0, 0, 1/s])
        for a in range(3):
            for b in range(3):
                Tr9[3*a+b, 3*b+a] = 1/(s*s)
        K_l0 = (Bs.T@sp.block_diag([Vsup[s_]*lam_m*np.outer(c, c) for s_ in range(ns)], format='csr')@Bs).tocsr()
        K_g = (Bs.T@sp.block_diag([Vsup[s_]*lam_m*(-lnJ)*Tr9 for s_ in range(ns)], format='csr')@Bs).tocsr()
        ctr = (1/s)*trm
        K_v = ((dT@ctr@Bs).T@sp.diags(2*lam_m*A*L/24)@(dT@ctr@Bs)).tocsr()
        Kref = (K_s0/s+K_l0+K_v).tocsr()[fd][:, fd]
        lu = splu(Kref.tocsc())
        Gs = (-(1-s)/s*Sk).tocsr()[fd][:, fd]
        Gp = K_g[fd][:, fd]
        th_a, z_a = lowest((Gs+Gp).tocsr(), lu, Kref)
        th_s, _ = lowest(Gs, lu, Kref, k=2)
        th_p, z_p = lowest(Gp, lu, Kref)
        z = z_a[:, 0].reshape(-1, 3); wt_ = (z*z).sum(1); wt_ /= wt_.sum()
        top = np.argsort(-wt_)[:3]; cell = free[top]
        print('  %-9s J %.3f: theta all %.3f (%.3f, %.3f), below -1: %d of 6; springs\' part alone %.3f, pressure\'s alone %.3f; the lowest mode on %.1f sites, top share %s; V / median %s, walls %s, A L / V %s, gg row x h %s' % (label, s**3, th_a[0], th_a[1], th_a[2], int((th_a < -1).sum()), th_s[0], th_p[0], 1/(wt_*wt_).sum(), np.array2string(wt_[top], precision=2), np.array2string(V[cell]/np.median(V), precision=3), nw[cell].tolist(), np.array2string((AL/V)[cell], precision=1), np.array2string(Bn[cell], precision=1)), flush=True)
