"""The cells' consistent mass for ClosureWarp (11-06; the clean model's 10-21 version: KB_DERIVATION.md 21.3).

Each cell's material moves with the affine field of its site and of the one-way cap's closure without the free-surface
completion (positions only, linear):
    v(X) = u_C + L_C(u) (X - X_C),   L_C(u) = (S_C(u) + G_C(u) Mcap_C) Mt_C^-1,
    S_C = sum_walls 1/2 A (u_j - u_i) (x) n  (k_S_bonds on u),   G_C = sum_e (u_k - u_C) (x) dX_e M2_C^-1  (k_closP's fit),
exact for every affine u (S_C = W Mb_C, G_C = W, Mt = Mb + Mcap). Its kinetic energy is
    T = 1/2 sum_C int_C rho |v|^2 = 1/2 u^T (Mh (x) I3) u,   L_C = sum_k u_k a_Ck^T,
    Mh_kl = sum_C [ m_C e_kC e_lC + m_C (e_kC a_Cl . d_C + e_lC a_Ck . d_C) + a_Ck^T J_C a_Cl ],
d_C = centroid - X_C and J_C = int rho r r^T (r = X - X_C) of the cell's material (its power cell inside its sphere),
sampled on a grid and scaled to the cell's volume V_C (unit density, as the model's mass). Mh is constant (rest
coefficients), symmetric positive definite, one scalar matrix for the three components; it carries the rotational
inertia exactly. The consistent weight is Mh g per site (row sums times g): the same total force and moment.
Chain cells (set_chain: rigid, with their own rotation) keep m_C on the diagonal only.

Why (scratch/consistent_mass_check.py, PAC-NeRF elastic scene 0): the lumped mass sits at the sites, and a surface
cell's site lies about one cell size inward of its material (median 17.6 mm; interior 4.2 mm). The modal mass along the
voxel FEM's modes was 0.84-0.88 (modes 9+ down to 0.68), so the modes ran 5-9 % fast. With Mh the modal mass is
1.005-1.05, and Rayleigh-Ritz gives 0.976 (modes 0-8) and 1.010 (modes 0-25) of the FEM's frequencies; the inertia about
the centre of mass equals the material's. Mh's weak directions (sites far inside their material) make diag-preconditioned
CG 5x slower; TrustCG preconditions the sites with the dense (Mh / (th dt)^2 + kbar)^-1 instead (13 iterations against
the lumped mass's 16 to 1e-6, scratch/cm_pcg_check.py)."""
import hashlib
import os

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree


def cell_moments(p, r, V, spacing=1.5e-3, cache=None, verbose=True):
    """d (n, 3) = centroid - site and J (n, 3, 3) = int r r^T about the site, per unit density with the masses V, of
    each cell's material (its power cell inside its sphere), from a grid of `spacing`; cache: an npz path, keyed by the
    geometry (written atomically, so parallel runs may share it)"""
    p, r, V = np.asarray(p, np.float64), np.asarray(r, np.float64), np.asarray(V, np.float64)
    n = len(p)
    key = hashlib.sha1(np.concatenate([p.ravel(), r, [spacing]]).tobytes()).hexdigest()
    if cache is not None and os.path.exists(cache):
        z = np.load(cache)
        if str(z['key']) == key:
            return z['d'], z['Jm']*V[:, None, None]
    Rm = r.max(); tree = cKDTree(np.c_[p, np.sqrt(Rm*Rm-r*r)])           # (power distance as a lifted Euclidean one)
    lo, hi = (p-r[:, None]).min(0), (p+r[:, None]).max(0)
    ax = [np.arange(lo[k]+spacing/2, hi[k], spacing) for k in range(3)]
    cnt = np.zeros(n); S1 = np.zeros((n, 3)); S2 = np.zeros((n, 3, 3))
    for ia in range(0, len(ax[0]), 8):
        G = np.stack(np.meshgrid(ax[0][ia:ia+8], ax[1], ax[2], indexing='ij'), -1).reshape(-1, 3)
        _, own = tree.query(np.c_[G, np.zeros(len(G))], workers=-1)
        q = ((G-p[own])**2).sum(1) <= r[own]**2                             # (inside its own sphere: the material)
        G, own = G[q], own[q]; rr = G-p[own]
        cnt += np.bincount(own, minlength=n)
        for k in range(3):
            S1[:, k] += np.bincount(own, rr[:, k], n)
            for l in range(k, 3):
                S2[:, k, l] += np.bincount(own, rr[:, k]*rr[:, l], n)
    for k in range(3):
        for l in range(k):
            S2[:, k, l] = S2[:, l, k]
    cs = np.maximum(cnt, 1)
    d, Jm = S1/cs[:, None], S2/cs[:, None, None]
    d[cnt == 0] = 0.; Jm[cnt == 0] = 0.
    if verbose:
        vs = cnt*spacing**3
        print('  cell moments: sampled volume %.5f vs sum V %.5f (per cell 1-99 %% %.3f-%.3f), %d cells under 20 samples, %d without'
              % (vs.sum(), V.sum(), *np.percentile(vs/V, [1, 99]), (cnt < 20).sum(), (cnt == 0).sum()), flush=True)
    if cache is not None:
        tmp = '%s.%d.tmp.npz' % (cache, os.getpid())
        np.savez(tmp, key=key, d=d, Jm=Jm, cnt=cnt, spacing=spacing)
        os.replace(tmp, cache)
    return d, Jm*V[:, None, None]


def mass_matrix(W, d, J):
    """Mh (n x n, scipy CSR) of a ClosureWarp W after set_cap_positions (and set_chain if any), per unit density"""
    st, n, P = W.st, W.n, W._pos
    c = lambda t: t.detach().cpu().numpy()
    m = c(st.V).astype(np.float64)
    chain = np.zeros(n, bool)
    if W.chain is not None:
        chain[c(W.chain)] = True
    bi, bj, A, nrm = c(st.i).astype(np.int64), c(st.j).astype(np.int64), c(st.A), c(st.nrm)
    w = .5*A[:, None]*nrm
    rc, rk, vecs = [bi, bi, bj, bj], [bj, bi, bj, bi], [w, -w, w, -w]      # S_C: both cells of a wall get 1/2 A (u_j - u_i) (x) n
    off, pk, dX, M2i = c(P['off']), c(P['pk']).astype(np.int64), c(P['dX']), c(P['M2i'])
    Mc, Mti = c(st.Mcap), c(st.Mt_inv)
    pc = np.repeat(np.arange(n), np.diff(off))
    cv = np.einsum('eij,ej->ei', Mc[pc], np.einsum('eij,ej->ei', M2i[pc], dX))   # G_C Mcap: (u_k - u_C) (x) Mcap M2^-1 dX_e
    rc += [pc, pc]; rk += [pk, pc]; vecs += [cv, -cv]
    C, K, Vv = np.concatenate(rc), np.concatenate(rk), np.concatenate(vecs)
    keep = ~chain[C]
    C, K, Vv = C[keep], K[keep], Vv[keep]
    uk, inv = np.unique(C*n+K, return_inverse=True)
    a = np.stack([np.bincount(inv, Vv[:, k], len(uk)) for k in range(3)], 1)
    aC, aK = uk//n, uk % n
    a = np.einsum('eij,ej->ei', Mti[aC], a)                                 # a_Ck = Mt_C^-1 (...) (Mt symmetric)
    X = c(st.X)
    err = max(np.abs(np.stack([np.bincount(aC, a[:, k], n) for k in range(3)], 1)).max(),
              max(np.abs(np.bincount(aC, X[aK, k]*a[:, l], n)-(k == l))[~chain].max() for k in range(3) for l in range(3)))
    assert err < 1e-8, 'consistent mass: L_C not exact for affine motion (%.1e)' % err
    starts = np.searchsorted(aC, np.arange(n+1))
    ri, ci, vv = [], [], []
    for Cc in range(n):
        if chain[Cc]:
            ri.append([Cc]); ci.append([Cc]); vv.append([m[Cc]])
            continue
        s0, s1 = starts[Cc], starts[Cc+1]
        ks, Ac = aK[s0:s1], a[s0:s1]
        e = (ks == Cc).astype(np.float64)
        Ad = Ac@d[Cc]
        blk = Ac@J[Cc]@Ac.T+m[Cc]*(np.outer(e, Ad)+np.outer(Ad, e)+np.outer(e, e))
        ri.append(np.repeat(ks, len(ks))); ci.append(np.tile(ks, len(ks))); vv.append(blk.ravel())
    Mh = sp.coo_matrix((np.concatenate(vv), (np.concatenate(ri), np.concatenate(ci))), shape=(n, n)).tocsr()
    Mh = (.5*(Mh+Mh.T)).tocsr()
    Mh.sum_duplicates(); Mh.sort_indices()
    return Mh


def mass_matrix_cols(W, d, J):
    """F1 (ClosureWarp.set_cap_columns): Mh over the n sites and the ncap cap columns (ndof x ndof, scipy CSR), per unit
    density. The material map x_C + F_C (X - X_C), F_C = (S_C + (G_C P_C + c_C m_C^T) Mc_C) Mt_C^-1, is linear in (x, c):
    L_C(u, cdot) = sum_k u_k a_Ck^T + cdot_C a_Cc^T with the fit's coefficients now through P (G P Mc) and the column's
    a_Cc = Mt^-1 Mc m. Exact for every affine motion with c's reference value m (u = W X + b, cdot = W m: L_C = W)"""
    st, n, P = W.st, W.n, W._pos
    N = W.ndof
    c = lambda t: t.detach().cpu().numpy()
    m = c(st.V).astype(np.float64)
    bi, bj, A, nrm = c(st.i).astype(np.int64), c(st.j).astype(np.int64), c(st.A), c(st.nrm)
    w = .5*A[:, None]*nrm
    rc, rk, vecs = [bi, bi, bj, bj], [bj, bi, bj, bi], [w, -w, w, -w]
    off, pk, dX, M2i = c(P['off']), c(P['pk']).astype(np.int64), c(P['dX']), c(P['M2i'])
    Mc, Mti, PA, mA = c(st.Mcap), c(st.Mt_inv), c(st.PA), c(st.mA)
    pc = np.repeat(np.arange(n), np.diff(off))
    cv = np.einsum('eij,ej->ei', Mc[pc], np.einsum('eij,ej->ei', PA[pc], np.einsum('eij,ej->ei', M2i[pc], dX)))   # (G P Mc: Mc P M2^-1 dX_e)
    cells = c(W.cap_cells).astype(np.int64)
    cidx = n+np.arange(len(cells))
    rc += [pc, pc, cells]; rk += [pk, pc, cidx]; vecs += [cv, -cv, np.einsum('eij,ej->ei', Mc[cells], mA[cells])]
    C, K, Vv = np.concatenate(rc), np.concatenate(rk), np.concatenate(vecs)
    uk, inv = np.unique(C*N+K, return_inverse=True)
    a = np.stack([np.bincount(inv, Vv[:, k], len(uk)) for k in range(3)], 1)
    aC, aK = uk//N, uk % N
    a = np.einsum('eij,ej->ei', Mti[aC], a)
    Xd = np.concatenate([c(st.X), mA[cells]], 0); tr = np.concatenate([np.ones(n), np.zeros(len(cells))])
    err = max(np.abs(np.stack([np.bincount(aC, tr[aK]*a[:, k], n) for k in range(3)], 1)).max(),
              max(np.abs(np.bincount(aC, Xd[aK, k]*a[:, l], n)-(k == l)).max() for k in range(3) for l in range(3)))
    assert err < 1e-8, 'consistent mass (F1): L_C not exact for affine motion (%.1e)' % err
    starts = np.searchsorted(aC, np.arange(n+1))
    ri, ci, vv = [], [], []
    for Cc in range(n):
        s0, s1 = starts[Cc], starts[Cc+1]
        ks, Ac = aK[s0:s1], a[s0:s1]
        e = (ks == Cc).astype(np.float64)
        Ad = Ac@d[Cc]
        blk = Ac@J[Cc]@Ac.T+m[Cc]*(np.outer(e, Ad)+np.outer(Ad, e)+np.outer(e, e))
        ri.append(np.repeat(ks, len(ks))); ci.append(np.tile(ks, len(ks))); vv.append(blk.ravel())
    Mh = sp.coo_matrix((np.concatenate(vv), (np.concatenate(ri), np.concatenate(ci))), shape=(N, N)).tocsr()
    Mh = (.5*(Mh+Mh.T)).tocsr()
    Mh.sum_duplicates(); Mh.sort_indices()
    return Mh
