"""Stabilization-free VEM on Power Foam polyhedra: how high must the gradient projection degree be?

For an element E (a power cell, or an agglomerate of cells) with vertex displacements u_v and the
face fan triangulation used everywhere in vem_geometry.py (v linear on each triangle (a, b, c_f),
c_f = face vertex mean), the gradient is projected onto matrix polynomials of degree l
(Chen & Sukumar 2022 / Xu 2024 idea, first-order enhanced space):

    int_E Pi_l grad v : G = int_dE v . G N dS - int_E Pi_1 v . div G dX      for all G in P_l^{3x3}

with the interior moments taken from the affine (l = 0) projection Pi_1 v = vbar + Fbar (X - Xbar)
(Fbar = our current VEM gradient, vbar = vertex mean). Exact for affine v (patch test) for every l.
Linear elastic stiffness K = int_E eps(Pi grad u) : C : eps(Pi grad u); spurious modes = number of
zero eigenvalues - 6. Volume integrals use signed cones from the vertex mean over the boundary
triangles (exact for polynomials, also for non-convex agglomerates).
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np


def monomials(l):
    return [e for d in range(l+1) for e in itertools.product(range(d+1), repeat=3) if sum(e) == d]


def gl(n):
    x, w = np.polynomial.legendre.leggauss(n)
    return .5*(x+1), .5*w


def tri_rule(n=5):
    """Collapsed Gauss rule on the reference triangle (barycentric l0, l1, l2), weights sum to 1/2."""
    x, w = gl(n)
    s, t = np.meshgrid(x, x, indexing='ij'); ws = np.outer(w, w)
    l1 = s*(1-t); l2 = t; l0 = 1-l1-l2
    return np.stack((l0, l1, l2), -1).reshape(-1, 3), (ws*(1-t)).ravel()


def tet_rule(n=5):
    """Collapsed Gauss rule on the reference tetrahedron (barycentric, 4), weights sum to 1/6."""
    x, w = gl(n)
    a, b, c = np.meshgrid(x, x, x, indexing='ij')
    wa = w[:, None, None]*w[None, :, None]*w[None, None, :]
    l3 = c; l2 = b*(1-c); l1 = a*(1-b)*(1-c); l0 = 1-l1-l2-l3
    return np.stack((l0, l1, l2, l3), -1).reshape(-1, 4), (wa*(1-b)*(1-c)**2).ravel()


def element_K(X, tris, l, lam=1., mu=1.):
    """tris: list of (a, b, face_vertex_list, sign); a, b global vertex ids. Returns K over the
    element's vertices (sorted ids), plus diagnostics."""
    verts = sorted({v for a, b, fvs, _ in tris for v in (a, b, *fvs)})
    loc = {v: i for i, v in enumerate(verts)}
    n = len(verts)
    Xe = X[verts]
    Xbar = Xe.mean(0)
    h = np.abs(Xe-Xbar).max()
    mons = monomials(l); nb = len(mons)
    E = np.array(mons)

    def phi(P):                                             # (q, nb) scaled monomials
        Z = (P-Xbar)/h
        return np.prod(Z[:, None, :]**E[None], -1)

    def dphi(P):                                            # (q, nb, 3) gradients
        Z = (P-Xbar)/h
        out = np.zeros((len(P), nb, 3))
        for j in range(3):
            Ej = E.copy(); Ej[:, j] -= 1
            coef = E[:, j]/h
            out[:, :, j] = coef*np.prod(Z[:, None, :]**np.maximum(Ej, 0)[None], -1)*(E[:, j] > 0)
        return out
    tb, tw = tri_rule(5); vb, vw = tet_rule(5)
    M = np.zeros((nb, nb)); vol = 0.
    Ig = np.zeros((nb, 3)); IXg = np.zeros((nb, 3, 3))        # int dphi_k/dX_j ; int (X - Xbar)_m dphi_k/dX_j
    S = np.zeros((nb, 3, n))                                # int_dE lambda_p phi_k N_j  (per vertex p)
    gbar = np.zeros((n, 3))                                 # affine projection weights: Fbar = sum_v u_v (x) gbar_v
    for a, b, fvs, sign in tris:
        c = X[fvs].mean(0)
        pa, pb = X[a], X[b]
        an = sign*.5*np.cross(pa-c, pb-c)                   # outward vector area
        A = np.linalg.norm(an)
        # boundary terms
        P = tb@np.stack((pa, pb, c))
        ph = phi(P)*(tw*2*A)[:, None]                       # dS = 2A dref
        for col, wts in ((0, {loc[a]: 1.}), (1, {loc[b]: 1.}), (2, {loc[v]: 1/len(fvs) for v in fvs})):
            m = ph.T@tb[:, col]                             # (nb,) int lambda phi dS
            for vi, wv in wts.items():
                S[:, :, vi] += wv*np.outer(m, an/A)
        for vi, wv in ((loc[a], 1/3), (loc[b], 1/3)):
            gbar[vi] += wv*an
        for v in fvs:
            gbar[loc[v]] += an/(3*len(fvs))
        # signed cone from Xbar over this triangle
        T = np.stack((Xbar, pa, pb, c))
        J = np.linalg.det(np.stack((pa-Xbar, pb-Xbar, c-Xbar)))
        Q = vb@T; wq = vw*J*sign                           # reversed faces: reversed cone orientation
        f, df = phi(Q), dphi(Q)
        M += (f*wq[:, None]).T@f
        vol += wq.sum()
        Ig += np.einsum('q,qkj->kj', wq, df)
        IXg += np.einsum('q,qm,qkj->kmj', wq, Q-Xbar, df)
    if vol < 0:                                             # orientation of the cone volumes
        M, Ig, IXg, vol = -M, -Ig, -IXg, -vol
    gbar /= vol
    # Pi grad u = sum_k phi_k C_k, C_k (3x3) = sum_l Minv_kl [ sum_v u_v (x) S[l, :, v] - interior_l ]
    # interior_l (i, j) = int (Pi_1 u)_i d_j phi_l = ubar_i Ig[l, j] + sum_m Fbar_im IXg[l, m, j]
    # map from u (n*3) to C (nb*9): C[k, i, j] = sum_v u_{v,i} R[k, j, v]
    R = S.copy()
    R -= Ig[:, :, None]/n                                   # ubar = mean of vertex displacements
    R -= np.einsum('lmj,vm->ljv', IXg, gbar)               # Fbar_im = sum_v u_{v,i} gbar_{v,m}
    cond = np.linalg.cond(M)
    Rk = np.linalg.solve(M, R.reshape(nb, -1)).reshape(nb, 3, n)
    # stiffness: energy = sum_kl M_kl [lam tr(eps_k) tr(eps_l) + 2 mu eps_k : eps_l], eps_k = sym(C_k)
    # build B: (nb, 3, 3, n, 3): C_k[i, j] derivative wrt u_{v, i'} = delta_{i i'} Rk[k, j, v]
    B = np.zeros((nb, 3, 3, n, 3))
    for i in range(3):
        B[:, i, :, :, i] = Rk
    Bs = .5*(B+B.transpose(0, 2, 1, 3, 4))
    tr = np.einsum('kiivd->kvd', Bs).reshape(nb, -1)
    Bf = Bs.reshape(nb, 9, -1)
    K = lam*tr.T@M@tr
    for a_ in range(9):
        K += 2*mu*Bf[:, a_, :].T@M@Bf[:, a_, :]
    # patch check: affine displacement reproduces a constant gradient
    G = np.array([[.1, .2, -.1], [.05, -.2, .3], [.0, .1, .15]])
    u = (Xe-Xbar)@G.T
    C = np.einsum('kjv,vi->kij', Rk, u)
    patch = np.abs(C[0]-G).max()+np.abs(C[1:]).max() if nb > 1 else np.abs(C[0]-G).max()
    return K, dict(n_vertices=n, n_basis=nb, volume=vol, cond_M=cond, patch_err=float(patch))


def spurious(K):
    w = np.linalg.eigvalsh(.5*(K+K.T))
    return int((np.abs(w) < 1e-10*w.max()).sum())-6, float(np.sort(w)[6]/w.max())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--geometry', type=Path, default=Path('output/pacnerf_vem/elastic_0_f0/vem_cap1_bodies_pyr.npz'))
    p.add_argument('--samples', type=int, default=40)
    p.add_argument('--degrees', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    p.add_argument('--out', type=Path, default=Path('output/pacnerf_vem/elastic_0_f0/sfvem_modes.json'))
    args = p.parse_args()
    d = np.load(args.geometry)
    X, fv, fs, fp = d['X'], d['face_vertices'], d['face_start'], d['face_pair']
    src, cell_elem = d['source_index'], d['cell_elem']
    k = np.diff(np.r_[fs, len(fv)])
    cell_of = {int(s): i for i, s in enumerate(src)}
    faces_of_cell = [[] for _ in range(len(src))]
    for f_, (i, j) in enumerate(fp):
        for side, s_ in ((0, 1.), (1, -1.)):
            c = cell_of.get(int((i, j)[side]))
            if c is not None:
                faces_of_cell[c].append((f_, s_))
    rng = np.random.default_rng(0)

    def tris_of(face_list):
        out = []
        for f_, s_ in face_list:
            vs = fv[fs[f_]:fs[f_]+k[f_]]
            for t in range(len(vs)):
                out.append((int(vs[t]), int(vs[(t+1) % len(vs)]), vs, s_))
        return out
    results = {}
    cells = rng.choice(len(src), args.samples, replace=False)
    bodies = rng.choice(cell_elem.max()+1, args.samples, replace=False)
    groups = {'cell': [tris_of(faces_of_cell[c]) for c in cells]}
    body_tris = []
    for b in bodies:
        members = set(np.flatnonzero(cell_elem == b).tolist())
        fl = []
        for c in members:
            for f_, s_ in faces_of_cell[c]:
                other = fp[f_][1] if s_ > 0 else fp[f_][0]
                if cell_of.get(int(other)) not in members:   # internal faces of the agglomerate cancel
                    fl.append((f_, s_))
        body_tris.append(tris_of(fl))
    groups['agglomerate'] = body_tris
    for name, tlist in groups.items():
        results[name] = {}
        for l in args.degrees:
            rows = []
            for tris in tlist:
                K, info = element_K(X, tris, l)
                sp, smallest = spurious(K)
                rows.append(dict(info, spurious=sp, smallest_nonzero_over_max=smallest))
            sp = np.array([r['spurious'] for r in rows]); nv = np.array([r['n_vertices'] for r in rows])
            r = dict(degree=l, basis=rows[0]['n_basis'], elements=len(rows), vertices_median=float(np.median(nv)),
                     vertices_max=int(nv.max()), clean_fraction=float(np.mean(sp <= 0)), spurious_max=int(sp.max()),
                     spurious_mean=float(sp.mean()), cond_M_max=float(max(r_['cond_M'] for r_ in rows)),
                     patch_err_max=float(max(r_['patch_err'] for r_ in rows)),
                     smallest_nonzero_median=float(np.median([r_['smallest_nonzero_over_max'] for r_ in rows])),
                     max_vertices_clean=int(nv[sp <= 0].max()) if (sp <= 0).any() else 0,
                     min_vertices_dirty=int(nv[sp > 0].min()) if (sp > 0).any() else None)
            results[name][l] = r
            print(name, json.dumps(r), flush=True)
            args.out.write_text(json.dumps(results, indent=1))


if __name__ == '__main__':
    main()
