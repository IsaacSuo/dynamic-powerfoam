"""Stabilization-free VEM elements on Power Foam cells (precomputation).

Elements are the material power cells (cells below --merge-frac x median volume are merged into
the neighbour sharing the largest face first; merged elements may be non-convex). On element E
with N vertices the gradient is projected onto matrix polynomials of degree l(E), the smallest l
with 6 dim P_l >= 3N - 6 (enough strain freedoms for every non-rigid vertex motion; checked
against spurious-mode counts in sfvem_modes.py):

    int_E Pi grad v : G = int_dE v . G N dS - int_E Pi_1 v . div G dX,   G in P_l^{3x3},

v linear on the face fan triangles (a, b, c_f), interior moments from the affine projection.
Stored per element: R (nb, 3, N) with  F(X) = sum_k phi_k(X) C_k,  C_k[i, j] = sum_v (x_v - xbar)_i R[k, j, v]
(constant fields have zero gradient, so centring by the vertex mean is exact), phi_k scaled
monomials about the vertex mean. Energy quadrature: one point per boundary-triangle cone from the
vertex mean (centroid, signed cone volume) -- many points, so the quadratic form keeps its rank.
Lumped mass: each cone's mass split equally over its 4 points (vertex mean -> all element
vertices, face centre -> face vertices).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from experiments.sfvem_modes import monomials, tet_rule, tri_rule
from experiments.vem_geometry import merge_small_cells


def degree_for(n, lmax=5):
    for l in range(lmax+1):
        if 6*len(monomials(l)) >= 3*n-6:
            return l
    return lmax


def element_data(X, tris, l):
    verts = sorted({v for a, b, fvs, _ in tris for v in (a, b, *fvs)})
    loc = {v: i for i, v in enumerate(verts)}
    n = len(verts)
    Xe = X[verts]; Xbar = Xe.mean(0); h = np.abs(Xe-Xbar).max()
    E = np.array(monomials(l)); nb = len(E)

    def phi(P):
        Z = (P-Xbar)/h
        return np.prod(Z[:, None, :]**E[None], -1)

    def dphi(P):
        Z = (P-Xbar)/h
        out = np.zeros((len(P), nb, 3))
        for j in range(3):
            Ej = E.copy(); Ej[:, j] -= 1
            out[:, :, j] = (E[:, j]/h)*np.prod(Z[:, None, :]**np.maximum(Ej, 0)[None], -1)*(E[:, j] > 0)
        return out
    tb, tw = tri_rule(5); vb, vw = tet_rule(5)
    M = np.zeros((nb, nb)); vol = 0.
    Ig = np.zeros((nb, 3)); IXg = np.zeros((nb, 3, 3)); S = np.zeros((nb, 3, n)); gbar = np.zeros((n, 3))
    qp, qw, mass_rows = [], [], []
    for a, b, fvs, sign in tris:
        c = X[fvs].mean(0); pa, pb = X[a], X[b]
        an = sign*.5*np.cross(pa-c, pb-c); A = np.linalg.norm(an)
        if A > 0:
            P = tb@np.stack((pa, pb, c)); ph = phi(P)*(tw*2*A)[:, None]
            for col, wts in ((0, {loc[a]: 1.}), (1, {loc[b]: 1.}), (2, {loc[v]: 1/len(fvs) for v in fvs})):
                m = ph.T@tb[:, col]
                for vi, wv in wts.items():
                    S[:, :, vi] += wv*np.outer(m, an/A)
        gbar[loc[a]] += an/3; gbar[loc[b]] += an/3
        for v in fvs:
            gbar[loc[v]] += an/(3*len(fvs))
        T = np.stack((Xbar, pa, pb, c))
        J = sign*np.linalg.det(np.stack((pa-Xbar, pb-Xbar, c-Xbar)))
        Q = vb@T; wq = vw*J
        f, df = phi(Q), dphi(Q)
        M += (f*wq[:, None]).T@f; vol += wq.sum()
        Ig += np.einsum('q,qkj->kj', wq, df); IXg += np.einsum('q,qm,qkj->kmj', wq, Q-Xbar, df)
        cv = J/6                                            # signed cone volume
        qp.append(T.mean(0)); qw.append(cv)
        mass_rows.append((cv, loc[a], loc[b], [loc[v] for v in fvs]))
    gbar /= vol
    R = S-Ig[:, :, None]/n-np.einsum('lmj,vm->ljv', IXg, gbar)
    Rk = np.linalg.solve(M, R.reshape(nb, -1)).reshape(nb, 3, n)
    qp = np.array(qp); qw = np.array(qw)
    mv = np.zeros(n)
    for cv, ia, ib, ifs in mass_rows:
        mv[ia] += cv/4; mv[ib] += cv/4
        mv[ifs] += cv/(4*len(ifs))
        mv += cv/(4*n)
    # checks: affine patch, quadrature integrates phi exactly? (one point per cone: only degree <= 1)
    G = np.array([[.1, .2, -.1], [.05, -.2, .3], [.0, .1, .15]])
    C = np.einsum('kjv,vi->kij', Rk, (Xe-Xbar)@G.T)
    patch = np.abs(C[0]-G).max()+(np.abs(C[1:]).max() if nb > 1 else 0.)
    return dict(verts=np.array(verts), Xbar=Xbar, h=h, exps=E, R=Rk, qp=qp, qw=qw, phi_q=phi(qp), mass=mv, M=M,
                volume=vol, cond=float(np.linalg.cond(M)), patch=float(patch))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--geometry', type=Path, default=Path('output/pacnerf_vem/elastic_0_f0/vem_cap1_bodies_pyr.npz'))
    p.add_argument('--cells', type=Path, default=Path('output/pacnerf_dem/elastic_0_f0/cells.npz'))
    p.add_argument('--merge-frac', type=float, default=.1)
    p.add_argument('--lmax', type=int, default=5)
    p.add_argument('--limit', type=int, default=0, help='only the first N elements (testing)')
    p.add_argument('--out', type=Path, default=Path('output/pacnerf_vem/elastic_0_f0/sfvem_cells.npz'))
    args = p.parse_args()
    d = np.load(args.geometry); cells = dict(np.load(args.cells))
    X, fv, fs, fp, src = d['X'], d['face_vertices'], d['face_start'], d['face_pair'], d['source_index']
    k = np.diff(np.r_[fs, len(fv)])
    group, merges = merge_small_cells(cells['masses'], cells['pairs'], cells['areas'], np.zeros(len(src), int), args.merge_frac)
    ne = group.max()+1
    gid = {int(s): int(group[i]) for i, s in enumerate(src)}
    faces = [[] for _ in range(ne)]
    for f_, (i, j) in enumerate(fp):
        gi, gj = gid.get(int(i), -1), gid.get(int(j), -1)
        if gi == gj:
            continue
        if gi >= 0:
            faces[gi].append((f_, 1.))
        if gj >= 0:
            faces[gj].append((f_, -1.))
    t0 = time.perf_counter()
    out = dict(verts=[], R=[], phi_q=[], qw=[], qp=[], mass=[], Xbar=[], h=[], degree=[], nv=[], nq=[], volume=[])
    stats = dict(cond=[], patch=[], negative_weights=0, negative_weight_volume=0.)
    for e in range(args.limit or ne):
        tris = []
        for f_, s_ in faces[e]:
            vs = fv[fs[f_]:fs[f_]+k[f_]]
            for t in range(len(vs)):
                tris.append((int(vs[t]), int(vs[(t+1) % len(vs)]), vs, s_))
        n = len({v for a, b, fvs, _ in tris for v in (a, b, *fvs)})
        l = degree_for(n, args.lmax)
        r = element_data(X, tris, l)
        for key in ('verts', 'R', 'phi_q', 'qw', 'qp', 'mass'):
            out[key].append(r[key])
        out['Xbar'].append(r['Xbar']); out['h'].append(r['h']); out['degree'].append(l)
        out['nv'].append(len(r['verts'])); out['nq'].append(len(r['qw'])); out['volume'].append(r['volume'])
        stats['cond'].append(r['cond']); stats['patch'].append(r['patch'])
        neg = r['qw'] < 0
        stats['negative_weights'] += int(neg.sum()); stats['negative_weight_volume'] += float(-r['qw'][neg].sum())
        if e % 500 == 0:
            print(f'{e}/{ne} elements, {time.perf_counter()-t0:.0f}s', flush=True)
    deg = np.array(out['degree']); nv = np.array(out['nv']); nq = np.array(out['nq'])
    summary = dict(elements=int(len(deg)), merges=int(merges), degree_counts={int(l): int((deg == l).sum()) for l in np.unique(deg)},
                   vertices_median=float(np.median(nv)), vertices_max=int(nv.max()), points_median=float(np.median(nq)),
                   points_total=int(nq.sum()), coefficient_rows=int(sum(len(monomials(l))*n for l, n in zip(deg, nv))),
                   cond_max=float(max(stats['cond'])), cond_median=float(np.median(stats['cond'])), patch_max=float(max(stats['patch'])),
                   negative_weights=stats['negative_weights'], negative_weight_volume_share=stats['negative_weight_volume']/float(np.sum(out['volume'])),
                   volume_total=float(np.sum(out['volume'])), seconds=time.perf_counter()-t0)
    print(json.dumps(summary, indent=1), flush=True)
    off_v = np.r_[0, np.cumsum(nv)]; off_q = np.r_[0, np.cumsum(nq)]
    nbs = np.array([len(monomials(l)) for l in deg]); off_k = np.r_[0, np.cumsum(nbs*3*nv)]; off_p = np.r_[0, np.cumsum(nq*nbs)]
    np.savez_compressed(args.out, X=X, verts=np.concatenate(out['verts']), off_v=off_v, degree=deg, nb=nbs,
                        R=np.concatenate([r.ravel() for r in out['R']]), off_R=off_k,
                        phi_q=np.concatenate([q.ravel() for q in out['phi_q']]), off_phi=off_p,
                        qw=np.concatenate(out['qw']), qp=np.concatenate(out['qp']), off_q=off_q,
                        mass_vol=np.concatenate(out['mass']), volume=np.array(out['volume']), Xbar=np.array(out['Xbar']),
                        group=group, source_index=src)
    args.out.with_suffix('.json').write_text(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
