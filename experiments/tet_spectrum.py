"""Soft-mode check for a linear tetrahedral sub-mesh of the power cells (+ vertex binding).

Each cell is split into tetrahedra (cell vertex mean, face vertex mean, face edge a-b); the two
virtual points are averages of vertices, so the unknowns are still the cell's vertices (or their
clusters when vertices closer than `tol` are bound). Linear-elastic stiffness = sum over tets of
|T| eps : C : eps with the exact P1 gradient of each tet; lumped mass |T|/4 per tet point.
Reported like sfvem_spectrum.py: omega h / c_s with h = cell volume^(1/3).
"""
import json
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

E_, NU, RHO = 23008.4, .30079, 1000.
LAM, MU = E_*NU/((1+NU)*(1-2*NU)), E_/(2*(1+NU))
CS = np.sqrt(MU/RHO)


def clusters(X, tol):
    if tol <= 0:
        return np.arange(len(X))
    pairs = cKDTree(X).query_pairs(tol, output_type='ndarray')
    g = sparse.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(X),)*2)
    return connected_components(g, directed=False)[1]


def cell_matrices(X, faces, fv, fs, k, cl):
    """faces: list of (face id, sign). Returns K, lumped m over the cell's clusters, volume, tet stats."""
    fvs = [fv[fs[f]:fs[f]+k[f]] for f, _ in faces]
    verts = np.unique(np.concatenate(fvs))
    apex = [(v, 1/len(verts)) for v in verts]
    cls = np.unique(cl[verts]); loc = {c: i for i, c in enumerate(cls)}; n = len(cls)
    K = np.zeros((3*n, 3*n)); m = np.zeros(n); vol = 0.; vols = []
    Ci = np.zeros((6, 6))
    for (f, s), vs in zip(faces, fvs):
        cf = [(v, 1/len(vs)) for v in vs]
        for t in range(len(vs)):
            a, b = vs[t], vs[(t+1) % len(vs)]
            pts = [apex, cf, [(a, 1.)], [(b, 1.)]]
            P = np.array([sum(w*X[v] for v, w in pt) for pt in pts])
            Dm = np.stack((P[1]-P[0], P[2]-P[0], P[3]-P[0]), 1)
            V = s*np.linalg.det(Dm)/6                    # outward-ordered faces -> positive for convex cells
            if abs(V) < 1e-18:
                continue
            Gi = np.linalg.inv(Dm)                       # rows: d(barycentric 1..3)/dX
            grads = np.vstack((-Gi.sum(0), Gi))          # (4, 3) shape-function gradients of the 4 points
            # vertex-level gradient weights (cluster level): g_c = sum over points of weight * grad
            gw = {}
            for pt, gr in zip(pts, grads):
                for v, w in pt:
                    c = loc[cl[v]]; gw[c] = gw.get(c, 0)+w*gr
            idx = np.array(list(gw)); G = np.array([gw[c] for c in idx])   # (q, 3)
            # B: strain (Voigt 6) from cluster displacements (q*3)
            q = len(idx); B = np.zeros((6, 3*q))
            for j in range(q):
                gx, gy, gz = G[j]
                B[:, 3*j:3*j+3] = [[gx, 0, 0], [0, gy, 0], [0, 0, gz], [gy, gx, 0], [0, gz, gy], [gz, 0, gx]]
            D = np.zeros((6, 6)); D[:3, :3] = LAM; D[range(3), range(3)] += 2*MU; D[range(3, 6), range(3, 6)] = MU
            Ke = abs(V)*B.T@D@B
            dofs = (3*idx[:, None]+np.arange(3)).ravel()
            K[np.ix_(dofs, dofs)] += Ke
            for pt in pts:
                for v, w in pt:
                    m[loc[cl[v]]] += RHO*abs(V)/4*w
            vol += V; vols.append(abs(V))
    return .5*(K+K.T), m, vol, np.array(vols)


def main():
    d = np.load('output/pacnerf_vem/elastic_0_f0/vem_cap1_bodies_pyr.npz')
    X, fv, fs, fp, src = d['X'], d['face_vertices'], d['face_start'], d['face_pair'], d['source_index']
    k = np.diff(np.r_[fs, len(fv)])
    cell_of = {int(s): i for i, s in enumerate(src)}
    faces = [[] for _ in src]
    for f, (i, j) in enumerate(fp):
        for side, s_ in ((0, 1.), (1, -1.)):
            c = cell_of.get(int((i, j)[side]))
            if c is not None:
                faces[c].append((f, s_))
    rng = np.random.default_rng(1)
    sample = rng.choice(len(src), 60, replace=False)
    summary = {}
    for tol in (0., 5e-4, 1e-3):
        cl = clusters(X, tol)
        rows = []
        for c in sample:
            K, m, vol, vols = cell_matrices(X, faces[c], fv, fs, k, cl)
            h = vol**(1/3)
            s = 1/np.sqrt(np.repeat(m, 3))
            w2 = np.linalg.eigvalsh(K*s[:, None]*s[None, :])
            w = np.sqrt(np.clip(w2, 0, None))*h/CS
            nonrigid = np.sort(w)[6:]
            rows.append(dict(n_dof=len(m)*3, zero=int((w2 < 1e-10*w2.max()).sum())-6, min=float(nonrigid.min()),
                             p05=float(np.quantile(nonrigid, .05)), below1=int((nonrigid < 1).sum()), max=float(w.max()),
                             min_tet_mm=float(np.cbrt(vols.min())*1e3)))
        r = dict(cells_with_zero_modes=int(sum(x['zero'] > 0 for x in rows)),
                 min_scaled_median=float(np.median([x['min'] for x in rows])), min_scaled_min=float(min(x['min'] for x in rows)),
                 p05_median=float(np.median([x['p05'] for x in rows])), modes_below_1_mean=float(np.mean([x['below1'] for x in rows])),
                 max_scaled_median=float(np.median([x['max'] for x in rows])), max_scaled_max=float(max(x['max'] for x in rows)))
        summary[f'bind {tol*1e3:.1f} mm'] = r
        print(f'bind {tol*1e3:.1f} mm:', json.dumps(r), flush=True)
    json.dump(summary, open('output/pacnerf_vem/elastic_0_f0/tet_spectrum_v1.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
