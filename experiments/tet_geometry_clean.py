"""Tetrahedral sub-mesh of the Power Foam cells without slivers (geometric cleanup, not physics patches).

Thin tets in tet_geometry.py came from the face fan about the face centre: 99% sat on flat face
triangles (centre, a, b), 2/3 of them on short edges a-b, 1/3 on elongated faces; none on flat
cells. Here, before cutting:
 1. vertices closer than --weld are merged geometrically (midpoint of the cluster) -- free vertices,
    unlike run-time binding (which shared one translation and resisted rigid rotation);
    consecutive duplicates are removed, faces with < 3 vertices and empty cells are dropped;
 2. each face polygon gets the triangulation with the largest smallest-altitude among the centre fan
    and the k vertex fans (shared by both cells of the face, so the sub-mesh stays conforming);
 3. tets (cell vertex mean, triangle) with exact P1 gradients; degenerate ones dropped.
Output has the tet_cells.npz layout (TetSolid). Reports thin-tet statistics and checks.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def weld(X, tol):
    pairs = cKDTree(X).query_pairs(tol, output_type='ndarray')
    g = sparse.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(X),)*2)
    n, lab = connected_components(g, directed=False)
    Y = np.zeros((n, 3)); np.add.at(Y, lab, X); Y /= np.bincount(lab)[:, None]
    return lab, Y


def min_altitude(P):
    """P (..., 3 points, 3): smallest altitude of each triangle."""
    a, b, c = P[..., 0, :], P[..., 1, :], P[..., 2, :]
    area2 = np.linalg.norm(np.cross(b-a, c-a), axis=-1)
    longest = np.maximum.reduce([np.linalg.norm(b-a, axis=-1), np.linalg.norm(c-b, axis=-1), np.linalg.norm(a-c, axis=-1)])
    return area2/np.maximum(longest, 1e-30)


def best_triangulation(Y, vs):
    """vs: cyclic vertex ids. Returns list of triangles as tuples of point codes:
    ('v', id) for a vertex, ('c',) for the face centre; and the chosen scheme."""
    k = len(vs)
    P = Y[vs]; c = P.mean(0)
    best, choice = -1., None
    # centre fan
    tris = np.stack((np.repeat(c[None], k, 0), P, np.roll(P, -1, 0)), 1)
    q = min_altitude(tris).min()
    if q > best:
        best, choice = q, ('centre', None)
    if k > 3:
        for s in range(k):                                   # fan from vertex s
            idx = [(s+i) % k for i in range(k)]
            tris = np.stack([np.stack((P[idx[0]], P[idx[i]], P[idx[i+1]])) for i in range(1, k-1)])
            q = min_altitude(tris).min()
            if q > best:
                best, choice = q, ('vertex', s)
    elif k == 3:
        q = min_altitude(P[None]).min()
        if q >= best:
            best, choice = q, ('vertex', 0)
    if choice[0] == 'centre':
        return [(('c',), ('v', vs[t]), ('v', vs[(t+1) % k])) for t in range(k)], 'centre', best
    s = choice[1]; idx = [(s+i) % k for i in range(k)]
    return [(('v', vs[idx[0]]), ('v', vs[idx[i]]), ('v', vs[idx[i+1]])) for i in range(1, k-1)], 'vertex', best


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--geometry', type=Path, default=Path('output/pacnerf_vem/elastic_0_f0/vem_cap1_bodies_pyr.npz'))
    p.add_argument('--weld', type=float, default=5e-4)
    p.add_argument('--min-height', type=float, default=5e-5, help='drop tets thinner than this (m), and inverted ones')
    p.add_argument('--out', type=Path, default=None)
    args = p.parse_args()
    out = args.out or Path(f'output/pacnerf_vem/elastic_0_f0/tet_clean_w{args.weld*1e3:g}mm.npz')
    d = np.load(args.geometry)
    X0, fv, fs, fp, src = d['X'], d['face_vertices'], d['face_start'], d['face_pair'], d['source_index']
    k0 = np.diff(np.r_[fs, len(fv)])
    lab, Y = weld(X0, args.weld) if args.weld > 0 else (np.arange(len(X0)), X0.copy())
    nc = len(src)
    cell_of = -np.ones(max(fp.max(), src.max())+1, int); cell_of[src] = np.arange(nc)
    faces = []                                               # (welded cycle, pair cells)
    dropped = 0
    for f in range(len(fs)):
        vs = lab[fv[fs[f]:fs[f]+k0[f]]]
        keep = vs != np.roll(vs, 1)
        vs = vs[keep]
        if len(vs) < 3 or len(np.unique(vs)) < 3:
            dropped += 1; continue
        faces.append((vs, cell_of[fp[f, 0]], cell_of[fp[f, 1]]))
    # cell apex = mean of its (welded) vertices
    cell_verts = [set() for _ in range(nc)]
    for vs, c0, c1 in faces:
        for c in (c0, c1):
            if c >= 0:
                cell_verts[c].update(vs.tolist())
    alive = np.array([len(s) >= 4 for s in cell_verts])
    # apex = the cell's volume centroid (welding can push the vertex mean outside a face -> inverted
    # tets), expressed as the affine combination of the cell's vertices closest to equal weights:
    # w = 1/n + A^T (A A^T)^-1 (b - A 1/n), A = [X^T; 1^T], b = [centroid; 1] (exact for affine motion)
    cent_num = np.zeros((nc, 3)); cent_vol = np.zeros(nc)
    for vs, c0, c1 in faces:
        P = Y[vs]; c = P.mean(0)
        for t in range(len(vs)):
            a_, b_ = P[t], P[(t+1) % len(vs)]
            for cc, sg in ((c0, 1.), (c1, -1.)):
                if cc >= 0:
                    vol6 = sg*np.dot(c, np.cross(a_, b_))          # signed cone volume x 6 from the origin
                    cent_vol[cc] += vol6; cent_num[cc] += vol6*(c+a_+b_)/4
    apex_w = []
    for cc, sset in enumerate(cell_verts):
        vl = sorted(sset)
        if not alive[cc]:
            apex_w.append((vl, np.ones(len(vl))/max(len(vl), 1))); continue
        cen = cent_num[cc]/cent_vol[cc]
        A = np.vstack((Y[vl].T, np.ones(len(vl)))); w0 = np.ones(len(vl))/len(vl)
        w = w0+A.T@np.linalg.solve(A@A.T, np.r_[cen, 1.]-A@w0)
        apex_w.append((vl, w))
    apex = np.array([Y[vl].T@w if len(vl) else np.zeros(3) for vl, w in apex_w])
    # triangulate faces, build tets: points = vertices | face centres (only faces using the centre) | apexes
    nvY = len(Y)
    fc_rows_f, fc_rows_v, fc_rows_w, centre_of_face = [], [], [], {}
    tet_pts, tet_cell, schemes, quals = [], [], {'centre': 0, 'vertex': 0}, []
    for fi, (vs, c0, c1) in enumerate(faces):
        tris, scheme, q = best_triangulation(Y, vs)
        schemes[scheme] += 1; quals.append(q)
        if scheme == 'centre':
            cid = len(centre_of_face); centre_of_face[fi] = cid
            fc_rows_f += [cid]*len(vs); fc_rows_v += vs.tolist(); fc_rows_w += [1/len(vs)]*len(vs)
        for c, sign in ((c0, 1.), (c1, -1.)):
            if c < 0 or not alive[c]:
                continue
            for t in tris:
                ids = []
                for code in t:
                    ids.append(code[1] if code[0] == 'v' else ('c', fi))
                if sign < 0:
                    ids = [ids[0], ids[2], ids[1]]           # reverse orientation for the other side
                tet_pts.append((('a', c), *ids)); tet_cell.append(c)
    nfc = len(centre_of_face)
    def pid(x):
        if isinstance(x, tuple) and x[0] == 'c':
            return nvY+centre_of_face[x[1]]
        if isinstance(x, tuple) and x[0] == 'a':
            return nvY+nfc+x[1]
        return int(x)
    pts = np.array([[pid(x) for x in t] for t in tet_pts])
    tet_cell = np.array(tet_cell)
    fcen = np.zeros((nfc, 3)); np.add.at(fcen, np.array(fc_rows_f), Y[np.array(fc_rows_v)]*np.array(fc_rows_w)[:, None])
    allP = np.concatenate((Y, fcen, apex))
    P = allP[pts]
    Dm = np.stack((P[:, 1]-P[:, 0], P[:, 2]-P[:, 0], P[:, 3]-P[:, 0]), 2)
    det = np.linalg.det(Dm)
    V = det/6
    ok = np.abs(det) > 1e-18
    neg = int((V[ok] <= 0).sum())
    Gi_all = np.linalg.inv(np.where(ok[:, None, None], Dm, np.eye(3)))
    h_all = 1/np.linalg.norm(np.concatenate((-Gi_all.sum(1, keepdims=True), Gi_all), 1), axis=2).max(1)
    thin_drop = ok & (V > 0) & (h_all < args.min_height)
    drop_cells = np.unique(tet_cell[(ok & (V <= 0)) | thin_drop])
    n_thin_drop = int(thin_drop.sum()); vol_drop = float(np.abs(V[(ok & (V <= 0)) | thin_drop]).sum())
    ok &= (V > 0) & ~thin_drop
    pts, tet_cell, Dm, V = pts[ok], tet_cell[ok], Dm[ok], V[ok]
    Gi = np.linalg.inv(Dm)
    grads = np.concatenate((-Gi.sum(1, keepdims=True), Gi), 1)
    # apex rows
    ap_cell = np.concatenate([[c]*len(vl) for c, (vl, w) in enumerate(apex_w) if alive[c]])
    ap_vert = np.concatenate([vl for c, (vl, w) in enumerate(apex_w) if alive[c]])
    ap_w = np.concatenate([w for c, (vl, w) in enumerate(apex_w) if alive[c]])
    fc_face, fc_vert, fc_w = np.array(fc_rows_f), np.array(fc_rows_v), np.array(fc_rows_w)
    # lumped mass: each cell's volume split equally over its vertices (tet-wise |t|/4 left corners on
    # small tets 30-40x lighter than the median and set the explicit step)
    cv_tmp = np.bincount(tet_cell, np.abs(V), minlength=nc)
    vm = np.zeros(nvY)
    for cc, (vl, w) in enumerate(apex_w):
        if alive[cc] and len(vl):
            vm[vl] += cv_tmp[cc]/len(vl)
    # compact vertices
    used = np.unique(np.concatenate((pts[pts < nvY], fc_vert, ap_vert)))
    vmap = -np.ones(nvY, int); vmap[used] = np.arange(len(used))
    shift = nvY-len(used)
    pts_c = np.where(pts < nvY, vmap[np.minimum(pts, nvY-1)], pts-shift)
    # stats / checks
    size = np.cbrt(np.abs(V)); height = 1/np.linalg.norm(grads, axis=2).max(1)
    cellvol = np.bincount(tet_cell, np.abs(V), minlength=nc)
    cs = np.cbrt(np.maximum(cellvol, 1e-30))[tet_cell]
    A = np.array([[1.1, .05, 0], [0, .95, .08], [.03, 0, 1.02]])
    F = np.einsum('tpi,tpj->tij', (allP@A.T)[pts], grads)
    summary = dict(weld_mm=args.weld*1e3, vertices=int(len(used)), vertices_before=int(len(X0)), faces=len(faces), faces_dropped=dropped,
                   cells_alive=int(alive.sum()), cells_dropped=int((~alive).sum()), schemes=schemes, tets=int(len(V)),
                   negative_tets_dropped=neg, thin_tets_dropped=n_thin_drop, dropped_volume_share=vol_drop/float(np.abs(V).sum()),
                   cells_touched_by_drops=int(len(drop_cells)), volume_total=float(np.abs(V).sum()),
                   volume_change_vs_original=float(np.abs(V).sum()/0.029756264945633274-1),
                   thin_tets_h_lt_0p3_size=int((height < .3*size).sum()), thin_tets_h_lt_0p1_cellsize=int((height < .1*cs).sum()),
                   tets_height_lt_0p05mm=int((height < 5e-5).sum()), height_mm_quantiles=(np.quantile(height, [0, .001, .01, .5])*1e3).tolist(),
                   patch_max=float(np.abs(F-A).max()))
    print(json.dumps(summary, indent=1), flush=True)
    np.savez_compressed(out, X=Y[used], sp_id=np.repeat(np.arange(len(V)), 4), sp_pt=pts_c.ravel(), sp_g=grads.reshape(-1, 3),
                        tet_volume=np.abs(V), tet_cell=tet_cell, fc_face=fc_face, fc_vert=vmap[fc_vert], fc_w=fc_w,
                        ap_cell=ap_cell, ap_vert=vmap[ap_vert], ap_w=ap_w, n_face=nfc, n_cell=nc, vertex_volume=vm[used],
                        cell_volume=cellvol, source_index=src, cells_with_drops=drop_cells)
    out.with_suffix('.json').write_text(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
