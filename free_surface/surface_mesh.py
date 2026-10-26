"""Continuous outer surface of the material (2026-10-22, step 1 of the continuous-surface model): the faces between material
and air cells of the foam's power diagram, as polygons over SHARED vertices. Read exactly from the regular triangulation
(lower hull of the lifted sites, as foam_surface_completion), no matching by distance: a power vertex is a tetrahedron's
orthocentre; a tetrahedron with both material and air sites gives a surface vertex; the material-air face of an edge
(material site m, air site a) is the cycle of the tetrahedra around that edge, ordered counter-clockwise about m -> a
(outward), by angle about the face's own centre (the face is convex; the bond line may miss it). The unbounded power diagram is used; the material cells of the DEM are additionally clipped by their own
balls, which on the hull-ghost foams cuts almost nothing (exposed sphere volume ~1e-5 of the material) - checked below.
Checks: every polygon edge shared by exactly two faces in opposite directions (closed, consistently oriented); Euler
characteristic; enclosed volume (divergence theorem over the polygons) against the material volume; per surface cell the
air-face area against the DEM's clipped faces (clipped_faces, as pacnerf_foam_geometry). Writes <dem>/surface_mesh.npz
(foam frame and DEM frame vertices, CSR polygons, owning cell / body, air site)."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import ConvexHull

from experiments.bread_material_geometry import clipped_faces, neighbors
from experiments.pacnerf_foam_geometry import visual_hull
from experiments.powersim_bread_demo import regular_candidates


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--run', type=Path, required=True)
    a.add_argument('--dem', type=Path, required=True, help='DEM directory in the PAC-NeRF world frame (cells.npz, geometry.npz)')
    a.add_argument('--dem-z', type=Path, default=None, help='its z-up copy (vertices also written in that frame)')
    a.add_argument('--scene-dir', type=Path, default=Path('/home/fangsuo/py/dynamic-powerfoam/data/pacnerf_powerfoam/elastic_0_f0'))
    args = a.parse_args()
    t0 = time.time()
    st = torch.load(args.run/'model.pt', map_location='cpu', weights_only=True)
    p = st['points'].double().numpy(); r = torch.nn.functional.softplus(st['radii'].double(), beta=100).numpy()
    cells, geo = np.load(args.dem/'cells.npz'), np.load(args.dem/'geometry.npz')
    if 'ghost_points' in cells.files:
        p = np.concatenate((p, cells['ghost_points'])); r = np.concatenate((r, cells['ghost_radii']))
    N = len(p); src = cells['source_index']; boc = geo['body_of_cell']
    material = np.zeros(N, bool); material[src] = True
    n_real = int(st['num_sites']) if 'num_sites' in st else len(st['points'])
    inside = visual_hull(p, args.scene_dir); inside[n_real:] = False
    solid = material | inside                  # real sites inside the hull left out of the DEM (bounded volume 0) are not air
    print('real sites inside the hull but not material (zero bounded volume): %d' % int((inside & ~material).sum()), flush=True)
    cell_of = -np.ones(N, int); cell_of[src] = np.arange(len(src))
    offset = np.median(p[src]-cells['points'], axis=0)                      # foam -> PAC-NeRF world: x - offset
    # regular triangulation (all sites)
    scale = np.ptp(p, axis=0).max(); z = (p-p.mean(0))/scale
    hull = ConvexHull(np.column_stack((z, (z*z).sum(1)-(r/scale)**2)))
    T = hull.simplices[hull.equations[:, -2] < -1e-10]
    nm = solid[T].sum(1)
    mixed = T[(nm > 0) & (nm < 4)]
    print('sites %d (material %d); tetrahedra %d, mixed %d; %.0f s' % (N, material.sum(), len(T), len(mixed), time.time()-t0), flush=True)
    # orthocentres: 2 (p_k - p_0) . v = |p_k|^2 - |p_0|^2 - w_k + w_0
    P = p[mixed]; w = r[mixed]**2
    A = 2*(P[:, 1:]-P[:, :1])
    b = (P[:, 1:]**2).sum(-1)-(P[:, :1]**2).sum(-1)-w[:, 1:]+w[:, :1]
    s6 = np.linalg.det(A)
    flat = np.abs(s6) <= 1e-9*np.median(np.abs(s6))          # exactly flat: a site, its mirror and another such pair (one plane)
    vert = np.empty((len(mixed), 3))
    vert[~flat] = np.linalg.solve(A[~flat], b[~flat][..., None])[..., 0]
    for t in np.flatnonzero(flat):                            # the vertex is any point of a line: the one nearest the tet's centroid
        c = P[t].mean(0)
        vert[t] = c+np.linalg.lstsq(A[t], b[t]-A[t]@c, rcond=None)[0]
    print('flat mixed tetrahedra (vertex not unique, nearest point of its line): %d' % flat.sum(), flush=True)
    # material-air edges of the mixed tetrahedra
    ea = np.array([0, 0, 0, 1, 1, 2]); eb = np.array([1, 2, 3, 2, 3, 3])
    u, v = mixed[:, ea].ravel(), mixed[:, eb].ravel()
    tet = np.repeat(np.arange(len(mixed)), 6)
    ma = solid[u] != solid[v]
    u, v, tet = u[ma], v[ma], tet[ma]
    m_site = np.where(solid[u], u, v); a_site = np.where(solid[u], v, u)
    key = m_site.astype(np.int64)*N+a_site
    keys, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    F = len(keys)
    fm, fa = (keys//N).astype(int), (keys % N).astype(int)
    # order each face's tetrahedra counter-clockwise about m -> a
    e = p[fa]-p[fm]; e /= np.linalg.norm(e, axis=1, keepdims=True)
    ref = np.where(np.abs(e[:, :1]) < .9, np.array([[1., 0, 0]]), np.array([[0, 1., 0]]))
    uu = np.cross(e, ref); uu /= np.linalg.norm(uu, axis=1, keepdims=True); ww = np.cross(e, uu)
    # cyclic order from the combinatorics: the tetrahedra around edge (m, a) are the edges of its link cycle (their two other
    # sites); walking the cycle orders them exactly, also where vertices coincide (flat tetrahedra)
    order = np.argsort(inv, kind='stable')
    face_off = np.concatenate(([0], np.cumsum(cnt)))
    face_vert = np.empty(len(order), np.int64); open_faces = 0
    for f in range(F):
        ts = tet[order[face_off[f]:face_off[f+1]]]
        other = [tuple(x for x in mixed[t_] if x != fm[f] and x != fa[f]) for t_ in ts]
        by = {}
        for k_, (x_, y_) in enumerate(other):
            by.setdefault(x_, []).append(k_); by.setdefault(y_, []).append(k_)
        seq, prev, k_ = [0], None, 0
        cur = other[0][1]
        while len(seq) < len(ts):
            nxt_ = [j_ for j_ in by[cur] if j_ != k_]
            if not nxt_:
                open_faces += 1; break
            k_ = nxt_[0]; seq.append(k_)
            cur = other[k_][0] if other[k_][1] == cur else other[k_][1]
        ring = ts[seq] if len(seq) == len(ts) else ts
        V_ = vert[ring]; c_ = V_.mean(0)
        if (np.cross(V_-c_, np.roll(V_, -1, axis=0)-c_).sum(0)@e[f]) < 0:          # orient counter-clockwise about m -> a
            ring = ring[::-1]
        face_vert[face_off[f]:face_off[f+1]] = ring
    print('faces whose tetrahedra do not close a cycle: %d' % open_faces, flush=True)
    # orientation by propagation (opposite directions on every shared edge): a zero-area face (coinciding vertices of the
    # flat tetrahedra) has no geometric orientation; the bulk keeps the counter-clockwise one, so the majority sign is used
    from collections import deque
    faces_ = [face_vert[face_off[f]:face_off[f+1]] for f in range(F)]
    ef = {}
    for f, P_ in enumerate(faces_):
        for k_ in range(len(P_)):
            a_, b_ = int(P_[k_]), int(P_[(k_+1) % len(P_)])
            ef.setdefault((min(a_, b_), max(a_, b_)), []).append(f)
    sign = np.zeros(F, int); conflicts = 0
    for f0 in range(F):
        if sign[f0]:
            continue
        sign[f0] = 1; q_ = deque([f0])
        while q_:
            f = q_.popleft(); P_ = faces_[f] if sign[f] > 0 else faces_[f][::-1]
            for k_ in range(len(P_)):
                a_, b_ = int(P_[k_]), int(P_[(k_+1) % len(P_)])
                for g in ef[(min(a_, b_), max(a_, b_))]:
                    if g == f:
                        continue
                    Q_ = faces_[g]; fwd = any(int(Q_[j_]) == a_ and int(Q_[(j_+1) % len(Q_)]) == b_ for j_ in range(len(Q_)))
                    need = -1 if fwd else 1
                    if sign[g] == 0:
                        sign[g] = need; q_.append(g)
                    elif sign[g] != need:
                        conflicts += 1
    if (sign < 0).sum() > (sign > 0).sum():
        sign = -sign
    for f in np.flatnonzero(sign < 0):
        face_vert[face_off[f]:face_off[f+1]] = face_vert[face_off[f]:face_off[f+1]][::-1]
    print('orientation propagated: faces reversed %d, conflicts %d' % (int((sign < 0).sum()), conflicts), flush=True)
    print('surface faces %d, vertices %d (vertices per face median %d, min %d); %.0f s' % (F, len(mixed), int(np.median(cnt)), cnt.min(), time.time()-t0), flush=True)
    # closedness: each directed polygon edge once, its reverse once
    nxt = np.empty_like(face_vert)
    for f0, f1 in zip(face_off[:-1], face_off[1:]):
        nxt[f0:f1] = np.roll(face_vert[f0:f1], -1)
    nv = len(mixed)
    dk = face_vert.astype(np.int64)*nv+nxt; rk = nxt.astype(np.int64)*nv+face_vert
    du, dc = np.unique(dk, return_counts=True)
    rev_ok = np.isin(rk, dk)
    und = np.unique(np.minimum(face_vert, nxt).astype(np.int64)*nv+np.maximum(face_vert, nxt), return_counts=True)[1]
    E_ = len(und)
    print('closed: directed edges used once %s, every edge has its reverse %s, undirected edges shared by two faces %d of %d; Euler V - E + F = %d' % (
        bool((dc == 1).all()), bool(rev_ok.all()), int((und == 2).sum()), E_, nv-E_+F), flush=True)
    # area vectors, centroids (fan from the vertex mean), enclosed volume
    av = np.zeros((F, 3)); cen = np.zeros((F, 3)); vol = 0.
    for f in range(F):
        V_ = vert[face_vert[face_off[f]:face_off[f+1]]]; c0 = V_.mean(0)
        tri = np.cross(V_-c0, np.roll(V_, -1, axis=0)-c0)*.5
        ta = np.linalg.norm(tri, axis=1)
        av[f] = tri.sum(0); cen[f] = ((c0+V_+np.roll(V_, -1, axis=0))/3*ta[:, None]).sum(0)/max(ta.sum(), 1e-300)
        vol += (tri.sum(0)@c0)/3
    out_ok = ((av*e).sum(1) > 0).mean()
    print('area vectors along m -> a: %.4f of the faces; enclosed volume %.6f m^3 (DEM material volume %.6f, ratio %.6f)' % (
        out_ok, vol, cells['masses'].sum(), vol/cells['masses'].sum()), flush=True)
    # against the DEM's clipped faces (same pairs)
    all_pairs, _, _ = regular_candidates(p, r)
    offs, ids = neighbors(N, all_pairs)                                   # every neighbour clips the face, as in the DEM build
    pr = np.column_stack((np.minimum(fm, fa), np.maximum(fm, fa)))
    area_c, _ = clipped_faces(p, r, pr, offs, ids, 64)
    an = np.linalg.norm(av, axis=1)
    rel = np.abs(an-area_c)/np.maximum(an, 1e-300)
    print('vs clipped faces: total area %.5f vs %.5f m^2; per face |rel. difference| median %.1e, 99%% %.1e, faces clipped by more than 1%% %d' % (
        an.sum(), area_c.sum(), np.median(rel), np.quantile(rel, .99), int((rel > .01).sum())), flush=True)
    res = dict(sites=N, faces=F, vertices=nv, edges=E_, euler=int(nv-E_+F), closed=bool((dc == 1).all() and rev_ok.all() and (und == 2).all()),
               enclosed_volume=float(vol), material_volume=float(cells['masses'].sum()), area_mesh=float(an.sum()), area_clipped=float(area_c.sum()))
    out = dict(vert_foam=vert, vert_world=vert-offset, face_off=face_off, face_vert=face_vert, face_cell=cell_of[fm], face_body=np.where(cell_of[fm] >= 0, boc[np.maximum(cell_of[fm], 0)], -1),
               face_air=fa, area_vec_world=av, centroid_world=cen-offset, area_clipped=area_c)
    if args.dem_z is not None:
        RZ = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
        out.update(vert_z=(vert-offset)@RZ.T, area_vec_z=av@RZ.T, centroid_z=(cen-offset)@RZ.T)
    np.savez_compressed(args.dem/'surface_mesh.npz', **out)
    (args.dem/'surface_mesh.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res), '%.0f s' % (time.time()-t0), flush=True)


if __name__ == '__main__':
    main()
