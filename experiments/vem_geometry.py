"""Power Foam material cells -> first-order VEM polyhedral elements (geometry only, no dynamics).

The power (Laguerre) diagram of all trained sites is already a conforming polyhedral mesh:
convex cells, planar faces shared by two cells, vertices shared by (generically) four cells.
Each material cell of the DEM run (cells.npz source_index, same 7193 cells) becomes one VEM
element whose unknowns are its vertices; sites, radii and colours only ride along.

Power vertices come from the lower lifted convex hull (regular triangulation): a lower facet
with plane n.(z, w) + e = 0, w = |z|^2 - rho^2, is the vertex x = -n_z / (2 n_w) (Qt-triangulated
coplanar facets share one plane, hence one vertex). The face between sites i < j is the cycle of
regular tetrahedra around edge ij, ordered counter-clockwise about p_j - p_i (outward for i).
Faces toward air cells close the material cells (pure Laguerre faces, no support-sphere clip).

Vertices closer than --weld are merged (union-find; short power edges from densification would
set the explicit time step). Faces stay closed cycles, so every element remains a closed surface;
welded faces may be non-planar and are fan-triangulated about their vertex mean (a virtual point
expressed through the face vertices, identical for both sides of an interior face).

Projection (VEM, k = 1): F_E = (1/|E|) sum_t int_t x dS (x) N_t = sum_v x_v (x) g_Ev, exact for
affine x. Checks: closedness (sum_t A_t N_t = 0), volumes vs the bounded (sphere-clipped) power
volumes of cells.npz, patch test F = A for x = A X + b.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull, cKDTree


def power_diagram(points, radii):
    """Regular tetrahedra (sites) and their power vertices."""
    mean = points.mean(0); scale = np.ptp(points, axis=0).max()
    z = (points-mean)/scale
    lifted = np.column_stack((z, (z*z).sum(1)-(radii/scale)**2))
    hull = ConvexHull(lifted)
    lower = hull.equations[:, 3] < -1e-10
    tets, eq = hull.simplices[lower], hull.equations[lower]
    vertex = -eq[:, :3]/(2*eq[:, 3:4])*scale+mean
    return tets, vertex


def material_faces(points, tets, material):
    """Faces (i < j, at least one material): ragged cycles of tet ids, CCW about p_j - p_i."""
    n = len(points)
    edges = np.concatenate([tets[:, [a, b]] for a in range(4) for b in range(a+1, 4)])
    owner = np.tile(np.arange(len(tets)), 6)
    edges = np.sort(edges, 1)
    keep = material[edges].any(1)
    edges, owner = edges[keep], owner[keep]
    key = edges[:, 0]*n+edges[:, 1]
    order = np.argsort(key, kind='stable'); key, edges, owner = key[order], edges[order], owner[order]
    start = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    group = np.repeat(np.arange(len(start)), np.diff(np.r_[start, len(key)]))
    return key, edges, owner, start, group


def order_faces(points, vertex, edges, owner, start, group):
    d = points[edges[:, 1]]-points[edges[:, 0]]
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    axis = np.where(np.abs(d[:, :1]) < .8, [[1., 0., 0.]], [[0., 1., 0.]])
    u = np.cross(d, axis); u /= np.linalg.norm(u, axis=1, keepdims=True); v = np.cross(d, u)
    c = vertex[owner]
    count = np.diff(np.r_[start, len(owner)])
    centre = np.add.reduceat(c, start)/count[:, None]
    rel = c-centre[group]
    angle = np.arctan2((rel*v).sum(1), (rel*u).sum(1))
    order = np.lexsort((angle, group))
    return owner[order]


def weld(vertex, tol):
    if tol <= 0:
        return np.arange(len(vertex)), vertex.copy()
    pairs = cKDTree(vertex).query_pairs(tol, output_type='ndarray')
    graph = sparse.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(vertex),)*2)
    _, label = connected_components(graph, directed=False)
    count = np.bincount(label)
    merged = np.zeros((label.max()+1, 3))
    np.add.at(merged, label, vertex)
    return label, merged/count[:, None]


def build_elements(face_vertices, face_start, face_pair, elem_of, X):
    """Fan triangles -> per element (vertex, g) projection weights, volumes, closure residuals.

    face_vertices: concatenated welded vertex ids of each face (CCW about p_j - p_i, cyclic
    duplicates already removed); face_pair: (i, j) global sites; elem_of: global site -> element
    or -1. Faces with the same element on both sides (agglomerated cells) are interior and skipped.
    """
    e_side = elem_of[face_pair]
    e_side = np.where((e_side[:, :1] == e_side[:, 1:]), -1, e_side)
    n_face = len(face_start)
    k = np.diff(np.r_[face_start, len(face_vertices)])
    fid = np.repeat(np.arange(n_face), k)
    pos = np.arange(len(face_vertices))-face_start[fid]
    nxt = face_start[fid]+(pos+1) % k[fid]
    a, b = face_vertices, face_vertices[nxt]
    centre = np.zeros((n_face, 3)); np.add.at(centre, fid, X[a]); centre /= k[:, None]
    an = .5*np.cross(X[a]-centre[fid], X[b]-centre[fid])        # triangle (a, b, centre) vector area
    face_an = np.zeros((n_face, 3)); np.add.at(face_an, fid, an)
    tri_mean = (X[a]+X[b]+centre[fid])/3
    vol_term = (tri_mean*an).sum(1)/3                           # divergence theorem, oriented i -> j
    rows = []                                                   # (element, vertex, g * |E|, sign)
    for side, sign in ((0, 1.), (1, -1.)):
        elem = e_side[:, side]
        has = elem >= 0
        tri = has[fid]
        # corner a and corner b of each triangle get an/3; face centre spreads face_an/3 over k vertices
        e_t = elem[fid[tri]]
        rows.append((e_t, a[tri], sign*an[tri]/3))
        rows.append((e_t, b[tri], sign*an[tri]/3))
        rows.append((e_t, a[tri], sign*face_an[fid[tri]]/(3*k[fid[tri]][:, None])))
    n_elem = elem_of.max()+1
    volume = np.zeros(n_elem); closure = np.zeros((n_elem, 3)); area = np.zeros(n_elem)
    for side, sign in ((0, 1.), (1, -1.)):
        elem = e_side[:, side]; has = elem >= 0
        tri = has[fid]
        np.add.at(volume, elem[fid[tri]], sign*vol_term[tri])
        np.add.at(closure, elem[fid[tri]], sign*an[tri])
        np.add.at(area, elem[fid[tri]], np.linalg.norm(an[tri], axis=1))
    e = np.concatenate([r[0] for r in rows]); v = np.concatenate([r[1] for r in rows])
    g = np.concatenate([r[2] for r in rows])
    # Merge duplicate (element, vertex) entries.
    key = e.astype(np.int64)*len(X)+v
    uniq, inv = np.unique(key, return_inverse=True)
    G = np.zeros((len(uniq), 3)); np.add.at(G, inv, g)
    e_u, v_u = uniq//len(X), uniq % len(X)
    G /= volume[e_u, None]
    return dict(elem=e_u, vert=v_u, g=G, volume=volume, closure=closure, area=area)


def projection(X, el):
    F = np.zeros((el['volume'].shape[0], 3, 3))
    np.add.at(F, el['elem'], X[el['vert'], :, None]*el['g'][:, None, :])
    return F


def pyramid_subcells(X, fv, fstart, fpair, cell_of, cell_elem):
    """Each material cell = union of pyramids (apex = mean of the cell's vertices, base = one face).

    Pyramids tile every cell (hence every agglomerated element) exactly and, for convex cells,
    have positive volume. Their VEM projections F_s = sum_v x_v (x) g_sv (boundary: base fan
    triangles about the face vertex mean + side triangles to the apex; apex and face centre are
    vertex means, so no new unknowns) feed the Wriggers / Cihan stabilisation
    (1 - beta) |E| psi(F_E) + beta sum_s |s| psi(F_s). Unlike tetrahedra on the short power edges,
    a pyramid has no needle-shaped pieces. Returns sparse (sub, vert, g), volumes, sub -> element,
    and the element -> vertex incidence over all vertices (for lumped mass).
    """
    n_face = len(fstart)
    k = np.diff(np.r_[fstart, len(fv)])
    fid = np.repeat(np.arange(n_face), k)
    pos = np.arange(len(fv))-fstart[fid]
    a, b = fv, fv[fstart[fid]+(pos+1) % k[fid]]
    centre = np.zeros((n_face, 3)); np.add.at(centre, fid, X[a]); centre /= k[:, None]
    an_base = .5*np.cross(X[a]-centre[fid], X[b]-centre[fid])        # outward for side 0
    inc_face, inc_cell, inc_sign = [], [], []
    c_both = cell_of[fpair]
    for side, sign in ((0, 1.), (1, -1.)):
        c = c_both[:, side]
        ok = (c >= 0) & (c != c_both[:, 1-side])                       # faces inside a merged cell vanish
        inc_face.append(np.flatnonzero(ok)); inc_cell.append(c[ok]); inc_sign.append(np.full(ok.sum(), sign))
    inc_face, inc_cell, inc_sign = map(np.concatenate, (inc_face, inc_cell, inc_sign))
    n_sub = len(inc_face)
    n_cell = cell_of.max()+1
    # cell vertex sets -> apex
    rep = k[inc_face]
    sub_of_corner = np.repeat(np.arange(n_sub), rep)
    off = np.repeat(fstart[inc_face], rep)+(np.arange(rep.sum())-np.repeat(np.cumsum(rep)-rep, rep))
    corner = fv[off]
    cv = np.unique(inc_cell[sub_of_corner].astype(np.int64)*len(X)+corner)
    cv_cell, cv_vert = cv//len(X), cv % len(X)
    n_cv = np.bincount(cv_cell, minlength=n_cell)
    apex = np.zeros((n_cell, 3)); np.add.at(apex, cv_cell, X[cv_vert]); apex /= n_cv[:, None]
    # per corner (= per base edge a->b of the face, for this sub-cell)
    sgn = inc_sign[sub_of_corner][:, None]
    A_, B_ = a[off], b[off]
    p = apex[inc_cell[sub_of_corner]]
    base = sgn*an_base[off]
    side = -sgn*.5*np.cross(X[A_]-p, X[B_]-p)
    cf = centre[fid[off]]
    vol = np.zeros(n_sub)
    np.add.at(vol, sub_of_corner, ((X[A_]+X[B_]+cf)*base).sum(1)/3+((X[A_]+X[B_]+p)*side).sum(1)/3)
    vol /= 3
    rows_s, rows_v, rows_g = [], [], []
    # direct corners a, b of both triangles
    for vv in (A_, B_):
        rows_s.append(sub_of_corner); rows_v.append(vv); rows_g.append((base+side)/3)
    # face centre: sum of base AN / 3 spread over the face vertices
    base_sum = np.zeros((n_sub, 3)); np.add.at(base_sum, sub_of_corner, base)
    rows_s.append(sub_of_corner); rows_v.append(A_); rows_g.append(base_sum[sub_of_corner]/(3*rep[sub_of_corner][:, None]))
    # apex: sum of side AN / 3 spread over the cell vertices
    side_sum = np.zeros((n_sub, 3)); np.add.at(side_sum, sub_of_corner, side)
    cell_start = np.r_[0, np.cumsum(n_cv)[:-1]]
    sub_cell = inc_cell
    rep2 = n_cv[sub_cell]
    s2 = np.repeat(np.arange(n_sub), rep2)
    off2 = np.repeat(cell_start[sub_cell], rep2)+(np.arange(rep2.sum())-np.repeat(np.cumsum(rep2)-rep2, rep2))
    rows_s.append(s2); rows_v.append(cv_vert[off2]); rows_g.append(side_sum[s2]/(3*rep2[s2][:, None]))
    S = np.concatenate(rows_s).astype(np.int64); V = np.concatenate(rows_v); G = np.concatenate(rows_g)
    key = S*len(X)+V
    uniq, inv = np.unique(key, return_inverse=True)
    g = np.zeros((len(uniq), 3)); np.add.at(g, inv, G)
    sub, vert = uniq//len(X), uniq % len(X)
    g /= vol[sub, None]
    ev = np.unique(cell_elem[cv_cell]*len(X)+cv_vert)
    # Same projection over "points" = [vertices, face centres, cell apexes]: centres and apexes are
    # computed once per step (fc_*, ap_* averaging rows), so each pyramid reads ~k+2 points instead
    # of its whole cell (3.8M -> ~0.8M rows). Point ids: vertex v, face nv + f, apex nv + n_face + c.
    nv = len(X)
    P_s = np.concatenate([sub_of_corner, sub_of_corner, np.arange(n_sub), np.arange(n_sub)]).astype(np.int64)
    P_p = np.concatenate([A_, B_, nv+inc_face, nv+n_face+sub_cell])
    P_g = np.concatenate([(base+side)/3, (base+side)/3, base_sum/3, side_sum/3])
    key = P_s*(nv+n_face+n_cell)+P_p
    uq, inv = np.unique(key, return_inverse=True)
    pg = np.zeros((len(uq), 3)); np.add.at(pg, inv, P_g)
    sp_id, sp_pt = uq//(nv+n_face+n_cell), uq % (nv+n_face+n_cell)
    pg /= vol[sp_id, None]
    return dict(sub_id=sub, sub_vert=vert, sub_g=g, sub_volume=vol, sub_elem=cell_elem[sub_cell],
                all_elem=ev//len(X), all_vert=ev % len(X), sub_cell=sub_cell, sub_face=inc_face,
                sp_id=sp_id, sp_pt=sp_pt, sp_g=pg, fc_face=fid, fc_vert=fv.copy(), fc_w=1/k[fid],
                ap_cell=cv_cell, ap_vert=cv_vert, ap_w=1/n_cv[cv_cell], n_face=n_face, n_cell=n_cell)


def merge_small_cells(volume, pairs, areas, cell_elem, frac):
    """Cells below frac x median cell volume join the neighbour (same element) sharing the largest
    face, smallest first, until every merged cell is above the threshold or has no such neighbour.
    Only the stabilisation sub-cells use the merged cells; elements are unchanged."""
    import heapq
    n = len(volume)
    thr = frac*np.median(volume)
    parent = np.arange(n); gvol = volume.astype(float).copy()
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    nbrs = [dict() for _ in range(n)]
    for (i, j), a in zip(pairs, areas):
        if cell_elem[i] == cell_elem[j]:
            nbrs[i][j] = nbrs[i].get(j, 0.)+a; nbrs[j][i] = nbrs[j].get(i, 0.)+a
    heap = [(gvol[i], i) for i in range(n) if gvol[i] < thr]
    heapq.heapify(heap)
    merges = 0
    while heap:
        v, i = heapq.heappop(heap)
        if find(i) != i or v != gvol[i] or gvol[i] >= thr:
            continue
        # neighbour groups by shared area
        share = {}
        for j, a in nbrs[i].items():
            r = find(j)
            if r != i:
                share[r] = share.get(r, 0.)+a
        if not share:
            continue
        r = max(share, key=share.get)
        parent[i] = r; gvol[r] += gvol[i]; merges += 1
        for j, a in nbrs[i].items():
            nbrs[r][j] = nbrs[r].get(j, 0.)+a
        if gvol[r] < thr:
            heapq.heappush(heap, (gvol[r], r))
    roots = np.array([find(i) for i in range(n)])
    _, group = np.unique(roots, return_inverse=True)
    return group, merges


def merge_pyramids(sc, fv, fstart, frac):
    """Merge pyramids of the same cell smaller than frac x cell volume into an edge-adjacent one.

    Greedy per cell: the smallest group below the threshold joins its smallest edge-adjacent
    neighbour (bases share >= 2 vertices), until none is below or the cell is one group. A group
    is the union of pyramids with a common apex over adjacent faces (closed, star-shaped), and its
    projection is the volume-weighted mean of its members': g_Gv = sum_m |m| g_mv / |G|."""
    k = np.diff(np.r_[fstart, len(fv)])
    n = len(sc['sub_volume'])
    vol = sc['sub_volume'].copy()
    cell = sc['sub_cell']
    group = np.arange(n)
    order = np.argsort(cell, kind='stable')
    bounds = np.flatnonzero(np.r_[True, cell[order][1:] != cell[order][:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        subs = order[a:b]
        if len(subs) < 2:
            continue
        verts = [set(fv[fstart[sc['sub_face'][s]]:fstart[sc['sub_face'][s]]+k[sc['sub_face'][s]]]) for s in subs]
        adj = {i: {j for j in range(len(subs)) if j != i and len(verts[i] & verts[j]) >= 2} for i in range(len(subs))}
        gvol = {i: vol[s] for i, s in enumerate(subs)}
        members = {i: [i] for i in range(len(subs))}
        total = sum(gvol.values())
        while len(gvol) > 1:
            i = min(gvol, key=gvol.get)
            if gvol[i] >= frac*total:
                break
            nb = [j for j in adj[i] if j in gvol]
            if not nb:                                   # isolated: join the smallest other group
                nb = [j for j in gvol if j != i]
            j = min(nb, key=gvol.get)
            gvol[j] += gvol.pop(i); members[j] += members.pop(i)
            adj[j] = (adj[j] | adj.pop(i)) - {i, j}
            for t in adj:
                if i in adj[t]:
                    adj[t].discard(i); adj[t].add(j) if t != j else None
        for j, mem in members.items():
            group[subs[mem]] = subs[j]
    uniq, gid = np.unique(group, return_inverse=True)
    gvol = np.bincount(gid, vol)
    key = gid[sc['sub_id']].astype(np.int64)*(sc['sub_vert'].max()+1)+sc['sub_vert']
    u, inv = np.unique(key, return_inverse=True)
    w = sc['sub_g']*vol[sc['sub_id'], None]
    g = np.zeros((len(u), 3)); np.add.at(g, inv, w)
    nvert = sc['sub_vert'].max()+1
    sub, vert = u//nvert, u % nvert
    g /= gvol[sub, None]
    first = np.zeros(len(uniq), int); first[gid] = np.arange(n)
    return dict(sc, sub_id=sub, sub_vert=vert, sub_g=g, sub_volume=gvol, sub_elem=sc['sub_elem'][first],
                sub_cell=cell[first], sub_face=sc['sub_face'][first], pyramids_before_merge=n)


def icosphere_directions(level):
    t = (1+5**.5)/2
    v = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t],
                  [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], float)
    hull = ConvexHull(v); f = hull.simplices
    for _ in range(level):
        e = np.unique(np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [0, 2]]]), 1), axis=0)
        v = np.concatenate([v, v[e].mean(1)])
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        f = ConvexHull(v).simplices
    return v/np.linalg.norm(v, axis=1, keepdims=True)


def cap_sites(pts, radii, material, level, k=64):
    """Air sites closing exposed support spheres of material cells by tangent planes.

    For a material site i and direction n, the tangent point t = p_i + r_i n lies on the bounded
    cell (Laguerre_i ∩ ball_i) iff no other site's ball contains it. A zero-radius ghost at t has the
    tangent plane at t as its power bisector with i, and its power distance |x - t|^2 >= 0 never
    beats a site inside whose ball x lies, so no bounded (rendered) cell loses volume; exposed parts
    of Laguerre cells are cut to the circumscribed polyhedron of the ball, in one conforming diagram.
    """
    if level < 0:
        return np.zeros((0, 3)), np.zeros(0)
    n = icosphere_directions(level)
    idx = np.flatnonzero(material)
    t = (pts[idx, None]+radii[idx, None, None]*n[None]).reshape(-1, 3)
    owner = np.repeat(idx, len(n))
    _, near = cKDTree(pts).query(t, k=k)
    inside = ((t[:, None]-pts[near])**2).sum(-1) < radii[near]**2*(1-1e-9)
    inside &= near != owner[:, None]
    exposed = ~inside.any(1)
    dirs = np.tile(n, (len(idx), 1))[exposed]
    o = owner[exposed]
    return pts[o]+radii[o, None]*dirs, np.zeros(len(o))


def faces_for(pts, radii, material, elem_of, offset, weld_tol, keep_interior=False):
    tets, vertex = power_diagram(pts, radii)
    key, edges, owner, start, group = material_faces(pts, tets, material)
    cyc = order_faces(pts, vertex, edges, owner, start, group)
    face_pair = edges[start]
    label, X = weld(vertex[np.unique(cyc)]-offset, weld_tol)
    remap = -np.ones(len(vertex), int); remap[np.unique(cyc)] = np.arange(len(np.unique(cyc)))
    fv = label[remap[cyc]]
    first = np.r_[True, group[1:] != group[:-1]]
    last_of = np.r_[np.flatnonzero(first)[1:]-1, len(fv)-1]
    prev = np.where(first, last_of[group], np.arange(len(fv))-1)
    keep = fv != fv[prev]
    k = np.bincount(group[keep], minlength=len(start))
    ok_face = k >= 3
    keep &= ok_face[group]
    new_id = -np.ones(len(start), int); new_id[ok_face] = np.arange(ok_face.sum())
    fv_k, fid_k = fv[keep], new_id[group[keep]]
    fstart = np.flatnonzero(np.r_[True, fid_k[1:] != fid_k[:-1]])
    pair_k = face_pair[ok_face]
    el = build_elements(fv_k, fstart, pair_k, elem_of, X)
    # drop vertices not on any element boundary (interior of agglomerates), unless the sub-cell
    # stabilisation needs them (keep_interior: all faces of material cells are returned)
    used_v = np.unique(fv_k) if keep_interior else np.unique(el['vert'])
    vmap = -np.ones(len(X), int); vmap[used_v] = np.arange(len(used_v))
    el['vert'] = vmap[el['vert']]
    e_side = elem_of[pair_k]
    boundary = (e_side[:, 0] != e_side[:, 1]) | keep_interior
    fk = np.diff(np.r_[fstart, len(fv_k)])
    fkeep = np.repeat(boundary, fk)
    fv_b = vmap[fv_k[fkeep]]
    fstart_b = np.r_[0, np.cumsum(fk[boundary])[:-1]]
    return X[used_v], fv_b, fstart_b, pair_k[boundary], el, dict(regular_tets=len(tets), dropped_faces=int((~ok_face).sum()))


def max_frequency(X, el, mu, lam, rho, gamma, iters=200, seed=0):
    """Largest omega of lumped-mass linear VEM (consistency + stabilisation gamma mu V/h^2 |r|^2 / n_v),
    by power iteration on M^-1 K. Only an estimate of the explicit step for planning."""
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    T = lambda a, dt=torch.float64: torch.as_tensor(a, dtype=dt, device=dev)
    Xt, e, v, g = T(X), T(el['elem'], torch.long), T(el['vert'], torch.long), T(el['g'])
    vol = T(el['volume']); ne = len(vol)
    nv = torch.bincount(e, minlength=ne).to(Xt.dtype)
    h2 = vol.abs()**(2/3)
    Xbar = torch.zeros(ne, 3, dtype=Xt.dtype, device=dev).index_add_(0, e, Xt[v])/nv[:, None]
    mass = torch.zeros(len(X), dtype=Xt.dtype, device=dev).index_add_(0, v, (rho*vol/nv)[e])

    def K(u):
        H = torch.zeros(ne, 3, 3, dtype=u.dtype, device=dev).index_add_(0, e, u[v][:, :, None]*g[:, None, :])
        eps = .5*(H+H.transpose(1, 2))
        ubar = torch.zeros(ne, 3, dtype=u.dtype, device=dev).index_add_(0, e, u[v])/nv[:, None]
        r = u[v]-ubar[e]-torch.einsum('eij,ej->ei', H[e], Xt[v]-Xbar[e])
        energy = (vol*(mu*(eps**2).sum((1, 2))+.5*lam*eps.diagonal(dim1=1, dim2=2).sum(1)**2)).sum()
        energy = energy+.5*gamma*mu*((vol/h2/nv)[e]*(r**2).sum(1)).sum()
        return energy
    u = torch.randn(len(X), 3, dtype=Xt.dtype, device=dev, generator=torch.Generator(dev).manual_seed(seed))
    lam_max = 0.
    for _ in range(iters):
        u = u/u.norm()
        _, Ku = torch.func.jvp(torch.func.grad(K), (torch.zeros_like(u),), (u,))
        w = Ku/mass[:, None]
        lam_max = float((u*w).sum()/(u*u).sum())
        u = w
    return lam_max**.5


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, default=Path('output/pacnerf_elastic0_f0_v1'))
    p.add_argument('--scene-dir', type=Path, default=Path('data/pacnerf_powerfoam/elastic_0_f0'))
    p.add_argument('--cells', type=Path, default=Path('output/pacnerf_dem/elastic_0_f0/cells.npz'))
    p.add_argument('--bodies', type=Path, default=Path('output/pacnerf_dem/elastic_0_f0/geometry.npz'))
    p.add_argument('--weld', type=float, default=1e-6, help='merge numerically coincident power vertices (m)')
    p.add_argument('--cap-level', type=int, nargs='+', default=[1, 2], help='-1 none; icosphere level of cap directions')
    p.add_argument('--agglomerate', nargs='+', default=['bodies', 'none'])
    p.add_argument('--pyramids', action='store_true', help='keep interior vertices, add pyramid sub-cells')
    p.add_argument('--merge-frac', type=float, default=0., help='merge pyramids below this fraction of their cell')
    p.add_argument('--cell-merge-frac', type=float, default=0.,
                   help='before cutting pyramids, merge cells below this fraction of the median cell volume')
    p.add_argument('--out', type=Path, default=Path('output/pacnerf_vem/elastic_0_f0'))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    state = torch.load(args.run/'model.pt', map_location='cpu', weights_only=True)
    pts0 = state['points'].double().numpy()
    radii0 = torch.nn.functional.softplus(state['radii'].double(), beta=100).numpy()
    offset = np.array(json.loads((args.scene_dir/'conversion.json').read_text())['offset'])
    cells = dict(np.load(args.cells)); body_of_cell = np.load(args.bodies)['body_of_cell']
    source = cells['source_index']
    material0 = np.zeros(len(pts0), bool); material0[source] = True
    E, nu, rho = 23008.4, .30079, 1000.
    lam, mu = E*nu/((1+nu)*(1-2*nu)), E/(2*(1+nu))
    rng = np.random.default_rng(0)
    reports = []
    for level in args.cap_level:
        gp, gr = cap_sites(pts0, radii0, material0, level)
        pts, radii = np.concatenate([pts0, gp]), np.concatenate([radii0, gr])
        material = np.r_[material0, np.zeros(len(gp), bool)]
        for agg in args.agglomerate:
            cell_elem = np.arange(len(source)) if agg == 'none' else body_of_cell
            elem_of = -np.ones(len(pts), int); elem_of[source] = cell_elem
            X, fv, fstart, fpair, el, info = faces_for(pts, radii, material, elem_of, offset, args.weld,
                                                       keep_interior=args.pyramids)
            vol = el['volume']
            extra = {}
            if args.pyramids:
                group = np.arange(len(source)); n_merges = 0
                if args.cell_merge_frac > 0:
                    group, n_merges = merge_small_cells(cells['masses'], cells['pairs'], cells['areas'],
                                                        cell_elem, args.cell_merge_frac)
                group_elem = np.zeros(group.max()+1, int); group_elem[group] = cell_elem
                cell_of = -np.ones(len(pts), int); cell_of[source] = group
                sc = pyramid_subcells(X, fv, fstart, fpair, cell_of, group_elem)
                sc['cell_group'] = group
                info.update(cell_merges=n_merges, merged_cells=int(group.max()+1))
                if args.merge_frac > 0:
                    sc = merge_pyramids(sc, fv, fstart, args.merge_frac)
                Fs = np.zeros((len(sc['sub_volume']), 3, 3))
                Ap = np.eye(3)+.3*rng.standard_normal((3, 3))
                np.add.at(Fs, sc['sub_id'], (X@Ap.T+1.)[sc['sub_vert'], :, None]*sc['sub_g'][:, None, :])
                per_elem = np.bincount(sc['sub_elem'], sc['sub_volume'], minlength=len(vol))
                hs = np.cbrt(np.abs(sc['sub_volume']))
                cvol = np.bincount(sc['sub_cell'], sc['sub_volume'])
                flat = np.zeros(len(sc['sub_volume'])); np.maximum.at(flat, sc['sub_id'], np.linalg.norm(sc['sub_g'], axis=1))
                flat *= np.cbrt(sc['sub_volume'])
                info.update(sub_over_cell_quantiles=np.quantile(sc['sub_volume']/cvol[sc['sub_cell']], [0, .01, .1, .5]).tolist(),
                            sub_flatness_quantiles=np.quantile(flat, [.5, .99, .999, 1]).tolist(),
                            sub_per_cell_median=float(np.median(np.bincount(sc['sub_cell']))))
                info.update(pyramids=len(sc['sub_volume']), pyramid_entries=len(sc['sub_g']),
                            pyramid_nonpositive=int((sc['sub_volume'] <= 0).sum()),
                            pyramid_volume_min_over_elem_h3=float((sc['sub_volume']/np.cbrt(vol[sc['sub_elem']])**3).min()),
                            pyramid_tiling_rel_err=float(np.abs(per_elem/vol-1).max()),
                            pyramid_patch_err=float(np.abs(Fs-Ap).max()),
                            pyramid_size_quantiles=np.quantile(hs, [0, .01, .5]).tolist())
                used = np.zeros(len(X), bool); used[el['vert']] = True; used[sc['sub_vert']] = True
                vm = -np.ones(len(X), int); vm[used] = np.arange(used.sum())
                info.update(unreferenced_vertices_removed=int((~used).sum()))
                X = X[used]; el['vert'] = vm[el['vert']]
                for key_ in ('sub_vert', 'all_vert'):
                    sc[key_] = vm[sc[key_]]
                if 'sp_pt' in sc:                              # point ids: vertices remapped, others shifted
                    nv_old, shift = len(vm), len(vm)-int(used.sum())
                    pt = sc['sp_pt']
                    sc['sp_pt'] = np.where(pt < nv_old, vm[np.minimum(pt, nv_old-1)], pt-shift)
                    for a_, w_, o_ in (('fc_vert', 'fc_w', 'fc_face'), ('ap_vert', 'ap_w', 'ap_cell')):
                        mv = vm[sc[a_]]; ok_ = mv >= 0
                        sc[a_], sc[w_], sc[o_] = mv[ok_], sc[w_][ok_], sc[o_][ok_]
                keep_all = sc['all_vert'] >= 0
                sc['all_vert'], sc['all_elem'] = sc['all_vert'][keep_all], sc['all_elem'][keep_all]
                fv = vm[fv]                                    # faces only for reference (-1: removed vertex)
                extra = sc
            bounded = np.bincount(cell_elem, cells['masses'])
            A = np.eye(3)+.3*rng.standard_normal((3, 3))
            patch = float(np.abs(projection(X@A.T+rng.standard_normal(3), el)-A).max())
            th = 1.3; R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
            rot = float(np.abs(projection(X@R.T, el)-R).max())
            nverts = np.bincount(el['elem'], minlength=len(vol))
            h = np.cbrt(np.abs(vol))
            omega = {f'gamma{gm:g}': max_frequency(X, el, mu, lam, rho, gm) for gm in (.1, 1.)}
            r = dict(cap_level=level, cap_sites=len(gp), agglomerate=agg, elements=len(vol), vertices=len(X),
                     unknowns=3*len(X), **info, vertices_per_element_median=float(np.median(nverts)),
                     max_closure_rel=float((np.linalg.norm(el['closure'], axis=1)/el['area']).max()),
                     patch_affine_max_err=patch, patch_rotation_max_err=rot,
                     negative_volumes=int((vol <= 0).sum()), total_volume=float(vol.sum()),
                     bounded_total=float(cells['masses'].sum()),
                     volume_ratio_quantiles=np.quantile(vol/bounded, [0, .01, .5, .99, 1]).tolist(),
                     element_size_quantiles=np.quantile(h, [0, .01, .5, .99]).tolist(),
                     vertex_bbox=[X.min(0).tolist(), X.max(0).tolist()],
                     omega_max=omega, steps_per_frame={k: float(np.ceil(w/(2*24))) for k, w in omega.items()})
            print(json.dumps(r), flush=True)
            reports.append(r)
            tag = ("_pyr" if args.pyramids else "")+(f"m{args.merge_frac:g}" if args.merge_frac > 0 else "")+(f"c{args.cell_merge_frac:g}" if args.cell_merge_frac > 0 else "")
            np.savez_compressed(args.out/f'vem_cap{level}_{agg}{tag}.npz', **extra, X=X, face_vertices=fv, face_start=fstart,
                                face_pair=fpair, elem=el['elem'], vert=el['vert'], g=el['g'], volume=vol,
                                cell_elem=cell_elem, sites=pts0[source]-offset, site_radii=radii0[source],
                                source_index=source)
    (args.out/f'vem_geometry{"_pyr" if args.pyramids else ""}.json').write_text(json.dumps(reports, indent=1))


if __name__ == '__main__':
    main()
