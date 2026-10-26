"""Exact geometry of a sphere surface (2026-10-22): the material is the union of the balls of power-diagram sites with
no air sites cutting it (free_surface/test_part.py); everything is read from the regular triangulation (lower hull of the
lifted sites; eight far dummy sites of radius 0, jittered so that no four lifted dummies are coplanar, close every cell
and never reach into a ball: pi_d(x) > 0 >= pi_i(x) there).
  corner  a triangle (i, j, k): its power edge is the segment between the orthocentres of its two tetrahedra, on the
          radical line of the three spheres; its points with pi = 0 lie on the three spheres and in the three cells,
          hence outside every other ball: a corner of the surface, shared by the patches i, j, k
  arc     an edge (i, j): the circle where the two spheres meet, in their radical plane; its parts outside every other
          ball (tested at each part's middle) are surface arcs, between two corners on the circle, or a full ring
  patch   the part of sphere i outside every other ball, bounded by its arcs. Each arc is oriented for each of its two
          patches with the patch on the left (about the sphere's outward normal: the side away from the other ball);
          oriented arcs chain at the corners into loops.
  face    the wall between cells i and j: the power face (orthocentres of the tetrahedra around the edge, in link order)
          clipped by the disk of the circle; exact area and centroid (polygon - disk intersection: triangles from the
          disk centre, inside pieces as triangles, outside pieces as circular sectors)
Patch integrals over the region left of a loop: area by Gauss-Bonnet from the boundary alone (arcs of geodesic curvature
-cos(alpha) / r seen from outside their cap, turning angles at the corners; a disk region), and by a signed fan from a
pole (gnomonic sectors, Gauss-Legendre; a negative result is the region holding the pole's antipode, so 4 pi r^2 is
added), which also gives int n dA and int n n^T dA."""
import numpy as np
import torch
from scipy.spatial import ConvexHull


def triangulation(p, r):
    """regular triangulation with eight far dummy sites (indices n..n+7): tetrahedra T, orthocentres O, and for the
    triangles of three material sites their two tetrahedra"""
    n = len(p)
    size = np.ptp(p, axis=0).max(); c0 = p.mean(0)
    box = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    P = np.concatenate((p, c0+10*size*(box+.1*np.random.default_rng(0).standard_normal(box.shape))))
    R = np.concatenate((r, np.zeros(8)))
    z = (P-c0)/size
    hull = ConvexHull(np.column_stack((z, (z*z).sum(1)-(R/size)**2)))
    T = hull.simplices[hull.equations[:, -2] < -1e-12]
    A_ = 2*(P[T[:, 1:]]-P[T[:, :1]])
    b_ = (P[T[:, 1:]]**2).sum(-1)-(P[T[:, :1]]**2).sum(-1)-R[T[:, 1:]]**2+R[T[:, :1]]**2
    O = np.linalg.solve(A_, b_[..., None])[..., 0]
    tri = np.sort(np.concatenate([T[:, [0, 1, 2]], T[:, [0, 1, 3]], T[:, [0, 2, 3]], T[:, [1, 2, 3]]]), axis=1)
    tet_of = np.tile(np.arange(len(T)), 4)
    key = (tri[:, 0].astype(np.int64)*len(P)+tri[:, 1])*len(P)+tri[:, 2]
    _, first, inv, cnt = np.unique(key, return_index=True, return_inverse=True, return_counts=True)
    assert cnt.max() == 2 and (tri[first[cnt == 1]] >= n).any(1).all(), 'a material triangle with one tetrahedron, or a triangle with three'
    two = np.flatnonzero(cnt[inv] == 2)
    order = two[np.argsort(inv[two], kind='stable')]
    tris = tri[order[::2]]; ta, tb = tet_of[order[::2]], tet_of[order[1::2]]
    mat = (tris < n).all(1)
    return P, R, T, O, tris[mat], ta[mat], tb[mat]


def min_power_others(x, p, r, own):
    """min over the sites not in own (rows of site indices) of the power distance of x"""
    pt, rt = torch.as_tensor(p, device='cuda'), torch.as_tensor(r, device='cuda')
    out = np.empty(len(x))
    for s in range(0, len(x), 4096):
        y = torch.as_tensor(x[s:s+4096], device='cuda')
        pw = torch.cdist(y, pt)**2-rt[None]**2
        pw.scatter_(1, torch.as_tensor(np.asarray(own[s:s+4096], np.int64), device='cuda'), np.inf)
        out[s:s+4096] = pw.min(1).values.cpu().numpy()
    return out


def circle(p, r, i, j):
    """the circle where spheres i and j meet: centre, radius, in-plane basis (u, v, u x v = n from i to j), and the signed
    distances of the two sites to its plane (h_i along n, h_j along -n); None when the balls do not meet"""
    d = p[j]-p[i]; L = np.linalg.norm(d); nn = d/L
    hi = (L*L+r[i]**2-r[j]**2)/(2*L)
    if hi*hi >= r[i]**2:
        return None
    ref = np.array([1., 0, 0]) if abs(nn[0]) < .9 else np.array([0, 1., 0])
    u = np.cross(nn, ref); u /= np.linalg.norm(u); v = np.cross(nn, u)
    return dict(c=p[i]+hi*nn, rho=np.sqrt(r[i]**2-hi*hi), u=u, v=v, n=nn, hi=hi, hj=L-hi)


def arc_point(C, t):
    return C['c']+C['rho']*(np.cos(t)*C['u']+np.sin(t)*C['v'])


def skeleton(p, r):
    n = len(p)
    P, R, T, O, tris, ta, tb = triangulation(p, r)
    D = O[tb]-O[ta]; W = O[ta]-p[tris[:, 0]]
    qa, qb, qc = (D*D).sum(1), 2*(D*W).sum(1), (W*W).sum(1)-r[tris[:, 0]]**2
    disc = qb*qb-4*qa*qc
    corners, corner_tri = [], []
    for t in np.flatnonzero((disc > 0) & (qa > 0)):
        for sgn in (-1, 1):
            u = (-qb[t]+sgn*np.sqrt(disc[t]))/(2*qa[t])
            if 0 <= u <= 1:
                corners.append(O[ta[t]]+u*D[t]); corner_tri.append(tris[t])
    corners = np.array(corners).reshape(-1, 3); corner_tri = np.array(corner_tri, int).reshape(-1, 3)
    on_circle = {}
    for c, (i, j, k) in enumerate(corner_tri):
        for e in ((i, j), (i, k), (j, k)):
            on_circle.setdefault(e, []).append(c)
    edges = sorted({(min(a, b), max(a, b)) for t in T for a in t for b in t if a != b and a < n and b < n})
    cand = []
    for (i, j) in edges:
        C = circle(p, r, i, j)
        if C is None:
            continue
        cs = on_circle.get((i, j), [])
        if cs:
            th = np.array([np.arctan2((corners[c]-C['c'])@C['v'], (corners[c]-C['c'])@C['u']) for c in cs]); o = np.argsort(th)
            cs = [cs[k] for k in o]; th = th[o]
            for k in range(len(cs)):
                cand.append((i, j, cs[k], cs[(k+1) % len(cs)], th[k], th[(k+1) % len(cs)]+(2*np.pi if k == len(cs)-1 else 0.), C))
        else:
            cand.append((i, j, -1, -1, 0., 2*np.pi, C))
    mids = np.array([arc_point(x[6], (x[4]+x[5])/2) for x in cand])
    q = min_power_others(mids, p, r, np.array([[x[0], x[1]] for x in cand]))
    arcs = [x for x, v in zip(cand, q) if v >= 0]
    pw_c = min_power_others(corners, p, r, corner_tri) if len(corners) else np.zeros(0)
    on3 = max(np.abs(((corners-p[corner_tri[:, k]])**2).sum(1)-r[corner_tri[:, k]]**2).max() for k in range(3)) if len(corners) else 0.
    ends = np.zeros(len(corners), int)
    for x in arcs:
        if x[2] >= 0:
            ends[x[2]] += 1; ends[x[3]] += 1
    return dict(P=P, R=R, T=T, O=O, corners=corners, corner_tri=corner_tri, arcs=arcs, edges=edges,
                check=dict(corners=len(corners), corner_power_to_others_min=float(pw_c.min()) if len(pw_c) else 0., corner_off_its_spheres_max=float(on3),
                           arcs=len(arcs), rings=int(sum(x[2] < 0 for x in arcs)), arc_ends_per_corner={int(k): int(v) for k, v in zip(*np.unique(ends, return_counts=True))}))


def oriented_loops(p, r, sk):
    """per patch, its arcs oriented with the patch on the left, chained at the corners: [(site, [(arc, forward)])]"""
    arcs = sk['arcs']
    by = {}
    for ai, (i, j, c0, c1, t0, t1, C) in enumerate(arcs):
        tm = (t0+t1)/2; x = arc_point(C, tm)
        tan = -np.sin(tm)*C['u']+np.cos(tm)*C['v']                    # increasing theta: counter-clockwise about n_ij
        for site, cap in ((i, C['n']), (j, -C['n'])):
            m = (x-p[site])/r[site]
            away = -(cap-(cap@m)*m)                                      # along the sphere, away from the other ball
            by.setdefault(site, []).append((ai, bool(np.cross(m, tan)@away > 0)))
    loops, broken = [], 0
    for site, lst in by.items():
        start = {}
        for ai, fwd in lst:
            a_ = arcs[ai]
            if a_[2] < 0:
                loops.append((site, [(ai, fwd)])); continue
            s_, e_ = (a_[2], a_[3]) if fwd else (a_[3], a_[2])
            broken += s_ in start
            start[s_] = (ai, fwd, e_)
        used = set()
        for s_ in list(start):
            if s_ in used:
                continue
            loop, c = [], s_
            while c not in used:
                if c not in start:
                    broken += 1; break
                used.add(c); ai, fwd, e_ = start[c]
                loop.append((ai, fwd)); c = e_
            broken += c != s_
            loops.append((site, loop))
    return loops, broken


def loop_pieces(sk, loop, piece=2*np.pi/24):
    """the loop's arcs in travel order as (circle, theta from, theta to), pieces of at most `piece` radians"""
    out = []
    for ai, fwd in loop:
        i, j, c0, c1, t0, t1, C = sk['arcs'][ai]
        a, b = (t0, t1) if fwd else (t1, t0)
        m = max(1, int(np.ceil(abs(b-a)/piece)))
        out += [(C, a+(b-a)*k/m, a+(b-a)*(k+1)/m) for k in range(m)]
    return out


def gauss_bonnet_area(p, r, sk, site, loop):
    """area of the disk region left of the loop: r^2 (2 pi - int kappa_g ds - sum of turning angles)"""
    rr = r[site]; kg = 0.; tang = []
    for ai, fwd in loop:
        i, j, c0, c1, t0, t1, C = sk['arcs'][ai]
        kg += -(C['hi'] if site == i else C['hj'])/rr*abs(t1-t0)         # -cos(alpha) per radian of the circle
        a, b = (t0, t1) if fwd else (t1, t0)
        sg = 1. if fwd else -1.
        tang.append((sg*(-np.sin(a)*C['u']+np.cos(a)*C['v']), sg*(-np.sin(b)*C['u']+np.cos(b)*C['v']), arc_point(C, b)))
    turn = 0.
    if sk['arcs'][loop[0][0]][2] >= 0:
        for k in range(len(tang)):
            t_end, x = tang[k][1], tang[k][2]; t_start = tang[(k+1) % len(tang)][0]
            m = (x-p[site])/rr
            turn += np.arctan2(np.cross(t_end, t_start)@m, t_end@t_start)
    return rr*rr*(2*np.pi-kg-turn)


def fan_integrals(p, r, sk, site, loop, nw=24, nt=24):
    """area, int n dA, int n n^T dA over the region left of the loop (signed gnomonic fan from the pole that keeps the
    boundary closest), and min over the boundary of pole . direction"""
    rr = r[site]; pc = p[site]
    pieces = loop_pieces(sk, loop)
    B = (np.array([arc_point(C, a) for C, a, b in pieces])-pc)/rr
    k = np.arange(400)+.5
    ph = np.arccos(1-2*k/400); th = np.pi*(1+5**.5)*k
    cands = np.column_stack((np.cos(th)*np.sin(ph), np.sin(th)*np.sin(ph), np.cos(ph)))
    m_ = B.mean(0)
    if np.linalg.norm(m_) > 1e-12:
        cands = np.concatenate((cands, [m_/np.linalg.norm(m_), -m_/np.linalg.norm(m_)]))
    pole = cands[np.argmax((cands@B.T).min(1))]
    gw, ww = np.polynomial.legendre.leggauss(nw); gw = .5*(gw+1); ww = .5*ww
    gt, wt = np.polynomial.legendre.leggauss(nt); gt = .5*(gt+1); wt = .5*wt
    A = 0.; a1 = np.zeros(3); a2 = np.zeros((3, 3))
    for C, a, b in pieces:
        t = a+(b-a)*gt
        bb = (C['c'][None]+C['rho']*(np.cos(t)[:, None]*C['u']+np.sin(t)[:, None]*C['v'])-pc)/rr
        db = C['rho']*(-np.sin(t)[:, None]*C['u']+np.cos(t)[:, None]*C['v'])/rr*(b-a)
        s = (1-gw)[:, None, None]*pole+gw[:, None, None]*bb[None]
        ns = np.linalg.norm(s, axis=-1); sh = s/ns[..., None]
        dA = rr*rr*gw[:, None]*(np.cross(bb[None]-pole, db[None])*sh).sum(-1)/ns**2*ww[:, None]*wt[None]
        A += dA.sum(); a1 += np.einsum('wt,wti->i', dA, sh); a2 += np.einsum('wt,wti,wtj->ij', dA, sh, sh)
    if A < 0:                                                          # the fan covered the region holding the antipode
        A += 4*np.pi*rr*rr; a2 += 4*np.pi*rr*rr/3*np.eye(3)
    return A, a1, a2, float((pole@B.T).min())


def face_polygons(p, sk):
    """for each material edge: the power face (orthocentres of the tetrahedra around the edge, in link order)"""
    T, O, n = sk['T'], sk['O'], len(p)
    around = {}
    for t, tet in enumerate(T):
        for x in range(4):
            for y in range(x+1, 4):
                a, b = sorted((tet[x], tet[y]))
                if b < n:
                    around.setdefault((a, b), []).append(t)
    polys = {}
    for (i, j), ts in around.items():
        other = [tuple(v for v in T[t] if v != i and v != j) for t in ts]
        at = {}
        for k, (x, y) in enumerate(other):
            at.setdefault(x, []).append(k); at.setdefault(y, []).append(k)
        seq, cur, k = [0], other[0][1], 0
        while len(seq) < len(ts):
            nxt = [q for q in at[cur] if q != k]
            if not nxt:
                break
            k = nxt[0]; seq.append(k)
            cur = other[k][0] if other[k][1] == cur else other[k][1]
        if len(seq) == len(ts):
            polys[(i, j)] = O[np.array(ts)[seq]]
    return polys


def disk_polygon(V2, rho):
    """area and first moment (about the disk centre) of a convex counter-clockwise polygon (2D) clipped by the disk"""
    A = 0.; M = np.zeros(2)
    for k in range(len(V2)):
        a, b = V2[k], V2[(k+1) % len(V2)]
        d = b-a; qa, qb, qc = d@d, 2*a@d, a@a-rho*rho
        ts = [0.]
        disc = qb*qb-4*qa*qc
        if disc > 0 and qa > 0:
            ts += [t for t in sorted(((-qb-np.sqrt(disc))/(2*qa), (-qb+np.sqrt(disc))/(2*qa))) if 0 < t < 1]
        ts.append(1.)
        for t0, t1 in zip(ts[:-1], ts[1:]):
            x0, x1, xm = a+t0*d, a+t1*d, a+.5*(t0+t1)*d
            if xm@xm <= rho*rho:
                ar = .5*(x0[0]*x1[1]-x0[1]*x1[0])
                A += ar; M += ar*(x0+x1)/3
            else:
                al = np.arctan2(x0[1], x0[0]); dl = np.arctan2(x0[0]*x1[1]-x0[1]*x1[0], x0@x1); be = al+dl
                A += .5*rho*rho*dl
                M += rho**3/3*np.array([np.sin(be)-np.sin(al), np.cos(al)-np.cos(be)])
    return A, M


def faces_exact(p, r, sk):
    """{(i, j): (area, centroid, circle)} of the walls with positive area"""
    out = {}
    for (i, j), V in face_polygons(p, sk).items():
        C = circle(p, r, i, j)
        if C is None:
            continue
        V2 = np.column_stack(((V-C['c'])@C['u'], (V-C['c'])@C['v']))
        if np.sum(V2[:, 0]*np.roll(V2[:, 1], -1)-np.roll(V2[:, 0], -1)*V2[:, 1]) < 0:
            V2 = V2[::-1]
        A, M = disk_polygon(V2, C['rho'])
        if A > 0:
            out[(i, j)] = (A, C['c']+(M[0]/A)*C['u']+(M[1]/A)*C['v'], C)
    return out
