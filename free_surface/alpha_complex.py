"""The alpha complex of the balls (the structure gpu_geometry.Geometry needs: the pairs sharing a wall, each wall's
third sites, the hidden sites) on the GPU without a global triangulation (2026-10-24, toward re-cutting the foam every
time step; qhull's regular triangulation took 0.4 s). Power Foam's BVH gives every overlapping pair (powerfoam.bvh,
~300 per site here, x22 the walls); each cell is then clipped on its own (one thread per cell, as Ray, Sokolov, Levy
2018 for Voronoi cells): the power cell of site i in its local frame, from the cube around its ball, cut by the
planes n . y <= h of its overlapping neighbours j (n = (p_j - p_i) / L, h = (L^2 + r_i^2 - r_j^2) / (2 L), the
radical plane), nearest first, until the next plane is farther than every vertex. Planes with h >= r_i hold the whole
ball and are skipped (they change the cell only outside it). The polytope is kept as triangles of plane indices (a
vertex = three planes); a cut removes the vertices beyond the plane and closes the hole with the boundary cycle of the
removed region (each boundary edge (a, b) gives the vertex (a, b, new)). Then, in the ball (radius r_i):
  wall (i, j)   the face on j's plane meets the open ball (its polygon's distance to the site < r_i)
  link k        an edge of that face, the one shared with face k, meets the open ball
  hidden        no wall and the site outside its own cell.
float64 throughout; overflow of the fixed per-cell storage is reported, never silent.
11-06 (the k1 cut's cost, free_surface/k1_alpha_bench.py; every output identical to the kernel before, on two states):
  a vertex test n . V > h is done in float32 first (V32, the vertices' float32 copies) and in float64 only when the float32
  difference lies within 2^-20 (|V|max + |h|) (the float32 rounding of n, V, h and the dot is below it, so outside it the
  sign is the float64 one); a plane's vertices are counted first and the removal marks written only when it cuts; the
  vertices' radius bound R2 (only the early stop reads it) in float32 rounded up by 2^-20 (never below the float64 max:
  a later stop tests planes that cut nothing); the walls phase computes each face vertex once (edge s's end is edge s+1's
  start: the same planes in the same order) and keeps each edge's plane and distance for the links (ea, ed2).
  8k foam at the landing: 14.1 -> 7.3 ms a call. The kernel before: alpha_variants.k_alpha_v mode 0."""
import numpy as np
import torch
import warp as wp

PMAX, TMAX, WMAX, LMAX = 192, 512, 128, 24                                      # (WMAX 64 -> 128, PMAX 128 -> 192, TMAX 256 -> 512
                                                                                # 10-28: a compressed lion cell; a cell with W faces
                                                                                # has up to 2 (W + 6) - 4 vertices)


@wp.func
def vert3(P: wp.array2d(dtype=wp.vec4d), i: int, a: int, b: int, c: int):
    na = wp.vec3d(P[i, a][0], P[i, a][1], P[i, a][2]); nb = wp.vec3d(P[i, b][0], P[i, b][1], P[i, b][2]); nc = wp.vec3d(P[i, c][0], P[i, c][1], P[i, c][2])
    bc = wp.cross(nb, nc); ca = wp.cross(nc, na); ab = wp.cross(na, nb)
    det = wp.dot(na, bc)
    return (P[i, a][3]*bc+P[i, b][3]*ca+P[i, c][3]*ab)/det


@wp.func
def to_f(v: wp.vec3d):
    return wp.vec3f(wp.float32(v[0]), wp.float32(v[1]), wp.float32(v[2]))


@wp.func
def seg_dist2(a: wp.vec3d, b: wp.vec3d):
    """squared distance of the origin to the segment [a, b]"""
    e = b-a
    ee = wp.dot(e, e)
    t = wp.float64(0.0)
    if ee > wp.float64(0.0):
        t = wp.clamp(-wp.dot(a, e)/ee, wp.float64(0.0), wp.float64(1.0))
    q = a+t*e
    return wp.dot(q, q)


@wp.kernel(enable_backward=False)
def k_alpha(p: wp.array(dtype=wp.vec3d), r: wp.array(dtype=wp.float64), coff: wp.array(dtype=int), cj: wp.array(dtype=int),
            cn: wp.array(dtype=wp.vec3d), ch: wp.array(dtype=wp.float64),
            P: wp.array2d(dtype=wp.vec4d), Pid: wp.array2d(dtype=int), T: wp.array2d(dtype=wp.vec3i), V: wp.array2d(dtype=wp.vec3d),
            rem: wp.array2d(dtype=int), ea: wp.array2d(dtype=int), eb: wp.array2d(dtype=int), nxt: wp.array2d(dtype=int),
            V32: wp.array2d(dtype=wp.vec3f), ed2: wp.array2d(dtype=wp.float64),
            walls: wp.array2d(dtype=int), links: wp.array3d(dtype=int), nwall: wp.array(dtype=int), nlink: wp.array2d(dtype=int),
            status: wp.array(dtype=int)):
    """status: 0 fine, 1 hidden, 2 plane storage overflow, 3 vertex storage overflow, 4 wall storage overflow,
    5 link storage overflow, 6 open boundary (a cut's hole did not close)"""
    i = wp.tid()
    R = r[i]
    zero = wp.float64(0.0)
    one = wp.float64(1.0)
    st = int(0)
    # the cube around the ball: planes 0..5 (+x, -x, +y, -y, +z, -z), its 8 corners
    P[i, 0] = wp.vec4d(one, zero, zero, R); P[i, 1] = wp.vec4d(-one, zero, zero, R)
    P[i, 2] = wp.vec4d(zero, one, zero, R); P[i, 3] = wp.vec4d(zero, -one, zero, R)
    P[i, 4] = wp.vec4d(zero, zero, one, R); P[i, 5] = wp.vec4d(zero, zero, -one, R)
    for k in range(6):
        Pid[i, k] = -1
    nplane = int(6)
    nt = int(0)
    for sx in range(2):
        for sy in range(2):
            for sz in range(2):
                a = sx; b = 2+sy; c = 4+sz
                if (sx+sy+sz) % 2 == 1:                                          # det(n_a, n_b, n_c) > 0
                    T[i, nt] = wp.vec3i(b, a, c)
                else:
                    T[i, nt] = wp.vec3i(a, b, c)
                tt = T[i, nt]
                V[i, nt] = vert3(P, i, tt[0], tt[1], tt[2])
                V32[i, nt] = to_f(V[i, nt])
                nt += 1
    R2 = wp.float64(wp.float64(3.0)*R*R)
    origin_in = int(1)
    for c in range(coff[i], coff[i+1]):
        h = ch[c]
        if h > zero and h*h >= R2:
            break
        n = cn[c]
        nrem = int(0)
        n32 = to_f(n)
        h32 = wp.float32(h)
        tol = wp.float32(9.5367431640625e-07)*(wp.float32(wp.sqrt(R2))+wp.abs(h32))   # 2^-20 (|V|max + |h|)
        for t in range(nt):                                                    # (counted first: most planes cut nothing)
            d32 = wp.dot(n32, V32[i, t])-h32
            if d32 > tol:
                nrem += 1
            elif d32 >= -tol:
                if wp.dot(n, V[i, t]) > h:
                    nrem += 1
        if nrem > 0:
            for t in range(nt):
                o = int(0)
                d32 = wp.dot(n32, V32[i, t])-h32
                if d32 > tol:
                    o = 1
                elif d32 >= -tol:
                    if wp.dot(n, V[i, t]) > h:
                        o = 1
                rem[i, t] = o
        if nrem == 0:
            continue
        if h <= zero:
            origin_in = 0
        if nrem == nt:                                                           # nothing left: the cell is empty
            nt = 0
            break
        if nplane >= PMAX:
            st = 2
            break
        q = nplane
        P[i, q] = wp.vec4d(n[0], n[1], n[2], h); Pid[i, q] = cj[c]
        nplane += 1
        # the removed region's boundary: edges of removed triangles whose reverse is not in a removed triangle
        # (11-08: the reverse is looked for among the removed triangles only, their indices listed in nxt, free until
        # the walls phase; the same edges in the same order. More removed than nxt holds: the search over all.)
        nb = int(0)
        if nrem <= nxt.shape[1]:
            nr = int(0)
            for t in range(nt):
                if rem[i, t] == 1:
                    nxt[i, nr] = t
                    nr += 1
            for ra in range(nr):
                tr = T[i, nxt[i, ra]]
                for e in range(3):
                    x = tr[e]; y = tr[(e+1) % 3]
                    inner = int(0)
                    for ra2 in range(nr):
                        if ra2 != ra:
                            u = T[i, nxt[i, ra2]]
                            for e2 in range(3):
                                if u[e2] == y and u[(e2+1) % 3] == x:
                                    inner = 1
                    if inner == 0:
                        ea[i, nb] = x; eb[i, nb] = y
                        nb += 1
        else:
            for t in range(nt):
                if rem[i, t] == 1:
                    tr = T[i, t]
                    for e in range(3):
                        x = tr[e]; y = tr[(e+1) % 3]
                        inner = int(0)
                        for t2 in range(nt):
                            if t2 != t and rem[i, t2] == 1:
                                u = T[i, t2]
                                for e2 in range(3):
                                    if u[e2] == y and u[(e2+1) % 3] == x:
                                        inner = 1
                        if inner == 0:
                            ea[i, nb] = x; eb[i, nb] = y
                            nb += 1
        # compact the kept triangles, then close the hole
        m = int(0)
        for t in range(nt):
            if rem[i, t] == 0:
                T[i, m] = T[i, t]; V[i, m] = V[i, t]; V32[i, m] = V32[i, t]
                m += 1
        if m+nb > TMAX:
            st = 3
            break
        for e in range(nb):
            T[i, m] = wp.vec3i(ea[i, e], eb[i, e], q)
            vn = vert3(P, i, ea[i, e], eb[i, e], q)
            V[i, m] = vn
            V32[i, m] = to_f(vn)
            m += 1
        nt = m
        R2f = wp.float32(0.0)
        for t in range(nt):
            R2f = wp.max(R2f, wp.dot(V32[i, t], V32[i, t]))
        R2 = wp.float64(R2f)*wp.float64(1.00000095367431640625)                # (1 + 2^-20: never below the float64 max)
    # walls and links in the ball
    nw = int(0)
    R2b = R*R
    if st == 0 and nt > 0:
        for f in range(6, nplane):
            # the face's cycle of neighbouring planes: triangle (f, k, l) -> nxt[k] = l
            for k in range(nplane):
                nxt[i, k] = -1
            cnt = int(0)
            start = int(-1)
            for t in range(nt):
                tr = T[i, t]
                for e in range(3):
                    if tr[e] == f:
                        nxt[i, tr[(e+1) % 3]] = tr[(e+2) % 3]
                        start = tr[(e+1) % 3]
                        cnt += 1
            if cnt < 3:
                continue
            nf = wp.vec3d(P[i, f][0], P[i, f][1], P[i, f][2])
            hf = P[i, f][3]
            q0 = hf*nf
            d2 = hf*hf                                                           # the plane's foot: inside the polygon?
            pos = int(0); neg = int(0)
            dmin = wp.float64(1.0e300)
            kk = int(start)                                                     # (a typed declaration: Warp carries it
            closed = int(1)                                                     # through the loop; 10-24 a plain k = start
            l = int(nxt[i, kk])                                                 # stayed at the start, one edge only)
            va = wp.vec3d(zero, zero, zero)
            if l < 0:
                closed = 0
            else:
                va = vert3(P, i, f, kk, l)                                      # (each vertex once: edge s's end is edge s + 1's
                for s in range(cnt):                                            # start; the edge's plane and distance kept)
                    lp = nxt[i, l]
                    if lp < 0:
                        closed = 0
                        break
                    vb = vert3(P, i, f, l, lp)
                    sg = wp.dot(wp.cross(vb-va, q0-va), nf)
                    if sg > zero:
                        pos += 1
                    if sg < zero:
                        neg += 1
                    dd = seg_dist2(va, vb)
                    dmin = wp.min(dmin, dd)
                    ea[i, s] = l
                    ed2[i, s] = dd
                    va = vb
                    kk = l
                    l = lp
            if closed == 0:
                st = 6
                break
            if pos == 0 or neg == 0:
                dmin = wp.min(dmin, d2)
            if dmin >= R2b:
                continue
            if nw >= WMAX:
                st = 4
                break
            walls[i, nw] = Pid[i, f]
            nl = int(0)
            for s in range(cnt):
                l = ea[i, s]
                if ed2[i, s] < R2b and Pid[i, l] >= 0:                         # the edge between the face and plane l
                    if nl >= LMAX:
                        st = 5
                    else:
                        links[i, nw, nl] = Pid[i, l]
                        nl += 1
            nlink[i, nw] = nl
            nw += 1
    nwall[i] = nw
    if st == 0 and (nt == 0 or (nw == 0 and origin_in == 0)):
        st = 1
    status[i] = st


class AlphaComplex:
    """alpha_complex(p, r) -> gpu_geometry structure (pairs, links, nbrs, hidden) and the raw per-cell lists"""

    def __init__(self, n, device='cuda'):
        z = lambda shape, dt: wp.zeros(shape, dtype=dt, device=device)
        self.n = n
        self.P, self.Pid = z((n, PMAX), wp.vec4d), z((n, PMAX), int)
        self.T, self.V, self.rem = z((n, TMAX), wp.vec3i), z((n, TMAX), wp.vec3d), z((n, TMAX), int)
        self.ea, self.eb, self.nxt = z((n, TMAX), int), z((n, TMAX), int), z((n, PMAX), int)
        self.walls, self.links = z((n, WMAX), int), z((n, WMAX, LMAX), int)
        self.V32, self.ed2 = z((n, TMAX), wp.vec3f), z((n, TMAX), wp.float64)          # (11-06: the vertices' float32 copies, the face edges' distances)
        self.nwall, self.nlink, self.status = z(n, int), z((n, WMAX), int), z(n, int)
        from powerfoam.bvh import AABBTree
        self.tree = AABBTree(device)
        self.f32, self.f32_falls = False, 0                                        # (the clipping in single precision, alpha_complex32; a cell that closes badly: the double kernel)
        self.ov_p, self.ov_margin, self.n_overlaps = None, 0., 0
        import os
        self.coop = os.environ.get('ALPHA_COOP', '1') != '0'                           # (11-08: the clip by alpha_coop, one warp a cell, the same outputs; False / ALPHA_COOP=0: k_alpha)

    def overlaps(self, p, r, margin=0.):
        """the overlapping pairs (Power Foam's BVH) with the radii raised by margin: a superset of the overlaps at any
        radii up to r (1 + margin) at these sites (|p_i - p_j| < r_i + r_j <= (1 + margin)(r0_i + r0_j)), kept with the
        sites they were found at (covers)"""
        rm = r*(1+margin)
        self.tree.update(p.float().contiguous(), rm.float().contiguous())
        adj, off = self.tree.build_cech_complex()
        self.ov_off, self.ov_adj, self.ov_key = off, adj, object()                    # (the BVH's CSR by cell; a new key each build)
        self.ov = None                                                                # (the expanded pair list, made when candidates asks)
        self.ov_p, self.ov_r, self.ov_margin = p.detach().clone(), rm, margin
        self.n_overlaps += 1

    def covers(self, p, r):
        """the kept overlaps still hold every pair overlapping at (p, r) (11-06, for keeping them across the cuts): each
        site's r_i + |p_i - p0_i| within its kept radius rm_i = r0_i (1 + margin), p0 the sites they were found at (then
        |p_i - p_j| < r_i + r_j gives |p0_i - p0_j| < rm_i + rm_j); 1e-6 spared for the BVH's float32"""
        if self.ov_p is None or self.ov_p.shape != p.shape:
            return False
        return bool((r+(p-self.ov_p).norm(dim=1) <= self.ov_r*(1-1e-6)).all())

    def candidates(self, p, r, reuse=False):
        """the planes of the overlapping pairs, nearest first per cell, those reaching the ball (h < r_i: a pair
        that does not overlap has h >= r_i, so any superset of the overlapping pairs gives the same planes); reuse: the
        kept overlaps if they still cover (covers), else rebuilt with the kept margin"""
        if not (reuse and self.covers(p, r)):
            self.overlaps(p, r, self.ov_margin)
        if self.ov is None:
            off = self.ov_off; deg = (off[1:]-off[:-1]).long()
            self.ov = (torch.repeat_interleave(torch.arange(len(p), device=p.device), deg), self.ov_adj.long())
        i, j = self.ov
        n = len(p)
        d = p[j]-p[i]
        L = d.norm(dim=1)
        h = (L*L+r[i]**2-r[j]**2)/(2*L)
        keep = (h < r[i]) & (L > 0)
        i, j, d, L, h = i[keep], j[keep], d[keep], L[keep], h[keep]
        o = torch.argsort(h, stable=True); o = o[torch.argsort(i[o], stable=True)]
        i, j, nn, h = i[o], j[o], (d/L[:, None])[o], h[o]
        coff = torch.zeros(n+1, dtype=torch.int64, device=p.device); coff[1:] = torch.cumsum(torch.bincount(i, minlength=n), 0)
        return coff, j, nn, h

    def __call__(self, p, r, reuse=False):
        coff, j, nn, h = self.candidates(p, r, reuse)
        wf = lambda t, dt: wp.from_torch(t.contiguous(), dtype=dt)
        done = False
        if self.f32:
            from free_surface.alpha_complex32 import k_alpha32
            if not hasattr(self, 'P32'):
                self.P32, self.V32 = wp.zeros((self.n, PMAX), dtype=wp.vec4f, device=self.P.device), wp.zeros((self.n, TMAX), dtype=wp.vec3f, device=self.P.device)
            wp.launch(k_alpha32, self.n, inputs=[wf(p.float(), wp.vec3f), wf(r.float(), wp.float32), wf(coff.int(), wp.int32), wf(j.int(), wp.int32), wf(nn.float(), wp.vec3f), wf(h.float(), wp.float32),
                                                 self.P32, self.Pid, self.T, self.V32, self.rem, self.ea, self.eb, self.nxt],
                      outputs=[self.walls, self.links, self.nwall, self.nlink, self.status])
            done = not bool((wp.to_torch(self.status) > 1).any())
            self.f32_falls += not done
        if not done and self.coop:
            from free_surface.alpha_coop import clip
            clip(wf(r, wp.float64), wf(coff.int(), wp.int32), wf(j.int(), wp.int32), wf(nn, wp.vec3d), wf(h, wp.float64),
                 self.walls, self.links, self.nwall, self.nlink, self.status, self.n)
        elif not done:
            wp.launch(k_alpha, self.n, inputs=[wf(p, wp.vec3d), wf(r, wp.float64), wf(coff.int(), wp.int32), wf(j.int(), wp.int32), wf(nn, wp.vec3d), wf(h, wp.float64),
                                               self.P, self.Pid, self.T, self.V, self.rem, self.ea, self.eb, self.nxt, self.V32, self.ed2],
                      outputs=[self.walls, self.links, self.nwall, self.nlink, self.status])
        return dict(walls=wp.to_torch(self.walls), links=wp.to_torch(self.links), nwall=wp.to_torch(self.nwall), nlink=wp.to_torch(self.nlink),
                    status=wp.to_torch(self.status), ncand=len(j))

    def structure_gpu(self, out, n):
        """the gpu_geometry structure from the cells' lists on the GPU (each wall from its smaller site's side): pairs
        sorted as _pack sorts them, their links (-1 padded), each site's neighbours, hidden (status 1)"""
        dev = out['walls'].device
        W, nw, L, nl = out['walls'].long(), out['nwall'].long(), out['links'].long(), out['nlink'].long()
        slots = torch.arange(W.shape[1], device=dev)
        m = slots[None, :] < nw[:, None]
        ii = torch.arange(n, device=dev)[:, None].expand_as(W)[m]
        ss = slots[None, :].expand_as(W)[m]
        jj = W[m]
        keep = ii < jj
        ii, ss, jj = ii[keep], ss[keep], jj[keep]
        o = torch.argsort(ii*n+jj)
        ii, ss, jj = ii[o], ss[o], jj[o]
        nlk = nl[ii, ss]
        width = max(int(nlk.max()) if len(nlk) else 1, 1)
        lk = L[ii, ss, :width]
        lk = torch.where(torch.arange(width, device=dev)[None, :] < nlk[:, None], lk, torch.full_like(lk, -1))
        a = torch.cat((ii, jj)); b = torch.cat((jj, ii))
        o = torch.argsort(a, stable=True); a, b = a[o], b[o]
        deg = torch.bincount(a, minlength=n)
        slot = torch.arange(len(a), device=dev)-(torch.cumsum(deg, 0)-deg)[a]
        nbrs = torch.full((n, max(int(deg.max()), 1)), -1, dtype=torch.long, device=dev)
        nbrs[a, slot] = b
        return dict(pairs=torch.stack((ii, jj), 1), links=lk, nbrs=nbrs, hidden=out['status'] == 1)

    def structure(self, out, n):
        """the gpu_geometry structure from the cells' lists (each wall from its smaller site's side)"""
        from free_surface.gpu_geometry import _pack
        W, L, nw, nl = (out[k].cpu().numpy() for k in ('walls', 'links', 'nwall', 'nlink'))
        ii = np.repeat(np.arange(n), nw)
        slot = np.concatenate([np.arange(c) for c in nw]) if len(ii) else np.zeros(0, int)
        jj = W[ii, slot]
        keep = ii < jj
        pairs = np.stack((ii[keep], jj[keep]), 1)
        tri = [(a, b, k) for a, b, s in zip(ii[keep], jj[keep], slot[keep]) for k in L[a, s, :nl[a, s]]]
        hidden = out['status'].cpu().numpy() == 1
        return _pack(pairs, np.array(tri, np.int64).reshape(-1, 3), n, hidden, out['walls'].device)
