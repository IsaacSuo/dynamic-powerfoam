"""k_alpha of alpha_complex.py in single precision (10-31), generated from it by substitution (no other change): the structure only, which pairs share a wall and which third sites cut it; gpu_geometry.Geometry then computes the values in double precision from that structure. A near-degenerate cell may close badly in single precision: its status says so and AlphaComplex then runs the double kernel. The kernel's time is mostly float64 under divergence on this GPU (UPDATED_LAGRANGE.md 21.5)."""
import warp as wp

from free_surface.alpha_complex import PMAX, TMAX, WMAX, LMAX


@wp.func
def vert3f(P: wp.array2d(dtype=wp.vec4f), i: int, a: int, b: int, c: int):
    na = wp.vec3f(P[i, a][0], P[i, a][1], P[i, a][2]); nb = wp.vec3f(P[i, b][0], P[i, b][1], P[i, b][2]); nc = wp.vec3f(P[i, c][0], P[i, c][1], P[i, c][2])
    bc = wp.cross(nb, nc); ca = wp.cross(nc, na); ab = wp.cross(na, nb)
    det = wp.dot(na, bc)
    return (P[i, a][3]*bc+P[i, b][3]*ca+P[i, c][3]*ab)/det


@wp.func
def seg_dist2f(a: wp.vec3f, b: wp.vec3f):
    """squared distance of the origin to the segment [a, b]"""
    e = b-a
    ee = wp.dot(e, e)
    t = wp.float32(0.0)
    if ee > wp.float32(0.0):
        t = wp.clamp(-wp.dot(a, e)/ee, wp.float32(0.0), wp.float32(1.0))
    q = a+t*e
    return wp.dot(q, q)


@wp.kernel(enable_backward=False)
def k_alpha32(p: wp.array(dtype=wp.vec3f), r: wp.array(dtype=wp.float32), coff: wp.array(dtype=int), cj: wp.array(dtype=int),
            cn: wp.array(dtype=wp.vec3f), ch: wp.array(dtype=wp.float32),
            P: wp.array2d(dtype=wp.vec4f), Pid: wp.array2d(dtype=int), T: wp.array2d(dtype=wp.vec3i), V: wp.array2d(dtype=wp.vec3f),
            rem: wp.array2d(dtype=int), ea: wp.array2d(dtype=int), eb: wp.array2d(dtype=int), nxt: wp.array2d(dtype=int),
            walls: wp.array2d(dtype=int), links: wp.array3d(dtype=int), nwall: wp.array(dtype=int), nlink: wp.array2d(dtype=int),
            status: wp.array(dtype=int)):
    """status: 0 fine, 1 hidden, 2 plane storage overflow, 3 vertex storage overflow, 4 wall storage overflow,
    5 link storage overflow, 6 open boundary (a cut's hole did not close)"""
    i = wp.tid()
    R = r[i]
    zero = wp.float32(0.0)
    one = wp.float32(1.0)
    st = int(0)
    # the cube around the ball: planes 0..5 (+x, -x, +y, -y, +z, -z), its 8 corners
    P[i, 0] = wp.vec4f(one, zero, zero, R); P[i, 1] = wp.vec4f(-one, zero, zero, R)
    P[i, 2] = wp.vec4f(zero, one, zero, R); P[i, 3] = wp.vec4f(zero, -one, zero, R)
    P[i, 4] = wp.vec4f(zero, zero, one, R); P[i, 5] = wp.vec4f(zero, zero, -one, R)
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
                V[i, nt] = vert3f(P, i, tt[0], tt[1], tt[2])
                nt += 1
    R2 = wp.float32(wp.float32(3.0)*R*R)
    origin_in = int(1)
    for c in range(coff[i], coff[i+1]):
        h = ch[c]
        if h > zero and h*h >= R2:
            break
        n = cn[c]
        nrem = int(0)
        for t in range(nt):
            o = int(0)
            if wp.dot(n, V[i, t]) > h:
                o = 1
            rem[i, t] = o
            nrem += o
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
        P[i, q] = wp.vec4f(n[0], n[1], n[2], h); Pid[i, q] = cj[c]
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
                T[i, m] = T[i, t]; V[i, m] = V[i, t]
                m += 1
        if m+nb > TMAX:
            st = 3
            break
        for e in range(nb):
            T[i, m] = wp.vec3i(ea[i, e], eb[i, e], q)
            V[i, m] = vert3f(P, i, ea[i, e], eb[i, e], q)
            m += 1
        nt = m
        R2 = zero
        for t in range(nt):
            R2 = wp.max(R2, wp.dot(V[i, t], V[i, t]))
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
            nf = wp.vec3f(P[i, f][0], P[i, f][1], P[i, f][2])
            hf = P[i, f][3]
            q0 = hf*nf
            d2 = hf*hf                                                           # the plane's foot: inside the polygon?
            pos = int(0); neg = int(0)
            dmin = wp.float32(1.0e30)
            kk = int(start)                                                     # (a typed declaration: Warp carries it
            closed = int(1)                                                     # through the loop; 10-24 a plain k = start
            for s in range(cnt):                                                # stayed at the start, one edge only)
                l = nxt[i, kk]
                if l < 0:
                    closed = 0
                    break
                lp = nxt[i, l]
                if lp < 0:
                    closed = 0
                    break
                va = vert3f(P, i, f, kk, l)
                vb = vert3f(P, i, f, l, lp)
                sg = wp.dot(wp.cross(vb-va, q0-va), nf)
                if sg > zero:
                    pos += 1
                if sg < zero:
                    neg += 1
                dmin = wp.min(dmin, seg_dist2f(va, vb))
                kk = l
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
            kk = int(start)
            for s in range(cnt):
                l = nxt[i, kk]
                lp = nxt[i, l]
                va = vert3f(P, i, f, kk, l)
                vb = vert3f(P, i, f, l, lp)
                if seg_dist2f(va, vb) < R2b and Pid[i, l] >= 0:                  # the edge between the face and plane l
                    if nl >= LMAX:
                        st = 5
                    else:
                        links[i, nw, nl] = Pid[i, l]
                        nl += 1
                kk = l
            nlink[i, nw] = nl
            nw += 1
    nwall[i] = nw
    if st == 0 and (nt == 0 or (nw == 0 and origin_in == 0)):
        st = 1
    status[i] = st
