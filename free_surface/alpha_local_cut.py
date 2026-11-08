"""Experimental local-boundary clipping kernel; reference remains alpha_complex."""
import numpy as np
import torch
import warp as wp
from free_surface.alpha_boundary import local_boundary

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
        nb = local_boundary(i, nt, nplane, nrem, T, rem, nxt, ea, eb)
        if nb < 0:
            # Degenerate/non-shellable boundary: retain the original search.
            nb = int(0)
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
