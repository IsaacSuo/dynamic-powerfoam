"""geometry_warp.k_geo_pair with the same outputs, faster (11-08):
  - the events' scratch (angle, sign, link) in local vectors, not in the global rows ea, ed, el (still in the signature)
  - each link's radical-line segment clipped by its two neighbours on the face's cycle only (the links come in the cut's
    walk order: alpha_complex's walls phase writes them along the face), not by every other link: the wall's polygon
    is convex, so its edge's segment is the line between the adjacent edges' half-planes; an edge dropped between two
    links (outside the ball) has its vertex outside the disk, where the circle bounds the segment, and the other link's
    half-plane holds the whole polygon (it cuts the line only beyond that vertex). The bounds are the same rhs / ad of
    the same links: the same bits whenever a neighbour is the binding one.
Used by GeoPipe / GroundPipe (FAST_PAIR=0: k_geo_pair)."""
import warp as wp

from free_surface.alpha_complex import LMAX
from free_surface.geometry_warp import rem2pi

vecE = wp.types.vector(length=2*LMAX, dtype=wp.float64)
vecEi = wp.types.vector(length=2*LMAX, dtype=wp.int32)


@wp.kernel(enable_backward=False)
def k_geo_pair_fast(p: wp.array(dtype=wp.vec3d), r: wp.array(dtype=wp.float64), P: wp.array2d(dtype=wp.int32), Cn: wp.array2d(dtype=wp.int32), K: int, tp: wp.float64, big: wp.float64, fin: wp.float64, thr: wp.float64,
               ea: wp.array2d(dtype=wp.float64), ed: wp.array2d(dtype=wp.float64), el: wp.array2d(dtype=wp.int32),
               nn_o: wp.array(dtype=wp.vec3d), c_o: wp.array(dtype=wp.vec3d), u_o: wp.array(dtype=wp.vec3d), v_o: wp.array(dtype=wp.vec3d),
               rho_o: wp.array(dtype=wp.float64), hi_o: wp.array(dtype=wp.float64), hj_o: wp.array(dtype=wp.float64),
               at0: wp.array2d(dtype=wp.float64), at1: wp.array2d(dtype=wp.float64), narc: wp.array(dtype=wp.int32),
               wA: wp.array(dtype=wp.float64), wc: wp.array(dtype=wp.vec3d), Vw: wp.array(dtype=wp.float64)):
    q = wp.tid()
    i = P[q, 0]
    j = P[q, 1]
    one = wp.float64(1.0)
    zero = wp.float64(0.0)
    half_ = wp.float64(0.5)
    pi_ = p[i]
    ri = r[i]
    rj = r[j]
    d = p[j]-pi_
    L = wp.length(d)
    nn = d/L
    hi = (L*L+ri*ri-rj*rj)/(wp.float64(2.0)*L)
    c = pi_+hi*nn
    rho = wp.sqrt(ri*ri-hi*hi)
    ref = wp.vec3d(one, zero, zero)
    if wp.abs(nn[0]) >= thr:
        ref = wp.vec3d(zero, one, zero)
    u = wp.normalize(wp.cross(nn, ref))
    v = wp.cross(nn, u)
    nn_o[q] = nn
    c_o[q] = c
    u_o[q] = u
    v_o[q] = v
    rho_o[q] = rho
    hi_o[q] = hi
    hj_o[q] = L-hi
    # each link's covered interval of the circle: two events (start +1, end -1); the wraps counted
    ea_ = vecE()
    ed_ = vecE()
    el_ = vecEi()
    anyfull = int(0)
    c0 = int(0)
    for m in range(K):
        k = Cn[q, m]
        s_ = big
        e_ = big
        dl = zero
        if k >= 0:
            g = c-p[k]
            A_ = wp.float64(2.0)*rho*wp.dot(g, u)
            B_ = wp.float64(2.0)*rho*wp.dot(g, v)
            C_ = wp.dot(g, g)+rho*rho-r[k]*r[k]
            R2 = A_*A_+B_*B_
            R = zero
            if R2 > zero:
                R = wp.sqrt(R2)
            if -C_ >= R:
                anyfull = 1
            if wp.abs(C_) < R:
                al = wp.acos(-C_/R)
                ph = wp.atan2(B_, A_)
                s_ = rem2pi(ph+al, tp)
                e0 = s_+tp-wp.float64(2.0)*al
                if e0 > tp:
                    c0 += 1
                e_ = rem2pi(e0, tp)
                dl = one
        ea_[m] = s_
        ed_[m] = dl
        el_[m] = k
        ea_[K+m] = e_
        ed_[K+m] = -dl
        el_[K+m] = k
    # sort the 2K events by angle (insertion sort, stable)
    for a in range(1, 2*K):
        x = ea_[a]
        y = ed_[a]
        z = el_[a]
        b = a
        while b > 0:
            if ea_[b-1] > x:
                ea_[b] = ea_[b-1]
                ed_[b] = ed_[b-1]
                el_[b] = el_[b-1]
                b -= 1
            else:
                break
        ea_[b] = x
        ed_[b] = y
        el_[b] = z
    # the sweep: uncovered intervals
    na = int(0)
    nev = int(0)
    cov = wp.float64(c0)
    for m in range(2*K):
        if ea_[m] < fin:
            nev += 1
        cov = cov+ed_[m]
        nx = big
        if m+1 < 2*K:
            nx = ea_[m+1]
        if anyfull == 0 and cov == zero and ea_[m] < fin and nx < fin:
            at0[q, na] = ea_[m]
            at1[q, na] = nx
            na += 1
    if anyfull == 0 and c0 == 0 and nev > 0:
        at0[q, na] = ea_[nev-1]
        at1[q, na] = ea_[0]+tp
        na += 1
    if anyfull == 0 and nev == 0:
        at0[q, na] = zero
        at1[q, na] = tp
        na += 1
    narc[q] = na
    # the wall: sectors of the arcs
    Aw = zero
    Mx = zero
    My = zero
    for m in range(na):
        t0 = at0[q, m]
        t1 = at1[q, m]
        Aw = Aw+half_*rho*rho*(t1-t0)
        Mx = Mx+(rho*rho*rho/wp.float64(3.0))*(wp.sin(t1)-wp.sin(t0))
        My = My+(rho*rho*rho/wp.float64(3.0))*(wp.cos(t0)-wp.cos(t1))
    # and the radical-line segments: pi_i <= pi_k  <=>  a . (s, t) <= beta in the plane x = c + s u + t v
    rho2 = rho*rho
    nl = int(0)
    for l in range(K):
        if Cn[q, l] >= 0:
            nl += 1
    for l in range(K):
        kl = Cn[q, l]
        if kl < 0:
            continue
        g = p[kl]-pi_
        av = wp.vec2d(wp.float64(2.0)*wp.dot(u, g), wp.float64(2.0)*wp.dot(v, g))
        beta = wp.dot(g, g)+ri*ri-r[kl]*r[kl]-wp.float64(2.0)*hi*wp.dot(nn, g)
        an2 = wp.dot(av, av)
        if an2 <= zero:
            continue
        foot = av*(beta/an2)
        sa = wp.sqrt(an2)
        dirn = wp.vec2d(-av[1]/sa, av[0]/sa)
        f2 = wp.dot(foot, foot)
        if f2 >= rho2:
            continue
        hh = wp.sqrt(rho2-f2)
        upv = big
        dnv = -big
        empty = int(0)
        for w2 in range(2):
            m = (l+nl-1) % nl
            if w2 == 1:
                m = (l+1) % nl
            if nl < 2 or m == l or (w2 == 1 and m == (l+nl-1) % nl):
                continue
            km = Cn[q, m]
            gm = p[km]-pi_
            am = wp.vec2d(wp.float64(2.0)*wp.dot(u, gm), wp.float64(2.0)*wp.dot(v, gm))
            bm = wp.dot(gm, gm)+ri*ri-r[km]*r[km]-wp.float64(2.0)*hi*wp.dot(nn, gm)
            ad = wp.dot(am, dirn)
            rhs = bm-wp.dot(am, foot)
            if ad > zero:
                upv = wp.min(upv, rhs/ad)
            elif ad < zero:
                dnv = wp.max(dnv, rhs/ad)
            elif rhs < zero:
                empty = 1
        if empty == 1 or not (wp.min(hh, upv) > wp.max(-hh, dnv)):
            continue
        tl = -hh
        if dnv > -hh:
            tl = dnv
        tu = hh
        if upv < hh:
            tu = upv
        P0 = foot+tl*dirn
        P1 = foot+tu*dirn
        cr = P0[0]*P1[1]-P0[1]*P1[0]
        Aw = Aw+half_*cr
        Mx = Mx+half_*cr*(P0[0]+P1[0])/wp.float64(3.0)
        My = My+half_*cr*(P0[1]+P1[1])/wp.float64(3.0)
    wA[q] = Aw
    cw = c
    if Aw > zero:
        cw = c+(Mx/Aw)*u+(My/Aw)*v
    wc[q] = cw
    wp.atomic_add(Vw, i, Aw*hi)
    wp.atomic_add(Vw, j, Aw*(L-hi))
