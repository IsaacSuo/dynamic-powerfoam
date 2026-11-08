"""gpu_geometry.Geometry's values in two Warp kernels (11-06, the k1 run's cost: Geometry's torch construction was 19 ms of a
40 ms call, about 4.5 calls a step at the landing). Same formulas, same structure input, no gradient (k1's cut is not
differentiated: update_ul detaches x, the radii solve has its own Jacobian). Values for what the run reads: the pairs i, j,
the circles (nn, c, rho, u, v, hi, hj), wall_A, wall_c, patch_n, patch_A, V; patch_nn is not computed (diagnostics only:
use Geometry there).
  k_geo_pair  one thread per pair: the circle; each link's covered interval, the events sorted (insertion sort in the
              pair's scratch row), the uncovered arcs by the sweep; the wall's area and first moment (sectors of the arcs,
              triangles of the radical-line segments, each segment's ends from the bounding lines); the walls' share of V
  k_geo_site  one thread per site over its (pair, side) entries: patch_n = 1/2 sum int m x dm; the pole (of 32 Fibonacci
              directions and +-the mean normal, the one whose least dot with the boundary samples is largest; Geometry's
              rule); the area by the fan from the pole (Van Oosterom - Strackee triangle + 2 beta + cos psi |dt| per
              piece), 4 pi when -P is exposed; V = (sum_walls A h + r A_patch) / 3
Summation orders differ from Geometry (index_add vs per-thread sums): values agree to rounding."""
import math

import numpy as np
import torch
import warp as wp

TWO_PI = 2*math.pi
BIG = 1e300
NCAND = 34                                                                       # (32 Fibonacci poles and +-the mean normal)
vecC = wp.types.vector(length=NCAND, dtype=wp.float64)


@wp.func
def rem2pi(x: wp.float64, tp: wp.float64):
    return x-tp*wp.floor(x/tp)


@wp.kernel(enable_backward=False)
def k_geo_pair(p: wp.array(dtype=wp.vec3d), r: wp.array(dtype=wp.float64), P: wp.array2d(dtype=wp.int32), Cn: wp.array2d(dtype=wp.int32), K: int, tp: wp.float64, big: wp.float64, fin: wp.float64, thr: wp.float64,
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
        ea[q, m] = s_
        ed[q, m] = dl
        el[q, m] = k
        ea[q, K+m] = e_
        ed[q, K+m] = -dl
        el[q, K+m] = k
    # sort the 2K events by angle (insertion sort, stable)
    for a in range(1, 2*K):
        x = ea[q, a]
        y = ed[q, a]
        z = el[q, a]
        b = a
        while b > 0:
            if ea[q, b-1] > x:
                ea[q, b] = ea[q, b-1]
                ed[q, b] = ed[q, b-1]
                el[q, b] = el[q, b-1]
                b -= 1
            else:
                break
        ea[q, b] = x
        ed[q, b] = y
        el[q, b] = z
    # the sweep: uncovered intervals
    na = int(0)
    nev = int(0)
    cov = wp.float64(c0)
    for m in range(2*K):
        if ea[q, m] < fin:
            nev += 1
        cov = cov+ed[q, m]
        nx = big
        if m+1 < 2*K:
            nx = ea[q, m+1]
        if anyfull == 0 and cov == zero and ea[q, m] < fin and nx < fin:
            at0[q, na] = ea[q, m]
            at1[q, na] = nx
            na += 1
    if anyfull == 0 and c0 == 0 and nev > 0:
        at0[q, na] = ea[q, nev-1]
        at1[q, na] = ea[q, 0]+tp
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
        for m in range(K):
            km = Cn[q, m]
            if km < 0 or m == l:
                continue
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


@wp.func
def emv(t: wp.float64, u: wp.vec3d, v: wp.vec3d):
    return wp.cos(t)*u+wp.sin(t)*v


@wp.kernel(enable_backward=False)
def k_geo_site(p: wp.array(dtype=wp.vec3d), r: wp.array(dtype=wp.float64), off: wp.array(dtype=wp.int32), ent: wp.array(dtype=wp.int32), npr: int,
               nn_o: wp.array(dtype=wp.vec3d), u_o: wp.array(dtype=wp.vec3d), v_o: wp.array(dtype=wp.vec3d), rho_o: wp.array(dtype=wp.float64),
               hi_o: wp.array(dtype=wp.float64), hj_o: wp.array(dtype=wp.float64), at0: wp.array2d(dtype=wp.float64), at1: wp.array2d(dtype=wp.float64),
               narc: wp.array(dtype=wp.int32), fib: wp.array(dtype=wp.vec3d), nfib: int, piece: wp.float64, big: wp.float64, fourpi: wp.float64,
               nbrs: wp.array2d(dtype=wp.int32), hidden: wp.array(dtype=wp.int32), stable_area: int, floor_enabled: int, floor_height: wp.float64,
               site_ids: wp.array(dtype=wp.int32), indexed: int, Vw: wp.array(dtype=wp.float64),
               pn_o: wp.array(dtype=wp.vec3d), pA_o: wp.array(dtype=wp.float64), V_o: wp.array(dtype=wp.float64), pole_o: wp.array(dtype=wp.vec3d)):
    s = wp.tid()
    if indexed == 1:
        s = site_ids[s]
    zero = wp.float64(0.0)
    one = wp.float64(1.0)
    half_ = wp.float64(0.5)
    rs = r[s]
    # pass 1: int m dA (x from the site)
    pn = wp.vec3d()
    for e in range(off[s], off[s+1]):
        t = ent[e]
        q = t % npr
        first = t < npr
        axis = nn_o[q]
        h = hi_o[q]
        if not first:
            axis = -nn_o[q]
            h = hj_o[q]
        u = u_o[q]
        v = v_o[q]
        rho = rho_o[q]
        for m in range(narc[q]):
            ta = at0[q, m]
            tb = at1[q, m]
            if first:
                ta = at1[q, m]
                tb = at0[q, m]
            D = tb-ta
            pn = pn+half_*(rho*wp.cross(h*axis, emv(tb, u, v)-emv(ta, u, v))+(rho*rho*D)*wp.cross(u, v))
    pn_o[s] = pn
    # pass 2: the pole, the candidate with the largest least dot over the boundary samples (piece starts and middles);
    # each sample computed once, the candidates' running minima in a local vector (index nfib: +nd, nfib + 1: -nd)
    npn = wp.length(pn)
    nd = wp.vec3d()
    if npn > zero:
        nd = pn/npn
    ncand = nfib+2
    sc = vecC()
    for ci in range(ncand):
        sc[ci] = big
    for e in range(off[s], off[s+1]):
        t = ent[e]
        q = t % npr
        first = t < npr
        axis = nn_o[q]
        h = hi_o[q]
        if not first:
            axis = -nn_o[q]
            h = hj_o[q]
        u = u_o[q]
        v = v_o[q]
        cs = h/rs
        sn = rho_o[q]/rs
        for m in range(narc[q]):
            ta = at0[q, m]
            tb = at1[q, m]
            if first:
                ta = at1[q, m]
                tb = at0[q, m]
            D = tb-ta
            npc = int(wp.ceil(wp.abs(D)/piece))
            if npc < 1:
                npc = 1
            for kk in range(npc):
                pa = ta+D*(wp.float64(kk)/wp.float64(npc))
                pb = ta+D*(wp.float64(kk+1)/wp.float64(npc))
                ma = cs*axis+sn*emv(pa, u, v)
                mm = cs*axis+sn*emv(half_*(pa+pb), u, v)
                for ci in range(nfib):
                    sc[ci] = wp.min(sc[ci], wp.min(wp.dot(fib[ci], ma), wp.dot(fib[ci], mm)))
                sc[nfib] = wp.min(sc[nfib], wp.min(wp.dot(nd, ma), wp.dot(nd, mm)))
                sc[nfib+1] = wp.min(sc[nfib+1], wp.min(-wp.dot(nd, ma), -wp.dot(nd, mm)))
    if npn == zero:
        sc[nfib] = -big*wp.float64(100.0)
        sc[nfib+1] = -big*wp.float64(100.0)
    best = int(0)
    bestv = -big*wp.float64(10.0)
    for ci in range(ncand):
        if sc[ci] > bestv:
            bestv = sc[ci]
            best = ci
    pole = nd
    if best < nfib:
        pole = fib[best]
    elif best == nfib+1:
        pole = -nd
    pole_o[s] = pole
    # pass 3: the area, fan from the pole
    Au = zero
    for e in range(off[s], off[s+1]):
        t = ent[e]
        q = t % npr
        first = t < npr
        axis = nn_o[q]
        h = hi_o[q]
        if not first:
            axis = -nn_o[q]
            h = hj_o[q]
        u = u_o[q]
        v = v_o[q]
        cs = h/rs
        sn = rho_o[q]/rs
        for m in range(narc[q]):
            ta = at0[q, m]
            tb = at1[q, m]
            if first:
                ta = at1[q, m]
                tb = at0[q, m]
            D = tb-ta
            sg = zero
            if D > zero:
                sg = one
            elif D < zero:
                sg = -one
            npc = int(wp.ceil(wp.abs(D)/piece))
            if npc < 1:
                npc = 1
            for kk in range(npc):
                pa = ta+D*(wp.float64(kk)/wp.float64(npc))
                pb = ta+D*(wp.float64(kk+1)/wp.float64(npc))
                ma = cs*axis+sn*emv(pa, u, v)
                mb = cs*axis+sn*emv(pb, u, v)
                Om = wp.float64(2.0)*wp.atan2(wp.dot(pole, wp.cross(ma, mb)), one+wp.dot(pole, ma)+wp.dot(ma, mb)+wp.dot(mb, pole))
                tau = sg*(-wp.sin(pa)*u+wp.cos(pa)*v)
                gg = mb-wp.dot(ma, mb)*ma
                if stable_area == 1:
                    # Same spherical-sector formula, without subtracting
                    # coincident endpoints. Ground clipping creates arcs as
                    # short as 1e-15 radians at degenerate power vertices.
                    mid = half_*(pa+pb)
                    tangent_mid = -wp.sin(mid)*u+wp.cos(mid)*v
                    chord = wp.float64(2.0)*sn*wp.sin(half_*(pb-pa))*tangent_mid
                    Om = wp.float64(2.0)*wp.atan2(wp.dot(pole, wp.cross(ma, chord)),
                        wp.float64(2.0)+wp.float64(2.0)*wp.dot(pole, ma)+wp.dot(pole, chord)+wp.dot(ma, chord))
                    gg = sg*(tangent_mid-wp.dot(ma,tangent_mid)*ma)
                bet = wp.atan2(wp.dot(wp.cross(tau, gg), ma), wp.dot(tau, gg))
                Au = Au+Om+wp.float64(2.0)*bet+cs*wp.abs(pb-pa)
    # is the pole's antipode exposed
    if hidden[s] == 0:
        ex = int(1)
        if floor_enabled == 1 and p[s][1]-rs*pole[1] < floor_height:
            ex = 0
        for b in range(nbrs.shape[1]):
            k = nbrs[s, b]
            if k >= 0:
                dX = p[s]-p[k]-rs*pole
                if not (wp.dot(dX, dX)-r[k]*r[k] > zero):
                    ex = 0
        if ex == 1:
            Au = Au+fourpi
    pA = rs*rs*Au
    pA_o[s] = pA
    V_o[s] = (Vw[s]+rs*pA)/wp.float64(3.0)


class GeometryWarp:
    """the values of gpu_geometry.Geometry(p, r, S) that k1's run reads (see the module docstring)"""
    _fib = {}

    def __init__(self, p, r, S, piece=math.pi/6, npole=32, integrate_sites=True):
        dev = p.device
        self.p, self.r, self.n = p, r, len(p)
        with torch.no_grad():
            P0 = S['pairs']; i0, j0 = P0[:, 0], P0[:, 1]
            L0 = (p[j0]-p[i0]).norm(dim=1)
            h0 = (L0*L0+r[i0]**2-r[j0]**2)/(2*torch.where(L0 > 0, L0, torch.ones_like(L0)))
            keep = (L0 > 0) & (r[i0]**2-h0*h0 > 0)
            self.P, self.Cn, self.nbrs, self.hidden = P0[keep], S['links'][keep], S['nbrs'], S['hidden']
        n, npr = self.n, len(self.P); K = self.Cn.shape[1]
        self.i, self.j = self.P[:, 0], self.P[:, 1]
        F = lambda *s_: wp.zeros(s_, dtype=wp.float64, device=str(dev)) if len(s_) > 1 else wp.zeros(s_[0], dtype=wp.float64, device=str(dev))
        V3 = lambda k_: wp.zeros(k_, dtype=wp.vec3d, device=str(dev))
        I32 = lambda t_: wp.from_torch(t_.to(torch.int32).contiguous(), dtype=wp.int32)
        pw, rw = wp.from_torch(p.detach().contiguous(), dtype=wp.vec3d), wp.from_torch(r.detach().contiguous(), dtype=wp.float64)
        Pw, Cw = wp.from_torch(self.P.to(torch.int32).contiguous(), dtype=wp.int32), wp.from_torch(self.Cn.to(torch.int32).contiguous(), dtype=wp.int32)
        ea, ed = F(max(npr, 1), 2*K), F(max(npr, 1), 2*K); el = wp.zeros((max(npr, 1), 2*K), dtype=wp.int32, device=str(dev))
        nn, c, u, v = V3(npr), V3(npr), V3(npr), V3(npr); rho, hi, hj = F(npr), F(npr), F(npr)
        at0, at1 = F(max(npr, 1), 2*K+1), F(max(npr, 1), 2*K+1); narc = wp.zeros(npr, dtype=wp.int32, device=str(dev))
        wA, wc, Vw = F(npr), V3(npr), F(n)
        if npr:
            wp.launch(k_geo_pair, npr, inputs=[pw, rw, Pw, Cw, K, TWO_PI, BIG, 1e299, .9, ea, ed, el], outputs=[nn, c, u, v, rho, hi, hj, at0, at1, narc, wA, wc, Vw], device=str(dev))
        # each site's (pair, side) entries: side i as q, side j as npr + q
        site = torch.cat((self.i, self.j)); o = torch.argsort(site, stable=True)
        cnt = torch.bincount(site, minlength=n); off = torch.zeros(n+1, dtype=torch.int64, device=dev); off[1:] = torch.cumsum(cnt, 0)
        assert npole+2 == NCAND, 'k_geo_site keeps NCAND candidates'
        key = (str(dev), npole)
        if key not in GeometryWarp._fib:
            k_ = torch.arange(npole, dtype=torch.float64, device=dev)+.5
            ph = torch.acos(1-2*k_/npole); th = math.pi*(1+5**.5)*k_
            GeometryWarp._fib[key] = wp.from_torch(torch.stack((torch.cos(th)*torch.sin(ph), torch.sin(th)*torch.sin(ph), torch.cos(ph)), 1).contiguous(), dtype=wp.vec3d)
        pn, pA, V, pole = V3(n), F(n), F(n), V3(n)
        if integrate_sites:
            wp.launch(k_geo_site, n, inputs=[pw, rw, I32(off), I32(o), npr, nn, u, v, rho, hi, hj, at0, at1, narc, GeometryWarp._fib[key], npole, float(piece), BIG, 4*math.pi,
                                             I32(self.nbrs), I32(self.hidden), 0, 0, 0.0, I32(o), 0], outputs=[Vw, pn, pA, V, pole], device=str(dev))
        T = lambda a_: wp.to_torch(a_, requires_grad=False)
        self.nn, self.c, self.u, self.v, self.rho, self.hi, self.hj = T(nn), T(c), T(u), T(v), T(rho), T(hi), T(hj)
        self.wall_A, self.wall_c, self.patch_n, self.patch_A, self.V, self.pole = T(wA), T(wc), T(pn), T(pA), T(V), T(pole)
        self.narc, self.arc_t0, self.arc_t1 = T(narc), T(at0), T(at1)
