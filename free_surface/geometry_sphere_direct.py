"""Experimental exact small-circle boundary integral; same geometric inputs as k_geo_site.

The solid-angle gauge retains the existing exposed-antipode correction, so no
assumption of a connected spherical patch or disk topology is introduced.
"""
import warp as wp
from free_surface.geometry_warp import emv, vecC

@wp.kernel(enable_backward=False)
def k_geo_site_direct(p: wp.array(dtype=wp.vec3d), r: wp.array(dtype=wp.float64), off: wp.array(dtype=wp.int32), ent: wp.array(dtype=wp.int32), npr: int,
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
    # A conservative analytic bound over each *whole* boundary circle.
    # The gauge is safe if 1+pole.m stays positive. Eight candidates suffice
    # in ordinary cells; difficult cells retain the original algorithm.
    npn = wp.length(pn)
    nd = wp.vec3d()
    if npn > zero:
        nd = pn/npn
    pole = wp.vec3d(one,zero,zero)
    clearance = -one
    for ci in range(8):
        candidate = wp.vec3d()
        if ci < 6:
            candidate[ci//2] = one
            if ci%2 == 1:
                candidate = -candidate
        elif ci == 6:
            candidate = nd
        else:
            candidate = -nd
        least = wp.float64(2.0)
        if ci >= 6 and npn == zero:
            least = -one
        for e in range(off[s],off[s+1]):
            t = ent[e]
            q = t%npr
            if narc[q] > 0:
                c = hi_o[q]/rs
                if t >= npr:
                    c = -hj_o[q]/rs
                aa = wp.dot(candidate,wp.cross(u_o[q],v_o[q]))
                bound = one+c*aa-(rho_o[q]/rs)*wp.sqrt(wp.max(zero,one-aa*aa))
                least = wp.min(least,bound)
        if least > clearance:
            clearance = least
            pole = candidate
    fast_ok = clearance > wp.float64(1.e-4)
    if not fast_ok:
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
    pole_o[s] = pole
    Au = zero
    if fast_ok:
        # Exact integral of q.(m x dm)/(1+q.m) over each complete small-circle arc.
        # Use the basis normal u x v: it reverses relative to axis on the second cell.
        Au = zero
        for e in range(off[s], off[s+1]):
            t = ent[e]
            q = t % npr
            first = t < npr
            c = hi_o[q]/rs
            if not first:
                c = -hj_o[q]/rs
            u = u_o[q]
            v = v_o[q]
            sn = rho_o[q]/rs
            aa = wp.dot(pole, wp.cross(u,v))
            bb = sn*wp.dot(pole,u)
            cc = sn*wp.dot(pole,v)
            den0 = one+c*aa
            delta = aa+c
            sign_delta = one
            if delta < zero:
                sign_delta = -one
            for m in range(narc[q]):
                ta = at0[q,m]
                tb = at1[q,m]
                if first:
                    ta = at1[q,m]
                    tb = at0[q,m]
                D = tb-ta
                mid = half_*(ta+tb)
                angle = wp.atan2(wp.abs(delta)*wp.sin(half_*D),
                    den0*wp.cos(half_*D)+bb*wp.cos(mid)+cc*wp.sin(mid))
                Au = Au+wp.float64(2.0)*sign_delta*angle-c*D
    else:
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

