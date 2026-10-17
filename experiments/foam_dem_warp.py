"""Fused Warp rollout of EnergyFoam (Neo-Hookean bodies, dashpots, gravity, no-slip ground).

Same model, formulas and step order as experiments/foam_dem_energy.py (EnergyFoam.forces,
damping_forces, step with ground_mode='noslip'), but each symplectic-Euler step is four
kernels (bonds: elastic + dashpot forces and the least-squares sums S; bodies: Neo-Hookean
stress G = dE/dF M^-1; bonds: body-energy forces; bodies: integration + ground) instead of
dozens of torch ops. model.kinematics='exact' uses the geometrically exact bond strains (BOND_EXACT.md:
log-map relative rotation, mid-frame stretch; bond_kernel_exact_det). deterministic=True (default) replaces the atomic scatters by per-bond
outputs gathered per body in a fixed order (two extra kernels), so the forward is bitwise
reproducible (the atomic version differs run to run by float rounding, amplified by the
hard ground switch to ~0.5 mm / ~12% gradient after a few frames). Warp generates the adjoint kernels; WarpRollout records one tape per
rollout with per-step state buffers and exposes the result to torch autograd (theta only).
Quaternions are stored (w, x, y, z) as vec4 to match the torch code.
"""
import numpy as np
import torch
import warp as wp


@wp.func
def qmat(q: wp.vec4):
    w, x, y, z = q[0], q[1], q[2], q[3]
    return wp.mat33(1.0-2.0*(y*y+z*z), 2.0*(x*y-w*z), 2.0*(x*z+w*y),
                    2.0*(x*y+w*z), 1.0-2.0*(x*x+z*z), 2.0*(y*z-w*x),
                    2.0*(x*z-w*y), 2.0*(y*z+w*x), 1.0-2.0*(x*x+y*y))


@wp.func
def qmul(a: wp.vec4, b: wp.vec4):
    return wp.vec4(a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3], a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2],
                   a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1], a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0])


@wp.func
def cols(a: wp.vec3, b: wp.vec3, c: wp.vec3):
    return wp.mat33(a[0], b[0], c[0], a[1], b[1], c[1], a[2], b[2], c[2])


@wp.kernel
def bond_kernel(x: wp.array(dtype=wp.vec3), q: wp.array(dtype=wp.vec4), v: wp.array(dtype=wp.vec3),
                w: wp.array(dtype=wp.vec3), theta: wp.array2d(dtype=float),
                bi: wp.array(dtype=int), bj: wp.array(dtype=int), n0: wp.array(dtype=wp.vec3),
                ri: wp.array(dtype=wp.vec3), rj: wp.array(dtype=wp.vec3), base: wp.array2d(dtype=float),
                cn: wp.array(dtype=float), ct: wp.array(dtype=float), cr: wp.array(dtype=float),
                lw: wp.array(dtype=float), dX: wp.array(dtype=wp.vec3), damping: int,
                f: wp.array(dtype=wp.vec3), tau: wp.array(dtype=wp.vec3), S: wp.array(dtype=wp.mat33)):
    e = wp.tid()
    i, j = bi[e], bj[e]
    Ri, Rj = qmat(q[i]), qmat(q[j])
    pi, pj = Ri@ri[e], Rj@rj[e]
    D = x[j]+pj-x[i]-pi
    d = 0.5*(wp.transpose(Ri)@D+wp.transpose(Rj)@D)
    Psi = wp.transpose(Ri)@Rj
    phi = 0.5*wp.vec3(Psi[2, 1]-Psi[1, 2], Psi[0, 2]-Psi[2, 0], Psi[1, 0]-Psi[0, 1])
    n = n0[e]
    k0 = base[e, 0]*wp.exp(0.5*(theta[i, 0]+theta[j, 0]))
    k1 = base[e, 1]*wp.exp(0.5*(theta[i, 1]+theta[j, 1]))
    k2 = base[e, 2]*wp.exp(0.5*(theta[i, 2]+theta[j, 2]))
    k3 = base[e, 3]*wp.exp(0.5*(theta[i, 3]+theta[j, 3]))
    dn = wp.dot(d, n)
    pn = wp.dot(phi, n)
    gd = k0*dn*n+k1*(d-dn*n)
    gp = k3*pn*n+k2*(phi-pn*n)
    Rgi, Rgj = Ri@gd, Rj@gd
    a = 0.5*(Rgi+Rgj)
    tr = Psi[0, 0]+Psi[1, 1]+Psi[2, 2]
    M = 0.5*(tr*gp-wp.transpose(Psi)@gp)
    T = Ri@M
    tj = wp.cross(pj, a)-0.5*wp.cross(D, Rgj)+T
    ti = -wp.cross(pi, a)-0.5*wp.cross(D, Rgi)-T
    fi = a
    fj = -a
    taui = -ti
    tauj = -tj
    if damping != 0:
        scale = wp.exp(0.5*(theta[i, 6]+theta[j, 6]))
        vrel = v[j]+wp.cross(w[j], pj)-v[i]-wp.cross(w[i], pi)
        axis = wp.normalize(x[j]-x[i])
        vn = wp.dot(vrel, axis)*axis
        fd = -scale*(cn[e]*vn+ct[e]*(vrel-vn))
        spin = scale*cr[e]*(w[j]-w[i])
        fj = fj+fd
        fi = fi-fd
        tauj = tauj+wp.cross(pj, fd)-spin
        taui = taui+wp.cross(pi, -fd)+spin
    wp.atomic_add(f, i, fi)
    wp.atomic_add(f, j, fj)
    wp.atomic_add(tau, i, taui)
    wp.atomic_add(tau, j, tauj)
    outer = lw[e]*wp.outer(x[j]-x[i], dX[e])   # both directions give the same product
    wp.atomic_add(S, i, outer)
    wp.atomic_add(S, j, outer)


@wp.func
def nh_bond_force(dn: float, L: float, k: float, nl: int, alpha: float):
    """Normal bond force. nl = 0: linear k dn. nl = 1: bond stretch lam = 1 + dn / L with the one-term Ogden
    incompressible uniaxial energy U = 2 k L^2 / (3 alpha^2) (lam^alpha + 2 lam^(-alpha/2) - 3) (a virtual-internal-
    bond potential; alpha = 2 is neo-Hookean): stiffness k at lam = 1 for every alpha, stiffening in compression (and,
    for alpha > 2, in tension) the more the larger alpha; continued quadratically below lam = 0.2 (finite when crushed)."""
    if nl == 0:
        return k*dn
    lam = 1.0+dn/L
    m = float(0.2)
    c = 2.0*k/(3.0*alpha)
    if lam < m:
        fm = c*L*(wp.pow(m, alpha-1.0)-wp.pow(m, -0.5*alpha-1.0))
        dfm = c*((alpha-1.0)*wp.pow(m, alpha-2.0)+(0.5*alpha+1.0)*wp.pow(m, -0.5*alpha-2.0))
        return fm+dfm*L*(lam-m)
    return c*L*(wp.pow(lam, alpha-1.0)-wp.pow(lam, -0.5*alpha-1.0))

@wp.kernel
def bond_kernel_det(x: wp.array(dtype=wp.vec3), q: wp.array(dtype=wp.vec4), v: wp.array(dtype=wp.vec3),
                w: wp.array(dtype=wp.vec3), theta: wp.array2d(dtype=float),
                bi: wp.array(dtype=int), bj: wp.array(dtype=int), n0: wp.array(dtype=wp.vec3),
                ri: wp.array(dtype=wp.vec3), rj: wp.array(dtype=wp.vec3), base: wp.array2d(dtype=float),
                cn: wp.array(dtype=float), ct: wp.array(dtype=float), cr: wp.array(dtype=float),
                lw: wp.array(dtype=float), dX: wp.array(dtype=wp.vec3), damping: int,
                bf: wp.array(dtype=wp.vec3), bti: wp.array(dtype=wp.vec3), btj: wp.array(dtype=wp.vec3),
                    bS: wp.array(dtype=wp.mat33), nl: int, alpha0: float, oc: int):
    e = wp.tid()
    i, j = bi[e], bj[e]
    Ri, Rj = qmat(q[i]), qmat(q[j])
    pi, pj = Ri@ri[e], Rj@rj[e]
    D = x[j]+pj-x[i]-pi
    d = 0.5*(wp.transpose(Ri)@D+wp.transpose(Rj)@D)
    Psi = wp.transpose(Ri)@Rj
    phi = 0.5*wp.vec3(Psi[2, 1]-Psi[1, 2], Psi[0, 2]-Psi[2, 0], Psi[1, 0]-Psi[0, 1])
    n = n0[e]
    k0 = base[e, 0]*wp.exp(0.5*(theta[i, 0]+theta[j, 0]))
    k1 = base[e, 1]*wp.exp(0.5*(theta[i, 1]+theta[j, 1]))
    k2 = base[e, 2]*wp.exp(0.5*(theta[i, 2]+theta[j, 2]))
    k3 = base[e, 3]*wp.exp(0.5*(theta[i, 3]+theta[j, 3]))
    dn = wp.dot(d, n)
    pn = wp.dot(phi, n)
    alpha = alpha0
    if oc >= 0:
        alpha = alpha0*wp.exp(0.5*(theta[i, oc]+theta[j, oc]))
    gd = nh_bond_force(dn, wp.length(dX[e]), k0, nl, alpha)*n+k1*(d-dn*n)
    gp = k3*pn*n+k2*(phi-pn*n)
    Rgi, Rgj = Ri@gd, Rj@gd
    a = 0.5*(Rgi+Rgj)
    tr = Psi[0, 0]+Psi[1, 1]+Psi[2, 2]
    M = 0.5*(tr*gp-wp.transpose(Psi)@gp)
    T = Ri@M
    tj = wp.cross(pj, a)-0.5*wp.cross(D, Rgj)+T
    ti = -wp.cross(pi, a)-0.5*wp.cross(D, Rgi)-T
    fi = a
    fj = -a
    taui = -ti
    tauj = -tj
    if damping != 0:
        scale = wp.exp(0.5*(theta[i, 6]+theta[j, 6]))
        vrel = v[j]+wp.cross(w[j], pj)-v[i]-wp.cross(w[i], pi)
        axis = wp.normalize(x[j]-x[i])
        vn = wp.dot(vrel, axis)*axis
        fd = -scale*(cn[e]*vn+ct[e]*(vrel-vn))
        spin = scale*cr[e]*(w[j]-w[i])
        fj = fj+fd
        fi = fi-fd
        tauj = tauj+wp.cross(pj, fd)-spin
        taui = taui+wp.cross(pi, -fd)+spin
    bf[e] = fi                                   # force on j is -fi (fj = -a + fd = -fi)
    bti[e] = taui
    btj[e] = tauj
    bS[e] = lw[e]*wp.outer(x[j]-x[i], dX[e])


@wp.kernel
def bond_kernel_det_visco(x: wp.array(dtype=wp.vec3), q: wp.array(dtype=wp.vec4), v: wp.array(dtype=wp.vec3),
                w: wp.array(dtype=wp.vec3), theta: wp.array2d(dtype=float),
                bi: wp.array(dtype=int), bj: wp.array(dtype=int), n0: wp.array(dtype=wp.vec3),
                ri: wp.array(dtype=wp.vec3), rj: wp.array(dtype=wp.vec3), base: wp.array2d(dtype=float),
                lw: wp.array(dtype=float), dX: wp.array(dtype=wp.vec3),
                zd: wp.array(dtype=wp.vec3), zp: wp.array(dtype=wp.vec3), g0: float, tau0: float, cg: int, ctau: int, dt: float,
                bf: wp.array(dtype=wp.vec3), bti: wp.array(dtype=wp.vec3), btj: wp.array(dtype=wp.vec3),
                bS: wp.array(dtype=wp.mat33), zd1: wp.array(dtype=wp.vec3), zp1: wp.array(dtype=wp.vec3), nl: int, alpha0: float, oc: int):
    """bond_kernel_det (small kinematics, no explicit dashpot) plus a Maxwell branch on every bond spring (Zener /
    standard linear solid): energy 1/2 k (s)^2 + g/2 k (s - z)^2 for each spring measure s (normal / tangential
    displacement d, bending / twist rotation phi), with the branch's viscous internal variable z relaxing towards s,
    dz/dt = (s - z) / tau, integrated exactly over the step with s held (z1 = s + (z - s) e^{-dt/tau}).
    g = g0 e^{theta[:, cg]}, tau = tau0 e^{theta[:, ctau]} (bond averages). Instantaneous stiffness (1 + g) k,
    relaxed stiffness k."""
    e = wp.tid()
    i, j = bi[e], bj[e]
    Ri, Rj = qmat(q[i]), qmat(q[j])
    pi, pj = Ri@ri[e], Rj@rj[e]
    D = x[j]+pj-x[i]-pi
    d = 0.5*(wp.transpose(Ri)@D+wp.transpose(Rj)@D)
    Psi = wp.transpose(Ri)@Rj
    phi = 0.5*wp.vec3(Psi[2, 1]-Psi[1, 2], Psi[0, 2]-Psi[2, 0], Psi[1, 0]-Psi[0, 1])
    n = n0[e]
    k0 = base[e, 0]*wp.exp(0.5*(theta[i, 0]+theta[j, 0]))
    k1 = base[e, 1]*wp.exp(0.5*(theta[i, 1]+theta[j, 1]))
    k2 = base[e, 2]*wp.exp(0.5*(theta[i, 2]+theta[j, 2]))
    k3 = base[e, 3]*wp.exp(0.5*(theta[i, 3]+theta[j, 3]))
    g = g0*wp.exp(0.5*(theta[i, cg]+theta[j, cg]))
    tau = tau0*wp.exp(0.5*(theta[i, ctau]+theta[j, ctau]))
    ed = d-zd[e]
    ep = phi-zp[e]
    dn = wp.dot(d, n)
    pn = wp.dot(phi, n)
    edn = wp.dot(ed, n)
    epn = wp.dot(ep, n)
    alpha = alpha0
    if oc >= 0:
        alpha = alpha0*wp.exp(0.5*(theta[i, oc]+theta[j, oc]))
    gd = nh_bond_force(dn, wp.length(dX[e]), k0, nl, alpha)*n+k1*(d-dn*n)+g*(k0*edn*n+k1*(ed-edn*n))
    gp = k3*pn*n+k2*(phi-pn*n)+g*(k3*epn*n+k2*(ep-epn*n))
    Rgi, Rgj = Ri@gd, Rj@gd
    a = 0.5*(Rgi+Rgj)
    tr = Psi[0, 0]+Psi[1, 1]+Psi[2, 2]
    M = 0.5*(tr*gp-wp.transpose(Psi)@gp)
    T = Ri@M
    tj = wp.cross(pj, a)-0.5*wp.cross(D, Rgj)+T
    ti = -wp.cross(pi, a)-0.5*wp.cross(D, Rgi)-T
    bf[e] = a
    bti[e] = -ti
    btj[e] = -tj
    bS[e] = lw[e]*wp.outer(x[j]-x[i], dX[e])
    r = wp.exp(-dt/tau)
    zd1[e] = d+(zd[e]-d)*r
    zp1[e] = phi+(zp[e]-phi)*r


@wp.func
def qconj(q: wp.vec4):
    return wp.vec4(q[0], -q[1], -q[2], -q[3])


@wp.func
def qlog(q: wp.vec4):
    """Rotation vector of a unit quaternion with q_w >= 0 (angle in [0, pi])."""
    v = wp.vec3(q[1], q[2], q[3])
    s = wp.length(v)
    w = q[0]
    ratio = float(0.0)
    if s < 1.0e-6:
        ratio = 2.0/w*(1.0-s*s/(3.0*w*w))
    else:
        ratio = 2.0*wp.atan2(s, w)/s
    return ratio*v


@wp.func
def skewm(v: wp.vec3):
    return wp.mat33(0.0, -v[2], v[1], v[2], 0.0, -v[0], -v[1], v[0], 0.0)


@wp.func
def jr(phi: wp.vec3):
    """SO(3) right Jacobian: I - (1 - cos t)/t^2 [phi]x + (t - sin t)/t^3 [phi]x^2."""
    t2 = wp.dot(phi, phi)
    a = float(0.0)
    b = float(0.0)
    if t2 < 1.0e-4:
        a = 0.5-t2/24.0
        b = 1.0/6.0-t2/120.0
    else:
        t = wp.sqrt(t2)
        sh = wp.sin(0.5*t)
        a = 2.0*sh*sh/t2
        b = (t-wp.sin(t))/(t2*t)
    K = skewm(phi)
    return wp.identity(3, dtype=float)-a*K+b*(K@K)


@wp.func
def jr_inv(phi: wp.vec3):
    """Inverse right Jacobian: I + [phi]x/2 + (1/t^2 - cot(t/2)/(2t)) [phi]x^2 (finite at t = pi)."""
    t2 = wp.dot(phi, phi)
    c = float(0.0)
    if t2 < 1.0e-4:
        c = 1.0/12.0+t2/720.0
    else:
        t = wp.sqrt(t2)
        c = 1.0/t2-wp.cos(0.5*t)/(wp.sin(0.5*t)*2.0*t)
    K = skewm(phi)
    return wp.identity(3, dtype=float)+0.5*K+c*(K@K)


@wp.kernel
def bond_kernel_exact_det(x: wp.array(dtype=wp.vec3), q: wp.array(dtype=wp.vec4), v: wp.array(dtype=wp.vec3),
                w: wp.array(dtype=wp.vec3), theta: wp.array2d(dtype=float),
                bi: wp.array(dtype=int), bj: wp.array(dtype=int), n0: wp.array(dtype=wp.vec3),
                ri: wp.array(dtype=wp.vec3), rj: wp.array(dtype=wp.vec3), base: wp.array2d(dtype=float),
                cn: wp.array(dtype=float), ct: wp.array(dtype=float), cr: wp.array(dtype=float),
                lw: wp.array(dtype=float), dX: wp.array(dtype=wp.vec3), damping: int,
                bf: wp.array(dtype=wp.vec3), bti: wp.array(dtype=wp.vec3), btj: wp.array(dtype=wp.vec3),
                    bS: wp.array(dtype=wp.mat33)):
    e = wp.tid()
    i, j = bi[e], bj[e]
    qi, qj = q[i], q[j]
    Ri, Rj = qmat(qi), qmat(qj)
    pi, pj = Ri@ri[e], Rj@rj[e]
    D = x[j]+pj-x[i]-pi
    qr = qmul(qconj(qi), qj)
    if qr[0] < 0.0:
        qr = -qr
    th = qlog(qr)
    Rm = qmat(qmul(qi, wp.normalize(qr+wp.vec4(1.0, 0.0, 0.0, 0.0))))
    d = wp.transpose(Rm)@D
    n = n0[e]
    k0 = base[e, 0]*wp.exp(0.5*(theta[i, 0]+theta[j, 0]))
    k1 = base[e, 1]*wp.exp(0.5*(theta[i, 1]+theta[j, 1]))
    k2 = base[e, 2]*wp.exp(0.5*(theta[i, 2]+theta[j, 2]))
    k3 = base[e, 3]*wp.exp(0.5*(theta[i, 3]+theta[j, 3]))
    dn = wp.dot(d, n)
    tn = wp.dot(th, n)
    gd = k0*dn*n+k1*(d-dn*n)
    gt = k3*tn*n+k2*(th-tn*n)
    G = Rm@gd
    hh = wp.cross(G, D)
    u = wp.transpose(Rm)@hh
    c = wp.transpose(jr_inv(th))@(gt+0.5*(wp.transpose(jr(0.5*th))@u))
    Rjc = Rj@c
    dEj = Rjc+wp.cross(pj, G)
    dEi = -Rjc+hh-wp.cross(pi, G)
    fi = G
    fj = -G
    taui = -dEi
    tauj = -dEj
    if damping != 0:
        scale = wp.exp(0.5*(theta[i, 6]+theta[j, 6]))
        vrel = v[j]+wp.cross(w[j], pj)-v[i]-wp.cross(w[i], pi)
        axis = wp.normalize(x[j]-x[i])
        vn = wp.dot(vrel, axis)*axis
        fd = -scale*(cn[e]*vn+ct[e]*(vrel-vn))
        spin = scale*cr[e]*(w[j]-w[i])
        fj = fj+fd
        fi = fi-fd
        tauj = tauj+wp.cross(pj, fd)-spin
        taui = taui+wp.cross(pi, -fd)+spin
    bf[e] = fi                                   # force on j is -fi (fj = -a + fd = -fi)
    bti[e] = taui
    btj[e] = tauj
    bS[e] = lw[e]*wp.outer(x[j]-x[i], dX[e])


@wp.kernel
def gather_kernel(off: wp.array(dtype=int), inc: wp.array(dtype=int), side: wp.array(dtype=float),
                  bf: wp.array(dtype=wp.vec3), bti: wp.array(dtype=wp.vec3), btj: wp.array(dtype=wp.vec3),
                  bS: wp.array(dtype=wp.mat33),
                  f: wp.array(dtype=wp.vec3), tau: wp.array(dtype=wp.vec3), S: wp.array(dtype=wp.mat33)):
    """Deterministic sums: each body adds its bonds in a fixed (CSR) order; side = +1 if it is
    the bond's i end, -1 if its j end."""
    b = wp.tid()
    fs = wp.vec3(0.0, 0.0, 0.0)
    ts = wp.vec3(0.0, 0.0, 0.0)
    Ss = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    for k in range(off[b], off[b+1]):
        e = inc[k]
        sg = side[k]
        fs = fs+sg*bf[e]
        if sg > 0.0:
            ts = ts+bti[e]
        else:
            ts = ts+btj[e]
        Ss = Ss+bS[e]
    f[b] = fs
    tau[b] = ts
    S[b] = Ss


@wp.kernel
def body_kernel(S: wp.array(dtype=wp.mat33), Minv: wp.array(dtype=wp.mat33), theta: wp.array2d(dtype=float),
                vol: wp.array(dtype=float), k_v: float, k_mu: float, barrier: float, G: wp.array(dtype=wp.mat33)):
    b = wp.tid()
    F = S[b]@Minv[b]
    c0 = wp.vec3(F[0, 0], F[1, 0], F[2, 0])
    c1 = wp.vec3(F[0, 1], F[1, 1], F[2, 1])
    c2 = wp.vec3(F[0, 2], F[1, 2], F[2, 2])
    cof = cols(wp.cross(c1, c2), wp.cross(c2, c0), wp.cross(c0, c1))
    J = wp.dot(c0, wp.cross(c1, c2))
    kv = k_v*wp.exp(theta[b, 4])*vol[b]
    mu = k_mu*wp.exp(theta[b, 5])*vol[b]
    m = barrier
    L = float(0.0)
    dL = float(0.0)
    if J < m:
        L = wp.log(m)+(J-m)/m-(J-m)*(J-m)/(2.0*m*m)
        dL = 1.0/m-(J-m)/(m*m)
    else:
        L = wp.log(J)
        dL = 1.0/J
    stress = mu*F+((kv*L-mu)*dL)*cof
    G[b] = stress@Minv[b]


@wp.kernel
def volume_kernel(G: wp.array(dtype=wp.mat33), bi: wp.array(dtype=int), bj: wp.array(dtype=int),
                  lw: wp.array(dtype=float), dX: wp.array(dtype=wp.vec3), f: wp.array(dtype=wp.vec3)):
    e = wp.tid()
    i, j = bi[e], bj[e]
    g1 = lw[e]*(G[i]@dX[e])     # direction a = i, b = j
    g2 = lw[e]*(G[j]@(-dX[e]))  # direction a = j, b = i
    wp.atomic_add(f, i, g1-g2)
    wp.atomic_add(f, j, g2-g1)


@wp.kernel
def volume_kernel_det(G: wp.array(dtype=wp.mat33), bi: wp.array(dtype=int), bj: wp.array(dtype=int),
                      lw: wp.array(dtype=float), dX: wp.array(dtype=wp.vec3), bg: wp.array(dtype=wp.vec3)):
    e = wp.tid()
    i, j = bi[e], bj[e]
    bg[e] = lw[e]*(G[i]@dX[e])-lw[e]*(G[j]@(-dX[e]))     # force on i; on j it is the negative


@wp.kernel
def gather_add_kernel(off: wp.array(dtype=int), inc: wp.array(dtype=int), side: wp.array(dtype=float),
                      bg: wp.array(dtype=wp.vec3), f_el: wp.array(dtype=wp.vec3), f: wp.array(dtype=wp.vec3)):
    b = wp.tid()
    fs = f_el[b]
    for k in range(off[b], off[b+1]):
        fs = fs+side[k]*bg[inc[k]]
    f[b] = fs


@wp.kernel
def integrate_kernel(x: wp.array(dtype=wp.vec3), q: wp.array(dtype=wp.vec4), v: wp.array(dtype=wp.vec3),
                     w: wp.array(dtype=wp.vec3), f: wp.array(dtype=wp.vec3), tau: wp.array(dtype=wp.vec3),
                     mass: wp.array(dtype=float), inertia: wp.array(dtype=float), fixed: wp.array(dtype=int),
                     contact: wp.array(dtype=float), gravity: wp.vec3, ground_p: wp.vec3, ground_n: wp.vec3,
                     use_ground: int, layer_dx: float, dt_ref: float, theta: wp.array2d(dtype=float),
                     friction0: float, restitution0: float, dt: float,
                     x1: wp.array(dtype=wp.vec3), q1: wp.array(dtype=wp.vec4), v1: wp.array(dtype=wp.vec3),
                     w1: wp.array(dtype=wp.vec3)):
    b = wp.tid()
    vn = v[b]
    wn = w[b]
    if fixed[b] == 0:
        vn = v[b]+dt*(f[b]/mass[b]+gravity)
        wn = w[b]+dt*tau[b]/inertia[b]
    if use_ground == 1:
        if wp.dot(x[b]-ground_p, ground_n) <= contact[b]:
            vn = wp.max(wp.dot(vn, ground_n), 0.0)*ground_n
            wn = wp.vec3(0.0, 0.0, 0.0)
    if use_ground == 2:
        # smooth contact: the velocity approaches a contact target at a rate set by W(h), the CDF
        # of PAC-NeRF's quadratic B-spline over the contact height h in units of layer_dx (1 at or
        # below -0.5, 0 above 1.5); the difference to the target is scaled by (1 - W)^(dt / dt_ref)
        # per step, so the per-time rate does not depend on dt. Target: tangential velocity and w
        # scaled by (1 - friction), inward normal velocity reflected to -restitution x itself,
        # outward kept. friction = sigmoid(theta[b, 7]), restitution = sigmoid(theta[b, 8] - logit
        # offset) when theta has 9 columns, else 1 and 0 (the hard rule's target).
        s = (wp.dot(x[b]-ground_p, ground_n)-contact[b])/layer_dx
        W = float(0.0)
        if s < -0.5:
            W = 1.0
        elif s < 0.5:
            W = 1.0-0.5*(s+0.5)*(s+0.5)
        elif s < 1.5:
            W = 0.5*(1.5-s)*(1.5-s)
        keep = wp.pow(1.0-W+1.0e-6, dt/dt_ref)            # smooth floor (a max() floor left a kink in the derivative)
        mu_f = float(1.0)
        rest = float(0.0)
        if theta.shape[1] > 8:
            mu_f = 1.0/(1.0+wp.exp(-(theta[b, 7]+friction0)))
            rest = 1.0/(1.0+wp.exp(-(theta[b, 8]+restitution0)))
        un = wp.dot(vn, ground_n)
        ut = vn-un*ground_n
        tn = un
        if un < 0.0:
            tn = -rest*un
        target = tn*ground_n+(1.0-mu_f)*ut
        vn = target+keep*(vn-target)
        tw = (1.0-mu_f)*wn
        wn = tw+keep*(wn-tw)
    if use_ground == 3:
        # Coulomb ground (the standard MPM friction boundary, applied to the body velocity): the velocity into the
        # ground is removed, scaled by the contact weight W(h) (W as above: the gap is smoothed over the layer, the
        # impulses are not), and the tangential speed is reduced by at most mu_f times the removed normal speed
        # (stick when that suffices, slide otherwise). No restitution parameter: the bounce comes from the elasticity
        # of the body; spins are left free (no couple at the boundary). mu_f = exp(theta[b, 7] + friction0),
        # friction0 = log(mu_f at theta = 0).
        s = (wp.dot(x[b]-ground_p, ground_n)-contact[b])/layer_dx
        W = float(0.0)
        if s < -0.5:
            W = 1.0
        elif s < 0.5:
            W = 1.0-0.5*(s+0.5)*(s+0.5)
        elif s < 1.5:
            W = 0.5*(1.5-s)*(1.5-s)
        un = wp.dot(vn, ground_n)
        if un < 0.0 and W > 0.0:
            mu_f = float(0.5)
            if theta.shape[1] > 7:
                mu_f = wp.exp(theta[b, 7]+friction0)
            dn = -W*un                                     # removed normal speed (>= 0)
            ut = vn-un*ground_n
            st = wp.length(ut)
            red = wp.min(st, mu_f*dn)                      # Coulomb: tangential change <= mu_f x normal change
            ut_new = ut
            if st > 1.0e-12:
                ut_new = ut*((st-red)/st)
            vn = (un+dn)*ground_n+ut_new
    xn = x[b]
    if fixed[b] == 0:
        xn = x[b]+dt*vn
    h = 0.5*dt*wn
    qs = wp.normalize(wp.vec4(1.0, h[0], h[1], h[2]))
    x1[b] = xn
    v1[b] = vn
    w1[b] = wn
    q1[b] = wp.normalize(qmul(qs, q[b]))


@wp.kernel
def hertz_integrate_kernel(x: wp.array(dtype=wp.vec3), q: wp.array(dtype=wp.vec4), v: wp.array(dtype=wp.vec3),
                           w: wp.array(dtype=wp.vec3), hc: wp.array(dtype=wp.vec4), f: wp.array(dtype=wp.vec3),
                           tau: wp.array(dtype=wp.vec3), mass: wp.array(dtype=float), inertia: wp.array(dtype=float),
                           fixed: wp.array(dtype=int), contact: wp.array(dtype=float), gravity: wp.vec3, ground_p: wp.vec3,
                           ground_n: wp.vec3, theta: wp.array2d(dtype=float), k_v: float, mu_ref: float, friction0: float,
                           kscale: float, coc: int, thornton: int, dt: float,
                           x1: wp.array(dtype=wp.vec3), q1: wp.array(dtype=wp.vec4), v1: wp.array(dtype=wp.vec3),
                           w1: wp.array(dtype=wp.vec3), hc1: wp.array(dtype=wp.vec4)):
    """Hertz-Mindlin contact of each body's contact sphere (radius R = contact[b]) with the rigid ground, then the
    symplectic Euler update (kscale multiplies F_n and k_t; 1 = the sphere-on-plane values). Normal (Hertz): F_n = 4/3 E* sqrt(R) d^1.5, d = penetration, E* = E / (1 - nu^2) of the
    body's own material (lam = k_v e^theta4, mu = mu_ref e^theta5); conservative, no restitution parameter. Tangential
    (Mindlin no-slip stiffness k_t = 8 G* sqrt(R d), G* = G / (2 - nu)), incremental with history: the tangential force
    of the previous step is rescaled by sqrt(d / d_prev) when the normal load decreases (Thornton et al. 2013), the
    contact-point slip increment adds -k_t v_s dt, and the result is capped at mu_f F_n (Coulomb, sliding);
    mu_f = exp(theta[b, 7] + friction0). Acts at the contact point x - R n (torque). hc = (F_t, d_prev) is state."""
    b = wp.tid()
    n = ground_n
    R = contact[b]
    reach = R
    if coc >= 0:                       # learned contact-surface offset of this body (m, + = reaches further down); no lower
        reach = R+theta[b, coc]        # bound: a negative reach moves the contact surface inside the sphere, so the surface
                                       # follows the offset smoothly for any value (F_n still uses the sphere radius R)
    d = reach-wp.dot(x[b]-ground_p, n)
    Fc = wp.vec3(0.0, 0.0, 0.0)
    Tc = wp.vec3(0.0, 0.0, 0.0)
    Ft_new = wp.vec3(0.0, 0.0, 0.0)
    d_store = float(0.0)
    if d > 0.0:
        lam = k_v*wp.exp(theta[b, 4])
        mu = mu_ref*wp.exp(theta[b, 5])
        nu = 0.5*lam/(lam+mu)
        E = mu*(3.0*lam+2.0*mu)/(lam+mu)
        Es = E/(1.0-nu*nu)
        Gs = mu/(2.0-nu)
        sR = wp.sqrt(R)
        sd = wp.sqrt(d)
        Fn = kscale*4.0/3.0*Es*sR*d*sd
        kt = kscale*8.0*Gs*sR*sd
        h = hc[b]
        Ft = wp.vec3(h[0], h[1], h[2])
        Ft = Ft-wp.dot(Ft, n)*n
        d_prev = h[3]
        if thornton != 0 and d_prev > 0.0 and d < d_prev:
            Ft = Ft*wp.sqrt(d/d_prev)
        rc = -R*n
        vc = v[b]+wp.cross(w[b], rc)
        vs = vc-wp.dot(vc, n)*n
        Ft = Ft-kt*dt*vs
        mu_f = wp.exp(theta[b, 7]+friction0)
        cap = mu_f*Fn
        L = wp.length(Ft)
        if L > cap:
            Ft = Ft*(cap/L)
        Ft_new = Ft
        d_store = d
        Fc = Fn*n+Ft
        Tc = wp.cross(rc, Ft)
    vn = v[b]
    wn = w[b]
    if fixed[b] == 0:
        vn = v[b]+dt*((f[b]+Fc)/mass[b]+gravity)
        wn = w[b]+dt*(tau[b]+Tc)/inertia[b]
    xn = x[b]
    if fixed[b] == 0:
        xn = x[b]+dt*vn
    hh = 0.5*dt*wn
    qs = wp.normalize(wp.vec4(1.0, hh[0], hh[1], hh[2]))
    x1[b] = xn
    v1[b] = vn
    w1[b] = wn
    q1[b] = wp.normalize(qmul(qs, q[b]))
    hc1[b] = wp.vec4(Ft_new[0], Ft_new[1], Ft_new[2], d_store)


@wp.kernel
def deformation_kernel(S: wp.array(dtype=wp.mat33), Minv: wp.array(dtype=wp.mat33), Fb: wp.array(dtype=wp.mat33)):
    b = wp.tid()
    Fb[b] = S[b]@Minv[b]


@wp.kernel
def residual_kernel(x: wp.array(dtype=wp.vec3), Fb: wp.array(dtype=wp.mat33), theta: wp.array2d(dtype=float),
                    bi: wp.array(dtype=int), bj: wp.array(dtype=int), lw: wp.array(dtype=float), dX: wp.array(dtype=wp.vec3),
                    vol: wp.array(dtype=float), den: wp.array(dtype=float), k_mu: float, bres: wp.array(dtype=wp.vec3)):
    """Non-affine residual of each body's least-squares fit (EnergyFoam residual=True):
    E = sum_a (mu_a V_a / den_a) sum_b w |(x_b - x_a) - F_a dX_ab|^2; dE/dF_a = 0 (F_a is the
    weighted least-squares fit), so only the explicit position terms act. Output: force on i."""
    e = wp.tid()
    i, j = bi[e], bj[e]
    y = x[j]-x[i]
    rij = y-Fb[i]@dX[e]
    rji = -y+Fb[j]@dX[e]
    ci = k_mu*wp.exp(theta[i, 5])*vol[i]/den[i]*lw[e]
    cj = k_mu*wp.exp(theta[j, 5])*vol[j]/den[j]*lw[e]
    bres[e] = 2.0*ci*rij-2.0*cj*rji


@wp.kernel
def gather_add2_kernel(off: wp.array(dtype=int), inc: wp.array(dtype=int), side: wp.array(dtype=float),
                       bg: wp.array(dtype=wp.vec3), bres: wp.array(dtype=wp.vec3), f_el: wp.array(dtype=wp.vec3),
                       f: wp.array(dtype=wp.vec3)):
    b = wp.tid()
    fs = f_el[b]
    for k in range(off[b], off[b+1]):
        fs = fs+side[k]*(bg[inc[k]]+bres[inc[k]])
    f[b] = fs


@wp.func
def nh_stress(F: wp.mat33, mu: float, kv: float, m: float):
    c0 = wp.vec3(F[0, 0], F[1, 0], F[2, 0])
    c1 = wp.vec3(F[0, 1], F[1, 1], F[2, 1])
    c2 = wp.vec3(F[0, 2], F[1, 2], F[2, 2])
    cof = cols(wp.cross(c1, c2), wp.cross(c2, c0), wp.cross(c0, c1))
    J = wp.dot(c0, wp.cross(c1, c2))
    L = float(0.0)
    dL = float(0.0)
    if J < m:
        L = wp.log(m)+(J-m)/m-(J-m)*(J-m)/(2.0*m*m)
        dL = 1.0/m-(J-m)/(m*m)
    else:
        L = wp.log(J)
        dL = 1.0/J
    return mu*F+((kv*L-mu)*dL)*cof


@wp.func
def eas_terms(F: wp.mat33, e: wp.vec3, mu: float, kv: float, m: float, sv: float):
    """g = d psi / d s and h = d2 psi / d s2 of psi(s) = mu/2 (|F|^2 - 3 + (e^{2s} - 1) a) - mu L(J e^s) + kv/2 L(J e^s)^2,
    a = |F e|^2 (the thickness log-stretch s of the enhanced strain), plus dg/dJ, dg/dmu, dg/dkv for the implicit adjoint."""
    Fe = F@e
    a = wp.dot(Fe, Fe)
    c0 = wp.vec3(F[0, 0], F[1, 0], F[2, 0])
    c1 = wp.vec3(F[0, 1], F[1, 1], F[2, 1])
    c2 = wp.vec3(F[0, 2], F[1, 2], F[2, 2])
    J = wp.dot(c0, wp.cross(c1, c2))
    es = wp.exp(sv)
    Jt = J*es
    L = float(0.0)
    dL = float(0.0)
    d2L = float(0.0)
    if Jt < m:
        L = wp.log(m)+(Jt-m)/m-(Jt-m)*(Jt-m)/(2.0*m*m)
        dL = 1.0/m-(Jt-m)/(m*m)
        d2L = -1.0/(m*m)
    else:
        L = wp.log(Jt)
        dL = 1.0/Jt
        d2L = -1.0/(Jt*Jt)
    g = mu*a*es*es+(kv*L-mu)*dL*Jt
    h = 2.0*mu*a*es*es+kv*(dL*Jt)*(dL*Jt)+(kv*L-mu)*(d2L*Jt*Jt+dL*Jt)
    # d g / d J (through Jt = J e^s): same structure as h with Jt -> es
    dgdJ = (kv*dL*dL*Jt+(kv*L-mu)*(d2L*Jt+dL))*es
    dgdmu = a*es*es-dL*Jt
    dgdkv = L*dL*Jt
    return wp.vec4(g, h, dgdJ, dgdmu), dgdkv, a, J


@wp.func
def eas_solve(F: wp.mat33, e: wp.vec3, mu: float, kv: float, m: float):
    """s = log(1 + eps) with d psi / d s = 0 (strictly convex in s above the barrier): 20 Newton steps, |step| <= 1,
    s kept in [-3, 3]; bodies with J <= 0 (inverted fit) are not enhanced. Custom adjoint below (implicit derivative)."""
    c0 = wp.vec3(F[0, 0], F[1, 0], F[2, 0])
    c1 = wp.vec3(F[0, 1], F[1, 1], F[2, 1])
    c2 = wp.vec3(F[0, 2], F[1, 2], F[2, 2])
    J = wp.dot(c0, wp.cross(c1, c2))
    sv = float(0.0)
    if J > 0.0:
        for it in range(20):
            t, dgdkv, a, J_ = eas_terms(F, e, mu, kv, m, sv)
            step = t[0]/wp.max(t[1], 1.0e-30)
            step = wp.clamp(step, -1.0, 1.0)
            sv = wp.clamp(sv-step, -3.0, 3.0)
    return wp.exp(sv)-1.0


@wp.func_grad(eas_solve)
def adj_eas_solve(F: wp.mat33, e: wp.vec3, mu: float, kv: float, m: float, adj_ret: float):
    """Implicit derivative of the converged s: ds = -(dg/dF : dF + dg/dmu dmu + dg/dkv dkv) / h, eps = e^s - 1."""
    eps = eas_solve(F, e, mu, kv, m)
    sv = wp.log(1.0+eps)
    t, dgdkv, a, J = eas_terms(F, e, mu, kv, m, sv)
    h = wp.max(t[1], 1.0e-30)
    lam_ = adj_ret*(1.0+eps)/h
    if J > 0.0:
        es = wp.exp(sv)
        Fe = F@e
        c0 = wp.vec3(F[0, 0], F[1, 0], F[2, 0])
        c1 = wp.vec3(F[0, 1], F[1, 1], F[2, 1])
        c2 = wp.vec3(F[0, 2], F[1, 2], F[2, 2])
        cof = cols(wp.cross(c1, c2), wp.cross(c2, c0), wp.cross(c0, c1))
        dgdF = 2.0*mu*es*es*wp.outer(Fe, e)+t[2]*cof
        wp.adjoint[F] += -lam_*dgdF
        wp.adjoint[mu] += -lam_*t[3]
        wp.adjoint[kv] += -lam_*dgdkv


@wp.func
def thin_pullback(P: wp.mat33, Ft: wp.mat33, e: wp.vec3, t1: wp.vec3, t2: wp.vec3):
    """dE/dF_t for F = F_t + n(F_t) e^T, n = normalize(F_t t1 x F_t t2), given dE/dF = P."""
    a = Ft@t1
    b = Ft@t2
    c = wp.cross(a, b)
    cn = wp.max(wp.length(c), 1.0e-30)
    nrm = c/cn
    Pe = P@e
    g = (Pe-wp.dot(Pe, nrm)*nrm)/cn
    return P+wp.outer(wp.cross(b, g), t1)+wp.outer(wp.cross(g, a), t2)


@wp.kernel
def shell_body_kernel(S: wp.array(dtype=wp.mat33), Minv: wp.array(dtype=wp.mat33), theta: wp.array2d(dtype=float),
                      vol: wp.array(dtype=float), k_v: float, k_mu: float, barrier: float,
                      thin: wp.array(dtype=int), te: wp.array(dtype=wp.vec3), t1: wp.array(dtype=wp.vec3), t2: wp.array(dtype=wp.vec3),
                      eas_on: wp.array(dtype=int), eas_dir: wp.array(dtype=wp.vec3),
                      x: wp.array(dtype=wp.vec3), ground_p: wp.vec3, ground_n: wp.vec3, contact: wp.array(dtype=float),
                      layer_dx: float, use_ground: int, Fb: wp.array(dtype=wp.mat33), G: wp.array(dtype=wp.mat33)):
    """Per body: F_t = S Minv; thin bodies: F = F_t + n e^T (shell closure); enhanced strain eps along eas_dir
    (free-surface normal, off while the body is in the contact layer): F~ = F (I + eps e e^T), d psi/d eps = 0 by a
    scalar Newton iteration; stress dE/dF = P(F~)(I + eps e e^T) pulled back through the closure; G = dE/dF_t Minv."""
    b = wp.tid()
    Ft = S[b]@Minv[b]
    F = Ft
    if thin[b] == 1:
        nrm = wp.cross(Ft@t1[b], Ft@t2[b])
        nrm = nrm/wp.max(wp.length(nrm), 1.0e-30)
        F = Ft+wp.outer(nrm, te[b])
    Fb[b] = F
    kv = k_v*wp.exp(theta[b, 4])*vol[b]
    mu = k_mu*wp.exp(theta[b, 5])*vol[b]
    on = eas_on[b]
    if use_ground == 1 and on == 1:
        if wp.dot(x[b]-ground_p, ground_n)-contact[b] < layer_dx:
            on = 0
    P = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if on == 1:
        e = eas_dir[b]
        eps = eas_solve(F, e, mu, kv, barrier)
        A = wp.identity(n=3, dtype=float)+eps*wp.outer(e, e)
        P = nh_stress(F@A, mu, kv, barrier)@A
    else:
        P = nh_stress(F, mu, kv, barrier)
    if thin[b] == 1:
        P = thin_pullback(P, Ft, te[b], t1[b], t2[b])
    G[b] = P@Minv[b]


@wp.kernel
def residual_normal_kernel(x: wp.array(dtype=wp.vec3), Fb: wp.array(dtype=wp.mat33), theta: wp.array2d(dtype=float),
                           bi: wp.array(dtype=int), bj: wp.array(dtype=int), lw: wp.array(dtype=float), dX: wp.array(dtype=wp.vec3),
                           vol: wp.array(dtype=float), den: wp.array(dtype=float), k_mu: float,
                           bres: wp.array(dtype=wp.vec3), bGi: wp.array(dtype=wp.mat33), bGj: wp.array(dtype=wp.mat33)):
    """Residual along the current bond direction: E = sum_a c_a sum_b w (r_ab . d)^2, r = y - F_a dX, d = y/|y|.
    Direct force on i (bres; on j the negative) and the per-end dE/dF_a contributions (bGi for a = i, bGj for a = j)."""
    e = wp.tid()
    i, j = bi[e], bj[e]
    y = x[j]-x[i]
    yn = wp.max(wp.length(y), 1.0e-30)
    d = y/yn
    ci = k_mu*wp.exp(theta[i, 5])*vol[i]/den[i]*lw[e]
    cj = k_mu*wp.exp(theta[j, 5])*vol[j]/den[j]*lw[e]
    # a = i, b = j
    r = y-Fb[i]@dX[e]
    s = wp.dot(r, d)
    v = d+(r-s*d)/yn
    gi = 2.0*ci*s*v                     # dE/dx_j (direct); dE/dx_i = -gi
    bGi[e] = -2.0*ci*s*wp.outer(d, dX[e])
    # a = j, b = i
    r2 = -y+Fb[j]@dX[e]
    s2 = wp.dot(r2, -d)
    v2 = -d+(r2-s2*(-d))/yn
    gj = 2.0*cj*s2*v2                   # dE/dx_i (direct); dE/dx_j = -gj
    bGj[e] = -2.0*cj*s2*wp.outer(-d, -dX[e])
    bres[e] = gi-gj                      # force on i = -(dE/dx_i) = -(-gi + gj)


@wp.kernel
def gather_mat_kernel(off: wp.array(dtype=int), inc: wp.array(dtype=int), side: wp.array(dtype=float),
                      bGi: wp.array(dtype=wp.mat33), bGj: wp.array(dtype=wp.mat33), Gres: wp.array(dtype=wp.mat33)):
    b = wp.tid()
    Gs = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    for k in range(off[b], off[b+1]):
        e = inc[k]
        if side[k] > 0.0:
            Gs = Gs+bGi[e]
        else:
            Gs = Gs+bGj[e]
    Gres[b] = Gs


@wp.kernel
def pullback_add_kernel(Gres: wp.array(dtype=wp.mat33), S: wp.array(dtype=wp.mat33), Minv: wp.array(dtype=wp.mat33),
                        thin: wp.array(dtype=int), te: wp.array(dtype=wp.vec3), t1: wp.array(dtype=wp.vec3), t2: wp.array(dtype=wp.vec3),
                        G: wp.array(dtype=wp.mat33), G2: wp.array(dtype=wp.mat33)):
    """G2 = G + pullback(Gres) Minv (the residual's dE/dF_a through the thin closure and the least-squares fit)."""
    b = wp.tid()
    P = Gres[b]
    if thin[b] == 1:
        P = thin_pullback(P, S[b]@Minv[b], te[b], t1[b], t2[b])
    G2[b] = G[b]+P@Minv[b]


@wp.kernel
def cosserat_bond_kernel(q: wp.array(dtype=wp.vec4), bi: wp.array(dtype=int), bj: wp.array(dtype=int), lw: wp.array(dtype=float),
                         dX: wp.array(dtype=wp.vec3), bth: wp.array(dtype=wp.vec3), bN: wp.array(dtype=wp.mat33)):
    """theta_ij = log(R_i^T R_j) (principal branch) and the curvature least-squares term w theta dX^T (same for both ends)."""
    e = wp.tid()
    i, j = bi[e], bj[e]
    qr = qmul(qconj(q[i]), q[j])
    if qr[0] < 0.0:
        qr = -qr
    th = qlog(qr)
    bth[e] = th
    bN[e] = lw[e]*wp.outer(th, dX[e])


@wp.kernel
def gather_mat_both_kernel(off: wp.array(dtype=int), inc: wp.array(dtype=int), bN: wp.array(dtype=wp.mat33), N: wp.array(dtype=wp.mat33)):
    b = wp.tid()
    Ns = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    for k in range(off[b], off[b+1]):
        Ns = Ns+bN[inc[k]]
    N[b] = Ns


@wp.kernel
def cosserat_body_kernel(S: wp.array(dtype=wp.mat33), N: wp.array(dtype=wp.mat33), Minv: wp.array(dtype=wp.mat33), q: wp.array(dtype=wp.vec4),
                         theta: wp.array2d(dtype=float), vol: wp.array(dtype=float), gyr: wp.array(dtype=float), k_v: float, k_mu: float, barrier: float,
                         thin: wp.array(dtype=int), te: wp.array(dtype=wp.vec3), t1: wp.array(dtype=wp.vec3), t2: wp.array(dtype=wp.vec3),
                         eas_on: wp.array(dtype=int), eas_dir: wp.array(dtype=wp.vec3),
                         x: wp.array(dtype=wp.vec3), ground_p: wp.vec3, ground_n: wp.vec3, contact: wp.array(dtype=float), layer_dx: float, use_ground: int,
                         bond_level: int,
                         Fb: wp.array(dtype=wp.mat33), G: wp.array(dtype=wp.mat33), QM: wp.array(dtype=wp.mat33), tau_b: wp.array(dtype=wp.vec3)):
    """shell_body_kernel plus the micropolar terms (COSSERAT_FOAM.md): stress of mu_c |skew(R^T F)|^2 (R (2 mu S), S = skew U),
    its torque 2 mu vee(B - B^T) with B = F S^T R^T, and the curvature stress Q = V [a1 tr K I + E I_z K] with
    K = N Minv, a1 = 2 mu (1 - nu) I_z; QM = Q Minv feeds the curvature torques of the bonds."""
    b = wp.tid()
    Ft = S[b]@Minv[b]
    F = Ft
    if thin[b] == 1:
        nrm = wp.cross(Ft@t1[b], Ft@t2[b])
        nrm = nrm/wp.max(wp.length(nrm), 1.0e-30)
        F = Ft+wp.outer(nrm, te[b])
    Fb[b] = F
    kv = k_v*wp.exp(theta[b, 4])*vol[b]
    mu = k_mu*wp.exp(theta[b, 5])*vol[b]
    on = eas_on[b]
    if use_ground == 1 and on == 1:
        if wp.dot(x[b]-ground_p, ground_n)-contact[b] < layer_dx:
            on = 0
    P = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if on == 1:
        e = eas_dir[b]
        eps = eas_solve(F, e, mu, kv, barrier)
        A = wp.identity(n=3, dtype=float)+eps*wp.outer(e, e)
        P = nh_stress(F@A, mu, kv, barrier)@A
    else:
        P = nh_stress(F, mu, kv, barrier)
    # micropolar coupling: U = R^T F, S = skew U
    R = qmat(q[b])
    U = wp.transpose(R)@F
    Sk = 0.5*(U-wp.transpose(U))
    P = P+R@(2.0*mu*Sk)
    B = F@wp.transpose(Sk)@wp.transpose(R)
    Bs = B-wp.transpose(B)
    tb = -2.0*mu*wp.vec3(Bs[2, 1], Bs[0, 2], Bs[1, 0])
    # curvature
    lam = kv
    E = mu*(3.0*lam+2.0*mu)/(lam+mu)
    nu = lam/(2.0*(lam+mu))
    Iz = gyr[b]
    K = N[b]@Minv[b]
    trK = K[0, 0]+K[1, 1]+K[2, 2]
    Q = (2.0*mu*(1.0-nu)*Iz*trK)*wp.identity(n=3, dtype=float)+(E*Iz)*K
    QM[b] = Q@Minv[b]
    if bond_level == 1:          # the Neo-Hookean and skew terms are evaluated per pair (bond_level_kernel)
        P = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        tb = wp.vec3(0.0, 0.0, 0.0)
    tau_b[b] = tb                # one write per element (a second write would still receive the first one's adjoint)
    if thin[b] == 1:
        P = thin_pullback(P, Ft, te[b], t1[b], t2[b])
    G[b] = P@Minv[b]


@wp.func
def skew_stress(F: wp.mat33, R: wp.mat33, mu: float):
    """mu |skew(R^T F)|^2: dW/dF = R (2 mu S), S = skew(R^T F) (single-output funcs: safe Warp adjoints)."""
    U = wp.transpose(R)@F
    Sk = 0.5*(U-wp.transpose(U))
    return R@(2.0*mu*Sk)


@wp.func
def skew_torque(F: wp.mat33, R: wp.mat33, mu: float):
    """Torque of mu |skew(R^T F)|^2: -2 mu vee(B - B^T), B = F S^T R^T."""
    U = wp.transpose(R)@F
    Sk = 0.5*(U-wp.transpose(U))
    B = F@wp.transpose(Sk)@wp.transpose(R)
    Bs = B-wp.transpose(B)
    return -2.0*mu*wp.vec3(Bs[2, 1], Bs[0, 2], Bs[1, 0])


@wp.kernel
def bond_level_kernel(x: wp.array(dtype=wp.vec3), q: wp.array(dtype=wp.vec4), Fb: wp.array(dtype=wp.mat33), theta: wp.array2d(dtype=float),
                      vol: wp.array(dtype=float), k_v: float, k_mu: float, barrier: float,
                      bi: wp.array(dtype=int), bj: wp.array(dtype=int), lw: wp.array(dtype=float), dX: wp.array(dtype=wp.vec3),
                      trM: wp.array(dtype=float), mid: int, micro: int,
                      bres: wp.array(dtype=wp.vec3), bGi: wp.array(dtype=wp.mat33), bGj: wp.array(dtype=wp.mat33),
                      bti: wp.array(dtype=wp.vec3), btj: wp.array(dtype=wp.vec3)):
    """Bond-level deformation gradient (Breitzman & Dayal 2018; EnergyFoam(bond_level=True)), both directions of bond
    e = (i, j), n = dX/L. mid = 0: F_ij = F_i (I - n n^T) + y/L n^T, F_ji = F_j (I - n n^T) + y/L n^T. mid = 1: with
    r = y - (F_i + F_j) dX / 2, F_ij = F_i + r/L n^T, F_ji = F_j + r/L n^T. Energy om_a W(F_a.) with om = lw L^2 / tr M_a,
    W = psi_NH + mu |skew(R_a^T F)|^2. Outputs: force on i (bres; on j the negative), dE/dF_i (bGi), dE/dF_j (bGj),
    torques on i, j from the skew term."""
    e = wp.tid()
    i, j = bi[e], bj[e]
    L = wp.length(dX[e])
    nrm = dX[e]/L
    y = x[j]-x[i]
    nn = wp.outer(nrm, nrm)
    Fi = Fb[i]
    Fj = Fb[j]
    F1 = Fi
    F2 = Fj
    if mid == 1:
        r = y-0.5*((Fi+Fj)@dX[e])
        F1 = Fi+wp.outer(r/L, nrm)
        F2 = Fj+wp.outer(r/L, nrm)
    else:
        F1 = Fi-Fi@nn+wp.outer(y/L, nrm)
        F2 = Fj-Fj@nn+wp.outer(y/L, nrm)
    w2 = lw[e]*L*L
    om_i = w2/trM[i]
    om_j = w2/trM[j]
    mu_i = k_mu*wp.exp(theta[i, 5])*vol[i]
    kv_i = k_v*wp.exp(theta[i, 4])*vol[i]
    mu_j = k_mu*wp.exp(theta[j, 5])*vol[j]
    kv_j = k_v*wp.exp(theta[j, 4])*vol[j]
    Ri = qmat(q[i])
    Rj = qmat(q[j])
    zero3 = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    Ps1 = zero3
    Ps2 = zero3
    t1 = wp.vec3(0.0, 0.0, 0.0)
    t2 = wp.vec3(0.0, 0.0, 0.0)
    if micro == 1:               # micropolar coupling mu |skew(R^T F)|^2; off for the classical (clean) model
        Ps1 = skew_stress(F1, Ri, mu_i)
        Ps2 = skew_stress(F2, Rj, mu_j)
        t1 = skew_torque(F1, Ri, mu_i)
        t2 = skew_torque(F2, Rj, mu_j)
    P1 = om_i*(nh_stress(F1, mu_i, kv_i, barrier)+Ps1)
    P2 = om_j*(nh_stress(F2, mu_j, kv_j, barrier)+Ps2)
    if mid == 1:
        Qh = 0.5*(P1+P2)@nn
        bGi[e] = P1-Qh
        bGj[e] = P2-Qh
    else:
        bGi[e] = P1-P1@nn
        bGj[e] = P2-P2@nn
    bres[e] = (P1+P2)@nrm/L          # dE/dx_j = (P1 + P2) n / L = -dE/dx_i, so this is the force on i
    bti[e] = om_i*t1
    btj[e] = om_j*t2


@wp.kernel
def curvature_torque_kernel(q: wp.array(dtype=wp.vec4), bth: wp.array(dtype=wp.vec3), QM: wp.array(dtype=wp.mat33),
                            bi: wp.array(dtype=int), bj: wp.array(dtype=int), lw: wp.array(dtype=float), dX: wp.array(dtype=wp.vec3),
                            cti: wp.array(dtype=wp.vec3), ctj: wp.array(dtype=wp.vec3)):
    """g = w (QM_i + QM_j) dX; c = J_r^-T(theta) g; torque on i = +R_j c, on j = -R_j c."""
    e = wp.tid()
    i, j = bi[e], bj[e]
    g = lw[e]*((QM[i]+QM[j])@dX[e])
    c = wp.transpose(jr_inv(bth[e]))@g
    Rjc = qmat(q[j])@c
    cti[e] = Rjc
    ctj[e] = -Rjc


@wp.kernel
def gather_tau_kernel(off: wp.array(dtype=int), inc: wp.array(dtype=int), side: wp.array(dtype=float), cti: wp.array(dtype=wp.vec3),
                      ctj: wp.array(dtype=wp.vec3), tau_b: wp.array(dtype=wp.vec3), tau_in: wp.array(dtype=wp.vec3), tau_out: wp.array(dtype=wp.vec3)):
    b = wp.tid()
    ts = tau_in[b]+tau_b[b]
    for k in range(off[b], off[b+1]):
        e = inc[k]
        if side[k] > 0.0:
            ts = ts+cti[e]
        else:
            ts = ts+ctj[e]
    tau_out[b] = ts


@wp.kernel
def damping_pre_kernel(x: wp.array(dtype=wp.vec3), q: wp.array(dtype=wp.vec4), bi: wp.array(dtype=int), bj: wp.array(dtype=int),
                       ri: wp.array(dtype=wp.vec3), rj: wp.array(dtype=wp.vec3), cn: wp.array(dtype=float),
                       ct: wp.array(dtype=float), cr: wp.array(dtype=float), theta: wp.array2d(dtype=float),
                       mass: wp.array(dtype=float), inertia: wp.array(dtype=float), fixed: wp.array(dtype=int), dt: float,
                       pre_pi: wp.array(dtype=wp.vec3), pre_pj: wp.array(dtype=wp.vec3), pre_Ainv: wp.array(dtype=wp.mat33),
                       pre_Dm: wp.array(dtype=wp.mat33), pre_m: wp.array(dtype=wp.vec4), pre_c: wp.array(dtype=float)):
    """The velocity-independent part of damping_pair_kernel, once per bond and step (x, q, theta do not change
    across the colours of one step): levers, dashpot matrix Dm, (I + dt K Dm)^-1, inverse masses, spin coefficient.
    Same expressions as damping_pair_kernel."""
    e = wp.tid()
    i, j = bi[e], bj[e]
    Ri, Rj = qmat(q[i]), qmat(q[j])
    pi, pj = Ri@ri[e], Rj@rj[e]
    mi = float(0.0)
    mj = float(0.0)
    Ii = float(0.0)
    Ij = float(0.0)
    if fixed[i] == 0:
        mi = 1.0/mass[i]
        Ii = 1.0/inertia[i]
    if fixed[j] == 0:
        mj = 1.0/mass[j]
        Ij = 1.0/inertia[j]
    axis = wp.normalize(x[j]-x[i])
    scale = wp.exp(0.5*(theta[i, 6]+theta[j, 6]))
    nn = wp.outer(axis, axis)
    Id = wp.identity(3, dtype=float)
    Dm = scale*(cn[e]*nn+ct[e]*(Id-nn))
    Si, Sj = skewm(pi), skewm(pj)
    K = (mi+mj)*Id-Ii*(Si@Si)-Ij*(Sj@Sj)
    pre_pi[e] = pi
    pre_pj[e] = pj
    pre_Ainv[e] = wp.inverse(Id+dt*(K@Dm))
    pre_Dm[e] = Dm
    pre_m[e] = wp.vec4(mi, mj, Ii, Ij)
    pre_c[e] = scale*cr[e]


@wp.kernel
def damping_apply_kernel(v_in: wp.array(dtype=wp.vec3), w_in: wp.array(dtype=wp.vec3), bond_of: wp.array2d(dtype=int), color: int,
                         bi: wp.array(dtype=int), bj: wp.array(dtype=int), pre_pi: wp.array(dtype=wp.vec3),
                         pre_pj: wp.array(dtype=wp.vec3), pre_Ainv: wp.array(dtype=wp.mat33), pre_Dm: wp.array(dtype=wp.mat33),
                         pre_m: wp.array(dtype=wp.vec4), pre_c: wp.array(dtype=float), dt: float,
                         v_out: wp.array(dtype=wp.vec3), w_out: wp.array(dtype=wp.vec3)):
    """The velocity-dependent part of damping_pair_kernel for one colour (same expressions, precomputed factors)."""
    b = wp.tid()
    e = bond_of[color, b]
    if e < 0:
        v_out[b] = v_in[b]
        w_out[b] = w_in[b]
        return
    i, j = bi[e], bj[e]
    pi, pj = pre_pi[e], pre_pj[e]
    vi, wi, vj, wj = v_in[i], w_in[i], v_in[j], w_in[j]
    mm = pre_m[e]
    mi, mj, Ii, Ij = mm[0], mm[1], mm[2], mm[3]
    Dm = pre_Dm[e]
    u = vj+wp.cross(wj, pj)-vi-wp.cross(wi, pi)
    un = pre_Ainv[e]@u
    P = -dt*(Dm@un)                                   # impulse on j at the contact, -P on i
    dvi = -mi*P
    dwi = -Ii*wp.cross(pi, P)
    dvj = mj*P
    dwj = Ij*wp.cross(pj, P)
    c = pre_c[e]
    sp = (wj+dwj)-(wi+dwi)
    L = -dt*c*sp/(1.0+dt*c*(Ii+Ij))                   # spin impulse on j, -L on i
    dwi = dwi-Ii*L
    dwj = dwj+Ij*L
    if b == i:
        v_out[b] = vi+dvi
        w_out[b] = wi+dwi
    else:
        v_out[b] = vj+dvj
        w_out[b] = wj+dwj


@wp.kernel
def damp_copy_kernel(v: wp.array(dtype=wp.vec3), w: wp.array(dtype=wp.vec3), vd: wp.array(dtype=wp.vec3), wd: wp.array(dtype=wp.vec3)):
    b = wp.tid()
    vd[b] = v[b]
    wd[b] = w[b]


@wp.kernel
def damp_copy_adj_kernel(avd: wp.array(dtype=wp.vec3), awd: wp.array(dtype=wp.vec3), av: wp.array(dtype=wp.vec3), aw: wp.array(dtype=wp.vec3)):
    b = wp.tid()
    av[b] = av[b]+avd[b]
    aw[b] = aw[b]+awd[b]


@wp.kernel
def damp_inplace_kernel(bonds: wp.array(dtype=int), off: int, vd: wp.array(dtype=wp.vec3), wd: wp.array(dtype=wp.vec3),
                        bi: wp.array(dtype=int), bj: wp.array(dtype=int), pre_pi: wp.array(dtype=wp.vec3),
                        pre_pj: wp.array(dtype=wp.vec3), pre_Ainv: wp.array(dtype=wp.mat33), pre_Dm: wp.array(dtype=wp.mat33),
                        pre_m: wp.array(dtype=wp.vec4), pre_c: wp.array(dtype=float), dt: float,
                        sv_i: wp.array(dtype=wp.vec3), sw_i: wp.array(dtype=wp.vec3), sv_j: wp.array(dtype=wp.vec3),
                        sw_j: wp.array(dtype=wp.vec3)):
    """One colour of the implicit dashpots, in place (bonds of a colour share no body), one thread per bond; same
    arithmetic as damping_apply_kernel. The pre-update velocities are saved per bond for damp_inplace_adj_kernel."""
    t = wp.tid()
    e = bonds[off+t]
    i, j = bi[e], bj[e]
    vi, wi, vj, wj = vd[i], wd[i], vd[j], wd[j]
    sv_i[e] = vi
    sw_i[e] = wi
    sv_j[e] = vj
    sw_j[e] = wj
    pi, pj = pre_pi[e], pre_pj[e]
    mm = pre_m[e]
    mi, mj, Ii, Ij = mm[0], mm[1], mm[2], mm[3]
    Dm = pre_Dm[e]
    u = vj+wp.cross(wj, pj)-vi-wp.cross(wi, pi)
    un = pre_Ainv[e]@u
    P = -dt*(Dm@un)
    dvi = -mi*P
    dwi = -Ii*wp.cross(pi, P)
    dvj = mj*P
    dwj = Ij*wp.cross(pj, P)
    c = pre_c[e]
    sp = (wj+dwj)-(wi+dwi)
    L = -dt*c*sp/(1.0+dt*c*(Ii+Ij))
    dwi = dwi-Ii*L
    dwj = dwj+Ij*L
    vd[i] = vi+dvi
    wd[i] = wi+dwi
    vd[j] = vj+dvj
    wd[j] = wj+dwj


@wp.kernel
def damp_inplace_adj_kernel(bonds: wp.array(dtype=int), off: int, avd: wp.array(dtype=wp.vec3), awd: wp.array(dtype=wp.vec3),
                            bi: wp.array(dtype=int), bj: wp.array(dtype=int), sv_i: wp.array(dtype=wp.vec3),
                            sw_i: wp.array(dtype=wp.vec3), sv_j: wp.array(dtype=wp.vec3), sw_j: wp.array(dtype=wp.vec3),
                            pre_pi: wp.array(dtype=wp.vec3), pre_pj: wp.array(dtype=wp.vec3), pre_Ainv: wp.array(dtype=wp.mat33),
                            pre_Dm: wp.array(dtype=wp.mat33), pre_m: wp.array(dtype=wp.vec4), pre_c: wp.array(dtype=float),
                            dt: float, a_pre_pi: wp.array(dtype=wp.vec3), a_pre_pj: wp.array(dtype=wp.vec3),
                            a_pre_Ainv: wp.array(dtype=wp.mat33), a_pre_Dm: wp.array(dtype=wp.mat33),
                            a_pre_m: wp.array(dtype=wp.vec4), a_pre_c: wp.array(dtype=float)):
    """Hand-written adjoint of damp_inplace_kernel for one colour (run last colour first): the adjoints of the
    post-update velocities of the bond's two bodies (avd, awd at i, j) are replaced in place by those of the
    pre-update velocities; the adjoints of the precomputed factors are added (each bond belongs to one colour)."""
    t = wp.tid()
    e = bonds[off+t]
    i, j = bi[e], bj[e]
    vi, wi, vj, wj = sv_i[e], sw_i[e], sv_j[e], sw_j[e]
    pi, pj = pre_pi[e], pre_pj[e]
    mm = pre_m[e]
    mi, mj, Ii, Ij = mm[0], mm[1], mm[2], mm[3]
    Dm = pre_Dm[e]
    A = pre_Ainv[e]
    c = pre_c[e]
    # forward recomputation
    u = vj+wp.cross(wj, pj)-vi-wp.cross(wi, pi)
    un = A@u
    P = -dt*(Dm@un)
    dwi0 = -Ii*wp.cross(pi, P)
    dwj0 = Ij*wp.cross(pj, P)
    sp = (wj+dwj0)-(wi+dwi0)
    g = 1.0+dt*c*(Ii+Ij)
    L = -dt*c*sp/g
    # adjoints of the outputs vi' = vi + dvi, wi' = wi + dwi, vj' = vj + dvj, wj' = wj + dwj
    a_vio, a_wio, a_vjo, a_wjo = avd[i], awd[i], avd[j], awd[j]
    a_vi = a_vio
    a_wi = a_wio
    a_vj = a_vjo
    a_wj = a_wjo
    a_mi = float(0.0)
    a_mj = float(0.0)
    a_Ii = float(0.0)
    a_Ij = float(0.0)
    # dwi = dwi0 - Ii L, dwj = dwj0 + Ij L
    a_dwi0 = a_wio
    a_dwj0 = a_wjo
    a_L = -Ii*a_wio+Ij*a_wjo
    a_Ii = a_Ii-wp.dot(a_wio, L)
    a_Ij = a_Ij+wp.dot(a_wjo, L)
    # L = -dt c sp / g, g = 1 + dt c (Ii + Ij)
    a_sp = (-dt*c/g)*a_L
    q_ = wp.dot(a_L, sp)
    a_c = q_*(-dt/(g*g))
    a_Ii = a_Ii+q_*(dt*c)*(dt*c)/(g*g)
    a_Ij = a_Ij+q_*(dt*c)*(dt*c)/(g*g)
    # sp = wj + dwj0 - wi - dwi0
    a_wj = a_wj+a_sp
    a_dwj0 = a_dwj0+a_sp
    a_wi = a_wi-a_sp
    a_dwi0 = a_dwi0-a_sp
    # dwj0 = Ij (pj x P), dvj = mj P, dwi0 = -Ii (pi x P), dvi = -mi P
    a_P = Ij*wp.cross(a_dwj0, pj)
    a_pj = Ij*wp.cross(P, a_dwj0)
    a_Ij = a_Ij+wp.dot(a_dwj0, wp.cross(pj, P))
    a_P = a_P+mj*a_vjo
    a_mj = a_mj+wp.dot(a_vjo, P)
    a_P = a_P-Ii*wp.cross(a_dwi0, pi)
    a_pi = -Ii*wp.cross(P, a_dwi0)
    a_Ii = a_Ii-wp.dot(a_dwi0, wp.cross(pi, P))
    a_P = a_P-mi*a_vio
    a_mi = a_mi-wp.dot(a_vio, P)
    # P = -dt Dm un, un = A u
    a_D = -dt*wp.outer(a_P, un)
    a_un = -dt*(wp.transpose(Dm)@a_P)
    a_A = wp.outer(a_un, u)
    a_u = wp.transpose(A)@a_un
    # u = vj + wj x pj - vi - wi x pi
    a_vj = a_vj+a_u
    a_vi = a_vi-a_u
    a_wj = a_wj+wp.cross(pj, a_u)
    a_pj = a_pj+wp.cross(a_u, wj)
    a_wi = a_wi-wp.cross(pi, a_u)
    a_pi = a_pi-wp.cross(a_u, wi)
    avd[i] = a_vi
    awd[i] = a_wi
    avd[j] = a_vj
    awd[j] = a_wj
    a_pre_pi[e] = a_pre_pi[e]+a_pi
    a_pre_pj[e] = a_pre_pj[e]+a_pj
    a_pre_Ainv[e] = a_pre_Ainv[e]+a_A
    a_pre_Dm[e] = a_pre_Dm[e]+a_D
    a_pre_m[e] = a_pre_m[e]+wp.vec4(a_mi, a_mj, a_Ii, a_Ij)
    a_pre_c[e] = a_pre_c[e]+a_c


@wp.kernel
def damping_pair_kernel(x: wp.array(dtype=wp.vec3), q: wp.array(dtype=wp.vec4), v_in: wp.array(dtype=wp.vec3),
                        w_in: wp.array(dtype=wp.vec3), bond_of: wp.array2d(dtype=int), color: int,
                        bi: wp.array(dtype=int), bj: wp.array(dtype=int), ri: wp.array(dtype=wp.vec3), rj: wp.array(dtype=wp.vec3),
                        cn: wp.array(dtype=float), ct: wp.array(dtype=float), cr: wp.array(dtype=float),
                        theta: wp.array2d(dtype=float), mass: wp.array(dtype=float), inertia: wp.array(dtype=float),
                        fixed: wp.array(dtype=int), dt: float, v_out: wp.array(dtype=wp.vec3), w_out: wp.array(dtype=wp.vec3)):
    """Implicit (backward Euler) dashpot of one bond per body in this colour (bonds of a colour share
    no body): the pair's contact-point relative velocity u (normal / tangential dashpots) and relative
    spin are solved exactly for the two-body system, (I + dt K D) u' = u with K = J M^-1 J^T, so any
    damping is unconditionally stable; linear and angular momentum of the pair are conserved. Both
    bodies of the pair evaluate the same solve (out of place, deterministic, adjoint friendly)."""
    b = wp.tid()
    e = bond_of[color, b]
    if e < 0:
        v_out[b] = v_in[b]
        w_out[b] = w_in[b]
        return
    i, j = bi[e], bj[e]
    Ri, Rj = qmat(q[i]), qmat(q[j])
    pi, pj = Ri@ri[e], Rj@rj[e]
    vi, wi, vj, wj = v_in[i], w_in[i], v_in[j], w_in[j]
    mi = float(0.0)
    mj = float(0.0)
    Ii = float(0.0)
    Ij = float(0.0)
    if fixed[i] == 0:
        mi = 1.0/mass[i]
        Ii = 1.0/inertia[i]
    if fixed[j] == 0:
        mj = 1.0/mass[j]
        Ij = 1.0/inertia[j]
    u = vj+wp.cross(wj, pj)-vi-wp.cross(wi, pi)
    axis = wp.normalize(x[j]-x[i])
    scale = wp.exp(0.5*(theta[i, 6]+theta[j, 6]))
    nn = wp.outer(axis, axis)
    Id = wp.identity(3, dtype=float)
    Dm = scale*(cn[e]*nn+ct[e]*(Id-nn))
    Si, Sj = skewm(pi), skewm(pj)
    K = (mi+mj)*Id-Ii*(Si@Si)-Ij*(Sj@Sj)
    un = wp.inverse(Id+dt*(K@Dm))@u
    P = -dt*(Dm@un)                                   # impulse on j at the contact, -P on i
    dvi = -mi*P
    dwi = -Ii*wp.cross(pi, P)
    dvj = mj*P
    dwj = Ij*wp.cross(pj, P)
    c = scale*cr[e]
    sp = (wj+dwj)-(wi+dwi)
    L = -dt*c*sp/(1.0+dt*c*(Ii+Ij))                   # spin impulse on j, -L on i
    dwi = dwi-Ii*L
    dwj = dwj+Ij*L
    if b == i:
        v_out[b] = vi+dvi
        w_out[b] = wi+dwi
    else:
        v_out[b] = vj+dvj
        w_out[b] = wj+dwj


@wp.kernel
def dil_strain_kernel(x: wp.array(dtype=wp.vec3), bi: wp.array(dtype=int), bj: wp.array(dtype=int), lw: wp.array(dtype=float),
                      dX: wp.array(dtype=wp.vec3), trM: wp.array(dtype=float), bti: wp.array(dtype=float), btj: wp.array(dtype=float)):
    """Dilatation body law (EnergyFoam body_law='dilatation'): per bond the contributions 3 om log(|y|/|dX|) to the volumetric
    strain of both ends, om_i = lw L^2 / tr M_i."""
    e = wp.tid()
    i, j = bi[e], bj[e]
    L0 = wp.length(dX[e])
    eps = wp.log(wp.length(x[j]-x[i])/L0)
    w2 = lw[e]*L0*L0
    bti[e] = 3.0*w2/trM[i]*eps
    btj[e] = 3.0*w2/trM[j]*eps


@wp.kernel
def gather_scalar_kernel(off: wp.array(dtype=int), inc: wp.array(dtype=int), side: wp.array(dtype=float),
                         bti: wp.array(dtype=float), btj: wp.array(dtype=float), out: wp.array(dtype=float)):
    b = wp.tid()
    s = float(0.0)
    for k in range(off[b], off[b+1]):
        e = inc[k]
        if side[k] > 0.0:
            s = s+bti[e]
        else:
            s = s+btj[e]
    out[b] = s


@wp.kernel
def dil_force_kernel(x: wp.array(dtype=wp.vec3), th_v: wp.array(dtype=float), theta: wp.array2d(dtype=float), vol: wp.array(dtype=float),
                     k_v: float, bi: wp.array(dtype=int), bj: wp.array(dtype=int), lw: wp.array(dtype=float), dX: wp.array(dtype=wp.vec3),
                     trM: wp.array(dtype=float), bg: wp.array(dtype=wp.vec3)):
    """dE/dy = (lam_i V_i theta_i 3 om_i + lam_j V_j theta_j 3 om_j) y / |y|^2 (y = x_j - x_i): force on i (on j the negative)."""
    e = wp.tid()
    i, j = bi[e], bj[e]
    y = x[j]-x[i]
    L0 = wp.length(dX[e])
    w2 = lw[e]*L0*L0
    ci = k_v*wp.exp(theta[i, 4])*vol[i]*th_v[i]*3.0*w2/trM[i]
    cj = k_v*wp.exp(theta[j, 4])*vol[j]*th_v[j]*3.0*w2/trM[j]
    bg[e] = ((ci+cj)/wp.dot(y, y))*y


@wp.kernel
def tet_vol_kernel(x: wp.array(dtype=wp.vec3), tets: wp.array2d(dtype=int), Vt: wp.array(dtype=float)):
    t = wp.tid()
    p0 = x[tets[t, 0]]
    Vt[t] = wp.dot(wp.cross(x[tets[t, 1]]-p0, x[tets[t, 2]]-p0), x[tets[t, 3]]-p0)/6.0


@wp.kernel
def node_volume_kernel(off: wp.array(dtype=int), inc: wp.array(dtype=int), Vt: wp.array(dtype=float), Vn: wp.array(dtype=float)):
    """Sum of the current volumes of the tetrahedra at each body (linear gather only: Warp's adjoint of a dynamic loop
    does not keep the loop-carried primal, so nonlinear functions of the sum go in a separate kernel)."""
    b = wp.tid()
    vs = float(0.0)
    for k in range(off[b], off[b+1]):
        vs = vs+Vt[inc[k]]
    Vn[b] = vs


@wp.kernel
def node_pressure_kernel(Vn: wp.array(dtype=float), V0: wp.array(dtype=float), has: wp.array(dtype=int), theta: wp.array2d(dtype=float),
                         k_v: float, m: float, pa: wp.array(dtype=float)):
    """True-volume body law (EnergyFoam body_law='nodalvol'): J_a = sum_t V_t / V0_a over the tetrahedra at a,
    pa = lam_a L(J) L'(J) / 4 (dE/dV_t of each tetrahedron at a), L = log J continued quadratically below m."""
    b = wp.tid()
    J = float(1.0)
    if has[b] == 1:
        J = Vn[b]/V0[b]
    L = float(0.0)
    dL = float(0.0)
    if J < m:
        L = wp.log(m)+(J-m)/m-(J-m)*(J-m)/(2.0*m*m)
        dL = 1.0/m-(J-m)/(m*m)
    else:
        L = wp.log(J)
        dL = 1.0/J
    pa[b] = float(has[b])*0.25*k_v*wp.exp(theta[b, 4])*L*dL


@wp.kernel
def tet_force_kernel(x: wp.array(dtype=wp.vec3), tets: wp.array2d(dtype=int), pa: wp.array(dtype=float), tg: wp.array(dtype=wp.vec3)):
    """dE/dx at the 4 corners of tetrahedron t: p_t dV_t/dx_k, p_t = sum of pa over the corners."""
    t = wp.tid()
    i0, i1, i2, i3 = tets[t, 0], tets[t, 1], tets[t, 2], tets[t, 3]
    pt = pa[i0]+pa[i1]+pa[i2]+pa[i3]
    d1 = x[i1]-x[i0]
    d2 = x[i2]-x[i0]
    d3 = x[i3]-x[i0]
    g1 = wp.cross(d2, d3)/6.0
    g2 = wp.cross(d3, d1)/6.0
    g3 = wp.cross(d1, d2)/6.0
    tg[4*t+0] = -pt*(g1+g2+g3)
    tg[4*t+1] = pt*g1
    tg[4*t+2] = pt*g2
    tg[4*t+3] = pt*g3


@wp.kernel
def node_force_kernel(off: wp.array(dtype=int), corner: wp.array(dtype=int), tg: wp.array(dtype=wp.vec3), f_in: wp.array(dtype=wp.vec3),
                      f_out: wp.array(dtype=wp.vec3)):
    b = wp.tid()
    fs = f_in[b]
    for k in range(off[b], off[b+1]):
        fs = fs-tg[corner[k]]
    f_out[b] = fs


class WarpFoam:
    """Device copies of an EnergyFoam (body_law='neohookean', ground_mode='noslip') for fused rollouts."""

    def __init__(self, model, deterministic=True, contact='hard', layer_dx=.002, dt_ref=1/4800, friction0=.5, restitution0=.3,
                 implicit_damping=False, graphs=False, contact_scale=1., visco=False, visco_g0=1., visco_tau0=1/120,
                 visco_cols=(8, 9), ogden_col=-1, contact_offset_col=-1):
        assert model.body_law in ('neohookean', 'dilatation', 'nodalvol') and model.ground_mode == 'noslip'
        f32 = lambda t: wp.from_torch(t.detach().float().contiguous())
        vec = lambda t: wp.from_torch(t.detach().float().contiguous(), dtype=wp.vec3)
        self.n, self.nb = model.n, len(model.i)
        self.bi = wp.from_torch(model.i.int().contiguous())
        self.bj = wp.from_torch(model.j.int().contiguous())
        self.n0, self.ri, self.rj = vec(model.n0), vec(model.ri), vec(model.rj)
        self.base = f32(model.base)
        self.cn, self.ct, self.cr = f32(model.c_n), f32(model.c_t), f32(model.c_r)
        # least-squares pairs: first half of ls_* are the (i, j) directions in bond order
        nb = self.nb
        assert torch.equal(model.ls_a[:nb], model.i) and torch.equal(model.ls_b[:nb], model.j)
        self.lw, self.dX = f32(model.ls_w[:nb]), vec(model.ls_dX[:nb])
        self.Minv = wp.from_torch(model.moment_inv.detach().float().contiguous(), dtype=wp.mat33)
        self.vol = f32(model.cell_volume)
        self.mass, self.inertia = f32(model.mass), f32(model.inertia)
        self.fixed = wp.from_torch(model.fixed.int().contiguous())
        self.contact = f32(model.contact_radius)
        self.k_v, self.k_mu, self.barrier = float(model.k_v), float(model.k_mu), float(model.barrier_J)
        self.mu_ref = float(getattr(model, 'mu_ref', model.k_mu))     # material shear modulus (Hertz-Mindlin contact)
        self.nl_bonds = int(bool(getattr(model, 'nonlinear_bonds', False)))   # Ogden / neo-Hookean normal bond springs
        self.ogden_alpha = float(getattr(model, 'ogden_alpha', 2.))
        self.ogden_col = int(ogden_col)          # theta column of log(alpha / ogden_alpha) per body (-1: fixed alpha)
        self.damping = int(model.damping > 0)
        self.gravity = wp.vec3(*model.gravity.tolist())
        self.ground_p = wp.vec3(*model.ground_point.tolist())
        self.ground_n = wp.vec3(*model.ground_normal.tolist())
        self.deterministic = deterministic
        self.kinematics = getattr(model, 'kinematics', 'small')
        self.residual = getattr(model, 'residual', False)
        # implicit_damping: dashpots solved per bond pair (edge colouring, exact two-body backward
        # Euler) before each elastic step instead of explicitly inside the bond kernel; the time step
        # is then limited by the elastic frequencies only (EnergyFoam.stable_dt(include_damping=False)).
        self.implicit_damping = implicit_damping and self.damping != 0
        self.graphs = graphs           # CUDA-graph replay of whole frames (forward passes; deterministic step only)
        assert not graphs or deterministic
        # graphs are captured / replayed on a dedicated stream (other Warp users, e.g. the renderer, may
        # leave the current stream on one that cannot be captured)
        self.stream = wp.Stream(wp.get_device('cuda')) if graphs else None
        assert not self.implicit_damping or deterministic
        if self.implicit_damping:
            i_np, j_np = model.i.cpu().numpy(), model.j.cpu().numpy()
            used = [set() for _ in range(self.n)]
            col = np.zeros(len(i_np), int)
            deg = np.bincount(np.concatenate((i_np, j_np)), minlength=self.n)
            for e_ in np.argsort(-(deg[i_np]+deg[j_np]), kind='stable'):
                a_, b_ = i_np[e_], j_np[e_]
                c_ = 0
                while c_ in used[a_] or c_ in used[b_]:
                    c_ += 1
                col[e_] = c_; used[a_].add(c_); used[b_].add(c_)
            self.n_colors = int(col.max())+1
            order_ = np.argsort(col, kind='stable')
            self.color_bonds = wp.array(order_.astype(np.int32), dtype=int, device='cuda')
            cnt_ = np.bincount(col, minlength=self.n_colors)
            self.color_off = [int(o_) for o_ in np.concatenate(([0], np.cumsum(cnt_)[:-1]))]
            self.color_cnt = [int(c_) for c_ in cnt_]
            bond_of = -np.ones((self.n_colors, self.n), np.int32)
            bond_of[col, i_np] = np.arange(len(i_np)); bond_of[col, j_np] = np.arange(len(i_np))
            self.bond_of = wp.array(bond_of, dtype=int, device='cuda')
        assert not self.residual or deterministic, 'residual term only in the deterministic step'
        if self.residual:
            self.den = f32(model.res_den)
        self.dilatation = getattr(model, 'body_law', '') == 'dilatation'
        if self.dilatation:
            assert deterministic and not self.residual
            wl2_ = model.ls_w*(model.ls_dX*model.ls_dX).sum(-1)
            self.trM = f32(torch.zeros(model.n, dtype=wl2_.dtype, device=wl2_.device).index_add(0, model.ls_a, wl2_))
        self.nodalvol = getattr(model, 'body_law', '') == 'nodalvol'
        if self.nodalvol:
            assert deterministic and not self.residual
            if getattr(model, 'tets', None) is None:
                model._setup_tets()
            T = model.tets.cpu().numpy().astype(np.int32)
            self.nt = len(T)
            self.tets = wp.array(T, dtype=int, device='cuda')
            order = np.argsort(T.reshape(-1), kind='stable')                 # corner index 4 t + k, grouped by body
            counts = np.bincount(T.reshape(-1), minlength=self.n)
            off_t = np.concatenate(([0], np.cumsum(counts))).astype(np.int32)
            self.t_off = wp.array(off_t, dtype=int, device='cuda')
            self.t_corner = wp.array(order.astype(np.int32), dtype=int, device='cuda')
            self.t_inc = wp.array((order//4).astype(np.int32), dtype=int, device='cuda')
            self.node_V0 = f32(model.node_V0)
            self.node_has = wp.from_torch(model.node_has.int().contiguous())
        self.bond_level = bool(getattr(model, 'bond_level', False))
        if self.bond_level:
            assert deterministic and getattr(model, 'cosserat', False) and not self.residual
            wl2 = model.ls_w*(model.ls_dX*model.ls_dX).sum(-1)
            self.trM = f32(torch.zeros(model.n, dtype=wl2.dtype, device=wl2.device).index_add(0, model.ls_a, wl2))
            self.bl_mid = int(bool(getattr(model, 'bond_level_mid', False)))
            self.bl_micro = int(bool(getattr(model, 'micropolar', True)))
        # shell path (BOND_BENDING.md): bond-direction residual, enhanced strain, thin-body closure
        thin_t = getattr(model, 'thin', None)
        self.shell = bool(getattr(model, 'enhanced', False)) or (thin_t is not None and bool(thin_t.any())) or \
            (bool(self.residual) and getattr(model, 'residual_mode', 'full') == 'normal') or bool(getattr(model, 'cosserat', False))
        self.cosserat = False
        if self.shell:
            assert deterministic and self.residual_mode_ok(model), 'shell path: deterministic, residual full not supported with shell'
            n_ = self.n
            dev_ = model.X.device
            z3 = lambda: torch.zeros(n_, 3, dtype=torch.float64, device=dev_)
            if thin_t is not None and bool(thin_t.any()):
                self.thin = wp.from_torch(thin_t.int().contiguous())
                self.te, self.t1, self.t2 = vec(model.thin_e), vec(model.thin_t1), vec(model.thin_t2)
            else:
                self.thin = wp.from_torch(torch.zeros(n_, dtype=torch.int32, device=dev_))
                self.te = self.t1 = self.t2 = vec(z3())
            self.cosserat = bool(getattr(model, 'cosserat', False))
            if self.cosserat:
                self.gyr = f32(model.gyration)
            if getattr(model, 'enhanced', False):
                self.eas_on = wp.from_torch(model.eas_on.int().contiguous()); self.eas_dir = vec(model.eas_dir)
            else:
                self.eas_on = wp.from_torch(torch.zeros(n_, dtype=torch.int32, device=dev_)); self.eas_dir = vec(z3())
        assert self.kinematics == 'small' or deterministic, 'exact bond kinematics only in the deterministic step'
        assert contact in ('hard', 'smooth', 'coulomb', 'hertz'), contact
        self.ground_mode = dict(hard=1, smooth=2, coulomb=3, hertz=4)[contact]
        self.hertz = contact == 'hertz'
        self.contact_scale = float(contact_scale)     # Hertz-Mindlin: multiplies F_n and k_t (1 = sphere-on-plane values)
        self.contact_offset_col = int(contact_offset_col)   # theta column of per-body contact-surface offsets (m), -1: none
        # simulation state: x, q, v, w (+ per-body Hertz-Mindlin contact history (F_t, d_prev))
        self.state_dtypes = (wp.vec3, wp.vec4, wp.vec3, wp.vec3)+((wp.vec4,) if self.hertz else ())
        self.state_sizes = (self.n,)*len(self.state_dtypes)
        # Zener bonds: per-bond viscous internal variables (displacement, rotation) appended to the state
        self.visco = bool(visco)
        if self.visco:
            assert deterministic and self.kinematics == 'small' and not self.shell, 'visco: small-kinematics deterministic springs only'
            self.i_visco = len(self.state_dtypes)
            self.state_dtypes = self.state_dtypes+(wp.vec3, wp.vec3)
            self.state_sizes = self.state_sizes+(self.nb, self.nb)
            self.visco_g0, self.visco_tau0 = float(visco_g0), float(visco_tau0)
            self.visco_cg, self.visco_ctau = int(visco_cols[0]), int(visco_cols[1])
        self.layer_dx, self.dt_ref = float(layer_dx), float(dt_ref)
        logit = lambda p: float(np.log(p/(1-p)))
        if contact in ('coulomb', 'hertz'):   # Coulomb coefficient mu_f = friction0 * exp(theta[:, 7]); theta[:, 8] unused
            self.friction0, self.restitution0 = float(np.log(friction0)), 0.
        else:
            self.friction0, self.restitution0 = logit(friction0), logit(restitution0)   # theta[:, 7:9] = 0 -> these values
        if deterministic:     # bond incidence per body, bodies then bonds in ascending order
            i, j = model.i.cpu().numpy(), model.j.cpu().numpy()
            body = np.concatenate((i, j)); bond = np.concatenate((np.arange(nb), np.arange(nb)))
            side = np.concatenate((np.ones(nb), -np.ones(nb)))
            order = np.lexsort((bond, body))
            off = np.zeros(self.n+1, np.int32); np.add.at(off, body+1, 1); off = np.cumsum(off).astype(np.int32)
            self.off = wp.array(off, dtype=int, device='cuda')
            self.inc = wp.array(bond[order].astype(np.int32), dtype=int, device='cuda')
            self.side = wp.array(side[order].astype(np.float32), dtype=float, device='cuda')

    @staticmethod
    def residual_mode_ok(model):
        return (not getattr(model, 'residual', False)) or getattr(model, 'residual_mode', 'full') == 'normal'

    def _buffers(self, grad, count=None):
        """Per-step work buffers. All arrays of `count` buffer sets (one set if None) are views into one arena
        (values) and one gradient arena, so zeroing them is one or two memsets instead of one per array (the step is
        launch bound: ~145 memsets per step otherwise). Same arrays, same contents, same kernels."""
        spec = self._buffer_spec()
        return _arena_sets(spec, grad, count)

    def _buffer_spec(self):
        """[(name, dtype, length)] of the per-step work buffers."""
        n = self.n
        out = {}
        z = lambda dtype: (dtype, n)
        out.update(f=z(wp.vec3), tau=z(wp.vec3), S=z(wp.mat33), G=z(wp.mat33))
        if self.deterministic:
            ze = lambda dtype: (dtype, self.nb)
            out.update(f_el=z(wp.vec3), bf=ze(wp.vec3), bti=ze(wp.vec3), btj=ze(wp.vec3), bS=ze(wp.mat33), bg=ze(wp.vec3))
            if self.residual:
                out.update(Fb=z(wp.mat33), bres=ze(wp.vec3))
            if getattr(self, 'dilatation', False):
                zs = lambda m_: (float, m_)
                out.update(dti=zs(self.nb), dtj=zs(self.nb), th_v=zs(self.n))
            if getattr(self, 'nodalvol', False):
                zs = lambda m_: (float, m_)
                out.update(Vt=zs(self.nt), Vn=zs(self.n), pa=zs(self.n), tg=(wp.vec3, 4*self.nt), f2=z(wp.vec3))
            if getattr(self, 'shell', False):
                out.update(Fb=z(wp.mat33), Gres=z(wp.mat33), G2=z(wp.mat33), bGi=ze(wp.mat33), bGj=ze(wp.mat33))
            if getattr(self, 'cosserat', False):
                out.update(bth=ze(wp.vec3), bN=ze(wp.mat33), N=z(wp.mat33), QM=z(wp.mat33), tau_b=z(wp.vec3), cti=ze(wp.vec3), ctj=ze(wp.vec3), tau2=z(wp.vec3))
            if getattr(self, 'bond_level', False):
                out.update(bres=ze(wp.vec3), bGi=ze(wp.mat33), bGj=ze(wp.mat33), Gres=z(wp.mat33), G2=z(wp.mat33),
                           blti=ze(wp.vec3), bltj=ze(wp.vec3), tau3=z(wp.vec3))
            if self.implicit_damping and getattr(self, 'damp_inplace', False) and getattr(self, 'damp_pre', True):   # in place: slower (65 latency-bound passes with a longer chain), kept for reference
                out.update(dvw_v=z(wp.vec3), dvw_w=z(wp.vec3), sv_i=ze(wp.vec3), sw_i=ze(wp.vec3), sv_j=ze(wp.vec3), sw_j=ze(wp.vec3))
            elif self.implicit_damping:
                for c_ in range(self.n_colors):
                    out[f'dv{c_}'] = z(wp.vec3); out[f'dw{c_}'] = z(wp.vec3)
            if self.implicit_damping:
                if getattr(self, 'damp_pre', True):
                    out.update(dp_pi=ze(wp.vec3), dp_pj=ze(wp.vec3), dp_Ainv=ze(wp.mat33), dp_Dm=ze(wp.mat33),
                               dp_m=ze(wp.vec4), dp_c=(float, self.nb))
        return [(k_, d_, m_) for k_, (d_, m_) in out.items()]

    def step(self, theta, s, s1, buf, dt):
        if self.deterministic:
            return self.step_det(theta, s, s1, buf, dt)
        x, q, v, w = s
        wp.launch(bond_kernel, self.nb, inputs=[x, q, v, w, theta, self.bi, self.bj, self.n0, self.ri, self.rj,
                                                self.base, self.cn, self.ct, self.cr, self.lw, self.dX, self.damping,
                                                buf['f'], buf['tau'], buf['S']])
        wp.launch(body_kernel, self.n, inputs=[buf['S'], self.Minv, theta, self.vol, self.k_v, self.k_mu,
                                               self.barrier, buf['G']])
        wp.launch(volume_kernel, self.nb, inputs=[buf['G'], self.bi, self.bj, self.lw, self.dX, buf['f']])
        wp.launch(integrate_kernel, self.n, inputs=[x, q, v, w, buf['f'], buf['tau'], self.mass, self.inertia,
                                                    self.fixed, self.contact, self.gravity, self.ground_p,
                                                    self.ground_n, self.ground_mode, self.layer_dx, self.dt_ref, theta, self.friction0, self.restitution0, dt, *s1[:4]])


def _step_det(self, theta, s, s1, buf, dt):
    x, q, v, w = s[:4]
    damping = self.damping
    if self.implicit_damping:
        if 'dvw_v' in buf:
            # in place: copy, then one launch per colour over that colour's bonds (no full-array copy per colour); these
            # launches stay off the tape and a hand-written adjoint is recorded instead (damp_inplace_adj_kernel)
            wp.launch(damping_pre_kernel, self.nb, inputs=[x, q, self.bi, self.bj, self.ri, self.rj, self.cn, self.ct, self.cr,
                                                           theta, self.mass, self.inertia, self.fixed, dt, buf['dp_pi'], buf['dp_pj'],
                                                           buf['dp_Ainv'], buf['dp_Dm'], buf['dp_m'], buf['dp_c']])
            vd, wd = buf['dvw_v'], buf['dvw_w']
            wp.launch(damp_copy_kernel, self.n, inputs=[v, w, vd, wd], record_tape=False)
            dpa = (buf['dp_pi'], buf['dp_pj'], buf['dp_Ainv'], buf['dp_Dm'], buf['dp_m'], buf['dp_c'])
            sva = (buf['sv_i'], buf['sw_i'], buf['sv_j'], buf['sw_j'])
            for c_ in range(self.n_colors):
                if self.color_cnt[c_]:
                    wp.launch(damp_inplace_kernel, self.color_cnt[c_], inputs=[self.color_bonds, self.color_off[c_], vd, wd, self.bi, self.bj,
                                                                               *dpa, dt, *sva], record_tape=False)
            tape_ = wp.context.runtime.tape
            if tape_ is not None and v.requires_grad:
                v_in, w_in, wf_ = v, w, self

                def _damp_backward(v_in=v_in, w_in=w_in, vd=vd, wd=wd, dpa=dpa, sva=sva, dt=dt, wf_=wf_):
                    for c_ in reversed(range(wf_.n_colors)):
                        if wf_.color_cnt[c_]:
                            wp.launch(damp_inplace_adj_kernel, wf_.color_cnt[c_], inputs=[wf_.color_bonds, wf_.color_off[c_], vd.grad, wd.grad,
                                                                                          wf_.bi, wf_.bj, *sva, *dpa, dt, *[a_.grad for a_ in dpa]],
                                      record_tape=False)
                    wp.launch(damp_copy_adj_kernel, wf_.n, inputs=[vd.grad, wd.grad, v_in.grad, w_in.grad], record_tape=False)
                tape_.record_func(backward=_damp_backward, arrays=[])
            v, w = vd, wd
        elif getattr(self, 'damp_pre', True) and 'dp_pi' in buf:
            # velocity-independent factors once per step, then the colour sweep (same arithmetic, shorter dependent chain)
            wp.launch(damping_pre_kernel, self.nb, inputs=[x, q, self.bi, self.bj, self.ri, self.rj, self.cn, self.ct, self.cr,
                                                           theta, self.mass, self.inertia, self.fixed, dt, buf['dp_pi'], buf['dp_pj'],
                                                           buf['dp_Ainv'], buf['dp_Dm'], buf['dp_m'], buf['dp_c']])
            for c_ in range(self.n_colors):
                wp.launch(damping_apply_kernel, self.n, inputs=[v, w, self.bond_of, c_, self.bi, self.bj, buf['dp_pi'], buf['dp_pj'],
                                                                buf['dp_Ainv'], buf['dp_Dm'], buf['dp_m'], buf['dp_c'], dt,
                                                                buf[f'dv{c_}'], buf[f'dw{c_}']])
                v, w = buf[f'dv{c_}'], buf[f'dw{c_}']
        else:
            for c_ in range(self.n_colors):
                wp.launch(damping_pair_kernel, self.n, inputs=[x, q, v, w, self.bond_of, c_, self.bi, self.bj, self.ri, self.rj,
                                                               self.cn, self.ct, self.cr, theta, self.mass, self.inertia, self.fixed, dt,
                                                               buf[f'dv{c_}'], buf[f'dw{c_}']])
                v, w = buf[f'dv{c_}'], buf[f'dw{c_}']
        damping = 0
    if getattr(self, 'visco', False):
        assert damping == 0, 'visco: implicit damping only'
        iv = self.i_visco
        wp.launch(bond_kernel_det_visco, self.nb, inputs=[x, q, v, w, theta, self.bi, self.bj, self.n0, self.ri, self.rj, self.base,
                                                          self.lw, self.dX, s[iv], s[iv+1], self.visco_g0, self.visco_tau0,
                                                          self.visco_cg, self.visco_ctau, dt,
                                                          buf['bf'], buf['bti'], buf['btj'], buf['bS'], s1[iv], s1[iv+1], self.nl_bonds, self.ogden_alpha, self.ogden_col])
    else:
        if self.kinematics == 'exact':
            assert not self.nl_bonds, 'nonlinear bonds: small kinematics only'
            wp.launch(bond_kernel_exact_det, self.nb, inputs=[x, q, v, w, theta, self.bi, self.bj, self.n0, self.ri, self.rj,
                                                              self.base, self.cn, self.ct, self.cr, self.lw, self.dX, damping,
                                                              buf['bf'], buf['bti'], buf['btj'], buf['bS']])
        else:
            wp.launch(bond_kernel_det, self.nb, inputs=[x, q, v, w, theta, self.bi, self.bj, self.n0, self.ri, self.rj,
                                                        self.base, self.cn, self.ct, self.cr, self.lw, self.dX, damping,
                                                        buf['bf'], buf['bti'], buf['btj'], buf['bS'], self.nl_bonds, self.ogden_alpha, self.ogden_col])
    wp.launch(gather_kernel, self.n, inputs=[self.off, self.inc, self.side, buf['bf'], buf['bti'], buf['btj'], buf['bS'],
                                             buf['f_el'], buf['tau'], buf['S']])
    if getattr(self, 'shell', False):
        assert not getattr(self, 'hertz', False), 'Hertz contact: springs / volume body laws only'
        tau_arr = buf['tau']
        if getattr(self, 'cosserat', False):
            if getattr(self, 'bl_micro', 1) == 1:    # curvature (micropolar); the classical model leaves N = 0 (no curvature)
                wp.launch(cosserat_bond_kernel, self.nb, inputs=[q, self.bi, self.bj, self.lw, self.dX, buf['bth'], buf['bN']])
                wp.launch(gather_mat_both_kernel, self.n, inputs=[self.off, self.inc, buf['bN'], buf['N']])
            wp.launch(cosserat_body_kernel, self.n, inputs=[buf['S'], buf['N'], self.Minv, q, theta, self.vol, self.gyr, self.k_v, self.k_mu, self.barrier,
                                                            self.thin, self.te, self.t1, self.t2, self.eas_on, self.eas_dir,
                                                            x, self.ground_p, self.ground_n, self.contact, self.layer_dx, 1, int(getattr(self, 'bond_level', False)),
                                                            buf['Fb'], buf['G'], buf['QM'], buf['tau_b']])
            wp.launch(curvature_torque_kernel, self.nb, inputs=[q, buf['bth'], buf['QM'], self.bi, self.bj, self.lw, self.dX, buf['cti'], buf['ctj']])
            wp.launch(gather_tau_kernel, self.n, inputs=[self.off, self.inc, self.side, buf['cti'], buf['ctj'], buf['tau_b'], buf['tau'], buf['tau2']])
            tau_arr = buf['tau2']
            if getattr(self, 'bond_level', False):
                wp.launch(bond_level_kernel, self.nb, inputs=[x, q, buf['Fb'], theta, self.vol, self.k_v, self.k_mu, self.barrier, self.bi, self.bj,
                                                              self.lw, self.dX, self.trM, self.bl_mid, self.bl_micro,
                                                              buf['bres'], buf['bGi'], buf['bGj'], buf['blti'], buf['bltj']])
                wp.launch(gather_mat_kernel, self.n, inputs=[self.off, self.inc, self.side, buf['bGi'], buf['bGj'], buf['Gres']])
                wp.launch(pullback_add_kernel, self.n, inputs=[buf['Gres'], buf['S'], self.Minv, self.thin, self.te, self.t1, self.t2,
                                                               buf['G'], buf['G2']])
                wp.launch(gather_tau_kernel, self.n, inputs=[self.off, self.inc, self.side, buf['blti'], buf['bltj'], buf['tau_b'], buf['tau2'], buf['tau3']])
                tau_arr = buf['tau3']
                wp.launch(volume_kernel_det, self.nb, inputs=[buf['G2'], self.bi, self.bj, self.lw, self.dX, buf['bg']])
                wp.launch(gather_add2_kernel, self.n, inputs=[self.off, self.inc, self.side, buf['bg'], buf['bres'], buf['f_el'], buf['f']])
                wp.launch(integrate_kernel, self.n, inputs=[x, q, v, w, buf['f'], tau_arr, self.mass, self.inertia,
                                                            self.fixed, self.contact, self.gravity, self.ground_p,
                                                            self.ground_n, self.ground_mode, self.layer_dx, self.dt_ref, theta, self.friction0, self.restitution0, dt, *s1[:4]])
                return
        else:
            wp.launch(shell_body_kernel, self.n, inputs=[buf['S'], self.Minv, theta, self.vol, self.k_v, self.k_mu, self.barrier,
                                                         self.thin, self.te, self.t1, self.t2, self.eas_on, self.eas_dir,
                                                         x, self.ground_p, self.ground_n, self.contact, self.layer_dx, 1, buf['Fb'], buf['G']])
        if self.residual:
            wp.launch(residual_normal_kernel, self.nb, inputs=[x, buf['Fb'], theta, self.bi, self.bj, self.lw, self.dX, self.vol,
                                                               self.den, self.k_mu, buf['bres'], buf['bGi'], buf['bGj']])
            wp.launch(gather_mat_kernel, self.n, inputs=[self.off, self.inc, self.side, buf['bGi'], buf['bGj'], buf['Gres']])
            wp.launch(pullback_add_kernel, self.n, inputs=[buf['Gres'], buf['S'], self.Minv, self.thin, self.te, self.t1, self.t2,
                                                           buf['G'], buf['G2']])
            wp.launch(volume_kernel_det, self.nb, inputs=[buf['G2'], self.bi, self.bj, self.lw, self.dX, buf['bg']])
            wp.launch(gather_add2_kernel, self.n, inputs=[self.off, self.inc, self.side, buf['bg'], buf['bres'], buf['f_el'], buf['f']])
        else:
            wp.launch(volume_kernel_det, self.nb, inputs=[buf['G'], self.bi, self.bj, self.lw, self.dX, buf['bg']])
            wp.launch(gather_add_kernel, self.n, inputs=[self.off, self.inc, self.side, buf['bg'], buf['f_el'], buf['f']])
        wp.launch(integrate_kernel, self.n, inputs=[x, q, v, w, buf['f'], tau_arr, self.mass, self.inertia,
                                                    self.fixed, self.contact, self.gravity, self.ground_p,
                                                    self.ground_n, self.ground_mode, self.layer_dx, self.dt_ref, theta, self.friction0, self.restitution0, dt, *s1[:4]])
        return
    if getattr(self, 'nodalvol', False):          # springs + true-volume term (tetrahedra); the bond-body force bg stays 0
        wp.launch(tet_vol_kernel, self.nt, inputs=[x, self.tets, buf['Vt']])
        wp.launch(node_volume_kernel, self.n, inputs=[self.t_off, self.t_inc, buf['Vt'], buf['Vn']])
        wp.launch(node_pressure_kernel, self.n, inputs=[buf['Vn'], self.node_V0, self.node_has, theta, self.k_v, self.barrier, buf['pa']])
        wp.launch(tet_force_kernel, self.nt, inputs=[x, self.tets, buf['pa'], buf['tg']])
    elif getattr(self, 'dilatation', False):        # springs + volumetric multi-body term (no averaged F)
        wp.launch(dil_strain_kernel, self.nb, inputs=[x, self.bi, self.bj, self.lw, self.dX, self.trM, buf['dti'], buf['dtj']])
        wp.launch(gather_scalar_kernel, self.n, inputs=[self.off, self.inc, self.side, buf['dti'], buf['dtj'], buf['th_v']])
        wp.launch(dil_force_kernel, self.nb, inputs=[x, buf['th_v'], theta, self.vol, self.k_v, self.bi, self.bj, self.lw, self.dX,
                                                     self.trM, buf['bg']])
    else:
        wp.launch(body_kernel, self.n, inputs=[buf['S'], self.Minv, theta, self.vol, self.k_v, self.k_mu,
                                               self.barrier, buf['G']])
        wp.launch(volume_kernel_det, self.nb, inputs=[buf['G'], self.bi, self.bj, self.lw, self.dX, buf['bg']])
    if self.residual:
        wp.launch(deformation_kernel, self.n, inputs=[buf['S'], self.Minv, buf['Fb']])
        wp.launch(residual_kernel, self.nb, inputs=[x, buf['Fb'], theta, self.bi, self.bj, self.lw, self.dX, self.vol,
                                                    self.den, self.k_mu, buf['bres']])
        wp.launch(gather_add2_kernel, self.n, inputs=[self.off, self.inc, self.side, buf['bg'], buf['bres'], buf['f_el'], buf['f']])
    else:
        wp.launch(gather_add_kernel, self.n, inputs=[self.off, self.inc, self.side, buf['bg'], buf['f_el'], buf['f']])
    f_arr = buf['f']
    if getattr(self, 'nodalvol', False):
        wp.launch(node_force_kernel, self.n, inputs=[self.t_off, self.t_corner, buf['tg'], buf['f'], buf['f2']])
        f_arr = buf['f2']
    if getattr(self, 'hertz', False):
        wp.launch(hertz_integrate_kernel, self.n, inputs=[x, q, v, w, s[4], f_arr, buf['tau'], self.mass, self.inertia, self.fixed,
                                                          self.contact, self.gravity, self.ground_p, self.ground_n, theta,
                                                          self.k_v, self.mu_ref, self.friction0, self.contact_scale, self.contact_offset_col,
                                                          int(getattr(self, 'thornton', 1)), dt, *s1[:5]])
        return
    wp.launch(integrate_kernel, self.n, inputs=[x, q, v, w, f_arr, buf['tau'], self.mass, self.inertia,
                                                self.fixed, self.contact, self.gravity, self.ground_p,
                                                self.ground_n, self.ground_mode, self.layer_dx, self.dt_ref, theta, self.friction0, self.restitution0, dt, *s1[:4]])


WarpFoam.step_det = _step_det


def _frame_graph(wf, th, start, other, buf, dt, n):
    """CUDA graph of n ping-pong steps from `start` (buffers zeroed inside); returns (graph, end state, spare)."""
    with wp.ScopedCapture() as cap:
        a, b = start, other
        for _ in range(n):
            _zero_sets([buf])
            wf.step(th, a, b, buf, dt)
            a, b = b, a
    return cap.graph, a, b


def _state_dtypes(wf):
    return getattr(wf, 'state_dtypes', (wp.vec3, wp.vec4, wp.vec3, wp.vec3))


def _state_sizes(wf):
    return getattr(wf, 'state_sizes', (wf.n,)*len(_state_dtypes(wf)))


class _ArenaDict(dict):
    """dict of array views into shared arenas (attributes arena / garena), see _arena_sets."""


def _arena_sets(spec, grad, count=None):
    """count (or 1) sets of zeroed arrays [(name, dtype, length)] as views into one value arena and, with grad, one
    gradient arena (256-byte aligned slices). Returns a dict (count None) or a list of dicts; each dict carries the
    shared arenas as .arena / .garena."""
    k = 1 if count is None else count
    offs, off = [], 0
    for _, d_, m_ in spec:
        off = (off+255)//256*256
        offs.append(off)
        off += m_*wp.types.type_size_in_bytes(d_)
    per = (off+255)//256*256
    arena = wp.zeros(per*k//4, dtype=float, device='cuda')
    garena = wp.zeros(per*k//4, dtype=float, device='cuda') if grad else None
    sets = []
    for s_ in range(k):
        d = _ArenaDict()
        for (nm_, d_, m_), o_ in zip(spec, offs):
            b_ = s_*per+o_
            g_ = wp.array(ptr=garena.ptr+b_, dtype=d_, shape=(m_,), device='cuda') if grad else None
            d[nm_] = wp.array(ptr=arena.ptr+b_, dtype=d_, shape=(m_,), device='cuda', grad=g_, requires_grad=grad)
        d.arena, d.garena = arena, garena
        sets.append(d)
    return sets[0] if count is None else sets


def _zero_sets(sets, values=True, grads=False):
    """Zero the arenas behind these buffer / state sets (each arena once)."""
    seen = set()
    for d in sets:
        for a_ in ((d.arena if values else None), (d.garena if grads else None)):
            if a_ is not None and a_.ptr not in seen:
                seen.add(a_.ptr)
                a_.zero_()


def _state_spec(wf):
    return [(str(i_), d, m_) for i_, (d, m_) in enumerate(zip(_state_dtypes(wf), _state_sizes(wf)))]


def _new_state(wf, grad=False):
    return tuple(wp.zeros(m_, dtype=d, device='cuda', requires_grad=grad) for d, m_ in zip(_state_dtypes(wf), _state_sizes(wf)))


def _new_states(wf, k, grad=False):
    """k states (tuples of arrays) from one arena; returns (list of tuples, list of the arena dicts)."""
    sets = _arena_sets(_state_spec(wf), grad, k)
    return [tuple(d[str(i_)] for i_ in range(len(d))) for d in sets], sets


def _pad_state(wf, state0):
    """state0 = (x, q, v, w) torch tensors; extra state (Hertz contact history, Zener bond variables) starts at zero."""
    dts, sz = _state_dtypes(wf), _state_sizes(wf)
    out = list(state0)
    for d, m_ in zip(dts[len(out):], sz[len(out):]):
        out.append(torch.zeros(m_, 4 if d == wp.vec4 else 3, dtype=state0[0].dtype, device=state0[0].device))
    return out


class _Rollout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, theta, wf, state0, dt, substeps, frames):
        with (wp.ScopedStream(wf.stream) if getattr(wf, 'stream', None) is not None else _Null()):
            return _Rollout._forward(ctx, theta, wf, state0, dt, substeps, frames)

    @staticmethod
    def backward(ctx, gpos, gquat):
        wf_ = getattr(ctx, 'wf', None)
        with (wp.ScopedStream(wf_.stream) if wf_ is not None and getattr(wf_, 'stream', None) is not None else _Null()):
            return _Rollout._backward(ctx, gpos, gquat)

    @staticmethod
    def _forward(ctx, theta, wf, state0, dt, substeps, frames):
        grad = theta.requires_grad
        th = wp.from_torch(theta.detach().float().contiguous(), requires_grad=grad)
        n = wf.n
        new = lambda dtype: wp.zeros(n, dtype=dtype, device='cuda', requires_grad=grad)
        dts = _state_dtypes(wf)
        s = tuple(wp.from_torch(t.detach().float().contiguous().clone(), dtype=d, requires_grad=grad)
                  for t, d in zip(_pad_state(wf, state0), dts))
        tape = wp.Tape() if grad else None
        frames_x, frames_q = ([s[0]], [s[1]]) if grad else ([wp.clone(s[0])], [wp.clone(s[1])])  # ping-pong reuses s
        if not grad:  # ping-pong buffers, no history
            other = _new_state(wf)
            buf = wf._buffers(False)
        if not grad and getattr(wf, 'graphs', False):
            graphs = {}
            cur, spare = s, other
            for f_ in range(frames):
                key = id(cur[0])
                if key not in graphs:
                    graphs[key] = _frame_graph(wf, th, cur, spare, buf, dt, substeps)
                g_, nxt, spr = graphs[key]
                wp.capture_launch(g_)
                cur, spare = nxt, spr
                frames_x.append(wp.clone(cur[0])); frames_q.append(wp.clone(cur[1]))
            pos = torch.stack([wp.to_torch(a_) for a_ in frames_x])
            quat = torch.stack([wp.to_torch(a_) for a_ in frames_q])
            ctx.tape, ctx.th, ctx.frames_x, ctx.frames_q = None, th, frames_x, frames_q
            return pos.clone(), quat.clone()
        with (tape if grad else _Null()):
            for k in range(frames*substeps):
                if grad:
                    s1 = _new_state(wf, True)
                    buf = wf._buffers(True)
                else:
                    s1 = other
                    _zero_sets([buf])
                wf.step(th, s, s1, buf, dt)
                if grad:
                    s = s1
                else:
                    s, other = s1, s
                if (k+1) % substeps == 0:
                    if grad:
                        frames_x.append(s[0]); frames_q.append(s[1])
                    else:
                        frames_x.append(wp.clone(s[0])); frames_q.append(wp.clone(s[1]))
        pos = torch.stack([wp.to_torch(a) for a in frames_x])
        quat = torch.stack([wp.to_torch(a) for a in frames_q])
        ctx.tape, ctx.th, ctx.frames_x, ctx.frames_q = tape, th, frames_x, frames_q
        return pos.clone(), quat.clone()

    @staticmethod
    def _backward(ctx, gpos, gquat):
        grads = {}
        for k in range(1, len(ctx.frames_x)):
            grads[ctx.frames_x[k]] = wp.from_torch(gpos[k].contiguous(), dtype=wp.vec3)
            grads[ctx.frames_q[k]] = wp.from_torch(gquat[k].contiguous(), dtype=wp.vec4)
        ctx.tape.backward(grads=grads)
        g = wp.to_torch(ctx.th.grad).clone()
        ctx.tape.zero(); ctx.tape = None
        return g, None, None, None, None, None


REST_FIELDS = (('n0', wp.vec3), ('ri', wp.vec3), ('rj', wp.vec3), ('base', None), ('cn', None), ('ct', None), ('cr', None),
               ('lw', None), ('dX', wp.vec3), ('Minv', wp.mat33), ('node_V0', None))


def set_rest(wf, rest, grad=False):
    """Replace the rest-geometry arrays of a WarpFoam (bond frames, levers, stiffnesses, dashpots, least-squares
    weights / vectors / moment inverses, tetrahedron node volumes) by these torch tensors, in REST_FIELDS order;
    grad=True makes the kernels accumulate their adjoints (returned by the checkpointed rollout)."""
    arrs = []
    for (name, dt_), t_ in zip(REST_FIELDS, rest):
        t_ = t_.detach().float().contiguous().clone()
        a_ = wp.from_torch(t_, dtype=dt_, requires_grad=grad) if dt_ is not None else wp.from_torch(t_, requires_grad=grad)
        setattr(wf, name, a_)
        arrs.append(a_)
    return arrs


class _RolloutCheckpointed(torch.autograd.Function):
    """Same rollout, segment checkpointing: the forward keeps only the state at the start of every
    `segment` substeps (ping-pong buffers, no tape); the backward replays the segments last to
    first with a tape over a reused pool of segment+1 states and buffers, seeds the segment end
    with the adjoint carried from later segments (plus the frame-output gradients that fall in
    it), and carries the start-state adjoint back. Exact gradients; memory ~ segment steps."""

    @staticmethod
    def forward(ctx, theta, x0, q0, v0, w0, wf, dt, substeps, frames, segment, *rest):
        # graphs are captured inside every call (forward and backward), after the rest arrays are set
        ctx.rest_arrays = set_rest(wf, rest, grad=True) if rest else []
        with (wp.ScopedStream(wf.stream) if getattr(wf, 'stream', None) is not None else _Null()):
            return _RolloutCheckpointed._forward(ctx, theta, wf, (x0, q0, v0, w0), dt, substeps, frames, segment)

    @staticmethod
    def backward(ctx, gpos, gquat):
        wf_ = getattr(ctx, 'wf', None)
        with (wp.ScopedStream(wf_.stream) if wf_ is not None and getattr(wf_, 'stream', None) is not None else _Null()):
            return _RolloutCheckpointed._backward(ctx, gpos, gquat)

    @staticmethod
    def _forward(ctx, theta, wf, state0, dt, substeps, frames, segment):
        n, total = wf.n, frames*substeps
        dtypes = _state_dtypes(wf)
        new = lambda dtype, grad=False: wp.zeros(n, dtype=dtype, device='cuda', requires_grad=grad)
        th = wp.from_torch(theta.detach().float().contiguous(), requires_grad=True)
        s = tuple(wp.from_torch(t.detach().float().contiguous().clone(), dtype=d) for t, d in zip(_pad_state(wf, state0), dtypes))
        other = _new_state(wf)
        buf = wf._buffers(False)
        starts = []
        frames_x, frames_q = [wp.to_torch(s[0]).clone()], [wp.to_torch(s[1]).clone()]
        if getattr(wf, 'graphs', False):
            # frames split into chunks of <= segment steps; one CUDA graph per (chunk length, start buffer)
            graphs, chunks = {}, []
            cur, spare = s, other
            for f_ in range(frames):
                k, left = f_*substeps, substeps
                while left > 0:
                    m_ = min(segment, left)
                    starts.append(tuple(wp.clone(a_) for a_ in cur)); chunks.append((k, m_))
                    key = (id(cur[0]), m_)
                    if key not in graphs:
                        graphs[key] = _frame_graph(wf, th, cur, spare, buf, dt, m_)
                    g_, nxt, spr = graphs[key]
                    wp.capture_launch(g_)
                    cur, spare = nxt, spr
                    k += m_; left -= m_
                frames_x.append(wp.to_torch(cur[0]).clone()); frames_q.append(wp.to_torch(cur[1]).clone())
            ctx.wf, ctx.th, ctx.starts, ctx.dt, ctx.substeps, ctx.total, ctx.segment = wf, th, starts, dt, substeps, total, segment
            ctx.chunks = chunks
            return torch.stack(frames_x), torch.stack(frames_q)
        for k in range(total):
            if k % segment == 0:
                starts.append(tuple(wp.clone(a) for a in s))
            _zero_sets([buf])
            wf.step(th, s, other, buf, dt)
            s, other = other, s
            if (k+1) % substeps == 0:
                frames_x.append(wp.to_torch(s[0]).clone()); frames_q.append(wp.to_torch(s[1]).clone())
        ctx.wf, ctx.th, ctx.starts, ctx.dt, ctx.substeps, ctx.total, ctx.segment = wf, th, starts, dt, substeps, total, segment
        ctx.chunks = [(i_*segment, min(segment, total-i_*segment)) for i_ in range(len(starts))]
        return torch.stack(frames_x), torch.stack(frames_q)

    @staticmethod
    def _backward(ctx, gpos, gquat):
        wf, th, L, sub = ctx.wf, ctx.th, ctx.segment, ctx.substeps
        dtypes = _state_dtypes(wf)
        new = lambda dtype: wp.zeros(wf.n, dtype=dtype, device='cuda', requires_grad=True)
        pool, pool_sets = _new_states(wf, L+1, True)          # one arena for the pool, one for all the work buffers
        bufs = wf._buffers(True, L)
        adj = [torch.zeros(m_, 4 if d == wp.vec4 else 3, device='cuda') for d, m_ in zip(dtypes, _state_sizes(wf))]
        g_theta = torch.zeros_like(wp.to_torch(th))
        rest_arrays = getattr(ctx, 'rest_arrays', [])
        for (name_, _), a_ in zip(REST_FIELDS, rest_arrays):     # another rollout may have replaced them since the forward
            setattr(wf, name_, a_)
        g_rest = [torch.zeros_like(wp.to_torch(a_)) for a_ in rest_arrays]
        use_graphs = getattr(wf, 'graphs', False)
        # wf.adj_decay = r: a second adjoint seeded with r^f x (frame-f output gradient), carried through the same
        # re-computed segments (second tape backward); it gives the initial-state gradient (returned) and its own
        # theta gradient (wf.g_theta_decay); theta / rest gradients returned are the plain ones (also wf.g_theta_plain)
        r_ = getattr(wf, 'adj_decay', None)
        adj2 = [a_.clone() for a_ in adj] if r_ is not None else None
        g_theta2 = torch.zeros_like(g_theta) if r_ is not None else None

        def zero_grads(m):
            _zero_sets(pool_sets, values=False, grads=True)
            _zero_sets(bufs, values=False, grads=True)
            th.grad.zero_()
            for a_ in rest_arrays:
                a_.grad.zero_()

        def seed(m, k0, carried, wgt):
            for a_, g_ in zip(pool[m], carried):
                wp.to_torch(a_.grad).add_(g_)
            for j in range(1, m+1):
                k = k0+j
                if k % sub == 0:
                    f = k//sub
                    w_ = 1. if wgt is None else (wgt**f if isinstance(wgt, float) else float(wgt[f]))
                    wp.to_torch(pool[j][0].grad).add_(w_*gpos[f])
                    wp.to_torch(pool[j][1].grad).add_(w_*gquat[f])
        cache = {}
        for seg in reversed(range(len(ctx.starts))):
            k0, m = ctx.chunks[seg]
            if use_graphs:
                if m not in cache:            # graphs: zero / taped forward / tape backward / zero grads, reused by every chunk of length m
                    with wp.ScopedCapture() as c0:
                        _zero_sets(pool_sets, values=True, grads=True)
                        _zero_sets(bufs, values=True, grads=True)
                        th.grad.zero_()
                        for a_ in rest_arrays:
                            a_.grad.zero_()
                    tape_m = wp.Tape()
                    with wp.ScopedCapture() as c1:
                        with tape_m:
                            for j in range(m):
                                wf.step(th, pool[j], pool[j+1], bufs[j], ctx.dt)
                    with wp.ScopedCapture() as c2:
                        tape_m.backward()
                    c3 = None
                    if r_ is not None:
                        with wp.ScopedCapture() as c3_:
                            zero_grads(m)
                        c3 = c3_.graph
                    cache[m] = (c0.graph, tape_m, c1.graph, c2.graph, c3)
                g0, tape_m, g1, g2, g3 = cache[m]
                wp.capture_launch(g0)
                for a_, b_ in zip(pool[0], ctx.starts[seg]):
                    wp.copy(a_, b_)
                wp.capture_launch(g1)
            else:
                _zero_sets(pool_sets, values=True, grads=True)
                _zero_sets(bufs, values=True, grads=True)
                th.grad.zero_()
                for a_ in rest_arrays:
                    a_.grad.zero_()
                for a_, b_ in zip(pool[0], ctx.starts[seg]):
                    wp.copy(a_, b_)
                tape = wp.Tape()
                with tape:
                    for j in range(m):
                        wf.step(th, pool[j], pool[j+1], bufs[j], ctx.dt)
            # seed: carried adjoint at the segment end + frame-output gradients inside the segment
            seed(m, k0, adj, None)
            if use_graphs:
                wp.capture_launch(g2)
            else:
                tape.backward()
            g_theta += wp.to_torch(th.grad)
            for g_, a_ in zip(g_rest, rest_arrays):
                g_ += wp.to_torch(a_.grad)
            adj = [wp.to_torch(a_.grad).clone() for a_ in pool[0]]
            if r_ is not None:            # second adjoint over the same taped segment
                if use_graphs:
                    wp.capture_launch(g3)
                else:
                    zero_grads(m)
                seed(m, k0, adj2, r_)
                if use_graphs:
                    wp.capture_launch(g2)
                else:
                    tape.backward()
                g_theta2 += wp.to_torch(th.grad)
                adj2 = [wp.to_torch(a_.grad).clone() for a_ in pool[0]]
            if not use_graphs:
                tape.reset()
        # adj: gradient with respect to the initial state (x, q, v, w, ...); output frame 0 is the initial state itself,
        # so its gradient (e.g. a camera rendered between frames 0 and 1) goes straight to x0 / q0 (weight r^0 = 1)
        n_ = gpos.shape[1]
        if r_ is not None:
            # r_: float (weights r^f) or per-frame weights; wf.adj2_stash: return the plain initial-state gradient and keep
            # the weighted one in wf.adj_init2 (caller picks per parameter), else return the weighted one
            wf.g_theta_plain, wf.g_theta_decay = g_theta.clone(), g_theta2
            w0_ = 1. if isinstance(r_, float) else float(r_[0])
            a2_ = [a_.clone() for a_ in adj2]
            a2_[0][:n_] += w0_*gpos[0]
            a2_[1][:n_] += w0_*gquat[0]
            if getattr(wf, 'adj2_stash', False):
                wf.adj_init2 = tuple(a2_[:4])
                p_ = [a_.clone() for a_ in adj[:4]]
                p_[0][:n_] += gpos[0]
                p_[1][:n_] += gquat[0]
                wf.adj_init_plain = tuple(p_)
            else:
                adj = adj2
        gx0, gq0 = adj[0].clone(), adj[1].clone()
        gx0[:n_] += gpos[0]
        gq0[:n_] += gquat[0]
        return (g_theta, gx0, gq0, adj[2], adj[3], None, None, None, None, None)+tuple(g_rest)


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def rollout(wf, theta, state0, dt, substeps, frames, segment=None, rest=None):
    """(positions (frames+1, n, 3), quaternions (frames+1, n, 4)); differentiable w.r.t. theta.
    segment: None keeps the whole tape (memory ~ frames*substeps steps); an int checkpoints every
    `segment` substeps (memory ~ segment steps, ~one extra forward in the backward).
    rest: optional tensors of the rest geometry (REST_FIELDS order); with segment and grad they are differentiable."""
    grad_rest = rest is not None and any(t_.requires_grad for t_ in rest)
    if segment is None or not (theta.requires_grad or grad_rest or any(t_.requires_grad for t_ in state0[:4])):
        if rest is not None:
            set_rest(wf, rest, grad=False)
        return _Rollout.apply(theta, wf, state0, float(dt), int(substeps), int(frames))
    x0, q0, v0, w0 = state0[:4]
    return _RolloutCheckpointed.apply(theta, x0, q0, v0, w0, wf, float(dt), int(substeps), int(frames), int(segment),
                                      *(rest if rest is not None else ()))
