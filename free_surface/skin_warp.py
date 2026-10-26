"""The skin model's energy and gradient in Warp, float64 (2026-10-23, W1): the same terms as sphere_model.SkinTerms
(BoundaryTerms, lever c, quad_ref, variance) with the bonds of the clean model (kinematics 'exact', no nodal volume),
theta = 0. Kernels in the order of the torch code:
  F        S_B from the bonds (A / 2 (x_j - x_i) (x) n) and the skin levers (w (y - x_B) (x) l), F_B = S_B Mt^-1;
           R_B = polar(F_B) by 12 Newton steps (R <- (R + R^-T) / 2), q_B = matrix_quat (Shepperd)
  bonds    exact kinematics (foam_dem_energy._bond_exact): theta = log(q_i* q_j) (principal), mid frame
           R_m = R(q_i sqrt(q_rel)), d = R_m^T (x_j + R_j r_j - x_i - R_i r_i); 1/2 (k_n dn^2 + k_s |d_t|^2 + k_b |th_t|^2 + k_t th_n^2)
           on d - dp, theta - thp (the plastic references, plastic.py; 0 elastic)
  Gamma    'sym_skin' (SkinTerms): rhs_B = sum over the bonds of B of wb A(dX)^T (F_j - F_i) (rows), s = N_B^-1 rhs_B (the 6 x 6
           of bonds and skin observations), the squares' deviation delta = delta_aff - 1/2 s . q(d), turned into the body
           frame (R_B^T delta)
  outer    o1 = mu r tr(D^T D N) per patch, o2 = 2 mu delta_aff . D c per entry (D = F - R P_p, P_p the cell's plastic
           stretch, I elastic), o3 and lateral per pair
  volume   walls (SkinTerms.wall_volume, Hermite), det F (V_res + cones), skin (delta_aff . cof(F) c / 3); log J with the
           model's barrier; V lam / 2 lj^2 and the variance lam A L / 24 (lj_i - lj_j)^2
  viscous  (SkinWarp.set_viscosity) each term again on its measures' change since the step's start, mu -> eta, lam -> lam_v
The gradient is Warp's reverse mode (wp.Tape) of the total."""
import math

import numpy as np
import torch
import warp as wp

f64 = torch.float64
NT = 9                                       # terms: bonds, volume, o1, o2, o3, lateral, contact (augmented Lagrangian), viscous (Et),
                                             # friction (augmented Lagrangian, 10-25)
NTC = wp.constant(NT)
NSLOT = wp.constant(256)                     # each term summed over 256 slots first (one address per term contends)


@wp.func
def quat_mul(a: wp.vec4d, b: wp.vec4d):
    return wp.vec4d(a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3], a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2],
                    a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1], a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0])


@wp.func
def quat_mat(q: wp.vec4d):
    w, x, y, z = q[0], q[1], q[2], q[3]
    one, two = wp.float64(1.0), wp.float64(2.0)
    return wp.mat33d(one-two*(y*y+z*z), two*(x*y-w*z), two*(x*z+w*y),
                     two*(x*y+w*z), one-two*(x*x+z*z), two*(y*z-w*x),
                     two*(x*z-w*y), two*(y*z+w*x), one-two*(x*x+y*y))


@wp.func
def mat_quat(R: wp.mat33d):
    """Shepperd, as sphere_model.matrix_quat (first largest of 4w^2, 4x^2, 4y^2, 4z^2)"""
    one, two = wp.float64(1.0), wp.float64(2.0)
    tr = R[0, 0]+R[1, 1]+R[2, 2]
    c0, c1, c2, c3 = one+tr, one+two*R[0, 0]-tr, one+two*R[1, 1]-tr, one+two*R[2, 2]-tr
    a21, a02, a10 = R[2, 1]-R[1, 2], R[0, 2]-R[2, 0], R[1, 0]-R[0, 1]
    b01, b02, b12 = R[0, 1]+R[1, 0], R[0, 2]+R[2, 0], R[1, 2]+R[2, 1]
    q = wp.vec4d()
    if c0 >= c1 and c0 >= c2 and c0 >= c3:
        s = wp.sqrt(c0)
        q = wp.vec4d(s/two, a21/(two*s), a02/(two*s), a10/(two*s))
    elif c1 >= c2 and c1 >= c3:
        s = wp.sqrt(c1)
        q = wp.vec4d(a21/(two*s), s/two, b01/(two*s), b02/(two*s))
    elif c2 >= c3:
        s = wp.sqrt(c2)
        q = wp.vec4d(a02/(two*s), b01/(two*s), s/two, b12/(two*s))
    else:
        s = wp.sqrt(c3)
        q = wp.vec4d(a10/(two*s), b02/(two*s), b12/(two*s), s/two)
    return q


@wp.func
def quat_log(q: wp.vec4d):
    v = wp.vec3d(q[1], q[2], q[3]); w = q[0]
    s = wp.sqrt(wp.dot(v, v)+wp.float64(1e-30))
    ratio = wp.float64(0.0)
    if s < wp.float64(1e-6):
        ratio = wp.float64(2.0)/wp.max(w, wp.float64(1e-12))*(wp.float64(1.0)-s*s/wp.max(wp.float64(3.0)*w*w, wp.float64(1e-12)))
    else:
        ratio = wp.float64(2.0)*wp.atan2(s, w)/s
    return ratio*v


@wp.func
def cof(F: wp.mat33d):
    f1 = wp.vec3d(F[0, 0], F[1, 0], F[2, 0]); f2 = wp.vec3d(F[0, 1], F[1, 1], F[2, 1]); f3 = wp.vec3d(F[0, 2], F[1, 2], F[2, 2])
    c1, c2, c3 = wp.cross(f2, f3), wp.cross(f3, f1), wp.cross(f1, f2)
    return wp.mat33d(c1[0], c2[0], c3[0], c1[1], c2[1], c3[1], c1[2], c2[2], c3[2])


@wp.kernel
def k_S_bonds(x: wp.array(dtype=wp.vec3d), bi: wp.array(dtype=int), bj: wp.array(dtype=int), A: wp.array(dtype=wp.float64),
              nrm: wp.array(dtype=wp.vec3d), S: wp.array(dtype=wp.mat33d)):
    e = wp.tid()
    i, j = bi[e], bj[e]
    t = wp.outer(x[j]-x[i], nrm[e])*(A[e]*wp.float64(0.5))
    wp.atomic_add(S, i, t); wp.atomic_add(S, j, t)


@wp.kernel
def k_S_skin(x: wp.array(dtype=wp.vec3d), y: wp.array(dtype=wp.vec3d), pid: wp.array(dtype=int), body: wp.array(dtype=int),
             w: wp.array(dtype=wp.float64), l: wp.array(dtype=wp.vec3d), S: wp.array(dtype=wp.mat33d)):
    e = wp.tid()
    b = body[e]
    wp.atomic_add(S, b, wp.outer(y[pid[e]]-x[b], l[e])*w[e])


@wp.kernel
def k_F(S: wp.array(dtype=wp.mat33d), Mt_inv: wp.array(dtype=wp.mat33d), F: wp.array(dtype=wp.mat33d), R: wp.array(dtype=wp.mat33d),
        q: wp.array(dtype=wp.vec4d)):
    b = wp.tid()
    Fb = S[b]@Mt_inv[b]
    Rb = Fb
    for it in range(12):
        Rb = (Rb+wp.transpose(wp.inverse(Rb)))*wp.float64(0.5)
    F[b] = Fb; R[b] = Rb; q[b] = mat_quat(Rb)


@wp.func
def bond_quad(d: wp.vec3d, th: wp.vec3d, n: wp.vec3d, ke: wp.vec4d):
    dn, tn = wp.dot(d, n), wp.dot(th, n)
    dt, tt = d-dn*n, th-tn*n
    return wp.float64(0.5)*(ke[0]*dn*dn+ke[1]*wp.dot(dt, dt)+ke[2]*wp.dot(tt, tt)+ke[3]*tn*tn)


@wp.kernel
def k_bonds(x: wp.array(dtype=wp.vec3d), q: wp.array(dtype=wp.vec4d), bi: wp.array(dtype=int), bj: wp.array(dtype=int),
            ri: wp.array(dtype=wp.vec3d), rj: wp.array(dtype=wp.vec3d), n0: wp.array(dtype=wp.vec3d), k: wp.array(dtype=wp.vec4d),
            dref: wp.array(dtype=wp.vec3d), thref: wp.array(dtype=wp.vec3d), matp: wp.array(dtype=wp.float64),
            dp: wp.array(dtype=wp.vec3d), thp: wp.array(dtype=wp.vec3d),
            terms: wp.array2d(dtype=wp.float64), bd: wp.array(dtype=wp.vec3d), bth: wp.array(dtype=wp.vec3d)):
    """the bond's measures d (mid frame) and theta into bd, bth; 1/2 k . (m - m_p)^2 into row 0 (m_p the plastic
    references dp, thp: 0 elastic) and the viscous counterpart eta / mu 1/2 k . (m - m_ref)^2 into row 7"""
    e = wp.tid()
    i, j = bi[e], bj[e]
    qi, qj = q[i], q[j]
    Ri, Rj = quat_mat(qi), quat_mat(qj)
    D = x[j]+Rj@rj[e]-x[i]-Ri@ri[e]
    qr = quat_mul(wp.vec4d(qi[0], -qi[1], -qi[2], -qi[3]), qj)
    if qr[0] < wp.float64(0.0):
        qr = -qr
    th = quat_log(qr)
    h = qr+wp.vec4d(wp.float64(1.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
    h = h/wp.length(h)
    Rm = quat_mat(quat_mul(qi, h))
    d = wp.transpose(Rm)@D
    bd[e] = d
    bth[e] = th
    n = n0[e]
    ke = k[e]
    wp.atomic_add(terms, 0, e % NSLOT, bond_quad(d-dp[e], th-thp[e], n, ke))
    wp.atomic_add(terms, 7, e % NSLOT, matp[4]*bond_quad(d-dref[e], th-thref[e], n, ke))


@wp.kernel
def k_G(F: wp.array(dtype=wp.mat33d), bi: wp.array(dtype=int), bj: wp.array(dtype=int), wb: wp.array(dtype=wp.float64),
        dX: wp.array(dtype=wp.vec3d), G: wp.array2d(dtype=wp.float64)):
    """the curvature fit's right-hand side (gamma 'sym_skin'): wb A(dX)^T (F_j - F_i) row by row, s = (S00, S11, S22, S01,
    S02, S12), into both ends (n, 3 x 6)"""
    e = wp.tid()
    i, j = bi[e], bj[e]
    dF = (F[j]-F[i])*wb[e]
    v = dX[e]
    for a in range(3):
        r = wp.vec3d(dF[a, 0], dF[a, 1], dF[a, 2])
        vals = wp.vector(v[0]*r[0], v[1]*r[1], v[2]*r[2], v[1]*r[0]+v[0]*r[1], v[2]*r[0]+v[0]*r[2], v[2]*r[1]+v[1]*r[2], dtype=wp.float64)
        for m in range(6):
            wp.atomic_add(G, i, a*6+m, vals[m]); wp.atomic_add(G, j, a*6+m, vals[m])


@wp.kernel
def k_walls(x: wp.array(dtype=wp.vec3d), F: wp.array(dtype=wp.mat33d), R: wp.array(dtype=wp.mat33d), bi: wp.array(dtype=int),
            bj: wp.array(dtype=int), hi: wp.array(dtype=wp.float64), hj: wp.array(dtype=wp.float64), L: wp.array(dtype=wp.float64),
            wri: wp.array(dtype=wp.vec3d), wrj: wp.array(dtype=wp.vec3d), wt: wp.array(dtype=wp.vec3d), A: wp.array(dtype=wp.float64),
            nrm: wp.array(dtype=wp.vec3d), wi: wp.array(dtype=wp.float64), wj: wp.array(dtype=wp.float64), Vd: wp.array(dtype=wp.float64)):
    e = wp.tid()
    i, j = bi[e], bj[e]
    half = wp.float64(0.5)
    Ff, Rf = (F[i]+F[j])*half, (R[i]+R[j])*half
    ai, aj = hj[e]/L[e], hi[e]/L[e]
    line = ai*(x[i]+R[i]@wri[e])+aj*(x[j]+R[j]@wrj[e])+(Ff-Rf)@wt[e]
    comp = ai*(x[i]+F[i]@wri[e])+aj*(x[j]+F[j]@wrj[e])
    xf = (line+comp)*half
    a = cof(Ff)@(nrm[e]*A[e])
    third = wp.float64(1.0)/wp.float64(3.0)
    wp.atomic_add(Vd, i, wi[e]*wp.dot(xf-x[i], a)*third)
    wp.atomic_add(Vd, j, -wj[e]*wp.dot(xf-x[j], a)*third)


# hand-written adjoints of k_bonds and k_walls (recorded with tape.record_func; Warp's generated ones recompute the
# forward with much local memory)
@wp.func
def quat_conj(q: wp.vec4d):
    return wp.vec4d(q[0], -q[1], -q[2], -q[3])


@wp.func
def quat_mat_adj(q: wp.vec4d, G: wp.mat33d):
    """sum_kl G_kl d quat_mat(q)_kl / dq (q not normalised)"""
    w, x, y, z = q[0], q[1], q[2], q[3]
    two, four = wp.float64(2.0), wp.float64(4.0)
    gw = two*(-z*G[0, 1]+y*G[0, 2]+z*G[1, 0]-x*G[1, 2]-y*G[2, 0]+x*G[2, 1])
    gx = two*(y*G[0, 1]+z*G[0, 2]+y*G[1, 0]-w*G[1, 2]+z*G[2, 0]+w*G[2, 1])-four*x*(G[1, 1]+G[2, 2])
    gy = two*(x*G[0, 1]+w*G[0, 2]+x*G[1, 0]+z*G[1, 2]-w*G[2, 0]+z*G[2, 1])-four*y*(G[0, 0]+G[2, 2])
    gz = two*(-w*G[0, 1]+x*G[0, 2]+w*G[1, 0]+y*G[1, 2]+x*G[2, 0]+y*G[2, 1])-four*z*(G[0, 0]+G[1, 1])
    return wp.vec4d(gw, gx, gy, gz)


@wp.func
def quat_log_adj(q: wp.vec4d, ft: wp.vec3d):
    """ft . d quat_log(q) / dq, both branches of quat_log (clamps as Warp's max adjoint: to the first argument if larger)"""
    v = wp.vec3d(q[1], q[2], q[3])
    w = q[0]
    s = wp.sqrt(wp.dot(v, v)+wp.float64(1e-30))
    two = wp.float64(2.0)
    ratio = wp.float64(0.0)
    drs = wp.float64(0.0)
    drw = wp.float64(0.0)
    if s < wp.float64(1e-6):
        wa = wp.max(w, wp.float64(1e-12))
        dwa = wp.float64(0.0)
        if w > wp.float64(1e-12):
            dwa = wp.float64(1.0)
        w3 = wp.max(wp.float64(3.0)*w*w, wp.float64(1e-12))
        dw3 = wp.float64(0.0)
        if wp.float64(3.0)*w*w > wp.float64(1e-12):
            dw3 = wp.float64(6.0)*w
        ratio = two/wa*(wp.float64(1.0)-s*s/w3)
        drs = two/wa*(-two*s/w3)
        drw = -two/(wa*wa)*dwa*(wp.float64(1.0)-s*s/w3)+two/wa*(s*s/(w3*w3))*dw3
    else:
        ratio = two*wp.atan2(s, w)/s
        drs = (two*w/(s*s+w*w)-ratio)/s
        drw = -two/(s*s+w*w)
    vf = wp.dot(v, ft)
    gv = ratio*ft+(vf*drs/s)*v
    return wp.vec4d(vf*drw, gv[0], gv[1], gv[2])


@wp.kernel(enable_backward=False)
def k_bonds_adj(x: wp.array(dtype=wp.vec3d), q: wp.array(dtype=wp.vec4d), bi: wp.array(dtype=int), bj: wp.array(dtype=int),
                ri: wp.array(dtype=wp.vec3d), rj: wp.array(dtype=wp.vec3d), n0: wp.array(dtype=wp.vec3d), k: wp.array(dtype=wp.vec4d),
                dref: wp.array(dtype=wp.vec3d), thref: wp.array(dtype=wp.vec3d), matp: wp.array(dtype=wp.float64),
                dp: wp.array(dtype=wp.vec3d), thp: wp.array(dtype=wp.vec3d),
                terms_adj: wp.array2d(dtype=wp.float64), x_adj: wp.array(dtype=wp.vec3d), q_adj: wp.array(dtype=wp.vec4d)):
    """k_bonds' adjoint: dE/dx and dE/dq through D = x_j + R_j r_j - x_i - R_i r_i, d = R_m^T D, R_m = R(q_i h),
    h = (q_r + 1) / |q_r + 1|, theta = log(q_r), q_r = +-conj(q_i) q_j; the seeds dE/dd, dE/dtheta of rows 0 and 7"""
    e = wp.tid()
    i, j = bi[e], bj[e]
    qi, qj = q[i], q[j]
    Ri, Rj = quat_mat(qi), quat_mat(qj)
    D = x[j]+Rj@rj[e]-x[i]-Ri@ri[e]
    qr = quat_mul(quat_conj(qi), qj)
    sg = wp.float64(1.0)
    if qr[0] < wp.float64(0.0):
        qr = -qr
        sg = wp.float64(-1.0)
    th = quat_log(qr)
    hr = qr+wp.vec4d(wp.float64(1.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
    hl = wp.length(hr)
    h = hr/hl
    qm = quat_mul(qi, h)
    Rm = quat_mat(qm)
    d = wp.transpose(Rm)@D
    n = n0[e]
    de, te = d-dp[e], th-thp[e]                                                      # (the elastic part: less the plastic)
    dn, tn = wp.dot(de, n), wp.dot(te, n)
    dt, tt = de-dn*n, te-tn*n
    dv, tv = d-dref[e], th-thref[e]
    dvn, tvn = wp.dot(dv, n), wp.dot(tv, n)
    dvt, tvt = dv-dvn*n, tv-tvn*n
    ke = k[e]
    adj = terms_adj[0, e % NSLOT]
    av = terms_adj[7, e % NSLOT]*matp[4]
    fd = adj*(ke[0]*dn*n+ke[1]*dt)+av*(ke[0]*dvn*n+ke[1]*dvt)                        # dE/dd
    ft = adj*(ke[2]*tt+ke[3]*tn*n)+av*(ke[2]*tvt+ke[3]*tvn*n)                        # dE/dtheta
    gD = Rm@fd
    gqi = quat_mat_adj(qi, -wp.outer(gD, ri[e]))
    gqj = quat_mat_adj(qj, wp.outer(gD, rj[e]))
    gqm = quat_mat_adj(qm, wp.outer(D, fd))
    gqi = gqi+quat_mul(gqm, quat_conj(h))                                            # q_m = q_i h
    gh = quat_mul(quat_conj(qi), gqm)
    gqr = (gh-h*wp.dot(h, gh))/hl+quat_log_adj(qr, ft)
    gqr = gqr*sg
    gqj = gqj+quat_mul(qi, gqr)                                                      # q_r = conj(q_i) q_j
    gqi = gqi+quat_conj(quat_mul(gqr, quat_conj(qj)))
    wp.atomic_add(x_adj, i, -gD)
    wp.atomic_add(x_adj, j, gD)
    wp.atomic_add(q_adj, i, gqi)
    wp.atomic_add(q_adj, j, gqj)


@wp.func
def cof_adj(F: wp.mat33d, m: wp.vec3d, ga: wp.vec3d):
    """ga . d (cof(F) m) / dF (cof's columns f2 x f3, f3 x f1, f1 x f2)"""
    f1 = wp.vec3d(F[0, 0], F[1, 0], F[2, 0])
    f2 = wp.vec3d(F[0, 1], F[1, 1], F[2, 1])
    f3 = wp.vec3d(F[0, 2], F[1, 2], F[2, 2])
    g1 = m[1]*wp.cross(ga, f3)+m[2]*wp.cross(f2, ga)
    g2 = m[0]*wp.cross(f3, ga)+m[2]*wp.cross(ga, f1)
    g3 = m[0]*wp.cross(ga, f2)+m[1]*wp.cross(f1, ga)
    return wp.mat33d(g1[0], g2[0], g3[0], g1[1], g2[1], g3[1], g1[2], g2[2], g3[2])


@wp.kernel(enable_backward=False)
def k_walls_adj(x: wp.array(dtype=wp.vec3d), F: wp.array(dtype=wp.mat33d), R: wp.array(dtype=wp.mat33d), bi: wp.array(dtype=int),
                bj: wp.array(dtype=int), hi: wp.array(dtype=wp.float64), hj: wp.array(dtype=wp.float64), L: wp.array(dtype=wp.float64),
                wri: wp.array(dtype=wp.vec3d), wrj: wp.array(dtype=wp.vec3d), wt: wp.array(dtype=wp.vec3d), A: wp.array(dtype=wp.float64),
                nrm: wp.array(dtype=wp.vec3d), wi: wp.array(dtype=wp.float64), wj: wp.array(dtype=wp.float64), Vd_adj: wp.array(dtype=wp.float64),
                x_adj: wp.array(dtype=wp.vec3d), F_adj: wp.array(dtype=wp.mat33d), R_adj: wp.array(dtype=wp.mat33d)):
    """k_walls' adjoint: Vd_i += w_i (x_f - x_i) . a / 3, Vd_j -= w_j (x_f - x_j) . a / 3, a = cof(F_f) A n,
    x_f = (line + comp) / 2"""
    e = wp.tid()
    i, j = bi[e], bj[e]
    half = wp.float64(0.5)
    Ff, Rf = (F[i]+F[j])*half, (R[i]+R[j])*half
    ai, aj = hj[e]/L[e], hi[e]/L[e]
    line = ai*(x[i]+R[i]@wri[e])+aj*(x[j]+R[j]@wrj[e])+(Ff-Rf)@wt[e]
    comp = ai*(x[i]+F[i]@wri[e])+aj*(x[j]+F[j]@wrj[e])
    xf = (line+comp)*half
    m = nrm[e]*A[e]
    a = cof(Ff)@m
    third = wp.float64(1.0)/wp.float64(3.0)
    ci = wi[e]*Vd_adj[i]*third
    cj = -wj[e]*Vd_adj[j]*third
    gxf = (ci+cj)*a
    gFf = cof_adj(Ff, m, ci*(xf-x[i])+cj*(xf-x[j]))
    q4 = wp.float64(0.25)
    gt = q4*wp.outer(gxf, wt[e])
    gwi = (ai*half)*wp.outer(gxf, wri[e])
    gwj = (aj*half)*wp.outer(gxf, wrj[e])
    wp.atomic_add(x_adj, i, ai*gxf-ci*a)
    wp.atomic_add(x_adj, j, aj*gxf-cj*a)
    wp.atomic_add(F_adj, i, gwi+gt+half*gFf)
    wp.atomic_add(F_adj, j, gwj+gt+half*gFf)
    wp.atomic_add(R_adj, i, gwi-gt)
    wp.atomic_add(R_adj, j, gwj-gt)


@wp.kernel
def k_sg(G: wp.array2d(dtype=wp.float64), Ninv: wp.array2d(dtype=wp.float64), sg: wp.array2d(dtype=wp.float64)):
    """the curvature fit per body: s = N_B^-1 rhs_B, sg[b, a 6 + m] (row a of Gamma, component m)"""
    b = wp.tid()
    for a in range(3):
        for m in range(6):
            sm = wp.float64(0.0)
            for k in range(6):
                sm = sm+Ninv[b, m*6+k]*G[b, a*6+k]
            sg[b, a*6+m] = sm


@wp.kernel
def k_entries(x: wp.array(dtype=wp.vec3d), y: wp.array(dtype=wp.vec3d), F: wp.array(dtype=wp.mat33d), R: wp.array(dtype=wp.mat33d),
              sg: wp.array2d(dtype=wp.float64), q6: wp.array2d(dtype=wp.float64),
              pid: wp.array(dtype=int), body: wp.array(dtype=int),
              d: wp.array(dtype=wp.vec3d), c: wp.array(dtype=wp.vec3d), matp: wp.array(dtype=wp.float64), aref: wp.array(dtype=wp.vec3d),
              Pref: wp.array(dtype=wp.mat33d), Pp: wp.array(dtype=wp.mat33d), dl: wp.array(dtype=wp.vec3d), Vd: wp.array(dtype=wp.float64),
              terms: wp.array2d(dtype=wp.float64), ba: wp.array(dtype=wp.vec3d)):
    """o2 = 2 mu daff . (F - R P_p) c = 2 mu (R^T daff) . (R^T F - P_p) c (P_p the cell's plastic stretch: I elastic);
    its viscous counterpart 2 eta (a - a_ref) . (R^T F - P_ref) c with a = R^T daff (into ba) into row 7"""
    e = wp.tid()
    b = body[e]
    Fb = F[b]
    Rt = wp.transpose(R[b])
    de = d[e]
    daff = y[pid[e]]-x[b]-Fb@de
    curv = wp.vec3d()
    for a in range(3):
        acc = wp.float64(0.0)
        for m in range(6):
            acc = acc+sg[b, a*6+m]*q6[e, m]
        curv[a] = acc
    delta = daff-curv*wp.float64(0.5)
    dl[e] = Rt@delta
    ce = c[e]
    wp.atomic_add(terms, 3, e % NSLOT, wp.float64(2.0)*matp[0]*wp.dot(daff, (Fb-R[b]@Pp[b])@ce))
    av = Rt@daff
    ba[e] = av
    wp.atomic_add(terms, 7, e % NSLOT, wp.float64(2.0)*matp[1]*wp.dot(av-aref[e], (Rt@Fb-Pref[b])@ce))
    wp.atomic_add(Vd, b, wp.dot(daff, cof(Fb)@ce)/wp.float64(3.0))


@wp.kernel(enable_backward=False)
def k_rows(dl: wp.array(dtype=wp.vec3d), dlref: wp.array(dtype=wp.vec3d), e_part: wp.array(dtype=int), off: wp.array(dtype=int),
           poff: wp.array(dtype=int), n3: wp.array(dtype=wp.float64), NL: wp.array(dtype=wp.mat33d), matp: wp.array(dtype=wp.float64),
           NLv: wp.array(dtype=wp.mat33d), vsame: int, g3: wp.array(dtype=wp.vec3d), gL: wp.array(dtype=wp.vec3d),
           gv: wp.array(dtype=wp.vec3d), terms: wp.array2d(dtype=wp.float64)):
    """o3 and the lateral term, both quadratic in the turned deviations: sum over a part's ordered pairs dl_a . M_ab dl_b.
    Per entry a the row sums g_a = sum_b N_ab dl_b (N_ab = M_ab + M_ba^T, the pairs of a part row by row from poff), the
    energy 1/2 dl_a . g_a; g_a is its gradient (k_rows_adj). The viscous counterpart on dl - dl_ref (o3 times eta / mu,
    the lateral term's NLv) into row 7, its row sums gv. N_ab is read as N_ba^T (n3 and N are symmetric that way, to the
    bit: sums of the same two terms), so that the threads of a part read consecutive pairs (10-24: the reads were s apart).
    vsame 1: NLv = (eta / mu) NL (one viscosity ratio for the shear and the lateral kappa, SkinWarp.set_viscosity), NLv
    not read"""
    a = wp.tid()
    smu = matp[4]
    k = e_part[a]
    o = off[k]
    s = off[k+1]-o
    p = poff[k]+(a-o)
    a3 = wp.vec3d()
    aL = wp.vec3d()
    av = wp.vec3d()
    for j in range(s):
        q = p+j*s
        db = dl[o+j]
        dv = db-dlref[o+j]
        c3 = n3[q]
        Nt = wp.transpose(NL[q])
        a3 = a3+c3*db
        aL = aL+Nt@db
        if vsame == 1:
            av = av+smu*(c3*dv+Nt@dv)
        else:
            av = av+(smu*c3)*dv+wp.transpose(NLv[q])@dv
    g3[a] = a3
    gL[a] = aL
    gv[a] = av
    da = dl[a]
    wp.atomic_add(terms, 4, a % NSLOT, wp.float64(0.5)*wp.dot(da, a3))
    wp.atomic_add(terms, 5, a % NSLOT, wp.float64(0.5)*wp.dot(da, aL))
    wp.atomic_add(terms, 7, a % NSLOT, wp.float64(0.5)*wp.dot(da-dlref[a], av))


@wp.kernel(enable_backward=False)
def k_rows_adj(g3: wp.array(dtype=wp.vec3d), gL: wp.array(dtype=wp.vec3d), gv: wp.array(dtype=wp.vec3d), terms_adj: wp.array2d(dtype=wp.float64),
               dl_adj: wp.array(dtype=wp.vec3d)):
    a = wp.tid()
    dl_adj[a] = dl_adj[a]+terms_adj[4, a % NSLOT]*g3[a]+terms_adj[5, a % NSLOT]*gL[a]+terms_adj[7, a % NSLOT]*gv[a]


@wp.kernel
def k_patches(F: wp.array(dtype=wp.mat33d), R: wp.array(dtype=wp.mat33d), cb: wp.array(dtype=int), cr: wp.array(dtype=wp.float64),
              N: wp.array(dtype=wp.mat33d), matp: wp.array(dtype=wp.float64), Pref: wp.array(dtype=wp.mat33d), Pp: wp.array(dtype=wp.mat33d),
              terms: wp.array2d(dtype=wp.float64)):
    """o1 = mu r tr(D^T D N), D = F - R P_p (D^T D = (R^T F - P_p)^2, P_p the cell's plastic stretch: I elastic); the
    viscous counterpart on R^T F - P_ref into row 7"""
    p = wp.tid()
    b = cb[p]
    D = F[b]-R[b]@Pp[b]
    wp.atomic_add(terms, 2, p % NSLOT, matp[0]*cr[p]*wp.trace(wp.transpose(D)@D@N[p]))
    Dv = wp.transpose(R[b])@F[b]-Pref[b]
    wp.atomic_add(terms, 7, p % NSLOT, matp[1]*cr[p]*wp.trace(wp.transpose(Dv)@Dv@N[p]))


@wp.kernel
def k_volume(F: wp.array(dtype=wp.mat33d), Vd: wp.array(dtype=wp.float64), Vdet: wp.array(dtype=wp.float64), V: wp.array(dtype=wp.float64),
             matp: wp.array(dtype=wp.float64), Lref: wp.array(dtype=wp.float64), barrier: wp.float64, lj: wp.array(dtype=wp.float64),
             terms: wp.array2d(dtype=wp.float64)):
    """V lam / 2 L^2 (L = log J with the barrier); the viscous counterpart V lam_v / 2 (L - L_ref)^2 into row 7"""
    b = wp.tid()
    J = (Vd[b]+wp.determinant(F[b])*Vdet[b])/V[b]
    L = wp.float64(0.0)
    if J < barrier:
        L = wp.log(barrier)+(J-barrier)/barrier-(J-barrier)*(J-barrier)/(wp.float64(2.0)*barrier*barrier)
    else:
        L = wp.log(J)
    lj[b] = L
    wp.atomic_add(terms, 1, b % NSLOT, V[b]*wp.float64(0.5)*matp[2]*L*L)
    Lv = L-Lref[b]
    wp.atomic_add(terms, 7, b % NSLOT, V[b]*wp.float64(0.5)*matp[3]*Lv*Lv)


@wp.kernel
def k_variance(lj: wp.array(dtype=wp.float64), Lref: wp.array(dtype=wp.float64), bi: wp.array(dtype=int), bj: wp.array(dtype=int),
               wv: wp.array(dtype=wp.float64), matp: wp.array(dtype=wp.float64), terms: wp.array2d(dtype=wp.float64)):
    e = wp.tid()
    i, j = bi[e], bj[e]
    dlj = lj[i]-lj[j]
    wp.atomic_add(terms, 1, e % NSLOT, wv[e]*dlj*dlj)
    dv = dlj-(Lref[i]-Lref[j])
    wp.atomic_add(terms, 7, e % NSLOT, matp[5]*wv[e]*dv*dv)


@wp.kernel(enable_backward=False)
def k_total(terms: wp.array2d(dtype=wp.float64), tsum: wp.array(dtype=wp.float64)):
    """the slots summed per term (one thread per slot); then k_total_loss; adjoint k_total_adj"""
    t, k = wp.tid()
    wp.atomic_add(tsum, t, terms[t, k])


@wp.kernel(enable_backward=False)
def k_total_loss(tsum: wp.array(dtype=wp.float64), tw: wp.array(dtype=wp.float64), loss: wp.array(dtype=wp.float64)):
    """the loss sum_t tw_t tsum_t (tw: SkinWarp.ttw, the terms' weights)"""
    s = wp.float64(0.0)
    for t in range(NTC):
        s = s+tw[t]*tsum[t]
    loss[0] = s


@wp.kernel(enable_backward=False)
def k_total_adj(tsum_adj: wp.array(dtype=wp.float64), loss_adj: wp.array(dtype=wp.float64), tw: wp.array(dtype=wp.float64),
                terms_adj: wp.array2d(dtype=wp.float64)):
    t, k = wp.tid()
    terms_adj[t, k] = terms_adj[t, k]+tsum_adj[t]+tw[t]*loss_adj[0]


@wp.kernel(enable_backward=False)
def k_prec_turn(Bin: wp.array(dtype=wp.mat33d), R: wp.array(dtype=wp.mat33d), owner: wp.array(dtype=int), gofs: wp.array(dtype=int),
                bofs: wp.array(dtype=int), grp: wp.array(dtype=int), Bt: wp.array(dtype=wp.mat33d)):
    """the group blocks turned by the owner body's rotation, R B_ab R^T, stored transposed at (b, a) (k_prec_apply's
    threads a then read neighbouring blocks); one thread per (group, point a)"""
    t = wp.tid()
    g = grp[t]
    o = gofs[g]
    s = gofs[g+1]-o
    a = t-o
    Rg = R[owner[g]]
    for b in range(s):
        Bt[bofs[g]+b*s+a] = wp.transpose(Rg@Bin[bofs[g]+a*s+b]@wp.transpose(Rg))


@wp.kernel(enable_backward=False)
def k_prec_apply(Bt: wp.array(dtype=wp.mat33d), r: wp.array(dtype=wp.vec3d), pts: wp.array(dtype=int), gofs: wp.array(dtype=int),
                 bofs: wp.array(dtype=int), grp: wp.array(dtype=int), z: wp.array(dtype=wp.vec3d)):
    """z += the group blocks times r (additive): row a of group g into its point"""
    t = wp.tid()
    g = grp[t]
    o = gofs[g]
    s = gofs[g+1]-o
    a = t-o
    acc = wp.vec3d()
    for b in range(s):
        acc = acc+wp.transpose(Bt[bofs[g]+b*s+a])@r[pts[o+b]]
    wp.atomic_add(z, pts[t], acc)


# the contact's penalty in the preconditioner (Sherman-Morrison on the contact cell's body group): a = om n (the gap's
# derivative per skin point of the group), u = B^-1 a, a.u; with coefficient rho, B^-1 <- B^-1 - rho u u^T / (1 + rho a.u)
@wp.kernel(enable_backward=False)
def k_sm_u(Bt: wp.array(dtype=wp.mat33d), cgrp: wp.array(dtype=int), gofs: wp.array(dtype=int), bofs: wp.array(dtype=int),
           om: wp.array(dtype=wp.float64), ng: wp.vec3d, Smax: int, U: wp.array(dtype=wp.vec3d)):
    c, a = wp.tid()
    G = cgrp[c]
    o = gofs[G]
    s = gofs[G+1]-o
    if a >= s:
        return
    u = wp.vec3d()
    for b in range(s):
        u = u+om[c*Smax+b]*(wp.transpose(Bt[bofs[G]+b*s+a])@ng)                   # block (a, b) is stored transposed at (b, a)
    U[c*Smax+a] = u


@wp.kernel(enable_backward=False)
def k_sm_coef(cgrp: wp.array(dtype=int), gofs: wp.array(dtype=int), om: wp.array(dtype=wp.float64), U: wp.array(dtype=wp.vec3d),
              ng: wp.vec3d, Smax: int, rho: wp.array(dtype=wp.float64), aua: wp.array(dtype=wp.float64), coef: wp.array(dtype=wp.float64)):
    c = wp.tid()
    G = cgrp[c]
    s = gofs[G+1]-gofs[G]
    t = wp.float64(0.0)
    for a in range(s):
        t = t+om[c*Smax+a]*wp.dot(ng, U[c*Smax+a])
    aua[c] = t
    r = rho[c]
    cf = wp.float64(0.0)
    if r > wp.float64(0.0):
        cf = r/(wp.float64(1.0)+r*t)
    coef[c] = cf


@wp.kernel(enable_backward=False)
def k_sm_update(Bt: wp.array(dtype=wp.mat33d), cgrp: wp.array(dtype=int), gofs: wp.array(dtype=int), bofs: wp.array(dtype=int),
                U: wp.array(dtype=wp.vec3d), coef: wp.array(dtype=wp.float64), Smax: int):
    c, a = wp.tid()
    cf = coef[c]
    if cf == wp.float64(0.0):
        return
    G = cgrp[c]
    o = gofs[G]
    s = gofs[G+1]-o
    if a >= s:
        return
    ua = U[c*Smax+a]
    for b in range(s):
        k = bofs[G]+b*s+a
        Bt[k] = Bt[k]-cf*wp.outer(U[c*Smax+b], ua)


class WarpPrec:
    """skin_solve.SkinSolver.preconditioner on Warp: the inverted group blocks stored packed (each group's s x s blocks of
    3 x 3, row by row, no padding), turned once per call (Rb) -> f (rv (k, 3 npt) -> same, torch), applied by one
    thread per (group, point); apply(r, z) adds the blocks times r to z (Warp arrays, for graphs)"""

    def __init__(self, B, g, npt):
        G, S, Lc, Lm, owner = g['G'], g['S'], g['Lc'], g['Lm'], g['owner']
        Binv = torch.linalg.inv(.5*(B+B.transpose(1, 2))).view(G, S, 3, S, 3).permute(0, 1, 3, 2, 4)      # (G, S, S, 3, 3)
        valid = Lm > 0
        s = valid.sum(1)
        gofs = torch.cat((s.new_zeros(1), s.cumsum(0))); bofs = torch.cat((s.new_zeros(1), (s*s).cumsum(0)))
        Bp = Binv[valid[:, :, None] & valid[:, None, :]].contiguous()                                     # g, a, b order
        pts = Lc[valid]; grp = torch.repeat_interleave(torch.arange(G, device=s.device), s)
        I = lambda t: wp.from_torch(t.to(torch.int32).contiguous(), dtype=wp.int32)
        self.Bin, self.Bt = wp.from_torch(Bp, dtype=wp.mat33d), wp.zeros(len(Bp), dtype=wp.mat33d, device='cuda')
        self.owner, self.gofs, self.bofs, self.grp, self.pts = I(owner), I(gofs), I(bofs), I(grp), I(pts)
        self.nt, self.npt = len(pts), npt
        self.ck = 0

    def set_contact(self, cgrp, Smax, normal):
        """contact cells' groups cgrp (k,), the groups' padded size Smax, the plane's normal"""
        k = len(cgrp)
        self.ck, self.Smax, self.ng = k, int(Smax), wp.vec3d(*[float(a) for a in normal])
        self.cgrp = wp.from_torch(cgrp.to(torch.int32).contiguous(), dtype=wp.int32)
        z = lambda n_, dt: wp.zeros(n_, dtype=dt, device='cuda')
        self.om, self.U, self.rho_act, self.aua, self.coef = z(k*self.Smax, wp.float64), z(k*self.Smax, wp.vec3d), z(k, wp.float64), z(k, wp.float64), z(k, wp.float64)
        self.aua_t, self.coef_t = z(k, wp.float64), z(k, wp.float64)                # (the friction's directions: taua kept)
        tt = lambda a: wp.to_torch(a, requires_grad=False)
        self.tom, self.trho_act, self.taua = tt(self.om), tt(self.rho_act), tt(self.aua)

    def contact_update(self, om, rho_act, direction=None):
        """after a turn: the penalty rho_act[c] (a a^T, a = om[c] n per slot of cell c's group) added to the blocks by
        Sherman-Morrison (rho_act 0: none; taua gets a.B^-1 a either way). direction: a = om[c] direction instead of n
        (the friction's tangents: the slip's derivative per slot is the gap's weight times the direction), taua kept"""
        self.tom.copy_(om.reshape(-1)); self.trho_act.copy_(rho_act)
        k, S_ = self.ck, self.Smax
        ng = self.ng if direction is None else wp.vec3d(*[float(a) for a in direction])
        aua, coef = (self.aua, self.coef) if direction is None else (self.aua_t, self.coef_t)
        wp.launch(k_sm_u, (k, S_), inputs=[self.Bt, self.cgrp, self.gofs, self.bofs, self.om, ng, S_], outputs=[self.U])
        wp.launch(k_sm_coef, k, inputs=[self.cgrp, self.gofs, self.om, self.U, ng, S_, self.rho_act], outputs=[aua, coef])
        wp.launch(k_sm_update, (k, S_), inputs=[self.Bt, self.cgrp, self.gofs, self.bofs, self.U, coef, S_])     # Bt in place

    def apply(self, r, z):
        wp.launch(k_prec_apply, self.nt, inputs=[self.Bt, r, self.pts, self.gofs, self.bofs, self.grp], outputs=[z])

    def __call__(self, Rb):
        wp.launch(k_prec_turn, self.nt, inputs=[self.Bin, wp.from_torch(Rb.contiguous(), dtype=wp.mat33d), self.owner, self.gofs, self.bofs,
                                                self.grp], outputs=[self.Bt])
        npt = self.npt
        def f(rv):
            out = torch.zeros(rv.shape[0], npt, 3, dtype=f64, device=rv.device)
            for k in range(rv.shape[0]):
                self.apply(wp.from_torch(rv[k].view(npt, 3).contiguous(), dtype=wp.vec3d), wp.from_torch(out[k], dtype=wp.vec3d))
            return out.reshape(rv.shape[0], -1)
        f.wprec = self
        return f


def warp_preconditioner(B, g, npt):
    return WarpPrec(B, g, npt)


# the inner CG iteration of SkinSolver.steihaug as one graph (WarpCG). Scalars sc: 0 s = max |d|, 1 d.Hd, 3 r.r, 4 r.z
# (new), 5 r.z (current); partial sums red (4, NSLOT) of max |d|, d.Hd, r.r, r.z, then one thread per slot into sc
@wp.kernel(enable_backward=False)
def k_cg_reset(sc: wp.array(dtype=wp.float64)):
    sc[0] = wp.float64(0.0)
    sc[1] = wp.float64(0.0)
    sc[3] = wp.float64(0.0)
    sc[4] = wp.float64(0.0)


@wp.kernel(enable_backward=False)
def k_cg_absmax(d: wp.array(dtype=wp.vec3d), red: wp.array2d(dtype=wp.float64)):
    i = wp.tid()
    v = d[i]
    wp.atomic_max(red, 0, i % NSLOT, wp.max(wp.max(wp.abs(v[0]), wp.abs(v[1])), wp.abs(v[2])))


@wp.kernel(enable_backward=False)
def k_cg_gather(red: wp.array2d(dtype=wp.float64), r0: int, sc: wp.array(dtype=wp.float64)):
    """slot k of partial row r0 + j into sc: row 0 by max into sc[0], rows 1, 2, 3 summed into sc[1], sc[3], sc[4]"""
    j, k = wp.tid()
    row = r0+j
    v = red[row, k]
    if row == 0:
        wp.atomic_max(sc, 0, v)
    elif row == 1:
        wp.atomic_add(sc, 1, v)
    else:
        wp.atomic_add(sc, row+1, v)


@wp.kernel(enable_backward=False)
def k_cg_perturb(y: wp.array(dtype=wp.vec3d), d: wp.array(dtype=wp.vec3d), sc: wp.array(dtype=wp.float64), h: wp.float64,
                 yp: wp.array(dtype=wp.vec3d)):
    i = wp.tid()
    yp[i] = y[i]+(h/wp.max(sc[0], wp.float64(1e-300)))*d[i]


@wp.kernel(enable_backward=False)
def k_cg_hd(gy: wp.array(dtype=wp.vec3d), g: wp.array(dtype=wp.vec3d), d: wp.array(dtype=wp.vec3d), sc: wp.array(dtype=wp.float64),
            h: wp.float64, Hd: wp.array(dtype=wp.vec3d), red: wp.array2d(dtype=wp.float64)):
    i = wp.tid()
    v = (gy[i]-g[i])*(wp.max(sc[0], wp.float64(1e-300))/h)
    Hd[i] = v
    wp.atomic_add(red, 1, i % NSLOT, wp.dot(d[i], v))


@wp.kernel(enable_backward=False)
def k_cg_step(d: wp.array(dtype=wp.vec3d), Hd: wp.array(dtype=wp.vec3d), sc: wp.array(dtype=wp.float64), p: wp.array(dtype=wp.vec3d),
              Hp: wp.array(dtype=wp.vec3d), r: wp.array(dtype=wp.vec3d)):
    """alpha = r.z / d.Hd (0 if d.Hd <= 0: the host then leaves along d_old)"""
    i = wp.tid()
    al = wp.float64(0.0)
    if sc[1] > wp.float64(0.0):
        al = sc[5]/sc[1]
    p[i] = p[i]+al*d[i]
    Hp[i] = Hp[i]+al*Hd[i]
    r[i] = r[i]-al*Hd[i]


@wp.kernel(enable_backward=False)
def k_cg_dots(r: wp.array(dtype=wp.vec3d), z: wp.array(dtype=wp.vec3d), red: wp.array2d(dtype=wp.float64)):
    i = wp.tid()
    ri = r[i]
    wp.atomic_add(red, 2, i % NSLOT, wp.dot(ri, ri))
    wp.atomic_add(red, 3, i % NSLOT, wp.dot(ri, z[i]))


@wp.kernel(enable_backward=False)
def k_cg_dir(z: wp.array(dtype=wp.vec3d), sc: wp.array(dtype=wp.float64), d: wp.array(dtype=wp.vec3d), d_old: wp.array(dtype=wp.vec3d)):
    """beta = r.z (new) / r.z (current)"""
    i = wp.tid()
    be = wp.float64(0.0)
    if sc[5] != wp.float64(0.0):
        be = sc[4]/sc[5]
    d_old[i] = d[i]
    d[i] = z[i]+be*d[i]


@wp.kernel(enable_backward=False)
def k_cg_shift(sc: wp.array(dtype=wp.float64)):
    sc[5] = sc[4]


class WarpCG:
    """one iteration of SkinSolver.steihaug's CG as a CUDA graph, speculatively (as if neither negative curvature nor the
    region's edge stopped it): s = max |d|, Hd by one-sided differences of the gradient (y + h d / s), alpha, p, Hp, r,
    z = prec(r), r.r, r.z, beta, d_old = d, d = z + beta d. step() returns d.Hd, r.r, r.z (one sync); the host forms
    alpha and beta from them by the same divisions and decides; at an exit it steps back along d_old. Buffers t (torch
    views): y, g (gradient there), d, d_old, Hd, p, Hp, r, z; tsc[5] holds the current r.z"""

    def __init__(self, W, wprec, h):
        n = W.npt
        self.a = {k: wp.zeros(n, dtype=wp.vec3d, device='cuda') for k in ('y', 'g', 'd', 'd_old', 'Hd', 'p', 'Hp', 'r', 'z')}
        self.t = {k: wp.to_torch(v, requires_grad=False) for k, v in self.a.items()}
        self.red = wp.zeros((4, 256), dtype=wp.float64, device='cuda')
        self.sc = wp.zeros(8, dtype=wp.float64, device='cuda'); self.tsc = wp.to_torch(self.sc, requires_grad=False)
        self.hsc = wp.zeros(8, dtype=wp.float64, device='cpu', pinned=True); self.hv = self.hsc.numpy()      # read back in the graph
        self.W, self.wprec, self.h = W, wprec, h
        self._iteration()                                                            # load the kernels outside the capture
        wp.capture_begin()
        self._iteration()
        self.graph = wp.capture_end()

    def _iteration(self):
        a, W, h, n = self.a, self.W, self.h, self.W.npt
        wp.launch(k_cg_reset, 1, inputs=[], outputs=[self.sc])
        self.red.zero_()
        wp.launch(k_cg_absmax, n, inputs=[a['d']], outputs=[self.red])
        wp.launch(k_cg_gather, (1, 256), inputs=[self.red, 0], outputs=[self.sc])
        wp.launch(k_cg_perturb, n, inputs=[a['y'], a['d'], self.sc, h], outputs=[W.yw])
        W._run_products()
        wp.launch(k_cg_hd, n, inputs=[W.yw.grad, a['g'], a['d'], self.sc, h], outputs=[a['Hd'], self.red])
        wp.launch(k_cg_gather, (1, 256), inputs=[self.red, 1], outputs=[self.sc])
        wp.launch(k_cg_step, n, inputs=[a['d'], a['Hd'], self.sc], outputs=[a['p'], a['Hp'], a['r']])
        a['z'].zero_()
        self.wprec.apply(a['r'], a['z'])
        wp.launch(k_cg_dots, n, inputs=[a['r'], a['z']], outputs=[self.red])
        wp.launch(k_cg_gather, (2, 256), inputs=[self.red, 2], outputs=[self.sc])
        wp.launch(k_cg_dir, n, inputs=[a['z'], self.sc], outputs=[a['d'], a['d_old']])
        wp.launch(k_cg_shift, 1, inputs=[], outputs=[self.sc])
        wp.copy(self.hsc, self.sc)

    def step(self):
        wp.capture_launch(self.graph)
        wp.synchronize_stream()
        hv = self.hv
        return float(hv[1]), float(hv[3]), float(hv[4])


# contact with a plane n . y >= c0 (10-23): each surface cell's exposed sphere (site p, radius r) carried by its affine
# field y = x_B + F_B (X - X_B); its lowest point is the minimum over the patch of n . F_B (X - X_B) (linear in X):
# the sphere's lowest point X = p - r c / |c| (c = F_B^T n) if exposed (outside every neighbour's sphere), else on an
# exposed boundary arc X(t) = cc + rho (cos t u + sin t v), t0 <= t <= t1 (its interior minimum or an end). One condition
# per cell, gap g = n . x_B + c . (X* - X_B) - c0 >= 0, by the augmented Lagrangian term
# Phi = (max(0, lam - rho g)^2 - lam^2) / (2 rho) (multiplier lam, numerical coefficient rho); dg/dx_B = n,
# dg/dF_B = n (x) (X* - X_B) (envelope: a minimum over material points)
@wp.kernel(enable_backward=False)
def k_contact(x: wp.array(dtype=wp.vec3d), y: wp.array(dtype=wp.vec3d), F: wp.array(dtype=wp.mat33d), X: wp.array(dtype=wp.vec3d),
              shift: int, ceoff: wp.array(dtype=int), cepid: wp.array(dtype=int), cew: wp.array(dtype=wp.float64), dbar: wp.array(dtype=wp.vec3d),
              cbody: wp.array(dtype=int),
              cp: wp.array(dtype=wp.vec3d), cr: wp.array(dtype=wp.float64), aoff: wp.array(dtype=int), acc: wp.array(dtype=wp.vec3d),
              arho: wp.array(dtype=wp.float64), au: wp.array(dtype=wp.vec3d), av: wp.array(dtype=wp.vec3d), at0: wp.array(dtype=wp.float64),
              at1: wp.array(dtype=wp.float64), noff: wp.array(dtype=int), nbp: wp.array(dtype=wp.vec3d), nbr: wp.array(dtype=wp.float64),
              ng: wp.vec3d, c0: wp.array(dtype=wp.float64), lam: wp.array(dtype=wp.float64), rho: wp.array(dtype=wp.float64), energy: int,
              cpar: wp.array(dtype=wp.float64), goff: wp.array(dtype=wp.float64), act: wp.array(dtype=int), only: wp.array(dtype=int),
              gap: wp.array(dtype=wp.float64), Xs: wp.array(dtype=wp.vec3d), cpos: wp.array(dtype=wp.vec3d), dphi: wp.array(dtype=wp.float64),
              terms: wp.array2d(dtype=wp.float64)):
    """the condition is cs g - goff >= 0 (written to gap), the term wt Phi (cpar = cs, wt, mode; 1, 1 and goff 0: g
    itself; the midpoint rule, joint_step: cs 2, wt 1/2, goff the last step's end gap, i.e. the gap at the step's end
    2 g_mid - g_n). mode 1: the active set held (act): lam - rho g unclamped on it, nothing off it (a smooth term within
    a solve; JointSolver.solve_contact moves the set between solves). only[0] 1 (the CG products, SkinWarp._run_products)
    in mode 1: the cells off the set are skipped (their force is 0 and their lam 0, so their term too; their gap and
    X* are not written, only the full pass's are read). (10-24: packing the set into the first warps gained nothing, the
    time is the set's own cells: per arc float64 atan2 / cos / sin)"""
    c = wp.tid()
    if only[0] == 1 and cpar[2] > wp.float64(0.5) and act[c] == 0:
        dphi[c] = wp.float64(0.0)
        return
    B = cbody[c]
    cv = wp.transpose(F[B])@ng
    p = cp[c]
    r = cr[c]
    best = wp.float64(1.0e30)
    bx = p
    cl = wp.length(cv)
    if cl > wp.float64(0.0):
        Xc = p-(r/cl)*cv
        ok = int(1)
        for k in range(noff[c], noff[c+1]):
            d = Xc-nbp[k]
            if wp.dot(d, d)-nbr[k]*nbr[k] <= wp.float64(0.0):
                ok = int(0)
        if ok == 1:
            best = wp.dot(cv, Xc-p)
            bx = Xc
    pi = wp.float64(3.141592653589793)
    for k in range(aoff[c], aoff[c+1]):
        t0, t1 = at0[k], at1[k]
        ts = wp.atan2(-wp.dot(cv, av[k]), -wp.dot(cv, au[k]))                        # the circle's lowest angle
        tm = wp.float64(0.5)*(t0+t1)
        dd = ts-tm
        dd = dd-wp.float64(2.0)*pi*wp.floor((dd+pi)/(wp.float64(2.0)*pi))
        for e in range(3):
            t = t0
            if e == 1:
                t = t1
            if e == 2:
                t = tm+dd
            if t >= t0 and t <= t1:
                Xa = acc[k]+arho[k]*(wp.cos(t)*au[k]+wp.sin(t)*av[k])
                val = wp.dot(cv, Xa-p)
                if val < best:
                    best = val
                    bx = Xa
    XB = X[B]
    w = bx-XB
    g = wp.float64(0.0)
    if shift == 1:                                                                   # carried by the skin's mean deviation:
        w = w-dbar[c]                                                                # x_B drops out (the weights sum to 1)
        pm = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
        for k in range(ceoff[c], ceoff[c+1]):
            pm = pm+cew[k]*y[cepid[k]]
        cpos[c] = F[B]@w+pm                                                          # the lowest point's position
        g = wp.dot(cv, w)+wp.dot(ng, pm)-c0[0]
    else:
        cpos[c] = x[B]+F[B]@w
        g = wp.dot(ng, x[B])+wp.dot(cv, w)-c0[0]
    cs, wt = cpar[0], cpar[1]
    g = cs*g-goff[c]
    lm = lam[c]
    rh = rho[c]
    mx = wp.max(lm-rh*g, wp.float64(0.0))
    if cpar[2] > wp.float64(0.5):
        mx = wp.float64(0.0)
        if act[c] == 1:
            mx = lm-rh*g
    gap[c] = g
    Xs[c] = w
    if energy == 1:
        dphi[c] = -mx*cs*wt
        wp.atomic_add(terms, 6, c % NSLOT, wt*(mx*mx-lm*lm)/(wp.float64(2.0)*rh))
    else:
        dphi[c] = wp.float64(0.0)


# contact at held material points (10-23, the skin's own surface): per skin region q, the point m* of its sphere patch on
# the deformed surface y(m) = x_B + F_B (p + r m - X_B) + sum_s W_s(m) delta_s (SkinSurface.lowest, held within a step):
# g = cW n.x_B + n.F_B wbar + sum_s W_s n.y_s - c0, wbar = p + r m* - X_B - sum_s W_s (Y_s - X_B), cW = 1 - sum_s W_s;
# linear in x and y; augmented Lagrangian as k_contact; regions with cand 0 hold no condition
@wp.kernel(enable_backward=False)
def k_cpts(x: wp.array(dtype=wp.vec3d), y: wp.array(dtype=wp.vec3d), F: wp.array(dtype=wp.mat33d), cand: wp.array(dtype=int),
           body: wp.array(dtype=int), wbar: wp.array(dtype=wp.vec3d), cW: wp.array(dtype=wp.float64), loc: wp.array(dtype=int),
           Wq: wp.array(dtype=wp.float64), Smax: int, ng: wp.vec3d, c0: wp.array(dtype=wp.float64), lam: wp.array(dtype=wp.float64),
           rho: wp.array(dtype=wp.float64), gap: wp.array(dtype=wp.float64), dphi: wp.array(dtype=wp.float64), terms: wp.array2d(dtype=wp.float64)):
    q = wp.tid()
    if cand[q] == 0:                                                                 # no condition (its gap reads 0)
        gap[q] = wp.float64(0.0)
        dphi[q] = wp.float64(0.0)
        return
    B = body[q]
    g = cW[q]*wp.dot(ng, x[B])+wp.dot(ng, F[B]@wbar[q])-c0[0]
    for s in range(Smax):
        k = loc[q*Smax+s]
        if k >= 0:
            g = g+Wq[q*Smax+s]*wp.dot(ng, y[k])
    lm = lam[q]
    rh = rho[q]
    mx = wp.max(lm-rh*g, wp.float64(0.0))
    gap[q] = g
    dphi[q] = -mx
    wp.atomic_add(terms, 6, q % NSLOT, (mx*mx-lm*lm)/(wp.float64(2.0)*rh))


@wp.kernel(enable_backward=False)
def k_cpts_adj(cand: wp.array(dtype=int), body: wp.array(dtype=int), wbar: wp.array(dtype=wp.vec3d), cW: wp.array(dtype=wp.float64),
               loc: wp.array(dtype=int), Wq: wp.array(dtype=wp.float64), Smax: int, ng: wp.vec3d, dphi: wp.array(dtype=wp.float64),
               terms_adj: wp.array2d(dtype=wp.float64), x_adj: wp.array(dtype=wp.vec3d), y_adj: wp.array(dtype=wp.vec3d), F_adj: wp.array(dtype=wp.mat33d)):
    q = wp.tid()
    if cand[q] == 0:
        return
    B = body[q]
    a = terms_adj[6, q % NSLOT]*dphi[q]
    wp.atomic_add(x_adj, B, (a*cW[q])*ng)
    wp.atomic_add(F_adj, B, a*wp.outer(ng, wbar[q]))
    for s in range(Smax):
        k = loc[q*Smax+s]
        if k >= 0:
            wp.atomic_add(y_adj, k, (a*Wq[q*Smax+s])*ng)


@wp.kernel(enable_backward=False)
def k_contact_adj(cbody: wp.array(dtype=int), Xs: wp.array(dtype=wp.vec3d), dphi: wp.array(dtype=wp.float64), ng: wp.vec3d,
                  shift: int, ceoff: wp.array(dtype=int), cepid: wp.array(dtype=int), cew: wp.array(dtype=wp.float64),
                  cpar: wp.array(dtype=wp.float64), act: wp.array(dtype=int), only: wp.array(dtype=int),
                  terms_adj: wp.array2d(dtype=wp.float64), x_adj: wp.array(dtype=wp.vec3d), y_adj: wp.array(dtype=wp.vec3d),
                  F_adj: wp.array(dtype=wp.mat33d)):
    c = wp.tid()
    if only[0] == 1 and cpar[2] > wp.float64(0.5) and act[c] == 0:                   # (k_contact skipped it: dphi 0)
        return
    B = cbody[c]
    a = terms_adj[6, c % NSLOT]*dphi[c]
    wp.atomic_add(F_adj, B, a*wp.outer(ng, Xs[c]))
    if shift == 1:
        for k in range(ceoff[c], ceoff[c+1]):
            wp.atomic_add(y_adj, cepid[k], (a*cew[k])*ng)
    else:
        wp.atomic_add(x_adj, B, a*ng)


# friction at the contact cells as a glue to the ground (10-25, SkinWarp.set_contact): the material point at the step's
# start lowest point (wfix, k_contact's Xs then; pn its position then) carried by the cell, its slip
# s = cs P_t (p - pn) (P_t = I - n n^T; the midpoint rule cs 2: the slip over the step), the glue's force
# f = proj_{|f| <= R}(lamt - k s) (lamt the force it held at the step's start, k = rhot its stiffness, R = mu lamR with
# lamR the normal multiplier at the step's start, held through its passes), from the term wt (psi_R(lamt - k s) - |lamt|^2 / 2) / k,
# psi_R(q) = |q|^2 / 2 (|q| <= R), R |q| - R^2 / 2 (beyond): dE/ds = -wt f (the yield as a return map within the solve).
# psi_R is convex and C1, and for R > 0 the kink of |q| at 0 lies inside the quadratic part (10-25: holding a stick /
# slip state per solve cycled with period 2). Held mode: a cell holds glue this step iff it touched at the step's start
# (lamR > 0: its existence, bound and held force all from the step's start; 10-26: on the pass's active set a
# cell unloading within the step had its glue switched by its own flips, gripped and tipped up, freed and sank, period
# 2 to the 50-pass cap); it breaks at the step's end (MidpointIntegrator: the force kept on the next set only).
# fpar[0] = mu (0: off, nothing written)
@wp.kernel(enable_backward=False)
def k_friction(x: wp.array(dtype=wp.vec3d), y: wp.array(dtype=wp.vec3d), F: wp.array(dtype=wp.mat33d), shift: int,
               ceoff: wp.array(dtype=int), cepid: wp.array(dtype=int), cew: wp.array(dtype=wp.float64), cbody: wp.array(dtype=int), ng: wp.vec3d,
               wfix: wp.array(dtype=wp.vec3d), pn: wp.array(dtype=wp.vec3d), lamt: wp.array(dtype=wp.vec3d), rhot: wp.array(dtype=wp.float64),
               lamR: wp.array(dtype=wp.float64), fpar: wp.array(dtype=wp.float64), cpar: wp.array(dtype=wp.float64), act: wp.array(dtype=int),
               only: wp.array(dtype=int), energy: int,
               slip: wp.array(dtype=wp.vec3d), ff: wp.array(dtype=wp.vec3d), terms: wp.array2d(dtype=wp.float64)):
    c = wp.tid()
    zero = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
    mu = fpar[0]
    held = cpar[2] > wp.float64(0.5)
    if mu <= wp.float64(0.0) or (held and lamR[c] <= wp.float64(0.0)):             # (glue: the step's start contact)
        ff[c] = zero
        if only[0] == 0:
            slip[c] = zero
        return
    B = cbody[c]
    p = F[B]@wfix[c]
    if shift == 1:
        for k in range(ceoff[c], ceoff[c+1]):
            p = p+cew[k]*y[cepid[k]]
    else:
        p = p+x[B]
    d = p-pn[c]
    s = cpar[0]*(d-wp.dot(d, ng)*ng)
    slip[c] = s
    rh = rhot[c]
    R = mu*wp.max(lamR[c], wp.float64(0.0))
    lt = lamt[c]
    q = lt-rh*s
    qn = wp.length(q)
    f = zero
    phi = wp.float64(0.0)
    if qn <= R:
        f = q
        phi = (wp.dot(q, q)-wp.dot(lt, lt))/(wp.float64(2.0)*rh)
    else:
        if qn > wp.float64(0.0):
            f = (R/qn)*q
        phi = (R*qn-wp.float64(0.5)*R*R-wp.float64(0.5)*wp.dot(lt, lt))/rh
    ff[c] = f
    if energy == 1:
        wp.atomic_add(terms, 8, c % NSLOT, cpar[1]*phi)


@wp.kernel(enable_backward=False)
def k_friction_adj(cbody: wp.array(dtype=int), wfix: wp.array(dtype=wp.vec3d), ff: wp.array(dtype=wp.vec3d), shift: int,
                   ceoff: wp.array(dtype=int), cepid: wp.array(dtype=int), cew: wp.array(dtype=wp.float64), fpar: wp.array(dtype=wp.float64),
                   cpar: wp.array(dtype=wp.float64), lamR: wp.array(dtype=wp.float64), terms_adj: wp.array2d(dtype=wp.float64),
                   x_adj: wp.array(dtype=wp.vec3d), y_adj: wp.array(dtype=wp.vec3d), F_adj: wp.array(dtype=wp.mat33d)):
    """dE/dp = -wt cs f (f in the plane), p = F wfix + sum cew y (or x_B + F wfix)"""
    c = wp.tid()
    if fpar[0] <= wp.float64(0.0) or (cpar[2] > wp.float64(0.5) and lamR[c] <= wp.float64(0.0)):
        return
    B = cbody[c]
    g = (-terms_adj[8, c % NSLOT]*cpar[0]*cpar[1])*ff[c]
    wp.atomic_add(F_adj, B, wp.outer(g, wfix[c]))
    if shift == 1:
        for k in range(ceoff[c], ceoff[c+1]):
            wp.atomic_add(y_adj, cepid[k], cew[k]*g)
    else:
        wp.atomic_add(x_adj, B, g)


def contact_geometry(p, r, body_of_cell):
    """the surface cells (sites with an exposed patch, boundary_skin.regions), their bodies, centres, radii, exposed
    boundary arcs (CSR aoff: cc, rho, u, v, t0 < t1) and power-neighbour spheres (CSR noff: centre, radius), torch"""
    from free_surface.boundary_skin import regions
    R, _ = regions(p, r)
    sites = torch.unique(R.gsite)
    pos = torch.full((len(p),), -1, dtype=torch.long, device=p.device); pos[sites] = torch.arange(len(sites), device=p.device)
    ent = torch.argsort(pos[R.site], stable=True)                                    # entries grouped by surface cell
    cnt = torch.bincount(pos[R.site], minlength=len(sites))
    nb = R.S['nbrs'][sites]; ok = nb >= 0
    ncnt = ok.sum(1); nidx = nb[ok]                                                   # row-major: cell by cell
    cs = lambda k: torch.cat((k.new_zeros(1), k.cumsum(0)))
    boc = torch.as_tensor(np.asarray(body_of_cell), device=p.device)
    return dict(site=sites, body=boc[sites], p=p[sites], r=r[sites], aoff=cs(cnt), cc=R.c[ent], rho=R.rho[ent], u=R.u[ent], v=R.v[ent],
                t0=R.t0[ent], t1=R.t1[ent], noff=cs(ncnt), nbp=p[nidx], nbr=r[nidx], R=R)


class SkinWarp:
    """built from a SkinTerms (BoundaryTerms) st; energy_grad(x, y) -> (terms (6,), dE/dx (n, 3), dE/dy (npt, 3)), torch.
    hand_adjoints: k_bonds and k_walls differentiated by k_bonds_adj / k_walls_adj (False: Warp's generated adjoints)"""
    hand_adjoints = True

    def __init__(self, st):
        wp.init()
        self.st = st; m = st.model
        assert m.kinematics == 'exact' and m.body_law == 'nodalvol' and not m.residual and st.quad_ref and st.lateral
        self.n, self.npt, self.nb, self.ne = st.n, st.npt, len(st.i), len(st.e_pid)
        dev = 'cuda'
        T = lambda t, dt=wp.float64: wp.from_torch(t.contiguous(), dtype=dt)
        I = lambda t: wp.from_torch(t.to(torch.int32).contiguous(), dtype=wp.int32)
        self.bi, self.bj = I(st.i), I(st.j)
        self.A, self.nrm, self.L = T(st.A), T(st.nrm, wp.vec3d), T(st.L)
        self.hi, self.hj, self.wri, self.wrj, self.wt = T(st.hi), T(st.hj), T(st.ri, wp.vec3d), T(st.rj, wp.vec3d), T(st.t, wp.vec3d)
        self.wi, self.wj = T(st.wi), T(st.wj)
        self.wb, self.dX = T(st.wb), T(st.dX, wp.vec3d)
        self._kb_t = m.base.clone().contiguous()                                    # (own memory: set_material rewrites it)
        self.mri, self.mrj, self.n0, self.kb = T(m.ri, wp.vec3d), T(m.rj, wp.vec3d), T(m.n0, wp.vec3d), wp.from_torch(self._kb_t, dtype=wp.vec4d)
        assert st.gamma == 'sym_skin'
        self.Mt_inv = T(st.Mt_inv, wp.mat33d)
        self.Ninv, self.q6 = T(st.g_Ninv.reshape(self.n, 36)), T(st.g_q)
        self.pid, self.body, self.ew, self.el = I(st.e_pid), I(st.e_body), T(st.e_w), T(st.e_l, wp.vec3d)
        self.ed, self.ec = T(st.e_d, wp.vec3d), T(st.e_c, wp.vec3d)
        # o3 and lateral as row sums: entries of a part are contiguous, its pairs all ordered (a, b) row by row
        pa, pb, pp = st.pr_a, st.pr_b, st.pr_part
        P = int(pp.max())+1
        e_part = torch.full((self.ne,), -1, dtype=torch.long, device=pa.device); e_part[pa] = pp
        cnt = torch.bincount(e_part, minlength=P)
        off = torch.cat((cnt.new_zeros(1), cnt.cumsum(0))); poff = torch.cat((cnt.new_zeros(1), (cnt*cnt).cumsum(0)))
        ia, ib, s_ = pa-off[pp], pb-off[pp], cnt[pp]
        assert bool((e_part >= 0).all()) and bool((torch.diff(e_part) >= 0).all())
        assert bool((torch.arange(len(pa), device=pa.device) == poff[pp]+ia*s_+ib).all())
        tr = poff[pp]+ib*s_+ia                                                      # the pair (b, a)
        r = st.p_r[pp]; mu_, lam_ = float(st.mu), float(m.k_v); kap = lam_*mu_/(lam_+2*mu_)
        M3 = mu_/r*st.pr_m
        ML = (r/3)[:, None, None]*(.5*mu_*(st.pr_U+st.pr_T.transpose(1, 2))+kap*st.pr_T)
        self.e_part, self.poff_e, self.poff = I(e_part), I(off), I(poff)
        self.n3, self.NL = T(M3+M3[tr]), T((ML+ML[tr].transpose(1, 2)).contiguous(), wp.mat33d)
        self.cb, self.cr, self.cN = I(st.c_body), T(st.c_r), T(st.c_N, wp.mat33d)
        Vdet = st.Vres+torch.zeros_like(st.V).index_add(0, st.c_body, st.c_r*st.c_A/3)
        self.Vdet, self.V = T(Vdet), T(st.V)
        self.lam = float(m.k_v); self.mu = float(st.mu); self.kap = self.lam*self.mu/(self.lam+2*self.mu)
        self.barrier = float(m.barrier_J)
        self.wv = T(.5*self.lam*st.A*st.L/12 if st.variance else torch.zeros_like(st.A))
        # set_material: the coefficients per unit material (bonds k and o3 n3 per mu, the variance wv per lam, the lateral
        # NL = mu NLm + kappa NLk) and the kernels' scalars in matp (mu, eta, lam, lam_v, eta / mu, lam_v / lam): the
        # material changes in place and the graphs stay
        self._kb_unit = self._kb_t/self.mu
        M3u = st.pr_m/r
        self._n3_unit = M3u+M3u[tr]
        MLm, MLk = (r/3)[:, None, None]*(.5*(st.pr_U+st.pr_T.transpose(1, 2))), (r/3)[:, None, None]*st.pr_T
        self._NLm, self._NLk = (MLm+MLm[tr].transpose(1, 2)).contiguous(), (MLk+MLk[tr].transpose(1, 2)).contiguous()
        self._wv_unit = .5*st.A*st.L/12 if st.variance else torch.zeros_like(st.A)
        tt_ = lambda a_: wp.to_torch(a_, requires_grad=False)
        self.tn3, self.tNL, self.twv = tt_(self.n3), tt_(self.NL), tt_(self.wv)
        self.matp = wp.array(np.array([self.mu, 0., self.lam, 0., 0., 0., 0., 0.]), dtype=wp.float64, device=dev)
        self.tmatp = tt_(self.matp)
        self.npairs, self.nc = len(st.pr_a), len(st.c_body)
        self.ct = self.cp = self.vis = None
        self._only0, self._only1 = wp.zeros(1, dtype=int, device=dev), wp.array([1], dtype=int, device=dev)   # (k_contact's only)
        self._lat = (r, st.pr_U, st.pr_T, tr)                                       # the lateral term's pieces (viscous NLv)
        # viscosity (set_viscosity): its coefficients 0 until set; the measures at the step's start m_ref (visc_ref)
        z = lambda n_, dt_: wp.zeros(n_, dtype=dt_, device=dev)
        self.eta = self.lamv = self.smu = self.slam = 0.
        self.visc_central = True                                                   # visc_hvp's differences
        self.visc_weight = 1.                                                      # total_mode
        self.NLv = z(len(pa), wp.mat33d)
        self.vsame = 1                                                             # (k_rows: NLv = smu NL; smu 0 here)
        self.ref = dict(d=z(self.nb, wp.vec3d), th=z(self.nb, wp.vec3d), P=z(self.n, wp.mat33d), a=z(self.ne, wp.vec3d), dl=z(self.ne, wp.vec3d),
                        L=z(self.n, wp.float64))
        self.tref = {k_: wp.to_torch(a_, requires_grad=False) for k_, a_ in self.ref.items()}
        # plasticity (plastic.Plasticity, 10-26): the elastic terms' plastic references, the bonds' dp, thp (row 0 on
        # m - m_p) and the cells' plastic stretch Pp (o1, o2 on R^T F - P_p); 0, 0 and I: elastic, the terms as before
        self._Pp_t = torch.eye(3, dtype=f64, device=dev).repeat(self.n, 1, 1).contiguous()
        self.plas = dict(dp=z(self.nb, wp.vec3d), thp=z(self.nb, wp.vec3d), Pp=wp.from_torch(self._Pp_t, dtype=wp.mat33d))
        self.tplas = {k_: wp.to_torch(a_, requires_grad=False) for k_, a_ in self.plas.items()}
        self.tw = wp.array(np.ones(NT), dtype=wp.float64, device=dev)              # the terms' weights in the loss
        self.ttw = wp.to_torch(self.tw, requires_grad=False)

    def set_viscosity(self, eta, zeta, dt):
        """Kelvin-Voigt viscosity on the elasticity's own measures (eta shear, zeta bulk, per unit density; the second Lame
        viscosity lam_v = zeta - 2 eta / 3), before prepare(). Each elastic term gets its counterpart on the measures' change
        since the step's start, m - m_ref (visc_ref: bond d and theta, R^T F, R^T daff, the turned deviations dl, log J), with
        mu -> eta and lam -> lam_v (the lateral term's kappa = lam mu / (lam + 2 mu) -> lam_v eta / (lam_v + 2 eta)); their
        sum Et is row 7. Et / dt is the incremental dissipation potential of the step: the skin's energy E + Et / dt (loss
        weights ttw, total_mode) puts the viscous forces into its equilibrium; visc_hvp gives the bodies' implicit update its
        products (Et's Hessian at the reference, where m - m_ref = 0: Gauss-Newton); 2 Et / dt the energy dissipated. The
        bodies' force comes from visc_ref (without the viscous term), not from the solve's dE/dx"""
        eta, zeta = float(eta), float(zeta)
        lamv = zeta-2*eta/3
        self.eta, self.lamv, self.smu, self.slam = eta, lamv, eta/self.mu, lamv/self.lam
        self.tmatp[1], self.tmatp[3], self.tmatp[4], self.tmatp[5] = eta, lamv, self.smu, self.slam
        r, U, T_, tr = self._lat
        kv = lamv*eta/(lamv+2*eta) if lamv+2*eta > 0 else 0.
        ML = (r/3)[:, None, None]*(.5*eta*(U+T_.transpose(1, 2))+kv*T_)
        self._NLv = (ML+ML[tr].transpose(1, 2)).contiguous()
        self.NLv = wp.from_torch(self._NLv, dtype=wp.mat33d)
        c2 = kv-self.smu*self.kap                                                  # NLv = smu NL + c2 (r/3) T (paired): c2 0
        self.vsame = int(abs(c2) <= 16*np.finfo(float).eps*max(abs(kv), abs(self.smu*self.kap)))   # but for its own rounding:
        # one ratio (zeta = beta (lam + 2 mu / 3), eta = beta mu), k_rows then reads NL only
        self.vis = dict(eta=eta, zeta=zeta, lamv=lamv, dt=float(dt))
        self.total_mode()

    def set_material(self, E=None, nu=None, beta=None, rho=1000.):
        """the material in place (no rebuild; the gradient graphs stay, 2026-10-25): E (Pa) and nu as the build's
        (moduli per unit density, rho = pacnerf_dem_forward.RHO), beta the Kelvin-Voigt time (eta = beta mu, lam_v = beta
        lam; None: the current ratios kept). The bonds' stiffnesses and o3 scale with mu, the variance with lam, the
        lateral term is mu NLm + kappa NLk (its viscous NLv likewise). Not changed: the preconditioner's blocks and the
        contact's rho (the solves' pace only). Viscosity can not be switched on here (prepare() fixes k_rows' NLv use)"""
        mu0, lam0 = self.mu, self.lam
        nu0, E0 = lam0/(2*(lam0+mu0)), mu0*(3*lam0+2*mu0)/(lam0+mu0)*rho
        E = E0 if E is None else float(E); nu = nu0 if nu is None else float(nu)
        mu, lam = E/(2*(1+nu))/rho, E*nu/((1+nu)*(1-2*nu))/rho
        s, sl = (self.smu, self.slam) if beta is None else (float(beta), float(beta))
        if self.vis is None and (s > 0 or sl > 0):
            raise RuntimeError('set_material: no viscosity at prepare(), it can not be switched on in place')
        kap = lam*mu/(lam+2*mu)
        eta, lamv = s*mu, sl*lam
        kv = lamv*eta/(lamv+2*eta) if lamv+2*eta > 0 else 0.
        if self.vis is not None:
            c2 = kv-s*kap
            if int(abs(c2) <= 16*np.finfo(float).eps*max(abs(kv), abs(s*kap))) != self.vsame:
                raise RuntimeError('set_material: the viscosity ratios change whether k_rows reads NLv (fixed at prepare())')
            self._NLv.copy_(eta*self._NLm+kv*self._NLk)
            self.vis.update(eta=eta, zeta=lamv+2*eta/3, lamv=lamv)
        self._kb_t.copy_(mu*self._kb_unit)
        self.tn3.copy_(mu*self._n3_unit)
        self.tNL.copy_(mu*self._NLm+kap*self._NLk)
        self.twv.copy_(lam*self._wv_unit)
        self.mu, self.lam, self.kap, self.eta, self.lamv, self.smu, self.slam = mu, lam, kap, eta, lamv, s, sl
        self.glue()
        self.tmatp[:6] = torch.tensor([mu, eta, lam, lamv, s, sl], dtype=self.tmatp.dtype, device=self.tmatp.device)

    def total_mode(self):
        """the loss is the terms' sum, the viscous one times visc_weight / dt (1: the skin's energy in a step of the
        split scheme; 2: joint_step's midpoint rule)"""
        self.ttw.fill_(1.)
        if self.vis is not None:
            self.ttw[7] = self.visc_weight/self.vis['dt']

    def visc_mode(self):
        """the loss is Et alone (visc_hvp)"""
        self.ttw.zero_(); self.ttw[7] = 1.

    def elastic_mode(self):
        """the loss without the viscous term (the same model without viscosity)"""
        self.ttw.fill_(1.); self.ttw[7] = 0.

    def visc_ref(self, x, y):
        """the measures at (x, y) become the reference m_ref (the step's start); returns grad's (energy, dE/dx, dE/dy) there
        without the viscous term: the bodies' force at the step's start. (A solve's dE/dx carries d(Et / dt)/dx, the
        viscous force once more and explicitly: the bodies take theirs only from the implicit update, visc_hvp)"""
        self.elastic_mode()
        out = self.grad(x, y)
        self.total_mode()
        r = self.tref
        r['d'].copy_(self.tbd); r['th'].copy_(self.tbth); r['a'].copy_(self.tba); r['dl'].copy_(self.tdl); r['L'].copy_(self.tlj)
        r['P'].copy_(self.tR.transpose(1, 2)@self.tF)
        return out

    def visc_hvp(self, x, y, wx, wy, h=None, central=None):
        """the viscous products D (wx, wy) = (dEt/dx, dEt/dy) differentiated along (wx, wy) at the reference (x, y) of
        visc_ref, the direction scaled to largest entry 1: central differences (step h, default 1e-7; error ~ (h / cell)^2)
        or, as dEt = 0 there, one-sided dEt(z + h w) / h (one gradient; default h 1e-9, error ~ h / cell); central
        default self.visc_central"""
        central = self.visc_central if central is None else central
        h = h or (1e-7 if central else 1e-9)
        s = max(float(wx.abs().max()), float(wy.abs().max()), 1e-300)
        self.visc_mode()
        _, gxp, gyp = self.grad(x+(h/s)*wx, y+(h/s)*wy)
        if central:
            _, gxm, gym = self.grad(x-(h/s)*wx, y-(h/s)*wy)
            c = s/(2*h)
            gxp, gyp = gxp-gxm, gyp-gym
        else:
            c = s/h
        self.total_mode()
        return gxp*c, gyp*c

    def dissipated(self, tsum):
        """the energy dissipated over the step, 2 Et / dt, from a solve's terms at the step's end"""
        return 2*float(tsum[7])/self.vis['dt'] if self.vis is not None else 0.

    def set_contact_points(self, body, loc):
        """after set_contact, before prepare(): the conditions at held material points, one per skin region (body (k,),
        loc (k, Smax) the region's skin points, -1 padding); their data (cand, wbar, cW, W) and the multipliers and
        coefficients (tlam, trho) are torch views filled per step (WarpSkinSolver.refresh_points)"""
        k, S_ = loc.shape
        I = lambda t: wp.from_torch(t.to(torch.int32).contiguous(), dtype=wp.int32)
        z = lambda n_, dt: wp.zeros(n_, dtype=dt, device='cuda')
        cp = dict(k=k, S=int(S_), body=I(body), loc=I(loc.reshape(-1)), cand=z(k, wp.int32), wbar=z(k, wp.vec3d), cW=z(k, wp.float64),
                  W=z(k*S_, wp.float64), lam=z(k, wp.float64), rho=z(k, wp.float64))
        tt = lambda a: wp.to_torch(a, requires_grad=False)
        self.tc = dict(cand=tt(cp['cand']), wbar=tt(cp['wbar']), cW=tt(cp['cW']), W=tt(cp['W']).view(k, S_))
        self.tlam, self.trho = tt(cp['lam']), tt(cp['rho'])
        self.cp = cp

    def set_contact(self, cg, normal=(0., 1., 0.), c0=0., patch_area=None, ent_a=None):
        """contact of every surface cell's exposed sphere with the plane normal . y >= c0 (cg: contact_geometry), before
        prepare(). Multipliers tlam (start 0) and coefficients trho (default (lam + 2 mu) A / r per cell, A its exposed
        area: the cell's outer layer as a column, only the multiplier iteration's pace) are torch views kept between calls;
        tgap the gaps of the last evaluation"""
        T = lambda t, dt=wp.float64: wp.from_torch(t.contiguous(), dtype=dt)
        I = lambda t: wp.from_torch(t.to(torch.int32).contiguous(), dtype=wp.int32)
        k = len(cg['site'])
        ct = dict(k=k, body=I(cg['body']), p=T(cg['p'], wp.vec3d), r=T(cg['r']), aoff=I(cg['aoff']), cc=T(cg['cc'], wp.vec3d), rho_a=T(cg['rho']),
                  u=T(cg['u'], wp.vec3d), v=T(cg['v'], wp.vec3d), t0=T(cg['t0']), t1=T(cg['t1']), noff=I(cg['noff']), nbp=T(cg['nbp'], wp.vec3d),
                  nbr=T(cg['nbr']), X=T(self.st.X, wp.vec3d), ng=wp.vec3d(*[float(a) for a in normal]),
                  c0=wp.array([float(c0)], dtype=wp.float64, device='cuda'))
        ct['normal'] = tuple(float(a_) for a_ in normal)
        self.tc0 = wp.to_torch(ct['c0'], requires_grad=False)                     # the plane's offset, changeable
        ct['lam'] = wp.zeros(k, dtype=wp.float64, device='cuda')
        # ent_a given: the patch carried by its skin's mean deviation, sum_e w_e delta_e with w_e = ent_a / exposed area
        # (the interpolant's area mean over the cell's regions); dbar = sum_e w_e (Y_e - X_B)
        st = self.st
        cidx = torch.full((self.n,), -1, dtype=torch.long, device=st.X.device); cidx[cg['body']] = torch.arange(k, device=st.X.device)
        ec = cidx[st.e_body]
        if ent_a is not None:
            assert patch_area is not None and len(ent_a) == len(st.e_pid) and bool((ec >= 0).all())
            order = torch.argsort(ec, stable=True)
            we = (ent_a/patch_area[ec])[order]
            cnt = torch.bincount(ec, minlength=k)
            dbar = torch.zeros(k, 3, dtype=st.X.dtype, device=st.X.device).index_add(0, ec, (ent_a/patch_area[ec])[:, None]*(st.Y[st.e_pid]-st.X[st.e_body]))
            ct.update(shift=1, ceoff=I(torch.cat((cnt.new_zeros(1), cnt.cumsum(0)))), cepid=I(st.e_pid[order]), cew=T(we), dbar=T(dbar, wp.vec3d))
            self.c_w = (ent_a/patch_area[ec])                                         # per entry, for the preconditioner
        else:
            ct.update(shift=0, ceoff=I(torch.zeros(k+1, dtype=torch.long, device=st.X.device)), cepid=I(torch.zeros(1, dtype=torch.long, device=st.X.device)),
                      cew=T(torch.zeros(1, dtype=st.X.dtype, device=st.X.device)), dbar=T(torch.zeros(k, 3, dtype=st.X.dtype, device=st.X.device), wp.vec3d))
            self.c_w = None
        A = patch_area if patch_area is not None else 4*math.pi*cg['r']**2
        ct['rho'] = T(((self.lam+2*self.mu)*A/cg['r']).to(torch.float64))
        self.tlam, self.trho = wp.to_torch(ct['lam'], requires_grad=False), wp.to_torch(ct['rho'], requires_grad=False)
        self._glue_unit = (A/cg['r']).to(torch.float64).contiguous()              # the friction glue's stiffness per mu (below)
        ct['cpar'] = wp.array(np.array([1., 1., 0.]), dtype=wp.float64, device='cuda')  # k_contact's cs, wt, mode and per-cell
        ct['goff'] = wp.zeros(k, dtype=wp.float64, device='cuda')                 # goff, act: the gap itself, the multipliers'
        ct['act'] = wp.zeros(k, dtype=wp.int32, device='cuda')                    # max (the midpoint rule: joint_step)
        self.tcpar, self.tgoff = wp.to_torch(ct['cpar'], requires_grad=False), wp.to_torch(ct['goff'], requires_grad=False)
        self.tact = wp.to_torch(ct['act'], requires_grad=False)
        # friction (k_friction) as a glue to the ground (10-25, the bonds' rule): a touching cell's glue holds the force lamt
        # (0 when it touches), stretched by the step's slip s to lamt - k s, yields at R = mu lam (the force stays at R, the
        # rest of the slip slides), breaks as the pressure goes (R -> 0); kept from step to step (MidpointIntegrator), not
        # iterated to zero slip. k (rhot) = fkappa mu A / r: the cell's outer layer (exposed area A, depth r) in shear, as
        # the normal's default (lam + 2 mu) A / r in compression (10-25: holding the stick exact by multiplier passes took
        # 13-40 passes per step at k ~ the local stiffness, and a stiffer k made the solves trip over the stick-slip jump).
        # mu in fpar[0] (0: off); fst marks the last step's side (1 stuck, 2 sliding: the preconditioner); wfix, pn the
        # step's material points and their start positions
        zv = lambda: wp.zeros(k, dtype=wp.vec3d, device='cuda')
        # The bound R = mu lamR: lamR the normal multipliers at the step's start, held through its passes (solve_contact;
        # 10-26: on the pass's own multipliers a cell joining the set started at 0, its glue held nothing that pass and
        # gripped the next with R = mu lam, a jump ~ mu: at mu 10 the stiff landing's set cycled to the 50-pass cap)
        ct.update(fpar=wp.zeros(2, dtype=wp.float64, device='cuda'), lamt=zv(), rhot=T(self.mu*self._glue_unit), fst=wp.zeros(k, dtype=wp.int32, device='cuda'),
                  wfix=zv(), pn=zv(), lamR=wp.zeros(k, dtype=wp.float64, device='cuda'))
        tt_ = lambda a_: wp.to_torch(a_, requires_grad=False)
        self.tfpar, self.tlamt, self.trhot, self.tfst, self.twfix, self.tpn, self.tlamR = (tt_(ct[a_]) for a_ in ('fpar', 'lamt', 'rhot', 'fst', 'wfix', 'pn', 'lamR'))
        self.fkappa = 100.                                                        # (10-25, the jack's drop at mu 0.3, 8 frames: glue x 1 / 10 / 100
        # -> features x10 - x1 0.67 px, x100 - x10 0.085 px (~1/k: the stuck limit within ~0.01 px), products 1.11 / 1.16 / 1.38 x the
        # frictionless, rejections 34 / 37 / 69; x100 matched the multiplier-exact stick of the Coulomb passes: the glue is stuck)
        self.ct = ct
        self.cg = cg

    def glue(self):
        """the friction glue's stiffness k = fkappa mu A / r (set_contact; set_material moves mu)"""
        if self.ct is not None:
            self.trhot.copy_(self.fkappa*self.mu*self._glue_unit)

    def _buffers(self):
        z = lambda shape, dt: wp.zeros(shape, dtype=dt, device='cuda', requires_grad=True)
        n = self.n
        B = dict(S=z(n, wp.mat33d), F=z(n, wp.mat33d), R=z(n, wp.mat33d), q=z(n, wp.vec4d), G=z((n, 18), wp.float64), sg=z((n, 18), wp.float64),
                    dl=z(self.ne, wp.vec3d), g3=z(self.ne, wp.vec3d), gL=z(self.ne, wp.vec3d), gv=z(self.ne, wp.vec3d), Vd=z(n, wp.float64),
                    lj=z(n, wp.float64), bd=z(self.nb, wp.vec3d), bth=z(self.nb, wp.vec3d), ba=z(self.ne, wp.vec3d),
                    terms=z((NT, 256), wp.float64), tsum=z(NT, wp.float64), loss=z(1, wp.float64))
        if self.ct is not None:
            k = self.ct['k']
            B.update(gap=z(k, wp.float64), Xs=z(k, wp.vec3d), dphi=z(k, wp.float64), cpos=z(k, wp.vec3d), slip=z(k, wp.vec3d), ff=z(k, wp.vec3d))
        if self.cp is not None:
            k = self.cp['k']
            B.update(gap_p=z(k, wp.float64), dphi_p=z(k, wp.float64))
        return B

    def _forward(self, x, y, B, tape=None, only=None):
        """the energy into B; with a tape, the hand-written adjoints of k_rows and k_total are recorded on it. only: k_contact's
        flag array (the CG products: the cells off a held active set skipped)"""
        only = self._only0 if only is None else only
        n = self.n
        rf, pl = self.ref, self.plas
        wp.launch(k_S_bonds, self.nb, inputs=[x, self.bi, self.bj, self.A, self.nrm], outputs=[B['S']])
        wp.launch(k_S_skin, self.ne, inputs=[x, y, self.pid, self.body, self.ew, self.el], outputs=[B['S']])
        wp.launch(k_F, n, inputs=[B['S'], self.Mt_inv], outputs=[B['F'], B['R'], B['q']])
        hand = self.hand_adjoints and tape is not None
        bonds_in = [x, B['q'], self.bi, self.bj, self.mri, self.mrj, self.n0, self.kb, rf['d'], rf['th'], self.matp, pl['dp'], pl['thp']]
        wp.launch(k_bonds, self.nb, inputs=bonds_in, outputs=[B['terms'], B['bd'], B['bth']], record_tape=not hand)
        if hand:
            tape.record_func(lambda: wp.launch(k_bonds_adj, self.nb, inputs=bonds_in+[B['terms'].grad], outputs=[x.grad, B['q'].grad]),
                             arrays=[x, B['q'], B['terms']])
        wp.launch(k_G, self.nb, inputs=[B['F'], self.bi, self.bj, self.wb, self.dX], outputs=[B['G']])
        wp.launch(k_sg, n, inputs=[B['G'], self.Ninv], outputs=[B['sg']])
        walls_in = [x, B['F'], B['R'], self.bi, self.bj, self.hi, self.hj, self.L, self.wri, self.wrj, self.wt, self.A, self.nrm, self.wi, self.wj]
        wp.launch(k_walls, self.nb, inputs=walls_in, outputs=[B['Vd']], record_tape=not hand)
        if hand:
            tape.record_func(lambda: wp.launch(k_walls_adj, self.nb, inputs=walls_in+[B['Vd'].grad], outputs=[x.grad, B['F'].grad, B['R'].grad]),
                             arrays=[x, B['F'], B['R'], B['Vd']])
        wp.launch(k_entries, self.ne, inputs=[x, y, B['F'], B['R'], B['sg'], self.q6, self.pid, self.body, self.ed, self.ec, self.matp,
                                              rf['a'], rf['P'], pl['Pp']],
                  outputs=[B['dl'], B['Vd'], B['terms'], B['ba']])
        wp.launch(k_rows, self.ne, inputs=[B['dl'], rf['dl'], self.e_part, self.poff_e, self.poff, self.n3, self.NL, self.matp, self.NLv, self.vsame],
                  outputs=[B['g3'], B['gL'], B['gv'], B['terms']], record_tape=False)
        if tape is not None:
            tape.record_func(lambda: wp.launch(k_rows_adj, self.ne, inputs=[B['g3'], B['gL'], B['gv'], B['terms'].grad], outputs=[B['dl'].grad]),
                             arrays=[B['dl'], B['terms']])
        wp.launch(k_patches, self.nc, inputs=[B['F'], B['R'], self.cb, self.cr, self.cN, self.matp, rf['P'], pl['Pp']], outputs=[B['terms']])
        wp.launch(k_volume, n, inputs=[B['F'], B['Vd'], self.Vdet, self.V, self.matp, rf['L'], self.barrier], outputs=[B['lj'], B['terms']])
        wp.launch(k_variance, self.nb, inputs=[B['lj'], rf['L'], self.bi, self.bj, self.wv, self.matp], outputs=[B['terms']])
        ct, cp = self.ct, self.cp
        if ct is not None:                                                           # cells' affine patches: the condition, or
            en = 0 if cp is not None else 1                                          # with held points only their gaps
            sh = [ct['shift'], ct['ceoff'], ct['cepid'], ct['cew']]
            wp.launch(k_contact, ct['k'], inputs=[x, y, B['F'], ct['X']]+sh+[ct['dbar'], ct['body'], ct['p'], ct['r'], ct['aoff'], ct['cc'], ct['rho_a'],
                                                  ct['u'], ct['v'], ct['t0'], ct['t1'], ct['noff'], ct['nbp'], ct['nbr'], ct['ng'], ct['c0'], ct['lam'],
                                                  ct['rho'], en, ct['cpar'], ct['goff'], ct['act'], only],
                      outputs=[B['gap'], B['Xs'], B['cpos'], B['dphi'], B['terms']], record_tape=False)
            if tape is not None and en:
                tape.record_func(lambda: wp.launch(k_contact_adj, ct['k'], inputs=[ct['body'], B['Xs'], B['dphi'], ct['ng']]+sh+[ct['cpar'], ct['act'], only, B['terms'].grad],
                                                   outputs=[x.grad, y.grad, B['F'].grad]), arrays=[x, y, B['F'], B['terms']])
            wp.launch(k_friction, ct['k'], inputs=[x, y, B['F']]+sh+[ct['body'], ct['ng'], ct['wfix'], ct['pn'], ct['lamt'], ct['rhot'], ct['lamR'],
                                                   ct['fpar'], ct['cpar'], ct['act'], only, en],
                      outputs=[B['slip'], B['ff'], B['terms']], record_tape=False)
            if tape is not None and en:
                tape.record_func(lambda: wp.launch(k_friction_adj, ct['k'], inputs=[ct['body'], ct['wfix'], B['ff']]+sh+[ct['fpar'], ct['cpar'], ct['lamR'], B['terms'].grad],
                                                   outputs=[x.grad, y.grad, B['F'].grad]), arrays=[x, y, B['F'], B['terms']])
        if cp is not None:
            pin = [cp['cand'], cp['body'], cp['wbar'], cp['cW'], cp['loc'], cp['W'], cp['S']]
            wp.launch(k_cpts, cp['k'], inputs=[x, y, B['F']]+pin+[ct['ng'], ct['c0'], cp['lam'], cp['rho']], outputs=[B['gap_p'], B['dphi_p'], B['terms']],
                      record_tape=False)
            if tape is not None:
                tape.record_func(lambda: wp.launch(k_cpts_adj, cp['k'], inputs=pin+[ct['ng'], B['dphi_p'], B['terms'].grad],
                                                   outputs=[x.grad, y.grad, B['F'].grad]), arrays=[x, y, B['F'], B['terms']])
        wp.launch(k_total, (NT, 256), inputs=[B['terms']], outputs=[B['tsum']], record_tape=False)
        wp.launch(k_total_loss, 1, inputs=[B['tsum'], self.tw], outputs=[B['loss']], record_tape=False)
        if tape is not None:
            tape.record_func(lambda: wp.launch(k_total_adj, (NT, 256), inputs=[B['tsum'].grad, B['loss'].grad, self.tw], outputs=[B['terms'].grad]),
                             arrays=[B['terms'], B['tsum'], B['loss']])

    def prepare(self, graph=True):
        """persistent buffers and the gradient pass (zero, forward, backward) captured as one CUDA graph; then
        grad(x, y) -> (energy, dE/dx, dE/dy) only copies x, y in and replays it"""
        self.B = self._buffers()
        self.xw = wp.zeros(self.n, dtype=wp.vec3d, device='cuda', requires_grad=True)
        self.yw = wp.zeros(self.npt, dtype=wp.vec3d, device='cuda', requires_grad=True)
        self.tape = wp.Tape()
        def run(products=False):
            self.tape.reset()                                                        # (a recapture records afresh)
            for k_, a in self.B.items():                                             # adjoints, and the intermediates
                a.grad.zero_()                                                       # summed into (the others are
                if k_ in ('S', 'G', 'Vd', 'terms', 'tsum'):                          # written whole); the inputs keep
                    a.zero_()                                                        # their values
            self.xw.grad.zero_(); self.yw.grad.zero_()
            with self.tape:
                self._forward(self.xw, self.yw, self.B, self.tape, self._only1 if products else self._only0)
            self.tape.backward(loss=self.B['loss'])
        self._run = run                                                              # the gradient pass, for other graphs (WarpCG)
        self._run_products = lambda: run(True)                                       # the same for the CG products (only)
        run(); self.tape.reset()                                                     # compile the kernels outside the capture
        self.graph = None
        if graph:
            wp.capture_begin()
            run()
            self.graph = wp.capture_end()
        else:
            self.run = lambda: (run(), self.tape.reset())
        tt = lambda a: wp.to_torch(a, requires_grad=False)
        self.tx, self.ty = tt(self.xw), tt(self.yw)
        self.tgx, self.tgy, self.tloss = tt(self.xw.grad), tt(self.yw.grad), tt(self.B['loss'])
        self.tR, self.tF = tt(self.B['R']), tt(self.B['F'])                          # R_B, F_B of the last grad call
        self.tsum = tt(self.B['tsum'])                                               # the terms of the last call
        self.tlj = tt(self.B['lj'])                                                  # log J per body (with the barrier), last call
        self.tbd, self.tbth, self.tba, self.tdl = tt(self.B['bd']), tt(self.B['bth']), tt(self.B['ba']), tt(self.B['dl'])   # measures (visc_ref)
        if self.ct is not None:
            self.tgap, self.tXs = tt(self.B['gap']), tt(self.B['Xs'])
            self.tcpos, self.tslip, self.tff = tt(self.B['cpos']), tt(self.B['slip']), tt(self.B['ff'])   # (friction)
        if self.cp is not None:                                                      # the conditions are the held points':
            self.tgap_cell, self.tgap = self.tgap, tt(self.B['gap_p'])               # the cells' affine gaps kept apart

    def grad(self, x, y):
        self.tx.copy_(x); self.ty.copy_(y)
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.run()
        return self.tloss[0].clone(), self.tgx.clone(), self.tgy.clone()

    def hvp_fd(self, x, y, V, h, g=None):
        """Hessian-vector products in y by differences of the gradient along each row v of V (k, 3 npt), scaled to
        largest entry 1 (the step is h): central (g(y + h v) - g(y - h v)) / (2 h), or one-sided (g(y + h v) - g) / h
        when g = dE/dy at (x, y) (flat) is given"""
        out = torch.empty_like(V)
        for k in range(V.shape[0]):
            s = V[k].abs().max().clamp_min(1e-300)
            v = (V[k]/s).view(-1, 3)
            gp = self.grad(x, y+h*v)[2]
            if g is None:
                d = (gp-self.grad(x, y-h*v)[2])/(2*h)
            else:
                d = (gp-g.view(-1, 3))/h
            out[k] = d.reshape(-1)*s
        return out

    def energy(self, x, y):
        B = self._buffers()
        self._forward(wp.from_torch(x.contiguous(), dtype=wp.vec3d), wp.from_torch(y.contiguous(), dtype=wp.vec3d), B)
        return wp.to_torch(B['tsum']).clone()

    def energy_grad(self, x, y):
        xw = wp.from_torch(x.detach().clone().contiguous(), dtype=wp.vec3d, requires_grad=True)
        yw = wp.from_torch(y.detach().clone().contiguous(), dtype=wp.vec3d, requires_grad=True)
        B = self._buffers()
        tape = wp.Tape()
        with tape:
            self._forward(xw, yw, B, tape)
        tape.backward(loss=B['loss'])
        return wp.to_torch(B['tsum']).clone(), wp.to_torch(xw.grad).clone(), wp.to_torch(yw.grad).clone()


def warp_solver(st, bsk, body_of_cell, W, h, hvp='forward', cg_graph=True):
    """skin_solve.SkinSolver with Warp's energy and gradient (one pass gives the energy, dE/dy and, besides, dE/dx and
    the rotations) and Hessian-vector products by differences of the gradient (step h): 'forward' one-sided from the
    gradient the solver already has at y, or 'central'. cg_graph (with 'forward' and the Warp preconditioner): each CG
    iteration one CUDA graph (WarpCG)"""
    from free_surface.skin_solve import SkinSolver

    class WarpSkinSolver(SkinSolver):
        def energy(self, x, y):
            return W.grad(x, y)[0]

        def grad(self, x, y):
            return W.grad(x, y)[2].reshape(-1)

        def eg(self, x, y):
            e, gx, gy = W.grad(x, y)
            c_ = W.ct is not None
            return e, gy.reshape(-1), (gx, W.tR.clone(), W.tgap.clone() if c_ else None, W.tsum.clone(), W.tXs.clone() if c_ else None)

        def contact_weights(self, Xs):
            """om (k, Smax): the gap of contact cell c along the normal per skin point (slot) of its body's group,
            dg/dy_e = w_e (l_e . Mt^-1 (X* - X_B)) n, X* - X_B = Xs"""
            st = self.st
            om_e = st.e_w*(st.e_l*torch.einsum('eij,ej->ei', st.Mt_inv[st.e_body], Xs[self.e_contact])).sum(1)
            if W.c_w is not None:                                                    # the mean deviation's direct share
                om_e = om_e+W.c_w
            om = torch.zeros(W.ct['k'], self.bodies['S'], dtype=om_e.dtype, device=om_e.device)
            return om.index_put_((self.e_contact, self.e_slot), om_e, accumulate=True)

        def prec_at(self, make_prec, x, y, aux):
            """the turned blocks, and the active contacts' penalties (Sherman-Morrison in their body's block)"""
            f = make_prec(aux[1])
            if W.ct is not None and getattr(getattr(f, 'wprec', None), 'ck', 0):              # (always: taua current)
                act = (W.tlam-W.trho*aux[2]) > 0
                om = self.points_weights() if W.cp is not None else self.contact_weights(aux[4])
                f.wprec.contact_update(om, W.trho*act)
            return f

        def points_setup(self, prec, surf, p, r, body_of_cell, interp='shepard0', kappa=100.):
            """contact at held material points (W.set_contact_points with surf.body, surf.I.loc_g): surf a SkinSurface
            on every region. Per region q: C_q = the largest sum_s |W_s| of its interpolant over sampled points (inside:
            the lowest() start points; arcs: their 3-point traces), so that it can reach the plane only if its cell's
            affine gap <= C_q max_s |n . delta_s|; its points' and its body's entries' slots in the body's group"""
            from free_surface.boundary_skin import lagrange3 as lagrange3_
            st, dev = self.st, self.dev
            self.surf_all, self.kappa = surf, kappa
            self.p_all, self.r_all, self.boc_all, self.interp = p, r, body_of_cell, interp
            self.nvec = torch.tensor(W.ct['normal'], dtype=st.X.dtype, device=dev)
            P = surf.P
            with torch.no_grad():
                th = surf.a0[..., None]+(surf.a1-surf.a0)[..., None]*(torch.arange(4, dtype=st.X.dtype, device=dev)+.5)/4
                fr = torch.tensor([.25, .5, .75], dtype=st.X.dtype, device=dev)
                smp = (1-fr[:, None])*surf.m0[:, None, None, None]+fr[:, None]*surf.arcpt(th)[:, :, :, None]
                smp = torch.cat((surf.m0[:, None], smp.reshape(P, -1, 3)), 1); smp = smp/smp.norm(dim=-1, keepdim=True)
                C = torch.zeros(P, dtype=st.X.dtype, device=dev)
                for q0 in range(0, smp.shape[1], 32):
                    sq = smp[:, q0:q0+32]
                    C = torch.maximum(C, surf.I.make_phi(sq)[0](sq).abs().sum(-1).amax(1))
                ta = torch.linspace(0, 1, 33, dtype=st.X.dtype, device=dev)
                tha = surf.a0[..., None]+(surf.a1-surf.a0)[..., None]*ta
                La = lagrange3_(tha, surf.a0[..., None], surf.a1[..., None]).abs().sum(-1)
                C = torch.maximum(C, torch.where(surf.vb[..., None], La, torch.zeros_like(La)).amax((1, 2)))
            self.C_reg = C
            gidx = self.gidx.cpu().numpy(); lists = self.bodies['lists']
            loc = surf.I.loc_g.cpu().numpy(); body = surf.body.cpu().numpy()
            slot = -np.ones_like(loc)
            for q in range(P):
                v = loc[q] >= 0
                slot[q, v] = np.searchsorted(lists[gidx[body[q]]], loc[q, v])
            self.p_slot = torch.as_tensor(slot, device=dev)
            eb = self.e_body.cpu().numpy(); order = np.argsort(eb, kind='stable'); start = np.searchsorted(eb[order], np.arange(self.n+1))
            pq, pe = [], []
            for q in range(P):
                e_ = order[start[body[q]]:start[body[q]+1]]; pq.append(np.full(len(e_), q)); pe.append(e_)
            self.p_pq, self.p_pe = torch.as_tensor(np.concatenate(pq), device=dev), torch.as_tensor(np.concatenate(pe), device=dev)
            cells = torch.as_tensor(np.searchsorted(W.cg['site'].cpu().numpy(), surf.gsite.cpu().numpy()), device=dev)
            assert bool((W.cg['site'][cells] == surf.gsite).all())
            self.p_cell = cells
            prec.set_contact(self.gidx[surf.body], self.bodies['S'], W.ct['normal'])
            self.union = torch.zeros(P, dtype=torch.bool, device=dev); self.surf_u = None
            return C

        def points_weights(self):
            """om (k, Smax) of the held points: dg/dy along the normal per slot of the region's body group (the direct
            W_s and the levers of F_B: w_e (l_e . Mt^-1 wbar))"""
            st, tc = self.st, W.tc
            q, e = self.p_pq, self.p_pe
            B = st.e_body[e]
            om_e = st.e_w[e]*(st.e_l[e]*torch.einsum('eij,ej->ei', st.Mt_inv[B], tc['wbar'][q])).sum(1)
            om = torch.zeros(W.cp['k'], self.bodies['S'], dtype=om_e.dtype, device=om_e.device)
            om.index_put_((q, self.e_slot[e]), om_e, accumulate=True)
            v = self.p_slot >= 0
            qq = torch.nonzero(v)
            om.index_put_((qq[:, 0], self.p_slot[v]), tc['W'][:, :self.p_slot.shape[1]][v], accumulate=True)
            return om*tc['cand'][:, None].to(om.dtype)

        def refresh_points(self, x, y, make_prec):
            """per step, before solve_contact: the regions that can reach the plane (their cell's affine gap <= C_q
            max_s |n . delta_s|) join the held set (with their sites' other regions); the lowest points of the held set's
            regions on the deformed surface (SkinSurface.lowest) become the held points (cand, wbar, cW, W); rho = kappa /
            a.B^-1 a for them at the current rotations. Returns the held count, the new ones and their lowest gap."""
            from free_surface.skin_contact import SkinSurface
            st, dev, surf, n = self.st, self.dev, self.surf_all, self.nvec
            e, gx, gy = W.grad(x, y)
            hcell = W.tgap_cell.clone(); Rb = W.tR.clone()
            with torch.no_grad():
                F = st.F(x, y)
                loc = surf.I.loc_g; lv = loc >= 0; lc = loc.clamp(min=0)
                Bq = surf.body
                dl = y[lc]-x[Bq][:, None]-torch.einsum('qij,qsj->qsi', F[Bq], st.Y[lc]-st.X[Bq][:, None])
                dn = torch.where(lv, (dl@n).abs(), torch.zeros_like(lv, dtype=dl.dtype)).amax(1)
                reach = hcell[self.p_cell] <= self.C_reg*dn
                sites = torch.unique(surf.gsite[reach])
                new = torch.isin(surf.gsite, sites) & ~self.union
            new_gap = float('nan')
            if bool(new.any()):
                self.union |= new
                self.surf_u = SkinSurface(st, surf.R, self.p_all, self.r_all, self.boc_all, self.interp, sl=torch.nonzero(self.union)[:, 0])
            if self.surf_u is None:
                return 0, 0, new_gap
            su = self.surf_u
            low = su.lowest(x, y, n, float(W.tc0[0]))
            with torch.no_grad():
                qi = su.sl
                XB = st.X[su.body]
                Wl = low['W']*su.lmask
                Yl = st.Y[su.loc]
                wbar = su.p+su.r[:, None]*low['m']-XB-torch.einsum('qs,qsi->qi', Wl, Yl-XB[:, None])
                tc = W.tc
                tc['cand'][qi] = 1
                tc['wbar'][qi] = wbar
                tc['cW'][qi] = 1-Wl.sum(1)
                tc['W'][qi] = 0.
                tc['W'][qi, :Wl.shape[1]] = Wl
                if bool(new.any()):
                    new_gap = float(low['gap'][new[qi]].min())
                f = make_prec(Rb)
                f.wprec.contact_update(self.points_weights(), torch.zeros(W.cp['k'], dtype=x.dtype, device=dev))
                aua = f.wprec.taua
                W.trho.copy_(torch.where((tc['cand'] > 0) & (aua > 0), self.kappa/aua, W.trho))
            return int(self.union.sum()), int(new.sum()), new_gap

        def contact_setup(self, prec, kappa=100.):
            """after W.set_contact, prepare() and the preconditioner: the contact cells' groups into prec, and
            rho_c = kappa / (a.B^-1 a) at rest (a the gap's derivative at the rest lowest point, B the body's block of
            K_yy at rest): kappa times the cell's contact stiffness with the rest of the skin held; the multiplier
            iteration then shrinks its error about 1 / (1 + kappa) per pass (rho only sets that pace)"""
            ct = W.ct
            cbody = torch.as_tensor(np.asarray(W.cg['body'].cpu()), device=self.dev)
            cidx = torch.full((self.n,), -1, dtype=torch.long, device=self.dev); cidx[cbody] = torch.arange(ct['k'], device=self.dev)
            self.e_contact = cidx[self.e_body]
            assert bool((self.e_contact >= 0).all()) and bool((self.e_pid == self.st.e_pid).all()) and bool((self.gidx[cbody] >= 0).all())
            prec.set_contact(self.gidx[cbody], self.bodies['S'], ct['normal'])
            W.grad(self.st.X, self.st.Y)
            prec(torch.eye(3, dtype=self.st.X.dtype, device=self.dev).expand(self.n, 3, 3).contiguous())
            prec.contact_update(self.contact_weights(W.tXs.clone()), torch.zeros(ct['k'], dtype=self.st.X.dtype, device=self.dev))
            W.trho.copy_(kappa/prec.taua)
            self.kappa = kappa
            return prec.taua.clone()

        def solve_contact(self, x, y, make_prec, ref, tol=1e-4, maxit=300, al_max=50):
            """the skin under the contact conditions (W.set_contact) by the augmented Lagrangian: solve with the
            multipliers W.tlam held, then lam <- max(0, lam - rho gap), until |lam change| / kappa <= tol ref (the force
            the violation left would carry at the cells' contact stiffness rho / kappa: the inner solve's scale; the
            change is then kept for the next call, the returned state is that of the last solve). rho is first set to
            kappa / a.B^-1 a at the last preconditioner's lowest points (contact_setup). Stats: the last solve's, with
            products summed over the passes, 'al' passes, 'dlam' the last change, 'lam' the multipliers max(0, lam - rho
            gap) at the returned skin (the contact forces)"""
            wp_ = getattr(make_prec, 'ck', 0) and make_prec
            if wp_:
                W.trho.copy_(torch.where(wp_.taua > 0, self.kappa/wp_.taua, W.trho))
            prods = 0
            for it in range(1, al_max+1):
                y, s_ = self.solve(x, y, make_prec, ref, tol=tol, maxit=maxit)
                prods += s_['products']
                lam_used = W.tlam.clone()
                lam_new = (W.tlam-W.trho*s_['aux'][2]).clamp(min=0)
                if W.cp is not None:
                    lam_new = lam_new*(W.tc['cand'] > 0)                                 # regions without a condition
                dl = float((lam_new-W.tlam).norm())
                W.tlam.copy_(lam_new)
                if dl/self.kappa <= tol*ref:
                    break
            s_.update(products=prods, al=it, dlam=dl, lam=lam_new, lam_used=lam_used)
            return y, s_

        def prec_rotations(self, x, y, aux):
            return aux[1]

        def preconditioner(self, B, g):
            return warp_preconditioner(B, g, self.npt)

        def hvp(self, x, y, g=None):
            return lambda V: W.hvp_fd(x, y, V, h, g if hvp == 'forward' else None)

        def steihaug(self, x, y, g, prec, delta, stop, cap):
            """SkinSolver.steihaug with each CG iteration one WarpCG graph (one-sided products, Warp preconditioner)"""
            if not (cg_graph and hvp == 'forward' and hasattr(prec, 'wprec')):
                return super().steihaug(x, y, g, prec, delta, stop, cap)
            if self._cg is None or self._cg.wprec is not prec.wprec:
                self._cg = WarpCG(W, prec.wprec, h)
            cg = self._cg; T = cg.t
            W.tx.copy_(x); T['y'].copy_(y); T['g'].copy_(g.view(-1, 3))
            r = T['r']; r.copy_(-g.view(-1, 3))
            z = prec(r.view(1, -1)).view(-1, 3); T['z'].copy_(z); T['d'].copy_(z)
            rz = float((r*z).sum())
            if delta is None:
                delta = rz**.5
            cg.tsc[5] = rz
            T['p'].zero_(); T['Hp'].zero_()
            pMp, pMd, dMd = 0., 0., rz
            k = 0
            def out(edge, neg):
                return T['p'].reshape(1, -1).clone(), T['Hp'].reshape(1, -1).clone(), pMp, delta, edge, neg, k
            while True:
                dHd, rr, rz1 = cg.step(); k += 1
                al = rz/dHd if dHd > 0 else 0.                                       # as on the device
                be = rz1/rz if rz != 0 else 0.
                def to_edge():
                    return (-pMd+(pMd**2+dMd*(delta**2-pMp))**.5)/dMd
                if dHd <= 0:                                                         # the graph took no step (alpha 0)
                    tau = to_edge(); T['p'].add_(T['d_old'], alpha=tau); T['Hp'].add_(T['Hd'], alpha=tau)
                    return out(True, 1)
                if pMp+2*al*pMd+al**2*dMd >= delta**2:                               # back from p + alpha d to the edge
                    tau = to_edge(); T['p'].add_(T['d_old'], alpha=tau-al); T['Hp'].add_(T['Hd'], alpha=tau-al)
                    return out(True, 0)
                pMp = pMp+2*al*pMd+al**2*dMd
                if rr**.5 <= stop or k >= cap:
                    return out(False, 0)
                pMd = be*(pMd+al*dMd); dMd = rz1+be**2*dMd; rz = rz1
    S = WarpSkinSolver(st, bsk, body_of_cell)
    S._cg = None
    return S
