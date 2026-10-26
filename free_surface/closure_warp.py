"""The skinless sphere model (sphere_closure.ClosureTerms) in Warp, float64 (2026-10-26): the bodies only, F_B closed on
the exposed spheres, the kernels of skin_warp where the terms are the same.
  S        k_S_bonds (the walls' S_B, Warp's adjoint)
  closure  k_closure: F_B Mt = S_B + F_s Mcap per body, F_s = R (I + e_s(U)) (sphere_closure.ClosureTerms.completion),
           Newton from the last root (Fw, kept between calls) with the analytic 9 x 9 Jacobian: for dF, with A = R^T dF,
           W = [w]x, (tr U I - U) w = 2 axial(skew A) (dR = R W), dU = sym A - (W U - U W) / 2, dF_s = dR (I + e_s) + R e_s(dU)
           (linear), dr = dF Mt - dF_s Mcap; up to iters iterations, each body stopping once its step is below 1e-10
           (then its error is ~1e-20); each step with the LU at the current iterate (at Fw kept from the last call, cold()
           after a reset), the Jacobian rebuilt at the new one, so the adjoint's is at the root. ClosureCG's products
           (product_iters 1): one step from the last point, F's error ~ (the step)^2, i.e. O(h) relative in the forward
           difference, as the difference's own error.
           Adjoint k_closure_adj: dF = J^-1 dS, so S_bar += J^-T F_bar
  R, q     k_F on F (S = F_B, Mt^-1 = I: the polar factor by its 12 Newton steps, Warp's adjoint)
  terms    k_bonds (hand adjoint), k_walls (hand adjoint), k_patches (o1), k_volume (walls + det F_B (V_res + r A / 3)),
           k_variance; no skin terms
  contact  k_contact with the affine caps (shift 0), multipliers 0 and rho = the cap's stiffness k_n = (lam + 2 mu) A / 3 r:
           the cap's own compliance in series with a rigid ground (sphere_closure_jack.py, PLAN), a one-sided spring;
           k_friction's glue with rhot = k_t = 4 mu A / 3 r (fkappa 4 / 3): the rigid glue and the cap's shear compliance
Viscosity (Kelvin-Voigt on the same measures: bonds d, theta; R^T F; log J) as SkinWarp's, references in self.ref."""
import math

import numpy as np
import torch
import warp as wp

from free_surface.skin_warp import (NT, NSLOT, SkinWarp, contact_geometry, k_S_bonds, k_F, k_bonds, k_bonds_adj, k_walls, k_walls_adj,
                                    bond_quad, mat_quat, quat_mat, quat_mul, quat_log, quat_conj, quat_mat_adj, quat_log_adj,
                                    k_patches, k_volume, k_variance, k_total, k_total_loss, k_total_adj, k_contact, k_contact_adj,
                                    k_friction, k_friction_adj)

f64 = torch.float64
mat99 = wp.types.matrix(shape=(9, 9), dtype=wp.float64)
vec9 = wp.types.vector(length=9, dtype=wp.float64)


@wp.func
def polar12(F: wp.mat33d):
    """the polar factor by Newton's R <- (R + R^-T) / 2 from F: 8 steps (|U - I| 0.5 reaches 1e-16 in 5)"""
    R = F
    for it in range(8):
        R = (R+wp.transpose(wp.inverse(R)))*wp.float64(0.5)
    return R


@wp.func
def es_of(e: wp.mat33d, P: wp.mat33d, mm: wp.mat33d, kap: wp.float64):
    et = P@e@P
    return et-mm*(kap*wp.trace(et))


@wp.func
def solve9(A: mat99, b: vec9):
    """Gaussian elimination with partial pivoting"""
    M = A
    v = b
    for c in range(9):
        piv = c
        best = wp.abs(M[c, c])
        for r in range(c+1, 9):
            if wp.abs(M[r, c]) > best:
                best = wp.abs(M[r, c])
                piv = r
        if piv != c:
            for k in range(9):
                t = M[c, k]
                M[c, k] = M[piv, k]
                M[piv, k] = t
            t2 = v[c]
            v[c] = v[piv]
            v[piv] = t2
        for r in range(c+1, 9):
            f = M[r, c]/M[c, c]
            for k in range(c, 9):
                M[r, k] = M[r, k]-f*M[c, k]
            v[r] = v[r]-f*v[c]
    x = vec9()
    for ii in range(9):
        r = 8-ii
        acc = v[r]
        for k in range(r+1, 9):
            acc = acc-M[r, k]*x[k]
        x[r] = acc/M[r, r]
    return x


@wp.func
def lu9(A: mat99):
    """LU with partial pivoting, packed (L below the diagonal, unit; U on and above); perm: row i of P A is row perm[i]"""
    M = A
    perm = vec9()
    for i in range(9):
        perm[i] = wp.float64(i)
    for c in range(9):
        piv = c
        best = wp.abs(M[c, c])
        for r in range(c+1, 9):
            if wp.abs(M[r, c]) > best:
                best = wp.abs(M[r, c])
                piv = r
        if piv != c:
            for k in range(9):
                t = M[c, k]
                M[c, k] = M[piv, k]
                M[piv, k] = t
            t2 = perm[c]
            perm[c] = perm[piv]
            perm[piv] = t2
        for r in range(c+1, 9):
            f = M[r, c]/M[c, c]
            M[r, c] = f
            for k in range(c+1, 9):
                M[r, k] = M[r, k]-f*M[c, k]
    return M, perm


@wp.func
def lu9_solve(LU: mat99, perm: vec9, b: vec9):
    """A x = b from lu9's factors"""
    y = vec9()
    for i in range(9):
        acc = b[int(perm[i])]
        for k in range(i):
            acc = acc-LU[i, k]*y[k]
        y[i] = acc
    x = vec9()
    for ii in range(9):
        r = 8-ii
        acc = y[r]
        for k in range(r+1, 9):
            acc = acc-LU[r, k]*x[k]
        x[r] = acc/LU[r, r]
    return x


@wp.func
def lu9_solve_t(LU: mat99, perm: vec9, b: vec9):
    """A^T x = b from lu9's factors: U^T w = b, L^T v = w, x[perm[i]] = v[i]"""
    w = vec9()
    for i in range(9):
        acc = b[i]
        for k in range(i):
            acc = acc-LU[k, i]*w[k]
        w[i] = acc/LU[i, i]
    v = vec9()
    for ii in range(9):
        r = 8-ii
        acc = w[r]
        for k in range(r+1, 9):
            acc = acc-LU[k, r]*v[k]
        v[r] = acc
    x = vec9()
    for i in range(9):
        x[int(perm[i])] = v[i]
    return x


@wp.func
def solve9_np(A: mat99, b: vec9):
    """Gaussian elimination without pivoting: every index static (the loops unrolled), so the matrix may stay in
    registers; the closure's Jacobian is dominated by Mt (x) I (SPD), its pivots positive"""
    M = A
    v = b
    for c in range(9):
        inv = wp.float64(1.0)/M[c, c]
        for r in range(c+1, 9):
            f = M[r, c]*inv
            for k in range(c+1, 9):
                M[r, k] = M[r, k]-f*M[c, k]
            v[r] = v[r]-f*v[c]
    x = vec9()
    for ii in range(9):
        r = 8-ii
        acc = v[r]
        for k in range(r+1, 9):
            acc = acc-M[r, k]*x[k]
        x[r] = acc/M[r, r]
    return x


@wp.func
def closure_jac(R: wp.mat33d, U: wp.mat33d, es: wp.mat33d, Mtb: wp.mat33d, Mcb: wp.mat33d, Pb: wp.mat33d, mmb: wp.mat33d, ka: wp.float64):
    """the closure residual's 9 x 9 Jacobian at F = R U (column k: dF = e_k)"""
    I = wp.identity(n=3, dtype=wp.float64)
    T = wp.inverse(wp.trace(U)*I-U)
    Jb = mat99()
    for k in range(9):
        dF = wp.mat33d()
        dF[k//3, k % 3] = wp.float64(1.0)
        A = wp.transpose(R)@dF
        K = (A-wp.transpose(A))*wp.float64(0.5)
        w = T@wp.vec3d(K[2, 1], K[0, 2], K[1, 0])*wp.float64(2.0)
        Wm = wp.skew(w)
        dU = (A+wp.transpose(A))*wp.float64(0.5)-(Wm@U-U@Wm)*wp.float64(0.5)
        dFs = R@Wm@(I+es)+R@es_of(dU, Pb, mmb, ka)
        dr = dF@Mtb-dFs@Mcb
        for i in range(9):
            Jb[i, k] = dr[i//3, i % 3]
    return Jb


@wp.kernel(enable_backward=False)
def k_closure(S: wp.array(dtype=wp.mat33d), Mt: wp.array(dtype=wp.mat33d), Mc: wp.array(dtype=wp.mat33d), P: wp.array(dtype=wp.mat33d),
              mm: wp.array(dtype=wp.mat33d), kap: wp.array(dtype=wp.float64), iters: int, Fw: wp.array(dtype=wp.mat33d),
              F: wp.array(dtype=wp.mat33d), J: wp.array(dtype=mat99), step: wp.array(dtype=wp.float64), perm: wp.array(dtype=vec9)):
    """Newton from Fw with the LU kept in J, perm (the Jacobian at Fw): each step with the LU at the current iterate, the
    Jacobian rebuilt at the new one (the next step's, and at the end the adjoint's)"""
    b = wp.tid()
    Fb = Fw[b]
    Sb, Mtb, Mcb, Pb, mmb, ka = S[b], Mt[b], Mc[b], P[b], mm[b], kap[0]
    I = wp.identity(n=3, dtype=wp.float64)
    LUb = J[b]
    pb = perm[b]
    R = polar12(Fb)
    A0 = wp.transpose(R)@Fb
    U = (A0+wp.transpose(A0))*wp.float64(0.5)
    es = es_of(U-I, Pb, mmb, ka)
    last = wp.float64(0.0)
    for it in range(iters):
        r = Fb@Mtb-Sb-R@(I+es)@Mcb
        rv = vec9()
        for i in range(9):
            rv[i] = -r[i//3, i % 3]
        dx = lu9_solve(LUb, pb, rv)
        last = wp.float64(0.0)
        for i in range(9):
            Fb[i//3, i % 3] = Fb[i//3, i % 3]+dx[i]
            last = wp.max(last, wp.abs(dx[i]))
        R = polar12(Fb)
        A0 = wp.transpose(R)@Fb
        U = (A0+wp.transpose(A0))*wp.float64(0.5)
        es = es_of(U-I, Pb, mmb, ka)
        LUb, pb = lu9(closure_jac(R, U, es, Mtb, Mcb, Pb, mmb, ka))
        if last < wp.float64(1e-10):                                                # quadratic: the next step ~1e-20
            break
    F[b] = Fb
    Fw[b] = Fb
    J[b] = LUb
    perm[b] = pb
    step[b] = last


@wp.kernel(enable_backward=False)
def k_closure_lu(Fw: wp.array(dtype=wp.mat33d), Mt: wp.array(dtype=wp.mat33d), Mc: wp.array(dtype=wp.mat33d), P: wp.array(dtype=wp.mat33d),
                 mm: wp.array(dtype=wp.mat33d), kap: wp.array(dtype=wp.float64), J: wp.array(dtype=mat99), perm: wp.array(dtype=vec9)):
    """the LU of the Jacobian at Fw (k_closure's starting factors after Fw is reset)"""
    b = wp.tid()
    Fb = Fw[b]
    R = polar12(Fb)
    A0 = wp.transpose(R)@Fb
    U = (A0+wp.transpose(A0))*wp.float64(0.5)
    es = es_of(U-wp.identity(n=3, dtype=wp.float64), P[b], mm[b], kap[0])
    LUb, pb = lu9(closure_jac(R, U, es, Mt[b], Mc[b], P[b], mm[b], kap[0]))
    J[b] = LUb
    perm[b] = pb


@wp.kernel(enable_backward=False)
def k_closure_adj(J: wp.array(dtype=mat99), perm: wp.array(dtype=vec9), F_adj: wp.array(dtype=wp.mat33d), S_adj: wp.array(dtype=wp.mat33d)):
    """S_bar += J^-T F_bar, J from k_closure's LU factors (J holds them)"""
    b = wp.tid()
    Fa = F_adj[b]
    rv = vec9()
    for i in range(9):
        rv[i] = Fa[i//3, i % 3]
    lam = lu9_solve_t(J[b], perm[b], rv)
    G = wp.mat33d()
    for i in range(9):
        G[i//3, i % 3] = lam[i]
    wp.atomic_add(S_adj, b, G)


@wp.kernel(enable_backward=False)
def k_closure_np(S: wp.array(dtype=wp.mat33d), Mt: wp.array(dtype=wp.mat33d), Mc: wp.array(dtype=wp.mat33d), P: wp.array(dtype=wp.mat33d),
              mm: wp.array(dtype=wp.mat33d), kap: wp.array(dtype=wp.float64), iters: int, Fw: wp.array(dtype=wp.mat33d),
              F: wp.array(dtype=wp.mat33d), J: wp.array(dtype=mat99), step: wp.array(dtype=wp.float64)):
    b = wp.tid()
    Fb = Fw[b]
    Sb, Mtb, Mcb, Pb, mmb, ka = S[b], Mt[b], Mc[b], P[b], mm[b], kap[0]
    I = wp.identity(n=3, dtype=wp.float64)
    Jb = mat99()
    last = wp.float64(0.0)
    for it in range(iters):
        R = polar12(Fb)
        A0 = wp.transpose(R)@Fb
        U = (A0+wp.transpose(A0))*wp.float64(0.5)
        es = es_of(U-I, Pb, mmb, ka)
        r = Fb@Mtb-Sb-R@(I+es)@Mcb
        T = wp.inverse(wp.trace(U)*I-U)
        for k in range(9):
            dF = wp.mat33d()
            dF[k//3, k % 3] = wp.float64(1.0)
            A = wp.transpose(R)@dF
            K = (A-wp.transpose(A))*wp.float64(0.5)
            w = T@wp.vec3d(K[2, 1], K[0, 2], K[1, 0])*wp.float64(2.0)
            Wm = wp.skew(w)
            dU = (A+wp.transpose(A))*wp.float64(0.5)-(Wm@U-U@Wm)*wp.float64(0.5)
            dFs = R@Wm@(I+es)+R@es_of(dU, Pb, mmb, ka)
            dr = dF@Mtb-dFs@Mcb
            for i in range(9):
                Jb[i, k] = dr[i//3, i % 3]
        rv = vec9()
        for i in range(9):
            rv[i] = -r[i//3, i % 3]
        dx = solve9_np(Jb, rv)
        last = wp.float64(0.0)
        for i in range(9):
            Fb[i//3, i % 3] = Fb[i//3, i % 3]+dx[i]
            last = wp.max(last, wp.abs(dx[i]))
        if last < wp.float64(1e-10):                                                # quadratic: the next step ~1e-20
            break
    F[b] = Fb
    Fw[b] = Fb
    J[b] = Jb
    step[b] = last


@wp.kernel(enable_backward=False)
def k_closure_adj_np(J: wp.array(dtype=mat99), F_adj: wp.array(dtype=wp.mat33d), S_adj: wp.array(dtype=wp.mat33d)):
    b = wp.tid()
    Fa = F_adj[b]
    rv = vec9()
    for i in range(9):
        rv[i] = Fa[i//3, i % 3]
    lam = solve9_np(wp.transpose(J[b]), rv)
    G = wp.mat33d()
    for i in range(9):
        G[i//3, i % 3] = lam[i]
    wp.atomic_add(S_adj, b, G)


# contact of the surface cells' balls with the plane n . x >= c0 (Hertz-Mindlin, as PFC's ball-wall; 2026-10-26, user:
# "一个球和平面的相交根本不用最低点"): the ball of cell c (centre x_B, its Power Foam radius R) overlaps the plane by
# delta = R - (n . x_e - c0), x_e the centre at the step's end (cs x_B - (cs - 1) x_n: cs 2 for the midpoint rule, x_B the
# midpoint), normal energy 8/15 E* R^1/2 delta^5/2 (force 4/3 E* R^1/2 delta^3/2 along n through the centre). Friction (mu
# > 0): the ball held at its centre while it touched at the step's start, Mindlin's k_t = 8 G* (R delta_n)^1/2, the force
# q = lamt - k_t s over the step's slip s = P (x_e - x_n), projected on |f| <= mu F_n (step start), the term
# (psi(q) - |lamt|^2 / 2) / k_t as k_friction. Rows 6 and 8, weight wt (cpar[1])
@wp.kernel(enable_backward=False)
def k_ball(x: wp.array(dtype=wp.vec3d), cbody: wp.array(dtype=int), R: wp.array(dtype=wp.float64), xn: wp.array(dtype=wp.vec3d),
           ng: wp.vec3d, bpar: wp.array(dtype=wp.float64), kt: wp.array(dtype=wp.float64), bound: wp.array(dtype=wp.float64),
           lamt: wp.array(dtype=wp.vec3d), delta: wp.array(dtype=wp.float64), fn: wp.array(dtype=wp.float64),
           ff: wp.array(dtype=wp.vec3d), terms: wp.array2d(dtype=wp.float64)):
    """bpar = (cs, wt, c0, E*, mu)"""
    c = wp.tid()
    B = cbody[c]
    cs, wt, c0, Es, mu = bpar[0], bpar[1], bpar[2], bpar[3], bpar[4]
    xe = cs*x[B]-(cs-wp.float64(1.0))*xn[c]
    d = R[c]-(wp.dot(ng, xe)-c0)
    delta[c] = d
    zero = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
    F = wp.float64(0.0)
    if d > wp.float64(0.0):
        sq = wp.sqrt(R[c]*d)
        F = wp.float64(4.0)/wp.float64(3.0)*Es*sq*d
        wp.atomic_add(terms, 6, c % NSLOT, wt*wp.float64(8.0)/wp.float64(15.0)*Es*sq*d*d)
    fn[c] = F
    f = zero
    if mu > wp.float64(0.0) and bound[c] > wp.float64(0.0):
        dx = xe-xn[c]
        s = dx-wp.dot(dx, ng)*ng
        k = kt[c]
        lt = lamt[c]
        q = lt-k*s
        qn = wp.length(q)
        Rb = bound[c]
        phi = wp.float64(0.0)
        if qn <= Rb:
            f = q
            phi = (wp.dot(q, q)-wp.dot(lt, lt))/(wp.float64(2.0)*k)
        else:
            f = (Rb/qn)*q
            phi = (Rb*qn-wp.float64(0.5)*Rb*Rb-wp.float64(0.5)*wp.dot(lt, lt))/k
        wp.atomic_add(terms, 8, c % NSLOT, wt*phi)
    ff[c] = f


@wp.kernel(enable_backward=False)
def k_ball_adj(cbody: wp.array(dtype=int), ng: wp.vec3d, bpar: wp.array(dtype=wp.float64), fn: wp.array(dtype=wp.float64),
               ff: wp.array(dtype=wp.vec3d), terms_adj: wp.array2d(dtype=wp.float64), x_adj: wp.array(dtype=wp.vec3d)):
    """dE/dx_B = -cs wt (F_n n + f) (the tangential force f in the plane)"""
    c = wp.tid()
    cs, wt = bpar[0], bpar[1]
    a6, a8 = terms_adj[6, c % NSLOT], terms_adj[8, c % NSLOT]
    wp.atomic_add(x_adj, cbody[c], -(cs*wt)*(a6*fn[c]*ng+a8*ff[c]))


@wp.kernel(enable_backward=False)
def k_bonds_fused(x: wp.array(dtype=wp.vec3d), q: wp.array(dtype=wp.vec4d), bi: wp.array(dtype=int), bj: wp.array(dtype=int),
                  ri: wp.array(dtype=wp.vec3d), rj: wp.array(dtype=wp.vec3d), n0: wp.array(dtype=wp.vec3d), k: wp.array(dtype=wp.vec4d),
                  dref: wp.array(dtype=wp.vec3d), thref: wp.array(dtype=wp.vec3d), matp: wp.array(dtype=wp.float64),
                  dp: wp.array(dtype=wp.vec3d), thp: wp.array(dtype=wp.vec3d), tw: wp.array(dtype=wp.float64),
                  terms: wp.array2d(dtype=wp.float64), bd: wp.array(dtype=wp.vec3d), bth: wp.array(dtype=wp.vec3d),
                  x_adj: wp.array(dtype=wp.vec3d), q_adj: wp.array(dtype=wp.vec4d)):
    """k_bonds and k_bonds_adj in one pass (2026-10-26, speed): the loss is sum_t tw_t terms_t, so the bonds' seeds are
    tw[0] and tw[7] before any backward pass; the forward's quaternions, logs and frames are computed once"""
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
    adj = tw[0]
    av = tw[7]*matp[4]
    bd[e] = d
    bth[e] = th
    wp.atomic_add(terms, 0, e % NSLOT, bond_quad(de, te, n, ke))
    wp.atomic_add(terms, 7, e % NSLOT, matp[4]*bond_quad(dv, tv, n, ke))
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



@wp.kernel
def k_F8(S: wp.array(dtype=wp.mat33d), F: wp.array(dtype=wp.mat33d), R: wp.array(dtype=wp.mat33d), q: wp.array(dtype=wp.vec4d)):
    """skin_warp.k_F with Mt^-1 = I and 8 Newton steps for the polar factor (|U - I| 0.5 reaches 1e-16 in 5; speed)"""
    b = wp.tid()
    Fb = S[b]
    Rb = Fb
    for it in range(8):
        Rb = (Rb+wp.transpose(wp.inverse(Rb)))*wp.float64(0.5)
    F[b] = Fb; R[b] = Rb; q[b] = mat_quat(Rb)


class ClosureWarp(SkinWarp):
    """built from a ClosureTerms st; grad(x) -> (loss, dE/dx), torch (prepare() first). Viscosity, contact and friction as
    SkinWarp's (set_viscosity, set_contact), the contact made the cap's series spring and the glue the cap's shear"""
    iters = 3

    def __init__(self, st):
        wp.init()
        self.st = st; m = st.model
        assert m.kinematics == 'exact' and m.body_law == 'nodalvol' and not m.residual
        self.n, self.nb = st.n, len(st.i)
        self.npt, self.ne = 1, 0                                                    # (a dummy skin: k_contact / k_friction take y)
        dev = 'cuda'
        T = lambda t, dt=wp.float64: wp.from_torch(t.contiguous(), dtype=dt)
        I = lambda t: wp.from_torch(t.to(torch.int32).contiguous(), dtype=wp.int32)
        self.bi, self.bj = I(st.i), I(st.j)
        self.A, self.nrm, self.L = T(st.A), T(st.nrm, wp.vec3d), T(st.L)
        self.hi, self.hj, self.wri, self.wrj, self.wt = T(st.hi), T(st.hj), T(st.ri, wp.vec3d), T(st.rj, wp.vec3d), T(st.t, wp.vec3d)
        self.wi, self.wj = T(st.wi), T(st.wj)
        self._kb_t = m.base.clone().contiguous()
        self.mri, self.mrj, self.n0, self.kb = T(m.ri, wp.vec3d), T(m.rj, wp.vec3d), T(m.n0, wp.vec3d), wp.from_torch(self._kb_t, dtype=wp.vec4d)
        self._Mt = (st.Mb+st.Mcap).contiguous(); self._Mc = st.Mcap.contiguous()
        self._P, self._mm = st.P.contiguous(), st.mm.contiguous()
        self.Mt, self.Mc, self.P, self.mmw = T(self._Mt, wp.mat33d), T(self._Mc, wp.mat33d), T(self._P, wp.mat33d), T(self._mm, wp.mat33d)
        self.ident = T(torch.eye(3, dtype=f64, device=st.X.device).repeat(self.n, 1, 1), wp.mat33d)
        self._Fw = torch.eye(3, dtype=f64, device=st.X.device).repeat(self.n, 1, 1).contiguous()
        self.Fw = wp.from_torch(self._Fw, dtype=wp.mat33d)                         # the closure's warm start (the last root)
        self.cb, self.cr, self.cN = I(st.c_body), T(st.c_r), T(st.c_N, wp.mat33d)
        Vdet = st.Vres+torch.zeros_like(st.V).index_add(0, st.c_body, st.c_r*st.c_A/3)
        self.Vdet, self.V = T(Vdet), T(st.V)
        self.lam = float(m.k_v); self.mu = float(st.mu); self.kap = self.lam*self.mu/(self.lam+2*self.mu)
        self.kapw = wp.array([self.lam/(self.lam+2*self.mu)], dtype=wp.float64, device=dev)   # the closure's lam / (lam + 2 mu)
        self.tkap = wp.to_torch(self.kapw, requires_grad=False)
        self._kb_unit = self._kb_t/self.mu                                          # (build's shell=False: kn, ks, kb, kt all ~ mu)
        self.barrier = float(m.barrier_J)
        self._wv_unit = (.5*st.A*st.L/12 if st.variance else torch.zeros_like(st.A)).contiguous()
        self._wv_t = (self.lam*self._wv_unit).contiguous()
        self.wv = T(self._wv_t)
        self.matp = wp.array(np.array([self.mu, 0., self.lam, 0., 0., 0., 0., 0.]), dtype=wp.float64, device=dev)
        self.tmatp = wp.to_torch(self.matp, requires_grad=False)
        self.nc = len(st.c_body)
        self.ct = self.cp = self.vis = self.ball = None
        self._only0, self._only1 = wp.zeros(1, dtype=int, device=dev), wp.array([1], dtype=int, device=dev)
        z = lambda n_, dt_: wp.zeros(n_, dtype=dt_, device=dev)
        self.eta = self.lamv = self.smu = self.slam = 0.
        self.visc_weight = 1.
        self.ref = dict(d=z(self.nb, wp.vec3d), th=z(self.nb, wp.vec3d), P=z(self.n, wp.mat33d), L=z(self.n, wp.float64))
        self.tref = {k_: wp.to_torch(a_, requires_grad=False) for k_, a_ in self.ref.items()}
        self._Pp_t = torch.eye(3, dtype=f64, device=dev).repeat(self.n, 1, 1).contiguous()
        self.plas = dict(dp=z(self.nb, wp.vec3d), thp=z(self.nb, wp.vec3d), Pp=wp.from_torch(self._Pp_t, dtype=wp.mat33d))
        self.tw = wp.array(np.ones(NT), dtype=wp.float64, device=dev)
        self.ttw = wp.to_torch(self.tw, requires_grad=False)
        self.fkappa = 4/3

    def set_viscosity(self, eta, zeta, dt):
        """Kelvin-Voigt on the bonds, o1 and the volume (SkinWarp.set_viscosity without the skin's lateral term)"""
        eta, zeta = float(eta), float(zeta)
        lamv = zeta-2*eta/3
        self.eta, self.lamv, self.smu, self.slam = eta, lamv, eta/self.mu, lamv/self.lam
        self.tmatp[1], self.tmatp[3], self.tmatp[4], self.tmatp[5] = eta, lamv, self.smu, self.slam
        self.vis = dict(eta=eta, zeta=zeta, lamv=lamv, dt=float(dt))
        self.total_mode()

    def set_contact(self, cg, normal=(0., 1., 0.), c0=0., patch_area=None, ent_a=None):
        """SkinWarp.set_contact with the affine caps; rho = the cap's stiffness (lam + 2 mu) A / 3 r, the glue's k_t = 4 mu A / 3 r"""
        SkinWarp.set_contact(self, cg, normal, c0, patch_area, None)
        self.trho.div_(3.)
        self.fkappa = 4/3                                                          # (SkinWarp.set_contact sets its 100)
        self.glue()

    def set_ball(self, cbody, R, normal=(0., 1., 0.), c0=0., mu=0.):
        """Hertz-Mindlin ball contact (k_ball) of the bodies cbody (the surface cells), ball radii R, before prepare();
        E* = E / (1 - nu^2), G* = G / (2 - nu) of the material (per unit density). Torch views: tball (cs, wt, c0, E*, mu),
        tbxn (the step's start centres), tbkt, tbbound, tblamt (the glue's stiffness, bound and held force)"""
        k = len(cbody); dev = 'cuda'
        T = lambda t, dt=wp.float64: wp.from_torch(t.contiguous(), dtype=dt)
        self._ball_moduli()
        z = lambda dt_: wp.zeros(k, dtype=dt_, device=dev)
        self.ball = dict(k=k, body=wp.from_torch(cbody.to(torch.int32).contiguous(), dtype=wp.int32), R=T(R.to(f64)),
                         ng=wp.vec3d(*[float(a_) for a_ in normal]), par=wp.array(np.array([1., 1., float(c0), self.Es, float(mu)]), dtype=wp.float64, device=dev),
                         xn=z(wp.vec3d), kt=z(wp.float64), bound=z(wp.float64), lamt=z(wp.vec3d))
        tt = lambda a_: wp.to_torch(a_, requires_grad=False)
        self.tball, self.tbxn, self.tbkt, self.tbbound, self.tblamt = (tt(self.ball[a_]) for a_ in ('par', 'xn', 'kt', 'bound', 'lamt'))
        self.tbR = tt(self.ball['R'])

    def _ball_moduli(self):
        nu = self.lam/(2*(self.lam+self.mu)); E = 2*self.mu*(1+nu)
        self.Es, self.Gs = E/(1-nu*nu), self.mu/(2-nu)

    def set_material(self, E=None, nu=None, beta=None, rho=1000.):
        """the material in place (no rebuild; the graphs stay; SkinWarp.set_material for this model): E (Pa) and nu as the
        build's (moduli per unit density, rho = pacnerf_dem_forward.RHO), beta the Kelvin-Voigt time (eta = beta mu,
        lam_v = beta lam; None: the current ratios kept). The bonds' stiffnesses scale with mu (build's shell=False), the
        variance with lam; the closure's lam / (lam + 2 mu); the balls' E*, G*. Not here: MidpointStep's blocks and
        constants (its update_material()). The cap contact (set_contact) is not rescaled. The closure's kept LU is rebuilt
        at the warm start (it depends on kappa)"""
        assert self.ct is None, 'set_material: set_contact\'s cap springs are not rescaled (use set_ball)'
        mu0, lam0 = self.mu, self.lam
        nu0, E0 = lam0/(2*(lam0+mu0)), mu0*(3*lam0+2*mu0)/(lam0+mu0)*rho
        E = E0 if E is None else float(E); nu = nu0 if nu is None else float(nu)
        mu, lam = E/(2*(1+nu))/rho, E*nu/((1+nu)*(1-2*nu))/rho
        s, sl = (self.smu, self.slam) if beta is None else (float(beta), float(beta))
        if self.vis is None and (s > 0 or sl > 0):
            raise RuntimeError('set_material: no viscosity (set_viscosity) before, it can not be switched on in place')
        eta, lamv = s*mu, sl*lam
        self._kb_t.copy_(mu*self._kb_unit)
        self._wv_t.copy_(lam*self._wv_unit)
        self.tkap[0] = lam/(lam+2*mu)
        self.tmatp[:6] = torch.tensor([mu, eta, lam, lamv, s, sl], dtype=self.tmatp.dtype, device=self.tmatp.device)
        if self.vis is not None:
            self.vis.update(eta=eta, zeta=lamv+2*eta/3, lamv=lamv)
        self.mu, self.lam, self.kap, self.eta, self.lamv, self.smu, self.slam = mu, lam, lam*mu/(lam+2*mu), eta, lamv, s, sl
        if self.ball is not None:
            self._ball_moduli()
            self.tball[3] = self.Es
        if getattr(self, 'B', None) is not None:                                  # (the closure's kept LU used the old kappa)
            self.closure_lu()

    def _buffers(self):
        z = lambda shape, dt: wp.zeros(shape, dtype=dt, device='cuda', requires_grad=True)
        n = self.n
        B = dict(S=z(n, wp.mat33d), Fc=z(n, wp.mat33d), J=wp.zeros(n, dtype=mat99, device='cuda'), cstep=wp.zeros(n, dtype=wp.float64, device='cuda'),
                 perm=wp.zeros(n, dtype=vec9, device='cuda'),
                 F=z(n, wp.mat33d), R=z(n, wp.mat33d), q=z(n, wp.vec4d), Vd=z(n, wp.float64), lj=z(n, wp.float64), bd=z(self.nb, wp.vec3d),
                 bth=z(self.nb, wp.vec3d), terms=z((NT, 256), wp.float64), tsum=z(NT, wp.float64), loss=z(1, wp.float64))
        if self.ct is not None:
            k = self.ct['k']
            B.update(gap=z(k, wp.float64), Xs=z(k, wp.vec3d), dphi=z(k, wp.float64), cpos=z(k, wp.vec3d), slip=z(k, wp.vec3d), ff=z(k, wp.vec3d))
        if self.ball is not None:
            k = self.ball['k']
            B.update(bdelta=z(k, wp.float64), bfn=z(k, wp.float64), bff=z(k, wp.vec3d))
        return B

    def _forward(self, x, y, B, tape=None, only=None):
        only = self._only0 if only is None else only
        n = self.n
        rf, pl = self.ref, self.plas
        wp.launch(k_S_bonds, self.nb, inputs=[x, self.bi, self.bj, self.A, self.nrm], outputs=[B['S']])
        wp.launch(k_closure, n, inputs=[B['S'], self.Mt, self.Mc, self.P, self.mmw, self.kapw, self.iters, self.Fw], outputs=[B['Fc'], B['J'], B['cstep'], B['perm']],
                  record_tape=False)
        if tape is not None:
            tape.record_func(lambda: wp.launch(k_closure_adj, n, inputs=[B['J'], B['perm'], B['Fc'].grad], outputs=[B['S'].grad]), arrays=[B['S'], B['Fc']])
        wp.launch(k_F8, n, inputs=[B['Fc']], outputs=[B['F'], B['R'], B['q']])
        hand = self.hand_adjoints and tape is not None
        bonds_in = [x, B['q'], self.bi, self.bj, self.mri, self.mrj, self.n0, self.kb, rf['d'], rf['th'], self.matp, pl['dp'], pl['thp']]
        if hand:                                                                    # (the gradient pass: forward and adjoint fused)
            wp.launch(k_bonds_fused, self.nb, inputs=bonds_in+[self.tw], outputs=[B['terms'], B['bd'], B['bth'], x.grad, B['q'].grad], record_tape=False)
        else:
            wp.launch(k_bonds, self.nb, inputs=bonds_in, outputs=[B['terms'], B['bd'], B['bth']])
        walls_in = [x, B['F'], B['R'], self.bi, self.bj, self.hi, self.hj, self.L, self.wri, self.wrj, self.wt, self.A, self.nrm, self.wi, self.wj]
        wp.launch(k_walls, self.nb, inputs=walls_in, outputs=[B['Vd']], record_tape=not hand)
        if hand:
            tape.record_func(lambda: wp.launch(k_walls_adj, self.nb, inputs=walls_in+[B['Vd'].grad], outputs=[x.grad, B['F'].grad, B['R'].grad]),
                             arrays=[x, B['F'], B['R'], B['Vd']])
        wp.launch(k_patches, self.nc, inputs=[B['F'], B['R'], self.cb, self.cr, self.cN, self.matp, rf['P'], pl['Pp']], outputs=[B['terms']])
        wp.launch(k_volume, n, inputs=[B['F'], B['Vd'], self.Vdet, self.V, self.matp, rf['L'], self.barrier], outputs=[B['lj'], B['terms']])
        wp.launch(k_variance, self.nb, inputs=[B['lj'], rf['L'], self.bi, self.bj, self.wv, self.matp], outputs=[B['terms']])
        ct = self.ct
        if ct is not None:
            sh = [ct['shift'], ct['ceoff'], ct['cepid'], ct['cew']]
            wp.launch(k_contact, ct['k'], inputs=[x, y, B['F'], ct['X']]+sh+[ct['dbar'], ct['body'], ct['p'], ct['r'], ct['aoff'], ct['cc'], ct['rho_a'],
                                              ct['u'], ct['v'], ct['t0'], ct['t1'], ct['noff'], ct['nbp'], ct['nbr'], ct['ng'], ct['c0'], ct['lam'],
                                              ct['rho'], 1, ct['cpar'], ct['goff'], ct['act'], only],
                      outputs=[B['gap'], B['Xs'], B['cpos'], B['dphi'], B['terms']], record_tape=False)
            if tape is not None:
                tape.record_func(lambda: wp.launch(k_contact_adj, ct['k'], inputs=[ct['body'], B['Xs'], B['dphi'], ct['ng']]+sh+[ct['cpar'], ct['act'], only, B['terms'].grad],
                                                   outputs=[x.grad, y.grad, B['F'].grad]), arrays=[x, y, B['F'], B['terms']])
            wp.launch(k_friction, ct['k'], inputs=[x, y, B['F']]+sh+[ct['body'], ct['ng'], ct['wfix'], ct['pn'], ct['lamt'], ct['rhot'], ct['lamR'],
                                               ct['fpar'], ct['cpar'], ct['act'], only, 1],
                      outputs=[B['slip'], B['ff'], B['terms']], record_tape=False)
            if tape is not None:
                tape.record_func(lambda: wp.launch(k_friction_adj, ct['k'], inputs=[ct['body'], ct['wfix'], B['ff']]+sh+[ct['fpar'], ct['cpar'], ct['lamR'], B['terms'].grad],
                                                   outputs=[x.grad, y.grad, B['F'].grad]), arrays=[x, y, B['F'], B['terms']])
        bl = self.ball
        if bl is not None:
            wp.launch(k_ball, bl['k'], inputs=[x, bl['body'], bl['R'], bl['xn'], bl['ng'], bl['par'], bl['kt'], bl['bound'], bl['lamt']],
                      outputs=[B['bdelta'], B['bfn'], B['bff'], B['terms']], record_tape=False)
            if tape is not None:
                tape.record_func(lambda: wp.launch(k_ball_adj, bl['k'], inputs=[bl['body'], bl['ng'], bl['par'], B['bfn'], B['bff'], B['terms'].grad],
                                                   outputs=[x.grad]), arrays=[x, B['terms']])
        wp.launch(k_total, (NT, 256), inputs=[B['terms']], outputs=[B['tsum']], record_tape=False)
        wp.launch(k_total_loss, 1, inputs=[B['tsum'], self.tw], outputs=[B['loss']], record_tape=False)
        if tape is not None:
            tape.record_func(lambda: wp.launch(k_total_adj, (NT, 256), inputs=[B['tsum'].grad, B['loss'].grad, self.tw], outputs=[B['terms'].grad]),
                             arrays=[B['terms'], B['tsum'], B['loss']])

    def prepare(self, graph=True):
        """the gradient pass (zero, forward, backward) as one CUDA graph; grad(x) copies x in and replays it"""
        self.B = self._buffers()
        self.cold()
        self.xw = wp.zeros(self.n, dtype=wp.vec3d, device='cuda', requires_grad=True)
        self.yw = wp.zeros(self.npt, dtype=wp.vec3d, device='cuda', requires_grad=True)
        wp.to_torch(self.xw, requires_grad=False).copy_(self.st.X)                  # (the warm-up runs at rest: at x = 0 the
        self.tape = wp.Tape()                                                       # closure is degenerate and its NaN would
                                                                                    # stay in the warm start)
        def run(products=False):
            self.tape.reset()
            for k_, a in self.B.items():
                if a.grad is not None:
                    a.grad.zero_()
                if k_ in ('S', 'Vd', 'terms', 'tsum'):
                    a.zero_()
            self.xw.grad.zero_(); self.yw.grad.zero_()
            with self.tape:
                self._forward(self.xw, self.yw, self.B, self.tape, self._only1 if products else self._only0)
            self.tape.backward(loss=self.B['loss'])
        self._run = run
        run(); self.tape.reset()
        self.graph = None
        if graph:
            wp.capture_begin()
            run()
            self.graph = wp.capture_end()
        else:
            self.run = lambda: (run(), self.tape.reset())
        self.cold()
        tt = lambda a: wp.to_torch(a, requires_grad=False)
        self.tx, self.tgx, self.tloss = tt(self.xw), tt(self.xw.grad), tt(self.B['loss'])
        self.tR, self.tF, self.tsum, self.tlj = tt(self.B['R']), tt(self.B['F']), tt(self.B['tsum']), tt(self.B['lj'])
        self.tbd, self.tbth, self.tcstep = tt(self.B['bd']), tt(self.B['bth']), tt(self.B['cstep'])
        if self.ct is not None:
            self.tgap, self.tXs = tt(self.B['gap']), tt(self.B['Xs'])
            self.tcpos, self.tslip, self.tff = tt(self.B['cpos']), tt(self.B['slip']), tt(self.B['ff'])
        if self.ball is not None:
            self.tbdelta, self.tbfn, self.tbff = tt(self.B['bdelta']), tt(self.B['bfn']), tt(self.B['bff'])

    def cold(self):
        """the closure's warm start reset to I (rest) with its Jacobian's LU"""
        self._Fw.copy_(torch.eye(3, dtype=f64, device=self._Fw.device).repeat(self.n, 1, 1))
        self.closure_lu()

    def closure_lu(self):
        """the LU kept with the warm start rebuilt at it (after a reset, or kappa's change in set_material)"""
        wp.launch(k_closure_lu, self.n, inputs=[self.Fw, self.Mt, self.Mc, self.P, self.mmw, self.kapw], outputs=[self.B['J'], self.B['perm']])

    def grad(self, x):
        self.tx.copy_(x)
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.run()
        return self.tloss[0].clone(), self.tgx.clone()

    def measures(self):
        """the viscosity's measures of the last grad call: bond d, theta; R^T F; log J"""
        return dict(d=self.tbd.clone(), th=self.tbth.clone(), P=self.tR.transpose(1, 2)@self.tF, L=self.tlj.clone())

    def visc_ref(self, x):
        """the measures at x become the references; returns grad's (energy, dE/dx) there without the viscous term"""
        self.elastic_mode()
        out = self.grad(x)
        self.total_mode()
        ms = self.measures()
        for k_, r_ in self.tref.items():
            r_.copy_(ms[k_])
        return out


# CG of the midpoint step on the GPU (2026-10-26): each iteration's perturbation, gradient pass, product, step and dots as
# kernels, several iterations captured as one CUDA graph; the host reads the residual between graphs. Scalars in sc:
SC_PMAX, SC_PHP, SC_RY, SC_RYN, SC_RR, SC_AL, SC_BE, SC_NEG = range(8)


@wp.kernel(enable_backward=False)
def k_pcg_absmax(p: wp.array(dtype=wp.vec3d), sc: wp.array(dtype=wp.float64)):
    i = wp.tid()
    a = p[i]
    wp.atomic_max(sc, SC_PMAX, wp.max(wp.abs(a[0]), wp.max(wp.abs(a[1]), wp.abs(a[2]))))


@wp.kernel(enable_backward=False)
def k_pcg_perturb(z: wp.array(dtype=wp.vec3d), p: wp.array(dtype=wp.vec3d), sc: wp.array(dtype=wp.float64), h: wp.float64,
                  xw: wp.array(dtype=wp.vec3d)):
    i = wp.tid()
    xw[i] = z[i]+(h/wp.max(sc[SC_PMAX], wp.float64(1e-300)))*p[i]


@wp.kernel(enable_backward=False)
def k_pcg_hd(gE: wp.array(dtype=wp.vec3d), g0: wp.array(dtype=wp.vec3d), p: wp.array(dtype=wp.vec3d), cm: wp.array(dtype=wp.float64),
             sc: wp.array(dtype=wp.float64), h: wp.float64, Hp: wp.array(dtype=wp.vec3d)):
    """H p = (4 m / dt^2) p + (dE/dx(z + h p / |p|_max) - dE/dx(z)) |p|_max / h; p . H p into sc"""
    i = wp.tid()
    hp = cm[i]*p[i]+(gE[i]-g0[i])*(wp.max(sc[SC_PMAX], wp.float64(1e-300))/h)
    Hp[i] = hp
    wp.atomic_add(sc, SC_PHP, wp.dot(p[i], hp))


@wp.kernel(enable_backward=False)
def k_pcg_alpha(sc: wp.array(dtype=wp.float64)):
    if sc[SC_PHP] > wp.float64(0.0):
        sc[SC_AL] = sc[SC_RY]/sc[SC_PHP]
    else:
        sc[SC_AL] = wp.float64(0.0)
        sc[SC_NEG] = wp.float64(1.0)
    sc[SC_RYN] = wp.float64(0.0)
    sc[SC_RR] = wp.float64(0.0)


@wp.kernel(enable_backward=False)
def k_pcg_update(p: wp.array(dtype=wp.vec3d), Hp: wp.array(dtype=wp.vec3d), Pinv: wp.array(dtype=wp.mat33d), sc: wp.array(dtype=wp.float64),
                 dz: wp.array(dtype=wp.vec3d), r: wp.array(dtype=wp.vec3d), y: wp.array(dtype=wp.vec3d)):
    i = wp.tid()
    al = sc[SC_AL]
    dz[i] = dz[i]+al*p[i]
    ri = r[i]-al*Hp[i]
    r[i] = ri
    yi = Pinv[i]@ri
    y[i] = yi
    wp.atomic_add(sc, SC_RYN, wp.dot(ri, yi))
    wp.atomic_add(sc, SC_RR, wp.dot(ri, ri))


@wp.kernel(enable_backward=False)
def k_pcg_beta(sc: wp.array(dtype=wp.float64)):
    sc[SC_BE] = sc[SC_RYN]/wp.max(sc[SC_RY], wp.float64(1e-300))
    sc[SC_RY] = sc[SC_RYN]
    sc[SC_PMAX] = wp.float64(0.0)
    sc[SC_PHP] = wp.float64(0.0)


@wp.kernel(enable_backward=False)
def k_pcg_dir(y: wp.array(dtype=wp.vec3d), sc: wp.array(dtype=wp.float64), p: wp.array(dtype=wp.vec3d)):
    i = wp.tid()
    p[i] = y[i]+sc[SC_BE]*p[i]


class ClosureCG:
    """preconditioned CG for the midpoint step's Newton system (H = 4 M / dt^2 + K by differences of W's gradient),
    per captured graph `per` iterations; W prepared (its gradient pass reused inside the graph). solve(z, rhs, g0, Pinv,
    tol_abs, maxit) -> (dz, iterations, negative curvature seen); torch tensors in, dz a torch view (copy it); or start()
    and run() on MidpointStep's device arrays"""
    def __init__(self, W, mass, dt, per=2, h=1e-9, product_iters=None):
        self.W, self.per, self.h = W, per, float(h)
        it0 = W.iters
        if product_iters is not None:                                              # (the closure's iterations in the
            W.iters = product_iters                                                # graphs below: one step = exact to h^2)
        n = W.n; dev = 'cuda'
        z = lambda dt_: wp.zeros(n, dtype=dt_, device=dev)
        self.a = dict(z=z(wp.vec3d), p=z(wp.vec3d), r=z(wp.vec3d), y=z(wp.vec3d), dz=z(wp.vec3d), Hp=z(wp.vec3d), g0=z(wp.vec3d),
                      Pinv=z(wp.mat33d), cm=wp.from_torch((4*mass.to(f64)/dt**2).contiguous(), dtype=wp.float64),
                      sc=wp.zeros(8, dtype=wp.float64, device=dev))
        self.t = {k_: wp.to_torch(a_, requires_grad=False) for k_, a_ in self.a.items()}

        def iteration():
            A = self.a
            wp.launch(k_pcg_absmax, n, inputs=[A['p']], outputs=[A['sc']])
            wp.launch(k_pcg_perturb, n, inputs=[A['z'], A['p'], A['sc'], self.h], outputs=[W.xw])
            W._run()                                                              # (no tape.reset here: it zeroes the gradients)
            wp.launch(k_pcg_hd, n, inputs=[W.xw.grad, A['g0'], A['p'], A['cm'], A['sc'], self.h], outputs=[A['Hp']])
            wp.launch(k_pcg_alpha, 1, inputs=[A['sc']])
            wp.launch(k_pcg_update, n, inputs=[A['p'], A['Hp'], A['Pinv'], A['sc']], outputs=[A['dz'], A['r'], A['y']])
            wp.launch(k_pcg_beta, 1, inputs=[A['sc']])
            wp.launch(k_pcg_dir, n, inputs=[A['y'], A['sc']], outputs=[A['p']])
        self.t['z'].copy_(W.st.X)                                                  # (the warm-up at rest: at x = 0 the closure's
        iteration()                                                                # NaN would stay in its warm start; compile
        wp.capture_begin()                                                         # outside the capture)
        for _ in range(per):
            iteration()
        self.graph = wp.capture_end()
        W.iters = it0
        W.cold()

    def solve(self, z, rhs, g0, Pinv, tol_abs, maxit=300):
        t = self.t
        t['z'].copy_(z); t['g0'].copy_(g0); t['Pinv'].copy_(Pinv); t['r'].copy_(rhs)
        y = torch.einsum('bij,bj->bi', Pinv, rhs)
        t['y'].copy_(y); t['p'].copy_(y); t['dz'].zero_()
        sc = t['sc']; sc.zero_(); sc[SC_RY] = (rhs*y).sum()
        it = 0
        while it < maxit:
            wp.capture_launch(self.graph)
            it += self.per
            rr, neg = sc[SC_RR].item(), sc[SC_NEG].item()
            if neg > 0 or math.sqrt(max(rr, 0.)) <= tol_abs:
                break
        return t['dz'], it, neg > 0

    def start(self, g, sgn):
        """solve's set-up on the device for rhs = sgn g (a Warp array); z, g0 and Pinv already in self.a (MidpointStep)"""
        A = self.a
        A['sc'].zero_()
        wp.launch(k_pcg_init, self.W.n, inputs=[g, float(sgn), A['Pinv']], outputs=[A['r'], A['y'], A['p'], A['dz'], A['sc']])

    def run(self, tol_abs, maxit=300):
        """solve's iterations after start(): (iterations, negative curvature seen); one host read per graph"""
        sc = self.t['sc']
        it = 0
        while it < maxit:
            wp.capture_launch(self.graph)
            it += self.per
            rr, _, _, neg = sc[SC_RR:SC_NEG+1].tolist()
            if neg > 0 or math.sqrt(max(rr, 0.)) <= tol_abs:
                break
        return it, neg > 0


@wp.kernel(enable_backward=False)
def k_pcg_init(g: wp.array(dtype=wp.vec3d), sgn: wp.float64, Pinv: wp.array(dtype=wp.mat33d), r: wp.array(dtype=wp.vec3d),
               y: wp.array(dtype=wp.vec3d), p: wp.array(dtype=wp.vec3d), dz: wp.array(dtype=wp.vec3d), sc: wp.array(dtype=wp.float64)):
    i = wp.tid()
    ri = sgn*g[i]
    yi = Pinv[i]@ri
    r[i] = ri; y[i] = yi; p[i] = yi
    dz[i] = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
    wp.atomic_add(sc, SC_RY, wp.dot(ri, yi))


# the midpoint step's own work on the GPU (2026-10-26, user: "先只做纯工程"): hertz_drop.py's per-step torch work (predictor,
# the balls' step-start data, Pi and its gradient, the block preconditioner, the line search's moves, the step's end and
# the viscosity's references) as kernels, the gradient pass with Pi's assembly as one graph; the host reads the scalars ps
# once after each Pi evaluation. Slots of ps:
PS_T, PS_PZ, PS_G2, PS_PQ, PS_GQ2, PS_SLOPE, PS_DZMAX, PS_VMAX = range(8)


@wp.kernel(enable_backward=False)
def k_mid_set(ps: wp.array(dtype=wp.float64), k: int, val: wp.float64):
    ps[k] = val


@wp.kernel(enable_backward=False)
def k_mid_begin(x: wp.array(dtype=wp.vec3d), v: wp.array(dtype=wp.vec3d), a: wp.array(dtype=wp.vec3d), dt: wp.float64,
                xt: wp.array(dtype=wp.vec3d), z: wp.array(dtype=wp.vec3d)):
    """the predictor: xt = x + dt v / 2, z = xt + dt^2 a / 4"""
    i = wp.tid()
    xti = x[i]+(dt/wp.float64(2.0))*v[i]
    xt[i] = xti
    z[i] = xti+(dt*dt/wp.float64(4.0))*a[i]


@wp.kernel(enable_backward=False)
def k_mid_ball(x: wp.array(dtype=wp.vec3d), cbody: wp.array(dtype=int), R: wp.array(dtype=wp.float64), ng: wp.vec3d, c0: wp.float64,
               g8: wp.float64, cb: wp.float64, xn: wp.array(dtype=wp.vec3d), kt: wp.array(dtype=wp.float64), bound: wp.array(dtype=wp.float64)):
    """the balls' step-start data: centre, Mindlin k_t = 8 G* (R delta_n)^1/2 (g8 = 8 G*), bound mu F_n (cb = 4 / 3 mu E*)"""
    c = wp.tid()
    xc = x[cbody[c]]
    dn = wp.max(R[c]-(wp.dot(ng, xc)-c0), wp.float64(0.0))
    sq = wp.sqrt(R[c]*dn)
    xn[c] = xc
    kt[c] = g8*sq
    bound[c] = cb*sq*dn


@wp.kernel(enable_backward=False)
def k_mid_query(z: wp.array(dtype=wp.vec3d), dz: wp.array(dtype=wp.vec3d), ps: wp.array(dtype=wp.float64), xw: wp.array(dtype=wp.vec3d)):
    """the point of a Pi evaluation: z + t dz (t = ps[PS_T]; z itself at t = 0); zeroes Pi's sums"""
    i = wp.tid()
    t = ps[PS_T]
    if t == wp.float64(0.0):
        xw[i] = z[i]
    else:
        xw[i] = z[i]+t*dz[i]
    if i == 0:
        ps[PS_PQ] = wp.float64(0.0)
        ps[PS_GQ2] = wp.float64(0.0)


@wp.kernel(enable_backward=False)
def k_mid_pi(zq: wp.array(dtype=wp.vec3d), xt: wp.array(dtype=wp.vec3d), x: wp.array(dtype=wp.vec3d), cm: wp.array(dtype=wp.float64),
             fg: wp.array(dtype=wp.vec3d), gE: wp.array(dtype=wp.vec3d), loss: wp.array(dtype=wp.float64), gz: wp.array(dtype=wp.vec3d),
             g0: wp.array(dtype=wp.vec3d), ps: wp.array(dtype=wp.float64)):
    """Pi(zq) = sum cm |zq - xt|^2 / 2 - fg . (zq - x) + E (cm = 4 m / dt^2) into ps[PS_PQ], its gradient into gz (|.|^2 into
    ps[PS_GQ2]) and E's into g0 (the last evaluation is always the accepted one)"""
    i = wp.tid()
    d = zq[i]-xt[i]
    g = cm[i]*d-fg[i]+gE[i]
    gz[i] = g
    g0[i] = gE[i]
    e = wp.float64(0.5)*cm[i]*wp.dot(d, d)-wp.dot(fg[i], zq[i]-x[i])
    if i == 0:
        e = e+loss[0]
    wp.atomic_add(ps, PS_PQ, e)
    wp.atomic_add(ps, PS_GQ2, wp.dot(g, g))


@wp.kernel(enable_backward=False)
def k_mid_accept(zq: wp.array(dtype=wp.vec3d), ps: wp.array(dtype=wp.float64), z: wp.array(dtype=wp.vec3d)):
    i = wp.tid()
    z[i] = zq[i]
    if i == 0:
        ps[PS_PZ] = ps[PS_PQ]
        ps[PS_G2] = ps[PS_GQ2]
        ps[PS_SLOPE] = wp.float64(0.0)
        ps[PS_DZMAX] = wp.float64(0.0)


@wp.kernel(enable_backward=False)
def k_mid_pre(R: wp.array(dtype=wp.mat33d), K: wp.array(dtype=wp.mat33d), c1: wp.float64, cm: wp.array(dtype=wp.float64),
              P: wp.array(dtype=wp.mat33d)):
    """the bodies' blocks R K_bb R^T (1 + 2 beta / dt) + 4 m / dt^2"""
    b = wp.tid()
    P[b] = R[b]@K[b]@wp.transpose(R[b])*c1+cm[b]*wp.identity(n=3, dtype=wp.float64)


@wp.kernel(enable_backward=False)
def k_mid_pre_ball(cbody: wp.array(dtype=int), R: wp.array(dtype=wp.float64), delta: wp.array(dtype=wp.float64), ng: wp.vec3d,
                   e4: wp.float64, kt: wp.array(dtype=wp.float64), bound: wp.array(dtype=wp.float64), mu: wp.float64,
                   P: wp.array(dtype=wp.mat33d)):
    """the touching balls' 2 k_n n n^T (k_n = 2 E* (R delta)^1/2; e4 = 4 E*) and the glued ones' 2 k_t (I - n n^T)"""
    c = wp.tid()
    nn = wp.outer(ng, ng)
    M = (e4*wp.sqrt(R[c]*wp.max(delta[c], wp.float64(0.0))))*nn
    if mu > wp.float64(0.0) and bound[c] > wp.float64(0.0):
        M = M+(wp.float64(2.0)*kt[c])*(wp.identity(n=3, dtype=wp.float64)-nn)
    wp.atomic_add(P, cbody[c], M)


@wp.kernel(enable_backward=False)
def k_mid_inv(P: wp.array(dtype=wp.mat33d), Pinv: wp.array(dtype=wp.mat33d)):
    b = wp.tid()
    Pinv[b] = wp.inverse(P[b])


@wp.kernel(enable_backward=False)
def k_mid_absmax(dz: wp.array(dtype=wp.vec3d), ps: wp.array(dtype=wp.float64)):
    i = wp.tid()
    a = dz[i]
    wp.atomic_max(ps, PS_DZMAX, wp.max(wp.abs(a[0]), wp.max(wp.abs(a[1]), wp.abs(a[2]))))


@wp.kernel(enable_backward=False)
def k_mid_slope(gz: wp.array(dtype=wp.vec3d), Pinv: wp.array(dtype=wp.mat33d), ps: wp.array(dtype=wp.float64), dz: wp.array(dtype=wp.vec3d)):
    """dz = P^-1 (-g) if CG gave nothing; the slope g . dz into ps[PS_SLOPE]"""
    i = wp.tid()
    d = dz[i]
    if not (ps[PS_DZMAX] > wp.float64(0.0)):
        d = Pinv[i]@(-gz[i])
        dz[i] = d
    wp.atomic_add(ps, PS_SLOPE, wp.dot(gz[i], d))


@wp.kernel(enable_backward=False)
def k_mid_visc_bonds(bd: wp.array(dtype=wp.vec3d), bth: wp.array(dtype=wp.vec3d), rd: wp.array(dtype=wp.vec3d), rth: wp.array(dtype=wp.vec3d)):
    """the viscosity's references at the step's end: 2 (midpoint measure) - start"""
    e = wp.tid()
    rd[e] = wp.float64(2.0)*bd[e]-rd[e]
    rth[e] = wp.float64(2.0)*bth[e]-rth[e]


@wp.kernel(enable_backward=False)
def k_mid_visc_bodies(R: wp.array(dtype=wp.mat33d), F: wp.array(dtype=wp.mat33d), lj: wp.array(dtype=wp.float64),
                      rP: wp.array(dtype=wp.mat33d), rL: wp.array(dtype=wp.float64)):
    b = wp.tid()
    rP[b] = wp.float64(2.0)*(wp.transpose(R[b])@F[b])-rP[b]
    rL[b] = wp.float64(2.0)*lj[b]-rL[b]


@wp.kernel(enable_backward=False)
def k_mid_lamt(ff: wp.array(dtype=wp.vec3d), delta: wp.array(dtype=wp.float64), lamt: wp.array(dtype=wp.vec3d)):
    """the glue's force kept to the next step while the ball touches"""
    c = wp.tid()
    if delta[c] > wp.float64(0.0):
        lamt[c] = ff[c]
    else:
        lamt[c] = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))


@wp.kernel(enable_backward=False)
def k_mid_end(z: wp.array(dtype=wp.vec3d), dt: wp.float64, m: wp.array(dtype=wp.float64), x: wp.array(dtype=wp.vec3d),
              v: wp.array(dtype=wp.vec3d), a: wp.array(dtype=wp.vec3d), mv: wp.array(dtype=wp.vec3d)):
    """x1 = 2 z - x, v1 = 4 (z - x) / dt - v, a = (v1 - v) / dt; sum m v1 into mv"""
    i = wp.tid()
    xi, vi, zi = x[i], v[i], z[i]
    v1 = (zi-xi)*(wp.float64(4.0)/dt)-vi
    x[i] = wp.float64(2.0)*zi-xi
    a[i] = (v1-vi)/dt
    v[i] = v1
    wp.atomic_add(mv, 0, m[i]*v1)


@wp.kernel(enable_backward=False)
def k_mid_vmax(mv: wp.array(dtype=wp.vec3d), mtot: wp.float64, ps: wp.array(dtype=wp.float64)):
    """the largest upward (y) velocity of the centre of mass so far; zeroes mv"""
    ps[PS_VMAX] = wp.max(ps[PS_VMAX], mv[0][1]/mtot)
    mv[0] = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))


class MidpointStep:
    """hertz_drop.py's implicit-midpoint step of a ClosureWarp W with ball contact (set_ball) with its work on the device:
    x, v, a (torch views tx, tv, ta), the glue's held force and the viscosity's references stay there; CG = ClosureCG(W, ...)
    (its z, g0, Pinv and dz are this step's); Kbb the bodies' rest blocks (closure_drop.probe_blocks), fg the external
    forces, muf the ground's friction. step(tol_abs, cg_tol_abs) -> (gradient evaluations, Newton iterations, |grad Pi|^2),
    then end(); after W.set_material, update_material()"""
    def __init__(self, W, CG, mass, Kbb, fg, dt, muf):
        self.W, self.CG, self.dt = W, CG, float(dt)
        n = W.n; dev = 'cuda'
        z = lambda dt_, k=n: wp.zeros(k, dtype=dt_, device=dev)
        A = CG.a
        self.a = dict(x=z(wp.vec3d), v=z(wp.vec3d), a=z(wp.vec3d), xt=z(wp.vec3d), gz=z(wp.vec3d), P=z(wp.mat33d), mv=z(wp.vec3d, 1),
                      m=wp.from_torch(mass.to(f64).contiguous(), dtype=wp.float64), cm=A['cm'], z=A['z'], g0=A['g0'], Pinv=A['Pinv'], dz=A['dz'],
                      K=wp.from_torch(Kbb.to(f64).contiguous(), dtype=wp.mat33d), fg=wp.from_torch(fg.to(f64).contiguous(), dtype=wp.vec3d))
        self.ps = wp.zeros(8, dtype=wp.float64, device=dev)
        tt = lambda a_: wp.to_torch(a_, requires_grad=False)
        self.tx, self.tv, self.ta, self.tps, self.tK = tt(self.a['x']), tt(self.a['v']), tt(self.a['a']), tt(self.ps), tt(self.a['K'])
        self.K0, self.mu0 = self.tK.clone(), W.mu
        self.mtot = float(mass.sum()); self.c0 = float(W.tball[2])
        self.update_material(muf)

        def pi():
            A = self.a
            wp.launch(k_mid_query, n, inputs=[A['z'], A['dz'], self.ps], outputs=[W.xw])
            W._run()
            wp.launch(k_mid_pi, n, inputs=[W.xw, A['xt'], A['x'], A['cm'], A['fg'], W.xw.grad, W.B['loss']], outputs=[A['gz'], A['g0'], self.ps])
        tt(self.a['z']).copy_(W.st.X); tt(self.a['x']).copy_(W.st.X); tt(self.a['xt']).copy_(W.st.X)
        pi()                                                                        # (compile at rest, outside the capture)
        wp.capture_begin()
        pi()
        self.graph = wp.capture_end()
        W.cold()
        self.tps.zero_(); self.tps[PS_VMAX] = -1e9

    def update_material(self, muf=None):
        """W's material (after W.set_material) into the step: the rest blocks scaled with mu (nu's change not in them: the
        solve's pace only, as identify.py's), the viscous factor 1 + 2 beta / dt, the balls' 8 G*, 4 E*; muf: the friction"""
        W = self.W
        if muf is not None:
            self.muf = float(muf); W.tball[4] = self.muf
        self.tK.copy_(self.K0*(W.mu/self.mu0))
        self.c1 = 1+2*W.smu/self.dt
        self.g8, self.cb, self.e4 = 8*W.Gs, self.muf*4/3*W.Es, 4*W.Es

    def set_state(self, x, v, a):
        self.tx.copy_(x); self.tv.copy_(v); self.ta.copy_(a)

    def _eval(self, t):
        wp.launch(k_mid_set, 1, inputs=[self.ps, PS_T, float(t)])
        wp.capture_launch(self.graph)
        return self.tps.tolist()

    def step(self, tol_abs, cg_tol_abs):
        W, A, bl, n = self.W, self.a, self.W.ball, self.W.n
        wp.launch(k_mid_begin, n, inputs=[A['x'], A['v'], A['a'], self.dt], outputs=[A['xt'], A['z']])
        wp.launch(k_mid_ball, bl['k'], inputs=[A['x'], bl['body'], bl['R'], bl['ng'], self.c0, self.g8, self.cb], outputs=[bl['xn'], bl['kt'], bl['bound']])
        g2 = self._eval(0.)[PS_GQ2]
        wp.launch(k_mid_accept, n, inputs=[W.xw, self.ps], outputs=[A['z']])
        prods, its = 1, 0
        while its < 50:
            if math.sqrt(g2) <= tol_abs:
                break
            its += 1
            wp.launch(k_mid_pre, n, inputs=[W.B['R'], A['K'], self.c1, A['cm']], outputs=[A['P']])
            wp.launch(k_mid_pre_ball, bl['k'], inputs=[bl['body'], bl['R'], W.B['bdelta'], bl['ng'], self.e4, bl['kt'], bl['bound'], self.muf],
                      outputs=[A['P']])
            wp.launch(k_mid_inv, n, inputs=[A['P']], outputs=[A['Pinv']])
            self.CG.start(A['gz'], -1.)
            k_, _ = self.CG.run(cg_tol_abs(g2))
            prods += k_
            wp.launch(k_mid_absmax, n, inputs=[A['dz']], outputs=[self.ps])
            wp.launch(k_mid_slope, n, inputs=[A['gz'], A['Pinv'], self.ps], outputs=[A['dz']])
            t = 1.
            while True:
                s = self._eval(t); prods += 1
                ok = math.isfinite(s[PS_PQ]) and math.isfinite(s[PS_GQ2])
                if ok and (s[PS_PQ] <= s[PS_PZ]+1e-4*t*s[PS_SLOPE] or t < 1e-6):
                    break
                if not ok and t < 1e-6:                                           # (a non-finite trial is never taken)
                    return prods, its, math.nan
                t *= .5
            wp.launch(k_mid_accept, n, inputs=[W.xw, self.ps], outputs=[A['z']])
            g2 = s[PS_GQ2]
        return prods, its, g2

    def end(self):
        """the step's end at the accepted midpoint (W's buffers hold its evaluation)"""
        W, A, bl, n = self.W, self.a, self.W.ball, self.W.n
        if W.vis is not None:
            wp.launch(k_mid_visc_bonds, W.nb, inputs=[W.B['bd'], W.B['bth']], outputs=[W.ref['d'], W.ref['th']])
            wp.launch(k_mid_visc_bodies, n, inputs=[W.B['R'], W.B['F'], W.B['lj']], outputs=[W.ref['P'], W.ref['L']])
        wp.launch(k_mid_lamt, bl['k'], inputs=[W.B['bff'], W.B['bdelta']], outputs=[bl['lamt']])
        wp.launch(k_mid_end, n, inputs=[A['z'], self.dt, A['m']], outputs=[A['x'], A['v'], A['a'], A['mv']])
        wp.launch(k_mid_vmax, 1, inputs=[A['mv'], self.mtot], outputs=[self.ps])

    def vmax(self):
        return self.tps[PS_VMAX].item()
