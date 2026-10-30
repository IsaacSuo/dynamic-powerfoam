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
           (now scheme A's coupled closure, k_closA_*; set_cap_positions: the one-way cap k_closP, F_s from the positions)
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
def frob2(M: wp.mat33d):
    a = wp.float64(0.0)
    for i in range(3):
        for j in range(3):
            a = a+M[i, j]*M[i, j]
    return a


@wp.func
def null_row_cross(M: wp.mat33d):
    """the unit vector of the largest cross product of two rows of M (M's near null vector)"""
    m0 = wp.vec3d(M[0, 0], M[0, 1], M[0, 2])
    m1 = wp.vec3d(M[1, 0], M[1, 1], M[1, 2])
    m2 = wp.vec3d(M[2, 0], M[2, 1], M[2, 2])
    v = wp.cross(m0, m1)
    c = wp.cross(m0, m2)
    if wp.dot(c, c) > wp.dot(v, v):
        v = c
    c = wp.cross(m1, m2)
    if wp.dot(c, c) > wp.dot(v, v):
        v = c
    return v/wp.length(v)


@wp.func
def least_eigvec(H: wp.mat33d):
    """the unit eigenvector of the least eigenvalue of a symmetric 3 x 3 H: the eigenvalue by the trigonometric formula,
    the vector the largest cross product of two rows of H - lam I"""
    q = (H[0, 0]+H[1, 1]+H[2, 2])/wp.float64(3.0)
    p1 = H[0, 1]*H[0, 1]+H[0, 2]*H[0, 2]+H[1, 2]*H[1, 2]
    p2 = (H[0, 0]-q)*(H[0, 0]-q)+(H[1, 1]-q)*(H[1, 1]-q)+(H[2, 2]-q)*(H[2, 2]-q)+wp.float64(2.0)*p1
    p = wp.sqrt(p2/wp.float64(6.0))
    I3 = wp.identity(n=3, dtype=wp.float64)
    Bm = (H-I3*q)/p
    r = wp.clamp(wp.determinant(Bm)*wp.float64(0.5), wp.float64(-1.0), wp.float64(1.0))
    phi = wp.acos(r)/wp.float64(3.0)
    lam = q+wp.float64(2.0)*p*wp.cos(phi+wp.float64(2.0943951023931957))
    v = null_row_cross(H-I3*lam)
    return null_row_cross(H-I3*wp.dot(v, H@v))                                    # (one Rayleigh refinement)




@wp.func
def rot_proper(F: wp.mat33d):
    """the nearest proper rotation of F (10-27): scaled Newton X <- (g X + X^-T / g) / 2, g = (|X^-1|_F / |X|_F)^1/2,
    12 steps (Higham; any stretch); det F < 0: the orthogonal factor Q (det -1) flipped along the least eigenvector v of
    H = Q^T F (least_eigvec, analytic): R = Q (I - 2 v v^T), continuous through det F = 0"""
    X = F
    for it in range(12):
        Xi = wp.inverse(X)
        g = wp.sqrt(wp.sqrt(frob2(Xi)/frob2(X)))
        X = (X*g+wp.transpose(Xi)/g)*wp.float64(0.5)
    if wp.determinant(F) < wp.float64(0.0):
        H = wp.transpose(X)@F
        H = (H+wp.transpose(H))*wp.float64(0.5)
        v = least_eigvec(H)
        X = X@(wp.identity(n=3, dtype=wp.float64)-wp.outer(v, v)*wp.float64(2.0))
    return X


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


@wp.func
def lm_factor(Jb: mat99):
    """the LU of J^T J + lam I, lam = 1e-14 tr(J^T J) / 9 (Levenberg-Marquardt; 10-26: a cell whose walls see two
    directions only and whose cap does not close the third has a singular J, cond 5e16 in the 30k jack: its step along
    the null direction 0 there, so that direction keeps its warm start; elsewhere a change ~1e-14 cond^2)"""
    M = wp.transpose(Jb)@Jb
    tr = wp.float64(0.0)
    for i in range(9):
        tr = tr+M[i, i]
    lam = wp.float64(1e-14)*tr/wp.float64(9.0)
    for i in range(9):
        M[i, i] = M[i, i]+lam
    return lu9(M)


@wp.kernel(enable_backward=False)
def k_closure(S: wp.array(dtype=wp.mat33d), Mt: wp.array(dtype=wp.mat33d), Mc: wp.array(dtype=wp.mat33d), P: wp.array(dtype=wp.mat33d),
              mm: wp.array(dtype=wp.mat33d), kap: wp.array(dtype=wp.float64), iters: int, Fw: wp.array(dtype=wp.mat33d),
              F: wp.array(dtype=wp.mat33d), J: wp.array(dtype=mat99), step: wp.array(dtype=wp.float64), perm: wp.array(dtype=vec9),
              Jr: wp.array(dtype=mat99)):
    """Levenberg-Marquardt Newton from Fw with the factors kept (Jr the Jacobian at Fw, J, perm the LU of
    Jr^T Jr + lam I, lm_factor): each step with the current iterate's, the Jacobian rebuilt at the new one (the next
    step's, and at the end the adjoint's)"""
    b = wp.tid()
    Fb = Fw[b]
    Sb, Mtb, Mcb, Pb, mmb, ka = S[b], Mt[b], Mc[b], P[b], mm[b], kap[0]
    I = wp.identity(n=3, dtype=wp.float64)
    LUb = J[b]
    pb = perm[b]
    Jb = Jr[b]
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
        dx = lu9_solve(LUb, pb, wp.transpose(Jb)@rv)
        last = wp.float64(0.0)
        for i in range(9):
            Fb[i//3, i % 3] = Fb[i//3, i % 3]+dx[i]
            last = wp.max(last, wp.abs(dx[i]))
        R = polar12(Fb)
        A0 = wp.transpose(R)@Fb
        U = (A0+wp.transpose(A0))*wp.float64(0.5)
        es = es_of(U-I, Pb, mmb, ka)
        Jb = closure_jac(R, U, es, Mtb, Mcb, Pb, mmb, ka)
        LUb, pb = lm_factor(Jb)
        if last < wp.float64(1e-10):                                                # quadratic: the next step ~1e-20
            break
    F[b] = Fb
    Fw[b] = Fb
    J[b] = LUb
    perm[b] = pb
    Jr[b] = Jb
    step[b] = last


@wp.kernel(enable_backward=False)
def k_closure_lu(Fw: wp.array(dtype=wp.mat33d), Mt: wp.array(dtype=wp.mat33d), Mc: wp.array(dtype=wp.mat33d), P: wp.array(dtype=wp.mat33d),
                 mm: wp.array(dtype=wp.mat33d), kap: wp.array(dtype=wp.float64), J: wp.array(dtype=mat99), perm: wp.array(dtype=vec9),
                 Jr: wp.array(dtype=mat99)):
    """the Jacobian at Fw and its lm_factor (k_closure's starting factors after Fw is reset)"""
    b = wp.tid()
    Fb = Fw[b]
    R = polar12(Fb)
    A0 = wp.transpose(R)@Fb
    U = (A0+wp.transpose(A0))*wp.float64(0.5)
    es = es_of(U-wp.identity(n=3, dtype=wp.float64), P[b], mm[b], kap[0])
    Jb = closure_jac(R, U, es, Mt[b], Mc[b], P[b], mm[b], kap[0])
    LUb, pb = lm_factor(Jb)
    J[b] = LUb
    perm[b] = pb
    Jr[b] = Jb


@wp.kernel(enable_backward=False)
def k_closure_adj(J: wp.array(dtype=mat99), perm: wp.array(dtype=vec9), Jr: wp.array(dtype=mat99), F_adj: wp.array(dtype=wp.mat33d),
                  S_adj: wp.array(dtype=wp.mat33d)):
    """S_bar += Jr (Jr^T Jr + lam I)^-1 F_bar (k_closure's linearization dF = (Jr^T Jr + lam I)^-1 Jr^T dS; J, perm its LU)"""
    b = wp.tid()
    Fa = F_adj[b]
    rv = vec9()
    for i in range(9):
        rv[i] = Fa[i//3, i % 3]
    lam = Jr[b]@lu9_solve(J[b], perm[b], rv)
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
def k_self(x: wp.array(dtype=wp.vec3d), xn: wp.array(dtype=wp.vec3d), si: wp.array(dtype=int), sk: wp.array(dtype=int),
           Rs: wp.array(dtype=wp.float64), d0: wp.array(dtype=wp.float64), Reff: wp.array(dtype=wp.float64), bpar: wp.array(dtype=wp.float64),
           Esp: wp.float64, fn: wp.array(dtype=wp.float64), nv: wp.array(dtype=wp.vec3d), terms: wp.array2d(dtype=wp.float64)):
    """self contact (10-29): a pair (i, k) of surface cells whose exposed caps met face to face (ClosureWarp.update_self);
    Hertz on the balls' overlap from its value at the contact's birth, at the step-end centres as k_ball: delta = R_i + R_k
    - |xe_k - xe_i| - d0, E = wt 8/15 E* (R_eff)^1/2 delta^5/2 (E* of two like bodies: E / 2 (1 - nu^2)); padding i == k"""
    c = wp.tid()
    i, k = si[c], sk[c]
    zero = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
    fn[c] = wp.float64(0.0)
    nv[c] = zero
    if i == k:
        return
    cs, wt = bpar[0], bpar[1]
    xi = cs*x[i]-(cs-wp.float64(1.0))*xn[i]
    xk = cs*x[k]-(cs-wp.float64(1.0))*xn[k]
    dv = xk-xi
    L = wp.length(dv)
    n = dv/L
    dl = Rs[c]-L-d0[c]
    nv[c] = n
    if dl > wp.float64(0.0):
        sq = wp.sqrt(Reff[c]*dl)
        fn[c] = wp.float64(4.0)/wp.float64(3.0)*Esp*sq*dl
        wp.atomic_add(terms, 6, c % NSLOT, wt*wp.float64(8.0)/wp.float64(15.0)*Esp*sq*dl*dl)


@wp.kernel(enable_backward=False)
def k_self_adj(si: wp.array(dtype=int), sk: wp.array(dtype=int), bpar: wp.array(dtype=wp.float64), fn: wp.array(dtype=wp.float64),
               nv: wp.array(dtype=wp.vec3d), terms_adj: wp.array2d(dtype=wp.float64), x_adj: wp.array(dtype=wp.vec3d)):
    """dE/dx_k = -cs wt F n, dE/dx_i = +cs wt F n (n from i to k)"""
    c = wp.tid()
    i, k = si[c], sk[c]
    if i == k:
        return
    g = bpar[0]*bpar[1]*terms_adj[6, c % NSLOT]*fn[c]
    wp.atomic_add(x_adj, k, -g*nv[c])
    wp.atomic_add(x_adj, i, g*nv[c])


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
    """skin_warp.k_F with Mt^-1 = I; R the nearest proper rotation (rot_proper, 10-27)"""
    b = wp.tid()
    Fb = S[b]
    Rb = rot_proper(Fb)
    F[b] = Fb; R[b] = Rb; q[b] = mat_quat(Rb)


# scheme A (10-27): the cap's completion from its rim, F_s = Fbar P + s n' m^T, Fbar_B = 1/2 F_B + 1/2 sum_e w_e F_k(e)
# (sphere_closure.ClosureTerms.completion_a); the surface cells coupled through Fbar. Rim entries gathered per cell (off,
# ek, ew: the cell's rim neighbours and weights) and per neighbour (offT, eTb, eTw: who has this cell on its rim).
@wp.func
def mat_vec9(M: wp.mat33d):
    v = vec9()
    for i in range(9):
        v[i] = M[i//3, i % 3]
    return v


@wp.func
def vec9_mat(v: vec9):
    M = wp.mat33d()
    for i in range(9):
        M[i//3, i % 3] = v[i]
    return M


@wp.func
def comp_a(Fb: wp.mat33d, t1: wp.vec3d, t2: wp.vec3d, m: wp.vec3d, P: wp.mat33d, kap: wp.float64):
    c1 = Fb@t1
    c2 = Fb@t2
    nt = wp.cross(c1, c2)
    a = wp.length(nt)
    s = wp.float64(1.0)-kap*(wp.sqrt(wp.dot(c1, c1)+wp.dot(c2, c2)+wp.float64(2.0)*a)-wp.float64(2.0))
    return Fb@P+wp.outer(nt*(s/a), m)


@wp.func
def comp_a_jvp(Fb: wp.mat33d, dFb: wp.mat33d, t1: wp.vec3d, t2: wp.vec3d, m: wp.vec3d, P: wp.mat33d, kap: wp.float64):
    """d comp_a (Fb) [dFb]"""
    c1 = Fb@t1
    c2 = Fb@t2
    d1 = dFb@t1
    d2 = dFb@t2
    nt = wp.cross(c1, c2)
    a = wp.length(nt)
    nn = nt/a
    dnt = wp.cross(d1, c2)+wp.cross(c1, d2)
    dnn = (dnt-nn*wp.dot(nn, dnt))/a
    q = wp.sqrt(wp.dot(c1, c1)+wp.dot(c2, c2)+wp.float64(2.0)*a)
    s = wp.float64(1.0)-kap*(q-wp.float64(2.0))
    ds = -kap*(wp.dot(c1, d1)+wp.dot(c2, d2)+wp.dot(nn, dnt))/q
    return dFb@P+wp.outer(nn*ds+dnn*s, m)


# the floor on the one-way cap's normal (10-30): the 2-ring fit's area a = |n_t| goes to 0 (n_t / a: 0 / 0, its derivative ~ 1 / a) though the normal stays
# near F_B's. n_t's component q along the step's reference normal n_r (the last accepted midpoint's) kept at f(q) >= eps / 2, C^1: q above eps, eps / 2 +
# q^2 / (2 eps) on [0, eps], eps / 2 below 0; then length 1: comp_a itself above the floor, the direction's sensitivity at most 2 / eps below it
@wp.func
def floored_nt(nt: wp.vec3d, nr: wp.vec3d, eps: wp.float64):
    q = wp.dot(nt, nr)
    if q >= eps:
        return nt
    f = eps*wp.float64(0.5)
    if q > wp.float64(0.0):
        f = f+q*q/(wp.float64(2.0)*eps)
    return nt+nr*(f-q)


@wp.func
def comp_f(Fb: wp.mat33d, t1: wp.vec3d, t2: wp.vec3d, m: wp.vec3d, P: wp.mat33d, kap: wp.float64, nr: wp.vec3d, eps: wp.float64):
    if eps <= wp.float64(0.0):
        return comp_a(Fb, t1, t2, m, P, kap)
    c1 = Fb@t1
    c2 = Fb@t2
    nt = wp.cross(c1, c2)
    s = wp.float64(1.0)-kap*(wp.sqrt(wp.dot(c1, c1)+wp.dot(c2, c2)+wp.float64(2.0)*wp.length(nt))-wp.float64(2.0))
    nf = floored_nt(nt, nr, eps)
    return Fb@P+wp.outer(nf*(s/wp.length(nf)), m)


@wp.func
def comp_f_jvp(Fb: wp.mat33d, dFb: wp.mat33d, t1: wp.vec3d, t2: wp.vec3d, m: wp.vec3d, P: wp.mat33d, kap: wp.float64, nr: wp.vec3d, eps: wp.float64):
    """d comp_f (Fb) [dFb], n_r fixed"""
    c1 = Fb@t1
    c2 = Fb@t2
    nt = wp.cross(c1, c2)
    q_ = wp.dot(nt, nr)
    if eps <= wp.float64(0.0) or q_ >= eps:
        return comp_a_jvp(Fb, dFb, t1, t2, m, P, kap)
    d1 = dFb@t1
    d2 = dFb@t2
    dnt = wp.cross(d1, c2)+wp.cross(c1, d2)
    a = wp.length(nt)
    nn = wp.vec3d()
    if a > wp.float64(0.0):
        nn = nt/a
    q = wp.sqrt(wp.dot(c1, c1)+wp.dot(c2, c2)+wp.float64(2.0)*a)
    s = wp.float64(1.0)-kap*(q-wp.float64(2.0))
    ds = -kap*(wp.dot(c1, d1)+wp.dot(c2, d2)+wp.dot(nn, dnt))/q
    fp = wp.float64(0.0)
    if q_ > wp.float64(0.0):
        fp = q_/eps
    nf = floored_nt(nt, nr, eps)
    ln = wp.length(nf)
    nh = nf/ln
    dnf = dnt+nr*((fp-wp.float64(1.0))*wp.dot(nr, dnt))
    dnh = (dnf-nh*wp.dot(nh, dnf))/ln
    return dFb@P+wp.outer(nh*ds+dnh*s, m)


@wp.func
def rim_mean(F: wp.array(dtype=wp.mat33d), off: wp.array(dtype=int), ek: wp.array(dtype=int), ew: wp.array(dtype=wp.float64), b: int):
    A = wp.mat33d()
    for e in range(off[b], off[b+1]):
        A = A+F[ek[e]]*ew[e]
    return A


@wp.kernel(enable_backward=False)
def k_closA_blocks(F: wp.array(dtype=wp.mat33d), off: wp.array(dtype=int), ek: wp.array(dtype=int), ew: wp.array(dtype=wp.float64),
                   Mt: wp.array(dtype=wp.mat33d), Mc: wp.array(dtype=wp.mat33d), t1: wp.array(dtype=wp.vec3d), t2: wp.array(dtype=wp.vec3d),
                   mA: wp.array(dtype=wp.vec3d), PA: wp.array(dtype=wp.mat33d), kap: wp.array(dtype=wp.float64),
                   LU: wp.array(dtype=mat99), perm: wp.array(dtype=vec9), C: wp.array(dtype=mat99)):
    """at F: the own block D = d r_B / d F_B (the rim's neighbours held; its LU) and C = d (F_s M_c) / d Fbar"""
    b = wp.tid()
    Fb = (F[b]+rim_mean(F, off, ek, ew, b))*wp.float64(0.5)
    D = mat99()
    Cm = mat99()
    for k in range(9):
        E = wp.mat33d()
        E[k//3, k % 3] = wp.float64(1.0)
        cM = comp_a_jvp(Fb, E, t1[b], t2[b], mA[b], PA[b], kap[0])@Mc[b]
        dr = E@Mt[b]-cM*wp.float64(0.5)
        for i in range(9):
            D[i, k] = dr[i//3, i % 3]
            Cm[i, k] = cM[i//3, i % 3]
    LUb, pb = lu9(D)
    LU[b] = LUb
    perm[b] = pb
    C[b] = Cm


@wp.kernel(enable_backward=False)
def k_closA_sweep(S: wp.array(dtype=wp.mat33d), Fin: wp.array(dtype=wp.mat33d), off: wp.array(dtype=int), ek: wp.array(dtype=int),
                  ew: wp.array(dtype=wp.float64), Mt: wp.array(dtype=wp.mat33d), Mc: wp.array(dtype=wp.mat33d), t1: wp.array(dtype=wp.vec3d),
                  t2: wp.array(dtype=wp.vec3d), mA: wp.array(dtype=wp.vec3d), PA: wp.array(dtype=wp.mat33d), kap: wp.array(dtype=wp.float64),
                  LU: wp.array(dtype=mat99), perm: wp.array(dtype=vec9), Fout: wp.array(dtype=wp.mat33d), step: wp.array(dtype=wp.float64)):
    """one chord-Newton block-Jacobi sweep: F_B -= D_B^-1 r_B(F_B; the neighbours of Fin), D kept"""
    b = wp.tid()
    Fo = Fin[b]
    Fb = (Fo+rim_mean(Fin, off, ek, ew, b))*wp.float64(0.5)
    r = Fo@Mt[b]-S[b]-comp_a(Fb, t1[b], t2[b], mA[b], PA[b], kap[0])@Mc[b]
    dx = lu9_solve(LU[b], perm[b], -mat_vec9(r))
    last = wp.float64(0.0)
    for i in range(9):
        last = wp.max(last, wp.abs(dx[i]))
    Fout[b] = Fo+vec9_mat(dx)
    step[b] = last


@wp.kernel(enable_backward=False)
def k_lin_sweep(rhs: wp.array(dtype=wp.mat33d), yin: wp.array(dtype=wp.mat33d), off: wp.array(dtype=int), ek: wp.array(dtype=int),
                ew: wp.array(dtype=wp.float64), C: wp.array(dtype=mat99), LU: wp.array(dtype=mat99), perm: wp.array(dtype=vec9),
                yout: wp.array(dtype=wp.mat33d)):
    """block-Jacobi on J y = rhs (J = D + O, O_Bk = -1/2 w_Bk C_B): y_B = D_B^-1 (rhs_B + 1/2 C_B sum_e w_e y_k(e))"""
    b = wp.tid()
    A = rim_mean(yin, off, ek, ew, b)
    v = mat_vec9(rhs[b])+(C[b]@mat_vec9(A))*wp.float64(0.5)
    yout[b] = vec9_mat(lu9_solve(LU[b], perm[b], v))


@wp.kernel(enable_backward=False)
def k_lin_sweep_t(rhs: wp.array(dtype=wp.mat33d), lin: wp.array(dtype=wp.mat33d), offT: wp.array(dtype=int), eTb: wp.array(dtype=int),
                  eTw: wp.array(dtype=wp.float64), C: wp.array(dtype=mat99), LU: wp.array(dtype=mat99), perm: wp.array(dtype=vec9),
                  lout: wp.array(dtype=wp.mat33d)):
    """block-Jacobi on J^T l = rhs: l_k = D_k^-T (rhs_k + 1/2 sum_{B: k on B's rim} w C_B^T l_B)"""
    k = wp.tid()
    v = mat_vec9(rhs[k])
    for e in range(offT[k], offT[k+1]):
        B = eTb[e]
        v = v+(wp.transpose(C[B])@mat_vec9(lin[B]))*(wp.float64(0.5)*eTw[e])
    lout[k] = vec9_mat(lu9_solve_t(LU[k], perm[k], v))


@wp.kernel(enable_backward=False)
def k_mat_lin(a: wp.array(dtype=wp.mat33d), ca: wp.float64, b_: wp.array(dtype=wp.mat33d), cb: wp.float64, out: wp.array(dtype=wp.mat33d)):
    i = wp.tid()
    out[i] = a[i]*ca+b_[i]*cb


@wp.kernel(enable_backward=False)
def k_mat_add_to(a: wp.array(dtype=wp.mat33d), out: wp.array(dtype=wp.mat33d)):
    i = wp.tid()
    out[i] = out[i]+a[i]


# the one-way cap (10-29, user: the surface must not act back on F_B): a surface cell's cap completion takes the surface's
# deformation from the positions alone, G_B = the least-squares affine fit of its 2-ring sites' offsets about the site
# (sum_e (x_k - x_B) (x) dX_e M2_B^-1, M2_B = sum_e dX_e (x) dX_e; exact for any affine motion), F_s = comp_a(G_B); then
# F_B = (S_B + F_s M_c) M_t^-1 directly: no fixed point, M_t positive definite (scheme A's own block cancelled under the
# caps' in-plane compression: the new lion's steps 428 / 437)
@wp.kernel(enable_backward=False)
def k_closP(x: wp.array(dtype=wp.vec3d), S: wp.array(dtype=wp.mat33d), poff: wp.array(dtype=int), pk: wp.array(dtype=int),
            pdX: wp.array(dtype=wp.vec3d), M2i: wp.array(dtype=wp.mat33d), Mti: wp.array(dtype=wp.mat33d), Mc: wp.array(dtype=wp.mat33d),
            t1: wp.array(dtype=wp.vec3d), t2: wp.array(dtype=wp.vec3d), mA: wp.array(dtype=wp.vec3d), PA: wp.array(dtype=wp.mat33d),
            kap: wp.array(dtype=wp.float64), nr: wp.array(dtype=wp.vec3d), eps: wp.array(dtype=wp.float64), G: wp.array(dtype=wp.mat33d),
            Fc: wp.array(dtype=wp.mat33d)):
    b = wp.tid()
    C = wp.mat33d()
    if poff[b+1] > poff[b]:
        Sx = wp.mat33d()
        xb = x[b]
        for e in range(poff[b], poff[b+1]):
            Sx = Sx+wp.outer(x[pk[e]]-xb, pdX[e])
        Gb = Sx@M2i[b]
        G[b] = Gb
        C = comp_f(Gb, t1[b], t2[b], mA[b], PA[b], kap[0], nr[b], eps[0])@Mc[b]
    Fc[b] = (S[b]+C)@Mti[b]


@wp.kernel(enable_backward=False)
def k_capref(G: wp.array(dtype=wp.mat33d), poff: wp.array(dtype=int), t1: wp.array(dtype=wp.vec3d), t2: wp.array(dtype=wp.vec3d),
             eps: wp.array(dtype=wp.float64), nr: wp.array(dtype=wp.vec3d)):
    """the next step's reference normal: the accepted midpoint's floored normal (G: W's buffers hold its evaluation)"""
    b = wp.tid()
    if poff[b+1] > poff[b]:
        nf = floored_nt(wp.cross(G[b]@t1[b], G[b]@t2[b]), nr[b], eps[0])
        nr[b] = nf/wp.length(nf)


@wp.kernel(enable_backward=False)
def k_Fw(S: wp.array(dtype=wp.mat33d), Mbi: wp.array(dtype=wp.mat33d), Fw: wp.array(dtype=wp.mat33d)):
    """the walls-only fit F_w = S_B Mb^-1 (positions only)"""
    b = wp.tid()
    Fw[b] = S[b]@Mbi[b]


# the walls' cap (10-29): scheme A's rim structure with the walls-only fits in place of F_B, G_B = 1/2 F_w(B) + 1/2 sum_e
# w_e F_w(k(e)), F_s = comp_a(G_B), F_B = (S_B + F_s M_c) M_t^-1 directly: one-way (F_w carries no cap, so no F_B -> cap ->
# F_B loop) and its input stays non-degenerate where the 2-ring position fit collapsed (new lion: a >= 0.12 vs 0)
@wp.kernel(enable_backward=False)
def k_closW(S: wp.array(dtype=wp.mat33d), Fw: wp.array(dtype=wp.mat33d), off: wp.array(dtype=int), ek: wp.array(dtype=int),
            ew: wp.array(dtype=wp.float64), Mti: wp.array(dtype=wp.mat33d), Mc: wp.array(dtype=wp.mat33d), t1: wp.array(dtype=wp.vec3d),
            t2: wp.array(dtype=wp.vec3d), mA: wp.array(dtype=wp.vec3d), PA: wp.array(dtype=wp.mat33d), kap: wp.array(dtype=wp.float64),
            G: wp.array(dtype=wp.mat33d), Fc: wp.array(dtype=wp.mat33d)):
    b = wp.tid()
    C = wp.mat33d()
    if off[b+1] > off[b]:
        Gb = (Fw[b]+rim_mean(Fw, off, ek, ew, b))*wp.float64(0.5)
        G[b] = Gb
        C = comp_a(Gb, t1[b], t2[b], mA[b], PA[b], kap[0])@Mc[b]
    Fc[b] = (S[b]+C)@Mti[b]


@wp.kernel(enable_backward=False)
def k_closW_adj(off: wp.array(dtype=int), Mti: wp.array(dtype=wp.mat33d), Mc: wp.array(dtype=wp.mat33d), t1: wp.array(dtype=wp.vec3d),
                t2: wp.array(dtype=wp.vec3d), mA: wp.array(dtype=wp.vec3d), PA: wp.array(dtype=wp.mat33d), kap: wp.array(dtype=wp.float64),
                G: wp.array(dtype=wp.mat33d), Fc_bar: wp.array(dtype=wp.mat33d), S_bar: wp.array(dtype=wp.mat33d), Gbar: wp.array(dtype=wp.mat33d)):
    """S_bar += Fc_bar M_t^-T; G_bar = d comp_a^T [Fc_bar M_t^-T M_c^T] (by its 9 directions), 0 off the caps"""
    b = wp.tid()
    Cb = Fc_bar[b]@wp.transpose(Mti[b])
    S_bar[b] = S_bar[b]+Cb
    Gb_ = wp.mat33d()
    if off[b+1] > off[b]:
        cbar = Cb@wp.transpose(Mc[b])
        Gb = G[b]
        for k in range(9):
            E = wp.mat33d()
            E[k//3, k % 3] = wp.float64(1.0)
            d = comp_a_jvp(Gb, E, t1[b], t2[b], mA[b], PA[b], kap[0])
            s = wp.float64(0.0)
            for i in range(3):
                for j in range(3):
                    s = s+cbar[i, j]*d[i, j]
            Gb_[k//3, k % 3] = s
    Gbar[b] = Gb_


@wp.kernel(enable_backward=False)
def k_Fw_adj(Gbar: wp.array(dtype=wp.mat33d), offT: wp.array(dtype=int), eTb: wp.array(dtype=int), eTw: wp.array(dtype=wp.float64),
             Mbi: wp.array(dtype=wp.mat33d), S_bar: wp.array(dtype=wp.mat33d)):
    """F_w(k)_bar = 1/2 G_bar(k) + 1/2 sum_{B: k on B's rim} w G_bar(B); S_bar += F_w_bar Mb^-T"""
    k = wp.tid()
    A = Gbar[k]*wp.float64(0.5)
    for e in range(offT[k], offT[k+1]):
        A = A+Gbar[eTb[e]]*(wp.float64(0.5)*eTw[e])
    S_bar[k] = S_bar[k]+A@wp.transpose(Mbi[k])


@wp.kernel(enable_backward=False)
def k_closP_adj(poff: wp.array(dtype=int), pk: wp.array(dtype=int), pdX: wp.array(dtype=wp.vec3d), M2i: wp.array(dtype=wp.mat33d),
                Mti: wp.array(dtype=wp.mat33d), Mc: wp.array(dtype=wp.mat33d), t1: wp.array(dtype=wp.vec3d), t2: wp.array(dtype=wp.vec3d),
                mA: wp.array(dtype=wp.vec3d), PA: wp.array(dtype=wp.mat33d), kap: wp.array(dtype=wp.float64), nr: wp.array(dtype=wp.vec3d),
                eps: wp.array(dtype=wp.float64), G: wp.array(dtype=wp.mat33d), Fc_bar: wp.array(dtype=wp.mat33d), S_bar: wp.array(dtype=wp.mat33d),
                x_bar: wp.array(dtype=wp.vec3d)):
    """S_bar += Fc_bar M_t^-T; G_bar = d comp_f^T [Fc_bar M_t^-T M_c^T] (by its 9 directions); x_k += G_bar M2^-T dX_e, x_B -= the sum"""
    b = wp.tid()
    Cb = Fc_bar[b]@wp.transpose(Mti[b])
    S_bar[b] = S_bar[b]+Cb
    if poff[b+1] > poff[b]:
        cbar = Cb@wp.transpose(Mc[b])
        Gb = G[b]
        Gbar = wp.mat33d()
        for k in range(9):
            E = wp.mat33d()
            E[k//3, k % 3] = wp.float64(1.0)
            d = comp_f_jvp(Gb, E, t1[b], t2[b], mA[b], PA[b], kap[0], nr[b], eps[0])
            s = wp.float64(0.0)
            for i in range(3):
                for j in range(3):
                    s = s+cbar[i, j]*d[i, j]
            Gbar[k//3, k % 3] = s
        Sxbar = Gbar@wp.transpose(M2i[b])
        tot = wp.vec3d()
        for e in range(poff[b], poff[b+1]):
            v = Sxbar@pdX[e]
            wp.atomic_add(x_bar, pk[e], v)
            tot = tot+v
        wp.atomic_add(x_bar, b, -tot)


# UL's F in place of the fitted skin (10-30, the user: the bonds carry the force, UL's F needs no skin): within a step, on the step-start cut, F_b = (I + G_b(x - x_n)) F_n with G_b = sum_e (dx_O - dx_S) (x) R_e
# UL's least-squares fit on the cut's walls (R_e = w d^T M_S^+), a tiny cell's G the V0-weighted mean of its non-tiny wall neighbours' (agg). Padding entries have R = 0, Aw = 0
@wp.kernel(enable_backward=False)
def k_Gul(x: wp.array(dtype=wp.vec3d), xn: wp.array(dtype=wp.vec3d), S: wp.array(dtype=int), O: wp.array(dtype=int), Rw: wp.array(dtype=wp.vec3d), G: wp.array(dtype=wp.mat33d)):
    e = wp.tid()
    s, o = S[e], O[e]
    wp.atomic_add(G, s, wp.outer((x[o]-xn[o])-(x[s]-xn[s]), Rw[e]))


@wp.kernel(enable_backward=False)
def k_Aul(G: wp.array(dtype=wp.mat33d), As: wp.array(dtype=int), Ao: wp.array(dtype=int), Aw: wp.array(dtype=wp.float64), Ga: wp.array(dtype=wp.mat33d)):
    e = wp.tid()
    wp.atomic_add(Ga, As[e], G[Ao[e]]*Aw[e])


@wp.kernel(enable_backward=False)
def k_Ful(G: wp.array(dtype=wp.mat33d), Ga: wp.array(dtype=wp.mat33d), used: wp.array(dtype=int), F0: wp.array(dtype=wp.mat33d), Fc: wp.array(dtype=wp.mat33d)):
    b = wp.tid()
    Gb = G[b]
    if used[b] != 0:
        Gb = Ga[b]
    Fc[b] = F0[b]+Gb@F0[b]


@wp.kernel(enable_backward=False)
def k_Ful_adj(Fb: wp.array(dtype=wp.mat33d), used: wp.array(dtype=int), F0: wp.array(dtype=wp.mat33d), Gb: wp.array(dtype=wp.mat33d)):
    """dE = Fb : (dG F0) = (Fb F0^T) : dG; a tiny cell's goes to its neighbours (k_Aul_adj)"""
    b = wp.tid()
    Gb[b] = wp.mat33d()
    if used[b] == 0:
        Gb[b] = Fb[b]@wp.transpose(F0[b])


@wp.kernel(enable_backward=False)
def k_Aul_adj(Fb: wp.array(dtype=wp.mat33d), F0: wp.array(dtype=wp.mat33d), As: wp.array(dtype=int), Ao: wp.array(dtype=int), Aw: wp.array(dtype=wp.float64), Gb: wp.array(dtype=wp.mat33d)):
    e = wp.tid()
    s = As[e]
    wp.atomic_add(Gb, Ao[e], (Fb[s]@wp.transpose(F0[s]))*Aw[e])


@wp.kernel(enable_backward=False)
def k_Gul_adj(Gb: wp.array(dtype=wp.mat33d), S: wp.array(dtype=int), O: wp.array(dtype=int), Rw: wp.array(dtype=wp.vec3d), x_bar: wp.array(dtype=wp.vec3d)):
    e = wp.tid()
    v = Gb[S[e]]@Rw[e]
    wp.atomic_add(x_bar, O[e], v)
    wp.atomic_add(x_bar, S[e], -v)


def st_V(W):
    return W.st.V


class ClosureWarp(SkinWarp):
    """built from a ClosureTerms st; grad(x) -> (loss, dE/dx), torch (prepare() first). Viscosity, contact and friction as
    SkinWarp's (set_viscosity, set_contact), the contact made the cap's series spring and the glue the cap's shear"""
    iters = 3
    sweep_mult = 1                                                                  # (10-28 test: n_lin x this)

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
        n_ = self.n                                                                 # (scheme A: the rim, by cell and by neighbour)
        o1 = torch.argsort(st.r_c, stable=True); o2 = torch.argsort(st.r_k, stable=True)
        offs = lambda key: torch.cat((torch.zeros(1, dtype=torch.long, device=key.device), torch.cumsum(torch.bincount(key, minlength=n_), 0)))
        self.off, self.ek, self.ew = I(offs(st.r_c)), I(st.r_k[o1]), T(st.r_w[o1])
        self.offT, self.eTb, self.eTw = I(offs(st.r_k)), I(st.r_c[o2]), T(st.r_w[o2])
        self.t1, self.t2, self.mA, self.PA = T(st.t1, wp.vec3d), T(st.t2, wp.vec3d), T(st.mA, wp.vec3d), T(st.PA, wp.mat33d)
        self.linear_closure = False                                                 # (TrustCG's products: the linearised closure)
        self.n_lin = 0
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
        self.ct = self.cp = self.vis = self.ball = self.selfc = None
        self.cap_pos = False                                                        # (set_cap_positions: the one-way cap)
        self.cap_eps = 0.                                                           # (set_cap_floor: the floor on its normal)
        self.cap_wall = False                                                       # (set_cap_walls: the walls' cap)
        self.cap_ul = False                                                         # (set_cap_ul: UL's F, no cap)
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

    def set_self_contact(self, r, K=20000, sigma=0.):
        """self contact between surface cells (k_self; 10-29, user: contact from the current configuration's exact cut),
        before prepare() and after set_ball (its cs, wt). r: the cells' rest radii (the cut's balls). update_self(x) at
        each step's start (MidpointStep.step) re-cuts at x and keeps the pair list; K the list's capacity. sigma > 0 (10-30):
        the colour code instead of the facing caps, see _color_code"""
        from free_surface.recut import Recut
        dev = 'cuda'; n = self.n
        self._rr = r.to(f64).contiguous(); self._recut = Recut(n, dev)
        st = self.st
        a, b = torch.minimum(st.i, st.j), torch.maximum(st.i, st.j)
        self._rest_keys = torch.unique(a.long()*n+b.long())
        G, _ = self._recut.geometry(st.X, self._rr)
        self._cap_prev, self._m_prev = self._caps(G)
        self._col_Q = self._color_code(G, sigma) if sigma > 0 else None
        self._pw = torch.zeros(0, dtype=f64, device=dev)
        z = lambda dt_: wp.zeros(K, dtype=dt_, device=dev)
        self.selfc = dict(K=K, i=z(wp.int32), k=z(wp.int32), Rs=z(wp.float64), d0=z(wp.float64), Reff=z(wp.float64), fn=z(wp.float64),
                          nv=z(wp.vec3d), xn=wp.zeros(n, dtype=wp.vec3d, device=dev), Esp=wp.float64(self.Es/2))
        tt = lambda a_: wp.to_torch(a_, requires_grad=False)
        self.tsc = {k_: tt(self.selfc[k_]) for k_ in ('i', 'k', 'Rs', 'd0', 'Reff', 'fn', 'xn')}
        self.tsc['d0'].fill_(1e300)
        self._pairs = torch.zeros(0, 2, dtype=torch.long, device=dev); self._pd0 = torch.zeros(0, dtype=f64, device=dev)
        self.self_stats = dict(pairs=0, born=0, left=0, cut_fail=0)
        self.self_births = []; self._self_calls = 0                               # (step, i, k, d0, rest distance, n.m_i, n.m_k) per birth

    def set_cap_positions(self):
        """the one-way cap (k_closP) in place of scheme A's coupled closure, before prepare(): each surface cell's 2-ring
        (the rest walls' graph, the cell excluded), the offsets dX_e, the fit's M2^-1, and M_t^-1"""
        st = self.st; n = self.n; dev = st.X.device
        ij = torch.cat([torch.stack([st.i, st.j]), torch.stack([st.j, st.i])], 1).long()
        A = torch.sparse_coo_tensor(ij, torch.ones(ij.shape[1], dtype=torch.float32, device=dev), (n, n)).coalesce()
        R2 = (torch.sparse.mm(A, A)+A).coalesce().indices()
        keep = (R2[0] != R2[1]) & st.cap[R2[0]]
        pc, pk = R2[0][keep], R2[1][keep]
        o = torch.argsort(pc, stable=True); pc, pk = pc[o], pk[o]
        dX = st.X[pk]-st.X[pc]
        M2 = torch.zeros(n, 3, 3, dtype=f64, device=dev).index_add(0, pc, dX[:, :, None]*dX[:, None, :])
        M2[~st.cap] = torch.eye(3, dtype=f64, device=dev)
        off = torch.cat((torch.zeros(1, dtype=torch.long, device=dev), torch.cumsum(torch.bincount(pc, minlength=n), 0)))
        T = lambda t, dt=wp.float64: wp.from_torch(t.contiguous(), dtype=dt)
        I = lambda t: wp.from_torch(t.to(torch.int32).contiguous(), dtype=wp.int32)
        self._pos = dict(off=off, pk=pk, dX=dX.contiguous(), M2i=torch.linalg.inv(M2).contiguous(), Mti=st.Mt_inv.contiguous())
        P = self._pos
        self.poff, self.pk, self.pdX = I(P['off']), I(P['pk']), T(P['dX'], wp.vec3d)
        self.M2i, self.Mti = T(P['M2i'], wp.mat33d), T(P['Mti'], wp.mat33d)
        P['nr'] = st.mA.clone().contiguous()
        self.capnr, self.capeps = T(P['nr'], wp.vec3d), wp.array([0.], dtype=wp.float64, device='cuda')
        self.cap_pos = True

    def set_cap_ul(self, agg=.1, Kw=80000, Ka=20000, vol_tol=1e-5):
        """UL's F instead of the fitted skin, after set_self_contact (its colours and cut), before prepare(): each step re-solves the radii for V0 det F_n and re-cuts (update_ul), the fit on the cut's walls
        with areas scaled by the colour similarity, F within the step k_Gul / k_Aul / k_Ful; the volume term on det F alone (no k_walls). Not for plasticity: the bonds are the rest bonds. Kw, Ka: capacities"""
        dev = 'cuda'; n = self.n
        self.cap_ul = True; self._ul_agg = agg; self._ul_tol = vol_tol
        self._rr0 = self._rr.clone(); self._ulr = self._rr.clone(); self._ulG = None
        self._Fn = torch.eye(3, dtype=f64, device=dev).repeat(n, 1, 1)
        self.ul_stats = dict(radii_fail=0, floored=0, walls=0, radii_it=0)
        z = lambda k, dt_: wp.zeros(k, dtype=dt_, device=dev)
        self.ul = dict(K=2*Kw, Ka=Ka, Kw=Kw, xn=z(n, wp.vec3d), S=z(2*Kw, wp.int32), O=z(2*Kw, wp.int32), R=z(2*Kw, wp.vec3d), As=z(Ka, wp.int32), Ao=z(Ka, wp.int32), Aw=z(Ka, wp.float64),
                       used=z(n, wp.int32), F0=z(n, wp.mat33d))
        tt = lambda a_: wp.to_torch(a_, requires_grad=False)
        self.tul = {k_: tt(self.ul[k_]) for k_ in ('xn', 'S', 'O', 'R', 'As', 'Ao', 'Aw', 'used', 'F0')}
        self.tul['xn'].copy_(self.st.X); self.tul['F0'].copy_(self._Fn)
        self.Vdet = wp.from_torch(self.st.V.contiguous(), dtype=wp.float64)

    def update_ul(self, x):
        """the step's start at x: radii for V0 det F_n (their failure keeps the last radii: they only shape the cut), the cut, the fit's entries R_e, the aggregation, x_n, F_n"""
        n = self.n; x = x.detach(); st = self.ul_stats
        J = torch.linalg.det(self._Fn); st['floored'] = int((J < .02).sum())
        try:
            r, G, S_, it, nrep = self._recut.solve_radii(x, self._ulr, st_V(self)*J.clamp(min=.02), tol=self._ul_tol, r_base=self._rr0)
            st['radii_it'] = it
        except RuntimeError:
            st['radii_fail'] += 1
            r = self._ulr; G, S_ = self._recut.geometry(x, r)
        self._ulr, self._rr, self._ulG = r, r, G
        kp = G.wall_A.detach() > 0
        i, j, A = G.i[kp].long(), G.j[kp].long(), G.wall_A.detach()[kp]
        a, b = torch.minimum(i, j), torch.maximum(i, j)
        s = torch.ones(len(i), dtype=f64, device=x.device); nw = ~torch.isin(a*n+b, self._rest_keys)
        if self._col_Q is not None:                                                 # (no colours: every wall counts in full)
            s[nw] = self._color_sim(a[nw], b[nw])
        d = x[j]-x[i]; w = A*s/d.norm(dim=1)
        q = w[:, None, None]*d[:, :, None]*d[:, None, :]
        M = torch.zeros(n, 3, 3, dtype=f64, device=x.device).index_add(0, i, q).index_add(0, j, q)
        Mi = torch.linalg.pinv(M)
        vd = w[:, None]*d
        Ri = torch.einsum('fa,fab->fb', vd, Mi[i]); Rj = torch.einsum('fa,fab->fb', -vd, Mi[j])
        m = len(i); U = self.ul; t = self.tul
        if m > U['Kw']:
            raise RuntimeError('update_ul: %d walls over the capacity %d' % (m, U['Kw']))
        st['walls'] = m
        t['S'].zero_(); t['O'].zero_(); t['R'].zero_()
        t['S'][:2*m] = torch.cat([i, j]).to(torch.int32); t['O'][:2*m] = torch.cat([j, i]).to(torch.int32); t['R'][:2*m] = torch.cat([Ri, Rj])
        V0 = st_V(self); tiny = V0 < self._ul_agg*V0.median()
        m1, m2 = tiny[i] & ~tiny[j], tiny[j] & ~tiny[i]
        as_, ao_ = torch.cat([i[m1], j[m2]]), torch.cat([j[m1], i[m2]])
        va = V0[ao_]*torch.cat([s[m1], s[m2]]); ka = va > 0; as_, ao_, va = as_[ka], ao_[ka], va[ka]
        den = torch.zeros(n, dtype=f64, device=x.device).index_add(0, as_, va)
        if len(as_) > U['Ka']:
            raise RuntimeError('update_ul: %d aggregation entries over the capacity %d' % (len(as_), U['Ka']))
        t['As'].zero_(); t['Ao'].zero_(); t['Aw'].zero_()
        k_ = len(as_); t['As'][:k_] = as_.to(torch.int32); t['Ao'][:k_] = ao_.to(torch.int32); t['Aw'][:k_] = va/den[as_]
        t['used'].copy_((tiny & (den > 0)).to(torch.int32))
        t['xn'].copy_(x); t['F0'].copy_(self._Fn)

    def end_ul(self):
        """after the step: F_n+1 = 2 F(z) - F_n, F linear in x and the buffers at the accepted midpoint z"""
        self._Fn = 2*self.tF-self._Fn

    def set_cap_floor(self, eps):
        """the floor on the one-way cap's normal (floored_nt), after set_cap_positions; 0: none, comp_a itself. The reference normal starts at the rest
        normal and MidpointStep.end() moves it to each accepted midpoint's"""
        self.cap_eps = float(eps)
        wp.to_torch(self.capeps, requires_grad=False)[0] = self.cap_eps

    def set_cap_walls(self):
        """the walls' cap (k_Fw, k_closW) in place of scheme A's coupled closure, before prepare(): Mb^-1 (a pseudo-inverse:
        cells with fewer than three wall directions get F_w = 0 there) and M_t^-1"""
        st = self.st
        self._capw = dict(Mbi=torch.linalg.pinv(st.Mb).contiguous(), Mti=st.Mt_inv.contiguous())
        self.Mbi = wp.from_torch(self._capw['Mbi'], dtype=wp.mat33d)
        self.Mti = wp.from_torch(self._capw['Mti'], dtype=wp.mat33d)
        self.cap_wall = True

    def _caps(self, G):
        pa = G.patch_A.detach(); pn = G.patch_n.detach()
        return pa > 0, pn/pn.norm(dim=1, keepdim=True).clamp(min=1e-300)

    def _color_code(self, G, sigma):
        """a static colour per cell (10-30, the user's idea, from the UL model). Each cell of the rest skin has its own colour, spread over the rest skin
        graph as exp(-d^2 / (2 sigma^2)), d in steps; the others inherit from their neighbours, as the average weighted by the rest A / L (a harmonic
        extension). The similarity of two cells is sum sqrt(p_i p_j); a contact's push is scaled by 1 - similarity. sigma: half similarity at the reach of
        deformation alone, A^(2/3) steps with A the stretch ratio the impact's energy density pays for (lion: A 4.72, 2.81 steps, sigma 1.19)"""
        import scipy.sparse as sp_
        from scipy.sparse.linalg import splu
        from scipy.sparse.csgraph import dijkstra
        n = self.n; X = self.st.X; kp = G.wall_A.detach() > 0
        i0, j0 = G.i[kp], G.j[kp]
        w = (G.wall_A.detach()[kp]/(X[j0]-X[i0]).norm(dim=1)).cpu().numpy()
        W = sp_.coo_matrix((w, (i0.cpu().numpy(), j0.cpu().numpy())), shape=(n, n)).tocsr(); W = W+W.T
        skin = (G.patch_A.detach() > 0).cpu().numpy(); sid, iid = np.where(skin)[0], np.where(~skin)[0]
        ds = dijkstra((W[sid][:, sid] > 0).astype(np.float64).tocsr(), unweighted=True)
        Ps = np.exp(-np.where(np.isfinite(ds), ds, 1e6)**2/(2*sigma**2)); Ps = Ps/Ps.sum(1, keepdims=True)
        P = np.zeros((n, len(sid)), dtype=np.float32); P[sid] = Ps
        L = (sp_.diags(np.asarray(W.sum(1)).ravel())-W).tocsc()
        P[iid] = splu(L[iid][:, iid].tocsc()).solve(np.asarray(W[iid][:, sid]@Ps)).astype(np.float32)
        return torch.as_tensor(np.sqrt(np.clip(P, 0, None)), device='cuda')

    def _color_sim(self, a, b, chunk=4096):
        s = torch.zeros(len(a), dtype=f64, device=a.device)
        for c0 in range(0, len(a), chunk):
            s[c0:c0+chunk] = (self._col_Q[a[c0:c0+chunk]]*self._col_Q[b[c0:c0+chunk]]).sum(1).to(f64).clamp(max=1.)
        return s

    def update_self(self, x):
        """the self contact list at the step's start x (torch, n x 3): re-cut at x with the rest radii; a wall (i, k) of
        the cut that is not a rest wall and not yet a contact becomes one when both cells had an exposed cap at the last
        cut and the caps face each other (n_ik . m_i > 0 > n_ik . m_k, n_ik from i to k, m the caps' mean normals there),
        with d0 its balls' overlap now (born without force); a contact leaves when its overlap is back to d0. With the
        colour code every such wall is a contact, its push scaled by 1 - the cells' similarity (10-30)"""
        n = self.n; dev = x.device
        x = x.detach(); self._self_calls += 1
        P = self._pairs
        if len(P):
            ov = self._rr[P[:, 0]]+self._rr[P[:, 1]]-(x[P[:, 1]]-x[P[:, 0]]).norm(dim=1)
            stay = ov > self._pd0
            self.self_stats['left'] += int((~stay).sum())
            self._pairs, self._pd0 = P[stay], self._pd0[stay]
            self._pw = self._pw[stay]
        try:
            G = self._ulG if self.cap_ul else self._recut.geometry(x, self._rr)[0]   # (cap_ul: the step's own cut, update_ul)
        except RuntimeError:
            self.self_stats['cut_fail'] += 1
            G = None
        if G is not None:
            keep = G.wall_A.detach() > 0
            i, j = G.i[keep].long(), G.j[keep].long()
            a, b = torch.minimum(i, j), torch.maximum(i, j)
            key = a*n+b
            old = torch.isin(key, self._rest_keys)
            if len(self._pairs):
                old |= torch.isin(key, self._pairs[:, 0]*n+self._pairs[:, 1])
            cand = ~old if self._col_Q is not None else ~old & self._cap_prev[a] & self._cap_prev[b]
            a, b = a[cand], b[cand]
            nab = x[b]-x[a]; nab = nab/nab.norm(dim=1, keepdim=True)
            if self._col_Q is not None:
                pw = 1-self._color_sim(a, b); face = pw > 0
            else:
                pw = torch.ones(len(a), dtype=f64, device=dev); face = ((nab*self._m_prev[a]).sum(1) > 0) & ((nab*self._m_prev[b]).sum(1) < 0)
            a, b, pw = a[face], b[face], pw[face]
            if len(a):
                d0 = self._rr[a]+self._rr[b]-(x[b]-x[a]).norm(dim=1)
                self._pairs = torch.cat([self._pairs, torch.stack([a, b], 1)]); self._pd0 = torch.cat([self._pd0, d0]); self._pw = torch.cat([self._pw, pw])
                self.self_stats['born'] += len(a)
                nf = nab[face]
                self.self_births.append(torch.stack([torch.full_like(d0, self._self_calls), a.to(f64), b.to(f64), d0, (self.st.X[b]-self.st.X[a]).norm(dim=1),
                                                     (nf*self._m_prev[a]).sum(1), (nf*self._m_prev[b]).sum(1)], 1).cpu())
            self._cap_prev, self._m_prev = self._caps(G)
        m = len(self._pairs); K = self.selfc['K']
        if m > K:
            raise RuntimeError('self contact: %d pairs over the capacity %d' % (m, K))
        t = self.tsc
        t['i'].zero_(); t['k'].zero_(); t['d0'].fill_(1e300); t['Rs'].zero_(); t['Reff'].fill_(1.)
        if m:
            P = self._pairs; ra, rb = self._rr[P[:, 0]], self._rr[P[:, 1]]
            t['i'][:m] = P[:, 0].to(torch.int32); t['k'][:m] = P[:, 1].to(torch.int32)
            t['Rs'][:m] = ra+rb; t['d0'][:m] = self._pd0; t['Reff'][:m] = ra*rb/(ra+rb)*self._pw**2       # (k_self's force goes as sqrt(Reff): the weight)
        t['xn'].copy_(x)
        self.self_stats['pairs'] = m

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
        zz = lambda dt_: wp.zeros(n, dtype=dt_, device='cuda')
        B = dict(S=z(n, wp.mat33d), Fc=z(n, wp.mat33d), J=wp.zeros(n, dtype=mat99, device='cuda'), Jr=wp.zeros(n, dtype=mat99, device='cuda'),
                 C=zz(mat99), Fx=zz(wp.mat33d), y0=zz(wp.mat33d), y1=zz(wp.mat33d), dS=zz(wp.mat33d), G=zz(wp.mat33d),
                 Fw=zz(wp.mat33d), Gbar=zz(wp.mat33d), Ga=zz(wp.mat33d),
                 cstep=wp.zeros(n, dtype=wp.float64, device='cuda'),
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
        if self.cap_ul:
            U = self.ul
            wp.launch(k_Gul, U['K'], inputs=[x, U['xn'], U['S'], U['O'], U['R']], outputs=[B['S']], record_tape=False)
            wp.launch(k_Aul, U['Ka'], inputs=[B['S'], U['As'], U['Ao'], U['Aw']], outputs=[B['Ga']], record_tape=False)
            wp.launch(k_Ful, n, inputs=[B['S'], B['Ga'], U['used'], U['F0']], outputs=[B['Fc']], record_tape=False)
            if tape is not None:
                def adj_ul():
                    wp.launch(k_Ful_adj, n, inputs=[B['Fc'].grad, U['used'], U['F0']], outputs=[B['Gbar']])
                    wp.launch(k_Aul_adj, U['Ka'], inputs=[B['Fc'].grad, U['F0'], U['As'], U['Ao'], U['Aw']], outputs=[B['Gbar']])
                    wp.launch(k_Gul_adj, U['K'], inputs=[B['Gbar'], U['S'], U['O'], U['R']], outputs=[x.grad])
                tape.record_func(adj_ul, arrays=[x, B['Fc']])
        else:
            wp.launch(k_S_bonds, self.nb, inputs=[x, self.bi, self.bj, self.A, self.nrm], outputs=[B['S']])
        if self.cap_ul:
            pass
        elif self.cap_wall:
            wp.launch(k_Fw, n, inputs=[B['S'], self.Mbi], outputs=[B['Fw']])
            cw = [self.Mti, self.Mc, self.t1, self.t2, self.mA, self.PA, self.kapw]
            wp.launch(k_closW, n, inputs=[B['S'], B['Fw'], self.off, self.ek, self.ew]+cw, outputs=[B['G'], B['Fc']])
            if tape is not None:
                def adj():
                    wp.launch(k_closW_adj, n, inputs=[self.off]+cw+[B['G'], B['Fc'].grad], outputs=[B['S'].grad, B['Gbar']])
                    wp.launch(k_Fw_adj, n, inputs=[B['Gbar'], self.offT, self.eTb, self.eTw, self.Mbi], outputs=[B['S'].grad])
                tape.record_func(adj, arrays=[B['S'], B['Fc']])
        elif self.cap_pos:
            cp = [self.poff, self.pk, self.pdX, self.M2i, self.Mti, self.Mc, self.t1, self.t2, self.mA, self.PA, self.kapw, self.capnr, self.capeps]
            wp.launch(k_closP, n, inputs=[x, B['S']]+cp, outputs=[B['G'], B['Fc']])
            if tape is not None:
                tape.record_func(lambda: wp.launch(k_closP_adj, n, inputs=cp+[B['G'], B['Fc'].grad], outputs=[B['S'].grad, x.grad]),
                                 arrays=[x, B['S'], B['Fc']])
        else:
            self._closure_a(B)
            if tape is not None:
                tape.record_func(lambda: self._closure_a_adj(B), arrays=[B['S'], B['Fc']])
        wp.launch(k_F8, n, inputs=[B['Fc']], outputs=[B['F'], B['R'], B['q']])
        hand = self.hand_adjoints and tape is not None
        bonds_in = [x, B['q'], self.bi, self.bj, self.mri, self.mrj, self.n0, self.kb, rf['d'], rf['th'], self.matp, pl['dp'], pl['thp']]
        if hand:                                                                    # (the gradient pass: forward and adjoint fused)
            wp.launch(k_bonds_fused, self.nb, inputs=bonds_in+[self.tw], outputs=[B['terms'], B['bd'], B['bth'], x.grad, B['q'].grad], record_tape=False)
        else:
            wp.launch(k_bonds, self.nb, inputs=bonds_in, outputs=[B['terms'], B['bd'], B['bth']])
        walls_in = [x, B['F'], B['R'], self.bi, self.bj, self.hi, self.hj, self.L, self.wri, self.wrj, self.wt, self.A, self.nrm, self.wi, self.wj]
        if not self.cap_ul:                                                         # (cap_ul: J = det F, Vdet = V)
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
        sc = self.selfc
        if sc is not None:
            wp.launch(k_self, sc['K'], inputs=[x, sc['xn'], sc['i'], sc['k'], sc['Rs'], sc['d0'], sc['Reff'], bl['par'], sc['Esp']],
                      outputs=[sc['fn'], sc['nv'], B['terms']], record_tape=False)
            if tape is not None:
                tape.record_func(lambda: wp.launch(k_self_adj, sc['K'], inputs=[sc['i'], sc['k'], bl['par'], sc['fn'], sc['nv'], B['terms'].grad],
                                                   outputs=[x.grad]), arrays=[x, B['terms']])
        wp.launch(k_total, (NT, 256), inputs=[B['terms']], outputs=[B['tsum']], record_tape=False)
        wp.launch(k_total_loss, 1, inputs=[B['tsum'], self.tw], outputs=[B['loss']], record_tape=False)
        if tape is not None:
            tape.record_func(lambda: wp.launch(k_total_adj, (NT, 256), inputs=[B['tsum'].grad, B['loss'].grad, self.tw], outputs=[B['terms'].grad]),
                             arrays=[B['terms'], B['tsum'], B['loss']])

    def prepare(self, graph=True):
        """the gradient pass (zero, forward, backward) as one CUDA graph; grad(x) copies x in and replays it. Scheme A's
        sweeps per solve n_lin from the block-Jacobi spectral radius rho at rest: rho^n_lin <= 1e-13 (+4)"""
        self.B = self._buffers()
        zz = lambda dt_: wp.zeros(self.n, dtype=dt_, device='cuda')
        self.sv = dict(F=zz(wp.mat33d), S=zz(wp.mat33d), J=zz(mat99), perm=zz(vec9), C=zz(mat99))
        self.cold()
        if self.cap_pos or self.cap_wall or self.cap_ul:                            # (the one-way caps, UL's F: no sweeps)
            self.rho, self.n_lin = 0., 0
        else:
            self.rho = self._rho()
            assert self.rho < 1, 'scheme A: the coupled closure\'s block-Jacobi does not contract (rho %.3f)' % self.rho
            self.n_lin = self.sweep_mult*max(8, int(math.ceil(math.log(1e-13)/math.log(max(self.rho, 1e-3))))+4)
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
                if k_ in ('S', 'Ga', 'Vd', 'terms', 'tsum'):
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
        """the closure's warm start reset to I (rest) with its blocks"""
        self._Fw.copy_(torch.eye(3, dtype=f64, device=self._Fw.device).repeat(self.n, 1, 1))
        self.closure_lu()

    def closure_lu(self):
        """the closure's blocks (D's LU, C) rebuilt at the warm start (after a reset, or kappa's change in set_material)"""
        self._blocks(self.Fw, self.B)

    def _blocks(self, F, B):
        wp.launch(k_closA_blocks, self.n, inputs=[F, self.off, self.ek, self.ew, self.Mt, self.Mc, self.t1, self.t2, self.mA, self.PA, self.kapw],
                  outputs=[B['J'], B['perm'], B['C']])

    def _closure_a(self, B):
        """scheme A's closure into B['Fc'] (and the warm start Fw), with its blocks at the result. Main: Newton
        block-Jacobi sweeps from Fw (n_lin of them, each body's own block rebuilt at every sweep's start; 10-28: with the
        blocks frozen at the start (chord) the new lion's step 428 diverged from a warm start whose residual was 1.6e-6:
        an inverted surface cell, very sensitive to its rim, swung out of the frozen blocks' reach and the chord sweeps
        grew x4-6 each; rebuilt, the sweeps converge to the same root, the cell still inverted, det -0.477), cstep the
        last sweep's step. linear_closure
        (TrustCG's products): from the root saved at z (save_closure), F = F_z + J_z^-1 (S - S_z) by n_lin linear sweeps
        (exact to O(|dS|^2)), the blocks then rebuilt at F for the adjoint"""
        n, N = self.n, self.n_lin
        geo = [self.off, self.ek, self.ew]
        if not self.linear_closure:
            Fa, Fb = self.Fw, B['Fx']
            for _ in range(N):
                self._blocks(Fa, B)                                                    # each sweep's own blocks at its start
                wp.launch(k_closA_sweep, n, inputs=[B['S'], Fa]+geo+[self.Mt, self.Mc, self.t1, self.t2, self.mA, self.PA, self.kapw, B['J'], B['perm']],
                          outputs=[Fb, B['cstep']])
                Fa, Fb = Fb, Fa
            wp.launch(k_mat_lin, n, inputs=[Fa, 1., Fa, 0.], outputs=[B['Fc']])
            if Fa is not self.Fw:
                wp.launch(k_mat_lin, n, inputs=[Fa, 1., Fa, 0.], outputs=[self.Fw])
        else:
            sv = self.sv
            wp.launch(k_mat_lin, n, inputs=[B['S'], 1., sv['S'], -1.], outputs=[B['dS']])
            B['y0'].zero_()
            ya, yb = B['y0'], B['y1']
            for _ in range(N):
                wp.launch(k_lin_sweep, n, inputs=[B['dS'], ya]+geo+[sv['C'], sv['J'], sv['perm']], outputs=[yb])
                ya, yb = yb, ya
            wp.launch(k_mat_lin, n, inputs=[sv['F'], 1., ya, 1.], outputs=[B['Fc']])
            wp.launch(k_mat_lin, n, inputs=[sv['F'], 1., ya, 1.], outputs=[self.Fw])
        self._blocks(B['Fc'], B)

    def _closure_a_adj(self, B):
        """S_bar += J^-T Fc_bar (block-Jacobi on J^T, n_lin sweeps, the blocks at Fc)"""
        n = self.n
        B['y0'].zero_()
        la, lb = B['y0'], B['y1']
        for _ in range(self.n_lin):
            wp.launch(k_lin_sweep_t, n, inputs=[B['Fc'].grad, la, self.offT, self.eTb, self.eTw, B['C'], B['J'], B['perm']], outputs=[lb])
            la, lb = lb, la
        wp.launch(k_mat_add_to, n, inputs=[la], outputs=[B['S'].grad])

    def save_closure(self):
        """the last evaluation's closure (the accepted z: F, S, D's LU, C) as the linearised products' base"""
        B, sv, n = self.B, self.sv, self.n
        wp.launch(k_mat_lin, n, inputs=[B['Fc'], 1., B['Fc'], 0.], outputs=[sv['F']])
        wp.launch(k_mat_lin, n, inputs=[B['S'], 1., B['S'], 0.], outputs=[sv['S']])
        wp.copy(sv['J'], B['J']); wp.copy(sv['perm'], B['perm']); wp.copy(sv['C'], B['C'])

    def _rho(self, its=400):
        """the block-Jacobi iteration's spectral radius at the blocks now (rest): power iteration on y -> -D^-1 O y"""
        B, n = self.B, self.n
        ty0 = wp.to_torch(B['y0'], requires_grad=False); ty1 = wp.to_torch(B['y1'], requires_grad=False)
        g = torch.Generator(device='cuda').manual_seed(0)
        ty0.copy_(torch.randn(ty0.shape, dtype=f64, device='cuda', generator=g)); B['dS'].zero_()
        rho = 0.
        for k in range(its):
            a, b = (B['y0'], B['y1']) if k % 2 == 0 else (B['y1'], B['y0'])
            na = float(wp.to_torch(a, requires_grad=False).norm())
            wp.launch(k_lin_sweep, n, inputs=[B['dS'], a, self.off, self.ek, self.ew, B['C'], B['J'], B['perm']], outputs=[b])
            tb = wp.to_torch(b, requires_grad=False); nb = float(tb.norm())
            rho = nb/na
            tb.div_(nb)
        return rho

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


# the trust-region Newton-CG of the midpoint step (2026-10-27, user: "改"; it replaces the line search, whose fall-back took
# any finite trial once t < 1e-6 - on the 30k jack a Pi rise of 1e107 - and whose CG had no bound on its step and drifted
# the closure's warm start through its products): Steihaug-Toint preconditioned CG in the preconditioner's norm, every
# product from the closure's state saved at z. Scalars in tr:
TR_RZ, TR_DHD, TR_PMP, TR_PMD, TR_DMD, TR_RR, TR_RZN, TR_AL, TR_BE, TR_PMAX, TR_DONE, TR_DONE0, TR_EDGE, TR_NEG, TR_DELTA, TR_STOP, \
    TR_K, TR_CAP, TR_GP, TR_PHP, TR_NPMP = range(21)


@wp.kernel(enable_backward=False)
def k_tr_init(g: wp.array(dtype=wp.vec3d), Pinv: wp.array(dtype=wp.mat33d), r: wp.array(dtype=wp.vec3d), y: wp.array(dtype=wp.vec3d),
              d: wp.array(dtype=wp.vec3d), p: wp.array(dtype=wp.vec3d), Hp: wp.array(dtype=wp.vec3d), tr: wp.array(dtype=wp.float64)):
    """r = -g, y = P^-1 r, d = y, p = H p = 0; r . y into tr"""
    i = wp.tid()
    ri = -g[i]
    yi = Pinv[i]@ri
    r[i] = ri
    y[i] = yi
    d[i] = yi
    zero = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
    p[i] = zero
    Hp[i] = zero
    wp.atomic_add(tr, TR_RZ, wp.dot(ri, yi))


@wp.kernel(enable_backward=False)
def k_tr_start(tr: wp.array(dtype=wp.float64)):
    """|d|_M^2 = r . y, p = 0; the region (tr[TR_DELTA] <= 0: the preconditioned gradient's length |P^-1 g|_M)"""
    tr[TR_DMD] = tr[TR_RZ]
    if not (tr[TR_DELTA] > wp.float64(0.0)):
        tr[TR_DELTA] = wp.sqrt(tr[TR_RZ])


@wp.kernel(enable_backward=False)
def k_tr_absmax(d: wp.array(dtype=wp.vec3d), tr: wp.array(dtype=wp.float64)):
    """|d|_max (the product's scale); the exit flag as at the iteration's start"""
    i = wp.tid()
    if i == 0:
        tr[TR_DONE0] = tr[TR_DONE]
    a = d[i]
    wp.atomic_max(tr, TR_PMAX, wp.max(wp.abs(a[0]), wp.max(wp.abs(a[1]), wp.abs(a[2]))))


@wp.kernel(enable_backward=False)
def k_tr_copy(Fa: wp.array(dtype=wp.mat33d), Ja: wp.array(dtype=mat99), pa: wp.array(dtype=vec9), Jra: wp.array(dtype=mat99),
              Fb: wp.array(dtype=wp.mat33d), Jb: wp.array(dtype=mat99), pb: wp.array(dtype=vec9), Jrb: wp.array(dtype=mat99)):
    """the closure's state (warm start, LU, pivots, Jacobian) a -> b"""
    b = wp.tid()
    Fb[b] = Fa[b]
    Jb[b] = Ja[b]
    pb[b] = pa[b]
    Jrb[b] = Jra[b]


@wp.kernel(enable_backward=False)
def k_tr_perturb(z: wp.array(dtype=wp.vec3d), d: wp.array(dtype=wp.vec3d), tr: wp.array(dtype=wp.float64), h: wp.float64,
                 xw: wp.array(dtype=wp.vec3d)):
    i = wp.tid()
    xw[i] = z[i]+(h/wp.max(tr[TR_PMAX], wp.float64(1e-300)))*d[i]


@wp.kernel(enable_backward=False)
def k_tr_hd(gE: wp.array(dtype=wp.vec3d), g0: wp.array(dtype=wp.vec3d), d: wp.array(dtype=wp.vec3d), cm: wp.array(dtype=wp.float64),
            tr: wp.array(dtype=wp.float64), h: wp.float64, Hd: wp.array(dtype=wp.vec3d)):
    """H d = (4 m / dt^2) d + (dE/dx(z + h d / |d|_max) - dE/dx(z)) |d|_max / h; d . H d into tr"""
    i = wp.tid()
    hd = cm[i]*d[i]+(gE[i]-g0[i])*(wp.max(tr[TR_PMAX], wp.float64(1e-300))/h)
    Hd[i] = hd
    wp.atomic_add(tr, TR_DHD, wp.dot(d[i], hd))


@wp.kernel(enable_backward=False)
def k_tr_alpha(tr: wp.array(dtype=wp.float64)):
    """the step along d: CG's alpha, or to the region's edge (negative curvature, or past it), which ends the CG"""
    if tr[TR_DONE0] > wp.float64(0.0):
        return
    tr[TR_K] = tr[TR_K]+wp.float64(1.0)
    pmp = tr[TR_PMP]
    pmd = tr[TR_PMD]
    dmd = wp.max(tr[TR_DMD], wp.float64(1e-300))
    d2 = tr[TR_DELTA]*tr[TR_DELTA]
    tau = (-pmd+wp.sqrt(wp.max(pmd*pmd+dmd*(d2-pmp), wp.float64(0.0))))/dmd
    dhd = tr[TR_DHD]
    if dhd <= wp.float64(0.0):
        tr[TR_AL] = tau
        tr[TR_EDGE] = wp.float64(1.0)
        tr[TR_NEG] = wp.float64(1.0)
        tr[TR_DONE] = wp.float64(1.0)
        tr[TR_NPMP] = d2
    else:
        al = tr[TR_RZ]/dhd
        npmp = pmp+wp.float64(2.0)*al*pmd+al*al*dmd
        if npmp >= d2:
            tr[TR_AL] = tau
            tr[TR_EDGE] = wp.float64(1.0)
            tr[TR_DONE] = wp.float64(1.0)
            tr[TR_NPMP] = d2
        else:
            tr[TR_AL] = al
            tr[TR_NPMP] = npmp
    tr[TR_RZN] = wp.float64(0.0)
    tr[TR_RR] = wp.float64(0.0)


@wp.kernel(enable_backward=False)
def k_tr_update(d: wp.array(dtype=wp.vec3d), Hd: wp.array(dtype=wp.vec3d), Pinv: wp.array(dtype=wp.mat33d), tr: wp.array(dtype=wp.float64),
                p: wp.array(dtype=wp.vec3d), Hp: wp.array(dtype=wp.vec3d), r: wp.array(dtype=wp.vec3d), y: wp.array(dtype=wp.vec3d)):
    """p += alpha d, H p += alpha H d; the residual and its preconditioned form unless this step ended the CG"""
    i = wp.tid()
    if tr[TR_DONE0] > wp.float64(0.0):
        return
    al = tr[TR_AL]
    p[i] = p[i]+al*d[i]
    Hp[i] = Hp[i]+al*Hd[i]
    if tr[TR_DONE] > wp.float64(0.0):
        return
    ri = r[i]-al*Hd[i]
    r[i] = ri
    yi = Pinv[i]@ri
    y[i] = yi
    wp.atomic_add(tr, TR_RZN, wp.dot(ri, yi))
    wp.atomic_add(tr, TR_RR, wp.dot(ri, ri))


@wp.kernel(enable_backward=False)
def k_tr_beta(tr: wp.array(dtype=wp.float64)):
    """|p|_M^2; the stop (|r| <= stop or cap products) or the next direction's beta and M-products"""
    if tr[TR_DONE0] == wp.float64(0.0):
        tr[TR_PMP] = tr[TR_NPMP]
        if tr[TR_DONE] == wp.float64(0.0):
            if wp.sqrt(wp.max(tr[TR_RR], wp.float64(0.0))) <= tr[TR_STOP] or tr[TR_K] >= tr[TR_CAP]:
                tr[TR_DONE] = wp.float64(1.0)
            else:
                be = tr[TR_RZN]/wp.max(tr[TR_RZ], wp.float64(1e-300))
                tr[TR_BE] = be
                tr[TR_PMD] = be*(tr[TR_PMD]+tr[TR_AL]*tr[TR_DMD])
                tr[TR_DMD] = tr[TR_RZN]+be*be*tr[TR_DMD]
                tr[TR_RZ] = tr[TR_RZN]
    tr[TR_PMAX] = wp.float64(0.0)
    tr[TR_DHD] = wp.float64(0.0)


@wp.kernel(enable_backward=False)
def k_tr_dir(y: wp.array(dtype=wp.vec3d), tr: wp.array(dtype=wp.float64), d: wp.array(dtype=wp.vec3d)):
    i = wp.tid()
    if tr[TR_DONE] > wp.float64(0.0):
        return
    d[i] = y[i]+tr[TR_BE]*d[i]


@wp.kernel(enable_backward=False)
def k_tr_pred(g: wp.array(dtype=wp.vec3d), p: wp.array(dtype=wp.vec3d), Hp: wp.array(dtype=wp.vec3d), tr: wp.array(dtype=wp.float64)):
    """g . p and p . H p (the model's decrease -(g . p + p . H p / 2))"""
    i = wp.tid()
    wp.atomic_add(tr, TR_GP, wp.dot(g[i], p[i]))
    wp.atomic_add(tr, TR_PHP, wp.dot(p[i], Hp[i]))


class TrustCG:
    """the midpoint step's trust-region subproblem on the GPU (Steihaug-Toint: preconditioned CG from p = 0 in the
    preconditioner's norm |p|_M = (p^T P p)^1/2; a direction of negative curvature or a step past the region ends it at
    the region's edge; else |r| <= stop or cap products). Products H d = 4 m / dt^2 d + (dE/dx(z + h d / |d|_max) - g0)
    |d|_max / h with the closure's state (warm start, LU, Jacobian) as saved at z (save() after every accepted point)
    restored before each one: the products' product_iters closure steps all start at z's root, as g0 did (the line
    search's ClosureCG let each product's result be the next one's start: on the 30k jack a drift of 3e-5 in one cell's F
    and a curvature -120 where the true one is +52). Per graph `per` iterations; nothing changes after the exit. Arrays
    in a: z, p (the step; MidpointStep's dz), g0 (dE/dx at z), Pinv, cm = 4 m / dt^2, tr (scalars)"""
    def __init__(self, W, mass, dt, per=2, h=1e-9, product_iters=1):
        self.W, self.per, self.h = W, per, float(h)
        n = W.n; dev = 'cuda'; B = W.B
        z = lambda dt_: wp.zeros(n, dtype=dt_, device=dev)
        self.a = dict(z=z(wp.vec3d), p=z(wp.vec3d), d=z(wp.vec3d), r=z(wp.vec3d), y=z(wp.vec3d), Hd=z(wp.vec3d), Hp=z(wp.vec3d),
                      g0=z(wp.vec3d), Pinv=z(wp.mat33d), cm=wp.from_torch((4*mass.to(f64)/dt**2).contiguous(), dtype=wp.float64),
                      tr=wp.zeros(24, dtype=wp.float64, device=dev), Fs=z(wp.mat33d), Js=z(mat99), ps=z(vec9), Jrs=z(mat99))
        self.t = {k_: wp.to_torch(a_, requires_grad=False) for k_, a_ in self.a.items()
                  if k_ in ('z', 'p', 'd', 'r', 'y', 'Hd', 'Hp', 'g0', 'Pinv', 'cm', 'tr')}
        W.linear_closure = True                                                     # (the products: the closure linearised at z)

        def iteration():
            A = self.a
            wp.launch(k_tr_absmax, n, inputs=[A['d']], outputs=[A['tr']])
            wp.launch(k_tr_perturb, n, inputs=[A['z'], A['d'], A['tr'], self.h], outputs=[W.xw])
            W._run()                                                              # (no tape.reset here: it zeroes the gradients)
            wp.launch(k_tr_hd, n, inputs=[W.xw.grad, A['g0'], A['d'], A['cm'], A['tr'], self.h], outputs=[A['Hd']])
            wp.launch(k_tr_alpha, 1, inputs=[A['tr']])
            wp.launch(k_tr_update, n, inputs=[A['d'], A['Hd'], A['Pinv'], A['tr']], outputs=[A['p'], A['Hp'], A['r'], A['y']])
            wp.launch(k_tr_beta, 1, inputs=[A['tr']])
            wp.launch(k_tr_dir, n, inputs=[A['y'], A['tr']], outputs=[A['d']])
        W.cold()
        self.t['z'].copy_(W.st.X)                                                  # (the warm-up at rest, a finished CG: nothing
        W.linear_closure = False; W.grad(W.st.X); self.save(); W.linear_closure = True   # moves; compiled outside the capture)
        self.t['tr'].zero_(); self.t['tr'][TR_DONE] = 1.
        iteration()
        wp.capture_begin()
        for _ in range(per):
            iteration()
        self.graph = wp.capture_end()
        W.linear_closure = False
        W.cold()

    def save(self):
        """the closure at W's last evaluation (the accepted z) as the linearised products' base"""
        self.W.save_closure()

    def solve(self, g, delta, stop, cap):
        """the subproblem for the gradient g (a Warp array) with z, g0, Pinv in a; delta <= 0: the preconditioned
        gradient's length. Returns dict(K products, EDGE, NEG, PMP |p|_M^2, DELTA, GP g . p, PHP p . H p); p in a['p']"""
        A, n = self.a, self.W.n
        tr = self.t['tr']
        tr.zero_(); tr[TR_DELTA] = float(delta); tr[TR_STOP] = float(stop); tr[TR_CAP] = float(cap)
        wp.launch(k_tr_init, n, inputs=[g, A['Pinv']], outputs=[A['r'], A['y'], A['d'], A['p'], A['Hp'], A['tr']])
        wp.launch(k_tr_start, 1, inputs=[A['tr']])
        it = 0
        while it < cap+self.per:
            wp.capture_launch(self.graph)
            it += self.per
            if tr[TR_DONE].item() > 0:
                break
        wp.launch(k_tr_pred, n, inputs=[g, A['p'], A['Hp']], outputs=[A['tr']])
        v = tr.tolist()
        return dict(K=int(v[TR_K]), EDGE=v[TR_EDGE] > 0, NEG=v[TR_NEG] > 0, PMP=v[TR_PMP], DELTA=v[TR_DELTA], GP=v[TR_GP], PHP=v[TR_PHP])


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
    x, v, a (torch views tx, tv, ta), the glue's held force and the viscosity's references stay there; CG = TrustCG(W, ...)
    (its z, g0, Pinv and p are this step's); Kbb the bodies' rest blocks (closure_drop.probe_blocks), fg the external
    forces, muf the ground's friction. step(tol_abs, cg_tol_abs) -> (gradient evaluations, Newton iterations, |grad Pi|^2),
    then end(); after W.set_material, update_material().
    Newton (10-27): trust-region Newton-CG on Pi (TrustCG's subproblem; the region carried through the step's iterations,
    started at the first preconditioned gradient's length): a trial is taken only when Pi falls by more than 1e-4 of the
    model's predicted decrease (or both are within round-off of 0); the region shrinks to 1/4 of the step when the ratio
    is below 1/4 (a rejected trial: z re-evaluated, the closure's state saved again) and doubles when above 3/4 at the
    edge; CG to max(cg_tol_abs(|g|^2), eta |g|). The Pi evaluations' closure: up to eval_iters Newton steps from the last
    root (each body stopping at its 1e-10 step). stats: the last step's rejected trials, edge and negative-curvature exits"""
    def __init__(self, W, CG, mass, Kbb, fg, dt, muf, eval_iters=12, eta=.01, cg_cap=300, newton_max=50):
        self.W, self.CG, self.dt = W, CG, float(dt)
        self.eta, self.cg_cap, self.newton_max = float(eta), int(cg_cap), int(newton_max)
        n = W.n; dev = 'cuda'
        z = lambda dt_, k=n: wp.zeros(k, dtype=dt_, device=dev)
        A = CG.a
        self.a = dict(x=z(wp.vec3d), v=z(wp.vec3d), a=z(wp.vec3d), xt=z(wp.vec3d), gz=z(wp.vec3d), P=z(wp.mat33d), mv=z(wp.vec3d, 1),
                      m=wp.from_torch(mass.to(f64).contiguous(), dtype=wp.float64), cm=A['cm'], z=A['z'], g0=A['g0'], Pinv=A['Pinv'], dz=A['p'],
                      K=wp.from_torch(Kbb.to(f64).contiguous(), dtype=wp.mat33d), fg=wp.from_torch(fg.to(f64).contiguous(), dtype=wp.vec3d))
        self.ps = wp.zeros(8, dtype=wp.float64, device=dev)
        tt = lambda a_: wp.to_torch(a_, requires_grad=False)
        self.tx, self.tv, self.ta, self.tps, self.tK = tt(self.a['x']), tt(self.a['v']), tt(self.a['a']), tt(self.ps), tt(self.a['K'])
        self.K0, self.mu0 = self.tK.clone(), W.mu
        self.mtot = float(mass.sum()); self.c0 = float(W.tball[2])
        self.update_material(muf)
        self.stats = {}

        def pi():
            A = self.a
            wp.launch(k_mid_query, n, inputs=[A['z'], A['dz'], self.ps], outputs=[W.xw])
            W._run()
            wp.launch(k_mid_pi, n, inputs=[W.xw, A['xt'], A['x'], A['cm'], A['fg'], W.xw.grad, W.B['loss']], outputs=[A['gz'], A['g0'], self.ps])
        tt(self.a['z']).copy_(W.st.X); tt(self.a['x']).copy_(W.st.X); tt(self.a['xt']).copy_(W.st.X)
        it0 = W.iters
        W.iters = int(eval_iters)                                                   # (the evaluations' closure steps)
        pi()                                                                        # (compile at rest, outside the capture)
        wp.capture_begin()
        pi()
        self.graph = wp.capture_end()
        W.iters = it0
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

    def _accept(self):
        """the last evaluation is z: Pi, |g|^2 kept, the closure's state saved for the products"""
        wp.launch(k_mid_accept, self.W.n, inputs=[self.W.xw, self.ps], outputs=[self.a['z']])
        self.CG.save()

    def step(self, tol_abs, cg_tol_abs):
        W, A, bl, n = self.W, self.a, self.W.ball, self.W.n
        if W.cap_ul:
            W.update_ul(self.tx)                                                   # (radii, cut, the fit at the step's start)
        if W.selfc is not None:
            W.update_self(self.tx)                                                 # (the self contact list at the step's start)
        wp.launch(k_mid_begin, n, inputs=[A['x'], A['v'], A['a'], self.dt], outputs=[A['xt'], A['z']])
        wp.launch(k_mid_ball, bl['k'], inputs=[A['x'], bl['body'], bl['R'], bl['ng'], self.c0, self.g8, self.cb], outputs=[bl['xn'], bl['kt'], bl['bound']])
        g2 = self._eval(0.)[PS_GQ2]
        self._accept()
        evals, its = 1, 0
        st = dict(rejected=0, edge=0, negative=0)
        delta, delta0 = -1., None
        while its < self.newton_max:
            if math.sqrt(g2) <= tol_abs:
                break
            its += 1
            wp.launch(k_mid_pre, n, inputs=[W.B['R'], A['K'], self.c1, A['cm']], outputs=[A['P']])
            wp.launch(k_mid_pre_ball, bl['k'], inputs=[bl['body'], bl['R'], W.B['bdelta'], bl['ng'], self.e4, bl['kt'], bl['bound'], self.muf],
                      outputs=[A['P']])
            wp.launch(k_mid_inv, n, inputs=[A['P']], outputs=[A['Pinv']])
            r = self.CG.solve(A['gz'], delta, max(cg_tol_abs(g2), self.eta*math.sqrt(g2)), self.cg_cap)
            evals += r['K']; st['edge'] += r['EDGE']; st['negative'] += r['NEG']
            delta = r['DELTA']
            if delta0 is None:
                delta0 = delta
            pred = -(r['GP']+.5*r['PHP'])
            s = self._eval(1.); evals += 1
            ok = math.isfinite(s[PS_PQ]) and math.isfinite(s[PS_GQ2])
            noise = 1e-14*abs(s[PS_PZ])
            ared = s[PS_PZ]-s[PS_PQ] if ok else -math.inf
            # (10-27: with no predicted decrease the ratio is not 1: the lion's step 339 took Pi rises of 2.98 and 1.97 so)
            rho = ared/pred if pred > noise else (1. if ared >= -noise else -math.inf)
            pM = delta if r['EDGE'] else math.sqrt(max(r['PMP'], 0.))
            if rho < .25 or pred <= noise:
                delta = .25*pM
            elif rho > .75 and r['EDGE']:
                delta = 2*delta
            if ok and rho > 1e-4:
                self._accept()
                g2 = s[PS_GQ2]
            else:
                st['rejected'] += 1
                g2 = self._eval(0.)[PS_GQ2]; evals += 1                              # back at z (its gradient, g0, the closure)
                self._accept()
                if not delta > 1e-12*delta0:                                          # the region collapsed: no decrease found
                    break
        self.stats = st
        return evals, its, g2

    def end(self):
        """the step's end at the accepted midpoint (W's buffers hold its evaluation)"""
        W, A, bl, n = self.W, self.a, self.W.ball, self.W.n
        if W.cap_ul:
            W.end_ul()
        if W.vis is not None:
            wp.launch(k_mid_visc_bonds, W.nb, inputs=[W.B['bd'], W.B['bth']], outputs=[W.ref['d'], W.ref['th']])
            wp.launch(k_mid_visc_bodies, n, inputs=[W.B['R'], W.B['F'], W.B['lj']], outputs=[W.ref['P'], W.ref['L']])
        if W.cap_eps > 0:
            wp.launch(k_capref, n, inputs=[W.B['G'], W.poff, W.t1, W.t2, W.capeps], outputs=[W.capnr])
        wp.launch(k_mid_lamt, bl['k'], inputs=[W.B['bff'], W.B['bdelta']], outputs=[bl['lamt']])
        wp.launch(k_mid_end, n, inputs=[A['z'], self.dt, A['m']], outputs=[A['x'], A['v'], A['a'], A['mv']])
        wp.launch(k_mid_vmax, 1, inputs=[A['mv'], self.mtot], outputs=[self.ps])

    def vmax(self):
        return self.tps[PS_VMAX].item()
