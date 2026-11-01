"""The ring spring with a yield limit, no switch: springs carry the force at every state (2026-11-01). Moduli per unit density, masses = volumes.
The corner to give up. A uniform flow balances only if every cell reads one stress; its virial is then that stress times M_i, and M_i is round only in the sum. So a uniform flow, an incremental potential and a convex cell cannot all hold (ring_flow.py's header). The potential is only what a trust-region solver asks for. Given up here: the model is a non-associated plasticity cell by cell, yield in the Frobenius norm of the strain, elastic metric M_i.
  state     q per half spring (the rest vector until the cell yields), P per cell for the outer layer's virtual springs q_k = P D_k, x. F_g = I + G(x) on the rest cut. A cell whose plastic strain passes eps_cut goes on the current cut: F_g = (I + G) F_g,n, P = I, L a running sum. eps_cut is a tolerated error of the flow stress: on the rest cut M carries the plastic stretch, M = 2 mu V U_p^2, and the plastic work per strain reads (2 e^(2 eps) + e^-eps) / 3, 1 + eps, of s_y (a uniform path, ring_yield_uniform.py: M's diagonal to .01). On the cut no such term
  reading   A = sum w d (x) q + F_g Mo P^T, M = sum w q (x) q + P Mo P^T, F = A M^-1, the whole cell's fit; R = polar(A), the whole cell's minimum, off the cut; polar(F) on it (patch, below); in its frame H = R^T F - I; the step's start H_n = R_n^T F_n - I, kept in the material frame so a spin reads nothing; the step's end H_e = (H(z) - (1 - th) H_n) / th. Yield on the tensor only, as the MPM: beta = min(1, e_y / |dev sym H_e|). Not on the remainder: a cell's wiggles are not its strain (the user, 10-31)
  force     half spring f = w [beta X K q + (1 - beta)(1 - th) R e_n K q + beta_r (d - F q) + (1 - beta_r)(1 - th) R r_n], X = F - (1 + v) R, e_n = dev H_n, r_n = R_n^T (d_n - F_n q); the outer layer through F_g: [(beta X + (1 - beta)(1 - th) R e_n) K P + beta_r (F_g - F P) + (1 - beta_r)(1 - th) R R_n^T (F_g,n - F_n P)] Mo. Pressure V0 kap_b L on L. Off the cut K = I, v = 0, kap_b = lam + (1 - beta) 2 mu / 3: the springs' volumetric part goes down with beta and the pressure takes it, so a flowing cell has no v R, whose negative stiffness in compression, |v| 2 mu V, does not stay when the shape's stiffness has gone. beta = beta_r = 1: w (d - R q), exactly ring_ul.py. No switch
  volume    on the cut one reading: L at kap, whatever beta; the fit's volume is free: v = tr H / 3, read and frozen with beta. Two readings, the fit's rho 2 mu / 3 by beta and L's, pumped a cell whose beta moves: on a settled body |L| grew from .15 to .57, KE 3.7e-4 to 6.2e-4 in .07 s. No explicit term: with H_n whole in the start's term and L - (1 - th) L_n in the pressure the two v_n cancel only on M = 2 mu V I; on rho = tr M / 6 mu V a flowing cell keeps (rho - 1) mu v_n against the implicit 3 kap v: in the stiff limit L_n+1 = -(1 + .31 (rho - 1)) L_n, a cell of the skin at rho 2: L -.62 to +1.09 in two steps, 20 % of the start's energy made
  patch     a uniform flow is what the potential was given up for, so it must balance: F uniform must give X K uniform. So on the cut R = polar(F), H = U - I, e = dev H, whatever M_i. With polar(F M) each cell read its own skew part 1/2 [e, D] of M's anisotropy D: |g| / s_y V^2/3 .09 median, .41 largest on the deep sites of a uniform flow, and in flow only beta_r w holds that, r / L about .28 e_y / beta_r, .03, measured .02: a landing's remainder took 23.5 % of the plastic work, with this 15 %. Now .0009, .0047, of order e_y D. The price: a cell alone is not free of torque, skew(R M_t F^T) of order e D; the sum is zero on a uniform state, as the sum of M_i is round: a free flight's angular momentum off by .19 % in 1.5 periods, all in plastic flow. Off the cut polar(F M) as the ring, whose elastic stiffness takes the same force at r / L .28 e
  metric    on the cut the walls are the current ones and q = F_e^-1 d, so M = F_e^-1 (sum w d (x) d) F_e^-T, about 2 mu V U_e^-2: with it the flow stress is off by 2 e (.936 at e_y .1 on a uniform path). The energy of the strain belongs on the moment of the walls as they stand, in the material frame: M_t = U M U, U the stretch of F_n: E_t = 1/2 beta tr(X M_t X^T), dE/dF = beta X M_t, a half's force w beta X K q, K = M_t M^-1, about U^2, one matrix on a uniform state. So 1/2 tr(H M_t H^T) is mu V |U - I|^2 at any elastic strain: the ring's elastic law. V is the rest's: M_t scaled by V0 / V of the walls as they stand, or a loop in shear and volume does work, .48 of its peak's energy at amp .1 (ring_yield_elastic.py, 11-02), 1e-10 with it. Off the cut M_t = M, K = I. (Weights pulled back by Nanson's formula instead: the current walls are not the affine image of an intermediate cut, M about 2 mu V (I - 1.2 e): .973.)
  remainder a Maxwell element, the stabilisation of a virtual element method, F being the projection (the normal equations): its share of the plastic work falls with h. beta_r = 1 / (1 + kappa (1 / beta - 1)), kappa = 5 at w (ring_flow.py's header), beta of the last step's end. Read within the step, at a reading just over the limit beta_r falls kappa times beta and lets go w r, |r| / L far over e_y: the force's change many times the tensor's, a gain over 1 on the outer map's fixed point, which then turns (.960, 1.283 e_y; on a settled body hundreds of cells at once). The first-order lag is a choice of time discretisation of the rate. kappa has a bound from above: a flowing cell has negative second-order work |b|^2 / 4 of 2 mu in a direction, b the part of n D across n (the non-associated price), which only the remainder holds, against beta_r
  viscosity set_viscosity(beta, dt), c = beta / (th dt): 1/2 c tr(S M_s S) + 1/2 c sum w |r - R r_n|^2 + 1/2 c V0 kap (L - L_n)^2, S = dev sym((F - F_n) F_n^-1), M_s = R_n M_t R_n^T, in parallel with the limit (not by beta). A half's force + w (c Y F_n^-T M^-1 q + c (r - R r_n)), Y = dev sym(S M_s). S has no R: linear in F, nothing for the outer iteration to chase; on X - R e_n, which holds a spin against the frozen R with c M_t (c 120 on the lion), the outer iterations stood at 1.5 tol on landing. A spin reads only the midpoint's second order: a turn of 90 degrees at a stretch, its work 1 / steps^3 (64 down at 4 times the steps). The normal equations still drop the fit out of the remainder's part
  frozen    beta, beta_r, R, v, L's linearisation C frozen: the force is the gradient of a convex quadratic, by the fit's normal equations (sum w r (x) q + r_o Mo P^T = 0, so d/dx through F drops out of the remainder's energy, and M M^-1 cancels in the tensor's):
            E = 1/2 beta tr(X M_t X^T) + (1 - beta)(1 - th) tr(R e_n M_t X^T) + 1/2 beta_r W_r + (1 - beta_r)(1 - th) W_r,n + V0 kap_b L^2 / 2
  solver    so a local-global iteration (YieldStep): z of the frozen problem, by the trust-region CG with its energy ratio valid there; then beta, R, v, C read at z, until the true |g| is small; Anderson over the last aa_m iterates, the combination with the least true |g| by its differences (in the mass's norm of G - z a stiff direction's residual does not show: the combination stood still at 6 tol), back to the plain iterate when the true |g| rises by aa_back. After hold_after outer iterations a cell under beta .2 keeps its R and C: its force goes with beta X, so R's error costs little, and on a cell far gone R can flip between two rotations (dropped: the same state, five times the iterations)
  update    at the step's end, the polar F_e = R_e U of a cell that yields: U_e = a exp(c dev log U), c holding the limit in the reading's measure, |dev (U_e - I)| = e_y (on |dev log U_e| = e_y a cell at e_y .1 read 1.02 at rest, beta under 1 with no increment, plastic work 25 % too much), a = exp(L / 3): only its shape is a state, its volume L's. a's log within +-.5 (L is a linear reading, and a sets only q's scale); q -> U_e^-1 R_e^T (d_e - beta_r r_e), r_e = d_e - F_e q: so F' = R_e U_e and r' = beta_r r, and q comes from d itself, however flat F_e read. The outer layer's virtual springs the same, P -> U_e^-1 R_e^T (F_g - beta_r (F_g - F_e P)); on the cut, F_g -> R_e U_e + beta_r (F_g - F_e)
  merging   a cell under agg of the median volume is one cell with its neighbour across its largest wall: A, M summed, F, R, beta shared, its gradient the group's volume mean, V_g G_g = sum V_i G_i, which keeps the cancellation across walls
  the cut   eps_cut < 0: every cell on it from the first step, no first yield, the elastic part too: the springs are rewritten on the newest walls every step, a uniform elastic loop comes back to 1e-13; a wave's F_e drifts by (amp^2 / steps) a loop. The state is the fit: F_e, what the last step's return left, and the cut only re-weights the quadrature of it. At fixed x and q new weights read F'' = A'' M''^-1, not F_e, as the normal equations hold the remainder on the old weights only: F'' - F_e = sum dw r (x) q M^-1, (dw / w)(r / L), against a limit of a percent: it was read as stress at the step's start, H_n to 89 e_y. So all of a cell's q and its outer layer go through T = F_e^-1 F'', one map: A = A'' T^T, M = T M'' T^T, F = F'' T^-1 = F_e exactly, the remainder is d - F'' q, the least for the new weights. A new spring reads none, q = F_e^-1 d, and leaves the fit alone. A cell's T too far from I (det under .1): all its q = F_e^-1 d, the remainder dropped. Each cell's q stand in its own frame, F_e^-1 d: a cell that changes group goes first through F_e,group^-1 F_e, which keeps its remainder, or the difference of the two F reads as one (that was the cut's energy: .019 of the springs to .001). Between two cells on the cut the wall's spring 2 w goes by t, the wall's place on the line of the sites (clipped .02 .. .98): M_i then has each pyramid's moment and a thin cell's M scales with its volume
  outer     on the cut q_k = F_e^-1 D_k, D_k the cut's own vectors, so Mo = F_e^-1 Mo_cur F_e^-T in q's frame (until 11-01 Mo_cur itself: right on no rotated cell), P = I, and F_g = (I + G) F_g,n reads q_k. F_g,n = F_e + s E, the remainder E = F_g - F_e on the part of the moment it had, s = tr Mo_old / tr Mo up to 1 (a cap born reads none, as a spring). (P F_g^-1 with F_g -> I, the same to first order, was tried: F_g over one step reads singular values .07 to 6 on a cell whose neighbours lie in a plane: the outer layer lost 70 % of the start's energy at the cuts, and made it in the steps)
Self contact not here. set_ball, start(), prepare(), update_ul; through YieldStep. ring_yield_check.py its checks, ring_yield_uniform.py a uniform path against closed forms, ring_yield_part.py, ring_yield_drop.py."""
import math

import numpy as np
import torch
import warp as wp

from free_surface.closure_warp import MidpointStep, PS_GQ2, PS_PQ, PS_PZ, k_Aul, k_F8, k_Gul, k_Gul_adj, k_ball_adj, k_mid_ball, k_mid_begin, k_mid_inv, k_mid_pre, k_mid_pre_ball, rot_proper
from free_surface.merged_ul import MergedUL
from free_surface.ring_flow import RingFlow, k_rf_F, k_rf_Fadj, k_rf_ball
from free_surface.skin_warp import NSLOT, NT, k_total, k_total_loss
from free_surface.ul_plastic import k_Aul_adj2, k_Ge, k_Ge_adj, k_gg_cell, k_gg_cell_adj, k_gg_wall, k_gg_wall_adj, k_var

f64 = torch.float64


@wp.kernel(enable_backward=False)
def k_ry_gs(G: wp.array(dtype=wp.mat33d), V0: wp.array(dtype=wp.float64), gid: wp.array(dtype=int), Gs: wp.array(dtype=wp.mat33d)):
    b = wp.tid()
    wp.atomic_add(Gs, gid[b], V0[b]*G[b])


@wp.kernel(enable_backward=False)
def k_ry_gm(Gs: wp.array(dtype=wp.mat33d), Vgi: wp.array(dtype=wp.float64), gid: wp.array(dtype=int), Gm: wp.array(dtype=wp.mat33d)):
    b = wp.tid()
    g = gid[b]
    Gm[b] = Vgi[g]*Gs[g]


@wp.kernel(enable_backward=False)
def k_ry_as(Gbm: wp.array(dtype=wp.mat33d), gid: wp.array(dtype=int), Ss: wp.array(dtype=wp.mat33d)):
    b = wp.tid()
    wp.atomic_add(Ss, gid[b], Gbm[b])


@wp.kernel(enable_backward=False)
def k_ry_ab(Ss: wp.array(dtype=wp.mat33d), V0: wp.array(dtype=wp.float64), Vgi: wp.array(dtype=wp.float64), gid: wp.array(dtype=int), Gb: wp.array(dtype=wp.mat33d)):
    b = wp.tid()
    g = gid[b]
    Gb[b] = (V0[b]*Vgi[g])*Ss[g]


@wp.kernel(enable_backward=False)
def k_ry_L(Fc: wp.array(dtype=wp.mat33d), Gm: wp.array(dtype=wp.mat33d), C: wp.array(dtype=wp.mat33d), Ln: wp.array(dtype=wp.float64), V0: wp.array(dtype=wp.float64), lam: wp.float64, k2: wp.float64, flag: wp.array(dtype=int), gid: wp.array(dtype=int),
           be: wp.array(dtype=wp.float64), Lz: wp.array(dtype=wp.float64), Cz: wp.array(dtype=wp.mat33d), Fz: wp.array(dtype=wp.mat33d), cv: wp.float64, L: wp.array(dtype=wp.float64), Lb: wp.array(dtype=wp.float64), terms: wp.array2d(dtype=wp.float64)):
    """L on its frozen linearisation; the springs' volumetric share that beta has let go, on the pressure; the viscous pressure on L - L_n"""
    b = wp.tid()
    half = wp.float64(0.5)
    lb = Lz[b]+wp.ddot(Cz[b], Fc[b]-Fz[b])
    if flag[b] != 0:
        lb = Ln[b]+wp.ddot(C[b], Gm[b])
    L[b] = lb
    s = lam+k2
    if flag[b] == 0:
        s = lam+k2*(wp.float64(1.0)-be[gid[b]])
    dl = lb-Ln[b]
    wp.atomic_add(terms, 1, b % NSLOT, V0[b]*half*s*(lb*lb+cv*dl*dl))
    wp.atomic_add(Lb, b, V0[b]*s*(lb+cv*dl))


@wp.kernel(enable_backward=False)
def k_ry_Lb(Lb: wp.array(dtype=wp.float64), C: wp.array(dtype=wp.mat33d), Cz: wp.array(dtype=wp.mat33d), flag: wp.array(dtype=int), Gbm: wp.array(dtype=wp.mat33d), Gbrm: wp.array(dtype=wp.mat33d)):
    b = wp.tid()
    if flag[b] != 0:
        Gbm[b] = Gbm[b]+Lb[b]*C[b]
    else:
        Gbrm[b] = Gbrm[b]+Lb[b]*Cz[b]


@wp.kernel(enable_backward=False)
def k_ry_A(x: wp.array(dtype=wp.vec3d), ri: wp.array(dtype=int), rj: wp.array(dtype=int), wi: wp.array(dtype=wp.float64), wj: wp.array(dtype=wp.float64), qi: wp.array(dtype=wp.vec3d), qj: wp.array(dtype=wp.vec3d), gid: wp.array(dtype=int), Am: wp.array(dtype=wp.mat33d)):
    e = wp.tid()
    i, j = ri[e], rj[e]
    if i == j:
        return
    d = x[j]-x[i]
    wp.atomic_add(Am, gid[i], wi[e]*wp.outer(d, qi[e]))
    wp.atomic_add(Am, gid[j], wj[e]*wp.outer(d, qj[e]))


@wp.kernel(enable_backward=False)
def k_ry_Aout(Fc: wp.array(dtype=wp.mat33d), MoP: wp.array(dtype=wp.mat33d), gid: wp.array(dtype=int), Am: wp.array(dtype=wp.mat33d)):
    b = wp.tid()
    wp.atomic_add(Am, gid[b], Fc[b]@MoP[b])


@wp.kernel(enable_backward=False)
def k_ry_Ff(Am: wp.array(dtype=wp.mat33d), Mti: wp.array(dtype=wp.mat33d), Ff: wp.array(dtype=wp.mat33d)):
    b = wp.tid()
    Ff[b] = Am[b]@Mti[b]


@wp.kernel(enable_backward=False)
def k_ry_cell(Ff: wp.array(dtype=wp.mat33d), Hn: wp.array(dtype=wp.mat33d), Rz: wp.array(dtype=wp.mat33d), Fc: wp.array(dtype=wp.mat33d), Ron: wp.array(dtype=wp.mat33d), Pc: wp.array(dtype=wp.mat33d), Mo: wp.array(dtype=wp.mat33d),
              be: wp.array(dtype=wp.float64), br: wp.array(dtype=wp.float64), ca: wp.float64, gid: wp.array(dtype=int), vm: wp.array(dtype=wp.float64), Mtt: wp.array(dtype=wp.mat33d), Kt: wp.array(dtype=wp.mat33d), Vz: wp.array(dtype=wp.mat33d), cv: wp.float64, Fb: wp.array(dtype=wp.mat33d), terms: wp.array2d(dtype=wp.float64)):
    """the tensor's energy once per group, term 0; the outer layer's remainder, term 5, with its viscous part; dE/dF_g, Vz the fit's viscous part"""
    b = wp.tid()
    half = wp.float64(0.5)
    one = wp.float64(1.0)
    g = gid[b]
    X = Ff[g]-vm[g]*Rz[g]
    Xn = Rz[g]@Hn[g]
    bb = be[g]
    cc = br[g]
    if g == b:
        wp.atomic_add(terms, 0, b % NSLOT, half*bb*wp.ddot(X@Mtt[b], X)+(one-bb)*ca*wp.ddot(Xn@Mtt[b], X))
    R0 = Fc[b]-Ff[g]@Pc[b]
    Rn = Rz[g]@Ron[b]
    Rv = R0-Rn
    wp.atomic_add(terms, 5, b % NSLOT, half*cc*wp.ddot(R0@Mo[b], R0)+(one-cc)*ca*wp.ddot(Rn@Mo[b], R0)+half*cv*wp.ddot(Rv@Mo[b], Rv))
    Fb[b] = (((bb*X+((one-bb)*ca)*Xn)@(Kt[g]@Pc[b]))+(Vz[g]@Pc[b])+cc*R0+((one-cc)*ca)*Rn+cv*Rv)@Mo[b]


@wp.kernel(enable_backward=False)
def k_ry_visc(Ff: wp.array(dtype=wp.mat33d), Fn: wp.array(dtype=wp.mat33d), Fni: wp.array(dtype=wp.mat33d), Ms: wp.array(dtype=wp.mat33d), Mti: wp.array(dtype=wp.mat33d), gid: wp.array(dtype=int), cv: wp.float64, Vz: wp.array(dtype=wp.mat33d), terms: wp.array2d(dtype=wp.float64)):
    """the fit's viscous energy on S = dev sym((F - F_n) F_n^-1), per group, term 0; Vz = dPsi/dF M^-1, what a half's q and the outer layer's P take"""
    b = wp.tid()
    if gid[b] != b:
        return
    half = wp.float64(0.5)
    third = wp.float64(1.0)/wp.float64(3.0)
    I = wp.identity(n=3, dtype=wp.float64)
    l = (Ff[b]-Fn[b])@Fni[b]
    s = half*(l+wp.transpose(l))
    S = s-(third*wp.trace(s))*I
    SM = S@Ms[b]
    Y = half*(SM+wp.transpose(SM))
    Y = Y-(third*wp.trace(Y))*I
    wp.atomic_add(terms, 0, b % NSLOT, half*cv*wp.ddot(SM, S))
    Vz[b] = cv*(Y@(wp.transpose(Fni[b])@Mti[b]))


@wp.kernel(enable_backward=False)
def k_ry_half(x: wp.array(dtype=wp.vec3d), ri: wp.array(dtype=int), rj: wp.array(dtype=int), wi: wp.array(dtype=wp.float64), wj: wp.array(dtype=wp.float64), qi: wp.array(dtype=wp.vec3d), qj: wp.array(dtype=wp.vec3d), dni: wp.array(dtype=wp.vec3d), dnj: wp.array(dtype=wp.vec3d), gid: wp.array(dtype=int),
              Ff: wp.array(dtype=wp.mat33d), Hn: wp.array(dtype=wp.mat33d), Rz: wp.array(dtype=wp.mat33d), be: wp.array(dtype=wp.float64), br: wp.array(dtype=wp.float64), ca: wp.float64, vm: wp.array(dtype=wp.float64), Kt: wp.array(dtype=wp.mat33d), Vz: wp.array(dtype=wp.mat33d), cv: wp.float64, terms: wp.array2d(dtype=wp.float64), x_adj: wp.array(dtype=wp.vec3d)):
    """each half's force, the springs carry it; the remainder's energy, term 4"""
    e = wp.tid()
    i, j = ri[e], rj[e]
    if i == j:
        return
    half = wp.float64(0.5)
    one = wp.float64(1.0)
    d = x[j]-x[i]
    gi = gid[i]
    q = qi[e]
    bb = be[gi]
    cc = br[gi]
    w = wi[e]
    r = d-Ff[gi]@q
    rn = Rz[gi]@dni[e]
    qk = Kt[gi]@q
    f = w*(bb*((Ff[gi]-vm[gi]*Rz[gi])@qk)+((one-bb)*ca)*(Rz[gi]@(Hn[gi]@qk))+cc*r+((one-cc)*ca)*rn+Vz[gi]@q+cv*(r-rn))
    wp.atomic_add(terms, 4, e % NSLOT, half*cc*w*wp.dot(r, r)+(one-cc)*ca*w*wp.dot(rn, r)+half*cv*w*wp.dot(r-rn, r-rn))
    gj = gid[j]
    q = qj[e]
    bb = be[gj]
    cc = br[gj]
    w = wj[e]
    r = d-Ff[gj]@q
    rn = Rz[gj]@dnj[e]
    qk = Kt[gj]@q
    f = f+w*(bb*((Ff[gj]-vm[gj]*Rz[gj])@qk)+((one-bb)*ca)*(Rz[gj]@(Hn[gj]@qk))+cc*r+((one-cc)*ca)*rn+Vz[gj]@q+cv*(r-rn))
    wp.atomic_add(terms, 4, e % NSLOT, half*cc*w*wp.dot(r, r)+(one-cc)*ca*w*wp.dot(rn, r)+half*cv*w*wp.dot(r-rn, r-rn))
    wp.atomic_add(x_adj, j, f)
    wp.atomic_add(x_adj, i, -f)


@wp.kernel(enable_backward=False)
def k_ry_start(Fn: wp.array(dtype=wp.mat33d), Mt: wp.array(dtype=wp.mat33d), Mtt: wp.array(dtype=wp.mat33d), gid: wp.array(dtype=int), gy: wp.array(dtype=int), Rn: wp.array(dtype=wp.mat33d), Hn: wp.array(dtype=wp.mat33d), Hd: wp.array(dtype=wp.mat33d)):
    """the step's start in the material frame, per group"""
    b = wp.tid()
    if gid[b] != b:
        return
    R = rot_proper(Fn[b]@Mtt[b])
    if gy[b] != 0:
        R = rot_proper(Fn[b])
    Rn[b] = R
    I = wp.identity(n=3, dtype=wp.float64)
    H = wp.transpose(R)@Fn[b]-I
    Hn[b] = H
    Hd[b] = H-(wp.trace(H)/wp.float64(3.0))*I


@wp.kernel(enable_backward=False)
def k_ry_read(Ff: wp.array(dtype=wp.mat33d), Hn: wp.array(dtype=wp.mat33d), Mt: wp.array(dtype=wp.mat33d), ey: wp.float64, th_: wp.float64, hold: int, gid: wp.array(dtype=int), gy: wp.array(dtype=int), Rz: wp.array(dtype=wp.mat33d), be: wp.array(dtype=wp.float64), ei: wp.array(dtype=wp.float64), vm: wp.array(dtype=wp.float64)):
    """the frozen part, per group, at the last evaluation"""
    b = wp.tid()
    if gid[b] != b:
        return
    one = wp.float64(1.0)
    I = wp.identity(n=3, dtype=wp.float64)
    R = Rz[b]
    if hold == 0 or be[b] >= wp.float64(0.2):
        if gy[b] != 0:
            R = rot_proper(Ff[b])
        else:
            R = rot_proper(Ff[b]@Mt[b])
        Rz[b] = R
    Hz = wp.transpose(R)@Ff[b]-I
    vm[b] = one
    if gy[b] != 0:
        vm[b] = one+wp.trace(Hz)/wp.float64(3.0)
    H = (Hz-(one-th_)*Hn[b])/th_
    s = wp.float64(0.5)*(H+wp.transpose(H))
    e = s-(wp.trace(s)/wp.float64(3.0))*I
    nr = wp.sqrt(wp.ddot(e, e))
    ei[b] = nr
    bb = one
    if nr > ey:
        bb = ey/nr
    be[b] = bb


@wp.kernel(enable_backward=False)
def k_ry_readL(Fc: wp.array(dtype=wp.mat33d), flag: wp.array(dtype=int), Lz: wp.array(dtype=wp.float64), Cz: wp.array(dtype=wp.mat33d), Fz: wp.array(dtype=wp.mat33d)):
    """an elastic cell's L = ln det F_g, linearised at the last evaluation (at J under 0.05 continued linearly in J)"""
    b = wp.tid()
    if flag[b] != 0:
        return
    F = Fc[b]
    a0 = wp.vec3d(F[0, 0], F[1, 0], F[2, 0])
    a1 = wp.vec3d(F[0, 1], F[1, 1], F[2, 1])
    a2 = wp.vec3d(F[0, 2], F[1, 2], F[2, 2])
    c0 = wp.cross(a1, a2)
    c1 = wp.cross(a2, a0)
    c2 = wp.cross(a0, a1)
    J = wp.dot(a0, c0)
    J0 = wp.float64(0.05)
    lb = wp.log(J0)+(J-J0)/J0
    dl = wp.float64(1.0)/J0
    if J > J0:
        lb = wp.log(J)
        dl = wp.float64(1.0)/J
    Lz[b] = lb
    Fz[b] = F
    Cz[b] = wp.matrix_from_cols(dl*c0, dl*c1, dl*c2)


class RingYield(RingFlow):
    """X, r, V0: the foam at rest; pairs, A, c: its walls (geometry.npz); E, nu, sigma_y per unit density; kappa: the remainder's rate over the cell's; eps_cut: the plastic strain that takes a cell onto the current cut; agg: cells under this of the median volume are merged (0: none); hold, ledger: as ring_flow.py"""
    C_cap = 4.

    def __init__(self, X, r, V0, pairs, A, c, E, nu, sigma_y, theta=.5, kappa=5., eps_cut=-1., agg_merge=.1, hold=False, ledger=False, **kw):
        super().__init__(X, r, V0, pairs, A, c, E, nu, sigma_y, theta=theta, kappa=kappa, hold=hold, ledger=ledger, **kw)
        self.eps_cut, self.agg_merge = float(eps_cut), float(agg_merge)
        self.Wp_cell = self.Wp_half = 0.
        self._Ft = self._Fti = None
        self.Ecut_parts = [0., 0., 0., 0.]

    def start(self):
        n, dev = self.n, self.dev
        z = lambda k_, dt_: wp.zeros(k_, dtype=dt_, device=dev)
        Ks = self.ul['Kw']
        self.ry = dict(Ff=z(n, wp.mat33d), Fn=z(n, wp.mat33d), Rn=z(n, wp.mat33d), Hn=z(n, wp.mat33d), Hd=z(n, wp.mat33d), gy=z(n, wp.int32), vm=z(n, wp.float64), Mtt=z(n, wp.mat33d), Kt=z(n, wp.mat33d), Vz=z(n, wp.mat33d), Fni=z(n, wp.mat33d), Ms=z(n, wp.mat33d), Ron=z(n, wp.mat33d), Rz=z(n, wp.mat33d), Mt=z(n, wp.mat33d), Mti=z(n, wp.mat33d), MoP=z(n, wp.mat33d), Pc=z(n, wp.mat33d), be=z(n, wp.float64), br=z(n, wp.float64), ei=z(n, wp.float64), Lz=z(n, wp.float64), Cz=z(n, wp.mat33d), Fz=z(n, wp.mat33d),
                       gid=z(n, wp.int32), Vgi=z(n, wp.float64), Gs=z(n, wp.mat33d), Grs=z(n, wp.mat33d), Gm=z(n, wp.mat33d), Grm=z(n, wp.mat33d), Gbm=z(n, wp.mat33d), Gbrm=z(n, wp.mat33d), Ss=z(n, wp.mat33d), Srs=z(n, wp.mat33d))
        tt = lambda a_: wp.to_torch(a_, requires_grad=False)
        self.ty = {k_: tt(a_) for k_, a_ in self.ry.items()}
        Y = self.ty
        Y['Pc'].copy_(torch.eye(3, dtype=f64, device=dev).expand(n, 3, 3)); Y['be'].fill_(1.); Y['br'].fill_(1.); Y['vm'].fill_(1.); Y['Rz'].copy_(Y['Pc']); Y['Cz'].copy_(Y['Pc']); Y['Fz'].copy_(Y['Pc'])
        self._eps = torch.zeros(n, dtype=f64, device=dev)
        self._sj = torch.ones(n, dtype=f64, device=dev)
        super().start()
        T = self.tr
        self._Mo0 = T['Mo'].clone()
        self._gid_rest()
        self._state_start(self.X)
        self._xs = self.X.clone()
        if self.eps_cut < 0:                                                               # (on the cut from the first step: no first yield to wait for)
            self._yld[:] = True
            T['flag'][:] = 1
            self._Ft = self._Fe.clone()

    def _gid_rest(self):
        """the groups on the rest walls"""
        V = self.V0.cpu().numpy(); n = self.n
        gid = np.arange(n)
        if self.agg_merge > 0:
            tiny = V < self.agg_merge*np.median(V)
            lo, hi = self.tlo.cpu().numpy(), self.thi.cpu().numpy()
            A = (self.tk0/(2*self.kshare)*(self.X[self.thi]-self.X[self.tlo]).norm(dim=1)).cpu().numpy()    # (mu A: for the order)
            o = np.argsort(A)
            lo, hi = lo[o], hi[o]
            m1 = tiny[lo] & ~tiny[hi]; gid[lo[m1]] = hi[m1]
            m2 = tiny[hi] & ~tiny[lo]; gid[hi[m2]] = lo[m2]
        self._set_gid(torch.as_tensor(gid, device=self.dev))

    def _set_gid(self, gid):
        Y = self.ty
        Y['gid'].copy_(gid.to(torch.int32))
        S = torch.zeros(self.n, dtype=f64, device=self.dev).index_add(0, gid, self.V0)
        Y['Vgi'].copy_(1/S.clamp(min=1e-300))
        self._gid = gid

    def _set_halves(self, x):
        """the half springs by their pair at x: the kept q, a new one reading no remainder on its cell's fit; mu A / L each, by t between two cells on the cut"""
        T = self.tr
        lo, hi, key, ms = self.tlo, self.thi, self.tkey, self._m_s
        pos = torch.searchsorted(self._qkey, key).clamp(max=len(self._qkey)-1); kept = self._qkey[pos] == key
        d = x[hi]-x[lo]
        qi, qj = d.clone(), d.clone()
        qi[kept] = self._qi[pos[kept]]; qj[kept] = self._qj[pos[kept]]
        if not bool(kept.all()) and self._Fti is not None:
            nw = ~kept
            qi[nw] = torch.einsum('fab,fb->fa', self._Fti[lo[nw]], d[nw]); qj[nw] = torch.einsum('fab,fb->fa', self._Fti[hi[nw]], d[nw])
        self._qkey, self._qi, self._qj = key.clone(), qi, qj
        w = self.tk0/(2*self.kshare)
        share = torch.full((ms,), .5, dtype=f64, device=self.dev)
        i_w, j_w, _, _, _, m = self._w
        lo_w, hi_w = torch.minimum(i_w, j_w), torch.maximum(i_w, j_w)
        kw = lo_w*self.n+hi_w
        tlo_w = torch.where(i_w < j_w, self.tul['wt'][:m], 1-self.tul['wt'][:m])
        o = torch.argsort(kw)
        pw = torch.searchsorted(kw[o], key).clamp(max=m-1); hw = kw[o][pw] == key
        both = self._yld[lo] & self._yld[hi] & hw
        share[both] = tlo_w[o][pw[both]].clamp(.02, .98)
        for k_ in ('i', 'j', 'wi', 'wj', 'qi', 'qj', 'dni', 'dnj'):
            T[k_].zero_()
        T['i'][:ms] = lo.to(torch.int32); T['j'][:ms] = hi.to(torch.int32); T['wi'][:ms] = 2*share*w; T['wj'][:ms] = 2*(1-share)*w; T['qi'][:ms] = qi; T['qj'][:ms] = qj

    def _cut(self, x):
        """the current cut at x: merged_ul.py's list and gradient, the halves; a cut cell's outer layer pushed onto it; its group on the current walls; the fit carried across"""
        Mo_old = self.tr['Mo'].clone()
        MergedUL.update_ul(self, x)
        self._cut_valid = True
        self._Fti = self._inv(self._Ft)
        self._set_halves(x)
        t, T, Y, yld = self.tul, self.tr, self.ty, self._yld
        Mc = t['Mc']
        T['Mo'].copy_(self._Mo0)
        Fi_, Pc = self._Fti[yld], Y['Pc'][yld]
        Mq = Fi_@(self.mu*(Mc[yld]+Mc[yld].transpose(1, 2)))@Fi_.transpose(1, 2)
        T['Mo'][yld] = Mq
        tr3 = lambda a_: a_.diagonal(dim1=1, dim2=2).sum(1)
        tn = tr3(Mq)
        s_ = torch.where(tn > 0, tr3(Pc@Mo_old[yld]@Pc.transpose(1, 2))/tn.clamp(min=1e-300), torch.zeros_like(tn)).clamp(0., 1.)
        self._Fe[yld] = self._Ft[yld]+s_[:, None, None]*(self._Fe[yld]@self._inv(Pc)-self._Ft[yld])
        Y['Pc'][yld] = self.I3
        if self.ball is not None:
            self.tbR.copy_(torch.where(yld, self.tbR, self._tbR0))
            self.tbkn.copy_(2*self.Es*self.tbR.clamp(min=0.)*.05**.5)
        if self.agg_merge > 0:
            V = self.V0.cpu().numpy()
            tiny = V < self.agg_merge*np.median(V)
            i_w, j_w, _, _, _, m = self._w
            i_n, j_n, a_ = i_w.cpu().numpy(), j_w.cpu().numpy(), t['wa'][:m].cpu().numpy()
            o = np.argsort(a_); i_n, j_n = i_n[o], j_n[o]
            gid = self._gid.cpu().numpy().copy(); y_ = yld.cpu().numpy()
            m1 = tiny[i_n] & ~tiny[j_n] & y_[i_n]; gid[i_n[m1]] = j_n[m1]
            m2 = tiny[j_n] & ~tiny[i_n] & y_[j_n]; gid[j_n[m2]] = i_n[m2]
            self._set_gid(torch.as_tensor(gid, device=self.dev))
        self._wv_masks()
        self._carry_fit(x)

    def _inv(self, A):
        """the inverse of 3 x 3 matrices, the pseudo-inverse where near singular"""
        a0, a1, a2 = A[:, :, 0], A[:, :, 1], A[:, :, 2]; c0, c1, c2 = torch.linalg.cross(a1, a2), torch.linalg.cross(a2, a0), torch.linalg.cross(a0, a1)
        det = (a0*c0).sum(1); sing = det.abs() <= 1e-10*(A*A).sum((1, 2))**1.5
        Ai = torch.stack([c0, c1, c2], 1)/torch.where(sing, torch.ones_like(det), det)[:, None, None]
        if bool(sing.any()):
            Ai[sing] = torch.linalg.pinv(A[sing].cpu()).to(A.device)
        return Ai

    def _carry_fit(self, x):
        """all of a group's q and its outer layer through T = F_e^-1 F'', so that the fit on the new weights is F_e; a group with a cell on the cut only"""
        T, Y, ms, lo, hi, n, g = self.tr, self.ty, self._m_s, self.tlo, self.thi, self.n, self._gid
        P, Mo = Y['Pc'], T['Mo']
        wi, wj = T['wi'][:ms], T['wj'][:ms]
        d = x[hi]-x[lo]
        qi, qj = self._qi, self._qj
        on = torch.zeros(n, dtype=torch.bool, device=self.dev)
        on[g[self._yld | (self._eps > 0)]] = True
        ch = ((self._Ft-self._Ft[g]).abs().sum((1, 2)) > 0) & on[g]
        self.ul_stats['frames'] = int(ch.sum())
        if bool(ch.any()):
            T0 = self.I3.expand(n, 3, 3).clone()
            T0[ch] = self._Fti[g[ch]]@self._Ft[ch]
            qi, qj = torch.einsum('fab,fb->fa', T0[lo], qi), torch.einsum('fab,fb->fa', T0[hi], qj)
            y_, e_ = ch & self._yld, ch & ~self._yld
            Mo[y_] = T0[y_]@Mo[y_]@T0[y_].transpose(1, 2)
            self._Fe[y_] = self._Fe[y_]@self._inv(T0[y_])
            P[e_] = T0[e_]@P[e_]
        MoP = Mo@P.transpose(1, 2)
        M = torch.zeros(n, 3, 3, dtype=f64, device=self.dev).index_add(0, g, P@MoP)
        M.index_add_(0, g[lo], wi[:, None, None]*qi[:, :, None]*qi[:, None, :]); M.index_add_(0, g[hi], wj[:, None, None]*qj[:, :, None]*qj[:, None, :])
        A = torch.zeros(n, 3, 3, dtype=f64, device=self.dev).index_add(0, g, self._Fe@MoP)
        A.index_add_(0, g[lo], wi[:, None, None]*d[:, :, None]*qi[:, None, :]); A.index_add_(0, g[hi], wj[:, None, None]*d[:, :, None]*qj[:, None, :])
        k = torch.nonzero(on)[:, 0]
        Tk = self._Fti[k]@(A[k]@self._inv(M[k]))
        dt_ = torch.linalg.det(Tk)
        bad = ~(dt_ > .1) | ~torch.isfinite(Tk).all(dim=(1, 2))
        Tm = self.I3.expand(n, 3, 3).clone()
        Tm[k] = torch.where(bad[:, None, None], self.I3, Tk)
        reset = torch.zeros(n, dtype=torch.bool, device=self.dev)
        reset[k[bad]] = True
        qi2, qj2 = torch.einsum('fab,fb->fa', Tm[g[lo]], qi), torch.einsum('fab,fb->fa', Tm[g[hi]], qj)
        Tg = Tm[g]
        el = ~self._yld
        Y['Pc'][el] = Tg[el]@P[el]
        y = self._yld
        Mo[y] = Tg[y]@Mo[y]@Tg[y].transpose(1, 2)
        self._Fe[y] = self._Fe[y]@self._inv(Tg[y])
        if bool(reset.any()):
            ri, rj = reset[g[lo]], reset[g[hi]]
            qi2[ri] = torch.einsum('fab,fb->fa', self._Fti[g[lo[ri]]], d[ri]); qj2[rj] = torch.einsum('fab,fb->fa', self._Fti[g[hi[rj]]], d[rj])
            rc = reset[g]
            Y['Pc'][rc & el] = self._Fti[g[rc & el]]@self._Fe[rc & el]
            self._Fe[rc & y] = self._Ft[g[rc & y]]
        self._qi, self._qj = qi2, qj2
        T['qi'][:ms] = qi2; T['qj'][:ms] = qj2
        if self.ledger:
            dT = (Tk-self.I3).flatten(1).norm(dim=1)
            self.ul_stats['T_med'], self.ul_stats['T_max'], self.ul_stats['T_reset'] = float(dT.median()), float(dT.max()), int(bad.sum())

    def _mark_groups(self):
        """the groups with a cell on the cut; this step's beta_r, per group its lead's"""
        gy = self.ty['gy']
        gy.zero_()
        gy[self._gid[self._yld]] = 1
        b_ = self._be_end if hasattr(self, '_be_end') else self.ty['be'][self._gid]
        self.ty['br'].copy_(1/(1+self.kappa*(1/b_.clamp(min=1e-12)-1)))

    def _state_start(self, x):
        """the step's start: M, its inverse, the fit at x, each half's remainder there"""
        T, Y, ms, lo, hi, n = self.tr, self.ty, self._m_s, self.tlo, self.thi, self.n
        g = self._gid
        P, Mo = Y['Pc'], T['Mo']
        Y['MoP'].copy_(Mo@P.transpose(1, 2))
        wi, wj = T['wi'][:ms], T['wj'][:ms]
        qi, qj = T['qi'][:ms], T['qj'][:ms]
        M = torch.zeros(n, 3, 3, dtype=f64, device=self.dev).index_add(0, g, P@Y['MoP'])
        M.index_add_(0, g[lo], wi[:, None, None]*qi[:, :, None]*qi[:, None, :]); M.index_add_(0, g[hi], wj[:, None, None]*qj[:, :, None]*qj[:, None, :])
        lead = g == torch.arange(n, device=self.dev)
        a0, a1, a2 = M[:, :, 0], M[:, :, 1], M[:, :, 2]; c0, c1, c2 = torch.linalg.cross(a1, a2), torch.linalg.cross(a2, a0), torch.linalg.cross(a0, a1)
        det = (a0*c0).sum(1); sing = det.abs() <= 1e-10*(M*M).sum((1, 2))**1.5
        Mi = torch.stack([c0, c1, c2], 1)/torch.where(sing, torch.ones_like(det), det)[:, None, None]
        if bool((sing & lead).any()):
            Mi[sing] = torch.linalg.pinv(M[sing])
        Y['Mt'].copy_(M); Y['Mti'].copy_(Mi)
        d = x[hi]-x[lo]
        Fe = self._Fe
        A = torch.zeros(n, 3, 3, dtype=f64, device=self.dev).index_add(0, g, Fe@Y['MoP'])
        A.index_add_(0, g[lo], wi[:, None, None]*d[:, :, None]*qi[:, None, :]); A.index_add_(0, g[hi], wj[:, None, None]*d[:, :, None]*qj[:, None, :])
        Fn = A@Mi
        Y['Fn'].copy_(Fn)
        Y['Mtt'].copy_(M); Y['Kt'].copy_(self.I3.expand(n, 3, 3))
        on_ = torch.zeros(n, dtype=torch.bool, device=self.dev); on_[g[self._yld]] = True
        on_ = on_ & lead
        if bool(on_.any()):
            Us_, Ss_, Vh_ = torch.linalg.svd(Fn[on_].cpu())
            Un_ = ((Vh_.transpose(1, 2)*Ss_[:, None, :])@Vh_).to(self.dev)
            Mt_ = Un_@M[on_]@Un_
            Vc_ = torch.zeros(n, dtype=f64, device=self.dev).index_add(0, g, 1/self.tul['Vi'])
            self._sj[on_] = (1/(Y['Vgi']*Vc_))[on_]
            Mt_ = self._sj[on_][:, None, None]*Mt_
            Y['Mtt'][on_] = .5*(Mt_+Mt_.transpose(1, 2))
            Y['Kt'][on_] = Y['Mtt'][on_]@Mi[on_]
        self._mark_groups()
        wp.launch(k_ry_start, n, inputs=[self.ry['Fn'], self.ry['Mt'], self.ry['Mtt'], self.ry['gid'], self.ry['gy']], outputs=[self.ry['Rn'], self.ry['Hn'], self.ry['Hd']])
        if self.cv > 0.:
            Y['Fni'][lead] = self._inv(Fn[lead])
            Y['Ms'][lead] = Y['Rn'][lead]@Y['Mtt'][lead]@Y['Rn'][lead].transpose(1, 2)
        Rn = Y['Rn'][g]
        T['dni'][:ms] = torch.einsum('fba,fb->fa', Rn[lo], d-torch.einsum('fab,fb->fa', Fn[g[lo]], qi)); T['dnj'][:ms] = torch.einsum('fba,fb->fa', Rn[hi], d-torch.einsum('fab,fb->fa', Fn[g[hi]], qj))
        Y['Ron'].copy_(Rn.transpose(1, 2)@(Fe-Fn[g]@P))
        self.tsp['F0'].copy_(Fe)

    def update_ul(self, x):
        """the step's start at x: the cut, once a cell has gone on it"""
        x = x.detach().clone()
        cut = bool(self._yld.any())
        e0 = self._parts(x) if cut and self.ledger else None
        self._cut_valid = False
        self._xs = x
        if cut:
            self._cut(x)
        else:
            ms = self._m_s
            self.tr['qi'][:ms] = self._qi; self.tr['qj'][:ms] = self._qj                 # (the returns, which a cut would have set)
            if self._Ft is not None and bool((self._eps > 0).any()):
                self._Fti = self._inv(self._Ft)
                self._carry_fit(x)
        self._state_start(x)
        if cut and self.ledger:
            e1 = self._parts(x)
            self.Ecut += sum(e1)-sum(e0)
            self.Ecut_parts = [a_+b_-c_ for a_, b_, c_ in zip(self.Ecut_parts, e1, e0)]
            lead_ = self._gid == torch.arange(self.n, device=self.dev)
            on_ = self._yld[lead_]
            self.ul_stats['fit_gap'] = float((self.ty['Fn'][lead_][on_]-self._Ft[lead_][on_]).flatten(1).norm(dim=1).max()) if bool(on_.any()) else 0.
        if self.ledger:
            Y, g = self.ty, self._gid
            lead = g == torch.arange(self.n, device=self.dev)
            Hs = .5*(Y['Hn']+Y['Hn'].transpose(1, 2)); tr_ = Hs.diagonal(dim1=1, dim2=2).sum(1)/3
            ed = (Hs-tr_[:, None, None]*self.I3).flatten(1).norm(dim=1)/self.ey
            self.ul_stats['Hn_med'], self.ul_stats['Hn_max'] = float(ed[lead].median()), float(ed[lead].max())
            Lg = torch.zeros(self.n, dtype=f64, device=self.dev).index_add(0, g, self.V0*self.tul['Ln'])*Y['Vgi']
            gap = (tr_-(torch.exp(Lg/3)-1)).abs()
            self.ul_stats['v_gap_med'], self.ul_stats['v_gap_max'] = float(gap[lead].median()), float(gap[lead].max())
            self.ul_stats['v_max'] = float(tr_[lead].abs().max())

    def _parts(self, x):
        """the springs, the outer layer, volume, the ground at x"""
        _, b = self.stored(x)
        return [self._Es, self._Eo, b, self.ground_store(x)]

    def run(self):
        """E and dE/dx at xw, all frozen but x: both gradients and their groups, F_g, L; A, the fit; the cells, the halves; back through both gradients"""
        B, U, Ur, Br, Rg, Y, S, n, Kw = self.B, self.ul, self.ur, self.Br, self.ring, self.ry, self.sp, self.n, self.ul['Kw']
        x, g = self.xw, self.xw.grad
        for k_ in ('G0', 'Ga', 'Sg', 'Lb', 'Gbe', 'Gb0', 'terms', 'tsum'):
            B[k_].zero_()
        for k_ in ('G0', 'Ga', 'Sg', 'Gbe', 'Gb0'):
            Br[k_].zero_()
        for k_ in ('Gs', 'Grs', 'Ss', 'Srs'):
            Y[k_].zero_()
        Rg['Am'].zero_()
        g.zero_()
        ca, gid, V0 = wp.float64(1.-self.th), Y['gid'], U['V0']
        wall = [U['wi'], U['wj'], U['wa'], U['wn'], U['wt'], U['wtau']]
        wallr = [Ur['wi'], Ur['wj'], Ur['wa'], Ur['wn'], Ur['wt'], Ur['wtau']]
        wp.launch(k_Gul, 2*Kw, inputs=[x, U['xn'], U['S'], U['O'], U['R']], outputs=[B['G0']])
        wp.launch(k_Aul, Kw, inputs=[B['G0'], U['As'], U['Ao'], U['Aw']], outputs=[B['Ga']])
        wp.launch(k_Ge, n, inputs=[B['G0'], B['Ga'], U['used']], outputs=[B['Ge']])
        wp.launch(k_gg_wall, Kw, inputs=[x, U['xn']]+wall+[B['Ge']], outputs=[B['Sg']])
        wp.launch(k_gg_cell, n, inputs=[x, U['xn'], B['Sg'], B['Ge'], U['a'], U['Mc'], U['Vi']], outputs=[B['G']])
        wp.launch(k_ry_gs, n, inputs=[B['G'], V0, gid], outputs=[Y['Gs']])
        wp.launch(k_ry_gm, n, inputs=[Y['Gs'], Y['Vgi'], gid], outputs=[Y['Gm']])
        wp.launch(k_Gul, Ur['Kl'], inputs=[x, Ur['xn'], Ur['S'], Ur['O'], Ur['R']], outputs=[Br['G0']])
        wp.launch(k_Aul, Ur['Ka'], inputs=[Br['G0'], Ur['As'], Ur['Ao'], Ur['Aw']], outputs=[Br['Ga']])
        wp.launch(k_Ge, n, inputs=[Br['G0'], Br['Ga'], Ur['used']], outputs=[Br['Ge']])
        wp.launch(k_gg_wall, Ur['Kw'], inputs=[x, Ur['xn']]+wallr+[Br['Ge']], outputs=[Br['Sg']])
        wp.launch(k_gg_cell, n, inputs=[x, Ur['xn'], Br['Sg'], Br['Ge'], Ur['a'], Ur['Mc'], Ur['Vi']], outputs=[Br['G']])
        wp.launch(k_ry_gs, n, inputs=[Br['G'], V0, gid], outputs=[Y['Grs']])
        wp.launch(k_ry_gm, n, inputs=[Y['Grs'], Y['Vgi'], gid], outputs=[Y['Grm']])
        wp.launch(k_rf_F, n, inputs=[Y['Gm'], Y['Grm'], S['F0'], Rg['flag']], outputs=[B['Fc']])
        cv = wp.float64(self.cv)
        wp.launch(k_ry_L, n, inputs=[B['Fc'], Y['Gm'], U['Q'], U['Ln'], V0, wp.float64(self.lam), wp.float64(2*self.mu/3), Rg['flag'], gid, Y['be'], Y['Lz'], Y['Cz'], Y['Fz'], cv], outputs=[B['L'], B['Lb'], B['terms']])
        wp.launch(k_var, Kw, inputs=[U['wi'], U['wj'], U['wv'], B['L']], outputs=[B['Lb'], B['terms']])
        wp.launch(k_var, Ur['Kw'], inputs=[Ur['wi'], Ur['wj'], Ur['wv'], B['L']], outputs=[B['Lb'], B['terms']])
        hs = [Rg['i'], Rg['j'], Rg['wi'], Rg['wj'], Rg['qi'], Rg['qj']]
        wp.launch(k_ry_A, Rg['K'], inputs=[x]+hs+[gid], outputs=[Rg['Am']])
        wp.launch(k_ry_Aout, n, inputs=[B['Fc'], Y['MoP'], gid], outputs=[Rg['Am']])
        wp.launch(k_ry_Ff, n, inputs=[Rg['Am'], Y['Mti']], outputs=[Y['Ff']])
        if self.cv > 0.:
            wp.launch(k_ry_visc, n, inputs=[Y['Ff'], Y['Fn'], Y['Fni'], Y['Ms'], Y['Mti'], gid, cv], outputs=[Y['Vz'], B['terms']])
        wp.launch(k_ry_cell, n, inputs=[Y['Ff'], Y['Hd'], Y['Rz'], B['Fc'], Y['Ron'], Y['Pc'], Rg['Mo'], Y['be'], Y['br'], ca, gid, Y['vm'], Y['Mtt'], Y['Kt'], Y['Vz'], cv], outputs=[B['Fb'], B['terms']])
        wp.launch(k_rf_Fadj, n, inputs=[B['Fb'], S['F0'], Rg['flag']], outputs=[Y['Gbm'], Y['Gbrm']])
        wp.launch(k_ry_half, Rg['K'], inputs=[x]+hs+[Rg['dni'], Rg['dnj'], gid, Y['Ff'], Y['Hd'], Y['Rz'], Y['be'], Y['br'], ca, Y['vm'], Y['Kt'], Y['Vz'], cv], outputs=[B['terms'], g])
        wp.launch(k_ry_Lb, n, inputs=[B['Lb'], U['Q'], Y['Cz'], Rg['flag']], outputs=[Y['Gbm'], Y['Gbrm']])
        bl = self.ball
        if bl is not None:
            wp.launch(k_rf_ball, bl['k'], inputs=[x, bl['body'], bl['R'], bl['xn'], bl['ng'], bl['par'], bl['kt'], bl['bound'], bl['lamt'], self.hd['stuck'], self.hd['kn'], wp.float64(self.sy*1.5**.5)], outputs=[B['bdelta'], B['bfn'], B['bff'], B['terms']])
            wp.launch(k_ball_adj, bl['k'], inputs=[bl['body'], bl['ng'], bl['par'], B['bfn'], B['bff'], self.tones], outputs=[g])
        wp.launch(k_ry_as, n, inputs=[Y['Gbm'], gid], outputs=[Y['Ss']])
        wp.launch(k_ry_ab, n, inputs=[Y['Ss'], V0, Y['Vgi'], gid], outputs=[B['Gb']])
        wp.launch(k_ry_as, n, inputs=[Y['Gbrm'], gid], outputs=[Y['Srs']])
        wp.launch(k_ry_ab, n, inputs=[Y['Srs'], V0, Y['Vgi'], gid], outputs=[Br['Gb']])
        wp.launch(k_gg_cell_adj, n, inputs=[B['Gb'], U['a'], U['Mc'], U['Vi']], outputs=[g, B['Gs'], B['Gbe']])
        wp.launch(k_gg_wall_adj, Kw, inputs=wall+[B['Gs']], outputs=[g, B['Gbe']])
        wp.launch(k_Ge_adj, n, inputs=[B['Gbe'], U['used']], outputs=[B['Gb0']])
        wp.launch(k_Aul_adj2, Kw, inputs=[B['Gbe'], U['As'], U['Ao'], U['Aw']], outputs=[B['Gb0']])
        wp.launch(k_Gul_adj, 2*Kw, inputs=[B['Gb0'], U['S'], U['O'], U['R']], outputs=[g])
        wp.launch(k_gg_cell_adj, n, inputs=[Br['Gb'], Ur['a'], Ur['Mc'], Ur['Vi']], outputs=[g, Br['Gs'], Br['Gbe']])
        wp.launch(k_gg_wall_adj, Ur['Kw'], inputs=wallr+[Br['Gs']], outputs=[g, Br['Gbe']])
        wp.launch(k_Ge_adj, n, inputs=[Br['Gbe'], Ur['used']], outputs=[Br['Gb0']])
        wp.launch(k_Aul_adj2, Ur['Ka'], inputs=[Br['Gbe'], Ur['As'], Ur['Ao'], Ur['Aw']], outputs=[Br['Gb0']])
        wp.launch(k_Gul_adj, Ur['Kl'], inputs=[Br['Gb0'], Ur['S'], Ur['O'], Ur['R']], outputs=[g])
        wp.launch(k_total, (NT, 256), inputs=[B['terms']], outputs=[B['tsum']])
        wp.launch(k_total_loss, 1, inputs=[B['tsum'], self.tw], outputs=[B['loss']])

    def reread(self, beta=True, hold=False):
        """beta, R, v, C at the last evaluation's x, frozen for the next inner solve; hold: a cell under beta .2 keeps its R and C"""
        Y, n = self.ry, self.n
        wp.launch(k_ry_read, n, inputs=[Y['Ff'], Y['Hn'], Y['Mtt'], wp.float64(self.ey), wp.float64(self.th), int(hold), Y['gid'], Y['gy']], outputs=[Y['Rz'], Y['be'], Y['ei'], Y['vm']])
        wp.launch(k_ry_readL, n, inputs=[self.B['Fc'], self.ring['flag']], outputs=[Y['Lz'], Y['Cz'], Y['Fz']])
        if bool(self._yld.any()):
            t = self.ty
            M = self.I3+t['Gm']/(2*self.th)
            a0, a1, a2 = M[:, :, 0], M[:, :, 1], M[:, :, 2]; c0, c1, c2 = torch.linalg.cross(a1, a2), torch.linalg.cross(a2, a0), torch.linalg.cross(a0, a1)
            det = (a0*c0).sum(1); ok = det > 1e-3
            C = torch.where(ok[:, None, None], torch.stack([c0, c1, c2], 2)/torch.where(ok, det, torch.ones_like(det))[:, None, None], self.I3)
            big = C.flatten(1).norm(dim=1) > self.C_cap
            if bool(big.any()):
                U, S, Vh = torch.linalg.svd(M[big].cpu())
                C[big] = (U*(1/S.clamp(min=.25))[:, None, :]@Vh).to(self.dev)
            if hold:
                keep = self.ty['be'][self._gid] < .2
                C[keep] = self.tul['Q'][keep]
            self.tul['Q'].copy_(C)

    def end_ul(self, th=.5):
        """after the step (the buffers at the accepted z): the plastic work, the returns on q and P, F_g, L; a cell past eps_cut goes on the cut"""
        assert th == self.th == .5, 'the plastic work is written for the midpoint'
        t, s, T, Y, n, I3, ms = self.tul, self.tsp, self.tr, self.ty, self.n, self.I3, self._m_s
        lo, hi, g = self.tlo, self.thi, self._gid
        ca = 1-th
        be, br, R, M, Ff, Fn = Y['be'], Y['br'], Y['Rz'], Y['Mtt'], Y['Ff'], Y['Fn']
        Fe = (Ff-ca*Fn)/th
        lead = g == torch.arange(n, device=self.dev)
        Hn = Y['Hn']; He = (R.transpose(1, 2)@Ff-I3-ca*Hn)/th
        Hnd = Y['Hd']
        vd = He.diagonal(dim1=1, dim2=2).sum(1)/3
        Hed = He-vd[:, None, None]*I3
        wc = .5*(1-be)*(((Hnd@M)*Hed).sum((1, 2))+be*((Hed@M)*Hed).sum((1, 2)))
        wc_ = float(wc[lead].sum())
        xe = (self.tx-ca*self._xs)/th
        de = xe[hi]-xe[lo]
        wi, wj, qi, qj, rni, rnj = T['wi'][:ms], T['wj'][:ms], T['qi'][:ms], T['qj'][:ms], T['dni'][:ms], T['dnj'][:ms]
        rei = de-torch.einsum('fab,fb->fa', Fe[g[lo]], qi); rej = de-torch.einsum('fab,fb->fa', Fe[g[hi]], qj)
        ci, cj = br[g[lo]], br[g[hi]]
        rni_ = torch.einsum('fab,fb->fa', R[g[lo]], rni); rnj_ = torch.einsum('fab,fb->fa', R[g[hi]], rnj)
        wh = .5*wi*(1-ci)*((rni_*rei).sum(1)+ci*(rei*rei).sum(1)); wj_ = .5*wj*(1-cj)*((rnj_*rej).sum(1)+cj*(rej*rej).sum(1))
        wh_ = float(wh.sum()+wj_.sum())
        if self.ledger:
            xz = self.tx
            dz_, dd_ = xz[hi]-xz[lo], de-(self._xs[hi]-self._xs[lo])
            Rl, Fl, Hl, bl = R[g[lo]], Ff[g[lo]], Y['Hd'][g[lo]], be[g[lo]]
            Rh, Fh, Hh, bh = R[g[hi]], Ff[g[hi]], Y['Hd'][g[hi]], be[g[hi]]
            sm = Y['vm']
            Kl, Kh = Y['Kt'][g[lo]], Y['Kt'][g[hi]]
            fti = wi[:, None]*(bl[:, None]*torch.einsum('fab,fb->fa', (Fl-sm[g[lo]][:, None, None]*Rl)@Kl, qi)+((1-bl)*ca)[:, None]*torch.einsum('fab,fb->fa', Rl@Hl@Kl, qi))
            ftj = wj[:, None]*(bh[:, None]*torch.einsum('fab,fb->fa', (Fh-sm[g[hi]][:, None, None]*Rh)@Kh, qj)+((1-bh)*ca)[:, None]*torch.einsum('fab,fb->fa', Rh@Hh@Kh, qj))
            fri = wi[:, None]*(ci[:, None]*(dz_-torch.einsum('fab,fb->fa', Fl, qi))+((1-ci)*ca)[:, None]*rni_)
            frj = wj[:, None]*(cj[:, None]*(dz_-torch.einsum('fab,fb->fa', Fh, qj))+((1-cj)*ca)[:, None]*rnj_)
            self.W_tensor = getattr(self, 'W_tensor', 0.)+float(((fti+ftj)*dd_).sum())
            self.W_rem = getattr(self, 'W_rem', 0.)+float(((fri+frj)*dd_).sum())
        self.Wp_cell += wc_; self.Wp_half += wh_; self.Wp += wc_+wh_
        if self.cv > 0.:                                                                   # (the viscous work over the step: its force . dx = 2 Psi / th, Psi a quadratic of z - x_n)
            lv = (Ff-Fn)@Y['Fni']
            Sv = .5*(lv+lv.transpose(1, 2))
            Sv = Sv-(Sv.diagonal(dim1=1, dim2=2).sum(1)/3)[:, None, None]*I3
            dzv = self.tx[hi]-self.tx[lo]
            rvi = dzv-torch.einsum('fab,fb->fa', Ff[g[lo]], qi)-rni_; rvj = dzv-torch.einsum('fab,fb->fa', Ff[g[hi]], qj)-rnj_
            Rv = self.tFc-Ff[g]@Y['Pc']-R[g]@Y['Ron']
            dLv = self.tL-t['Ln']
            self.Wv += self.cv/th*(float((((Sv@Y['Ms'])*Sv).sum((1, 2)))[lead].sum())+float((wi*(rvi*rvi).sum(1)).sum()+(wj*(rvj*rvj).sum(1)).sum())+float(((Rv@T['Mo'])*Rv).sum())+float((self.V0*(self.lam+2*self.mu/3)*dLv*dLv).sum()))
        F0 = s['F0']
        Fgend = (self.tFc-ca*F0)/th
        flag = self._yld
        yn = lead & ((be < 1) | (br < 1))
        Re = R.clone()
        Fprime = Fe.clone()
        if bool(yn.any()):
            U_, S_, Vh = torch.linalg.svd(Fe[yn].cpu())
            sg = torch.linalg.det(U_@Vh)
            S2 = S_.clone(); S2[:, 2] = S2[:, 2]*sg
            U2 = U_.clone(); U2[:, :, 2] = U2[:, :, 2]*sg[:, None]
            Rp = U2@Vh
            V_ = Vh.transpose(1, 2)
            lw = torch.log(S2.clamp(min=1e-6)); lm = lw.mean(1, keepdim=True); dv = lw-lm
            nr = dv.norm(dim=1, keepdim=True).clamp(min=1e-300)
            Lg = torch.zeros(n, dtype=f64, device=self.dev).index_add(0, g, self.V0*self.tL)*Y['Vgi']
            Ly = (Lg[yn]/th-ca/th*(torch.zeros(n, dtype=f64, device=self.dev).index_add(0, g, self.V0*t['Ln'])*Y['Vgi'])[yn]).cpu()[:, None]    # (the group's L at the step's end, linear over the step)
            la = (Ly/3).clamp(-.5, .5)
            a = torch.exp(la)
            sc = torch.minimum(torch.ones_like(nr), (self.ey/a)/nr)
            for _ in range(3):                                                             # (the limit in the reading's own measure, |dev (U_e - I)| = e_y: on |dev log U_e| = e_y a cell at e_y .1 read 1.02 at rest, beta under 1 with no increment, plastic work 25 % too much)
                lam_ = torch.exp(la+sc*dv)
                nd_ = (lam_-lam_.mean(1, keepdim=True)).norm(dim=1, keepdim=True).clamp(min=1e-300)
                sc = torch.minimum(torch.ones_like(nr), sc*self.ey/nd_)
            lue = la+sc*dv
            Uei = torch.zeros(n, 3, 3, dtype=f64, device=self.dev)
            Uei[yn] = (V_*torch.exp(-lue)[:, None, :]@V_.transpose(1, 2)).to(self.dev)
            Fprime[yn] = (U2*torch.exp(lue)[:, None, :]@Vh).to(self.dev)
            Re[yn] = Rp.to(self.dev)
            eps = torch.zeros(n, dtype=f64, device=self.dev)
            eps[yn] = ((1-sc)*nr)[:, 0].to(self.dev)
            self._eps = self._eps+eps[g]
            dep = torch.zeros(n, 3, 3, dtype=f64, device=self.dev)
            dep[yn] = (Rp@(V_*((1-sc)*dv)[:, None, :]@V_.transpose(1, 2))@Rp.transpose(1, 2)).to(self.dev)
            self.ep = self.ep+dep[g]
            yl, yh = yn[g[lo]], yn[g[hi]]
            ti = torch.einsum('fab,fb->fa', Uei[g[lo]], torch.einsum('fba,fb->fa', Re[g[lo]], de-ci[:, None]*rei)); tj = torch.einsum('fab,fb->fa', Uei[g[hi]], torch.einsum('fba,fb->fa', Re[g[hi]], de-cj[:, None]*rej))
            self._qi = torch.where(yl[:, None], ti, qi)
            self._qj = torch.where(yh[:, None], tj, qj)
        else:
            self._qi, self._qj = qi.clone(), qj.clone()
        # F_g, L
        Ln = t['Ln']
        Ln[flag] = Ln[flag]+((t['Q']*Y['Gm']).sum((1, 2))/th)[flag]
        J = torch.linalg.det(Fgend); J0 = .05
        Le = torch.where(J > J0, torch.log(J.clamp(min=J0)), torch.log(torch.tensor(J0, dtype=f64, device=self.dev))+(J-J0)/J0)
        Ln[~flag] = Le[~flag]
        Fnew = Fgend.clone()
        if bool(yn.any()):
            ymem = yn[g]
            Eo = Fgend-Fe[g]@Y['Pc']
            cm = ymem & flag
            Fnew[cm] = Fprime[g[cm]]+br[g[cm]][:, None, None]*Eo[cm]
            em = ymem & ~flag
            Y['Pc'][em] = Uei[g[em]]@(Re[g[em]].transpose(1, 2)@(Fgend[em]-br[g[em]][:, None, None]*Eo[em]))
        self._Fe = Fnew
        self._Ft = Fprime[g].clone()
        self._be_end = be[g].clone()
        # past eps_cut: onto the cut, its group with it
        over = (self._eps > self.eps_cut) & ~flag
        if bool(over.any()):
            self._yld = self._yld | over
            T['flag'][over] = 1
        self.ul_stats['flips'] = int((J <= 1e-3).sum()); self.ul_stats['yielding'] = int(yn.sum()); self.ul_stats['yielded'] = int((self._eps > 0).sum()); self.ul_stats['cut_cells'] = int(self._yld.sum())
        if self.hold:
            nw = (self.tbd > 0) & ~self._stuck & (self.tbR > 0)
            if bool(nw.any()):
                self._stuck = self._stuck | nw
                self.tbkn[nw] = 2*self.Es*self.tbR[nw]*.05**.5
                self.tstuck[nw] = 1

    def stored(self, x=None):
        """the elastic energy of the state at the step's end x (or at x): the springs and the outer layer, the whole cell's minimum; volume"""
        T, Y, ms, lo, hi, n = self.tr, self.ty, self._m_s, self.tlo, self.thi, self.n
        t, g = self.tul, self._gid
        if x is None:
            x = (self.tx-(1-self.th)*self._xs)/self.th
        d = x[hi]-x[lo]
        wi, wj = T['wi'][:ms], T['wj'][:ms]; qi, qj = self._qi[:ms], self._qj[:ms]
        Fe, P, Mo = self._Fe, Y['Pc'], T['Mo']
        MoP = Mo@P.transpose(1, 2)
        A = torch.zeros(n, 3, 3, dtype=f64, device=self.dev).index_add(0, g, Fe@MoP)
        A.index_add_(0, g[lo], wi[:, None, None]*d[:, :, None]*qi[:, None, :]); A.index_add_(0, g[hi], wj[:, None, None]*d[:, :, None]*qj[:, None, :])
        lead = g == torch.arange(n, device=self.dev)
        R = self.I3.expand(n, 3, 3).clone()
        Us, _, Vh = torch.linalg.svd(A[lead].cpu())
        Sg = torch.diag_embed(torch.stack([torch.ones(len(Us), dtype=f64), torch.ones(len(Us), dtype=f64), torch.linalg.det(Us@Vh)], 1))
        R[lead] = (Us@Sg@Vh).to(self.dev)
        Rm = R[g]
        mi = torch.einsum('fba,fb->fa', Rm[lo], d)-qi; mj = torch.einsum('fba,fb->fa', Rm[hi], d)-qj
        Es = float((.5*wi*(mi*mi).sum(1)).sum()+(.5*wj*(mj*mj).sum(1)).sum())
        H = Rm.transpose(1, 2)@Fe-P
        Eo = float((.5*((H@Mo)*H).sum((1, 2))).sum())
        # (on the cut the shape only: the volumetric part of the fit carries nothing)
        M = torch.zeros(n, 3, 3, dtype=f64, device=self.dev).index_add(0, g, P@MoP)
        M.index_add_(0, g[lo], wi[:, None, None]*qi[:, :, None]*qi[:, None, :]); M.index_add_(0, g[hi], wj[:, None, None]*qj[:, :, None]*qj[:, None, :])
        on = torch.zeros(n, dtype=torch.bool, device=self.dev); on[g[self._yld]] = True
        on = on & lead
        if bool(on.any()):
            Fg_ = A[on]@self._inv(M[on])
            Hg = R[on].transpose(1, 2)@Fg_-self.I3
            Sg_ = torch.linalg.svd(Fg_.cpu())
            Ug_ = ((Sg_[2].transpose(1, 2)*Sg_[1][:, None, :])@Sg_[2]).to(self.dev)
            Mg_ = self._sj[on][:, None, None]*(Ug_@M[on]@Ug_)
            Ht_ = Ug_-self.I3
            vg = Ht_.diagonal(dim1=1, dim2=2).sum(1)/3
            eg_ = Ht_-vg[:, None, None]*self.I3
            Es += float((.5*((eg_@Mg_)*eg_).sum((1, 2))-.5*((Hg@M[on])*Hg).sum((1, 2))).sum())
        self._Es, self._Eo = Es, Eo
        L = t['Ln']
        Ev = float((self.V0*.5*torch.where(self._yld, torch.full_like(L, self.lam+2*self.mu/3), torch.full_like(L, self.lam))*L*L).sum())
        Ev += float((self.turv*(L[self._irw]-L[self._jrw])**2).sum())
        if self._cut_valid:
            i_w, j_w, m = self._w[0], self._w[1], self._w[5]
            Ev += float((t['wv'][:m]*(L[i_w]-L[j_w])**2).sum())
        return Es+Eo, Ev

    def ul_blocks(self, lr_thr=None, lr_cap=0):
        """the diagonal blocks at the last reading: the pressure and the outer layer through the gradients' direct rows, the halves' w beta_r on each end and on the far end the tensor's part by the half's leverage w q M^-1 q"""
        assert lr_thr is None, 'no low-rank term here'
        n, t, dev, T, Y = self.n, self.tul, self.dev, self.tr, self.ty
        flag, g = self._yld, self._gid
        be, br = Y['be'], Y['br']
        K = torch.zeros(n, 3, 3, dtype=f64, device=dev)
        mod = torch.where(flag, torch.full_like(be, self.lam+2*self.mu/3), self.lam+(1-be[g])*2*self.mu/3)
        sc, kc, r = self._rrows
        el = ~flag[sc]
        if bool(el.any()):
            s_, k_, r_ = sc[el], kc[el], r[el]
            cr = torch.einsum('fab,fb->fa', Y['Cz'][s_], r_)
            rP = torch.einsum('fba,fb->fa', Y['Pc'][s_], r_)
            rM = (torch.einsum('fab,fb->fa', T['Mo'][s_], r_)*r_).sum(1)
            K.index_add_(0, k_, (self.V0[s_]*mod[s_])[:, None, None]*(cr[:, :, None]*cr[:, None, :])+(br[g[s_]]*rM)[:, None, None]*self.I3)
        if self._cut_valid and bool(flag.any()):
            i, j, Lw, key, _, m = self._w
            sc2, kc2, r2_ = self._rows_from(i, j, t['wa'][:m], t['wt'][:m], t['wn'][:m], t['Vi'], self._ml)
            fw = flag[sc2]
            s_, k_, r_ = sc2[fw], kc2[fw], r2_[fw]
            cr = torch.einsum('fab,fb->fa', t['Q'][s_], r_)
            rF = torch.einsum('fba,fb->fa', self.tsp['F0'][s_], r_)
            rM = (torch.einsum('fab,fb->fa', T['Mo'][s_], rF)*rF).sum(1)
            K.index_add_(0, k_, (self.V0[s_]*mod[s_])[:, None, None]*(cr[:, :, None]*cr[:, None, :])+(br[g[s_]]*rM)[:, None, None]*self.I3)
        ms = self._m_s
        lo, hi = self.tlo, self.thi
        wi, wj = T['wi'][:ms], T['wj'][:ms]
        qi, qj = T['qi'][:ms], T['qj'][:ms]
        Mi = Y['Mti']
        li = (torch.einsum('fab,fb->fa', Mi[g[lo]], qi)*qi).sum(1); lj = (torch.einsum('fab,fb->fa', Mi[g[hi]], qj)*qj).sum(1)
        bi, ci, bj, cj = be[g[lo]], br[g[lo]], be[g[hi]], br[g[hi]]
        wc = wi*ci+wj*cj
        ka = wc+wj*wj*(bj-cj)*lj; kb = wc+wi*wi*(bi-ci)*li
        K.index_add_(0, lo, ka[:, None, None]*self.I3); K.index_add_(0, hi, kb[:, None, None]*self.I3)
        if self.hold and bool(self._stuck.any()):
            off = self._stuck & (self.tbR > 0) & (self.tbd < .05*self.tbR)
            ng = torch.as_tensor([float(self.ball['ng'][k]) for k in range(3)], dtype=f64, device=dev)
            hz = 2*self.Es*torch.sqrt(self.tbR[off]*self.tbd[off].clamp(min=0.))
            K[off] += ((self.tbkn[off]-hz)/self.th)[:, None, None]*(ng[:, None]*ng[None, :])
        return K


class YieldStep(MidpointStep):
    """the local-global iteration: at z read beta, beta_r, R, C and the true |g|; then z of the frozen problem to eta_out of it, by the trust-region CG on Pi, its energy valid there (a quadratic, and Hertz); until |g| is at the tolerance. stats: outer iterations, the trace (products, edge, negative, stop, |g|, ratio, taken, outer)"""
    eta_out, inner_max, outer_max, cg_cap_max, hold_after = .1, 30, 200, 3000, 3
    aa_m, aa_back = 6, 1.5

    def solve(self, tol_abs, cg_tol_abs):
        W, A, bl, n = self.W, self.a, self.W.ball, self.W.n
        wp.launch(k_mid_begin, n, inputs=[A['x'], A['v'], A['a'], self.dt, self.th], outputs=[A['xt'], A['z']])
        if self.smooth_pred and self._vp is not None:
            h_ = self.th*self.dt
            self.tz.copy_(self.tx+.5*h_*(self.tv+self._vp)+.5*(.5*h_*self.dt+h_*h_)*(self.ta+self._ap))
        wp.launch(k_mid_ball, bl['k'], inputs=[A['x'], bl['body'], bl['R'], bl['ng'], self.c0, self.g8, self.cb], outputs=[bl['xn'], bl['kt'], bl['bound']])
        if getattr(W, 'hold', False):
            W.hold_start(self.tx, self.muf)
        evals, its, outer = 1, 0, 0
        st = dict(rejected=0, edge=0, negative=0, trace=[], outer=0)
        aa_on = self.aa_m > 0
        hist_G, hist_f, z_plain, g_last = [], [], None, math.inf
        self._eval(0.)
        delta = self.delta_start if self.carry_delta and self.delta_start else -1.
        cap = self.cg_cap
        while True:
            W.reread(hold=outer >= self.hold_after)
            g2 = self._eval(0.)[PS_GQ2]; evals += 1
            self._accept()
            g_ = math.sqrt(g2)
            if aa_on and g_ > self.aa_back*g_last and z_plain is not None:
                # (the extrapolated point read worse than the plain one was: back to it, the history dropped)
                self.tz.copy_(z_plain)
                z_plain = None; hist_G, hist_f = [], []
                st['aa_back'] = st.get('aa_back', 0)+1
                self._eval(0.)
                W.reread(hold=outer >= self.hold_after)
                g2 = self._eval(0.)[PS_GQ2]; evals += 2
                self._accept()
                g_ = math.sqrt(g2)
            g_last = g_
            g_k = wp.to_torch(self.a['gz'], requires_grad=False).clone() if aa_on else None
            if g_ <= tol_abs or outer >= self.outer_max:
                break
            outer += 1
            target = max(.5*tol_abs, self.eta_out*g_)
            delta0 = None
            for _ in range(self.inner_max):
                its += 1
                if outer == 1 and its == 1 and self.pre_ul and self._pre_age <= 0:
                    self._pre_age = self.pre_every
                    self.tK.copy_(W.ul_blocks())
                wp.launch(k_mid_pre, n, inputs=[W.B['R'], A['K'], self.c1, A['cm']], outputs=[A['P']])
                wp.launch(k_mid_pre_ball, bl['k'], inputs=[bl['body'], bl['R'], W.B['bdelta'], bl['ng'], self.e4, bl['kt'], bl['bound'], self.muf], outputs=[A['P']])
                wp.launch(k_mid_inv, n, inputs=[A['P']], outputs=[A['Pinv']])
                stop = max(cg_tol_abs(g2), self.eta*math.sqrt(g2)); gi_ = math.sqrt(g2)
                r = self.CG.solve(A['gz'], delta, stop, cap)
                if r['K'] >= cap and not r['EDGE'] and not r['NEG']:
                    cap = min(2*cap, self.cg_cap_max)
                evals += r['K']; st['edge'] += r['EDGE']; st['negative'] += r['NEG']
                delta = r['DELTA']
                if delta0 is None:
                    delta0 = delta
                pred = -(r['GP']+.5*r['PHP'])
                s = self._eval(1.); evals += 1
                ok = math.isfinite(s[PS_PQ]) and math.isfinite(s[PS_GQ2])
                noise = 1e-14*abs(s[PS_PZ])
                ared = s[PS_PZ]-s[PS_PQ] if ok else -math.inf
                rho = ared/pred if pred > noise else (1. if ared >= -noise else -math.inf)
                take = ok and rho > 1e-4
                pM = delta if r['EDGE'] else math.sqrt(max(r['PMP'], 0.))
                if not take or rho < .25 or pred <= noise:
                    delta = .25*pM
                elif rho > .75 and r['EDGE']:
                    delta = 2*delta
                st['trace'].append((r['K'], r['EDGE'], r['NEG'], stop, gi_, rho, take, outer))
                if take:
                    self._accept()
                    g2 = s[PS_GQ2]
                else:
                    st['rejected'] += 1
                    g2 = self._eval(0.)[PS_GQ2]; evals += 1
                    self._accept()
                    if not delta > 1e-12*delta0:
                        break
                if math.sqrt(g2) <= target:
                    break
            if aa_on:
                # z_k -> G_k, the inner solve: the combination of the last iterates with the least true |g|, by its differences (in the mass's norm of G - z a stiff direction's residual does not show: the combination stood still at |g| 6 tol)
                G_k = self.tz.clone()
                z_plain = G_k
                hist_G.append(G_k); hist_f.append(g_k)
                if len(hist_f) > self.aa_m+1:
                    hist_G.pop(0); hist_f.pop(0)
                if len(hist_f) >= 2:
                    dF = torch.stack([(hist_f[k+1]-hist_f[k]).flatten() for k in range(len(hist_f)-1)], 1)
                    dG = torch.stack([(hist_G[k+1]-hist_G[k]).flatten() for k in range(len(hist_G)-1)], 1)
                    A_ = dF.T@dF
                    b_ = dF.T@g_k.flatten()
                    A_ = A_+1e-10*A_.diagonal().mean()*torch.eye(A_.shape[0], dtype=A_.dtype, device=A_.device)
                    gam = torch.linalg.solve(A_.cpu(), b_.cpu()).to(A_.device)
                    self.tz.copy_(G_k-(dG@gam).reshape(-1, 3))
                    self._eval(0.); evals += 1
                    self._accept()
        st['outer'] = outer
        self.stats = st
        self.delta_start = delta if delta > 0. and outer <= 20 else None
        return evals, its, g2
