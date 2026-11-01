"""The ring spring with a yield limit, no switch: springs carry the force at every state (2026-11-01). Moduli per unit density, masses = volumes.
The corner to give up. A uniform flow balances only if every cell reads one stress; its virial is then that stress times M_i, and M_i is round only in the sum. So a uniform flow, an incremental potential and a convex cell cannot all hold (ring_flow.py's header). The potential is only what a trust-region solver asks for. Given up here: the model is a non-associated plasticity cell by cell, yield in the Frobenius norm of the strain, elastic metric M_i.
  state     q per half spring (the rest vector until the cell yields), P per cell for the outer layer's virtual springs q_k = P D_k, x. F_g = I + G(x) on the rest cut. A cell whose plastic strain passes eps_cut goes on the current cut: F_g = (I + G) F_g,n, P = I, L a running sum (rest rows read poorly once the plastic stretch's anisotropy of M passes what the foam's own has: e^(2 eps) ~ 1 + delta, delta about .3)
  reading   A = sum w d (x) q + F_g Mo P^T, M = sum w q (x) q + P Mo P^T, F = A M^-1, the whole cell's fit; R = polar(A), the whole cell's minimum; in its frame H = R^T F - I; the step's start H_n = R_n^T F_n - I, kept in the material frame so a spin reads nothing (read in R(z), F_n - R gave a first-order false strain: the flight's outer iteration diverged at step 2); the step's end H_e = (H(z) - (1 - th) H_n) / th. Yield on the tensor only, as the MPM: beta = min(1, e_y / |dev sym H_e|). Not on the remainder: a cell's wiggles are not its strain (the user, 10-31)
  force     half spring f = w [beta (F - R) q + (1 - beta)(1 - th) R H_n q + beta_r (d - F q) + (1 - beta_r)(1 - th) R r_n q], r_n = R_n^T (d_n - F_n q); the outer layer through F_g: [(beta (F - R) + (1 - beta)(1 - th) R H_n) P + beta_r (F_g - F P) + (1 - beta_r)(1 - th) R R_n^T (F_g,n - F_n P)] Mo. Pressure V0 [lam L + (1 - beta) 2 mu / 3 (L - (1 - th) L_n)] on L: the springs' volumetric part goes down with beta and the pressure takes it, the volumetric stress 3 kap v whatever beta, so a flowing cell's force has no v R: its negative stiffness in compression, |v| 2 mu V, does not stay when the shape's stiffness has gone with beta
            beta = beta_r = 1: w (d - R q), exactly ring_ul.py. No switch
  remainder a Maxwell element, beta_r = 1 / (1 + kappa (1 / beta - 1)), kappa = 5 at w (ring_flow.py's header)
  frozen    beta, beta_r, R, L's linearisation C frozen: the force is the gradient of a convex quadratic, by the fit's normal equations (sum w r (x) q + r_o Mo P^T = 0, so d/dx through F drops out of the remainder's energy, and M M^-1 cancels in the tensor's):
            E = 1/2 beta tr(X M X^T) + (1 - beta)(1 - th) tr(R H_n M X^T) + 1/2 beta_r W_r + (1 - beta_r)(1 - th) W_r,n + V0 [1/2 lam L^2 + (1 - beta) 2 mu / 3 (1/2 L^2 - (1 - th) L_n L)], X = F - R
  solver    so a local-global iteration (YieldStep): z of the frozen problem, by the trust-region CG with its energy ratio valid there; then beta, R, C read at z, until the true |g| is small; after hold_after outer iterations a cell under beta .2 keeps its R and C: its force goes with beta (F - R), so R's error costs little, and on a cell far gone R can flip between two rotations. For one norm term the outer map contracts by 2 mu beta / (k + 2 mu beta), k the inertia's, and it is a majorisation (IRLS) when M is round
  torque    per cell skew(A P^T) = 0 by R^T A symmetric and the normal equations, at the fixed point where R = polar(A(z)), but for the start's term, second order: (1 - beta)(1 - th) skew(H H_n M)
  update    at the step's end, the polar F_e = R_e U of a cell that yields: U_e = a exp(c dev log U), c holding the limit, a = exp(L / 3): only its shape is a state, its volume L's (on nothing else the fit's volume drifts, once beta has given its volumetric share to the pressure: a cell read .29 by the fit with L about 0, then its R flipped 125 degrees at every outer iteration). a's log within +-.5 (L is a linear reading, and a sets only q's scale); q -> U_e^-1 R_e^T (d_e - beta_r r_e), r_e = d_e - F_e q: so F' = R_e U_e and r' = beta_r r, and q comes from d itself, however flat F_e read (the reading must end on the limit: the step's start term (1 - beta)(1 - th) R H_n has no limit of its own, and an increment bounded per step left H_n past it: plastic work -.05, then 1e8). The outer layer's virtual springs the same, P -> U_e^-1 R_e^T (F_g - beta_r (F_g - F_e P)); on the cut, P = I, F_g -> R_e U_e + beta_r (F_g - F_e)
  merging   a cell under agg of the median volume is one cell with its neighbour across its largest wall: A, M summed, F, R, beta shared, its gradient the group's volume mean, V_g G_g = sum V_i G_i, which keeps the cancellation across walls
Self contact not here. set_ball, start(), prepare(), update_ul; through YieldStep. ring_yield_check.py its checks, ring_yield_part.py, ring_yield_drop.py."""
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
def k_ry_L(Fc: wp.array(dtype=wp.mat33d), Gm: wp.array(dtype=wp.mat33d), C: wp.array(dtype=wp.mat33d), Ln: wp.array(dtype=wp.float64), V0: wp.array(dtype=wp.float64), lam: wp.float64, k2: wp.float64, ca: wp.float64, flag: wp.array(dtype=int), gid: wp.array(dtype=int),
           be: wp.array(dtype=wp.float64), Lz: wp.array(dtype=wp.float64), Cz: wp.array(dtype=wp.mat33d), Fz: wp.array(dtype=wp.mat33d), L: wp.array(dtype=wp.float64), Lb: wp.array(dtype=wp.float64), terms: wp.array2d(dtype=wp.float64)):
    """L on its frozen linearisation; the springs' volumetric share that beta has let go, on the pressure"""
    b = wp.tid()
    half = wp.float64(0.5)
    lb = Lz[b]+wp.ddot(Cz[b], Fc[b]-Fz[b])
    if flag[b] != 0:
        lb = Ln[b]+wp.ddot(C[b], Gm[b])
    L[b] = lb
    s = k2*(wp.float64(1.0)-be[gid[b]])
    wp.atomic_add(terms, 1, b % NSLOT, V0[b]*(half*lam*lb*lb+s*(half*lb*lb-ca*Ln[b]*lb)))
    wp.atomic_add(Lb, b, V0[b]*(lam*lb+s*(lb-ca*Ln[b])))


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
def k_ry_cell(Ff: wp.array(dtype=wp.mat33d), Hn: wp.array(dtype=wp.mat33d), Rz: wp.array(dtype=wp.mat33d), Mt: wp.array(dtype=wp.mat33d), Fc: wp.array(dtype=wp.mat33d), Ron: wp.array(dtype=wp.mat33d), Pc: wp.array(dtype=wp.mat33d), Mo: wp.array(dtype=wp.mat33d),
              be: wp.array(dtype=wp.float64), br: wp.array(dtype=wp.float64), ca: wp.float64, gid: wp.array(dtype=int), Fb: wp.array(dtype=wp.mat33d), terms: wp.array2d(dtype=wp.float64)):
    """the tensor's energy once per group, term 0; the outer layer's remainder, term 5; dE/dF_g"""
    b = wp.tid()
    half = wp.float64(0.5)
    one = wp.float64(1.0)
    g = gid[b]
    X = Ff[g]-Rz[g]
    Xn = Rz[g]@Hn[g]
    bb = be[g]
    cc = br[g]
    if g == b:
        wp.atomic_add(terms, 0, b % NSLOT, half*bb*wp.ddot(X@Mt[b], X)+(one-bb)*ca*wp.ddot(Xn@Mt[b], X))
    R0 = Fc[b]-Ff[g]@Pc[b]
    Rn = Rz[g]@Ron[b]
    wp.atomic_add(terms, 5, b % NSLOT, half*cc*wp.ddot(R0@Mo[b], R0)+(one-cc)*ca*wp.ddot(Rn@Mo[b], R0))
    Fb[b] = (((bb*X+((one-bb)*ca)*Xn)@Pc[b])+cc*R0+((one-cc)*ca)*Rn)@Mo[b]


@wp.kernel(enable_backward=False)
def k_ry_half(x: wp.array(dtype=wp.vec3d), ri: wp.array(dtype=int), rj: wp.array(dtype=int), wi: wp.array(dtype=wp.float64), wj: wp.array(dtype=wp.float64), qi: wp.array(dtype=wp.vec3d), qj: wp.array(dtype=wp.vec3d), dni: wp.array(dtype=wp.vec3d), dnj: wp.array(dtype=wp.vec3d), gid: wp.array(dtype=int),
              Ff: wp.array(dtype=wp.mat33d), Hn: wp.array(dtype=wp.mat33d), Rz: wp.array(dtype=wp.mat33d), be: wp.array(dtype=wp.float64), br: wp.array(dtype=wp.float64), ca: wp.float64, terms: wp.array2d(dtype=wp.float64), x_adj: wp.array(dtype=wp.vec3d)):
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
    f = w*(bb*((Ff[gi]-Rz[gi])@q)+((one-bb)*ca)*(Rz[gi]@(Hn[gi]@q))+cc*r+((one-cc)*ca)*rn)
    wp.atomic_add(terms, 4, e % NSLOT, half*cc*w*wp.dot(r, r)+(one-cc)*ca*w*wp.dot(rn, r))
    gj = gid[j]
    q = qj[e]
    bb = be[gj]
    cc = br[gj]
    w = wj[e]
    r = d-Ff[gj]@q
    rn = Rz[gj]@dnj[e]
    f = f+w*(bb*((Ff[gj]-Rz[gj])@q)+((one-bb)*ca)*(Rz[gj]@(Hn[gj]@q))+cc*r+((one-cc)*ca)*rn)
    wp.atomic_add(terms, 4, e % NSLOT, half*cc*w*wp.dot(r, r)+(one-cc)*ca*w*wp.dot(rn, r))
    wp.atomic_add(x_adj, j, f)
    wp.atomic_add(x_adj, i, -f)


@wp.kernel(enable_backward=False)
def k_ry_start(Fn: wp.array(dtype=wp.mat33d), Mt: wp.array(dtype=wp.mat33d), gid: wp.array(dtype=int), Rn: wp.array(dtype=wp.mat33d), Hn: wp.array(dtype=wp.mat33d)):
    """the step's start in the material frame, per group"""
    b = wp.tid()
    if gid[b] != b:
        return
    R = rot_proper(Fn[b]@Mt[b])
    Rn[b] = R
    Hn[b] = wp.transpose(R)@Fn[b]-wp.identity(n=3, dtype=wp.float64)


@wp.kernel(enable_backward=False)
def k_ry_read(Ff: wp.array(dtype=wp.mat33d), Hn: wp.array(dtype=wp.mat33d), Mt: wp.array(dtype=wp.mat33d), ey: wp.float64, th_: wp.float64, kappa: wp.float64, hold: int, gid: wp.array(dtype=int), Rz: wp.array(dtype=wp.mat33d), be: wp.array(dtype=wp.float64), br: wp.array(dtype=wp.float64), ei: wp.array(dtype=wp.float64)):
    """the frozen part, per group, at the last evaluation"""
    b = wp.tid()
    if gid[b] != b:
        return
    one = wp.float64(1.0)
    I = wp.identity(n=3, dtype=wp.float64)
    R = Rz[b]
    if hold == 0 or be[b] >= wp.float64(0.2):
        R = rot_proper(Ff[b]@Mt[b])
        Rz[b] = R
    H = (wp.transpose(R)@Ff[b]-I-(one-th_)*Hn[b])/th_
    s = wp.float64(0.5)*(H+wp.transpose(H))
    e = s-(wp.trace(s)/wp.float64(3.0))*I
    nr = wp.sqrt(wp.ddot(e, e))
    ei[b] = nr
    bb = one
    if nr > ey:
        bb = ey/nr
    be[b] = bb
    br[b] = one/(one+kappa*(one/bb-one))


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

    def __init__(self, X, r, V0, pairs, A, c, E, nu, sigma_y, theta=.5, kappa=5., eps_cut=.15, agg_merge=.1, hold=False, ledger=False, **kw):
        super().__init__(X, r, V0, pairs, A, c, E, nu, sigma_y, theta=theta, kappa=kappa, hold=hold, ledger=ledger, **kw)
        self.eps_cut, self.agg_merge = float(eps_cut), float(agg_merge)
        self.Wp_cell = self.Wp_half = 0.

    def start(self):
        n, dev = self.n, self.dev
        z = lambda k_, dt_: wp.zeros(k_, dtype=dt_, device=dev)
        Ks = self.ul['Kw']
        self.ry = dict(Ff=z(n, wp.mat33d), Fn=z(n, wp.mat33d), Rn=z(n, wp.mat33d), Hn=z(n, wp.mat33d), Ron=z(n, wp.mat33d), Rz=z(n, wp.mat33d), Mt=z(n, wp.mat33d), Mti=z(n, wp.mat33d), MoP=z(n, wp.mat33d), Pc=z(n, wp.mat33d), be=z(n, wp.float64), br=z(n, wp.float64), ei=z(n, wp.float64), Lz=z(n, wp.float64), Cz=z(n, wp.mat33d), Fz=z(n, wp.mat33d),
                       gid=z(n, wp.int32), Vgi=z(n, wp.float64), Gs=z(n, wp.mat33d), Grs=z(n, wp.mat33d), Gm=z(n, wp.mat33d), Grm=z(n, wp.mat33d), Gbm=z(n, wp.mat33d), Gbrm=z(n, wp.mat33d), Ss=z(n, wp.mat33d), Srs=z(n, wp.mat33d))
        tt = lambda a_: wp.to_torch(a_, requires_grad=False)
        self.ty = {k_: tt(a_) for k_, a_ in self.ry.items()}
        Y = self.ty
        Y['Pc'].copy_(torch.eye(3, dtype=f64, device=dev).expand(n, 3, 3)); Y['be'].fill_(1.); Y['br'].fill_(1.); Y['Rz'].copy_(Y['Pc']); Y['Cz'].copy_(Y['Pc']); Y['Fz'].copy_(Y['Pc'])
        self._eps = torch.zeros(n, dtype=f64, device=dev)
        super().start()
        T = self.tr
        self._Mo0 = T['Mo'].clone()
        self._gid_rest()
        self._state_start(self.X)
        self._xs = self.X.clone()

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
        """the half springs by their pair at x: the kept q, a new one reading no remainder in its cell's last frame; equal halves, mu A / L each"""
        T = self.tr
        lo, hi, key, ms = self.tlo, self.thi, self.tkey, self._m_s
        pos = torch.searchsorted(self._qkey, key).clamp(max=len(self._qkey)-1); kept = self._qkey[pos] == key
        d = x[hi]-x[lo]
        qi, qj = d.clone(), d.clone()
        qi[kept] = self._qi[pos[kept]]; qj[kept] = self._qj[pos[kept]]
        if not bool(kept.all()) and hasattr(self, '_Re'):
            nw = ~kept
            qi[nw] = torch.einsum('fba,fb->fa', self._Re[self._gid[lo[nw]]], d[nw])/(1+self._ve[self._gid[lo[nw]]])[:, None]; qj[nw] = torch.einsum('fba,fb->fa', self._Re[self._gid[hi[nw]]], d[nw])/(1+self._ve[self._gid[hi[nw]]])[:, None]
        self._qkey, self._qi, self._qj = key.clone(), qi, qj
        w = self.tk0/(2*self.kshare)
        for k_ in ('i', 'j', 'wi', 'wj', 'qi', 'qj', 'dni', 'dnj'):
            T[k_].zero_()
        T['i'][:ms] = lo.to(torch.int32); T['j'][:ms] = hi.to(torch.int32); T['wi'][:ms] = w; T['wj'][:ms] = w; T['qi'][:ms] = qi; T['qj'][:ms] = qj

    def _cut(self, x):
        """the current cut at x: merged_ul.py's list and gradient, the halves; a cut cell's outer layer on it, P = I; its group on the current walls"""
        MergedUL.update_ul(self, x)
        self._cut_valid = True
        self._set_halves(x)
        t, T, Y, yld = self.tul, self.tr, self.ty, self._yld
        Mc = t['Mc']
        T['Mo'].copy_(torch.where(yld[:, None, None], self.mu*(Mc+Mc.transpose(1, 2)), self._Mo0))
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

    def _state_start(self, x):
        """the step's start: M, its inverse, the fit at x, each half's remainder there"""
        T, Y, ms, lo, hi, n = self.tr, self.ty, self._m_s, self.tlo, self.thi, self.n
        g = self._gid
        P, Mo = Y['Pc'], T['Mo']
        Y['MoP'].copy_(Mo@P.transpose(1, 2))
        w = T['wi'][:ms]
        qi, qj = T['qi'][:ms], T['qj'][:ms]
        M = torch.zeros(n, 3, 3, dtype=f64, device=self.dev).index_add(0, g, P@Y['MoP'])
        M.index_add_(0, g[lo], w[:, None, None]*qi[:, :, None]*qi[:, None, :]); M.index_add_(0, g[hi], w[:, None, None]*qj[:, :, None]*qj[:, None, :])
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
        A.index_add_(0, g[lo], w[:, None, None]*d[:, :, None]*qi[:, None, :]); A.index_add_(0, g[hi], w[:, None, None]*d[:, :, None]*qj[:, None, :])
        Fn = A@Mi
        Y['Fn'].copy_(Fn)
        wp.launch(k_ry_start, n, inputs=[self.ry['Fn'], self.ry['Mt'], self.ry['gid']], outputs=[self.ry['Rn'], self.ry['Hn']])
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
        self._state_start(x)
        if cut and self.ledger:
            e1 = self._parts(x)
            self.Ecut += sum(e1)-sum(e0)
            self.Ecut_parts = [a_+b_-c_ for a_, b_, c_ in zip(getattr(self, 'Ecut_parts', [0., 0., 0.]), e1, e0)]
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
        a, b = self.stored(x)
        return [a, b, self.ground_store(x)]

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
        wp.launch(k_ry_L, n, inputs=[B['Fc'], Y['Gm'], U['Q'], U['Ln'], V0, wp.float64(self.lam), wp.float64(2*self.mu/3), ca, Rg['flag'], gid, Y['be'], Y['Lz'], Y['Cz'], Y['Fz']], outputs=[B['L'], B['Lb'], B['terms']])
        wp.launch(k_var, Kw, inputs=[U['wi'], U['wj'], U['wv'], B['L']], outputs=[B['Lb'], B['terms']])
        wp.launch(k_var, Ur['Kw'], inputs=[Ur['wi'], Ur['wj'], Ur['wv'], B['L']], outputs=[B['Lb'], B['terms']])
        hs = [Rg['i'], Rg['j'], Rg['wi'], Rg['wj'], Rg['qi'], Rg['qj']]
        wp.launch(k_ry_A, Rg['K'], inputs=[x]+hs+[gid], outputs=[Rg['Am']])
        wp.launch(k_ry_Aout, n, inputs=[B['Fc'], Y['MoP'], gid], outputs=[Rg['Am']])
        wp.launch(k_ry_Ff, n, inputs=[Rg['Am'], Y['Mti']], outputs=[Y['Ff']])
        wp.launch(k_ry_cell, n, inputs=[Y['Ff'], Y['Hn'], Y['Rz'], Y['Mt'], B['Fc'], Y['Ron'], Y['Pc'], Rg['Mo'], Y['be'], Y['br'], ca, gid], outputs=[B['Fb'], B['terms']])
        wp.launch(k_rf_Fadj, n, inputs=[B['Fb'], S['F0'], Rg['flag']], outputs=[Y['Gbm'], Y['Gbrm']])
        wp.launch(k_ry_half, Rg['K'], inputs=[x]+hs+[Rg['dni'], Rg['dnj'], gid, Y['Ff'], Y['Hn'], Y['Rz'], Y['be'], Y['br'], ca], outputs=[B['terms'], g])
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
        """beta, beta_r, R, C at the last evaluation's x, frozen for the next inner solve; hold: a cell under beta .2 keeps its R and C"""
        Y, n = self.ry, self.n
        Rp_ = self.ty['Rz'].clone() if getattr(self, 'watch', None) else None
        wp.launch(k_ry_read, n, inputs=[Y['Ff'], Y['Hn'], Y['Mt'], wp.float64(self.ey), wp.float64(self.th), wp.float64(self.kappa), int(hold), Y['gid']], outputs=[Y['Rz'], Y['be'], Y['br'], Y['ei']])
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
        if Rp_ is not None and getattr(self, 'watch_on', False):
            for c_ in self.watch:
                g_ = int(self._gid[c_])
                tY = self.ty
                dR = Rp_[g_].T@tY['Rz'][g_]
                ang = float(torch.arccos(((torch.trace(dR)-1)/2).clamp(-1, 1)))*57.3
                Sf = torch.linalg.svdvals(tY['Ff'][g_]); Sg = torch.linalg.svdvals(self.tFc[c_])
                print('    cell %d (group %d, cut %d): reading/e_y %.1f beta %.4f beta_r %.4f; R turned %.1f deg; fit sv %.3f %.3f; F_g sv %.3f %.3f; L %.4f |C| %.2f |Cz| %.2f' % (c_, g_, int(self._yld[c_]), float(tY['ei'][g_])/self.ey, float(tY['be'][g_]), float(tY['br'][g_]), ang, float(Sf[2]), float(Sf[0]), float(Sg[2]), float(Sg[0]), float(self.tL[c_]), float(self.tul['Q'][c_].norm()), float(tY['Cz'][c_].norm())), flush=True)

    def end_ul(self, th=.5):
        """after the step (the buffers at the accepted z): the plastic work, the returns on q and P, F_g, L; a cell past eps_cut goes on the cut"""
        assert th == self.th == .5, 'the plastic work is written for the midpoint'
        t, s, T, Y, n, I3, ms = self.tul, self.tsp, self.tr, self.ty, self.n, self.I3, self._m_s
        lo, hi, g = self.tlo, self.thi, self._gid
        ca = 1-th
        be, br, R, M, Ff, Fn = Y['be'], Y['br'], Y['Rz'], Y['Mt'], Y['Ff'], Y['Fn']
        Fe = (Ff-ca*Fn)/th
        lead = g == torch.arange(n, device=self.dev)
        Hn = Y['Hn']; He = (R.transpose(1, 2)@Ff-I3-ca*Hn)/th
        Hnd = Hn-(Hn.diagonal(dim1=1, dim2=2).sum(1)/3)[:, None, None]*I3; Hed = He-(He.diagonal(dim1=1, dim2=2).sum(1)/3)[:, None, None]*I3
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
        self.Wp_cell += wc_; self.Wp_half += wh_; self.Wp += wc_+wh_
        F0 = s['F0']
        Fgend = (self.tFc-ca*F0)/th
        flag = self._yld
        yn = lead & ((be < 1) | (br < 1))
        Re = R.clone()
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
            lue = la+sc*dv
            Uei = torch.zeros(n, 3, 3, dtype=f64, device=self.dev)
            Uei[yn] = (V_*torch.exp(-lue)[:, None, :]@V_.transpose(1, 2)).to(self.dev)
            Fprime = Fe.clone()
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
        self._Re, self._ve = Re, (He.diagonal(dim1=1, dim2=2).sum(1)/3)
        # past eps_cut: onto the cut, its group with it
        over = (self._eps > self.eps_cut) & ~flag
        if bool(over.any()):
            self._Fe[over] = self._Fe[over]@torch.linalg.inv(Y['Pc'][over])
            Y['Pc'][over] = I3
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
        w = T['wi'][:ms]; qi, qj = self._qi[:ms], self._qj[:ms]
        Fe, P, Mo = self._Fe, Y['Pc'], T['Mo']
        MoP = Mo@P.transpose(1, 2)
        A = torch.zeros(n, 3, 3, dtype=f64, device=self.dev).index_add(0, g, Fe@MoP)
        A.index_add_(0, g[lo], w[:, None, None]*d[:, :, None]*qi[:, None, :]); A.index_add_(0, g[hi], w[:, None, None]*d[:, :, None]*qj[:, None, :])
        lead = g == torch.arange(n, device=self.dev)
        R = self.I3.expand(n, 3, 3).clone()
        Us, _, Vh = torch.linalg.svd(A[lead].cpu())
        Sg = torch.diag_embed(torch.stack([torch.ones(len(Us), dtype=f64), torch.ones(len(Us), dtype=f64), torch.linalg.det(Us@Vh)], 1))
        R[lead] = (Us@Sg@Vh).to(self.dev)
        Rm = R[g]
        mi = torch.einsum('fba,fb->fa', Rm[lo], d)-qi; mj = torch.einsum('fba,fb->fa', Rm[hi], d)-qj
        Es = float((.5*w*(mi*mi).sum(1)).sum()+(.5*w*(mj*mj).sum(1)).sum())
        H = Rm.transpose(1, 2)@Fe-P
        Eo = float((.5*((H@Mo)*H).sum((1, 2))).sum())
        L = t['Ln']
        Ev = float((self.V0*.5*self.lam*L*L).sum())
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
        mod = self.lam+(1-be[g])*2*self.mu/3
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
        w = T['wi'][:ms]
        qi, qj = T['qi'][:ms], T['qj'][:ms]
        Mi = Y['Mti']
        li = w*(torch.einsum('fab,fb->fa', Mi[g[lo]], qi)*qi).sum(1); lj = w*(torch.einsum('fab,fb->fa', Mi[g[hi]], qj)*qj).sum(1)
        bi, ci, bj, cj = be[g[lo]], br[g[lo]], be[g[hi]], br[g[hi]]
        ka = w*(ci+cj+(bj-cj)*lj); kb = w*(ci+cj+(bi-ci)*li)
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
        self._eval(0.)
        delta = self.delta_start if self.carry_delta and self.delta_start else -1.
        cap = self.cg_cap
        while True:
            W.reread(hold=outer >= self.hold_after)
            g2 = self._eval(0.)[PS_GQ2]; evals += 1
            self._accept()
            g_ = math.sqrt(g2)
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
        st['outer'] = outer
        self.stats = st
        self.delta_start = delta if delta > 0. and outer <= 20 else None
        return evals, its, g2
