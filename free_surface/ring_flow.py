"""The elastic ring and the flowing cell joined (2026-11-01; the user: springs carry the force, and the two are not one or the other). Moduli per unit density, masses = volumes.
Why not a limit on the springs (ring_plastic.py, four forms, none right). A uniform flow balances only if every cell reads one stress P(D); its virial is then P(D) M_i, M_i = sum w q (x) q + Mo. For a potential M_i dP/dD must be symmetric. Elastic dP/dD is constant: no conflict. In flow
dP/dD = (e_y / |D|)(I - D^ (x) D^), symmetric for every direction only if M_i is round, and it is not, cell by cell (only the sum is): so a uniform flow, a potential and a convex cell cannot all hold on the springs' metric. A yield limit needs the metric V_i I, which the divergence-theorem gradient has (merged_ul.py).
  elastic cell  flag 0: exactly RingUL (ring_ul.py): half springs on the rest q, w = mu A / L each; the rest cut's gradient F_o = I + G(x) for the outer layer and for L = ln det F_o, V0 lam / 2 L^2; R the whole cell's minimum, read at every evaluation. Nothing flagged: no cut at all, RingUL's force, except its viscosity (not here yet)
  flowing cell  flag 1: V0 W(e(F_o)) + V0 kap / 2 L^2 (merged_ul.py's cell on the current cut, F_o = (I + G) F_o,n, R and C frozen at Newton's start, L = L_n + tr(C^T G)), each half spring r = d - p - G p, one-sided: the part of the relative motion its own cell's gradient does not explain; p = F_o q at the flag, d - kept remainder after that. F_o's volume is L's at each step's end: only its shape is a state (a cell flattened in a step kept det F_o near 0 for good otherwise, and its shape read shrank with it)
  the two forms S_i = 1/2 tr(H M_i H^T) + 1/2 sum w |r|^2 and V_i mu |H|^2 + 1/2 sum w |r|^2 hold one elastic energy on two quadratures: equal on a uniform strain over the whole body, cell by cell by M_i's anisotropy. At a limit of 1 % the flag changes a cell's energy by about (M_i / 2 mu V_i - 1) mu V_i e_y^2, zero in the mean
  flag          at Newton's start on the predictor, when the reading e passes flag_at e_y, and a step whose solution has an elastic cell past e_y is solved again with it flagged (the solve is repeatable until end()): the limit then holds inside the step. One way. A flagged cell under the limit is elastic
  remainder     a Maxwell element, not a stiffness to switch. Its rate relaxes at kappa g, g = |D| / e_y the cell's flow rate, so w beta_r r with beta_r = 1 / (1 + kappa (1 / beta - 1)), beta the cell's return factor, frozen at Newton's start as merged_ul.py's. A fluctuation D' at om over a flow of rate g: true excess dissipation mu g |D'_perp|^2 / (g^2 + om^2); with the springs' energy s' times the FE's (s' = 2 at w), yield on the tensor only:
                5 s' / 2 . kappa (1 + x^2) / (kappa^2 + x^2) of it, x = om / g, the 5/4 since only the 4 of 5 deviatoric directions across the flow count. Slow fluctuations: kappa = 5 at w, exact. In steady flow the viscosity is that of a fixed k = w / 5 at kappa 1 (merged_ul.py's 1/6, by trial, reads 0.83). The stiffness never switches: beta = 1 gives the spring itself
  shares        between two flowing cells the wall's spring 2 mu A / L goes by t = h / L (clipped .02 .. .98): the one-sided reading has the cell's gradient in it, and a tiny cell's gradient is its gain; on the rest list halves, as the ring
  cut           none while nothing is flagged, then each step: as merged_ul.py, x -> G linear within a step on the current cut. Rest list while a cell of the pair has not flagged, current walls when both have, born at r = 0, gone with their wall. Ground balls: the rest radii on elastic cells, the cut's on flowing ones
Self contact not here. set_ball, start(), prepare(), update_ul; through FlowStep. ring_flow_check.py its checks, ring_flow_drop.py the landing."""
import torch
import warp as wp

from free_surface.closure_warp import k_Aul, k_F8, k_Gul, k_Gul_adj, k_ball_adj, rot_proper
from free_surface.merged_ul import MergedUL
from free_surface.ring_plastic import RingStep
from free_surface.ring_ul import RingUL
from free_surface.skin_warp import NSLOT, NT, k_total, k_total_loss
from free_surface.ul_plastic import k_Aul_adj2, k_Ge, k_Ge_adj, k_gg_cell, k_gg_cell_adj, k_gg_wall, k_gg_wall_adj, k_var, polar_batch

f64 = torch.float64


@wp.kernel(enable_backward=False)
def k_rf_shape(Fc: wp.array(dtype=wp.mat33d), Rh: wp.array(dtype=wp.mat33d), en: wp.array(dtype=wp.mat33d), V0: wp.array(dtype=wp.float64), mu: wp.float64, sy: wp.float64, th_: wp.float64, pic: wp.array(dtype=int), bfz: wp.array(dtype=wp.float64),
               ei: wp.array(dtype=wp.float64), be: wp.array(dtype=wp.float64), er: wp.array(dtype=wp.mat33d), Fb: wp.array(dtype=wp.mat33d), terms: wp.array2d(dtype=wp.float64)):
    """merged_ul.py's k_cell_shape, with the secant for the products: past the limit W = s_y |e| is a cone, its Newton model has no curvature along e and runs through the apex (ratios -.7, .9, -.4 for 300 iterations on a small limit). With pic the force is linear, 2 mu beta(z) r: the quadratic upper bound of the cone at z, so a step of the model lowers W. Equal to the true force at z itself"""
    b = wp.tid()
    I = wp.identity(n=3, dtype=wp.float64)
    A = wp.transpose(Rh[b])@Fc[b]
    s = wp.float64(0.5)*(A+wp.transpose(A))-I
    e = s-(wp.trace(s)/wp.float64(3.0))*I
    ca = wp.float64(1.0)-th_
    r = (e-ca*en[b])/th_
    nr = wp.sqrt(wp.ddot(r, r))
    ei[b] = nr
    er[b] = r
    two_mu = wp.float64(2.0)*mu
    w = mu*nr*nr
    bb = wp.float64(1.0)
    if two_mu*nr > sy:
        w = sy*nr-sy*sy/(wp.float64(2.0)*two_mu)
        bb = sy/(two_mu*nr)
    be[b] = bb
    wp.atomic_add(terms, 0, b % NSLOT, V0[b]*th_*th_*w)
    if ca > wp.float64(0.0):
        wp.atomic_add(terms, 3, b % NSLOT, V0[b]*(two_mu*ca*wp.ddot(en[b], e-en[b])+(wp.float64(1.0)-th_*th_)*mu*wp.ddot(en[b], en[b])))
    if pic[0] != 0:
        bb = bfz[b]
    Fb[b] = V0[b]*(Rh[b]@((two_mu*ca)*en[b]+(th_*bb*two_mu)*r))


@wp.kernel(enable_backward=False)
def k_rf_F(G: wp.array(dtype=wp.mat33d), Gr: wp.array(dtype=wp.mat33d), F0: wp.array(dtype=wp.mat33d), flag: wp.array(dtype=int), Fc: wp.array(dtype=wp.mat33d)):
    """the cell's F: flowing, a running product on the current cut; elastic, a function of x on the rest cut"""
    b = wp.tid()
    if flag[b] != 0:
        Fc[b] = F0[b]+G[b]@F0[b]
    else:
        Fc[b] = wp.identity(n=3, dtype=wp.float64)+Gr[b]


@wp.kernel(enable_backward=False)
def k_rf_Fadj(Fb: wp.array(dtype=wp.mat33d), F0: wp.array(dtype=wp.mat33d), flag: wp.array(dtype=int), Gb: wp.array(dtype=wp.mat33d), Gbr: wp.array(dtype=wp.mat33d)):
    b = wp.tid()
    if flag[b] != 0:
        Gb[b] = Fb[b]@wp.transpose(F0[b])
        Gbr[b] = wp.mat33d()
    else:
        Gb[b] = wp.mat33d()
        Gbr[b] = Fb[b]


@wp.kernel(enable_backward=False)
def k_rf_L(Fc: wp.array(dtype=wp.mat33d), G: wp.array(dtype=wp.mat33d), C: wp.array(dtype=wp.mat33d), Ln: wp.array(dtype=wp.float64), V0: wp.array(dtype=wp.float64), kc: wp.array(dtype=wp.float64), flag: wp.array(dtype=int),
           L: wp.array(dtype=wp.float64), dL: wp.array(dtype=wp.mat33d), Lb: wp.array(dtype=wp.float64), terms: wp.array2d(dtype=wp.float64)):
    """L and its energy, the modulus by the cell's form (kc: lam on the springs, which hold 2 mu of the volumetric; kap on the cell). Flowing: L_n + tr(C^T G). Elastic: ln det F, under J 0.05 continued linearly in J, dL/dF"""
    b = wp.tid()
    half = wp.float64(0.5)
    lb = Ln[b]+wp.trace(wp.transpose(C[b])@G[b])
    if flag[b] == 0:
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
        dL[b] = wp.matrix_from_cols(dl*c0, dl*c1, dl*c2)
    L[b] = lb
    wp.atomic_add(terms, 1, b % NSLOT, V0[b]*half*kc[b]*lb*lb)
    wp.atomic_add(Lb, b, V0[b]*kc[b]*lb)


@wp.kernel(enable_backward=False)
def k_rf_Lb(Lb: wp.array(dtype=wp.float64), C: wp.array(dtype=wp.mat33d), dL: wp.array(dtype=wp.mat33d), flag: wp.array(dtype=int), Gb: wp.array(dtype=wp.mat33d), Gbr: wp.array(dtype=wp.mat33d)):
    """dE/dL through L's reading, onto the cell's own gradient"""
    b = wp.tid()
    if flag[b] != 0:
        Gb[b] = Gb[b]+Lb[b]*C[b]
    else:
        Gbr[b] = Gbr[b]+Lb[b]*dL[b]


@wp.kernel(enable_backward=False)
def k_rf_A(x: wp.array(dtype=wp.vec3d), ri: wp.array(dtype=int), rj: wp.array(dtype=int), wi: wp.array(dtype=wp.float64), wj: wp.array(dtype=wp.float64), qi: wp.array(dtype=wp.vec3d), qj: wp.array(dtype=wp.vec3d), Am: wp.array(dtype=wp.mat33d)):
    """sum w d (x) q per cell (read only on the elastic cells)"""
    e = wp.tid()
    i, j = ri[e], rj[e]
    if i == j:
        return
    d = x[j]-x[i]
    wp.atomic_add(Am, i, wi[e]*wp.outer(d, qi[e]))
    wp.atomic_add(Am, j, wj[e]*wp.outer(d, qj[e]))


@wp.kernel(enable_backward=False)
def k_rf_R(Am: wp.array(dtype=wp.mat33d), Fc: wp.array(dtype=wp.mat33d), Mo: wp.array(dtype=wp.mat33d), flag: wp.array(dtype=int), Rc: wp.array(dtype=wp.mat33d)):
    """the elastic cell's rotation, its whole minimum"""
    b = wp.tid()
    if flag[b] != 0:
        return
    Rc[b] = rot_proper(Am[b]+Fc[b]@Mo[b])


@wp.kernel(enable_backward=False)
def k_rf_outer(Fc: wp.array(dtype=wp.mat33d), Rc: wp.array(dtype=wp.mat33d), Mo: wp.array(dtype=wp.mat33d), flag: wp.array(dtype=int), Fb: wp.array(dtype=wp.mat33d), terms: wp.array2d(dtype=wp.float64)):
    """the elastic cell's outer layer, H = R^T F_o - I, 1/2 tr(H Mo H^T); term 5 as RingUL; onto dE/dF"""
    b = wp.tid()
    if flag[b] != 0:
        return
    H = wp.transpose(Rc[b])@Fc[b]-wp.identity(n=3, dtype=wp.float64)
    Z = H@Mo[b]
    wp.atomic_add(terms, 5, b % NSLOT, wp.float64(0.5)*wp.ddot(Z, H))
    Fb[b] = Fb[b]+Rc[b]@Z


@wp.kernel(enable_backward=False)
def k_rf_half(x: wp.array(dtype=wp.vec3d), G: wp.array(dtype=wp.mat33d), Rc: wp.array(dtype=wp.mat33d), flag: wp.array(dtype=int), ri: wp.array(dtype=int), rj: wp.array(dtype=int), wi: wp.array(dtype=wp.float64), wj: wp.array(dtype=wp.float64), qi: wp.array(dtype=wp.vec3d), qj: wp.array(dtype=wp.vec3d),
              dni: wp.array(dtype=wp.vec3d), dnj: wp.array(dtype=wp.vec3d), bi: wp.array(dtype=wp.float64), bj: wp.array(dtype=wp.float64), th_: wp.float64, sri: wp.array(dtype=wp.vec3d), srj: wp.array(dtype=wp.vec3d), terms: wp.array2d(dtype=wp.float64), x_adj: wp.array(dtype=wp.vec3d), Gb: wp.array(dtype=wp.mat33d)):
    """each half by its cell's form. Elastic: 1/2 w |R^T d - q|^2, term 0. Flowing: the remainder at the step's end r_e = (d - p - G p - (1 - th) dn) / th; th^2 1/2 w b r_e^2 + (1 - th) w dn . (r - dn) + (1 - th^2) 1/2 w dn^2, term 4, as merged_ul.py's spring; its force w ((1 - th) dn + th b r_e), and -force (x) p onto dE/dG"""
    e = wp.tid()
    i, j = ri[e], rj[e]
    if i == j:
        return
    half = wp.float64(0.5)
    ca = wp.float64(1.0)-th_
    d = x[j]-x[i]
    f = wp.vec3d()
    if flag[i] == 0:
        a = wp.transpose(Rc[i])@d-qi[e]
        wp.atomic_add(terms, 0, e % NSLOT, half*wi[e]*wp.dot(a, a))
        f = f+wi[e]*(Rc[i]@a)
    else:
        p = qi[e]
        dn = dni[e]
        r = d-p-G[i]@p
        re = (r-ca*dn)/th_
        sri[e] = re
        w = wi[e]
        b = bi[e]
        wp.atomic_add(terms, 4, e % NSLOT, th_*th_*half*w*b*wp.dot(re, re)+ca*w*wp.dot(dn, r-dn)+(wp.float64(1.0)-th_*th_)*half*w*wp.dot(dn, dn))
        fi = w*(ca*dn+(th_*b)*re)
        f = f+fi
        wp.atomic_add(Gb, i, -wp.outer(fi, p))
    if flag[j] == 0:
        a = wp.transpose(Rc[j])@d-qj[e]
        wp.atomic_add(terms, 0, e % NSLOT, half*wj[e]*wp.dot(a, a))
        f = f+wj[e]*(Rc[j]@a)
    else:
        p = qj[e]
        dn = dnj[e]
        r = d-p-G[j]@p
        re = (r-ca*dn)/th_
        srj[e] = re
        w = wj[e]
        b = bj[e]
        wp.atomic_add(terms, 4, e % NSLOT, th_*th_*half*w*b*wp.dot(re, re)+ca*w*wp.dot(dn, r-dn)+(wp.float64(1.0)-th_*th_)*half*w*wp.dot(dn, dn))
        fi = w*(ca*dn+(th_*b)*re)
        f = f+fi
        wp.atomic_add(Gb, j, -wp.outer(fi, p))
    wp.atomic_add(x_adj, j, f)
    wp.atomic_add(x_adj, i, -f)


@wp.kernel(enable_backward=False)
def k_rf_ball(x: wp.array(dtype=wp.vec3d), cbody: wp.array(dtype=int), R: wp.array(dtype=wp.float64), xn: wp.array(dtype=wp.vec3d), ng: wp.vec3d, bpar: wp.array(dtype=wp.float64), kt: wp.array(dtype=wp.float64), bound: wp.array(dtype=wp.float64),
              lamt: wp.array(dtype=wp.vec3d), stuck: wp.array(dtype=int), kn: wp.array(dtype=wp.float64), delta: wp.array(dtype=wp.float64), fn: wp.array(dtype=wp.float64), ff: wp.array(dtype=wp.vec3d), terms: wp.array2d(dtype=wp.float64)):
    """closure_warp.py's k_ball, with the hold: a ball that has touched is linear below a sinking of 5 % of its radius, C1 with Hertz there, and so pulls when it leaves, its rest at a third of that (Hertz down to 0 would have the stiffness jump from k_n to 0 at d = 0, where a thousand balls rest: Newton's ratio cycled -.06, .73, .47 for 300 iterations). bpar = (cs, wt, c0, E*, mu)"""
    c = wp.tid()
    B = cbody[c]
    cs, wt, c0, Es, mu = bpar[0], bpar[1], bpar[2], bpar[3], bpar[4]
    xe = cs*x[B]-(cs-wp.float64(1.0))*xn[c]
    d = R[c]-(wp.dot(ng, xe)-c0)
    delta[c] = d
    F = wp.float64(0.0)
    dr = wp.float64(0.05)*R[c]
    if stuck[c] != 0 and R[c] > wp.float64(0.0) and d < dr:
        sr = wp.sqrt(R[c]*dr)
        Fr = wp.float64(4.0)/wp.float64(3.0)*Es*sr*dr
        dd = d-dr
        F = Fr+kn[c]*dd
        wp.atomic_add(terms, 6, c % NSLOT, wt*(wp.float64(8.0)/wp.float64(15.0)*Es*sr*dr*dr+Fr*dd+wp.float64(0.5)*kn[c]*dd*dd))
    elif d > wp.float64(0.0):
        sq = wp.sqrt(R[c]*d)
        F = wp.float64(4.0)/wp.float64(3.0)*Es*sq*d
        wp.atomic_add(terms, 6, c % NSLOT, wt*wp.float64(8.0)/wp.float64(15.0)*Es*sq*d*d)
    fn[c] = F
    f = wp.vec3d()
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


class RingFlow(RingUL):
    """X, r, V0: the foam at rest; pairs, A, c: its walls (geometry.npz); E, nu, sigma_y per unit density; kappa: the remainder's rate over the cell's (5 at the springs' w); flag_at: the predictor's reading, in e_y, that flags a cell; hold: a ball that has touched does not leave the ground (the MPM's sticky ground holds tension; in the plane only the friction, as merged_ul.py: a site stands a ball's radius above the ground, and pinned without a limit it lifts the no-slip plane by that: landing y +20 % against +8 %); ledger: account the energy that the flag and the cut change at fixed x"""
    C_cap = 4.

    def __init__(self, X, r, V0, pairs, A, c, E, nu, sigma_y, theta=.5, kappa=5., flag_at=.5, hold=False, ledger=False, **kw):
        super().__init__(X, r, V0, pairs, A, c, E, nu, theta=theta, sigma_y=sigma_y, **kw)
        self.kappa, self.flag_at = float(kappa), float(flag_at)
        self.hold, self.ledger = bool(hold), bool(ledger)
        self.Wp_half = 0.
        self.Eflag = self.Ecut = 0.
        self.n_flagged_end = 0
        self._cut_valid = False

    def start(self):
        """the rest cut, once: the elastic cells' gradient, frozen in its own arrays at its own size; the half springs at rest"""
        n, dev = self.n, self.dev
        MergedUL.update_ul(self, self.X)
        t, U = self.tul, self.ul
        i_w, j_w, Lw, _, _, m = self._w; ml = self._ml
        k3 = max(1, int((t['Aw'] != 0).sum()))
        tt = lambda a_: wp.to_torch(a_, requires_grad=False)
        z = lambda k_, dt_: wp.zeros(k_, dtype=dt_, device=dev)
        Ur = dict(xn=z(n, wp.vec3d), S=z(2*ml, wp.int32), O=z(2*ml, wp.int32), R=z(2*ml, wp.vec3d), As=z(k3, wp.int32), Ao=z(k3, wp.int32), Aw=z(k3, wp.float64), used=z(n, wp.int32),
                  wi=z(m, wp.int32), wj=z(m, wp.int32), wa=z(m, wp.float64), wn=z(m, wp.vec3d), wt=z(m, wp.float64), wtau=z(m, wp.vec3d), a=z(n, wp.vec3d), Mc=z(n, wp.mat33d), Vi=z(n, wp.float64))
        for k_, a_ in Ur.items():
            v_ = tt(a_)
            v_.copy_(t[k_][:len(v_)])
        Ur.update(Kl=2*ml, Ka=k3, Kw=m, wv=z(m, wp.float64))
        self.ur, self.turv = Ur, tt(Ur['wv'])
        self._xs = self.X.clone()
        self._wvr0 = self.lam*t['wa'][:m]*Lw/24
        self._irw, self._jrw = i_w.clone(), j_w.clone()
        self.turv.copy_(self._wvr0)
        self.Br = dict(G0=z(n, wp.mat33d), Ga=z(n, wp.mat33d), Ge=z(n, wp.mat33d), Sg=z(n, wp.mat33d), G=z(n, wp.mat33d), Gb=z(n, wp.mat33d), Gs=z(n, wp.mat33d), Gbe=z(n, wp.mat33d), Gb0=z(n, wp.mat33d))
        # the rest rows for the blocks
        A_, tw, nn, Vi = t['wa'][:m], t['wt'][:m], t['wn'][:m], t['Vi']
        self._rrows = self._rows_from(i_w, j_w, A_, tw, nn, Vi, ml)
        self._tbR0 = self.tbR.clone() if self.ball is not None else None
        Ks = U['Kw']
        self.hd = dict(stuck=z(n, wp.int32), kn=z(n, wp.float64), pic=z(1, wp.int32), bfz=z(n, wp.float64))
        self.tpic, self.tbfz = tt(self.hd['pic']), tt(self.hd['bfz'])
        self.ring = dict(K=Ks, i=z(Ks, wp.int32), j=z(Ks, wp.int32), wi=z(Ks, wp.float64), wj=z(Ks, wp.float64), qi=z(Ks, wp.vec3d), qj=z(Ks, wp.vec3d), dni=z(Ks, wp.vec3d), dnj=z(Ks, wp.vec3d), bi=z(Ks, wp.float64), bj=z(Ks, wp.float64), sri=z(Ks, wp.vec3d), srj=z(Ks, wp.vec3d),
                         Am=z(n, wp.mat33d), Rc=z(n, wp.mat33d), Mo=z(n, wp.mat33d), dL=z(n, wp.mat33d), flag=z(n, wp.int32), kc=z(n, wp.float64), V0f=z(n, wp.float64))
        self.tr = {k_: tt(a_) for k_, a_ in self.ring.items() if k_ != 'K'}
        T = self.tr
        T['kc'].fill_(self.lam)
        Mc = t['Mc']
        T['Mo'].copy_(self.mu*(Mc+Mc.transpose(1, 2)))
        self._qkey = self.tkey.clone()
        D = self.X[self.thi]-self.X[self.tlo]
        self._qi, self._qj = D.clone(), D.clone()
        zv = torch.zeros_like(D)
        self._dli, self._dlj = zv.clone(), zv.clone()
        self._set_halves(self.X)
        self._cut_valid = False
        self._wv_masks()
        ms = self._m_s
        Mr = T['Mo'].clone()
        Mr.index_add_(0, self.tlo, T['wi'][:ms, None, None]*D[:, :, None]*D[:, None, :]); Mr.index_add_(0, self.thi, T['wj'][:ms, None, None]*D[:, :, None]*D[:, None, :])
        self._Mri = torch.linalg.pinv(Mr)
        self.tstuck, self.tbkn = tt(self.hd['stuck']), tt(self.hd['kn'])
        self._stuck = torch.zeros(n, dtype=torch.bool, device=dev)
        self._Rst = torch.zeros(n, dtype=f64, device=dev)
        self.tbkt, self.tbd = tt(self.ball['kt']), tt(self.B['bdelta'])

    def _rows_from(self, i, j, A, tw, nn, Vi, ml):
        """the gradient's direct rows on each cell (sc, kc, r): for the blocks"""
        n, t, dev = self.n, self.tul, self.dev
        ci, cj = (A*Vi[i])[:, None]*nn, (A*Vi[j])[:, None]*nn
        s_ = [i, i, j, j]; k_ = [i, j, i, j]; r_ = [(1-tw)[:, None]*ci, tw[:, None]*ci, -(1-tw)[:, None]*cj, -tw[:, None]*cj]
        ids = torch.arange(n, device=dev)
        s_.append(ids); k_.append(ids); r_.append(Vi[:, None]*t['a'])
        S, O, R_ = t['S'][:2*ml].long(), t['O'][:2*ml].long(), t['R'][:2*ml]
        cap = (t['Mc'].abs().sum((1, 2)) > 0)
        q = cap[S]
        rm = torch.einsum('fba,fb->fa', t['Mc'][S[q]], R_[q])*Vi[S[q]][:, None]
        s_ += [S[q], S[q]]; k_ += [O[q], S[q]]; r_ += [rm, -rm]
        sa, ka, ra = torch.cat(s_), torch.cat(k_), torch.cat(r_)
        u, inv = torch.unique(sa*n+ka, return_inverse=True)
        r = torch.zeros(len(u), 3, dtype=f64, device=dev).index_add(0, inv, ra)
        return u//n, u % n, r

    def _set_halves(self, x):
        """the half springs by their pair at x: an elastic cell's rest q, a flowing cell's p = d - its kept remainder; a new one with no remainder; the shares between two flowing cells"""
        dev, t, T, yld = self.dev, self.tul, self.tr, self._yld
        i_w, j_w, _, _, _, m = self._w
        lo, hi, key, ms = self.tlo, self.thi, self.tkey, self._m_s
        pos = torch.searchsorted(self._qkey, key).clamp(max=len(self._qkey)-1); kept = self._qkey[pos] == key
        d = x[hi]-x[lo]
        qi, qj = d.clone(), d.clone()
        dli = torch.zeros(ms, 3, dtype=f64, device=dev); dlj = torch.zeros(ms, 3, dtype=f64, device=dev)
        qi[kept] = self._qi[pos[kept]]; qj[kept] = self._qj[pos[kept]]
        dli[kept] = self._dli[pos[kept]]; dlj[kept] = self._dlj[pos[kept]]
        self._qkey, self._qi, self._qj, self._dli, self._dlj = key.clone(), qi, qj, dli, dlj
        fl, fh = yld[lo], yld[hi]
        pi_ = torch.where(fl[:, None], d-dli, qi); pj_ = torch.where(fh[:, None], d-dlj, qj)
        w = self.tk0/(2*self.kshare)
        lo_w, hi_w = torch.minimum(i_w, j_w), torch.maximum(i_w, j_w)
        kw = lo_w*self.n+hi_w
        tlo_w = torch.where(i_w < j_w, t['wt'][:m], 1-t['wt'][:m])
        o = torch.argsort(kw)
        pw = torch.searchsorted(kw[o], key).clamp(max=m-1); hw = kw[o][pw] == key
        share = torch.full((ms,), .5, dtype=f64, device=dev)
        both = fl & fh & hw
        share[both] = tlo_w[o][pw[both]].clamp(.02, .98)
        for k_ in ('i', 'j', 'wi', 'wj', 'qi', 'qj', 'dni', 'dnj', 'bi', 'bj', 'sri', 'srj'):
            T[k_].zero_()
        T['i'][:ms] = lo.to(torch.int32); T['j'][:ms] = hi.to(torch.int32); T['wi'][:ms] = 2*share*w; T['wj'][:ms] = 2*(1-share)*w; T['qi'][:ms] = pi_; T['qj'][:ms] = pj_; T['dni'][:ms] = dli; T['dnj'][:ms] = dlj; T['bi'][:ms] = 1.; T['bj'][:ms] = 1.

    def _wv_masks(self):
        """the variance: on the rest walls between two elastic cells, on the current walls otherwise (kap between two flowing cells)"""
        yld, t = self._yld, self.tul
        self.turv.copy_(torch.where(~yld[self._irw] & ~yld[self._jrw], self._wvr0, torch.zeros_like(self._wvr0)))
        if self._cut_valid:
            i_w, j_w, Lw, _, _, m = self._w
            bw, an = yld[i_w] & yld[j_w], yld[i_w] | yld[j_w]
            md = torch.where(bw, torch.full_like(Lw, self.kap), torch.full_like(Lw, self.lam))
            t['wv'][:m] = torch.where(an, md*t['wa'][:m]*Lw/24, torch.zeros_like(Lw))
        else:
            t['wv'].zero_()

    def _cut(self, x):
        """the current cut at x: merged_ul.py's list, gradient and the cells' start strains; the halves; the ground balls"""
        MergedUL.update_ul(self, x)
        self._cut_valid = True
        self._set_halves(x)
        if self.ball is not None:
            self.tbR.copy_(torch.where(self._yld, self.tbR, self._tbR0))
            self.tbR[self._stuck] = self._Rst[self._stuck]                               # (a ball that touches keeps its radius: a new one changes the contact's energy at fixed x, 9 % of the start's energy by 0.28 s of the landing against 0.8 %)
        self._wv_masks()

    def _start_cells(self):
        """the cells' start strains without a cut"""
        s = self.tsp
        s['F0'].copy_(self._Fe)
        wp.launch(k_F8, self.n, inputs=[self.sp['F0']], outputs=[self.B['Fp'], self.B['Rp'], self.B['qp']])
        s['Rh'].copy_(self.tRp)
        A_ = self.tRp.transpose(1, 2)@self._Fe; e_ = .5*(A_+A_.transpose(1, 2))-self.I3
        s['en'].copy_(e_-(e_.diagonal(dim1=1, dim2=2).sum(1)/3)[:, None, None]*self.I3)

    def update_ul(self, x):
        """the step's start at x: the cut, once anything has flagged"""
        x = x.detach().clone()
        cut = bool(self._yld.any())
        e0 = self.energy_at(x) if cut and self.ledger else 0.
        self._cut_valid = False
        if cut:
            self._xs = x
            self._cut(x)
            if self.ledger:
                self.Ecut += self.energy_at(x)-e0
        else:
            self._xs = x
            self._start_cells()
            self.tr['bi'].fill_(1.); self.tr['bj'].fill_(1.)

    def flag_new(self, mask):
        """these cells go to the flowing form from the step's start on: their halves' p = F_o q, dn = d_n - p; the outer layer into the cell. Number flagged"""
        new = mask & ~self._yld
        k = int(new.sum())
        if k == 0:
            return 0
        e0 = self.energy_at(self._xs) if self.ledger else 0.
        if not self._cut_valid:
            self._cut(self._xs)
        T, ms = self.tr, self._m_s
        lo, hi = self.tlo, self.thi
        F0 = self.tsp['F0']
        d = self._xs[hi]-self._xs[lo]
        a, b = new[lo], new[hi]
        qi_, qj_, dni_, dnj_ = T['qi'][:ms], T['qj'][:ms], T['dni'][:ms], T['dnj'][:ms]
        # p on the whole cell's fit: r = d - F_fit q is what S_i's split calls the remainder; on F_o the difference of the two readings would be charged to it, 1/2 tr((F_fit - F_o) M (F_fit - F_o)^T), never negative
        Af = (F0@T['Mo']).index_add(0, lo, T['wi'][:ms, None, None]*d[:, :, None]*qi_[:, None, :]).index_add(0, hi, T['wj'][:ms, None, None]*d[:, :, None]*qj_[:, None, :])
        Ff = Af@self._Mri
        pa = torch.einsum('fab,fb->fa', Ff[lo[a]], qi_[a]); pb = torch.einsum('fab,fb->fa', Ff[hi[b]], qj_[b])
        qi_[a] = pa; dni_[a] = d[a]-pa
        qj_[b] = pb; dnj_[b] = d[b]-pb
        self._dli[a] = d[a]-pa; self._dlj[b] = d[b]-pb
        self._yld = self._yld | new
        T['Mo'][new] = 0.; T['flag'][new] = 1; T['kc'][new] = self.kap; T['V0f'][new] = self.V0[new]
        self._wv_masks()
        if self.ledger:
            self.Eflag += self.energy_at(self._xs)-e0
        return k

    def run(self):
        """E and dE/dx at xw: both gradients, F; L; the elastic cells' rotations; the cells; the halves; back through both gradients"""
        B, U, Ur, Br, Rg, S, n, Kw = self.B, self.ul, self.ur, self.Br, self.ring, self.sp, self.n, self.ul['Kw']
        x, g = self.xw, self.xw.grad
        for k_ in ('G0', 'Ga', 'Sg', 'Lb', 'Gbe', 'Gb0', 'terms', 'tsum'):
            B[k_].zero_()
        for k_ in ('G0', 'Ga', 'Sg', 'Gbe', 'Gb0'):
            Br[k_].zero_()
        Rg['Am'].zero_()
        g.zero_()
        th_ = wp.float64(self.th)
        wall = [U['wi'], U['wj'], U['wa'], U['wn'], U['wt'], U['wtau']]
        wallr = [Ur['wi'], Ur['wj'], Ur['wa'], Ur['wn'], Ur['wt'], Ur['wtau']]
        wp.launch(k_Gul, 2*Kw, inputs=[x, U['xn'], U['S'], U['O'], U['R']], outputs=[B['G0']])
        wp.launch(k_Aul, Kw, inputs=[B['G0'], U['As'], U['Ao'], U['Aw']], outputs=[B['Ga']])
        wp.launch(k_Ge, n, inputs=[B['G0'], B['Ga'], U['used']], outputs=[B['Ge']])
        wp.launch(k_gg_wall, Kw, inputs=[x, U['xn']]+wall+[B['Ge']], outputs=[B['Sg']])
        wp.launch(k_gg_cell, n, inputs=[x, U['xn'], B['Sg'], B['Ge'], U['a'], U['Mc'], U['Vi']], outputs=[B['G']])
        wp.launch(k_Gul, Ur['Kl'], inputs=[x, Ur['xn'], Ur['S'], Ur['O'], Ur['R']], outputs=[Br['G0']])
        wp.launch(k_Aul, Ur['Ka'], inputs=[Br['G0'], Ur['As'], Ur['Ao'], Ur['Aw']], outputs=[Br['Ga']])
        wp.launch(k_Ge, n, inputs=[Br['G0'], Br['Ga'], Ur['used']], outputs=[Br['Ge']])
        wp.launch(k_gg_wall, Ur['Kw'], inputs=[x, Ur['xn']]+wallr+[Br['Ge']], outputs=[Br['Sg']])
        wp.launch(k_gg_cell, n, inputs=[x, Ur['xn'], Br['Sg'], Br['Ge'], Ur['a'], Ur['Mc'], Ur['Vi']], outputs=[Br['G']])
        wp.launch(k_rf_F, n, inputs=[B['G'], Br['G'], S['F0'], Rg['flag']], outputs=[B['Fc']])
        wp.launch(k_rf_L, n, inputs=[B['Fc'], B['G'], U['Q'], U['Ln'], U['V0'], Rg['kc'], Rg['flag']], outputs=[B['L'], Rg['dL'], B['Lb'], B['terms']])
        wp.launch(k_var, Kw, inputs=[U['wi'], U['wj'], U['wv'], B['L']], outputs=[B['Lb'], B['terms']])
        wp.launch(k_var, Ur['Kw'], inputs=[Ur['wi'], Ur['wj'], Ur['wv'], B['L']], outputs=[B['Lb'], B['terms']])
        wp.launch(k_rf_A, Rg['K'], inputs=[x, Rg['i'], Rg['j'], Rg['wi'], Rg['wj'], Rg['qi'], Rg['qj']], outputs=[Rg['Am']])
        wp.launch(k_rf_R, n, inputs=[Rg['Am'], B['Fc'], Rg['Mo'], Rg['flag']], outputs=[Rg['Rc']])
        wp.launch(k_rf_shape, n, inputs=[B['Fc'], S['Rh'], S['en'], Rg['V0f'], wp.float64(self.mu), wp.float64(self.sy), th_, self.hd['pic'], self.hd['bfz']], outputs=[B['ei'], B['be'], B['er'], B['Fb'], B['terms']])
        wp.launch(k_rf_outer, n, inputs=[B['Fc'], Rg['Rc'], Rg['Mo'], Rg['flag']], outputs=[B['Fb'], B['terms']])
        wp.launch(k_rf_Fadj, n, inputs=[B['Fb'], S['F0'], Rg['flag']], outputs=[B['Gb'], Br['Gb']])
        wp.launch(k_rf_half, Rg['K'], inputs=[x, B['G'], Rg['Rc'], Rg['flag'], Rg['i'], Rg['j'], Rg['wi'], Rg['wj'], Rg['qi'], Rg['qj'], Rg['dni'], Rg['dnj'], Rg['bi'], Rg['bj'], th_], outputs=[Rg['sri'], Rg['srj'], B['terms'], g, B['Gb']])
        wp.launch(k_rf_Lb, n, inputs=[B['Lb'], U['Q'], Rg['dL'], Rg['flag']], outputs=[B['Gb'], Br['Gb']])
        bl = self.ball
        if bl is not None:
            wp.launch(k_rf_ball, bl['k'], inputs=[x, bl['body'], bl['R'], bl['xn'], bl['ng'], bl['par'], bl['kt'], bl['bound'], bl['lamt'], self.hd['stuck'], self.hd['kn']], outputs=[B['bdelta'], B['bfn'], B['bff'], B['terms']])
            wp.launch(k_ball_adj, bl['k'], inputs=[bl['body'], bl['ng'], bl['par'], B['bfn'], B['bff'], self.tones], outputs=[g])
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

    def reread(self, beta=False):
        """R and C at the point the solve stands on; beta: Newton's start: the predictor's reading flags cells (read again on the new forms at the same point), the halves' beta_r"""
        if beta and self.flag_new(self.tei > self.flag_at*self.ey):
            wp.capture_launch(self.graph)
        MergedUL.reread(self, beta=False)
        C = self.tul['Q']
        big = C.flatten(1).norm(dim=1) > self.C_cap
        if bool(big.any()):
            # a cell the predictor flattens in the half step: C = (I + G / 2 th)^-T frozen on it would make it stiff for the whole step, for a flattening that may not come (one cell of volume 0.07 at |C| 25: first |g| 4e4 weights, 300 Newton). C is free as far as the energy goes: at th = 1/2 work = store for any C. So the half step's stretch is read down to 1/4
            M = self.I3+self.tG[big]/(2*self.th)
            U, S, Vh = torch.linalg.svd(M.cpu())
            C[big] = (U*(1/S.clamp(min=.25))[:, None, :]@Vh).to(self.dev)
        if beta:
            T, ms, lo, hi = self.tr, self._m_s, self.tlo, self.thi
            br = 1/(1+self.kappa*(1/self.tbe.clamp(min=1e-6)-1))
            T['bi'][:ms] = br[lo]; T['bj'][:ms] = br[hi]

    def end_ul(self, th=.5):
        """after the step (the buffers at the accepted z; x -> F linear, so the end's F and G are exact): the flowing halves' remainders, the cells' returns, F_o, L; the plastic work"""
        assert th == self.th, 'the model and the step on one theta'
        t, s, T, n, I3, ms = self.tul, self.tsp, self.tr, self.n, self.I3, self._m_s
        lo, hi = self.tlo, self.thi
        flag = self._yld
        fl, fh = flag[lo], flag[hi]
        wi, wj, bi, bj, sri, srj, dni, dnj = T['wi'][:ms], T['wj'][:ms], T['bi'][:ms], T['bj'][:ms], T['sri'][:ms], T['srj'][:ms], T['dni'][:ms], T['dnj'][:ms]
        if th == 1.:
            wa = .5*wi*bi*(1-bi)*(sri*sri).sum(1); wb = .5*wj*bj*(1-bj)*(srj*srj).sum(1)
        else:
            wa = .5*wi*(1-bi)*((sri*dni).sum(1)+bi*(sri*sri).sum(1)); wb = .5*wj*(1-bj)*((srj*dnj).sum(1)+bj*(srj*srj).sum(1))
        wh = float(wa[fl].sum()+wb[fh].sum())
        self.Wp_half += wh; self.Wp += wh
        self._dli = torch.where(fl[:, None], bi[:, None]*sri, self._dli); self._dlj = torch.where(fh[:, None], bj[:, None]*srj, self._dlj)
        be = self.tbe.clone()
        yn = (be < 1) & flag
        if bool(yn.any()):
            er, en = self.ter, s['en']
            if th == 1.:
                wc = (self.V0*self.sy*(self.tei-self.ey))[yn]
            else:
                wc = (self.V0*self.mu*(1-be)*((en*er).sum((1, 2))+be*(er*er).sum((1, 2))))[yn]
            self.Wp += float(wc.sum()); self.Wp_cell += float(wc.sum())
        s['Fend'].copy_((self.tFc-(1-th)*s['F0'])/th)
        wp.launch(k_F8, n, inputs=[self.sp['Fend']], outputs=[self.B['Fp'], self.B['Rp'], self.B['qp']])
        Fe, R = s['Fend'].clone(), self.tRp.clone()
        # L at the step's end: flowing, the frozen C's account; elastic, ln det F as the kernel reads it
        Ln = t['Ln']
        Ln[flag] = Ln[flag]+((t['Q']*self.tG).sum((1, 2))/th)[flag]
        J = torch.linalg.det(Fe); J0 = .05
        Le = torch.where(J > J0, torch.log(J.clamp(min=J0)), torch.log(torch.tensor(J0, dtype=f64, device=self.dev))+(J-J0)/J0)
        Ln[~flag] = Le[~flag]
        self.ul_stats['flips'] = int((J[flag] <= 1e-3).sum()) if bool(flag.any()) else 0
        if bool(flag.any()):
            # the flowing cell's stretch: its shape read, the limit held, its volume L's
            A_ = R[flag].transpose(1, 2)@Fe[flag]
            w, v = torch.linalg.eigh((.5*(A_+A_.transpose(1, 2))).cpu())           # (on the CPU, as merged_ul.py)
            w, v = w.to(A_.device), v.to(A_.device)
            lw = torch.log(w.clamp(min=1e-6)); lm = lw.mean(1, keepdim=True); dv = lw-lm
            nr = dv.norm(dim=1, keepdim=True).clamp(min=1e-300); a = torch.exp(Ln[flag]/3)[:, None]
            sc = torch.where(yn[flag][:, None], torch.minimum(torch.ones_like(nr), (self.ey/a)/nr), torch.ones_like(nr))
            Ue = v@torch.diag_embed(a*torch.exp(sc*dv))@v.transpose(1, 2)
            Fe[flag] = R[flag]@Ue
            Q = polar_batch(I3+self.tG/th)
            self.ep = Q@self.ep@Q.transpose(1, 2)
            self.ep[flag] = self.ep[flag]+R[flag]@(v@torch.diag_embed((1-sc[:, 0, None])*dv)@v.transpose(1, 2))@R[flag].transpose(1, 2)
        self._Fe = Fe
        # (an elastic cell still past the limit after the step's repeats: flagged here, its spring state read at the step's end, its stress kept for one step)
        over = (be < 1) & ~flag
        self.n_flagged_end = int(over.sum())
        if bool(over.any()):
            xe = (self.tx-(1-th)*self._xs)/th
            d = xe[hi]-xe[lo]
            a, b = over[lo], over[hi]
            self._dli[a] = d[a]-torch.einsum('fab,fb->fa', Fe[lo[a]], self._qi[a]); self._dlj[b] = d[b]-torch.einsum('fab,fb->fa', Fe[hi[b]], self._qj[b])
            self._yld = self._yld | over
            T['Mo'][over] = 0.; T['flag'][over] = 1; T['kc'][over] = self.kap; T['V0f'][over] = self.V0[over]
        self.ul_stats['yielding'] = int(yn.sum()); self.ul_stats['yielded'] = int(self._yld.sum())
        if self.hold:
            nw = (self.tbd > 0) & ~self._stuck & (self.tbR > 0)
            if bool(nw.any()):
                self._stuck = self._stuck | nw
                self._Rst[nw] = self.tbR[nw]
                self.tbkn[nw] = 2*self.Es*self.tbR[nw]*.05**.5                           # (Hertz's stiffness at a sinking of 5 % of the radius)
                self.tstuck[nw] = 1

    def ground_store(self, x):
        """the balls' elastic energy at x: normal, and the glue's as it was kept"""
        if self.ball is None:
            return 0.
        c0, Es = float(self.tball[2]), self.Es
        ng = torch.as_tensor([float(self.ball['ng'][k]) for k in range(3)], dtype=f64, device=self.dev)
        R = self.tbR
        d = R-((x*ng).sum(1)-c0)
        dr = .05*R
        lin = self._stuck & (R > 0) & (d < dr) if self.hold else torch.zeros_like(self._stuck)
        on = (d > 0) & (R > 0) & ~lin
        En = float((8/15*Es*torch.sqrt(R[on])*d[on]**2.5).sum())
        if self.hold and bool(lin.any()):
            r_, dd, kn_ = R[lin], (d-dr)[lin], self.tbkn[lin]
            sr = torch.sqrt(r_*dr[lin])
            En += float((8/15*Es*sr*dr[lin]**2+4/3*Es*sr*dr[lin]*dd+.5*kn_*dd*dd).sum())
        lt = wp.to_torch(self.ball['lamt'], requires_grad=False)
        kt = self.tbkt
        ok = kt > 0
        return En+float((.5*(lt[ok]*lt[ok]).sum(1)/kt[ok]).sum())

    def energy_at(self, x):
        a, b = self.stored(x)
        return a+b+self.ground_store(x)

    def stored(self, x=None):
        """the elastic energy of the state after the last end_ul, at the step's end x (or at x): shape (springs and outer layer on the elastic cells, the cell and the kept remainders on the flowing), volume"""
        T, ms, lo, hi = self.tr, self._m_s, self.tlo, self.thi
        Fe, flag, t = self._Fe, self._yld, self.tul
        if x is None:
            x = self.tx if self.th == 1. else (self.tx-(1-self.th)*self._xs)/self.th
        d = x[hi]-x[lo]
        qi, qj, wi, wj = self._qi, self._qj, T['wi'][:ms], T['wj'][:ms]
        Mo = T['Mo']
        fl, fh = flag[lo], flag[hi]
        Es = Eo = Ec = 0.
        if not bool(flag.all()):
            A = (Fe@Mo).index_add(0, lo, wi[:, None, None]*d[:, :, None]*qi[:, None, :]).index_add(0, hi, wj[:, None, None]*d[:, :, None]*qj[:, None, :])
            R = self.I3.expand(self.n, 3, 3).clone()
            Us, _, Vh = torch.linalg.svd(A[~flag].cpu())                                 # (log only; not polar_batch, whose det floor reads I on a cell of anisotropic M)
            Sg = torch.diag_embed(torch.stack([torch.ones(len(Us), dtype=f64), torch.ones(len(Us), dtype=f64), torch.linalg.det(Us@Vh)], 1))
            R[~flag] = (Us@Sg@Vh).to(self.dev)
            mi = torch.einsum('fba,fb->fa', R[lo], d)-qi; mj = torch.einsum('fba,fb->fa', R[hi], d)-qj
            Es = float((.5*wi*(mi*mi).sum(1))[~fl].sum()+(.5*wj*(mj*mj).sum(1))[~fh].sum())
            H = R.transpose(1, 2)@Fe-self.I3
            Eo = float((.5*((H@Mo)*H).sum((1, 2)))[~flag].sum())
        Eh = float((.5*wi*(self._dli*self._dli).sum(1))[fl].sum()+(.5*wj*(self._dlj*self._dlj).sum(1))[fh].sum())
        if bool(flag.any()):
            wp.launch(k_F8, self.n, inputs=[wp.from_torch(Fe.contiguous(), dtype=wp.mat33d)], outputs=[self.B['Fp'], self.B['Rp'], self.B['qp']])
            A_ = self.tRp.transpose(1, 2)@Fe; e_ = .5*(A_+A_.transpose(1, 2))-self.I3
            e_ = e_-(e_.diagonal(dim1=1, dim2=2).sum(1)/3)[:, None, None]*self.I3
            Ec = float((self.V0*self.mu*(e_*e_).sum((1, 2)))[flag].sum())
        L = t['Ln']
        Ev = float((self.V0*.5*T['kc']*L*L).sum())
        Ev += float((self.turv*(L[self._irw]-L[self._jrw])**2).sum())
        if self._cut_valid:
            i_w, j_w, m = self._w[0], self._w[1], self._w[5]
            Ev += float((t['wv'][:m]*(L[i_w]-L[j_w])**2).sum())
        return Es+Eo+Eh+Ec, Ev

    def ul_blocks(self, lr_thr=None, lr_cap=0):
        """the diagonal blocks at the last evaluation: through the gradient's direct rows, rest rows on the elastic cells (the pressure, the outer layer), the current ones on the flowing (the shape at its return factor, the pressure); the halves' w, a flowing one's at beta_r, on each end"""
        assert lr_thr is None, 'no low-rank term here'
        n, t, dev, T = self.n, self.tul, self.dev, self.tr
        flag = self._yld
        K = torch.zeros(n, 3, 3, dtype=f64, device=dev)
        sc, kc, r = self._rrows
        el = ~flag[sc]
        if bool(el.any()):
            s_, k_, r_ = sc[el], kc[el], r[el]
            cr = torch.einsum('fab,fb->fa', T['dL'][s_], r_)
            rM = (torch.einsum('fab,fb->fa', T['Mo'][s_], r_)*r_).sum(1)
            K.index_add_(0, k_, (self.V0[s_]*self.lam)[:, None, None]*(cr[:, :, None]*cr[:, None, :])+rM[:, None, None]*self.I3)
        if self._cut_valid and bool(flag.any()):
            i, j, Lw, key, _, m = self._w
            sc2, kc2, r2_ = self._rows_from(i, j, t['wa'][:m], t['wt'][:m], t['wn'][:m], t['Vi'], self._ml)
            fw = flag[sc2]
            s_, k_, r_ = sc2[fw], kc2[fw], r2_[fw]
            rr = r_[:, :, None]*r_[:, None, :]; rn = (r_*r_).sum(1)
            cr = torch.einsum('fab,fb->fa', t['Q'][s_], r_)
            K.index_add_(0, k_, (self.V0[s_]*self.mu*self.tbe[s_])[:, None, None]*(rn[:, None, None]*self.I3+rr/3)+(self.V0[s_]*self.kap)[:, None, None]*(cr[:, :, None]*cr[:, None, :]))
        ms = self._m_s
        lo, hi = self.tlo, self.thi
        one = torch.ones(ms, dtype=f64, device=dev)
        wa = T['wi'][:ms]*torch.where(flag[lo], T['bi'][:ms], one); wb = T['wj'][:ms]*torch.where(flag[hi], T['bj'][:ms], one)
        kb = (wa+wb)[:, None, None]*self.I3
        K.index_add_(0, lo, kb); K.index_add_(0, hi, kb)
        if self.hold and bool(self._stuck.any()):
            off = self._stuck & (self.tbR > 0) & (self.tbd < .05*self.tbR)               # (the step's blocks have Hertz's stiffness at d: the rest of k_n)
            ng = torch.as_tensor([float(self.ball['ng'][k]) for k in range(3)], dtype=f64, device=dev)
            hz = 2*self.Es*torch.sqrt(self.tbR[off]*self.tbd[off].clamp(min=0.))
            K[off] += ((self.tbkn[off]-hz)/self.th)[:, None, None]*(ng[:, None]*ng[None, :])
        return K


class FlowStep(RingStep):
    """RingStep's Newton, repeated when its solution has an elastic cell past the limit, with those cells flagged from the step's start (up to max_redo). stats: the repeats, the cells flagged by them"""
    max_redo = 5

    def step(self, tol_abs, cg_tol_abs):
        self.begin()
        return self.solve_flagged(tol_abs, cg_tol_abs)

    def solve_flagged(self, tol_abs, cg_tol_abs):
        W = self.W
        evals, its, g2 = self.solve(tol_abs, cg_tol_abs)
        redo = extra = 0
        for _ in range(self.max_redo):
            over = (W.tei > W.ey) & ~W._yld
            if not bool(over.any()):
                break
            extra += W.flag_new(over)
            redo += 1
            e_, i_, g2 = self.solve(tol_abs, cg_tol_abs)
            evals += e_; its += i_
        self.stats['redo'] = redo; self.stats['redo_cells'] = extra
        return evals, its, g2
