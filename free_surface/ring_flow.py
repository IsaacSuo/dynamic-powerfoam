"""The elastic ring and the flowing cell joined (2026-11-01; the user: springs carry the force, and the two are not one or the other). Moduli per unit density, masses = volumes.
Why not a limit on the springs (ring_plastic.py, four forms, none right). A uniform flow balances only if every cell reads one stress P(D); its virial is then P(D) M_i, M_i = sum w q (x) q + Mo. For a potential M_i dP/dD must be symmetric. Elastic dP/dD is constant: no conflict. In flow
dP/dD = (e_y / |D|)(I - D^ (x) D^), symmetric for every direction only if M_i is round, and it is not, cell by cell (only the sum is): so a uniform flow, a potential and a convex cell cannot all hold on the springs' metric. A yield limit needs the metric V_i I, which the divergence-theorem gradient has (merged_ul.py).
  elastic cell  flag 0: RingUL's half springs on the rest q, w = mu A / L each, the outer layer on F_o, R the whole cell's minimum, read at every evaluation; V0 lam / 2 L^2
  flowing cell  flag 1: V0 W(e(F_o)) + V0 kap / 2 L^2 (merged_ul.py's cell, R and C frozen at Newton's start), each half spring r = d - p - G p, one-sided: the part of the relative motion its own cell's gradient does not explain; p = F_o q at the flag, d - kept remainder after that
  the two forms S_i = 1/2 tr(H M_i H^T) + 1/2 sum w |r|^2 and V_i mu |H|^2 + 1/2 sum w |r|^2 hold one elastic energy on two quadratures: equal on a uniform strain over the whole body, cell by cell by M_i's anisotropy. At a limit of 1 % the flag changes a cell's energy by about (M_i / 2 mu V_i - 1) mu V_i e_y^2, zero in the mean
  flag          at Newton's start on the predictor, when the reading e passes flag_at e_y, and a step whose solution has an elastic cell past e_y is solved again with it flagged (the solve is repeatable until end()): the limit then holds inside the step. One way. A flagged cell under the limit is elastic. No limit: nothing flags, RingUL's lion (no cut needed then: use ring_ul.py)
  remainder     a Maxwell element, not a stiffness to switch. Its rate relaxes at kappa g, g = |D| / e_y the cell's flow rate, so w beta_r r with beta_r = 1 / (1 + kappa (1 / beta - 1)), beta the cell's return factor, frozen at Newton's start as merged_ul.py's. A fluctuation D' at om over a flow of rate g: true excess dissipation mu g |D'_perp|^2 / (g^2 + om^2); with the springs' energy s' times the FE's (s' = 2 at w), yield on the tensor only:
                5 s' / 2 . kappa (1 + x^2) / (kappa^2 + x^2) of it, x = om / g, the 5/4 since only the 4 of 5 deviatoric directions across the flow count. Slow fluctuations: kappa = 5 at w, exact. In steady flow the viscosity is that of a fixed k = w / 5 at kappa 1 (merged_ul.py's 1/6, by trial, reads 0.83). The stiffness never switches: beta = 1 gives the spring itself
  shares        between two flowing cells the wall's spring 2 mu A / L goes by t = h / L (clipped .02 .. .98): the one-sided reading has the cell's gradient in it, and a tiny cell's gradient is its gain; on the rest list halves, as the ring
  cut           as merged_ul.py: x -> G linear within a step on the current cut, F_o = (I + G) F_o,n, L = L_n + tr(C^T G), C frozen. Rest list while a cell of the pair has not flagged, current walls when both have, born at r = 0, gone with their wall
Self contact not here. set_ball, start(), prepare(), update_ul; through FlowStep. ring_flow_check.py its checks, ring_flow_drop.py the landing."""
import torch
import warp as wp

from free_surface.closure_warp import k_Aul, k_F8, k_Gul, k_Gul_adj, k_ball, k_ball_adj, rot_proper
from free_surface.merged_ul import MergedUL, k_cell_shape
from free_surface.ring_plastic import RingStep
from free_surface.ring_ul import RingUL
from free_surface.skin_warp import NSLOT, NT, k_total, k_total_loss
from free_surface.spring_plastic import k_Fs, k_Fs_adj
from free_surface.ul_plastic import k_Aul_adj2, k_Ge, k_Ge_adj, k_Lb, k_gg_cell, k_gg_cell_adj, k_gg_wall, k_gg_wall_adj, k_var, polar_batch

f64 = torch.float64


@wp.kernel(enable_backward=False)
def k_rf_L(G: wp.array(dtype=wp.mat33d), C: wp.array(dtype=wp.mat33d), Ln: wp.array(dtype=wp.float64), V0: wp.array(dtype=wp.float64), kc: wp.array(dtype=wp.float64), L: wp.array(dtype=wp.float64),
           Lb: wp.array(dtype=wp.float64), terms: wp.array2d(dtype=wp.float64)):
    """L = L_n + tr(C^T G), the modulus by the cell's form: lam on the springs (they hold 2 mu of the volumetric), kap on the cell"""
    b = wp.tid()
    Lc = Ln[b]+wp.trace(wp.transpose(C[b])@G[b])
    L[b] = Lc
    wp.atomic_add(terms, 1, b % NSLOT, V0[b]*wp.float64(0.5)*kc[b]*Lc*Lc)
    wp.atomic_add(Lb, b, V0[b]*kc[b]*Lc)


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
    """the elastic cell's outer layer, H = R^T F_o - I, 1/2 tr(H Mo H^T); term 4; onto dE/dF"""
    b = wp.tid()
    if flag[b] != 0:
        return
    H = wp.transpose(Rc[b])@Fc[b]-wp.identity(n=3, dtype=wp.float64)
    Z = H@Mo[b]
    wp.atomic_add(terms, 4, b % NSLOT, wp.float64(0.5)*wp.ddot(Z, H))
    Fb[b] = Fb[b]+Rc[b]@Z


@wp.kernel(enable_backward=False)
def k_rf_half(x: wp.array(dtype=wp.vec3d), G: wp.array(dtype=wp.mat33d), Rc: wp.array(dtype=wp.mat33d), flag: wp.array(dtype=int), ri: wp.array(dtype=int), rj: wp.array(dtype=int), wi: wp.array(dtype=wp.float64), wj: wp.array(dtype=wp.float64), qi: wp.array(dtype=wp.vec3d), qj: wp.array(dtype=wp.vec3d),
              dni: wp.array(dtype=wp.vec3d), dnj: wp.array(dtype=wp.vec3d), bi: wp.array(dtype=wp.float64), bj: wp.array(dtype=wp.float64), th_: wp.float64, sri: wp.array(dtype=wp.vec3d), srj: wp.array(dtype=wp.vec3d), terms: wp.array2d(dtype=wp.float64), x_adj: wp.array(dtype=wp.vec3d), Gb: wp.array(dtype=wp.mat33d)):
    """each half by its cell's form. Elastic: 1/2 w |R^T d - q|^2, term 4. Flowing: the remainder at the step's end r_e = (d - p - G p - (1 - th) dn) / th; th^2 1/2 w b r_e^2 + (1 - th) w dn . (r - dn) + (1 - th^2) 1/2 w dn^2, term 5, as merged_ul.py's spring; its force w ((1 - th) dn + th b r_e), and -force (x) p onto dE/dG"""
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
        wp.atomic_add(terms, 4, e % NSLOT, half*wi[e]*wp.dot(a, a))
        f = f+wi[e]*(Rc[i]@a)
    else:
        p = qi[e]
        dn = dni[e]
        r = d-p-G[i]@p
        re = (r-ca*dn)/th_
        sri[e] = re
        w = wi[e]
        b = bi[e]
        wp.atomic_add(terms, 5, e % NSLOT, th_*th_*half*w*b*wp.dot(re, re)+ca*w*wp.dot(dn, r-dn)+(wp.float64(1.0)-th_*th_)*half*w*wp.dot(dn, dn))
        fi = w*(ca*dn+(th_*b)*re)
        f = f+fi
        wp.atomic_add(Gb, i, -wp.outer(fi, p))
    if flag[j] == 0:
        a = wp.transpose(Rc[j])@d-qj[e]
        wp.atomic_add(terms, 4, e % NSLOT, half*wj[e]*wp.dot(a, a))
        f = f+wj[e]*(Rc[j]@a)
    else:
        p = qj[e]
        dn = dnj[e]
        r = d-p-G[j]@p
        re = (r-ca*dn)/th_
        srj[e] = re
        w = wj[e]
        b = bj[e]
        wp.atomic_add(terms, 5, e % NSLOT, th_*th_*half*w*b*wp.dot(re, re)+ca*w*wp.dot(dn, r-dn)+(wp.float64(1.0)-th_*th_)*half*w*wp.dot(dn, dn))
        fi = w*(ca*dn+(th_*b)*re)
        f = f+fi
        wp.atomic_add(Gb, j, -wp.outer(fi, p))
    wp.atomic_add(x_adj, j, f)
    wp.atomic_add(x_adj, i, -f)


class RingFlow(RingUL):
    """X, r, V0: the foam at rest; pairs, A, c: its walls (geometry.npz); E, nu, sigma_y per unit density; kappa: the remainder's rate over the cell's (5 at the springs' w); flag_at: the predictor's reading, in e_y, that flags a cell"""

    def __init__(self, X, r, V0, pairs, A, c, E, nu, sigma_y, theta=.5, kappa=5., flag_at=.5, **kw):
        super().__init__(X, r, V0, pairs, A, c, E, nu, theta=theta, sigma_y=sigma_y, **kw)
        self.kappa, self.flag_at = float(kappa), float(flag_at)
        self.Wp_half = 0.
        self.n_flagged_end = 0

    def start(self):
        n, dev = self.n, self.dev
        z = lambda k_, dt_: wp.zeros(k_, dtype=dt_, device=dev)
        Ks = self.ul['Kw']
        self.ring = dict(K=Ks, i=z(Ks, wp.int32), j=z(Ks, wp.int32), wi=z(Ks, wp.float64), wj=z(Ks, wp.float64), qi=z(Ks, wp.vec3d), qj=z(Ks, wp.vec3d), dni=z(Ks, wp.vec3d), dnj=z(Ks, wp.vec3d), bi=z(Ks, wp.float64), bj=z(Ks, wp.float64), sri=z(Ks, wp.vec3d), srj=z(Ks, wp.vec3d),
                         Am=z(n, wp.mat33d), Rc=z(n, wp.mat33d), Mo=z(n, wp.mat33d), flag=z(n, wp.int32), kc=z(n, wp.float64), V0f=z(n, wp.float64))
        tt = lambda a_: wp.to_torch(a_, requires_grad=False)
        self.tr = {k_: tt(a_) for k_, a_ in self.ring.items() if k_ != 'K'}
        self.tr['kc'].fill_(self.lam)
        self._qkey = self.tkey.clone()
        D = self.X[self.thi]-self.X[self.tlo]
        self._qi, self._qj = D.clone(), D.clone()
        zv = torch.zeros_like(D)
        self._dli, self._dlj = zv.clone(), zv.clone()
        self.update_ul(self.X)

    def update_ul(self, x):
        """the step's start at x: the cut, the spring list, the gradient (MergedUL.update_ul; self._Fe the running F_o); the half springs by their pair: an elastic cell's q, a flowing cell's p = d - its kept remainder; the shares; the forms' moduli"""
        self._xs = x.detach().clone(); x = self._xs
        MergedUL.update_ul(self, x)
        dev, t, T, yld = self.dev, self.tul, self.tr, self._yld
        i_w, j_w, Lw, _, _, m = self._w
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
        # the shares between two flowing cells: by the wall's place on the pair
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
        Mc = t['Mc']
        T['Mo'].copy_(self.mu*(Mc+Mc.transpose(1, 2))); T['Mo'][yld] = 0.
        T['flag'].copy_(yld.to(torch.int32))
        T['kc'].copy_(torch.where(yld, torch.full_like(self.V0, self.kap), torch.full_like(self.V0, self.lam)))
        T['V0f'].copy_(torch.where(yld, self.V0, torch.zeros_like(self.V0)))
        bw = yld[i_w] & yld[j_w]
        t['wv'][:m] = torch.where(bw, torch.full_like(Lw, self.kap), torch.full_like(Lw, self.lam))*t['wa'][:m]*Lw/24

    def flag_new(self, mask):
        """these cells go to the flowing form from the step's start on: their halves' p = F_o q, dn = d_n - p; the outer layer into the cell. Number flagged"""
        new = mask & ~self._yld
        k = int(new.sum())
        if k == 0:
            return 0
        T, ms = self.tr, self._m_s
        lo, hi = self.tlo, self.thi
        F0 = self.tsp['F0']
        d = self._xs[hi]-self._xs[lo]
        a, b = new[lo], new[hi]
        pa = torch.einsum('fab,fb->fa', F0[lo[a]], T['qi'][:ms][a]); pb = torch.einsum('fab,fb->fa', F0[hi[b]], T['qj'][:ms][b])
        qi_, qj_, dni_, dnj_ = T['qi'][:ms], T['qj'][:ms], T['dni'][:ms], T['dnj'][:ms]
        qi_[a] = pa; dni_[a] = d[a]-pa
        qj_[b] = pb; dnj_[b] = d[b]-pb
        self._dli[a] = d[a]-pa; self._dlj[b] = d[b]-pb
        self._yld = self._yld | new
        T['Mo'][new] = 0.; T['flag'][new] = 1; T['kc'][new] = self.kap; T['V0f'][new] = self.V0[new]
        return k

    def run(self):
        """E and dE/dx at xw: the gradient and F_o; the elastic cells' rotations; the cells; the halves; back through the gradient"""
        B, U, Rg, S, n, Kw = self.B, self.ul, self.ring, self.sp, self.n, self.ul['Kw']
        x, g = self.xw, self.xw.grad
        for k_ in ('G0', 'Ga', 'Sg', 'Lb', 'Gbe', 'Gb0', 'terms', 'tsum'):
            B[k_].zero_()
        Rg['Am'].zero_()
        g.zero_()
        th_ = wp.float64(self.th)
        wp.launch(k_Gul, 2*Kw, inputs=[x, U['xn'], U['S'], U['O'], U['R']], outputs=[B['G0']])
        wp.launch(k_Aul, Kw, inputs=[B['G0'], U['As'], U['Ao'], U['Aw']], outputs=[B['Ga']])
        wp.launch(k_Ge, n, inputs=[B['G0'], B['Ga'], U['used']], outputs=[B['Ge']])
        wall = [U['wi'], U['wj'], U['wa'], U['wn'], U['wt'], U['wtau']]
        wp.launch(k_gg_wall, Kw, inputs=[x, U['xn']]+wall+[B['Ge']], outputs=[B['Sg']])
        wp.launch(k_gg_cell, n, inputs=[x, U['xn'], B['Sg'], B['Ge'], U['a'], U['Mc'], U['Vi']], outputs=[B['G']])
        wp.launch(k_Fs, n, inputs=[B['G'], S['F0']], outputs=[B['Fc']])
        wp.launch(k_rf_L, n, inputs=[B['G'], U['Q'], U['Ln'], U['V0'], Rg['kc']], outputs=[B['L'], B['Lb'], B['terms']])
        wp.launch(k_var, Kw, inputs=[U['wi'], U['wj'], U['wv'], B['L']], outputs=[B['Lb'], B['terms']])
        wp.launch(k_rf_A, Rg['K'], inputs=[x, Rg['i'], Rg['j'], Rg['wi'], Rg['wj'], Rg['qi'], Rg['qj']], outputs=[Rg['Am']])
        wp.launch(k_rf_R, n, inputs=[Rg['Am'], B['Fc'], Rg['Mo'], Rg['flag']], outputs=[Rg['Rc']])
        wp.launch(k_cell_shape, n, inputs=[B['Fc'], S['Rh'], S['en'], Rg['V0f'], wp.float64(self.mu), wp.float64(self.sy), th_], outputs=[B['ei'], B['be'], B['er'], B['Fb'], B['terms']])
        wp.launch(k_rf_outer, n, inputs=[B['Fc'], Rg['Rc'], Rg['Mo'], Rg['flag']], outputs=[B['Fb'], B['terms']])
        wp.launch(k_Fs_adj, n, inputs=[B['Fb'], S['F0']], outputs=[B['Gb']])
        wp.launch(k_rf_half, Rg['K'], inputs=[x, B['G'], Rg['Rc'], Rg['flag'], Rg['i'], Rg['j'], Rg['wi'], Rg['wj'], Rg['qi'], Rg['qj'], Rg['dni'], Rg['dnj'], Rg['bi'], Rg['bj'], th_], outputs=[Rg['sri'], Rg['srj'], B['terms'], g, B['Gb']])
        wp.launch(k_Lb, n, inputs=[B['Lb'], U['Q']], outputs=[B['Gb']])
        bl = self.ball
        if bl is not None:
            wp.launch(k_ball, bl['k'], inputs=[x, bl['body'], bl['R'], bl['xn'], bl['ng'], bl['par'], bl['kt'], bl['bound'], bl['lamt']], outputs=[B['bdelta'], B['bfn'], B['bff'], B['terms']])
            wp.launch(k_ball_adj, bl['k'], inputs=[bl['body'], bl['ng'], bl['par'], B['bfn'], B['bff'], self.tones], outputs=[g])
        wp.launch(k_gg_cell_adj, n, inputs=[B['Gb'], U['a'], U['Mc'], U['Vi']], outputs=[g, B['Gs'], B['Gbe']])
        wp.launch(k_gg_wall_adj, Kw, inputs=wall+[B['Gs']], outputs=[g, B['Gbe']])
        wp.launch(k_Ge_adj, n, inputs=[B['Gbe'], U['used']], outputs=[B['Gb0']])
        wp.launch(k_Aul_adj2, Kw, inputs=[B['Gbe'], U['As'], U['Ao'], U['Aw']], outputs=[B['Gb0']])
        wp.launch(k_Gul_adj, 2*Kw, inputs=[B['Gb0'], U['S'], U['O'], U['R']], outputs=[g])
        wp.launch(k_total, (NT, 256), inputs=[B['terms']], outputs=[B['tsum']])
        wp.launch(k_total_loss, 1, inputs=[B['tsum'], self.tw], outputs=[B['loss']])

    def reread(self, beta=False):
        """R and C at the point the solve stands on; beta: Newton's start: the predictor's reading flags cells, the halves' beta_r"""
        MergedUL.reread(self, beta=False)
        if beta:
            self.flag_new(self.tei > self.flag_at*self.ey)
            T, ms, lo, hi = self.tr, self._m_s, self.tlo, self.thi
            br = 1/(1+self.kappa*(1/self.tbe.clamp(min=1e-6)-1))
            T['bi'][:ms] = br[lo]; T['bj'][:ms] = br[hi]

    def end_ul(self, th=.5):
        """after the step (the buffers at the accepted z; x -> F linear, so the end's F and G are exact): the flowing halves' remainders, the cells' returns, F_o, L; the plastic work"""
        assert th == self.th, 'the model and the step on one theta'
        t, s, T, n, I3, ms = self.tul, self.tsp, self.tr, self.n, self.I3, self._m_s
        lo, hi = self.tlo, self.thi
        fl, fh = self._yld[lo], self._yld[hi]
        wi, wj, bi, bj, sri, srj, dni, dnj = T['wi'][:ms], T['wj'][:ms], T['bi'][:ms], T['bj'][:ms], T['sri'][:ms], T['srj'][:ms], T['dni'][:ms], T['dnj'][:ms]
        if th == 1.:
            wa = .5*wi*bi*(1-bi)*(sri*sri).sum(1); wb = .5*wj*bj*(1-bj)*(srj*srj).sum(1)
        else:
            wa = .5*wi*(1-bi)*((sri*dni).sum(1)+bi*(sri*sri).sum(1)); wb = .5*wj*(1-bj)*((srj*dnj).sum(1)+bj*(srj*srj).sum(1))
        wh = float(wa[fl].sum()+wb[fh].sum())
        self.Wp_half += wh; self.Wp += wh
        self._dli = torch.where(fl[:, None], bi[:, None]*sri, self._dli); self._dlj = torch.where(fh[:, None], bj[:, None]*srj, self._dlj)
        be = self.tbe.clone(); flag = self._yld
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
        Q = polar_batch(I3+self.tG/th)
        self.ep = Q@self.ep@Q.transpose(1, 2)
        if bool(yn.any()):
            A_ = R[yn].transpose(1, 2)@Fe[yn]
            w, v = torch.linalg.eigh((.5*(A_+A_.transpose(1, 2))).cpu())           # (on the CPU, as merged_ul.py)
            w, v = w.to(A_.device), v.to(A_.device)
            lw = torch.log(w.clamp(min=1e-6)); lm = lw.mean(1, keepdim=True); dv = lw-lm
            nr = dv.norm(dim=1, keepdim=True).clamp(min=1e-300); a = torch.exp(lm)
            sc = torch.minimum(torch.ones_like(nr), (self.ey/a)/nr)
            Ue = v@torch.diag_embed(torch.exp(lm+sc*dv))@v.transpose(1, 2)
            Fe[yn] = R[yn]@Ue
            self.ep[yn] = self.ep[yn]+R[yn]@(v@torch.diag_embed((1-sc)*dv)@v.transpose(1, 2))@R[yn].transpose(1, 2)
        self._Fe = Fe
        # (an elastic cell still past the limit after the step's repeats: flagged here, its spring state read at the step's end, its stress kept for one step)
        over = (be < 1) & ~flag
        self.n_flagged_end = int(over.sum())
        if bool(over.any()):
            xe = (self.tx-(1-th)*self._xs)/th
            d = xe[hi]-xe[lo]
            a, b = over[lo], over[hi]
            self._dli[a] = d[a]-torch.einsum('fab,fb->fa', Fe[lo[a]], T['qi'][:ms][a]); self._dlj[b] = d[b]-torch.einsum('fab,fb->fa', Fe[hi[b]], T['qj'][:ms][b])
            self._yld = self._yld | over
        self.ul_stats['flips'] = int((torch.linalg.det(s['Fend']) <= 1e-3).sum()); self.ul_stats['yielding'] = int(yn.sum()); self.ul_stats['yielded'] = int(self._yld.sum())
        t['Ln'].add_((t['Q']*self.tG).sum((1, 2))/th)

    def stored(self):
        """the elastic energy of the state after the last end_ul, at the step's end x: shape (springs and outer layer on the elastic cells, the cell and the kept remainders on the flowing), volume"""
        T, ms, lo, hi = self.tr, self._m_s, self.tlo, self.thi
        Fe, flag, t = self._Fe, self._yld, self.tul
        x = self.tx if self.th == 1. else (self.tx-(1-self.th)*self._xs)/self.th
        d = x[hi]-x[lo]
        qi, qj, wi, wj = self._qi, self._qj, T['wi'][:ms], T['wj'][:ms]
        Mo = T['Mo']
        A = (Fe@Mo).index_add(0, lo, wi[:, None, None]*d[:, :, None]*qi[:, None, :]).index_add(0, hi, wj[:, None, None]*d[:, :, None]*qj[:, None, :])
        R = self.I3.expand(self.n, 3, 3).clone()
        if not bool(flag.all()):
            Us, _, Vh = torch.linalg.svd(A[~flag].cpu())                                 # (log only; not polar_batch, whose det floor reads I on a cell of anisotropic M)
            Sg = torch.diag_embed(torch.stack([torch.ones(len(Us), dtype=f64), torch.ones(len(Us), dtype=f64), torch.linalg.det(Us@Vh)], 1))
            R[~flag] = (Us@Sg@Vh).to(self.dev)
        fl, fh = flag[lo], flag[hi]
        mi = torch.einsum('fba,fb->fa', R[lo], d)-qi; mj = torch.einsum('fba,fb->fa', R[hi], d)-qj
        Es = float((.5*wi*(mi*mi).sum(1))[~fl].sum()+(.5*wj*(mj*mj).sum(1))[~fh].sum())
        H = R.transpose(1, 2)@Fe-self.I3
        Eo = float((.5*((H@Mo)*H).sum((1, 2)))[~flag].sum())
        Eh = float((.5*wi*(self._dli*self._dli).sum(1))[fl].sum()+(.5*wj*(self._dlj*self._dlj).sum(1))[fh].sum())
        Rp = polar_batch(Fe)
        A_ = Rp.transpose(1, 2)@Fe; e_ = .5*(A_+A_.transpose(1, 2))-self.I3
        e_ = e_-(e_.diagonal(dim1=1, dim2=2).sum(1)/3)[:, None, None]*self.I3
        Ec = float((self.V0*self.mu*(e_*e_).sum((1, 2)))[flag].sum())
        L = t['Ln']
        Ev = float((self.V0*.5*T['kc']*L*L).sum())
        wv, i_w, j_w, m = t['wv'], self._w[0], self._w[1], self._w[5]
        Ev += float((wv[:m]*(L[i_w]-L[j_w])**2).sum())
        return Es+Eo+Eh+Ec, Ev

    def ul_blocks(self, lr_thr=None, lr_cap=0):
        """the diagonal blocks at the last evaluation: through the gradient's direct rows the flowing cell's shape at its return factor, the pressure, the elastic cell's outer layer; the halves' w, a flowing one's at beta_r, on each end"""
        assert lr_thr is None, 'no low-rank term here'
        n, t, dev, T, s = self.n, self.tul, self.dev, self.tr, self.tsp
        i, j, Lw, key, _, m = self._w; ml = self._ml
        A, tw, nn, Vi = t['wa'][:m], t['wt'][:m], t['wn'][:m], t['Vi']
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
        r = torch.zeros(len(u), 3, dtype=f64, device=dev).index_add(0, inv, ra); sc, kc = u//n, u % n
        rr = r[:, :, None]*r[:, None, :]; r2 = (r*r).sum(1)
        cr = torch.einsum('fab,fb->fa', t['Q'][sc], r)
        rF = torch.einsum('fba,fb->fa', s['F0'][sc], r)
        rM = (torch.einsum('fab,fb->fa', T['Mo'][sc], rF)*rF).sum(1)
        K = torch.zeros(n, 3, 3, dtype=f64, device=dev)
        K.index_add_(0, kc, (T['V0f'][sc]*self.mu*self.tbe[sc])[:, None, None]*(r2[:, None, None]*self.I3+rr/3)+(self.V0[sc]*T['kc'][sc])[:, None, None]*(cr[:, :, None]*cr[:, None, :])+rM[:, None, None]*self.I3)
        ms = self._m_s
        lo, hi = self.tlo, self.thi
        wa = T['wi'][:ms]*torch.where(self._yld[lo], T['bi'][:ms], torch.ones(ms, dtype=f64, device=dev)); wb = T['wj'][:ms]*torch.where(self._yld[hi], T['bj'][:ms], torch.ones(ms, dtype=f64, device=dev))
        kb = (wa+wb)[:, None, None]*self.I3
        K.index_add_(0, lo, kb); K.index_add_(0, hi, kb)
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
