"""ring_ul.py with a yield limit (2026-11-01). One formula with the elastic end: beta = 1 is RingUL's force.
  reading   per half spring in its cell's frame m = R^T d - q; the outer layer's H = R^T F_o - I, its virtual springs. R the whole cell's minimum of the step's start state.
  split     the cell's spring energy S = 1/2 sum w |m|^2 + 1/2 tr(H Mo H^T) splits exactly: v = (sum w m . q + tr(H Mo)) / T, T = sum w |q|^2 + tr Mo, a volumetric part 1/2 T v^2 and a distortion W = S - 1/2 T v^2 (m - v q orthogonal to q in that sum).
  limit     rho^2 = W / (mu V_s) <= e_y^2, beta = min(1, e_y / rho), V_s = T / (6 mu) the volume the cell's half springs stand for: each carries half of its wall's two pyramids, so sum V_s = sum V on a closed foam, cell by cell not (with the cell's own V a small cell beside large ones read a uniform strain up to 2.29 times, 1 % 0.85). So rho^2 = 3 tr(e M e) / tr M on a uniform strain,
            with only M's anisotropy in it. The cell's energy E = psi(W) + 1/2 T v^2, psi = W up to W_y = mu V_s e_y^2, then 2 (W W_y)^1/2 - W_y.
  force     w R (beta m + (1 - beta) v q) on the half spring, R (beta H + (1 - beta) v I) Mo on F_o: the exact gradient of E at any increment. E stays stationary in R (the torque of the returned force is beta times the elastic one, 0), so nothing for dR, and R need not be read again after the return. The step is symmetric, and convex in x at a frozen R
            (2026-11-01, first form: the limit on |dev U| of the fit F + 3/2 of the remainder, exact on a uniform flow, ring_checks.py --test 1, 6, 12, but its force is a gradient only to first order in 1 - beta, and in the landing's first impact beta reached 0.01: Newton's ratio -99 to 120, stuck at 0.03 weights. That form's 3/2 is not integrable.)
  return    at the step's end, from the step's start state: q -> q + (1 - beta)(m - v q), F_o -> R (I + beta H + (1 - beta) v I). The plastic work s_y de = 2 beta (1 - beta) W: what the return lets go, (1 - beta^2) W, less the trial's overshoot
  price     W in the springs' metric: under a uniform strain rho varies with the direction of e against M, so a uniform flow is not exactly uniform: the mean stress about 1 - Var(3 tr(e M e) / (tr M |e|^2)) / 8 of the limit. The remainder counts once, where its FE share is half: the excess dissipation of a rough motion 1.3-1.8 of the FE's
  cut       the gradient on the current cut, as merged_ul.py: x -> G linear within a step, F_o = (I + G) F_o,n, L = L_n + tr(C^T G), C frozen at Newton's start. Springs at their rest weight while a cell of the pair has not yielded, at the current wall's when both have; gone with their wall, born at r = 0, q = R^T d / (1 + v).
Self contact not here. set_ball, start(), prepare(); through RingStep. ring_plastic_check.py its checks, ring_drop.py the landing."""
import math

import torch
import warp as wp

from free_surface.closure_warp import PS_GQ2, PS_PQ, PS_PZ, k_Aul, k_Gul, k_Gul_adj, k_ball, k_ball_adj, k_mid_ball, k_mid_begin, k_mid_inv, k_mid_pre, k_mid_pre_ball, rot_proper
from free_surface.merged_ul import MergedStep, MergedUL, k_Lc
from free_surface.ring_ul import RingUL, k_ring_A
from free_surface.skin_warp import NSLOT, k_total, k_total_loss
from free_surface.spring_plastic import k_Fs, k_Fs_adj
from free_surface.ul_plastic import k_Aul_adj2, k_Ge, k_Ge_adj, k_Lb, k_gg_cell, k_gg_cell_adj, k_gg_wall, k_gg_wall_adj, k_var

f64 = torch.float64


@wp.kernel(enable_backward=False)
def k_rp_R(Am: wp.array(dtype=wp.mat33d), Fc: wp.array(dtype=wp.mat33d), Mo: wp.array(dtype=wp.mat33d), Rc: wp.array(dtype=wp.mat33d)):
    b = wp.tid()
    Rc[b] = rot_proper(Am[b]+Fc[b]@Mo[b])


@wp.kernel(enable_backward=False)
def k_rp_m(x: wp.array(dtype=wp.vec3d), ri: wp.array(dtype=int), rj: wp.array(dtype=int), rw: wp.array(dtype=wp.float64), qi: wp.array(dtype=wp.vec3d), qj: wp.array(dtype=wp.vec3d), Rc: wp.array(dtype=wp.mat33d),
           mi: wp.array(dtype=wp.vec3d), mj: wp.array(dtype=wp.vec3d), Sq: wp.array(dtype=wp.float64), Sm: wp.array(dtype=wp.float64)):
    """each half's reading; per cell sum w m . q, sum w |m|^2"""
    e = wp.tid()
    i, j = ri[e], rj[e]
    if i == j:
        return
    d = x[j]-x[i]
    w = rw[e]
    a = wp.transpose(Rc[i])@d-qi[e]
    b = wp.transpose(Rc[j])@d-qj[e]
    mi[e] = a
    mj[e] = b
    wp.atomic_add(Sq, i, w*wp.dot(a, qi[e]))
    wp.atomic_add(Sm, i, w*wp.dot(a, a))
    wp.atomic_add(Sq, j, w*wp.dot(b, qj[e]))
    wp.atomic_add(Sm, j, w*wp.dot(b, b))


@wp.kernel(enable_backward=False)
def k_rp_cell(Fc: wp.array(dtype=wp.mat33d), Rc: wp.array(dtype=wp.mat33d), Mo: wp.array(dtype=wp.mat33d), Tm: wp.array(dtype=wp.float64), Sq: wp.array(dtype=wp.float64), Sm: wp.array(dtype=wp.float64), ey: wp.float64,
              Hr: wp.array(dtype=wp.mat33d), vc: wp.array(dtype=wp.float64), Wd: wp.array(dtype=wp.float64), rho: wp.array(dtype=wp.float64), be: wp.array(dtype=wp.float64), Fb: wp.array(dtype=wp.mat33d), terms: wp.array2d(dtype=wp.float64)):
    """the split, the limit, the cell's energy (term 0) and dE/dF_o"""
    b = wp.tid()
    half = wp.float64(0.5)
    one = wp.float64(1.0)
    I = wp.identity(n=3, dtype=wp.float64)
    H = wp.transpose(Rc[b])@Fc[b]-I
    Hr[b] = H
    Z = H@Mo[b]
    v = (Sq[b]+wp.trace(Z))/Tm[b]
    W = wp.max(half*(Sm[b]+wp.ddot(Z, H)-v*v*Tm[b]), wp.float64(0.0))
    r = wp.sqrt(wp.float64(6.0)*W/Tm[b])
    bb = one
    E = W
    if r > ey:
        bb = ey/r
        Wy = ey*ey*Tm[b]/wp.float64(6.0)
        E = wp.float64(2.0)*wp.sqrt(W*Wy)-Wy
    vc[b] = v
    Wd[b] = W
    rho[b] = r
    be[b] = bb
    wp.atomic_add(terms, 0, b % NSLOT, E+half*Tm[b]*v*v)
    Fb[b] = Rc[b]@((H*bb+I*((one-bb)*v))@Mo[b])


@wp.kernel(enable_backward=False)
def k_rp_force(ri: wp.array(dtype=int), rj: wp.array(dtype=int), rw: wp.array(dtype=wp.float64), qi: wp.array(dtype=wp.vec3d), qj: wp.array(dtype=wp.vec3d), mi: wp.array(dtype=wp.vec3d), mj: wp.array(dtype=wp.vec3d), Rc: wp.array(dtype=wp.mat33d),
               be: wp.array(dtype=wp.float64), vc: wp.array(dtype=wp.float64), x_adj: wp.array(dtype=wp.vec3d)):
    e = wp.tid()
    i, j = ri[e], rj[e]
    if i == j:
        return
    one = wp.float64(1.0)
    fi = be[i]*mi[e]+((one-be[i])*vc[i])*qi[e]
    fj = be[j]*mj[e]+((one-be[j])*vc[j])*qj[e]
    f = rw[e]*(Rc[i]@fi+Rc[j]@fj)
    wp.atomic_add(x_adj, j, f)
    wp.atomic_add(x_adj, i, -f)


class RingStep(MergedStep):
    """MergedStep with the residual as a second way to take a trial, down to 0.9 of itself: near the solution Pi's change is rounding (the frozen C's volume term is the gradient of its own energy, but the ground's end-of-step balls and the variance add their round-off). The solution is g = 0; Pi only guards the way there"""

    def solve(self, tol_abs, cg_tol_abs):
        W, A, bl, n = self.W, self.a, self.W.ball, self.W.n
        wp.launch(k_mid_begin, n, inputs=[A['x'], A['v'], A['a'], self.dt, self.th], outputs=[A['xt'], A['z']])
        if self.smooth_pred and self._vp is not None:
            h_ = self.th*self.dt
            self.tz.copy_(self.tx+.5*h_*(self.tv+self._vp)+.5*(.5*h_*self.dt+h_*h_)*(self.ta+self._ap))
        wp.launch(k_mid_ball, bl['k'], inputs=[A['x'], bl['body'], bl['R'], bl['ng'], self.c0, self.g8, self.cb], outputs=[bl['xn'], bl['kt'], bl['bound']])
        if W.stick:
            W.tbbound.copy_(torch.minimum(W.tbbound, W.tau_y*W.footprint(self.tx, W._ulr)))
        self._eval(0.)
        W.reread(beta=True)
        g2 = self._eval(0.)[PS_GQ2]
        self._accept()
        evals, its = 2, 0
        st = dict(rejected=0, edge=0, negative=0, trace=[])
        delta, delta0 = (self.delta_start if self.carry_delta and self.delta_start else -1.), None
        while its < self.newton_max:
            if math.sqrt(g2) <= tol_abs:
                break
            its += 1
            if its == 1 and self.pre_ul and self._pre_age <= 0:
                self._pre_age = self.pre_every
                self.tK.copy_(W.ul_blocks())
            wp.launch(k_mid_pre, n, inputs=[W.B['R'], A['K'], self.c1, A['cm']], outputs=[A['P']])
            wp.launch(k_mid_pre_ball, bl['k'], inputs=[bl['body'], bl['R'], W.B['bdelta'], bl['ng'], self.e4, bl['kt'], bl['bound'], self.muf], outputs=[A['P']])
            wp.launch(k_mid_inv, n, inputs=[A['P']], outputs=[A['Pinv']])
            stop = max(cg_tol_abs(g2), self.eta*math.sqrt(g2)); g_ = math.sqrt(g2)
            r = self.CG.solve(A['gz'], delta, stop, self.cg_cap)
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
            take = ok and (rho > 1e-4 or s[PS_GQ2] < .81*g2)
            pM = delta if r['EDGE'] else math.sqrt(max(r['PMP'], 0.))
            if not take or rho < .25 or pred <= noise:
                delta = .25*pM
            elif rho > .75 and r['EDGE']:
                delta = 2*delta
            st['trace'].append((r['K'], r['EDGE'], r['NEG'], stop, g_, rho, take))
            if take:
                self._accept()
                g2 = s[PS_GQ2]
            else:
                st['rejected'] += 1
                g2 = self._eval(0.)[PS_GQ2]; evals += 1
                self._accept()
                if not delta > 1e-12*delta0:
                    break
        self.stats = st
        self.delta_start = delta if delta > 0. else None
        return evals, its, g2


class RingPlastic(RingUL):
    """X, r, V0: the foam at rest; pairs, A, c: its walls (geometry.npz); E, nu, sigma_y per unit density"""

    def __init__(self, X, r, V0, pairs, A, c, E, nu, sigma_y, theta=.5, **kw):
        super().__init__(X, r, V0, pairs, A, c, E, nu, theta=theta, sigma_y=sigma_y, **kw)

    def start(self):
        n, dev = self.n, self.dev
        z = lambda k_, dt_: wp.zeros(k_, dtype=dt_, device=dev)
        Ks = self.ul['Kw']
        self.ring = dict(K=Ks, i=z(Ks, wp.int32), j=z(Ks, wp.int32), w=z(Ks, wp.float64), qi=z(Ks, wp.vec3d), qj=z(Ks, wp.vec3d), mi=z(Ks, wp.vec3d), mj=z(Ks, wp.vec3d),
                         Am=z(n, wp.mat33d), Rc=z(n, wp.mat33d), Mo=z(n, wp.mat33d), Tm=z(n, wp.float64), Sq=z(n, wp.float64), Sm=z(n, wp.float64), Hr=z(n, wp.mat33d), vc=z(n, wp.float64), Wd=z(n, wp.float64), rho=z(n, wp.float64))
        tt = lambda a_: wp.to_torch(a_, requires_grad=False)
        self.tr = {k_: tt(a_) for k_, a_ in self.ring.items() if k_ != 'K'}
        self._qkey = self.tkey.clone()
        D = self.X[self.thi]-self.X[self.tlo]
        self._qi, self._qj = D.clone(), D.clone()
        self._Re = self.I3.expand(n, 3, 3).clone(); self._ve = torch.zeros(n, dtype=f64, device=dev)
        self.store = dict(springs=0., vol=0., var=0., outer=0.)
        self.update_ul(self.X)

    def update_ul(self, x):
        """the step's start at x: the cut, the spring list, the gradient (MergedUL.update_ul; self._Fe the outer layer's F_o); the half springs by their pair, a new one with no remainder; Mo, T"""
        self._xs = x.detach().clone(); x = self._xs
        MergedUL.update_ul(self, x)
        dev, t, T = self.dev, self.tul, self.tr
        m = self._w[5]; Lw = self._w[2]
        t['wv'][:m] = self.lam*t['wa'][:m]*Lw/24                                       # (the springs hold 2 mu of the volumetric: lam here)
        lo, hi, key, ms = self.tlo, self.thi, self.tkey, self._m_s
        pos = torch.searchsorted(self._qkey, key).clamp(max=len(self._qkey)-1); kept = self._qkey[pos] == key
        qi = torch.zeros(ms, 3, dtype=f64, device=dev); qj = torch.zeros(ms, 3, dtype=f64, device=dev)
        qi[kept] = self._qi[pos[kept]]; qj[kept] = self._qj[pos[kept]]
        if not bool(kept.all()):
            nw = ~kept
            d = x[hi[nw]]-x[lo[nw]]
            qi[nw] = torch.einsum('fba,fb->fa', self._Re[lo[nw]], d)/(1+self._ve[lo[nw]])[:, None]; qj[nw] = torch.einsum('fba,fb->fa', self._Re[hi[nw]], d)/(1+self._ve[hi[nw]])[:, None]
        self._qkey, self._qi, self._qj = key.clone(), qi, qj
        w = self.tk0/(2*self.kshare)
        for k_ in ('i', 'j', 'w', 'qi', 'qj'):
            T[k_].zero_()
        T['i'][:ms] = lo.to(torch.int32); T['j'][:ms] = hi.to(torch.int32); T['w'][:ms] = w; T['qi'][:ms] = qi; T['qj'][:ms] = qj
        Mc = t['Mc']
        T['Mo'].copy_(self.mu*(Mc+Mc.transpose(1, 2)))
        Tm = T['Mo'].diagonal(dim1=1, dim2=2).sum(1).clone()
        Tm.index_add_(0, lo, w*(qi*qi).sum(1)); Tm.index_add_(0, hi, w*(qj*qj).sum(1))
        T['Tm'].copy_(Tm)

    def run(self):
        """E and dE/dx at xw: the gradient; R; the readings; the split, the limit, the force; back through the gradient"""
        B, U, Rg, S, n, Kw = self.B, self.ul, self.ring, self.sp, self.n, self.ul['Kw']
        x, g = self.xw, self.xw.grad
        for k_ in ('G0', 'Ga', 'Sg', 'Lb', 'Gbe', 'Gb0', 'terms', 'tsum'):
            B[k_].zero_()
        Rg['Am'].zero_(); Rg['Sq'].zero_(); Rg['Sm'].zero_()
        g.zero_()
        wp.launch(k_Gul, 2*Kw, inputs=[x, U['xn'], U['S'], U['O'], U['R']], outputs=[B['G0']])
        wp.launch(k_Aul, Kw, inputs=[B['G0'], U['As'], U['Ao'], U['Aw']], outputs=[B['Ga']])
        wp.launch(k_Ge, n, inputs=[B['G0'], B['Ga'], U['used']], outputs=[B['Ge']])
        wall = [U['wi'], U['wj'], U['wa'], U['wn'], U['wt'], U['wtau']]
        wp.launch(k_gg_wall, Kw, inputs=[x, U['xn']]+wall+[B['Ge']], outputs=[B['Sg']])
        wp.launch(k_gg_cell, n, inputs=[x, U['xn'], B['Sg'], B['Ge'], U['a'], U['Mc'], U['Vi']], outputs=[B['G']])
        wp.launch(k_Fs, n, inputs=[B['G'], S['F0']], outputs=[B['Fc']])
        wp.launch(k_ring_A, Rg['K'], inputs=[x, Rg['i'], Rg['j'], Rg['w'], Rg['qi'], Rg['qj']], outputs=[Rg['Am']])
        wp.launch(k_rp_R, n, inputs=[Rg['Am'], B['Fc'], Rg['Mo']], outputs=[Rg['Rc']])
        wp.launch(k_rp_m, Rg['K'], inputs=[x, Rg['i'], Rg['j'], Rg['w'], Rg['qi'], Rg['qj'], Rg['Rc']], outputs=[Rg['mi'], Rg['mj'], Rg['Sq'], Rg['Sm']])
        wp.launch(k_rp_cell, n, inputs=[B['Fc'], Rg['Rc'], Rg['Mo'], Rg['Tm'], Rg['Sq'], Rg['Sm'], wp.float64(self.ey)], outputs=[Rg['Hr'], Rg['vc'], Rg['Wd'], Rg['rho'], B['be'], B['Fb'], B['terms']])
        wp.launch(k_rp_force, Rg['K'], inputs=[Rg['i'], Rg['j'], Rg['w'], Rg['qi'], Rg['qj'], Rg['mi'], Rg['mj'], Rg['Rc'], B['be'], Rg['vc']], outputs=[g])
        wp.launch(k_Fs_adj, n, inputs=[B['Fb'], S['F0']], outputs=[B['Gb']])
        wp.launch(k_Lc, n, inputs=[B['G'], U['Q'], U['Ln'], U['V0'], wp.float64(self.lam)], outputs=[B['L'], B['Lb'], B['terms']])
        wp.launch(k_var, Kw, inputs=[U['wi'], U['wj'], U['wv'], B['L']], outputs=[B['Lb'], B['terms']])
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
        wp.launch(k_total, (9, 256), inputs=[B['terms']], outputs=[B['tsum']])
        wp.launch(k_total_loss, 1, inputs=[B['tsum'], self.tw], outputs=[B['loss']])

    def end_ul(self, th=.5):
        """at the accepted z: the state read at the step's end x_e = x_n + (z - x_n) / th from the step's start state, and returned; the ground's buffers kept as they are at z"""
        assert th == self.th, 'the model and the step on one theta'
        t, T, B = self.tul, self.tr, self.B
        z = self.tx.clone()
        bk = {k_: wp.to_torch(B[k_], requires_grad=False).clone() for k_ in ('bff', 'bdelta', 'bfn')}
        trG = (t['Q']*self.tG).sum((1, 2))/th
        self.grad((z-(1-th)*self._xs)/th)
        ms = self._m_s
        be = self.tbe.clone(); yn = be < 1
        v, W, R, H = T['vc'], T['Wd'], T['Rc'], T['Hr']
        self.Wp += float((2*be*(1-be)*W).sum())
        lo, hi = self.tlo, self.thi
        qi, qj, mi, mj = T['qi'][:ms], T['qj'][:ms], T['mi'][:ms], T['mj'][:ms]
        self._qi = qi+(1-be[lo])[:, None]*(mi-v[lo][:, None]*qi)
        self._qj = qj+(1-be[hi])[:, None]*(mj-v[hi][:, None]*qj)
        Hn = be[:, None, None]*H+((1-be)*v)[:, None, None]*self.I3
        self._Fe = R@(self.I3+Hn)
        self._Re, self._ve = R.clone(), v.clone()
        self.store = dict(springs=float((be*be*W+.5*T['Tm']*v*v).sum()), vol=float(self.tsum[1]), var=float(self.tsum[2]), outer=0.)
        self._yld = self._yld | yn
        t['Ln'].add_(trG)
        self.ul_stats['flips'] = int((torch.linalg.det(self._Fe) <= 0).sum()); self.ul_stats['yielding'] = int(yn.sum()); self.ul_stats['yielded'] = int(self._yld.sum())
        self.spring_store = 0.
        for k_, a_ in bk.items():
            wp.to_torch(B[k_], requires_grad=False).copy_(a_)

    def ul_blocks(self, lr_thr=None, lr_cap=0):
        """the diagonal blocks at the last evaluation: the springs' w (beta_i + beta_j) I on each end, the outer layer and the pressure through the gradient's direct rows, on this step's cut"""
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
        cr = torch.einsum('fab,fb->fa', t['Q'][sc], r)
        rF = torch.einsum('fba,fb->fa', s['F0'][sc], r)
        rM = (torch.einsum('fab,fb->fa', T['Mo'][sc], rF)*rF).sum(1)
        K = torch.zeros(n, 3, 3, dtype=f64, device=dev)
        K.index_add_(0, kc, (self.V0[sc]*self.lam)[:, None, None]*(cr[:, :, None]*cr[:, None, :])+rM[:, None, None]*self.I3)
        ms = self._m_s
        kb = (T['w'][:ms]*(self.tbe[self.tlo]+self.tbe[self.thi]))[:, None, None]*self.I3
        K.index_add_(0, self.tlo, kb); K.index_add_(0, self.thi, kb)
        return K
