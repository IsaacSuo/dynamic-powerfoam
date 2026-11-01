"""The ring spring (2026-11-01, checks in ring_checks.py; the user: springs carry the force). Each wall two half springs, one per cell, w = mu A / L each; each cell's half springs and its outer layer share one rotation. Moduli per unit density, masses = volumes.
  state     q_i, q_j per wall (the rest vector D until the cell yields), P per cell for the outer layer (its virtual springs q_k = P D_k), x
  gradient  F_g = I + G, G ul_plastic.py's divergence-theorem gradient on the rest cut, a function of x (u = x - X): for the outer layer and the pressure, whose force must cancel across every wall
  rotation  the whole cell's minimum, R = polar(A), A = sum w d (x) q + F_g Mo P^T, Mo = 2 mu Mc the outer moment. Read at every evaluation; no term for it in the force, E being stationary in R
            (R read from the fit F = A M^-1 reads a uniform strain exactly but has a net torque: on a free flight the energy rose 1 % and the angular momentum 0.75 % in 4 periods, ring_checks.py --test 10)
  energy    E = sum_cells [sum 1/2 w |d - R q|^2 + 1/2 tr((F_g - R P) Mo (F_g - R P)^T)] + sum V0 lam / 2 L^2 + variance, L = ln det F_g (at J under 0.05 continued linearly in J), variance lam A L / 24 (L_i - L_j)^2.
            The springs hold the deviatoric part and 2 mu of the volumetric; so lam here, not kap. No bending or torsion: the modes do not ask for them (ring_checks.py --test 5, 11)
  viscous   Kelvin-Voigt on the same readings in the cell's frame over the step: m = R^T d - q, H = R^T F_g - P, L, against their values at the step's start, times cv = beta / (th dt). Here E is not stationary in R, so the torque's force is written out: dE = -omega . tau,
            (tr S I - S) omega = 2 axial(skew(R^T dA)), S = R^T A symmetric
  ground    Hertz on the rest radii of the cells with a cap, as MergedRest. No cut after the rest cut while nothing yields
The elastic end, through MidpointStep (ring_lion.py); ring_plastic.py the same with a yield limit."""
import math

import numpy as np
import torch
import warp as wp

from free_surface.closure_warp import k_Aul, k_Gul, k_Gul_adj, k_ball, k_ball_adj, rot_proper
from free_surface.merged_ul import MergedUL
from free_surface.skin_warp import NSLOT, k_total, k_total_loss
from free_surface.ul_plastic import k_Aul_adj2, k_Ge, k_Ge_adj, k_Lb, k_gg_cell, k_gg_cell_adj, k_gg_wall, k_gg_wall_adj, k_var

f64 = torch.float64


@wp.kernel(enable_backward=False)
def k_Fc(G: wp.array(dtype=wp.mat33d), Fc: wp.array(dtype=wp.mat33d)):
    b = wp.tid()
    Fc[b] = wp.identity(n=3, dtype=wp.float64)+G[b]


@wp.kernel(enable_backward=False)
def k_ring_A(x: wp.array(dtype=wp.vec3d), ri: wp.array(dtype=int), rj: wp.array(dtype=int), rw: wp.array(dtype=wp.float64), qi: wp.array(dtype=wp.vec3d), qj: wp.array(dtype=wp.vec3d), Am: wp.array(dtype=wp.mat33d)):
    """sum w d (x) q per cell"""
    e = wp.tid()
    i, j = ri[e], rj[e]
    if i == j:
        return
    d = x[j]-x[i]
    wp.atomic_add(Am, i, rw[e]*wp.outer(d, qi[e]))
    wp.atomic_add(Am, j, rw[e]*wp.outer(d, qj[e]))


@wp.kernel(enable_backward=False)
def k_ring_R(Am: wp.array(dtype=wp.mat33d), Fc: wp.array(dtype=wp.mat33d), MoP: wp.array(dtype=wp.mat33d), Rc: wp.array(dtype=wp.mat33d), Si: wp.array(dtype=wp.mat33d)):
    """the whole cell's rotation and (tr S I - S)^-1, the polar's derivative (0 at its singular line: the viscous torque's force is dropped there)"""
    b = wp.tid()
    A = Am[b]+Fc[b]@MoP[b]
    R = rot_proper(A)
    Rc[b] = R
    S = wp.transpose(R)@A
    S = (S+wp.transpose(S))*wp.float64(0.5)
    T = wp.identity(n=3, dtype=wp.float64)*wp.trace(S)-S
    tr = wp.trace(T)
    d = wp.determinant(T)
    if wp.abs(d) > wp.float64(1e-9)*tr*tr*tr:
        Si[b] = wp.inverse(T)
    else:
        Si[b] = wp.mat33d()


@wp.kernel(enable_backward=False)
def k_ring_spring(x: wp.array(dtype=wp.vec3d), ri: wp.array(dtype=int), rj: wp.array(dtype=int), rw: wp.array(dtype=wp.float64), qi: wp.array(dtype=wp.vec3d), qj: wp.array(dtype=wp.vec3d), Rc: wp.array(dtype=wp.mat33d),
                  min_: wp.array(dtype=wp.vec3d), mjn: wp.array(dtype=wp.vec3d), cv: wp.float64, mi: wp.array(dtype=wp.vec3d), mj: wp.array(dtype=wp.vec3d), tau: wp.array(dtype=wp.vec3d), terms: wp.array2d(dtype=wp.float64), x_adj: wp.array(dtype=wp.vec3d)):
    """each half's reading in its cell's frame, its energy and force; term 0, viscous 7"""
    e = wp.tid()
    i, j = ri[e], rj[e]
    if i == j:
        return
    half = wp.float64(0.5)
    d = x[j]-x[i]
    w = rw[e]
    di = wp.transpose(Rc[i])@d
    dj = wp.transpose(Rc[j])@d
    a = di-qi[e]
    b = dj-qj[e]
    mi[e] = a
    mj[e] = b
    wp.atomic_add(terms, 0, e % NSLOT, half*w*(wp.dot(a, a)+wp.dot(b, b)))
    fi = w*a
    fj = w*b
    if cv > wp.float64(0.0):
        da = a-min_[e]
        db = b-mjn[e]
        wc = w*cv
        wp.atomic_add(terms, 7, e % NSLOT, half*wc*(wp.dot(da, da)+wp.dot(db, db)))
        fi = fi+wc*da
        fj = fj+wc*db
        wp.atomic_add(tau, i, wc*wp.cross(di, da))
        wp.atomic_add(tau, j, wc*wp.cross(dj, db))
    f = Rc[i]@fi+Rc[j]@fj
    wp.atomic_add(x_adj, j, f)
    wp.atomic_add(x_adj, i, -f)


@wp.kernel(enable_backward=False)
def k_ring_outer(Fc: wp.array(dtype=wp.mat33d), Rc: wp.array(dtype=wp.mat33d), Pc: wp.array(dtype=wp.mat33d), Mo: wp.array(dtype=wp.mat33d), Hn: wp.array(dtype=wp.mat33d), cv: wp.float64,
                 Hr: wp.array(dtype=wp.mat33d), Fb: wp.array(dtype=wp.mat33d), tau: wp.array(dtype=wp.vec3d), terms: wp.array2d(dtype=wp.float64)):
    """the outer layer, H = R^T F_g - P, 1/2 tr(H Mo H^T); term 5, viscous 7; Fb = dE/dF_g from here on"""
    b = wp.tid()
    half = wp.float64(0.5)
    Rt = wp.transpose(Rc[b])
    Ft = Rt@Fc[b]
    H = Ft-Pc[b]
    Hr[b] = H
    Z = H@Mo[b]
    wp.atomic_add(terms, 5, b % NSLOT, half*wp.ddot(Z, H))
    if cv > wp.float64(0.0):
        D = H-Hn[b]
        Zv = D@Mo[b]
        wp.atomic_add(terms, 7, b % NSLOT, half*cv*wp.ddot(Zv, D))
        Z = Z+cv*Zv
        Y = (Ft@Mo[b])@wp.transpose(D)
        wp.atomic_add(tau, b, cv*wp.vec3d(Y[1, 2]-Y[2, 1], Y[2, 0]-Y[0, 2], Y[0, 1]-Y[1, 0]))
    Fb[b] = Rc[b]@Z


@wp.kernel(enable_backward=False)
def k_ring_tau(tau: wp.array(dtype=wp.vec3d), Si: wp.array(dtype=wp.mat33d), Rc: wp.array(dtype=wp.mat33d), MoP: wp.array(dtype=wp.mat33d), RY: wp.array(dtype=wp.mat33d), Fb: wp.array(dtype=wp.mat33d)):
    """dE = -(R [Y]x) : dA, Y = (tr S I - S)^-1 tau: the part through F_g here"""
    b = wp.tid()
    Y = Si[b]@tau[b]
    Yx = wp.matrix_from_rows(wp.vec3d(wp.float64(0.0), -Y[2], Y[1]), wp.vec3d(Y[2], wp.float64(0.0), -Y[0]), wp.vec3d(-Y[1], Y[0], wp.float64(0.0)))
    M = Rc[b]@Yx
    RY[b] = M
    Fb[b] = Fb[b]-M@wp.transpose(MoP[b])


@wp.kernel(enable_backward=False)
def k_ring_tau_adj(ri: wp.array(dtype=int), rj: wp.array(dtype=int), rw: wp.array(dtype=wp.float64), qi: wp.array(dtype=wp.vec3d), qj: wp.array(dtype=wp.vec3d), RY: wp.array(dtype=wp.mat33d), x_adj: wp.array(dtype=wp.vec3d)):
    """the same through the half springs' d"""
    e = wp.tid()
    i, j = ri[e], rj[e]
    if i == j:
        return
    v = -rw[e]*(RY[i]@qi[e]+RY[j]@qj[e])
    wp.atomic_add(x_adj, j, v)
    wp.atomic_add(x_adj, i, -v)


@wp.kernel(enable_backward=False)
def k_ring_vol(Fc: wp.array(dtype=wp.mat33d), Ln: wp.array(dtype=wp.float64), V0: wp.array(dtype=wp.float64), lam: wp.float64, ekap: wp.float64, L: wp.array(dtype=wp.float64), dL: wp.array(dtype=wp.mat33d), Lb: wp.array(dtype=wp.float64), terms: wp.array2d(dtype=wp.float64)):
    """L = ln det F_g, dL/dF; V0 lam / 2 L^2, term 1, viscous 7; dE/dL into Lb"""
    b = wp.tid()
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
    L[b] = lb
    d = lb-Ln[b]
    half = wp.float64(0.5)
    wp.atomic_add(terms, 1, b % NSLOT, V0[b]*half*lam*lb*lb)
    wp.atomic_add(terms, 7, b % NSLOT, V0[b]*half*ekap*d*d)
    dL[b] = wp.matrix_from_cols(dl*c0, dl*c1, dl*c2)
    Lb[b] = V0[b]*(lam*lb+ekap*d)


class RingUL(MergedUL):
    """X, r, V0: the foam at rest; pairs, A, c: its walls (geometry.npz); E, nu per unit density. set_ball, set_viscosity, then rest_cut(), then prepare(); through MidpointStep"""

    def __init__(self, X, r, V0, pairs, A, c, E, nu, theta=.5, sigma_y=float('inf'), **kw):
        super().__init__(X, r, V0, pairs, A, c, E, nu, sigma_y, theta=theta, kshare=1/6, **kw)
        self._cut_done = False

    def rest_cut(self):
        """the rest cut, once: the gradient's arrays, the half springs at rest"""
        MergedUL.update_ul(self, self.X)
        n, dev, t = self.n, self.dev, self.tul
        i, j, Lw, key, _, m = self._w
        A, tw, nn, Vi = t['wa'][:m], t['wt'][:m], t['wn'][:m], t['Vi']
        t['wv'][:m] = self.lam*A*Lw/24                                                  # (the springs hold 2 mu of the volumetric: lam here)
        lo, hi = torch.minimum(i, j), torch.maximum(i, j)
        D = self.X[hi]-self.X[lo]
        z = lambda k_, dt_: wp.zeros(k_, dtype=dt_, device=dev)
        Ks = self.ul['Kw']
        self.ring = dict(K=Ks, i=z(Ks, wp.int32), j=z(Ks, wp.int32), w=z(Ks, wp.float64), qi=z(Ks, wp.vec3d), qj=z(Ks, wp.vec3d), mi=z(Ks, wp.vec3d), mj=z(Ks, wp.vec3d), min=z(Ks, wp.vec3d), mjn=z(Ks, wp.vec3d),
                         Am=z(n, wp.mat33d), Rc=z(n, wp.mat33d), Si=z(n, wp.mat33d), tau=z(n, wp.vec3d), RY=z(n, wp.mat33d), Mo=z(n, wp.mat33d), MoP=z(n, wp.mat33d), Pc=z(n, wp.mat33d), Hr=z(n, wp.mat33d), Hn=z(n, wp.mat33d), dL=z(n, wp.mat33d))
        tt = lambda a_: wp.to_torch(a_, requires_grad=False)
        self.tr = {k_: tt(a_) for k_, a_ in self.ring.items() if k_ != 'K'}
        T = self.tr
        T['i'][:m] = lo.to(torch.int32); T['j'][:m] = hi.to(torch.int32); T['w'][:m] = self.mu*A/Lw; T['qi'][:m] = D; T['qj'][:m] = D
        Mc = t['Mc']
        T['Mo'].copy_(self.mu*(Mc+Mc.transpose(1, 2)))
        T['Pc'].copy_(self.I3.expand(n, 3, 3)); T['MoP'].copy_(T['Mo'])
        self._m = m
        self._rows()
        self._cut_done = True

    def _rows(self):
        """the gradient's rows on each cell (the rest cut), for the blocks"""
        n, t, dev = self.n, self.tul, self.dev
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
        self._rw = torch.zeros(len(u), 3, dtype=f64, device=dev).index_add(0, inv, ra); self._sc, self._kc = u//n, u % n
        # the springs' diagonal, fixed while the walls are the rest walls
        T = self.tr; w = T['w'][:m]
        self._Kd = torch.zeros(n, dtype=f64, device=dev).index_add(0, T['i'][:m].long(), 2*w).index_add(0, T['j'][:m].long(), 2*w)

    def run(self):
        """E and dE/dx at xw"""
        B, U, Rg, n, Kw = self.B, self.ul, self.ring, self.n, self.ul['Kw']
        x, g = self.xw, self.xw.grad
        for k_ in ('G0', 'Ga', 'Sg', 'Lb', 'Gbe', 'Gb0', 'terms', 'tsum'):
            B[k_].zero_()
        Rg['Am'].zero_(); Rg['tau'].zero_()
        g.zero_()
        cv = wp.float64(self.cv)
        ke = (x, Rg['i'], Rg['j'], Rg['w'], Rg['qi'], Rg['qj'])
        wp.launch(k_Gul, 2*Kw, inputs=[x, U['xn'], U['S'], U['O'], U['R']], outputs=[B['G0']])
        wp.launch(k_Aul, Kw, inputs=[B['G0'], U['As'], U['Ao'], U['Aw']], outputs=[B['Ga']])
        wp.launch(k_Ge, n, inputs=[B['G0'], B['Ga'], U['used']], outputs=[B['Ge']])
        wall = [U['wi'], U['wj'], U['wa'], U['wn'], U['wt'], U['wtau']]
        wp.launch(k_gg_wall, Kw, inputs=[x, U['xn']]+wall+[B['Ge']], outputs=[B['Sg']])
        wp.launch(k_gg_cell, n, inputs=[x, U['xn'], B['Sg'], B['Ge'], U['a'], U['Mc'], U['Vi']], outputs=[B['G']])
        wp.launch(k_Fc, n, inputs=[B['G']], outputs=[B['Fc']])
        wp.launch(k_ring_A, Rg['K'], inputs=list(ke), outputs=[Rg['Am']])
        wp.launch(k_ring_R, n, inputs=[Rg['Am'], B['Fc'], Rg['MoP']], outputs=[Rg['Rc'], Rg['Si']])
        wp.launch(k_ring_spring, Rg['K'], inputs=list(ke)+[Rg['Rc'], Rg['min'], Rg['mjn'], cv], outputs=[Rg['mi'], Rg['mj'], Rg['tau'], B['terms'], g])
        wp.launch(k_ring_outer, n, inputs=[B['Fc'], Rg['Rc'], Rg['Pc'], Rg['Mo'], Rg['Hn'], cv], outputs=[Rg['Hr'], B['Fb'], Rg['tau'], B['terms']])
        if self.cv > 0.:
            wp.launch(k_ring_tau, n, inputs=[Rg['tau'], Rg['Si'], Rg['Rc'], Rg['MoP']], outputs=[Rg['RY'], B['Fb']])
            wp.launch(k_ring_tau_adj, Rg['K'], inputs=[Rg['i'], Rg['j'], Rg['w'], Rg['qi'], Rg['qj'], Rg['RY']], outputs=[g])
        wp.launch(k_ring_vol, n, inputs=[B['Fc'], U['Ln'], U['V0'], wp.float64(self.lam), wp.float64(self.cv*self.lam)], outputs=[B['L'], Rg['dL'], B['Lb'], B['terms']])
        wp.launch(k_var, Kw, inputs=[U['wi'], U['wj'], U['wv'], B['L']], outputs=[B['Lb'], B['terms']])
        wp.launch(k_Lb, n, inputs=[B['Lb'], Rg['dL']], outputs=[B['Fb']])
        bl = self.ball
        if bl is not None:
            wp.launch(k_ball, bl['k'], inputs=[x, bl['body'], bl['R'], bl['xn'], bl['ng'], bl['par'], bl['kt'], bl['bound'], bl['lamt']], outputs=[B['bdelta'], B['bfn'], B['bff'], B['terms']])
            wp.launch(k_ball_adj, bl['k'], inputs=[bl['body'], bl['ng'], bl['par'], B['bfn'], B['bff'], self.tones], outputs=[g])
        wp.launch(k_gg_cell_adj, n, inputs=[B['Fb'], U['a'], U['Mc'], U['Vi']], outputs=[g, B['Gs'], B['Gbe']])
        wp.launch(k_gg_wall_adj, Kw, inputs=wall+[B['Gs']], outputs=[g, B['Gbe']])
        wp.launch(k_Ge_adj, n, inputs=[B['Gbe'], U['used']], outputs=[B['Gb0']])
        wp.launch(k_Aul_adj2, Kw, inputs=[B['Gbe'], U['As'], U['Ao'], U['Aw']], outputs=[B['Gb0']])
        wp.launch(k_Gul_adj, 2*Kw, inputs=[B['Gb0'], U['S'], U['O'], U['R']], outputs=[g])
        wp.launch(k_total, (9, 256), inputs=[B['terms']], outputs=[B['tsum']])
        wp.launch(k_total_loss, 1, inputs=[B['tsum'], self.tw], outputs=[B['loss']])

    def update_ul(self, x):
        """the step's start: no cut while nothing yields; its x for the step's end"""
        self._xs = x.detach().clone()

    def end_ul(self, th=.5):
        """at the accepted z. The viscous work of the step, 2 E_7(z) / th (the reading linear over the step: its change is D / th). The state read at the step's end x_e = x_n + (z - x_n) / th: the rotation is not linear in x, so read, not extrapolated; then z again, for the ground's buffers"""
        assert th == self.th, 'the model and the step on one theta'
        z = self.tx.clone()
        self.Wv += 2*float(self.tsum[7])/th
        self.grad((z-(1-th)*self._xs)/th)
        T, t = self.tr, self.tul
        T['min'].copy_(T['mi']); T['mjn'].copy_(T['mj']); T['Hn'].copy_(T['Hr']); t['Ln'].copy_(self.tL)
        self._Fe = self.tFc.clone()
        self.store = dict(springs=float(self.tsum[0]), vol=float(self.tsum[1]), var=float(self.tsum[2]), outer=float(self.tsum[5]))
        self.grad(z)

    def stored(self):
        return self.store['springs']+self.store['outer'], self.store['vol']+self.store['var']

    def ul_blocks(self, lr_thr=None, lr_cap=0):
        """the diagonal blocks at the last evaluation: the springs' 2 w I on each end (the rotation's relief left out), the outer layer and the pressure through the gradient's direct rows"""
        assert lr_thr is None, 'no low-rank term here'
        sc, kc, r = self._sc, self._kc, self._rw
        T = self.tr
        cr = torch.einsum('fab,fb->fa', T['dL'][sc], r)
        rM = (torch.einsum('fab,fb->fa', T['Mo'][sc], r)*r).sum(1)
        K = torch.zeros(self.n, 3, 3, dtype=f64, device=self.dev)
        K.index_add_(0, kc, (self.V0[sc]*self.lam)[:, None, None]*(cr[:, :, None]*cr[:, None, :])+rM[:, None, None]*self.I3)
        return K+self._Kd[:, None, None]*self.I3


