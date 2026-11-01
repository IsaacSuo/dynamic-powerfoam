"""The elastic end of the merged model read from the rest shape (2026-11-01; tl_check.py its static checks). What changes against merged_ul.py at no limit, whose F and p are running sums over the steps' cuts, so what a step misreads stays as a false permanent strain:
  cut      the rest cut, once. No radii, no re-cut; the ground's balls keep their rest radii
  F        a function of x: V F_i = sum_walls A x_w (x) n_out + x_i (x) a_i + Fe_i Mc_i, x_w = (1 - t) x_i + t x_j + 1/2 (Fe_i + Fe_j) tau, with u = x - X in ul_plastic.py's kernels, xn = X, F0 = I
  r        one reading per cell per wall: r_iw = x_w - x_i - F_i (c_w - X_i). The divergence theorem gives sum_w A r_iw (x) n = 0 on a closed cell at any x, so the pyramid gradient F_i + (beta / h) r (x) n has no cross term with F_i. E_w = beta^2 / 3 (A / h) [mu / 2 |r|^2 + 1/2 (kap + mu / 3)(r . R_i n)^2], h from the cell's centroid.
           beta^2 = 3 is the value at which an orthogonal mesh gives dual linear FE's energy. The cap's own term (mu / 2 tr((Fe - F) Mc (Fe - F)^T)) is left out: it changed no mode in tl_check.py
  shape    V0 mu |dev sym(R^T F)|^2 at z, R frozen from Newton's start, as merged_ul.py
  volume   V0 kap / 2 L^2, L = ln det F(z), a function of x too (at J under 0.05 continued linearly in J). merged_ul.py's frozen linear reading, carried as a history, left L up to 7.5 off ln det F on a part's free oscillation, its tiny cells
  viscous  Kelvin-Voigt on the same readings' change over the step: the shape's e(z) - e_n, L(z) - L_n, r(z) - r_n, each with its elastic form times cv = beta / (th dt). No inverse of F, and a rigid spin reads none
  through MergedStep. Yield is not here: this is the limit's infinite end only"""
import math

import numpy as np
import torch
import warp as wp

from free_surface.closure_warp import k_F8, k_Aul, k_Gul, k_Gul_adj, k_ball, k_ball_adj
from free_surface.merged_ul import MergedUL, k_cell_shape
from free_surface.skin_warp import NSLOT, NT, k_total, k_total_loss
from free_surface.spring_plastic import k_Fs, k_Fs_adj
from free_surface.ul_plastic import k_Aul_adj2, k_Ge, k_Ge_adj, k_gg_cell, k_gg_cell_adj, k_gg_wall, k_gg_wall_adj

f64 = torch.float64
J0 = .05


def log_J(J):
    """ln J, continued linearly in J at J <= J0, and its derivative"""
    q = J > J0
    return torch.where(q, torch.log(torch.where(q, J, torch.ones_like(J))), math.log(J0)+(J-J0)/J0), torch.where(q, 1/torch.where(q, J, torch.ones_like(J)), torch.full_like(J, 1/J0))


def cofactor(F):
    a0, a1, a2 = F[:, :, 0], F[:, :, 1], F[:, :, 2]
    return torch.stack([torch.linalg.cross(a1, a2), torch.linalg.cross(a2, a0), torch.linalg.cross(a0, a1)], 2)


@wp.kernel(enable_backward=False)
def k_half(x: wp.array(dtype=wp.vec3d), X: wp.array(dtype=wp.vec3d), G: wp.array(dtype=wp.mat33d), Ge: wp.array(dtype=wp.mat33d), wi: wp.array(dtype=int), wj: wp.array(dtype=int), wt: wp.array(dtype=wp.float64), wtau: wp.array(dtype=wp.vec3d),
           cI: wp.array(dtype=wp.vec3d), cJ: wp.array(dtype=wp.vec3d), wn: wp.array(dtype=wp.vec3d), wI: wp.array(dtype=wp.float64), wJ: wp.array(dtype=wp.float64), Rh: wp.array(dtype=wp.mat33d), rIn: wp.array(dtype=wp.vec3d), rJn: wp.array(dtype=wp.vec3d),
           muh: wp.float64, knh: wp.float64, cv: wp.float64, rI: wp.array(dtype=wp.vec3d), rJ: wp.array(dtype=wp.vec3d), terms: wp.array2d(dtype=wp.float64), x_adj: wp.array(dtype=wp.vec3d), Gb: wp.array(dtype=wp.mat33d), Gbe: wp.array(dtype=wp.mat33d)):
    """each side's remainder at z, its energy and force; into x, the cell's gradient and the fits through the wall's value"""
    e = wp.tid()
    i, j = wi[e], wj[e]
    if i == j:
        return
    half = wp.float64(0.5)
    t = wt[e]
    ca = wp.float64(1.0)-t
    ui = x[i]-X[i]
    uj = x[j]-X[j]
    uw = ca*ui+t*uj+half*((Ge[i]+Ge[j])@wtau[e])
    ri = uw-ui-G[i]@cI[e]
    rj = uw-uj-G[j]@cJ[e]
    rI[e] = ri
    rJ[e] = rj
    n = wn[e]
    ni = Rh[i]@n
    nj = Rh[j]@n
    fi = wI[e]*(muh*ri+(knh*wp.dot(ri, ni))*ni)
    fj = wJ[e]*(muh*rj+(knh*wp.dot(rj, nj))*nj)
    wp.atomic_add(terms, 5, e % NSLOT, half*(wp.dot(fi, ri)+wp.dot(fj, rj)))
    if cv > wp.float64(0.0):
        di = ri-rIn[e]
        dj = rj-rJn[e]
        gi = (cv*wI[e])*(muh*di+(knh*wp.dot(di, ni))*ni)
        gj = (cv*wJ[e])*(muh*dj+(knh*wp.dot(dj, nj))*nj)
        wp.atomic_add(terms, 7, e % NSLOT, half*(wp.dot(gi, di)+wp.dot(gj, dj)))
        fi = fi+gi
        fj = fj+gj
    fs = fi+fj
    wp.atomic_add(x_adj, i, ca*fs-fi)
    wp.atomic_add(x_adj, j, t*fs-fj)
    wp.atomic_add(Gb, i, -wp.outer(fi, cI[e]))
    wp.atomic_add(Gb, j, -wp.outer(fj, cJ[e]))
    M = half*wp.outer(fs, wtau[e])
    wp.atomic_add(Gbe, i, M)
    wp.atomic_add(Gbe, j, M)


@wp.kernel(enable_backward=False)
def k_visc_shape(Rh: wp.array(dtype=wp.mat33d), er: wp.array(dtype=wp.mat33d), en: wp.array(dtype=wp.mat33d), V0: wp.array(dtype=wp.float64), emu: wp.float64, th_: wp.float64, terms: wp.array2d(dtype=wp.float64), Fb: wp.array(dtype=wp.mat33d)):
    """e(z) - e_n = th (e_end - e_n); term 7; into dE/dF"""
    b = wp.tid()
    dv = th_*(er[b]-en[b])
    wp.atomic_add(terms, 7, b % NSLOT, V0[b]*emu*wp.ddot(dv, dv))
    Fb[b] = Fb[b]+V0[b]*(Rh[b]@((wp.float64(2.0)*emu)*dv))


@wp.kernel(enable_backward=False)
def k_vol(Fc: wp.array(dtype=wp.mat33d), Ln: wp.array(dtype=wp.float64), V0: wp.array(dtype=wp.float64), kap: wp.float64, ekap: wp.float64, L: wp.array(dtype=wp.float64), terms: wp.array2d(dtype=wp.float64), Fb: wp.array(dtype=wp.mat33d)):
    """V0 kap / 2 L^2, L = ln det F(z), and the viscous part on L - L_n; into dE/dF"""
    b = wp.tid()
    F = Fc[b]
    a0 = wp.vec3d(F[0, 0], F[1, 0], F[2, 0])
    a1 = wp.vec3d(F[0, 1], F[1, 1], F[2, 1])
    a2 = wp.vec3d(F[0, 2], F[1, 2], F[2, 2])
    c0 = wp.cross(a1, a2)
    c1 = wp.cross(a2, a0)
    c2 = wp.cross(a0, a1)
    J = wp.dot(a0, c0)
    J0_ = wp.float64(0.05)
    lb = wp.log(J0_)+(J-J0_)/J0_
    dl = wp.float64(1.0)/J0_
    if J > J0_:
        lb = wp.log(J)
        dl = wp.float64(1.0)/J
    L[b] = lb
    d = lb-Ln[b]
    half = wp.float64(0.5)
    wp.atomic_add(terms, 1, b % NSLOT, V0[b]*half*kap*lb*lb)
    wp.atomic_add(terms, 7, b % NSLOT, V0[b]*half*ekap*d*d)
    s = V0[b]*dl*(kap*lb+ekap*d)
    Fb[b] = Fb[b]+wp.matrix_from_cols(s*c0, s*c1, s*c2)


class MergedRest(MergedUL):
    """as MergedUL at sigma_y inf; set_ball, set_viscosity, then rest_cut(), then prepare(). beta2 the remainders' share: 3 as dual linear FE, 1 the pyramid's own volume"""

    def __init__(self, X, r, V0, pairs, A, c, E, nu, theta=.5, beta2=3., **kw):
        super().__init__(X, r, V0, pairs, A, c, E, nu, float('inf'), theta=theta, kshare=1/6, **kw)
        self.beta2 = float(beta2)
        self.muh, self.knh = self.beta2/3*self.mu, self.beta2/3*(self.kap+self.mu/3)
        self._cut_done = False

    def rest_cut(self):
        """the rest cut's arrays, once, and the remainders' from them"""
        MergedUL.update_ul(self, self.X)
        n, dev = self.n, self.dev
        t, U = self.tul, self.ul
        m = U_m = self.ul_stats['walls']
        Kw = U['Kw']
        z = lambda k_, dt_: wp.zeros(k_, dtype=dt_, device=dev)
        H = dict(cI=z(Kw, wp.vec3d), cJ=z(Kw, wp.vec3d), wI=z(Kw, wp.float64), wJ=z(Kw, wp.float64), rIn=z(Kw, wp.vec3d), rJn=z(Kw, wp.vec3d), rI=z(Kw, wp.vec3d), rJ=z(Kw, wp.vec3d), Lc=z(n, wp.float64), Xr=wp.from_torch(self.X, dtype=wp.vec3d))
        self.H = H
        tt = lambda a_: wp.to_torch(a_, requires_grad=False)
        self.th_ = {k_: tt(a_) for k_, a_ in H.items() if k_ != 'Xr'}
        i, j, A, nh, tw, tau = t['wi'][:m].long(), t['wj'][:m].long(), t['wa'][:m], t['wn'][:m], t['wt'][:m], t['wtau'][:m]
        X = self.X
        d = X[j]-X[i]; L = d.norm(dim=1)
        c = X[i]+tw[:, None]*d+tau
        hi = ((c-X[i])*nh).sum(1); hj = L-hi
        # the cell's centroid: pyramids from the site on the walls, a spherical sector on the cap
        V = self._V
        Cm = torch.zeros(n, 3, dtype=f64, device=dev).index_add(0, i, (A*hi/4)[:, None]*(c-X[i])).index_add(0, j, (A*hj/4)[:, None]*(c-X[j]))
        af = torch.zeros(n, 3, dtype=f64, device=dev).index_add(0, i, -A[:, None]*nh).index_add(0, j, A[:, None]*nh)
        r = self._ulr
        Cen = X+(Cm+(r*r/4)[:, None]*af)/V[:, None]
        hci = ((c-Cen[i])*nh).sum(1); hcj = ((Cen[j]-c)*nh).sum(1)
        h = float(self.V0.median())**(1/3)
        self.h_stats = (float(torch.minimum(hci, hcj).min())/h, int((torch.minimum(hci, hcj) <= 0).sum()), m)
        floor = 1e-3*h
        self.th_['cI'][:m] = c-X[i]; self.th_['cJ'][:m] = c-X[j]
        self.th_['wI'][:m] = A/hci.clamp(min=floor); self.th_['wJ'][:m] = A/hcj.clamp(min=floor)
        self._cut_done = True
        self._F_rest_blocks()

    def _F_rest_blocks(self):
        """the gradient's rows on each cell (fixed: the rest cut), and the remainders' diagonal blocks through them, once"""
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
        # a side's coefficient on site k: the wall's value, minus its own site, minus the row dotted with the lever
        H, sc, kc, rw = self.th_, self._sc, self._kc, self._rw
        cnt = torch.bincount(sc, minlength=n); zero = torch.zeros(1, dtype=torch.long, device=dev); first = torch.cat([zero, torch.cumsum(cnt, 0)[:-1]])
        Kh = torch.zeros(n, 3, 3, dtype=f64, device=dev)
        I3 = self.I3; nnt = nn[:, :, None]*nn[:, None, :]
        for own, oth, D, w_, ao, bo in ((i, j, H['cI'][:m], H['wI'][:m], -tw, tw), (j, i, H['cJ'][:m], H['wJ'][:m], -(1-tw), 1-tw)):
            rep = cnt[own]; e1 = torch.repeat_interleave(torch.arange(m, device=dev), rep)
            pos = torch.arange(int(rep.sum()), device=dev)-torch.cat([zero, torch.cumsum(rep, 0)[:-1]])[e1]+first[own][e1]
            ee = torch.arange(m, device=dev)
            uu, iv = torch.unique(torch.cat([e1, ee, ee])*n+torch.cat([kc[pos], own, oth]), return_inverse=True)
            al = torch.zeros(len(uu), dtype=f64, device=dev).index_add(0, iv, torch.cat([-(rw[pos]*D[e1]).sum(1), ao, bo]))
            e2, k2 = uu//n, uu % n
            Kh.index_add_(0, k2, (al*al*w_[e2])[:, None, None]*(self.muh*I3+self.knh*nnt[e2]))
        self._Kh = Kh

    def set_rest_state(self):
        """the state at rest: F_n = I, L_n = 0, remainders 0"""
        self._Fe = self.I3.expand(self.n, 3, 3).clone()
        self.tul['Ln'].zero_()
        self.th_['rIn'].zero_(); self.th_['rJn'].zero_()

    def run(self):
        """E and dE/dx at xw"""
        B, U, S, H, n, Kw = self.B, self.ul, self.sp, self.H, self.n, self.ul['Kw']
        x, g = self.xw, self.xw.grad
        for k_ in ('G0', 'Ga', 'Sg', 'Lb', 'Gbe', 'Gb0', 'terms', 'tsum'):
            B[k_].zero_()
        g.zero_()
        kap, mu, sy, th_ = wp.float64(self.kap), wp.float64(self.mu), wp.float64(self.sy), wp.float64(self.th)
        wp.launch(k_Gul, 2*Kw, inputs=[x, U['xn'], U['S'], U['O'], U['R']], outputs=[B['G0']])
        wp.launch(k_Aul, Kw, inputs=[B['G0'], U['As'], U['Ao'], U['Aw']], outputs=[B['Ga']])
        wp.launch(k_Ge, n, inputs=[B['G0'], B['Ga'], U['used']], outputs=[B['Ge']])
        wall = [U['wi'], U['wj'], U['wa'], U['wn'], U['wt'], U['wtau']]
        wp.launch(k_gg_wall, Kw, inputs=[x, U['xn']]+wall+[B['Ge']], outputs=[B['Sg']])
        wp.launch(k_gg_cell, n, inputs=[x, U['xn'], B['Sg'], B['Ge'], U['a'], U['Mc'], U['Vi']], outputs=[B['G']])
        wp.launch(k_Fs, n, inputs=[B['G'], S['F0']], outputs=[B['Fc']])
        wp.launch(k_cell_shape, n, inputs=[B['Fc'], S['Rh'], S['en'], U['V0'], mu, sy, th_], outputs=[B['ei'], B['be'], B['er'], B['Fb'], B['terms']])
        cv = wp.float64(self.cv)
        if self.cv > 0.:
            wp.launch(k_visc_shape, n, inputs=[S['Rh'], B['er'], S['en'], U['V0'], wp.float64(self.cv*self.mu), th_], outputs=[B['terms'], B['Fb']])
        wp.launch(k_vol, n, inputs=[B['Fc'], U['Ln'], U['V0'], kap, wp.float64(self.cv*self.kap)], outputs=[B['L'], B['terms'], B['Fb']])
        wp.launch(k_Fs_adj, n, inputs=[B['Fb'], S['F0']], outputs=[B['Gb']])
        wp.launch(k_half, Kw, inputs=[x, H['Xr'], B['G'], B['Ge'], U['wi'], U['wj'], U['wt'], U['wtau'], H['cI'], H['cJ'], U['wn'], H['wI'], H['wJ'], S['Rh'], H['rIn'], H['rJn'], wp.float64(self.muh), wp.float64(self.knh), cv],
                  outputs=[H['rI'], H['rJ'], B['terms'], g, B['Gb'], B['Gbe']])
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

    def _read_C(self, F):
        """dL/dF at F, for the blocks"""
        cof = cofactor(F)
        self.tul['Q'].copy_(log_J((F[:, :, 0]*cof[:, :, 0]).sum(1))[1][:, None, None]*cof)

    def set_state_at(self, x):
        """F_n, L_n, the remainders read at x"""
        self.grad(x)
        self._Fe = self.tFc.clone()
        self.tul['Ln'].copy_(log_J(torch.linalg.det(self._Fe))[0])
        self.th_['rIn'].copy_(self.th_['rI']); self.th_['rJn'].copy_(self.th_['rJ'])

    def update_ul(self, x):
        """the step's start: no cut. R, dL/dF at F_n, e_n; x's only use is in the kernels"""
        s = self.tsp
        s['F0'].copy_(self.I3.expand(self.n, 3, 3))
        Fn = self._Fe
        wp.launch(k_F8, self.n, inputs=[wp.from_torch(Fn.contiguous(), dtype=wp.mat33d)], outputs=[self.B['Fp'], self.B['Rp'], self.B['qp']])
        s['Rh'].copy_(self.tRp)
        A_ = self.tRp.transpose(1, 2)@Fn; e_ = .5*(A_+A_.transpose(1, 2))-self.I3
        s['en'].copy_(e_-(e_.diagonal(dim1=1, dim2=2).sum(1)/3)[:, None, None]*self.I3)
        self._read_C(Fn)

    def reread(self, beta=False):
        """at Newton's start, at z0: R and C frozen for the step"""
        wp.launch(k_F8, self.n, inputs=[self.B['Fc']], outputs=[self.B['Fp'], self.B['Rp'], self.B['qp']])
        self.tsp['Rh'].copy_(self.tRp)
        self._read_C(self.tFc)

    def end_ul(self, th=.5):
        """the buffers at the accepted z; F, r linear in x, so their ends are exact; L read from the end's F"""
        assert th == self.th, 'the model and the step on one theta'
        t, s, H, n = self.tul, self.tsp, self.th_, self.n
        m = self.ul_stats['walls']
        Rh = s['Rh']; i, j = t['wi'][:m].long(), t['wj'][:m].long(); nn = t['wn'][:m]
        ni, nj = torch.einsum('fab,fb->fa', Rh[i], nn), torch.einsum('fab,fb->fa', Rh[j], nn)
        rIz, rJz = H['rI'][:m], H['rJ'][:m]
        rIn, rJn = H['rIn'][:m], H['rJn'][:m]
        wI, wJ = H['wI'][:m], H['wJ'][:m]
        Fe = (self.tFc-(1-th)*self._Fe)/th
        if self.cv > 0.:                                                            # (the viscous forces' work over the step)
            dv = th*(self.ter-s['en'])
            dL = self.tL-t['Ln']
            cof = cofactor(self.tFc); dlz = log_J((self.tFc[:, :, 0]*cof[:, :, 0]).sum(1))[1]
            di, dj = rIz-rIn, rJz-rJn
            self.Wv += float((self.V0*self.cv*2*self.mu*(dv*dv).sum((1, 2))).sum())/th+float((self.V0*self.cv*self.kap*dL*dlz*(cof*(Fe-self._Fe)).sum((1, 2))).sum())
            self.Wv += float((self.cv*wI*(self.muh*(di*di).sum(1)+self.knh*(di*ni).sum(1)**2)).sum()+(self.cv*wJ*(self.muh*(dj*dj).sum(1)+self.knh*(dj*nj).sum(1)**2)).sum())/th
        self._Fe = Fe
        t['Ln'].copy_(log_J(torch.linalg.det(Fe))[0])
        H['rIn'][:m] = (rIz-(1-th)*rIn)/th; H['rJn'][:m] = (rJz-(1-th)*rJn)/th
        rIe, rJe = H['rIn'][:m], H['rJn'][:m]
        self.spring_store = .5*float((wI*(self.muh*(rIe*rIe).sum(1)+self.knh*(rIe*ni).sum(1)**2)).sum()+(wJ*(self.muh*(rJe*rJe).sum(1)+self.knh*(rJe*nj).sum(1)**2)).sum())
        self.ul_stats['flips'] = int((torch.linalg.det(Fe) <= 0).sum())

    def ul_blocks(self, lr_thr=None, lr_cap=0):
        """the diagonal blocks at the frozen R, C: the cell's through its rows, the remainders' through theirs at rest"""
        assert lr_thr is None, 'no low-rank term here'
        sc, kc, r = self._sc, self._kc, self._rw
        Rh = self.tsp['Rh']
        Rr = torch.einsum('fab,fb->fa', Rh[sc], r)
        r2 = (r*r).sum(1)
        cr = torch.einsum('fab,fb->fa', self.tul['Q'][sc], r)
        K = torch.zeros(self.n, 3, 3, dtype=f64, device=self.dev)
        K.index_add_(0, kc, (self.V0[sc]*self.mu)[:, None, None]*(r2[:, None, None]*self.I3+Rr[:, :, None]*Rr[:, None, :]/3)+(self.V0[sc]*self.kap)[:, None, None]*(cr[:, :, None]*cr[:, None, :]))
        return K+Rh@self._Kh@Rh.transpose(1, 2)
