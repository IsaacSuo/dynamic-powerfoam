"""One model for elastic and plastic bodies (2026-10-31; the user: not one or the other, they can be joined). The limit is the only switch. Moduli per unit density, masses = volumes.
A spring's reading splits: x_j - x_i - R T = (F - R) T + (x_j - x_i - F T), the cell's shape change and a remainder. So the shape part goes to the cell, the remainder stays on the spring:
  state      each cell F (elastic: its stretch is what the limit holds), each spring p, what its rest offset T has become. x -> F is linear within a step, F = (I + G(x)) F_n, G the divergence-theorem gradient of ul_plastic.py over the connected walls
  cell       e = dev sym(R^T F): R frozen for the step, read at Newton's start, so the step is convex; R is the rotation nearest F, so its error costs nothing to first order, and the state is F alone. V0 W(e), W von Mises: mu e^2 up to e_y = s_y / (2 mu), then s_y e - s_y^2 / (4 mu). Pressure: V0 kap / 2 L^2, L = L_n + tr(C^T G) within the step, L_n + tr(C^T G) / th at its end, C = (I + G / (2 th))^-T frozen as R: at th = 1/2 its work is its store, a rotation reads 0, ln J is read to third order
             (10-31: tr(Q^T (I + G)) - 3 within the step with ln det at its end made 8.3 % of the start's energy by 0.27 s of the landing, everything else 0.03 %, as a cell flattened in one step reads little within it and a lot at its end). Variance as ul_plastic.py
  spring     r = x_j - x_i - p_n - 1/2 (G_i + G_j) p_n, linear in x and needing no rotation of its own, energy 1/2 k beta r^2, beta the mean of the two cells' return factors, frozen as R. k = mu A / 3 L, 1/6 of the DEM's bond 2 mu A / L, which carries all the shear: here the uniform part is the cell's and the spring holds only the sub-cell strain r / L over the two pyramids, volume A L / 3 (kshare; 1 gave 27 % of the plastic work to the springs and spread y +14 % against the MPM, 1/6 +5.6 %), of the rest wall while
             a cell of the pair has not yielded, of the current wall when both have. The lever is p, not each cell's F T (10-31, 1/2 (F_i + F_j) T with T = Fbar^-1 (d - r): in flow F is about a rotation and neighbours differ by up to a half turn,
             so Fbar went singular, smallest sigma 1e-5, lever 81631 L, the step's first |g| 5.85e7 weights, 99.6 % of it on the springs; without them 25-32)
  step       theta method, forces at z. The return belongs to the step's end: e_e = e_n + (e(z) - e_n) / th, its force (1 - th) 2 mu e_n + th beta 2 mu e_e, potential (1 - th) B(e_n, e - e_n) + th^2 W(e_e); the same on r. Unyielded it is the
             plain theta method; at th = 1/2 it keeps the elastic energy and yielding only dissipates
  update     at x^n+1: F's deviatoric log stretch held at the limit, the remainder scaled by beta, p_n+1 = x_j - x_i - beta r. No limit reached: F the total deformation, p the rest offset carried by the pair's mean gradient (1/2 (F_i + F_j) T up to (G_i - G_j)(F_i - F_j) T / 4 a step)
  connection the spring list, not the cut. A spring goes only when both cells have yielded and its wall is gone (its force went with the area); a new wall gets a spring when both have yielded, r = 0, otherwise it does not count: the cell treats it
             as a free surface, so nothing pulls across it. Springs without a wall stay in the cell's fit. Self contact is not in this file
Limit inf: the list and the total F never change, an elastic body. Small limit: F is about a rotation, p is reset to the flow, ul_plastic.py with a stiffer spring. No switch between them."""
import math
import time

import numpy as np
import torch
import warp as wp

from free_surface.closure_warp import MidpointStep, PS_GQ2, PS_PQ, PS_PZ, k_Aul, k_F8, k_Gul, k_Gul_adj, k_ball, k_ball_adj, k_mid_ball, k_mid_begin, k_mid_inv, k_mid_pre, k_mid_pre_ball
from free_surface.skin_warp import NSLOT, NT, k_total, k_total_loss
from free_surface.spring_plastic import k_Fs, k_Fs_adj
from free_surface.ul_plastic import PlasticUL, k_Aul_adj2, k_Ge, k_Ge_adj, k_Lb, k_gg_cell, k_gg_cell_adj, k_gg_wall, k_gg_wall_adj, k_var, polar_batch

f64 = torch.float64


@wp.kernel(enable_backward=False)
def k_Lc(G: wp.array(dtype=wp.mat33d), C: wp.array(dtype=wp.mat33d), Ln: wp.array(dtype=wp.float64), V0: wp.array(dtype=wp.float64), kap: wp.float64, L: wp.array(dtype=wp.float64),
         Lb: wp.array(dtype=wp.float64), terms: wp.array2d(dtype=wp.float64)):
    """L = L_n + tr(C^T G) and its energy; dE/dL into Lb"""
    b = wp.tid()
    Lc = Ln[b]+wp.trace(wp.transpose(C[b])@G[b])
    L[b] = Lc
    wp.atomic_add(terms, 1, b % NSLOT, V0[b]*wp.float64(0.5)*kap*Lc*Lc)
    wp.atomic_add(Lb, b, V0[b]*kap*Lc)


@wp.kernel(enable_backward=False)
def k_cell_shape(Fc: wp.array(dtype=wp.mat33d), Rh: wp.array(dtype=wp.mat33d), en: wp.array(dtype=wp.mat33d), V0: wp.array(dtype=wp.float64), mu: wp.float64, sy: wp.float64, th_: wp.float64,
                 ei: wp.array(dtype=wp.float64), be: wp.array(dtype=wp.float64), er: wp.array(dtype=wp.mat33d), Fb: wp.array(dtype=wp.mat33d), terms: wp.array2d(dtype=wp.float64)):
    """the cell's shape energy read at the step's end, beta and dE/dF (the springs add to Fb after)"""
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
    Fb[b] = V0[b]*(Rh[b]@((two_mu*ca)*en[b]+(th_*bb*two_mu)*r))


@wp.kernel(enable_backward=False)
def k_spring(x: wp.array(dtype=wp.vec3d), G: wp.array(dtype=wp.mat33d), si: wp.array(dtype=int), sj: wp.array(dtype=int), sP: wp.array(dtype=wp.vec3d), sdn: wp.array(dtype=wp.vec3d), sk: wp.array(dtype=wp.float64),
             sb: wp.array(dtype=wp.float64), th_: wp.float64, sr: wp.array(dtype=wp.vec3d), terms: wp.array2d(dtype=wp.float64), x_adj: wp.array(dtype=wp.vec3d), Gb: wp.array(dtype=wp.mat33d)):
    """the remainder at z, at the step's end into sr, and its force"""
    e = wp.tid()
    i, j = si[e], sj[e]
    if i == j:
        return
    half = wp.float64(0.5)
    p = sP[e]
    d = x[j]-x[i]-p-half*(G[i]@p+G[j]@p)
    ca = wp.float64(1.0)-th_
    dn = sdn[e]
    r = (d-ca*dn)/th_
    sr[e] = r
    k, b = sk[e], sb[e]
    wp.atomic_add(terms, 5, e % NSLOT, th_*th_*half*k*b*wp.dot(r, r)+ca*k*wp.dot(dn, d-dn)+(wp.float64(1.0)-th_*th_)*half*k*wp.dot(dn, dn))
    f = k*(ca*dn+(th_*b)*r)
    wp.atomic_add(x_adj, j, f)
    wp.atomic_add(x_adj, i, -f)
    PG = -half*wp.outer(f, p)
    wp.atomic_add(Gb, i, PG)
    wp.atomic_add(Gb, j, PG)


def _mk(W):
    """timing only (11-06, the assembly's blocks): a synchronised time mark when W._marks is a list"""
    mk = getattr(W, '_marks', None)
    if mk is not None:
        torch.cuda.synchronize(); mk.append(time.perf_counter())


class MergedUL(PlasticUL):
    """X, r, V0: the foam at rest; pairs, A, c: its walls (geometry.npz); E, nu, sigma_y per unit density (inf: never yields); theta as the step's set_theta. kshare scales the springs' k. Through MergedStep."""

    def __init__(self, X, r, V0, pairs, A, c, E, nu, sigma_y, theta=.5, kshare=1/6, grad='gg', agg=.1, vol_tol=1e-5, Kw=None, dev='cuda'):
        super().__init__(X, r, V0, E, nu, sigma_y, grad, agg, vol_tol, Kw, dev)
        assert theta in (.5, 1.), 'the plastic work is written for the midpoint and for backward Euler'
        n = self.n; X_ = self.X
        self.th = float(theta); self.kshare = float(kshare); self.ey = self.sy/(2*self.mu)
        p = torch.as_tensor(np.asarray(pairs).astype(np.int64), device=dev); A0 = torch.as_tensor(A, dtype=f64, device=dev)
        lo, hi = torch.minimum(p[:, 0], p[:, 1]), torch.maximum(p[:, 0], p[:, 1])
        self.tlo, self.thi = lo, hi
        d = X_[hi]-X_[lo]; L = d.norm(dim=1)
        self.tk0 = self.kshare*2*self.mu*A0/L; self.tkey = lo*n+hi
        self.tdl = torch.zeros_like(d)                                              # (the remainder kept across a step)
        o = torch.argsort(self.tkey)
        self.tlo, self.thi, self.tk0, self.tkey, self.tdl = lo[o], hi[o], self.tk0[o], self.tkey[o], self.tdl[o]
        self.restkey = self.tkey.clone()
        Ks = self.ul['Kw']
        z = lambda k, dt_: wp.zeros(k, dtype=dt_, device=dev)
        self.sp = dict(K=Ks, si=z(Ks, wp.int32), sj=z(Ks, wp.int32), sP=z(Ks, wp.vec3d), sdn=z(Ks, wp.vec3d), sk=z(Ks, wp.float64), sb=z(Ks, wp.float64), sr=z(Ks, wp.vec3d),
                       F0=z(n, wp.mat33d), Rh=z(n, wp.mat33d), en=z(n, wp.mat33d), Fend=z(n, wp.mat33d))
        tt = lambda a_: wp.to_torch(a_, requires_grad=False)
        self.tsp = {k_: tt(a_) for k_, a_ in self.sp.items() if k_ != 'K'}
        self.B = dict(G0=z(n, wp.mat33d), Ga=z(n, wp.mat33d), Ge=z(n, wp.mat33d), Sg=z(n, wp.mat33d), G=z(n, wp.mat33d), Gb=z(n, wp.mat33d), Gs=z(n, wp.mat33d), Gbe=z(n, wp.mat33d), Gb0=z(n, wp.mat33d),
                      L=z(n, wp.float64), Lb=z(n, wp.float64), ei=z(n, wp.float64), be=z(n, wp.float64), er=z(n, wp.mat33d), Fc=z(n, wp.mat33d), Fb=z(n, wp.mat33d), R=z(n, wp.mat33d),
                      Fp=z(n, wp.mat33d), Rp=z(n, wp.mat33d), qp=z(n, wp.vec4d), terms=wp.zeros((NT, 256), dtype=wp.float64, device=dev), tsum=wp.zeros(NT, dtype=wp.float64, device=dev),
                      loss=wp.zeros(1, dtype=wp.float64, device=dev), bdelta=z(n, wp.float64), bfn=z(n, wp.float64), bff=z(n, wp.vec3d))
        self.tFc, self.tRp, self.tG, self.tL, self.tei, self.tbe, self.ter, self.tR = (tt(self.B[k_]) for k_ in ('Fc', 'Rp', 'G', 'L', 'ei', 'be', 'er', 'R'))
        self.tbe.fill_(1.); self.tR.copy_(self.I3.expand(n, 3, 3))
        self._Fe = self.I3.expand(n, 3, 3).clone()
        self._yld = torch.zeros(n, dtype=torch.bool, device=dev)
        self.ul_stats.update(rows=0, born=0, closed=0, D_min=0., ghost=0, open_walls=0)

    def prepare(self):
        n, dev = self.n, self.dev
        self.tw = wp.array(np.ones(NT), dtype=wp.float64, device=dev)
        self.tones = wp.array(np.ones((NT, 256)), dtype=wp.float64, device=dev)
        self.xw = wp.zeros(n, dtype=wp.vec3d, device=dev, requires_grad=True)
        tt = lambda a_: wp.to_torch(a_, requires_grad=False)
        self.tx, self.tgx, self.tloss, self.tsum = tt(self.xw), tt(self.xw.grad), tt(self.B['loss']), tt(self.B['tsum'])
        self.tx.copy_(self.X)
        self.run()
        wp.capture_begin()
        self.run()
        self.graph = wp.capture_end()

    def run(self):
        """E and dE/dx at xw: the gradient, F, the cells, the springs, then the chain back through F and G"""
        B, U, S, n, Kw = self.B, self.ul, self.sp, self.n, self.ul['Kw']
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
        wp.launch(k_Lc, n, inputs=[B['G'], U['Q'], U['Ln'], U['V0'], kap], outputs=[B['L'], B['Lb'], B['terms']])
        wp.launch(k_var, Kw, inputs=[U['wi'], U['wj'], U['wv'], B['L']], outputs=[B['Lb'], B['terms']])
        wp.launch(k_cell_shape, n, inputs=[B['Fc'], S['Rh'], S['en'], U['V0'], mu, sy, th_], outputs=[B['ei'], B['be'], B['er'], B['Fb'], B['terms']])
        wp.launch(k_Fs_adj, n, inputs=[B['Fb'], S['F0']], outputs=[B['Gb']])
        wp.launch(k_spring, S['K'], inputs=[x, B['G'], S['si'], S['sj'], S['sP'], S['sdn'], S['sk'], S['sb'], th_], outputs=[S['sr'], B['terms'], g, B['Gb']])
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

    def update_ul(self, x):
        """the step's start at x: radii for V0 exp(L_n), the cut; the spring list; the fit and gradient on the connected walls; the springs' rows and the cells' start strains"""
        n = self.n; x = x.detach().clone(); st = self.ul_stats; t, s = self.tul, self.tsp; dev = x.device; yld = self._yld   # (x owned: the cut's sites G.p, read by the next start)
        J = torch.exp(t['Ln']); st['floored'] = int((J < .02).sum())
        Vt = self.V0*J.clamp(min=.02)
        # the radii's start: the last cut's radii, or (radii_start 'predict', 11-06) predicted to first order from the last cut G0
        # by this step's site motion and targets (Recut.predict: J dw = Vt - V - dvol(G0, x - p0)); only when G0 is the last
        # cut of this state (its gen: a restored snapshot keeps an older geometry object)
        r0, G0 = self._ulr, getattr(self, '_ulG', None)
        if getattr(self, 'radii_start', 'last') == 'predict' and G0 is not None and getattr(G0, 'gen', -1) == getattr(self, '_ul_gen', 0):
            r0 = self._recut.predict(G0, x-G0.p, Vt, rtol=getattr(self, 'radii_pred_rtol', 1e-3))
        try:
            r, G, S_, it, nrep = self._recut.solve_radii(x, r0, Vt, tol=self._ul_tol, r_base=self._rr0)
            st['radii_it'] = it
        except RuntimeError:
            st['radii_fail'] += 1
            r = self._ulr; G, S_ = self._recut.geometry(x, r)
        self._ul_gen = getattr(self, '_ul_gen', 0)+1; G.gen = self._ul_gen
        self._ulr, self._ulG = r, G
        _mk(self)
        mode = getattr(self, 'asm', 'torch')
        if mode == 'warp':
            self.asm_warp.run(self, x, r, G)
        elif mode == 'check':
            self.asm_warp.check(self, x, r, G)
        else:
            self._assemble_torch(x, r, G)
        Fe = self._Fe
        t['Q'].copy_(self.I3.expand(n, 3, 3))
        s['F0'].copy_(Fe)
        wp.launch(k_F8, n, inputs=[self.sp['F0']], outputs=[self.B['Fp'], self.B['Rp'], self.B['qp']])
        s['Rh'].copy_(self.tRp)
        A_ = self.tRp.transpose(1, 2)@Fe; e_ = .5*(A_+A_.transpose(1, 2))-self.I3
        s['en'].copy_(e_-(e_.diagonal(dim1=1, dim2=2).sum(1)/3)[:, None, None]*self.I3)
        _mk(self)

    def _assemble_torch(self, x, r, G):
        """the cut's assembly as before 11-06 (torch): the spring list, the connected walls, the fit, the aggregation, the
        wall and cell data (update_ul's body from the springs to the walls' buffers)"""
        n = self.n; st = self.ul_stats; t, s = self.tul, self.tsp; dev = x.device; yld = self._yld
        kp = G.wall_A.detach() > 0
        i, j, A, c = G.i[kp].long(), G.j[kp].long(), G.wall_A.detach()[kp], G.wall_c.detach()[kp]
        lo, hi = torch.minimum(i, j), torch.maximum(i, j); key = lo*n+hi
        # the spring list: a new wall with both cells yielded gets one; a spring whose cells have both yielded goes with its wall
        both_w = yld[lo] & yld[hi]
        new = both_w & ~torch.isin(key, self.tkey)
        both_s = yld[self.tlo] & yld[self.thi]
        stay = ~both_s | torch.isin(self.tkey, key)
        st['closed'] = int((~stay).sum()); st['born'] = int(new.sum())
        lo_s = torch.cat([self.tlo[stay], lo[new]]); hi_s = torch.cat([self.thi[stay], hi[new]])
        ds = x[hi_s]-x[lo_s]
        nnew = int(new.sum())
        dl_ = torch.cat([self.tdl[stay], torch.zeros(nnew, 3, dtype=f64, device=dev)])
        k0_ = torch.cat([self.tk0[stay], torch.zeros(nnew, dtype=f64, device=dev)])
        key_s = lo_s*n+hi_s
        o = torch.argsort(key_s)
        lo_s, hi_s, ds, dl_, k0_, key_s = lo_s[o], hi_s[o], ds[o], dl_[o], k0_[o], key_s[o]
        self.tlo, self.thi, self.tkey = lo_s, hi_s, key_s
        m_s = len(key_s)
        if m_s > self.sp['K']:
            raise RuntimeError('update_ul: %d springs over the capacity %d' % (m_s, self.sp['K']))
        both_y = yld[lo_s] & yld[hi_s]
        p_ = ds-dl_                                                                 # (the offset that reads the kept remainder at x; a new spring's r = 0)
        # the connected walls, by their pair; k of the current wall when both cells have yielded
        pos = torch.searchsorted(key_s, key).clamp(max=m_s-1); conn = key_s[pos] == key
        order = torch.argsort(key); pw = torch.searchsorted(key[order], key_s).clamp(max=len(key)-1); has_wall = key[order][pw] == key_s
        Aw = torch.zeros(m_s, dtype=f64, device=dev); Aw[has_wall] = A[order][pw[has_wall]]
        k_ = k0_.clone(); q_ = both_y & has_wall
        Ls_ = ds.norm(dim=1)
        k_[q_] = self.kshare*2*self.mu*Aw[q_]/Ls_[q_]
        self.tP, self.tdl, self.tk0 = p_, dl_.clone(), k_.clone()
        for k2 in ('si', 'sj', 'sP', 'sdn', 'sk'):
            s[k2].zero_()
        s['si'][:m_s] = lo_s.to(torch.int32); s['sj'][:m_s] = hi_s.to(torch.int32); s['sP'][:m_s] = p_; s['sdn'][:m_s] = dl_; s['sk'][:m_s] = k_; s['sb'].fill_(1.)
        self._m_s = m_s
        _mk(self)
        ic, jc, Ac, cc = i[conn], j[conn], A[conn], c[conn]
        m = len(ic); U = self.ul
        if m > U['Kw']:
            raise RuntimeError('update_ul: %d walls over the capacity %d' % (m, U['Kw']))
        st['walls'] = m; st['open_walls'] = int((~conn).sum()); st['rows'] = m_s
        V = G.V.detach(); self._V = V
        d = x[jc]-x[ic]; Lw = d.norm(dim=1); nn = d/Lw[:, None]; w = Ac/Lw
        # the fit: the connected walls, and the springs without a wall at their rest weight
        gh = ~has_wall
        st['ghost'] = int(gh.sum())
        _mk(self)
        li, lj, dd, ww = torch.cat([ic, lo_s[gh]]), torch.cat([jc, hi_s[gh]]), torch.cat([d, ds[gh]]), torch.cat([w, k_[gh]/(self.kshare*2*self.mu)])
        q = ww[:, None, None]*dd[:, :, None]*dd[:, None, :]
        M = torch.zeros(n, 3, 3, dtype=f64, device=dev).index_add(0, li, q).index_add(0, lj, q)
        a0, a1, a2 = M[:, :, 0], M[:, :, 1], M[:, :, 2]; c0, c1, c2 = torch.linalg.cross(a1, a2), torch.linalg.cross(a2, a0), torch.linalg.cross(a0, a1)
        det = (a0*c0).sum(1); sing = det.abs() <= 1e-10*(M*M).sum((1, 2))**1.5
        Mi = torch.stack([c0, c1, c2], 1)/torch.where(sing, torch.ones_like(det), det)[:, None, None]
        if bool(sing.any()):
            Mi[sing] = torch.linalg.pinv(M[sing])
        vd = ww[:, None]*dd
        Ri = torch.einsum('fa,fab->fb', vd, Mi[li]); Rj = torch.einsum('fa,fab->fb', -vd, Mi[lj])
        ml = len(li)
        if ml > U['Kw']:
            raise RuntimeError('update_ul: %d fit entries over the capacity %d' % (ml, U['Kw']))
        t['S'].zero_(); t['O'].zero_(); t['R'].zero_()
        t['S'][:2*ml] = torch.cat([li, lj]).to(torch.int32); t['O'][:2*ml] = torch.cat([lj, li]).to(torch.int32); t['R'][:2*ml] = torch.cat([Ri, Rj])
        self._ml = ml
        _mk(self)
        tiny = self.V0 < self._ul_agg*self.V0.median()
        m1, m2 = tiny[ic] & ~tiny[jc], tiny[jc] & ~tiny[ic]
        as_, ao_ = torch.cat([ic[m1], jc[m2]]), torch.cat([jc[m1], ic[m2]])
        va = self.V0[ao_]
        den = torch.zeros(n, dtype=f64, device=dev).index_add(0, as_, va)
        t['As'].zero_(); t['Ao'].zero_(); t['Aw'].zero_()
        k3 = len(as_); t['As'][:k3] = as_.to(torch.int32); t['Ao'][:k3] = ao_.to(torch.int32); t['Aw'][:k3] = va/den[as_]
        self._k3 = k3                                                              # (the aggregation's live length, for k1_follow_fast; nothing else reads it)
        t['used'].copy_((tiny & (den > 0)).to(torch.int32))
        _mk(self)
        tw = ((cc-x[ic])*nn).sum(1)/Lw
        cap = G.patch_A.detach() > 0
        openc = cap.clone()                                                         # (a wall that does not count is a free surface to its cells)
        if bool((~conn).any()):
            openc[i[~conn]] = True; openc[j[~conn]] = True
        a_ = torch.zeros(n, 3, dtype=f64, device=dev).index_add(0, ic, -Ac[:, None]*nn).index_add(0, jc, Ac[:, None]*nn)
        Mw = (torch.zeros(n, 3, 3, dtype=f64, device=dev).index_add(0, ic, Ac[:, None, None]*(cc-x[ic])[:, :, None]*nn[:, None, :])
              .index_add(0, jc, -Ac[:, None, None]*(cc-x[jc])[:, :, None]*nn[:, None, :]))
        Mc = V[:, None, None]*self.I3-Mw
        Mc[~openc] = 0.; a_[~openc] = 0.
        for k2 in ('wi', 'wj', 'wa', 'wn', 'wt', 'wtau', 'wv'):
            t[k2].zero_()
        t['wi'][:m] = ic.to(torch.int32); t['wj'][:m] = jc.to(torch.int32); t['wa'][:m] = Ac; t['wn'][:m] = nn; t['wt'][:m] = tw
        t['wtau'][:m] = cc-(x[ic]+tw[:, None]*d); t['wv'][:m] = self.kap*Ac*Lw/24
        t['a'].copy_(a_); t['Mc'].copy_(Mc); t['Vi'].copy_(1/V.clamp(min=1e-12*float(V.median())))
        t['xn'].copy_(x)
        if self.ball is not None:
            self.tbR.copy_(torch.where(cap, r, -torch.ones_like(r)))
        self._w = (ic, jc, Lw, key, None, m)
        _mk(self)

    def reread(self, beta=False, keep_C=False):
        """after an evaluation at the point the solve stands on: R and Q read there, frozen for the next solve; beta: the springs' beta too, once per step at Newton's start (10-31: read at every accepted point it moved
        the target: each solve's ratio 1, |g| after the re-read jumping between 0.2 and 3 weights, no trend, on 300 Newton iterations)"""
        s, m = self.tsp, self._m_s
        wp.launch(k_F8, self.n, inputs=[self.B['Fc']], outputs=[self.B['Fp'], self.B['Rp'], self.B['qp']])
        s['Rh'].copy_(self.tRp)
        if keep_C:                                                                  # (k1_lag: C = F_rel(x_n)^-T set by the rebase, kept)
            if beta:
                s['sb'][:m] = .5*(self.tbe[self.tlo]+self.tbe[self.thi])
            return
        self.tul['Q'].copy_(self.C_at())
        if beta:
            s['sb'][:m] = .5*(self.tbe[self.tlo]+self.tbe[self.thi])

    def C_at(self):
        """C = (I + G / 2 theta)^-T at the last evaluation (G in tG); a cell flipped there reads the plain trace (C = I)"""
        M = self.I3+self.tG/(2*self.th)
        a0, a1, a2 = M[:, :, 0], M[:, :, 1], M[:, :, 2]; c0, c1, c2 = torch.linalg.cross(a1, a2), torch.linalg.cross(a2, a0), torch.linalg.cross(a0, a1)
        det = (a0*c0).sum(1); ok = det > 1e-3
        return torch.where(ok[:, None, None], torch.stack([c0, c1, c2], 2)/torch.where(ok, det, torch.ones_like(det))[:, None, None], self.I3)

    def end_ul(self, th=.5, Fri=None):
        """after the step (the buffers at the accepted z; x -> F linear, so the end's F and G are exact): the cells' returns, the springs' remainders, F_n+1, L_n+1, the plastic work.
        end_mode 'warp' (11-06): end_warp.EndWarp, the same update on the device, its bookkeeping read by flush; 'check': both, compared"""
        mode = getattr(self, 'end_mode', 'torch')
        if Fri is not None:
            assert mode == 'warp', 'the cut kept over steps (k1_lag) needs end_warp'
        if mode == 'warp':
            return self.end_warp.run(self, th, Fri)
        if mode == 'check':
            self.end_check_left = getattr(self, 'end_check_left', 1)-1
            if self.end_check_left <= 0:
                self.end_mode = 'warp'
            return self.end_warp.check(self, th)
        return self._end_torch(th)

    def _end_torch(self, th):
        assert th == self.th, 'the model and the step on one theta'
        t, s, n, I3, m = self.tul, self.tsp, self.n, self.I3, self._m_s
        Gs = self.tG/th
        Q = polar_batch(I3+Gs)
        A = Q.transpose(1, 2)@(I3+Gs); H = .5*(A+A.transpose(1, 2))-I3
        be = self.tbe.clone(); yn = be < 1
        sb, sk, sr, sdn = s['sb'][:m], s['sk'][:m], s['sr'][:m], s['sdn'][:m]
        rr = (sr*sr).sum(1)
        if th == 1.:
            wsp = .5*sk*sb*(1-sb)*rr
        else:
            wsp = .5*sk*(1-sb)*((sr*sdn).sum(1)+sb*rr)
        self.Wp_spring = getattr(self, 'Wp_spring', 0.)+float(wsp.sum())
        self.Wp += float(wsp.sum())
        if bool(yn.any()):
            er, en = self.ter, s['en']
            if th == 1.:
                wc = (self.V0*self.sy*(self.tei-self.ey))[yn]
            else:
                wc = (self.V0*self.mu*(1-be)*((en*er).sum((1, 2))+be*(er*er).sum((1, 2))))[yn]
            self.Wp += float(wc.sum()); self.Wp_cell += float(wc.sum())
            self.ul_stats['D_min'] = float((wc/(self.V0[yn]*self.mu*self.ey**2)).min())
        self.tdl = sb[:, None]*sr
        self.spring_store = float((.5*sk*(self.tdl*self.tdl).sum(1)).sum())
        new = self._yld | yn
        s['Fend'].copy_((self.tFc-(1-th)*s['F0'])/th)
        wp.launch(k_F8, n, inputs=[self.sp['Fend']], outputs=[self.B['Fp'], self.B['Rp'], self.B['qp']])
        Fe, R = s['Fend'].clone(), self.tRp.clone()
        self.ep = Q@self.ep@Q.transpose(1, 2)
        if bool(yn.any()):
            A_ = R[yn].transpose(1, 2)@Fe[yn]
            w, v = torch.linalg.eigh((.5*(A_+A_.transpose(1, 2))).cpu())           # (on the CPU: on the GPU torch's reserved memory grew with the count squared, 25 GB at 3800 cells)
            w, v = w.to(A_.device), v.to(A_.device)
            lw = torch.log(w.clamp(min=1e-6)); lm = lw.mean(1, keepdim=True); dv = lw-lm
            nr = dv.norm(dim=1, keepdim=True).clamp(min=1e-300); a = torch.exp(lm)
            sc = torch.minimum(torch.ones_like(nr), (self.ey/a)/nr)                 # (dev(U - I) ~ a c: held at e_y)
            Ue = v@torch.diag_embed(torch.exp(lm+sc*dv))@v.transpose(1, 2)
            Fe[yn] = R[yn]@Ue
            self.ep[yn] = self.ep[yn]+R[yn]@(v@torch.diag_embed((1-sc)*dv)@v.transpose(1, 2))@R[yn].transpose(1, 2)
        self._Fe = Fe
        self._yld = new
        self.Hsum += (self.V0[:, None, None]*H).sum(0)/self.V0.sum()
        self.ul_stats['flips'] = int((torch.linalg.det(I3+Gs) <= 1e-3).sum()); self.ul_stats['yielding'] = int(yn.sum()); self.ul_stats['yielded'] = int(new.sum())
        t['Ln'].add_((t['Q']*self.tG).sum((1, 2))/th)

    def stored(self):
        """the elastic energy after the last end_ul: shape, on the F kept; volume"""
        Fe = self._Fe
        wp.launch(k_F8, self.n, inputs=[wp.from_torch(Fe.contiguous(), dtype=wp.mat33d)], outputs=[self.B['Fp'], self.B['Rp'], self.B['qp']])
        A_ = self.tRp.transpose(1, 2)@Fe; e_ = .5*(A_+A_.transpose(1, 2))-self.I3
        e_ = e_-(e_.diagonal(dim1=1, dim2=2).sum(1)/3)[:, None, None]*self.I3
        return float((self.V0*self.mu*(e_*e_).sum((1, 2))).sum()), float((self.V0*.5*self.kap*self.tul['Ln']**2).sum())

    def ul_blocks(self, lr_thr=None, lr_cap=0):
        """the diagonal blocks for the preconditioner, at the step's start: the cell's through the gradient's direct weights (shape at the return factor, volume), the fit's only through M_s; beta k on each spring's two ends"""
        assert lr_thr is None, 'no low-rank term here'
        n, t, s, dev = self.n, self.tul, self.tsp, self.dev
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
        mut = self.mu*self.tbe
        rr = r[:, :, None]*r[:, None, :]; r2 = (r*r).sum(1)
        cr = torch.einsum('fab,fb->fa', t['Q'][sc], r)                              # (dL/du = C r)
        K = torch.zeros(n, 3, 3, dtype=f64, device=dev)
        K.index_add_(0, kc, (self.V0[sc]*mut[sc])[:, None, None]*(r2[:, None, None]*self.I3+rr/3)+(self.V0[sc]*self.kap)[:, None, None]*(cr[:, :, None]*cr[:, None, :]))
        ms = self._m_s
        kb = (s['sk'][:ms]*s['sb'][:ms])[:, None, None]*self.I3
        K.index_add_(0, self.tlo, kb); K.index_add_(0, self.thi, kb)
        return K


class MergedStep(MidpointStep):
    """MidpointStep's Newton with R, Q and beta read once, at Newton's start, frozen for the step: the step is one convex problem. Read again at each accepted point they moved the target: (10-31 landing) each solve's ratio 1 but |g| after the
    re-read without a trend; with beta frozen a cell whose I + G has singular values 0.09 and 4.6 in a step kept jumping |g| to 160 weights every 10 iterations. The state stays the total F: the frozen R only reads it within the step"""

    c_two_pass = False                                                              # (11-06, the expert's single-variable check: C read again at the first solve's root, R and beta kept)
    slide, slide_thr, slide_max, slide_delta = False, .25, 6, 0.                    # (the sliding step at the wall: on when the cut is below slide_thr, up to slide_max active cells, |y|_M <= slide_delta)
    flip_guard = 0.                                                                 # (> 0: the step's end F must keep det(I + dG) >= this on every cell; the Newton step shortened to it)
    c1_identity = False                                                             # (pass 1's C = I)
    patch_check_py = False                                                          # (the first trigger also solved by the python path, compared)
    patch_cg, patch_ncell, patch_eta, patch_log = False, 6, .005, print                 # (11-06: the stalled-CG fallback on, its group size, its stop, where it logs)
    c2_refresh = False                                                              # (11-06: pass 2 rebuilds the preconditioner's blocks with C2)
    pre_force = False                                                               # (replay arm: the blocks rebuilt at the step's start whatever their age)
    newton_log = False                                                              # (11-06 logging only: per Newton iteration the region, the reductions, the CG's exit, the true linear residual, the step)

    def solve(self, tol_abs, cg_tol_abs):
        W, A, bl, n = self.W, self.a, self.W.ball, self.W.n
        wp.launch(k_mid_begin, n, inputs=[A['x'], A['v'], A['a'], self.dt, self.th], outputs=[A['xt'], A['z']])
        if self.smooth_pred and self._vp is not None:
            h_ = self.th*self.dt
            self.tz.copy_(self.tx+.5*h_*(self.tv+self._vp)+.5*(.5*h_*self.dt+h_*h_)*(self.ta+self._ap))
        wp.launch(k_mid_ball, bl['k'], inputs=[A['x'], bl['body'], bl['R'], bl['ng'], self.c0, self.g8, self.cb], outputs=[bl['xn'], bl['kt'], bl['bound']])
        if W.stick:                                                                 # (the glue's limit: mu F_n as k_mid_ball wrote it, up to the material's shear yield on the cell's plan view)
            W.tbbound.copy_(torch.minimum(W.tbbound, W.tau_y*W.footprint(self.tx, W._ulr)))
        self._eval(0.)
        self._guard_pred = 0
        if self.flip_guard > 0:                                                     # (11-07: the predictor itself must be feasible: G(x_n) = 0 on the fresh cut and G is linear in x, so
            A1_ = W.I3+W.tG/self.th                                                 # det(I + t G(z) / theta) on the cubic; z moved to x_n + t (z - x_n) if needed, then evaluated again)
            if float(torch.linalg.det(A1_).min()) < self.flip_guard:
                self._guard_pred = .9*self._first_crossing(W.I3.expand_as(W.tG), W.tG/self.th, self.flip_guard)
                self.tz.copy_(self.tx+self._guard_pred*(self.tz-self.tx))
                self._eval(0.)
        W.reread(beta=True)
        if self.c1_identity:                                                        # (11-07: pass 1 on C read at the step's start, G(x_n) = 0 on the fresh cut, so C = I; R, beta as read; pass 2 reads C at the root)
            W.tul['Q'].copy_(W.I3.expand_as(W.tul['Q']))
        g2 = self._eval(0.)[PS_GQ2]
        self._accept()
        if getattr(self, 'probe', None) is not None:                                # (logging only: it leaves the last evaluation and accept as here)
            self.probe(self)
        st = dict(rejected=0, edge=0, negative=0, trace=[], nlog=[] if self.newton_log else None)
        if self._guard_pred:
            st['guard_pred'] = self._guard_pred
        self._nst = getattr(self, '_nst', 0)+1                                      # (steps solved: the blocks' age is counted in them)
        if self.pre_force and self.pre_ul:                                          # (replay arm: the blocks rebuilt now, at this state and C1, whatever their age)
            self._pre_age = self.pre_every
            self.tK.copy_(W.ul_blocks()); self._blk_at = self._nst
        st['blk_age'] = self._nst-getattr(self, '_blk_at', -10**9) if self._pre_age > 0 or self.pre_force else None   # (None: rebuilt at this step's first iteration, if it has one)
        delta = self.delta_start if self.carry_delta and self.delta_start else -1.
        evals, its, g2, delta = self._newton(g2, delta, tol_abs, cg_tol_abs, st, 2, 0, 1)
        st['its1'], st['g1'] = its, math.sqrt(max(g2, 0.))
        if self.c_two_pass:
            if not math.sqrt(g2) <= tol_abs:                                        # (11-06 expert: no second pass on an unconverged first one; the step fails as is, nothing committed by the driver)
                st['pass1_failed'] = True
            else:
                evals, its, g2, delta = self.pass2(tol_abs, cg_tol_abs, st, evals, its, delta, self.c2_refresh)
        self.stats = st
        self.delta_start = delta if delta > 0. else None
        return evals, its, g2

    def pass2(self, tol_abs, cg_tol_abs, st, evals, its, delta, refresh):
        """pass 2 from the first pass's root z1 (W's buffers hold its evaluation): only C read again there, R and beta as read at the predictor; refresh: the
        preconditioner's blocks rebuilt with C2 (they hold kappa (C r)(C r)^T); the step's end (L_n+1 = L_n + C : dG, the plastic work) then on this C"""
        W = self.W
        Rh0, sb0 = W.tsp['Rh'].clone(), W.tsp['sb'][:W._m_s].clone()
        C1 = W.tul['Q'].clone(); C2 = W.C_at()
        its1, ev1, g1 = its, evals, st['g1']
        W.tul['Q'].copy_(C2)
        if refresh and self.pre_ul:
            self.tK.copy_(W.ul_blocks()); self._blk_at = self._nst
        g2 = self._eval(0.)[PS_GQ2]; evals += 1
        self._accept()
        g2s = g2
        evals, its, g2, delta = self._newton(g2, delta, tol_abs, cg_tol_abs, st, evals, max(its, 1), 2)
        its = its if its1 > 0 else its-1
        C3 = W.C_at()                                                               # (C at the second pass's root: what is still frozen)
        d12 = (C2-C1).flatten(1).abs().max(1).values; d23 = (C3-C2).flatten(1).abs().max(1).values
        st['two_pass'] = dict(its1=its1, evals1=ev1, g1=g1, g_start2=math.sqrt(max(g2s, 0.)), its2=its-its1, evals2=evals-ev1, refreshed=bool(refresh),
                              dC12=float(d12.max()), dC12_cell=int(d12.argmax()), dC23=float(d23.max()), dC23_cell=int(d23.argmax()),
                              dC23_q99=float(torch.quantile(d23, .99)), Rh_same=bool(torch.equal(Rh0, W.tsp['Rh'])), sb_same=bool(torch.equal(sb0, W.tsp['sb'][:W._m_s])))
        return evals, its, g2, delta

    @staticmethod
    def _first_crossing(A0, D, m, bis=60):
        return MergedStep._first_crossing_cell(A0, D, m, bis)[0]

    @staticmethod
    def _first_crossing_cell(A0, D, m, bis=60):
        """the first t in (0, 1] at which min_k det(A0_k + t D_k) falls below m (1. if never). Per cell det(A0 + t D) is a cubic in t: its
        coefficients from four samples, its derivative's roots split [0, 1] into monotone intervals, the first interval entered above m and
        left below it is bisected on the polynomial; the smallest crossing over the cells"""
        dt, dev = A0.dtype, A0.device
        ts = torch.tensor([0., 1/3, 2/3, 1.], dtype=dt, device=dev)
        f = torch.linalg.det(A0[None]+ts[:, None, None, None]*D[None]).T-m           # (n, 4)
        f = f-torch.clamp(f[:, :1], max=0.)*(1+1e-9)                                  # (a cell at or marginally below the margin at t = 0: its crossing is falling below its own det there)
        V = torch.stack([ts**k for k in range(4)], 1)                                # (4, 4) Vandermonde
        c = torch.linalg.solve(V, f.T).T                                             # (n, 4): c0 + c1 t + c2 t^2 + c3 t^3
        n = c.shape[0]
        # stationary points of the cubic: 3 c3 t^2 + 2 c2 t + c1 = 0
        a2, a1, a0 = 3*c[:, 3], 2*c[:, 2], c[:, 1]
        disc = a1*a1-4*a2*a0
        quad = a2.abs() > 1e-300
        sq = torch.sqrt(disc.clamp(min=0.))
        r1 = torch.where(quad, (-a1-sq)/torch.where(quad, 2*a2, torch.ones_like(a2)), torch.where(a1.abs() > 1e-300, -a0/torch.where(a1.abs() > 1e-300, a1, torch.ones_like(a1)), torch.full_like(a0, 2.)))
        r2 = torch.where(quad & (disc >= 0), (-a1+sq)/torch.where(quad, 2*a2, torch.ones_like(a2)), torch.full_like(a0, 2.))
        r1 = torch.where((disc >= 0) | ~quad, r1, torch.full_like(r1, 2.))
        bp = torch.stack([torch.zeros(n, dtype=dt, device=dev), r1.clamp(0., 1.), r2.clamp(0., 1.), torch.ones(n, dtype=dt, device=dev)], 1).sort(1).values   # (n, 4) breakpoints
        poly = lambda t: ((c[:, 3]*t+c[:, 2])*t+c[:, 1])*t+c[:, 0]
        best = torch.full((n,), 1., dtype=dt, device=dev)
        found = torch.zeros(n, dtype=torch.bool, device=dev)
        for j in range(3):
            lo, hi = bp[:, j].clone(), bp[:, j+1].clone()
            cand = (~found) & (hi > lo) & (poly(lo) >= 0) & (poly(hi) < 0)          # (monotone on the interval: one crossing)
            if not bool(cand.any()):
                continue
            l_, h_ = lo.clone(), hi.clone()
            for _ in range(bis):
                mid = .5*(l_+h_); ok = poly(mid) >= 0
                l_ = torch.where(ok, mid, l_); h_ = torch.where(ok, h_, mid)
            best = torch.where(cand, l_, best); found = found | cand
        if not bool(found.any()):
            return 1., -1
        k = int(torch.where(found, best, torch.full_like(best, 2.)).argmin())
        return float(best[k]), k

    def _det_grad(self, c, z):
        """d det(I + G_c / theta) / dx on the cell's 2-hop wall stencil, central differences (the test version; analytic later)"""
        W = self.W; dev = W.dev; th = self.th; m = W._w[5]
        wi, wj = W.tul['wi'][:m].long(), W.tul['wj'][:m].long()
        nb1 = torch.unique(torch.cat([wj[wi == c], wi[wj == c], torch.tensor([c], device=dev)]))
        sel = torch.zeros(W.n, dtype=torch.bool, device=dev); sel[nb1] = True
        nb2 = torch.unique(torch.cat([wj[sel[wi]], wi[sel[wj]], nb1]))
        xw = wp.to_torch(W.xw, requires_grad=False); h = 1e-7
        nvec = torch.zeros(W.n, 3, dtype=f64, device=dev); evals = 0
        detc = lambda: float(torch.det(W.I3+W.tG[c]/th))
        for k in nb2.tolist():
            for a in range(3):
                xp = z.clone(); xp[k, a] += h; xw.copy_(xp); W._run(); dp = detc()
                xm = z.clone(); xm[k, a] -= h; xw.copy_(xm); W._run(); dm = detc()
                nvec[k, a] = (dp-dm)/(2*h); evals += 2
        return nvec, evals

    def _slide(self, st, m, cblk, pmp_full, evals):
        """a step along the wall from z (the buffers at z's evaluation): the active cells = the cell that blocked the Newton step and the cells with det
        within 1.5 m at z; their det gradients orthonormalised; the preconditioned gradient projected onto their tangent, scaled to the blocked
        Newton step's M-norm, cut at the first crossing, Armijo on the energy; accepted into z"""
        W, A, CG = self.W, self.a, self.CG; dev = W.dev; th = self.th
        z = self.tz.clone(); gz = wp.to_torch(A['gz'], requires_grad=False).clone()
        xw = wp.to_torch(W.xw, requires_grad=False)
        G0 = W.tG.clone(); A0 = W.I3+G0/th
        d0 = torch.linalg.det(A0)
        act = sorted(set(torch.nonzero(d0 < 1.5*m).flatten().tolist()+([cblk] if cblk >= 0 else [])))[:self.slide_max]
        N, raw = [], {}
        for c in act:
            nv, e_ = self._det_grad(c, z); evals += e_; raw[c] = nv.clone()
            for q in N:
                nv = nv-float((nv*q).sum())*q
            nn = float(nv.norm())
            if nn > 1e-300:
                N.append(nv/nn)
        xw.copy_(z); W._run()
        gt = gz.clone()
        for q in N:
            gt = gt-float((gt*q).sum())*q
        Pinv = CG.t['Pinv']; y = -torch.einsum('kab,kb->ka', Pinv, gt)
        for q in N:
            y = y-float((y*q).sum())*q
        slope = float((gz*y).sum())
        lift = []
        for c in act:                                                              # (first order along each active cell's raw det gradient: det back to 1.2 m, so the step leaves the wall)
            nv = raw[c]
            need = 1.2*m-float(d0[c])
            if need > 0:
                lift.append(nv*(need/max(float((nv*nv).sum()), 1e-300)))
        if lift:
            y = y+sum(lift)
        st['slide_lift'] = st.get('slide_lift', 0)+len(lift)
        if not slope < 0.:
            st['slide_fail'] = st.get('slide_fail', 0)+1
            return dict(taken=False, evals=evals)
        P = wp.to_torch(A['P'], requires_grad=False)
        yM = math.sqrt(float((y*torch.einsum('kab,kb->ka', P, y)).sum()))
        L = pmp_full if self.slide_delta <= 0 else min(pmp_full, self.slide_delta)
        sc = L/max(yM, 1e-300); y = y*sc; slope *= sc
        xw.copy_(z+y); W._run(); D = (W.tG-G0)/th; evals += 1
        tc, cblk2 = self._first_crossing_cell(A0, D, m)
        t = min(1., .9*tc)
        dlin = {c: float((raw[c]*y).sum()) for c in act}                           # (first-order det change of each active cell along y: should be >= 0 after the lift)
        dact = {c: float(torch.det(A0[c]+1e-3*D[c]))-float(d0[c]) for c in act}    # (the actual change at t = 1e-3, from the cubic)
        wp.to_torch(CG.a['p'], requires_grad=False).copy_(y)
        e0 = self._eval(0.)[PS_PZ]; evals += 1
        taken = False
        for _ in range(12):
            s_ = self._eval(t); evals += 1
            dmin = float(torch.linalg.det(W.I3+W.tG/th).min())
            if math.isfinite(s_[PS_PQ]) and dmin >= m*(1-1e-6) and s_[PS_PQ]-e0 <= 1e-4*t*slope:
                taken = True; break
            t *= .5
        st['slide'] = st.get('slide', 0)+1
        st.setdefault('slide_log', []).append((len(act), len(N), slope, tc, t, (s_[PS_PQ]-e0) if taken else float('nan'), taken, math.sqrt(max(s_[PS_GQ2], 0.)) if taken else float('nan'),
                                               cblk2, float(d0[cblk2]) if cblk2 >= 0 else float('nan'), act[0], dlin[act[0]], dact[act[0]], float(d0[act[0]])))
        if taken:
            self._accept()
            return dict(taken=True, evals=evals, g2=s_[PS_GQ2])
        st['slide_fail'] = st.get('slide_fail', 0)+1
        self._eval(0.); evals += 1; self._accept()
        return dict(taken=False, evals=evals)

    def _fresh_Hp(self):
        """logging only: H p by one new product (as TrustCG's iteration: d = p, the finite difference at z), not the CG's accumulated Hp; W's
        buffers left at z + h p / |p|_max (the next evaluation overwrites them; MergedUL's gradient keeps no state between calls)"""
        from free_surface.closure_warp import k_tr_absmax, k_tr_perturb, k_tr_hd, TR_PMAX, TR_DHD
        CG, W = self.CG, self.W; A, t, n = CG.a, CG.t, W.n
        t['d'].copy_(t['p']); t['tr'][TR_PMAX] = 0.; t['tr'][TR_DHD] = 0.
        wp.launch(k_tr_absmax, n, inputs=[A['d']], outputs=[A['tr']])
        wp.launch(k_tr_perturb, n, inputs=[A['z'], A['d'], A['tr'], CG.h], outputs=[W.xw])
        W._run()
        wp.launch(k_tr_hd, n, inputs=[W.xw.grad, A['g0'], A['d'], A['cm'], A['tr'], CG.h], outputs=[A['Hd']])
        return t['Hd']

    def _newton(self, g2, delta, tol_abs, cg_tol_abs, st, evals, its, pass_):
        W, A, bl, n = self.W, self.a, self.W.ball, self.W.n
        delta0, its0 = None, its
        krylov = None
        if getattr(self, 'krylov_tr', False):
            if self.flip_guard > 0 or self.slide:
                raise ValueError('krylov-tr currently supports the unconstrained production path')
            if getattr(self, '_krylov', None) is None:
                if getattr(self, 'krylov_gpu', False):
                    from free_surface.k1_krylov_device import DeviceKrylovTR
                    self._krylov = DeviceKrylovTR(self)
                else:
                    from free_surface.k1_krylov import KrylovTR
                    self._krylov = KrylovTR(self)
            krylov = self._krylov
            krylov.invalidate()  # new pass, including a possible C reread
        while its-its0 < self.newton_max:
            if math.sqrt(g2) <= tol_abs:
                break
            its += 1
            if its == 1 and self.pre_ul and self._pre_age <= 0:
                self._pre_age = self.pre_every
                self.tK.copy_(W.ul_blocks()); self._blk_at = getattr(self, '_nst', 0); st['blk_age'] = 0
            G0_ = W.tG.clone() if self.flip_guard > 0 else None                    # (G at z: the buffers hold z's evaluation here, before the CG's products perturb them)
            if krylov is None or not krylov.valid:
                wp.launch(k_mid_pre, n, inputs=[W.B['R'], A['K'], self.c1, A['cm']], outputs=[A['P']])
                wp.launch(k_mid_pre_ball, bl['k'], inputs=[bl['body'], bl['R'], W.B['bdelta'], bl['ng'], self.e4, bl['kt'], bl['bound'], self.muf], outputs=[A['P']])
                wp.launch(k_mid_inv, n, inputs=[A['P']], outputs=[A['Pinv']])
            stop = max(cg_tol_abs(g2), self.eta*math.sqrt(g2)); g_ = math.sqrt(g2)
            r = (self.CG.solve if krylov is None else krylov.solve)(A['gz'], delta, stop, self.cg_cap)
            evals += r['K']; st['edge'] += r['EDGE']; st['negative'] += r['NEG']; st['cg_std'] = st.get('cg_std', 0)+r['K']   # (the standard CG's products, the capped ones included)
            if krylov is None and self.patch_cg and r['K'] >= self.cg_cap and not r['EDGE'] and not r['NEG']:   # (11-06: the stalled-CG fallback, k1_patchcg: the same subproblem again, at this point)
                from free_surface.k1_patchcg import solve as patch_solve
                rp = patch_solve(self, A['gz'], delta, self.cg_cap, self.patch_log, ncell=self.patch_ncell, eta=self.patch_eta, tag=' pass %d it %d' % (pass_, its), check_py=self.patch_check_py)
                self.patch_check_py = self.patch_check_py and getattr(self, 'patch_check_all', False)   # (once, or every trigger in a replay)
                st['patch'] = st.get('patch', 0)+1; st['patch_evals'] = st.get('patch_evals', 0)+rp['evals']; st['patch_time'] = st.get('patch_time', 0.)+rp['time']
                st['patch_checks'] = st.get('patch_checks', 0)+rp['evals_checks']; st['patch_time_checks'] = st.get('patch_time_checks', 0.)+rp['time_checks']
                st['patch_res'] = st.get('patch_res', [])+[(rp['res_split'], rp['res_cent'])]
                st['patch_K'] = st.get('patch_K', 0)+rp['K']; st['patch_cols'] = st.get('patch_cols', 0)+rp['evals']-rp['K']
                evals += rp['evals']; r = dict(r, **{k_: rp[k_] for k_ in ('K', 'EDGE', 'NEG', 'PMP', 'DELTA', 'GP', 'PHP', 'RR')})
            delta = r['DELTA']
            if delta0 is None:
                delta0 = delta
            pred = -(r['GP']+.5*r['PHP'])
            if st['nlog'] is not None:                                              # (before the evaluation at z + p: the products' base is z)
                t_ = self.CG.t; gz_ = wp.to_torch(A['gz'], requires_grad=False); p_ = t_['p']
                acc = float((gz_+t_['Hp']).norm()); fresh = float((gz_+self._fresh_Hp()).norm()); st['log_products'] = st.get('log_products', 0)+1   # (not in evals: the cost compared)
                pn = p_.norm(dim=1); kmax = int(pn.argmax())
                rec = dict(pass_=pass_, it=its, delta=delta, K=r['K'], exit='N' if r['NEG'] else ('E' if r['EDGE'] else ('C' if r['K'] >= self.cg_cap else 'T')),
                           rr=math.sqrt(max(r['RR'], 0.)), stop=stop, res_acc=acc, res_true=fresh, pM=math.sqrt(max(r['PMP'], 0.)), p2=float(p_.norm()), pmax=float(pn[kmax]), pcell=kmax,
                           g=g_, pred=pred)
            s = self._eval(1.); evals += 1
            tstar, feas = 1., True
            if self.flip_guard > 0:                                                 # (11-07, the expert's first layer, made trustworthy: the step's end F keeps det(I + G / theta) >= margin on every cell.
                m_ = self.flip_guard; I3_ = W.I3                                    # G is linear in x: along the line z + t p the det is a cubic in t, its FIRST crossing of the margin from z bounds the step;
                D_ = (W.tG-G0_)/self.th; A0_ = I3_+G0_/self.th                     # every accepted point is verified; an infeasible trial is rejected; z is always the last feasible point)
                dmin0 = float(torch.linalg.det(A0_).min())
                if dmin0 < m_*(1-1e-6):                                             # (cannot happen: z was verified when accepted; counted apart and the solve stops)
                    st['guard_breach_z'] = st.get('guard_breach_z', 0)+1; break
                tc_, cblk_ = self._first_crossing_cell(A0_, D_, m_)
                pmp_full = math.sqrt(max(r['PMP'], 0.))
                st['guard_dmin'] = min(st.get('guard_dmin', math.inf), float(torch.linalg.det(A0_+D_).min()))
                if tc_ < 1.:
                    tstar = .9*tc_
                    st['guard'] = st.get('guard', 0)+1; st['guard_t'] = min(st.get('guard_t', 1.), tstar)
                    if tstar < 1e-3 or (its > 1 and st['guard_log'][-1][2] < 1. and abs(st['guard_log'][-1][1]-dmin0) < 1e-3*m_):
                        st['guard_wall'] = st.get('guard_wall', 0)+1                 # (the solver stalled at the wall: shortened again with z's det unchanged at the margin)
                    s = self._eval(tstar); evals += 1                                # (the trial at z + t* p; the model's decrease along the same line)
                    r = dict(r, GP=tstar*r['GP'], PHP=tstar*tstar*r['PHP'], PMP=tstar*tstar*r['PMP'], EDGE=False)   # (the model's decrease of the step actually taken; the region is not the wall's business)
                    pred = -(r['GP']+.5*r['PHP'])
                dtr_ = float(torch.linalg.det(I3_+W.tG/self.th).min())              # (the trial verified on its own evaluation)
                feas = dtr_ >= m_*(1-1e-9)
                if not feas:
                    st['guard_breach'] = st.get('guard_breach', 0)+1
                st.setdefault('guard_log', []).append((its, dmin0, tc_, tstar, dtr_, int(torch.linalg.det(I3_+W.tG/self.th).argmin())))
            ok = math.isfinite(s[PS_PQ]) and math.isfinite(s[PS_GQ2])
            noise = 1e-14*abs(s[PS_PZ])
            ared = s[PS_PZ]-s[PS_PQ] if ok else -math.inf
            rho = ared/pred if pred > noise else (1. if ared >= -noise else -math.inf)
            pM = delta if r['EDGE'] else math.sqrt(max(r['PMP'], 0.))
            if rho < .25 or pred <= noise or not feas:
                delta = .25*pM
            elif rho > .75 and r['EDGE']:
                delta = 2*delta
            taken = ok and rho > 1e-4 and feas
            st['trace'].append((r['K'], r['EDGE'], r['NEG'], stop, g_, rho, taken, math.sqrt(max(r.get('RR', 0.), 0.))))   # (8th: the CG's last |r|, logging)
            if st['nlog'] is not None:
                rec.update(noise_branch=not pred > noise, ared=ared, rho=rho, g_trial=math.sqrt(s[PS_GQ2]) if ok else math.inf, taken=taken, delta_next=delta)
                st['nlog'].append(rec)
            if taken:
                self._accept()
                if krylov is not None:
                    krylov.invalidate()
                g2 = s[PS_GQ2]
                if self.slide and self.flip_guard > 0 and tstar < self.slide_thr:  # (11-07 expert: a severely cut Newton step is taken, then a step along the wall from there, the
                    took = self._slide(st, m_, cblk_, pmp_full, evals)              # blocking cell's det gradient projected out; the next Newton direction is fresh)
                    evals = took['evals']
                    if took['taken']:
                        g2 = took['g2']
            else:
                st['rejected'] += 1
                g2 = self._eval(0.)[PS_GQ2]; evals += 1
                self._accept()
                if not delta > 1e-12*delta0:
                    break
        return evals, its, g2, delta
