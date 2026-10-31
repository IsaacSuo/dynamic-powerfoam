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
             sb: wp.array(dtype=wp.float64), th_: wp.float64, cv: wp.float64, sr: wp.array(dtype=wp.vec3d), terms: wp.array2d(dtype=wp.float64), x_adj: wp.array(dtype=wp.vec3d), Gb: wp.array(dtype=wp.mat33d)):
    """the remainder at z, at the step's end into sr, and its force; cv = beta / (th dt), the viscous part k beta dr/dt"""
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
    if cv > wp.float64(0.0):
        dv = d-dn
        wp.atomic_add(terms, 7, e % NSLOT, half*k*cv*wp.dot(dv, dv))
        f = f+(k*cv)*dv
    wp.atomic_add(x_adj, j, f)
    wp.atomic_add(x_adj, i, -f)
    PG = -half*wp.outer(f, p)
    wp.atomic_add(Gb, i, PG)
    wp.atomic_add(Gb, j, PG)


@wp.kernel(enable_backward=False)
def k_cell_visc(G: wp.array(dtype=wp.mat33d), V0: wp.array(dtype=wp.float64), emu: wp.float64, ekap: wp.float64, terms: wp.array2d(dtype=wp.float64), Gb: wp.array(dtype=wp.mat33d)):
    """Kelvin-Voigt on the step's gradient, sym G / (th dt) the rate: emu = beta mu / (th dt), ekap = beta kap / (th dt); term 7; into Gb"""
    b = wp.tid()
    I = wp.identity(n=3, dtype=wp.float64)
    s = wp.float64(0.5)*(G[b]+wp.transpose(G[b]))
    tr = wp.trace(s)
    e = s-(tr/wp.float64(3.0))*I
    wp.atomic_add(terms, 7, b % NSLOT, V0[b]*(emu*wp.ddot(e, e)+wp.float64(0.5)*ekap*tr*tr))
    Gb[b] = Gb[b]+V0[b]*((wp.float64(2.0)*emu)*e+(ekap*tr)*I)


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
        self.cv = 0.; self.Wv = 0.

    def set_viscosity(self, beta, dt):
        """Kelvin-Voigt with the elastic moduli times beta on the cells' shape and volume and on the springs, before prepare() and before the step is built (its preconditioner's factor 1 + beta / (th dt) follows smu). For a body that rings: at rest it adds no stress"""
        self.smu = float(beta); self.cv = float(beta)/(self.th*float(dt))

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
        wp.launch(k_spring, S['K'], inputs=[x, B['G'], S['si'], S['sj'], S['sP'], S['sdn'], S['sk'], S['sb'], th_, wp.float64(self.cv)], outputs=[S['sr'], B['terms'], g, B['Gb']])
        if self.cv > 0.:
            wp.launch(k_cell_visc, n, inputs=[B['G'], U['V0'], wp.float64(self.cv*self.mu), wp.float64(self.cv*self.kap)], outputs=[B['terms'], B['Gb']])
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
        n = self.n; x = x.detach(); st = self.ul_stats; t, s = self.tul, self.tsp; dev = x.device; yld = self._yld
        J = torch.exp(t['Ln']); st['floored'] = int((J < .02).sum())
        try:
            r, G, S_, it, nrep = self._recut.solve_radii(x, self._ulr, self.V0*J.clamp(min=.02), tol=self._ul_tol, r_base=self._rr0)
            st['radii_it'] = it
        except RuntimeError:
            st['radii_fail'] += 1
            r = self._ulr; G, S_ = self._recut.geometry(x, r)
        self._ulr, self._ulG = r, G
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
        Fe = self._Fe
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
        tiny = self.V0 < self._ul_agg*self.V0.median()
        m1, m2 = tiny[ic] & ~tiny[jc], tiny[jc] & ~tiny[ic]
        as_, ao_ = torch.cat([ic[m1], jc[m2]]), torch.cat([jc[m1], ic[m2]])
        va = self.V0[ao_]
        den = torch.zeros(n, dtype=f64, device=dev).index_add(0, as_, va)
        t['As'].zero_(); t['Ao'].zero_(); t['Aw'].zero_()
        k3 = len(as_); t['As'][:k3] = as_.to(torch.int32); t['Ao'][:k3] = ao_.to(torch.int32); t['Aw'][:k3] = va/den[as_]
        t['used'].copy_((tiny & (den > 0)).to(torch.int32))
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
        t['Q'].copy_(self.I3.expand(n, 3, 3))
        s['F0'].copy_(Fe)
        wp.launch(k_F8, n, inputs=[self.sp['F0']], outputs=[self.B['Fp'], self.B['Rp'], self.B['qp']])
        s['Rh'].copy_(self.tRp)
        A_ = self.tRp.transpose(1, 2)@Fe; e_ = .5*(A_+A_.transpose(1, 2))-self.I3
        s['en'].copy_(e_-(e_.diagonal(dim1=1, dim2=2).sum(1)/3)[:, None, None]*self.I3)

    def reread(self, beta=False):
        """after an evaluation at the point the solve stands on: R and Q read there, frozen for the next solve; beta: the springs' beta too, once per step at Newton's start (10-31: read at every accepted point it moved
        the target: each solve's ratio 1, |g| after the re-read jumping between 0.2 and 3 weights, no trend, on 300 Newton iterations)"""
        s, m = self.tsp, self._m_s
        wp.launch(k_F8, self.n, inputs=[self.B['Fc']], outputs=[self.B['Fp'], self.B['Rp'], self.B['qp']])
        s['Rh'].copy_(self.tRp)
        M = self.I3+self.tG/(2*self.th)                                             # (U['Q'] holds C; a cell flipped at this point reads the plain trace)
        a0, a1, a2 = M[:, :, 0], M[:, :, 1], M[:, :, 2]; c0, c1, c2 = torch.linalg.cross(a1, a2), torch.linalg.cross(a2, a0), torch.linalg.cross(a0, a1)
        det = (a0*c0).sum(1); ok = det > 1e-3
        self.tul['Q'].copy_(torch.where(ok[:, None, None], torch.stack([c0, c1, c2], 2)/torch.where(ok, det, torch.ones_like(det))[:, None, None], self.I3))
        if beta:
            s['sb'][:m] = .5*(self.tbe[self.tlo]+self.tbe[self.thi])

    def end_ul(self, th=.5):
        """after the step (the buffers at the accepted z; x -> F linear, so the end's F and G are exact): the cells' returns, the springs' remainders, F_n+1, L_n+1, the plastic work"""
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
        if self.cv > 0.:                                                            # (the viscous work over the step: its force . dx)
            sG = .5*(self.tG+self.tG.transpose(1, 2)); trG = sG.diagonal(dim1=1, dim2=2).sum(1); eG = sG-(trG/3)[:, None, None]*I3
            self.Wv += float((self.V0*self.cv*(2*self.mu*(eG*eG).sum((1, 2))+self.kap*trG*trG)).sum())/th
            self.Wv += float((sk*self.cv*((sr-sdn)*(sr-sdn)).sum(1)).sum())*th
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
        W.reread(beta=True)
        g2 = self._eval(0.)[PS_GQ2]
        self._accept()
        if getattr(self, 'probe', None) is not None:                                # (logging only: it leaves the last evaluation and accept as here)
            self.probe(self)
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
            pM = delta if r['EDGE'] else math.sqrt(max(r['PMP'], 0.))
            if rho < .25 or pred <= noise:
                delta = .25*pM
            elif rho > .75 and r['EDGE']:
                delta = 2*delta
            st['trace'].append((r['K'], r['EDGE'], r['NEG'], stop, g_, rho, ok and rho > 1e-4))
            if ok and rho > 1e-4:
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
