"""Plasticity on the DEM's springs (2026-10-31; the user: DEM can break and rebuild springs too). Moduli per unit density, masses = volumes.
  springs    each wall one spring in two halves, one per cell, in parallel, shares s = h / L of the wall's pyramids (clamped at 0); the reading is the DEM's: D = x_j + R_j r_j - x_i - R_i r_i, d = R_m^T D,
             theta = log(R_i^T R_j), R from each cell's F. A half reads without the volume: d - m - (L_o - L_ref,o) / 3 L_h n, m its memory, L_o the owner's ln J
  limit      per cell, on its whole store: U_i = sum over its halves of 1/2 k . reading^2 + the outer layer mu r tr(e N e), e = dev sym(R^T F); e_i = (U_i / (mu V_i))^1/2; E_i = V_i W(e_i), W as von Mises:
             mu e^2 up to e_y = s_y / (2 mu), then s_y e - s_y^2 / (4 mu). So dE_i / dU_i = beta_i = min(1, e_y / e_i): every part of the cell's store scaled by one beta, a true potential of x
  update     at x^n+1 (theta = 1): a yielding cell's halves keep beta (reading), F keeps beta dev ln U, what goes is the plastic strain. The bound is on the state itself
  walls      a cell that has yielded reads the current cut each step: its halves' k = s 2 mu A / L (bending 2 mu A^2 / 4 pi L, torsion twice that), levers, normal, length; only the strain gamma = reading / L goes on,
             so the force goes with the current area. A wall that closes takes its half along, the force gone with the area. A new wall gets its halves when both cells have yielded, gamma 0;
             otherwise nothing here (self contact is not in this file). A cell that has not yielded keeps its rest halves whether or not their walls exist: sigma_y inf never leaves them
  pressure   not limited and not in the halves: V0 kap / 2 L^2, L = L_n + tr(Q^T (I + G)) - 3, Q frozen at Newton's start, exact ln det at the step's end, variance on the current walls; G the divergence-theorem
             gradient of ul_plastic.py, also the one in F = (I + G) F_n
Not as ClosureWarp at sigma_y inf: the springs' volume part lives on J for every cell, G is 'gg', and theta = 1 (the bound needs it: the midpoint's step end 2 clip(m + D / 2) - m reaches three times the limit on reversal)."""
import math

import numpy as np
import torch
import warp as wp

from free_surface.closure_warp import k_Aul, k_F8, k_Gul, k_Gul_adj, k_ball, k_ball_adj
from free_surface.skin_warp import NSLOT, NT, bond_quad, k_total, k_total_loss, quat_conj, quat_log, quat_log_adj, quat_mat, quat_mat_adj, quat_mul
from free_surface.ul_plastic import PlasticUL, k_Aul_adj2, k_Ge, k_Ge_adj, k_Lb, k_gg_cell, k_gg_cell_adj, k_gg_wall, k_gg_wall_adj, k_var, polar_batch

f64 = torch.float64


@wp.kernel(enable_backward=False)
def k_Fs(G: wp.array(dtype=wp.mat33d), F0: wp.array(dtype=wp.mat33d), Fc: wp.array(dtype=wp.mat33d)):
    b = wp.tid()
    Fc[b] = F0[b]+G[b]@F0[b]


@wp.kernel(enable_backward=False)
def k_Fs_adj(Fcb: wp.array(dtype=wp.mat33d), F0: wp.array(dtype=wp.mat33d), Gb: wp.array(dtype=wp.mat33d)):
    b = wp.tid()
    Gb[b] = Fcb[b]@wp.transpose(F0[b])


@wp.kernel(enable_backward=False)
def k_L(G: wp.array(dtype=wp.mat33d), Q: wp.array(dtype=wp.mat33d), Ln: wp.array(dtype=wp.float64), V0: wp.array(dtype=wp.float64), kap: wp.float64, L: wp.array(dtype=wp.float64),
        Lb: wp.array(dtype=wp.float64), terms: wp.array2d(dtype=wp.float64)):
    """L = L_n + tr(Q^T (I + G)) - 3 and its energy; dE/dL into Lb"""
    b = wp.tid()
    Lc = Ln[b]+wp.trace(wp.transpose(Q[b])@(wp.identity(n=3, dtype=wp.float64)+G[b]))-wp.float64(3.0)
    L[b] = Lc
    wp.atomic_add(terms, 1, b % NSLOT, V0[b]*wp.float64(0.5)*kap*Lc*Lc)
    wp.atomic_add(Lb, b, V0[b]*kap*Lc)


@wp.kernel(enable_backward=False)
def k_hframes(q: wp.array(dtype=wp.vec4d), hi: wp.array(dtype=int), hj: wp.array(dtype=int), Rm: wp.array(dtype=wp.mat33d), th: wp.array(dtype=wp.vec3d)):
    """the mid frame and the relative rotation of each row, as k_hread reads them"""
    e = wp.tid()
    i, j = hi[e], hj[e]
    if i == j:
        return
    qr = quat_mul(quat_conj(q[i]), q[j])
    if qr[0] < wp.float64(0.0):
        qr = -qr
    th[e] = quat_log(qr)
    h = qr+wp.vec4d(wp.float64(1.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
    Rm[e] = quat_mat(quat_mul(q[i], h/wp.length(h)))


@wp.kernel(enable_backward=False)
def k_hread(x: wp.array(dtype=wp.vec3d), q: wp.array(dtype=wp.vec4d), hi: wp.array(dtype=int), hj: wp.array(dtype=int), ho: wp.array(dtype=int), ri: wp.array(dtype=wp.vec3d), rj: wp.array(dtype=wp.vec3d),
            hn: wp.array(dtype=wp.vec3d), hk: wp.array(dtype=wp.vec4d), dp: wp.array(dtype=wp.vec3d), thp: wp.array(dtype=wp.vec3d), hL: wp.array(dtype=wp.float64), L: wp.array(dtype=wp.float64),
            Lref: wp.array(dtype=wp.float64), U: wp.array(dtype=wp.float64), de: wp.array(dtype=wp.vec3d), te: wp.array(dtype=wp.vec3d)):
    """a half's reading and its energy into its owner's store"""
    e = wp.tid()
    i, j = hi[e], hj[e]
    if i == j:
        return
    qi, qj = q[i], q[j]
    D = x[j]+quat_mat(qj)@rj[e]-x[i]-quat_mat(qi)@ri[e]
    qr = quat_mul(quat_conj(qi), qj)
    if qr[0] < wp.float64(0.0):
        qr = -qr
    h = qr+wp.vec4d(wp.float64(1.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
    Rm = quat_mat(quat_mul(qi, h/wp.length(h)))
    o = ho[e]
    n = hn[e]
    a = wp.transpose(Rm)@D-dp[e]-((L[o]-Lref[o])/wp.float64(3.0)*hL[e])*n
    t = quat_log(qr)-thp[e]
    de[e] = a
    te[e] = t
    wp.atomic_add(U, o, bond_quad(a, t, n, hk[e]))


@wp.kernel(enable_backward=False)
def k_oread(F: wp.array(dtype=wp.mat33d), R: wp.array(dtype=wp.mat33d), orr: wp.array(dtype=wp.float64), oN: wp.array(dtype=wp.mat33d), mu: wp.float64, U: wp.array(dtype=wp.float64)):
    """the outer layer, the cell's store's part on its own F"""
    b = wp.tid()
    if orr[b] == wp.float64(0.0):
        return
    A = wp.transpose(R[b])@F[b]
    I = wp.identity(n=3, dtype=wp.float64)
    s = wp.float64(0.5)*(A+wp.transpose(A))-I
    s = s-(wp.trace(s)/wp.float64(3.0))*I
    wp.atomic_add(U, b, mu*orr[b]*wp.trace(s@oN[b]@s))


@wp.kernel(enable_backward=False)
def k_cell_store(U: wp.array(dtype=wp.float64), Vr: wp.array(dtype=wp.float64), mu: wp.float64, sy: wp.float64, ei: wp.array(dtype=wp.float64), be: wp.array(dtype=wp.float64), terms: wp.array2d(dtype=wp.float64)):
    """E_i = V_i W(e_i) and beta_i = dE_i / dU_i"""
    b = wp.tid()
    e = wp.sqrt(wp.max(U[b], wp.float64(0.0))/(mu*Vr[b]))
    ei[b] = e
    w = mu*e*e
    bb = wp.float64(1.0)
    if wp.float64(2.0)*mu*e > sy:
        w = sy*e-sy*sy/(wp.float64(4.0)*mu)
        bb = sy/(wp.float64(2.0)*mu*e)
    be[b] = bb
    wp.atomic_add(terms, 0, b % NSLOT, Vr[b]*w)


@wp.kernel(enable_backward=False)
def k_hforce(x: wp.array(dtype=wp.vec3d), q: wp.array(dtype=wp.vec4d), hi: wp.array(dtype=int), hj: wp.array(dtype=int), ho: wp.array(dtype=int), ri: wp.array(dtype=wp.vec3d), rj: wp.array(dtype=wp.vec3d),
             hn: wp.array(dtype=wp.vec3d), hk: wp.array(dtype=wp.vec4d), hL: wp.array(dtype=wp.float64), de: wp.array(dtype=wp.vec3d), te: wp.array(dtype=wp.vec3d), be: wp.array(dtype=wp.float64),
             x_adj: wp.array(dtype=wp.vec3d), q_adj: wp.array(dtype=wp.vec4d), Lb: wp.array(dtype=wp.float64)):
    """k_bonds_fused's adjoint on the half's reading, scaled by its owner's beta; d(reading)/dL_o = - L_h / 3 n into Lb"""
    e = wp.tid()
    i, j = hi[e], hj[e]
    if i == j:
        return
    qi, qj = q[i], q[j]
    Ri, Rj = quat_mat(qi), quat_mat(qj)
    D = x[j]+Rj@rj[e]-x[i]-Ri@ri[e]
    qr = quat_mul(quat_conj(qi), qj)
    sg = wp.float64(1.0)
    if qr[0] < wp.float64(0.0):
        qr = -qr
        sg = wp.float64(-1.0)
    hr = qr+wp.vec4d(wp.float64(1.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
    hl = wp.length(hr)
    h = hr/hl
    qm = quat_mul(qi, h)
    Rm = quat_mat(qm)
    n = hn[e]
    a, t = de[e], te[e]
    an, tn = wp.dot(a, n), wp.dot(t, n)
    at, tt = a-an*n, t-tn*n
    ke = hk[e]
    o = ho[e]
    bb = be[o]
    fd = bb*(ke[0]*an*n+ke[1]*at)
    ft = bb*(ke[2]*tt+ke[3]*tn*n)
    wp.atomic_add(Lb, o, -(hL[e]/wp.float64(3.0))*wp.dot(fd, n))
    gD = Rm@fd
    gqi = quat_mat_adj(qi, -wp.outer(gD, ri[e]))
    gqj = quat_mat_adj(qj, wp.outer(gD, rj[e]))
    gqm = quat_mat_adj(qm, wp.outer(D, fd))
    gqi = gqi+quat_mul(gqm, quat_conj(h))
    gh = quat_mul(quat_conj(qi), gqm)
    gqr = (gh-h*wp.dot(h, gh))/hl+quat_log_adj(qr, ft)
    gqr = gqr*sg
    gqj = gqj+quat_mul(qi, gqr)
    gqi = gqi+quat_conj(quat_mul(gqr, quat_conj(qj)))
    wp.atomic_add(x_adj, i, -gD)
    wp.atomic_add(x_adj, j, gD)
    wp.atomic_add(q_adj, i, gqi)
    wp.atomic_add(q_adj, j, gqj)


@wp.kernel(enable_backward=False)
def k_oforce(F: wp.array(dtype=wp.mat33d), R: wp.array(dtype=wp.mat33d), orr: wp.array(dtype=wp.float64), oN: wp.array(dtype=wp.mat33d), mu: wp.float64, be: wp.array(dtype=wp.float64), Fb: wp.array(dtype=wp.mat33d),
             Rb: wp.array(dtype=wp.mat33d)):
    """d(mu r tr(e N e)) = S : d(R^T F), S = dev(mu r (N e + e N)): dF gets R S, dR gets F S"""
    b = wp.tid()
    if orr[b] == wp.float64(0.0):
        return
    A = wp.transpose(R[b])@F[b]
    I = wp.identity(n=3, dtype=wp.float64)
    s = wp.float64(0.5)*(A+wp.transpose(A))-I
    s = s-(wp.trace(s)/wp.float64(3.0))*I
    S = (be[b]*mu*orr[b])*(oN[b]@s+s@oN[b])
    S = S-(wp.trace(S)/wp.float64(3.0))*I
    wp.atomic_add(Fb, b, R[b]@S)
    wp.atomic_add(Rb, b, F[b]@S)


class SpringPlastic(PlasticUL):
    """X, r, V0, E, nu, sigma_y as PlasticUL (sigma_y inf: no cell ever yields); the interface of MidpointStep and TrustCG through PlasticStep. Kh: the rows' capacity (default twice the walls')"""

    def __init__(self, X, r, V0, E, nu, sigma_y, grad='gg', agg=.1, vol_tol=1e-5, Kw=None, Kh=None, dev='cuda'):
        super().__init__(X, r, V0, E, nu, sigma_y, grad, agg, vol_tol, Kw, dev)
        n = self.n; X_ = self.X
        self.ey = self.sy/(2*self.mu)
        G0, _ = self._recut.geometry(X_, self._rr0)
        kp = G0.wall_A.detach() > 0
        i, j, A, c = G0.i[kp].long(), G0.j[kp].long(), G0.wall_A.detach()[kp], G0.wall_c.detach()[kp]
        lo, hi = torch.minimum(i, j), torch.maximum(i, j)
        every = torch.ones_like(A, dtype=torch.bool)
        self._rest = tuple(torch.cat([a_, b_]) for a_, b_ in zip(self._halves(X_, lo, hi, A, c, lo, every), self._halves(X_, lo, hi, A, c, hi, every)))
        cap0 = G0.patch_A.detach() > 0
        self._or0 = torch.where(cap0, self._rr0, torch.zeros_like(self._rr0)); self._oN0 = torch.where(cap0[:, None, None], G0.patch_nn.detach(), torch.zeros(n, 3, 3, dtype=f64, device=dev))
        Kh = 2*self.ul['Kw'] if Kh is None else Kh
        z = lambda k, dt_: wp.zeros(k, dtype=dt_, device=dev)
        self.sp = dict(Kh=Kh, hi=z(Kh, wp.int32), hj=z(Kh, wp.int32), ho=z(Kh, wp.int32), ri=z(Kh, wp.vec3d), rj=z(Kh, wp.vec3d), hn=z(Kh, wp.vec3d), hk=z(Kh, wp.vec4d), dp=z(Kh, wp.vec3d), thp=z(Kh, wp.vec3d),
                       hL=z(Kh, wp.float64), de=z(Kh, wp.vec3d), te=z(Kh, wp.vec3d), Rm=z(Kh, wp.mat33d), th=z(Kh, wp.vec3d), Lref=z(n, wp.float64), Vr=z(n, wp.float64), orr=z(n, wp.float64), oN=z(n, wp.mat33d), F0=z(n, wp.mat33d))
        tt = lambda a_: wp.to_torch(a_, requires_grad=False)
        self.tsp = {k_: tt(a_) for k_, a_ in self.sp.items() if k_ != 'Kh'}
        self.B = dict(G0=z(n, wp.mat33d), Ga=z(n, wp.mat33d), Ge=z(n, wp.mat33d), Sg=z(n, wp.mat33d), G=z(n, wp.mat33d), Gb=z(n, wp.mat33d), Gs=z(n, wp.mat33d), Gbe=z(n, wp.mat33d), Gb0=z(n, wp.mat33d),
                      L=z(n, wp.float64), Lb=z(n, wp.float64), U=z(n, wp.float64), ei=z(n, wp.float64), be=z(n, wp.float64), Fc=z(n, wp.mat33d), F=z(n, wp.mat33d), R=z(n, wp.mat33d), q=z(n, wp.vec4d),
                      Fcb=z(n, wp.mat33d), Fb=z(n, wp.mat33d), Rb=z(n, wp.mat33d), qb=z(n, wp.vec4d), terms=wp.zeros((NT, 256), dtype=wp.float64, device=dev), tsum=wp.zeros(NT, dtype=wp.float64, device=dev),
                      loss=wp.zeros(1, dtype=wp.float64, device=dev), bdelta=z(n, wp.float64), bfn=z(n, wp.float64), bff=z(n, wp.vec3d))
        self.tF, self.tR, self.tq, self.tG, self.tL, self.tei, self.tbe, self.tU = (tt(self.B[k_]) for k_ in ('F', 'R', 'q', 'G', 'L', 'ei', 'be', 'U'))
        self.tbe.fill_(1.)
        self._Fe = self.I3.expand(n, 3, 3).clone()
        self._yld = torch.zeros(n, dtype=torch.bool, device=dev)
        self._mkey = torch.zeros(0, dtype=torch.long, device=dev); self._mg = torch.zeros(0, 3, dtype=f64, device=dev); self._mt = torch.zeros(0, 3, dtype=f64, device=dev)
        self.ul_stats.update(rows=0, born=0, closed=0)

    def _halves(self, x, lo, hi, A, c, own, keep):
        """the rows of the walls (lo, hi) for the owners own where keep: (lo, hi, own, levers r_lo, r_hi as seen from x, normal, k (n, 4), L). The stiffnesses as the DEM's build at alpha = 1"""
        lo, hi, A, c, own = lo[keep], hi[keep], A[keep], c[keep], own[keep]
        d = x[hi]-x[lo]; L = d.norm(dim=1); nh = d/L[:, None]
        hl = ((c-x[lo])*nh).sum(1).clamp(min=0); hh = (L-((c-x[lo])*nh).sum(1)).clamp(min=0)
        s = torch.where(own == lo, hl, hh)/(hl+hh)
        sec = A*A/(4*math.pi)
        k = (2*self.mu*s/L)[:, None]*torch.stack([A, A, sec, 2*sec], 1)
        return lo, hi, own, c-x[lo], c-x[hi], nh, k, L

    def prepare(self):
        n, dev = self.n, self.dev
        self.tw = wp.array(np.ones(NT), dtype=wp.float64, device=dev)
        self.tones = wp.array(np.ones((NT, 256)), dtype=wp.float64, device=dev)
        self.xw = wp.zeros(n, dtype=wp.vec3d, device=dev, requires_grad=True)
        tt = lambda a_: wp.to_torch(a_, requires_grad=False)
        self.tx, self.tgx, self.tloss, self.tsum = tt(self.xw), tt(self.xw.grad), tt(self.B['loss']), tt(self.B['tsum'])
        self.thde, self.thte = self.tsp['de'], self.tsp['te']
        self.tx.copy_(self.X)
        self.run()
        wp.capture_begin()
        self.run()
        self.graph = wp.capture_end()

    def run(self):
        """E and dE/dx at xw: forward, the cells' stores and betas, the forces, then the chain through F and G"""
        B, U, S, n, Kw, Kh = self.B, self.ul, self.sp, self.n, self.ul['Kw'], self.sp['Kh']
        x, g = self.xw, self.xw.grad
        for k_ in ('G0', 'Ga', 'Sg', 'Lb', 'U', 'Gbe', 'Gb0', 'Fcb', 'Fb', 'Rb', 'qb', 'terms', 'tsum'):
            B[k_].zero_()
        g.zero_()
        kap, mu, sy = wp.float64(self.kap), wp.float64(self.mu), wp.float64(self.sy)
        wp.launch(k_Gul, 2*Kw, inputs=[x, U['xn'], U['S'], U['O'], U['R']], outputs=[B['G0']])
        wp.launch(k_Aul, Kw, inputs=[B['G0'], U['As'], U['Ao'], U['Aw']], outputs=[B['Ga']])
        wp.launch(k_Ge, n, inputs=[B['G0'], B['Ga'], U['used']], outputs=[B['Ge']])
        wall = [U['wi'], U['wj'], U['wa'], U['wn'], U['wt'], U['wtau']]
        wp.launch(k_gg_wall, Kw, inputs=[x, U['xn']]+wall+[B['Ge']], outputs=[B['Sg']])
        wp.launch(k_gg_cell, n, inputs=[x, U['xn'], B['Sg'], B['Ge'], U['a'], U['Mc'], U['Vi']], outputs=[B['G']])
        wp.launch(k_Fs, n, inputs=[B['G'], S['F0']], outputs=[B['Fc']])
        wp.launch(k_F8, n, inputs=[B['Fc']], outputs=[B['F'], B['R'], B['q']])
        wp.launch(k_L, n, inputs=[B['G'], U['Q'], U['Ln'], U['V0'], kap], outputs=[B['L'], B['Lb'], B['terms']])
        wp.launch(k_var, Kw, inputs=[U['wi'], U['wj'], U['wv'], B['L']], outputs=[B['Lb'], B['terms']])
        rows = [S['hi'], S['hj'], S['ho'], S['ri'], S['rj'], S['hn'], S['hk']]
        wp.launch(k_hread, Kh, inputs=[x, B['q']]+rows+[S['dp'], S['thp'], S['hL'], B['L'], S['Lref']], outputs=[B['U'], S['de'], S['te']])
        wp.launch(k_oread, n, inputs=[B['F'], B['R'], S['orr'], S['oN'], mu], outputs=[B['U']])
        wp.launch(k_cell_store, n, inputs=[B['U'], S['Vr'], mu, sy], outputs=[B['ei'], B['be'], B['terms']])
        wp.launch(k_hforce, Kh, inputs=[x, B['q']]+rows+[S['hL'], S['de'], S['te'], B['be']], outputs=[g, B['qb'], B['Lb']])
        wp.launch(k_oforce, n, inputs=[B['F'], B['R'], S['orr'], S['oN'], mu, B['be']], outputs=[B['Fb'], B['Rb']])
        wp.launch(k_F8, n, inputs=[B['Fc']], outputs=[B['F'], B['R'], B['q']], adj_inputs=[B['Fcb']], adj_outputs=[B['Fb'], B['Rb'], B['qb']], adjoint=True)
        wp.launch(k_Fs_adj, n, inputs=[B['Fcb'], S['F0']], outputs=[B['Gb']])
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
        """the step's start at x: PlasticUL's radii, cut, fit and gradient data; the frames of F_n; the rows: rest halves of the cells that have not yielded, the current walls' halves of those that have"""
        super().update_ul(x)
        x = x.detach(); n, dev, s = self.n, self.dev, self.tsp; yld = self._yld; G = self._ulG; st = self.ul_stats
        s['F0'].copy_(self._Fe)
        wp.launch(k_F8, n, inputs=[self.sp['F0']], outputs=[self.B['F'], self.B['R'], self.B['q']])
        lo0, hi0, own0, ri0, rj0, nh0, k0, L0 = self._rest
        keep0 = ~yld[own0]
        m0 = int(keep0.sum())
        kp = G.wall_A.detach() > 0
        i, j, A, c = G.i[kp].long(), G.j[kp].long(), G.wall_A.detach()[kp], G.wall_c.detach()[kp]
        lo, hi = torch.minimum(i, j), torch.maximum(i, j)
        rows, born, kept = [], 0, 0
        both = yld[lo] & yld[hi]
        for own in (lo, hi):
            key = own*n*n+lo*n+hi
            found = torch.zeros_like(key, dtype=torch.bool); pos = torch.zeros_like(key)
            if len(self._mkey):
                pos = torch.searchsorted(self._mkey, key).clamp(max=len(self._mkey)-1)
                found = self._mkey[pos] == key
            make = yld[own] & (found | both)
            born += int((make & ~found).sum()); kept += int((make & found).sum())
            g_ = torch.zeros(int(make.sum()), 3, dtype=f64, device=dev); t_ = torch.zeros_like(g_)
            fm = found[make]
            g_[fm] = self._mg[pos[make][fm]]; t_[fm] = self._mt[pos[make][fm]]
            rows.append(self._halves(x, lo, hi, A, c, own, make)+(g_, t_))
        cols = [torch.cat([a_, b_]) for a_, b_ in zip(rows[0], rows[1])]
        lo1, hi1, own1, ci, cj, nh1, k1, L1, g1, t1 = cols
        m1 = len(lo1); m = m0+m1
        if m > self.sp['Kh']:
            raise RuntimeError('update_ul: %d rows over the capacity %d' % (m, self.sp['Kh']))
        for k_ in ('hi', 'hj', 'ho', 'ri', 'rj', 'hn', 'hk', 'dp', 'thp', 'hL'):
            s[k_].zero_()
        s['hi'][:m] = torch.cat([lo0[keep0], lo1]).to(torch.int32); s['hj'][:m] = torch.cat([hi0[keep0], hi1]).to(torch.int32); s['ho'][:m] = torch.cat([own0[keep0], own1]).to(torch.int32)
        s['hk'][:m] = torch.cat([k0[keep0], k1]); s['hL'][:m] = torch.cat([L0[keep0], L1])
        s['ri'][:m0] = ri0[keep0]; s['rj'][:m0] = rj0[keep0]; s['hn'][:m0] = nh0[keep0]
        if m1:                                                                      # (the levers in the cells' frames, D = 0 at x; the normal in the mid frame; the memory as dp = - reading, thp)
            Rs = self.tR
            s['ri'][m0:m] = torch.einsum('fba,fb->fa', Rs[lo1], ci); s['rj'][m0:m] = torch.einsum('fba,fb->fa', Rs[hi1], cj)
            wp.launch(k_hframes, self.sp['Kh'], inputs=[self.B['q'], self.sp['hi'], self.sp['hj']], outputs=[self.sp['Rm'], self.sp['th']])
            s['hn'][m0:m] = torch.einsum('fba,fb->fa', s['Rm'][m0:m], nh1)
            s['dp'][m0:m] = -g1*L1[:, None]; s['thp'][m0:m] = s['th'][m0:m]-t1
        V = G.V.detach(); Vmin = 1e-9*float(self.V0.median())
        s['Vr'].copy_(torch.where(yld, V.clamp(min=Vmin), self.V0)); s['Lref'].copy_(torch.where(yld, self.tul['Ln'], torch.zeros_like(self.V0)))
        cap = G.patch_A.detach() > 0
        rc = torch.where(cap, self._ulr, torch.zeros_like(self._ulr)); Nc = torch.where(cap[:, None, None], G.patch_nn.detach(), torch.zeros(n, 3, 3, dtype=f64, device=dev))
        s['orr'].copy_(torch.where(yld, rc, self._or0)); s['oN'].copy_(torch.where(yld[:, None, None], Nc, self._oN0))
        self._rows = (torch.cat([own0[keep0], own1]), torch.cat([lo0[keep0], lo1]), torch.cat([hi0[keep0], hi1]), m)
        st['rows'] = m; st['born'] = born; st['closed'] = len(self._mkey)-kept

    def set_rotation(self, z):
        """the step's frozen rotation of L's linear reading, at Newton's start z (Q = I for that evaluation)"""
        self.grad(z)
        self.tul['Q'].copy_(polar_batch(self.I3+self.tG))

    def end_ul(self, th=1.):
        """after the step (the buffers at the accepted x^n+1): the cells' returns, the halves' memories, F_n+1, L_n+1, the plastic work"""
        assert th == 1., 'the return belongs to the step end: theta 1 only'
        t, s, n, I3 = self.tul, self.tsp, self.n, self.I3
        Q, G = t['Q'], self.tG
        A = Q.transpose(1, 2)@(I3+G); H = .5*(A+A.transpose(1, 2))-I3
        be, ei = self.tbe.clone(), self.tei.clone(); yn = be < 1
        if bool(yn.any()):
            wc = float(((s['Vr']*self.sy*(ei-self.ey))[yn]).sum())
            self.Wp += wc; self.Wp_cell += wc
        new = self._yld | yn
        own, lo, hi, m = self._rows
        keep = new[own]
        key = (own*n*n+lo*n+hi)[keep]
        o = torch.argsort(key)
        self._mkey = key[o]
        self._mg = ((be[own, None]*s['de'][:m]/s['hL'][:m, None])[keep])[o]
        self._mt = ((be[own, None]*s['te'][:m])[keep])[o]
        Fe, R = self.tF.clone(), self.tR.clone()
        Qt = Q.transpose(1, 2)
        self.ep = Q@self.ep@Qt
        if bool(yn.any()):
            A_ = R[yn].transpose(1, 2)@Fe[yn]; w, v = torch.linalg.eigh(.5*(A_+A_.transpose(1, 2)))
            lw = torch.log(w.clamp(min=1e-6)); lm = lw.mean(1, keepdim=True); dv = lw-lm; b_ = be[yn, None]
            Ue = v@torch.diag_embed(torch.exp(lm+b_*dv))@v.transpose(1, 2)
            Fe[yn] = R[yn]@Ue
            self.ep[yn] = self.ep[yn]+R[yn]@(v@torch.diag_embed((1-b_)*dv)@v.transpose(1, 2))@R[yn].transpose(1, 2)
        self._Fe = Fe
        self._yld = new
        self.Hsum += (self.V0[:, None, None]*H).sum(0)/self.V0.sum()
        Fr = I3+G; det = torch.linalg.det(Fr); flip = det <= 1e-3
        tr = H.diagonal(dim1=1, dim2=2).sum(1)
        self.ul_stats['flips'] = int(flip.sum()); self.ul_stats['yielding'] = int(yn.sum()); self.ul_stats['yielded'] = int(new.sum())
        t['Ln'].add_(torch.where(flip, tr, torch.log(det.clamp(min=1e-3))))

    def stored(self):
        """the elastic energy after the last end_ul: the stores at their limit or below, the volume"""
        ec = self.tei.clamp(max=self.ey) if math.isfinite(self.sy) else self.tei
        return float((self.tsp['Vr']*self.mu*ec*ec).sum()), float((self.V0*.5*self.kap*self.tul['Ln']**2).sum())

    def ul_blocks(self, lr_thr=None, lr_cap=0):
        """the diagonal blocks for the preconditioner, at the step's start: the volume through the gradient's direct weights as PlasticUL.ul_blocks, each half's beta k on its two ends; R's chain left out"""
        assert lr_thr is None, 'no low-rank term here'
        n, t, s, dev = self.n, self.tul, self.tsp, self.dev
        i, j, Lw, key, sg, m = self._w
        A, tw, nn, Vi = t['wa'][:m], t['wt'][:m], t['wn'][:m], t['Vi']
        ci, cj = (A*Vi[i])[:, None]*nn, (A*Vi[j])[:, None]*nn
        s_ = [i, i, j, j]; k_ = [i, j, i, j]; r_ = [(1-tw)[:, None]*ci, tw[:, None]*ci, -(1-tw)[:, None]*cj, -tw[:, None]*cj]
        ids = torch.arange(n, device=dev)
        s_.append(ids); k_.append(ids); r_.append(Vi[:, None]*t['a'])
        S, O, R_ = t['S'][:2*m].long(), t['O'][:2*m].long(), t['R'][:2*m]
        cap = (t['Mc'].abs().sum((1, 2)) > 0)
        q = cap[S]
        rm = torch.einsum('fba,fb->fa', t['Mc'][S[q]], R_[q])*Vi[S[q]][:, None]
        s_ += [S[q], S[q]]; k_ += [O[q], S[q]]; r_ += [rm, -rm]
        sa, ka, ra = torch.cat(s_), torch.cat(k_), torch.cat(r_)
        u, inv = torch.unique(sa*n+ka, return_inverse=True)
        r = torch.zeros(len(u), 3, dtype=f64, device=dev).index_add(0, inv, ra); sc, kc = u//n, u % n
        K = torch.zeros(n, 3, 3, dtype=f64, device=dev)
        K.index_add_(0, kc, (self.V0[sc]*self.kap)[:, None, None]*r[:, :, None]*r[:, None, :])
        own, lo, hi, mr = self._rows
        kb = (self.tbe[own]*s['hk'][:mr, 0])[:, None, None]*self.I3
        K.index_add_(0, lo, kb); K.index_add_(0, hi, kb)
        return K
