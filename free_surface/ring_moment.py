"""RingYield with the springs not tied to the walls: the cut goes on every step, for the volume, the contact and the candidates; a spring's weight moves only with a plastic return.
  a return gives up the part lam = 1 - c of the cell's deviatoric strain; its halves give up the same part of their distance to the current walls' weights, w' = w + lam (A / L - w); the outer layer as a set of halves
    square lattice in simple shear, plastic shear g, elastic e: M fixes the three pairs, w_old = 1 - g, w_new = g, while the walls read g + e: dw / (wall - w) = dg / e = lam
    uniform flow: lam is the same everywhere, and any mix of closed weights is closed, so the patch test holds
    no return, lam = 0: q, weights, outer layer, group do not move, whatever the walls do; the springs' energy 1/2 sum w |d - (1 + v) R q|^2 does not see the cut. e_y to 0: lam to 1, the walls' weights
  force: K = I, the plain derivative of 1/2 tr(X M X^T); R = polar(F), the same in all cells under the same F; a torque of M's non-roundness times e is the price (the theorem)
  a new wall gets a spring of weight 0 if its cells are two springs apart (a fold's two sheets are not: a wall between them is a free surface to both); a spring leaves when it has no wall and next to no weight
  colour contact (set_colour_contact, 11-02; ClosureWarp's of 10-30): a wall is material or contact by the cells' place at rest, the colour code (exp(-d^2 / 2 sigma^2) on the rest skin graph, the inside harmonic); a new wall's
    spring is born only between like colours (similarity sum sqrt(p_i p_j) >= sim_birth) in place of the two springs. A wall without a spring is a free surface to both cells (no volume flux, no gradient).
    Check on the first lion's locked states (lock_colour): the weightless springs split at .5 (rest distance 8-9 h under, 1-2 h over); the unlike ones out, the legs' lowest omega^2 3135-8662 -> 161-216
  the push: on the state, not on a history. Every pair of unlike colours (a static set) within rho_i + rho_k pushes as the two half cells along the line, a column under uniaxial strain lambda = |d| / (rho_i + rho_k), in the
    log measure as the volume: E = 1/2 M V_c (ln lambda)^2 (1 - similarity), M = lam + 2 mu, V_c the two cells' mean rest volume; 0 with its force at the touch, infinite as the centres meet, so they never pass
    (11-02, Hertz on rho_i + rho_k - |d| was tried: finite at d = 0, E_max = 8/15 E* sqrt(R_eff) (rho_i + rho_k)^5/2, and the push turns over past it; 560 of 600 leg-torso pairs came in over its speed sqrt(2 E_max / mu),
    4.3 m/s against 1.95, and the paws went in as with no push at all). rho = rho0 |F_n^T n|, the cell's extent along the line
    of a ball of radius rho0 carried by its fit F_n at the step's start (n the line there, both frozen over the step, as R and beta), rho0 half the cell's rest distance to its neighbours by wall area (the face neighbours:
    two cells side by side at their own size read delta 0). Its partners found by distance each step, within reach and a margin of h. (11-02 first the push on the approach since the wall's birth, ClosureWarp's d0:
    with balls of 2 h the cut has walls between unlike cells 1.6-3.9 h apart across the air, so the push began far before touching, and a pair that slid past left and came back nearer, born without force:
    10 % born under .82 h at t .34, the nearest leg-torso pair .42 h apart read delta .02; the paws went in .02 s later than without it)
  jump share (set_jump_share, 11-02): one piece of material or two is whether the inverse motion is continuous between two cells: J = X_j - X_i - int F^-1 dx along the current
    segment (exact by the fundamental theorem in one piece, whatever the flow; the rest jump across an interface). A wall without a spring counts as material by s = the static
    colour's similarity of X_i + int F^-1 dx to j (the jump's size measured through the material), the rest of it a free surface; such a pair pushes by 1 - s. The colour alone read
    compressed skin as contact: on the saved states 99-298 walls a state with colour < .5 had J ~ 0, rest 2.5-3.6 h apart, now about half, the rest segment inside the body (jump_check)"""
import numpy as np
import scipy.sparse as sp
import torch

import math

import warp as wp

from free_surface.closure_warp import k_mid_ball, k_mid_lamt
from free_surface.ring_yield import RingYield, YieldStep

f64 = torch.float64


class RingMoment(RingYield):
    stat_keys = ('rows', 'ghost', 'born', 'closed', 'm_err', 'm_err_max', 'c_err', 'c_err_max', 'lam_med', 'lam_max', 'undone', 'dw', 'gap', 'ghost_w')

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._wi = None

    ghost_in_fit = False                                                                   # (a spring without a wall has no flux: the volume is the walls')

    rest_ground = False                                                                    # (the ground's radius a cell's at rest, not the cut's: the ground's energy a function of x alone)

    def _cut(self, x):
        super()._cut(x)
        if self.rest_ground and self.ball is not None:
            self.tbR.copy_(self._tbR0)
            self.tbkn.copy_(2*self.Es*self.tbR.clamp(min=0.)*.05**.5)

    def _radii_start(self):
        """the weights r^2 extrapolated over the last step: the same solve, fewer cuts"""
        r1, r0 = self._ulr, getattr(self, '_ulr_last', None)
        self._ulr_last = r1.clone()
        if not self.radii_extrap or r0 is None:
            return r1
        return torch.sqrt((2*r1*r1-r0*r0).clamp(min=(.25*r1)**2))

    elastic_births = True
    radii_extrap = False                                                               # (on the 30 steps: the same time, no gain)
    _col_Q = None
    sim_birth = .5
    jump_share = False

    def set_jump_share(self):
        """the share of a wall without a spring by the jump of the inverse motion across it (11-02, after set_colour_contact): J = X_j - X_i - int F^-1 dx along x_i -> x_j,
        0 in one piece of material, the rest jump across an interface; s = the static colour's similarity of the rest point X_i + int F^-1 dx to j (and from j), the wall counting
        by s in the gradient, the volume and its smoothing, its cells' closure taking the rest; a pair without a spring pushes by 1 - s. No weightless springs: the share is the wall's"""
        self.jump_share = True
        self.elastic_births = False

    def _sim_at(self, y, b):
        """the static colour's similarity of rest points y to cells b: the rest power cell holding y and the next, by their power gap (continuous across the face, the second's
        weight 1/2 on it and under 5 % a quarter cell inside)"""
        X, w2 = self.X.float(), (self._rr0**2).float()
        tau2 = self._h**2/6
        out = torch.empty(len(y), dtype=f64, device=y.device)
        for c0 in range(0, len(y), 4096):
            pw = torch.cdist(y[c0:c0+4096].float(), X)**2-w2[None]
            v, k = torch.topk(pw, 2, dim=1, largest=False)
            g = torch.exp(-(v[:, 1]-v[:, 0]).double().clamp(min=0.)/tau2)
            bb = b[c0:c0+4096]
            out[c0:c0+4096] = (color_sim(self._col_Q, k[:, 0], bb)+g*color_sim(self._col_Q, k[:, 1], bb))/(1+g)
        return out

    def jump_sim(self, a, b, d, t):
        """the jump test's similarity of pairs (a, b), d = x_b - x_a, t the wall's place on the line (the step's start state F_e): from both ends, averaged"""
        Fi = self._inv(self._Fe[self._gid])
        I_ = torch.einsum('fab,fb->fa', Fi[a], t[:, None]*d)+torch.einsum('fab,fb->fa', Fi[b], (1-t)[:, None]*d)
        return .5*(self._sim_at(self.X[a]+I_, b)+self._sim_at(self.X[b]-I_, a))

    def _wall_share(self, i, j, A, c, x, conn):
        mw = conn.to(f64)
        if not self.jump_share:
            return mw
        k = torch.nonzero(~conn)[:, 0]
        if len(k):
            a, b = i[k], j[k]
            d = x[b]-x[a]
            t = (((c[k]-x[a])*d).sum(1)/(d*d).sum(1)).clamp(0., 1.)
            mw[k] = self.jump_sim(a, b, d, t)
        kk = torch.minimum(i, j)*self.n+torch.maximum(i, j)
        o = torch.argsort(kk)
        self._wkey, self._wshare = kk[o], mw[o]
        st = self.ul_stats
        st['share_n'] = len(k); st['share_hi'] = int((mw[k] > .5).sum()); st['share_mid'] = int(((mw[k] > .1) & (mw[k] < .9)).sum())
        st['share_mean'] = float(mw[k].mean()) if len(k) else 0.
        return mw

    def set_colour_contact(self, sigma, K=262144):
        """the colour code on the rest cut, each cell's rho0 and the contact list's buffers, before prepare(); K the list's capacity. E* of two like bodies, E / 2 (1 - nu^2), as closure_warp.py"""
        n, dev = self.n, self.dev
        self._col_Q = color_code(self, self._rr0, sigma)
        G, _ = self._recut.geometry(self.X, self._rr0)
        kp = G.wall_A.detach() > 0
        i0, j0, A0 = G.i[kp].long(), G.j[kp].long(), G.wall_A.detach()[kp]
        L0 = (self.X[j0]-self.X[i0]).norm(dim=1)
        sA = torch.zeros(n, dtype=f64, device=dev).index_add(0, i0, A0).index_add(0, j0, A0)
        sAL = torch.zeros(n, dtype=f64, device=dev).index_add(0, i0, A0*L0).index_add(0, j0, A0*L0)
        self._rho0 = .5*sAL/sA.clamp(min=1e-300)
        self._h = float(self.V0.median())**(1/3)
        z = lambda k_, dt_: wp.zeros(k_, dtype=dt_, device=dev)
        self.selfc = dict(K=K, i=z(K, wp.int32), k=z(K, wp.int32), Rs=z(K, wp.float64), Kc=z(K, wp.float64), fn=z(K, wp.float64), nv=z(K, wp.vec3d), xn=z(n, wp.vec3d))
        self.Mp = self.lam+2*self.mu                                                       # (the P-wave modulus: uniaxial strain)
        tt = lambda a_: wp.to_torch(a_, requires_grad=False)
        self.tsc = {k_: tt(self.selfc[k_]) for k_ in ('i', 'k', 'Rs', 'Kc', 'fn', 'xn')}
        self._pairs = torch.zeros(0, 2, dtype=torch.long, device=dev); self._pw = torch.zeros(0, dtype=f64, device=dev)
        self.self_stats = dict(pairs=0, active=0, d_max=0.)

    def contact_pairs(self, x, F, margin):
        """the unlike pairs without a spring within reach + margin at x: (a, b, rho_a, rho_b, 1 - similarity), rho = rho0 |F^T n| (F per cell)"""
        n = self.n
        rmax = self._rho0*F.flatten(1).norm(dim=1)                                         # (|F|_F over its largest singular value: a superset)
        cut = 2*float(rmax.max())+margin
        xf = x.float(); I_, J_ = [], []
        for c0 in range(0, n, 1024):
            ii, jj = torch.nonzero(torch.cdist(xf[c0:c0+1024], xf) < cut, as_tuple=True)
            ii = ii+c0; k = ii < jj
            I_.append(ii[k]); J_.append(jj[k])
        a, b = torch.cat(I_), torch.cat(J_)
        L = (x[b]-x[a]).norm(dim=1)
        k = (L < rmax[a]+rmax[b]+margin) & ~torch.isin(a*n+b, self.tkey)
        a, b = a[k], b[k]
        if self.jump_share:                                                            # (reach first; the share: a wall's as update_ul read it, the others' on the half-way split)
            d = x[b]-x[a]; L = d.norm(dim=1); nn = d/L[:, None]
            ra = self._rho0[a]*torch.einsum('fba,fb->fa', F[a], nn).norm(dim=1)
            rb = self._rho0[b]*torch.einsum('fba,fb->fa', F[b], nn).norm(dim=1)
            k = L < ra+rb+margin
            a, b, d, L, ra, rb = a[k], b[k], d[k], L[k], ra[k], rb[k]
            s = torch.zeros(len(a), dtype=f64, device=x.device)
            isw = torch.zeros(len(a), dtype=torch.bool, device=x.device)
            if getattr(self, '_wkey', None) is not None and len(self._wkey) and len(a):
                pos = torch.searchsorted(self._wkey, a*n+b).clamp(max=len(self._wkey)-1); isw = self._wkey[pos] == a*n+b
                s[isw] = self._wshare[pos[isw]]
            nw = ~isw
            if bool(nw.any()):
                s[nw] = self.jump_sim(a[nw], b[nw], d[nw], torch.full((int(nw.sum()),), .5, dtype=f64, device=x.device))
            k = (1-s) > 1e-3
            return a[k], b[k], ra[k], rb[k], (1-s)[k], L[k]
        sim = color_sim(self._col_Q, a, b)
        k = sim < self.sim_birth
        a, b, sim = a[k], b[k], sim[k]
        d = x[b]-x[a]; L = d.norm(dim=1); nn = d/L[:, None]
        ra = self._rho0[a]*torch.einsum('fba,fb->fa', F[a], nn).norm(dim=1)
        rb = self._rho0[b]*torch.einsum('fba,fb->fa', F[b], nn).norm(dim=1)
        k = L < ra+rb+margin
        return a[k], b[k], ra[k], rb[k], 1-sim[k], L[k]

    def update_self(self, x):
        """the contact list at the step's start x (MidpointStep.begin, after update_ul): every unlike pair within reach + h, rho on the fit F_n there, frozen over the step; k_ry_contact on lambda = |d| / (rho_a + rho_b),
        Kc = 1/2 M V_c (1 - similarity), V_c the two cells' mean rest volume"""
        st = self.self_stats
        x = x.detach()
        F = self.ty['Fn'][self._gid]
        a, b, ra, rb, pw, L = self.contact_pairs(x, F, (.5 if self.jump_share else 1.)*self._h)   # (a step's relative motion is under .3 h; with the share every near pair is a candidate)
        m = len(a); K = self.selfc['K']
        if m > K:
            raise RuntimeError('update_self: %d contact pairs over the capacity %d' % (m, K))
        self._pairs, self._pw = torch.stack([a, b], 1), pw
        t = self.tsc
        t['i'].zero_(); t['k'].zero_(); t['Rs'].zero_(); t['Kc'].zero_()
        if m:
            t['i'][:m] = a.to(torch.int32); t['k'][:m] = b.to(torch.int32)
            t['Rs'][:m] = ra+rb; t['Kc'][:m] = .5*self.Mp*.5*(self.V0[a]+self.V0[b])*pw
            dl = (ra+rb-L)/(ra+rb)
            st['active'], st['d_max'] = int((dl > 0).sum()), float(dl.max())
        t['xn'].copy_(x)
        st['pairs'] = m

    def _skin(self, Mq, Mo_old, yld):
        """the outer layer is a set of unseen halves: Mo their weights, carried by the return as q (end_ul), toward the current cut's by the same part. At the first cut, the cut's"""
        if not self._skin_set:
            self._skin_set = True
            return Mq
        lam = self._ret[self._gid][yld][:, None, None]
        return Mo_old+lam*(Mq-Mo_old)

    def _regroup(self, yld):
        """a tiny cell with its heaviest half: its own weights do not move without a return of its own"""
        ms, lo, hi = self._m_s, self.tlo, self.thi
        tiny = self.V0 < self.agg_merge*self.V0.median()
        own = torch.cat([lo, hi]); oth = torch.cat([hi, lo]); w_ = torch.cat([self._wi, self._wj])
        k = tiny[own] & ~tiny[oth] & (w_ > 0)
        o = torch.argsort(w_[k]).cpu()
        gid = np.array(self._gid.cpu())
        gid[np.array(own[k].cpu())[o]] = np.array(oth[k].cpu())[o]                         # (numpy, in order: the last, the heaviest, stands)
        self._set_gid(torch.as_tensor(gid, device=self.dev))

    def _fit_on(self):
        """only a group with a return: its weights moved; a cell without one is not touched"""
        on = torch.zeros(self.n, dtype=torch.bool, device=self.dev)
        on[self._ret > 0] = True
        return on

    def _metric(self, Un_, M_, on_):
        """the weights hold M where the rest had it, so the force is the plain derivative of 1/2 tr(X M X^T): K = I"""
        return M_

    def start(self):
        super().start()
        self._wi = None                                                                    # (the rest weights: the first cut's, as RingYield's on the cut)
        self._skin_set = False

    def _spring_rule(self, lo, hi, key, yld):
        if self._wi is None:
            return super()._spring_rule(lo, hi, key, yld)
        wt = self._wi+self._wj
        ret = (self._ret[self._gid[self.tlo]] > 0) | (self._ret[self._gid[self.thi]] > 0)
        stay = torch.isin(self.tkey, key) | (wt > self._wgone) | ((wt > 0) & ~ret)          # (next to no weight: it leaves, in a return only)
        new = ~torch.isin(key, self.tkey)
        if bool(new.any()) and self._col_Q is not None:                                    # (like colours only: material at rest; the rest a contact, update_self)
            k = torch.nonzero(new)[:, 0]
            near = color_sim(self._col_Q, lo[k], hi[k]) >= self.sim_birth
            new[k[~near]] = False
        elif bool(new.any()):
            a_, b_ = np.array(self.tlo.cpu()), np.array(self.thi.cpu())
            A = sp.csr_matrix((np.ones(len(a_)), (a_, b_)), shape=(self.n, self.n))
            A = A+A.T
            A2 = (A@A).tocsr()
            k = torch.nonzero(new)[:, 0]
            near = np.asarray(A2[np.array(lo[k].cpu()), np.array(hi[k].cpu())]).ravel() > 0
            new[k[torch.as_tensor(~near, device=self.dev)]] = False
        if not self.elastic_births:                                                        # (a wall that no return has reached is a free surface to its cells: no bond, not even through the volume)
            new = new & ((self._ret[self._gid[lo]] > 0) | (self._ret[self._gid[hi]] > 0))
        return new, stay

    def _spring_k(self, x, lo_s, hi_s, key_s, ds, k0_, both_y, has_wall, Aw, wall):
        if self._wi is None:
            return super()._spring_k(x, lo_s, hi_s, key_s, ds, k0_, both_y, has_wall, Aw, wall)
        pos = torch.searchsorted(self._qkey, key_s).clamp(max=len(self._qkey)-1); kept = self._qkey[pos] == key_s
        ms = len(key_s)
        z = lambda: torch.zeros(ms, dtype=f64, device=self.dev)
        wi, wj = z(), z()
        wi[kept], wj[kept] = self._wi[pos[kept]], self._wj[pos[kept]]
        Fti = self._inv(self._Ft)
        qi, qj = torch.einsum('fab,fb->fa', Fti[lo_s], ds), torch.einsum('fab,fb->fa', Fti[hi_s], ds)
        ki, kj = kept & (wi > 0), kept & (wj > 0)                                          # (a half of weight 0 has no memory: q reads no remainder until it has weight)
        qi[ki], qj[kj] = self._qi[pos[ki]], self._qj[pos[kj]]
        i, j, c, order, pw = wall
        dw = x[j]-x[i]; Lw = dw.norm(dim=1)
        tw = ((c-x[i])*dw).sum(1)/Lw**2
        tlo = torch.where(i < j, tw, 1-tw)[order]
        share = torch.full((ms,), .5, dtype=f64, device=self.dev)
        share[has_wall] = tlo[pw[has_wall]].clamp(.02, .98)
        ww = z()
        ww[has_wall] = self.mu*Aw[has_wall]/ds.norm(dim=1)[has_wall]
        wi, wj = self._reweight(lo_s, hi_s, qi, qj, wi, wj, 2*share*ww, 2*(1-share)*ww, has_wall)
        self._wi, self._wj = wi, wj
        self._qkey_n, self._qi_n, self._qj_n = key_s.clone(), qi, qj
        return self.kshare*(wi+wj)

    def _set_halves(self, x):
        T = self.tr
        if self._wi is None:
            super()._set_halves(x)
            ms = self._m_s
            self._wi, self._wj = T['wi'][:ms].clone(), T['wj'][:ms].clone()
            self._Ms, self._Cs = self._moments(self.tlo, self.thi, self._qi, self._qj, self._wi, self._wj)
            m = self._Ms.diagonal(dim1=1, dim2=2).sum(1)/3
            self._sm = m.clamp(min=1e-9*float(m.median()))
            self._sc = self._sm/self.V0**(1/3)
            self._Mlast = self._Ms.clone()
            self._ret = torch.zeros(self.n, dtype=f64, device=self.dev)
            self._wgone = 1e-3*float((self._wi+self._wj).median())                         # (a spring without a wall under this carries next to nothing: it leaves)
            return
        ms, lo, hi = self._m_s, self.tlo, self.thi
        self._qkey, self._qi, self._qj = self._qkey_n, self._qi_n, self._qj_n
        for k_ in ('i', 'j', 'wi', 'wj', 'qi', 'qj', 'dni', 'dnj'):
            T[k_].zero_()
        T['i'][:ms] = lo.to(torch.int32); T['j'][:ms] = hi.to(torch.int32); T['wi'][:ms] = self._wi; T['wj'][:ms] = self._wj; T['qi'][:ms] = self._qi; T['qj'][:ms] = self._qj

    def _moments(self, lo, hi, qi, qj, wi, wj):
        n, dev = self.n, self.dev
        M = torch.zeros(n, 3, 3, dtype=f64, device=dev)
        M.index_add_(0, lo, wi[:, None, None]*qi[:, :, None]*qi[:, None, :]); M.index_add_(0, hi, wj[:, None, None]*qj[:, :, None]*qj[:, None, :])
        s = wi[:, None]*qi+wj[:, None]*qj
        C = torch.zeros(n, 3, dtype=f64, device=dev).index_add(0, lo, s).index_add(0, hi, -s)
        return M, C

    def _reweight(self, lo, hi, qi, qj, wi, wj, wwi, wwj, has_wall):
        """after this step's return: each cell's halves toward the walls' weights by the lambda that best undoes what the return did to its M. No return, no lambda"""
        st, sm = self.ul_stats, self._sm
        M, _ = self._moments(lo, hi, qi, qj, wi, wj)
        Mw, _ = self._moments(lo, hi, qi, qj, wwi, wwj)
        E = self._Mlast-M
        lam = self._ret[self._gid]
        wn_i, wn_j = wi+lam[lo]*(wwi-wi), wj+lam[hi]*(wwj-wj)
        Mn, Cn = self._moments(lo, hi, qi, qj, wn_i, wn_j)
        bg = self.V0 >= self.agg_merge*self.V0.median()
        me, ce = ((self._Ms-Mn)/sm[:, None, None]).flatten(1).norm(dim=1), ((self._Cs-Cn)/self._sc[:, None]).norm(dim=1)
        st['lam_med'], st['lam_max'] = float(lam.median()), float(lam.max())
        st['m_err'], st['m_err_max'], st['c_err'], st['c_err_max'] = float(me[bg].median()), float(me[bg].max()), float(ce[bg].median()), float(ce[bg].max())
        st['undone'] = float(((Mn-self._Mlast)/sm[:, None, None]).flatten(1).norm(dim=1)[bg].median()/(E/sm[:, None, None]).flatten(1).norm(dim=1)[bg].median().clamp(min=1e-300))
        st['dw'] = float((wn_i-wi).abs().sum()+(wn_j-wj).abs().sum())/float((wi+wj).sum())
        st['gap'] = float((wn_i-wwi).abs().sum()+(wn_j-wwj).abs().sum())/float((wwi+wwj).sum())
        st['ghost_w'] = float((wn_i[~has_wall].sum()+wn_j[~has_wall].sum())/(wn_i.sum()+wn_j.sum()))
        return wn_i, wn_j

    def _v6(self, M):
        s2 = 2**.5
        return torch.stack([M[:, 0, 0], M[:, 1, 1], M[:, 2, 2], s2*M[:, 0, 1], s2*M[:, 0, 2], s2*M[:, 1, 2]], 1)

    def _sum6(self, lo, hi, ri, rj, ai, aj):
        return torch.zeros(self.n, 6, dtype=f64, device=self.dev).index_add(0, lo, ai[:, None]*ri).index_add(0, hi, aj[:, None]*rj)

    def _r6(self, q):
        s2 = 2**.5
        return torch.stack([q[:, 0]*q[:, 0], q[:, 1]*q[:, 1], q[:, 2]*q[:, 2], s2*q[:, 0]*q[:, 1], s2*q[:, 0]*q[:, 2], s2*q[:, 1]*q[:, 2]], 1)

    def end_ul(self, th=.5):
        T, ms, Y = self.tr, self._m_s, self.ty
        self._Mlast, _ = self._moments(self.tlo, self.thi, T['qi'][:ms], T['qj'][:ms], T['wi'][:ms], T['wj'][:ms])     # (M before this step's return)
        Fe = ((Y['Ff']-(1-th)*Y['Fn'])/th)[self._gid].clone()
        super().end_ul(th)
        m = self._ret[self._gid] > 0
        if bool(m.any()):
            Tr = self._inv(self._Ft[m])@Fe[m]                                              # (the return on the outer layer, as on q)
            T['Mo'][m] = Tr@T['Mo'][m]@Tr.transpose(1, 2)


def exact_blocks(W, z, g0, cm, h=1e-9):
    """the diagonal 3 x 3 blocks of the frozen problem's Hessian, cm on the diagonal included: by products on the colours of the springs' graph squared, nothing assumed of the terms. Returns the blocks and the products used"""
    n, dev = W.n, W.dev
    a_, b_ = np.array(W.tlo.cpu()), np.array(W.thi.cpu())
    A = sp.csr_matrix((np.ones(len(a_)), (a_, b_)), shape=(n, n)); A = A+A.T
    A2 = (A+A@A).tocsr()
    col = np.full(n, -1)
    ip, jx = A2.indptr, A2.indices
    for i in np.argsort(-np.diff(ip)):                                                     # (greedy, the largest first)
        c = col[jx[ip[i]:ip[i+1]]]
        used = set(c[c >= 0].tolist())
        k = 0
        while k in used:
            k += 1
        col[i] = k
    nc = int(col.max())+1
    col_t = torch.as_tensor(col, device=dev)
    K = torch.zeros(n, 3, 3, dtype=f64, device=dev)
    for c in range(nc):
        m = col_t == c
        for ax in range(3):
            d = torch.zeros(n, 3, dtype=f64, device=dev); d[m, ax] = 1.
            _, g1 = W.grad(z+h*d)
            K[m, :, ax] = ((g1-g0)/h)[m]
    K = .5*(K+K.transpose(1, 2))
    K = K.nan_to_num(0., 0., 0.)
    ev, Q = torch.linalg.eigh(K.cpu())                                                     # (a preconditioner: positive; near a contact's kink or a flipped cell a block may not be)
    W.ul_stats['block_min'] = float((ev.to(dev)[:, 0]/cm).min()); W.ul_stats['block_neg'] = int((ev.to(dev)[:, 0] < 0).sum())
    ev = ev.to(dev).clamp(min=0.)+cm[:, None]
    Q = Q.to(dev)
    K = (Q*ev[:, None, :])@Q.transpose(1, 2)
    return K, 3*nc


class ExplicitStep(YieldStep):
    """leapfrog between the steps of the model: the cut, the weights and the return every dt as before (theta = 1: the return's stress read at x, q frozen in between, as one implicit step's constitutive update), the forces at substeps dt / ns. ns from the largest frequency of M^-1 K at the step's start (warm power iteration on the frozen forces), to cfl of the stability limit. No CG, no outer iteration"""
    cfl, ns_max, power_its = .7, 20000, 8

    def _omega2(self, x, g0, m3):
        W = self.W
        d = self._pv+.1*torch.randn_like(x) if getattr(self, '_pv', None) is not None else torch.randn_like(x)         # (the noise: a new mode above the old one grows out of it)
        its = self.power_its if getattr(self, '_pv', None) is not None else 60
        lam = 0.
        for _ in range(its):
            s_ = float(d.abs().max())
            d = d/s_
            _, g1 = W.grad(x+1e-9*d)
            Kd = (g1-g0)/1e-9
            lam = float((d*Kd).sum()/(m3*d*d).sum())
            d = Kd/m3
        self._pv = d/float(d.abs().max())
        # the ground: not in the frozen forces until it touches. A ball's stiffness at twice its reference depth on its own mass; a cell that may reach it within the step
        R = W.tbR
        m = m3[:, 0]
        near = (R > 0)
        lg = 0.
        if bool(near.any()):
            ng = torch.as_tensor([float(W.ball['ng'][k]) for k in range(3)], dtype=f64, device=W.dev)
            dep = R-((x*ng).sum(1)-float(W.tball[2]))
            reach = dep+2*self.dt*self.tv.norm(dim=1) > 0
            k_ = 2*W.Es*torch.sqrt(R*torch.maximum(dep+2*self.dt*self.tv.norm(dim=1), .1*R).clamp(min=0.))
            k_ = torch.where(near & reach, k_, torch.zeros_like(k_))
            lg = float((k_/m).max())
        self._lam_ground = lg
        return max(lam, lg, 1.)

    def solve(self, tol_abs, cg_tol_abs):
        W, A, bl = self.W, self.a, self.W.ball
        wp.launch(k_mid_ball, bl['k'], inputs=[A['x'], bl['body'], bl['R'], bl['ng'], self.c0, self.g8, self.cb], outputs=[bl['xn'], bl['kt'], bl['bound']])
        if getattr(W, 'hold', False):
            W.hold_start(self.tx, self.muf)
        m3 = self._m3; fg = self._fg
        x, v = self.tx.clone(), self.tv.clone()
        W.grad(x); W.reread()
        _, g = W.grad(x); g = g.clone()
        lam = self._omega2(x, g, m3)
        ns = min(int(math.ceil(self.dt*math.sqrt(1.15*lam)/(2*self.cfl))), self.ns_max)
        ds = self.dt/ns
        W.grad(x)
        a = (fg-g)/m3
        for k in range(ns):
            v = v+.5*ds*a
            x = x+ds*v
            _, g = W.grad(x)
            a = (fg-g)/m3
            v = v+.5*ds*a
            W.reread_light()
        W.reread()
        _, g = W.grad(x)
        a = (fg-g.clone())/m3
        self._xe, self._ve, self._ae = x, v, a
        self.stats = dict(rejected=0, edge=0, negative=0, trace=[], outer=0, ns=ns, lam=lam)
        ok = bool(torch.isfinite(x).all()) and bool(torch.isfinite(v).all())
        return ns, ns, 0. if ok else math.inf

    def end(self):
        W, A, bl = self.W, self.a, self.W.ball
        W.end_ul(self.th)
        wp.launch(k_mid_lamt, bl['k'], inputs=[W.B['bff'], W.B['bdelta']], outputs=[bl['lamt']])
        self.tx.copy_(self._xe); self.tv.copy_(self._ve); self.ta.copy_(self._ae); self.tz.copy_(self._xe)


def color_code(W, r0, sigma):
    """ClosureWarp._color_code (10-30, the user's colour) on a RingMoment: a static colour per cell from the rest skin graph, exp(-d^2 / (2 sigma^2)) in
    steps, the inside by the harmonic extension on the rest A / L; the similarity of two cells sum sqrt(p_i p_j). Returns sqrt(p), n x skin cells"""
    import scipy.sparse as sp_
    from scipy.sparse.linalg import splu
    from scipy.sparse.csgraph import dijkstra
    n = W.n; X = W.X
    G, _ = W._recut.geometry(X, r0)
    kp = G.wall_A.detach() > 0
    i0, j0 = G.i[kp], G.j[kp]
    w = (G.wall_A.detach()[kp]/(X[j0]-X[i0]).norm(dim=1)).cpu().numpy()
    Wm = sp_.coo_matrix((w, (i0.cpu().numpy(), j0.cpu().numpy())), shape=(n, n)).tocsr(); Wm = Wm+Wm.T
    skin = (G.patch_A.detach() > 0).cpu().numpy(); sid, iid = np.where(skin)[0], np.where(~skin)[0]
    ds = dijkstra((Wm[sid][:, sid] > 0).astype(np.float64).tocsr(), unweighted=True)
    Ps = np.exp(-np.where(np.isfinite(ds), ds, 1e6)**2/(2*sigma**2)); Ps = Ps/Ps.sum(1, keepdims=True)
    Pm = np.zeros((n, len(sid)), dtype=np.float32); Pm[sid] = Ps
    Lm = (sp_.diags(np.asarray(Wm.sum(1)).ravel())-Wm).tocsc()
    Pm[iid] = splu(Lm[iid][:, iid].tocsc()).solve(np.asarray(Wm[iid][:, sid]@Ps)).astype(np.float32)
    return torch.as_tensor(np.sqrt(np.clip(Pm, 0, None)), device=W.dev)


def color_sim(Q, a, b, chunk=4096):
    s = torch.zeros(len(a), dtype=f64, device=a.device)
    for c0 in range(0, len(a), chunk):
        s[c0:c0+chunk] = (Q[a[c0:c0+chunk]]*Q[b[c0:c0+chunk]]).sum(1).to(f64).clamp(max=1.)
    return s
