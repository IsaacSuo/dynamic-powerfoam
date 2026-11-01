"""RingYield with the springs not tied to the walls: the cut goes on every step, for the volume, the contact and the candidates; a spring's weight moves only with a plastic return.
  a return gives up the part lam = 1 - c of the cell's deviatoric strain; its halves give up the same part of their distance to the current walls' weights, w' = w + lam (A / L - w); the outer layer as a set of halves
    square lattice in simple shear, plastic shear g, elastic e: M fixes the three pairs, w_old = 1 - g, w_new = g, while the walls read g + e: dw / (wall - w) = dg / e = lam
    uniform flow: lam is the same everywhere, and any mix of closed weights is closed, so the patch test holds
    no return, lam = 0: q, weights, outer layer, group do not move, whatever the walls do; the springs' energy 1/2 sum w |d - (1 + v) R q|^2 does not see the cut. e_y to 0: lam to 1, the walls' weights
  force: K = I, the plain derivative of 1/2 tr(X M X^T); R = polar(F), the same in all cells under the same F; a torque of M's non-roundness times e is the price (the theorem)
  a new wall gets a spring of weight 0 if its cells are two springs apart (a fold's two sheets are not: a wall between them is a free surface to both); a spring leaves when it has no wall and next to no weight"""
import numpy as np
import scipy.sparse as sp
import torch

from free_surface.ring_yield import RingYield

f64 = torch.float64


class RingMoment(RingYield):
    stat_keys = ('rows', 'ghost', 'born', 'closed', 'm_err', 'm_err_max', 'c_err', 'c_err_max', 'lam_med', 'lam_max', 'undone', 'dw', 'gap', 'ghost_w')

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._wi = None

    ghost_in_fit = False                                                                   # (a spring without a wall has no flux: the volume is the walls')

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
        if bool(new.any()):
            a_, b_ = np.array(self.tlo.cpu()), np.array(self.thi.cpu())
            A = sp.csr_matrix((np.ones(len(a_)), (a_, b_)), shape=(self.n, self.n))
            A = A+A.T
            A2 = (A@A).tocsr()
            k = torch.nonzero(new)[:, 0]
            near = np.asarray(A2[np.array(lo[k].cpu()), np.array(hi[k].cpu())]).ravel() > 0
            new[k[torch.as_tensor(~near, device=self.dev)]] = False
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
