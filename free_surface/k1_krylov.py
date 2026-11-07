"""Persistent, reorthogonalized Lanczos trust-region solve for MergedStep.

One context belongs to one Newton point and one fixed preconditioner. Rejected
trials may change its radius, but accepted points and C rereads invalidate it.
The physical energy, acceptance rule, and step-end update belong to MergedStep.
"""
import math
import time
import numpy as np
import scipy.linalg as sla
import torch
import warp as wp

from free_surface.k1_tangent import Tangent


def _spectral(B, rhs, radius):
    """General symmetric reduced problem, including the indefinite hard case."""
    ev, U = sla.eigh(B, check_finite=False)
    a = U.T @ rhs
    lower = max(0., -float(ev[0]))
    eps = np.finfo(float).eps * max(1., float(np.max(np.abs(ev)))) * 16
    den = ev + lower
    singular = den <= eps
    y = np.zeros_like(a)
    y[~singular] = -a[~singular] / den[~singular]
    if np.linalg.norm(a[singular]) <= eps * max(1., np.linalg.norm(rhs)) and np.linalg.norm(y) <= radius:
        if lower > 0:
            y[np.flatnonzero(singular)[0]] = math.sqrt(max(radius**2 - y@y, 0.))
        return U@y, lower
    hi = max(lower + 1., np.linalg.norm(rhs)/radius + lower)
    for _ in range(100):
        if np.linalg.norm(a/(ev+hi)) <= radius:
            break
        hi = lower + 2*(hi-lower)
    lo = lower
    for _ in range(60):
        mid = .5*(lo+hi)
        if np.linalg.norm(a/(ev+mid)) > radius:
            lo = mid
        else:
            hi = mid
    return U@(-a/(ev+hi)), hi


def reduced_step(diag, off, rhs, radius, warm=0., dense=None):
    """Tridiagonal shifted Cholesky and safeguarded Newton on the radius."""
    if dense is not None:
        return _spectral(dense, rhs, radius)
    band = np.zeros((2, len(diag)))
    band[0] = diag
    band[1, :-1] = off

    def at(lam):
        work = band.copy()
        work[0] += lam
        factor = sla.cholesky_banded(work, lower=True, check_finite=False)
        y = sla.cho_solve_banded((factor, True), -rhs, check_finite=False)
        return y, factor

    try:
        y, _ = at(0.)
    except sla.LinAlgError:
        B = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)
        return _spectral(B, rhs, radius)
    if np.linalg.norm(y) <= radius:
        return y, 0.
    lo = 0.
    hi = max(float(warm), float(np.linalg.norm(rhs)/radius), 1e-14)
    for _ in range(100):
        yhi, _ = at(hi)
        if np.linalg.norm(yhi) <= radius:
            break
        hi *= 2
    lam = min(max(float(warm), lo), hi)
    if not lo < lam < hi:
        lam = .5*(lo+hi)
    for _ in range(45):
        y, factor = at(lam)
        norm = np.linalg.norm(y)
        if norm <= radius:
            hi, yhi = lam, y
            if radius-norm <= 1e-9*radius:
                break
        else:
            lo = lam
        invy = sla.cho_solve_banded((factor, True), y, check_finite=False)
        deriv = -(y@invy)/norm
        candidate = lam-(norm-radius)/deriv
        lam = candidate if lo < candidate < hi else .5*(lo+hi)
    return yhi, hi


class KrylovTR:
    def __init__(self, ST):
        self.ST, self.CG = ST, ST.CG
        self.T = Tangent(ST.CG)
        self.valid = False
        self.storage = None
        self.records = []
        self.graphs = {}
        self.graph_build_seconds = 0.

    def invalidate(self):
        self.valid = False

    def _map(self, v, transpose=False):
        mat = self.L.transpose(1, 2) if transpose else self.L
        out = torch.einsum('nij,nj->ni', mat, v.reshape(-1, 3))
        if self.group is not None:
            matj = self.Lj.T if transpose else self.Lj
            out[self.group] = (matj @ v.reshape(-1, 3)[self.group].flatten()).reshape(-1, 3)
        return out

    def _untranspose(self, v):
        out = torch.linalg.solve_triangular(self.L.transpose(1, 2), v.reshape(-1, 3, 1), upper=True).squeeze(-1)
        if self.group is not None:
            out[self.group] = torch.linalg.solve_triangular(self.Lj.T, v.reshape(-1, 3)[self.group].reshape(-1, 1), upper=True).reshape(-1, 3)
        return out

    def _begin(self, g, cap, joint=None):
        self.T.setbase()
        self.g = wp.to_torch(g).clone()
        self.gnorm = float(self.g.norm())
        self.group = None
        L = torch.linalg.cholesky(self.CG.t['Pinv'])
        if not hasattr(self, 'L'):
            self.L = torch.empty_like(L)
        self.L.copy_(L)
        self.P = wp.to_torch(self.ST.a['P']).clone()
        self.joint_cost = 0
        if joint is not None:
            # Captured joint kernels must keep the same index/matrix addresses;
            # only their contents change between Newton points.
            if not hasattr(self, 'joint_indices') or self.joint_indices.numel() != len(joint):
                self.joint_indices = torch.empty(len(joint), dtype=torch.long, device=self.g.device)
                self.joint_L = torch.empty((3*len(joint), 3*len(joint)), dtype=self.g.dtype, device=self.g.device)
                self.graphs = {}
            self.joint_indices.copy_(torch.as_tensor(joint, device=self.g.device))
            self.group = self.joint_indices
            columns = []
            for cell in joint:
                for axis in range(3):
                    self.CG.t['d'].zero_()
                    self.CG.t['d'][cell, axis] = 1.
                    wp.capture_launch(self.T.graph)
                    columns.append(self.CG.t['Hd'][self.group].flatten().clone())
            J = torch.stack(columns, 1)
            J = .5*(J+J.T)
            ev, U = torch.linalg.eigh(J)
            floor = self.CG.t['cm'][self.group].min()
            self.J = (U*ev.clamp(min=floor)) @ U.T
            self.joint_L.copy_(torch.linalg.cholesky(torch.linalg.inv(self.J)))
            self.Lj = self.joint_L
            self.joint_cost = len(columns)
        self.b = self._map(self.g, True).flatten()
        self.bn = float(self.b.norm())
        n = self.b.numel()
        if self.storage is None or self.storage[0].shape != (cap+1, n):
            self.graphs = {}
            self.storage = (torch.empty((cap+1, n), dtype=self.b.dtype, device=self.b.device),
                            torch.empty((cap, n), dtype=self.b.dtype, device=self.b.device),
                            torch.empty(cap, dtype=self.b.dtype, device=self.b.device),
                            torch.empty(cap, dtype=self.b.dtype, device=self.b.device))
        self.Q, self.AQ, self.diag, self.off = self.storage
        self.Q[0].copy_(-self.b/self.bn)
        self.k = 0
        self.lam = 0.
        self.dense_mode = False
        self.breakdown = False
        self.valid = True

    def _advance(self, start, end, capturing=False):
        for j in range(start, end):
            q = self.Q[j]
            self.CG.t['d'].copy_(self._map(q))
            if capturing:
                self.T.run()
            else:
                wp.capture_launch(self.T.graph)
            v = self._map(self.CG.t['Hd'], True).flatten()
            self.AQ[j].copy_(v)
            self.diag[j] = torch.dot(q, v)
            for _ in range(2):
                v -= (self.Q[:j+1] @ v) @ self.Q[:j+1]
            beta = v.norm()
            self.off[j] = beta
            self.Q[j+1].copy_(v/beta.clamp(min=1e-300))

    def _extend(self, end):
        key = (self.group is not None, self.k, end)
        if key not in self.graphs:
            begin = time.perf_counter()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream()), sync_enter=False, sync_exit=False):
                    self._advance(self.k, end, capturing=True)
            self.graphs[key] = graph
            self.graph_build_seconds += time.perf_counter()-begin
        self.graphs[key].replay()
        self.k = end

    def _candidate(self, radius):
        k = self.k
        # Both the represented operator and RHS use the actually stored basis.
        host = torch.stack((self.diag[:k], self.off[:k], self.Q[:k] @ self.b)).cpu().numpy()
        dd, oo, rhs = host[0], host[1, :-1], host[2]
        dense = None
        if self.dense_mode:
            B = self.Q[:k] @ self.AQ[:k].T
            dense = (.5*(B+B.T)).cpu().numpy()
        coef, self.lam = reduced_step(dd, oo, rhs, radius, self.lam, dense)
        self.lam = float(self.lam)
        c = torch.as_tensor(coef, dtype=self.b.dtype, device=self.b.device)
        y = c @ self.Q[:k]
        Ay = c @ self.AQ[:k]
        p = self._map(y)
        hp = self._untranspose(Ay)
        pp = self._untranspose(y)
        residual = self.g + hp + self.lam*pp
        # A large projected residual means the tridiagonal representation lost
        # information under finite-precision reorthogonalization. Use Q^T H Q.
        projected = self.Q[:k] @ self._map(residual, True).flatten()
        projection_error = float(projected.norm())/max(self.bn, 1e-300)
        return p, hp, pp, y, residual, projection_error

    def solve(self, g, delta, stop, cap, allow_joint=True):
        reused = self.valid and self.k > 0
        if not self.valid:
            self._begin(g, cap)
        initial_k = self.k
        radius = float(delta) if delta > 0 else self.bn
        products = self.joint_cost
        self.joint_cost = 0
        checks = 0
        certified = False
        while True:
            if self.k == 0:
                self._extend(1)
            p, hp, pp, y, residual, projection_error = self._candidate(radius)
            if projection_error > 1e-7 and not self.dense_mode:
                self.dense_mode = True
                continue
            rn = float(residual.norm())
            if rn <= stop or self.k >= cap:
                self.CG.t['d'].copy_(p)
                wp.capture_launch(self.T.graph)
                checks += 1
                hp = self.CG.t['Hd'].clone()
                true_residual = self.g+hp+self.lam*pp
                rn = float(true_residual.norm())
                certified = rn <= stop
                if certified or self.k >= cap:
                    break
                if not self.dense_mode:
                    self.dense_mode = True
                    continue
            # Check short spaces cheaply, then advance in blocks of five.
            end = min(cap, 2 if self.k == 1 else (5 if self.k < 5 else self.k+5))
            self._extend(end)
        products += self.k-initial_k+checks
        self.CG.t['p'].copy_(p)
        self.CG.t['Hp'].copy_(hp)
        self.CG.t['r'].copy_(-self.g-hp)
        pmp, gp, php, rr = torch.stack((torch.dot(y,y), (self.g*p).sum(), (p*hp).sum(), (self.g+hp).square().sum())).tolist()
        record = dict(basis=self.k, added=self.k-initial_k, reused=reused, products=products,
                      joint=None if self.group is None else self.group.tolist(), dense=self.dense_mode,
                      kkt_relative=rn/max(self.gnorm, 1e-300), certified=certified,
                      radius=radius, multiplier=self.lam, norm=math.sqrt(pmp), projection_error=projection_error)
        self.records.append(record)
        result = dict(K=products, EDGE=self.lam>0, NEG=False, PMP=pmp, DELTA=radius, GP=gp, PHP=php, RR=rr)
        # Preserve the production fallback's scope: an exhausted interior solve.
        if not certified and self.k >= cap and self.lam == 0 and self.group is None and allow_joint:
            group = torch.argsort(self.CG.t['r'].norm(dim=1), descending=True)[:6].tolist()
            self._begin(g, cap, joint=group)
            result = self.solve(g, radius, stop, cap, allow_joint=False)
            result['K'] += products
        return result
