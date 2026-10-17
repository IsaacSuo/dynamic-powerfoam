"""First-order VEM elastodynamics on Power Foam polyhedra (vem_geometry.py), differentiable in torch.

Unknowns are the polyhedron vertices x_v. Per element E (an agglomerate of material cells):
  F_E = sum_v (x_v - xbar_E) (x) g_Ev                    (VEM projection, exact for affine x)
  r_Ev = x_v - xbar_E - F_E (X_v - Xbar_E)              (non-affine remainder, objective)
  energy = sum_E |E| psi(F_E) + 1/2 gamma mu sum_E |E| / (h_E^2 n_E) sum_v |r_Ev|^2
psi is PAC-NeRF's compressible Neo-Hookean, mu/2 (|F|^2 - 3) - mu log J + lam/2 (log J)^2, with
log J continued quadratically below J = barrier (finite under inversion), as in EnergyFoam.
The second term is the standard VEM stabilisation (h_E = |E|^(1/3), n_E vertices of E); the
nonaffine part has no other stiffness. Forces are closed form (checked against autograd).

Lumped mass rho |E| / n_E per vertex. Symplectic Euler (velocity first), gravity, ground at y = 0:
'noslip' (the DEM's): a vertex that would cross the plane in this step loses its tangential
velocity and lands on it; it may leave again. 'layer' (default): PAC-NeRF's sticky ground as an
MPM particle sees it -- grid nodes at y <= 0 are zeroed, so a point at height y keeps the share
1 - W(y / dx) of its velocity, W the quadratic B-spline weight of those nodes (dx = 2 cm, the GT
grid); applied as (1 - W)^(dt / dt_mpm) so the loss rate per unit time matches MPM's step 1/4800.
A smooth layer a few cm thick instead of pinning single vertices; the body may leave the ground.

Probes (e.g. the DEM body centres, or Power Foam sites) ride with their element's affine map
xbar_E + F_E (P - Xbar_E); vertices are shared, so neighbouring elements' maps nearly agree.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

RHO = 1000.


def _matvec(A, b):
    """Batched 3x3 @ 3: elementwise (einsum dispatches millions of tiny cuBLAS gemv, ~10x slower)."""
    return (A*b[:, None, :]).sum(2)


def polar_newton(F, iters=8):
    """Rotation of the polar decomposition, R <- (R + R^-T) / 2 from R = F (det F > 0); used detached."""
    R = F
    for _ in range(iters):
        c0, c1, c2 = R[:, :, 0], R[:, :, 1], R[:, :, 2]
        cof = torch.stack((torch.cross(c1, c2, dim=1), torch.cross(c2, c0, dim=1), torch.cross(c0, c1, dim=1)), 2)
        J = (c0*cof[:, :, 0]).sum(1)
        R = .5*(R+cof/J[:, None, None])                           # R^-T = cof / det
    return R


def lame(E, nu):
    return E*nu/((1+nu)*(1-2*nu)), E/(2*(1+nu))


class VEMSolid:
    """params = (E, nu, s): s is gamma (stab='gamma', vertex-residual penalty) or beta (stab='pyramid':
    energy (1 - beta) sum_E |E| psi(F_E) + beta sum_s |s| psi(F_s) over the pyramid sub-cells of
    vem_geometry.py --pyramids, Wriggers' mixed VEM-FEM stabilisation as in Cihan et al. 2020), or
    beta for stab='pyrlin': sum_E |E| psi(F_E) + beta mu / 2 sum_s |s| |F_s - F_E(s)|^2 -- the same
    sub-cells, but a constant-stiffness penalty on their deviation from the element's F: zero for
    affine and rigid motion, objective, rank-sufficient, and it does not stiffen when a small
    sub-cell is crushed (the pyramid-NH variants blew up there)."""
    def __init__(self, geometry, device='cuda', dtype=torch.float32, barrier=.05, stab=None, min_sub=1e-6, max_flat=10.,
                 sub_law='snh', points=True, cluster_tol=0.):
        d = dict(np.load(geometry)) if isinstance(geometry, (str, Path)) else geometry
        self.stab = stab or ('pyramid' if 'sub_id' in d else 'gamma')
        self.sub_law = sub_law
        T = lambda a, dt=dtype: torch.as_tensor(np.asarray(a), dtype=dt, device=device)
        self.device, self.dtype, self.barrier = device, dtype, barrier
        self.X = T(d['X']); self.nv = len(self.X)
        self.e, self.v = T(d['elem'], torch.long), T(d['vert'], torch.long)
        self.g = T(d['g']); self.volume = T(d['volume']); self.ne = len(self.volume)
        self.count = torch.bincount(self.e, minlength=self.ne).to(dtype)
        self.Xbar = self._mean(self.X[self.v])
        self.Xrel = self.X[self.v]-self.Xbar[self.e]
        self.stab_w = self.volume/(self.volume**(2/3)*self.count)       # |E| / (h^2 n)
        if self.stab in ('pyramid', 'pyrlin', 'pyrtan'):
            # drop sub-cells below min_sub x their element's volume (round-off dominated F, negligible
            # energy) and very flat ones (max |g| h_s > max_flat: they alone set the explicit step;
            # the face is still held by the neighbour's pyramid on its other side)
            flat = np.zeros(len(d['sub_volume']))
            np.maximum.at(flat, d['sub_id'], np.linalg.norm(d['sub_g'], axis=1))
            flat *= np.cbrt(np.abs(d['sub_volume']))
            keep_sub = (d['sub_volume'] >= min_sub*d['volume'][d['sub_elem']]) & (flat <= max_flat)
            self.dropped_sub_count = int((~keep_sub).sum())
            new = -np.ones(len(keep_sub), int); new[keep_sub] = np.arange(keep_sub.sum())
            rows = keep_sub[d['sub_id']]
            self.s_id, self.s_v = T(new[d['sub_id'][rows]], torch.long), T(d['sub_vert'][rows], torch.long)
            self.s_g, self.s_vol = T(d['sub_g'][rows]), T(d['sub_volume'][keep_sub])
            self.s_elem = T(d['sub_elem'][keep_sub], torch.long); self.ns = int(keep_sub.sum())
            self.points = 'sp_id' in d and points
            if self.points:
                # point-based rows: [vertices, face centres, cell apexes], centres/apexes averaged once per step
                prow = keep_sub[d['sp_id']]
                self.p_id, self.p_pt = T(new[d['sp_id'][prow]], torch.long), T(d['sp_pt'][prow], torch.long)
                self.p_g = T(d['sp_g'][prow])
                self.nf, self.nc = int(d['n_face']), int(d['n_cell'])
                self.fc_face, self.fc_vert, self.fc_w = T(d['fc_face'], torch.long), T(d['fc_vert'], torch.long), T(d['fc_w'])
                self.ap_cell, self.ap_vert, self.ap_w = T(d['ap_cell'], torch.long), T(d['ap_vert'], torch.long), T(d['ap_w'])
            self.dropped_sub_volume = float(d['sub_volume'][~keep_sub].sum()/d['volume'].sum())
            ae, av = T(d['all_elem'], torch.long), T(d['all_vert'], torch.long)
            n_all = torch.bincount(ae, minlength=self.ne).to(dtype)
            self.mass = torch.zeros(self.nv, dtype=dtype, device=device).index_add_(0, av, (RHO*self.volume/n_all)[ae])
        else:
            self.mass = torch.zeros(self.nv, dtype=dtype, device=device).index_add_(0, self.v, (RHO*self.volume/self.count)[self.e])
        self.gravity = T([0., -9.8, 0.])
        self.ground = True
        self.ground_mode, self.layer_dx, self.layer_dt_ref = 'noslip', .02, 1/4800
        self.friction, self.band = .5, .01
        self.contact_omega, self.contact_zeta = 2000., .3
        self.cell_elem = d.get('cell_elem')
        self.ncl = 0
        if cluster_tol > 0:
            self.set_clusters(cluster_tol)

    def set_clusters(self, tol):
        """Vertices closer than tol (transitively) move together: one shared displacement per cluster.
        The rest geometry is unchanged (volumes, projections exact); only the vertex freedoms are
        reduced, so no zero-energy mode can appear, and short edges / sliver faces cannot shear."""
        from scipy import sparse
        from scipy.sparse.csgraph import connected_components
        from scipy.spatial import cKDTree
        X = self.X.double().cpu().numpy()
        pairs = cKDTree(X).query_pairs(tol, output_type='ndarray')
        g = sparse.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(X),)*2)
        ncl, cl = connected_components(g, directed=False)
        order = np.lexsort((X[:, 1], cl))                 # per cluster: lowest member first
        first = np.r_[True, cl[order][1:] != cl[order][:-1]]
        low = order[first]
        dev = self.X.device
        self.ncl, self.cl = int(ncl), torch.as_tensor(cl, device=dev)
        self.rep = self.low = torch.as_tensor(low, device=dev)
        self.cmass = torch.zeros(self.ncl, dtype=self.dtype, device=dev).index_add_(0, self.cl, self.mass)
        size = np.bincount(cl)
        lo = np.full((ncl, 3), np.inf); hi = np.full((ncl, 3), -np.inf)
        np.minimum.at(lo, cl, X); np.maximum.at(hi, cl, X)
        self.cluster_stats = dict(tol=tol, vertices=len(X), clusters=int(ncl), clustered_vertices=int((size[cl] > 1).sum()),
                                  max_members=int(size.max()), max_extent_mm=float(np.linalg.norm(hi-lo, axis=1).max()*1e3))
        return self.cluster_stats

    def _mean(self, a):
        return torch.zeros(self.ne, 3, dtype=a.dtype, device=a.device).index_add_(0, self.e, a)/self.count[:, None]

    def kinematics(self, x):
        xv = x[self.v]
        xbar = self._mean(xv)
        rel = xv-xbar[self.e]
        F = torch.zeros(self.ne, 3, 3, dtype=x.dtype, device=x.device).index_add_(0, self.e, rel[:, :, None]*self.g[:, None, :])
        r = rel-_matvec(F[self.e], self.Xrel)
        return xbar, F, r

    def _pyrtan_parts(self, x, xbar, F):
        """Tangent-consistent stabilisation: second-order term of sum_s |s| psi(F_s) about the element F_E
        (the first-order term vanishes since F_E is the volume mean of F_s): dF_s = F_s - F_E rotated back
        by the element's polar rotation R (detached), Q = R^T dF_s; strain e = sym Q charged with the
        Lame moduli, rotation w = skew Q with kappa mu (params[2], 0 = pure tangent)."""
        R = polar_newton(F.detach())
        self._Rs = R[self.s_elem]
        Q = self._Rs.transpose(1, 2)@(self.sub_F(x, xbar)-F[self.s_elem])
        return Q, .5*(Q+Q.transpose(1, 2)), .5*(Q-Q.transpose(1, 2))

    def _points(self, x):
        z = lambda n: torch.zeros(n, 3, dtype=x.dtype, device=x.device)
        yf = z(self.nf).index_add_(0, self.fc_face, x[self.fc_vert]*self.fc_w[:, None])
        ya = z(self.nc).index_add_(0, self.ap_cell, x[self.ap_vert]*self.ap_w[:, None])
        return torch.cat((x, yf, ya))

    def sub_F(self, x, xbar):
        if self.points:
            rel = self._points(x)[self.p_pt]-xbar[self.s_elem[self.p_id]]
            return torch.zeros(self.ns, 3, 3, dtype=x.dtype, device=x.device).index_add_(0, self.p_id, rel[:, :, None]*self.p_g[:, None, :])
        rel = x[self.s_v]-xbar[self.s_elem[self.s_id]]
        return torch.zeros(self.ns, 3, 3, dtype=x.dtype, device=x.device).index_add_(0, self.s_id, rel[:, :, None]*self.s_g[:, None, :])

    def sub_forces(self, x, A):
        """Vertex forces -sum_s A_s g_s (A = dE/dF_s per sub-cell), through the points if used."""
        if not self.points:
            return torch.zeros_like(x).index_add_(0, self.s_v, -_matvec(A[self.s_id], self.s_g))
        n = len(x)
        fp = torch.zeros(n+self.nf+self.nc, 3, dtype=x.dtype, device=x.device).index_add_(0, self.p_pt, -_matvec(A[self.p_id], self.p_g))
        f = fp[:n].clone()
        f.index_add_(0, self.fc_vert, fp[n+self.fc_face]*self.fc_w[:, None])
        f.index_add_(0, self.ap_vert, fp[n+self.nf+self.ap_cell]*self.ap_w[:, None])
        return f

    def _psi_P(self, F, lam, mu, stress=True, law='nh'):
        """'nh': PAC-NeRF's Neo-Hookean (log J barrier). 'snh': stable Neo-Hookean (Smith et al. 2018)
        mu/2 (|F|^2 - 3) + lam'/2 (J - a)^2 - const, lam' = lam + mu, a = 1 + mu/lam': same Lame
        moduli at small strain, zero at rest / rigid, finite stress under collapse and inversion."""
        c0, c1, c2 = F[:, :, 0], F[:, :, 1], F[:, :, 2]
        cof = torch.stack((torch.cross(c1, c2, dim=1), torch.cross(c2, c0, dim=1), torch.cross(c0, c1, dim=1)), 2)
        J = (c0*cof[:, :, 0]).sum(1)                                    # cof = dJ/dF, no inverse
        if law == 'snh':
            lp = lam+mu; a = 1+mu/lp
            psi = .5*mu*((F*F).sum((1, 2))-3)+.5*lp*((J-a)**2-(1-a)**2)
            return psi, (mu*F+(lp*(J-a))[:, None, None]*cof if stress else None)
        L, dL = self._log_J(J)
        psi = .5*mu*((F*F).sum((1, 2))-3)-mu*L+.5*lam*L*L
        if not stress:
            return psi, None
        return psi, mu*F+((lam*L-mu)*dL)[:, None, None]*cof

    def _log_J(self, J):
        m = self.barrier
        below = J < m
        Jc = torch.where(below, torch.full_like(J, m), J)
        L = torch.where(below, np.log(m)+(J-m)/m-(J-m)**2/(2*m*m), torch.log(Jc))
        dL = torch.where(below, 1/m-(J-m)/(m*m), 1/Jc)
        return L, dL

    def energy(self, x, params):
        E, nu, s = params
        lam, mu = lame(E, nu)
        xbar, F, r = self.kinematics(x)
        psi, _ = self._psi_P(F, lam, mu, stress=False)
        if self.stab == 'pyrlin':
            D = self.sub_F(x, xbar)-F[self.s_elem]
            return (self.volume*psi).sum()+.5*s*mu*(self.s_vol*(D*D).sum((1, 2))).sum()
        if self.stab == 'pyrtan':
            Q, e, w = self._pyrtan_parts(x, xbar, F)
            tr = e.diagonal(dim1=1, dim2=2).sum(1)
            dens = lam*tr*tr+2*mu*(e*e).sum((1, 2))+s*mu*(w*w).sum((1, 2))
            return (self.volume*psi).sum()+.5*(self.s_vol*dens).sum()
        if self.stab == 'pyramid':
            psi_s, _ = self._psi_P(self.sub_F(x, xbar), lam, mu, stress=False, law=self.sub_law)
            return (1-s)*(self.volume*psi).sum()+s*(self.s_vol*psi_s).sum()
        stab = .5*s*mu*(self.stab_w[self.e]*(r*r).sum(1)).sum()
        return (self.volume*psi).sum()+stab

    def forces(self, x, params):
        E, nu, gamma = params
        lam, mu = lame(E, nu)
        xbar, F, r = self.kinematics(x)
        _, P = self._psi_P(F, lam, mu)
        if self.stab == 'pyrtan':
            Q, e, w = self._pyrtan_parts(x, xbar, F)
            tr = e.diagonal(dim1=1, dim2=2).sum(1)
            eye = torch.eye(3, dtype=x.dtype, device=x.device)
            Sg = lam*tr[:, None, None]*eye+2*mu*e+gamma*mu*w                  # d dens/2 / dQ
            cD = self.s_vol[:, None, None]*(self._Rs@Sg)                       # dE/dF_s (R detached)
            back = torch.zeros(self.ne, 3, 3, dtype=x.dtype, device=x.device).index_add_(0, self.s_elem, cD)
            VP = self.volume[:, None, None]*P-back
            f = -_matvec(VP[self.e], self.g)
            return torch.zeros_like(x).index_add_(0, self.v, f)+self.sub_forces(x, cD)
        if self.stab == 'pyrlin':
            cD = (gamma*mu*self.s_vol)[:, None, None]*(self.sub_F(x, xbar)-F[self.s_elem])
            back = torch.zeros(self.ne, 3, 3, dtype=x.dtype, device=x.device).index_add_(0, self.s_elem, cD)
            VP = self.volume[:, None, None]*P-back
            f = -_matvec(VP[self.e], self.g)
            return torch.zeros_like(x).index_add_(0, self.v, f)+self.sub_forces(x, cD)
        if self.stab == 'pyramid':
            beta = gamma
            VP = ((1-beta)*self.volume)[:, None, None]*P
            f = -_matvec(VP[self.e], self.g)
            _, Ps = self._psi_P(self.sub_F(x, xbar), lam, mu, law=self.sub_law)
            VPs = (beta*self.s_vol)[:, None, None]*Ps
            return torch.zeros_like(x).index_add_(0, self.v, f)+self.sub_forces(x, VPs)
        VP = self.volume[:, None, None]*P
        f = -_matvec(VP[self.e], self.g)
        rsum = torch.zeros(self.ne, 3, dtype=x.dtype, device=x.device).index_add_(0, self.e, r)
        S = torch.zeros(self.ne, 3, 3, dtype=x.dtype, device=x.device).index_add_(0, self.e, r[:, :, None]*self.Xrel[:, None, :])
        c = (gamma*mu*self.stab_w)[self.e]
        grad = r-rsum[self.e]/self.count[self.e, None]-_matvec(S[self.e], self.g)
        f = f-c[:, None]*grad
        return torch.zeros_like(x).index_add_(0, self.v, f)

    def damping_forces(self, x, v, params):
        """Kelvin-Voigt viscosity with time constant tau = params[3] (stiffness-proportional: damping
        ratio tau omega / 2 per mode). Element: Cauchy stress 2 mu tau sym(dF/dt F^-1), P = sigma cof;
        pyrlin sub-cells: beta mu tau |s| d(F_s - F_E)/dt. Zero for rigid motion of the element,
        dissipative (power 2 mu tau J |D|^2 |E| + beta mu tau |s| |dD_s/dt|^2 >= 0)."""
        E, nu, beta, tau = params
        _, mu = lame(E, nu)
        _, F, _ = self.kinematics(x)
        vbar, Fd, _ = self.kinematics(v)                 # same projection applied to velocities: dF/dt
        c0, c1, c2 = F[:, :, 0], F[:, :, 1], F[:, :, 2]
        cof = torch.stack((torch.cross(c1, c2, dim=1), torch.cross(c2, c0, dim=1), torch.cross(c0, c1, dim=1)), 2)
        J = (c0*cof[:, :, 0]).sum(1)
        Finv = cof.transpose(1, 2)/J.clamp(min=1e-3)[:, None, None]
        L = Fd@Finv
        D = .5*(L+L.transpose(1, 2))
        PE = (2*mu*tau*self.volume)[:, None, None]*(D@cof)
        if self.stab == 'pyrlin':
            A = (beta*mu*tau*self.s_vol)[:, None, None]*(self.sub_F(v, vbar)-Fd[self.s_elem])
            PE = PE-torch.zeros(self.ne, 3, 3, dtype=x.dtype, device=x.device).index_add_(0, self.s_elem, A)
            return torch.zeros_like(x).index_add_(0, self.v, -_matvec(PE[self.e], self.g))+self.sub_forces(x, A)
        return torch.zeros_like(x).index_add_(0, self.v, -_matvec(PE[self.e], self.g))

    def _ground(self, x, v, dt):
        if self.ground and self.ground_mode == 'penalty':
            # smooth contact: normal acceleration omega_c^2 * penetration + damping (never pulling), and a
            # regularised Coulomb friction proportional to it -- the impact is spread over ~pi/omega_c
            pen = (-x[:, 1]).clamp(min=0)
            touching = pen > 0
            a_n = self.contact_omega**2*pen-2*self.contact_zeta*self.contact_omega*v[:, 1]*touching
            a_n = a_n.clamp(min=0)
            vy = v[:, 1]+dt*a_n
            vt = torch.stack((v[:, 0], v[:, 2]), 1)
            speed = (vt*vt).sum(1).add(1e-6).sqrt()
            keep = (1-self.friction*a_n*dt/speed).clamp(min=0)
            vt = vt*keep[:, None]
            return torch.stack((vt[:, 0], vy, vt[:, 1]), 1)
        if self.ground and self.ground_mode == 'layer':
            t = x[:, 1]/self.layer_dx
            W = torch.where(t < -.5, torch.ones_like(t),
                torch.where(t < .5, 1-.5*(t+.5)**2, torch.where(t < 1.5, .5*(1.5-t)**2, torch.zeros_like(t))))
            keep = torch.clamp(1-W, min=1e-6)**(dt/self.layer_dt_ref)
            v = v*keep[:, None]
        elif self.ground and self.ground_mode in ('coulomb', 'band'):
            y_next = x[:, 1]+dt*v[:, 1]
            hit = y_next <= 0
            vy = torch.where(hit, torch.maximum(v[:, 1], -x[:, 1]/dt), v[:, 1])
            vt = torch.stack((v[:, 0], v[:, 2]), 1)
            if self.ground_mode == 'coulomb':
                # impulse friction: tangential speed reduced by mu_f x the normal velocity change
                dvn = (vy-v[:, 1]).clamp(min=0)
                speed = vt.norm(dim=1)
                keep = torch.where(speed > 0, (1-self.friction*dvn/speed.clamp(min=1e-12)).clamp(min=0), torch.zeros_like(speed))
                vt = vt*keep[:, None]
            else:
                # no-slip over a contact band: vertices below `band` (or hitting) lose tangential velocity
                stick = hit | (x[:, 1] <= self.band)
                vt = torch.where(stick[:, None], torch.zeros_like(vt), vt)
            v = torch.stack((vt[:, 0], vy, vt[:, 1]), 1)
        elif self.ground:
            y_next = x[:, 1]+dt*v[:, 1]
            hit = (y_next <= 0)[:, None]
            v_hit = torch.stack((torch.zeros_like(x[:, 0]), -x[:, 1]/dt, torch.zeros_like(x[:, 0])), 1)
            v_hit = torch.stack((v_hit[:, 0], torch.maximum(v[:, 1], v_hit[:, 1]), v_hit[:, 2]), 1)
            v = torch.where(hit, v_hit, v)
        return v

    def step(self, x, v, params, dt):
        f = self.forces(x, params[:3])
        if len(params) > 3:
            f = f+self.damping_forces(x, v, params)
        if self.ncl:
            # clustered vertices share one displacement: sum forces / masses per cluster, contact on the lowest member
            fc = torch.zeros(self.ncl, 3, dtype=x.dtype, device=x.device).index_add_(0, self.cl, f)
            vc = v[self.rep]+dt*(fc/self.cmass[:, None]+self.gravity)
            vc = self._ground(x[self.low], vc, dt)
            v = vc[self.cl]
        else:
            v = v+dt*(f/self.mass[:, None]+self.gravity)
            v = self._ground(x, v, dt)
        return x+dt*v, v

    def probe(self, x, P, elem):
        xbar, F, _ = self.kinematics(x)
        Xbar = self.Xbar
        return xbar[elem]+_matvec(F[elem], P-Xbar[elem])

    def max_frequency(self, params, iterations=100, seed=0):
        """Largest omega of M^-1 K at the reference (power iteration, forward-mode K)."""
        gen = torch.Generator(device='cpu').manual_seed(seed)
        n = self.ncl or self.nv
        y = torch.randn(n, 3, generator=gen, dtype=torch.float64).to(self.device, self.dtype)
        sm = (self.cmass if self.ncl else self.mass).sqrt()[:, None]
        expand = (lambda u: u[self.cl]) if self.ncl else (lambda u: u)
        reduce = (lambda f: torch.zeros(n, 3, dtype=f.dtype, device=f.device).index_add_(0, self.cl, f)) if self.ncl else (lambda f: f)
        value = torch.tensor(0.)
        for _ in range(iterations):
            y = y/torch.linalg.vector_norm(y)
            _, Ky = torch.func.jvp(lambda zz: reduce(self.forces(self.X+expand(zz), params)), (torch.zeros_like(y),), (y/sm,))
            z = -Ky/sm
            value = (y*z).sum(); y = z
        return float(value.clamp(min=0).sqrt())

    def rollout(self, params, x, v, dt, substeps, frames, P=None, elem=None, compile=False, checkpoint=True, chunk=24):
        """Probe positions at every frame (frames+1, n, 3). With gradients, every `chunk` substeps are
        one checkpoint segment (only segment inputs are stored; a frame of a few hundred substeps of
        ~4M sub-cell rows would not fit in 12 GB)."""
        from torch.utils.checkpoint import checkpoint as ckpt
        if compile and getattr(self, '_compiled', None) is None:
            self._compiled = torch.compile(self.step, dynamic=False)
        step = self._compiled if compile else self.step
        dt = torch.as_tensor(dt, dtype=self.dtype, device=self.device)
        params = tuple(torch.as_tensor(p, dtype=self.dtype, device=self.device) for p in params)

        def advance(x, v, n, *params):
            for _ in range(n):
                x, v = step(x, v, params, dt)
            return x, v
        out = [self.probe(x, P, elem) if P is not None else x]
        grad = any(p.requires_grad for p in params)
        for _ in range(frames):
            done = 0
            while done < substeps:
                n = min(chunk, substeps-done)
                if checkpoint and grad:
                    x, v = ckpt(advance, x, v, n, *params, use_reentrant=False)
                else:
                    x, v = advance(x, v, n, *params)
                done += n
            out.append(self.probe(x, P, elem) if P is not None else x)
        return torch.stack(out), (x, v)


def checks(geometry, device, stages, out_path, stab=None):
    """Checks in float64; each result is printed and saved as soon as it is computed.
    static: forces vs autograd, rigid motion, momentum balance, patch test, omega_max.
    energy: conservation without gravity/ground for dt = c / omega.  fall: free fall.
    grad: gradient of a short rollout (frames 2 without, 7 with ground contact) vs finite differences."""
    dev = device
    m = VEMSolid(geometry, dev, torch.float64, stab=stab)
    E, nu = 23008.4, .30079
    s0 = {'pyramid': .4, 'pyrlin': 1.}.get(m.stab, 1.)
    params = tuple(torch.tensor(p, dtype=torch.float64, device=dev) for p in (E, nu, s0))
    gen = torch.Generator(device='cpu').manual_seed(1)
    rnd = lambda *s: torch.randn(*s, generator=gen, dtype=torch.float64).to(dev)
    out = json.loads(out_path.read_text()) if out_path.exists() else {}

    def put(k, val):
        out[k] = val
        print(json.dumps({k: val}), flush=True)
        out_path.write_text(json.dumps(out, indent=1))
    th = 1.3
    R = torch.tensor([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]], dtype=torch.float64, device=dev)
    A = torch.eye(3, dtype=torch.float64, device=dev)+.15*rnd(3, 3)
    t0 = time.perf_counter()
    if 'static' in stages:
        x = (m.X@A.T)@R.T+.002*rnd(m.nv, 3)+rnd(3)
        xg = x.clone().requires_grad_(True)
        fa = -torch.autograd.grad(m.energy(xg, params), xg)[0]
        fc = m.forces(x, params)
        put('force_vs_autograd_rel', float((fa-fc).norm()/fa.norm()))
        xc = m.X*torch.tensor([1., .03, 1.], dtype=torch.float64, device=dev)+.001*rnd(m.nv, 3)
        xg = xc.clone().requires_grad_(True)
        fa2 = -torch.autograd.grad(m.energy(xg, params), xg)[0]
        put('force_vs_autograd_rel_compressed', float((fa2-m.forces(xc, params)).norm()/fa2.norm()))
        xr = m.X@R.T+rnd(3)
        put('rigid_energy_over_deformed', float(m.energy(xr, params)/m.energy(x, params)))
        put('rigid_force_rel', float(m.forces(xr, params).norm()/fc.norm()))
        put('net_force_rel', float(fc.sum(0).norm()/fc.norm(dim=1).sum()))
        put('net_torque_rel', float(torch.cross(x, fc, dim=1).sum(0).norm()/(x.norm(dim=1)*fc.norm(dim=1)).sum()))
        _, F, r = m.kinematics(m.X@A.T+rnd(3))
        put('affine_F_err', float((F-A).abs().max())); put('affine_remainder', float(r.abs().max()))
        put('omega_max', m.max_frequency(params))
        put('static_seconds', time.perf_counter()-t0)
    omega = out.get('omega_max') or m.max_frequency(params)
    if 'energy' in stages:
        m.ground = False; m.gravity = torch.zeros(3, dtype=torch.float64, device=dev)
        S = torch.diag(torch.tensor([1.03, .98, 1.01], dtype=torch.float64, device=dev))
        c0 = (m.X*m.mass[:, None]).sum(0)/m.mass.sum()
        x0 = (m.X-c0)@S.T+c0
        w = torch.tensor([0., 3., 1.], dtype=torch.float64, device=dev)
        v0 = torch.cross(w.expand_as(x0), x0-c0, dim=1)
        T_total = .05
        for c in (1., .5, .25):
            dt = c/omega; n = int(np.ceil(T_total/dt)); dt = T_total/n
            xs, vs = x0.clone(), v0.clone()
            e0 = float(m.energy(xs, params)); total0 = e0+float(.5*(m.mass*(vs*vs).sum(1)).sum())
            worst = 0.
            for _ in range(n):
                xs, vs = m.step(xs, vs, params, dt)
                tot = float(m.energy(xs, params))+float(.5*(m.mass*(vs*vs).sum(1)).sum())
                worst = max(worst, abs(tot-total0))
            put(f'energy_err_over_strain_energy_dt{c:g}_over_omega', worst/e0)
        m.ground = True; m.gravity = torch.tensor([0., -9.8, 0.], dtype=torch.float64, device=dev)
    if 'fall' in stages:
        m.ground = False
        dt = .5/omega; n = int(.1/dt); xs, vs = m.X.clone(), torch.zeros_like(m.X)
        for _ in range(n):
            xs, vs = m.step(xs, vs, params, dt)
        cy = float(((xs-m.X)[:, 1]*m.mass).sum()/m.mass.sum()); t = n*dt
        put('free_fall_err_and_g_t_dt_over_2', [abs(cy+4.9*t*t), 4.9*t*dt])
        put('free_fall_max_strain_energy', float(m.energy(xs, params)))
        m.ground = True
    if 'grad' in stages:
        # No ground (smooth): a deforming motion (spin + stretching velocity) so that E, nu,
        # gamma all matter; then the same with ground contact, where finite differences are
        # only meaningful if they agree across step sizes (contact sets switch with parameters).
        P, elem = m.Xbar, torch.arange(m.ne, device=dev)
        c0 = (m.X*m.mass[:, None]).sum(0)/m.mass.sum()
        spin = torch.cross(torch.tensor([0., 4., 2.], dtype=torch.float64, device=dev).expand(m.nv, 3), m.X-c0, dim=1)
        stretch = (m.X-c0)*torch.tensor([3., -2., 1.], dtype=torch.float64, device=dev)
        bump = .3*torch.sin(25*(m.X-c0))
        v_init = torch.tensor([-.41, -.5, -.066], dtype=torch.float64, device=dev)+spin+stretch+bump
        substeps = int(np.ceil(omega/24))
        for ground, frames in ((False, 2), (False, 4), (True, 7)):
            m.ground = ground
            with torch.no_grad():
                target, _ = m.rollout((E*1.1, nu-.02, .8*s0), m.X, v_init, 1/24/substeps, substeps, frames, P=P, elem=elem)

            def loss_fn(z):
                p = (E*torch.exp(z[0]), nu+z[1], s0*torch.exp(z[2]))
                pos, _ = m.rollout(p, m.X, v_init, 1/24/substeps, substeps, frames, P=P, elem=elem)
                return ((pos-target)**2).mean()
            z = torch.zeros(3, dtype=torch.float64, device=dev, requires_grad=True)
            t1 = time.perf_counter()
            loss = loss_fn(z)
            ad = torch.autograd.grad(loss, z)[0].cpu().numpy()
            t_ad = time.perf_counter()-t1
            fds = {}
            for h in (1e-4, 1e-5, 1e-6):
                fd = []
                for i in range(3):
                    dz = torch.zeros(3, dtype=torch.float64, device=dev); dz[i] = h
                    with torch.no_grad():
                        fd.append(float((loss_fn(z+dz)-loss_fn(z-dz))/(2*h)))
                fds[f'h{h:g}'] = fd
            rel = {k: (np.abs(ad-np.array(v))/np.abs(np.array(v))).tolist() for k, v in fds.items()}
            put(f'grad_{"ground" if ground else "free"}_frames{frames}', dict(
                loss=float(loss), autograd=ad.tolist(), finite_diff=fds, rel_err=rel, substeps=substeps,
                forward_backward_seconds=t_ad))
        m.ground = True
    return out


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--geometry', type=Path, default=Path('output/pacnerf_vem/elastic_0_f0/vem_cap1_bodies.npz'))
    a.add_argument('--checks', nargs='*', help='stages: static energy fall grad')
    a.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a.add_argument('--stab', default=None, choices=(None, 'gamma', 'pyramid', 'pyrlin', 'pyrtan'))
    args = a.parse_args()
    if args.checks is not None:
        checks(args.geometry, args.device, args.checks or ['static', 'energy', 'fall', 'grad'],
               args.geometry.parent/f'checks_{args.geometry.stem}{"_"+args.stab if args.stab else ""}.json', args.stab)


if __name__ == '__main__':
    main()
