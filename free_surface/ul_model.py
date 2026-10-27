"""Updated-Lagrangian Power cells (UPDATED_LAGRANGE.md, 2026-10-27), torch reference. State per cell: site x_i, velocity v_i,
deformation gradient F_i (history, against rest), radius r_i (the weight w = r^2). A step from n to n+1 (backward Euler):
  geometry  the power cells cut by their balls at (x^n, r^n) (recut.Recut, GPU): walls (i, j) with area A, centroid c,
            normal n = (x_j - x_i) / l, position along the line h = (c - x_i).n; volumes; exposed caps
  gradient  L: G_s = sum_walls (dx_o - dx_s) (x) w d^T M_s^-1 (w = A / l, M_s = sum w d d^T), carried to the cell's centroid
            with the least-squares gradient of the gradients: Gc_s = G_s + sum w (d^T M_s^-1 delta_s)(G_o - G_s), delta_s =
            F_s^n (c0_s - X_s) (the rest centroid offset carried affinely); exact for affine increments
  energy    Pi(x) = sum m / (2 dt^2) |x - x^n - dt v^n|^2 - sum m g.x + sum V0 psi(F) + E_stab + E_jump (+ contact), with
            F_s(x) = (I + Gc_s(x - x^n)) F_s^n and neo-Hookean psi = mu / 2 (tr C - 3) - mu ln J + lam / 2 (ln J)^2;
            E_stab (S-NSNI, Wang & Hillman 2025, on the Green-Lagrange strain E = (C - I) / 2):
            1/2 sum_s sum_kl Jm_s^kl dE_k : D : dE_l, dE_k = (1 / V_s) sum_walls A (h / l)(E_o - E_s) n_k, Jm_s = det F F J0 F^T
            the cell's second moment about its centroid carried affinely;
            E_jump: 1/2 sum_walls (A / 3 l)(mu |delta|^2 + (lam + mu)(delta.n)^2), delta = (x_j - x_i) - (F_i + F_j)(X_j - X_i) / 2
  solve     trust-region Newton-CG on Pi (autograd gradient, Hessian-vector products by double backward)
  update    v = (x - x^n) / dt, F = F(x); radii solved so that every cell's volume is V0 det F (recut.solve_radii)
Units as the DEM model's (moduli per unit density, E / 1000 for density 1000; masses = volumes)."""
import math

import numpy as np
import torch

from free_surface.recut import Recut

f64 = torch.float64


def cell_moments_grid(X, r, V, spacing, dev='cuda'):
    """per cell: centroid offset from the site and second moment about the centroid (grid samples assigned by power
    distance inside the balls, scaled to the exact volume V); cells without a sample get zeros"""
    pt, rt = torch.as_tensor(X, device=dev), torch.as_tensor(r, device=dev)
    lo, hi = (X-r[:, None]).min(0), (X+r[:, None]).max(0)
    n = len(X); cnt = np.zeros(n); s1 = np.zeros((n, 3)); s2 = np.zeros((n, 3, 3))
    ax = [np.arange(lo[k]+spacing/2, hi[k], spacing) for k in range(3)]
    for x0 in ax[0]:
        Gg = np.stack(np.meshgrid([x0], ax[1], ax[2], indexing='ij'), -1).reshape(-1, 3)
        for c0 in range(0, len(Gg), 8192):
            y = torch.as_tensor(Gg[c0:c0+8192], device=dev)
            v_, a_ = (torch.cdist(y, pt)**2-rt[None]**2).min(1)
            ok = (v_ <= 0).cpu().numpy()
            if ok.any():
                a_ = a_.cpu().numpy()[ok]; R_ = Gg[c0:c0+8192][ok]-X[a_]
                np.add.at(cnt, a_, 1.); np.add.at(s1, a_, R_); np.add.at(s2, a_, R_[:, :, None]*R_[:, None, :])
    ok = cnt > 0
    dlt = np.where(ok[:, None], s1/np.maximum(cnt, 1)[:, None], 0.)
    J = np.where(ok[:, None, None], (s2/np.maximum(cnt, 1)[:, None, None]-dlt[:, :, None]*dlt[:, None, :])*V[:, None, None], 0.)
    return dlt, J, cnt


class ULModel:
    def __init__(self, X0, r0, V0, dlt0, J0, E, nu, rho=1., dev='cuda', jump=True, stab=True):
        t = lambda a: torch.as_tensor(np.asarray(a), dtype=f64, device=dev)
        self.dev, self.n = dev, len(X0)
        self.X0, self.r0, self.V0, self.dlt0, self.J0 = t(X0), t(r0), t(V0), t(dlt0), t(J0)
        self.mu = E/(2*(1+nu)); self.lam = E*nu/((1+nu)*(1-2*nu))
        self.m = rho*self.V0
        self.jump, self.stab = jump, stab
        self.recut = Recut(self.n, dev)
        self.I3 = torch.eye(3, dtype=f64, device=dev)

    # ---- geometry and the step's operators
    def geometry(self, x, r):
        G, S = self.recut.geometry(x, r)
        keep = G.wall_A.detach() > 0
        P = G.P[keep]
        return dict(i=P[:, 0], j=P[:, 1], A=G.wall_A.detach()[keep], c=G.wall_c.detach()[keep], V=G.V.detach(),
                    capA=G.patch_A.detach(), hidden=S['hidden'])

    def operators(self, x, F, geo):
        """the step's linear maps, built at the step start (x^n, F^n, geometry)"""
        i, j, A, c = geo['i'], geo['j'], geo['A'], geo['c']
        d = x[j]-x[i]; L = d.norm(dim=1); nrm = d/L[:, None]
        h = ((c-x[i])*nrm).sum(1); hj = L-h; w = A/L
        n = self.n
        M = torch.zeros(n, 3, 3, dtype=f64, device=self.dev)
        dd = w[:, None, None]*d[:, :, None]*d[:, None, :]
        M.index_add_(0, i, dd); M.index_add_(0, j, dd)
        Mi = torch.linalg.inv(M)
        delta = torch.einsum('nab,nb->na', F, self.dlt0)
        detF = torch.linalg.det(F)
        Jm = detF[:, None, None]*F@self.J0@F.transpose(1, 2)
        ends = []
        for s_, o_, dv, nn, hh in ((i, j, d, nrm, h), (j, i, -d, -nrm, hj)):
            row = torch.einsum('fa,fab->fb', w[:, None]*dv, Mi[s_])              # w d^T M_s^-1
            cen = (row*delta[s_]).sum(1)                                          # w d^T M_s^-1 delta_s
            smo = (A*hh/L)[:, None]*nn/geo['V'][s_][:, None]                       # A (h / l) n / V_s
            ends.append((s_, o_, row, cen, smo))
        Kj = (A/(3*L))
        return dict(i=i, j=j, ends=ends, Jm=Jm, Kj=Kj, x0=x.clone(), F0=F.clone(), L=L, nrm=nrm, A=A)

    # ---- the energy
    def def_grad(self, x, ops):
        dx = x-ops['x0']
        G = torch.zeros(self.n, 3, 3, dtype=f64, device=self.dev)
        for s_, o_, row, cen, smo in ops['ends']:
            G = G.index_add(0, s_, (dx[o_]-dx[s_])[:, :, None]*row[:, None, :])
        Gc = G
        for s_, o_, row, cen, smo in ops['ends']:
            Gc = Gc.index_add(0, s_, cen[:, None, None]*(G[o_]-G[s_]))
        return (self.I3+Gc)@ops['F0']

    def elastic(self, x, ops, terms=False):
        F = self.def_grad(x, ops)
        J = torch.linalg.det(F)
        C = F.transpose(1, 2)@F
        lnJ = torch.log(J.clamp(min=1e-300))
        psi = .5*self.mu*(C.diagonal(dim1=1, dim2=2).sum(1)-3)-self.mu*lnJ+.5*self.lam*lnJ**2
        e = dict(cells=(self.V0*psi).sum())
        if bool((J <= 0).any()):
            e['cells'] = e['cells']*0+math.inf
        if self.stab:
            Eg = .5*(C-self.I3)
            dE = torch.zeros(self.n, 3, 3, 3, dtype=f64, device=self.dev)        # [s, k, a, b]
            for s_, o_, row, cen, smo in ops['ends']:
                dE = dE.index_add(0, s_, smo[:, :, None, None]*(Eg[o_]-Eg[s_])[:, None])
            tr = dE.diagonal(dim1=2, dim2=3).sum(2)                               # [s, k]
            e['stab'] = .5*(torch.einsum('skl,sk,sl->', ops['Jm'], tr, tr)*self.lam+2*self.mu*torch.einsum('skl,skab,slab->', ops['Jm'], dE, dE))
        if self.jump:
            i, j = ops['i'], ops['j']
            dlt = (x[j]-x[i])-.5*torch.einsum('fab,fb->fa', F[i]+F[j], self.X0[j]-self.X0[i])
            nc = (x[j]-x[i])/(x[j]-x[i]).norm(dim=1, keepdim=True)
            e['jump'] = .5*(ops['Kj']*(self.mu*(dlt*dlt).sum(1)+(self.lam+self.mu)*(dlt*nc).sum(1)**2)).sum()
        return (e, F) if terms else sum(e.values())
