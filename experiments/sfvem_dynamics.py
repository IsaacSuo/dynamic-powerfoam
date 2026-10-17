"""Stabilization-free VEM dynamics on Power Foam cells (elements from sfvem_geometry.py).

Per element: C_k = sum_v (x_v - xbar) (x) R_k[:, v] (polynomial coefficients of the projected
deformation gradient), F(X_q) = sum_k phi_k(X_q) C_k at the quadrature points, energy
sum_q w_q psi_NH(F_q) (PAC-NeRF's Neo-Hookean, log J continued below J = 0.05). No stabilisation
term: the degree of each element is high enough that every non-rigid vertex motion strains it.
Elements are bucketed by degree and padded, so the projection, the point evaluation and their
transposes are batched matrix products; vertex forces are one scatter-add.
Mass: lumped from the cone volumes (sfvem_geometry). Time stepping, ground, rollout: VEMSolid.
params = (E, nu, unused) for compatibility.
"""
import numpy as np
import torch

from experiments.vem_dynamics import RHO, VEMSolid, lame


class SFVEMSolid(VEMSolid):
    def __init__(self, path, device='cuda', dtype=torch.float32, barrier=.05, cells=None, law='nh'):
        d = dict(np.load(path))
        self.law = law
        self.device, self.dtype, self.barrier = device, dtype, barrier
        self.stab = 'sfvem'; self.ncl = 0
        T = lambda a, dt=dtype: torch.as_tensor(np.asarray(a), dtype=dt, device=device)
        used = np.unique(d['verts'])
        gmap = -np.ones(len(d['X']), int); gmap[used] = np.arange(len(used))
        self.X = T(d['X'][used]); self.nv = len(used)
        deg, nb, off_v, off_R, off_q, off_phi = d['degree'], d['nb'], d['off_v'], d['off_R'], d['off_q'], d['off_phi']
        nv_e = np.diff(off_v); nq_e = np.diff(off_q)
        self.ne = len(deg)
        mass = np.zeros(self.nv)
        np.add.at(mass, gmap[d['verts']], RHO*d['mass_vol'])
        self.mass = T(mass)
        self.buckets = []
        for l in np.unique(deg):
            es = np.flatnonzero(deg == l); k = int(nb[es[0]])
            Np, Qp = int(nv_e[es].max()), int(nq_e[es].max())
            vid = np.zeros((len(es), Np), int); vm = np.zeros((len(es), Np))
            R = np.zeros((len(es), k, 3, Np)); Ph = np.zeros((len(es), Qp, k)); W = np.zeros((len(es), Qp))
            for r, e in enumerate(es):
                n, q = nv_e[e], nq_e[e]
                vid[r, :n] = gmap[d['verts'][off_v[e]:off_v[e+1]]]; vm[r, :n] = 1
                R[r, :, :, :n] = d['R'][off_R[e]:off_R[e+1]].reshape(k, 3, n)
                Ph[r, :q] = d['phi_q'][off_phi[e]:off_phi[e+1]].reshape(q, k)
                W[r, :q] = np.maximum(d['qw'][off_q[e]:off_q[e+1]], 0.)     # 264 tiny negative cone weights (non-convex merges) dropped
            self.buckets.append(dict(degree=int(l), elems=T(es, torch.long), vid=T(vid, torch.long), vm=T(vm),
                                     n=T(nv_e[es]), R=T(R.reshape(len(es), k*3, Np)), Ph=T(Ph), W=T(W), k=k, Np=Np, Qp=Qp))
        self.gravity = T([0., -9.8, 0.]); self.ground = True
        self.ground_mode, self.layer_dx, self.layer_dt_ref, self.friction, self.band = 'noslip', .02, 1/4800, .5, .01
        self.contact_omega, self.contact_zeta = 2000., .3
        self.volume = T(d['volume'])
        # probe: element vertex means; bodies (DEM) = volume-weighted mean over their cells' elements
        self.group = d['group']
        if cells is not None:
            body_of_cell, cell_vol = cells
            nbody = int(body_of_cell.max()+1)
            Wb = np.zeros((nbody, self.ne))
            np.add.at(Wb, (body_of_cell, self.group), cell_vol)
            Wb /= Wb.sum(1, keepdims=True)
            self.Wb = T(Wb).to_sparse()

    def _bucket_F(self, x, b):
        xe = x[b['vid']]*b['vm'][..., None]
        xbar = xe.sum(1)/b['n'][:, None]
        rel = (xe-xbar[:, None])*b['vm'][..., None]
        C = torch.bmm(b['R'], rel)                                    # (E, k*3, 3): [k, j] rows, i cols
        C = C.view(-1, b['k'], 3, 3).transpose(2, 3).reshape(-1, b['k'], 9)   # [k, i, j]
        F = torch.bmm(b['Ph'], C).view(-1, b['Qp'], 3, 3)
        return F, xbar

    def energy(self, x, params):
        E, nu = params[0], params[1]
        lam, mu = lame(E, nu)
        tot = 0.
        for b in self.buckets:
            F, _ = self._bucket_F(x, b)
            psi, _ = self._psi_P(F.reshape(-1, 3, 3), lam, mu, stress=False, law=self.law)
            tot = tot+(b['W'].reshape(-1)*psi).sum()
        return tot

    def _assemble(self, f, b, P):
        """f -= sum_q w_q P_q : dF_q/dx (P per point, (E*Qp, 3, 3))."""
        Pw = (P.view(-1, b['Qp'], 9)*b['W'][..., None])
        dC = torch.bmm(b['Ph'].transpose(1, 2), Pw).view(-1, b['k'], 3, 3)      # [k, i, j]
        dC = dC.permute(0, 2, 1, 3).reshape(-1, 3, b['k']*3)                      # (E, i, k*3 [k, j])
        g = torch.bmm(dC, b['R']).transpose(1, 2)*b['vm'][..., None]              # (E, Np, 3)
        f.index_add_(0, b['vid'].reshape(-1), -g.reshape(-1, 3))

    def forces(self, x, params):
        E, nu = params[0], params[1]
        lam, mu = lame(E, nu)
        f = torch.zeros_like(x)
        for b in self.buckets:
            F, _ = self._bucket_F(x, b)
            _, P = self._psi_P(F.reshape(-1, 3, 3), lam, mu, law=self.law)
            self._assemble(f, b, P)
        return f

    def damping_forces(self, x, v, params):
        """Kelvin-Voigt viscosity, time constant tau = params[3]: sigma = 2 mu tau sym(dF/dt F^-1), P = sigma cof."""
        E, nu, _, tau = params
        _, mu = lame(E, nu)
        f = torch.zeros_like(x)
        for b in self.buckets:
            F, _ = self._bucket_F(x, b)
            Fd, _ = self._bucket_F(v, b)
            F, Fd = F.reshape(-1, 3, 3), Fd.reshape(-1, 3, 3)
            c0, c1, c2 = F[:, :, 0], F[:, :, 1], F[:, :, 2]
            cof = torch.stack((torch.cross(c1, c2, dim=1), torch.cross(c2, c0, dim=1), torch.cross(c0, c1, dim=1)), 2)
            J = (c0*cof[:, :, 0]).sum(1)
            Finv = cof.transpose(1, 2)/torch.where(J.abs() < 1e-3, torch.full_like(J, 1e-3), J)[:, None, None]
            L = Fd@Finv
            D = .5*(L+L.transpose(1, 2))
            self._assemble(f, b, (2*mu*tau)*(D@cof))
        return f

    def element_centres(self, x):
        c = torch.zeros(self.ne, 3, dtype=x.dtype, device=x.device)
        for b in self.buckets:
            xe = x[b['vid']]*b['vm'][..., None]
            c[b['elems']] = xe.sum(1)/b['n'][:, None]
        return c

    def probe(self, x, P=None, elem=None):
        c = self.element_centres(x)
        return torch.sparse.mm(self.Wb, c) if hasattr(self, 'Wb') else c

    def min_J(self, x):
        out = []
        for b in self.buckets:
            F, _ = self._bucket_F(x, b)
            J = torch.linalg.det(F.reshape(-1, 3, 3)).view(-1, b['Qp'])
            out.append(J[b['W'] != 0])
        return torch.cat(out)
