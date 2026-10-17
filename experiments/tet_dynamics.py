"""Neo-Hookean dynamics on the linear tetrahedral sub-mesh of the Power Foam cells (tet_geometry.py).

Energy sum_t |t| psi_NH(F_t), F_t exact P1 gradient of tet t from its 4 points (vertex, vertex,
face centre, cell apex; centres averaged from vertices every step). No stabilisation, parameters
(E, nu) only (a third entry is ignored). Vertex binding (cluster_tol) as in VEMSolid. Ground,
time stepping, rollout, max_frequency, damping hooks: VEMSolid. Probe: DEM bodies as the
volume-weighted mean of their cells' apexes.
"""
import numpy as np
import torch

from experiments.vem_dynamics import RHO, VEMSolid, lame


class TetSolid(VEMSolid):
    def __init__(self, path, device='cuda', dtype=torch.float32, barrier=.05, cluster_tol=5e-4, cells=None, law='nh',
                 mass_file=None):
        d = dict(np.load(path))
        self.device, self.dtype, self.barrier, self.law = device, dtype, barrier, law
        self.stab, self.points = 'tet', True
        T = lambda a, dt=dtype: torch.as_tensor(np.asarray(a), dtype=dt, device=device)
        self.X = T(d['X']); self.nv = len(self.X)
        self.p_id, self.p_pt, self.p_g = T(d['sp_id'], torch.long), T(d['sp_pt'], torch.long), T(d['sp_g'])
        self.s_vol = T(d['tet_volume']); self.ns = len(self.s_vol)
        self.tet_cell = T(d['tet_cell'], torch.long); self.cell_vol = T(d['cell_volume'])
        # dense layout: every tet has exactly 4 points (apex, face centre, a, b) -> no scatter for F,
        # one stress read per tet for the forces
        self.tet_pts = self.p_pt.view(-1, 4); self.tet_g = self.p_g.view(-1, 4, 3)
        self.dense = False                                  # compiled sparse path is as fast (0.79 vs 0.85 ms/step)
        self.nf, self.nc = int(d['n_face']), int(d['n_cell'])
        self.fc_face, self.fc_vert, self.fc_w = T(d['fc_face'], torch.long), T(d['fc_vert'], torch.long), T(d['fc_w'])
        self.ap_cell, self.ap_vert, self.ap_w = T(d['ap_cell'], torch.long), T(d['ap_vert'], torch.long), T(d['ap_w'])
        vm = d['vertex_volume'].copy()
        vm[vm <= 0] = vm[vm > 0].min()                       # vertices outside any tet (none expected)
        self.mass = T(RHO*vm)
        self.gravity = T([0., -9.8, 0.]); self.ground = True
        self.ground_mode, self.layer_dx, self.layer_dt_ref, self.friction, self.band = 'noslip', .02, 1/4800, .5, .01
        self.contact_omega, self.contact_zeta = 2000., .3
        self.ncl = 0
        if cluster_tol > 0:
            self.set_clusters(cluster_tol)
        self.added_mass_fraction = 0.
        if mass_file is not None:                            # selective mass scaling (tet_mass_scaling.py)
            ms = np.load(mass_file)
            assert self.ncl and np.array_equal(ms['cl'], self.cl.cpu().numpy()), 'mass file built for other clusters'
            c0 = float(self.cmass.sum())
            self.cmass = T(ms['cmass'])
            self.added_mass_fraction = float(self.cmass.sum())/c0-1
        if cells is not None:
            body_of_cell, cell_vol = cells
            Wb = np.zeros((int(body_of_cell.max()+1), self.nc))
            np.add.at(Wb, (body_of_cell, np.arange(self.nc)), cell_vol)
            Wb /= Wb.sum(1, keepdims=True)
            self.Wb = T(Wb).to_sparse()

    def tet_F(self, x):
        if self.dense:
            P = self._points(x)[self.tet_pts]                                  # (T, 4, 3)
            rel = P-P[:, :1]                                                   # centre on the apex (float32)
            return (rel[:, :, :, None]*self.tet_g[:, :, None, :]).sum(1)      # (T, 3, 3) = sum_p rel_p (x) g_p (elementwise: no tiny batched GEMM)
        pts = self._points(x)
        rel = pts[self.p_pt]-pts[self.p_pt[::4]].repeat_interleave(4, 0)     # centre on each tet's apex
        return torch.zeros(self.ns, 3, 3, dtype=x.dtype, device=x.device).index_add_(0, self.p_id, rel[:, :, None]*self.p_g[:, None, :])

    def energy(self, x, params):
        lam, mu = lame(params[0], params[1])
        psi, _ = self._psi_P(self.tet_F(x), lam, mu, stress=False, law=self.law)
        return (self.s_vol*psi).sum()

    def forces(self, x, params):
        lam, mu = lame(params[0], params[1])
        _, P = self._psi_P(self.tet_F(x), lam, mu, law=self.law)
        A = self.s_vol[:, None, None]*P
        if not self.dense:
            return self.sub_forces(x, A)
        n = len(x)
        fpt = -(A[:, None, :, :]*self.tet_g[:, :, None, :]).sum(-1)            # (T, 4, 3): -A g_p per tet point
        fp = torch.zeros(n+self.nf+self.nc, 3, dtype=x.dtype, device=x.device).index_add_(0, self.tet_pts.reshape(-1), fpt.reshape(-1, 3))
        f = fp[:n].clone()
        f.index_add_(0, self.fc_vert, fp[n+self.fc_face]*self.fc_w[:, None])
        f.index_add_(0, self.ap_vert, fp[n+self.nf+self.ap_cell]*self.ap_w[:, None])
        return f

    def damping_forces(self, x, v, params):
        """Artificial bulk viscosity (von Neumann-Richtmyer / LS-DYNA default form), the standard shock
        dissipation of explicit codes: for a tet in compression (volumetric rate d = tr(dF/dt F^-1) < 0),
        pressure q = rho l (Q1^2 l d^2 - Q2 c d), l = (J |t|)^(1/3), c the P-wave speed, Q1 = 1.5,
        Q2 = 0.06, both scaled by params[3]; Cauchy stress -q I, P = -q cof. Zero in expansion and for
        rigid motion; the quadratic term only matters at steep compression fronts."""
        E, nu, _, s = params
        lam, mu = lame(E, nu)
        c = torch.sqrt((lam+2*mu)/RHO)
        F, Fd = self.tet_F(x), self.tet_F(v)
        c0, c1, c2 = F[:, :, 0], F[:, :, 1], F[:, :, 2]
        cof = torch.stack((torch.cross(c1, c2, dim=1), torch.cross(c2, c0, dim=1), torch.cross(c0, c1, dim=1)), 2)
        J = (c0*cof[:, :, 0]).sum(1)
        Jc = torch.where(J.abs() < 1e-3, torch.full_like(J, 1e-3), J)
        d = (Fd*cof).sum((1, 2))/Jc                                   # tr(dF/dt F^-1) = dJ/dt / J
        ell = (J.abs()*self.s_vol).clamp(min=1e-30)**(1/3)
        q = torch.where(d < 0, RHO*ell*s*(2.25*ell*d*d-.06*c*d), torch.zeros_like(d))
        return self.sub_forces(x, -(self.s_vol*q)[:, None, None]*cof)

    def probe(self, x, P=None, elem=None):
        apex = self._points(x)[self.nv+self.nf:]
        return torch.sparse.mm(self.Wb, apex) if hasattr(self, 'Wb') else apex

    def tet_J(self, x):
        return torch.linalg.det(self.tet_F(x))

    def cell_F(self, x):
        """Volume-weighted mean tet F per cell (for rendering / diagnostics)."""
        F = self.tet_F(x)*self.s_vol[:, None, None]
        if not hasattr(self, 'tet_cell'):
            raise AttributeError('tet_cell not loaded')
        out = torch.zeros(self.nc, 3, 3, dtype=x.dtype, device=x.device).index_add_(0, self.tet_cell, F)
        return out/self.cell_vol[:, None, None]
