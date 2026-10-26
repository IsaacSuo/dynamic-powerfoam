"""The sphere model without skin degrees of freedom (2026-10-26; user: delete the skin, the surface from the foam's own
spheres). Degrees of freedom: the body positions only. Each surface cell's exposed sphere is the material's free surface:
  F_B      [S_B + F_s r N] [M_B + r N]^-1 per body: S_B, M_B the walls' bonds as in SkinTerms; r N = int (X - X_c) (x) n dA
           over the exposed sphere (X - X_c = r n on it), the moment the skin carried with lever 'c' (sum_q d_q (x) c_q),
           now with the cap's own deformation F_s instead of skin points:
             F_s = R_B (I + e_s),  e = sym(R_B^T F_B) - I,  e_s = P e P - lam / (lam + 2 mu) tr(P e P) m m^T
           (m the cap's mean normal at rest, P = I - m m^T: the tangential block kept, the normal column free, sigma . m = 0,
           plane stress about m). F_s depends on F_B: solved per body by Newton (9 unknowns), started from the polar
           factor of the fit with the cap turned along (SVD, no gradient); the derivative is the implicit one at the
           root. Exact for rigid motions and for uniform plane-stress states about each cap's m.
  energy   bonds (R_B = polar(F_B)), volume (walls + det F_B (V_res + r A / 3), no skin deviation), outer layer
           o1 = mu r tr(D^T D N), D = F_B - R_B; the skin's own terms (cross, deviation, lateral) and Gamma are gone
  skin     geometric, no degrees of freedom: a corner is where its three cells' deformed spheres meet,
           |F_k^-1 (y - x_k)| = r_k; an arc midpoint lies on its two cells' deformed spheres and in the plane through the
           arc's centre spanned by its direction from the centre and the circle's axis, both carried by the two cells'
           mean F (exact when they agree). Newton from the cells' mean affine image."""
import torch

from experiments.foam_surface_completion import polar_newton
from free_surface.boundary_skin import BoundaryTerms, corners_and_mids
from free_surface.gpu_geometry import Geometry, structure_triangulation
from free_surface.sphere_model import matrix_quat
from free_surface.surface_model import det3


def sym(M):
    return .5*(M+M.transpose(1, 2))


class ClosureTerms(BoundaryTerms):
    """the sphere model's walls, patches and volumes (BoundaryTerms, lever 'c'; its skin arrays only give each patch's
    int n dA), F_B closed on the exposed spheres, the energy without skin terms"""
    def __init__(self, model, geometry, p, r, skin, iters=12):
        super().__init__(model, geometry, p, r, skin, quad_ref=False, lever='c')
        X = self.X; n = self.n; dt, dev = X.dtype, X.device
        z33 = torch.zeros(n, 3, 3, dtype=dt, device=dev)
        half = (self.A*self.L/2)[:, None, None]*self.nrm[:, :, None]*self.nrm[:, None, :]
        self.Mb = z33.index_add(0, self.i, half).index_add(0, self.j, half)
        self.Mcap = z33.index_add(0, self.c_body, self.c_r[:, None, None]*self.c_N)
        self.Mt_inv = torch.linalg.inv(self.Mb+self.Mcap)
        self.Ms = z33.index_add(0, self.e_body, self.e_d[:, :, None]*self.e_l[:, None, :])     # the skin's moment (diagnostic: = Mcap)
        nint = torch.zeros(n, 3, dtype=dt, device=dev).index_add(0, self.e_body, self.e_c)    # int n dA over the body's caps
        self.cap = torch.zeros(n, dtype=torch.bool, device=dev); self.cap[self.c_body] = True
        self.m = torch.where(self.cap[:, None], nint/nint.norm(dim=1, keepdim=True).clamp(min=1e-300), torch.zeros_like(nint))
        self.mm = self.m[:, :, None]*self.m[:, None, :]
        self.P = torch.eye(3, dtype=dt, device=dev)-self.mm
        self.iters = iters
        self.kap0 = float(model.k_v)/(float(model.k_v)+2*self.mu)

    def walls(self, x):
        d = x[self.j]-x[self.i]
        t = (self.A/2)[:, None, None]*d[:, :, None]*self.nrm[:, None, :]
        return torch.zeros(self.n, 3, 3, dtype=x.dtype, device=x.device).index_add(0, self.i, t).index_add(0, self.j, t)

    def completion(self, F, kap):
        """the cap's deformation F_s (plane stress about m in the body's frame)"""
        R = polar_newton(F)
        e = sym(R.transpose(1, 2)@F)-torch.eye(3, dtype=F.dtype, device=F.device)
        et = self.P@e@self.P
        es = et-kap*et.diagonal(dim1=1, dim2=2).sum(1)[:, None, None]*self.mm
        return R@(torch.eye(3, dtype=F.dtype, device=F.device)+es)

    def residual(self, F, S, kap):
        return F@(self.Mb+self.Mcap)-S-self.completion(F, kap)@self.Mcap

    def closure(self, S, kap=None, iters=None):
        """F_B from the walls' S_B (n, 3, 3); returns F_B and the last Newton step (max abs). The plain fixed point
        contracts only by the cap's share of the moment along the rotations (up to 0.94 on the part: 1e-5 after 60
        iterations), so Newton with each body's 9 x 9 Jacobian (forward mode, exact). The returned F_B is one more
        Newton step with the Jacobian held and the graph on S and kap: at the root its value is the root and its first
        derivative the implicit one (second derivatives are not: Hessians by differences of gradients)."""
        kap = self.kap0 if kap is None else kap
        n = len(S)
        with torch.no_grad():
            Sd = S.detach(); kd = kap.detach() if torch.is_tensor(kap) else kap
            if getattr(self, 'F_warm', None) is not None and self.F_warm.shape == S.shape:
                F = self.F_warm                                                     # the last root (time stepping)
            else:
                U, _, Vh = torch.linalg.svd(Sd@self.Mt_inv)
                sg = torch.sign(torch.linalg.det(U@Vh))
                Rs = U@torch.diag_embed(torch.stack((torch.ones_like(sg), torch.ones_like(sg), sg), 1))@Vh
                F = (Sd+Rs@self.Mcap)@self.Mt_inv
            ch = float('nan')
            for _ in range(max(1, self.iters if iters is None else iters)):
                J = self.jacobian(F, Sd, kd)                                        # held for the last step: one Newton
                dF = torch.linalg.solve(J, -self.residual(F, Sd, kd).reshape(n, 9)).view(n, 3, 3)   # step off the root
                F = F+dF; ch = float(dF.abs().max())
                if ch < 1e-14:
                    break
            if getattr(self, 'warm', False):
                self.F_warm = F
        return F-torch.linalg.solve(J, self.residual(F, S, kap).reshape(n, 9)).view(n, 3, 3), ch

    def jacobian(self, F, S, kap):
        """d residual / d F per body (n, 9, 9), row-major flattening (the 9 directions batched)"""
        n = len(F)
        E9 = torch.eye(9, dtype=F.dtype, device=F.device).view(9, 1, 3, 3).expand(9, n, 3, 3)
        cols = torch.func.vmap(lambda T: torch.func.jvp(lambda F_: self.residual(F_, S, kap), (F,), (T,))[1])(E9)
        return cols.reshape(9, n, 9).permute(1, 2, 0)

    def F(self, x, y=None, kap=None):
        return self.closure(self.walls(x), kap)[0]

    def energy_terms(self, theta, x, y=None):
        lam = self.model.k_v*torch.exp(theta[0, 4])
        return self.energy_F(theta, x, self.F(x, kap=lam/(lam+2*self.mu)))

    def energy_F(self, theta, x, F):
        """the energy with F_B given (the closure's root, or any F for its derivatives)"""
        m = self.model
        lam = m.k_v*torch.exp(theta[0, 4]); sm = torch.exp(theta[:, 5]); mu = self.mu
        R = polar_newton(F); q = matrix_quat(R)
        bonds = m.energy(theta, x, q)
        if getattr(m, 'body_law', None) == 'nodalvol':
            bonds = bonds-m._nodalvol(theta, x)[0]
        D = F-R
        Dc = D[self.c_body]
        o1 = (mu*self.c_r*sm[self.c_body]*torch.einsum('cki,ckj,cij->c', Dc, Dc, self.c_N)).sum()
        Vd = self.wall_volume(x, F, R)+det3(F)*(self.Vres+torch.zeros_like(self.V).index_add(0, self.c_body, self.c_r*self.c_A/3))
        lj, _ = m._log_J(Vd/self.V)
        vol = (self.V*.5*lam*lj**2).sum()
        if self.variance:
            vol = vol+(.5*lam*self.A*self.L/12*(lj[self.i]-lj[self.j])**2).sum()
        return dict(bonds=bonds, volume=vol, outer_stretch=o1)

    def energy_x(self, theta, x):
        return sum(self.energy_terms(theta, x).values())


class SphereSkin:
    """the skin points (corners, then one midpoint per arc, as boundary_skin.build) as the meeting points of the cells'
    deformed spheres; one body per cell, the sites the bodies' rest positions"""
    def __init__(self, p, r, newton=30):
        S = structure_triangulation(p.cpu().numpy(), r.cpu().numpy(), device=p.device)
        G = Geometry(p, r, S)
        Y, gid, nc, _ = corners_and_mids(G, p)
        ap = G.arc_pair; i, j = G.i[ap], G.j[ap]
        tri = torch.zeros(nc, 3, dtype=torch.long, device=p.device)
        tri[gid[:, 0]] = torch.stack((i, j, G.arc_k0), 1); tri[gid[:, 2]] = torch.stack((i, j, G.arc_k1), 1)
        self.Y, self.nc, self.tri, self.pair = Y, nc, tri, torch.stack((i, j), 1)
        self.c, self.axis = G.c[ap], G.nn[ap]
        self.w = Y[nc:]-self.c                                     # the midpoint's direction from the arc's centre
        self.p, self.r, self.newton = p, r, newton

    def points(self, x, F):
        """skin positions for body positions x and deformation gradients F (n, 3, 3); and the largest residuals
        (sphere equations / r, midpoint planes / r)"""
        p, r = self.p, self.r
        Fi = torch.linalg.inv(F)
        Gm = Fi.transpose(1, 2)@Fi                                 # |F^-1 v|^2 = v^T G v
        img = lambda k, Z: x[k]+torch.einsum('...ij,...j->...i', F[k], Z-p[k])
        # corners
        k = self.tri; Yc = self.Y[:self.nc]
        yc = img(k, Yc[:, None, :]).mean(1)
        for _ in range(self.newton):
            v = yc[:, None, :]-x[k]
            Gv = torch.einsum('ckij,ckj->cki', Gm[k], v)
            phi = (v*Gv).sum(-1)-r[k]**2
            yc = yc-torch.linalg.solve(2*Gv, phi)
        res_c = ((yc[:, None, :]-x[k])*torch.einsum('ckij,ckj->cki', Gm[k], yc[:, None, :]-x[k])).sum(-1)-r[k]**2
        res_c = float((res_c.abs()/(2*r[k]**2)).max()) if len(k) else 0.
        # midpoints
        k = self.pair; Ym = self.Y[self.nc:]
        cc = img(k, self.c[:, None, :]).mean(1)
        Fb = .5*(F[k[:, 0]]+F[k[:, 1]])
        nrm = torch.cross(torch.einsum('aij,aj->ai', Fb, self.w), torch.einsum('aij,aj->ai', Fb, self.axis), dim=1)
        ym = img(k, Ym[:, None, :]).mean(1)
        for _ in range(self.newton):
            v = ym[:, None, :]-x[k]
            Gv = torch.einsum('akij,akj->aki', Gm[k], v)
            phi = torch.cat(((v*Gv).sum(-1)-r[k]**2, ((ym-cc)*nrm).sum(1, keepdim=True)), 1)
            Jm = torch.cat((2*Gv, nrm[:, None, :]), 1)
            ym = ym-torch.linalg.solve(Jm, phi)
        v = ym[:, None, :]-x[k]
        sph = ((v*torch.einsum('akij,akj->aki', Gm[k], v)).sum(-1)-r[k]**2).abs()/(2*r[k]**2)
        pln = ((ym-cc)*nrm).sum(1).abs()/(nrm.norm(dim=1)*r[k].min(1).values)
        return torch.cat((yc, ym)), res_c, float(sph.max()), float(pln.max())
