"""The sphere model without skin degrees of freedom (2026-10-26; user: delete the skin, the surface from the foam's own
spheres). Degrees of freedom: the body positions only. Each surface cell's exposed sphere is the material's free surface:
  F_B      [S_B + F_s r N] [M_B + r N]^-1 per body: S_B, M_B the walls' bonds as in SkinTerms; r N = int (X - X_c) (x) n dA
           over the exposed sphere (X - X_c = r n on it), the moment the skin carried with lever 'c' (sum_q d_q (x) c_q),
           now with the cap's deformation F_s instead of skin points. Scheme A (10-27, user: "推清楚" / "做吧"): the cap's
           tangential part from its rim, carried by the walls the rim lies on (each wall's map the mean F of its two
           cells), the normal column by the free surface:
             Fbar_B = 1/2 F_B + 1/2 sum_e w_e F_k(e)   (w_e the rim's arc length on wall e, normalised; k(e) the other cell)
             F_s = Fbar P + s n' m^T,  c_a = Fbar t_a,  n' = c1 x c2 / |c1 x c2|,
             s = 1 - lam / (lam + 2 mu) (sqrt(|c1|^2 + |c2|^2 + 2 |c1 x c2|) - 2)
           (m the cap's mean normal at rest, t_a its tangents, P = I - m m^T; = R_s [U_t (+) (1 - kappa tr(U_t - I))] of
           F~ = Fbar t1 t1^T + Fbar t2 t2^T + n' m^T, sigma . m = 0). The cap fixes only the normal shear and the plane-stress
           normal; the former F_s = R_B (I + e_s(sym(R_B^T F_B) - I)) took the rotation and the tangential strain from F_B
           itself, so those 6 directions rested on the walls alone (lion: 34 cells of closure cond 1e3-3e17, now 3-98).
           The surface cells are coupled through Fbar: solved by block-Jacobi Newton sweeps (each body's 9 x 9 block, the
           neighbours lagged; rho 0.79 on the lion, 0.39 on the jack), the derivative the implicit one of the coupled system
           (Jacobi on J and J^T). Exact for rest, rigid motions and uniform plane-stress states about each cap's m.
  energy   bonds (R_B = polar(F_B)), volume (walls + det F_B (V_res + r A / 3), no skin deviation), outer layer
           o1 = mu r tr(D^T D N), D = F_B - R_B; the skin's own terms (cross, deviation, lateral) and Gamma are gone
  skin     geometric, no degrees of freedom: a corner is where its three cells' deformed spheres meet,
           |F_k^-1 (y - x_k)| = r_k; an arc midpoint lies on its two cells' deformed spheres and in the plane through the
           arc's centre spanned by its direction from the centre and the circle's axis, both carried by the two cells'
           mean F (exact when they agree). Newton from the cells' mean affine image."""
import numpy as np
import torch

from experiments.foam_surface_completion import polar_newton
from free_surface.boundary_skin import corners_and_mids, split_rings
from free_surface.gpu_geometry import Geometry, structure_triangulation
from free_surface.sphere_model import matrix_quat
from free_surface.surface_model import cof3, det3


def sym(M):
    return .5*(M+M.transpose(1, 2))


def rot_proper(F, steps=12):
    """the nearest proper rotation of F (n, 3, 3), closure_warp.rot_proper: scaled Newton (Higham), and for det F < 0 the
    orthogonal factor flipped along the least eigenvector of its symmetric factor (analytic, as least_eigvec)"""
    X = F
    fro = lambda M: (M*M).sum((1, 2))
    for _ in range(steps):
        Xi = torch.linalg.inv(X)
        g = (fro(Xi)/fro(X))**.25
        X = .5*(g[:, None, None]*X+Xi.transpose(1, 2)/g[:, None, None])
    neg = torch.linalg.det(F) < 0
    if bool(neg.any()):
        Xn, Fn = X[neg], F[neg]
        H = sym(Xn.transpose(1, 2)@Fn)
        q = H.diagonal(dim1=1, dim2=2).sum(1)/3
        I3 = torch.eye(3, dtype=F.dtype, device=F.device)
        p = torch.sqrt(((H-q[:, None, None]*I3)**2).sum((1, 2))/6)
        r = (torch.linalg.det((H-q[:, None, None]*I3)/p[:, None, None])/2).clamp(-1, 1)
        lam = q+2*p*torch.cos(torch.acos(r)/3+2.0943951023931957)

        def null_row_cross(M):
            cs = torch.stack((torch.linalg.cross(M[:, 0], M[:, 1]), torch.linalg.cross(M[:, 0], M[:, 2]), torch.linalg.cross(M[:, 1], M[:, 2])), 1)
            k = (cs*cs).sum(2).argmax(1)
            v = cs[torch.arange(len(k), device=F.device), k]
            return v/v.norm(dim=1, keepdim=True)
        v = null_row_cross(H-lam[:, None, None]*I3)
        v = null_row_cross(H-torch.einsum('bi,bij,bj->b', v, H, v)[:, None, None]*I3)   # (one Rayleigh refinement)
        X = X.clone()
        X[neg] = Xn@(torch.eye(3, dtype=F.dtype, device=F.device)-2*v[:, :, None]*v[:, None, :])
    return X


class _CoupledSolve(torch.autograd.Function):
    """x = J^-1 b with the coupled closure's Jacobian J held at F (backward: J^-T)"""
    @staticmethod
    def forward(ctx, b, st, F, kap):
        ctx.st, ctx.F, ctx.kap = st, F, kap
        with torch.no_grad():
            return st.coupled_solve(F, kap, b)

    @staticmethod
    def backward(ctx, g):
        with torch.no_grad():
            return ctx.st.coupled_solve(ctx.F, ctx.kap, g, transpose=True), None, None, None


class ClosureTerms:
    """the skinless model's terms, standalone (2026-10-26; user: the geometric skin is used by nothing; the skin model's
    set-up it inherited before held 4-15 GB in a batched pseudo-inverse and failed on the 30k jack): the walls as SkinTerms
    has them (the DEM geometry, one body per cell), each surface cell's exposed sphere from the exact geometry
    (gpu_geometry.Geometry: r N = r int n n^T dA, its area, the mean normal m from int n dA; the cells with exposed area),
    the volumes; F_B closed on the exposed spheres, the energy without skin terms. skin: unused (the callers' signature)"""
    def __init__(self, model, geometry, p, r, skin=None, iters=12, variance=True):
        dt, dev = model.X.dtype, model.X.device
        tt = lambda a: torch.as_tensor(np.asarray(a), dtype=dt, device=dev)
        assert np.array_equal(np.asarray(geometry['body_of_cell']), np.arange(len(p))), 'one body per cell (foam_exact\'s dem_exact_single)'
        self.model, self.variance = model, variance
        X = model.X; self.X = X; n = len(X); self.n = n
        self.i, self.j = model.i, model.j
        self.A = tt(geometry['areas']); c = tt(geometry['contacts'])
        d = X[self.j]-X[self.i]; self.L = d.norm(dim=1); self.nrm = d/self.L[:, None]
        self.hi = ((c-X[self.i])*self.nrm).sum(1); self.hj = self.L-self.hi
        self.ri, self.rj = c-X[self.i], c-X[self.j]
        self.t = c-(X[self.i]+self.hi[:, None]*self.nrm)
        self.wi, self.wj = (self.hi > 0).to(dt), (self.hj > 0).to(dt)
        self.V = tt(geometry['masses'])
        z1 = torch.zeros(n, dtype=dt, device=dev)
        front = z1.index_add(0, self.i, self.wi*self.A*self.hi/3).index_add(0, self.j, self.wj*self.A*self.hj/3)
        pt, rt = tt(p), tt(r)
        G = Geometry(pt, rt, structure_triangulation(np.asarray(p), np.asarray(r), device=dev))
        self.c_body = torch.nonzero(G.patch_A > 0).flatten()                       # the exposed spheres
        self.c_r, self.c_A, self.c_N = rt[self.c_body], G.patch_A[self.c_body].detach(), G.patch_nn[self.c_body].detach()
        self.Vres = self.V-front-z1.index_add(0, self.c_body, self.c_r*self.c_A/3)
        self.mu = float(model.mu_ref)
        z33 = torch.zeros(n, 3, 3, dtype=dt, device=dev)
        half = (self.A*self.L/2)[:, None, None]*self.nrm[:, :, None]*self.nrm[:, None, :]
        self.Mb = z33.index_add(0, self.i, half).index_add(0, self.j, half)
        self.Mcap = z33.index_add(0, self.c_body, self.c_r[:, None, None]*self.c_N)
        self.Mt_inv = torch.linalg.inv(self.Mb+self.Mcap)
        nint = G.patch_n.detach()                                                   # int n dA over the body's caps
        self.cap = torch.zeros(n, dtype=torch.bool, device=dev); self.cap[self.c_body] = True
        self.m = torch.where(self.cap[:, None], nint/nint.norm(dim=1, keepdim=True).clamp(min=1e-300), torch.zeros_like(nint))
        self.mm = self.m[:, :, None]*self.m[:, None, :]
        self.P = torch.eye(3, dtype=dt, device=dev)-self.mm
        self.iters = iters
        self.kap0 = float(model.k_v)/(float(model.k_v)+2*self.mu)
        self._scheme_a(G)

    def _scheme_a(self, G, K=16):
        """scheme A's rest data: the rim entries (cell r_c, its neighbour r_k across the wall the arc lies on, weight r_w =
        the arc's length over the cell's rim; both cells of each arc), the caps' tangent frames t1, t2 about m (interior
        cells: any frame, M_c = 0 there)"""
        split_rings(G)
        n, dev, dt = self.n, self.X.device, self.X.dtype
        ap = G.arc_pair; ai, aj = G.P[ap, 0], G.P[ap, 1]
        Y = torch.stack([G.arc_points(G.arc_t0+(G.arc_t1-G.arc_t0)*(k/K)) for k in range(K+1)], 1).detach()
        alen = (Y[:, 1:]-Y[:, :-1]).norm(dim=2).sum(1)
        rc, rk, rl = torch.cat((ai, aj)), torch.cat((aj, ai)), torch.cat((alen, alen))
        tot = torch.zeros(n, dtype=dt, device=dev).index_add(0, rc, rl)
        assert bool((tot[self.cap] > 0).all()), 'a surface cell without a rim'
        self.r_c, self.r_k, self.r_w = rc, rk, rl/tot[rc]
        ez = torch.tensor([0., 0., 1.], dtype=dt, device=dev)
        self.mA = torch.where(self.cap[:, None], self.m, ez)
        a_ = torch.where((self.mA[:, 0].abs() < .9)[:, None], torch.tensor([1., 0., 0.], dtype=dt, device=dev), torch.tensor([0., 1., 0.], dtype=dt, device=dev))
        t1 = torch.linalg.cross(self.mA, a_); self.t1 = t1/t1.norm(dim=1, keepdim=True)
        self.t2 = torch.linalg.cross(self.mA, self.t1)
        self.PA = torch.eye(3, dtype=dt, device=dev)-self.mA[:, :, None]*self.mA[:, None, :]
        self.Mt = self.Mb+self.Mcap

    def abar(self, F):
        """sum_e w_e F_k(e) per cell (the rim's neighbours' mean F)"""
        return torch.zeros_like(F).index_add(0, self.r_c, self.r_w[:, None, None]*F[self.r_k])

    def completion_a(self, Fb, kap):
        """F_s from the rim's mean deformation Fbar (scheme A)"""
        c1 = torch.einsum('bij,bj->bi', Fb, self.t1); c2 = torch.einsum('bij,bj->bi', Fb, self.t2)
        nt = torch.linalg.cross(c1, c2); a = nt.norm(dim=1)
        s = 1-kap*(torch.sqrt((c1*c1).sum(1)+(c2*c2).sum(1)+2*a)-2)
        return Fb@self.PA+s[:, None, None]*(nt/a[:, None])[:, :, None]*self.mA[:, None, :]

    def residual_a(self, F, S, kap, Ab=None):
        """F M_t - S - F_s(Fbar) M_c; Ab given: the neighbours' part held"""
        Ab = self.abar(F) if Ab is None else Ab
        return F@self.Mt-S-self.completion_a(.5*F+.5*Ab, kap)@self.Mcap

    def jacobian_a(self, F, S, kap, Ab):
        """each body's own 9 x 9 block of d residual_a / dF (the neighbours' part Ab held), row-major"""
        n = len(F)
        E9 = torch.eye(9, dtype=F.dtype, device=F.device).view(9, 1, 3, 3).expand(9, n, 3, 3)
        cols = torch.func.vmap(lambda T: torch.func.jvp(lambda F_: self.residual_a(F_, S, kap, Ab), (F,), (T,))[1])(E9)
        return cols.reshape(9, n, 9).permute(1, 2, 0)

    def coupled_solve(self, F, kap, b, transpose=False, tol=1e-14, maxit=20000):
        """J x = b (or J^T x = b) for the coupled closure's Jacobian J at F, block-Jacobi on the own blocks"""
        n = len(F)
        D = self.jacobian_a(F, torch.zeros_like(F), kap, self.abar(F))
        if transpose:
            D = D.transpose(1, 2)
            _, vjp = torch.func.vjp(lambda F_: self.residual_a(F_, torch.zeros_like(F_), kap), F)
            Jx = lambda X_: vjp(X_)[0]
        else:
            Jx = lambda X_: torch.func.jvp(lambda F_: self.residual_a(F_, torch.zeros_like(F_), kap), (F,), (X_,))[1]
        x = torch.zeros_like(b); bn = float(b.norm())
        for _ in range(maxit):
            r_ = b-Jx(x)
            if float(r_.norm()) <= tol*bn:
                break
            x = x+torch.linalg.solve(D, r_.reshape(n, 9)).view(n, 3, 3)
        return x

    def wall_volume(self, x, F, R):
        """SphereModel.wall_volume: the walls' pyramids from the bodies' centres to the deformed walls"""
        i, j = self.i, self.j
        Ff, Rf = .5*(F[i]+F[j]), .5*(R[i]+R[j])
        mv = lambda M_, v: torch.einsum('bij,bj->bi', M_, v)
        line = (self.hj/self.L)[:, None]*(x[i]+mv(R[i], self.ri))+(self.hi/self.L)[:, None]*(x[j]+mv(R[j], self.rj))+mv(Ff-Rf, self.t)
        comp = (self.hj/self.L)[:, None]*(x[i]+mv(F[i], self.ri))+(self.hi/self.L)[:, None]*(x[j]+mv(F[j], self.rj))
        xf = .5*(line+comp)
        a = mv(cof3(Ff), self.A[:, None]*self.nrm)
        vi = (self.wi*((xf-x[i])*a).sum(1))/3; vj = (self.wj*(-(xf-x[j])*a).sum(1))/3
        return torch.zeros(self.n, dtype=x.dtype, device=x.device).index_add(0, i, vi).index_add(0, j, vj)

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
        """F_B from the walls' S_B (n, 3, 3); returns F_B and the last sweep's step (max abs). Scheme A's coupled closure
        by block-Jacobi Newton sweeps (each body's own 9 x 9 block, forward mode, the rim's neighbours lagged) to a 1e-14
        step. The returned F_B is one more Newton step with the coupled Jacobian held (_CoupledSolve) and the graph on S
        and kap: at the root its value is the root and its first derivative the implicit one (second derivatives are
        not: Hessians by differences of gradients). On the part: rest / rigid 1e-15, gradient vs differences O(h^2)."""
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
            for _ in range(20000 if iters is None else iters):                     # (block-Jacobi Newton sweeps)
                Ab = self.abar(F)
                J = self.jacobian_a(F, Sd, kd, Ab)
                dF = torch.linalg.solve(J, -self.residual_a(F, Sd, kd, Ab).reshape(n, 9)).view(n, 3, 3)
                F = F+dF; ch = float(dF.abs().max())
                if ch < 1e-14:
                    break
            if getattr(self, 'warm', False):
                self.F_warm = F
        return F-_CoupledSolve.apply(self.residual_a(F, S, kap), self, F, kd), ch

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
        R = rot_proper(F); q = matrix_quat(R)
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
