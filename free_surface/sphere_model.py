"""Bonded cells with a continuous sphere surface (2026-10-22, step 3 for the synthetic part; sphere_skin.py,
part_geometry.py, sphere_points.py). Degrees of freedom: body positions x_B and the skin points y_q (corners, arc points,
part points; shared where the surface is shared). Each patch (the part of cell c's sphere left free, cell c of body B,
site X_c, radius r) is a set of parts with shape functions phi_q; the surface there is
  y(X) = x_B + F_B (X - X_B) + sum_q phi_q(X) delta_q,   delta_q = y_q - x_B - F_B (Y_q - X_B)
(the cell's affine field plus the interpolated skin deviation; exact for affine maps). Every term is an integral over
the true patch, with the rest integrals a_q = int phi_q dA, c_q = int phi_q n dA, m_qp = int phi_q phi_p dA, A, N = int
n n^T dA:
  outer layer  sum_patches (mu / r) int |y(X) - x_bar_c - R_B r n|^2 dA     (x_bar_c = x_B + F_B (X_c - X_B))
               = mu r tr(D^T D N) + 2 mu sum_q delta_q . D c_q + (mu / r) sum_qp m_qp delta_q . delta_p,  D = F_B - R_B
               (fibres of length r from the site, turning with the body; for affine maps the first term alone: the
               sphere's share of the cell's moment, sum_walls A sym((h n + t) (x) n) + r N = V I, so with the bonds'
               A L n n the mu energy is mu V |U - I|^2 exactly)
  volume       walls (Hermite, CleanTerms.J) + sum_patches (1/3) int (y - x_bar_c) . cof(F_B) n dA + det(F_B) V_res
               = walls + det(F_B) r A / 3 + (1/3) sum_q delta_q . cof(F_B) c_q + det(F_B) V_res,
               V_res = V_B - front walls - sum r A / 3 at rest (walls behind the site, faces inside merged bodies)
  lateral      the outer layer's deformation along the surface comes from the surface itself (shared, continuous), its
               normal column is free (sigma . n = 0 <=> dW / d(F n) = 0 with the tangential part fixed): the energy of the
               surface's tangential strain beyond the cell's, eps_t = sym(P grad_s (R_B^T delta) P) (P = I - n n^T, rest
               frame; the deviation turned back by the body's rotation, 10-23: objective), with the normal
               relaxed (eps_nn = -lam / (lam + 2 mu) tr eps_t, eps_nt = 0): mu eps_t : eps_t + lam mu / (lam + 2 mu) (tr eps_t)^2,
               over the cone from the site (delta grown linearly along the radius: r / 3 per unit area); zero for affine maps
  reference    (quad_ref, 10-22) the skin terms above (cross, deviation, lateral) compare the skin with the cell's affine
               field plus its curvature, delta = y_q - x_B - F_B d - 1/2 Gamma_B[d, d], Gamma_B = dF/dX fitted over the bonds:
               in the parallel-bond picture the walls' moments already carry how the strain varies across a cell (the
               surface side of a wall stretched more in bending), so only what no bond sees, the skin's roughness below
               the cell scale, is charged here; the volume keeps the skin's true position (delta_aff)
  gradient     F_B = [S_B + sum_q w_q (y_q - x_B) (x) l_q] [M_B + sum_q w_q (Y_q - X_B) (x) l_q]^-1, l_q the normal part of
               the lever (10-22: the skin informs the normal column only; with the full lever the flat-face uniform
               free-surface test left 2 % on the skin points),
               w_q = a_q / |Y_q - X_B| (the point's moment a_q x its distance, as a bond's A x L / 2)
Bonds, bond rotations R_B = polar(F_B) and the variance term as the clean model (grad_weights.MidpointTerms' L / 2
weights). Polynomial det / cofactor (second derivatives at F = I)."""
import numpy as np
import torch

from experiments.foam_surface_completion import polar_newton
from free_surface.surface_model import cof3, det3


def matrix_quat(R):
    """Unit quaternions (w, x, y, z) of rotation matrices (n, 3, 3) at any angle (10-23; the earlier
    foam_surface_completion.matrix_quat divides by w = sqrt(1 + tr) / 2, which vanishes at a half turn). Shepperd: from the
    largest of 4w^2 = 1 + tr, 4x^2 = 1 + 2 R_00 - tr, 4y^2, 4z^2 (at least 1, as they add up to 4). The sign follows the
    branch; the bond energy uses q only through quat_matrix and quat_rel_principal, which do not see it. All four branches
    are computed with their squares clamped finite and the largest gathered, so the derivatives stay finite."""
    R00, R11, R22 = R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]
    tr = R00+R11+R22
    c = torch.stack((1+tr, 1+2*R00-tr, 1+2*R11-tr, 1+2*R22-tr), 1)
    s = torch.sqrt(c.clamp(min=1e-12))                                               # 2|w|, 2|x|, 2|y|, 2|z|
    a21, a02, a10 = R[:, 2, 1]-R[:, 1, 2], R[:, 0, 2]-R[:, 2, 0], R[:, 1, 0]-R[:, 0, 1]   # 4wx, 4wy, 4wz
    b01, b02, b12 = R[:, 0, 1]+R[:, 1, 0], R[:, 0, 2]+R[:, 2, 0], R[:, 1, 2]+R[:, 2, 1]   # 4xy, 4xz, 4yz
    s0, s1, s2, s3 = s.unbind(1)
    qs = torch.stack((torch.stack((s0/2, a21/(2*s0), a02/(2*s0), a10/(2*s0)), 1),
                      torch.stack((a21/(2*s1), s1/2, b01/(2*s1), b02/(2*s1)), 1),
                      torch.stack((a02/(2*s2), b01/(2*s2), s2/2, b12/(2*s2)), 1),
                      torch.stack((a10/(2*s3), b02/(2*s3), b12/(2*s3), s3/2), 1)), 1)  # (n, branch, 4)
    k = c.argmax(1)
    return torch.gather(qs, 1, k[:, None, None].expand(-1, 1, 4))[:, 0]


def skin_rest(skin, pts, p, r):
    """rest positions of the skin points: corners, arc points (equal angles), part points (site + r pole)"""
    nc = int(pts['n_corners']); aoff = pts['arc_off']; nsub = pts['arc_nsub']
    Y = np.zeros((int(pts['n_points']), 3))
    Y[:nc] = skin['corners']
    for ai in range(len(nsub)):
        cc = skin['arc_circle'][ai]; t0, t1 = skin['arc_theta'][ai]
        for s in range(1, nsub[ai]):
            t = t0+(t1-t0)*s/nsub[ai]
            Y[nc+aoff[ai]+s-1] = cc[0:3]+cc[3]*(np.cos(t)*cc[4:7]+np.sin(t)*cc[7:10])
    base = nc+int(aoff[-1])
    site = pts['part_site']
    Y[base:base+len(site)] = p[site]+r[site, None]*pts['part_pole']
    return Y


class SkinTerms:
    """the model for any skin given as arrays (sphere patches: SphereTerms; flat faces: free_surface/flat_skin.py):
    Y rest skin points; ent: per (part, point) entry pid, body, a = int phi dA, c = int phi n dA; parts: off (entries per
    part), body, h (outer-layer thickness: the sphere radius, or the site-to-face distance), m (and k, T, U) flattened
    with moff; patches: body, h, A, N = int n n^T dA (one per sphere patch or flat face)"""
    def __init__(self, model, geometry, Y, ent, parts, patches, variance=True, quad_ref=True, gamma='sym_skin'):
        dt, dev = model.X.dtype, model.X.device
        self.quad_ref = quad_ref
        tt = lambda a: torch.as_tensor(np.asarray(a), dtype=dt, device=dev)
        ti = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.long, device=dev)
        self.model, self.variance = model, variance
        X = model.X; self.X = X; n = len(X); self.n = n
        self.i, self.j = model.i, model.j
        self.A = tt(geometry['areas']); c = tt(geometry['contacts'])
        d = X[self.j]-X[self.i]; self.L = d.norm(dim=1); self.nrm = d/self.L[:, None]
        self.hi = ((c-X[self.i])*self.nrm).sum(1); self.hj = self.L-self.hi
        self.ri, self.rj = c-X[self.i], c-X[self.j]
        self.t = c-(X[self.i]+self.hi[:, None]*self.nrm)
        self.wi, self.wj = (self.hi > 0).to(dt), (self.hj > 0).to(dt)
        nn = self.nrm[:, :, None]*self.nrm[:, None, :]
        half = (self.A*self.L/2)[:, None, None]*nn
        Mb = torch.zeros(n, 3, 3, dtype=dt, device=dev).index_add(0, self.i, half).index_add(0, self.j, half)
        self.V = tt(geometry['masses'])
        front = torch.zeros(n, dtype=dt, device=dev).index_add(0, self.i, self.wi*self.A*self.hi/3).index_add(0, self.j, self.wj*self.A*self.hj/3)
        # skin entries: gradient weight a / |Y - X_B| (the point's moment: its area share x its distance)
        self.Y = tt(Y); self.npt = len(Y)
        self.e_pid = ti(ent['pid']); self.e_body = ti(ent['body']); self.e_c = tt(ent['c'])
        self.e_d = self.Y[self.e_pid]-X[self.e_body]
        # a skin point informs only the normal column of F_B (the tangential columns are the walls' bonds): its lever in
        # the fit is the normal part of d, l = (d . n) n (n the point's surface normal: radial on a sphere, so l = d
        # there); weight a / |l| (moment a x distance); M_skin = sum w d (x) l keeps the fit exact for affine maps
        en = tt(ent['n'])
        self.e_l = (self.e_d*en).sum(1, keepdim=True)*en
        self.e_w = tt(ent['a'])/self.e_l.norm(dim=1)
        Ms = torch.zeros(n, 3, 3, dtype=dt, device=dev).index_add(0, self.e_body, self.e_w[:, None, None]*self.e_d[:, :, None]*self.e_l[:, None, :])
        self.Mt_inv = torch.linalg.inv(Mb+Ms)
        # curvature: Gamma_B = d F / d X by least squares over the bonds, (F_j - F_B) ~ Gamma_B (X_j - X_B), the bonds'
        # L / 2 weights (moment Mb); symmetric in its two derivative indices
        # pseudo-inverse: a cell with two walls (new foam, 10-22) has a rank-2 bond moment; along the missing direction
        # the bonds carry no curvature information, and no curvature reference is used there (numerical rank only)
        self.Mb_inv = torch.linalg.pinv(Mb.cpu(), hermitian=True).to(dev)        # (on the CPU: the GPU batched eigensolver took 4 GB
                                                                                    # of workspace for the 8k jack, 15 GB for 30k, 10-26)
        self.dX = X[self.j]-X[self.i]; self.wb = self.A/(2*self.L)
        # gamma 'sym_skin' (10-23): Gamma_i (symmetric in its derivative indices, 6 unknowns) by least squares over the bonds,
        # wb |F_j - F_B - Gamma dX|^2, plus the skin points as observations that F does not change toward them,
        # (|c| / |d|) |Gamma d|^2, the same moment as the skin's lever in the F_B fit (sum (|c| / |d|) d d^T = sum d (x) c when d
        # is along c): no amplification where the bonds barely see a direction, and the seen components exact (the
        # symmetry gives the through-thickness ones from the in-plane ones). 'bonds': the earlier fit (Mb pinv, symmetrised).
        self.gamma = gamma
        def amat(v):                                          # S v = A(v) s, s = (S00, S11, S22, S01, S02, S12)
            z = torch.zeros_like(v[:, 0])
            return torch.stack((torch.stack((v[:, 0], z, z, v[:, 1], v[:, 2], z), 1), torch.stack((z, v[:, 1], z, v[:, 0], z, v[:, 2]), 1),
                                torch.stack((z, z, v[:, 2], z, v[:, 0], v[:, 1]), 1)), 1)
        if gamma == 'sym_skin':
            self.g_A = amat(self.dX)                                                          # (nb, 3, 6)
            AtA = self.wb[:, None, None]*self.g_A.transpose(1, 2)@self.g_A
            Ne = torch.zeros(n, 6, 6, dtype=dt, device=dev).index_add(0, self.i, AtA).index_add(0, self.j, AtA)
            Ad = amat(self.e_d)
            om = self.e_c.norm(dim=1)/self.e_d.norm(dim=1)
            Ne = Ne.index_add(0, self.e_body, om[:, None, None]*Ad.transpose(1, 2)@Ad)
            self.g_Ninv = torch.linalg.pinv(Ne, hermitian=True)
            d = self.e_d
            self.g_q = torch.stack((d[:, 0]**2, d[:, 1]**2, d[:, 2]**2, 2*d[:, 0]*d[:, 1], 2*d[:, 0]*d[:, 2], 2*d[:, 1]*d[:, 2]), 1)
        # parts' point pairs (a, b) with m_ab and, for the lateral term, T_ab = int g_a g_b^T, U_ab = int (g_a . g_b) P
        off = parts['off']; P = len(off)-1
        pa, pb, pp = [], [], []
        for k in range(P):
            s_ = off[k+1]-off[k]
            pa.append(off[k]+np.repeat(np.arange(s_), s_)); pb.append(off[k]+np.tile(np.arange(s_), s_)); pp.append(np.full(s_*s_, k))
        self.pr_a, self.pr_b, self.pr_part = ti(np.concatenate(pa)), ti(np.concatenate(pb)), ti(np.concatenate(pp))
        self.pr_m = tt(parts['m'])
        self.pr_k = tt(parts['k']) if 'k' in parts else None
        self.lateral = 'T' in parts
        if self.lateral:
            self.pr_T = tt(np.asarray(parts['T']).reshape(-1, 3, 3)); self.pr_U = tt(np.asarray(parts['U']).reshape(-1, 3, 3))
        self.p_r = tt(parts['h']); self.p_body = ti(parts['body'])
        # patches: outer-layer moment h N, cone / pyramid volume h A / 3
        self.c_body = ti(patches['body']); self.c_r = tt(patches['h']); self.c_A = tt(patches['A']); self.c_N = tt(patches['N'])
        cone = self.c_r*self.c_A/3
        self.Vres = self.V-front-torch.zeros(n, dtype=dt, device=dev).index_add(0, self.c_body, cone)
        self.mu = float(model.mu_ref)

    def F(self, x, y):
        d = x[self.j]-x[self.i]
        t = (self.A/2)[:, None, None]*d[:, :, None]*self.nrm[:, None, :]
        S = torch.zeros(self.n, 3, 3, dtype=x.dtype, device=x.device).index_add(0, self.i, t).index_add(0, self.j, t)
        g = self.e_w[:, None, None]*(y[self.e_pid]-x[self.e_body])[:, :, None]*self.e_l[:, None, :]
        return S.index_add(0, self.e_body, g)@self.Mt_inv

    def wall_volume(self, x, F, R):
        i, j = self.i, self.j
        Ff, Rf = .5*(F[i]+F[j]), .5*(R[i]+R[j])
        mv = lambda M_, v: torch.einsum('bij,bj->bi', M_, v)
        line = (self.hj/self.L)[:, None]*(x[i]+mv(R[i], self.ri))+(self.hi/self.L)[:, None]*(x[j]+mv(R[j], self.rj))+mv(Ff-Rf, self.t)
        comp = (self.hj/self.L)[:, None]*(x[i]+mv(F[i], self.ri))+(self.hi/self.L)[:, None]*(x[j]+mv(F[j], self.rj))
        xf = .5*(line+comp)
        a = mv(cof3(Ff), self.A[:, None]*self.nrm)
        vi = (self.wi*((xf-x[i])*a).sum(1))/3; vj = (self.wj*(-(xf-x[j])*a).sum(1))/3
        return torch.zeros(self.n, dtype=x.dtype, device=x.device).index_add(0, i, vi).index_add(0, j, vj)

    def energy_terms(self, theta, x, y):
        F = self.F(x, y)
        R = polar_newton(F); q = matrix_quat(R)
        m = self.model
        bonds = m.energy(theta, x, q)
        if getattr(m, 'body_law', None) == 'nodalvol':
            bonds = bonds-m._nodalvol(theta, x)[0]
        lam = m.k_v*torch.exp(theta[0, 4]); sm = torch.exp(theta[:, 5]); mu = self.mu
        D = F-R
        Fe = F[self.e_body]
        delta_aff = y[self.e_pid]-x[self.e_body]-torch.einsum('eij,ej->ei', Fe, self.e_d)   # skin minus the cell's affine field
        delta = delta_aff
        if self.quad_ref and self.gamma == 'sym_skin':
            rb = self.wb[:, None, None]*torch.einsum('bkm,bik->bim', self.g_A, F[self.j]-F[self.i])        # (nb, 3, 6)
            rhs = torch.zeros(self.n, 3, 6, dtype=x.dtype, device=x.device).index_add(0, self.i, rb).index_add(0, self.j, rb)
            sg = torch.einsum('bmn,bin->bim', self.g_Ninv, rhs)
            delta = delta_aff-.5*torch.einsum('eim,em->ei', sg[self.e_body], self.g_q)
        elif self.quad_ref:                                   # minus the cell's curvature too: what the bonds' moments carry
            g = self.wb[:, None, None, None]*(F[self.j]-F[self.i])[:, :, :, None]*self.dX[:, None, None, :]
            G = torch.zeros(self.n, 3, 3, 3, dtype=x.dtype, device=x.device).index_add(0, self.i, g).index_add(0, self.j, g)
            Gam = torch.einsum('bijl,blk->bijk', G, self.Mb_inv)
            Gam = .5*(Gam+Gam.transpose(2, 3))
            delta = delta_aff-.5*torch.einsum('eijk,ej,ek->ei', Gam[self.e_body], self.e_d, self.e_d)
        # outer layer
        Dc = D[self.c_body]
        o1 = (mu*self.c_r*sm[self.c_body]*torch.einsum('cki,ckj,cij->c', Dc, Dc, self.c_N)).sum()
        # first order (the normal column's coupling to the skin) with the skin's true deviation: zero force at uniform
        # free-surface states; the squares below (roughness) with the curvature-aware deviation
        o2 = (2*mu*sm[self.e_body]*(delta_aff*torch.einsum('eij,ej->ei', D[self.e_body], self.e_c)).sum(1)).sum()
        # the squares in the body's frame (10-23): the lateral strain pairs the deviation with rest-frame surface gradients
        # and projectors, so the deviation is turned back by the body's rotation first (objective under rotations; at
        # rest R = I, so the Hessian there is unchanged)
        delta = torch.einsum('eji,ej->ei', R[self.e_body], delta)
        da, db = delta[self.pr_a], delta[self.pr_b]
        mu_p = (mu*sm[self.p_body])[self.pr_part]; r_p = self.p_r[self.pr_part]
        o3 = (mu_p/r_p*self.pr_m*(da*db).sum(1)).sum()
        terms = {}
        if self.lateral:
            kap = lam*mu_p/(lam+2*mu_p)
            tU = torch.einsum('ni,nij,nj->n', da, self.pr_U, db)
            tT = torch.einsum('ni,nij,nj->n', da, self.pr_T, db)
            tTs = torch.einsum('ni,nij,nj->n', db, self.pr_T, da)             # sum_ab delta_a^T T_ba delta_b, relabelled
            terms['outer_lateral'] = (r_p/3*(.5*mu_p*(tU+tTs)+kap*tT)).sum()
        # volume
        detF = det3(F)
        Vd = self.wall_volume(x, F, R)+detF*(self.Vres+torch.zeros_like(self.V).index_add(0, self.c_body, self.c_r*self.c_A/3))
        Vd = Vd.index_add(0, self.e_body, (delta_aff*torch.einsum('eij,ej->ei', cof3(Fe), self.e_c)).sum(1)/3)
        lj, _ = m._log_J(Vd/self.V)
        vol = (self.V*.5*lam*lj**2).sum()
        if self.variance:
            vol = vol+(.5*lam*self.A*self.L/12*(lj[self.i]-lj[self.j])**2).sum()
        return dict(bonds=bonds, volume=vol, outer_stretch=o1, outer_cross=o2, outer_skin=o3, **terms)

    def energy_xy(self, theta, x, y):
        return sum(self.energy_terms(theta, x, y).values())

    def affine(self, Fa, c, t=None):
        t = torch.zeros(3, dtype=self.X.dtype, device=self.X.device) if t is None else t
        return c+(self.X-c)@Fa.T+t, c+(self.Y-c)@Fa.T+t


class SphereTerms(SkinTerms):
    def __init__(self, model, geometry, p, r, skin, pts, variance=True, quad_ref=True):
        boc = np.asarray(geometry['body_of_cell'])
        off = pts['part_off']; site = pts['part_site']
        e_part = np.repeat(np.arange(len(off)-1), np.diff(off))
        Yr = skin_rest(skin, pts, p, r)
        nrad = (Yr[pts['part_pid']]-p[site[e_part]])/r[site[e_part], None]       # radial: the sphere's own normal
        ent = dict(pid=pts['part_pid'], body=boc[site[e_part]], a=pts['part_a'], c=pts['part_c'], n=nrad)
        parts = dict(off=off, body=boc[site], h=r[site], m=pts['part_m'])
        for k_ in ('k', 'T', 'U'):
            if 'part_'+k_ in pts:
                parts[k_] = pts['part_'+k_]
        cells = np.flatnonzero(skin['patch_area'] > 0)
        patches = dict(body=boc[cells], h=r[cells], A=skin['patch_area'][cells], N=skin['patch_nn'][cells])
        super().__init__(model, geometry, Yr, ent, parts, patches, variance, quad_ref)
