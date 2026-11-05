"""Tetrahedron-free volume, rotation constraint and completion (KB_DERIVATION.md, 2026-10-20), large deformation, torch.

Cell gradient (linear in the positions, exact for any affine map, objective: F(Rx) = R F(x)):
  S_B = sum_{front walls} A (h_B/L) (x_j - x_B) n^T            (walls with h_B > 0; a wall behind the site is left out)
  F_B = [S_B + F_N M_air,B] (M_B + M_air,B)^-1,   F_N = (sum_{nbrs} S_k)(sum_{nbrs} M_k)^-1
  M_B = sum_{front walls} A h_B n n^T,  M_air,B = sum_{completion springs of B} (k / 2 mu) l l^T   (air faces, patches)
Rotation: R_B = polar(F_B) (constraint). Wall position (Hermite, the mean of two estimates, exact for affine maps):
  line = (h_j/L)(x_i + R_i r_i) + (h_i/L)(x_j + R_j r_j) + (F_f - R_f) t,   comp = (h_j/L)(x_i + F_i r_i) + (h_i/L)(x_j + F_j r_j)
  with F_f, R_f the two sides' means, t the in-plane offset of the contact point from the bond line.
Cell volume (deformed polyhedron, exact for affine maps): V'_B = 1/3 sum_{front walls} (x_f - x_B).cof(F_f) A n_B
  + det(F_B) (V_B - 1/3 sum_{front walls} A h_B)   (air faces, walls behind the site and merged-face residue on F_B).
Energy: bonds (kn, ks, kb, kt; rotations R_B) + sum_B V_B lam/2 (ln J_B)^2 + sum_bonds lam/2 (A L/12)(ln J_i - ln J_j)^2
  + sum_springs k/2 |(F_B - R_B) l|^2.
"""
import numpy as np
import torch

from experiments.foam_surface_completion import matrix_quat, polar_newton


def cof(F):
    return torch.linalg.det(F)[:, None, None]*torch.linalg.inv(F).transpose(1, 2)


def polar_svd(F):
    U, S, Vh = torch.linalg.svd(F)
    d = torch.sign(torch.linalg.det(U@Vh))
    D = torch.ones_like(S); D[:, 2] = d
    return U@torch.diag_embed(D)@Vh


class CleanTerms:
    def __init__(self, model, sc, geometry, variance=True):
        dt, dev = model.X.dtype, model.X.device
        tt = lambda a: torch.as_tensor(np.asarray(a), dtype=dt, device=dev)
        X = model.X; self.X = X; n = len(X); self.n = n
        self.i, self.j = model.i, model.j
        self.A = tt(geometry['areas']); c = tt(geometry['contacts'])
        d = X[self.j]-X[self.i]; self.L = d.norm(dim=1); self.nrm = d/self.L[:, None]
        self.hi = ((c-X[self.i])*self.nrm).sum(1); self.hj = self.L-self.hi
        self.ri, self.rj = c-X[self.i], c-X[self.j]
        self.t = c-(X[self.i]+self.hi[:, None]*self.nrm)
        self.wi, self.wj = (self.hi > 0).to(dt), (self.hj > 0).to(dt)
        nn = self.nrm[:, :, None]*self.nrm[:, None, :]
        self.M = torch.zeros(n, 3, 3, dtype=dt, device=dev).index_add(0, self.i, (self.wi*self.A*self.hi)[:, None, None]*nn).index_add(
            0, self.j, (self.wj*self.A*self.hj)[:, None, None]*nn)
        mu = float(model.mu_ref); l = sc.vs_lever
        self.Mair = torch.zeros(n, 3, 3, dtype=dt, device=dev).index_add(0, sc.vs_body, (sc.vs_k/(2*mu))[:, None, None]*l[:, :, None]*l[:, None, :])
        self.MN_inv = torch.linalg.inv(self.nbr_sum(self.M))
        self.Mt_inv = torch.linalg.inv(self.M+self.Mair)
        self.V = sc.V0
        front = torch.zeros(n, dtype=dt, device=dev).index_add(0, self.i, self.wi*self.A*self.hi/3).index_add(0, self.j, self.wj*self.A*self.hj/3)
        self.Vrest = self.V-front                    # air faces, walls behind the site, merged-face residue
        self.sc, self.model, self.variance = sc, model, variance
        self.polar = polar_svd

    def nbr_sum(self, T):
        return torch.zeros_like(T).index_add(0, self.i, T[self.j]).index_add(0, self.j, T[self.i])

    def F(self, x):
        d = x[self.j]-x[self.i]
        on = lambda w, h: (w*self.A*h/self.L)[:, None, None]*d[:, :, None]*self.nrm[:, None, :]
        S = torch.zeros(self.n, 3, 3, dtype=x.dtype, device=x.device).index_add(0, self.i, on(self.wi, self.hi)).index_add(0, self.j, on(self.wj, self.hj))
        FN = self.nbr_sum(S)@self.MN_inv
        return (S+FN@self.Mair)@self.Mt_inv

    def J(self, x, F, R):
        i, j = self.i, self.j
        Ff, Rf = .5*(F[i]+F[j]), .5*(R[i]+R[j])
        mv = lambda M_, v: torch.einsum('bij,bj->bi', M_, v)
        line = (self.hj/self.L)[:, None]*(x[i]+mv(R[i], self.ri))+(self.hi/self.L)[:, None]*(x[j]+mv(R[j], self.rj))+mv(Ff-Rf, self.t)
        comp = (self.hj/self.L)[:, None]*(x[i]+mv(F[i], self.ri))+(self.hi/self.L)[:, None]*(x[j]+mv(F[j], self.rj))
        xf = .5*(line+comp)
        a = mv(cof(Ff), self.A[:, None]*self.nrm)                      # deformed area vector, outward from i
        vi = (self.wi*((xf-x[i])*a).sum(1))/3; vj = (self.wj*(-(xf-x[j])*a).sum(1))/3
        Vd = torch.zeros(self.n, dtype=x.dtype, device=x.device).index_add(0, i, vi).index_add(0, j, vj)
        Vd = Vd+torch.linalg.det(F)*self.Vrest
        return Vd/self.V

    def volume_energy(self, x, F, R, lam):
        lj, _ = self.model._log_J(self.J(x, F, R))
        E = (self.V*.5*lam*lj**2).sum()
        if self.variance:
            E = E+(.5*lam*self.A*self.L/12*(lj[self.i]-lj[self.j])**2).sum()
        return E

    def completion_energy(self, F, R, theta=None):
        """k/2 |(F_B - R_B) l|^2 with the shear multiplier exp(theta[:, 5]) (as cl_comp_kernel)"""
        sc = self.sc
        dd = torch.einsum('mij,mj->mi', (F-R)[sc.vs_body], sc.vs_lever)
        k = sc.vs_k if theta is None else sc.vs_k*torch.exp(theta[sc.vs_body, 5])
        return .5*(k*(dd*dd).sum(-1)).sum()

    def energy(self, theta, x, smooth_polar=False):
        """total elastic energy (per unit density) of the clean model at positions x; theta as the model's (kn ks kb kt
        multipliers in columns 0-3, lam in 4); returns (E, R)"""
        F = self.F(x)
        R = polar_newton(F) if smooth_polar else self.polar(F)
        q = matrix_quat(R)
        m = self.model
        bonds = m.energy(theta, x, q)
        if getattr(m, 'body_law', None) == 'nodalvol':
            bonds = bonds-m._nodalvol(theta, x)[0]
        lam = m.k_v*torch.exp(theta[0, 4])
        return bonds+self.volume_energy(x, F, R, lam)+self.completion_energy(F, R, theta), R


def np_data(ct):
    """Arrays for WarpFoam (completion dict keys cl_*): per bond and per body, float64 numpy (bonds in the model's order)."""
    K = (ct.MN_inv@ct.Mair@ct.Mt_inv)                     # F_B = S_B Mt^-1 + S_nbrs K
    c = lambda t_: t_.detach().cpu().numpy()
    # ground wall (KB_DERIVATION.md 20): per completion spring the air face's area and site-to-face distance (0 for exposed
    # patches and faces with the site outside, which never touch as walls); MN^-1 to drop a held face from K
    sc = ct.sc
    nair = len(sc.air_faces['body'])
    area = np.zeros(len(sc.vs_k)); h = np.zeros(len(sc.vs_k))
    area[:nair] = sc.air_faces['area']
    Ls = c(sc.vs_lever.norm(dim=1))
    h[:nair] = c(sc.vs_k[:nair])*Ls[:nair]**2/(2*float(ct.model.mu_ref)*area[:nair])
    area[h <= 0] = 0.
    return dict(cl_ci=c(ct.wi*ct.A*ct.hi/ct.L), cl_cj=c(ct.wj*ct.A*ct.hj/ct.L), cl_hi=c(ct.hi), cl_hj=c(ct.hj), cl_L=c(ct.L),
                cl_A=c(ct.A), cl_t=c(ct.t), cl_wi=c(ct.wi), cl_wj=c(ct.wj), cl_var=c(ct.A*ct.L/12*(1. if ct.variance else 0.)),
                cl_Mtinv=c(ct.Mt_inv), cl_K=c(K), cl_Vrest=c(ct.Vrest), cl_MNinv=c(ct.MN_inv), cl_gc_area=area, cl_gc_h=np.maximum(h, 0.))


def frequency_ratio(model, ct, iterations=60, seed=0):
    """Highest frequency at rest of the clean energy (rotations slaved by Newton polar) over the model's own (free
    rotations), by power iteration as foam_surface_completion.frequency_ratio; multiplies the substep count."""
    from experiments.foam_dem_energy import quat_mul, quat_small
    n = model.n; dt_, dev = model.X.dtype, model.X.device
    th = torch.zeros(n, 9, dtype=dt_, device=dev)
    x0, q0, _, _ = model.initial_state()
    out = []
    for kind in ('model', 'clean'):
        mvec = torch.cat((model.mass[:, None].expand(-1, 3), model.inertia[:, None].expand(-1, 3)), 1) if kind == 'model' else model.mass[:, None].expand(-1, 3)
        gen = torch.Generator(device='cpu').manual_seed(seed)
        y = torch.randn(*mvec.shape, generator=gen, dtype=torch.float64).to(dev, dt_)
        val = 0.
        for _ in range(iterations):
            y = y/torch.linalg.vector_norm(y)
            z = torch.zeros(*mvec.shape, dtype=dt_, device=dev, requires_grad=True)
            if kind == 'model':
                E_ = model.energy(th[:, :7], x0+z[:, :3], quat_mul(quat_small(z[:, 3:]), q0))
            else:
                E_ = ct.energy(th, x0+z, smooth_polar=True)[0]
            gz, = torch.autograd.grad(E_, z, create_graph=True)
            Hy, = torch.autograd.grad((gz*(y/mvec.sqrt())).sum(), z)
            zz = Hy/mvec.sqrt(); val = float((y*zz).sum()); y = zz.detach()
        out.append(np.sqrt(max(val, 0.)))
    return out[1]/out[0], out


def slaved_rot_invmass(ct, mass):
    """Per body the largest eigenvalue of sum_c J_Bc M_c^-1 J_Bc^T, J_Bc = d theta_B / d x_c of the slaved rotation
    (theta_B = axial skew of F_B - I at rest, linear in x): the inverse rotational inertia a torque on B sees through
    the wall gradient (the PFC contact step bound, as SurfaceCompletion.slaved_rot_invmass for the tetrahedra)."""
    n = ct.n; X = ct.X
    def theta(xf):
        F = ct.F(xf.view(n, 3))
        W = .5*(F-F.transpose(1, 2))
        return torch.stack((W[:, 2, 1], W[:, 0, 2], W[:, 1, 0]), 1).reshape(-1)
    Jm = torch.func.jacrev(theta, chunk_size=256)(X.reshape(-1))   # (3n, 3n), linear map; 256 rows per pass (all at once: > 10 GB)
    Jm = Jm/torch.sqrt(mass.to(Jm.dtype)).repeat_interleave(3)[None, :]
    out = torch.zeros(n, dtype=Jm.dtype, device=Jm.device)
    for s in range(0, n, 256):
        rows = Jm[3*s:3*min(n, s+256)].view(-1, 3, 3*n)
        Bk = rows@rows.transpose(1, 2)
        out[s:s+256] = torch.linalg.eigvalsh(Bk)[:, -1]
    return out


def surface_weights(ct, sp, chunk=256):
    """Surface contact points carried by the wall-gradient affine field, x_s = x_B + F_B (X_s - X_B) = sum_c a_c x_c
    (F_B is linear in the positions and acts component-wise with scalar weights, so the a_c are the same for x, y, z).
    sp: SurfaceCompletion.surface_points() (sp_X rest positions, sp_owner bodies). Returns the point -> (body, a) CSR and
    its transpose with the keys of SurfaceCompletion.np_data (sp_*, bs_*), replacing the tetrahedra's weights."""
    n = ct.n; dev, dt = ct.X.device, ct.X.dtype
    Xs = torch.as_tensor(sp['sp_X'], dtype=dt, device=dev); own = torch.as_tensor(sp['sp_owner'], device=dev)
    lever = Xs-ct.X[own]
    def phi(y):                                      # one coordinate of every body -> that coordinate of every point
        x = torch.stack((y, torch.zeros_like(y), torch.zeros_like(y)), 1)
        F = ct.F(x)
        return y[own]+torch.einsum('pj,pj->p', F[own][:, 0, :], lever)
    eye = torch.eye(n, dtype=dt, device=dev)
    cols = []
    for s in range(0, n, chunk):
        cols.append(torch.func.vmap(lambda e_: torch.func.jvp(phi, (torch.zeros(n, dtype=dt, device=dev),), (e_,))[1])(eye[s:s+chunk]))
    W = torch.cat(cols, 0).T                                                         # (points, bodies)
    W = torch.where(W.abs() > 1e-12, W, torch.zeros_like(W))
    pi_, bi_ = torch.nonzero(W, as_tuple=True)
    a = W[pi_, bi_]
    sum_err = float((W.sum(1)-1).abs().max()); pos_err = float((W@ct.X-Xs).abs().max())
    pi_, bi_, a = pi_.cpu().numpy(), bi_.cpu().numpy(), a.cpu().numpy()
    npt = len(Xs)
    sp_off = np.concatenate(([0], np.cumsum(np.bincount(pi_, minlength=npt))))
    o_ = np.argsort(bi_, kind='stable')
    bs_off = np.concatenate(([0], np.cumsum(np.bincount(bi_, minlength=n))))
    out = dict(sp_off=sp_off, sp_body=bi_, sp_a=a, sp_owner=np.asarray(sp['sp_owner']), bs_off=bs_off, bs_point=pi_[o_], bs_a=a[o_])
    return out, dict(points=npt, bodies_per_point=len(a)/npt, sum_minus_one=sum_err, rest_position_error=pos_err)


def consistent_mass_data(ct, spacing=.004):
    """Consistent mass of the cell's own affine motion x(X) = x_B + F_B (X - X_B) (KB_DERIVATION.md 21.3):
    T = 1/2 sum_B [m |v_B|^2 + 2 m v_B . dF_B d_B + tr(dF_B J_B dF_B^T)], dF_B = F(v) (the wall gradient of the velocities).
    The cells are sampled on a grid (local_energy_map.material_points) for d_B = centroid - X_B and the second moment J_B
    about X_B, scaled to the body masses. M is constant (material coordinates, constant coefficients). Returns the Warp
    data (cm_*): d, J, the diagonal of M per component, and the extreme eigenvalues of diag(M)^-1 M (dense; Chebyshev)."""
    from prototypes.kb_derivation.local_energy_map import material_points
    sc, m = ct.sc, ct.model
    n, X, mass = ct.n, ct.X, m.mass.to(ct.X.dtype)
    dt, dev = X.dtype, X.device
    P, b = material_points(sc, spacing)
    P = torch.as_tensor(P, dtype=dt, device=dev); b = torch.as_tensor(b, device=dev)
    cnt = torch.zeros(n, dtype=dt, device=dev).index_add(0, b, torch.ones(len(b), dtype=dt, device=dev))
    assert bool((cnt > 0).all()), 'consistent mass: a body without sample points (finer spacing)'
    s_ = mass/(cnt*spacing**3)
    r = P-X[b]
    d = torch.zeros(n, 3, dtype=dt, device=dev).index_add(0, b, r)*(spacing**3)*s_[:, None]/mass[:, None]
    J = torch.zeros(n, 3, 3, dtype=dt, device=dev).index_add(0, b, r[:, :, None]*r[:, None, :])*(spacing**3)*s_[:, None, None]

    def Mv(v):                                   # M v = dT/dv
        v = v.view(n, 3)
        def T(u):
            dF = ct.F(u.view(n, 3))
            uu = u.view(n, 3)
            return .5*((mass*(uu*uu).sum(1)).sum()+2*(mass*(uu*torch.einsum('bij,bj->bi', dF, d)).sum(1)).sum()
                       + torch.einsum('bij,bjk,bik->', dF, J, dF))
        return torch.func.grad(T)(v.reshape(-1))
    n3 = 3*n
    eye = torch.eye(n3, dtype=dt, device=dev)
    M = torch.cat([torch.func.vmap(Mv)(eye[s:s+64]) for s in range(0, n3, 64)])
    M = .5*(M+M.T)
    dg = torch.diagonal(M).clone()
    Ds = 1/torch.sqrt(dg)
    ev = torch.linalg.eigvalsh(Ds[:, None]*M*Ds[None, :])
    c = lambda t_: t_.detach().cpu().numpy()
    return dict(cl_cm_d=c(d), cl_cm_J=c(J), cl_cm_diag=c(dg.view(n, 3)), cm_lo=float(ev[0]), cm_hi=float(ev[-1])), M
