"""Energy-based, finite-rotation bonded DEM on Power Foam cells (differentiable, torch).

State per cell: centre x_i, unit quaternion q_i (body -> world, identity at reference),
velocity v_i, world angular velocity w_i. All forces come from one total energy, so rigid
motions are force free and energy is conserved by construction. Forces are closed form, so a
rollout needs only first-order autograd (checkpointed per frame in rollout()).

Bond b between cells i, j (reference normal n0 from i to j, reference levers r_i = c - X_i,
r_j = c - X_j to the shared contact point c):
    D     = (x_j + R_j r_j) - (x_i + R_i r_i)        relative contact displacement (world)
    d     = (R_i^T + R_j^T) D / 2                    mean of the two body-frame views
    Psi   = R_i^T R_j,  phi = vee(Psi - Psi^T) / 2   relative rotation (|phi| = sin angle)
    E_b   = kn/2 (d.n0)^2 + ks/2 |d_t|^2 + kt/2 (phi.n0)^2 + kb/2 |phi_t|^2.
Both measures are frame invariant and linearize to the LinearizedFoam bond kinematics, so
the reference Hessian is the LinearizedFoam stiffness. Forces and torques are closed form
(forces()); forces_autograd() differentiates the same energy for checking.

Volume (Power Foam air cells, option (a)): the reference geometry of every cell, including
cells bordering air, comes from the bounded power diagram (cell volume V_i, face areas A_ij).
The deformation gradient of a material cell is the weighted least-squares fit over its
material neighbours (weights A_ij/d_ij), exact for any affine motion on any cell geometry,
    F_i = [sum_j w (x_j - x_i) (X_j - X_i)^T] [sum_j w (X_j - X_i)(X_j - X_i)^T]^-1,
    E_v = sum_i k_v V_i / 2 (det F_i - 1)^2,
which is frame invariant. On the cube lattice its linearization equals the exact power
volume rows with mirrored air ghosts (power_volume_rows). Air cells do not move.

Gravity and a ground plane are optional: set_environment(gravity, ground_point, ground_normal,
contact_radius, ground_mode). A cell touches when its centre is within contact_radius of the
plane; after the velocity kick a touching cell gets
    'sticky':   v = w = 0 (stays put; PAC-NeRF's MPMSimulator.surface_sticky taken literally),
    'noslip':   tangential v = 0, inward normal v = 0, outward kept, w = 0 (can lift off).
    'layer':    smooth sticky layer, the MPM grid boundary seen through the B-spline kernel:
                all velocity components scaled by (1 - W(h))^(dt/dt_ref) (see step()).
PAC-NeRF zeroes grid velocities only at nodes on/below y = 0, so its objects do bounce off the
ground (simulation_data/elastic/0 leaves the floor after frame 8).

Parameters theta (cells x 5, log multipliers of kn, ks, kb, kt, k_v) follow LinearizedFoam:
bond stiffness = base * exp((theta_i + theta_j)/2), volume k_v * exp(theta_i4).
body_law='neohookean' replaces the volume term by PAC-NeRF's compressible Neo-Hookean energy of
the same least-squares F (6th column: mu multiplier); log J is continued quadratically below
barrier_J so inverted bodies stay finite. Interior cube-lattice bonds with kn = ks = 2 a mu A/d
act like an isotropic material (lam 0, mu a mu), so bodies take mu (1 - a) and lam.
"""
import numpy as np
import torch

from experiments.bonded_dem_linear_gpu import LinearBondsGPU


def quat_mul(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack((aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
                        aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw), -1)


def quat_conj(q):
    return q*q.new_tensor([1., -1., -1., -1.])


def quat_matrix(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), -1).reshape(q.shape[:-1]+(3, 3))


def quat_small(v):
    """Unit quaternion of a (small) rotation vector v; exact derivative at v = 0."""
    q = torch.cat((torch.ones_like(v[..., :1]), .5*v), -1)
    return q/torch.linalg.vector_norm(q, dim=-1, keepdim=True)


def skew(v):
    z = torch.zeros_like(v[..., 0])
    return torch.stack((z, -v[..., 2], v[..., 1], v[..., 2], z, -v[..., 0], -v[..., 1], v[..., 0], z), -1).reshape(v.shape[:-1]+(3, 3))


def quat_rel_principal(qi, qj):
    """q_rel = conj(q_i) q_j with non-negative scalar part (principal representative)."""
    q = quat_mul(quat_conj(qi), qj)
    return torch.where(q[..., :1] < 0, -q, q)


def quat_log(q):
    """Rotation vector of a unit quaternion with q_w >= 0 (angle in [0, pi]); smooth at 0."""
    v, w = q[..., 1:], q[..., 0]
    s = torch.sqrt((v*v).sum(-1)+1e-30)             # smooth at v = 0 (the norm has no second derivative there)
    small = s < 1e-6
    s_safe = torch.where(small, torch.ones_like(s), s)
    ratio = torch.where(small, 2/w.clamp(min=1e-12)*(1-s*s/(3*w*w).clamp(min=1e-12)), 2*torch.atan2(s, w)/s_safe)
    return ratio[..., None]*v


def quat_sqrt(q):
    """Geodesic half rotation of a unit quaternion with q_w >= 0: normalize(1 + q)."""
    h = q+torch.cat((torch.ones_like(q[..., :1]), torch.zeros_like(q[..., 1:])), -1)
    return h/torch.linalg.vector_norm(h, dim=-1, keepdim=True)


def _so3_coeffs(phi):
    t2 = (phi*phi).sum(-1)
    t = torch.sqrt(t2.clamp(min=1e-30))
    small = t < 1e-3
    ts = torch.where(small, torch.ones_like(t), t)
    a = torch.where(small, .5-t2/24, (1-torch.cos(ts))/(ts*ts))                       # J_r: -a [phi]x
    b = torch.where(small, 1/6-t2/120, (ts-torch.sin(ts))/(ts**3))                   # J_r: +b [phi]x^2
    near_pi = (np.pi-t) < 1e-4
    tp = torch.where(small | near_pi, torch.full_like(t, 1.), ts)
    c = torch.where(small, 1/12+t2/720,
                    torch.where(near_pi, 1/(t*t), 1/(tp*tp)-(1+torch.cos(tp))/(2*tp*torch.sin(tp))))  # J_r^-1: +c [phi]x^2
    return a, b, c


def so3_jr(phi):
    a, b, _ = _so3_coeffs(phi)
    K = skew(phi)
    return torch.eye(3, dtype=phi.dtype, device=phi.device)-a[..., None, None]*K+b[..., None, None]*(K@K)


def so3_jr_inv(phi):
    _, _, c = _so3_coeffs(phi)
    K = skew(phi)
    return torch.eye(3, dtype=phi.dtype, device=phi.device)+.5*K+c[..., None, None]*(K@K)


class EnergyFoam:
    def __init__(self, geometry, k_v=400., volume=True, device='cpu', young=1200., shear=480.,
                 body_law='volume', k_mu=0., barrier_J=.05, damping=0., dtype=torch.float64,
                 inertia_scale=1., kinematics='small', residual=False, enhanced=False, thin_ratio=0., thin_bend_factor=1.,
                 one_sided=.15, cosserat=False, bond_level=False, micropolar=True, viscosity=0.):
        assert body_law in ('volume', 'neohookean', 'dilatation', 'nodalvol'), body_law
        # body_law='nodalvol' (SPRING_DILATATION.md section 2): true volumes. Tetrahedra on the body centres (Delaunay, edges
        # <= 1.8 x the local bond length); per body J_a = sum_{t at a} V_t(x) / sum_{t at a} V_t(X) (nodal averaging, no
        # volumetric locking); energy sum_a V0_a lam_a (log J_a)^2 / 2 with V0_a = sum_{t at a} V_t(X) / 4.
        # body_law='dilatation' (SPRING_DILATATION.md): no averaged F; per body a volumetric strain from its bonds' log stretches,
        # theta_a = 3 sum_b om_ab log(|y_ab| / |dX_ab|), om_ab = w L^2 / tr M_a (sum 1), energy V_a lam_a theta_a^2 / 2 (lam_a =
        # k_v exp(theta[:, 4])); the springs carry all of mu (build with alpha = 1). Central forces, no inverse of M.
        assert kinematics in ('small', 'exact'), kinematics
        assert residual in (False, True, 'full', 'normal'), residual
        # residual='normal': only the component of the residual along the current bond direction is penalised
        # (Ganzenmuller 2015 radial hourglass control): zero for affine motion and for the transverse quadratic
        # part of bending, which the full residual would penalise as if it were a loose mode.
        self.residual_mode = 'normal' if residual == 'normal' else 'full'
        # enhanced=True: enhanced assumed strain along each body's free-surface normal (geometry['air_normal'], zero
        # for interior bodies): F~ = F (I + eps e e^T) with eps condensed per body (d psi / d eps = 0, i.e. zero normal
        # stress on the free side), which lets a rigid-body discretisation represent the through-thickness Poisson
        # strain of thin members that the neighbour fit cannot see (BOND_BENDING.md).
        self._enhanced_req, self._air_normal = bool(enhanced), geometry.get('air_normal', None)   # finalised after the volume setup
        # residual=True: non-affine residual energy of each body's least-squares fit,
        # mu_b V_i sum_j w_ij |(x_j - x_i) - F_i dX_ij|^2 / sum_j w_ij |dX_ij|^2 (zero for homogeneous
        # deformation; controls the loosely held / non-affine modes; forces only along r since F_i
        # minimises the same weighted residual).
        self.residual = residual
        self.kinematics = kinematics   # bond strain measures: 'small' (sin-angle, averaged frames) or 'exact' (BOND_EXACT.md)
        self.body_law, self.k_mu, self.barrier_J = body_law, k_mu, barrier_J
        self.damping = damping
        self.dtype = dtype
        self.device = device
        t = lambda a, dtype=torch.float64: torch.as_tensor(np.asarray(a), dtype=dtype, device=device)
        bonds = LinearBondsGPU(geometry['points'], geometry['radii'], geometry['pairs'], geometry['areas'],
                               geometry['contacts'], geometry['masses'], geometry['fixed'], young, shear,
                               device=device)
        self.X = bonds.p
        self.n = len(self.X)
        self.i, self.j = bonds.i, bonds.j
        self.n0, self.ri, self.rj = bonds.n, bonds.ri, bonds.rj
        self.base = torch.stack((bonds.kn, bonds.ks, bonds.kb, bonds.kt), 1)
        # inertia_scale > 1: rotational inertia scaling (standard DEM), for micro-rotations only.
        self.mass, self.inertia = bonds.mass[:, 0], inertia_scale*bonds.mass[:, 3]
        self.fixed = t(geometry['fixed'], torch.bool)
        self.set_environment()
        m_pair = self.mass[self.i]*self.mass[self.j]/(self.mass[self.i]+self.mass[self.j])
        I_pair = self.inertia[self.i]*self.inertia[self.j]/(self.inertia[self.i]+self.inertia[self.j])
        self.c_n = 2*damping*torch.sqrt(self.base[:, 0]*m_pair)
        self.c_t = 2*damping*torch.sqrt(self.base[:, 1]*m_pair)
        self.c_r = 2*damping*torch.sqrt(self.base[:, 2]*I_pair)
        # Explicit dashpots need dt < 2 / rate: per-body upper bounds of the damping rates
        # (translation sum c / m, rotation sum (c |lever|^2 + c_r) / I); see stable_dt().
        c = torch.maximum(self.c_n, self.c_t)
        both = torch.cat((self.i, self.j))
        lever2 = torch.cat(((self.ri**2).sum(-1), (self.rj**2).sum(-1)))
        self.gamma_v = torch.zeros_like(self.mass).index_add(0, both, torch.cat((c, c)))/self.mass
        self.gamma_w = torch.zeros_like(self.mass).index_add(
            0, both, torch.cat((c, c))*lever2+torch.cat((self.c_r, self.c_r)))/self.inertia
        # cosserat=True (COSSERAT_FOAM.md): the body energy is the geometrically exact micropolar law
        #   V [psi_NH(C) + mu_c |skew(R^T F)|^2 + 1/2 (a1 (tr K)^2 + a2 |sym K|^2 + a3 |skew K|^2)],
        # K_i = [sum_j w theta_ij dX^T] M^-1 from the exact relative rotations theta_ij = log(R_i^T R_j); the bond
        # springs are removed (their dashpots keep the reference scale); mu_c = mu; curvature constants from the
        # body's second moment I_z = 0.2 r^2 (a2 = a3 = E I_z, a1 = 2 mu (1 - nu) I_z), so no new parameters.
        self.cosserat = bool(cosserat)
        # bond_level=True (Breitzman & Dayal 2018, JMPS 110, "Bond-level deformation gradients and energy averaging in
        # peridynamics"): the local energy W(F, R) = psi_NH(F) + mu_c |skew(R^T F)|^2 is evaluated per directed neighbour
        # pair a -> b at F_ab = F_a (I - n n^T) + (y_b - y_a) / L n^T (n = dX/L: the pair's own stretch replaces the
        # least-squares average along the pair) and averaged, V_a sum_b om_ab W(F_ab) with om_ab = w_ab L_ab^2 / tr M_a.
        # Homogeneous deformation: F_ab = F, the classical energy. Any motion of one body relative to its neighbours
        # changes some F_ab, so there is no zero-energy mode and no penalty term is needed (residual off); a pair
        # crushed to zero length has det F_ab = 0 (barrier). Requires cosserat=True; enhanced strain is not used.
        self.bond_level = bool(bond_level)
        # micropolar=False (CLEAN_FOAM.md): classical continuum. With cosserat + bond_level: no rotation coupling (mu_c) and no
        # curvature energy; the bodies' rotations are not integrated (orientation = polar(F) where needed). viscosity: kinematic
        # shear viscosity eta / rho (m^2/s) of the objective dissipation Phi = sum_a V_a sum_b om_ab eta |sym(dF_ab F_ab^-1)|^2,
        # scaled per body by exp(theta[:, 6]) (the damping column).
        self.micropolar = bool(micropolar)
        self.k_eta = float(viscosity)
        assert self.micropolar or (self.cosserat and self.bond_level), 'micropolar=False needs cosserat + bond_level'
        assert not self.bond_level or (self.cosserat and not residual and not enhanced), 'bond_level: cosserat, no residual, no EAS'
        if self.cosserat:
            self.base = torch.zeros_like(self.base)
            self.gyration = .2*torch.as_tensor(np.asarray(geometry['radii']), dtype=torch.float64, device=device)**2   # I_z per unit volume
        self.volume = volume
        if volume:
            self.k_v = k_v
            self.cell_volume = t(geometry['masses'])  # unit density: masses are power-cell volumes
            # Least-squares deformation gradient over material neighbours, both directions.
            a = torch.cat((self.i, self.j)); b = torch.cat((self.j, self.i))
            length = torch.linalg.vector_norm(self.X[b]-self.X[a], dim=1)
            w = torch.cat((t(geometry['areas']),)*2)/length
            dX = self.X[b]-self.X[a]
            moment = torch.zeros(self.n, 3, 3, dtype=torch.float64, device=device)
            moment.index_add_(0, a, w[:, None, None]*dX[:, :, None]*dX[:, None, :])
            self.ls_a, self.ls_b, self.ls_w, self.ls_dX = a, b, w, dX
            # Thin (sliver / one-layer) bodies: the neighbour moment matrix is nearly singular along one direction e
            # (eigenvalue ratio < thin_ratio), so the least-squares F is undefined there. Shell closure: fit F only in
            # the tangent plane (pseudo-inverse) and let the thickness column follow the deformed tangent plane,
            # F e = n(F t1, F t2) = normalize(F t1 x F t2) (Kirchhoff-Love), with the thickness stretch from the
            # enhanced strain when enhanced=True. Continuous in the data; exact for stretched / bent sheets.
            evals, evecs = torch.linalg.eigh(moment)
            ratio = evals[:, 0]/evals[:, -1]
            self.thin = ratio < thin_ratio
            self.thin_ratio = thin_ratio
            inv = torch.zeros_like(moment)
            for k in (0, 1, 2):
                term = evecs[:, :, k:k+1]*evecs[:, :, k:k+1].transpose(1, 2)/evals[:, k, None, None].clamp(min=1e-300)
                inv = inv+(torch.where(self.thin[:, None, None], torch.zeros_like(term), term) if k == 0 else term)
            if bool(self.thin.any()):
                self.thin_e = evecs[:, :, 0]                                   # weak direction
                self.thin_t1, self.thin_t2 = evecs[:, :, 1], evecs[:, :, 2]
                flip = (torch.linalg.cross(self.thin_t1, self.thin_t2)*self.thin_e).sum(-1) < 0
                self.thin_t2 = torch.where(flip[:, None], -self.thin_t2, self.thin_t2)   # t1 x t2 = e
            self.moment_inv = inv
            if bool(self.thin.any()) and thin_bend_factor != 1.:
                # bonds at thin bodies: the shell closure gives their Neo-Hookean term no in-cell bending, so the bond
                # must carry the full plate bending stiffness E I / ((1 - nu^2) L) instead of the bond's share 2 alpha mu I / L:
                # factor 1 / (alpha (1 - nu)) per thin end (half each end), BOND_BENDING.md section 3.
                ends = (self.thin[self.i].to(torch.float64)+self.thin[self.j].to(torch.float64))/2
                self.base[:, 2:4] = self.base[:, 2:4]*(1+(thin_bend_factor-1)*ends[:, None])
            self.res_den = torch.zeros(self.n, dtype=torch.float64, device=device).index_add_(0, a, w*(dX*dX).sum(-1))
        self.enhanced = self._enhanced_req and volume and ((self._air_normal is not None) or one_sided > 0 or thin_ratio > 0)
        if self.enhanced:
            if self._air_normal is not None:
                e = torch.as_tensor(np.asarray(self._air_normal), dtype=torch.float64, device=device)
            elif volume and one_sided > 0:
                # free-surface normal from the neighbour distribution: bodies whose weighted first moment is a
                # sizeable fraction of the total (neighbours on one side only) are surface bodies, normal = -m1
                m1 = torch.zeros(self.n, 3, dtype=torch.float64, device=device).index_add_(0, self.ls_a, self.ls_w[:, None]*self.ls_dX)
                tot = torch.zeros(self.n, dtype=torch.float64, device=device).index_add_(0, self.ls_a, self.ls_w*torch.linalg.vector_norm(self.ls_dX, dim=-1))
                frac = torch.linalg.vector_norm(m1, dim=-1)/tot.clamp(min=1e-30)
                e = torch.where((frac > one_sided)[:, None], -m1, torch.zeros_like(m1))
            else:
                e = torch.zeros(self.n, 3, dtype=torch.float64, device=device)
            on = torch.linalg.vector_norm(e, dim=-1) > 1e-9
            e = torch.where(on[:, None], e/torch.linalg.vector_norm(e, dim=-1, keepdim=True).clamp(min=1e-12), torch.zeros_like(e))
            if volume and thin_ratio > 0 and bool(self.thin.any()):   # thin bodies: enhance along the weak direction
                e = torch.where(self.thin[:, None], self.thin_e, e); on = on | self.thin
            self.eas_on, self.eas_dir = on, e
        else:
            self.eas_on = torch.zeros(self.n, dtype=torch.bool, device=device)
            self.eas_dir = torch.zeros(self.n, 3, dtype=torch.float64, device=device)
        # Geometry is prepared in float64, then everything is cast to the working dtype.
        for name, value in list(vars(self).items()):
            if torch.is_tensor(value) and value.is_floating_point():
                setattr(self, name, value.to(dtype))

    def set_environment(self, gravity=None, ground_point=None, ground_normal=None, contact_radius=None,
                        ground_mode='sticky', layer_dx=.01, layer_dt_ref=1/4800):
        assert ground_mode in ('sticky', 'noslip', 'layer'), ground_mode
        self.ground_mode, self.layer_dx, self.layer_dt_ref = ground_mode, layer_dx, layer_dt_ref
        t = lambda a: None if a is None else torch.as_tensor(np.asarray(a), dtype=self.dtype, device=self.device)
        self.gravity = t(gravity)
        self.ground_point, self.ground_normal = t(ground_point), t(ground_normal)
        if self.ground_normal is not None:
            self.ground_normal = self.ground_normal/torch.linalg.vector_norm(self.ground_normal)
        self.contact_radius = t(contact_radius)

    def initial_state(self):
        z = torch.zeros_like(self.X)
        q = torch.zeros(self.n, 4, dtype=self.dtype, device=self.device); q[:, 0] = 1
        return self.X.clone(), q, z.clone(), z.clone()

    def deformation_gradient(self, x):
        dx = x[self.ls_b]-x[self.ls_a]
        S = torch.zeros(self.n, 3, 3, dtype=x.dtype, device=x.device)
        S = S.index_add(0, self.ls_a, self.ls_w[:, None, None]*dx[:, :, None]*self.ls_dX[:, None, :])
        F = S@self.moment_inv
        if getattr(self, 'thin', None) is not None and bool(self.thin.any()):
            e, t1, t2 = self.thin_e, self.thin_t1, self.thin_t2
            nrm = torch.linalg.cross((F@t1[..., None])[..., 0], (F@t2[..., None])[..., 0])
            nrm = nrm/torch.linalg.vector_norm(nrm, dim=-1, keepdim=True).clamp(min=1e-30)
            F = torch.where(self.thin[:, None, None], F+nrm[:, :, None]*e[:, None, :], F)
        return F

    def _bond_kinematics(self, theta, x, R):
        Ri, Rj = R[self.i], R[self.j]
        pi, pj = (Ri@self.ri[..., None])[..., 0], (Rj@self.rj[..., None])[..., 0]
        D = x[self.j]+pj-x[self.i]-pi
        d = .5*((Ri.transpose(1, 2)@D[..., None])[..., 0]+(Rj.transpose(1, 2)@D[..., None])[..., 0])
        Psi = Ri.transpose(1, 2)@Rj
        A = Psi-Psi.transpose(1, 2)
        phi = .5*torch.stack((A[:, 2, 1], A[:, 0, 2], A[:, 1, 0]), -1)
        k = self.base*torch.exp(.5*(theta[self.i, :4]+theta[self.j, :4]))
        dn, pn = (d*self.n0).sum(-1), (phi*self.n0).sum(-1)
        dt, pt = d-dn[:, None]*self.n0, phi-pn[:, None]*self.n0
        return Ri, Rj, pi, pj, D, Psi, k, dn, dt, pn, pt

    def _bond_exact(self, theta, x, q):
        """Geometrically exact bond strains (BOND_EXACT.md): relative rotation vector theta =
        log(R_i^T R_j), mid frame R_m = R_i exp(theta/2), stretch d = R_m^T D."""
        qi, qj = q[self.i], q[self.j]
        Ri, Rj = quat_matrix(qi), quat_matrix(qj)
        pi, pj = (Ri@self.ri[..., None])[..., 0], (Rj@self.rj[..., None])[..., 0]
        D = x[self.j]+pj-x[self.i]-pi
        qr = quat_rel_principal(qi, qj)
        th = quat_log(qr)
        Rm = quat_matrix(quat_mul(qi, quat_sqrt(qr)))
        d = (Rm.transpose(1, 2)@D[..., None])[..., 0]
        k = self.base*torch.exp(.5*(theta[self.i, :4]+theta[self.j, :4]))
        n = self.n0
        dn, tn = (d*n).sum(-1), (th*n).sum(-1)
        dt, tt = d-dn[:, None]*n, th-tn[:, None]*n
        return Ri, Rj, Rm, pi, pj, D, th, d, k, dn, dt, tn, tt

    def _volume_kinematics(self, theta, x):
        F = self.deformation_gradient(x)
        c0, c1, c2 = F[:, :, 0], F[:, :, 1], F[:, :, 2]
        cof = torch.stack((torch.linalg.cross(c1, c2), torch.linalg.cross(c2, c0), torch.linalg.cross(c0, c1)), -1)
        J = (c0*cof[:, :, 0]).sum(-1)
        kv = self.k_v*torch.exp(theta[:, 4])*self.cell_volume
        return J, cof, kv, F

    def _log_J(self, J):
        """log J, continued quadratically below barrier_J (finite under inversion); and d/dJ."""
        m = self.barrier_J
        below = J < m
        Jc = torch.where(below, torch.full_like(J, m), J)
        L = torch.where(below, np.log(m)+(J-m)/m-(J-m)**2/(2*m*m), torch.log(Jc))
        dL = torch.where(below, 1/m-(J-m)/(m*m), 1/Jc)
        return L, dL

    def _nh(self, F, mu, kv):
        c0, c1, c2 = F[:, :, 0], F[:, :, 1], F[:, :, 2]
        cof = torch.stack((torch.linalg.cross(c1, c2), torch.linalg.cross(c2, c0), torch.linalg.cross(c0, c1)), -1)
        J = (c0*cof[:, :, 0]).sum(-1)
        L, dL = self._log_J(J)
        energy = .5*mu*((F*F).sum((1, 2))-3)-mu*L+.5*kv*L*L
        stress = mu[:, None, None]*F+((kv*L-mu)*dL)[:, None, None]*cof
        return energy, stress

    def _enhance(self, F, mu, kv, iters=20):
        """Thickness stretch 1 + eps = exp(s) per enhanced body, from d psi(F (I + eps e e^T)) / d s = 0.
        With a = |F e|^2 and J~ = J exp(s): psi(s) = mu/2 (|F|^2 - 3 + (e^{2s} - 1) a) - mu L(J~) + kv/2 L(J~)^2,
        strictly convex in s (log-stretch), so Newton from s = 0 with a step limit converges; eps = 0 where not enhanced."""
        e = self.eas_dir
        a = ((F@e[..., None])[..., 0]**2).sum(-1)
        c0, c1, c2 = F[:, :, 0], F[:, :, 1], F[:, :, 2]
        J = (c0*torch.linalg.cross(c1, c2)).sum(-1)
        m = self.barrier_J
        sv = torch.zeros(self.n, dtype=F.dtype, device=F.device)
        on = self.eas_on & (J > 0)                                   # inverted fits are not enhanced
        for _ in range(iters):
            Jt = J*torch.exp(sv)
            below = Jt < m
            Jc = torch.where(below, torch.full_like(Jt, m), Jt)
            L = torch.where(below, np.log(m)+(Jt-m)/m-(Jt-m)**2/(2*m*m), torch.log(Jc))
            dL = torch.where(below, 1/m-(Jt-m)/(m*m), 1/Jc)
            d2L = torch.where(below, torch.full_like(Jt, -1/(m*m)), -1/(Jc*Jc))
            g = mu*a*torch.exp(2*sv)+(kv*L-mu)*dL*Jt
            h = 2*mu*a*torch.exp(2*sv)+kv*(dL*Jt)**2+(kv*L-mu)*(d2L*Jt*Jt+dL*Jt)
            step = torch.where(on, g/h.clamp(min=1e-30), torch.zeros_like(g)).clamp(-1., 1.)
            sv = (sv-step).clamp(-3., 3.)
        return torch.exp(sv)-1

    def _body_energy_and_stress(self, theta, x):
        """Per-body energy and dE/dF. 'volume': k_v V (J-1)^2 / 2. 'neohookean' (PAC-NeRF's law):
        V [mu/2 (|F|^2 - 3) - mu log J + lam/2 (log J)^2], lam = k_v e^theta4, mu = k_mu e^theta5."""
        J, cof, kv, F = self._volume_kinematics(theta, x)
        if self.body_law == 'volume':
            return .5*kv*(J-1)**2, (kv*(J-1))[:, None, None]*cof
        mu = self.k_mu*torch.exp(theta[:, 5])*self.cell_volume
        if getattr(self, 'enhanced', False):
            eps = self._enhance(F, mu, kv)
            A = torch.eye(3, dtype=F.dtype, device=F.device)+eps[:, None, None]*(self.eas_dir[:, :, None]*self.eas_dir[:, None, :])
            energy, P = self._nh(F@A, mu, kv)
            return energy, P@A                      # envelope: d psi / d eps = 0, A symmetric
        return self._nh(F, mu, kv)

    def _residual_energy(self, theta, x):
        c, r = self._residual(theta, x)
        if self.residual_mode == 'normal':
            d = x[self.ls_b]-x[self.ls_a]
            d = d/torch.linalg.vector_norm(d, dim=-1, keepdim=True)
            return (c*(r*d).sum(-1)**2).sum()
        return (c*(r*r).sum(-1)).sum()

    def energy(self, theta, x, q):
        if self.kinematics == 'exact':
            *_, k, dn, dt, pn, pt = self._bond_exact(theta, x, q)
        else:
            _, _, _, _, _, _, k, dn, dt, pn, pt = self._bond_kinematics(theta, x, quat_matrix(q))
        if getattr(self, 'nonlinear_bonds', False):      # Ogden (alpha = 2: neo-Hookean) normal bond energy (foam_dem_warp.nh_bond_force)
            al = float(getattr(self, 'ogden_alpha', 2.))
            L = self.ls_dX[:len(self.i)].norm(dim=-1)
            lam = 1+dn/L; m = .2
            c = 2*k[:, 0]/(3*al)
            U_ = lambda l_: c*L**2/al*(l_**al+2*l_**(-al/2)-3)
            Um = U_(torch.full_like(lam, m)); fm = c*L*(m**(al-1)-m**(-al/2-1)); dfm = c*((al-1)*m**(al-2)+(al/2+1)*m**(-al/2-2))
            u = L*(lam-m)
            En = torch.where(lam < m, Um+fm*u+.5*dfm*u**2, U_(lam.clamp(min=m)))
        else:
            En = .5*k[:, 0]*dn**2
        energy = (En+.5*(k[:, 1]*(dt*dt).sum(-1)+k[:, 2]*(pt*pt).sum(-1)+k[:, 3]*pn**2)).sum()
        if self.volume and getattr(self, 'bond_level', False):
            energy = energy+self._bond_level_energy(theta, x, q)
        elif self.volume and self.body_law == 'dilatation':
            energy = energy+self._dilatation(theta, x)[0]
        elif self.volume and self.body_law == 'nodalvol':
            energy = energy+self._nodalvol(theta, x)[0]
        elif self.volume:
            energy = energy+self._body_energy_and_stress(theta, x)[0].sum()
        if getattr(self, 'cosserat', False) and getattr(self, 'micropolar', True):
            energy = energy+self._cosserat_energy(theta, x, q, skew=not getattr(self, 'bond_level', False)).sum()
        if self.residual:
            energy = energy+self._residual_energy(theta, x)
        return energy

    def curvature(self, q):
        """K_i = [sum_j w theta_ij dX_ij^T] M_i^-1 (thin bodies: tangent pseudo-inverse), theta_ij = log(R_i^T R_j)
        in the frame of i; the reverse pair contributes the same term (theta_ji = -theta_ij, dX_ji = -dX_ij)."""
        th = quat_log(quat_rel_principal(q[self.i], q[self.j]))                         # (nb, 3)
        nb = len(self.i)
        th_dir = torch.cat((th, -th), 0)                                                 # directed pairs ls_a -> ls_b
        Kn = torch.zeros(self.n, 3, 3, dtype=q.dtype, device=q.device)
        Kn = Kn.index_add(0, self.ls_a, self.ls_w[:, None, None]*th_dir[:, :, None]*self.ls_dX[:, None, :])
        return Kn@self.moment_inv

    def _setup_tets(self, edge_factor=1.8):
        """Delaunay tetrahedra of the body centres, kept when every edge <= edge_factor x the larger of its two ends' median
        bond length (removes tetrahedra bridging gaps or concavities), oriented positively."""
        from scipy.spatial import Delaunay
        X = self.X.detach().cpu().double().numpy()
        i_, j_ = self.i.cpu().numpy(), self.j.cpu().numpy()
        bl = np.linalg.norm(X[j_]-X[i_], axis=1)
        med = np.zeros(self.n)
        for b_ in range(self.n):
            pass
        per = [[] for _ in range(self.n)]
        for e_, (a_, c_) in enumerate(zip(i_, j_)):
            per[a_].append(bl[e_]); per[c_].append(bl[e_])
        med = np.array([np.median(p_) if p_ else 0. for p_ in per])
        try:
            T = Delaunay(X).simplices
        except Exception:
            try:
                T = Delaunay(X, qhull_options='QJ').simplices
            except Exception:                                             # flat (one-layer) set: no volume
                T = np.zeros((0, 4), np.int64)
        keep = np.ones(len(T), bool)
        for u_ in range(4):
            for v_ in range(u_+1, 4):
                a_, c_ = T[:, u_], T[:, v_]
                keep &= np.linalg.norm(X[a_]-X[c_], axis=1) <= edge_factor*np.maximum(med[a_], med[c_])
        P = X[T]
        vol = np.einsum('ij,ij->i', np.cross(P[:, 1]-P[:, 0], P[:, 2]-P[:, 0]), P[:, 3]-P[:, 0])/6
        keep &= np.abs(vol) > 1e-9*np.abs(vol).max(initial=0.)               # flat slivers (co-spherical sites)
        T, vol = T[keep], vol[keep]
        flip = vol < 0
        T[flip, 2], T[flip, 3] = T[flip, 3].copy(), T[flip, 2].copy()
        self.tets = torch.as_tensor(T, dtype=torch.long, device=self.X.device)
        V = torch.as_tensor(np.abs(vol), dtype=self.X.dtype, device=self.X.device)
        self.tet_V0 = V
        self.node_V0 = torch.zeros(self.n, dtype=self.X.dtype, device=self.X.device).index_add(0, self.tets.reshape(-1), V.repeat_interleave(4))
        self.node_has = self.node_V0 > 1e-12*float(V.max() if len(V) else 1.)

    def _tet_volumes(self, x):
        P = x[self.tets]
        return (torch.linalg.cross(P[:, 1]-P[:, 0], P[:, 2]-P[:, 0])*(P[:, 3]-P[:, 0])).sum(-1)/6, P

    def _nodalvol(self, theta, x):
        if getattr(self, 'tets', None) is None:
            self._setup_tets()
        Vt, P = self._tet_volumes(x)
        Vn = torch.zeros(self.n, dtype=x.dtype, device=x.device).index_add(0, self.tets.reshape(-1), Vt.repeat_interleave(4))
        J = torch.where(self.node_has, Vn/self.node_V0.clamp(min=1e-300), torch.ones_like(Vn))
        lam = self.k_v*torch.exp(theta[:, 4])
        logJ, dlogJ = self._log_J(J)                  # log J, continued quadratically below barrier_J (finite when inverted)
        e = torch.where(self.node_has, .25*self.node_V0*.5*lam*logJ**2, torch.zeros_like(J))   # V0_a = node_V0 / 4
        return e.sum(), J, lam, Vt, P

    def _nodalvol_forces(self, theta, x, f):
        """dE/dx_k = sum_t p_t dV_t/dx_k, p_t = sum_{a in t} lam_a log J_a / (4 J_a) (the 1/4 of V0_a cancels node_V0)."""
        _, J, lam, Vt, P = self._nodalvol(theta, x)
        logJ, dlogJ = self._log_J(J)
        pa = torch.where(self.node_has, .25*lam*logJ*dlogJ, torch.zeros_like(J))
        pt = pa[self.tets].sum(1)
        cr = torch.linalg.cross
        d1, d2, d3 = P[:, 1]-P[:, 0], P[:, 2]-P[:, 0], P[:, 3]-P[:, 0]
        g1, g2, g3 = cr(d2, d3)/6, cr(d3, d1)/6, cr(d1, d2)/6
        g0 = -(g1+g2+g3)
        G = torch.stack((g0, g1, g2, g3), 1)*pt[:, None, None]                 # dE/dx per tet corner
        return f.index_add(0, self.tets.reshape(-1), -G.reshape(-1, 3))

    def _dilatation(self, theta, x):
        """Energy, per-body volumetric strain theta_a, and per-pair (y, |y|, om) of the dilatation body law."""
        a, b = self.ls_a, self.ls_b
        y = x[b]-x[a]
        ly = torch.linalg.vector_norm(y, dim=-1)
        L0 = torch.linalg.vector_norm(self.ls_dX, dim=-1)
        wl2 = self.ls_w*L0**2
        trM = torch.zeros(self.n, dtype=x.dtype, device=x.device).index_add(0, a, wl2)
        om = wl2/trM[a]
        th_v = 3*torch.zeros(self.n, dtype=x.dtype, device=x.device).index_add(0, a, om*torch.log(ly/L0))
        lam = self.k_v*torch.exp(theta[:, 4])*self.cell_volume
        return (.5*lam*th_v**2).sum(), th_v, y, ly, om, lam

    def _dilatation_forces(self, theta, x, f):
        """dE/dy_ab = lam_a theta_a 3 om_ab y / |y|^2 (central, along the current pair direction)."""
        _, th_v, y, ly, om, lam = self._dilatation(theta, x)
        g = ((lam*th_v)[self.ls_a]*3*om/ly**2)[:, None]*y              # dE/dy_b (direct pair)
        return f.index_add(0, self.ls_a, g).index_add(0, self.ls_b, -g)

    def _bond_level_kinematics(self, theta, x):
        """Per directed pair a -> b: F_ab = F_a (I - n n^T) + (y_b - y_a)/L n^T, the pair weight w L^2 / tr M_a, mu_a, kv_a."""
        F = self.deformation_gradient(x)
        a, b = self.ls_a, self.ls_b
        L = torch.linalg.vector_norm(self.ls_dX, dim=-1, keepdim=True)
        nrm = self.ls_dX/L
        Fa = F[a]
        if getattr(self, 'bond_level_mid', False):
            # chord compared with the two-end (trapezoid) gradient: F_ab = F_a + (y_ab - (F_a + F_b)/2 dX)/L n^T
            Fab = Fa+((x[b]-x[a]-.5*((Fa+F[b])@self.ls_dX[..., None])[..., 0])/L)[:, :, None]*nrm[:, None, :]
        else:
            Fab = Fa-(Fa@nrm[..., None])*nrm[:, None, :]+((x[b]-x[a])/L)[:, :, None]*nrm[:, None, :]
        # pair weight w L^2 / tr M_a: then sum_b om n n^T = M_a / tr M_a and under a homogeneous stress the nodal
        # forces equal those of the averaged-F law exactly (same patch-test behaviour); for Power cells (w = A/L)
        # om is the pyramid (face area x distance) fraction of the cell.
        wl2 = self.ls_w*(L[:, 0]**2)
        trM = torch.zeros(self.n, dtype=x.dtype, device=x.device).index_add(0, a, wl2)
        om = wl2/trM[a]
        mu = (self.k_mu*torch.exp(theta[:, 5])*self.cell_volume)[a]
        kv = (self.k_v*torch.exp(theta[:, 4])*self.cell_volume)[a]
        return Fab, nrm, L, om, mu, kv

    def _bond_level_energy(self, theta, x, q):
        Fab, nrm, L, om, mu, kv = self._bond_level_kinematics(theta, x)
        e_nh, _ = self._nh(Fab, mu, kv)
        if not getattr(self, 'micropolar', True):
            return (om*e_nh).sum()
        R = quat_matrix(q)[self.ls_a]
        U = R.transpose(1, 2)@Fab
        Sk = .5*(U-U.transpose(1, 2))
        return (om*(e_nh+mu*(Sk*Sk).sum((1, 2)))).sum()

    def _bond_level_forces(self, theta, x, q, f, tau):
        """Closed form: P_ab = om [P_NH(F_ab) + R_a (2 mu S_ab)], S_ab = skew(R_a^T F_ab); dE/dF_a += P_ab (I - n n^T)
        (pulled back through the fit and the thin closure), dE/dy_b = -dE/dy_a = P_ab n / L, torque on a
        -2 mu om vee(B - B^T), B = F_ab S_ab^T R_a^T."""
        Fab, nrm, L, om, mu, kv = self._bond_level_kinematics(theta, x)
        _, Pnh = self._nh(Fab, mu, kv)
        micro = getattr(self, 'micropolar', True)
        if micro:
            R = quat_matrix(q)[self.ls_a]
            U = R.transpose(1, 2)@Fab
            Sk = .5*(U-U.transpose(1, 2))
            P = om[:, None, None]*(Pnh+R@(2*mu[:, None, None]*Sk))
        else:
            P = om[:, None, None]*Pnh
        return self._bond_level_pullback(P, nrm, L, x, f, tau if not micro else self._skew_torques(om, mu, Fab, Sk, R, tau))

    def _skew_torques(self, om, mu, Fab, Sk, R, tau):
        B = Fab@Sk.transpose(1, 2)@R.transpose(1, 2)
        Bs = B-B.transpose(1, 2)
        t_ = -2*(om*mu)[:, None]*torch.stack((Bs[:, 2, 1], Bs[:, 0, 2], Bs[:, 1, 0]), -1)
        return tau.index_add(0, self.ls_a, t_)

    def _bond_level_pullback(self, P, nrm, L, x, f, tau):
        """Forces from per-pair stresses P_ab = dW/dF_ab (or dPhi/d dF_ab): through F_ab (two-end or one-end), the
        least-squares fit, the thin closure, and the pair's own chord."""
        Pn = (P@nrm[..., None])[..., 0]
        Pnn = Pn[:, :, None]*nrm[:, None, :]
        if getattr(self, 'bond_level_mid', False):     # dE/dF_a += P - Pnn/2, dE/dF_b += -Pnn/2 (r = y - (F_a + F_b) dX / 2)
            GF = torch.zeros(self.n, 3, 3, dtype=x.dtype, device=x.device).index_add(0, self.ls_a, P-.5*Pnn).index_add(0, self.ls_b, -.5*Pnn)
        else:
            GF = torch.zeros(self.n, 3, 3, dtype=x.dtype, device=x.device).index_add(0, self.ls_a, P-Pnn)
        G = self._thin_pullback(GF, x)@self.moment_inv
        g = self.ls_w[:, None]*(G[self.ls_a]@self.ls_dX[..., None])[..., 0]+Pn/L     # dE/dy_b (fit + direct)
        f = f.index_add(0, self.ls_a, g).index_add(0, self.ls_b, -g)
        return f, tau

    def deformation_rate(self, x, v):
        """dF/dt of deformation_gradient(x) along velocities v (thin closure: n' = (I - n n^T) c' / |c|)."""
        dv = v[self.ls_b]-v[self.ls_a]
        Sv = torch.zeros(self.n, 3, 3, dtype=x.dtype, device=x.device).index_add(0, self.ls_a, self.ls_w[:, None, None]*dv[:, :, None]*self.ls_dX[:, None, :])
        Fd = Sv@self.moment_inv
        if getattr(self, 'thin', None) is not None and bool(self.thin.any()):
            dx = x[self.ls_b]-x[self.ls_a]
            S = torch.zeros(self.n, 3, 3, dtype=x.dtype, device=x.device).index_add(0, self.ls_a, self.ls_w[:, None, None]*dx[:, :, None]*self.ls_dX[:, None, :])
            Ft = S@self.moment_inv
            t1, t2, e = self.thin_t1, self.thin_t2, self.thin_e
            a_, b_ = (Ft@t1[..., None])[..., 0], (Ft@t2[..., None])[..., 0]
            ad, bd = (Fd@t1[..., None])[..., 0], (Fd@t2[..., None])[..., 0]
            c = torch.linalg.cross(a_, b_); cn = torch.linalg.vector_norm(c, dim=-1, keepdim=True).clamp(min=1e-30); nrm = c/cn
            cd = torch.linalg.cross(ad, b_)+torch.linalg.cross(a_, bd)
            nd = (cd-(cd*nrm).sum(-1, keepdim=True)*nrm)/cn
            Fd = torch.where(self.thin[:, None, None], Fd+nd[:, :, None]*e[:, None, :], Fd)
        return Fd

    def _viscous_pair(self, theta, x, v):
        """Per directed pair: D_ab = sym(dF_ab F_ab^-1), dF_ab built from velocities like F_ab from positions;
        Phi_ab = om V_a eta_a |D_ab|^2 and P_v = dPhi_ab / d dF_ab = 2 om V eta D G^T (G = F_ab^-1)."""
        Fab, nrm, L, om, mu, kv = self._bond_level_kinematics(theta, x)
        Fd = self.deformation_rate(x, v)
        a, b = self.ls_a, self.ls_b
        if getattr(self, 'bond_level_mid', False):
            Fdab = Fd[a]+((v[b]-v[a]-.5*((Fd[a]+Fd[b])@self.ls_dX[..., None])[..., 0])/L)[:, :, None]*nrm[:, None, :]
        else:
            Fdab = Fd[a]-(Fd[a]@nrm[..., None])*nrm[:, None, :]+((v[b]-v[a])/L)[:, :, None]*nrm[:, None, :]
        Gi = torch.linalg.inv(Fab)
        Lv = Fdab@Gi
        D = .5*(Lv+Lv.transpose(1, 2))
        eta = (self.k_eta*torch.exp(theta[:, 6])*self.cell_volume)[a] if theta.shape[1] > 6 else self.k_eta*self.cell_volume[a]
        return D, Gi, om, eta, nrm, L

    def dissipation(self, theta, x, v):
        D, Gi, om, eta, nrm, L = self._viscous_pair(theta, x, v)
        return (om*eta*(D*D).sum((1, 2))).sum()

    def viscous_forces(self, theta, x, v):
        """-dPhi/dv (closed form, same pullback as the elastic stress)."""
        D, Gi, om, eta, nrm, L = self._viscous_pair(theta, x, v)
        Pv = (2*om*eta)[:, None, None]*(D@Gi.transpose(1, 2))
        f, _ = self._bond_level_pullback(Pv, nrm, L, x, torch.zeros_like(x), torch.zeros_like(x))
        return f

    def max_damping_rate(self, theta, iterations=60, seed=0):
        """Largest viscous decay rate at the reference (power iteration on M^-1 C, C v = -viscous_forces(v))."""
        x0 = self.X
        g = torch.Generator(device='cpu').manual_seed(seed)
        y = torch.randn(self.n, 3, generator=g, dtype=torch.float64).to(self.device, self.dtype)
        value = torch.tensor(0.)
        for _ in range(iterations):
            y = y/torch.linalg.vector_norm(y)
            z = -self.viscous_forces(theta, x0, y/self.mass[:, None].sqrt())/self.mass[:, None].sqrt()
            value = (y*z).sum()
            y = z
        return float(value.clamp(min=0))

    def _cosserat_energy(self, theta, x, q, skew=True):
        """Per body: V [mu_c |skew(R^T F)|^2 + 1/2 (a1 (tr K)^2 + a2 |sym K|^2 + a3 |skew K|^2)] with mu_c = mu,
        a2 = a3 = E I_z, a1 = 2 mu (1 - nu) I_z (E, nu from the body's current lam, mu)."""
        F = self.deformation_gradient(x)
        R = quat_matrix(q)
        U = R.transpose(1, 2)@F
        skewU = .5*(U-U.transpose(1, 2))
        mu = self.k_mu*torch.exp(theta[:, 5])*self.cell_volume
        lam = self.k_v*torch.exp(theta[:, 4])*self.cell_volume
        E = mu*(3*lam+2*mu)/(lam+mu)
        nu = lam/(2*(lam+mu))
        Iz = self.gyration
        K = self.curvature(q)
        trK = K.diagonal(dim1=-2, dim2=-1).sum(-1)
        a1 = 2*mu*(1-nu)*Iz
        return (mu*(skewU*skewU).sum((1, 2)) if skew else 0.)+.5*(a1*trK*trK+E*Iz*(K*K).sum((1, 2)))

    def _residual(self, theta, x):
        """Per directed neighbour pair (ls_a -> ls_b): coefficient mu_b V w / den and residual r."""
        F = self.deformation_gradient(x)
        a, b = self.ls_a, self.ls_b
        r = (x[b]-x[a])-(F[a]@self.ls_dX[..., None])[..., 0]
        mu = self.k_mu*torch.exp(theta[:, 5])*self.cell_volume
        c = (mu/self.res_den)[a]*self.ls_w
        return c, r

    def _thin_pullback(self, P, x):
        """dE/dF_t for thin bodies given dE/dF = P, F = F_t + n(F_t) e^T with n = normalize(F_t t1 x F_t t2)."""
        if getattr(self, 'thin', None) is None or not bool(self.thin.any()):
            return P
        dx = x[self.ls_b]-x[self.ls_a]
        S = torch.zeros(self.n, 3, 3, dtype=x.dtype, device=x.device)
        S = S.index_add(0, self.ls_a, self.ls_w[:, None, None]*dx[:, :, None]*self.ls_dX[:, None, :])
        Ft = S@self.moment_inv
        a_, b_ = (Ft@self.thin_t1[..., None])[..., 0], (Ft@self.thin_t2[..., None])[..., 0]
        c = torch.linalg.cross(a_, b_); cn = torch.linalg.vector_norm(c, dim=-1, keepdim=True).clamp(min=1e-30); nrm = c/cn
        Pe = (P@self.thin_e[..., None])[..., 0]                      # dE/dn
        g = (Pe-(Pe*nrm).sum(-1, keepdim=True)*nrm)/cn               # dE/dc
        dA, dB = torch.linalg.cross(b_, g), torch.linalg.cross(g, a_)
        extra = dA[:, :, None]*self.thin_t1[:, None, :]+dB[:, :, None]*self.thin_t2[:, None, :]
        return torch.where(self.thin[:, None, None], P+extra, P)

    def _residual_forces(self, theta, x, f):
        if self.residual_mode == 'normal':
            # E = sum c (r . d)^2, r = y - F_a dX, d = y/|y|: direct part on x_a, x_b and the part through F_a
            F = self.deformation_gradient(x)
            a, b = self.ls_a, self.ls_b
            y = x[b]-x[a]; yn = torch.linalg.vector_norm(y, dim=-1, keepdim=True).clamp(min=1e-30); d = y/yn
            r = y-(F[a]@self.ls_dX[..., None])[..., 0]
            mu = self.k_mu*torch.exp(theta[:, 5])*self.cell_volume
            c = (mu/self.res_den)[a]*self.ls_w
            sdot = (r*d).sum(-1)
            v = d+(r-(r*d).sum(-1, keepdim=True)*d)/yn
            g = (2*c*sdot)[:, None]*v                                   # dE/dx_b (direct); dE/dx_a = -g
            Gres = torch.zeros(self.n, 3, 3, dtype=x.dtype, device=x.device)
            Gres = Gres.index_add(0, a, (-2*c*sdot)[:, None, None]*d[:, :, None]*self.ls_dX[:, None, :])   # dE/dF_a
            G = self._thin_pullback(Gres, x)@self.moment_inv
            g2 = self.ls_w[:, None]*(G[a]@self.ls_dX[..., None])[..., 0]
            return f.index_add(0, a, g+g2).index_add(0, b, -g-g2)
        c, r = self._residual(theta, x)
        g = 2*c[:, None]*r                                  # dE/dx_b (envelope: dE/dF_a = 0)
        return f.index_add(0, self.ls_a, g).index_add(0, self.ls_b, -g)

    def _bond_forces_exact(self, theta, x, q):
        """Closed-form bond force / world torque for kinematics='exact' (derivation: BOND_EXACT.md)."""
        Ri, Rj, Rm, pi, pj, D, th, d, k, dn, dt, tn, tt = self._bond_exact(theta, x, q)
        n = self.n0
        gd = k[:, :1]*dn[:, None]*n+k[:, 1:2]*dt
        gt = k[:, 3:4]*tn[:, None]*n+k[:, 2:3]*tt
        G = (Rm@gd[..., None])[..., 0]                             # dE/dx_j
        cross = torch.linalg.cross
        h = cross(G, D)
        u = (Rm.transpose(1, 2)@h[..., None])[..., 0]
        c = (so3_jr_inv(th).transpose(1, 2)@(gt+.5*(so3_jr(.5*th).transpose(1, 2)@u[..., None])[..., 0])[..., None])[..., 0]
        Rjc = (Rj@c[..., None])[..., 0]
        dEj = Rjc+cross(pj, G)                                     # dE/d(world rotation of j)
        dEi = -Rjc+h-cross(pi, G)
        f = torch.zeros_like(x).index_add(0, self.i, G).index_add(0, self.j, -G)
        tau = torch.zeros_like(x).index_add(0, self.i, -dEi).index_add(0, self.j, -dEj)
        return f, tau

    def _cosserat_forces(self, theta, x, q, skew=True):
        """Closed form for the micropolar terms. skew term W = mu_c V |skew(R^T F)|^2: dW/dF = R (2 mu_c V S),
        dW/d(world rotation) = 2 mu_c V vee(B - B^T), B = F S^T R^T. Curvature term W = 1/2 V [a1 (tr K)^2 + E I_z |K|^2],
        K = N M^-1, N = sum w theta dX^T: Q = dW/dK, g = w (Q_i M_i^-1 + Q_j M_j^-1) dX_ij per bond, and with
        theta = log(R_i^T R_j), delta theta = J_r^-1(theta) R_j^T (d omega_j - d omega_i), so the world torques are
        tau_j = -R_j J_r^-T g, tau_i = +R_j J_r^-T g. Returns the extra stress (per body), forces (none) and torques."""
        F = self.deformation_gradient(x)
        R = quat_matrix(q)
        U = R.transpose(1, 2)@F
        S = .5*(U-U.transpose(1, 2))
        mu = self.k_mu*torch.exp(theta[:, 5])*self.cell_volume
        lam = self.k_v*torch.exp(theta[:, 4])*self.cell_volume
        E = mu*(3*lam+2*mu)/(lam+mu)
        nu = lam/(2*(lam+mu))
        Iz = self.gyration
        P_skew = R@(2*mu[:, None, None]*S)
        B = F@S.transpose(1, 2)@R.transpose(1, 2)
        Bs = B-B.transpose(1, 2)
        tau = -2*mu[:, None]*torch.stack((Bs[:, 2, 1], Bs[:, 0, 2], Bs[:, 1, 0]), -1)
        if not skew:                    # bond_level: the skew term is evaluated per pair (_bond_level_forces)
            P_skew, tau = torch.zeros_like(P_skew), torch.zeros_like(tau)
        K = self.curvature(q)
        trK = K.diagonal(dim1=-2, dim2=-1).sum(-1)
        a1 = 2*mu*(1-nu)*Iz
        eye = torch.eye(3, dtype=x.dtype, device=x.device)
        Q = (a1*trK)[:, None, None]*eye+(E*Iz)[:, None, None]*K
        QM = Q@self.moment_inv
        th = quat_log(quat_rel_principal(q[self.i], q[self.j]))
        nb = len(self.i)
        dX = self.ls_dX[:nb]
        g = self.ls_w[:nb, None]*((QM[self.i]+QM[self.j])@dX[..., None])[..., 0]
        c = (so3_jr_inv(th).transpose(1, 2)@g[..., None])[..., 0]
        Rjc = (R[self.j]@c[..., None])[..., 0]
        tau = tau.index_add(0, self.j, -Rjc).index_add(0, self.i, Rjc)
        return P_skew, tau

    def forces(self, theta, x, q):
        """Closed-form force and world torque, -dE/dx and -dE/d(world rotation)."""
        if self.kinematics == 'exact':
            f, tau = self._bond_forces_exact(theta, x, q)
            if self.volume and getattr(self, 'bond_level', False):
                if getattr(self, 'micropolar', True):
                    _, tau_c = self._cosserat_forces(theta, x, q, skew=False)    # curvature torques only
                    tau = tau+tau_c
                f, tau = self._bond_level_forces(theta, x, q, f, tau)
            elif self.volume and self.body_law == 'dilatation':
                f = self._dilatation_forces(theta, x, f)
            elif self.volume and self.body_law == 'nodalvol':
                f = self._nodalvol_forces(theta, x, f)
            elif self.volume:
                P = self._body_energy_and_stress(theta, x)[1]
                if getattr(self, 'cosserat', False):
                    P_skew, tau_c = self._cosserat_forces(theta, x, q)
                    P = P+P_skew; tau = tau+tau_c
                G = self._thin_pullback(P, x)@self.moment_inv
                g = self.ls_w[:, None]*(G[self.ls_a]@self.ls_dX[..., None])[..., 0]
                f = f.index_add(0, self.ls_a, g).index_add(0, self.ls_b, -g)
            if self.residual:
                f = self._residual_forces(theta, x, f)
            return f, tau
        Ri, Rj, pi, pj, D, Psi, k, dn, dt, pn, pt = self._bond_kinematics(theta, x, quat_matrix(q))
        gd = k[:, :1]*dn[:, None]*self.n0+k[:, 1:2]*dt
        gp = k[:, 3:4]*pn[:, None]*self.n0+k[:, 2:3]*pt
        Rgi, Rgj = (Ri@gd[..., None])[..., 0], (Rj@gd[..., None])[..., 0]
        a = .5*(Rgi+Rgj)                                         # dE/dx_j = a, dE/dx_i = -a
        trace = Psi.diagonal(dim1=1, dim2=2).sum(-1)
        M = .5*(trace[:, None]*gp-(Psi.transpose(1, 2)@gp[..., None])[..., 0])
        T = (Ri@M[..., None])[..., 0]
        cross = torch.linalg.cross
        tj = cross(pj, a)-.5*cross(D, Rgj)+T                     # dE/d(world rotation of j)
        ti = -cross(pi, a)-.5*cross(D, Rgi)-T
        f = torch.zeros_like(x).index_add(0, self.i, a).index_add(0, self.j, -a)
        tau = torch.zeros_like(x).index_add(0, self.i, -ti).index_add(0, self.j, -tj)
        if self.volume and self.body_law == 'dilatation':
            f = self._dilatation_forces(theta, x, f)
        elif self.volume and self.body_law == 'nodalvol':
            f = self._nodalvol_forces(theta, x, f)
        elif self.volume:
            G = self._thin_pullback(self._body_energy_and_stress(theta, x)[1], x)@self.moment_inv  # dE/dS, S = sum w dx dX^T
            g = self.ls_w[:, None]*(G[self.ls_a]@self.ls_dX[..., None])[..., 0]
            f = f.index_add(0, self.ls_a, g).index_add(0, self.ls_b, -g)
        if self.residual:
            f = self._residual_forces(theta, x, f)
        return f, tau

    def forces_autograd(self, theta, x, q):
        """Reference: autograd of energy() under a world-frame rotation perturbation."""
        rot = torch.zeros_like(x, requires_grad=True)
        x = x.detach().requires_grad_(True)
        E = self.energy(theta, x, quat_mul(quat_small(rot), q))
        gx, gr = torch.autograd.grad(E, (x, rot))
        return -gx, -gr

    def damping_forces(self, x, q, v, w, theta=None):
        """Bond dashpots on relative contact velocity (normal / tangential, along the current
        bond axis) and relative spin, each damping * critical for the bond's two-body oscillator
        at reference stiffness. Zero for rigid motion; dissipative, not from the energy.
        A 7th theta column, if present, is a log multiplier of the dashpots (bond: mean of ends)."""
        R = quat_matrix(q)
        pi, pj = (R[self.i]@self.ri[..., None])[..., 0], (R[self.j]@self.rj[..., None])[..., 0]
        cross = torch.linalg.cross
        vrel = v[self.j]+cross(w[self.j], pj)-v[self.i]-cross(w[self.i], pi)
        axis = x[self.j]-x[self.i]
        axis = axis/torch.linalg.vector_norm(axis, dim=-1, keepdim=True)
        vn = (vrel*axis).sum(-1, keepdim=True)*axis
        scale = torch.exp(.5*(theta[self.i, 6]+theta[self.j, 6]))[:, None] if theta is not None and theta.shape[1] > 6 else 1.
        fj = -scale*(self.c_n[:, None]*vn+self.c_t[:, None]*(vrel-vn))
        spin = scale*self.c_r[:, None]*(w[self.j]-w[self.i])
        tj = cross(pj, fj)-spin
        ti = cross(pi, -fj)+spin
        f = torch.zeros_like(x).index_add(0, self.j, fj).index_add(0, self.i, -fj)
        tau = torch.zeros_like(x).index_add(0, self.j, tj).index_add(0, self.i, ti)
        return f, tau

    def step(self, theta, state, dt, external=None):
        """Symplectic Euler (velocity first); isotropic inertia, so no gyroscopic term."""
        x, q, v, w = state
        f, tau = self.forces(theta, x, q)
        if getattr(self, 'k_eta', 0.) > 0:
            f = f+self.viscous_forces(theta, x, v)
        if not getattr(self, 'micropolar', True):
            tau = torch.zeros_like(tau)
        if self.damping > 0:
            fd, td = self.damping_forces(x, q, v, w, theta)
            f = f+fd; tau = tau+td
        if external is not None:
            f = f+external[0]; tau = tau+external[1]
        if self.gravity is not None:
            f = f+self.mass[:, None]*self.gravity
        free = (~self.fixed)[:, None]
        v = torch.where(free, v+dt*f/self.mass[:, None], v)
        w = torch.where(free, w+dt*tau/self.inertia[:, None], w)
        if self.ground_normal is not None and self.ground_mode == 'layer':
            # MPM sticky grid analog: weight W of grid nodes on/below the ground seen by the
            # body's lowest point through PAC-NeRF's quadratic B-spline (CDF of the kernel), at
            # grid spacing layer_dx; velocities scaled by (1 - W)^(dt / layer_dt_ref), so the
            # loss rate per unit time equals MPM's at its step layer_dt_ref. Smooth in height.
            s = ((x-self.ground_point)@self.ground_normal-self.contact_radius)/self.layer_dx
            W = torch.where(s < -.5, torch.ones_like(s),
                torch.where(s < .5, 1-.5*(s+.5)**2, torch.where(s < 1.5, .5*(1.5-s)**2, torch.zeros_like(s))))
            keep = (1-W+1e-6)**(dt/self.layer_dt_ref)          # smooth floor (a clamp left a kink in the derivative)
            v, w = v*keep[:, None], w*keep[:, None]
        elif self.ground_normal is not None:
            touching = ((x-self.ground_point)@self.ground_normal <= self.contact_radius)[:, None]
            if self.ground_mode == 'sticky':
                v_contact = torch.zeros_like(v)
            else:
                vn = v@self.ground_normal
                v_contact = torch.clamp(vn, min=0.)[:, None]*self.ground_normal
            v = torch.where(touching, v_contact, v)
            w = torch.where(touching, torch.zeros_like(w), w)
        x = torch.where(free, x+dt*v, x)
        q = quat_mul(quat_small(dt*w), q)
        q = q/torch.linalg.vector_norm(q, dim=-1, keepdim=True)
        return x, q, v, w

    def max_frequency(self, theta, iterations=80, seed=0):
        """Largest angular frequency at the reference (power iteration on M^-1/2 K M^-1/2),
        K applied by forward-mode differentiation of the closed-form forces. Fixed cells excluded.
        Explicit symplectic Euler is stable for dt < 2 / omega_max."""
        x0, q0, _, _ = self.initial_state()
        m = torch.cat((self.mass[:, None].expand(-1, 3), self.inertia[:, None].expand(-1, 3)), 1)
        free = (~self.fixed)[:, None].to(m.dtype)
        def force(z):
            f, tau = self.forces(theta, x0+z[:, :3], quat_mul(quat_small(z[:, 3:]), q0))
            return torch.cat((f, tau), 1)
        g = torch.Generator(device='cpu').manual_seed(seed)
        y = torch.randn(self.n, 6, generator=g, dtype=torch.float64).to(self.device, self.dtype)*free
        value = torch.tensor(0.)
        for _ in range(iterations):
            y = y/torch.linalg.vector_norm(y)
            _, Ky = torch.func.jvp(force, (torch.zeros(self.n, 6, dtype=self.dtype, device=self.device),),
                                   (y/m.sqrt(),))
            z = -Ky/m.sqrt()*free
            value = (y*z).sum()
            y = z
        return float(value.clamp(min=0).sqrt())

    def stable_dt(self, theta, safety=.5, include_damping=True):
        """safety x min(2 / omega_max, 2 / largest explicit damping rate); include_damping=False when
        the dashpots are integrated implicitly (WarpFoam implicit_damping)."""
        rate = float(torch.maximum(self.gamma_v, self.gamma_w).max()) if self.damping > 0 and include_damping else 0.
        if self.damping > 0 and include_damping and theta.shape[1] > 6:
            rate *= float(torch.exp(theta[:, 6]).max())
        if getattr(self, 'k_eta', 0.) > 0:              # explicit viscous term: dt < 2 / largest decay rate
            rate = max(rate, self.max_damping_rate(theta))
        return safety*2/max(self.max_frequency(theta), rate, 1e-30)

    def rollout(self, theta, state, dt, substeps, frames, checkpoint=True, compile=False, quaternions=False):
        """Positions at every frame (frames+1, n, 3); per-frame gradient checkpointing.

        compile=True fuses each step with torch.compile (about 4x faster on GPU for ~4k cells)."""
        from torch.utils.checkpoint import checkpoint as ckpt
        if compile and getattr(self, '_compiled_step', None) is None:
            self._compiled_step = torch.compile(self.step, dynamic=False)
        step = self._compiled_step if compile else self.step
        if compile:  # a tensor dt is a graph input: changing the step size does not recompile
            dt = torch.as_tensor(dt, dtype=self.dtype, device=self.device)

        def advance(x, q, v, w, theta):
            s = (x, q, v, w)
            for _ in range(substeps):
                s = step(theta, s, dt)
            return s
        out, quats = [state[0]], [state[1]]
        for _ in range(frames):
            if checkpoint and theta.requires_grad:
                state = ckpt(advance, *state, theta, use_reentrant=False)
            else:
                state = advance(*state, theta)
            out.append(state[0]); quats.append(state[1])
        if quaternions:
            return torch.stack(out), torch.stack(quats), state
        return torch.stack(out), state

    def kinetic(self, state):
        _, _, v, w = state
        return .5*((self.mass*(v*v).sum(-1)).sum()+(self.inertia*(w*w).sum(-1)).sum())
