"""Float64, 3D parallel-bond *component*, with fixed reference connections.

Source: Itasca 9.6 Linear Parallel Bond Model, eqs. (5)-(15), and Contact
Resolution, eqs. (2), (5), (7), (12). Archived in references/bonded_dem/.
This is our implementation, not PFC code or a reproduction of a paper fixture.

Scope: elastic normal/shear forces, bend/twist couples, irreversible brittle
failure. No frictional contact branch, damping, softening, mass assignment,
Power Foam radius law, or automatic creation/deletion of connections.

We integrate the published rate law at a supplied midpoint configuration.
World-space tangential history is transported by exact shortest-arc rotations
and axial spin, rather than the documentation's first-order frame update.
Finite-step frame transport is an approximation; convergence must be checked.
Force/couple signs below are those acting on endpoint i; n points i -> j.
Positions are body centers of mass in this baseline, not arbitrary generators.
"""

import numpy as np


def _dot(a, b):
    return np.einsum('...i,...i->...', a, b)


def _rotate_axis(v, axis, angle):
    c, s = np.cos(angle)[:, None], np.sin(angle)[:, None]
    return c*v + s*np.cross(axis, v) + (1-c)*_dot(axis, v)[:, None]*axis


def _transport(v, n0, n1, spin):
    """Transport tangent vectors, including common rotation about the normal."""
    cross = np.cross(n0, n1)
    cosine = np.clip(_dot(n0, n1), -1., 1.)
    if np.any(cosine < np.cos(.25)):
        raise ValueError('Contact normal turns > 0.25 rad per half-step; subdivide')
    # Rodrigues' shortest-arc rotation; well-defined also for equal normals.
    out = v + np.cross(cross, v)
    out += np.cross(cross, np.cross(cross, v))/(1+cosine)[:, None]
    bisector = n0+n1
    bisector /= np.linalg.norm(bisector, axis=1)[:, None]
    return _rotate_axis(out, n1, _dot(spin, bisector))


class ParallelBonds:
    """Stateful bonded interaction; advance exactly once per physical step.

    bond_radius determines a fixed circular section. kbar_n and kbar_s have
    units stress/length (NOT force/length). The caller may choose E_b/L0 and
    G_b/L0, but these are connection parameters, not calibrated bulk moduli.
    Particle radii affect the classical ball-ball contact location only.
    All bonds start stress-free, regardless of initial sphere overlap.
    """

    def __init__(self, reference, pairs, radii, bond_radius, kbar_n, kbar_s,
                 tensile_strength=np.inf, cohesion=np.inf,
                 friction_angle=0., moment_factor=1.):
        self.x = np.asarray(reference, dtype=np.float64).copy()
        if (self.x.ndim != 2 or self.x.shape[1] != 3
                or not np.isfinite(self.x).all()):
            raise ValueError('reference must be a finite (N, 3) array')
        raw_pairs = np.asarray(pairs)
        if (raw_pairs.ndim != 2 or raw_pairs.shape[1] != 2 or len(raw_pairs) == 0
                or not np.issubdtype(raw_pairs.dtype, np.integer)):
            raise ValueError('pairs must be a nonempty integer (M, 2) array')
        self.pairs = raw_pairs.astype(np.int64, copy=True)
        if np.any(self.pairs < 0) or np.any(self.pairs >= len(self.x)):
            raise ValueError('Invalid endpoint index')
        canonical = np.sort(self.pairs, axis=1)
        if (np.any(canonical[:, 0] == canonical[:, 1])
                or len(np.unique(canonical, axis=0)) != len(canonical)):
            raise ValueError('Self-bonds and duplicate undirected bonds are invalid')
        self.i, self.j = self.pairs.T
        self.radii = np.broadcast_to(np.asarray(radii, dtype=float), (len(self.x),)).copy()
        if not np.isfinite(self.radii).all() or np.any(self.radii <= 0):
            raise ValueError('Particle radii must be positive and finite')
        count = len(self.pairs)

        def parameter(value, name, allow_inf=False):
            out = np.broadcast_to(np.asarray(value, dtype=float), (count,)).copy()
            valid = ~np.isnan(out) if allow_inf else np.isfinite(out)
            if not valid.all() or np.any(out <= 0):
                raise ValueError(f'{name} must be positive' + (' or +inf' if allow_inf else ' and finite'))
            return out

        self.rb = parameter(bond_radius, 'bond_radius')
        self.area = np.pi*self.rb**2
        self.second_moment = np.pi*self.rb**4/4
        self.polar_moment = 2*self.second_moment
        kn = parameter(kbar_n, 'kbar_n')
        ks = parameter(kbar_s, 'kbar_s')
        self.kn, self.ks = kn*self.area, ks*self.area
        self.kb, self.kt = kn*self.second_moment, ks*self.polar_moment
        self.tensile_strength = parameter(tensile_strength, 'tensile_strength', True)
        self.cohesion = parameter(cohesion, 'cohesion', True)
        if not np.isfinite(friction_angle) or not 0 <= friction_angle < np.pi/2:
            raise ValueError('friction_angle must be in [0, pi/2) radians')
        if not np.isfinite(moment_factor) or not 0 <= moment_factor <= 1:
            raise ValueError('moment_factor must be in [0, 1]')
        self.tan_phi = np.tan(friction_angle)
        self.beta = float(moment_factor)
        self.normal, _ = self.geometry(self.x)
        self.reference_length = np.linalg.norm(self.x[self.j]-self.x[self.i], axis=1)
        self.fn = np.zeros(count)            # positive tension
        self.fs = np.zeros((count, 3))       # force on i
        self.mt = np.zeros(count)            # twist couple on i, along n
        self.mb = np.zeros((count, 3))       # bending couple on i
        self.failure = np.zeros(count, dtype=np.int8)  # 0 intact, 1 tension, 2 shear
        self.break_energy = np.zeros(count)
        self.work = 0.                       # constitutive work, not external work
        self.time = 0.

    @property
    def intact(self):
        return self.failure == 0

    def geometry(self, x):
        """Classical ball-ball contact point; not the weighted power face."""
        d = x[self.j]-x[self.i]
        length = np.linalg.norm(d, axis=1)
        if np.any(length <= 1e-12):
            raise ValueError('Coincident centers are outside this baseline')
        normal = d/length[:, None]
        c = .5*(x[self.i]+x[self.j])
        c += .5*(self.radii[self.i]-self.radii[self.j])[:, None]*normal
        return normal, c

    def elastic_energy_per_bond(self):
        return .5*(self.fn**2/self.kn + _dot(self.fs, self.fs)/self.ks
                   + self.mt**2/self.kt + _dot(self.mb, self.mb)/self.kb)

    def loads(self):
        """Assemble forces and torques about centers, including force lever arms."""
        n, c = self.geometry(self.x)
        f = self.fn[:, None]*n + self.fs
        couple = self.mt[:, None]*n + self.mb
        force, torque = np.zeros_like(self.x), np.zeros_like(self.x)
        np.add.at(force, self.i, f)
        np.add.at(force, self.j, -f)
        np.add.at(torque, self.i, np.cross(c-self.x[self.i], f)+couple)
        np.add.at(torque, self.j, np.cross(c-self.x[self.j], -f)-couple)
        return force, torque

    def advance(self, x_next, velocity_mid, omega_mid, dt, *, x_mid=None):
        """Integrate bond history using midpoint center/velocity/spin samples.

        x_mid defaults to (old+new)/2, appropriate for a straight drift step.
        For prescribed curved trajectories supply the true midpoint position
        and its analytic velocities. This function does NOT integrate bodies.
        omega is world angular velocity; arbitrary render quaternions are not
        mechanical rotations. Subdivide large angular steps before calling.
        """
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('dt must be positive and finite')

        def state(value, name):
            a = np.asarray(value, dtype=np.float64)
            if a.shape != self.x.shape or not np.isfinite(a).all():
                raise ValueError(f'{name} must be finite with shape {self.x.shape}')
            return a

        end = state(x_next, 'x_next')
        mid = .5*(self.x+end) if x_mid is None else state(x_mid, 'x_mid')
        velocity = state(velocity_mid, 'velocity_mid')
        omega = state(omega_mid, 'omega_mid')
        if np.max(np.linalg.norm(omega, axis=1))*dt > .25:
            raise ValueError('Body rotation > 0.25 rad per step; subdivide')
        nm, contact = self.geometry(mid)
        ne, _ = self.geometry(end)
        spin = .5*(omega[self.i]+omega[self.j])*dt
        # Validate both transports before changing history (a rejected step
        # must not partly mutate the model).
        fs0 = _transport(self.fs, self.normal, nm, .5*spin)
        mb0 = _transport(self.mb, self.normal, nm, .5*spin)
        _transport(np.zeros_like(fs0), nm, ne, .5*spin)
        du = (velocity[self.j] + np.cross(omega[self.j], contact-mid[self.j])
              - velocity[self.i] - np.cross(omega[self.i], contact-mid[self.i]))*dt
        dtheta = (omega[self.j]-omega[self.i])*dt
        dn, twist = _dot(du, nm), _dot(dtheta, nm)
        ds, bend = du-dn[:, None]*nm, dtheta-twist[:, None]*nm
        active = self.intact
        dn, twist = dn*active, twist*active
        ds, bend = ds*active[:, None], bend*active[:, None]
        fn = self.fn+self.kn*dn
        fs = fs0+self.ks[:, None]*ds
        mt = self.mt+self.kt*twist
        mb = mb0+self.kb[:, None]*bend
        work = .5*((self.fn+fn)*dn + _dot(fs0+fs, ds)
                   + (self.mt+mt)*twist + _dot(mb0+mb, bend))
        sigma = fn/self.area + self.beta*np.linalg.norm(mb, axis=1)*self.rb/self.second_moment
        tau = np.linalg.norm(fs, axis=1)/self.area + self.beta*np.abs(mt)*self.rb/self.polar_moment
        tensile = active & (sigma > self.tensile_strength)
        shear = active & ~tensile & (tau > self.cohesion-fn/self.area*self.tan_phi)
        broken = tensile | shear
        energy_trial = .5*(fn**2/self.kn + _dot(fs, fs)/self.ks
                           + mt**2/self.kt + _dot(mb, mb)/self.kb)
        self.break_energy[broken] += energy_trial[broken]
        self.failure[tensile], self.failure[shear] = 1, 2
        fn[broken], fs[broken], mt[broken], mb[broken] = 0., 0., 0., 0.
        self.fn, self.mt = fn, mt
        self.fs = _transport(fs, nm, ne, .5*spin)
        self.mb = _transport(mb, nm, ne, .5*spin)
        self.work += float(work.sum())
        self.x, self.normal = end.copy(), ne
        self.time += dt
        return dict(newly_broken=int(broken.sum()),
                    energy=float(self.elastic_energy_per_bond().sum()),
                    removed_bond_energy=float(self.break_energy.sum()))
