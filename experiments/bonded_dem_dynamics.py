"""Stage-2 helpers: reference tangent and many-body isotropic-inertia dynamics.

The tangent is the stress-free linearization of bonded_dem_core, assembled
from contact-point translation and relative spin. It is NOT a finite-strain
replacement for the history-dependent bond law. The body integrator uses the
actual bond law; no frozen stiffness is used to compute dynamic forces.
"""

import numpy as np
from scipy import sparse
from scipy.spatial.transform import Rotation


def skew(v):
    out = np.zeros(v.shape[:-1]+(3, 3))
    out[..., 0, 1], out[..., 0, 2] = -v[..., 2], v[..., 1]
    out[..., 1, 0], out[..., 1, 2] = v[..., 2], -v[..., 0]
    out[..., 2, 0], out[..., 2, 1] = -v[..., 1], v[..., 0]
    return out


def reference_stiffness(bonds):
    """Sparse K for q=[u_x,u_y,u_z,theta_x,theta_y,theta_z] per body.

    From U=1/2 du.D.du + 1/2 dtheta.H.dtheta at an unstressed state.
    The force lever-arm torques are automatically included by B.T D B.
    """
    if (np.any(bonds.failure) or np.any(bonds.fn) or np.any(bonds.fs)
            or np.any(bonds.mt) or np.any(bonds.mb)):
        raise ValueError('Reference tangent requires an intact, stress-free state')
    n, c = bonds.geometry(bonds.x)
    ri, rj = c-bonds.x[bonds.i], c-bonds.x[bonds.j]
    count = len(n)
    b, a = np.zeros((count, 3, 12)), np.zeros((count, 3, 12))
    b[:, :, :3], b[:, :, 3:6] = -np.eye(3), skew(ri)
    b[:, :, 6:9], b[:, :, 9:12] = np.eye(3), -skew(rj)
    a[:, :, 3:6], a[:, :, 9:12] = -np.eye(3), np.eye(3)
    nn = n[:, :, None]*n[:, None, :]
    d = bonds.ks[:, None, None]*np.eye(3)+(bonds.kn-bonds.ks)[:, None, None]*nn
    h = bonds.kb[:, None, None]*np.eye(3)+(bonds.kt-bonds.kb)[:, None, None]*nn
    local = np.einsum('eai,eab,ebj->eij', b, d, b)+np.einsum('eai,eab,ebj->eij', a, h, a)
    indices = np.concatenate((6*bonds.i[:, None]+np.arange(6),
                              6*bonds.j[:, None]+np.arange(6)), axis=1)
    rows = np.broadcast_to(indices[:, :, None], local.shape).ravel()
    cols = np.broadcast_to(indices[:, None, :], local.shape).ravel()
    size = 6*len(bonds.x)
    return sparse.coo_matrix((local.ravel(), (rows, cols)), shape=(size, size)).tocsr()


def mass_vector(mass, inertia):
    return np.column_stack((np.repeat(np.asarray(mass)[:, None], 3, axis=1),
                            np.repeat(np.asarray(inertia)[:, None], 3, axis=1))).ravel()


def reference_dt_bound(stiffness, mass, inertia, fixed):
    """Conservative linear bound 2/omega_max via absolute row-sum bound.

    Only a reference-state bound, not a nonlinear adaptive stability guarantee.
    Caller chooses an accuracy factor and verifies dt convergence.
    """
    free = np.flatnonzero(~np.repeat(fixed, 6))
    inverse = 1/np.sqrt(mass_vector(mass, inertia)[free])
    scaled = sparse.diags(inverse)@stiffness[free][:, free]@sparse.diags(inverse)
    bound = np.max(np.asarray(abs(scaled).sum(axis=1)))
    if not np.isfinite(bound) or bound <= 0:
        raise ValueError('No positive finite reference frequency bound')
    return float(2/np.sqrt(bound))


class BondedDynamics:
    """Kick-drift-kick bodies, scalar isotropic inertia and stationary clamps.

    Mass/inertia are supplied explicitly, not inferred from rendering radii.
    Orientations are active world rotation matrices; no optical coordinates.
    External loads passed to step are held constant for that step. No damping,
    moving grips, general inertia tensor, collision, or implicit solve yet.
    """

    def __init__(self, bonds, mass, inertia, fixed=None):
        self.bonds = bonds
        count = len(bonds.x)
        self.mass = np.broadcast_to(np.asarray(mass, dtype=float), (count,)).copy()
        self.inertia = np.broadcast_to(np.asarray(inertia, dtype=float), (count,)).copy()
        if (not np.isfinite(self.mass).all() or not np.isfinite(self.inertia).all()
                or np.any(self.mass <= 0) or np.any(self.inertia <= 0)):
            raise ValueError('Positive finite mass and scalar inertia are required')
        self.fixed = np.zeros(count, dtype=bool) if fixed is None else np.asarray(fixed, dtype=bool).copy()
        if self.fixed.shape != (count,):
            raise ValueError('fixed must contain one flag per body')
        self.velocity = np.zeros_like(bonds.x)
        self.omega = np.zeros_like(bonds.x)
        self.rotation = np.broadcast_to(np.eye(3), (count, 3, 3)).copy()
        self.time = 0.

    def energy(self):
        kinetic = .5*np.sum(self.mass[:, None]*self.velocity**2+self.inertia[:, None]*self.omega**2)
        return float(kinetic+self.bonds.elastic_energy_per_bond().sum())

    def momentum(self):
        p = self.mass[:, None]*self.velocity
        return p.sum(0), (np.cross(self.bonds.x, p)+self.inertia[:, None]*self.omega).sum(0)

    def step(self, dt, force_external=None, torque_external=None):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('dt must be positive and finite')

        def external(value):
            if value is None:
                return np.zeros_like(self.bonds.x)
            out = np.asarray(value, dtype=float)
            if out.shape != self.bonds.x.shape or not np.isfinite(out).all():
                raise ValueError('External loads must match body positions and be finite')
            return out

        fe, te = external(force_external), external(torque_external)
        f, t = self.bonds.loads()
        vh = self.velocity+.5*dt*(f+fe)/self.mass[:, None]
        wh = self.omega+.5*dt*(t+te)/self.inertia[:, None]
        vh[self.fixed], wh[self.fixed] = 0., 0.
        end = self.bonds.x+dt*vh
        self.bonds.advance(end, vh, wh, dt)
        self.rotation = Rotation.from_rotvec(dt*wh).as_matrix()@self.rotation
        f, t = self.bonds.loads()
        self.velocity = vh+.5*dt*(f+fe)/self.mass[:, None]
        self.omega = wh+.5*dt*(t+te)/self.inertia[:, None]
        self.velocity[self.fixed], self.omega[self.fixed] = 0., 0.
        self.time += dt
