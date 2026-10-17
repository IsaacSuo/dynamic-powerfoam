"""Float64 reference TLSPH, fixed material graph, no rendering dependencies.

Corrected gradients and antisymmetric elastic forces follow Ganzenmueller
(2015), sections 2.3-2.4. The central, symmetric hourglass force follows eq. 32,
using reference volumes and a 3D cubic spline kernel. This is an independent
implementation, not the author's code or an exact reproduction of his fixtures.
"""
import numpy as np
from scipy.spatial import cKDTree


class TLSPH:
    def __init__(self, reference, volumes, h, young=1000., poisson=.3,
                 density=1., hourglass=10.):
        self.X = np.asarray(reference, dtype=np.float64).copy()
        self.V = np.asarray(volumes, dtype=np.float64).copy()
        self.mass = self.V * density
        self.h, self.young, self.alpha = h, young, hourglass
        self.mu = young / (2 * (1 + poisson))
        self.lam = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
        self.wave_speed = np.sqrt((self.lam + 2 * self.mu) / density)
        pairs = cKDTree(self.X).query_pairs(2*h, output_type='ndarray')
        self.i, self.j = pairs.T
        self.R = self.X[self.j] - self.X[self.i]
        self.distance = np.linalg.norm(self.R, axis=1)
        if np.any(self.distance < 1e-12) or np.any(self.V <= 0):
            raise ValueError('Coincident particles or nonpositive volumes')
        q = self.distance / h
        self.W = np.where(q < 1, 1-1.5*q*q+.75*q**3, .25*(2-q)**3)/(np.pi*h**3)
        dw = np.where(q < 1, -3*q+2.25*q*q, -.75*(2-q)**2)/(np.pi*h**4)
        raw = -dw[:, None] * self.R / self.distance[:, None]
        moment = np.zeros((len(self.X), 3, 3))
        outer = self.R[:, :, None] * raw[:, None, :]
        np.add.at(moment, self.i, self.V[self.j, None, None]*outer)
        np.add.at(moment, self.j, self.V[self.i, None, None]*outer)
        self.condition = np.linalg.cond(moment)
        if not np.isfinite(self.condition).all() or self.condition.max() > 1e8:
            raise ValueError('Reference neighborhood does not resolve 3D deformation')
        correction = np.linalg.inv(moment).transpose(0, 2, 1)
        self.gi = np.einsum('eab,eb->ea', correction[self.i], raw)
        self.gj = np.einsum('eab,eb->ea', correction[self.j], -raw)
        self.degree = np.bincount(pairs.ravel(), minlength=len(self.X))

    def deformation(self, x):
        dx = x[self.j] - x[self.i]
        F = np.zeros((len(x), 3, 3))
        np.add.at(F, self.i, self.V[self.j, None, None]*dx[:, :, None]*self.gi[:, None, :])
        np.add.at(F, self.j, -self.V[self.i, None, None]*dx[:, :, None]*self.gj[:, None, :])
        return F

    def evaluate(self, x, alpha=None):
        F = self.deformation(x)
        J = np.linalg.det(F)
        if J.min() <= 0:
            raise ValueError(f'Inverted material: min det F = {J.min()}')
        logJ = np.log(J)
        invT = np.linalg.inv(F).transpose(0, 2, 1)
        P = self.mu*(F-invT) + self.lam*logJ[:, None, None]*invT
        psi = .5*self.mu*(np.sum(F*F, axis=(1, 2))-3) - self.mu*logJ + .5*self.lam*logJ**2
        pair = (self.V[self.i]*self.V[self.j])[:, None]*(
            np.einsum('eab,eb->ea', P[self.i], self.gi) -
            np.einsum('eab,eb->ea', P[self.j], self.gj))
        elastic = np.zeros_like(x)
        np.add.at(elastic, self.i, pair)
        np.add.at(elastic, self.j, -pair)
        dx = x[self.j] - x[self.i]
        length = np.linalg.norm(dx, axis=1)
        if length.min() < 1e-12:
            raise ValueError('Coincident current particles')
        direction = dx / length[:, None]
        # Both endpoint errors expressed in the common i -> j orientation.
        error = np.einsum('eab,eb->ea', F[self.i]+F[self.j], self.R)-2*dx
        delta_sum = np.sum(error*direction, axis=1)
        strength = self.alpha if alpha is None else alpha
        coefficient = -.5*strength*self.young*self.V[self.i]*self.V[self.j]*self.W/self.distance**2
        hg_pair = (coefficient*delta_sum)[:, None]*direction
        hg = np.zeros_like(x)
        np.add.at(hg, self.i, hg_pair)
        np.add.at(hg, self.j, -hg_pair)
        return elastic+hg, dict(F=F, J=J, elastic=elastic, hourglass=hg,
                               energy=float(self.V@psi),
                               hg_error=float(np.sqrt(np.mean((delta_sum/self.distance)**2))))


def beam(resolution=4):
    """Known 2 x .5 x .5 solid; x endpoint quadrature, yz cell centers."""
    dx = .5/resolution
    xx = np.linspace(0, 2, 4*resolution+1)
    yy = (np.arange(resolution)+.5)*dx-.25
    X = np.stack(np.meshgrid(xx, yy, yy, indexing='ij'), axis=-1).reshape(-1, 3)
    V = np.full(len(X), dx**3)
    V[(X[:, 0] == 0) | (X[:, 0] == 2)] *= .5
    fixed = X[:, 0] <= .25+1e-9
    tip = X[:, 0] == 2
    return X, V, fixed, tip, dx
