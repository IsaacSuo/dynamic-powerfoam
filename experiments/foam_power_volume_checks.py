"""Static checks of the power-cell volume term (LinearizedFoam volume_mode='power').

1. Uniform strain: J_i - 1 = tr(eps) for every cell, surface cells included (air ghosts).
2. Rigid translation / infinitesimal rotation: J_i - 1 = 0.
3. Pure bending of the bar about y: top-layer cells J > 1, bottom-layer J < 1, and the
   tetrahedral nodal average for comparison.
4. Affine moduli of the term: C11 = C12 = k_v, C44 = 0.
"""
import json
import sys

import numpy as np

from experiments.foam_linear_inverse import LinearizedFoam


def field(points, gradient, spin=None):
    u = np.zeros((len(points), 6))
    u[:, :3] = points@gradient.T
    if spin is not None:
        u[:, 3:] = spin
    return u.ravel()


def run(label, geometry):
    power = LinearizedFoam(geometry, volume=True, volume_mode='power', k_v=1.)
    tet = LinearizedFoam(geometry, volume=True, volume_mode='tet', k_v=1.)
    p = power.points
    rng = np.random.default_rng(0)
    eps = rng.normal(size=(3, 3)); eps = .5*(eps+eps.T)
    J = power.volume_B@field(p, eps)
    out = dict(cells=len(p), affine_max_error=float(np.abs(J-np.trace(eps)).max()))
    w = rng.normal(size=3)
    W = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    out['rotation_max'] = float(np.abs(power.volume_B@field(p, W, w)).max())
    out['translation_max'] = float(np.abs(power.volume_B@np.tile([.3, -.2, .1, 0, 0, 0], len(p))).max())
    # Bending about y: u_x = -kappa x z, u_z = kappa x^2 / 2 (curvature kappa).
    kappa = 1e-3
    u = np.zeros((len(p), 6))
    u[:, 0] = -kappa*p[:, 0]*p[:, 2]; u[:, 2] = .5*kappa*p[:, 0]**2; u[:, 4] = kappa*p[:, 0]
    u = u.ravel()
    top, bottom = p[:, 2] == p[:, 2].max(), p[:, 2] == p[:, 2].min()
    for name, model in (('power', power), ('tet', tet)):
        Jb = model.volume_B@u
        out[f'bending_{name}_top_mean'] = float(Jb[top].mean())
        out[f'bending_{name}_bottom_mean'] = float(Jb[bottom].mean())
    out['bending_exact_top'] = float(-kappa*p[top, 2].mean())
    # Affine moduli of the volume term alone.
    K = (power.volume_B.T@(power.volume_base[:, None]*power.volume_B.toarray()))
    V = float(np.sum(geometry['masses']))
    E = []
    for i in range(6):
        e = np.zeros((3, 3))
        if i < 3:
            e[i, i] = 1
        else:
            a, b = [(1, 2), (0, 2), (0, 1)][i-3]; e[a, b] = e[b, a] = .5
        E.append(field(p, e))
    E = np.array(E).T
    C = E.T@K@E/V
    out.update(C11=float(C[0, 0]), C12=float(C[0, 1]), C44=float(C[3, 3]))
    print(label, json.dumps(out))
    return out


if __name__ == '__main__':
    for path in sys.argv[1:]:
        run(path, dict(np.load(path)))
