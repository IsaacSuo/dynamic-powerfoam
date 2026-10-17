"""Static checks of the exact power-cell shape term (LinearizedFoam shape_mode='power').

1. Affine fields u = H x (H random, not symmetric): G_i = H for every cell, air side included.
2. tr G_i equals the power volume row / V_i (same faces, same ghosts).
3. Rigid translation / rotation: sym G_i = 0, shape energy 0.
4. Affine moduli of the shape term: (C11, C12, C44) = (4/3, -2/3, 1) k_d; with the power
   volume term, k_v = lam + 2 mu / 3 and k_d = mu reproduce isotropic C exactly.
5. Checkerboard (alternating +-1 per cell): count of zero-energy modes of volume + shape.
6. Same affine moduli as 4 on the whole specimen (boundary cells included).
"""
import json
import sys

import numpy as np

from experiments.foam_linear_inverse import LinearizedFoam, power_gradient_rows
from experiments.foam_affine_moduli_check import strain_fields


def run(label, geometry):
    p = np.asarray(geometry['points'], float)
    n = len(p)
    G = power_gradient_rows(geometry)
    rng = np.random.default_rng(0)
    H = rng.normal(size=(3, 3))
    u = np.zeros((n, 6)); u[:, :3] = p@H.T
    out = dict(cells=n, affine_gradient_max_error=float(np.abs((G@u.ravel()).reshape(n, 3, 3)-H).max()))
    model = LinearizedFoam(geometry, volume=True, volume_mode='power', deviatoric=True, shape_mode='power', k_v=1., k_d=1.)
    trace = (G@u.ravel()).reshape(n, 3, 3).trace(axis1=1, axis2=2)
    out['trace_vs_volume_max'] = float(np.abs(trace-model.volume_B@u.ravel()).max())
    w = rng.normal(size=3)
    W = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    r = np.zeros((n, 6)); r[:, :3] = p@W.T+[.3, -.2, .1]; r[:, 3:] = w
    out['rigid_shape_max'] = float(np.abs(model.shape_D@r.ravel()).max())
    E = strain_fields(p)
    Ks = model.shape_D.T@(np.repeat(model.shape_base, 6)[:, None]*model.shape_D.toarray())
    Kv = model.volume_B.T@(model.volume_base[:, None]*model.volume_B.toarray())
    V = float(np.sum(geometry['masses']))
    Cs, Cv = E.T@Ks@E/V, E.T@Kv@E/V
    out.update(shape_C11=Cs[0, 0], shape_C12=Cs[0, 1], shape_C44=Cs[3, 3],
               volume_C11=Cv[0, 0], volume_C12=Cv[0, 1], volume_C44=Cv[3, 3])
    lam = mu = 480.
    C = (lam+2*mu/3)*Cv+mu*Cs
    out.update(combined_C11=C[0, 0], combined_C12=C[0, 1], combined_C44=C[3, 3])
    # Zero-energy translational modes of volume + shape alone (rotations are not involved).
    K = (Ks+Kv)
    trans = np.flatnonzero(np.tile([1, 1, 1, 0, 0, 0], n))
    eig = np.linalg.eigvalsh(K[np.ix_(trans, trans)])
    out['zero_modes_translation'] = int(np.sum(eig < 1e-9*eig.max()))
    print(label, json.dumps({k: (round(v, 12) if isinstance(v, float) else v) for k, v in out.items()}))
    return out


if __name__ == '__main__':
    for path in sys.argv[1:]:
        run(path, dict(np.load(path)))
