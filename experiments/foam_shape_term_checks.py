"""Static checks of the deviatoric (shape) multi-body term in LinearizedFoam.

Uses only the shape-term tangent (all pair/volume contributions excluded) on the
16-cell FEM patch and the regular 64-cell patch with 8 interior cells.
"""
import json
import numpy as np
from experiments.foam_linear_inverse import LinearizedFoam


def shape_tangent(model):
    theta = np.zeros((model.n, model.parameter_count))
    weight = np.repeat(model.shape_base, 6)
    return (model.shape_D.T@(weight[:, None]*model.shape_D.toarray())), theta


def field(points, gradient):
    """Center translations u = G X, zero rotations (6 DOFs per cell)."""
    u = np.zeros((len(points), 6))
    u[:, :3] = points@gradient.T
    return u.ravel()


def run(name, geometry):
    model = LinearizedFoam(geometry, deviatoric=True, k_d=1.)
    K, _ = shape_tangent(model)
    volume = float(np.sum(geometry['masses']))
    out = dict(cells=model.n, symmetric=float(np.abs(K-K.T).max()/np.abs(K).max()),
               min_eigenvalue_relative=float(np.linalg.eigvalsh(K).min()/np.linalg.eigvalsh(K).max()))
    rng = np.random.default_rng(0)
    shear = rng.normal(size=(3, 3)); shear = .5*(shear+shear.T); shear -= np.trace(shear)/3*np.eye(3)
    u = field(model.points, shear)
    expected = volume*np.sum(shear*shear)  # E = k_d |dev eps|^2 V with k_d = 1
    out['affine_shear_energy_relative_error'] = float(abs(.5*u@K@u/expected-1))
    out['dilatation_energy'] = float(.5*field(model.points, np.eye(3))@K@field(model.points, np.eye(3)))
    spin = rng.normal(size=3)
    W = np.array([[0, -spin[2], spin[1]], [spin[2], 0, -spin[0]], [-spin[1], spin[0], 0]])
    rot = field(model.points, W).reshape(-1, 6); rot[:, 3:] = spin
    out['rigid_rotation_force'] = float(np.abs(K@rot.ravel()).max()/np.abs(K@u).max())
    trans = np.zeros((model.n, 6)); trans[:, :3] = rng.normal(size=3)
    out['translation_force'] = float(np.abs(K@trans.ravel()).max()/np.abs(K@u).max())
    # Patch test: under a uniform deviatoric strain, cells away from the boundary feel no net force.
    force = (K@u).reshape(-1, 6)[:, :3]
    lo, hi = model.points.min(0), model.points.max(0)
    interior = np.all((model.points > lo+1e-9) & (model.points < hi-1e-9), axis=1)
    out['interior_cells'] = int(interior.sum())
    out['interior_force_relative'] = (float(np.abs(force[interior]).max()/np.abs(force).max())
                                      if interior.any() else None)
    print(name, json.dumps(out))
    return out


if __name__ == '__main__':
    results = {}
    results['fem16'] = run('fem16', dict(np.load('output/foam_fem/reference_edge007/patch.npz')))
    results['regular64'] = run('regular64', dict(np.load('output/foam_fem/moduli_v1/patch64.npz')))
