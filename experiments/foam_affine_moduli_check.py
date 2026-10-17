"""Affine (homogeneous-strain) moduli of each term of LinearizedFoam, term by term.

Applies the six unit Voigt strains as center fields u = eps X (symmetric, so zero
cell rotation) and reports C_ij = e_i^T K e_j / V for each term separately.
"""
import json
import numpy as np
import scipy.sparse as sp
from experiments.foam_linear_inverse import LinearizedFoam


def strain_fields(points):
    fields = []
    for i in range(6):
        eps = np.zeros((3, 3))
        if i < 3:
            eps[i, i] = 1
        else:
            a, b = [(1, 2), (0, 2), (0, 1)][i-3]
            eps[a, b] = eps[b, a] = .5  # engineering shear gamma = 1
        u = np.zeros((len(points), 6))
        u[:, :3] = points@eps.T
        fields.append(u.ravel())
    return np.array(fields).T


def terms(model):
    out = {}
    for m, name in enumerate(('pair normal (kn)', 'pair shear (ks)', 'bending (kb)', 'twist (kt)')):
        blocks = model.base[:, m, None, None]*model.unit[:, m]
        out[name] = sp.coo_matrix((blocks.ravel(), (model._rows, model._cols)), shape=(6*model.n,)*2).tocsr()
    out['volume (k_v=1)'] = model.volume_B.T@sp.diags(model.volume_base/model.volume_term.k_v)@model.volume_B
    out['shape (k_d=1)'] = model.shape_D.T@sp.diags(np.repeat(model.shape_base/400., 6))@model.shape_D
    return out


def run(label, geometry):
    model = LinearizedFoam(geometry, volume=True, deviatoric=True, k_d=400.)
    E = strain_fields(model.points)
    volume = float(np.sum(geometry['masses']))
    print(f'== {label}: {model.n} cells, material volume {volume:.3f}')
    print('%-18s %9s %9s %9s %9s %9s' % ('term', 'C11', 'C12', 'C44', 'C11-C22', 'C44-C55'))
    results = {}
    for name, K in terms(model).items():
        C = E.T@(K@E)/volume
        results[name] = C.tolist()
        print('%-18s %9.3f %9.3f %9.3f %9.1e %9.1e' % (name, C[0, 0], C[0, 1], C[3, 3], C[0, 0]-C[1, 1], C[3, 3]-C[4, 4]))
    return results


if __name__ == '__main__':
    run('regular 4x4x4', dict(np.load('output/foam_fem/moduli_v1/patch64.npz')))
    run('FEM patch 4x2x2', dict(np.load('output/foam_fem/reference_edge007/patch.npz')))
