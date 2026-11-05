"""the voxel FEM's trilinear mode interpolation (pacnerf_modes.py's), shared by the free-vibration scripts"""
import numpy as np
from scipy.spatial import cKDTree


def interpolate(fem, vecs, Xq):
    lo, h, grid = fem['lo'], float(fem['h']), fem['node_grid']
    key = lambda g_: (g_[:, 0]*100003+g_[:, 1])*100003+g_[:, 2]
    order = np.argsort(key(grid)); keys = key(grid)[order]

    def node_of(g_):
        k_ = key(g_); i_ = np.clip(np.searchsorted(keys, k_), 0, len(keys)-1)
        return np.where(keys[i_] == k_, order[i_], -1)
    t = (Xq-lo)/h; b = np.floor(t).astype(int); fr = t-b
    U = vecs.reshape(len(grid), 3, -1)
    out = np.zeros((len(Xq), 3, vecs.shape[1])); ok = np.ones(len(Xq), bool)
    for c in np.array(np.meshgrid([0, 1], [0, 1], [0, 1], indexing='ij')).reshape(3, -1).T:
        n_ = node_of(b+c); ok &= n_ >= 0
        w = np.prod(np.where(c[None] == 1, fr, 1-fr), axis=1)
        out += np.where((n_ >= 0)[:, None, None], w[:, None, None]*U[np.maximum(n_, 0)], 0.)
    if (~ok).any():
        _, near = cKDTree(fem['nodes']).query(Xq[~ok]); out[~ok] = U[near]
    return out, ok
