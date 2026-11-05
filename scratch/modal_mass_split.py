"""the modal mass along the voxel FEM's elastic modes (talk-pages round01_exact, h 0.008, interpolated trilinearly as
pacnerf_modes.py), unit density: (a) the material, GT particles each 1/N of our total volume, (b) our model, V at the
sites, (c) V at the cells' material centroids (GT particles assigned by power distance); FEM's own sum m phi^2 for
reference. A Rayleigh frequency goes with 1 / sqrt(modal mass): (b)/(a) < 1 makes our modes fast with the stiffness right"""
import numpy as np
from pathlib import Path
from plyfile import PlyData
from scipy.spatial import cKDTree
ROOT = Path(__file__).resolve().parents[1]
OFF = np.array([.5, -.05, .5])
fem = dict(np.load('/home/fangsuo/py/dynamic-powerfoam/.claude/worktrees/talk-pages/output/round01_exact/fem_modes_h0.008.npz'))


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


c_ = np.load(ROOT/'inverse_data/pacnerf_e0/foam/dem_exact_single/cells.npz'); p, r = c_['points'].astype(np.float64), c_['radii'].astype(np.float64)
V = np.load(ROOT/'inverse_data/pacnerf_e0/foam/dem_exact_single/geometry.npz')['masses']
P = np.stack([PlyData.read('/home/fangsuo/py/dynamic-powerfoam/data/pacnerf/simulation_data/elastic/0/0.ply')['vertex'][c] for c in 'xyz'], 1).astype(np.float64)+OFF
R = r.max(); _, own = cKDTree(np.c_[p, np.sqrt(R*R-r*r)]).query(np.c_[P, np.zeros(len(P))])
cnt = np.bincount(own, minlength=len(p)); cen = np.stack([np.bincount(own, P[:, k], len(p)) for k in range(3)], 1)/np.maximum(cnt, 1)[:, None]
cen[cnt == 0] = p[cnt == 0]
K = 12
phiP, okP = interpolate(fem, fem['V'][:, :K], P); phiS, okS = interpolate(fem, fem['V'][:, :K], p); phiC, okC = interpolate(fem, fem['V'][:, :K], cen)
print('inside the voxel mesh: GT particles %.3f, sites %.3f, centroids %.3f' % (okP.mean(), okS.mean(), okC.mean()))
mP = V.sum()/len(P)
Ma = mP*(phiP**2).sum((0, 1)); Mb = (V[:, None]*(phiS**2).sum(1)).sum(0); Mc = (V[:, None]*(phiC**2).sum(1)).sum(0)
Mf = (fem['masses'][:, None]*(fem['V'][:, :K].reshape(len(fem['masses']), 3, K)**2).sum(1)).sum(0)
print('mode | FEM f (Hz) | modal mass: FEM nodes | material (GT particles) | ours at sites | ours at centroids | sites / material -> f x')
for j in range(K):
    print('%4d | %6.3f | %.3f | %.3f | %.3f | %.3f | %.3f -> %.3f' % (j, fem['f'][j], Mf[j], Ma[j], Mb[j], Mc[j], Mb[j]/Ma[j], np.sqrt(Ma[j]/Mb[j])))
