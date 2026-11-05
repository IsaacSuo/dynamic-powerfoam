"""where our inertia goes: GT particles (the material, uniform mass) assigned to our cells by power distance
(argmin |x - p|^2 - r^2), so each cell's own material centroid; the moment of inertia about the clamp's centre for the
sway (axis horizontal, normal to the kick's direction) and the polar second moment about the centre of mass, for
 (a) the GT particles, (b) our masses V at the sites (the model), (c) our masses V at their cells' material centroids;
(b)/(a) = the model's inertia against the material's, split into (c)/(a) (partition, shape) and (b)/(c) (site vs centroid)"""
import numpy as np
from pathlib import Path
from plyfile import PlyData
from scipy.spatial import cKDTree
ROOT = Path(__file__).resolve().parents[1]
OFF = np.array([.5, -.05, .5])
c_ = np.load(ROOT/'inverse_data/pacnerf_e0/foam/dem_exact_single/cells.npz'); p, r = c_['points'].astype(np.float64), c_['radii'].astype(np.float64)
V = np.load(ROOT/'inverse_data/pacnerf_e0/foam/dem_exact_single/geometry.npz')['masses']
P = np.stack([PlyData.read('/home/fangsuo/py/dynamic-powerfoam/data/pacnerf/simulation_data/elastic/0/0.ply')['vertex'][c] for c in 'xyz'], 1).astype(np.float64)+OFF
R = r.max(); lift = np.c_[p, np.sqrt(R*R-r*r)]
_, own = cKDTree(lift).query(np.c_[P, np.zeros(len(P))])
cnt = np.bincount(own, minlength=len(p)); cen = np.stack([np.bincount(own, P[:, k], len(p)) for k in range(3)], 1)/np.maximum(cnt, 1)[:, None]
empty = cnt == 0; cen[empty] = p[empty]
off = np.linalg.norm(cen-p, axis=1)
print('cells without GT particles %d; site-centroid offset (mm): median %.1f, 90 %% %.1f; surface vs interior by count: %s'
      % (empty.sum(), 1e3*np.median(off), 1e3*np.percentile(off, 90), ''))
Sz = np.load(ROOT/'out_inverse/clamp/sets_hc04.npz'); piv = P[Sz['clamp_m']].mean(0)
kd = np.array([-0.99, 0., -0.16]); kd /= np.linalg.norm(kd); ax = np.cross(kd, [0., 1., 0.]); ax /= np.linalg.norm(ax)


def J(X, m, o, a=None):
    d = X-o; return (m*((d*d).sum(1)-((d@a)**2 if a is not None else 0))).sum()/m.sum()


mP = np.ones(len(P))
for name, o, a in [('sway about the clamp', piv, ax), ('polar about the com', None, None)]:
    oa = P.mean(0) if o is None else o; ob = (V[:, None]*p).sum(0)/V.sum() if o is None else o; oc = (V[:, None]*cen).sum(0)/V.sum() if o is None else o
    Ja, Jb, Jc = J(P, mP, oa, a), J(p, V, ob, a), J(cen, V, oc, a)
    print('%-22s per unit mass: GT %.5f | ours at sites %.5f | ours at centroids %.5f -> model/material %.3f = partition %.3f x site %.3f'
          % (name, Ja, Jb, Jc, Jb/Ja, Jc/Ja, Jb/Jc))
print('volume share by particle count against V: corr %.3f, sum V %.5f' % (np.corrcoef(cnt, V)[0, 1], V.sum()))
