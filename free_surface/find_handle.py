"""Where is the handle of a genus-g surface_mesh.npz? Greedy homotopy generators (Erickson & Whittlesey 2005) from many
basepoints: shortest-path tree T from the basepoint, a maximum spanning tree of the dual graph over the edges not in T
weighted by the loop length d(x, u) + |uv| + d(x, v); the 2g edges left over each close a non-contractible loop through
the basepoint. The shortest loop found over the basepoints locates the handle; it is drawn over the surface (PNG)."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import dijkstra, minimum_spanning_tree

S = np.load(sys.argv[1]); out = Path(sys.argv[2]); nb = int(sys.argv[3]) if len(sys.argv) > 3 else 40
V, off, fv = S['vert_world'], S['face_off'], S['face_vert']
F = len(off)-1; nv = len(V)
face_of = np.repeat(np.arange(F), np.diff(off))
nxt = np.empty_like(fv)
for f0, f1 in zip(off[:-1], off[1:]):
    nxt[f0:f1] = np.roll(fv[f0:f1], -1)
a, b = np.minimum(fv, nxt), np.maximum(fv, nxt)
key = a.astype(np.int64)*nv+b
ekeys, einv = np.unique(key, return_inverse=True)
eu, ev = (ekeys//nv).astype(int), (ekeys % nv).astype(int)
E = len(ekeys)
ef = np.full((E, 2), -1)
for i, (e, f) in enumerate(zip(einv, face_of)):
    ef[e, 0 if ef[e, 0] < 0 else 1] = f
L = np.linalg.norm(V[eu]-V[ev], axis=1)+1e-12
G = sparse.coo_matrix((np.r_[L, L], (np.r_[eu, ev], np.r_[ev, eu])), shape=(nv, nv)).tocsr()
print('vertices %d, edges %d, faces %d, genus %d' % (nv, E, F, (2-(nv-E+F))//2), flush=True)
rng = np.random.default_rng(0)
best = None
for x in rng.choice(nv, nb, replace=False):
    d, pred = dijkstra(G, indices=int(x), return_predecessors=True)
    tree = (pred[eu] == ev) | (pred[ev] == eu)
    sigma = d[eu]+L+d[ev]
    nt = np.flatnonzero(~tree)
    W = sigma.max()+1-sigma[nt]                                 # maximum spanning tree by minimising the complement
    D = sparse.coo_matrix((W, (ef[nt, 0], ef[nt, 1])), shape=(F, F)).tocsr()
    mst = minimum_spanning_tree(D+D.T).tocoo()
    used = set(zip(np.minimum(mst.row, mst.col), np.maximum(mst.row, mst.col)))
    left = [e for e in nt if (min(ef[e]), max(ef[e])) not in used]
    for e in left:
        if best is None or sigma[e] < best[0]:
            path = lambda s: ([s] + (path(pred[s]) if pred[s] >= 0 else []))
            pu, pv = [], []
            s = eu[e]
            while s >= 0:
                pu.append(s); s = pred[s]
            s = ev[e]
            while s >= 0:
                pv.append(s); s = pred[s]
            best = (sigma[e], int(x), pu[::-1]+pv)
print('shortest non-contractible loop found: %.1f mm, %d vertices (basepoint %d)' % (1e3*best[0], len(best[2]), best[1]), flush=True)
loop = V[best[2]]
# the loop's part farthest from the basepoint is where it wraps the handle
far = loop[np.argmax(np.linalg.norm(loop-loop[0], axis=1))]
print('loop bounding box (world, m): %s .. %s; farthest point from the basepoint %s' % (np.round(loop.min(0), 3).tolist(), np.round(loop.max(0), 3).tolist(), np.round(far, 3).tolist()), flush=True)
fig = plt.figure(figsize=(14, 7))
for i, (el, az) in enumerate([(20, -60), (20, 120)]):
    ax = fig.add_subplot(1, 2, i+1, projection='3d')
    ax.scatter(V[::7, 0], V[::7, 2], V[::7, 1], s=.3, c='0.75')
    ax.plot(loop[:, 0], loop[:, 2], loop[:, 1], 'r-', lw=2)
    ax.view_init(el, az); ax.set_axis_off()
plt.tight_layout(); plt.savefig(out, dpi=60)
np.save(out.with_suffix('.npy'), loop)
