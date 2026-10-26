"""Render a surface_mesh.npz (PAC-NeRF world, y up) from six directions, faces shaded by their normal (PNG next to it)."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

S = np.load(sys.argv[1]); out = Path(sys.argv[2])
V, off, fv = S['vert_world'], S['face_off'], S['face_vert']
tris = []
for f in range(len(off)-1):
    P = V[fv[off[f]:off[f+1]]]; c = P.mean(0)
    for k in range(len(P)):
        tris.append((c, P[k], P[(k+1) % len(P)]))
T = np.array(tris)
n = np.cross(T[:, 1]-T[:, 0], T[:, 2]-T[:, 0]); n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
fig = plt.figure(figsize=(18, 11))
views = [(20, -60), (20, 30), (20, 120), (20, 210), (85, 0), (-85, 0)]
lo, hi = V.min(0), V.max(0); mid = (lo+hi)/2; R = (hi-lo).max()/2
for i, (el, az) in enumerate(views):
    ax = fig.add_subplot(2, 3, i+1, projection='3d')
    e = np.radians(el); a = np.radians(az)
    view = np.array([np.cos(e)*np.cos(a), np.cos(e)*np.sin(a), np.sin(e)])
    vz = np.array([view[0], view[2], view[1]])                      # matplotlib z-up <- world y-up
    shade = .25+.75*np.clip((n@np.array([.4, .8, .45])), 0, 1)
    pc = Poly3DCollection(T[:, :, [0, 2, 1]], facecolors=plt.cm.Blues(.35+.6*shade), edgecolors='none', linewidths=0)
    ax.add_collection3d(pc)
    ax.set_xlim(mid[0]-R, mid[0]+R); ax.set_ylim(mid[2]-R, mid[2]+R); ax.set_zlim(mid[1]-R, mid[1]+R)
    ax.view_init(el, az); ax.set_axis_off(); ax.set_title('elev %d azim %d' % (el, az))
plt.tight_layout(); plt.savefig(out, dpi=60)
print('wrote', out)
