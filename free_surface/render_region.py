"""Zoom on a region of a surface_mesh.npz: faces within --radius of a centre, shaded, cut open, with an optional loop (.npy)
and the material / air sites there; six views (PNG)."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

a = argparse.ArgumentParser()
a.add_argument('mesh'); a.add_argument('out')
a.add_argument('--centre', type=float, nargs=3, required=True)
a.add_argument('--radius', type=float, default=.03)
a.add_argument('--loop', default=None)
args = a.parse_args()
S = np.load(args.mesh)
V, off, fv = S['vert_world'], S['face_off'], S['face_vert']
c = np.array(args.centre)
keep = [f for f in range(len(off)-1) if np.linalg.norm(S['centroid_world'][f]-c) < args.radius]
polys = [V[fv[off[f]:off[f+1]]] for f in keep]
an = S['area_vec_world'][keep]; an = an/np.maximum(np.linalg.norm(an, axis=1, keepdims=True), 1e-30)
shade = .3+.7*np.clip(an@np.array([.4, .8, .45]), 0, 1)
loop = np.load(args.loop) if args.loop else None
fig = plt.figure(figsize=(18, 11))
for i, (el, az) in enumerate([(20, -60), (20, 30), (20, 120), (20, 210), (85, 0), (-85, 0)]):
    ax = fig.add_subplot(2, 3, i+1, projection='3d')
    pc = Poly3DCollection([P[:, [0, 2, 1]] for P in polys], facecolors=plt.cm.Blues(.3+.6*shade), edgecolors='k', linewidths=.2, alpha=.9)
    ax.add_collection3d(pc)
    if loop is not None:
        ax.plot(loop[:, 0], loop[:, 2], loop[:, 1], 'r-', lw=2.5)
    r = args.radius
    ax.set_xlim(c[0]-r, c[0]+r); ax.set_ylim(c[2]-r, c[2]+r); ax.set_zlim(c[1]-r, c[1]+r)
    ax.view_init(el, az); ax.set_title('elev %d azim %d (faces within %.0f mm)' % (el, az, 1e3*r))
plt.tight_layout(); plt.savefig(args.out, dpi=60)
print('faces drawn', len(keep))
