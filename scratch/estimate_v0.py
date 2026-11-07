"""the start velocity of a PAC-NeRF scene from its photos (11-07, for torus / bird, which have no published truth): per
view and frame the object's mask by background subtraction against the empty scene (frame -1), its centroid; per
frame the 3D point nearest all views' centroid rays (least squares); then c(t) = c0 + v0 t - g t^2 / 2 (g 9.8 along
-y, t = k / 24 s) fitted to frames 0..hit-1 (PAC-NeRF's hit_frame 5). Checks: frame 0's subtraction mask against the
training mask (IoU), c0 against the converted foam's volume centroid, and on scenes with a truth, v0 against
gt/elastic.json. Scene frame = PAC-NeRF world + (0.5, -0.05, 0.5); velocities are frame-independent.
  python scratch/estimate_v0.py <photos dir> <transforms_all.json> <frame-0 masks dir> <foam dem_exact_single dir> [truth v]"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

photos, trf, mdir, foam = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
truth = np.array([float(x) for x in sys.argv[5:8]]) if len(sys.argv) > 7 else None
import os
HIT, FPS, THR = int(os.environ.get("HIT", "5")), 24, float(os.environ.get("THR", ".12"))
tr = json.loads(trf.read_text()); W = int(tr['w']); H = int(tr['h']); f = .5*W/math.tan(.5*tr['camera_angle_x'])
views = []
for fr in tr['frames']:
    nm = Path(fr['file_path']).name.split('_')[1]
    c2w = np.array(fr['transform_matrix'], float)
    views.append((nm, c2w))
load = lambda nm, k: np.asarray(Image.open(photos/('r_%s_%d.png' % (nm, k))).convert('RGB'), np.float32)/255.


def mask(nm, k):
    d = np.abs(load(nm, k)-load(nm, -1)).max(-1) > THR
    d = ndimage.binary_opening(d, iterations=1)
    lab, nl = ndimage.label(d)
    if nl == 0:
        return d
    big = np.argmax(np.bincount(lab.ravel())[1:])+1
    return lab == big


def ray(nm, c2w, px, py):
    """the camera ray through pixel (px, py): origin, unit direction (OpenGL c2w: camera looks along -z, y up)"""
    d = np.array([(px-(W-1)/2)/f, -(py-(H-1)/2)/f, -1.])
    d = c2w[:3, :3]@d
    return c2w[:3, 3], d/np.linalg.norm(d)


ious = []
for nm, c2w in views:
    m0 = np.asarray(Image.open(mdir/('r_%s_0.png' % nm)), np.float32)/255. > .5
    m = mask(nm, 0)
    ious.append((m & m0).sum()/max((m | m0).sum(), 1))
print('frame 0: subtraction mask vs training mask IoU per view %s' % ' '.join('%.3f' % u for u in ious))
C = []
for k in range(HIT):
    A = np.zeros((3, 3)); b = np.zeros(3)
    for nm, c2w in views:
        m = mask(nm, k); ys, xs = np.nonzero(m)
        o, d = ray(nm, c2w, xs.mean(), ys.mean())
        P = np.eye(3)-np.outer(d, d); A += P; b += P@o
    C.append(np.linalg.solve(A, b))
C = np.array(C); t = np.arange(HIT)/FPS
# c(t) - (-g t^2 / 2 along y) = c0 + v0 t
Y = C.copy(); Y[:, 1] += .5*9.8*t**2
G = np.stack([np.ones_like(t), t], 1)
sol, *_ = np.linalg.lstsq(G, Y, rcond=None); c0, v0 = sol
res = Y-G@sol
g = np.load(foam/'geometry.npz'); cf = (g['points']*g['masses'][:, None]).sum(0)/g['masses'].sum()
print('silhouette-centroid triangulation, frames 0..%d (scene frame, m): %s' % (HIT-1, np.round(C, 4).tolist()))
print('fit: c0 %s, v0 %s m/s, residual max %.2f mm' % (np.round(c0, 4).tolist(), np.round(v0, 4).tolist(), 1e3*np.abs(res).max()))
print('foam volume centroid %s (c0 - it: %s mm)' % (np.round(cf, 4).tolist(), np.round(1e3*(c0-cf), 1).tolist()))
if truth is not None:
    print('truth v0 %s: error %s m/s (|err| %.3f)' % (np.round(truth, 4).tolist(), np.round(v0-truth, 4).tolist(), np.linalg.norm(v0-truth)))
