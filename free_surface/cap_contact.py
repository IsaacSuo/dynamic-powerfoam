"""The ground acting on the exposed caps only (11-06, user: no Hertz). The body is the union of the cells' spheres, its
surface the caps: the sphere points outside every other sphere (step 0, scratch/cap_bottom_check.py: the jack's bottom is
all exposed caps facing down, its lowest point 0.5 mm off the GT's). Each cap is sampled at rest by Fibonacci directions
on its sphere (area weight 4 pi r^2 / N per kept point); the points move with their cell, x_q = x_B + F_B (X_q - X_B).
The contact is a stiff penalty on each point's penetration and a glue at each point (ClosureWarp.set_cap_contact): its
compliance comes from the body, not from a Hertz ball whose sphere is mostly other cells' material (median 1.1 % of a
contact ball is its own exposed cap)."""
import hashlib
import os

import numpy as np
from scipy.spatial import cKDTree


def cap_quadrature(p, r, cells, N=2000, cache=None, verbose=True):
    """the exposed caps of `cells` sampled: (cell (Q,), offset X_q - X_B (Q, 3), area weight (Q,)); cache: an npz path,
    keyed by the geometry"""
    p, r = np.asarray(p, np.float64), np.asarray(r, np.float64)
    cells = np.asarray(cells, np.int64)
    key = hashlib.sha1(np.concatenate([p.ravel(), r, cells.astype(np.float64), [N]]).tobytes()).hexdigest()
    if cache is not None and os.path.exists(cache):
        z = np.load(cache)
        if str(z['key']) == key:
            return z['cell'], z['off'], z['w']
    k = np.arange(N)+.5; ph = np.arccos(1-2*k/N); th = np.pi*(1+5**.5)*k
    U = np.c_[np.cos(th)*np.sin(ph), np.cos(ph), np.sin(th)*np.sin(ph)]
    tree = cKDTree(p); rmax = r.max()
    cl, off, w = [], [], []
    for i in cells:
        x = p[i]+r[i]*U
        nb = np.array([j for j in tree.query_ball_point(p[i], r[i]+rmax) if j != i], dtype=np.int64)
        keep = np.ones(N, bool)
        if len(nb):
            near = nb[np.linalg.norm(p[nb]-p[i], axis=1) < r[i]+r[nb]]
            for s in range(0, len(near), 64):
                b = near[s:s+64]
                keep &= (((x[:, None, :]-p[b][None])**2).sum(-1) >= r[b][None]**2).all(1)
        cl.append(np.full(keep.sum(), i)); off.append(r[i]*U[keep]); w.append(np.full(keep.sum(), 4*np.pi*r[i]**2/N))
    cl, off, w = np.concatenate(cl), np.concatenate(off), np.concatenate(w)
    if verbose:
        cnt = np.bincount(cl, minlength=len(p))[cells]
        print('  cap quadrature: %d points on %d caps (per cap median %d, %d caps without a point), area %.4f m^2' % (
            len(cl), len(cells), int(np.median(cnt)), (cnt == 0).sum(), w.sum()), flush=True)
    if cache is not None:
        tmp = '%s.%d.tmp.npz' % (cache, os.getpid())
        np.savez(tmp, key=key, cell=cl, off=off, w=w)
        os.replace(tmp, cache)
    return cl, off, w
