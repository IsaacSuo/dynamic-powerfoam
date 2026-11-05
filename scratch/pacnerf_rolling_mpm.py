"""MPM's contact layer as its sticky ground sees it: the floor nodes (y <= 0, velocity set to 0 every substep) reach every
particle whose quadratic B-spline stencil holds them, y < 1.5 dx; the weight of the floor node w0(y/dx) = 0.75 - x^2 (x < .5),
.5 (1.5 - x)^2 (x < 1.5); dx = 0.0104 from the particle count (p_vol = (dx/2)^3, our volume). Per interval k -> k+1 the
w0-weighted forward velocity of those particles (u_p) and its split as in pacnerf_rolling.py"""
import sys
from pathlib import Path
import numpy as np
from plyfile import PlyData
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pacnerf_rolling import M, kabsch, ef, FPS, G0
DX = float(sys.argv[1]) if len(sys.argv) > 1 else .0104
w = np.ones(len(M[0]))/len(M[0])


def w0(y):
    x = np.maximum(y, 0)/DX
    return np.where(x < .5, .75-x**2, np.where(x < 1.5, .5*(1.5-x)**2, 0.))


print('dx %.4f | k->k+1 | sum w0 (particle equivalents) | a_k | v_c | spin | def | u_p | ratio' % DX)
prev = None
for k in range(4, 13):
    R, c0, c1 = kabsch(M[k], M[k+1], w); vc = (c1-c0)*FPS
    a = (vc@ef-prev)*FPS if prev is not None else np.nan; prev = vc@ef
    q = .5*(w0(M[k][:, 1]-G0)+w0(M[k+1][:, 1]-G0))
    if q.sum() > 0:
        q_ = q/q.sum(); r = M[k]-c0
        up = (q_[:, None]*(M[k+1]-M[k])).sum(0)*FPS; sp = (q_[:, None]*(r@(R-np.eye(3)).T)).sum(0)*FPS
        print(' %2d->%2d | %7.1f | %+6.2f | %+.3f | %+.3f | %+.3f | %+.3f | %+.2f' % (k, k+1, q.sum(), a, vc@ef, sp@ef, (up-vc-sp)@ef, up@ef, (up@ef)/(vc@ef)))
    else:
        print(' %2d->%2d |       0 | %+6.2f | %+.3f | no layer' % (k, k+1, a, vc@ef))
