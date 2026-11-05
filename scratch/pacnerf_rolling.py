"""rolling test, MPM against ours: per frame interval k -> k+1 (24 fps, forward differences), along the horizontal
direction of v0 ('forward'):
  v_c    the centre of mass's velocity
  a_k    the ground's horizontal force per unit mass at frame k, (v_c[k, k+1] - v_c[k-1, k]) * 24  (exact: gravity and the
         internal forces have no net horizontal part); sign + = pushes forward, - = brakes
  u_p    the contact layer's own velocity (points whose lowest part is within 5.5 mm of the ground at both ends)
  split  u_p = v_c + spin + def: spin = the whole body's best-fit rotation R (Kabsch, k -> k+1) carried to the contact
         points, (R - I)(p_k - c_k) * 24; def = the rest, the contact layer's motion relative to the rigid fit
  ratio  u_p / v_c: 1 = slides with no spin, 0 = rolls, < 0 = the bottom runs backward (spins faster than rolling)"""
import sys
from pathlib import Path
import numpy as np
from plyfile import PlyData
ROOT = Path(__file__).resolve().parents[1]
OFF = np.array([.5, -.05, .5]); G0 = -.05; FPS = 24; LAYER = 5.5e-3
ef = np.array([-0.4078950600549248, 0., -0.06629882732047176]); ef /= np.linalg.norm(ef)
V = np.load(ROOT/'inverse_data'/'pacnerf_e0'/'foam'/'dem_exact_single'/'geometry.npz')['masses']
R0 = np.load(ROOT/'inverse_data'/'pacnerf_e0'/'foam'/'dem_exact_single'/'cells.npz')['radii']
M = [np.stack([PlyData.read(str(Path('/home/fangsuo/py/dynamic-powerfoam/data/pacnerf/simulation_data/elastic/0')/('%d.ply' % k)))['vertex'][a] for a in 'xyz'], 1)+OFF for k in range(14)]


def kabsch(P0, P, w):
    a0, a = (w[:, None]*P0).sum(0), (w[:, None]*P).sum(0)
    U_, _, Vt = np.linalg.svd(((P-a)*w[:, None]).T@(P0-a0)); D = np.diag([1., 1., np.sign(np.linalg.det(U_@Vt))])
    return U_@D@Vt, a0, a


def table(name, Ps, w, bottom):
    w = w/w.sum(); rows = []
    for k in range(4, len(Ps)-1):
        R, c0, c1 = kabsch(Ps[k], Ps[k+1], w); vc = (c1-c0)*FPS
        on = (Ps[k][:, 1]-bottom-G0 < LAYER) & (Ps[k+1][:, 1]-bottom-G0 < LAYER)
        if on.any():
            wo = w[on]/w[on].sum(); r = Ps[k][on]-c0
            up = (wo[:, None]*(Ps[k+1][on]-Ps[k][on])).sum(0)*FPS
            spin = (wo[:, None]*(r@(R-np.eye(3)).T)).sum(0)*FPS
            rows.append((k, on.sum(), vc@ef, spin@ef, (up-vc-spin)@ef, up@ef))
        else:
            rows.append((k, 0, vc@ef, np.nan, np.nan, np.nan))
    print('== %s' % name)
    print(' k->k+1 | contact pts | a_k (m/s^2) | v_c | spin | def | u_p | ratio u_p/v_c   (forward components, m/s)')
    for i, (k, n, vc, sp, de, up) in enumerate(rows):
        a = (vc-rows[i-1][2])*FPS if i > 0 else np.nan
        print(' %2d->%2d | %6d | %+6.2f | %+.3f | %s' % (k, k+1, n, a, vc, ('%+.3f | %+.3f | %+.3f | %+.2f' % (sp, de, up, up/vc)) if n else 'no contact'))


if __name__ == '__main__':
  table('MPM', M, np.ones(len(M[0])), 0.)
  for p in sys.argv[1:]:
    z = np.load(p); table('%s (beta %.4f)' % (Path(p).parent.name, float(np.exp(z['theta'][2]))), list(z['x'].astype(np.float64)), V, R0)
