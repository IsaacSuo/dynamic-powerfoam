"""the ground's impulse on the jack, MPM against ours (10-26's open step: where does MPM's ground act?), per unit mass:
velocities by central differences of the frames (24 fps), the centre of mass's velocity v_c and the angular momentum
about it l = sum m (x - c) x (v - v_c) / M; from frame 4 (before contact) to frame k the ground's horizontal impulse
dv_x (gravity has none) and torque impulse dl (gravity has none about c); the impulse's equivalent height below the
centre h = dl_z / dv_x; also the vertical impulse dv_y + g dt and the moment it would carry about z at a lever
(x_p - x_c): dl_z = h dv_x + (x_p - x_c)(dv_y + g t), so with h fixed at the centre's height above the ground the
vertical impulse's lever follows"""
import sys
from pathlib import Path
import numpy as np
from plyfile import PlyData
ROOT = Path(__file__).resolve().parents[1]
OFF = np.array([.5, -.05, .5]); G0 = -.05; DTF = 1/24
V = np.load(ROOT/'inverse_data'/'pacnerf_e0'/'foam'/'dem_exact_single'/'geometry.npz')['masses']
M = [np.stack([PlyData.read(str(Path('/home/fangsuo/py/dynamic-powerfoam/data/pacnerf/simulation_data/elastic/0')/('%d.ply' % k)))['vertex'][a] for a in 'xyz'], 1)+OFF for k in range(14)]


def series(Ps, w):
    w = w/w.sum(); out = []
    for k in range(1, len(Ps)-1):
        v = (Ps[k+1]-Ps[k-1])/(2*DTF); c = (w[:, None]*Ps[k]).sum(0); vc = (w[:, None]*v).sum(0)
        l = (w[:, None]*np.cross(Ps[k]-c, v-vc)).sum(0)
        Iz = (w*((Ps[k][:, 0]-c[0])**2+(Ps[k][:, 1]-c[1])**2)).sum()
        out.append((k, c, vc, l, Iz))
    return out


runs = [('MPM', series(M, np.ones(len(M[0]))))]
for p in sys.argv[1:]:
    z = np.load(p); runs.append(('beta %.4f' % float(np.exp(z['theta'][2])), series(list(z['x'].astype(np.float64)), V)))
for name, S in runs:
    k4 = [s for s in S if s[0] == 4][0]
    print('== %s (per unit mass; dv, dl from frame 4)' % name)
    print(' k | com height | v_c x, y | dv_x | dl_z (m^2/s) | h = dl_z / dv_x (m) | omega_z = l_z / (I_z / M) | dv_y + g t')
    for k, c, vc, l, Iz in S:
        if k < 4:
            continue
        dvx = vc[0]-k4[2][0]; dl = l-k4[3]; dvy = vc[1]-k4[2][1]+9.8*(k-4)*DTF
        h = dl[2]/dvx if abs(dvx) > 1e-3 else float('nan')
        print('%2d | %.4f | %+.3f %+.3f | %+.3f | %+.5f | %s | %+.2f | %+.3f' % (k, c[1]-G0, vc[0], vc[1], dvx, dl[2], ('%+.3f' % h) if h == h else '  -   ', l[2]/Iz, dvy))
