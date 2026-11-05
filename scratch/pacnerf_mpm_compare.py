"""our PAC-NeRF scene 0 forward (show_frames.npz) against the MPM ground truth particles (simulation_data/elastic/0/<k>.ply,
frame k at k / 24 s, world frame; ours in the scene frame = world + (0.5, -0.05, 0.5)): per frame the centre of mass
(ours mass-weighted by cell volume, MPM's particles equal weight), its velocity, the rigid rotation since frame 0 by
Kabsch (angle and axis), the height above the ground, the spreads"""
import sys
from pathlib import Path
import numpy as np
from plyfile import PlyData
ROOT = Path(__file__).resolve().parents[1]
run = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT/'out_inverse'/'pacnerf_e0_show'/'show_frames.npz'
OFF = np.array([.5, -.05, .5])
z = np.load(run); X = z['x'].astype(np.float64)
V = np.load(ROOT/'inverse_data'/'pacnerf_e0'/'foam'/'dem_exact_single'/'geometry.npz')['masses']
K = len(X)
mpm = []
for k in range(K):
    v = PlyData.read(str(Path('/home/fangsuo/py/dynamic-powerfoam/data/pacnerf/simulation_data/elastic/0')/('%d.ply' % k)))['vertex']
    mpm.append(np.stack([v['x'], v['y'], v['z']], 1)+OFF)


def kabsch(P0, P):
    c0, c = P0.mean(0), P.mean(0)
    U, _, Vt = np.linalg.svd((P-c).T@(P0-c0))
    D = np.diag([1., 1., np.sign(np.linalg.det(U@Vt))])
    R = U@D@Vt
    ang = np.degrees(np.arccos(np.clip((np.trace(R)-1)/2, -1, 1)))
    ax = np.array([R[2, 1]-R[1, 2], R[0, 2]-R[2, 0], R[1, 0]-R[0, 1]]); ax = ax/max(np.linalg.norm(ax), 1e-12)
    return ang, ax


def com_w(P, w):
    return (w[:, None]*P).sum(0)/w.sum()


G0 = -.05
print('frame | com y above the ground: MPM, ours | com x: MPM, ours | com z: MPM, ours | rotation since frame 0 (deg): MPM, ours | axis MPM / ours')
cm_m, cm_o = [], []
for k in range(K):
    cm = mpm[k].mean(0); co = com_w(X[k], V); cm_m.append(cm); cm_o.append(co)
    am, axm = kabsch(mpm[0], mpm[k]); ao, axo = kabsch(X[0], X[k])
    print('%5d | %.4f %.4f | %+.4f %+.4f | %+.4f %+.4f | %6.1f %6.1f | %s / %s' % (
        k, cm[1]-G0, co[1]-G0, cm[0], co[0], cm[2], co[2], am, ao, np.round(axm, 2).tolist(), np.round(axo, 2).tolist()))
cm_m, cm_o = np.array(cm_m), np.array(cm_o)
vm, vo = np.diff(cm_m, axis=0)*24, np.diff(cm_o, axis=0)*24
print('velocity of the centre of mass (m/s, between frames): frame | MPM vx vy vz | ours vx vy vz')
for k in range(K-1):
    print('%2d-%2d | %+.3f %+.3f %+.3f | %+.3f %+.3f %+.3f' % (k, k+1, *vm[k], *vo[k]))
