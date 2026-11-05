"""the clamped test's sets, MPM (GT particles) and ours (cells) from the same rest body (frame 0, scene frame):
  clamp: the GT particles within hc of the ground at frame 6 (the impact's contact region), taken back to frame 0; a cell is
         clamped when its centre's nearest GT particle (frame 0) is
  groups: all, the 4 arm tips (mpm_dx_summary.sets: radius > .75 of the reach, within 22.5 deg of the arm's axis), the core
args: hc (m), out npz"""
import sys
import numpy as np
from scipy.spatial import cKDTree
hc, out = float(sys.argv[1]), sys.argv[2]
sys.argv = sys.argv[:1]
import mpm_dx_summary as S
G0w = S.M[0]-S.OFF                                                                     # (world frame, ground y = 0)
from plyfile import PlyData
G6 = np.stack([PlyData.read('/home/fangsuo/py/dynamic-powerfoam/data/pacnerf/simulation_data/elastic/0/6.ply')['vertex'][c] for c in 'xyz'], 1)
cl_m = G6[:, 1] < hc
X = np.load(S.ROOT/'inverse_data'/'pacnerf_e0'/'foam'/'dem_exact_single'/'cells.npz')['points'].astype(np.float64)   # (scene frame)
_, nn = cKDTree(S.M[0]).query(X)
cl_d = cl_m[nn]
core_m, tips_m = S.sets(S.M[0]); core_d, tips_d = S.sets(X)
lower = [j for j in range(4) if j not in S.upper]
print('hc %.3f: clamped MPM particles %d (%.2f %% of the mass), our cells %d (%.2f %% of the volume); upper arms %s, lower %s' % (
    hc, cl_m.sum(), 100*cl_m.mean(), cl_d.sum(), 100*S.V[cl_d].sum()/S.V.sum(), list(S.upper), lower))
for j in range(4):
    print('  tip %d: MPM %d particles (clamped %d), ours %d cells (clamped %d)' % (j, tips_m[j].sum(), (tips_m[j] & cl_m).sum(), tips_d[j].sum(), (tips_d[j] & cl_d).sum()))
np.savez(out, hc=hc, clamp_m=cl_m, clamp_d=cl_d, groups_m=np.stack([np.ones_like(cl_m)]+list(tips_m)+[core_m]),
         groups_d=np.stack([np.ones_like(cl_d)]+list(tips_d)+[core_d]), upper=np.array(S.upper), lower=np.array(lower))
