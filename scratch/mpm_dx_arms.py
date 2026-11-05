"""the two upper arms' tip displacement in the body frame (mpm_dx_summary.py's construction), projected on each tip's main
direction of motion in GT (largest-variance axis over frames 5-13, sign: GT's frame-8 value negative), mm, frames 5..13"""
import sys
import numpy as np
sys.argv, args = sys.argv[:1], sys.argv[1:]
import mpm_dx_summary as S
ref = S.summary(S.M, np.ones(len(S.M[0])))['tip'][:, S.upper]          # (14, 2, 3)
axes = []
for j in range(2):
    T = ref[5:14, j]; _, _, Vt = np.linalg.svd(T-T.mean(0)); a = Vt[0]
    axes.append(a if ref[8, j]@a < 0 else -a)
runs = [('GT (dx 0.020)', ref)]
for p in args:
    z = np.load(p)
    if 'dx' in z:
        X = z['x'].astype(np.float64)[:14]+S.OFF; runs.append(('MPM dx %.4f%s' % (float(z['dx']), ' s' if int(z['split']) == 2 else ''), S.summary(list(X), np.ones(X.shape[1]))['tip'][:, S.upper]))
    else:
        runs.append(('DEM beta %.4f' % float(np.exp(z['theta'][2])), S.summary(list(z['x'].astype(np.float64)), S.V)['tip'][:, S.upper]))
for j in range(2):
    print('upper arm %d, frames 5..13 (mm)' % j)
    for n, t in runs:
        print('  %-16s %s' % (n, ' '.join('%+6.0f' % (t[k, j]@axes[j]) for k in range(5, 14))))
