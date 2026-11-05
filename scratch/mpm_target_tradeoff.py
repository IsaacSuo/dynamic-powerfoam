"""our beta runs against two targets: GT (MPM dx 0.02) and the converged MPM (first argument, dx 0.0078 split): the upper
arms' tip displacement (mpm_dx_arms.py's axes), RMS of the difference over frames 6-13 (mm), the time the swing crosses
zero after its frame-8 trough (frames, linear between frames), and the centre's height at frame 12 (m)"""
import sys
import numpy as np
sys.argv, args = sys.argv[:1], sys.argv[1:]
import mpm_dx_summary as S
import mpm_dx_arms as A


def series(p):
    z = np.load(p)
    if 'dx' in z:
        X = z['x'].astype(np.float64)[:14]+S.OFF; s = S.summary(list(X), np.ones(X.shape[1]))
    else:
        s = S.summary(list(z['x'].astype(np.float64)), S.V)
    return np.stack([s['tip'][:, S.upper][:, j]@A.axes[j] for j in range(2)], 1), s['h'][12]


def cross(t):
    out = []
    for j in range(2):
        k = next(k for k in range(8, 13) if t[k, j] < 0 <= t[k+1, j]); out.append(k+(-t[k, j])/(t[k+1, j]-t[k, j]))
    return out


gt = (np.stack([A.ref[:, j]@A.axes[j] for j in range(2)], 1), S.ref['h'][12])
cv = series(args[0])
print('#####')
print('target         | zero crossing arm 0 / arm 1 (frame) | height 12 (m)')
for n, (t, h) in [('GT dx 0.020', gt), ('MPM dx 0.0078', cv)]:
    c = cross(t); print('%-15s| %.2f / %.2f | %.3f' % (n, c[0], c[1], h))
print('run            | arms RMS vs GT, vs converged (mm) | zero crossing arm 0 / 1 | height 12')
for p in args[1:]:
    t, h = series(p); c = cross(t); b = float(np.exp(np.load(p)['theta'][2]))
    r = lambda T: np.sqrt(((t[6:14]-T[6:14])**2).mean())
    print('DEM beta %.4f | %5.1f  %5.1f | %.2f / %.2f | %.3f' % (b, r(gt[0]), r(cv[0]), c[0], c[1], h))
