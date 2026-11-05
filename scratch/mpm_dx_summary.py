"""MPM re-runs at several dx (mpm_pacnerf_repro.py npz, world frame) and our DEM runs (show_frames.npz, scene frame), all
against the GT (= MPM at dx 0.02): the ground's horizontal force per unit mass a_k along the forward direction (from the
centre of mass, exact), the horizontal speed lost by frame 6->7 and after it (to 12->13), the speed left at 12->13, the
rotation since frame 0 (whole-body Kabsch) at frames 9 / 11 / 13, the centre's height at 9 / 12, and the two upper arms'
tip displacement in the body frame (pacnerf_beta_summary.py's construction): RMS of the difference to GT over frames 6-13"""
import sys
from pathlib import Path
import numpy as np
from plyfile import PlyData
ROOT = Path(__file__).resolve().parents[1]
OFF = np.array([.5, -.05, .5]); G0 = -.05; FPS = 24
ef = np.array([-0.4078950600549248, 0., -0.06629882732047176]); ef /= np.linalg.norm(ef)
V = np.load(ROOT/'inverse_data'/'pacnerf_e0'/'foam'/'dem_exact_single'/'geometry.npz')['masses']
M = [np.stack([PlyData.read('/home/fangsuo/py/dynamic-powerfoam/data/pacnerf/simulation_data/elastic/0/%d.ply' % k)['vertex'][a] for a in 'xyz'], 1)+OFF for k in range(14)]
c0 = M[0].mean(0); _, U = np.linalg.eigh(np.cov((M[0]-c0).T)); nrm, e1, e2 = U[:, 0], U[:, 2], U[:, 1]
polar = lambda P: (np.arctan2((P-c0)@e2, (P-c0)@e1), np.linalg.norm((P-c0)-np.outer((P-c0)@nrm, nrm), axis=1))
ang, rad = polar(M[0]); Rmax = rad.max()
bins = np.linspace(-np.pi, np.pi, 361)
reach = np.array([rad[(ang >= bins[i]) & (ang < bins[i+1])].max(initial=0) for i in range(360)])
sm = np.convolve(np.r_[reach, reach, reach], np.ones(15)/15, 'same')[360:720]; peaks = []
for i in np.argsort(-sm):
    a = .5*(bins[i]+bins[i+1])
    if all(abs((a-p+np.pi) % (2*np.pi)-np.pi) > np.pi/4 for p in peaks):
        peaks.append(a)
    if len(peaks) == 4:
        break
peaks = np.array(sorted(peaks)); dirs = [np.cos(p)*e1+np.sin(p)*e2 for p in peaks]; upper = np.argsort([-d_[1] for d_ in dirs])[:2]


def sets(P0):
    a_, r_ = polar(P0); d = np.abs((a_[:, None]-peaks[None]+np.pi) % (2*np.pi)-np.pi); arm = np.argmin(d, 1)
    return r_ < .35*Rmax, [(arm == j) & (d.min(1) < np.pi/8) & (r_ > .75*Rmax) for j in range(4)]


def kabsch(P0, P, w):
    w = w/w.sum(); a0, a = (w[:, None]*P0).sum(0), (w[:, None]*P).sum(0)
    U_, _, Vt = np.linalg.svd(((P-a)*w[:, None]).T@(P0-a0)); D = np.diag([1., 1., np.sign(np.linalg.det(U_@Vt))]); return U_@D@Vt, a0, a


def summary(Ps, w):
    core, tips = sets(Ps[0]); tip, rot = [], []
    com = np.array([(w[:, None]*P).sum(0)/w.sum() for P in Ps])
    for k in range(len(Ps)):
        R, a0, a = kabsch(Ps[0][core], Ps[k][core], w[core]); q = (Ps[k]-a)@R+a0
        tip.append([((w[t, None]*(q[t]-Ps[0][t])).sum(0))/w[t].sum() for t in tips])
        Rw, _, _ = kabsch(Ps[0], Ps[k], w); rot.append(np.degrees(np.arccos(np.clip((np.trace(Rw)-1)/2, -1, 1))))
    vf = np.diff(com, axis=0)@ef*FPS                       # vf[k] = interval k -> k+1
    return dict(a=np.diff(vf)*FPS, vf=vf, rot=np.array(rot), h=com[:, 1]-G0, tip=1e3*np.array(tip))


ref = summary(M, np.ones(len(M[0])))
print('run | a_k, k = 5..11 (m/s^2) | lost by 6->7, after | left 12->13 | rot 9/11/13 (deg) | height 9/12 (m) | upper-arm RMS vs GT (mm)')
rows = [('GT (MPM dx 0.020)', ref)]
for p in sys.argv[1:]:
    z = np.load(p)
    if 'dx' in z:
        X = z['x'].astype(np.float64)[:14]+OFF; rows.append(('MPM dx %.4f%s' % (float(z['dx']), ' split' if int(z['split']) == 2 else ''), summary(list(X), np.ones(X.shape[1]))))
    else:
        rows.append(('DEM beta %.4f' % float(np.exp(z['theta'][2])), summary(list(z['x'].astype(np.float64)), V)))
for name, s in rows:
    d = s['tip'][6:14][:, upper]-ref['tip'][6:14][:, upper]
    print('%-20s| %s | %.3f %.3f | %+.3f | %.1f %.1f %.1f | %.3f %.3f | %.1f' % (
        name, ' '.join('%+5.2f' % s['a'][k-1] for k in range(5, 12)), s['vf'][4]-s['vf'][6], s['vf'][6]-s['vf'][12], s['vf'][12],
        s['rot'][9], s['rot'][11], s['rot'][13], s['h'][9], s['h'][12], np.sqrt((d**2).mean())))
