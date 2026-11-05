"""the jack's arms after the rigid motion is taken off (user 11-06: the two upper arms' bounce looks wrong): frame 0's
plane of the cross by PCA (its normal the least-variance axis), the arms as the 4 angular sectors around the
centre in that plane (centred on the 4 directions of largest reach), each arm's tip = its points beyond 75 % of the
arm's reach; per frame a rigid fit (Kabsch) on the core (within 35 % of the reach), then each tip's mean displacement
in that body frame against frame 0, split into along the arm, sideways in the plane, out of the plane (mm), MPM
particles against our cells (volume-weighted)"""
import sys
from pathlib import Path
import numpy as np
from plyfile import PlyData
ROOT = Path(__file__).resolve().parents[1]
run = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT/'out_inverse'/'pacnerf_e0_show'/'show_frames.npz'
OFF = np.array([.5, -.05, .5])
X = np.load(run)['x'].astype(np.float64); K = len(X)
V = np.load(ROOT/'inverse_data'/'pacnerf_e0'/'foam'/'dem_exact_single'/'geometry.npz')['masses']
M = []
for k in range(K):
    v = PlyData.read(str(Path('/home/fangsuo/py/dynamic-powerfoam/data/pacnerf/simulation_data/elastic/0')/('%d.ply' % k)))['vertex']
    M.append(np.stack([v['x'], v['y'], v['z']], 1)+OFF)
c0 = M[0].mean(0)
w_, U = np.linalg.eigh(np.cov((M[0]-c0).T))
nrm, e1, e2 = U[:, 0], U[:, 2], U[:, 1]
def polar(P):
    d = P-c0; return np.arctan2(d@e2, d@e1), np.linalg.norm(d-np.outer(d@nrm, nrm), axis=1)
ang, rad = polar(M[0])
bins = np.linspace(-np.pi, np.pi, 361); reach = np.array([rad[(ang >= bins[i]) & (ang < bins[i+1])].max(initial=0) for i in range(360)])
sm = np.convolve(np.r_[reach, reach, reach], np.ones(15)/15, 'same')[360:720]
peaks = []
for i in np.argsort(-sm):
    a = .5*(bins[i]+bins[i+1])
    if all(abs((a-p+np.pi) % (2*np.pi)-np.pi) > np.pi/4 for p in peaks):
        peaks.append(a)
    if len(peaks) == 4:
        break
peaks = np.array(sorted(peaks)); dirs = [np.cos(p)*e1+np.sin(p)*e2 for p in peaks]
def arm_of(P):
    a, _ = polar(P); d = np.abs((a[:, None]-peaks[None]+np.pi) % (2*np.pi)-np.pi); return np.argmin(d, 1), d.min(1)
up = np.array([dd[1] for dd in dirs])                                              # (world y of each arm's direction)
order = np.argsort(-up)
print('the cross: normal %s; arms (world direction, reach mm): %s; the two upper arms: %s' % (
    np.round(nrm, 2).tolist(), [(np.round(d_, 2).tolist(), round(1e3*reach[np.argmin(abs(.5*(bins[:-1]+bins[1:])-p))], 1)) for d_, p in zip(dirs, peaks)], order[:2].tolist()))
Rmax = rad.max()
def sets(P0):
    a_, d_ = arm_of(P0); _, r_ = polar(P0)
    core = r_ < .35*Rmax
    tips = [(a_ == j) & (d_ < np.pi/8) & (r_ > .75*Rmax) for j in range(4)]
    return core, tips
cm_, tm_ = sets(M[0]); co_, to_ = sets(X[0])
print('points: MPM core %d, tips %s; ours core %d, tips %s' % (cm_.sum(), [int(t.sum()) for t in tm_], co_.sum(), [int(t.sum()) for t in to_]))
def kabsch(P0, P, w):
    w = w/w.sum(); a0, a = (w[:, None]*P0).sum(0), (w[:, None]*P).sum(0)
    U_, _, Vt = np.linalg.svd(((P-a)*w[:, None]).T@(P0-a0)); D = np.diag([1., 1., np.sign(np.linalg.det(U_@Vt))]); return U_@D@Vt, a0, a
def tipdisp(Ps, core, tips, w):
    out = []
    for k in range(K):
        R, a0, a = kabsch(Ps[0][core], Ps[k][core], w[core])
        q = (Ps[k]-a)@R+a0                                                            # (the frame-k points mapped back to frame 0's body frame)
        row = []
        for j in range(4):
            t = tips[j]; dd = ((w[t, None]*(q[t]-Ps[0][t])).sum(0))/w[t].sum()
            side = np.cross(nrm, dirs[j])
            row.append((dd@dirs[j], dd@side, dd@nrm))
        out.append(row)
    return 1e3*np.array(out)                                                          # (frames, arms, 3) mm
dm = tipdisp(M, cm_, tm_, np.ones(len(M[0]))); do = tipdisp(X, co_, to_, V)
for j in order[:2]:
    print('upper arm %d (direction %s): tip displacement in the body frame (mm) | along the arm: MPM, ours | sideways in the plane: MPM, ours | out of the plane: MPM, ours' % (
        j, np.round(dirs[j], 2).tolist()))
    for k in range(K):
        print('  frame %2d | %+6.1f %+6.1f | %+6.1f %+6.1f | %+6.1f %+6.1f' % (k, dm[k, j, 0], do[k, j, 0], dm[k, j, 1], do[k, j, 1], dm[k, j, 2], do[k, j, 2]))
for j in order[2:]:
    print('lower arm %d: largest |tip displacement| over the frames: MPM %.1f mm, ours %.1f mm' % (j, np.abs(dm[:, j]).max(), np.abs(do[:, j]).max()))
