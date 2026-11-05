"""several scene-0 forwards (show_frames.npz) against MPM, side by side: the two upper arms' tip displacement in the body
frame (pacnerf_arms.py's construction; RMS over frames 6-13 of the difference, all 3 components), the centre of mass's
height above the ground (frames 9-13), its horizontal speed after frame 9, the rotation since frame 0 at frames 9, 11, 13"""
import sys
from pathlib import Path
import numpy as np
from plyfile import PlyData
ROOT = Path(__file__).resolve().parents[1]
runs = sys.argv[1:]
OFF = np.array([.5, -.05, .5]); G0 = -.05
V = np.load(ROOT/'inverse_data'/'pacnerf_e0'/'foam'/'dem_exact_single'/'geometry.npz')['masses']
M = [np.stack([PlyData.read(str(Path('/home/fangsuo/py/dynamic-powerfoam/data/pacnerf/simulation_data/elastic/0')/('%d.ply' % k)))['vertex'][a] for a in 'xyz'], 1)+OFF for k in range(14)]
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


def motion(Ps, w):
    core, tips = sets(Ps[0]); out, com, rot = [], [], []
    for k in range(len(Ps)):
        R, a0, a = kabsch(Ps[0][core], Ps[k][core], w[core]); q = (Ps[k]-a)@R+a0
        out.append([((w[t, None]*(q[t]-Ps[0][t])).sum(0))/w[t].sum() for t in tips])
        com.append((w[:, None]*Ps[k]).sum(0)/w.sum())
        Rw, _, _ = kabsch(Ps[0], Ps[k], w); rot.append(np.degrees(np.arccos(np.clip((np.trace(Rw)-1)/2, -1, 1))))
    return 1e3*np.array(out), np.array(com), np.array(rot)


R0 = np.load(ROOT/'inverse_data'/'pacnerf_e0'/'foam'/'dem_exact_single'/'cells.npz')['radii']


def slip(Ps, bottom, ks=range(6, 10)):
    """the median horizontal step, mm per frame, of the points touching the ground at both ends of a frame (bottom: the
    point's height of its lowest part above its centre)"""
    out = []
    for k in ks:
        on = (Ps[k][:, 1]-bottom-G0 < 5.5e-3) & (Ps[k+1][:, 1]-bottom-G0 < 5.5e-3)
        d = Ps[k+1][on][:, [0, 2]]-Ps[k][on][:, [0, 2]]
        out.append(1e3*np.median(np.linalg.norm(d, axis=1)) if on.any() else float('nan'))
    return out


tm, cm, rm = motion(M, np.ones(len(M[0])))
print('slip of the ground layer (mm per frame, frames 6-7 .. 9-10): MPM %s' % np.round(slip(M, 0.), 1).tolist())
print('run | upper arms: RMS of the tip difference, frames 6-13 (mm) | com height frames 9..13 (m) | vx frames 9-13 | rotation (deg) frames 9 / 11 / 13')
print('MPM |   -   | %s | %+.3f | %.1f / %.1f / %.1f' % (' '.join('%.3f' % (cm[k, 1]-G0) for k in range(9, 14)), (cm[13, 0]-cm[9, 0])*24/4, rm[9], rm[11], rm[13]))
for r in runs:
    z = np.load(r); X = z['x'].astype(np.float64); b = float(np.exp(z['theta'][2]))
    to, co, ro = motion(list(X), V)
    print('   slip of our ground layer (cells whose sphere is within 5.5 mm of the ground), %s: %s' % (Path(r).parent.name, np.round(slip(list(X), R0), 1).tolist()))
    d = to[6:14][:, upper]-tm[6:14][:, upper]
    print('beta %.4f | %5.1f | %s | %+.3f | %.1f / %.1f / %.1f' % (b, np.sqrt((d**2).mean()), ' '.join('%.3f' % (co[k, 1]-G0) for k in range(9, 14)),
                                                       (co[13, 0]-co[9, 0])*24/4, ro[9], ro[11], ro[13]), flush=True)
