"""The landing's shape against the MPM by the mass-weighted spreads (2026-11-01, as memory plastic-flow-design: heights and widths by outermost points misled, the caps against the particles): per frame file, at the last saved time of each MPM frame asked, the spread of x, y, z over its own rest and the centre of mass above the ground over its own rest bias.
  python free_surface/flow_spreads.py --foam <dir> <frames.npz> ..."""
import argparse
import io
import zipfile
from pathlib import Path

import numpy as np
from plyfile import PlyData

ROOT = Path('/home/fangsuo/py/dynamic-powerfoam/data/pacnerf')
OFFSET = np.array([.5, -.05, .5])
a_ = argparse.ArgumentParser()
a_.add_argument('--foam', type=Path, required=True)
a_.add_argument('--frames', type=int, nargs='+', default=[6, 9, 12, 15])
a_.add_argument('--scene', type=int, default=0)
a_.add_argument('files', type=Path, nargs='+')
args = a_.parse_args()
g = np.load(args.foam/'dem_exact_single'/'geometry.npz')
X0, V0 = g['points'], g['masses']
Z = zipfile.ZipFile(ROOT/'simulation_data.zip')


def truth(k, scene=None):
    v = PlyData.read(io.BytesIO(Z.read('simulation_data/plasticine/%d/%d.ply' % (args.scene if scene is None else scene, k))))['vertex']
    return np.stack([v['x'], v['y'], v['z']], 1).astype(np.float64)+OFFSET


p0_, pk_ = truth(0, 0), truth(0)
c0_, ck_ = p0_.mean(0), pk_.mean(0)
U_, _, Vt_ = np.linalg.svd((pk_-ck_).T@(p0_-c0_))
X0 = ck_+(X0-c0_)@(U_@np.diag([1., 1., np.sign(np.linalg.det(U_@Vt_))])@Vt_).T


def spread(x, m):
    c = (m[:, None]*x).sum(0)/m.sum()
    return np.sqrt((m[:, None]*(x-c)**2).sum(0)/m.sum()), c


GROUND = -.05
s0, c0 = spread(X0, V0)
p0 = truth(0); ones = np.ones(len(p0)); s0g, c0g = spread(p0, ones)
for k in args.frames:
    pk = truth(k); sg, cg = spread(pk, np.ones(len(pk)))
    h_g = cg[1]-GROUND-(c0g[1]-c0[1])
    print('frame %2d t %.3f  MPM: spread x %.3f y %.4f z %.3f (of rest), com above the ground %.4f' % (k, k/24, sg[0]/s0g[0], sg[1]/s0g[1], sg[2]/s0g[2], h_g), flush=True)
    for f in args.files:
        z = np.load(f)
        t = z['t']; x = z['x']
        o = np.argmin(np.abs(t-k/24))
        if abs(t[o]-k/24) > .01:
            print('   %-16s no frame near' % f.stem, flush=True)
            continue
        s, c = spread(x[o], V0)
        print('   %-16s spread x %.3f (%+.1f %%) y %.4f (%+.1f %%) z %.3f (%+.1f %%), com %.4f (%+.1f %%)' % (f.stem, s[0]/s0[0], 100*(s[0]/s0[0]/(sg[0]/s0g[0])-1), s[1]/s0[1], 100*(s[1]/s0[1]/(sg[1]/s0g[1])-1), s[2]/s0[2], 100*(s[2]/s0[2]/(sg[2]/s0g[2])-1), c[1]-GROUND, 100*((c[1]-GROUND)/h_g-1)), flush=True)
