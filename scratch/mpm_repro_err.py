"""each MPM re-run (mpm_pacnerf_repro.py npz) against the GT particles, frames 1..15: per-particle RMS distance (mm),
the centre of mass's distance (mm), the lowest particle's height (mm, GT in brackets)"""
import sys
import numpy as np
from plyfile import PlyData
D = '/home/fangsuo/py/dynamic-powerfoam/data/pacnerf/simulation_data/elastic/0/%d.ply'
G = [np.stack([PlyData.read(D % k)['vertex'][c] for c in 'xyz'], 1).astype(np.float64) for k in range(16)]
for p in sys.argv[1:]:
    z = np.load(p); X = z['x'].astype(np.float64)
    print('== %s  dx %.5f dt %.2e split %d' % (p.split('/')[-1], float(z['dx']), float(z['dt']), int(z['split'])))
    print(' frame: ' + ' '.join('%6d' % k for k in range(4, len(X))))
    print(' rms  : ' + ' '.join('%6.1f' % (1e3*np.sqrt(((X[k]-G[k])**2).sum(1).mean())) for k in range(4, len(X))))
    print(' com  : ' + ' '.join('%6.1f' % (1e3*np.linalg.norm(X[k].mean(0)-G[k].mean(0))) for k in range(4, len(X))))
    print(' low  : ' + ' '.join('%6.1f' % (1e3*X[k][:, 1].min()) for k in range(4, len(X))))
    print(' (GT) : ' + ' '.join('%6.1f' % (1e3*G[k][:, 1].min()) for k in range(4, len(X))))
