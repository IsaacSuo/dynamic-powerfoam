"""free vibration against the voxel FEM, set-up: the FEM's elastic modes 0-11 (talk-pages round01_exact h 0.008, scene
frame) at the GT particles (MPM) and at our sites; an initial velocity v0 = a sum_{j<9} c_j phi_j (c_j random signs,
seed 0) scaled so that the largest displacement amplitude is about 3 mm at the lowest frequency; for the particles the
rigid part taken off (uniform mass: mean velocity and the angular velocity of the angular momentum). Our sites' rigid part
is taken off by dem_freevib.py with the model's own mass. Saves out_inverse/freevib/prep.npz"""
import sys
from pathlib import Path
import numpy as np
from plyfile import PlyData
sys.path.insert(0, str(Path(__file__).resolve().parent))
from consistent_mass_check_interp import interpolate
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'out_inverse'/'freevib'; OUT.mkdir(parents=True, exist_ok=True)
OFF = np.array([.5, -.05, .5])
fem = dict(np.load('/home/fangsuo/py/dynamic-powerfoam/.claude/worktrees/talk-pages/output/round01_exact/fem_modes_h0.008.npz'))
K, KX = 12, 9
P = np.stack([PlyData.read('/home/fangsuo/py/dynamic-powerfoam/data/pacnerf/simulation_data/elastic/0/0.ply')['vertex'][c] for c in 'xyz'], 1).astype(np.float64)+OFF
S = np.load(ROOT/'inverse_data/pacnerf_e0/foam/dem_exact_single/cells.npz')['points'].astype(np.float64)
phiP, okP = interpolate(fem, fem['V'][:, :K], P); phiS, okS = interpolate(fem, fem['V'][:, :K], S)
c = np.where(np.random.default_rng(0).random(KX) < .5, -1., 1.)
a = 3e-3*2*np.pi*fem['f'][0]/np.abs(phiP[:, :, :KX]@c).max()
v0P = a*(phiP[:, :, :KX]@c); v0S = a*(phiS[:, :, :KX]@c)
x = P-P.mean(0); v0P -= v0P.mean(0)
I = (x*x).sum()*np.eye(3)-x.T@x; w = np.linalg.solve(I, np.cross(x, v0P).sum(0)); v0P -= np.cross(w, x)
print('FEM f 0-11 (Hz) %s; inside the voxel mesh: particles %.4f, sites %.4f; a %.4g; |v0| max particles %.3f, sites %.3f m/s'
      % (np.round(fem['f'][:K], 3).tolist(), okP.mean(), okS.mean(), a, np.abs(v0P).max(), np.abs(v0S).max()))
np.savez(OUT/'prep.npz', phiP=phiP.astype(np.float32), phiS=phiS, v0P=v0P, v0S=v0S, c=c, a=a, f=fem['f'][:K], off=OFF)
