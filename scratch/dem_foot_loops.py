"""per foot, our DEM (lion_inverse.forward of PAC-NeRF elastic scene 0, consistent mass, beta 0.003, the cap contact alpha 30:
the ground's forces at the exposed caps' points), as mpm_foot_loops.py: every step (1/1200 s) the ground's normal and
forward force on the front and on the back foot (the cap points ahead of / behind the centre of mass along e_f, at the
step's midpoint z), and each foot's squash: the mean height over the ground of the same material as the MPM's (the GT
frame-0 particles within 50 mm of the lowest one), each particle carried by its power cell's affine map x_B + F_B (X - X_B)
at z. Per unit mass. args: out [frames]"""
import sys
from pathlib import Path
import numpy as np
import torch
from plyfile import PlyData
from scipy.spatial import cKDTree
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import free_surface.lion_inverse as li
out = sys.argv[1]; frames = int(sys.argv[2]) if len(sys.argv) > 2 else 11
li.set_scene('pacnerf_e0'); li.CAPC = 30.
H = .05; OFF = np.array([.5, -.05, .5])
EF = np.array([-0.4078950600549248, 0., -0.06629882732047176]); EF /= np.linalg.norm(EF)
X0 = np.stack([PlyData.read('/home/fangsuo/py/dynamic-powerfoam/data/pacnerf/simulation_data/elastic/0/0.ply')['vertex'][c] for c in 'xyz'], 1).astype(np.float64)
c0 = X0.mean(0)
lab0 = np.where(X0[:, 1] < X0[:, 1].min()+H, np.where((X0-c0)@EF > 0, 1, 2), 0)
Xf = X0[lab0 > 0]+OFF; labf = lab0[lab0 > 0]
cells, _, _ = li.load_foam()
p, r = np.asarray(cells['points'], np.float64), np.asarray(cells['radii'], np.float64)
Rm = r.max()
_, cell = cKDTree(np.c_[p, np.sqrt(Rm**2-r**2)]).query(np.c_[Xf, np.zeros(len(Xf))])   # (the power cell: nearest in the lifted space)
print('foot material: front %d, back %d particles; in cells %d' % ((labf == 1).sum(), (labf == 2).sum(), len(np.unique(cell))), flush=True)
dev = 'cuda'
ef = torch.tensor(EF, dtype=torch.float64, device=dev)
Bp = torch.as_tensor(cell, device=dev).long(); Xp = torch.as_tensor(Xf, dtype=torch.float64, device=dev); Lp = torch.as_tensor(labf, device=dev)
rows = []


def probe(s, W, ST, cbody, Rb):
    n = W.n
    z = ST.tz[:n]; F = W.tF; m = W.st.V; M = float(m.sum())
    d = torch.as_tensor(W.cmass['d'], device=dev)
    c = (m[:, None]*(z+torch.einsum('nij,nj->ni', F, d))).sum(0)/M
    cq = W.tcq; cb = cq['cb'].long()
    xq = z[cb]+torch.einsum('nij,nj->ni', F[cb], cq['rq'])
    front = (xq-c)@ef > 0
    fn = cq['fnq']; ft = cq['ffq']@ef
    xp = z[Bp]+torch.einsum('nij,nj->ni', F[Bp], Xp-W.st.X[Bp])
    hf = float(xp[Lp == 1, li.UP].mean())-li.C0; hb = float(xp[Lp == 2, li.UP].mean())-li.C0
    rows.append(np.r_[s*li.DT, c.cpu().numpy(), float(fn[front].sum())/M, float(fn[~front].sum())/M, float(ft[front].sum())/M, float(ft[~front].sum())/M, hf, hb])


o = li.forward(np.array([10.04361, .30079413, -5.809143]), frames, verbose=True, on_step=probe)
np.savez(out, rows=np.array(rows), cols='t, com (3), Nf, Nb, Tf, Tb (per unit mass), hf, hb (m over the ground)')
print('saved', out, flush=True)
