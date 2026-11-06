"""the ground's grip per foot in our DEM (dem_foot_loops.py's run: the cap contact alpha 30), as mpm_foot_grip.py: every step
per cap point its glue force (forward, along e_f) and normal force; per foot (points ahead of / behind the centre of mass):
the forward force T, the normal force N, the area of the points glued (bound > 0) and pressed (normal force > 0) (their
rest area weights), the pressed caps (cells), and the change of T since the last step split by points: held (glued in
both steps: their own change), new (glued now only), lost (glued before only: their force before). Per unit mass.
args: out [frames]"""
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
rows, grip = [], []
prev = {}


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
    h = float((W.st.V.sum()/n)**(1/3)); wq = cq['kq']/(li.CAPC*(W.lam+2*W.mu)/h)
    act = cq['bound'] > 0; tf = ft/M; prs = fn > 0
    if not prev:
        prev.update(act=torch.zeros_like(act), fwd=front.clone(), tf=torch.zeros_like(tf))
    row = [s*li.DT]
    for ft_ in (front, ~front):
        m_ = act & ft_
        held = m_ & prev['act']; new = m_ & ~prev['act']; lost = prev['act'] & ~act & (prev['fwd'] if ft_ is front else ~prev['fwd'])
        row += [float(tf[m_].sum()), float(fn[m_ | (prs & ft_)].sum())/M, float(wq[m_].sum()), float(wq[prs & ft_].sum()),
                float((tf[held]-prev['tf'][held]).sum()), float(tf[new].sum()), -float(prev['tf'][lost].sum()), int(torch.unique(cb[prs & ft_]).numel())]
    grip.append(row); prev.update(act=act.clone(), fwd=front.clone(), tf=tf.clone())


o = li.forward(np.array([10.04361, .30079413, -5.809143]), frames, verbose=True, on_step=probe)
np.savez(out, rows=np.array(rows), cols='t, com (3), Nf, Nb, Tf, Tb (per unit mass), hf, hb (m over the ground)', grip=np.array(grip),
         gcols='t; per foot (front, back): T, N, area glued, area pressed (m^2, rest), dT held, T new, -T lost (per unit mass), pressed caps')
print('saved', out, flush=True)
