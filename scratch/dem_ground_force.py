"""the ground's action on our DEM at every step (1/1200 s), lion_inverse.forward of PAC-NeRF elastic scene 0 (the current
default: consistent mass) with a probe: per touching ball c (cell B, radius R) the force on the site f_c = F_n n + f_t at
the step's midpoint z (the midpoint rule's force), and where it acts:
  site       x_B (what the model does: the force on the site, no lever)
  ball       x_B - R n (the rigid ball's own lowest point)
  material   x_B - R F_B F_B^T n / |F_B^T n| (the cell's deformed sphere's lowest point, F_B at z)
Records the total force, the torque about the material's centre of mass (sum m (x + F_B d) / M) for each point, the
centre of pressure (F_n-weighted) for each, the touching count, and per frame 5..9 the touching balls' (point, impulse).
args: lnbeta, out"""
import sys
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import free_surface.lion_inverse as li
lnb, out = float(sys.argv[1]), sys.argv[2]
li.set_scene('pacnerf_e0')
if len(sys.argv) > 3:
    li.GLUE = sys.argv[3]
if len(sys.argv) > 4:
    li.SHAPE = sys.argv[4]
if len(sys.argv) > 5:
    li.CAPC = float(sys.argv[5])
rec = dict(t=[], com=[], F=[], tau_model=[], cop_model=[], tau_site=[], tau_ball=[], tau_mat=[], cop_site=[], cop_ball=[], cop_mat=[], Fn=[], n=[])
maps = {k: [] for k in range(5, 10)}


def probe(s, W, ST, cbody, Rb):
    n = W.n; dev = 'cuda'
    up = torch.tensor([0., 1., 0.], dtype=torch.float64, device=dev)
    z = ST.tz[:n]; F = W.tF
    d = torch.as_tensor(W.cmass['d'], device=dev) if W.cmass is not None else torch.zeros(n, 3, dtype=torch.float64, device=dev)
    m = W.st.V
    c = (m[:, None]*(z+torch.einsum('nij,nj->ni', F, d))).sum(0)/m.sum()
    fn, ff = W.tbfn, W.tbff
    if W.capc is not None:                                                          # (the cap contact: its points are the contacts)
        cq = W.tcq; on = cq['fnq'] > 0; t = s*li.DT
        rec['t'].append(t); rec['com'].append(c.cpu().numpy())
        if not bool(on.any()):
            for k in ('F', 'tau_model', 'cop_model', 'tau_site', 'tau_ball', 'tau_mat', 'cop_site', 'cop_ball', 'cop_mat'):
                rec[k].append(np.zeros(3))
            rec['Fn'].append(0.); rec['n'].append(0)
            return
        B = cq['cb'][on].long(); xq = z[B]+torch.einsum('nij,nj->ni', F[B], cq['rq'][on])
        f = cq['fnq'][on][:, None]*up+cq['ffq'][on]; w = cq['fnq'][on]/cq['fnq'][on].sum()
        tq = torch.linalg.cross(xq-c, f).sum(0).cpu().numpy(); cp = (w[:, None]*xq).sum(0).cpu().numpy()
        rec['F'].append(f.sum(0).cpu().numpy())
        for k in ('tau_model', 'tau_site', 'tau_ball', 'tau_mat'):
            rec[k].append(tq)
        for k in ('cop_model', 'cop_site', 'cop_ball', 'cop_mat'):
            rec[k].append(cp)
        rec['Fn'].append(float(cq['fnq'].sum())); rec['n'].append(int(on.sum()))
        fr = int(t*li.FPS)
        if fr in maps:
            maps[fr].append(np.c_[xq.cpu().numpy(), (f*li.DT).cpu().numpy(), B.cpu().numpy()])
        return
    on = fn > 0
    t = s*li.DT
    rec['t'].append(t); rec['com'].append(c.cpu().numpy())
    if not bool(on.any()):
        for k in ('F', 'tau_model', 'cop_model', 'tau_site', 'tau_ball', 'tau_mat', 'cop_site', 'cop_ball', 'cop_mat'):
            rec[k].append(np.zeros(3))
        rec['Fn'].append(0.); rec['n'].append(0)
        return
    B = cbody[on]; R = Rb[on]; f = fn[on][:, None]*up+ff[on]
    xs = z[B]; xb = xs-R[:, None]*up
    FB = F[B]; Ftn = torch.einsum('nji,j->ni', FB, up)
    xm = xs-R[:, None]*torch.einsum('nij,nj->ni', FB, Ftn)/Ftn.norm(dim=1, keepdim=True)
    w = fn[on]/fn[on].sum()
    rec['F'].append(f.sum(0).cpu().numpy())
    xt = xs+torch.einsum('nij,nj->ni', FB, W.tbroff[on]) if W.glue_point == 'material' else xs   # (where the model applies the tangential force)
    xq = xm if W.ball_shape == 'deformed' else xs                                  # (where the model applies the normal force)
    rec['cop_model'].append((w[:, None]*xq).sum(0).cpu().numpy())
    rec['tau_model'].append((torch.linalg.cross(xq-c, fn[on][:, None]*up)+torch.linalg.cross(xt-c, ff[on])).sum(0).cpu().numpy())
    for nm, xp in (('site', xs), ('ball', xb), ('mat', xm)):
        rec['tau_'+nm].append(torch.linalg.cross(xp-c, f).sum(0).cpu().numpy())
        rec['cop_'+nm].append((w[:, None]*xp).sum(0).cpu().numpy())
    rec['Fn'].append(float(fn[on].sum())); rec['n'].append(int(on.sum()))
    fr = int(t*li.FPS)
    if fr in maps:
        maps[fr].append(np.c_[xm.cpu().numpy(), (f*li.DT).cpu().numpy(), B.cpu().numpy()])


o = li.forward(np.array([10.04361, .30079413, lnb]), 12, save_frames=False, on_step=probe)
sv = {k: np.array(v) for k, v in rec.items()}
for k, v in maps.items():
    sv['map%d' % k] = np.concatenate(v) if v else np.zeros((0, 7))
sv['M'] = float(li.load_foam()[1]['masses'].sum())
np.savez(out, **sv)
print('saved', out)
