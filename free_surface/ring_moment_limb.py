"""Who holds a limb: the lion's elastic state rebuilt at a saved frame, each limb's net force split by term (tensor, remainder and outer layer, volume, ground; gravity), against its weight and along the way out from the body's axis; the energies. The same model at another run's frame: does it want that shape?
  python free_surface/ring_moment_limb.py --foam <dir> --frames new.npz --other old.npz --k 48"""
import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from free_surface.ring_moment import RingMoment

f64, dev = torch.float64, 'cuda'
a_ = argparse.ArgumentParser()
a_.add_argument('--foam', type=Path, required=True)
a_.add_argument('--frames', type=Path, required=True)
a_.add_argument('--other', type=Path, default=None)
a_.add_argument('--k', type=int, default=48)
a_.add_argument('--H', type=float, default=.14)
args = a_.parse_args()
E, nu, C0 = 10648.614517785627/1000, .3912951704721983, -.2
g = np.load(args.foam/'dem_exact_single'/'geometry.npz'); cl = np.load(args.foam/'dem_exact_single'/'cells.npz')
X0, r0, V0, P = g['points'], cl['radii'], g['masses'], g['pairs']; n = len(X0)
z0 = X0[:, 2]-X0[:, 2].min()
low = z0 < args.H; idx = -np.ones(n, int); idx[low] = np.arange(low.sum())
m_ = low[P[:, 0]] & low[P[:, 1]]; p_ = idx[P[m_]]
nc, lab = connected_components(coo_matrix((np.ones(len(p_)), (p_[:, 0], p_[:, 1])), shape=(low.sum(), low.sum())), directed=False)
ci = np.nonzero(low)[0]
sz = np.bincount(lab); parts = [ci[lab == k] for k in np.argsort(-sz)[:5] if sz[k] > 50]
limbs = sorted(parts, key=lambda a: (X0[a, 0].mean(), X0[a, 1].mean()))
names = ['%s (rest x %.2f y %.2f, %d cells)' % ('limb', X0[a, 0].mean(), X0[a, 1].mean(), len(a)) for a in limbs]
mass = torch.as_tensor(V0, device=dev, dtype=f64)
W = RingMoment(X0, r0, V0, P, g['areas'], g['contacts'], E, nu, 1e9, theta=.5, kappa=5., eps_cut=-1.)
W.set_ball((0., 0., 1.), C0, 0.)
W.start(); W.prepare()
W.update_ul(W.X)
wq = (W._wi.clone(), W._wj.clone(), W._qi.clone(), W._qj.clone(), W.tkey.clone(), W.tlo.clone(), W.thi.clone(), W._qkey.clone(), W._Ms.clone(), W._Mlast.clone())
Mo0 = W.tr['Mo'].clone(); gid0 = W._gid.clone()
Zs = [('run', np.load(args.frames))]
if args.other is not None:
    Zs.append(('other', np.load(args.other)))
for name, Z in Zs:
    W._wi, W._wj, W._qi, W._qj, W.tkey, W.tlo, W.thi, W._qkey, W._Ms, W._Mlast = [a.clone() for a in wq]
    W._m_s = len(W.tkey)
    W.tr['Mo'].copy_(Mo0); W._set_gid(gid0)
    k = args.k
    x = torch.as_tensor(Z['x'][k], device=dev, dtype=f64); F = torch.as_tensor(Z['F'][k], device=dev, dtype=f64)
    W._Ft = F.clone(); W._Fe = F.clone()
    Ln = torch.as_tensor(Z['L'][k] if 'L' in Z.files else np.log(np.clip(np.linalg.det(Z['F'][k]), 1e-3, None)), device=dev, dtype=f64)
    W.tul['Ln'].copy_(Ln)
    W.update_ul(x)
    W.grad(x); W.reread(); _, f = W.grad(x); ftot = -f.clone()
    es = W.tsum.clone()
    Y, T = W.ty, W.tr
    ms = W._m_s
    sv = {k_: Y[k_].clone() for k_ in ('be', 'br', 'Hd')}; dn = T['dni'].clone(), T['dnj'].clone(); L_ = W.tul['Ln'].clone(); Lr = W.ring['Ln'] if False else None
    Y['be'].zero_(); Y['Hd'].zero_()
    _, f = W.grad(x); fnoT = -f.clone()
    Y['be'].copy_(sv['be']); Y['Hd'].copy_(sv['Hd'])
    Y['br'].zero_(); T['dni'].zero_(); T['dnj'].zero_()
    _, f = W.grad(x); fnoR = -f.clone()
    Y['br'].copy_(sv['br']); T['dni'].copy_(dn[0]); T['dnj'].copy_(dn[1])
    W.tul['Ln'].zero_()
    _, f = W.grad(x); fnoV = -f.clone()
    W.tul['Ln'].copy_(L_)
    fT, fR, fV = ftot-fnoT, ftot-fnoR, ftot-fnoV
    fG = ftot-fT-fR-fV
    grav = (-9.8*mass)[:, None]*torch.tensor([0., 0., 1.], dtype=f64, device=dev)
    com = (mass[:, None]*x).sum(0)/mass.sum()
    print('%s, t %.3f: extent %s; energy by term %s; sum over all cells of tensor / remainder / volume forces (should vanish) %.2e %.2e %.2e' % (name, float(Z['t'][k]), np.round(np.ptp(Z['x'][k], axis=0), 3).tolist(), {k_: float('%.3e' % float(es[k_])) for k_ in range(len(es)) if abs(float(es[k_])) > 1e-12}, float(fT.sum(0).norm()), float(fR.sum(0).norm()), float(fV.sum(0).norm())), flush=True)
    Ep = float((9.8*mass*x[:, 2]).sum())
    print('  ENERGY tensor %.5f remainder %.5f outer %.5f volume %.5f ground %.5f; elastic+ground %.5f; potential %.5f; TOTAL %.5f' % (float(es[0]), float(es[4]), float(es[5]), float(es[1]+es[2]), float(es[6]), float(es[0]+es[1]+es[2]+es[4]+es[5]+es[6]), Ep, float(es[0]+es[1]+es[2]+es[4]+es[5]+es[6])+Ep), flush=True)
    ax = torch.tensor([1., 0., 0.], dtype=f64, device=dev)
    for nm, a in zip(names, limbs):
        ii = torch.as_tensor(a, device=dev)
        wt = float(mass[ii].sum())*9.8
        c = x[ii].mean(0)
        out = c-com; out = out-(out*ax).sum()*ax; out[2] = 0.; out = out/out.norm().clamp(min=1e-9)
        s = lambda f_: np.round((f_[ii].sum(0)/wt).cpu().numpy(), 2).tolist()
        o = lambda f_: float((f_[ii].sum(0)*out).sum())/wt
        print('  %s: centre %s, %.3f from the axis; net force / its weight: tensor %s out %+.2f; remainder %s out %+.2f; volume %s out %+.2f; ground+other %s out %+.2f; total out %+.2f; L median %.3f' % (nm, np.round(c.cpu().numpy(), 3).tolist(), float((c-com-((c-com)*ax).sum()*ax)[1].abs()), s(fT), o(fT), s(fR), o(fR), s(fV), o(fV), s(fG), o(fG), o(ftot+grav), float(L_[ii].median())), flush=True)
