"""One leg's fold, from the run's own frames only (elastic: q, w, outer layer at rest, so any frame can be rebuilt). A: slice by slice along the leg, over time: length, strain, energy. B: the energy along a straight path from the folded leg to its shape before the fold, the root fixed, per term.
  python free_surface/ring_moment_fold.py --foam <dir> --frames <frames.npz> [--leg 0]"""
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
a_.add_argument('--leg', type=int, default=0)
a_.add_argument('--k', type=int, nargs='+', default=[18, 20, 21, 24, 36, 72])
a_.add_argument('--pre', type=int, default=18)
args = a_.parse_args()
E, nu, C0 = 10648.614517785627/1000, .3912951704721983, -.2
g = np.load(args.foam/'dem_exact_single'/'geometry.npz'); cl = np.load(args.foam/'dem_exact_single'/'cells.npz')
X0, r0, V0, P = g['points'], cl['radii'], g['masses'], g['pairs']; n = len(X0)
Z = np.load(args.frames); h = float(np.median(V0)**(1/3)); Vm = float(np.median(V0))
zr = X0[:, 2]-X0[:, 2].min(); low = zr < .18
idx = -np.ones(n, int); idx[low] = np.arange(low.sum())
m = low[P[:, 0]] & low[P[:, 1]]; p = idx[P[m]]
_, lab = connected_components(coo_matrix((np.ones(len(p)), (p[:, 0], p[:, 1])), shape=(low.sum(), low.sum())), directed=False)
ids = np.nonzero(low)[0]
legs = sorted([ids[lab == k] for k in np.argsort(-np.bincount(lab))[:4]], key=lambda a: X0[a, 0].mean())
a = legs[args.leg]
edges = np.linspace(.02, .17, 8)
bins = [a[(zr[a] >= edges[i]) & (zr[a] < edges[i+1])] for i in range(7)]
at = torch.as_tensor(a, device=dev)
L0 = [np.linalg.norm(X0[bins[i+1]].mean(0)-X0[bins[i]].mean(0)) for i in range(6)]
TERMS = {0: 'tensor', 1: 'volume', 2: 'var', 4: 'remainder', 5: 'outer'}


def rebuild(k):
    W = RingMoment(X0, r0, V0, P, g['areas'], g['contacts'], E, nu, 1e9, theta=.5, kappa=5., eps_cut=-1.)
    W.set_ball((0., 0., 1.), C0, 0.)
    W.start(); W.prepare(); W.update_ul(W.X)
    x = torch.as_tensor(Z['x'][k], device=dev, dtype=f64)
    W._Ft = torch.as_tensor(Z['F'][k], device=dev, dtype=f64); W._Fe = W._Ft.clone()
    W.tul['Ln'].copy_(torch.as_tensor(Z['L'][k], device=dev, dtype=f64))
    W.update_ul(x); W.grad(x); W.reread(); W.grad(x)
    return W, x


def cellwise(W):
    Y = W.ty
    Xm = Y['Ff']-Y['vm'][:, None, None]*Y['Rz']
    e = .5*((Xm@Y['Mtt'])*Xm).sum((1, 2))*(W._gid == torch.arange(n, device=dev))
    return e, torch.linalg.svdvals(Y['Ff'].cpu()).to(dev), torch.linalg.det(Y['Ff']), Xm.flatten(1).norm(dim=1)


print('leg %d: %d cells; slices from the paw up: cells %s; rest segment lengths %s' % (args.leg, len(a), [len(b) for b in bins], np.round(L0, 3).tolist()), flush=True)
for k in args.k:
    W, x = rebuild(k)
    e, sv, dt, Xn = cellwise(W)
    Lc = torch.exp(W.tul['Ln'])
    xn = Z['x'][k]
    seg = [np.linalg.norm(xn[bins[i+1]].mean(0)-xn[bins[i]].mean(0))/L0[i] for i in range(6)]
    print('t %.3f: end to end %.3f; segment length / rest %s; leg tensor energy %.3e' % (float(Z['t'][k]), np.linalg.norm(xn[bins[-1]].mean(0)-xn[bins[0]].mean(0)), np.round(seg, 2).tolist(), float(e[at].sum())), flush=True)
    for i, b in enumerate(bins):
        bt = torch.as_tensor(b, device=dev)
        g_ = W._gid[bt]
        print('    slice %d: energy %.2e; F sv median %.2f %.2f %.2f, smallest %.2f; det F median %.2f; e^L median %.2f; |X| median %.2f; own group share %.2f; springs of the slice / cell %.1f, with a wall %.1f' % (i, float(e[bt].sum()), *[float(sv[bt, j].median()) for j in range(3)], float(sv[bt, 2].min()), float(dt[bt].median()), float(Lc[bt].median()), float(Xn[bt].median()), float((g_ == bt).double().mean()), float(((W.tlo.unsqueeze(1) == bt).any(1) | (W.thi.unsqueeze(1) == bt).any(1)).sum())*2/len(b), float(((W.tul['wi'][:W._w[5]].long().unsqueeze(1) == bt).any(1) | (W.tul['wj'][:W._w[5]].long().unsqueeze(1) == bt).any(1)).sum())*2/len(b)), flush=True)
# B: the straight path from the last frame's folded leg to its shape before the fold, root aligned, kept above the ground; recut at each point, L carried along the path
for births in (True, False):
    k = args.k[-1]
    W, x = rebuild(k)
    if not births:
        W.elastic_births = False
        keep = (W._wi+W._wj > 0)
        W.tlo, W.thi, W.tkey, W._wi, W._wj, W._qkey, W._qi, W._qj, W.tk0, W.tdl = W.tlo[keep], W.thi[keep], W.tkey[keep], W._wi[keep], W._wj[keep], W._qkey[keep], W._qi[keep], W._qj[keep], W.tk0[keep], W.tdl[keep]
        W.update_ul(x); W.grad(x); W.reread(); W.grad(x)
    # a hinge, not a straight line between shapes (which squeezes the leg): the part below the kink turned rigidly about it toward its shape before the fold, a smooth blend across the kink
    xpre = torch.as_tensor(Z['x'][args.pre], device=dev, dtype=f64)
    lowp = torch.as_tensor(np.concatenate(bins[:3]), device=dev)
    joint = x[torch.as_tensor(bins[4], device=dev)].mean(0)
    c1, c0 = x[lowp].mean(0), xpre[lowp].mean(0)
    jp = xpre[torch.as_tensor(bins[4], device=dev)].mean(0)
    U, _, Vt = torch.linalg.svd(((xpre[torch.as_tensor(np.concatenate(bins[-2:]), device=dev)]-jp).T@(x[torch.as_tensor(np.concatenate(bins[-2:]), device=dev)]-joint)).cpu())
    S = torch.diag(torch.tensor([1., 1., float(torch.linalg.det(U@Vt))], dtype=f64))
    R0 = (Vt.T@S@U.T).to(dev)                                                              # (the part above the kink: pre-fold frame to now)
    pre_low = (xpre[lowp]-jp)@R0.T
    U2, _, Vt2 = torch.linalg.svd(((x[lowp]-joint).T@pre_low).cpu())
    S2 = torch.diag(torch.tensor([1., 1., float(torch.linalg.det(U2@Vt2))], dtype=f64))
    Rh = (Vt2.T@S2@U2.T).to(dev)                                                           # (now to unfolded, for the lower part)
    ang = float(torch.arccos(((Rh.diagonal().sum()-1)/2).clamp(-1., 1.)))
    ax = torch.stack([Rh[2, 1]-Rh[1, 2], Rh[0, 2]-Rh[2, 0], Rh[1, 0]-Rh[0, 1]]); ax = ax/ax.norm().clamp(min=1e-12)
    zcell = torch.as_tensor(zr[a], device=dev, dtype=f64)
    zj = float(zr[bins[4]].mean()); zb = float(zr[bins[2]].mean())
    wt_ = ((zj-zcell)/(zj-zb)).clamp(0., 1.)                                               # (1 below the kink, 0 at it)
    print('the hinge: %.0f degrees to unfold, about %s, at z %.3f' % (np.degrees(ang), np.round(ax.cpu().numpy(), 2).tolist(), float(joint[2])), flush=True)

    def pos(sf):
        th_ = sf*ang*wt_
        r = x[at]-joint
        c_, s_ = torch.cos(th_)[:, None], torch.sin(th_)[:, None]
        axv = ax.expand(len(a), 3)
        rr = r*c_+torch.linalg.cross(axv, r)*s_+axv*((axv*r).sum(1, keepdim=True))*(1-c_)
        y_ = x.clone(); y_[at] = joint+rr
        return y_
    wg = 9.8*torch.as_tensor(V0, device=dev, dtype=f64)
    lo, hi = W.tlo, W.thi
    inleg = torch.zeros(n, dtype=torch.bool, device=dev); inleg[at] = True
    zero = (W._wi+W._wj) == 0
    print('weightless springs %s: %d in all, touching this leg %d; walls %d, open %d' % ('connected' if births else 'free surfaces', int(zero.sum()), int((zero & (inleg[lo] | inleg[hi])).sum()), W.ul_stats.get('walls', 0), W.ul_stats.get('open_walls', 0)), flush=True)
    base = None; last = x.clone()
    for s in np.linspace(0., 1., 11):
        y = pos(float(s))
        if s > 0:
            W.grad(y)
            W.tul['Ln'].add_((W.tul['Q']*W.ty['Gm']).sum((1, 2)))
            W._Ft = W.ty['Ff'].clone(); W._Fe = W._Ft.clone()
            W.update_ul(y)
        W.grad(y); W.reread(); W.grad(y)
        t = W.tsum.clone(); t = t.sum(1) if t.dim() > 1 else t
        v = {TERMS[j]: float(t[j]) for j in TERMS}; v['ground'] = float(W.ground_store(y)); v['gravity'] = float((wg[at]*y[at, 2]).sum())
        if base is None:
            base = dict(v)
        print('  s %.1f: total %+.3e; %s; leg e^L %.3f..%.3f' % (s, sum(v.values())-sum(base.values()), {j: float('%+.2e' % (v[j]-base[j])) for j in v}, float(torch.exp(W.tul['Ln'][at]).min()), float(torch.exp(W.tul['Ln'][at]).max())), flush=True)
        if float(torch.exp(W.tul['Ln'][at]).max()) > 5.:
            print('  the volume reading has left the range: stop', flush=True)
            break
