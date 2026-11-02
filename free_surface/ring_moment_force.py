"""At a real saved state of a run (nothing rebuilt): is the force at x what the frames' motion asks, and on a leg, per term, along its own motion and along straightening in the ground's plane."""
import numpy as np
import torch
import warp as wp
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from free_surface.closure_warp import k_mid_ball

f64, dev = torch.float64, 'cuda'
TERMS = {0: 'tensor', 1: 'volume', 2: 'var', 4: 'remainder', 5: 'outer', 6: 'ground'}


def probe(W, ST, mass, fg, V0, X0, P, dt):
    n = W.n; A, bl = ST.a, W.ball; m3 = mass[:, None]
    wp.launch(k_mid_ball, bl['k'], inputs=[A['x'], bl['body'], bl['R'], bl['ng'], ST.c0, ST.g8, ST.cb], outputs=[bl['xn'], bl['kt'], bl['bound']])
    x, v, a = ST.tx.clone(), ST.tv.clone(), ST.ta.clone()
    W.update_ul(x)
    W.grad(x); W.reread(); _, g0 = W.grad(x); g0 = g0.clone()

    def terms(y):
        W.grad(y)
        t = W.tsum.clone(); t = t.sum(1) if t.dim() > 1 else t
        return {TERMS[k]: float(t[k]) for k in TERMS}
    res = m3*a-fg+g0                                                                       # (what the elastic force and gravity leave of m a: the viscous force and the step's form)
    wc = fg.norm(dim=1)
    print('t %.4f: per cell over its own weight: |elastic force| median %.1f, 90 %% %.1f; |m a - gravity| median %.1f; what is left median %.2f, 90 %% %.1f, 99 %% %.1f; speed median %.3f' % (float(W.tul['xn'].shape[0])*0+0., float((g0.norm(dim=1)/wc).median()), float(torch.quantile(g0.norm(dim=1)/wc, .9)), float(((m3*a-fg).norm(dim=1)/wc).median()), float((res.norm(dim=1)/wc).median()), float(torch.quantile(res.norm(dim=1)/wc, .9)), float(torch.quantile(res.norm(dim=1)/wc, .99)), float(v.norm(dim=1).median())), flush=True)
    zr = X0[:, 2]-X0[:, 2].min(); low = zr < .18
    idx = -np.ones(n, int); idx[low] = np.arange(low.sum())
    m = low[P[:, 0]] & low[P[:, 1]]; p = idx[P[m]]
    _, lab = connected_components(coo_matrix((np.ones(len(p)), (p[:, 0], p[:, 1])), shape=(low.sum(), low.sum())), directed=False)
    ids = np.nonzero(low)[0]
    legs = sorted([ids[lab == k] for k in np.argsort(-np.bincount(lab))[:4]], key=lambda a_: X0[a_, 0].mean())
    e0 = terms(x)
    up = torch.tensor([0., 0., 1.], dtype=f64, device=dev)
    ed = np.linspace(0., .17, 8)
    for li, a_ in enumerate(legs[:2]):
        at = torch.as_tensor(a_, device=dev)
        bins = [a_[(zr[a_] >= ed[i]) & (zr[a_] < ed[i+1])] for i in range(7)]
        lo_ = torch.as_tensor(np.concatenate(bins[:4]), device=dev)
        mm = mass[lo_].sum()
        print('  leg %d: lower part mass %.2e, mean velocity %s; net force on it: elastic %s, m a %s, gravity z %.2e; left %s' % (li, float(mm), np.round((m3[lo_]*v[lo_]).sum(0).cpu().numpy()/float(mm), 3).tolist(), np.round((-g0[lo_]).sum(0).cpu().numpy(), 5).tolist(), np.round((m3[lo_]*a[lo_]).sum(0).cpu().numpy(), 5).tolist(), float(fg[lo_].sum(0)[2]), np.round(res[lo_].sum(0).cpu().numpy(), 5).tolist()), flush=True)
        # along its own motion
        d = torch.zeros_like(x); d[at] = v[at]
        nr = float((d*d).sum().sqrt()); d = d/max(nr, 1e-12)
        eps = 1e-5
        ep, em = terms(x+eps*d), terms(x-eps*d)
        print('    along its own motion, -dE/ds per term: %s' % {k: float('%+.1e' % (-(ep[k]-em[k])/(2*eps))) for k in ep}, flush=True)
        for j in (3, 5):
            joint = x[torch.as_tensor(bins[j], device=dev)].mean(0)
            below = torch.as_tensor(np.concatenate(bins[:j]), device=dev)
            rot = torch.linalg.cross(up.expand(len(below), 3), x[below]-joint)
            pc, tc = x[torch.as_tensor(bins[0], device=dev)].mean(0), x[torch.as_tensor(bins[-1], device=dev)].mean(0)
            if float((rot[:len(bins[0])].mean(0)*(pc-tc)).sum()) < 0:
                rot = -rot
            e_ = 1e-3
            xp, xm = x.clone(), x.clone(); xp[below] += e_*rot; xm[below] -= e_*rot
            ep, em = terms(xp), terms(xm)
            print('    joint at slice %d in the ground plane, torque, + straightens: %s; viscous torque the left force gives %+.1e; angular speed there %+.2f' % (j, {k: float('%+.1e' % (-(ep[k]-em[k])/(2*e_))) for k in ep}, float((res[below]*rot).sum()), float((m3[below]*v[below]*rot).sum()/(m3[below]*rot*rot).sum())), flush=True)
