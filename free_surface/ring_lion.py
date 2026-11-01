"""ring_ul.py on the lion, as merged_rest_lion.py: the foam and conditions of dem_contact_lion.py (round_clean, GT E nu, mu 0, dt 1/1200, 1.2 s, beta), no self contact, no yield. Frames (x, F, t, L) every --every steps from t = 0, for dem_gt_compare.py; the log line-buffered; no timeout.
  python free_surface/ring_lion.py --foam <dir with dem_exact_single> --save <dir> --tag <tag> [--beta 0.05]"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import warp as wp

from free_surface.closure_warp import MidpointStep, TrustCG
from free_surface.ring_ul import RingUL

f64, dev = torch.float64, 'cuda'
wp_t = lambda a_: wp.to_torch(a_, requires_grad=False)
a_ = argparse.ArgumentParser()
a_.add_argument('--foam', type=Path, required=True)
a_.add_argument('--save', type=Path, default=Path('output/lion'))
a_.add_argument('--tag', default='ring')
a_.add_argument('--beta', type=float, default=.05)
a_.add_argument('--T', type=float, default=1.2)
a_.add_argument('--every', type=int, default=20)
a_.add_argument('--dt', type=float, default=1/1200)
a_.add_argument('--tol', type=float, default=1e-3)
a_.add_argument('--newton-max', type=int, default=300)
a_.add_argument('--pre-every', type=int, default=4)
args = a_.parse_args()
E, nu, C0, v0 = 10648.614517785627/1000, .3912951704721983, -.2, (0., 0., -.245)
g = np.load(args.foam/'dem_exact_single'/'geometry.npz'); cl = np.load(args.foam/'dem_exact_single'/'cells.npz')
X0, r0, V0 = g['points'], cl['radii'], g['masses']; n = len(X0)
args.save.mkdir(parents=True, exist_ok=True)
mass = torch.as_tensor(V0, dtype=f64, device=dev); M = float(mass.sum()); M3 = mass[:, None]
up = torch.tensor([0., 0., 1.], dtype=f64, device=dev)
W = RingUL(X0, r0, V0, g['pairs'], g['areas'], g['contacts'], E, nu, theta=.5)
W.set_ball((0., 0., 1.), C0, 0.)
W.set_viscosity(args.beta, args.dt)
W.rest_cut()
W.prepare()
fg = -9.8*M3*up; wref = float(fg.norm()); tol = args.tol*wref
CG = TrustCG(W, mass, args.dt, per=2, product_iters=1)
ST = MidpointStep(W, CG, mass, W.ul_blocks(), fg, args.dt, 0.)
ST.pre_ul = True; ST.pre_every = args.pre_every; ST.smooth_pred = ST.carry_delta = True; ST.newton_max = args.newton_max
ST.set_theta(.5, mass)
v = torch.tensor(v0, dtype=f64, device=dev).expand(n, 3).clone()
ST.set_state(W.X.clone(), v, fg/M3)
W.store = dict(springs=0., vol=0., var=0., outer=0.)
H0 = .5*float((M3*ST.tv**2).sum())+float((-fg*ST.tx).sum())
T = W.tr; m = W._m
lo0, hi0 = T['i'][:m].long(), T['j'][:m].long(); L0 = (W.X[hi0]-W.X[lo0]).norm(dim=1)
h = float(np.median(V0))**(1/3)
print('lion, ring springs: %d cells, h %.2f mm, %d walls; E / rho %.3f, nu %.4f, beta %g, dt 1/%d, tol %g of the weight; ground z = %g, frictionless; the start\'s energy %.5f; touching radii on %d cells' % (n, 1e3*h, m, E, nu, args.beta, round(1/args.dt), args.tol, C0, H0, int((W.tbR > 0).sum())), flush=True)
# the front legs, by their rest position
X_ = W.X
fr = (X_[:, 2] < .43) & (X_[:, 0] < .45); ymid = float(X_[:, 1].median()); legs = dict(a=fr & (X_[:, 1] < ymid), b=fr & (X_[:, 1] >= ymid))
edges = np.linspace(.26, .43, 7)
bins = {k_: [torch.nonzero(mk & (X_[:, 2] >= float(edges[i])) & (X_[:, 2] < float(edges[i+1]))).flatten() for i in range(6)] for k_, mk in legs.items()}


def centre_line(x, k_):
    c = [x[b_].mean(0) for b_ in bins[k_]]
    return float(sum((c[i+1]-c[i]).norm() for i in range(5)))


print('front legs\' centre lines at rest: %.3f, %.3f' % (centre_line(X_, 'a'), centre_line(X_, 'b')), flush=True)
xs, Fs, Ls, ts = [X_.cpu().numpy()], [np.tile(np.eye(3), (n, 1, 1))], [np.zeros(n)], [0.]


def save():
    np.savez(args.save/('frames_%s.npz' % args.tag), x=np.array(xs), F=np.array(Fs), L=np.array(Ls), t=np.array(ts))


t0 = t_saved = time.time()
nit, npr = [], []
nsteps = int(round(args.T/args.dt))
for s in range(1, nsteps+1):
    e_, its, g2 = ST.step(tol, lambda g_: .5*tol)
    if not math.isfinite(g2) or math.sqrt(g2) > tol:
        tr_ = ST.stats['trace']
        gz = wp_t(ST.a['gz']).norm(dim=1); top = torch.argsort(gz, descending=True)[:6]
        sf = torch.linalg.svdvals(W.tFc.cpu()).to(dev)
        print('step %d, t %.4f: the solve did not converge (|g| / weight %s after %d Newton iterations); CG %s; |g| %s; F_g smallest min %.3f, largest max %.3f; largest |g| cells %s, V0 / median %s' % (s, s*args.dt, math.sqrt(g2)/wref if math.isfinite(g2) else g2, its, [a__[0] for a__ in tr_][:30], ['%.3g' % (a__[4]/wref) for a__ in tr_][:30], float(sf[:, 2].min()), float(sf[:, 0].max()), top.tolist(), np.array2string(V0[top.cpu().numpy()]/np.median(V0), precision=3)), flush=True)
        np.savez(args.save/('fail_%s.npz' % args.tag), x=ST.tx.cpu().numpy(), v=ST.tv.cpu().numpy(), a=ST.ta.cpu().numpy(), z=ST.tz.cpu().numpy(), step=s)
        save()
        break
    ST.end()
    nit.append(its); npr.append(e_)
    if s % args.every == 0:
        xs.append(ST.tx.cpu().numpy()); Fs.append(W._Fe.cpu().numpy()); Ls.append(W.tul['Ln'].cpu().numpy()); ts.append(s*args.dt)
        ek = .5*float((M3*ST.tv**2).sum()); ed, ev = W.stored(); eg = float((-fg*ST.tx).sum())
        sf = torch.linalg.svdvals(W._Fe.cpu())
        J = torch.linalg.det(W._Fe)
        st_ = (ST.tx[hi0]-ST.tx[lo0]).norm(dim=1)/L0
        bd = W.B['bdelta'].numpy()
        print('t %.4f (%.0f s): KE %.3e springs+outer %.3e vol %.3e W_v %.3e ground %.3e, left %.3f; com up %.5f, lowest %.4f; J %.3f..%.3f, inverted %d; F_g singular %.3f..%.3f; rest-pair stretch 99.9 %% %.2f, max %.2f; touching %d; legs %.3f %.3f; last %d steps: products %d, Newton %d, max %d, rejected %d' % (
            s*args.dt, time.time()-t0, ek, ed, ev, W.Wv, float(W.tsum[6]), 1-(ek+ed+ev+W.Wv+eg)/H0, float((M3*ST.tx)[:, 2].sum())/M, float(ST.tx[:, 2].min()), float(J.min()), float(J.max()), int((J <= 0).sum()), float(sf[:, 2].min()), float(sf[:, 0].max()),
            float(torch.quantile(st_, .999)), float(st_.max()), int((bd > 0).sum()), centre_line(ST.tx, 'a'), centre_line(ST.tx, 'b'), args.every, sum(npr[-args.every:]), sum(nit[-args.every:]), max(nit[-args.every:]), ST.stats['rejected']), flush=True)
        if time.time()-t_saved > 60:
            save(); t_saved = time.time()
save()
ni, np_ = np.array(nit), np.array(npr)
print('Newton per step: median %d, 99 %% %d, max %d, steps above 50: %d of %d; products per step: median %d, max %d, total %d' % (np.median(ni), np.percentile(ni, 99), ni.max(), int((ni > 50).sum()), len(ni), np.median(np_), np_.max(), np_.sum()), flush=True)
print('done: %.0f s' % (time.time()-t0), flush=True)
