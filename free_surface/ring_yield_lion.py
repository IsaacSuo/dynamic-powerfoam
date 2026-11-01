"""ring_yield.py on the lion, as ring_lion.py: the foam and conditions of dem_contact_lion.py (round_clean, GT E nu, mu 0, dt 1/1200, 1.2 s), no self contact, the limit out of reach, every cell on the cut from the first step, beta on the model's own viscosity (ring_yield.py's header; --beta 0 and --eps-cut 1e9: the controls, both blow up on landing, as ring_lion.py at beta 0). Frames (x, F, t, L) every --every steps from t = 0, for dem_gt_compare.py; the log line-buffered; no timeout.
  python free_surface/ring_yield_lion.py --foam <dir with dem_exact_single> --save <dir> --tag <tag>"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import warp as wp

from free_surface.closure_warp import TrustCG
from free_surface.ring_yield import RingYield, YieldStep

f64, dev = torch.float64, 'cuda'
wp_t = lambda a_: wp.to_torch(a_, requires_grad=False)
a_ = argparse.ArgumentParser()
a_.add_argument('--foam', type=Path, required=True)
a_.add_argument('--save', type=Path, default=Path('output/lion'))
a_.add_argument('--tag', default='ry')
a_.add_argument('--T', type=float, default=1.2)
a_.add_argument('--every', type=int, default=20)
a_.add_argument('--dt', type=float, default=1/1200)
a_.add_argument('--tol', type=float, default=1e-3)
a_.add_argument('--pre-every', type=int, default=4)
a_.add_argument('--aa', type=int, default=6)
a_.add_argument('--hold-after', type=int, default=3)
a_.add_argument('--eps-cut', type=float, default=-1.)
a_.add_argument('--beta', type=float, default=.05)
args = a_.parse_args()
E, nu, C0, v0 = 10648.614517785627/1000, .3912951704721983, -.2, (0., 0., -.245)
g = np.load(args.foam/'dem_exact_single'/'geometry.npz'); cl = np.load(args.foam/'dem_exact_single'/'cells.npz')
X0, r0, V0 = g['points'], cl['radii'], g['masses']; n = len(X0)
args.save.mkdir(parents=True, exist_ok=True)
mass = torch.as_tensor(V0, dtype=f64, device=dev); M = float(mass.sum()); M3 = mass[:, None]
up = torch.tensor([0., 0., 1.], dtype=f64, device=dev)
W = RingYield(X0, r0, V0, g['pairs'], g['areas'], g['contacts'], E, nu, 1e9, theta=.5, kappa=5., eps_cut=args.eps_cut)
W.set_ball((0., 0., 1.), C0, 0.)
if args.beta > 0:
    W.set_viscosity(args.beta, args.dt)
W.start()
W.prepare()
fg = -9.8*M3*up; wref = float(fg.norm()); tol = args.tol*wref
CG = TrustCG(W, mass, args.dt, per=2, product_iters=1)
W.update_ul(W.X)
W.grad(W.X); W.reread(); W.grad(W.X)
ST = YieldStep(W, CG, mass, W.ul_blocks(), fg, args.dt, 0.)
ST.pre_ul = True; ST.pre_every = args.pre_every; ST.smooth_pred = ST.carry_delta = True
ST.set_theta(.5, mass)
ST.aa_m = args.aa
ST.hold_after = args.hold_after
ST.set_state(W.X.clone(), torch.tensor(v0, dtype=f64, device=dev).expand(n, 3).clone(), fg/M3)
H0 = .5*float((M3*ST.tv**2).sum())+float((-fg*ST.tx).sum())
h = float(np.median(V0))**(1/3)
print('lion, ring springs on the cut: %d cells, h %.2f mm; E / rho %.3f, nu %.4f, beta %g, dt 1/%d, tol %g of the weight; ground z = %g, frictionless; the start\'s energy %.5f' % (n, 1e3*h, E, nu, args.beta, round(1/args.dt), args.tol, C0, H0), flush=True)
xs, Fs, Ls, ts = [W.X.cpu().numpy()], [np.tile(np.eye(3), (n, 1, 1))], [np.zeros(n)], [0.]


def save():
    np.savez(args.save/('frames_%s.npz' % args.tag), x=np.array(xs), F=np.array(Fs), L=np.array(Ls), t=np.array(ts))


t0 = t_saved = time.time()
nit, npr, nout = [], [], []
nsteps = int(round(args.T/args.dt))
for s in range(1, nsteps+1):
    e_, its, g2 = ST.step(tol, lambda g_: .5*tol)
    if not math.isfinite(g2) or math.sqrt(g2) > tol:
        tr_ = ST.stats['trace']
        gz = wp_t(ST.a['gz']).norm(dim=1); top = torch.argsort(gz, descending=True)[:6]
        gl = W._gid == torch.arange(n, device=dev)
        Y_ = W.ty
        bad = {k_: int((~torch.isfinite(Y_[k_][gl]).all(dim=(1, 2))).sum()) for k_ in ('Fn', 'Ff', 'Mt', 'Mti', 'Mtt', 'Kt', 'Fni', 'Ms', 'Vz', 'Rz', 'Rn', 'Hn', 'Cz')}
        badx = {k_: int((~torch.isfinite(v_).all(dim=1)).sum()) for k_, v_ in (('x', ST.tx), ('z', ST.tz), ('g', wp_t(ST.a['gz'])))}
        bo = torch.nonzero(gl & ~torch.isfinite(Y_['Ff']).all(dim=(1, 2)))[:, 0]
        sn = torch.linalg.svdvals(torch.nan_to_num(Y_['Fn'][gl], nan=0., posinf=0., neginf=0.).cpu())
        print('step %d, t %.4f: the solve did not converge (|g| / weight %s, %d outer, %d inner); |g| %s; F_n smallest min %.3f, largest max %.3f; non-finite, groups: %s, sites: %s; groups with a bad fit %s, V0 / median %s; largest |g| cells %s, V0 / median %s' % (s, s*args.dt, math.sqrt(g2)/wref if math.isfinite(g2) else g2, ST.stats['outer'], its, ['%.3g' % (a__[4]/wref) for a__ in tr_][-20:], float(sn[:, 2].min()), float(sn[:, 0].max()), bad, badx, bo[:12].tolist(), np.array2string(V0[bo[:12].cpu().numpy()]/np.median(V0), precision=3), top.tolist(), np.array2string(V0[top.cpu().numpy()]/np.median(V0), precision=3)), flush=True)
        np.savez(args.save/('fail_%s.npz' % args.tag), x=ST.tx.cpu().numpy(), v=ST.tv.cpu().numpy(), a=ST.ta.cpu().numpy(), z=ST.tz.cpu().numpy(), step=s)
        save()
        break
    ST.end()
    nit.append(its); npr.append(e_); nout.append(ST.stats['outer'])
    if s % args.every == 0:
        xs.append(ST.tx.cpu().numpy()); Fs.append(W._Ft.cpu().numpy()); Ls.append(W.tul['Ln'].cpu().numpy()); ts.append(s*args.dt)
        ek = .5*float((M3*ST.tv**2).sum()); ed, ev = W.stored(); eg = float((-fg*ST.tx).sum())
        Lm = W.tul['Ln']
        sf = torch.linalg.svdvals(W._Ft.cpu())
        bd = W.B['bdelta'].numpy()
        print('t %.4f (%.0f s): KE %.3e springs+outer %.3e vol %.3e ground %.3e W_v %.3e, balance / start %.4f; com up %.5f, lowest %.4f; e^L %.3f..%.3f; F_e singular %.3f..%.3f; touching %d, ground/weight %.2f; walls %d, new %d, closed %d; radii %d (fails %d), flips %d; last %d steps: products %d, outer median %d max %d, rejected %d' % (
            s*args.dt, time.time()-t0, ek, ed, ev, float(W.tsum[6]), W.Wv, (ek+ed+ev+float(W.tsum[6])+W.Wv+eg)/H0, float((M3*ST.tx)[:, 2].sum())/M, float(ST.tx[:, 2].min()), float(torch.exp(Lm).min()), float(torch.exp(Lm).max()), float(sf[:, 2].min()), float(sf[:, 0].max()),
            int((bd > 0).sum()), float(W.B['bfn'].numpy().sum())/wref, W.ul_stats.get('walls', 0), W.ul_stats.get('born', 0), W.ul_stats.get('closed', 0), W.ul_stats.get('radii_it', 0), W.ul_stats.get('radii_fail', 0), W.ul_stats.get('flips', 0), args.every, sum(npr[-args.every:]), int(np.median(nout[-args.every:])), max(nout[-args.every:]), ST.stats['rejected']), flush=True)
        if time.time()-t_saved > 60:
            save(); t_saved = time.time()
save()
ni, no, np_ = np.array(nit), np.array(nout), np.array(npr)
print('steps %d of %d; outer per step: median %d, 99 %% %d, max %d; products per step: median %d, max %d, total %d' % (len(ni), nsteps, np.median(no), np.percentile(no, 99), no.max(), np.median(np_), np_.max(), np_.sum()), flush=True)
print('done: %.0f s' % (time.time()-t0), flush=True)
