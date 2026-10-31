"""ul_plastic.py on PAC-NeRF's plasticine scene 0 against its MPM particles (2026-10-31): the shape of elastic scene 0 (frame 0 identical), the scene's E, nu, yield stress and start
velocity (data/pacnerf/gt/plasticine.json), gravity, the ground at y = -0.05 with friction large enough for the MPM's sticky ground (ours = theirs + (0.5, -0.05, 0.5), as plastic_drop.py).
Per frame (1/24 s): height and widths (ours: the cap cells' outermost points), the centre of mass, Chamfer in both directions against 30000 of the particles.
Frames saved on the way (x, radii, L, en, ep), every --every steps; the log line-buffered; no timeout.
  python free_surface/ul_plastic_drop.py --foam <dir with dem_exact_single> --save <dir> --tag <tag>"""
import argparse
import io
import json
import math
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import warp as wp
from plyfile import PlyData

from free_surface.closure_warp import TrustCG
from free_surface.ul_plastic import PlasticStep, PlasticUL

f64, dev = torch.float64, 'cuda'
wp_t = lambda a_: wp.to_torch(a_, requires_grad=False)
ROOT = Path('/home/fangsuo/py/dynamic-powerfoam/data/pacnerf')
OFFSET = np.array([.5, -.05, .5])
a_ = argparse.ArgumentParser()
a_.add_argument('--foam', type=Path, required=True)
a_.add_argument('--save', type=Path, default=Path('output/plastic'))
a_.add_argument('--tag', default='drop')
a_.add_argument('--frames', type=int, default=15)
a_.add_argument('--dt', type=float, default=1/1200)
a_.add_argument('--tol', type=float, default=1e-3)
a_.add_argument('--mu', type=float, default=10., help="the ground's friction: large for the MPM's sticky ground")
a_.add_argument('--grad', default='gg')
a_.add_argument('--every', type=int, default=10)
a_.add_argument('--pre-every', type=int, default=4)
a_.add_argument('--log-from', type=int, default=10**9, help="every step from this one on: the start's |g|, the balls' overlap and radii, the Newton trace")
a_.add_argument('--newton-max', type=int, default=300)
args = a_.parse_args()
gt = json.loads((ROOT/'gt'/'plasticine.json').read_text())['0']
E, nu, ys = gt['E']/1000, gt['nu'], gt['ys']/1000
g = np.load(args.foam/'dem_exact_single'/'geometry.npz'); cl = np.load(args.foam/'dem_exact_single'/'cells.npz')
X0, r0, V0 = g['points'], cl['radii'], g['masses']; n = len(X0)
args.save.mkdir(parents=True, exist_ok=True)
Z = zipfile.ZipFile(ROOT/'simulation_data.zip')


def truth(k):
    v = PlyData.read(io.BytesIO(Z.read('simulation_data/plasticine/0/%d.ply' % k)))['vertex']
    return np.stack([v['x'], v['y'], v['z']], 1).astype(np.float64)+OFFSET


rng = np.random.default_rng(0)
Gt = [truth(k) for k in range(args.frames+1)]
pick = rng.choice(len(Gt[0]), 30000, replace=False)
Gp = [torch.as_tensor(p[pick], device=dev).float() for p in Gt]


def chamfer(a, b, chunk=1024):
    """the mean nearest distance a -> b, b -> a (mm)"""
    a = a.float()

    def one(p, q):
        return torch.cat([torch.cdist(p[i:i+chunk], q).min(1).values for i in range(0, len(p), chunk)]).mean()
    return 1e3*float(one(a, b)), 1e3*float(one(b, a))


mass = torch.as_tensor(V0, dtype=f64, device=dev); M = float(mass.sum()); M3 = mass[:, None]
W = PlasticUL(X0, r0, V0, E, nu, ys, grad=args.grad)
W.set_ball((0., 1., 0.), -.05, args.mu)
W.update_ul(W.X)
W.prepare()
up = torch.tensor([0., 1., 0.], dtype=f64, device=dev)
fg = -9.8*M3*up; wref = float(fg.norm()); tol = args.tol*wref
CG = TrustCG(W, mass, args.dt, per=2, product_iters=1)
ST = PlasticStep(W, CG, mass, W.ul_blocks(), fg, args.dt, args.mu)
ST.pre_ul = True; ST.pre_every = args.pre_every; ST.smooth_pred = ST.carry_delta = True; ST.newton_max = args.newton_max
ST.set_theta(1., mass)
ST.set_state(W.X.clone(), torch.tensor(gt['v'], dtype=f64, device=dev).expand(n, 3).clone(), fg/M3)
h = float(np.median(V0))**(1/3)
print('plasticine 0: E %.1f, nu %.4f, yield %.1f Pa (yield strain %.4f), v0 %s; foam %s: %d cells, h %.2f mm; dt 1/%d, tol %g of the weight, friction %g, grad %s; MPM %d particles' % (
    gt['E'], nu, gt['ys'], gt['ys']/gt['E'], np.round(gt['v'], 4).tolist(), args.foam.name, n, 1e3*h, round(1/args.dt), args.tol, args.mu, args.grad, len(Gt[0])), flush=True)


def ours():
    """the sites and the cap cells' outermost points, on the last cut's radii and cap directions"""
    x = ST.tx; a = W.tul['a']; c = a.norm(dim=1) > 0
    return torch.cat([x, x[c]+W._ulr[c, None]*a[c]/a[c].norm(dim=1, keepdim=True)])


def row(k):
    o = ours(); gp = Gt[k]
    oh = (o.max(0).values-o.min(0).values).cpu().numpy(); gh = gp.max(0)-gp.min(0)
    c1, c2 = chamfer(o, Gp[k])
    com = ((M3*ST.tx).sum(0)/M).cpu().numpy()
    print('FRAME %2d t %.3f: height %.3f (MPM %.3f); x width %.3f (%.3f), z width %.3f (%.3f); com %s (%s); low %.4f (%.4f); Chamfer ours->MPM %.1f, MPM->ours %.1f mm' % (
        k, k/24, oh[1], gh[1], oh[0], gh[0], oh[2], gh[2], np.round(com, 4).tolist(), np.round(gp.mean(0), 4).tolist(), float(o[:, 1].min()), gp[:, 1].min(), c1, c2), flush=True)
    return (k, k/24, *oh, *gh, *com, *gp.mean(0), float(o[:, 1].min()), gp[:, 1].min(), c1, c2)


rows = [row(0)]
xs, rs, Ls, ens, eps, ts = [], [], [], [], [], []


def save():
    np.savez(args.save/('frames_%s.npz' % args.tag), x=np.array(xs), r=np.array(rs), L=np.array(Ls), en=np.array(ens), ep=np.array(eps), t=np.array(ts), rows=np.array(rows))


t0 = t_saved = time.time()
Y2 = float((mass*(W.X[:, 1]-(M3*W.X).sum(0)[1]/M)**2).sum())
nit, npr = [], []
nf = int(round(1/(24*args.dt))); nsteps = args.frames*nf
H0 = .5*float((M3*ST.tv**2).sum())+float((-fg*ST.tx).sum())
for s in range(1, nsteps+1):
    t_s = time.time()
    R_prev = W.tbR.clone()
    e_, its, g2 = ST.step(tol, lambda g_: .5*tol)
    if s >= args.log_from:
        tr_ = ST.stats['trace']; bd = torch.as_tensor(W.B['bdelta'].numpy(), device=dev); R_ = W.tbR; tc = bd > 0
        both = (R_ > 0) & (R_prev > 0); gz = wp_t(ST.a['gz']).norm(dim=1); top = torch.argsort(gz, descending=True)[:4]
        print('  step %d t %.4f: Newton %d, products %d, rejected %d; |g| %s; CG %s; touching %d, overlap / R max %.3f, overlap / h max %.3f, ground/weight %.2f; R / h of the touching max %.2f; this cut R / last max %.2f min %.2f, new caps %d, overlap of the new caps / h max %.3f; radii %d, walls %d; |eps_p| max %.3f; end |g| top cells %s' % (
            s, s*args.dt, its, e_, ST.stats['rejected'], ['%.3g' % (a__[4]/wref) for a__ in tr_][:4], [a__[0] for a__ in tr_][:6], int(tc.sum()), float((bd[tc]/R_[tc]).max()) if bool(tc.any()) else 0., float(bd.max())/h, float(W.B['bfn'].numpy().sum())/wref,
            float(R_[tc].max())/h if bool(tc.any()) else 0., float((R_[both]/R_prev[both]).max()), float((R_[both]/R_prev[both]).min()), int(((R_ > 0) & ~(R_prev > 0)).sum()), float(bd[(R_ > 0) & ~(R_prev > 0)].max())/h if bool(((R_ > 0) & ~(R_prev > 0)).any()) else 0., W.ul_stats['radii_it'], W.ul_stats['walls'], float(W.ep.flatten(1).norm(dim=1).max()),
            ' '.join('%d: %.2f (delta/h %.3f, V0/med %.2f)' % (c_, float(gz[c_])/wref, float(bd[c_])/h, float(W.V0[c_]/W.V0.median())) for c_ in top.tolist())), flush=True)
    if not math.isfinite(g2) or math.sqrt(g2) > tol:
        tr_ = ST.stats['trace']
        print('step %d, t %.4f: the solve did not converge (|g| / weight %s after %d Newton iterations); CG %s; |g| %s; ratio %s' % (s, s*args.dt, math.sqrt(g2)/wref if math.isfinite(g2) else g2, its, [a__[0] for a__ in tr_][:30], ['%.3g' % (a__[4]/wref) for a__ in tr_][:30], ['%.2g' % a__[5] for a__ in tr_][:30]), flush=True)
        break
    ST.end()
    nit.append(its); npr.append(e_)
    if s % args.every == 0:
        xs.append(ST.tx.cpu().numpy()); rs.append(W._ulr.cpu().numpy()); Ls.append(W.tul['Ln'].cpu().numpy()); ens.append(W.tul['en'].cpu().numpy()); eps.append(W.ep.cpu().numpy()); ts.append(s*args.dt)
    if s % 20 == 0:
        i_, j_, Lw, _, _, m = W._w
        ek = .5*float((M3*ST.tv**2).sum()); ed, ev = W.stored(); eg = float((-fg*ST.tx).sum())
        bd = W.B['bdelta'].numpy()
        Es_ = float(((M3*(ST.tx-ST.tx.mean(0)))[:, 1]*(ST.tx[:, 1]-(M3*ST.tx).sum(0)[1]/M)).sum())
        print('  integrated mean strain yy %.4f xx %.4f zz %.4f; cells\' share of W_p %.3f; site height std / rest %.4f, its log %.4f; mean |ep| %.3f; frozen beta error mean %.3f max %.3f; memory |delta| / L median %.4f, 99 %% %.4f, max %.4f' % (float(W.Hsum[1, 1]), float(W.Hsum[0, 0]), float(W.Hsum[2, 2]), W.Wp_cell/max(W.Wp, 1e-300), (Es_/Y2)**.5, .5*math.log(Es_/Y2), float((mass*W.ep.flatten(1).norm(dim=1)).sum())/M, *W.ul_stats['beta_err'], *W.ul_stats['delta_L']), flush=True)
        print('t %.4f (%.0f s): KE %.3e dev %.3e vol %.3e jump %.3e var %.3e W_p %.3e; com y %.4f; yielding %d cells, %d of %d walls; J %.3f..%.3f, |eps_p| median %.3f max %.3f; touching %d, ground/weight %.2f; min wall L / h %.3f, max r / h %.2f, radii %d (fails %d), flips %d; products %d, Newton %d, rejected %d, negative %d; %.2f s' % (
            s*args.dt, time.time()-t0, ek, ed, ev, float(W.tsum[3]), float(W.tsum[2]), W.Wp, float((M3*ST.tx)[:, 1].sum())/M, W.ul_stats['yielding'], W.ul_stats['wall_yielding'], m, float(torch.exp(W.tul['Ln']).min()), float(torch.exp(W.tul['Ln']).max()),
            float(W.ep.flatten(1).norm(dim=1).median()), float(W.ep.flatten(1).norm(dim=1).max()), int((bd > 0).sum()), float(W.B['bfn'].numpy().sum())/wref, float(Lw.min())/h, float(W._ulr.max())/h, W.ul_stats['radii_it'], W.ul_stats['radii_fail'], W.ul_stats['flips'], e_, its, ST.stats['rejected'], ST.stats['negative'], time.time()-t_s), flush=True)
    if s % nf == 0:
        rows.append(row(s//nf))
        save(); t_saved = time.time()
    elif time.time()-t_saved > 60:
        save(); t_saved = time.time()
save()
ni, np_ = np.array(nit), np.array(npr)
print('Newton per step: median %d, 99 %% %d, max %d, steps above 50: %d of %d; products per step: median %d, max %d, total %d; plastic work %.4f, cells\' share %.3f' % (np.median(ni), np.percentile(ni, 99), ni.max(), int((ni > 50).sum()), len(ni), np.median(np_), np_.max(), np_.sum(), W.Wp, W.Wp_cell/max(W.Wp, 1e-300)), flush=True)
print('done: %.0f s' % (time.time()-t0), flush=True)
