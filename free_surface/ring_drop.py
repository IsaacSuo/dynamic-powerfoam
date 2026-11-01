"""ring_plastic.py on PAC-NeRF's plasticine scene 0 against its MPM particles, as merged_drop.py: the shape of elastic scene 0, the scene's E, nu, yield stress and start velocity, gravity, the ground at y = -0.05 with friction large enough for the MPM's sticky ground.
Per frame (1/24 s): height and widths (ours: the cap cells' outermost points), the centre of mass, Chamfer in both directions against 30000 of the particles. Frames saved on the way (x, radii, L, yielded), every --every steps; the log line-buffered; no timeout.
  python free_surface/ring_drop.py --foam <dir with dem_exact_single> --save <dir> --tag <tag>"""
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
from free_surface.ring_plastic import RingPlastic, RingStep

f64, dev = torch.float64, 'cuda'
wp_t = lambda a_: wp.to_torch(a_, requires_grad=False)
ROOT = Path('/home/fangsuo/py/dynamic-powerfoam/data/pacnerf')
OFFSET = np.array([.5, -.05, .5])
a_ = argparse.ArgumentParser()
a_.add_argument('--foam', type=Path, required=True)
a_.add_argument('--save', type=Path, default=Path('output/spring'))
a_.add_argument('--tag', default='ring')
a_.add_argument('--frames', type=int, default=15)
a_.add_argument('--dt', type=float, default=1/1200)
a_.add_argument('--tol', type=float, default=1e-3)
a_.add_argument('--mu', type=float, default=10., help="the ground's friction: large for the MPM's sticky ground")
a_.add_argument('--kappa', type=float, default=2.)
a_.add_argument('--every', type=int, default=10)
a_.add_argument('--pre-every', type=int, default=4)
a_.add_argument('--newton-max', type=int, default=300)
a_.add_argument('--steps', type=int, default=0, help='stop after this step (0: at the last frame)')
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
W = RingPlastic(X0, r0, V0, g['pairs'], g['areas'], g['contacts'], E, nu, ys, theta=.5, kappa=args.kappa)
W.set_ball((0., 1., 0.), -.05, args.mu)
W.start()
W.prepare()
up = torch.tensor([0., 1., 0.], dtype=f64, device=dev)
fg = -9.8*M3*up; wref = float(fg.norm()); tol = args.tol*wref
CG = TrustCG(W, mass, args.dt, per=2, product_iters=1)
W.update_ul(W.X)
W.grad(W.X)
ST = RingStep(W, CG, mass, W.ul_blocks(), fg, args.dt, args.mu)
ST.pre_ul = True; ST.pre_every = args.pre_every; ST.smooth_pred = ST.carry_delta = True; ST.newton_max = args.newton_max
ST.set_theta(.5, mass)
ST.set_state(W.X.clone(), torch.tensor(gt['v'], dtype=f64, device=dev).expand(n, 3).clone(), fg/M3)
h = float(np.median(V0))**(1/3)
print('plasticine 0, ring springs: E %.1f, nu %.4f, yield %.1f Pa (yield strain %.4f, cell limit e_y %.4f), kappa %g, v0 %s; foam %s: %d cells, h %.2f mm; dt 1/%d, tol %g of the weight, friction %g; MPM %d particles' % (
    gt['E'], nu, gt['ys'], gt['ys']/gt['E'], W.ey, args.kappa, np.round(gt['v'], 4).tolist(), args.foam.name, n, 1e3*h, round(1/args.dt), args.tol, args.mu, len(Gt[0])), flush=True)


def ours():
    x = ST.tx; a = W.tul['a']; c = a.norm(dim=1) > 0
    return torch.cat([x, x[c]+W._ulr[c, None]*a[c]/a[c].norm(dim=1, keepdim=True)])


def spreads(p):
    q = torch.quantile(p.float(), torch.tensor([.01, .99], device=p.device), dim=0)
    return (q[1]-q[0]).cpu().numpy()


def row(k):
    o = ours(); gp = Gt[k]
    oh = (o.max(0).values-o.min(0).values).cpu().numpy(); gh = gp.max(0)-gp.min(0)
    c1, c2 = chamfer(o, Gp[k])
    com = ((M3*ST.tx).sum(0)/M).cpu().numpy()
    so, sg = spreads(ST.tx), spreads(Gp[k])
    print('FRAME %2d t %.3f: height %.3f (MPM %.3f); x width %.3f (%.3f), z width %.3f (%.3f); spreads of the sites 1-99 %% x %.3f (%.3f) y %.3f (%.3f) z %.3f (%.3f); com %s (%s); low %.4f (%.4f); Chamfer ours->MPM %.1f, MPM->ours %.1f mm' % (
        k, k/24, oh[1], gh[1], oh[0], gh[0], oh[2], gh[2], so[0], sg[0], so[1], sg[1], so[2], sg[2], np.round(com, 4).tolist(), np.round(gp.mean(0), 4).tolist(), float(o[:, 1].min()), gp[:, 1].min(), c1, c2), flush=True)
    return (k, k/24, *oh, *gh, *com, *gp.mean(0), float(o[:, 1].min()), gp[:, 1].min(), c1, c2)


rows = [row(0)]
xs, rs, Ls, ys_, ts = [], [], [], [], []


def save():
    np.savez(args.save/('frames_%s.npz' % args.tag), x=np.array(xs), r=np.array(rs), L=np.array(Ls), yld=np.array(ys_), t=np.array(ts), rows=np.array(rows))


t0 = t_saved = time.time()
nit, npr = [], []
nf = int(round(1/(24*args.dt))); nsteps = args.steps or args.frames*nf
H0 = .5*float((M3*ST.tv**2).sum())+float((-fg*ST.tx).sum())
for s in range(1, nsteps+1):
    t_s = time.time()
    e_, its, g2 = ST.step(tol, lambda g_: .5*tol)
    if not math.isfinite(g2) or math.sqrt(g2) > tol:
        tr_ = ST.stats['trace']
        g_ = wp_t(ST.a['gz']).norm(dim=1); top = torch.argsort(g_, descending=True)[:8]
        sf = torch.linalg.svdvals(W.tFc.cpu()).to(dev)
        print('  at the failure: F_o smallest singular value min %.3f, largest max %.3f; beta min %.3f; rho / e_y max %.1f' % (float(sf[:, 2].min()), float(sf[:, 0].max()), float(W.tbe.min()), float(W.tr['rho'].max())/W.ey), flush=True)
        print('  largest |g| cells (index: |g| / weight; F_o\'s singular values; beta; yielded; V0 / median; height / h):', flush=True)
        for c_ in top.tolist():
            print('    %d: %.2f; %.3f %.3f; beta %.3f; %d; V %.2f; y %.1f' % (c_, float(g_[c_])/wref, float(sf[c_, 2]), float(sf[c_, 0]), float(W.tbe[c_]), int(W._yld[c_]), float(W.V0[c_]/W.V0.median()), float(ST.tz[c_, 1]+.05)/h), flush=True)
        np.savez(args.save/('fail_%s.npz' % args.tag), x=ST.tx.cpu().numpy(), v=ST.tv.cpu().numpy(), a=ST.ta.cpu().numpy(), z=ST.tz.cpu().numpy(), step=s, g=wp_t(ST.a['gz']).cpu().numpy())
        print('step %d, t %.4f: the solve did not converge (|g| / weight %s after %d Newton iterations); CG %s; |g| %s; ratio %s' % (s, s*args.dt, math.sqrt(g2)/wref if math.isfinite(g2) else g2, its, [a__[0] for a__ in tr_][:30], ['%.3g' % (a__[4]/wref) for a__ in tr_][:30], ['%.2g' % a__[5] for a__ in tr_][:30]), flush=True)
        break
    ST.end()
    nit.append(its); npr.append(e_)
    if s % args.every == 0:
        xs.append(ST.tx.cpu().numpy()); rs.append(W._ulr.cpu().numpy()); Ls.append(W.tul['Ln'].cpu().numpy()); ys_.append(W._yld.cpu().numpy()); ts.append(s*args.dt)
    if s % 20 == 0:
        i_, j_, Lw, _, _, m = W._w
        ek = .5*float((M3*ST.tv**2).sum()); ed, ev = W.stored(); eg = float((-fg*ST.tx).sum())
        bd = W.B['bdelta'].numpy()
        print('t %.4f (%.0f s): KE %.3e springs+outer %.3e vol %.3e W_p %.3e, left %.3f of the start\'s energy; com y %.4f; flowing %d cells, yielded so far %d; walls %d, springs %d, new %d, closed %d, without a wall %d; J %.3f..%.3f; touching %d, ground/weight %.2f; min wall L / h %.3f, max r / h %.2f, radii %d (fails %d), flips %d; rho / e_y median %.2f, max %.1f, beta min %.3f; products %d, Newton %d, rejected %d, negative %d; %.2f s' % (
            s*args.dt, time.time()-t0, ek, ed, ev, W.Wp, 1-(ek+ed+ev+W.Wp+eg)/H0, float((M3*ST.tx)[:, 1].sum())/M, W.ul_stats['yielding'], W.ul_stats['yielded'], m, W._m_s, W.ul_stats['born'], W.ul_stats['closed'], W.ul_stats['ghost'], float(torch.exp(W.tul['Ln']).min()), float(torch.exp(W.tul['Ln']).max()),
            int((bd > 0).sum()), float(W.B['bfn'].numpy().sum())/wref, float(Lw.min())/h, float(W._ulr.max())/h, W.ul_stats['radii_it'], W.ul_stats['radii_fail'], W.ul_stats['flips'], float(W.tr['rho'].median())/W.ey, float(W.tr['rho'].max())/W.ey, float(W.tbe.min()), e_, its, ST.stats['rejected'], ST.stats['negative'], time.time()-t_s), flush=True)
    if s % nf == 0:
        rows.append(row(s//nf))
        save(); t_saved = time.time()
    elif time.time()-t_saved > 60:
        save(); t_saved = time.time()
save()
ni, np_ = np.array(nit), np.array(npr)
print('Newton per step: median %d, 99 %% %d, max %d, steps above 50: %d of %d; products per step: median %d, max %d, total %d; plastic work %.4f' % (np.median(ni), np.percentile(ni, 99), ni.max(), int((ni > 50).sum()), len(ni), np.median(np_), np_.max(), np_.sum(), W.Wp), flush=True)
print('done: %.0f s' % (time.time()-t0), flush=True)
