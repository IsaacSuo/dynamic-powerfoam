"""ring_yield.py on PAC-NeRF's plasticine scene 0 against its MPM particles, as merged_drop.py: the shape of elastic scene 0, the scene's E, nu, yield stress and start velocity, gravity, the ground at y = -0.05 with friction large enough for the MPM's sticky ground.
Per frame (1/24 s): height and widths (ours: the cap cells' outermost points), the centre of mass, Chamfer in both directions against 30000 of the particles. Frames saved on the way (x, radii, L, yielded), every --every steps; the log line-buffered; no timeout.
  python free_surface/ring_yield_drop.py --foam <dir with dem_exact_single> --save <dir> --tag <tag>"""
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
from free_surface.ring_yield import RingYield, YieldStep

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
a_.add_argument('--kappa', type=float, default=5.)
a_.add_argument('--eps-cut', type=float, default=.15)
a_.add_argument('--every', type=int, default=10)
a_.add_argument('--pre-every', type=int, default=4)
a_.add_argument('--newton-max', type=int, default=300)
a_.add_argument('--steps', type=int, default=0, help='stop after this step (0: at the last frame)')
a_.add_argument('--scene', type=int, default=0)
a_.add_argument('--watch', type=int, nargs='*', default=None)
a_.add_argument('--trace-from', type=int, default=None)
a_.add_argument('--aa', type=int, default=6, help='the outer iteration: Anderson over this many differences (0: none)')
a_.add_argument('--ground-mono', action='store_true')
a_.add_argument('--contact-point', action='store_true')
a_.add_argument('--contact-alpha', type=float, default=1.)
a_.add_argument('--br-lag', action='store_true', default=True)
a_.add_argument('--hold-after', type=int, default=3)
a_.add_argument('--no-relax', action='store_true', default=True)
a_.add_argument('--save-state', type=int, nargs='*', default=[], help='checkpoints after these steps')
a_.add_argument('--load-state', type=Path, default=None)
a_.add_argument('--first-every', type=int, default=0)
a_.add_argument('--first-from', type=int, default=240)
a_.add_argument('--watch-from', type=int, default=0)
a_.add_argument('--soft-tol', type=float, default=.1, help='of the weight: a step that ends over the tolerance but under this goes on, counted and logged (.1 is at most a .1 m/s kick to one site)')
a_.add_argument('--no-hold', action='store_true', help="the ground only presses; the MPM's sticky ground holds tension, up to the material's yield here")
args = a_.parse_args()
gt = json.loads((ROOT/'gt'/'plasticine.json').read_text())[str(args.scene)]
E, nu, ys = gt['E']/1000, gt['nu'], gt['ys']/1000
g = np.load(args.foam/'dem_exact_single'/'geometry.npz'); cl = np.load(args.foam/'dem_exact_single'/'cells.npz')
X0, r0, V0, C0 = g['points'], cl['radii'], g['masses'], g['contacts']; n = len(X0)
args.save.mkdir(parents=True, exist_ok=True)
Z = zipfile.ZipFile(ROOT/'simulation_data.zip')


def truth(k, scene=None):
    v = PlyData.read(io.BytesIO(Z.read('simulation_data/plasticine/%d/%d.ply' % (args.scene if scene is None else scene, k))))['vertex']
    return np.stack([v['x'], v['y'], v['z']], 1).astype(np.float64)+OFFSET


# every scene starts as scene 0 turned, particle by particle: the foam of scene 0 turned with it
p0_, pk_ = truth(0, 0), truth(0)
c0_, ck_ = p0_.mean(0), pk_.mean(0)
U_, _, Vt_ = np.linalg.svd((pk_-ck_).T@(p0_-c0_))
R_s = U_@np.diag([1., 1., np.sign(np.linalg.det(U_@Vt_))])@Vt_
print('scene %d: E %.1f, nu %.4f, yield %.1f; turned from scene 0 by %.1f degrees, particle residual %.1e' % (args.scene, gt['E'], nu, gt['ys'], np.degrees(np.arccos(np.clip((np.trace(R_s)-1)/2, -1, 1))), np.abs((p0_-c0_)@R_s.T-(pk_-ck_)).max()), flush=True)
X0 = ck_+(X0-c0_)@R_s.T; C0 = ck_+(C0-c0_)@R_s.T
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
W = RingYield(X0, r0, V0, g["pairs"], g["areas"], C0, E, nu, ys, theta=.5, kappa=args.kappa, eps_cut=args.eps_cut, ground_mono=args.ground_mono, contact_point=args.contact_point, contact_alpha=args.contact_alpha, br_lag=args.br_lag, hold=not args.no_hold, ledger=True)
W.set_ball((0., 1., 0.), -.05, args.mu)
W.start()
W.prepare()
up = torch.tensor([0., 1., 0.], dtype=f64, device=dev)
fg = -9.8*M3*up; wref = float(fg.norm()); tol = args.tol*wref
CG = TrustCG(W, mass, args.dt, per=2, product_iters=1)
W.update_ul(W.X)
W.grad(W.X); W.reread(); W.grad(W.X)
ST = YieldStep(W, CG, mass, W.ul_blocks(), fg, args.dt, args.mu)
ST.pre_ul = True; ST.pre_every = args.pre_every; ST.smooth_pred = ST.carry_delta = True
ST.set_theta(.5, mass)
ST.trace_from = args.trace_from
ST.aa_m = args.aa
ST.relax_on = not args.no_relax
ST.hold_after = args.hold_after


class _First:
    s = 0

    def __call__(self):
        return args.first_every > 0 and self.s >= args.first_from and self.s % args.first_every == 0


ST.trace_first = _First()
ST.set_state(W.X.clone(), torch.tensor(gt['v'], dtype=f64, device=dev).expand(n, 3).clone(), fg/M3)
h = float(np.median(V0))**(1/3)
print('plasticine, ring springs with the tensor limit: E %.1f, nu %.4f, yield %.1f Pa (yield strain %.4f, cell limit e_y %.4f), kappa %g, v0 %s; foam %s: %d cells, h %.2f mm; dt 1/%d, tol %g of the weight, friction %g; MPM %d particles' % (
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


def pack(o):
    out = {}
    for k_, v_ in o.__dict__.items():
        if isinstance(v_, (bool, int, float, str, type(None))):
            out[k_] = v_
        elif isinstance(v_, torch.Tensor):
            out[k_] = v_.detach().cpu().clone()
        elif isinstance(v_, wp.array):
            out[k_] = ('wp', v_.numpy().copy())
        elif isinstance(v_, (tuple, list)) and all(isinstance(a__, (torch.Tensor, int, float, type(None))) for a__ in v_):
            out[k_] = (type(v_).__name__, [a__.detach().cpu().clone() if isinstance(a__, torch.Tensor) else a__ for a__ in v_])
        elif isinstance(v_, dict):
            d_ = {}
            for k2, v2 in v_.items():
                if isinstance(v2, wp.array):
                    d_[k2] = ('wp', v2.numpy().copy())
                elif isinstance(v2, torch.Tensor):
                    d_[k2] = v2.detach().cpu().clone()
                elif isinstance(v2, (bool, int, float, str, type(None))):
                    d_[k2] = v2
            out[k_] = ('dict', d_)
    return out


def wp_put(a, v):
    t_ = wp.to_torch(a, requires_grad=False)
    t_.copy_(torch.as_tensor(v, device=t_.device, dtype=t_.dtype))


def unpack(o, st):
    for k_, v_ in st.items():
        cur = getattr(o, k_, None)
        if isinstance(v_, torch.Tensor):
            if isinstance(cur, torch.Tensor) and cur.shape == v_.shape:
                cur.copy_(v_.to(cur.device))
            else:
                setattr(o, k_, v_.to(dev))
        elif isinstance(v_, tuple) and v_[0] == 'wp':
            if isinstance(cur, wp.array):
                wp_put(cur, v_[1])
        elif isinstance(v_, tuple) and v_[0] == 'dict':
            if isinstance(cur, dict):
                for k2, v2 in v_[1].items():
                    c2 = cur.get(k2)
                    if isinstance(v2, tuple) and v2[0] == 'wp':
                        if isinstance(c2, wp.array):
                            wp_put(c2, v2[1])
                    elif isinstance(v2, torch.Tensor):
                        if isinstance(c2, torch.Tensor) and c2.shape == v2.shape:
                            c2.copy_(v2.to(c2.device))
                        else:
                            cur[k2] = v2.to(dev)
                    else:
                        cur[k2] = v2
            else:
                setattr(o, k_, {k2: (v2.to(dev) if isinstance(v2, torch.Tensor) else v2) for k2, v2 in v_[1].items()})
        elif isinstance(v_, tuple) and v_[0] in ('tuple', 'list'):
            a__ = [b_.to(dev) if isinstance(b_, torch.Tensor) else b_ for b_ in v_[1]]
            setattr(o, k_, tuple(a__) if v_[0] == 'tuple' else a__)
        else:
            setattr(o, k_, v_)


s_first = 1
if args.load_state is not None:
    ck = torch.load(args.load_state, weights_only=False)
    unpack(W, ck['W']); unpack(ST, ck['ST'])
    ST.trace_from, ST.aa_m, ST.hold_after, ST.relax_on = args.trace_from, args.aa, args.hold_after, not args.no_relax        # (the command line over the checkpoint's)
    W.br_lag, W.contact_point, W.ground_mono = args.br_lag, args.contact_point, args.ground_mono
    s_first = ck['s']+1
    print('from step %d of %s' % (ck['s'], args.load_state), flush=True)
t0 = t_saved = time.time()
nit, npr, nrd, ncl, nfe = [], [], [], [], []
nsoft = 0
nf = int(round(1/(24*args.dt))); nsteps = args.steps or args.frames*nf
H0 = .5*float((M3*ST.tv**2).sum())+float((-fg*ST.tx).sum())
W.watch = args.watch
for s in range(s_first, nsteps+1):
    t_s = time.time()
    W.watch_on = bool(args.watch) and s >= args.watch_from
    ST.trace_first.s = s
    if W.watch_on:
        print('  step %d' % s, flush=True)
    e_, its, g2 = ST.step(tol, lambda g_: .5*tol)
    if math.isfinite(g2) and tol < math.sqrt(g2) <= args.soft_tol*wref:
        nsoft += 1
        print('  step %d, t %.4f: over the tolerance, |g| / weight %.4f after %d Newton iterations; going on' % (s, s*args.dt, math.sqrt(g2)/wref, its), flush=True)
    elif not math.isfinite(g2) or math.sqrt(g2) > tol:
        tr_ = ST.stats['trace']
        g_ = wp_t(ST.a['gz']).norm(dim=1); top = torch.argsort(g_, descending=True)[:8]
        sf = torch.linalg.svdvals(W.tFc.cpu()).to(dev)
        print('  at the failure: F_o smallest singular value min %.3f, largest max %.3f; beta min %.3f; rho / e_y max %.1f' % (float(sf[:, 2].min()), float(sf[:, 0].max()), float(W.ty['be'].min()), float(W.ty['ei'].max())/W.ey), flush=True)
        Cn_ = W.tul['Q'].flatten(1).norm(dim=1); oc_ = torch.argsort(Cn_, descending=True)[:5]
        print('  the frozen C: |C| median %.2f, 99 %% %.2f, largest %s at cells %s, V / median %s; cells with |C| over 6: %d; first |g| %.3g weights' % (float(Cn_.median()), float(torch.quantile(Cn_, .99)), ['%.1f' % float(Cn_[c_]) for c_ in oc_], oc_.tolist(), ['%.2f' % float(W.V0[c_]/W.V0.median()) for c_ in oc_], int((Cn_ > 6).sum()), tr_[0][4]/wref if tr_ else 0.), flush=True)
        print('  largest |g| cells (index: |g| / weight; F_o\'s singular values; beta; yielded; V0 / median; height / h):', flush=True)
        for c_ in top.tolist():
            print('    %d: %.2f; %.3f %.3f; beta %.3f; %d; V %.2f; y %.1f' % (c_, float(g_[c_])/wref, float(sf[c_, 2]), float(sf[c_, 0]), float(W.ty['be'][W._gid[c_]]), int(W._yld[c_]), float(W.V0[c_]/W.V0.median()), float(ST.tz[c_, 1]+.05)/h), flush=True)
        np.savez(args.save/('fail_%s.npz' % args.tag), x=ST.tx.cpu().numpy(), v=ST.tv.cpu().numpy(), a=ST.ta.cpu().numpy(), z=ST.tz.cpu().numpy(), step=s, g=wp_t(ST.a['gz']).cpu().numpy())
        print('step %d, t %.4f: the solve did not converge (|g| / weight %s after %d Newton iterations); CG %s; |g| %s; ratio %s' % (s, s*args.dt, math.sqrt(g2)/wref if math.isfinite(g2) else g2, its, [a__[0] for a__ in tr_][:30], ['%.3g' % (a__[4]/wref) for a__ in tr_][:30], ['%.2g' % a__[5] for a__ in tr_][:30]), flush=True)
        break
    ST.end()
    nit.append(its); npr.append(e_); nrd.append(ST.stats['outer']); ncl.append(int(W.ul_stats['cut_cells'])); nfe.append(ST.stats['rejected'])
    if s in args.save_state:
        torch.save(dict(W=pack(W), ST=pack(ST), s=s), args.save/('state_%s_%d.pt' % (args.tag, s)))
    if s % args.every == 0:
        xs.append(ST.tx.cpu().numpy()); rs.append(W._ulr.cpu().numpy()); Ls.append(W.tul['Ln'].cpu().numpy()); ys_.append(W._yld.cpu().numpy()); ts.append(s*args.dt)
    if s % 20 == 0:
        i_, j_, Lw, _, _, m = W._w
        ek = .5*float((M3*ST.tv**2).sum()); ed, ev = W.stored(); eg = float((-fg*ST.tx).sum()); eb = W.ground_store(ST.tx)
        print('  work of the springs over the steps: the tensor %.4e, the remainder %.4e; plastic work by the formula, cells %.4e, halves %.4e' % (getattr(W, 'W_tensor', 0.), getattr(W, 'W_rem', 0.), W.Wp_cell, W.Wp_half), flush=True)
        print('  the cut made, tensor volumetric, cross, deviatoric, remainder: on beta under .5 %s; the rest %s' % (['%.2e' % a__ for a__ in W.Ecut_t[:4]], ['%.2e' % a__ for a__ in W.Ecut_t[4:]]), flush=True)
        print('  the cut made: springs %.3e, outer %.3e, volume %.3e, ground %.3e (without the carried fit: springs %.3e, outer %.3e); T - I median %.4f, max %.3f, reset %d; fit gap %.1e; H_n / e_y median %.2f, max %.1f; springs v against L: gap median %.4f, max %.3f, |v| max %.3f' % (*W.Ecut_parts, *W.Ecut_w, W.ul_stats.get('T_med', 0.), W.ul_stats.get('T_max', 0.), W.ul_stats.get('T_reset', 0), W.ul_stats.get('fit_gap', 0.), W.ul_stats.get('Hn_med', 0.), W.ul_stats.get('Hn_max', 0.), W.ul_stats.get('v_gap_med', 0.), W.ul_stats.get('v_gap_max', 0.), W.ul_stats.get('v_max', 0.)), flush=True)
        bd = W.B['bdelta'].numpy()
        print('t %.4f (%.0f s): KE %.3e springs+outer %.3e vol %.3e ground %.3e W_p %.3e, flags made %.2e, cuts made %.2e; left %.4f of the start\'s energy; com y %.4f; flowing %d cells, yielded so far %d, on the cut %d; outer %d; walls %d, springs %d, new %d, closed %d, without a wall %d; J %.3f..%.3f; touching %d, ground/weight %.2f; min wall L / h %.3f, max r / h %.2f, radii %d (fails %d), flips %d; rho / e_y median %.2f, max %.1f, beta min %.3f; products %d, Newton %d, rejected %d, negative %d; %.2f s' % (
            s*args.dt, time.time()-t0, ek, ed, ev, eb, W.Wp, W.Eflag, W.Ecut, 1-(ek+ed+ev+eb+W.Wp+eg-W.Eflag-W.Ecut)/H0, float((M3*ST.tx)[:, 1].sum())/M, W.ul_stats['yielding'], W.ul_stats['yielded'], W.ul_stats['cut_cells'], ST.stats['outer'], m, W._m_s, W.ul_stats['born'], W.ul_stats['closed'], W.ul_stats['ghost'], float(torch.exp(W.tul['Ln']).min()), float(torch.exp(W.tul['Ln']).max()),
            int((bd > 0).sum()), float(W.B['bfn'].numpy().sum())/wref, float(Lw.min())/h, float(W._ulr.max())/h, W.ul_stats['radii_it'], W.ul_stats['radii_fail'], W.ul_stats['flips'], float(W.ty['ei'][W._gid == torch.arange(n, device=dev)].median())/W.ey, float(W.ty['ei'].max())/W.ey, float(W.ty['be'].min()), e_, its, ST.stats['rejected'], ST.stats['negative'], time.time()-t_s), flush=True)
    if s % nf == 0:
        rows.append(row(s//nf))
        save(); t_saved = time.time()
    elif time.time()-t_saved > 60:
        save(); t_saved = time.time()
save()
ni, np_ = np.array(nit), np.array(npr)
print('Newton per step: median %d, 99 %% %d, max %d, steps above 50: %d of %d; products per step: median %d, max %d, total %d; plastic work %.4f (cells %.4f, halves %.4f); outer iterations per step median %d, max %d; on the cut at the end %d; rejected %d; steps over the tolerance %d' % (np.median(ni), np.percentile(ni, 99), ni.max(), int((ni > 50).sum()), len(ni), np.median(np_), np_.max(), np_.sum(), W.Wp, W.Wp_cell, W.Wp_half, int(np.median(nrd)), int(np.max(nrd)), int(ncl[-1]) if ncl else 0, int(np.sum(nfe)), nsoft), flush=True)
print('done: %.0f s' % (time.time()-t0), flush=True)
