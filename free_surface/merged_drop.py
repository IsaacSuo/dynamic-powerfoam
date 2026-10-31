"""merged_ul.py on PAC-NeRF's plasticine scene 0 against its MPM particles (2026-10-31): the shape of elastic scene 0 (frame 0 identical), the scene's E, nu, yield stress and start
velocity (data/pacnerf/gt/plasticine.json), gravity, the ground at y = -0.05 with friction large enough for the MPM's sticky ground (ours = theirs + (0.5, -0.05, 0.5), as plastic_drop.py).
Per frame (1/24 s): height and widths (ours: the cap cells' outermost points), the centre of mass, Chamfer in both directions against 30000 of the particles.
Frames saved on the way (x, radii, L, ep, yielded), every --every steps; the log line-buffered; no timeout.
  python free_surface/merged_drop.py --foam <dir with dem_exact_single> --save <dir> --tag <tag>"""
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
from free_surface.merged_ul import MergedStep, MergedUL

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
a_.add_argument('--mu', type=float, default=10., help="the ground's friction: large for the MPM's sticky ground; with --stick shear it only turns the glue on")
a_.add_argument('--stick', choices=('coulomb', 'shear'), default='coulomb', help="the glue's limit: mu F_n, or the material's shear yield on the cell's footprint (PlasticUL.set_stick)")
a_.add_argument('--grad', default='gg')
a_.add_argument('--every', type=int, default=10)
a_.add_argument('--pre-every', type=int, default=4)
a_.add_argument('--log-from', type=int, default=10**9, help="every step from this one on: the start's |g|, the balls' overlap and radii, the Newton trace")
a_.add_argument('--newton-max', type=int, default=300)
a_.add_argument('--theta', type=float, default=.5)
a_.add_argument('--kshare', type=float, default=1.)
a_.add_argument('--log-mem', type=int, default=10**9)
a_.add_argument('--probe-from', type=int, default=10**9, help="every step from this one on, at its first point: the springs' part of |g|, the levers F T, the neighbours' relative rotation (logging only)")
a_.add_argument('--budget-from', type=int, default=10**9, help="every step from this one on: the energy account by channel, each one's stored change minus its forces' work over the step (logging only)")
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
W = MergedUL(X0, r0, V0, g['pairs'], g['areas'], g['contacts'], E, nu, ys, theta=args.theta, kshare=args.kshare, grad=args.grad)
W.set_ball((0., 1., 0.), -.05, args.mu)
if args.stick == 'shear':
    W.set_stick()
W.update_ul(W.X)
W.prepare()
up = torch.tensor([0., 1., 0.], dtype=f64, device=dev)
fg = -9.8*M3*up; wref = float(fg.norm()); tol = args.tol*wref
CG = TrustCG(W, mass, args.dt, per=2, product_iters=1)
ST = MergedStep(W, CG, mass, W.ul_blocks(), fg, args.dt, args.mu)
ST.pre_ul = True; ST.pre_every = args.pre_every; ST.smooth_pred = ST.carry_delta = True; ST.newton_max = args.newton_max
ST.set_theta(args.theta, mass)
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
xs, rs, Ls, ys_, eps, ts = [], [], [], [], [], []


def save():
    np.savez(args.save/('frames_%s.npz' % args.tag), x=np.array(xs), r=np.array(rs), L=np.array(Ls), yld=np.array(ys_), ep=np.array(eps), t=np.array(ts), rows=np.array(rows))
    if probes:
        np.save(args.save/('probe_%s.npy' % args.tag), np.array(probes))


probes = []


def probe(ST_):
    """at the step's first point, after the re-read: the springs' part of |g| (their k off for one evaluation, then back), the levers p of their chain through G, the neighbours' relative rotation (F no longer enters the springs). Logging only"""
    s_, m_, lo, hi = W.tsp, W._m_s, W.tlo, W.thi
    g_all = wp_t(ST_.a['gz']).clone()
    sk_ = s_['sk'].clone(); s_['sk'].zero_()
    ST_._eval(0.); g_no = wp_t(ST_.a['gz']).clone()
    s_['sk'].copy_(sk_)
    ST_._eval(0.); ST_._accept()
    n_all, n_no, n_sp = float(g_all.norm())/wref, float(g_no.norm())/wref, float((g_all-g_no).norm())/wref
    gn = g_all.norm(dim=1); top = int(gn.argmax())
    F0, T, k_, b_, r_, dn_ = s_['F0'], s_['sP'][:m_], s_['sk'][:m_], s_['sb'][:m_], s_['sr'][:m_], s_['sdn'][:m_]
    dl = (ST_.tx[hi]-ST_.tx[lo]).norm(dim=1)
    ai = aj = T.norm(dim=1)
    lev = torch.maximum(ai, aj)/dl
    sg = torch.linalg.svdvals((.5*(F0[lo]+F0[hi])).cpu())[:, 2].to(dev)
    U, _, Vh = torch.linalg.svd(F0.cpu())
    U[:, :, 2] *= torch.sign(torch.linalg.det(U@Vh))[:, None]
    R = (U@Vh).to(dev)
    ang = torch.rad2deg(torch.acos((((R[lo]*R[hi]).sum((1, 2))-1)/2).clamp(-1, 1)))
    Vi = W.tul['Vi']
    f_ = k_*(((1-W.th)*dn_+W.th*b_[:, None]*r_).norm(dim=1))
    e_ = int((f_*torch.maximum(ai*Vi[lo], aj*Vi[hi])).argmax()); c1, c2 = int(lo[e_]), int(hi[e_])
    ring = torch.cat([hi[lo == c1], lo[hi == c1], hi[lo == c2], lo[hi == c2], torch.tensor([c1, c2], device=dev)])
    q_ = lambda a__, p_: float(torch.quantile(a__, p_))
    med = W.V0.median()
    print('  lever step %d t %.4f: first |g| %.3g, without the springs %.3g, the springs\' part %.3g; top site %d, |g| %.3g, V0/med %.2f; Fbar smallest sigma: min %.3g, 0.1 %% %.3f, 1 %% %.3f; |p| / |d|: median %.2f, 99 %% %.2f, 99.9 %% %.2f, max %.3g, above 3: %d, 10: %d, 100: %d; |p| / |d| max %.3g; relative rotation in degrees: median %.1f, 99 %% %.1f, 99.9 %% %.1f, max %.2f, above 90: %d, 150: %d; |r| / |d| at the start max %.3g; the top spring (%d, %d): lever %.3g, sigma %.3g, rotation %.2f, |r| / |d| %.3g, |G_i - G_j| %.3g, k %.3g, beta %.3f, V0/med %.2f %.2f; the top site in its rings: %d' % (
        s, s*args.dt, n_all, n_no, n_sp, top, float(gn[top])/wref, float(W.V0[top]/med), float(sg.min()), q_(sg, .001), q_(sg, .01), q_(lev, .5), q_(lev, .99), q_(lev, .999), float(lev.max()), int((lev > 3).sum()), int((lev > 10).sum()), int((lev > 100).sum()), float((T.norm(dim=1)/dl).max()),
        q_(ang, .5), q_(ang, .99), q_(ang, .999), float(ang.max()), int((ang > 90).sum()), int((ang > 150).sum()), float((r_.norm(dim=1)/dl).max()), c1, c2, float(lev[e_]), float(sg[e_]), float(ang[e_]), float(r_[e_].norm()/dl[e_]), float((W.tG[c1]-W.tG[c2]).norm()), float(k_[e_]), float(b_[e_]), float(W.V0[c1]/med), float(W.V0[c2]/med), int(bool((ring == top).any()))), flush=True)
    probes.append((s, s*args.dt, n_all, n_no, n_sp, float(sg.min()), q_(lev, .999), float(lev.max()), q_(ang, .999), float(ang.max()), float((r_.norm(dim=1)/dl).max()), float(lev[e_]), float(sg[e_]), int(bool((ring == top).any()))))
    # the first |g| by what the cells put on their walls: pressure and variance, shape; the ground's largest ball
    Lb_, Fb_ = wp_t(W.B['Lb']), wp_t(W.B['Fb'])
    ic_, jc_, m2_ = W._w[0], W._w[1], W._w[5]; wa_ = W.tul['wa'][:m2_]
    Am = torch.zeros(W.n, dtype=f64, device=dev); Am.index_reduce_(0, ic_, wa_, 'amax'); Am.index_reduce_(0, jc_, wa_, 'amax')
    pr, sh = Lb_.abs()*Vi*Am*3**.5/wref, (Fb_@F0.transpose(1, 2)).flatten(1).norm(dim=1)*Vi*Am/wref
    fb_ = torch.as_tensor(W.B['bfn'].numpy(), device=dev).abs()/wref

    def cell(c_):
        sv_, sf_ = torch.linalg.svdvals((W.I3+W.tG[c_]).cpu()), torch.linalg.svdvals(F0[c_].cpu())
        return '%d: V0/med %.2f, V / (V0 J) %.3f, J %.3f, L(z) %.3f, beta %.3f, e_i / e_y %.1f, I + G %.3f %.3f, F %.3f %.3f, y/h %.1f' % (c_, float(W.V0[c_]/med), float(1/(Vi[c_]*W.V0[c_]*torch.exp(W.tul['Ln'][c_]))), float(torch.exp(W.tul['Ln'][c_])), float(W.tL[c_]), float(W.tbe[c_]), float(W.tei[c_])/W.ey, float(sv_[2]), float(sv_[0]), float(sf_[2]), float(sf_[0]), float(ST_.tz[c_, 1]+.05)/h)
    cp_, cs_ = int(pr.argmax()), int(sh.argmax())
    print('  channels step %d: pressure |dE/dL| A / V max %.3g at %s; shape max %.3g at %s; ground largest ball %.3g at %d' % (s, float(pr[cp_]), cell(cp_), float(sh[cs_]), cell(cs_), float(fb_.max()), int(fb_.argmax())), flush=True)


budgets = []
bud = dict(prev=None, cum=np.zeros(6))


def energies():
    """stored shape and volume after the last end, kinetic, gravity's, the plastic work (cells, springs) of the state now"""
    E_s, E_v = W.stored()
    return dict(s=E_s, v=E_v, k=.5*float((M3*ST.tv**2).sum()), g=float((-fg*ST.tx).sum()), wc=float(getattr(W, 'Wp_cell', 0.)), ws=float(getattr(W, 'Wp_spring', 0.)))


def budget_mid():
    """at the accepted z, before the step's end: the forces' work over the step by channel, gradient . dx with dx = (z - x_n) / th, G linear in x"""
    th, G, Q, F0 = W.th, W.tG, W.tul['Q'], W.tsp['F0']
    s_, m_ = W.tsp, W._m_s
    k_, b_, r_, dn_ = s_['sk'][:m_], s_['sb'][:m_], s_['sr'][:m_], s_['sdn'][:m_]
    f_ = k_[:, None]*((1-th)*dn_+th*b_[:, None]*r_)
    t_, mw = W.tul, W._w[5]
    A_ = G/th
    return dict(WL=wp_t(W.B['Lb'])*(Q*G).sum((1, 2))/th, Ws=float((wp_t(W.B['Fb'])*(G@F0)).sum())/th, Wsp=float((f_*(r_-dn_)).sum()), Wt=float((W.tgx*(ST.tz-ST.tx)).sum())/th, Ln0=t_['Ln'].clone(), Lz=W.tL.clone(), A=A_.clone(),
                Esp0=float((.5*k_*(dn_*dn_).sum(1)).sum()), wv=t_['wv'][:mw].clone(), wi=t_['wi'][:mw].long().clone(), wj=t_['wj'][:mw].long().clone(), Q=Q.clone(), G=G.clone())


def budget_end(e0, m):
    """after the step's end: each channel's stored change minus its work; what is left of the balance is the ground's work"""
    th, kap, V0_ = W.th, W.kap, W.V0
    e1 = energies()
    Ln1 = W.tul['Ln']; Ln0, Lz = m['Ln0'], m['Lz']
    Evar0, Evar1 = float((m['wv']*(Ln0[m['wi']]-Ln0[m['wj']])**2).sum()), float((m['wv']*(Ln1[m['wi']]-Ln1[m['wj']])**2).sum())
    Esp1 = W.spring_store
    WL = float(m['WL'].sum())
    D_L = (e1['v']-e0['v'])+(Evar1-Evar0)-WL
    D_s = (e1['s']-e0['s'])+(e1['wc']-e0['wc'])-m['Ws']
    D_sp = (Esp1-m['Esp0'])+(e1['ws']-e0['ws'])-m['Wsp']
    W_c = m['Wt']-WL-m['Ws']-m['Wsp']
    pv = bud['prev']
    jk, jv = (m['Esp0']-pv['Esp'], Evar0-pv['Evar']) if pv is not None else (0., 0.)
    bud['prev'] = dict(Esp=Esp1, Evar=Evar1)
    bud['cum'] += np.array([D_L, D_s, D_sp, jk, jv, -W_c])/H0
    Dc = .5*V0_*kap*(Ln1**2-Ln0**2)-V0_*kap*Lz*(m['Q']*m['G']).sum((1, 2))/th
    A_ = m['A']; tr2 = (A_*A_.transpose(1, 2)).sum((1, 2))
    form = -V0_*kap*Lz*.5*tr2
    na = A_.flatten(1).norm(dim=1)
    c_ = int(Dc.argmax())
    sv_ = torch.linalg.svdvals((W.I3+A_[c_]).cpu())
    left = 1-(e1['k']+e1['g']+e1['s']+e1['v']+Evar1+Esp1+W.Wp)/H0
    print('  budget step %d t %.4f: left %.4f; cumulative / H0: pressure and variance %.5f, shape %.5f, springs %.5f, the springs\' jump at the cut %.5f, variance\'s jump %.5f, into the ground %.5f; this step / H0: pressure %.3e (formula kap V0 L(z) (-1/2 tr A^2) %.3e; cells above 0 %.3e, below 0 %.3e); kinetic + gravity + work %.2e; top cell %d: %.3e, L(z) %.3f, J %.3f -> %.3f, tr A^2 %.3f, I + A %.3f %.3f, V0/med %.2f, y/h %.1f; cells with |A| above 0.3: %d, 1: %d' % (
        s, s*args.dt, left, *bud['cum'].tolist(), float(Dc.sum())/H0, float(form.sum())/H0, float(Dc[Dc > 0].sum())/H0, float(Dc[Dc < 0].sum())/H0, (e1['k']-e0['k']+e1['g']-e0['g']+m['Wt'])/H0, c_, float(Dc[c_])/H0, float(Lz[c_]), float(torch.exp(Ln0[c_])), float(torch.exp(Ln1[c_])), float(tr2[c_]), float(sv_[2]), float(sv_[0]), float(V0_[c_]/V0_.median()), float(ST.tx[c_, 1]+.05)/h, int((na > .3).sum()), int((na > 1).sum())), flush=True)
    budgets.append((s, s*args.dt, left, *bud['cum'].tolist(), float(Dc.sum())/H0, float(form.sum())/H0, float(Dc[Dc > 0].sum())/H0, float(Dc[Dc < 0].sum())/H0, c_, float(Dc[c_])/H0))
    np.save(args.save/('budget_%s.npy' % args.tag), np.array(budgets))


t0 = t_saved = time.time()
Y2 = float((mass*(W.X[:, 1]-(M3*W.X).sum(0)[1]/M)**2).sum())
nit, npr = [], []
nf = int(round(1/(24*args.dt))); nsteps = args.steps or args.frames*nf
H0 = .5*float((M3*ST.tv**2).sum())+float((-fg*ST.tx).sum())
for s in range(1, nsteps+1):
    t_s = time.time()
    R_prev = W.tbR.clone()
    yb_ = W._yld.clone()
    ST.probe = probe if s >= args.probe_from else None
    bud_e0 = energies() if s >= args.budget_from else None
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
        g_ = wp_t(ST.a['gz']).norm(dim=1); top = torch.argsort(g_, descending=True)[:8]; I3_ = W.I3
        Gz = W.tG; sv = torch.linalg.svdvals(I3_+Gz); sf = torch.linalg.svdvals(W.tFc); dt_ = torch.linalg.det(I3_+Gz)
        print('  at the failure: I + G(z) smallest singular value 1%% %.3f, min %.3f, cells below 0.2: %d, det <= 0: %d; F(z) smallest singular value min %.3f, largest max %.3f; beta min %.3f; e_i / e_y max %.1f' % (float(torch.quantile(sv[:, 2], .01)), float(sv[:, 2].min()), int((sv[:, 2] < .2).sum()), int((dt_ <= 0).sum()), float(sf[:, 2].min()), float(sf[:, 0].max()), float(W.tbe.min()), float(W.tei.max())/W.ey), flush=True)
        print('  largest |g| cells (index: |g| / weight; smallest and largest singular value of I + G, of F; beta; yielded; V0 / median; height / h):', flush=True)
        for c_ in top.tolist():
            print('    %d: %.2f; %.3f %.3f; %.3f %.3f; beta %.3f; %d; V %.2f; y %.1f' % (c_, float(g_[c_])/wref, float(sv[c_, 2]), float(sv[c_, 0]), float(sf[c_, 2]), float(sf[c_, 0]), float(W.tbe[c_]), int(W._yld[c_]), float(W.V0[c_]/W.V0.median()), float(ST.tz[c_, 1]+.05)/h), flush=True)
        np.savez(args.save/('fail_%s.npz' % args.tag), x=ST.tx.cpu().numpy(), v=ST.tv.cpu().numpy(), a=ST.ta.cpu().numpy(), z=ST.tz.cpu().numpy(), step=s, g=wp_t(ST.a['gz']).cpu().numpy())
        print('step %d, t %.4f: the solve did not converge (|g| / weight %s after %d Newton iterations); CG %s; |g| %s; ratio %s' % (s, s*args.dt, math.sqrt(g2)/wref if math.isfinite(g2) else g2, its, [a__[0] for a__ in tr_][:30], ['%.3g' % (a__[4]/wref) for a__ in tr_][:30], ['%.2g' % a__[5] for a__ in tr_][:30]), flush=True)
        break
    if s >= args.log_mem:
        ov_ = len(W._recut.A.ov[0]) if hasattr(W._recut.A, 'ov') else 0
        print('  mem step %d t %.4f: torch allocated %.2f GB, reserved %.2f GB, peak allocated %.2f GB; overlap pairs %d; walls %d; yielded %d; r / r0 max %.2f; J min %.3f; Newton %d, products %d' % (s, s*args.dt, torch.cuda.memory_allocated()/2**30, torch.cuda.memory_reserved()/2**30, torch.cuda.max_memory_allocated()/2**30, ov_, W.ul_stats['walls'], int(W._yld.sum()), float((W._ulr/W._rr0).max()), float(torch.exp(W.tul['Ln']).min()), its, e_), flush=True)
    bud_m = budget_mid() if s >= args.budget_from else None
    ST.end()
    if s >= args.budget_from:
        budget_end(bud_e0, bud_m)
    nit.append(its); npr.append(e_)
    if s % args.every == 0:
        xs.append(ST.tx.cpu().numpy()); rs.append(W._ulr.cpu().numpy()); Ls.append(W.tul['Ln'].cpu().numpy()); ys_.append(W._yld.cpu().numpy()); eps.append(W.ep.cpu().numpy()); ts.append(s*args.dt)
    if s % 20 == 0:
        i_, j_, Lw, _, _, m = W._w
        ek = .5*float((M3*ST.tv**2).sum()); ed, ev = W.stored(); eg = float((-fg*ST.tx).sum())
        bd = W.B['bdelta'].numpy()
        Es_ = float(((M3*(ST.tx-ST.tx.mean(0)))[:, 1]*(ST.tx[:, 1]-(M3*ST.tx).sum(0)[1]/M)).sum())
        print('  integrated mean strain yy %.4f xx %.4f zz %.4f; site height std / rest %.4f, its log %.4f; mean |ep| %.3f; springs %d, new %d, closed %d this step, without a wall %d; smallest one-cell dissipation / limit store %.2e' % (float(W.Hsum[1, 1]), float(W.Hsum[0, 0]), float(W.Hsum[2, 2]), (Es_/Y2)**.5, .5*math.log(Es_/Y2), float((mass*W.ep.flatten(1).norm(dim=1)).sum())/M, W.ul_stats['rows'], W.ul_stats['born'], W.ul_stats['closed'], W.ul_stats['ghost'], W.ul_stats.get('D_min', 0.)), flush=True)
        print('t %.4f (%.0f s): KE %.3e dev %.3e vol %.3e var %.3e springs %.3e W_p %.3e (springs %.3f), left %.3f of the start\'s energy; com y %.4f; yielding %d cells, yielded so far %d; walls %d; J %.3f..%.3f, |eps_p| median %.3f max %.3f; touching %d, ground/weight %.2f; min wall L / h %.3f, max r / h %.2f, radii %d (fails %d), flips %d; products %d, Newton %d, rejected %d, negative %d; %.2f s' % (
            s*args.dt, time.time()-t0, ek, ed, ev, float(W.tsum[2]), W.spring_store, W.Wp, W.Wp_spring/max(W.Wp, 1e-300), 1-(ek+ed+ev+float(W.tsum[2])+W.spring_store+W.Wp+eg)/H0, float((M3*ST.tx)[:, 1].sum())/M, W.ul_stats['yielding'], W.ul_stats['yielded'], m, float(torch.exp(W.tul['Ln']).min()), float(torch.exp(W.tul['Ln']).max()),
            float(W.ep.flatten(1).norm(dim=1).median()), float(W.ep.flatten(1).norm(dim=1).max()), int((bd > 0).sum()), float(W.B['bfn'].numpy().sum())/wref, float(Lw.min())/h, float(W._ulr.max())/h, W.ul_stats['radii_it'], W.ul_stats['radii_fail'], W.ul_stats['flips'], e_, its, ST.stats['rejected'], ST.stats['negative'], time.time()-t_s), flush=True)
    if s % nf == 0:
        rows.append(row(s//nf))
        save(); t_saved = time.time()
    elif time.time()-t_saved > 60:
        save(); t_saved = time.time()
save()
ni, np_ = np.array(nit), np.array(npr)
print('Newton per step: median %d, 99 %% %d, max %d, steps above 50: %d of %d; products per step: median %d, max %d, total %d; plastic work %.4f' % (np.median(ni), np.percentile(ni, 99), ni.max(), int((ni > 50).sum()), len(ni), np.median(np_), np_.max(), np_.sum(), W.Wp), flush=True)
print('done: %.0f s' % (time.time()-t0), flush=True)
