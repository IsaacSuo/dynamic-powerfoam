"""Material identification of the Vid2Sim lion with the old DEM (fitted skin, one-way cap, Hertz ground, no self contact,
as dem_contact_lion.py's dem_old1 run on 10-31: the smoothed start and the carried region on), pure elasticity.

Unknowns theta = (ln E, nu, ln beta); fixed from the dataset: the foam's rest state (inverse_data/lion/foam, the
fourier_lion_8k_round_clean cells), start velocity (0, 0, -g dt / 2) with Vid2Sim's dt 0.05 (vid2sim_to_springgaus.py),
gravity -z, the ground z = -0.2 in the foam's frame (-0.7 in the world: x_world = x_foam - offset), frictionless (mu 0).
No self contact: Vid2Sim's ground truth has none (user, 11-06). beta is bounded below by --beta-min (default 0.05: the
lion's one-way cap needs it, user 10-29).

Observation: per view and frame (20 fps, frame k at t = k / 20 s, 60 steps of 1/1200 s) the silhouette of the union of the
cells' spheres (centre x, radius r0 det(F_B)^(1/3), r0 the foam's radii; at rest this union is exactly the material
region) by an anti-aliased ray-sphere test (Warp), reduced to identify.moments' features: centroid, second-moment widths,
sqrt(area), in pixels. Ground truth: the alpha masks a_<view>_<frame> (inverse_data/lion/masks, vid2sim_to_springgaus.py).
Protocol (10-18, Vid2Sim benchmark): fit frames 1..15 in all views but the held one (5); report the held view and the
future frames 16..23 (IoU of the silhouettes at 0.5), log10 E and nu errors against gt_phys_params (E 10648.6, nu 0.391).

Commands (run from the repository root; python with warp 1.10, torch 2.9, as requirements-sim.txt):
  python -u free_surface/lion_inverse.py check                    frame 0: our silhouette vs the masks per view (no simulation)
  python -u free_surface/lion_inverse.py run --theta lnE nu lnbeta --frames K --out f.npz [--tol 1e-3] [--save-frames]
                                                                  one forward (one process, one material): features per frame
  python -u free_surface/lion_inverse.py lm --out DIR [--obs real|synth --truth ...] [--start ...] [--jobs N]
                                                                  Levenberg-Marquardt, the Jacobian by central differences of
                                                                  whole runs, all of an iteration's runs in parallel processes
  python -u free_surface/lion_inverse.py eval --theta ... --out DIR   frames 0..23, IoU per view and frame, the errors
A run whose step does not converge exits 1 (its trial is a rejection in lm). No timeout anywhere."""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = ROOT/'inverse_data'/'lion'
DT, UP, C0, V0 = 1/1200, 2, -.2, (0., 0., -.245)
FPS, SPF, NFRAMES = 20, 60, 24
GT = dict(E=10648.614517785627, nu=.3912951704721983)
PARAMS = ('lnE', 'nu', 'lnbeta')
NU_RANGE = (.01, .49)


# ---------------------------------------------------------------- the observation
def cameras():
    meta = json.loads((DATA/'cameras.json').read_text())
    off = np.array(json.loads((DATA/'conversion.json').read_text())['offset'], float)
    cams = [meta['cameras'][nm] for nm in meta['camera_names']]
    return meta, cams, off


def gt_masks():
    """(views, frames, H, W) alpha in [0, 1], float32"""
    from PIL import Image
    meta, _, _ = cameras()
    return np.stack([np.stack([np.asarray(Image.open(DATA/'masks'/nm/('%02d.png' % f)), np.float32)/255. for f in range(NFRAMES)])
                     for nm in meta['camera_names']])


def moments_np(a):
    """identify.moments for one map: centroid x, y, widths x, y, sqrt(area) (pixel centres at integer coordinates)"""
    H, W = a.shape
    ys, xs = np.arange(H, dtype=np.float64)[:, None], np.arange(W, dtype=np.float64)[None, :]
    m0 = max(float(a.sum()), 1e-6)
    cx, cy = float((a*xs).sum())/m0, float((a*ys).sum())/m0
    sx = math.sqrt(max(float((a*(xs-cx)**2).sum())/m0, 1e-12)); sy = math.sqrt(max(float((a*(ys-cy)**2).sum())/m0, 1e-12))
    return np.array([cx, cy, sx, sy, math.sqrt(m0)])


def gt_features():
    path = DATA/'gt_features.npz'
    if path.exists():
        return np.load(path)['f']
    A = gt_masks()
    f = np.stack([np.stack([moments_np(A[v, k].astype(np.float64)) for v in range(A.shape[0])]) for k in range(NFRAMES)])
    np.savez(path, f=f)                                                              # (frames, views, 5)
    return f


class Silhouette:
    """the union of spheres seen by each camera: per sphere its pixels' coverage clamp(1/2 + (r - d) f / t, 0, 1) (d the
    centre's distance to the pixel's ray, t its depth along the ray: one pixel of anti-aliasing), the maximum over spheres"""

    def __init__(self):
        import torch
        import warp as wp
        self.torch, self.wp = torch, wp
        meta, cams, off = cameras()
        self.W, self.H = meta['width'], meta['height']
        self.R = torch.tensor(np.array([c['R'] for c in cams]), dtype=torch.float64, device='cuda')
        self.t = torch.tensor(np.array([c['t'] for c in cams]), dtype=torch.float64, device='cuda')
        self.K = np.array([c['K'] for c in cams])
        self.off = torch.tensor(off, dtype=torch.float64, device='cuda')
        self.nv = len(cams)
        self.img = wp.zeros((self.nv, self.H, self.W), dtype=wp.float32, device='cuda')
        self.timg = wp.to_torch(self.img)

        @wp.kernel
        def k_sil(pc: wp.array2d(dtype=wp.vec3f), r: wp.array(dtype=wp.float32), fx: wp.array(dtype=wp.float32),
                  cx: wp.array(dtype=wp.float32), cy: wp.array(dtype=wp.float32), Wd: int, Hd: int, img: wp.array3d(dtype=wp.float32)):
            v, i = wp.tid()
            p = pc[v, i]
            z = p[2]
            if z <= 1e-6:
                return
            f = fx[v]
            ux = f*p[0]/z+cx[v]
            uy = f*p[1]/z+cy[v]
            R = int(wp.ceil(1.1*f*r[i]/z+2.0))
            x0 = wp.max(int(wp.floor(ux))-R, 0)
            x1 = wp.min(int(wp.floor(ux))+R+1, Wd-1)
            y0 = wp.max(int(wp.floor(uy))-R, 0)
            y1 = wp.min(int(wp.floor(uy))+R+1, Hd-1)
            for y in range(y0, y1+1):
                for x in range(x0, x1+1):
                    d = wp.normalize(wp.vec3f((wp.float32(x)-cx[v])/f, (wp.float32(y)-cy[v])/f, 1.0))
                    t = wp.dot(p, d)
                    dist = wp.length(p-t*d)
                    a = wp.clamp(0.5+(r[i]-dist)*f/t, 0.0, 1.0)
                    if a > 0.0:
                        wp.atomic_max(img, v, y, x, a)
        self.k_sil = k_sil
        self.fx = wp.array(self.K[:, 0, 0].astype(np.float32), dtype=wp.float32, device='cuda')
        self.cx = wp.array(self.K[:, 0, 2].astype(np.float32), dtype=wp.float32, device='cuda')
        self.cy = wp.array(self.K[:, 1, 2].astype(np.float32), dtype=wp.float32, device='cuda')
        H, W = self.H, self.W
        self.ys = torch.arange(H, dtype=torch.float64, device='cuda')[:, None]
        self.xs = torch.arange(W, dtype=torch.float64, device='cuda')[None, :]

    def render(self, x, r):
        """x (n, 3) foam frame, r (n,): (views, H, W) coverage (torch float32, a view into the Warp image)"""
        torch, wp = self.torch, self.wp
        xw = x.to(torch.float64)-self.off
        pc = torch.einsum('vij,nj->vni', self.R, xw)+self.t[:, None, :]
        self.img.zero_()
        wp.launch(self.k_sil, (self.nv, x.shape[0]), inputs=[wp.from_torch(pc.to(torch.float32).contiguous(), dtype=wp.vec3f),
                                                              wp.from_torch(r.to(torch.float32).contiguous()), self.fx, self.cx, self.cy,
                                                              self.W, self.H], outputs=[self.img])
        return self.timg

    def features(self, x, r):
        """(views, 5): identify.moments of each view's coverage"""
        a = self.render(x, r).to(self.torch.float64)
        m0 = a.sum((1, 2)).clamp(min=1e-6)
        cx = (a*self.xs).sum((1, 2))/m0; cy = (a*self.ys).sum((1, 2))/m0
        sx = ((a*(self.xs-cx[:, None, None])**2).sum((1, 2))/m0).clamp(min=1e-12).sqrt()
        sy = ((a*(self.ys-cy[:, None, None])**2).sum((1, 2))/m0).clamp(min=1e-12).sqrt()
        return self.torch.stack((cx, cy, sx, sy, m0.sqrt()), 1)


def load_foam():
    F = DATA/'foam'
    cells = dict(np.load(F/'dem_exact_single'/'cells.npz')); g = dict(np.load(F/'dem_exact_single'/'geometry.npz'))
    site = np.load(F/'patch_site.npz')['patch_site']
    return cells, g, site


# ---------------------------------------------------------------- one forward
def forward(theta, frames, tol=1e-3, save_frames=False, verbose=True):
    """one run at theta = (lnE, nu, lnbeta) to frame `frames`: dict(f (frames + 1, views, 5), x, r per frame if asked,
    stats); raises RuntimeError at a step that does not converge"""
    import torch
    import warp as wp
    from experiments.pacnerf_dem_forward import build
    from free_surface.closure_drop import colouring, probe_blocks
    from free_surface.closure_warp import ClosureWarp, MidpointStep, TrustCG
    from free_surface.sphere_closure import ClosureTerms
    f64, dev = torch.float64, 'cuda'
    E, nu, beta = math.exp(theta[0]), float(theta[1]), math.exp(theta[2])
    t0 = time.time()
    cells, g, site_np = load_foam()
    m = build(g, E, nu, dev, 'noslip', .25, .3, f64, kinematics='exact', nodalvol=True, nonlinear=False, ogden_alpha=2.)
    st = ClosureTerms(m, g, cells['points'], cells['radii'])
    W = ClosureWarp(st)
    W.set_viscosity(beta*W.mu, beta*(W.lam+2*W.mu/3), DT)
    W.visc_weight = 2.; W.total_mode()
    boc = torch.as_tensor(np.asarray(g['body_of_cell']), device=dev).long(); site = torch.as_tensor(site_np, device=dev).long()
    cbody = boc[site]; Rb = torch.as_tensor(cells['radii'], dtype=f64, device=dev)[site]
    upt = tuple(1. if k == UP else 0. for k in range(3))
    W.set_ball(cbody, Rb, upt, C0, 0.)                                               # (frictionless; no self contact)
    W.set_cap_positions(); W.set_cap_floor(0.)
    W.prepare()
    n = W.n; X = st.X; mass = m.mass.to(f64); M3 = mass[:, None]
    up = torch.tensor(upt, dtype=f64, device=dev); fg = -9.8*M3*up; ref = float(fg.norm())
    Kbb = probe_blocks(W, X, colouring(st.i.cpu().numpy(), st.j.cpu().numpy(), n))
    W.visc_ref(X)
    W.tball[0], W.tball[1] = 2., .5
    CG = TrustCG(W, mass, DT, per=2, product_iters=1); ST = MidpointStep(W, CG, mass, Kbb, fg, DT, 0.)
    ST.smooth_pred = ST.carry_delta = True                                           # (dem_old1's: the same solution, faster)
    tl = tol*ref
    ST.set_state(X, torch.tensor(V0, dtype=f64, device=dev).expand_as(X).clone(), fg/M3)
    sil = Silhouette()
    r0 = torch.as_tensor(cells['radii'], dtype=f64, device=dev)
    fs, xs, rs = [sil.features(X, r0).cpu().numpy()], [X.cpu().numpy().astype(np.float32)], [r0.cpu().numpy().astype(np.float32)]
    tb = time.time()-t0; t1 = time.time(); prods = 0
    for s in range(1, frames*SPF+1):
        p_, its, g2 = ST.step(tl, lambda g2_: .5*tl)
        if not (math.isfinite(g2) and math.sqrt(g2) <= tl):
            raise RuntimeError('step %d (t %.4f s): the solve did not converge (|grad| / weight %s after %d Newton iterations)' % (
                s, s*DT, math.sqrt(g2)/ref if math.isfinite(g2) else g2, its))
        ST.end(); prods += p_
        if s % SPF:
            continue
        x = ST.tx.clone()
        W.grad(x); Fb = W.tF.clone()
        r = r0*torch.linalg.det(Fb).abs().clamp(min=1e-6)**(1/3)
        fs.append(sil.features(x, r).cpu().numpy())
        if save_frames:
            xs.append(x.cpu().numpy().astype(np.float32)); rs.append(r.cpu().numpy().astype(np.float32))
        if verbose:
            print('  frame %d (t %.2f s, %.0f s): com z %.4f m, balls touching %d, min J %.3f, products so far %d' % (
                s//SPF, s*DT, time.time()-t1, float((M3*x).sum(0)[UP]/mass.sum()), int((W.tbdelta > 0).sum()), math.exp(float(W.tlj.min())), prods), flush=True)
    out = dict(f=np.stack(fs), stats=dict(build=tb, seconds=time.time()-t1, products=prods))
    if save_frames:
        out.update(x=np.stack(xs), r=np.stack(rs))
    return out


def cmd_run(a):
    th = np.array(a.theta, float)
    print('run: E %.2f, nu %.4f, beta %.5f; frames 0..%d; tol %g' % (math.exp(th[0]), th[1], math.exp(th[2]), a.frames, a.tol), flush=True)
    try:
        o = forward(th, a.frames, a.tol, a.save_frames)
    except RuntimeError as e:
        print('FAILED: %s' % e, flush=True)
        np.savez(a.out, failed=str(e), theta=th)
        sys.exit(1)
    np.savez(a.out, theta=th, tol=a.tol, f=o['f'], **{k: o[k] for k in ('x', 'r') if k in o}, stats=json.dumps(o['stats']))
    print('done: build %.0f s, run %.0f s, products %d -> %s' % (o['stats']['build'], o['stats']['seconds'], o['stats']['products'], a.out), flush=True)


# ---------------------------------------------------------------- frame 0 against the masks
def cmd_check(a):
    import torch
    cells, g, _ = load_foam()
    sil = Silhouette()
    x = torch.as_tensor(cells['points'], dtype=torch.float64, device='cuda'); r = torch.as_tensor(cells['radii'], dtype=torch.float64, device='cuda')
    img = sil.render(x, r).cpu().numpy() > .5
    A = gt_masks()[:, 0] > .5
    fo = sil.features(x, r).cpu().numpy(); fg_ = gt_features()[0]
    meta, _, _ = cameras()
    for v, nm in enumerate(meta['camera_names']):
        iou = (img[v] & A[v]).sum()/(img[v] | A[v]).sum()
        print('%s: IoU %.4f; features ours - GT (px): centroid %+.2f %+.2f, widths %+.2f %+.2f, sqrt area %+.2f' % (nm, iou, *(fo[v]-fg_[v])), flush=True)
    sx, sy = np.abs(fo-fg_).mean(0)[:2]
    print('mean |centroid| difference %.2f / %.2f px' % (sx, sy), flush=True)


# ---------------------------------------------------------------- parallel runs
def run_batch(thetas, frames, tol, out, jobs, tag):
    """each theta in its own process (`run`), at most `jobs` at once: list of feature arrays (None for a failed run);
    tol one value or one per theta"""
    tols = list(tol) if isinstance(tol, (list, tuple)) else [tol]*len(thetas)
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(ROOT)+os.pathsep+os.environ.get('PYTHONPATH', ''))
    paths = [out/('%s_%d.npz' % (tag, k)) for k in range(len(thetas))]
    todo, live, t0 = list(range(len(thetas))), {}, time.time()
    while todo or live:
        while todo and len(live) < jobs:
            k = todo.pop(0); th = thetas[k]
            if paths[k].exists():
                paths[k].unlink()
            log = open(out/('%s_%d.log' % (tag, k)), 'w')
            cmd = [sys.executable, '-u', str(Path(__file__).resolve()), 'run', '--theta', *['%.17g' % v for v in th], '--frames', str(frames),
                   '--tol', '%g' % tols[k], '--out', str(paths[k])]
            live[k] = (subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT), env=env), log)
        time.sleep(2)
        for k in list(live):
            p, log = live[k]
            if p.poll() is not None:
                log.close(); del live[k]
    res = []
    for k, pth in enumerate(paths):
        z = np.load(pth) if pth.exists() else None
        res.append(None if z is None or 'failed' in z.files else z['f'])
    print('   %d runs (%s) in %.0f s, %d failed' % (len(thetas), tag, time.time()-t0, sum(r_ is None for r_ in res)), flush=True)
    return res


# ---------------------------------------------------------------- Levenberg-Marquardt
def cmd_lm(a):
    out = a.out; out.mkdir(parents=True, exist_ok=True)
    K, held = a.frames, list(a.held)
    fit = [v for v in range(12) if v not in held]
    lnb_min = math.log(a.beta_min)
    if a.obs == 'real':
        obs = gt_features()
    else:
        print('synthetic observations: one run at the truth %s' % a.truth, flush=True)
        obs = run_batch([np.array(a.truth)], K, a.tol, out/'runs', 1, 'truth')[0]
        if obs is None:
            raise SystemExit('the truth run failed')
    rel = (lambda f_: f_-f_[:1]) if a.relative else (lambda f_: f_)              # (each side's own frame 0 taken off: the
    obs = rel(obs)                                                                   # static shape / registration bias, 1-4.5 px)
    flat = lambda f_: (rel(f_)[1:K+1][:, fit]-obs[1:K+1][:, fit]).reshape(-1)
    held_c = lambda f_: float(((rel(f_)[1:K+1][:, held]-obs[1:K+1][:, held])**2).sum())
    path = out/'lm.json'
    hist = json.loads(path.read_text()) if a.resume and path.exists() else []
    th = np.array([e['theta'] for e in hist if e['accepted']][-1]) if hist else np.array(a.start, float)
    it0 = max([e['it'] for e in hist], default=0)

    def clamp(t_):
        t_ = t_.copy(); t_[1] = min(max(t_[1], NU_RANGE[0]), NU_RANGE[1]); t_[2] = max(t_[2], lnb_min)
        return t_

    def log(it, t_, c_, f_, acc, info):
        e = dict(it=it, theta=t_.tolist(), E=math.exp(t_[0]), nu=t_[1], beta=math.exp(t_[2]), cost=c_, accepted=acc,
                 rms=math.sqrt(c_/(K*len(fit)*5)) if math.isfinite(c_) else None, held=held_c(f_) if f_ is not None else None,
                 log10E_err=abs(t_[0]-math.log(GT['E']))/math.log(10), nu_err=abs(t_[1]-GT['nu']), **info)
        hist.append(e); path.write_text(json.dumps(hist, indent=1))
        print('it %d %s: E %.1f (log10 error %.3f), nu %.4f, beta %.4f; cost %.4e (RMS %.3f px; held view %s)' % (
            it, 'accepted' if acc else 'rejected', e['E'], e['log10E_err'], e['nu'], e['beta'], c_, e['rms'] if e['rms'] else float('nan'),
            '%.4e' % e['held'] if e['held'] is not None else '-'), flush=True)

    def jac(t_):
        """central differences, all 2p runs at once; a parameter at its bound: the pair moved inside"""
        ths, steps = [], []
        for j in range(3):
            tp, tm = t_.copy(), t_.copy(); tp[j] += a.eps; tm[j] -= a.eps
            if j == 1 and tm[1] < NU_RANGE[0]:
                tm[1], tp[1] = NU_RANGE[0], NU_RANGE[0]+2*a.eps
            if j == 1 and tp[1] > NU_RANGE[1]:
                tp[1], tm[1] = NU_RANGE[1], NU_RANGE[1]-2*a.eps
            if j == 2 and tm[2] < lnb_min:
                tm[2], tp[2] = lnb_min, lnb_min+2*a.eps
            ths += [tp, tm]; steps.append(tp[j]-tm[j])
        fs = run_batch(ths, K, a.tol, out/'runs', a.jobs, 'jac')
        if any(f_ is None for f_ in fs):
            raise SystemExit('a difference run failed: %s' % [k for k, f_ in enumerate(fs) if f_ is None])
        J_ = np.stack([(flat(fs[2*j])-flat(fs[2*j+1]))/steps[j] for j in range(3)], 1)
        np.savez(out/'jac.npz', theta=t_, J=J_)
        return J_

    # the start: the run at tol and at --ref-tol (the forward's own error), together
    f, f_ref = run_batch([th, th], K, [a.tol, a.ref_tol], out/'runs', a.jobs, 'start')
    if f is None or f_ref is None:
        raise SystemExit('the start run failed')
    dvec = flat(f)-flat(f_ref)
    print('the forward\'s error at the start (tol %g against %g): |d| %.3e px' % (a.tol, a.ref_tol, float(np.linalg.norm(dvec))), flush=True)
    r = flat(f); c, radius = float(r@r), math.inf
    if not hist:
        log(0, th, c, f, True, dict(radius=radius))
    J = jac(th)
    for it in range(it0+1, it0+a.iters+1):
        D = np.diag(J.T@J); dn = lambda s_: float(np.sqrt((D*s_*s_).sum()))

        def step(mu):
            if mu == 0:
                return np.linalg.lstsq(J, -r, rcond=None)[0]
            return np.linalg.lstsq(np.vstack((J, np.diag(np.sqrt(mu*D)))), np.concatenate((-r, np.zeros(3))), rcond=None)[0]
        res = float(np.linalg.norm(J@np.linalg.lstsq(J, dvec, rcond=None)[0]))
        while True:
            mu, s_ = 0., step(0.)
            if dn(s_) > radius:
                lo, hi = 0., 1.
                while dn(step(hi)) > radius:
                    lo, hi = hi, 2*hi
                for _ in range(60):
                    m_ = .5*(lo+hi)
                    lo, hi = (m_, hi) if dn(step(m_)) > radius else (lo, m_)
                mu, s_ = hi, step(hi)
            tt = clamp(th+s_); s_ = tt-th
            pred, slope = float(np.linalg.norm(J@s_)), float(2*r@(J@s_))
            if pred <= res:
                print('LM: the predicted change |J s| %.3e is not above the forward\'s error in J\'s range %.3e: stop' % (pred, res), flush=True)
                return finish(a, th)
            if slope >= 0:
                print('LM: the step (at a bound) does not descend: stop', flush=True)
                return finish(a, th)
            f_t = run_batch([tt], K, a.tol, out/'runs', 1, 'trial')[0]
            c_t = float(flat(f_t)@flat(f_t)) if f_t is not None else math.inf
            kap = c_t-c-slope
            al = -slope/(2*kap) if kap > 0 else math.inf
            if not math.isfinite(c_t):
                al = .25                                                             # (a failed trial: a quarter of the step)
            radius = al*dn(s_)
            info = dict(damping=mu, step=dn(s_), pred=pred, resolution=res, pred_cost=c+slope+float((J@s_)@(J@s_)), alpha=al, radius=radius)
            if c_t < c:
                th, f, r, c = tt, f_t, flat(f_t), c_t
                log(it, th, c, f, True, info)
                J = jac(th)
                break
            log(it, tt, c_t, f_t, False, info)
    return finish(a, th)


def finish(a, th):
    print('final theta: E %.2f, nu %.4f, beta %.5f; log10 E error %.3f, nu error %.4f' % (
        math.exp(th[0]), th[1], math.exp(th[2]), abs(th[0]-math.log(GT['E']))/math.log(10), abs(th[1]-GT['nu'])), flush=True)
    (a.out/'final_theta.json').write_text(json.dumps(dict(theta=th.tolist(), E=math.exp(th[0]), nu=th[1], beta=math.exp(th[2]))))
    return th


# ---------------------------------------------------------------- evaluation
def cmd_eval(a):
    a.out.mkdir(parents=True, exist_ok=True)
    th = np.array(a.theta, float)
    o = forward(th, NFRAMES-1, a.tol, save_frames=True, verbose=True)
    import torch
    sil = Silhouette()
    A = gt_masks() > .5
    iou = np.zeros((A.shape[0], NFRAMES))
    for k in range(NFRAMES):
        img = sil.render(torch.as_tensor(o['x'][k], device='cuda'), torch.as_tensor(o['r'][k], device='cuda')).cpu().numpy() > .5
        for v in range(A.shape[0]):
            iou[v, k] = (img[v] & A[v, k]).sum()/max((img[v] | A[v, k]).sum(), 1)
    held = list(a.held); fit = [v for v in range(A.shape[0]) if v not in held]
    rep = dict(theta=th.tolist(), E=math.exp(th[0]), nu=th[1], beta=math.exp(th[2]),
               log10E_err=abs(th[0]-math.log(GT['E']))/math.log(10), nu_err=abs(th[1]-GT['nu']),
               iou_fit_views_train=float(iou[fit][:, 1:16].mean()), iou_held_view_train=float(iou[held][:, 1:16].mean()),
               iou_future_all_views=float(iou[:, 16:].mean()), iou_future_held_view=float(iou[held][:, 16:].mean()),
               iou=iou.tolist(), stats=o['stats'])
    (a.out/'eval.json').write_text(json.dumps(rep, indent=1))
    np.savez(a.out/'eval_frames.npz', x=o['x'], r=o['r'], f=o['f'])
    print('eval: E %.2f (log10 error %.3f), nu %.4f (error %.4f), beta %.5f | IoU: fit views frames 1-15 %.4f, held view %.4f; future frames 16-23 '
          'all views %.4f, held view %.4f' % (rep['E'], rep['log10E_err'], rep['nu'], rep['nu_err'], rep['beta'], rep['iou_fit_views_train'],
                                             rep['iou_held_view_train'], rep['iou_future_all_views'], rep['iou_future_held_view']), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('check')
    q = sub.add_parser('run')
    q.add_argument('--theta', type=float, nargs=3, required=True)
    q.add_argument('--frames', type=int, default=15)
    q.add_argument('--tol', type=float, default=1e-3)
    q.add_argument('--out', type=Path, required=True)
    q.add_argument('--save-frames', action='store_true')
    q = sub.add_parser('lm')
    q.add_argument('--out', type=Path, required=True)
    q.add_argument('--obs', choices=('real', 'synth'), default='real')
    q.add_argument('--truth', type=float, nargs=3, default=[math.log(GT['E']), GT['nu'], math.log(.05)], help='--obs synth: theta of the observations')
    q.add_argument('--start', type=float, nargs=3, default=[math.log(GT['E']*3), .3, math.log(.1)])
    q.add_argument('--frames', type=int, default=15)
    q.add_argument('--held', type=int, nargs='*', default=[5])
    q.add_argument('--tol', type=float, default=1e-3)
    q.add_argument('--ref-tol', type=float, default=2e-3)
    q.add_argument('--eps', type=float, default=1e-2, help='central differences\' step in theta')
    q.add_argument('--beta-min', type=float, default=.05)
    q.add_argument('--iters', type=int, default=15)
    q.add_argument('--jobs', type=int, default=7, help='runs at once (an iteration has 6 difference runs)')
    q.add_argument('--resume', action='store_true')
    q.add_argument('--relative', type=int, default=1, help='1: features minus their own frame 0 (both sides)')
    q = sub.add_parser('eval')
    q.add_argument('--theta', type=float, nargs=3, required=True)
    q.add_argument('--out', type=Path, required=True)
    q.add_argument('--held', type=int, nargs='*', default=[5])
    q.add_argument('--tol', type=float, default=1e-3)
    a = p.parse_args()
    dict(check=cmd_check, run=cmd_run, lm=cmd_lm, eval=cmd_eval)[a.cmd](a)


if __name__ == '__main__':
    main()
