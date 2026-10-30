"""PAC-NeRF elastic scene 0 (the jack) with the skinless model (2026-10-26): identify.py's scene (the foam's rest state
in the model.pt frame, a uniform start velocity v0, gravity -9.8 y, the ground y = --ground, frames every 1/24 s; per
frame and view the rendered silhouette's centroid, second-moment widths and sqrt(area) in pixels, Power Foam's renderer
with the PowerSim update x, R = polar(F_B), r0 det(F_B)^(1/3)), run with closure_warp's ClosureWarp, Hertz-Mindlin balls
on the ground and MidpointStep (hertz_drop.py's step) instead of the skin model. theta = (lnE, nu, lnbeta, v0x, v0y,
v0z, lnmu) is set in place (ClosureWarp.set_material, MidpointStep.update_material); E in Pa as identify's.
  --mode lm: identify.lm's LM from --start on the real masks (the fit views; the held view reported), the Jacobian by
           central differences of whole runs (--eps in theta: 14 runs), computed only at accepted points; the stop at
           the forward's own error (f at --tol against --ref-tol, identify's); a failed trial is a rejection
  --mode forward: one run at --theta against the real masks' features (identify.py's, frames 1..--frames); --show:
           also a picture per view in --show (renderer camera order): rows the real photo with the rendered
           silhouette's outline (red), and the render over the real background (r_<view>_-1.png); columns frames 0..
  PYTHONPATH=<main repo>:<worktree> python free_surface/closure_scene.py"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import warp as wp

from free_surface import identify as idf                                       # (its import puts the renderer's code on the path)
from experiments.pacnerf_dem_forward import build                               # noqa: E402
from free_surface.closure_drop import colouring, probe_blocks                   # noqa: E402
from free_surface.closure_warp import ClosureWarp, MidpointStep, TrustCG        # noqa: E402
from free_surface.sphere_closure import ClosureTerms                            # noqa: E402

f64, dev = torch.float64, 'cuda'
rd = idf.rd
PARAMS = idf.PARAMS
BOUNDS = dict(nu=(.005, .495))                                                   # (the difference runs' range: lam finite)


class Scene:
    """the model built at --theta's material; run(theta) -> features (frames, views, 5), stats"""
    def __init__(self, args):
        self.args = a = args
        self.wstream = wp.get_stream()                                             # Warp's own, before the renderer
        cells = np.load(a.out/'dem_exact_single'/'cells.npz')
        g = dict(np.load(a.out/'dem_exact_single'/'geometry.npz'))
        bsk = dict(np.load(a.out/'boundary_skin.npz'))
        th = a.theta
        t0 = time.time()
        m = build(g, math.exp(th[0]), th[1], dev, 'noslip', .25, .3, f64, kinematics='exact', nodalvol=True, nonlinear=False, ogden_alpha=2.)
        st = ClosureTerms(m, g, cells['points'], cells['radii'], bsk)
        W = ClosureWarp(st)
        beta = math.exp(th[2])
        W.set_viscosity(beta*W.mu, beta*(W.lam+2*W.mu/3), a.dt)
        W.visc_weight = 2.; W.total_mode()
        boc = torch.as_tensor(np.asarray(g['body_of_cell']), device=dev).long()
        site = torch.as_tensor(bsk['patch_site'], device=dev).long()
        cbody = boc[site]; Rb = torch.as_tensor(cells['radii'], dtype=f64, device=dev)[site]
        W.set_ball(cbody, Rb, (0., 1., 0.), a.ground, math.exp(th[6]))
        W.prepare()
        self.W, self.X, self.cbody, self.Rb = W, st.X, cbody, Rb
        self.mass = m.mass.to(f64)
        up = torch.tensor([0., 1., 0.], dtype=f64, device=dev)
        self.fg = -9.8*self.mass[:, None]*up
        self.ref = float(self.fg.norm())
        self.color = colouring(st.i.cpu().numpy(), st.j.cpu().numpy(), W.n)
        Kbb = probe_blocks(W, self.X, self.color)
        self.probed = (th[0], th[1])
        W.tball[0], W.tball[1] = 2., .5                                            # (the step's end: 2 z - x_n, weight 1/2)
        self.CG = TrustCG(W, self.mass, a.dt, per=a.cg_check, product_iters=1)
        self.ST = MidpointStep(W, self.CG, self.mass, Kbb, self.fg, a.dt, math.exp(th[6]))
        tb = time.time()-t0
        self.render = getattr(a, 'render', True)                                  # (False: no renderer, no features; e.g.
        self.nfit = len(a.fit)                                                     # another foam than rd.RUN's)
        if self.render:
            model, data = rd.load(rd.RUN)                                          # (identify.Problem's)
            src = torch.as_tensor(cells['source_index'], device=model.device)
            with torch.no_grad():
                for name in rd.NAMES:
                    setattr(model, name, torch.nn.Parameter(getattr(model, name).data[src].clone()))
                model.rebuild_adjacency()
            self.model, self.up = model, rd.Update(model)
            assert float((self.up.p0.double()-self.X).abs().max()) < 1e-6, 'the DEM cells are not the foam\'s sites'
            self.cams = [idf.ss_camera(data.cameras[v], a.ss) for v in a.fit+a.held]
            self._base_cams = {v: data.cameras[v] for v in a.fit+a.held}
        self.steps = int(round(a.frames/24/a.dt)); self.every = int(round(1/(24*a.dt)))
        assert abs(self.every*a.dt*24-1) < 1e-9
        wp.set_stream(self.wstream)
        print('scene: %d bodies, %d surface balls, ground y %.3f; model, blocks and graphs %.0f s; %d steps, %d frames, %d + %d views at %d x %d rays' % (
            W.n, len(cbody), a.ground, tb, self.steps, a.frames, self.nfit, len(a.held), a.ss, a.ss), flush=True)

    @torch.no_grad()
    def features(self, x, F, radii=None):
        self.up.apply(x.float(), F.float(), radii)
        f = torch.stack([idf.moments(rd.render(self.model, c)[1].reshape(c.height, c.width).double(), self.args.ss) for c in self.cams])
        wp.set_stream(self.wstream)                                                # (the rasterizer sets its own; identify.features)
        return f

    def run(self, th, tol=None, verbose=True):
        a, W, ST = self.args, self.W, self.ST
        tol = a.tol if tol is None else tol
        W.set_material(E=math.exp(th[0]), nu=th[1], beta=math.exp(th[2]))
        W.cold(); W.tblamt.zero_(); W.tbbound.zero_()
        if (th[0], th[1]) != self.probed:                                         # the blocks probed again at the new E, nu
            ST.K0, ST.mu0 = probe_blocks(W, self.X, self.color), W.mu              # (10-26: scaled by mu only, nu 0.4 -> 0.49
            self.probed = (th[0], th[1])                                           # cost 2.7 x the products; a probe ~1.5 s)
        ST.update_material(math.exp(th[6]))
        W.visc_ref(self.X)
        v0 = torch.tensor(th[3:6], dtype=f64, device=dev).expand_as(self.X).clone()
        ST.set_state(self.X, v0, self.fg/self.mass[:, None])
        fs, prods, newton, tr = [], 0, 0, 0.
        self.states = [(self.X.clone(), torch.eye(3, dtype=f64, device=dev).expand(len(self.X), 3, 3).clone())]
        tl = tol*self.ref
        torch.cuda.synchronize(); t0 = time.time()
        for s in range(1, self.steps+1):
            p_, its, g2 = ST.step(tl, lambda g2_: a.cg_abs*tl)
            if not (math.isfinite(g2) and math.sqrt(g2) <= tl):
                raise RuntimeError('step %d: the solve did not converge (|grad| / weight %s after %d Newton iterations)' % (s, math.sqrt(g2)/self.ref if math.isfinite(g2) else g2, its))
            ST.end()
            prods += p_; newton = max(newton, its)
            if s % self.every:
                continue
            x = ST.tx.clone()
            W.grad(x); F = W.tF.clone()
            self.states.append((x, F))
            if self.render:
                torch.cuda.synchronize(); t1 = time.time()
                fs.append(self.features(x, F))
                torch.cuda.synchronize(); tr += time.time()-t1
            if verbose:
                print('  frame %d (t %.3f s): com y %.4f m, balls touching %d, min J %.3f' % (s//self.every, s*a.dt, float((self.mass[:, None]*x).sum(0)[1]/self.mass.sum()),
                                                                                          int((W.tbdelta > 0).sum()), math.exp(float(W.tlj.min()))), flush=True)
        return (torch.stack(fs) if fs else None), dict(seconds=time.time()-t0-tr, render=tr, products=prods, newton_max=newton)

    @torch.no_grad()
    def show(self, cams, photos, crop, path):
        """the last run's frames (self.states) against the real photos; cams: renderer camera indices, photos(c, k) ->
        the real image (H, W, 3) in [0, 1] (k -1: the background), crop[c] = (x0, y0, size). Per camera three rows: the
        photo with the rendered silhouettes' outlines (red: the radii r0 det(F_B)^(1/3), PowerSim's; blue: the rest
        radii r0), the render with the scaled radii over the real background, the same with the rest radii"""
        from PIL import Image, ImageDraw
        nk = len(self.states)
        rules = (('scaled', None, (1., 0., 0.)), ('rest', self.up.r0, (0., .3, 1.)))
        tiles = []
        for c in cams:
            bg = torch.as_tensor(photos(c, -1), device=dev, dtype=torch.float32)
            x0, y0, sz = crop[c]
            rows = [[], [], []]
            for k, (x, F) in enumerate(self.states):
                out = torch.as_tensor(photos(c, k), device=dev, dtype=torch.float32)
                for j, (_, rr, col) in enumerate(rules):
                    self.up.apply(x.float(), F.float(), rr)
                    c1 = self._base_cams[c]
                    rgb, al = rd.render(self.model, c1)
                    H, Wd = c1.height, c1.width
                    rgb, al = rgb.reshape(H, Wd, 3).float(), al.reshape(H, Wd, 1).float()
                    rows[1+j].append((rgb+(1-al)*bg).clamp(0, 1)[y0:y0+sz, x0:x0+sz].cpu().numpy())
                    m = (al[..., 0] > .5)
                    er = m.clone()
                    er[1:-1, 1:-1] = m[1:-1, 1:-1] & m[:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, :-2] & m[1:-1, 2:]
                    out[m & ~er] = torch.tensor(col, device=dev)
                rows[0].append(out[y0:y0+sz, x0:x0+sz].cpu().numpy())
            tiles.append((c, rows))
            wp.set_stream(self.wstream)
        sz = max(t_[1][0][0].shape[0] for t_ in tiles)
        head, lab, nr = 16, 70, 3
        img = np.ones((head+len(tiles)*nr*sz, lab+nk*sz, 3))
        for i, (c, rows) in enumerate(tiles):
            for r_ in range(nr):
                for k in range(nk):
                    t_ = rows[r_][k]
                    y = head+(nr*i+r_)*sz
                    img[y:y+t_.shape[0], lab+k*sz:lab+k*sz+t_.shape[1]] = t_
        im = Image.fromarray((img*255).astype(np.uint8))
        d = ImageDraw.Draw(im)
        for k in range(nk):
            d.text((lab+k*sz+3, 2), 'frame %d (%.3f s)' % (k, k/24), fill=(0, 0, 0))
        for i, (c, _) in enumerate(tiles):
            y = head+nr*i*sz
            d.text((2, y+4), 'camera %d\nphoto;\nred: r0 J^1/3\nblue: r0' % c, fill=(0, 0, 0))
            d.text((2, y+sz+4), 'camera %d\nours, radii\nr0 J^1/3' % c, fill=(0, 0, 0))
            d.text((2, y+2*sz+4), 'camera %d\nours, radii\nr0 (rest)' % c, fill=(0, 0, 0))
        im.save(path)
        print('wrote %s' % path, flush=True)


def lm(S, th, f_obs, args):
    """identify.lm (the radius from the measured cost along each step, the stop at the forward's own error) with J by
    central differences, built only at an accepted point; writes <out>/identify/<tag>.json"""
    nf = S.nfit
    path = args.out/'identify'/('%s.json' % args.tag)
    hist = json.loads(path.read_text()) if args.resume else []
    if args.resume:
        th = np.array([e['theta'] for e in hist if e['accepted']][-1])
        print('resume from %s: its last accepted theta %s' % (path, np.round(th, 5).tolist()), flush=True)
    it0 = max([e['it'] for e in hist], default=0)

    def run_f(t_, tol=None):
        try:
            f_, st_ = S.run(t_, tol=tol, verbose=False)
            return f_.cpu().numpy(), st_
        except RuntimeError as e:
            print('   the run failed: %s' % e, flush=True)
            return None, None
    flat = lambda f_: (f_[:, :nf]-f_obs[:, :nf]).reshape(-1)
    held = lambda f_: float(((f_[:, nf:]-f_obs[:, nf:])**2).sum())
    cost = lambda r_: float(r_@r_)

    jpath = args.out/'identify'/('%s_jac.npz' % args.tag)                         # (the last Jacobian, reused on a restart)

    def jac(t_):
        if jpath.exists():
            zj = np.load(jpath)
            if np.array_equal(zj['theta'], t_) and float(zj['eps']) == args.eps:
                print('   Jacobian read from %s' % jpath, flush=True)
                return zj['J']
        t0 = time.time(); cols = []
        for j in range(len(t_)):
            e = args.eps
            lo, hi = BOUNDS.get(PARAMS[j], (-math.inf, math.inf))
            for _ in range(3):                                                     # (a failed run: the step halved)
                tp, tm = t_.copy(), t_.copy()
                tp[j], tm[j] = min(t_[j]+e, hi), min(t_[j]+e, hi)-2*e              # (both moved inside a bound: the
                if tm[j] < lo:                                                     # derivative a step from it)
                    tm[j], tp[j] = lo, lo+2*e
                fp, _ = run_f(tp); fm, _ = run_f(tm)
                if fp is not None and fm is not None:
                    break
                e /= 2
            else:
                raise RuntimeError('the difference runs of %s failed' % PARAMS[j])
            cols.append((flat(fp)-flat(fm))/(tp[j]-tm[j]))
        print('   Jacobian: 14 runs, %.0f s' % (time.time()-t0), flush=True)
        J_ = np.stack(cols, 1)
        np.savez(jpath, theta=t_, eps=args.eps, J=J_)
        return J_

    def log(it, t_, c_, f_, st_, acc, info):
        it += it0
        e = dict(it=it, theta=t_.tolist(), E=math.exp(t_[0]), nu=t_[1], beta=math.exp(t_[2]), v0=t_[3:6].tolist(), mu=math.exp(t_[6]), cost=c_,
                 held=held(f_) if f_ is not None else None, rms_fit=math.sqrt(c_/(f_obs[:, :nf].size)) if math.isfinite(c_) else None,
                 accepted=acc, seconds=st_['seconds'] if st_ else None, products=st_['products'] if st_ else None, **info)
        hist.append(e)
        path.write_text(json.dumps(hist, indent=1))
        print('it %d %s: E %.1f, nu %.4f, beta %.5f, v0 %s, mu %.4f; cost %.4e (RMS %.3f px; held view %s)' % (
            it, 'accepted' if acc else 'rejected', e['E'], e['nu'], e['beta'], np.round(e['v0'], 4).tolist(), e['mu'], c_,
            e['rms_fit'] if e['rms_fit'] is not None else float('nan'), '%.4e' % e['held'] if e['held'] is not None else '-'), flush=True)
        if 'damping' in info:
            print('   damping %.2e, |s| %.3e, |J s| %.3e (forward error in J\'s range %.3e), cost predicted %.4e, a* %.3g -> radius %.3e' % (
                info['damping'], info['step'], info['pred'], info['resolution'], info['pred_cost'], info['alpha'], info['radius']), flush=True)

    f, st = run_f(th)
    f_ref, _ = run_f(th, tol=args.ref_tol)
    dvec = flat(f)-flat(f_ref)
    print('the forward\'s error at the start (tol %.0e against %.0e): |d| %.3e px over the fit views' % (args.tol, args.ref_tol, float(np.linalg.norm(dvec))), flush=True)
    r = flat(f); c, radius = cost(r), math.inf
    log(0, th, c, f, st, True, dict(radius=radius))
    J = jac(th)
    for it in range(1, args.iters+1):
        D = np.diag(J.T@J)
        dn = lambda s_: float(np.sqrt((D*s_*s_).sum()))

        def step(mu):
            if mu == 0:
                return np.linalg.lstsq(J, -r, rcond=None)[0]
            return np.linalg.lstsq(np.vstack((J, np.diag(np.sqrt(mu*D)))), np.concatenate((-r, np.zeros(len(D)))), rcond=None)[0]
        res = float(np.linalg.norm(J@np.linalg.lstsq(J, dvec, rcond=None)[0]))
        while True:
            mu, s_ = 0., step(0.)
            if dn(s_) > radius:
                lo, hi = 0., 1.
                while dn(step(hi)) > radius:
                    lo, hi = hi, 2*hi
                while True:
                    m = .5*(lo+hi)
                    if not lo < m < hi:
                        break
                    lo, hi = (m, hi) if dn(step(m)) > radius else (lo, m)
                mu, s_ = hi, step(hi)
            tt = th+s_
            tt[1] = min(max(tt[1], .01), .49)
            s_ = tt-th
            pred, slope = float(np.linalg.norm(J@s_)), float(2*r@(J@s_))
            if pred <= res:
                print('LM: the step\'s predicted change |J s| %.3e is not above the forward\'s error in J\'s range %.3e, stop' % (pred, res), flush=True)
                return th, hist
            if slope >= 0:
                print('LM: the step (nu at its bound) does not descend, stop', flush=True)
                return th, hist
            f_t, st_t = run_f(tt)
            c_t = cost(flat(f_t)) if f_t is not None else math.inf
            kap = c_t-c-slope
            a = -slope/(2*kap) if kap > 0 else math.inf
            if not math.isfinite(c_t):
                a = .25                                                            # (a failed trial: a quarter of the step)
            radius = a*dn(s_)
            info = dict(damping=mu, step=dn(s_), pred=pred, resolution=res, pred_cost=c+slope+float((J@s_)@(J@s_)), alpha=a, radius=radius)
            if c_t < c:
                th, f, r, c = tt, f_t, flat(f_t), c_t
                log(it, th, c, f, st_t, True, info)
                J = jac(th)
                break
            log(it, tt, c_t, f_t, st_t, False, info)
    return th, hist


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument('--mode', choices=('forward', 'lm'), default='forward')
    a_.add_argument('--out', type=Path, default=Path('output/round01_pruned_exact'))
    a_.add_argument('--real', type=Path, default=Path('output/round01_pruned_exact/identify/real_features.npz'),
                    help='the masks\' features (identify.py\'s real_check; frames 0.., the renderer\'s camera order)')
    a_.add_argument('--theta', type=float, nargs=7, default=[math.log(23008.38), .3008, math.log(.005), -.408, -.5, -.066, math.log(.5)],
                    help='lnE (Pa), nu, ln beta (s), v0 (m/s), ln mu; default identify.py\'s --truth')
    a_.add_argument('--ground', type=float, default=-.05)
    a_.add_argument('--frames', type=int, default=13)
    a_.add_argument('--fit', type=int, nargs='+', default=list(range(10)))
    a_.add_argument('--held', type=int, nargs='+', default=[10])
    a_.add_argument('--ss', type=int, default=2, help='rays per pixel k x k (identify.ss_camera)')
    a_.add_argument('--dt', type=float, default=1/1200)
    a_.add_argument('--tol', type=float, default=1e-3)
    a_.add_argument('--cg-abs', type=float, default=.5)
    a_.add_argument('--cg-check', type=int, default=2)
    a_.add_argument('--tag', default='closure_scene')
    a_.add_argument('--start', type=float, nargs=7, default=[math.log(23008.38*.5), .4, math.log(.02), -.204, -.25, -.033, math.log(.3)],
                    help='--mode lm: the start (identify.py\'s --start)')
    a_.add_argument('--eps', type=float, default=1e-2, help='--mode lm: the central differences\' step in theta (identify\'s --probe-eps)')
    a_.add_argument('--ref-tol', type=float, default=2e-3, help='--mode lm: the forward\'s own error from a second run at this tol (identify\'s)')
    a_.add_argument('--iters', type=int, default=15)
    a_.add_argument('--resume', action='store_true', help='--mode lm: from the last accepted theta of <tag>.json (the log continued; the '
                    'Jacobian from <tag>_jac.npz when it is that theta\'s)')
    a_.add_argument('--show', type=int, nargs='*', default=None, help='cameras (renderer order) for the picture; empty: 0 6 10')
    a_.add_argument('--photos', type=Path, default=Path('/home/fangsuo/py/dynamic-powerfoam/data/pacnerf/data/elastic/0/data'), help='r_<view>_<frame>.png')
    args = a_.parse_args()
    S = Scene(args)
    z = np.load(args.real)
    assert list(z['cameras']) == list(args.fit+args.held) and int(z['frames'][0]) == 0 and len(z['frames']) > args.frames
    f_obs = z['f'][1:args.frames+1]
    if args.mode == 'lm':
        th, hist = lm(S, np.array(args.start, dtype=float), f_obs, args)
        print('LM end: E %.1f, nu %.4f, beta %.5f, v0 %s, mu %.4f (scene truth E 23008.4, nu 0.3008); %d entries in %s' % (
            math.exp(th[0]), th[1], math.exp(th[2]), np.round(th[3:6], 4).tolist(), math.exp(th[6]), len(hist), args.out/'identify'/('%s.json' % args.tag)), flush=True)
        return
    th = np.array(args.theta, dtype=float)
    f, st = S.run(th)
    f = f.cpu().numpy()
    d = f-f_obs
    nf = S.nfit
    print('theta: E %.1f, nu %.4f, beta %.4f, v0 %s, mu %.3f; forward %.1f s (%d gradient evaluations, Newton per step max %d), rendering %.1f s' % (
        math.exp(th[0]), th[1], math.exp(th[2]), np.round(th[3:6], 4).tolist(), math.exp(th[6]), st['seconds'], st['products'], st['newton_max'], st['render']), flush=True)
    rms = lambda v: float(np.sqrt((v**2).mean()))
    print('rendered - observed (px), RMS over the fit views; per frame: centroid x, y | widths x, y | sqrt(area) || held view all five', flush=True)
    for k in range(len(d)):
        print('  frame %2d: %6.2f %6.2f | %6.2f %6.2f | %6.2f || %6.2f' % (k+1, rms(d[k, :nf, 0]), rms(d[k, :nf, 1]), rms(d[k, :nf, 2]), rms(d[k, :nf, 3]),
                                                                      rms(d[k, :nf, 4]), rms(d[k, nf:])), flush=True)
    print('cost (fit views, sum of squares) %.4e; RMS all fit features %.3f px, held view %.3f px' % (float((d[:, :nf]**2).sum()), rms(d[:, :nf]), rms(d[:, nf:])), flush=True)
    np.savez(args.out/'identify'/('%s.npz' % args.tag), f=f, f_obs=f_obs, theta=th, seconds=st['seconds'], render=st['render'], products=st['products'])
    if args.show is not None:
        from PIL import Image
        cams = args.show or [0, 6, 10]
        views = list(z['views'])
        photos = lambda c, k: np.asarray(Image.open(args.photos/('r_%d_%d.png' % (views[c], k))).convert('RGB'), dtype=np.float32)/255
        fo = z['f'][:args.frames+1]                                                # (frames 0.., views, 5): centroid, widths
        crop = {}
        for c in cams:
            cx, cy, wx, wy = fo[:, c, 0], fo[:, c, 1], fo[:, c, 2].max(), fo[:, c, 3].max()
            sz = int(math.ceil(max(cx.max()-cx.min()+6*wx, cy.max()-cy.min()+6*wy)))
            x0 = int(round(.5*(cx.max()+cx.min())-sz/2)); y0 = int(round(.5*(cy.max()+cy.min())-sz/2))
            crop[c] = (max(x0, 0), max(y0, 0), sz)
        S.show(cams, photos, crop, args.out/'identify'/('%s_show.png' % args.tag))
        fr = np.stack([S.features(x_, F_, S.up.r0).cpu().numpy() for x_, F_ in S.states[1:]])
        print('signed mean over the fit views, rendered - observed (px): radii r0 det(F)^(1/3) | rest radii r0', flush=True)
        for k in range(len(d)):
            a_, b_ = d[k, :nf].mean(0), (fr[k]-f_obs[k])[:nf].mean(0)
            print('  frame %2d: cy %+6.2f wx %+6.2f wy %+6.2f sqrtA %+6.2f | cy %+6.2f wx %+6.2f wy %+6.2f sqrtA %+6.2f' % ((k+1,)+tuple(a_[1:])+tuple(b_[1:])), flush=True)
        dr = fr-f_obs
        print('RMS all fit features: scaled radii %.3f px, rest radii %.3f px; held view %.3f / %.3f px' % (rms(d[:, :nf]), rms(dr[:, :nf]), rms(d[:, nf:]), rms(dr[:, nf:])), flush=True)


if __name__ == '__main__':
    main()
