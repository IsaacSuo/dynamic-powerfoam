"""LM identification of theta = (lnE, nu, lnbeta, v0x, v0y, v0z) from silhouette features, the Jacobian from the tangent
sensitivities (tangent.py) (2026-10-25). Forward: the joint midpoint rule from the foam's rest state (the DEM cells are
the foam's material sites, in the model.pt frame), uniform start velocity v0, gravity -9.8 y, the frictionless ground
plane y = --ground (PAC-NeRF scene 0: -0.05 in the scene frame), frames every 1/24 s. Per frame and view the rendered
silhouette's features (Power Foam's renderer, the PowerSim update of the 7478 cells: x, R = polar(F_B), r0
det(F_B)^(1/3)): centroid, second-moment widths, sqrt(area), in pixels. Jacobian: per parameter the state tangents at
the frame (bodies x, the integer-time skin y; F_B along them by a one-sided difference) and the features' central
difference along them (max displacement --delta), k x k rays per pixel (--ss, ss_camera). LM on r = f(theta) - f_obs (lm(): the radius from
the measured cost along each step, the stop at the forward's own error), every trial run with its tangents (the next
iteration's J). The material changes in place (SkinWarp.set_material); the preconditioner's
blocks are scaled with E and the viscous factor (the K probe at the defaults: nu's change not in them).
  --mode synth: observations from the model at --truth, then LM from --start; views --fit, held out --held.
  --mode probe: the end-to-end Jacobian (tangents + render differences) at --start against central differences of
                whole runs (--probe params, relative eps --probe-eps) over --frames.
Writes <out>/identify/<tag>.json (per iteration theta, cost, mu, times) and the observations' npz."""
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import warp as wp

from free_surface import render_drop as rd                                     # noqa: E402  (thin-surface's renderer: its powerfoam,
sys.path.remove(rd.THIN); sys.path.append(rd.THIN)                             # configs, data_loader now loaded; the rest, e.g.
                                                                                # experiments, from the usual path)

from free_surface.joint_step import MidpointIntegrator                          # noqa: E402
from free_surface.skin_contact_check import f64, dev                            # noqa: E402
from free_surface.skin_dynamics_check import build_model, parser                # noqa: E402
from free_surface.tangent import Tangent                                        # noqa: E402

PARAMS = ('lnE', 'nu', 'lnbeta', 'v0x', 'v0y', 'v0z', 'lnmu')                    # (lnmu: the ground's friction, 10-25)


def moments(a, k=1):
    """centroid, second-moment widths, sqrt(area) of an alpha map in the original pixels (k: the map has k x k samples per
    pixel, ss_camera's; sample centres at (J + 1/2) / k - 1/2)"""
    H, W = a.shape
    ys = ((torch.arange(H, device=a.device, dtype=a.dtype)+.5)/k-.5)[:, None]
    xs = ((torch.arange(W, device=a.device, dtype=a.dtype)+.5)/k-.5)[None, :]
    m0 = a.sum().clamp(min=1e-6)
    cx = (a*xs).sum()/m0; cy = (a*ys).sum()/m0
    sx = ((a*(xs-cx)**2).sum()/m0).clamp(min=1e-12).sqrt(); sy = ((a*(ys-cy)**2).sum()/m0).clamp(min=1e-12).sqrt()
    return torch.stack((cx, cy, sx, sy, (m0/k**2).sqrt()))


def ss_camera(c, k):
    """k x k rays per pixel: k W x k H pixels centred on the original pixels' sub-pixel centres (the camera's x =
    2 j / (W - 1) - 1 spans the pixel centres; right, up scaled by (k W - 1) / (k (W - 1)), (k H - 1) / (k (H - 1))).
    10-25 (rigid shifts of the rest foam, feature_smooth_ss): one ray per pixel leaves sub-pixel roughness (0.012 px per
    0.05 mm, a 0.35 px dip at the trained pose: the fringe fitted at the pixel centres); k 2: no dip, the features'
    central difference over +-2 mm within 1.1-1.8 % of the slope; k 4: 0.3-0.4 %"""
    if k == 1:
        return c
    W, H = c.width, c.height
    return replace(c, right=c.right*((k*W-1)/(k*(W-1))), up=c.up*((k*H-1)/(k*(H-1))), width=k*W, height=k*H, ray_maps=None)


class Problem:
    def __init__(self, args):
        self.args = args
        b = build_model(args, contact=dict(normal=(0., 1., 0.), c0=args.ground, points=False, shift=True))
        self.wstream = wp.get_stream()                                             # Warp's own, before the renderer (features)
        self.b, self.st, self.W, self.S, self.mass = b, b['st'], b['W'], b['S'], b['mass']
        self.E0, self.beta0 = args.E, args.beta
        self.blocks0, self.kb0, self.vb0 = b['blocks'], b['kblocks'], b['vblocks']
        up = torch.tensor([0., 1., 0.], dtype=f64, device=dev)
        self.fg = -9.8*self.mass[:, None]*up
        self.ref = float(self.fg.norm())
        model, data = rd.load(rd.RUN)
        src = torch.as_tensor(np.load(Path(args.out)/'dem_exact_single'/'cells.npz')['source_index'], device=model.device)
        with torch.no_grad():
            for name in rd.NAMES:
                setattr(model, name, torch.nn.Parameter(getattr(model, name).data[src].clone()))
            model.rebuild_adjacency()
        self.model, self.up = model, rd.Update(model)
        assert float((self.up.p0.double()-self.st.X).abs().max()) < 1e-6, 'the DEM cells are not the foam\'s sites'
        self.cams = {k: [ss_camera(data.cameras[v], k) for v in args.fit+args.held] for k in args.ss}
        self.nfit = len(args.fit)
        self.steps = int(round(args.frames/24/args.dt)); self.every = int(round(1/(24*args.dt)))
        assert abs(self.every*args.dt*24-1) < 1e-9
        self.copies = [b['make_copy']() for _ in range(args.copies)]               # the tangents' CGs at once (tangent.Tangent)
        free, total = torch.cuda.mem_get_info()
        print('%d product copies; GPU memory in use %.2f of %.2f GB' % (len(self.copies), (total-free)/1e9, total/1e9), flush=True)

    def set_material(self, th):
        E, nu, beta = math.exp(th[0]), th[1], math.exp(th[2])
        W, b, S, dt = self.W, self.b, self.S, self.args.dt
        W.tfpar[0] = math.exp(th[6])                                               # the friction (its glue: set_material moves k)
        W.set_material(E=E, nu=nu, beta=beta)
        vis = lambda be: 1+2*be/dt                                                # (build_model: midpoint's viscous factor)
        Bk = self.blocks0*(E/self.E0)*(vis(beta)/vis(self.beta0))
        b['prec'] = S.preconditioner(Bk, S.bodies)
        S.contact_setup(b['prec'], 100.)
        b['kblocks'] = self.kb0*(E/self.E0)
        b['vblocks'] = self.vb0*(E*beta/(self.E0*self.beta0)) if self.vb0 is not None else None

    @torch.no_grad()
    def features(self, x, F):
        """per supersampling k of --ss: (views, 5)"""
        self.up.apply(x.float(), F.float())
        out = {}
        for k, cams in self.cams.items():
            out[k] = torch.stack([moments(rd.render(self.model, c)[1].reshape(c.height, c.width).double(), k) for c in cams])
        # 10-25: the rasterizer sets wp.stream_from_torch(...) as Warp's stream on each call, and the previous wrapper's
        # __del__ unregisters that handle; a later graph capture (a new JointCG at a contact) then fails "unknown stream"
        wp.set_stream(self.wstream)
        return out

    def run(self, th, tangents=True, tol=None):
        """per supersampling k: features (frames, views, 5) and, with tangents, their Jacobian (frames, views, 5, 6); stats"""
        a, W = self.args, self.W
        tol = a.tol if tol is None else tol
        t0 = time.time()
        self.set_material(th)
        v0 = torch.tensor(th[3:6], dtype=f64, device=dev).expand_as(self.st.X).clone()
        W.tlam.zero_(); W.tact.zero_(); W.tlamt.zero_(); W.tfst.zero_()           # (not the last run's contact)
        I = MidpointIntegrator(self.b, a.dt, self.st.X.clone(), v0, self.st.Y.clone(), self.fg, tol=tol, ref=self.ref, contact=True)
        Tg = Tangent(I, PARAMS, tol=a.tan_tol, al_tol=a.tan_tol, start=a.tan_start, copies=self.copies) if tangents else None
        f, Jf, prods, cgs = [], [], 0, {p: 0 for p in PARAMS}
        for s in range(1, self.steps+1):
            if Tg is None:
                s_ = I.step()
            else:
                s_, info = Tg.step()
                for p in PARAMS:
                    cgs[p] += info['params'][p]['cg_iterations']
            prods += int(s_['products'])
            if s % self.every:
                continue
            x, y = I.x.clone(), I.y_int.clone()
            W.grad(x, y); F = W.tF.clone()
            fx = self.features(x, F)
            f.append(fx)
            if Tg is None:
                continue
            cols = {k: [] for k in fx}
            for p in PARAMS:
                xd, yd = Tg.T[p]['x'], Tg.T[p]['y']
                sc = max(float(xd.abs().max()), float(yd.abs().max()))
                if sc == 0:
                    for k in cols:
                        cols[k].append(torch.zeros_like(fx[k]))
                    continue
                h = 1e-7/sc
                W.grad(x+h*xd, y+h*yd); Fd = (W.tF-F)/h
                eps = a.delta/max(float(xd.abs().max()), 1e-300)
                fp, fm = self.features(x+eps*xd, F+eps*Fd), self.features(x-eps*xd, F-eps*Fd)
                for k in cols:
                    cols[k].append((fp[k]-fm[k])/(2*eps))
            Jf.append({k: torch.stack(v, -1) for k, v in cols.items()})
        st = dict(seconds=time.time()-t0, products=prods, tangent_cg=cgs if Tg else None)
        fo = {k: torch.stack([d[k] for d in f]) for k in self.cams}
        Jo = {k: torch.stack([d[k] for d in Jf]) for k in self.cams} if Tg else None
        return fo, Jo, st


def make_parser():
    a = parser()
    a.add_argument('--mode', choices=('synth', 'probe', 'real'), default='synth')
    a.add_argument('--real', default='output/round01_pruned_exact/identify/real_features.npz', help='--mode real: the masks\' features (real_check)')
    a.add_argument('--probe', nargs='+', default=['lnE', 'v0y'])
    a.add_argument('--probe-eps', type=float, default=1e-2)
    a.add_argument('--probe-tol', type=float, default=1e-6, help='the probe\'s difference runs (the 1e-3 solve\'s scatter would enter them)')
    a.add_argument('--probe-reuse', action='store_true', help='the tangent run\'s J and f from <tag>_probe_tangent.npz (same theta, ss)')
    a.add_argument('--truth', type=float, nargs=7, default=[math.log(23008.38), .3008, math.log(.005), -.408, -.5, -.066, math.log(.5)])
    a.add_argument('--start', type=float, nargs=7, default=[math.log(23008.38*.5), .4, math.log(.02), -.204, -.25, -.033, math.log(.3)])
    a.add_argument('--ground', type=float, default=-.05)
    a.add_argument('--frames', type=int, default=14)
    a.add_argument('--fit', type=int, nargs='+', default=list(range(10)))
    a.add_argument('--held', type=int, nargs='+', default=[10])
    a.add_argument('--tol', type=float, default=1e-3)
    a.add_argument('--tan-tol', type=float, default=5e-6, help='the tangents\' CG (|r| <= tol |b|) and constraint (10-25: 0.1-0.3 % over 0.2 s)')
    a.add_argument('--tan-start', choices=('predict', 'inertia', 'last'), default='predict')
    a.add_argument('--copies', type=int, default=4, help='product SkinWarps (~1.2 GB each; 4: 10 of 12.8 GB in use)')
    a.add_argument('--delta', type=float, default=2e-3, help='the features\' central difference: largest body displacement (m)')
    a.add_argument('--ss', type=int, nargs='+', default=[2], help='rays per pixel k x k (ss_camera); the first for the fit, all in probe')
    a.add_argument('--iters', type=int, default=15)
    a.add_argument('--ref-tol', type=float, default=2e-3, help='a second forward at --start: the forward\'s own error, estimated by the '
                   'difference to it, sets LM\'s stop (10-25: the error ~ tol, f(2e-3) - f(1e-3) 1.8 x the error against 1e-6 at synth14\'s '
                   'start, its stop unchanged; 252 s against 712 s)')
    a.add_argument('--tag', default='synth')
    a.set_defaults(dt=1/1200, beta=.01, integrator='midpoint')
    return a


def lm(run_, th, f, Jf, st, f_obs, nf, dvec, iters, log):
    """LM with a radius from the measured cost (2026-10-25). Each trial: the Gauss-Newton step, or with damping
    mu diag(J^T J) to radius (in that diagonal's norm) when it is longer; then the cost along the step fitted by
    c + slope a + kap a^2 (slope 2 r.J s from the tangents, kap from the trial's cost), the next radius a* |s| with the
    fit's minimum a* = -slope / (2 kap) (none when kap <= 0; a rejection gives a* <= 1/2). Stop when the trial's
    predicted change |J s| is not above |P_J dvec|, dvec the forward's own error (tol against a converged solve): the
    parameter error that error leaves; finer steps are below what the forward resolves. run_(th, tangents=True) ->
    f, J, stats (fit views first nf); log(it, th, c, f, st, accepted, info)"""
    def flat(f_, J_):
        return (f_[:, :nf]-f_obs[:, :nf]).reshape(-1).cpu().numpy(), J_[:, :nf].reshape(-1, len(PARAMS)).cpu().numpy()
    cost = lambda r_: float(r_@r_)
    r, J = flat(f, Jf)
    c, radius = cost(r), math.inf
    log(0, th, c, f, st, True, dict(radius=radius))
    for it in range(1, iters+1):
        D = np.diag(J.T@J)
        dn = lambda s_: float(np.sqrt((D*s_*s_).sum()))
        def step(mu):
            if mu == 0:
                return np.linalg.lstsq(J, -r, rcond=None)[0]
            return np.linalg.lstsq(np.vstack((J, np.diag(np.sqrt(mu*D)))), np.concatenate((-r, np.zeros(len(D)))), rcond=None)[0]
        res = float(np.linalg.norm(J@np.linalg.lstsq(J, dvec, rcond=None)[0]))
        while True:
            mu, s = 0., step(0.)
            if dn(s) > radius:                                                     # mu with |s(mu)| = radius (monotone), to
                lo, hi = 0., 1.                                                    # the last bit by bisection
                while dn(step(hi)) > radius:
                    lo, hi = hi, 2*hi
                while True:
                    m = .5*(lo+hi)
                    if not lo < m < hi:
                        break
                    lo, hi = (m, hi) if dn(step(m)) > radius else (lo, m)
                mu, s = hi, step(hi)
            tt = th+s
            tt[1] = min(max(tt[1], .01), .49)
            s = tt-th
            pred, slope = float(np.linalg.norm(J@s)), float(2*r@(J@s))
            if pred <= res:
                print('LM: the step\'s predicted change |J s| %.3e is not above the forward\'s error in J\'s range %.3e, stop' % (pred, res), flush=True)
                return th
            if slope >= 0:
                print('LM: the step (nu at its bound) does not descend, stop', flush=True)
                return th
            f_t, Jf_t, st_t = run_(tt)
            r_t, J_t = flat(f_t, Jf_t)
            c_t = cost(r_t)
            kap = c_t-c-slope
            a = -slope/(2*kap) if kap > 0 else math.inf
            radius = a*dn(s)
            info = dict(mu=mu, step=dn(s), pred=pred, resolution=res, pred_cost=c+slope+float((J@s)@(J@s)), alpha=a, radius=radius)
            if c_t < c:
                th, f, r, J, c = tt, f_t, r_t, J_t, c_t
                log(it, th, c, f, st_t, True, info)
                break
            log(it, tt, c_t, f_t, st_t, False, info)
    return th


def main():
    args = make_parser().parse_args()
    d = Path(args.out)/'identify'; d.mkdir(parents=True, exist_ok=True)
    P = Problem(args)
    if args.mode == 'probe':
        return probe(P, args)
    k0 = args.ss[0]

    def run_(th_, tangents=True, tol=None):
        f_, J_, st_ = P.run(th_, tangents=tangents, tol=tol)
        return f_[k0], (J_[k0] if J_ is not None else None), st_
    fo = d/('%s_obs_ss%d.npz' % (args.tag, k0))
    if args.mode == 'real':                                                        # the masks' features (frames 0.., the renderer's
        z = np.load(args.real)                                                     # camera order), frames 1..frames
        assert list(z['cameras']) == list(args.fit+args.held) and int(z['frames'][0]) == 0 and len(z['frames']) > args.frames
        f_obs = torch.as_tensor(z['f'][1:args.frames+1], device=dev)
        print('observations: the real masks (%s), frames 1..%d, views %s' % (args.real, args.frames, list(z['views'])), flush=True)
    elif fo.exists():
        f_obs = torch.as_tensor(np.load(fo)['f'], device=dev)
        print('observations read from %s' % fo, flush=True)
    else:
        f_obs, _, st = run_(args.truth, tangents=False)
        np.savez(fo, f=f_obs.cpu().numpy(), truth=np.array(args.truth))
        print('observations at the truth %s: %.0f s, %d products' % (np.round(args.truth, 4).tolist(), st['seconds'], st['products']), flush=True)
    nf = P.nfit
    th = np.array(args.start, dtype=float)
    f, Jf, st = run_(th)
    fe = d/('%s_fwd_error_ss%d.npz' % (args.tag, k0))                            # the forward's own error at --start
    z = np.load(fe) if fe.exists() else None
    if z is not None and np.array_equal(z['theta'], th) and float(z['tol']) == args.tol and float(z['ref_tol']) == args.ref_tol:
        dvec = z['dvec']
        print('the forward\'s error read from %s' % fe, flush=True)
    else:
        f_ref, _, st_r = run_(th, tangents=False, tol=args.ref_tol)
        dvec = (f[:, :nf]-f_ref[:, :nf]).reshape(-1).cpu().numpy()
        np.savez(fe, theta=th, tol=args.tol, ref_tol=args.ref_tol, dvec=dvec)
        print('the forward\'s error at the start (tol %.0e against %.0e, %.0f s): |d| %.3e px over the fit views' % (
            args.tol, args.ref_tol, st_r['seconds'], float(np.linalg.norm(dvec))), flush=True)
    hist = []
    held = lambda f_: float(((f_[:, nf:]-f_obs[:, nf:])**2).sum())

    def log(it, th_, c_, f_, st_, acc, info):
        e = dict(it=it, theta=th_.tolist(), E=math.exp(th_[0]), nu=th_[1], beta=math.exp(th_[2]), v0=th_[3:6].tolist(), mu=math.exp(th_[6]), cost=c_, held=held(f_),
                 accepted=acc, seconds=st_['seconds'], products=st_['products'], tangent_cg=st_['tangent_cg'], **info)
        hist.append(e)
        (d/('%s.json' % args.tag)).write_text(json.dumps(hist, indent=1))
        print('it %d %s: E %.1f, nu %.4f, beta %.5f, v0 %s, mu %.4f; cost %.4e (held view %.4e); %.0f s, products %d, tangent CG %s' % (
            it, 'accepted' if acc else 'rejected', e['E'], e['nu'], e['beta'], np.round(e['v0'], 4).tolist(), e['mu'], c_, e['held'], st_['seconds'],
            st_['products'], st_['tangent_cg']), flush=True)
        if it:
            print('   mu %.2e, |s| %.3e, |J s| %.3e (forward error in J\'s range %.3e), cost predicted %.4e, a* %.3g -> radius %.3e' % (
                info['mu'], info['step'], info['pred'], info['resolution'], info['pred_cost'], info['alpha'], info['radius']), flush=True)
    lm(run_, th, f, Jf, st, f_obs, nf, dvec, args.iters, log)
    if args.mode == 'synth':
        print('truth E %.1f, nu %.4f, beta %.5f, v0 %s, mu %.3f' % (math.exp(args.truth[0]), args.truth[1], math.exp(args.truth[2]), args.truth[3:6],
              math.exp(args.truth[6])), flush=True)


def probe(P, args):
    th = np.array(args.start, dtype=float)
    ft = Path(args.out)/'identify'/('%s_probe_tangent.npz' % args.tag)
    if args.probe_reuse:
        d = np.load(ft)
        assert np.array_equal(d['theta'], th), 'the saved tangent run is at another theta'
        f, Jf = ({k: torch.as_tensor(d['%s_ss%d' % (nm, k)], device=dev) for k in args.ss} for nm in ('f', 'J'))
        print('tangent run: from %s' % ft, flush=True)
    else:
        f, Jf, st = P.run(th)
        print('tangent run: %.0f s, products %d, tangent CG %s' % (st['seconds'], st['products'], st['tangent_cg']), flush=True)
        np.savez(ft, theta=th, **{'J_ss%d' % k: Jf[k].cpu().numpy() for k in args.ss}, **{'f_ss%d' % k: f[k].cpu().numpy() for k in args.ss})
    runs = {}
    for p in args.probe:
        j = PARAMS.index(p)
        e = args.probe_eps
        tp, tm = th.copy(), th.copy(); tp[j] += e; tm[j] -= e
        fp, _, sp = P.run(tp, tangents=False, tol=args.probe_tol); fm, _, _ = P.run(tm, tangents=False, tol=args.probe_tol)
        print('%s: difference runs at tol %.0e, %.0f s each' % (p, args.probe_tol, sp['seconds']), flush=True)
        runs.update({'%s_%s_ss%d' % (p, sg, k): v[k].cpu().numpy() for sg, v in (('plus', fp), ('minus', fm)) for k in args.ss})
        for k in args.ss:
            fd = (fp[k]-fm[k])/(2*e); T = Jf[k][..., j]
            nfd = fd.reshape(len(fd), -1).norm(dim=1); err = (T-fd).reshape(len(fd), -1).norm(dim=1)
            print('%s, %d x %d rays: per frame |J - FD| / max_frames |FD|: %s; |FD| per frame %s; all frames |J - FD| / |FD| %.2e' % (
                p, k, k, ' '.join('%.1e' % float(v) for v in err/nfd.max()), ' '.join('%.1e' % float(v) for v in nfd),
                float((T-fd).norm()/fd.norm())), flush=True)
            q = fd.reshape(-1).abs().argsort()[-5:]
            print('   the 5 largest FD entries: FD %s, J %s' % (np.round(fd.reshape(-1)[q].cpu().numpy(), 3).tolist(), np.round(T.reshape(-1)[q].cpu().numpy(), 3).tolist()), flush=True)
    np.savez(Path(args.out)/'identify'/('%s_probe.npz' % args.tag), theta=th, **{'J_ss%d' % k: Jf[k].cpu().numpy() for k in args.ss},
             **{'f_ss%d' % k: f[k].cpu().numpy() for k in args.ss}, **runs)


if __name__ == '__main__':
    main()
