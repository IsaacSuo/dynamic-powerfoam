"""The skin's start for the step's solve (2026-10-24): at deep compression the first pass is slow on the skin alone, and its
start error is the skin's (split_pass_check.py: the bodies' 100x smaller). From a saved state (skin_drop_check.py
--save-state N, the midpoint rule) N steps three times, the deviation from the cells' affine field extrapolated over the
last midpoints by coefficients c (y_s = sum c_k y_k + e, e = affine(x_new - sum c_k x_k, e), predict_skin's form):
linear (2, -1, the drop's), quadratic (3, -3, 1), filtered (1.25, 0.5, -0.75: linear through the means of consecutive
pairs, a step-to-step alternation cancels). Per step: first-pass and total products, passes, the skin's correction
c_n = y_mid - y_s (the solved midpoint minus the start) and cos(c_n, c_n-1) (+1: the deviation drifts smoothly, the
quadratic should help; -1: it alternates, the filtered should); on the linear run also each predictor's error offline.
  python free_surface/skin_predict_check.py --out output/round01_pruned_exact --dt 2.0833333e-4 --tols 1e-4 --beta 0.01 --integrator midpoint --state output/round01_pruned_exact/skin_drop_state_step900.npz --nsteps 20"""
import time
from pathlib import Path

import numpy as np
import torch

from free_surface.joint_step import MidpointIntegrator
from free_surface.skin_contact_check import f64, dev
from free_surface.skin_dynamics_check import build_model, parser

COEF = dict(linear=(2., -1.), quadratic=(3., -3., 1.), filtered=(1.25, .5, -.75))


def extrapolate(S, x_new, xh, yh, c, iters=4):
    """predict_skin with coefficients c over the midpoint history xh, yh (most recent first)"""
    dx = x_new-sum(c_*x_ for c_, x_ in zip(c, xh))
    e = torch.zeros_like(yh[0]).index_add(0, S.e_pid, dx[S.e_body])/S.cnt[:, None]
    for _ in range(iters):
        e = S.affine_skin(dx, e)
    return sum(c_*y_ for c_, y_ in zip(c, yh))+e


def main():
    a = parser()
    a.add_argument('--state', type=Path, required=True)
    a.add_argument('--gravity', type=float, default=9.8)
    a.add_argument('--nsteps', type=int, default=20)
    args = a.parse_args()
    b = build_model(args, contact=dict(normal=(0., 1., 0.), c0=0., shift=True))
    st, W, S, prec, mass = b['st'], b['W'], b['S'], b['prec'], b['mass']
    z0 = np.load(args.state)
    T = lambda k: torch.as_tensor(z0[k], device=dev)
    fg = -args.gravity*mass[:, None]*torch.tensor([0., 1., 0.], dtype=f64, device=dev)
    ref = float(fg.norm()); tol, dt = args.tols[0], args.dt
    I = MidpointIntegrator(b, dt, T('x'), T('v'), T('ym1'), fg, tol=tol, ref=ref, contact=True)
    J, n = I.J, W.n
    step0 = int(args.state.stem.split('step')[-1])

    def restore():
        I.x, I.v, I.a, I.xm1, I.ym1, I.xm2, I.ym2 = (T(k) for k in ('x', 'v', 'a', 'xm1', 'ym1', 'xm2', 'ym2'))
        W.tlam.copy_(T('lam')); W.tact.copy_(T('act')); W.tgoff.copy_(T('goff')); W.trho.copy_(T('rho'))
        for k_, r_ in W.tref.items():
            r_.copy_(T('ref_'+k_))
        W.total_mode()
        J.set_step(I.x, I.x+.5*dt*I.v, fg)
        z = torch.cat((I.xm1, I.ym1))                                              # the same preconditioner state (its
        _, _, aux = J.eg(None, z)                                                  # a.B^-1 a sets rho at the solve's
        J.prec_at(prec, None, z, aux)                                              # start) for every run
        W.trho.copy_(T('rho'))

    runs = {}
    for name, c in COEF.items():
        restore()
        H = dict(x3=None, y3=None)
        cur = {}
        def pred(S_, x_new, x1, x2, y1, y2):
            if len(c) == 3 and H['x3'] is not None:
                ys = extrapolate(S_, x_new, (x1, x2, H['x3']), (y1, y2, H['y3']), c)
            else:                                                                  # the first step: no third midpoint
                ys = extrapolate(S_, x_new, (x1, x2), (y1, y2), COEF['linear'])
            cur['ys'], cur['hist'] = ys, (x_new, (x1, x2, H['x3']), (y1, y2, H['y3']))
            return ys
        I.predict_skin = pred
        rows, hist, c_prev = [], [], None
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for k in range(args.nsteps):
            x_old, x2_old, y2_old = I.x, I.xm2, I.ym2
            xs = I.x+.5*dt*I.v+.25*dt*dt*I.a
            s_ = I.step()
            H['x3'], H['y3'] = x2_old, y2_old
            cy = s_['ym']-cur['ys']; cx = s_['xm']-xs
            cos = float((cy*c_prev).sum()/(cy.norm()*c_prev.norm())) if c_prev is not None else float('nan')
            c_prev = cy
            p = s_['passes']
            rows.append((step0+k, p[0][0], s_['products'], len(p), np.hypot(p[0][4], p[0][5]), float(cy.norm(dim=1).max()), float(cx.norm(dim=1).max()), cos))
            hist.append((cur['hist'], s_['ym'].clone()))
        torch.cuda.synchronize()
        runs[name] = rows
        print('== %s %s: %d steps, %.1f s' % (name, c, args.nsteps, time.perf_counter()-t0), flush=True)
        print('   step  first  total passes  start res | skin corr max (m)  bodies corr max (m)  cos(c_n, c_n-1)', flush=True)
        for r in rows:
            print('   %4d  %5d  %5d  %d       %.2e | %.2e           %.2e             %+.3f' % r, flush=True)
        R = np.array([r[1:] for r in rows], dtype=float)
        print('   median: first %.0f, total %.0f, passes %.1f, start residual %.2e, skin correction %.2e m; cos median %+.3f' % (
            np.median(R[:, 0]), np.median(R[:, 1]), np.median(R[:, 2]), np.median(R[:, 3]), np.median(R[:, 4]), np.nanmedian(R[:, 6])), flush=True)
        if name == 'linear':                                                       # every predictor's error on this run's history
            err = {k_: [] for k_ in COEF}
            for (x_new, xh, yh), ym in hist:
                if xh[2] is None:
                    continue
                for k_, c_ in COEF.items():
                    ys = extrapolate(S, x_new, xh[:len(c_)], yh[:len(c_)], c_)
                    err[k_].append(float((ym-ys).norm(dim=1).max()))
            print('   offline on this history, the skin start error max (m) per step, median: ' + ', '.join(
                '%s %.2e' % (k_, np.median(v_)) for k_, v_ in err.items()), flush=True)


if __name__ == '__main__':
    main()
