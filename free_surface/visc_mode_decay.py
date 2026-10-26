"""Kelvin-Voigt viscosity against stiffness-proportional damping theory (2026-10-23): eta = beta mu, zeta = beta (lam +
2 mu / 3) make the viscous stress beta times the elastic stress rate, so a mode of angular frequency omega has damping
ratio beta omega / 2. Start in one FEM mode (interpolated to the bodies, rigid part removed, largest displacement
--amp), at rest with the skin solved (consistent: skin and bodies both still), no ground or gravity; symplectic Euler
as free_vib_check.py, once without viscosity (the same model, SkinWarp.elastic_mode) and once with. The modal
coordinate q = sum m u . (x - X) / sum m |u|^2 fitted by c + A exp(-s t) cos(w t + p): omega_n = sqrt(w^2 + s^2),
damping ratio s / omega_n against beta omega_n / 2 (omega_n of the undamped run); kinetic + elastic + dissipated.
  python free_surface/visc_mode_decay.py --out output/round01_pruned_exact --fem output/round01_exact/fem_modes_h0.008.npz --beta 0.005 --modes 0 20"""
import time

import numpy as np
import torch
from scipy.optimize import curve_fit

from experiments.dem_fem_mode_match import interpolate
from free_surface.skin_contact_check import f64, dev
from free_surface.skin_dynamics_check import build_model, parser, predict_skin, visc_body_update


def mode_shape(cache, k, X, mass):
    """FEM mode k at the bodies, its rigid part (M-weighted translation and rotation about the centre of mass)
    removed, largest displacement 1"""
    return scaled_shape(cache, k, X, mass)[0]


def scaled_shape(cache, k, X, mass):
    """mode_shape and the factor s the FEM mode was divided by"""
    fem = dict(nodes=cache['nodes'], node_grid=cache['node_grid'], lo=cache['lo'], h=float(cache['h']), masses=cache['masses'])
    Ux, _ = interpolate(fem, cache['V'][:, [k]], X.cpu().numpy())
    u = torch.as_tensor(Ux[:, :, 0], dtype=f64, device=dev)
    m = mass[:, None]; M = float(mass.sum())
    c = (m*X).sum(0)/M; r = X-c
    u = u-(m*u).sum(0)/M
    I = (mass[:, None, None]*((r*r).sum(1)[:, None, None]*torch.eye(3, dtype=f64, device=dev)-r[:, :, None]*r[:, None, :])).sum(0)
    w = torch.linalg.solve(I, (m*torch.linalg.cross(r, u)).sum(0))
    u = u-torch.linalg.cross(w.expand_as(r), r)
    s = float(u.norm(dim=1).max())
    return u/s, s


def run(b, u, amp, steps, dt, tol, every, damped, vtol=1e-9, integ="split"):
    """records (t, q, kinetic, elastic, dissipated) per unit mass every `every` steps"""
    st, W, S, prec, mass = b['st'], b['W'], b['S'], b['prec'], b['mass']
    X, Y = st.X, st.Y
    M = float(mass.sum()); mu = mass[:, None]*u; uMu = float((mu*u).sum())
    W.elastic_mode()                                                               # the start without viscosity
    x = X+amp*u; v = torch.zeros_like(X)
    y0 = Y.clone()
    for _ in range(5):
        y0 = S.affine_skin(x, y0)
    y0, _ = S.solve(x, y0, prec, float(S.grad(x, y0).norm()), tol=1e-8, maxit=1000)
    yp = yc = y0; xp = x.clone()                                                  # at rest: one step back the same
    if damped:
        W.total_mode()
    ref = float((9.8*mass).norm())
    e, gx, _ = W.grad(x, yc); fx = -gx; el = float(W.tsum[:6].sum())
    rec, prods, cgs = [], [], []
    Dv = 0.
    if integ == 'midpoint':                                                        # bodies and skin in one implicit midpoint step:
        from free_surface.joint_step import MidpointIntegrator                     # elastic at integer times: at the skin
        I = MidpointIntegrator(b, dt, x, v, yc, torch.zeros_like(x), tol=tol, ref=ref)   # extrapolated (I.E_int)
        if not damped:
            W.ttw[7] = 0.; I.no_visc = True
        E_prev = el
        for s in range(steps+1):
            if s % every == 0:
                rec.append([s*dt, float((mu*(I.x-X)).sum())/uMu, .5*float((mass[:, None]*I.v*I.v).sum())/M, el/M, I.dissipated/M])
            if s == steps:
                break
            st_ = I.step(); prods.append(st_['products'])
            el = I.E_int                                                           # (at the skin extrapolated to integer times)
        W.visc_weight = 1.; W.total_mode()
        return np.array(rec), int(np.median(prods)), 0
    for s in range(steps+1):
        if s % every == 0:
            rec.append([s*dt, float((mu*(x-X)).sum())/uMu, .5*float((mass[:, None]*v*v).sum())/M, el/M, Dv/M])
        if s == steps:
            break
        if damped:                                                                 # the step's start the reference, the bodies'
            fx = -W.visc_ref(x, yc)[1]                                             # force there without the viscous term
        v = v+dt*fx/mass[:, None]
        if damped:
            v, k_ = visc_body_update(W, x, yc, mass, v, (yc-yp)/dt, dt, b['vblocks'], tol=vtol); cgs.append(k_)
        x_old = x
        x = x+dt*v
        yi = predict_skin(S, x, x_old, xp, yc, yp)
        xp = x_old
        yn, st_ = S.solve(x, yi, prec, ref, tol=tol, maxit=300)
        prods.append(st_['products'])
        ts = st_['aux'][3]; el = float(ts[:6].sum())
        if damped:
            Dv += W.dissipated(ts)
        fx = -st_['aux'][0]
        yp, yc = yc, yn
    W.total_mode()
    return np.array(rec), int(np.median(prods)), (int(np.median(cgs)) if cgs else 0)


def fit(t, q, w0):
    """c + A exp(-s t) cos(w t + p) by least squares from omega w0; returns (w, s, rms residual / A)"""
    f = lambda t_, c, A, s, w, p: c+A*np.exp(-s*t_)*np.cos(w*t_+p)
    P, _ = curve_fit(f, t, q, p0=[0., q[0], 0., w0, 0.], maxfev=20000)
    return abs(P[3]), P[2], float(np.sqrt(np.mean((f(t, *P)-q)**2)))/abs(P[1])


def main():
    a = parser()
    a.add_argument('--modes', type=int, nargs='+', default=[0, 20])
    a.add_argument('--periods', type=float, default=2.5, help='run length in FEM periods')
    a.add_argument('--every', type=int, default=4)
    a.add_argument('--no-free', action='store_true', help='skip the run without viscosity (theory from the viscous fit\'s omega_n)')
    a.add_argument('--tag', default='', help='suffix of the output file')
    args = a.parse_args()
    assert args.beta > 0 and args.eta is None and args.zeta is None
    b = build_model(args)
    cache = np.load(args.fem)
    dt, tol = args.dt, args.tols[0]
    out = {}
    for k in args.modes:
        u = mode_shape(cache, k, b['st'].X, b['mass'])
        steps = int(np.ceil(args.periods/cache['f'][k]/dt))
        w_fem = 2*np.pi*cache['f'][k]
        res = {}
        for damped in ((True,) if args.no_free else (False, True)):
            t0 = time.time()
            rec, pr, cg = run(b, u, args.amp, steps, dt, tol, args.every, damped, args.visc_tol, args.integrator)
            w, s, r = fit(rec[:, 0], rec[:, 1], w_fem)
            wn = np.hypot(w, s)
            H = rec[:, 2]+rec[:, 3]+rec[:, 4]
            res[damped] = (wn, s/wn)
            print('mode %d (FEM %.2f Hz), %s: %d steps (%.3f s), %.0f s; fit: omega_n %.3f rad/s (%.3f Hz, x%.4f FEM), damping ratio %.4f, rms residual / amplitude %.3f; '
                  'kinetic + elastic + dissipated: start %.4e, change max %.2e min %.2e (relative %.1e); dissipated %.4e; products per step median %d, viscous CG %d' % (
                      k, cache['f'][k], 'viscous' if damped else 'no viscosity', steps, steps*dt, time.time()-t0, wn, wn/2/np.pi, wn/w_fem, s/wn, r,
                      H[0], (H-H[0]).max(), (H-H[0]).min(), np.abs(H-H[0]).max()/H[0], rec[-1, 4], pr, cg), flush=True)
            out['mode%d_%s' % (k, 'visc' if damped else 'free')] = rec
        w0, z0 = res.get(False, (res[True][0], 0.))
        print('mode %d: damping ratio %.4f with viscosity, %.4f without; theory beta omega / 2 = %.4f (omega %.3f %s): ratio %.3f' % (
            k, res[True][1], z0, args.beta*w0/2, w0, 'of the viscous fit' if args.no_free else 'undamped', (res[True][1]-z0)/(args.beta*w0/2)), flush=True)
    np.savez(b['out']/('visc_mode_decay_beta%g%s.npz' % (args.beta, args.tag)), **out)


if __name__ == '__main__':
    main()
