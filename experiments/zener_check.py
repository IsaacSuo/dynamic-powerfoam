"""Check of the Zener bond springs (WarpFoam visco=True) on a two-body oscillator.

Two bodies joined by one bond, no gravity, ground far away, no dashpots; the pair is released with a relative
velocity along the bond, so only the normal spring k0 acts: m_r u'' = -k0 (u + g (u - z)), z' = (u - z) / tau.
Characteristic equation (s = -decay + i omega): m_r s^2 + k0 (1 + g s tau / (1 + s tau)) = 0, a cubic; the
oscillatory root is compared with the frequency and decay fitted to the simulated separation. Limits: g = 0 (plain
spring), tau -> inf (stiffness (1 + g) k0, no loss), tau -> 0 (stiffness k0, no loss).
"""
import argparse

import numpy as np
import torch
import warp as wp
from scipy.optimize import curve_fit

from experiments.foam_dem_warp import WarpFoam, _new_state, _pad_state, _state_dtypes
from experiments.pacnerf_dem_forward import build


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--a', type=float, default=.004)
    p.add_argument('--E', type=float, default=4e4)
    p.add_argument('--gs', type=float, nargs='+', default=[0., 1., 1., 1., 3.])
    p.add_argument('--taus', type=float, nargs='+', default=[.01, 1e3, 1e-7, .001, .0005])
    args = p.parse_args()
    wp.init()
    dev = 'cuda'
    a = args.a
    g = dict(points=np.array([[0., 0., 1.], [a, 0., 1.]]), radii=np.full(2, a*np.sqrt(1/2.4)), pairs=np.array([[0, 1]]),
             areas=np.array([a*a]), contacts=np.array([[a/2, 0., 1.]]), masses=np.full(2, a**3), fixed=np.zeros(2, bool))
    for gv, tau in zip(args.gs, args.taus):
        m = build(g, args.E, .3, dev, 'noslip', .25, .3, torch.float32, nodalvol=True)
        m.set_environment(gravity=[0, 0, 0], ground_point=[0, 0, -10.], ground_normal=[0, 0, 1],
                          contact_radius=m.contact_radius.cpu().numpy(), ground_mode='noslip')
        wf = WarpFoam(m, contact='hertz', implicit_damping=True, visco=True, visco_g0=max(gv, 1e-12), visco_tau0=tau, visco_cols=(8, 9))
        th = torch.zeros(2, 10, dtype=torch.float32, device=dev)
        th[:, 6] = -30.                       # no dashpots
        k0 = float(m.base[0, 0]); mr = float(m.mass[0])/2
        w0 = np.sqrt(k0*(1+gv)/mr)
        dt = 2*np.pi/w0/400
        x0, q0, _, w_ = m.initial_state()
        v = torch.zeros_like(x0); v[0, 0] = -.01; v[1, 0] = .01
        dts = _state_dtypes(wf)
        st = tuple(wp.from_torch(t.float().contiguous().clone(), dtype=d) for t, d in zip(_pad_state(wf, (x0, q0, v, w_)), dts))
        st1 = _new_state(wf)
        buf = wf._buffers(False)
        thw = wp.from_torch(th)
        T = 6*2*np.pi/np.sqrt(k0/mr)
        n = int(T/dt)
        sep, ts = [], []
        for k in range(n):
            for b_ in buf.values():
                b_.zero_()
            wf.step(thw, st, st1, buf, dt)
            st, st1 = st1, st
            if k % 4 == 0:
                xx = wp.to_torch(st[0])
                sep.append(float(xx[1, 0]-xx[0, 0])-a); ts.append((k+1)*dt)
        ts, sep = np.array(ts), np.array(sep)
        f = lambda t, A, lam, om, ph, c: c+A*np.exp(-lam*t)*np.sin(om*t+ph)
        om0 = np.sqrt(k0/mr)
        best = None
        for wguess in (om0, om0*np.sqrt(1+gv)):
            try:
                pp, _ = curve_fit(f, ts, sep, p0=(np.abs(sep).max(), 1., wguess, 0., 0.), maxfev=40000)
                r = ((f(ts, *pp)-sep)**2).sum()
                if best is None or r < best[0]:
                    best = (r, pp)
            except RuntimeError:
                pass
        # analytic: mr s^3 tau + mr s^2 + k0 (1 + g) tau s + k0 = 0  (multiply by (1 + s tau))
        roots = np.roots([mr*tau, mr, k0*(1+gv)*tau, k0]) if gv > 0 else np.roots([mr, 0., k0])
        osc = roots[np.argmax(np.abs(roots.imag))]
        # exact linear ODE (fine RK45): m_r u'' = -k0 (u + g (u - z)), z' = (u - z) / tau, u(0) = 0, u'(0) = 0.02
        from scipy.integrate import solve_ivp
        rhs = lambda t, y: [y[1], -k0*(y[0]+gv*(y[0]-y[2]))/mr, (y[0]-y[2])/tau if gv > 0 else 0.]
        sol = solve_ivp(rhs, (0, ts[-1]), [0., .02, 0.], t_eval=ts, rtol=1e-10, atol=1e-14, method='Radau')
        err = np.sqrt(((sol.y[0]-sep)**2).mean())/np.abs(sol.y[0]).max()
        print('   trajectory vs exact ODE: rms error %.2e of the peak amplitude' % err)
        print('g %.2f tau %.1e s | simulated: omega %.2f rad/s, decay %.3f /s | analytic: omega %.2f, decay %.3f | '
              'plain spring omega %.2f, instantaneous %.2f' % (gv, tau, abs(best[1][2]), best[1][1], abs(osc.imag), -osc.real,
                                                             om0, om0*np.sqrt(1+gv)), flush=True)


if __name__ == '__main__':
    main()
