"""Free vibration without ground or gravity (2026-10-23), the counterpart of mpm_drop_chunked.py --ground none --gravity 0
--squeeze s: the bodies at rest with v_y = -s (y - y_com), symplectic Euler, the skin solved every step (no contact,
tolerance relative to the bodies' weights |m g| as in the drop); every --every steps the total energy (kinetic + elastic)
and the bodies' height spread (second moment about the centre of mass, 1-99 % extent). Writes free_vib_check.npz.
  python free_surface/free_vib_check.py --out output/round01_pruned_exact --dt 2.0833333e-4 --steps 1600 --tols 1e-4"""
import time

import numpy as np
import torch

from free_surface.skin_contact_check import f64, dev
from free_surface.skin_dynamics_check import build_model, parser, predict_skin, visc_body_update


def main():
    a = parser()
    a.add_argument('--squeeze', type=float, default=11.127)
    a.add_argument('--every', type=int, default=20)
    args = a.parse_args()
    b = build_model(args)
    st, W, S, prec, mass = b['st'], b['W'], b['S'], b['prec'], b['mass']
    X, Y = st.X, st.Y
    M = float(mass.sum())
    ycom = float((mass*X[:, 1]).sum())/M
    v = torch.zeros_like(X); v[:, 1] = -args.squeeze*(X[:, 1]-ycom)
    ref = float((9.8*mass).norm()); tol, dt = args.tols[0], args.dt
    x = X.clone(); yc = Y.clone()
    xp = x-dt*v; yp = predict_skin(S, xp, x, x, yc, yc)                           # one step back: the skin moving with the bodies
    e, gx, _ = W.grad(x, yc); fx = -gx; el = float(W.tsum[:6].sum())
    rec, prods, cgs = [], [], []
    Dv = 0.                                                                        # energy the viscosity took
    t0 = time.time()
    for s in range(args.steps+1):
        if s % args.every == 0:
            K = .5*float((mass[:, None]*v*v).sum())/M; El = el/M
            yy = x[:, 1]; yc_ = float((mass*yy).sum())/M
            q = torch.quantile(yy, torch.tensor([.01, .99], dtype=f64, device=dev))
            rec.append([s*dt, K, El, K+El, float((mass*(yy-yc_)**2).sum())/M, float(q[1]-q[0]), Dv/M])
        if s == args.steps:
            break
        if W.vis is not None:                                                    # viscosity: the step's start the reference, the
            fx = -W.visc_ref(x, yc)[1]                                           # bodies' force there without the viscous term
        v = v+dt*fx/mass[:, None]
        if W.vis is not None:                                                    # and the bodies implicit there
            v, k_ = visc_body_update(W, x, yc, mass, v, (yc-yp)/dt, dt, b['vblocks'], tol=args.visc_tol); cgs.append(k_)
        x_old = x
        x = x+dt*v
        yi = predict_skin(S, x, x_old, xp, yc, yp)
        xp = x_old
        yn, st_ = S.solve(x, yi, prec, ref, tol=tol, maxit=300)
        prods.append(st_['products'])
        ts = st_['aux'][3]; el = float(ts[:6].sum()); Dv += W.dissipated(ts)
        fx = -st_['aux'][0]
        yp, yc = yc, yn
    rec = np.array(rec)
    H = rec[:, 3]+rec[:, 6]
    print('free vibration, squeeze %.3f /s: %d steps, %.0f s; kinetic at start %.4f J/kg; kinetic + elastic + dissipated change max %+.2e min %+.2e last %+.2e J/kg (relative %.1e); '
          'products per step median %d; viscous CG per step median %s; energy taken by viscosity %.4f J/kg, kinetic + elastic left %.4f' % (
              args.squeeze, args.steps, time.time()-t0, rec[0, 1], (H-H[0]).max(), (H-H[0]).min(), H[-1]-H[0], np.abs(H-H[0]).max()/H[0],
              int(np.median(prods)), int(np.median(cgs)) if cgs else '-', rec[-1, 6], rec[-1, 3]), flush=True)
    np.savez(b['out']/('free_vib_check%s.npz' % ('_beta%g' % args.beta if args.beta else '')), rec=rec, cols=np.array(['t', 'K', 'elastic', 'K+elastic', 'y2', 'extent', 'dissipated']))


if __name__ == '__main__':
    main()
