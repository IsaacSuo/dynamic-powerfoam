"""Where is the first pass slow (2026-10-24)? At a saved state (skin_drop_check.py --save-state N, the midpoint rule) the
step's first contact pass (active set and multipliers held) from the integrator's start z_s = (x_s, y_s), three ways:
joint (JointSolver.solve, the drop's) -> z* = (x*, y*); the skin alone from y_s with the bodies held at x* (WarpSkinSolver
.solve, its group blocks turned, the held contacts by Sherman-Morrison); the bodies alone from x_s with the skin held at
y* (Newton-CG on Pi(x, y*), the joint's body blocks c_in m + R (K_B + 2 D_B / dt) R^T, one-sided products, each CG to
0.01 of the residual as SkinSolver.solve). Products per e-fold of the residual: slow skin alone -> coupling among the
skin's groups; slow bodies alone -> among the bodies; both fast -> between the bodies and the skin.
  python free_surface/split_pass_check.py --out output/round01_pruned_exact --dt 2.0833333e-4 --tols 1e-4 --beta 0.01 --integrator midpoint --state output/round01_pruned_exact/skin_drop_state_step900.npz"""
import math
import time
from pathlib import Path

import numpy as np
import torch

from free_surface.joint_step import MidpointIntegrator
from free_surface.skin_contact_check import f64, dev
from free_surface.skin_dynamics_check import build_model, parser


def main():
    a = parser()
    a.add_argument('--state', type=Path, required=True)
    a.add_argument('--gravity', type=float, default=9.8)
    args = a.parse_args()
    b = build_model(args, contact=dict(normal=(0., 1., 0.), c0=0., shift=True))
    st, W, S, prec, mass = b['st'], b['W'], b['S'], b['prec'], b['mass']
    z0 = np.load(args.state)
    T = lambda k: torch.as_tensor(z0[k], device=dev)
    fg = -args.gravity*mass[:, None]*torch.tensor([0., 1., 0.], dtype=f64, device=dev)
    ref = float(fg.norm()); tol, dt = args.tols[0], args.dt
    I = MidpointIntegrator(b, dt, T('x'), T('v'), T('ym1'), fg, tol=tol, ref=ref, contact=True)
    I.x, I.v, I.a, I.xm1, I.ym1, I.xm2, I.ym2 = (T(k) for k in ('x', 'v', 'a', 'xm1', 'ym1', 'xm2', 'ym2'))
    def restore():
        W.tlam.copy_(T('lam')); W.tact.copy_(T('act')); W.tgoff.copy_(T('goff')); W.trho.copy_(T('rho'))
        for k_, r_ in W.tref.items():
            r_.copy_(T('ref_'+k_))
        W.total_mode()
    restore()
    J, n = I.J, W.n
    x, v = I.x, I.v
    xs = x+.5*dt*v+.25*dt*dt*I.a                                                   # the integrator's start (MidpointIntegrator.step)
    ys = I.predict_skin(S, xs, I.xm1, I.xm2, I.ym1, I.ym2)
    zs = torch.cat((xs, ys))
    J.set_step(x, x+.5*dt*v, fg)
    print('state %s: %d bodies, %d skin points, %d active contacts' % (args.state.name, n, W.npt, int((W.tact > 0).sum())), flush=True)
    def report(name, r0, s_, ms):
        ef = math.log(r0/max(s_['residual'], 1e-300))
        print('%-28s: start residual %.3e, %4d products, %2d outer, %d rejected, %d negative; residual %.2e; %.1f products per e-fold; %.0f ms' % (
            name, r0, s_['products'], s_['outer'], s_['rejected'], s_['negative'], s_['residual'], s_['products']/ef, ms), flush=True)

    # joint (the drop's first pass)
    g0 = J.eg(None, zs)[1].view(-1, 3)
    r0 = float(g0.norm())/ref
    print('joint start residual: bodies %.3e, skin %.3e of |m g|' % (float(g0[:n].norm())/ref, float(g0[n:].norm())/ref), flush=True)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    z, s_ = J.solve(None, zs.clone(), prec, ref, tol=tol, maxit=2000)
    torch.cuda.synchronize()
    report('joint', r0, s_, 1e3*(time.perf_counter()-t0))
    xa, ya = z[:n].clone(), z[n:].clone()
    print('the pass moved: bodies |x* - x_s| %.3e m (max %.3e), skin |y* - y_s| %.3e m (max %.3e)' % (
        float((xa-xs).norm()), float((xa-xs).norm(dim=1).max()), float((ya-ys).norm()), float((ya-ys).norm(dim=1).max())), flush=True)

    # the skin alone, bodies held at x*
    restore()
    def skin_prec_at(mp, x_, y_, aux):                                             # the held active set (as JointSolver.prec_at)
        f = mp(aux[1])
        f.wprec.contact_update(S.contact_weights(aux[4]), W.trho*float(W.tcpar[1]*W.tcpar[0]**2)*(W.tact > 0))
        return f
    S.prec_at = skin_prec_at
    r0s = float(S.eg(xa, ys)[1].norm())/ref
    torch.cuda.synchronize(); t0 = time.perf_counter()
    y1, s_ = S.solve(xa, ys.clone(), prec, ref, tol=tol, maxit=2000)
    torch.cuda.synchronize()
    report('skin alone (bodies at x*)', r0s, s_, 1e3*(time.perf_counter()-t0))
    print('   |y - y*| / |y* - y_s| %.1e' % float((y1-ya).norm()/(ya-ys).norm()), flush=True)

    # the bodies alone, skin held at y*
    restore()
    m = mass[:, None]
    def grad_x(x_):
        return W.grad(x_, ya)[1]+J.c_in*m*(x_-J.x_t)-J.f_ext
    Bb0 = J.Kb+J.wvis/dt*J.Db
    eye = torch.eye(3, dtype=f64, device=dev)
    xb, prods, outer, h = xs.clone(), 0, 0, J.h
    g = grad_x(xb); r0b = float(g.norm())/ref; res = r0b
    torch.cuda.synchronize(); t0 = time.perf_counter()
    while res > tol and prods < 2000:
        outer += 1
        W.grad(xb, ya); R = W.tR
        Pi = torch.linalg.inv(J.c_in*mass[:, None, None]*eye+R@Bb0@R.transpose(1, 2))
        P = lambda r_: torch.einsum('nij,nj->ni', Pi, r_)
        stop = max(tol, .01*res)*ref
        r = -g; zz = P(r); d = zz.clone(); rz = float((r*zz).sum()); p = torch.zeros_like(xb)
        while True:
            s = float(d.abs().max())
            Hd = (grad_x(xb+(h/s)*d)-g)*(s/h); prods += 1
            al = rz/float((d*Hd).sum())
            p += al*d; r -= al*Hd
            if float(r.norm()) <= stop or prods >= 2000:
                break
            zz = P(r); rz1 = float((r*zz).sum()); d = zz+(rz1/rz)*d; rz = rz1
        xb = xb+p
        g = grad_x(xb); res = float(g.norm())/ref
    torch.cuda.synchronize()
    report('bodies alone (skin at y*)', r0b, dict(residual=res, products=prods, outer=outer, rejected=0, negative=0), 1e3*(time.perf_counter()-t0))
    print('   |x - x*| / |x* - x_s| %.1e' % float((xb-xa).norm()/(xa-xs).norm()), flush=True)


if __name__ == '__main__':
    main()
