"""How fast the joint solve (joint_step.JointSolver: bodies and skin, the block-diagonal preconditioner) converges
against the skin's own (bodies held, the turned group blocks), 2026-10-24: at rest (no contact, the references at rest,
x_n = X, v_n = 0, no external force: the minimum is the rest state), from the same random start (bodies and skin moved by
--amp), each solved to a residual 1e-6 of its start; products per e-fold ln(r0 / r) of the residual. The midpoint
rule's weights: inertia 4 m / dt^2, the viscous term 2 Et / dt.
  python free_surface/joint_cond_check.py --out output/round01_pruned_exact --dt 2.0833333e-4 --beta 0.01 --integrator midpoint"""
import math

import torch

from free_surface.joint_step import JointSolver
from free_surface.skin_contact_check import f64, dev
from free_surface.skin_dynamics_check import build_model, parser


def main():
    a = parser()
    a.add_argument('--perturb', type=float, default=1e-4)
    args = a.parse_args()
    assert args.integrator == 'midpoint'
    b = build_model(args)
    st, W, S, prec, mass = b['st'], b['W'], b['S'], b['prec'], b['mass']
    X, Y = st.X, st.Y
    W.visc_weight = 2.
    if W.vis is not None:
        W.visc_ref(X, Y)
    W.total_mode()
    gen = torch.Generator(device=dev).manual_seed(0)
    dx = args.perturb*torch.randn(X.shape, generator=gen, dtype=f64, device=dev)
    dy = args.perturb*torch.randn(Y.shape, generator=gen, dtype=f64, device=dev)

    def run(name, solve, r0):
        lines = []
        _, s_ = solve(r0, lines.append)
        rate = s_['products']/math.log(1/s_['residual']) if s_['residual'] < 1 else float('nan')
        print('%s: start residual %.3e; %d products to %.1e of it (%d outer, %d rejected, %d negative); products per e-fold %.1f; at 1e-2 / 1e-4 / 1e-6: %s' % (
            name, r0, s_['products'], s_['residual'], s_['outer'], s_['rejected'], s_['negative'], rate,
            ' / '.join(str(s_['hits'][m]) for m in (1e-2, 1e-4, 1e-6))), flush=True)
        for l_ in lines[:12]:
            print('   ', l_, flush=True)

    # the skin alone, bodies held at X
    ys = Y+dy
    r0 = float(S.grad(X, ys).norm())
    run('skin alone (bodies held)', lambda r, log: S.solve(X, ys, prec, r, tol=1e-6, maxit=4000, log=log), r0)
    # bodies and skin
    J = JointSolver(S, W, mass, b['kblocks'], b['vblocks'], args.dt)
    J.set_step(X, torch.zeros_like(X), torch.zeros_like(X))
    z = torch.cat((X+dx, Y+dy))
    r0 = float(J.eg(None, z)[1].norm())
    run('bodies and skin (joint)', lambda r, log: J.solve(None, z, prec, r, tol=1e-6, maxit=4000, log=log), r0)
    # bodies and skin, only the bodies moved at the start
    z = torch.cat((X+dx, Y.clone()))
    r0 = float(J.eg(None, z)[1].norm())
    run('joint, bodies moved only', lambda r, log: J.solve(None, z, prec, r, tol=1e-6, maxit=4000, log=log), r0)


if __name__ == '__main__':
    main()
