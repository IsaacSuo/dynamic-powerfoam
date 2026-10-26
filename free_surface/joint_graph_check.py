"""JointCG (each CG iteration of the joint solve one CUDA graph) against the torch Steihaug (2026-10-24), at a saved state
(skin_drop_check.py --save-state N, the midpoint rule): one product H d (the graph's first iteration against
JointSolver.hvp), then the step's first contact pass (JointSolver.solve with the active set held) from the same start
both ways: products, outer iterations, residual, |z_graph - z_torch| / |z - z_start|, wall time per product.
  python free_surface/joint_graph_check.py --out output/round01_pruned_exact --dt 2.0833333e-4 --tols 1e-4 --beta 0.01 --integrator midpoint --state output/round01_pruned_exact/skin_drop_state_step349.npz"""
import time
from pathlib import Path

import numpy as np
import torch

from free_surface.joint_step import JointCG, MidpointIntegrator
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
    restore()
    J, n = I.J, W.n
    x, v = I.x, I.v
    zs = torch.cat((x+.5*dt*v+.25*dt*dt*I.a, I.predict_skin(S, x+.5*dt*v+.25*dt*dt*I.a, I.xm1, I.xm2, I.ym1, I.ym2)))
    J.set_step(x, x+.5*dt*v, fg)
    # one product
    e0, g0, aux = J.eg(None, zs)
    P = J.prec_at(prec, None, zs, aux)
    d = P(-g0[None])
    Hd_t = J.hvp(None, zs, g0)(d)
    cg = JointCG(W, P.wprec, mass, J.c_in, J.h)
    Tt = cg.t; zv, gv, dv = zs.view(-1, 3), g0.view(-1, 3), d.view(-1, 3)
    Tt['zx'].copy_(zv[:n]); Tt['zy'].copy_(zv[n:]); Tt['gx'].copy_(gv[:n]); Tt['gy'].copy_(gv[n:]); Tt['dx'].copy_(dv[:n]); Tt['dy'].copy_(dv[n:])
    Tt['rx'].copy_(-gv[:n]); Tt['ry'].copy_(-gv[n:])
    cg.tPi.copy_(P.Pi); cg.txt.copy_(J.x_t); cg.tf.copy_(J.f_ext); cg.tsc[5] = float((g0*g0).sum())
    for s in 'xy':
        Tt['p'+s].zero_(); Tt['Hp'+s].zero_()
    dHd, _, _ = cg.step()
    Hd_g = torch.cat((Tt['Hdx'], Tt['Hdy'])).reshape(1, -1)
    print('one product: |Hd_graph - Hd_torch| / |Hd| %.2e; d.Hd graph %.10e, torch %.10e' % (
        float((Hd_g-Hd_t).norm()/Hd_t.norm()), dHd, float((d*Hd_t).sum())), flush=True)
    # the first pass both ways
    res = {}
    for use in (False, True):
        restore(); J.use_graph = use
        torch.cuda.synchronize(); t0 = time.perf_counter()
        z, s_ = J.solve(None, zs.clone(), prec, ref, tol=tol, maxit=2000)
        torch.cuda.synchronize(); t1 = time.perf_counter()
        res[use] = z
        print('%s: %d products, %d outer, %d rejected, residual %.2e ref; %.0f ms, %.2f ms per product' % (
            'graph' if use else 'torch', s_['products'], s_['outer'], s_['rejected'], s_['residual'], 1e3*(t1-t0), 1e3*(t1-t0)/max(s_['products'], 1)), flush=True)
    print('|z_graph - z_torch| / |z - z_start| %.2e' % float((res[True]-res[False]).norm()/(res[False]-zs).norm()), flush=True)


if __name__ == '__main__':
    main()
