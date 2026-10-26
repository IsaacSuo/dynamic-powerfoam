"""Where a step's time goes besides the CG products (2026-10-24: the drop spent ~68 ms per contact step outside them,
17 ms per free-fall step). From a saved state (skin_drop_check.py --save-state N, the midpoint rule) N steps of
MidpointIntegrator.step with each part timed between synchronisations (counts and ms per step): the skin's start
(predict_skin), JointSolver.eg (a full gradient with the inertia), prec_at, steihaug (the CG, the products inside), the
rest of each solve and of solve_contact, elastic_at (the energy record), every SkinWarp.grad call (nested in the above;
outside them: the measures' update), and the step's remainder.
  python free_surface/step_profile.py --out output/round01_pruned_exact --dt 2.0833333e-4 --tols 1e-4 --beta 0.01 --integrator midpoint --state output/round01_pruned_exact/skin_drop_state_step900.npz --nsteps 20"""
import time
from collections import defaultdict
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
    a.add_argument('--nsteps', type=int, default=20)
    args = a.parse_args()
    b = build_model(args, contact=dict(normal=(0., 1., 0.), c0=0., shift=True))
    st, W, S, prec, mass = b['st'], b['W'], b['S'], b['prec'], b['mass']
    z0 = np.load(args.state)
    T = lambda k: torch.as_tensor(z0[k], device=dev)
    fg = -args.gravity*mass[:, None]*torch.tensor([0., 1., 0.], dtype=f64, device=dev)
    ref = float(fg.norm()); tol, dt = args.tols[0], args.dt
    I = MidpointIntegrator(b, dt, T('x'), T('v'), T('ym1'), fg, tol=tol, ref=ref, contact=True)
    I.x, I.v, I.a, I.xm1, I.ym1, I.xm2, I.ym2 = (T(k) for k in ('x', 'v', 'a', 'xm1', 'ym1', 'xm2', 'ym2'))
    W.tlam.copy_(T('lam')); W.tact.copy_(T('act')); W.tgoff.copy_(T('goff')); W.trho.copy_(T('rho'))
    for k_, r_ in W.tref.items():
        r_.copy_(T('ref_'+k_))
    W.total_mode()
    J = I.J
    I.step()                                                                       # (warm: graphs captured, kernels built)

    tm, nc = defaultdict(float), defaultdict(int)
    def timed(name, f):
        def g(*a_, **k_):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            r = f(*a_, **k_)
            torch.cuda.synchronize(); tm[name] += time.perf_counter()-t0; nc[name] += 1
            return r
        return g
    prods = [0]
    st_h = J.steihaug
    def steihaug(*a_, **k_):
        r = st_h(*a_, **k_)
        prods[0] += r[-1]
        return r
    I.predict_skin = timed('predict_skin', I.predict_skin)
    J.eg = timed('eg', J.eg)
    J.prec_at = timed('prec_at', J.prec_at)
    J.steihaug = timed('steihaug', steihaug)
    J.solve = timed('solve', J.solve)
    J.solve_contact = timed('solve_contact', J.solve_contact)
    I.elastic_at = timed('elastic_at', I.elastic_at)
    W.grad = timed('W.grad', W.grad)
    step = timed('step', I.step)
    for _ in range(args.nsteps):
        step()
    ns = args.nsteps
    ms = lambda k: 1e3*tm[k]/ns
    print('%d steps from %s: %.1f products per step, %.1f ms per step' % (ns, args.state.name, prods[0]/ns, ms('step')), flush=True)
    print('   %-34s %8s %10s' % ('part', 'calls/step', 'ms/step'), flush=True)
    for k in ('step', 'solve_contact', 'solve', 'steihaug', 'eg', 'prec_at', 'predict_skin', 'elastic_at', 'W.grad'):
        print('   %-34s %8.1f %10.2f' % (k, nc[k]/ns, ms(k)), flush=True)
    print('   steihaug per product %.3f ms' % (1e3*tm['steihaug']/max(prods[0], 1)), flush=True)
    print('   the rest of the solves (outside steihaug, eg, prec_at) %.2f ms/step; of solve_contact outside its solves %.2f ms/step' % (
        ms('solve')-ms('steihaug')-ms('eg')-ms('prec_at'), ms('solve_contact')-ms('solve')), flush=True)
    print('   outside solve_contact: predict_skin %.2f, elastic_at %.2f, the rest of the step %.2f ms/step (W.grad calls outside eg and elastic_at: %.1f per step)' % (
        ms('predict_skin'), ms('elastic_at'), ms('step')-ms('solve_contact')-ms('predict_skin')-ms('elastic_at'),
        (nc['W.grad']-nc['eg']-nc['elastic_at'])/ns), flush=True)
    print('   outside the CG altogether: %.2f ms/step (%.0f %% of the step)' % (ms('step')-ms('steihaug'), 100*(1-tm['steihaug']/tm['step'])), flush=True)


if __name__ == '__main__':
    main()
