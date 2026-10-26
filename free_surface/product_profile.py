"""Where a product's 2.7 ms go (2026-10-24). At a saved state (skin_drop_check.py --save-state N, the midpoint rule):
the gradient pass (SkinWarp's graph) and one JointCG iteration (its graph) timed by replays; one JointCG iteration run
outside the graph under wp.timing (each kernel, memset, memcpy on its own, the sum against the graph's time: the rest is
launch overhead); and this GPU's float64 against float32 throughput by a multiply-add loop (independent chains).
  python free_surface/product_profile.py --out output/round01_pruned_exact --dt 2.0833333e-4 --tols 1e-4 --beta 0.01 --integrator midpoint --state output/round01_pruned_exact/skin_drop_state_step900.npz"""
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import warp as wp

from free_surface.joint_step import JointCG, MidpointIntegrator
from free_surface.skin_contact_check import f64, dev
from free_surface.skin_dynamics_check import build_model, parser


@wp.kernel(enable_backward=False)
def k_fma64(out: wp.array(dtype=wp.float64), n_it: int):
    i = wp.tid()
    a = wp.float64(i)*wp.float64(1e-9)
    b = a+wp.float64(1.0)
    c = a+wp.float64(2.0)
    d = a+wp.float64(3.0)
    m = wp.float64(0.999999)
    s = wp.float64(1e-7)
    for k in range(n_it):
        a = a*m+s
        b = b*m+s
        c = c*m+s
        d = d*m+s
    out[i] = a+b+c+d


@wp.kernel(enable_backward=False)
def k_fma32(out: wp.array(dtype=wp.float32), n_it: int):
    i = wp.tid()
    a = wp.float32(i)*wp.float32(1e-9)
    b = a+wp.float32(1.0)
    c = a+wp.float32(2.0)
    d = a+wp.float32(3.0)
    m = wp.float32(0.999999)
    s = wp.float32(1e-7)
    for k in range(n_it):
        a = a*m+s
        b = b*m+s
        c = c*m+s
        d = d*m+s
    out[i] = a+b+c+d


def replay_ms(f, reps):
    for _ in range(3):
        f()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(reps):
        f()
    torch.cuda.synchronize()
    return 1e3*(time.perf_counter()-t0)/reps


def main():
    a = parser()
    a.add_argument('--state', type=Path, required=True)
    a.add_argument('--gravity', type=float, default=9.8)
    args = a.parse_args()

    # the GPU's float64 / float32 multiply-add rate
    nt, nit = 1 << 20, 1024
    for name, k_, dt_ in (('float32', k_fma32, wp.float32), ('float64', k_fma64, wp.float64)):
        out = wp.zeros(nt, dtype=dt_, device='cuda')
        ms = replay_ms(lambda: wp.launch(k_, nt, inputs=[out, nit]), 5)
        print('%s multiply-add: %.2f ms for %.2e flops, %.0f GFLOP/s' % (name, ms, 8.*nt*nit, 8.*nt*nit/ms/1e6), flush=True)

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
    J, n = I.J, W.n
    x, v = I.x, I.v
    xs = x+.5*dt*v+.25*dt*dt*I.a
    zs = torch.cat((xs, I.predict_skin(S, xs, I.xm1, I.xm2, I.ym1, I.ym2)))
    J.set_step(x, x+.5*dt*v, fg)
    print('sizes: %d bodies, %d skin points, %d bonds, %d skin entries, %d contact cells (%d active)' % (
        n, W.npt, W.nb, W.ne, W.ct['k'], int((W.tact > 0).sum())), flush=True)

    # the gradient pass and one CG iteration, as graphs
    zx, zy = zs[:n].contiguous(), zs[n:].contiguous()
    W.tx.copy_(zx); W.ty.copy_(zy)
    ms_graph = replay_ms(lambda: wp.capture_launch(W.graph), 200)
    ms_grad = replay_ms(lambda: W.grad(zx, zy), 200)
    e0, g0, aux = J.eg(None, zs)
    P = J.prec_at(prec, None, zs, aux)
    d = P(-g0[None])
    cg = JointCG(W, P.wprec, mass, J.c_in, J.h)
    Tt = cg.t; zv, gv, dv = zs.view(-1, 3), g0.view(-1, 3), d.view(-1, 3)
    def reset_cg():
        Tt['zx'].copy_(zv[:n]); Tt['zy'].copy_(zv[n:]); Tt['gx'].copy_(gv[:n]); Tt['gy'].copy_(gv[n:]); Tt['dx'].copy_(dv[:n]); Tt['dy'].copy_(dv[n:])
        Tt['rx'].copy_(-gv[:n]); Tt['ry'].copy_(-gv[n:])
        cg.tPi.copy_(P.Pi); cg.txt.copy_(J.x_t); cg.tf.copy_(J.f_ext); cg.tsc[5] = float((g0*g0).sum())
        for s in 'xy':
            Tt['p'+s].zero_(); Tt['Hp'+s].zero_()
    reset_cg()
    ms_cg = replay_ms(cg.step, 50)
    print('gradient pass graph %.3f ms (W.grad with its copies %.3f ms); one JointCG iteration (graph, one host sync) %.3f ms' % (
        ms_graph, ms_grad, ms_cg), flush=True)

    # one CG iteration outside the graph, each activity timed
    reps = 5
    reset_cg()
    cg._iteration(); wp.synchronize()
    wp.timing_begin(cuda_filter=wp.TIMING_KERNEL | wp.TIMING_MEMSET | wp.TIMING_MEMCPY)
    for _ in range(reps):
        cg._iteration()
    res = wp.timing_end()
    tot, cnt, kind = defaultdict(float), defaultdict(int), {}
    for r in res:
        tot[r.name] += r.elapsed/reps; cnt[r.name] += 1; kind[r.name] = r.filter
    names = sorted(tot, key=lambda k_: -tot[k_])
    by = defaultdict(float)
    for k_ in names:
        by[{wp.TIMING_KERNEL: 'kernels', wp.TIMING_MEMSET: 'memsets', wp.TIMING_MEMCPY: 'memcpys'}.get(kind[k_], str(kind[k_]))] += tot[k_]
    s_all = sum(tot.values())
    print('one CG iteration outside the graph: %d activities, GPU time %.3f ms (%s); the graph took %.3f ms: %.3f ms launch / gap overhead' % (
        sum(cnt.values())//reps, s_all, ', '.join('%s %.3f' % kv for kv in by.items()), ms_cg, ms_cg-s_all), flush=True)
    print('   %-60s %6s %9s %6s' % ('activity', 'count', 'ms', 'share'), flush=True)
    acc = 0.
    for k_ in names:
        acc += tot[k_]
        print('   %-60s %6d %9.4f %5.1f%%  (cum %5.1f%%)' % (k_[:60], cnt[k_]//reps, tot[k_], 100*tot[k_]/s_all, 100*acc/s_all), flush=True)


if __name__ == '__main__':
    main()
