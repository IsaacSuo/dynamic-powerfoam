"""Where the time of a dynamics step goes (2026-10-23, the per-step skin solve on Warp): (1) each kernel's share of one
gradient (forward and backward, no graph), the graph replay time; (2) a few dynamics steps timed plain, then again with
the gradient calls (split into Hessian-vector products and the rest), rotations, affine skin and preconditioner timed
(synchronised) and counted; the remainder is Python, small torch ops and syncs.
  python free_surface/skin_speed_profile.py --out output/round01_pruned_exact --fem output/round01_exact/fem_modes_h0.008.npz --dt 2.0833333e-4 --steps 20 --tols 1e-4"""
import time
from collections import defaultdict

import torch
import warp as wp

from free_surface.skin_dynamics_check import build, integrate, parser
from free_surface import skin_warp
from free_surface.skin_warp import SkinWarp


def timed(fn, acc, key):
    def f(*a, **k):
        torch.cuda.synchronize(); t = time.perf_counter()
        r = fn(*a, **k)
        torch.cuda.synchronize(); acc[key][0] += time.perf_counter()-t; acc[key][1] += 1
        return r
    return f


def cpu_profile(b, tol, steps, dt, top=30):
    """Python's view of the same steps (cProfile): own time per function; the syncs (item, tolist, float) hold the
    waits for the GPU, the rest is launching"""
    import cProfile
    import pstats
    integrate(b, tol, 3, dt)
    pr = cProfile.Profile()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    pr.enable()
    _, prods = integrate(b, tol, steps, dt)
    torch.cuda.synchronize()
    pr.disable()
    tot = time.perf_counter()-t0
    print('cProfile: %d steps, %.1f ms per step (with profiling overhead), products %s; own time per step, top %d:' % (
        steps, 1e3*tot/steps, prods, top), flush=True)
    st = pstats.Stats(pr)
    rows = sorted(((v[2], v[1], k) for k, v in st.stats.items()), reverse=True)[:top]   # (tottime, ncalls, (file, line, name))
    for tt_, nc, (f, ln, nm) in rows:
        print('  %7.2f ms %7.1f calls   %s:%d %s' % (1e3*tt_/steps, nc/steps, f.split('/')[-1], ln, nm), flush=True)


def main():
    a = parser(); a.add_argument('--cprofile', action='store_true', help='only the cProfile part')
    args = a.parse_args()
    b = build(args)
    if args.cprofile:
        cpu_profile(b, args.tols[0], args.steps, args.dt)
        return
    st, W, S, x0, y0 = b['st'], b['W'], b['S'], b['x0'], b['y0']
    # (1) kernels of one gradient
    W2 = SkinWarp(st); W2.prepare(graph=False)
    W2.grad(x0, y0); torch.cuda.synchronize()
    tot = defaultdict(float)
    with wp.ScopedTimer('grad', cuda_filter=wp.TIMING_KERNEL, synchronize=True, print=False) as t:
        for _ in range(5):
            W2.grad(x0, y0)
    for r in t.timing_results:
        tot[r.name] += r.elapsed/5
    print('kernels of one gradient (no graph), ms:', flush=True)
    for k, v in sorted(tot.items(), key=lambda kv: -kv[1]):
        print('  %-60s %.3f' % (k[:60], v), flush=True)
    print('  sum %.3f ms; sizes: bonds %d, entries %d, pairs %d, patches %d, bodies %d, skin points %d' % (
        sum(tot.values()), W.nb, W.ne, W.npairs, W.nc, W.n, W.npt), flush=True)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(50):
        W.grad(x0, y0)
    torch.cuda.synchronize()
    print('graph replay incl. copies: %.3f ms per gradient' % (1e3*(time.perf_counter()-t0)/50), flush=True)
    # (2) dynamics steps
    tol = args.tols[0]
    integrate(b, tol, 3, args.dt)                                                  # warm-up
    torch.cuda.synchronize(); t0 = time.perf_counter()
    _, prods = integrate(b, tol, args.steps, args.dt)
    torch.cuda.synchronize(); plain = time.perf_counter()-t0
    print('plain: %d steps at tol %.0e, %.1f ms per step, products per step %s' % (args.steps, tol, 1e3*plain/args.steps, prods), flush=True)
    acc = defaultdict(lambda: [0., 0])
    g0, h0 = W.grad, W.hvp_fd
    W.grad = timed(g0, acc, 'gradient (other)')
    def hvp(x, y, V, h, g=None):                                                 # its gradients are counted apart
        W.grad = timed(g0, acc, 'gradient (in products)')
        try:
            return h0(x, y, V, h, g)
        finally:
            W.grad = timed(g0, acc, 'gradient (other)')
    W.hvp_fd = hvp
    S.rotations = timed(S.rotations, acc, 'rotations (torch)')
    skin_warp.WarpCG.step = timed(skin_warp.WarpCG.step, acc, 'CG iteration graph + read')
    S.steihaug = timed(S.steihaug, acc, 'steihaug (whole, incl. CG)')
    S.affine_skin = timed(S.affine_skin, acc, 'affine skin (torch)')
    make = b['prec']
    def make_t(Rb):
        f = make(Rb)
        ft = timed(f, acc, 'preconditioner apply')
        if hasattr(f, 'wprec'):                                                  # keeps the graph path
            ft.wprec = f.wprec
        return ft
    b2 = dict(b, prec=timed(make_t, acc, 'preconditioner make'))
    torch.cuda.synchronize(); t0 = time.perf_counter()
    _, prods = integrate(b2, tol, args.steps, args.dt)
    torch.cuda.synchronize(); inst = time.perf_counter()-t0
    print('instrumented: %.1f ms per step, products per step %s' % (1e3*inst/args.steps, prods), flush=True)
    for k, (tm, c) in sorted(acc.items(), key=lambda kv: -kv[1][0]):
        print('  %-28s %6.1f ms per step, %5.1f calls per step, %.3f ms per call' % (k, 1e3*tm/args.steps, c/args.steps, 1e3*tm/max(c, 1)), flush=True)
    nested = ('CG iteration graph + read', 'preconditioner apply', 'gradient (in products)')                     # inside steihaug
    rest = inst-sum(tm for k_, (tm, _) in acc.items() if k_ not in nested)
    print('  %-28s %6.1f ms per step' % ('rest (python, torch, syncs)', 1e3*rest/args.steps), flush=True)


if __name__ == '__main__':
    main()
