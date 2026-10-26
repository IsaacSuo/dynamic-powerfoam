"""Would running several simulations at once speed the new model up (2026-10-24)? A product is ~94 small GPU operations
(gradient pass 1.5 of 1.8 ms per CG iteration, product_profile.py), most kernels a few ten thousand threads: if the
GPU idles between and within them, K independent copies on K streams should take much less than K times one. Here K
gradient passes (SkinWarp instances on the same geometry, each its own buffers and graph, the state of a saved step:
skin_drop_check.py --save-state N, the midpoint rule's settings) are replayed K = 1, 2, 4, 8: one after another on one
stream, and each on its own stream at once; time per pass per copy. Prediction: if the GPU idles, 4 at once take ~1.5-2.5
times one (throughput x1.6-2.7); if it is already full, ~4 times.
  python free_surface/batch_premise_check.py --out output/round01_pruned_exact --dt 8.3333332e-4 --tols 1e-4 --beta 0.01 --integrator midpoint --state output/round01_pruned_exact/skin_drop_state_step900.npz"""
import time
from pathlib import Path

import numpy as np
import torch
import warp as wp

from free_surface.skin_contact_check import f64, dev
from free_surface.skin_dynamics_check import build_model, parser
from free_surface.skin_warp import SkinWarp, contact_geometry


def main():
    a = parser()
    a.add_argument('--state', type=Path, required=True)
    a.add_argument('--copies', type=int, nargs='+', default=[1, 2, 4, 8])
    a.add_argument('--reps', type=int, default=50)
    args = a.parse_args()
    b = build_model(args, contact=dict(normal=(0., 1., 0.), c0=0., shift=True))
    st, W0 = b['st'], b['W']
    out = Path(args.out)
    bsk = dict(np.load(out/'boundary_skin.npz')); g = dict(np.load(out/'dem_exact_single'/'geometry.npz'))
    cells = np.load(out/'dem_exact_single'/'cells.npz')
    T64 = lambda a_: torch.as_tensor(a_, dtype=f64, device=dev)
    z0 = np.load(args.state)
    T = lambda k: torch.as_tensor(z0[k], device=dev)

    def make():
        W = SkinWarp(st)
        if W0.vis is not None:
            W.set_viscosity(W0.vis['eta'], W0.vis['zeta'], args.dt)
        cg = contact_geometry(T64(cells['points']), T64(cells['radii']), g['body_of_cell'])
        W.set_contact(cg, (0., 1., 0.), 0., T64(bsk['patch_A']), T64(bsk['ent_a']))
        W.prepare()
        return W

    def state(W):                                                                  # the midpoint rule's settings at the saved step
        W.visc_weight = 2.
        W.total_mode()
        W.tcpar[0], W.tcpar[1], W.tcpar[2] = 2., .5, 1.
        W.tlam.copy_(T('lam')); W.tact.copy_(T('act')); W.tgoff.copy_(T('goff')); W.trho.copy_(T('rho'))
        for k_, r_ in W.tref.items():
            r_.copy_(T('ref_'+k_))
        W.tx.copy_(T('xm1')); W.ty.copy_(T('ym1'))
        wp.capture_launch(W.graph); wp.synchronize()
        return float(W.tloss[0])

    Kmax = max(args.copies)
    t0 = time.perf_counter()
    Ws = [W0]+[make() for _ in range(Kmax-1)]
    loss = [state(W) for W in Ws]
    print('%d gradient instances built in %.1f s; their losses at the state agree: %s (spread %.1e); GPU memory in use %.2f GB' % (
        Kmax, time.perf_counter()-t0, bool(max(loss)-min(loss) <= 1e-12*abs(loss[0])), max(loss)-min(loss), torch.cuda.mem_get_info()[1]/1e9-torch.cuda.mem_get_info()[0]/1e9), flush=True)
    streams = [wp.Stream(dev) for _ in range(Kmax)]

    def run(K, concurrent):
        def once():
            for k in range(K):
                wp.capture_launch(Ws[k].graph, stream=streams[k] if concurrent else None)
        for _ in range(3):
            once()
        wp.synchronize()
        t1 = time.perf_counter()
        for _ in range(args.reps):
            once()
        wp.synchronize()
        return 1e3*(time.perf_counter()-t1)/args.reps

    one = run(1, False)
    print('one gradient pass: %.3f ms' % one, flush=True)
    print('  K   one stream: ms per round, per copy   |  K streams: ms per round, per copy  |  throughput vs one copy', flush=True)
    for K in args.copies:
        s, c = run(K, False), run(K, True)
        print('%3d   %8.3f  %8.3f                 |  %8.3f  %8.3f              |  x%.2f' % (K, s, s/K, c, c/K, K*one/c), flush=True)


if __name__ == '__main__':
    main()
