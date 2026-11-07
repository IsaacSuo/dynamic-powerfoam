"""The alpha kernel on real inputs (a k1 snapshot's sites ST.tx and radii W._ulr: the first cut call of that step),
timed per variant and checked output by output against the kernel before 11-06 (11-06, toward a faster cut; bench only).
alpha_variants.k_alpha_v: generated from the kernel before 11-06 (gen_alpha_variants.py in that job's tmp), with a switch:
  mode 0  the kernel before 11-06 (the reference)
  mode 1  clipping only, no walls / links phase (time only: the split)
  mode 2  A: vertex tests in float32 first, float64 only within the rounding bound (must match exactly)
  mode 3  + the vertices' radius bound R2 (the early stop) in float32, rounded up (must match exactly)
  mode 4  + the walls phase computing each face vertex once (the same planes in the same order; must match exactly)
  mode 5  + each plane's vertices counted first, the removal marks written only when it cuts (must match exactly)
python free_surface/k1_alpha_bench.py snapshot.pt [more.pt ...]"""
import sys
import time

import torch
import warp as wp

from free_surface.alpha_complex import AlphaComplex, k_alpha, PMAX, TMAX, WMAX, LMAX
from free_surface.alpha_variants import k_alpha_v


def load(path):
    s = torch.load(path, weights_only=False)
    v = s['values']
    get = lambda name: next(val[1] for k, val in v.items() if '.'.join(str(a[1]) for a in k) == name)
    return get('ST.tx').cuda().double().contiguous(), get('W._ulr').cuda().double().contiguous(), s.get('step')


def bench(path, reps=20, dev='cuda'):
    p, r, step = load(path); n = len(p)
    A = AlphaComplex(n, dev)
    A.overlaps(p, r, .05)
    coff, j, nn, h = A.candidates(p, r, reuse=True)
    wf = lambda t, dt: wp.from_torch(t.contiguous(), dtype=dt)
    ins = [wf(p, wp.vec3d), wf(r, wp.float64), wf(coff.int(), wp.int32), wf(j.int(), wp.int32), wf(nn, wp.vec3d), wf(h, wp.float64)]

    def scratch():
        z = lambda shape, dt: wp.zeros(shape, dtype=dt, device=dev)
        return dict(P=z((n, PMAX), wp.vec4d), Pid=z((n, PMAX), int), T=z((n, TMAX), wp.vec3i), V=z((n, TMAX), wp.vec3d), rem=z((n, TMAX), int),
                    ea=z((n, TMAX), int), eb=z((n, TMAX), int), nxt=z((n, PMAX), int), walls=z((n, WMAX), int), links=z((n, WMAX, LMAX), int),
                    nwall=z(n, int), nlink=z((n, WMAX), int), status=z(n, int), V32=z((n, TMAX), wp.vec3f), ed2=z((n, TMAX), wp.float64))

    def launcher(mode, S):
        sc = [S[k] for k in ('P', 'Pid', 'T', 'V', 'rem', 'ea', 'eb', 'nxt')]; o_ = [S[k] for k in ('walls', 'links', 'nwall', 'nlink', 'status')]
        if mode is None:                                                            # (alpha_complex.k_alpha as it is now)
            return lambda: wp.launch(k_alpha, n, inputs=ins+sc+[S['V32'], S['ed2']]+o_, device=dev)
        return lambda: wp.launch(k_alpha_v, n, inputs=ins+sc+o_+[S['V32'], S['ed2'], mode], device=dev)

    def timeit(fn):
        fn(); wp.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        wp.synchronize_device(dev)
        return 1e3*(time.perf_counter()-t0)/reps

    def out(S):
        T_ = lambda a_: wp.to_torch(a_).clone()
        return {k: T_(S[k]) for k in ('walls', 'links', 'nwall', 'nlink', 'status')}

    def same(a, b):
        if not torch.equal(a['nwall'], b['nwall']) or not torch.equal(a['status'], b['status']):
            return 'nwall/status differ in %d cells' % int(((a['nwall'] != b['nwall']) | (a['status'] != b['status'])).sum())
        m = torch.arange(WMAX, device=a['nwall'].device)[None, :] < a['nwall'][:, None]
        if not torch.equal(a['walls'][m], b['walls'][m]) or not torch.equal(a['nlink'][m], b['nlink'][m]):
            return 'walls/nlink differ'
        ml = m[:, :, None] & (torch.arange(LMAX, device=m.device)[None, None, :] < a['nlink'][:, :, None])
        if not torch.equal(a['links'][ml], b['links'][ml]):
            return 'links differ in %d entries' % int((a['links'][ml] != b['links'][ml]).sum())
        return 'identical'

    deg = (coff[1:]-coff[:-1]).float()
    print('%s (step %s): %d cells, candidates per cell median %.0f max %.0f; status %s' % (path.split('/')[-1], step, n, float(deg.median()), float(deg.max()),
          torch.unique(A(p, r, reuse=True)['status']).tolist()), flush=True)
    S0 = scratch(); f0 = launcher(0, S0); t0 = timeit(f0); O0 = out(S0)
    walls_ = int(O0['nwall'].sum()); links_ = int(O0['nlink'].sum())
    print('  before 11-06 (mode 0)   %7.2f ms   walls %d (pairs %d), links %d' % (t0, walls_, walls_//2, links_), flush=True)
    S = scratch(); f = launcher(None, S); t = timeit(f)
    print('  k_alpha now             %7.2f ms   %s' % (t, same(O0, out(S))), flush=True)
    for mode, name in ((1, 'clipping only (before)'), (2, 'A: float32 tests'), (3, 'A + R2 in float32'), (4, '+ walls: vertex once'), (5, '+ count before marking')):
        S = scratch(); f = launcher(mode, S); t = timeit(f); O = out(S)
        print('  %-23s %7.2f ms   %s' % (name, t, same(O0, O) if mode != 1 else '(no walls phase: time only)'), flush=True)


if __name__ == '__main__':
    wp.init()
    for path_ in sys.argv[1:]:
        bench(path_)
