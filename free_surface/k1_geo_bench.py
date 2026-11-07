"""geo_pipe.GeoPipe against the geometry call before it (AlphaComplex.candidates -> k_alpha -> structure_gpu ->
geometry_warp.GeometryWarp) on real inputs: a k1 snapshot's sites ST.tx and radii W._ulr (11-06, stage 1; bench only).
Both on the same overlaps. Compared: the candidates, the structure (pairs, links exactly; neighbours as sets; hidden),
the values (relative to each array's largest magnitude); timed with a sync around each call.
python free_surface/k1_geo_bench.py snapshot.pt [more.pt ...]"""
import sys
import time

import torch
import warp as wp

from free_surface.alpha_complex import AlphaComplex
from free_surface.geo_pipe import GeoPipe
from free_surface.geometry_warp import GeometryWarp
from free_surface.k1_alpha_bench import load


def old_call(A, p, r, n):
    out = A(p, r, reuse=True)
    bad = out['status'].clone(); bad[bad == 1] = 0
    if bool((bad > 0).any()):
        raise RuntimeError('status %s' % torch.unique(out['status']).tolist())
    S = A.structure_gpu(out, n)
    return GeometryWarp(p, r, S), S


def rel(a, b):
    a, b = a.double(), b.double()
    return float((a-b).abs().max()/b.abs().max().clamp(min=1e-300))


def bench(path, reps=10, dev='cuda'):
    p, r, step = load(path); n = len(p)
    A = AlphaComplex(n, dev); A.ov_margin = .05
    A.overlaps(p, r, .05)
    # the candidates
    coff, j, nn, h = A.candidates(p, r, reuse=True)
    pipe = GeoPipe(n, dev)
    G1, S1 = pipe(A, p, r)
    c_ok = torch.equal(coff.to(torch.int32), pipe.coff) and torch.equal(j.to(torch.int32), pipe.cj[:len(j)])
    print('%s (step %s): %d cells, %d candidates; candidates: offsets and sites %s, h rel %.1e, n rel %.1e' % (
        path.split('/')[-1], step, n, len(j), 'identical' if c_ok else 'DIFFER', rel(pipe.ch[:len(h)], h) if c_ok else -1, rel(pipe.cn[:len(h)], nn) if c_ok else -1), flush=True)
    G0, S0 = old_call(A, p, r, n)
    G1, S1 = pipe(A, p, r)
    eq = lambda a_, b_: a_.shape == b_.shape and torch.equal(a_.long(), b_.long())
    srt = lambda t_: torch.sort(t_, 1).values
    print('  structure: pairs %s (%d), links %s (width %d / %d), hidden %s, neighbours as sets %s' % (
        'identical' if eq(S0['pairs'], S1['pairs']) else 'DIFFER', len(S0['pairs']), 'identical' if eq(S0['links'], S1['links']) else 'DIFFER',
        S0['links'].shape[1], S1['links'].shape[1], 'identical' if eq(S0['hidden'], S1['hidden']) else 'DIFFER',
        'identical' if eq(srt(S0['nbrs']), srt(S1['nbrs'])) else 'DIFFER'), flush=True)
    if eq(G0.P, G1.P) and eq(G0.Cn, G1.Cn):
        print('  values: pairs and links identical (%d); relative to max: %s' % (len(G0.P), ', '.join('%s %.1e' % (k_, rel(getattr(G1, k_), getattr(G0, k_))) for k_ in
              ('wall_A', 'wall_c', 'patch_A', 'patch_n', 'V', 'rho', 'hi', 'nn', 'pole'))), flush=True)
        nz = (G0.patch_A > 0)
        print('  caps: %d sites; poles identical on %d' % (int(nz.sum()), int((G0.pole[nz] == G1.pole[nz]).all(1).sum())), flush=True)
    else:
        print('  values: the kept pairs DIFFER (%d / %d)' % (len(G0.P), len(G1.P)), flush=True)
    for nm, f in (('before', lambda: old_call(A, p, r, n)), ('GeoPipe', lambda: pipe(A, p, r))):
        f(); torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            f()
        torch.cuda.synchronize()
        print('  %-8s %7.2f ms a call (overlaps kept)' % (nm, 1e3*(time.perf_counter()-t0)/reps), flush=True)


if __name__ == '__main__':
    wp.config.quiet = True
    wp.init()
    for path_ in sys.argv[1:]:
        bench(path_)
