"""alpha_coop (tier 1 + tier 2) vs alpha_complex.k_alpha (the worktree's, B): walls, links, nwall, nlink, status identical; times."""
import sys, time
import numpy as np, torch, warp as wp
WT = '/home/fangsuo/py/dynamic-powerfoam/.claude/worktrees/solid-fv-ale'  # (its frames; the kernels from the cwd's free_surface)
sys.path.insert(0, '.')  # run from the worktree root: python free_surface/alpha_coop_check.py [all|frame]
import free_surface.alpha_complex as AC
import free_surface.alpha_coop as C
(KC, WPB_), (K2, WPB2) = C.make_pair()
wp.init(); dev = 'cuda'
which = sys.argv[1] if len(sys.argv) > 1 else 'all'
def states():
    z = np.load(WT+'/output/ale/gravity_drop_adaptive/frames_gravity.npz')
    ks = range(len(z['t'])) if which == 'all' else [int(which)]
    for k in ks:
        for m in ((0., 4e-5, 4e-4) if which == 'all' else (0.,)):
            yield 'ale f%02d +%.0e' % (k, m), z['x'][k]+z['v'][k]*m, z['r'][k]
    if which == 'all':
        z = np.load('/home/fangsuo/py/dynamic-powerfoam/output/discrete_plasticity/batch/plasticine0/frames_plasticine0_kry.npz')
        for k in range(0, len(z['t']), 4):
            yield 'k1 p0 f%02d' % k, z['x'][k], z['r'][k]
def timeit(f, reps=20):
    f(); wp.synchronize(); ts = []
    for _ in range(reps):
        wp.synchronize(); t0 = time.perf_counter(); f(); wp.synchronize(); ts.append(time.perf_counter()-t0)
    return 1e3*float(np.median(ts))
cache = {}; bad = 0; cnt = 0; T = [0., 0.]
for name, x, r in states():
    n = len(x)
    if n not in cache:
        cache[n] = (AC.AlphaComplex(n, dev), AC.AlphaComplex(n, dev))
    A, B = cache[n]
    p = torch.as_tensor(x, device=dev, dtype=torch.float64); rr = torch.as_tensor(r, device=dev, dtype=torch.float64)
    coff, j, nn, h = A.candidates(p, rr, False)
    wf = lambda q, dt: wp.from_torch(q.contiguous(), dtype=dt)
    c = [wf(p, wp.vec3d), wf(rr, wp.float64), wf(coff.int(), wp.int32), wf(j.int(), wp.int32), wf(nn, wp.vec3d), wf(h, wp.float64)]
    keys = ('walls', 'links', 'nwall', 'nlink', 'status')
    def orig():
        wp.launch(AC.k_alpha, n, inputs=c+[A.P, A.Pid, A.T, A.V, A.rem, A.ea, A.eb, A.nxt, A.V32, A.ed2, A.walls, A.links, A.nwall, A.nlink, A.status])
    def coop():
        wp.launch(KC, n*32, inputs=c[1:]+[B.walls, B.links, B.nwall, B.nlink, B.status, n, AC.WMAX, AC.LMAX], block_dim=32*WPB_)
    def fb():
        wp.launch(K2, n*32, inputs=c[1:]+[B.walls, B.links, B.nwall, B.nlink, B.status, n, AC.WMAX, AC.LMAX], block_dim=32*WPB2)
    for S in (A, B):
        for k in keys: getattr(S, k).zero_()
    orig(); coop(); wp.synchronize()
    n7 = int((wp.to_torch(B.status) == 7).sum())
    fb(); wp.synchronize()
    o = {k: wp.to_torch(getattr(A, k)).clone() for k in keys}; q = {k: wp.to_torch(getattr(B, k)).clone() for k in keys}
    diff = [k for k in keys if not torch.equal(o[k], q[k])]
    if diff:
        bad += 1
        cells = torch.zeros(n, dtype=torch.bool, device=dev)
        for k in diff:
            d = (o[k] != q[k]); cells |= d.reshape(n, -1).any(1)
        ids = torch.nonzero(cells).flatten()[:5].tolist()
        info = 'DIFF %s cells %d e.g. %s status o/c %s/%s nwall %s/%s' % (','.join(diff), int(cells.sum()), ids, o['status'][ids].tolist(), q['status'][ids].tolist(), o['nwall'][ids].tolist(), q['nwall'][ids].tolist())
    else:
        info = 'IDENTICAL'
    cnt += 1
    to = timeit(orig); tc = timeit(coop); tcf = timeit(lambda: (coop(), fb()))
    T[0] += to; T[1] += tcf
    print('%-15s n %d | orig %.2f ms | coop %.2f ms, +tier2 %.2f ms (tier-2 cells %d) | %s' % (name, n, to, tc, tcf, n7, info), flush=True)
print('cases %d differing %d; orig total %.1f ms, coop+tier2 total %.1f ms' % (cnt, bad, *T))
