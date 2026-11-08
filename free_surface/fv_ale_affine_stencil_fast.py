"""fv_ale_affine_stencil.build + compact on the GPU in three Warp kernels (11-08), for the stencils that need no growth.

The reference (torch) builds the directed group-pair keys by a global unique/sort, scatters them into rows, forms the
normal matrices three times and the rows twice when nothing grows, and reads its summary back with several syncs:
about 7 ms a call, 2.2 calls a step. Here:
  k_count / k_fill  one thread a wall (g_i != g_j, A > 0): each group's (neighbour group, area) list, both directions
  k_rows            one thread a group: its list sorted by the neighbour (insertion), equal neighbours merged (their
                    areas summed), the weights A / |d|^2 (d between the group centroids), the normal matrix, its
                    eigen-decomposition (fv_ale_symmetric_eigen.jacobi3's arithmetic, as a function), the ratio, the
                    pseudo-inverse (good: ratio >= minimum and the largest eigenvalue > 0), the summary by atomics
  k_edges           one thread a group: the directed supports in the reference's order (owner, then neighbour)
One read back: whether any root would grow (ratio < transition_ratio: its stencil gains supports through faces,
fv_ale_affine_stencil.build's loop). Then the whole call is the reference build + compact, unchanged; otherwise the
reference's no-growth result, to rounding (the sums' order: the merged areas, the normal matrix).
build_compact(b, capacity) -> (values, summary, capacity, count) as build + compact give them."""
import torch
import warp as wp

from free_surface.fv_ale_reconstruction import group_centroids


@wp.func
def jacobi3_f(A: wp.mat33d):
    """fv_ale_symmetric_eigen.jacobi3 for one matrix (the same arithmetic): ascending eigenvalues, the vectors as columns"""
    M = A
    R = wp.identity(n=3, dtype=wp.float64)
    for sweep in range(10):
        for pair in range(3):
            p = int(0); q = int(1)
            if pair == 1:
                p = 0; q = 2
            elif pair == 2:
                p = 1; q = 2
            off = M[p, q]
            threshold = wp.float64(1e-16)*(wp.abs(M[p, p])+wp.abs(M[q, q]))+wp.float64(1e-30)
            if wp.abs(off) > threshold:
                tau = (M[q, q]-M[p, p])/(wp.float64(2.)*off)
                sign = wp.float64(1.)
                if tau < wp.float64(0.):
                    sign = wp.float64(-1.)
                t = sign/(wp.abs(tau)+wp.sqrt(wp.float64(1.)+tau*tau))
                c = wp.float64(1.)/wp.sqrt(wp.float64(1.)+t*t)
                s = t*c
                pp = M[p, p]; qq = M[q, q]
                M[p, p] = pp-t*off; M[q, q] = qq+t*off
                M[p, q] = wp.float64(0.); M[q, p] = wp.float64(0.)
                for k in range(3):
                    if k != p and k != q:
                        kp = M[k, p]; kq = M[k, q]
                        M[k, p] = c*kp-s*kq; M[p, k] = M[k, p]
                        M[k, q] = s*kp+c*kq; M[q, k] = M[k, q]
                    rp = R[k, p]; rq = R[k, q]
                    R[k, p] = c*rp-s*rq; R[k, q] = s*rp+c*rq
    d = wp.vec3d(M[0, 0], M[1, 1], M[2, 2])
    for p in range(2):
        for q in range(p+1, 3):
            if d[q] < d[p]:
                old = d[p]; d[p] = d[q]; d[q] = old
                for k in range(3):
                    old_r = R[k, p]; R[k, p] = R[k, q]; R[k, q] = old_r
    return d, R


@wp.kernel(enable_backward=False)
def k_count(i: wp.array(dtype=wp.int64), j: wp.array(dtype=wp.int64), A: wp.array(dtype=wp.float64), g: wp.array(dtype=wp.int64),
            cnt: wp.array(dtype=wp.int32)):
    w = wp.tid()
    gi = g[i[w]]
    gj = g[j[w]]
    if gi != gj and A[w] > wp.float64(0.0):
        wp.atomic_add(cnt, int(gi), 1)
        wp.atomic_add(cnt, int(gj), 1)


@wp.kernel(enable_backward=False)
def k_fill(i: wp.array(dtype=wp.int64), j: wp.array(dtype=wp.int64), A: wp.array(dtype=wp.float64), g: wp.array(dtype=wp.int64),
           off: wp.array(dtype=wp.int32), cur: wp.array(dtype=wp.int32), nb: wp.array(dtype=wp.int64), ar: wp.array(dtype=wp.float64)):
    w = wp.tid()
    gi = g[i[w]]
    gj = g[j[w]]
    if gi != gj and A[w] > wp.float64(0.0):
        a = wp.atomic_add(cur, int(gi), 1)
        nb[off[int(gi)]+a] = gj
        ar[off[int(gi)]+a] = A[w]
        b = wp.atomic_add(cur, int(gj), 1)
        nb[off[int(gj)]+b] = gi
        ar[off[int(gj)]+b] = A[w]


@wp.kernel(enable_backward=False)
def k_rows(off: wp.array(dtype=wp.int32), nb: wp.array(dtype=wp.int64), ar: wp.array(dtype=wp.float64), centers: wp.array(dtype=wp.vec3d),
           g: wp.array(dtype=wp.int64), minimum_ratio: wp.float64, transition_ratio: wp.float64,
           rowlen: wp.array(dtype=wp.int32), wt: wp.array(dtype=wp.float64), inv: wp.array(dtype=wp.mat33d), stats: wp.array(dtype=wp.int32),
           min_ratio: wp.array(dtype=wp.float64)):
    """stats: 0 roots that would grow (ratio < transition), 1 bad roots (ratio < minimum or no positive eigenvalue),
    2 fallback roots (not good), 3 the longest row, 4 the supports"""
    o = wp.tid()
    a = off[o]
    b = off[o+1]
    for t in range(a+1, b):
        x = nb[t]
        y = ar[t]
        u = int(t)
        while u > a:
            if nb[u-1] > x:
                nb[u] = nb[u-1]
                ar[u] = ar[u-1]
                u -= 1
            else:
                break
        nb[u] = x
        ar[u] = y
    m = int(0)
    for t in range(a, b):
        if m > 0 and nb[a+m-1] == nb[t]:
            ar[a+m-1] = ar[a+m-1]+ar[t]
        else:
            nb[a+m] = nb[t]
            ar[a+m] = ar[t]
            m += 1
    rowlen[o] = m
    co = centers[o]
    N = wp.mat33d()
    for t in range(a, a+m):
        d = centers[int(nb[t])]-co
        w = ar[t]/wp.max(wp.dot(d, d), wp.float64(1e-300))
        wt[t] = w
        N = N+w*wp.outer(d, d)
    eig, R = jacobi3_f(N)
    ratio = eig[0]/wp.max(eig[2], wp.float64(1e-300))
    good = ratio >= minimum_ratio and eig[2] > wp.float64(0.0)
    D = wp.mat33d()
    if good:
        D = wp.diag(wp.vec3d(wp.float64(1.0)/wp.max(eig[0], wp.float64(1e-300)), wp.float64(1.0)/wp.max(eig[1], wp.float64(1e-300)),
                             wp.float64(1.0)/wp.max(eig[2], wp.float64(1e-300))))
    inv[o] = R@D@wp.transpose(R)
    root = g[o] == wp.int64(o)
    if root:
        if ratio < transition_ratio:
            wp.atomic_add(stats, 0, 1)
        if ratio < minimum_ratio or eig[2] <= wp.float64(0.0):
            wp.atomic_add(stats, 1, 1)
        if not good:
            wp.atomic_add(stats, 2, 1)
        wp.atomic_min(min_ratio, 0, ratio)
    wp.atomic_max(stats, 3, m)
    wp.atomic_add(stats, 4, m)


@wp.kernel(enable_backward=False)
def k_edges(off: wp.array(dtype=wp.int32), eoff: wp.array(dtype=wp.int64), rowlen: wp.array(dtype=wp.int32), nb: wp.array(dtype=wp.int64),
            wt: wp.array(dtype=wp.float64), owner: wp.array(dtype=wp.int64), neighbor: wp.array(dtype=wp.int64), weight: wp.array(dtype=wp.float64)):
    o = wp.tid()
    a = off[o]
    e = int(eoff[o])
    for t in range(rowlen[o]):
        owner[e+t] = wp.int64(o)
        neighbor[e+t] = nb[a+t]
        weight[e+t] = wt[a+t]


@torch.no_grad()
def build_compact(b, capacity=0, minimum_ratio=1e-3, transition_ratio=.01):
    """fv_ale_affine_stencil.build(b, minimum_ratio, transition_ratio=...) followed by compact(values, capacity)"""
    g = b['groups']; n = len(g); dev = g.device
    i, j, A = b['i'].long().contiguous(), b['j'].long().contiguous(), b['A'].contiguous()
    W = lambda t_, d_: wp.from_torch(t_.contiguous(), dtype=d_)
    gl = g.long().contiguous()
    centers = group_centroids(b).contiguous()
    cnt = torch.zeros(n, dtype=torch.int32, device=dev)
    if len(i):
        wp.launch(k_count, len(i), inputs=[W(i, wp.int64), W(j, wp.int64), W(A, wp.float64), W(gl, wp.int64)], outputs=[W(cnt, wp.int32)], device=str(dev))
    off = torch.zeros(n+1, dtype=torch.int32, device=dev); torch.cumsum(cnt, 0, dtype=torch.int32, out=off[1:])
    total = 2*len(i)                                                               # (an upper bound: no read)
    nb = torch.empty(max(total, 1), dtype=torch.long, device=dev); ar = torch.empty(max(total, 1), dtype=A.dtype, device=dev)
    cur = torch.zeros(n, dtype=torch.int32, device=dev)
    if len(i):
        wp.launch(k_fill, len(i), inputs=[W(i, wp.int64), W(j, wp.int64), W(A, wp.float64), W(gl, wp.int64), W(off, wp.int32), W(cur, wp.int32)],
                  outputs=[W(nb, wp.int64), W(ar, wp.float64)], device=str(dev))
    rowlen = torch.empty(n, dtype=torch.int32, device=dev); wt = torch.empty_like(ar)
    inv = torch.empty(n, 3, 3, dtype=A.dtype, device=dev)
    stats = torch.zeros(5, dtype=torch.int32, device=dev); mr = torch.full((1,), float('inf'), dtype=A.dtype, device=dev)
    wp.launch(k_rows, n, inputs=[W(off, wp.int32), W(nb, wp.int64), W(ar, wp.float64), W(centers, wp.vec3d), W(gl, wp.int64), float(minimum_ratio), float(transition_ratio)],
              outputs=[W(rowlen, wp.int32), W(wt, wp.float64), W(inv, wp.mat33d), W(stats, wp.int32), W(mr, wp.float64)], device=str(dev))
    grow, bad, fallback, width, count = stats.tolist(); min_root = float(mr)      # (the one read)
    if grow:
        from free_surface.fv_ale_affine_stencil import build, compact
        values, summary = build(b, minimum_ratio=minimum_ratio, transition_ratio=transition_ratio)
        values, capacity, count = compact(values, capacity)
        build_compact.reference_calls += 1
        return values, summary, capacity, count
    capacity = max(capacity, ((count+32767)//32768)*32768)
    eoff = torch.zeros(n, dtype=torch.long, device=dev); torch.cumsum(rowlen[:-1].long(), 0, out=eoff[1:])
    owner = torch.zeros(capacity, dtype=torch.long, device=dev); neighbor = torch.zeros(capacity, dtype=torch.long, device=dev)
    weight = torch.zeros(capacity, dtype=A.dtype, device=dev)
    wp.launch(k_edges, n, inputs=[W(off, wp.int32), W(eoff, wp.int64), W(rowlen, wp.int32), W(nb, wp.int64), W(wt, wp.float64)],
              outputs=[W(owner, wp.int64), W(neighbor, wp.int64), W(weight, wp.float64)], device=str(dev))
    values = {'affine_centers': centers, 'affine_inverse': inv, 'affine_owner': owner, 'affine_neighbor': neighbor, 'affine_edge_weight': weight}
    summary = {'initial_bad_roots': bad, 'bad_roots_by_growth': [bad], 'fallback_roots': fallback, 'extra_directed_supports': 0,
               'width': max(16, ((width+15)//16)*16), 'minimum_root_ratio': min_root, 'minimum_ratio_required': minimum_ratio,
               'transition_ratio': transition_ratio, 'transition_roots': 0, 'fully_extended_roots': 0}
    return values, summary, capacity, count


build_compact.reference_calls = 0
