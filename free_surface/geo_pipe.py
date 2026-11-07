"""One cut's geometry call on the GPU with one host read (11-06, stage 1 of the explicit step's restructuring; the same
cut as Recut.geometry's AlphaComplex -> structure_gpu -> geometry_warp.GeometryWarp, to rounding):
  candidates  one thread per cell over its BVH overlaps (the kept CSR, AlphaComplex.overlaps): h = (L^2 + r_i^2 -
              r_j^2) / (2 L) (L = 0: never kept), all of the cell's overlaps kept sorted by (h, the overlap's position)
              from call to call: a heap sort when the overlaps are found again, else h again in the last order and an
              insertion sort (h moves little between calls: nearly sorted); the kept, h < r_i, are the sorted prefix
              (AlphaComplex.candidates: the same order by two global stable sorts), copied compact with j, n, h
  clip        alpha_complex.k_alpha, unchanged
  structure   each cell's walls to larger sites, sorted by the site (structure_gpu's key order i n + j), their links
              (-1 padded), the circle's existence (GeometryWarp's keep: L > 0, r_i^2 - h^2 > 0), the kept pairs
              compacted; each site's entries (side i: its own kept pairs in order; side j: atomic cursors) and its
              neighbours (atomic cursors; the exposure test reads them as a set)
  one read    the pair counts, the links' width, the largest degree, the status flags, the overflow flag
  values      geometry_warp.k_geo_pair, k_geo_site, unchanged; outputs on torch's caching allocator (each call owns
              its arrays: the solve keeps the start's geometry beside its trials)
Summation orders differ (atomic cursors, as k_geo_pair's volumes already do): values agree to rounding.
GeoPipe(n)(A, p, r) -> (G, S) as Recut.geometry returns them; A: the AlphaComplex holding the overlaps and the clip's
storage. timer: a callable(label) for the profiler (synchronised marks), None in runs."""
import math

import torch
import warp as wp

from free_surface.alpha_complex import LMAX, WMAX, k_alpha
from free_surface.geometry_warp import BIG, NCAND, TWO_PI, k_geo_pair, k_geo_site

f64, i32 = torch.float64, torch.int32


@wp.func
def hless(hk: wp.array(dtype=wp.float64), ek: wp.array(dtype=wp.int32), a: int, b: int):
    """(h, position) at a before the one at b"""
    return hk[a] < hk[b] or (hk[a] == hk[b] and ek[a] < ek[b])


@wp.func
def hswap(hk: wp.array(dtype=wp.float64), ek: wp.array(dtype=wp.int32), a: int, b: int):
    th = hk[a]
    te = ek[a]
    hk[a] = hk[b]
    ek[a] = ek[b]
    hk[b] = th
    ek[b] = te


@wp.func
def hsift(hk: wp.array(dtype=wp.float64), ek: wp.array(dtype=wp.int32), base: int, start: int, last: int):
    root = int(start)
    while 2*root+1 <= last:
        child = 2*root+1
        sw = int(root)
        if hless(hk, ek, base+sw, base+child):
            sw = child
        if child+1 <= last:
            if hless(hk, ek, base+sw, base+child+1):
                sw = child+1
        if sw == root:
            break
        hswap(hk, ek, base+root, base+sw)
        root = sw


@wp.func
def hval(p: wp.array(dtype=wp.vec3d), r: wp.array(dtype=wp.float64), i: int, j: int, big: wp.float64):
    d = p[j]-p[i]
    L = wp.length(d)
    h = big
    if L > wp.float64(0.0):
        h = (L*L+r[i]*r[i]-r[j]*r[j])/(wp.float64(2.0)*L)
    return h


@wp.kernel(enable_backward=False)
def k_cand_full(p: wp.array(dtype=wp.vec3d), r: wp.array(dtype=wp.float64), off: wp.array(dtype=wp.int32), adj: wp.array(dtype=wp.int32), big: wp.float64,
                hk: wp.array(dtype=wp.float64), ek: wp.array(dtype=wp.int32)):
    """the overlaps found again: each cell's whole segment sorted by (h, position), a heap sort"""
    i = wp.tid()
    a = off[i]
    b = off[i+1]
    for e in range(a, b):
        ek[e] = e
        hk[e] = hval(p, r, i, adj[e], big)
    cnt = b-a
    if cnt > 1:
        for s in range(cnt//2-1, -1, -1):
            hsift(hk, ek, a, s, cnt-1)
        for t in range(cnt-1, 0, -1):
            hswap(hk, ek, a, a+t)
            hsift(hk, ek, a, 0, t-1)


@wp.kernel(enable_backward=False)
def k_hval(p: wp.array(dtype=wp.vec3d), r: wp.array(dtype=wp.float64), cell: wp.array(dtype=wp.int32), adj: wp.array(dtype=wp.int32), big: wp.float64,
           ek: wp.array(dtype=wp.int32), hk: wp.array(dtype=wp.float64)):
    """each call, one thread per overlap: h again at its place in the kept order"""
    t = wp.tid()
    hk[t] = hval(p, r, cell[t], adj[ek[t]], big)


@wp.kernel(enable_backward=False)
def k_cand(r: wp.array(dtype=wp.float64), off: wp.array(dtype=wp.int32), hk: wp.array(dtype=wp.float64), ek: wp.array(dtype=wp.int32), kc: wp.array(dtype=wp.int32)):
    """then per cell: an insertion sort by (h, position) (nearly sorted), the kept prefix h < r_i"""
    i = wp.tid()
    a = off[i]
    b = off[i+1]
    for t in range(a+1, b):
        x = hk[t]
        y = ek[t]
        u = int(t)
        while u > a:
            if hk[u-1] > x or (hk[u-1] == x and ek[u-1] > y):
                hk[u] = hk[u-1]
                ek[u] = ek[u-1]
                u -= 1
            else:
                break
        hk[u] = x
        ek[u] = y
    ri = r[i]
    cnt = int(0)
    for t in range(a, b):
        if hk[t] < ri:
            cnt += 1
        else:
            break
    kc[i] = cnt


@wp.kernel(enable_backward=False)
def k_cand_out(p: wp.array(dtype=wp.vec3d), cell: wp.array(dtype=wp.int32), off: wp.array(dtype=wp.int32), adj: wp.array(dtype=wp.int32), hk: wp.array(dtype=wp.float64),
               ek: wp.array(dtype=wp.int32), kc: wp.array(dtype=wp.int32), coff: wp.array(dtype=wp.int32),
               cj: wp.array(dtype=wp.int32), cn: wp.array(dtype=wp.vec3d), ch: wp.array(dtype=wp.float64)):
    """one thread per overlap: the kept prefix copied compact"""
    t = wp.tid()
    i = cell[t]
    u = t-off[i]
    if u >= kc[i]:
        return
    o = coff[i]+u
    j = adj[ek[t]]
    d = p[j]-p[i]
    cj[o] = j
    cn[o] = d/wp.length(d)
    ch[o] = hk[t]


@wp.kernel(enable_backward=False)
def k_pair_count(walls: wp.array2d(dtype=int), nwall: wp.array(dtype=int), nlink: wp.array2d(dtype=int), status: wp.array(dtype=int),
                 pc: wp.array(dtype=wp.int32), cnt: wp.array(dtype=wp.int32)):
    """cnt: 2 the links' width, 4 the largest bad status (> 1)"""
    i = wp.tid()
    st = status[i]
    if st > 1:
        wp.atomic_max(cnt, 4, st)
    c = int(0)
    km = int(0)
    for s in range(nwall[i]):
        if walls[i, s] > i:
            c += 1
            km = wp.max(km, nlink[i, s])
    pc[i] = c
    wp.atomic_max(cnt, 2, km)


@wp.kernel(enable_backward=False)
def k_pair_fill(p: wp.array(dtype=wp.vec3d), r: wp.array(dtype=wp.float64), walls: wp.array2d(dtype=int), links: wp.array3d(dtype=int),
                nwall: wp.array(dtype=int), nlink: wp.array2d(dtype=int), poff: wp.array(dtype=wp.int32), cap: int, ordr: wp.array2d(dtype=wp.int32),
                pairs: wp.array2d(dtype=wp.int32), plinks: wp.array2d(dtype=wp.int32), keep: wp.array(dtype=wp.int32), deg: wp.array(dtype=wp.int32),
                cnt: wp.array(dtype=wp.int32)):
    """each cell's walls to larger sites by the site (insertion sort of the slots), written at poff[i]; cnt 5: overflow"""
    i = wp.tid()
    c = int(0)
    for s in range(nwall[i]):
        j = walls[i, s]
        if j > i:
            b = int(c)
            while b > 0:
                if walls[i, ordr[i, b-1]] > j:
                    ordr[i, b] = ordr[i, b-1]
                    b -= 1
                else:
                    break
            ordr[i, b] = s
            c += 1
    o = poff[i]
    zero = wp.float64(0.0)
    for t in range(c):
        q = o+t
        if q >= cap:
            wp.atomic_max(cnt, 5, 1)
            break
        s = ordr[i, t]
        j = walls[i, s]
        pairs[q, 0] = i
        pairs[q, 1] = j
        nl = nlink[i, s]
        for m in range(LMAX):
            if m < nl:
                plinks[q, m] = links[i, s, m]
            else:
                plinks[q, m] = -1
        L0 = wp.length(p[j]-p[i])
        kp = int(0)
        if L0 > zero:
            h0 = (L0*L0+r[i]*r[i]-r[j]*r[j])/(wp.float64(2.0)*L0)
            if r[i]*r[i]-h0*h0 > zero:
                kp = 1
        keep[q] = kp
        wp.atomic_add(deg, i, 1)
        wp.atomic_add(deg, j, 1)


@wp.kernel(enable_backward=False)
def k_pair_compact(pairs: wp.array2d(dtype=wp.int32), plinks: wp.array2d(dtype=wp.int32), keep: wp.array(dtype=wp.int32), kscan: wp.array(dtype=wp.int32),
                   poff: wp.array(dtype=wp.int32), n: int, P: wp.array2d(dtype=wp.int32), Cn: wp.array2d(dtype=wp.int32),
                   ci: wp.array(dtype=wp.int32), cjn: wp.array(dtype=wp.int32)):
    """the kept pairs in order (kscan inclusive); per site its kept pairs as the smaller (ci) and the larger (cjn) site"""
    q = wp.tid()
    if q >= poff[n]:
        return
    if keep[q] == 0:
        return
    f = kscan[q]-1
    i = pairs[q, 0]
    j = pairs[q, 1]
    P[f, 0] = i
    P[f, 1] = j
    for m in range(LMAX):
        Cn[f, m] = plinks[q, m]
    wp.atomic_add(ci, i, 1)
    wp.atomic_add(cjn, j, 1)


@wp.kernel(enable_backward=False)
def k_ent_i(poff: wp.array(dtype=wp.int32), kscan: wp.array(dtype=wp.int32), keep: wp.array(dtype=wp.int32), ci: wp.array(dtype=wp.int32),
            soff: wp.array(dtype=wp.int32), ent: wp.array(dtype=wp.int32)):
    """side i: the site's own kept pairs, in order (first entries: the pair's index)"""
    s = wp.tid()
    a = poff[s]
    base = int(0)
    if a > 0:
        base = kscan[a-1]
    o = soff[s]
    for t in range(ci[s]):
        ent[o+t] = base+t


@wp.kernel(enable_backward=False)
def k_ent_j(P: wp.array2d(dtype=wp.int32), ci: wp.array(dtype=wp.int32), soff: wp.array(dtype=wp.int32), npr: int,
            cur: wp.array(dtype=wp.int32), ent: wp.array(dtype=wp.int32)):
    """side j: npr + the pair's index, after the site's own (atomic cursors)"""
    f = wp.tid()
    j = P[f, 1]
    k = wp.atomic_add(cur, j, 1)
    ent[soff[j]+ci[j]+k] = npr+f


@wp.kernel(enable_backward=False)
def k_nbrs(pairs: wp.array2d(dtype=wp.int32), poff: wp.array(dtype=wp.int32), n: int, cur: wp.array(dtype=wp.int32), nbrs: wp.array2d(dtype=wp.int32)):
    q = wp.tid()
    if q >= poff[n]:
        return
    i = pairs[q, 0]
    j = pairs[q, 1]
    nbrs[i, wp.atomic_add(cur, i, 1)] = j
    nbrs[j, wp.atomic_add(cur, j, 1)] = i


class PipeGeometry:
    """the values Recut and the run read, as GeometryWarp names them (torch arrays owned by this call)"""
    pass


class GeoPipe:
    _fib = {}

    def __init__(self, n, dev='cuda', piece=math.pi/6, npole=32, cap_pairs=None):
        self.n, self.dev, self.piece, self.npole = n, dev, piece, npole
        self.cap = int(cap_pairs or 10*n)
        Z = lambda *s_, dt=i32: torch.zeros(*s_, dtype=dt, device=dev)
        self.kc, self.coff, self.pc, self.poff = Z(n), Z(n+1), Z(n), Z(n+1)
        self.ordr = Z(n, WMAX)
        self.cnt = Z(8)
        self.ncap = 0
        self.timer = None
        self.sorted_key = None
        assert npole+2 == NCAND
        key = (str(dev), npole)
        if key not in GeoPipe._fib:
            k_ = torch.arange(npole, dtype=f64, device=dev)+.5
            ph = torch.acos(1-2*k_/npole); th = math.pi*(1+5**.5)*k_
            GeoPipe._fib[key] = wp.from_torch(torch.stack((torch.cos(th)*torch.sin(ph), torch.sin(th)*torch.sin(ph), torch.cos(ph)), 1).contiguous(), dtype=wp.vec3d)

    def _mark(self, label):
        if self.timer is not None:
            self.timer(label)

    def _cand_storage(self, total):
        if total > self.ncap:
            self.sorted_key = None
            self.ncap = int(total*1.1)+1024
            Z = lambda dt: torch.zeros(self.ncap, dtype=dt, device=self.dev)
            self.hk, self.ek, self.cj, self.ch = Z(f64), Z(i32), Z(i32), Z(f64)
            self.cn = torch.zeros(self.ncap, 3, dtype=f64, device=self.dev)

    def __call__(self, A, p, r):
        n, dev = self.n, self.dev
        W_ = lambda t_, d_: wp.from_torch(t_, dtype=d_)
        p = p.contiguous(); r = r.contiguous()
        pw, rw = W_(p, wp.vec3d), W_(r, wp.float64)
        self._mark('start')
        # the overlaps, kept while they cover (AlphaComplex.covers / overlaps)
        if not A.covers(p, r):
            A.overlaps(p, r, A.ov_margin)
        if getattr(A, 'ov_csr', None) is None or A.ov_csr[0] is not A.ov_key:
            off_, adj_ = A.ov_off.to(i32).contiguous(), A.ov_adj.to(i32).contiguous()
            cell_ = torch.repeat_interleave(torch.arange(n, dtype=i32, device=dev), (off_[1:]-off_[:-1]).long(), output_size=len(adj_))
            A.ov_csr = (A.ov_key, off_, adj_, cell_)
        _, off_, adj_, cell_ = A.ov_csr
        self._cand_storage(len(adj_))
        offw, adjw = W_(off_, wp.int32), W_(adj_, wp.int32)
        hkw, ekw = W_(self.hk, wp.float64), W_(self.ek, wp.int32)
        self._mark('overlaps')
        # candidates
        if self.sorted_key is not A.ov_key:                                          # (the overlaps found again: the full sort)
            wp.launch(k_cand_full, n, inputs=[pw, rw, offw, adjw, BIG], outputs=[hkw, ekw], device=dev)
            self.sorted_key = A.ov_key
        if len(adj_):
            wp.launch(k_hval, len(adj_), inputs=[pw, rw, W_(cell_, wp.int32), adjw, BIG, ekw], outputs=[hkw], device=dev)
        wp.launch(k_cand, n, inputs=[rw, offw, hkw, ekw], outputs=[W_(self.kc, wp.int32)], device=dev)
        torch.cumsum(self.kc, 0, dtype=i32, out=self.coff[1:])
        if len(adj_):
            wp.launch(k_cand_out, len(adj_), inputs=[pw, W_(cell_, wp.int32), offw, adjw, hkw, ekw, W_(self.kc, wp.int32), W_(self.coff, wp.int32)],
                      outputs=[W_(self.cj, wp.int32), W_(self.cn, wp.vec3d), W_(self.ch, wp.float64)], device=dev)
        self._mark('candidates')
        # clip
        wp.launch(k_alpha, n, inputs=[pw, rw, W_(self.coff, wp.int32), W_(self.cj, wp.int32), W_(self.cn, wp.vec3d), W_(self.ch, wp.float64),
                                      A.P, A.Pid, A.T, A.V, A.rem, A.ea, A.eb, A.nxt, A.V32, A.ed2],
                  outputs=[A.walls, A.links, A.nwall, A.nlink, A.status], device=dev)
        self._mark('clip')
        # structure
        while True:
            cap = self.cap
            self.cnt.zero_()
            wp.launch(k_pair_count, n, inputs=[A.walls, A.nwall, A.nlink, A.status], outputs=[W_(self.pc, wp.int32), W_(self.cnt, wp.int32)], device=dev)
            torch.cumsum(self.pc, 0, dtype=i32, out=self.poff[1:])
            pairs = torch.empty(cap, 2, dtype=i32, device=dev); plinks = torch.empty(cap, LMAX, dtype=i32, device=dev)
            keep = torch.zeros(cap, dtype=i32, device=dev); deg = torch.zeros(n, dtype=i32, device=dev)
            wp.launch(k_pair_fill, n, inputs=[pw, rw, A.walls, A.links, A.nwall, A.nlink, W_(self.poff, wp.int32), cap, W_(self.ordr, wp.int32)],
                      outputs=[W_(pairs, wp.int32), W_(plinks, wp.int32), W_(keep, wp.int32), W_(deg, wp.int32), W_(self.cnt, wp.int32)], device=dev)
            kscan = torch.cumsum(keep, 0, dtype=i32)
            self.cnt[0] = self.poff[n]; self.cnt[1] = kscan[-1]; self.cnt[3] = deg.max()
            P = torch.empty(cap, 2, dtype=i32, device=dev); Cn = torch.empty(cap, LMAX, dtype=i32, device=dev)
            ci = torch.zeros(n, dtype=i32, device=dev); cjn = torch.zeros(n, dtype=i32, device=dev)
            wp.launch(k_pair_compact, cap, inputs=[W_(pairs, wp.int32), W_(plinks, wp.int32), W_(keep, wp.int32), W_(kscan, wp.int32), W_(self.poff, wp.int32), n],
                      outputs=[W_(P, wp.int32), W_(Cn, wp.int32), W_(ci, wp.int32), W_(cjn, wp.int32)], device=dev)
            soff = torch.zeros(n+1, dtype=i32, device=dev); torch.cumsum(ci+cjn, 0, dtype=i32, out=soff[1:])
            self._mark('structure')
            m, npr, K, Dmax, bits, over = self.cnt[:6].tolist()                    # (the one read)
            self._mark('read')
            if over:
                self.cap = int(1.5*max(cap, m))+1024
                continue
            break
        if bits:
            raise RuntimeError('alpha complex: per-cell storage overflow or an open cut (status up to %d)' % bits)
        if m > 0.9*self.cap:
            self.cap = int(1.3*m)+1024
        K = max(K, 1); Dmax = max(Dmax, 1)
        # entries and neighbours
        ent = torch.empty(max(2*npr, 1), dtype=i32, device=dev)
        wp.launch(k_ent_i, n, inputs=[W_(self.poff, wp.int32), W_(kscan, wp.int32), W_(keep, wp.int32), W_(ci, wp.int32), W_(soff, wp.int32)], outputs=[W_(ent, wp.int32)], device=dev)
        if npr:
            cur = torch.zeros(n, dtype=i32, device=dev)
            wp.launch(k_ent_j, npr, inputs=[W_(P, wp.int32), W_(ci, wp.int32), W_(soff, wp.int32), npr], outputs=[W_(cur, wp.int32), W_(ent, wp.int32)], device=dev)
        nbrs = torch.full((n, Dmax), -1, dtype=i32, device=dev)
        if m:
            curn = torch.zeros(n, dtype=i32, device=dev)
            wp.launch(k_nbrs, m, inputs=[W_(pairs, wp.int32), W_(self.poff, wp.int32), n], outputs=[W_(curn, wp.int32), W_(nbrs, wp.int32)], device=dev)
        hidden = wp.to_torch(A.status) == 1
        # values
        E = lambda *s_: torch.empty(*s_, dtype=f64, device=dev)
        np_ = max(npr, 1)
        ea, ed, el = E(np_, 2*K), E(np_, 2*K), torch.empty(np_, 2*K, dtype=i32, device=dev)
        nn, c, u, v = E(np_, 3), E(np_, 3), E(np_, 3), E(np_, 3)
        rho, hi, hj = E(np_), E(np_), E(np_)
        at0, at1, narc = E(np_, 2*K+1), E(np_, 2*K+1), torch.empty(np_, dtype=i32, device=dev)
        wA, wc, Vw = E(np_), E(np_, 3), torch.zeros(n, dtype=f64, device=dev)
        Pk, Ck = P[:np_].contiguous(), Cn[:np_, :K].contiguous()
        V3 = lambda t_: W_(t_, wp.vec3d)
        F_ = lambda t_: W_(t_, wp.float64)
        if npr:
            wp.launch(k_geo_pair, npr, inputs=[pw, rw, W_(Pk, wp.int32), W_(Ck, wp.int32), K, TWO_PI, BIG, 1e299, .9, F_(ea), F_(ed), W_(el, wp.int32)],
                      outputs=[V3(nn), V3(c), V3(u), V3(v), F_(rho), F_(hi), F_(hj), F_(at0), F_(at1), W_(narc, wp.int32), F_(wA), V3(wc), F_(Vw)], device=dev)
        pn, pA, V, pole = E(n, 3), E(n), E(n), E(n, 3)
        wp.launch(k_geo_site, n, inputs=[pw, rw, W_(soff, wp.int32), W_(ent, wp.int32), np_, V3(nn), V3(u), V3(v), F_(rho), F_(hi), F_(hj), F_(at0), F_(at1),
                                         W_(narc, wp.int32), GeoPipe._fib[(str(dev), self.npole)], self.npole, float(self.piece), BIG, 4*math.pi,
                                         W_(nbrs, wp.int32), W_(hidden.to(i32), wp.int32), int(getattr(self,'stable_area',False)), 0, 0.0, F_(Vw)], outputs=[V3(pn), F_(pA), F_(V), V3(pole)], device=dev)
        self._mark('values')
        G = PipeGeometry()
        G.p, G.r, G.n = p, r, n
        G.P, G.Cn, G.nbrs, G.hidden = Pk[:npr], Ck[:npr], nbrs, hidden            # (int32; only i, j as the callers index with them)
        G.i, G.j = G.P[:, 0].long(), G.P[:, 1].long()
        G.nn, G.c, G.u, G.v, G.rho, G.hi, G.hj = nn[:npr], c[:npr], u[:npr], v[:npr], rho[:npr], hi[:npr], hj[:npr]
        G.wall_A, G.wall_c, G.patch_n, G.patch_A, G.V, G.pole = wA[:npr], wc[:npr], pn, pA, V, pole
        G.narc, G.arc_t0, G.arc_t1 = narc[:npr], at0[:npr], at1[:npr]
        G.site_off, G.site_ent = soff, ent[:2*npr]
        S = dict(pairs=pairs[:m], links=plinks[:m, :K], nbrs=nbrs, hidden=hidden)  # (int32 views: the solve reads hidden, diagnostics the rest)
        return G, S
