"""The fixed ground as one more clipping plane of the cut (11-08), instead of clipping the finished geometry afterwards
(fv_ale_ground_local: re-clip the walls that dip below the ground as disks against all neighbours, re-integrate those
cells' spheres; 5-9 ms a cut, mostly fixed overhead).

The ground half-space y >= H, seen from cell i, is a power constraint: pi_i(x) <= pi_k(x) for a virtual site k of i at
p_k = p_i - (0, 2 r_i, 0) with r_k^2 = r_i (5 r_i - 4 h_i), h_i = y_i - H: the radical plane of i and k is y = H (at
distance h_i from p_i along n = (0, -1, 0)), for either sign of h_i, and r_k^2 > r_i^2 (|h_i| < r_i). Cell i's power cell
cut by that plane is its cell above the ground; the face (i, k) is its ground face; an edge with the ground is a link k.
Every downstream formula (k_geo_pair's covered intervals and segments, k_geo_site's exposure test) is a power-distance
test, so it reads the ground exactly through k's p, r: no kernel changes.

k is a candidate of i only (inserted into i's sorted candidate list), never in the BVH: as another cell's candidate it
would pass through that cell's ground edges (four sites at equal power along the line: a degenerate cut). Virtual
site of cell i: index n + i (no count to read back); cells whose ball does not reach below the ground (h_i >= r_i) get
no plane, their virtual sites are referenced by nothing. The sites' arrays are n + n long, the cells' n.

GroundPipe(n)(A, p, r, H) -> G over the 2 n sites (pairs (i, n + i): the ground faces); plane_geometry(clip, p, r) -> the
PowerControlVolumes mesh with the fields fv_ale_ground_local.local_geometry gives. Default of GroundClip (11-08); FV_GROUND_PLANE=0: the local re-clip."""
import math

import torch
import warp as wp

from free_surface.alpha_complex import LMAX, AlphaComplex, k_alpha
from free_surface.geo_pipe import (BIG, GeoPipe, PipeGeometry, f64, i32, hval, k_cand, k_cand_full, k_ent_i, k_ent_j, k_hval, k_nbrs,
                                   k_pair_compact, k_pair_count, k_pair_fill, pair_kernel, site_launch)
from free_surface.geometry_warp import TWO_PI, k_geo_pair, k_geo_site


@wp.kernel(enable_backward=False)
def k_aug(p: wp.array(dtype=wp.vec3d), r: wp.array(dtype=wp.float64), H: wp.float64, floor_start: int,
          pa: wp.array(dtype=wp.vec3d), ra: wp.array(dtype=wp.float64), gh: wp.array(dtype=wp.float64), need: wp.array(dtype=wp.int32)):
    """the cells' ground heights, which reach below it, the 2 n sites (the virtual site of i at n + i)"""
    i = wp.tid()
    ri = r[i]
    y = p[i][1]-H
    gh[i] = y
    nd = int(0)
    rv2 = ri*ri
    if y < ri:
        nd = 1
        rv2 = ri*(wp.float64(5.0)*ri-wp.float64(4.0)*y)
    need[i] = nd
    pa[i] = p[i]
    ra[i] = ri
    q = p[i]
    q[1] = q[1]-wp.float64(2.0)*ri
    pa[floor_start+i] = q
    ra[floor_start+i] = wp.sqrt(rv2)


@wp.kernel(enable_backward=False)
def k_cand_out_g(p: wp.array(dtype=wp.vec3d), cell: wp.array(dtype=wp.int32), off: wp.array(dtype=wp.int32), adj: wp.array(dtype=wp.int32), hk: wp.array(dtype=wp.float64),
                 ek: wp.array(dtype=wp.int32), kc: wp.array(dtype=wp.int32), coff2: wp.array(dtype=wp.int32), need: wp.array(dtype=wp.int32), gh: wp.array(dtype=wp.float64),
                 cj: wp.array(dtype=wp.int32), cn: wp.array(dtype=wp.vec3d), ch: wp.array(dtype=wp.float64)):
    """geo_pipe.k_cand_out written at its place in the list with the ground plane: one further after the plane"""
    t = wp.tid()
    i = cell[t]
    u = t-off[i]
    if u >= kc[i]:
        return
    sh = int(0)
    if need[i] == 1 and hk[t] > gh[i]:
        sh = 1
    o = coff2[i]+u+sh
    j = adj[ek[t]]
    d = p[j]-p[i]
    cj[o] = j
    cn[o] = d/wp.length(d)
    ch[o] = hk[t]


@wp.kernel(enable_backward=False)
def k_ground_put(off: wp.array(dtype=wp.int32), hk: wp.array(dtype=wp.float64), kc: wp.array(dtype=wp.int32), coff2: wp.array(dtype=wp.int32),
                 need: wp.array(dtype=wp.int32), gh: wp.array(dtype=wp.float64), floor_start: int,
                 cj: wp.array(dtype=wp.int32), cn: wp.array(dtype=wp.vec3d), ch: wp.array(dtype=wp.float64)):
    """the ground plane (n + i, (0, -1, 0), h_i) after the kept planes with h <= h_i (a binary search: the kept are sorted)"""
    i = wp.tid()
    if need[i] == 0:
        return
    a = off[i]
    h = gh[i]
    lo = int(0)
    hi = int(kc[i])
    while lo < hi:
        mid = (lo+hi)//2
        if hk[a+mid] <= h:
            lo = mid+1
        else:
            hi = mid
    o = coff2[i]+lo
    cj[o] = floor_start+i
    cn[o] = wp.vec3d(wp.float64(0.0), wp.float64(-1.0), wp.float64(0.0))
    ch[o] = h


@wp.kernel(enable_backward=False)
def k_vacuum_count(p: wp.array(dtype=wp.vec3d), r: wp.array(dtype=wp.float64), start: int, end: int,
                   big: wp.float64, count: wp.array(dtype=wp.int32)):
    """Test every moving generator against each real ball, as in augmented_structure."""
    i = wp.tid()
    hits = int(0)
    for j in range(start, end):
        h = hval(p, r, i, j, big)
        if h == big or not wp.isfinite(h):
            count[i] = -1
            return
        if h < r[i]:
            hits += 1
    count[i] = hits


@wp.kernel(enable_backward=False)
def k_vacuum_put(p: wp.array(dtype=wp.vec3d), r: wp.array(dtype=wp.float64), start: int, end: int,
                 big: wp.float64, kc: wp.array(dtype=wp.int32), need: wp.array(dtype=wp.int32),
                 count: wp.array(dtype=wp.int32), off: wp.array(dtype=wp.int32),
                 cj: wp.array(dtype=wp.int32), cn: wp.array(dtype=wp.vec3d), ch: wp.array(dtype=wp.float64)):
    """Insert moving constraints into the existing sorted real/floor segment."""
    i = wp.tid()
    if count[i] <= 0:
        return
    begin = off[i]
    stop = begin+kc[i]+need[i]
    for j in range(start, end):
        h = hval(p, r, i, j, big)
        if h < r[i]:
            pos = stop
            while pos > begin:
                if ch[pos-1] <= h:
                    break
                cj[pos] = cj[pos-1]
                cn[pos] = cn[pos-1]
                ch[pos] = ch[pos-1]
                pos -= 1
            d = p[j]-p[i]
            cj[pos] = j
            cn[pos] = d/wp.length(d)
            ch[pos] = h
            stop += 1


class UnifiedCutFallback(RuntimeError):
    """Use the complete-constraint reference for a difficult moving-plane cut."""


class GroundPipe(GeoPipe):
    """Clip real cells against their neighbors, fixed ground and optional moving generators."""

    def __call__(self, A, p, r, H, vacuum=None):
        n, dev = self.n, self.dev
        floor_start = n if vacuum is None else len(vacuum[1])
        N = floor_start+n
        W_ = lambda t_, d_: wp.from_torch(t_, dtype=d_)
        p = p.contiguous(); r = r.contiguous()
        pw, rw = W_(p, wp.vec3d), W_(r, wp.float64)
        # IDs are real sites, moving generators, then each cell's private floor site.
        if getattr(self, '_an', 0) != N:
            self._an = N
            self.pa, self.ra = torch.empty(N, 3, dtype=f64, device=dev), torch.empty(N, dtype=f64, device=dev)
            self.gh, self.need = torch.empty(n, dtype=f64, device=dev), torch.empty(n, dtype=i32, device=dev)
            self.coff2 = torch.zeros(n+1, dtype=i32, device=dev)
        pa, ra, gh, need = self.pa, self.ra, self.gh, self.need
        wp.launch(k_aug, n, inputs=[pw, rw, float(H), floor_start], outputs=[W_(pa, wp.vec3d), W_(ra, wp.float64), W_(gh, wp.float64), W_(need, wp.int32)], device=dev)
        if vacuum is not None:
            pp, rr, redundant = vacuum
            pa[n:floor_start].copy_(pp[n:])
            ra[n:floor_start].copy_(rr[n:])
            need.masked_fill_(redundant[:n], 0)
        paw, raw_ = W_(pa, wp.vec3d), W_(ra, wp.float64)
        self._mark('start')
        # the overlaps of the real sites only, kept while they cover
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
        # candidates (GeoPipe's), then the ground inserted
        if self.sorted_key is not A.ov_key:
            wp.launch(k_cand_full, n, inputs=[pw, rw, offw, adjw, BIG], outputs=[hkw, ekw], device=dev)
            self.sorted_key = A.ov_key
        if len(adj_):
            wp.launch(k_hval, len(adj_), inputs=[pw, rw, W_(cell_, wp.int32), adjw, BIG, ekw], outputs=[hkw], device=dev)
        wp.launch(k_cand, n, inputs=[rw, offw, hkw, ekw], outputs=[W_(self.kc, wp.int32)], device=dev)
        coff2 = self.coff2
        counts = self.kc+need
        if vacuum is not None:
            vacuum_count = torch.empty_like(need)
            wp.launch(k_vacuum_count, n, inputs=[paw, raw_, n, floor_start, BIG],
                      outputs=[W_(vacuum_count, wp.int32)], device=dev)
            counts += vacuum_count.clamp_min(0)
        torch.cumsum(counts, 0, dtype=i32, out=coff2[1:])
        cap2 = self.ncap+n*(1+floor_start-n)
        if getattr(self, '_gcap', 0) < cap2:
            self._gcap = cap2
            self.cj2, self.ch2 = torch.zeros(cap2, dtype=i32, device=dev), torch.zeros(cap2, dtype=f64, device=dev)
            self.cn2 = torch.zeros(cap2, 3, dtype=f64, device=dev)
        gw = [W_(coff2, wp.int32), W_(need, wp.int32), W_(gh, wp.float64)]
        out2 = [W_(self.cj2, wp.int32), W_(self.cn2, wp.vec3d), W_(self.ch2, wp.float64)]
        if len(adj_):
            wp.launch(k_cand_out_g, len(adj_), inputs=[pw, W_(cell_, wp.int32), offw, adjw, hkw, ekw, W_(self.kc, wp.int32)]+gw, outputs=out2, device=dev)
        wp.launch(k_ground_put, n, inputs=[offw, hkw, W_(self.kc, wp.int32), *gw, floor_start], outputs=out2, device=dev)
        if vacuum is not None:
            wp.launch(k_vacuum_put, n, inputs=[paw, raw_, n, floor_start, BIG, W_(self.kc, wp.int32),
                      gw[1], W_(vacuum_count, wp.int32), gw[0]], outputs=out2, device=dev)
        self._mark('candidates')
        # clip (n cells; the virtual sites are ids only)
        cand = [W_(coff2, wp.int32), W_(self.cj2, wp.int32), W_(self.cn2, wp.vec3d), W_(self.ch2, wp.float64)]
        if getattr(A, 'coop', True):
            from free_surface.alpha_coop import clip
            clip(rw, *cand, A.walls, A.links, A.nwall, A.nlink, A.status, n, device=dev)
        else:
            wp.launch(k_alpha, n, inputs=[pw, rw]+cand+[A.P, A.Pid, A.T, A.V, A.rem, A.ea, A.eb, A.nxt, A.V32, A.ed2],
                      outputs=[A.walls, A.links, A.nwall, A.nlink, A.status], device=dev)
        self._mark('clip')
        # structure: the pairs from the cells (n), the sites' lists over N
        while True:
            cap = self.cap
            self.cnt.zero_()
            if vacuum is not None:
                self.cnt[4] = (vacuum_count.min() < 0).to(i32)*8
            wp.launch(k_pair_count, n, inputs=[A.walls, A.nwall, A.nlink, A.status], outputs=[W_(self.pc, wp.int32), W_(self.cnt, wp.int32)], device=dev)
            torch.cumsum(self.pc, 0, dtype=i32, out=self.poff[1:])
            pairs = torch.empty(cap, 2, dtype=i32, device=dev); plinks = torch.empty(cap, LMAX, dtype=i32, device=dev)
            keep = torch.zeros(cap, dtype=i32, device=dev); deg = torch.zeros(N, dtype=i32, device=dev)
            wp.launch(k_pair_fill, n, inputs=[paw, raw_, A.walls, A.links, A.nwall, A.nlink, W_(self.poff, wp.int32), cap, W_(self.ordr, wp.int32)],
                      outputs=[W_(pairs, wp.int32), W_(plinks, wp.int32), W_(keep, wp.int32), W_(deg, wp.int32), W_(self.cnt, wp.int32)], device=dev)
            kscan = torch.cumsum(keep, 0, dtype=i32)
            self.cnt[0] = self.poff[n]; self.cnt[1] = kscan[-1]; self.cnt[3] = deg.max()
            P = torch.empty(cap, 2, dtype=i32, device=dev); Cn = torch.empty(cap, LMAX, dtype=i32, device=dev)
            ci = torch.zeros(N, dtype=i32, device=dev); cjn = torch.zeros(N, dtype=i32, device=dev)
            wp.launch(k_pair_compact, cap, inputs=[W_(pairs, wp.int32), W_(plinks, wp.int32), W_(keep, wp.int32), W_(kscan, wp.int32), W_(self.poff, wp.int32), n],
                      outputs=[W_(P, wp.int32), W_(Cn, wp.int32), W_(ci, wp.int32), W_(cjn, wp.int32)], device=dev)
            soff = torch.zeros(N+1, dtype=i32, device=dev); torch.cumsum(ci+cjn, 0, dtype=i32, out=soff[1:])
            self._mark('structure')
            m, npr, K, Dmax, bits, over = self.cnt[:6].tolist()                    # (the one read)
            self._mark('read')
            if over:
                self.cap = int(1.5*max(cap, m))+1024
                continue
            break
        if bits == 8 and vacuum is not None:
            raise UnifiedCutFallback('singular real/vacuum plane')
        if bits:
            error = UnifiedCutFallback if vacuum is not None else RuntimeError
            raise error('alpha complex: per-cell storage overflow or an open cut (status up to %d)' % bits)
        if m > 0.9*self.cap:
            self.cap = int(1.3*m)+1024
        K = max(K, 1); Dmax = max(Dmax, 1)
        ent = torch.empty(max(2*npr, 1), dtype=i32, device=dev)
        wp.launch(k_ent_i, n, inputs=[W_(self.poff, wp.int32), W_(kscan, wp.int32), W_(keep, wp.int32), W_(ci, wp.int32), W_(soff, wp.int32)], outputs=[W_(ent, wp.int32)], device=dev)
        if npr:
            cur = torch.zeros(N, dtype=i32, device=dev)
            wp.launch(k_ent_j, npr, inputs=[W_(P, wp.int32), W_(ci, wp.int32), W_(soff, wp.int32), npr], outputs=[W_(cur, wp.int32), W_(ent, wp.int32)], device=dev)
        nbrs = torch.full((N, Dmax), -1, dtype=i32, device=dev)
        if m:
            curn = torch.zeros(N, dtype=i32, device=dev)
            wp.launch(k_nbrs, m, inputs=[W_(pairs, wp.int32), W_(self.poff, wp.int32), n], outputs=[W_(curn, wp.int32), W_(nbrs, wp.int32)], device=dev)
        hidden = torch.cat((wp.to_torch(A.status) == 1, torch.ones(N-n, dtype=torch.bool, device=dev)))
        # Integrate faces at all site types, and spherical surfaces of the real cells.
        E = lambda *s_: torch.empty(*s_, dtype=f64, device=dev)
        np_ = max(npr, 1)
        ea, ed, el = E(np_, 2*K), E(np_, 2*K), torch.empty(np_, 2*K, dtype=i32, device=dev)
        nn, c, u, v = E(np_, 3), E(np_, 3), E(np_, 3), E(np_, 3)
        rho, hi, hj = E(np_), E(np_), E(np_)
        at0, at1, narc = E(np_, 2*K+1), E(np_, 2*K+1), torch.empty(np_, dtype=i32, device=dev)
        wA, wc, Vw = E(np_), E(np_, 3), torch.zeros(N, dtype=f64, device=dev)
        Pk, Ck = P[:np_].contiguous(), Cn[:np_, :K].contiguous()
        V3 = lambda t_: W_(t_, wp.vec3d)
        F_ = lambda t_: W_(t_, wp.float64)
        if npr:
            wp.launch(pair_kernel(), npr, inputs=[paw, raw_, W_(Pk, wp.int32), W_(Ck, wp.int32), K, TWO_PI, BIG, 1e299, .9, F_(ea), F_(ed), W_(el, wp.int32)],
                      outputs=[V3(nn), V3(c), V3(u), V3(v), F_(rho), F_(hi), F_(hj), F_(at0), F_(at1), W_(narc, wp.int32), F_(wA), V3(wc), F_(Vw)], device=dev)
        pn, pA, V, pole = E(n, 3), E(n), E(n), E(n, 3)
        site_kernel = k_geo_site
        if getattr(self, 'direct_sphere', False):
            from free_surface.geometry_sphere_direct import k_geo_site_direct
            site_kernel = k_geo_site_direct
        site_launch(site_kernel, n, inputs=[paw, raw_, W_(soff, wp.int32), W_(ent, wp.int32), np_, V3(nn), V3(u), V3(v), F_(rho), F_(hi), F_(hj), F_(at0), F_(at1),
                                          W_(narc, wp.int32), GeoPipe._fib[(str(dev), self.npole)], self.npole, float(self.piece), BIG, 4*math.pi,
                                          W_(nbrs, wp.int32), W_(hidden.to(i32), wp.int32), int(getattr(self, 'stable_area', False)), 0, 0.0, W_(ent, wp.int32), 0, F_(Vw)],
                  outputs=[V3(pn), F_(pA), F_(V), V3(pole)], device=dev)
        self._mark('values')
        G = PipeGeometry()
        G.p, G.r, G.n, G.n_cells = pa, ra, N, n
        G.P, G.Cn, G.nbrs, G.hidden = Pk[:npr], Ck[:npr], nbrs, hidden
        G.i, G.j = G.P[:, 0].long(), G.P[:, 1].long()
        G.nn, G.c, G.u, G.v, G.rho, G.hi, G.hj = nn[:npr], c[:npr], u[:npr], v[:npr], rho[:npr], hi[:npr], hj[:npr]
        G.wall_A, G.wall_c, G.patch_n, G.patch_A, G.V, G.pole = wA[:npr], wc[:npr], pn, pA, V, pole
        G.narc, G.arc_t0, G.arc_t1 = narc[:npr], at0[:npr], at1[:npr]
        G.site_off, G.site_ent = soff, ent[:2*npr]
        return G


class LazyReal:
    """mesh.geometry: real_view(G) made at the first attribute read (the run reads none of it)"""

    def __init__(self, G, p, r):
        self._args, self._view = (G, p, r), None

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        if self._view is None:
            self._view = real_view(*self._args)
        return getattr(self._view, name)


def real_view(G, p, r):
    """G without the virtual sites: the pairs between cells, the neighbour lists without the ground ids (mesh.geometry)"""
    n = G.n_cells
    g = PipeGeometry()
    real = torch.nonzero(G.j < n).flatten()
    for k in ('P', 'Cn', 'i', 'j', 'nn', 'c', 'u', 'v', 'rho', 'hi', 'hj', 'wall_A', 'wall_c', 'narc', 'arc_t0', 'arc_t1'):
        setattr(g, k, getattr(G, k)[real])
    nb = G.nbrs[:n]
    g.nbrs = torch.where(nb >= n, torch.full_like(nb, -1), nb)
    g.p, g.r, g.n, g.hidden = p, r, n, G.hidden[:n]
    g.patch_n, g.patch_A, g.V, g.pole = G.patch_n, G.patch_A, G.V, G.pole
    g.site_off = g.site_ent = None                                                 # (the entries index G's pairs)
    return g


def plane_geometry(clip, p, r, planes=None):
    """Build material geometry from one cut, including any moving vacuum faces."""
    from free_surface.fv_ale_geometry import PowerControlVolumes
    n = len(r); dev = p.device; H = float(clip.ground)
    if getattr(clip, 'plane_size', 0) != n:
        clip.plane_size = n
        clip.plane_A = AlphaComplex(n, clip.device); clip.plane_A.ov_margin = .05      # (Recut's margin)
        clip.plane_pipe = GroundPipe(n, clip.device); clip.plane_pipe._cand_storage(1); clip.plane_pipe.stable_area = True
        if clip.recut is not None:
            clip.plane_pipe.direct_sphere = getattr(clip.recut.pipe, 'direct_sphere', False)
    vacuum = None if planes is None else planes.generators(p, r, H)
    G = clip.plane_pipe(clip.plane_A, p.contiguous(), r.contiguous(), H, vacuum=vacuum)
    floor_start = n if vacuum is None else len(vacuum[1])
    h = p[:, 1]-H
    real = G.j < n
    hidden = G.hidden[:n] | (h <= -r)
    volume = torch.where(hidden, torch.zeros_like(G.V), G.V)
    hidden = hidden | (volume == 0)
    capA = torch.where(hidden, torch.zeros_like(G.patch_A), G.patch_A)
    pn = torch.where(hidden[:, None], torch.zeros_like(G.patch_n), G.patch_n)
    keep = torch.nonzero(real & (G.wall_A > 0)).flatten()
    mesh = PowerControlVolumes(p.clone(), r.clone(), volume, G.i[keep], G.j[keep], G.wall_A[keep], G.nn[keep], G.wall_c[keep], capA, pn)
    floor = torch.nonzero(G.j >= floor_start).flatten()
    owner = G.i[floor]
    mesh.ground_area = torch.zeros_like(r).index_add(0, owner, G.wall_A[floor])
    mesh.ground_centroid = p.clone(); mesh.ground_centroid[owner] = G.wall_c[floor]
    mesh.ground_normal = torch.zeros_like(p); mesh.ground_normal[:, 1] = -1
    mesh.hidden = hidden
    mesh.geometry = LazyReal(G, p, r)
    mesh.ground_geometry_info = {'method': 'ground_plane_cut', 'stable_sphere_area': True, 'local_ground_update': False}
    if planes is not None:
        moving = (~real)&(G.j < floor_start)
        small = G.wall_A <= 1e-10*math.pi*r[G.i].square()
        if bool((moving&small).any()):
            raise UnifiedCutFallback('moving face requires precise disk integration')
        if not bool(torch.isfinite(G.wall_A).all() & torch.isfinite(volume).all() & (volume >= 0).all()
                    & torch.isfinite(capA).all() & torch.isfinite(pn).all()):
            raise UnifiedCutFallback('unified cut has invalid areas or volumes')
        mesh.planes = planes.clone()
        mesh.plane_i, mesh.plane_j = G.i[moving], G.j[moving]-n
        mesh.plane_area, mesh.plane_normal, mesh.plane_centroid = G.wall_A[moving], G.nn[moving], G.wall_c[moving]
        pp = vacuum[0]
        mesh.plane_length = (pp[G.j[moving]]-pp[mesh.plane_i]).norm(dim=1)
        mesh.ground_geometry_info['method'] = 'unified_plane_cut'
    return mesh
