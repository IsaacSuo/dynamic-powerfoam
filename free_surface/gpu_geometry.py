"""Exact geometry of the union of balls cut by the power diagram, differentiable in the sites and radii, on the GPU
(2026-10-22; step 1 toward recomputing it every time step). Two layers:
  structure  discrete, no gradient: the pairs of sites that may share a wall, for each pair the third sites whose
             half-planes may cut it (its link), each site's neighbours, the hidden sites (no power cell).
             structure_triangulation: edges and links of the regular triangulation (qhull, CPU; 0.14 s for 8k sites).
             structure_overlaps: every overlapping pair and its common overlapping neighbours (a superset giving the same
             values; for tests on small parts: the new foam has ~300 overlapping neighbours per site). Enough because
             a point of the wall (i, j) with pi_k < pi_i <= 0 lies in both balls i and j, and a site's exposed sphere is
             its sphere inside its power cell.
  values     smooth functions of the sites and radii given the structure, closed form, no quadrature:
  circles  pairs whose spheres meet: centre c, radius rho, normal n from i to j, basis u, v (u x v = n), distances h_i,
           h_j of the sites to the plane.
  arcs     each third ball covers one interval of the circle's angle (pi_k = C + R cos(theta - phi0) < 0); the
           uncovered intervals by a sweep over the sorted interval ends; their ends are the corners.
  walls    the disk clipped by the third sites' radical lines; area and first moment exactly (sectors from the disk
           centre for the arcs, triangles for the line segments). Which line bounds each segment is chosen without
           gradient; the segment ends are computed from the chosen lines only.
  patches  sphere s inside its power cell, bounded by the arcs (site i travels its circle with decreasing angle, site j
           with increasing: the patch on the left). On the unit sphere, m the normal, psi the angular radius of the
           cap cut by the other ball (cos psi = h / r):
             int m dA     = 1/2 sum int m x dm
             int m m^T dA = A/3 I - 1/6 sum int (nu m^T + m nu^T) ds, nu = (axis - cos psi m) / sin psi the outward
                            conormal (on the sphere the Laplace-Beltrami operator gives Delta x x^T = 2 I - 6 x x^T)
             A            = sum over arc pieces of the signed fan from a pole P: the geodesic triangle (P, a, b) (Van
                            Oosterom - Strackee) plus the signed segment between the arc and the geodesic a -> b
                            (Gauss-Bonnet: 2 beta + cos psi |dt|, beta the angle from the arc to the geodesic at a);
                            + 4 pi when -P is exposed. The value does not depend on P; P is chosen (no gradient) to keep
                            -P far from the boundary.
  volumes  V_i = 1/3 (sum_walls A h_i + r_i A_patch) (divergence theorem from the site).
Away from the configurations where the structure or the choices switch, values and gradients are exact."""
import math

import numpy as np
import torch

from free_surface.sphere_skin import triangulation


def _pack(pairs, triples, n, hidden, device):
    """pairs (E, 2) with i < j, triples (M, 3) = (i, j, k): k in the link of (i, j) -> padded links and neighbours"""
    key = pairs[:, 0].astype(np.int64)*n+pairs[:, 1]
    o = np.argsort(key); pairs, key = pairs[o], key[o]
    row = np.searchsorted(key, triples[:, 0].astype(np.int64)*n+triples[:, 1])
    assert len(triples) == 0 or (key[row] == triples[:, 0].astype(np.int64)*n+triples[:, 1]).all()
    cnt = np.bincount(row, minlength=len(pairs))
    o = np.argsort(row, kind='stable'); slot = np.arange(len(row))-(np.cumsum(cnt)-cnt)[row[o]]
    links = -np.ones((len(pairs), max(cnt.max(initial=0), 1)), np.int64); links[row[o], slot] = triples[o, 2]
    a = np.concatenate((pairs[:, 0], pairs[:, 1])); b = np.concatenate((pairs[:, 1], pairs[:, 0]))
    o = np.argsort(a, kind='stable'); a, b = a[o], b[o]
    deg = np.bincount(a, minlength=n); slot = np.arange(len(a))-(np.cumsum(deg)-deg)[a]
    nbrs = -np.ones((n, max(deg.max(initial=0), 1)), np.int64); nbrs[a, slot] = b
    t = lambda x: torch.as_tensor(x, device=device)
    return dict(pairs=t(pairs), links=t(links), nbrs=t(nbrs), hidden=t(hidden))


def structure_triangulation(p, r, device='cuda'):
    """edges between real sites of the regular triangulation, their links (real third sites), neighbours, hidden sites"""
    p = np.asarray(p, float); r = np.asarray(r, float); n = len(p)
    _, _, T, _, _, _, _ = triangulation(p, r)
    f = np.unique(np.sort(np.concatenate([T[:, [0, 1, 2]], T[:, [0, 1, 3]], T[:, [0, 2, 3]], T[:, [1, 2, 3]]]), axis=1), axis=0)
    f = f[(f < n).all(1)]
    e = np.sort(np.concatenate([T[:, [a, b]] for a in range(4) for b in range(a+1, 4)]), axis=1)
    e = np.unique(e[(e < n).all(1)], axis=0)
    hidden = np.ones(n, bool); hidden[T[T < n]] = False
    return _pack(e, np.concatenate([f[:, [0, 1, 2]], f[:, [0, 2, 1]], f[:, [1, 2, 0]]]), n, hidden, device)


def structure_overlaps(p, r, chunk=2048):
    """every overlapping pair (containment included), its common overlapping neighbours, neighbours (GPU)"""
    p, r = torch.as_tensor(p), torch.as_tensor(r)
    n = len(p); out = []
    for s in range(0, n, chunk):
        d = torch.cdist(p[s:s+chunk], p, compute_mode='donot_use_mm_for_euclid_dist')
        ii, jj = torch.nonzero(d < r[s:s+chunk, None]+r[None], as_tuple=True); ii = ii+s
        out.append(torch.stack((ii[ii < jj], jj[ii < jj]), 1))
    O = torch.cat(out).cpu().numpy()
    a = np.concatenate((O[:, 0], O[:, 1])); b = np.concatenate((O[:, 1], O[:, 0]))
    adj = [set() for _ in range(n)]
    for x, y in zip(a, b):
        adj[x].add(y)
    tri = [(i, j, k) for i, j in O for k in adj[i] & adj[j]]
    return _pack(O, np.array(tri, np.int64).reshape(-1, 3), n, np.zeros(n, bool), p.device)


def take(x, idx):
    """x[idx] along the first axis, with a fast backward (index_add; advanced indexing sorts the repeated indices)"""
    return torch.index_select(x, 0, idx.reshape(-1)).reshape(*idx.shape, *x.shape[1:])


def fibonacci(m, dt, dev):
    k = torch.arange(m, dtype=dt, device=dev)+.5
    ph = torch.acos(1-2*k/m); th = math.pi*(1+5**.5)*k
    return torch.stack((torch.cos(th)*torch.sin(ph), torch.sin(th)*torch.sin(ph), torch.cos(ph)), 1)


class Geometry:
    def __init__(self, p, r, S, piece=math.pi/6, npole=32):
        """p (n, 3), r (n): float64 tensors (may require grad); S: a structure (structure_triangulation / _overlaps)"""
        self.p, self.r, self.n = p, r, len(p)
        with torch.no_grad():
            P0 = S['pairs']; i0, j0 = P0[:, 0], P0[:, 1]
            L0 = (p[j0]-p[i0]).norm(dim=1)
            h0 = (L0*L0+r[i0]**2-r[j0]**2)/(2*torch.where(L0 > 0, L0, torch.ones_like(L0)))
            keep = (L0 > 0) & (r[i0]**2-h0*h0 > 0)
            self.P, self.Cn, self.nbrs, self.hidden = P0[keep], S['links'][keep], S['nbrs'], S['hidden']
        i, j = self.P[:, 0], self.P[:, 1]
        dt, dev = p.dtype, p.device
        pi_, pj_, ri, rj = take(p, i), take(p, j), take(r, i), take(r, j)
        d = pj_-pi_; L = d.norm(dim=1); nn = d/L[:, None]
        hi = (L*L+ri**2-rj**2)/(2*L)
        c = pi_+hi[:, None]*nn; rho = torch.sqrt(ri**2-hi**2)
        with torch.no_grad():
            ref = torch.where(nn[:, :1].abs() < .9, torch.tensor([1., 0, 0], dtype=dt, device=dev), torch.tensor([0, 1., 0], dtype=dt, device=dev))
        u = torch.cross(nn, ref, dim=1); u = u/u.norm(dim=1, keepdim=True); v = torch.cross(nn, u, dim=1)
        self.i, self.j, self.nn, self.c, self.rho, self.u, self.v, self.hi, self.hj = i, j, nn, c, rho, u, v, hi, L-hi
        self._arcs()
        self._walls()
        self._patches(piece, npole)
        Vw = torch.zeros(self.n, dtype=dt, device=dev).index_add(0, i, self.wall_A*hi).index_add(0, j, self.wall_A*(L-hi))
        self.V = (Vw+r*self.patch_A)/3

    # ------------------------------------------------------------------ arcs of every circle
    def _arcs(self):
        p, r, Cn = self.p, self.r, self.Cn
        valid = Cn >= 0; k = Cn.clamp(min=0)
        g = (take(p, self.i)[:, None]-take(p, k))+self.hi[:, None, None]*self.nn[:, None]   # c - p_k
        A = 2*self.rho[:, None]*(g*self.u[:, None]).sum(-1); B = 2*self.rho[:, None]*(g*self.v[:, None]).sum(-1)
        C = (g*g).sum(-1)+self.rho[:, None]**2-take(r, k)**2
        R2 = A*A+B*B
        R = torch.sqrt(torch.where(R2 > 0, R2, torch.ones_like(R2)))*(R2 > 0)
        full = valid & (-C >= R)                               # this ball covers the whole circle
        part = valid & (C.abs() < R)
        al = torch.acos(torch.where(part, -C/torch.where(part, R, torch.ones_like(R)), torch.zeros_like(C)))
        phi0 = torch.atan2(torch.where(part, B, torch.zeros_like(B)), torch.where(part, A, torch.ones_like(A)))
        two = 2*math.pi
        s = torch.remainder(phi0+al, two); e = s+two-2*al
        wraps = part & (e > two)
        e = torch.remainder(e, two)
        inf = torch.full_like(s, float('inf'))
        ang = torch.cat((torch.where(part, s, inf), torch.where(part, e, inf)), 1)
        dlt = torch.cat((part.to(s.dtype), -part.to(s.dtype)), 1)
        lab = torch.cat((Cn, Cn), 1)
        o = torch.argsort(ang.detach(), dim=1)
        ang, dlt, lab = torch.gather(ang, 1, o), torch.gather(dlt, 1, o), torch.gather(lab, 1, o)
        c0 = wraps.sum(1).to(s.dtype)
        cov = c0[:, None]+torch.cumsum(dlt, 1)
        nxt = torch.cat((ang[:, 1:], torch.full_like(ang[:, :1], float('inf'))), 1)
        anyfull = full.any(1)
        ok = (cov == 0) & torch.isfinite(ang) & torch.isfinite(nxt) & ~anyfull[:, None]
        nev = torch.isfinite(ang).sum(1); last = (nev-1).clamp(min=0)
        wrap_ok = (c0 == 0) & (nev > 0) & ~anyfull
        ring = (nev == 0) & ~anyfull                           # no ball touches the circle: a full ring
        # arcs: (pair, theta from, theta to, ball whose cover ends at the start, ball whose cover starts at the end)
        pr, m = torch.nonzero(ok, as_tuple=True)
        W = ang.shape[1]; fa, fn = ang.reshape(-1), nxt.reshape(-1)
        t0 = take(fa, pr*W+m); t1 = take(fn, pr*W+m); k0 = lab[pr, m]; k1 = lab[pr, m+1]
        pw = torch.nonzero(wrap_ok)[:, 0]
        t0 = torch.cat((t0, take(fa, pw*W+last[pw]))); t1 = torch.cat((t1, take(fa, pw*W)+two))
        k0 = torch.cat((k0, lab[pw, last[pw]])); k1 = torch.cat((k1, lab[pw, 0]))
        pr = torch.cat((pr, pw))
        prr = torch.nonzero(ring)[:, 0]
        z = torch.zeros(len(prr), dtype=t0.dtype, device=t0.device)
        self.arc_pair = torch.cat((pr, prr)); self.arc_t0 = torch.cat((t0, z)); self.arc_t1 = torch.cat((t1, z+two))
        self.arc_k0 = torch.cat((k0, torch.full_like(prr, -1))); self.arc_k1 = torch.cat((k1, torch.full_like(prr, -1)))

    def arc_points(self, t):
        ap = self.arc_pair
        return take(self.c, ap)+take(self.rho, ap)[:, None]*(torch.cos(t)[:, None]*take(self.u, ap)+torch.sin(t)[:, None]*take(self.v, ap))

    # ------------------------------------------------------------------ walls
    def _walls(self):
        p, r, Cn = self.p, self.r, self.Cn
        i = self.i; npr, K = Cn.shape; dt, dev = p.dtype, p.device
        ap = self.arc_pair; rho_a = take(self.rho, ap); t0, t1 = self.arc_t0, self.arc_t1
        A = torch.zeros(npr, dtype=dt, device=dev).index_add(0, ap, .5*rho_a**2*(t1-t0))
        M = torch.zeros(npr, 2, dtype=dt, device=dev).index_add(0, ap, (rho_a**3/3)[:, None]*torch.stack((torch.sin(t1)-torch.sin(t0), torch.cos(t0)-torch.cos(t1)), 1))
        # lines: pi_i <= pi_k  <=>  a . (s, t) <= beta in the plane x = c + s u + t v
        valid = Cn >= 0; k = Cn.clamp(min=0)
        g = take(p, k)-take(p, i)[:, None]
        a = 2*torch.stack(((self.u[:, None]*g).sum(-1), (self.v[:, None]*g).sum(-1)), -1)
        beta = (g*g).sum(-1)+take(r, i)[:, None]**2-take(r, k)**2-2*self.hi[:, None]*(self.nn[:, None]*g).sum(-1)
        an2 = (a*a).sum(-1)
        line = valid & (an2 > 0)
        an2s = torch.where(line, an2, torch.ones_like(an2))
        foot = a*(beta/an2s)[..., None]
        dirn = torch.stack((-a[..., 1], a[..., 0]), -1)/torch.sqrt(an2s)[..., None]       # the region on the left
        f2 = (foot*foot).sum(-1)
        rho2 = self.rho[:, None]**2
        inside = line & (f2 < rho2)
        half = torch.sqrt(torch.where(inside, rho2-f2, torch.ones_like(f2)))
        with torch.no_grad():                                  # which half-plane bounds each segment end
            ad = (a[:, None]*dirn[:, :, None]).sum(-1)                         # [p, l, k] (broadcast: einsum went through
            rhs = beta[:, None, :]-(a[:, None]*foot[:, :, None]).sum(-1)       # cuBLAS GEMMs, 10-28)
            other = valid[:, None, :] & ~torch.eye(K, dtype=torch.bool, device=dev)[None]
            bound = rhs/torch.where(ad != 0, ad, torch.ones_like(ad))
            upv, upk = torch.where(other & (ad > 0), bound, torch.full_like(bound, float('inf'))).min(-1)
            dnv, dnk = torch.where(other & (ad < 0), bound, torch.full_like(bound, -float('inf'))).max(-1)
            empty = (other & (ad == 0) & (rhs < 0)).any(-1)
            seg = inside & (torch.minimum(half, upv) > torch.maximum(-half, dnv)) & ~empty
            up_line, dn_line = upv < half, dnv > -half
        def bnd(kk):
            ak = torch.gather(a, 1, kk[..., None].expand(-1, -1, 2)); bk = torch.gather(beta, 1, kk)
            den = (ak*dirn).sum(-1)
            return (bk-(ak*foot).sum(-1))/torch.where(den != 0, den, torch.ones_like(den))
        tl = torch.where(dn_line, bnd(dnk), -half); tu = torch.where(up_line, bnd(upk), half)
        P0 = foot+tl[..., None]*dirn; P1 = foot+tu[..., None]*dirn
        cr = P0[..., 0]*P1[..., 1]-P0[..., 1]*P1[..., 0]
        A = A+torch.where(seg, .5*cr, torch.zeros_like(cr)).sum(1)
        M = M+torch.where(seg[..., None], (.5*cr)[..., None]*(P0+P1)/3, torch.zeros_like(P0)).sum(1)
        self.wall_A = A
        Mc = torch.where((A > 0)[:, None], M/torch.where(A > 0, A, torch.ones_like(A))[:, None], torch.zeros_like(M))
        self.wall_c = self.c+Mc[:, :1]*self.u+Mc[:, 1:]*self.v

    # ------------------------------------------------------------------ sphere patches
    def _patches(self, piece, npole):
        p, r = self.p, self.r
        dt, dev = p.dtype, p.device
        ap = self.arc_pair; na = len(ap)
        site = torch.cat((self.i[ap], self.j[ap]))
        nna = take(self.nn, ap)
        axis = torch.cat((nna, -nna))                                 # towards the other ball: the covered cap
        h = torch.cat((take(self.hi, ap), take(self.hj, ap)))
        rho, u, v = take(self.rho, ap).repeat(2), take(self.u, ap).repeat(2, 1), take(self.v, ap).repeat(2, 1)
        first = torch.arange(2*na, device=dev) < na
        t0, t1 = self.arc_t0.repeat(2), self.arc_t1.repeat(2)
        ta = torch.where(first, t1, t0); tb = torch.where(first, t0, t1)     # site i: decreasing angle, site j: increasing
        rs = take(r, site); cs = h/rs; sn = rho/rs
        e = lambda t: torch.cos(t)[:, None]*u+torch.sin(t)[:, None]*v
        D = tb-ta; sg = torch.sign(D.detach())
        # int n dA = 1/2 int x x dx (x from the site, c - p = h axis)
        vec = .5*(rho[:, None]*torch.cross(h[:, None]*axis, e(tb)-e(ta), dim=1)+(rho**2*D)[:, None]*torch.cross(u, v, dim=1))
        self.patch_n = torch.zeros(self.n, 3, dtype=dt, device=dev).index_add(0, site, vec)
        # int (axis m^T + m axis^T - 2 cos psi m m^T) |dt| over each arc (unit sphere)
        S1 = (torch.sin(tb)-torch.sin(ta))[:, None]*u+(torch.cos(ta)-torch.cos(tb))[:, None]*v
        Im = (cs*D)[:, None]*axis+sn[:, None]*S1
        s2 = torch.sin(2*tb)-torch.sin(2*ta)
        Icc, Iss, Isc = D/2+s2/4, D/2-s2/4, (torch.sin(tb)**2-torch.sin(ta)**2)/2
        o_ = lambda x, y: x[:, :, None]*y[:, None, :]
        S2 = Icc[:, None, None]*o_(u, u)+Isc[:, None, None]*(o_(u, v)+o_(v, u))+Iss[:, None, None]*o_(v, v)
        Imm = (cs*cs*D)[:, None, None]*o_(axis, axis)+(sn*cs)[:, None, None]*(o_(axis, S1)+o_(S1, axis))+(sn*sn)[:, None, None]*S2
        Q = sg[:, None, None]*(o_(axis, Im)+o_(Im, axis)-2*cs[:, None, None]*Imm)
        Qs = torch.zeros(self.n, 3, 3, dtype=dt, device=dev).index_add(0, site, Q)
        # area: arc pieces, fan from a pole
        npc = torch.ceil((D.detach().abs()/piece)).clamp(min=1).long()
        rep = torch.repeat_interleave(torch.arange(2*na, device=dev), npc)
        kk = torch.arange(len(rep), device=dev)-torch.repeat_interleave(torch.cumsum(npc, 0)-npc, npc)
        f0 = kk.to(dt)/npc[rep].to(dt); f1 = (kk+1).to(dt)/npc[rep].to(dt)
        tar, Dr = take(ta, rep), take(D, rep)
        pa = tar+Dr*f0; pb = tar+Dr*f1
        csr, snr, axr, ur, vr, sr = take(cs, rep), take(sn, rep), take(axis, rep), take(u, rep), take(v, rep), site[rep]
        er = lambda t: torch.cos(t)[:, None]*ur+torch.sin(t)[:, None]*vr
        ma = csr[:, None]*axr+snr[:, None]*er(pa); mb = csr[:, None]*axr+snr[:, None]*er(pb)
        with torch.no_grad():
            nd = self.patch_n/self.patch_n.norm(dim=1, keepdim=True).clamp(min=1e-300)
            cand = torch.cat((fibonacci(npole, dt, dev)[None].expand(self.n, -1, -1), nd[:, None], -nd[:, None]), 1)
            mm = csr[:, None]*axr+snr[:, None]*er(.5*(pa+pb))
            smp = torch.cat((ma, mm)); ss = torch.cat((sr, sr))
            dots = (cand[ss]*smp[:, None, :]).sum(-1)
            score = torch.full((self.n, cand.shape[1]), float('inf'), dtype=dt, device=dev).scatter_reduce(0, ss[:, None].expand(-1, cand.shape[1]), dots, reduce='amin', include_self=True)
            score[:, npole:][(self.patch_n.norm(dim=1) == 0)] = -float('inf')        # no mean normal to offer
            best = score.argmax(1)
            pole = cand[torch.arange(self.n, device=dev), best]
            self.pole_gap = math.pi-torch.acos(score.max(1).values.clamp(-1, 1))     # angle from -P to the boundary
        Pl = pole[sr]
        Om = 2*torch.atan2((Pl*torch.cross(ma, mb, dim=1)).sum(1), 1+(Pl*ma).sum(1)+(ma*mb).sum(1)+(mb*Pl).sum(1))
        tau = sg[rep][:, None]*(-torch.sin(pa)[:, None]*ur+torch.cos(pa)[:, None]*vr)
        gg = mb-(ma*mb).sum(1, keepdim=True)*ma
        bet = torch.atan2((torch.cross(tau, gg, dim=1)*ma).sum(1), (tau*gg).sum(1))
        Au = torch.zeros(self.n, dtype=dt, device=dev).index_add(0, sr, Om+2*bet+csr*(pb-pa).abs())
        with torch.no_grad():                                  # is the pole's antipode exposed
            nb = self.nbrs; kn = nb.clamp(min=0)
            dX = p[:, None]-p[kn]-r[:, None, None]*pole[:, None]
            exposed = torch.where(nb >= 0, (dX*dX).sum(-1)-r[kn]**2 > 0, torch.ones_like(nb, dtype=torch.bool)).all(1) & ~self.hidden
        Au = Au+4*math.pi*exposed.to(dt)
        r2 = r*r
        self.patch_A = r2*Au
        self.patch_nn = r2[:, None, None]*((Au/3)[:, None, None]*torch.eye(3, dtype=dt, device=dev)-Qs/6)
        self.pole, self.pole_idx, self.exposed = pole, best, exposed
