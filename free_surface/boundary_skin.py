"""Boundary-only skin of a sphere surface (2026-10-22, step 2): skin points only at the corners and the arc midpoints;
inside a patch the skin deviation is interpolated from them, with no points of its own and no per-patch decisions
(no star test, no chord cuts, no arc refinement). The energy is unchanged (sphere_model.SkinTerms); only the integrals
a_q = int phi_q dA, c_q = int phi_q n dA, m_qp = int phi_q phi_p dA, T_ab = int g_a g_b^T dA, U_ab = int (g_a . g_b) P dA
(g the surface gradient) come from this interpolation.
  arcs     each arc carries 3 skin points (start corner, midpoint, end corner); along it the deviation is in
           span{1, cos, sin} of the circle angle, so an affine map is followed exactly and both patches of the arc share
           the trace. Corners are identified by their three sites and the side of the sites' plane.
  regions  the patch of a site; a site with several boundary loops: separate islands (the loops' Gauss-Bonnet disk
           areas add up to the exposed area) are separate regions; loops bounding one region (holes) stay together.
  inside   (interp 'shepard', default) u = Pi u + S[trace - Pi u]: Pi u the affine function of the point fitted to the
           arc traces by least squares in arc length (exact for affine data: the interpolated skin lies on the sphere,
           bulge included, and sum_q c_q (x) m_q = int n n^T dA); S the transfinite Shepard interpolation of the residual
           traces, kernel 1 / |X - Y|^2 over the arcs (normalised, near a smooth boundary piece it is the half-plane
           Poisson kernel, i.e. locally the harmonic extension; positive everywhere, so defined outside the patch too);
           it tends to the trace on every arc (exact seams), with no parameter that changes the interpolant. Boundary
           integrals by Gauss-Legendre after a sinh map around each arc's nearest point (near-singular kernel), the
           nodes placed for the evaluation points and kept fixed under forward differentiation.
           (interp 'mvc': mean value coordinates of the arcs cut into k_sub chords in the patch's stereographic plane;
           seams match only at the chord vertices and the interpolant changes with k_sub; kept for comparison.)
  integrals  signed fan from m0 = int n dA / |int n dA| to every arc piece (k_sub pieces per arc), Gauss-Legendre
           nq x nq per piece (regions not star-shaped from m0 are covered with signed overlaps); surface gradients by
           forward differentiation along two tangents."""
import math
import time
from types import SimpleNamespace

import numpy as np
import torch

from free_surface.gpu_geometry import Geometry, structure_triangulation
from free_surface.sphere_model import SkinTerms


def split_rings(G):
    """a full ring (an arc with no third ball on its circle, k0 = k1 = -1, 0 to 2 pi) as three arcs of 2 pi / 3 whose
    ends are corners of the ring's own (10-26, the lion with its air cells); the arc integrals are additive, so nothing
    else changes. In place, once; G.arc_ring / arc_sub: the ring and the piece (-1 for the other arcs)."""
    if hasattr(G, 'arc_ring'):
        return
    ring = G.arc_k0 < 0; nr = int(ring.sum()); keep = ~ring
    if nr == 0:
        G.arc_ring = G.arc_sub = torch.full_like(G.arc_pair, -1)
        return
    m = torch.arange(3, device=G.arc_pair.device).repeat(nr)
    t0 = G.arc_t0[ring].repeat_interleave(3)+(2*math.pi/3)*m
    neg = lambda k: torch.full((k,), -1, dtype=G.arc_k0.dtype, device=G.arc_k0.device)
    G.arc_pair = torch.cat((G.arc_pair[keep], G.arc_pair[ring].repeat_interleave(3)))
    G.arc_t0, G.arc_t1 = torch.cat((G.arc_t0[keep], t0)), torch.cat((G.arc_t1[keep], t0+2*math.pi/3))
    G.arc_k0, G.arc_k1 = torch.cat((G.arc_k0[keep], neg(3*nr))), torch.cat((G.arc_k1[keep], neg(3*nr)))
    G.arc_ring = torch.cat((neg(int(keep.sum())), torch.arange(nr, device=m.device).repeat_interleave(3)))
    G.arc_sub = torch.cat((neg(int(keep.sum())), m))


def corners_and_mids(G, p):
    """global skin ids: corners (deduplicated) then one midpoint per arc; per arc its (start corner, mid, end corner)"""
    split_rings(G)
    n = len(p); ap = G.arc_pair; na = len(ap)
    i, j = G.P[ap, 0], G.P[ap, 1]
    x0, x1, xm = G.arc_points(G.arc_t0), G.arc_points(G.arc_t1), G.arc_points(.5*(G.arc_t0+G.arc_t1))
    def key(k, x):
        tri = torch.sort(torch.stack((i, j, k), 1), 1).values
        a, b, c = p[tri[:, 0]], p[tri[:, 1]], p[tri[:, 2]]
        side = (torch.cross(b-a, c-a, dim=1)*(x-a)).sum(1) > 0
        return ((tri[:, 0]*n+tri[:, 1])*n+tri[:, 2])*2+side.long()
    rg = G.arc_ring >= 0                                                         # a ring's corners: keyed by the ring and the piece
    keys = torch.cat((torch.where(rg, -(1+3*G.arc_ring+G.arc_sub), key(G.arc_k0, x0)), torch.where(rg, -(1+3*G.arc_ring+(G.arc_sub+1) % 3), key(G.arc_k1, x1))))
    _, cid, cnt = torch.unique(keys, return_inverse=True, return_counts=True)
    nc = len(cnt)
    xs = torch.cat((x0, x1))
    Xc = torch.zeros(nc, 3, dtype=p.dtype, device=p.device).index_add(0, cid, xs)/cnt[:, None]
    spread = float((xs-Xc[cid]).norm(dim=1).max())
    gid = torch.stack((cid[:na], nc+torch.arange(na, device=p.device), cid[na:]), 1)
    return torch.cat((Xc, xm)), gid, nc, dict(corner_uses=torch.bincount(cnt).tolist(), corner_spread=spread)


def regions(p, r, k_sub=4, G=None, S=None):
    """The skin points and the regions with their boundary entries (every arc on its two spheres, travelled with the
    patch on the left), shared by the integrals (build) and the interpolant at any point (interpolant)."""
    dev, dt = p.device, p.dtype
    with torch.no_grad():
        if G is None:
            S = structure_triangulation(p.cpu().numpy(), r.cpu().numpy(), device=dev)
            G = Geometry(p, r, S)
        Y, gid, nc, diag = corners_and_mids(G, p)
        npt = len(Y); ap = G.arc_pair; na = len(ap)
        # boundary entries: every arc on its two spheres, travelled with the patch on the left
        site = torch.cat((G.i[ap], G.j[ap])); first = torch.arange(2*na, device=dev) < na
        t0, t1 = G.arc_t0.repeat(2), G.arc_t1.repeat(2)
        ta = torch.where(first, t1, t0); tb = torch.where(first, t0, t1)
        c_, rho_, u_, v_ = G.c[ap].repeat(2, 1), G.rho[ap].repeat(2), G.u[ap].repeat(2, 1), G.v[ap].repeat(2, 1)
        gids = gid.repeat(2, 1)
        sk = torch.arange(k_sub+1, dtype=dt, device=dev)/k_sub
        th = ta[:, None]+(tb-ta)[:, None]*sk[None]                                   # (B, k+1) along travel
        tm, beta = .5*(t0+t1), .5*(t1-t0)
        ph = th-tm[:, None]
        sb2 = torch.sin(beta/2)[:, None]**2; sp2 = torch.sin(ph/2)**2; sr = torch.sin(ph)/torch.sin(beta)[:, None]
        L = torch.stack((.5*(sp2/sb2-sr), (sb2-sp2)/sb2, .5*(sp2/sb2+sr)), -1)        # (B, k+1, 3): start, mid, end
        unit = lambda t, sel: (c_[sel][..., None, :]+rho_[sel][..., None, None]*(torch.cos(t)[..., None]*u_[sel][..., None, :]+torch.sin(t)[..., None]*v_[sel][..., None, :])
                               -p[site[sel]][..., None, :])/r[site[sel]][..., None, None]
        diag['arcs_beyond_0.9pi'] = int((beta > .9*math.pi).sum())
        diag['arc_sides_cap_over_hemisphere'] = int((torch.cat((G.hi[ap], G.hj[ap])) < 0).sum())     # these arcs bulge out of the chord polygon
        # boundary loops per site (start corner -> end corner along travel) and their Gauss-Bonnet areas (unit sphere:
        # 2 pi - int kappa_g - turning angles); a site with several loops: separate islands when the loops' disk areas add
        # up to the exposed area, one region (holes) when they exceed it by whole spheres; each island is its own region
        cs = torch.where(first, gids[:, 2], gids[:, 0]).cpu().numpy(); ce = torch.where(first, gids[:, 0], gids[:, 2]).cpu().numpy()
        site_np = site.cpu().numpy()
        h_e = torch.cat((G.hi[ap], G.hj[ap])); axis_e = torch.cat((G.nn[ap], -G.nn[ap]))
        sg = torch.sign(tb-ta)
        tang = lambda t: sg[:, None]*(-torch.sin(t)[:, None]*u_+torch.cos(t)[:, None]*v_)
        tau_s, tau_e = tang(ta).cpu().numpy(), tang(tb).cpu().numpy()
        kg = (-(h_e/r[site])*(tb-ta).abs()).cpu().numpy()
        e_ = lambda t: torch.cos(t)[:, None]*u_+torch.sin(t)[:, None]*v_
        x_end = ((c_+rho_[:, None]*e_(tb)-p[site])/r[site][:, None]).cpu().numpy()
        vec = .5*(rho_[:, None]*torch.cross(h_e[:, None]*axis_e, e_(tb)-e_(ta), dim=1)+(rho_**2*(tb-ta))[:, None]*torch.cross(u_, v_, dim=1))
        rn = r.cpu().numpy(); A_site = G.patch_A.cpu().numpy()
        region = np.full(2*na, -1); rsite, rarea, rloops = [], [], []
        by_site = {}
        for e in range(2*na):
            by_site.setdefault(int(site_np[e]), []).append(e)
        gb_err, n_isl, n_holes = 0., 0, 0
        for s_, es in by_site.items():
            start = {cs[e]: e for e in es}
            loops_, seen = [], set()
            for e0 in es:
                if e0 in seen:
                    continue
                lp = []; e = e0
                while e not in seen:
                    seen.add(e); lp.append(e); e = start[ce[e]]
                loops_.append(lp)
            areas = []
            for lp in loops_:
                turn = sum(math.atan2(float(np.cross(tau_e[e], tau_s[lp[(k_+1) % len(lp)]])@x_end[e]), float(tau_e[e]@tau_s[lp[(k_+1) % len(lp)]])) for k_, e in enumerate(lp))
                areas.append(rn[s_]**2*(2*math.pi-sum(kg[e] for e in lp)-turn))
            if len(loops_) == 1:
                gb_err = max(gb_err, abs(areas[0]-A_site[s_])/(4*math.pi*rn[s_]**2)); groups = [[0]]
            elif abs(sum(areas)-A_site[s_]) <= abs(sum(areas)-A_site[s_]-4*math.pi*rn[s_]**2*(len(loops_)-1)):
                groups = [[q] for q in range(len(loops_))]; n_isl += 1
            else:
                groups = [list(range(len(loops_)))]; n_holes += 1
            for g_ in groups:
                rid = len(rsite); rsite.append(s_); rloops.append(len(g_))
                rarea.append(areas[g_[0]] if len(g_) == 1 else A_site[s_])
                for q in g_:
                    region[loops_[q]] = rid
        diag.update(gauss_bonnet_vs_exact_area_max=gb_err, sites_with_islands=n_isl, sites_with_holes=n_holes)
        inv = torch.as_tensor(region, device=dev); ng = len(rsite)
        gsite = torch.as_tensor(np.array(rsite), device=dev); loops = np.array(rloops)
        A_reg = torch.as_tensor(np.array(rarea), dtype=dt, device=dev)
        n_reg = torch.zeros(ng, 3, dtype=dt, device=dev).index_add(0, inv, vec)
        m0 = n_reg/n_reg.norm(dim=1, keepdim=True)
        ref = torch.where(m0[:, :1].abs() < .9, torch.tensor([1., 0, 0], dtype=dt, device=dev), torch.tensor([0, 1., 0], dtype=dt, device=dev))
        e1 = torch.cross(m0, ref, dim=1); e1 = e1/e1.norm(dim=1, keepdim=True); e2 = torch.cross(m0, e1, dim=1)
        # where m0 and -m0 sit: on the exposed sphere or not (power-cell test against the site's neighbours)
        nb = S['nbrs'][gsite]; kn = nb.clamp(min=0)
        def inside(mv):
            X = p[gsite]+r[gsite, None]*mv
            pw = ((X[:, None]-p[kn])**2).sum(-1)-r[kn]**2
            return torch.where(nb >= 0, pw > 0, torch.ones_like(pw, dtype=torch.bool)).all(1)
        diag['m0_outside_patch'] = int((~inside(m0)).sum()); diag['antipode_inside_patch'] = int(inside(-m0).sum())
        diag['regions'] = ng
        order = torch.argsort(inv); counts = torch.bincount(inv, minlength=ng); offs = torch.cumsum(counts, 0)-counts
    return SimpleNamespace(G=G, S=S, Y=Y, gid=gid, nc=nc, npt=npt, ap=ap, na=na, site=site, t0=t0, t1=t1, sgn=torch.sign(tb-ta),
                           c=c_, rho=rho_, u=u_, v=v_, gids=gids, k_sub=k_sub, th=th, L=L, unit=unit, inv=inv, ng=ng, gsite=gsite,
                           loops=loops, A=A_reg, n=n_reg, m0=m0, e1=e1, e2=e2, inside=inside, order=order, counts=counts, offs=offs), diag


def lagrange3(thv, a0v, a1v):
    """span{1, cos, sin} through an arc's start, middle, end"""
    tmv, bv = .5*(a0v+a1v), .5*(a1v-a0v)
    phv = thv-tmv; s2 = torch.sin(bv/2)**2; p2 = torch.sin(phv/2)**2; srv = torch.sin(phv)/torch.sin(bv)
    return torch.stack((.5*(p2/s2-srv), (s2-p2)/s2, .5*(p2/s2+srv)), -1)


def interpolant(R, sl, p, r, interp='shepard', nbd=16):
    """The interpolant inside the regions sl (one padded batch: P regions, Bmax boundary entries ent, Smax local skin
    points loc_g, -1 padding): make_phi(mb) -> (phi, den), phi(m) maps unit directions m (P, Q, 3) from the region's site
    to the weights (P, Q, Smax) of its local skin points (build: the interpolations)."""
    dev, dt = p.device, p.dtype
    counts, order, offs, gids, gsite, k_sub = R.counts, R.order, R.offs, R.gids, R.gsite, R.k_sub
    m0, e1, e2, th, L, unit, t0, t1, sgn_all = R.m0, R.e1, R.e2, R.th, R.L, R.unit, R.t0, R.t1, R.sgn
    c_, rho_, u_, v_, Y = R.c, R.rho, R.u, R.v, R.Y
    gN, wN = np.polynomial.legendre.leggauss(nbd)
    gN = torch.as_tensor(.5*(gN+1), dtype=dt, device=dev); wN = torch.as_tensor(.5*wN, dtype=dt, device=dev)
    gP, wP = np.polynomial.legendre.leggauss(16)
    gP = torch.as_tensor(.5*(gP+1), dtype=dt, device=dev); wP = torch.as_tensor(.5*wP, dtype=dt, device=dev)
    P = len(sl)
    nE = counts[sl]; Bmax = int(nE.max())
    slot = torch.arange(Bmax, device=dev)
    valid_b = slot[None] < nE[:, None]
    ent = torch.where(valid_b, order[(offs[sl][:, None]+slot[None]).clamp(max=len(order)-1)], torch.zeros_like(slot[None]))   # (P, Bmax)
    # local skin ids
    g3 = torch.where(valid_b[..., None], gids[ent], torch.full_like(gids[ent], -1))
    loc_list = [torch.unique(g3[q][g3[q] >= 0]) for q in range(P)]
    Smax = max(len(x) for x in loc_list)
    loc_g = torch.full((P, Smax), -1, dtype=torch.long, device=dev)
    for q in range(P):
        loc_g[q, :len(loc_list[q])] = loc_list[q]
    loc3 = torch.searchsorted(torch.where(loc_g >= 0, loc_g, torch.full_like(loc_g, 2**62)), g3.clamp(min=0).reshape(P, -1)).reshape(P, Bmax, 3)
    ve = valid_b[:, :, None].expand(-1, -1, k_sub).reshape(P, -1)
    M0, E1, E2 = m0[sl], e1[sl], e2[sl]
    rr = r[gsite[sl]]; ps_ = p[gsite[sl]]
    def stereo(m):
        sh = (P,)+(1,)*(m.dim()-2)+(3,)
        return torch.stack(((m*E1.view(sh)).sum(-1), (m*E2.view(sh)).sum(-1)), -1)/(1+(m*M0.view(sh)).sum(-1, keepdim=True))
    if interp == 'mvc':
        # polygon vertices (unit vectors) and their trace weights
        mv = unit(th[ent], ent)                                                 # (P, Bmax, k+1, 3)
        Lv = L[ent]                                                             # (P, Bmax, k+1, 3)
        V = stereo(mv)                                                          # (P, Bmax, k+1, 2)
        va, vb = V[:, :, :-1].reshape(P, -1, 2), V[:, :, 1:].reshape(P, -1, 2)  # edges (P, Bmax*k, 2)
        La, Lb = Lv[:, :, :-1].reshape(P, -1, 3), Lv[:, :, 1:].reshape(P, -1, 3)
        ve = valid_b[:, :, None].expand(-1, -1, k_sub).reshape(P, -1)
        le = loc3[:, :, None, :].expand(-1, -1, k_sub, -1).reshape(P, -1, 3)
        def phi_mvc(m):                                                          # m (P, Q, 3) -> (P, Q, Smax)
            x = stereo(m)[:, :, None, :]
            da, db = va[:, None]-x, vb[:, None]-x
            ra, rb = da.norm(dim=-1), db.norm(dim=-1)
            A2 = da[..., 0]*db[..., 1]-da[..., 1]*db[..., 0]; D = (da*db).sum(-1)
            t = torch.where(ve[:, None], A2/(ra*rb+D), torch.zeros_like(A2))
            wa, wb = t/ra, t/rb
            coef = wa[..., None]*La[:, None]+wb[..., None]*Lb[:, None]           # (P, Q, E, 3)
            num = torch.zeros(P, m.shape[1], Smax, dtype=dt, device=dev).scatter_add(2, le[:, None].expand(-1, m.shape[1], -1, -1).reshape(P, m.shape[1], -1), coef.reshape(P, m.shape[1], -1))
            return num/(wa+wb).sum(-1, keepdim=True)
        make_phi = lambda mb: (phi_mvc, None)
    else:
        # u = Pi u + transfinite Shepard interpolation (kernel 1 / distance^2: near a smooth boundary piece its normalised
        # form is the half-plane Poisson kernel, i.e. locally the harmonic extension; positive everywhere) of the arcs'
        # residual traces (trace - Pi u, in span{1, cos, sin})
        a0e, a1e = t0[ent], t1[ent]                                             # increasing-angle range (P, Bmax)
        sg_e = torch.where(valid_b, sgn_all[ent], torch.zeros_like(sgn_all[ent]))   # travel direction (patch on the left)
        ce, roe, ue, vve = c_[ent], rho_[ent], u_[ent], v_[ent]
        def arcpt(thv, lead):                                                    # thv (P, *lead, Bmax, n) -> unit point, d/dtheta
            sh = (P,)+(1,)*lead+(Bmax, 1, 3)
            cv, uv, vv = ce.view(sh), ue.view(sh), vve.view(sh)
            rv = roe.view(sh[:-1]+(1,)); pv = ps_.view((P,)+(1,)*(lead+2)+(3,)); rsv = rr.view((P,)+(1,)*(lead+2)+(1,))
            co, si = torch.cos(thv)[..., None], torch.sin(thv)[..., None]
            return (cv+rv*(co*uv+si*vv)-pv)/rsv, rv*(-si*uv+co*vv)/rsv
        def stereo_d(m, dm):
            sh = (P,)+(1,)*(m.dim()-2)+(3,)
            a1, a2 = (m*E1.view(sh)).sum(-1), (m*E2.view(sh)).sum(-1); w = 1+(m*M0.view(sh)).sum(-1)
            d1, d2, dw = (dm*E1.view(sh)).sum(-1), (dm*E2.view(sh)).sum(-1), (dm*M0.view(sh)).sum(-1)
            return torch.stack((a1/w, a2/w), -1), torch.stack(((d1*w-a1*dw)/w**2, (d2*w-a2*dw)/w**2), -1)
        # Pi: affine functions of the point (basis 1, (m - m0) / kappa) fitted to the traces, least squares in arc length
        thg = a0e[..., None]+(a1e-a0e)[..., None]*gP                            # (P, Bmax, 16)
        mg, _ = arcpt(thg, 0)
        wg = torch.where(valid_b[..., None], (roe*(a1e-a0e))[..., None]*wP, torch.zeros_like(thg))
        kap = torch.where(valid_b[..., None], (mg-M0[:, None, None]).norm(dim=-1), torch.zeros_like(thg)).amax((1, 2)).clamp(min=1e-12)
        bfun = lambda m: torch.cat((torch.ones_like(m[..., :1]), (m-M0.view((P,)+(1,)*(m.dim()-2)+(3,)))/kap.view((P,)+(1,)*(m.dim()-1))), -1)
        bg = bfun(mg)                                                           # (P, Bmax, 16, 4)
        Mpi = torch.einsum('pbn,pbni,pbnj->pij', wg, bg, bg)
        rhs_e = torch.einsum('pbn,pbni,pbnk->pibk', wg, bg, lagrange3(thg, a0e[..., None], a1e[..., None]))
        rhs = torch.zeros(P, 4, Smax, dtype=dt, device=dev).scatter_add(2, loc3.reshape(P, 1, -1).expand(-1, 4, -1), rhs_e.reshape(P, 4, -1))
        Pi = torch.linalg.solve(Mpi, rhs)*(interp != 'shepard0')                # (P, 4, Smax): Pi u = b(m) . Pi d; 'shepard0': no Pi
        BY = bfun((Y[loc_g.clamp(min=0)]-ps_[:, None])/rr[:, None, None])        # (P, Smax, 4)
        BYPi = torch.einsum('psk,pkt->pst', BY, Pi)
        vmask = valid_b[:, None, :, None]
        def make_phi(mb):
            """the interpolant at points near mb (P, Q, 3): boundary nodes placed for mb (sinh map around each arc's
            nearest point), kept fixed so that forward derivatives are those of the quadrature rule"""
            with torch.no_grad():
                Q = mb.shape[1]
                X = ps_[:, None]+rr[:, None, None]*mb
                rel = X[:, :, None]-ce[:, None]                                  # (P, Q, Bmax, 3)
                thp = torch.atan2((rel*vve[:, None]).sum(-1), (rel*ue[:, None]).sum(-1))
                tmv = .5*(a0e+a1e)[:, None]
                thp = tmv+torch.remainder(thp-tmv+math.pi, 2*math.pi)-math.pi
                ths = torch.minimum(torch.maximum(thp, a0e[:, None]), a1e[:, None])
                ms, _ = arcpt(ths[..., None], 1)
                eps = ((ms[..., 0, :]-mb[:, :, None]).norm(dim=-1)*rr[:, None, None]/roe[:, None]).clamp(min=1e-14)   # distance / |dY/dtheta|
                TR = torch.asinh((a1e[:, None]-ths)/eps); TL = torch.asinh((ths-a0e[:, None])/eps)
                thR = ths[..., None]+eps[..., None]*torch.sinh(TR[..., None]*gN); wR = eps[..., None]*torch.cosh(TR[..., None]*gN)*TR[..., None]*wN
                thL = ths[..., None]-eps[..., None]*torch.sinh(TL[..., None]*gN); wL = eps[..., None]*torch.cosh(TL[..., None]*gN)*TL[..., None]*wN
                thn = torch.cat((thL, thR), -1)
                wn = torch.cat((wL, wR), -1)*(roe*valid_b)[:, None, :, None]        # arc length, (P, Q, Bmax, 2 nb)
                mn, _ = arcpt(thn, 1)                                            # boundary nodes on the unit sphere
                Lw = lagrange3(thn, a0e[:, None, :, None], a1e[:, None, :, None])*wn[..., None]
                idx = loc3[:, None, :, None, :].expand(-1, Q, -1, thn.shape[-1], -1).reshape(P, Q, -1)
            def kern(m):                                                         # 1 / distance^2 (chord, unit sphere)
                K = 1/((mn-m[:, :, None, None, :])**2).sum(-1)
                return torch.where(vmask, K, torch.zeros_like(K))
            def phi(m):
                K = kern(m)
                num = torch.zeros(P, m.shape[1], Smax, dtype=dt, device=dev).scatter_add(2, idx, (K[..., None]*Lw).reshape(P, m.shape[1], -1))
                Psi = num/(K*wn).sum((-1, -2))[..., None]
                return torch.einsum('pqk,pks->pqs', bfun(m), Pi)+Psi-torch.einsum('pqs,pst->pqt', Psi, BYPi)
            with torch.no_grad():
                den = (kern(mb)*wn).sum((-1, -2))
            return phi, den
    return SimpleNamespace(P=P, Bmax=Bmax, valid_b=valid_b, ent=ent, loc_list=loc_list, Smax=Smax, loc_g=loc_g, loc3=loc3, ve=ve,
                           M0=M0, E1=E1, E2=E2, rr=rr, ps=ps_, make_phi=make_phi)


def build(p, r, k_sub=4, nq=12, budget=6e6, G=None, S=None, data=None, interp='shepard', nbd=16, eta=1e-7):
    """p, r: float64 cuda tensors. Returns the rest skin points Y, kind (0 corner, 1 midpoint), per-patch integrals and
    diagnostics. data (npt, 3), optional: skin values for the seam continuity check."""
    dev, dt = p.device, p.dtype
    t_start = time.time()
    R, diag = regions(p, r, k_sub, G, S)
    G, Y, gid, nc, npt, ap, na, site, sgn_all = R.G, R.Y, R.gid, R.nc, R.npt, R.ap, R.na, R.site, R.sgn
    c_, rho_, u_, v_, th, unit, inside, m0 = R.c, R.rho, R.u, R.v, R.th, R.unit, R.inside, R.m0
    inv, ng, gsite, loops, A_reg, n_reg, counts = R.inv, R.ng, R.gsite, R.loops, R.A, R.n, R.counts
    with torch.no_grad():
        gq, wq = np.polynomial.legendre.leggauss(nq)
        gq = torch.as_tensor(.5*(gq+1), dtype=dt, device=dev); wq = torch.as_tensor(.5*wq, dtype=dt, device=dev)
    out = dict(a=[], c=[], m=[], T=[], U=[], gid=[], site=[])
    cont = []                                                    # (entry, sub-edge, value) at the chords' arc midpoints
    sumA = torch.zeros(ng, dtype=dt, device=dev); sumC = torch.zeros(ng, 3, dtype=dt, device=dev)
    min_phi, neg_w, min_eig = 0., 0., float('inf')
    pp_minphi = torch.zeros(ng, dtype=dt, device=dev); pp_eig = torch.zeros(ng, dtype=dt, device=dev); pp_neg = torch.zeros(ng, dtype=dt, device=dev)
    # chunks: as many regions as fit the budget (elements of the largest per-point x boundary array)
    cnt_np = counts.cpu().numpy()
    cost = lambda P_, Bm: P_*Bm*k_sub*nq*nq*(Bm*k_sub if interp == 'mvc' else Bm*2*nbd)
    spans, s0 = [], 0
    while s0 < ng:
        s1, Bm = s0+1, cnt_np[s0]
        while s1 < ng and cost(s1-s0+1, max(Bm, cnt_np[s1])) <= budget:
            Bm = max(Bm, cnt_np[s1]); s1 += 1
        spans.append((s0, s1)); s0 = s1
    diag['chunks'] = len(spans)
    aff_err, pu_err, den_min, den_neg = 0., 0., float('inf'), 0
    for s0, s1 in spans:
        sl = torch.arange(s0, s1, device=dev)
        I = interpolant(R, sl, p, r, interp, nbd)
        P, Bmax, valid_b, ent, loc_list, loc_g, ve = I.P, I.Bmax, I.valid_b, I.ent, I.loc_list, I.loc_g, I.ve
        M0, E2, rr, ps_, make_phi = I.M0, I.E2, I.rr, I.ps, I.make_phi
        # quadrature: fan from m0 over every chord's arc piece
        with torch.no_grad():
            thp = th[ent]                                                            # (P, Bmax, k+1)
            pa_, pb_ = thp[..., :-1], thp[..., 1:]                                   # (P, Bmax, k)
            tt = pa_[..., None]+(pb_-pa_)[..., None]*gq                              # (P, Bmax, k, nt)
            bb = unit(tt, ent[..., None])                                            # (P, Bmax, k, nt, 3)
            dbb = (rho_[ent][..., None, None, None]*(-torch.sin(tt)[..., None]*u_[ent][:, :, None, None]+torch.cos(tt)[..., None]*v_[ent][:, :, None, None])
                   /r[site[ent]][..., None, None, None])*(pb_-pa_)[..., None, None]
            Pl = M0[:, None, None, None, None, :]                                    # (P, 1, 1, 1, 1, 3)
            bw = bb[:, :, :, None]; dw = dbb[:, :, :, None]                          # (P, Bmax, k, 1, nt, 3)
            gw_ = gq[:, None, None]                                                  # (nw, 1, 1)
            s_ = (1-gw_)*Pl+gw_*bw                                                   # (P, Bmax, k, nw, nt, 3)
            ns = s_.norm(dim=-1); sh_ = s_/ns[..., None]
            cr = torch.cross((bw-Pl).expand_as(s_), dw.expand_as(s_), dim=-1)
            dA = (rr**2)[:, None, None, None, None]*gq[:, None]*(cr*sh_).sum(-1)/ns**2*wq[:, None]*wq[None, :]
            dA = torch.where(valid_b[:, :, None, None, None], dA, torch.zeros_like(dA))
            mq = sh_.reshape(P, -1, 3); wgt = dA.reshape(P, -1)
            mq = torch.where((wgt != 0)[..., None], mq, M0[:, None, :].expand_as(mq))
            t1_ = torch.cross(mq, E2[:, None].expand_as(mq), dim=-1); t1_ = t1_/t1_.norm(dim=-1, keepdim=True); t2_ = torch.cross(mq, t1_, dim=-1)
        phi, den = make_phi(mq)
        f, d1 = torch.func.jvp(phi, (mq,), (t1_,)); _, d2 = torch.func.jvp(phi, (mq,), (t2_,))
        with torch.no_grad():
            grad = (d1[..., None]*t1_[:, :, None, :]+d2[..., None]*t2_[:, :, None, :])/rr[:, None, None, None]   # (P, Q, S, 3)
            Pg = torch.eye(3, dtype=dt, device=dev)-mq[..., :, None]*mq[..., None, :]
            a = torch.einsum('pq,pqs->ps', wgt, f); c = torch.einsum('pq,pqs,pqi->psi', wgt, f, mq)
            Mm = torch.einsum('pq,pqs,pqt->pst', wgt, f, f)
            T = torch.einsum('pq,pqsi,pqtj->pstij', wgt, grad, grad)
            U = torch.einsum('pq,pqsk,pqtk,pqij->pstij', wgt, grad, grad, Pg)
            sumA[sl] = a.sum(1); sumC[sl] = c.sum(1)
            vm = wgt != 0
            Xi = torch.einsum('pqs,psi->pqi', f, Y[loc_g.clamp(min=0)]*(loc_g >= 0)[..., None])
            aff_err = max(aff_err, float(((Xi-ps_[:, None]-rr[:, None, None]*mq).norm(dim=-1)/rr[:, None])[vm].max()))
            pu_err = max(pu_err, float((f.sum(-1)-1).abs()[vm].max()))
            if den is not None:
                den_min = min(den_min, float(den[vm].min())); den_neg += int((den[vm] <= 0).sum())
            min_phi = min(min_phi, float(f[vm].min())); neg_w = neg_w+float(wgt.clamp(max=0).sum())
            pp_minphi[sl] = torch.where(vm[..., None], f, torch.zeros_like(f)).amin((1, 2)); pp_neg[sl] = -wgt.clamp(max=0).sum(1)/wgt.sum(1)
            for q in range(P):
                s_n = len(loc_list[q])
                ev = torch.linalg.eigvalsh(Mm[q, :s_n, :s_n])
                min_eig = min(min_eig, float(ev[0]/ev[-1])); pp_eig[sl[q]] = ev[0]/ev[-1]
                out['a'].append(a[q, :s_n]); out['c'].append(c[q, :s_n]); out['m'].append(Mm[q, :s_n, :s_n].reshape(-1))
                out['T'].append(T[q, :s_n, :s_n].reshape(-1, 3, 3)); out['U'].append(U[q, :s_n, :s_n].reshape(-1, 3, 3))
                out['gid'].append(loc_list[q]); out['site'].append(gsite[sl[q]].repeat(s_n))
            if data is not None:                                                     # values at the pieces' arc midpoints
                tmid = .5*(pa_+pb_)
                me = unit(tmid, ent).reshape(P, -1, 3)
                if interp != 'mvc':                                                  # just inside the patch (the limit is the trace)
                    dme = (rho_[ent][..., None, None]*(-torch.sin(tmid)[..., None]*u_[ent][:, :, None]+torch.cos(tmid)[..., None]*v_[ent][:, :, None])
                           /r[site[ent]][..., None, None]*sgn_all[ent][..., None, None]).reshape(P, -1, 3)
                    inw = torch.cross(me, dme, dim=-1); inw = inw/inw.norm(dim=-1, keepdim=True).clamp(min=1e-300)
                    me = me+eta*inw; me = me/me.norm(dim=-1, keepdim=True)
                fv = make_phi(me)[0](me)
                val = torch.einsum('pqs,psi->pqi', fv, data[loc_g.clamp(min=0)])
                cont.append((ent[:, :, None].expand(-1, -1, k_sub).reshape(P, -1)[ve], torch.arange(k_sub, device=dev).repeat(P, Bmax)[ve], val[ve]))
    A_exact, n_exact = A_reg, n_reg
    sph = 4*math.pi*r[gsite]**2
    diag.update(sumA_minus_A_of_sphere_max=float(((sumA-A_exact).abs()/sph).max()), sumC_minus_intn_of_sphere_max=float(((sumC-n_exact).norm(dim=1)/sph).max()),
                min_phi=min_phi, negative_weight_share=-neg_w/float(A_exact.sum()), mass_matrix_min_eig_ratio=min_eig, seconds=time.time()-t_start,
                affine_reproduction_max=aff_err, partition_of_unity_max=pu_err, tmvi_denominator_min=den_min, tmvi_denominator_nonpositive=den_neg)
    bulge = torch.zeros(ng, dtype=torch.bool, device=dev); bulge[inv[torch.cat((G.hi[ap], G.hj[ap])) < 0]] = True
    diag['per_patch'] = dict(site=gsite.cpu().numpy(), min_phi=pp_minphi.cpu().numpy(), min_eig=pp_eig.cpu().numpy(), neg_share=pp_neg.cpu().numpy(),
                             sumA_err=((sumA-A_exact).abs()/sph).cpu().numpy(), loops=loops, m0_out=(~inside(m0)).cpu().numpy(), bulge=bulge.cpu().numpy(),
                             frac=(A_exact/sph).cpu().numpy(), narcs=counts.cpu().numpy())
    if data is not None:
        e_, s_, v_ = (torch.cat(x) for x in zip(*cont))
        inc = torch.where(e_ < na, k_sub-1-s_, s_)                                  # sub-edge index along increasing angle
        key = torch.where(e_ < na, e_, e_-na)*k_sub+inc                              # same arc point from both sides
        o = torch.argsort(key*2+(e_ >= na).long())
        ks, vs = key[o], v_[o]
        assert bool((ks[0::2] == ks[1::2]).all())
        # the arc trace at those points (span{1, cos, sin} of the 3 skin points)
        arc = ks[0::2]//k_sub; sub = ks[0::2] % k_sub
        tq = G.arc_t0[arc]+(G.arc_t1[arc]-G.arc_t0[arc])*(sub.to(dt)+.5)/k_sub
        tmq, bq = .5*(G.arc_t0[arc]+G.arc_t1[arc]), .5*(G.arc_t1[arc]-G.arc_t0[arc])
        pq_ = tq-tmq; sb2q = torch.sin(bq/2)**2; sp2q = torch.sin(pq_/2)**2; srq = torch.sin(pq_)/torch.sin(bq)
        Lq = torch.stack((.5*(sp2q/sb2q-srq), (sb2q-sp2q)/sb2q, .5*(sp2q/sb2q+srq)), -1)
        trace = torch.einsum('qk,qki->qi', Lq, data[gid[arc]])
        scale = float(data.norm(dim=1).max())
        diag.update(seam_mismatch_max=float((vs[0::2]-vs[1::2]).norm(dim=1).max())/scale, off_trace_max=float(torch.maximum((vs[0::2]-trace).norm(dim=1), (vs[1::2]-trace).norm(dim=1)).max())/scale)
    psites = torch.unique(site)
    cat = lambda k_: torch.cat(out[k_]).cpu().numpy()
    off = np.concatenate(([0], np.cumsum([len(x) for x in out['gid']])))
    return dict(Y=Y.cpu().numpy(), kind=(torch.arange(npt, device=dev) >= nc).long().cpu().numpy(), n_corners=nc,
                ent_pid=cat('gid'), ent_site=cat('site'), ent_a=cat('a'), ent_c=cat('c'),
                part_off=off, part_site=gsite.cpu().numpy(), part_m=cat('m'), part_T=cat('T'), part_U=cat('U'),
                patch_site=psites.cpu().numpy(), patch_A=G.patch_A[psites].cpu().numpy(), patch_N=G.patch_nn[psites].cpu().numpy()), diag


class BoundaryTerms(SkinTerms):
    """the sphere-skin model with the boundary-only skin (build); bodies from the DEM geometry (body_of_cell).
    lever 'point': a skin point enters the F_B fit with its area share times its own normal (SkinTerms); 'c': with
    its share of int n dA, c_q (the same on flat faces; on spheres with an interpolation reproducing affine fields the
    skin's moment sum_q d_q (x) c_q is then exactly r N, the moment the outer layer's energy uses)"""
    def __init__(self, model, geometry, p, r, skin, variance=True, quad_ref=True, lever='point', gamma='sym_skin'):
        boc = np.asarray(geometry['body_of_cell'])
        site = skin['ent_site']; Y = skin['Y']
        ent = dict(pid=skin['ent_pid'], body=boc[site], a=skin['ent_a'], c=skin['ent_c'], n=(Y[skin['ent_pid']]-p[site])/r[site, None])
        ps = skin['part_site']
        parts = dict(off=skin['part_off'], body=boc[ps], h=r[ps], m=skin['part_m'], T=skin['part_T'], U=skin['part_U'])
        pz = skin['patch_site']
        patches = dict(body=boc[pz], h=r[pz], A=skin['patch_A'], N=skin['patch_N'])
        super().__init__(model, geometry, Y, ent, parts, patches, variance, quad_ref, gamma)
        if lever == 'c':
            self.e_l, self.e_w = self.e_c, torch.ones_like(self.e_w)
            half = (self.A*self.L/2)[:, None, None]*self.nrm[:, :, None]*self.nrm[:, None, :]
            Mb = torch.zeros(self.n, 3, 3, dtype=self.X.dtype, device=self.X.device).index_add(0, self.i, half).index_add(0, self.j, half)
            Ms = torch.zeros_like(Mb).index_add(0, self.e_body, self.e_d[:, :, None]*self.e_l[:, None, :])
            self.Mt_inv = torch.linalg.inv(Mb+Ms)
