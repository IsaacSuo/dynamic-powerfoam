"""Skin points of a sphere surface and their rest integrals (2026-10-22, step 2 for the part; sphere_skin.py,
part_geometry.py). Boundary points: the corners (shared by three patches) and the middle of every arc (two patches);
each patch region (a loop of part_geometry, a disk) is cut into edges between consecutive boundary points (half arcs).
A region is fanned from a point of its own on the sphere: per edge, the sector (point, edge start, edge end). The fan
has a positive area element everywhere iff (b x b').P > 0 along the boundary (b the boundary direction, b' its
derivative, P the point's direction: the sign of the gnomonic area element for every w), i.e. iff the region is
star-shaped from P; P maximises the minimum of (b x b').P / |b x b'|. A region star-shaped from no point (its arcs bulge
in too far) is cut by a chord (the great circle between two of its boundary points, inside the region) into two parts
star-shaped from points of their own, recursively; the chord is a straight edge of both parts' fans, so the
interpolation along it uses its two end points only (continuous). Shape functions on a sector: barycentric coordinates
of the radial projection onto the plane of its three points (partition of unity, continuous along spokes and chords,
>= 0 since the arcs bulge into the region). Per part and point q: a_q = int phi_q dA, c_q = int phi_q n dA,
m_qp = int phi_q phi_p dA (Gauss-Legendre on each sector). Checks: star value > 0 for every part; per region sum a_q =
area and sum c_q = int n dA (sphere_skin.fan_integrals); m positive definite; a_q >= 0."""
import argparse
from pathlib import Path

import numpy as np
import torch

from free_surface.sphere_skin import fan_integrals, skeleton


def loop_edges(S, k, nsub):
    """boundary point keys ('c', corner) / ('m', arc, s) in travel order, and the edges from each to the next; arc ai is
    cut into nsub[ai] edges by nsub[ai] - 1 points at equal angles (shared by both of its patches)"""
    keys, edges = [], []
    for ai, f in zip(S['loop_arc'][S['loop_off'][k]:S['loop_off'][k+1]], S['loop_fwd'][S['loop_off'][k]:S['loop_off'][k+1]]):
        i, j, c0, c1 = S['arc_ij'][ai]
        t0, t1 = S['arc_theta'][ai]; m = nsub[ai]
        cc = S['arc_circle'][ai]; C = dict(c=cc[0:3], rho=cc[3], u=cc[4:7], v=cc[7:10])
        th = [t0+(t1-t0)*q/m for q in range(m+1)]
        idx = list(range(m+1)) if f else list(range(m, -1, -1))
        keys.append(('c', c0 if f else c1))
        for q in range(m):
            if q > 0:
                keys.append(('m', ai, idx[q]))
            edges.append(('arc', C, th[idx[q]], th[idx[q+1]]))
    return keys, edges


def key_dir(S, k_, pc, rr, nsub):
    if k_[0] == 'c':
        return (S['corners'][k_[1]]-pc)/rr
    cc = S['arc_circle'][k_[1]]; t0, t1 = S['arc_theta'][k_[1]]; t = t0+(t1-t0)*k_[2]/nsub[k_[1]]
    return (cc[0:3]+cc[3]*(np.cos(t)*cc[4:7]+np.sin(t)*cc[7:10])-pc)/rr


def edge_eval(e, xi, pc, rr):
    """unit directions b(xi) and db/dxi along an edge"""
    if e[0] == 'arc':
        C, a, b = e[1], e[2], e[3]
        t = a+(b-a)*xi
        bb = (C['c'][None]+C['rho']*(np.cos(t)[:, None]*C['u']+np.sin(t)[:, None]*C['v'])-pc)/rr
        return bb, C['rho']*(-np.sin(t)[:, None]*C['u']+np.cos(t)[:, None]*C['v'])/rr*(b-a)
    A, B = e[1], e[2]
    s = (1-xi)[:, None]*A+xi[:, None]*B; ns = np.linalg.norm(s, axis=1, keepdims=True); bb = s/ns
    return bb, ((B-A)[None]-(bb*((B-A)[None]*bb).sum(1, keepdims=True)))/ns


def star(edges, pc, rr, per=24, start=None):
    """the direction maximising min (b x b').P / |b x b'| over the boundary, and that minimum; sampled at the
    Gauss-Legendre nodes of each edge (the quadrature's own points: at a cusp corner the tangent is radial and the
    value there is 0 from any interior point); the search starts from `start` when given"""
    xi = .5*(np.polynomial.legendre.leggauss(per)[0]+1)
    G = []
    for e in edges:
        b, db = edge_eval(e, xi, pc, rr)
        g = np.cross(b, db); G.append(g/np.linalg.norm(g, axis=1, keepdims=True))
    G = np.concatenate(G)
    j = np.arange(2000)+.5
    ph = np.arccos(1-2*j/2000); th = np.pi*(1+5**.5)*j
    C = np.column_stack((np.cos(th)*np.sin(ph), np.sin(th)*np.sin(ph), np.cos(ph)))
    if start is not None:
        C = np.concatenate((C, [start]))
    best = C[np.argmax((C@G.T).min(1))]; val = (best@G.T).min(); step = .05
    rng = np.random.default_rng(0)
    for _ in range(300):
        T = best[None]+step*rng.standard_normal((64, 3)); T /= np.linalg.norm(T, axis=1, keepdims=True)
        v = (T@G.T).min(1)
        if v.max() > val:
            best, val = T[np.argmax(v)], v.max()
        else:
            step *= .8
    return best, float(val)


def exposed(x, site, p, r):
    pt, rt = torch.as_tensor(p, device='cuda'), torch.as_tensor(r, device='cuda')
    pw = torch.cdist(torch.as_tensor(x, device='cuda'), pt)**2-rt[None]**2
    return (pw.argmin(1) == site).cpu().numpy()


def decompose(keys, edges, dirs, site, p, r, depth=0, start=None):
    """[(pole, keys, edges, star value)] parts star-shaped from their poles (edge e runs from keys[e] to keys[e+1])"""
    pc, rr = p[site], r[site]
    P, v = star(edges, pc, rr, start=start)
    if v > 0 or depth > 6:
        return [(P, keys, edges, v)]
    best = None
    n = len(keys)
    for a in range(n):
        for b in range(a+2, n):
            if a == 0 and b == n-1:
                continue
            A, B = dirs[keys[a]], dirs[keys[b]]
            w = np.linspace(0, 1, 34)[1:-1]
            s = (1-w)[:, None]*A+w[:, None]*B; s /= np.linalg.norm(s, axis=1, keepdims=True)
            if not exposed(pc+rr*s, site, p, r).all():
                continue
            k1, e1 = keys[a:b+1], edges[a:b]+[('chord', B, A)]
            k2, e2 = keys[b:]+keys[:a+1], edges[b:]+edges[:a]+[('chord', A, B)]
            v1, v2 = star(e1, pc, rr)[1], star(e2, pc, rr)[1]
            if best is None or min(v1, v2) > best[0]:
                best = (min(v1, v2), (k1, e1), (k2, e2))
    if best is None:
        return [(P, keys, edges, v)]
    return decompose(*best[1], dirs, site, p, r, depth+1)+decompose(*best[2], dirs, site, p, r, depth+1)


def part_rest(pole, keys, edges, dirs, pc, rr, nw=16, nt=16):
    """point keys of the part (its own point first), a, c, m, k (k_qp = int grad_s phi_q . grad_s phi_p dA, the surface
    gradient on the sphere), T_qp = int g_q g_p^T dA and U_qp = int (g_q . g_p)(I - n n^T) dA (g = grad_s phi; for the
    strain of the surface deviation in the tangent plane), area, smallest area element / (w r^2)"""
    gw, ww = np.polynomial.legendre.leggauss(nw); gw = .5*(gw+1); ww = .5*ww
    gt, wt = np.polynomial.legendre.leggauss(nt); gt = .5*(gt+1); wt = .5*wt
    loc = {'own': 0}
    for k_ in keys:
        loc.setdefault(k_, len(loc))
    npnt = len(loc)
    a_ = np.zeros(npnt); c_ = np.zeros((npnt, 3)); m_ = np.zeros((npnt, npnt)); kmat = np.zeros((npnt, npnt)); T_ = np.zeros((npnt, npnt, 3, 3)); U_ = np.zeros((npnt, npnt, 3, 3)); area = 0.; emin = np.inf
    for e_i, e in enumerate(edges):
        ka, kb = keys[e_i], keys[(e_i+1) % len(keys)]
        ids = [0, loc[ka], loc[kb]]
        Ginv = np.linalg.inv(np.stack([pole, dirs[ka], dirs[kb]], 1))
        bb, db = edge_eval(e, gt, pc, rr)
        s = (1-gw)[:, None, None]*pole+gw[:, None, None]*bb[None]
        ns = np.linalg.norm(s, axis=-1); sh = s/ns[..., None]
        el = rr*rr*gw[:, None]*(np.cross(bb[None]-pole, db[None])*sh).sum(-1)/ns**2
        emin = min(emin, float((el/gw[:, None]).min()/rr**2))
        dA = el*ww[:, None]*wt[None]
        co = np.einsum('ij,wtj->wti', Ginv, sh); S_ = co.sum(-1, keepdims=True); phi = co/S_
        gr = (Ginv[None, None]-phi[..., None]*Ginv.sum(0)[None, None, None])/S_[..., None]     # d phi / d s_hat
        gr = (gr-(gr*sh[:, :, None, :]).sum(-1, keepdims=True)*sh[:, :, None, :])/rr          # surface gradient
        kmat[np.ix_(ids, ids)] += np.einsum('wt,wtqi,wtpi->qp', dA, gr, gr)
        Pn = np.eye(3)[None, None]-sh[..., :, None]*sh[..., None, :]
        T_[np.ix_(ids, ids)] += np.einsum('wt,wtqi,wtpj->qpij', dA, gr, gr)
        U_[np.ix_(ids, ids)] += np.einsum('wt,wtqk,wtpk,wtij->qpij', dA, gr, gr, Pn)
        area += dA.sum()
        np.add.at(a_, ids, np.einsum('wt,wtq->q', dA, phi))
        np.add.at(c_, ids, np.einsum('wt,wtq,wti->qi', dA, phi, sh))
        m_[np.ix_(ids, ids)] += np.einsum('wt,wtq,wtp->qp', dA, phi, phi)
    order = [None]*npnt
    for k_, v_ in loc.items():
        order[v_] = k_
    return order, a_, c_, m_, kmat, T_, U_, area, emin


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--part', type=Path, default=Path('output/test_part'))
    args = a.parse_args()
    st = torch.load(args.part/'run'/'model.pt', map_location='cpu', weights_only=True)
    p = st['points'].double().numpy(); r = torch.nn.functional.softplus(st['radii'].double(), beta=100).numpy()
    S = dict(np.load(args.part/'sphere_skin.npz'))
    assert (S['arc_ij'][:, 2] >= 0).all(), 'rings (arcs without corners) are not handled'
    sk = skeleton(p, r)
    nl, nc, na = len(S['loop_site']), len(S['corners']), len(S['arc_ij'])
    nsub = np.full(na, 2)
    ref = []
    for k in range(nl):
        lp = list(zip(S['loop_arc'][S['loop_off'][k]:S['loop_off'][k+1]], S['loop_fwd'][S['loop_off'][k]:S['loop_off'][k+1]]))
        ref.append(fan_integrals(p, r, sk, S['loop_site'][k], lp)[:2])
    for it in range(5):                                   # refine the arcs of regions with a folded part, until none
        parts, folded = [], set()
        for k in range(nl):
            site = S['loop_site'][k]; pc, rr = p[site], r[site]
            keys, edges = loop_edges(S, k, nsub)
            dirs = {k_: key_dir(S, k_, pc, rr, nsub) for k_ in keys}
            dec = decompose(keys, edges, dirs, site, p, r, start=ref[k][1]/np.linalg.norm(ref[k][1]))
            for pole, kk, ee, v in dec:
                if v <= 0:
                    folded.add(k)
                parts.append((k, site, pole, kk, ee, v, dirs, len(dec) > 1))
        print('pass %d: arcs refined %d, regions with a folded part %d' % (it, int((nsub > 2).sum()), len(folded)), flush=True)
        if not folded:
            break
        for k in folded:
            for ai in S['loop_arc'][S['loop_off'][k]:S['loop_off'][k+1]]:
                nsub[ai] *= 2
    out, worst = [], dict(star=np.inf, emin=np.inf, area=0., cn=0., eig=np.inf, amin=np.inf)
    sums = {}
    for k, site, pole, kk, ee, v, dirs, was_split in parts:
        pc, rr = p[site], r[site]
        order, a_, c_, m_, kg, Tq, Uq, area, emin = part_rest(pole, kk, ee, dirs, pc, rr)
        worst['tu'] = max(worst.get('tu', 0.), float(np.abs(np.einsum('qpii->qp', Tq)-kg).max()/np.abs(kg).max()),
                          float(np.abs(np.einsum('qpii->qp', Uq)-2*kg).max()/np.abs(kg).max()))
        worst['krow'] = max(worst.get('krow', 0.), float(np.abs(kg.sum(1)).max()/np.abs(kg).max()))
        ev = np.linalg.eigvalsh(m_)
        worst['star'] = min(worst['star'], v); worst['emin'] = min(worst['emin'], emin)
        worst['eig'] = min(worst['eig'], ev[0]/ev[-1]); worst['amin'] = min(worst['amin'], a_.min()/area)
        A_s, n_s = sums.get(k, (0., np.zeros(3)))
        sums[k] = (A_s+area, n_s+c_.sum(0))
        out.append(dict(site=site, region=k, pole=pole, keys=order, a=a_, c=c_, m=m_, k=kg, T=Tq, U=Uq))
    for k, (A_s, n_s) in sums.items():
        worst['area'] = max(worst['area'], abs(A_s/ref[k][0]-1)); worst['cn'] = max(worst['cn'], np.linalg.norm(n_s-ref[k][1])/ref[k][0])
    parts = out
    split = sum(1 for k in range(nl) if sum(pt['region'] == k for pt in parts) > 1)
    print('regions %d, cut by chords %d, parts %d | star value min %.3f, area element / r^2 min %.3f | per region |sum a_q / area - 1| max %.1e, '
          '|sum c_q - int n dA| / area max %.1e | m smallest / largest eigenvalue min %.2e | a_q / area min %.4f | gradient matrix row sums / max entry %.1e | '
          'tr T = k, tr U = 2 k: max deviation %.1e' % (
              nl, split, len(parts), worst['star'], worst['emin'], worst['area'], worst['cn'], worst['eig'], worst['amin'], worst['krow'], worst['tu']), flush=True)
    # global point ids: corners, arc middles, part points
    aoff = np.concatenate(([0], np.cumsum(nsub-1)))                  # arc ai's points: nc + aoff[ai] + (s - 1)
    na_pts = int(aoff[-1])
    def gid(k_, pi_):
        if k_ == 'own':
            return nc+na_pts+pi_
        return k_[1] if k_[0] == 'c' else nc+aoff[k_[1]]+k_[2]-1
    pid = [np.array([gid(k_, pi_) for k_ in pt['keys']]) for pi_, pt in enumerate(parts)]
    npts = nc+na_pts+len(parts)
    print('skin points: corners %d + arc points %d + part points %d = %d' % (nc, na_pts, len(parts), npts), flush=True)
    sizes = np.array([len(x) for x in pid])
    np.savez_compressed(args.part/'sphere_points.npz', part_site=np.array([pt['site'] for pt in parts]), part_region=np.array([pt['region'] for pt in parts]),
                        part_pole=np.array([pt['pole'] for pt in parts]), part_off=np.concatenate(([0], np.cumsum(sizes))), part_pid=np.concatenate(pid),
                        part_a=np.concatenate([pt['a'] for pt in parts]), part_c=np.concatenate([pt['c'] for pt in parts]),
                        part_moff=np.concatenate(([0], np.cumsum(sizes**2))), part_m=np.concatenate([pt['m'].ravel() for pt in parts]),
                        part_k=np.concatenate([pt['k'].ravel() for pt in parts]),
                        part_T=np.concatenate([pt['T'].ravel() for pt in parts]), part_U=np.concatenate([pt['U'].ravel() for pt in parts]),
                        n_corners=nc, n_arcs=na, n_points=npts, arc_nsub=nsub, arc_off=aoff)


if __name__ == '__main__':
    main()
