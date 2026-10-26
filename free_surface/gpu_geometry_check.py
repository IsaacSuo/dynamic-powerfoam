"""Checks of free_surface/gpu_geometry.py (2026-10-22).
  two balls   cap area, lens volume, wall area, int n dA, int n n^T dA in closed form.
  part        the two structures (triangulation links / all overlaps) against each other; against the exact CPU
              geometry (part_geometry.py: dem_exact_single, sphere_skin.npz): walls, corners, arc angle per circle, patch
              area / int n / int n n^T, volumes; every cell closed (sum of wall area vectors + int n dA = 0).
  gradients   directional derivatives of six weighted sums (volumes, patch area, int n, int n n^T, wall areas, wall
              first moments) along random site and radius directions against central differences.
  new foam    timing of structure (CPU), forward, backward; memory; closure; against its DEM (64-gon walls, sampled
              sphere: agreement only to ~1e-3 expected)."""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

from free_surface.gpu_geometry import Geometry, structure_overlaps, structure_triangulation

F = torch.nn.functional
dev, f64 = 'cuda', torch.float64


def load(path):
    st = torch.load(path, map_location='cpu', weights_only=True)
    return st['points'].double().to(dev), F.softplus(st['radii'].double(), beta=100).to(dev)


def keymap(P, n):
    P = np.asarray(P); return {int(a)*n+int(b): k for k, (a, b) in enumerate(P)}


def closure(G):
    vec = G.wall_A[:, None]*G.nn
    s = G.patch_n.clone().index_add(0, G.i, vec).index_add(0, G.j, -vec)
    tot = G.patch_A.clone().index_add(0, G.i, G.wall_A).index_add(0, G.j, G.wall_A)
    m = tot > 0
    return float((s.norm(dim=1)[m]/tot[m]).max().detach())


def two_balls():
    p = torch.tensor([[0., 0, 0], [.013, .004, -.002]], dtype=f64, device=dev); r = torch.tensor([.01, .007], dtype=f64, device=dev)
    S = structure_overlaps(p, r)
    G = Geometry(p, r, S)
    L = float((p[1]-p[0]).norm()); n = ((p[1]-p[0])/L).cpu().numpy()
    out = []
    for s, sg in ((0, 1.), (1, -1.)):
        rr = float(r[s]); h = (L*L+float(r[0])**2-float(r[1])**2)/(2*L) if s == 0 else L-(L*L+float(r[0])**2-float(r[1])**2)/(2*L)
        c = h/rr; a = sg*n
        A = 2*math.pi*rr*rr*(1+c)                                   # sphere minus the cap beyond the plane
        nvec = -math.pi*rr*rr*(1-c*c)*a                             # = - int over the cap
        aa = 2*math.pi/3*(1+c**3); tt = 2*math.pi/3+math.pi*c-math.pi*c**3/3
        nn = rr*rr*(tt*np.eye(3)+(aa-tt)*np.outer(a, a))
        hc = rr-h; V = 4/3*math.pi*rr**3-math.pi*hc*hc*(3*rr-hc)/3
        out.append(max(abs(float(G.patch_A[s])-A)/(4*math.pi*rr*rr), np.abs(G.patch_n[s].cpu().numpy()-nvec).max()/(4*math.pi*rr*rr),
                       np.abs(G.patch_nn[s].cpu().numpy()-nn).max()/(4*math.pi*rr*rr), abs(float(G.V[s])-V)/V))
    rho2 = float(r[0])**2-((L*L+float(r[0])**2-float(r[1])**2)/(2*L))**2
    print('two balls: |patch area, int n, int n n^T, volume - closed form| / scale max %.1e; wall area / (pi rho^2) - 1 = %.1e' % (
        max(out), float(G.wall_A[0])/(math.pi*rho2)-1), flush=True)


def part(path):
    p, r = load(path/'run'/'model.pt'); n = len(p); s = float(r.median())
    pn, rn = p.cpu().numpy(), r.cpu().numpy()
    t = time.time(); St = structure_triangulation(pn, rn); tt = time.time()-t
    So = structure_overlaps(p, r)
    Gt, Go = Geometry(p, r, St), Geometry(p, r, So)
    sph = 4*math.pi*r**2
    # the two structures
    mt, mo = keymap(Gt.P.cpu(), n), keymap(Go.P.cpu(), n)
    com = [k for k in mt if k in mo]
    wa = max(abs(float(Gt.wall_A[mt[k]]-Go.wall_A[mo[k]])) for k in com)/s**2
    only_o = [mo[k] for k in mo if k not in mt]
    print('part: %d sites; structure from the triangulation %.3f s: %d circle pairs (links up to %d); from overlaps %d circle pairs (up to %d common neighbours)' % (
        n, tt, len(Gt.P), Gt.Cn.shape[1], len(Go.P), Go.Cn.shape[1]), flush=True)
    print('  triangulation vs overlaps: walls |dA| / s^2 max %.1e, overlap-only pairs wall area max %.1e s^2; patch area %.1e, int n %.1e, int nn %.1e (of the sphere area); '
          'volume %.1e (of s^3); arcs %d vs %d' % (
              wa, float(Go.wall_A[only_o].abs().max())/s**2 if only_o else 0., float(((Gt.patch_A-Go.patch_A).abs()/sph).max()),
              float(((Gt.patch_n-Go.patch_n).norm(dim=1)/sph).max()), float(((Gt.patch_nn-Go.patch_nn).flatten(1).norm(dim=1)/sph).max()),
              float((Gt.V-Go.V).abs().max())/s**3, len(Gt.arc_pair), len(Go.arc_pair)), flush=True)
    G = Gt
    # against the exact CPU geometry
    dem = np.load(path/'dem_exact_single'/'cells.npz'); sk = np.load(path/'sphere_skin.npz')
    cp = np.sort(dem['pairs'], axis=1); mc = keymap(cp, n); mg = keymap(G.P.cpu(), n)
    A = G.wall_A.detach().cpu().numpy(); C = G.wall_c.detach().cpu().numpy()
    pos = [k for k in mg if A[mg[k]] > 0]
    miss = [k for k in mc if k not in mg]; extra = [k for k in pos if k not in mc]
    both = [k for k in mc if k in mg and A[mg[k]] > 0]
    zero = [k for k in mc if k in mg and A[mg[k]] == 0]                      # positive on the CPU, zero here
    dA = np.array([abs(A[mg[k]]-dem['areas'][mc[k]])/dem['areas'][mc[k]] for k in both])
    dAs = np.array([abs(A[mg[k]]-dem['areas'][mc[k]]) for k in both])/s**2
    dC = np.array([np.linalg.norm(C[mg[k]]-dem['contacts'][mc[k]]) for k in both])/s
    print('  walls: GPU with area > 0 %d, CPU %d; CPU walls missing on the GPU %d, GPU walls with area > 0 not on the CPU %d (largest %.1e s^2); '
          'CPU walls with area > 0 that are 0 here %d (CPU area max %.1e s^2) | both > 0: area |rel diff| median %.1e max %.1e (max %.1e s^2), centroid max %.1e s' % (
              len(pos), len(cp), len(miss), len(extra), max([A[mg[k]] for k in extra], default=0.)/s**2,
              len(zero), max([dem['areas'][mc[k]] for k in zero], default=0.)/s**2, np.median(dA), dA.max(), dAs.max(), dC.max()), flush=True)
    X = torch.cat((G.arc_points(G.arc_t0), G.arc_points(G.arc_t1))).detach().cpu().numpy()
    Cc = sk['corners']
    d1 = np.sqrt(((X[:, None]-Cc[None])**2).sum(-1)); dgc = d1.min(1).max()/s; dcg = d1.min(0).max()/s
    def on_sphere(x, k):
        kk = torch.as_tensor(k, device=dev)
        return ((torch.as_tensor(x, device=dev)-p[kk])**2).sum(1)/r[kk]**2-1
    ea = G.arc_k0 >= 0
    x0 = G.arc_points(G.arc_t0)[ea].detach(); x1 = G.arc_points(G.arc_t1)[ea].detach()
    pi_, pj_ = G.i[G.arc_pair][ea], G.j[G.arc_pair][ea]
    res = max(float(on_sphere(x, k).abs().max()) for x in (x0, x1) for k in (pi_, pj_))
    res = max(res, float(on_sphere(x0, G.arc_k0[ea]).abs().max()), float(on_sphere(x1, G.arc_k1[ea]).abs().max()))
    ccpu = max(float(on_sphere(Cc, sk['corner_tri'][:, q]).abs().max()) for q in range(3))
    print('  corners on their three spheres: |x - p|^2 / r^2 - 1 max: GPU %.1e, CPU %.1e' % (res, ccpu), flush=True)
    # uncovered angle per circle
    arc_ij = sk['arc_ij']; th = sk['arc_theta']
    tot_c = {}
    for (i, j, _, _), (t0, t1) in zip(arc_ij, th):
        tot_c[int(i)*n+int(j)] = tot_c.get(int(i)*n+int(j), 0.)+(t1-t0)
    tg = torch.zeros(len(G.P), dtype=f64, device=dev).index_add(0, G.arc_pair, G.arc_t1-G.arc_t0).cpu().numpy()
    dth = max(abs(tg[mg[k]]-v) for k, v in tot_c.items())
    gk = [k for k in mg if tg[mg[k]] > 0 and k not in tot_c]
    print('  arcs: GPU %d, CPU %d; corners: every arc end on a CPU corner within %.1e s, every CPU corner reached within %.1e s; uncovered angle per circle |diff| max %.1e rad, '
          'circles with arcs only on the GPU %d' % (len(G.arc_pair), len(arc_ij), dgc, dcg, dth, len(gk)), flush=True)
    rr = r.cpu().numpy(); sp = 4*np.pi*rr**2
    pa = np.abs(G.patch_A.detach().cpu().numpy()-sk['patch_area'])/sp
    pv = np.linalg.norm(G.patch_n.detach().cpu().numpy()-sk['patch_n'], axis=1)/sp
    pm = np.linalg.norm((G.patch_nn.detach().cpu().numpy()-sk['patch_nn']).reshape(n, 9), axis=1)/sp
    V = G.V.detach().cpu().numpy(); Vc = dem['masses']
    print('  patches (of the sphere area; CPU: fan quadrature agreeing with Gauss-Bonnet to 1.8e-13): area max %.1e, int n max %.1e, int n n^T max %.1e | '
          'antipode of the pole to the boundary min %.2f rad | volumes |rel diff| median %.1e max %.1e, of s^3 max %.1e, total %.10e vs %.10e | closure %.1e' % (
              pa.max(), pv.max(), pm.max(), float(G.pole_gap[torch.as_tensor(sk['patch_area'] > 0, device=dev)].min()),
              np.median(np.abs(V/Vc-1)), np.abs(V/Vc-1).max(), (np.abs(V-Vc)/s**3).max(), V.sum(), Vc.sum(), closure(G)), flush=True)
    return p, r, St, s


def gradcheck(p, r, S, s, h=1e-6, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    n = len(p)
    def terms(pp, rr):
        G = Geometry(pp, rr, S)
        m = len(G.P)
        W = lambda *shape: torch.randn(*shape, generator=torch.Generator(device=dev).manual_seed(hash(shape) % 2**31), dtype=f64, device=dev)
        mom = G.wall_A[:, None]*(G.wall_c-pp[G.i])
        return G, dict(volume=(W(n, 1)[:, 0]*G.V).sum()/s**3, area=(W(n, 2)[:, 0]*G.patch_A).sum()/s**2, int_n=(W(n, 3)*G.patch_n).sum()/s**2,
                       int_nn=(W(n, 3, 3)*G.patch_nn).sum()/s**2, wall_area=(W(m, 1)[:, 0]*G.wall_A).sum()/s**2, wall_moment=(W(m, 3)*mom).sum()/s**3)
    dp = torch.randn(n, 3, generator=g, dtype=f64, device=dev)*s; dr = torch.randn(n, generator=g, dtype=f64, device=dev)*s*.1
    pp = p.clone().requires_grad_(True); rr = r.clone().requires_grad_(True)
    G0, T = terms(pp, rr)
    res = []
    for name, val in T.items():
        gp, gr = torch.autograd.grad(val, (pp, rr), retain_graph=True)
        for tag, ad, (pa, ra, pb, rb) in (('sites', float((gp*dp).sum()), (p+h*dp, r, p-h*dp, r)), ('radii', float((gr*dr).sum()), (p, r+h*dr, p, r-h*dr))):
            Ga, Ta = terms(pa, ra); Gb, Tb = terms(pb, rb)
            fd = float(Ta[name]-Tb[name])/(2*h)
            same = len(Ga.arc_pair) == len(Gb.arc_pair) == len(G0.arc_pair) and bool((Ga.pole_idx == Gb.pole_idx).all() & (Ga.pole_idx == G0.pole_idx).all())
            res.append('%s/%s %.1e%s' % (name, tag, abs(ad-fd)/max(abs(fd), 1e-300), '' if same else ' (structure changed)'))
    print('  gradients, |autograd - central difference| / |central difference| (step %.0e s): %s' % (h, ', '.join(res)), flush=True)


def foam(path, dem_dir):
    p, r = load(path); n = len(p); s = float(r.median())
    t = time.time(); S = structure_triangulation(p.cpu().numpy(), r.cpu().numpy()); tt = time.time()-t
    fw, bw = [], []
    for _ in range(6):
        pp = p.clone().requires_grad_(True); rr = r.clone().requires_grad_(True)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t = time.time()
        G = Geometry(pp, rr, S)
        torch.cuda.synchronize(); fw.append(time.time()-t); t = time.time()
        (G.V.sum()+G.patch_A.sum()*s+G.patch_n.sum()*s+G.patch_nn.sum()*s+G.wall_A.sum()*s).backward()
        torch.cuda.synchronize(); bw.append(time.time()-t)
    mem = torch.cuda.max_memory_allocated()/2**20
    print('new foam: %d sites, hidden %d; structure (CPU triangulation) %.2f s; circle pairs %d, links up to %d, neighbours up to %d; arcs %d | '
          'forward %.1f ms, backward %.1f ms (median of the last 5), peak GPU memory %.0f MB' % (
              n, int(S['hidden'].sum()), tt, len(G.P), G.Cn.shape[1], S['nbrs'].shape[1], len(G.arc_pair),
              1e3*np.median(fw[1:]), 1e3*np.median(bw[1:]), mem), flush=True)
    c = np.load(dem_dir/'cells.npz'); src = c['source_index']
    V = G.V.detach().cpu().numpy(); A = G.wall_A.detach().cpu().numpy()
    mg = keymap(G.P.cpu(), n)
    dp_ = np.sort(src[c['pairs']], axis=1)
    ad = np.array([A[mg[int(a)*n+int(b)]] if int(a)*n+int(b) in mg else 0. for a, b in dp_])
    rel = np.abs(ad/c['areas']-1)
    mat = np.zeros(n, bool); mat[src] = True
    print('  closure %.1e; antipode of the pole to the boundary min %.2f rad; material volume %.6f (DEM %.6f), non-material sites hold %.2e | '
          'against the DEM: cell volume |rel diff| median %.1e, 99%% %.1e; wall area |rel diff| median %.1e, 99%% %.1e (DEM walls missing here %d)' % (
              closure(G), float(G.pole_gap[G.patch_A > 0].min()), V[mat].sum(), c['masses'].sum(), V[~mat].sum(),
              np.median(np.abs(V[src]/c['masses']-1)), np.quantile(np.abs(V[src]/c['masses']-1), .99), np.median(rel), np.quantile(rel, .99),
              int(sum(int(a)*n+int(b) not in mg for a, b in dp_))), flush=True)


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--part', type=Path, default=Path('output/test_part'))
    a.add_argument('--foam', type=Path, default=Path('/home/fangsuo/py/dynamic-powerfoam/.claude/worktrees/thin-surface/output/pacnerf_elastic0_f0_round01_8k/model.pt'))
    a.add_argument('--foam_dem', type=Path, default=Path('/home/fangsuo/py/dynamic-powerfoam/.claude/worktrees/thin-surface/output/pacnerf_dem_round01_8k/elastic_0_zup'))
    args = a.parse_args()
    two_balls()
    p, r, S, s = part(args.part)
    gradcheck(p, r, S, s)
    foam(args.foam, args.foam_dem)


if __name__ == '__main__':
    main()
