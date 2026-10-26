"""Checks of the skin's lowest points (skin_contact.py), 2026-10-23, on the part and the new foam, for three plane
normals each:
  exact      rest, a rotation, and a uniform stretch + shear + rotation (x = A X + t, y = A Y + t): the skin is then the
             affine image of the sphere patches, and the lowest point of each region is known in closed form (the
             ellipsoid's lowest point if it lies in the region, else the lowest point of its arcs' ellipse arcs)
  sampling   a smooth non-uniform deformation of the bodies and the skin plus a rough skin (random skin offsets): no
             random sample of the region (1024, uniform on the cap holding it) lies lower than the computed lowest point
  gradient   d gap / d (x, y) at the held point against central differences of the whole computation, on the lowest
             regions
  timing     one lowest() over all regions"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

from experiments.pacnerf_dem_forward import build as dem_build
from free_surface.boundary_skin import BoundaryTerms, regions
from free_surface.skin_contact import SkinSurface

f64, dev = torch.float64, 'cuda'


def setup(case, skin, interp):
    if case == 'part':
        st = torch.load('output/test_part/run/model.pt', map_location='cpu', weights_only=True)
        p, r = st['points'].double().numpy(), torch.nn.functional.softplus(st['radii'].double(), beta=100).numpy()
        base = Path('output/test_part')
    else:
        base = Path('output/round01_exact')
        cells = np.load(base/'dem_exact_single'/'cells.npz'); p, r = cells['points'], cells['radii']
    g = dict(np.load(base/'dem_exact_single'/'geometry.npz')); bsk = dict(np.load(base/skin))
    m = dem_build(g, 23008.38, .3008, dev, 'noslip', .25, .3, f64, kinematics='exact', nodalvol=True, nonlinear=False, ogden_alpha=2.)
    terms = BoundaryTerms(m, g, p, r, bsk, lever='c')
    pt, rt = torch.as_tensor(p, dtype=f64, device=dev), torch.as_tensor(r, dtype=f64, device=dev)
    R, _ = regions(pt, rt)
    return terms, SkinSurface(terms, R, pt, rt, g['body_of_cell'], interp=interp)


def exact_lowest(S, A, t, n, c):
    """closed-form lowest point of every region under x -> A x + t"""
    w = A.T@n
    ge = (S.p@A.T+t)@n-c-S.r*w.norm()
    me = (-w/w.norm()).expand_as(S.p)
    gi = torch.where(S.inside(me), ge, torch.full_like(ge, float('inf')))
    al, be = S.rho*(S.u@w), S.rho*(S.v@w)
    g0 = (S.cc@A.T+t)@n-c
    th = torch.atan2(-be, -al)
    tm = .5*(S.a0+S.a1); th = tm+torch.remainder(th-tm+math.pi, 2*math.pi)-math.pi
    cand = torch.stack((torch.where((th >= S.a0) & (th <= S.a1), th, S.a0), S.a0, S.a1), -1)
    ga = (g0[..., None]+al[..., None]*torch.cos(cand)+be[..., None]*torch.sin(cand)).amin(-1)
    ga = torch.where(S.vb, ga, torch.full_like(ga, float('inf'))).amin(1)
    return torch.minimum(gi, ga), gi < ga


def samples(S, N=1024, seed=1):
    """random samples of every region, independent of lowest()'s starts: uniform on the spherical cap around m0 that holds
    the region's corners and arc midpoints (kept where inside)"""
    g = torch.Generator(device=dev).manual_seed(seed)
    ends = torch.cat((S.arcpt(S.a0), S.arcpt(.5*(S.a0+S.a1))), 1)
    vb = torch.cat((S.vb, S.vb), 1)
    ang = torch.where(vb, torch.acos(((ends*S.R.m0[:, None]).sum(-1)).clamp(-1, 1)), torch.zeros_like(vb, dtype=f64)).amax(1)
    cz = 1-(1-torch.cos(ang))[:, None]*torch.rand(S.P, N, generator=g, dtype=f64, device=dev)
    ps = 2*math.pi*torch.rand(S.P, N, generator=g, dtype=f64, device=dev)
    sz = (1-cz**2).clamp(min=0).sqrt()
    s = cz[..., None]*S.R.m0[:, None]+sz[..., None]*(torch.cos(ps)[..., None]*S.R.e1[:, None]+torch.sin(ps)[..., None]*S.R.e2[:, None])
    return s, torch.ones(S.P, N, dtype=torch.bool, device=dev)


def sampled_min(S, x, y, n, c, chunk=24):
    s, ok = samples(S)
    best = torch.full((S.P,), float('inf'), dtype=f64, device=dev)
    for q0 in range(0, s.shape[1], chunk):
        sq = s[:, q0:q0+chunk]
        with torch.no_grad():
            W = S.weights(sq)
            g = S.points(x, y, sq, W)@n-c
            ins = torch.stack([S.inside(sq[:, j]) for j in range(sq.shape[1])], 1) & ok[:, q0:q0+chunk]
        best = torch.minimum(best, torch.where(ins, g, torch.full_like(g, float('inf'))).amin(1))
    return best


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--case', choices=('part', 'foam'), default='part')
    a.add_argument('--skin', default='boundary_skin.npz')
    a.add_argument('--interp', default='shepard', help="the skin file's interpolation: part boundary_skin.npz 'shepard' (Pi), foam boundary_skin.npz 'shepard0'")
    args = a.parse_args()
    t0 = time.time()
    terms, S = setup(args.case, args.skin, args.interp)
    X, Y = terms.X, terms.Y
    size = float((X.max(0).values-X.min(0).values).max()); cen = X.mean(0)
    rmed = float(S.r.median())
    print('[%s] %s (%s): %d bodies, %d skin points, %d regions (%d on sites with islands); setup %.0f s' % (
        args.case, args.skin, args.interp, len(X), len(Y), S.P, 0 if S.multi is None else len(S.multi[0]), time.time()-t0), flush=True)
    gen = torch.Generator(device=dev).manual_seed(0)
    normals = torch.nn.functional.normalize(torch.randn(3, 3, generator=gen, dtype=f64, device=dev), dim=1)
    def rot(v):
        th = v.norm(); k = v/th; K = torch.tensor([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], dtype=f64, device=dev)
        return torch.eye(3, dtype=f64, device=dev)+torch.sin(th)*K+(1-torch.cos(th))*K@K
    Rq = rot(torch.tensor([.4, -1.1, .7], dtype=f64, device=dev))
    U = torch.tensor([[.82, .07, -.05], [.07, 1.13, .09], [-.05, .09, 1.24]], dtype=f64, device=dev)
    states = dict(rest=torch.eye(3, dtype=f64, device=dev), rotation=Rq, stretch=Rq@U)
    # exact states
    for nm, A in states.items():
        t = torch.tensor([.01, -.02, .03], dtype=f64, device=dev)*size
        x, y = (X-cen)@A.T+cen+t, (Y-cen)@A.T+cen+t
        for k, n in enumerate(normals):
            c = float(((x@n).min()))
            t1 = time.time()
            low = S.lowest(x, y, n, c)
            torch.cuda.synchronize(); dtl = time.time()-t1
            ref, rin = exact_lowest(S, A, t+cen-cen@A.T, n, c)
            err = (low['gap']-ref).abs()/S.r
            print('[%s] exact %-8s normal %d: |gap - exact| / r max %.1e (median %.1e); lowest point inside %d / on arcs %d (exact: %d inside), kind '
                  'mismatches %d; Newton %d iterations, not converged %d (held at the boundary %d), |grad| / scale max over converged %.1e; %.2f s' % (
                      args.case, nm, k, float(err.max()), float(err.median()), int((low['kind'] == 0).sum()), int((low['kind'] == 1).sum()), int(rin.sum()),
                      int(((low['kind'] == 0) != rin).sum()), low['iterations'], int((~low['converged']).sum()), int(low['stuck'].sum()),
                      float(low['grad_rel'][low['converged']].max()), dtl), flush=True)
    # non-uniform deformation + rough skin
    kv = torch.randn(3, 3, generator=gen, dtype=f64, device=dev)*2*math.pi/size
    amp = torch.randn(3, 3, generator=gen, dtype=f64, device=dev)*.03*size
    ph = torch.rand(3, generator=gen, dtype=f64, device=dev)*2*math.pi
    field = lambda Z: (torch.sin(Z@kv.T+ph)@amp.T)
    x = X+field(X)
    y = Y+field(Y)+torch.randn(Y.shape, generator=gen, dtype=f64, device=dev)*.05*rmed
    for k, n in enumerate(normals):
        c = float((x@n).min())
        low = S.lowest(x, y, n, c)
        sm = sampled_min(S, x, y, n, c)
        d = (sm-low['gap'])/S.r
        fin = torch.isfinite(sm)
        print('[%s] deformed normal %d: random samples (1024 per region) min - lowest, / r: min %.1e (negative = a sample lower than the lowest point), median %.1e, max %.1e '
              'over %d regions with inside samples; lowest point inside %d / on arcs %d; Newton not converged %d (held %d), and of those lower than '
              'their arcs %d' % (args.case, k, float(d[fin].min()), float(d[fin].median()), float(d[fin].max()), int(fin.sum()), int((low['kind'] == 0).sum()),
                                 int((low['kind'] == 1).sum()), int((~low['converged']).sum()), int(low['stuck'].sum()),
                                 int((~low['converged'] & (low['g_interior'] < low['g_arc'])).sum())), flush=True)
        # gradient: the 20 lowest regions, a random direction of (x, y)
        sel = torch.argsort(low['gap'])[:20]
        dx = torch.randn(X.shape, generator=gen, dtype=f64, device=dev); dy = torch.randn(Y.shape, generator=gen, dtype=f64, device=dev)
        _, jv = torch.func.jvp(lambda x_, y_: S.gap(x_, y_, n, c, low), (x, y), (dx, dy))
        h = 1e-6*size
        gp = S.lowest(x+h*dx, y+h*dy, n, c)['gap']; gm = S.lowest(x-h*dx, y-h*dy, n, c)['gap']
        fd = (gp-gm)/(2*h)
        rel = ((jv-fd).abs()/fd.abs().clamp(min=1e-30))[sel]
        print('[%s] gradient normal %d (20 lowest regions, kinds %s): |autodiff - central difference| / |central difference| max %.1e, median %.1e' % (
            args.case, k, low['kind'][sel].tolist(), float(rel.max()), float(rel.median())), flush=True)
    print('[%s] done (%.0f s)' % (args.case, time.time()-t0), flush=True)


if __name__ == '__main__':
    main()
