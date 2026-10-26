"""recut.Recut (all GPU) against the CPU route (qhull structure, scipy solve: retess_check.py's) along the same path:
the sites moved from rest to a saved step's bodies in --path pieces, each piece's radii solved from the last for
V = V0 (1 + s (J - 1)) (J the model's volume ratio there); first the structure_gpu packing against _pack at rest;
per piece the Newton iterations and the GPU route's time; at the end both routes' radii and volumes compared.
  python free_surface/recut_check.py --out output/round01_pruned_exact --dt 2.0833333e-4 --tols 1e-4 --beta 0.01 --integrator midpoint --state output/round01_pruned_exact/skin_drop_state_step900.npz"""
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import torch

from free_surface.gpu_geometry import Geometry, structure_triangulation
from free_surface.recut import Recut
from free_surface.skin_contact_check import f64, dev
from free_surface.skin_dynamics_check import build_model, parser


def sync():
    torch.cuda.synchronize()


def main():
    a = parser()
    a.add_argument('--state', type=Path, required=True)
    a.add_argument('--path', type=int, default=50)
    a.add_argument('--cpu', type=int, default=1, help='also the CPU route (qhull + scipy), about 1.5 min')
    args = a.parse_args()
    out = Path(args.out)
    cells = np.load(out/'dem_exact_single'/'cells.npz')
    p0 = torch.as_tensor(cells['points'], dtype=f64, device=dev); r0 = torch.as_tensor(cells['radii'], dtype=f64, device=dev)
    b = build_model(args)
    W = b['W']
    z0 = np.load(args.state)
    for k_, r_ in W.tref.items():
        r_.copy_(torch.as_tensor(z0['ref_'+k_], device=dev))
    p1 = torch.as_tensor(z0['xm1'], dtype=f64, device=dev)
    W.grad(p1, torch.as_tensor(z0['ym1'], dtype=f64, device=dev))
    J = torch.exp(W.tlj).double()
    n = len(p0)
    R = Recut(n)

    # the GPU packing against _pack at rest: identical values
    o = R.A(p0, r0)
    S_g, S_c = R.A.structure_gpu(o, n), R.A.structure(o, n)
    G_g, G_c = Geometry(p0, r0, S_g), Geometry(p0, r0, S_c)
    V0 = G_g.V.detach()
    print('packing on the GPU against _pack: pairs equal %s; volumes max relative difference %.1e, exposed areas %.1e' % (
        bool(torch.equal(S_g['pairs'], S_c['pairs'])), float(((G_g.V-G_c.V).abs()/G_c.V).max()), float((G_g.patch_A-G_c.patch_A).abs().max())), flush=True)
    for _ in range(3):                                                             # warm
        R.geometry(p0, r0)
    sync(); t0 = time.perf_counter()
    for _ in range(10):
        R.geometry(p0, r0)
    sync(); tg = (time.perf_counter()-t0)/10
    print('one geometry (BVH, clipping, packing, values) on the GPU: %.1f ms' % (1e3*tg), flush=True)

    # the path on the GPU route
    r = r0.clone(); its = []; times = []
    R.stats = dict(geometry=0, cg=0, overlaps=0)
    for k in range(1, args.path+1):
        sk = k/args.path
        pk = p0+sk*(p1-p0); Vt = V0*(1+sk*(J-1))
        sync(); t1 = time.perf_counter()
        r, G, S, it, nrep = R.solve_radii(pk, r, Vt)
        sync(); times.append(time.perf_counter()-t1); its.append(it)
        if nrep:
            print('   piece %d: %d hidden sites repaired' % (k, nrep), flush=True)
    rg, Vg = r.clone(), G.V.detach()
    print('GPU route along %d pieces: Newton %s per piece, geometry evaluations %.1f, overlap builds %.1f and CG iterations %.0f per piece; time per piece median %.0f ms (max %.0f); '
          'final max |V / target - 1| %.1e, radii r / r0 - 1 p0 %+.3e p50 %+.3e p100 %+.3e' % (
              args.path, np.bincount(its).tolist(), R.stats['geometry']/args.path, R.stats['overlaps']/args.path, R.stats['cg']/args.path, 1e3*np.median(times), 1e3*max(times),
              float((Vg/Vt-1).abs().max()), float((rg/r0-1).min()), float((rg/r0-1).median()), float((rg/r0-1).max())), flush=True)

    # the worst cell: its volume against the size of the terms it sums (V = (sum_walls A h + r A_patch) / 3, h of either sign)
    e = (Vg/Vt-1).abs(); w_ = int(e.argmax())
    keep = G.wall_A > 0
    on = keep & ((G.i == w_) | (G.j == w_))
    hh = torch.where(G.i[on] == w_, G.hi[on], G.hj[on])
    terms = float((G.wall_A[on]*hh).abs().sum()+rg[w_]*G.patch_A[w_])/3
    print('worst cell %d: |V / target - 1| %.1e; V %.3e (target %.3e, %.1e of the median cell), sum of its terms\' sizes %.3e: ratio %.1e, '
          'so its rounding about %.0e relative; walls %d, of them with the site behind (h < 0) %d' % (
              w_, float(e[w_]), float(Vg[w_]), float(Vt[w_]), float(Vt[w_]/Vt.median()), terms, terms/float(Vg[w_]), 1.1e-16*terms/float(Vg[w_]),
              int(on.sum()), int((hh < 0).sum())), flush=True)

    if args.cpu:                                                                   # the CPU route (retess_check.py's), same path
        def geo_cpu(pk, r):
            S = structure_triangulation(pk.cpu().numpy(), r.cpu().numpy())
            return Geometry(pk, r, S), S

        def hess(G, r):
            keep = G.wall_A > 0
            i, j = G.i[keep], G.j[keep]
            L = (G.p[j]-G.p[i]).norm(dim=1)
            c = (G.wall_A[keep]/(2*L)).detach()
            d = torch.zeros(n, dtype=f64, device=dev).index_add(0, i, c).index_add(0, j, c)+G.patch_A.detach()/(2*r)
            I, Jj, C = i.cpu().numpy(), j.cpu().numpy(), c.cpu().numpy()
            return sp.coo_matrix((np.r_[-C, -C, d.cpu().numpy()], (np.r_[I, Jj, np.arange(n)], np.r_[Jj, I, np.arange(n)])), shape=(n, n)).tocsc()
        r = r0.clone(); t1 = time.perf_counter()
        for k in range(1, args.path+1):
            sk = k/args.path
            pk = p0+sk*(p1-p0); Vt = V0*(1+sk*(J-1))
            G, S = geo_cpu(pk, r)
            res = Vt-G.V.detach(); it = 0
            while float((res/Vt).abs().max()) > 1e-9 and it < 30:
                it += 1
                dw = torch.as_tensor(spl.spsolve(hess(G, r), res.cpu().numpy()), dtype=f64, device=dev)
                w = r*r
                if float((dw/w).abs().max()) < 1e-10:
                    break
                st, nr = 1., float(res.norm())
                while True:
                    wt = w+st*dw
                    Gt, St = geo_cpu(pk, torch.sqrt(wt)); rt = Vt-Gt.V.detach()
                    if int(St['hidden'].sum()) == 0 and bool((Gt.V > 0).all()) and float(rt.norm()) <= (1-st/2)*nr:
                        break
                    st *= .5
                r, G, res = torch.sqrt(wt), Gt, rt
        print('CPU route: %.0f s for the path; radii GPU against CPU max relative difference %.1e; volumes %.1e' % (
            time.perf_counter()-t1, float((rg/r-1).abs().max()), float((Vg/G.V.detach()-1).abs().max())), flush=True)


if __name__ == '__main__':
    main()
