"""the cap sampling's convergence for the foot's hard contact (the reviewer's item 2, 11-07). Run with CAP_N = 1000 / 2000
/ 4000 (the points per sphere of cap_quadrature): from the 6 mm state (the set rebuilt for this sampling), the clamp
pushed 6.0 -> 6.125 ... 6.5 mm (the second cap's entry; its lowest gap and multiplier per step) and then 7.0 mm (two
cells bearing, the third 0.36 mm off). Saves per delta: the free sites, reaction, energy, active cells, the gap of cell
5460, to out_inverse/static_contact/sampling_N<N>.pt. 'compare' as argv[1] (run with CAP_N=4000): loads the three
files, compares reaction / energy / displacement at 7.0 mm, the entry delta of cell 5460 (gap -> 0 interpolated), and
checks the coarse solutions' penetration on the fine candidate set"""
import os
import sys
from pathlib import Path
import numpy as np
import torch
MODE = sys.argv[1] if len(sys.argv) > 1 else 'run'
sys.argv = [sys.argv[0], '0.04', '3', '1', '40', '1']
sys.path.insert(0, str(Path(__file__).resolve().parent))
import static_contact_kkt as kk
from static_contact_kkt import *                                                      # noqa
N = int(os.environ.get('CAP_N', '2000')); OUT = kk.OUT
DELTAS = [6e-3, 6.125e-3, 6.25e-3, 6.375e-3, 6.5e-3, 7e-3]
if MODE == 'run':
    print('sampling N %d: %d candidates on the foot' % (N, len(kk.candF)), flush=True)
    z = torch.load(str(OUT/'multi_step2.pt')); xf = z['xf']
    A = torch.zeros(0, dtype=torch.long, device=dev); lamA = torch.zeros(0, dtype=f64, device=dev)
    rec = {}
    for delta in DELTAS:
        xfix = x_touch.clone(); xfix[:, 1] -= delta; prob = kk.KKT(xfix)
        xf, A, lamA, _, ok = kk.solve_tr(prob, xf, A, lamA, None, 'N %d delta %.3f mm' % (N, 1e3*delta), maxn=40)   # (the set is rebuilt for this sampling: a large change, the TR solver)
        react, cf = kk.final(prob, xf, A, lamA, None, 'N %d, delta %.3f mm, %s' % (N, 1e3*delta, 'converged' if ok else 'NOT converged'))
        x = prob.full(xf); e = float(W.grad(x)[0]); g = prob.cons(x)[0]
        gcell = torch.full((n,), 1e9, dtype=f64, device=dev).scatter_reduce(0, cl[kk.candF], g, 'amin')
        lam_cell = torch.zeros(n, dtype=f64, device=dev).index_add(0, cl[kk.candF][A], lamA) if len(A) else torch.zeros(n, dtype=f64, device=dev)
        cells_ = sorted(torch.unique(cl[kk.candF][A]).tolist()) if len(A) else []
        print('   -> cells %s; gap 5460 %.4f mm, lambda 5460 %.3e; reaction %.6f; E %.6e' % (cells_, 1e3*float(gcell[5460]), float(lam_cell[5460]), float(react[1]), e), flush=True)
        rec[delta] = dict(xf=xf.clone().cpu(), react=float(react[1]), E=e, cells=cells_, gap5460=float(gcell[5460]), lam5460=float(lam_cell[5460]), ok=ok, lam_cells={int(c): float(lam_cell[c]) for c in cells_})
        if not ok:
            break
    torch.save(dict(N=N, rec=rec), str(OUT/('sampling_N%d.pt' % N)))
    print('saved sampling_N%d.pt' % N, flush=True)
else:
    assert N == 4000, 'compare with CAP_N=4000 (the fine candidates for the penetration check)'
    R = {Nn: torch.load(str(OUT/('sampling_N%d.pt' % Nn)))['rec'] for Nn in (1000, 2000, 4000)}
    print('== state (a): delta 7.0 mm, two cells bearing')
    ref = R[4000][7e-3]
    for Nn in (1000, 2000, 4000):
        r = R[Nn][7e-3]
        dx = (r['xf']-ref['xf']).norm(dim=1)
        print('   N %4d: reaction %.6f (vs 4000: %+.2e, rel %.1e), E %.6e (rel %.1e), site displacement vs 4000: max %.4f mm median %.4f mm; cells %s, per-cell lambda %s' % (
            Nn, r['react'], r['react']-ref['react'], abs(r['react']-ref['react'])/ref['react'], r['E'], abs(r['E']-ref['E'])/ref['E'], 1e3*float(dx.max()), 1e3*float(dx.median()), r['cells'], {c: round(v, 6) for c, v in r['lam_cells'].items()}))
    print('== state (b): the second cap (5460) entering, 6.0 -> 6.5 mm')
    for Nn in (1000, 2000, 4000):
        ds = [d for d in DELTAS if d <= 6.5e-3 and d in R[Nn]]
        gaps = [R[Nn][d]['gap5460'] for d in ds]; lams = [R[Nn][d]['lam5460'] for d in ds]
        # entry: the last delta with gap > 0 and the first with the cell active, interpolated on the gap
        onset = None
        for i in range(len(ds)-1):
            if gaps[i] > 0 and gaps[i+1] <= 1e-12:
                # the gap closes ~ linearly before contact; extrapolate from the previous two gaps when available
                if i >= 1:
                    slope = (gaps[i]-gaps[i-1])/(ds[i]-ds[i-1]); onset = ds[i]-gaps[i]/slope
                else:
                    onset = ds[i]+gaps[i]/(gaps[i]+1e-30)*(ds[i+1]-ds[i])
                break
        print('   N %4d: gap 5460 (mm) %s; lambda 5460 %s; entry delta %s mm' % (
            Nn, ' '.join('%.4f' % (1e3*v) for v in gaps), ' '.join('%.2e' % v for v in lams), ('%.4f' % (1e3*onset)) if onset else '-'))
    print('== the coarse solutions on the fine candidate set (N 4000): penetration')
    for Nn in (1000, 2000):
        for d in (6.5e-3, 7e-3):
            xfix = x_touch.clone(); xfix[:, 1] -= d; prob = kk.KKT(xfix)
            g = prob.cons(prob.full(R[Nn][d]['xf'].to(dev)))[0]
            print('   N %d solution at %.1f mm: max penetration on the N 4000 points %.4f mm (points < 0: %d of %d)' % (Nn, 1e3*d, 1e3*float((-g).clamp(min=0).max()), int((g < 0).sum()), len(g)))
    print('done')
