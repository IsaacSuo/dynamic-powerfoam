"""the reviewer's two checks on the foot's KKT solution (11-07):
 1 the working-set cut: from the saved 3 mm compression and the stick states, re-solve with RANK_TOL 1e-2 / 1e-3 / 1e-4
   and with every candidate point duplicated; compare the free sites' displacement, total reaction, energy, residuals.
 2 multi-point contact and release: the per-cell lowest gaps at 3 mm; the clamp pushed until a second (and third) cap
   touches, then unloaded until one leaves; equilibrium and the set at every step, the reaction against delta.
args: none (uses static_contact_kkt's foot, H 0.04)"""
import sys
from pathlib import Path
import numpy as np
import torch
SKIP1_FLAG = len(sys.argv) > 1 and sys.argv[1] == 'skip1'
sys.argv = [sys.argv[0], '0.04', '3', '1', '40', '1']
sys.path.insert(0, str(Path(__file__).resolve().parent))
import static_contact_kkt as kk
from static_contact_kkt import *                                                      # noqa
OUT = kk.OUT


def state(xf, lamA, A):
    x = kk.KKT(xfix).full(xf); e = float(W.grad(x)[0])
    return dict(xf=xf.clone(), E=e, react=float(lamA.sum()), nA=len(A), cells=sorted(torch.unique(cl[kk.candF][A]).tolist()) if len(A) else [])


# ================================================================ 1: the cut and duplicated points
SKIP1 = SKIP1_FLAG
z = torch.load(str(OUT/'kkt_d3.0.pt')); xf0, A0, lam0 = z['xf'], z['A'], z['lamA']
xfix = x_touch.clone(); xfix[:, 1] -= 3e-3
if not SKIP1:
  print('== 1. the working-set cut, at delta 3 mm', flush=True)
  ref = None
  for tol in (1e-2, 1e-3, 1e-4):
      kk.RANK_TOL = tol; prob = kk.KKT(xfix)
      xf, A, lamA, _, ok = kk.solve(prob, xf0.clone(), A0.clone(), lam0.clone(), None, 'tol %g' % tol, maxn=25)
      react, cf = kk.final(prob, xf, A, lamA, None, 'tol %g, %s' % (tol, 'converged' if ok else 'NOT converged'))
      st_ = state(xf, lamA, A)
      if ref is None:
          ref = st_
      print('   -> |A| %d on cells %s; against tol 1e-2: max site displacement difference %.3e mm, reaction diff %.2e (rel %.1e), E diff %.2e (rel %.1e)' % (
          st_['nA'], st_['cells'], 1e3*float((st_['xf']-ref['xf']).norm(dim=1).max()), st_['react']-ref['react'], abs(st_['react']-ref['react'])/abs(ref['react']), st_['E']-ref['E'], abs(st_['E']-ref['E'])/abs(ref['E'])), flush=True)
  kk.RANK_TOL = 1e-2
  # duplicated candidates
  cand_orig = kk.candF.clone(); kk.candF = torch.cat([cand_orig, cand_orig]); kk.QF = len(kk.candF)
  prob = kk.KKT(xfix); A_dup = torch.cat([A0, A0+len(cand_orig)]); lam_dup = torch.cat([lam0, torch.zeros_like(lam0)])
  xf, A, lamA, _, ok = kk.solve(prob, xf0.clone(), A_dup, lam_dup, None, 'duplicated', maxn=25)
  react, cf = kk.final(prob, xf, A, lamA, None, 'duplicated points, %s' % ('converged' if ok else 'NOT converged'))
  st_ = state(xf, lamA, A)
  print('   -> |A| %d (over %d candidates); against the original: max site displacement difference %.3e mm, reaction diff %.2e (rel %.1e), E diff %.2e (rel %.1e)' % (
      st_['nA'], kk.QF, 1e3*float((st_['xf']-ref['xf']).norm(dim=1).max()), st_['react']-ref['react'], abs(st_['react']-ref['react'])/abs(ref['react']), st_['E']-ref['E'], abs(st_['E']-ref['E'])/abs(ref['E'])), flush=True)
  kk.candF = cand_orig; kk.QF = len(cand_orig)
# the stick state
print('== 1b. the cut, at the stick state', flush=True)
z = torch.load(str(OUT/'kkt_stick.pt')); xfS, AS, lamS, lamTS, anchors = z['xf'], z['A'], z['lamA'], z['lamT'], z['anchors']
xfixS = x_touch.clone(); xfixS[:, 1] -= 3e-3; xfixS[:, 0] += 1e-3
refS = None
for tol in (() if SKIP1 else (1e-2, 1e-3, 1e-4)):
    kk.RANK_TOL = tol; prob = kk.KKT(xfixS, anchors)
    xf, A, lamA, lamT, ok = kk.solve(prob, xfS.clone(), AS.clone(), lamS.clone(), lamTS.clone(), 'stick tol %g' % tol, maxn=25)
    react, cf = kk.final(prob, xf, A, lamA, lamT, 'stick tol %g, %s' % (tol, 'converged' if ok else 'NOT converged'))
    if refS is None:
        refS = (xf.clone(), react.clone())
    print('   -> against tol 1e-2: max site displacement difference %.3e mm, reaction vector diff %.2e' % (1e3*float((xf-refS[0]).norm(dim=1).max()), float((react-refS[1]).norm())), flush=True)
kk.RANK_TOL = 1e-2

# ================================================================ 2: multi-point contact and release
print('\n== 2. multi-point contact and release', flush=True)
prob = kk.KKT(xfix); g, _, _ = prob.cons(prob.full(xf0))
gcell = torch.full((n,), 1e9, dtype=f64, device=dev).scatter_reduce(0, cl[candF], g, 'amin')
order = torch.argsort(gcell)[:6]
print('   at delta 3 mm: the lowest gaps per cell (mm): %s' % ', '.join('%d: %.3f' % (int(c), 1e3*float(gcell[c])) for c in order), flush=True)
xf, A, lamA = xf0.clone(), A0.clone(), lam0.clone(); curve = []
# adaptive: load by 1.5 mm until a second cell touches (the neighbours are lifted by the first contact: a gap closes at
# ~ 1/3 the clamp's rate), one more step, then unload by 1.5 mm until one cell is left, then back to 3 mm
delta, k, phase, loaded_max = 3e-3, 0, 'load', None
while True:
    xfix = x_touch.clone(); xfix[:, 1] -= delta; prob = kk.KKT(xfix)
    cells_before = sorted(torch.unique(cl[kk.candF][A]).tolist()) if len(A) else []
    xf, A, lamA, _, ok = kk.solve(prob, xf, A, lamA, None, 'step %d delta %.2f mm' % (k, 1e3*delta), maxn=40)
    react, cf = kk.final(prob, xf, A, lamA, None, 'step %d, delta %.2f mm, %s' % (k, 1e3*delta, 'converged' if ok else 'NOT converged'))
    cells_after = sorted(torch.unique(cl[kk.candF][A]).tolist()) if len(A) else []
    per_cell = {int(c): float(lamA[cl[kk.candF][A] == c].sum()) for c in cells_after}
    print('   cells in contact: %s -> %s; reaction per cell %s' % (cells_before, cells_after, {c: round(v, 6) for c, v in per_cell.items()}), flush=True)
    curve.append((delta, float(react[1]), len(cells_after), ok))
    torch.save(dict(xf=xf, A=A, lamA=lamA, delta=delta), str(OUT/('multi_step%d.pt' % k)))
    if not ok:
        print('STOP at step %d' % k); break
    k += 1
    if phase == 'load':
        if len(cells_after) >= 2 and loaded_max is None:
            loaded_max = delta+1.5e-3                                                  # (one more step with two cells, then unload)
        if loaded_max is not None and delta >= loaded_max-1e-12:
            phase = 'unload'; delta -= 1.5e-3
        elif delta >= 12e-3-1e-12:
            print('   no second cell by 12 mm'); break
        else:
            delta += 1.5e-3
    else:
        if len(cells_after) <= 1 or delta <= 3e-3+1e-12:
            if abs(delta-3e-3) < 1e-12:
                break
            delta = 3e-3
        else:
            delta = max(delta-1.5e-3, 3e-3)
    if k > 16:
        break
print('   reaction against delta along the path: ' + ', '.join('%.2f mm: %.5f (%d cells%s)' % (1e3*d, r, c, '' if o else ', NOT converged') for d, r, c, o in curve), flush=True)
print('done', flush=True)
