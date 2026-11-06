"""from the converged 6 mm state of the foot (static_contact_kkt_checks: multi_step2.pt), the clamp pushed in steps of
argv[1] mm (default 0.25) up to argv[2] mm (default 8): where the second cap first touches, and whether that step
converges with the solver as it is (the reviewer's item 1). args: [step mm] [max mm] [solver: kkt | tr]"""
import sys
from pathlib import Path
import torch
sys.argv_ = sys.argv[1:]
sys.argv = [sys.argv[0], '0.04', '3', '1', '40', '1']
sys.path.insert(0, str(Path(__file__).resolve().parent))
import static_contact_kkt as kk
from static_contact_kkt import *                                                      # noqa
STEP = 1e-3*float(sys.argv_[0]) if len(sys.argv_) > 0 else .25e-3
DMAX = 1e-3*float(sys.argv_[1]) if len(sys.argv_) > 1 else 8e-3
SOLVER = sys.argv_[2] if len(sys.argv_) > 2 else 'kkt'
MAXN_ = int(sys.argv_[3]) if len(sys.argv_) > 3 else 40
START = sys.argv_[4] if len(sys.argv_) > 4 else 'multi_step2.pt'                    # (a negative step unloads from it)
z = torch.load(str(kk.OUT/START)); xf, A, lamA, delta = z['xf'], z['A'], z['lamA'], float(z['delta'])
print('start: %s, delta %.2f mm, |A| %d; steps of %.2f mm, solver %s' % (START, 1e3*delta, len(A), 1e3*STEP, SOLVER), flush=True)
solve = kk.solve if SOLVER == 'kkt' else kk.solve_tr
curve = []
while (delta < DMAX-1e-12) if STEP > 0 else (delta > DMAX+1e-12):
    delta += STEP
    xfix = x_touch.clone(); xfix[:, 1] -= delta; prob = kk.KKT(xfix)
    before = sorted(torch.unique(cl[kk.candF][A]).tolist()) if len(A) else []
    xf, A, lamA, _, ok = solve(prob, xf, A, lamA, None, 'delta %.2f mm' % (1e3*delta), maxn=MAXN_)
    react, cf = kk.final(prob, xf, A, lamA, None, 'delta %.2f mm, %s' % (1e3*delta, 'converged' if ok else 'NOT converged'))
    after = sorted(torch.unique(cl[kk.candF][A]).tolist()) if len(A) else []
    g = prob.cons(prob.full(xf))[0]
    gcell = torch.full((n,), 1e9, dtype=f64, device=dev).scatter_reduce(0, cl[kk.candF], g, 'amin')
    low = torch.argsort(gcell)[:3]
    print('   cells in contact %s -> %s; lowest gaps %s' % (before, after, ', '.join('%d: %.4f mm' % (int(c), 1e3*float(gcell[c])) for c in low)), flush=True)
    curve.append((delta, float(react[1]), len(after), ok))
    torch.save(dict(xf=xf, A=A, lamA=lamA, delta=delta), str(kk.OUT/('fine_%s_d%.2f.pt' % (SOLVER, 1e3*delta))))
    if not ok:
        print('STOP: not converged at delta %.2f mm' % (1e3*delta)); break
print('reaction against delta: ' + ', '.join('%.2f: %.5f (%d%s)' % (1e3*d, r, c, '' if o else ' NOT conv') for d, r, c, o in curve), flush=True)
print('done', flush=True)
