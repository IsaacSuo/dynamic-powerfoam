"""the static rest test's inner solver as a diagnostic (the reviewer, 11-07): trust-region Newton-CG (Steihaug) on the
augmented objective phi (static_contact_common), Hessian-vector products by central differences of grad phi (the active
set's flips within a product counted), block-Jacobi preconditioner from the rest elastic blocks Kbb (probe_blocks; its
definiteness checked, regularised if needed) plus rho sum_q j_q^T j_q over every site a contact row reaches (j_q = the
row of the active constraint, by autograd). Steps: (0) a common start: L-BFGS for 1000 iterations from the just-touching
state at gravity x 0.25, rho 2e3, saved; (1) the products' step sensitivity and symmetry there; (2) from the start, the same
budget of gradient evaluations: L-BFGS against Newton-CG, |grad phi| against evaluations; (3) if Newton-CG wins, the ALM
rounds at rho 2e3 fixed, inner tolerance loosened then tightened, residuals before and after each multiplier update; then
gravity x 0.5, 0.75, 1. args: [budget evals (1500)] [rho (2e3)]"""
import math
import sys
import time
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize
sys.path.insert(0, str(Path(__file__).resolve().parent))
from static_contact_common import *                                                   # noqa
from free_surface.closure_drop import colouring, probe_blocks
BUDGET = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
RHO = float(sys.argv[2]) if len(sys.argv) > 2 else 2e3
OUT = ROOT/'out_inverse'/'static_contact'
OUT.mkdir(exist_ok=True)
print('weight %.4g (per unit density); contact candidates %d' % (WEIGHT, Q), flush=True)

# ---- the rest elastic blocks
Kbb = probe_blocks(W, X, colouring(st.i.cpu().numpy(), st.j.cpu().numpy(), n))
W.elastic_mode()
ev = torch.linalg.eigvalsh(Kbb)
print('rest blocks Kbb: eigenvalues min %.3e, median %.3e, max %.3e; blocks with min eigenvalue <= 0: %d' % (float(ev.min()), float(ev[:, 0].median()), float(ev.max()), int((ev[:, 0] <= 0).sum())), flush=True)
REG = max(0., -float(ev.min()))+1e-3*float(ev[:, 0].median())
print('  regularisation added to every block: %.3e' % REG)


def precond(prob, x):
    """block-Jacobi: Kbb + REG I + rho sum over the active rows of j^T j (every site the row reaches); returns the blocks' inverse"""
    P = Kbb.clone(); P += REG*torch.eye(3, dtype=f64, device=dev)
    xg = x.clone().requires_grad_(True); y_, _ = yq(xg); gq = y_[:, 1]-C0
    act = torch.nonzero(prob.lam_n-prob.rho*gq > 0)[:, 0]
    rows = [(act, y_[:, 1])]
    if prob.anchors is not None:
        rows += [(prob.anchors[0], y_[:, 0]), (prob.anchors[0], y_[:, 2])]
    cnt = 0
    for idx, comp in rows:
        for q in idx.tolist():
            j = torch.autograd.grad(comp[q], xg, retain_graph=True)[0]
            nz = torch.nonzero(j.abs().sum(1) > 0)[:, 0]
            P[nz] += prob.rho*j[nz][:, :, None]*j[nz][:, None, :]; cnt += 1
    return torch.linalg.inv(P), cnt


def hv(prob, x, g0, d, h=1e-6):
    """H d by central differences of grad phi; the active set's flips between the two sides"""
    s_ = float(d.abs().max())
    if s_ == 0:
        return torch.zeros_like(d), 0
    hh = h/s_
    _, gp, mp = prob.phi(x+hh*d, True); _, gm, mm = prob.phi(x-hh*d, True)
    return (gp-gm)/(2*hh), int((mp != mm).sum())


def newton_cg(prob, x, tol, max_newton=60, cg_cap=200, budget=None, delta=None, log=''):
    """trust-region Newton-CG on phi: returns x, |grad phi|, evaluations used, the Newton trace"""
    f0, g0, _ = prob.phi(x); e0 = prob.evals
    Pinv, ncon = precond(prob, x)
    if delta is None:
        delta = 10*float(torch.einsum('nij,nj->ni', Pinv, g0).norm())
    trace = []
    for it in range(max_newton):
        gn = float(g0.norm())
        if gn <= tol or (budget is not None and prob.evals-e0 >= budget):
            break
        # Steihaug CG in the preconditioner's norm
        p = torch.zeros_like(x); r = -g0.clone(); y = torch.einsum('nij,nj->ni', Pinv, r); d = y.clone()
        ry = float((r*y).sum()); edge = neg = False; flips = 0; K = 0
        stop = max(min(.5, math.sqrt(gn/WEIGHT))*gn, .5*tol)
        for K in range(1, cg_cap+1):
            Hd, fl = hv(prob, x, g0, d); flips += fl
            dHd = float((d*Hd).sum())
            if dHd <= 0:
                neg = True
                tau = _to_edge(p, d, delta, Pinv); p = p+tau*d; break
            alpha = ry/dHd; pn = p+alpha*d
            if _pnorm(pn, Pinv) >= delta:
                edge = True
                tau = _to_edge(p, d, delta, Pinv); p = p+tau*d; break
            p = pn; r = r-alpha*Hd
            if float(r.norm()) <= stop:
                break
            y = torch.einsum('nij,nj->ni', Pinv, r); ryn = float((r*y).sum()); d = y+(ryn/ry)*d; ry = ryn
            if budget is not None and prob.evals-e0 >= budget:
                break
        Hp, _ = hv(prob, x, g0, p)
        pred = -(float((g0*p).sum())+.5*float((p*Hp).sum()))
        f1, g1, _ = prob.phi(x+p)
        ared = f0-f1; rho_ = ared/pred if pred > 0 else -1.
        taken = rho_ > 1e-4
        if taken:
            x = x+p; f0, g0 = f1, g1
            Pinv, ncon = precond(prob, x)
        pn_ = _pnorm(p, Pinv)
        if rho_ < .25:
            delta = .25*pn_
        elif rho_ > .75 and (edge or neg):
            delta = 2*delta
        trace.append((it, K, edge, neg, flips, gn, rho_, taken, prob.evals-e0))
        if log:
            print('     %s Newton %d: CG %d (edge %d, negative %d, active flips in the products %d), |grad phi| %.3e -> %.3e, ratio %.3f, %s, evals %d' % (
                log, it, K, edge, neg, flips, gn, float(g0.norm()), rho_, 'taken' if taken else 'rejected', prob.evals-e0), flush=True)
    return x, float(g0.norm()), prob.evals-e0, trace, delta


def _pnorm(p, Pinv):
    Pp = torch.linalg.solve(Pinv, p[:, :, None])[:, :, 0]
    return math.sqrt(max(float((p*Pp).sum()), 0.))


def _to_edge(p, d, delta, Pinv):
    """tau >= 0 with |p + tau d|_P = delta"""
    Pd = torch.linalg.solve(Pinv, d[:, :, None])[:, :, 0]; Pp = torch.linalg.solve(Pinv, p[:, :, None])[:, :, 0]
    a = float((d*Pd).sum()); b = 2*float((p*Pd).sum()); c = float((p*Pp).sum())-delta*delta
    return (-b+math.sqrt(max(b*b-4*a*c, 0.)))/(2*a)


def lbfgs(prob, x, maxiter, gtol=0.):
    xs = []
    def fg(xf):
        xx = torch.as_tensor(xf.reshape(n, 3), device=dev)
        f, gr, _ = prob.phi(xx)
        xs.append(float(gr.norm()))
        return f, gr.cpu().numpy().ravel()
    r = minimize(fg, x.cpu().numpy().ravel(), jac=True, method='L-BFGS-B', options=dict(maxiter=maxiter, maxfun=3*maxiter, gtol=gtol, ftol=1e-16, maxcor=30))
    return torch.as_tensor(r.x.reshape(n, 3), device=dev), xs, r


# ---- (0) the common start
y0_, _ = yq(X); x0 = X.clone(); x0[:, 1] -= float(y0_[:, 1].min())-C0
LF = .25
start_file = OUT/('start_g%.2f_rho%g.pt' % (LF, RHO))
prob = Problem(LF*FG, RHO, torch.zeros(Q, dtype=f64, device=dev))
if start_file.exists():
    x_s = torch.load(str(start_file))['x']
    print('start: loaded %s' % start_file.name)
else:
    t0 = time.time(); x_s, hist, r = lbfgs(prob, x0, 1000)
    print('start: L-BFGS 1000 iterations from the just-touching state at gravity x %.2f, rho %g: |grad phi| %.3e -> %.3e (%.0f s, %d evals: Warp %.1f ms, torch %.1f ms each)' % (
        LF, RHO, hist[0], hist[-1], time.time()-t0, prob.evals, 1e3*prob.t_warp/prob.evals, 1e3*prob.t_torch/prob.evals), flush=True)
    torch.save(dict(x=x_s), str(start_file))
res = prob.residuals(x_s, updated=False); report(res, 'start, current multipliers (all 0)')

# ---- (1) the products at the start
f_, g_, _ = prob.phi(x_s); gen = torch.Generator(device='cpu').manual_seed(0)
d1 = torch.randn(n, 3, generator=gen, dtype=f64).to(dev); d2 = torch.randn(n, 3, generator=gen, dtype=f64).to(dev)
Hd_ref, fl = hv(prob, x_s, g_, d1, 1e-6)
for h in (1e-5, 1e-7, 1e-8):
    Hd, fl2 = hv(prob, x_s, g_, d1, h)
    print('products: h %.0e against 1e-6: |diff| / |H d| %.2e (active flips %d / %d)' % (h, float((Hd-Hd_ref).norm()/Hd_ref.norm()), fl2, fl))
Hd2, _ = hv(prob, x_s, g_, d2, 1e-6)
print('products: symmetry d1 . H d2 %+.10e against d2 . H d1 %+.10e (rel %.1e)' % (float((d1*Hd2).sum()), float((d2*Hd_ref).sum()), abs(float((d1*Hd2).sum())-float((d2*Hd_ref).sum()))/abs(float((d1*Hd2).sum()))), flush=True)

# ---- (2) the same budget: L-BFGS against Newton-CG
print('\n== inner solvers from the common start, budget %d gradient evaluations' % BUDGET, flush=True)
pL = Problem(LF*FG, RHO, torch.zeros(Q, dtype=f64, device=dev)); t0 = time.time()
xL, hist, r = lbfgs(pL, x_s, BUDGET)
marks = [hist[min(k, len(hist)-1)] for k in (0, 100, 300, 600, 1000, BUDGET-1)]
print('   L-BFGS: |grad phi| at evals 0 / 100 / 300 / 600 / 1000 / %d: %s (%.0f s; %s)' % (BUDGET, ' '.join('%.2e' % v for v in marks), time.time()-t0, r.message[:40]), flush=True)
pN = Problem(LF*FG, RHO, torch.zeros(Q, dtype=f64, device=dev)); t0 = time.time()
xN, gN, used, trace, delta = newton_cg(pN, x_s, tol=1e-6*WEIGHT, budget=BUDGET, log='NCG')
print('   Newton-CG: |grad phi| %.2e -> %.2e in %d evals (%.0f s)' % (hist[0], gN, used, time.time()-t0), flush=True)
better = gN < hist[-1]
print('   Newton-CG %s L-BFGS at equal budget' % ('beats' if better else 'does not beat'), flush=True)
if not better:
    print('STOP: the inner solver question is open'); sys.exit(0)

# ---- (3) the ALM rounds with Newton-CG, rho fixed
x = xN; prob = pN
for LF in (.25, .5, .75, 1.):
    prob.load = LF*FG
    tol = 1e-3*WEIGHT
    print('\n== gravity x %.2f, rho %g fixed' % (LF, RHO), flush=True)
    for it in range(15):
        x, gn, used, trace, delta = newton_cg(prob, x, tol=tol, delta=delta)
        rb = prob.residuals(x, updated=False); ra = prob.residuals(x, updated=True)
        print('  outer %d: inner tol %.1e, |grad phi| %.2e in %d evals (%d Newton); before the update: KKT %.2e, pen %.3e mm, comp %.2e | after: KKT %.2e, comp %.2e, active %d, |delta lambda| / |lambda| %.2e' % (
            it, tol/WEIGHT, gn/WEIGHT, used, len(trace), rb['stat'], 1e3*rb['pen'], rb['comp'], ra['stat'], ra['comp'], ra['active'],
            float((ra['lam']-rb['lam']).norm()/max(float(ra['lam'].norm()), 1e-300))), flush=True)
        prob.update(ra)
        if ra['pen'] < 1e-6 and ra['comp'] < 1e-9 and ra['stat'] < 1e-4:
            break
        tol = max(tol/3, 1e-6*WEIGHT)
        if rb['det_min'] <= 0:
            print('STOP: inverted (min det F_B %.3f)' % rb['det_min']); torch.save(dict(x=x, lam=prob.lam_n), str(OUT/'fail.pt')); sys.exit(0)
    report(ra, 'gravity x %.2f, final (updated multipliers)' % LF)
    feet = torch.unique(cl[torch.nonzero(ra['lam'] > 0)[:, 0]])
    print('   touching cells %d; their F_B yy median %.4f, min %.4f; support (lambda-weighted centre) %s' % (len(feet), float(ra['Fyy'][feet].median()), float(ra['Fyy'][feet].min()),
          np.round(((ra['lam'][:, None]*ra['y']).sum(0)/ra['lam'].sum()).cpu().numpy(), 4)), flush=True)
    torch.save(dict(x=x, lam=prob.lam_n), str(OUT/('A_g%.2f.pt' % LF)))
print('done', flush=True)
