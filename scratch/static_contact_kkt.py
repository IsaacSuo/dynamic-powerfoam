"""the foot under displacement control with a primal-dual active-set Newton on the KKT system (the reviewer, 11-07; no rho,
no penalty): unknowns the free sites x_F (the lowest H m; every other site rigidly fixed and pushed down by delta) and
the active points' multipliers lambda_A; constraints g_q = y_q . up - C0 >= 0 on every cap point of the free cells
(y_q = x_B + F_s r_q, the fitted skin's own map), stick (phase 2): (y_k - a_k)_h = 0 on the anchored points.
Per iteration (set A fixed): J_A by autograd (its rank and smallest singular value reported), H_L = d^2 (E - lambda . g)
by central differences of grad L over the free DOFs (dense), the reduced curvature Z^T H_L Z on null(J_A) (the clamp
removes the rigid modes; a negative one is reported with its shape, energy and gap changes); the KKT step with a shift on
H_L when the reduced Hessian is indefinite; an l1 merit line search (E + mu (sum_A |g| + sum_not A max(0, -g)));
then the set update over all candidates: drop lambda < 0 only if NNLS finds no lambda >= 0 giving the same J^T lambda
(redundancy reported), add g < 0. Pass: penetration, lambda >= 0, complementarity, stationarity, total force = the clamp's,
moment. args: [H (0.04)] [deltas mm (0.5,1,2,3)] [delta_h mm (1)] [max Newton per delta (30)] [stick 0/1 (1)]"""
import math
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import nnls
from scipy.linalg import qr as sqr
sys.path.insert(0, str(Path(__file__).resolve().parent))
from static_contact_common import *                                                   # noqa
H = float(sys.argv[1]) if len(sys.argv) > 1 else .04
DELTAS = [1e-3*float(a) for a in (sys.argv[2].split(',') if len(sys.argv) > 2 else ['0.5', '1', '2', '3'])]
DH = 1e-3*float(sys.argv[3]) if len(sys.argv) > 3 else 1e-3
MAXN = int(sys.argv[4]) if len(sys.argv) > 4 else 30
STICK = int(sys.argv[5]) if len(sys.argv) > 5 else 1
RANK_TOL = float(os.environ.get('RANK_TOL', '1e-2'))                                 # (the working set's relative singular-value cut)
OUT = ROOT/'out_inverse'/'static_contact'; OUT.mkdir(exist_ok=True)
y0_, _ = yq(X); x_touch = X.clone(); x_touch[:, 1] -= float(y0_[:, 1].min())-C0
free = torch.nonzero(X[:, 1] < float(X[:, 1].min())+H)[:, 0]; nF = len(free); N3 = 3*nF
fixed = torch.ones(n, dtype=torch.bool, device=dev); fixed[free] = False
candF = torch.nonzero((~fixed)[cl])[:, 0]; QF = len(candF)
print('foot: %d free sites (lowest %.0f mm), %d candidate points; the clamp (every other site) rigid; weight %.4g for scale' % (nF, 1e3*H, QF, WEIGHT), flush=True)
W.elastic_mode()


class KKT:
    def __init__(self, xfix, anchors=None):
        self.xfix, self.anchors = xfix, anchors                                       # anchors: (indices into candF, a_h (K, 2))
        self.K = 0 if anchors is None else len(anchors[0])

    def full(self, xf):
        x = self.xfix.clone(); x[free] = xf; return x

    def cons(self, x):
        """g (QF,) and the stick residuals c (K, 2) at x, with y"""
        y_ = yq(x)[0][candF]; g = y_[:, 1]-C0
        c = (y_[self.anchors[0]][:, [0, 2]]-self.anchors[1]) if self.anchors is not None else None
        return g, c, y_

    def gradL(self, xf, A, lamA, lamT):
        """E, grad_x L on the free DOFs with L = E - lamA . g_A + lamT . c (lamT (K, 2)), and g, c"""
        x = self.full(xf); e, ge = W.grad(x)
        xg = x.clone().requires_grad_(True); g, c, _ = self.cons(xg)
        Lc = -(g[A]*lamA).sum() if len(A) else g.sum()*0
        if self.anchors is not None:
            Lc = Lc+(lamT*c).sum()
        Lc.backward()
        return float(e), (ge+xg.grad)[free], g.detach(), (c.detach() if c is not None else None)

    def jac(self, xf, A):
        """rows of the active constraints over the free DOFs: J (m, N3), m = |A| + 2 K"""
        x = self.full(xf); xg = x.clone().requires_grad_(True); g, c, _ = self.cons(xg)
        rows = [g[q] for q in A.tolist()]
        if self.anchors is not None:
            rows += [c[k, j] for k in range(self.K) for j in range(2)]
        J = torch.zeros(len(rows), N3, dtype=f64, device=dev)
        for i, r_ in enumerate(rows):
            J[i] = torch.autograd.grad(r_, xg, retain_graph=True)[0][free].reshape(-1)
        return J

    def hessL(self, xf, A, lamA, lamT, h=1e-6):
        Hm = torch.zeros(N3, N3, dtype=f64, device=dev)
        for k in range(N3):
            e = torch.zeros(nF, 3, dtype=f64, device=dev); e.view(-1)[k] = h
            Hm[:, k] = (self.gradL(xf+e, A, lamA, lamT)[1]-self.gradL(xf-e, A, lamA, lamT)[1]).reshape(-1)/(2*h)
        return .5*(Hm+Hm.T)

    def merit(self, xf, A, mu):
        x = self.full(xf); e = float(W.grad(x)[0]); g, c, _ = self.cons(x)
        viol = float(g[A].abs().sum())+float((-g).clamp(min=0)[~torch.isin(torch.arange(QF, device=dev), A)].sum())
        if c is not None:
            viol += float(c.abs().sum())
        return e+mu*viol, e, g, c


def solve(prob, xf, A, lamA, lamT, tag, maxn=MAXN):
    lamT = torch.zeros(prob.K, 2, dtype=f64, device=dev) if (lamT is None and prob.anchors is not None) else lamT
    for it in range(maxn):
        t0 = time.time()
        e, gL, g, c = prob.gradL(xf, A, lamA, lamT)
        m = len(A)+2*prob.K
        J = prob.jac(xf, A) if m else torch.zeros(0, N3, dtype=f64, device=dev)
        rA = torch.cat([g[A], c.reshape(-1)]) if c is not None else g[A]
        # the Jacobian's rank, the reduced curvature
        rank, smin, rank_num = (0, float('nan'), 0)
        if m:
            U, S, Vh = torch.linalg.svd(J.cpu(), full_matrices=True); S = S.to(dev); Vh = Vh.to(dev)
            rank_num = int((S > 1e-10*S[0]).sum())
            rank = int((S > RANK_TOL*S[0]).sum()); smin = float(S[rank-1]) if rank else 0.   # (the working rank: rows nearly dependent
            Z = Vh[rank:].T                                                             # on a cap are not enforced separately; null(J_work))
        else:
            Z = torch.eye(N3, dtype=f64, device=dev)
        Hm = prob.hessL(xf, A, lamA, lamT)
        Hr = Z.T@Hm@Z; evr, vr = torch.linalg.eigh(Hr.cpu()); evr = evr.to(dev); vr = vr.to(dev)
        shift = 0.
        if float(evr[0]) < 0:
            shift = -float(evr[0])*1.01+1e-9*float(evr.abs().max())
            d = (Z@vr[:, 0]).reshape(nF, 3); dn = d/d.norm()
            eps_ = 1e-4
            ep = prob.merit(xf+eps_*dn, A, 0.)[1]; em = prob.merit(xf-eps_*dn, A, 0.)[1]
            gp = prob.cons(prob.full(xf+eps_*dn))[0]
            cellmax = int(free[torch.argmax(d.norm(dim=1))])
            print('     %s it %d: NEGATIVE reduced curvature %.3e: shape concentrated on cell %d (%.0f %% of |d|^2 on its 5 largest sites), E(+/-1e-4 along it) - E %+.3e / %+.3e, gap change at the active points max %+.3e mm, min %+.3e mm; H_L shifted by %.3e' % (
                tag, it, float(evr[0]), cellmax, 100*float((d.norm(dim=1).sort(descending=True)[0][:5]**2).sum()/(d*d).sum()), ep-e, em-e, 1e3*float((gp-g)[A].max()) if len(A) else 0., 1e3*float((gp-g)[A].min()) if len(A) else 0., shift), flush=True)
        # the working set: a maximal independent subset of the rows (pivoted QR of J^T); the dependent rows keep multiplier 0
        if m:
            _, _, piv = sqr(J.cpu().numpy().T, pivoting=True, mode='economic'); piv = torch.as_tensor(piv[:rank].copy(), device=dev)
        else:
            piv = torch.zeros(0, dtype=torch.long, device=dev)
        Jp = J[piv]; rp = rA[piv]
        sgn_all = torch.cat([-torch.ones(len(A), dtype=f64, device=dev), torch.ones(2*prob.K, dtype=f64, device=dev)]) if m else torch.zeros(0, dtype=f64, device=dev)
        sgn = sgn_all[piv]
        # the KKT step: [H + shift I, sgn J_p^T; J_p, 0] [dx; dmu] = [-gL; -r_p]  (L = E - lam . g + lamT . c)
        Hs = Hm+shift*torch.eye(N3, dtype=f64, device=dev); mp = len(piv)
        Kmat = torch.zeros(N3+mp, N3+mp, dtype=f64, device=dev); Kmat[:N3, :N3] = Hs
        if mp:
            Kmat[:N3, N3:] = (sgn[None, :]*Jp.T); Kmat[N3:, :N3] = Jp
        rhs = torch.cat([-gL.reshape(-1), -rp]) if mp else -gL.reshape(-1)
        sol = torch.linalg.solve(Kmat.cpu(), rhs.cpu()).to(dev)
        dx = sol[:N3].reshape(nF, 3)
        lam_all = torch.cat([lamA, lamT.reshape(-1)]) if prob.anchors is not None else lamA.clone()
        if os.environ.get('KKT_DEBUG'):
            lin = float((Kmat@sol-rhs).norm()); gpred = gL.reshape(-1)+Hs@sol[:N3]-(sgn[None, :]*Jp.T)@sol[N3:] if mp else gL.reshape(-1)+Hs@sol[:N3]
            gq_act, _, _ = prob.cons(prob.full(xf+dx)); gLn = prob.gradL(xf+dx, A, lamA, lamT)[1]
            lam_try = lam_all.clone(); lam_try[piv] += sol[N3:]
            gLn2 = prob.gradL(xf+dx, A, lam_try[:len(A)], (lam_try[len(A):].reshape(prob.K, 2) if prob.anchors is not None else lamT))[1]
            print('     DEBUG it %d: |K sol - rhs| %.1e, cond(K) %.1e, |dx| max %.3e m, dmu %s; predicted |grad L| at x+dx %.2e, actual (old lam) %.2e, actual (new lam) %.2e; gaps at the working set after the full step %s mm (linear target 0)' % (
                it, lin, float(torch.linalg.cond(Kmat.cpu())), float(dx.abs().max()), np.round(sol[N3:].cpu().numpy(), 4), float(gpred.norm()), float(gLn.norm()), float(gLn2.norm()), np.round(1e3*gq_act[A[piv[piv < len(A)]]].cpu().numpy(), 4) if mp else '-'), flush=True)
        lam_new_all = torch.zeros_like(lam_all); lam_new_all[piv] = lam_all[piv]+sol[N3:]
        lamA_new = lam_new_all[:len(A)]; lamT_new = lam_new_all[len(A):].reshape(prob.K, 2) if prob.anchors is not None else None
        # the l1 merit line search with mu above the multipliers
        mu = 2*max(float(lamA_new.abs().max()) if len(A) else 0., float(lamT_new.abs().max()) if lamT_new is not None else 0., 1e-3*WEIGHT)
        f0, _, _, _ = prob.merit(xf, A, mu); dd = float((gL*dx).sum())-mu*float(rA.abs().sum())   # (the merit's directional derivative)
        a_ = 1.
        for _ in range(30):
            f1, _, _, _ = prob.merit(xf+a_*dx, A, mu)
            if f1 <= f0+1e-4*a_*dd:
                break
            a_ *= .5
        xf = xf+a_*dx; lamA = lamA+a_*(lamA_new-lamA)
        if lamT_new is not None:
            lamT = lamT+a_*(lamT_new-lamT)
        # the set update over all candidates
        g, c, y_ = prob.cons(prob.full(xf))
        drop = torch.nonzero(lamA < -1e-12)[:, 0]; redundant = 0
        if len(drop) and rank < len(A)+2*prob.K:
            # NNLS: is there lambda >= 0 with J_A^T lambda = J_A^T lamA?
            JA = J[:len(A)].cpu().numpy(); target = JA.T@lamA.cpu().numpy()
            sol_, res_ = nnls(JA.T, target, maxiter=10*len(A))
            if res_ < 1e-10*max(np.linalg.norm(target), 1e-300):
                lamA = torch.as_tensor(sol_, dtype=f64, device=dev); redundant = len(drop); drop = torch.nonzero(lamA < -1e-12)[:, 0]
        keep = torch.ones(len(A), dtype=torch.bool, device=dev); keep[drop] = False
        A_new = A[keep]; lam_new = lamA[keep]
        viol = torch.nonzero(g < -1e-9)[:, 0]; add = viol[~torch.isin(viol, A_new)]
        A2 = torch.cat([A_new, add]); lam2 = torch.cat([lam_new, torch.zeros(len(add), dtype=f64, device=dev)])
        e, gL2, _, _ = prob.gradL(xf, A2, lam2, lamT)
        print('   %s it %d: |A| %d (rank %d of numerical %d, s_min %.2e, working set %d), reduced curvature min %.3e, step %.3g, |grad L| / weight %.2e, max penetration %.3e mm, dropped %d (redundant %d), added %d; %.0f s' % (
            tag, it, len(A), rank, rank_num, smin, mp, float(evr[0]), a_, float(gL2.norm())/WEIGHT, 1e3*float((-g).clamp(min=0).max()), len(drop), redundant, len(add), time.time()-t0), flush=True)
        A, lamA = A2, lam2
        converged = float(gL2.norm())/WEIGHT < 1e-6 and float((-g).clamp(min=0).max()) < 1e-9 and len(add) == 0 and len(drop) == 0 and (c is None or float(c.abs().max()) < 1e-9)
        if converged:
            break
    return xf, A, lamA, lamT, converged


def solve_tr(prob, xf, A, lamA, lamT, tag, maxn=MAXN, delta0=1e-3):
    """the reviewer's items 2-3 (11-07): a trust-region SQP on the working set. Per iteration: multipliers by least squares
    on the stationarity (grad E = sum -sgn_i J_i^T lambda_i); H_L with them; the normal step dx_n = -J_p^+ r_p; the
    tangential step Z dz from the trust-region subproblem on the reduced model (exact, in the eigenbasis of Z^T H_L Z;
    an indefinite model is handled, nothing deleted for it); a second-order correction dx_c = -J_p^+ r(x + dx) tried
    first; acceptance by the l1 merit's actual / predicted decrease; logged: the merit's directional derivative, the
    actual decrease with and without the correction, the penetration of every candidate after the step. Set update:
    drop working rows with lambda < 0 (NNLS redundancy check), add violated candidates, the deepest per cell first"""
    lamT = torch.zeros(prob.K, 2, dtype=f64, device=dev) if (lamT is None and prob.anchors is not None) else lamT
    Delta = delta0; mu = 1e-3*WEIGHT; converged = False
    for it in range(maxn):
        t0 = time.time()
        e, gE, g, c = prob.gradL(xf, A[:0], lamA[:0], torch.zeros_like(lamT) if lamT is not None else None)   # grad E alone
        m = len(A)+2*prob.K
        J = prob.jac(xf, A) if m else torch.zeros(0, N3, dtype=f64, device=dev)
        rA = torch.cat([g[A], c.reshape(-1)]) if c is not None else g[A]
        sgn_all = torch.cat([-torch.ones(len(A), dtype=f64, device=dev), torch.ones(2*prob.K, dtype=f64, device=dev)])
        rank = rank_num = 0; smin = float('nan')
        if m:
            U, S, Vh = torch.linalg.svd(J.cpu(), full_matrices=True); S = S.to(dev); Vh = Vh.to(dev)
            rank_num = int((S > 1e-10*S[0]).sum()); rank = int((S > RANK_TOL*S[0]).sum()); smin = float(S[rank-1]) if rank else 0.
            Z = Vh[rank:].T
            _, _, piv = sqr(J.cpu().numpy().T, pivoting=True, mode='economic'); piv = torch.as_tensor(piv[:rank].copy(), device=dev)
        else:
            Z = torch.eye(N3, dtype=f64, device=dev); piv = torch.zeros(0, dtype=torch.long, device=dev)
        Jp, rp, sgn = J[piv], rA[piv], sgn_all[piv]; mp = len(piv)
        # the multipliers by least squares: J_p^T y = grad E, lambda = -sgn y
        lam_all = torch.zeros(m, dtype=f64, device=dev)
        if mp:
            y_ls = torch.linalg.lstsq(Jp.T.cpu(), gE.reshape(-1, 1).cpu()).solution[:, 0].to(dev)
            lam_all[piv] = -sgn*y_ls
        lamA_ls = lam_all[:len(A)]; lamT_ls = lam_all[len(A):].reshape(prob.K, 2) if prob.anchors is not None else None
        gL = gE.reshape(-1)+((sgn_all[:, None]*J)*lam_all[:, None]).sum(0) if m else gE.reshape(-1)
        Hm = prob.hessL(xf, A, lamA_ls, lamT_ls)
        # normal step, reduced model
        dx_n = -(Jp.T@torch.linalg.solve((Jp@Jp.T).cpu(), rp.cpu()).to(dev)) if mp else torch.zeros(N3, dtype=f64, device=dev)
        gr = Z.T@(gL+Hm@dx_n); Hr = Z.T@Hm@Z
        evr, vr = torch.linalg.eigh(Hr.cpu()); evr = evr.to(dev); vr = vr.to(dev)
        cvec = vr.T@gr
        def dz_of(sig):
            return -vr@(cvec/(evr+sig))
        sig_min = max(0., -float(evr[0]))*(1+1e-6)+1e-12
        dz = dz_of(sig_min); sig = sig_min; hard = False
        if float(dz.norm()) > Delta:
            lo, hi = sig_min, sig_min+1.
            while float(dz_of(hi).norm()) > Delta:
                hi *= 2
            for _ in range(80):
                mid = .5*(lo+hi)
                if float(dz_of(mid).norm()) > Delta:
                    lo = mid
                else:
                    hi = mid
            sig = hi; dz = dz_of(sig)
        elif float(evr[0]) <= 0:                                                      # (the hard case: fill the radius along the negative eigenvector)
            hard = True; tau = math.sqrt(max(Delta*Delta-float(dz.norm())**2, 0.)); dz = dz+tau*vr[:, 0]
        dx = (dx_n+Z@dz)
        # merit, directional derivative, predicted decrease
        mu = max(mu, 1.5*float(lam_all.abs().max()) if m else mu)
        f0, _, _, _ = prob.merit(xf, A, mu)
        dd = float(gE.reshape(-1)@dx)-mu*float(rp.abs().sum())
        pred = -(float(gL@dx)+.5*float(dx@(Hm@dx)))+mu*(float(rp.abs().sum())-float((rp+Jp@dx).abs().sum()))
        dxm = dx.reshape(nF, 3)
        f1, _, g1, _ = prob.merit(xf+dxm, A, mu); pen1 = float((-g1).clamp(min=0).max())
        # second-order correction on the working rows
        g2_, c2_, _ = prob.cons(prob.full(xf+dxm)); r2 = (torch.cat([g2_[A], c2_.reshape(-1)]) if c2_ is not None else g2_[A])[piv]
        dxc = -(Jp.T@torch.linalg.solve((Jp@Jp.T).cpu(), r2.cpu()).to(dev)).reshape(nF, 3) if mp else torch.zeros_like(dxm)
        f2, _, g2m, _ = prob.merit(xf+dxm+dxc, A, mu); pen2 = float((-g2m).clamp(min=0).max())
        use_soc = f2 < f1
        fnew = f2 if use_soc else f1; step = dxm+dxc if use_soc else dxm
        ared = f0-fnew; ratio = ared/pred if pred > 0 else (-1. if ared <= 0 else 1.)
        accepted = ared > 0 and (ratio > .1 or pred <= 0)
        if accepted:
            xf = xf+step
        dzn = float(dz.norm())
        if ratio < .25 or not accepted:
            Delta = max(.25*min(Delta, float(step.norm())), 1e-7)
        elif ratio > .75 and dzn > .9*Delta:
            Delta = 2*Delta
        # the set update (after an accepted step, on the LS multipliers; else only add the violated)
        g, c, y_ = prob.cons(prob.full(xf))
        drop = torch.zeros(0, dtype=torch.long, device=dev); redundant = 0
        if accepted and len(A):
            drop = torch.nonzero(lamA_ls < -1e-12*max(float(lamA_ls.abs().max()), 1e-300))[:, 0]
            if len(drop) and rank < m:
                JA = J[:len(A)].cpu().numpy(); target = JA.T@lamA_ls.cpu().numpy()
                sol_, res_ = nnls(JA.T, target, maxiter=10*len(A))
                if res_ < 1e-10*max(np.linalg.norm(target), 1e-300):
                    lamA_ls = torch.as_tensor(sol_, dtype=f64, device=dev); redundant = len(drop); drop = torch.nonzero(lamA_ls < -1e-12)[:, 0]
        keep = torch.ones(len(A), dtype=torch.bool, device=dev); keep[drop] = False
        A_new = A[keep]; lam_new = lamA_ls[keep]
        viol = torch.nonzero(g < -1e-9)[:, 0]; viol = viol[~torch.isin(viol, A_new)]
        add = torch.zeros(0, dtype=torch.long, device=dev)
        if len(viol):                                                                # (the deepest per cell first)
            cells_v = cl[candF][viol]; order = torch.argsort(g[viol])
            seen = set(); picks = []
            for q in viol[order].tolist():
                cc_ = int(cl[candF][q])
                if cc_ not in seen:
                    seen.add(cc_); picks.append(q)
            add = torch.as_tensor(picks, device=dev)
        A2 = torch.cat([A_new, add]); lam2 = torch.cat([lam_new, torch.zeros(len(add), dtype=f64, device=dev)])
        stat = float(gL.norm())/WEIGHT
        print('   %s it %d: |A| %d (rank %d/%d, s_min %.2e), red. curv. min %.3e%s, Delta %.2e, |dz| %.2e, |dx_n| %.2e; merit dd %+.3e, pred %+.3e, ared %+.3e (no SOC) / %+.3e (SOC), ratio %.2f, %s%s; pen. after %.3e mm (no SOC %.3e); stationarity(LS lambda) %.2e; dropped %d (redundant %d), added %d; %.0f s' % (
            tag, it, len(A), rank, rank_num, smin, float(evr[0]), ' HARD' if hard else '', Delta, dzn, float(dx_n.norm()), dd, pred, f0-f1, f0-f2, ratio,
            'accepted' if accepted else 'REJECTED', ' (SOC)' if (accepted and use_soc) else '', 1e3*(pen2 if use_soc else pen1) if accepted else 1e3*float((-g).clamp(min=0).max()), 1e3*pen1, stat, len(drop), redundant, len(add), time.time()-t0), flush=True)
        if accepted:
            A, lamA = A2, lam2
        converged = accepted and stat < 1e-6 and float((-g).clamp(min=0).max()) < 1e-9 and len(add) == 0 and len(drop) == 0 and (c is None or float(c.abs().max()) < 1e-9) and float(rp.abs().max()) < 1e-9
        if converged:
            lamA = lamA_ls[keep] if len(A) else lamA
            if prob.anchors is not None:
                lamT = lamT_ls
            break
    return xf, A, lamA, lamT, converged


def final(prob, xf, A, lamA, lamT, tag):
    x = prob.full(xf); e, ge = W.grad(x); g, c, y_ = prob.cons(x)
    xg = x.clone().requires_grad_(True); g2, c2, _ = prob.cons(xg)
    Lc = -(g2[A]*lamA).sum() if len(A) else g2.sum()*0
    if c2 is not None:
        Lc = Lc+(lamT*c2).sum()
    Lc.backward(); JTl = -xg.grad
    gL = (ge-JTl)[free]; clamp_force = -ge[fixed].sum(0)
    fq = torch.zeros(QF, 3, dtype=f64, device=dev); fq[A, 1] = lamA
    if c2 is not None:
        fq[prob.anchors[0], 0] -= lamT[:, 0]; fq[prob.anchors[0], 2] -= lamT[:, 1]
    react = fq.sum(0); mom = torch.linalg.cross(y_, fq).sum(0)
    # the moment of the clamp's force: the elastic forces on the fixed sites at their positions
    mom_clamp = torch.linalg.cross(x[fixed], -ge[fixed]).sum(0)
    det = torch.linalg.det(W.tF)
    print('== %s: penetration max %.3e mm (all %d candidates), lambda_n min %.3e (>= 0: %s), complementarity max |lambda g| %.2e, stationarity |grad E - J^T lambda| / weight %.2e (max entry %.2e)%s' % (
        tag, 1e3*float((-g).clamp(min=0).max()), QF, float(lamA.min()) if len(A) else 0., bool((lamA >= 0).all()), float((lamA*g[A]).abs().max()) if len(A) else 0.,
        float(gL.norm())/WEIGHT, float(gL.abs().max())/WEIGHT, ('; stick residual max %.3e mm' % (1e3*float(c.abs().max()))) if c is not None else ''))
    print('   reactions on the ground %s against the clamp force %s (|diff| / |clamp| %.1e); moments about the origin: reactions %s, clamp %s (|sum| / |clamp| %.1e)' % (
        np.round(react.cpu().numpy(), 5), np.round(clamp_force.cpu().numpy(), 5), float((react-clamp_force).norm()/clamp_force.norm()), np.round(mom.cpu().numpy(), 6), np.round(mom_clamp.cpu().numpy(), 6), float((mom-mom_clamp).norm()/mom_clamp.norm())))
    print('   active %d points on %d cells, reaction per cell max %.4g; foot F_B yy median %.4f min %.4f; min det F_B (free) %.4f; E %.4e' % (
        len(A), len(torch.unique(cl[candF][A])) if len(A) else 0, float(torch.zeros(n, dtype=f64, device=dev).index_add(0, cl[candF][A], lamA).max()) if len(A) else 0.,
        float(W.tF[free][:, 1, 1].median()), float(W.tF[free][:, 1, 1].min()), float(det[free].min()), e), flush=True)
    return react, clamp_force


if __name__ == '__main__':
    xf = x_touch[free].clone(); A = torch.zeros(0, dtype=torch.long, device=dev); lamA = torch.zeros(0, dtype=f64, device=dev)
    curve = []
    for delta in DELTAS:
        xfix = x_touch.clone(); xfix[:, 1] -= delta; prob = KKT(xfix)
        xf, A, lamA, _, ok = solve(prob, xf, A, lamA, None, 'delta %.1f mm' % (1e3*delta))
        react, cf = final(prob, xf, A, lamA, None, 'delta %.1f mm, %s' % (1e3*delta, 'converged' if ok else 'NOT converged'))
        curve.append((delta, float(react[1]), float(cf[1]), ok))
        torch.save(dict(xf=xf, A=A, lamA=lamA, delta=delta), str(OUT/('kkt_d%.1f.pt' % (1e3*delta))))
        if not ok:
            print('STOP at delta %.1f mm' % (1e3*delta)); break
    print('reaction against delta: ' + ', '.join('%.1f mm: %.4g (clamp %.4g, %s)' % (1e3*d, r, c, 'ok' if o else 'no') for d, r, c, o in curve), flush=True)
    if STICK and curve[-1][3]:
        yA = prob.cons(prob.full(xf))[2]; anchors = (A.clone(), yA[A][:, [0, 2]].clone())
        xfix = x_touch.clone(); xfix[:, 1] -= DELTAS[-1]; xfix[:, 0] += DH; probS = KKT(xfix, anchors)
        print('\n== stick: %d anchors at delta %.1f mm, the clamp pushed sideways by %.1f mm' % (len(A), 1e3*DELTAS[-1], 1e3*DH), flush=True)
        xf2, A2, lam2, lamT, ok = solve(probS, xf, A, lamA, None, 'stick')
        final(probS, xf2, A2, lam2, lamT, 'stick, %s' % ('converged' if ok else 'NOT converged'))
        print('   anchored points still in the normal active set: %d of %d' % (int(torch.isin(anchors[0], A2).sum()), len(anchors[0])))
        torch.save(dict(xf=xf2, A=A2, lamA=lam2, lamT=lamT, anchors=anchors), str(OUT/'kkt_stick.pt'))
    print('done', flush=True)
