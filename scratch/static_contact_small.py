"""the hard-contact rest, shrunk (11-07, the user: validate the whole-body balance first, then a small deformable test):
 0 rigid: the whole body as a rigid body (translation + rotation, 6 unknowns), the weight against the hard ground
   (ALM on the 6 unknowns, Newton with the exact FD Hessian): sum lambda = weight, moment balance, active points, penetration.
 1 foot, displacement controlled: the sites of the lowest H m free (a few hundred), every other site rigidly fixed and
   pushed down by delta in steps; the full fitted-skin energy; contact candidates = the free cells' cap points; the ALM's
   inner problem solved by Newton with the dense Hessian by central differences of grad phi over the free DOFs, the active
   set fixed within an iteration (semismooth), Armijo line search; multipliers updated between inner solves, rho fixed.
   Reports per delta: the inner convergence (|grad phi|, iterations), penetration, complementarity, KKT stationarity,
   total reaction (= the force the clamp must push with: sum over the fixed sites of -grad E), the foot's deformation.
 2 stick on the foot: anchors at delta's converged state, the clamp pushed sideways by delta_h, tangential equalities.
args: [H (0.05)] [rho (2e3)] [deltas mm list, default 0.5,1,2,3] [delta_h mm (1)]"""
import math
import sys
import time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from static_contact_common import *                                                   # noqa
H = float(sys.argv[1]) if len(sys.argv) > 1 else .05
RHO = float(sys.argv[2]) if len(sys.argv) > 2 else 2e3
DELTAS = [1e-3*float(a) for a in (sys.argv[3].split(',') if len(sys.argv) > 3 else ['0.5', '1', '2', '3'])]
DH = 1e-3*float(sys.argv[4]) if len(sys.argv) > 4 else 1e-3
OUT = ROOT/'out_inverse'/'static_contact'; OUT.mkdir(exist_ok=True)
y0_, _ = yq(X); x_touch = X.clone(); x_touch[:, 1] -= float(y0_[:, 1].min())-C0     # (just touching)
I3 = torch.eye(3, dtype=f64, device=dev)


def alm_terms(y_, lam, rho, anchors=None, lam_t=None):
    gq = y_[:, 1]-C0; t = torch.clamp(lam-rho*gq, min=0.)
    val = ((t*t).sum()-(lam*lam).sum())/(2*rho)
    if anchors is not None:
        r_ = y_[anchors[0]][:, [0, 2]]-anchors[1]; val = val+(lam_t*r_).sum()+.5*rho*(r_*r_).sum()
    return val, t > 0


# ================================================================ 0: rigid
print('== 0. rigid body on the ground (6 unknowns), weight %.4g' % WEIGHT, flush=True)
cm = (wg[:, None]*X).sum(0)/Wtot
cand0 = torch.arange(Q, device=dev)                                                 # (every cap point: a rotated body must not pass the ground anywhere)
print('   candidates: all %d cap points' % Q)


def rigid_x(p):
    t, th = p[:3], p[3:]
    z = th[0]*0
    K = torch.stack([torch.stack([z, -th[2], th[1]]), torch.stack([th[2], z, -th[0]]), torch.stack([-th[1], th[0], z])])
    a2 = (th*th).sum(); a = torch.sqrt(a2+1e-30)
    R = I3+torch.sin(a)/a*K+(1-torch.cos(a))/(a2+1e-30)*(K@K)
    return (x_touch-cm)@R.T+cm+t


def rigid_phi(p, lam, rho):
    x = rigid_x(p); y_ = x[cl[cand0]]+torch.einsum('qij,qj->qi', Fs_torch(x)[0][cl[cand0]], roff[cand0])   # (F_s of a rigid motion = R; recomputed anyway)
    val, act = alm_terms(y_, lam, rho); return val-(FG*x).sum(), y_, act


lam0 = torch.zeros(len(cand0), dtype=f64, device=dev); p = torch.zeros(6, dtype=f64, device=dev)
for it in range(25):
    for newton in range(30):
        pg = p.clone().requires_grad_(True); f, y_, act = rigid_phi(pg, lam0, RHO); gr = torch.autograd.grad(f, pg)[0]
        if float(gr.norm()) < 1e-10*WEIGHT:
            break
        Hm = torch.zeros(6, 6, dtype=f64, device=dev); h = 1e-7
        for k in range(6):
            e = torch.zeros(6, dtype=f64, device=dev); e[k] = h
            pa = (p+e).clone().requires_grad_(True); ga = torch.autograd.grad(rigid_phi(pa, lam0, RHO)[0], pa)[0]
            pb = (p-e).clone().requires_grad_(True); gb = torch.autograd.grad(rigid_phi(pb, lam0, RHO)[0], pb)[0]
            Hm[:, k] = (ga-gb)/(2*h)
        Hm = .5*(Hm+Hm.T); ev = torch.linalg.eigvalsh(Hm.cpu()).to(dev)
        E6 = torch.eye(6, dtype=f64, device=dev)
        step = -torch.linalg.solve((Hm+(max(0., -float(ev.min()))*1.001+1e-12*float(ev.abs().max()))*E6).cpu(), gr.cpu()).to(dev)
        if newton == 0:
            print('     rigid ALM %d: |grad| %.3e, H eig min %.3e max %.3e' % (it, float(gr.norm()), float(ev.min()), float(ev.max())), flush=True)
        a_ = 1.
        for _ in range(40):
            if float(rigid_phi(p+a_*step, lam0, RHO)[0]) <= float(f)+1e-4*a_*float(gr@step):
                break
            a_ *= .5
        p = p+a_*step
    f, y_, act = rigid_phi(p, lam0, RHO); gq = y_[:, 1]-C0; lam_new = torch.clamp(lam0-RHO*gq, min=0.)
    pen = float((-gq).clamp(min=0).max()); comp = float((lam_new*gq).abs().max()); dl = float((lam_new-lam0).norm()/max(float(lam_new.norm()), 1e-300))
    lam0 = lam_new
    if pen < 1e-9 and comp < 1e-12 and dl < 1e-8:
        break
x = rigid_x(p); fq = lam0[:, None]*UP
print('   %d ALM rounds: translation %s mm, rotation %s mrad; penetration %.2e mm, complementarity %.2e, active %d' % (
    it+1, np.round(1e3*p[:3].cpu().numpy(), 3), np.round(1e3*p[3:].cpu().numpy(), 3), 1e3*pen, comp, int((lam0 > 0).sum())))
mom = torch.linalg.cross(y_, fq).sum(0)+torch.linalg.cross(x, FG).sum(0)
print('   sum lambda %.6f against the weight %.6f (rel %.1e); moment of reactions + weight |sum| / |moment of the weight| %.1e; support cells %s' % (
    float(lam0.sum()), WEIGHT, abs(float(lam0.sum())-WEIGHT)/WEIGHT, float(mom.norm()/torch.linalg.cross(x, FG).sum(0).norm()), torch.unique(cl[cand0][lam0 > 0]).tolist()), flush=True)

if len(sys.argv) > 5 and sys.argv[5] == 'rigid-only':
    sys.exit(0)
# ================================================================ 1: the foot, displacement controlled
free = torch.nonzero(X[:, 1] < float(X[:, 1].min())+H)[:, 0]; nF = len(free)
fixed = torch.ones(n, dtype=torch.bool, device=dev); fixed[free] = False
candF = torch.nonzero((~fixed)[cl])[:, 0]
print('\n== 1. the foot: %d free sites (lowest %.0f mm), %d contact candidates; the rest rigidly fixed and pushed down' % (nF, 1e3*H, len(candF)), flush=True)
sel_free = lambda a: a[free]


class Foot:
    def __init__(self, rho, lam, anchors=None, lam_t=None):
        self.rho, self.lam, self.anchors, self.lam_t = rho, lam, anchors, lam_t
        self.evals = 0

    def set_clamp(self, xfix):
        self.xfix = xfix

    def full(self, xf):
        x = self.xfix.clone(); x[free] = xf; return x

    def phi(self, xf, want_act=False):
        x = self.full(xf); e, ge = W.grad(x)
        xg = x.clone().requires_grad_(True); y_ = yq(xg)[0][candF]
        val, act = alm_terms(y_, self.lam, self.rho, self.anchors, self.lam_t); val.backward(); self.evals += 1
        return float(e)+float(val), (ge+xg.grad)[free], (act if want_act else None), y_.detach()

    def hessian(self, xf, h=1e-6):
        """dense (3 nF x 3 nF) by central differences of grad phi, the active set fixed at xf's (the max() replaced by its branch)"""
        _, _, act, _ = self.phi(xf, True); lam_fix = torch.where(act, self.lam, torch.zeros_like(self.lam))
        Hm = torch.zeros(3*nF, 3*nF, dtype=f64, device=dev)
        for k in range(3*nF):
            e = torch.zeros(nF, 3, dtype=f64, device=dev); e.view(-1)[k] = h
            Hm[:, k] = (self._grad_fixed(xf+e, act, lam_fix)-self._grad_fixed(xf-e, act, lam_fix)).reshape(-1)/(2*h)
        return .5*(Hm+Hm.T)

    def _grad_fixed(self, xf, act, lam_fix):
        x = self.full(xf); e, ge = W.grad(x)
        xg = x.clone().requires_grad_(True); y_ = yq(xg)[0][candF]; gq = y_[:, 1]-C0
        t = (lam_fix-self.rho*gq)*act
        val = ((t*t).sum())/(2*self.rho)
        if self.anchors is not None:
            r_ = y_[self.anchors[0]][:, [0, 2]]-self.anchors[1]; val = val+(self.lam_t*r_).sum()+.5*self.rho*(r_*r_).sum()
        val.backward(); self.evals += 1
        return (ge+xg.grad)[free]

    def inner(self, xf, tol, max_newton=25, log=''):
        for it in range(max_newton):
            f, gr, act, _ = self.phi(xf, True); gn = float(gr.norm())
            if gn <= tol:
                return xf, gn, it
            t0 = time.time(); Hm = self.hessian(xf); ev = torch.linalg.eigvalsh(Hm)
            shift = max(0., -float(ev.min()))*1.01+1e-9*float(ev.abs().max())
            step = -torch.linalg.solve(Hm+shift*torch.eye(3*nF, dtype=f64, device=dev), gr.reshape(-1)).reshape(nF, 3)
            a_ = 1.; gd = float((gr*step).sum()); flips = 0
            for _ in range(40):
                f1, _, act1, _ = self.phi(xf+a_*step, True)
                if f1 <= f+1e-4*a_*gd:
                    flips = int((act1 != act).sum()); break
                a_ *= .5
            xf = xf+a_*step
            if log:
                print('     %s Newton %d: |grad phi| %.3e, H min eig %.3e (shift %.1e), step %.1f (%.0f s), active flips along the step %d' % (log, it, gn, float(ev.min()), shift, a_, time.time()-t0, flips), flush=True)
        f, gr, _, _ = self.phi(xf); return xf, float(gr.norm()), max_newton

    def residuals(self, xf):
        x = self.full(xf); y_ = yq(x)[0][candF]; gq = y_[:, 1]-C0
        lam_new = torch.clamp(self.lam-self.rho*gq, min=0.)
        lt = self.lam_t+self.rho*(y_[self.anchors[0]][:, [0, 2]]-self.anchors[1]) if self.anchors is not None else None
        xg = x.clone().requires_grad_(True); y2 = yq(xg)[0][candF]
        Lc = -(y2[:, 1]*lam_new).sum()
        if lt is not None:
            Lc = Lc+(lt*(y2[self.anchors[0]][:, [0, 2]]-self.anchors[1])).sum()
        Lc.backward(); JTl = -xg.grad
        e, ge = W.grad(x); gL = (ge-JTl)[free]
        clamp_force = -ge[fixed].sum(0)                                               # (the force the clamp exerts: minus the elastic force on it)
        fq = lam_new[:, None]*UP
        if lt is not None:
            fq = fq.clone(); fq[self.anchors[0], 0] += lt[:, 0]; fq[self.anchors[0], 2] += lt[:, 1]
        det = torch.linalg.det(W.tF)
        return dict(E=float(e), pen=float((-gq).clamp(min=0).max()), comp=float((lam_new*gq).abs().max()), stat=float(gL.norm()/WEIGHT), stat_inf=float(gL.abs().max()/WEIGHT),
                    lam=lam_new, lam_t=lt, react=fq.sum(0), clamp=clamp_force, active=int((lam_new > 0).sum()), det_min=float(det[free].min()), a_min=float(yq(x)[1][cap][(~fixed)[cap]].min()),
                    Fyy=W.tF[free][:, 1, 1], y=y_, rt=float((y_[self.anchors[0]][:, [0, 2]]-self.anchors[1]).norm(dim=1).max()) if self.anchors is not None else 0.)


def run_alm(prob, xf, tag, rounds=20):
    tol = 1e-3*WEIGHT
    for it in range(rounds):
        xf, gn, nit = prob.inner(xf, tol, log=tag if it == 0 else '')
        res = prob.residuals(xf)
        print('   %s outer %d: inner |grad phi| %.2e in %d Newton; after the update: pen %.3e mm, comp %.2e, KKT %.2e (max %.2e), active %d, |delta lambda| / |lambda| %.2e' % (
            tag, it, gn/WEIGHT, nit, 1e3*res['pen'], res['comp'], res['stat'], res['stat_inf'], res['active'], float((res['lam']-prob.lam).norm()/max(float(res['lam'].norm()), 1e-300))), flush=True)
        done = res['pen'] < 1e-7 and res['comp'] < 1e-10 and res['stat'] < 1e-5 and float((res['lam']-prob.lam).norm()) < 1e-6*max(float(res['lam'].norm()), 1e-300)
        prob.lam = res['lam']
        if prob.anchors is not None:
            prob.lam_t = res['lam_t']
        if done:
            break
        tol = max(tol/3, 1e-7*WEIGHT)
    return xf, res


xf = x_touch[free].clone(); prob = Foot(RHO, torch.zeros(len(candF), dtype=f64, device=dev)); results = []
for delta in DELTAS:
    xfix = x_touch.clone(); xfix[:, 1] -= delta; prob.set_clamp(xfix)
    xf, res = run_alm(prob, xf, 'd %.1f mm' % (1e3*delta))
    print('   [delta %.1f mm] reaction sum lambda %s against the clamp force %s (|diff| / |clamp| %.1e); active %d on cells %d; foot F_B yy median %.4f min %.4f; min det F_B %.4f, min cap area factor %.4f; E %.4e' % (
        1e3*delta, np.round(res['react'].cpu().numpy(), 5), np.round(res['clamp'].cpu().numpy(), 5), float((res['react']-res['clamp']).norm()/res['clamp'].norm()), res['active'],
        len(torch.unique(cl[candF][res['lam'] > 0])), float(res['Fyy'].median()), float(res['Fyy'].min()), res['det_min'], res['a_min'], res['E']), flush=True)
    results.append((delta, float(res['react'][1]), res['active'], res['pen'], res['stat']))
    torch.save(dict(xf=xf, lam=prob.lam, delta=delta), str(OUT/('foot_d%.1f.pt' % (1e3*delta))))
print('   reaction against delta: ' + ', '.join('%.1f mm: %.4g (weight %.4g)' % (1e3*d, r, WEIGHT) for d, r, *_ in results), flush=True)

# ================================================================ 2: stick on the foot
act = torch.nonzero(prob.lam > 0)[:, 0]; yA = yq(prob.full(xf))[0][candF]; anchors = (act, yA[act][:, [0, 2]].clone())
print('\n== 2. stick: %d anchored points at delta %.1f mm; the clamp pushed sideways by %.1f mm' % (len(act), 1e3*DELTAS[-1], 1e3*DH), flush=True)
probS = Foot(RHO, prob.lam.clone(), anchors, torch.zeros(len(act), 2, dtype=f64, device=dev))
xfix = x_touch.clone(); xfix[:, 1] -= DELTAS[-1]; xfix[:, 0] += DH; probS.set_clamp(xfix)
xf2, res = run_alm(probS, xf, 'stick')
print('   [stick] reactions %s against the clamp force %s (|diff| / |clamp| %.1e); tangential residual max %.3e mm; anchored points with lambda_n <= 0: %d of %d; min det F_B %.4f' % (
    np.round(res['react'].cpu().numpy(), 5), np.round(res['clamp'].cpu().numpy(), 5), float((res['react']-res['clamp']).norm()/res['clamp'].norm()), 1e3*res['rt'], int((res['lam'][act] <= 0).sum()), len(act), res['det_min']), flush=True)
torch.save(dict(xf=xf2, lam=probS.lam, lam_t=probS.lam_t, anchors=anchors), str(OUT/'foot_stick.pt'))
print('done', flush=True)
