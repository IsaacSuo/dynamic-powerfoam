"""the original fitted skin resting on the ground under hard contact constraints (the reviewer's step 1, 11-07): the
elastic model E_old(x) untouched (one-way cap, free completion, no F1 / clip / penalty / glue), the contact points
y_q(x) = x_B + F_s(x) r_q (the cap quadrature points on the model's own surface; F_s re-implemented in torch, checked
against the Warp closure), the ground as constraints solved by an augmented Lagrangian over the sites only (inner L-BFGS
on E_old - f_g . x + the ALM terms; the energy's gradient from Warp, the constraints' J^T lambda by torch autograd).
Test A (normal only): the body translated so its lowest point just touches y = C0, gravity (the forward's consistent
weight M g) raised in steps; positioning conditions on the two horizontal translations and the rotation about the
vertical (reported as forces). Test B (stick): from A's solution the active points' positions become anchors, a horizontal
load of 10 % of the weight is added, the anchored points' tangential positions held (equality, ALM), anchors fixed
throughout. Reports per solve: max penetration over all candidate points, lambda_n >= 0, complementarity max |lambda g|,
stationarity |grad L| / weight, total reaction and moment against the load, min det F_B, min cap area factor, the feet's
vertical stretch; B: the tangential position residual. Stops (saves the state) on non-convergence or inversion.
args: [rho0 (default 2e3)] [lbfgs maxiter (3000)]"""
import math
import sys
import time
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import free_surface.lion_inverse as li
from experiments.pacnerf_dem_forward import build
from free_surface.cap_contact import cap_quadrature
from free_surface.closure_warp import ClosureWarp
from free_surface.sphere_closure import ClosureTerms
RHO0 = float(sys.argv[1]) if len(sys.argv) > 1 else 2e3
MAXIT = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
f64, dev = torch.float64, 'cuda'
li.set_scene('pacnerf_e0')
cells, g, site_np = li.load_foam()
m_ = build(g, li.GT['E'], li.GT['nu'], dev, 'noslip', .25, .3, f64, kinematics='exact', nodalvol=True, nonlinear=False, ogden_alpha=2.)
st = ClosureTerms(m_, g, cells['points'], cells['radii'])
W = ClosureWarp(st)
W.set_viscosity(0., 0., li.DT); W.visc_weight = 2.; W.total_mode()
site = torch.as_tensor(site_np, device=dev).long()
W.set_ball(site, torch.as_tensor(cells['radii'], dtype=f64, device=dev)[site], (0., 1., 0.), -10., 0.)   # (the balls' plane far away: no contact in E)
W.set_cap_positions(); W.set_cap_floor(0.)
W.cm_cache = str(li.FOAM/'dem_exact_single'/'cell_moments.npz')
W.prepare()
n = W.n; X = st.X.clone(); UP = torch.tensor([0., 1., 0.], dtype=f64, device=dev); C0 = li.C0
P_ = W._pos; off, pk, dX, M2i = P_['off'], P_['pk'], P_['dX'], P_['M2i']
cap = torch.nonzero(off[1:] > off[:-1])[:, 0]
pc = torch.repeat_interleave(cap, (off[cap+1]-off[cap]))                            # (the fit's pairs: cell, neighbour)
kap = float(W.tkap[0]); t1, t2, mA, PA = st.t1, st.t2, st.mA, st.PA
cl, roff, wq = cap_quadrature(cells['points'], cells['radii'], st.c_body.cpu().numpy(), cache=str(li.FOAM/'dem_exact_single'/'cap_quadrature.npz'), verbose=False)
cl = torch.as_tensor(cl, device=dev).long(); roff = torch.as_tensor(roff, dtype=f64, device=dev); wq = torch.as_tensor(wq, dtype=f64, device=dev)
Q = len(cl)
wg = torch.as_tensor(W.cmass['rowsum'], dtype=f64, device=dev)                      # (the consistent weight per site, per unit density)
Wtot = float(wg.sum()); FG = -9.8*wg[:, None]*UP
print('sites %d, cap cells %d, contact candidates %d; weight %.4g (per unit density, x 9.8: %.4g)' % (n, len(cap), Q, Wtot, 9.8*Wtot), flush=True)


def Fs_torch(x):
    """every cell's F_s (I off the caps), the free completion in torch: G = sum_e (x_k - x_B) (x) dX_e M2^-1, c1 = G t1,
    c2 = G t2, n_t = c1 x c2, s = 1 - kap (sqrt(|c1|^2 + |c2|^2 + 2 |n_t|) - 2), F_s = G P + s n_t / |n_t| m^T"""
    Sx = torch.zeros(n, 3, 3, dtype=x.dtype, device=dev).index_add(0, pc, (x[pk]-x[pc])[:, :, None]*dX[:, None, :])
    G = Sx@M2i
    c1 = torch.einsum('nij,nj->ni', G, t1); c2 = torch.einsum('nij,nj->ni', G, t2)
    nt = torch.linalg.cross(c1, c2, dim=1); a = nt.norm(dim=1)
    s = 1-kap*(torch.sqrt((c1*c1).sum(1)+(c2*c2).sum(1)+2*a)-2)
    Fs = G@PA+(nt/a.clamp(min=1e-300)[:, None]*s[:, None])[:, :, None]*mA[:, None, :]
    I3 = torch.eye(3, dtype=x.dtype, device=dev).expand(n, 3, 3)
    iscap = torch.zeros(n, dtype=torch.bool, device=dev); iscap[cap] = True
    return torch.where(iscap[:, None, None], Fs, I3), a


def yq(x):
    Fs, a = Fs_torch(x)
    return x[cl]+torch.einsum('qij,qj->qi', Fs[cl], roff), a


# ---- the torch F_s against the Warp closure, J^T lambda by autograd against virtual work, resultant and moment
gen = torch.Generator(device='cpu').manual_seed(0)
rnd = lambda *s: torch.randn(*s, generator=gen, dtype=f64).to(dev)
xt = X+2e-3*rnd(n, 3)
W.grad(xt); Fs_w = W.tFs.clone(); Fs_t, _ = Fs_torch(xt)
print('torch F_s against Warp: max |diff| %.2e on the caps' % float((Fs_t[cap]-Fs_w[cap]).abs().max()), flush=True)
lam_t = rnd(Q, 3)*(torch.rand(Q, generator=gen, dtype=f64).to(dev) < .01)[:, None]   # (a test force on 1 % of the points)
xg = xt.clone().requires_grad_(True); y, _ = yq(xg); (y*lam_t).sum().backward(); JTl = xg.grad.clone()
v = rnd(n, 3); v /= v.abs().max(); h = 1e-6
work = float((((yq(xt+h*v)[0]-yq(xt-h*v)[0])/(2*h))*lam_t).sum())
mom_s = torch.linalg.cross(xt, JTl).sum(0); mom_q = torch.linalg.cross(y.detach(), lam_t).sum(0)
print('J^T lambda: virtual work (J^T lambda) . v %+.10e against sum lambda . dy %+.10e (rel %.1e); resultant |sum J^T lambda - sum lambda| %.1e; moment sum x_k x (J^T lambda)_k against sum y_q x lambda_q: |diff| / |moment| %.1e' % (
    float((JTl*v).sum()), work, abs(float((JTl*v).sum())-work)/abs(work), float((JTl.sum(0)-lam_t.sum(0)).norm()), float((mom_s-mom_q).norm()/mom_q.norm())), flush=True)

# ---- the ALM solve
def solve(x0, load, lam_n, rho, anchors=None, lam_t=None, pos=True, outer=12, tag=''):
    """min E(x) - load . x s.t. y_q . up - C0 >= 0 (all points), anchors: (index, a (K, 2)) the tangential positions held.
    ALM: phi = E - load . x + sum_q [max(0, lam_q - rho g_q)^2 - lam_q^2] / (2 rho) + (stick) sum_k [lam_t . r_k + rho/2 |r_k|^2],
    r_k = (y_k - a_k)_h; positioning (pos): kp/2 (|mean_w dx_h|^2 + theta_v^2 ...) as forces. Returns x, lam_n, lam_t, report"""
    x = x0.clone(); kp = 1e3*Wtot
    Xh = X[:, [0, 2]]; c_h = (wg[:, None]*Xh).sum(0)/Wtot; rh = Xh-c_h
    for it in range(outer):
        def phi(xf):
            xx = torch.as_tensor(xf.reshape(n, 3), device=dev)
            e, ge = W.grad(xx)
            xg_ = xx.clone().requires_grad_(True)
            y_, _ = yq(xg_); gq = y_[:, 1]-C0
            t = torch.clamp(lam_n-rho*gq, min=0.)
            val = e-(load*xx).sum()+((t*t).sum()-(lam_n*lam_n).sum())/(2*rho)
            cterm = ((t*t).sum())/(2*rho)
            if anchors is not None:
                r_ = y_[anchors[0]][:, [0, 2]]-anchors[1]
                cterm = cterm+(lam_t*r_).sum()+.5*rho*(r_*r_).sum()
                val = val+(lam_t*r_).sum()+.5*rho*(r_*r_).sum()
            if pos:
                dh = xg_[:, [0, 2]]-Xh; mh = (wg[:, None]*dh).sum(0)/Wtot
                th = (wg*(rh[:, 0]*dh[:, 1]-rh[:, 1]*dh[:, 0])).sum()/(wg*(rh*rh).sum(1)).sum()
                cterm = cterm+.5*kp*((mh*mh).sum()+th*th); val = val+.5*kp*((mh*mh).sum()+th*th)
            cterm.backward()
            gr = ge-load+xg_.grad
            return float(val), gr.cpu().numpy().ravel()
        t0 = time.time()
        r = minimize(phi, x.cpu().numpy().ravel(), jac=True, method='L-BFGS-B', options=dict(maxiter=MAXIT, maxfun=2*MAXIT, gtol=1e-12, ftol=1e-16, maxcor=30))
        x = torch.as_tensor(r.x.reshape(n, 3), device=dev)
        y_, a_ = yq(x); gq = y_[:, 1]-C0
        lam_new = torch.clamp(lam_n-rho*gq, min=0.)
        pen = float((-gq).clamp(min=0).max()); comp = float((lam_new*gq).abs().max())
        # stationarity of the Lagrangian with the updated multipliers: grad E - load - J^T lambda (+ positioning)
        xg_ = x.clone().requires_grad_(True); y2, _ = yq(xg_)
        Lc = -(y2[:, 1]*lam_new).sum()
        if anchors is not None:
            r_ = y2[anchors[0]][:, [0, 2]]-anchors[1]; lam_t_new = lam_t+rho*r_
            Lc = Lc+(lam_t_new*r_).sum()
        if pos:
            dh = xg_[:, [0, 2]]-Xh; mh = (wg[:, None]*dh).sum(0)/Wtot
            th = (wg*(rh[:, 0]*dh[:, 1]-rh[:, 1]*dh[:, 0])).sum()/(wg*(rh*rh).sum(1)).sum()
            Lc = Lc+.5*kp*((mh*mh).sum()+th*th)
        Lc.backward()
        e, ge = W.grad(x)
        gL = ge-load+xg_.grad
        stat = float(gL.norm()/(9.8*Wtot))
        fpos = float(xg_.grad.norm()) if pos else 0.
        rt = float(r_.norm(dim=1).max()) if anchors is not None else 0.
        print('  %s outer %d: rho %.3g, L-BFGS %d its (%.0f s, %s), E %.6e; max penetration %.3e mm, active %d, complementarity %.2e, stationarity |grad L| / weight %.2e, tangential residual %.2e mm' % (
            tag, it, rho, r.nit, time.time()-t0, r.message[:30], e, 1e3*pen, int((lam_new > 0).sum()), comp, stat, 1e3*rt), flush=True)
        lam_n = lam_new
        if anchors is not None:
            lam_t = lam_t_new
        if pen < 1e-6 and comp < 1e-9 and stat < 1e-4 and rt < 1e-6:
            break
        rho = min(rho*3, 1e8)
    W.grad(x); F = W.tF; det = torch.linalg.det(F)
    rep = dict(x=x, lam_n=lam_n, lam_t=lam_t, pen=pen, comp=comp, stat=stat, det_min=float(det.min()), a_min=float(a_[cap].min()), fpos=fpos,
               react=(lam_new[:, None]*UP).sum(0) if anchors is None else (lam_new[:, None]*UP).sum(0)+torch.stack([lam_t.sum(0)[0], torch.zeros((), dtype=f64, device=dev), lam_t.sum(0)[1]]),
               y=y_, Fyy=F[:, 1, 1])
    return rep


def report(rep, load, tag):
    x, y_ = rep['x'], rep['y']; lam = rep['lam_n']
    fq = lam[:, None]*UP
    if rep['lam_t'] is not None:
        fq = fq.clone(); fq[anchors[0], 0] += rep['lam_t'][:, 0]; fq[anchors[0], 2] += rep['lam_t'][:, 1]
    tot = fq.sum(0); L = load.sum(0)
    mom_c = torch.linalg.cross(y_, fq).sum(0); mom_l = torch.linalg.cross(x, load).sum(0)
    low = torch.nonzero(lam > 0)[:, 0]; feet = torch.unique(cl[low])
    print('== %s: total reaction %s against -load %s (|diff| / |load| %.1e); moment of the reactions + load about the origin |sum| / |moment of the load| %.1e; positioning force %.2e of the weight' % (
        tag, np.round(tot.cpu().numpy(), 5), np.round(-L.cpu().numpy(), 5), float((tot+L).norm()/L.norm()), float((mom_c+mom_l).norm()/mom_l.norm()), rep['fpos']/(9.8*Wtot)))
    print('   max penetration %.3e mm, complementarity %.2e, stationarity %.2e; min det F_B %.4f, min cap area factor %.4f; touching cells %d, their F_B yy median %.4f min %.4f; lambda >= 0 all: %s' % (
        1e3*rep['pen'], rep['comp'], rep['stat'], rep['det_min'], rep['a_min'], len(feet), float(rep['Fyy'][feet].median()), float(rep['Fyy'][feet].min()), bool((lam >= 0).all())), flush=True)


# ---- test A: normal contact, gravity in steps
y0_, _ = yq(X); x = X.clone(); x[:, 1] -= float(y0_[:, 1].min())-C0                  # (just touching)
lam_n = torch.zeros(Q, dtype=f64, device=dev)
ok = True
for lf in (.25, .5, .75, 1.):
    rep = solve(x, lf*FG, lam_n, RHO0, tag='A g x %.2f' % lf)
    x, lam_n = rep['x'], rep['lam_n']
    report(rep, lf*FG, 'A, gravity x %.2f' % lf)
    if rep['det_min'] <= 0 or rep['stat'] > 1e-3 or rep['pen'] > 1e-5:
        ok = False; print('STOP: test A did not converge or inverted (det %.3f, stat %.2e, pen %.2e)' % (rep['det_min'], rep['stat'], rep['pen'])); break
torch.save(dict(x=x, lam_n=lam_n), str(ROOT/'out_inverse'/'static_contact_A.pt'))
if ok:
    # ---- test B: stick anchors from A's active points, a horizontal load of 10 % of the weight
    act = torch.nonzero(lam_n > 0)[:, 0]; yA, _ = yq(x); anchors = (act, yA[act][:, [0, 2]].clone())
    loadB = FG+torch.tensor([-.1*9.8, 0., 0.], dtype=f64, device=dev)*wg[:, None]
    print('test B: %d anchored points on %d cells; horizontal load %.4g (10 %% of the weight)' % (len(act), len(torch.unique(cl[act])), .1*9.8*Wtot), flush=True)
    rep = solve(x, loadB, lam_n, RHO0, anchors=anchors, lam_t=torch.zeros(len(act), 2, dtype=f64, device=dev), pos=False, tag='B')
    report(rep, loadB, 'B, stick + 10 % horizontal load')
    print('   anchored points with lambda_n <= 0 (pulling or released): %d of %d; tangential residual max %.3e mm' % (int((rep['lam_n'][act] <= 0).sum()), len(act), 1e3*float((rep['y'][act][:, [0, 2]]-anchors[1]).norm(dim=1).max())))
    torch.save(dict(x=rep['x'], lam_n=rep['lam_n'], lam_t=rep['lam_t'], anchors=anchors), str(ROOT/'out_inverse'/'static_contact_B.pt'))
print('done', flush=True)
