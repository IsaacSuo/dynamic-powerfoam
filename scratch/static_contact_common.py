"""shared by the static contact tests (static_contact_rest.py, static_contact_newton.py): the original fitted skin
E_old(x) (one-way cap, free completion; no F1 / clip / penalty / glue), the contact map y_q(x) = x_B + F_s(x) r_q on the
cap quadrature points (F_s in torch, = the Warp closure's), the consistent weight, the augmented objective
    phi(x) = E(x) - load . x + sum_q [max(0, lam_q - rho g_q)^2 - lam_q^2] / (2 rho)        (g_q = y_q . up - C0 >= 0)
           + sum_k [lam_t,k . r_k + rho / 2 |r_k|^2]  (stick: r_k = (y_k - a_k)_h)  + kp / 2 (|mean_w dx_h|^2 + theta_v^2)  (positioning)
and the residuals: |grad phi| (the inner objective), the KKT stationarity |grad E - load - J^T lambda + f_pos| with the
multipliers stated (before / after the ALM update), max penetration, complementarity max |lambda g|, the positioning
force, the tangential residual. Everything per unit density."""
import math
import sys
import time
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import free_surface.lion_inverse as li
from experiments.pacnerf_dem_forward import build
from free_surface.cap_contact import cap_quadrature
from free_surface.closure_warp import ClosureWarp
from free_surface.sphere_closure import ClosureTerms
f64, dev = torch.float64, 'cuda'
li.set_scene('pacnerf_e0')
cells, g, site_np = li.load_foam()
m_ = build(g, li.GT['E'], li.GT['nu'], dev, 'noslip', .25, .3, f64, kinematics='exact', nodalvol=True, nonlinear=False, ogden_alpha=2.)
st = ClosureTerms(m_, g, cells['points'], cells['radii'])
W = ClosureWarp(st)
W.set_viscosity(0., 0., li.DT); W.visc_weight = 2.; W.total_mode()
site = torch.as_tensor(site_np, device=dev).long()
W.set_ball(site, torch.as_tensor(cells['radii'], dtype=f64, device=dev)[site], (0., 1., 0.), -10., 0.)
W.set_cap_positions(); W.set_cap_floor(0.)
W.cm_cache = str(li.FOAM/'dem_exact_single'/'cell_moments.npz')
W.prepare()
n = W.n; X = st.X.clone(); UP = torch.tensor([0., 1., 0.], dtype=f64, device=dev); C0 = li.C0
P_ = W._pos; off, pk, dX, M2i = P_['off'], P_['pk'], P_['dX'], P_['M2i']
cap = torch.nonzero(off[1:] > off[:-1])[:, 0]
pc = torch.repeat_interleave(cap, (off[cap+1]-off[cap]))
kap = float(W.tkap[0]); t1, t2, mA, PA = st.t1, st.t2, st.mA, st.PA
iscap = torch.zeros(n, dtype=torch.bool, device=dev); iscap[cap] = True
import os
CAP_N = int(os.environ.get('CAP_N', '2000'))                                        # (points per sphere; 2000 = the cached default)
cl, roff, wq = cap_quadrature(cells['points'], cells['radii'], st.c_body.cpu().numpy(), N=CAP_N,
                              cache=str(li.FOAM/'dem_exact_single'/('cap_quadrature.npz' if CAP_N == 2000 else 'cap_quadrature_N%d.npz' % CAP_N)), verbose=False)
cl = torch.as_tensor(cl, device=dev).long(); roff = torch.as_tensor(roff, dtype=f64, device=dev); wq = torch.as_tensor(wq, dtype=f64, device=dev)
Q = len(cl)
wg = torch.as_tensor(W.cmass['rowsum'], dtype=f64, device=dev)
Wtot = float(wg.sum()); WEIGHT = 9.8*Wtot; FG = -9.8*wg[:, None]*UP
Xh = X[:, [0, 2]]; c_h = (wg[:, None]*Xh).sum(0)/Wtot; rh = Xh-c_h; Irh = float((wg*(rh*rh).sum(1)).sum())
KP = 1e3*Wtot


def Fs_torch(x):
    Sx = torch.zeros(n, 3, 3, dtype=x.dtype, device=dev).index_add(0, pc, (x[pk]-x[pc])[:, :, None]*dX[:, None, :])
    G = Sx@M2i
    c1 = torch.einsum('nij,nj->ni', G, t1); c2 = torch.einsum('nij,nj->ni', G, t2)
    nt = torch.linalg.cross(c1, c2, dim=1); a = nt.norm(dim=1)
    s = 1-kap*(torch.sqrt((c1*c1).sum(1)+(c2*c2).sum(1)+2*a)-2)
    Fs = G@PA+(nt/a.clamp(min=1e-300)[:, None]*s[:, None])[:, :, None]*mA[:, None, :]
    return torch.where(iscap[:, None, None], Fs, torch.eye(3, dtype=x.dtype, device=dev).expand(n, 3, 3)), a


def yq(x):
    Fs, a = Fs_torch(x)
    return x[cl]+torch.einsum('qij,qj->qi', Fs[cl], roff), a


class Problem:
    """the augmented objective for a load, multipliers lam_n (Q,), optional stick anchors (idx, a_h (K, 2)) with lam_t (K, 2),
    rho, positioning on / off"""
    def __init__(self, load, rho, lam_n, anchors=None, lam_t=None, pos=True):
        self.load, self.rho, self.lam_n, self.anchors, self.lam_t, self.pos = load, float(rho), lam_n, anchors, lam_t, pos
        self.evals = 0; self.t_warp = self.t_torch = 0.

    def terms(self, xg, want_mask=False):
        """the constraint + positioning part of phi at xg (requires grad) and the active mask of the max()"""
        y_, a_ = yq(xg); gq = y_[:, 1]-C0
        t = torch.clamp(self.lam_n-self.rho*gq, min=0.)
        val = ((t*t).sum()-(self.lam_n*self.lam_n).sum())/(2*self.rho)
        if self.anchors is not None:
            r_ = y_[self.anchors[0]][:, [0, 2]]-self.anchors[1]
            val = val+(self.lam_t*r_).sum()+.5*self.rho*(r_*r_).sum()
        if self.pos:
            dh = xg[:, [0, 2]]-Xh; mh = (wg[:, None]*dh).sum(0)/Wtot
            th = (wg*(rh[:, 0]*dh[:, 1]-rh[:, 1]*dh[:, 0])).sum()/Irh
            val = val+.5*KP*((mh*mh).sum()+th*th)
        return val, (t > 0) if want_mask else None

    def phi(self, x, want_mask=False):
        """phi and grad phi (torch); the active mask of the max() if asked"""
        t0 = time.time(); e, ge = W.grad(x); t1_ = time.time()
        xg = x.clone().requires_grad_(True)
        val, mask = self.terms(xg, want_mask); val.backward()
        self.t_warp += t1_-t0; self.t_torch += time.time()-t1_; self.evals += 1
        return float(e)+float(val)-float((self.load*x).sum()), ge-self.load+xg.grad, mask

    def pos_force(self, x):
        if not self.pos:
            return torch.zeros_like(x)
        xg = x.clone().requires_grad_(True)
        dh = xg[:, [0, 2]]-Xh; mh = (wg[:, None]*dh).sum(0)/Wtot
        th = (wg*(rh[:, 0]*dh[:, 1]-rh[:, 1]*dh[:, 0])).sum()/Irh
        (.5*KP*((mh*mh).sum()+th*th)).backward()
        return xg.grad

    def residuals(self, x, updated=True):
        """at x: penetration, the multipliers (updated: lam <- max(0, lam - rho g), lam_t <- lam_t + rho r; else the current),
        complementarity, the KKT stationarity |grad E - load - J^T lambda + f_pos| / weight, the positioning force, the
        reactions' resultant and moment, the tangential residual"""
        y_, a_ = yq(x); gq = y_[:, 1]-C0
        lam = torch.clamp(self.lam_n-self.rho*gq, min=0.) if updated else self.lam_n
        lt = None
        if self.anchors is not None:
            r_ = y_[self.anchors[0]][:, [0, 2]]-self.anchors[1]
            lt = self.lam_t+self.rho*r_ if updated else self.lam_t
        xg = x.clone().requires_grad_(True); y2, _ = yq(xg)
        Lc = -(y2[:, 1]*lam).sum()
        if lt is not None:
            Lc = Lc+(lt*(y2[self.anchors[0]][:, [0, 2]]-self.anchors[1])).sum()
        Lc.backward(); JTl = -xg.grad
        e, ge = W.grad(x); fpos = self.pos_force(x)
        gL = ge-self.load-JTl+fpos
        fq = lam[:, None]*UP
        if lt is not None:
            fq = fq.clone(); fq[self.anchors[0], 0] += lt[:, 0]; fq[self.anchors[0], 2] += lt[:, 1]
        det = torch.linalg.det(W.tF)
        return dict(E=float(e), pen=float((-gq).clamp(min=0).max()), comp=float((lam*gq).abs().max()), stat=float(gL.norm()/WEIGHT), stat_inf=float(gL.abs().max()/WEIGHT),
                    fpos=float(fpos.sum(0).norm()/WEIGHT), fpos_max=float(fpos.abs().max()/WEIGHT), react=fq.sum(0), load=self.load.sum(0),
                    mom=(torch.linalg.cross(y_, fq).sum(0)+torch.linalg.cross(x, self.load).sum(0)-torch.linalg.cross(x, fpos).sum(0)), mom_load=torch.linalg.cross(x, self.load).sum(0),
                    active=int((lam > 0).sum()), lam=lam, lam_t=lt, det_min=float(det.min()), a_min=float(a_[cap].min()), Fyy=W.tF[:, 1, 1].clone(), y=y_,
                    rt=float((y_[self.anchors[0]][:, [0, 2]]-self.anchors[1]).norm(dim=1).max()) if self.anchors is not None else 0.)

    def update(self, res):
        self.lam_n = res['lam']
        if self.anchors is not None:
            self.lam_t = res['lam_t']


def report(res, tag):
    print('   [%s] E %.6e; penetration max %.3e mm, active %d, complementarity %.2e; KKT stationarity |grad L| / weight %.2e (max entry %.2e); positioning force %.2e of the weight (max entry %.2e)' % (
        tag, res['E'], 1e3*res['pen'], res['active'], res['comp'], res['stat'], res['stat_inf'], res['fpos'], res['fpos_max']), flush=True)
    print('        reactions %s against -load %s (|diff| / |load| %.1e); moment balance |sum| / |moment of the load| %.1e; min det F_B %.4f, min cap area factor %.4f%s' % (
        np.round(res['react'].cpu().numpy(), 5), np.round(-res['load'].cpu().numpy(), 5), float((res['react']+res['load']).norm()/res['load'].norm()),
        float(res['mom'].norm()/res['mom_load'].norm()), res['det_min'], res['a_min'], ('; tangential residual max %.3e mm' % (1e3*res['rt'])) if res['rt'] else ''), flush=True)
