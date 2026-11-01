"""ring_yield.py under a prescribed uniform motion, where everything has a closed form: no solve, no contact, no comparison with another code. All sites move by x = Q(s) F(s) X, F = diag(l, l^-1/2, l^-1/2), l up to --lmax, back to where the stress is zero, then --loops elastic loops; Q a rotation with s (--rotate, degrees over the path). On the deep cells, read:
  the limit       after yield |dev sym H| = e_y, whatever the path; the stress zero at the analytic residual stretch, ln l_res = ln l_max - e_y / (3/2)^1/2
  plastic work    per unit plastic strain and volume against s_y: on the rest cut, M carries the plastic stretch U_p = diag(e^eps, e^(-eps/2), e^(-eps/2)), M = 2 mu V U_p^2 in the sum, so (2 e^(2 eps) + e^-eps) / 3 = 1 + eps + ...: a first-order error of the flow stress. On the cut: 1
  elastic energy  1/2 tr(e M e^T) over mu V |e|^2, e the analytic stretch; a closed elastic loop's work against its largest energy: 0 if the elastic part is a potential
  python free_surface/ring_yield_uniform.py --foam <dir with dem_exact_single> [--ey .01] [--eps-cut 0] [--lmax 1.2] [--steps 60] [--loops 1] [--rotate 0]"""
import argparse
import math
from pathlib import Path

import numpy as np
import torch

from free_surface.ring_yield import RingYield

f64, dev = torch.float64, 'cuda'
a_ = argparse.ArgumentParser()
a_.add_argument('--foam', type=Path, required=True)
a_.add_argument('--ey', type=float, default=.01)
a_.add_argument('--eps-cut', type=float, default=0.)
a_.add_argument('--lmax', type=float, default=1.2)
a_.add_argument('--steps', type=int, default=60)
a_.add_argument('--loops', type=int, default=1)
a_.add_argument('--rotate', type=float, default=0.)
a_.add_argument('--E', type=float, default=23.00838)
a_.add_argument('--nu', type=float, default=.3008)
a_.add_argument('--every', type=int, default=10)
args = a_.parse_args()
g = np.load(args.foam/'dem_exact_single'/'geometry.npz'); cl = np.load(args.foam/'dem_exact_single'/'cells.npz')
X0, r0, V0, C0 = g['points'], cl['radii'], g['masses'], g['contacts']; n = len(X0)
mu_ = args.E/(2*(1+args.nu))
sy_ = 2*mu_*args.ey                                                                  # (the code's s_y = (2 / 3)^1/2 sigma_y, e_y = s_y / 2 mu)
W = RingYield(X0, r0, V0, g['pairs'], g['areas'], C0, args.E, args.nu, sy_/math.sqrt(2/3), theta=.5, kappa=5., eps_cut=args.eps_cut, ledger=True)
W.set_ball((0., 1., 0.), -100., 0.)
W.start()
W.prepare()
print('e_y %.5f (asked %.5f), s_y %.5f; %d cells' % (W.ey, args.ey, W.sy, n), flush=True)
X = W.X.clone()
Xc = (W.V0[:, None]*X).sum(0)/W.V0.sum()
# the deep cells: two walls from a cap on the rest walls
cap = W.tul['Mc'].abs().sum((1, 2)) > 0
lo, hi = W.tlo, W.thi
near = cap.clone()
for _ in range(2):
    nx = near.clone()
    nx[lo[near[hi]]] = True; nx[hi[near[lo]]] = True
    near = nx
deep = ~near
print('deep cells %d of %d, volume %.3f of the body' % (int(deep.sum()), n, float(W.V0[deep].sum()/W.V0.sum())), flush=True)
sq32 = math.sqrt(1.5)
ly = W.ey/sq32                                                                       # (the log stretch at yield along this path)
lres = math.log(args.lmax)-ly


def path():
    """(log stretch) by steps: up, back to the zero of the stress, then elastic loops"""
    lm = math.log(args.lmax)
    p = list(np.linspace(0., lm, args.steps+1))
    p += list(np.linspace(lm, lres, max(args.steps//4, 4)+1)[1:])
    for _ in range(args.loops):
        amp = .8*ly
        p += list(lres+amp*np.sin(np.linspace(0., 2*np.pi, args.steps//2+1)[1:]))
    return np.array(p)


P = path()
ax = torch.tensor([.3, .5, .81], dtype=f64, device=dev); ax = ax/ax.norm()


def motion(k):
    l = math.exp(P[k])
    F = torch.diag(torch.tensor([l, l**-.5, l**-.5], dtype=f64, device=dev))
    a = math.radians(args.rotate)*k/(len(P)-1)
    K = torch.tensor([[0., -ax[2], ax[1]], [ax[2], 0., -ax[0]], [-ax[1], ax[0], 0.]], dtype=f64, device=dev)
    Q = torch.eye(3, dtype=f64, device=dev)+math.sin(a)*K+(1-math.cos(a))*K@K
    return Xc+(X-Xc)@(Q@F).T


def analytic(l_now, lp):
    """the elastic log stretch's size, signed, along the path, and the plastic log stretch, given the last ones"""
    e = l_now-lp
    if abs(e) > ly:
        lp = l_now-math.copysign(ly, e)
        e = l_now-lp
    return e, lp


lp = 0.
Wp_deep = 0.; Wa = 0.
E_max = 0.; loop_w = 0.; in_loop = False
x = motion(0)
W.update_ul(x)
rows = []
for k in range(1, len(P)):
    xn = x
    x = motion(k)
    W.update_ul(xn)
    z = .5*(xn+x)
    W.grad(z); W.reread(); _, gz = W.grad(z)
    Y, T = W.ty, W.tr
    glead = W._gid == torch.arange(n, device=dev)
    dgroup = torch.zeros(n, dtype=torch.bool, device=dev); dgroup[W._gid[~deep]] = True    # (a group with any cell not deep is out)
    use = glead & ~dgroup
    R, Ff, M, be = Y['Rz'], Y['Ff'], Y['Mtt'], Y['be']
    Hn = Y['Hn']; He = 2*(R.transpose(1, 2)@Ff-W.I3)-Hn
    trM = M.diagonal(dim1=1, dim2=2).sum(1).clamp(min=1e-300)
    vd = He.diagonal(dim1=1, dim2=2).sum(1)/3
    Hed = He-vd[:, None, None]*W.I3
    Hnd = Y['Hd']
    wc = .5*(1-be)*(((Hnd@M)*Hed).sum((1, 2))+be*((Hed@M)*Hed).sum((1, 2)))
    Vg = 1/Y['Vgi'].clamp(min=1e-300)
    V_use = float(Vg[use].sum())
    lp0 = lp
    e_a, lp = analytic(P[k], lp)
    dl = abs(lp-lp0)
    Wp_deep += float(wc[use].sum()); Wa += W.sy*sq32*dl*V_use
    rd = (Y['ei'][use]/W.ey)
    # a loop's work: -g . dx over the deep cells' interior sites would need the walls' share: the whole body's internal work is read
    dw = float((gz*(x-xn)).sum())
    if k % args.every == 0:
        gs = gz.norm(dim=1)[deep]/(W.sy*W.V0[deep]**(2/3))
        print('    the plastic patch test: |g| / (s_y V^2/3) on the deep sites, median %.4f, 90 %% %.4f, 99 %% %.4f, max %.4f; net moment of the whole force %.3e over s_y V^4/3 ... %.3e' % (float(gs.median()), float(torch.quantile(gs, .9)), float(torch.quantile(gs, .99)), float(gs.max()), float(torch.linalg.cross(z-Xc, gz).sum(0).norm()), float(W.sy*W.V0.median()**(4/3))), flush=True)
        Mm = (M[use].sum(0)/(2*W.mu*V_use)).cpu().numpy()
        Hmn = Hnd[use].mean(0).cpu().numpy(); Hme = Hed[use].mean(0).cpu().numpy()
        wcn = float(wc[use].sum())
        print('    deep, mean: M / 2 mu V diag %s, off %.4f; H_n dev diag %s; H_e dev diag %s; beta median %.4f; this step W_c / (s_y sq32 dl V) %.4f; tr(H_e M H_n) / V 2 mu e_y^2 %.4f; V read / V0 %.4f' % (np.round(np.diag(Mm), 4).tolist(), float(np.abs(Mm-np.diag(np.diag(Mm))).max()), np.round(np.diag(Hmn), 5).tolist(), np.round(np.diag(Hme), 5).tolist(), float(be[use].median()), wcn/max(W.sy*sq32*dl*V_use, 1e-300), float(((Hed[use]@M[use])*Hnd[use]).sum())/(V_use*2*W.mu*W.ey**2), float((W.tul['Vi'][use].pow(-1)).sum()/W.V0[use].sum())), flush=True)
    W.end_ul(.5)
    F_t = W._Ft[use]
    Us = torch.linalg.svdvals(F_t.cpu())
    Hend = .5*(F_t+F_t.transpose(1, 2))
    # the stored state, deep cells: |dev log U_e| against the analytic
    lg = torch.log(Us); dv = lg-lg.mean(1, keepdim=True)
    dn = dv.norm(dim=1)
    rows.append((k, P[k], e_a, lp, float(dn.median()), float(torch.quantile(dn, .01)), float(torch.quantile(dn, .99))))
    a_e = abs(e_a)*sq32
    # the energy of the deep cells at the step's end, the tensor's, on M after the return's q: read at the next start
    if k % args.every == 0 or k == len(P)-1:
        ratio = Wp_deep/Wa if Wa > 0 else 0.
        eps = lp
        print('step %3d, ln l %+.4f: analytic |dev log U_e| %.5f, read median %.5f, 1..99 %% %.5f..%.5f; reading / e_y at z median %.3f, max %.2f; deep plastic work / s_y dV %.4f (rest cut would be %.4f at eps %.3f); cut made %s; springs new %d' % (
            k, P[k], a_e, float(dn.median()), float(torch.quantile(dn, .01)), float(torch.quantile(dn, .99)), float(rd.median()), float(rd.max()), ratio, (2*math.exp(2*eps)+math.exp(-eps))/3, eps, ['%.2e' % v for v in W.Ecut_parts], W.ul_stats.get('born', 0)), flush=True)
    if k >= args.steps+max(args.steps//4, 4):
        loop_w += dw
print('the loops: the body\'s internal work %.4e' % loop_w, flush=True)
E_el = mu_*float(W.V0.sum())*(ly*sq32)**2
print('  against mu V (e at the limit)^2 of the whole body %.4e: ratio %.4f' % (E_el, loop_w/E_el), flush=True)
print('done', flush=True)
