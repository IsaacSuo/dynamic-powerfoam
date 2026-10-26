"""Implicit-midpoint drop of the part with the skinless sphere model (sphere_closure.py), torch prototype before a Warp
port (2026-10-26). The step's midpoint z of the bodies minimises
  Pi(z) = 2 / h^2 sum m |z - x_n - h v_n / 2|^2 - f . (z - x_n) + E(z) + contact(z) + glue(z),
x_n+1 = 2 z - x_n, v_n+1 = 4 (z - x_n) / h - v_n; E with F_B from the closure (warm-started from the last root).
  contact  each candidate cell's lowest point x* = x + F_B d (d the rest lever: the sphere's bottom when exposed, else the
           cap's lowest skin point), gap g = n . x* - c0. The cap may leave its free-surface state under its own load
           (energy 1/2 dq^T V_cap diag(lam + 2 mu, mu, mu) dq, moving x* only); in series with a rigid ground that is a
           one-sided spring of the cap's stiffness, 1/2 k_n min(0, g)^2, k_n = (lam + 2 mu) A_cap / 3 r: no multiplier
           passes, no penalty parameter
  glue     (--mu) a cell pressed at the step's start (p = k_n max(0, -g_n) > 0) holds an anchor a; the tangential offset
           s = P (x* - a) costs 1/2 k_t |s|^2 up to the Coulomb bound mu p (then mu p (|s| - s_y / 2), s_y = mu p / k_t),
           k_t = 4 mu_s A_cap / 3 r (the cap's shear compliance; the glue itself rigid); after the step the anchor slides to
           the bound, and is dropped when the cell leaves the ground
  solver   Newton-CG on Pi: products by forward differences of the gradient, block-Jacobi preconditioner (the rest K_bb +
           4 m / h^2 + k_n n n^T on pressed cells), Armijo backtracking; to |grad Pi| <= tol |m g| (the body's weight)
Per step: products (gradient evaluations, line search included), Newton iterations, pressed cells, the largest cap
compression -g / r, and H = kinetic + elastic + contact + glue + gravity against its start.
  PYTHONPATH=<main repo>:<worktree> python free_surface/sphere_closure_drop.py [--mu 0.5]"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

from experiments.pacnerf_dem_forward import build
from free_surface.sphere_closure import ClosureTerms
from free_surface.sphere_closure_modes import closure_stiffness

f64, dev = torch.float64, 'cuda'


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument('--part', type=Path, default=Path('output/test_part'))
    a_.add_argument('--h', type=float, default=1/1200)
    a_.add_argument('--T', type=float, default=.1)
    a_.add_argument('--v0', type=float, default=1., help='downward start speed (m/s)')
    a_.add_argument('--gap0', type=float, default=.002, help='start height of the lowest point above the ground (m)')
    a_.add_argument('--mu', type=float, default=0., help='Coulomb friction (glue); 0: frictionless')
    a_.add_argument('--tol', type=float, default=1e-3)
    a_.add_argument('--band', type=float, default=.015, help='contact candidates: lowest points within this of the lowest (m)')
    a_.add_argument('--tag', default='sphere_closure_drop')
    args = a_.parse_args()
    stt = torch.load(args.part/'run'/'model.pt', map_location='cpu', weights_only=True)
    p = stt['points'].double().numpy(); r = torch.nn.functional.softplus(stt['radii'].double(), beta=100).numpy()
    g = dict(np.load(args.part/'dem_exact_single'/'geometry.npz'))
    m = build(g, 23008.38, .3008, dev, 'noslip', .25, .3, f64, kinematics='exact', nodalvol=True, nonlinear=False, ogden_alpha=2.)
    new = ClosureTerms(m, g, p, r, dict(np.load(args.part/'boundary_skin.npz')))
    n = new.n; h = args.h
    th = torch.zeros(n, 9, dtype=f64, device=dev)
    X = new.X; I3 = torch.eye(3, dtype=f64, device=dev)
    mass = m.mass.to(f64); M3 = mass[:, None]
    lam_, mu_ = float(m.k_v), new.mu
    up = torch.tensor([0., 0., 1.], dtype=f64, device=dev)
    fg = -9.8*M3*up
    ref = float(fg.norm())

    # contact candidates and levers (rest frame, ground normal +z)
    pt, rt = torch.as_tensor(p, dtype=f64, device=dev), torch.as_tensor(r, dtype=f64, device=dev)
    bot = pt-rt[:, None]*up
    pw = torch.cdist(bot, pt)**2-rt**2; pw[torch.arange(n, device=dev), torch.arange(n, device=dev)] = float('inf')
    exposed = (pw.min(1).values >= 0) & new.cap
    hY = (new.Y*up).sum(1)
    lowY = torch.full((n,), float('inf'), dtype=f64, device=dev).scatter_reduce(0, new.e_body, hY[new.e_pid], 'amin')
    argY = torch.zeros(n, 3, dtype=f64, device=dev)
    sel = hY[new.e_pid] == lowY[new.e_body]; argY[new.e_body[sel]] = new.Y[new.e_pid[sel]]
    low = torch.where(exposed[:, None], bot, argY)
    hl = torch.where(new.cap, (low*up).sum(1), torch.full((n,), float('inf'), dtype=f64, device=dev))
    cid = torch.nonzero(hl <= hl.min()+args.band).flatten()
    d = (low-X)[cid]
    Acap = torch.zeros(n, dtype=f64, device=dev).index_add(0, new.c_body, new.c_A)[cid]
    kn, kt = (lam_+2*mu_)*Acap/(3*rt[cid]), 4*mu_*Acap/(3*rt[cid])
    c0 = float(hl.min())-args.gap0
    print('part: %d bodies; contact candidates %d (lowest points within %.0f mm); cap stiffness k_n median %.3g, k_t %.3g (per unit density); '
          'ground %.4f m, %.1f mm below the lowest point; start speed %.2f m/s down; h 1/%.0f, friction mu %g' % (
              n, len(cid), 1e3*args.band, float(kn.median()), float(kt.median()), c0, 1e3*args.gap0, args.v0, 1/h, args.mu), flush=True)

    # preconditioner blocks at rest
    K, _ = closure_stiffness(new, th)
    a = torch.arange(n, device=dev)
    Kbb = K.view(n, 3, n, 3)[a, :, a, :].clone(); del K; torch.cuda.empty_cache()
    new.warm = True; new.F_warm = None

    P_t = I3-up[:, None]*up[None, :]
    st = dict(anchor=torch.zeros(len(cid), 3, dtype=f64, device=dev), glued=torch.zeros(len(cid), dtype=torch.bool, device=dev),
              p=torch.zeros(len(cid), dtype=f64, device=dev))

    def parts(z, need_grad):
        """E, contact, glue at z (with the graph when need_grad); also the lowest points and gaps"""
        z = z.detach().clone().requires_grad_(need_grad)
        F = new.F(z)
        E = sum(new.energy_F(th, z, F).values())
        xs = z[cid]+torch.einsum('cij,cj->ci', F[cid], d)
        gap = (xs*up).sum(1)-c0
        Ec = (.5*kn*torch.clamp(gap, max=0.)**2).sum()
        Eg = torch.zeros((), dtype=f64, device=dev)
        if args.mu > 0 and bool(st['glued'].any()):
            s = (xs-st['anchor'])@P_t; sn = torch.sqrt((s*s).sum(1).clamp(min=1e-60))     # (no NaN gradient at 0)
            sy = args.mu*st['p']/kt
            hub = torch.where(sn <= sy, .5*kt*sn**2, args.mu*st['p']*(sn-.5*sy))
            Eg = torch.where(st['glued'], hub, torch.zeros_like(hub)).sum()
        return z, E, Ec, Eg, xs, gap

    def Pi_grad(z, xt, xn, need_grad=True):
        z_, E, Ec, Eg, xs, gap = parts(z, need_grad)
        Pi = 2/h**2*(M3*(z_-xt)**2).sum()-(fg*(z_-xn)).sum()+E+Ec+Eg
        if not need_grad:
            return float(Pi), None, gap
        gr = torch.autograd.grad(Pi, z_)[0]
        return float(Pi), gr, gap

    def precond(gap):
        Pb = Kbb+(4*mass/h**2)[:, None, None]*I3
        pressed = gap < 0
        Pb[cid[pressed]] += kn[pressed][:, None, None]*up[:, None]*up[None, :]
        if args.mu > 0:
            Pb[cid[st['glued']]] += kt[st['glued']][:, None, None]*P_t
        Pi_ = torch.linalg.inv(Pb)
        return lambda v: torch.einsum('bij,bj->bi', Pi_, v)

    def step(xn, vn):
        xt = xn+h/2*vn
        z = xt+h*h/4*fg/M3; prods = 0                                         # free fall solves itself
        Pz, gz, gap = Pi_grad(z, xt, xn); prods += 1
        its = 0
        for its in range(1, 51):
            if float(gz.norm()) <= args.tol*ref:
                its -= 1; break
            Pinv = precond(gap.detach())
            # CG on H dz = -g (H by forward differences of the gradient), to 0.1 of |g| or negative curvature
            b = -gz; dz = torch.zeros_like(z); rr = b.clone(); y = Pinv(rr); pp = y.clone(); ry = float((rr*y).sum())
            for _ in range(200):
                eps = 1e-9/max(float(pp.abs().max()), 1e-300)
                Hp = (Pi_grad(z+eps*pp, xt, xn)[1]-gz)/eps; prods += 1
                pHp = float((pp*Hp).sum())
                if pHp <= 0:
                    if not bool(dz.abs().max() > 0):
                        dz = pp
                    break
                al = ry/pHp; dz = dz+al*pp; rr = rr-al*Hp
                if float(rr.norm()) <= .1*float(b.norm()):
                    break
                y = Pinv(rr); ry_new = float((rr*y).sum()); pp = y+ry_new/ry*pp; ry = ry_new
            slope = float((gz*dz).sum()); t_ = 1.
            while True:
                Pn = Pi_grad(z+t_*dz, xt, xn, need_grad=False)[0]; prods += 1
                if Pn <= Pz+1e-4*t_*slope or t_ < 1e-6:
                    break
                t_ *= .5
            z = z+t_*dz
            Pz, gz, gap = Pi_grad(z, xt, xn); prods += 1
        x1 = 2*z-xn; v1 = 4*(z-xn)/h-vn
        return x1, v1, prods, its, float(gz.norm())/ref

    def energy(x, v):
        _, E, Ec, Eg, xs, gap = parts(x, False)
        return float(.5*(M3*v*v).sum()+E+Ec+Eg-(fg*x).sum()), xs, gap

    x = X+(0.*up)[None]; v = (-args.v0*up).expand(n, 3).clone()
    H0, xs, gap = energy(x, v)
    KE0 = float(.5*(M3*v*v).sum())
    nsteps = int(round(args.T/h)); t0 = time.time(); tot = 0
    rec = []
    for k in range(1, nsteps+1):
        # glue: pressed at the step's start -> anchored (new anchors where the lowest point is now)
        if args.mu > 0:
            pn = kn*torch.clamp(-gap, min=0.)
            new_g = (pn > 0) & ~st['glued']
            st['anchor'][new_g] = xs.detach()[new_g]
            st['glued'] = pn > 0; st['p'] = pn
        x, v, prods, its, res = step(x, v)
        tot += prods
        H, xs, gap = energy(x, v)
        if args.mu > 0:                                                    # slide the anchors to the Coulomb bound
            s = (xs.detach()-st['anchor'])@P_t; sn = s.norm(dim=1); sy = args.mu*st['p']/kt
            slip = st['glued'] & (sn > sy)
            st['anchor'][slip] = st['anchor'][slip]+((1-sy/sn)[:, None]*s)[slip]
        pressed = int((gap < 0).sum())
        comp = float((torch.clamp(-gap, min=0.)/rt[cid]).max())
        com = (M3*x).sum(0)/mass.sum(); vcom = (M3*v).sum(0)/mass.sum()
        rec.append((k*h, float(com[2]), float(vcom[2]), pressed, comp, (H-H0)/KE0, prods, its))
        if k % 5 == 0 or k <= 3:
            print('step %3d (t %.4f, %.1f s): products %d, Newton %d, |grad| / weight %.1e; pressed %d, largest cap compression %.3f r; '
                  'com z %.5f, vz %.3f m/s; H - H0 = %.2e of the start kinetic energy' % (
                      k, k*h, time.time()-t0, prods, its, res, pressed, comp, float(com[2]), float(vcom[2]), (H-H0)/KE0), flush=True)
    R = np.array(rec)
    inc = R[:, 3] > 0
    print('total: %d products in %.0f s (%d steps; %.1f per step, %.1f per step in contact); largest cap compression %.3f r; '
          'H drift max |dH| %.2e of the start kinetic energy; com vz at the end %.3f m/s' % (
              tot, time.time()-t0, nsteps, R[:, 6].mean(), R[inc, 6].mean() if inc.any() else float('nan'), R[:, 4].max(),
              np.abs(R[:, 5]).max(), R[-1, 2]), flush=True)
    np.savez(args.part/('%s_mu%g.npz' % (args.tag, args.mu)), rec=R)


if __name__ == '__main__':
    main()
