"""The jack's drop with the skinless sphere model and Hertz-Mindlin ball contact (closure_warp.ClosureWarp.set_ball;
2026-10-26), set up as closure_drop.py / skin_drop_check.py (8k foam, at rest with its lowest ball point --height above
y = 0, gravity -y, Kelvin-Voigt beta, implicit midpoint dt 1/1200, tol 1e-3 of |m g|, 0.5 s).
  contact  every surface cell's Power Foam ball (centre x_B, radius R) against the plane: Hertz 4/3 E* R^1/2 delta^3/2
           along n through the centre, at the step's end centre 2 z - x_n; no lowest-point search, no F_B, no multipliers
  glue     (--mu) a ball touching at the step's start held at its centre, Mindlin k_t = 8 G* (R delta_n)^1/2, Coulomb bound
           mu F_n (step start), the force kept to the next step while it touches
  solver   Newton-CG on Pi(z) = 2 / dt^2 sum m |z - x_n - dt v_n / 2|^2 - f . (z - x_n) + E(z) + contact + 2 Et / dt;
           products by forward differences of the gradient, CG scalars kept on the GPU (a sync every --cg-check
           iterations; closure_warp.MidpointStep: the step's own work as kernels, the host reading a few scalars per Pi
           evaluation), block-Jacobi P = R K_bb R^T (1 + 2 beta / dt) + 4 m / dt^2 + the touching balls' 2 k_n n n^T
           (k_n = 2 E* (R delta)^1/2) and the glued ones' 2 k_t (I - n n^T); Armijo backtracking, the accepted trial's
           gradient reused. No energy record (user: speed first).
  scene    (10-27, the lion) --up the up axis (gravity along -up), --ground the plane's height along it with the body left
           where its foam puts it (default: the plane at 0 and the lowest ball point --height above it), --v0 the
           initial velocity
  PYTHONPATH=<main repo>:<worktree> python free_surface/hertz_drop.py [--mu 0.3]"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

from experiments.pacnerf_dem_forward import build
from free_surface.closure_drop import colouring, probe_blocks
from free_surface.closure_warp import ClosureCG, ClosureWarp, MidpointStep, TrustCG
from free_surface.sphere_closure import ClosureTerms

f64, dev = torch.float64, 'cuda'


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument('--out', type=Path, default=Path('output/round01_pruned_exact'))
    a_.add_argument('--height', type=float, default=.02)
    a_.add_argument('--dt', type=float, default=1/1200)
    a_.add_argument('--T', type=float, default=.5)
    a_.add_argument('--beta', type=float, default=.001)
    a_.add_argument('--mu', type=float, default=0.)
    a_.add_argument('--tol', type=float, default=1e-3)
    a_.add_argument('--cg-tol', type=float, default=.1, help='CG to this fraction of the Newton residual')
    a_.add_argument('--cg-abs', type=float, default=.5, help='> 0: CG to this fraction of the Newton tolerance (absolute), '
                    'so that one Newton step usually meets it; 0: to --cg-tol of the current residual')
    a_.add_argument('--cg-check', type=int, default=2)
    a_.add_argument('--graph-cg', action=argparse.BooleanOptionalAction, default=True, help='CG iterations as CUDA graphs (ClosureCG), --cg-check per graph')
    a_.add_argument('--gpu-step', action=argparse.BooleanOptionalAction, default=True, help="the step's own work as kernels (closure_warp.MidpointStep; "
                    'needs --graph-cg); --no-gpu-step: the torch loop')
    a_.add_argument('--product-iters', type=int, default=1, help="the closure's Newton iterations in the CG products (from the last point; "
                    "1: F exact to h^2, as the forward difference); 0: ClosureWarp.iters, each body to its 1e-10 step")
    a_.add_argument('--E', type=float, default=23008.38)
    a_.add_argument('--nu', type=float, default=.3008)
    a_.add_argument('--tag', default='hertz_drop')
    a_.add_argument('--up', type=int, default=1, help='the up axis (0 x, 1 y, 2 z)')
    a_.add_argument('--ground', type=float, default=None, help='the plane at this height along up, the body as its foam has it')
    a_.add_argument('--v0', type=float, nargs=3, default=[0., 0., 0.])
    args = a_.parse_args()
    cells = np.load(args.out/'dem_exact_single'/'cells.npz')
    g = dict(np.load(args.out/'dem_exact_single'/'geometry.npz'))
    bsk = dict(np.load(args.out/'boundary_skin.npz'))
    p, r = cells['points'], cells['radii']
    t0 = time.time()
    m = build(g, args.E, args.nu, dev, 'noslip', .25, .3, f64, kinematics='exact', nodalvol=True, nonlinear=False, ogden_alpha=2.)
    st = ClosureTerms(m, g, p, r, bsk)
    W = ClosureWarp(st)
    dt = args.dt
    if args.beta > 0:
        W.set_viscosity(args.beta*W.mu, args.beta*(W.lam+2*W.mu/3), dt)
        W.visc_weight = 2.; W.total_mode()
    boc = torch.as_tensor(np.asarray(g['body_of_cell']), device=dev).long()
    site = torch.as_tensor(bsk['patch_site'], device=dev).long()
    cbody = boc[site]; Rb = torch.as_tensor(r, dtype=f64, device=dev)[site]
    upt = tuple(1. if k == args.up else 0. for k in range(3)); c0 = 0. if args.ground is None else args.ground
    W.set_ball(cbody, Rb, upt, c0, args.mu)
    W.prepare()
    n = W.n; X = st.X
    mass = m.mass.to(f64); M3 = mass[:, None]; Mtot = float(mass.sum())
    up = torch.tensor(upt, dtype=f64, device=dev); I3 = torch.eye(3, dtype=f64, device=dev); Pt = I3-up[:, None]*up[None, :]
    nn_ = up[:, None]*up[None, :]
    fg = -9.8*M3*up; ref = float(fg.norm())
    low = float(((X[cbody]*up).sum(1)-Rb).min())
    x = X+(args.height-low)*up if args.ground is None else X.clone()
    v = torch.tensor(args.v0, dtype=f64, device=dev).expand_as(x).clone()
    print('the lowest ball point %.4f m above the plane' % (float(((x[cbody]*up).sum(1)-Rb).min())-c0), flush=True)
    color = colouring(st.i.cpu().numpy(), st.j.cpu().numpy(), n)
    Kbb = probe_blocks(W, x, color)
    if args.beta > 0:
        W.visc_ref(x)
    W.tball[0], W.tball[1] = 2., .5                                              # the step's end: 2 z - x_n, weight 1/2
    if args.gpu_step and args.graph_cg:                                          # (the trust-region step, 10-27)
        CG = TrustCG(W, mass, dt, per=args.cg_check, product_iters=args.product_iters or 1)
        ST = MidpointStep(W, CG, mass, Kbb, fg, dt, args.mu)
    else:                                                                       # (the torch loop's line search)
        CG = ClosureCG(W, mass, dt, per=args.cg_check, product_iters=args.product_iters or None) if args.graph_cg else None
        ST = None
    Es, Gs = W.Es, W.Gs
    print('jack: %d bodies, %d surface balls (radius median %.4f m); built in %.0f s; E* %.3f, G* %.3f (per unit density); beta %g, mu %g, '
          'dt 1/%.0f, tol %.0e' % (n, len(cbody), float(Rb.median()), time.time()-t0, Es, Gs, args.beta, args.mu, 1/dt, args.tol), flush=True)
    lamt = torch.zeros(len(cbody), 3, dtype=f64, device=dev)
    a = fg/M3
    nsteps = int(round(args.T/dt)); every = max(1, round(1/(240*dt)))
    rec, prods_all, its_all = [], [], []
    vmax = -1e9

    def record(s, x, v, fn, dend, prods, its):
        com = (M3*x).sum(0)/Mtot; vcom = (M3*v).sum(0)/Mtot
        on = dend > 0
        rec.append((s*dt, float(com@up), float(vcom@up), float(fn.sum())/(9.8*Mtot), int(on.sum()), float((dend.clamp(min=0.)/Rb).max()),
                    math.exp(float(W.tlj.min())), prods, its))
        print('t %.4f s (%.1f s): com up %.5f m, v up %+.4f m/s; contact force / weight %.3f on %d balls, overlap max %.3f R (%.1f mm); min J %.3f; '
              'products %d, Newton %d' % (s*dt, time.time()-t1, float(com@up), float(vcom@up), float(fn.sum())/(9.8*Mtot), int(on.sum()),
                                         rec[-1][5], 1e3*float(dend.max()), rec[-1][6], prods, its), flush=True)
    torch.cuda.synchronize(); t1 = time.time()
    if ST is not None:
        ST.set_state(x, v, a)
        cgt = (lambda g2: args.cg_abs*args.tol*ref) if args.cg_abs > 0 else (lambda g2: args.cg_tol*math.sqrt(g2))
        for s in range(1, nsteps+1):
            prods, its, g2 = ST.step(args.tol*ref, cgt)
            if not math.isfinite(g2) or math.sqrt(g2) > args.tol*ref:
                print('step %d: the solve did not converge (|grad| / weight %.2e after %d Newton iterations); stopping' % (s, math.sqrt(g2)/ref, its), flush=True)
                break
            ST.end()
            prods_all.append(prods); its_all.append(its)
            if s % every == 0 or s == nsteps:
                record(s, ST.tx, ST.tv, W.tbfn, W.tbdelta, prods, its)
        vmax = ST.vmax() if args.up == 1 else max([r_[2] for r_ in rec], default=-1e9); x = ST.tx.clone()
    for s in range(1, (nsteps if ST is None else 0)+1):                             # (--no-gpu-step: the torch loop)
        xc = x[cbody]
        dn = (Rb-((xc*up).sum(1)-c0)).clamp(min=0.)
        W.tbxn.copy_(xc)
        W.tbkt.copy_(8*Gs*torch.sqrt(Rb*dn))
        W.tbbound.copy_(args.mu*4/3*Es*torch.sqrt(Rb*dn)*dn)
        W.tblamt.copy_(lamt)
        xt = x+dt/2*v
        z = xt+dt*dt/4*a
        prods = [0]

        def Pi_grad(z_):
            prods[0] += 1
            l_, gE = W.grad(z_)
            last['gE'] = gE
            return 2/dt**2*(M3*(z_-xt)**2).sum()-(fg*(z_-x)).sum()+l_, 4/dt**2*M3*(z_-xt)-fg+gE
        last = {}
        Pz, gz = Pi_grad(z)
        its = 0
        while its < 50:
            gn = float(gz.norm())
            if gn <= args.tol*ref:
                break
            its += 1
            R = W.tR.clone()
            Pb = (R@Kbb@R.transpose(1, 2))*(1+2*args.beta/dt)+(4*mass/dt**2)[:, None, None]*I3
            d_ = W.tbdelta.clamp(min=0.)
            Pb.index_add_(0, cbody, (4*Es*torch.sqrt(Rb*d_))[:, None, None]*nn_)
            if args.mu > 0:
                Pb.index_add_(0, cbody, (2*W.tbkt*(W.tbbound > 0))[:, None, None]*Pt)
            Pi_ = torch.linalg.inv(Pb)
            Pinv = lambda q: torch.einsum('bij,bj->bi', Pi_, q)
            if CG is not None:
                dz, k_, _ = CG.solve(z, -gz, last['gE'], Pi_, args.cg_abs*args.tol*ref if args.cg_abs > 0 else args.cg_tol*gn)
                dz = dz.clone(); prods[0] += k_
            else:
                rr = -gz; dz = torch.zeros_like(z); y = Pinv(rr); pp = y.clone(); ry = (rr*y).sum()
                neg = torch.zeros((), dtype=torch.bool, device=dev)
                for k in range(300):
                    pmax = pp.abs().max()
                    eps = 1e-9/pmax
                    Hp = (Pi_grad(z+eps*pp)[1]-gz)/eps
                    pHp = (pp*Hp).sum()
                    bad = pHp <= 0
                    neg = neg | bad
                    al = torch.where(bad, torch.zeros_like(pHp), ry/pHp)
                    dz = dz+al*pp; rr = rr-al*Hp
                    y = Pinv(rr); ry_new = (rr*y).sum(); pp = y+(ry_new/ry)*pp; ry = ry_new
                    if (k+1) % args.cg_check == 0:
                        if bool(neg) or float(rr.norm()) <= args.cg_tol*gn:
                            break
            if not bool(dz.abs().max() > 0):
                dz = Pinv(-gz)
            slope = float((gz*dz).sum()); t_ = 1.; Pz_ = float(Pz)
            while True:
                Pn, gn_ = Pi_grad(z+t_*dz)
                if float(Pn) <= Pz_+1e-4*t_*slope or t_ < 1e-6:
                    break
                t_ *= .5
            z = z+t_*dz; Pz, gz = Pn, gn_
        if not math.isfinite(float(gz.norm())) or float(gz.norm()) > args.tol*ref:
            print('step %d: the solve did not converge (|grad| / weight %.2e after %d Newton iterations); stopping' % (s, float(gz.norm())/ref, its), flush=True)
            break
        fend = W.tbff.clone(); dend = W.tbdelta.clone(); fn = W.tbfn.clone()
        if args.beta > 0:
            ms = W.measures()
            for k_, r_ in W.tref.items():
                r_.copy_(2*ms[k_]-r_)
        x1 = 2*z-x; v1 = 4*(z-x)/dt-v
        a = (v1-v)/dt
        x, v = x1, v1
        lamt = fend*(dend > 0)[:, None]
        prods_all.append(prods[0]); its_all.append(its)
        com = (M3*x).sum(0)/Mtot; vcom = (M3*v).sum(0)/Mtot
        vmax = max(vmax, float(vcom@up))
        if s % every == 0 or s == nsteps:
            record(s, x, v, fn, dend, prods[0], its)
    torch.cuda.synchronize()
    P_ = np.array(prods_all); R_ = np.array(rec)
    if len(R_) == 0:
        return
    i = int(np.argmin(R_[:, 1]))
    print('total %.1f s for %d steps: products per step median %.0f, max %d (all %d); Newton per step median %.0f, max %d; com up min %.4f m at t %.3f; '
          'largest upward com velocity %.4f m/s; at the end com up %.4f, v up %+.4f; overlap max %.3f R; min J %.3f' % (
              time.time()-t1, len(P_), np.median(P_), P_.max(), P_.sum(), np.median(its_all), max(its_all), R_[i, 1], R_[i, 0], vmax,
              R_[-1, 1], R_[-1, 2], R_[:, 5].max(), R_[:, 6].min()), flush=True)
    np.savez(args.out/('%s_beta%g_mu%g.npz' % (args.tag, args.beta, args.mu)), rec=R_, prods=P_, its=np.array(its_all), x=x.cpu().numpy())


if __name__ == '__main__':
    main()
