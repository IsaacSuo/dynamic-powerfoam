"""The jack's drop with the skinless sphere model in Warp (closure_warp.ClosureWarp), the implicit midpoint rule
(2026-10-26), set up as skin_drop_check.py's runs for a direct comparison (skin_drop_beta0.001_dt4_tol3.log: 8k foam,
PAC-NeRF elastic scene 0's E and nu, at rest with its lowest exposed point --height above the ground y = 0, gravity -y,
Kelvin-Voigt beta, dt 1/1200, tol 1e-3 of |m g|, 0.5 s, frictionless by default).
  step     the midpoint z minimises Pi(z) = 2 / dt^2 sum m |z - x_n - dt v_n / 2|^2 - f . (z - x_n) + E(z) + Phi(z)
           + 2 Et(z) / dt (E, contact, glue and viscous terms from ClosureWarp), x_n+1 = 2 z - x_n, v_n+1 = 4 (z - x_n) / dt - v_n;
           the viscous references m_n+1 = 2 m(z) - m_n, dissipated 8 Et / dt (joint_step.MidpointIntegrator's rule)
  contact  the cap's series spring at the step's end gap 2 g(z) - g_n (k_contact cs 2, wt 1/2, goff g_n; no multipliers)
  glue     (--mu) k_friction with the step's start normal force k_n max(0, -g_n) as its bound's pressure, the force kept to
           the next step while the cell stays pressed, the material points held at integer times
  solver   Newton-CG on Pi, products by forward differences of the gradient (one CUDA-graph gradient each), block-Jacobi
           P = R K_bb R^T (1 + 2 beta / dt) + 4 m / dt^2 + the pressed cells' 2 k_n n n^T and the glued 2 k_t (I - n n^T)
           (K_bb probed at rest with a distance-4 colouring), Armijo backtracking, to |grad Pi| <= tol |m g|
Records every 1/240 s: H = K + elastic + contact + gravity (+ dissipated) against H_0, centre of mass, the lowest end gap
(-: the cap compressed), contact force / weight and pressed cells, products and Newton iterations, min J.
  PYTHONPATH=<main repo>:<worktree> python free_surface/closure_drop.py [--beta 0.001] [--mu 0] [--T 0.5]"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from experiments.pacnerf_dem_forward import build
from free_surface.closure_warp import ClosureWarp
from free_surface.skin_warp import contact_geometry
from free_surface.sphere_closure import ClosureTerms

f64, dev = torch.float64, 'cuda'


def colouring(i_, j_, n, dist=4):
    """greedy colouring with no two bodies of a colour within dist bonds"""
    Adj = sp.coo_matrix((np.ones(len(i_)), (i_, j_)), shape=(n, n)).tocsr(); Adj = ((Adj+Adj.T+sp.eye(n)) > 0).astype(np.float64)
    D = Adj.copy()
    for _ in range(dist-1):
        D = ((D@Adj) > 0).astype(np.float64)
    D = D.tocsr()
    color = -np.ones(n, int)
    for b in np.argsort(-np.diff(D.indptr)):
        used = set(color[D.indices[D.indptr[b]:D.indptr[b+1]]].tolist())
        c = 0
        while c in used:
            c += 1
        color[b] = c
    return color


def probe_blocks(W, X, color, eps=1e-7):
    """the bodies' 3 x 3 diagonal blocks of the elastic Hessian at X by central differences of the gradient; the closure
    converged at X first (each evaluation starts from the last root: 10-26, from a root 5 % away its 3 iterations left
    steps ~17 and indefinite blocks)"""
    n = len(X); B = torch.zeros(n, 3, 3, dtype=f64, device=dev)
    W.elastic_mode()
    for _ in range(20):
        W.grad(X)
        if float(W.tcstep.max()) < 1e-10:
            break
    else:
        raise RuntimeError('probe_blocks: the closure does not converge at X')
    for c in range(color.max()+1):
        msk = torch.as_tensor(color == c, device=dev)
        for k in range(3):
            v = torch.zeros(n, 3, dtype=f64, device=dev); v[msk, k] = eps
            d = (W.grad(X+v)[1]-W.grad(X-v)[1])/(2*eps)
            B[msk, :, k] = d[msk]
    W.total_mode()
    return .5*(B+B.transpose(1, 2))


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument('--out', type=Path, default=Path('output/round01_pruned_exact'))
    a_.add_argument('--height', type=float, default=.02)
    a_.add_argument('--dt', type=float, default=1/1200)
    a_.add_argument('--T', type=float, default=.5)
    a_.add_argument('--beta', type=float, default=.001)
    a_.add_argument('--mu', type=float, default=0.)
    a_.add_argument('--tol', type=float, default=1e-3)
    a_.add_argument('--E', type=float, default=23008.38)
    a_.add_argument('--nu', type=float, default=.3008)
    a_.add_argument('--contact', choices=('series', 'rigid', 'ball'), default='ball',
                    help='ball: each surface cell\'s sphere, stretched by the cell along n (its |F^T n| at the step\'s start), a bound on the cell\'s '
                         'centre alone, held exactly (the normal force through the centre, as a ball on a plane); series: the cap\'s spring k_n; '
                         'rigid (a diagnostic): a penalty kappa / (n^T K_bb^-1 n) on the affine cap\'s lowest point')
    a_.add_argument('--kappa', type=float, default=100.)
    a_.add_argument('--kscale', type=float, default=1., help='series: the cap spring times this (a diagnostic)')
    a_.add_argument('--verbose-from', type=int, default=0, help='ball: every step from this one printed with its solve')
    a_.add_argument('--tag', default='closure_drop')
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
    pt, rt = torch.as_tensor(p, dtype=f64, device=dev), torch.as_tensor(r, dtype=f64, device=dev)
    cg = contact_geometry(pt, rt, g['body_of_cell'])
    assert (cg['site'].cpu().numpy() == bsk['patch_site']).all()
    W.set_contact(cg, (0., 1., 0.), 0., torch.as_tensor(bsk['patch_A'], dtype=f64, device=dev))
    W.prepare()
    n = W.n; X = st.X
    mass = m.mass.to(f64); M3 = mass[:, None]; Mtot = float(mass.sum())
    up = torch.tensor([0., 1., 0.], dtype=f64, device=dev)
    fg = -9.8*M3*up; ref = float(fg.norm())
    color = colouring(st.i.cpu().numpy(), st.j.cpu().numpy(), n)
    W.tcpar[0], W.tcpar[1], W.tcpar[2] = 1., 1., 0.; W.tgoff.zero_()
    W.grad(X)
    shift = (args.height-float(W.tgap.min()))*up
    x = X+shift; v = torch.zeros_like(x)
    Kbb = probe_blocks(W, x, color)
    cbody = torch.as_tensor(cg['body'], device=dev).long()
    if args.contact == 'rigid':
        W.trho.copy_(args.kappa/torch.einsum('i,cij,j->c', up, torch.linalg.inv(Kbb[cbody]), up))
    else:
        W.trho.mul_(args.kscale); W.trhot.mul_(args.kscale)                     # (the glue's shear compliance likewise)
    print('jack: %d bodies, %d contact cells; built in %.0f s (%d colours for the blocks); beta %g, mu %g, dt 1/%.0f, tol %.0e of |m g|; '
          'contact %s, its stiffness median %.3g' % (n, W.ct['k'], time.time()-t0, color.max()+1, args.beta, args.mu, 1/dt, args.tol, args.contact, float(W.trho.median())), flush=True)
    I3 = torch.eye(3, dtype=f64, device=dev); Pt = I3-up[:, None]*up[None, :]
    W.tfpar[0] = args.mu
    # each body's walls (CSR over both ends) for the gap gradients
    nbw = len(st.i)
    ends = torch.cat((st.i, st.j)); wid = torch.arange(nbw, device=dev).repeat(2)
    order = torch.argsort(ends, stable=True); wid_s = wid[order]
    woff = torch.zeros(n+1, dtype=torch.long, device=dev); woff[1:] = torch.cumsum(torch.bincount(ends, minlength=n), 0)
    import warp as wp
    Jtor = wp.to_torch(W.B['Jr'], requires_grad=False).view(n, 9, 9)              # (k_closure keeps the Jacobian there)

    def gap_vectors(cs):
        """d g_c / dx for contact cells cs at the last grad call: n e_B + d(n . F_B w)/dx, F_B's change through the
        closure's Jacobian, dF = J^-1 dS_B, dS_B = sum over B's walls A / 2 (dx_j - dx_i) (x) nrm; (m, n, 3)"""
        B = cbody[cs]; m_ = len(cs)
        w = W.tXs[cs]
        rhs = (up[None, :, None]*w[:, None, :]).reshape(m_, 9)                    # vec(n (x) w), row-major
        T = torch.linalg.solve(Jtor[B].transpose(1, 2), rhs).view(m_, 3, 3)
        cnt = woff[B+1]-woff[B]
        ci = torch.repeat_interleave(torch.arange(m_, device=dev), cnt)
        start = torch.repeat_interleave(woff[B], cnt)
        first = torch.repeat_interleave(torch.cumsum(cnt, 0)-cnt, cnt)
        e = wid_s[start+torch.arange(int(cnt.sum()), device=dev)-first]
        u = (st.A[e]/2)[:, None]*torch.einsum('kij,kj->ki', T[ci], st.nrm[e])
        C = torch.zeros(m_, n, 3, dtype=f64, device=dev)
        C.index_put_((ci, st.j[e]), u, accumulate=True)
        C.index_put_((ci, st.i[e]), -u, accumulate=True)
        C[torch.arange(m_, device=dev), B] += up
        return C

    def state_energy(x_):
        """elastic and contact energy at x_ with the true gap (and the lowest points there for the glue)"""
        tw = W.ttw.clone(); cp = W.tcpar.clone(); go = W.tgoff.clone(); fp = float(W.tfpar[0])
        W.ttw.zero_(); W.ttw[:6 if args.contact == 'ball' else 7] = 1.; W.tcpar[0], W.tcpar[1], W.tcpar[2] = 1., 1., 0.; W.tgoff.zero_(); W.tfpar[0] = 0.
        W.grad(x_)
        out = float(W.tsum[:6].sum()), float(W.tsum[6]), W.tgap.clone(), W.tXs.clone(), W.tcpos.clone(), math.exp(float(W.tlj.min()))
        W.ttw.copy_(tw); W.tcpar.copy_(cp); W.tgoff.copy_(go); W.tfpar[0] = fp
        return out
    if args.beta > 0:
        W.visc_ref(x)
    E0, C0, gap_n, wfix, pn, _ = state_energy(x)
    H0 = E0+C0-float((fg*x).sum())
    mgh = 9.8*Mtot*args.height
    W.tgoff.copy_(gap_n); W.tcpar[0], W.tcpar[1], W.tcpar[2] = 2., .5, 0.
    lamt = torch.zeros(W.ct['k'], 3, dtype=f64, device=dev)
    a = fg/M3
    dissipated = 0.
    nsteps = int(round(args.T/dt)); every = max(1, round(1/(240*dt)))
    rec, prods_all, its_all = [], [], []
    vmax = -1e9
    t1 = time.time()
    rb = torch.as_tensor(cg['r'], dtype=f64, device=dev)
    ball_act = torch.zeros(len(rb), dtype=torch.bool, device=dev)
    if args.contact == 'ball':
        W.ttw[6] = 0.                                                             # (k_contact's term off; its gaps still read)

    def solve_ball(x, v, a):
        """the midpoint z with the bounds n . x_n+1,B >= h_B = r_B |F_B^T n| (step start) for the surface cells, i.e.
        n . z_B >= (h_B + n . x_n,B) / 2: an active set (moved between solves by the multipliers' signs and the bounds),
        within a solve the active cells' normal coordinate fixed (gradients, directions and the preconditioner projected);
        the multiplier lam_B = n . dPi/dz_B is the contact force"""
        nonlocal ball_act
        xt = x+dt/2*v
        Fn = W.tF[cbody]                                                          # (the last grad call: x_n)
        hB = rb*torch.linalg.norm(Fn.transpose(1, 2)@up, dim=1)
        bnd = .5*(hB+(x[cbody]*up).sum(1))
        z = xt+dt*dt/4*a
        cnt = [0]

        def Pi_grad(z_):
            cnt[0] += 1
            l_, gE = W.grad(z_)
            return float(2/dt**2*(M3*(z_-xt)**2).sum()-(fg*(z_-x)).sum())+float(l_), 4/dt**2*M3*(z_-xt)-fg+gE
        A = ball_act.clone(); its = 0; passes = 0
        while True:
            passes += 1
            Ab = cbody[A]

            def proj(q, Ab=Ab):
                q = q.clone(); q[Ab] -= up*(q[Ab]@up)[:, None]
                return q
            z = z.clone(); z[Ab] += up*(bnd[A]-z[Ab]@up)[:, None]
            Pz, gz = Pi_grad(z); gp = proj(gz)
            while float(gp.norm()) > args.tol*ref and its < 50*passes:
                its += 1
                R = W.tR.clone()
                Pb = (R@Kbb@R.transpose(1, 2))*(1+2*args.beta/dt)+(4*mass/dt**2)[:, None, None]*I3
                Pi_ = torch.linalg.inv(Pb)
                Pinv = lambda q: proj(torch.einsum('bij,bj->bi', Pi_, proj(q)))
                b = -gp; dz = torch.zeros_like(z); rr = b.clone(); y = Pinv(rr); pp = y.clone(); ry = float((rr*y).sum())
                for _ in range(300):
                    eps = 1e-9/max(float(pp.abs().max()), 1e-300)
                    Hp = proj((Pi_grad(z+eps*pp)[1]-gz)/eps)
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
                    Pn = Pi_grad(z+t_*dz)[0]
                    if Pn <= Pz+1e-4*t_*slope or t_ < 1e-6:
                        break
                    t_ *= .5
                z = z+t_*dz
                Pz, gz = Pi_grad(z); gp = proj(gz)
            lam = (gz[cbody]*up).sum(1)
            nz = (z[cbody]*up).sum(1)
            A_new = (A & (lam > 0)) | (~A & (nz < bnd-1e-13))
            if bool(torch.equal(A_new, A)) or passes >= 20:
                break
            A = A_new
        ball_act = A
        return z, cnt[0], its, passes, torch.where(A, lam, torch.zeros_like(lam)), gp, hB
    for s in range(1, nsteps+1):
        if args.contact == 'ball':
            z, prods, its, passes, lam_ball, gz, hB = solve_ball(x, v, a)
            if args.verbose_from and s >= args.verbose_from:
                act = lam_ball > 0
                Fa = W.tF[cbody[act]] if bool(act.any()) else W.tF[:1]
                print('  step %d: products %d, Newton %d, passes %d, |grad| / weight %.1e; active %d, force / weight %.3f, max %.3f; '
                      'the active cells\' J min %.3f, |F - I| max %.3f; body J min %.3f; closure step max %.1e' % (
                          s, prods, its, passes, float(gz.norm())/ref, int(act.sum()), float(lam_ball.sum())/(9.8*Mtot), float(lam_ball.max())/(9.8*Mtot),
                          float(torch.linalg.det(Fa).min()), float((Fa-I3).abs().max()), float(torch.linalg.det(W.tF).min()), float(W.tcstep.max())), flush=True)
            if not math.isfinite(float(gz.norm())) or float(gz.norm()) > args.tol*ref:
                print('step %d: the solve did not converge (|grad| / weight %.2e after %d Newton iterations, %d passes); stopping' % (
                    s, float(gz.norm())/ref, its, passes), flush=True)
                break
        else:
            xt = x+dt/2*v
            z = xt+dt*dt/4*a
            W.twfix.copy_(wfix); W.tpn.copy_(pn); W.tlamt.copy_(lamt)
            W.tlamR.copy_(W.trho*torch.clamp(-W.tgoff, min=0.))                        # the step's start normal force
            prods = 0

            def Pi_grad(z_):
                nonlocal prods
                prods += 1
                l_, gE = W.grad(z_)
                return float(2/dt**2*(M3*(z_-xt)**2).sum()-(fg*(z_-x)).sum())+float(l_), 4/dt**2*M3*(z_-xt)-fg+gE
            held = args.contact == 'rigid'                                           # rigid: the contact set held within each
            if held:                                                                  # solve (a smooth quadratic on it, nothing off
                W.tcpar[2] = 1.                                                       # it; the penalty's kink outside the Newton
                W.tact.copy_((W.tgoff < 0).to(W.tact.dtype))                          # model, PLAN 10-23), moved between solves
            passes = 0; its = 0                                                       # to the end gaps' signs until it settles
            while True:
              passes += 1
              Pz, gz = Pi_grad(z)
              while float(gz.norm()) > args.tol*ref and its < 50*passes:
                  its += 1
                  R = W.tR.clone()
                  Pb = (R@Kbb@R.transpose(1, 2))*(1+2*args.beta/dt)+(4*mass/dt**2)[:, None, None]*I3
                  pressed = (W.tact > 0) if held else (W.tgap < 0)
                  if args.contact == 'series':
                      Pb.index_add_(0, cbody[pressed], (2*W.trho[pressed])[:, None, None]*(up[:, None]*up[None, :]).expand(int(pressed.sum()), 3, 3))
                  if args.mu > 0:
                      gl = W.tlamR > 0
                      Pb.index_add_(0, cbody[gl], (2*W.trhot[gl])[:, None, None]*Pt.expand(int(gl.sum()), 3, 3))
                  Pi_ = torch.linalg.inv(Pb)
                  Pblk = lambda q: torch.einsum('bij,bj->bi', Pi_, q)
                  if args.contact == 'rigid' and bool(pressed.any()):
                      # the pressed cells' gap gradients (the cell and its wall neighbours through F_B) as a low-rank
                      # correction, P = P_b + C diag(2 rho) C^T, inverted by Woodbury
                      C = gap_vectors(torch.nonzero(pressed).flatten())                 # (m, n, 3)
                      U = torch.einsum('bij,mbj->mbi', Pi_, C)
                      Cap = torch.einsum('mbi,kbi->mk', C, U)+torch.diag(1/(2*W.trho[pressed]))
                      Lc = torch.linalg.cholesky(Cap)
                      def Pinv(q, C=C, U=U, Lc=Lc):
                          y_ = Pblk(q)
                          s_ = torch.cholesky_solve(torch.einsum('mbi,bi->m', C, y_)[:, None], Lc)[:, 0]
                          return y_-torch.einsum('mbi,m->bi', U, s_)
                  else:
                      Pinv = Pblk
                  b = -gz; dz = torch.zeros_like(z); rr = b.clone(); y = Pinv(rr); pp = y.clone(); ry = float((rr*y).sum())
                  for _ in range(300):
                      eps = 1e-9/max(float(pp.abs().max()), 1e-300)
                      Hp = (Pi_grad(z+eps*pp)[1]-gz)/eps
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
                      Pn = Pi_grad(z+t_*dz)[0]
                      if Pn <= Pz+1e-4*t_*slope or t_ < 1e-6:
                          break
                      t_ *= .5
                  z = z+t_*dz
                  Pz, gz = Pi_grad(z)
              if not held:
                  break
              act_new = (W.tgap < 0)
              if bool(torch.equal(act_new, W.tact > 0)) or passes >= 20:
                  break
              W.tact.copy_(act_new.to(W.tact.dtype))
            if not math.isfinite(float(gz.norm())) or float(gz.norm()) > args.tol*ref:
                print('step %d: the solve did not converge (|grad| / weight %.2e after %d Newton iterations, %d passes); stopping' % (
                    s, float(gz.norm())/ref, its, passes), flush=True)
                break
        # the solve's last gradient was at z: end gaps, glue forces, viscous terms and measures there
        gap_end = W.tgap.clone(); ff = W.tff.clone(); Et = float(W.tsum[7])
        ms = W.measures() if args.beta > 0 else None
        x1 = 2*z-x; v1 = 4*(z-x)/dt-v
        a = (v1-v)/dt
        x, v = x1, v1
        if args.beta > 0:
            dissipated += 8*Et/dt
            for k_, r_ in W.tref.items():
                r_.copy_(2*ms[k_]-r_)
        lamt = ff*(gap_end < 0)[:, None]
        W.tgoff.copy_(gap_end)
        E_el, E_c, gap_true, wfix, pn, minJ = state_energy(x)
        prods_all.append(prods); its_all.append(its)
        com = (M3*x).sum(0)/Mtot; vcom = (M3*v).sum(0)/Mtot
        vmax = max(vmax, float(vcom[1]))
        if s % every == 0 or s == nsteps:
            K = .5*float((M3*v*v).sum()); H = K+E_el+E_c-float((fg*x).sum())
            force = float((W.trho*torch.clamp(-gap_true, min=0.)).sum())
            pr = int((gap_true < 0).sum())
            comp = float((torch.clamp(-gap_true, min=0.)/torch.as_tensor(cg['r'], device=dev)).max())
            if args.contact == 'ball':                                            # the multipliers; the cells' compression r (|F^T n| - 1)
                force = float(lam_ball.sum()); pr = int((lam_ball > 0).sum())
                comp = float(torch.clamp(rb-rb*torch.linalg.norm(W.tF[cbody].transpose(1, 2)@up, dim=1), min=0.).div(rb)[lam_ball > 0].max()) if pr else 0.
            rec.append((s*dt, float(com[1]), float(vcom[1]), (H+dissipated-H0)/mgh, dissipated/mgh, force/(9.8*Mtot), pr, comp, minJ, prods, its))
            print('t %.4f s (%.0f s): H + dissipated - H_0 %+.2e of m g h, dissipated %.4f m g h; com y %.5f m, vy %+.4f m/s; contact force / weight %.3f on %d cells, '
                  'largest cap compression %.3f r; min J %.3f; products %d, Newton %d' % (
                      s*dt, time.time()-t1, (H+dissipated-H0)/mgh, dissipated/mgh, float(com[1]), float(vcom[1]), force/(9.8*Mtot), pr, comp, minJ, prods, its), flush=True)
    P_ = np.array(prods_all); R_ = np.array(rec)
    print('total %.0f s for %d steps: products per step median %.0f, max %d (all %d); Newton per step median %.0f, max %d; largest upward com velocity %.4f m/s; '
          'dissipated %.4f of m g h; |H + dissipated - H_0| max %.2e of m g h; largest cap compression %.3f r; min J %.3f' % (
              time.time()-t1, nsteps, np.median(P_), P_.max(), P_.sum(), np.median(its_all), max(its_all), vmax, dissipated/mgh,
              np.abs(R_[:, 3]).max(), R_[:, 7].max(), R_[:, 8].min()), flush=True)
    np.savez(args.out/('%s_%s%s_beta%g_mu%g.npz' % (args.tag, args.contact, '' if args.kscale == 1 else '_k%g' % args.kscale, args.beta, args.mu)), rec=R_, prods=P_, its=np.array(its_all), x=x.cpu().numpy())


if __name__ == '__main__':
    main()
