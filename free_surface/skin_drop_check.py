"""Drop on a frictionless ground (2026-10-23, the per-step skin solve with contact, skin_warp.k_contact): the object at
rest, its lowest exposed sphere point --height above the ground y = 0 (PAC-NeRF world, as mpm_drop_energy.py), gravity
along -y, symplectic Euler for the bodies (lumped masses = volumes: energies per unit density), the skin solved every
step under the contact conditions (augmented Lagrangian, SkinWarp.set_contact / solve_contact). Viscosity (--beta, or
--eta / --zeta; 10-24): on the elasticity's own measures, the step's start the reference (SkinWarp.visc_ref), the bodies
implicit there (visc_body_update), the skin's energy with the step's dissipation potential. The skin's residual
tolerance is relative to the bodies' weights |(m_B g)| (at rest the elastic forces vanish). Every --every steps: total
energy H = kinetic + elastic + gravity + dissipated (+ the augmented term, ~0 once converged), centre of mass height and
velocity, lowest gap, contact force sum / weight and count, products and passes. Writes skin_drop_check[_beta..].npz.
Step size (10-24, the convergence check with beta 0.001 / 0.01 at dt = 1/4800 s, 2, 4, 8 dt: positions second order,
the bodies' error at 1/1200 s ~34 um max, 13 um rms, the dissipation ~0.1 %, contact forces' instants first order): the
default --dt is 1/1200 s for the implicit integrators (midpoint, bdf2), 1/4800 s for split (its bodies explicit); a new
material or scene: run h and 2h, the error of h ~ their difference / 3. --every defaults to a record every 1/240 s.
  python free_surface/skin_drop_check.py --out output/round01_pruned_exact --steps 600 --tols 1e-4 --shift"""
import math
import time

import numpy as np
import torch
import warp as wp

from free_surface.skin_contact_check import f64, dev
from free_surface.skin_dynamics_check import build_model, parser, predict_skin, visc_body_update


def main():
    a = parser()
    a.add_argument('--height', type=float, default=.02)
    a.add_argument('--every', type=int, default=None, help='steps between records (default: every 1/240 s)')
    a.add_argument('--gravity', type=float, default=9.8)
    a.add_argument('--al-max', type=int, default=50)
    a.add_argument('--stop-on-anomaly', action='store_true')
    a.add_argument('--shift', action='store_true', help='the cells\' affine patches carried by their skin\'s mean deviation')
    a.add_argument('--points', action='store_true', help='contact at held points of the skin\'s own surface (else the cells\' affine patches)')
    a.add_argument('--tag', default='', help='suffix of the output file')
    a.add_argument('--record-F', action='store_true', help='also every body\'s F_B (the last gradient pass: the integer-time state) at the records, for rendering')
    a.add_argument('--energy-audit', action='store_true', help='the midpoint rule: the energy record\'s drift split into its sources, per step (5 more gradients)')
    a.add_argument('--save-state', type=int, default=0, help='the midpoint integrator\'s state at the start of this step into skin_drop_state_step<N>.npz')
    a.add_argument('--predict', choices=('linear', 'affine'), default='affine',
                   help='the skin\'s start: linear extrapolation, or the cells\' affine field at the new bodies plus the extrapolated deviation (predict_skin)')
    a.set_defaults(integrator='midpoint', dt=None)                                 # (10-24: bodies and skin in one step; bdf2 kept as an option)
    args = a.parse_args()
    if args.dt is None:                                                            # (the step-size check, docstring)
        args.dt = 1/4800 if args.integrator == 'split' else 1/1200
    if args.every is None:
        args.every = max(1, round(1/(240*args.dt)))
    b = build_model(args, contact=dict(normal=(0., 1., 0.), c0=0., points=args.points, shift=args.shift))
    st, W, S, prec, mass = b['st'], b['W'], b['S'], b['prec'], b['mass']
    X, Y = st.X, st.Y
    up = torch.tensor([0., 1., 0.], dtype=f64, device=dev)
    W.grad(X, Y)
    shift = (args.height-float((W.tgap_cell if args.points else W.tgap).min()))*up
    x, y = X+shift, Y+shift
    fg = -args.gravity*mass[:, None]*up
    ref = float(fg.norm())
    tol, dt = args.tols[0], args.dt
    M = float(mass.sum())
    e, gx, _ = W.grad(x, y)
    tsum = W.tsum.clone()
    v = torch.zeros_like(x)
    fx = -gx+fg
    rec = []
    yp = yc = y; xp = x.clone()                                                   # the skin and the bodies one step back
    prods, passes = [], []
    lam = torch.zeros(len(W.tlam), dtype=f64, device=dev); gap = W.tgap.clone()
    held_info = (0, 0, float('nan'))
    Hlast = None
    Xs_last, X_rec, lam_rec, F_rec = None, [], [], []
    eye = torch.eye(3, dtype=f64, device=dev)
    def motion(x, v, y, lam, Xs):
        """centre of mass (x, z) and its velocity, the angular velocity about it (I^-1 L), the centre of pressure (x, z):
        the cells' contact points (affine patch's lowest point, mean skin shift's tangential part left out) weighted by
        their contact forces"""
        com3 = (mass[:, None]*x).sum(0)/M; vc = (mass[:, None]*v).sum(0)/M
        r_, u = x-com3, v-vc
        L = (mass[:, None]*torch.linalg.cross(r_, u)).sum(0)
        I = (mass[:, None, None]*((r_*r_).sum(1)[:, None, None]*eye-r_[:, :, None]*r_[:, None, :])).sum(0)
        om = torch.linalg.solve(I, L)
        cop = [float('nan')]*2
        if not args.points and Xs is not None and float(lam.sum()) > 0:
            cb = W.cg['body']
            pts = x[cb]+torch.einsum('cij,cj->ci', st.F(x, y)[cb], Xs+wp.to_torch(W.ct['dbar'], requires_grad=False))
            w = lam/lam.sum(); cp = (w[:, None]*pts).sum(0); cop = [float(cp[0]), float(cp[2])]
        return [float(com3[0]), float(com3[2]), float(vc[0]), float(vc[2]), float(om[0]), float(om[1]), float(om[2])]+cop
    gmin = (lambda g_: g_[W.tc['cand'] > 0].min() if bool((W.tc['cand'] > 0).any()) else torch.tensor(float('inf'))) if args.points else (lambda g_: g_.min())
    prev = cur = None
    Dv, cgs = 0., []                                                               # energy the viscosity took, its CG per step
    mid = args.integrator != 'split'
    pinfo, nrej = [], 0
    if mid:
        from free_surface.joint_step import BDF2Integrator, MidpointIntegrator
        I = MidpointIntegrator(b, dt, x, v, y, fg, tol=tol, ref=ref, contact=True, al_max=args.al_max) if args.integrator == 'midpoint' else BDF2Integrator(b, dt, x, v, y, fg, tol=tol, ref=ref, al_max=args.al_max)
        Emid = Ee_int = float(tsum[:6].sum())
    if mid and args.energy_audit:                                                  # the midpoint rule's exact balance per step:
        assert args.integrator == 'midpoint'                                       # dK + dU = r . dz - dz . grad(E + Phi + 2 Et / dt)(mid),
        wts = lambda *w: torch.tensor(w, dtype=f64, device=dev)                    # dz = (2 (x_m - x_n), 2 (y_m - y_n)) (y_n the skin
        wE, wC, wV = wts(1, 1, 1, 1, 1, 1, 0, 0), wts(0, 0, 0, 0, 0, 0, 1, 0), wts(0, 0, 0, 0, 0, 0, 0, 2/dt)   # extrapolated to
        def masked(w, xx, yy):                                                     # integer times); so the record's drift is
            W.ttw.copy_(w); l_, gx_, gy_ = W.grad(xx, yy)                          # (elastic recorded - true change) + (true
            return float(l_), gx_, gy_                                             # change - dz . grad E) + (dissipated - dz .
        y_int = y.clone(); E_n = masked(wE, x, y)[0]; W.total_mode()               # grad 2 Et / dt) + (Phi - contact work - dz .
        aud = dict(E_avg=0., E_quad=0., visc=0., contact=0., resid=0.)             # grad Phi) + r . dz
    t0 = time.time()
    print('start: lowest point %.4f m above the ground, centre of mass y %.4f m; weight %.4e (per unit density); tol %.0e of |m g| = %.3e; '
          'dt %.3e s, %d steps; free fall to the ground %.4f s' % (float((W.tgap_cell if args.points else W.tgap).min()), float((mass*x[:, 1]).sum())/M, M*args.gravity, tol, ref,
                                                                  dt, args.steps, (2*args.height/args.gravity)**.5), flush=True)
    for s in range(0, args.steps+1):
        if s % args.every == 0 or s == args.steps:
            K = .5*float((mass[:, None]*v*v).sum()); Ee = Ee_int if mid else float(tsum[:6].sum()); Phi = float(tsum[6])
            Ug = args.gravity*float((mass*x[:, 1]).sum())
            act = lam > 0
            rec.append([s*dt, K, Ee, Ug, Phi, K+Ee+Ug+Phi+Dv-(I.contact_work if mid else 0.), float((mass*x[:, 1]).sum())/M, float((mass*v[:, 1]).sum())/M, float(gmin(gap)),
                        float(lam.sum())/(M*args.gravity), int(act.sum()), sum(prods[-args.every:]), max(passes[-args.every:], default=0)]
                       +motion(x, v, yc, lam, Xs_last)+[Dv])
            X_rec.append(x.float().cpu().numpy()); lam_rec.append(lam.float().cpu().numpy())
            if args.record_F:
                F_rec.append(W.tF.float().cpu().numpy())
            r_ = rec[-1]
            print('t %.4f s: H %.6e (K %.3e, elastic %.3e, gravity %.6e, AL term %.1e, dissipated %.3e); com y %.5f m, vy %+.4f m/s; lowest gap %+.2e m; '
                  'contact force / weight %.3f on %d cells; products last %d steps %d, passes max %d, viscous CG max %d; min J %.3f; '
                  'rho of the active / median max %.1e; %.0f s' % (
                      r_[0], r_[5], r_[1], r_[2], r_[3], r_[4], Dv, r_[6], r_[7], r_[8], r_[9], r_[10], args.every, r_[11], r_[12],
                      max(cgs[-args.every:], default=0), math.exp(float(W.tlj.min())),
                      float(W.trho[act].max()/W.trho.median()) if bool(act.any()) else 0., time.time()-t0), flush=True)
        if s == args.steps:
            break
        if mid:                                                                  # bodies and skin in one implicit midpoint step
            if args.save_state == s+1 and args.integrator == 'midpoint':           # the state at the start of step s + 1
                np.savez(b['out']/('skin_drop_state_step%d.npz' % (s+1)), **{k_: t_.cpu().numpy() for k_, t_ in dict(
                    x=I.x, v=I.v, a=I.a, xm1=I.xm1, ym1=I.ym1, xm2=I.xm2, ym2=I.ym2, lam=W.tlam, act=W.tact, goff=W.tgoff, rho=W.trho,
                    **{'ref_'+k2: r2 for k2, r2 in W.tref.items()}).items()})
                print('   state at the start of step %d saved' % (s+1), flush=True)
            if args.energy_audit:
                x_n, v_n, D0, Wc0, Phi0, Eint0 = I.x.clone(), I.v.clone(), I.dissipated, I.contact_work, float(tsum[6]), Ee_int
            st_ = I.step()
            for k_, p_ in enumerate(st_['passes']):                              # (step, pass, products, outer, rejected, negative,
                pinfo.append((s+1, k_+1)+tuple(p_))                              # start residual on bodies, on skin / ref)
            for it_, lines in I.J.rejected_logs:                                 # passes with many rejected steps (the first 6)
                nrej += 1
                if nrej <= 6:
                    print('   step %d, pass %d: %s' % (s+1, it_, ' | '.join(lines)), flush=True)
            I.J.rejected_logs.clear()
            if (s+1) % args.every == 0:
                q = np.array([p_ for p_ in pinfo if p_[0] > s+1-args.every])
                f1, fl = q[q[:, 1] == 1], q[q[:, 1] > 1]
                print('   steps %d-%d: passes %d; products per pass %.1f (first %.1f, later %s); outer per pass %.1f; rejected %d, negative %d; '
                      'start residual / ref first pass: bodies median %.1e, skin %.1e; later passes: bodies %s, skin %s' % (
                          s+2-args.every, s+1, len(q), q[:, 2].mean(), f1[:, 2].mean(), '%.1f' % fl[:, 2].mean() if len(fl) else '-', q[:, 3].mean(),
                          int(q[:, 4].sum()), int(q[:, 5].sum()), np.median(f1[:, 6]), np.median(f1[:, 7]),
                          '%.1e' % np.median(fl[:, 6]) if len(fl) else '-', '%.1e' % np.median(fl[:, 7]) if len(fl) else '-'), flush=True)
            x, v, yc = I.x, I.v, st_['ym']                                       # (the skin: at the midpoint)
            prods.append(st_['products']); passes.append(st_['al'])
            gap, tsum, lam = st_['aux'][2], st_['aux'][3], st_['lam']
            Xs_last = st_['aux'][4]
            Dv = I.dissipated
            Emid, Eprev = float(tsum[:6].sum()), Emid
            Ee_int = I.E_int if args.integrator == 'midpoint' else Emid         # elastic at the step's end (midpoint: at the skin extrapolated)
            Hs = .5*float((mass[:, None]*v*v).sum())+Ee_int+float(tsum[6])+args.gravity*float((mass*x[:, 1]).sum())+Dv-I.contact_work
            if args.energy_audit:
                xm_, ym_ = st_['xm'], st_['ym']
                dX, dY = 2*(xm_-x_n), 2*(ym_-y_int)
                dot = lambda gx_, gy_: float((dX*gx_).sum()+(dY*gy_).sum())
                TE = dot(*masked(wE, xm_, ym_)[1:]); TC = dot(*masked(wC, xm_, ym_)[1:])
                TV = dot(*masked(wV, xm_, ym_)[1:]) if W.vis is not None else 0.
                y_next = 2*ym_-y_int
                E_np1 = masked(wE, 2*xm_-x_n, y_next)[0]
                W.total_mode(); _, gxT, gyT = W.grad(xm_, ym_)
                rx = 4/dt**2*mass[:, None]*(xm_-x_n-.5*dt*v_n)-fg+gxT
                aud['E_avg'] += (Ee_int-Eint0)-(E_np1-E_n)
                aud['E_quad'] += (E_np1-E_n)-TE
                aud['visc'] += (I.dissipated-D0)-TV
                aud['contact'] += (float(tsum[6])-Phi0)-(I.contact_work-Wc0)-TC
                aud['resid'] += dot(rx, gyT)
                E_n, y_int = E_np1, y_next
                if (s+1) % args.every == 0:
                    print('   audit: H - H_0 %+.4e = elastic averaging %+.3e + elastic quadrature %+.3e + viscous %+.3e + contact %+.3e + residual %+.3e '
                          '(sum %+.4e)' % (Hs-rec[0][5], aud['E_avg'], aud['E_quad'], aud['visc'], aud['contact'], aud['resid'], sum(aud.values())), flush=True)
            if not math.isfinite(Hs) or st_['products'] > 6000 or (Hlast is not None and abs(Hs/Hlast-1) > 1e-4):
                print('anomaly at step %d (t %.5f s): H %.6e (previous step %.6e), products %d, passes %d, last residual %.1e' % (
                    s+1, (s+1)*dt, Hs, Hlast if Hlast is not None else float('nan'), st_['products'], st_['al'], st_['residual']), flush=True)
                if not math.isfinite(Hs) or args.stop_on_anomaly:
                    break
            Hlast = Hs
            continue
        if W.vis is not None:                                                    # viscosity: the step's start the reference, the
            fx = -W.visc_ref(x, yc)[1]+fg                                        # bodies' force there without the viscous term
        v = v+dt*fx/mass[:, None]
        if W.vis is not None:                                                    # and the bodies implicit there, with the skin's
            v, k_ = visc_body_update(W, x, yc, mass, v, (yc-yp)/dt, dt, b['vblocks'], tol=args.visc_tol); cgs.append(k_)   # last velocity
            # (10-24: its velocity predicted for this step, predict_skin with v*, is exact in free fall but diverged at the
            # landing: where the ground holds the skin the prediction is wrong, the bodies are not slowed while the solve
            # counts the dissipation)
        x_old = x
        x = x+dt*v
        yi = 2*yc-yp if args.predict == 'linear' else predict_skin(S, x, x_old, xp, yc, yp)
        xp = x_old
        if args.points:
            held_info = S.refresh_points(x, yi, prec)
            if held_info[1] and held_info[2] < -1e-9:
                print('  step %d: %d regions joined the held set with lowest gap %.2e m (held %d)' % (s+1, held_info[1], held_info[2], held_info[0]), flush=True)
        yn, st_ = S.solve_contact(x, yi, prec, ref, tol=tol, maxit=300, al_max=args.al_max)
        prods.append(st_['products']); passes.append(st_['al'])
        gx, gap, tsum, lam = st_['aux'][0], st_['aux'][2], st_['aux'][3], st_['lam']
        Dv += W.dissipated(tsum)
        Xs_last = st_['aux'][4]
        prev, cur = cur, dict(x=x, v=v, y=yn, fx=-gx+fg, tsum=tsum, lam_used=st_['lam_used'], lam=lam, rho=W.trho.clone(), gap=gap)
        fx = -gx+fg
        yp, yc = yc, yn
        Hs = .5*float((mass[:, None]*v*v).sum())+float(tsum[:7].sum())+args.gravity*float((mass*x[:, 1]).sum())+Dv
        if not math.isfinite(Hs) or st_['products'] > 3000 or (Hlast is not None and abs(Hs/Hlast-1) > 1e-4):
            act = lam > 0
            print('anomaly at step %d (t %.5f s): H %.6e (previous step %.6e), products %d, passes %d, last residual %.1e, multiplier change %.1e; '
                  'active %d, lowest gap %.2e m, min J %.3f; rho of the active / median: max %.1e; saved' % (
                      s+1, (s+1)*dt, Hs, Hlast if Hlast is not None else float('nan'), st_['products'], st_['al'], st_['residual'], st_['dlam'],
                      int(act.sum()), float(gmin(gap)), math.exp(float(W.tlj.min())), float(W.trho[act].max()/W.trho.median()) if bool(act.any()) else 0.), flush=True)
            np.savez(b['out']/'skin_drop_anomaly.npz', step=s+1, x=x.cpu().numpy(), v=v.cpu().numpy(), y=yc.cpu().numpy(), yp=yp.cpu().numpy(),
                     lam=W.tlam.cpu().numpy(), rho=W.trho.cpu().numpy(), gap=gap.cpu().numpy(),
                     **{'%s_%s' % (w_, k_): t_.cpu().numpy() for w_, d_ in (('prev', prev), ('cur', cur)) if d_ is not None for k_, t_ in d_.items()})
            if not math.isfinite(Hs) or args.stop_on_anomaly:
                break
        Hlast = Hs
    rec = np.array(rec)
    H = rec[:, 5]
    print('H / H_0 - 1: max %.2e, min %.2e, last %.2e; dissipated %.4e (%.3f of the drop\'s m g h); lowest gap over the run %.2e m; '
          'products per step median %d, max %d; passes max %d; viscous CG per step median %s; %.0f s' % (
              (H/H[0]-1).max(), (H/H[0]-1).min(), H[-1]/H[0]-1, Dv, Dv/(M*args.gravity*args.height), rec[:, 8].min(),
              int(np.median(prods)), max(prods), max(passes), int(np.median(cgs)) if cgs else '-', time.time()-t0), flush=True)
    if mid:
        print(args.integrator+': the contact force\'s work (new contacts; in H) %.4e, %.3f of the dissipated' % (
            I.contact_work, I.contact_work/Dv if Dv else float('nan')), flush=True)
    vis_tag = '' if W.vis is None else '_eta%g_zeta%g' % (W.vis['eta'], W.vis['zeta']) if args.beta == 0 else '_beta%g' % args.beta
    np.savez(b['out']/('skin_drop_check%s%s.npz' % (vis_tag, args.tag)),rec=rec, cols=np.array(['t', 'K', 'elastic', 'gravity', 'AL', 'H', 'com_y', 'com_vy', 'gap_min',
                                                                     'contact/weight', 'active', 'products', 'passes', 'com_x', 'com_z', 'com_vx', 'com_vz',
                                                                     'om_x', 'om_y', 'om_z', 'cop_x', 'cop_z', 'dissipated']),
             prods=np.array(prods), passes=np.array(passes), X=np.stack(X_rec), lam=np.stack(lam_rec), pass_info=np.array(pinfo), **(dict(F=np.stack(F_rec)) if F_rec else {}))


if __name__ == '__main__':
    main()
