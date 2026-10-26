"""W3, part 1 (2026-10-23): dynamics with the skin solved every step, no contact, gravity or damping. Bodies by
symplectic Euler (v += dt f / m, x += dt v, f = -dE/dx at the step's solved skin), lumped masses m.mass; the skin from
the linear extrapolation of the last two, trust-region solve on Warp (preconditioner: probed K_rest blocks per body) to
a tolerance tol of |dE/dy| relative to the cells' affine skin. Start: the first 6 FEM modes at random amplitudes
(largest body displacement --amp), at rest. --case part: the synthetic part (its K_yy probed here). Total energy H = 1/2 sum m |v|^2 + E every 20 steps, for tol 1e-3, 1e-4, 1e-6."""
import argparse
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from experiments.dem_fem_mode_match import interpolate
from free_surface.skin_contact_check import setup, f64, dev
from free_surface.skin_warp import SkinWarp, contact_geometry, warp_solver


def parser():
    a = argparse.ArgumentParser()
    a.add_argument('--steps', type=int, default=480)
    a.add_argument('--dt', type=float, default=1/4800)
    a.add_argument('--tols', type=float, nargs='+', default=[1e-3, 1e-4, 1e-6])
    a.add_argument('--case', choices=('foam', 'part'), default='foam')
    a.add_argument('--amp', type=float, default=.005)
    a.add_argument('--out', type=Path, default=None, help='a foam_exact.py directory (default: the case\'s)')
    a.add_argument('--fem', type=Path, default=None, help='FEM modes for the start (default: the case\'s)')
    a.add_argument('--hvp', choices=('forward', 'central'), default='forward', help='Hessian-vector products by one-sided or central differences')
    a.add_argument('--cg', choices=('graph', 'torch'), default='graph', help='each CG iteration one CUDA graph (forward only) or torch ops')
    a.add_argument('--beta', type=float, default=0., help='Kelvin-Voigt viscosity as a time: eta = beta mu, zeta = beta (lam + 2 mu / 3) (s)')
    a.add_argument('--eta', type=float, default=None, help='shear viscosity per unit density (m^2/s), overrides beta')
    a.add_argument('--zeta', type=float, default=None, help='bulk viscosity per unit density (m^2/s), overrides beta')
    a.add_argument('--visc-fd', choices=('central', 'forward'), default='forward', help='the viscous products by central or one-sided differences (10-24: the same to 1e-6 on mode 20, 1.3x faster)')
    a.add_argument('--integrator', choices=('split', 'midpoint', 'bdf2'), default='split', help='bodies by symplectic Euler then the skin (viscosity: bodies implicit with the skin\'s last velocity), or bodies and skin in one implicit step: the midpoint rule or BDF2 (joint_step)')
    a.add_argument('--E', type=float, default=23008.38, help='Young\'s modulus per unit density (default: PAC-NeRF elastic scene 0)')
    a.add_argument('--nu', type=float, default=.3008, help='Poisson\'s ratio')
    a.add_argument('--visc-tol', type=float, default=1e-6, help='the bodies\' viscous update: CG residual relative to its right-hand side (1e-6 vs 1e-9: the same to 1e-6, 17 vs 32 iterations)')
    return a


def build_model(args, contact=None):
    """the model, Warp gradient, solver, preconditioner and masses; contact: dict(normal, c0) of a ground plane
    (SkinWarp.set_contact, every surface cell's exposed sphere)"""
    out = args.out or Path('output/round01_exact' if args.case == 'foam' else 'output/test_part')
    if args.out is None:
        st, _ = setup(args.case, 'boundary_skin.npz', 'shepard0' if args.case == 'foam' else 'shepard')
    else:
        from experiments.pacnerf_dem_forward import build as dem_build
        from free_surface.boundary_skin import BoundaryTerms
        cells = np.load(out/'dem_exact_single'/'cells.npz')
        gg = dict(np.load(out/'dem_exact_single'/'geometry.npz'))
        m = dem_build(gg, getattr(args, 'E', 23008.38), getattr(args, 'nu', .3008), dev, 'noslip', .25, .3, f64, kinematics='exact', nodalvol=True, nonlinear=False, ogden_alpha=2.)
        st = BoundaryTerms(m, gg, cells['points'], cells['radii'], dict(np.load(out/'boundary_skin.npz')), lever='c')
    X, Y, npt = st.X, st.Y, st.npt; v3 = 3*npt
    W = SkinWarp(st)
    bsk = dict(np.load(out/'boundary_skin.npz')); g = dict(np.load(out/'dem_exact_single'/'geometry.npz'))
    beta = getattr(args, 'beta', 0.)
    eta = args.eta if getattr(args, 'eta', None) is not None else beta*W.mu
    zeta = args.zeta if getattr(args, 'zeta', None) is not None else beta*(W.lam+2*W.mu/3)
    if eta > 0 or zeta > 0:
        W.set_viscosity(eta, zeta, args.dt)
        W.visc_central = getattr(args, 'visc_fd', 'central') == 'central'
        print('Kelvin-Voigt viscosity per unit density: eta %.4g, zeta %.4g m^2/s (mu %.4g, lam %.4g)' % (eta, zeta, W.mu, W.lam), flush=True)
    if contact is not None:
        cells = np.load(out/'dem_exact_single'/'cells.npz')
        cg = contact_geometry(torch.as_tensor(cells['points'], dtype=f64, device=dev), torch.as_tensor(cells['radii'], dtype=f64, device=dev),
                              g['body_of_cell'])
        assert (cg['site'].cpu().numpy() == bsk['patch_site']).all()
        W.set_contact(cg, contact['normal'], contact['c0'], torch.as_tensor(bsk['patch_A'], dtype=f64, device=dev),
                      torch.as_tensor(bsk['ent_a'], dtype=f64, device=dev) if contact.get('shift') else None)
        if contact.get('points'):                                                   # at held points of the skin's surface
            from free_surface.skin_contact import SkinSurface
            pc, rc = torch.as_tensor(cells['points'], dtype=f64, device=dev), torch.as_tensor(cells['radii'], dtype=f64, device=dev)
            surf = SkinSurface(st, cg['R'], pc, rc, g['body_of_cell'], 'shepard0')
            W.set_contact_points(surf.body, surf.I.loc_g)
    W.prepare()
    S = warp_solver(st, bsk, g['body_of_cell'], W, 1e-7, args.hvp, args.cg == 'graph')
    kf = out/('skin_Kyy_rest_csr_%s.npz' % st.gamma)                              # K_yy at rest depends on the curvature fit
    # (and on E, nu: probed once at the defaults; with other --E / --nu only the preconditioner is that of the defaults)
    if st.gamma == 'bonds' and (out/'skin_Kyy_rest.npz').exists():                 # skin_step_cost.py's, from before sym_skin
        z = np.load(out/'skin_Kyy_rest.npz'); P_, Q_, vals = z['P'], z['Q'], z['vals']
        rows = (3*P_[:, None, None]+np.arange(3)[None, :, None]).repeat(3, 2); cols = (3*Q_[:, None, None]+np.arange(3)[None, None, :]).repeat(3, 1)
        K = sp.coo_matrix((vals.ravel(), (rows.ravel(), cols.ravel())), shape=(v3, v3)).tocsr(); K = (.5*(K+K.T)).tocsr()
    elif kf.exists():
        K = sp.load_npz(kf)
    else:
        from free_surface.skin_solve import SkinSolver
        K = SkinSolver(st, bsk, g['body_of_cell']).probe_K()
        sp.save_npz(kf, K)
    Bk = S.blocks_probed(K, S.bodies)
    integ = getattr(args, 'integrator', 'split'); mid = integ != 'split'
    if W.vis is not None:                                                            # the skin's Hessian K + w Et''/dt, Et'' ~ eta / mu K
        Bk = Bk*(1+dict(split=1., midpoint=2., bdf2=1.5)[integ]*W.smu/args.dt)       # (w: the viscous term's weight times dt)
    prec = S.preconditioner(Bk, S.bodies)
    if contact is not None and contact.get('points'):
        C = S.points_setup(prec, surf, pc, rc, g['body_of_cell'], 'shepard0', contact.get('kappa', 100.))
        print('contact at held points: %d regions; interpolant bound C = max sum |W|: median %.3f, max %.3f; rho = %g x the contact stiffness' % (
            len(C), float(C.median()), float(C.max()), contact.get('kappa', 100.)), flush=True)
    elif contact is not None:
        kc = 1/S.contact_setup(prec, contact.get('kappa', 100.))
        print('contact: %d surface cells; contact stiffness (skin else held) 1 / a.B^-1 a: median %.3e, min %.3e, max %.3e; rho = %g x that' % (
            len(kc), float(kc.median()), float(kc.min()), float(kc.max()), contact.get('kappa', 100.)), flush=True)
    mass = st.model.mass.to(f64)
    print('masses: sum %.4e, cell volumes sum %.4e (energy per unit density: mass = volume)' % (float(mass.sum()), float(st.V.sum())), flush=True)
    from free_surface.joint_step import body_blocks
    vb = body_blocks(W, st, 'viscous') if W.vis is not None else None
    kb = body_blocks(W, st, 'elastic') if mid else None

    def make_copy():
        """another SkinWarp of the same model (its own buffers and gradient graph, ~1.2 GB on the 8k foam); its mutable
        state (material, weights, references, contact) is the caller's to copy (tangent.sync_state)"""
        Wc = SkinWarp(st)
        if W.vis is not None:
            Wc.set_viscosity(eta, zeta, args.dt)
            Wc.visc_central = W.visc_central
        if contact is not None:
            Wc.set_contact(cg, contact['normal'], contact['c0'], torch.as_tensor(bsk['patch_A'], dtype=f64, device=dev),
                           torch.as_tensor(bsk['ent_a'], dtype=f64, device=dev) if contact.get('shift') else None)
            if contact.get('points'):
                Wc.set_contact_points(surf.body, surf.I.loc_g)
        Wc.prepare()
        return Wc
    return dict(out=out, st=st, W=W, S=S, prec=prec, blocks=Bk, mass=mass, vblocks=vb, kblocks=kb, make_copy=make_copy)


def build(args):
    """build_model and the solved start state (x0, y0): the first 6 FEM modes at random amplitudes"""
    b = build_model(args)
    out, st, W, S, prec = b['out'], b['st'], b['W'], b['S'], b['prec']
    X, Y = st.X, st.Y
    cache = np.load(args.fem or out/('fem_modes_h0.008.npz' if args.case == 'foam' else 'fem_modes_h0.0015.npz'))
    print('lowest FEM frequency %.2f Hz (period %.4f s); run %.4f s' % (cache['f'][0], 1/cache['f'][0], args.steps*args.dt), flush=True)
    fem = dict(nodes=cache['nodes'], node_grid=cache['node_grid'], lo=cache['lo'], h=float(cache['h']), masses=cache['masses'])
    Ux, _ = interpolate(fem, cache['V'][:, :6], X.cpu().numpy())
    U = torch.as_tensor(Ux.transpose(2, 0, 1), dtype=f64, device=dev)
    am = torch.as_tensor(np.random.default_rng(1).standard_normal(6), dtype=f64, device=dev)
    u0 = torch.einsum('k,kni->ni', am, U); u0 = u0*args.amp/float(u0.norm(dim=1).max())
    x0 = X+u0
    y0 = Y.clone()
    for _ in range(5):
        y0 = S.affine_skin(x0, y0)
    y0, s0 = S.solve(x0, y0, prec, float(S.grad(x0, y0).norm()), tol=1e-8, maxit=1000)     # finite differences: ~1e-9 floor
    E0 = float(W.grad(x0, y0)[0])
    print('start: largest displacement %.1f mm, elastic energy %.6e, skin cold start %d products' % (1e3*args.amp, E0, s0['products']), flush=True)
    return dict(b, x0=x0, y0=y0)


def predict_skin(S, x_new, x, x_prev, y, y_prev, iters=4):
    """the skin's start for the solve at the bodies' new positions x_new. The skin = the cells' affine field
    a(x, y) (S.affine_skin, linear in x and y) + its deviation; x_new is known, so only the deviation is extrapolated:
    y = a(x_new, y) + 2 dev_n - dev_n-1, i.e. y = 2 y_n - y_n-1 + e, e = a(x_new - 2 x_n + x_n-1, e) by fixed point
    from the bodies' mean increment per skin point (exact for a uniform acceleration, free fall)"""
    dx = x_new-2*x+x_prev
    e = torch.zeros_like(y).index_add(0, S.e_pid, dx[S.e_body])/S.cnt[:, None]
    for _ in range(iters):
        e = S.affine_skin(dx, e)
    return 2*y-y_prev+e


def visc_body_update(W, x, y, mass, v_star, ydot, dt, blocks=None, tol=1e-9, maxit=500):
    """Kelvin-Voigt viscosity on the bodies, implicit: (M + dt D_xx) v = M v* - dt D_xy ydot, D the viscous products
    at the step's start (x, y), right after W.visc_ref there (SkinWarp.visc_hvp); conjugate gradients preconditioned by
    M + dt R_B B R_B^T per body (blocks B: joint_step.body_blocks 'viscous'; R_B of visc_ref's call), else by M. Returns v and iterations"""
    if blocks is not None:
        R = W.tR.clone()
        Pi = torch.linalg.inv(mass[:, None, None]*torch.eye(3, dtype=f64, device=dev)+dt*R@blocks@R.transpose(1, 2))
        prec = lambda r_: torch.einsum('nij,nj->ni', Pi, r_)
    else:
        prec = lambda r_: r_/mass[:, None]
    zx, zy = torch.zeros_like(v_star), torch.zeros_like(ydot)
    b = mass[:, None]*v_star-dt*W.visc_hvp(x, y, zx, ydot)[0]
    A = lambda u: mass[:, None]*u+dt*W.visc_hvp(x, y, u, zy)[0]
    u = v_star.clone(); r = b-A(u); z = prec(r); p = z.clone(); rz = float((r*z).sum()); bn = float(b.norm())
    for k in range(1, maxit+1):
        Ap = A(p); al = rz/float((p*Ap).sum())
        u = u+al*p; r = r-al*Ap
        if float(r.norm()) <= tol*bn:
            break
        z = prec(r); rz1 = float((r*z).sum()); p = z+(rz1/rz)*p; rz = rz1
    return u, k


def integrate(b, tol, steps, dt):
    """symplectic Euler from (x0, y0) at rest; total energy every 20 steps and products per step. The force on the
    bodies comes from the solve's last evaluation (at the skin it returns)"""
    W, S, prec, mass, x0, y0 = (b[k] for k in ('W', 'S', 'prec', 'mass', 'x0', 'y0'))
    x, v = x0.clone(), torch.zeros_like(x0)
    yp = yc = y0
    H, prods = [], []
    e, fx, _ = W.grad(x, yc)
    for s in range(1, steps+1):
        if (s-1) % 20 == 0:
            H.append(float(e)+.5*float((mass[:, None]*v*v).sum()))
        v = v-dt*fx/mass[:, None]
        x = x+dt*v
        yi = 2*yc-yp
        rf = float(S.grad(x, S.affine_skin(x, yi)).norm())
        yn, st_ = S.solve(x, yi, prec, rf, tol=tol, maxit=300)
        prods.append(st_['products'])
        e, fx = st_['energy'], st_['aux'][0]
        yp, yc = yc, yn
    H.append(float(e)+.5*float((mass[:, None]*v*v).sum()))
    return np.array(H), prods


def main():
    args = parser().parse_args()
    b = build(args)
    out, dt = b['out'], args.dt
    for tol in args.tols:
        t1 = time.time()
        H, prods = integrate(b, tol, args.steps, dt)
        print('tol %.0e: %d steps (%.3f s); H / H_0 - 1 at every 20 steps: max %.2e, min %.2e, last %.2e; products per step median %d, max %d; %.0f s' % (
            tol, args.steps, args.steps*dt, (H/H[0]-1).max(), (H/H[0]-1).min(), H[-1]/H[0]-1, np.median(prods), max(prods), time.time()-t1), flush=True)
        np.save(out/('skin_dynamics_H_tol%.0e_dt%.0e.npy' % (tol, dt)), H)


if __name__ == '__main__':
    main()
