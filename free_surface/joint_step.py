"""The bodies and the skin in one implicit step (2026-10-24): the implicit midpoint rule for the bodies, the skin at the
step's midpoint (massless: in equilibrium there), viscosity (SkinWarp.set_viscosity) on the same midpoint velocities.
Per step the midpoint state z = (x_m, y_m) minimises
  Pi(z) = 2 / dt^2 sum_B m_B |x_m - x_n - dt v_n / 2|^2 - f_ext . x_m + E(x_m, y_m) + Phi_c(x_m, y_m) + 2 Et(x_m, y_m) / dt
(Et the viscous counterpart on the measures' change from the step's start m_n: SkinWarp loss weight 2 / dt). Its
stationarity is the midpoint rule M (v_n+1 - v_n) / dt = f_ext - dE/dx(mid) - D v_mid with x_n+1 = 2 x_m - x_n,
v_n+1 = 4 (x_m - x_n) / dt - v_n, and the skin's balance dE/dy(mid) + D_y (v_mid, ydot_mid) = 0: the bodies and the skin
feel the viscous forces of the same velocities; the step dissipates 8 Et / dt. The references go on as
m_n+1 = 2 m(mid) - m_n. Without viscosity the rule keeps the energy of a linear system exactly (no damping). Contact:
at the step's end, 2 g(mid) - g_n >= 0 (k_contact's cpar, goff; held at the midpoint the bodies' end overshot into the
ground and the next step pushed them out: the energy rose), so that its force does work lam (g_end - g_n) only at a new
contact (contact_work).
Solver: SkinSolver.solve (trust-region Newton-CG) and the augmented Lagrangian of WarpSkinSolver.solve_contact on z;
products by one-sided differences of the gradient (one Warp pass gives dE/dx and dE/dy); preconditioner block
diagonal: the skin's (turned group blocks, contact by Sherman-Morrison) and per body 4 m / dt^2 + R (K_B + 2 D_B / dt) R^T
(K_B, D_B: body_blocks at rest)."""
import numpy as np
import scipy.sparse as sp
import torch
import warp as wp

from free_surface.skin_contact_check import f64, dev
from free_surface.skin_solve import SkinSolver
from free_surface.skin_warp import (NSLOT, k_cg_absmax, k_cg_dir, k_cg_dots, k_cg_gather, k_cg_hd, k_cg_perturb, k_cg_reset,
                                    k_cg_shift, k_cg_step)


@wp.kernel(enable_backward=False)
def k_joint_hd_x(gx: wp.array(dtype=wp.vec3d), xp: wp.array(dtype=wp.vec3d), xt: wp.array(dtype=wp.vec3d), mass: wp.array(dtype=wp.float64),
                 f: wp.array(dtype=wp.vec3d), cin: wp.float64, g: wp.array(dtype=wp.vec3d), d: wp.array(dtype=wp.vec3d),
                 sc: wp.array(dtype=wp.float64), h: wp.float64, Hd: wp.array(dtype=wp.vec3d), red: wp.array2d(dtype=wp.float64)):
    """the bodies' part of H d by one-sided differences: the gradient at x + h d / s with the inertia c_in m (x - x_t) - f"""
    i = wp.tid()
    gt = gx[i]+cin*mass[i]*(xp[i]-xt[i])-f[i]
    v = (gt-g[i])*(wp.max(sc[0], wp.float64(1e-300))/h)
    Hd[i] = v
    wp.atomic_add(red, 1, i % NSLOT, wp.dot(d[i], v))


@wp.kernel(enable_backward=False)
def k_body_prec(Pi: wp.array(dtype=wp.mat33d), r: wp.array(dtype=wp.vec3d), z: wp.array(dtype=wp.vec3d)):
    i = wp.tid()
    z[i] = Pi[i]@r[i]


class JointCG:
    """WarpCG for the joint vector (bodies x, skin y; two arrays each): one iteration of the Steihaug CG as one CUDA
    graph: s = max |d|, H d by one-sided differences of the joint gradient (W._run at z + h d / s; the bodies' inertia
    added), alpha, p, Hp, r, z = P r (bodies: Pi; skin: WarpPrec.apply), r.r, r.z, beta, d. Torch views t[name + 'x' /
    'y']; tPi, txt, tf set per outer iteration / step"""

    def __init__(self, W, wprec, mass, c_in, h):
        n, npt = W.n, W.npt
        self.W, self.wprec, self.h, self.c_in, self.n = W, wprec, h, float(c_in), n
        names = ('z', 'g', 'd', 'd_old', 'Hd', 'p', 'Hp', 'r', 'zz')
        self.a = {k+s: wp.zeros(n if s == 'x' else npt, dtype=wp.vec3d, device='cuda') for k in names for s in 'xy'}
        self.t = {k: wp.to_torch(v, requires_grad=False) for k, v in self.a.items()}
        self.Pi = wp.zeros(n, dtype=wp.mat33d, device='cuda'); self.tPi = wp.to_torch(self.Pi, requires_grad=False)
        self.xt = wp.zeros(n, dtype=wp.vec3d, device='cuda'); self.txt = wp.to_torch(self.xt, requires_grad=False)
        self.f = wp.zeros(n, dtype=wp.vec3d, device='cuda'); self.tf = wp.to_torch(self.f, requires_grad=False)
        self.m = wp.from_torch(mass.contiguous(), dtype=wp.float64)
        self.red = wp.zeros((4, 256), dtype=wp.float64, device='cuda')
        self.sc = wp.zeros(8, dtype=wp.float64, device='cuda'); self.tsc = wp.to_torch(self.sc, requires_grad=False)
        self.hsc = wp.zeros(8, dtype=wp.float64, device='cpu', pinned=True); self.hv = self.hsc.numpy()
        self._iteration()
        wp.capture_begin()
        self._iteration()
        self.graph = wp.capture_end()

    def _iteration(self):
        a, W, h, n, npt = self.a, self.W, self.h, self.n, self.W.npt
        wp.launch(k_cg_reset, 1, inputs=[], outputs=[self.sc])
        self.red.zero_()
        wp.launch(k_cg_absmax, n, inputs=[a['dx']], outputs=[self.red])
        wp.launch(k_cg_absmax, npt, inputs=[a['dy']], outputs=[self.red])
        wp.launch(k_cg_gather, (1, 256), inputs=[self.red, 0], outputs=[self.sc])
        wp.launch(k_cg_perturb, n, inputs=[a['zx'], a['dx'], self.sc, h], outputs=[W.xw])
        wp.launch(k_cg_perturb, npt, inputs=[a['zy'], a['dy'], self.sc, h], outputs=[W.yw])
        W._run_products()
        wp.launch(k_joint_hd_x, n, inputs=[W.xw.grad, W.xw, self.xt, self.m, self.f, self.c_in, a['gx'], a['dx'], self.sc, h],
                  outputs=[a['Hdx'], self.red])
        wp.launch(k_cg_hd, npt, inputs=[W.yw.grad, a['gy'], a['dy'], self.sc, h], outputs=[a['Hdy'], self.red])
        wp.launch(k_cg_gather, (1, 256), inputs=[self.red, 1], outputs=[self.sc])
        for s, m in (('x', n), ('y', npt)):
            wp.launch(k_cg_step, m, inputs=[a['d'+s], a['Hd'+s], self.sc], outputs=[a['p'+s], a['Hp'+s], a['r'+s]])
        wp.launch(k_body_prec, n, inputs=[self.Pi, a['rx']], outputs=[a['zzx']])
        a['zzy'].zero_()
        self.wprec.apply(a['ry'], a['zzy'])
        for s, m in (('x', n), ('y', npt)):
            wp.launch(k_cg_dots, m, inputs=[a['r'+s], a['zz'+s]], outputs=[self.red])
        wp.launch(k_cg_gather, (2, 256), inputs=[self.red, 2], outputs=[self.sc])
        for s, m in (('x', n), ('y', npt)):
            wp.launch(k_cg_dir, m, inputs=[a['zz'+s], self.sc], outputs=[a['d'+s], a['d_old'+s]])
        wp.launch(k_cg_shift, 1, inputs=[], outputs=[self.sc])
        wp.copy(self.hsc, self.sc)

    def step(self):
        wp.capture_launch(self.graph)
        wp.synchronize_stream()
        hv = self.hv
        return float(hv[1]), float(hv[3]), float(hv[4])

    def launch(self, stream=None):
        """one iteration on stream (several JointCGs on their own SkinWarps at once: tangent.Tangent); read() after a
        synchronize"""
        wp.capture_launch(self.graph, stream=stream)

    def read(self):
        hv = self.hv
        return float(hv[1]), float(hv[3]), float(hv[4])


def body_blocks(W, st, which):
    """the bodies' 3 x 3 blocks at rest of the elastic Hessian with the skin held ('elastic', one-sided differences of
    the gradient without the viscous term) or of the viscous products ('viscous', SkinWarp.visc_hvp, reference at
    rest), by probing: bodies coloured so that no two of a colour share a bond neighbour, one product per colour and
    axis (a block picks up its colour's bodies three or more bonds away: a preconditioner's accuracy); symmetrised"""
    X, Y, n = st.X, st.Y, st.n
    i, j = st.i.cpu().numpy(), st.j.cpu().numpy()
    A = sp.coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n))
    A = ((A+A.T+sp.eye(n)) > 0).astype(np.int32).tocsr()
    A2 = (A@A).tocsr()
    col = -np.ones(n, dtype=int)
    for a in range(n):
        used = set(col[A2.indices[A2.indptr[a]:A2.indptr[a+1]]].tolist())
        c = 0
        while c in used:
            c += 1
        col[a] = c
    B = torch.zeros(n, 3, 3, dtype=f64, device=dev); zy = torch.zeros_like(Y)
    colt = torch.as_tensor(col, device=dev)
    if which == 'viscous':
        W.visc_ref(X, Y)
    else:
        W.elastic_mode(); W.ttw[6] = 0.; g0 = W.grad(X, Y)[1]; h = 1e-7              # (without the contact term)
    for c in range(int(col.max())+1):
        m = colt == c
        for k in range(3):
            w = torch.zeros_like(X); w[m, k] = 1.
            if which == 'viscous':
                B[m, :, k] = W.visc_hvp(X, Y, w, zy)[0][m]
            else:
                B[m, :, k] = ((W.grad(X+h*w, Y)[1]-g0)/h)[m]
    W.total_mode()
    print('%s blocks of the bodies at rest by %d colours (%d products)' % (which, int(col.max())+1, 3*(int(col.max())+1)), flush=True)
    return .5*(B+B.transpose(1, 2))


def tangents(normal):
    """two unit vectors spanning the plane normal to `normal` (the friction's directions)"""
    n = np.asarray(normal, dtype=float)
    n = n/np.linalg.norm(n)
    e = np.eye(3)[int(np.argmin(np.abs(n)))]
    t1 = np.cross(n, e); t1 /= np.linalg.norm(t1)
    return t1, np.cross(n, t1)


class JointSolver:
    """SkinSolver's trust-region solve and the contact's augmented Lagrangian on z = (x, y) stacked (n + npt, 3): Pi =
    c_in / 2 sum m |x - x_t|^2 - f_ext . (x - x_n) + the SkinWarp loss (its viscous weight W.visc_weight / dt = wvis / dt);
    the midpoint rule c_in 4 / dt^2, x_t = x_n + dt v_n / 2, wvis 2; BDF2 c_in 9 / (4 dt^2), wvis 3 / 2"""
    solve = SkinSolver.solve
    steihaug = SkinSolver.steihaug

    def __init__(self, S, W, mass, Kb, Db, dt, c_in, wvis, h=1e-8):
        # h: the products' one-sided step; 1e-8 as Tangent's (10-25: with the friction glue x 100 it balances the truncation
        # with the rounding), so a forward is the same with or without tangents
        self.S, self.W, self.mass, self.Kb, self.Db, self.dt, self.h = S, W, mass, Kb, Db, dt, h
        self.c_in, self.wvis = c_in, wvis
        self.n = W.n
        self.st = S.st
        self.rejected_logs = []
        self.use_graph, self._jcg = True, None                                     # each CG iteration one graph (JointCG)

    def steihaug(self, x, z, g, prec, delta, stop, cap):
        """SkinSolver.steihaug with each CG iteration one JointCG graph (as WarpSkinSolver.steihaug)"""
        if not (self.use_graph and getattr(prec, 'wprec', None) is not None and hasattr(prec, 'Pi')):
            return SkinSolver.steihaug(self, x, z, g, prec, delta, stop, cap)
        if self._jcg is None or self._jcg.wprec is not prec.wprec:
            self._jcg = JointCG(self.W, prec.wprec, self.mass, self.c_in, self.h)
        cg, n = self._jcg, self.n
        T = cg.t
        zv, gv = z.view(-1, 3), g.view(-1, 3)
        T['zx'].copy_(zv[:n]); T['zy'].copy_(zv[n:]); T['gx'].copy_(gv[:n]); T['gy'].copy_(gv[n:])
        cg.tPi.copy_(prec.Pi); cg.txt.copy_(self.x_t); cg.tf.copy_(self.f_ext)
        r = -g[None]
        zz = prec(r)
        rv, zzv = r.view(-1, 3), zz.view(-1, 3)
        T['rx'].copy_(rv[:n]); T['ry'].copy_(rv[n:]); T['dx'].copy_(zzv[:n]); T['dy'].copy_(zzv[n:])
        rz = float((r*zz).sum())
        if delta is None:
            delta = rz**.5
        cg.tsc[5] = rz
        for s in 'xy':
            T['p'+s].zero_(); T['Hp'+s].zero_()
        pMp, pMd, dMd = 0., 0., rz
        k = 0
        def out(edge, neg):
            return (torch.cat((T['px'], T['py'])).reshape(1, -1).clone(), torch.cat((T['Hpx'], T['Hpy'])).reshape(1, -1).clone(),
                    pMp, delta, edge, neg, k)
        def back(tau):                                                             # along d_old (the graph took alpha d)
            for s in 'xy':
                T['p'+s].add_(T['d_old'+s], alpha=tau); T['Hp'+s].add_(T['Hd'+s], alpha=tau)
        while True:
            dHd, rr, rz1 = cg.step(); k += 1
            al = rz/dHd if dHd > 0 else 0.                                         # as on the device
            be = rz1/rz if rz != 0 else 0.
            def to_edge():
                return (-pMd+(pMd**2+dMd*(delta**2-pMp))**.5)/dMd
            if dHd <= 0:                                                           # the graph took no step (alpha 0)
                back(to_edge())
                return out(True, 1)
            if pMp+2*al*pMd+al**2*dMd >= delta**2:                                 # back from p + alpha d to the edge
                back(to_edge()-al)
                return out(True, 0)
            pMp = pMp+2*al*pMd+al**2*dMd
            if rr**.5 <= stop or k >= cap:
                return out(False, 0)
            pMd = be*(pMd+al*dMd); dMd = rz1+be**2*dMd; rz = rz1

    def set_step(self, x_n, x_t, f_ext):
        self.x_n, self.x_t, self.f_ext = x_n, x_t, f_ext

    def split(self, z):
        z = z.view(-1, 3)
        return z[:self.n], z[self.n:]

    def eg(self, x, z):
        """Pi, its gradient (flat) and the aux of WarpSkinSolver.eg: (dE/dx without inertia, R_B, gaps, terms, X*)"""
        W, m = self.W, self.mass[:, None]
        xm, ym = self.split(z)
        e, gx, gy = W.grad(xm, ym)
        d = xm-self.x_t
        e = e+.5*self.c_in*float((m*d*d).sum())-float((self.f_ext*(xm-self.x_n)).sum())   # (relative to the step's start:
        # a constant like m g y_n (~0.07 here) would leave the trust region's ratio to the rounding of e0 - e1)
        g = torch.cat((gx+self.c_in*m*d-self.f_ext, gy)).reshape(-1)
        c_ = W.ct is not None
        return e, g, (gx, W.tR.clone(), W.tgap.clone() if c_ else None, W.tsum.clone(), W.tXs.clone() if c_ else None,
                      W.tslip.clone() if c_ else None, W.tff.clone() if c_ else None)

    def hvp(self, x, z, g):
        """one-sided differences of the gradient along each row (scaled to largest entry 1, step h)"""
        h = self.h
        def H(V):
            out = torch.empty_like(V)
            for k in range(V.shape[0]):
                s = V[k].abs().max().clamp_min(1e-300)
                out[k] = (self.eg(x, z+(h/s)*V[k].view(-1, 3))[1]-g)*(s/h)
            return out
        return H

    def prec_at(self, make_prec, x, z, aux):
        """the skin's preconditioner (WarpSkinSolver.prec_at: turned blocks, active contacts) and the bodies' blocks
        c_in m + R (K_B + wvis D_B / dt) R^T"""
        n, dt, W = self.n, self.dt, self.W
        if W.ct is not None and float(W.tcpar[2]) > .5:                            # the held active set: its penalty
            fs = make_prec(aux[1])                                                 # wt cs^2 rho a a^T
            if getattr(getattr(fs, 'wprec', None), 'ck', 0):
                om = self.S.contact_weights(aux[4])
                wcs = float(W.tcpar[1]*W.tcpar[0]**2)
                fs.wprec.contact_update(om, W.trho*wcs*(W.tact > 0))
                if float(W.tfpar[0]) > 0:                                          # friction glue, the side at this iterate (10-25:
                    q = W.tlamt-W.trhot[:, None]*aux[5]                            # not the last step's marks): holding k, yielding
                    R = float(W.tfpar[0])*W.tlamR.clamp(min=0)                     # k R / |q| (in both tangents; the yield's is its
                    qn = q.norm(dim=1)                                             # perpendicular part's only)
                    k_t = torch.where(W.tlamR > 0, torch.where(qn <= R, W.trhot, W.trhot*R/qn.clamp(min=1e-300)), torch.zeros_like(W.trhot))   # (the glue: the step's start contact)
                    for t in tangents(W.ct['normal']):
                        fs.wprec.contact_update(om, k_t*wcs, direction=t)
        else:
            fs = self.S.prec_at(make_prec, None, None, aux)
        R = aux[1]
        Bb = self.Kb if self.Db is None else self.Kb+self.wvis/dt*self.Db
        Pi = torch.linalg.inv(self.c_in*self.mass[:, None, None]*torch.eye(3, dtype=f64, device=dev)+R@Bb@R.transpose(1, 2))
        def f(r):
            k = r.shape[0]
            r = r.view(k, -1, 3)
            zb = torch.einsum('nij,knj->kni', Pi, r[:, :n])
            zs = fs(r[:, n:].reshape(k, -1)).view(k, -1, 3)
            return torch.cat((zb, zs), 1).reshape(k, -1)
        f.wprec = getattr(fs, 'wprec', None)
        f.Pi = Pi                                                                  # (for JointCG)
        return f

    def solve_contact(self, z, make_prec, ref, tol=1e-4, maxit=600, al_max=50, theta=0., hook=None):
        """WarpSkinSolver.solve_contact on z: the solve with the multipliers held, then lam <- max(0, lam - rho gap)
        until |lam change| / kappa <= tol ref and the active set stays. theta > 0, inexact passes: a solve stops once its
        residual is theta times the force the multiplier update would bring from the current gaps (the last pass then
        goes to tol). ⚠ 10-24, theta 0.1: once new cells touched next to active ones the active set cycled (50 passes,
        energy -4.9 %): there each pass only halves the multipliers' error (rho is set per cell from its own skin
        block; neighbours sharing skin points stiffen its gap to ~2 rho), and loose gaps flip the set; default off.
        hook(z) (10-26, plastic.Plasticity.step_merged): another outer update (the plastic references), True once it has
        settled, called only when the contact has settled (multipliers and set) for the current references; the passes
        then go on from the multipliers and set they reached, and stop when both have settled"""
        W, S = self.W, self.S
        if W.ct is None:
            prods = 0
            for it in range(1, al_max+1):
                z, s_ = self.solve(None, z, make_prec, ref, tol=tol, maxit=maxit)
                prods += s_['products']
                if hook is None or hook(z):
                    break
            s_.update(products=prods, al=it, dlam=0., lam=None, lam_used=None)
            return z, s_
        wp_ = getattr(make_prec, 'ck', 0) and make_prec
        if wp_:
            W.trho.copy_(torch.where(wp_.taua > 0, S.kappa/wp_.taua, W.trho))
        fr = float(W.tfpar[0]) > 0                                                 # friction: the glue's force at the step's start
        if fr:                                                                     # and its bound's multipliers (R = mu lamR) held
            W.tlamR.copy_(W.tlam)                                                  # through the step's passes
        lamt_used, fst_used = (W.tlamt.clone(), W.tfst.clone()) if fr else (None, None)   # held through the step (its update:
        alinfo = []                                                                # MidpointIntegrator); per pass: moved, |dlam| / kappa
        prods = 0
        info = []                                                                  # per pass: products, outer, rejected, negative,
        tol_fn = (lambda aux: theta*float(((W.tlam-W.trho*aux[2]).clamp(min=0)-W.tlam).norm())/ref) if theta > 0 else None   # the start's residual
        for it in range(1, al_max+1):                                              # on the bodies and the skin
            g0 = self.eg(None, z)[1].view(-1, 3)
            rb, rs = float(g0[:self.n].norm())/ref, float(g0[self.n:].norm())/ref
            lines = []
            z, s_ = self.solve(None, z, make_prec, ref, tol=tol, maxit=maxit, log=lines.append, tol_fn=tol_fn)
            info.append((s_['products'], s_['outer'], s_['rejected'], s_['negative'], rb, rs))
            if s_['rejected'] > 5:                                                 # the outer iterations of a pass with many rejections
                self.rejected_logs.append((it, lines))
            prods += s_['products']
            lam_used = W.tlam.clone()
            lr = W.tlam-W.trho*s_['aux'][2]
            lam_new = lr.clamp(min=0)
            moved = 0
            act = (lr > 0).to(W.tact.dtype)
            joined = left = []
            if float(W.tcpar[2]) > .5:                                             # the active set for the next solve: the
                moved = int((act != W.tact).sum())                                 # multipliers' max (penetrating cells join,
                joined = torch.nonzero((act > 0) & (W.tact == 0))[:, 0].tolist()  # (the record: which cells, 10-26)
                left = torch.nonzero((act == 0) & (W.tact > 0))[:, 0].tolist()
                lam_new = torch.where((act > 0) & (W.tact == 0), torch.zeros_like(lam_new), lam_new)   # pulling ones leave). A
                W.tact.copy_(act)                                                  # joining cell starts at 0: off the set it went
            dlv = (lam_new-W.tlam).abs()
            dl = float((lam_new-W.tlam).norm())                                    # in unpenalised, rho times that depth was ~100x
            ib = int(dlv.argmax())
            alinfo.append(dict(moved=moved, dl=dl/S.kappa, joined=joined, left=left, gaps={c: float(s_['aux'][2][c]) for c in (joined+left)[:8]},
                               lr={c: float(lr[c]) for c in (joined+left)[:8]}, top=ib, top_dl=float(dlv[ib])/S.kappa, top_lam=float(lam_new[ib]),
                               top_gap=float(s_['aux'][2][ib]), residual=float(s_['residual']), products=int(s_['products']), rejected=int(s_['rejected'])))
            W.tlam.copy_(lam_new)                                                  # its force (10-24, step 349)
            if dl/S.kappa <= tol*ref and moved == 0 and (hook is None or hook(z)):  # the hook only on a settled contact
                break                                                              # (10-26: called every pass, each update
            # taking the other as fixed, the multipliers' step rho (elastic) and the plastic return cycled at 2 of 17
            # landing steps, period 3 and 2, to the 50-pass cap); on its change the passes go on from these multipliers
        if s_['residual'] > tol:                                                   # the last pass ended loose: to tol
            lam_used = W.tlam.clone()                                              # (the multipliers this solve holds)
            z, s2 = self.solve(None, z, make_prec, ref, tol=tol, maxit=maxit)
            info.append((s2['products'], s2['outer'], s2['rejected'], s2['negative'], s_['residual'], 0.))
            prods += s2['products']
            s_ = dict(s2, lam=None)
        s_.update(products=prods, al=it, dlam=dl, lam=lam_new, lam_used=lam_used, passes=info, lamt_used=lamt_used, fst_used=fst_used,
                  al_info=alinfo)
        return z, s_


class MidpointIntegrator:
    """the implicit midpoint rule with the skin at the midpoints (JointSolver). State: x, v (bodies, integer times), a
    (the last step's mean acceleration (v_n+1 - v_n) / dt), the last two midpoints (bodies and skin) for the skin's
    start (skin_dynamics_check.predict_skin), the references m_n (SkinWarp.tref) and the energy dissipated"""

    def __init__(self, b, dt, x0, v0, y0, f_ext, tol=1e-4, ref=None, contact=False, al_max=50):
        from free_surface.skin_dynamics_check import predict_skin
        self.predict_skin = predict_skin
        st, W, S = b['st'], b['W'], b['S']
        self.b, self.st, self.W, self.S, self.prec, self.mass = b, st, W, S, b['prec'], b['mass']
        self.dt, self.f_ext, self.tol, self.contact, self.al_max = dt, f_ext, tol, contact, al_max
        self.ref = ref if ref is not None else float((9.8*self.mass).norm())
        W.visc_weight = 2.
        W.total_mode()
        self.J = JointSolver(S, W, self.mass, b['kblocks'], b['vblocks'], dt, 4/dt**2, 2.)
        self.x, self.v = x0.clone(), v0.clone()
        if W.ct is not None:                                                       # the gap itself (set_contact's cpar, goff), not
            W.tcpar[0], W.tcpar[1], W.tcpar[2] = 1., 1., 0.                        # the last run's end gaps 2 g_mid - g_n (10-25: a
            W.tgoff.zero_()                                                        # second run started from those, hung or diverged)
        _, gx, _ = W.visc_ref(x0, y0) if W.vis is not None else W.grad(x0, y0)       # m_0, and the force at the start
        self.a = (f_ext-gx)/self.mass[:, None]
        if W.ct is not None:                                                       # the contact at the step's end: 2 g_mid - g_n
            W.tgoff.copy_(W.tgap)                                                  # >= 0 (k_contact's cs 2, wt 1/2, goff g_n), so that
            W.tcpar[0], W.tcpar[1] = 2., .5                                        # the contact force does no work but at a new contact
            W.tcpar[2] = 1.                                                        # an active set held within each solve (no kinks
            W.tact.copy_((W.tlam > 0).to(W.tact.dtype))                            # for the trust region), moved between them
            W.tfst.copy_(W.tact)                                                   # friction: the active cells start stuck
        self.xm1, self.ym1 = x0-.5*dt*v0, y0.clone()                               # the midpoints one and two steps back
        self.xm2, self.ym2 = self.xm1-dt*v0, y0.clone()
        if not torch.equal(v0, torch.zeros_like(v0)):                              # the skin moving with the bodies
            self.ym1 = predict_skin(S, self.xm1, x0, x0, y0, y0)
            self.ym2 = predict_skin(S, self.xm2, self.xm1, x0, self.ym1, y0)
        self.dissipated = self.contact_work = self.friction_work = 0.
        self.Em = None
        self.y_int = y0.clone()                                                    # the skin at integer times, extrapolated
        self.E_int = self.elastic_at(x0, y0)                                       # y_n+1 = 2 y_m - y_n, and the elastic energy there
        self.hold_points()

    def hold_points(self):
        """the friction's material points for the next step: each contact cell's lowest point at the integer time (the
        last grad call, elastic_at) and its position there"""
        W = self.W
        if W.ct is not None:
            self.wfix, self.pn = W.tXs.clone(), W.tcpos.clone()

    def elastic_at(self, x, y):
        """the elastic energy (terms 0-5) at (x, y): the record's elastic energy at integer times, at the skin extrapolated
        from the midpoints, the one the rule's balance holds with (10-24 audit: the mean of two midpoints' energies left
        -2.6e-6 at the landing, all of the record's drift; this closes it to ~2e-9)"""
        W = self.W
        tw = W.ttw.clone()
        W.ttw.zero_(); W.ttw[:6] = 1.
        e = float(W.grad(x, y)[0])
        W.ttw.copy_(tw)
        return e

    def step(self, z0=None):
        """one step; returns the solve's stats (with 'aux' at the midpoint). z0: the solve's start (bodies and skin at
        the midpoint) instead of the predictors (plastic.Plasticity.step: the same step again)"""
        W, dt, n = self.W, self.dt, self.W.n
        x, v = self.x, self.v
        if z0 is None:
            xm = x+.5*dt*v+.25*dt*dt*self.a
            ym = self.predict_skin(self.S, xm, self.xm1, self.xm2, self.ym1, self.ym2)
        else:
            xm, ym = z0[:n], z0[n:]
        if W.ct is not None:
            W.twfix.copy_(self.wfix); W.tpn.copy_(self.pn)
        self.J.set_step(x, x+.5*dt*v, self.f_ext)
        z, s_ = self.J.solve_contact(torch.cat((xm, ym)), self.prec, self.ref, tol=self.tol, al_max=self.al_max, hook=getattr(self, 'hook', None))
        xm, ym = z[:n], z[n:]
        vmid = 2*(xm-x)/dt
        vn = 2*vmid-v
        self.a = (vn-v)/dt
        self.x, self.v = 2*xm-x, vn
        self.xm2, self.ym2, self.xm1, self.ym1 = self.xm1, self.ym1, xm, ym
        ts = s_['aux'][3]
        self.Em = ts
        if W.ct is not None:                                                       # the contact force's work lam (g_end - g_n):
            if s_['lam'] is not None:                                              # 0 in lasting contact, < 0 at a new one
                self.contact_work += float((s_['lam']*(s_['aux'][2]-W.tgoff)).sum())
            W.tgoff.copy_(s_['aux'][2])                                            # the end gaps, the next step's g_n
            if float(W.tfpar[0]) > 0:                                              # the friction's work f . slip (<= 0), and the
                f, sl = s_['aux'][6], s_['aux'][5]                                 # glue's force kept for the next step on the
                self.friction_work += float((f*sl).sum())                          # next active set (a leaving cell's glue breaks)
                q = s_['lamt_used']-W.trhot[:, None]*sl
                R = float(W.tfpar[0])*W.tlamR.clamp(min=0)
                on = W.tact > 0
                W.tlamt.copy_(f*on[:, None])
                W.tfst.copy_(torch.where(on, torch.where(q.norm(dim=1) <= R, 1, 2), 0).to(W.tfst.dtype))
        if W.vis is not None and not getattr(self, 'no_visc', False):
            self.dissipated += 8*float(ts[7])/dt
            W.grad(xm, ym)                                                         # the measures at the midpoint
            r = W.tref
            mid = dict(d=W.tbd, th=W.tbth, a=W.tba, dl=W.tdl, L=W.tlj, P=W.tR.transpose(1, 2)@W.tF)
            for k_ in r:
                r[k_].copy_(2*mid[k_]-r[k_])
        self.y_int = 2*ym-self.y_int
        self.E_int = self.elastic_at(self.x, self.y_int)
        self.hold_points()
        s_['xm'], s_['ym'], s_['vmid'] = xm, ym, vmid
        return s_


def measures(W):
    """the elasticity's measures of SkinWarp's last grad call (copies): bond d, theta; R^T daff; dl; log J; R^T F"""
    return dict(d=W.tbd.clone(), th=W.tbth.clone(), a=W.tba.clone(), dl=W.tdl.clone(), L=W.tlj.clone(), P=W.tR.transpose(1, 2)@W.tF)


class BDF2Integrator:
    """BDF2 (2026-10-24, L-stable and second order): every force at the step's end, bodies and skin in one solve
    (JointSolver on z = (x_n+1, y_n+1)). With v_n+1 = (3 x_n+1 - 4 x_n + x_n-1) / (2 dt) and
    M (3 v_n+1 - 4 v_n + v_n-1) / (2 dt) = f_ext - dE/dx - D v_n+1, the step minimises
      Pi = 9 / (8 dt^2) sum m |x - x_t|^2 - f_ext . (x - x_n) + E + Phi_c + 3 Et / (2 dt),
      x_t = (4 x_n - x_n-1) / 3 + 2 dt (4 v_n - v_n-1) / 9,
    Et on the measures' change from m_b = (4 m_n - m_n-1) / 3 (the measures' rate (3 m - 4 m_n + m_n-1) / (2 dt)); the
    step dissipates about 9 Et / (2 dt). Numerical damping ratio about (w dt)^3 / 4 (w dt 0.01: 2.5e-7, 0.1: 2.5e-4), the
    stiff local modes of the small cells (10-40x above what the cells can carry) die within a step. Contact at the end
    (k_contact cs 1, wt 1), the active set held within each solve. Start: the history of a constant acceleration a_0
    (x_-1 = x_0 - dt v_0 + dt^2 a_0 / 2, v_-1 = v_0 - dt a_0; free fall is then exact), the measures' m_-1 = m_0."""

    def __init__(self, b, dt, x0, v0, y0, f_ext, tol=1e-4, ref=None, al_max=50):
        from free_surface.skin_dynamics_check import predict_skin
        self.predict_skin = predict_skin
        st, W, S = b['st'], b['W'], b['S']
        self.b, self.st, self.W, self.S, self.prec, self.mass = b, st, W, S, b['prec'], b['mass']
        self.dt, self.f_ext, self.tol, self.al_max = dt, f_ext, tol, al_max
        self.ref = ref if ref is not None else float((9.8*self.mass).norm())
        W.visc_weight = 1.5
        W.total_mode()
        self.J = JointSolver(S, W, self.mass, b['kblocks'], b['vblocks'], dt, 9/(4*dt*dt), 1.5)
        if W.ct is not None:                                                       # the gap itself (as MidpointIntegrator)
            W.tcpar[0], W.tcpar[1], W.tcpar[2] = 1., 1., 0.
            W.tgoff.zero_()
        e, gx, _ = W.visc_ref(x0, y0) if W.vis is not None else W.grad(x0, y0)     # the force at the start
        self.a = (f_ext-gx)/self.mass[:, None]
        self.x, self.v = x0.clone(), v0.clone()
        self.xp, self.vp = x0-dt*v0+.5*dt*dt*self.a, v0-dt*self.a
        self.y = y0.clone()
        self.yp = predict_skin(S, self.xp, x0, x0, y0, y0)                         # the skin moving with that history
        self.m_n = measures(W) if W.vis is not None else None
        self.m_p = self.m_n
        if W.ct is not None:
            self.g_n = W.tgap.clone()
            W.tgoff.zero_()
            W.tcpar[0], W.tcpar[1], W.tcpar[2] = 1., 1., 1.
            W.tact.copy_((W.tlam > 0).to(W.tact.dtype))
        self.dissipated = self.contact_work = 0.

    def step(self):
        """one step; returns the solve's stats (aux at the step's end)"""
        W, dt, n = self.W, self.dt, self.W.n
        x, v, xp, vp = self.x, self.v, self.xp, self.vp
        x_t = (4*x-xp)/3+2*dt*(4*v-vp)/9
        xs = x+dt*v+.5*dt*dt*self.a
        ys = self.predict_skin(self.S, xs, x, xp, self.y, self.yp)
        if W.vis is not None:                                                      # the references m_b
            for k_, r_ in W.tref.items():
                r_.copy_((4*self.m_n[k_]-self.m_p[k_])/3)
        self.J.set_step(x, x_t, self.f_ext)
        z, s_ = self.J.solve_contact(torch.cat((xs, ys)), self.prec, self.ref, tol=self.tol, al_max=self.al_max)
        x1, y1 = z[:n], z[n:]
        v1 = (3*x1-4*x+xp)/(2*dt)
        self.a = (3*v1-4*v+vp)/(2*dt)
        self.xp, self.vp, self.x, self.v = x, v, x1, v1
        self.yp, self.y = self.y, y1
        ts = s_['aux'][3]
        if W.ct is not None:                                                       # the contact force's work lam (g_end - g_n):
            if s_['lam'] is not None:                                              # 0 in lasting contact, < 0 at a new one
                self.contact_work += float((s_['lam']*(s_['aux'][2]-self.g_n)).sum())
            self.g_n = s_['aux'][2].clone()
        if W.vis is not None:
            self.dissipated += 4.5*float(ts[7])/dt
            W.grad(x1, y1)                                                         # the measures at the step's end
            self.m_p, self.m_n = self.m_n, measures(W)
        s_['xm'], s_['ym'] = x1, y1
        return s_
