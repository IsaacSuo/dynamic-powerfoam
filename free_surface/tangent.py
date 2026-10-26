"""Forward (tangent) sensitivities of the joint implicit midpoint step (joint_step.MidpointIntegrator) to the material
parameters and the start's velocity (2026-10-25). Per step the midpoint z = (x_m, y_m) solves
  G(z) = grad Pi_0(z) - sum_A lam_a grad g_a(z) = 0,   c_a(z) = 2 g_a(z) - g_n,a = 0 on the active set A
(Pi_0: inertia, gravity, E, 2 Et / dt; the augmented Lagrangian converged with its set held, k_contact mode 1). For a
parameter theta (dots: d / d theta):
  H zdot - J^T lamdot = c_in M xdot_t - (dG/dm_n) mdot_n - dG/dtheta,    2 J zdot = gdot_n  (on A)
with H the solve's Hessian at z (the products of JointCG: one-sided differences of the gradient), J = grad g on A,
x_t = x_n + dt v_n / 2, c_in = 4 / dt^2; then
  xdot_n+1 = 2 xdot_m - xdot_n,  vdot_n+1 = 4 (xdot_m - xdot_n) / dt - vdot_n,  mdot_n+1 = 2 (dm/dz) zdot - mdot_n,
  gdot_n+1 = 2 (dg/dz) zdot - gdot_n (every contact cell; 0 on A),  ydot_n+1 = 2 ydot_m - ydot_n (the skin at integer times)
(m: the viscous references SkinWarp.tref, g: the end gaps k_contact's goff). The constraint by the linear augmented
Lagrangian: the products carry the contact term's 2 rho J^T J, so (H + 2 rho J^T J) zdot = rhs + J^T q with q = delta +
rho gdot_n, then delta <- delta - rho (2 J zdot - gdot_n) (at its end lamdot = delta), until the active cells' end-gap
rates are al_tol of the largest |zdot| (the same units; 10-25: relative to the largest gap rate the tangential v0
parameters, whose gap rates are noise, ran to al_max); J^T q is the contact term's gradient change between the
multipliers lam and lam + q (k_contact: grad Phi_c = -(lam - rho g) J on A, wt cs = 1). Each step's solve starts from
(start 'predict', default) the forward's own predictor in tangent form: xdot_n + dt vdot_n / 2 + dt^2 adot_n / 4 (adot =
(vdot_n+1 - vdot_n) / dt) and predict_skin of the tangents' midpoints (linear in its inputs); or ('last') from the last
step's zdot, or ('inertia') the predictor without adot (10-25: its start residual was 1-2 x smaller than with adot in most
of a drop). The parameters' CGs of each augmented-Lagrangian pass run at once in lockstep, one on W and one per product
copy (copies: build_model's make_copy, state by sync_state; each its own stream; 2026-10-24 four gradient graphs on four
streams gave x1.46 the throughput of one). The CG stops at |r| <= tol |b| (10-25: |b| is mostly the inertia c_in M xdot_t, and the tangents' error went
~linearly with tol, 0.2 % at 1e-6, 6-25 % at 1e-4 over 0.2 s; a stop relative to the start's residual has none in free
flight for v0, whose tangent there is the predictor up to the products' noise).
dG/dtheta: 'lnE' grad E + (2/dt) grad Et (every elastic and viscous coefficient is of degree 1 in E at fixed nu: mu,
lam, lam mu / (lam + 2 mu), eta = beta mu, lam_v = beta lam), 'lnbeta' (2/dt) grad Et, 'nu' the one-sided difference of
the gradients (without contact) with the material moved in place to nu + h (SkinWarp.set_material; or a second SkinWarp
W_nu built there), 'v0x' / 'v0y' / 'v0z' none (the
start's velocity: vdot_0 = e_k on every body). Term weights SkinWarp.ttw select the parts (device array: the graphs
stay). Directional derivatives (references, measures, gaps) by one-sided differences, step h relative to the
direction's largest entry, as the products."""
import torch
import warp as wp

from free_surface.joint_step import JointCG

NAMES = ('d', 'th', 'a', 'dl', 'L', 'P')


def measures_now(W):
    """the viscous measures of W's last gradient pass (joint_step.MidpointIntegrator.step's set)"""
    return dict(d=W.tbd.clone(), th=W.tbth.clone(), a=W.tba.clone(), dl=W.tdl.clone(), L=W.tlj.clone(), P=W.tR.transpose(1, 2)@W.tF)


def sync_state(dst, src):
    """a product copy's mutable state from W (both from build_model): the material (set_material's arrays and scalars),
    the terms' weights, the viscous references, the contact's multipliers, penalties, set, end gaps, mode and plane"""
    dst._kb_t.copy_(src._kb_t); dst.tn3.copy_(src.tn3); dst.tNL.copy_(src.tNL); dst.twv.copy_(src.twv); dst.tmatp.copy_(src.tmatp)
    if src.vis is not None:
        dst._NLv.copy_(src._NLv); dst.vis = dict(src.vis)
    for a in ('mu', 'lam', 'kap', 'eta', 'lamv', 'smu', 'slam', 'visc_weight'):
        setattr(dst, a, getattr(src, a))
    dst.ttw.copy_(src.ttw)
    for k, r in src.tref.items():
        dst.tref[k].copy_(r)
    if src.ct is not None:
        for a in ('tlam', 'trho', 'tcpar', 'tgoff', 'tact', 'tc0', 'tfpar', 'tlamt', 'trhot', 'tfst', 'twfix', 'tpn', 'tlamR'):
            getattr(dst, a).copy_(getattr(src, a))


def make_W_nu(args, dnu, out):
    """a SkinWarp of the same foam at nu + dnu (the viscosity's eta = beta mu', lam_v = beta lam' with it; no contact),
    for dG/dnu"""
    import numpy as np
    from experiments.pacnerf_dem_forward import build as dem_build
    from free_surface.boundary_skin import BoundaryTerms
    from free_surface.skin_contact_check import f64, dev
    from free_surface.skin_warp import SkinWarp
    cells = np.load(out/'dem_exact_single'/'cells.npz')
    gg = dict(np.load(out/'dem_exact_single'/'geometry.npz'))
    m = dem_build(gg, args.E, args.nu+dnu, dev, 'noslip', .25, .3, f64, kinematics='exact', nodalvol=True, nonlinear=False, ogden_alpha=2.)
    st = BoundaryTerms(m, gg, cells['points'], cells['radii'], dict(np.load(out/'boundary_skin.npz')), lever='c')
    W = SkinWarp(st)
    if args.beta > 0:
        W.set_viscosity(args.beta*W.mu, args.beta*(W.lam+2*W.mu/3), args.dt)
    W.visc_weight = 2.
    W.total_mode()
    W.prepare()
    return W


class Tangent:
    """the tangents of every parameter in params, carried along MidpointIntegrator I's steps: step() runs I.step() and
    then each parameter's linear solve. W_nu, dnu: make_W_nu's (needed for 'nu'). State per parameter: x, v (bodies),
    y (skin at integer times), m (references), g (end gaps)"""

    def __init__(self, I, params, W_nu=None, dnu=None, tol=1e-9, h=1e-8, al_tol=1e-9, al_max=30, maxit=4000, start='predict',
                 log_starts=False, copies=(), floor=True, seed=False):
        self.I, self.W, self.J = I, I.W, I.J
        if I.J.h != h:                                                             # the products' one-sided step, the forward's
            I.J.h, I.J._jcg = h, None                                              # too (its graph recaptured; 10-25, the friction
        # glue x 100 at step 100: h 1e-7 -> 1e-8 took the tangents' step errors 1e-4 -> 1e-5 (the truncation, near the
        # glue's stiffness; 1e-8 ~ sqrt(eps) balances it with the rounding)
        W = self.W
        self.params, self.W_nu, self.dnu = list(params), W_nu, dnu
        self.tol, self.h, self.al_tol, self.al_max, self.maxit = tol, h, al_tol, al_max, maxit
        self.start, self.log_starts = start, log_starts
        self.copies = list(copies)                                                 # product SkinWarps (build_model's make_copy)
        self.streams = [None]+[wp.Stream(wp.get_device()) for _ in self.copies]
        self._ccg = None
        self.floor, self.use_seed = floor, seed
        z = torch.zeros_like
        kc = len(W.tlam) if W.ct is not None else 0
        self.T = {}
        for p in self.params:
            t = dict(x=z(I.x), v=z(I.v), y=torch.zeros(W.npt, 3, dtype=I.x.dtype, device=I.x.device), m={k: z(r) for k, r in W.tref.items()},
                     g=torch.zeros(kc, dtype=I.x.dtype, device=I.x.device), zm=None, a=z(I.x))
            if p.startswith('v0'):
                t['v'][:, 'xyz'.index(p[2])] = 1.
            t['xm1'] = t['x']-.5*I.dt*t['v']; t['xm2'] = t['xm1']-I.dt*t['v']       # the integrator's start history, tangent
            t['ym1'] = I.predict_skin(I.S, t['xm1'], t['x'], t['x'], t['y'], t['y'])
            t['ym2'] = I.predict_skin(I.S, t['xm2'], t['xm1'], t['x'], t['ym1'], t['y'])
            if kc:                                                                 # the friction glue's force and its held points
                t['wd'], t['pd'], t['fd'] = (torch.zeros(kc, 3, dtype=I.x.dtype, device=I.x.device) for _ in range(3))   # (0 at the start)
            self.T[p] = t
        self.cbody = wp.to_torch(W.ct['body']).long() if kc else None
        self.log = []

    # ---------------------------------------------------------------- pieces
    def _grad_w(self, w, xm, ym, Wx=None):
        """the gradient (dx, dy) at (xm, ym) with term weights w (8,), the weights restored"""
        Wx = Wx or self.W
        keep = Wx.ttw.clone()
        Wx.ttw.copy_(w)
        _, gx, gy = Wx.grad(xm, ym)
        Wx.ttw.copy_(keep)
        return gx, gy

    def _w(self, terms):
        w = torch.zeros_like(self.W.ttw)
        for k, v in terms.items():
            w[k] = v
        return w

    def dG_dref(self, xm, ym, md, m_n):
        """(dG/dm_n) mdot: the viscous term's gradient with the references moved along mdot (it is linear in them)"""
        W, h = self.W, self.h
        s = max(float(v.abs().max()) for v in md.values())
        if s == 0:
            return torch.zeros_like(xm), torch.zeros_like(ym)
        w = self._w({7: float(self.W.ttw[7])})
        for k in NAMES:
            W.tref[k].copy_(m_n[k]+(h/s)*md[k])
        gx1, gy1 = self._grad_w(w, xm, ym)
        for k in NAMES:
            W.tref[k].copy_(m_n[k])
        gx0, gy0 = self._grad_w(w, xm, ym)
        return (gx1-gx0)*(s/h), (gy1-gy0)*(s/h)

    def dG_dtheta(self, p, xm, ym, m_n):
        W, t7 = self.W, float(self.W.ttw[7])
        if p == 'lnE':
            return self._grad_w(self._w({**{k: 1. for k in range(6)}, 7: t7}), xm, ym)
        if p == 'lnbeta':
            return self._grad_w(self._w({7: t7}), xm, ym)
        if p == 'nu':
            w = self._w({**{k: 1. for k in range(6)}, 7: t7})
            gx0, gy0 = self._grad_w(w, xm, ym)
            dnu = self.dnu or 1e-6
            if self.W_nu is not None:
                Wn = self.W_nu
                for k in NAMES:
                    Wn.tref[k].copy_(m_n[k])
                _, gx1, gy1 = Wn.grad(xm, ym)
            else:                                                              # in place
                nu0 = W.lam/(2*(W.lam+W.mu))
                W.set_material(nu=nu0+dnu)
                gx1, gy1 = self._grad_w(w, xm, ym)
                W.set_material(nu=nu0)
            return (gx1-gx0)/dnu, (gy1-gy0)/dnu
        return torch.zeros_like(xm), torch.zeros_like(ym)                         # v0*: only the start

    def contact_JT(self, xm, ym, q):
        """J^T q on A: -(grad Phi_c at lam + q - at lam)"""
        W = self.W
        w = self._w({6: 1.})
        lam0 = W.tlam.clone()
        gx0, gy0 = self._grad_w(w, xm, ym)
        W.tlam.copy_(lam0+q)
        gx1, gy1 = self._grad_w(w, xm, ym)
        W.tlam.copy_(lam0)
        return -(gx1-gx0), -(gy1-gy0)

    def friction_JT(self, xm, ym, dn, qt, dmu, dk, dw, dp):
        """the glue's generalised force change at fixed z (flat as contact_JT) when its bound's multipliers (R = mu lamR,
        the step's start ones) move by dn, its held force by qt, mu by dmu, its stiffness k by dk, the step's held material
        points by dw and their start positions by dp (the slip s = cs P_t (F w + sum cew y - pn) and, 10-25, the force's
        lever: dE/dF = -f (x) w): one-sided differences of the friction term's gradient (psi_R is not linear in them)"""
        W = self.W
        lsc = max(float(W.twfix.abs().max()), 1e-300)
        sc = max(float(dn.abs().max())/max(float(W.tlamR.abs().max()), 1e-300), float(qt.abs().max())/max(float(W.tlamt.abs().max()),
                 float(W.tfpar[0])*float(W.tlamR.abs().max()), 1e-300), abs(dmu)/max(float(W.tfpar[0]), 1e-300),
                 float((dk.abs()/W.trhot.clamp(min=1e-300)).max()), float(dw.abs().max())/lsc, float(dp.abs().max())/lsc)
        if sc == 0:
            return torch.zeros_like(xm), torch.zeros_like(ym)
        e = self.h/sc
        w = self._w({8: 1.})
        lam0, lt0, mu0, k0, w0, p0 = W.tlamR.clone(), W.tlamt.clone(), float(W.tfpar[0]), W.trhot.clone(), W.twfix.clone(), W.tpn.clone()
        gx0, gy0 = self._grad_w(w, xm, ym)
        W.tlamR.add_(dn, alpha=e); W.tlamt.add_(qt, alpha=e); W.tfpar[0] = mu0+e*dmu; W.trhot.add_(dk, alpha=e)
        W.twfix.add_(dw, alpha=e); W.tpn.add_(dp, alpha=e)
        gx1, gy1 = self._grad_w(w, xm, ym)
        W.tlamR.copy_(lam0); W.tlamt.copy_(lt0); W.tfpar[0] = mu0; W.trhot.copy_(k0); W.twfix.copy_(w0); W.tpn.copy_(p0)
        return -(gx1-gx0)/e, -(gy1-gy0)/e

    def along(self, xm, ym, zd, meas0, gap0, slip0=None):
        """the measures', the gaps' and the friction's slips' derivatives along zd (one-sided)"""
        W, h, n = self.W, self.h, self.W.n
        s = float(zd.abs().max())
        if s == 0:
            return ({k: torch.zeros_like(v) for k, v in meas0.items()}, torch.zeros_like(gap0) if gap0 is not None else None,
                    torch.zeros_like(slip0) if slip0 is not None else None)
        W.grad(xm+(h/s)*zd[:n], ym+(h/s)*zd[n:])
        m1 = measures_now(W)
        dm = {k: (m1[k]-meas0[k])*(s/h) for k in meas0}
        dg = (W.tgap-gap0)*(s/h) if gap0 is not None else None
        dsl = (W.tslip-slip0)*(s/h) if slip0 is not None else None
        return dm, dg, dsl

    def _cgs(self, prec):
        """the JointCGs: on W (J's own) and one per product copy, rebuilt when the skin's WarpPrec changes"""
        J = self.J
        if J._jcg is None or J._jcg.wprec is not prec.wprec:
            J._jcg = JointCG(self.W, prec.wprec, J.mass, J.c_in, J.h)
        if self._ccg is None or (self._ccg and self._ccg[0].wprec is not prec.wprec):
            self._ccg = [JointCG(Wc, prec.wprec, J.mass, J.c_in, J.h) for Wc in self.copies]
        return [J._jcg]+self._ccg

    def pcg_many(self, z, g0, prec, bs, x0s, record=False):
        """H zd = b for each b of bs (flat (n + npt) x 3) by preconditioned CG on JointCG graphs, 1 + len(copies) at once
        in lockstep (each its own SkinWarp and stream, one synchronize per iteration; a slot whose system is done takes
        the next); x0s: starts, their residuals from one product each (J.hvp). Stop: |r| <= tol max(|b|, I.ref) (with
        floor; I.ref the forward's force scale, the weight: a force the forward does not resolve is not solved for, e.g.
        the material parameters' right sides in free flight, the forward's leftover residual). Per system (sol,
        iterations, negative curvature, |r| / |b|, start |r| / |b|); record (one system): also its directions d and H d"""
        J, n = self.J, self.W.n
        cgs = self._cgs(prec)
        fl = self.I.ref if self.floor else 0.
        out, r0, rst, k, neg, rr = [None]*len(bs), {}, {}, {}, {}, {}
        basis = ([], [])
        queue, slot = list(range(len(bs))), [None]*len(cgs)

        def setup(j, i):
            cg, b, x0 = cgs[j], bs[i], x0s[i]
            T = cg.t
            r = b.reshape(1, -1).clone()
            if x0 is not None and float(x0.abs().max()) > 0:
                r = r-J.hvp(None, z, g0)(x0.reshape(1, -1))
            T['zx'].copy_(z[:n]); T['zy'].copy_(z[n:]); T['gx'].copy_(g0.view(-1, 3)[:n]); T['gy'].copy_(g0.view(-1, 3)[n:])
            cg.tPi.copy_(prec.Pi); cg.txt.copy_(J.x_t); cg.tf.copy_(J.f_ext)
            zz = prec(r)
            rv, zv = r.view(-1, 3), zz.view(-1, 3)
            T['rx'].copy_(rv[:n]); T['ry'].copy_(rv[n:]); T['dx'].copy_(zv[:n]); T['dy'].copy_(zv[n:])
            cg.tsc[5] = float((r*zz).sum())
            for s_ in 'xy':
                T['p'+s_].zero_(); T['Hp'+s_].zero_()
            r0[i], rst[i] = float(b.norm()), float(r.norm())
            k[i], neg[i], rr[i] = 0, 0, 0.
            slot[j] = i

        def fill():
            took = False
            for j in range(len(cgs)):
                if slot[j] is None and queue:
                    setup(j, queue.pop(0)); took = True
            if took:
                torch.cuda.synchronize()                                           # (the starts, before the other streams)
        fill()
        while any(i is not None for i in slot):
            for j, i in enumerate(slot):
                if i is not None:
                    cgs[j].launch(self.streams[j])
            wp.synchronize()
            for j, i in enumerate(slot):
                if i is None:
                    continue
                T = cgs[j].t
                dHd, rr[i], _ = cgs[j].read(); k[i] += 1
                if record:
                    basis[0].append(torch.cat((T['d_oldx'], T['d_oldy'])).reshape(-1).clone())
                    basis[1].append(torch.cat((T['Hdx'], T['Hdy'])).reshape(-1).clone())
                if dHd <= 0:
                    neg[i] = 1
                elif rr[i]**.5 > self.tol*max(r0[i], fl) and k[i] < self.maxit:
                    continue
                sol = torch.cat((T['px'], T['py'])).clone()
                if x0s[i] is not None:
                    sol = sol+x0s[i].view(-1, 3)
                out[i] = (sol, k[i], neg[i], rr[i]**.5/max(r0[i], 1e-300), rst[i]/max(r0[i], 1e-300))
                slot[j] = None
            fill()
        if record:
            return out, (torch.stack(basis[0]), torch.stack(basis[1]))
        return out

    def seed(self, basis, z, g0, b, x0):
        """x0 + D c, the Galerkin correction of the start on another system's CG directions D (H D stored): (D^T H D) c =
        D^T (b - H x0) (symmetrised, Jacobi-scaled; directions below its rounding dropped)"""
        D, HD = basis
        r = b.reshape(-1).clone()
        if x0 is not None and float(x0.abs().max()) > 0:
            r = r-self.J.hvp(None, z, g0)(x0.reshape(1, -1)).reshape(-1)
        A = D@HD.T
        A = .5*(A+A.T)
        sc = A.diagonal().clamp(min=1e-300).rsqrt()
        w, V = torch.linalg.eigh(sc[:, None]*A*sc[None, :])
        keep = w > w.max()*w.shape[0]*torch.finfo(w.dtype).eps
        c = sc*(V[:, keep]@((V[:, keep].T@(sc*(D@r)))/w[keep]))
        return (x0.reshape(-1) if x0 is not None else 0.)+D.T@c

    def pcg(self, z, g0, prec, b, x0=None):
        return self.pcg_many(z, g0, prec, [b], [x0])[0]

    def _start(self, T, dt, adot):
        I = self.I
        xs = T['x']+.5*dt*T['v']+(.25*dt*dt*T['a'] if adot else 0.)
        return torch.cat((xs, I.predict_skin(I.S, xs, T['xm1'], T['xm2'], T['ym1'], T['ym2'])))

    # ---------------------------------------------------------------- one step
    def step(self):
        I, W, J = self.I, self.W, self.J
        dt, n = I.dt, W.n
        con = W.ct is not None
        x_n, v_n = I.x.clone(), I.v.clone()
        m_n = {k: r.clone() for k, r in W.tref.items()}
        g_n = W.tgoff.clone() if con else None
        s_ = I.step()
        post = dict(ref={k: r.clone() for k, r in W.tref.items()}, goff=W.tgoff.clone() if con else None,
                    lam=W.tlam.clone() if con else None, act=W.tact.clone() if con else None,
                    lamt=W.tlamt.clone() if con else None, fst=W.tfst.clone() if con else None)
        xm, ym = s_['xm'], s_['ym']
        settled = True
        fr = con and float(W.tfpar[0]) > 0
        if con:                                                                    # the multipliers and set of the last solve
            lam_used = s_.get('lam_used')
            if lam_used is None:
                lam_used = W.tlam.clone()
            settled = s_['al'] < I.al_max
            W.tlam.copy_(lam_used); W.tgoff.copy_(g_n)
            if fr:                                                                 # the friction's (fst: the side marks)
                W.tlamt.copy_(s_['lamt_used']); W.tfst.copy_(s_['fst_used'])
        for k in NAMES:
            W.tref[k].copy_(m_n[k])
        z = torch.cat((xm, ym))
        J.set_step(x_n, x_n+.5*dt*v_n, I.f_ext)
        _, g0, aux = J.eg(None, z)
        meas0 = measures_now(W)
        gap0 = W.tgap.clone() if con else None
        if fr:                                                                     # the friction at the solution: slip s, the
            slip0, Fc = aux[5], W.tF[self.cbody].clone()                           # cells' F at the midpoint, q = lamt - rho_t s,
            mu = float(W.tfpar[0])                                                 # R = mu lamR (the step's start multipliers,
            R = mu*W.tlamR.clamp(min=0)                                            # held through its passes); the side as the solve's
            qf = W.tlamt-W.trhot[:, None]*slip0
            qn = qf.norm(dim=1)
            qh = qf/qn.clamp(min=1e-300)[:, None]
            stickf = qn <= R
            nv = torch.tensor(W.ct['normal'], dtype=xm.dtype, device=xm.device)
            Pt = lambda v_: v_-(v_@nv)[:, None]*nv
            cs = float(W.tcpar[0])
            kfac = {'lnE': 1., 'nu': -1/(1+float(W.lam/(2*(W.lam+W.mu))))}         # dk / k: k = mu_s A / r, mu_s = E / (2 (1 + nu))
        prec = J.prec_at(I.prec, None, z, aux)
        act = (W.tact > 0) if con else None
        anyact = con and bool(act.any())
        rho = W.trho.clone() if con else None
        info = dict(step_residual=float(g0.norm())/I.ref, settled=settled, active=int(act.sum()) if con else 0, params={})
        S = {}
        for p in self.params:                                                      # the right sides (the state as the solve's)
            T = self.T[p]
            dx_m, dy_m = self.dG_dref(xm, ym, T['m'], m_n)
            tx, ty = self.dG_dtheta(p, xm, ym, m_n)
            rx = J.c_in*I.mass[:, None]*(T['x']+.5*dt*T['v'])-dx_m-tx
            ry = -dy_m-ty
            zd = T['zm'] if self.start == 'last' else self._start(T, dt, self.start == 'predict')
            S[p] = dict(rhs=torch.cat((rx, ry)), delta=torch.zeros_like(T['g']) if con else None, zd=zd, its=0, k=0, neg=0,
                        gend=None, dm=None, starts={}, done=False)
            if fr:                                                                 # the slip rate's part not along zdot (the held
                S[p].update(dmu=mu if p == 'lnmu' else 0., dk=W.trhot*kfac.get(p, 0.), dsl=None,            # point's material and start
                            et=cs*Pt(torch.einsum('cij,cj->ci', Fc, T['wd'])-T['pd']),       # rates: F_mid wdot - pdot_n); k = mu_s A / r;
                            dR=(T['lamdot'] if 'lamdot' in T else torch.zeros_like(T['g']))*(W.tlamR > 0))   # the bound's multipliers'
            # rate: the step's start ones, the last step's end lamdot
        for Wc in self.copies:
            sync_state(Wc, W)
        for its in range(1, self.al_max+1):                                        # the augmented Lagrangian's passes, all
            live = [p for p in self.params if not S[p]['done']]                    # parameters' CGs of a pass at once
            if not live:
                break
            bs = []
            for p in live:
                T, st_ = self.T[p], S[p]
                b = st_['rhs']
                if anyact:
                    q = torch.where(act, st_['delta']+rho*T['g'], torch.zeros_like(st_['delta']))
                    cx, cy = self.contact_JT(xm, ym, q)
                    b = b+torch.cat((cx, cy))
                    if fr:                                                         # the glue's: its bound's multipliers by dR, its
                        cx, cy = self.friction_JT(xm, ym, st_['dR'],                                                         # held force,
                                                  T['fd']*(W.tlamR > 0)[:, None], st_['dmu'], st_['dk'], T['wd'], T['pd'])   # mu, k, the
                        b = b+torch.cat((cx, cy))                                  # held points (slip and lever)
                if its == 1 and self.log_starts:                                   # the first pass's start residuals / |b|
                    bn = max(float(b.norm()), 1e-300)
                    for nm, s0 in (('last', T['zm']), ('predict', self._start(T, dt, True)), ('inertia', self._start(T, dt, False))):
                        if s0 is None or float(s0.abs().max()) == 0:
                            st_['starts'][nm] = 1.; continue
                        st_['starts'][nm] = float((b.reshape(1, -1)-J.hvp(None, z, g0)(s0.reshape(1, -1))).norm())/bn
                bs.append(b)
            x0s = [S[p]['zd'] for p in live]
            if self.use_seed and len(live) > 1:                                    # the first alone, its directions seed the rest
                r1, basis = self.pcg_many(z, g0, prec, bs[:1], x0s[:1], record=True)
                x0s = x0s[:1]+[self.seed(basis, z, g0, b_, x_).view(-1, 3) for b_, x_ in zip(bs[1:], x0s[1:])]
                del basis
                res = r1+self.pcg_many(z, g0, prec, bs[1:], x0s[1:])
            else:
                res = self.pcg_many(z, g0, prec, bs, x0s)
            for p, (zd, k, neg, rel, r_st) in zip(live, res):
                T, st_ = self.T[p], S[p]
                st_.update(zd=zd, its=its); st_['k'] += k; st_['neg'] += neg
                if its == 1:
                    st_['starts']['used'] = r_st
                    st_['starts']['b_ref'] = float(bs[live.index(p)].norm())/I.ref     # |b| / the weight
                if not con:
                    st_['done'] = True
                    continue
                st_['dm'], dg, dsl = self.along(xm, ym, zd, meas0, gap0, slip0 if fr else None)
                st_['gend'] = gend = dg-T['g']                                     # 2 J zdot - gdot_n
                if not anyact:
                    st_['done'] = True
                    continue
                zmax = max(float(zd.abs().max()), 1e-300)
                err = float(gend[act].abs().max())/zmax
                st_['delta'] = torch.where(act, st_['delta']-rho*gend, torch.zeros_like(st_['delta']))
                st_['dsl'] = dsl
                st_['done'] = err <= self.al_tol
        for p in self.params:                                                      # carried to the step's end
            T, st_ = self.T[p], S[p]
            zd = st_['zd']
            dm = st_['dm'] if con else self.along(xm, ym, zd, meas0, gap0)[0]
            T['zm'] = zd.clone()
            xd_m, yd_m = zd[:n], zd[n:]
            xd_n, vd_n = T['x'], T['v']
            T['x'] = 2*xd_m-xd_n
            T['v'] = 4*(xd_m-xd_n)/dt-vd_n
            T['y'] = 2*yd_m-T['y']
            T['a'] = (T['v']-vd_n)/dt
            T['xm2'], T['ym2'], T['xm1'], T['ym1'] = T['xm1'], T['ym1'], xd_m.clone(), yd_m.clone()
            for k in NAMES:
                T['m'][k] = 2*dm[k]-T['m'][k]
            if con:
                T['g'] = st_['gend'].clone()
                T['lamdot'] = torch.where(act, st_['delta'], torch.zeros_like(st_['delta']))
            gend = st_['gend']
            info['params'][p] = dict(al_iterations=st_['its'], cg_iterations=st_['k'], negative=st_['neg'], starts=st_['starts'],
                                     gap_rel=float(gend[act].abs().max())/max(float(zd.abs().max()), 1e-300) if anyact else 0.)
        if fr:                                                                     # the glue's force at the step's end (the
            on = post['act'] > 0                                                   # forward: f on the next active set): the
            for p in self.params:                                                  # projection linearised, stuck qdot, sliding
                T, st_ = self.T[p], S[p]                                           # Rdot qhat + R / |q| (I - qhat qhat^T) qdot,
                dsl = st_['dsl'] if st_['dsl'] is not None else torch.zeros_like(slip0)   # qdot = fdot_n - kdot s - k sdot
                qd = T['fd']-st_['dk'][:, None]*slip0-W.trhot[:, None]*(dsl+st_['et'])
                Rd = mu*st_['dR']+st_['dmu']*W.tlamR.clamp(min=0)
                sl = Rd[:, None]*qh+(R/qn.clamp(min=1e-300))[:, None]*(qd-qh*(qh*qd).sum(1, keepdim=True))
                T['fd'] = torch.where(stickf[:, None], qd, sl)*on[:, None]
            for p in self.params:                                                  # the next step's held points' rates: each
                T = self.T[p]                                                      # cell's lowest point at the integer time and
                sc = max(float(T['x'].abs().max()), float(T['y'].abs().max()))
                if sc == 0:
                    T['wd'].zero_(); T['pd'].zero_()
                    continue
                e = self.h/sc
                W.grad(I.x+e*T['x'], I.y_int+e*T['y'])
                T['wd'], T['pd'] = (W.tXs-I.wfix)/e, (W.tcpos-I.pn)/e
        for k in NAMES:                                                            # back to the step's end
            W.tref[k].copy_(post['ref'][k])
        if con:
            W.tgoff.copy_(post['goff']); W.tlam.copy_(post['lam']); W.tact.copy_(post['act'])
            W.tlamt.copy_(post['lamt']); W.tfst.copy_(post['fst'])
        self.log.append(info)
        return s_, info
