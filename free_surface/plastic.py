"""Plasticity on the skin model (2026-10-26, first version): each cell keeps a plastic (permanent) deformation, a log
strain eps_p (symmetric, traceless: the volume never yields), and the elastic terms are measured from the references
it implies (SkinWarp.plas):
  o1, o2   R^T F - P_p, P_p = exp(eps_p) the cell's plastic stretch (its body frame)
  bonds    d - dp, dp = (P_i - I) r_i - (P_j - I) r_j: each half of the bond (cell centre to the wall's contact point,
           r in the body frame) stretched by its own cell's P_p. With one P_p everywhere the body at x = P_p X is free of
           stress; between cells of different P_p what does not fit is the residual stress
  bonds    theta - thp, thp the rotation between the two cells in the stress-free state of a linear plastic field (from
           its fitted gradient, thp(), with branch()'s second-order dp): off by default, the cells' plastic strain taken
           as piecewise constant (10-26, Plasticity.__init__)
Not done: the skin terms' references (0: their deviations vanish under a uniform flow).
Yield (von Mises, as PAC-NeRF's MPM), per cell: the elastic log strain eps_e = log U - eps_p (U = R^T F of F_B, eps_e
coaxial with eps_p assumed), |2 mu dev eps_e| <= sqrt(2/3) sigma_y; beyond it the radial return eps_p += (1 -
sqrt(2/3) sigma_y / |2 mu dev eps_e|) dev eps_e. Two ways in time:
  update(x, y)  after the step (solved on the old references): 10-26 on the part, it dissipates the overshoot's
                elastic energy, 1/2 E de^2 a step beyond the plastic work: at dt 1/1200 (strain 0.0075 a step, yield
                0.01) 22-26 % too little permanent strain, 1/4800 still 6-11 %
  step(I)       within the step (midpoint rule): eps_n+1 the return of the step's end strain from eps_n, the step
                solved on the references of the mean eps_n+1/2 (the midpoint's stress then on the yield surface), by
                passes until eps_n+1 settles; the dissipation is the references' two switches (at the start eps_n ->
                eps_n+1/2, at the end eps_n+1/2 -> eps_n+1), sigma_y de exactly in 1-d
The elastic energy at the same state before less after a switch is the plastic dissipation.
10-26, the yield on the model's own stress (criterion 'virial', step_virial): the F_B stress above and the stress the
terms on the references carry, -(1/V) dE/deps_p, differ by 20-35 % per cell in bending (plastic_virial_check.py); the
step is min over z and delta of Pi(z; eps_n + delta) + sum sigma^ V |delta| (sigma^ = sqrt(2/3) sigma_y), which puts
the latter on the yield surface exactly; solved by blocks (z: the step's solve; delta at fixed z: solve_delta, E
being quadratic in P_p there)."""
import math

import torch

from free_surface.skin_contact_check import f64, dev


def sym_fun(A, f):
    """f of symmetric matrices (n, 3, 3) by eigenvalues"""
    w, V = torch.linalg.eigh(.5*(A+A.transpose(-1, -2)))
    return V@torch.diag_embed(f(w))@V.transpose(-1, -2)


def dev_part(A):
    return A-torch.diagonal(A, dim1=-2, dim2=-1).sum(-1)[..., None, None]/3*torch.eye(3, dtype=A.dtype, device=A.device)


def exp_dexp(eps):
    """exp of symmetric matrices (n, 3, 3) and what its derivative needs: the eigenvectors V and the divided
    differences G (Daleckii-Krein: Dexp(eps)[d] = V (G o V^T d V) V^T, self-adjoint)"""
    lam, V = torch.linalg.eigh(eps)
    el = torch.exp(lam)
    dl = lam[:, :, None]-lam[:, None, :]
    close = dl.abs() <= 1e-9
    G = torch.where(close, .5*(el[:, :, None]+el[:, None, :]), (el[:, :, None]-el[:, None, :])/torch.where(close, torch.ones_like(dl), dl))
    return V@torch.diag_embed(el)@V.transpose(1, 2), V, G


class Plasticity:
    """W: SkinWarp, st: its SkinTerms, sigma_y: the yield stress per unit density (as W.mu); criterion 'FB' (2 mu dev
    log U of F_B, step_fb) or 'virial' (the yield on -(1/V) dE/deps_p, the stress the model's own terms carry,
    step_virial). 10-26: the two differ by 20-35 % per cell in bending (plastic_virial_check.py), but 'virial' is not
    well posed here: the energy's curvature in all cells' plastic strains has soft modes (smallest 0.016, 1 % below
    0.19 of the continuum's 2 mu V) alternating between neighbours, and with every yielding cell's driving force on the
    surface they took the plastic flow (the part: single cells to |eps_p| 3.9 at body strain ~0.1, then exp overflowed).
    'FB' ties each cell's plastic strain to its own fitted deformation: the default"""

    def __init__(self, W, st, sigma_y, criterion='FB'):
        self.W, self.sigma_y, self.criterion = W, float(sigma_y), criterion
        self.bi, self.bj = st.i.long().to(dev), st.j.long().to(dev)
        self.ri, self.rj = st.model.ri.to(device=dev, dtype=f64), st.model.rj.to(device=dev, dtype=f64)
        self.n0 = st.model.n0.to(device=dev, dtype=f64)
        self.cb, self.cr, self.cN = st.c_body.long(), st.c_r.to(f64), st.c_N.to(f64)
        self.eb, self.ep, self.ed, self.ec = st.e_body.long(), st.e_pid.long(), st.e_d.to(f64), st.e_c.to(f64)
        self.V = st.V.to(f64)
        # the plastic strain's gradient per cell (thp): least squares over its bonds, sum wb (eps_j - eps_i) dX^T =
        # (grad eps) sum wb dX dX^T (wb = A / 2L and dX = X_j - X_i as the model's curvature fit)
        self.dX, self.wb = st.dX.to(device=dev, dtype=f64), st.wb.to(device=dev, dtype=f64)
        Mb = self.wb[:, None, None]*self.dX[:, :, None]*self.dX[:, None, :]
        M = torch.zeros(W.n, 3, 3, dtype=f64, device=dev).index_add(0, self.bi, Mb).index_add(0, self.bj, Mb)
        self.Minv = torch.linalg.pinv(M)
        # 10-26, the user: piecewise constant ("允许误差吧。就用分片常数"): each cell's plastic strain uniform in it, so a
        # bond's references come from its two cells only (dp = eps_i r_i - eps_j r_j, thp 0: uniform cells do not curve;
        # the misfit at the walls is the residual stress), local as the F_B return is: the returns dissipate (never
        # negative on the part). The linear reconstruction (rotation thp, branch gradient: exact for smooth fields, the
        # part's smooth plastic bending springs back 1.2 % against 4.6 %) reads the neighbours' strains through the
        # fitted gradient: a cell's return then moves its neighbours' bonds regardless of its own yield (dissipation
        # -2.2e-8, 8 % of the largest step's) and the gradient of cell-to-cell scatter (scatter / cell size) times r^2
        # amplifies it (max |eps_p| 0.28 -> 0.63, passes +44 %). Off; the error of the constant cells should fall as
        # (cell / thickness)^2 (not measured)
        self.rotation = False                                                      # (thp from the gradient; False: 0)
        self.gradient = False                                                      # (dp's second-order branch part; False: off)
        self.merged = True                                                         # (step: the return inside the step's passes)
        rr = self.ri.norm(dim=1)+self.rj.norm(dim=1)                                # the bonds' and o1's curvature in P,
        self._Lb_i, self._Lb_j = self.ri.norm(dim=1)*rr, self.rj.norm(dim=1)*rr      # Gershgorin per cell (times k, mu)
        self._Lo = torch.zeros(W.n, dtype=f64, device=dev).index_add(0, self.cb, 2*self.cr*torch.linalg.eigvalsh(self.cN)[:, -1])
        self.eps = torch.zeros(W.n, 3, 3, dtype=f64, device=dev)
        self.eps_prev = self.eps.clone()
        self.dissipated = 0.
        self.apply()

    def grad_eps(self, eps):
        """the plastic strain's gradient per cell, G[n, a, b, k] = d_k eps_ab (least squares over the cell's bonds)"""
        de = eps[self.bj]-eps[self.bi]                                               # (nb, 3, 3)
        rhs = self.wb[:, None, None, None]*de[:, :, :, None]*self.dX[:, None, None, :]
        R = torch.zeros(self.W.n, 3, 3, 3, dtype=f64, device=dev).index_add(0, self.bi, rhs).index_add(0, self.bj, rhs)
        return torch.einsum('nabk,nkl->nabl', R, self.Minv)

    def thp(self, G):
        """the bonds' plastic rotation (10-26): in the stress-free state of a plastic strain field the cells' rotations
        differ by omega_j - omega_i = (d omega) dX, d_k omega_ab = d_b eps_ak - d_a eps_bk (small strain), the gradient
        the two cells' mean; thp = axial(omega_j - omega_i) (theta = log(q_i* q_j) ~ phi_j - phi_i)"""
        Gb = .5*(G[self.bi]+G[self.bj])
        T = torch.einsum('eakb,ek->eab', Gb, self.dX)                                # sum_k d_b eps_ak dX_k
        dw = T-T.transpose(1, 2)                                                     # omega_j - omega_i
        return torch.stack((dw[:, 2, 1], dw[:, 0, 2], dw[:, 1, 0]), 1)

    def branch(self, G, r, idx):
        """a half-bond's second-order part (10-26): along the branch r from the cell's centre to the contact point the
        plastic displacement gradient H = eps + omega changes, int_0^1 H(s r) r ds = H r + 1/2 (d_k H_ab) r_b r_k with
        d_k H_ab = d_k d_b u_a = d_k eps_ab + d_b eps_ak - d_a eps_bk; the omega r parts cancel between the two halves
        (the cells turn), the rest: sum_bk d_k eps_ab r_b r_k - 1/2 sum_bk d_a eps_bk r_b r_k"""
        Gi = G[idx]
        return torch.einsum('eabk,eb,ek->ea', Gi, r, r)-.5*torch.einsum('ebka,eb,ek->ea', Gi, r, r)

    def apply(self, eps=None, P=None):
        """the references from eps_p (default the state's), or from the plastic stretch P itself (thp then kept)"""
        W, I3 = self.W, torch.eye(3, dtype=f64, device=dev)
        corr = 0.
        if P is None:
            e_ = self.eps if eps is None else eps
            P = sym_fun(e_, torch.exp)
            G = self.grad_eps(e_) if (self.rotation or self.gradient) else None
            if self.rotation:
                W.tplas['thp'].copy_(self.thp(G))
            else:
                W.tplas['thp'].zero_()
            if self.gradient:
                corr = self.branch(G, self.ri, self.bi)-self.branch(G, self.rj, self.bj)
        W.tplas['Pp'].copy_(P)
        W.tplas['dp'].copy_(torch.einsum('eab,eb->ea', P[self.bi]-I3, self.ri)-torch.einsum('eab,eb->ea', P[self.bj]-I3, self.rj)+corr)

    def elastic(self, x, y):
        """the elastic energy (terms 0-5) at (x, y); W.tF, W.tR there"""
        W = self.W
        tw = W.ttw.clone()
        W.ttw.zero_(); W.ttw[:6] = 1.
        e = float(W.grad(x, y)[0])
        W.ttw.copy_(tw)
        return e

    def elastic_grad(self, x, y):
        """the elastic terms' gradient at (x, y) (bodies and skin, copies); W's buffers there"""
        W = self.W
        tw = W.ttw.clone()
        W.ttw.zero_(); W.ttw[:6] = 1.
        _, gx, gy = W.grad(x, y)
        gx, gy = gx.clone(), gy.clone()
        W.ttw.copy_(tw)
        return gx, gy

    def dE_dP(self, P, x, y):
        """dE/dP_p per cell (n, 3, 3) of the terms on the plastic references at fixed (x, y) (W's buffers from a grad
        there: bond d, F, R; d, F, R do not depend on P_p): E is quadratic in P_p there (bonds 1/2 k |d - dp|^2 with dp
        linear in P, o1 quadratic, o2 linear)"""
        W, I3 = self.W, torch.eye(3, dtype=f64, device=dev)
        G = torch.zeros_like(P)
        dp = torch.einsum('eab,eb->ea', P[self.bi]-I3, self.ri)-torch.einsum('eab,eb->ea', P[self.bj]-I3, self.rj)
        u, nn, k = W.tbd-dp, self.n0, W._kb_t
        un = (u*nn).sum(1, keepdim=True)
        g = k[:, :1]*un*nn+k[:, 1:2]*(u-un*nn)                                     # dE/dd
        G.index_add_(0, self.bi, -torch.einsum('ea,eb->eab', g, self.ri))
        G.index_add_(0, self.bj, torch.einsum('ea,eb->eab', g, self.rj))
        F, R = W.tF, W.tR
        cb, eb = self.cb, self.eb
        G.index_add_(0, cb, -2*W.mu*self.cr[:, None, None]*R[cb].transpose(1, 2)@(F[cb]-R[cb]@P[cb])@self.cN)
        daff = y[self.ep]-x[eb]-torch.einsum('eab,eb->ea', F[eb], self.ed)
        G.index_add_(0, eb, -2*W.mu*torch.einsum('ea,eb->eab', torch.einsum('eba,eb->ea', R[eb], daff), self.ec))
        return G

    def dE_deps(self, eps, x, y):
        """dE/deps_p per cell, deviatoric (the volume never yields), and P_p = exp(eps_p)"""
        P, V, Gd = exp_dexp(eps)
        S = self.dE_dP(P, x, y)
        S = .5*(S+S.transpose(1, 2))
        return dev_part(V@(Gd*(V.transpose(1, 2)@S@V))@V.transpose(1, 2)), P

    def lin(self, eps_n):
        """the plastic stretch linearised at eps_n: P(delta) = P_n + Dexp(eps_n)[delta] (its pieces)"""
        return exp_dexp(eps_n)

    def grad_lin(self, delta, lin, x, y):
        """d/ddelta of E on P(delta) (deviatoric) and P(delta)"""
        Pn, V, Gd = lin
        Vt = V.transpose(1, 2)
        P = Pn+V@(Gd*(Vt@delta@V))@Vt
        S = self.dE_dP(P, x, y)
        S = .5*(S+S.transpose(1, 2))
        return dev_part(V@(Gd*(Vt@S@V))@Vt), P

    def solve_delta(self, eps_n, delta0, x, y, tol, maxit=2000):
        """at fixed (x, y): delta = argmin E(P(delta)) + sum sigma^ V |delta| (sigma^ = sqrt(2/3) sigma_y; P(delta) =
        exp(eps_n) + Dexp(eps_n)[delta], the plastic stretch linearised at the step's start: the midpoint's references,
        the step's increment 2 delta; E is then exactly quadratic in delta), by accelerated proximal gradient (FISTA),
        per cell the step 1 / L, L Gershgorin's bound of that quadratic's curvature (the bonds' k |r_i| (|r_i| + |r_j|),
        o1's 2 mu r max eig N, times Dexp's largest divided difference squared), the proximal map the shrink of |delta|
        by sigma^ V / L: a radial return on the cell's own driving force -dE/ddelta. (10-26: on exp itself, a strained
        part's first iterates reached |delta| ~ 1, the exp's curvature outgrew L and it diverged.) Stops when the gradient
        mapping L |delta - y| is within tol sigma^ V in every cell (the yield's stationarity to tol). Returns delta,
        P(delta), iterations, residual"""
        W = self.W
        thr = math.sqrt(2/3)*self.sigma_y*self.V
        kmax = W._kb_t[:, :2].max(1).values
        Lp = torch.zeros(W.n, dtype=f64, device=dev).index_add(0, self.bi, kmax*self._Lb_i).index_add(0, self.bj, kmax*self._Lb_j)+W.mu*self._Lo
        lin = self.lin(eps_n)
        L = Lp*lin[2].flatten(1).max(1).values**2
        x_, xp, t = delta0.clone(), delta0.clone(), 1.
        for it in range(1, maxit+1):
            t1 = .5*(1+math.sqrt(1+4*t*t))
            yv = x_+((t-1)/t1)*(x_-xp)
            g, _ = self.grad_lin(yv, lin, x, y)
            if it == 1 and not bool(torch.isfinite(g).all()):
                bad = (~torch.isfinite(g.flatten(1))).any(1)
                raise FloatingPointError('solve_delta: the driving force not finite in %d cells (first %s) at the start; L finite %s, min %.3e; '
                                         'W.tbd finite %s, F %s, R %s; delta0 finite %s; lin finite %s' % (
                                             int(bad.sum()), torch.nonzero(bad)[:5, 0].tolist(), bool(torch.isfinite(L).all()), float(L.min()),
                                             bool(torch.isfinite(W.tbd).all()), bool(torch.isfinite(W.tF).all()), bool(torch.isfinite(W.tR).all()),
                                             bool(torch.isfinite(delta0).all()), all(bool(torch.isfinite(a_).all()) for a_ in lin)))
            w = yv-g/L[:, None, None]
            nw = w.flatten(1).norm(dim=1)
            xn = (1-(thr/L)/nw.clamp(min=1e-300)).clamp(min=0)[:, None, None]*w
            res = float((L*(xn-yv).flatten(1).norm(dim=1)/thr).max())
            xp, x_, t = x_, xn, t1
            if res <= tol:
                break
        return x_, self.grad_lin(x_, lin, x, y)[1], it, res

    def yield_check(self, eps_n, delta, x, y):
        """the stationarity at (x, y) of delta from eps_n (the linearised P): yielding cells (delta != 0) |dev tau| /
        (sigma^ V) - 1 and the cosine of tau with delta, elastic cells their largest |dev tau| / (sigma^ V)"""
        g, _ = self.grad_lin(delta, self.lin(eps_n), x, y)
        thr = math.sqrt(2/3)*self.sigma_y*self.V
        tn = g.flatten(1).norm(dim=1)/thr
        y_ = delta.flatten(1).norm(dim=1) > 0
        cos = -(g*delta).flatten(1).sum(1)/(g.flatten(1).norm(dim=1)*delta.flatten(1).norm(dim=1)).clamp(min=1e-300)
        return dict(yielding=int(y_.sum()), surf=float((tn[y_]-1).abs().max()) if bool(y_.any()) else 0.,
                    cos=float(cos[y_].min()) if bool(y_.any()) else 1., elastic_max=float(tn[~y_].max()) if bool((~y_).any()) else 0.)

    def ret(self, eps_n):
        """the return from eps_n at the strain of the last grad call (W.tF, W.tR): eps_n+1, the cells yielding"""
        W = self.W
        U = W.tR.transpose(1, 2)@W.tF
        s = dev_part(sym_fun(U, torch.log)-eps_n)
        q = 2*W.mu*s.flatten(1).norm(dim=1)
        f = (1-math.sqrt(2/3)*self.sigma_y/q.clamp(min=1e-300)).clamp(min=0)
        return eps_n+f[:, None, None]*s, int((f > 0).sum())

    def update(self, x, y):
        """after a step: the return at (x, y); the energy dissipated, the cells yielding"""
        e0 = self.elastic(x, y)
        eps, n_y = self.ret(self.eps)
        if n_y:
            self.eps = eps
            self.apply()
        e1 = self.elastic(x, y) if n_y else e0
        self.dissipated += e0-e1
        return dict(yielding=n_y, dW=e0-e1, e_after=e1)

    def step_fb(self, I, max_passes=20):
        """one step of the midpoint integrator I with the return within it; I.E_int on the end references. Between
        passes the step's start is restored: the bodies, the skin and their history, the contact's and the glue's
        state (multipliers, active set, end gaps, held force and points) and the viscous references and records"""
        W = self.W
        keep = {k: getattr(I, k).clone() for k in ('x', 'v', 'a', 'xm1', 'ym1', 'xm2', 'ym2', 'y_int', 'wfix', 'pn') if hasattr(I, k)}
        keep_s = {k: getattr(I, k) for k in ('dissipated', 'contact_work', 'friction_work')}
        keep_w = {k: getattr(W, k).clone() for k in ('tgoff', 'tlam', 'tact', 'tlamt', 'tfst', 'trho', 'trhot', 'tlamR')} if W.ct is not None else {}
        keep_r = {k: r.clone() for k, r in W.tref.items()}
        eps_n = self.eps
        x_n, y_n = keep['x'], keep['y_int']
        e_n = I.E_int
        nxt = eps_n+(eps_n-self.eps_prev)                                           # (the start: the last step's rate)
        z0, prods, log = None, 0, []
        for p in range(1, max_passes+1):
            mid = .5*(eps_n+nxt)
            self.apply(mid)
            for k, v in keep.items():
                setattr(I, k, v.clone())
            for k, v in keep_s.items():
                setattr(I, k, v)
            for k, v in keep_w.items():
                getattr(W, k).copy_(v)
            for k, v in keep_r.items():
                W.tref[k].copy_(v)
            s_ = I.step(z0)
            prods += int(s_['products'])
            z0 = torch.cat((s_['xm'], s_['ym']))
            self.elastic(I.x, I.y_int)                                              # (the end strain: W.tF, W.tR)
            new, n_y = self.ret(eps_n)
            xm, ym = s_['xm'], s_['ym']                                              # the stop (10-26, as step_virial): the
            gx0, gy0 = self.elastic_grad(xm, ym)                                     # elastic force at the midpoint moved by
            self.apply(.5*(eps_n+new))                                               # the new references by no more than the
            gx1, gy1 = self.elastic_grad(xm, ym)                                     # solve's tolerance (was: eps_n+1's change
            self.apply(mid)                                                          # below tol x the yield strain, 5-7 passes)
            change = math.sqrt(float(((gx1-gx0)**2).sum())+float(((gy1-gy0)**2).sum()))/I.ref
            log.append((int(s_['products']), int(s_.get('al', 0)), change, n_y))
            nxt = new
            if change <= I.tol:
                break
        # the dissipation: the switches at the start (eps_n -> mid) and at the end (mid -> eps_n+1); mid is the last
        # pass's, eps_n+1 within the solve's tolerance of its own
        e_mid_n = self.elastic(x_n, y_n)
        e_mid_end = self.elastic(I.x, I.y_int)
        self.eps_prev, self.eps = eps_n, nxt
        self.apply()
        e_end = self.elastic(I.x, I.y_int)
        dW = (e_n-e_mid_n)+(e_mid_end-e_end)
        self.dissipated += dW
        I.E_int = e_end
        return s_, dict(yielding=n_y, dW=dW, passes=p, change=change, products=prods, lim=I.tol, log=log)

    def step_merged(self, I):
        """step_fb's return (F_B, eps_n+1 the return of the step's end strain from eps_n, the step on eps_n+1/2) with
        its passes inside the step's own (10-26: JointSolver.solve_contact's hook): the plastic references are updated
        each time the contact has settled for them, and the contact then goes on from the multipliers and set it reached
        (step_fb restarted them from the step's start each time: 3-4 plastic passes x 2-5 contact passes per step, each
        plastic pass as dear as the first; updating both after every solve cycled); one I.step, nothing to restore. The plastic part settles when the elastic force at
        the midpoint moves by no more than the solve's tolerance with the new references. The dissipation: the
        references' switches around the step as the last solve ran it"""
        W, n = self.W, self.W.n
        eps_n = self.eps
        x_n, y_n, e_n = I.x.clone(), I.y_int.clone(), I.E_int
        nxt = eps_n+(eps_n-self.eps_prev)                                           # (the start: the last step's rate)
        st = dict(nxt=nxt, mid=.5*(eps_n+nxt), used=None, log=[], settled=False)
        self.apply(st['mid'])

        def hook(z):
            xm, ym = z[:n], z[n:]
            st['used'] = st['mid']                                                 # (the references this solve ran on)
            self.elastic(2*xm-x_n, 2*ym-y_n)                                        # (the step's end strain: W.tF, W.tR)
            new, n_y = self.ret(eps_n)
            gx0, gy0 = self.elastic_grad(xm, ym)
            mid = .5*(eps_n+new)
            self.apply(mid)
            gx1, gy1 = self.elastic_grad(xm, ym)
            change = math.sqrt(float(((gx1-gx0)**2).sum())+float(((gy1-gy0)**2).sum()))/I.ref
            st['log'].append((change, n_y))
            st['nxt'], st['settled'] = new, change <= I.tol
            if st['settled']:                                                      # (the references stay those solved on: the
                self.apply(st['used'])                                              # contact's passes may still go on, and its
            else:                                                                  # last loose solve runs on them)
                st['mid'] = mid
            return st['settled']
        I.hook = hook
        try:
            s_ = I.step()
        finally:
            I.hook = None
        used = st['used'] if st['used'] is not None else st['mid']
        self.apply(used)                                                             # the dissipation: the switches around the
        e_mid_n = self.elastic(x_n, y_n)                                             # step as the last solve ran it
        e_mid_end = self.elastic(I.x, I.y_int)
        self.eps_prev, self.eps = eps_n, st['nxt']
        self.apply()
        e_end = self.elastic(I.x, I.y_int)
        dW = (e_n-e_mid_n)+(e_mid_end-e_end)
        self.dissipated += dW
        I.E_int = e_end
        n_y = st['log'][-1][1] if st['log'] else 0
        return s_, dict(yielding=n_y, dW=dW, passes=len(st['log']), products=int(s_['products']), lim=I.tol, settled=st['settled'],
                        log=[(int(s_['products']), int(s_.get('al', 0)), ch, ny) for ch, ny in st['log']])

    def step(self, I, max_passes=20):
        if self.criterion == 'virial':
            return self.step_virial(I, max_passes)
        return self.step_merged(I) if self.merged else self.step_fb(I, max_passes)

    def _keep(self, I):
        W = self.W
        k = {('I', a): getattr(I, a).clone() for a in ('x', 'v', 'a', 'xm1', 'ym1', 'xm2', 'ym2', 'y_int', 'wfix', 'pn') if hasattr(I, a)}
        k.update({('s', a): getattr(I, a) for a in ('dissipated', 'contact_work', 'friction_work')})
        if W.ct is not None:
            k.update({('W', a): getattr(W, a).clone() for a in ('tgoff', 'tlam', 'tact', 'tlamt', 'tfst', 'trho', 'trhot', 'tlamR')})
        k.update({('r', a): r.clone() for a, r in W.tref.items()})
        return k

    def _restore(self, I, k):
        W = self.W
        for (g, a), v in k.items():
            if g == 'I':
                setattr(I, a, v.clone())
            elif g == 's':
                setattr(I, a, v)
            elif g == 'W':
                getattr(W, a).copy_(v)
            else:
                W.tref[a].copy_(v)

    def step_virial(self, I, max_passes=20):
        """one step of the midpoint integrator I with the return within it, the yield on the model's own stress
        (10-26): min over z and delta of the step's Pi(z; eps_n + delta) + sum sigma^ V |delta| by blocks: z by the
        step's solve on the references P(delta) (the plastic stretch linearised at eps_n, solve_delta), delta by
        solve_delta at that midpoint; until the elastic terms' force at the midpoint moves by no more than the solve's
        tolerance (|d grad| / ref <= tol) when delta is updated. eps_n+1 = eps_n + 2 delta (exp exact); the dissipation
        the references' switches (start: eps_n -> P(delta used), end: that -> eps_n+1)"""
        keep = self._keep(I)
        eps_n = self.eps
        x_n, y_n, e_n = keep[('I', 'x')], keep[('I', 'y_int')], I.E_int
        Pn, Vn, Gn = self.lin(eps_n)
        Vt = Vn.transpose(1, 2)
        P_of = lambda d_: Pn+Vn@(Gn*(Vt@d_@Vn))@Vt
        delta = .5*(eps_n-self.eps_prev)                                            # (the start: the last step's rate)
        z0, prods, log = None, 0, []
        for p in range(1, max_passes+1):
            used = delta
            self.apply(P=P_of(used))
            self._restore(I, keep)
            s_ = I.step(z0)
            prods += int(s_['products'])
            xm, ym = s_['xm'], s_['ym']
            z0 = torch.cat((xm, ym))
            gx0, gy0 = self.elastic_grad(xm, ym)                                     # (W's buffers at the midpoint)
            delta, P_new, its, res = self.solve_delta(eps_n, used, xm, ym, I.tol)
            if not bool(torch.isfinite(delta).all()):                                # (10-26: a NaN seen once; say where)
                bad = (~torch.isfinite(delta.flatten(1))).any(1)
                raise FloatingPointError('solve_delta: %d cells not finite (first %s), pass %d, %d iterations; eps_n finite %s, |eps_n| max %.3e; '
                                         'z finite %s; used finite %s' % (int(bad.sum()), torch.nonzero(bad)[:5, 0].tolist(), p, its,
                                                                         bool(torch.isfinite(eps_n).all()), float(eps_n.abs().max()),
                                                                         bool(torch.isfinite(z0).all()), bool(torch.isfinite(used).all())))
            self.apply(P=P_new)
            gx1, gy1 = self.elastic_grad(xm, ym)
            dg = math.sqrt(float(((gx1-gx0)**2).sum())+float(((gy1-gy0)**2).sum()))/I.ref
            log.append((int(s_['products']), int(s_.get('al', 0)), dg/I.tol, int((delta.flatten(1).norm(dim=1) > 0).sum()), its))
            if dg <= I.tol:
                break
        chk = self.yield_check(eps_n, delta, xm, ym) if self.sigma_y < math.inf else None
        self.apply(P=P_of(used))                                                     # the dissipation: the switches around the
        e_mid_n = self.elastic(x_n, y_n)                                             # step as it ran (on eps_n + used)
        e_mid_end = self.elastic(I.x, I.y_int)
        self.eps_prev, self.eps = eps_n, eps_n+2*delta
        self.apply()
        e_end = self.elastic(I.x, I.y_int)
        dW = (e_n-e_mid_n)+(e_mid_end-e_end)
        self.dissipated += dW
        I.E_int = e_end
        return s_, dict(yielding=log[-1][3], dW=dW, passes=p, products=prods, lim=I.tol, log=[(a, c, ch*I.tol, n) for a, c, ch, n, _ in log],
                        inner=[l_[4] for l_ in log], check=chk)

    def strain(self, mass):
        """the mass-weighted mean plastic strain (3, 3) and the largest |eps_p| (Frobenius)"""
        w = mass/mass.sum()
        return (w[:, None, None]*self.eps).sum(0), float(self.eps.flatten(1).norm(dim=1).max())
