"""Updated-Lagrangian Power cells (UPDATED_LAGRANGE.md, 2026-10-27), torch reference. State per cell: site x_i, velocity v_i,
deformation gradient F_i (history, against rest), radius r_i (the weight w = r^2). A step from n to n+1 (backward Euler):
  geometry  the power cells cut by their balls at (x^n, r^n) (recut.Recut, GPU): walls (i, j) with area A, centroid c,
            normal n = (x_j - x_i) / l, position along the line h = (c - x_i).n; volumes; exposed caps
  gradient  L: G_s = sum_walls (dx_o - dx_s) (x) w d^T M_s^-1 (w = A / l, M_s = sum w d d^T), carried to the cell's centroid
            (the cell's material: sites are not at its centre; exact FEM gradients at the sites give 0.82 of the bending
            energy, at the centroids 0.93), delta_s the centroid's offset from the step's cut (cut_centroids; centroid
            'carried', the first form: F_s^n (c0_s - X_s), the rest offset carried affinely, left its cell under
            compression, 10-28):
            shift 'gradient' (the first form): the least-squares gradient of the gradients,
                       Gc_s = G_s + sum w (d^T M_s^-1 delta_s)(G_o - G_s) (10-27: on the trained 8k foam its coefficients
                       reach 25-66 on surface cells, kappa 8.7e5)
            shift 'displacement' (default): each centroid's increment from its own site, u_s = dx_s + G_s delta_s, then
                       L over the centroids, Gc_s = sum_walls (u_o - u_s) (x) w' e^T M'_s^-1, e = c_o - c_s, w' = A / |e|,
                       M'_s = sum w' e e^T (levers between the cells' material, not between the sites)
            both exact for affine increments
  energy    Pi(x) = sum m / (2 dt^2) |x - x^n - dt v^n|^2 - sum m g.x + sum V0 psi(F) + E_stab + E_jump (+ contact), with
            F_s(x) = (I + Gc_s(x - x^n)) F_s^n and neo-Hookean psi = mu / 2 (tr C - 3) - mu ln J + lam / 2 (ln J)^2;
            E_stab (S-NSNI, Wang & Hillman 2025, on the Green-Lagrange strain E = (C - I) / 2):
            1/2 sum_s sum_kl Jm_s^kl dE_k : D : dE_l, dE_k = (1 / V_s) sum_walls A t (E_o - E_s) n_k, Jm_s = det F F J0 F^T
            the cell's second moment about its centroid carried affinely; t where the wall's plane cuts the line between
            the two cells: smooth 'site' (the first form) x_s -> x_o (t = h / l; 7.6 % of the 8k foam's walls lie
            outside, up to 6 l away), smooth 'centroid' (default) c_s -> c_o (the cells are convex and the wall is a
            face of both, so t is in (0, 1))
            E_jump: 1/2 sum_walls (A / 3 l)(mu |delta|^2 + (lam + mu)(delta.n)^2), delta = (x_j - x_i) - (F_i + F_j)(X_j - X_i) / 2
  solve     trust-region Newton-CG on Pi, the gradient and Hessian-vector products written out (state / hvp; autograd on
            elastic() is the reference)
  update    v = (x - x^n) / dt, F = F(x); radii solved so that every cell's volume is V0 det F (recut.solve_radii)
Units as the DEM model's (moduli per unit density, E / 1000 for density 1000; masses = volumes)."""
import math
import time

import numpy as np
import torch

from free_surface.recut import Recut

f64 = torch.float64


def clock():
    torch.cuda.synchronize()
    return time.perf_counter()


def cell_moments_grid(X, r, V, spacing, dev='cuda'):
    """per cell: centroid offset from the site and second moment about the centroid (grid samples assigned by power
    distance inside the balls, scaled to the exact volume V); cells without a sample get zeros"""
    pt, rt = torch.as_tensor(X, device=dev), torch.as_tensor(r, device=dev)
    lo, hi = (X-r[:, None]).min(0), (X+r[:, None]).max(0)
    n = len(X); cnt = np.zeros(n); s1 = np.zeros((n, 3)); s2 = np.zeros((n, 3, 3))
    ax = [np.arange(lo[k]+spacing/2, hi[k], spacing) for k in range(3)]
    for x0 in ax[0]:
        Gg = np.stack(np.meshgrid([x0], ax[1], ax[2], indexing='ij'), -1).reshape(-1, 3)
        for c0 in range(0, len(Gg), 8192):
            y = torch.as_tensor(Gg[c0:c0+8192], device=dev)
            v_, a_ = (torch.cdist(y, pt)**2-rt[None]**2).min(1)
            ok = (v_ <= 0).cpu().numpy()
            if ok.any():
                a_ = a_.cpu().numpy()[ok]; R_ = Gg[c0:c0+8192][ok]-X[a_]
                np.add.at(cnt, a_, 1.); np.add.at(s1, a_, R_); np.add.at(s2, a_, R_[:, :, None]*R_[:, None, :])
    ok = cnt > 0
    dlt = np.where(ok[:, None], s1/np.maximum(cnt, 1)[:, None], 0.)
    J = np.where(ok[:, None, None], (s2/np.maximum(cnt, 1)[:, None, None]-dlt[:, :, None]*dlt[:, None, :])*V[:, None, None], 0.)
    return dlt, J, cnt


class ULModel:
    def __init__(self, X0, r0, V0, dlt0, J0, E, nu, rho=1., dev='cuda', jump=True, stab=True, shift='displacement', smooth='centroid',
                 centroid='cut'):
        t = lambda a: torch.as_tensor(np.asarray(a), dtype=f64, device=dev)
        self.dev, self.n = dev, len(X0)
        self.X0, self.r0, self.V0, self.dlt0, self.J0 = t(X0), t(r0), t(V0), t(dlt0), t(J0)
        self.mu = E/(2*(1+nu)); self.lam = E*nu/((1+nu)*(1-2*nu))
        self.m = rho*self.V0
        self.jump, self.stab, self.shift, self.smooth, self.centroid = jump, stab, shift, smooth, centroid
        self.recut = Recut(self.n, dev)
        self.I3 = torch.eye(3, dtype=f64, device=dev)

    # ---- geometry and the step's operators
    def geometry(self, x, r, GS=None):
        """GS: a geometry already cut at (x, r) (solve_radii's last), else cut here"""
        G, S = self.recut.geometry(x, r) if GS is None else GS
        keep = G.wall_A.detach() > 0
        P = G.P[keep]
        return dict(i=P[:, 0], j=P[:, 1], A=G.wall_A.detach()[keep], c=G.wall_c.detach()[keep], V=G.V.detach(),
                    capA=G.patch_A.detach(), hidden=S['hidden'], r=r)

    def cut_centroids(self, x, geo):
        """each cell's centroid offset from its site, from its cut alone: cones from the site, the walls as pyramids
        (centroid 3/4 of the way to the wall's centroid), the cap as a spherical sector (first moment r^2 / 4 times the
        cap's area vector a = -sum_walls A n, by closure): V (c - x) = sum_walls (A h / 4)(c_w - x) + (r^2 / 4) a (10-28:
        against the grid at rest 0.006 h median on the 8k foam; the F-carried offsets left their own cells under a
        uniform 10 % compression in 23 cells)"""
        i, j, A, cw = geo['i'], geo['j'], geo['A'], geo['c']
        d = x[j]-x[i]; nrm = d/d.norm(dim=1, keepdim=True)
        mom = torch.zeros(self.n, 3, dtype=f64, device=self.dev); a = torch.zeros_like(mom)
        for s_, nn in ((i, nrm), (j, -nrm)):
            mom.index_add_(0, s_, (A*((cw-x[s_])*nn).sum(1)/4)[:, None]*(cw-x[s_])); a.index_add_(0, s_, -A[:, None]*nn)
        return (mom+(geo['r']**2/4)[:, None]*a)/geo['V'][:, None]

    def operators(self, x, F, geo):
        """the step's linear maps, built at the step start (x^n, F^n, geometry)"""
        i, j, A, c = geo['i'], geo['j'], geo['A'], geo['c']
        d = x[j]-x[i]; L = d.norm(dim=1); nrm = d/L[:, None]
        h = ((c-x[i])*nrm).sum(1); hj = L-h; w = A/L
        n = self.n
        M = torch.zeros(n, 3, 3, dtype=f64, device=self.dev)
        dd = w[:, None, None]*d[:, :, None]*d[:, None, :]
        M.index_add_(0, i, dd); M.index_add_(0, j, dd)
        Mi = torch.linalg.inv(M)
        delta = self.cut_centroids(x, geo) if self.centroid == 'cut' else torch.einsum('nab,nb->na', F, self.dlt0)
        detF = torch.linalg.det(F)
        Jm = detF[:, None, None]*F@self.J0@F.transpose(1, 2)
        ends = []
        cc = x+delta
        for s_, o_, dv, nn, hh in ((i, j, d, nrm, h), (j, i, -d, -nrm, hj)):
            row = torch.einsum('fa,fab->fb', w[:, None]*dv, Mi[s_])              # w d^T M_s^-1
            cen = (row*delta[s_]).sum(1)                                          # w d^T M_s^-1 delta_s
            if self.smooth == 'centroid':                                         # where the wall's plane cuts c_s -> c_o
                t = ((c-cc[s_])*nn).sum(1)/((cc[o_]-cc[s_])*nn).sum(1)
            else:                                                                 # ... cuts x_s -> x_o
                t = hh/L
            smo = (A*t)[:, None]*nn/geo['V'][s_][:, None]                          # A t n / V_s
            ends.append((s_, o_, row, cen, smo))
        Kj = (A/(3*L))
        crow = None
        if self.shift == 'displacement':
            e = cc[j]-cc[i]; wc = A/e.norm(dim=1)
            Mc = torch.zeros(n, 3, 3, dtype=f64, device=self.dev)
            ee = wc[:, None, None]*e[:, :, None]*e[:, None, :]
            Mc.index_add_(0, i, ee); Mc.index_add_(0, j, ee)
            Mci = torch.linalg.inv(Mc)
            crow = [torch.einsum('fa,fab->fb', wc[:, None]*e, Mci[i]), torch.einsum('fa,fab->fb', -wc[:, None]*e, Mci[j])]
        cat = lambda k: torch.cat([e_[k] for e_ in ends])                          # both ends of every wall in one list
        return dict(i=i, j=j, ends=ends, Jm=Jm, Kj=Kj, x0=x.clone(), F0=F.clone(), L=L, nrm=nrm, A=A, delta=delta, crow=crow,
                    S=cat(0), O=cat(1), R=cat(2), CEN=cat(3), SM=cat(4), RC=torch.cat(crow) if crow is not None else None,
                    b=self.X0[j]-self.X0[i])

    # ---- the energy
    def def_grad(self, x, ops):
        dx = x-ops['x0']
        G = torch.zeros(self.n, 3, 3, dtype=f64, device=self.dev)
        for s_, o_, row, cen, smo in ops['ends']:
            G = G.index_add(0, s_, (dx[o_]-dx[s_])[:, :, None]*row[:, None, :])
        if self.shift == 'displacement':
            u = dx+torch.einsum('nab,nb->na', G, ops['delta'])
            Gc = torch.zeros_like(G)
            for (s_, o_, row, cen, smo), rc in zip(ops['ends'], ops['crow']):
                Gc = Gc.index_add(0, s_, (u[o_]-u[s_])[:, :, None]*rc[:, None, :])
        else:
            Gc = G
            for s_, o_, row, cen, smo in ops['ends']:
                Gc = Gc.index_add(0, s_, cen[:, None, None]*(G[o_]-G[s_]))
        return (self.I3+Gc)@ops['F0']

    def elastic(self, x, ops, terms=False):
        F = self.def_grad(x, ops)
        J = torch.linalg.det(F)
        C = F.transpose(1, 2)@F
        lnJ = torch.log(J.clamp(min=1e-300))
        psi = .5*self.mu*(C.diagonal(dim1=1, dim2=2).sum(1)-3)-self.mu*lnJ+.5*self.lam*lnJ**2
        e = dict(cells=(self.V0*psi).sum())
        if bool((J <= 0).any()):
            e['cells'] = e['cells']*0+math.inf
        if self.stab:
            Eg = .5*(C-self.I3)
            dE = torch.zeros(self.n, 3, 3, 3, dtype=f64, device=self.dev)        # [s, k, a, b]
            for s_, o_, row, cen, smo in ops['ends']:
                dE = dE.index_add(0, s_, smo[:, :, None, None]*(Eg[o_]-Eg[s_])[:, None])
            tr = dE.diagonal(dim1=2, dim2=3).sum(2)                               # [s, k]
            e['stab'] = .5*(torch.einsum('skl,sk,sl->', ops['Jm'], tr, tr)*self.lam+2*self.mu*torch.einsum('skl,skab,slab->', ops['Jm'], dE, dE))
        if self.jump:
            i, j = ops['i'], ops['j']
            dlt = (x[j]-x[i])-.5*torch.einsum('fab,fb->fa', F[i]+F[j], self.X0[j]-self.X0[i])
            nc = (x[j]-x[i])/(x[j]-x[i]).norm(dim=1, keepdim=True)
            e['jump'] = .5*(ops['Kj']*(self.mu*(dlt*dlt).sum(1)+(self.lam+self.mu)*(dlt*nc).sum(1)**2)).sum()
        return (e, F) if terms else sum(e.values())

    # ---- the energy, its gradient and Hessian-vector products written out (the solve's path; elastic() is the autograd
    # reference they are checked against). F is linear in x: dF = Gc(dx) F0, so every product is the chain
    # dx -> dGc -> dF -> each term's second derivative -> back through the adjoint of the chain
    def kin(self, v, ops):
        """Gc(v) for an increment v of the sites (F = (I + Gc) F0)"""
        S, O = ops['S'], ops['O']
        G = torch.zeros(self.n, 3, 3, dtype=f64, device=self.dev).index_add_(0, S, (v[O]-v[S])[:, :, None]*ops['R'][:, None, :])
        if self.shift == 'displacement':
            u = v+torch.einsum('nab,nb->na', G, ops['delta'])
            return torch.zeros_like(G).index_add_(0, S, (u[O]-u[S])[:, :, None]*ops['RC'][:, None, :])
        return G.index_add(0, S, ops['CEN'][:, None, None]*(G[O]-G[S]))

    def kin_T(self, Gb, ops):
        """the adjoint of kin: the sites' gradient from dU/dGc"""
        S, O = ops['S'], ops['O']
        z = torch.zeros(self.n, 3, dtype=f64, device=self.dev)
        if self.shift == 'displacement':
            q = torch.einsum('fab,fb->fa', Gb[S], ops['RC'])
            ub = z.index_add(0, O, q).index_add_(0, S, -q)
            Gb = ub[:, :, None]*ops['delta'][:, None, :]
        else:
            W = ops['CEN'][:, None, None]*Gb[S]
            ub = z.clone()
            Gb = Gb.index_add(0, O, W).index_add_(0, S, -W)
        q = torch.einsum('fab,fb->fa', Gb[S], ops['R'])
        return ub.index_add_(0, O, q).index_add_(0, S, -q)

    def _stab_pull(self, dE, ops):
        """dU/dE per cell from the strains' smoothed derivatives dE [s, k, a, b] (and the energy's density Q)"""
        S, O, SM = ops['S'], ops['O'], ops['SM']
        tr = dE.diagonal(dim1=2, dim2=3).sum(2)
        Q = 2*self.mu*torch.einsum('skl,slab->skab', ops['Jm'], dE)+self.lam*torch.einsum('skl,sl->sk', ops['Jm'], tr)[:, :, None, None]*self.I3
        W = torch.einsum('fk,fkab->fab', SM, Q[S])
        return torch.zeros(self.n, 3, 3, dtype=f64, device=self.dev).index_add_(0, O, W).index_add_(0, S, -W), Q

    def _stab_push(self, E, ops):
        S, O, SM = ops['S'], ops['O'], ops['SM']
        return torch.zeros(self.n, 3, 3, 3, dtype=f64, device=self.dev).index_add_(0, S, SM[:, :, None, None]*(E[O]-E[S])[:, None])

    def state(self, y, ops):
        """everything the gradient and the products at y need; None where some J <= 0"""
        Gc = self.kin(y-ops['x0'], ops)
        F = (self.I3+Gc)@ops['F0']
        J = torch.linalg.det(F)
        if bool((J <= 0).any()):
            return None
        FiT = torch.linalg.inv(F).transpose(1, 2); lnJ = torch.log(J)
        C = F.transpose(1, 2)@F
        e = dict(cells=float((self.V0*(.5*self.mu*(C.diagonal(dim1=1, dim2=2).sum(1)-3)-self.mu*lnJ+.5*self.lam*lnJ**2)).sum()))
        P = self.V0[:, None, None]*(self.mu*(F-FiT)+self.lam*lnJ[:, None, None]*FiT)
        st = dict(F=F, FiT=FiT, lnJ=lnJ)
        if self.stab:
            dE = self._stab_push(.5*(C-self.I3), ops)
            Eb, Q = self._stab_pull(dE, ops)
            e['stab'] = .5*float((dE*Q).sum())
            P = P+F@Eb; st['Eb'] = Eb
        yg = torch.zeros(self.n, 3, dtype=f64, device=self.dev)
        if self.jump:
            i, j, b, Kj = ops['i'], ops['j'], ops['b'], ops['Kj'][:, None]
            a = y[j]-y[i]; la = a.norm(dim=1, keepdim=True); nc = a/la
            dl = a-.5*torch.einsum('fab,fb->fa', F[i]+F[j], b); sn = (dl*nc).sum(1, keepdim=True)
            e['jump'] = .5*float((Kj*(self.mu*(dl*dl).sum(1, keepdim=True)+(self.lam+self.mu)*sn**2)).sum())
            gd = Kj*(self.mu*dl+(self.lam+self.mu)*sn*nc); gn = Kj*(self.lam+self.mu)*sn*dl
            tn = gn-nc*(nc*gn).sum(1, keepdim=True)
            ga = gd+tn/la
            PF = -.5*gd[:, :, None]*b[:, None, :]
            P = P.index_add(0, i, PF).index_add_(0, j, PF)
            yg.index_add_(0, j, ga).index_add_(0, i, -ga)
            st.update(nc=nc, la=la, dl=dl, sn=sn, gn=gn, tn=tn)
        st['e'] = e
        st['grad'] = self.kin_T(P@ops['F0'].transpose(1, 2), ops)+yg
        return st

    def hvp(self, st, d, ops):
        """the elastic energy's Hessian at the state st times d (n, 3)"""
        F, FiT, lnJ = st['F'], st['FiT'], st['lnJ']
        dF = self.kin(d, ops)@ops['F0']
        tr = torch.einsum('nab,nab->n', FiT, dF)                                   # tr(F^-1 dF)
        A_ = FiT@dF.transpose(1, 2)@FiT
        dP = self.V0[:, None, None]*(self.mu*dF+(self.mu-self.lam*lnJ)[:, None, None]*A_+self.lam*tr[:, None, None]*FiT)
        if self.stab:
            dEl = F.transpose(1, 2)@dF; dEl = .5*(dEl+dEl.transpose(1, 2))
            dEb, _ = self._stab_pull(self._stab_push(dEl, ops), ops)
            dP = dP+dF@st['Eb']+F@dEb
        dy = torch.zeros(self.n, 3, dtype=f64, device=self.dev)
        if self.jump:
            i, j, b, Kj = ops['i'], ops['j'], ops['b'], ops['Kj'][:, None]
            nc, la, dl, sn, gn, tn = st['nc'], st['la'], st['dl'], st['sn'], st['gn'], st['tn']
            dot = lambda p, q: (p*q).sum(1, keepdim=True)
            da = d[j]-d[i]
            ddl = da-.5*torch.einsum('fab,fb->fa', dF[i]+dF[j], b)
            dnc = (da-nc*dot(nc, da))/la
            ds = dot(ddl, nc)+dot(dl, dnc)
            dgd = Kj*(self.mu*ddl+(self.lam+self.mu)*(ds*nc+sn*dnc))
            dgn = Kj*(self.lam+self.mu)*(ds*dl+sn*ddl)
            dga = dgd+(dgn-dnc*dot(nc, gn)-nc*dot(dnc, gn)-nc*dot(nc, dgn))/la-tn*dot(nc, da)/la**2
            dPF = -.5*dgd[:, :, None]*b[:, None, :]
            dP = dP.index_add(0, i, dPF).index_add_(0, j, dPF)
            dy.index_add_(0, j, dga).index_add_(0, i, -dga)
        return self.kin_T(dP@ops['F0'].transpose(1, 2), ops)+dy

    # ---- the step's solve: trust-region Newton-CG (Steihaug-Toint, as skin_solve.SkinSolver), the region in the norm of
    # a diagonal preconditioner per site (mass / dt^2 plus the stiffness scale (lam + 2 mu) sum_walls A / l)
    def tr_newton(self, eg, hvp, y, dg, ref, noise, tol=1e-9, maxit=5000):
        """minimises Pi from y (eg(y): Pi, its gradient (None where Pi is infinite) and what hvp(aux, d) needs); stops at
        |dPi/dy| <= tol ref, or at the energy's rounding: noise (a step whose predicted decrease is below it is accepted
        when the gradient shrinks, else the solve ends there). The inner CG stops at min(0.01, res) times the residual
        (quadratic outer convergence). The region starts at the larger of twice the last call's largest accepted step
        and the preconditioned gradient's length. Returns y, Pi(y), stats"""
        stats = dict(products=0, outer=0, rejected=0, negative=0, floor=False)
        e0, g, aux = eg(y); yg = y
        if g is None:
            raise RuntimeError('tr_newton: the start has no finite energy')
        res = float(g.norm())/ref
        delta = max(getattr(self, 'delta', 0.), float((g.detach()**2/dg[:, None]).sum())**.5)
        big = 0.
        while res > tol and stats['products'] < maxit:
            stats['outer'] += 1
            r_ = -g.detach(); z = r_/dg[:, None]; d = z.clone(); rz = float((r_*z).sum())
            p = torch.zeros_like(r_); Hp = torch.zeros_like(r_); pMp, pMd, dMd = 0., 0., rz
            edge = False; stop = max(tol, min(.01, res)*res)*ref
            while True:                                                           # Steihaug-Toint CG
                Hd = hvp(aux, d); stats['products'] += 1
                dHd = float((d*Hd).sum())
                to_edge = lambda: (-pMd+(pMd**2+dMd*(delta**2-pMp))**.5)/dMd
                if dHd <= 0:
                    tau = to_edge(); p, Hp = p+tau*d, Hp+tau*Hd; edge = True; stats['negative'] += 1; break
                al = rz/dHd
                if pMp+2*al*pMd+al**2*dMd >= delta**2:
                    tau = to_edge(); p, Hp = p+tau*d, Hp+tau*Hd; edge = True; break
                p = p+al*d; Hp = Hp+al*Hd; pMp = pMp+2*al*pMd+al**2*dMd
                r_ = r_-al*Hd; z = r_/dg[:, None]; rz1 = float((r_*z).sum())
                if float(r_.norm()) <= stop or stats['products'] >= maxit:
                    break
                be = rz1/rz; pMd = be*(pMd+al*dMd); dMd = rz1+be**2*dMd; d = z+be*d; rz = rz1
            pred = -float((g.detach()*p).sum()+.5*(p*Hp).sum())
            yg1 = yg+p; e1, g1, aux1 = eg(yg1)
            pM = delta if edge else pMp**.5
            if pred <= noise:                                                     # below the energy's rounding
                if g1 is not None and float(g1.norm()) < res*ref:
                    e0, g, yg, aux = e1, g1, yg1, aux1; res = float(g.norm())/ref; big = max(big, pM)
                    continue
                stats['floor'] = True; break
            rho = (e0-e1)/pred
            if rho < .25:
                delta = .25*pM
            elif rho > .75 and edge:
                delta = 2*delta
            if rho > 1e-4:
                e0, g, yg, aux = e1, g1, yg1, aux1; res = float(g.norm())/ref; big = max(big, pM)
            else:
                stats['rejected'] += 1
        stats['residual'] = res
        if big > 0:
            self.delta = 2*big
        return yg, e0, stats

    def step(self, st, dt, grav=None, ref=1., tol=1e-9, maxit=5000, analytic=True, vol_tol=1e-5, ground=None, ground_r=None):
        """one backward-Euler step from st = dict(x, v, F, r[, GS, r_prev]); returns the new state and a report
        (energies at the start on this step's operators, at the end, the solve's stats, the volume fit's). analytic: the
        written-out gradient and products (state / hvp), else autograd on elastic(). The volume fit starts from the
        weights extrapolated in time (2 w^n - w^n-1, where positive) and stops at vol_tol (relative; each step fits
        V0 det F afresh, so its error does not add up; 10-27 8k: 1 Newton iteration, 9e-6, against 3 from w^n to 1e-9).
        ground = (n, c, E*): the plane n.x = c (n unit, the body on its + side), Hertz on the ball (centre x, this step's
        radius r) of every cell with an exposed cap at the step's start: energy 8/15 E* R^1/2 delta^5/2, delta = R - (n.x
        - c) > 0, frictionless (E* = E / (1 - nu^2) against a rigid ground); ground_r: fixed contact radii instead of
        r (10-27 test: the volume fit's radii fed back into the contact on the jack's first-touching cluster)"""
        x, v, F, r = st['x'], st['v'], st['F'], st['r']
        tm = {}; t0 = clock()
        geo = self.geometry(x, r, st.get('GS'))
        ops = self.operators(x, F, geo)
        tm['operators'] = clock()-t0; t0 = clock()
        gv = torch.zeros(3, dtype=f64, device=self.dev) if grav is None else torch.as_tensor(grav, dtype=f64, device=self.dev)
        mdt = self.m[:, None]/(2*dt*dt); xp = x+dt*v
        if ground is not None:
            gn = torch.as_tensor(ground[0], dtype=f64, device=self.dev); gc = float(ground[1])
            rc_ = r if ground_r is None else ground_r
            cb = torch.nonzero(geo['capA'] > 0).flatten(); cR = float(ground[2])*torch.sqrt(rc_[cb]); Rb = rc_[cb]
        def contact(y):
            """the ground's energy, gradient and the products' normal stiffness per ball"""
            if ground is None:
                return 0., 0., None
            dl = (Rb-(y[cb]@gn-gc)).clamp(min=0.)
            return (float((8/15*cR*dl**2.5).sum()), torch.zeros_like(y).index_add_(0, cb, -(4/3*cR*dl**1.5)[:, None]*gn),
                    2*cR*dl.sqrt())
        def Pi(y):
            e = (mdt*(y-xp)**2).sum()-(self.m[:, None]*gv*y).sum()+self.elastic(y, ops)
            if ground is not None:
                e = e+(8/15*cR*(Rb-(y[cb]@gn-gc)).clamp(min=0.)**2.5).sum()
            return e
        w = geo['A']/(x[geo['j']]-x[geo['i']]).norm(dim=1)
        k = torch.zeros(self.n, dtype=f64, device=self.dev).index_add_(0, geo['i'], w).index_add_(0, geo['j'], w)
        dg = self.m/dt**2+(self.lam+2*self.mu)*k
        y0 = xp+dt*dt*gv
        if ground is not None:
            dg = dg.index_add(0, cb, contact(y0)[2])
        if not math.isfinite(float(Pi(y0))):
            y0 = x
        e_start, _ = self.elastic(x, ops, terms=True)
        # the energy's rounding: tr C ~ 3 per cell carries the cells' sum's absolute error, eps (mu + lam) sum V0; the
        # inertia's |y - xp|^2 and the other terms are sums of non-negative parts at most comparable
        noise = 64*torch.finfo(f64).eps*float((self.mu+self.lam)*self.V0.sum())
        if analytic:
            def eg(y):
                a_ = self.state(y, ops)
                if a_ is None:
                    return math.inf, None, None
                ec, gc_, a_['kc'] = contact(y)
                return (float((mdt*(y-xp)**2).sum()-(self.m[:, None]*gv*y).sum())+sum(a_['e'].values())+ec,
                        2*mdt*(y-xp)-self.m[:, None]*gv+a_['grad']+gc_, a_)
            def hvp(a_, d):
                h_ = 2*mdt*d+self.hvp(a_, d, ops)
                if ground is not None:
                    h_ = h_.index_add(0, cb, (a_['kc']*(d[cb]@gn))[:, None]*gn)
                return h_
        else:
            def eg(y):
                y = y.detach().requires_grad_(True)
                e = Pi(y)
                if not bool(torch.isfinite(e)):
                    return math.inf, None, None
                g, = torch.autograd.grad(e, y, create_graph=True)
                return float(e.detach()), g.detach(), (g, y)
            def hvp(a_, d):
                return torch.autograd.grad(a_[0], a_[1], d, retain_graph=True)[0]
        y, pi, stats = self.tr_newton(eg, hvp, y0, dg, ref, noise, tol, maxit)
        e_end, F1 = self.elastic(y, ops, terms=True)
        tm['solve'] = clock()-t0; t0 = clock()
        Vt = self.V0*torch.linalg.det(F1)
        w_ = r*r
        if 'r_prev' in st:
            w2 = 2*w_-st['r_prev']**2
            w_ = torch.where(w2 > 0, w2, w_)
        r1, G1, S1, it, nrep = self.recut.solve_radii(y, torch.sqrt(w_), Vt, tol=vol_tol)
        tm['radii'] = clock()-t0
        verr = float(((G1.V.detach()-Vt)/Vt).abs().max())
        rep = dict(e_start={k_: float(v_) for k_, v_ in e_start.items()}, e_end={k_: float(v_) for k_, v_ in e_end.items()},
                   solve=stats, radii_it=it, repairs=nrep, vol_err=verr, walls=(geo['i'], geo['j'], geo['A']),
                   minJ=float((Vt/self.V0).min()), time=tm, radii_start=self.recut.stats.get('start_ratio'))
        if ground is not None:
            dl = (Rb-(y[cb]@gn-gc)).clamp(min=0.)
            rep['ground'] = dict(force=float((4/3*cR*dl**1.5).sum()), touching=int((dl > 0).sum()), overlap=float((dl/Rb).max()),
                                 energy=float((8/15*cR*dl**2.5).sum()), balls=len(cb))
        return dict(x=y, v=(y-x)/dt, F=F1.detach(), r=r1, GS=(G1, S1), r_prev=r), rep
