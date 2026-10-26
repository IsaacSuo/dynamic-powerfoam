"""Contact of the skin with a plane (2026-10-23, offline). On every region of the boundary skin (a patch of the sphere of
cell c: site p, radius r, body B) the deformed surface is (sphere_model.SkinTerms)
  y(m) = x_B + F_B (p + r m - X_B) + sum_s W_s(m) delta_s,   delta_s = y_s - x_B - F_B (Y_s - X_B)
(m the unit direction from the site, W the boundary skin's interpolant, boundary_skin.interpolant; on an arc W is the
arc's span{1, cos, sin} trace and y = sum_k L_k y_k), linear in the body positions x and the skin points y for a fixed
material point. The plane n . y >= c holds on the whole surface iff it holds at every region's lowest point, arcs
included, so there is one condition per region:
  interior  Newton on the sphere in tangent coordinates (gradient and Hessian by forward differentiation of the
            interpolant with its boundary nodes fixed at the current point), from the lowest of the region's m0 and points
            toward its arcs, with backtracking, kept inside the region (exposed on the sphere; a site with several islands:
            the island with the nearest boundary)
  arcs      g(phi) = g_mid + A (1 - cos phi) + B sin phi (phi from the arc's middle): closed-form minimum, or an end
The region's gap is the lower of the two. Its derivatives with respect to x and y are those at the lowest point with the
material point held (envelope theorem: it is a minimum over the material points)."""
import math

import numpy as np
import torch

from free_surface.boundary_skin import interpolant, lagrange3


def tangents(m):
    ref = torch.where(m[:, :1].abs() < .9, torch.tensor([1., 0, 0], dtype=m.dtype, device=m.device), torch.tensor([0, 1., 0], dtype=m.dtype, device=m.device))
    t1 = torch.cross(m, ref, dim=1); t1 = t1/t1.norm(dim=1, keepdim=True)
    return t1, torch.cross(m, t1, dim=1)


class SkinSurface:
    """terms: SkinTerms (BoundaryTerms) of the skin built by boundary_skin.build with the same p, r and interp; R:
    boundary_skin.regions(p, r) (the same structure, so the same skin points); body_of_cell: the DEM's body per site;
    sl: the regions (default all; with every region of each of their sites: the islands' inside test needs them)."""

    def __init__(self, terms, R, p, r, body_of_cell, interp='shepard0', nbd=16, sl=None):
        dev, dt = terms.X.device, terms.X.dtype
        assert float((R.Y-terms.Y).abs().max()) <= 1e-9*float(terms.Y.abs().max()), 'skin points differ from the terms'
        sl = torch.arange(R.ng, device=dev) if sl is None else sl
        gs = R.gsite[sl]
        self.terms, self.R, self.P, self.sl, self.gsite, self.m0 = terms, R, len(sl), sl, gs, R.m0[sl]
        I = interpolant(R, sl, p, r, interp, nbd)
        self.I = I
        self.body = torch.as_tensor(np.asarray(body_of_cell), device=dev)[gs]
        self.p, self.r = p[gs], r[gs]
        nb = R.S['nbrs'][gs]; kn = nb.clamp(min=0)                                  # exposed: outside every neighbour's sphere
        def exposed(mv):
            X = p[gs]+r[gs, None]*mv
            pw = ((X[:, None]-p[kn])**2).sum(-1)-r[kn]**2
            return torch.where(nb >= 0, pw > 0, torch.ones_like(pw, dtype=torch.bool)).all(1)
        self.exposed = exposed
        self.loc = I.loc_g.clamp(min=0); self.lmask = (I.loc_g >= 0).to(dt)
        ent = I.ent; self.vb = I.valid_b
        self.a0, self.a1 = R.t0[ent], R.t1[ent]
        self.cc, self.rho, self.u, self.v = R.c[ent], R.rho[ent], R.u[ent], R.v[ent]
        self.gid3, self.slot3 = R.gids[ent], I.loc3                                      # (P, Bmax, 3): start, mid, end
        # trust radius: the region's angular extent from m0 (to its corners)
        mc = self.arcpt(self.a0)
        ang = torch.acos(((mc*self.m0[:, None]).sum(-1)).clamp(-1, 1))
        self.rad = torch.where(self.vb, ang, torch.zeros_like(ang)).amax(1)
        # sites with several regions (islands): every boundary entry of the site with its region, for the inside test
        cnt = torch.bincount(gs, minlength=int(gs.max())+1)
        multi = torch.nonzero(cnt[gs] > 1)[:, 0]
        self.multi = None
        if len(multi):
            rows = []
            for k in multi.tolist():
                sib = torch.nonzero(gs == gs[k])[:, 0]
                rows.append([(int(q), b) for q in sib.tolist() for b in range(int(self.vb[q].sum()))])
            E = max(len(x) for x in rows)
            own = torch.full((len(multi), E), -1, dtype=torch.long, device=dev); eb = torch.zeros_like(own)
            for i_, row in enumerate(rows):
                own[i_, :len(row)] = torch.tensor([q for q, _ in row], device=dev); eb[i_, :len(row)] = torch.tensor([b for _, b in row], device=dev)
            self.multi = (multi, own, eb)

    def arcpt(self, th, sel=None):
        """unit directions (from the region's site) of the arc points at angles th (P, Bmax, ...) of every entry"""
        sh = th.shape+(3,)
        ex = lambda a: a.view(a.shape[:2]+(1,)*(th.dim()-2)+a.shape[2:])
        X = ex(self.cc)+ex(self.rho)[..., None]*(torch.cos(th)[..., None]*ex(self.u)+torch.sin(th)[..., None]*ex(self.v))
        return ((X-self.p.view((-1,)+(1,)*(th.dim()-1)+(3,)))/self.r.view((-1,)+(1,)*th.dim())).expand(sh)

    def inside(self, m):
        """m (P, 3): in the region (exposed; for islands also nearest to the region's own boundary)"""
        ok = self.exposed(m)
        if self.multi is not None:
            k, own, eb = self.multi
            q = own.clamp(min=0)
            X = self.p[k][:, None]+self.r[k][:, None, None]*m[k][:, None]                  # (K, 3) -> (K, 1, 3)
            cc, u, v, rho = self.cc[q, eb], self.u[q, eb], self.v[q, eb], self.rho[q, eb]
            a0, a1 = self.a0[q, eb], self.a1[q, eb]
            rel = X-cc
            th = torch.atan2((rel*v).sum(-1), (rel*u).sum(-1))
            tm = .5*(a0+a1); th = tm+torch.remainder(th-tm+math.pi, 2*math.pi)-math.pi
            th = torch.minimum(torch.maximum(th, a0), a1)
            d = (cc+rho[..., None]*(torch.cos(th)[..., None]*u+torch.sin(th)[..., None]*v)-X).norm(dim=-1)
            d = torch.where(own >= 0, d, torch.full_like(d, float('inf')))
            ok = ok.clone(); ok[k] = ok[k] & (own.gather(1, d.argmin(1, keepdim=True))[:, 0] == k)
        return ok

    def fields(self, x, y):
        """per region: F_B, base = x_B + F_B (p - X_B), local deviations delta (P, Smax, 3)"""
        T = self.terms
        F = T.F(x, y)
        FB, xB, XB = F[self.body], x[self.body], T.X[self.body]
        dl = (y[self.loc]-xB[:, None]-torch.einsum('pij,psj->psi', FB, T.Y[self.loc]-XB[:, None]))*self.lmask[..., None]
        return FB, xB+torch.einsum('pij,pj->pi', FB, self.p-XB), dl

    def points(self, x, y, m, W):
        """surface points (P, Q, 3) at directions m (P, Q, 3) with interpolant weights W (P, Q, Smax); differentiable in x, y"""
        FB, base, dl = self.fields(x, y)
        return base[:, None]+self.r[:, None, None]*torch.einsum('pij,pqj->pqi', FB, m)+torch.einsum('pqs,psi->pqi', W, dl)

    def weights(self, m):
        """interpolant weights (P, Q, Smax) at directions m (P, Q, 3), boundary nodes placed for m"""
        phi, _ = self.I.make_phi(m)
        return phi(m)

    def lowest(self, x, y, n, c, iters=40, tol=1e-9, rule="both"):
        """per region the lowest point over the plane n . y = c: gap (P), direction m (P, 3), weights W (P, Smax), kind
        (0 interior, 1 arc), and Newton diagnostics; all detached"""
        with torch.no_grad():
            FB, base, dl = self.fields(x, y)
            g0 = base@n-c; a = self.r[:, None]*torch.einsum('pji,j->pi', FB, n); dn = dl@n          # g(m) = g0 + a.m + W(m).dn
            gfun = lambda phi, m: g0[:, None]+(a[:, None]*m).sum(-1)+(phi(m)*dn[:, None]).sum(-1)
            P = self.P
            # arcs: closed form on every entry
            gk = y[self.gid3]@n-c                                                               # (P, Bmax, 3)
            beta = .5*(self.a1-self.a0); s2 = torch.sin(beta/2)**2
            A = (.5*(gk[..., 0]+gk[..., 2])-gk[..., 1])/(2*s2); B = (gk[..., 2]-gk[..., 0])/(2*torch.sin(beta))
            ph = torch.atan2(-B, A)
            ph = torch.where(ph.abs() <= beta, ph, -beta)
            cand = torch.stack((ph, -beta, beta), -1)
            gc = gk[..., 1:2]+A[..., None]*(1-torch.cos(cand))+B[..., None]*torch.sin(cand)
            j = gc.argmin(-1, keepdim=True)
            ga = torch.where(self.vb, gc.gather(-1, j)[..., 0], torch.full_like(A, float('inf')))
            pha = cand.gather(-1, j)[..., 0]
            ba = ga.argmin(1)
            g_arc = ga.gather(1, ba[:, None])[:, 0]
            # interior: start from the lowest inside sample, m0 and 1/4, 1/2, 3/4 of the way toward 4 points of every arc (the
            # region is the sphere minus caps, its arcs bulge into it: a descent can stop against an arc with the lowest
            # point on its far side, so the start has to be in the right basin)
            R = self.R
            ar = torch.arange(P, device=x.device)
            th = self.a0[..., None]+(self.a1-self.a0)[..., None]*(torch.arange(4, dtype=x.dtype, device=x.device)+.5)/4
            fr = torch.tensor([.25, .5, .75], dtype=x.dtype, device=x.device)
            smp = (1-fr[:, None])*self.m0[:, None, None, None]+fr[:, None]*self.arcpt(th)[:, :, :, None]   # (P, Bmax, 4, 3, 3)
            okv = self.vb[:, :, None, None].expand(smp.shape[:-1]).reshape(P, -1)
            smp = torch.cat((self.m0[:, None], smp.reshape(P, -1, 3)), 1); okv = torch.cat((torch.ones_like(okv[:, :1]), okv), 1)
            smp = smp/smp.norm(dim=-1, keepdim=True)
            g = torch.full((P,), float('inf'), dtype=x.dtype, device=x.device); m = self.m0.clone()
            for q0 in range(0, smp.shape[1], 32):
                sq = smp[:, q0:q0+32]
                gq = gfun(self.I.make_phi(sq)[0], sq)
                ins = torch.stack([self.inside(sq[:, k]) for k in range(sq.shape[1])], 1) & okv[:, q0:q0+32]
                gq = torch.where(ins, gq, torch.full_like(gq, float('inf')))
                j = gq.argmin(1); gj = gq.gather(1, j[:, None])[:, 0]
                upd = gj < g
                g = torch.where(upd, gj, g); m = torch.where(upd[:, None], sq[ar, j], m)
            have = torch.isfinite(g)
            rad = self.rad.clone()
            E1 = torch.zeros(P, 2, dtype=x.dtype, device=x.device); E1[:, 0] = 1
            E2 = torch.zeros_like(E1); E2[:, 1] = 1
            conv = torch.zeros(P, dtype=torch.bool, device=x.device); stuck = torch.zeros_like(conv)
            gnorm = torch.zeros(P, dtype=x.dtype, device=x.device)
            scale = a.norm(dim=1)+dn.abs().amax(1)
        for it in range(iters):
            with torch.no_grad():
                phi, _ = self.I.make_phi(m[:, None])
                t1, t2 = tangents(m)
            def gs(s):
                mm = m+s[:, :1]*t1+s[:, 1:]*t2
                return gfun(phi, (mm/mm.norm(dim=1, keepdim=True))[:, None])[:, 0]
            d1 = lambda s: torch.func.jvp(gs, (s,), (E1,))[1]
            d2 = lambda s: torch.func.jvp(gs, (s,), (E2,))[1]
            s0 = torch.zeros_like(E1)
            h11 = torch.func.jvp(d1, (s0,), (E1,))[1].detach(); h12 = torch.func.jvp(d1, (s0,), (E2,))[1].detach()
            h22 = torch.func.jvp(d2, (s0,), (E2,))[1].detach()
            gr = torch.stack((d1(s0), d2(s0)), 1).detach()
            with torch.no_grad():
                gnorm = gr.norm(dim=1)
                det = h11*h22-h12**2
                pd = (h11 > 0) & (det > 0)
                sN = -torch.stack((h22*gr[:, 0]-h12*gr[:, 1], h11*gr[:, 1]-h12*gr[:, 0]), 1)/torch.where(pd, det, torch.ones_like(det))[:, None]
                # converged: gradient at 1e-9, or the Newton decrement 1/2 g^T H^-1 g at 1e-16 (gradient ~1e-8: above the
                # quadrature noise of the gradient, ~4e-9, from the boundary nodes moving with the point)
                conv = have & ((gnorm <= tol*scale) | (pd & (-.5*(gr*sN).sum(1) <= 1e-16*scale)))
                if bool((conv | ~have | stuck).all()):
                    break
                # the boundary decides: no lower than the arcs here, and the surface curves away or the full Newton step
                # leaves the region (for a surface bulging toward the plane the lowest point is then on the boundary)
                mf = m+sN[:, :1]*t1+sN[:, 1:]*t2
                leave = ~self.inside(mf/mf.norm(dim=1, keepdim=True))
                why = dict(both=~pd | leave, leave=pd & leave, away=~pd, none=torch.zeros_like(pd))[rule]
                stuck |= have & ~conv & (g >= g_arc) & why
                step = torch.where(pd[:, None], sN, -gr/gnorm.clamp(min=1e-300)[:, None]*rad[:, None])
                sl = step.norm(dim=1)
                step = step*torch.where(sl > rad, rad/sl.clamp(min=1e-300), torch.ones_like(sl))[:, None]
                todo = have & ~conv & ~stuck
                acc = torch.zeros(P, dtype=torch.bool, device=x.device)
                mnew, gnew = m.clone(), g.clone()
                for _ in range(40):
                    mt = m+step[:, :1]*t1+step[:, 1:]*t2; mt = mt/mt.norm(dim=1, keepdim=True)
                    gt = gfun(self.I.make_phi(mt[:, None])[0], mt[:, None])[:, 0]
                    ok = todo & ~acc & self.inside(mt) & (gt <= g+1e-4*(gr*step).sum(1))
                    mnew = torch.where(ok[:, None], mt, mnew); gnew = torch.where(ok, gt, gnew); acc |= ok
                    if bool((acc | ~todo).all()):
                        break
                    step = torch.where((todo & ~acc)[:, None], .5*step, step)
                stuck |= todo & ~acc                     # no admissible decrease: held at the boundary (the arcs decide)
                rad = torch.where(acc, torch.maximum(rad, 2*(mnew-m).norm(dim=1)).minimum(self.rad), rad)
                m, g = mnew, gnew
        with torch.no_grad():
            Wi = self.weights(m[:, None])[:, 0]
            arc = ~have | (g_arc < g)
            th = .5*(self.a0+self.a1)+pha                                                      # (P, Bmax)
            ar = torch.arange(P, device=x.device)
            thb = th[ar, ba]
            ma = self.arcpt(th[..., None])[ar, ba, 0]
            Lb = lagrange3(thb, self.a0[ar, ba], self.a1[ar, ba])                               # (P, 3)
            Wa = torch.zeros_like(Wi).scatter_(1, self.slot3[ar, ba], Lb)
            gap = torch.where(arc, g_arc, g)
            out = dict(gap=gap, m=torch.where(arc[:, None], ma, m), W=torch.where(arc[:, None], Wa, Wi), kind=arc.long(),
                       g_interior=torch.where(have, g, torch.full_like(g, float('inf'))), g_arc=g_arc, converged=conv, stuck=stuck,
                       grad_rel=gnorm/scale, iterations=it+1)
        return out

    def gap(self, x, y, n, c, low):
        """the gaps at the held lowest points low (from lowest), differentiable in x, y"""
        return self.points(x, y, low['m'][:, None], low['W'][:, None])[:, 0]@n-c
