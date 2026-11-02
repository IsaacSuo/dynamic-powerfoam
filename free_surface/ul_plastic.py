"""Plasticity on the per-step cut (2026-10-31; the user: UL's welding shows it suits plasticity, its elastic body is another
concept). No rest shape and no permanent neighbours: connection is the current cut's walls, and each cell keeps only its elastic
state, a deviatoric Hencky strain e and L = ln J. Bonds, the outer layer and the colours are not here (the cell energy takes the
whole V0); the permanent bonds stay ClosureWarp's, for elastic bodies. Moduli per unit density, masses = volumes, as the others.
A step from n to n + 1, backward Euler (MidpointStep at theta = 1):
  cut       radii for V = V0 exp(L_n), walls (i, j): area A, centroid c, n from i to j, L = |x_j - x_i|, t = (c - x_i).n / L,
            tau = c - (x_i + t (x_j - x_i))
  gradient  'gg' (divergence theorem, ul-tet's gg_a on the current cut): V_s G_s = sum_walls A u_w (x) n_out + u_s (x) a_s + G0_s M_s,
            u_w = (1 - t) u_i + t u_j + 1/2 (G0_i + G0_j) tau shared by the two cells, so a uniform Cauchy stress gives no force at
            an interior site; a_s = -sum A n_out, M_s = V_s I - sum A (c - x_s) (x) n_out (0 and 0 without an exposed cap). G0 the
            sites' least-squares fit (tiny cells: their neighbours' mean). 'fit': G = G0 (for comparison: fails that test)
  strain    frozen for the step: Q the rotation of I + G at Newton's start. H = sym(Q^T (I + G)) - I, linear in x. e_tr = e_n + dev H,
            L = L_n + tr H: rotation-free to second order in the rotation's error, and no det F, whose second-order term gives the
            negative curvature -p |W|^2 that per-cell fits do not cancel
  energy    Pi = inertia + sum V0 (W(e_tr) + kap / 2 L^2) + variance + jump + ground. W the condensed von Mises potential: mu |e|^2 up to
            2 mu |e| = s_y, then s_y |e| - s_y^2 / (4 mu), s_y = (2 / 3)^1/2 sigma_y (as PAC-NeRF's return). Each term is convex
            in x, so the step's only non-convexity is the ground's: the geometry's nonlinearity lives in the re-cut
  variance  kap A L / 24 (L_i - L_j)^2 on the current walls (the J field's variation inside the cells)
  jump      the sites' relative motion the gradients do not explain: dt = (u_j - u_i) - 1/2 (G_i + G_j)(x_j - x_i), a sub-cell strain gamma = delta / L over the wall's
            volume A L / 3, so k = mu A / (3 L). It has no yield of its own: in a flowing cell D = Dm + Df with mean Df = 0, s_y |D| averages to s_y |Dm| + s_y <|Df_perp|^2> / (2 |Dm|),
            so the fluctuation's dissipation is second order, a viscosity s_y / (2 |Dm|); a force limit on the wall would add first-order friction, the yield counted twice (10-31: a
            limit tau_y A took 29 % of the plastic work, tau_y A / 3 20 %, and each sliding wall starts each step on its knee: Newton failed). The wall follows its cells' return:
            E = 1/2 k beta |delta_n + dt|^2, delta_n+1 = beta (delta_n + dt), beta the mean of the two cells' min(1, s_y / (2 mu |e_tr|)), frozen at Newton's start. Quadratic in x.
            beta 1: the elastic memory; fast flow: force (A / 3 L)(s_y / 2 |D|)[v], independent of dt. A new wall starts at 0, a wall gone takes its memory along (its area is 0 by then)
  update    at x^n+1: e, delta returned to their limits and turned by Q, L_n+1 = L_n + ln det(I + G) (exact, not the step's linear tr H)
Not here: the stabilisation S-NSNI, the normal part of UL's jump (lam + mu), rendering's F, flow's other materials."""
import math
from types import SimpleNamespace

import numpy as np
import torch
import warp as wp

from free_surface.closure_warp import MidpointStep, k_Aul, k_Gul, k_Gul_adj, k_ball, k_ball_adj
from free_surface.recut import Recut
from free_surface.skin_warp import NSLOT, NT, k_total, k_total_loss

f64 = torch.float64


@wp.kernel(enable_backward=False)
def k_Ge(G0: wp.array(dtype=wp.mat33d), Ga: wp.array(dtype=wp.mat33d), used: wp.array(dtype=int), Ge: wp.array(dtype=wp.mat33d)):
    b = wp.tid()
    Ge[b] = G0[b]
    if used[b] != 0:
        Ge[b] = Ga[b]


@wp.kernel(enable_backward=False)
def k_gg_wall(x: wp.array(dtype=wp.vec3d), xn: wp.array(dtype=wp.vec3d), wi: wp.array(dtype=int), wj: wp.array(dtype=int), wa: wp.array(dtype=wp.float64),
              wn: wp.array(dtype=wp.vec3d), wt: wp.array(dtype=wp.float64), wtau: wp.array(dtype=wp.vec3d), Ge: wp.array(dtype=wp.mat33d), Sg: wp.array(dtype=wp.mat33d)):
    e = wp.tid()
    i, j = wi[e], wj[e]
    if i == j:
        return
    t = wt[e]
    uw = (wp.float64(1.0)-t)*(x[i]-xn[i])+t*(x[j]-xn[j])+wp.float64(0.5)*((Ge[i]+Ge[j])@wtau[e])
    M = wa[e]*wp.outer(uw, wn[e])
    wp.atomic_add(Sg, i, M)
    wp.atomic_add(Sg, j, -M)


@wp.kernel(enable_backward=False)
def k_gg_cell(x: wp.array(dtype=wp.vec3d), xn: wp.array(dtype=wp.vec3d), Sg: wp.array(dtype=wp.mat33d), Ge: wp.array(dtype=wp.mat33d), a: wp.array(dtype=wp.vec3d),
              Mc: wp.array(dtype=wp.mat33d), Vi: wp.array(dtype=wp.float64), G: wp.array(dtype=wp.mat33d)):
    b = wp.tid()
    G[b] = (Sg[b]+wp.outer(x[b]-xn[b], a[b])+Ge[b]@Mc[b])*Vi[b]


@wp.kernel(enable_backward=False)
def k_cell(G: wp.array(dtype=wp.mat33d), Q: wp.array(dtype=wp.mat33d), en: wp.array(dtype=wp.mat33d), Ln: wp.array(dtype=wp.float64), V0: wp.array(dtype=wp.float64),
           par: wp.array(dtype=wp.float64), L: wp.array(dtype=wp.float64), enrm: wp.array(dtype=wp.float64), Gb: wp.array(dtype=wp.mat33d), terms: wp.array2d(dtype=wp.float64)):
    """par = (mu, kap, s_y); Gb = dE/dG so far"""
    b = wp.tid()
    I = wp.identity(n=3, dtype=wp.float64)
    A = wp.transpose(Q[b])@(I+G[b])
    H = wp.float64(0.5)*(A+wp.transpose(A))-I
    tr = wp.trace(H)
    Lb = Ln[b]+tr
    L[b] = Lb
    e = en[b]+H-(tr/wp.float64(3.0))*I
    nr = wp.sqrt(wp.ddot(e, e))
    enrm[b] = nr
    mu, kap, sy = par[0], par[1], par[2]
    w = mu*nr*nr
    s = (wp.float64(2.0)*mu)*e
    if wp.float64(2.0)*mu*nr > sy:
        w = sy*nr-sy*sy/(wp.float64(4.0)*mu)
        s = (sy/nr)*e
    wp.atomic_add(terms, 0, b % NSLOT, V0[b]*w)
    wp.atomic_add(terms, 1, b % NSLOT, V0[b]*wp.float64(0.5)*kap*Lb*Lb)
    Gb[b] = V0[b]*(Q[b]@(s+(kap*Lb)*I))


@wp.kernel(enable_backward=False)
def k_var(wi: wp.array(dtype=int), wj: wp.array(dtype=int), wv: wp.array(dtype=wp.float64), L: wp.array(dtype=wp.float64), Lb: wp.array(dtype=wp.float64), terms: wp.array2d(dtype=wp.float64)):
    e = wp.tid()
    i, j = wi[e], wj[e]
    if i == j:
        return
    dl = L[i]-L[j]
    wp.atomic_add(terms, 2, e % NSLOT, wv[e]*dl*dl)
    g = wp.float64(2.0)*wv[e]*dl
    wp.atomic_add(Lb, i, g)
    wp.atomic_add(Lb, j, -g)


@wp.kernel(enable_backward=False)
def k_Lb(Lb: wp.array(dtype=wp.float64), Q: wp.array(dtype=wp.mat33d), Gb: wp.array(dtype=wp.mat33d)):
    """dL/dG = Q"""
    b = wp.tid()
    Gb[b] = Gb[b]+Lb[b]*Q[b]


@wp.kernel(enable_backward=False)
def k_jump(x: wp.array(dtype=wp.vec3d), xn: wp.array(dtype=wp.vec3d), G: wp.array(dtype=wp.mat33d), wi: wp.array(dtype=int), wj: wp.array(dtype=int), wd: wp.array(dtype=wp.vec3d),
           wdn: wp.array(dtype=wp.vec3d), wk: wp.array(dtype=wp.float64), wb: wp.array(dtype=wp.float64), wdl: wp.array(dtype=wp.vec3d), terms: wp.array2d(dtype=wp.float64),
           Gb: wp.array(dtype=wp.mat33d), x_adj: wp.array(dtype=wp.vec3d)):
    e = wp.tid()
    i, j = wi[e], wj[e]
    if i == j:
        return
    half = wp.float64(0.5)
    dv = wdn[e]+(x[j]-xn[j])-(x[i]-xn[i])-half*((G[i]+G[j])@wd[e])
    wdl[e] = dv
    k = wk[e]*wb[e]
    f = k*dv
    wp.atomic_add(terms, 3, e % NSLOT, half*wp.dot(f, dv))
    wp.atomic_add(x_adj, j, f)
    wp.atomic_add(x_adj, i, -f)
    PF = -half*wp.outer(f, wd[e])
    wp.atomic_add(Gb, i, PF)
    wp.atomic_add(Gb, j, PF)


@wp.kernel(enable_backward=False)
def k_gg_cell_adj(Gb: wp.array(dtype=wp.mat33d), a: wp.array(dtype=wp.vec3d), Mc: wp.array(dtype=wp.mat33d), Vi: wp.array(dtype=wp.float64), x_adj: wp.array(dtype=wp.vec3d),
                  Gs: wp.array(dtype=wp.mat33d), Gbe: wp.array(dtype=wp.mat33d)):
    """(dE/dG) / V: the walls' flux gets it, the cap's u_s (x) a_s and G0 M_s directly"""
    b = wp.tid()
    Gh = Gb[b]*Vi[b]
    Gs[b] = Gh
    wp.atomic_add(x_adj, b, Gh@a[b])
    wp.atomic_add(Gbe, b, Gh@wp.transpose(Mc[b]))


@wp.kernel(enable_backward=False)
def k_gg_wall_adj(wi: wp.array(dtype=int), wj: wp.array(dtype=int), wa: wp.array(dtype=wp.float64), wn: wp.array(dtype=wp.vec3d), wt: wp.array(dtype=wp.float64), wtau: wp.array(dtype=wp.vec3d),
                  Gs: wp.array(dtype=wp.mat33d), x_adj: wp.array(dtype=wp.vec3d), Gbe: wp.array(dtype=wp.mat33d)):
    e = wp.tid()
    i, j = wi[e], wj[e]
    if i == j:
        return
    t = wt[e]
    ub = wa[e]*((Gs[i]-Gs[j])@wn[e])
    wp.atomic_add(x_adj, i, (wp.float64(1.0)-t)*ub)
    wp.atomic_add(x_adj, j, t*ub)
    M = wp.float64(0.5)*wp.outer(ub, wtau[e])
    wp.atomic_add(Gbe, i, M)
    wp.atomic_add(Gbe, j, M)


@wp.kernel(enable_backward=False)
def k_Ge_adj(Gbe: wp.array(dtype=wp.mat33d), used: wp.array(dtype=int), Gb0: wp.array(dtype=wp.mat33d)):
    b = wp.tid()
    if used[b] == 0:
        wp.atomic_add(Gb0, b, Gbe[b])


@wp.kernel(enable_backward=False)
def k_Aul_adj2(Gbe: wp.array(dtype=wp.mat33d), As: wp.array(dtype=int), Ao: wp.array(dtype=int), Aw: wp.array(dtype=wp.float64), Gb0: wp.array(dtype=wp.mat33d)):
    e = wp.tid()
    if Aw[e] == wp.float64(0.0):
        return
    wp.atomic_add(Gb0, Ao[e], Aw[e]*Gbe[As[e]])


def polar_batch(F, its=40):
    """the nearest rotations of F (n, 3, 3) by Newton's iteration, to its rounding; I where det F <= 0.1 (a fit gone bad: no rotation is read from it)"""
    I3 = torch.eye(3, dtype=F.dtype, device=F.device)
    ok = torch.linalg.det(F) > .1
    R = torch.where(ok[:, None, None], F, I3)
    for _ in range(its):
        R1 = .5*(R+torch.linalg.inv(R).transpose(1, 2))
        ch = float((R1-R).abs().max())
        R = R1
        if ch < 1e-14:
            break
    return R


class PlasticUL:
    """X, r, V0: the foam at rest (sites, radii, volumes), E, nu, sigma_y per unit density (sigma_y inf: no yield). grad 'gg' or 'fit'.
    The interface MidpointStep and TrustCG need; update_ul(x) at each step's start, set_rotation(z) at Newton's start (PlasticStep),
    end_ul at its end. Kw: the walls' capacity (default 3 times the rest walls; padding costs next to nothing; over it update_ul stops)"""
    iters = 3
    linear_closure = False
    vis = selfc = None
    cap_ul, cap_eps = True, 0.
    smu = 0.
    visc_weight = 1.

    def __init__(self, X, r, V0, E, nu, sigma_y, grad='gg', agg=.1, vol_tol=1e-5, Kw=None, dev='cuda'):
        wp.init()
        self.n = n = len(X); self.dev = dev; self.grad_kind = grad
        self.X = torch.as_tensor(X, dtype=f64, device=dev).contiguous()
        self.V0 = torch.as_tensor(V0, dtype=f64, device=dev).contiguous()
        self._rr0 = torch.as_tensor(r, dtype=f64, device=dev).contiguous()
        self._ulr = self._rr0.clone()                                               # (not the same tensor: a state's load wrote its cut radii into the rest ones, 11-02)
        self.st = SimpleNamespace(X=self.X, V=self.V0)
        self.mu, self.lam = E/(2*(1+nu)), E*nu/((1+nu)*(1-2*nu))
        self.kap = self.lam+2*self.mu/3
        self.sy = math.sqrt(2/3)*sigma_y
        self.tau_y = sigma_y/math.sqrt(3)
        self.stick = False                                                          # (set_stick: the glue's limit from the material's shear yield)
        self._ul_agg, self._ul_tol = agg, vol_tol
        self._recut = Recut(n, dev); self._recut.A.f32 = True
        G, _ = self._recut.geometry(self.X, self._ulr)
        nb = int((G.wall_A.detach() > 0).sum())
        Kw = 3*nb if Kw is None else Kw
        z = lambda k, dt_: wp.zeros(k, dtype=dt_, device=dev)
        self.ul = dict(Kw=Kw, xn=z(n, wp.vec3d), S=z(2*Kw, wp.int32), O=z(2*Kw, wp.int32), R=z(2*Kw, wp.vec3d), As=z(Kw, wp.int32), Ao=z(Kw, wp.int32), Aw=z(Kw, wp.float64), used=z(n, wp.int32),
                       wi=z(Kw, wp.int32), wj=z(Kw, wp.int32), wa=z(Kw, wp.float64), wn=z(Kw, wp.vec3d), wt=z(Kw, wp.float64), wtau=z(Kw, wp.vec3d), wd=z(Kw, wp.vec3d), wdn=z(Kw, wp.vec3d),
                       wk=z(Kw, wp.float64), wb=z(Kw, wp.float64), wv=z(Kw, wp.float64), a=z(n, wp.vec3d), Mc=z(n, wp.mat33d), Vi=z(n, wp.float64), Q=z(n, wp.mat33d), en=z(n, wp.mat33d),
                       Ln=z(n, wp.float64), V0=wp.from_torch(self.V0, dtype=wp.float64))
        tt = lambda a_: wp.to_torch(a_, requires_grad=False)
        self.tul = {k_: tt(a_) for k_, a_ in self.ul.items() if k_ not in ('Kw', 'V0')}
        self.I3 = torch.eye(3, dtype=f64, device=dev)
        self.tul['xn'].copy_(self.X); self.tul['Q'].copy_(self.I3.expand(n, 3, 3))
        self.par = wp.array(np.array([self.mu, self.kap, self.sy]), dtype=wp.float64, device=dev)
        self.ep = torch.zeros(n, 3, 3, dtype=f64, device=dev)                       # (the accumulated plastic strain, a passive record)
        self.Wp = self.Wp_cell = 0.                                                 # (the plastic work so far: all, and the cells' share)
        self.Hsum = torch.zeros(3, 3, dtype=f64, device=dev)
        self.ul_stats = dict(radii_fail=0, floored=0, walls=0, radii_it=0, yielding=0, wall_yielding=0, flips=0)
        self._mem_key = torch.zeros(0, dtype=torch.long, device=dev); self._mem_d = torch.zeros(0, 3, dtype=f64, device=dev)
        self._ball_moduli(E, nu)
        self.ball = None

    def _ball_moduli(self, E, nu):
        self.Es, self.Gs = E/(1-nu*nu), self.mu/(2-nu)

    def set_ball(self, normal=(0., 1., 0.), c0=0., mu=0.):
        """Hertz-Mindlin on the balls of the cells with an exposed cap at the step's start, before prepare(): one slot per cell, the radius
        this step's (update_ul; -1 without a cap: it never touches), so a cell keeps its glue's held force"""
        n, dev = self.n, self.dev
        z = lambda dt_: wp.zeros(n, dtype=dt_, device=dev)
        self.ball = dict(k=n, body=wp.from_torch(torch.arange(n, device=dev).to(torch.int32), dtype=wp.int32), R=z(wp.float64),
                         ng=wp.vec3d(*[float(a_) for a_ in normal]), par=wp.array(np.array([1., 1., float(c0), self.Es, float(mu)]), dtype=wp.float64, device=dev),
                         xn=z(wp.vec3d), kt=z(wp.float64), bound=z(wp.float64), lamt=z(wp.vec3d))
        tt = lambda a_: wp.to_torch(a_, requires_grad=False)
        self.tball, self.tbR, self.tbbound = tt(self.ball['par']), tt(self.ball['R']), tt(self.ball['bound'])
        self.tbR.fill_(-1.)

    def set_stick(self):
        """the ground no-slip for this material, after set_ball: a no-slip wall under a perfectly plastic material shears in a layer far thinner than a cell (the MPM: zero velocity at
        the ground, the lowest particles a few mm above it, moving), so the wall's traction is tau_y, von Mises friction m = 1, whatever the pressure. So the glue's limit is tau_y A_c, A_c the
        cell's own footprint (footprint), in place of mu F_n. The step's friction switch tball[4] (> 0) only turns the glue on; its stiffness stays Mindlin's"""
        self.stick = True

    def footprint(self, x, r, g=None):
        """each touching cell's plan view: the column over a point y of the plane goes to the touching cell of least power |y - p|^2 - r^2 (p the site's projection), so the plane is cut by the 2-d power
        diagram of the circles of radius r; the areas add up to the union of the discs, nothing counted twice. Not the ball cut by the ground (r^2 - d^2): the balls are bumps touching with their tips, discs of 5-6 mm
        on cells of 15 mm, a footprint 2-6 times the MPM's contact area too small (10-31), while the material under a site reaches the ground all across its section. On a raster of spacing g (default a quarter of the median cell size)"""
        A = torch.zeros(self.n, dtype=f64, device=x.device); ng = torch.tensor([self.ball['ng'][k_] for k_ in range(3)], dtype=f64, device=x.device)
        d = x@ng-float(self.tball[2])
        c = torch.nonzero((d < r) & (d > -r)).flatten()
        if len(c) == 0:
            return A
        k0 = int(torch.argmin(ng.abs())); e0 = torch.zeros(3, dtype=f64, device=x.device); e0[k0] = 1.
        u = torch.linalg.cross(ng, e0); u = u/u.norm(); v = torch.linalg.cross(ng, u)
        rc = r[c]; ac = rc*rc; P = torch.stack([x[c]@u, x[c]@v], 1)
        g = float(self.V0.median())**(1/3)/4 if g is None else g
        lo, hi = (P-rc[:, None]).min(0).values, (P+rc[:, None]).max(0).values
        gx = torch.arange(float(lo[0])+g/2, float(hi[0]), g, dtype=f64, device=x.device); gy = torch.arange(float(lo[1])+g/2, float(hi[1]), g, dtype=f64, device=x.device)
        Y = torch.stack(torch.meshgrid(gx, gy, indexing='ij'), -1).reshape(-1, 2)
        cnt = torch.zeros(len(c), dtype=f64, device=x.device)
        for s in range(0, len(Y), 8192):
            pw = torch.cdist(Y[s:s+8192].float(), P.float())**2-ac.float()[None]
            mn, k = pw.min(1)
            cnt.index_add_(0, k[mn <= 0], torch.ones(int((mn <= 0).sum()), dtype=f64, device=x.device))
        A[c] = cnt*g*g
        return A

    def prepare(self):
        n, dev = self.n, self.dev
        z = lambda dt_: wp.zeros(n, dtype=dt_, device=dev)
        Kw = self.ul['Kw']
        self.B = dict(G0=z(wp.mat33d), Ga=z(wp.mat33d), Ge=z(wp.mat33d), Sg=z(wp.mat33d), G=z(wp.mat33d), Gb=z(wp.mat33d), Gs=z(wp.mat33d), Gbe=z(wp.mat33d), Gb0=z(wp.mat33d),
                      L=z(wp.float64), Lb=z(wp.float64), enrm=z(wp.float64), R=z(wp.mat33d), wdl=wp.zeros(Kw, dtype=wp.vec3d, device=dev), terms=wp.zeros((NT, 256), dtype=wp.float64, device=dev),
                      tsum=wp.zeros(NT, dtype=wp.float64, device=dev), loss=wp.zeros(1, dtype=wp.float64, device=dev), bdelta=z(wp.float64), bfn=z(wp.float64), bff=z(wp.vec3d))
        self.tw = wp.array(np.ones(NT), dtype=wp.float64, device=dev)
        self.tones = wp.array(np.ones((NT, 256)), dtype=wp.float64, device=dev)
        self.xw = wp.zeros(n, dtype=wp.vec3d, device=dev, requires_grad=True)
        tt = lambda a_: wp.to_torch(a_, requires_grad=False)
        self.tx, self.tgx, self.tloss, self.tR, self.tG, self.tL, self.tenrm, self.twdl, self.tsum = (tt(a_) for a_ in (self.xw, self.xw.grad, self.B['loss'], self.B['R'], self.B['G'], self.B['L'],
                                                                                                                      self.B['enrm'], self.B['wdl'], self.B['tsum']))
        self.tR.copy_(self.I3.expand(n, 3, 3))
        self.tx.copy_(self.X)
        self.run()
        wp.capture_begin()
        self.run()
        self.graph = wp.capture_end()

    def run(self):
        """E and dE/dx at xw: the forward kernels, then the adjoints written out"""
        B, U, n, Kw = self.B, self.ul, self.n, self.ul['Kw']
        x, g = self.xw, self.xw.grad
        for k_ in ('G0', 'Ga', 'Sg', 'Lb', 'Gbe', 'Gb0', 'terms', 'tsum'):
            B[k_].zero_()
        g.zero_()
        wp.launch(k_Gul, 2*Kw, inputs=[x, U['xn'], U['S'], U['O'], U['R']], outputs=[B['G0']])
        wp.launch(k_Aul, Kw, inputs=[B['G0'], U['As'], U['Ao'], U['Aw']], outputs=[B['Ga']])
        wp.launch(k_Ge, n, inputs=[B['G0'], B['Ga'], U['used']], outputs=[B['Ge']])
        wall = [U['wi'], U['wj'], U['wa'], U['wn'], U['wt'], U['wtau']]
        wp.launch(k_gg_wall, Kw, inputs=[x, U['xn']]+wall+[B['Ge']], outputs=[B['Sg']])
        wp.launch(k_gg_cell, n, inputs=[x, U['xn'], B['Sg'], B['Ge'], U['a'], U['Mc'], U['Vi']], outputs=[B['G']])
        wp.launch(k_cell, n, inputs=[B['G'], U['Q'], U['en'], U['Ln'], U['V0'], self.par], outputs=[B['L'], B['enrm'], B['Gb'], B['terms']])
        wp.launch(k_var, Kw, inputs=[U['wi'], U['wj'], U['wv'], B['L']], outputs=[B['Lb'], B['terms']])
        wp.launch(k_Lb, n, inputs=[B['Lb'], U['Q']], outputs=[B['Gb']])
        wp.launch(k_jump, Kw, inputs=[x, U['xn'], B['G'], U['wi'], U['wj'], U['wd'], U['wdn'], U['wk'], U['wb']], outputs=[B['wdl'], B['terms'], B['Gb'], g])
        bl = self.ball
        if bl is not None:
            wp.launch(k_ball, bl['k'], inputs=[x, bl['body'], bl['R'], bl['xn'], bl['ng'], bl['par'], bl['kt'], bl['bound'], bl['lamt']], outputs=[B['bdelta'], B['bfn'], B['bff'], B['terms']])
            wp.launch(k_ball_adj, bl['k'], inputs=[bl['body'], bl['ng'], bl['par'], B['bfn'], B['bff'], self.tones], outputs=[g])
        wp.launch(k_gg_cell_adj, n, inputs=[B['Gb'], U['a'], U['Mc'], U['Vi']], outputs=[g, B['Gs'], B['Gbe']])
        wp.launch(k_gg_wall_adj, Kw, inputs=wall+[B['Gs']], outputs=[g, B['Gbe']])
        wp.launch(k_Ge_adj, n, inputs=[B['Gbe'], U['used']], outputs=[B['Gb0']])
        wp.launch(k_Aul_adj2, Kw, inputs=[B['Gbe'], U['As'], U['Ao'], U['Aw']], outputs=[B['Gb0']])
        wp.launch(k_Gul_adj, 2*Kw, inputs=[B['Gb0'], U['S'], U['O'], U['R']], outputs=[g])
        wp.launch(k_total, (NT, 256), inputs=[B['terms']], outputs=[B['tsum']])
        wp.launch(k_total_loss, 1, inputs=[B['tsum'], self.tw], outputs=[B['loss']])

    def _run(self):
        self.run()

    def grad(self, x):
        self.tx.copy_(x)
        wp.capture_launch(self.graph)
        return self.tloss[0].clone(), self.tgx.clone()

    def cold(self):
        pass

    def save_closure(self):
        pass

    def total_mode(self):
        pass

    def update_ul(self, x):
        """the step's start at x: radii for V0 exp(L_n) (their failure keeps the last radii: they only shape the cut), the cut, the fit's
        entries, the walls' and caps' data, each wall's memory found by its pair"""
        n = self.n; x = x.detach(); st = self.ul_stats; t = self.tul; dev = x.device
        J = torch.exp(t['Ln']); st['floored'] = int((J < .02).sum())
        try:
            r, G, S_, it, nrep = self._recut.solve_radii(x, self._ulr, self.V0*J.clamp(min=.02), tol=self._ul_tol, r_base=self._rr0)
            st['radii_it'] = it
        except RuntimeError:
            st['radii_fail'] += 1
            r = self._ulr; G, S_ = self._recut.geometry(x, r)
        self._ulr, self._ulG = r, G
        kp = G.wall_A.detach() > 0
        i, j, A, c = G.i[kp].long(), G.j[kp].long(), G.wall_A.detach()[kp], G.wall_c.detach()[kp]
        m = len(i); U = self.ul
        if m > U['Kw']:
            raise RuntimeError('update_ul: %d walls over the capacity %d' % (m, U['Kw']))
        st['walls'] = m
        V = G.V.detach(); self._V = V
        d = x[j]-x[i]; Lw = d.norm(dim=1); nn = d/Lw[:, None]; w = A/Lw
        q = w[:, None, None]*d[:, :, None]*d[:, None, :]
        M = torch.zeros(n, 3, 3, dtype=f64, device=dev).index_add(0, i, q).index_add(0, j, q)
        a0, a1, a2 = M[:, :, 0], M[:, :, 1], M[:, :, 2]; c0, c1, c2 = torch.linalg.cross(a1, a2), torch.linalg.cross(a2, a0), torch.linalg.cross(a0, a1)   # (the inverse by cofactors; the SVD only near singular)
        det = (a0*c0).sum(1); sing = det.abs() <= 1e-10*(M*M).sum((1, 2))**1.5
        Mi = torch.stack([c0, c1, c2], 1)/torch.where(sing, torch.ones_like(det), det)[:, None, None]
        if bool(sing.any()):
            Mi[sing] = torch.linalg.pinv(M[sing])
        vd = w[:, None]*d
        Ri = torch.einsum('fa,fab->fb', vd, Mi[i]); Rj = torch.einsum('fa,fab->fb', -vd, Mi[j])
        t['S'].zero_(); t['O'].zero_(); t['R'].zero_()
        t['S'][:2*m] = torch.cat([i, j]).to(torch.int32); t['O'][:2*m] = torch.cat([j, i]).to(torch.int32); t['R'][:2*m] = torch.cat([Ri, Rj])
        tiny = self.V0 < self._ul_agg*self.V0.median()                              # (the aggregation, as ClosureWarp.update_ul)
        m1, m2 = tiny[i] & ~tiny[j], tiny[j] & ~tiny[i]
        as_, ao_ = torch.cat([i[m1], j[m2]]), torch.cat([j[m1], i[m2]])
        va = self.V0[ao_]
        den = torch.zeros(n, dtype=f64, device=dev).index_add(0, as_, va)
        t['As'].zero_(); t['Ao'].zero_(); t['Aw'].zero_()
        k_ = len(as_); t['As'][:k_] = as_.to(torch.int32); t['Ao'][:k_] = ao_.to(torch.int32); t['Aw'][:k_] = va/den[as_]
        t['used'].copy_((tiny & (den > 0)).to(torch.int32))
        tw = ((c-x[i])*nn).sum(1)/Lw
        cap = G.patch_A.detach() > 0
        a_ = torch.zeros(n, 3, dtype=f64, device=dev).index_add(0, i, -A[:, None]*nn).index_add(0, j, A[:, None]*nn)
        Mw = (torch.zeros(n, 3, 3, dtype=f64, device=dev).index_add(0, i, A[:, None, None]*(c-x[i])[:, :, None]*nn[:, None, :])
              .index_add(0, j, -A[:, None, None]*(c-x[j])[:, :, None]*nn[:, None, :]))
        Mc = V[:, None, None]*self.I3-Mw
        Mc[~cap] = 0.; a_[~cap] = 0.
        wa = A
        if self.grad_kind == 'fit':                                                 # (G = G0: no flux, the whole cell on the fit)
            wa = torch.zeros_like(A); a_ = torch.zeros_like(a_); Mc = V[:, None, None]*self.I3
        for k2 in ('wi', 'wj', 'wa', 'wn', 'wt', 'wtau', 'wd', 'wdn', 'wk', 'wb', 'wv'):
            t[k2].zero_()
        t['wi'][:m] = i.to(torch.int32); t['wj'][:m] = j.to(torch.int32); t['wa'][:m] = wa; t['wn'][:m] = nn; t['wt'][:m] = tw
        t['wtau'][:m] = c-(x[i]+tw[:, None]*d); t['wd'][:m] = d
        t['wk'][:m] = self.mu*A/(3*Lw); t['wb'][:m] = 1.; t['wv'][:m] = self.kap*A*Lw/24
        lo, hi = torch.minimum(i, j), torch.maximum(i, j); key = lo*n+hi; sg = torch.where(i < j, 1., -1.).to(f64)[:, None]
        if len(self._mem_key):
            pos = torch.searchsorted(self._mem_key, key).clamp(max=len(self._mem_key)-1)
            found = self._mem_key[pos] == key
            t['wdn'][:m][found] = self._mem_d[pos[found]]*sg[found]
        t['a'].copy_(a_); t['Mc'].copy_(Mc); t['Vi'].copy_(1/V.clamp(min=1e-12*float(V.median())))
        t['xn'].copy_(x)
        if self.ball is not None:
            self.tbR.copy_(torch.where(cap, r, -torch.ones_like(r)))
        self._w = (i, j, Lw, key, sg, m)
        t['Q'].copy_(self.I3.expand(n, 3, 3))

    def set_rotation(self, z):
        """the step's frozen rotation, of I + G at Newton's start z (Q = I for that evaluation), and the walls' beta from the cells' trial strains there"""
        self.grad(z)
        self.tul['Q'].copy_(polar_batch(self.I3+self.tG))
        self.grad(z)
        i, j, Lw, key, sg, m = self._w
        b = self._beta(self.tenrm)
        self.tul['wb'][:m] = .5*(b[i]+b[j])

    def _beta(self, nr):
        return torch.minimum(torch.ones_like(nr), self.sy/(2*self.mu*nr).clamp(min=1e-300))

    def end_ul(self, th=1.):
        """after the step (the buffers at the accepted x^n+1): the returns and the update"""
        assert th == 1., 'the return belongs to the step end: theta 1 only'
        t = self.tul; Q, G = t['Q'], self.tG; I3 = self.I3
        A = Q.transpose(1, 2)@(I3+G); H = .5*(A+A.transpose(1, 2))-I3
        tr = H.diagonal(dim1=1, dim2=2).sum(1)
        e = t['en']+H-(tr/3)[:, None, None]*I3
        nr = e.flatten(1).norm(dim=1)
        sc = self._beta(nr)
        er = sc[:, None, None]*e
        dep = e-er
        wc = float((self.V0*self.sy*dep.flatten(1).norm(dim=1)).sum()) if math.isfinite(self.sy) else 0.
        self.Wp += wc; self.Wp_cell += wc
        self.Hsum += (self.V0[:, None, None]*H).sum(0)/self.V0.sum()                # (the volume mean of the step's strain, in the world's frame: passive)
        Qt = Q.transpose(1, 2)
        t['en'].copy_(Q@er@Qt)
        self.ep = Q@(self.ep+dep)@Qt
        Fr = I3+G; det = torch.linalg.det(Fr); flip = det <= 1e-3
        self.ul_stats['flips'] = int(flip.sum()); self.ul_stats['yielding'] = int((sc < 1).sum())
        t['Ln'].add_(torch.where(flip, tr, torch.log(det.clamp(min=1e-3))))
        i, j, Lw, key, sg, m = self._w
        dl = self.twdl[:m]; nd = dl.norm(dim=1); k, wb = t['wk'][:m], t['wb'][:m]
        self.ul_stats['wall_yielding'] = int((wb < 1).sum())
        self.Wp += float((.5*k*wb*(1-wb)*nd*nd).sum())                              # (what the step's energy held less what the memory keeps)
        be = .5*(sc[i]+sc[j])-wb                                                     # (the frozen beta against the cells' at the step's end)
        self.ul_stats['beta_err'] = (float(be.abs().mean()), float(be.abs().max()))
        Qf = .5*(Q[i]+Q[j])
        dn = torch.einsum('fab,fb->fa', Qf, wb[:, None]*dl)*sg                       # (the memory oriented low index -> high)
        q_ = dn.norm(dim=1)/Lw
        self.ul_stats['delta_L'] = (float(q_.median()), float(torch.quantile(q_, .99)), float(q_.max()))
        o = torch.argsort(key); self._mem_key = key[o]; self._mem_d = dn[o]

    def stored(self):
        """the elastic energy after the returns of the last end_ul: cells (deviatoric, volumetric), variance and jump at the state as it is"""
        t = self.tul
        return float((self.V0*self.mu*(t['en'].flatten(1)**2).sum(1)).sum()), float((self.V0*.5*self.kap*t['Ln']**2).sum())

    def ul_blocks(self, lr_thr=None, lr_cap=0):
        """the diagonal blocks for the preconditioner, at the step's start (a preconditioner: the walls' direct weights, the cap's, the fit's only through M_s; the tangential
        offsets' G0 and the variance left out). G_s = sum_k u_k (x) r_sk: V0 (mu_t (|r|^2 I + r r^T / 3) + kap r r^T), mu_t = s_y / (2 |e_tr|) on the yielding cells; the jump's k beta"""
        assert lr_thr is None, 'no low-rank term here'
        n = self.n; t = self.tul; dev = self.dev; i, j, Lw, key, sg, m = self._w
        A, tw, nn, Vi = t['wa'][:m], t['wt'][:m], t['wn'][:m], t['Vi']
        ci, cj = (A*Vi[i])[:, None]*nn, (A*Vi[j])[:, None]*nn
        s_ = [i, i, j, j]; k_ = [i, j, i, j]; r_ = [(1-tw)[:, None]*ci, tw[:, None]*ci, -(1-tw)[:, None]*cj, -tw[:, None]*cj]
        ids = torch.arange(n, device=dev)
        s_.append(ids); k_.append(ids); r_.append(Vi[:, None]*t['a'])
        S, O, R_ = t['S'][:2*m].long(), t['O'][:2*m].long(), t['R'][:2*m]
        cap = (t['Mc'].abs().sum((1, 2)) > 0)
        q = cap[S]
        rm = torch.einsum('fba,fb->fa', t['Mc'][S[q]], R_[q])*Vi[S[q]][:, None]
        s_ += [S[q], S[q]]; k_ += [O[q], S[q]]; r_ += [rm, -rm]
        sa, ka, ra = torch.cat(s_), torch.cat(k_), torch.cat(r_)
        u, inv = torch.unique(sa*n+ka, return_inverse=True)
        r = torch.zeros(len(u), 3, dtype=f64, device=dev).index_add(0, inv, ra); s, k = u//n, u % n
        nr = self.tenrm
        mut = torch.minimum(self.mu*torch.ones_like(nr), self.sy/(2*nr).clamp(min=1e-300))
        rr = r[:, :, None]*r[:, None, :]; r2 = (r*r).sum(1)
        K = torch.zeros(n, 3, 3, dtype=f64, device=dev)
        K.index_add_(0, k, (self.V0[s]*mut[s])[:, None, None]*(r2[:, None, None]*self.I3+rr/3)+(self.V0[s]*self.kap)[:, None, None]*rr)
        kt = (t['wk'][:m]*t['wb'][:m])[:, None, None]*self.I3
        K.index_add_(0, i, kt); K.index_add_(0, j, kt)
        return K


class PlasticStep(MidpointStep):
    """MidpointStep at theta = 1 with the step's rotation read at Newton's start, before its first evaluation"""
    def begin(self):
        super().begin()
        self._fresh = True

    def _eval(self, t):
        if getattr(self, '_fresh', False):
            self._fresh = False
            W = self.W
            W.set_rotation(self.tz)
            if W.stick:                                                             # (after k_mid_ball, which wrote mu F_n there; at the step's start as it does)
                W.tbbound.copy_(torch.where(W.tbR > 0, W.tau_y*W.footprint(self.tx, W._ulr), torch.zeros_like(W.tbR)))
        return super()._eval(t)
