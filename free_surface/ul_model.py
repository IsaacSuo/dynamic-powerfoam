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
  f_mode    'total' (default, UPDATED_LAGRANGE.md section 20): F = kin(x) fitted against material reference vectors;
            fit_w 'rest' (default): on the rest neighbours with the rest weights and rest offsets, the carry with the rest
            cut's centroid offsets: a constant linear map, so F is a function of the positions alone (no drift, any path;
            the fit's condition fixed at its rest value; no guarantee of a correct, positive det F under any large
            deformation). fit_w 'wall' (current cut's walls; references from pair_refs, new_ref 'rest' X_j - X_i or
            'contact' Fb^-1 d at the first meeting) is not a function of the positions alone (the weights depend on the
            radii, which depend on det F) and inverted a cell on the 8k drop when its walls shrank to four; fit_w 'lens'
            (every overlapping pair, lens-volume weights) is robust but soft (part 0.62-0.74 of FEM against 0.82-0.90).
            'increment' (the first form): F = (I + Gc(x - x^n)) F^n, a product of increments each fitted on its own cut,
            which need not invert one another (least squares and inversion do not commute, the carry differs between
            cuts, the radii feed F's error back): F drifts, 10-28
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


def mv(A, b):
    """batched 3 x 3 matrix times vector by broadcasting (10-28: einsum / @ on tiny float64 matrices went through cuBLAS
    GEMM kernels, 75 % of a product's GPU time)"""
    return (A*b[..., None, :]).sum(-1)


def mm(A, B):
    """batched 3 x 3 matrix product by broadcasting"""
    return (A[..., :, :, None]*B[..., None, :, :]).sum(-2)


class GraphedProduct:
    """a product d -> f(aux, d) replayed from a CUDA graph while aux (the Newton iterate) stays the same: captured at the
    third call for one aux (a capture costs ~ 20 ms on the 8k foam, a replay 0.95 ms against 1.76 ms plain, 10-28). Each
    graph has its own memory pool, freed with it (10-28 lion: a pool handle shared by the graphs of successive steps,
    reused after they were all freed, tripped PyTorch 2.9's allocator assert use_count > 0). When to capture: a capture
    pays off after ~25 products (20 ms / 0.8 ms saved per product); the number an iterate will take is predicted by the
    last iterate's (kept across steps): at least 25, capture at the third call, else only after 25 calls (10-28 lion,
    deep compression: ~500 Newton iterates of ~4 products each, a capture per iterate cost more than the products)"""
    breakeven, last, captures = 25, None, 0

    def __init__(self, f, after=2):
        self.f, self.after, self.key, self.calls, self.graph = f, after, None, 0, None

    def __call__(self, aux, d):
        if aux is not self.key:
            if self.key is not None:
                GraphedProduct.last = self.calls
            self.key, self.calls, self.graph = aux, 0, None
        self.calls += 1
        last = GraphedProduct.last
        after = self.after if last is None or last >= self.breakeven else self.breakeven
        if self.graph is None and self.calls > after:
            GraphedProduct.captures += 1
            self.d = d.clone()
            s_ = torch.cuda.Stream(); s_.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s_):
                self.f(aux, self.d)
            torch.cuda.current_stream().wait_stream(s_)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.out = self.f(aux, self.d)
        if self.graph is None:
            return self.f(aux, d)
        self.d.copy_(d); self.graph.replay()
        return self.out.clone()


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
                 centroid='cut', site_w='A/l', f_mode='total', new_ref='rest', fit_w='rest', tet_w=0., tet_qmin=.1, wall_w=0., cell_w=1., cellhg_w=0., cellhg_k=1, mass='diag', grad='fit', fit_cap=None, carry_cap=None, f_substeps=1, agg=0., beta=0., precond='scalar', jump_n='current', self_contact=None, sc_thr=2.5, sc_birth=False, sc_eps=.25, cut_bonds=False, bond_delta=4., bond_eps=.25, bond_floor=0.):
        t = lambda a: torch.as_tensor(np.asarray(a), dtype=f64, device=dev)
        self.tet_w, self.tet_qmin = tet_w, tet_qmin
        self.wall_w, self.cell_w, self.cellhg_w, self.cellhg_k = wall_w, cell_w, cellhg_w, cellhg_k
        self.dev, self.n = dev, len(X0)
        self.X0, self.r0, self.V0, self.dlt0, self.J0 = t(X0), t(r0), t(V0), t(dlt0), t(J0)
        self.mu = E/(2*(1+nu)); self.lam = E*nu/((1+nu)*(1-2*nu))
        self.m = rho*self.V0
        self.jump, self.stab, self.shift, self.smooth, self.centroid, self.site_w = jump, stab, shift, smooth, centroid, site_w
        self.recut = Recut(self.n, dev)
        self.I3 = torch.eye(3, dtype=f64, device=dev)
        self.f_mode, self.new_ref, self.fit_w = f_mode, new_ref, fit_w
        self.fit_cap, self.carry_cap, self.f_substeps = fit_cap, carry_cap, f_substeps
        self.beta = beta                                                               # (10-29: Kelvin-Voigt, as the DEM's: eta = beta mu, zeta =
                                                                                       # beta (lam + 2 mu / 3), see visc_terms)
        self.precond = precond                                                         # (10-29: the CG's preconditioner, see step / block_precond)
        self.jump_n = jump_n                                                           # (10-29: the jump term's normal, see _jn)
        self.sc, self.sc_prev, self.sc_stats = self_contact, None, {}                   # (10-29: self-contact, None / 'air' / 'compat',
        self.sc_thr, self.h_med = sc_thr, float(self.V0.median())**(1/3)               # see classify_walls / _sc)
        self.sc_birth = sc_birth                                                       # (the overlap from the wall's birth, see _sc)
        self.sc_eps = sc_eps
        self.bond_floor = bond_floor                                                   # (10-29: the rest pairs' jump stiffness floor)
        if bond_floor > 0 or self_contact == 'rest':
            assert f_mode != 'total' and shift == 'none', 'bond_floor / self_contact rest: f_mode increment, shift none'
            G0, _ = self.recut.geometry(self.X0, self.r0); kp = G0.wall_A > 0
            a0, b0 = torch.minimum(G0.i[kp], G0.j[kp]), torch.maximum(G0.i[kp], G0.j[kp])
            K0 = G0.wall_A.detach()[kp]/(3*(self.X0[b0]-self.X0[a0]).norm(dim=1))
            o = torch.argsort(a0*self.n+b0)
            self.rb_i, self.rb_j, self.rb_K = a0[o], b0[o], K0[o]; self.rb_keys = (a0*self.n+b0)[o]
            self.rb_A = G0.wall_A.detach()[kp][o]
        self.cut_bonds = cut_bonds                                                     # (10-29: the recut over the bond family, see
        if cut_bonds:                                                                  # _bond_family / _contact_pairs)
            t0_ = time.time()
            self.bonds, self.sc_rest_keys, self.sc_rest_ov = self._bond_family(bond_delta, bond_eps)
            self.recut.A.allow = self.bonds
            self.recut_full = Recut(self.n, dev)                                       # (the full recut: where parts meet)
            print('cut_bonds: %d bonds (%.1f per cell; delta %g h, eps %g h), %d pairs overlapping at rest outside them; %.0f s' % (
                len(self.bonds), 2*len(self.bonds)/self.n, bond_delta, bond_eps, len(self.sc_rest_keys), time.time()-t0_), flush=True)
        if self_contact == 'bond':                                                     # (the rest walls: always material)
            G0, _ = self.recut.geometry(self.X0, self.r0); kp = G0.wall_A > 0
            self.sc_rest = torch.sort(torch.minimum(G0.i[kp], G0.j[kp])*self.n+torch.maximum(G0.i[kp], G0.j[kp])).values
        if self_contact == 'skin':                                                     # (the rest skin: the cells with an exposed cap
            import scipy.sparse as sp_                                                 # at rest; the pairs within 2 steps on it, i.e.
            G0, _ = self.recut.geometry(self.X0, self.r0)                              # skin neighbours or sharing a skin neighbour)
            kp = G0.wall_A > 0; a0, b0 = G0.i[kp].cpu().numpy(), G0.j[kp].cpu().numpy()
            skin = (G0.patch_A.detach() > 0).cpu().numpy(); ks = skin[a0] & skin[b0]
            As = sp_.coo_matrix((np.ones(int(ks.sum())), (a0[ks], b0[ks])), shape=(self.n, self.n)).tocsr(); As = ((As+As.T) > 0).astype(np.float64)
            near = sp_.triu((As+As@As) > 0, k=1).tocoo()
            self.sc_skin = torch.as_tensor(skin, device=dev)
            self.sc_near = torch.sort(torch.as_tensor(near.row.astype(np.int64)*self.n+near.col, device=dev)).values
        self.E_sc = E/(2*(1-nu**2))                                                    # (two balls of the same material)
        self.agg = agg                                                                 # (10-29: cells below agg x the median volume take their
                                                                                       # neighbours' fit, see operators)
        self.deg_cells = torch.zeros(0, dtype=torch.long, device=dev); self.deg_q = torch.zeros(0, 3, dtype=f64, device=dev)
        if f_mode == 'total':                                                          # UPDATED_LAGRANGE.md section 20
            geo0 = self.geometry(self.X0, self.r0)
            self.carry0 = self.cut_centroids(self.X0, geo0)                            # the material centroid offsets
            i0, j0 = geo0['i'], geo0['j']
            self.mem_a, self.mem_b = torch.minimum(i0, j0), torch.maximum(i0, j0)      # the pair memory, D oriented a -> b
            self.mem_D = self.X0[self.mem_b]-self.X0[self.mem_a]
            self.rest_walls, self.rest_fit = (i0, j0, geo0['A']), None
            self.null_site, self.null_cen = torch.zeros(self.n, 3, 3, dtype=f64, device=dev), torch.zeros(self.n, 3, 3, dtype=f64, device=dev)
            if fit_w == 'rest':
                self._degenerate(i0, j0)
        if tet_w > 0:
            self._material_tets(geo0['i'], geo0['j'])
        if wall_w > 0:
            self._material_walls(geo0)
        if cellhg_w > 0:
            self.hg_i, self.hg_j, self.hg_A = geo0['i'], geo0['j'], geo0['A']
        self.grad = 'fit'
        if grad == 'corrected':
            if mass == 'consistent':
                raise NotImplementedError("grad 'corrected' with mass 'consistent'")
            self._corrected_operator()
        elif grad in ('gg_a', 'gg_b', 'gg_c'):
            if mass == 'consistent':
                raise NotImplementedError("grad 'gg_*' with mass 'consistent'")
            self._gg_operator(grad[-1])
        elif grad != 'fit':
            raise ValueError('grad: fit, corrected, gg_a, gg_b or gg_c')
        self.grad = grad
        self.mass = mass
        if mass == 'consistent':
            self._consistent_mass(rho)
        elif mass != 'diag':
            raise ValueError('mass: diag or consistent')
            self.hg_den = 2*torch.zeros(self.n, dtype=f64, device=self.dev).index_add_(0, self.hg_i, self.hg_A).index_add_(0, self.hg_j, self.hg_A)

    def _material_tets(self, i0, j0):
        """the tetrahedra between the sites (10-29, the UL lion's sites crushed beyond F: the site tetrahedra below 5 cell
        sizes held 0.53 of their volume where V0 det F said 0.83, 18 % inverted; the jump term is linear and per pair):
        the rest regular triangulation (the rest cut's dual; lower hull of (X, |X|^2 - r^2)), the tetrahedra whose six
        edges are rest walls. Their energy (tet_energy) is the Bregman remainder of psi's volumetric part U(J) = -mu ln J
        + lam / 2 ln^2 J at the tetrahedron's own J_t (exact from its four sites) about the mean det F of its four cells:
        zero with its force for affine motion (nothing re-counted), >= 0 (U convex for J < e^(1 + mu / lam)), rotation
        free (the full-F remainder is negative for a tetrahedron rotated against its cells under compression: initial
        stress), infinite when the tetrahedron inverts"""
        from scipy.spatial import ConvexHull
        X = self.X0.cpu().numpy(); r = self.r0.cpu().numpy(); n = self.n
        gg = X-X.mean(0)
        hull = ConvexHull(np.c_[gg, (gg**2).sum(1)-r**2], qhull_options='Qt Qbb Qc Q12')
        T = hull.simplices[hull.equations[:, 3] < 0]
        rest = set((torch.minimum(i0, j0)*n+torch.maximum(i0, j0)).tolist())
        E6 = np.stack([T[:, [0, 0, 0, 1, 1, 2]], T[:, [1, 2, 3, 2, 3, 3]]], 2)
        k6 = np.minimum(E6[:, :, 0], E6[:, :, 1]).astype(np.int64)*n+np.maximum(E6[:, :, 0], E6[:, :, 1])
        T = T[np.array([all(int(q) in rest for q in row) for row in k6])]
        Dm = np.stack([X[T[:, 1]]-X[T[:, 0]], X[T[:, 2]]-X[T[:, 0]], X[T[:, 3]]-X[T[:, 0]]], 2)
        neg = np.linalg.det(Dm) < 0
        T[neg] = T[neg][:, [0, 2, 1, 3]]                                                   # positive orientation
        Dm = np.stack([X[T[:, 1]]-X[T[:, 0]], X[T[:, 2]]-X[T[:, 0]], X[T[:, 3]]-X[T[:, 0]]], 2)
        # slivers left out (10-29 lion: 15 % of these tetrahedra have under 0.05 of a regular tetrahedron's volume on
        # their longest edge; 1 % of a cell size of random site noise inverted 51 of all, 0 of those with quality >= 0.1)
        Lmax = np.max([np.linalg.norm(X[T[:, a]]-X[T[:, b]], axis=1) for a, b in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))], 0)
        good = np.linalg.det(Dm)/6/(Lmax**3/(6*2**.5)) >= self.tet_qmin
        T, Dm = T[good], Dm[good]
        self.tets = torch.as_tensor(T, dtype=torch.long, device=self.dev)
        self.tet_Dmi = torch.as_tensor(np.linalg.inv(Dm), dtype=f64, device=self.dev)
        self.tet_V = torch.as_tensor(np.linalg.det(Dm)/6, dtype=f64, device=self.dev)

    def _material_walls(self, geo0):
        """the rest walls as the energy's integration domains (10-29, edge-based smoothing ES-FEM / Breitzman & Dayal 2018
        bond-level F): each cell's rest volume shared among its walls by A0 l0 (sum of the walls' volumes = the body's),
        wall_energy integrates psi there"""
        i0, j0, A0 = geo0['i'], geo0['j'], geo0['A']
        D = self.X0[j0]-self.X0[i0]; l0 = D.norm(dim=1)
        q = A0*l0
        tot = torch.zeros(self.n, dtype=f64, device=self.dev).index_add_(0, i0, q).index_add_(0, j0, q)
        self.wall_i, self.wall_j, self.wall_D = i0, j0, D
        self.wall_V = self.V0[i0]*q/tot[i0]+self.V0[j0]*q/tot[j0]

    def wall_energy(self, y, ops):
        """wall_w sum_walls V_w psi(F_w), F_w = Fb + (d - Fb D) (x) D / |D|^2 (exact along the pair: F_w D = d; across it the
        mean of the two cells' F), neo-Hookean; inf where some det F_w <= 0. Returns the energy and min det F_w"""
        if not ops['total']:
            raise RuntimeError('wall_energy: f_mode total only')
        F = self.def_grad(y, ops)
        i, j, D = self.wall_i, self.wall_j, self.wall_D
        Fb = .5*(F[i]+F[j])
        dl = (y[j]-y[i])-(Fb*D[:, None, :]).sum(-1)
        Fw = Fb+dl[:, :, None]*(D/(D*D).sum(1, keepdim=True))[:, None, :]
        J = torch.linalg.det(Fw)
        if bool((J <= 0).any()):
            return y.sum()*0+math.inf, float(J.min())
        lnJ = torch.log(J)
        psi = .5*self.mu*((Fw*Fw).sum((1, 2))-3)-self.mu*lnJ+.5*self.lam*lnJ**2
        return self.wall_w*(self.wall_V*psi).sum(), float(J.min())

    def _consistent_mass(self, rho):
        """the kinetic energy of the model's own displacement field (10-29): in cell s the field is u_c,s + Gc_s (X - c_s),
        u_c = v + G0 carry (the site fit carried to the centroid), Gc the centroid fit, both linear in the site increments v
        with scalar weights and constant in f_mode 'total' (rest operators); its exact kinetic energy per cell is
        1/2 rho [V_s |u_c,s'|^2 + Gc_s' : (Gc_s' J0_s)], J0_s = int (X - c)(X - c)^T dV, so KE = 1/2 sum_k v_k'^T Mc v_k'
        (k the component) with the n x n matrix Mc = rho (P^T diag(V) P + sum_kl Q_k^T diag(J0_kl) Q_l), u_c = P v,
        (Gc)_ik = (Q_k v_i). Exact total mass and rigid motions; the lumped masses Mc 1 = rho P^T V carry gravity and the
        centre of mass. 10-29 part: the compression test's deepest modes 3-5x the continuum's under the site masses, 0.35-0.75x
        under Mc (a light site turning a heavy neighbour cell's F carried no kinetic energy). The degenerate cells' free
        columns stay massless"""
        if self.f_mode != 'total' or self.shift != 'displacement':
            raise RuntimeError("mass 'consistent': f_mode 'total' and shift 'displacement' only")
        n = self.n
        I = self.I3.expand(n, 3, 3).contiguous()
        geo = self.geometry(self.X0, self.r0)
        ops = self.operators(self.X0, I, geo, self.pair_refs(self.X0, I, self.r0, geo))
        S, O = ops['KS'], ops['KO']
        w = (ops['R']*ops['carry'][S]).sum(1)
        P = torch.zeros(n, n, dtype=f64, device=self.dev)
        P.index_put_((S, O), w, accumulate=True); P.index_put_((S, S), -w, accumulate=True)
        P = P+torch.eye(n, dtype=f64, device=self.dev)
        Q = []
        for k in range(3):
            B = torch.zeros(n, n, dtype=f64, device=self.dev)
            B.index_put_((S, O), ops['RC'][:, k], accumulate=True); B.index_put_((S, S), -ops['RC'][:, k], accumulate=True)
            Q.append(B@P)
        M = P.T@(self.V0[:, None]*P)
        for k in range(3):
            for l in range(3):
                M = M+Q[k].T@(self.J0[:, k, l][:, None]*Q[l])
        self.Mc = rho*.5*(M+M.T)
        self.m = self.Mc.sum(1); self.mdiag = torch.diagonal(self.Mc).clone()

    def _corrected_operator(self):
        """the fit's F with the integration constraint (10-29): the two-step fit (rest neighbours, carried to the centroid)
        is a linear map G_s = sum_a u_a (x) W_sa; its discrete divergence of a constant b_a = sum_s V_s W_sa is O(1) at the
        interior sites (one step: sum_s V_s M_s^-1 A_as n_as, zero only for equal isotropic V_s M_s^-1), so a uniform stress
        pushes them (part patch test: deviation 0.24-0.93 |H| h, energy 8-24 % below affine). The smallest change delta
        (sum_s V_s |delta_s|^2, on the same stencils) keeping every cell exact for affine fields (sum_a delta_sa = 0,
        sum_a delta_sa X_a = 0) and making b_a = 0 at the sites without an exposed cap: per component, delta_s = -Pi_s lambda
        on the cell's stencil, Pi_s the projection off the affine patterns [1, X] there, N lambda = b on those sites,
        N = sum_s V_s E_s^T Pi_s E_s. Surface sites unconstrained (they carry the boundary terms). The result is used as a
        one-step operator (ops 'onestep'), for the energy and the forces alike (variational; cf. VCI, Chen, Hillman, Rueter
        2013, which corrects the test functions only)"""
        if self.f_mode != 'total' or self.shift != 'displacement':
            raise RuntimeError("grad 'corrected': f_mode 'total' and shift 'displacement' only")
        n = self.n
        I = self.I3.expand(n, 3, 3).contiguous()
        geo = self.geometry(self.X0, self.r0)
        ops = self.operators(self.X0, I, geo, self.pair_refs(self.X0, I, self.r0, geo))
        S, O = ops['KS'], ops['KO']
        w = (ops['R']*ops['carry'][S]).sum(1)
        P = torch.zeros(n, n, dtype=f64, device=self.dev)
        P.index_put_((S, O), w, accumulate=True); P.index_put_((S, S), -w, accumulate=True)
        P = P+torch.eye(n, dtype=f64, device=self.dev)
        W = torch.zeros(3, n, n, dtype=f64, device=self.dev)
        for k in range(3):
            B = torch.zeros(n, n, dtype=f64, device=self.dev)
            B.index_put_((S, O), ops['RC'][:, k], accumulate=True); B.index_put_((S, S), -ops['RC'][:, k], accumulate=True)
            W[k] = B@P
        del P, B
        mask = W.abs().sum(0) > 1e-12*float(W.abs().max())
        cap = geo['capA'] > 0; inner = torch.nonzero(~cap).flatten()
        b = torch.einsum('s,ksa->ka', self.V0, W)                                          # (3, n) the divergence of a constant
        N = torch.zeros(n, n, dtype=f64, device=self.dev); Pis, idxs = [], []
        for c in range(n):
            idx = torch.nonzero(mask[c]).flatten()
            Bm = torch.cat([torch.ones(len(idx), 1, dtype=f64, device=self.dev), self.X0[idx]-self.X0[c]], 1)
            Pi = torch.eye(len(idx), dtype=f64, device=self.dev)-Bm@torch.linalg.solve(Bm.T@Bm, Bm.T)
            N[idx[:, None], idx[None, :]] += self.V0[c]*Pi
            Pis.append(Pi); idxs.append(idx)
        lam = torch.zeros(3, n, dtype=f64, device=self.dev)
        lam[:, inner] = torch.linalg.solve(N[inner][:, inner], b[:, inner].T).T
        W2 = W.clone(); dn = torch.zeros(n, dtype=f64, device=self.dev); wn = torch.zeros(n, dtype=f64, device=self.dev)
        for c in range(n):
            idx = idxs[c]; d = -(Pis[c]@lam[:, idx].T).T                                  # (3, k)
            W2[:, c, idx] += d
            dn[c] = d.norm(); wn[c] = W[:, c, idx].norm()
        b2 = torch.einsum('s,ksa->ka', self.V0, W2)
        ex0 = W2.sum(2).abs().max(); ex1 = (torch.einsum('ksa,al->ksl', W2, self.X0)-torch.eye(3, dtype=f64, device=self.dev)[:, None, :]).abs().max()
        rel = (dn/wn).cpu().numpy()
        self.cor_stats = dict(b_before=float((b[:, inner].norm(dim=0)/self.V0[inner]**(2/3)).median()),
                              b_after=float((b2[:, inner].norm(dim=0)/self.V0[inner]**(2/3)).max()), exact_const=float(ex0), exact_lin=float(ex1),
                              rel_median=float(np.median(rel)), rel_p90=float(np.percentile(rel, 90)), rel_max=float(rel.max()),
                              rel_total=float(dn.norm()/wn.norm()))
        cs, ca = torch.nonzero(mask & ~torch.eye(n, dtype=torch.bool, device=self.dev), as_tuple=True)
        self.cor_ops = (cs, ca, W2[:, cs, ca].T.contiguous())

    def _capped_fits(self, i, j, w, lev, x, cc, M, Mc, wc, e, delta):
        """the current-cut fits with a cap on their conditioning (10-29): a cell whose walls leave a direction nearly
        unseen (a tiny cell down to 4 walls: cond(M) 81, its increment's fit det(I + G) -0.11 where the forward step's
        exact inverse is 1.01) takes that direction from the fit over its 1- and 2-ring (uniform weights): G = S M (M^2 +
        eps^2)^-1 + G2 eps^2 (M^2 + eps^2)^-1. Exact for affine fields (S = H M, G2 = H); per eigen-direction the 2-ring
        share eps^2 / (lambda^2 + eps^2). eps^2 = (lambda_max / fit_cap)^2 s, s = clamp((cond - c0) / (fit_cap - c0), 0, 1)^2,
        c0 = fit_cap / 2: cells with cond(M) <= c0 keep the plain fit exactly and need no 2-ring rows (rest part: cond
        median 2.6, 99 % 14.3), eps continuous in the geometry. The same for the centroid fit (levers between the current
        cut's centroids). With the centroid carry (shift 'displacement') the site fit enters F only through the carried
        displacement u_c = v + G0 delta, so its eps is set by what matters there instead (10-29, cell 1674 at the loop's
        step 12: site fit cond 8.7, centroid fit 2.4, but |delta| 0.69 h, det(I + Gc) -0.05): the sensitivity kappa =
        |c| of the carried displacement's weights c_o = w_o lev_o^T A delta, A = M (M^2 + eps^2)^-1, kappa^2 = delta^T A W A
        delta, W = sum w^2 lev lev^T; eps = 0 where kappa(0) <= carry_cap, else the smallest eps with kappa = carry_cap
        (bisection; rest part: kappa median 0.135, 99 % 0.81). Returns the pair lists and rows: the walls' (both ends),
        then the 2-ring pairs of the cells under either cap"""
        import scipy.sparse as sp_
        n = self.n; c0 = self.fit_cap/2
        def eps2_of(Mw):
            ev = torch.linalg.eigvalsh(Mw)
            lmax = torch.maximum(ev[:, -1], 1e-3*ev[:, -1].median())                 # (a cell without walls: the 2-ring fit)
            cond = lmax/ev[:, 0].clamp(min=1e-300)
            ramp = ((cond-c0)/(self.fit_cap-c0)).clamp(0., 1.)**2
            return ((lmax/self.fit_cap)**2*ramp)[:, None, None]
        if Mc is not None and self.carry_cap is not None:                             # the site fit capped by the carry's sensitivity
            W = torch.zeros(n, 3, 3, dtype=f64, device=self.dev); q = (w**2)[:, None, None]*lev[:, :, None]*lev[:, None, :]
            W.index_add_(0, i, q); W.index_add_(0, j, q)
            def kap(e2):
                A_ = M@torch.linalg.inv(M@M+e2[:, None, None]*self.I3)
                Ad = torch.einsum('nab,nb->na', A_, delta)
                return torch.sqrt(torch.einsum('na,nab,nb->n', Ad, W, Ad))
            evm = torch.linalg.eigvalsh(M); lmax = torch.maximum(evm[:, -1], 1e-3*evm[:, -1].median())
            degen = evm[:, 0] < 1e-10*lmax                                               # rank-deficient: always capped
            need = degen | (kap(torch.where(degen, lmax**2, torch.zeros_like(lmax))) > self.carry_cap)
            lo = torch.full((n,), -12., dtype=f64, device=self.dev); hi = torch.zeros(n, dtype=f64, device=self.dev)   # log10(eps / lambda_max)
            for _ in range(40):
                mid = .5*(lo+hi); big = kap((lmax*10**mid)**2) > self.carry_cap
                lo = torch.where(big, mid, lo); hi = torch.where(big, hi, mid)
            e2s = torch.where(need, (lmax*10**hi)**2, torch.zeros_like(lmax))[:, None, None]
        else:
            e2s = eps2_of(M)
        e2c = eps2_of(Mc) if Mc is not None else torch.zeros_like(e2s)
        act = ((e2s[:, 0, 0] > 0) | (e2c[:, 0, 0] > 0)).cpu().numpy()
        ii, jj = i.cpu().numpy(), j.cpu().numpy()
        Ad = sp_.coo_matrix((np.ones(2*len(ii)), (np.r_[ii, jj], np.r_[jj, ii])), shape=(n, n)).tocsr()
        A2 = ((Ad+Ad@Ad) > 0).tocoo(); m_ = (A2.row != A2.col) & act[A2.row]
        a2 = torch.as_tensor(A2.row[m_], device=self.dev); b2 = torch.as_tensor(A2.col[m_], device=self.dev)
        def cap(Mw, eps2, levw, ww, pts):
            N = torch.linalg.inv(Mw@Mw+eps2*self.I3)                                    # (M^2 + eps^2)^-1
            A_ = Mw@N; B_ = eps2*N                                                     # data and 2-ring shares
            rw = torch.cat([torch.einsum('fab,fb->fa', A_[i], ww[:, None]*levw), torch.einsum('fab,fb->fa', A_[j], -ww[:, None]*levw)])
            if len(a2) == 0:
                return rw
            l2 = pts[b2]-pts[a2]
            M2 = torch.zeros(n, 3, 3, dtype=f64, device=self.dev).index_add_(0, a2, l2[:, :, None]*l2[:, None, :])
            M2i = torch.linalg.pinv(M2)                                                # (cells without 2-ring rows: unused)
            return torch.cat([rw, torch.einsum('fab,fbc,fc->fa', B_[a2], M2i[a2], l2)])
        KS, KO = torch.cat([i, j, a2]), torch.cat([j, i, b2])
        R_ = cap(M, e2s, lev, w, x)
        RC_ = cap(Mc, e2c, e, wc, cc) if Mc is not None else None
        self.cap_stats = dict(cells=int(act.sum()), site=int((e2s[:, 0, 0] > 0).sum()), centroid=int((e2c[:, 0, 0] > 0).sum()), pairs2=len(a2))
        return KS, KO, R_, RC_

    def update_F(self, x, y, F, r, GS=None):
        """f_mode 'increment', F at the step end in f_substeps increments along the straight path x -> y, each fitted on a
        cut at its own start (the step start's radii r; the first on GS): for affine motion the increments telescope to the
        exact one; a fit's error enters per substep, so a step's large non-affine increment is not multiplied in at once
        (10-29: the loop along the part impact's path inverted cells at the run's dt, not at dt / 4). The step's solve keeps
        its single-increment F (the difference is second order in the step)"""
        if len(self.deg_cells):
            raise NotImplementedError('update_F: degenerate cells')
        K = self.f_substeps; Fk = F; geo_p = None
        for k in range(K):
            xa = x+(k/K)*(y-x); xb = x+((k+1)/K)*(y-x)
            try:                                                                       # a substep's cut with the step start's radii
                geo_k = self.geometry(xa, r, GS if k == 0 else None)                    # can leave a cell without walls: then the
                ops_k = self.operators(xa, Fk, geo_k, None)                            # last substep's walls, at this substep's
            except RuntimeError:                                                       # positions
                if geo_p is None:
                    raise
                self.recut.stats['substep_fallback'] = self.recut.stats.get('substep_fallback', 0)+1
                geo_k = geo_p; ops_k = self.operators(xa, Fk, geo_k, None)
            Fk = (self.I3+self.kin(xb-xa, ops_k))@Fk; geo_p = geo_k
        return Fk

    def _gg_operator(self, kind):
        """the cells' gradient by the divergence theorem on the rest cut (10-29): V_s G_s = sum_walls A u_w (x) n + cap,
        u_w the displacement at the wall's centroid, shared by the two cells: linear along the sites' line to the wall's
        plane, the centroid's offset from that line carried by the mean of the two cells' site-level fits G0 (rest
        neighbours, A / l weights); the cap (exposed sphere) integral u_s (x) a_s + Gcap M_s, a_s = -sum_walls A n,
        M_s = V_s I - sum_walls A (c_w - X_s) (x) n (zero for interior cells: no fit and no inverse there). Gcap:
        'a' the cell's own G0; 'b' the area-weighted mean of the G0 of the cell and its wall neighbours; 'c' G0's tangential
        columns (mean cap normal m = a_s / |a_s|) with the normal column from the linear free-surface closure
        (eps_nt = 0, eps_nn = -lam / (lam + 2 mu) tr_t eps). Exact for affine fields with 'a' and 'b'; 'c' only for those
        with a traction-free cap. Total Lagrangian (F = I + G(x - X0)); stored as a one-step tensor operator
        dG_s = sum_a T_sa (u_a - u_s)"""
        if self.f_mode != 'total':
            raise RuntimeError("grad 'gg_*': f_mode 'total' only")
        n, dv_ = self.n, self.dev
        geo = self.geometry(self.X0, self.r0)
        i, j, A, c = geo['i'], geo['j'], geo['A'], geo['c']
        D = self.X0[j]-self.X0[i]; L = D.norm(dim=1); nn = D/L[:, None]
        Vg = geo['V']
        t = ((c-self.X0[i])*nn).sum(1)/L
        tau = c-(self.X0[i]+t[:, None]*D)
        w = A/L
        M0 = torch.zeros(n, 3, 3, dtype=f64, device=dv_); dd = w[:, None, None]*D[:, :, None]*D[:, None, :]
        M0.index_add_(0, i, dd); M0.index_add_(0, j, dd)
        M0i = torch.linalg.pinv(M0)
        a = torch.zeros(n, 3, dtype=f64, device=dv_).index_add_(0, i, -A[:, None]*nn).index_add_(0, j, A[:, None]*nn)
        Mc = Vg[:, None, None]*self.I3-(torch.zeros(n, 3, 3, dtype=f64, device=dv_)
                                          .index_add_(0, i, A[:, None, None]*(c-self.X0[i])[:, :, None]*nn[:, None, :])
                                          .index_add_(0, j, -A[:, None, None]*(c-self.X0[j])[:, :, None]*nn[:, None, :]))
        cap = geo['capA'] > 0
        Mc[~cap] = 0.; a[~cap] = 0.
        m_ = a/a.norm(dim=1, keepdim=True).clamp(min=1e-300); Pt = self.I3-m_[:, :, None]*m_[:, None, :]
        kap = self.lam/(self.lam+2*self.mu)
        wA = torch.zeros(n, dtype=f64, device=dv_).index_add_(0, i, A).index_add_(0, j, A)
        def G0f(u):
            du = u[j]-u[i]
            return (torch.zeros(n, 3, 3, dtype=f64, device=dv_)
                    .index_add_(0, i, w[:, None, None]*du[:, :, None]*torch.einsum('fab,fb->fa', M0i[i], D)[:, None, :])
                    .index_add_(0, j, w[:, None, None]*du[:, :, None]*torch.einsum('fab,fb->fa', M0i[j], D)[:, None, :]))
        def apply(u):
            G0 = G0f(u)
            uw = (1-t)[:, None]*u[i]+t[:, None]*u[j]+.5*torch.einsum('fab,fb->fa', G0[i]+G0[j], tau)
            S = (torch.zeros(n, 3, 3, dtype=f64, device=dv_).index_add_(0, i, A[:, None, None]*uw[:, :, None]*nn[:, None, :])
                 .index_add_(0, j, -A[:, None, None]*uw[:, :, None]*nn[:, None, :]))
            if kind == 'a':
                Gc_ = G0
            elif kind == 'b':
                Gc_ = (wA[:, None, None]*G0).index_add(0, i, A[:, None, None]*G0[j]).index_add_(0, j, A[:, None, None]*G0[i])/(2*wA)[:, None, None]
            else:
                Ht = G0@Pt
                cc = -torch.einsum('nab,nb->na', Pt@G0.transpose(1, 2), m_)-kap*torch.einsum('naa->n', Ht)[:, None]*m_
                Gc_ = Ht+cc[:, :, None]*m_[:, None, :]
            return (S+u[:, :, None]*a[:, None, :]+Gc_@Mc)/Vg[:, None, None]
        Lm = torch.zeros(n, 3, 3, n, 3, dtype=f64, device=dv_); e = torch.zeros(n, 3, dtype=f64, device=dv_)
        for s_ in range(n):
            for k in range(3):
                e[s_, k] = 1.; Lm[:, :, :, s_, k] = apply(e); e[s_, k] = 0.
        blk = Lm.abs().sum((1, 2, 4)) > 1e-12*float(Lm.abs().max())
        blk.fill_diagonal_(False)
        cs, ca = torch.nonzero(blk, as_tuple=True)
        T = Lm[cs, :, :, ca, :]                                                            # (P, 3, 3, 3): dG_s[i, j] / du_a[c]
        Xa = self.X0+torch.randn(3, dtype=f64, device=dv_)
        Ht_ = torch.randn(3, 3, dtype=f64, device=dv_)
        self.gg_stats = dict(kind=kind, pairs=len(cs), V_geo_vs_V0=float(((Vg-self.V0)/self.V0).abs().max()),
                             interior_M_max=float((Vg[:, None, None]*self.I3-(torch.zeros(n, 3, 3, dtype=f64, device=dv_)
                                          .index_add_(0, i, A[:, None, None]*(c-self.X0[i])[:, :, None]*nn[:, None, :])
                                          .index_add_(0, j, -A[:, None, None]*(c-self.X0[j])[:, :, None]*nn[:, None, :])))[~cap].abs().max()/Vg[~cap].min()),
                             translation=float(Lm.sum(3).abs().max()), affine=float((apply(Xa@Ht_.T)-Ht_).abs().max()))
        self.gg_ops = (cs, ca, T)
        self.gg_off = self.I3-apply(self.X0) if kind == 'c' else None                # the closure is affine in the positions: F = G(x) + (I - G(X0))
        del Lm

    def kinetic(self, v):
        """the kinetic energy of the site velocities v (n, 3)"""
        if self.mass == 'consistent':
            return float(.5*(v*(self.Mc@v)).sum())
        return float(.5*(self.m[:, None]*v**2).sum())

    def cellhg_energy(self, y, ops):
        """the cells' psi with the cell-scale fluctuation's own quadratic (geometric) term dropped (10-29; geometric
        hourglassing: Bieber, Auricchio, Reali, Bischoff, IJNME 2023, modified Green-Lagrange strain; Reese et al.: no
        geometric stiffness for the hourglass part): Fb_s = (sum_o A_so F_o + (sum_o A_so) F_s) / (2 sum_o A_so) over the
        rest neighbours (exact for affine fields; cellhg_k passes, eigenvalues of the pass in [0, 1], so the kept geometric
        part m (2 - m) is in [0, 1]), Hh = F - Fb, C' = F^T F - Hh^T Hh, neo-Hookean in C' (J' = det C'^1/2).
        Objective (Hh turns with a rigid rotation); at rest and for affine fields identical to the cells' psi"""
        if not ops['total']:
            raise RuntimeError('cellhg_energy: f_mode total only')
        F = self.def_grad(y, ops)
        i, j, A = self.hg_i, self.hg_j, self.hg_A
        w = (self.hg_den/2)[:, None, None]
        Fb = F
        for _ in range(self.cellhg_k):
            S = torch.zeros_like(F).index_add(0, i, A[:, None, None]*Fb[j]).index_add(0, j, A[:, None, None]*Fb[i])
            Fb = (S+w*Fb)/(2*w)
        Hh = F-Fb
        C = F.transpose(1, 2)@F-Hh.transpose(1, 2)@Hh
        dC = torch.linalg.det(C)
        if bool((dC <= 0).any()):
            return y.sum()*0+math.inf, float(dC.min())
        lnJ = .5*torch.log(dC)
        psi = .5*self.mu*(C.diagonal(dim1=1, dim2=2).sum(1)-3)-self.mu*lnJ+.5*self.lam*lnJ**2
        return self.cellhg_w*(self.V0*psi).sum(), float(dC.min())

    def tet_energy(self, y, ops):
        """tet_w sum V_t [U(J_t) - U(Jb) - U'(Jb)(J_t - Jb)], U(J) = -mu ln J + lam / 2 ln^2 J, J_t = det(Ds Dm^-1) from the
        tetrahedron's sites, Jb the mean det F of its four cells; inf where a tetrahedron has J_t <= 0. Returns the energy
        and min J_t"""
        if not ops['total']:
            raise RuntimeError('tet_energy: f_mode total only')
        J = torch.linalg.det(self.def_grad(y, ops))
        xs, T = y[:self.n], self.tets
        Ds = torch.stack([xs[T[:, 1]]-xs[T[:, 0]], xs[T[:, 2]]-xs[T[:, 0]], xs[T[:, 3]]-xs[T[:, 0]]], 2)
        Jt = torch.linalg.det(Ds@self.tet_Dmi)
        Jb = J[T].mean(1)
        if bool((Jt <= 0).any()) or bool((Jb <= 0).any()):
            return y.sum()*0+math.inf, float(Jt.min())
        U = lambda J_: -self.mu*torch.log(J_)+.5*self.lam*torch.log(J_)**2
        dU = (-self.mu+self.lam*torch.log(Jb))/Jb
        return self.tet_w*(self.tet_V*(U(Jt)-U(Jb)-dU*(Jt-Jb))).sum(), float(Jt.min())

    def _degenerate(self, i0, j0):
        """cells whose rest walls span fewer than three directions (fewer than 3 walls; 10-28 lion: one cell with 2, its F
        off I by 14 at rest): the fits fix F only on the levers' span (pseudo-inverses, null projectors null_site /
        null_cen); the carried centroid offset is projected onto the site levers' span (u = x + G delta then moves with
        the material point X + P delta, exact under affine motion, for the cell and its neighbours' centroid levers); F's
        columns along the centroid levers' null directions q become unknowns of the step, F = fit + sum b q^T (rows after
        the sites in the solve's vector, no mass), set by the energy's minimum: dPi/db = P q = 0, traction free along q
        (UPDATED_LAGRANGE.md section 5, static condensation of the unseen direction only; the cell's energy stays)"""
        n = self.n
        nw = torch.bincount(torch.cat([i0, j0]), minlength=n)
        cells = torch.nonzero(nw < 3).flatten().tolist()
        nbrs = {k: torch.cat([j0[i0 == k], i0[j0 == k]]) for k in cells}
        def null(L):
            if len(L) == 0:
                return torch.eye(3, dtype=f64, device=self.dev)
            return torch.linalg.svd(L, full_matrices=True)[2][len(L):].T          # (3, 3 - rank), the walls' directions independent
        for k in cells:
            Q = null(self.X0[nbrs[k]]-self.X0[k]); P = Q@Q.T
            self.null_site[k] = P; self.carry0[k] = self.carry0[k]-P@self.carry0[k]
        rc, rq = [], []
        for k in cells:
            Q = null(self.X0[nbrs[k]]-self.X0[k]+self.carry0[nbrs[k]]-self.carry0[k])
            self.null_cen[k] = Q@Q.T
            rc += [k]*Q.shape[1]; rq.append(Q.T)
        if cells:
            self.deg_cells = torch.tensor(rc, dtype=torch.long, device=self.dev); self.deg_q = torch.cat(rq)

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

    def lens_pairs(self, x, r, chunk=1024):
        """every pair of overlapping balls (a < b) and the volume of their lens: pi (R + r - d)^2 (d^2 + 2 d (R + r) -
        3 (R - r)^2) / (12 d), the smaller ball's volume when one holds the other; 0 when the balls just touch, so a pair
        enters and leaves with weight 0"""
        A_, B_ = [], []
        for c0 in range(0, self.n, chunk):
            d = torch.cdist(x[c0:c0+chunk], x)
            ov = d < r[c0:c0+chunk, None]+r[None]
            a, b = torch.nonzero(ov, as_tuple=True); a = a+c0
            k = a < b; A_.append(a[k]); B_.append(b[k])
        a, b = torch.cat(A_), torch.cat(B_)
        d = (x[b]-x[a]).norm(dim=1); R1, R2 = r[a], r[b]
        V = math.pi*(R1+R2-d)**2*(d**2+2*d*(R1+R2)-3*(R1-R2)**2)/(12*d)
        V = torch.where(d <= (R1-R2).abs(), 4/3*math.pi*torch.minimum(R1, R2)**3, V)
        return a, b, V

    def pair_refs(self, x, F, r, geo):
        """f_mode 'total': every wall's material reference vector (oriented i -> j) from the pair memory; a pair met for
        the first time takes D = Fb^-1 d (new_ref 'contact': Fb = (F_i + F_j) / 2, d = x_j - x_i; its jump starts at 0;
        carries the fit's error at that moment) or the rest offset X_j - X_i (new_ref 'rest': F a function of the
        positions alone); pairs whose balls no longer overlap are forgotten first (a wall needs the overlap)"""
        n = self.n
        keep = (x[self.mem_b]-x[self.mem_a]).norm(dim=1) < r[self.mem_a]+r[self.mem_b]
        forgot = int((~keep).sum())
        self.mem_a, self.mem_b, self.mem_D = self.mem_a[keep], self.mem_b[keep], self.mem_D[keep]
        i, j = geo['i'], geo['j']
        lo, hi = torch.minimum(i, j), torch.maximum(i, j); sg = torch.where(i < j, 1., -1.).to(f64)[:, None]
        key = lo*n+hi; mk = self.mem_a*n+self.mem_b
        order = torch.argsort(mk); mks = mk[order]
        pos = torch.searchsorted(mks, key).clamp(max=len(mks)-1)
        found = mks[pos] == key
        D = torch.zeros(len(i), 3, dtype=f64, device=self.dev)
        D[found] = self.mem_D[order[pos[found]]]*sg[found]
        new = ~found
        if bool(new.any()):
            if self.new_ref == 'rest':                                                 # the two material points' rest offset
                Dn = self.X0[j[new]]-self.X0[i[new]]
            else:
                Dn = torch.linalg.solve(.5*(F[i[new]]+F[j[new]]), (x[j[new]]-x[i[new]])[:, :, None])[:, :, 0]
            D[new] = Dn
            self.mem_a, self.mem_b = torch.cat([self.mem_a, lo[new]]), torch.cat([self.mem_b, hi[new]])
            self.mem_D = torch.cat([self.mem_D, Dn*sg[new]])
        self.pair_stats = dict(new=int(new.sum()), forgotten=forgot, memory=len(self.mem_a))
        return D

    def operators(self, x, F, geo, D=None):
        """the step's linear maps, built at the step start (x^n, F^n, geometry); f_mode 'total': the fits' levers are the
        walls' material reference vectors D (pair_refs; at rest the rest offsets) and F = kin(x) itself (section 20)"""
        i, j, A, c = geo['i'], geo['j'], geo['A'], geo['c']
        if 'contact' in geo:                                                           # (self_contact: the fits, the jump term, the
            mk = ~geo['contact']                                                       # stabilisation and the aggregation over the
            i, j, A, c = i[mk], j[mk], A[mk], c[mk]                                    # material walls; the centroids from all)
        d = x[j]-x[i]; L = d.norm(dim=1); nrm = d/L[:, None]
        total = self.f_mode == 'total'
        if total and D is None:
            D = self.X0[j]-self.X0[i]
        lev = D if total else d
        h = ((c-x[i])*nrm).sum(1); hj = L-h; w = A/L if self.site_w == 'A/l' else A
        n = self.n
        cur = not (total and self.fit_w in ('rest', 'lens'))                           # the fits on the current cut (else unused:
        if cur:                                                                        # 10-28 lion, a cell that lost its walls
            M = torch.zeros(n, 3, 3, dtype=f64, device=self.dev)                       # made them singular)
            dd = w[:, None, None]*lev[:, :, None]*lev[:, None, :]
            M.index_add_(0, i, dd); M.index_add_(0, j, dd)
            Mi = torch.linalg.pinv(M) if self.fit_cap is not None or self.shift == 'none' else torch.linalg.inv(M)   # (capped / shift 'none': a cell may be rank-deficient)
        delta = self.cut_centroids(x, geo) if self.centroid == 'cut' else torch.einsum('nab,nb->na', F, self.dlt0)
        detF = torch.linalg.det(F)
        Jm = detF[:, None, None]*F@self.J0@F.transpose(1, 2)
        ends = []
        cc = x+delta
        for s_, o_, dv, nn, hh in ((i, j, lev, nrm, h), (j, i, -lev, -nrm, hj)):
            row = torch.einsum('fa,fab->fb', w[:, None]*dv, Mi[s_]) if cur else None   # w d^T M_s^-1
            cen = (row*delta[s_]).sum(1) if cur else None                         # w d^T M_s^-1 delta_s
            if self.smooth == 'centroid':                                         # where the wall's plane cuts c_s -> c_o
                t = ((c-cc[s_])*nn).sum(1)/((cc[o_]-cc[s_])*nn).sum(1)
            else:                                                                 # ... cuts x_s -> x_o
                t = hh/L
            smo = (A*t)[:, None]*nn/geo['V'][s_][:, None]                          # A t n / V_s
            ends.append((s_, o_, row, cen, smo))
        Kj = (A/(3*L))
        crow = None
        if self.shift == 'displacement' and cur:
            e = D+self.carry0[j]-self.carry0[i] if total else cc[j]-cc[i]               # material / current centroid levers
            wc = A/e.norm(dim=1)
            Mc = torch.zeros(n, 3, 3, dtype=f64, device=self.dev)
            ee = wc[:, None, None]*e[:, :, None]*e[:, None, :]
            Mc.index_add_(0, i, ee); Mc.index_add_(0, j, ee)
            Mci = torch.linalg.pinv(Mc) if self.fit_cap is not None else torch.linalg.inv(Mc)
            crow = [torch.einsum('fa,fab->fb', wc[:, None]*e, Mci[i]), torch.einsum('fa,fab->fb', -wc[:, None]*e, Mci[j])]
        cat = lambda k: torch.cat([e_[k] for e_ in ends])                          # both ends of every wall in one list
        KS, KO = cat(0), cat(1)
        R_, RC_ = (cat(2), torch.cat(crow) if crow is not None else None) if cur else (None, None)
        if cur and self.fit_cap is not None:
            KS, KO, R_, RC_ = self._capped_fits(i, j, w, lev, x, cc, M, Mc if self.shift == 'displacement' else None, wc if self.shift == 'displacement' else None,
                                                e if self.shift == 'displacement' else None, delta)
        if total and self.fit_w == 'rest':                                             # the fits on the rest neighbours with the
            if self.rest_fit is None:                                                  # rest weights: a constant linear map
                i0, j0, A0 = self.rest_walls                                           # (section 20, 10-28)
                D0 = self.X0[j0]-self.X0[i0]; w0 = A0/D0.norm(dim=1) if self.site_w == 'A/l' else A0
                E0 = D0+self.carry0[j0]-self.carry0[i0]; wc0 = A0/E0.norm(dim=1)
                M0 = torch.zeros(n, 3, 3, dtype=f64, device=self.dev); Mc0 = torch.zeros_like(M0)
                for M_, L_, w_ in ((M0, D0, w0), (Mc0, E0, wc0)):
                    q = w_[:, None, None]*L_[:, :, None]*L_[:, None, :]
                    M_.index_add_(0, i0, q); M_.index_add_(0, j0, q)
                pinv = lambda M_, P_: (lambda s_: torch.linalg.inv(M_+s_*P_)-P_/s_)(M_.diagonal(dim1=1, dim2=2).sum(1)[:, None, None]/3+(M_.abs().sum((1, 2)) == 0)[:, None, None])
                M0i, Mc0i = pinv(M0, self.null_site), pinv(Mc0, self.null_cen)    # inverses (pseudo- for degenerate cells)
                self.rest_fit = (torch.cat([i0, j0]), torch.cat([j0, i0]),
                                 torch.cat([torch.einsum('fa,fab->fb', w0[:, None]*D0, M0i[i0]), torch.einsum('fa,fab->fb', -w0[:, None]*D0, M0i[j0])]),
                                 torch.cat([torch.einsum('fa,fab->fb', wc0[:, None]*E0, Mc0i[i0]), torch.einsum('fa,fab->fb', -wc0[:, None]*E0, Mc0i[j0])]))
            KS, KO, R_, RC_ = self.rest_fit
        if total and self.fit_w == 'lens':                                             # the fits over every overlapping pair,
            a, b, wl = self.lens_pairs(x, geo['r'])                                    # weighted by the lens volume, with the
            Dl = self.X0[b]-self.X0[a]                                                 # rest offsets (section 20, 10-28)
            El = Dl+self.carry0[b]-self.carry0[a]
            Ml = torch.zeros(n, 3, 3, dtype=f64, device=self.dev); Mcl = torch.zeros_like(Ml)
            for M_, L_ in ((Ml, Dl), (Mcl, El)):
                q = wl[:, None, None]*L_[:, :, None]*L_[:, None, :]
                M_.index_add_(0, a, q); M_.index_add_(0, b, q)
            Mli, Mcli = torch.linalg.inv(Ml), torch.linalg.inv(Mcl)
            KS, KO = torch.cat([a, b]), torch.cat([b, a])
            R_ = torch.cat([torch.einsum('fa,fab->fb', wl[:, None]*Dl, Mli[a]), torch.einsum('fa,fab->fb', -wl[:, None]*Dl, Mli[b])])
            RC_ = torch.cat([torch.einsum('fa,fab->fb', wl[:, None]*El, Mcli[a]), torch.einsum('fa,fab->fb', -wl[:, None]*El, Mcli[b])])
        ops_ = dict(i=i, j=j, ends=ends, Jm=Jm, Kj=Kj, x0=torch.zeros(self.n+len(self.deg_cells), 3, dtype=f64, device=self.dev) if total else x.clone(),
                    F0=self.I3.expand(self.n, 3, 3).contiguous() if total else F.clone(), total=total, L=L, nrm=nrm, A=A,
                    delta=delta, carry=self.carry0 if total else delta, crow=crow,
                    S=cat(0), O=cat(1), R=R_, CEN=cat(3) if cur else None, SM=cat(4), RC=RC_, KS=KS, KO=KO,
                    b=D if total else self.X0[j]-self.X0[i])
        if total and self.grad == 'corrected':                                        # (the integration-constraint fit, 10-29)
            ops_.update(KS=self.cor_ops[0], KO=self.cor_ops[1], R=self.cor_ops[2], onestep=True)
        if self.shift == 'none':                                                       # (10-29: the site-level fit alone, no centroid step)
            ops_['onestep'] = True
        if self.agg > 0:                                                               # (10-29: aggregation of the tiny cells' fit, the small-cell
            tiny = self.V0 < self.agg*self.V0.median()                                 # problem, cf. cell merging / aggregated FEM: a cell below
            m1 = tiny[i] & ~tiny[j]; m2 = tiny[j] & ~tiny[i]                           # agg x the median volume takes the V0-weighted mean of its
            as_, ao_ = torch.cat([i[m1], j[m2]]), torch.cat([j[m1], i[m2]])            # non-tiny wall neighbours' increment on this cut)
            den = torch.zeros(n, dtype=f64, device=self.dev).index_add_(0, as_, self.V0[ao_])
            ops_['agg'] = (as_, ao_, self.V0[ao_]/den[as_], tiny & (den > 0))
        if 'contact' in geo and bool(geo['contact'].any()):                            # (self_contact: the pairs across the contact
            ci, cj = geo['i'][geo['contact']], geo['j'][geo['contact']]                # walls, see _sc)
            dd_ = x[cj]-x[ci]; rr = geo['r']
            db = geo['sc_db'][geo['contact']] if self.sc_birth else 0.
            ops_['sc'] = (ci, cj, dd_/dd_.norm(dim=1, keepdim=True), rr[ci]+rr[cj]-db, self.E_sc*torch.sqrt(rr[ci]*rr[cj]/(rr[ci]+rr[cj])))
        if self.sc == 'rest':                                                          # (10-29: the material connection is the rest one:
            ri, rj, rA = self.rb_i, self.rb_j, self.rb_A                               # the fit and the jump term over every rest pair,
            dr = x[rj]-x[ri]; Lr = dr.norm(dim=1); wr = rA/Lr                          # rest areas, current offsets; the same as the
            Mr = torch.zeros(n, 3, 3, dtype=f64, device=self.dev)                      # cut's fit at rest)
            qr = wr[:, None, None]*dr[:, :, None]*dr[:, None, :]
            Mr.index_add_(0, ri, qr); Mr.index_add_(0, rj, qr)
            Mri = torch.linalg.pinv(Mr)
            ops_.update(KS=torch.cat([ri, rj]), KO=torch.cat([rj, ri]),
                        R=torch.cat([torch.einsum('fa,fab->fb', wr[:, None]*dr, Mri[ri]), torch.einsum('fa,fab->fb', -wr[:, None]*dr, Mri[rj])]),
                        i=ri, j=rj, b=self.X0[rj]-self.X0[ri], Kj=rA/(3*Lr))
        if self.bond_floor > 0:                                                        # (10-29: the rest pairs' jump stiffness floor;
            kw = torch.minimum(i, j)*n+torch.maximum(i, j)                             # a rest pair without a material wall now joins
            pos = torch.searchsorted(self.rb_keys, kw).clamp(max=len(self.rb_keys)-1)  # the jump term as a virtual wall)
            isr = self.rb_keys[pos] == kw
            Kj2 = Kj.clone(); Kj2[isr] = torch.maximum(Kj[isr], self.bond_floor*self.rb_K[pos[isr]])
            miss = ~torch.isin(self.rb_keys, kw)
            ai, aj = self.rb_i[miss], self.rb_j[miss]
            ops_.update(i=torch.cat([i, ai]), j=torch.cat([j, aj]), Kj=torch.cat([Kj2, self.bond_floor*self.rb_K[miss]]),
                        b=torch.cat([ops_['b'], self.X0[aj]-self.X0[ai]]))
            self.floor_stats = dict(floored=int((Kj2 > Kj).sum()), virtual=int(miss.sum()))
        if total and self.grad.startswith('gg_'):                                     # (the divergence-theorem gradient, 10-29)
            ops_.update(KS=self.gg_ops[0], KO=self.gg_ops[1], T=self.gg_ops[2], onestepT=True)
            if self.gg_off is not None:
                ops_['F_off'] = self.gg_off
        return ops_

    # ---- the energy
    def def_grad(self, x, ops):
        """x: the sites, then the degenerate cells' free columns (deg_cells, deg_q) when there are any"""
        dx = x-ops['x0']
        cols, dx = dx[self.n:], dx[:self.n]
        S, O = ops['KS'], ops['KO']
        if ops.get('onestepT'):
            G = torch.zeros(self.n, 3, 3, dtype=f64, device=self.dev).index_add(0, S, torch.einsum('pijc,pc->pij', ops['T'], dx[O]-dx[S]))
        else:
            G = torch.zeros(self.n, 3, 3, dtype=f64, device=self.dev).index_add(0, S, (dx[O]-dx[S])[:, :, None]*ops['R'][:, None, :])
        if ops.get('onestep') or ops.get('onestepT'):
            Gc = G
        elif self.shift == 'displacement':
            u = dx+torch.einsum('nab,nb->na', G, ops['carry'])
            Gc = torch.zeros_like(G).index_add(0, S, (u[O]-u[S])[:, :, None]*ops['RC'][:, None, :])
        else:
            Gc = G.index_add(0, S, ops['CEN'][:, None, None]*(G[O]-G[S]))
        if 'agg' in ops:
            Gc = self._agg(Gc, ops)
        if len(cols):
            Gc = Gc.index_add(0, self.deg_cells, cols[:, :, None]*self.deg_q[:, None, :])
        Fx = Gc@ops['F0'] if ops['total'] else (self.I3+Gc)@ops['F0']
        return Fx+ops['F_off'] if 'F_off' in ops else Fx

    def elastic(self, x, ops, terms=False):
        F = self.def_grad(x, ops)
        J = torch.linalg.det(F)
        C = F.transpose(1, 2)@F
        lnJ = torch.log(J.clamp(min=1e-300))
        psi = .5*self.mu*(C.diagonal(dim1=1, dim2=2).sum(1)-3)-self.mu*lnJ+.5*self.lam*lnJ**2
        e = dict(cells=self.cell_w*(self.V0*psi).sum())
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
            dlt = (x[j]-x[i])-.5*torch.einsum('fab,fb->fa', F[i]+F[j], ops['b'])
            nc = self._jn(x, ops)
            e['jump'] = .5*(ops['Kj']*(self.mu*(dlt*dlt).sum(1)+(self.lam+self.mu)*(dlt*nc).sum(1)**2)).sum()
        if 'sc' in ops:
            e['selfc'] = self._sc(x, ops)[0]
        if self.tet_w > 0:
            e['tet'], _ = self.tet_energy(x, ops)
        if self.wall_w > 0:
            e['wall'], _ = self.wall_energy(x, ops)
        if self.cellhg_w > 0:
            e['cellhg'], _ = self.cellhg_energy(x, ops)
        return (e, F) if terms else sum(e.values())

    def _jn(self, y, ops):
        """the jump term's normal per wall: along the pair at y (jump_n 'current'), or along the pair at the step's cut
        (jump_n 'start', 10-29: fixed within the step, so the term is exactly quadratic in the positions; the wall's
        normal of the cut. Near-twin sites, 0.07 h apart on the lion, turned the current normal by ~90 degrees within one
        Newton step and broke the solve's quadratic model)"""
        if self.jump_n == 'start':
            y = ops['x0']
        a = y[ops['j']]-y[ops['i']]
        return a/a.norm(dim=1, keepdim=True)

    def classify_walls(self, x, r, geo, F=None, K=24, chunk=512):
        """self_contact (10-29): each wall of this cut is 'contact' or 'material' by where it grew. A wall that was in the
        last classified cut keeps its label. A new wall (i, j) is contact when, on that last cut, the segment between the
        two sites crossed air or a contact wall, else material: a wall grown where there was material is the material
        rearranging itself, one grown in air or out of a contact interface is two surfaces touching (or sliding along each
        other) (self_contact 'air'). Along the segment the owners (least power distance) at K points; between two consecutive different
        owners a, b the exact point where their power distances are equal: air when it lies outside a's ball (10-29: the
        gaps closing between the lion's leg and chest were too thin for any sample, even 200, to land in; the exact
        crossing found them), across a contact wall when (a, b) is one. 10-29: on the lion the air test also took ~17 %
        of the new walls between material neighbours (the dimples between neighbouring surface balls), and the bottom
        layer fell apart. self_contact 'compat': a new wall is contact when the pair's offset is not explained by the
        deformation of either side from rest, max_k |x_j - x_i - F_k (X0_j - X0_i)| > sc_thr h (k = i, j; one side at a
        time: the mean of the two F of a fold's sides collapses the offset; on the lion's frames the foreign walls,
        >= 4 rest hops, 99 % above h, median 7 h; new walls of material neighbours, 1-2 hops, at most 2.4 h). A contact
        wall never turns material while it lives (no adhesion). The first cut (no last one): every wall material"""
        n = self.n
        i, j = geo['i'], geo['j']
        key = torch.minimum(i, j)*n+torch.maximum(i, j)
        contact = torch.zeros(len(i), dtype=torch.bool, device=self.dev); born = torch.zeros_like(contact)
        db = torch.zeros(len(i), dtype=f64, device=self.dev)                            # (each contact wall's overlap at its birth)
        pv = self.sc_prev
        if pv is not None:
            pos = torch.searchsorted(pv['keys'], key).clamp(max=len(pv['keys'])-1)
            found = pv['keys'][pos] == key
            contact[found] = pv['contact'][pos[found]]; db[found] = pv['db'][pos[found]]
            born = ~found
            if self.sc in ('bond', 'rest'):
                pass
            elif self.sc == 'skin':                                                    # (the rest-skin rule: the pair alone decides,
                pass                                                                   # every wall each step, below)
            elif bool(born.any()) and self.sc == 'cap':                                 # (the user's rule, 10-29: outside a cap is
                bi, bj = i[born], j[born]; xp, rp = pv['x'], pv['r']                    # air, so caps meeting never make a connection:
                D = xp[bj]-xp[bi]; L2 = (D*D).sum(1)                                    # the two balls apart on the last cut. Or the
                apart = L2.sqrt() >= rp[bi]+rp[bj]                                      # contact interface growing / sliding: the lens
                q = xp[bi]+((L2+rp[bi]**2-rp[bj]**2)/(2*L2))[:, None]*D                 # point (equal power on the segment) owned on the
                own = torch.empty(len(q), dtype=torch.long, device=self.dev)           # last cut by a third cell in contact with i or j)
                for c0 in range(0, len(q), chunk):
                    own[c0:c0+chunk] = (((q[c0:c0+chunk, None, :]-xp[None])**2).sum(2)-(rp**2)[None]).argmin(1)
                ck = pv['keys'][pv['contact']]
                inh = (torch.isin(torch.minimum(own, bi)*n+torch.maximum(own, bi), ck) | torch.isin(torch.minimum(own, bj)*n+torch.maximum(own, bj), ck)) & (own != bi) & (own != bj)
                contact[born] = apart | inh
                self.sc_stats_extra = dict(apart=int(apart.sum()), inherited=int((inh & ~apart).sum()))
            elif bool(born.any()) and self.sc == 'compat':
                bi, bj = i[born], j[born]
                d0, dx = self.X0[bj]-self.X0[bi], x[bj]-x[bi]
                mis = torch.maximum((dx-mv(F[bi], d0)).norm(dim=1), (dx-mv(F[bj], d0)).norm(dim=1))
                contact[born] = mis > self.sc_thr*self.h_med
            elif bool(born.any()):
                bi, bj = i[born], j[born]
                xp, rp = pv['x'], pv['r']
                s_ = torch.arange(1, K+1, dtype=f64, device=self.dev)/(K+1)
                pts = (xp[bi][:, None, :]+s_[None, :, None]*(xp[bj]-xp[bi])[:, None, :]).reshape(-1, 3)
                own = torch.empty(len(pts), dtype=torch.long, device=self.dev); air = torch.empty(len(pts), dtype=torch.bool, device=self.dev)
                for c0 in range(0, len(pts), chunk):                                   # (the owner: the least power distance;
                    pw = ((pts[c0:c0+chunk, None, :]-xp[None])**2).sum(2)-(rp**2)[None]   # air where it is positive)
                    m_, a_ = pw.min(1); own[c0:c0+chunk] = a_; air[c0:c0+chunk] = m_ > 0
                seq = torch.cat([bi[:, None], own.view(-1, K), bj[:, None]], 1)
                a, b = seq[:, :-1], seq[:, 1:]
                z_, o_ = torch.zeros(len(bi), 1, dtype=f64, device=self.dev), torch.ones(len(bi), 1, dtype=f64, device=self.dev)
                sv = torch.cat([z_, s_[None].expand(len(bi), K), o_], 1)[:, :-1]
                P, D = xp[bi][:, None, :], (xp[bj]-xp[bi])[:, None, :]
                Pa, Pb = xp[a], xp[b]                                                   # (pow_a = pow_b is linear in s)
                c1 = 2*((Pb-Pa)*D).sum(2); c0_ = ((P-Pa)**2).sum(2)-((P-Pb)**2).sum(2)-rp[a]**2+rp[b]**2
                q = P+torch.where(c1.abs() > 0, -c0_/c1, sv)[..., None]*D
                air_x = (a != b) & ((((q-Pa)**2).sum(2)-rp[a]**2) > 0)
                cross = torch.isin(torch.minimum(a, b)*n+torch.maximum(a, b), pv['keys'][pv['contact']]) & (a != b)
                contact[born] = air.view(-1, K).any(1) | air_x.any(1) | cross.any(1)
        if self.sc == 'rest':                                                          # (10-29: material iff a rest pair; a function of the pair)
            contact = ~torch.isin(key, self.rb_keys)
        if self.sc == 'bond':                                                          # (10-29: the bond family with visibility; the new
            sel = born if pv is not None else torch.ones_like(contact)                # walls only, a function of the pair)
            if bool(sel.any()):
                bi, bj = i[sel], j[sel]
                far = (self.X0[bj]-self.X0[bi]).norm(dim=1) > self.sc_thr*self.h_med
                hid = self._rest_outside(bi, bj) > self.sc_eps*self.h_med
                contact[sel] = (far | hid) & ~torch.isin(key[sel], self.sc_rest)
        if self.sc == 'skin':                                                          # (10-29, the user's rest-surface rule: contact
            contact = self.sc_skin[i] & self.sc_skin[j] & ~torch.isin(key, self.sc_near)   # = two pieces of the rest skin >= 3 steps
        o = torch.argsort(key)                                                         # apart on it came together)
        nb = born & contact
        if bool(nb.any()):
            db[nb] = (r[i[nb]]+r[j[nb]]-(x[j[nb]]-x[i[nb]]).norm(dim=1)).clamp(min=0.)
        geo['sc_db'] = db
        self.sc_prev = dict(x=x.clone(), r=r.clone(), keys=key[o], contact=contact[o], db=db[o])
        self.sc_stats = dict(walls=len(i), contact=int(contact.sum()), born=int(born.sum()), born_contact=int((contact & born).sum()))
        return contact

    def _bond_family(self, delta, eps):
        """the bond family (10-29; the peridynamic horizon with the meshless visibility criterion): the rest walls, plus
        the pairs within delta h at rest whose rest segment stays in the body (no point more than eps h outside the rest
        union of balls), as sorted keys min n + max; and the pairs outside it whose balls overlap at rest, with that
        overlap (the contact counts from it)"""
        n, X, r, h = self.n, self.X0, self.r0, self.h_med
        G0, _ = self.recut.geometry(X, r); kp = G0.wall_A > 0
        rest = torch.minimum(G0.i[kp], G0.j[kp])*n+torch.maximum(G0.i[kp], G0.j[kp])
        A_, B_ = [], []
        for c0 in range(0, n, 1024):
            a, b = torch.nonzero(torch.cdist(X[c0:c0+1024], X) <= delta*h, as_tuple=True); a = a+c0
            k = a < b; A_.append(a[k]); B_.append(b[k])
        a, b = torch.cat(A_), torch.cat(B_)
        vis = torch.cat([self._rest_outside(a[c0:c0+65536], b[c0:c0+65536]) for c0 in range(0, len(a), 65536)]) <= eps*h
        bonds = torch.unique(torch.cat([a[vis]*n+b[vis], rest]))
        key = a*n+b; ov = r[a]+r[b]-(X[b]-X[a]).norm(dim=1)
        m = (ov > 0) & ~torch.isin(key, bonds)
        o = torch.argsort(key[m])
        return bonds, key[m][o], ov[m][o]

    def _contact_pairs(self, x, r, ops):
        """cut_bonds: the walls of the full recut at the step's start between cells outside each other's bond family (two
        parts' caps meeting), as ops['sc'] for _sc (Hertz between their balls along the step's pair direction, from 0
        when the caps first meet; less the rest overlap for pairs already so at rest: none, the rest walls are bonds)"""
        n = self.n
        Gf, _ = self.recut_full.geometry(x, r); kf = Gf.wall_A > 0
        i, j = torch.minimum(Gf.i[kf], Gf.j[kf]), torch.maximum(Gf.i[kf], Gf.j[kf])
        key = i*n+j; nb = ~torch.isin(key, self.bonds); i, j, key = i[nb], j[nb], key[nb]
        d = x[j]-x[i]; L = d.norm(dim=1)
        off = (r[i]+r[j]-L).clamp(min=0.)                                             # (a pair's first touch: its overlap now;
        pv = getattr(self, 'cb_prev', None)                                            # one met before keeps its first touch's)
        if pv is not None and len(pv[0]) and len(i):
            pos = torch.searchsorted(pv[0], key).clamp(max=len(pv[0])-1)
            f_ = pv[0][pos] == key; off[f_] = pv[1][pos[f_]]
        o_ = torch.argsort(key); self.cb_prev = (key[o_], off[o_])
        if len(i):
            ops['sc'] = (i, j, d/L[:, None], r[i]+r[j]-off, self.E_sc*torch.sqrt(r[i]*r[j]/(r[i]+r[j])))
        self.sc_keys_last = key
        self.sc_stats = dict(walls=0, contact=len(i), born=0, born_contact=0)

    def _rest_outside(self, a, b, K=16, chunk=1024):
        """per pair (a, b) the largest distance outside the rest union of balls along the rest segment X0_a -> X0_b (the
        visibility criterion; negative: the segment inside the body)"""
        s_ = torch.linspace(0, 1, K+2, dtype=f64, device=self.dev)[1:-1]
        P = (self.X0[a][:, None, :]+s_[None, :, None]*(self.X0[b]-self.X0[a])[:, None, :]).reshape(-1, 3)
        d = torch.empty(len(P), dtype=f64, device=self.dev)
        for c0 in range(0, len(P), chunk):
            d[c0:c0+chunk] = (torch.cdist(P[c0:c0+chunk], self.X0)-self.r0[None]).min(1).values
        return d.view(-1, K).max(1).values

    def _sc(self, y, ops):
        """self-contact (10-29): Hertz between the balls across each contact wall, along the pair's direction on the step's
        cut n0 (fixed within the step, as the jump term's normal): overlap d = r_i + r_j - n0 . (y_j - y_i) with the step
        start's radii, energy 8/15 E* sqrt(R) d^5/2 (E* = E / (2 (1 - nu^2)), R = r_i r_j / (r_i + r_j)); frictionless,
        compression only, as the ground. Returns the energy, the gradient and the products' stiffness 2 E* sqrt(R) d^1/2"""
        ci, cj, n0, rs, c = ops['sc']
        dl = (rs-((y[cj]-y[ci])*n0).sum(1)).clamp(min=0.)
        f = 4/3*c*dl**1.5
        g = torch.zeros_like(y).index_add_(0, ci, f[:, None]*n0).index_add_(0, cj, -f[:, None]*n0)
        return (8/15*c*dl**2.5).sum(), g, 2*c*dl.sqrt()

    def wall_jump(self, y, ops):
        """the jump term's energy per wall at y (as elastic() takes it) and the walls' keys min(i, j) n + max(i, j)"""
        F = self.def_grad(y, ops)
        i, j = ops['i'], ops['j']
        a = y[j]-y[i]
        dl = a-.5*mv(F[i]+F[j], ops['b']); nc = self._jn(y, ops)
        e = .5*ops['Kj']*(self.mu*(dl*dl).sum(1)+(self.lam+self.mu)*(dl*nc).sum(1)**2)
        return torch.minimum(i, j)*self.n+torch.maximum(i, j), e

    # ---- the energy, its gradient and Hessian-vector products written out (the solve's path; elastic() is the autograd
    # reference they are checked against). F is linear in x: dF = Gc(dx) F0, so every product is the chain
    # dx -> dGc -> dF -> each term's second derivative -> back through the adjoint of the chain
    def kin(self, v, ops):
        """Gc(v) for an increment v of the sites (F = (I + Gc) F0), then of the degenerate cells' free columns if any"""
        S, O = ops['KS'], ops['KO']
        cols, v = v[self.n:], v[:self.n]
        if ops.get('onestepT'):
            G = torch.zeros(self.n, 3, 3, dtype=f64, device=self.dev).index_add_(0, S, torch.einsum('pijc,pc->pij', ops['T'], v[O]-v[S]))
        else:
            G = torch.zeros(self.n, 3, 3, dtype=f64, device=self.dev).index_add_(0, S, (v[O]-v[S])[:, :, None]*ops['R'][:, None, :])
        if ops.get('onestep') or ops.get('onestepT'):
            Gc = G
        elif self.shift == 'displacement':
            u = v+mv(G, ops['carry'])
            Gc = torch.zeros_like(G).index_add_(0, S, (u[O]-u[S])[:, :, None]*ops['RC'][:, None, :])
        else:
            Gc = G.index_add(0, S, ops['CEN'][:, None, None]*(G[O]-G[S]))
        if 'agg' in ops:
            Gc = self._agg(Gc, ops)
        if len(cols):
            Gc = Gc.index_add_(0, self.deg_cells, cols[:, :, None]*self.deg_q[:, None, :])
        return Gc

    def _agg(self, G, ops):
        """the tiny cells' G replaced by their non-tiny neighbours' V0-weighted mean (linear; _agg_T its adjoint)"""
        as_, ao_, aw, used = ops['agg']
        G2 = G.clone(); G2[used] = 0.
        return G2.index_add(0, as_, aw[:, None, None]*G[ao_])

    def _agg_T(self, Gb, ops):
        as_, ao_, aw, used = ops['agg']
        G2 = Gb.clone(); G2[used] = 0.
        return G2.index_add(0, ao_, aw[:, None, None]*Gb[as_])

    def kin_T(self, Gb, ops):
        """the adjoint of kin: the sites' gradient from dU/dGc (then the degenerate cells' free columns' if any)"""
        S, O = ops['KS'], ops['KO']
        if 'agg' in ops:
            Gb = self._agg_T(Gb, ops)
        Gb0 = Gb
        z = torch.zeros(self.n, 3, dtype=f64, device=self.dev)
        if ops.get('onestepT'):
            q = torch.einsum('pijc,pij->pc', ops['T'], Gb[S])
            ub = z.index_add(0, O, q).index_add_(0, S, -q)
            return torch.cat([ub, mv(Gb0[self.deg_cells], self.deg_q)]) if len(self.deg_cells) and ops['total'] else ub
        if ops.get('onestep'):
            ub = z.clone()
        elif self.shift == 'displacement':
            q = mv(Gb[S], ops['RC'])
            ub = z.index_add(0, O, q).index_add_(0, S, -q)
            Gb = ub[:, :, None]*ops['carry'][:, None, :]
        else:
            W = ops['CEN'][:, None, None]*Gb[S]
            ub = z.clone()
            Gb = Gb.index_add(0, O, W).index_add_(0, S, -W)
        q = mv(Gb[S], ops['R'])
        ub = ub.index_add_(0, O, q).index_add_(0, S, -q)
        return torch.cat([ub, mv(Gb0[self.deg_cells], self.deg_q)]) if len(self.deg_cells) and ops['total'] else ub

    def _stab_pull(self, dE, ops):
        """dU/dE per cell from the strains' smoothed derivatives dE [s, k, a, b] (and the energy's density Q)"""
        S, O, SM = ops['S'], ops['O'], ops['SM']
        tr = dE.diagonal(dim1=2, dim2=3).sum(2)
        Jm = ops['Jm']
        Q = 2*self.mu*(Jm[:, :, :, None, None]*dE[:, None]).sum(2)+self.lam*(Jm*tr[:, None, :]).sum(2)[:, :, None, None]*self.I3
        W = (SM[:, :, None, None]*Q[S]).sum(1)
        return torch.zeros(self.n, 3, 3, dtype=f64, device=self.dev).index_add_(0, O, W).index_add_(0, S, -W), Q

    def _stab_push(self, E, ops):
        S, O, SM = ops['S'], ops['O'], ops['SM']
        return torch.zeros(self.n, 3, 3, 3, dtype=f64, device=self.dev).index_add_(0, S, SM[:, :, None, None]*(E[O]-E[S])[:, None])

    def state(self, y, ops):
        """everything the gradient and the products at y need; None where some J <= 0"""
        Gc = self.kin(y-ops['x0'], ops)
        F = Gc if ops['total'] else mm(self.I3+Gc, ops['F0'])
        if 'F_off' in ops:                                                             # (gg_c: the closure acts on I + G(x - X0))
            F = F+ops['F_off']
        J = torch.linalg.det(F)
        if bool((J <= 0).any()):
            return None
        FiT = torch.linalg.inv(F).transpose(1, 2); lnJ = torch.log(J)
        C = mm(F.transpose(1, 2), F)
        e = dict(cells=self.cell_w*float((self.V0*(.5*self.mu*(C.diagonal(dim1=1, dim2=2).sum(1)-3)-self.mu*lnJ+.5*self.lam*lnJ**2)).sum()))
        P = self.cell_w*self.V0[:, None, None]*(self.mu*(F-FiT)+self.lam*lnJ[:, None, None]*FiT)
        st = dict(F=F, FiT=FiT, lnJ=lnJ)
        if self.stab:
            dE = self._stab_push(.5*(C-self.I3), ops)
            Eb, Q = self._stab_pull(dE, ops)
            e['stab'] = .5*float((dE*Q).sum())
            P = P+mm(F, Eb); st['Eb'] = Eb
        yg = torch.zeros_like(y)
        if self.jump:
            i, j, b, Kj = ops['i'], ops['j'], ops['b'], ops['Kj'][:, None]
            a = y[j]-y[i]; la = a.norm(dim=1, keepdim=True); nc = a/la if self.jump_n == 'current' else self._jn(y, ops)
            dl = a-.5*mv(F[i]+F[j], b); sn = (dl*nc).sum(1, keepdim=True)
            e['jump'] = .5*float((Kj*(self.mu*(dl*dl).sum(1, keepdim=True)+(self.lam+self.mu)*sn**2)).sum())
            gd = Kj*(self.mu*dl+(self.lam+self.mu)*sn*nc); gn = Kj*(self.lam+self.mu)*sn*dl
            if self.jump_n == 'current':
                tn = gn-nc*(nc*gn).sum(1, keepdim=True)
                ga = gd+tn/la
            else:
                tn, ga = None, gd
            PF = -.5*gd[:, :, None]*b[:, None, :]
            P = P.index_add(0, i, PF).index_add_(0, j, PF)
            yg.index_add_(0, j, ga).index_add_(0, i, -ga)
            st.update(nc=nc, la=la, dl=dl, sn=sn, gn=gn, tn=tn)
        if 'sc' in ops:
            E_, g_, st['ksc'] = self._sc(y, ops); e['selfc'] = float(E_); yg = yg+g_
        if self.tet_w > 0 or self.wall_w > 0 or self.cellhg_w > 0:                                           # (autograd: their products by one
            with torch.enable_grad():                                                   # backward through the retained gradient
                yt = y.detach().requires_grad_(True)                                    # graph)
                Ea = 0.
                for nm, fn in (('tet', self.tet_energy), ('wall', self.wall_energy), ('cellhg', self.cellhg_energy)):
                    if getattr(self, nm+'_w') > 0:
                        E_, j_ = fn(yt, ops)
                        if not math.isfinite(float(E_)):
                            return None
                        e[nm] = float(E_); st[nm+'_minJ'] = j_; Ea = Ea+E_
                gt, = torch.autograd.grad(Ea, yt, create_graph=True)
            st['ag'] = (gt, yt)
            yg = yg+gt.detach()
        Parg = P if ops['total'] else mm(P, ops['F0'].transpose(1, 2))
        if 'visc' in ops:
            Uv, Pv = self.visc_terms(Gc, ops)
            e['visc'] = float(Uv)
            st['visc_grad'] = self.kin_T(Pv, ops)
            Parg = Parg+Pv
        st['e'] = e
        st['grad'] = self.kin_T(Parg, ops)+yg
        return st

    def visc_terms(self, Gc, ops):
        """Kelvin-Voigt (10-29; the DEM's definition: eta = beta mu, zeta = beta (lam + 2 mu / 3)) in the step's incremental
        potential: dt V_n [eta |dev d|^2 + zeta / 2 (tr d)^2], d = sym(dG) / dt the rate of deformation of the step's
        increment dG = (F - F_n) F_n^-1 (f_mode 'increment': the kinematic map's own increment), V_n = V0 det F_n. Quadratic in
        the positions, zero for a rigid rotation to first order. Returns the potential and dU / d(kinematic map's G)"""
        vs = ops['visc']; dt, Vn = vs['dt'], vs['Vn']
        dG = Gc if not ops['total'] else (Gc+ops.get('F_off', 0.)-vs['Fn'])@vs['Fni']
        D = .5*(dG+dG.transpose(1, 2)); tr = D.diagonal(dim1=1, dim2=2).sum(1)
        dev = D-tr[:, None, None]/3*self.I3
        eta, zeta = self.beta*self.mu, self.beta*(self.lam+2*self.mu/3)
        Uv = ((Vn/dt)*(eta*(dev*dev).sum((1, 2))+.5*zeta*tr*tr)).sum()
        Pv = (Vn/dt)[:, None, None]*(2*eta*dev+zeta*tr[:, None, None]*self.I3)
        if ops['total']:
            Pv = Pv@vs['Fni'].transpose(1, 2)
        return Uv, Pv

    def hvp(self, st, d, ops):
        """the elastic energy's Hessian at the state st times d (n, 3)"""
        F, FiT, lnJ = st['F'], st['FiT'], st['lnJ']
        dF = self.kin(d, ops)
        if not ops['total']:
            dF = mm(dF, ops['F0'])
        tr = (FiT*dF).sum((1, 2))                                                  # tr(F^-1 dF)
        A_ = mm(mm(FiT, dF.transpose(1, 2)), FiT)
        dP = self.cell_w*self.V0[:, None, None]*(self.mu*dF+(self.mu-self.lam*lnJ)[:, None, None]*A_+self.lam*tr[:, None, None]*FiT)
        if self.stab:
            dEl = mm(F.transpose(1, 2), dF); dEl = .5*(dEl+dEl.transpose(1, 2))
            dEb, _ = self._stab_pull(self._stab_push(dEl, ops), ops)
            dP = dP+mm(dF, st['Eb'])+mm(F, dEb)
        dy = torch.zeros_like(d)
        if self.jump:
            i, j, b, Kj = ops['i'], ops['j'], ops['b'], ops['Kj'][:, None]
            nc, la, dl, sn, gn, tn = st['nc'], st['la'], st['dl'], st['sn'], st['gn'], st['tn']
            dot = lambda p, q: (p*q).sum(1, keepdim=True)
            da = d[j]-d[i]
            ddl = da-.5*mv(dF[i]+dF[j], b)
            if self.jump_n == 'current':
                dnc = (da-nc*dot(nc, da))/la
                ds = dot(ddl, nc)+dot(dl, dnc)
                dgd = Kj*(self.mu*ddl+(self.lam+self.mu)*(ds*nc+sn*dnc))
                dgn = Kj*(self.lam+self.mu)*(ds*dl+sn*ddl)
                dga = dgd+(dgn-dnc*dot(nc, gn)-nc*dot(dnc, gn)-nc*dot(nc, dgn))/la-tn*dot(nc, da)/la**2
            else:
                dgd = Kj*(self.mu*ddl+(self.lam+self.mu)*dot(ddl, nc)*nc); dga = dgd
            dPF = -.5*dgd[:, :, None]*b[:, None, :]
            dP = dP.index_add(0, i, dPF).index_add_(0, j, dPF)
            dy.index_add_(0, j, dga).index_add_(0, i, -dga)
        if 'sc' in ops:
            ci, cj, n0 = ops['sc'][:3]
            q = st['ksc']*((d[cj]-d[ci])*n0).sum(1)
            dy.index_add_(0, cj, q[:, None]*n0).index_add_(0, ci, -q[:, None]*n0)
        if 'ag' in st:
            dy = dy+torch.autograd.grad(st['ag'][0], st['ag'][1], d, retain_graph=True)[0]
        dParg = dP if ops['total'] else mm(dP, ops['F0'].transpose(1, 2))
        if 'visc' in ops:
            g_ = self.kin(d, ops)
            if ops['total']:
                g_ = g_+ops['visc']['Fn']                                            # (visc_terms subtracts F_n: the product's increment)
                g_ = g_-ops.get('F_off', 0.)
            dParg = dParg+self.visc_terms(g_, ops)[1]
        return self.kin_T(dParg, ops)+dy

    def block_precond(self, x, F, ops, dt, mass):
        """per site the 3 x 3 diagonal block of the step's Hessian at its start (10-29, the CG's preconditioner): mass (per
        site, already / dt^2) times I; the cells' neo-Hookean term (f_mode 'increment', onestep: cell c's increment is
        G_c = sum over its pairs (v_O - v_S) (x) R, a tiny cell's its neighbours' V0-weighted mean, so site a enters it as
        dG_c = dv_a (x) g_ca and dF = dv_a (x) h, h = F0^T g_ca: the block is V0 [mu |h|^2 I + (mu + lam - lam ln J) u u^T],
        u = F^-T h, the coefficient clamped at 0); the viscous term's (Vn / dt) [eta |g|^2 I + (eta / 3 + zeta) g g^T]; the
        jump term's direct part K (mu I + (lam + mu) n n^T) at each wall end. Left out: the stabilisation and the jump
        term's dependence through F (a preconditioner only; block_check in tmp compares with the probed blocks)"""
        assert ops.get('onestep') and not ops.get('onestepT') and not ops['total'], 'block_precond: f_mode increment, onestep only'
        n, f64_, dev = self.n, x.dtype, x.device
        S, O, R = ops['KS'], ops['KO'], ops['R']
        cell, site, g = torch.cat([S, S]), torch.cat([O, S]), torch.cat([R, -R])
        if 'agg' in ops:                                                               # (the tiny cells' entries: their
            as_, ao_, aw, used = ops['agg']                                            # neighbours', re-labelled and weighted)
            o = torch.argsort(cell); cell, site, g = cell[o], site[o], g[o]
            cnt = torch.bincount(cell, minlength=n); off = torch.cumsum(cnt, 0)-cnt
            rep = cnt[ao_]; tot = int(rep.sum())
            start = torch.repeat_interleave(torch.cumsum(rep, 0)-rep, rep)
            idx = torch.repeat_interleave(off[ao_], rep)+torch.arange(tot, device=dev)-start
            keep = ~used[cell]
            cell = torch.cat([cell[keep], torch.repeat_interleave(as_, rep)]); site = torch.cat([site[keep], site[idx]])
            g = torch.cat([g[keep], g[idx]*torch.repeat_interleave(aw, rep)[:, None]])
        key, inv = torch.unique(cell*n+site, return_inverse=True)                     # (a site in a cell once: its g summed)
        g = torch.zeros(len(key), 3, dtype=f64_, device=dev).index_add_(0, inv, g); cell, site = key//n, key % n
        I3 = self.I3
        F0 = ops['F0']; FiT = torch.linalg.inv(F0).transpose(1, 2); lnJ = torch.log(torch.linalg.det(F0))
        h = torch.einsum('emk,em->ek', F0[cell], g); u = torch.einsum('eij,ej->ei', FiT[cell], h)
        co = (self.mu+self.lam-self.lam*lnJ).clamp(min=0.)[cell]
        blk = (self.cell_w*self.V0[cell])[:, None, None]*(self.mu*(h*h).sum(1)[:, None, None]*I3+co[:, None, None]*u[:, :, None]*u[:, None, :])
        if 'visc' in ops:
            eta, zeta = self.beta*self.mu, self.beta*(self.lam+2*self.mu/3)
            Vn = ops['visc']['Vn'][cell]/dt
            blk = blk+Vn[:, None, None]*(eta*(g*g).sum(1)[:, None, None]*I3+(eta/3+zeta)*g[:, :, None]*g[:, None, :])
        B = torch.zeros(n, 3, 3, dtype=f64_, device=dev).index_add_(0, site, blk)+mass[:, None, None]*I3
        if self.jump:
            i, j = ops['i'], ops['j']
            nc = (x[j]-x[i])/(x[j]-x[i]).norm(dim=1, keepdim=True)
            bj = ops['Kj'][:, None, None]*(self.mu*I3+(self.lam+self.mu)*nc[:, :, None]*nc[:, None, :])
            B = B.index_add(0, i, bj).index_add_(0, j, bj)
        if len(self.deg_cells):
            B = torch.cat([B, ((self.lam+2*self.mu)*self.V0[self.deg_cells])[:, None, None]*I3])
        return B

    # ---- the step's solve: trust-region Newton-CG (Steihaug-Toint, as skin_solve.SkinSolver), the region in the norm of
    # a preconditioner per site: diagonal (mass / dt^2 plus the stiffness scale (lam + 2 mu) sum_walls A / l, times
    # 1 + beta / dt with precond 'visc') or the 3 x 3 diagonal blocks of the step's Hessian (precond 'block', 10-29)
    def tr_newton(self, eg, hvp, y, dg, ref, noise, tol=1e-9, maxit=5000):
        """minimises Pi from y (eg(y): Pi, its gradient (None where Pi is infinite) and what hvp(aux, d) needs); stops at
        |dPi/dy| <= tol ref, or at the energy's rounding: noise (a step whose predicted decrease is below it is accepted
        when the gradient shrinks, else the solve ends there). The inner CG stops at min(0.01, res) times the residual
        (quadratic outer convergence). The region starts at the larger of twice the last call's largest accepted step
        and the preconditioned gradient's length. Returns y, Pi(y), stats"""
        stats = dict(products=0, outer=0, rejected=0, negative=0, floor=False, t_hvp=0., t_eg=0., inf=0, edge=0)
        if dg.dim() == 1:
            prec = lambda r: r/dg[:, None]
        else:
            Bi = torch.linalg.inv(dg); prec = lambda r: torch.einsum('nij,nj->ni', Bi, r)
        e0, g, aux = eg(y); yg = y
        if g is None:
            raise RuntimeError('tr_newton: the start has no finite energy')
        res = float(g.norm())/ref
        delta = max(getattr(self, 'delta', 0.), float((g.detach()*prec(g.detach())).sum())**.5)
        big = 0.
        while res > tol and stats['products'] < maxit:
            stats['outer'] += 1
            r_ = -g.detach(); z = prec(r_); d = z.clone(); rz = float((r_*z).sum())
            p = torch.zeros_like(r_); Hp = torch.zeros_like(r_); pMp, pMd, dMd = 0., 0., rz
            edge = False; stop = max(tol, min(.01, res)*res)*ref
            while True:                                                           # Steihaug-Toint CG
                t_ = time.perf_counter()
                Hd = hvp(aux, d); stats['products'] += 1
                dHd = float((d*Hd).sum()); stats['t_hvp'] += time.perf_counter()-t_
                to_edge = lambda: (-pMd+(pMd**2+dMd*(delta**2-pMp))**.5)/dMd
                if dHd <= 0:
                    tau = to_edge(); p, Hp = p+tau*d, Hp+tau*Hd; edge = True; stats['negative'] += 1; break
                al = rz/dHd
                if pMp+2*al*pMd+al**2*dMd >= delta**2:
                    tau = to_edge(); p, Hp = p+tau*d, Hp+tau*Hd; edge = True; break
                p = p+al*d; Hp = Hp+al*Hd; pMp = pMp+2*al*pMd+al**2*dMd
                r_ = r_-al*Hd; z = prec(r_); rz1 = float((r_*z).sum())
                if float(r_.norm()) <= stop or stats['products'] >= maxit:
                    break
                be = rz1/rz; pMd = be*(pMd+al*dMd); dMd = rz1+be**2*dMd; d = z+be*d; rz = rz1
            pred = -float((g.detach()*p).sum()+.5*(p*Hp).sum())
            t_ = time.perf_counter()
            yg1 = yg+p; e1, g1, aux1 = eg(yg1); stats['t_eg'] += time.perf_counter()-t_
            stats['inf'] += g1 is None; stats['edge'] += edge
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

    def step(self, st, dt, grav=None, ref=1., tol=1e-9, maxit=5000, analytic=True, vol_tol=1e-5, ground=None, ground_r=None,
             f_update='forward', radii_start='predict', account=False):
        """one backward-Euler step from st = dict(x, v, F, r[, GS, r_prev]); returns the new state and a report
        (energies at the start on this step's operators, at the end, the solve's stats, the volume fit's). analytic: the
        written-out gradient and products (state / hvp), else autograd on elastic(). The volume fit starts from the
        weights given by radii_start (below) and stops at vol_tol (relative; each step fits
        V0 det F afresh, so its error does not add up; 10-27 8k: 1 Newton iteration, 9e-6, against 3 from w^n to 1e-9).
        ground = (n, c, E*): the plane n.x = c (n unit, the body on its + side), Hertz on the ball (centre x, this step's
        radius r) of every cell with an exposed cap at the step's start: energy 8/15 E* R^1/2 delta^5/2, delta = R - (n.x
        - c) > 0, frictionless (E* = E / (1 - nu^2) against a rigid ground); ground_r: fixed contact radii instead of
        r (10-27 test: the volume fit's radii fed back into the contact on the jack's first-touching cluster).
        f_update: 'forward' F^n+1 = (I + A) F^n, A = Gc(x^n+1 - x^n) on the step start's cut (the solve's own F);
        'midpoint' (10-28 test of the F histories' drift): also B = Gc on the step end's cut (the radii fitted to the
        forward F first), F^n+1 = (I - B/2)^-1 (I + A/2) F^n (exact for exact A = f - I, B = I - f^-1; symmetric in the
        two ends), then the radii fitted again. radii_start: 'predict' the weights from this step's motion and targets to
        first order on the step start's cut (Recut.predict; 10-28 8k bending: start error 8e-4 against 3e-2 from the
        time extrapolation, 1 Newton iteration instead of 2, 2 cuts per step instead of 3), 'extrapolate' them in time
        (2 w^n - w^n-1, where positive). account (10-28): rep['acct'], the step's energy account. Within the step (fixed
        potential W = elastic + ground): Delta(K + W + gravity) = -D_K - B + (Y - X).g, D_K = 1/2 sum m |v^n+1 - v^n|^2
        (backward Euler's kinetic dissipation), B = W(X) - W(Y) - grad W(Y).(X - Y) (Bregman; >= 0 for convex W), g the
        solve's final gradient of Pi (its residual's work); X, Y the step's start and end (sites, then free columns). For
        the jumps between steps: the jump term per wall and the ground energy per ball at the start and the end"""
        x, v, F, r = st['x'], st['v'], st['F'], st['r']
        nr = len(self.deg_cells)                                                       # the degenerate cells' free columns:
        cols = st['cols'] if 'cols' in st else self.deg_q.clone()                       # rows after the sites, no mass (F = I
        ext = lambda a, b_: torch.cat([a, b_]) if nr else a                            # at rest: b = q)
        tm = {}; t0 = clock()
        GS = st['GS'] if 'GS' in st else self.recut.geometry(x, r)
        geo = self.geometry(x, r, GS)
        if self.sc:
            geo['contact'] = self.classify_walls(x, r, geo, F)
        ops = self.operators(x, F, geo, self.pair_refs(x, F, r, geo) if self.f_mode == 'total' else None)
        if self.beta > 0:
            ops['visc'] = dict(dt=dt, Vn=self.V0*torch.linalg.det(F), Fn=F, Fni=torch.linalg.inv(F))
        if self.cut_bonds:
            self._contact_pairs(x, r, ops)
        tm['operators'] = clock()-t0; t0 = clock()
        gv = torch.zeros(3, dtype=f64, device=self.dev) if grav is None else torch.as_tensor(grav, dtype=f64, device=self.dev)
        mdt = ext(self.m[:, None]/(2*dt*dt), torch.zeros(nr, 1, dtype=f64, device=self.dev)); xp = ext(x+dt*v, cols)
        mg = ext(self.m[:, None], torch.zeros(nr, 1, dtype=f64, device=self.dev))
        cons = self.mass == 'consistent'
        def inert(dy):
            """the inertia term's energy and gradient (or product) for dy = y - xp (or a direction)"""
            if cons:
                a_ = self.Mc@dy[:self.n]/(dt*dt)
                return .5*(dy[:self.n]*a_).sum(), ext(a_, torch.zeros(nr, 3, dtype=f64, device=self.dev))
            return (mdt*dy**2).sum(), 2*mdt*dy
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
            e = inert(y-xp)[0]-(mg*gv*y).sum()+self.elastic(y, ops)
            if ground is not None:
                e = e+(8/15*cR*(Rb-(y[cb]@gn-gc)).clamp(min=0.)**2.5).sum()
            return e
        w = geo['A']/(x[geo['j']]-x[geo['i']]).norm(dim=1)
        k = torch.zeros(self.n, dtype=f64, device=self.dev).index_add_(0, geo['i'], w).index_add_(0, geo['j'], w)
        if self.precond == 'block':
            dg = self.block_precond(x, F, ops, dt, (self.mdiag if cons else self.m)/dt**2)
        else:
            sk = 1+self.beta/dt if self.precond == 'visc' else 1.                         # (the viscous term: beta / dt times the
            dg = ext((self.mdiag if cons else self.m)/dt**2+(self.lam+2*self.mu)*sk*k,    # small-strain elastic Hessian)
                     (self.lam+2*self.mu)*self.V0[self.deg_cells])
        y0 = ext(x+dt*v+dt*dt*gv, cols)
        if ground is not None:
            kc0 = contact(y0)[2]
            dg = dg.index_add(0, cb, kc0) if dg.dim() == 1 else dg.index_add(0, cb, kc0[:, None, None]*torch.outer(gn, gn))
        if 'sc' in ops:                                                                # (self-contact: its normal stiffness, as the ground's)
            ci, cj, n0 = ops['sc'][:3]; ks0 = self._sc(y0, ops)[2]
            for s_ in (ci, cj):
                dg = dg.index_add(0, s_, ks0) if dg.dim() == 1 else dg.index_add(0, s_, ks0[:, None, None]*n0[:, :, None]*n0[:, None, :])
        if not math.isfinite(float(Pi(y0))):
            y0 = ext(x, cols)
        e_start, _ = self.elastic(ext(x, cols), ops, terms=True)
        if account:
            X0e = ext(x, cols)
            acct = dict(wall_start=self.wall_jump(X0e, ops), U0={k_: float(v_) for k_, v_ in e_start.items()},
                        K0=self.kinetic(v), P0=-float((mg*gv*X0e).sum()))
            if ground is not None:
                acct['ball_start'] = (cb, 8/15*cR*(Rb-(x[cb]@gn-gc)).clamp(min=0.)**2.5)
        # the energy's rounding: tr C ~ 3 per cell carries the cells' sum's absolute error, eps (mu + lam) sum V0; the
        # inertia's |y - xp|^2 and the other terms are sums of non-negative parts at most comparable
        noise = 64*torch.finfo(f64).eps*float((self.mu+self.lam)*self.V0.sum())
        if analytic:
            def eg(y):
                a_ = self.state(y, ops)
                if a_ is None:
                    return math.inf, None, None
                ec, gc_, a_['kc'] = contact(y)
                ei, gi = inert(y-xp)
                return (float(ei-(mg*gv*y).sum())+sum(a_['e'].values())+ec,
                        gi-mg*gv+a_['grad']+gc_, a_)
            def hvp_(a_, d):
                h_ = inert(d)[1]+self.hvp(a_, d, ops)
                if ground is not None:
                    h_ = h_.index_add(0, cb, (a_['kc']*(d[cb]@gn))[:, None]*gn)
                return h_
            hvp = GraphedProduct(hvp_) if self.dev == 'cuda' and self.tet_w <= 0 and self.wall_w <= 0 and self.cellhg_w <= 0 else hvp_   # (autograd in the product)
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
        if self.f_mode != 'total' and self.f_substeps > 1:
            F1 = self.update_F(x, y[:self.n], F, r, GS)
        tet_min = self.tet_energy(y, ops)[1] if self.tet_w > 0 else None
        wall_min = self.wall_energy(y, ops)[1] if self.wall_w > 0 else None
        if account:
            Y = y; gf = eg(Y)[1]
            if gf is None or not analytic:
                raise RuntimeError('account: the end has no finite energy (or the autograd path)')
            gW = gf-inert(Y-xp)[1]+mg*gv
            if 'visc' in ops:
                sY = self.state(Y, ops)
                gW = gW-sY['visc_grad']; acct['DV'] = 2*sY['e']['visc']                    # (quadratic in Y - X: grad . (Y - X) = 2 U)
            acct.update(wall_end=self.wall_jump(Y, ops), U1={k_: float(v_) for k_, v_ in e_end.items()},
                        K1=self.kinetic((Y[:self.n]-x)/dt), P1=-float((mg*gv*Y).sum()),
                        DK=self.kinetic((Y[:self.n]-x)/dt-v), res=float((gf*(Y-X0e)).sum()))
            G0 = float(acct['ball_start'][1].sum()) if ground is not None else 0.
            G1 = 0.
            if ground is not None:
                acct['ball_end'] = (cb, 8/15*cR*(Rb-(Y[cb]@gn-gc)).clamp(min=0.)**2.5); G1 = float(acct['ball_end'][1].sum())
            acct['B'] = sum(acct['U0'].values())+G0-sum(acct['U1'].values())-G1-float((gW*(X0e-Y)).sum())
            acct['H0'] = acct['K0']+sum(acct['U0'].values())+G0+acct['P0']; acct['H1'] = acct['K1']+sum(acct['U1'].values())+G1+acct['P1']
        y, cols = y[:self.n], y[self.n:]
        tm['solve'] = clock()-t0; t0 = clock()
        Vt = self.V0*torch.linalg.det(F1)
        if radii_start == 'predict':
            r_ = self.recut.predict(GS[0], y-x, Vt)
        else:
            w_ = r*r
            if 'r_prev' in st:
                w2 = 2*w_-st['r_prev']**2
                w_ = torch.where(w2 > 0, w2, w_)
            r_ = torch.sqrt(w_)
        r1, G1, S1, it, nrep = self.recut.solve_radii(y, r_, Vt, tol=vol_tol, r_base=r)
        if f_update == 'midpoint' and self.f_mode != 'total':
            ops1 = self.operators(y, F1, self.geometry(y, r1, (G1, S1)))
            A_, B_ = self.kin(y-x, ops), self.kin(y-x, ops1)
            F1 = torch.linalg.solve(self.I3-B_/2, (self.I3+A_/2)@F)
            Vt = self.V0*torch.linalg.det(F1)
            r1, G1, S1, it2, nrep2 = self.recut.solve_radii(y, r1, Vt, tol=vol_tol); it += it2; nrep += nrep2
        tm['radii'] = clock()-t0
        verr = float(((G1.V.detach()-Vt)/Vt).abs().max())
        if verr > vol_tol:                                                             # (10-28: a failed fit went on silently)
            raise RuntimeError('the volume fit stopped at %.1e of the target (tol %.0e; iterations %d, repairs %d, recut stats %s)' % (
                verr, vol_tol, it, nrep, {k_: v_ for k_, v_ in self.recut.stats.items() if k_ in ('floor', 'overflow')}))
        rep = dict(e_start={k_: float(v_) for k_, v_ in e_start.items()}, e_end={k_: float(v_) for k_, v_ in e_end.items()},
                   solve=stats, radii_it=it, repairs=nrep, vol_err=verr, walls=(geo['i'], geo['j'], geo['A']),
                   minJ=float((Vt/self.V0).min()), time=tm, radii_start=self.recut.stats.get('start_ratio'))
        if ground is not None:
            dl = (Rb-(y[cb]@gn-gc)).clamp(min=0.)
            rep['ground'] = dict(force=float((4/3*cR*dl**1.5).sum()), touching=int((dl > 0).sum()), overlap=float((dl/Rb).max()),
                                 energy=float((8/15*cR*dl**2.5).sum()), balls=len(cb))
        if self.sc or self.cut_bonds:
            rep['sc'] = dict(self.sc_stats, overlap=0., energy=0.)
            if 'sc' in ops:
                ci, cj, n0, rs, c_ = ops['sc']
                dl_ = (rs-((y[cj]-y[ci])*n0).sum(1)).clamp(min=0.)
                rep['sc'].update(overlap=float((dl_/torch.minimum(r[ci], r[cj])).max()), energy=float((8/15*c_*dl_**2.5).sum()))
        if tet_min is not None:
            rep['tet_minJ'] = tet_min
        if wall_min is not None:
            rep['wall_minJ'] = wall_min
        if nr:
            rep['cols'] = cols.norm(dim=1)                                             # the free columns' lengths (stretch along q)
        if account:
            rep['acct'] = acct
        return dict(x=y, v=(y-x)/dt, F=F1.detach(), r=r1, GS=(G1, S1), r_prev=r, cols=cols), rep
