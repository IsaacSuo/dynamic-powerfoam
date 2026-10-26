"""Solving the connected skin for given bodies (2026-10-23): the skin is massless, so every time step needs y with
dE/dy (x, y) = 0 (sphere_model.SkinTerms). The reference for a later Warp version.
  solver   trust-region Newton-CG (Steihaug-Toint, preconditioned; the region in the preconditioner's norm): a direction of
           negative curvature or a step past the region ends the CG at the region's edge; the region grows or shrinks by
           the ratio of the actual to the predicted energy decrease, so a start far from the solution (a cold start, where
           the energy need not be convex) still decreases the energy every accepted step
  blocks   preconditioner of small dense blocks over groups of skin points (each region's, or each body's), added over
           the groups and turned by the group's body rotation:
           'probed'  blocks of K_yy at rest (skin_step_cost.py, skin_Kyy_rest.npz)
           'skin'    the skin's own squares per region, (mu / r) m_ab I + (r / 3) [mu / 2 (U_ab + T_ab^T) + kappa T_ab]
           'model'   per body, a quadratic model of everything a body's skin points move: F_B is linear in the skin,
                     dF = sum_q u_q (x) V_q (V_q = Mt^-T sum of the point's levers), so the deviation delta_e = u_e - dF d_e
                     is linear too; terms: the outer layer as its perfect square (mu / r) int |r sym(dF) n + sum phi delta|^2
                     (positive; the energy's own o1 + o2 + o3 are not a perfect square), the lateral squares, the volume
                     lambda / (2 V) (V tr dF + sum_e delta_e . c_e / 3)^2, the bonds' rotational stiffness 1/2 th^T K_rot th,
                     th = axial(skew dF) = 1/2 sum_q V_q x u_q (K_rot probed on the bond energy, bodies coloured by bonds);
                     left out: the curvature reference Gamma and the volume variance (couplings between bodies)
           'skin' and 'model' are a group's own terms (element matrices): assembled into the global matrix first
           (assemble: a skin point shared by several groups gets all their parts), then the groups' blocks taken from it;
           all built from the structure alone, for any structure; 'model' leaves out Gamma and the variance and
           underestimates big-lever arc midpoints (1.5 % of the points at less than half their stiffness, 10-23)
  rest     rest_model: the skin Hessian at rest written out term by term as the energy has it (lever, delta_aff, Gamma,
           o1 / o2 with D = sym dF, the squares, the volume with the walls' dW / dF and the variance, the bonds through
           axial(skew dF)); matches the energy's Hessian-vector products to round-off (skin_rest_model_check.py). The bond
           rotation Hessian and dW / dF are probed on those two functions alone (rest_pieces); a Warp version writes them
           per bond / wall
  start    the cells' affine skin: each skin point at the mean of its cells' affine fields"""
import numpy as np
import torch

from experiments.foam_surface_completion import polar_newton
from free_surface.sphere_model import matrix_quat

f64 = torch.float64


class SkinSolver:
    def __init__(self, st, bsk, body_of_cell):
        self.st = st; dev = st.X.device; self.dev = dev
        self.n, self.npt = st.n, st.npt
        self.th = torch.zeros(self.n, 9, dtype=f64, device=dev)
        boc = np.asarray(body_of_cell)
        off = np.asarray(bsk['part_off'])
        self.off = torch.as_tensor(off, device=dev)
        self.regions = self._groups([bsk['ent_pid'][off[k]:off[k+1]] for k in range(len(off)-1)], boc[bsk['part_site']])
        ent_body = boc[bsk['ent_site']]
        self.e_pid, self.e_body = torch.as_tensor(bsk['ent_pid'], device=dev), torch.as_tensor(ent_body, device=dev)
        self.cnt = torch.zeros(self.npt, dtype=f64, device=dev).index_add(0, self.e_pid, torch.ones(len(self.e_pid), dtype=f64, device=dev))
        bodies = np.unique(ent_body)
        self.bodies = self._groups([np.unique(bsk['ent_pid'][ent_body == b]) for b in bodies], bodies)
        gidx = -np.ones(self.n, int); gidx[bodies] = np.arange(len(bodies)); self.gidx = torch.as_tensor(gidx, device=dev)
        # each entry's slot in its body's point list
        le = np.zeros(len(ent_body), int)
        for k, b in enumerate(bodies):
            sel = np.flatnonzero(ent_body == b)
            le[sel] = np.searchsorted(self.bodies['lists'][k], bsk['ent_pid'][sel])
        self.e_slot = torch.as_tensor(le, device=dev)
        # bodies coloured by bonds (for the rotational stiffness)
        nb = [set() for _ in range(self.n)]
        for a_, b_ in zip(st.i.tolist(), st.j.tolist()):
            nb[a_].add(b_); nb[b_].add(a_)
        col = -np.ones(self.n, int)
        for b_ in range(self.n):
            used = {col[c_] for c_ in nb[b_]}
            k = 0
            while k in used:
                k += 1
            col[b_] = k
        self.bond_colour = torch.as_tensor(col, device=dev)

    def _groups(self, lists, owner):
        S = max(len(l) for l in lists)
        L = -np.ones((len(lists), S), int)
        for k, l in enumerate(lists):
            L[k, :len(l)] = l
        Lt = torch.as_tensor(L, device=self.dev)
        return dict(lists=lists, S=S, G=len(lists), Lc=Lt.clamp(min=0), Lm=(Lt >= 0).to(f64), owner=torch.as_tensor(np.asarray(owner), device=self.dev))

    def energy(self, x, y):
        return self.st.energy_xy(self.th, x, y)

    def grad(self, x, y):
        return torch.func.grad(lambda yy: self.energy(x, yy.view(-1, 3)))(y.reshape(-1))

    def hvp(self, x, y, g=None):
        """exact Hessian-vector products in y (g, the gradient at y, is for subclasses that difference it)"""
        f = lambda yy: torch.func.grad(lambda z: self.energy(x, z.view(-1, 3)))(yy)
        y0 = y.reshape(-1)
        return torch.func.vmap(lambda v: torch.func.jvp(f, (y0,), (v,))[1])

    def eg(self, x, y):
        """energy, dE/dy (flat) and whatever else the same evaluation gives (subclasses; None here)"""
        return self.energy(x, y), self.grad(x, y), None

    def rotations(self, x, y):
        return polar_newton(self.st.F(x, y))

    def prec_rotations(self, x, y, aux):
        """the rotations that turn the preconditioner at (x, y); aux from eg there"""
        return self.rotations(x, y)

    def prec_at(self, make_prec, x, y, aux):
        """the preconditioner for the step from (x, y)"""
        return make_prec(self.prec_rotations(x, y, aux))

    def affine_skin(self, x, y):
        """each skin point at the mean of its cells' affine fields (F_B from the bonds and the skin y)"""
        st = self.st
        F = st.F(x, y)
        pos = x[self.e_body]+torch.einsum('eij,ej->ei', F[self.e_body], st.Y[self.e_pid]-st.X[self.e_body])
        return torch.zeros_like(st.Y).index_add(0, self.e_pid, pos)/self.cnt[:, None]

    def _pad(self, B, g):
        pad = (g['Lm'] == 0).repeat_interleave(3, 1)
        return B+torch.diag_embed(pad.to(f64))

    def probe_K(self, x=None, y=None, batch=8):
        """K_yy at (x, y) (default rest) as scipy CSR by coloured probing of the energy's Hessian-vector products: two skin
        points couple when their cells are within two bonds (Gamma); a colour holds no two points coupled to a common
        point (skin_step_cost.py); couplings beyond two bonds (the variance term) are left out (~1e-6)"""
        import scipy.sparse as sp
        st, n, npt = self.st, self.n, self.npt
        x = st.X if x is None else x; y = st.Y if y is None else y
        i_, j_ = st.i.cpu().numpy(), st.j.cpu().numpy()
        A1 = sp.coo_matrix((np.ones(2*len(i_)), (np.r_[i_, j_], np.r_[j_, i_])), shape=(n, n)).tocsr()+sp.eye(n, format='csr'); A1.data[:] = 1
        A2 = (A1@A1).astype(bool).astype(float)
        ep, eb = self.e_pid.cpu().numpy(), self.e_body.cpu().numpy()
        Pc = sp.coo_matrix((np.ones(len(ep)), (ep, eb)), shape=(npt, n)).tocsr(); Pc.data[:] = 1
        C2 = (Pc@A2@Pc.T).tocsr(); C2.data[:] = 1
        C4 = (C2@C2).tocsr()
        colour = -np.ones(npt, int)
        for q in range(npt):
            used = np.unique(colour[C4.indices[C4.indptr[q]:C4.indptr[q+1]]])
            k = 0
            for u in used[used >= 0]:
                if u != k:
                    break
                k += 1
            colour[q] = k
        C2 = C2.tocoo(); P_, Q_ = C2.row, C2.col
        col = torch.as_tensor(colour, device=self.dev); Qt = torch.as_tensor(Q_, device=self.dev); Pt = torch.as_tensor(P_, device=self.dev)
        H = self.hvp(x, y)
        vals = torch.zeros(len(P_), 3, 3, dtype=f64, device=self.dev)
        probes = [(k, d) for k in range(int(colour.max())+1) for d in range(3)]
        for s0 in range(0, len(probes), batch):
            ch = probes[s0:s0+batch]
            V = torch.zeros(len(ch), npt, 3, dtype=f64, device=self.dev)
            for b, (k, d) in enumerate(ch):
                V[b, col == k, d] = 1.
            Kv = H(V.reshape(len(ch), -1)).view(len(ch), npt, 3)
            for b, (k, d) in enumerate(ch):
                sel = torch.nonzero(col[Qt] == k)[:, 0]
                vals[sel, :, d] = Kv[b, Pt[sel]]
        vals = vals.cpu().numpy()
        rows = (3*P_[:, None, None]+np.arange(3)[None, :, None]).repeat(3, 2); cols = (3*Q_[:, None, None]+np.arange(3)[None, None, :]).repeat(3, 1)
        K = sp.coo_matrix((vals.ravel(), (rows.ravel(), cols.ravel())), shape=(3*npt, 3*npt)).tocsr()
        return (.5*(K+K.T)).tocsr()

    def assemble(self, B, g):
        """the global matrix (scipy CSR, 3 npt square) of element blocks B over groups g (each group's own terms; a skin
        point shared by several groups gets all their parts, as K_yy does)"""
        import scipy.sparse as sp
        S3 = 3*g['S']
        idx = (3*g['Lc'][:, :, None]+torch.arange(3, device=self.dev)).reshape(g['G'], S3)
        real = g['Lm'].repeat_interleave(3, 1)
        m = (real[:, :, None]*real[:, None, :]) > 0
        rows = idx[:, :, None].expand(-1, -1, S3)[m].cpu().numpy(); cols = idx[:, None, :].expand(-1, S3, -1)[m].cpu().numpy()
        n3 = 3*self.npt
        return sp.coo_matrix((B[m].cpu().numpy(), (rows, cols)), shape=(n3, n3)).tocsr()

    def blocks_probed(self, K, g):
        """the groups' blocks of a global matrix K (scipy CSR, 3 npt square): K_yy at rest, or an assembled model"""
        S3 = 3*g['S']
        B = np.zeros((g['G'], S3, S3))
        for k, l in enumerate(g['lists']):
            i3 = (3*np.asarray(l)[:, None]+np.arange(3)).ravel()
            B[k, :len(i3), :len(i3)] = K[i3][:, i3].toarray(); B[k, len(i3):, len(i3):] = np.eye(S3-len(i3))
        return torch.as_tensor(B, dtype=f64, device=self.dev)

    def _square_coeff(self):
        """per pair (a, b) of a part's entries: the outer_skin + outer_lateral coefficient C_ab (energy delta_a^T C_ab delta_b)"""
        st = self.st
        lam = float(st.model.k_v); mu = st.mu
        kap = lam*mu/(lam+2*mu)
        r_p = st.p_r[st.pr_part]
        C = (mu/r_p*st.pr_m)[:, None, None]*torch.eye(3, dtype=f64, device=self.dev)
        if st.lateral:
            C = C+(r_p/3)[:, None, None]*(.5*mu*(st.pr_U+st.pr_T.transpose(1, 2))+kap*st.pr_T)
        return C

    def blocks_skin(self):
        """the skin's own squares (outer_skin, outer_lateral) per region, body frame"""
        st, g = self.st, self.regions
        part = st.pr_part
        C = self._square_coeff()
        la, lb = st.pr_a-self.off[part], st.pr_b-self.off[part]
        Hb = torch.zeros(g['G'], g['S'], g['S'], 3, 3, dtype=f64, device=self.dev)
        Hb.index_put_((part, la, lb), C, accumulate=True)
        Hb.index_put_((part, lb, la), C.transpose(1, 2), accumulate=True)
        return self._pad(Hb.permute(0, 1, 3, 2, 4).reshape(g['G'], 3*g['S'], 3*g['S']), g)

    def rot_stiffness(self, x, R):
        """the bond energy's 3 x 3 stiffness to each body's rotation (turned by a small rotation phi: R <- exp([phi]) R;
        second order of exp is exact for the Hessian at phi = 0)"""
        m, th = self.st.model, self.th
        def skew(p):
            z = torch.zeros_like(p[:, 0])
            return torch.stack((torch.stack((z, -p[:, 2], p[:, 1]), 1), torch.stack((p[:, 2], z, -p[:, 0]), 1), torch.stack((-p[:, 1], p[:, 0], z), 1)), 1)
        I3 = torch.eye(3, dtype=f64, device=self.dev)
        def Eb(phi):
            W = skew(phi.view(-1, 3))
            q = matrix_quat((I3+W+.5*W@W)@R)
            e = m.energy(th, x, q)
            if getattr(m, 'body_law', None) == 'nodalvol':
                e = e-m._nodalvol(th, x)[0]
            return e
        gE = torch.func.grad(Eb)
        z0 = torch.zeros(3*self.n, dtype=f64, device=self.dev)
        Kr = torch.zeros(self.n, 3, 3, dtype=f64, device=self.dev)
        for c in range(int(self.bond_colour.max())+1):
            sel = self.bond_colour == c
            for k in range(3):
                v = torch.zeros(self.n, 3, dtype=f64, device=self.dev); v[sel, k] = 1.
                Hv = torch.func.jvp(gE, (z0,), (v.reshape(-1),))[1].view(-1, 3)
                Kr[sel, :, k] = Hv[sel]
        return .5*(Kr+Kr.transpose(1, 2))

    def blocks_model(self, x=None, R=None, chunk=12, terms=('o1', 'o2', 'squares', 'volume', 'rotation')):
        """per body, the Hessian of the quadratic model above in its skin points' displacements (rest frame); terms: which
        of the model's terms to include"""
        st, g, dev = self.st, self.bodies, self.dev
        x = st.X if x is None else x
        R = torch.eye(3, dtype=f64, device=dev).expand(self.n, 3, 3) if R is None else R
        mu, lam = st.mu, float(st.model.k_v)
        Kr = self.rot_stiffness(x, R)[g['owner']]
        gb = self.gidx[self.e_body]; le = self.e_slot
        v = st.e_w[:, None]*torch.einsum('eji,ej->ei', st.Mt_inv[self.e_body], st.e_l)
        Vq = torch.zeros(g['G'], g['S'], 3, dtype=f64, device=dev).index_put_((gb, le), v, accumulate=True)
        gc = self.gidx[st.c_body]
        VB = st.V[g['owner']]
        pa, pb = st.pr_a, st.pr_b
        C = self._square_coeff()
        def Q(U):
            dF = torch.einsum('bqi,bqj->bij', U, Vq)
            Sy = .5*(dF+dF.transpose(1, 2))
            de = U[gb, le]-torch.einsum('eij,ej->ei', dF[gb], st.e_d)
            o1 = mu*(st.c_r*torch.einsum('cij,cjk,cki->c', Sy[gc], st.c_N, Sy[gc])).sum()
            o2 = 2*mu*(de*torch.einsum('eij,ej->ei', Sy[gb], st.e_c)).sum()
            sq = torch.einsum('ni,nij,nj->n', de[pa], C, de[pb]).sum()
            dV = VB*torch.diagonal(dF, dim1=1, dim2=2).sum(1)+torch.zeros(g['G'], dtype=f64, device=dev).index_add(0, gb, (de*st.e_c).sum(1)/3)
            vol = (lam/(2*VB)*dV**2).sum()
            tht = .5*torch.cross(Vq, U, dim=2).sum(1)
            rot = .5*torch.einsum('bi,bij,bj->b', tht, Kr, tht).sum()
            return sum(dict(o1=o1, o2=o2, squares=sq, volume=vol, rotation=rot)[t] for t in terms)
        gQ = torch.func.grad(Q)
        S3 = 3*g['S']
        H = torch.zeros(g['G'], S3, S3, dtype=f64, device=dev)
        for k0 in range(0, S3, chunk):
            ks = torch.arange(k0, min(k0+chunk, S3), device=dev)
            E = torch.zeros(len(ks), g['G'], g['S'], 3, dtype=f64, device=dev)
            E[torch.arange(len(ks), device=dev), :, ks//3, ks % 3] = 1.
            cols = torch.func.vmap(gQ)(E)                                            # (k, G, S, 3)
            H[:, :, k0:k0+len(ks)] = cols.reshape(len(ks), g['G'], S3).permute(1, 2, 0)
        H = .5*(H+H.transpose(1, 2))
        real = g['Lm'].repeat_interleave(3, 1)
        return self._pad(H*real[:, :, None]*real[:, None, :], g)

    def _colour2(self):
        """bodies coloured so that no two of a colour are within two bonds (each body's closed neighbourhood holds at
        most one of every colour)"""
        if getattr(self, 'col2', None) is None:
            nb = [set([b_]) for b_ in range(self.n)]
            for a_, b_ in zip(self.st.i.tolist(), self.st.j.tolist()):
                nb[a_].add(b_); nb[b_].add(a_)
            col = -np.ones(self.n, int)
            for b_ in range(self.n):
                used = {col[c_] for k_ in nb[b_] for c_ in nb[k_]}
                k = 0
                while k in used:
                    k += 1
                col[b_] = k
            self.col2 = torch.as_tensor(col, device=self.dev)
        return self.col2

    def rest_pieces(self):
        """at rest: the bond energy's Hessian in the bodies' rotations (diagonal blocks, and the (i, j) block of every
        bond), and the walls' volume derivative dW_k / dF_b for b = k and its bond neighbours (k's own row of the bond:
        (k, k), and (i, j), (j, i) per bond); probed on these two functions with the distance-two colouring"""
        st, m, th, dev, n = self.st, self.st.model, self.th, self.dev, self.n
        col = self._colour2(); ncol = int(col.max())+1
        I3 = torch.eye(3, dtype=f64, device=dev)
        def skew(p):
            z = torch.zeros_like(p[:, 0])
            return torch.stack((torch.stack((z, -p[:, 2], p[:, 1]), 1), torch.stack((p[:, 2], z, -p[:, 0]), 1), torch.stack((-p[:, 1], p[:, 0], z), 1)), 1)
        def Eb(phi):
            W = skew(phi.view(-1, 3))
            e = m.energy(th, st.X, matrix_quat(I3+W+.5*W@W))
            if getattr(m, 'body_law', None) == 'nodalvol':
                e = e-m._nodalvol(th, st.X)[0]
            return e
        gE = torch.func.grad(Eb)
        z0 = torch.zeros(3*n, dtype=f64, device=dev)
        Kd = torch.zeros(n, 3, 3, dtype=f64, device=dev); Kij = torch.zeros(len(st.i), 3, 3, dtype=f64, device=dev)
        wall = lambda F: st.wall_volume(st.X, F, polar_newton(F))
        Fid = I3.expand(n, 3, 3).contiguous()
        Wd = torch.zeros(n, 3, 3, dtype=f64, device=dev); Wij = torch.zeros(len(st.i), 3, 3, dtype=f64, device=dev); Wji = torch.zeros_like(Wij)
        for c in range(ncol):
            sel = col == c
            for k in range(3):
                v = torch.zeros(n, 3, dtype=f64, device=dev); v[sel, k] = 1.
                Hv = torch.func.jvp(gE, (z0,), (v.reshape(-1),))[1].view(n, 3)
                Kd[sel, :, k] = Hv[sel]
                bj = sel[st.j]
                Kij[bj, :, k] = Hv[st.i[bj]]
            for a in range(3):
                for b in range(3):
                    dF = torch.zeros(n, 3, 3, dtype=f64, device=dev); dF[sel, a, b] = 1.
                    dW = torch.func.jvp(wall, (Fid,), (dF,))[1]
                    Wd[sel, a, b] = dW[sel]
                    bj, bi = sel[st.j], sel[st.i]
                    Wij[bj, a, b] = dW[st.i[bj]]                                     # d W_i / d F_j
                    Wji[bi, a, b] = dW[st.j[bi]]                                     # d W_j / d F_i
        return dict(Kd=.5*(Kd+Kd.transpose(1, 2)), Kij=Kij, Wd=Wd, Wij=Wij, Wji=Wji, colours=ncol)

    def rest_model(self, pieces=None):
        """Q(U), U (npt, 3) skin displacements: the energy's second-order part at rest (F = I, R = I, delta = 0), every
        term as SkinTerms writes it: dF_b = sum u (x) v (the lever), delta_aff = u - dF d, the curvature reference Gamma
        from the bonds' dF differences, o1 / o2 with D = sym dF, the squares with delta, the volume through dVd = dW
        (walls) + tr dF (V_res + cones) + sum delta_aff . c / 3 and the variance term, the bonds through the rotations
        theta = axial(skew dF). Its Hessian is K_yy at rest; every first derivative at rest is zero, so no second-order
        parts of the maps enter."""
        st, dev, n = self.st, self.dev, self.n
        P = self.rest_pieces() if pieces is None else pieces
        lam, mu = float(st.model.k_v), st.mu
        kap = lam*mu/(lam+2*mu)
        v = st.e_w[:, None]*torch.einsum('eji,ej->ei', st.Mt_inv[self.e_body], st.e_l)
        eb, ep = self.e_body, self.e_pid
        Vdet = st.Vres+torch.zeros(n, dtype=f64, device=dev).index_add(0, st.c_body, st.c_r*st.c_A/3)
        r_p = st.p_r[st.pr_part]
        def Q(U):
            dF = torch.zeros(n, 3, 3, dtype=f64, device=dev).index_add(0, eb, U[ep][:, :, None]*v[:, None, :])
            Sy = .5*(dF+dF.transpose(1, 2))
            daff = U[ep]-torch.einsum('eij,ej->ei', dF[eb], st.e_d)
            rb = st.wb[:, None, None]*torch.einsum('bkm,bik->bim', st.g_A, dF[st.j]-dF[st.i])
            rhs = torch.zeros(n, 3, 6, dtype=f64, device=dev).index_add(0, st.i, rb).index_add(0, st.j, rb)
            dl = daff-.5*torch.einsum('eim,em->ei', torch.einsum('bmn,bin->bim', st.g_Ninv, rhs)[eb], st.g_q)
            o1 = mu*(st.c_r*torch.einsum('cki,ckj,cij->c', Sy[st.c_body], Sy[st.c_body], st.c_N)).sum()
            o2 = 2*mu*(daff*torch.einsum('eij,ej->ei', Sy[eb], st.e_c)).sum()
            da, db = dl[st.pr_a], dl[st.pr_b]
            o3 = (mu/r_p*st.pr_m*(da*db).sum(1)).sum()
            lat = torch.zeros((), dtype=f64, device=dev)
            if st.lateral:
                tU = torch.einsum('ni,nij,nj->n', da, st.pr_U, db); tT = torch.einsum('ni,nij,nj->n', da, st.pr_T, db)
                tTs = torch.einsum('ni,nij,nj->n', db, st.pr_T, da)
                lat = (r_p/3*(.5*mu*(tU+tTs)+kap*tT)).sum()
            dW = (P['Wd']*dF).sum((1, 2))
            dW = dW.index_add(0, st.i, (P['Wij']*dF[st.j]).sum((1, 2))).index_add(0, st.j, (P['Wji']*dF[st.i]).sum((1, 2)))
            dVd = dW+torch.diagonal(dF, dim1=1, dim2=2).sum(1)*Vdet+torch.zeros(n, dtype=f64, device=dev).index_add(0, eb, (daff*st.e_c).sum(1)/3)
            dlj = dVd/st.V
            vol = (st.V*.5*lam*dlj**2).sum()
            if st.variance:
                vol = vol+(.5*lam*st.A*st.L/12*(dlj[st.i]-dlj[st.j])**2).sum()
            tht = .5*torch.stack((dF[:, 2, 1]-dF[:, 1, 2], dF[:, 0, 2]-dF[:, 2, 0], dF[:, 1, 0]-dF[:, 0, 1]), 1)
            bonds = .5*torch.einsum('bi,bij,bj->', tht, P['Kd'], tht)+torch.einsum('ki,kij,kj->', tht[st.i], P['Kij'], tht[st.j])
            return dict(o1=o1, o2=o2, o3=o3, lateral=lat, volume=vol, bonds=bonds)
        return Q

    def preconditioner(self, B, g):
        """additive group blocks, turned by each group's body rotation: returns make(Rb) -> (rv (k, 3 npt) -> same)"""
        Binv = torch.linalg.inv(.5*(B+B.transpose(1, 2)))
        G, S, npt, dev = g['G'], g['S'], self.npt, self.dev
        Lc, Lm, owner = g['Lc'], g['Lm'], g['owner']
        def make(Rb):
            Rr = Rb[owner]
            def f(rv):
                k = rv.shape[0]
                rl = rv.view(k, npt, 3)[:, Lc]*Lm[None, :, :, None]
                rl = torch.einsum('pji,bpsj->bpsi', Rr, rl).reshape(k, G, -1)
                zl = torch.einsum('pij,bpj->bpi', Binv, rl).view(k, G, S, 3)
                zl = torch.einsum('pij,bpsj->bpsi', Rr, zl)*Lm[None, :, :, None]
                return torch.zeros(k, npt, 3, dtype=f64, device=dev).index_add(1, Lc.reshape(-1), zl.reshape(k, -1, 3)).reshape(k, -1)
            return f
        return make

    def steihaug(self, x, y, g, prec, delta, stop, cap):
        """the trust-region subproblem by preconditioned CG from p = 0 (Steihaug-Toint): stops at |r| <= stop, after cap
        products, at negative curvature or at the region's edge |p|_M = delta (None: the preconditioned gradient's
        length). Returns p, Hp (1, 3 npt), |p|_M^2, delta, edge, negative exit (0 / 1), products"""
        H = self.hvp(x, y, g)
        r = -g[None]; z = prec(r); d = z.clone(); rz = float((r*z).sum())
        if delta is None:
            delta = rz**.5
        p = torch.zeros_like(r); Hp = torch.zeros_like(r)
        pMp, pMd, dMd = 0., 0., rz
        k = 0
        while True:
            Hd = H(d); k += 1
            dHd = float((d*Hd).sum())
            def to_edge():
                return (-pMd+(pMd**2+dMd*(delta**2-pMp))**.5)/dMd
            if dHd <= 0:
                tau = to_edge()
                return p+tau*d, Hp+tau*Hd, pMp, delta, True, 1, k
            al = rz/dHd
            if pMp+2*al*pMd+al**2*dMd >= delta**2:
                tau = to_edge()
                return p+tau*d, Hp+tau*Hd, pMp, delta, True, 0, k
            p = p+al*d; Hp = Hp+al*Hd; pMp = pMp+2*al*pMd+al**2*dMd
            r = r-al*Hd
            z = prec(r)
            rr, rz1 = torch.stack(((r*r).sum(), (r*z).sum())).tolist()                    # one sync for both
            if rr**.5 <= stop or k >= cap:
                return p, Hp, pMp, delta, False, 0, k
            be = rz1/rz
            pMd = be*(pMd+al*dMd); dMd = rz1+be**2*dMd; d = z+be*d; rz = rz1

    def solve(self, x, y, make_prec, ref, tol=1e-6, maxit=600, marks=(1e-2, 1e-4, 1e-6), log=None, tol_fn=None):
        """trust-region Newton-CG from y; ref the force scale (|dE/dy| is reported relative to it). Returns y and stats:
        products, products when the residual first fell below each mark, negative-curvature exits, rejected steps.
        tol_fn(aux): a looser tolerance from the current state (the residual stops at max(tol, tol_fn(aux)))"""
        stats = dict(products=0, hits={m: None for m in marks}, negative=0, rejected=0, outer=0)
        e0, g, aux = self.eg(x, y); e0 = float(e0); res = float(g.norm())/ref
        tl = max(tol, tol_fn(aux)) if tol_fn is not None else tol
        for m in marks:
            if res <= m:
                stats['hits'][m] = 0
        delta = None
        while res > tl and stats['products'] < maxit:
            stats['outer'] += 1
            prec = self.prec_at(make_prec, x, y, aux)
            p, Hp, pMp, delta, edge, neg, k = self.steihaug(x, y, g, prec, delta, max(tl, .01*res)*ref, maxit-stats['products'])
            stats['products'] += k; stats['negative'] += neg
            pred = -float((g[None]*p).sum()+.5*(p*Hp).sum())
            ynew = y+p.view(-1, 3)
            e1, g1, aux1 = self.eg(x, ynew); e1 = float(e1)
            ared = e0-e1
            noise = 1e-14*abs(e0)
            rho = ared/pred if pred > noise else 1.
            pM = pMp**.5 if not edge else delta
            if rho < .25:
                delta = .25*pM
            elif rho > .75 and edge:
                delta = 2*delta
            if rho > 1e-4 or (pred <= noise and ared >= -noise):
                y, e0, g, aux = ynew, e1, g1, aux1
                res = float(g.norm())/ref
                if tol_fn is not None:
                    tl = max(tol, tol_fn(aux))
                for m in marks:
                    if stats['hits'][m] is None and res <= m:
                        stats['hits'][m] = stats['products']
            else:
                stats['rejected'] += 1
            if log is not None:
                log('outer %d: products %d, residual %.1e, ratio %.2f (actual %.2e, predicted %.2e, energy %.3e), region %.2e%s' % (
                    stats['outer'], stats['products'], res, rho, ared, pred, e0, delta, ', edge' if edge else ''))
        stats['residual'] = res; stats['energy'] = e0; stats['aux'] = aux                    # at the returned y
        return y, stats
