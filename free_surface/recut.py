"""Re-cutting the foam at moved sites with the radii solved for given cell volumes, all on the GPU (2026-10-24; toward
rebuilding the cells from the bodies' motion every step). The structure is the alpha complex (alpha_complex.py: Power
Foam's BVH, then one clipping thread per cell), the values gpu_geometry.Geometry (exact, the union of balls cut by the
power diagram). The radii: damped Newton on the weights w = r^2 (semi-discrete optimal transport's) for V(w) = target,
with the exact symmetric Jacobian dV_i/dw_j = -A_ij / (2 L_ij) on a wall, dV_i/dw_i = sum_j A_ij / (2 L_ij) +
A_patch,i / (2 r_i) (a volume moves only by its boundary's normal velocity; checked against differences in
retess_check.py), solved by CG preconditioned by its diagonal; the full step halved until every cell keeps a positive
volume and the residual falls; it stops when the volumes hold to tol or the step reaches the rounding. A hidden site
(no volume: no derivative) first gets the weight that makes it own its position (w_i = max_j (w_j - |p_i - p_j|^2))."""
import numpy as np
import torch

from free_surface.alpha_complex import AlphaComplex
from free_surface.gpu_geometry import Geometry


class GraphedCG:
    """Recut.cg's iterations, check of them per replay of one CUDA graph, on fixed buffers: the pairs padded to a capacity
    with c = 0 (they add nothing), the graph captured again only when the pairs outgrow it. The same operations in the
    same order as the plain loop (the stop is read at the same iterations)"""

    def __init__(self, n, check, dev):
        self.n, self.check, self.dev, self.cap, self.graph = n, check, dev, 0, None

    def _body(self):
        for _ in range(self.check):
            Hq = self.d*self.q-torch.zeros_like(self.q).index_add(0, self.i, self.c*self.q[self.j]).index_add(0, self.j, self.c*self.q[self.i])
            al = self.rz/(self.q@Hq)
            self.x.add_(al*self.q); self.rr.sub_(al*Hq)
            z = self.rr/self.d; rz1 = self.rr@z
            self.q.mul_(rz1/self.rz).add_(z); self.rz.copy_(rz1)

    def _capture(self, m):
        self.cap = int(1.25*m)+64
        f64, n, dev = torch.float64, self.n, self.dev
        self.i = torch.zeros(self.cap, dtype=torch.long, device=dev); self.j = torch.zeros_like(self.i)
        self.c = torch.zeros(self.cap, dtype=f64, device=dev)
        self.d, self.q = torch.ones(n, dtype=f64, device=dev), torch.ones(n, dtype=f64, device=dev)
        self.x, self.rr = torch.zeros(n, dtype=f64, device=dev), torch.zeros(n, dtype=f64, device=dev)
        self.rz = torch.ones((), dtype=f64, device=dev)
        s_ = torch.cuda.Stream(device=dev); s_.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(s_):
            self._body()
        torch.cuda.current_stream(dev).wait_stream(s_)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._body()

    def __call__(self, J, b, scale, rtol, maxit):
        i, j, c, d = J
        m = len(i)
        if self.graph is None or m > self.cap:
            self._capture(m)
        self.i[:m] = i; self.i[m:] = 0; self.j[:m] = j; self.j[m:] = 0; self.c[:m] = c; self.c[m:] = 0
        self.d.copy_(d); self.x.zero_(); self.rr.copy_(b); z = b/d; self.q.copy_(z); self.rz.copy_(b@z)
        lim = rtol*(b/scale).abs().max()
        k = 0
        while k < maxit:
            self.graph.replay(); k += self.check
            if bool((self.rr/scale).abs().max() <= lim):
                break
        return self.x.clone(), k


class Recut:
    def __init__(self, n, device='cuda', margin=.05):
        self.n, self.dev, self.margin = n, device, margin
        self.A = AlphaComplex(n, device)
        self.stats = dict(geometry=0, cg=0, overlaps=0)
        self.gcg = None

    def geometry(self, p, r, reuse=False):
        """reuse: the overlap candidates of the last call at these sites while they still cover (AlphaComplex)"""
        if not reuse:
            self.A.overlaps(p, r, self.margin); self.stats['overlaps'] += 1
        out = self.A(p, r, reuse=True)
        bad = out['status'].clone(); bad[bad == 1] = 0
        if bool((bad > 0).any()):
            raise RuntimeError('alpha complex: per-cell storage overflow or an open cut (status %s)' % torch.unique(out['status']).tolist())
        S = self.A.structure_gpu(out, self.n)
        self.stats['geometry'] += 1
        return Geometry(p, r, S), S

    def jacobian(self, G, r):
        keep = G.wall_A > 0
        i, j = G.i[keep], G.j[keep]
        L = (G.p[j]-G.p[i]).norm(dim=1)
        c = (G.wall_A[keep]/(2*L)).detach()
        d = torch.zeros(self.n, dtype=r.dtype, device=r.device).index_add(0, i, c).index_add(0, j, c)+G.patch_A.detach()/(2*r)
        d = torch.where(d > 0, d, d.max())                                         # (hidden: repaired before any solve)
        return i, j, c, d

    def dvol(self, G, dp):
        """the volumes' change when the sites of G (cut at p, r) move by dp at fixed weights, to first order. A volume
        moves only by its boundary's normal velocity. The wall (i, j), n from i to j, L = |p_j - p_i|, lies on the
        power-equal plane |y - p_i|^2 - w_i = |y - p_j|^2 - w_j; its point y moves along n by ((y - p_i).dp_i - (y -
        p_j).dp_j) / L, linear in y, so dV_i = A / L ((c - p_i).dp_i - (c - p_j).dp_j) with c the wall's centroid, and
        dV_j = -dV_i. The cap moves with its site: dV_i = a_i.dp_i, a_i = -sum_walls A n its area vector (closure). A
        common translation keeps every volume (sum_walls A n + a = 0)"""
        keep = G.wall_A > 0
        i, j = G.i[keep], G.j[keep]
        A, c, p = G.wall_A.detach()[keep], G.wall_c.detach()[keep], G.p.detach()
        d = p[j]-p[i]; L = d.norm(dim=1)
        dw = (A/L)*(((c-p[i])*dp[i]).sum(1)-((c-p[j])*dp[j]).sum(1))
        a = torch.zeros_like(p).index_add(0, i, -(A/L)[:, None]*d).index_add(0, j, (A/L)[:, None]*d)
        return (a*dp).sum(1).index_add(0, i, dw).index_add(0, j, -dw)

    def predict(self, G, dp, Vt, rtol=1e-3):
        """radii for the sites of G moved by dp and the volumes Vt, to first order from G: J dw = Vt - V - dvol(G, dp)
        with the weights' Jacobian at G (10-28: the volume fit's start from the step's own motion instead of from the
        weights extrapolated in time)"""
        r = G.r.detach()
        dw = self.cg(self.jacobian(G, r), Vt-G.V.detach()-self.dvol(G, dp), Vt, rtol=rtol)
        w = r*r+dw
        return torch.sqrt(torch.where(w > 0, w, r*r))

    def cg(self, J, b, scale, rtol=1e-12, maxit=2000, check=10):
        """CG preconditioned by the diagonal; the scalars stay on the GPU, the residual is read every check iterations;
        it stops when every cell's residual relative to its scale (the target volume) is rtol times the start's
        largest (10-24: an overall norm left the tiny cells' corrections loose, 6e-9 against the tol 1e-9). On CUDA the
        check iterations between two reads are one replay of GraphedCG (10-28: 0.45 ms per iteration plain on the 8k
        foam, nearly all kernel launches)"""
        i, j, c, d = J
        if b.is_cuda:
            if self.gcg is None or self.gcg.check != check:
                self.gcg = GraphedCG(self.n, check, b.device)
            x, k = self.gcg(J, b, scale, rtol, maxit)
            self.stats['cg'] += k
            return x
        def H(x):
            return d*x-torch.zeros_like(x).index_add(0, i, c*x[j]).index_add(0, j, c*x[i])
        x = torch.zeros_like(b); rr = b.clone(); z = rr/d; q = z.clone(); rz = rr@z; lim = rtol*(b/scale).abs().max()
        for k in range(maxit):
            Hq = H(q); al = rz/(q@Hq)
            x += al*q; rr -= al*Hq
            if (k+1) % check == 0 and bool((rr/scale).abs().max() <= lim):
                break
            z = rr/d; rz1 = rr@z; q = z+(rz1/rz)*q; rz = rz1
        self.stats['cg'] += k+1
        return x

    def repair(self, p, r, hid):
        """each hidden site's weight raised until it owns its own position (over the sites whose balls could reach it)"""
        w = r*r
        for i_ in torch.nonzero(hid).flatten().tolist():
            d2 = ((p-p[i_])**2).sum(1)
            reach = (d2 < (r.max()+r[i_])**2); reach[i_] = False
            need = float((w[reach]-d2[reach]).max())
            w[i_] = max(float(w[i_]), need*(1+1e-9)+1e-12*float(w.max()))
        return torch.sqrt(w)

    def solve_radii(self, p, r, Vt, tol=1e-9, maxit=200, r_base=None):
        """radii from r with V = Vt at sites p; returns r, its geometry and structure, Newton iterations, repairs.
        r_base: radii r is a step from (the last step's for a predicted start): a start whose cut overflows the storage
        is moved halfway back to r_base in w until it does not (as a refused trial of the Newton step; 10-28 lion, a
        predicted weight x 30). The repair (each hidden or empty site's weight raised until it owns its position) is a
        relaxation of the difference constraints w_j - w_i <= |p_i - p_j|^2, done in rounds until none is violated; a
        chain of n sites needs at most n rounds (10-28 lion: 2 hidden sites spread to 307 over 21 rounds, and the old
        cap of 20 left two empty cells silently, the line search then had no step). maxit 30 -> 200 (10-28 lion, deep
        compression: 30 iterations ended at 3.9e-4); the caller checks the error"""
        while True:
            try:
                G, S = self.geometry(p, r)
                break
            except RuntimeError:
                if r_base is None or float(((r*r-r_base*r_base)/(r_base*r_base)).abs().max()) <= 4*torch.finfo(r.dtype).eps:
                    raise                                                      # (the base itself overflows: no start left)
                self.stats['overflow'] = self.stats.get('overflow', 0)+1
                r = torch.sqrt(.5*(r*r+r_base*r_base))
        nrep = 0
        for rd in range(self.n+1):
            hid = S['hidden'] | (G.V.detach() <= 0)
            if not bool(hid.any()):
                break
            if rd == self.n:
                raise RuntimeError('solve_radii: %d sites still hidden or empty after %d repair rounds' % (int(hid.sum()), rd))
            r = self.repair(p, r, hid); nrep += int(hid.sum())
            G, S = self.geometry(p, r)
        res = Vt-G.V.detach()
        q = G.V.detach()/Vt                                                        # V / target at the start: extremes, 1 / 99 %,
        off = (q < .5) | (q > 2)                                                   # the material share outside [0.5, 2]
        self.stats['start_ratio'] = (float(q.min()), float(q.max()), float(torch.quantile(q, .01)), float(torch.quantile(q, .99)),
                                     float(Vt[off].sum()/Vt.sum()), torch.nonzero(off).flatten())
        it = 0
        while True:
            err = float((res/Vt).abs().max())
            if err <= tol or it >= maxit:
                break
            it += 1
            dw = self.cg(self.jacobian(G, r), res, Vt, rtol=min(.1, err))       # inexact Newton: the solve as fine as the error
            w = r*r
            if float((dw/w).abs().max()) <= 4*torch.finfo(w.dtype).eps:            # the step changes no weight: converged
                break                                                              # (10-24: a 1e-10 bound stopped tiny cells, big
                                                                                   # radii, at 6e-9 volume error)
            st, nr = 1., float(res.norm())
            while True:
                wt = w+st*dw
                if bool((wt > 0).all()):
                    try:
                        Gt, St = self.geometry(p, torch.sqrt(wt), reuse=True)
                    except RuntimeError:                                       # a trial weight so large its cell overflows the
                        Gt = None                                              # storage: a failed trial (10-27: tiny cells near
                        self.stats['overflow'] = self.stats.get('overflow', 0)+1   # twins asked for w x 18 at full step)
                    if Gt is not None:
                        rt = Vt-Gt.V.detach()
                        if not bool(St['hidden'].any()) and bool((Gt.V > 0).all()) and float(rt.norm()) <= (1-st/2)*nr:
                            break
                st *= .5
                if st*float((dw/w).abs().max()) <= torch.finfo(w.dtype).eps:   # no step that still moves a weight lowers
                    self.stats['floor'] = self.stats.get('floor', 0)+1        # the residual: the volumes are at their own
                    return r, G, S, it, nrep                                   # evaluation floor (10-25: radii x (1 +
                                                                               # 1e-13) moved a cell's V by 6e-9, so tol
                                                                               # 1e-9 can lie below it)
            r, G, S, res = torch.sqrt(wt), Gt, St, rt
        return r, G, S, it, nrep


class GroundView:
    """the real cells of a cut with mirror sites (GroundRecut): walls between real cells (i, j, wall_A, wall_c), V, patch_A, p, r of the real sites; the ground faces
    (each real cell's wall with its own mirror: g_i, g_A, g_c, on the plane) apart; walls to another cell's mirror (none of positive area: a cell's mirror lies under
    the plane wherever the cell owns the point above it) counted in other_A_max"""

    def __init__(self, G, n, parent):
        kp = G.wall_A.detach() > 0
        i, j = G.i, G.j
        real = (i < n) & (j < n)
        self.i, self.j, self.wall_A, self.wall_c = i[real], j[real], G.wall_A[real], G.wall_c[real]
        gm = kp & (i < n) & (j >= n)
        own = gm & (parent[j.clamp(max=len(parent)-1)] == i)
        self.g_i, self.g_A, self.g_c = i[own], G.wall_A.detach()[own], G.wall_c.detach()[own]
        oth = gm & ~own
        self.other_A_max = float(G.wall_A.detach()[oth].max()) if bool(oth.any()) else 0.
        self.V, self.patch_A, self.p, self.r = G.V[:n], G.patch_A[:n], G.p[:n], G.r[:n]
        self.n_mirror = len(G.V)-n


class GroundRecut(Recut):
    """the cut bounded by a rigid plane n . y = c0 (n the up normal): every cell whose ball reaches the plane gets a mirror site, its reflection with the same radius; the
    power plane of a site and its mirror is the plane itself, so the cut of the real cells ends at it and nothing of theirs lies under it. The radii stay tied (a mirror's
    is its parent's): the Newton on the real cells only, its Jacobian over the real walls (a ground face does not move when both weights move alike), so Recut's solve
    holds unchanged on the view. A real cell is cut against the real cells and its own mirror only (another cell's mirror never bounds it above the plane, and kept it
    ties four sites on every edge of a ground face), a mirror against its parent only (its own cell is not used). cap: the alpha complex's capacity (real + mirrors)"""

    def __init__(self, n, c0, normal=(0., 0., 1.), device='cuda', margin=.05, cap=None):
        super().__init__(n, device, margin)
        self.c0, self.ng = float(c0), torch.tensor(normal, dtype=torch.float64, device=device)
        self.ng = self.ng/self.ng.norm()
        self.A = AlphaComplex(cap if cap is not None else 2*n, device)

    def geometry(self, p, r, reuse=False):
        n = self.n
        h = p@self.ng-self.c0
        M = torch.nonzero(h < r)[:, 0]
        P = torch.cat([p, p[M]-2*h[M, None]*self.ng[None, :]]); R = torch.cat([r, r[M]])
        parent = torch.cat([torch.arange(n, device=p.device), M])
        if len(P) > self.A.n:
            raise RuntimeError('GroundRecut: %d sites with the mirrors over the capacity %d' % (len(P), self.A.n))
        self.A.overlaps(P, R, self.margin); self.stats['overlaps'] += 1                # (the mirrors change with the radii: no reuse)
        a, b = self.A.ov                                                               # (each real cell sees the real cells and its own mirror only, a mirror its parent only: above
        ra, rb = a < n, b < n                                                          # the plane |y - p_k'|^2 = |y - p_k|^2 + 4 h_k (n.y - c0) > |y - p_k|^2, so another cell's mirror
        keep = (ra & rb) | (ra & ~rb & (parent[b] == a)) | (~ra & rb & (parent[a] == b))   # never bounds a real cell; kept, it made i, k, i', k' co-spherical on every edge of a
        self.A.ov = (a[keep], b[keep])                                                 # ground face, and the clipping read that tie at random: 11-03 step 336, three cells V < 0)
        out = self.A(P, R, reuse=True)
        bad = out['status'].clone(); bad[bad == 1] = 0
        if bool((bad > 0).any()):
            raise RuntimeError('alpha complex: per-cell storage overflow or an open cut (status %s)' % torch.unique(out['status']).tolist())
        S = self.A.structure_gpu(out, len(P))
        self.stats['geometry'] += 1
        G = Geometry(P, R, S)
        return GroundView(G, n, parent), dict(hidden=S['hidden'][:n], pairs=S['pairs'], mirrors=len(M))
