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


class Recut:
    def __init__(self, n, device='cuda', margin=.05):
        self.n, self.dev, self.margin = n, device, margin
        self.A = AlphaComplex(n, device)
        self.stats = dict(geometry=0, cg=0, overlaps=0)

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

    def cg(self, J, b, scale, rtol=1e-12, maxit=2000, check=10):
        """CG preconditioned by the diagonal; the scalars stay on the GPU, the residual is read every check iterations;
        it stops when every cell's residual relative to its scale (the target volume) is rtol times the start's
        largest (10-24: an overall norm left the tiny cells' corrections loose, 6e-9 against the tol 1e-9)"""
        i, j, c, d = J
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

    def solve_radii(self, p, r, Vt, tol=1e-9, maxit=30):
        """radii from r with V = Vt at sites p; returns r, its geometry and structure, Newton iterations, repairs"""
        G, S = self.geometry(p, r)
        nrep = 0
        for _ in range(20):
            hid = S['hidden'] | (G.V.detach() <= 0)
            if not bool(hid.any()):
                break
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
