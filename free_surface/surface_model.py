"""Continuous outer surface (2026-10-22, step 2 of free_surface/PLAN.md): the surface vertices y_v (shared by the 3-4 air
faces that meet there, surface_mesh.py) are degrees of freedom next to the body positions x_B. Every air face f of body
B (rest vertices Y_v, area A, outward normal n) is represented by its area centroid
  y_c = sum_v b_v y_v,  b_v = (a_k-1 + a_k) / 3A + 1 / 3k  (fan from the vertex mean, a_k the fan triangles' areas)
(fixed positive weights, exact for affine maps), rest lever r = X_c - X_B = h n + t (h signed distance to the face
plane, t in-plane offset). The face enters in three places, all through y_c, as the ground wall of KB_DERIVATION.md 20:
  gradient  F_B = [S_B + sum_f w_f (y_c - x_B) (x) r] [M_B + sum_f w_f r r^T]^-1,  w_f = A |h| / |r|^2 (the face's moment
            A x its distance, like a bond's A x L/2); S_B, M_B the bonds' L/2 weights (grad_weights.MidpointTerms). No
            neighbour fill F_N.
  volume    V'_B = Hermite walls (as CleanTerms.J) + sum_f (y_c - x_B).cof(F_B) A n / 3 + det(F_B) V_res,
            V_res = V_B - walls - sum_f A h / 3 at rest (walls behind the site, merged-face residue, ball clipping)
  bond      half bond kn = ks = 2 mu A / h on x_B + R_B h n + F_B t - y_c for h > 0: the normal part turns with R_B,
            the in-plane offset follows F_B (as the walls' line position), so for any affine map it equals the completion
            k/2 |(F_B - R_B) l|^2; none for h <= 0, as now.
Every force reaches the vertices through the centroids: a face in balance leaves all its vertices in balance. (A first
version, 10-22, used the site's foot on the face plane and the vertices' own polygon for the pyramid: the foot lay
outside 68 % of the new shape's faces, weights up to 3e14; and the polygon pyramid put the volume change on the moved
vertices while the walls' share went through F_B to all of the face's vertices: 4 % of a face force left on lattice
vertices next to the edges at uniform uniaxial stress.)
One more term for the vertex motions the centroids do not see (sliding in the face plane, warping):
  surface   sum_f sum_v mu |A h / 3| / k_f |y_v - x_B - F_B (Y_v - X_B)|^2 / |Y_v - X_B|^2: the extra strain of the
            pyramid over its own affine field, zero for any affine map.
The completion springs and exposed sphere patches of the clean model are dropped (the mesh is the whole surface).
Polynomial det / cofactor: torch.linalg.det's second derivative is NaN at F = I (face_modes.py)."""
import numpy as np
import torch

from experiments.foam_surface_completion import matrix_quat, polar_newton
from free_surface.grad_weights import MidpointTerms


def det3(F):
    return (F[..., 0, 0]*(F[..., 1, 1]*F[..., 2, 2]-F[..., 1, 2]*F[..., 2, 1])-F[..., 0, 1]*(F[..., 1, 0]*F[..., 2, 2]-F[..., 1, 2]*F[..., 2, 0])
            + F[..., 0, 2]*(F[..., 1, 0]*F[..., 2, 1]-F[..., 1, 1]*F[..., 2, 0]))


def cof3(F):
    f1, f2, f3 = F[..., :, 0], F[..., :, 1], F[..., :, 2]
    return torch.stack((torch.cross(f2, f3, dim=-1), torch.cross(f3, f1, dim=-1), torch.cross(f1, f2, dim=-1)), -1)


def face_rest(Y, off, fv):
    """Rest polygons (numpy): area vectors and centroid weights (entries face, vertex, weight); zero-area faces: vertex mean"""
    F = len(off)-1
    area = np.zeros((F, 3)); ew = np.empty(len(fv))
    for f in range(F):
        P = Y[fv[off[f]:off[f+1]]]; k = len(P); c = P.mean(0)
        tri = .5*np.cross(P-c, np.roll(P, -1, axis=0)-c)
        a = tri.sum(0); A = np.linalg.norm(a)
        area[f] = a
        if A > 0:
            ak = tri@(a/A)
            ew[off[f]:off[f+1]] = (np.roll(ak, 1)+ak)/(3*A)+1/(3*k)
        else:
            ew[off[f]:off[f+1]] = 1/k
    return area, np.repeat(np.arange(F), np.diff(off)), np.asarray(fv), ew


class SurfaceTerms(MidpointTerms):
    def __init__(self, model, sc, geometry, mesh, variance=True, surface_term=True):
        """mesh: dict with Y (rest vertices, model frame), face_off, face_vert, face_body"""
        super().__init__(model, sc, geometry, variance)
        dt, dev = self.X.dtype, self.X.device
        tt = lambda a: torch.as_tensor(np.asarray(a), dtype=dt, device=dev)
        ti = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.long, device=dev)
        Y = np.asarray(mesh['Y'], float); off = np.asarray(mesh['face_off']); fv = np.asarray(mesh['face_vert']); fb = np.asarray(mesh['face_body'])
        area, ef, ev, ew = face_rest(Y, off, fv)
        self.nf, self.nv = len(off)-1, len(Y)
        self.Y = tt(Y); self.fb = ti(fb)
        self.ef, self.ev, self.ew = ti(ef), ti(ev), tt(ew)
        self.fAn = tt(area); self.fA = self.fAn.norm(dim=1)
        self.fn = torch.where(self.fA[:, None] > 0, self.fAn/self.fA.clamp(min=1e-300)[:, None], torch.zeros_like(self.fAn))
        Xc = self.centroid(self.Y)
        self.r = Xc-self.X[self.fb]
        self.fh = (self.r*self.fn).sum(1)
        self.rn = self.fh[:, None]*self.fn; self.rt = self.r-self.rn
        # gradient: bonds (MidpointTerms.M) + air faces w r r^T, moment A |h|; no neighbour fill
        r2 = (self.r*self.r).sum(1)
        self.fw = self.fA*self.fh.abs()/r2.clamp(min=1e-300)
        self.Mface = torch.zeros(self.n, 3, 3, dtype=dt, device=dev).index_add(0, self.fb, self.fw[:, None, None]*self.r[:, :, None]*self.r[:, None, :])
        self.Mt_inv = torch.linalg.inv(self.M+self.Mface)
        # volume: residue on det F_B
        pyr0 = self.fA*self.fh/3
        self.Vres = self.Vrest-torch.zeros_like(self.Vrest).index_add(0, self.fb, pyr0)
        # half bonds (h > 0)
        mu = float(model.mu_ref); self.mu = mu
        self.kh = torch.where(self.fh > 0, 2*mu*self.fA/self.fh.clamp(min=1e-300), torch.zeros_like(self.fh))
        # surface term per (face, vertex) entry
        kf = torch.as_tensor(np.diff(off), dtype=dt, device=dev)
        self.s_body = self.fb[self.ef]
        self.s_d = self.Y[self.ev]-self.X[self.s_body]
        self.s_w = mu*pyr0.abs()[self.ef]/kf[self.ef]/(self.s_d*self.s_d).sum(1) if surface_term else torch.zeros(len(fv), dtype=dt, device=dev)

    def centroid(self, y):
        return torch.zeros(self.nf, 3, dtype=y.dtype, device=y.device).index_add(0, self.ef, self.ew[:, None]*y[self.ev])

    def F(self, x, y):
        d = x[self.j]-x[self.i]
        t = (self.A/2)[:, None, None]*d[:, :, None]*self.nrm[:, None, :]
        S = torch.zeros(self.n, 3, 3, dtype=x.dtype, device=x.device).index_add(0, self.i, t).index_add(0, self.j, t)
        g = self.fw[:, None, None]*(self.centroid(y)-x[self.fb])[:, :, None]*self.r[:, None, :]
        return S.index_add(0, self.fb, g)@self.Mt_inv

    def wall_volume(self, x, F, R):
        """CleanTerms.J's Hermite walls (deformed pyramids of the front walls), without its det(F) V_rest"""
        i, j = self.i, self.j
        Ff, Rf = .5*(F[i]+F[j]), .5*(R[i]+R[j])
        mv = lambda M_, v: torch.einsum('bij,bj->bi', M_, v)
        line = (self.hj/self.L)[:, None]*(x[i]+mv(R[i], self.ri))+(self.hi/self.L)[:, None]*(x[j]+mv(R[j], self.rj))+mv(Ff-Rf, self.t)
        comp = (self.hj/self.L)[:, None]*(x[i]+mv(F[i], self.ri))+(self.hi/self.L)[:, None]*(x[j]+mv(F[j], self.rj))
        xf = .5*(line+comp)
        a = mv(cof3(Ff), self.A[:, None]*self.nrm)
        vi = (self.wi*((xf-x[i])*a).sum(1))/3; vj = (self.wj*(-(xf-x[j])*a).sum(1))/3
        return torch.zeros(self.n, dtype=x.dtype, device=x.device).index_add(0, i, vi).index_add(0, j, vj)

    def energy_terms(self, theta, x, y):
        F = self.F(x, y)
        R = polar_newton(F); q = matrix_quat(R)
        m = self.model
        bonds = m.energy(theta, x, q)
        if getattr(m, 'body_law', None) == 'nodalvol':
            bonds = bonds-m._nodalvol(theta, x)[0]
        lam = m.k_v*torch.exp(theta[0, 4])
        yc = self.centroid(y); Fb = F[self.fb]
        pyr = ((yc-x[self.fb])*torch.einsum('fij,fj->fi', cof3(Fb), self.fAn)).sum(1)/3
        Vd = self.wall_volume(x, F, R).index_add(0, self.fb, pyr)+det3(F)*self.Vres
        lj, _ = m._log_J(Vd/self.V)
        vol = (self.V*.5*lam*lj**2).sum()
        if self.variance:
            vol = vol+(.5*lam*self.A*self.L/12*(lj[self.i]-lj[self.j])**2).sum()
        sm = torch.exp(theta[:, 5])
        p = x[self.fb]+torch.einsum('fij,fj->fi', R[self.fb], self.rn)+torch.einsum('fij,fj->fi', Fb, self.rt)
        hb = (.5*self.kh*sm[self.fb]*((p-yc)**2).sum(1)).sum()
        dl = y[self.ev]-x[self.s_body]-torch.einsum('eij,ej->ei', F[self.s_body], self.s_d)
        surf = (self.s_w*sm[self.s_body]*(dl*dl).sum(1)).sum()
        return dict(bonds=bonds, volume=vol, half_bond=hb, surface=surf)

    def energy_xy(self, theta, x, y):
        return sum(self.energy_terms(theta, x, y).values())

    def affine(self, Fa, c, t=None):
        """bodies and vertices on the affine map x = c + Fa (X - c) (+ t)"""
        t = torch.zeros(3, dtype=self.X.dtype, device=self.X.device) if t is None else t
        return c+(self.X-c)@Fa.T+t, c+(self.Y-c)@Fa.T+t
