"""Primitive update operators for rendering a Power Foam driven by DEM bodies.

rigid:  p = x_b + R_b (p0 - X_b), frame rotated by the body rotation R_b, radius fixed.
affine: PowerSim-style update with the body's least-squares deformation gradient F_b
        (EnergyFoam.deformation_gradient, the F of the Neo-Hookean body energy):
        p = x_b + F_b (p0 - X_b); polar rotation R_b of F_b rotates the frame and the
        spherical-Voronoi radiance axes; radius r0 det(F_b)^(1/3).
PowerFoam stores the normal/tangents as ROWS of the quaternion's matrix, so a world rotation R
updates q0 -> q0 (x) conj(quat(R)) (PowerSim eq. 7); radiance axes are world vectors: a -> R a.
blend:  continuous deformation field (linear blend skinning / embedded deformation): every cell
        mixes the affine maps x_k + F_k (p0 - X_k) of its K nearest bodies with Gaussian weights
        of its rest distance (width = `width` x median nearest-body spacing); rotation and radius
        come from the blended F. Exact for rigid and homogeneous motion, continuous across body
        boundaries, so neighbouring bodies' cells no longer tear apart.
Positions, cell rotations (surface planes) and cell sizes keep their autograd graph (for training; the rotation factor uses
the polar-decomposition adjoint, stable at rigid motion); adjacency is rebuilt from detached copies.
"""
import numpy as np
import torch

from experiments.foam_dynamic_render import quat_mul


def quat_from_matrix(R):
    """Unit quaternion (w, x, y, z) of rotation matrices, branch-stable (Shepperd)."""
    m = R
    t = m[..., 0, 0]+m[..., 1, 1]+m[..., 2, 2]
    cands = torch.stack((
        torch.stack((1+t, m[..., 2, 1]-m[..., 1, 2], m[..., 0, 2]-m[..., 2, 0], m[..., 1, 0]-m[..., 0, 1]), -1),
        torch.stack((m[..., 2, 1]-m[..., 1, 2], 1+m[..., 0, 0]-m[..., 1, 1]-m[..., 2, 2], m[..., 0, 1]+m[..., 1, 0], m[..., 0, 2]+m[..., 2, 0]), -1),
        torch.stack((m[..., 0, 2]-m[..., 2, 0], m[..., 0, 1]+m[..., 1, 0], 1-m[..., 0, 0]+m[..., 1, 1]-m[..., 2, 2], m[..., 1, 2]+m[..., 2, 1]), -1),
        torch.stack((m[..., 1, 0]-m[..., 0, 1], m[..., 0, 2]+m[..., 2, 0], m[..., 1, 2]+m[..., 2, 1], 1-m[..., 0, 0]-m[..., 1, 1]+m[..., 2, 2]), -1)), -2)
    key = torch.stack((t, m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]), -1).argmax(-1)
    q = torch.gather(cands, -2, key[..., None, None].expand(*key.shape, 1, 4))[..., 0, :]
    return q/q.norm(dim=-1, keepdim=True)


def _skew(h):
    z = torch.zeros_like(h[..., 0])
    return torch.stack((torch.stack((z, -h[..., 2], h[..., 1]), -1), torch.stack((h[..., 2], z, -h[..., 0]), -1),
                        torch.stack((-h[..., 1], h[..., 0], z), -1)), -2)


class _PolarRotation(torch.autograd.Function):
    """Rotation factor R of F = R S (S symmetric). Forward by SVD; backward by the derivative of the polar factor
    itself, not of the SVD: with R^T dR = [w]x, the skew part of R^T dF gives (tr(S) I - S) w = vee(R^T dF - dF^T R),
    so dL/dF = R [h]x with h = (tr(S) I - S)^-1 vee(R^T G - G^T R), G = dL/dR. tr(S) I - S has eigenvalues
    s_i + s_j (sums of singular values, ~2 for small strain), so it stays well conditioned when the singular values
    coincide (rigid motion), where the SVD adjoint (1 / (s_i^2 - s_j^2)) blows up."""

    @staticmethod
    def forward(ctx, F):
        U, _, Vh = torch.linalg.svd(F)
        flip = torch.det(U@Vh) < 0
        U = torch.where(flip[:, None, None], U*torch.tensor([1., 1., -1.], dtype=U.dtype, device=U.device), U)
        R = U@Vh
        S = R.transpose(-1, -2)@F
        ctx.save_for_backward(R, .5*(S+S.transpose(-1, -2)))
        return R

    @staticmethod
    def backward(ctx, G):
        R, S = ctx.saved_tensors
        K = R.transpose(-1, -2)@G
        K = K-K.transpose(-1, -2)
        g = torch.stack((K[..., 2, 1], K[..., 0, 2], K[..., 1, 0]), -1)
        M = (S[..., 0, 0]+S[..., 1, 1]+S[..., 2, 2])[..., None, None]*torch.eye(3, dtype=S.dtype, device=S.device)-S
        h = torch.linalg.solve(M, g[..., None])[..., 0]
        return R@_skew(h)


def polar_rotation(F):
    return _PolarRotation.apply(F)


def inv_softplus(y, beta=100.):
    return y+torch.log(-torch.expm1(-beta*y))/beta


class DeformedFoam:
    """Holds the rest state of the kept cells and applies an update operator per frame."""

    def __init__(self, foam, owner, body_rest, offset):
        self.foam = foam
        self.owner = torch.as_tensor(owner, device=foam.device)
        self.p0 = foam.points.data.clone()
        self.q0 = foam.quaternions.data/foam.quaternions.data.norm(dim=-1, keepdim=True)
        self.r0 = foam.get_radii().detach().clone()
        self.axis0 = foam.texel_sv_axis.data.clone()
        self.offset = torch.as_tensor(offset, dtype=self.p0.dtype, device=foam.device)
        self.lever = self.p0-(torch.as_tensor(body_rest, dtype=self.p0.dtype, device=foam.device)+self.offset)[self.owner]

    def prepare_blend(self, body_rest, K=8, width=1.):
        from scipy.spatial import cKDTree
        X = np.asarray(body_rest, float)
        spacing = float(np.median(cKDTree(X).query(X, k=2)[0][:, 1]))
        p0 = (self.p0.double().cpu().numpy()-self.offset.double().cpu().numpy())
        d, idx = cKDTree(X).query(p0, k=K)
        h = width*spacing
        w = np.exp(-.5*(d/h)**2)
        w /= w.sum(1, keepdims=True)
        dev, dt = self.p0.device, self.p0.dtype
        self.blend_idx = torch.as_tensor(idx, device=dev)
        self.blend_w = torch.as_tensor(w, dtype=dt, device=dev)
        Xt = torch.as_tensor(X, dtype=dt, device=dev)+self.offset
        self.blend_lever = self.p0[:, None, :]-Xt[self.blend_idx]       # (cells, K, 3)

    def prepare_tet(self, body_rest, tets, k=24, min_quality=.05):
        """Embed every cell in a tetrahedron of the bodies (tets: (T, 4) body indices, the physics' mesh): barycentric
        coordinates of the cell's rest centre. Cells inside a tetrahedron use it (positions are then a convex combination);
        cells outside the mesh use, among the k tetrahedra nearest by centroid, the one with quality >= min_quality whose
        barycentric coordinates are least negative (linear extrapolation of that tetrahedron's map)."""
        from scipy.spatial import cKDTree
        X = np.asarray(body_rest, float)
        T = np.asarray(tets, int)
        p0 = self.p0.double().cpu().numpy()-self.offset.double().cpu().numpy()
        P = X[T]
        D0 = np.stack((P[:, 1]-P[:, 0], P[:, 2]-P[:, 0], P[:, 3]-P[:, 0]), -1)          # columns: edges from corner 0
        vol = np.linalg.det(D0)/6
        L = np.max([np.linalg.norm(P[:, a]-P[:, b], axis=1) for a in range(4) for b in range(a+1, 4)], axis=0)
        quality = 6*np.sqrt(2)*np.abs(vol)/np.maximum(L**3, 1e-300)
        D0inv = np.linalg.inv(D0+np.eye(3)*1e-30)
        tree = cKDTree(P.mean(1))
        _, cand = tree.query(p0, k=min(k, len(T)))
        lam3 = np.einsum('ckij,ckj->cki', D0inv[cand], p0[:, None, :]-P[cand, 0])           # (cells, k, 3)
        bary = np.concatenate((1-lam3.sum(-1, keepdims=True), lam3), -1)                     # (cells, k, 4)
        mn = bary.min(-1)
        inside = mn >= -1e-9
        score = np.where(inside, 1.+quality[cand], np.where(quality[cand] >= min_quality, mn, -np.inf))
        pick = np.argmax(score, axis=1)
        rows = np.arange(len(p0))
        dev, dt = self.p0.device, self.p0.dtype
        self.tet_T = torch.as_tensor(T, device=dev)
        self.tet_c = torch.as_tensor(cand[rows, pick], device=dev)
        self.tet_bary = torch.as_tensor(bary[rows, pick], dtype=dt, device=dev)
        self.tet_inside = torch.as_tensor(inside[rows, pick], device=dev)
        self.tet_D0inv = torch.as_tensor(D0inv, dtype=dt, device=dev)
        self.tet_vol = torch.as_tensor(np.abs(vol), dtype=dt, device=dev)
        n = len(X)
        node_v = np.bincount(T.reshape(-1), weights=np.repeat(np.abs(vol), 4), minlength=n)
        self.tet_node_v = torch.as_tensor(np.maximum(node_v, 1e-300), dtype=dt, device=dev)
        return dict(cells=len(p0), inside=int(inside[rows, pick].sum()), min_bary=float(bary[rows, pick].min()))

    def tet_fields(self, body_x):
        """Per tetrahedron F_t = D(x) D0^-1; per body the volume-weighted average of its tetrahedra's F."""
        x = body_x.to(self.p0.dtype)
        P = x[self.tet_T]
        D = torch.stack((P[:, 1]-P[:, 0], P[:, 2]-P[:, 0], P[:, 3]-P[:, 0]), -1)
        Ft = D@self.tet_D0inv
        n = len(x)
        Fn = torch.zeros(n, 3, 3, dtype=x.dtype, device=x.device).index_add(
            0, self.tet_T.reshape(-1), (self.tet_vol[:, None, None]*Ft).repeat_interleave(4, 0))/self.tet_node_v[:, None, None]
        return Ft, Fn

    def apply(self, body_x, body_R=None, body_F=None, mode='affine'):
        """body_x: (bodies, 3) world centres; rigid needs body_R, affine needs body_F."""
        f = self.foam
        o = self.owner
        x = body_x.to(self.p0.dtype)[o]+self.offset
        if mode == 'tet':
            # embedded in the bodies' tetrahedra: positions by barycentric interpolation (no fitted F); rotation and scale
            # from the corners' volume-averaged tetrahedron gradients
            xb = body_x.to(self.p0.dtype)
            corners = self.tet_T[self.tet_c]                                       # (cells, 4)
            p = (self.tet_bary[..., None]*xb[corners]).sum(1)+self.offset
            _, Fn = self.tet_fields(xb)                                            # differentiable: cell rotation and size follow x
            wF = self.tet_bary.clamp(min=0)
            wF = wF/wF.sum(1, keepdim=True).clamp(min=1e-12)
            Fc = (wF[..., None, None]*Fn[corners]).sum(1)
            R = polar_rotation(Fc)
            scale = torch.det(Fc).clamp(min=1e-3)**(1/3)
        elif mode == 'blend':
            F = body_F.to(self.p0.dtype)
            idx, w = self.blend_idx, self.blend_w[..., None]
            xk = body_x.to(self.p0.dtype)[idx]+self.offset                          # (cells, K, 3)
            p = (w*(xk+(F[idx]@self.blend_lever[..., None])[..., 0])).sum(1)
            Fc = (w[..., None]*F[idx]).sum(1)
            R = polar_rotation(Fc)
            scale = torch.det(Fc).clamp(min=1e-3)**(1/3)
        elif mode == 'rigid':
            R = body_R.to(self.p0.dtype)[o]
            p = x+(R@self.lever[..., None])[..., 0]
            scale = None
        else:
            F = body_F.to(self.p0.dtype)
            R = polar_rotation(F)[o]
            p = x+(F[o]@self.lever[..., None])[..., 0]
            scale = torch.det(F).clamp(min=1e-3)[o]**(1/3)
        qR = quat_from_matrix(R)                                                   # cell surface planes rotate with R (silhouette)
        q = quat_mul(self.q0, qR*torch.tensor([1., -1., -1., -1.], dtype=qR.dtype, device=qR.device))
        f._parameters.pop('points', None)
        f.points = p
        f._parameters.pop('quaternions', None)
        f.quaternions = q
        k = self.axis0.shape[1]//3
        axis = self.axis0.view(len(o), -1, 3)                     # (cells, K*sv_dof, 3) world vectors
        f._parameters.pop('texel_sv_axis', None)
        f.texel_sv_axis = (R.detach()[:, None]@axis[..., None])[..., 0].reshape(self.axis0.shape)     # radiance axes: colour only
        f._parameters.pop('radii', None)
        f.radii = inv_softplus(self.r0*scale) if scale is not None else inv_softplus(self.r0)
        with torch.no_grad():
            f.aabb_tree.update(p.detach(), f.get_radii())
            f.adjacency, f.adjacency_offsets = f.aabb_tree.build_cech_complex()
