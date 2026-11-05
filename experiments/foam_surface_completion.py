"""Surface completion of the bonded DEM's stiffness from the Power Foam's own geometry (2026-10-19).

Under a uniform strain the spring network (kn = ks = 2 mu A / L) gives mu tr(eps^2 T), T = sum_bonds A L n n^T, and the
nodal tetrahedra give the volume term. Both are exact on any Power diagram in the interior: per cell
V I = sum_faces A (c_f - x) (x) n, and the face-centroid terms of a shared face cancel between its two cells. Both miss
the outer layer, though: on the PAC-NeRF jack the bonds give 0.827 and the Delaunay tets 0.830.

- Volume. J_B is the nodal average (Bonet & Burton 1998) over the tetrahedra of the foam's regular triangulation (power
  weights r^2; its edges are the bonds), with all four corners material. Merged cells collapse to their body centre;
  tetrahedra with a repeated body are dropped. The energy is sum_B V_B lam/2 L(J_B)^2, where V_B is the body's foam
  volume, not the tetrahedra's share of it. Any tetrahedral average is exact under a uniform strain, so this restores
  the missing outer part of lam. A surface body's J comes from its inward tetrahedra, i.e. the outer half cell is
  taken to change volume like the material inside it.
- Springs. Per material-air face, a virtual spring kn = ks = 2 mu A h / L^2 (h the material side of the centre line,
  L = |X_air - X_cell|). It runs from the body to its mirror point X_B - (X_air - X_cell) inside the material, whose
  motion is interpolated by the containing tetrahedron. The spring is measured in the body frame
  (x_B + R_B (X_m - X_B)), so it is objective. Per exposed sphere patch (cell clipped by its own ball, no neighbour
  cell), up to three virtual springs run along the eigenvectors of int r n n^T dA (k = 2 mu s / r^2, lever r), to
  mirror points as well. Under a uniform strain the mirror point gives the same stretch as the air site would.
  Together with the bonds this gives T = V I up to the face-centroid offset term (traceless; about +-0.8% by
  direction on the jack).

Torch only (float64 checks, dem_surface_completion_check.py). Energies are per unit density, as EnergyFoam's.
"""
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import ConvexHull, cKDTree
from scipy.stats import qmc

from numba import njit

from experiments.bread_material_geometry import clipped_faces, neighbors, volume_from_faces
from experiments.foam_dem_energy import quat_matrix
from experiments.powersim_bread_demo import regular_candidates



def matrix_quat(R):
    """Unit quaternions (w, x, y, z) of rotation matrices (n, 3, 3), differentiable (Shepperd-free form, w >= 0 branch
    blended by the largest diagonal choice is not needed for rotations near identity used here: w = sqrt(1 + tr) / 2)."""
    tr = R[:, 0, 0]+R[:, 1, 1]+R[:, 2, 2]
    w = .5*torch.sqrt((1+tr).clamp(min=1e-12))
    return torch.stack((w, (R[:, 2, 1]-R[:, 1, 2])/(4*w), (R[:, 0, 2]-R[:, 2, 0])/(4*w), (R[:, 1, 0]-R[:, 0, 1])/(4*w)), 1)


@njit(cache=True)
def face_boundary_points(p, r, pairs, offsets, ids, segments, spacing, cap):
    """Points on the clipped power faces between pairs[f, 0] and pairs[f, 1] (clipped_faces' polygon): the centroid and
    the polygon boundary resampled at equal arc length, at most `spacing` apart (at least 4, at most cap - 1)."""
    count = len(pairs)
    out = np.zeros((count, cap, 3)); num = np.zeros(count, np.int64)
    for edge in range(count):
        i, j = pairs[edge]
        d = p[j]-p[i]; length = np.sqrt(np.sum(d*d)); normal = d/length
        a = (length*length+r[i]*r[i]-r[j]*r[j])/(2*length)
        disk2 = r[i]*r[i]-a*a
        if disk2 <= 0:
            continue
        axis = np.array([1., 0., 0.]) if abs(normal[0]) < .8 else np.array([0., 1., 0.])
        u = np.cross(normal, axis); u /= np.sqrt(np.sum(u*u)); v = np.cross(normal, u)
        origin = p[i]+a*normal
        capacity = segments+offsets[i+1]-offsets[i]+4
        polygon, temporary = np.empty((capacity, 2)), np.empty((capacity, 2))
        size = segments
        for t in range(segments):
            theta = 2*np.pi*t/segments
            polygon[t, 0], polygon[t, 1] = np.sqrt(disk2)*np.cos(theta), np.sqrt(disk2)*np.sin(theta)
        for neighbor in range(offsets[i], offsets[i+1]):
            k = ids[neighbor]
            if k == j:
                continue
            delta = p[k]-p[i]
            alpha, beta = np.sum(delta*u), np.sum(delta*v)
            bound = .5*(np.sum(delta*delta)+r[i]*r[i]-r[k]*r[k])-a*np.sum(delta*normal)
            new_size = 0
            if size == 0:
                break
            last = size-1
            for current in range(size):
                va = alpha*polygon[last, 0]+beta*polygon[last, 1]-bound
                vb = alpha*polygon[current, 0]+beta*polygon[current, 1]-bound
                if (va <= 0) != (vb <= 0):
                    fraction = va/(va-vb)
                    temporary[new_size] = polygon[last]+fraction*(polygon[current]-polygon[last])
                    new_size += 1
                if vb <= 0:
                    temporary[new_size] = polygon[current]; new_size += 1
                last = current
            size = new_size
            polygon, temporary = temporary, polygon
        if size < 3:
            continue
        twice, cx, cy, per = 0., 0., 0., 0.
        for t in range(size):
            nxt = (t+1) % size
            cross = polygon[t, 0]*polygon[nxt, 1]-polygon[nxt, 0]*polygon[t, 1]
            twice += cross
            cx += (polygon[t, 0]+polygon[nxt, 0])*cross
            cy += (polygon[t, 1]+polygon[nxt, 1])*cross
            per += np.sqrt((polygon[nxt, 0]-polygon[t, 0])**2+(polygon[nxt, 1]-polygon[t, 1])**2)
        if twice <= 1e-14*disk2:
            continue
        out[edge, 0] = origin+(cx/(3*twice))*u+(cy/(3*twice))*v
        m = min(max(int(np.ceil(per/spacing)), 4), cap-1)
        t, acc = 0, 0.
        seg = np.sqrt((polygon[1 % size, 0]-polygon[0, 0])**2+(polygon[1 % size, 1]-polygon[0, 1])**2)
        for q in range(m):
            target = per*q/m
            while acc+seg < target and t < size-1:
                acc += seg; t += 1
                seg = np.sqrt((polygon[(t+1) % size, 0]-polygon[t, 0])**2+(polygon[(t+1) % size, 1]-polygon[t, 1])**2)
            f = (target-acc)/seg if seg > 0 else 0.
            px = polygon[t, 0]+f*(polygon[(t+1) % size, 0]-polygon[t, 0])
            py = polygon[t, 1]+f*(polygon[(t+1) % size, 1]-polygon[t, 1])
            out[edge, 1+q] = origin+px*u+py*v
        num[edge] = 1+m
    return out, num


class SurfaceCompletion:
    """Volume term and virtual springs of the completed model for one DEM (pacnerf_foam_geometry output) and its foam run."""

    def __init__(self, run, dem, model, sphere_exponent=12, k_search=48, min_quality=.05, verbose=True):
        run, dem = Path(run), Path(dem)
        state = torch.load(run/'model.pt', map_location='cpu', weights_only=True)
        p = state['points'].double().numpy()
        r = torch.nn.functional.softplus(state['radii'].double(), beta=100).numpy()
        cells, geo = np.load(dem/'cells.npz'), np.load(dem/'geometry.npz')
        src, boc = cells['source_index'], geo['body_of_cell']
        if 'ghost_points' in cells.files:      # hull_ghosts.py air sites (foam frame), as in pacnerf_foam_geometry --ghosts
            p = np.concatenate((p, cells['ghost_points'])); r = np.concatenate((r, cells['ghost_radii']))
        # DEM frame = similarity transform of the foam frame (scale, rotation, shift; e.g. the _zup DEMs, Spring-Gaus k)
        P0, P1 = p[src], np.asarray(cells['points'], float)
        m0, m1 = P0.mean(0), P1.mean(0)
        U_, S_, Vt_ = np.linalg.svd((P1-m1).T@(P0-m0))
        D_ = np.eye(3); D_[2, 2] = np.sign(np.linalg.det(U_@Vt_))
        Rm = U_@D_@Vt_
        sc_ = float((S_*np.diag(D_)).sum()/((P0-m0)**2).sum())
        tv = m1-sc_*Rm@m0
        assert np.abs(sc_*P0@Rm.T+tv-P1).max() < 1e-6*max(1., float(np.ptp(P1))), 'foam run does not match the DEM'
        self.frame = dict(scale=sc_, rotation=Rm, shift=tv)
        XB = model.X.detach().cpu().double().numpy()
        assert np.allclose(XB, geo['points'], atol=1e-9), 'model is not built from this DEM'
        N, nB = len(p), len(XB)
        material = np.zeros(N, bool); material[src] = True
        body_of = -np.ones(N, int); body_of[src] = boc
        Xs = sc_*p@Rm.T+tv                                               # foam sites in the DEM frame
        # faces and bounded volumes exactly as pacnerf_foam_geometry
        all_pairs, possible, active_ids = regular_candidates(p, r)
        active = np.zeros(N, bool); active[active_ids] = True
        offs, ids = neighbors(N, all_pairs)
        cand = all_pairs[possible]
        area, cen = clipped_faces(p, r, cand, offs, ids, 64)
        keep = area > 0
        pairs, area, cen = cand[keep], area[keep], cen[keep]
        volume, surface, _ = volume_from_faces(p, r, pairs, area, cen, offs, ids, active, 12)
        volume, area_d = volume*sc_**3, area*sc_**2                      # DEM units
        V0 = np.bincount(boc, volume[src], nB)
        assert np.allclose(V0, geo['masses'], rtol=1e-9), 'body volumes differ from the DEM masses'
        # regular triangulation, all-material tetrahedra collapsed to bodies
        scale = np.ptp(p, axis=0).max(); z = (p-p.mean(0))/scale
        hull = ConvexHull(np.column_stack((z, (z*z).sum(1)-(r/scale)**2)))
        T = hull.simplices[hull.equations[:, -2] < -1e-10]
        T = T[material[T].all(1)]
        Tb = body_of[T]
        distinct = np.array([len(set(row)) == 4 for row in Tb])
        Tb = Tb[distinct]
        P = XB[Tb]
        s6 = np.einsum('ij,ij->i', np.cross(P[:, 1]-P[:, 0], P[:, 2]-P[:, 0]), P[:, 3]-P[:, 0])
        flat = np.abs(s6) <= 1e-9*np.abs(s6).max()
        Tb, s6 = Tb[~flat], s6[~flat]
        Tb[s6 < 0, 2], Tb[s6 < 0, 3] = Tb[s6 < 0, 3].copy(), Tb[s6 < 0, 2].copy()
        vol0 = np.abs(s6)/6
        node_V = np.bincount(Tb.ravel(), np.repeat(vol0, 4), nB)
        has = node_V > 0
        P = XB[Tb]
        D0 = np.stack((P[:, 1]-P[:, 0], P[:, 2]-P[:, 0], P[:, 3]-P[:, 0]), -1)
        Lmx = np.max([np.linalg.norm(P[:, a]-P[:, b], axis=1) for a in range(4) for b in range(a+1, 4)], axis=0)
        qual = 6*np.sqrt(2)*vol0/Lmx**3
        D0inv = np.linalg.inv(D0)
        tree = cKDTree(P.mean(1))

        def locate(Xm):
            """Containing tetrahedron and barycentric coordinates of rest points Xm; outside the mesh, the one among the
            k_search nearest (by centroid) with quality >= min_quality whose coordinates are least negative."""
            _, cand_ = tree.query(Xm, k=min(k_search, len(Tb)))
            lam3 = np.einsum('ckij,ckj->cki', D0inv[cand_], Xm[:, None, :]-P[cand_, 0])
            bary = np.concatenate((1-lam3.sum(-1, keepdims=True), lam3), -1)
            mn = bary.min(-1)
            inside = mn >= -1e-9
            score = np.where(inside, 1.+qual[cand_], np.where(qual[cand_] >= min_quality, mn, -np.inf))
            pick = np.argmax(score, axis=1); rows = np.arange(len(Xm))
            return cand_[rows, pick], bary[rows, pick], inside[rows, pick]
        # virtual springs: material-air faces
        ob = material[pairs].sum(1) == 1
        c_, g_ = pairs[ob, 0].copy(), pairs[ob, 1].copy()
        sw = ~material[c_]; c_[sw], g_[sw] = g_[sw], c_[sw].copy()
        d_ = Xs[g_]-Xs[c_]; L_ = np.linalg.norm(d_, axis=1)
        nf_ = (p[g_]-p[c_])/np.linalg.norm(p[g_]-p[c_], axis=1)[:, None]  # foam frame, as the face centroids
        h_ = sc_*np.einsum('ij,ij->i', cen[ob]-p[c_], nf_)
        vs_body, vs_d, vs_k = [body_of[c_]], [d_], [2*model.mu_ref*area_d[ob]*np.maximum(h_, 0.)/L_**2]
        self.air_faces = dict(body=body_of[c_], area=area_d[ob], normal=d_/L_[:, None])          # material-air faces (DEM frame, outward)
        # exposed sphere patches: int r n n^T dA = r^3 int n n^T dOmega (rays as sphere_volume_terms)
        smp = qmc.Sobol(2, scramble=True, seed=812).random_base2(sphere_exponent)
        zz, ph = 1-2*smp[:, 0], 2*np.pi*smp[:, 1]
        rays = np.column_stack((np.sqrt(1-zz*zz)*np.cos(ph), np.sqrt(1-zz*zz)*np.sin(ph), zz))
        n_patch, V_patch = 0, 0.
        for c in np.flatnonzero(material & (surface > 0)):
            nb = ids[offs[c]:offs[c+1]]; dl = p[nb]-p[c]
            bound = .5*((dl*dl).sum(1)+r[c]**2-r[nb]**2)
            inside = np.all(r[c]*(rays@dl.T) <= bound[None, :], axis=1)
            if not inside.any():
                continue
            R_ = rays[inside]@Rm.T                                           # DEM frame
            rd = sc_*r[c]
            S = rd**3*4*np.pi/len(rays)*R_.T@R_
            V_patch += rd**3*4*np.pi/len(rays)*inside.sum()/3
            ev, evec = np.linalg.eigh(S)
            for s_k, e_k in zip(ev, evec.T):
                if s_k <= 1e-12*ev[-1]:
                    continue
                e_k = e_k if e_k@R_.mean(0) >= 0 else -e_k
                vs_body.append(np.array([body_of[c]])); vs_d.append((rd*e_k)[None]); vs_k.append(np.array([2*model.mu_ref*s_k/rd**2]))
                n_patch += 1
        vs_body, vs_d, vs_k = np.concatenate(vs_body), np.concatenate(vs_d), np.concatenate(vs_k)
        Xm = XB[vs_body]-vs_d                                              # mirror points, inside the material
        vs_tet, vs_bary, vs_in = locate(Xm)
        self.np_data = dict(tets=Tb, node_V=node_V, has=has, V0=V0, vs_body=vs_body, vs_corners=Tb[vs_tet], vs_bary=vs_bary,
                            vs_lever=Xm-XB[vs_body], vs_k=vs_k, D0inv=D0inv, vol0=vol0)
        dev, dt = model.X.device, model.X.dtype
        tt_ = lambda a: torch.as_tensor(np.asarray(a), dtype=dt, device=dev)
        ti_ = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.long, device=dev)
        self.n_body = nB
        self.Tb, self.vol6 = ti_(Tb), tt_(6*vol0)
        self.node_V, self.has = tt_(node_V), torch.as_tensor(has, device=dev)
        self.V0 = tt_(V0)
        self.vs_body, self.vs_k, self.vs_lever = ti_(vs_body), tt_(vs_k), tt_(Xm-XB[vs_body])
        # rotation coupling (off unless rot_coupling > 0): rest edge matrices of the tetrahedra on the body centres
        self.rot_coupling = 0.
        self.D0inv = tt_(np.linalg.inv(D0))
        self.vs_corners, self.vs_bary = self.Tb[ti_(vs_tet)], tt_(vs_bary)
        self.surface_bodies = np.unique(np.r_[body_of[c_], body_of[np.flatnonzero(material & (surface > 0))]])
        self._cell_args = dict(p=p, r=r, material=material, body_of=body_of, Rm=Rm, sc=sc_, tv=tv, XB=XB, locate=locate)
        self._surface_args = dict(pairs=np.column_stack((c_, g_)), c=c_, offs=offs, ids=ids, surface=surface)   # material cell first (its ball bounds the face)
        self.cell_W = None
        old_cov = float(model.node_V0.sum()/4/model.cell_volume.sum()) if getattr(model, 'node_V0', None) is not None else float('nan')
        self.stats = dict(tets=len(Tb), dropped_repeated_body=int((~distinct).sum()), dropped_flat=int(flat.sum()),
                          bodies_without_tet=int((~has).sum()), tet_coverage=float(vol0.sum()/V0.sum()), delaunay_coverage=old_cov,
                          faces=int(ob.sum()), faces_h_nonpositive=int((h_ <= 0).sum()), patch_springs=n_patch,
                          patch_volume_share=float(V_patch/V0.sum()), mirror_outside=int((~vs_in).sum()),
                          mirror_min_bary=float(vs_bary.min()), mirror_tet_quality_min=float(qual[vs_tet].min()),
                          surface_bodies=len(self.surface_bodies))
        if verbose:
            print('surface completion: %s' % {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.stats.items()}, flush=True)

    def body_J(self, x):
        Q = x[self.Tb]
        Vt = (torch.linalg.cross(Q[:, 1]-Q[:, 0], Q[:, 2]-Q[:, 0])*(Q[:, 3]-Q[:, 0])).sum(-1)/6
        Vn = torch.zeros(self.n_body, dtype=x.dtype, device=x.device).index_add(0, self.Tb.reshape(-1), Vt.repeat_interleave(4))
        return torch.where(self.has, Vn/self.node_V.clamp(min=1e-300), torch.ones_like(Vn))

    def body_J_cell(self, x):
        """KB_DERIVATION.md section 12: J_B = sum_t w_Bt det F_t over the body's own cell (the cell weights of the rotation
        constraint) instead of the nodal average over its whole star; bodies without weights keep the nodal J."""
        cb, ct, cw = self.cell_weights()
        Jt = torch.linalg.det(self.tet_F(x))
        Jc = torch.zeros(self.n_body, dtype=x.dtype, device=x.device).index_add(0, cb, cw*Jt[ct])
        return torch.where(self.has_c, Jc, self.body_J(x))

    def volume_energy(self, model, theta, x, q):
        cell = getattr(self, 'cell_volume', False)
        J = self.body_J_cell(x) if cell else self.body_J(x)
        lam = model.k_v*torch.exp(theta[:, 4])
        logJ, _ = model._log_J(J)
        has = (self.has | self.has_c) if cell else self.has
        return torch.where(has, self.V0*.5*lam*logJ**2, torch.zeros_like(J)).sum()

    def strain_completion_energy(self, model, theta, x, q):
        """KB_DERIVATION.md section 12: the virtual springs' outer half-layer stiffness on the body's own cell-average strain,
        sum_s k_s/2 |F_B l_s - R_B l_s|^2 = k_s/2 |(U_B - I) l_s|^2 with R_B = polar(F_B) under the constraint: the same
        energy as the springs under any homogeneous deformation, without the mirror point and the body's rotation over the
        lever (which penalised the local rotation gradient)."""
        F = self.body_F_cell(x)[self.vs_body]
        R = quat_matrix(q)[self.vs_body]
        d = torch.einsum('mij,mj->mi', F-R, self.vs_lever)
        k = self.vs_k*torch.exp(theta[self.vs_body, 5])
        return .5*(k*(d*d).sum(-1)).sum()

    def virtual_energy(self, model, theta, x, q):
        R = quat_matrix(q)
        target = (self.vs_bary[:, :, None]*x[self.vs_corners]).sum(1)
        pred = x[self.vs_body]+torch.einsum('mij,mj->mi', R[self.vs_body], self.vs_lever)
        k = self.vs_k*torch.exp(theta[self.vs_body, 5])
        return .5*(k*((target-pred)**2).sum(-1)).sum()

    def cell_weights(self, samples=4e6, batch=200000, verbose=True, cache=None):
        """ROTATION_CONSTRAINT.md (cell average): w_Bt = volume of body B's cells inside foam tetrahedron t, so that
        F_B = sum_t w_Bt F_t / V_B is the volume mean of the tetrahedra's piecewise-constant F over the body's own material.
        Material outside the tetrahedra takes the F of the tetrahedron locate() picks for it (as the mirror points).
        Regular grid of about `samples` points in the material; a point is in foam cell c when c is its power-nearest
        site and inside c's ball (power distance <= 0). Cached; also stored in np_data (cw_body, cw_tet, cw_w)."""
        if self.cell_W is not None:
            return self.cell_W
        if cache is not None and Path(cache).exists():
            d_ = np.load(cache)
            assert int(d_['n_body']) == len(self.V0) and int(d_['n_tet']) == len(self.Tb) and float(d_['samples']) == samples, 'stale cell-weight cache'
            return self._set_cell_W(d_['cw_body'], d_['cw_tet'], d_['cw_w'], d_['cw_has'], dict(cached=str(cache)), verbose)
        a = self._cell_args
        p, r, material, body_of, Rm, sc_, tv, XB, locate = (a[k] for k in ('p', 'r', 'material', 'body_of', 'Rm', 'sc', 'tv', 'XB', 'locate'))
        V0 = self.V0.detach().cpu().double().numpy()
        h = (V0.sum()/samples)**(1/3)
        rmax = r.max()
        lift = cKDTree(np.column_stack((p, np.sqrt(rmax**2-r**2))))
        src = np.flatnonzero(material)
        lo = (sc_*p[src]@Rm.T+tv).min(0)-sc_*r[src].max(); hi = (sc_*p[src]@Rm.T+tv).max(0)+sc_*r[src].max()
        axes = [np.arange(lo[k]+h/2, hi[k], h) for k in range(3)]
        n_tot = np.prod([len(ax) for ax in axes])
        eb, et = [], []
        for s0 in range(0, n_tot, batch):
            idx = np.arange(s0, min(s0+batch, n_tot))
            i0, rem = np.divmod(idx, len(axes[1])*len(axes[2])); i1, i2 = np.divmod(rem, len(axes[2]))
            X = np.column_stack((axes[0][i0], axes[1][i1], axes[2][i2]))
            P = (X-tv)@Rm/sc_                                                 # foam frame
            _, c = lift.query(np.column_stack((P, np.zeros(len(P)))))
            keep = material[c] & (((P-p[c])**2).sum(1) <= r[c]**2)
            if not keep.any():
                continue
            t_, _, _ = locate(X[keep])
            eb.append(body_of[c[keep]]); et.append(t_)
        eb, et = np.concatenate(eb), np.concatenate(et)
        key = eb.astype(np.int64)*len(self.Tb)+et
        uk, cnt = np.unique(key, return_counts=True)
        cw_body, cw_tet = (uk//len(self.Tb)).astype(np.int64), (uk % len(self.Tb)).astype(np.int64)
        vol = cnt*h**3
        Vs = np.bincount(cw_body, vol, len(V0))
        cw_w = vol/Vs[cw_body]                                                  # normalised per body
        rel = Vs/np.maximum(V0, 1e-300)-1
        stats = dict(spacing_mm=1e3*h, entries=len(cw_w), bodies_with_weights=int((Vs > 0).sum()),
                     sampled_vs_true_volume_median=float(np.median(np.abs(rel))), max=float(np.abs(rel).max()))
        if cache is not None:
            Path(cache).parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, cw_body=cw_body, cw_tet=cw_tet, cw_w=cw_w, cw_has=Vs > 0, n_body=len(V0), n_tet=len(self.Tb), samples=samples)
        return self._set_cell_W(cw_body, cw_tet, cw_w, Vs > 0, stats, verbose)

    def _set_cell_W(self, cw_body, cw_tet, cw_w, cw_has, stats, verbose):
        self.cell_stats = stats
        if verbose:
            print('cell average: %s' % {k: (round(v, 4) if isinstance(v, float) else v) for k, v in stats.items()}, flush=True)
        dev, dt = self.V0.device, self.V0.dtype
        self.cell_W = (torch.as_tensor(cw_body, device=dev), torch.as_tensor(cw_tet, device=dev), torch.as_tensor(cw_w, dtype=dt, device=dev))
        self.has_c = torch.as_tensor(np.asarray(cw_has, bool), device=dev)
        self.np_data.update(cw_body=np.asarray(cw_body), cw_tet=np.asarray(cw_tet), cw_w=np.asarray(cw_w), cw_has=np.asarray(cw_has, bool))
        return self.cell_W

    def surface_points(self, spacing=.005, ray_exponent=8, verbose=True):
        """ROTATION_CONSTRAINT.md surface contact: points on the foam's outer surface (material-air faces: centroid +
        boundary at most `spacing` apart; exposed sphere patches: rays of a 2^ray_exponent Sobol set), each on a body B.
        It moves with the body's cell-average deformation, x_s = x_B + F_B (X_s - X_B), which is linear in the body
        positions: x_s = sum_c a_c x_c with constant a_c (sum 1; exact for any affine motion). Stored in np_data as the
        point -> (body, a) CSR (sp_*) and its transpose (bs_*). Bodies without cell weights carry no points."""
        if getattr(self, 'sp', None) is not None:
            return self.sp
        a_, s_ = self._cell_args, self._surface_args
        p, r, body_of, Rm, sc_, tv, XB = (a_[k] for k in ('p', 'r', 'body_of', 'Rm', 'sc', 'tv', 'XB'))
        pts, num = face_boundary_points(p, r, s_['pairs'].astype(np.int64), s_['offs'], s_['ids'], 64, spacing/sc_, 24)
        keep = np.arange(pts.shape[1])[None, :] < num[:, None]
        Pf = pts[keep]; cf = np.repeat(s_['c'], num)
        smp = qmc.Sobol(2, scramble=True, seed=813).random_base2(ray_exponent)
        zz, ph = 1-2*smp[:, 0], 2*np.pi*smp[:, 1]
        rays = np.column_stack((np.sqrt(1-zz*zz)*np.cos(ph), np.sqrt(1-zz*zz)*np.sin(ph), zz))
        Pp, cp_ = [], []
        offs, ids = s_['offs'], s_['ids']
        for c in np.flatnonzero(a_['material'] & (s_['surface'] > 0)):
            nb = ids[offs[c]:offs[c+1]]; dl = p[nb]-p[c]
            bound = .5*((dl*dl).sum(1)+r[c]**2-r[nb]**2)
            inside = np.all(r[c]*(rays@dl.T) <= bound[None, :], axis=1)
            if inside.any():
                Pp.append(p[c]+r[c]*rays[inside]); cp_.append(np.full(int(inside.sum()), c))
        Pall = np.concatenate([Pf]+Pp); call = np.concatenate([cf]+cp_)
        Xs = sc_*Pall@Rm.T+tv                                                    # DEM frame
        owner = body_of[call]
        cb, ct, cw = (t_.cpu().numpy() for t_ in self.cell_weights())
        hasc = self.has_c.cpu().numpy()
        ok = hasc[owner]
        Xs, owner = Xs[ok], owner[ok]
        lever = Xs-XB[owner]
        D0i = self.D0inv.detach().cpu().double().numpy(); Tb = self.Tb.cpu().numpy()
        g = np.concatenate((-D0i.sum(1, keepdims=True), D0i), 1)                # (nt, 4, 3) grad lambda_k
        o_ = np.argsort(cb, kind='stable'); cb, ct, cw = cb[o_], ct[o_], cw[o_]
        boff = np.concatenate(([0], np.cumsum(np.bincount(cb, minlength=self.n_body))))
        rows, cols, vals = [np.arange(len(Xs))], [owner], [np.ones(len(Xs))]
        for s_i in range(len(Xs)):
            b = owner[s_i]; j0, j1 = boff[b], boff[b+1]
            coef = cw[j0:j1, None]*(g[ct[j0:j1]]@lever[s_i])                     # (entries, 4)
            rows.append(np.full(coef.size, s_i)); cols.append(Tb[ct[j0:j1]].ravel()); vals.append(coef.ravel())
        rows, cols, vals = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
        key = rows*self.n_body+cols
        uk, inv = np.unique(key, return_inverse=True)
        A = np.zeros(len(uk)); np.add.at(A, inv, vals)
        sp_point, sp_body = uk//self.n_body, uk % self.n_body
        nzero = np.abs(A) > 1e-12
        sp_point, sp_body, A = sp_point[nzero], sp_body[nzero], A[nzero]
        sp_off = np.concatenate(([0], np.cumsum(np.bincount(sp_point, minlength=len(Xs)))))
        o_ = np.argsort(sp_body, kind='stable')
        bs_off = np.concatenate(([0], np.cumsum(np.bincount(sp_body, minlength=self.n_body))))
        err = np.abs(np.bincount(sp_point, A*XB[sp_body][:, 0], len(Xs))-Xs[:, 0]).max()
        self.sp = dict(sp_off=sp_off, sp_body=sp_body, sp_a=A, sp_owner=owner, bs_off=bs_off, bs_point=sp_point[o_], bs_a=A[o_], sp_X=Xs)
        self.np_data.update({k_: v_ for k_, v_ in self.sp.items() if k_ != 'sp_X'})
        if verbose:
            print('surface points: %d (faces %d, patches %d; dropped %d on bodies without cell weights), %.1f bodies per point, rest position '
                  'from the weights vs X_s %.1e m, sum of weights - 1 max %.1e' % (len(Xs), len(Pf), len(Pall)-len(Pf), int((~ok).sum()),
                  len(A)/len(Xs), err, np.abs(np.bincount(sp_point, A, len(Xs))-1).max()), flush=True)
        return self.sp

    def surface_x(self, x):
        """Surface point positions sum_c a_c x_c (torch)."""
        sp = self.surface_points(verbose=False)
        pt = torch.as_tensor(sp['bs_point'], device=x.device); bd = torch.as_tensor(np.sort(sp['sp_body'], kind='stable'), device=x.device)
        a = torch.as_tensor(sp['bs_a'], dtype=x.dtype, device=x.device)
        return torch.zeros(len(sp['sp_owner']), 3, dtype=x.dtype, device=x.device).index_add(0, pt, a[:, None]*x[bd])

    def tet_F(self, x):
        Q = x[self.Tb]
        Dx = torch.stack((Q[:, 1]-Q[:, 0], Q[:, 2]-Q[:, 0], Q[:, 3]-Q[:, 0]), -1)
        return Dx@self.D0inv

    def body_F_cell(self, x):
        """Cell-average F_B (cell_weights); bodies without weights: identity."""
        cb, ct, cw = self.cell_weights()
        S = torch.zeros(self.n_body, 3, 3, dtype=x.dtype, device=x.device).index_add(0, cb, cw[:, None, None]*self.tet_F(x)[ct])
        eye = torch.eye(3, dtype=x.dtype, device=x.device).expand(self.n_body, 3, 3)
        return torch.where(self.has_c[:, None, None], S, eye)

    def slaved_rot_invmass(self, mass):
        """Bound on the inverse rotational inertia of each slaved rotation (cell average, at rest): the rotation is
        theta_B = sum_c [v_Bc]x x_c with v_Bc = 1/2 sum w_Bt grad(lambda_k) over the corners k of t at c, so a torque
        tau on it moves the corners by M^-1 sum_c [v_Bc]x^T tau; |dtheta| <= sum_c |v_Bc|^2 / m_c |tau|."""
        cb, ct, cw = (a_.cpu().numpy() for a_ in self.cell_weights())
        D0i = self.D0inv.detach().cpu().double().numpy(); Tb = self.Tb.cpu().numpy()
        g = np.concatenate((-D0i.sum(1, keepdims=True), D0i), 1)                # (nt, 4, 3): grad lambda_k = rows of D0^-1
        key = (cb[:, None]*self.n_body+Tb[ct]).ravel()
        vec = (.5*cw[:, None, None]*g[ct]).reshape(-1, 3)
        uk, inv = np.unique(key, return_inverse=True)
        V = np.zeros((len(uk), 3)); np.add.at(V, inv, vec)
        m_ = np.asarray(mass.detach().cpu().double())
        out = np.bincount(uk//self.n_body, (V*V).sum(1)/m_[uk % self.n_body], self.n_body)
        return torch.as_tensor(out, dtype=self.V0.dtype, device=self.V0.device)

    def body_F(self, x):
        """Per body the volume-weighted mean of the affine deformation gradients F_t = D(x) D(X)^-1 of the foam
        tetrahedra at it (bodies without a tetrahedron: identity)."""
        Q = x[self.Tb]
        Dx = torch.stack((Q[:, 1]-Q[:, 0], Q[:, 2]-Q[:, 0], Q[:, 3]-Q[:, 0]), -1)
        Ft = Dx@self.D0inv
        w = (self.vol6/6)[:, None, None]*Ft
        S = torch.zeros(self.n_body, 3, 3, dtype=x.dtype, device=x.device).index_add(0, self.Tb.reshape(-1), w.repeat_interleave(4, 0))
        eye = torch.eye(3, dtype=x.dtype, device=x.device).expand(self.n_body, 3, 3)
        return torch.where(self.has[:, None, None], S/self.node_V.clamp(min=1e-300)[:, None, None], eye)

    def rotation_energy(self, model, theta, x, q):
        """Micropolar coupling (COSSERAT_FOAM.md: mu_c |skew(R^T F)|^2, mu_c = rot_coupling x mu) with F the body's mean
        foam-tetrahedron deformation gradient: zero when the body turns with the surrounding material (any affine motion,
        rigid rotation); penalises the body spinning on its own."""
        U = quat_matrix(q).transpose(1, 2)@self.body_F(x)
        Sk = .5*(U-U.transpose(1, 2))
        mu_c = self.rot_coupling*model.mu_ref*torch.exp(theta[:, 5])
        return torch.where(self.has, self.V0*mu_c*(Sk*Sk).sum((1, 2)), torch.zeros_like(self.V0)).sum()

    def polar_R(self, x):
        """R_B = polar(F_B) (ROTATION_CONSTRAINT.md, F_B the cell average body_F_cell); also U_B. Bodies without
        weights: identity."""
        F = self.body_F_cell(x)
        Uu, Sg, Vh = torch.linalg.svd(F)
        d = torch.sign(torch.linalg.det(Uu@Vh))
        Dg = torch.ones_like(Sg); Dg[:, 2] = d
        R = Uu@torch.diag_embed(Dg)@Vh
        U = Vh.transpose(1, 2)@torch.diag_embed(Sg*Dg)@Vh
        return R, U

    def constrained_energy(self, model, theta, x):
        """Energy with every body's rotation slaved to polar(F_B(x)) (positions only)."""
        R, _ = self.polar_R(x)
        return self.energy(model, theta, x, matrix_quat(R))

    def energy(self, model, theta, x, q):
        """Completed elastic energy: the model's bonds (its own volume term removed) + new volume term + virtual springs."""
        base = model.energy(theta, x, q)
        if getattr(model, 'body_law', None) == 'nodalvol':
            base = base-model._nodalvol(theta, x)[0]
        comp = self.strain_completion_energy if getattr(self, 'strain_completion', False) else self.virtual_energy
        out = base+self.volume_energy(model, theta, x, q)+comp(model, theta, x, q)
        return out+self.rotation_energy(model, theta, x, q) if self.rot_coupling > 0 else out


def replicate(data, K, n0):
    """Completion data (SurfaceCompletion.np_data) of K disjoint copies of an n0-body model (springgaus_train batch_model)."""
    out = dict(tets=np.concatenate([data['tets']+k*n0 for k in range(K)]),
               vs_body=np.concatenate([data['vs_body']+k*n0 for k in range(K)]),
               vs_corners=np.concatenate([data['vs_corners']+k*n0 for k in range(K)]))
    for key in ('node_V', 'has', 'V0', 'vs_bary', 'vs_lever', 'vs_k', 'D0inv', 'vol0'):
        out[key] = np.concatenate([data[key]]*K)
    for key in [k_ for k_ in data if k_.startswith('cl_')]:     # clean model (prototypes/kb_derivation/clean_model.np_data): per bond / body
        out[key] = np.concatenate([data[key]]*K)
    for key in [k_ for k_ in data if k_.startswith('cm_')]:     # consistent-mass scalars (eigenvalue bounds): the same for every copy
        out[key] = data[key]
    if 'cw_body' in data:
        nt0 = len(data['tets'])
        out.update(cw_body=np.concatenate([data['cw_body']+k*n0 for k in range(K)]), cw_tet=np.concatenate([data['cw_tet']+k*nt0 for k in range(K)]),
                   cw_w=np.concatenate([data['cw_w']]*K), cw_has=np.concatenate([data['cw_has']]*K))
    if 'sp_off' in data:                # surface contact points: point -> body CSR and its transpose, per copy
        np0, nnz = len(data['sp_owner']), len(data['sp_body'])
        csr = lambda o_: np.concatenate([np.asarray(o_[:-1])+k*nnz for k in range(K)]+[[K*nnz]])
        out.update(sp_off=csr(data['sp_off']), sp_body=np.concatenate([data['sp_body']+k*n0 for k in range(K)]), sp_a=np.concatenate([data['sp_a']]*K),
                   sp_owner=np.concatenate([data['sp_owner']+k*n0 for k in range(K)]), bs_off=csr(data['bs_off']),
                   bs_point=np.concatenate([data['bs_point']+k*np0 for k in range(K)]), bs_a=np.concatenate([data['bs_a']]*K))
    return out


def polar_newton(F, iterations=12):
    """Rotation of the polar decomposition by Newton's iteration R <- (R + R^-T)/2: smooth to any order, also at F = I
    where the SVD's derivative is singular (used for the Hessian of the constrained energy)."""
    R = F
    for _ in range(iterations):
        R = .5*(R+torch.linalg.inv(R).transpose(1, 2))
    return R


def frequency_ratio(model, sc, iterations=40, seed=0, constrained=False):
    """Highest angular frequency at rest of the completed energy over the model's own (power iteration on M^-1/2 H
    M^-1/2, H by double backward); multiplies the substep count of the model's stable_dt. constrained: rotations of
    bodies with a tetrahedron slaved to polar(F_B) (ROTATION_CONSTRAINT.md), their rotation entries carry no energy."""
    th = torch.zeros(model.n, 7, dtype=model.X.dtype, device=model.X.device)
    x0, q0, _, _ = model.initial_state()
    mvec = torch.cat((model.mass[:, None].expand(-1, 3), model.inertia[:, None].expand(-1, 3)), 1)
    from experiments.foam_dem_energy import quat_mul, quat_small
    out = []
    def slaved(x, q):
        return sc.energy(model, th, x, torch.where(sc.has_c[:, None], matrix_quat(polar_newton(sc.body_F_cell(x))), q))
    for en in (lambda x, q: model.energy(th, x, q), slaved if constrained else (lambda x, q: sc.energy(model, th, x, q))):
        gen = torch.Generator(device='cpu').manual_seed(seed)
        y = torch.randn(model.n, 6, generator=gen, dtype=torch.float64).to(model.X.device, model.X.dtype)
        val = 0.
        for _ in range(iterations):
            y = y/torch.linalg.vector_norm(y)
            z = torch.zeros(model.n, 6, dtype=model.X.dtype, device=model.X.device, requires_grad=True)
            E_ = en(x0+z[:, :3], quat_mul(quat_small(z[:, 3:]), q0))
            gz, = torch.autograd.grad(E_, z, create_graph=True)
            Hy, = torch.autograd.grad((gz*(y/mvec.sqrt())).sum(), z)
            zz = Hy/mvec.sqrt(); val = float((y*zz).sum()); y = zz.detach()
        out.append(np.sqrt(max(val, 0.)))
    return out[1]/out[0], out
