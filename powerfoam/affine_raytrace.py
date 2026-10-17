"""Forward-only shared-metric bounded Power Foam reference renderer.

World-space ellipsoid/plane intersections, canonical texture, world-length
extinction. All cells share one invertible, orientation-preserving affine map.
This correctness prototype scans cells in painter order; it is not a BVH or
adjacency-walking acceleration benchmark. No displacement or directional color.
"""

import numpy as np
import warp as wp


@wp.kernel(enable_backward=False)
def shared_metric_kernel(
    ray_origins: wp.array(dtype=wp.vec3),
    ray_directions: wp.array(dtype=wp.vec3),
    metric: wp.mat33,
    inverse_f: wp.mat33,
    translation: wp.vec3,
    centers: wp.array(dtype=wp.vec3),
    radii: wp.array(dtype=float),
    normals: wp.array(dtype=wp.vec3),
    densities: wp.array(dtype=float),
    texture_sites: wp.array2d(dtype=wp.vec3),
    texture_colors: wp.array2d(dtype=wp.vec3),
    face_normals: wp.array(dtype=wp.vec3),
    face_offsets: wp.array(dtype=float),
    adjacency_offsets: wp.array(dtype=int),
    cell_order: wp.array2d(dtype=int),
    per_ray_order: int,
    num_cells: int,
    num_texels: int,
    rgb_out: wp.array(dtype=wp.vec3),
    alpha_out: wp.array(dtype=float),
    first_depth_out: wp.array(dtype=float),
    first_cell_out: wp.array(dtype=int),
    normal_out: wp.array(dtype=wp.vec3),
):
    ray = wp.tid()
    origin, direction = ray_origins[ray], ray_directions[ray]
    qa = wp.dot(direction, metric * direction)
    transmittance = float(1.0)
    rgb, normal_sum = wp.vec3(0.0), wp.vec3(0.0)
    first_depth, first_cell = float(0.0), int(-1)
    order_row = int(0)
    if per_ray_order != 0:
        order_row = ray
    for order_index in range(num_cells):
        if transmittance < 1.0e-6:
            break
        i = cell_order[order_row, order_index]
        center = centers[i]
        oc = origin - center
        # Closest point in metric M avoids subtracting two large discriminant
        # terms for distant cameras. Directions are unit WORLD vectors.
        closest_t = -wp.dot(direction, metric * oc) / qa
        closest_offset = oc + closest_t * direction
        residual = radii[i] * radii[i] - wp.dot(closest_offset, metric * closest_offset)
        if residual <= 0.0:
            continue
        half_length = wp.sqrt(residual / qa)
        near = wp.max(0.0, closest_t - half_length)
        far = closest_t + half_length
        if far <= near:
            continue
        # The shared quadratic term cancels: each competitor clips a plane.
        for edge in range(adjacency_offsets[i], adjacency_offsets[i + 1]):
            n = face_normals[edge]
            numerator = face_offsets[edge] - wp.dot(n, origin)
            denominator = wp.dot(n, direction)
            if wp.abs(denominator) < 1.0e-10:
                if numerator < 0.0:
                    far = near
            else:
                t_face = numerator / denominator
                if denominator > 0.0:
                    far = wp.min(far, t_face)
                else:
                    near = wp.max(near, t_face)
            if far <= near:
                break
        if far <= near:
            continue
        # The occupied half of the zero-displacement dipole is n.(x-p)<=0.
        n = normals[i]
        numerator = wp.dot(n, center - origin)
        denominator = wp.dot(n, direction)
        if wp.abs(denominator) < 1.0e-10:
            if numerator < 0.0:
                continue
        else:
            surface_t = numerator / denominator
            if denominator > 0.0:
                far = wp.min(far, surface_t)
            else:
                near = wp.max(near, surface_t)
        if far <= near:
            continue
        canonical_query = inverse_f * (origin + near * direction - translation)
        color_sum, weight_sum = wp.vec3(0.0), float(0.0)
        for k in range(num_texels):
            displacement = canonical_query - texture_sites[i, k]
            weight = wp.exp(-10.0 * wp.dot(displacement, displacement) / (radii[i] * radii[i]))
            color_sum += weight * texture_colors[i, k]
            weight_sum += weight
        color = color_sum / wp.max(weight_sum, 1.0e-20)
        alpha = 1.0 - wp.exp(-densities[i] * (far - near))
        contribution = transmittance * alpha
        if first_cell < 0 and alpha > 0.0:
            first_cell, first_depth = i, near
        rgb += contribution * color
        normal_sum += contribution * n
        transmittance *= 1.0 - alpha
    rgb_out[ray] = rgb
    alpha_out[ray] = 1.0 - transmittance
    first_depth_out[ray] = first_depth
    first_cell_out[ray] = first_cell
    normal_out[ray] = normal_sum


class SharedAffineRayTracer:
    def __init__(self, centers, radii, normals, texel_sites, texel_colors, density=1000., device="cuda:0"):
        self.p0 = np.asarray(centers, dtype=np.float64)
        self.r0 = np.asarray(radii, dtype=np.float64)
        self.n0 = np.asarray(normals, dtype=np.float64)
        self.sites = np.asarray(texel_sites, dtype=np.float64)
        self.colors = np.asarray(texel_colors, dtype=np.float64)
        n = len(self.p0)
        self.density = np.broadcast_to(np.asarray(density, dtype=np.float64), (n,)).copy()
        if self.p0.shape != (n, 3) or self.r0.shape != (n,) or self.n0.shape != (n, 3):
            raise ValueError("Invalid cell attribute shapes")
        if self.sites.ndim != 3 or self.sites.shape != self.colors.shape or self.sites.shape[0] != n or self.sites.shape[2] != 3:
            raise ValueError("Texture sites/colors must have shape [cells, texels, 3]")
        attributes = (self.p0, self.r0, self.n0, self.sites, self.colors, self.density)
        if not all(np.isfinite(a).all() for a in attributes) or np.any(self.r0 <= 0) or np.any(self.density < 0):
            raise ValueError("Attributes must be finite; radii positive and densities nonnegative")
        if np.any(np.linalg.norm(self.n0, axis=1) == 0):
            raise ValueError("Normals must be nonzero")
        wp.init()
        self.device = device
        self.radii_wp = self._array(self.r0, float)
        self.density_wp = self._array(self.density, float)
        self.sites_wp = self._array(self.sites, wp.vec3)
        self.colors_wp = self._array(self.colors, wp.vec3)
        # Invertible GLOBAL affine maps preserve pairwise support intersection.
        distance = np.linalg.norm(self.p0[:, None] - self.p0[None], axis=-1)
        overlap = distance < self.r0[:, None] + self.r0[None]
        np.fill_diagonal(overlap, False)
        self.edge_i, self.edge_j = np.nonzero(overlap)
        offsets = np.r_[0, np.cumsum(overlap.sum(1))].astype(np.int32)
        self.offsets_wp = self._array(offsets, int)

    def _array(self, values, dtype):
        typ = np.int32 if dtype is int else np.float32
        return wp.array(np.ascontiguousarray(values, dtype=typ), dtype=dtype, device=self.device)

    def render(self, origins, directions, deformation, translation=None):
        f = np.asarray(deformation, dtype=np.float64)
        t = np.zeros(3) if translation is None else np.asarray(translation, dtype=np.float64)
        if f.shape != (3, 3) or t.shape != (3,) or not np.isfinite(f).all() or not np.isfinite(t).all():
            raise ValueError("Expected finite F[3,3] and translation[3]")
        if np.linalg.det(f) <= 0 or np.linalg.cond(f) > 1e5:
            raise ValueError("F must preserve orientation and be sufficiently well-conditioned")
        origins = np.asarray(origins, dtype=np.float64)
        directions = np.asarray(directions, dtype=np.float64)
        if origins.shape != directions.shape or origins.ndim != 2 or origins.shape[1] != 3:
            raise ValueError("Origins/directions must have shape [rays,3]")
        if not np.isfinite(origins).all() or not np.isfinite(directions).all():
            raise ValueError("Rays must be finite")
        if not np.allclose(np.linalg.norm(directions, axis=1), 1., atol=1e-7, rtol=1e-7):
            raise ValueError("Ray directions must be unit world vectors")
        inv = np.linalg.inv(f)
        metric = inv.T @ inv
        centers = self.p0 @ f.T + t
        normals = self.n0 @ inv
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        i, j = self.edge_i, self.edge_j
        face_n = (centers[j] - centers[i]) @ metric
        face_b = np.sum(face_n * (centers[i] + centers[j]) * .5, axis=1) + .5*(self.r0[i]**2-self.r0[j]**2)
        common_origin = np.all(origins == origins[:1])
        sort_origins = origins[:1] if common_origin else origins
        delta = sort_origins[:, None] - centers
        order = np.argsort(np.einsum("nki,ij,nkj->nk", delta, metric, delta) - self.r0**2, axis=1).astype(np.int32)
        count = len(origins)
        out = [wp.zeros(count, dtype=dtype, device=self.device) for dtype in (wp.vec3, float, float, int, wp.vec3)]
        wp.launch(shared_metric_kernel, dim=count, inputs=[
            self._array(origins, wp.vec3), self._array(directions, wp.vec3),
            wp.mat33(*metric.ravel().astype(float)), wp.mat33(*inv.ravel().astype(float)), wp.vec3(*t),
            self._array(centers, wp.vec3), self.radii_wp, self._array(normals, wp.vec3), self.density_wp,
            self.sites_wp, self.colors_wp, self._array(face_n, wp.vec3), self._array(face_b, float),
            self.offsets_wp, self._array(order, int), int(not common_origin), len(centers), self.sites.shape[1],
            *out], device=self.device)
        wp.synchronize_device(self.device)
        return {key: value.numpy() for key, value in zip(("rgb", "alpha", "depth", "owner", "normal"), out)}
