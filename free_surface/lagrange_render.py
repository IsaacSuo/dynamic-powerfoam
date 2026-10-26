"""Power Foam rendered with its cells carried by the material (2026-10-25; forward only). Power Foam's renderer cuts
every cell anew from the current sites and radii (ball, power faces with the overlapping balls, the textured plane), so
moving the sites (PowerSim's update) re-partitions space: exact only for similarity motions; under a squash the walls
leave the material and the partly filled cells trade space (render_drop.py: the window at the flattened arm). Here each
cell keeps its rest geometry and is mapped by its own affine map x = x_i + F_i (X - X_i):
  ball and textured plane   the ray is taken into the cell's rest frame, X(t) = X_i + G_i (o - x_i) + t G_i d with
                            G_i = F_i^-1 (same t: the world path length), and cut against the rest ball, the rest plane
                            and the rest texels, so the ball becomes an ellipsoid and the plane and texture follow F_i;
  walls                     the rest power face of every rest adjacency entry (i, j), {X : n.X = c}, n = X_j - X_i, is
                            carried by the pair's mean map (its rest point taken by both cells' maps and averaged, the
                            normal by (F_i + F_j)^-T / 2): one plane shared by the two cells, so no gap and no overlap.
At F = I, x = X every value equals Power Foam's (the walls in float32 here, Power Foam keeps the neighbour offsets in
float16). The adjacency is the rest one (Power Foam's Cech complex at rest). Visibility and depth order: Power Foam's
tile kernels on the bounding spheres (x_i, r_i |F_i|_2), sorted by their power distance from the eye. The view-dependent
colour: the radiance axes turned by R_i = polar(F_i), evaluated at the deformed texel sites."""
import torch
import warp as wp

from powerfoam.camera import WarpCamera, get_ray_dir
from powerfoam.rendering_math import ray_sphere_intersect
from powerfoam.texture import plane_intersection_fwd

TILE_WIDTH = 8
TILE_SIZE = TILE_WIDTH*TILE_WIDTH


def make_kernel(num_texel_sites, is_pinhole):
    @wp.kernel(enable_backward=False)
    def lagrange_forward(
        camera: WarpCamera,
        cur: wp.array(dtype=wp.vec3f),
        rest: wp.array(dtype=wp.vec3f),
        rad: wp.array(dtype=wp.float32),
        ginv: wp.array(dtype=wp.mat33f),
        nsigma: wp.array(dtype=wp.vec4f),
        texel_sites: wp.array2d(dtype=wp.vec3f),
        texel_rgbh: wp.array2d(dtype=wp.vec4f),
        adj_off: wp.array(dtype=wp.int32),
        planes: wp.array(dtype=wp.vec4f),
        tile_prims: wp.array(dtype=wp.int32),
        tile_offsets: wp.array(dtype=wp.int64),
        threshold: float,
        color_out: wp.array2d(dtype=wp.vec3f),
        log_t_out: wp.array2d(dtype=wp.float32),
    ):
        tid = wp.tid()
        tile_idx = tid//TILE_SIZE
        tiles_h = 1+(camera.height-1)//TILE_WIDTH
        tile_i = tile_idx % tiles_h
        tile_j = tile_idx//tiles_h
        idx_in_tile = tid % TILE_SIZE
        pix_i = tile_i*TILE_WIDTH+idx_in_tile % TILE_WIDTH
        pix_j = tile_j*TILE_WIDTH+idx_in_tile//TILE_WIDTH
        oob = pix_i >= camera.height or pix_j >= camera.width
        pix_i = wp.min(pix_i, camera.height-1)
        pix_j = wp.min(pix_j, camera.width-1)
        if is_pinhole:
            ray_d = get_ray_dir(camera, float(pix_i), float(pix_j))
            ray_d = ray_d/wp.length(ray_d)
            ray_o = camera.eye
        else:
            ray_o = wp.vec3f(camera.ray_maps[pix_i, pix_j, 0], camera.ray_maps[pix_i, pix_j, 1], camera.ray_maps[pix_i, pix_j, 2])
            ray_d = wp.vec3f(camera.ray_maps[pix_i, pix_j, 3], camera.ray_maps[pix_i, pix_j, 4], camera.ray_maps[pix_i, pix_j, 5])
            ray_d = ray_d/wp.length(ray_d)

        rgb = wp.vec3f(0.0, 0.0, 0.0)
        log_t = float(0.0)
        start = tile_offsets[tile_idx]
        total = wp.int32(tile_offsets[tile_idx+1]-start)
        for k in range(total):
            p = tile_prims[start+wp.int64(k)]
            trans = wp.exp(log_t)
            max_trans = wp.tile_max(wp.tile(trans))[0]
            if max_trans < threshold:
                break
            g = ginv[p]
            xr = rest[p]
            r = rad[p]
            eo = xr+g@(ray_o-cur[p])                           # the ray in the cell's rest frame (same t)
            ed = g@ray_d
            hit, t_near, t_far = ray_sphere_intersect(eo, ed, xr, r)
            any_hit = wp.tile_sum(wp.tile(wp.int32(hit)))[0]
            if any_hit == 0:
                continue
            for a in range(adj_off[p], adj_off[p+1]):
                pl = planes[a]
                n = wp.vec3f(pl[0], pl[1], pl[2])
                dp = wp.dot(ray_d, n)
                t_face = (pl[3]-wp.dot(ray_o, n))/dp
                if hit:
                    t_far = wp.min(t_face, t_far) if dp >= 0.0 else t_far
                    t_near = wp.max(t_face, t_near) if dp < 0.0 else t_near
            ns = nsigma[p]
            _, _, height, _, t_surf, dps, color, _ = plane_intersection_fwd(
                eo, ed, t_near, xr, wp.vec3f(ns[0], ns[1], ns[2]), r, (not oob) and hit, texel_sites[p], texel_rgbh[p], num_texel_sites)
            t_far = wp.min(t_surf, t_far) if dps >= 0.0 else t_far
            t_near = wp.max(t_surf, t_near) if dps < 0.0 else t_near
            dt = t_far-t_near
            if hit and dt > 0.0:
                dlt = -ns[3]*dt
                alpha = 1.0-wp.exp(dlt)
                rgb += color*alpha*wp.exp(log_t)
                log_t += dlt
        if not oob:
            color_out[pix_i, pix_j] = rgb
            log_t_out[pix_i, pix_j] = log_t

    return lagrange_forward


class LagrangeRender:
    """the model's rest state (call at rest, with its rest adjacency built); render(cam, x, F) with the cells' current
    sites x (N, 3) and maps F (N, 3, 3)"""

    def __init__(self, model):
        self.m = model
        with torch.no_grad():
            self.X = model.points.detach().float().clone()
            self.r = model.get_radii().detach().float().clone()
            self.normal = model.get_normals().detach().float().clone()
            self.sigma = model.get_density().detach().float().clone()
            t_, b_ = model.get_tangents()
            off = model.texel_sites.detach()*self.r[:, None, None]
            self.texel_off = (off[..., 0:1]*t_[:, None, :]+off[..., 1:2]*b_[:, None, :]).float()     # rest offsets from the site
            self.texel_rest = (self.X[:, None]+self.texel_off).contiguous()
            self.height = (model.texel_height.detach()*self.r[:, None]).float()
            self.axis0 = model.texel_sv_axis.detach().clone()
            self.adj = model.adjacency.long().clone()
            self.adj_off = model.adjacency_offsets.int().clone()
            n = len(self.X)
            self.ai = torch.repeat_interleave(torch.arange(n, device=self.X.device), (self.adj_off[1:]-self.adj_off[:-1]).long())
            Xi, Xj = self.X[self.ai], self.X[self.adj]
            nn = Xj-Xi
            c = .5*((Xj*Xj).sum(1)-(Xi*Xi).sum(1)+self.r[self.ai]**2-self.r[self.adj]**2)
            self.wall_n = nn
            self.wall_c = Xi+((c-(nn*Xi).sum(1))/(nn*nn).sum(1))[:, None]*nn      # a point of each rest power face
        self.kernel = make_kernel(int(model.args.num_texel_sites), bool(model.args.is_pinhole))
        self.threshold = getattr(model.rasterizer, 'transmittance_threshold', 1e-3)

    def planes(self, x, F):
        i, j = self.ai, self.adj
        xc = .5*(x[i]+(F[i]@(self.wall_c-self.X[i])[..., None])[..., 0]+x[j]+(F[j]@(self.wall_c-self.X[j])[..., None])[..., 0])
        n = (torch.linalg.inv(.5*(F[i]+F[j])).transpose(1, 2)@self.wall_n[..., None])[..., 0]
        return torch.cat((n, (n*xc).sum(1, keepdim=True)), 1).contiguous()

    @torch.no_grad()
    def render(self, cam, x, F):
        m = self.m
        x = x.float().contiguous(); F = F.float()
        U, S, Vh = torch.linalg.svd(F)
        R = U@Vh
        m.texel_sv_axis = torch.nn.Parameter((R[:, None]@self.axis0.view(len(R), -1, 3)[..., None])[..., 0].reshape(self.axis0.shape))
        att_sites, att_values, att_temps = m.get_att_sv()
        texel_world = x[:, None]+(F[:, None]@self.texel_off[..., None])[..., 0]
        rgb_t = m.sv.forward(texel_world.reshape(-1, 3).contiguous(), cam, att_sites, att_values, att_temps).view(len(x), -1, 3)
        texel_rgbh = torch.cat((rgb_t.float(), self.height[..., None]), -1).contiguous()
        planes = self.planes(x, F)
        G = torch.linalg.inv(F).contiguous()
        bound = torch.cat((x, (self.r*S[:, 0])[:, None]), 1).contiguous()       # bounding spheres for visibility
        rz = m.rasterizer
        dev = x.device
        tiles_h = 1+(cam.height-1)//TILE_WIDTH; tiles_w = 1+(cam.width-1)//TILE_WIDTH
        total_tiles = tiles_h*tiles_w
        with wp.ScopedDevice(str(dev)):
            wp.set_stream(wp.stream_from_torch(torch.cuda.current_stream()))
            counts = torch.zeros(total_tiles+1, dtype=torch.int32, device=dev)
            wcam = cam.to_warp()
            assert m.args.is_pinhole
            wp.launch(rz.count_visible_kernel, dim=len(x), inputs=[wcam, bound, counts], block_dim=256)
            offsets = torch.cumsum(counts.long(), 0)
            nint = int(offsets[-1])
            prims = torch.zeros(nint, dtype=torch.int32, device=dev)
            keys = torch.zeros(nint, dtype=torch.int64, device=dev)
            wc = torch.zeros(total_tiles, dtype=torch.int32, device=dev)
            wp.launch(rz.write_visible_kernel, dim=len(x), inputs=[wcam, bound, offsets, prims, keys, wc], block_dim=256)
            prims = prims[torch.argsort(keys)].contiguous()
            color = torch.zeros(cam.height, cam.width, 3, dtype=torch.float32, device=dev)
            log_t = torch.zeros(cam.height, cam.width, dtype=torch.float32, device=dev)
            wp.launch(self.kernel, dim=total_tiles*TILE_SIZE, block_dim=TILE_SIZE, inputs=[
                wcam, wp.from_torch(x, dtype=wp.vec3f), wp.from_torch(self.X.contiguous(), dtype=wp.vec3f), self.r.contiguous(),
                wp.from_torch(G, dtype=wp.mat33f), torch.cat((self.normal, self.sigma[:, None]), 1).contiguous(),
                wp.from_torch(self.texel_rest, dtype=wp.vec3f), wp.from_torch(texel_rgbh, dtype=wp.vec4f),
                self.adj_off, planes, prims, offsets, float(self.threshold), wp.from_torch(color, dtype=wp.vec3f), log_t])
        return color.clamp(0, 1), (1-torch.exp(log_t)).clamp(0, 1)
