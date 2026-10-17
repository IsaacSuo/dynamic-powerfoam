"""Load the official PowerSim bread checkpoint in the local Power Foam renderer.

No retraining or appearance simplification. Official camera bundle replaces
training images. CPU deserialization uses weights_only=True.
"""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np


CHECKPOINT = Path('data/powersim/powerfoam_ckpt/bread_roll@ec2ced65')
ATTRIBUTES = ('points', 'density', 'radii', 'quaternions', 'texel_sites',
              'texel_height', 'texel_sv_axis', 'texel_sv_rgb')


def load_scene(folder=CHECKPOINT):
    import torch
    import warp as wp
    import yaml
    from powerfoam.scene import PowerfoamScene
    from powerfoam.bvh import AABBTree
    from powerfoam.rasterize import Rasterizer
    from powerfoam.color_fn import SphericalVoronoi
    wp.init()
    config = yaml.safe_load((folder/'config.yaml').read_text())
    checkpoint = torch.load(folder/'model.pt', map_location='cpu', weights_only=True)
    cameras = torch.load(folder/'cameras_test.pt', map_location='cpu', weights_only=True)
    if cameras['format'] != 'powersim.cameras.v1':
        raise ValueError('Unsupported camera bundle')
    settings = SimpleNamespace(**config)
    model = PowerfoamScene(settings)
    device = torch.device('cuda:0')
    model.aabb_tree = AABBTree(device)
    model.rasterizer = Rasterizer(settings, device)
    model.sv = SphericalVoronoi(settings, device)
    model.sv.fov_cos_cutoff = 0.
    for name in ATTRIBUTES:
        value = checkpoint[name]
        if not torch.isfinite(value).all():
            raise ValueError(f'Nonfinite checkpoint attribute {name}')
        setattr(model, name, torch.nn.Parameter(value.to(device).contiguous(), requires_grad=False))
    model.rebuild_adjacency()
    return model, cameras, config, checkpoint


def camera_from_bundle(bundle, index, width=512):
    import torch
    from powerfoam.camera import TorchCamera
    old_w, old_h = map(int, bundle['img_wh'])
    height = round(width*old_h/old_w)
    # Bread is pinhole; a non-pinhole bundle requires its actual ray directions.
    if bundle['cam_ray_dirs'] is not None:
        raise ValueError('This adapter currently supports pinhole bundles only')
    cam = TorchCamera(bundle['eyes'][index].float().cuda(),
                      bundle['rights'][index].float().cuda(), bundle['ups'][index].float().cuda(),
                      width, height)
    cam.ray_maps = cam._build_pinhole_ray_maps()
    return cam


def render_image(model, camera, label=''):
    from experiments.full_model_rigid_probe import render
    from PIL import Image, ImageDraw
    result = render(model, camera)
    if not all(np.isfinite(v).all() for v in result.values()) or result['alpha'].max() < .5:
        raise ValueError('Invalid or empty render')
    # The checkpoint was trained with black background; keep it unchanged.
    image = Image.fromarray(np.uint8(np.clip(result['rgb'], 0, 1)*255+.5))
    if label:
        ImageDraw.Draw(image).text((10, 10), label, fill='white')
    return image, result


def inspect_and_render(model, bundle, config, checkpoint, out):
    from PIL import Image
    import torch
    p = checkpoint['points'].numpy()
    radii = model.get_radii().detach().cpu().numpy()
    density = model.get_density().detach().cpu().numpy()
    offsets = model.adjacency_offsets.cpu().numpy()
    degree = np.diff(offsets)
    levels = [0, .01, .1, .5, .9, .99, 1]
    report = dict(source='https://huggingface.co/datasets/Tung11/PowerSim',
                  checkpoint='bread_roll@ec2ced65', particles=len(p),
                  checkpoint_shapes={k: list(v.shape) for k, v in checkpoint.items() if hasattr(v, 'shape')},
                  point_quantiles=np.quantile(p, levels, axis=0).tolist(), quantile_levels=levels,
                  radius_quantiles=np.quantile(radii, levels).tolist(),
                  optical_density_quantiles=np.quantile(density, levels).tolist(),
                  graph_offset_shape=list(offsets.shape), directed_adjacency_entries=len(model.adjacency),
                  degree_quantiles=np.quantile(degree, levels).tolist(),
                  cameras=len(bundle['eyes']), native_image_size=list(map(int, bundle['img_wh'])),
                  config_num_texel_sites=config['num_texel_sites'], config_sv_dof=config['sv_dof'],
                  model_sha256=hashlib.sha256((CHECKPOINT/'model.pt').read_bytes()).hexdigest(), views=[])
    indices = np.linspace(0, len(bundle['eyes'])-1, 8).astype(int)
    sheet = Image.new('RGB', (512*4, 512*2), 'black')
    for number, index in enumerate(indices):
        camera = camera_from_bundle(bundle, int(index))
        image, result = render_image(model, camera, f'Official bread checkpoint | view {index}')
        image.save(out/f'static_{index:03d}.png')
        sheet.paste(image, ((number%4)*512, (number//4)*512))
        report['views'].append(dict(index=int(index), foreground_pixels=int((result['alpha']>.5).sum()),
                                    max_alpha=float(result['alpha'].max())))
    sheet.save(out/'static_views.png')
    (out/'checkpoint_audit.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, indent=2), flush=True)
    return report


def regular_candidates(points, radii):
    """Lower lifted convex hull -> regular-triangulation edge candidates.

    Qhull floating-point geometry; degeneracy triangulation may add zero-area
    neighbors. Sphere-plane feasibility is necessary, not sufficient, for a
    positive-area bounded power face. This is NOT a final material bond graph.
    """
    from scipy.spatial import ConvexHull
    p, r = np.asarray(points, dtype=float), np.asarray(radii, dtype=float)
    scale = np.ptp(p, axis=0).max()
    if scale <= 0 or np.any(r <= 0):
        raise ValueError('Invalid geometry for weighted lifting')
    z = (p-p.mean(0))/scale
    lifted = np.column_stack((z, np.sum(z*z, axis=1)-(r/scale)**2))
    hull = ConvexHull(lifted)
    lower = hull.simplices[hull.equations[:, -2] < -1e-10]
    pairs = np.concatenate([lower[:, [i, j]] for i in range(4) for j in range(i+1, 4)])
    pairs = np.unique(np.sort(pairs, axis=1), axis=0)
    distance = np.linalg.norm(p[pairs[:, 1]]-p[pairs[:, 0]], axis=1)
    ri, rj = r[pairs[:, 0]], r[pairs[:, 1]]
    a = (distance**2+ri**2-rj**2)/(2*distance)
    possible = (distance < ri+rj) & (abs(a) < ri) & (abs(distance-a) < rj)
    return pairs, possible, np.unique(lower)


def graph_audit(out):
    import torch
    from scipy.spatial import Delaunay
    # Equal weights must reproduce ordinary Delaunay edges on generic points.
    rng = np.random.default_rng(928)
    x = rng.normal(size=(30, 3))
    pair, _, _ = regular_candidates(x, np.ones(30))
    tetra = Delaunay(x).simplices
    expected = np.concatenate([tetra[:, [i, j]] for i in range(4) for j in range(i+1, 4)])
    expected = np.unique(np.sort(expected, axis=1), axis=0)
    assert np.array_equal(pair, expected)
    # Small-weight center is hidden by eight sufficiently large corner balls.
    corners = np.array(np.meshgrid([-1., 1.], [-1., 1.], [-1., 1.])).reshape(3, -1).T
    _, _, active = regular_candidates(np.vstack((corners, [0., 0., 0.])), np.r_[np.full(8, 2.), .01])
    assert 8 not in active
    state = torch.load(CHECKPOINT/'model.pt', map_location='cpu', weights_only=True)
    points = state['points'].numpy()
    radii = torch.nn.functional.softplus(state['radii'], beta=100).numpy()
    pairs, possible, active = regular_candidates(points, radii)
    np.savez_compressed(out/'regular_candidates.npz', pairs=pairs, possible=possible,
                        active_unbounded=active)
    audit = json.loads((out/'checkpoint_audit.json').read_text())
    report = dict(particles=len(points), unbounded_power_active_sites=len(active),
                  hidden_unbounded_sites=len(points)-len(active), regular_triangulation_edges=len(pairs),
                  sphere_plane_feasible_candidates=int(possible.sum()),
                  current_cech_undirected_candidates=audit['directed_adjacency_entries']//2,
                  candidate_reduction_factor=float(audit['directed_adjacency_entries']/2/possible.sum()),
                  equal_radius_delaunay_check=True, hidden_site_check=True,
                  scope='Reference adjacency audit only. Not yet clipped positive-area faces, occupied-volume masses, or a simulated material graph.',
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'graph_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=Path('output/powersim_bread'))
    parser.add_argument('--graph-only', action='store_true')
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    if args.graph_only:
        graph_audit(args.out)
        return
    model, cameras, config, checkpoint = load_scene()
    inspect_and_render(model, cameras, config, checkpoint, args.out)
    graph_audit(args.out)


if __name__ == '__main__':
    main()
