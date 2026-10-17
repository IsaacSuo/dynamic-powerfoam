"""Frozen-appearance affine probe; run with `python -m experiments.affine_probe`.

Uses exact 3D bounded-power membership and the repository's texture weight
formula. GPU panels call the unchanged Power Foam Rasterizer.visualize path.
This is a planar material patch, not an MPM simulation or trained reconstruction.
"""

import argparse
import csv
import importlib.metadata
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


MODES = ("fixed", "volume", "cover")


def membership(query, centers, radii):
    """Owner=-1 outside the union. All power constraints, no graph approximation."""
    owner = np.empty(len(query), dtype=np.int64)
    for start in range(0, len(query), 4096):
        q = query[start:start + 4096]
        power = np.sum((q[:, None] - centers[None]) ** 2, axis=-1) - radii**2
        ids = power.argmin(axis=1)
        ids[power[np.arange(len(q)), ids] > 0] = -1
        owner[start:start + len(q)] = ids
    return owner


def texture(query, owner, sites, colors, radii):
    """Matches exp(-10 * squared_world_distance / radius**2), zero displacement."""
    valid = owner >= 0
    rgb = np.zeros((len(query), 3))
    ids = owner[valid]
    d2 = np.sum((query[valid, None] - sites[ids]) ** 2, axis=-1)
    weights = np.exp(-10 * d2 / radii[ids, None] ** 2)
    rgb[valid] = np.sum(weights[..., None] * colors[ids], axis=1) / np.maximum(
        weights.sum(axis=1, keepdims=True), 1e-20
    )
    return rgb


def radius_rule(r0, deformation, mode):
    if mode == "fixed":
        return r0.copy()
    if mode == "volume":
        return r0 * np.cbrt(np.linalg.det(deformation))
    return r0 * np.linalg.svd(deformation, compute_uv=False).max()


def scene(seed, uniform):
    rng = np.random.default_rng(seed)
    xx, yy = np.meshgrid(np.linspace(-1.1, 1.1, 11), np.linspace(-1.1, 1.1, 11))
    centers = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size)))
    centers[:, :2] += rng.uniform(-0.025, 0.025, (len(centers), 2))
    radii = np.full(len(centers), 0.22)
    if not uniform:
        radii *= rng.uniform(0.8, 1.2, len(centers))
    tx, ty = np.meshgrid(np.linspace(-0.65, 0.65, 4), [-0.45, 0.45])
    offsets = np.column_stack((tx.ravel(), ty.ravel(), np.zeros(8)))
    sites = centers[:, None] + radii[:, None, None] * offsets
    parity = np.floor((sites[..., :2] + 1.6) / 0.32).sum(axis=-1).astype(int) % 2
    palette = np.array([[0.95, 0.78, 0.20], [0.07, 0.22, 0.65]])
    return centers, radii, sites, palette[parity]


def transforms():
    angle = np.deg2rad(32)
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    return {
        "identity": (np.eye(3), np.zeros(3)),
        "rigid": (rotation, np.array([0.18, -0.12, 0])),
        "scale_1.6": (1.6 * np.eye(3), np.zeros(3)),
        "stretch_J1": (np.diag([2.0, 0.5, 1.0]), np.zeros(3)),
        "shear_J1": (np.array([[1., 0.9, 0.], [0., 1., 0.], [0., 0., 1.]]), np.zeros(3)),
    }


def save_image(path, rgb):
    Image.fromarray(np.uint8(np.clip(rgb, 0, 1) * 255 + 0.5)).save(path)


class GPURenderer:
    def __init__(self, resolution):
        import torch
        import warp as wp
        from powerfoam.bvh import AABBTree
        from powerfoam.camera import TorchCamera
        from powerfoam.rasterize import Rasterizer, VisOptions

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; use --no-gpu only for the analytic probe")
        wp.init()
        self.torch, self.wp = torch, wp
        self.device = torch.device("cuda:0")
        self.renderer = Rasterizer(SimpleNamespace(render_objective="volume",
            num_texel_sites=8, is_pinhole=True, disable_coop_prim_load=False,
            disable_coop_adj_load=False), self.device)
        self.tree = AABBTree(self.device)
        self.camera = TorchCamera(self.tensor([0, 0, 8]), self.tensor([3/8, 0, 0]),
                                  self.tensor([0, 3/8, 0]), resolution, resolution)
        self.camera.ray_maps = self.camera._build_pinhole_ray_maps()
        self.options = VisOptions()
        self.options.transmittance_threshold = 1e-4
        self.options.max_intersections = 1024
        self.options.depth_quantile = 0.5
        self.options.bkgd_color = wp.vec3f(0, 0, 0)

    def tensor(self, x):
        return self.torch.as_tensor(np.ascontiguousarray(x), dtype=self.torch.float32,
                                    device=self.device)

    def render(self, centers, radii, sites, colors, deformation, explicit_normals=None):
        t = self.torch
        points, r = self.tensor(centers), self.tensor(radii)
        self.tree.update(points, r)
        adjacency, offsets = self.tree.build_cech_complex()
        # Check actual GPU graph against brute-force sphere overlaps.
        d = np.linalg.norm(centers[:, None] - centers[None], axis=-1)
        expected = d < radii[:, None] + radii[None]
        np.fill_diagonal(expected, False)
        rows = np.repeat(np.arange(len(centers)), offsets.diff().cpu().numpy())
        actual = np.zeros_like(expected)
        actual[rows, adjacency.cpu().numpy()] = True
        np.testing.assert_array_equal(actual, expected)
        normal = np.linalg.solve(deformation.T, np.array([0., 0., 1.]))
        normal /= np.linalg.norm(normal)
        normals = np.tile(normal, (len(points), 1)) if explicit_normals is None else explicit_normals
        with t.no_grad():
            result = self.renderer.visualize(self.camera, points, r,
                self.tensor(np.full(len(points), 1000.)),
                self.tensor(normals),
                self.tensor(sites), self.tensor(colors), self.tensor(np.zeros(colors.shape[:2])),
                adjacency, offsets, self.options)
        t.cuda.synchronize()
        rgb, alpha = result[0].cpu().numpy(), result[3].cpu().numpy()
        assert np.isfinite(rgb).all() and np.isfinite(alpha).all()
        assert alpha.max() > 0.99, "GPU render unexpectedly empty"
        return rgb, alpha


def analytic_checks():
    # A point switches owner under volume-preserving affine stretch even when
    # it stays inside both its original owner's support and the union.
    centers = np.array([[0., 0., 0.], [1., 1., 0.]])
    q = np.array([[0.9, 0., 0.]])
    r = np.array([2., 2.])
    f = np.diag([2., 0.5, 1.])
    assert membership(q, centers, r)[0] == 0
    for mode in MODES:
        assert membership(q @ f.T, centers @ f.T, radius_rule(r, f, mode))[0] == 1
    return "two-cell ownership counterexample: passed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("output/affine_probe"))
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--uniform-radii", action="store_true")
    parser.add_argument("--no-gpu", action="store_true")
    args = parser.parse_args()
    if args.resolution < 32:
        parser.error("--resolution must be at least 32")
    args.out.mkdir(parents=True, exist_ok=True)
    print(analytic_checks(), flush=True)
    centers, r0, sites, colors = scene(args.seed, args.uniform_radii)
    cases = transforms()
    res = args.resolution
    gx, gy = np.meshgrid(np.linspace(-0.95, 0.95, res), np.linspace(0.95, -0.95, res))
    samples = np.column_stack((gx.ravel(), gy.ravel(), np.zeros(gx.size)))
    owner0 = membership(samples, centers, r0)
    valid0 = owner0 >= 0
    ref = texture(samples, owner0, sites, colors, r0)
    assert valid0.mean() > 0.99, "Reference patch has insufficient coverage"
    gpu = None if args.no_gpu else GPURenderer(res)
    wx, wy = np.meshgrid(np.linspace(-3, 3, res), np.linspace(3, -3, res))
    pixels = np.column_stack((wx.ravel(), wy.ravel(), np.zeros(wx.size)))
    figures = {}
    for kind, cols in [("material", 4), ("ownership", 3), ("gpu", 4)]:
        if kind == "gpu" and gpu is None:
            continue
        figures[kind] = plt.subplots(len(cases), cols, figsize=(3.5*cols, 3*len(cases)), squeeze=False)
    records, assertions = [], ["two-cell counterexample"]
    for row, (name, (f, shift)) in enumerate(cases.items()):
        current_points = centers @ f.T + shift
        current_sites = sites @ f.T + shift
        current_samples = samples @ f.T + shift
        inverse_samples = (current_samples - shift) @ np.linalg.inv(f).T
        np.testing.assert_allclose(inverse_samples, samples, atol=1e-12)
        # All image errors use the frozen represented texture as reference,
        # not the unattainable sharp checkerboard used to assign texel colors.
        ax = figures["material"][1][row, 0]
        ax.imshow(ref.reshape(res, res, 3)); ax.set_title(f"{name}: advected reference")
        if gpu is not None:
            canonical_pixels = (pixels - shift) @ np.linalg.inv(f).T
            pixel_owner0 = membership(canonical_pixels, centers, r0)
            expected = texture(canonical_pixels, pixel_owner0, sites, colors, r0).reshape(res, res, 3)
            figures["gpu"][1][row, 0].imshow(expected)
            figures["gpu"][1][row, 0].set_title(f"{name}: analytic surface reference")
            save_image(args.out / f"{name}_reference.png", expected)
        for col, mode in enumerate(MODES):
            r = radius_rule(r0, f, mode)
            owner = membership(current_samples, current_points, r)
            covered = valid0 & (owner >= 0)
            switched = covered & (owner != owner0)
            missed = valid0 & (owner < 0)
            naive = texture(current_samples, owner, current_sites, colors, r)
            # Isolate metric distortion by keeping the reference owner, even
            # where its deformed support no longer contains the query.
            metric_only = texture(current_samples, owner0, current_sites, colors, r)
            # Isolate ownership changes using the correct reference metric.
            pullback = texture(inverse_samples, owner, sites, colors, r0)
            restored = texture(inverse_samples, owner0, sites, colors, r0)
            np.testing.assert_allclose(restored, ref, atol=1e-12)
            distances = np.linalg.norm(current_points[:, None] - current_points[None], axis=-1)
            edges = distances < r[:, None] + r[None]
            np.fill_diagonal(edges, False)
            degree = edges.sum(axis=1)
            record = dict(case=name, mode=mode, support_miss_pct=100*missed.sum()/valid0.sum(),
                owner_switch_pct_covered=100*switched.sum()/covered.sum(),
                identity_loss_pct=100*(missed | switched).sum()/valid0.sum(),
                texture_metric_mae=float(np.abs(metric_only-ref)[valid0].mean()),
                ownership_only_mae_covered=float(np.abs(pullback-ref)[covered].mean()),
                combined_mae_covered=float(np.abs(naive-ref)[covered].mean()),
                combined_mae_including_gaps=float(np.abs(naive-ref)[valid0].mean()),
                pullback_fixed_owner_mae=float(np.abs(restored-ref)[valid0].mean()),
                mean_degree=float(degree.mean()), max_degree=int(degree.max()))
            if name in ("identity", "rigid") or (name == "scale_1.6" and mode != "fixed"):
                np.testing.assert_array_equal(owner, owner0)
                np.testing.assert_allclose(naive, ref, atol=1e-12)
                assertions.append(f"{name}/{mode}: ownership and texture equivariance")
            if mode == "cover":
                assert not missed.any(), "Spectral radius must cover every advected reference sample"
                assertions.append(f"{name}/cover: no support misses")
            a = figures["material"][1][row, col+1]
            a.imshow(naive.reshape(res, res, 3))
            a.set_title(f"{mode}: material-view MAE {record['combined_mae_including_gaps']:.4f}")
            ownership_rgb = np.full((len(samples), 3), 0.92)
            ownership_rgb[switched] = [1., 0.38, 0.08]
            ownership_rgb[missed] = [0.12, 0.22, 0.75]
            ownership_rgb[~valid0] = 0
            a = figures["ownership"][1][row, col]
            a.imshow(ownership_rgb.reshape(res, res, 3))
            a.set_title(f"{name}/{mode}\nswitch {record['owner_switch_pct_covered']:.1f}% | miss {record['support_miss_pct']:.1f}%")
            if gpu is not None:
                rgb, alpha = gpu.render(current_points, r, current_sites, colors, f)
                save_image(args.out / f"{name}_{mode}_gpu.png", rgb)
                save_image(args.out / f"{name}_{mode}_alpha.png", alpha)
                figures["gpu"][1][row, col+1].imshow(rgb)
                figures["gpu"][1][row, col+1].set_title(f"{mode}: original GPU rasterizer")
                if name == "identity":
                    mask = (pixel_owner0.reshape(res, res) >= 0) & (alpha > 0.999)
                    error = float(np.abs(rgb-expected)[mask].mean())
                    record["gpu_reference_opaque_mae"] = error
                    assert error < 0.015, f"GPU/analytic reference mismatch: {error}"
            records.append(record)
            print(f"{name:12} {mode:6} miss={record['support_miss_pct']:6.2f}% "
                  f"switch={record['owner_switch_pct_covered']:6.2f}% "
                  f"metric_MAE={record['texture_metric_mae']:.5f} degree={degree.mean():.2f}", flush=True)
    titles = {"material": "Frozen texture in material coordinates (shape change removed)",
              "ownership": "Material ownership: orange = switched; blue = unsupported; gray = retained",
              "gpu": "World view: analytic advected surface vs original Power Foam GPU volume rasterization"}
    for kind, (fig, axes) in figures.items():
        for ax in axes.ravel():
            ax.axis("off")
        fig.suptitle(titles[kind], fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        fig.savefig(args.out / f"{kind}_comparison.png", dpi=150)
        plt.close(fig)
    metadata = dict(seed=args.seed, resolution=res, uniform_radii=args.uniform_radii,
                    cells=len(centers), texels_per_cell=8, reference_coverage=float(valid0.mean()),
                    gpu_enabled=gpu is not None, assertions=assertions,
                    transforms={name: {"F": f.tolist(), "translation": t.tolist()} for name, (f,t) in cases.items()},
                    versions={pkg: importlib.metadata.version(pkg) for pkg in ("numpy", "matplotlib")})
    if gpu is not None:
        metadata["versions"].update({pkg: importlib.metadata.version(pkg) for pkg in ("torch", "warp-lang")})
        metadata["gpu"] = gpu.torch.cuda.get_device_name()
    (args.out / "results.json").write_text(json.dumps(dict(metadata=metadata, metrics=records), indent=2)+"\n")
    fields = sorted(set().union(*(r.keys() for r in records)))
    with (args.out / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(records)
    np.savez_compressed(args.out / "reference_state.npz", centers=centers, radii=r0, texel_sites=sites, texel_rgb=colors)
    print(f"Saved results to {args.out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
