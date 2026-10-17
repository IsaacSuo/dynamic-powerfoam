"""Small coordinate-equivariance diagnostic using PowerfoamScene.forward.

Hand-authored parameters, original spatial texture/displacement/SV/rasterization.
No learned reconstruction, physical illumination model, or nonrigid experiment.
Run: python -m experiments.full_model_rigid_probe
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import torch
from torch import nn
import warp as wp

from powerfoam.bvh import AABBTree
from powerfoam.camera import TorchCamera
from powerfoam.color_fn import SphericalVoronoi
from powerfoam.rasterize import Rasterizer
from powerfoam.scene import PowerfoamScene


def tensor(value):
    return torch.as_tensor(np.array(value, copy=True), dtype=torch.float32, device="cuda")


def fixture(count):
    rng = np.random.default_rng(25)
    if count == 1:
        points = np.zeros((1, 3))
    else:
        x, y = np.meshgrid([-.65, 0, .65], [-.65, 0, .65])
        points = np.column_stack([x.ravel(), y.ravel(), .04*np.sin(3*x.ravel())])
    uv = np.column_stack([np.tile([-.6, -.2, .2, .6], 2), np.repeat([-.4, .4], 4)])
    # scene.get_normals uses the first ROW of the quaternion rotation matrix.
    quat = Rotation.from_euler("y", 90, degrees=True).as_quat()
    quat = np.roll(quat, 1)
    axis = rng.normal(size=(count, 8, 8, 3))
    axis = 5*axis / np.linalg.norm(axis, axis=-1, keepdims=True)
    site_color = np.column_stack([.30*np.sin(4*uv[:, 0]),
                                 .25*np.cos(4*uv[:, 0]), .3*uv[:, 1]])
    values = site_color[None, :, None, :] + rng.uniform(-.22, .22, (count, 8, 8, 3))
    return {
        "points": points,
        # softplus(beta=100) differs negligibly from these positive values.
        "radii": np.full(count, .52),
        "density": np.full(count, 8.),
        "quaternions": np.tile(quat, (count, 1)),
        "texel_sites": np.tile(uv, (count, 1, 1)),
        "texel_height": np.tile(.22*np.sin(4*uv[:, 0]) + .10*uv[:, 1], (count, 1)),
        "texel_sv_axis": axis.reshape(count, 8, 24),
        "texel_sv_rgb": values.reshape(count, 8, 24),
    }


def camera(resolution, rotation=None, shift=None):
    rotation = np.eye(3) if rotation is None else rotation
    shift = np.zeros(3) if shift is None else shift
    result = TorchCamera(tensor(rotation @ [0., 0., 8.] + shift),
                         tensor(rotation @ [.19, 0., 0.]),
                         tensor(rotation @ [0., .19, 0.]), resolution, resolution)
    result.ray_maps = result._build_pinhole_ray_maps()
    return result


def transformed(state, rotation, shift, rotate_frame=True, rotate_sv=True):
    result = {key: value.copy() for key, value in state.items()}
    result["points"] = state["points"] @ rotation.T + shift
    if rotate_frame:
        frames = Rotation.from_quat(np.roll(state["quaternions"], -1, axis=-1)).as_matrix()
        result["quaternions"] = np.roll(
            Rotation.from_matrix(frames @ rotation.T).as_quat(), 1, axis=-1)
    if rotate_sv:
        axes = state["texel_sv_axis"].reshape(-1, 8, 8, 3)
        result["texel_sv_axis"] = (axes @ rotation.T).reshape(-1, 8, 24)
    return result


def load_state(model, state):
    for name, value in state.items():
        setattr(model, name, nn.Parameter(tensor(value)))
    model.rebuild_adjacency()


@torch.no_grad()
def render(model, cam):
    depth_q = torch.full((cam.height, cam.width, 1), .5, device="cuda")
    outputs = model.forward(cam, depth_quantiles=depth_q)
    return {name: outputs[index].detach().cpu().numpy()
            for name, index in [("rgb", 0), ("alpha", 1), ("depth", 4)]}


def compare(reference, value):
    foreground = (reference["alpha"] > .05) | (value["alpha"] > .05)
    error = np.abs(reference["rgb"] - value["rgb"])
    depth_mask = (reference["depth"][..., 0] > 0) & (value["depth"][..., 0] > 0)
    return {
        "rgb_mae_all": float(error.mean()),
        "rgb_mae_foreground": float(error[foreground].mean()),
        "rgb_max": float(error.max()),
        "pixels_rgb_error_above_0.01": int((error.max(axis=-1) > .01).sum()),
        "alpha_mae": float(np.abs(reference["alpha"]-value["alpha"]).mean()),
        "depth_mae_both_valid": float(np.abs(reference["depth"]-value["depth"])[depth_mask].mean()),
        "foreground_pixels": int(foreground.sum()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=192)
    parser.add_argument("--out", type=Path, default=Path("output/full_model_rigid"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("Original Power Foam rasterizer requires CUDA")
    wp.init()
    settings = SimpleNamespace(render_objective="volume", num_texel_sites=8,
                               sv_dof=8, is_pinhole=True,
                               disable_coop_prim_load=False, disable_coop_adj_load=False)
    model = PowerfoamScene(settings)
    model.aabb_tree = AABBTree(torch.device("cuda:0"))
    model.rasterizer = Rasterizer(settings, torch.device("cuda:0"))
    model.sv = SphericalVoronoi(settings, torch.device("cuda:0"))
    rotation = Rotation.from_euler("xyz", [24, -31, 17], degrees=True).as_matrix()
    shift = np.array([.3, -.2, .4])
    base_camera = camera(args.resolution)
    moved_camera = camera(args.resolution, rotation, shift)
    model.sv.fov_cos_cutoff = SphericalVoronoi.compute_fov_cos_cutoff(base_camera)
    results = {
        "scope": "Hand-authored coordinate-equivariance diagnostic; original full scene forward, no training or physical-lighting claim",
        "parameters": {"sites_per_cell": 8, "sv_axes_per_site": 8, "nonzero_heights": True,
                       "resolution": args.resolution, "rotation_xyz_degrees": [24, -31, 17],
                       "translation": shift.tolist(), "seed": 25},
        "cases": {},
    }
    for count in [1, 9]:
        state = fixture(count)
        load_state(model, state)
        reference = render(model, base_camera)
        assert reference["alpha"].max() > .9, "Empty or unexpectedly transparent fixture"
        moved = transformed(state, rotation, shift)
        cases = {
            "rigid_complete": (moved, moved_camera),
            "omit_frame_rotation": (transformed(state, rotation, shift, rotate_frame=False), moved_camera),
            "omit_sv_rotation": (transformed(state, rotation, shift, rotate_sv=False), moved_camera),
            "zero_height": ({**state, "texel_height": np.zeros_like(state["texel_height"])}, base_camera),
            "flatten_local_color": ({**state, "texel_sv_rgb": np.broadcast_to(
                state["texel_sv_rgb"].mean(axis=1, keepdims=True), state["texel_sv_rgb"].shape).copy()}, base_camera),
        }
        renders = {"reference": reference}
        metrics = {}
        for name, (case, cam) in cases.items():
            load_state(model, case)
            value = render(model, cam)
            assert all(np.isfinite(v).all() for v in value.values()), name
            metrics[name] = compare(reference, value)
            renders[name] = value
        results["cases"][str(count)] = metrics
        fig, axes = plt.subplots(2, len(renders), figsize=(18, 6))
        for column, (name, value) in enumerate(renders.items()):
            axes[0, column].imshow(np.clip(value["rgb"], 0, 1))
            axes[0, column].set_title(name.replace("_", "\n"))
            error = np.abs(reference["rgb"]-value["rgb"]).max(axis=-1)
            axes[1, column].imshow(error, vmin=0, vmax=.1, cmap="magma")
            axes[1, column].set_title("max RGB error (0–0.1)")
            for row in range(2):
                axes[row, column].axis("off")
            Image.fromarray((np.clip(value["rgb"], 0, 1)*255+.5).astype(np.uint8)).save(
                args.out / f"{count}_cells_{name}.png")
        fig.suptitle(f"{count} cell(s), original PowerfoamScene.forward, nonzero detail heights + directional color")
        fig.tight_layout()
        fig.savefig(args.out / f"{count}_cells_comparison.png", dpi=120)
        plt.close(fig)
        print(json.dumps({"cells": count, "metrics": metrics}, indent=2), flush=True)
    (args.out / "results.json").write_text(json.dumps(results, indent=2)+"\n")


if __name__ == "__main__":
    main()
