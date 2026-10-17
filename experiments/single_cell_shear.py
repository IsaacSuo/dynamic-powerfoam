"""Fit local detail parameters to affine-pulled-back original scene renders.

Original PowerfoamScene.forward supplies both target and candidate evaluations.
Artificial single-cell diagnostic, not a trained dynamic scene or physical light model.
"""
import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
import torch
import warp as wp

from powerfoam.bvh import AABBTree
from powerfoam.camera import TorchCamera
from powerfoam.color_fn import SphericalVoronoi
from powerfoam.rasterize import Rasterizer
from powerfoam.scene import PowerfoamScene
from experiments.full_model_rigid_probe import fixture, tensor, load_state


def make_model(pinhole=False):
    settings = SimpleNamespace(render_objective="volume", num_texel_sites=8,
                               sv_dof=8, is_pinhole=pinhole,
                               disable_coop_prim_load=False, disable_coop_adj_load=False)
    model = PowerfoamScene(settings)
    model.aabb_tree = AABBTree(torch.device("cuda:0"))
    model.rasterizer = Rasterizer(settings, torch.device("cuda:0"))
    model.sv = SphericalVoronoi(settings, torch.device("cuda:0"))
    model.sv.fov_cos_cutoff = 0.
    return model


def make_camera(resolution, angle=0.):
    rot = Rotation.from_euler("y", angle, degrees=True).as_matrix()
    # Existing visibility kernels test dot(center-eye, cross(up,right)) > .1
    # without normalizing that cross product. A .105 half-FOV is falsely culled
    # at this distance; .15 is inside the current implementation's valid regime.
    cam = TorchCamera(tensor(rot @ [0., 0., 8.]), tensor(rot @ [.15, 0., 0.]),
                      tensor(rot @ [0., .15, 0.]), resolution, resolution)
    cam.ray_maps = cam._build_pinhole_ray_maps()
    return cam


def inverse_camera(cam, matrix):
    inv = torch.linalg.inv(tensor(matrix))
    origins = cam.ray_maps[..., :3] @ inv.T
    raw_dirs = cam.ray_maps[..., 3:] @ inv.T
    scale = raw_dirs.norm(dim=-1)
    rays = torch.cat([origins, raw_dirs / scale[..., None]], dim=-1).contiguous()
    # The actual rays are provided explicitly; frame is used only for culling/SV FOV.
    eye = inv @ cam.eye
    forward = -eye / eye.norm()
    right = torch.linalg.cross(forward, tensor([0., 1., 0.]))
    right = right / right.norm()
    up = torch.linalg.cross(right, forward)
    pulled = TorchCamera(eye, .2*right, .2*up, cam.width, cam.height, rays)
    return pulled, scale


def evaluate(model, cam):
    quantiles = torch.full((cam.height, cam.width, 1), .5, device="cuda")
    output = model.forward(cam, depth_quantiles=quantiles)
    alpha = output[1]
    color = output[0] / alpha.clamp_min(1e-6)[..., None]
    # One front-facing cell, constant sigma: median t = entry t + log(2)/sigma.
    surface = output[4][..., 0] - math.log(2)/model.get_density()[0]
    return {"rgb": output[0], "alpha": alpha, "color": color, "surface": surface,
            "depth_valid": output[4][..., 0] > 0}


def sphere_hit(origins, directions, radius):
    # Only analytic support coverage; all appearance and surface queries use original forward.
    a = (directions*directions).sum(-1)
    b = (origins*directions).sum(-1)
    c = (origins*origins).sum(-1)-radius*radius
    disc = b*b-a*c
    return (disc >= 0) & ((-b+torch.sqrt(disc.clamp_min(0)))/a > 0)


@torch.no_grad()
def target_for(source, cam, matrix):
    pulled, scale = inverse_camera(cam, matrix)
    ref = evaluate(source, pulled)
    target_surface = ref["surface"] / scale
    # Keep world extinction fixed; correct the changed canonical ray length.
    alpha = -torch.expm1(torch.log1p(-ref["alpha"].clamp(max=1-1e-7))/scale)
    r = float(source.get_radii()[0])
    origins, directions = cam.ray_maps[..., :3], cam.ray_maps[..., 3:]
    base_t = -origins[..., 2]/directions[..., 2]
    base_point = origins + base_t[..., None]*directions
    canonical_point = base_point @ torch.linalg.inv(tensor(matrix)).T
    # Fixed common interior, independent of optimized parameters; boundary is reported separately.
    interior = (base_point.norm(dim=-1) < .70*r) & (canonical_point.norm(dim=-1) < .70*r)
    interior &= ref["depth_valid"] & (alpha > .55)
    target_support = sphere_hit(pulled.ray_maps[..., :3], pulled.ray_maps[..., 3:], r)
    candidate_support = sphere_hit(origins, directions, r)
    return {"color": ref["color"], "surface": target_surface, "alpha": alpha,
            "rgb": ref["color"]*alpha[..., None], "mask": interior,
            "support": {
                "target_rays": int(target_support.sum()),
                "miss_pct": float(100*(target_support & ~candidate_support).sum()/target_support.sum()),
                "extra_pct_of_candidate": float(100*(candidate_support & ~target_support).sum()/candidate_support.sum()),
                "iou": float((target_support & candidate_support).sum()/(target_support | candidate_support).sum()),
            }}


def loss_for(pred, target, radius):
    mask = target["mask"]
    color = (pred["color"][mask]-target["color"][mask]).square().mean()
    surface = ((pred["surface"][mask]-target["surface"][mask])/radius).square().mean()
    return color+surface


@torch.no_grad()
def metrics(pred, target, radius):
    mask = target["mask"]
    valid = mask & pred["depth_valid"]
    return {
        "interior_pixels": int(mask.sum()),
        "missing_surface_pct": float(100*(mask & ~pred["depth_valid"]).sum()/mask.sum()),
        "color_mae": float((pred["color"][mask]-target["color"][mask]).abs().mean()),
        "surface_rmse_over_radius": float(((pred["surface"][valid]-target["surface"][valid])/radius).square().mean().sqrt()),
        "rgb_mae_all": float((pred["rgb"]-target["rgb"]).abs().mean()),
        "alpha_mae_all": float((pred["alpha"]-target["alpha"]).abs().mean()),
    }


def snapshot(model):
    return {name: value.detach().cpu().numpy().copy() for name, value in model.named_parameters()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--train-resolution", type=int, default=96)
    parser.add_argument("--eval-resolution", type=int, default=192)
    parser.add_argument("--out", type=Path, default=Path("output/single_cell_shear"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    wp.init()
    state = fixture(1)
    source, candidate = make_model(), make_model()
    load_state(source, state)
    radius = float(source.get_radii()[0].detach())
    train_cam = make_camera(args.train_resolution)
    cameras = {"same_view_dense": make_camera(args.eval_resolution),
               "heldout_12deg": make_camera(args.eval_resolution, 12)}
    # Ensure the generic-ray route reproduces the existing pinhole route at identity.
    pinhole = make_model(True)
    load_state(pinhole, state)
    with torch.no_grad():
        direct, generic = evaluate(pinhole, train_cam), evaluate(source, train_cam)
        path_error = float((direct["rgb"]-generic["rgb"]).abs().max())
        path_mae = float((direct["rgb"]-generic["rgb"]).abs().mean())
        identity = target_for(source, train_cam, np.eye(3))
        identity_metrics = metrics(generic, identity, radius)
    print(json.dumps({"identity":identity_metrics,"path_rgb_max":path_error,
                      "path_rgb_mae":path_mae}),flush=True)
    # Different floating-point ray normalization paths affect silhouette pixels.
    # Bound mean AND worst pixel, rather than requiring bit-identical routes.
    assert path_mae < 1e-5 and path_error < 1e-3, f"Generic/pinhole mismatch: {path_mae}, {path_error}"
    assert int((generic["alpha"] > .9).sum()) > 100, "Identity view was culled or empty"
    assert identity_metrics["color_mae"] < 1e-5
    assert identity_metrics["surface_rmse_over_radius"] < 1e-5
    results = {"scope": "Single-cell affine pullback of original forward; shared fixed SV coefficients, no dynamic scene fitting",
               "settings": vars(args) | {"out": str(args.out)},
               "identity_checks": {"generic_pinhole_rgb_max": path_error,
                                   "generic_pinhole_rgb_mae": path_mae, **identity_metrics},
               "shears": {}}
    for shear in [.5, 1.]:
        matrix = np.array([[1., shear, 0.], [0., 1., 0.], [0., 0., 1.]])
        train_target = target_for(source, train_cam, matrix)
        targets = {name: target_for(source, cam, matrix) for name, cam in cameras.items()}
        # Transport sites in world space, then express them in unchanged local frame.
        with torch.no_grad():
            tangent, bitangent = source.get_tangents()
            basis = torch.stack([tangent[0], bitangent[0]], dim=0).cpu().numpy()
        advected = {k: v.copy() for k, v in state.items()}
        advected["texel_sites"] = (state["texel_sites"] @ basis @ matrix.T) @ basis.T
        cases = {"unchanged": state, "advected_sites": advected}
        histories = {}
        for mode in ["fit_sites", "fit_sites_and_heights"]:
            load_state(candidate, advected)
            names = ["texel_sites"] + (["texel_height"] if mode.endswith("heights") else [])
            for name, p in candidate.named_parameters():
                p.requires_grad_(name in names)
            optimizer = torch.optim.Adam([getattr(candidate, n) for n in names], lr=.005)
            history, best_loss, best = [], float("inf"), snapshot(candidate)
            for step in range(args.steps+1):
                optimizer.zero_grad(set_to_none=True)
                pred = evaluate(candidate, train_cam)
                loss = loss_for(pred, train_target, radius)
                score = float(loss.detach())
                assert math.isfinite(score)
                if score < best_loss:
                    best_loss, best = score, snapshot(candidate)
                if step % 50 == 0 or step == args.steps:
                    history.append({"step": step, "loss": score})
                    print(f"shear={shear} {mode} {step}/{args.steps} loss={score:.7f}", flush=True)
                if step == args.steps:
                    break
                loss.backward()
                for name in names:
                    assert torch.isfinite(getattr(candidate, name).grad).all(), name
                optimizer.step()
                with torch.no_grad():
                    candidate.texel_sites.clamp_(-1.5, 1.5)
                    candidate.texel_height.clamp_(-.45, .45)
            cases[mode], histories[mode] = best, history
        section = {"support_by_view": {n:t["support"] for n,t in targets.items()},
                   "optimization_history": histories, "metrics": {}}
        for view, cam in cameras.items():
            target = targets[view]
            rows = [("target", target)]
            section["metrics"][view] = {}
            for name, values in cases.items():
                load_state(candidate, values)
                with torch.no_grad():
                    pred = evaluate(candidate, cam)
                    section["metrics"][view][name] = metrics(pred, target, radius)
                rows.append((name, pred))
                np.savez(args.out / f"shear_{shear}_{name}.npz", **values)
            fig, axes = plt.subplots(3, len(rows), figsize=(15, 8))
            mask = target["mask"].cpu().numpy()
            for col, (name, value) in enumerate(rows):
                axes[0,col].imshow(value["rgb"].detach().cpu().numpy().clip(0,1))
                axes[0,col].contour(mask, levels=[.5], colors=["white"], linewidths=.5)
                axes[0,col].set_title(name.replace("_", "\n"))
                color_error = (value["color"]-target["color"]).abs().mean(-1).detach().cpu().numpy()
                height_error = ((value["surface"]-target["surface"])/radius).abs().detach().cpu().numpy()
                for row, error, limit, label in [(1,color_error,.1,"Color MAE, 0–0.1"),
                                                (2,height_error,.1,"Surface error/r, 0–0.1")]:
                    axes[row,col].imshow(np.ma.masked_where(~mask,error),vmin=0,vmax=limit,cmap="magma")
                    axes[row,col].set_title(label)
                for row in range(3): axes[row,col].axis("off")
            fig.suptitle(f"Single cell, x += {shear} y, {view}; white outline: fixed evaluation interior")
            fig.tight_layout()
            fig.savefig(args.out/f"shear_{shear}_{view}.png",dpi=120)
            plt.close(fig)
        results["shears"][str(shear)] = section
        (args.out/"results.json").write_text(json.dumps(results,indent=2)+"\n")
        print(json.dumps({"shear":shear,"metrics":section["metrics"]},indent=2),flush=True)


if __name__ == "__main__":
    main()
