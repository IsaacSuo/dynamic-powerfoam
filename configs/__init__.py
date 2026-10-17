from collections.abc import Sequence
import dataclasses
from dataclasses import dataclass
import inspect
from typing import Optional, Union

import configargparse


def str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() == "true":
        return True
    elif v.lower() == "false":
        return False
    else:
        raise configargparse.ArgumentTypeError("Boolean value expected. True or False.")


def add_group(parser: configargparse.ArgParser, group_class):
    fields = dict(inspect.getmembers(group_class))["__dataclass_fields__"]
    for name, field in fields.items():
        required = field.default == dataclasses.MISSING
        t = field.type
        nargs = None
        if hasattr(t, "__origin__") and hasattr(t, "__args__"):
            origin = t.__origin__
            if origin is Union and len(t.__args__) == 2 and type(None) in t.__args__:
                for arg in t.__args__:
                    if arg != type(None):
                        t = arg
                        break
                if required:
                    field.default = None
                    required = False
            elif issubclass(origin, Sequence):
                t = t.__args__[0]
                nargs = "+"
            else:
                raise ValueError(f"Unsupported type {t} for field {name}")

        if required and t == bool:
            parser.add_argument(
                f"--{name}", type=str_to_bool, nargs="?", const=True, required=True
            )
        elif required:
            parser.add_argument(f"--{name}", nargs=nargs, type=t, required=True)
        elif t == bool:
            assert field.default == False
            parser.add_argument(f"--{name}", action="store_true")
        else:
            parser.add_argument(f"--{name}", nargs=nargs, type=t, default=field.default)

    def extract(args):
        kwargs = {}
        for arg in vars(args).items():
            if arg[0] in fields:
                kwargs[arg[0]] = arg[1]
        return group_class(**kwargs)

    return extract


@dataclass(kw_only=True)
class Params:
    # Experiement parameters
    iterations: int
    normal_weight: float
    contribution_weight: float
    interpenetration_weight: float
    densify_from: int
    densify_until: int
    experiment_name: str = ""
    dry_run: bool = False
    viewer: bool = False
    # Enable external normal supervision.  When False (default), the only
    # normal-related term in the loss is the renderer's internal
    # ``normal_err`` regulariser.  When True, an additional MSE supervision
    # term is added against either Metric3D normals (if ``use_metric3d`` is
    # also True) or finite-difference normals computed from the rendered
    # depth map.
    normal_supervision: bool = False

    # Dataset parameters
    dataset: str
    data_path: str
    scene: str
    alpha_format_on_disk: str
    downsample: list[int]
    downsample_iterations: list[int]
    use_metric3d: bool = False
    is_pinhole: bool = False
    eval: bool = False

    # Model parameters
    init_type: str
    init_points: int
    final_points: int
    bkgd_color: list[float]
    disable_coop_prim_load: bool = False
    disable_coop_adj_load: bool = False
    render_objective: str = "volume"
    sv_dof: int
    num_texel_sites: int
    # Mechanics-aware spacing (0 = off): adjacent sites (k nearest) closer than
    # spacing_kappa x their mean local scale (median distance to the spacing_k nearest
    # sites, held fixed between neighbour rebuilds) are pushed apart; duplicated cells are
    # placed spacing_kappa x local scale away instead of 0.05 x radius.
    spacing_weight: float = 0.0
    spacing_kappa: float = 0.5
    spacing_k: int = 12
    # Centroidal Voronoi (Lloyd) regularisation, 0 = off: every cvt_every iterations the
    # centroid of each bounded power cell is estimated by Monte Carlo (cvt_samples per cell,
    # power-distance owner inside the owner's support sphere); the loss pulls sites towards
    # their (detached) centroids, |p - c|^2 / L^2 with L the local k-NN median spacing.
    cvt_weight: float = 0.0
    cvt_every: int = 200
    cvt_samples: int = 32

    # Optimizer parameters
    points_lr_init: float
    points_lr_final: float
    density_lr_init: float
    density_lr_final: float
    radii_lr_init: float
    radii_lr_final: float
    quaternions_lr_init: float
    quaternions_lr_final: float
    texel_sites_lr_init: float
    texel_sites_lr_final: float
    texel_sv_axis_lr_init: float
    texel_sv_axis_lr_final: float
    texel_sv_rgb_lr_init: float
    texel_sv_rgb_lr_final: float
    texel_height_lr_init: float
    texel_height_lr_final: float
