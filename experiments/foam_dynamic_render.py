"""Render a trained Power Foam whose cells move with DEM rigid bodies.

load(run): the trained PowerfoamScene and its dataset cameras (from the run's config.yaml).
keep_cells(model, index): drop all other cells (air) and rebuild adjacency.
move(model, points, rotations): rigid per-cell motion (centres, normals via quaternions;
texel sites follow the cell frame); adjacency rebuilt for the new positions.

check_air_removal(): material-only vs full foam at rest, per view: alpha IoU, PSNR against the
captured RGBA (object mask), so the air cells can be left out of dynamic rendering.
"""
import argparse
import json
from pathlib import Path

import configargparse
import numpy as np
import torch
import warp as wp
from PIL import Image

from configs import Params, add_group
from data_loader import DataHandler
from powerfoam.scene import PowerfoamScene


def load(run):
    parser = configargparse.ArgParser()
    get_params = add_group(parser, Params)
    parser.add_argument('-c', '--config', is_config_file=True)
    args = get_params(parser.parse_args(['-c', str(Path(run)/'config.yaml')]))
    wp.init()
    data = DataHandler(args)
    data.reload('all', downsample=args.downsample[-1])
    model = PowerfoamScene(args)
    model.initialize_from_dataset(data, device='cuda')
    model.load_pt(str(Path(run)/'model.pt'))
    return model, data


RINGS = 2
NAMES = ('points', 'radii', 'quaternions', 'density', 'texel_sites', 'texel_sv_axis', 'texel_sv_rgb', 'texel_height')


def keep_cells(model, index):
    index = torch.as_tensor(index, device=model.device)
    for name in NAMES:
        setattr(model, name, torch.nn.Parameter(getattr(model, name).data[index].clone()))
    model.rebuild_adjacency()
    model.rest_points = model.points.data.clone()
    model.rest_quaternions = model.quaternions.data.clone()


def quat_mul(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack((aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
                        aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw), -1)


def move(model, points, cell_quaternions=None):
    """Set cell centres (and, if given, world rotations as unit quaternions per cell)."""
    model.points = torch.nn.Parameter(points.to(model.points.dtype))
    if cell_quaternions is not None:
        q = quat_mul(cell_quaternions.to(model.rest_quaternions.dtype),
                     model.rest_quaternions/model.rest_quaternions.norm(dim=-1, keepdim=True))
        model.quaternions = torch.nn.Parameter(q)
    model.rebuild_adjacency()


def render(model, camera):
    h, w = camera.height, camera.width
    q = .5*torch.ones(h, w, 1, device=model.device)
    rgb, alpha = model.forward(camera, depth_quantiles=q)[:2]
    return rgb, alpha


def air_shell(model, material, body_of_material, rings=1):
    """Air cells within `rings` regular-triangulation hops of the material, each attached to
    the body of its nearest material neighbour (they move rigidly with it). Returns
    (cell indices: material then shell, body index per kept cell)."""
    from experiments.powersim_bread_demo import regular_candidates
    p = model.points.detach().double().cpu().numpy()
    r = model.get_radii().detach().double().cpu().numpy()
    pairs, possible, _ = regular_candidates(p, r)
    pairs = pairs[possible]
    n = len(p)
    body = -np.ones(n, int); body[material] = body_of_material
    owner = body.copy()
    for _ in range(rings):
        new = owner.copy()
        dist = np.full(n, np.inf)
        for a, b in ((0, 1), (1, 0)):
            i, j = pairs[:, a], pairs[:, b]
            m = (owner[i] < 0) & (owner[j] >= 0)
            d = np.linalg.norm(p[i[m]]-p[j[m]], axis=1)
            order = np.argsort(-d)  # nearest last, so it wins the assignment
            ii, jj, dd = i[m][order], j[m][order], d[order]
            better = dd < dist[ii]
            new[ii[better]] = owner[jj[better]]; dist[ii[better]] = dd[better]
        owner = new
    shell = np.flatnonzero((owner >= 0) & (body < 0))
    keep = np.r_[material, shell]
    return keep, owner[keep]


def check_air_removal(run, cells_npz, out):
    model, data = load(run)
    cells = np.load(cells_npz)
    rows = []
    with torch.no_grad():
        full = [render(model, cam) for cam in data.cameras]
        geometry = np.load(Path(cells_npz).parent/'geometry.npz')
        keep, _ = air_shell(model, cells["source_index"], geometry["body_of_cell"], rings=RINGS)
        print('material', len(cells['source_index']), 'shell', len(keep)-len(cells['source_index']), 'of', len(model.points))
        keep_cells(model, keep)
        mat = [render(model, cam) for cam in data.cameras]
    for k, ((rf, af), (rm, am)) in enumerate(zip(full, mat)):
        gt_alpha = data.alphas[k].cuda() if hasattr(data, 'alphas') else None
        mask_gt = (gt_alpha > .5) if gt_alpha is not None else None
        a_f, a_m = af.squeeze(-1) > .5, am.squeeze(-1) > .5
        iou = lambda a, b: float((a & b).sum()/(a | b).sum())
        rgb_gt = data.rgbs[k].cuda()
        psnr = lambda x: float(-10*torch.log10(((x.clamp(0, 1)-rgb_gt)**2).mean()))
        rows.append(dict(view=k, iou_full_vs_kept=iou(a_f, a_m),
                         iou_full_vs_gt=iou(a_f, mask_gt) if mask_gt is not None else None,
                         iou_kept_vs_gt=iou(a_m, mask_gt) if mask_gt is not None else None,
                         psnr_full=psnr(rf), psnr_kept=psnr(rm)))
        if k in (0, 4):
            im = torch.cat((rf.clamp(0, 1), rm.clamp(0, 1)), 1).cpu().numpy()
            Image.fromarray((im*255).astype(np.uint8)).save(Path(out)/f'air_removal_view{k}.png')
    summary = {key: float(np.mean([r[key] for r in rows])) for key in rows[0] if key != 'view' and rows[0][key] is not None}
    (Path(out)/'air_removal.json').write_text(json.dumps(dict(summary=summary, views=rows), indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--run', default='output/pacnerf_elastic0_f0_v1')
    a.add_argument('--cells', default='output/pacnerf_dem/elastic_0_f0/cells.npz')
    a.add_argument('--out', default='output/pacnerf_dem/elastic_0_f0/render')
    args = a.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    check_air_removal(args.run, args.cells, args.out)
