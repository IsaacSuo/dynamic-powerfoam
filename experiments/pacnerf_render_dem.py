"""Render the reconstructed Power Foam driven by a DEM trajectory and compare with the video.

Kept cells: material cells + a two-ring air shell (foam_dynamic_render.air_shell), each moving
rigidly with its DEM body: p(t) = x_b(t) + R_b(t) (p0 - X_b), orientation q_b(t) q0. For every
frame and view, the rendered alpha (> 0.5) is compared with the object mask of the captured
image (pacnerf_to_powerfoam.object_mask against the object-free frame) by IoU. Reference rows:
the foam left at rest (no motion) and the DEM before / after training. Writes a GIF of one view.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from experiments.foam_dynamic_render import air_shell, keep_cells, load, move, render
from experiments.pacnerf_dem_forward import build
from experiments.pacnerf_to_powerfoam import object_mask


def body_rotation(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y), 2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), -1).reshape(q.shape[:-1]+(3, 3))


def dem_trajectory(geometry, gt, theta7, frames, alpha=.25, damping=.3):
    model = build(geometry, gt['E'], gt['nu'], 'cuda', 'noslip', alpha, damping, torch.float32)
    theta = torch.tensor(theta7, dtype=model.dtype, device='cuda').expand(model.n, 7)
    substeps = int(np.ceil(1/24/model.stable_dt(theta)))
    x0, q0, _, w0 = model.initial_state()
    v0 = torch.tensor(gt['v'], dtype=model.dtype, device='cuda').expand(model.n, 3).clone()
    with torch.no_grad():
        pos, quat, _ = model.rollout(theta, (x0, q0, v0, w0), 1/24/substeps, substeps, frames, compile=True, quaternions=True)
    return pos.double(), quat.double()


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--run', default='output/pacnerf_elastic0_f0_v1')
    a.add_argument('--dem', type=Path, default=Path('output/pacnerf_dem/elastic_0_f0'))
    a.add_argument('--scene', default='elastic/0')
    a.add_argument('--frames', type=int, default=13, help='captured motion frames r_v_1..r_v_13')
    a.add_argument('--gif-view', type=int, default=0)
    args = a.parse_args()
    gt = json.loads(Path('data/pacnerf/gt/elastic.json').read_text())[args.scene.split('/')[1]]
    geometry = dict(np.load(args.dem/'geometry.npz'))
    cells = dict(np.load(args.dem/'cells.npz'))
    offset = np.array(json.loads(Path('data/pacnerf_powerfoam/elastic_0_f0/conversion.json').read_text())['offset'])
    foam, data = load(args.run)
    keep, owner = air_shell(foam, cells['source_index'], geometry['body_of_cell'], rings=2)
    keep_cells(foam, keep)
    rest = foam.points.data.double()
    X = torch.tensor(geometry['points']+offset, device='cuda')            # body centres in scene coords
    owner_t = torch.tensor(owner, device='cuda')
    lever = rest-X[owner_t]
    offset_t = torch.tensor(offset, device='cuda')
    report = json.loads((args.dem/'train_v2/report.json').read_text())
    runs = dict(rest=None, before=[0.]*7, after=report['best']['parameters'])
    src = Path('data/pacnerf/data')/args.scene/'data'
    masks = {}
    for k in range(args.frames+1):
        for v in range(len(data.cameras)):
            img = np.asarray(Image.open(src/f'r_{v}_{k}.png').convert('RGB'), float)/255
            bg = np.asarray(Image.open(src/f'r_{v}_-1.png').convert('RGB'), float)/255
            masks[k, v] = torch.tensor(object_mask(img, bg), device='cuda')
    # dataset cameras are ordered as transforms_all.json (file names r_<v>_0)
    order = [int(Path(f['file_path']).stem.split('_')[1]) for f in json.loads(Path('data/pacnerf_powerfoam/elastic_0_f0/transforms_all.json').read_text())['frames']]
    results, frames_gif = {}, {}
    for name, theta in runs.items():
        if theta is not None:
            pos, quat = dem_trajectory(geometry, gt, theta, args.frames)
        ious = np.zeros((args.frames+1, len(order)))
        for k in range(args.frames+1):
            with torch.no_grad():
                if theta is None:
                    move(foam, rest.float())
                else:
                    R = body_rotation(quat[k])[owner_t]
                    p = pos[k][owner_t]+offset_t
                    move(foam, (p+(R@lever[..., None])[..., 0]).float(), quat[k][owner_t].float())
                for c, cam in enumerate(data.cameras):
                    rgb, alpha = render(foam, cam)
                    m = alpha.squeeze(-1) > .5
                    g = masks[k, order[c]]
                    ious[k, c] = float((m & g).sum()/(m | g).sum())
                    if c == order.index(args.gif_view):
                        frames_gif.setdefault(name, []).append((rgb.clamp(0, 1).cpu().numpy(), m.cpu().numpy()))
        results[name] = dict(iou_per_frame=ious.mean(1).tolist(), iou_mean_frames_1_on=float(ious[1:].mean()),
                             iou_train_frames_1_10=float(ious[1:11].mean()), iou_test_frames_11_on=float(ious[11:].mean()))
        print(name, json.dumps({k: v for k, v in results[name].items() if k != 'iou_per_frame'}), flush=True)
    out = args.dem/'render'
    out.mkdir(exist_ok=True)
    (out/'dem_render_iou.json').write_text(json.dumps(results, indent=1))
    # GIF: captured frame | DEM before | DEM after (render), with mask outlines of the capture.
    images = []
    for k in range(args.frames+1):
        cap = np.asarray(Image.open(src/f'r_{args.gif_view}_{k}.png').convert('RGB'), float)/255
        tiles = [cap]
        for name in ('before', 'after'):
            rgb, m = frames_gif[name][k]
            tiles.append(np.where(m[..., None], rgb, .5*cap))
        images.append(Image.fromarray((np.concatenate(tiles, 1)*255).astype(np.uint8)))
    gif = Path('output/report_v2/pacnerf_elastic0_渲染对比.gif')
    images[0].save(gif, save_all=True, append_images=images[1:], duration=[300]*args.frames+[1200], loop=0)
    print(gif)


if __name__ == '__main__':
    main()
