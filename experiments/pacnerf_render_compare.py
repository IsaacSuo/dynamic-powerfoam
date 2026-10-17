"""Rigid vs affine (PowerSim-style) primitive updates on DEM trajectories of PAC-NeRF elastic/0.

For the DEM before / after (3D-trajectory) training, render all views and captured frames with
both update operators and score silhouette IoU and photometric PSNR of the object composited
over the view's object-free frame (r_<v>_-1) against the captured frame. Writes a GIF of view 0.
"""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from experiments.foam_deform_render import DeformedFoam
from experiments.foam_dynamic_render import air_shell, keep_cells, load, render
from experiments.pacnerf_dem_forward import build
from experiments.pacnerf_render_dem import body_rotation
from experiments.pacnerf_to_powerfoam import object_mask

DEM = Path('output/pacnerf_dem/elastic_0_f0')
FRAMES = 13
MODES = ('rigid', 'affine', 'blend')
RUNS = lambda report: (('after', report['best']['parameters']),)


def main():
    gt = json.loads(Path('data/pacnerf/gt/elastic.json').read_text())['0']
    geometry = dict(np.load(DEM/'geometry.npz')); cells = dict(np.load(DEM/'cells.npz'))
    offset = json.loads(Path('data/pacnerf_powerfoam/elastic_0_f0/conversion.json').read_text())['offset']
    foam, data = load('output/pacnerf_elastic0_f0_v1')
    keep, owner = air_shell(foam, cells['source_index'], geometry['body_of_cell'], rings=2)
    keep_cells(foam, keep)
    for name in ('radii', 'density', 'texel_sites', 'texel_sv_axis', 'texel_sv_rgb', 'texel_height', 'quaternions'):
        getattr(foam, name).requires_grad_(False)
    deformed = DeformedFoam(foam, owner, geometry['points'], offset)
    deformed.prepare_blend(geometry['points'])
    order = [int(Path(f['file_path']).stem.split('_')[1]) for f in
             json.loads(Path('data/pacnerf_powerfoam/elastic_0_f0/transforms_all.json').read_text())['frames']]
    src = Path('data/pacnerf/data/elastic/0/data')
    cap, bg, masks = {}, {}, {}
    for c, v in enumerate(order):
        bg[c] = torch.tensor(np.asarray(Image.open(src/f'r_{v}_-1.png').convert('RGB'), np.float32)/255, device='cuda')
        for k in range(FRAMES+1):
            img = np.asarray(Image.open(src/f'r_{v}_{k}.png').convert('RGB'), np.float32)/255
            cap[k, c] = torch.tensor(img, device='cuda')
            masks[k, c] = torch.tensor(object_mask(img.astype(float), bg[c].cpu().numpy().astype(float)), device='cuda')
    report = json.loads((DEM/'train_v2/report.json').read_text())
    model = build(geometry, gt['E'], gt['nu'], 'cuda', 'noslip', .25, .3, torch.float32)
    x0, q0, _, w0 = model.initial_state()
    v0 = torch.tensor(gt['v'], dtype=model.dtype, device='cuda').expand(model.n, 3).clone()
    results, gif = {}, {}
    for tag, params in RUNS(report):
        theta = torch.tensor(params, dtype=model.dtype, device='cuda').expand(model.n, 7)
        sub = int(np.ceil(1/24/model.stable_dt(theta)))
        with torch.no_grad():
            pos, quat, _ = model.rollout(theta, (x0, q0, v0, w0), 1/24/sub, sub, FRAMES, compile=True, quaternions=True)
        for mode in MODES:
            iou = np.zeros((FRAMES+1, len(order))); psnr = np.zeros_like(iou)
            for k in range(FRAMES+1):
                with torch.no_grad():
                    deformed.apply(pos[k], body_R=body_rotation(quat[k]), body_F=model.deformation_gradient(pos[k]), mode=mode)
                    for c, cam in enumerate(data.cameras):
                        rgb, alpha = render(foam, cam)
                        a_ = alpha.reshape(*alpha.shape[:2], 1).clamp(0, 1)
                        comp = rgb.clamp(0, 1)+(1-a_)*bg[c]
                        m, g = a_[..., 0] > .5, masks[k, c]
                        iou[k, c] = float((m & g).sum()/(m | g).sum())
                        psnr[k, c] = float(-10*torch.log10(((comp-cap[k, c])**2).mean()))
                        if c == order.index(0):
                            gif.setdefault((tag, mode), []).append(comp.cpu().numpy())
            results[f'{tag}_{mode}'] = dict(iou_frames_1_10=float(iou[1:11].mean()), iou_frames_11_13=float(iou[11:].mean()),
                                            psnr_frames_1_10=float(psnr[1:11].mean()), psnr_frames_11_13=float(psnr[11:].mean()),
                                            psnr_frame0=float(psnr[0].mean()))
            print(tag, mode, json.dumps(results[f'{tag}_{mode}']), flush=True)
    (DEM/'render/render_compare_blend.json').write_text(json.dumps(results, indent=1))
    frames = []
    for k in range(FRAMES+1):
        row = [cap[k, order.index(0)].cpu().numpy()]+[gif['after', m][k] for m in MODES]
        frames.append(Image.fromarray((np.concatenate(row, 1).clip(0, 1)*255).astype(np.uint8)))
    out = Path('output/report_v2/pacnerf_elastic0_渲染方式对比.gif')
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=[300]*FRAMES+[1200], loop=0)
    print(out)


if __name__ == '__main__':
    main()
