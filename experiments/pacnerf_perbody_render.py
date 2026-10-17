"""GIF: captured video vs DEM with shared parameters vs per-body parameters (PAC-NeRF elastic/0).

Rows: a training view (0) and a held-out view (3). Columns: captured frame, shared 7 parameters
(silhouette_v1), per-body parameters (perbody_smooth1e-2). Blend rendering composited over the
view's object-free frame; frames 11-13 were never used in training.
"""
import json
from pathlib import Path

import numpy as np
import torch
import warp as wp
from PIL import Image, ImageDraw, ImageFont

from experiments.foam_deform_render import DeformedFoam
from experiments.foam_dem_warp import WarpFoam, rollout as warp_rollout
from experiments.foam_dynamic_render import air_shell, keep_cells, load, render
from experiments.pacnerf_dem_forward import build

DEM = Path('output/pacnerf_dem/elastic_0_f0')
FRAMES, VIEWS = 13, (0, 3)
FONT = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 20)


def main():
    gt = json.loads(Path('data/pacnerf/gt/elastic.json').read_text())['0']
    geometry = dict(np.load(DEM/'geometry.npz')); cells = dict(np.load(DEM/'cells.npz'))
    offset = json.loads(Path('data/pacnerf_powerfoam/elastic_0_f0/conversion.json').read_text())['offset']
    foam, data = load('output/pacnerf_elastic0_f0_v1')
    keep, owner = air_shell(foam, cells['source_index'], geometry['body_of_cell'], rings=2)
    keep_cells(foam, keep)
    deformed = DeformedFoam(foam, owner, geometry['points'], offset)
    deformed.prepare_blend(geometry['points'], K=24, width=2.)
    order = [int(Path(f['file_path']).stem.split('_')[1]) for f in
             json.loads(Path('data/pacnerf_powerfoam/elastic_0_f0/transforms_all.json').read_text())['frames']]
    src = Path('data/pacnerf/data/elastic/0/data')
    model = build(geometry, gt['E'], gt['nu'], 'cuda', 'noslip', .25, .3, torch.float32)
    wf = WarpFoam(model)
    x0, q0, _, w0 = model.initial_state()
    v0 = torch.tensor(gt['v'], dtype=model.dtype, device='cuda').expand(model.n, 3).clone()
    shared = torch.tensor(json.loads((DEM/'silhouette_v1/report.json').read_text())['best']['parameters'],
                          dtype=model.dtype, device='cuda').expand(model.n, 7)
    perbody = torch.tensor(np.load(DEM/'perbody_smooth1e-2/theta_best.npz')['theta'], dtype=model.dtype, device='cuda')
    columns = {}
    for name, theta in (('shared', shared), ('perbody', perbody)):
        sub = int(np.ceil(1/24/model.stable_dt(theta)))
        with torch.no_grad():
            pos, quat = warp_rollout(wf, theta, (x0, q0, v0, w0), 1/24/sub, sub, FRAMES)
            for v in VIEWS:
                c = order.index(v)
                bg = torch.tensor(np.asarray(Image.open(src/f'r_{v}_-1.png').convert('RGB'), np.float32)/255, device='cuda')
                for k in range(FRAMES+1):
                    deformed.apply(pos[k], body_F=model.deformation_gradient(pos[k]), mode='blend')
                    rgb, alpha = render(foam, data.cameras[c])
                    a_ = alpha.reshape(*alpha.shape[:2], 1).clamp(0, 1)
                    columns[name, v, k] = (rgb.clamp(0, 1)+(1-a_)*bg).cpu().numpy()
    labels = ('真实视频', '共用 7 个参数', '每块各学一组')
    images = []
    for k in range(FRAMES+1):
        rows = []
        for v in VIEWS:
            cap = np.asarray(Image.open(src/f'r_{v}_{k}.png').convert('RGB'), np.float32)/255
            rows.append(np.concatenate([cap, columns['shared', v, k], columns['perbody', v, k]], 1))
        im = Image.fromarray((np.concatenate(rows, 0).clip(0, 1)*255).astype(np.uint8))
        draw = ImageDraw.Draw(im)
        for c, text in enumerate(labels):
            draw.text((c*400+10, 8), text, fill=(20, 24, 32), font=FONT)
        for r, v in enumerate(VIEWS):
            tag = f'视角 {v}（训练用）' if v == 0 else f'视角 {v}（训练没用过）'
            draw.text((10, r*400+372), tag, fill=(20, 24, 32), font=FONT)
        stage = '训练帧' if k <= 10 else '没训练过的帧'
        label = f'第 {k} 帧 · {stage}'
        draw.text((1190-draw.textlength(label, font=FONT), 372), label, fill=(20, 24, 32), font=FONT)
        images.append(im)
    out = Path('output/report_v2/pacnerf_elastic0_每块各学一组.gif')
    images[0].save(out, save_all=True, append_images=images[1:], duration=[350]*FRAMES+[1500], loop=0)
    print(out)


if __name__ == '__main__':
    main()
