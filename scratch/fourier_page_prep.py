"""the page of the ten Fourier elastic forwards (11-07): per scene and frame the cameras 0 and 3 rows of
fourier_render.py's frame_<k>.png (PAC-NeRF / MPM, ours, ours + MPM outline), each panel 300 px, JPEG ->
out_inverse/fourier_page/img/e<s>_v<c>_<k>.jpg; data.json: per scene E, nu, beta, run time, per frame the IoU per
camera (0, 3, 6, 10), the centre of mass (ours: cells by volume; MPM: particles, equal weight; scene frame) height above
the ground and horizontal offset ours - MPM.
  python scratch/fourier_page_prep.py"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import free_surface.lion_inverse as li

OUT = ROOT/'out_inverse'/'fourier_page'; (OUT/'img').mkdir(parents=True, exist_ok=True)
OFF = np.array([.5, -.05, .5]); C0 = -.05
VROW = {0: 0, 3: 1}                                                                 # (the cameras' rows in frame_<k>.png: views 0 3 6 10)
scenes = []
for s in range(10):
    name = 'fourier_elastic%d' % s
    li.set_scene(name)
    d = ROOT/'out_inverse'/'fourier_forward'/name
    simf = d/('sim_K14.npz' if s == 1 else 'sim_K13.npz')
    z = np.load(simf); st = json.loads(str(z['stats']))
    V = np.load(li.FOAM/'dem_exact_single'/'geometry.npz')['masses']
    iou = json.load(open(d/'render'/'iou.json'))['iou']
    frames = []
    for k in range(14):
        im = Image.open(d/'render'/('frame_%02d.png' % k)).convert('RGB')
        W3, H4 = im.size; w, h = W3//3, H4//4
        for c, row in VROW.items():
            im.crop((0, row*h, W3, (row+1)*h)).resize((900, 300), Image.LANCZOS).save(OUT/'img'/('e%d_v%d_%02d.jpg' % (s, c, k)), quality=82)
        x = z['x'][k].astype(np.float64); cm = (x*V[:, None]).sum(0)/V.sum()
        v = PlyData.read(str(Path(li.SCENE['mpm'])/('%d.ply' % k)))['vertex']
        p = np.stack([v['x'], v['y'], v['z']], 1).astype(np.float64)+OFF; cg = p.mean(0)
        frames.append(dict(iou=[round(u, 3) for u in iou[k]], h=round(cm[1]-C0, 4), h_mpm=round(cg[1]-C0, 4),
                           dxz=round(1e3*float(np.hypot(*(cm-cg)[[0, 2]])), 1)))
    scenes.append(dict(scene=s, E=li.GT['E'], nu=li.GT['nu'], beta=li.BETA_MIN, v0=[round(u, 3) for u in li.V0],
                       run=round(st['seconds']), build=round(st['build']), cells=len(V), frames=frames))
    I = np.array(iou)
    print('scene %d: IoU %.3f / %.3f / %.3f; final height ours %.3f MPM %.3f, horizontal offset %.1f mm' % (
        s, I[0].mean(), I[1:6].mean(), I[6:14].mean(), frames[-1]['h'], frames[-1]['h_mpm'], frames[-1]['dxz']), flush=True)
json.dump(dict(scenes=scenes, fps=24), open(OUT/'data.json', 'w'))
print('wrote %s (%d images)' % (OUT, len(list((OUT/'img').iterdir()))))
