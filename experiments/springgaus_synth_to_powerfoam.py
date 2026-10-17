"""Convert one frame of a Spring-Gaus MPM-synthetic object (10 Blender views, RGBA) into a Power Foam
Blender-format scene.

Cameras: camera.json c2w is the Blender/OpenGL convention (Spring-Gaus flips columns 1-2 to get
OpenCV), K has the principal point at the image centre. The objects already lie inside [0, 1]^3
(Spring-Gaus XYZ_MIN/MAX ~ [0.3, 0.8]), so the scene is the simulation world itself (offset 0,
metres, gravity -y, ground plane y = physical.json BC). Alpha = the rendered alpha (exact mask).
Sanity check: the GT particle centroid of that frame projects inside every mask.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

SRC = Path('data/springgaus/mpm_synthetic_ext/mpm_synthetic')
OUT = Path('data/springgaus_synth_powerfoam')


def read_ply(path):
    b = open(path, 'rb').read()
    h = b.index(b'end_header\n')+11
    return np.frombuffer(b[h:], dtype=np.float32).reshape(-1, 6)[:, :3]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--obj', default='torus')
    p.add_argument('--frame', type=int, default=0)
    a = p.parse_args()
    render = SRC/'render'/a.obj
    out = OUT/f'{a.obj}_f{a.frame}'
    (out/'images').mkdir(parents=True, exist_ok=True)
    cams = json.loads((render/'camera.json').read_text())
    phys = json.loads((render/'physical.json').read_text())
    centre = read_ply(SRC/'simulation'/a.obj/f'{a.frame:03d}.ply').mean(0)
    frames, checks = [], {}
    for cam in cams:
        K, c2w = np.array(cam['K']), np.array(cam['c2w'])
        name = f"{cam['camera']}"
        rgba = np.asarray(Image.open(render/name/f'{a.frame:03d}.png').convert('RGBA'))
        H, W = rgba.shape[:2]
        assert abs(K[0, 2]-W/2) < 1 and abs(K[1, 2]-H/2) < 1, 'principal point not centred'
        Image.fromarray(rgba, 'RGBA').save(out/'images'/f'{name}.png')
        x = np.linalg.inv(c2w)@np.append(centre, 1)          # OpenGL camera: looks along -z
        u = K[0, 0]*x[0]/-x[2]+K[0, 2]; v = -K[1, 1]*x[1]/-x[2]+K[1, 2]
        mask = rgba[..., 3] > 127
        ys, xs = np.nonzero(mask)
        checks[name] = dict(projected=[round(u, 1), round(v, 1)], mask_centroid=[round(xs.mean(), 1), round(ys.mean(), 1)],
                            inside=bool(mask[int(np.clip(v, 0, H-1)), int(np.clip(u, 0, W-1))]))
        frames.append(dict(file_path=f'images/{name}', transform_matrix=c2w.tolist()))
    meta = dict(camera_angle_x=2*math.atan(.5*W/K[0, 0]), w=W, h=H, frames=frames)
    for split in ('all', 'train', 'test'):
        (out/f'transforms_{split}.json').write_text(json.dumps(meta, indent=1))
    info = dict(source=str(render), frame=a.frame, offset=[0., 0., 0.], physical=phys, gt_centroid=centre.tolist(),
                ground=phys[1]['BC']['GROUND'], views=len(frames), checks=checks,
                note='scene = simulation world (offset 0); OpenGL c2w; ground plane from physical.json')
    (out/'conversion.json').write_text(json.dumps(info, indent=1))
    bad = [k for k, c in checks.items() if not c['inside']]
    print(out, len(frames), 'views; GT centroid outside mask in', bad or 'none')
    for k, c in sorted(checks.items()):
        print(k, c)


if __name__ == '__main__':
    main()
