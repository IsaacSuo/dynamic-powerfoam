"""Vid2Sim GSO test object, frame 0 (12 views, RGBA a_<v>_0.png) -> Power Foam Blender scene.

Cameras: transforms_train.json (NeRF format, OpenGL c2w, centred principal point). The visual hull of
the 12 masks (grid carve, inside every view) gives the object box; the scene is the world shifted by
`offset` so the box centre is at (0.5, 0.5, 0.5) (no scaling: the box must fit in the unit cube).
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

SRC = Path('data/vid2sim/ext')
OUT = Path('data/vid2sim_powerfoam')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--obj', default='backpack')
    a = p.parse_args()
    src = SRC/a.obj
    meta = json.loads((src/'transforms_train.json').read_text())
    frames = meta['frames']
    imgs = [np.asarray(Image.open(src/'data'/f"a_{Path(f['file_path']).stem.split('_')[1]}_0.png").convert('RGBA')) for f in frames]
    H, W = imgs[0].shape[:2]
    focal = .5*W/math.tan(.5*meta['camera_angle_x'])
    ax = np.linspace(-1.2, 1.2, 160)
    grid = np.stack(np.meshgrid(ax, ax, ax, indexing='ij'), -1).reshape(-1, 3)
    inside = np.ones(len(grid), bool)
    for f, im in zip(frames, imgs):
        c2w = np.array(f['transform_matrix'])
        cam = (np.c_[grid, np.ones(len(grid))]@np.linalg.inv(c2w).T)[:, :3]
        z = -cam[:, 2]
        u = focal*cam[:, 0]/z+W/2; v = -focal*cam[:, 1]/z+H/2
        ok = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        hit = np.zeros(len(grid), bool)
        hit[ok] = im[v[ok].astype(int), u[ok].astype(int), 3] > 127
        inside &= hit
    hull = grid[inside]
    lo, hi = hull.min(0), hull.max(0)
    assert (hi-lo).max() < .9, ('object larger than the unit cube', hi-lo)
    offset = .5-(lo+hi)/2
    out = OUT/f'{a.obj}_f0'
    (out/'images').mkdir(parents=True, exist_ok=True)
    fr = []
    for f, im in zip(frames, imgs):
        name = 'r_'+Path(f['file_path']).stem.split('_')[1]
        Image.fromarray(im, 'RGBA').save(out/'images'/f'{name}.png')
        c2w = np.array(f['transform_matrix']); c2w[:3, 3] += offset
        fr.append(dict(file_path=f'images/{name}', transform_matrix=c2w.tolist()))
    m = dict(camera_angle_x=meta['camera_angle_x'], w=W, h=H, frames=fr)
    for split in ('all', 'train', 'test'):
        (out/f'transforms_{split}.json').write_text(json.dumps(m, indent=1))
    info = dict(source=str(src), offset=offset.tolist(), hull_min=lo.tolist(), hull_max=hi.tolist(), hull_voxels=int(inside.sum()))
    (out/'conversion.json').write_text(json.dumps(info, indent=1))
    print(json.dumps(info))


if __name__ == '__main__':
    main()
