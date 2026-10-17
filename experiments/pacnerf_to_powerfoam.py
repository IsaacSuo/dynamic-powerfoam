"""Convert one PAC-NeRF frame (all views) into a Power Foam Blender-format scene.

Alpha comes from background subtraction against each view's object-free frame (r_<v>_-1):
colour difference AND chromaticity change (rejects the object's same-hue shadow), then
opening, closing, hole filling and the largest component. Cameras keep PAC-NeRF's metric
world (OpenGL convention, y up, ground y = 0), shifted by `offset` so the simulation box
xyz_min..xyz_max maps into Power Foam's random_bounded init cube [0, 1]^3; no scaling, so
physical units are unchanged. Sanity check: the box centre projects inside every mask.
--full-scene keeps the background (alpha = 1) for a whole-scene foam; masks/ holds the object masks.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

SRC = Path('data/pacnerf/data')
OUT = Path('data/pacnerf_powerfoam')


def object_mask(image, background, colour=.06, chroma=.03):
    diff = np.abs(image-background).max(-1)
    chrom = lambda a: a/(a.sum(-1, keepdims=True)+1e-3)
    mask = (diff > colour) & (np.abs(chrom(image)-chrom(background)).max(-1) > chroma)
    mask = ndimage.binary_opening(mask, iterations=1)
    mask = ndimage.binary_fill_holes(ndimage.binary_closing(mask, iterations=2))
    labels, count = ndimage.label(mask)
    if count:
        sizes = ndimage.sum(mask, labels, range(1, count+1))
        mask = labels == 1+int(np.argmax(sizes))
    return mask


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', default='elastic/0')
    p.add_argument('--frame', type=int, default=0)
    p.add_argument('--box-min', type=float, nargs=3, default=[-.5, .1, -.5], help='PAC-NeRF config xyz_min')
    p.add_argument('--box-max', type=float, nargs=3, default=[.5, 1., .5], help='PAC-NeRF config xyz_max')
    p.add_argument('--full-scene', action='store_true', help='keep the whole image (alpha = 1); object masks saved to masks/')
    args = p.parse_args()
    src = SRC/args.scene
    out = OUT/(f"{args.scene.replace('/', '_')}_f{args.frame}"+('_full' if args.full_scene else ''))
    (out/'images').mkdir(parents=True, exist_ok=True)
    (out/'masks').mkdir(exist_ok=True)
    box_min, box_max = np.array(args.box_min), np.array(args.box_max)
    extent = box_max-box_min
    assert np.all(extent <= 1.), 'box larger than the unit init cube; scaling would change units'
    offset = .5-(box_min+box_max)/2  # box centre -> (0.5, 0.5, 0.5)
    entries = json.loads((src/'all_data.json').read_text())
    frames, checks = [], {}
    angle = None
    for e in entries:
        name = Path(e['file_path']).stem              # r_<view>_<frame>
        _, view, frame = name.split('_')
        if int(frame) != args.frame:
            continue
        K = np.array(e['intrinsic'])
        W, H = int(round(2*K[0, 2])), int(round(2*K[1, 2]))
        angle = 2*math.atan(.5*W/K[0, 0])
        image = np.asarray(Image.open(src/'data'/f'{name}.png').convert('RGB'), float)/255
        background = np.asarray(Image.open(src/'data'/f'r_{view}_-1.png').convert('RGB'), float)/255
        mask = object_mask(image, background)
        Image.fromarray((mask*255).astype(np.uint8)).save(out/'masks'/f'{name}.png')
        alpha = np.ones_like(mask, float) if args.full_scene else mask.astype(float)
        rgba = np.concatenate([image, alpha[..., None]], -1)
        Image.fromarray((rgba*255).round().astype(np.uint8), 'RGBA').save(out/'images'/f'{name}.png')
        c2w = np.eye(4); c2w[:3] = np.array(e['c2w'])
        # Box centre projected with the OpenGL convention (camera looks along -z).
        centre = (box_min+box_max)/2
        cam = np.linalg.inv(c2w)@np.append(centre, 1)
        u = K[0, 0]*cam[0]/-cam[2]+K[0, 2]; v = -K[1, 1]*cam[1]/-cam[2]+K[1, 2]
        ys, xs = np.nonzero(mask)
        checks[name] = dict(projected=[round(u, 1), round(v, 1)], mask_centroid=[round(xs.mean(), 1), round(ys.mean(), 1)],
                            inside=bool(mask[int(np.clip(v, 0, H-1)), int(np.clip(u, 0, W-1))]), mask_pixels=int(mask.sum()))
        shifted = c2w.copy(); shifted[:3, 3] += offset
        frames.append(dict(file_path=f'images/{name}', transform_matrix=shifted.tolist()))
    meta = dict(camera_angle_x=angle, w=W, h=H, frames=frames)
    for split in ('all', 'train', 'test'):
        (out/f'transforms_{split}.json').write_text(json.dumps(meta, indent=1))
    info = dict(source=str(src), frame=args.frame, offset=offset.tolist(), box_min=box_min.tolist(),
                box_max=box_max.tolist(), ground_y_in_scene=float(offset[1]), views=len(frames), checks=checks,
                note='scene = PAC-NeRF world + offset; OpenGL c2w; ground plane y = ground_y_in_scene.')
    (out/'conversion.json').write_text(json.dumps(info, indent=1))
    bad = [k for k, c in checks.items() if not c['inside']]
    print(out, len(frames), 'views; box centre outside mask in', bad or 'none')
    for k, c in sorted(checks.items()):
        print(k, c)


if __name__ == '__main__':
    main()
