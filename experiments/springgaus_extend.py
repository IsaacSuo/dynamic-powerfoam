"""Frames after the 20-frame Spring-Gaus sequence (held-out future frames) for evaluation.

The sequence ends right after the bounce, but the videos (videos_compress, 120 fps) continue. This script decodes the
frames that follow the sequence (sequential decoding from just before the sequence, so frame indices match
extract_video.py), undistorts them like springgaus_dynamic.prepare, and makes object masks by background difference
against the object-free frame of each camera (springgaus_ground.object_mask, rest/background_<cam>.png).

Checks written to <out>/check.json:
- frame alignment: the decoded frame of the last sequence frame against the dataset jpg (mean absolute difference);
- mask quality: background-difference masks of the 20 sequence frames against the dataset masks (IoU per camera),
  i.e. how far the new masks can be trusted relative to the training masks.

Output: data/springgaus_powerfoam/<obj>_dynamic_ext/{images,masks}/<cam>/<kk>.png for k = 0..N-1 (0..19 are copies of
the training frames, 20.. are new), so frame k means the same instant as in <obj>_dynamic.
"""
import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from experiments.springgaus_dynamic import CAMS, cameras
from experiments.springgaus_ground import object_mask

SRC = Path('data/springgaus/real_capture/dynamic')
OUT = Path('data/springgaus_powerfoam')


def decode_range(cam, first, count):
    """Frames first .. first+count-1 of the video by sequential decoding (seek a few frames earlier, then read)."""
    v = cv2.VideoCapture(str(SRC/'videos_compress'/f'{cam}.MP4'))
    start = max(first-240, 0)
    v.set(cv2.CAP_PROP_POS_FRAMES, start)
    pos = int(v.get(cv2.CAP_PROP_POS_FRAMES))
    out = []
    while pos < first+count:
        ok, im = v.read()
        assert ok, (cam, pos)
        if pos >= first:
            out.append(im)
        pos += 1
    v.release()
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--obj', default='burger')
    p.add_argument('--frames', type=int, default=50, help='total frames (sequence frames 0..19 included)')
    a = p.parse_args()
    seq = json.loads((SRC/'sequences'/a.obj/'0.json').read_text())
    dyn = OUT/f'{a.obj}_dynamic'
    dst = OUT/f'{a.obj}_dynamic_ext'
    cams = cameras('calib')
    check = dict(alignment={}, mask_iou={})
    for c in CAMS:
        (dst/'images'/c).mkdir(parents=True, exist_ok=True); (dst/'masks'/c).mkdir(parents=True, exist_ok=True)
        base = int(Path(seq[c][0]).stem)
        n_seq = len(seq[c])
        frames = decode_range(c, base, a.frames)
        # alignment: decoded vs dataset jpg of the last sequence frame (and its neighbours, to see an off-by-one)
        ref = cv2.imread(str(SRC/'videos_images'/c/seq[c][-1])).astype(np.float32)
        check['alignment'][c] = {str(d): float(np.abs(frames[n_seq-1+d].astype(np.float32)-ref).mean()) for d in (-1, 0, 1)}
        bg = cv2.imread(str(dyn/'rest'/f'background_{c}.png'))
        und = lambda im: cv2.undistort(im, cams[c]['K_raw'], cams[c]['dist'], None, cams[c]['K'])
        ious = []
        for k in range(a.frames):
            if k < n_seq:                               # training frames: copy, and score the new masks against them
                shutil.copy(dyn/'images'/c/f'{k:02d}.png', dst/'images'/c/f'{k:02d}.png')
                shutil.copy(dyn/'masks'/c/f'{k:02d}.png', dst/'masks'/c/f'{k:02d}.png')
                mine = object_mask(und(frames[k]), bg)
                ref_m = cv2.imread(str(dyn/'masks'/c/f'{k:02d}.png'), cv2.IMREAD_GRAYSCALE) > 127
                ious.append(float((mine & ref_m).sum()/max((mine | ref_m).sum(), 1)))
            else:
                im = und(frames[k])
                cv2.imwrite(str(dst/'images'/c/f'{k:02d}.png'), im)
                cv2.imwrite(str(dst/'masks'/c/f'{k:02d}.png'), object_mask(im, bg).astype(np.uint8)*255)
        check['mask_iou'][c] = dict(mean=float(np.mean(ious)), min=float(np.min(ious)), per_frame=[round(x, 4) for x in ious])
        print(c, 'alignment (mean |decoded - jpg| at offsets -1, 0, +1):', {k_: round(v_, 2) for k_, v_ in check['alignment'][c].items()},
              '| background-difference mask IoU vs dataset masks, frames 0-%d: mean %.4f min %.4f' % (n_seq-1, np.mean(ious), np.min(ious)), flush=True)
    meta = json.loads((dyn/'cameras.json').read_text()); meta['frames'] = a.frames
    (dst/'cameras.json').write_text(json.dumps(meta, indent=1))
    (dst/'check.json').write_text(json.dumps(check, indent=1))
    print('wrote', dst, flush=True)


if __name__ == '__main__':
    main()
