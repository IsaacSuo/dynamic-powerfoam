"""Automatic free-flight segment: the frames before the first ground contact, from the silhouettes alone.

Before any contact the object only feels gravity: its centre of mass follows a parabola (constant acceleration) and
it is rigid, so the silhouette centroid follows a smooth (near-quadratic, perspective) curve and its area and aspect
ratio change only slowly (rotation, perspective). For each camera, frames are added one at a time: a quadratic fitted
to the centroid over the frames so far predicts the next one, and the aspect ratio and area are compared with their
linear trend. The first frame where the prediction error exceeds `k` times the noise (the fit residual, floored) in
any camera is the contact frame; the free flight ends `margin` frames before it (the frame just before contact may
already be decelerating or motion-blurred into the contact).
"""
import numpy as np


def moments(m):
    m = np.asarray(m, np.float64)
    H, W = m.shape
    ys, xs = np.mgrid[0:H, 0:W]
    A = m.sum()
    if A <= 0:
        return np.full(4, np.nan)
    cx, cy = (m*xs).sum()/A, (m*ys).sum()/A
    sx = np.sqrt((m*(xs-cx)**2).sum()/A); sy = np.sqrt((m*(ys-cy)**2).sum()/A)
    return np.array([cx, cy, np.sqrt(A), sy/sx])


def detect(masks, k=6., margin=1, min_frames=4, floor_px=.3, floor_aspect=.01):
    """masks: (frames, cams, H, W) in [0, 1]. Returns dict(last_flight, contact, per-camera scores)."""
    F, C = masks.shape[:2]
    feats = np.array([[moments(masks[f, c]) for c in range(C)] for f in range(F)])     # (F, C, 4)
    contact = F
    detail = {}
    for c in range(C):
        for n in range(min_frames, F):
            t = np.arange(n, dtype=float)
            bad = []
            for j, floor in ((0, floor_px), (1, floor_px)):            # centroid: quadratic
                y = feats[:n, c, j]
                p = np.polyfit(t, y, 2)
                res = y-np.polyval(p, t)
                sig = max(res.std()*np.sqrt(n/max(n-3, 1)), floor)
                bad.append(abs(feats[n, c, j]-np.polyval(p, n))/sig)
            for j, floor in ((2, floor_px), (3, floor_aspect)):        # size and aspect: linear trend
                y = feats[:n, c, j]
                p = np.polyfit(t, y, 1)
                res = y-np.polyval(p, t)
                sig = max(res.std()*np.sqrt(n/max(n-2, 1)), floor)
                bad.append(abs(feats[n, c, j]-np.polyval(p, n))/sig)
            if max(bad) > k:
                detail[c] = dict(contact=n, scores=np.round(bad, 1).tolist())
                contact = min(contact, n)
                break
    last = contact-1-margin
    return dict(contact=int(contact), last_flight=int(last), flight_frames=list(range(last+1)), per_camera=detail,
                enough=bool(last+1 >= min_frames))


if __name__ == '__main__':
    import argparse
    import json
    from pathlib import Path

    from PIL import Image
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dyn', type=Path, default=Path('data/springgaus_powerfoam/burger_dynamic'))
    p.add_argument('--downsample', type=int, default=2)
    a = p.parse_args()
    meta = json.loads((a.dyn/'cameras.json').read_text())
    cams = sorted(meta['cameras'])
    W, H = meta['width']//a.downsample, meta['height']//a.downsample
    m = np.array([[np.asarray(Image.open(a.dyn/'masks'/cn/f'{f:02d}.png').convert('L').resize((W, H), Image.BILINEAR), np.float32)/255
                   for cn in cams] for f in range(meta['frames'])])
    r = detect(m)
    print(json.dumps(dict(r, cameras=cams, metadata_hit_frame=meta.get('hit_frame')), indent=1))
