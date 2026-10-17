"""Spring-Gaus real capture, dynamic part: undistorted masks/images and static -> dynamic registration.

prepare: the 3 synced videos (C0733, C0787, C0801; 120 fps) of one sequence, undistorted with the
chessboard calibration (cameras_calib.json: K, 5 distortion coefficients, cv2 rvec/tvec = world ->
camera) into an exact pinhole with the calibrated focal lengths and the principal point moved to
the image centre (the Power Foam rasterizer assumes a centred principal point). Saved to data/springgaus_powerfoam/
<obj>_dynamic/{masks,images}/<cam>/<frame:02d>.png and cameras.json. World: metres, gravity +z.
hull: strict static visual hull samples (hull_colmap.npy) used by register.
register: similarity x_dyn = R s (x_colmap - mean) + t (Spring-Gaus convention: euler 'XYZ' degrees,
mean over the static object), fitted so the static visual hull (hull_colmap.npy) projects onto the
frame-0 masks of all 3 views: symmetric 2D chamfer (pixels) between projected hull voxels and mask
pixels. Initial values from Spring-Gaus' configs. Reports per-view mask IoU of the splatted hull.
--intrinsics sg uses Spring-Gaus' own choice instead (static COLMAP focal scaled by 1920/4752,
principal point at the image centre, no undistortion) for comparison.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation

SRC = Path('data/springgaus/real_capture')
OUT = Path('data/springgaus_powerfoam')
CAMS = ('C0733', 'C0787', 'C0801')
INIT = dict(burger=([60, 0, 0], [.12, .29, -.21], .15), bun=([30, -90, 0], [.14, .31, -.23], .1),
            dog=([-30, -10, 180], [.13, .27, -.19], .1), pig=([90, 0, 90], [.09, .26, -.23], .1),
            potato=([180, 180, 0], [.19, .29, -.24], .1))
GROUND = dict(burger=-.025, dog=-.025, pig=-.025, bun=-.01, potato=-.01)


def cameras(intrinsics='calib', obj='burger'):
    calib = json.loads((SRC/'dynamic'/'cameras_calib.json').read_text())
    if intrinsics == 'calib':
        K_raw = np.array(calib['camera_matrix']); dist = np.array(calib['distortion_coefficient'])
        # Undistort INTO a pinhole with the principal point at the image centre (OpenCV pixel-centre
        # convention: (W-1)/2, (H-1)/2): Power Foam's rasterizer assumes a symmetric frustum
        # (camera.right/up half-extents), so an off-centre principal point (954, 515 here) would shift
        # every render by tens of pixels against the ray maps / masks.
        K = K_raw.copy(); K[0, 2], K[1, 2] = (1920-1)/2, (1080-1)/2
    else:
        import pycolmap
        fx, fy, _, _ = list(pycolmap.Reconstruction(str(SRC/'static'/'colmap'/obj)).cameras.values())[0].params
        K = np.array([[fx*1920/4752, 0, 960], [0, fy*1080/2672, 540], [0, 0, 1]]); dist = None; K_raw = K
    out = {}
    for c in CAMS:
        R, _ = cv2.Rodrigues(np.array(calib[c]['rvecs']))
        out[c] = dict(K=K, K_raw=K_raw, dist=dist, R=R, t=np.array(calib[c]['tvecs']).reshape(3))
    return out


def prepare(obj, seq=0):
    info = json.loads((SRC/'dynamic'/'sequences'/obj/f'{seq}.json').read_text())
    cams = cameras('calib')
    dst = OUT/f'{obj}_dynamic'
    for c in CAMS:
        K, K_raw, dist = cams[c]['K'], cams[c]['K_raw'], cams[c]['dist']
        for kind in ('masks', 'images'):
            (dst/kind/c).mkdir(parents=True, exist_ok=True)
        for i, name in enumerate(info[c]):
            img = cv2.imread(str(SRC/'dynamic'/'videos_images'/c/name))
            mask = cv2.imread(str(SRC/'dynamic'/'videos_masks'/c/name.replace('.jpg', '.png')), cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(str(dst/'images'/c/f'{i:02d}.png'), cv2.undistort(img, K_raw, dist, None, K))
            cv2.imwrite(str(dst/'masks'/c/f'{i:02d}.png'), cv2.undistort(mask, K_raw, dist, None, K))
    meta = dict(obj=obj, seq=seq, fps=120, hit_frame=info['hit_frame'], frames=len(info[CAMS[0]]), width=1920, height=1080,
                gravity=[0, 0, 9.8], ground_z=GROUND[obj], ground_normal=[0, 0, -1],
                cameras={c: dict(K=cams[c]['K'].tolist(), R=cams[c]['R'].tolist(), t=cams[c]['t'].tolist()) for c in CAMS})
    (dst/'cameras.json').write_text(json.dumps(meta, indent=1))
    print('prepared', dst, meta['frames'], 'frames, hit frame', meta['hit_frame'])


def hull_samples(obj, n=2000000, seed=1):
    """Strict visual hull of the static capture (inside the mask of every kept view), sampled
    uniformly in the foam cube and saved in COLMAP coordinates (hull_colmap.npy) for register."""
    from experiments.pacnerf_foam_geometry import visual_hull_colmap
    sd = OUT/obj
    conv = json.loads((sd/'conversion.json').read_text())
    x = np.random.default_rng(seed).random((n, 3))
    x = x[visual_hull_colmap(x, sd)]
    np.save(sd/'hull_colmap.npy', ((x-.5)/conv['scale']+np.array(conv['centre'])).astype(np.float32))
    print('strict hull samples', len(x))


def load_masks(obj, frame, intrinsics):
    if intrinsics == 'calib':
        return [np.asarray(Image.open(OUT/f'{obj}_dynamic'/'masks'/c/f'{frame:02d}.png')) > 127 for c in CAMS]
    info = json.loads((SRC/'dynamic'/'sequences'/obj/'0.json').read_text())
    return [np.asarray(Image.open(SRC/'dynamic'/'videos_masks'/c/info[c][frame].replace('.jpg', '.png')).convert('L')) > 127
            for c in CAMS]


def project(x, cam):
    K = torch.as_tensor(cam['K'], dtype=x.dtype); R = torch.as_tensor(cam['R'], dtype=x.dtype)
    t = torch.as_tensor(cam['t'], dtype=x.dtype)
    xc = x@R.T+t
    uv = xc[:, :2]/xc[:, 2:3]
    return uv@K[:2, :2].T+K[:2, 2], xc[:, 2]


def splat(uv, shape, r=2):
    img = np.zeros(shape, bool)
    u = uv[:, 0].round().astype(int); v = uv[:, 1].round().astype(int)
    for du in range(-r, r+1):
        for dv in range(-r, r+1):
            ok = (u+du >= 0) & (u+du < shape[1]) & (v+dv >= 0) & (v+dv < shape[0])
            img[v[ok]+dv, u[ok]+du] = True
    return img


def register(obj, frame=0, intrinsics='calib', iterations=600, seed=0):
    torch.manual_seed(seed)
    hull = torch.tensor(np.load(OUT/obj/'hull_colmap.npy'), dtype=torch.float64)
    mean = hull.mean(0)
    cams = cameras(intrinsics, obj)
    masks = load_masks(obj, frame, intrinsics)
    targets = []
    for m in masks:
        ys, xs = np.nonzero(m)
        pick = np.random.default_rng(seed).choice(len(xs), min(4000, len(xs)), replace=False)
        targets.append(torch.tensor(np.c_[xs[pick], ys[pick]]+.5, dtype=torch.float64))
    r0, t0, s0 = INIT[obj]
    R0 = torch.tensor(Rotation.from_euler('XYZ', r0, degrees=True).as_matrix())
    w = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    t = torch.tensor(t0, dtype=torch.float64, requires_grad=True)
    logs = torch.tensor(np.log(s0), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([dict(params=[w], lr=1e-2), dict(params=[t], lr=1e-3), dict(params=[logs], lr=1e-2)])

    def transform(x):
        W = torch.zeros(3, 3, dtype=x.dtype)
        W[0, 1], W[0, 2], W[1, 2] = -w[2], w[1], -w[0]
        W = W-W.T
        return (torch.exp(logs)*(x-mean))@(torch.matrix_exp(W)@R0).T+t

    def loss_fn(sub):
        x = transform(hull[sub])
        total = 0
        for c, tgt in zip(CAMS, targets):
            uv, _ = project(x, cams[c])
            d = torch.cdist(uv, tgt)
            total = total+d.min(1).values.mean()+d.min(0).values.mean()
        return total/len(CAMS)
    rng = np.random.default_rng(seed)
    log = []
    for it in range(iterations):
        sub = torch.as_tensor(rng.choice(len(hull), min(4000, len(hull)), replace=False))
        loss = loss_fn(sub)
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 100 == 0 or it == iterations-1:
            log.append((it, float(loss))); print(it, 'chamfer px %.2f' % float(loss), 's %.4f' % float(torch.exp(logs)), flush=True)
    with torch.no_grad():
        x = transform(hull)
        ious = []
        dst = OUT/f'{obj}_dynamic'/f'register_{intrinsics}'; dst.mkdir(parents=True, exist_ok=True)
        tiles = []
        for c, m in zip(CAMS, masks):
            uv, _ = project(x, cams[c])
            sil = splat(uv.numpy(), m.shape)
            ious.append(float((sil & m).sum()/(sil | m).sum()))
            over = np.zeros(m.shape+(3,), np.uint8); over[..., 0] = m*255; over[..., 1] = sil*255
            tiles.append(over)
        Image.fromarray(np.concatenate(tiles, 0)[::2, ::2]).save(dst/f'frame{frame:02d}.png')
        R = (torch.matrix_exp(torch.tensor([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]]))@R0).numpy()
        res = dict(obj=obj, frame=frame, intrinsics=intrinsics, R=R.tolist(), t=t.tolist(), s=float(torch.exp(logs)),
                   colmap_mean=mean.tolist(), iou=dict(zip(CAMS, ious)), chamfer_log=log,
                   size_m=((hull.max(0).values-hull.min(0).values)*torch.exp(logs)).tolist(),
                   note='x_dyn = R s (x_colmap - colmap_mean) + t')
        (dst/f'frame{frame:02d}.json').write_text(json.dumps(res, indent=1))
    print('IoU', {c: round(i, 3) for c, i in zip(CAMS, ious)}, 'scale', res['s'], 'size (m)', np.round(res['size_m'], 3))
    return res


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('cmd', choices=['prepare', 'hull', 'register'])
    p.add_argument('--obj', default='burger')
    p.add_argument('--frame', type=int, default=0)
    p.add_argument('--intrinsics', default='calib', choices=['calib', 'sg'])
    p.add_argument('--iterations', type=int, default=600)
    a = p.parse_args()
    if a.cmd == 'prepare':
        prepare(a.obj)
    elif a.cmd == 'hull':
        hull_samples(a.obj)
    else:
        register(a.obj, a.frame, a.intrinsics, a.iterations)
