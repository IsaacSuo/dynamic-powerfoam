"""Measure the ground (table) plane of a Spring-Gaus dynamic capture from the full videos.

The 20-frame sequences end right after the bounce, but the videos (videos_compress, 120 fps)
continue: the object comes to rest on the table. Two independent measurements, both in the
calibrated dynamic world (metres):

extract: undistorted frames of the 3 cameras at the same instants (sync.json offsets) and object
  masks by background difference against an object-free frame of the same camera (--background)
  and saturated colour, largest component -> data/springgaus_powerfoam/<obj>_dynamic/rest/.
rest: the static foam (scale of the registration used for the DEM, or --fit-scale) is fitted to
  the rest masks and images of all 3 views (render composited over the image: L1 + silhouette L2;
  yaw scan then Adam; --upright keeps the object upright, rotation about the up axis only);
  the ground is the lowest material point along gravity, i.e. a supporting plane, reported
  with the mean height of the lowest 1% of material sites for robustness.
rim: the table top edge: a 3D circle (centre, radius, plane normal) fitted so its projection lies
  on the boundary of the bright table region in all 3 views; the circle's plane is the table.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from experiments.springgaus_dynamic import CAMS, cameras

SRC = Path('data/springgaus/real_capture/dynamic')
OUT = Path('data/springgaus_powerfoam')


def video_frame(cam, index):
    v = cv2.VideoCapture(str(SRC/'videos_compress'/f'{cam}.MP4'))
    v.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, im = v.read()
    v.release()
    assert ok, (cam, index)
    return im


def object_mask(bgr, bg):
    """Background difference (object-free frame of the same camera) AND saturated colour, then
    morphology, largest component, holes filled."""
    diff = np.abs(bgr.astype(np.int16)-bg.astype(np.int16)).max(-1)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = (diff > 40) & (hsv[..., 1] > 155)          # shadows on the table stay below ~140
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return np.zeros(m.shape, bool)
    big = 1+int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cnts, _ = cv2.findContours((lab == big).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros(m.shape, np.uint8)
    cv2.drawContours(filled, cnts, -1, 1, -1)
    return filled.astype(bool)


def extract(obj, frames_c0733, background_c0733):
    seq = json.loads((SRC/'sequences'/obj/'0.json').read_text())
    base = {c: int(Path(seq[c][0]).stem) for c in CAMS}
    cams = cameras('calib')
    dst = OUT/f'{obj}_dynamic'/'rest'
    dst.mkdir(parents=True, exist_ok=True)
    und = lambda c, f: cv2.undistort(video_frame(c, f-base['C0733']+base[c]), cams[c]['K_raw'], cams[c]['dist'], None, cams[c]['K'])
    bgs = {c: und(c, background_c0733) for c in CAMS}
    for c in CAMS:
        cv2.imwrite(str(dst/f'background_{c}.png'), bgs[c])
    for f in frames_c0733:
        for c in CAMS:
            im = und(c, f)
            cv2.imwrite(str(dst/f'{f}_{c}.png'), im)
            cv2.imwrite(str(dst/f'{f}_{c}_mask.png'), object_mask(im, bgs[c]).astype(np.uint8)*255)
    print('extracted', frames_c0733, '->', dst)


def triangulate(points2d, cams_meta):
    """Least-squares 3D point closest to the back-projected rays (pixel coordinates, pinhole)."""
    A = np.zeros((3, 3)); b = np.zeros(3)
    for (u, v), cam in zip(points2d, cams_meta):
        K, R, t = np.array(cam['K']), np.array(cam['R']), np.array(cam['t'])
        c = -R.T@t
        d = R.T@np.linalg.solve(K, [u, v, 1.]); d /= np.linalg.norm(d)
        P = np.eye(3)-np.outer(d, d)
        A += P; b += P@c
    return np.linalg.solve(A, b)


def rest_fit(obj, frames, run, dem, registration, downsample=2, iterations=200, fit_scale=False, upright=False, tag=''):
    import torch
    from scipy.spatial.transform import Rotation
    from experiments.foam_deform_render import DeformedFoam
    from experiments.foam_dynamic_render import load, render
    from experiments.pacnerf_foam_geometry import foam_to_dynamic
    from experiments.springgaus_train import foam_cameras, skew
    dyn = OUT/f'{obj}_dynamic'
    meta = json.loads((dyn/'cameras.json').read_text())
    A, b, k = foam_to_dynamic(OUT/obj, dyn/'register_calib'/registration)
    R0 = A/k
    foam, _ = load(run)
    for name in ('radii', 'density', 'texel_sites', 'texel_sv_axis', 'texel_sv_rgb', 'texel_height', 'points', 'quaternions'):
        getattr(foam, name).requires_grad_(False)
    c = foam.points.detach().mean(0)
    deformed = DeformedFoam(foam, np.zeros(foam.points.shape[0], int), c[None].cpu().numpy(), np.zeros(3))
    cams = foam_cameras(meta, A, b, k, downsample)
    W, H = cams[0].width, cams[0].height
    rest = dyn/'rest'
    load_img = lambda path, mode: np.asarray(Image.open(path).convert(mode).resize((W, H), Image.BILINEAR), np.float32)/255
    data = []
    for f in frames:
        for ci, cam in enumerate(CAMS):
            data.append((ci, torch.tensor(load_img(rest/f'{f}_{cam}.png', 'RGB'), device='cuda'),
                         torch.tensor(load_img(rest/f'{f}_{cam}_mask.png', 'L'), device='cuda')))
    # initial position: triangulated mask centroids (frame 0 of the list), in foam coordinates
    cents = []
    for ci, cam in enumerate(CAMS):
        m = np.asarray(Image.open(rest/f'{frames[0]}_{cam}_mask.png')) > 127
        ys, xs = np.nonzero(m); cents.append((xs.mean()+.5, ys.mean()+.5))
    X = triangulate(cents, [meta['cameras'][cc] for cc in CAMS])
    d0 = np.linalg.solve(A, X-b)-c.cpu().numpy()
    up_f = R0.T@(-np.array(meta['gravity'])/np.linalg.norm(meta['gravity']))
    t32 = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32, device='cuda')

    def loss_at(Q, d, sigma=None):
        if sigma is None:
            deformed.apply(c[None]+d[None], body_R=Q[None], mode='rigid')
        else:
            deformed.apply(c[None]+d[None], body_R=Q[None], body_F=(sigma*Q)[None], mode='affine')
        total = 0.
        ious = []
        for ci, img, m in data:
            rgb, alpha = render(foam, cams[ci])
            alpha = alpha.reshape(H, W)
            comp = rgb.clamp(0, 1)+(1-alpha.clamp(0, 1))[..., None]*img
            npx = m.sum().clamp(min=1)
            total = total+(comp-img).abs().mean()*W*H/npx+((alpha-m)**2).mean()*W*H/npx
            pm, qm = alpha > .5, m > .5
            ious.append(float((pm & qm).sum()/(pm | qm).sum()))
        return total/len(data), ious
    scores = []
    with torch.no_grad():
        for yaw in range(0, 360, 10):
            Q = t32(Rotation.from_rotvec(np.radians(yaw)*up_f).as_matrix())
            scores.append(float(loss_at(Q, t32(d0))[0]))
    yaw = 10*int(np.argmin(scores))
    print('yaw scan best %d deg (loss %.4f, worst %.4f)' % (yaw, min(scores), max(scores)), flush=True)
    up_t = t32(up_f)
    yaw_t = t32(np.radians(yaw)).requires_grad_(True)       # upright: rotation about up only
    w = t32(np.radians(yaw)*up_f).requires_grad_(True)       # free: full axis-angle
    d = t32(d0).requires_grad_(True)
    ls = t32(0.).requires_grad_(fit_scale)
    rot = (lambda: torch.matrix_exp(skew(yaw_t*up_t))) if upright else (lambda: torch.matrix_exp(skew(w)))
    groups = [dict(params=[yaw_t if upright else w], lr=1e-2), dict(params=[d], lr=2e-3)]
    if fit_scale:
        groups.append(dict(params=[ls], lr=3e-3))
    opt = torch.optim.Adam(groups)
    for it in range(iterations):
        opt.zero_grad()
        l, _ = loss_at(rot(), d, torch.exp(ls) if fit_scale else None)
        l.backward(); opt.step()
        if it % 50 == 0:
            print(it, 'loss %.4f sigma %.4f' % (float(l.detach()), float(torch.exp(ls))), flush=True)
    with torch.no_grad():
        Q = rot(); sigma = float(torch.exp(ls)); l, ious = loss_at(Q, d, torch.exp(ls) if fit_scale else None)
        tiles = []
        for ci, img, m in data[-len(CAMS):]:
            rgb, alpha = render(foam, cams[ci]); alpha = alpha.reshape(H, W)
            comp = (rgb.clamp(0, 1)+(1-alpha.clamp(0, 1))[..., None]*img).cpu().numpy()
            mm = m.cpu().numpy() > .5; edge = mm ^ np.roll(mm, 1, 0) | mm ^ np.roll(mm, 1, 1)
            comp[edge] = [0, 1, 0]
            ys, xs = np.nonzero(mm); cy, cx = int(ys.mean()), int(xs.mean())
            tiles.append(np.concatenate((np.pad(img.cpu().numpy(), ((80, 80), (80, 80), (0, 0)))[cy:cy+160, cx:cx+160],
                                         np.pad(comp, ((80, 80), (80, 80), (0, 0)))[cy:cy+160, cx:cx+160]), 0))
        Image.fromarray((np.concatenate(tiles, 1)*255).astype(np.uint8)).resize((3*160*3, 2*160*3)).save(dyn/'rest'/f'fit{tag}.png')
    Q, d, cn = Q.cpu().numpy().astype(float), d.detach().cpu().numpy().astype(float), c.cpu().numpy().astype(float)
    # DEM bodies at rest: rest foam pose applied to the frame-0 body centres
    g = dict(np.load(Path(dem)/'geometry.npz'))
    body_f = (g['points']-b)@np.linalg.inv(A).T
    body_rest = (cn+sigma*(body_f-cn)@Q.T+d)@A.T+b
    contact = sigma*(3*g['masses']/(4*np.pi))**(1/3)
    gdir = np.array(meta['gravity'])/np.linalg.norm(meta['gravity'])
    h = body_rest@gdir
    support = float((h+contact).max())
    tilt = float(np.degrees(np.arccos(np.clip(abs((Q@up_f)@up_f), -1, 1))))
    centre_world = A@(cn+d)+b
    proj = []
    for cc in CAMS:
        K_, R_, t_ = (np.array(meta['cameras'][cc][k_]) for k_ in ('K', 'R', 't'))
        q_ = K_@(R_@centre_world+t_); proj.append((q_[:2]/q_[2]).tolist())
    alpha_cent = []
    with torch.no_grad():
        for ci in range(len(CAMS)):
            _, al = render(foam, cams[ci]); al = al.reshape(H, W).cpu().numpy() > .5
            ys_, xs_ = np.nonzero(al); alpha_cent.append([(xs_.mean()+.5)*downsample, (ys_.mean()+.5)*downsample])
    res = dict(frames=frames, registration=registration, fit_scale=fit_scale, upright=upright, sigma=sigma,
               Q=Q.tolist(), d=d.tolist(), model_centre_world=centre_world.tolist(), model_centre_projected_px=proj,
               rendered_alpha_centroid_px=alpha_cent, body_rest_mean_world=body_rest.mean(0).tolist(),
               height_mm=float(1000*(h+contact).max()-1000*(h-contact).min()), iou=dict(zip([f'{f}_{cc}' for f in frames for cc in CAMS], ious)),
               loss=float(l), yaw_deg=yaw, tilt_from_upright_deg=tilt,
               ground_height_along_gravity_dem_contact=support,
               ground_height_lowest_1pct_mean=float(np.sort(h+contact)[-max(1, len(h)//100):].mean()),
               lowest_body_centre=float(h.max()), config_ground_z=meta['ground_z'],
               offset_vs_config=support-meta['ground_z'],
               note='heights are x . gravity_direction (dynamic world z, +z down); the DEM touches when '
                    'centre height + contact radius >= ground height')
    (dyn/'rest'/f'ground_rest{tag}.json').write_text(json.dumps(res, indent=1))
    print(json.dumps({k_: v for k_, v in res.items() if k_ != 'iou'}, indent=1))
    print('IoU', {k_: round(v, 3) for k_, v in res['iou'].items()})


def table_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 2] > 120) & (hsv[..., 1] < 90)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    m = (lab == 1+int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))


def rim_fit(obj, tilt=False):
    """3D circle of the table top: the far half of the circle (seen against the room, so it is the
    top edge, not the table's side) must project onto the table-region boundary in every view."""
    from scipy.ndimage import distance_transform_edt, map_coordinates
    from scipy.optimize import least_squares
    dyn = OUT/f'{obj}_dynamic'
    meta = json.loads((dyn/'cameras.json').read_text())
    views = []
    for c in CAMS:
        m = table_mask(cv2.imread(str(dyn/'rest'/f'background_{c}.png')))
        edge = (m ^ np.roll(m, 1, 0)) | (m ^ np.roll(m, 1, 1))
        edge[:6] = edge[-6:] = 0; edge[:, :6] = edge[:, -6:] = 0
        cam = meta['cameras'][c]
        views.append((distance_transform_edt(~edge.astype(bool)), np.array(cam['K']), np.array(cam['R']), np.array(cam['t'])))
    ang = np.linspace(0, 2*np.pi, 720, endpoint=False)

    def circle(p):
        cx, cy, z, r = p[:4]
        n = np.array([0., 0., 1.])
        if tilt:
            from scipy.spatial.transform import Rotation
            n = Rotation.from_rotvec([p[4], p[5], 0.]).apply(n)
        a = np.cross(n, [1., 0., 0.]); a /= np.linalg.norm(a); b_ = np.cross(n, a)
        return np.array([cx, cy, z])+r*(np.cos(ang)[:, None]*a+np.sin(ang)[:, None]*b_), np.array([cx, cy, z])

    def residuals(p):
        pts, ctr = circle(p)
        out = []
        for dist, K, R, t in views:
            eye = -R.T@t
            far = np.linalg.norm(pts-eye, axis=1) > np.linalg.norm(ctr-eye)       # far half of the rim
            x = (pts@R.T+t)@K.T
            u, v = x[:, 0]/x[:, 2], x[:, 1]/x[:, 2]
            ok = far & (u > 6) & (u < dist.shape[1]-6) & (v > 6) & (v < dist.shape[0]-6)
            r_ = np.full(len(ang), 50.)                                           # off-image samples: constant
            r_[ok] = map_coordinates(dist, [v[ok], u[ok]], order=1)
            out.append(np.where(far, r_, 0.))
        return np.concatenate(out)
    best = None
    for z in np.linspace(-.04, .02, 7):
        for r in (.25, .3, .35, .4, .45):
            for cx in np.linspace(-.2, .4, 7):
                for cy in np.linspace(0., .6, 7):
                    p = np.array([cx, cy, z, r]+([0., 0.] if tilt else []))
                    c_ = float(np.sum(np.minimum(residuals(p), 50.)**2))
                    if best is None or c_ < best[0]:
                        best = (c_, p)
    sol = least_squares(residuals, best[1], loss='soft_l1', f_scale=3.)
    res = residuals(sol.x)
    used = res[res > 0]
    pts, ctr = circle(sol.x)
    normal_tilt = 0.
    if tilt:
        from scipy.spatial.transform import Rotation
        n = Rotation.from_rotvec([sol.x[4], sol.x[5], 0.]).apply([0., 0., 1.]); normal_tilt = float(np.degrees(np.arccos(abs(n[2]))))
    out = dict(centre=ctr.tolist(), radius=float(sol.x[3]), plane_z_at_centre=float(sol.x[2]), tilt=tilt, normal_tilt_deg=normal_tilt,
               params=sol.x.tolist(), residual_px_median=float(np.median(used)), residual_px_p90=float(np.percentile(used, 90)))
    (dyn/'rest'/f'ground_rim{"_tilt" if tilt else ""}.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    tiles = []
    for c, (dist, K, R, t) in zip(CAMS, views):
        im = cv2.imread(str(dyn/'rest'/f'background_{c}.png'))
        x = (pts@R.T+t)@K.T
        for u, v in zip(x[:, 0]/x[:, 2], x[:, 1]/x[:, 2]):
            if 0 <= u < im.shape[1] and 0 <= v < im.shape[0]:
                cv2.circle(im, (int(u), int(v)), 3, (0, 0, 255), -1)
        tiles.append(cv2.resize(im, (640, 360)))
    cv2.imwrite(str(dyn/'rest'/f'rim{"_tilt" if tilt else ""}.png'), np.concatenate(tiles, 1))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('cmd', choices=['extract', 'rest', 'rim'])
    p.add_argument('--obj', default='burger')
    p.add_argument('--frames', type=int, nargs='+', default=[32480, 32520], help='C0733 video frame indices')
    p.add_argument('--background', type=int, default=32700, help='C0733 frame with an empty table')
    p.add_argument('--run', default=None)
    p.add_argument('--dem', default=None)
    p.add_argument('--registration', default='initial_state_upright_nospin.json')
    p.add_argument('--fit-scale', action='store_true', help='rest: also fit a scale sigma about the object centre')
    p.add_argument('--upright', action='store_true', help='rest: rotation about the up axis only')
    p.add_argument('--tag', default='')
    p.add_argument('--tilt', action='store_true', help='rim: also fit the plane normal')
    a = p.parse_args()
    if a.cmd == 'extract':
        extract(a.obj, a.frames, a.background)
    elif a.cmd == 'rest':
        rest_fit(a.obj, a.frames, a.run or f'output/springgaus_{a.obj}_spacing8k', a.dem or f'output/springgaus_dem/{a.obj}_spacing8k_mf0.1',
                 a.registration, fit_scale=a.fit_scale, upright=a.upright, tag=a.tag)
    elif a.cmd == 'rim':
        rim_fit(a.obj, tilt=a.tilt)
