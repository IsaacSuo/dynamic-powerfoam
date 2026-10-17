"""Static -> dynamic registration refined with colour (the burger is almost round, so silhouettes
leave its orientation ambiguous: the mask-only fit tilted it ~45 deg and showed the never-
photographed bottom to C0801).

The whole static foam moves rigidly in its own frame about the object centre c:
p' = c + Q (p - c) + d. Loss on frame 0 of all 3 views (as Spring-Gaus' registration): L1 of the
render composited over the captured image (so only object pixels count) + silhouette L2.
Global search over Q, scored without gradients: --starts random rotations (plus identity), and
(--upright) the "upright" family: in the static capture the object rests on the chessboard, so
the board normal (RANSAC plane of the COLMAP points outside the object box, oriented toward
the cameras) is mapped to the dynamic world's up (-gravity), times yaw about it every 10 deg,
also upside down. The best --refine are optimised with Adam (axis-angle increment, d, log scale
sigma about c). The winner updates the registration: R_new = R Q, s_new = s sigma, t_new = new
image of the COLMAP mean; written to register_calib/frame00_rgb.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation

from experiments.foam_deform_render import DeformedFoam
from experiments.foam_dynamic_render import load, render
from experiments.pacnerf_foam_geometry import foam_to_dynamic
from experiments.springgaus_dynamic import CAMS
from experiments.springgaus_train import foam_cameras, skew

OUT = Path('data/springgaus_powerfoam')


def board_up(obj, R0, meta):
    """Static capture: the object rests on the chessboard. Board normal (RANSAC plane of the COLMAP
    points outside the object box, oriented toward the cameras) and the dynamic world's up
    (-gravity), both in the foam frame (R0 = rotation of the foam -> dynamic map)."""
    import pycolmap
    recon = pycolmap.Reconstruction(str(OUT/obj/'sparse'/'0'))
    pts = np.array([q.xyz for q in recon.points3D.values()])
    pts = pts[np.abs(pts-.5).max(1) > .45]
    rng = np.random.default_rng(0); best_in = None
    for _ in range(2000):
        tri = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(tri[1]-tri[0], tri[2]-tri[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        inl = np.abs((pts-tri[0])@n) < .01
        if best_in is None or inl.sum() > best_in.sum():
            best_in = inl
    q = pts[best_in]; n = np.linalg.svd(q-q.mean(0))[2][-1]
    eyes = np.array([im.projection_center() for im in recon.images.values()])
    if ((eyes-q.mean(0))@n).mean() < 0:
        n = -n
    up_f = R0.T@(-np.array(meta['gravity'])/np.linalg.norm(meta['gravity']))
    print('board plane inliers %d of %d, normal %s, up in foam frame %s' % (best_in.sum(), len(pts), np.round(n, 3), np.round(up_f, 3)), flush=True)
    return n, up_f


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--obj', default='burger')
    p.add_argument('--run', default=None)
    p.add_argument('--registration', default='frame00.json')
    p.add_argument('--downsample', type=int, default=2)
    p.add_argument('--starts', type=int, default=200)
    p.add_argument('--refine', type=int, default=4)
    p.add_argument('--iterations', type=int, default=150)
    p.add_argument('--silhouette-weight', type=float, default=1.)
    p.add_argument('--upright', action='store_true')
    a = p.parse_args()
    run = a.run or f'output/springgaus_{a.obj}_v2'
    dyn = OUT/f'{a.obj}_dynamic'
    reg_path = dyn/'register_calib'/a.registration
    reg = json.loads(reg_path.read_text())
    meta = json.loads((dyn/'cameras.json').read_text())
    conv = json.loads((OUT/a.obj/'conversion.json').read_text())
    A, b, k = foam_to_dynamic(OUT/a.obj, reg_path)
    foam, _ = load(run)
    for name in ('radii', 'density', 'texel_sites', 'texel_sv_axis', 'texel_sv_rgb', 'texel_height', 'points', 'quaternions'):
        getattr(foam, name).requires_grad_(False)
    p0 = foam.points.detach()
    c = p0.mean(0)                                    # object-only foam: centre of the sites
    deformed = DeformedFoam(foam, np.zeros(len(p0), int), c[None].cpu().numpy(), np.zeros(3))
    cams = foam_cameras(meta, A, b, k, a.downsample)
    W, H = cams[0].width, cams[0].height
    imgs, masks = [], []
    for name in CAMS:
        imgs.append(torch.tensor(np.asarray(Image.open(dyn/'images'/name/'00.png').convert('RGB').resize((W, H), Image.LANCZOS),
                                            np.float32)/255, device='cuda'))
        masks.append(torch.tensor(np.asarray(Image.open(dyn/'masks'/name/'00.png').convert('L').resize((W, H), Image.BILINEAR),
                                             np.float32)/255, device='cuda'))

    def loss_at(Q, d, sigma=None):
        if sigma is None:
            deformed.apply(c[None]+d[None], body_R=Q[None], mode='rigid')
        else:
            F = (sigma*Q)[None]
            deformed.apply(c[None]+d[None], body_R=Q[None], body_F=F, mode='affine')
        total, parts = 0., []
        for cam, img, m in zip(cams, imgs, masks):
            rgb, alpha = render(foam, cam)
            alpha = alpha.reshape(H, W)
            comp = rgb.clamp(0, 1)+(1-alpha.clamp(0, 1))[..., None]*img
            l_rgb = (comp-img).abs().mean()*W*H/m.sum().clamp(min=1)       # per object pixel
            l_sil = ((alpha-m)**2).mean()*W*H/m.sum().clamp(min=1)
            parts.append((float(l_rgb), float(l_sil), float(((alpha > .5) & (m > .5)).sum()/((alpha > .5) | (m > .5)).sum())))
            total = total+l_rgb+a.silhouette_weight*l_sil
        return total/len(cams), parts

    rots = np.concatenate((np.eye(3)[None], Rotation.random(a.starts, random_state=0).as_matrix()))
    if a.upright:
        n, up_f = board_up(a.obj, A/k, meta)
        align = Rotation.align_vectors([up_f], [n])[0]
        flip = Rotation.align_vectors([-up_f], [n])[0]
        fam = [Rotation.from_rotvec(np.radians(y)*up_f)*base for base in (align, flip) for y in range(0, 360, 10)]
        rots = np.concatenate((rots, np.array([r.as_matrix() for r in fam])))
        n_upright = len(fam)
    zero = torch.zeros(3, device='cuda')
    scores = []
    with torch.no_grad():
        for i, Q in enumerate(rots):
            l, _ = loss_at(torch.tensor(Q, dtype=torch.float32, device='cuda'), zero)
            scores.append(float(l))
    order = np.argsort(scores)
    if a.upright:
        up_scores = scores[-n_upright:]
        print('best upright score %.4f (yaw %d, %s), best random %.4f' % (min(up_scores), 10*(int(np.argmin(up_scores)) % 36),
              'upside down' if np.argmin(up_scores) >= 36 else 'upright', min(scores[:-n_upright])), flush=True)
    print('identity score %.4f, best random %.4f (rank of identity %d)' % (scores[0], scores[order[0]], int(np.where(order == 0)[0][0])), flush=True)
    results = []
    for i in order[:a.refine]:
        Qs = torch.tensor(rots[i], dtype=torch.float32, device='cuda')
        w = torch.zeros(3, device='cuda', requires_grad=True)
        d = torch.zeros(3, device='cuda', requires_grad=True)
        ls = torch.zeros((), device='cuda', requires_grad=True)
        opt = torch.optim.Adam([dict(params=[w], lr=1e-2), dict(params=[d], lr=1e-3), dict(params=[ls], lr=2e-3)])
        for it in range(a.iterations):
            opt.zero_grad()
            l, _ = loss_at(torch.matrix_exp(skew(w))@Qs, d, torch.exp(ls))
            l.backward(); opt.step()
        with torch.no_grad():
            Q = torch.matrix_exp(skew(w))@Qs
            l, parts = loss_at(Q, d, torch.exp(ls))
        results.append(dict(start=int(i), start_score=scores[i], loss=float(l), parts=parts, Q=Q.cpu().numpy(), d=d.detach().cpu().numpy(),
                            sigma=float(torch.exp(ls)), upright_family=bool(a.upright and i >= len(rots)-n_upright)))
        print('start %d (sigma %.3f): score %.4f -> %.4f | per view (rgb L1, sil, IoU) %s' % (i, results[-1]['sigma'], scores[i], float(l), np.round(parts, 3).tolist()), flush=True)
    best = min(results, key=lambda r: r['loss'])
    # Update the registration: foam x -> c + Q (x - c) + d, then the old foam -> dynamic map.
    Q, d, cn, sigma = best['Q'].astype(float), best['d'].astype(float), c.cpu().numpy().astype(float), best['sigma']
    R_new = np.array(reg['R'])@Q        # COLMAP -> foam has no rotation
    mean_foam = conv['scale']*(np.array(reg['colmap_mean'])-np.array(conv['centre']))+.5
    t_new = A@(cn+sigma*Q@(mean_foam-cn)+d)+b
    out = dict(reg, R=R_new.tolist(), t=t_new.tolist(), s=reg['s']*sigma, refined_from=str(reg_path), method='rgb L1 + silhouette, frame 0, 3 views',
               start_rank_scores=[scores[i] for i in order[:a.refine]], refine=[{k_: v for k_, v in r.items() if k_ not in ('Q', 'd')} for r in results], upright=a.upright,
               iou=dict(zip(CAMS, [p_[2] for p_ in best['parts']])))
    (dyn/'register_calib'/'frame00_rgb.json').write_text(json.dumps(out, indent=1))
    angle = np.degrees(Rotation.from_matrix(Q).magnitude())
    print('best start %d (upright family %s, sigma %.3f), rotation change %.1f deg, shift %s (foam units), IoU %s' % (best['start'], best['upright_family'], sigma, angle, np.round(d, 4), out['iou']))


if __name__ == '__main__':
    main()
