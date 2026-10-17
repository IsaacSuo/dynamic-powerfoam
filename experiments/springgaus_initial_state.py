"""Joint initial-state fit on the pre-hit (free flight, rigid) frames of a Spring-Gaus sequence.

Why: the object falls at ~3 m/s, i.e. ~2.5 cm per 120 fps frame (a third of the burger). The
three videos are synced only to whole frames (sync.json) and the frames are motion blurred, so
no single pose fits frame 0 in all views (registration IoU stalls at ~0.85).

Unknowns: pose correction of the registration (rotation increment, shift d, scale sigma about the
foam centre c, as springgaus_register_rgb), initial velocity v0 and angular velocity w0, a
sub-frame time offset per camera (camera 0 fixed at 0) and the exposure time tau (blur: render at
t + {-1/2, 0, 1/2} tau and average). Motion: x(t) = com + exp(t [w0]) (x_reg - com) + v0 t + g t^2/2
in the dynamic world, mapped to the foam frame. Loss over frames 0..hit-1, all 3 views (state
estimation, like Spring-Gaus' registration; the held-out view is only used before the impact):
L1 of the render composited over the captured image (per object pixel) + silhouette L2.
Writes register_calib/initial_state.json: registration (same format as springgaus_dynamic
register, so pacnerf_foam_geometry can rebuild the DEM from it), v0, w0, offsets, tau.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from experiments.foam_deform_render import DeformedFoam
from experiments.foam_dynamic_render import load, render
from experiments.pacnerf_foam_geometry import foam_to_dynamic
from experiments.springgaus_dynamic import CAMS
from experiments.springgaus_train import foam_cameras, skew

OUT = Path('data/springgaus_powerfoam')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--obj', default='burger')
    p.add_argument('--run', default=None)
    p.add_argument('--registration', default='frame00_rgb.json')
    p.add_argument('--downsample', type=int, default=2)
    p.add_argument('--iterations', type=int, default=300)
    p.add_argument('--blur', action='store_true', help='fit an exposure time (3 sub-renders per frame)')
    p.add_argument('--silhouette-weight', type=float, default=1.)
    p.add_argument('--upright', action='store_true', help='start from the static board normal mapped to up, best yaw of 36')
    p.add_argument('--no-spin', action='store_true', help='fix w0 = 0')
    p.add_argument('--tag', default='')
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
    dev = 'cuda'
    p0 = foam.points.detach()
    c = p0.mean(0)
    deformed = DeformedFoam(foam, np.zeros(len(p0), int), c[None].cpu().numpy(), np.zeros(3))
    cams = foam_cameras(meta, A, b, k, a.downsample)
    W, H = cams[0].width, cams[0].height
    hit, fps = meta['hit_frame'], meta['fps']
    frames = list(range(hit))
    imgs, masks = {}, {}
    for f in frames:
        for ci, name in enumerate(CAMS):
            imgs[f, ci] = torch.tensor(np.asarray(Image.open(dyn/'images'/name/f'{f:02d}.png').convert('RGB').resize((W, H), Image.LANCZOS),
                                                  np.float32)/255, device=dev)
            masks[f, ci] = torch.tensor(np.asarray(Image.open(dyn/'masks'/name/f'{f:02d}.png').convert('L').resize((W, H), Image.BILINEAR),
                                                   np.float32)/255, device=dev)
    t32 = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32, device=dev)
    A_t, b_t, R0 = t32(A), t32(b), t32(A/k)
    Ainv = torch.linalg.inv(A_t)
    g = t32(meta['gravity'])
    w_rot = torch.zeros(3, device=dev, requires_grad=True)
    d = torch.zeros(3, device=dev, requires_grad=True)
    ls = torch.zeros((), device=dev, requires_grad=True)
    v0 = t32([0., 0., 3.]).requires_grad_(True)
    w0 = torch.zeros(3, device=dev, requires_grad=True)
    offs = torch.zeros(len(CAMS)-1, device=dev, requires_grad=True)          # in frames
    log_tau = t32(np.log(.3)).requires_grad_(a.blur)                          # exposure, in frames
    groups = [dict(params=[w_rot], lr=1e-2), dict(params=[d], lr=1e-3), dict(params=[ls], lr=2e-3),
              dict(params=[v0], lr=2e-2), dict(params=[offs], lr=2e-2)]
    if not a.no_spin:
        groups.append(dict(params=[w0], lr=.2))
    if a.blur:
        groups.append(dict(params=[log_tau], lr=2e-2))
    opt = torch.optim.Adam(groups)

    def pose(t):
        """Foam -> foam affine map (M, image of c) at time t (seconds)."""
        sigma = torch.exp(ls)
        Q = torch.matrix_exp(skew(w_rot))
        cr = A_t@(c+d)+b_t                                     # registered centre (world)
        com = cr                                               # rotate about the object centre
        Rt = torch.matrix_exp(t*skew(w0))
        x = com+Rt@(cr-com)+t*v0+.5*t*t*g
        M = sigma*(R0.T@Rt@R0@Q)
        return M, Ainv@(x-b_t)

    def frame_terms(f, ci):
        offset = 0. if ci == 0 else offs[ci-1]
        subs = (-.5, 0., .5) if a.blur else (0.,)
        rgb_acc, al_acc = 0., 0.
        for s in subs:
            t = (f+offset+s*torch.exp(log_tau))/fps
            M, xc = pose(t)
            deformed.apply(xc[None], body_F=M[None], mode='affine')
            rgb, alpha = render(foam, cams[ci])
            rgb_acc = rgb_acc+rgb.clamp(0, 1)/len(subs); al_acc = al_acc+alpha.reshape(H, W).clamp(0, 1)/len(subs)
        comp = rgb_acc+(1-al_acc)[..., None]*imgs[f, ci]
        npx = masks[f, ci].sum().clamp(min=1)
        return (comp-imgs[f, ci]).abs().mean()*W*H/npx, ((al_acc-masks[f, ci])**2).mean()*W*H/npx, al_acc

    def total(collect=False):
        loss, ious = 0., np.zeros((len(frames), len(CAMS)))
        for f in frames:
            for ci in range(len(CAMS)):
                l_rgb, l_sil, al = frame_terms(f, ci)
                loss = loss+l_rgb+a.silhouette_weight*l_sil
                if collect:
                    pm, qm = al > .5, masks[f, ci] > .5
                    ious[f, ci] = float((pm & qm).sum()/(pm | qm).sum())
        return loss/(len(frames)*len(CAMS)), ious

    if a.upright:
        from scipy.spatial.transform import Rotation
        from experiments.springgaus_register_rgb import board_up
        n, up_f = board_up(a.obj, A/k, meta)
        align = Rotation.align_vectors([up_f], [n])[0]
        scan = []
        with torch.no_grad():
            for yaw in range(0, 360, 10):
                w_rot.copy_(t32((Rotation.from_rotvec(np.radians(yaw)*up_f)*align).as_rotvec()))
                scan.append(float(total()[0]))
            yaw = 10*int(np.argmin(scan))
            w_rot.copy_(t32((Rotation.from_rotvec(np.radians(yaw)*up_f)*align).as_rotvec()))
        print('upright yaw scan: best %d deg, loss %.4f (worst %.4f)' % (yaw, min(scan), max(scan)), flush=True)
    with torch.no_grad():
        l0, iou0 = total(True)
    print('start loss %.4f, IoU per cam %s' % (float(l0), np.round(iou0.mean(0), 3)), flush=True)
    for it in range(a.iterations):
        opt.zero_grad()
        loss, _ = total()
        loss.backward(); opt.step()
        if it % 25 == 0 or it == a.iterations-1:
            print(it, 'loss %.4f sigma %.3f v0 %s w0 %s offsets %s tau %.2f' % (float(loss), float(torch.exp(ls)), np.round(v0.tolist(), 3),
                  np.round(w0.tolist(), 2), np.round(offs.tolist(), 3), float(torch.exp(log_tau))), flush=True)
    with torch.no_grad():
        l1, iou1 = total(True)
        Q = torch.matrix_exp(skew(w_rot)).cpu().numpy().astype(float)
        sigma = float(torch.exp(ls)); dn = d.cpu().numpy().astype(float); cn = c.cpu().numpy().astype(float)
    mean_foam = conv['scale']*(np.array(reg['colmap_mean'])-np.array(conv['centre']))+.5
    out = dict(reg, R=(np.array(reg['R'])@Q).tolist(), t=(A@(cn+sigma*Q@(mean_foam-cn)+dn)+b).tolist(), s=reg['s']*sigma,
               refined_from=str(reg_path), method='joint pre-hit fit (pose, scale, v0, w0, camera time offsets, exposure)',
               v0=v0.tolist(), w0=w0.tolist(), rotation_centre_world=(A@(cn+dn)+b).tolist(),
               time_offsets_frames=dict(zip(CAMS, [0.]+offs.tolist())), exposure_frames=float(torch.exp(log_tau)) if a.blur else 0.,
               loss_start=float(l0), loss_end=float(l1), iou_start=iou0.tolist(), iou_end=iou1.tolist(),
               iou_end_per_cam=dict(zip(CAMS, iou1.mean(0).tolist())), upright_start=a.upright, no_spin=a.no_spin)
    name = 'initial_state'+('_blur' if a.blur else '')+a.tag+'.json'
    (dyn/'register_calib'/name).write_text(json.dumps(out, indent=1))
    print('end loss %.4f, IoU per cam %s, per frame %s' % (float(l1), np.round(iou1.mean(0), 3), np.round(iou1.mean(1), 3)))


if __name__ == '__main__':
    main()
