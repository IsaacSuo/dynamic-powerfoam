"""Train the shared DEM parameters from video silhouettes only (no 3D supervision).

Per iteration: DEM rollout over the training frames; every kept Power Foam cell (material +
2-ring air shell) moves rigidly with its body; the foam is rendered (differentiable w.r.t. cell
centres) in the training views and the alpha map is compared with the captured object mask
(L2 per pixel). Gradients go alpha -> cell centres -> body positions/rotations -> DEM rollout
-> 7 shared log multipliers (pacnerf_dem_train). Cell orientations follow the bodies but are
detached (they only shade). Views: 6 for training, 5 held out (as in GIC); frames 1-10 train.

--render-mode blend uses the continuous deformation field; --rgb-weight > 0 adds PowerSim's
photometric term on the render composited over the view's object-free frame.

Evaluation: silhouette IoU (and PSNR) per split (train/held-out views x train/held-out frames) and the 3D
body-trajectory normalized RMSE against the GT particles (never used in training).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from experiments.foam_dynamic_render import air_shell, keep_cells, load, render
from experiments.pacnerf_dem_forward import build, cell_targets
from experiments.pacnerf_render_dem import body_rotation
from experiments.pacnerf_to_powerfoam import object_mask
from experiments.foam_deform_render import DeformedFoam
from powerfoam.metrics import ssim_eval
from experiments.foam_dem_warp import WarpFoam, rollout as warp_rollout


def set_points(foam, points, quats):
    """Differentiable cell centres: replace the Parameter by a graph tensor."""
    foam._parameters.pop('points', None)
    foam.points = points
    from experiments.foam_dynamic_render import quat_mul
    q = quat_mul(quats.detach(), foam.rest_quaternions/foam.rest_quaternions.norm(dim=-1, keepdim=True))
    foam._parameters.pop('quaternions', None)
    foam.quaternions = q
    with torch.no_grad():
        foam.aabb_tree.update(points.detach(), foam.get_radii())
        foam.adjacency, foam.adjacency_offsets = foam.aabb_tree.build_cech_complex()


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--run', default='output/pacnerf_elastic0_f0_v1')
    a.add_argument('--dem', type=Path, default=Path('output/pacnerf_dem/elastic_0_f0'))
    a.add_argument('--scene', default='elastic/0')
    a.add_argument('--frames', type=int, default=13)
    a.add_argument('--train-frames', type=int, default=10)
    a.add_argument('--train-views', type=int, nargs='+', default=[0, 2, 4, 6, 8, 10])
    a.add_argument('--iterations', type=int, default=40)
    a.add_argument('--lr', type=float, default=.05)
    a.add_argument('--out', type=Path, default=Path('output/pacnerf_dem/elastic_0_f0/silhouette_v1'))
    a.add_argument('--render-mode', default='rigid', choices=('rigid', 'blend'),
                   help="primitive update (foam_deform_render): 'blend' = continuous field, K=24, width 2")
    a.add_argument('--rgb-weight', type=float, default=0., help='add (1-l) L2 + l (1 - SSIM) of the composited image')
    a.add_argument('--ssim-lambda', type=float, default=.2)
    a.add_argument('--engine', default='torch', choices=('torch', 'warp'), help='DEM rollout: torch ops or fused Warp kernels')
    a.add_argument('--init-from', type=Path, help="report.json whose 'best' shared parameters start (and, per body, anchor) the fit")
    a.add_argument('--per-body', action='store_true', help='learn a per-body deviation (bodies x 7) on top of the start')
    a.add_argument('--smooth-weight', type=float, default=0., help='per body: weight of mean squared theta jump across bonds')
    a.add_argument('--init-noise', type=float, default=0., help='per body: start deviations ~ Uniform(-a, a) in log space (seed 0)')
    args = a.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    gt = json.loads(Path('data/pacnerf/gt/elastic.json').read_text())[args.scene.split('/')[1]]
    geometry = dict(np.load(args.dem/'geometry.npz')); cells = dict(np.load(args.dem/'cells.npz'))
    offset = torch.tensor(json.loads(Path('data/pacnerf_powerfoam/elastic_0_f0/conversion.json').read_text())['offset'],
                          dtype=torch.float32, device='cuda')
    foam, data = load(args.run)
    keep, owner = air_shell(foam, cells['source_index'], geometry['body_of_cell'], rings=2)
    keep_cells(foam, keep)
    for name in ('radii', 'density', 'texel_sites', 'texel_sv_axis', 'texel_sv_rgb', 'texel_height'):
        getattr(foam, name).requires_grad_(False)
    deformed = DeformedFoam(foam, owner, geometry['points'], offset)
    if args.render_mode == 'blend':
        deformed.prepare_blend(geometry['points'], K=24, width=2.)
    order = [int(Path(f['file_path']).stem.split('_')[1]) for f in
             json.loads(Path('data/pacnerf_powerfoam/elastic_0_f0/transforms_all.json').read_text())['frames']]
    src = Path('data/pacnerf/data')/args.scene/'data'
    masks, caps, bgs = {}, {}, {}
    for c, v in enumerate(order):
        bg = np.asarray(Image.open(src/f'r_{v}_-1.png').convert('RGB'), float)/255
        bgs[c] = torch.tensor(bg, dtype=torch.float32, device='cuda')
        for k in range(args.frames+1):
            img = np.asarray(Image.open(src/f'r_{v}_{k}.png').convert('RGB'), float)/255
            masks[k, c] = torch.tensor(object_mask(img, bg), dtype=torch.float32, device='cuda')
            caps[k, c] = torch.tensor(img, dtype=torch.float32, device='cuda')
    train_cams = [order.index(v) for v in args.train_views]
    test_cams = [c for c in range(len(order)) if c not in train_cams]
    targets, has, _ = cell_targets(cells, geometry['body_of_cell'], geometry['points'],
                                   Path('data/pacnerf/simulation_data')/args.scene, args.frames)
    model = build(geometry, gt['E'], gt['nu'], 'cuda', 'noslip', .25, .3, torch.float32)
    x0, q0, _, w0 = model.initial_state()
    v0 = torch.tensor(gt['v'], dtype=model.dtype, device='cuda').expand(model.n, 3).clone()
    base = torch.zeros(7, dtype=model.dtype, device='cuda')
    if args.init_from is not None:
        base = torch.tensor(json.loads(args.init_from.read_text())['best']['parameters'], dtype=model.dtype, device='cuda')
    if args.per_body:
        g0 = torch.Generator(device='cpu').manual_seed(0)
        start = (2*torch.rand(model.n, 7, generator=g0, dtype=torch.float64)-1)*args.init_noise
        shared = start.to(device='cuda', dtype=model.dtype).requires_grad_(True)
        theta_of = lambda p: base.expand(model.n, 7)+p
    else:
        shared = base.clone().requires_grad_(True)
        theta_of = lambda p: p.expand(model.n, 7)
    optimizer = torch.optim.Adam([shared], lr=args.lr)

    def summary(p):
        th = theta_of(p.detach())
        if not args.per_body:
            return th[0].tolist()
        return dict(mean=th.mean(0).tolist(), std=th.std(0).tolist(), min=th.min(0).values.tolist(), max=th.max(0).values.tolist())

    wf = WarpFoam(model) if args.engine == 'warp' else None

    def rollout(frames, substeps, grad=True):
        theta = theta_of(shared if grad else shared.detach())
        if wf is not None:
            return warp_rollout(wf, theta, (x0, q0, v0, w0), 1/24/substeps, substeps, frames)
        pos, quat, _ = model.rollout(theta, (x0, q0, v0, w0), 1/24/substeps, substeps, frames, compile=True, quaternions=True)
        return pos, quat

    def place(pos, quat, k):
        deformed.apply(pos[k], body_R=body_rotation(quat[k]), body_F=model.deformation_gradient(pos[k]),
                       mode=args.render_mode)

    def view_terms(cam, k, c):
        rgb, alpha = render(foam, cam)
        alpha = alpha.reshape(*alpha.shape[:2])
        comp = rgb.clamp(0, 1)+(1-alpha.clamp(0, 1))[..., None]*bgs[c]
        return alpha, comp

    def evaluate(tag, substeps):
        with torch.no_grad():
            pos, quat = rollout(args.frames, substeps, grad=False)
            iou = np.zeros((args.frames+1, len(order))); psnr = np.zeros_like(iou)
            for k in range(args.frames+1):
                place(pos, quat, k)
                for c, cam in enumerate(data.cameras):
                    alpha, comp = view_terms(cam, k, c)
                    m, g = alpha > .5, masks[k, c] > .5
                    iou[k, c] = float((m & g).sum()/(m | g).sum())
                    psnr[k, c] = float(-10*torch.log10(((comp-caps[k, c])**2).mean()))
        tr, te = slice(1, args.train_frames+1), slice(args.train_frames+1, args.frames+1)
        P = pos.cpu().numpy(); P = P-P[:1]+targets[:1]
        def nrmse(lo, hi):
            pp, tt = P[lo:hi+1, has], targets[lo:hi+1, has]
            return float(np.sqrt(np.mean((pp-tt)**2)/np.mean((tt-targets[:1, has])**2)))
        return dict(iou_trainviews_trainframes=float(iou[tr][:, train_cams].mean()),
                    iou_testviews_trainframes=float(iou[tr][:, test_cams].mean()),
                    iou_trainviews_testframes=float(iou[te][:, train_cams].mean()),
                    iou_testviews_testframes=float(iou[te][:, test_cams].mean()),
                    psnr_testviews_trainframes=float(psnr[tr][:, test_cams].mean()),
                    psnr_testviews_testframes=float(psnr[te][:, test_cams].mean()),
                    traj_nrmse_frames_1_10=nrmse(1, args.train_frames), traj_nrmse_frames_11_13=nrmse(args.train_frames+1, args.frames),
                    traj_nrmse_frames_1_13=nrmse(1, args.frames), parameters=summary(shared), substeps=substeps)

    substeps = int(np.ceil(1/24/model.stable_dt(theta_of(shared.detach()))))
    report = dict(config={k: str(v) for k, v in vars(args).items()}, initial=evaluate('initial', substeps))
    print('initial', json.dumps(report['initial']), flush=True)
    history, best = [], (np.inf, shared.detach().clone())
    start = time.perf_counter()
    for it in range(args.iterations):
        if it and it % 5 == 0:
            substeps = int(np.ceil(1/24/model.stable_dt(theta_of(shared.detach()))))
        optimizer.zero_grad()
        pos, quat = rollout(args.train_frames, substeps)
        loss = 0.
        for k in range(1, args.train_frames+1):
            place(pos, quat, k)
            for c in train_cams:
                alpha, comp = view_terms(data.cameras[c], k, c)
                loss = loss+((alpha-masks[k, c])**2).mean()
                if args.rgb_weight > 0:
                    l = args.ssim_lambda
                    loss = loss+args.rgb_weight*((1-l)*((comp-caps[k, c])**2).mean()+l*(1-ssim_eval(comp, caps[k, c])))
        loss = loss/(args.train_frames*len(train_cams))
        value = float(loss.detach())
        if args.per_body and args.smooth_weight > 0:
            th = theta_of(shared)
            loss = loss+args.smooth_weight*((th[model.i]-th[model.j])**2).sum(-1).mean()
        loss.backward()
        if not np.isfinite(value) or not torch.isfinite(shared.grad).all():
            print('non-finite at iteration', it, flush=True); break
        if value < best[0]:
            best = (value, shared.detach().clone())
        optimizer.step()
        with torch.no_grad():
            shared.clamp_(-3, 3)
        history.append(dict(iteration=it, silhouette_mse=value, grad_norm=float(shared.grad.norm()), parameters=summary(shared),
                            substeps=substeps, seconds=time.perf_counter()-start))
        print(json.dumps(history[-1]), flush=True)
        (args.out/'history.json').write_text(json.dumps(history, indent=1))
    with torch.no_grad():
        shared.copy_(best[1])
    substeps = int(np.ceil(1/24/model.stable_dt(theta_of(shared.detach()))))
    report['best'] = evaluate('best', substeps)
    report['seconds'] = time.perf_counter()-start
    names = ('kn', 'ks', 'kb', 'kt', 'lam', 'mu', 'damping')
    best_theta = theta_of(best[1])
    report['multipliers'] = dict(zip(names, np.exp(best_theta.mean(0).cpu().numpy()).tolist()))
    np.savez_compressed(args.out/'theta_best.npz', theta=best_theta.cpu().numpy())
    (args.out/'report.json').write_text(json.dumps(report, indent=1))
    print('best', json.dumps(report['best']), json.dumps(report['multipliers']), flush=True)


if __name__ == '__main__':
    main()
