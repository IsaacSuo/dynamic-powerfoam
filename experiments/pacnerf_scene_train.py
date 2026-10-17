"""Alternating physics / appearance training on a whole-scene Power Foam with gradient routing.

Scene: whole-scene foam split by foam_scene_split into an object foam (object cells + 2-ring
shell, driven by the DEM bodies through the continuous blend field) and a static background foam.
Image = object rgb + (1 - object alpha) * background rgb (object in front of the background).

Per pixel, with R = captured object mask and P = rendered object alpha > 0.5:
  physics  : silhouette (alpha - R)^2 over the image + RGB L2 over U = dilate(R or P)
             (object colours and background frozen);
  object colours (shared over frames): RGB L2 over the intersection erode(R and P), geometry frozen,
             frame 0 included (anchors colours where the geometry is exact) + L2 to the start;
  background (geometry fixed, per-frame darkening d = exp(-softplus(z)) in (0, 1] per cell):
             RGB L2 over B = outside dilate(R or P) by a wider margin, object frozen; smoothness of
             z across neighbouring background cells and adjacent frames.
Disagreeing pixels (only one mask says object) therefore drive the physics only; colour
parameters only learn where both masks agree. Appearance steps reuse the last trajectory.
Evaluation: held-out views / frames PSNR and IoU, and the 3D trajectory error (GT particles).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn
from PIL import Image

from experiments.foam_deform_render import DeformedFoam
from experiments.foam_dem_warp import WarpFoam, rollout as warp_rollout
from experiments.foam_dynamic_render import keep_cells, load
from experiments.foam_scene_split import frame_names, render_hooked, transparent_shell
from experiments.pacnerf_dem_forward import build, cell_targets
from experiments.pacnerf_to_powerfoam import object_mask


def dilate(m, r):
    return Fn.max_pool2d(m[None, None].float(), 2*r+1, 1, r)[0, 0] > .5 if r > 0 else m


def erode(m, r):
    return ~dilate(~m, r)


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--run', default='output/pacnerf_elastic0_f0_full_v1')
    a.add_argument('--scene-dir', default='data/pacnerf_powerfoam/elastic_0_f0_full')
    a.add_argument('--split', type=Path, default=Path('output/pacnerf_dem/elastic_0_f0/scene_split/split.npz'))
    a.add_argument('--dem', type=Path, default=Path('output/pacnerf_dem/elastic_0_f0'))
    a.add_argument('--theta', type=Path, default=Path('output/pacnerf_dem/elastic_0_f0/perbody_smooth1e-2/theta_best.npz'))
    a.add_argument('--scene', default='elastic/0')
    a.add_argument('--frames', type=int, default=13)
    a.add_argument('--train-frames', type=int, default=10)
    a.add_argument('--train-views', type=int, nargs='+', default=[0, 2, 4, 6, 8, 10])
    a.add_argument('--cycles', type=int, default=30)
    a.add_argument('--appearance-steps', type=int, default=3)
    a.add_argument('--lr-physics', type=float, default=.02)
    a.add_argument('--lr-color', type=float, default=2e-3)
    a.add_argument('--lr-shadow', type=float, default=.05)
    a.add_argument('--smooth-weight', type=float, default=1e-2)
    a.add_argument('--color-anchor', type=float, default=1.)
    a.add_argument('--shadow-smooth', type=float, default=1e-3)
    a.add_argument('--margin-union', type=int, default=3)
    a.add_argument('--margin-background', type=int, default=8)
    a.add_argument('--margin-intersection', type=int, default=2)
    a.add_argument('--out', type=Path, default=Path('output/pacnerf_dem/elastic_0_f0/scene_train_v1'))
    args = a.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    gt = json.loads(Path('data/pacnerf/gt/elastic.json').read_text())[args.scene.split('/')[1]]
    geometry = dict(np.load(args.dem/'geometry.npz')); cells = dict(np.load(args.dem/'cells.npz'))
    offset = json.loads((Path(args.scene_dir)/'conversion.json').read_text())['offset']
    split = np.load(args.split)
    fg, data = load(args.run)
    keep_cells(fg, split['object_keep'])
    transparent_shell(fg, len(split['object_cells']))
    bg, _ = load(args.run)
    keep_cells(bg, split['background_cells'])
    for foam in (fg, bg):
        for p_ in ('points', 'radii', 'density', 'quaternions', 'texel_sites', 'texel_sv_axis', 'texel_sv_rgb', 'texel_height'):
            getattr(foam, p_).requires_grad_(False)
    deformed = DeformedFoam(fg, np.zeros(len(split['object_keep']), int), geometry['points'], offset)
    deformed.prepare_blend(geometry['points'], K=24, width=2.)
    names = frame_names(args.scene_dir)
    order = [int(n.split('_')[1]) for n in names]
    train_cams = [order.index(v) for v in args.train_views]
    test_cams = [c for c in range(len(order)) if c not in train_cams]
    src = Path('data/pacnerf/data')/args.scene/'data'
    caps, masks = {}, {}
    for c, v in enumerate(order):
        bgimg = np.asarray(Image.open(src/f'r_{v}_-1.png').convert('RGB'), float)/255   # only to make the stand-in masks
        for k in range(args.frames+1):
            img = np.asarray(Image.open(src/f'r_{v}_{k}.png').convert('RGB'), float)/255
            caps[k, c] = torch.tensor(img, dtype=torch.float32, device='cuda')
            masks[k, c] = torch.tensor(object_mask(img, bgimg), device='cuda')
    targets, has, _ = cell_targets(cells, geometry['body_of_cell'], geometry['points'],
                                   Path('data/pacnerf/simulation_data')/args.scene, args.frames)
    model = build(geometry, gt['E'], gt['nu'], 'cuda', 'noslip', .25, .3, torch.float32)
    wf = WarpFoam(model)
    x0, q0, _, w0 = model.initial_state()
    v0 = torch.tensor(gt['v'], dtype=model.dtype, device='cuda').expand(model.n, 3).clone()
    theta0 = torch.tensor(np.load(args.theta)['theta'], dtype=model.dtype, device='cuda')
    delta = torch.zeros_like(theta0, requires_grad=True)
    opt_phys = torch.optim.Adam([delta], lr=args.lr_physics)
    color0 = fg.texel_sv_rgb.data.clone()
    color = torch.nn.Parameter(color0.clone())
    fg.texel_sv_rgb = color
    opt_color = torch.optim.Adam([color], lr=args.lr_color)
    nb = len(split['background_cells'])
    z = torch.full((args.frames+1, nb), -6., device='cuda', requires_grad=True)   # darkening ~ 1 - 2.5e-3
    opt_shadow = torch.optim.Adam([z], lr=args.lr_shadow)
    from experiments.powersim_bread_demo import regular_candidates
    bp, br = bg.points.detach().double().cpu().numpy(), bg.get_radii().detach().double().cpu().numpy()
    pairs, possible, _ = regular_candidates(bp, br)
    bpairs = torch.tensor(pairs[possible], device='cuda')

    def theta():
        return theta0+delta

    def background(k, c, grad=False):
        d = torch.exp(-Fn.softplus(z[k] if grad else z[k].detach()))
        return render_hooked(bg, data.cameras[c], lambda t: t*d[:, None, None])[0].clamp(0, 1)

    def place(pos, k):
        deformed.apply(pos[k], body_F=model.deformation_gradient(pos[k]), mode='blend')

    def regions(alpha, k, c):
        R, P = masks[k, c], alpha.detach() > .5
        U = dilate(R | P, args.margin_union)
        B = ~dilate(R | P, args.margin_background)
        I = erode(R & P, args.margin_intersection)
        return R, U, B, I

    def substeps_now():
        return int(np.ceil(1/24/model.stable_dt(theta().detach())))

    def evaluate(tag):
        sub = substeps_now()
        with torch.no_grad():
            pos, _ = warp_rollout(wf, theta().detach(), (x0, q0, v0, w0), 1/24/sub, sub, args.frames)
            iou = np.zeros((args.frames+1, len(order))); psnr = np.zeros_like(iou); psnr_obj = np.zeros_like(iou)
            for k in range(args.frames+1):
                place(pos, k)
                for c, cam in enumerate(data.cameras):
                    rgb, alpha = render_hooked(fg, cam)
                    comp = rgb.clamp(0, 1)+(1-alpha.clamp(0, 1))[..., None]*background(k, c)
                    R, U, B, I = regions(alpha, k, c)
                    P = alpha > .5
                    iou[k, c] = float((P & R).sum()/(P | R).sum())
                    err = (comp-caps[k, c])**2
                    psnr[k, c] = float(-10*torch.log10(err.mean()))
                    psnr_obj[k, c] = float(-10*torch.log10(err[U].mean()))
                    if c == order.index(3) and tag in ('initial', 'best'):
                        Image.fromarray((comp.cpu().numpy()*255).astype(np.uint8)).save(args.out/f'{tag}_view3_frame{k:02d}.png')
        P_ = pos.cpu().numpy(); P_ = P_-P_[:1]+targets[:1]
        def nrmse(lo, hi):
            pp, tt = P_[lo:hi+1, has], targets[lo:hi+1, has]
            return float(np.sqrt(np.mean((pp-tt)**2)/np.mean((tt-targets[:1, has])**2)))
        tr, te = slice(1, args.train_frames+1), slice(args.train_frames+1, args.frames+1)
        return dict(iou_testviews_trainframes=float(iou[tr][:, test_cams].mean()), iou_testviews_testframes=float(iou[te][:, test_cams].mean()),
                    psnr_testviews_trainframes=float(psnr[tr][:, test_cams].mean()), psnr_testviews_testframes=float(psnr[te][:, test_cams].mean()),
                    psnr_object_region_testviews_testframes=float(psnr_obj[te][:, test_cams].mean()),
                    traj_nrmse_frames_1_10=nrmse(1, args.train_frames), traj_nrmse_frames_11_13=nrmse(args.train_frames+1, args.frames),
                    traj_nrmse_frames_1_13=nrmse(1, args.frames), substeps=sub)

    report = dict(config={k: str(v) for k, v in vars(args).items()}, initial=evaluate('initial'))
    print('initial', json.dumps(report['initial']), flush=True)
    history, start = [], time.perf_counter()
    for cycle in range(args.cycles):
        # ---- physics step: object colours and background frozen --------------------------------
        sub = substeps_now()
        with torch.no_grad():
            bgs = {(k, c): background(k, c) for k in range(1, args.train_frames+1) for c in train_cams}
        opt_phys.zero_grad()
        pos, _ = warp_rollout(wf, theta(), (x0, q0, v0, w0), 1/24/sub, sub, args.train_frames)
        loss_phys, n_terms = 0., 0
        for k in range(1, args.train_frames+1):
            place(pos, k)
            for c in train_cams:
                with torch.no_grad():
                    color.requires_grad_(False)
                rgb, alpha = render_hooked(fg, data.cameras[c])
                R, U, B, I = regions(alpha, k, c)
                comp = rgb.clamp(0, 1)+(1-alpha.clamp(0, 1))[..., None]*bgs[k, c]
                loss_phys = loss_phys+((alpha-R.float())**2).mean()+((comp-caps[k, c])**2)[U].sum()/U.numel()
                n_terms += 1
        loss_phys = loss_phys/n_terms
        th = theta()
        total = loss_phys+args.smooth_weight*((th[model.i]-th[model.j])**2).sum(-1).mean()
        total.backward()
        opt_phys.step()
        color.requires_grad_(True)
        trajectory = pos.detach()
        # ---- appearance steps: trajectory frozen ------------------------------------------------
        for _ in range(args.appearance_steps):
            opt_color.zero_grad(); opt_shadow.zero_grad()
            loss_color, loss_bg = 0., 0.
            for k in range(0, args.train_frames+1):
                with torch.no_grad():
                    place(trajectory, k)
                for c in train_cams:
                    rgb, alpha = render_hooked(fg, data.cameras[c])          # colour grads only (geometry fixed)
                    R, U, B, I = regions(alpha, k, c)
                    bg_now = background(k, c, grad=True)
                    a_ = alpha.detach().clamp(0, 1)[..., None]
                    comp_col = rgb.clamp(0, 1)+(1-a_)*bg_now.detach()
                    comp_bg = rgb.detach().clamp(0, 1)+(1-a_)*bg_now
                    loss_color = loss_color+((comp_col-caps[k, c])**2)[I].sum()/I.numel()
                    loss_bg = loss_bg+((comp_bg-caps[k, c])**2)[B].sum()/B.numel()
            m = (args.train_frames+1)*len(train_cams)
            reg_color = args.color_anchor*((color-color0)**2).mean()
            reg_shadow = args.shadow_smooth*(((z[:, bpairs[:, 0]]-z[:, bpairs[:, 1]])**2).mean()+((z[1:]-z[:-1])**2).mean())
            (loss_color/m+reg_color+loss_bg/m+reg_shadow).backward()
            opt_color.step(); opt_shadow.step()
        history.append(dict(cycle=cycle, physics=float(loss_phys), color=float(loss_color/m), background=float(loss_bg/m),
                            darkening_min=float(torch.exp(-Fn.softplus(z.detach())).min()), substeps=sub,
                            seconds=time.perf_counter()-start))
        print(json.dumps(history[-1]), flush=True)
        (args.out/'history.json').write_text(json.dumps(history, indent=1))
    report['final'] = evaluate('best')
    report['seconds'] = time.perf_counter()-start
    np.savez_compressed(args.out/'state.npz', theta=theta().detach().cpu().numpy(), color=color.detach().cpu().numpy(),
                        z=z.detach().cpu().numpy())
    (args.out/'report.json').write_text(json.dumps(report, indent=1))
    print('final', json.dumps(report['final']), flush=True)


if __name__ == '__main__':
    main()
