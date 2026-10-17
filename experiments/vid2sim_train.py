"""Vid2Sim GSO object: learn (E, nu) of the DEM from multi-view silhouettes, compare with the GT.

Data (Vid2Sim test set): 12 views, 24 frames, one frame per simulation step of 0.05 s (their
simulator: reduced Simplicits LBS, backward Euler, floor penalty), zero initial velocity (matched
exactly in free fall by our v0 = g dt / 2, see below), gravity
-z, floor z = -0.7, rho = 1000. Frames 1..--train-frames train, the rest are future; --eval-view held out.
Scene = world + offset (vid2sim_to_powerfoam); DEM geometry from pacnerf_foam_geometry.

Parameters (shared): log E, nu (sigmoid to (0, 0.49)), damping, friction, restitution logits. All bond
stiffnesses and the bodies' mu scale with mu(E, nu), the bodies' lam with lam(E, nu); so the learned
E, nu are the material directly. Model options: --residual (non-affine residual energy),
--kinematics small|exact. Loss: silhouette L2 (rendered alpha vs GT alpha) of the training views.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from experiments.foam_deform_render import DeformedFoam
from experiments.foam_dem_warp import WarpFoam, rollout as warp_rollout
from experiments.foam_dynamic_render import air_shell, keep_cells, load, render
from experiments.pacnerf_dem_forward import build
from experiments.pacnerf_render_dem import body_rotation

SRC = Path('data/vid2sim/ext')


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--obj', default='backpack')
    a.add_argument('--E', type=float, default=2e4, help='initial E (deliberately off the GT)')
    a.add_argument('--nu', type=float, default=.3)
    a.add_argument('--train-frames', type=int, default=15)
    a.add_argument('--frames', type=int, default=23)
    a.add_argument('--eval-view', type=int, default=5)
    a.add_argument('--iterations', type=int, default=40)
    a.add_argument('--lr', type=float, default=.05)
    a.add_argument('--residual', type=int, default=1)
    a.add_argument('--kinematics', default='exact', choices=('small', 'exact'))
    a.add_argument('--contact', default='smooth', choices=('hard', 'smooth'))
    a.add_argument('--layer-dx', type=float, default=.01)
    a.add_argument('--segment', type=int, default=200)
    a.add_argument('--implicit-damping', type=int, default=1, help='pairwise implicit dashpots (time step from elasticity only)')
    a.add_argument('--graphs', type=int, default=1, help='CUDA-graph replay of the rollout (forward and backward)')
    a.add_argument('--time-only', type=int, default=0)
    a.add_argument('--tag', default='spacing8k', help='static run output/vid2sim_<obj>_<tag>; DEM output/vid2sim_dem/<obj>[_<tag> unless spacing8k]')
    a.add_argument('--params-from', type=Path, default=None, help='render-check 6: report.json whose best_params to render')
    a.add_argument('--shell', type=int, default=0, help='1: BOND_BENDING.md model')
    a.add_argument('--cosserat', type=int, default=0, help='1: COSSERAT_FOAM.md body law (implies --shell)')
    a.add_argument('--render-check', type=int, default=0, help='render the best run in --out with F / rotation / F-except-ill-conditioned (no training)')
    a.add_argument('--out', type=Path, required=True)
    args = a.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dev, f32 = 'cuda', torch.float32
    import warp as wp
    wp.init()
    wp.set_mempool_release_threshold('cuda', 0)        # return freed blocks to the driver (12 GB card)
    run = f'output/vid2sim_{args.obj}_{args.tag}'
    dem = Path(f'output/vid2sim_dem/{args.obj}'+('' if args.tag == 'spacing8k' else f'_{args.tag}'))
    scene = Path(f'data/vid2sim_powerfoam/{args.obj}_f0')
    offset = np.array(json.loads((scene/'conversion.json').read_text())['offset'])
    gt = yaml.safe_load(open(SRC/args.obj/'gt_phys_params.yaml'))
    dt_frame, gravity, ground_p, ground_n = .05, [0., 0., -9.8], [0., 0., -.7], [0., 0., 1.]
    geometry = dict(np.load(dem/'geometry.npz')); cells = dict(np.load(dem/'cells.npz'))
    foam, data = load(run)
    views = [int(Path(f['file_path']).name.split('_')[1]) for f in json.loads((scene/'transforms_all.json').read_text())['frames']]
    keep, owner = air_shell(foam, cells['source_index'], geometry['body_of_cell'], rings=2)
    keep_cells(foam, keep)
    for n_ in ('radii', 'density', 'texel_sites', 'texel_sv_axis', 'texel_sv_rgb', 'texel_height'):
        getattr(foam, n_).requires_grad_(False)
    deformed = DeformedFoam(foam, owner, geometry['points'], offset)
    deformed.prepare_blend(geometry['points'], K=24, width=2.)
    cams = data.cameras
    W, H = cams[0].width, cams[0].height
    masks = {}
    for f in range(args.frames+1):
        for c, v in enumerate(views):
            al = np.asarray(Image.open(SRC/args.obj/'data'/f'a_{v}_{f}.png').convert('RGBA'), np.float32)[..., 3]/255
            masks[f, c] = torch.tensor(al, device=dev)
    eval_c = views.index(args.eval_view)
    train_cams = [c for c in range(len(views)) if c != eval_c]

    if args.cosserat:
        args.shell = 1
    model = build(geometry, args.E, args.nu, dev, 'noslip', .25, .3, f32, kinematics=args.kinematics, residual=bool(args.residual), shell=bool(args.shell), cosserat=bool(args.cosserat))
    model.set_environment(gravity=gravity, ground_point=ground_p, ground_normal=ground_n,
                          contact_radius=model.contact_radius.cpu().numpy(), ground_mode='noslip')
    x0, q0, v0, w0 = model.initial_state()
    # Their backward Euler with one step per frame falls g dt^2 n(n+1)/2 by frame n; an initial
    # velocity g dt/2 makes our (accurate) free fall hit exactly those positions at every frame.
    v0 = (v0+torch.tensor(gravity, dtype=f32, device=dev)*dt_frame/2).contiguous()
    lam0, mu0 = args.E*args.nu/((1+args.nu)*(1-2*args.nu)), args.E/(2*(1+args.nu))
    logit = lambda p: float(np.log(p/(1-p)))
    params = torch.tensor([np.log(args.E), logit(args.nu/.49), 0., 0., 0.], dtype=f32, device=dev, requires_grad=True)

    def material(p):
        E = torch.exp(p[0]); nu = .49*torch.sigmoid(p[1])
        return E, nu, E*nu/((1+nu)*(1-2*nu)), E/(2*(1+nu))

    def theta_of(p):
        E, nu, lam, mu = material(p)
        lm, ll = torch.log(mu/mu0), torch.log(lam/lam0)
        row = torch.stack((lm, lm, lm, lm, ll, lm, p[2], p[3], p[4]))
        return row.expand(model.n, 9)
    optimizer = torch.optim.Adam([params], lr=args.lr)
    wf = WarpFoam(model, contact=args.contact, layer_dx=args.layer_dx, implicit_damping=bool(args.implicit_damping),
                  graphs=bool(args.graphs))

    def rollout(n, substeps, grad=True):
        th = theta_of(params if grad else params.detach())
        return warp_rollout(wf, th, (x0, q0, v0, w0), dt_frame/substeps, substeps, n, args.segment if grad else None)

    def place(pos, quat, f):
        deformed.apply(pos[f], body_R=body_rotation(quat[f]), body_F=model.deformation_gradient(pos[f]), mode='blend')

    def silhouette(c):
        _, alpha = render(foam, cams[c])
        return alpha.reshape(H, W)

    model_dt, dt_safety = model, .5
    if args.shell:   # time step from the enhancement-off (stiffest) configuration, float64, safety .4 (see vid2sim_traj_train)
        model_dt, dt_safety = build(geometry, args.E, args.nu, dev, 'noslip', .25, .3, torch.float64, kinematics=args.kinematics,
                                    residual=bool(args.residual), shell=True, shell_eas=False, cosserat=bool(args.cosserat)), .4

    def substeps_now():
        th7 = theta_of(params.detach())[:, :7]
        return int(np.ceil(dt_frame/model_dt.stable_dt(th7.to(model_dt.dtype), safety=dt_safety, include_damping=not wf.implicit_damping)))

    def summary():
        E, nu, _, _ = material(params.detach())
        return dict(E=float(E), nu=float(nu), damping=float(torch.exp(params[2])), E_over_gt=float(E)/gt['yms'])

    def evaluate(substeps):
        with torch.no_grad():
            pos, quat = rollout(args.frames, substeps, grad=False)
            ious = np.zeros((args.frames+1, len(views)))
            for f in range(args.frames+1):
                place(pos, quat, f)
                for c in range(len(views)):
                    al = silhouette(c) > .5; m = masks[f, c] > .5
                    ious[f, c] = float((al & m).sum()/(al | m).sum())
        tr, te = slice(1, args.train_frames+1), slice(args.train_frames+1, args.frames+1)
        return dict(iou_trainviews_trainframes=float(ious[tr][:, train_cams].mean()), iou_evalview_trainframes=float(ious[tr, eval_c].mean()),
                    iou_trainviews_futureframes=float(ious[te][:, train_cams].mean()), iou_evalview_futureframes=float(ious[te, eval_c].mean()),
                    iou_per_frame=ious.mean(1).tolist(), substeps=substeps, **summary())

    def train_step(substeps):
        optimizer.zero_grad()
        import os
        dbg = os.environ.get('MEMDEBUG')
        def mem(tag):
            if dbg:
                print('MEM %-22s torch alloc %.2f GB (max %.2f) reserved %.2f | warp pool %.2f GB (high %.2f)' % (
                    tag, torch.cuda.memory_allocated()/1e9, torch.cuda.max_memory_allocated()/1e9, torch.cuda.memory_reserved()/1e9,
                    wp.get_mempool_used_mem_current('cuda:0')/1e9, wp.get_mempool_used_mem_high('cuda:0')/1e9), flush=True)
        mem('before rollout')
        pos, quat = rollout(args.train_frames, substeps)
        mem('after rollout fwd')
        # frame-by-frame backward through the renderer (only one frame's render graph alive at a time),
        # then one backward of the accumulated dL/dpos, dL/dquat through the rollout: same gradient.
        pos_d = pos.detach().requires_grad_(True); quat_d = quat.detach().requires_grad_(True)
        norm = args.train_frames*len(train_cams)
        total = 0.
        for f in range(1, args.train_frames+1):
            place(pos_d, quat_d, f)
            lf = 0.
            for c in train_cams:
                lf = lf+((silhouette(c)-masks[f, c])**2).mean()
            (lf/norm).backward()
            total += float(lf.detach())/norm
            if f in (1, args.train_frames):
                mem(f'after frame {f} render bwd')
        outs = [(t, g) for t, g in ((pos, pos_d.grad), (quat, quat_d.grad)) if g is not None]
        torch.autograd.backward([t for t, _ in outs], [g for _, g in outs])
        mem('after rollout bwd')
        return total

    if args.render_check == 6:
        return params_render(args, params, rollout, substeps_now, place, foam, cams, views, train_cams, H, W)
    if args.render_check == 5:
        return gt_material_check(args, params, gt, logit, evaluate, substeps_now, rollout, place, foam, cams, views, train_cams, H, W)
    if args.render_check == 4:
        return tangle_check(args, params, model, rollout, substeps_now, theta_of)
    if args.render_check == 3:
        return jump_check(args, params, model, rollout, substeps_now)
    if args.render_check == 2:
        return affine_check(args, model, deformed, foam, cams, train_cams, H, W)
    if args.render_check:
        return render_check(args, params, model, rollout, substeps_now, deformed, foam, cams, views, masks, train_cams, eval_c, H, W)
    substeps = substeps_now()
    if args.time_only:
        t0 = time.perf_counter()
        for _ in range(args.time_only):
            train_step(substeps)
        torch.cuda.synchronize()
        print('bodies %d, substeps/frame %d, %.1f s per iteration' % (model.n, substeps, (time.perf_counter()-t0)/args.time_only))
        return
    show = lambda r: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != 'iou_per_frame'}
    report = dict(config={k: str(v) for k, v in vars(args).items()}, gt=gt, initial=evaluate(substeps))
    print('initial', json.dumps(show(report['initial'])), flush=True)
    history, best = [], (np.inf, params.detach().clone())
    start = time.perf_counter()
    for it in range(args.iterations):
        if it and it % 5 == 0:
            substeps = substeps_now()
        value = train_step(substeps)
        torch.cuda.empty_cache()
        if not np.isfinite(value) or not torch.isfinite(params.grad).all():
            print('non-finite at iteration', it, flush=True); break
        if value < best[0]:
            best = (value, params.detach().clone())
        optimizer.step()
        history.append(dict(iteration=it, silhouette_mse=value, substeps=substeps, seconds=time.perf_counter()-start, **summary()))
        print(json.dumps(history[-1]), flush=True)
        (args.out/'history.json').write_text(json.dumps(history, indent=1))
    with torch.no_grad():
        params.copy_(best[1])
    report['best'] = evaluate(substeps_now())
    report['best_params'] = params.detach().cpu().tolist()
    # video: GT (object on white) | simulation over white | silhouettes (red GT, green sim), 3 views
    from PIL import ImageDraw
    with torch.no_grad():
        pos, quat = rollout(args.frames, substeps_now(), grad=False)
        show_c = [train_cams[0], train_cams[len(train_cams)//2], eval_c]
        frames_img = []
        for f in range(args.frames+1):
            place(pos, quat, f)
            tiles = []
            for c in show_c:
                gt_rgb = np.asarray(Image.open(SRC/args.obj/'data'/f'm_{views[c]}_{f}.png').convert('RGB'), np.float32)/255
                rgb, al = render(foam, cams[c]); al = al.reshape(H, W).clamp(0, 1)
                sim = (rgb.clamp(0, 1)+(1-al)[..., None]).cpu().numpy()
                ov = np.zeros_like(sim); ov[..., 0] = masks[f, c].cpu().numpy() > .5; ov[..., 1] = al.cpu().numpy() > .5
                tiles.append(np.concatenate((gt_rgb, np.clip(sim, 0, 1), ov), 1))
            img = Image.fromarray((np.concatenate(tiles, 0)*255).astype(np.uint8)).resize((3*W//2*1, len(show_c)*H//2))
            d = ImageDraw.Draw(img)
            d.text((6, 4), 'frame %d (%s)' % (f, 'initial' if f == 0 else ('train' if f <= args.train_frames else 'FUTURE')), fill=(255, 0, 0))
            for k_, c in enumerate(show_c):
                d.text((6, k_*H//2+H//2-14), 'view %d%s' % (views[c], ' (held out)' if c == eval_c else ''), fill=(255, 0, 0))
            frames_img.append(img)
        frames_img[0].save(args.out/'compare.gif', save_all=True, append_images=frames_img[1:], duration=250, loop=0)
    report['seconds'] = time.perf_counter()-start
    (args.out/'report.json').write_text(json.dumps(report, indent=1))
    print('best', json.dumps(show(report['best'])), flush=True)


def render_check(args, params, model, rollout, substeps_now, deformed, foam, cams, views, masks, train_cams, eval_c, H, W):
    """Is the debris after impact caused by the least-squares F of ill-conditioned bodies? Same rollout (best
    parameters of --out), three ways to move the foam sites in blend mode: the body F (as trained), the body
    rotation only, and F except on ill-conditioned bodies (neighbour moment matrix min/max eigenvalue below
    the 2nd percentile), which use their rotation. Per-frame IoU over all views, a GIF, F statistics."""
    from PIL import ImageDraw
    rep = json.loads((args.out/'report.json').read_text())
    with torch.no_grad():
        params.copy_(torch.tensor(rep['best_params'], dtype=params.dtype, device=params.device))
        pos, quat = rollout(args.frames, substeps_now(), grad=False)
        G = torch.linalg.inv(model.moment_inv.double())
        ev = torch.linalg.eigvalsh(G)
        cond = (ev[:, 0]/ev[:, -1]).float()
        bad = cond < torch.quantile(cond, .02)
        print('ill-conditioned bodies: %d (min/max eigenvalue < %.3f)' % (int(bad.sum()), float(cond[bad].max())), flush=True)
        variants = ('F', 'R', 'rigid_owner')
        ious = {v: np.zeros((args.frames+1, len(views))) for v in variants}
        stats, frames_img = [], []
        show_c = [train_cams[0], train_cams[len(train_cams)//2], eval_c]
        for f in range(args.frames+1):
            R = body_rotation(quat[f]).to(pos.dtype)
            F = model.deformation_gradient(pos[f]).to(pos.dtype)
            sv = torch.linalg.svdvals(F.double())
            dev_ = (sv-1).abs().max(1).values
            stats.append(dict(frame=f, F_dev_bad_median=float(dev_[bad].median()), F_dev_good_median=float(dev_[~bad].median()),
                              F_dev_bad_max=float(dev_[bad].max()), F_dev_good_p99=float(torch.quantile(dev_[~bad].float(), .99)),
                              share_of_dev_over_0p5_that_is_bad=float(bad[dev_ > .5].float().mean()) if bool((dev_ > .5).any()) else 0.))
            tiles = []
            for v in variants:
                if v == 'rigid_owner':                      # each site follows only its own body, no blending
                    deformed.apply(pos[f], body_R=R, mode='rigid')
                else:
                    deformed.apply(pos[f], body_R=R, body_F=F if v == 'F' else R, mode='blend')
                col = []
                for c in range(len(views)):
                    rgb, al = render(foam, cams[c]); al = al.reshape(H, W)
                    m = masks[f, c] > .5
                    ious[v][f, c] = float(((al > .5) & m).sum()/((al > .5) | m).sum())
                    if c in show_c[:2]:
                        col.append(np.clip((rgb.clamp(0, 1)+(1-al.clamp(0, 1))[..., None]).cpu().numpy(), 0, 1))
                tiles.append(np.concatenate(col, 0))
            gt = np.concatenate([np.asarray(Image.open(SRC/args.obj/'data'/f'm_{views[c]}_{f}.png').convert('RGB'), np.float32)/255 for c in show_c[:2]], 0)
            img = Image.fromarray((np.concatenate([gt]+tiles, 1)*255).astype(np.uint8))
            d = ImageDraw.Draw(img)
            for k_, name in enumerate(('GT', 'blend, body F (as trained)', 'blend, rotation only', 'no blend: own body rigid')):
                d.text((k_*W+6, 4), name+('  frame %d' % f if k_ == 0 else ''), fill=(255, 0, 0))
            frames_img.append(img)
        frames_img[0].save(args.out/'render_check.gif', save_all=True, append_images=frames_img[1:], duration=300, loop=0)
        # do the bodies themselves leave their neighbourhood? non-affine residual of each body relative to the
        # least-squares F of its neighbours, in units of the body's own rest neighbour distance
        rest_len = torch.zeros(model.n, device=pos.device, dtype=torch.float64).index_add_(0, model.ls_a, model.ls_dX.double().norm(dim=-1))
        rest_len = rest_len/torch.bincount(model.ls_a, minlength=model.n).clamp(min=1)
        surf = torch.bincount(model.ls_a, minlength=model.n) < 9
        for f in (0, 6, 7, 8, 12, 20):
            x = pos[f].double(); Fd = model.deformation_gradient(pos[f]).double()
            r = (x[model.ls_b]-x[model.ls_a])-(Fd[model.ls_a]@model.ls_dX.double()[..., None])[..., 0]
            rb = torch.zeros(model.n, dtype=torch.float64, device=x.device).index_add_(0, model.ls_b, r.norm(dim=-1))
            rb = rb/torch.bincount(model.ls_b, minlength=model.n).clamp(min=1)/rest_len
            q = lambda t: tuple(float(torch.quantile(t.float(), u)) for u in (.5, .99))+(float(t.max()),)
            print('frame %2d non-affine offset / spacing: all median %.3f p99 %.3f max %.2f | <9 neighbours median %.3f p99 %.3f max %.2f | bodies below floor+1cm %d' % (
                (f,)+q(rb)+q(rb[surf])+(int((x[:, 2] < -.69).sum()),)))
    out = dict(bad_bodies=int(bad.sum()), iou_per_frame={v: ious[v].mean(1).tolist() for v in variants},
               iou_after_impact={v: float(ious[v][8:].mean()) for v in variants}, F_stats=stats)
    (args.out/'render_check.json').write_text(json.dumps(out, indent=1))
    for v in variants:
        print('%-14s IoU per frame %s | after impact (frames 8-) %.4f' % (v, np.round(ious[v].mean(1), 3).tolist(), out['iou_after_impact'][v]))
    for s_ in stats[::3]:
        print('frame %(frame)2d |sv-1| max: ill-cond median %(F_dev_bad_median).3f max %(F_dev_bad_max).2f | others median %(F_dev_good_median).3f p99 %(F_dev_good_p99).3f | share of |sv-1|>0.5 that is ill-cond %(share_of_dev_over_0p5_that_is_bad).2f' % s_)


def affine_check(args, model, deformed, foam, cams, train_cams, H, W):
    """Representation test, no physics: move every body by one global affine map x = c + A (X - c) (all
    body F = A, rotation = polar(A)) and render. A perfectly smooth deformation; anything that looks like
    debris here comes from the foam representation (sites moved, Power partition recomputed), not from
    the simulation."""
    from PIL import ImageDraw
    from experiments.foam_deform_render import polar_rotation
    X = model.X.float() if hasattr(model, 'X') else None
    X = model.initial_state()[0].float()
    c = X.mean(0)
    def rot(ax, deg):
        t = np.deg2rad(deg); C, S = np.cos(t), np.sin(t)
        M = np.eye(3); i, j = [k for k in range(3) if k != ax]
        M[i, i] = C; M[i, j] = -S; M[j, i] = S; M[j, j] = C
        return M
    cases = [('rest', np.eye(3)), ('rotation 30 deg', rot(2, 30)), ('uniform scale 0.85', .85*np.eye(3)),
             ('squash z 0.8, spread xy 1.12', np.diag([1.12, 1.12, .8])), ('squash z 0.65, spread xy 1.24', np.diag([1.24, 1.24, .65])),
             ('shear xz 0.3', np.array([[1, 0, .3], [0, 1, 0], [0, 0, 1.]]))]
    shows = [train_cams[0], train_cams[len(train_cams)//2]]
    rows = []
    with torch.no_grad():
        for name, A in cases:
            At = torch.tensor(A, dtype=torch.float32, device=X.device)
            pos = c+(X-c)@At.T
            F = At.expand(model.n, 3, 3).contiguous()
            R = polar_rotation(F.double()).float()
            deformed.apply(pos, body_R=R, body_F=F, mode='blend')
            tiles = []
            for cam in shows:
                rgb, al = render(foam, cams[cam]); al = al.reshape(H, W).clamp(0, 1)
                tiles.append(np.clip((rgb.clamp(0, 1)+(1-al)[..., None]).cpu().numpy(), 0, 1))
            img = Image.fromarray((np.concatenate(tiles, 0)*255).astype(np.uint8))
            ImageDraw.Draw(img).text((6, 4), name, fill=(255, 0, 0))
            rows.append(img)
    W_ = sum(r.width for r in rows)
    sheet = Image.new('RGB', (W_, rows[0].height), 'white')
    x_ = 0
    for r in rows:
        sheet.paste(r, (x_, 0)); x_ += r.width
    sheet.save(args.out/'affine_check.png')
    print(args.out/'affine_check.png')


def jump_check(args, params, model, rollout, substeps_now):
    """The DEM's own map: every cell moves rigidly with its body, phi(X) = x_i + R_i (X - X_i). Across a
    shared face it jumps by D = (x_j + R_j r_j) - (x_i + R_i r_i) (the bond displacement at the face
    point). Debris = faces where this jump is large compared with the bond length. Where are they, are
    they on the floor, how much elastic energy do they hold, and do they persist?"""
    from experiments.foam_dem_energy import quat_matrix
    rep = json.loads((args.out/'report.json').read_text())
    with torch.no_grad():
        params.copy_(torch.tensor(rep['best_params'], dtype=params.dtype, device=params.device))
        pos, quat = rollout(args.frames, substeps_now(), grad=False)
        th = theta_from(params, model) if False else None
        L = (model.X[model.j]-model.X[model.i]).norm(dim=-1).double()
        nbonds = torch.bincount(torch.cat((model.i, model.j)), minlength=model.n)
        surf = nbonds < 9
        floor_z = -.7
        rc = model.contact_radius.double()
        prev = None
        for f in (0, 6, 7, 8, 10, 12, 16, 20, 23):
            x = pos[f].double(); q = quat[f].double()
            Ri, Rj = quat_matrix(q[model.i]), quat_matrix(q[model.j])
            D = x[model.j]+(Rj@model.rj.double()[..., None])[..., 0]-x[model.i]-(Ri@model.ri.double()[..., None])[..., 0]
            jump = D.norm(dim=-1)/L
            rel = torch.rad2deg(torch.arccos((((Ri.transpose(1, 2)@Rj).diagonal(dim1=-2, dim2=-1).sum(-1)-1)/2).clamp(-1, 1)))
            on_floor = (x[:, 2]-rc < floor_z+.01)
            bf = on_floor[model.i] | on_floor[model.j]
            bs = surf[model.i] | surf[model.j]
            big = jump > .3
            v = (x-prev).norm(dim=-1)/.05 if prev is not None else torch.zeros(model.n, dtype=x.dtype, device=x.device)
            prev = x
            q_ = lambda t: (float(t.median()), float(torch.quantile(t.float(), .99)), float(t.max()))
            print('frame %2d | face jump / bond length: median %.3f p99 %.3f max %.2f | bonds > 0.3: %d (%.2f%%), of which touching floor %.2f, on surface bodies %.2f | rel. rotation of those: median %.0f deg | all bonds rel. rot p99 %.0f deg | bodies on floor %d | median speed %.3f m/s' % (
                (f,)+q_(jump)+(int(big.sum()), 100*float(big.float().mean()), float(bf[big].float().mean()) if big.any() else 0., float(bs[big].float().mean()) if big.any() else 0.,
                 float(rel[big].median()) if big.any() else 0., float(torch.quantile(rel.float(), .99)), int(on_floor.sum()), float(v.median()))), flush=True)
            if f == 20:
                ids = torch.nonzero(big).flatten()
                persistent = ids
        # persistence: of the bonds with a big jump at frame 20, how many already had it at frame 8?
        x8 = pos[8].double(); q8 = quat[8].double()
        Ri, Rj = quat_matrix(q8[model.i]), quat_matrix(q8[model.j])
        D8 = x8[model.j]+(Rj@model.rj.double()[..., None])[..., 0]-x8[model.i]-(Ri@model.ri.double()[..., None])[..., 0]
        j8 = D8.norm(dim=-1)/L
        print('big-jump bonds at frame 20: %d; %.2f of them already > 0.3 at frame 8' % (len(persistent), float((j8[persistent] > .3).float().mean()) if len(persistent) else 0.))
        # spatial: height of big-jump bonds above floor (bond midpoint), vs all bonds
        x = pos[20].double(); mid = .5*(x[model.i]+x[model.j])[:, 2]-floor_z
        if len(persistent):
            print('height above floor of big-jump bonds (cm): median %.1f, p10 %.1f, p90 %.1f | all bonds median %.1f | sofa height %.1f' % (
                100*float(mid[persistent].median()), 100*float(torch.quantile(mid[persistent].float(), .1)), 100*float(torch.quantile(mid[persistent].float(), .9)),
                100*float(mid.median()), 100*float(x[:, 2].max()-x[:, 2].min())))


def tangle_check(args, params, model, rollout, substeps_now, theta_of):
    """Are the persistent large face jumps after impact legitimate weight-bearing compression, or bodies
    that were squeezed out / pushed through their neighbours (a new local minimum)? For the bonds with
    jump > 0.3 at the last frame: normal vs shear decomposition (exact kinematics, mid frame), inverted
    bonds (lambda_n < 0), interpenetrating non-bonded body pairs, and the elastic energy they hold."""
    from scipy.spatial import cKDTree
    rep = json.loads((args.out/'report.json').read_text())
    with torch.no_grad():
        params.copy_(torch.tensor(rep['best_params'], dtype=params.dtype, device=params.device))
        E_now = float(torch.exp(params[0]))
        pos, quat = rollout(args.frames, substeps_now(), grad=False)
        th = theta_of(params)[:, :7].contiguous()
        L = (model.X[model.j]-model.X[model.i]).norm(dim=-1)
        Lmed = float(L.median())
        height = float(model.X[:, 2].max()-model.X[:, 2].min())
        print('E %.0f Pa: static strain under own weight rho g h / E = %.2f (h = %.2f m)' % (E_now, 1000*9.8*height/E_now, height))
        bonded = set(map(tuple, torch.stack((torch.minimum(model.i, model.j), torch.maximum(model.i, model.j)), 1).cpu().numpy().tolist()))
        v6 = (pos[6]-pos[5]).norm(dim=-1)/.05
        KE = float(.5*(model.mass*v6**2).sum())
        print('kinetic energy before impact (frame 5->6 speeds, median %.2f m/s): %.3f J' % (float(v6.median()), KE))
        big_last = None
        for f in (0, 7, 8, 12, 23):
            x, q = pos[f], quat[f]
            *_, k, dn, dt, pn, pt = model._bond_exact(th, x, q)
            dtn = dt.norm(dim=-1)
            jump = torch.sqrt(dn**2+dtn**2)/L
            lam = 1+dn/L
            eb = .5*(k[:, 0]*dn**2+k[:, 1]*dtn**2+k[:, 2]*(pt*pt).sum(-1)+k[:, 3]*pn**2)
            ebody = model._body_energy_and_stress(th, x)[0]
            c, r = model._residual(th, x); eres = (c*(r*r).sum(-1)).sum()
            Etot = float(eb.sum()+ebody.sum()+eres)
            big = jump > .3
            if f == 23:
                big_last = big
            comp = big & (dn < 0) & (dn.abs() >= dtn)
            tens = big & (dn > 0) & (dn.abs() >= dtn)
            shear = big & (dtn > dn.abs())
            inv = lam < 0
            xc = x.cpu().numpy()
            tree = cKDTree(xc)
            close = tree.query_pairs(.35*Lmed)
            nonb = [pq for pq in close if tuple(sorted(pq)) not in bonded]
            print('frame %2d | elastic energy %.3f J (bonds %.3f, body NH %.3f, residual %.3f) = %.2f x impact KE | jump>0.3: %d  -> normal compression %d, normal tension %d, shear-dominated %d | inverted bonds (lambda_n<0) %d, lambda_n<0.3: %d | body pairs closer than 0.35 spacing: %d, of which NOT bonded %d | energy in jump>0.3 bonds: %.1f%% of bond energy' % (
                f, Etot, float(eb.sum()), float(ebody.sum()), float(eres), Etot/KE, int(big.sum()), int(comp.sum()), int(tens.sum()), int(shear.sum()),
                int(inv.sum()), int((lam < .3).sum()), len(close), len(nonb), 100*float(eb[big].sum()/eb.sum()) if eb.sum() > 0 else 0.), flush=True)
            if f == 23:
                zmid = (.5*(x[model.i]+x[model.j])[:, 2]+.7)
                lowb = zmid < .04
                print('   frame 23, jump>0.3 bonds within 4 cm of the floor: %d (compression %d, tension %d, shear %d); higher up: %d (compression %d, tension %d, shear %d)' % (
                    int((big & lowb).sum()), int((comp & lowb).sum()), int((tens & lowb).sum()), int((shear & lowb).sum()),
                    int((big & ~lowb).sum()), int((comp & ~lowb).sum()), int((tens & ~lowb).sum()), int((shear & ~lowb).sum())))
                q_ = lambda t: tuple(float(torch.quantile(t.float(), u)) for u in (.5, .9)) + (float(t.max()),)
                print('   frame 23, jump>0.3 bonds: |shear|/L median %.2f p90 %.2f max %.2f ; lambda_n median %.2f min %.2f ; relative rotation (deg) median %.0f max %.0f' % (
                    q_(dtn[big]/L[big]) + (float(lam[big].median()), float(lam[big].min())) + (float(torch.rad2deg(pt.norm(dim=-1)[big]).median()), float(torch.rad2deg(pt.norm(dim=-1)[big]).max()))))
                print('   all bonds at frame 23: lambda_n median %.3f p1 %.3f ; |shear|/L median %.3f p99 %.3f' % (
                    float(lam.median()), float(torch.quantile(lam.float(), .01)), float((dtn/L).median()), float(torch.quantile((dtn/L).float(), .99))))


def gt_material_check(args, params, gt, logit, evaluate, substeps_now, rollout, place, foam, cams, views, train_cams, H, W):
    """No training: forward rollout at the ground-truth material (E, nu from gt_phys_params.yaml), with the
    default damping/contact and with the trained damping/contact, against the trained parameters.
    Per-frame IoU over all views, plus an image strip (GT | variants) at a few frames."""
    from PIL import ImageDraw
    rep = json.loads((args.out/'report.json').read_text())
    tr = torch.tensor(rep['best_params'], dtype=params.dtype, device=params.device)
    E, nu = gt['yms'], gt['prs']
    variants = {'GT material, default damping+contact': torch.tensor([np.log(E), logit(nu/.49), 0., 0., 0.], dtype=params.dtype, device=params.device)}
    out, rolls = {}, {}
    for name, p in variants.items():
        with torch.no_grad():
            params.copy_(p)
            sub = substeps_now()
            r = evaluate(sub)
            rolls[name] = rollout(args.frames, sub, grad=False)
        out[name] = r
        iou = np.array(r['iou_per_frame'])
        print('%-40s E %6.0f nu %.3f damping %5.1f substeps %4d | IoU frames 1-16 %.3f, 17-23 %.3f, after impact (8-) %.3f | per frame %s' % (
            name, r['E'], r['nu'], r['damping'], sub, iou[1:17].mean(), iou[17:].mean(), iou[8:].mean(), np.round(iou, 2).tolist()), flush=True)
    (args.out/'gt_material_check.json').write_text(json.dumps(out, indent=1))
    frames_show, c = (6, 8, 11, 14, 20), train_cams[0]
    rows = []
    with torch.no_grad():
        row = [np.asarray(Image.open(SRC/args.obj/'data'/f'm_{views[c]}_{f}.png').convert('RGB'), np.float32)/255 for f in frames_show]
        rows.append(np.concatenate(row, 1))
        for name, (pos, quat) in rolls.items():
            row = []
            for f in frames_show:
                place(pos, quat, f)
                rgb, al = render(foam, cams[c]); al = al.reshape(H, W).clamp(0, 1)
                row.append(np.clip((rgb.clamp(0, 1)+(1-al)[..., None]).cpu().numpy(), 0, 1))
            rows.append(np.concatenate(row, 1))
    img = Image.fromarray((np.concatenate(rows, 0)*255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    for k, name in enumerate(['GT video']+list(variants)):
        d.text((6, k*H+4), name, fill=(255, 0, 0))
    for k, f in enumerate(frames_show):
        d.text((k*W+6, H-14), 'frame %d' % f, fill=(255, 0, 0))
    img.save(args.out/'gt_material_check.png')
    print(args.out/'gt_material_check.png')


def params_render(args, params, rollout, substeps_now, place, foam, cams, views, train_cams, H, W):
    """Render the DEM with the parameters of --params-from (e.g. the 3D-trajectory training) next to the released
    video and the regenerated ground truth (data/vid2sim_gt/<obj>, surface samples of the deformed mesh projected to
    the camera): GIF over all frames (views 0 and 3) and a strip of selected frames."""
    import trimesh
    from PIL import ImageDraw
    from scipy import ndimage
    rep = json.loads(args.params_from.read_text())
    gt = np.load(Path(f'data/vid2sim_gt/{args.obj}')/'trajectory.npz'); V, Fc = gt['vertices'], gt['faces']
    meta = json.loads((SRC/args.obj/'transforms_simulation.json').read_text())
    f = .5*W/np.tan(.5*meta['camera_angle_x'])

    def gt_mask(t, c):
        pts, _ = trimesh.sample.sample_surface(trimesh.Trimesh(V[t], Fc, process=False), 150000)
        w2c = np.linalg.inv(np.array(meta['frames'][c]['transform_matrix']))
        cam = pts@w2c[:3, :3].T+w2c[:3, 3]; z = -cam[:, 2]
        u = (f*cam[:, 0]/z+W/2).astype(int); v = (-f*cam[:, 1]/z+H/2).astype(int)
        ok = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        img = np.zeros((H, W), bool); img[v[ok], u[ok]] = True
        return ndimage.binary_closing(img, iterations=2)
    with torch.no_grad():
        params.copy_(torch.tensor(rep['best_params'], dtype=params.dtype, device=params.device))
        pos, quat = rollout(args.frames, substeps_now(), grad=False)
        show_c = [train_cams[0], 3]
        frames_img, ious = [], []
        for t in range(args.frames+1):
            place(pos, quat, t)
            tiles = []
            for c in show_c:
                vid = views[c]
                video = np.asarray(Image.open(SRC/args.obj/'data'/f'm_{vid}_{t}.png').convert('RGB'), np.float32)/255
                rgb, al = render(foam, cams[c]); al = al.reshape(H, W).clamp(0, 1)
                sim = np.clip((rgb.clamp(0, 1)+(1-al)[..., None]).cpu().numpy(), 0, 1)
                g = gt_mask(t, c); d = al.cpu().numpy() > .5
                ious.append(float((g & d).sum()/max((g | d).sum(), 1)))
                gt_img = np.ones((H, W, 3), np.float32); gt_img[g] = (.75, .75, .75)
                ov = np.zeros((H, W, 3), np.float32); ov[..., 0] = g; ov[..., 1] = d
                tiles.append(np.concatenate((video, gt_img, sim, ov), 1))
            img = Image.fromarray((np.concatenate(tiles, 0)*255).astype(np.uint8))
            dr = ImageDraw.Draw(img)
            for k_, name in enumerate(('released video', 'regenerated 3D GT (our generator)', 'DEM, 3D-trained params', 'overlay: red GT, green DEM')):
                dr.text((k_*W+6, 4), name+('   frame %d' % t if k_ == 0 else ''), fill=(255, 0, 0))
            for k_, c in enumerate(show_c):
                dr.text((6, k_*H+H-14), 'view %d' % views[c], fill=(255, 0, 0))
            frames_img.append(img)
    out = args.params_from.parent
    frames_img[0].save(out/'compare_regen.gif', save_all=True, append_images=frames_img[1:], duration=300, loop=0)
    sel = [6, 8, 11, 14, 20, 23]
    strip = Image.new('RGB', (frames_img[0].width, frames_img[0].height*len(sel)))
    for k_, t in enumerate(sel):
        strip.paste(frames_img[t], (0, k_*frames_img[0].height))
    strip.save(out/'compare_regen_strip.png')
    ious = np.array(ious).reshape(args.frames+1, len(show_c))
    print('DEM vs regenerated GT silhouette IoU per frame:', np.round(ious.mean(1), 3).tolist())
    print('mean frames 1-16 %.3f, 17-23 %.3f' % (ious[1:17].mean(), ious[17:].mean()))
    print(out/'compare_regen.gif')


if __name__ == '__main__':
    main()
