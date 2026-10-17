"""Spring-Gaus MPM-synthetic object: learn DEM material from multi-view silhouettes, check against GT.

Scene = simulation world (offset 0, metres, gravity and ground plane from physical.json, initial
velocity from physical.json, 0.03 s between frames). Static foam: frame 0, all 10 views
(springgaus_synth_to_powerfoam + train.py with the spacing constraint). DEM: pacnerf_foam_geometry.
Training: Warp rollout (deterministic, segment checkpointing), blend render of the kept cells
(material + 2-ring air shell), silhouette L2 against the rendered alpha of the training views,
frames 1..--train-frames. Split as Spring-Gaus: --eval-cam held out, the other 9 views train.
Parameters: 7 log multipliers (+ friction, restitution logits with --contact smooth), shared or
per body. The model is built at --E/--nu (default: deliberately off the GT) so the fit has to
find the material; theta = 0 is that start.

Evaluation: silhouette IoU (train views / eval view x train / future frames) and the 3D trajectory
normalized RMSE of body centres against the GT MPM particles (frame-0 particles assigned to the
nearest body centre; never used in training). Effective (E, nu) from the body-energy multipliers
(lam, mu) at density 1000 (only E/rho is identifiable; Spring-Gaus does not state its density).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.spatial import cKDTree

from experiments.foam_deform_render import DeformedFoam
from experiments.foam_dem_warp import WarpFoam, rollout as warp_rollout
from experiments.foam_dynamic_render import air_shell, keep_cells, load, render
from experiments.pacnerf_dem_forward import build
from experiments.pacnerf_render_dem import body_rotation
from experiments.springgaus_synth_to_powerfoam import SRC, read_ply


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--obj', default='torus')
    a.add_argument('--run', default=None, help='default output/springgaus_synth_<obj>_spacing8k')
    a.add_argument('--dem', type=Path, default=None, help='default output/springgaus_synth_dem/<obj>')
    a.add_argument('--eval-cam', type=int, default=5)
    a.add_argument('--train-frames', type=int, default=19)
    a.add_argument('--frames', type=int, default=29, help='last frame evaluated')
    a.add_argument('--E', type=float, default=2e5)
    a.add_argument('--nu', type=float, default=.3)
    a.add_argument('--iterations', type=int, default=40)
    a.add_argument('--lr', type=float, default=.05)
    a.add_argument('--contact', default='smooth', choices=('hard', 'smooth'))
    a.add_argument('--layer-dx', type=float, default=.005)
    a.add_argument('--per-body', action='store_true')
    a.add_argument('--smooth-weight', type=float, default=1e-2)
    a.add_argument('--init-from', type=Path)
    a.add_argument('--segment', type=int, default=200)
    a.add_argument('--time-only', type=int, default=0, help='run N training iterations, print timing, exit')
    a.add_argument('--out', type=Path, required=True)
    args = a.parse_args()
    args.run = args.run or f'output/springgaus_synth_{args.obj}_spacing8k'
    args.dem = args.dem or Path(f'output/springgaus_synth_dem/{args.obj}')
    args.out.mkdir(parents=True, exist_ok=True)
    dev, f32 = 'cuda', torch.float32
    render_dir = SRC/'render'/args.obj
    phys = json.loads((render_dir/'physical.json').read_text())
    dt_frame = float(json.loads((render_dir/'frame.json').read_text())[1]['001'])
    ground_p, ground_n = phys[1]['BC']['GROUND']
    geometry = dict(np.load(args.dem/'geometry.npz')); cells = dict(np.load(args.dem/'cells.npz'))
    foam, data = load(args.run)
    names = [Path(f['file_path']).name for f in json.loads(
        (Path('data/springgaus_synth_powerfoam')/f'{args.obj}_f0'/'transforms_all.json').read_text())['frames']]
    keep, owner = air_shell(foam, cells['source_index'], geometry['body_of_cell'], rings=2)
    keep_cells(foam, keep)
    for n_ in ('radii', 'density', 'texel_sites', 'texel_sv_axis', 'texel_sv_rgb', 'texel_height'):
        getattr(foam, n_).requires_grad_(False)
    deformed = DeformedFoam(foam, owner, geometry['points'], np.zeros(3))
    deformed.prepare_blend(geometry['points'], K=24, width=2.)
    cams = data.cameras
    W, H = cams[0].width, cams[0].height
    masks = {}
    for f in range(args.frames+1):
        for c, n_ in enumerate(names):
            al = np.asarray(Image.open(render_dir/n_/f'{f:03d}.png').convert('RGBA'), np.float32)[..., 3]/255
            masks[f, c] = torch.tensor(al, device=dev)
    eval_c = names.index(f'camera_{args.eval_cam:03d}')
    train_cams = [c for c in range(len(names)) if c != eval_c]
    # GT body trajectories: frame-0 particles -> nearest body centre
    X = geometry['points']
    p0 = read_ply(SRC/'simulation'/args.obj/'000.ply')
    owner_p = cKDTree(X).query(p0)[1]
    count = np.bincount(owner_p, minlength=len(X))
    has = count > 0
    targets = []
    for f in range(args.frames+1):
        pf = read_ply(SRC/'simulation'/args.obj/f'{f:03d}.ply')
        s = np.zeros_like(X); np.add.at(s, owner_p, pf)
        targets.append(s/np.maximum(count, 1)[:, None])
    targets = np.array(targets)

    model = build(geometry, args.E, args.nu, dev, 'noslip', .25, .3, f32)
    model.set_environment(gravity=phys[1]['GRAVITY'], ground_point=ground_p, ground_normal=ground_n,
                          contact_radius=model.contact_radius.cpu().numpy(), ground_mode='noslip')
    x0, q0, _, w0 = model.initial_state()
    v0 = torch.tensor(phys[1]['INIT_VELOCITY'], dtype=f32, device=dev).expand_as(x0).contiguous()
    ncol = 9 if args.contact == 'smooth' else 7
    base = torch.zeros(ncol, dtype=f32, device=dev)
    if args.init_from is not None:
        start = json.loads(args.init_from.read_text())['best']['parameters']
        start = start['mean'] if isinstance(start, dict) else start
        base[:min(len(start), ncol)] = torch.tensor(start[:ncol], dtype=f32, device=dev)
    if args.per_body:
        shared = torch.zeros(model.n, ncol, dtype=f32, device=dev, requires_grad=True)
        theta_of = lambda p: base.expand(model.n, ncol)+p
    else:
        shared = base.clone().requires_grad_(True)
        theta_of = lambda p: p.expand(model.n, ncol)
    optimizer = torch.optim.Adam([shared], lr=args.lr)
    wf = WarpFoam(model, contact=args.contact, layer_dx=args.layer_dx)

    def summary(p):
        th = theta_of(p.detach())
        return th[0].tolist() if not args.per_body else dict(mean=th.mean(0).tolist(), std=th.std(0).tolist())

    def rollout(n, substeps, grad=True):
        theta = theta_of(shared if grad else shared.detach())
        return warp_rollout(wf, theta, (x0, q0, v0, w0), dt_frame/substeps, substeps, n, args.segment if grad else None)

    def place(pos, quat, f):
        deformed.apply(pos[f], body_R=body_rotation(quat[f]), body_F=model.deformation_gradient(pos[f]), mode='blend')

    def silhouette(c):
        _, alpha = render(foam, cams[c])
        return alpha.reshape(H, W)

    def material(th_mean):
        lam0, mu0 = args.E*args.nu/((1+args.nu)*(1-2*args.nu)), args.E/(2*(1+args.nu))
        lam, mu = lam0*np.exp(th_mean[4]), mu0*np.exp(th_mean[5])
        return dict(E=float(mu*(3*lam+2*mu)/(lam+mu)), nu=float(lam/(2*(lam+mu))))

    def evaluate(substeps):
        with torch.no_grad():
            pos, quat = rollout(args.frames, substeps, grad=False)
            ious = np.zeros((args.frames+1, len(names)))
            for f in range(args.frames+1):
                place(pos, quat, f)
                for c in range(len(names)):
                    al = silhouette(c) > .5; m = masks[f, c] > .5
                    ious[f, c] = float((al & m).sum()/(al | m).sum())
        P = pos.cpu().numpy()

        def nrmse(lo, hi):
            pp, tt = P[lo:hi+1, has], targets[lo:hi+1, has]
            return float(np.sqrt(np.mean((pp-tt)**2)/np.mean((tt-targets[:1, has])**2)))
        tr, te = slice(1, args.train_frames+1), slice(args.train_frames+1, args.frames+1)
        th = theta_of(shared.detach()).mean(0).cpu().numpy()
        return dict(iou_trainviews_trainframes=float(ious[tr][:, train_cams].mean()), iou_evalview_trainframes=float(ious[tr, eval_c].mean()),
                    iou_trainviews_futureframes=float(ious[te][:, train_cams].mean()), iou_evalview_futureframes=float(ious[te, eval_c].mean()),
                    traj_nrmse_trainframes=nrmse(1, args.train_frames), traj_nrmse_futureframes=nrmse(args.train_frames+1, args.frames),
                    traj_nrmse_all=nrmse(1, args.frames), material_effective=material(th),
                    iou_per_frame_cam=ious.tolist(), parameters=summary(shared), substeps=substeps)

    def substeps_now():
        return int(np.ceil(dt_frame/model.stable_dt(theta_of(shared.detach()))))

    def train_step(substeps):
        optimizer.zero_grad()
        pos, quat = rollout(args.train_frames, substeps)
        loss = 0.
        for f in range(1, args.train_frames+1):
            place(pos, quat, f)
            for c in train_cams:
                loss = loss+((silhouette(c)-masks[f, c])**2).mean()
        loss = loss/(args.train_frames*len(train_cams))
        value = float(loss.detach())
        if args.per_body and args.smooth_weight > 0:
            th = theta_of(shared)
            loss = loss+args.smooth_weight*((th[model.i]-th[model.j])**2).sum(-1).mean()
        loss.backward()
        return value

    substeps = substeps_now()
    if args.time_only:
        t0 = time.perf_counter()
        for _ in range(args.time_only):
            train_step(substeps)
        torch.cuda.synchronize()
        print('bodies %d, substeps/frame %d, %.1f s per training iteration' % (model.n, substeps, (time.perf_counter()-t0)/args.time_only))
        return
    report = dict(config={k_: str(v) for k_, v in vars(args).items()}, gt=phys[0], initial=evaluate(substeps))
    show = lambda r: {k_: (round(v, 4) if isinstance(v, float) else v) for k_, v in r.items() if k_ not in ('iou_per_frame_cam', 'parameters')}
    print('initial', json.dumps(show(report['initial'])), flush=True)
    history, best = [], (np.inf, shared.detach().clone())
    start = time.perf_counter()
    for it in range(args.iterations):
        if it and it % 5 == 0:
            substeps = substeps_now()
        value = train_step(substeps)
        if not np.isfinite(value) or not torch.isfinite(shared.grad).all():
            print('non-finite at iteration', it, flush=True); break
        if value < best[0]:
            best = (value, shared.detach().clone())
        optimizer.step()
        with torch.no_grad():
            shared.clamp_(-4, 4)
        history.append(dict(iteration=it, silhouette_mse=value, grad_norm=float(shared.grad.norm()), parameters=summary(shared),
                            material=material(theta_of(shared.detach()).mean(0).cpu().numpy()), substeps=substeps,
                            seconds=time.perf_counter()-start))
        print(json.dumps(history[-1]), flush=True)
        (args.out/'history.json').write_text(json.dumps(history, indent=1))
    with torch.no_grad():
        shared.copy_(best[1])
    report['best'] = evaluate(substeps_now())
    report['seconds'] = time.perf_counter()-start
    mean = theta_of(best[1]).mean(0).cpu().numpy()
    report['multipliers'] = dict(zip(('kn', 'ks', 'kb', 'kt', 'lam', 'mu', 'damping'), np.exp(mean[:7]).tolist()))
    if ncol == 9:
        sig = lambda z: float(1/(1+np.exp(-z)))
        report['multipliers'].update(friction=sig(mean[7]+wf.friction0), restitution=sig(mean[8]+wf.restitution0))
    np.savez_compressed(args.out/'theta_best.npz', theta=theta_of(best[1]).cpu().numpy())
    (args.out/'report.json').write_text(json.dumps(report, indent=1))
    print('best', json.dumps(show(report['best'])), json.dumps(report['multipliers']), flush=True)


if __name__ == '__main__':
    main()
