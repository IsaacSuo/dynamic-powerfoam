"""Loss landscape of the shared parameters on the E x damping plane (forward rollouts only, no gradients):
is the plateau of the gradient training a flat valley with a lower point, or the floor of the model?
Same setup as vid2sim_traj_train.py (shell model, nu fixed at the ground truth, friction / restitution from the
best report), training loss = normalized MSE of body centres over frames 1..16 against the LBS ground truth."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
import warp as wp

from experiments.foam_dem_warp import WarpFoam, rollout as warp_rollout
from experiments.pacnerf_dem_forward import build
from experiments.vid2sim_traj_train import lbs_targets


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--obj', default='sofa')
    a.add_argument('--report', type=Path, default=Path('output/vid2sim_dem/sofa/traj_train_shell/report.json'))
    a.add_argument('--E', type=float, nargs='+', default=[12e3, 15e3, 18e3, 22e3, 26e3, 30e3])
    a.add_argument('--damping', type=float, nargs='+', default=[15., 25., 40., 60., 85., 120.])
    a.add_argument('--train-frames', type=int, default=16)
    a.add_argument('--frames', type=int, default=23)
    a.add_argument('--out', type=Path, default=Path('output/vid2sim_dem/sofa/loss_sweep'))
    args = a.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    wp.init(); wp.set_mempool_release_threshold('cuda', 0)
    dev, f32 = 'cuda', torch.float32
    gt = yaml.safe_load(open(f'data/vid2sim/ext/{args.obj}/gt_phys_params.yaml')); nu = float(gt['prs'])
    rep = json.loads(args.report.read_text()); p = rep['best_params']
    geometry = dict(np.load(f'output/vid2sim_dem/{args.obj}/geometry.npz'))
    E0 = 2e4
    model = build(geometry, E0, nu, dev, 'noslip', .25, .3, f32, kinematics='exact', residual=True, shell=True)
    model.set_environment(gravity=[0., 0., -9.8], ground_point=[0., 0., -.7], ground_normal=[0., 0., 1.], contact_radius=model.contact_radius.cpu().numpy(), ground_mode='noslip')
    model_dt = build(geometry, E0, nu, dev, 'noslip', .25, .3, torch.float64, kinematics='exact', residual=True, shell=True, shell_eas=False)
    wf = WarpFoam(model, contact='smooth', layer_dx=.01, implicit_damping=True, graphs=True)
    x0, q0, v0, w0 = model.initial_state(); v0 = (v0+torch.tensor([0., 0., -9.8], dtype=f32, device=dev)*.05/2).contiguous()
    targets, _ = lbs_targets(Path(f'data/vid2sim_gt/{args.obj}'), geometry['points'], dev); targets = targets[:args.frames+1]
    lam0, mu0 = E0*nu/((1+nu)*(1-2*nu)), E0/(2*(1+nu))
    denom_tr = ((targets[1:args.train_frames+1]-targets[:1])**2).sum(-1).mean()
    denom_all = ((targets[1:]-targets[:1])**2).sum(-1).mean()

    def theta_of(E, damping):
        mu, lam = E/(2*(1+nu)), E*nu/((1+nu)*(1-2*nu)); lm, ll = np.log(mu/mu0), np.log(lam/lam0)
        return torch.tensor([lm, lm, lm, lm, ll, lm, np.log(damping), p[3], p[4]], dtype=f32, device=dev).expand(model.n, 9).contiguous()

    def losses(E, damping):
        th = theta_of(E, damping)
        sub = int(np.ceil(.05/model_dt.stable_dt(th[:, :7].double(), safety=.4, include_damping=False)))
        for k in range(4):
            with torch.no_grad():
                pos, _ = warp_rollout(wf, th, (x0, q0, v0, w0), .05/sub, sub, args.frames, None)
            if bool(torch.isfinite(pos).all()):
                break
            sub = int(sub*1.5)
        ltr = float(((pos[1:args.train_frames+1]-targets[1:args.train_frames+1])**2).sum(-1).mean()/denom_tr)
        lall = float(((pos[1:]-targets[1:])**2).sum(-1).mean()/denom_all)
        return ltr, lall, sub
    grid = np.zeros((len(args.E), len(args.damping), 2)); subs = np.zeros((len(args.E), len(args.damping)), int)
    t0 = time.perf_counter()
    print('train loss (normalized MSE, frames 1-16) | columns: damping %s' % args.damping, flush=True)
    for i, E in enumerate(args.E):
        row = []
        for j, dmp in enumerate(args.damping):
            ltr, lall, sub = losses(E, dmp); grid[i, j] = (ltr, lall); subs[i, j] = sub
            row.append('%.5f' % ltr)
        print('E %6.0f: %s   (%.0f s)' % (E, '  '.join(row), time.perf_counter()-t0), flush=True)
    i, j = np.unravel_index(grid[..., 0].argmin(), grid[..., 0].shape)
    print('minimum of the grid: E %.0f damping %.0f -> train loss %.5f (all frames %.5f); gradient training best: train loss %.5f at E %.0f damping %.1f' % (
        args.E[i], args.damping[j], grid[i, j, 0], grid[i, j, 1], min(h['loss'] for h in json.loads((args.report.parent/'history.json').read_text())), np.exp(p[0]), np.exp(p[2])))
    np.savez(args.out/'sweep.npz', E=args.E, damping=args.damping, loss_train=grid[..., 0], loss_all=grid[..., 1], substeps=subs)
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    cs = ax.contourf(args.damping, args.E, np.log10(grid[..., 0]), levels=20, cmap='viridis'); plt.colorbar(cs, label='log10 train loss')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('damping multiplier'); ax.set_ylabel('E (Pa)'); ax.set_title('sofa, shell model, nu fixed: train loss on the E x damping plane')
    ax.plot([np.exp(p[2])], [np.exp(p[0])], 'r*', ms=12, label='gradient training best'); ax.axhline(24107, color='w', ls='--', lw=.8, label='E of the ground truth'); ax.legend()
    plt.tight_layout(); plt.savefig('output/report_v2/vid2sim_sofa_损失面_E×阻尼.png', dpi=110); print('saved figure')


if __name__ == '__main__':
    main()
