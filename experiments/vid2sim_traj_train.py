"""Train the DEM material on the regenerated Vid2Sim 3D ground truth (experiments/vid2sim_regen_gt.py): full
3D trajectory supervision, no rendering in the loss.

Targets: the generator's LBS deformation field evaluated at every DEM body's rest centre, x_t(X_i) = X_i +
sum_h w_h(X_i) T_h,t [X_i; 1] (the field is defined everywhere, so bodies reconstructed outside the GSO mesh get
the smooth extension). Loss: normalized MSE of body centres over frames 1..--train-frames; later frames held out.
Same model and parameters as vid2sim_train.py (E, nu, damping, friction, restitution; exact bond kinematics,
residual term, implicit damping, CUDA graphs); Adam; stable substeps recomputed every 5 iterations.
Report: nRMSE train / held-out / all, per-body error (surface vs interior, height), learned parameters, and the
body-centre error of the released parameters. best_params is saved in the vid2sim_train.py report format.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from experiments.foam_dem_warp import WarpFoam, rollout as warp_rollout
from experiments.pacnerf_dem_forward import build
from experiments.vid2sim_regen_gt import SimplicitsMLP, lbs

SRC = Path('data/vid2sim/ext')


def lbs_targets(gt_dir, X, dev):
    """(frames+1, n, 3) positions of the rest points X under the generator's handles z_t."""
    tr = np.load(gt_dir/'trajectory.npz')
    model = SimplicitsMLP().to(dev)
    model.load_state_dict(torch.load(gt_dir/'lbs_model.pth', map_location=dev, weights_only=True)); model.eval()
    Xt = torch.tensor(X, dtype=torch.float32, device=dev)
    with torch.no_grad():
        w = torch.cat((model(Xt), torch.ones((Xt.shape[0], 1), device=dev)), 1)
        z = torch.tensor(tr['z'], dtype=torch.float32, device=dev)                  # (frames+1, 132)
        T = lbs(Xt, z.reshape(z.shape[0], -1, 3, 4), w).transpose(0, 1)            # (frames+1, n, 3)
    return T, tr


def nrmse(pred, target, lo, hi):
    p, t = pred[lo:hi+1], target[lo:hi+1]
    return float(torch.sqrt(((p-t)**2).sum(-1).mean()/((t-target[:1])**2).sum(-1).mean()))


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--obj', default='sofa')
    a.add_argument('--gt', type=Path, default=None, help='regenerated GT dir (default data/vid2sim_gt/<obj>)')
    a.add_argument('--gt-targets', type=Path, default=None, help='targets.npz from vid2sim_mpm_gt.py (continuum GT) instead of the LBS field')
    a.add_argument('--E', type=float, default=2e4, help='initial E (deliberately off the GT)')
    a.add_argument('--nu', type=float, default=.3)
    a.add_argument('--fix-nu', type=int, default=0, help='1: nu fixed at the ground-truth value, not trained')
    a.add_argument('--init-from', type=Path, default=None, help='continue from the best_params of this report.json')
    a.add_argument('--per-body', type=int, default=0, help='1: per-body deviations (n x 5) from the (frozen) shared parameters')
    a.add_argument('--smooth-weight', type=float, default=1e-2, help='per-body: w x mean over bonds |delta_i - delta_j|^2')
    a.add_argument('--lr-body', type=float, default=.02)
    a.add_argument('--multipliers', type=int, default=0, help='1: 7 independent log multipliers (kn ks kb kt lam mu damping) + friction + restitution instead of E, nu')
    a.add_argument('--train-frames', type=int, default=16)
    a.add_argument('--frames', type=int, default=23)
    a.add_argument('--iterations', type=int, default=40)
    a.add_argument('--lr', type=float, default=.1)
    a.add_argument('--lr-damping', type=float, default=None, help='learning rate of the damping multiplier (default: --lr)')
    a.add_argument('--residual', type=int, default=1)
    a.add_argument('--kinematics', default='exact', choices=('small', 'exact'))
    a.add_argument('--contact', default='smooth', choices=('hard', 'smooth'))
    a.add_argument('--layer-dx', type=float, default=.01)
    a.add_argument('--segment', type=int, default=100)
    a.add_argument('--tag', default='spacing8k')
    a.add_argument('--shell', type=int, default=0, help='1: BOND_BENDING.md model (bond-direction residual, enhanced strain, thin closure)')
    a.add_argument('--cosserat', type=int, default=0, help='1: COSSERAT_FOAM.md micropolar body law (implies --shell), no bond springs')
    a.add_argument('--dt-safety', type=float, default=None, help='stable_dt safety factor (default .5; .35 for --shell: the rest-state estimate has little margin at impact)')
    a.add_argument('--out', type=Path, required=True)
    args = a.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dev, f32 = 'cuda', torch.float32
    import warp as wp
    wp.init(); wp.set_mempool_release_threshold('cuda', 0)
    dem = Path(f'output/vid2sim_dem/{args.obj}'+('' if args.tag == 'spacing8k' else f'_{args.tag}'))
    gt_dir = args.gt or Path(f'data/vid2sim_gt/{args.obj}')
    gt = yaml.safe_load(open(SRC/args.obj/'gt_phys_params.yaml'))
    dt_frame, gravity, ground_p, ground_n = .05, [0., 0., -9.8], [0., 0., -.7], [0., 0., 1.]
    geometry = dict(np.load(dem/'geometry.npz'))
    if args.fix_nu:                                   # before build(): theta = 0 must be the (E, nu) reference
        args.nu = float(gt['prs'])
        print('nu fixed at the ground truth %.4f' % args.nu, flush=True)
    if args.cosserat:
        args.shell = 1
    model = build(geometry, args.E, args.nu, dev, 'noslip', .25, .3, f32, kinematics=args.kinematics, residual=bool(args.residual), shell=bool(args.shell), cosserat=bool(args.cosserat))
    safety = args.dt_safety if args.dt_safety is not None else (.4 if args.shell else .5)
    model_dt = model
    if args.shell:
        print('%s model: thin bodies %d, enhanced %d, residual %s, dt safety %.2f' % ('cosserat' if args.cosserat else 'shell', int(model.thin.sum()), int(model.eas_on.sum()), model.residual_mode, safety), flush=True)
        # omega_max of the stiffest configuration: the enhancement is switched off for bodies in the contact layer,
        # which raises their normal stiffness (EAS on: 2067 rad/s, off: 3588 on the sofa), so the time step is set
        # by the model without enhancement (float64 twin)
        model_dt = build(geometry, args.E, args.nu, dev, 'noslip', .25, .3, torch.float64, kinematics=args.kinematics, residual=bool(args.residual), shell=True, shell_eas=False, cosserat=bool(args.cosserat))
    model.set_environment(gravity=gravity, ground_point=ground_p, ground_normal=ground_n,
                          contact_radius=model.contact_radius.cpu().numpy(), ground_mode='noslip')
    x0, q0, v0, w0 = model.initial_state()
    v0 = (v0+torch.tensor(gravity, dtype=f32, device=dev)*dt_frame/2).contiguous()   # the GT's backward-Euler free fall
    if args.gt_targets is not None:
        tz = np.load(args.gt_targets)
        T = torch.tensor(tz['targets'], dtype=torch.float32, device=dev)
        targets = T-T[:1]+x0[None]                        # particle-mean motion applied to the body centres (frame-0 gap removed)
        tr = dict(seed='mpm E %.0f nu %.3f dx %.3f' % (float(tz['E']), float(tz['nu']), float(tz['dx'])))
        gt_dir = args.gt_targets
    else:
        targets, tr = lbs_targets(gt_dir, geometry['points'], dev)
    targets = targets[:args.frames+1]
    print('bodies %d, GT %s (E %.0f nu %.3f, seed %s), target frames %d' % (model.n, gt_dir, gt['yms'], gt['prs'], tr['seed'], len(targets)), flush=True)
    lam0, mu0 = args.E*args.nu/((1+args.nu)*(1-2*args.nu)), args.E/(2*(1+args.nu))
    logit = lambda p: float(np.log(p/(1-p)))
    params = torch.tensor([np.log(args.E), logit(args.nu/.49), 0., 0., 0.], dtype=f32, device=dev, requires_grad=True)

    def material(p):
        E = torch.exp(p[0]); nu = .49*torch.sigmoid(p[1])
        return E, nu, E*nu/((1+nu)*(1-2*nu)), E/(2*(1+nu))

    def theta_of(p):
        E, nu, lam, mu = material(p)
        lm, ll = torch.log(mu/mu0), torch.log(lam/lam0)
        return torch.stack((lm, lm, lm, lm, ll, lm, p[2], p[3], p[4])).expand(model.n, 9)

    def summary():
        E, nu, _, _ = material(params.detach())
        return dict(E=float(E), nu=float(nu), damping=float(torch.exp(params[2])), friction=float(torch.sigmoid(params[3])),
                    restitution=float(torch.sigmoid(params[4]+logit(.3))), E_over_gt=float(E)/gt['yms'])
    gt_params = torch.tensor([np.log(gt['yms']), logit(gt['prs']/.49), 0., 0., 0.], dtype=f32, device=dev)
    if args.multipliers:
        # theta columns directly: log multipliers of kn ks kb kt lam mu damping (0 = the build's E, nu split), friction, restitution
        params = torch.zeros(9, dtype=f32, device=dev, requires_grad=True)
        theta_of = lambda p: p.expand(model.n, 9)

        def summary():
            m = torch.exp(params.detach()[:7]).tolist()
            return dict(kn=m[0], ks=m[1], kb=m[2], kt=m[3], lam=m[4], mu=m[5], damping=m[6], friction=float(torch.sigmoid(params[7])),
                        restitution=float(torch.sigmoid(params[8]+logit(.3))), E_equiv_from_mu_over_gt=m[5]*args.E/gt['yms'],
                        bend_over_normal=m[2]/m[0], shear_over_normal=m[1]/m[0], lam_over_mu=m[4]/m[5])
        lam_gt, mu_gt = gt['yms']*gt['prs']/((1+gt['prs'])*(1-2*gt['prs'])), gt['yms']/(2*(1+gt['prs']))
        lm, ll = np.log(mu_gt/mu0), np.log(lam_gt/lam0)
        gt_params = torch.tensor([lm, lm, lm, lm, ll, lm, 0., 0., 0.], dtype=f32, device=dev)
    if args.init_from is not None:
        with torch.no_grad():
            params.copy_(torch.tensor(json.loads(args.init_from.read_text())['best_params'], dtype=f32, device=dev))
        print('continuing from', args.init_from, flush=True)
    var = params                                       # the optimised variable
    current_theta = lambda grad: theta_of(params if grad else params.detach())
    if args.per_body:
        params.requires_grad_(False)
        delta = torch.zeros(model.n, 5, dtype=f32, device=dev, requires_grad=True)
        var = delta

        def theta_pb(d):
            p = params[None]+d
            E = torch.exp(p[:, 0]); nu = .49*torch.sigmoid(p[:, 1])
            lam, mu = E*nu/((1+nu)*(1-2*nu)), E/(2*(1+nu))
            lm, ll = torch.log(mu/mu0), torch.log(lam/lam0)
            return torch.stack((lm, lm, lm, lm, ll, lm, p[:, 2], p[:, 3], p[:, 4]), 1)
        current_theta = lambda grad: theta_pb(delta if grad else delta.detach())
        nb_ = torch.bincount(torch.cat((model.i, model.j)), minlength=model.n)
        surf_ = nb_ < 9

        def summary():
            p = (params[None]+delta.detach())
            E = torch.exp(p[:, 0]); damp = torch.exp(p[:, 2]); fr = torch.sigmoid(p[:, 3])
            q = lambda t, u: float(torch.quantile(t, u))
            return dict(E_median=q(E, .5), E_p10=q(E, .1), E_p90=q(E, .9), E_median_over_gt=q(E, .5)/gt['yms'],
                        E_interior_median=float(E[~surf_].median()), E_surface_median=float(E[surf_].median()),
                        damping_median=q(damp, .5), damping_p10=q(damp, .1), damping_p90=q(damp, .9),
                        friction_median=q(fr, .5), delta_rms=float(delta.detach().pow(2).mean().sqrt()))
        print('per-body deviations from the shared parameters (frozen): %s, smooth weight %g, lr %g' % (
            json.dumps({k: round(v, 4) for k, v in dict(E=float(torch.exp(params[0])), nu=float(.49*torch.sigmoid(params[1])), damping=float(torch.exp(params[2]))).items()}),
            args.smooth_weight, args.lr_body), flush=True)
    if args.per_body:
        optimizer = torch.optim.Adam([delta], lr=args.lr_body)
    elif args.lr_damping is None:
        optimizer = torch.optim.Adam([params], lr=args.lr)
    else:
        # Adam with a separate rate for the damping entry: per-entry rates via a mask on a second copy of the step
        class SplitAdam:
            def __init__(self, p, lr, lr_damp, idx=2):
                self.p, self.idx = p, idx
                self.a = torch.optim.Adam([p], lr=lr); self.lr, self.lr_damp = lr, lr_damp
            def zero_grad(self): self.a.zero_grad()
            def step(self):
                scale = torch.ones_like(self.p); scale[self.idx] = self.lr_damp/self.lr
                g = self.p.grad
                # Adam's update is linear in lr: scale the damping component's effective lr by scaling its step
                before = self.p.detach().clone()
                self.a.step()
                with torch.no_grad():
                    self.p.copy_(before+(self.p-before)*scale)
        optimizer = SplitAdam(params, args.lr, args.lr_damping)
        print('damping learning rate %.3g (others %.3g)' % (args.lr_damping, args.lr), flush=True)
    wf = WarpFoam(model, contact=args.contact, layer_dx=args.layer_dx, implicit_damping=True, graphs=True)

    sub_scale = [1.]                                       # multiplier on the stable estimate, raised when a rollout blows up

    def rollout(n, substeps, grad=True):
        th = current_theta(grad)
        for _ in range(5):
            sub = int(np.ceil(substeps*sub_scale[0]))
            pos, quat = warp_rollout(wf, th, (x0, q0, v0, w0), dt_frame/sub, sub, n, args.segment if grad else None)
            if bool(torch.isfinite(pos).all()) or sub_scale[0] >= 4.:
                return pos, quat
            sub_scale[0] = min(4., sub_scale[0]*1.5)
            print('non-finite rollout at %d substeps: raising to x%.2f of the estimate' % (sub, sub_scale[0]), flush=True)
        return pos, quat

    def substeps_now():
        return int(np.ceil(dt_frame/model_dt.stable_dt(current_theta(False)[:, :7].double(), safety=safety, include_damping=False)))

    nb = torch.bincount(torch.cat((model.i, model.j)), minlength=model.n)
    surface = (nb < 9).cpu().numpy()
    height = geometry['points'][:, 2]-geometry['points'][:, 2].min()

    def evaluate(substeps):
        with torch.no_grad():
            pos, _ = rollout(args.frames, substeps, grad=False)
        pos = pos.to(targets.dtype)
        err = ((pos-targets)**2).sum(-1).sqrt()                                            # (frames+1, n)
        per_body = err[1:].mean(0).cpu().numpy()
        tr_, te = args.train_frames, args.frames
        return dict(nrmse_train=nrmse(pos, targets, 1, tr_), nrmse_future=nrmse(pos, targets, tr_+1, te), nrmse_all=nrmse(pos, targets, 1, te),
                    rms_mm_per_frame=[float(1000*err[f].pow(2).mean().sqrt()) for f in range(args.frames+1)],
                    body_err_mm=dict(surface=float(1000*per_body[surface].mean()), interior=float(1000*per_body[~surface].mean()),
                                     lowest_quarter=float(1000*per_body[height < np.quantile(height, .25)].mean()),
                                     top_quarter=float(1000*per_body[height > np.quantile(height, .75)].mean()), max=float(1000*per_body.max())),
                    substeps=substeps, **summary())

    def train_step(substeps):
        optimizer.zero_grad()
        pos, _ = rollout(args.train_frames, substeps)
        loss = ((pos[1:]-targets[1:args.train_frames+1])**2).sum(-1).mean()/((targets[1:args.train_frames+1]-targets[:1])**2).sum(-1).mean()
        total = loss
        if args.per_body:
            smooth = args.smooth_weight*((delta[model.i]-delta[model.j])**2).sum(-1).mean()
            total = loss+smooth
            train_step.last_smooth = float(smooth)
        total.backward()
        return float(loss)

    show = lambda r: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != 'rms_mm_per_frame'}
    substeps = substeps_now()
    report = dict(config={k: str(v) for k, v in vars(args).items()}, gt=gt, initial=evaluate(substeps))
    print('initial', json.dumps(show(report['initial'])), flush=True)
    history, best = [], (np.inf, params.detach().clone())
    start = time.perf_counter()
    for it in range(args.iterations):
        if it and it % 5 == 0:
            substeps = substeps_now()
        value = train_step(substeps)
        torch.cuda.empty_cache()
        if not np.isfinite(value) or not torch.isfinite(var.grad).all():
            print('non-finite at iteration', it, flush=True); break
        if args.fix_nu and not args.multipliers:
            if args.per_body:
                var.grad[:, 1] = 0.
            else:
                var.grad[1] = 0.
        if value < best[0]:
            best = (value, var.detach().clone())
        optimizer.step()
        history.append(dict(iteration=it, loss=value, substeps=int(np.ceil(substeps*sub_scale[0])), seconds=time.perf_counter()-start,
                            **({'smooth': train_step.last_smooth} if args.per_body else {}), **summary()))
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in history[-1].items()}), flush=True)
        (args.out/'history.json').write_text(json.dumps(history, indent=1))
    with torch.no_grad():
        var.copy_(best[1])
    report['best'] = evaluate(substeps_now())
    report['best_params'] = params.detach().cpu().tolist()
    if args.per_body:
        np.savez_compressed(args.out/'best_delta.npz', delta=delta.detach().cpu().numpy(), shared=params.detach().cpu().numpy())
        (args.out/'report.json').write_text(json.dumps(report, indent=1))
        print('best', json.dumps(show(report['best'])), flush=True)
        return
    # the released material (and default damping/contact) under the same supervision, for reference
    with torch.no_grad():
        params.copy_(gt_params)
    report['gt_material_default_rest'] = evaluate(substeps_now())
    report['seconds'] = time.perf_counter()-start
    (args.out/'report.json').write_text(json.dumps(report, indent=1))
    print('best', json.dumps(show(report['best'])), flush=True)
    print('GT material, default damping/contact', json.dumps(show(report['gt_material_default_rest'])), flush=True)


if __name__ == '__main__':
    main()
