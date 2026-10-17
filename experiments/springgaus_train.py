"""Spring-Gaus real capture: learn initial velocity, then DEM material, from masks of the dynamic videos.

Frames: physics in the dynamic world (metres, gravity +z 9.8, ground z = ground_z with normal -z,
120 fps, frame 0 = registered static foam); rendering in the foam frame of the static
reconstruction. x_dyn = A x_foam + b with A = k R0 (pacnerf_foam_geometry.foam_to_dynamic), so
body centres map back by A^-1 (x - b) and body rotations / deformation gradients by R0^T (.) R0.
The 3 dynamic cameras (undistorted pinhole) are re-expressed in the foam frame (cam_from_foam
= [Rc R0 | (Rc b + tc)/k]) and rendered at 1/--downsample resolution.

Stage 'velocity' (Spring-Gaus' velocity stage): before the hit the object is rigid, so the foam
moves as x(t) = c + v0 t + g t^2/2, R(t) = exp(t [w0]) about the centre of mass; v0, w0 fit the
masks of frames 0..hit-1 in the training views (silhouette L2). Saved to velocity.json.
--registration (default: the joint pre-hit fit, springgaus_initial_state --upright --no-spin) also
supplies v0, w0 and the per-camera sub-frame time offsets; each view is rendered at frame + offset.
Stage 'material': DEM rollout (Warp) from the rest geometry with v = v0 + w0 x (x - c), shared
7 log multipliers (+ friction and restitution logits with --contact smooth; optionally per body) fit to masks of frames 1..--train-frames in the training
views. Evaluation: mask IoU per split (train/held-out view x train/future frames).
Spring-Gaus split: --eval-cam 0 (C0733) held out, the other 2 cameras train.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from experiments.foam_dynamic_render import air_shell, keep_cells, load, render
from experiments.foam_deform_render import DeformedFoam
from experiments.pacnerf_dem_forward import build
from experiments.pacnerf_foam_geometry import foam_to_dynamic
from experiments.pacnerf_render_dem import body_rotation
from experiments.foam_dem_warp import WarpFoam, rollout as warp_rollout
from experiments.springgaus_dynamic import CAMS
from powerfoam.camera import TorchCamera

OUT = Path('data/springgaus_powerfoam')


def phys_obs_np(series, k_imp):
    """numpy / torch-free-of-graph version of the training phys_obs (feature series (frames, 6), impact index)."""
    import torch as _t
    a_ = _t.tensor(np.asarray(series)[:, 5], dtype=_t.float64)
    t = _t.arange(len(a_)-k_imp-1, dtype=_t.float64); y = a_[k_imp+1:]
    Tg = _t.arange(3., 10.01, .05, dtype=_t.float64); kg = _t.arange(0., .61, .04, dtype=_t.float64)
    TT, KK = _t.meshgrid(Tg, kg, indexing='ij'); TT, KK = TT.reshape(-1), KK.reshape(-1)
    env = _t.exp(-KK[:, None]*t[None])
    B = _t.stack((_t.ones_like(env), env*_t.cos(2*np.pi*t[None]/TT[:, None]), env*_t.sin(2*np.pi*t[None]/TT[:, None])), -1)
    coef = _t.linalg.solve(B.transpose(1, 2)@B+1e-6*_t.eye(3, dtype=_t.float64), B.transpose(1, 2)@y[None, :, None])
    res = ((B@coef)[..., 0]-y[None]).pow(2).sum(-1)
    w = _t.softmax(-res/(.05*res.min()+1e-8), 0)
    period = float((w*TT).sum()); decay = max(float((w*KK).sum()), 1e-3)
    amp = max(float(1-(-.01*_t.logsumexp(-a_/.01, 0))), 1e-3)
    return np.array([period, decay, amp, (period/amp)**2])


def foam_cameras(meta, A, b, k, downsample):
    """TorchCameras of the dynamic views in foam coordinates (OpenCV axes, as data_loader/colmap.py)."""
    R0 = A/k
    W, H = meta['width']//downsample, meta['height']//downsample
    out = []
    for c in CAMS:
        cam = meta['cameras'][c]
        K = np.array(cam['K'], float)
        fx, fy = K[0, 0]/downsample, K[1, 1]/downsample
        cx, cy = (K[0, 2]+.5)/downsample-.5, (K[1, 2]+.5)/downsample-.5        # OpenCV pixel-centre convention
        # Power Foam's rasterizer projects with a symmetric frustum (right/up half-extents): the
        # principal point must be the image centre or renders shift against the ray maps.
        assert abs(cx-(W-1)/2) < .51 and abs(cy-(H-1)/2) < .51, ('principal point not centred', cx, cy, W, H)
        Rc, tc = np.array(cam['R']), np.array(cam['t'])
        Rf = Rc@R0; tf = (Rc@b+tc)/k
        c2w_R = Rf.T; eye = -Rf.T@tf
        x, y = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
        d = np.stack(((x-cx)/fx, (y-cy)/fy, np.ones_like(x)), -1).reshape(-1, 3)
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        dw = d@c2w_R.T
        rays = np.concatenate((np.broadcast_to(eye, dw.shape), dw), -1).reshape(H, W, 6)
        right = c2w_R@np.array([(W-1)/2/fx, 0, 0]); up = c2w_R@np.array([0, -(H-1)/2/fy, 0])
        f32 = lambda a: torch.tensor(np.ascontiguousarray(a), dtype=torch.float32)
        out.append(TorchCamera(eye=f32(eye), right=f32(right), up=f32(up), width=W, height=H, ray_maps=f32(rays)))
    return out


def skew(w):
    z = torch.zeros((), dtype=w.dtype, device=w.device)
    return torch.stack((torch.stack((z, -w[2], w[1])), torch.stack((w[2], z, -w[0])), torch.stack((-w[1], w[0], z))))


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('stage', choices=('velocity', 'material', 'scan', 'render', 'gradcheck', 'gradcheck2', 'hessian', 'sweep', 'debris', 'framediag', 'bottom', 'carve', 'timeshift', 'frameview', 'features', 'jitter', 'energy', 'featbatch', 'cfieldcheck', 'gradall', 'rendercheck', 'rendercheck2', 'rendercheck3', 'rendercheck4', 'rasterref', 'finesweep', 'lossdecomp', 'sag', 'flightfit', 'kprofile', 'simcheck', 'dampcheck'))
    a.add_argument('--obj', default='burger')
    a.add_argument('--run', default=None, help='static foam run (default output/springgaus_<obj>_v2)')
    a.add_argument('--dem', type=Path, default=None, help='default output/springgaus_dem/<obj>')
    a.add_argument('--eval-cam', type=int, default=0)
    a.add_argument('--downsample', type=int, default=2)
    a.add_argument('--train-frames', type=int, default=14, help='material: frames 1..N train, the rest are future')
    a.add_argument('--ogden-learn', type=int, default=0, help='learn the Ogden exponent (extra theta column, log(alpha / ogden_alpha))')
    a.add_argument('--learn-init', type=int, default=0, help='learn corrections to the initial velocity and spin')
    a.add_argument('--freeze-spin', type=int, default=0, help='--learn-init: keep the spin correction dw0 at its start value '
                   '(only the initial-velocity correction dv0 is learned)')
    a.add_argument('--learn-pose', type=int, default=0, help='learn a rigid correction of the frame-0 pose (rotation about the '
                   'rotation centre + shift), applied to positions, orientations and velocities')
    a.add_argument('--lr-rot', type=float, default=3e-3, help='learning rate of the pose rotation correction (rad)')
    a.add_argument('--lr-shift', type=float, default=.05, help='learning rate of the pose shift correction (mm)')
    a.add_argument('--learn-offsets', type=int, default=0, help='learn the sub-frame time offsets of cameras 1.. (camera 0 fixed)')
    a.add_argument('--lr-offs', type=float, default=.01, help='learning rate of the camera time offsets (frames)')
    a.add_argument('--prior-vz', type=float, default=.02, help='prior sigma of the vertical initial-velocity correction (m/s; 0: none)')
    a.add_argument('--prior-spin', type=float, default=.2, help='prior sigma of the spin correction (rad/s; 0: none)')
    a.add_argument('--lr-init', type=float, default=.02, help='learning rate of the initial velocity / spin corrections')
    a.add_argument('--feat-weight', type=float, default=1., help="loss 'feat': weight of the normalised moment features")
    a.add_argument('--iou-weight', type=float, default=1., help="loss 'feat': weight of 1 - soft IoU")
    a.add_argument('--prior-weight', type=float, default=0., help='weak Gaussian priors on the weakly identifiable parameters: '
                   'log friction multiplier (sigma 0.3), vertical initial-velocity correction (0.02 m/s), spin correction (0.2 rad/s)')
    a.add_argument('--contact-field', type=int, default=0, help='hertz: learnable contact-surface offset field over the bottom, '
                   'offset(u, v) = c0 + c1 u + c2 v + c3 u^2 + c4 v^2 + c5 uv (mm; u, v body-frame horizontal coordinates / object radius)')
    a.add_argument('--flight-init', type=int, default=0, help='initial-state corrections learned from the free flight only: the '
                   'main loss treats them as constants; a second, short pass weights each frame by 1 before the first ground contact '
                   'and exp(-(f - f_contact + 1) / tau) after it, and updates only the corrections')
    a.add_argument('--flight-tau', type=float, default=1., help='frames; decay of the flight-pass weights after the first contact')
    a.add_argument('--pos-boost', type=float, default=0., help="loss 'feat': per-frame multiplier of the centroid (cx, cy) "
                   "terms: 1 + A before the first ground contact, 1 + A exp(-(f - f_contact + 1) / --pos-tau) after it (0: off); "
                   "the free flight pins the initial velocity, post-contact offsets are left to the contact field")
    a.add_argument('--init-decay', type=float, default=0., help='frames tau (0: off): the initial-state corrections (position, '
                   'pose, velocity, spin) and the contact-field c0 take their gradient with each frame f weighted by exp(-f / tau) '
                   '(a second, discounted adjoint through the same rollout); everything else keeps the plain gradient')
    a.add_argument('--init-cut', type=int, default=0, help='1: the initial position / velocity corrections (dx0, dv0) take '
                   'their gradient only from frames before the first contact, min(simulated, video) (a second adjoint with frame '
                   'weights 1 before / 0 from that frame); pose, spin, contact field and material keep the plain gradient. '
                   '2: also the pose and spin corrections (rot0, dw0) and the camera time offsets (rendered frames f with '
                   'f + 1 < cut) - everything about the body before the impact comes from the free flight. 3: as 2 but the '
                   'spin keeps the plain gradient (a spin rotates the body by only ~w t over the short flight)')
    a.add_argument('--tilt-relative', type=int, default=0, help='1: learn the tilt of the contacting bottom relative to the '
                   'ground at the impact (dtilt, mm, in contact-field c1 c2 units) instead of c1 c2; c1 c2 = dtilt - tilt(rot0 + dw0 t_hit) '
                   '(small-angle equivalent of the body pose and spin corrections at the metadata hit frame, rot0 held constant '
                   'there), so the spin no longer changes the impact tilt and only carries the angular momentum')
    a.add_argument('--window-start', type=int, default=0, help='training-window curriculum (0: off): train on frames 1..n with '
                   'n = window-start at iteration 0, one more frame every --window-step iterations, up to --train-frames; the '
                   'physical-observable / slope terms and the best-iterate choice wait for the full window')
    a.add_argument('--window-step', type=int, default=1, help='--window-start: iterations per added frame')
    a.add_argument('--pos-tau', type=float, default=1., help='frames; decay of the --pos-boost centroid weight after the first contact')
    a.add_argument('--cfield-fix0', type=int, default=0, help='keep the uniform contact-field term c0 fixed (it only shifts the '
                   'contact surface along the normal, i.e. the ground height)')
    a.add_argument('--thornton', type=int, default=1, help='hertz: rescale the tangential force by sqrt(d / d_prev) on unloading (0: off; diagnostics)')
    a.add_argument('--check-loss', choices=('pos', 'sil', 'mom'), default='sil', help='gradall: loss on the body positions (random fixed '
                   'weights, no rendering) or the silhouette MSE against the masks')
    a.add_argument('--check-frames', type=int, default=0, help='gradall: frames used (0: --train-frames)')
    a.add_argument('--check-scale', type=float, default=1., help='gradall: multiplies every finite-difference step')
    a.add_argument('--rest-field', type=Path, default=None, help='sag_field.npz (stage sag): self-weight sag of the scanned shape')
    a.add_argument('--rest-scale', type=float, default=0., help='remove this multiple of the sag field from the scanned rest shape')
    a.add_argument('--learn-rest', choices=('off', 'free', 'coupled'), default='off', help='unloaded rest shape X(s) = X_scan - s '
                   'u_rel (--rest-field), s learned (free) or tied to the shear modulus s = G_ref / G (coupled); the rest geometry '
                   'is interpolated linearly between models built at s = 0 and s = 1 (same bonds and tetrahedra)')
    a.add_argument('--rest-check', type=int, default=0, help='--learn-rest: compare the interpolated rest geometry with exact builds')
    a.add_argument('--rest-init', type=float, default=1., help='--learn-rest free: starting s')
    a.add_argument('--lr-rest', type=float, default=.05, help='--learn-rest free: learning rate of s')
    a.add_argument('--rest-uniform', type=float, default=1., help='uniform scale of the rest shape about its bottom centre')
    a.add_argument('--flight-frames', type=int, default=None, help='flightfit: use frames 0..n-1 (default: automatic detection)')
    a.add_argument('--flight-synthetic', type=Path, default=None, help='flightfit: json of known values; targets rendered from them')
    a.add_argument('--sag-frames', type=int, default=60, help='sag: frames simulated for the body to settle on the ground')
    a.add_argument('--sag-damping', type=float, default=10., help='sag: damping multiplier while settling')
    a.add_argument('--supersample', type=int, default=2, help='render silhouettes at this many times the resolution and '
                   'average-pool (anti-aliasing of the gradient; must divide --downsample)')
    a.add_argument('--raster-threshold', type=float, default=1e-3, help='rasterizer early-stop transmittance (Power Foam default 1e-3)')
    a.add_argument('--ref-band', choices=('edge', 'inner', 'outer', 'all'), default='edge', help='rasterref: which pixels to sample')
    a.add_argument('--raster-nocoop', type=int, default=0, help='rendercheck4: rasterizer without cooperative (tile-shared) primitive / adjacency loads')
    a.add_argument('--check-only', nargs='*', default=None, help='gradall: only these scalars (e.g. logE alpha c1)')
    a.add_argument('--check-decay', type=float, default=None, help='gradall: check --init-decay: initial-state and c0 gradients of '
                   'the discounted adjoint (factor r per frame) against the plain gradient of the loss with frame f weighted by r^f')
    a.add_argument('--check-cut', type=int, default=None, help='gradall: check --init-cut with the cut at this frame: dv0 / dx0 '
                   'from the second adjoint against the plain gradient of the loss with frames >= cut weighted 0')
    a.add_argument('--check-scan', type=float, nargs='*', default=None, help='gradall: print the loss at these offsets of each '
                   '--check-only scalar (1-D scan; shows whether the loss is smooth on the finite-difference scale)')
    a.add_argument('--check-single', type=int, default=0, help='gradall: one finite-difference step only (no h/4)')
    a.add_argument('--check-multi', type=float, nargs='*', default=None, help='gradall: step multipliers of the base step (overrides h, h/4)')
    a.add_argument('--check-cfield', type=float, nargs='+', default=None, help='cfieldcheck: contact-field coefficients (mm)')
    a.add_argument('--cfield-terms', type=int, default=6, help='number of contact-field coefficients learned (3: height + tilt only)')
    a.add_argument('--lr-cfield', type=float, default=.1, help='learning rate of the contact-field coefficients (mm)')
    a.add_argument('--pos-scale', type=float, default=0., help="loss 'feat': normalise the centroid (cx, cy) by this many px "
                   "instead of its std over the frames (0: std)")
    a.add_argument('--profile', type=int, default=0, help='print per-iteration timings (rollout, render + loss, backward; render-only backward at iteration 1)')
    a.add_argument('--slope-weight', type=float, default=0., help="loss 'feat': weight of the horizontal centroid velocities "
                   "before and after the impact (image direction perpendicular to the projected gravity; least-squares slopes over "
                   "frames 1..impact-2 and impact+2..N, per training camera; impact = strongest deceleration along gravity in the real video), in units of their noise: the pre-impact one pins the "
                   "initial velocity, the change across the impact is the contact impulse (contact-surface tilt)")
    a.add_argument('--slope-floor', type=float, default=.05, help='px / frame; lower bound of the noise scale of --slope-weight')
    a.add_argument('--phys-weight', type=float, default=0., help="loss 'feat': weight of the physical observables (log oscillation "
                   "period and decay, log squash amplitude, log impact / oscillation stiffness ratio)")
    a.add_argument('--batch-json', type=Path, default=None, help="featbatch: list of {name, parameters, dv0, dw0}")
    a.add_argument('--save-masks', type=int, default=0, help='featbatch: also save every simulated silhouette (uint8) per run')
    a.add_argument('--synthetic', type=Path, default=None, help='json with best.parameters (and optional dv0, dw0): replace the '
                   'real masks by silhouettes simulated with these parameters (identifiability / recovery test)')
    a.add_argument('--loss', default='mse', choices=('mse', 'iou', 'feat'),
                   help='material: silhouette loss per view, mse (alpha vs mask) or iou (1 - soft IoU of alpha and mask)')
    a.add_argument('--fix-nu', action='store_true', help='material, --param Enu: keep nu at its --init-from value')
    a.add_argument('--fix-E', action='store_true', help='material, --param Enu: keep E at its --init-from value')
    a.add_argument('--E', type=float, default=2e4, help='initial Young modulus (Pa at 1000 kg/m^3; only E/rho matters)')
    a.add_argument('--nu', type=float, default=.3)
    a.add_argument('--iterations', type=int, default=40)
    a.add_argument('--lr', type=float, default=.05)
    a.add_argument('--lr-final', type=float, default=1., help='cosine learning-rate decay over the iterations down to this '
                   'fraction of every learning rate (1: constant)')
    a.add_argument('--adam-beta2', type=float, default=.999, help='Adam second-moment decay (0.99: ~100-step memory, so the step '
                   'size follows the current gradient scale instead of the large early gradients; torch default 0.999)')
    a.add_argument('--render-mode', default='blend', choices=('rigid', 'blend', 'tet'),
                   help="tet: appearance cells embedded in the bodies' tetrahedra (barycentric; no fitted F)")
    a.add_argument('--per-body', action='store_true')
    a.add_argument('--smooth-weight', type=float, default=1e-2)
    a.add_argument('--init-from', type=Path, help="material report.json whose 'best' parameters start the fit")
    a.add_argument('--registration', default='initial_state_upright_nospin.json',
                   help='register_calib/<file>: pose (+ v0, w0, rotation centre, camera time offsets from springgaus_initial_state)')
    a.add_argument('--contact', default='hard', choices=('hard', 'smooth', 'coulomb', 'hertz'),
                   help="ground: hard no-slip switch, smooth no-slip layer, or 'coulomb' (no penetration + Coulomb friction, "
                        "mu_f = friction0 exp(theta7), no restitution parameter)")
    a.add_argument('--friction0', type=float, default=.5, help='smooth contact: friction at theta = 0; coulomb: Coulomb coefficient at theta = 0')
    a.add_argument('--restitution0', type=float, default=.3, help='smooth contact: restitution at theta = 0')
    a.add_argument('--layer-dx', type=float, default=.002, help='smooth contact: B-spline layer width (m)')
    a.add_argument('--config-ground', action='store_true', help="use the Spring-Gaus config ground_z instead of the measured table")
    a.add_argument('--ground-offset', type=float, default=0., help='m, added to ground_z (+ = lower, gravity is +z)')
    a.add_argument('--segment', type=int, default=100, help='checkpoint the DEM rollout every N substeps (0: keep the whole tape)')
    a.add_argument('--scan', type=float, nargs='+', default=[0, -1.1, -2.3, -3.4, -4.6, -5.7],
                   help="scan: log offsets added to the 6 stiffness multipliers (forward only)")
    a.add_argument('--viz', action='store_true', help='scan: save sim (green) vs mask (red) overlays')
    a.add_argument('--theta-npz', type=Path, help='render: per-body theta (theta_best.npz) instead of --init-from')
    a.add_argument('--cosserat', type=int, default=0, help='1: COSSERAT_FOAM.md body law (+ shell closures), no bond springs')
    a.add_argument('--truevol', type=int, default=0, help='1: SPRING_DILATATION.md section 2 - springs carry all of mu (their four '
                   'stiffnesses follow the mu column), true-volume tetrahedra term carries lam')
    a.add_argument('--implicit-damping', type=int, default=1)
    a.add_argument('--graphs', type=int, default=1)
    a.add_argument('--check-params', type=float, nargs='+', help='gradcheck: shared parameters at which to recompute the gradient')
    a.add_argument('--check-substeps', type=int, default=0, help='gradcheck: base substeps (0: the stability estimate)')
    a.add_argument('--skip-toggles', action='store_true', help='gradcheck2: skip the EAS / thin on-off gradients')
    a.add_argument('--hess-times', type=float, nargs='+', default=[7.0, 8.0, 8.1, 8.2, 8.5, 9.0, 10.0])
    a.add_argument('--param', default='lammu', choices=('lammu', 'Enu'),
                   help="material columns 4, 5: log multipliers of lam, mu ('lammu'), or 'Enu': column 4 = log multiplier of E, "
                        "column 5 = offset of z with nu = 0.5 sigmoid(z) (z = logit(2 nu0) at 0), mapped to the lam / mu multipliers")
    a.add_argument('--sweep-E', type=float, nargs='+', default=[.5, .71, 1., 1.41, 2.], help='sweep: E factors on the learned E')
    a.add_argument('--sweep-nu', type=float, nargs='+', default=[.3, .4, .45, .47, .476, .48, .49])
    a.add_argument('--diag-from', type=int, default=6)
    a.add_argument('--carve-step', type=float, default=5e-4)
    a.add_argument('--substep-scale', type=float, default=1., help='multiply the stable-step substep count')
    a.add_argument('--contact-scale', type=float, default=1., help='hertz contact: stiffness multiplier')
    a.add_argument('--nonlinear-bonds', type=int, default=0, help='Ogden normal bond springs (truevol)')
    a.add_argument('--ogden-alpha', type=float, default=2., help='Ogden exponent of the nonlinear bonds (2 = neo-Hookean)')
    a.add_argument('--visco', type=int, default=0, help='Zener (Maxwell branch) on every bond spring; theta gets 2 extra columns (log g, log tau)')
    a.add_argument('--visco-g0', type=float, default=1., help='Zener branch stiffness ratio at theta = 0')
    a.add_argument('--visco-tau0', type=float, default=1/120, help='Zener relaxation time (s) at theta = 0')
    a.add_argument('--view-columns', default='r7,s7,r8,s8,s8.875,r9,s9',
                   help='frameview: comma list, r<frame> real image, s<time> simulation (fractional frames allowed)')
    a.add_argument('--diag-to', type=int, default=12)
    a.add_argument('--out', type=Path, required=True)
    args = a.parse_args()
    args.run = args.run or f'output/springgaus_{args.obj}_v2'
    args.dem = args.dem or Path(f'output/springgaus_dem/{args.obj}')
    args.out.mkdir(parents=True, exist_ok=True)
    dyn = OUT/f'{args.obj}_dynamic'
    meta = json.loads((dyn/'cameras.json').read_text())
    reg = json.loads((dyn/'register_calib'/args.registration).read_text())
    A, b, k = foam_to_dynamic(OUT/args.obj, dyn/'register_calib'/args.registration)
    offsets = [reg.get('time_offsets_frames', {}).get(c, 0.) for c in CAMS]
    R0 = A/k
    Ainv = np.linalg.inv(A)
    geometry = dict(np.load(args.dem/'geometry.npz')); cells = dict(np.load(args.dem/'cells.npz'))
    foam, _ = load(args.run)
    keep, owner = air_shell(foam, cells['source_index'], geometry['body_of_cell'], rings=2)
    keep_cells(foam, keep)
    for name in ('radii', 'density', 'texel_sites', 'texel_sv_axis', 'texel_sv_rgb', 'texel_height'):
        getattr(foam, name).requires_grad_(False)
    body_rest_f = (geometry['points']-b)@Ainv.T
    deformed = DeformedFoam(foam, owner, body_rest_f, np.zeros(3))
    if args.render_mode == 'blend':
        deformed.prepare_blend(body_rest_f, K=24, width=2.)
    cams = foam_cameras(meta, A, b, k, args.downsample)
    W, H = cams[0].width, cams[0].height
    masks = {}
    for f in range(meta['frames']):
        for c, name in enumerate(CAMS):
            m = Image.open(dyn/'masks'/name/f'{f:02d}.png').convert('L').resize((W, H), Image.BILINEAR)
            masks[f, c] = torch.tensor(np.asarray(m, np.float32)/255, device='cuda')
    train_cams = [c for c in range(len(CAMS)) if c != args.eval_cam]
    dev, f32 = 'cuda', torch.float32
    Ainv_t = torch.tensor(Ainv, dtype=f32, device=dev); b_t = torch.tensor(b, dtype=f32, device=dev)
    R0_t = torch.tensor(R0, dtype=f32, device=dev)
    to_foam = lambda x: (x-b_t)@Ainv_t.T
    conj = lambda M: R0_t.T@M@R0_t
    fps = meta['fps']; hit = meta['hit_frame']
    g = torch.tensor(meta['gravity'], dtype=f32, device=dev)
    X = torch.tensor(geometry['points'], dtype=f32, device=dev)
    mass = torch.tensor(geometry['masses'], dtype=f32, device=dev)
    com = (mass[:, None]*X).sum(0)/mass.sum()

    foam.rasterizer.transmittance_threshold = args.raster_threshold

    ss_ = int(args.supersample)
    if ss_ > 1:      # anti-aliasing: render at ss x the resolution and average-pool back (edges are ~1 px sharp, so a
        # single sample per pixel makes the local gradient wobble around the macroscopic slope)
        assert args.downsample % ss_ == 0, 'supersample must divide downsample'
        cams_ss = foam_cameras(meta, A, b, k, args.downsample//ss_)
        assert cams_ss[0].width == ss_*W and cams_ss[0].height == ss_*H

    def silhouette(c):
        if ss_ > 1:
            _, alpha = render(foam, cams_ss[c])
            return torch.nn.functional.avg_pool2d(alpha.reshape(1, 1, ss_*H, ss_*W), ss_)[0, 0]
        _, alpha = render(foam, cams[c])
        return alpha.reshape(H, W)

    def moments_t(a_):
        """Differentiable silhouette features (px): centroid cx, cy, second-moment widths sx, sy, sqrt(area)."""
        H_, W_ = a_.shape
        ys = torch.arange(H_, device=a_.device, dtype=a_.dtype)[:, None]; xs = torch.arange(W_, device=a_.device, dtype=a_.dtype)[None, :]
        m0 = a_.sum().clamp(min=1e-6)
        cx = (a_*xs).sum()/m0; cy = (a_*ys).sum()/m0
        sx = ((a_*(xs-cx)**2).sum()/m0).clamp(min=1e-6).sqrt(); sy = ((a_*(ys-cy)**2).sum()/m0).clamp(min=1e-6).sqrt()
        return torch.stack((cx, cy, sx, sy, m0.sqrt()))

    def iou(alpha, m):
        p, q = alpha > .5, m > .5
        return float((p & q).sum()/(p | q).sum())

    if args.stage == 'velocity':
        v0 = torch.tensor([0., 0., 3.], device=dev, requires_grad=True)     # Spring-Gaus INIT_VELOCITY
        w0 = torch.zeros(3, device=dev, requires_grad=True)
        opt = torch.optim.Adam([dict(params=[v0], lr=.05), dict(params=[w0], lr=.5)])
        eye = torch.eye(3, device=dev)

        def place(f):
            t = f/fps
            R = torch.matrix_exp(t*skew(w0))
            x = com+(X-com)@R.T+t*v0+.5*t*t*g
            deformed.apply(to_foam(x), body_R=conj(R).expand(len(X), 3, 3), body_F=conj(R).expand(len(X), 3, 3),
                           mode='rigid' if args.render_mode == 'rigid' else 'blend')
        history = []
        for it in range(args.iterations):
            opt.zero_grad()
            loss = 0.
            for f in range(1, hit):
                place(f)
                for c in train_cams:
                    loss = loss+((silhouette(c)-masks[f, c])**2).mean()
            loss = loss/((hit-1)*len(train_cams))
            loss.backward(); opt.step()
            history.append(dict(iteration=it, loss=float(loss), v0=v0.tolist(), w0=w0.tolist()))
            print(json.dumps(history[-1]), flush=True)
        with torch.no_grad():
            ious = np.zeros((hit, len(CAMS)))
            for f in range(hit):
                place(f)
                for c in range(len(CAMS)):
                    ious[f, c] = iou(silhouette(c), masks[f, c])
        res = dict(v0=v0.tolist(), w0=w0.tolist(), com=com.tolist(), frames=f'0..{hit-1}', train_cams=train_cams,
                   iou_per_frame_cam=ious.tolist(), iou_train=float(ious[1:, train_cams].mean()),
                   iou_eval=float(ious[1:, args.eval_cam].mean()), iou_frame0=ious[0].tolist(), history=history)
        (args.out/'velocity.json').write_text(json.dumps(res, indent=1))
        print('velocity', json.dumps({k_: res[k_] for k_ in ('v0', 'w0', 'iou_train', 'iou_eval', 'iou_frame0')}))
        return

    if 'v0' in reg:       # joint initial-state fit
        vel = dict(v0=reg['v0'], w0=reg['w0'], com=reg['rotation_centre_world'])
    else:
        vel = json.loads((args.out.parent/'velocity'/'velocity.json').read_text())
    rim = dyn/'rest'/'ground_rim.json'          # measured table plane (springgaus_ground rim), else the config value
    ground_z = json.loads(rim.read_text())['plane_z_at_centre'] if rim.exists() and not args.config_ground else meta['ground_z']
    print('ground z %.4f (%s)' % (ground_z, 'measured table rim' if rim.exists() and not args.config_ground else 'Spring-Gaus config'), flush=True)
    kw_model = dict(kinematics='exact', residual=True, shell=True, cosserat=True) if args.cosserat else dict(nodalvol=True) if args.truevol else {}
    if args.nonlinear_bonds:
        kw_model['nonlinear'] = True
        kw_model['ogden_alpha'] = args.ogden_alpha
    if args.rest_field is not None and args.rest_scale != 0:
        # unloaded rest shape: the scanned part of the geometry was captured resting on a support, compressed by its
        # own weight; remove s x the self-weight sag field (stage 'sag'), taken relative to the supported bottom layer
        # (the bottom itself, filled from the free-flight visual hull, is already unloaded). The appearance cells keep
        # their embedding from the scanned positions, so they follow the corrected bodies.
        fz = np.load(args.rest_field)
        Rm_ = fz['R_upright']; uu_ = fz['u']@Rm_.T                     # upright frame
        gd_ = np.asarray(meta['gravity'], float); gd_ = gd_/np.linalg.norm(gd_); up_ = -gd_
        P_ = geometry['points']; m_ = geometry['masses']; cm_ = (m_[:, None]*P_).sum(0)/m_.sum()
        h_ = ((P_-cm_)@Rm_.T)@up_
        bot = h_ < h_.min()+4e-3
        uu_ = uu_-float((uu_[bot]@up_).mean())*up_[None, :]
        u_rel = uu_@Rm_                                                 # scan frame, bottom layer ~ 0
        geometry = dict(geometry)
        geometry['points'] = P_-args.rest_scale*u_rel
        if 'contacts' in geometry and 'pairs' in geometry:
            pr_ = geometry['pairs']
            geometry['contacts'] = geometry['contacts']-args.rest_scale*.5*(u_rel[pr_[:, 0]]+u_rel[pr_[:, 1]])
        print('rest shape: scanned geometry minus %.2f x the self-weight sag (top moves %+.2f mm, bottom %+.2f mm along up)' % (
            args.rest_scale, -args.rest_scale*float((uu_[h_ > h_.max()-4e-3]@up_).mean())*1e3, -args.rest_scale*float((uu_[bot]@up_).mean())*1e3), flush=True)
    if args.rest_uniform != 1.:
        # uniform scale of the rest shape about the centre of its bottom layer (registration scale error); volumes,
        # areas and radii follow
        geometry = dict(geometry)
        gd_ = np.asarray(meta['gravity'], float); gd_ = gd_/np.linalg.norm(gd_)
        P_ = geometry['points']; dn_ = P_@gd_
        o_ = P_[dn_ > dn_.max()-4e-3].mean(0)
        k_ = args.rest_uniform
        geometry['points'] = o_+(P_-o_)*k_
        if 'contacts' in geometry:
            geometry['contacts'] = o_+(geometry['contacts']-o_)*k_
        geometry['masses'] = geometry['masses']*k_**3
        if 'areas' in geometry:
            geometry['areas'] = geometry['areas']*k_**2
        if 'radii' in geometry:
            geometry['radii'] = geometry['radii']*k_
        print('rest shape: uniform scale %.4f about the bottom centre' % k_, flush=True)
    model = build(geometry, args.E, args.nu, dev, 'noslip', .25, .3, f32, **kw_model)
    if args.render_mode == 'tet':     # embed the appearance cells in the bodies' tetrahedra (the true-volume mesh)
        if getattr(model, 'tets', None) is None:
            model._setup_tets()
        print('tet embedding', deformed.prepare_tet(body_rest_f, model.tets.cpu().numpy()), flush=True)
    rest_A = rest_B = None
    if args.learn_rest != 'off':
        # rest geometry at s = 0 (this model) and s = 1 (rest shifted by -u_rel), same bonds and tetrahedra
        assert args.rest_scale == 0, '--learn-rest replaces --rest-scale'
        fz_ = np.load(args.rest_field)
        Rr_ = fz_['R_upright']; uu_ = fz_['u']@Rr_.T
        gd_ = np.asarray(meta['gravity'], float); gd_ = gd_/np.linalg.norm(gd_); up_ = -gd_
        P_ = geometry['points']; m_ = geometry['masses']; cm_ = (m_[:, None]*P_).sum(0)/m_.sum()
        h_ = ((P_-cm_)@Rr_.T)@up_; bot_ = h_ < h_.min()+4e-3
        uu_ = uu_-float((uu_[bot_]@up_).mean())*up_[None, :]
        u_rel_np = uu_@Rr_
        gB = dict(geometry); gB['points'] = P_-u_rel_np
        if 'contacts' in gB and 'pairs' in gB:
            gB['contacts'] = gB['contacts']-.5*(u_rel_np[gB['pairs'][:, 0]]+u_rel_np[gB['pairs'][:, 1]])
        kwB = dict(kw_model)
        nodal_ = kwB.get('nodalvol', False)

        def same_tets(mB_):            # tetrahedra of the s = 0 model (the Delaunay topology may flip), volumes at mB_'s rest
            if nodal_:
                mB_.tets = model.tets
                PB = mB_.X[model.tets]
                VB = ((torch.linalg.cross(PB[:, 1]-PB[:, 0], PB[:, 2]-PB[:, 0])*(PB[:, 3]-PB[:, 0])).sum(-1)/6).abs()
                mB_.tet_V0 = VB
                mB_.node_V0 = torch.zeros(mB_.n, dtype=VB.dtype, device=VB.device).index_add(0, model.tets.reshape(-1), VB.repeat_interleave(4))
                mB_.node_has = model.node_has
            return mB_
        modelB = same_tets(build(gB, args.E, args.nu, dev, 'noslip', .25, .3, f32, **kwB))
        nb_ = len(model.i)

        def rest_arrays(m__):
            out_ = [m__.n0, m__.ri, m__.rj, m__.base, m__.c_n, m__.c_t, m__.c_r, m__.ls_w[:nb_], m__.ls_dX[:nb_], m__.moment_inv,
                    m__.node_V0 if getattr(m__, 'node_V0', None) is not None else torch.zeros(m__.n, device=dev)]
            return [t__.detach().float().contiguous() for t__ in out_]
        rest_A, rest_B = rest_arrays(model), rest_arrays(modelB)
        XA_rest, XB_rest = model.X.detach().float(), modelB.X.detach().float()
        mu_log_ref = float(fz_['mu_log']) if 'mu_log' in fz_.files else None

        def rest_s(grad=True):
            if args.learn_rest == 'free':
                return sig_t[0] if grad else sig_t.detach()[0]
            assert mu_log_ref is not None, 'coupled rest shape: the sag field needs mu_log (rerun stage sag)'
            col5 = theta_of(shared if grad else shared.detach())[:, 5].mean()
            return torch.exp(mu_log_ref-col5)                   # s = G_ref / G

        LA_bond = rest_A[8].norm(dim=-1)

        def rest_interp(s_):
            # linear in s (exact): positions, levers ri / rj, bond vectors dX; from the bond length L(s) = |dX(s)| (exact):
            # normals dX / L, stiffnesses ~ A / L, least-squares weights A / L, dashpots ~ sqrt(k m); the moment inverse
            # and nodal volumes are interpolated linearly (relative error ~1e-3 over s in [0, 1.5])
            out_ = [a__+s_*(b__-a__) for a__, b__ in zip(rest_A, rest_B)]
            dX_ = out_[8]
            L_ = dX_.norm(dim=-1)
            r_ = LA_bond/L_
            out_[0] = dX_/L_[:, None]
            out_[3] = rest_A[3]*r_[:, None]
            out_[4], out_[5], out_[6] = (rest_A[k__]*r_.sqrt() for k__ in (4, 5, 6))
            out_[7] = rest_A[7]*r_
            return out_

        def rest_now(grad=True):
            return rest_interp(rest_s(grad))
        if args.rest_check:
            for s_chk in (.5, 1.5):
                gC = dict(geometry); gC['points'] = P_-s_chk*u_rel_np
                gC['contacts'] = geometry['contacts']-s_chk*.5*(u_rel_np[geometry['pairs'][:, 0]]+u_rel_np[geometry['pairs'][:, 1]])
                mC = same_tets(build(gC, args.E, args.nu, dev, 'noslip', .25, .3, f32, **kwB))
                exact_ = rest_arrays(mC)
                interp_ = rest_interp(torch.tensor(s_chk, device=dev))
                print('rest interpolation check at s = %.1f: max relative error per array %s' % (s_chk, [
                    '%.1e' % float((i__-c__).abs().max()/c__.abs().max().clamp(min=1e-30)) for i__, c__ in zip(interp_, exact_)]), flush=True)
        print('learned rest shape (%s): bodies move up to %.2f mm per unit s; rest arrays differ by up to %s' % (
            args.learn_rest, float((XB_rest-XA_rest).norm(dim=-1).max())*1e3,
            ['%.2g' % float(((b__-a__).abs().max()/a__.abs().max().clamp(min=1e-30))) for a__, b__ in zip(rest_A, rest_B)]), flush=True)
    model.set_environment(gravity=meta['gravity'], ground_point=[0, 0, ground_z+args.ground_offset], ground_normal=meta['ground_normal'],
                          contact_radius=model.contact_radius.cpu().numpy(), ground_mode='noslip')
    from experiments.foam_dem_warp import rollout as _warp_rollout_raw      # (warp_rollout is local to main below)

    def warp_rollout(wf_, theta_, state_, dt_, sub_, n_, seg_=None):
        if rest_A is not None and wf_ is wf:
            return _warp_rollout_raw(wf_, theta_, state_, dt_, sub_, n_, seg_, rest=rest_now(torch.is_grad_enabled()))
        return _warp_rollout_raw(wf_, theta_, state_, dt_, sub_, n_, seg_)
    model_dt, dt_safety = model, .5
    if args.cosserat:        # time step from the enhancement-off (stiffest) configuration, float64, safety .4
        model_dt, dt_safety = build(geometry, args.E, args.nu, dev, 'noslip', .25, .3, torch.float64, shell_eas=False, **kw_model), .4
        print('cosserat model: thin bodies %d, enhanced %d' % (int(model.thin.sum()), int(model.eas_on.sum())), flush=True)
    x0, q0, _, _ = model.initial_state()
    v0t = torch.tensor(vel['v0'], dtype=f32, device=dev); w0t = torch.tensor(vel['w0'], dtype=f32, device=dev)
    com_t = torch.tensor(vel['com'], dtype=f32, device=dev)
    vinit = (v0t+torch.cross(w0t.expand_as(x0), x0-com_t, dim=-1)).contiguous()
    winit = w0t.expand_as(x0).clone()
    dv0 = torch.zeros(3, dtype=f32, device=dev, requires_grad=True)     # learned initial velocity / spin corrections
    dw0 = torch.zeros(3, dtype=f32, device=dev, requires_grad=True)
    rot0 = torch.zeros(3, dtype=f32, device=dev, requires_grad=True)       # frame-0 pose correction: rotation vector (rad)
    dx0 = torch.zeros(3, dtype=f32, device=dev, requires_grad=True)        # and shift (mm)
    offs_t = torch.tensor(offsets[1:], dtype=f32, device=dev, requires_grad=True)   # camera time offsets (frames), camera 0 fixed
    sig_t = torch.tensor([args.rest_init], dtype=f32, device=dev, requires_grad=True)   # --learn-rest free
    dtilt = torch.zeros(2, dtype=f32, device=dev, requires_grad=True)       # --tilt-relative: impact tilt (mm)
    extras = dict(rot0=rot0, dx0=dx0, offs=offs_t, sig=sig_t, dtilt=dtilt)

    if args.init_from is not None:       # resume the initial-state corrections too
        js_ = json.loads(args.init_from.read_text())
        if 'dv0' in js_ and args.learn_init:
            with torch.no_grad():
                dv0.copy_(torch.tensor(js_['dv0'], dtype=f32, device=dev)); dw0.copy_(torch.tensor(js_['dw0'], dtype=f32, device=dev))
        with torch.no_grad():
            for k_, t_ in extras.items():
                if k_ in js_:
                    t_.copy_(torch.tensor(js_[k_], dtype=f32, device=dev))

    def rotvec_quat(r):
        th = torch.sqrt((r*r).sum()+1e-12)
        return torch.cat((torch.cos(th/2)[None], torch.sin(th/2)/th*r))

    def quat_mul_t(a, b):          # (w, x, y, z), a applied after b
        w1, x1, y1, z1 = a.unbind(-1); w2, x2, y2, z2 = b.unbind(-1)
        return torch.stack((w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2, w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2), -1)

    def init_state(grad=True):
        if not args.learn_init:
            return x0, q0, vinit, winit
        dv, dw = (dv0, dw0) if grad else (dv0.detach(), dw0.detach())
        w_ = w0t+dw
        xs_, qs_, cen_ = x0, q0, com_t
        if rest_A is not None:
            s_ = rest_s(grad)
            xs_ = (XA_rest+s_*(XB_rest-XA_rest)).contiguous()
        if args.learn_pose:
            rr, dd = (rot0, dx0) if grad else (rot0.detach(), dx0.detach())
            Rq = torch.matrix_exp(skew(rr))
            cen_ = com_t+1e-3*dd
            xs_ = (cen_+(xs_-com_t)@Rq.T).contiguous()
            qs_ = quat_mul_t(rotvec_quat(rr).expand_as(q0), q0).contiguous()
        return xs_, qs_, (v0t+dv+torch.cross(w_.expand_as(xs_), xs_-cen_, dim=-1)).contiguous(), w_.expand_as(xs_).contiguous()
    ncol = 9 if args.contact == 'smooth' else 8 if args.contact in ('coulomb', 'hertz') else 7   # smooth: friction, restitution logits; coulomb / hertz: log mu_f
    ogden_col = -1
    if args.nonlinear_bonds and args.ogden_learn:
        ogden_col = ncol; ncol += 1      # log(alpha / ogden_alpha)
    contact_col = -1
    if args.contact_field:
        assert args.contact == 'hertz'
        contact_col = ncol; ncol += 1    # per-body contact-surface offset (m), filled from the contact field
    if args.visco:
        ncol += 2                        # Zener: log g, log tau multipliers (last two columns)
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
    if args.param == 'Enu':
        # (E, nu) parameterisation: E = E0 e^p4, nu = 0.5 sigmoid(z0 + p5); the same material set as (lam, mu) but nu
        # is bounded in (0, 0.5) by construction (no clamp needed near incompressibility) and E, nu act on separate effects
        assert not args.per_body
        nu0 = args.nu
        z0 = float(np.log(2*nu0/(1-2*nu0)))
        lam0, mu0 = nu0/((1+nu0)*(1-2*nu0)), 1/(2*(1+nu0))          # per unit E

        def theta_of(p):
            E = torch.exp(p[4])
            nu = .5*torch.sigmoid(z0+p[5])
            lam, mu = E*nu/((1+nu)*(1-2*nu)), E/(2*(1+nu))
            th = torch.cat((p[:4], torch.log(lam/lam0)[None], torch.log(mu/mu0)[None], p[6:]))
            return th.expand(model.n, ncol)

        def e_nu(p):
            return dict(E_multiplier=float(torch.exp(p[4])), nu=float(.5*torch.sigmoid(z0+p[5])))
    if args.truevol:                 # springs carry mu: columns 0-3 (kn, ks, kb, kt multipliers) follow column 5
        _theta_base = theta_of
        theta_of = lambda p_: (lambda t_: torch.cat((t_[:, 5:6].expand(-1, 4), t_[:, 4:]), 1))(_theta_base(p_))
    cfield = torch.zeros(6, dtype=f32, device=dev, requires_grad=True)     # contact field coefficients (mm)
    if contact_col >= 0:
        gdc = torch.tensor(meta['gravity'], dtype=f32, device=dev); gdc = gdc/gdc.norm()
        e1c = torch.linalg.cross(gdc, torch.tensor([1., 0., 0.], device=dev)); e1c = e1c/e1c.norm(); e2c = torch.linalg.cross(gdc, e1c)
        uc, vc_ = (X-com)@e1c, (X-com)@e2c
        Rob = float(torch.sqrt(uc**2+vc_**2).max())
        uc, vc_ = uc/Rob, vc_/Rob
        cbasis = torch.stack((torch.ones_like(uc), uc, vc_, uc**2, vc_**2, uc*vc_), 1)
        if args.init_from is not None and 'cfield' in json.loads(args.init_from.read_text()):
            with torch.no_grad():
                cfield.copy_(torch.tensor(json.loads(args.init_from.read_text())['cfield'], dtype=f32, device=dev))
        _theta_c = theta_of

        # --tilt-relative: a small body rotation phi lowers body b by Rob (-u_b phi.e2 + v_b phi.e1), i.e. contact-field
        # c1 = -Rob phi.e2, c2 = Rob phi.e1 (mm); phi at the hit = rot0 + dw0 t_hit (corrections only; constant base pose / spin
        # drop out of the reparametrisation)
        t_hit = float(meta['hit_frame'])/fps

        def c_eq(rot_, w_):
            ph_ = rot_+w_*t_hit
            return 1e3*Rob*torch.stack((-(ph_*e2c).sum(), (ph_*e1c).sum()))

        def cfield_now(grad=True):
            cf_ = cfield if grad else cfield.detach()
            if not args.tilt_relative:
                return cf_
            dt_, dw_ = (dtilt, dw0) if grad else (dtilt.detach(), dw0.detach())
            return torch.cat((cf_[:1], dt_-c_eq(rot0.detach(), dw_), cf_[3:]))

        def theta_of(p_, _f=_theta_c):
            th_ = _f(p_)
            cf_ = cfield_now(torch.is_grad_enabled() and p_.requires_grad)
            off_ = 1e-3*(cbasis@cf_)
            return torch.cat((th_[:, :contact_col], off_[:, None], th_[:, contact_col+1:]), 1)
    optimizer = torch.optim.Adam([dict(params=[shared], lr=args.lr)]+([dict(params=[dv0]+([] if args.freeze_spin else [dw0]), lr=args.lr_init)] if args.learn_init else [])
                                 +([dict(params=[rot0], lr=args.lr_rot), dict(params=[dx0], lr=args.lr_shift)] if args.learn_init and args.learn_pose else [])
                                 +([dict(params=[offs_t], lr=args.lr_offs)] if args.learn_offsets else [])
                                 +([dict(params=[sig_t], lr=args.lr_rest)] if args.learn_rest == 'free' else [])
                                 +([dict(params=[cfield], lr=args.lr_cfield)] if contact_col >= 0 else [])
                                 +([dict(params=[dtilt], lr=args.lr_cfield)] if args.tilt_relative else []), betas=(.9, args.adam_beta2))
    if args.tilt_relative:
        assert contact_col >= 0 and args.learn_init
        with torch.no_grad():
            if args.init_from is None or 'dtilt' not in json.loads(args.init_from.read_text()):
                dtilt.copy_(cfield[1:3]+c_eq(rot0, dw0))
        print('tilt-relative: dtilt %s mm (t_hit %.4f s)' % (np.round(dtilt.detach().cpu().numpy(), 3).tolist(), t_hit), flush=True)
        if args.learn_pose:          # geometry check of c_eq: lowering of every body under a small pose rotation, fitted on (u, v)
            with torch.no_grad():
                keep_r = rot0.clone()
                gd_ = torch.tensor(meta['gravity'], dtype=f32, device=dev); gd_ = gd_/gd_.norm()
                x_a = init_state(False)[0].clone()
                for ph_ in ([.01, 0., 0.], [0., .01, 0.]):
                    rot0.copy_(keep_r+torch.tensor(ph_, dtype=f32, device=dev))
                    low_ = 1e3*((init_state(False)[0]-x_a)@gd_)                          # mm, + = towards the ground
                    A_ = cbasis[:, :3].to(f32)
                    fit_ = torch.linalg.lstsq(A_, low_[:, None]).solution[:, 0]
                    print('tilt-relative check, rotation %s: fitted lowering c1 c2 %s mm | c_eq %s mm' % (
                        ph_, np.round(fit_[1:].cpu().numpy(), 4).tolist(),
                        np.round(c_eq(torch.tensor(ph_, dtype=f32, device=dev), torch.zeros(3, device=dev)).cpu().numpy(), 4).tolist()), flush=True)
                rot0.copy_(keep_r)
    wf = WarpFoam(model, contact=args.contact, layer_dx=args.layer_dx, friction0=args.friction0, restitution0=args.restitution0, contact_scale=args.contact_scale,
                  visco=bool(args.visco), visco_g0=args.visco_g0, visco_tau0=args.visco_tau0, visco_cols=(ncol-2, ncol-1), ogden_col=ogden_col, contact_offset_col=contact_col,
                  implicit_damping=bool(args.implicit_damping), graphs=bool(args.graphs))
    wf.thornton = args.thornton
    sub_scale = [1.]
    frames = meta['frames']-1

    def summary(p):
        th = theta_of(p.detach())
        return th[0].tolist() if not args.per_body else dict(mean=th.mean(0).tolist(), std=th.std(0).tolist())

    def cut_init_grads():
        """After a backward with wf.adj2_stash: dv0 / dx0 gradients from the weighted adjoint (wf.adj_init2) instead of
        the plain one (wf.adj_init_plain); other terms of their gradient (priors) are kept."""
        tens_ = [dv0]+([dx0] if args.learn_pose else [])+([dw0] if args.init_cut == 2 else [])+([rot0] if args.learn_pose and args.init_cut in (2, 3) else [])
        st_ = init_state(True)
        pairs_ = [(t_, (a_-b_)[:t_.shape[0]].to(t_.dtype)) for a_, b_, t_ in zip(wf.adj_init2, wf.adj_init_plain, st_) if t_.requires_grad]
        g_ = torch.autograd.grad([p_[0] for p_ in pairs_], tens_, grad_outputs=[p_[1] for p_ in pairs_], allow_unused=True)
        for t_, gg_ in zip(tens_, g_):
            if gg_ is not None:
                t_.grad = gg_ if t_.grad is None else t_.grad+gg_

    real_contact = [None]
    offs_cut = [10**9]          # --init-cut 2: rendered frames >= this give the camera offsets no gradient

    def cut_weights(pos, n_out):
        """Frame weights 1 before min(first simulated contact, first video contact), 0 from it on."""
        if real_contact[0] is None:
            from experiments.flight_detect import detect
            md_ = np.stack([np.stack([masks[f, c].cpu().numpy() for c in range(len(CAMS))]) for f in range(meta['frames'])])
            real_contact[0] = int(detect(md_)['contact'])
            print('video first contact frame %d' % real_contact[0], flush=True)
        with torch.no_grad():
            gp_t = torch.tensor([wf.ground_p[0], wf.ground_p[1], wf.ground_p[2]], dtype=f32, device=dev)
            gn_t = torch.tensor(meta['ground_normal'], dtype=f32, device=dev)
            reach = model.contact_radius.to(f32)+(theta_of(shared.detach())[:, contact_col] if contact_col >= 0 else 0.)
            touching = (((pos.detach()-gp_t)@gn_t) < reach[None]).any(1)
            fs_ = int(torch.nonzero(touching)[0]) if bool(touching.any()) else len(pos)
        f_cut = min(fs_, real_contact[0])
        return [1. if f < f_cut else 0. for f in range(n_out)], fs_, f_cut

    def rollout(n, substeps, grad=True, init_grad=None):
        theta = theta_of(shared if grad else shared.detach())
        for _ in range(5):
            sub = int(np.ceil(substeps*sub_scale[0]))
            pos, quat = warp_rollout(wf, theta, init_state(grad if init_grad is None else init_grad), 1/fps/sub, sub, n, args.segment or None)
            if bool(torch.isfinite(pos).all()) or sub_scale[0] >= 4.:
                return pos, quat
            sub_scale[0] = min(4., sub_scale[0]*2.)
            print('non-finite rollout at %d substeps: raising to x%.2f of the estimate' % (sub, sub_scale[0]), flush=True)
        return pos, quat

    def place(pos, quat, f, c=0):
        """State at frame f + this camera's time offset (linear between the neighbouring frames)."""
        t = f+(offsets[c] if not (args.learn_offsets and c > 0) else (offs_t[c-1] if f+1 < offs_cut[0] else offs_t[c-1].detach()))
        lo = int(np.clip(np.floor(float(t)), 0, len(pos)-2)); u = t-lo
        x = (1-u)*pos[lo]+u*pos[lo+1]
        q1 = quat[lo+1]*torch.sign((quat[lo]*quat[lo+1]).sum(-1, keepdim=True))
        q = (1-u)*quat[lo]+u*q1
        q = q/q.norm(dim=-1, keepdim=True)
        R = conj(body_rotation(q))
        F = conj(model.deformation_gradient(x))
        deformed.apply(to_foam(x), body_R=R, body_F=F, mode=args.render_mode)

    def evaluate(substeps):
        with torch.no_grad():
            pos, quat = rollout(frames, substeps, grad=False)
            ious = np.zeros((frames+1, len(CAMS)))
            for f in range(frames+1):
                for c in range(len(CAMS)):
                    place(pos, quat, f, c)
                    ious[f, c] = iou(silhouette(c), masks[f, c])
        tr, te = slice(1, args.train_frames+1), slice(args.train_frames+1, frames+1)
        return dict(iou_trainviews_trainframes=float(ious[tr][:, train_cams].mean()),
                    iou_evalview_trainframes=float(ious[tr, args.eval_cam].mean()),
                    iou_trainviews_futureframes=float(ious[te][:, train_cams].mean()),
                    iou_evalview_futureframes=float(ious[te, args.eval_cam].mean()),
                    iou_per_frame_cam=ious.tolist(), parameters=summary(shared), substeps=substeps)

    def substeps_now():
        th = theta_of(shared.detach())[:, :7].to(model_dt.dtype)
        vs = 1.
        if args.visco:                   # instantaneous spring stiffness (1 + g) k: frequencies x sqrt(1 + g)
            vs = float(np.sqrt(1+args.visco_g0*np.exp(float(theta_of(shared.detach())[:, ncol-2].max()))))
        return int(np.ceil(vs*args.substep_scale/fps/model_dt.stable_dt(th, safety=dt_safety, include_damping=not wf.implicit_damping)))
    if args.synthetic is not None:     # recovery test: the 'real' masks are simulated with known parameters
        syn = json.loads(args.synthetic.read_text())
        keep_ = shared.detach().clone(); keep_dv, keep_dw = dv0.detach().clone(), dw0.detach().clone(); keep_cf = cfield.detach().clone()
        # the truth's start-state errors are applied whether or not they are learned (a fixed start then carries them as a
        # registration error); the flags are restored after the masks are made
        keep_flags = (args.learn_init, args.learn_pose, args.learn_offsets)
        args.learn_init = int(args.learn_init or any(k_ in syn for k_ in ('dv0', 'dw0', 'rot0', 'dx0')))
        args.learn_pose = int(args.learn_pose or any(k_ in syn for k_ in ('rot0', 'dx0')))
        args.learn_offsets = int(args.learn_offsets or 'offs' in syn)
        with torch.no_grad():
            shared.copy_(torch.tensor(syn['best']['parameters'][:ncol], dtype=f32, device=dev).expand_as(shared))
            if args.learn_init:
                dv0.copy_(torch.tensor(syn.get('dv0', [0., 0., 0.]), dtype=f32, device=dev))
                dw0.copy_(torch.tensor(syn.get('dw0', [0., 0., 0.]), dtype=f32, device=dev))
            cfield.copy_(torch.tensor(syn.get('cfield', [0.]*6), dtype=f32, device=dev))
            keep_ex = {k_: t_.detach().clone() for k_, t_ in extras.items()}
            for k_, t_ in extras.items():
                if k_ in syn:
                    t_.copy_(torch.tensor(syn[k_], dtype=f32, device=dev))
            if args.tilt_relative and 'dtilt' not in syn:     # the truth's impact tilt from its c1 c2 and pose / spin
                dtilt.copy_(cfield[1:3]+c_eq(rot0, dw0))
            subs = substeps_now()
            st_s = init_state(False)
            print('synthetic start vs registration: positions moved up to %.3f mm, mean velocity change %s m/s, offsets %s' % (
                float((st_s[0]-x0).norm(dim=-1).max())*1e3, np.round((st_s[2]-vinit).mean(0).cpu().numpy(), 4).tolist(),
                np.round(offs_t.detach().cpu().numpy(), 3).tolist() if args.learn_offsets else offsets[1:]), flush=True)
            pos_s, quat_s = warp_rollout(wf, theta_of(shared), st_s, 1/fps/subs, subs, frames)
            for f in range(frames+1):
                for c in range(len(CAMS)):
                    place(pos_s, quat_s, f, c)
                    masks[f, c] = (silhouette(c).detach() > .5).float()
            shared.copy_(keep_); dv0.copy_(keep_dv); dw0.copy_(keep_dw); cfield.copy_(keep_cf)      # back to the --init-from start
            for k_, t_ in extras.items():
                t_.copy_(keep_ex[k_])
        args.learn_init, args.learn_pose, args.learn_offsets = keep_flags
        print('synthetic masks from', args.synthetic, flush=True)
    real_feat = {}
    if args.stage == 'render':
        # real | simulation composited over the empty-table frame | silhouettes (red real, green sim)
        with torch.no_grad():
            if args.theta_npz is not None:
                th = torch.tensor(np.load(args.theta_npz)['theta'], dtype=f32, device=dev)
                theta_of_r = th
            else:
                theta_of_r = theta_of(shared.detach())
            sub = int(np.ceil(1/fps/model_dt.stable_dt(theta_of_r[:, :7].to(model_dt.dtype), safety=dt_safety, include_damping=not wf.implicit_damping)))
            pos, quat = warp_rollout(wf, theta_of_r, init_state(False), 1/fps/sub, sub, frames)
            def place_r(f, c):
                t = f+offsets[c]; lo = int(np.clip(np.floor(t), 0, len(pos)-2)); u = t-lo
                x = (1-u)*pos[lo]+u*pos[lo+1]
                q = (1-u)*quat[lo]+u*quat[lo+1]*torch.sign((quat[lo]*quat[lo+1]).sum(-1, keepdim=True)); q = q/q.norm(dim=-1, keepdim=True)
                deformed.apply(to_foam(x), body_R=conj(body_rotation(q)), body_F=conj(model.deformation_gradient(x)), mode=args.render_mode)
            order = [c for c in range(len(CAMS)) if c != args.eval_cam]+[args.eval_cam]
            tiles = {}
            for c in order:
                bg = np.asarray(Image.open(dyn/'rest'/f'background_{CAMS[c]}.png').convert('RGB').resize((W, H)), np.float32)/255
                for f in range(frames+1):
                    real = np.asarray(Image.open(dyn/'images'/CAMS[c]/f'{f:02d}.png').convert('RGB').resize((W, H)), np.float32)/255
                    place_r(f, c)
                    rgb, al = render(foam, cams[c]); al = al.reshape(H, W).clamp(0, 1)
                    sim = (rgb.clamp(0, 1)+(1-al)[..., None]*torch.tensor(bg, device=dev)).cpu().numpy()
                    m, s_ = masks[f, c].cpu().numpy() > .5, al.cpu().numpy() > .5
                    ov = np.zeros_like(real); ov[..., 0] = m; ov[..., 1] = s_
                    tiles[f, c] = (real, sim, ov, m | s_)
            frames_img = []
            for f in range(frames+1):
                rows = []
                for c in order:
                    ext = max(max(np.ptp(np.nonzero(tiles[g, c][3])[0]), np.ptp(np.nonzero(tiles[g, c][3])[1])) for g in range(frames+1))
                    half = int(.75*ext)+8
                    ys_, xs_ = np.nonzero(tiles[f, c][3]); cy_, cx_ = int(ys_.mean()), int(xs_.mean())
                    crop = lambda im: np.pad(im, ((half, half), (half, half), (0, 0)))[cy_:cy_+2*half, cx_:cx_+2*half]
                    row = np.concatenate([crop(t) for t in tiles[f, c][:3]], 1)
                    rows.append(np.asarray(Image.fromarray((row*255).astype(np.uint8)).resize((3*260, 260))))
                wmax = max(r.shape[1] for r in rows)
                rows = [np.pad(r, ((0, 0), (0, wmax-r.shape[1]), (0, 0))) for r in rows]
                img = Image.fromarray(np.concatenate(rows, 0))
                from PIL import ImageDraw
                d = ImageDraw.Draw(img)
                d.text((8, 6), 'frame %d  (%s)' % (f, 'train' if 1 <= f <= args.train_frames else ('initial' if f == 0 else 'FUTURE')), fill=(255, 255, 0))
                for k_, c in enumerate(order):
                    d.text((8, 260*k_+244), CAMS[c]+(' (held-out view)' if c == args.eval_cam else ' (train view)'), fill=(255, 255, 0))
                frames_img.append(img)
            frames_img[0].save(args.out/'compare.gif', save_all=True, append_images=frames_img[1:], duration=250, loop=0)
            for f in (0, 7, 8, 10, 14, 19):
                frames_img[f].save(args.out/f'compare_frame{f:02d}.png')
        print('saved', args.out/'compare.gif', flush=True)
        return
    if args.stage == 'scan':
        rows = []
        for c in args.scan:
            with torch.no_grad():
                shared.copy_(base.expand_as(shared)); shared[..., :6] += c
            ev = evaluate(substeps_now())
            if args.viz:
                with torch.no_grad():
                    pos, quat = rollout(frames, substeps_now(), grad=False)
                    rows_img = []
                    for c_ in range(len(CAMS)):
                        tiles = []
                        for f in (4, 7, 8, 10, 12, 15, 19):
                            place(pos, quat, f, c_)
                            al = silhouette(c_).cpu().numpy() > .5
                            m_ = masks[f, c_].cpu().numpy() > .5
                            o = np.zeros(al.shape+(3,), np.uint8); o[..., 0] = m_*255; o[..., 1] = al*255
                            tiles.append(o[::2, ::2])
                        rows_img.append(np.concatenate(tiles, 1))
                    Image.fromarray(np.concatenate(rows_img, 0)).save(args.out/f'overlay_{c:+.1f}.png')
            per_frame = np.array(ev['iou_per_frame_cam']).mean(1)
            rows.append(dict(log_offset=c, stiffness_factor=float(np.exp(c)), **{k_: v for k_, v in ev.items() if k_ != 'parameters'}))
            print('offset %+.1f (x%.4f) substeps %d | train views: train %.3f future %.3f | eval view: train %.3f future %.3f | per frame %s' % (
                c, np.exp(c), ev['substeps'], ev['iou_trainviews_trainframes'], ev['iou_trainviews_futureframes'],
                ev['iou_evalview_trainframes'], ev['iou_evalview_futureframes'], np.round(per_frame[6:], 2).tolist()), flush=True)
        (args.out/'scan.json').write_text(json.dumps(rows, indent=1))
        return
    if args.stage == 'framediag':
        # per frame and camera around the impact: IoU, silhouette area, centroid offset, height and width (px), sim vs real;
        # zoomed overlays (red real, green sim, yellow both) of every diagnosed frame (overlays.npz) and a 7..10 grid
        with torch.no_grad():
            th = theta_of(shared.detach()); sub = substeps_now()
            pos, quat = warp_rollout(wf, th, init_state(False), 1/fps/sub, sub, frames)
        def stats(m_):
            m_ = m_ > .5
            if not m_.any():
                return dict(area=0, cy=np.nan, cx=np.nan, h=0, w=0)
            ys, xs = np.nonzero(m_)
            return dict(area=int(m_.sum()), cy=float(ys.mean()), cx=float(xs.mean()), h=int(np.ptp(ys))+1, w=int(np.ptp(xs))+1)
        rows, tiles = [], {}
        for f in range(max(0, args.diag_from), min(frames, args.diag_to)+1):
            for c in range(len(CAMS)):
                place(pos, quat, f, c)
                sim = silhouette(c).detach().cpu().numpy(); real = masks[f, c].cpu().numpy()
                a, b_ = sim > .5, real > .5
                iou_ = float((a & b_).sum()/max((a | b_).sum(), 1))
                ss, rs = stats(sim), stats(real)
                r = dict(frame=f, cam=CAMS[c], eval=(c == args.eval_cam), iou=iou_, area_ratio=ss['area']/max(rs['area'], 1),
                         dy=ss['cy']-rs['cy'], dx=ss['cx']-rs['cx'], h_sim=ss['h'], h_real=rs['h'], w_sim=ss['w'], w_real=rs['w'])
                rows.append(r)
                print('frame %2d %-6s%s IoU %.3f | area sim/real %.2f | centroid shift dx %+5.1f dy %+5.1f px | height %3d vs %3d | width %3d vs %3d' % (
                    f, CAMS[c], '*' if r['eval'] else ' ', iou_, r['area_ratio'], r['dx'], r['dy'], ss['h'], rs['h'], ss['w'], rs['w']), flush=True)
                ov = np.zeros(a.shape+(3,), np.uint8); ov[..., 0] = b_*255; ov[..., 1] = a*255
                ys_, xs_ = np.nonzero(a | b_); cy_, cx_ = int(ys_.mean()), int(xs_.mean()); hh = 60
                tiles[f, c] = np.pad(ov, ((hh, hh), (hh, hh), (0, 0)))[cy_:cy_+2*hh, cx_:cx_+2*hh]
        np.savez_compressed(args.out/'overlays.npz', frames=np.array([k_[0] for k_ in tiles]), cams=np.array([k_[1] for k_ in tiles]),
                            tiles=np.stack(list(tiles.values())))                 # red real, green sim, yellow both
        if all((f, c) in tiles for f in range(7, 11) for c in range(len(CAMS))):
            grid = np.concatenate([np.concatenate([tiles[f, c] for f in range(7, 11)], 1) for c in range(len(CAMS))], 0)
            Image.fromarray(grid).resize((grid.shape[1]*2, grid.shape[0]*2), Image.NEAREST).save(args.out/'impact_overlays.png')
        (args.out/'framediag.json').write_text(json.dumps(rows, indent=1))
        return
    if args.stage == 'timeshift':
        # which simulated instant best matches each real frame around the impact? rollout sampled 8x per frame; for real
        # frame f (camera time offset included) IoU and width against the simulation at f + s, s in [-1, +1.5] frames
        K = 8
        with torch.no_grad():
            th = theta_of(shared.detach()); sub = substeps_now(); sub = int(np.ceil(sub/K))*K
            pos, quat = warp_rollout(wf, th, init_state(False), 1/fps/sub, sub//K, K*frames)
        def stats_w(m_):
            xs = np.nonzero(m_)[1]
            return int(np.ptp(xs))+1 if len(xs) else 0
        rows = []
        shifts = np.arange(-K, int(1.5*K)+1)/K
        for f in range(args.diag_from, args.diag_to+1):
            for c in range(len(CAMS)):
                real = masks[f, c].cpu().numpy() > .5
                for s_ in shifts:
                    t = (f+offsets[c]+s_)*K
                    lo = int(np.clip(np.floor(t), 0, len(pos)-2)); u = t-lo
                    x = (1-u)*pos[lo]+u*pos[lo+1]
                    q1 = quat[lo+1]*torch.sign((quat[lo]*quat[lo+1]).sum(-1, keepdim=True)); q = (1-u)*quat[lo]+u*q1; q = q/q.norm(dim=-1, keepdim=True)
                    deformed.apply(to_foam(x), body_R=conj(body_rotation(q)), body_F=conj(model.deformation_gradient(x)), mode=args.render_mode)
                    sim = silhouette(c).detach().cpu().numpy() > .5
                    rows.append(dict(frame=f, cam=CAMS[c], shift=float(s_), iou=float((sim & real).sum()/max((sim | real).sum(), 1)),
                                     w_sim=stats_w(sim), w_real=stats_w(real)))
            best = {}
            for s_ in shifts:
                best[s_] = np.mean([r['iou'] for r in rows if r['frame'] == f and r['shift'] == s_])
            sb = max(best, key=best.get)
            print('real frame %2d: IoU at shift 0 %.3f | best shift %+.3f frames -> IoU %.3f | widths at best: sim %s real %s' % (
                f, best[0.], sb, best[sb], [r['w_sim'] for r in rows if r['frame'] == f and r['shift'] == sb],
                [r['w_real'] for r in rows if r['frame'] == f and r['shift'] == sb]), flush=True)
            print('    IoU vs shift: '+' '.join('%+.2f:%.2f' % (s_, best[s_]) for s_ in shifts), flush=True)
        (args.out/'timeshift.json').write_text(json.dumps(rows, indent=1))
        return
    if args.stage == 'frameview':
        # side-by-side crops per camera: real video frames (r<f>) and the simulation rendered over the empty-table
        # background (s<t>, t may be fractional; camera time offset included), all cropped around the same window
        K = 8
        with torch.no_grad():
            th = theta_of(shared.detach()); sub = substeps_now(); sub = int(np.ceil(sub/K))*K
            pos, quat = warp_rollout(wf, th, init_state(False), 1/fps/sub, sub//K, K*frames)
        cols = args.view_columns.split(',')
        from PIL import ImageDraw, ImageFont
        import subprocess
        cjk = subprocess.run(['fc-match', '-f', '%{file}', 'Noto Sans CJK SC:lang=zh'], capture_output=True, text=True).stdout
        font = ImageFont.truetype(cjk, 20)
        hh, zoom, top = 75, 2, 30
        rows_img = []
        for c in range(len(CAMS)):
            bg = np.asarray(Image.open(dyn/'rest'/f'background_{CAMS[c]}.png').convert('RGB').resize((W, H)), np.float32)/255
            tiles = []
            fs = [float(k[1:]) for k in cols]
            ms = [masks[int(round(f_)), c].cpu().numpy() > .5 for f_ in fs]
            ys, xs = np.nonzero(np.any(ms, 0)); cy, cx = int(ys.mean()), int(xs.mean())
            for k in cols:
                t_ = float(k[1:])
                if k[0] == 'r':
                    img = np.asarray(Image.open(dyn/'images'/CAMS[c]/f'{int(t_):02d}.png').convert('RGB').resize((W, H), Image.BILINEAR), np.float32)/255
                else:
                    t = (t_+offsets[c])*K
                    lo = int(np.clip(np.floor(t), 0, len(pos)-2)); u = t-lo
                    with torch.no_grad():
                        x = (1-u)*pos[lo]+u*pos[lo+1]
                        q1 = quat[lo+1]*torch.sign((quat[lo]*quat[lo+1]).sum(-1, keepdim=True)); q = (1-u)*quat[lo]+u*q1; q = q/q.norm(dim=-1, keepdim=True)
                        deformed.apply(to_foam(x), body_R=conj(body_rotation(q)), body_F=conj(model.deformation_gradient(x)), mode=args.render_mode)
                        rgb, alpha = render(foam, cams[c])
                    rgb = rgb.reshape(H, W, 3).clamp(0, 1).cpu().numpy(); al = alpha.reshape(H, W, 1).cpu().numpy()
                    img = rgb+(1-al)*bg
                crop = np.pad(img, ((hh, hh), (hh, hh), (0, 0)))[cy:cy+2*hh, cx:cx+2*hh]
                im = Image.fromarray((np.clip(crop, 0, 1)*255).astype(np.uint8)).resize((2*hh*zoom, 2*hh*zoom), Image.LANCZOS)
                canvas = Image.new('RGB', (im.width, im.height+top), (245, 244, 240)); canvas.paste(im, (0, top))
                lab = ('真实 帧 %d' % int(t_)) if k[0] == 'r' else ('模拟 帧 %g' % t_)
                ImageDraw.Draw(canvas).text((im.width//2, 4), f'{CAMS[c]}  {lab}', fill=(30, 30, 30), font=font, anchor='ma')
                tiles.append(np.asarray(canvas))
            rows_img.append(np.concatenate(tiles, 1))
        Image.fromarray(np.concatenate(rows_img, 0)).save(args.out/'frameview.png')
        print('saved', args.out/'frameview.png', flush=True)
        return
    if args.stage == 'features':
        # silhouette moment features per frame and camera, simulation (soft alpha) and real (mask): area, centroid
        # (cx, cy), second central moments (sxx, syy, sxy) -> sqrt widths and principal-axis angle. Saves features.json
        def moments(a_):
            H_, W_ = a_.shape
            ys = torch.arange(H_, device=a_.device, dtype=a_.dtype)[:, None]; xs = torch.arange(W_, device=a_.device, dtype=a_.dtype)[None, :]
            m0 = a_.sum(); cx = (a_*xs).sum()/m0; cy = (a_*ys).sum()/m0
            sxx = (a_*(xs-cx)**2).sum()/m0; syy = (a_*(ys-cy)**2).sum()/m0; sxy = (a_*(xs-cx)*(ys-cy)).sum()/m0
            return dict(area=float(m0), cx=float(cx), cy=float(cy), sx=float(sxx.sqrt()), sy=float(syy.sqrt()),
                        angle=float(0.5*torch.atan2(2*sxy, sxx-syy)))
        with torch.no_grad():
            th = theta_of(shared.detach()); sub = substeps_now()
            pos, quat = warp_rollout(wf, th, init_state(False), 1/fps/sub, sub, frames)
            out = dict(params=summary(shared), frames=list(range(frames+1)), cams=CAMS, sim={}, real={})
            for c, name in enumerate(CAMS):
                out['sim'][name] = []; out['real'][name] = []
                for f in range(frames+1):
                    place(pos, quat, f, c)
                    out['sim'][name].append(moments(silhouette(c).reshape(H, W)))
                    out['real'][name].append(moments(masks[f, c]))
        (args.out/'features.json').write_text(json.dumps(out))
        print('features saved', flush=True)
        return
    if args.stage == 'jitter':
        # short-wavelength (body-scale) motion around the impact: rollout sampled K times per frame; per sample the
        # displacement Laplacian lap_i = u_i - mean of u over i's bonded neighbours (u = x - X, rigid motion cancels to
        # first order), rms over all bodies and over the lowest 15% (contact side); and the kinetic energy split into the
        # centre-of-mass part and the rest (velocities by central differences). Saves jitter.json
        K = 8
        with torch.no_grad():
            th = theta_of(shared.detach()); sub = substeps_now(); sub = int(np.ceil(sub/K))*K
            nf = min(frames, 14)
            pos, _ = warp_rollout(wf, th, init_state(False), 1/fps/sub, sub//K, K*nf)
        P = pos.double().cpu().numpy(); X0 = P[0]
        bi_, bj_ = model.i.cpu().numpy(), model.j.cpu().numpy(); n_ = model.n
        deg = np.bincount(np.concatenate((bi_, bj_)), minlength=n_).astype(float)
        m_ = model.mass.double().cpu().numpy() if hasattr(model, 'mass') else np.ones(n_)
        gd = np.asarray(meta['gravity'], float); gd /= np.linalg.norm(gd)
        low = (X0@gd) >= np.quantile(X0@gd, .85)
        rows = []
        for k in range(len(P)):
            u = P[k]-X0
            s_ = np.zeros_like(u); np.add.at(s_, bi_, u[bj_]); np.add.at(s_, bj_, u[bi_])
            lap = u-s_/np.maximum(deg, 1)[:, None]
            r = dict(t_frames=k/K, lap_rms_mm=1e3*float(np.sqrt((lap**2).sum(1).mean())), lap_rms_bottom_mm=1e3*float(np.sqrt((lap[low]**2).sum(1).mean())))
            if 0 < k < len(P)-1:
                v = (P[k+1]-P[k-1])*fps*K/2
                vc = (m_[:, None]*v).sum(0)/m_.sum()
                r.update(ke_com=float(.5*m_.sum()*(vc**2).sum()), ke_rest=float(.5*(m_[:, None]*(v-vc)**2).sum()))
            rows.append(r)
        for r in rows[::K//2]:
            print('t %5.2f fr | Laplacian rms all %.3f mm, bottom %.3f mm | KE com %.3g, rest %.3g' % (
                r['t_frames'], r['lap_rms_mm'], r['lap_rms_bottom_mm'], r.get('ke_com', np.nan), r.get('ke_rest', np.nan)), flush=True)
        (args.out/'jitter.json').write_text(json.dumps(rows))
        return
    if args.stage == 'energy':
        # energy budget, every substep up to --diag-to: kinetic (translation + spin), elastic (model.energy), gravity;
        # material damping dissipation measured exactly as the kinetic-energy drop of the implicit pair-dashpot passes
        # (re-run on the pre-step state into scratch buffers; identical to the ones inside the step), split by region
        # (lowest 15% of the bodies along gravity vs the rest); the contact layer's drop measured from the velocities
        # just before / after its projection (normal, tangential, spin); integration error = what is left
        import warp as wp
        from experiments.foam_dem_warp import damping_pair_kernel
        assert wf.implicit_damping
        th_t = theta_of(shared.detach()).float().contiguous(); th = wp.from_torch(th_t)
        sub = substeps_now(); dt = 1/fps/sub; n_ = wf.n
        mk = lambda t, d: wp.from_torch(t.detach().float().contiguous().clone(), dtype=d)
        from experiments.foam_dem_warp import _pad_state, _state_dtypes
        dts_ = _state_dtypes(wf)
        st = tuple(mk(t_, d_) for t_, d_ in zip(_pad_state(wf, init_state(False)), dts_))
        st1 = tuple(wp.zeros(n_, dtype=d, device='cuda') for d in dts_)
        buf = wf._buffers(False)
        scratch = [(wp.zeros(n_, dtype=wp.vec3, device='cuda'), wp.zeros(n_, dtype=wp.vec3, device='cuda')) for _ in range(wf.n_colors)]
        m_t, I_t = model.mass.double(), model.inertia.double()
        g_t = torch.tensor(meta['gravity'], dtype=torch.float64, device=dev)
        gd = g_t/g_t.norm()
        low = (x0.double()@gd) >= torch.quantile(x0.double()@gd, .85)
        ke = lambda v, w: .5*(m_t*(v.double()**2).sum(1)+I_t*(w.double()**2).sum(1))
        gn = torch.tensor(meta['ground_normal'], dtype=torch.float64, device=dev); gn = gn/gn.norm()
        def total(s_):
            x, q, v, w = (wp.to_torch(a_) for a_ in s_[:4])
            with torch.no_grad():
                U = float(model.energy(th_t, x, q))
            K = float(ke(v, w).sum()); G = float(-(m_t*(x.double()@g_t)).sum())
            if getattr(wf, 'hertz', False):      # Hertz normal potential 8/15 E* sqrt(R) d^2.5 + Mindlin spring |F_t|^2 / (2 k_t)
                lam_ = model.k_v*torch.exp(th_t[:, 4].double()); mu_ = model.mu_ref*torch.exp(th_t[:, 5].double())
                nu_ = .5*lam_/(lam_+mu_); E_ = mu_*(3*lam_+2*mu_)/(lam_+mu_); Es_ = E_/(1-nu_**2); Gs_ = mu_/(2-nu_)
                R_ = model.contact_radius.double(); d_ = (R_-(x.double()-model.ground_point.double())@gn).clamp(min=0)
                hcs = wp.to_torch(s_[4]).double()
                kt_ = 8*Gs_*(R_*d_).sqrt()
                kt_ = wf.contact_scale*kt_
                U = U+float((wf.contact_scale*8/15*Es_*R_.sqrt()*d_**2.5).sum())+float(torch.where(kt_ > 0, (hcs[:, :3]**2).sum(1)/(2*kt_.clamp(min=1e-30)), torch.zeros_like(kt_)).sum())
                pen_stats.append((float(d_.max()), int((d_ > 0).sum())))
            vc = (m_t[:, None]*v.double()).sum(0)/m_t.sum()
            return K, U, G, vc
        nsteps = (args.diag_to+1)*sub; every = max(1, sub//8)
        pen_stats = []
        D = torch.zeros(n_, dtype=torch.float64, device=dev)
        Cn, Ct, Cs = (torch.zeros(n_, dtype=torch.float64, device=dev) for _ in range(3))
        K0, U0, G0, _ = total(st); E0 = K0+U0+G0
        rows = []
        for k in range(nsteps):
            v0, w0 = wp.to_torch(st[2]), wp.to_torch(st[3])
            k0 = ke(v0, w0)
            vv, ww = st[2], st[3]
            for c_ in range(wf.n_colors):
                wp.launch(damping_pair_kernel, n_, inputs=[st[0], st[1], vv, ww, wf.bond_of, c_, wf.bi, wf.bj, wf.ri, wf.rj, wf.cn, wf.ct, wf.cr,
                                                           th, wf.mass, wf.inertia, wf.fixed, dt, scratch[c_][0], scratch[c_][1]])
                vv, ww = scratch[c_]
            vd, wd = wp.to_torch(vv).double(), wp.to_torch(ww).double()
            D += k0-ke(vd, wd)
            for a_ in buf.values():
                a_.zero_()
            wf.step(th, st, st1, buf, dt)
            # contact layer: velocity just before the contact projection (damped velocity + force and gravity kick) vs
            # the stepped velocity; kinetic-energy drop split into normal / tangential translation and spin
            if getattr(wf, 'hertz', False):       # conservative contact force: its work is in the contact potential, not a loss
                st, st1 = st1, st
                if (k+1) % every == 0:
                    K, U, G, vc = total(st)
                    rows.append(dict(t_frames=(k+1)/sub, kinetic=K, elastic=U, gravity=G, total=K+U+G, lost=E0-(K+U+G),
                                     damping=float(D.sum()), damping_bottom=float(D[low].sum()), contact_and_error=E0-(K+U+G)-float(D.sum()),
                                     contact_normal=0., contact_tangential=0., contact_spin=0., integration_error=E0-(K+U+G)-float(D.sum()),
                                     vz_com=float(vc@gd), max_penetration_mm=1e3*pen_stats[-1][0], n_contact=pen_stats[-1][1]))
                continue
            fb = wp.to_torch(buf['f2'] if 'f2' in buf else buf['f']).double(); tb = wp.to_torch(buf['tau']).double()
            free = (wp.to_torch(wf.fixed) == 0).double()[:, None]
            vpre = vd+free*dt*(fb/m_t[:, None]+g_t); wpre = wd+free*dt*tb/I_t[:, None]
            v1_, w1_ = wp.to_torch(st1[2]).double(), wp.to_torch(st1[3]).double()
            un0, un1 = vpre@gn, v1_@gn
            Cn += .5*m_t*(un0**2-un1**2)
            Ct += .5*m_t*(((vpre-un0[:, None]*gn)**2).sum(1)-((v1_-un1[:, None]*gn)**2).sum(1))
            Cs += .5*I_t*((wpre**2).sum(1)-(w1_**2).sum(1))
            st, st1 = st1, st
            if (k+1) % every == 0:
                K, U, G, vc = total(st)
                rows.append(dict(t_frames=(k+1)/sub, kinetic=K, elastic=U, gravity=G, total=K+U+G, lost=E0-(K+U+G),
                                 damping=float(D.sum()), damping_bottom=float(D[low].sum()), contact_and_error=E0-(K+U+G)-float(D.sum()),
                                 contact_normal=float(Cn.sum()), contact_tangential=float(Ct.sum()), contact_spin=float(Cs.sum()),
                                 integration_error=E0-(K+U+G)-float(D.sum())-float((Cn+Ct+Cs).sum()),
                                 vz_com=float(vc@gd)))
        ref = rows[0]['kinetic']
        print('energies relative to the kinetic energy just after release (%.4g); positive vz = moving down' % ref, flush=True)
        for r in rows[::4]:
            print('t %5.2f fr | kin %.3f el %.3f grav %+.3f | lost %.3f = damping %.3f (bottom %.3f) + contact/err %.3f | vz %+.3f m/s' % (
                r['t_frames'], r['kinetic']/ref, r['elastic']/ref, (r['gravity']-G0)/ref, r['lost']/ref, r['damping']/ref,
                r['damping_bottom']/ref, r['contact_and_error']/ref, r['vz_com']), flush=True)
        (args.out/'energy.json').write_text(json.dumps(dict(E0=E0, K0=K0, rows=rows)))
        return
    if args.stage == 'featbatch':
        # silhouette features (feat6: cx, cy, sx, sy, sqrt area, aspect) of every frame and camera for a list of parameter
        # sets in one process (sensitivity / identifiability Jacobians). Saves featbatch.json {name: [frames][cams][6]}
        # optional per run: 'tilt' [a, b] (rad) rotates the initial body positions / orientations about the centre of mass
        # around the two horizontal axes (perpendicular to gravity); 'ground_offset' (m, + = lower) shifts the ground
        import warp as wp
        runs = json.loads(args.batch_json.read_text())
        out = {}
        gdir_ = torch.tensor(meta['gravity'], dtype=f32, device=dev); gdir_ = gdir_/gdir_.norm()
        e1_ = torch.linalg.cross(gdir_, torch.tensor([1., 0., 0.], device=dev)); e1_ = e1_/e1_.norm(); e2_ = torch.linalg.cross(gdir_, e1_)
        gp0 = wf.ground_p
        for r_ in runs:
            with torch.no_grad():
                shared.copy_(torch.tensor(r_['parameters'][:ncol], dtype=f32, device=dev).expand_as(shared))
                dv0.copy_(torch.tensor(r_.get('dv0', [0., 0., 0.]), dtype=f32, device=dev))
                dw0.copy_(torch.tensor(r_.get('dw0', [0., 0., 0.]), dtype=f32, device=dev))
                cfield.copy_(torch.tensor(r_.get('cfield', [0.]*6), dtype=f32, device=dev))
                go = float(r_.get('ground_offset', 0.))
                gn_ = torch.tensor(meta['ground_normal'], dtype=f32, device=dev)
                wf.ground_p = wp.vec3(gp0[0]-go*float(gn_[0]), gp0[1]-go*float(gn_[1]), gp0[2]-go*float(gn_[2]))
                xs, qs, vs, ws = init_state(False)
                ta, tb = r_.get('tilt', [0., 0.])
                rv = ta*e1_+tb*e2_; ang = float(rv.norm())
                if ang > 0:
                    ax_ = rv/ang
                    K_ = torch.tensor([[0., -ax_[2], ax_[1]], [ax_[2], 0., -ax_[0]], [-ax_[1], ax_[0], 0.]], device=dev)
                    Rm_ = torch.eye(3, device=dev)+np.sin(ang)*K_+(1-np.cos(ang))*K_@K_
                    xs = (com_t+(xs-com_t)@Rm_.T).contiguous()
                    qr = torch.cat((torch.tensor([np.cos(ang/2)], device=dev), ax_*np.sin(ang/2)))
                    w1, x1_, y1, z1 = qr; w2, x2, y2, z2 = qs.unbind(-1)
                    qs = torch.stack((w1*w2-x1_*x2-y1*y2-z1*z2, w1*x2+x1_*w2+y1*z2-z1*y2, w1*y2-x1_*z2+y1*w2+z1*x2, w1*z2+x1_*y2-y1*x2+z1*w2), -1).contiguous()
                    w_ = w0t+dw0
                    vs = (v0t+dv0+torch.cross(w_.expand_as(xs), xs-com_t, dim=-1)).contiguous()
                sub = substeps_now()
                pos, quat = warp_rollout(wf, theta_of(shared), (xs, qs, vs, ws), 1/fps/sub, sub, frames)
                fr, sil_ = [], []
                for f in range(frames+1):
                    row = []
                    for c in range(len(CAMS)):
                        place(pos, quat, f, c)
                        a_ = silhouette(c).detach()
                        m_ = moments_t(a_)
                        row.append(torch.cat((m_, (m_[3]/m_[2])[None])).tolist())
                        if args.save_masks:
                            sil_.append((a_.clamp(0, 1)*255).round().to(torch.uint8).cpu().numpy())
                    fr.append(row)
                if args.save_masks:
                    com_ = ((mass[None, :, None]*pos.detach()).sum(1)/mass.sum()).cpu().numpy()      # (frames + 1, 3) world centre of mass
                    np.savez_compressed(args.out/('masks_%s.npz' % r_['name']), m=np.stack(sil_).reshape(frames+1, len(CAMS), *sil_[0].shape), com=com_)
            out[r_['name']] = fr
            print('featbatch', r_['name'], 'substeps', sub, flush=True)
        wf.ground_p = gp0
        real = [[(lambda m_: torch.cat((m_, (m_[3]/m_[2])[None])).tolist())(moments_t(masks[f, c])) for c in range(len(CAMS))] for f in range(frames+1)]
        if args.save_masks:
            np.savez_compressed(args.out/'masks_real.npz', m=np.stack([np.stack([(masks[f, c].clamp(0, 1)*255).round().to(torch.uint8).cpu().numpy()
                                                                                for c in range(len(CAMS))]) for f in range(frames+1)]))
        (args.out/'featbatch.json').write_text(json.dumps(dict(runs=out, real=real, cams=CAMS)))
        return
    if args.stage == 'carve':
        # visual hull of the object in its rest frame from the free-flight frames 0..hit-1 (3 cameras x hit frames):
        # per frame and camera the rigid placement (mass-weighted Kabsch fit of the rest bodies to the simulated
        # flight, at the camera's time offset); a rest-frame grid point is kept if it projects inside the full-resolution
        # real mask in every view. Saves the grid, the per-point inside count and the views (carve.npz)
        with torch.no_grad():
            th = theta_of(shared.detach()); sub = substeps_now()
            pos, _ = warp_rollout(wf, th, init_state(False), 1/fps/sub, sub, hit)
        P = pos.double().cpu().numpy(); P0 = geometry['points']; w_ = geometry['masses']/geometry['masses'].sum()
        c0 = w_@P0
        gd = np.asarray(meta['gravity'], float); gd /= np.linalg.norm(gd)
        e1 = np.cross(gd, [1., 0, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(gd, e1)
        Bm = np.stack((e1, e2, gd), 1)                                       # rest frame -> (e1, e2, down)
        q_ = (P0-c0)@Bm; r_eq = (3*geometry['masses']/(4*np.pi))**(1/3)
        step = args.carve_step
        lo_ = q_.min(0)-[.01, .01, .01]; hi_ = q_.max(0)+[.01, .01, .015]
        axes = [np.arange(lo_[k]+step/2, hi_[k], step) for k in range(3)]
        G = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
        Xg = c0+G@Bm.T
        count = np.zeros(len(Xg), int); views = 0
        for f in range(hit):
            for c, name in enumerate(CAMS):
                t = f+offsets[c]; lo = int(np.clip(np.floor(t), 0, len(P)-2)); u = t-lo
                Y = (1-u)*P[lo]+u*P[lo+1]
                yc = w_@Y; Hm = ((P0-c0)*w_[:, None]).T@(Y-yc)
                U, _, Vt = np.linalg.svd(Hm); D_ = np.diag([1, 1, np.sign(np.linalg.det(Vt.T@U.T))])
                Rr = Vt.T@D_@U.T
                xw = yc+(Xg-c0)@Rr.T
                cam = meta['cameras'][name]; K = np.array(cam['K'], float)
                xc = xw@np.array(cam['R']).T+np.array(cam['t'])
                uv_ = xc@K.T; uu, vv = uv_[:, 0]/uv_[:, 2], uv_[:, 1]/uv_[:, 2]
                m_ = np.asarray(Image.open(dyn/'masks'/name/f'{f:02d}.png').convert('L')) > 127
                ok = (xc[:, 2] > 0) & (uu >= 0) & (uu < m_.shape[1]-1) & (vv >= 0) & (vv < m_.shape[0]-1)
                hitm = np.zeros(len(Xg), bool); hitm[ok] = m_[np.round(vv[ok]).astype(int), np.round(uu[ok]).astype(int)]
                count += hitm; views += 1
        hull = count == views
        depth = G[:, 2]
        low_body = (q_[:, 2]+r_eq).max(); low_hull = depth[hull].max()
        print('carve: %d views, grid %d points (step %.1f mm), hull %d points = %.1f cm^3; lowest hull point %.1f mm below the lowest body surface' % (
            views, len(Xg), 1e3*step, int(hull.sum()), hull.sum()*step**3*1e6, 1e3*(low_hull-low_body)), flush=True)
        print('hull volume %.1f cm^3 vs material volume %.1f cm^3' % (hull.sum()*step**3*1e6, geometry['masses'].sum()*1e6), flush=True)
        np.savez_compressed(args.out/'carve.npz', points=Xg, count=count, views=views, hull=hull, step=step, gravity=gd, frame=Bm, centre=c0)
        return
    if args.stage == 'bottom':
        # is the reconstructed bottom complete? (a) geometry: cross-section area of the union of the bodies' equivalent
        # spheres against depth above the lowest point (along gravity); (b) frames 0..hit (rigid flight): per camera, the
        # silhouette extent along the image direction of gravity (bottom / top edge, sim vs real) and the width profile
        # just above the bottom edge
        P0 = geometry['points']; r_eq = (3*geometry['masses']/(4*np.pi))**(1/3)
        gd = np.asarray(meta['gravity'], float); gd /= np.linalg.norm(gd)
        e1 = np.cross(gd, [1., 0, 0] if abs(gd[0]) < .9 else [0, 1., 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(gd, e1)
        h = P0@gd; low = h.max()
        uv = np.stack((P0@e1, P0@e2), 1); res = 5e-4
        lo_ = uv.min(0)-.01; n_ = np.ceil((uv.max(0)+.01-lo_)/res).astype(int)
        gx, gy = np.meshgrid(lo_[0]+(np.arange(n_[0])+.5)*res, lo_[1]+(np.arange(n_[1])+.5)*res, indexing='ij')
        def section(depth):
            plane = low+max(r_eq)-depth                 # depth measured from the lowest sphere surface
            rr2 = r_eq**2-(h-plane)**2; sel = rr2 > 0
            img = np.zeros(gx.shape, bool)
            for (u_, v_), q2 in zip(uv[sel], rr2[sel]):
                i0, i1 = int((u_-np.sqrt(q2)-lo_[0])/res), int((u_+np.sqrt(q2)-lo_[0])/res)+2
                j0, j1 = int((v_-np.sqrt(q2)-lo_[1])/res), int((v_+np.sqrt(q2)-lo_[1])/res)+2
                img[i0:i1, j0:j1] |= (gx[i0:i1, j0:j1]-u_)**2+(gy[i0:i1, j0:j1]-v_)**2 <= q2
            return img
        height = h.max()-h.min()+2*max(r_eq)
        depths = [1e-3, 2e-3, 3e-3, 5e-3, 8e-3, 12e-3, 20e-3, height/2]
        secs = {d_: section(d_) for d_ in depths}
        amax = max(s_.sum() for s_ in secs.values())
        geo = [dict(depth_mm=1e3*d_, area_cm2=float(secs[d_].sum()*res*res*1e4), fraction_of_max=float(secs[d_].sum()/amax)) for d_ in depths]
        print('object height along gravity %.1f mm (body centres %.1f mm + sphere radii)' % (1e3*height, 1e3*(h.max()-h.min())))
        for r in geo:
            print('cross-section %5.1f mm above the bottom: %.2f cm^2 (%.2f of the largest)' % (r['depth_mm'], r['area_cm2'], r['fraction_of_max']), flush=True)
        tiles_g = [np.pad(secs[d_].T[::-1].astype(np.uint8)*255, 2, constant_values=128) for d_ in depths]
        Image.fromarray(np.concatenate(tiles_g, 1)).resize((sum(t.shape[1] for t in tiles_g)*2, tiles_g[0].shape[0]*2), Image.NEAREST).save(args.out/'bottom_sections.png')
        with torch.no_grad():
            th = theta_of(shared.detach()); sub = substeps_now()
            pos, quat = warp_rollout(wf, th, init_state(False), 1/fps/sub, sub, max(hit, args.diag_to))
        gd_t = torch.tensor(gd, dtype=f32, device=dev)
        rows, tiles = [], []
        for c in range(len(CAMS)):
            place(pos, quat, 0, c); a0 = silhouette(c).detach().cpu().numpy() > .5
            place(pos+.01*gd_t, quat, 0, c)
            a1 = silhouette(c).detach().cpu().numpy() > .5
            c0, c1 = np.argwhere(a0).mean(0), np.argwhere(a1).mean(0)
            u = (c1-c0)/np.linalg.norm(c1-c0); v = np.array([-u[1], u[0]])          # image direction of gravity, (row, col)
            px_per_mm = np.linalg.norm(c1-c0)/10.
            row_tiles = []
            for f in range(max(hit, args.diag_to)+1):
                place(pos, quat, f, c)
                sim = silhouette(c).detach().cpu().numpy() > .5; real = masks[f, c].cpu().numpy() > .5
                def ext(m_):
                    p_ = np.argwhere(m_).astype(float); s_ = p_@u; t_ = p_@v
                    bot = s_.max(); prof = []
                    for k0 in range(0, 20, 2):                                    # width at 0-2, 2-4, ... px above the bottom
                        sel_ = (s_ <= bot-k0) & (s_ > bot-k0-2)
                        prof.append(float(np.ptp(t_[sel_])+1) if sel_.any() else 0.)
                    return bot, s_.min(), prof
                sb, st_, sp = ext(sim); rb, rt, rp = ext(real)
                iou_ = float((sim & real).sum()/max((sim | real).sum(), 1))
                rows.append(dict(frame=f, cam=CAMS[c], iou=iou_, px_per_mm=px_per_mm, bottom_shift_px=sb-rb, top_shift_px=st_-rt,
                                 height_sim_px=sb-st_, height_real_px=rb-rt, width_profile_sim=sp, width_profile_real=rp))
                print('frame %d %-6s IoU %.3f | along gravity (%.1f px/mm): bottom edge sim-real %+5.1f px, top edge %+5.1f px, height %.0f vs %.0f px | width 0-2/4-6/10-12/18-20 px above bottom: sim %s real %s' % (
                    f, CAMS[c], iou_, px_per_mm, sb-rb, st_-rt, sb-st_, rb-rt, [int(sp[k]) for k in (0, 2, 5, 9)], [int(rp[k]) for k in (0, 2, 5, 9)]), flush=True)
                if f in (0, hit, 9):
                    ov = np.zeros(sim.shape+(3,), np.uint8); ov[..., 0] = real*255; ov[..., 1] = sim*255
                    ys_, xs_ = np.nonzero(sim | real); cy_, cx_ = int(ys_.mean()), int(xs_.mean()); hh = 60
                    row_tiles.append(np.pad(ov, ((hh, hh), (hh, hh), (0, 0)))[cy_:cy_+2*hh, cx_:cx_+2*hh])
            tiles.append(np.concatenate(row_tiles, 1))
        grid = np.concatenate(tiles, 0)
        Image.fromarray(grid).resize((grid.shape[1]*3, grid.shape[0]*3), Image.NEAREST).save(args.out/'bottom_overlays.png')
        (args.out/'bottom.json').write_text(json.dumps(dict(geometry=geo, height_mm=1e3*height, frames=rows), indent=1))
        return
    if args.stage == 'debris':
        # which bodies leave the object? rollout with the --init-from parameters; per body the largest stretch of its bonds
        # over the frames; flagged bodies' connectivity, tetrahedra, neighbour-moment ratio and height
        with torch.no_grad():
            th = theta_of(shared.detach())
            sub = substeps_now()
            pos, _ = warp_rollout(wf, th, init_state(False), 1/fps/sub, sub, frames)
        P = pos.double().cpu().numpy()
        bi_, bj_ = model.i.cpu().numpy(), model.j.cpu().numpy()
        L0 = np.linalg.norm(P[0][bj_]-P[0][bi_], axis=1)
        st = np.linalg.norm(P[:, bj_]-P[:, bi_], axis=2)/L0                   # (frames+1, bonds)
        bmax = np.zeros(model.n); np.maximum.at(bmax, bi_, st.max(0)); np.maximum.at(bmax, bj_, st.max(0))
        # a body is debris if all its bonds end stretched > 2x (it left its neighbours)
        bend = np.full(model.n, np.inf); np.minimum.at(bend, bi_, st[-1]); np.minimum.at(bend, bj_, st[-1])
        deg = np.bincount(np.concatenate((bi_, bj_)), minlength=model.n)
        areas = np.asarray(geometry['areas'])
        asum = np.bincount(np.concatenate((bi_, bj_)), weights=np.concatenate((areas, areas)), minlength=model.n)
        vol = model.cell_volume.double().cpu().numpy()
        M = torch.zeros(model.n, 3, 3, dtype=torch.float64, device=dev).index_add(0, model.ls_a, (model.ls_w[:, None, None]*model.ls_dX[:, :, None]*model.ls_dX[:, None, :]).double())
        ev = torch.linalg.eigvalsh(M).cpu().numpy(); qr = ev[:, 0]/np.maximum(ev[:, 2], 1e-300)
        has = model.node_has.cpu().numpy() if getattr(model, 'node_has', None) is not None else np.ones(model.n, bool)
        ntet = np.bincount(model.tets.cpu().numpy().reshape(-1), minlength=model.n) if getattr(model, 'tets', None) is not None else np.zeros(model.n, int)
        z = P[0][:, 2]; gnd = float(model.ground_point[2])
        flag = np.flatnonzero(bend > 2.)
        print('bodies %d; debris (all bonds > 2x at the end): %d; bodies with some bond > 3x during the rollout: %d' % (
            model.n, len(flag), int((bmax > 3).sum())), flush=True)
        print('population medians: bonds %d, bond area / cell^(2/3) %.2f, tets %d, moment ratio %.3f' % (
            np.median(deg), np.median(asum/vol**(2/3)), np.median(ntet), np.median(qr)), flush=True)
        for b in flag[np.argsort(-bend[flag])][:20]:
            print('  body %5d: bonds %d, bond area / cell^(2/3) %.3f, cell volume / median %.2f, tets %d (has %s), moment ratio %.4f, '
                  'height above lowest body %.1f mm (object height %.1f mm), final min bond stretch %.1f' % (
                      b, deg[b], asum[b]/vol[b]**(2/3), vol[b]/np.median(vol), ntet[b], bool(has[b]), qr[b],
                      1e3*(z[b]-z.min()), 1e3*np.ptp(z), bend[b]), flush=True)
        np.savez_compressed(args.out/'debris.npz', bend=bend, bmax=bmax, deg=deg, ntet=ntet, qr=qr, flag=flag)
        # render side: each appearance cell blends x_k + F_k lever over its nearest bodies (F_k = least-squares fit); compare
        # with the same blend using only the rotation polar(F_k) - a large difference means the fitted F throws the cell
        from experiments.foam_deform_render import polar_rotation
        idx_b, w_b, lev = deformed.blend_idx, deformed.blend_w[..., None], deformed.blend_lever
        worst = None
        for f in range(frames+1):
            xf = to_foam(pos[f].to(dev).float())
            F = conj(model.deformation_gradient(pos[f].to(dev).float())).to(deformed.p0.dtype)
            xk = xf.to(deformed.p0.dtype)[idx_b]
            pb = (w_b*(xk+(F[idx_b]@lev[..., None])[..., 0])).sum(1)
            Rk = polar_rotation(F)
            pr = (w_b*(xk+(Rk[idx_b]@lev[..., None])[..., 0])).sum(1)
            dev_c = (pb-pr).norm(dim=-1)
            if worst is None or float(dev_c.max()) > float(worst[0].max()):
                worst = (dev_c, f, F)
        dev_c, f_w, F_w = worst
        scale = float((deformed.p0.max(0).values-deformed.p0.min(0).values).max())
        far = torch.nonzero(dev_c > .05*scale).flatten()
        print('render cells %d; worst frame %d: cells displaced by the fitted F (vs rotation only) > 5%% of the object size: %d, max %.2f x object size' % (
            len(dev_c), f_w, len(far), float(dev_c.max())/scale), flush=True)
        if len(far):
            bodies = torch.unique(idx_b[far][:, :3].flatten()).cpu().numpy()
            sv = torch.linalg.svdvals(F_w[torch.as_tensor(bodies, device=dev)]).cpu().numpy()
            print('  nearest bodies of those cells: %d; their neighbour-moment ratio median %.4f (population %.3f), bonds median %d, '
                  'height above lowest body median %.1f mm; singular values of their F at that frame: max %.2f min %.3f' % (
                      len(bodies), np.median(qr[bodies]), np.median(qr), np.median(deg[bodies]), 1e3*np.median(z[bodies]-z.min()),
                      float(sv.max()), float(sv.min())), flush=True)
            lev_far = lev[far].norm(dim=-1).mean(1).cpu().numpy()
            lev_all = lev.norm(dim=-1).mean(1).cpu().numpy()
            print('  mean lever (cell to its blend bodies) of those cells %.4f vs all cells %.4f (foam units)' % (np.median(lev_far), np.median(lev_all)), flush=True)
        return
    if args.stage == 'sweep':
        # loss surface over (E, nu) around a fitted (E, nu) run (--init-from its report.json: E_nu and the damping /
        # friction / restitution columns); training loss (frames 1..train_frames, train views) and future-frame loss
        rep_ = json.loads(args.init_from.read_text())
        E_fit, nu_fit = rep_['E_nu']['E_multiplier'], rep_['E_nu']['nu']
        th_fit = torch.tensor(rep_['best']['parameters'], dtype=f32, device=dev)
        nu0 = args.nu
        lam0, mu0 = nu0/((1+nu0)*(1-2*nu0)), 1/(2*(1+nu0))
        rows = []
        for fE in args.sweep_E:
            for nu in args.sweep_nu:
                E = E_fit*fE
                lam, mu = E*nu/((1+nu)*(1-2*nu)), E/(2*(1+nu))
                th = th_fit.clone(); th[4] = float(np.log(lam/lam0)); th[5] = float(np.log(mu/mu0))
                with torch.no_grad():
                    shared.copy_(th)
                    sub = substeps_now()
                    t_ = time.perf_counter()
                    pos, quat = warp_rollout(wf, theta_of(shared.detach()), init_state(False), 1/fps/sub, sub, frames)
                    lt, lf = 0., 0.
                    for f in range(1, frames+1):
                        for c in train_cams:
                            place(pos, quat, f, c)
                            v_ = float(((silhouette(c)-masks[f, c])**2).mean())
                            if f <= args.train_frames:
                                lt += v_
                            else:
                                lf += v_
                r = dict(E_factor=fE, E_multiplier=E, nu=nu, substeps=sub, finite=bool(torch.isfinite(pos).all()),
                         train_loss=lt/(args.train_frames*len(train_cams)), future_loss=lf/max(1, (frames-args.train_frames)*len(train_cams)),
                         seconds=time.perf_counter()-t_)
                rows.append(r); print(json.dumps(r), flush=True)
                (args.out/'sweep.json').write_text(json.dumps(dict(E_fit=E_fit, nu_fit=nu_fit, rows=rows), indent=1))
        return
    if args.stage == 'hessian':
        # lowest eigenvalues of the mass-normalised stiffness M^-1/2 H M^-1/2 (H: Hessian of the elastic energy in
        # positions and world rotation vectors, float64 torch autograd) at post-impact states of a smooth rollout;
        # a negative eigenvalue -k is a mode that grows like exp(sqrt(k) t). EAS off for bodies in the contact layer
        # as in the Warp kernel. Variants: full, EAS off, EAS + thin closure off.
        import scipy.sparse.linalg as sla
        from experiments.foam_dem_energy import quat_mul, quat_small
        with torch.no_grad():
            shared.copy_(torch.tensor(args.check_params[:ncol], dtype=f32, device=dev))
        sub = args.check_substeps or 2*substeps_now()
        per = 10
        sub = per*int(np.ceil(sub/per))
        with torch.no_grad():
            pos_d, quat_d = warp_rollout(wf, theta_of(shared.detach()), init_state(False), 1/fps/sub, sub//per, per*11)
        m64 = build(geometry, args.E, args.nu, dev, 'noslip', .25, .3, torch.float64, **kw_model)
        th64 = theta_of(shared.detach())[:, :7].double()
        eas0, thin0 = m64.eas_on.clone(), m64.thin.clone()
        mvec = torch.cat((m64.mass[:, None].expand(-1, 3), m64.inertia[:, None].expand(-1, 3)), 1).reshape(-1)
        isq = 1/torch.sqrt(mvec)
        out = []
        for t in args.hess_times:
            k = int(round(t*per))
            x = pos_d[k].double(); q = quat_d[k].double(); q = q/q.norm(dim=-1, keepdim=True)
            if not bool(torch.isfinite(x).all()):
                print('t %.1f: non-finite state' % t, flush=True); continue
            in_layer = ((x-model.ground_point.double())*model.ground_normal.double()).sum(-1)-model.contact_radius.double() < args.layer_dx
            F = m64.deformation_gradient(x); J = torch.linalg.det(F)
            for name, eas_on, thin_on, mid in (('bond-level', False, True, False), ('bond-level, two-end gradient', False, True, True)):
                m64.bond_level_mid = mid
                m64.eas_on = eas0.clone() if eas_on == 'all' else (eas0 & ~in_layer) if eas_on else torch.zeros_like(eas0)
                m64.thin = thin0.clone() if thin_on else torch.zeros_like(thin0)
                xr = x.clone().requires_grad_(True); rr = torch.zeros_like(x, requires_grad=True)
                En = m64.energy(th64, xr, quat_mul(quat_small(rr), q))
                gx, gr = torch.autograd.grad(En, (xr, rr), create_graph=True)
                gvec = torch.cat((gx, gr), 1).reshape(-1)
                count = [0]

                def mv(v):
                    count[0] += 1
                    vt = torch.as_tensor(np.asarray(v).reshape(-1), dtype=torch.float64, device=dev)*isq
                    hv = torch.autograd.grad((gvec*vt).sum(), (xr, rr), retain_graph=True)
                    return (torch.cat(hv, 1).reshape(-1)*isq).cpu().numpy()
                N6 = 6*m64.n
                op = sla.LinearOperator((N6, N6), matvec=mv, dtype=np.float64)
                t_ = time.perf_counter(); mv(np.ones(N6)); print('t %.1f %s: %.3f s per Hessian-vector product' % (t, name, time.perf_counter()-t_), flush=True)
                lo, vlo = sla.eigsh(op, k=3, which='SA', tol=1e-3, ncv=40, maxiter=600)
                hi = sla.eigsh(op, k=1, which='LA', tol=1e-3, maxiter=600, return_eigenvectors=False)
                v0_ = vlo[:, 0].reshape(m64.n, 6); w_b = (v0_**2).sum(1); top = np.argsort(-w_b)[:5]
                rot_share = float((v0_[:, 3:]**2).sum()/(v0_**2).sum())          # mass-normalised: share of the mode in rotation dofs
                bodies = [dict(body=int(b), share=float(w_b[b]), thin=bool(thin0[b]), eas=bool(eas0[b]), in_layer=bool(in_layer[b]),
                               J=float(J[b]), Fnorm=float(torch.linalg.matrix_norm(F[b]))) for b in top]
                r = dict(t=t, variant=name, lowest=lo.tolist(), highest=float(hi[0]), omega_max=float(np.sqrt(max(hi[0], 0))),
                         omega_max_dt=float(np.sqrt(max(hi[0], 0))/fps/sub), substeps=sub, growth_per_frame=float(np.exp(np.sqrt(max(-lo[0], 0))/fps)),
                         matvecs=count[0], rot_share=rot_share, top_bodies=bodies, J_min=float(J.min()), in_layer=int(in_layer.sum()))
                out.append(r); print(json.dumps(r), flush=True)
                (args.out/'hessian.json').write_text(json.dumps(out, indent=1))
        return
    if args.stage == 'gradcheck2':
        # (1) EAS / thin-closure states along the trajectory (Newton end state: clamp, residual, curvature h, barrier);
        # (2) frame-10 gradient with EAS / thin closure switched off; (3) central finite differences of the loss
        with torch.no_grad():
            shared.copy_(torch.tensor(args.check_params[:ncol], dtype=f32, device=dev))
        base_sub = args.check_substeps or substeps_now()
        out = dict()
        th = theta_of(shared.detach())
        mu_b = model.k_mu*torch.exp(th[:, 5])*model.cell_volume
        kv_b = model.k_v*torch.exp(th[:, 4])*model.cell_volume
        m = model.barrier_J
        gp, gn = model.ground_point, model.ground_normal

        def eas_stats(x):
            dx = x[model.ls_b]-x[model.ls_a]
            S_ = torch.zeros(model.n, 3, 3, dtype=x.dtype, device=x.device).index_add(0, model.ls_a, model.ls_w[:, None, None]*dx[:, :, None]*model.ls_dX[:, None, :])
            Ft = S_@model.moment_inv
            F = model.deformation_gradient(x)
            st = dict()
            if bool(model.thin.any()):
                c = torch.linalg.cross((Ft@model.thin_t1[..., None])[..., 0], (Ft@model.thin_t2[..., None])[..., 0])
                cn = torch.linalg.vector_norm(c, dim=-1)[model.thin]
                st['thin_cross_min'] = float(cn.min()); st['thin_cross_lt_0.1'] = int((cn < .1).sum())
            on = model.eas_on & (((x-gp)*gn).sum(-1)-model.contact_radius >= args.layer_dx)
            e = model.eas_dir
            a = ((F@e[..., None])[..., 0]**2).sum(-1)
            J = torch.linalg.det(F)
            on = on & (J > 0)
            sv = torch.zeros(model.n, dtype=F.dtype, device=F.device)
            for _ in range(21):
                Jt = J*torch.exp(sv); below = Jt < m
                Jc = torch.where(below, torch.full_like(Jt, m), Jt)
                L = torch.where(below, np.log(m)+(Jt-m)/m-(Jt-m)**2/(2*m*m), torch.log(Jc))
                dL = torch.where(below, 1/m-(Jt-m)/(m*m), 1/Jc)
                d2L = torch.where(below, torch.full_like(Jt, -1/(m*m)), -1/(Jc*Jc))
                g = mu_b*a*torch.exp(2*sv)+(kv_b*L-mu_b)*dL*Jt
                h = 2*mu_b*a*torch.exp(2*sv)+kv_b*(dL*Jt)**2+(kv_b*L-mu_b)*(d2L*Jt*Jt+dL*Jt)
                if _ < 20:
                    sv = torch.where(on, (sv-(g/h.clamp(min=1e-30)).clamp(-1., 1.)).clamp(-3., 3.), sv)
            scale = mu_b+kv_b
            rel_g = (g.abs()/scale)[on]; rel_h = (h/scale)[on]
            st.update(eas_active=int(on.sum()), at_clamp=int((sv.abs()[on] > 2.999).sum()), not_converged=int((rel_g > 1e-3).sum()),
                      h_small=int((rel_h < 1e-2).sum()), h_nonpos=int((rel_h <= 0).sum()), in_barrier=int((Jt[on] < m).sum()),
                      s_min=float(sv[on].min()) if bool(on.any()) else 0., s_max=float(sv[on].max()) if bool(on.any()) else 0.,
                      J_min=float(J.min()), Fnorm_max=float(torch.linalg.matrix_norm(F).max()), inverted=int((J <= 0).sum()),
                      inv_h_max=float((1/rel_h.clamp(min=1e-30)).max()) if bool(on.any()) else 0.)
            return st
        with torch.no_grad():
            sub_d = 10*int(np.ceil(base_sub/10))
            pos_d, _ = warp_rollout(wf, th, init_state(False), 1/fps/sub_d, sub_d//10, 10*args.train_frames)
        rows = []
        for k in range(len(pos_d)):
            r = dict(t_frame=k/10, **eas_stats(pos_d[k])); rows.append(r)
            if k % 5 == 0 or r['at_clamp'] or r['h_nonpos'] or r['not_converged']:
                print('state', json.dumps(r), flush=True)
        out['states'] = rows

        def run(sub, frames_used, grad=True, wfx=None):
            wfx = wfx or wf
            if grad:
                shared.grad = None
            ctx = torch.enable_grad() if grad else torch.no_grad()
            with ctx:
                pos, quat = warp_rollout(wfx, theta_of(shared if grad else shared.detach()), init_state(False), 1/fps/sub, sub, max(frames_used)+1,
                                         args.segment or None)
                loss = 0.
                for f in frames_used:
                    for c in train_cams:
                        place(pos, quat, f, c)
                        loss = loss+((silhouette(c)-masks[f, c])**2).mean()
                loss = loss/(len(frames_used)*len(train_cams))
            if not grad:
                return float(loss)
            loss.backward()
            return float(loss.detach()), shared.grad.detach().cpu().numpy().copy(), bool(torch.isfinite(pos).all())
        variants = dict(base=wf)
        for name, eas_off, thin_off in () if args.skip_toggles else (('eas_off', True, False), ('thin_off', False, True), ('eas_thin_off', True, True)):
            w_ = WarpFoam(model, contact=args.contact, layer_dx=args.layer_dx, friction0=args.friction0, restitution0=args.restitution0,
                          implicit_damping=bool(args.implicit_damping), graphs=bool(args.graphs))
            import warp as wp_
            if eas_off:
                w_.eas_on = wp_.from_torch(torch.zeros(model.n, dtype=torch.int32, device=dev))
            if thin_off:
                w_.thin = wp_.from_torch(torch.zeros(model.n, dtype=torch.int32, device=dev))
            variants[name] = w_
        out['toggles'] = []
        if args.skip_toggles:
            variants = {}
        for name, w_ in variants.items():
            for k in (1, 2):
                lv, g, fin = run(base_sub*k, [10], wfx=w_)
                r = dict(variant=name, substeps=base_sub*k, frame=10, loss=lv, finite=fin, grad=g.tolist(), grad_norm=float(np.linalg.norm(g)))
                out['toggles'].append(r); print('toggle', json.dumps({k_: v for k_, v in r.items() if k_ != 'grad'}), np.round(g[4:], 6).tolist(), flush=True)
        out['fd'] = []
        frames_full = list(range(1, args.train_frames+1))
        p0 = shared.detach().clone()
        for k in (1, 2):
            for fu, label in (([10], 'frame10'), (frames_full, 'all')):
                lv, g, _ = run(base_sub*k, fu)
                for comp in (4, 5):
                    for d in (1e-2, 1e-3):
                        vals = []
                        for sgn in (1, -1):
                            with torch.no_grad():
                                shared.copy_(p0); shared[comp] += sgn*d
                            vals.append(run(base_sub*k, fu, grad=False))
                        with torch.no_grad():
                            shared.copy_(p0)
                        r = dict(substeps=base_sub*k, loss=label, comp=comp, delta=d, fd=(vals[0]-vals[1])/(2*d), adjoint=float(g[comp]),
                                 loss_plus=vals[0], loss_minus=vals[1], loss0=lv)
                        out['fd'].append(r); print('fd', json.dumps(r), flush=True)
        (args.out/'gradcheck2.json').write_text(json.dumps(out, indent=1))
        return
    if args.stage == 'flightfit':
        # free-flight state + unloaded rest shape from the frames before the first contact (detected automatically):
        # rest shape X(sigma, k) = o + k (X_scan - sigma u_rel - o) (u_rel: self-weight sag field relative to the
        # supported bottom layer, stage 'sag'; o: bottom centre), rigidly posed (rotation / shift correction about the
        # centre of mass) and moving x(t) = c + d + exp(t [w]) Rc (X - c) + v t + g t^2 / 2. Unknowns: pose
        # correction, k, sigma, v, w, time offsets of cameras 1, 2 (camera 0 fixed). Loss: silhouette L2, all cameras.
        from experiments.flight_detect import detect
        from experiments.foam_deform_render import polar_rotation as _pr  # noqa: F401 (render path)
        md = np.stack([np.stack([masks[f, c].cpu().numpy() for c in range(len(CAMS))]) for f in range(meta['frames'])])
        det = detect(md)
        fl = det['flight_frames'] if args.flight_frames is None else list(range(args.flight_frames))
        print('free flight: frames %s (first contact frame %d, metadata hit_frame %s)' % (fl, det['contact'], meta.get('hit_frame')), flush=True)
        fz = np.load(args.rest_field)
        Rm_ = fz['R_upright']; uu_ = fz['u']@Rm_.T
        gd_ = np.asarray(meta['gravity'], float); gd_ = gd_/np.linalg.norm(gd_); up_ = -gd_
        Xs = x0.detach().double().cpu().numpy(); m_ = mass.double().cpu().numpy(); cm_ = (m_[:, None]*Xs).sum(0)/m_.sum()
        h_ = ((Xs-cm_)@Rm_.T)@up_; bot = h_ < h_.min()+4e-3
        uu_ = uu_-float((uu_[bot]@up_).mean())*up_[None, :]
        u_rel = torch.tensor(uu_@Rm_, dtype=f32, device=dev)
        Xs_t = x0.detach().clone()
        o_t = Xs_t[torch.tensor(bot, device=dev)].mean(0)
        gt_ = torch.tensor(meta['gravity'], dtype=f32, device=dev)
        P = dict(rot=torch.zeros(3, device=dev), shift=torch.zeros(3, device=dev), logk=torch.zeros((), device=dev),
                 sigma=torch.zeros((), device=dev), v=v0t.clone(), w=w0t.clone(), offs=torch.tensor(offsets[1:], dtype=f32, device=dev))

        def rest_of(P_):
            return o_t+torch.exp(P_['logk'])*(Xs_t-P_['sigma']*u_rel-o_t)

        def bodies_at(P_, t):
            Xr = rest_of(P_)
            cmr = (mass[:, None]*Xr).sum(0)/mass.sum()
            Rc = torch.matrix_exp(skew(P_['rot']))
            Rt = torch.matrix_exp(t*skew(P_['w']))
            R_ = Rt@Rc
            return cmr+P_['shift']+(Xr-cmr)@R_.T+t*P_['v']+.5*t*t*gt_, R_

        def sil_at(P_, f, c):
            t = (f+(0. if c == 0 else P_['offs'][c-1]))/fps
            xb, R_ = bodies_at(P_, t)
            Rb = conj(R_.expand(len(xb), 3, 3)@body_rotation(q0))
            deformed.apply(to_foam(xb), body_R=Rb, body_F=conj(model.deformation_gradient(xb)), mode=args.render_mode)
            return silhouette(c)
        target = {}
        if args.flight_synthetic is not None:     # recovery test: targets rendered from known values
            js = json.loads(args.flight_synthetic.read_text())
            Pt = {k__: (torch.tensor(js[k__], dtype=f32, device=dev) if k__ in js else v__.clone()) for k__, v__ in P.items()}
            with torch.no_grad():
                for f in fl:
                    for c in range(len(CAMS)):
                        target[f, c] = sil_at(Pt, f, c).detach()
            print('synthetic targets from', {k__: js[k__] for k__ in js}, flush=True)
        else:
            for f in fl:
                for c in range(len(CAMS)):
                    target[f, c] = masks[f, c]
        for v__ in P.values():
            v__.requires_grad_(True)
        lrs = dict(rot=5e-3, shift=5e-4, logk=2e-3, sigma=5e-2, v=1e-2, w=5e-2, offs=1e-2)
        opt_ = torch.optim.Adam([dict(params=[P[k__]], lr=lrs[k__]) for k__ in P])

        def total_():
            L_ = 0.
            for f in fl:
                for c in range(len(CAMS)):
                    a_ = sil_at(P, f, c)
                    npx = target[f, c].sum().clamp(min=1)
                    L_ = L_+((a_-target[f, c])**2).sum()/npx
            return L_/(len(fl)*len(CAMS))
        for it in range(args.iterations):
            opt_.zero_grad()
            L_ = total_()
            L_.backward()
            opt_.step()
            if it % 10 == 0 or it == args.iterations-1:
                print('%3d loss %.5f | sigma %.3f k %.4f rot %s shift(mm) %s v %s w %s offs %s' % (
                    it, float(L_), float(P['sigma']), float(torch.exp(P['logk'])), np.round(P['rot'].tolist(), 4),
                    np.round((P['shift']*1e3).tolist(), 2), np.round(P['v'].tolist(), 3), np.round(P['w'].tolist(), 3),
                    np.round(P['offs'].tolist(), 3)), flush=True)
        with torch.no_grad():
            ious = []
            for f in fl:
                row_ = []
                for c in range(len(CAMS)):
                    a_ = sil_at(P, f, c); ious.append(iou(a_, target[f, c])); row_.append(ious[-1])
                    ms_, mr_ = moments_t(a_), moments_t(target[f, c])
                    row_.append(float(ms_[3]/ms_[2]-mr_[3]/mr_[2]))
                print('  frame %d: IoU / aspect sim-real per camera %s' % (f, np.round(row_, 3).tolist()), flush=True)
        E_ref = float(fz['E_mult']); sig_ = float(P['sigma'])
        res = dict(flight_frames=fl, contact_frame=det['contact'], sigma=sig_, k=float(torch.exp(P['logk'])), rot=P['rot'].tolist(),
                   shift=P['shift'].tolist(), v0=P['v'].tolist(), w0=P['w'].tolist(), time_offsets_frames=dict(zip(CAMS, [0.]+P['offs'].tolist())),
                   iou_mean=float(np.mean(ious)), sag_field=str(args.rest_field), sag_E_mult=E_ref,
                   E_mult_from_sag=(E_ref/sig_ if sig_ > 1e-3 else None))
        (args.out/'flight_state.json').write_text(json.dumps(res, indent=1))
        print('flight fit: IoU %.4f, sigma %.3f -> E multiplier from the self-weight sag %s' % (
            float(np.mean(ious)), sig_, ('%.3f' % (E_ref/sig_)) if sig_ > 1e-3 else 'n/a'), flush=True)
        return
    if args.stage == 'sag':
        # self-weight sag of the scanned (rest) shape: the body turned upright (symmetry axis = largest principal
        # moment along gravity), set on the ground at rest, heavily damped dynamics until it settles; reports the
        # height / width change and saves the displacement field (body frame of the scan, rigid motion removed)
        with torch.no_grad():
            m_ = mass.double(); xs = x0.double(); cm = (m_[:, None]*xs).sum(0)/m_.sum(); dd = xs-cm
            I_ = (m_[:, None, None]*((dd*dd).sum(1)[:, None, None]*torch.eye(3, dtype=torch.float64, device=dev)-dd[:, :, None]*dd[:, None, :])).sum(0)
            ev, V_ = torch.linalg.eigh(I_); ax = V_[:, 2]
            gd = torch.tensor(meta['gravity'], dtype=torch.float64, device=dev); gd = gd/gd.norm()
            if float(ax@gd) < 0:
                ax = -ax                                          # axis pointing down (along gravity), towards the bottom
            k_ = torch.linalg.cross(ax, gd); sn = float(k_.norm()); cs = float(ax@gd)
            if sn > 1e-9:
                k_ = k_/sn; K_ = torch.tensor([[0., -k_[2], k_[1]], [k_[2], 0., -k_[0]], [-k_[1], k_[0], 0.]], dtype=torch.float64, device=dev)
                Rm = torch.eye(3, dtype=torch.float64, device=dev)+sn*K_+(1-cs)*K_@K_
            else:
                Rm = torch.eye(3, dtype=torch.float64, device=dev)
            xu = cm+dd@Rm.T                                       # upright
            gp = torch.tensor([wf.ground_p[0], wf.ground_p[1], wf.ground_p[2]], dtype=torch.float64, device=dev)
            gn = torch.tensor(meta['ground_normal'], dtype=torch.float64, device=dev)
            hgt = (xu-gp)@gn-model.contact_radius.double()
            xu = xu+float(hgt.min())*(-gn)*(-1.)                  # lowest contact sphere touching the ground
            xu = xu-float(((xu-gp)@gn-model.contact_radius.double()).min())*gn
            th = theta_of(shared.detach()).clone()
            th[:, 6] += float(np.log(args.sag_damping))                  # settle fast (statics do not depend on damping)
            if contact_col >= 0:
                th[:, contact_col] = 0.                                 # flat support (no contact-field offsets)
            qr = torch.cat((torch.tensor([np.cos(np.arccos(np.clip(cs, -1, 1))/2)], dtype=torch.float64, device=dev), k_*np.sin(np.arccos(np.clip(cs, -1, 1))/2)))
            w1, x1_, y1, z1 = qr; w2, x2, y2, z2 = q0.double().unbind(-1)
            qu = torch.stack((w1*w2-x1_*x2-y1*y2-z1*z2, w1*x2+x1_*w2+y1*z2-z1*y2, w1*y2-x1_*z2+y1*w2+z1*x2, w1*z2+x1_*y2-y1*x2+z1*w2), -1)
            sub = substeps_now()
            nfr = args.sag_frames
            pos, quat = warp_rollout(wf, th, (xu.float().contiguous(), qu.float().contiguous(), torch.zeros_like(x0), torch.zeros_like(x0)),
                                     1/fps/sub, sub, nfr)
            xe = pos[-1].double()
            vlast = float((pos[-1]-pos[-2]).norm(dim=-1).max())*fps
            # displacement in the upright frame with the rigid part (translation, rotation) removed (Kabsch)
            a_ = xu-(m_[:, None]*xu).sum(0)/m_.sum(); b_ = xe-(m_[:, None]*xe).sum(0)/m_.sum()
            U, S_, Vh = torch.linalg.svd((m_[:, None]*b_).T@a_)
            Rk = U@Vh
            if torch.det(Rk) < 0:
                U[:, -1] *= -1; Rk = U@Vh
            u_ = b_@Rk-a_                                         # sag displacement, upright frame
            up = -gd
            hgt_r = float((a_@up).max()-(a_@up).min()); hgt_e = float((b_@Rk@up).max()-(b_@Rk@up).min())
            hor = a_-(a_@up)[:, None]*up; hor_e = (b_@Rk)-((b_@Rk)@up)[:, None]*up
            wid_r = float(2*hor.norm(dim=-1).max()); wid_e = float(2*hor_e.norm(dim=-1).max())
            top = (a_@up) > (a_@up).max()-3e-3
            print('sag (E x%.3f, nu %.3f, alpha %.2f; %d frames, max speed at the end %.2e m/s): height %.2f -> %.2f mm (%.2f %%), '
                  'width %.2f -> %.2f mm (%+.2f %%), top surface moves %.2f mm down' % (
                      float(torch.exp(shared[4])), .5*float(torch.sigmoid(z0+shared[5])) if 'z0' in dir() else float('nan'), float(args.ogden_alpha*torch.exp(shared[ogden_col])) if ogden_col >= 0 else float('nan'),
                      nfr, vlast, hgt_r*1e3, hgt_e*1e3, 100*(hgt_e-hgt_r)/hgt_r, wid_r*1e3, wid_e*1e3, 100*(wid_e-wid_r)/wid_r, -float((u_[top]@up).mean())*1e3), flush=True)
            # back to the scan frame: rest shape of the scan = upright rest; displacement field rotated back
            u_scan = u_@Rm                                        # (Rm^T applied to row vectors' inverse) scan-frame displacement
            np.savez(args.out/'sag_field.npz', u=u_scan.cpu().numpy(), R_upright=Rm.cpu().numpy(), E_mult=float(torch.exp(shared[4])),
                     mu_log=float(theta_of(shared.detach())[:, 5].mean()), height_mm=hgt_r*1e3, height_loaded_mm=hgt_e*1e3)
        return
    if args.stage == 'finesweep':
        # static foam, all cell centres and texel sites translated along world x by t (fixed adjacency): silhouette
        # centroid cx at fine steps, to see whether the forward is smooth below a pixel (steps vs local slope)
        with torch.no_grad():
            deformed.apply(to_foam(x0.detach()), body_R=conj(body_rotation(q0)), body_F=conj(model.deformation_gradient(x0.detach())), mode=args.render_mode)
            f_ = foam
            P0 = f_.points.detach().clone(); r0_ = f_.get_radii().detach().clone(); n0_ = f_.get_normals().detach().clone()
            tg_, bt_ = f_.get_tangents(); off_ = f_.texel_sites*r0_[:, None, None]
            off_ = off_[..., 0:1]*tg_[:, None, :]+off_[..., 1:2]*bt_[:, None, :]
            ts0 = (P0[:, None, :]+off_).detach().clone()
            th0 = (f_.texel_height*r0_[:, None]).detach().clone(); d0_ = f_.get_density().detach().clone()
            a_s, a_v, a_t = f_.get_att_sv()
            rgb0 = f_.sv.forward(ts0.view(-1, 3), cams[0], a_s, a_v, a_t).view(P0.shape[0], f_.args.num_texel_sites, 3).detach()
            adj_, adjo_ = f_.adjacency.clone(), f_.adjacency_offsets.clone()
            cmw = ((mass[:, None]*x0).sum(0)/mass.sum())
            u_ = (to_foam(cmw+1e-3*torch.tensor([1., 0., 0.], device=dev))-to_foam(cmw))       # 1 mm along world x
        qd = .5*torch.ones(H, W, 1, device=dev)
        c = train_cams[0]
        def cxof(t_):
            out = f_.rasterizer.forward(cams[c], qd, P0+t_*u_, r0_, d0_, n0_, ts0+t_*u_, rgb0, th0, adj_, adjo_, None, False)
            return moments_t(out[1].reshape(H, W))[0]
        ts = np.linspace(-.25, .25, 51)                                                    # mm (~0.06 px per step)
        with torch.no_grad():
            cx_ = np.array([float(cxof(float(t_))) for t_ in ts])
        tt = torch.zeros((), device=dev, requires_grad=True)
        g0, = torch.autograd.grad(cxof(tt), tt)
        sl = np.diff(cx_)/np.diff(ts)
        print('%s fine sweep of cx over +-0.25 mm (51 points): overall slope %.4f px/mm, autodiff at 0 %.4f' % (CAMS[c], np.polyfit(ts, cx_, 1)[0], float(g0)), flush=True)
        print('  local slopes between neighbouring points: %s' % np.round(sl, 3).tolist(), flush=True)
        print('  residual of cx about the line (px): %s' % np.round(cx_-np.polyval(np.polyfit(ts, cx_, 1), ts), 4).tolist(), flush=True)
        return
    if args.stage == 'rasterref':
        # float64 PyTorch reference of the rasterizer's per-ray integral for a set of silhouette-edge pixels (sphere
        # clip, power faces of the adjacency, the cell's surface plane with the texel-height offset, optical depth
        # sigma dt): alpha and d(sum w_p alpha_p)/d(cell centres) against the Warp rasterizer, per camera
        with torch.no_grad():
            deformed.apply(to_foam(x0.detach()), body_R=conj(body_rotation(q0)), body_F=conj(model.deformation_gradient(x0.detach())), mode=args.render_mode)
            f_ = foam
            P0 = f_.points.detach().clone(); r0_ = f_.get_radii().detach().clone(); n0_ = f_.get_normals().detach().clone()
            tg_, bt_ = f_.get_tangents(); off_ = f_.texel_sites*r0_[:, None, None]
            off_ = off_[..., 0:1]*tg_[:, None, :]+off_[..., 1:2]*bt_[:, None, :]
            ts0 = (P0[:, None, :]+off_).detach().clone()
            th0 = (f_.texel_height*r0_[:, None]).detach().clone(); d0_ = f_.get_density().detach().clone()
            a_s, a_v, a_t = f_.get_att_sv()
            rgb0 = f_.sv.forward(ts0.view(-1, 3), cams[0], a_s, a_v, a_t).view(P0.shape[0], f_.args.num_texel_sites, 3).detach()
            adj_, adjo_ = f_.adjacency.clone(), f_.adjacency_offsets.clone()
        qd = .5*torch.ones(H, W, 1, device=dev)
        D_ = torch.float64
        nadj = (adjo_[1:]-adjo_[:-1]).long()
        owner_ = torch.repeat_interleave(torch.arange(len(P0), device=dev), nadj)

        info_ = {}

        def ref_alpha(P, o, d, drop=()):
            # P (N, 3) float64 with grad; o (3,), d (3,) unit; drop: terms whose dependence on the centres is cut
            r = r0_.to(D_); c = P
            cd = c.detach()
            oc = o-(cd if 'sphere' in drop else c); b = (oc*d).sum(-1); qc = (oc*oc).sum(-1)-r*r; disc = b*b-qc
            hit = disc >= 0
            sq = torch.sqrt(disc.clamp(min=0))
            tfar = -b+sq; tnear = -b-sq
            hit = hit & (tfar >= 0)
            tnear = torch.where(tnear < 0, torch.zeros_like(tnear), tnear)
            ids = torch.nonzero(hit)[:, 0]
            if len(ids) == 0:
                return torch.zeros((), dtype=D_, device=dev)
            tn = tnear[ids]; tf = tfar[ids]
            # faces of the hit cells
            sel = hit[owner_]
            ei = owner_[sel]; ej = adj_.long()[sel]
            cj = cd[ej] if 'face_adj' in drop else c[ej]
            ci = cd[ei] if 'face_own' in drop else c[ei]
            if 'fp16' in drop:            # Warp's forward: neighbour = centre + float16(neighbour - centre), radius float16
                dlt = (cd[ej]-cd[ei]).float()
                cj = cj+(dlt.half().to(D_)-dlt.to(D_)).detach()+((cd[ej]-cd[ei]).float().to(D_)-(cd[ej]-cd[ei])).detach()
            nrm = cj-ci
            rj_ = r[ej].float().half().to(D_) if 'fp16' in drop else r[ej]
            offs = .5*((cj*cj).sum(-1)-(ci*ci).sum(-1)+r[ei]**2-rj_**2)
            dp = (nrm*d).sum(-1); t = (offs-(nrm*o).sum(-1))/dp
            pos = torch.full((len(P0),), -1, dtype=torch.long, device=dev); pos[ids] = torch.arange(len(ids), device=dev)
            k = pos[ei]
            big = torch.full((len(ids),), 1e30, dtype=D_, device=dev)
            tf = torch.minimum(tf, big.scatter_reduce(0, k[dp >= 0], t[dp >= 0], 'amin', include_self=True))
            tn = torch.maximum(tn, (-big).scatter_reduce(0, k[dp < 0], t[dp < 0], 'amax', include_self=True))
            # surface plane with the texel-height offset (plane_intersection_fwd)
            cc = cd[ids] if 'plane' in drop else c[ids]; nn = n0_[ids].to(D_); rr = r[ids]
            pdp = (nn*d).sum(-1)
            ts_ = ts0[ids].to(D_)-P0[ids].to(D_)[:, None, :]+cc[:, None, :]             # texel sites move with the centre
            _ts = ((cc-o)*nn).sum(-1)/pdp
            _tq = torch.where(pdp >= 0, tn, torch.maximum(tn, _ts))
            if 'tquery' in drop:
                _tq = _tq.detach()
            pt = o+_tq[:, None]*d
            wgt = torch.exp(-10.*((pt[:, None, :]-ts_)**2).sum(-1)/rr[:, None]**2)
            hgt = (wgt*th0[ids].to(D_)).sum(-1)/wgt.sum(-1).clamp(min=1e-20)
            if 'height' in drop:
                hgt = hgt.detach()
            tsurf = (((cc-o)*nn).sum(-1)+hgt)/pdp
            if 'info' in drop:
                fs_ = big.scatter_reduce(0, k[dp >= 0], t[dp >= 0], 'amin', include_self=True)
                ns_ = (-big).scatter_reduce(0, k[dp < 0], t[dp < 0], 'amax', include_self=True)
                info_['ids'] = ids; info_['near'] = torch.where(pdp < 0, torch.where(tsurf > tn, 2, torch.where(ns_ > tnear[ids], 1, 0)),
                                                                torch.where(ns_ > tnear[ids], 1, 0))
            tf_pre = tf
            far_plane = (pdp >= 0) & (tsurf < tf)
            tf = torch.where(pdp >= 0, torch.minimum(tsurf, tf), tf)
            tn = torch.where(pdp < 0, torch.maximum(tsurf, tn), tn)
            dt = tf-tn
            od = (d0_[ids].to(D_)*dt*(dt > 0)).sum()
            if 'info' in drop:
                info_['pdp'] = pdp.detach(); info_['disc'] = disc[ids].detach(); info_['b'] = b[ids].detach()
                info_['far'] = torch.where(far_plane, 2, torch.where(fs_ < tfar[ids], 1, 0)); info_['dt'] = dt.detach(); info_['sigma'] = d0_[ids]
            if 'double_far' in drop:      # emulate: dL/dt_far also applied to the pre-plane far boundary when the plane clips it
                od = od+(d0_[ids].to(D_)*(tf_pre-tf_pre.detach())*(far_plane & (dt > 0))).sum()
            return 1-torch.exp(-od)
        for c in train_cams:
            with torch.no_grad():
                a_full = f_.rasterizer.forward(cams[c], qd, P0, r0_, d0_, n0_, ts0, rgb0, th0, adj_, adjo_, None, False)[1].reshape(H, W)
            band = {'edge': (.05, .95), 'inner': (.95, .99999), 'outer': (1e-5, .05), 'all': (1e-6, 2.)}[args.ref_band]
            edge = torch.nonzero((a_full > band[0]) & (a_full < band[1]))
            gen = torch.Generator(device=dev).manual_seed(2)
            pick = edge[torch.randperm(len(edge), generator=gen, device=dev)[:40]] if args.ref_band != 'all' else edge
            print('pixel band %s: alpha in (%g, %g), %d candidates' % (args.ref_band, band[0], band[1], len(edge)), flush=True)
            wpx = torch.randn(len(pick), device=dev, generator=gen)
            if args.ref_band == 'all':      # first-order change of the silhouette centroid cx: sum_p (j - cx) alpha_p / A
                A_ = a_full.sum(); cx_ = (a_full*torch.arange(W, device=dev)[None, :]).sum()/A_
                wpx = (pick[:, 1].float()-cx_)/A_
            Wm = torch.zeros(H, W, device=dev); Wm[pick[:, 0], pick[:, 1]] = wpx
            Pw = P0.clone().requires_grad_(True)
            tsw = (ts0-P0[:, None, :]).detach()+Pw[:, None, :]                          # texel sites follow the centres
            aw = f_.rasterizer.forward(cams[c], qd, Pw, r0_, d0_, n0_, tsw, rgb0, th0, adj_, adjo_, None, False)[1].reshape(H, W)
            gw, = torch.autograd.grad((aw*Wm).sum(), Pw)
            rays = cams[c].ray_maps.to(dev)
            Pr = P0.to(D_).clone().requires_grad_(True)
            Lr = 0.; ar = []
            for (i_, j_), w_ in zip(pick.tolist(), wpx.tolist()):
                o = rays[i_, j_, :3].to(D_); d = rays[i_, j_, 3:].to(D_); d = d/d.norm()
                a_ = ref_alpha(Pr, o, d); ar.append(float(a_)); Lr = Lr+w_*a_
            gr, = torch.autograd.grad(Lr, Pr)
            gw64_ = gw.double()
            if args.ref_band == 'all':
                for k_ in range(3):
                    e_ = torch.zeros(3, dtype=D_, device=dev); e_[k_] = 1.
                    print('    d cx / d(translate foam axis %d): warp %+.5e  reference %+.5e' % (k_, float((gw64_*e_).sum()), float((gr*e_).sum())), flush=True)
                aw_ = a_full[pick[:, 0], pick[:, 1]].double()
                ar_ = torch.tensor(ar, dtype=D_, device=dev)
                print('    alpha warp - reference over all pixels: max %.4f, sum %.3f' % (float((aw_-ar_).abs().max()), float((aw_-ar_).sum())), flush=True)
                continue
            for drop in (('fp16',),):
                Pr2 = P0.to(D_).clone().requires_grad_(True); L2 = 0.
                for (i_, j_), w_ in zip(pick.tolist(), wpx.tolist()):
                    o = rays[i_, j_, :3].to(D_); d = rays[i_, j_, 3:].to(D_); d = d/d.norm()
                    L2 = L2+w_*ref_alpha(Pr2, o, d, drop)
                g2, = torch.autograd.grad(L2, Pr2)
                print('    reference without %-18s: vs warp relative error %.3f cosine %.4f | vs full reference relative change %.3f' % (
                    '+'.join(drop), float((gw64_-g2).norm()/g2.norm()), float((gw64_*g2).sum()/(gw64_.norm()*g2.norm())),
                    float((g2-gr).norm()/gr.norm())), flush=True)
            errs = []
            for q_, (i_, j_) in enumerate(pick.tolist()):
                W1 = torch.zeros(H, W, device=dev); W1[i_, j_] = 1.
                Pw1 = P0.clone().requires_grad_(True)
                ts1 = (ts0-P0[:, None, :]).detach()+Pw1[:, None, :]
                a1 = f_.rasterizer.forward(cams[c], qd, Pw1, r0_, d0_, n0_, ts1, rgb0, th0, adj_, adjo_, None, False)[1].reshape(H, W)
                g1, = torch.autograd.grad((a1*W1).sum(), Pw1)
                o = rays[i_, j_, :3].to(D_); d = rays[i_, j_, 3:].to(D_); d = d/d.norm()
                Pr1 = P0.to(D_).clone().requires_grad_(True)
                gref, = torch.autograd.grad(ref_alpha(Pr1, o, d, ('info',)), Pr1)
                e1 = float((g1.double()-gref).norm()/gref.norm().clamp(min=1e-30))
                errs.append((e1, q_, i_, j_, g1.double(), gref, {k__: v__.clone() for k__, v__ in info_.items()}))
            errs.sort(key=lambda z: -z[0])
            print('    per pixel relative error: median %.3f, worst %s' % (
                float(np.median([z[0] for z in errs])), ' '.join('%.2f' % z[0] for z in errs[:6])), flush=True)
            for e1, q_, i_, j_, g1, gref, inf in errs[:1]:
                o = rays[i_, j_, :3].to(D_); d = rays[i_, j_, 3:].to(D_); d = d/d.norm()
                ids = inf['ids']; b__ = int(ids[torch.argmax((g1[ids]-gref[ids]).norm(dim=-1))])
                hh = 2e-3*float(r0_[b__])
                print('    cell %d: radius %.3e, material %s, neighbours %d, texel sites %d' % (b__, float(r0_[b__]), b__ < len(cells['source_index']),
                      int(nadj[b__]), ts0.shape[1]), flush=True)
                for k__ in range(3):
                    fw, fr = [], []
                    for sg in (1, -1):
                        Pp = P0.clone(); Pp[b__, k__] += sg*hh
                        tsp = (ts0-P0[:, None, :])+Pp[:, None, :]
                        with torch.no_grad():
                            fw.append(float(f_.rasterizer.forward(cams[c], qd, Pp, r0_, d0_, n0_, tsp, rgb0, th0, adj_, adjo_, None, False)[1].reshape(H, W)[i_, j_]))
                        Pq = P0.to(D_).clone(); Pq[b__, k__] += sg*hh
                        with torch.no_grad():
                            fr.append(float(ref_alpha(Pq, o, d)))
                    print('      axis %d: warp autodiff %+.4e  warp central diff %+.4e | reference autodiff %+.4e  reference central diff %+.4e' % (
                        k__, float(g1[b__, k__]), (fw[0]-fw[1])/(2*hh), float(gref[b__, k__]), (fr[0]-fr[1])/(2*hh)), flush=True)
                for drop in (('double_far',), ('double_far', 'height'), ('double_far', 'plane'), ('double_far', 'face_adj'),
                             ('double_far', 'face_own'), ('double_far', 'sphere'), ('double_far', 'tquery')):
                    Pr1 = P0.to(D_).clone().requires_grad_(True)
                    gv, = torch.autograd.grad(ref_alpha(Pr1, o, d, drop), Pr1)
                    print('    pixel (%d, %d) cell %d: reference %-22s grad %s | warp %s' % (i_, j_, b__, '+'.join(drop),
                          np.round(gv[b__].cpu().numpy(), 3).tolist(), np.round(g1[b__].cpu().numpy(), 3).tolist()), flush=True)
            for e1, q_, i_, j_, g1, gref, inf in errs[:2]:
                ids = inf['ids']
                dg = (g1[ids]-gref[ids]).norm(dim=-1)
                order = torch.argsort(-dg)[:6]
                print('    pixel (%d, %d) error %.3f; worst cells: [id, near (0 sphere 1 face 2 plane), far, dt, sigma, |warp|, |ref|, |diff|, '
                      'plane n.d, sphere disc / b^2]' % (i_, j_, e1), flush=True)
                for k__ in order.tolist():
                    b__ = int(ids[k__])
                    print('       %6d %d %d %+.4e %.1f  %.3e %.3e %.3e  %+.2e %.2e' % (b__, int(inf['near'][k__]), int(inf['far'][k__]), float(inf['dt'][k__]),
                          float(inf['sigma'][k__]), float(g1[b__].norm()), float(gref[b__].norm()), float(dg[k__]), float(inf['pdp'][k__]),
                          float(inf['disc'][k__]/inf['b'][k__]**2)), flush=True)
                other = (g1-gref).norm(dim=-1); other[ids] = 0
                print('       gradient outside the ray\'s sphere-hit cells: warp-ref norm %.3e (cells %d)' % (float(other.norm()), int((other > 1e-9).sum())), flush=True)
            awp = a_full[pick[:, 0], pick[:, 1]].double().cpu().numpy(); ar = np.array(ar)
            gw64 = gw.double(); nz = gr.norm(dim=-1) > 1e-6*gr.norm(dim=-1).max()
            cos = float((gw64*gr).sum()/(gw64.norm()*gr.norm()))
            print('%s: %d edge pixels | alpha warp vs reference: max |diff| %.4f, mean %.4f | grad wrt centres: |warp| / |ref| %.3f, '
                  'cosine %.4f, relative error %.3f, cells with grad %d' % (
                      CAMS[c], len(pick), float(np.abs(awp-ar).max()), float(np.abs(awp-ar).mean()), float(gw64.norm()/gr.norm()), cos,
                      float((gw64-gr).norm()/gr.norm()), int(nz.sum())), flush=True)
            # rigid translation of everything (texel sites included) along foam axes
            for k_ in range(3):
                e_ = torch.zeros(3, dtype=D_, device=dev); e_[k_] = 1.
                print('    d/d(translate foam axis %d): warp %+.5e  reference %+.5e' % (k_, float((gw64*e_).sum()), float((gr*e_).sum())), flush=True)
        return
    if args.stage == 'rendercheck4':
        # rasterizer inputs one at a time (static foam, rest pose): silhouette centroid (cx, cy) derivative along a
        # perturbation of only the cell centres, only the texel sites, both (rigid), radii, normals, texel heights,
        # density; autodiff (one pass per output) vs central differences
        with torch.no_grad():
            deformed.apply(to_foam(x0.detach()), body_R=conj(body_rotation(q0)), body_F=conj(model.deformation_gradient(x0.detach())), mode=args.render_mode)
            f_ = foam
            P0 = f_.points.detach().clone(); r0_ = f_.get_radii().detach().clone(); n0_ = f_.get_normals().detach().clone()
            tg_, bt_ = f_.get_tangents(); off_ = f_.texel_sites*r0_[:, None, None]
            off_ = off_[..., 0:1]*tg_[:, None, :]+off_[..., 1:2]*bt_[:, None, :]
            ts0 = (P0[:, None, :]+off_).detach().clone()
            th0 = (f_.texel_height*r0_[:, None]).detach().clone(); d0_ = f_.get_density().detach().clone()
            a_s, a_v, a_t = f_.get_att_sv()
            rgb0 = f_.sv.forward(ts0.view(-1, 3), cams[0], a_s, a_v, a_t).view(P0.shape[0], f_.args.num_texel_sites, 3).detach()
            adj = (f_.adjacency.clone(), f_.adjacency_offsets.clone())
            cmw = ((mass[:, None]*x0).sum(0)/mass.sum())
            ufoam = [(to_foam(cmw+1e-3*e_)-to_foam(cmw)) for e_ in torch.eye(3, device=dev)]   # 1 mm along world x, y, z
        if args.raster_nocoop:
            import copy as _copy
            from powerfoam.rasterize import Rasterizer
            a2 = _copy.copy(f_.args); a2.disable_coop_prim_load = True; a2.disable_coop_adj_load = True
            f_.rasterizer = Rasterizer(a2, f_.device, f_.attr_dtype)
            print('rasterizer: cooperative loads disabled', flush=True)
        qd = .5*torch.ones(H, W, 1, device=dev)
        gen = torch.Generator(device=dev).manual_seed(1)
        axis_ = torch.nn.functional.normalize(torch.randn(3, device=dev, generator=gen), dim=0)

        def cen(c, P_, ts_, r_, n_, th_, d_):
            out = f_.rasterizer.forward(cams[c], qd, P_, r_, d_, n_, ts_, rgb0, th_, adj[0], adj[1], None, False)
            return moments_t(out[1].reshape(H, W))
        nm__ = len(cells['source_index'])
        print('cells %d: material %d (density median %.2f), air shell %d (density median %.2f)' % (
            len(d0_), nm__, float(d0_[:nm__].median()), len(d0_)-nm__, float(d0_[nm__:].median())), flush=True)
        for c in train_cams:
            base = dict(P_=P0, ts_=ts0, r_=r0_, n_=n0_, th_=th0, d_=d0_)
            dirs = {}
            for k_, ax_ in enumerate('xyz'):
                dirs['centres %s' % ax_] = dict(P_=ufoam[k_].expand_as(P0))
                dirs['texel sites %s' % ax_] = dict(ts_=ufoam[k_].expand_as(ts0))
                dirs['rigid %s' % ax_] = dict(P_=ufoam[k_].expand_as(P0), ts_=ufoam[k_].expand_as(ts0))
            solid = torch.zeros(len(d0_), 1, device=dev); solid[:len(cells['source_index'])] = 1.     # material cells first, then the air shell
            for k_, ax_ in enumerate('xyz'):
                dirs['material centres %s' % ax_] = dict(P_=ufoam[k_].expand_as(P0)*solid)
                dirs['shell centres %s' % ax_] = dict(P_=ufoam[k_].expand_as(P0)*(1-solid))
            dirs['radii x1.01'] = dict(r_=.01*r0_)
            dirs['normals rot 0.01'] = dict(n_=.01*torch.linalg.cross(axis_.expand_as(n0_), n0_))
            dirs['heights x1.1'] = dict(th_=.1*th0)
            dirs['density x1.1'] = dict(d_=.1*d0_)
            for nm_, dv in dirs.items():
                fd = []
                for h in (1., .5):
                    with torch.no_grad():
                        pp = cen(c, **{k__: base[k__]+h*dv.get(k__, 0.) for k__ in base})
                        mm = cen(c, **{k__: base[k__]-h*dv.get(k__, 0.) for k__ in base})
                    fd.append(((pp[:2]-mm[:2])/(2*h)).cpu().numpy())
                ad = []
                for j in range(2):
                    leaves = {k__: base[k__].clone().requires_grad_(k__ in dv) for k__ in base}
                    g_ = torch.autograd.grad(cen(c, **leaves)[j], [leaves[k__] for k__ in dv])
                    ad.append(float(sum((gg*dv[k__]).sum() for gg, k__ in zip(g_, dv))))
                print('%s %-18s d(cx, cy): autodiff %+.4f %+.4f | central diff %+.4f %+.4f (h) %+.4f %+.4f (h/2)' % (
                    CAMS[c], nm_, ad[0], ad[1], fd[0][0], fd[0][1], fd[1][0], fd[1][1]), flush=True)
        return
    if args.stage == 'rendercheck3':
        # rasterizer only: the static foam (rest bodies, frame 0 pose) rigidly translated by t along world x / y / z.
        # Sweep t over +-2 px: silhouette centroid vs the exact projected motion (slope and jitter about a line), with the
        # adjacency rebuilt every render (as in training) or kept from t = 0; autodiff d centroid / dt at t = 0.
        xs0 = x0.detach().clone()
        cm = ((mass[:, None]*xs0).sum(0)/mass.sum()).double().cpu().numpy()
        R_id = conj(body_rotation(q0))
        for c in train_cams:
            cam_ = meta['cameras'][CAMS[c]]; K_ = np.array(cam_['K'], float)/args.downsample; K_[2, 2] = 1.
            R_c = np.array(cam_['R']); t_c = np.array(cam_['t'])
            pr = lambda x_: (lambda u_: u_[:2]/u_[2])(K_@(R_c@x_+t_c))
            Jp = np.stack([(pr(cm+1e-5*e_)-pr(cm-1e-5*e_))/2e-5 for e_ in np.eye(3)], 1)*1e-3     # px per mm
            mm_px = 1./float(np.abs(Jp).max())                                                   # mm per ~1 px
            for k_, ax_ in enumerate('xyz'):
                e_ = torch.zeros(3, device=dev); e_[k_] = 1e-3                                  # 1 mm
                def cen(t_, adj=None):
                    xx = xs0+t_*e_
                    deformed.apply(to_foam(xx), body_R=R_id, body_F=conj(model.deformation_gradient(xx)), mode=args.render_mode)
                    if adj is not None:
                        foam.adjacency, foam.adjacency_offsets = adj
                    return moments_t(silhouette(c))
                with torch.no_grad():
                    cen(0.)
                    adj0 = (foam.adjacency.clone(), foam.adjacency_offsets.clone())
                ts = np.linspace(-2, 2, 21)*mm_px
                res = {}
                for nm_, adj in (('rebuilt', None), ('fixed', adj0)):
                    with torch.no_grad():
                        cc = np.array([cen(float(t_), adj)[:2].double().cpu().numpy() for t_ in ts])
                    fit = [np.polyfit(ts, cc[:, j], 1) for j in range(2)]
                    jit = [float(np.std(cc[:, j]-np.polyval(fit[j], ts))) for j in range(2)]
                    res[nm_] = (fit[0][0], fit[1][0], jit)
                tt = torch.zeros((), device=dev, requires_grad=True)
                ax, = torch.autograd.grad(cen(tt)[0], tt)
                tt = torch.zeros((), device=dev, requires_grad=True)
                ay, = torch.autograd.grad(cen(tt)[1], tt)
                print('%s translate %s (px per mm): exact %+.3f %+.3f | autodiff %+.3f %+.3f | sweep slope, adjacency rebuilt %+.3f %+.3f '
                      '(jitter %.3f %.3f px), fixed %+.3f %+.3f (jitter %.3f %.3f px)' % (
                          CAMS[c], ax_, Jp[0, k_], Jp[1, k_], float(ax), float(ay), res['rebuilt'][0], res['rebuilt'][1], *res['rebuilt'][2],
                          res['fixed'][0], res['fixed'][1], *res['fixed'][2]), flush=True)
        return
    if args.stage == 'rendercheck2':
        # render-only gradient check at pixel scale. (a) Along the loss gradient itself (largest derivative): body
        # displacement scaled to 1 px rms, central differences at 0.25 .. 2 px vs autodiff. (b) A known answer: rigid
        # translation of all bodies by t along world x / y / z; the silhouette centroid must move by the camera's
        # projected displacement (perspective Jacobian at the centre of mass), compared with autodiff and differences.
        sub = substeps_now()
        with torch.no_grad():
            pos, quat = warp_rollout(wf, theta_of(shared.detach()), init_state(False), 1/fps/sub, sub, args.train_frames+1)
        for f in (5, 9, 14):
            xf, qf = pos[f].clone(), quat[f].clone()
            cm = ((mass[:, None]*xf).sum(0)/mass.sum()).double().cpu().numpy()
            for c in train_cams:
                cam_ = meta['cameras'][CAMS[c]]; K_ = np.array(cam_['K'], float)/args.downsample; K_[2, 2] = 1.
                R_c = np.array(cam_['R']); t_c = np.array(cam_['t'])
                pr = lambda x_: (lambda u_: u_[:2]/u_[2])(K_@(R_c@x_+t_c))
                Jp = np.stack([(pr(cm+1e-5*e_)-pr(cm-1e-5*e_))/2e-5 for e_ in np.eye(3)], 1)     # px per m
                mm_px = 1e-3/float(np.abs(Jp[:, :2]).max())                                    # metres per ~1 px

                def render_x(x_):
                    R_ = conj(body_rotation(qf)); F_ = conj(model.deformation_gradient(x_))
                    deformed.apply(to_foam(x_), body_R=R_, body_F=F_, mode=args.render_mode)
                    return silhouette(c)
                lossf = lambda x_: ((render_x(x_)-masks[f, c])**2).mean()
                xg = xf.clone().requires_grad_(True)
                gx, = torch.autograd.grad(lossf(xg), xg)
                d = gx/gx.norm(dim=-1).pow(2).mean().sqrt()*mm_px                   # rms displacement ~1 px
                ad = float((gx*d).sum())
                fds = []
                for h in (.25, .5, 1., 2.):
                    with torch.no_grad():
                        fds.append((float(lossf(xf+h*d))-float(lossf(xf-h*d)))/(2*h))
                print('frame %2d %s (a) along the gradient, 1 px rms: autodiff %.4e | central diff at 0.25/0.5/1/2 px %s | ratios %s' % (
                    f, CAMS[c], ad, ' '.join('%.4e' % v_ for v_ in fds), ' '.join('%.3f' % (ad/v_) for v_ in fds)), flush=True)
                for k_, ax_ in enumerate('xyz'):
                    e_ = torch.zeros(3, device=dev); e_[k_] = 1.
                    tt = torch.zeros((), device=dev, requires_grad=True)
                    gcx, = torch.autograd.grad(moments_t(render_x(xf+tt*e_))[0], tt)     # one pass per output: the
                    tt = torch.zeros((), device=dev, requires_grad=True)                # rasterizer's backward is single-use
                    gcy, = torch.autograd.grad(moments_t(render_x(xf+tt*e_))[1], tt)
                    fdx, fdy = [], []
                    for hpx in (.5, 1., 2.):
                        h = hpx*mm_px
                        with torch.no_grad():
                            mp, mn = moments_t(render_x(xf+h*e_)), moments_t(render_x(xf-h*e_))
                        fdx.append(float(mp[0]-mn[0])/(2*h)*1e-3); fdy.append(float(mp[1]-mn[1])/(2*h)*1e-3)
                    print('    translate %s: d centroid / d t (px per mm): expected (projection) %+.3f %+.3f | autodiff %+.3f %+.3f | '
                          'central diff at 0.5/1/2 px  x %s  y %s' % (ax_, Jp[0, k_]*1e-3, Jp[1, k_]*1e-3, float(gcx)*1e-3, float(gcy)*1e-3,
                                                                     ' '.join('%+.3f' % v_ for v_ in fdx), ' '.join('%+.3f' % v_ for v_ in fdy)), flush=True)
        return
    if args.stage == 'rendercheck':
        # silhouette loss of one frame / camera as a function of the body positions only (no simulation): autodiff
        # vs central differences along rigid and deformation directions about the centre of mass
        sub = substeps_now()
        with torch.no_grad():
            pos, quat = warp_rollout(wf, theta_of(shared.detach()), init_state(False), 1/fps/sub, sub, args.train_frames+1)
        gen = torch.Generator(device=dev).manual_seed(0)
        for f in (5, 8, 9, 10, 14):
            xf, qf = pos[f].clone(), quat[f].clone()
            cm = (mass[:, None]*xf).sum(0)/mass.sum()
            r = xf-cm
            ez = torch.tensor([0., 0., 1.], device=dev)
            dirs = dict(tx=torch.tensor([1., 0., 0.], device=dev).expand_as(r)*1e-4,
                        tz=ez.expand_as(r)*1e-4,
                        scale=r*1e-3,
                        squash=torch.stack((r[:, 0], r[:, 1], -2*r[:, 2]), -1)*1e-3,
                        rot_y=torch.linalg.cross(torch.tensor([0., 1., 0.], device=dev).expand_as(r), r)*1e-3,
                        shear_xz=torch.stack((r[:, 2], 0*r[:, 1], 0*r[:, 0]), -1)*1e-3,
                        random=torch.sin(r@torch.randn(3, 3, device=dev, generator=gen)*60.)*1e-4)
            for c in train_cams:
                def loss_x(x_):
                    R_ = conj(body_rotation(qf)); F_ = conj(model.deformation_gradient(x_))
                    deformed.apply(to_foam(x_), body_R=R_, body_F=F_, mode=args.render_mode)
                    return ((silhouette(c)-masks[f, c])**2).mean()
                xg = xf.clone().requires_grad_(True)
                gx, = torch.autograd.grad(loss_x(xg), xg)
                out = []
                for nm, dv in dirs.items():
                    ad = float((gx*dv).sum())
                    fds = []
                    for h in (1., .25):
                        with torch.no_grad():
                            fds.append((float(loss_x(xf+h*dv))-float(loss_x(xf-h*dv)))/(2*h))
                    out.append('%s %.3f/%.3f' % (nm, ad/fds[0] if fds[0] else np.nan, ad/fds[1] if fds[1] else np.nan))
                print('frame %2d %s: autodiff / central diff (h, h/4): %s' % (f, CAMS[c], '  '.join(out)), flush=True)
        return
    if args.stage == 'dampcheck':
        # few-step adjoint check of the in-place damping (hand-written adjoint) against Warp's autodiff of the
        # out-of-place kernels: random relative velocities (so the dashpots act), random output adjoints
        import warp as wp_
        from experiments.foam_dem_warp import _pad_state, _state_dtypes, _new_state
        sub = substeps_now(); dt_ = 1/fps/sub
        M = 3
        gen = torch.Generator(device=dev).manual_seed(1)
        st0 = list(_pad_state(wf, init_state(False)))
        st0[2] = st0[2]+.1*torch.randn(st0[2].shape, device=dev, generator=gen)
        st0[3] = st0[3]+5.*torch.randn(st0[3].shape, device=dev, generator=gen)
        seeds = [torch.randn(t_.shape, device=dev, generator=gen) for t_ in st0[:4]]
        th_t = theta_of(shared.detach()).float().contiguous()
        res = {}
        for tag_, pre_, inpl_ in (('old', False, False), ('new', True, True)):
            wf.damp_pre, wf.damp_inplace = pre_, inpl_
            s0 = tuple(wp_.from_torch(t_.detach().float().contiguous().clone(), dtype=d_, requires_grad=True) for t_, d_ in zip(st0, _state_dtypes(wf)))
            th_ = wp_.from_torch(th_t.clone(), requires_grad=True)
            pool = [s0]+[_new_state(wf, True) for _ in range(M)]
            bufs = [wf._buffers(True) for _ in range(M)]
            tape = wp_.Tape()
            with tape:
                for j in range(M):
                    wf.step(th_, pool[j], pool[j+1], bufs[j], dt_)
            for a_, g_ in zip(pool[M][:4], seeds):
                wp_.to_torch(a_.grad).copy_(g_)
            tape.backward()
            res[tag_] = dict(out_x=wp_.to_torch(pool[M][0]).clone(), out_v=wp_.to_torch(pool[M][2]).clone(),
                             g_x=wp_.to_torch(s0[0].grad).clone(), g_q=wp_.to_torch(s0[1].grad).clone(), g_v=wp_.to_torch(s0[2].grad).clone(),
                             g_w=wp_.to_torch(s0[3].grad).clone(), g_theta=wp_.to_torch(th_.grad).clone())
        for k_ in res['old']:
            a_, b_ = res['old'][k_], res['new'][k_]
            print('dampcheck %d steps %-8s max|new-old| %.3e  max|old| %.3e  relative %.2e' % (
                M, k_, float((b_-a_).abs().max()), float(a_.abs().max()), float((b_-a_).abs().max()/a_.abs().max().clamp(min=1e-30))), flush=True)
        wf.damp_pre, wf.damp_inplace = True, False
        return
    if args.stage == 'simcheck':
        # bit-level regression check of the simulator: positions of a plain and a differentiable rollout and every
        # gradient of a fixed functional, saved for comparison between code versions
        sub = substeps_now()
        nfr = args.check_frames or 10
        gen = torch.Generator(device=dev).manual_seed(0)
        wpos = torch.randn(nfr, model.n, 3, device=dev, generator=gen)
        if getattr(wf, 'implicit_damping', False):      # in-process A/B of the damping kernels (old, new, new again)
            res_ab = {}
            for tag_, pre_, inpl_ in (('old', False, False), ('pre', True, False), ('new', True, True), ('new2', True, True)):
                wf.damp_pre, wf.damp_inplace = pre_, inpl_
                for t_ in (shared, dv0, dw0, cfield):
                    t_.grad = None
                with torch.no_grad():
                    pa_, _ = warp_rollout(wf, theta_of(shared), init_state(False), 1/fps/sub, sub, nfr+1, args.segment or None)
                pg_, _ = warp_rollout(wf, theta_of(shared), init_state(True), 1/fps/sub, sub, nfr+1, args.segment or None)
                ((pg_[1:nfr+1]-pg_[:1])*wpos).sum().backward()
                res_ab[tag_] = dict(pos=pa_.cpu().numpy(), g_shared=shared.grad.cpu().numpy(),
                                    g_cfield=cfield.grad.cpu().numpy() if cfield.grad is not None else np.zeros(6),
                                    **(dict(g_dv0=dv0.grad.cpu().numpy(), g_dw0=dw0.grad.cpu().numpy()) if args.learn_init else {}))
            for k_ in res_ab['old']:
                a_, p_, b_, c_ = res_ab['old'][k_], res_ab['pre'][k_], res_ab['new'][k_], res_ab['new2'][k_]
                print('damping A/B %-9s in-place vs old %.3e | vs pre-only %.3e | in-place vs itself %.3e | pre-only vs old %.3e  (max |old| %.3e)' % (
                    k_, np.abs(b_-a_).max(), np.abs(b_-p_).max(), np.abs(b_-c_).max(), np.abs(p_-a_).max(), np.abs(a_).max()), flush=True)
            wf.damp_pre, wf.damp_inplace = True, False
        with torch.no_grad():
            p0, q0_ = warp_rollout(wf, theta_of(shared), init_state(False), 1/fps/sub, sub, nfr+1, args.segment or None)
        for t_ in (shared, dv0, dw0, cfield):
            t_.grad = None
        p1, q1_ = warp_rollout(wf, theta_of(shared), init_state(True), 1/fps/sub, sub, nfr+1, args.segment or None)
        ((p1[1:nfr+1]-p1[:1])*wpos).sum().backward()
        out_ = dict(pos_plain=p0.cpu().numpy(), quat_plain=q0_.cpu().numpy(), pos_grad=p1.detach().cpu().numpy(),
                    g_shared=shared.grad.cpu().numpy(), g_cfield=cfield.grad.cpu().numpy() if cfield.grad is not None else np.zeros(6))
        if args.learn_init:
            out_.update(g_dv0=dv0.grad.cpu().numpy(), g_dw0=dw0.grad.cpu().numpy())
        np.savez(args.out/'simcheck.npz', **out_)
        print('simcheck saved', args.out/'simcheck.npz', {k_: float(np.abs(v_).max()) for k_, v_ in out_.items()}, flush=True)
        return
    if args.stage == 'kprofile':
        # per-kernel GPU time of the simulation step (Warp CUDA-event timing; nsys has no kernel trace under WSL)
        import warp as wp_
        from experiments.foam_dem_warp import _pad_state, _state_dtypes, _new_state, _frame_graph
        try:
            from experiments.foam_dem_warp import _zero_sets as _zs
        except ImportError:
            def _zs(sets):
                for d_ in sets:
                    for a_ in d_.values():
                        a_.zero_()
        sub = substeps_now(); dt_ = 1/fps/sub
        th_ = wp_.from_torch(theta_of(shared.detach()).float().contiguous())

        def fresh(grad=False):
            return tuple(wp_.from_torch(t_.detach().float().contiguous().clone(), dtype=d_, requires_grad=grad)
                         for t_, d_ in zip(_pad_state(wf, init_state(False)), _state_dtypes(wf)))

        def summarise(res, title, nsteps):
            agg = {}
            for r_ in res:
                k_ = r_.name.split('(')[0][:60]
                a_ = agg.setdefault(k_, [0., 0])
                a_[0] += r_.elapsed; a_[1] += 1
            tot = sum(a_[0] for a_ in agg.values())
            print('%s: GPU time %.3f ms per step (%d steps, %d timed ops)' % (title, tot/nsteps, nsteps, sum(a_[1] for a_ in agg.values())), flush=True)
            for k_, (t_, c_) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:18]:
                print('   %6.1f%%  %8.4f ms/step  calls/step %5.1f  avg %7.1f us  %s' % (100*t_/tot, t_/nsteps, c_/nsteps, 1e3*t_/c_, k_), flush=True)
        print('model: bodies %d, bonds %d, tets %d, substeps/frame %d' % (wf.n, wf.nb, getattr(wf, 'nt', 0), sub), flush=True)
        N = 200
        s_, o_ = fresh(), _new_state(wf); buf_ = wf._buffers(False)
        for _ in range(20):                                   # warm-up (module load)
            _zs([buf_])
            wf.step(th_, s_, o_, buf_, dt_); s_, o_ = o_, s_
        wp_.synchronize()
        t0 = time.perf_counter()
        for _ in range(N):
            _zs([buf_])
            wf.step(th_, s_, o_, buf_, dt_); s_, o_ = o_, s_
        wp_.synchronize()
        print('forward, plain launches: wall %.3f ms per step' % ((time.perf_counter()-t0)*1e3/N), flush=True)
        g_, nx_, sp_ = _frame_graph(wf, th_, s_, o_, buf_, dt_, N)
        wp_.capture_launch(g_); wp_.synchronize()
        t0 = time.perf_counter()
        for _ in range(3):
            wp_.capture_launch(g_)
        wp_.synchronize()
        print('forward, CUDA graph: wall %.3f ms per step' % ((time.perf_counter()-t0)*1e3/(3*N)), flush=True)
        # cost split inside CUDA graphs: full step, step without the implicit damping, a chain of empty kernels
        def graph_ms(fn, reps=3):
            with wp_.ScopedCapture() as cap_:
                for _ in range(N):
                    fn()
            wp_.capture_launch(cap_.graph); wp_.synchronize()
            t_ = time.perf_counter()
            for _ in range(reps):
                wp_.capture_launch(cap_.graph)
            wp_.synchronize()
            return (time.perf_counter()-t_)*1e3/(reps*N)
        st_ = [s_, o_]

        def one_step():
            _zs([buf_]); wf.step(th_, st_[0], st_[1], buf_, dt_); st_[0], st_[1] = st_[1], st_[0]
        t_full = graph_ms(one_step)
        keep_d = wf.implicit_damping
        wf.implicit_damping = False
        t_nod = graph_ms(one_step)
        wf.implicit_damping = keep_d
        noop_out = wp_.zeros(wf.n, dtype=float, device='cuda')

        @wp_.kernel
        def _noop(a: wp_.array(dtype=float)):
            i = wp_.tid()
            a[i] = a[i]+1.0
        t_noop = graph_ms(lambda: [wp_.launch(_noop, wf.n, inputs=[noop_out]) for _ in range(wf.n_colors)])
        print('CUDA graph, forward per step: full %.3f ms, without implicit damping %.3f ms -> damping %.3f ms (%d colours); '
              '%d empty kernels %.3f ms' % (t_full, t_nod, t_full-t_nod, wf.n_colors, wf.n_colors, t_noop), flush=True)
        wp_.timing_begin(cuda_filter=wp_.TIMING_ALL)
        for _ in range(N):
            _zs([buf_])
            wf.step(th_, s_, o_, buf_, dt_); s_, o_ = o_, s_
        summarise(wp_.timing_end(), 'forward (kernels + memsets)', N)
        # backward of M taped steps
        M = 25
        pool = [fresh(True)]+[_new_state(wf, True) for _ in range(M)]
        bufs = [wf._buffers(True) for _ in range(M)]
        th_.requires_grad = True
        th_.grad = wp_.zeros_like(th_)
        tape = wp_.Tape()
        with tape:
            for j in range(M):
                wf.step(th_, pool[j], pool[j+1], bufs[j], dt_)
        wp_.to_torch(pool[M][0].grad).fill_(1.)
        wp_.synchronize()
        t0 = time.perf_counter(); tape.backward(); wp_.synchronize()
        print('backward, plain launches: wall %.3f ms per step' % ((time.perf_counter()-t0)*1e3/M), flush=True)
        def bwd_graph_ms(damp):
            wf.implicit_damping = damp
            tp_ = wp_.Tape()
            with tp_:
                for j in range(M):
                    wf.step(th_, pool[j], pool[j+1], bufs[j], dt_)
            with wp_.ScopedCapture() as cap_:
                tp_.backward()
            wf.implicit_damping = keep_d
            wp_.capture_launch(cap_.graph); wp_.synchronize()
            t_ = time.perf_counter()
            for _ in range(3):
                wp_.capture_launch(cap_.graph)
            wp_.synchronize()
            return (time.perf_counter()-t_)*1e3/(3*M)
        b_full, b_nod = bwd_graph_ms(keep_d), bwd_graph_ms(False)
        print('CUDA graph, adjoint per step: full %.3f ms, without implicit damping %.3f ms -> damping adjoint %.3f ms' % (
            b_full, b_nod, b_full-b_nod), flush=True)
        tape.zero(); wp_.to_torch(pool[M][0].grad).fill_(1.)
        wp_.timing_begin(cuda_filter=wp_.TIMING_ALL)
        tape.backward()
        summarise(wp_.timing_end(), 'backward (adjoint kernels + memsets)', M)
        return
    if args.stage == 'gradall':
        # autodiff vs central differences for every learned scalar (shared material columns, initial velocity / spin
        # corrections, contact-field c1 c2) at the --init-from point; loss on positions (no rendering) or silhouettes
        if args.check_cfield is not None:
            with torch.no_grad():
                cfield.copy_(torch.tensor((args.check_cfield+[0.]*6)[:6], dtype=f32, device=dev))
        sub = substeps_now()
        nfr = args.check_frames or args.train_frames
        gen = torch.Generator(device=dev).manual_seed(0)
        wpos = torch.randn(nfr, model.n, 3, device=dev, generator=gen)/np.sqrt(nfr*model.n)
        wmom = torch.randn(nfr, 12, device=dev, generator=gen)/np.sqrt(nfr*12)

        def loss_of(grad, fw_=None):
            pos, quat = warp_rollout(wf, theta_of(shared), init_state(grad), 1/fps/sub, sub, nfr+1, args.segment or None)
            if args.check_loss == 'pos':
                return ((pos[1:nfr+1]-pos[:1])*wpos).sum()*1e3              # mm-weighted displacement functional
            if args.check_loss == 'mom':                                       # mass-weighted centre and second moments per frame
                mw = (mass/mass.sum()).to(pos.dtype)
                P = pos[1:nfr+1].double(); mw = mw.double()                  # float64 sums: small differences survive
                c_ = (mw[None, :, None]*P).sum(1)                             # (frames, 3)
                d_ = P-c_[:, None]
                M2 = torch.einsum('b,fbi,fbj->fij', mw, d_, d_)               # (frames, 3, 3)
                feats = torch.cat((c_*1e3, M2.reshape(nfr, 9)*1e6), 1)        # mm, mm^2
                if fw_ is not None:                                            # frame weights (frames 1..nfr)
                    feats = feats*fw_[:, None]
                return (feats*wmom.double()).sum()

            l_ = 0.
            for f in range(1, nfr+1):
                for c in train_cams:
                    place(pos, quat, f, c)
                    l_ = l_+((silhouette(c)-masks[f, c])**2).mean()
            return l_/(nfr*len(train_cams))
        if args.check_cut is not None:
            assert args.check_loss == 'mom'
            K_ = args.check_cut
            tens_ = dict(dv0=dv0, dw0=dw0)
            if args.learn_pose:
                tens_.update(rot0=rot0, dx0=dx0)
            res_ = {}
            for mode_ in ('adjoint', 'weighted', 'plain'):
                for t_ in (shared, dv0, dw0, cfield, rot0, dx0, offs_t, sig_t, dtilt):
                    t_.grad = None
                if mode_ == 'adjoint':
                    wf.adj_decay = [1. if f < K_ else 0. for f in range(nfr+2)]; wf.adj2_stash = True
                    loss_of(True).backward()
                    cut_init_grads()
                    wf.adj_decay = None; wf.adj2_stash = False
                elif mode_ == 'weighted':
                    fw_ = torch.tensor([1. if f < K_ else 0. for f in range(1, nfr+1)], dtype=torch.float64, device=dev)
                    loss_of(True, fw_).backward()
                else:
                    loss_of(True).backward()
                res_[mode_] = {k_: t_.grad.detach().cpu().numpy().copy() for k_, t_ in tens_.items()}
            for k_ in res_['adjoint']:
                ref_ = 'weighted' if k_ in ('dv0', 'dx0') or args.init_cut == 2 or (args.init_cut == 3 and k_ == 'rot0') else 'plain'
                a_, b_ = res_['adjoint'][k_], res_[ref_][k_]
                print('  cut check K=%d %-5s two-adjoint %s | %s loss %s | max rel diff %.2e' % (
                    K_, k_, np.array2string(a_, precision=5), ref_, np.array2string(b_, precision=5),
                    float(np.abs(a_-b_).max()/max(np.abs(b_).max(), 1e-30))), flush=True)
            return
        if args.check_decay is not None:
            assert args.check_loss == 'mom'
            r_ = args.check_decay
            tens_ = dict(dv0=dv0, dw0=dw0)
            if args.learn_pose:
                tens_.update(rot0=rot0, dx0=dx0)
            res_ = {}
            for mode_ in ('adjoint', 'weighted'):
                for t_ in (shared, dv0, dw0, cfield, rot0, dx0, offs_t, sig_t, dtilt):
                    t_.grad = None
                if mode_ == 'adjoint':
                    wf.adj_decay = r_
                    loss_of(True).backward()
                    wf.adj_decay = None
                    c0_ = 1e-3*float(wf.g_theta_decay[:, contact_col].sum()) if contact_col >= 0 else 0.
                else:
                    fw_ = torch.tensor([r_**f for f in range(1, nfr+1)], dtype=torch.float64, device=dev)
                    loss_of(True, fw_).backward()
                    c0_ = float(cfield.grad[0]) if contact_col >= 0 else 0.
                res_[mode_] = {k_: t_.grad.detach().cpu().numpy().copy() for k_, t_ in tens_.items()}
                res_[mode_]['c0'] = np.array([c0_])
            for k_ in res_['adjoint']:
                a_, b_ = res_['adjoint'][k_], res_['weighted'][k_]
                print('  decay check r=%.3f %-5s discounted adjoint %s | weighted loss %s | max rel diff %.2e' % (
                    r_, k_, np.array2string(a_, precision=5), np.array2string(b_, precision=5),
                    float(np.abs(a_-b_).max()/max(np.abs(b_).max(), 1e-30))), flush=True)
            return
        for t_ in (shared, dv0, dw0, cfield, rot0, dx0, offs_t, sig_t, dtilt):
            t_.grad = None
        loss_of(True).backward()
        names, tens, idx, steps = [], [], [], []
        for k, nm in ((4, 'logE'), (5, 'nu_z'), (6, 'damping'), (7, 'friction'), (ogden_col, 'alpha')):
            if k >= 0:
                names.append(nm); tens.append(shared); idx.append(k); steps.append(.02)
        for k in range(3):
            names.append('dv0_%s' % 'xyz'[k]); tens.append(dv0); idx.append(k); steps.append(2e-3)
        for k in range(3):
            names.append('dw0_%s' % 'xyz'[k]); tens.append(dw0); idx.append(k); steps.append(.05)
        if contact_col >= 0:
            for k in ((0, 1, 2) if not args.cfield_fix0 else (1, 2)):
                names.append('c%d' % k); tens.append(cfield); idx.append(k); steps.append(.1)
        if args.learn_pose:
            for k in range(3):
                names.append('rot0_%s' % 'xyz'[k]); tens.append(rot0); idx.append(k); steps.append(5e-3)
            for k in range(3):
                names.append('dx0_%s' % 'xyz'[k]); tens.append(dx0); idx.append(k); steps.append(.2)
        if args.learn_offsets:
            for k in range(len(offs_t)):
                names.append('offs_%d' % (k+1)); tens.append(offs_t); idx.append(k); steps.append(.02)
        if args.learn_rest == 'free':
            names.append('rest_s'); tens.append(sig_t); idx.append(0); steps.append(.05)
        if args.tilt_relative:
            for k in range(2):
                names.append('dtilt_%d' % (k+1)); tens.append(dtilt); idx.append(k); steps.append(.1)
        print('gradall: loss %s, %d frames, %d substeps' % (args.check_loss, nfr, sub), flush=True)
        rows = []
        for nm, t_, k, h0 in zip(names, tens, idx, steps):
            if args.check_only and nm not in args.check_only:
                continue
            ga = float(t_.grad.reshape(-1)[k]) if t_.dim() == 1 else float(t_.grad[..., k].sum())
            if args.check_scan:
                vals = []
                for o_ in args.check_scan:
                    with torch.no_grad():
                        t_.view(-1)[k] += o_
                        vals.append(float(loss_of(False)))
                        t_.view(-1)[k] -= o_
                    print('  scan %-9s offset %+.4f loss %+.9e' % (nm, o_, vals[-1]), flush=True)
                rows.append(dict(name=nm, autodiff=ga, scan=list(zip(args.check_scan, vals))))
                print('  scan %-9s autodiff slope at 0: %+.6e' % (nm, ga), flush=True)
                continue
            fds = []
            hs_ = [m_*h0*args.check_scale for m_ in args.check_multi] if args.check_multi else \
                ((h0*args.check_scale,) if args.check_single else (h0*args.check_scale, .25*h0*args.check_scale))
            for h in hs_:
                vals = []
                for sgn in (1, -1):
                    with torch.no_grad():
                        t_.view(-1)[k] += sgn*h
                        vals.append(float(loss_of(False)))
                        t_.view(-1)[k] -= sgn*h
                fds.append((vals[0]-vals[1])/(2*h))
            rows.append(dict(name=nm, autodiff=ga, fd=fds))
            print('  %-9s autodiff %+.6e | central diff %s | ratio %s' % (nm, ga, ' '.join('%+.6e' % v_ for v_ in fds),
                  ' '.join('%.4f' % (ga/v_ if v_ else np.nan) for v_ in fds)), flush=True)
        (args.out/('gradall_%s_%d.json' % (args.check_loss, nfr))).write_text(json.dumps(rows, indent=1))
        return
    if args.stage == 'cfieldcheck':
        # autodiff gradient of a silhouette loss w.r.t. the contact-field coefficients against central differences, at
        # --check-cfield (e.g. offsets below -R, where the old reach clamp cut the gradient)
        with torch.no_grad():
            cfield.copy_(torch.tensor((args.check_cfield+[0.]*6)[:6], dtype=f32, device=dev))
        sub = substeps_now()
        nfr = min(args.train_frames, 14)

        def loss_of():
            pos, quat = warp_rollout(wf, theta_of(shared), init_state(False), 1/fps/sub, sub, nfr+1, args.segment or None)
            l_ = 0.
            for f in range(1, nfr+1):
                for c in train_cams:
                    place(pos, quat, f, c)
                    l_ = l_+((silhouette(c)-masks[f, c])**2).mean()
            return l_/(nfr*len(train_cams))
        cfield.grad = None
        l0 = loss_of()
        l0.backward()
        with torch.no_grad():
            l0n = loss_of()
        print('loss on the taped (checkpointed) path %.9e, on the no-grad path %.9e' % (float(l0), float(l0n)), flush=True)
        ga = cfield.grad.detach().cpu().numpy().copy()
        for k in range(3):
            for h in (.02, .1, .3):
                vals = []
                for sgn in (1, -1):
                    with torch.no_grad():
                        cfield[k] += sgn*h
                    vals.append(float(loss_of()))     # same (taped) rollout path as the autodiff evaluation
                    with torch.no_grad():
                        cfield[k] -= sgn*h
                fd = (vals[0]-vals[1])/(2*h)
                print('c%d step %.2f mm: autodiff %.6e  central difference %.6e  ratio %.4f' % (k, h, ga[k], fd, ga[k]/fd if fd else np.nan), flush=True)
        return
    if args.stage == 'gradcheck':
        # where does a gradient spike come from? same parameters, (a) substeps x1/x2/x4, (b) wider contact layer,
        # (c) loss restricted to one frame at a time (which frame's adjoint carries the spike)
        with torch.no_grad():
            shared.copy_(torch.tensor(args.check_params[:ncol], dtype=f32, device=dev))
        base_sub = args.check_substeps or substeps_now()

        def grad_of(sub, frames_used):
            shared.grad = None
            pos, quat = warp_rollout(wf, theta_of(shared), init_state(False), 1/fps/sub, sub, max(frames_used)+1, args.segment or None)
            finite = bool(torch.isfinite(pos).all())
            loss = 0.
            for f in frames_used:
                for c in train_cams:
                    place(pos, quat, f, c)
                    loss = loss+((silhouette(c)-masks[f, c])**2).mean()
            loss = loss/(len(frames_used)*len(train_cams))
            loss.backward()
            g = shared.grad.detach().cpu().numpy()
            return dict(substeps=sub, frames=list(frames_used), finite_forward=finite, loss=float(loss.detach()),
                        grad_norm=float(np.linalg.norm(g)), grad=g.tolist())
        rows = []
        all_frames = list(range(1, args.train_frames+1))
        for k in (1, 2, 4):
            rows.append(dict(test='substeps x%d' % k, **grad_of(base_sub*k, all_frames)))
            print(json.dumps({k_: v for k_, v in rows[-1].items() if k_ != 'grad'}), np.round(rows[-1]['grad'], 4).tolist(), flush=True)
        wf_saved = wf
        for dx in (2*args.layer_dx, 4*args.layer_dx):
            wf = WarpFoam(model, contact=args.contact, layer_dx=dx, friction0=args.friction0, restitution0=args.restitution0,
                          implicit_damping=bool(args.implicit_damping), graphs=bool(args.graphs))
            rows.append(dict(test='layer_dx %.4f' % dx, **grad_of(base_sub, all_frames)))
            print(json.dumps({k_: v for k_, v in rows[-1].items() if k_ != 'grad'}), np.round(rows[-1]['grad'], 4).tolist(), flush=True)
        wf = wf_saved
        for f in all_frames:
            for k in (1, 2):
                rows.append(dict(test='frame %d substeps x%d' % (f, k), **grad_of(base_sub*k, [f])))
                print(json.dumps({k_: v for k_, v in rows[-1].items() if k_ not in ('grad', 'frames')}), flush=True)
        (args.out/'gradcheck.json').write_text(json.dumps(rows, indent=1))
        return
    substeps = substeps_now()
    report = dict(config={k_: str(v) for k_, v in vars(args).items()}, velocity=dict(v0=vel['v0'], w0=vel['w0']), time_offsets=offsets,
                  initial=evaluate(substeps))
    print('initial', json.dumps({k_: v for k_, v in report['initial'].items() if k_ != 'iou_per_frame_cam'}), flush=True)
    def _vp_res(p_, a_, t_):
        """Variable-projection residual of a damped cosine a ~ c0 + e^(-k t) (c1 cos 2 pi t / T + c2 sin 2 pi t / T), p_ = (T, k):
        the linear coefficients are solved in closed form, so the residual is a smooth function of (T, k) and the data."""
        env = torch.exp(-p_[1]*t_)
        B = torch.stack((torch.ones_like(t_), env*torch.cos(2*np.pi*t_/p_[0]), env*torch.sin(2*np.pi*t_/p_[0])), -1)
        coef = torch.linalg.solve(B.T@B+1e-9*torch.eye(3, device=B.device, dtype=B.dtype), B.T@a_)
        return ((B@coef-a_)**2).sum()

    def fit_period(a_):
        """(T, k) of the best damped-cosine fit: grid arg-min, then damped Newton on the residual (no gradient), then one
        implicit-function step dp/da = -H^-1 d(grad_p res)/da at the optimum (exact first-order derivative, no softmax
        temperature: the estimate is a continuous function of the data). k at its lower bound 0: T only."""
        dt_ = a_.dtype
        a64 = a_.double()
        t_ = torch.arange(len(a_), device=a_.device, dtype=torch.float64)
        with torch.no_grad():
            Tg = torch.arange(3., 10.01, .05, device=a_.device, dtype=torch.float64)
            kg = torch.arange(0., .61, .04, device=a_.device, dtype=torch.float64)
            TT, KK = torch.meshgrid(Tg, kg, indexing='ij'); TT, KK = TT.reshape(-1), KK.reshape(-1)
            env = torch.exp(-KK[:, None]*t_[None])
            B = torch.stack((torch.ones_like(env), env*torch.cos(2*np.pi*t_[None]/TT[:, None]), env*torch.sin(2*np.pi*t_[None]/TT[:, None])), -1)
            coef = torch.linalg.solve(B.transpose(1, 2)@B+1e-9*torch.eye(3, device=a_.device, dtype=torch.float64), B.transpose(1, 2)@a64.detach()[None, :, None])
            res = ((B@coef)[..., 0]-a64.detach()[None]).pow(2).sum(-1)
            i_ = int(torch.argmin(res))
            p = torch.stack((TT[i_], KK[i_]))
        lo = torch.tensor([3., 0.], device=a_.device, dtype=torch.float64); hi = torch.tensor([10., .6], device=a_.device, dtype=torch.float64)

        def grad_hess(p0_, a0_, free_):
            q_ = p0_.clone().requires_grad_(True)
            r_ = _vp_res(q_, a0_, t_)
            g_, = torch.autograd.grad(r_, q_, create_graph=True)
            H_ = torch.stack([torch.autograd.grad(g_[j], q_, retain_graph=True)[0] for j in range(2)])
            return r_, g_[free_], H_[free_][:, free_], q_
        lam = 1e-3
        a0 = a64.detach()
        for _ in range(30):
            free = [0, 1] if float(p[1]) > 1e-6 else [0]
            r0, g0, H0, _ = grad_hess(p, a0, free)
            r0, g0, H0 = r0.detach(), g0.detach(), H0.detach()
            if float(g0.norm()) < 1e-12:
                break
            for _ in range(20):
                step = torch.linalg.solve(H0+lam*torch.eye(len(free), device=a_.device, dtype=torch.float64)*(1+H0.diagonal().abs().max()), g0)
                pn = p.clone(); pn[free] = pn[free]-step
                pn = torch.maximum(torch.minimum(pn, hi), lo)
                with torch.no_grad():
                    rn = _vp_res(pn, a0, t_)
                if float(rn) <= float(r0):
                    p, lam = pn, max(lam/3, 1e-9)
                    break
                lam *= 3
            else:
                break
        free = [0, 1] if float(p[1]) > 1e-6 else [0]
        _, g1, H1, q1 = grad_hess(p, a64, free)            # g1 depends on the data a_ (graph kept)
        H1 = H1.detach()
        if bool(torch.linalg.eigvalsh(0.5*(H1+H1.T)).min() > 0):
            dp = torch.linalg.solve(H1, g1)
            pf = torch.zeros(2, device=a_.device, dtype=torch.float64)+p.detach()
            pf = pf.index_put((torch.tensor(free, device=a_.device),), p.detach()[free]-dp)
        else:                                                # not a strict minimum: no gradient through the fit
            pf = p.detach()
        return pf[0].to(dt_), pf[1].to(dt_)

    def phys_obs(series, k_imp):
        """Physical observables of one camera's feature series (frames 1..N, feat6): oscillation period and decay of the
        aspect ratio after the impact (frames k_imp+2..N; damped cosine fitted continuously, fit_period: grid start, Newton
        refinement, implicit-function gradient), squash amplitude 1 - soft-min aspect, and the stiffness ratio
        (period / amplitude)^2 (impact stiffness / oscillation stiffness for a linear oscillator hit at the same speed)."""
        asp = series[:, 5]
        period, decay = fit_period(asp[k_imp+1:])              # series index 0 is frame 1
        amp = 1-(-.01*torch.logsumexp(-asp/.01, 0))                 # 1 - soft-min aspect
        return torch.stack((period, decay.clamp(min=1e-3), amp.clamp(min=1e-3), (period/amp.clamp(min=1e-3))**2))

    def feat6(a_):
        m_ = moments_t(a_)
        return torch.cat((m_, (m_[3]/m_[2])[None]))           # cx, cy, sx, sy, sqrt(area), aspect sy / sx
    feat_scale, k_imp, real_phys = {}, {}, {}
    hdir, slope_w, real_slope, slope_sig, k_sl = {}, {}, {}, {}, {}
    if args.loss == 'feat':
        # each feature normalised by its own variation over the training frames in the real video (per camera), so the
        # small shape oscillation (sx, sy, aspect: a few px / percent) weighs as much as the centroid trajectory
        floor = torch.tensor([.5, .5, .5, .5, .5, .01], device=dev)
        for c in train_cams:
            series = torch.stack([feat6(masks[f, c]).detach() for f in range(1, args.train_frames+1)])
            for k_, f in enumerate(range(1, args.train_frames+1)):
                real_feat[f, c] = series[k_]
            feat_scale[c] = torch.maximum(series.std(0), floor)
            if args.pos_scale > 0:          # centroid: absolute pixel error (a few px of offset must count), not its range
                feat_scale[c][:2] = args.pos_scale
            print('feature scales %s: %s' % (CAMS[c], np.round(feat_scale[c].cpu().numpy(), 3).tolist()), flush=True)
            if args.phys_weight > 0 or args.slope_weight > 0:
                k_imp[c] = int(torch.argmin(series[:, 5]))          # most squashed real frame (index into frames 1..N)
            if args.phys_weight > 0:
                real_phys[c] = phys_obs(series, k_imp[c]).detach()
                print('real physical observables %s (impact frame %d): period %.2f decay %.3f squash %.3f ratio %.1f' % (
                    CAMS[c], k_imp[c]+1, *real_phys[c].tolist()), flush=True)
            if args.slope_weight > 0:
                # image direction perpendicular to gravity at the object (perspective projection of com and com + 1 cm g)
                cam_ = meta['cameras'][CAMS[c]]; K_ = np.array(cam_['K'], float); R_ = np.array(cam_['R']); t_ = np.array(cam_['t'])
                gd_ = np.asarray(meta['gravity'], float); gd_ = gd_/np.linalg.norm(gd_)
                pr_ = lambda x_: (lambda u_: u_[:2]/u_[2])(K_@(R_@x_+t_))
                c0_ = com.double().cpu().numpy()
                gi_ = pr_(c0_+.01*gd_)-pr_(c0_); gi_ = gi_/np.linalg.norm(gi_)
                hdir[c] = torch.tensor([-gi_[1], gi_[0]], dtype=f32, device=dev)
                nfr = args.train_frames
                # impact: strongest deceleration of the centroid along the projected gravity (most negative second
                # difference), so it also holds without a bounce or a visible squash
                s_ = (series[:, :2]@torch.tensor(gi_, dtype=f32, device=dev)).double().cpu().numpy()
                k_sl[c] = int(np.argmin(s_[2:]-2*s_[1:-1]+s_[:-2]))+1                  # series index (frame - 1)
                win_pre, win_post = list(range(0, k_sl[c]-1)), list(range(k_sl[c]+2, nfr))   # series indices (frame - 1)
                for nm_, wn_ in (('pre', win_pre), ('post', win_post)):
                    tt_ = torch.tensor(wn_, dtype=f32, device=dev); tt_ = tt_-tt_.mean()
                    slope_w[c, nm_] = (wn_, tt_/(tt_**2).sum())
                h_ = series[:, :2]@hdir[c]
                real_slope[c], slope_sig[c] = [], []
                for nm_ in ('pre', 'post'):
                    wn_, w_ = slope_w[c, nm_]
                    y_ = h_[wn_]; tt_ = torch.tensor(wn_, dtype=f32, device=dev)
                    sl_ = (w_*y_).sum(); res_ = y_-(y_.mean()+sl_*(tt_-tt_.mean()))
                    sig_ = res_.std()*np.sqrt(len(wn_)/max(len(wn_)-2, 1))*w_.norm()
                    real_slope[c].append(sl_); slope_sig[c].append(max(float(sig_), args.slope_floor))
                real_slope[c] = torch.stack(real_slope[c]); slope_sig[c] = torch.tensor(slope_sig[c], dtype=f32, device=dev)
                print('real horizontal centroid velocity %s (direction %s, impact frame %d, windows %d-%d / %d-%d): pre %.3f post %.3f px/frame, noise scale %s' % (
                    CAMS[c], np.round(hdir[c].cpu().numpy(), 3).tolist(), k_sl[c]+1, win_pre[0]+1, win_pre[-1]+1, win_post[0]+1, win_post[-1]+1, *real_slope[c].tolist(), np.round(slope_sig[c].cpu().numpy(), 3).tolist()), flush=True)
    if args.stage == 'lossdecomp':
        # gradient of each loss component (as weighted in training) w.r.t. every learned scalar at the --init-from point
        comps = ['iou', 'cx', 'cy', 'sx', 'sy', 'sqrtA', 'aspect', 'period', 'decay', 'squash', 'ratio', 'prior']
        sub = substeps_now()
        nf_ = args.train_frames
        names_ = [('logE', shared, 4), ('nu_z', shared, 5), ('damping', shared, 6), ('friction', shared, 7)]+(
            [('alpha', shared, ogden_col)] if ogden_col >= 0 else [])+[('dv0_%s' % a_, dv0, k_) for k_, a_ in enumerate('xyz')]+[
            ('dw0_%s' % a_, dw0, k_) for k_, a_ in enumerate('xyz')]+([('c1', cfield, 1), ('c2', cfield, 2)] if contact_col >= 0 else [])
        rows = {}
        for comp in comps:
            for t_ in (shared, dv0, dw0, cfield):
                t_.grad = None
            pos, quat = rollout(nf_+1, sub)
            L_ = 0.; series_ = {}
            for f in range(1, nf_+1):
                for c in train_cams:
                    if comp == 'prior':
                        continue
                    place(pos, quat, f, c)
                    a_ = silhouette(c)
                    fs = feat6(a_)
                    series_.setdefault(c, []).append(fs)
                    if comp == 'iou':
                        inter = (a_*masks[f, c]).sum()
                        L_ = L_+args.iou_weight*(1-inter/(a_.sum()+masks[f, c].sum()-inter).clamp(min=1e-6))/(nf_*len(train_cams))
                    elif comp in ('cx', 'cy', 'sx', 'sy', 'sqrtA', 'aspect'):
                        j_ = ['cx', 'cy', 'sx', 'sy', 'sqrtA', 'aspect'].index(comp)
                        L_ = L_+args.feat_weight*((fs[j_]-real_feat[f, c][j_])/feat_scale[c][j_])**2/(nf_*len(train_cams))
            if comp in ('period', 'decay', 'squash', 'ratio'):
                j_ = ['period', 'decay', 'squash', 'ratio'].index(comp)
                for c in train_cams:
                    po = phys_obs(torch.stack(series_[c]), k_imp[c])
                    L_ = L_+args.phys_weight*(torch.log(po[j_])-torch.log(real_phys[c][j_]))**2/len(train_cams)
            if comp == 'prior':
                gdir = torch.tensor(meta['gravity'], dtype=f32, device=dev); gdir = gdir/gdir.norm()
                L_ = (shared[7]/.3)**2+((dv0@gdir)/.02)**2+(dw0**2).sum()/.2**2+(cfield**2).sum()/2.**2
                L_ = args.prior_weight*L_+0.*pos.sum()
            L_.backward()
            rows[comp] = (float(L_), [float(t_.grad.reshape(-1)[k_]) if t_.grad is not None else 0. for _, t_, k_ in names_])
            print('component %-7s value %.4f done' % (comp, float(L_)), flush=True)
        print('\ngradient of each loss component (rows) w.r.t. each parameter (columns); natural units: log params per 0.1, '
              'dv0 per 0.01 m/s, dw0 per 0.1 rad/s, c per 1 mm (shared nu_z raw)', flush=True)
        unit_ = [.1, 1., .1, .1, .1]+[.01]*3+[.1]*3+[1., 1.]
        print('%-8s %8s ' % ('', 'value')+' '.join('%9s' % n_ for n_, _, _ in names_), flush=True)
        tot = np.zeros(len(names_))
        for comp in comps:
            v_, g_ = rows[comp]; g_ = np.array(g_)*np.array(unit_[:len(names_)]); tot += g_
            print('%-8s %8.4f ' % (comp, v_)+' '.join('%+9.4f' % x_ for x_ in g_), flush=True)
        print('%-8s %8s ' % ('total', '')+' '.join('%+9.4f' % x_ for x_ in tot), flush=True)
        (args.out/'lossdecomp.json').write_text(json.dumps(dict(params=[n_ for n_, _, _ in names_], rows=rows), indent=1))
        return
    lr0s = [g_['lr'] for g_ in optimizer.param_groups]
    history, best = [], (np.inf, shared.detach().clone(), dv0.detach().clone(), dw0.detach().clone(), cfield.detach().clone())
    grad_norms = []
    start = time.perf_counter()
    for it in range(args.iterations):
        if it and it % 5 == 0:
            substeps = substeps_now()
        optimizer.zero_grad()
        flight = bool(args.flight_init and args.learn_init)
        if args.profile:
            torch.cuda.synchronize(); _t0 = time.perf_counter()
        nwin = min(args.train_frames, args.window_start+it//max(args.window_step, 1)) if args.window_start > 0 else args.train_frames
        full_win = nwin == args.train_frames
        pos, quat = rollout(nwin+1, substeps, init_grad=not flight)
        if args.init_cut:
            cutw_, f_sim, f_cut = cut_weights(pos, len(pos))
            offs_cut[0] = f_cut if args.init_cut in (2, 3) else 10**9
        if args.profile:
            torch.cuda.synchronize(); _t1 = time.perf_counter()
            if it == 1:                     # render + loss backward alone (sim positions detached)
                pd_, qd_ = pos.detach().requires_grad_(), quat.detach().requires_grad_()
                _l = 0.
                for f in range(1, args.train_frames+1):
                    for c in train_cams:
                        place(pd_, qd_, f, c); a_ = silhouette(c)
                        _l = _l+((feat6(a_)-real_feat[f, c])**2).sum()+(a_*masks[f, c]).sum()*1e-6
                torch.cuda.synchronize(); _r0 = time.perf_counter()
                _l.backward(); torch.cuda.synchronize()
                print('profile: render+loss forward alone %.1f s, its backward %.1f s' % (_r0-_t1, time.perf_counter()-_r0), flush=True)
                torch.cuda.synchronize(); _t1 = time.perf_counter()
        loss = 0.
        sim_series, phys_log = {}, {}
        if args.pos_boost > 0:          # first frame with a body inside its (field-shifted) contact reach
            with torch.no_grad():
                gp_t = torch.tensor([wf.ground_p[0], wf.ground_p[1], wf.ground_p[2]], dtype=f32, device=dev)
                gn_t = torch.tensor(meta['ground_normal'], dtype=f32, device=dev)
                reach = model.contact_radius.to(f32)+(theta_of(shared.detach())[:, contact_col] if contact_col >= 0 else 0.)
                touching = (((pos.detach()-gp_t)@gn_t) < reach[None]).any(1)
                f_c = int(torch.nonzero(touching)[0]) if bool(touching.any()) else args.train_frames+1
        for f in range(1, nwin+1):
            if args.pos_boost > 0:
                wpos = 1.+args.pos_boost*(1. if f < f_c else float(np.exp(-(f-f_c+1)/args.pos_tau)))
                fw = torch.tensor([wpos, wpos, 1., 1., 1., 1.], dtype=f32, device=dev)
            else:
                fw = 1.
            for c in train_cams:
                place(pos, quat, f, c)
                a_ = silhouette(c)
                if args.loss in ('iou', 'feat'):
                    inter = (a_*masks[f, c]).sum()
                    l_iou = 1-inter/(a_.sum()+masks[f, c].sum()-inter).clamp(min=1e-6)
                    if args.loss == 'iou':
                        loss = loss+l_iou
                    else:
                        fs = feat6(a_)
                        sim_series.setdefault(c, []).append(fs)
                        loss = loss+args.iou_weight*l_iou+args.feat_weight*(fw*((fs-real_feat[f, c])/feat_scale[c])**2).sum()
                else:
                    loss = loss+((a_-masks[f, c])**2).mean()
        loss = loss/(nwin*len(train_cams))
        if args.loss == 'feat' and args.phys_weight > 0 and full_win:
            l_phys = 0.
            for c in train_cams:
                po = phys_obs(torch.stack(sim_series[c]), k_imp[c])
                l_phys = l_phys+((torch.log(po)-torch.log(real_phys[c]))**2).sum()
                phys_log[c] = po.detach().cpu().numpy().round(3).tolist()
            loss = loss+args.phys_weight*l_phys/len(train_cams)
        if args.loss == 'feat' and args.slope_weight > 0 and full_win:
            l_sl = 0.
            for c in train_cams:
                h_ = torch.stack(sim_series[c])[:, :2]@hdir[c]
                sl_ = torch.stack([(slope_w[c, nm_][1]*h_[slope_w[c, nm_][0]]).sum() for nm_ in ('pre', 'post')])
                l_sl = l_sl+(((sl_-real_slope[c])/slope_sig[c])**2).sum()
                phys_log.setdefault(c, []).extend(sl_.detach().cpu().numpy().round(3).tolist())
            loss = loss+args.slope_weight*l_sl/len(train_cams)
        value = float(loss.detach())
        if args.prior_weight > 0:              # weak priors (not part of the reported data loss)
            gdir = torch.tensor(meta['gravity'], dtype=f32, device=dev); gdir = gdir/gdir.norm()
            prior = (shared[7]/.3)**2 if args.contact in ('coulomb', 'hertz') else 0.
            if args.learn_init:
                if args.prior_vz > 0:
                    prior = prior+((dv0@gdir)/args.prior_vz)**2
                if args.prior_spin > 0:
                    prior = prior+(dw0**2).sum()/args.prior_spin**2
            if contact_col >= 0:
                prior = prior+(cfield_now()**2).sum()/2.**2               # 2 mm
            loss = loss+args.prior_weight*prior
        if args.per_body and args.smooth_weight > 0:
            th = theta_of(shared)
            loss = loss+args.smooth_weight*((th[model.i]-th[model.j])**2).sum(-1).mean()
        if args.profile:
            torch.cuda.synchronize(); _t2 = time.perf_counter()
        assert not (args.init_decay > 0 and args.init_cut)
        wf.adj_decay = float(np.exp(-1./args.init_decay)) if args.init_decay > 0 else None
        if args.init_cut:
            wf.adj_decay = cutw_
            wf.adj2_stash = True
        loss.backward()
        if args.init_cut:
            cut_init_grads()
            wf.adj2_stash = False
            offs_cut[0] = 10**9
            if it % 5 == 0:
                print('init cut: simulated first contact frame %d, video %d -> dx0 / dv0 from frames < %d' % (f_sim, real_contact[0], f_cut), flush=True)
        wf.adj_decay = None
        if args.init_decay > 0 and contact_col >= 0 and cfield.grad is not None:
            # c0 from the discounted adjoint: theta[:, contact_col] = 1e-3 (cbasis @ cfield), cbasis[:, 0] = 1
            cfield.grad[0] += 1e-3*float((wf.g_theta_decay-wf.g_theta_plain)[:, contact_col].sum())
        if args.window_start > 0 and it % 5 == 0:
            print('training window: frames 1..%d' % nwin, flush=True)
        if args.profile:
            torch.cuda.synchronize()
            print('profile: substeps/frame %d | rollout forward %.1f s, render + loss %.1f s, backward %.1f s' % (
                substeps, _t1-_t0, _t2-_t1, time.perf_counter()-_t2), flush=True)
        if args.pos_boost > 0 and it % 5 == 0:
            print('pos boost: first contact at frame %d' % f_c, flush=True)
        if flight:
            # second pass: the initial-state corrections from the free flight (weights 1 before the first contact,
            # decaying after it); material / contact-field gradients of this pass are discarded
            with torch.no_grad():
                Pd = pos.detach()
                gp_t = torch.tensor([wf.ground_p[0], wf.ground_p[1], wf.ground_p[2]], dtype=f32, device=dev)
                gn_t = torch.tensor(meta['ground_normal'], dtype=f32, device=dev)
                reach = model.contact_radius.to(f32)+(theta_of(shared.detach())[:, contact_col] if contact_col >= 0 else 0.)
                touching = (((Pd-gp_t)@gn_t) < reach[None]).any(1)
                f_c = int(torch.nonzero(touching)[0]) if bool(touching.any()) else args.train_frames+1
            nfl = min(args.train_frames, f_c+3)
            keep_g = [t_.grad.clone() if t_.grad is not None else None for t_ in (shared, cfield)]
            pos_f, quat_f = rollout(nfl+1, substeps, init_grad=True)
            l_fl = 0.
            for f in range(1, nfl+1):
                wgt = 1. if f < f_c else float(np.exp(-(f-f_c+1)/args.flight_tau))
                for c in train_cams:
                    place(pos_f, quat_f, f, c)
                    a_ = silhouette(c)
                    inter = (a_*masks[f, c]).sum()
                    l_iou = 1-inter/(a_.sum()+masks[f, c].sum()-inter).clamp(min=1e-6)
                    lf = args.iou_weight*l_iou
                    if args.loss == 'feat':
                        lf = lf+args.feat_weight*(((feat6(a_)-real_feat[f, c])/feat_scale[c])**2).sum()
                    l_fl = l_fl+wgt*lf
            (l_fl/(nfl*len(train_cams))).backward()
            for t_, g_ in zip((shared, cfield), keep_g):
                t_.grad = g_
            if it % 5 == 0:
                print('flight pass: first contact at frame %d, %d frames' % (f_c, nfl), flush=True)
        if contact_col >= 0 and args.cfield_terms < 6 and cfield.grad is not None:
            cfield.grad[args.cfield_terms:] = 0.
        if contact_col >= 0 and args.cfield_fix0 and cfield.grad is not None:
            cfield.grad[0] = 0.
        if args.fix_nu or args.fix_E:
            assert args.param == 'Enu' and not args.per_body
            if args.fix_nu:
                shared.grad[5] = 0.
            if args.fix_E:
                shared.grad[4] = 0.
        if not np.isfinite(value) or not torch.isfinite(shared.grad).all():
            print('non-finite at iteration', it, flush=True); break
        # spike guard: the adjoint of a near-unstable explicit rollout can explode while the forward stays finite;
        # a single such gradient poisons Adam's moments for many iterations -> clip to 20x the running median norm
        gn = float(shared.grad.norm())
        if len(grad_norms) >= 3 and gn > 20*float(np.median(grad_norms)):
            print('gradient spike at iteration %d (%.3g vs median %.3g): clipped' % (it, gn, float(np.median(grad_norms))), flush=True)
            shared.grad.mul_(20*float(np.median(grad_norms))/gn)
        grad_norms.append(min(gn, 20*float(np.median(grad_norms))) if len(grad_norms) >= 3 else gn)
        # exact state this gradient was taken at, and the gradient itself (reproduce spikes later)
        raw_ = dict(grad_norm_unclipped=gn, shared=shared.detach().cpu().tolist(), cfield=cfield.detach().cpu().tolist(),
                    grad_shared=shared.grad.detach().cpu().tolist() if shared.grad is not None else None,
                    grad_cfield=cfield.grad.detach().cpu().tolist() if cfield.grad is not None else None)
        if args.learn_init:
            raw_.update(dv0=dv0.detach().cpu().tolist(), dw0=dw0.detach().cpu().tolist())
        raw_.update({k_: t_.detach().cpu().tolist() for k_, t_ in extras.items()})
        if value < best[0] and full_win:
            best = (value, shared.detach().clone(), dv0.detach().clone(), dw0.detach().clone(), cfield.detach().clone(),
                    {k_: t_.detach().clone() for k_, t_ in extras.items()})
        optimizer.step()
        if args.tilt_relative:                    # keep cfield[1:3] = the derived c1 c2 (logs, report, saved state)
            with torch.no_grad():
                cfield[1:3] = cfield_now(False)[1:3]
        if args.lr_final < 1:
            frac = args.lr_final+(1-args.lr_final)*.5*(1+np.cos(np.pi*min(it+1, args.iterations-1)/max(args.iterations-1, 1)))
            for g_, lr0_ in zip(optimizer.param_groups, lr0s):
                g_['lr'] = lr0_*frac
        with torch.no_grad():
            if args.param == 'Enu':        # nu is bounded by the sigmoid: clamp every column except z
                keep = shared[5].clone(); shared.clamp_(-3, 3); shared[5] = keep
            else:
                shared.clamp_(-3, 3)
        if args.param == 'Enu':
            th_now = theta_of(shared.detach())[0]
            print('E x%.3f nu %.4f' % tuple(e_nu(shared.detach()).values())+' | damping x%.3f friction %.3f' % (float(torch.exp(th_now[6])), wf.friction0 and float(np.exp(float(th_now[7])+wf.friction0)) if args.contact in ('coulomb', 'hertz') else float(th_now[7]))
                  +(' | alpha %.3f' % (args.ogden_alpha*float(torch.exp(th_now[ogden_col]))) if ogden_col >= 0 else '')
                  +(' | dv0 %s dw0 %s' % (np.round(dv0.detach().cpu().numpy(), 4).tolist(), np.round(dw0.detach().cpu().numpy(), 3).tolist()) if args.learn_init else '')
                  +(' | contact field (mm) %s' % np.round(cfield.detach().cpu().numpy(), 3).tolist() if contact_col >= 0 else '')
                  +(' | rot0 %s dx0(mm) %s' % (np.round(rot0.detach().cpu().numpy(), 4).tolist(), np.round(dx0.detach().cpu().numpy(), 3).tolist()) if args.learn_pose else '')
                  +(' | offs %s' % np.round(offs_t.detach().cpu().numpy(), 3).tolist() if args.learn_offsets else '')
                  +(' | rest s %.3f' % float(rest_s(False)) if rest_A is not None else ''), flush=True)
        if phys_log:
            print('sim physical observables (period, decay, squash, ratio[, h-velocity pre, post]): %s' % {CAMS[c]: v_ for c, v_ in phys_log.items()}, flush=True)
        history.append(dict(iteration=it, silhouette_mse=value, loss_kind=args.loss, grad_norm=float(shared.grad.norm()), parameters=summary(shared),
                            substeps=substeps, seconds=time.perf_counter()-start, raw_before_step=raw_))
        print(json.dumps(history[-1]), flush=True)
        (args.out/'history.json').write_text(json.dumps(history, indent=1))
    with torch.no_grad():
        shared.copy_(best[1])
        if len(best) > 2:
            dv0.copy_(best[2]); dw0.copy_(best[3]); cfield.copy_(best[4])
        if len(best) > 5:
            for k_, t_ in extras.items():
                t_.copy_(best[5][k_])
    report['best'] = evaluate(substeps_now())
    report['init_correction'] = dict(dv0=dv0.detach().cpu().tolist(), dw0=dw0.detach().cpu().tolist())
    report['contact_field_mm'] = cfield.detach().cpu().tolist()
    if ogden_col >= 0:
        report['ogden_alpha'] = args.ogden_alpha*float(torch.exp(theta_of(shared.detach())[:, ogden_col].mean()))
    report['seconds'] = time.perf_counter()-start
    names = ('kn', 'ks', 'kb', 'kt', 'lam', 'mu', 'damping')
    best_theta = theta_of(best[1])
    mean = best_theta.mean(0).cpu().numpy()
    report['multipliers'] = dict(zip(names, np.exp(mean[:7]).tolist()))
    if args.param == 'Enu':
        report['E_nu'] = e_nu(best[1])
    if ncol == 9:
        sig = lambda z: float(1/(1+np.exp(-z)))
        report['multipliers'].update(friction=sig(mean[7]+wf.friction0), restitution=sig(mean[8]+wf.restitution0))
    elif ncol == 8:
        report['multipliers'].update(coulomb_mu=float(np.exp(mean[7]+wf.friction0)))
    np.savez_compressed(args.out/'theta_best.npz', theta=best_theta.cpu().numpy())
    if not args.per_body:          # raw optimiser parameters, for --init-from (resume)
        (args.out/'best_shared.json').write_text(json.dumps(dict(best=dict(parameters=best[1].cpu().tolist()),
                                                             dv0=dv0.detach().cpu().tolist(), dw0=dw0.detach().cpu().tolist(),
                                                             cfield=cfield.detach().cpu().tolist(),
                                                             **{k_: t_.detach().cpu().tolist() for k_, t_ in extras.items()})))
    (args.out/'report.json').write_text(json.dumps(report, indent=1))
    print('best', json.dumps({k_: v for k_, v in report['best'].items() if k_ != 'iou_per_frame_cam'}), json.dumps(report['multipliers']), flush=True)


if __name__ == '__main__':
    main()
