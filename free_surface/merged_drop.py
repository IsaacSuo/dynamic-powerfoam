"""merged_ul.py on PAC-NeRF's plasticine scene 0 against its MPM particles (2026-10-31): the shape of elastic scene 0 (frame 0 identical), the scene's E, nu, yield stress and start
velocity (data/pacnerf/gt/plasticine.json), gravity, the ground at y = -0.05 with friction large enough for the MPM's sticky ground (ours = theirs + (0.5, -0.05, 0.5), as plastic_drop.py).
Per frame (1/24 s): height and widths (ours: the cap cells' outermost points), the centre of mass, Chamfer in both directions against 30000 of the particles.
Frames saved on the way (x, radii, L, ep, yielded), every --every steps; the log line-buffered; no timeout.
  python free_surface/merged_drop.py --foam <dir with dem_exact_single> --save <dir> --tag <tag>
Server form (11-06, branch plasticine-k1; the model untouched, this is k1, the accepted landing of 11-01):
  --scene k (0-9): the scene's E, nu, yield, v0 and its start (scene 0's foam turned, R, c0, ck) from
  <foam>/plasticine_scenes.json (made once from PAC-NeRF's data), so no PAC-NeRF data is needed; with --data holding
  simulation_data.zip the FRAME lines and Chamfer are against the MPM as before, without it our shape only.
  Exit code 1 when a step's solve did not converge."""
import argparse
import io
import json
import math
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import warp as wp

from free_surface.closure_warp import TrustCG, k_Gul_adj
from free_surface.merged_ul import MergedStep, MergedUL
from free_surface.ul_plastic import k_Aul_adj2, k_Ge_adj, k_gg_cell_adj, k_gg_wall_adj

f64, dev = torch.float64, 'cuda'
wp_t = lambda a_: wp.to_torch(a_, requires_grad=False)
OFFSET = np.array([.5, -.05, .5])
a_ = argparse.ArgumentParser()
a_.add_argument('--foam', type=Path, required=True)
a_.add_argument('--scene', type=int, default=0, help="PAC-NeRF plasticine scene 0-9 (<foam>/plasticine_scenes.json)")
a_.add_argument('--truth', default=None, help="the MPM truth's folder in simulation_data.zip (default plasticine/<scene>; cat, playdoh for the demo objects)")
a_.add_argument('--data', type=Path, default=Path(__file__).resolve().parents[1]/'data'/'pacnerf', help="PAC-NeRF's data: with simulation_data.zip the frames are compared with the MPM")
a_.add_argument('--save', type=Path, default=Path('output/plastic'))
a_.add_argument('--tag', default='drop')
a_.add_argument('--frames', type=int, default=15)
a_.add_argument('--dt', type=float, default=1/1200)
a_.add_argument('--tol', type=float, default=1e-3)
a_.add_argument('--mu', type=float, default=10., help="the ground's friction: large for the MPM's sticky ground; with --stick shear it only turns the glue on")
a_.add_argument('--stick', choices=('coulomb', 'shear'), default='coulomb', help="the glue's limit: mu F_n, or the material's shear yield on the cell's footprint (PlasticUL.set_stick)")
a_.add_argument('--grad', default='gg')
a_.add_argument('--every', type=int, default=10)
a_.add_argument('--pre-every', type=int, default=4)
a_.add_argument('--log-from', type=int, default=10**9, help="every step from this one on: the start's |g|, the balls' overlap and radii, the Newton trace")
a_.add_argument('--newton-max', type=int, default=300)
a_.add_argument('--theta', type=float, default=.5)
a_.add_argument('--kshare', type=float, default=1/6)
a_.add_argument('--log-mem', type=int, default=10**9)
a_.add_argument('--probe-from', type=int, default=10**9, help="every step from this one on, at its first point: the springs' part of |g|, the levers F T, the neighbours' relative rotation (logging only)")
a_.add_argument('--budget-from', type=int, default=10**9, help="every step from this one on: the energy account by channel, each one's stored change minus its forces' work over the step (logging only)")
a_.add_argument('--steps', type=int, default=0, help='stop after this step (0: at the last frame)')
a_.add_argument('--hold-from', type=int, default=10**9, help="every --hold-every steps from this one on: each touching cell's ground force against tau_y A, and what pushes it: pressure, shape, springs (logging only)")
a_.add_argument('--hold-every', type=int, default=5)
a_.add_argument('--omega-at', type=int, nargs='*', default=[], help="logging only: at these steps' start, k1's omega_max by Lanczos on its operator there (explicit step limit), then the step as before")
a_.add_argument('--omega-k', type=int, default=120)
a_.add_argument('--floor-check', type=float, nargs='*', default=[], help="logging only, with --omega-at: omega_max with k1's own fit floored at these eps (k1_slave.floor_fit), the arrays restored after")
a_.add_argument('--screen-variants', default='V0', help="with --follow-screen: the follow rules to compare, comma separated (k1_follow_screen.VARIANTS)")
a_.add_argument('--radii-trace', type=int, default=0, help='logging only: from this step on, each step\'s radii Newton (error, worst cell, halvings) and geometry calls')
a_.add_argument('--radii-rtol', type=float, default=0., help='the radii Newton\'s CG relative tolerance: 0 keeps min(0.1, err); otherwise min(this, err)')
a_.add_argument('--radii-start', default=None, choices=['last', 'predict'], help="the radii solve's start each cut: the last cut's radii, or predicted to first order from the last cut by the step's site motion (Recut.predict); default predict (11-06: explicit start error 1e-1 -> 9e-4, geometry calls 3.5 -> 2.0 a step; implicit 3.6 -> 2.1)")
a_.add_argument('--asm', default='warp', choices=['torch', 'warp'], help="the cut's assembly: MergedUL._assemble_torch, or asm_warp.AsmWarp (the same outputs in Warp with one host read, 11-06)")
a_.add_argument('--asm-check', type=int, default=0, help='with --asm warp: the first this many cuts run both assemblies on the same input and print the largest differences (ASMCHECK)')
a_.add_argument('--end', default=None, choices=['torch', 'warp'], help="the step's end (MergedUL.end_ul): torch, or end_warp.EndWarp (the same update on the device, the bookkeeping read where logged, 11-06); default warp")
a_.add_argument('--f-volume-test', choices=['off', 'observe', 'match-l'], default='off', help='diagnostic: positive F volume matched to L after return, with energy accounting; stop before inversion; main implicit loop only')
a_.add_argument('--end-check', type=int, default=0, help='with --end warp: the first this many ends run both on the same state and print the largest differences (ENDCHECK)')
a_.add_argument('--cg', default='warp', choices=['torch', 'warp'], help="the radii's CG: recut.GraphedCG (torch), or cg_warp.WarpCG on geo_pipe's site lists (11-06; with --geom pipe)")
a_.add_argument('--alpha-f32', action='store_true', help='the alpha complex clipped in single precision as before 11-06 (default now double)')
a_.add_argument('--geom', default='pipe', choices=['torch', 'warp', 'pipe'], help="the cut's geometry values: gpu_geometry.Geometry (torch) or geometry_warp.GeometryWarp (the same values in Warp kernels; no patch_nn, so not with --diag2); pipe: the whole geometry call in Warp with one host read (geo_pipe.GeoPipe, 11-06)")
a_.add_argument('--snap-test', type=int, default=0, help='at this step: take the full state, run the step, put the state back, run it again, compare (k1_snapshot.py)')
a_.add_argument('--snap-at', type=int, nargs='*', default=[], help='save the full state at the start of these steps (snap_<tag>_step<s>.pt)')
a_.add_argument('--snap-stall-err', type=float, default=0., help='with --snap-stall: only a stall whose error is at least this')
a_.add_argument('--snap-stall', action='store_true', help='keep the full state at each step\'s start; save it the first time the radii solve stalls (Recut.st_min)')
a_.add_argument('--snap-hard', type=int, default=0, help='keep the full state at each step\'s start; save it (up to 3 times) when the step needs at least this many Newton iterations')
a_.add_argument('--snap-fail', action='store_true', help='keep the full state at each step\'s start on the device; save it when the step fails')
a_.add_argument('--replay', type=Path, default=None, help='a saved full state: run that step again (--replay-mode) and stop')
a_.add_argument('--replay-steps', type=int, default=1, help='with --replay asis: run this many steps (step and end), then save the positions (xend_<tag>.pt)')
a_.add_argument('--replay-compare', type=Path, default=None, help='with --replay: after the asis run, the step\'s end against this saved state of the next step')
a_.add_argument('--replay-mode', default='asis', help='comma separated: asis (unchanged), c_only (the frozen C read at x_n, C = I; rotation and the springs\' beta as read)')
a_.add_argument('--explicit', action='store_true', help='step explicitly with the follow map instead of the implicit midpoint (k1_explicit_run.py; needs --theta 1)')
a_.add_argument('--dt-explicit', type=float, default=5.2e-5, help='the explicit step (0.9 x 2 / omega_max of the V0 follow map, 3.45e4 at rest and at the impact state measured)')
a_.add_argument('--follow', default='v0', choices=['v0', 'none'], help='with --explicit: the V0 follow map rebuilt each cut, or none')
a_.add_argument('--follow-every', type=int, default=1)
a_.add_argument('--t-end', type=float, default=0., help='with --explicit: stop at this time (0: at the last frame)')
a_.add_argument('--time-from', type=int, default=1, help='with --explicit: time the steps from this one on (before it: warm-up, a restored state\'s first cut)')
a_.add_argument('--cut-every', type=int, default=1, help='with --explicit: a new cut (and follow map) every this many steps, the steps between on the kept cut with the state rebased (k1_lag.py)')
a_.add_argument('--cut-eps', type=float, default=0., help='with --explicit: also a new cut when a cell\'s strain since the cut passes this (k1_lag.strain_max)')
a_.add_argument('--lag-omega', type=int, nargs='*', default=[], help='with --cut-every / --cut-eps: at these steps the step limit on the state (k1_lag.omega_max)')
a_.add_argument('--lag-strain-log', type=float, default=0., help='logging only, with --cut-every / --cut-eps: at each cut whose largest strain since the last one passes this, its spread and the most strained cells')
a_.add_argument('--diag-from', type=int, default=0, help='logging only, with --explicit: from this step a line a step (KE, the fastest site, the most strained cell, the follow map, the cut), and the step limit by Lanczos once the KE grows 5 % in a step')
a_.add_argument('--lag-strain-sample', action='store_true', help='logging only: the same at the first cut of each log window')
a_.add_argument('--lag-patch', type=int, default=0, help='with --explicit: the affine patch test of the kept cut over this many steps from the (replayed) state, then stop')
a_.add_argument('--tprof', type=int, default=0, help='timing only, with --explicit: a torch.profiler trace of this many steps from --time-from, GPU time against wall time per part (k1_tprof.py)')
a_.add_argument('--struct-count', action='store_true', help='logging only, with --explicit: how many walls, links, hidden flags change within a cut and from cut to cut (k1_profile.install_struct_count)')
a_.add_argument('--trace-f', action='store_true', help="logging only: after each step's end update, the cells whose F's stretch hit the floor or inverted, and each cell's first F collapse; --trace-cells watched every step from --trace-from (k1_fail_trace.py)")
a_.add_argument('--trace-cells', default='7344,6297')
a_.add_argument('--c-two-pass', action='store_true', help='11-06 the expert\'s single-variable check: after the solve, C (only) read again at its root and the step solved again (MergedStep.c_two_pass)')
a_.add_argument('--op-at', type=int, default=40, help='--replay-mode opcheck: the first pass stopped after this many Newton iterations (the stalled point)')
a_.add_argument('--flip-guard', type=float, default=0., help='11-07 the expert\'s feasibility check: the step\'s end F keeps det(I + dG) >= this on every cell, the Newton step shortened to the feasible length (0: off)')
a_.add_argument('--slide', type=float, default=0., help='with --flip-guard: when a Newton step is cut below this fraction, a step along the wall instead (the active cells\' det gradients projected out of the preconditioned gradient), Armijo on the energy')
a_.add_argument('--slide-delta', type=float, default=0., help='the sliding step\'s length bound |y|_M (0: the blocked Newton step\'s own M-norm)')
a_.add_argument('--c1', default='predictor', choices=['predictor', 'start'], help='11-07: where pass 1 reads C: at the predictor (as before) or at the step\'s start (C = I); pass 2 reads it at the root either way')
a_.add_argument('--resume', type=Path, default=None, help='a saved full state: continue the run from that step (the main loop, all logging as usual)')
a_.add_argument('--snap-drift', type=float, default=0., help='keep the full state at each step\'s start; save it (up to 3 times) when a cell\'s log det F - L first falls below -this after the step')
a_.add_argument('--krylov-tr', action='store_true', help='persistent Lanczos trust-region solve with the model analytic tangent; default off')
a_.add_argument('--krylov-gpu', action='store_true', help='same Krylov trust-region search with device-controlled iterations; implies --krylov-tr')
a_.add_argument('--patch-cg', action='store_true', help='11-06 the stalled-CG fallback (k1_patchcg): a CG at the cap is solved again at the same point with the residual cells\' joint block and the column-based split product')
a_.add_argument('--patch-eta', type=float, default=.005)
a_.add_argument('--patch-check-py', action='store_true', help='the first fallback also solved by the python path (k1_patchcg.solve_py), compared')
a_.add_argument('--c2-refresh', action='store_true', help='with --c-two-pass: pass 2 rebuilds the preconditioner\'s blocks with C2')
a_.add_argument('--snap-costliest', action='store_true', help='keep the full state at the start of the step with the most products so far; save it at the end (or the failure)')
a_.add_argument('--snap-cap', type=int, default=0, help='keep the full state at each step\'s start; save it (up to 2 times) when the first pass has at least this many CG exits at the cap (and at a failure with --snap-fail)')
a_.add_argument('--newton-log', action='store_true', help='logging only: per Newton iteration region, predicted / actual reduction, CG exit, the true linear residual (one more product), the step; steps above 50 iterations to nlog_<tag>.json, all at the failure')
a_.add_argument('--vol-account', type=int, nargs='*', default=None, help='logging only: the account of log det F against L each step for these cells, and the most shrunk cells every 25 steps (k1_vol_account.py)')
a_.add_argument('--trace-from', type=int, default=200)
a_.add_argument('--profile', action='store_true', help="timing only: each phase of the run CUDA-synchronised, and every 50 steps the V0 follow map's rebuild at that cut (k1_profile.py)")
a_.add_argument('--follow-screen', action='store_true', help="logging only, with --omega-at: the nested follow levels' speed, one-step fidelity and cost (k1_follow_screen.py)")
a_.add_argument('--diag2', action='store_true', help="logging only, with --omega-at: k1_diag2.run (differences, Lanczos convergence, integration constraint, cross terms, consistent follow)")
a_.add_argument('--follow-check', action='store_true', help="logging only, with --omega-at: the tiny and the stiff particles following an anchor's affine motion (k1_slave.follow_omega)")
a_.add_argument('--slave-check', action='store_true', help="logging only, with --omega-at: also the reduced operator with the stiff particles following their neighbours' fit (k1_slave.py)")
a_.add_argument('--fv-ale-remap-check', action='store_true', help='isolated FV-ALE geometry/remap trial on the full foam; no solid-dynamics or landing claim')
args = a_.parse_args()
gt = json.loads((args.foam/'plasticine_scenes.json').read_text())['scenes'][str(args.scene)]
E, nu, ys = gt['E']/1000, gt['nu'], gt['ys']/1000
g = np.load(args.foam/'dem_exact_single'/'geometry.npz'); cl = np.load(args.foam/'dem_exact_single'/'cells.npz')
X0, r0, V0, C0 = g['points'], cl['radii'], g['masses'], g['contacts']; n = len(X0)
if args.scene != 0:                                                                  # (scene k's start: scene 0's foam turned)
    R_s, c0_, ck_ = (np.array(gt[k_]) for k_ in ('R', 'c0', 'ck'))
    X0 = ck_+(X0-c0_)@R_s.T; C0 = ck_+(C0-c0_)@R_s.T
args.save.mkdir(parents=True, exist_ok=True)
if args.fv_ale_remap_check:
    from free_surface.fv_ale_check import run as fv_ale_check
    check = fv_ale_check(X0, r0, args)
    raise SystemExit(0 if check['passed'] else 2)
have_truth = (args.data/'simulation_data.zip').exists()
Z = zipfile.ZipFile(args.data/'simulation_data.zip') if have_truth else None


def truth(k):
    from plyfile import PlyData
    v = PlyData.read(io.BytesIO(Z.read('simulation_data/%s/%d.ply' % (args.truth or 'plasticine/%d' % args.scene, k))))['vertex']
    return np.stack([v['x'], v['y'], v['z']], 1).astype(np.float64)+OFFSET


if have_truth:
    rng = np.random.default_rng(0)
    Gt = [truth(k) for k in range(args.frames+1)]
    pick = rng.choice(len(Gt[0]), 30000, replace=False)
    Gp = [torch.as_tensor(p[pick], device=dev).float() for p in Gt]


def chamfer(a, b, chunk=1024):
    """the mean nearest distance a -> b, b -> a (mm)"""
    a = a.float()

    def one(p, q):
        return torch.cat([torch.cdist(p[i:i+chunk], q).min(1).values for i in range(0, len(p), chunk)]).mean()
    return 1e3*float(one(a, b)), 1e3*float(one(b, a))


mass = torch.as_tensor(V0, dtype=f64, device=dev); M = float(mass.sum()); M3 = mass[:, None]
W = MergedUL(X0, r0, V0, g['pairs'], g['areas'], C0, E, nu, ys, theta=args.theta, kshare=args.kshare, grad=args.grad)
W.set_ball((0., 1., 0.), -.05, args.mu)
if args.stick == 'shear':
    W.set_stick()
if args.radii_rtol > 0:
    W._recut.cg_rtol = lambda err_, t_=args.radii_rtol: min(t_, err_)
W.radii_start = args.radii_start or 'predict'                                   # (11-06: implicit too: the cut 138 -> 51 ms a step at the landing's start)
W._recut.use_wcg = args.cg == 'warp'
if args.alpha_f32:
    W._recut.A.f32 = True
if args.geom in ('warp', 'pipe'):
    from free_surface.geometry_warp import GeometryWarp
    W._recut.geom_cls = GeometryWarp
if args.geom == 'pipe':
    from free_surface.geo_pipe import GeoPipe
    W._recut.pipe = GeoPipe(len(X0))
if args.asm == 'warp':
    from free_surface.asm_warp import AsmWarp
    W.asm_warp = AsmWarp(W); W.asm = 'check' if args.asm_check > 0 else 'warp'; W.asm_check_left = args.asm_check
if (args.end or 'warp') == 'warp':                                                  # (11-06: implicit too, checked at theta 1/2: Ln, ep 1e-15, F_e 1e-14, counts equal)
    from free_surface.end_warp import EndWarp
    W.end_warp = EndWarp(W); W.end_mode = 'check' if args.end_check > 0 else 'warp'; W.end_check_left = args.end_check
    W.end_flush_each = not args.explicit                                            # (the implicit driver's logs read W.Wp and the counts each step)
W.update_ul(W.X)
W.prepare()
up = torch.tensor([0., 1., 0.], dtype=f64, device=dev)
fg = -9.8*M3*up; wref = float(fg.norm()); tol = args.tol*wref
CG = TrustCG(W, mass, args.dt, per=2, product_iters=1)
ST = MergedStep(W, CG, mass, W.ul_blocks(), fg, args.dt, args.mu)
ST.pre_ul = True; ST.pre_every = args.pre_every; ST.smooth_pred = ST.carry_delta = True; ST.newton_max = args.newton_max
ST.set_theta(args.theta, mass)
ST.c_two_pass, ST.newton_log, ST.c2_refresh = args.c_two_pass, args.newton_log, args.c2_refresh
ST.krylov_tr = args.krylov_tr or args.krylov_gpu
ST.krylov_gpu = args.krylov_gpu
ST.patch_cg, ST.patch_eta, ST.patch_log = args.patch_cg, args.patch_eta, lambda m_: print(m_, flush=True)
ST.patch_check_py = args.patch_check_py; ST.patch_check_all = args.replay is not None; ST.c1_identity = args.c1 == 'start'; ST.flip_guard = args.flip_guard; ST.slide, ST.slide_thr, ST.slide_delta = args.slide > 0, args.slide, args.slide_delta
TP, NLOG, GUARD = [], {}, []
ST.set_state(W.X.clone(), torch.tensor(gt['v'], dtype=f64, device=dev).expand(n, 3).clone(), fg/M3)
if args.profile and not args.explicit:
    from free_surface.k1_profile import install as prof_install, report as prof_report
    PROF = prof_install(W, ST)
VACC = None
if args.vol_account is not None:                                                    # (k1_vol_account: log det F against L, step by step)
    from free_surface.k1_vol_account import install as vacc_install
    VACC = vacc_install(W, ST, lambda: s, watch=tuple(args.vol_account), log=lambda m_: print(m_, flush=True))
if args.trace_f:
    from free_surface.k1_fail_trace import install as ftrace_install
    ftrace_install(W, ST, lambda: s, watch=tuple(int(c_) for c_ in args.trace_cells.split(',')), trace_from=args.trace_from)
h = float(np.median(V0))**(1/3)
print('plasticine %d: E %.1f, nu %.4f, yield %.1f Pa (yield strain %.4f), v0 %s; foam %s: %d cells, h %.2f mm; dt 1/%d, tol %g of the weight, friction %g, grad %s; MPM %d particles' % (
    args.scene, gt['E'], nu, gt['ys'], gt['ys']/gt['E'], np.round(gt['v'], 4).tolist(), args.foam.name, n, 1e3*h, round(1/args.dt), args.tol, args.mu, args.grad, len(Gt[0]) if have_truth else 0), flush=True)
if not have_truth:
    print('no MPM particles at %s: the FRAME lines give our shape only (compare afterwards with flow_spreads.py on the frames)' % args.data, flush=True)


def ours():
    """the sites and the cap cells' outermost points, on the last cut's radii and cap directions"""
    x = ST.tx; a = W.tul['a']; c = a.norm(dim=1) > 0
    return torch.cat([x, x[c]+W._ulr[c, None]*a[c]/a[c].norm(dim=1, keepdim=True)])


def row(k):
    o = ours()
    if not have_truth:
        oh = (o.max(0).values-o.min(0).values).cpu().numpy(); com = ((M3*ST.tx).sum(0)/M).cpu().numpy(); nan = float('nan')
        print('FRAME %2d t %.3f: height %.3f; x width %.3f, z width %.3f; com %s; low %.4f (no MPM here)' % (k, k/24, oh[1], oh[0], oh[2], np.round(com, 4).tolist(), float(o[:, 1].min())), flush=True)
        return (k, k/24, *oh, nan, nan, nan, *com, nan, nan, nan, float(o[:, 1].min()), nan, nan, nan)
    gp = Gt[k]
    oh = (o.max(0).values-o.min(0).values).cpu().numpy(); gh = gp.max(0)-gp.min(0)
    c1, c2 = chamfer(o, Gp[k])
    com = ((M3*ST.tx).sum(0)/M).cpu().numpy()
    print('FRAME %2d t %.3f: height %.3f (MPM %.3f); x width %.3f (%.3f), z width %.3f (%.3f); com %s (%s); low %.4f (%.4f); Chamfer ours->MPM %.1f, MPM->ours %.1f mm' % (
        k, k/24, oh[1], gh[1], oh[0], gh[0], oh[2], gh[2], np.round(com, 4).tolist(), np.round(gp.mean(0), 4).tolist(), float(o[:, 1].min()), gp[:, 1].min(), c1, c2), flush=True)
    return (k, k/24, *oh, *gh, *com, *gp.mean(0), float(o[:, 1].min()), gp[:, 1].min(), c1, c2)


rows = [row(0)]
xs, rs, Ls, ys_, eps, ts, Fs_ = [], [], [], [], [], [], []


def save():
    np.savez(args.save/('frames_%s.npz' % args.tag), x=np.array(xs), r=np.array(rs), L=np.array(Ls), yld=np.array(ys_), ep=np.array(eps), t=np.array(ts), rows=np.array(rows), F=np.array(Fs_, dtype=np.float32))
    if probes:
        np.save(args.save/('probe_%s.npy' % args.tag), np.array(probes))


probes = []


def probe(ST_):
    """at the step's first point, after the re-read: the springs' part of |g| (their k off for one evaluation, then back), the levers p of their chain through G, the neighbours' relative rotation (F no longer enters the springs). Logging only"""
    s_, m_, lo, hi = W.tsp, W._m_s, W.tlo, W.thi
    g_all = wp_t(ST_.a['gz']).clone()
    sk_ = s_['sk'].clone(); s_['sk'].zero_()
    ST_._eval(0.); g_no = wp_t(ST_.a['gz']).clone()
    s_['sk'].copy_(sk_)
    ST_._eval(0.); ST_._accept()
    n_all, n_no, n_sp = float(g_all.norm())/wref, float(g_no.norm())/wref, float((g_all-g_no).norm())/wref
    gn = g_all.norm(dim=1); top = int(gn.argmax())
    F0, T, k_, b_, r_, dn_ = s_['F0'], s_['sP'][:m_], s_['sk'][:m_], s_['sb'][:m_], s_['sr'][:m_], s_['sdn'][:m_]
    dl = (ST_.tx[hi]-ST_.tx[lo]).norm(dim=1)
    ai = aj = T.norm(dim=1)
    lev = torch.maximum(ai, aj)/dl
    sg = torch.linalg.svdvals((.5*(F0[lo]+F0[hi])).cpu())[:, 2].to(dev)
    U, _, Vh = torch.linalg.svd(F0.cpu())
    U[:, :, 2] *= torch.sign(torch.linalg.det(U@Vh))[:, None]
    R = (U@Vh).to(dev)
    ang = torch.rad2deg(torch.acos((((R[lo]*R[hi]).sum((1, 2))-1)/2).clamp(-1, 1)))
    Vi = W.tul['Vi']
    f_ = k_*(((1-W.th)*dn_+W.th*b_[:, None]*r_).norm(dim=1))
    e_ = int((f_*torch.maximum(ai*Vi[lo], aj*Vi[hi])).argmax()); c1, c2 = int(lo[e_]), int(hi[e_])
    ring = torch.cat([hi[lo == c1], lo[hi == c1], hi[lo == c2], lo[hi == c2], torch.tensor([c1, c2], device=dev)])
    q_ = lambda a__, p_: float(torch.quantile(a__, p_))
    med = W.V0.median()
    print('  lever step %d t %.4f: first |g| %.3g, without the springs %.3g, the springs\' part %.3g; top site %d, |g| %.3g, V0/med %.2f; Fbar smallest sigma: min %.3g, 0.1 %% %.3f, 1 %% %.3f; |p| / |d|: median %.2f, 99 %% %.2f, 99.9 %% %.2f, max %.3g, above 3: %d, 10: %d, 100: %d; |p| / |d| max %.3g; relative rotation in degrees: median %.1f, 99 %% %.1f, 99.9 %% %.1f, max %.2f, above 90: %d, 150: %d; |r| / |d| at the start max %.3g; the top spring (%d, %d): lever %.3g, sigma %.3g, rotation %.2f, |r| / |d| %.3g, |G_i - G_j| %.3g, k %.3g, beta %.3f, V0/med %.2f %.2f; the top site in its rings: %d' % (
        s, s*args.dt, n_all, n_no, n_sp, top, float(gn[top])/wref, float(W.V0[top]/med), float(sg.min()), q_(sg, .001), q_(sg, .01), q_(lev, .5), q_(lev, .99), q_(lev, .999), float(lev.max()), int((lev > 3).sum()), int((lev > 10).sum()), int((lev > 100).sum()), float((T.norm(dim=1)/dl).max()),
        q_(ang, .5), q_(ang, .99), q_(ang, .999), float(ang.max()), int((ang > 90).sum()), int((ang > 150).sum()), float((r_.norm(dim=1)/dl).max()), c1, c2, float(lev[e_]), float(sg[e_]), float(ang[e_]), float(r_[e_].norm()/dl[e_]), float((W.tG[c1]-W.tG[c2]).norm()), float(k_[e_]), float(b_[e_]), float(W.V0[c1]/med), float(W.V0[c2]/med), int(bool((ring == top).any()))), flush=True)
    probes.append((s, s*args.dt, n_all, n_no, n_sp, float(sg.min()), q_(lev, .999), float(lev.max()), q_(ang, .999), float(ang.max()), float((r_.norm(dim=1)/dl).max()), float(lev[e_]), float(sg[e_]), int(bool((ring == top).any()))))
    # the first |g| by what the cells put on their walls: pressure and variance, shape; the ground's largest ball
    Lb_, Fb_ = wp_t(W.B['Lb']), wp_t(W.B['Fb'])
    ic_, jc_, m2_ = W._w[0], W._w[1], W._w[5]; wa_ = W.tul['wa'][:m2_]
    Am = torch.zeros(W.n, dtype=f64, device=dev); Am.index_reduce_(0, ic_, wa_, 'amax'); Am.index_reduce_(0, jc_, wa_, 'amax')
    pr, sh = Lb_.abs()*Vi*Am*3**.5/wref, (Fb_@F0.transpose(1, 2)).flatten(1).norm(dim=1)*Vi*Am/wref
    fb_ = torch.as_tensor(W.B['bfn'].numpy(), device=dev).abs()/wref

    def cell(c_):
        sv_, sf_ = torch.linalg.svdvals((W.I3+W.tG[c_]).cpu()), torch.linalg.svdvals(F0[c_].cpu())
        return '%d: V0/med %.2f, V / (V0 J) %.3f, J %.3f, L(z) %.3f, beta %.3f, e_i / e_y %.1f, I + G %.3f %.3f, F %.3f %.3f, y/h %.1f' % (c_, float(W.V0[c_]/med), float(1/(Vi[c_]*W.V0[c_]*torch.exp(W.tul['Ln'][c_]))), float(torch.exp(W.tul['Ln'][c_])), float(W.tL[c_]), float(W.tbe[c_]), float(W.tei[c_])/W.ey, float(sv_[2]), float(sv_[0]), float(sf_[2]), float(sf_[0]), float(ST_.tz[c_, 1]+.05)/h)
    cp_, cs_ = int(pr.argmax()), int(sh.argmax())
    print('  channels step %d: pressure |dE/dL| A / V max %.3g at %s; shape max %.3g at %s; ground largest ball %.3g at %d' % (s, float(pr[cp_]), cell(cp_), float(sh[cs_]), cell(cs_), float(fb_.max()), int(fb_.argmax())), flush=True)
    if n_all > 1.:                                                                  # (11-06: a jump at the step's start, the random failures)
        st_ = W.ul_stats; sr_ = W._recut.stats.get('start_ratio')
        top5 = torch.argsort(gn, descending=True)[:5].tolist()
        print('  JUMP step %d t %.4f: first |g| %.3g weights; the cut: radii iterations %s, fails %s, floored %s, springs born %s, closed %s, flips %s, V / target at the radii start min %.3g max %.3g; '
              'the largest |g| cells: %s' % (s, s*args.dt, n_all, st_.get('radii_it'), st_.get('radii_fail'), st_.get('floored'), st_.get('born'), st_.get('closed'), st_.get('flips'),
                                             sr_[0] if sr_ else float('nan'), sr_[1] if sr_ else float('nan'),
                                             ' | '.join('|g| %.3g, %s' % (float(gn[c_])/wref, cell(c_)) for c_ in top5)), flush=True)


holds = []
ht_ = {k_: wp.zeros(n, dtype=wp.mat33d, device=dev) for k_ in ('Gb', 'Gs', 'Gbe', 'Gb0')}; ht_['g'] = wp.zeros(n, dtype=wp.vec3d, device=dev)


def chain(Gb_):
    """the sites' force from dE/dG = Gb_, through the same kernels as MergedUL.run, on arrays of its own"""
    U, Kw = W.ul, W.ul['Kw']
    wall = [U['wi'], U['wj'], U['wa'], U['wn'], U['wt'], U['wtau']]
    wp_t(ht_['Gb']).copy_(Gb_)
    for k_ in ('Gbe', 'Gb0', 'g'):
        ht_[k_].zero_()
    wp.launch(k_gg_cell_adj, n, inputs=[ht_['Gb'], U['a'], U['Mc'], U['Vi']], outputs=[ht_['g'], ht_['Gs'], ht_['Gbe']])
    wp.launch(k_gg_wall_adj, Kw, inputs=wall+[ht_['Gs']], outputs=[ht_['g'], ht_['Gbe']])
    wp.launch(k_Ge_adj, n, inputs=[ht_['Gbe'], U['used']], outputs=[ht_['Gb0']])
    wp.launch(k_Aul_adj2, Kw, inputs=[ht_['Gbe'], U['As'], U['Ao'], U['Aw']], outputs=[ht_['Gb0']])
    wp.launch(k_Gul_adj, 2*Kw, inputs=[ht_['Gb0'], U['S'], U['O'], U['R']], outputs=[ht_['g']])
    return -wp_t(ht_['g']).clone()


def hold():
    """at the accepted z: for each touching cell with a plan area, the glue's force f against tau_y A, and the horizontal forces on its site: pressure with variance, shape, springs. Logging only"""
    th, s_, m_, lo, hi = W.th, W.tsp, W._m_s, W.tlo, W.thi
    x, zz = ST.tx, ST.tz
    A = W.footprint(x, W._ulr)
    f = wp_t(W.B['bff'])
    q = (W.tbbound > 0) & (A > 0)
    if not bool(q.any()):
        return
    B_, Lb_, Fb_, C_, F0_ = wp_t(W.B['Gb']).clone(), wp_t(W.B['Lb']).clone(), wp_t(W.B['Fb']).clone(), W.tul['Q'], s_['F0']
    Gp, Gs_ = Lb_[:, None, None]*C_, Fb_@F0_.transpose(1, 2)
    Fp, Fs, Fc_ = chain(Gp), chain(Gs_), chain(B_-Gp-Gs_)
    k_, b_, r_, dn_ = s_['sk'][:m_], s_['sb'][:m_], s_['sr'][:m_], s_['sdn'][:m_]
    fs = k_[:, None]*((1-th)*dn_+th*b_[:, None]*r_)
    Fd = torch.zeros_like(x).index_add(0, hi, -fs).index_add(0, lo, fs)
    Fsp = Fc_+Fd
    bn = torch.as_tensor(W.B['bfn'].numpy(), device=dev)
    # the check: these with the ball's force are all of the internal force
    res = -W.tgx-(Fp+Fs+Fsp+f+bn[:, None]*up)
    hz = torch.tensor([1., 0., 1.], dtype=f64, device=dev)
    c = (x[q]*hz).mean(0)
    d = (x-c)*hz; R = d.norm(dim=1); er = d/R.clamp(min=1e-12)[:, None]
    lim = W.tau_y*A
    fh = (f*hz).norm(dim=1)
    rt = (fh/lim.clamp(min=1e-300))[q]
    Rb = torch.quantile(R[q], .95)
    ring = [R[q] < .5*Rb, (R[q] >= .5*Rb) & (R[q] < .8*Rb), R[q] >= .8*Rb]
    ou = lambda F_: (((F_*er).sum(1))/lim.clamp(min=1e-300))[q]
    md = lambda a__: float(a__.median()) if len(a__) else 0.
    qt = lambda a__, p_: float(torch.quantile(a__, p_))
    atb = (fh[q] >= .999*W.tbbound[q]).float().mean()
    out = [s, s*args.dt, int(q.sum()), float(A[q].sum()), qt(rt, .25), qt(rt, .5), qt(rt, .75), qt(rt, .9), float((rt > 1).float().mean()), float((rt > 2).float().mean()), float((rt > 3).float().mean()), float(fh[q].sum()/lim[q].sum()), md(rt[ring[0]]), md(rt[ring[1]]), md(rt[ring[2]]), md(-ou(f)), md(ou(Fp)), md(ou(Fs)), md(ou(Fsp)), md(ou(Fp)[ring[2]]), md(ou(Fs)[ring[2]]), md(ou(Fsp)[ring[2]]), float(atb), float(res[q].norm(dim=1).max()/lim[q].max())]
    print('  hold step %d t %.4f: touching with area %d, plan area %.3f; |f| / tau_y A: 25 %% %.2f, median %.2f, 75 %% %.2f, 90 %% %.2f; above 1 %.2f, 2 %.2f, 3 %.2f; sum f / sum limit %.2f; rings inner %.2f, middle %.2f, edge %.2f; outward, all: ground holds %.2f, pressure %.2f, shape %.2f, springs %.2f; edge: pressure %.2f, shape %.2f, springs %.2f; at the glue\'s bound %.2f; balance error / largest limit %.1e' % tuple(out), flush=True)
    holds.append(out)
    np.save(args.save/('hold_%s.npy' % args.tag), np.array(holds))


budgets = []
bud = dict(prev=None, cum=np.zeros(6))


def energies():
    """stored shape and volume after the last end, kinetic, gravity's, the plastic work (cells, springs) of the state now"""
    E_s, E_v = W.stored()
    return dict(s=E_s, v=E_v, k=.5*float((M3*ST.tv**2).sum()), g=float((-fg*ST.tx).sum()), wc=float(getattr(W, 'Wp_cell', 0.)), ws=float(getattr(W, 'Wp_spring', 0.)))


def budget_mid():
    """at the accepted z, before the step's end: the forces' work over the step by channel, gradient . dx with dx = (z - x_n) / th, G linear in x"""
    th, G, Q, F0 = W.th, W.tG, W.tul['Q'], W.tsp['F0']
    s_, m_ = W.tsp, W._m_s
    k_, b_, r_, dn_ = s_['sk'][:m_], s_['sb'][:m_], s_['sr'][:m_], s_['sdn'][:m_]
    f_ = k_[:, None]*((1-th)*dn_+th*b_[:, None]*r_)
    t_, mw = W.tul, W._w[5]
    A_ = G/th
    return dict(WL=wp_t(W.B['Lb'])*(Q*G).sum((1, 2))/th, Ws=float((wp_t(W.B['Fb'])*(G@F0)).sum())/th, Wsp=float((f_*(r_-dn_)).sum()), Wt=float((W.tgx*(ST.tz-ST.tx)).sum())/th, Ln0=t_['Ln'].clone(), Lz=W.tL.clone(), A=A_.clone(),
                Esp0=float((.5*k_*(dn_*dn_).sum(1)).sum()), wv=t_['wv'][:mw].clone(), wi=t_['wi'][:mw].long().clone(), wj=t_['wj'][:mw].long().clone(), Q=Q.clone(), G=G.clone())


def budget_end(e0, m):
    """after the step's end: each channel's stored change minus its work; what is left of the balance is the ground's work"""
    th, kap, V0_ = W.th, W.kap, W.V0
    e1 = energies()
    Ln1 = W.tul['Ln']; Ln0, Lz = m['Ln0'], m['Lz']
    Evar0, Evar1 = float((m['wv']*(Ln0[m['wi']]-Ln0[m['wj']])**2).sum()), float((m['wv']*(Ln1[m['wi']]-Ln1[m['wj']])**2).sum())
    Esp1 = W.spring_store
    WL = float(m['WL'].sum())
    D_L = (e1['v']-e0['v'])+(Evar1-Evar0)-WL
    D_s = (e1['s']-e0['s'])+(e1['wc']-e0['wc'])-m['Ws']
    D_sp = (Esp1-m['Esp0'])+(e1['ws']-e0['ws'])-m['Wsp']
    W_c = m['Wt']-WL-m['Ws']-m['Wsp']
    pv = bud['prev']
    jk, jv = (m['Esp0']-pv['Esp'], Evar0-pv['Evar']) if pv is not None else (0., 0.)
    bud['prev'] = dict(Esp=Esp1, Evar=Evar1)
    bud['cum'] += np.array([D_L, D_s, D_sp, jk, jv, -W_c])/H0
    Dc = .5*V0_*kap*(Ln1**2-Ln0**2)-V0_*kap*Lz*(m['Q']*m['G']).sum((1, 2))/th
    A_ = m['A']; tr2 = (A_*A_.transpose(1, 2)).sum((1, 2))
    form = -V0_*kap*Lz*.5*tr2
    na = A_.flatten(1).norm(dim=1)
    c_ = int(Dc.argmax())
    sv_ = torch.linalg.svdvals((W.I3+A_[c_]).cpu())
    left = 1-(e1['k']+e1['g']+e1['s']+e1['v']+Evar1+Esp1+W.Wp)/H0
    print('  budget step %d t %.4f: left %.4f; cumulative / H0: pressure and variance %.5f, shape %.5f, springs %.5f, the springs\' jump at the cut %.5f, variance\'s jump %.5f, into the ground %.5f; this step / H0: pressure %.3e (formula kap V0 L(z) (-1/2 tr A^2) %.3e; cells above 0 %.3e, below 0 %.3e); kinetic + gravity + work %.2e; top cell %d: %.3e, L(z) %.3f, J %.3f -> %.3f, tr A^2 %.3f, I + A %.3f %.3f, V0/med %.2f, y/h %.1f; cells with |A| above 0.3: %d, 1: %d' % (
        s, s*args.dt, left, *bud['cum'].tolist(), float(Dc.sum())/H0, float(form.sum())/H0, float(Dc[Dc > 0].sum())/H0, float(Dc[Dc < 0].sum())/H0, (e1['k']-e0['k']+e1['g']-e0['g']+m['Wt'])/H0, c_, float(Dc[c_])/H0, float(Lz[c_]), float(torch.exp(Ln0[c_])), float(torch.exp(Ln1[c_])), float(tr2[c_]), float(sv_[2]), float(sv_[0]), float(V0_[c_]/V0_.median()), float(ST.tx[c_, 1]+.05)/h, int((na > .3).sum()), int((na > 1).sum())), flush=True)
    budgets.append((s, s*args.dt, left, *bud['cum'].tolist(), float(Dc.sum())/H0, float(form.sum())/H0, float(Dc[Dc > 0].sum())/H0, float(Dc[Dc < 0].sum())/H0, c_, float(Dc[c_])/H0))
    np.save(args.save/('budget_%s.npy' % args.tag), np.array(budgets))


t0 = t_saved = time.time()
Y2 = float((mass*(W.X[:, 1]-(M3*W.X).sum(0)[1]/M)**2).sum())
nit, npr = [], []
nf = int(round(1/(24*args.dt))); nsteps = args.steps or args.frames*nf
def omega_probe(st, hfd=1e-7):
    """Lanczos (full reorthogonalisation) on M^-1/2 K M^-1/2 at the step's first point, K by central differences of k1's
    gradient pass; afterwards that point evaluated and accepted again (the step goes on as without the probe)"""
    k = args.omega_k; Xp = st.tz.clone(); Ms = torch.sqrt(W.V0)[:, None]

    def gr(x_):
        W.tx.copy_(x_); wp.capture_launch(W.graph)
        return W.tgx.clone()
    Av = lambda v_: ((gr(Xp+hfd*v_/Ms)-gr(Xp-hfd*v_/Ms))/(2*hfd))/Ms
    Q = torch.zeros(k+1, n, 3, dtype=f64, device=dev)
    q_ = torch.randn(n, 3, dtype=f64, device=dev, generator=torch.Generator(device=dev).manual_seed(0)); Q[0] = q_/q_.norm()
    al, be = [], []
    for it in range(k):
        w_ = Av(Q[it]); a = float((w_*Q[it]).sum()); al.append(a)
        w_ = w_-a*Q[it]-(be[-1]*Q[it-1] if it > 0 else 0.)
        for _ in range(2):
            c_ = torch.einsum('kab,ab->k', Q[:it+1], w_); w_ = w_-torch.einsum('k,kab->ab', c_, Q[:it+1])
        b_ = float(w_.norm()); be.append(b_); Q[it+1] = w_/b_
    lam_, S_ = np.linalg.eigh(np.diag(al)+np.diag(be[:-1], 1)+np.diag(be[:-1], -1))
    om = np.sqrt(np.clip(lam_, 0, None)); target = 2*4800.
    y_ = torch.as_tensor(S_[:, -1], dtype=f64, device=dev); v_ = torch.einsum('k,kab->ab', y_, Q[:k]); wv = (v_*v_).sum(1)
    top = torch.argsort(wv, descending=True)[:3].tolist(); capc = W.tul['Mc'].abs().sum((1, 2)) > 0; Vm = float(W.V0.median())
    bd = torch.as_tensor(W.B['bdelta'].numpy(), device=dev)
    print('  OMEGA step %d t %.4f: omega_max %.3e -> explicit limit %.3e s (%.2f of the MPM substep); modes above %.0f: %d of the top %d; fastest mode on %s (%.0f %%; %s); touching %d' % (
        s, s*args.dt, om[-1], 2/om[-1], 2/om[-1]*4800, target, int((om > target).sum()), k, top, 100*float(wv[top].sum()),
        ', '.join('%s V/med %.2f' % ('surface' if bool(capc[c_]) else 'inside', float(W.V0[c_])/Vm) for c_ in top), int((bd > 0).sum())), flush=True)
    if args.follow_check:                                                           # (the stiff and the tiny particles following an anchor, k1_slave.follow_omega)
        from free_surface.k1_slave import column_omega, follow_omega
        follow_omega(W, gr, Xp, column_omega(W) > target, k=k, target=target, check_plumbing=(s == args.omega_at[0]), log=lambda m_: print(m_, flush=True))
    if args.slave_check:                                                            # (the stiff particles following their neighbours' fit)
        from free_surface.k1_slave import column_omega, reduced_omega
        sel = column_omega(W) > target
        for rec in reduced_omega(W, gr, Xp, sel, k=k, target=target):
            print('    SLAVE round %d: %d particles follow their neighbours\' fit (%d left free: no spanning masters), reduced omega_max %.3e -> %.2f of the MPM substep; '
                  'modes above the target %d; fastest on %s (%.0f %%); lumped master mass min %.2f of its own' % (
                      rec['round'], rec['slaved'], rec['free'], rec['om'], 2/rec['om']*4800, rec['above'], rec['top'][0], 100*rec['top'][1], rec['mr_min']), flush=True)
    if args.follow_screen:                                                          # (the follow map level by level, k1_follow_screen.py)
        from free_surface.k1_follow_screen import screen
        for var_ in args.screen_variants.split(','):
            screen(W, gr, Xp, s, lambda m_: print(m_, flush=True), variant=var_)
    if args.diag2:                                                                  # (the expert's diagnostics, k1_diag2.py)
        from free_surface.k1_diag2 import run as diag2
        diag2(W, gr, Xp, s, lambda m_: print(m_, flush=True))
    for eps_ in args.floor_check:                                                  # (k1's own fit floored, the raise filled by good neighbours' fits)
        from free_surface.k1_slave import floor_fit, lanczos
        tt_ = W.tul; ml_ = W._ml; keep_ = (tt_['S'].clone(), tt_['O'].clone(), tt_['R'].clone())
        S_f, O_f, R_f, info = floor_fit(W, eps_)
        if len(S_f) > len(tt_['S']):
            print('    FLOOR eps %g: %d entries over the capacity %d' % (eps_, len(S_f), len(tt_['S'])), flush=True); continue
        tt_['S'].zero_(); tt_['O'].zero_(); tt_['R'].zero_()
        tt_['S'][:len(S_f)] = S_f.to(torch.int32); tt_['O'][:len(S_f)] = O_f.to(torch.int32); tt_['R'][:len(S_f)] = R_f
        omf, Sf_, Qf_ = lanczos(Av, n, k)
        yf = torch.as_tensor(Sf_[:, -1], dtype=f64, device=dev); vf = torch.einsum('k,kab->ab', yf, Qf_); wf = (vf*vf).sum(1); tf = torch.argsort(wf, descending=True)[:3].tolist()
        print('    FLOOR eps %g: %d cells floored (%d with no good neighbour: the floor alone), uniform-strain error of the fit %.1e (those %.1e); '
              'omega_max %.3e -> %.2f of the MPM substep; modes above the target %d of %d; fastest on %s (%.0f %%; %s)' % (
                  eps_, info['floored'], info['nogood'], info['err_good'], info['err_nogood'], omf[-1], 2/omf[-1]*4800, int((omf > target).sum()), k, tf,
                  100*float(wf[tf].sum()/wf.sum()), ', '.join('%s V/med %.2f' % ('surface' if bool(capc[c_]) else 'inside', float(W.V0[c_])/Vm) for c_ in tf)), flush=True)
        tt_['S'].copy_(keep_[0]); tt_['O'].copy_(keep_[1]); tt_['R'].copy_(keep_[2])
    st._eval(0.); st._accept()


H0 = .5*float((M3*ST.tv**2).sum())+float((-fg*ST.tx).sum())
failed = False
from free_surface.k1_snapshot import take as snap_take, restore as snap_restore, to_host as snap_host, inventory as snap_inv
ROOTS = dict(W=W, ST=ST, CG=CG)
SNAP = None


def snap_save(snap, s_, kind):
    h_ = snap_host(snap); h_['step'] = s_
    path_ = args.save/('snap_%s_%s_step%d.pt' % (args.tag, kind, s_)); torch.save(h_, path_)
    print('  SNAP saved %s (%s)' % (path_, snap_inv(snap)), flush=True)


if args.explicit:                                                                   # (the follow version's actual run, then stop)
    from free_surface.k1_explicit_run import run as run_explicit
    if args.replay is not None:
        snapE = torch.load(args.replay, weights_only=False); snap_restore(ROOTS, snapE)
        W._ulG = None                                                               # (a snapshot keeps no geometry: the first cut starts from the restored radii)
        W.th = 1.; ST.set_theta(1., mass); print('explicit from the saved state of step %d (t %.4f)' % (snapE['step'], snapE['step']*args.dt), flush=True)
    t_end_ = args.t_end if args.t_end > 0 else nsteps*args.dt
    if args.lag_patch:                                                              # (k1_lag.patch_test: the same affine motion, a cut each step against one cut)
        from free_surface.k1_lag import patch_test
        for nm_, D_ in (('strain', [[1e-3, 4e-4, 0.], [4e-4, -6e-4, 2e-4], [0., 2e-4, -3e-4]]), ('strain and rotation', [[1e-3, 3e-3, 0.], [-2e-3, -6e-4, 2e-3], [1e-3, -2e-3, -3e-4]])):
            print('LAG patch test: %s' % nm_, flush=True)
            patch_test(W, ST, ROOTS, D_, args.lag_patch, log=lambda m_: print(m_, flush=True), trace=True)
        sys.exit(0)
    if args.profile:                                                                # (the cut split by synchronised timers, k1_profile.install_cut)
        from free_surface.k1_profile import install_cut, report_cut
        PCUT = install_cut(W, skip=args.time_from-1)
    if args.struct_count:
        from free_surface.k1_profile import install_struct_count, report_struct
        SCNT = install_struct_count(W)
    TPR = None
    if args.tprof:
        from free_surface.k1_tprof import install as tprof_install, report as tprof_report
        tprof_install(W); TPR = dict(start=max(args.time_from, 2), n=args.tprof, path=args.save/('tprof_%s.json' % args.tag))
    def ex_frame(k_):                                                               # (each frame: the MPM comparison row, and the state kept for frames_<tag>.npz)
        rows.append(row(k_)); xs.append(ST.tx.cpu().numpy()); rs.append(W._ulr.cpu().numpy()); Ls.append(W.tul['Ln'].cpu().numpy())
        ys_.append(W._yld.cpu().numpy()); eps.append(W.ep.flatten(1).norm(dim=1).cpu().numpy()); ts.append(k_/24)
    xe_, ve_ = run_explicit(W, ST, mass, fg, args.dt_explicit, t_end_, ex_frame, lambda m_: print(m_, flush=True), follow=args.follow, follow_every=args.follow_every,
                            time_from=args.time_from, tprof=TPR, cut_every=args.cut_every, cut_eps=args.cut_eps, omega_at=tuple(args.lag_omega), strain_log=args.lag_strain_log, strain_sample=args.lag_strain_sample, diag_from=args.diag_from)
    if args.profile:
        report_cut(PCUT, PCUT.n['update_ul'], log=lambda m_: print(m_, flush=True))
    if args.struct_count:
        report_struct(SCNT, log=lambda m_: print(m_, flush=True), skip=max(args.time_from-1, 1))
    if TPR is not None:
        tprof_report(TPR['path'], args.tprof, log=lambda m_: print(m_, flush=True))
    torch.save(dict(x=xe_.cpu(), F=W._Fe.cpu(), L=W.tul['Ln'].cpu()), args.save/('xend_%s.pt' % args.tag))
    if len(xs):
        save()
    print('done: %.0f s' % (time.time()-t0), flush=True)
    sys.exit(0)
def c_harm(s_, label, C_used, top=5):
    """at the current evaluation (G in W.tG): per cell the frozen error (C_used - C(z)) : dG, the actual a - b = log det(I + dG) - C_used : dG, the share of |g|^2;
    the cells with the largest |C_used - C(z)| and those with the largest |a - b|"""
    dG = W.tG/ST.th; Cz = W.C_at(); gq = wp_t(ST.a['gz']).pow(2).sum(1); gs = float(gq.sum())
    sg_, la_ = torch.linalg.slogdet(W.I3+dG); ab = la_-(C_used*dG).sum((1, 2)); ef = ((C_used-Cz)*dG).sum((1, 2)); dc = (C_used-Cz).flatten(1).abs().max(1).values
    ok = sg_ > 0; ab = torch.where(ok, ab, torch.zeros_like(ab))
    fmt = lambda c_: '%d: |dC| %.3g, (C_used - C(z)) : dG %+.2e, a - b %+.2e, |g|^2 share %.2e, V0/med %.3f' % (c_, float(dc[c_]), float(ef[c_]), float(ab[c_]), float(gq[c_])/max(gs, 1e-300), float(W.V0[c_]/W.V0.median()))
    print('CHARM step %d %s: sum |a - b| %.3e (max %.3e), sum |(C_used - C(z)) : dG| %.3e; cells |dC| > 0.1: %d, > 1: %d; flipped I + dG %d' % (
        s_, label, float(ab.abs().sum()), float(ab.abs().max()), float(ef.abs().sum()), int((dc > .1).sum()), int((dc > 1).sum()), int((~ok).sum())), flush=True)
    print('    largest |dC|: %s' % '; '.join(fmt(c_) for c_ in torch.argsort(dc, descending=True)[:top].tolist()), flush=True)
    print('    largest |a - b|: %s' % '; '.join(fmt(c_) for c_ in torch.argsort(ab.abs(), descending=True)[:top].tolist()), flush=True)


def pre_report(s_, label, e_, its_, g2_, sec):
    """one solve's Newton log grouped by the CG's exit (C cap, E edge, N negative curvature, T tolerance)"""
    st_ = ST.stats; nl_ = st_.get('nlog') or []
    torch.cuda.synchronize()
    print('PRE step %d %s: %s after %d Newton iterations, %d products, %.1f s; |g| / weight first %.3g, last %.3g (tol %.3g); blocks age %s; noise-branch rho %d' % (
        s_, label, 'converged' if math.sqrt(g2_) <= tol else 'NOT converged', its_, e_, sec, nl_[0]['g']/wref if nl_ else float('nan'), math.sqrt(max(g2_, 0.))/wref, tol/wref,
        st_.get('blk_age'), sum(1 for q_ in nl_ if q_.get('noise_branch'))), flush=True)
    for ex_ in 'CETN':
        q_ = [r_ for r_ in nl_ if r_['exit'] == ex_]
        if q_:
            med = lambda v_: float(np.median(v_))
            print('    exit %s: %d iterations, products %d (median %d); true linear residual / |g| median %.3g (min %.3g, max %.3g); ared / pred median %.4f (10 %% %.3f, 90 %% %.3f); taken %d; |g| after / before median %.3g' % (
                ex_, len(q_), sum(r_['K'] for r_ in q_), med([r_['K'] for r_ in q_]), med([r_['res_true']/r_['g'] for r_ in q_]), min(r_['res_true']/r_['g'] for r_ in q_), max(r_['res_true']/r_['g'] for r_ in q_),
                med([r_['ared']/r_['pred'] if r_['pred'] > 0 else float('nan') for r_ in q_]), float(np.nanpercentile([r_['ared']/r_['pred'] if r_['pred'] > 0 else float('nan') for r_ in q_], 10)),
                float(np.nanpercentile([r_['ared']/r_['pred'] if r_['pred'] > 0 else float('nan') for r_ in q_], 90)), sum(r_['taken'] for r_ in q_), med([r_['g_trial']/r_['g'] for r_ in q_])), flush=True)


if args.replay is not None:                                                         # (a saved step run again, then stop)
    snapF = torch.load(args.replay, weights_only=False); s = snapF['step']
    reread0 = W.reread
    snap_restore(ROOTS, snapF)
    from free_surface.k1_vol_account import drift_stats
    sg0_, ld0_ = torch.linalg.slogdet(W._Fe)
    print('REPLAY state at the start of step %d: drift log det F - L: %s; det F <= 0: %d' % (s, drift_stats(ld0_-W.tul['Ln']), int((sg0_ <= 0).sum())), flush=True)
    for mode in args.replay_mode.split(','):
        snap_restore(ROOTS, snapF); ST.probe = None
        if mode == 'anchor_check':                                                  # (k1_follow_warp.anchor_map_warp against k1_slave.anchor_map on this state's cut)
            from free_surface.k1_slave import anchor_map, column_omega
            from free_surface.k1_follow_warp import anchor_map_warp
            mt_ = W._w[5]; t_ = W.tul; wi_, wj_, wa_ = t_['wi'][:mt_].long(), t_['wj'][:mt_].long(), t_['wa'][:mt_]; X_ = t_['xn'].clone()
            sel_ = (W.V0 < W._ul_agg*W.V0.median()) | (column_omega(W) > 9600.)
            res_ = {}
            for nm_, fn_ in (('python', anchor_map), ('warp', anchor_map_warp)):
                fn_(X_, wi_, wj_, wa_, sel_); torch.cuda.synchronize(); t0_ = time.perf_counter()
                for _ in range(3):
                    out_ = fn_(X_, wi_, wj_, wa_, sel_)
                torch.cuda.synchronize(); res_[nm_] = (out_, 1e3*(time.perf_counter()-t0_)/3)
            (sp_, kp_, wpy_, fp_, mp_), tp_ = res_['python']; (sw_, kw_, ww_, fw_, mw_), tw_ = res_['warp']
            rows = lambda s__, k__, w__: {(int(a__), int(b__)): float(c__) for a__, b__, c__ in zip(s__.tolist(), k__.tolist(), w__.tolist())}
            Rp, Rw = rows(sp_, kp_, wpy_), rows(sw_, kw_, ww_)
            same_keys = set(Rp) == set(Rw); dmax = max((abs(Rp[k__]-Rw.get(k__, 0.)) for k__ in Rp), default=0.)
            print('ANCHOR %d slaved: python %.1f ms, warp %.1f ms; entries %d vs %d, same (s, k) set %s, max |w diff| %.1e; free %s vs %s; max |w| %.3f vs %.3f' % (
                int(sel_.sum()), tp_, tw_, len(Rp), len(Rw), same_keys, dmax, sorted(fp_)[:5], sorted(fw_)[:5], float(np.max(mp_)), float(np.max(mw_))), flush=True)
            if not same_keys:
                d1 = sorted(set(Rp)-set(Rw))[:6]; d2 = sorted(set(Rw)-set(Rp))[:6]; print('ANCHOR only python %s, only warp %s' % (d1, d2), flush=True)
            continue
        if mode == 'column_check':                                                  # (k1_column_warp against k1_slave.column_omega)
            from free_surface.k1_slave import column_omega
            from free_surface.k1_column_warp import column_omega_warp
            res_ = {}
            from free_surface.k1_column_warp import column_omega_sort
            column_omega_warp = column_omega_sort
            for nm_, fn_ in (('torch', column_omega), ('warp', column_omega_warp)):
                fn_(W); torch.cuda.synchronize(); t0_ = time.perf_counter()
                for _ in range(5):
                    o_ = fn_(W)
                torch.cuda.synchronize(); res_[nm_] = (o_, 1e3*(time.perf_counter()-t0_)/5)
            a_, b_ = res_['torch'][0], res_['warp'][0]
            import free_surface.k1_column_warp as KCW
            KCW.TIMING = True; tt_ = []
            for _ in range(5):
                column_omega_warp(W); torch.cuda.synchronize(); e_ = time.perf_counter(); T_ = KCW._T
                tt_.append((T_['prep']-T_['start'], T_['kernel']-T_['prep'], e_-T_['kernel']))
            KCW.TIMING = False
            print('COLUMN warp parts (median ms): preparation %.1f, kernel %.1f, after %.1f' % tuple(1e3*float(np.median([r_[i_] for r_ in tt_])) for i_ in range(3)), flush=True)
            print('COLUMN torch %.1f ms, warp %.1f ms; max |warp - torch| / torch %.1e; above 9600: %d vs %d, differing cells %d' % (
                res_['torch'][1], res_['warp'][1], float(((b_-a_).abs()/a_.abs().clamp(min=1e-300)).max()), int((a_ > 9600).sum()), int((b_ > 9600).sum()), int(((a_ > 9600) != (b_ > 9600)).sum())), flush=True)
            continue
        if mode == 'fast_check':                                                    # (k1_follow_fast.FollowFast against the explicit run's FollowMapGPU2 path)
            from free_surface.k1_column_warp import column_omega_sort
            from free_surface.k1_follow import FollowMapGPU2
            from free_surface.k1_follow_warp import neighbours
            from free_surface.k1_follow_fast import FollowFast
            mt_ = W._w[5]; t_ = W.tul; wi_, wj_, wa_ = t_['wi'][:mt_].long(), t_['wj'][:mt_].long(), t_['wa'][:mt_]; X_ = t_['xn'].clone(); m_ = W.V0
            W._k3 = int((W.tul['Aw'] != 0).sum())                                    # (the restored cut's aggregation length: older snapshots do not hold it)
            A_ = FollowMapGPU2(W.n); B_ = FollowFast(W.n)
            def old_():
                om__ = column_omega_sort(W); A_.reset(); nb__ = neighbours(X_, wi_, wj_, wa_)
                A_.add(X_, wi_, wj_, wa_, W.V0 < W._ul_agg*W.V0.median(), nb=nb__); A_.add(X_, wi_, wj_, wa_, om__ > 9600., nb=nb__); A_.build(m_)
                return om__
            def new_():
                return B_.rebuild(W, X_, m_)
            oa_, ob_ = old_(), new_()
            g_ = torch.Generator(device='cuda').manual_seed(4)
            y_ = torch.randn(W.n, 3, dtype=torch.float64, device='cuda', generator=g_); r_ = torch.randn(W.n, 3, dtype=torch.float64, device='cuda', generator=g_)
            r2_ = torch.randn(W.n, 3, dtype=torch.float64, device='cuda', generator=g_)
            rel_ = lambda a__, b__: float((a__-b__).norm()/b__.norm().clamp(min=1e-300))
            same_ = bool(torch.equal(A_.slave, B_.slave)); ya_ = y_*A_.master[:, None]
            x1_, x2_ = B_.accel2(r_, r2_)
            print('FAST om rel %.1e; slaved %d vs %d, same set %s; nnz %d vs %d; P %.1e, P^T %.1e, M_r^-1 %.1e; accel2 vs two solves %.1e / %.1e' % (
                rel_(ob_, oa_), int(A_.slave.sum()), int(B_.slave.sum()), same_, int(A_.ta['ptr'][A_.ns]), B_.nnz, rel_(B_.P(ya_), A_.P(ya_)), rel_(B_.Pt(r_), A_.Pt(r_)),
                rel_(B_.force_to_accel(r_), A_.force_to_accel(r_)), rel_(x1_, A_.force_to_accel(r_)), rel_(x2_, A_.force_to_accel(r2_))), flush=True)
            ta__, tb__ = [], []
            for _ in range(8):
                torch.cuda.synchronize(); t0_ = time.perf_counter(); old_(); torch.cuda.synchronize(); ta__.append(1e3*(time.perf_counter()-t0_))
                torch.cuda.synchronize(); t0_ = time.perf_counter(); new_(); torch.cuda.synchronize(); tb__.append(1e3*(time.perf_counter()-t0_))
            print('FAST rebuild, alternating, median (min): old %.1f (%.1f) ms, new %.1f (%.1f) ms' % (float(np.median(ta__)), min(ta__), float(np.median(tb__)), min(tb__)), flush=True)
            continue
        if mode == 'build_check':                                                   # (FollowMapGPU against FollowMap on this state's V0 map)
            from free_surface.k1_column_warp import column_omega_sort
            from free_surface.k1_follow import FollowMap, FollowMapGPU
            mt_ = W._w[5]; t_ = W.tul; wi_, wj_, wa_ = t_['wi'][:mt_].long(), t_['wj'][:mt_].long(), t_['wa'][:mt_]; X_ = t_['xn'].clone()
            om_ = column_omega_sort(W); m_ = W.V0
            maps = {}
            from free_surface.k1_follow import FollowMapGPU2
            for nm_, cls_ in (('old', FollowMap), ('gpu', FollowMapGPU2)):
                F_ = cls_(W.n); F_.add(X_, wi_, wj_, wa_, W.V0 < W._ul_agg*W.V0.median()); F_.add(X_, wi_, wj_, wa_, om_ > 9600.); maps[nm_] = F_
            ts_ = {}
            for nm_, F_ in maps.items():
                ts_[nm_] = []
                for _ in range(6):
                    torch.cuda.synchronize(); t0_ = time.perf_counter(); F_.build(m_); torch.cuda.synchronize(); ts_[nm_].append(1e3*(time.perf_counter()-t0_))
            B0_ = maps['gpu']; B0_.timing = {}; parts_ = []
            for _ in range(5):
                B0_.build(m_); T_ = B0_.timing
                parts_.append((T_['compose']-T_['t0'], T_['fill']-T_['compose'], T_['assemble']-T_['fill'], T_['factor']-T_['assemble']))
            B0_.timing = None
            print('BUILD gpu parts (median ms): compose %.1f, sort and fill %.1f, assemble S %.1f, Cholesky inverse %.1f (ns %d)' % (tuple(1e3*float(np.median([p__[i__] for p__ in parts_])) for i__ in range(4))+(B0_.ns,)), flush=True)
            g_ = torch.Generator(device='cuda').manual_seed(3); A_, B_ = maps['old'], maps['gpu']
            y_ = torch.randn(W.n, 3, dtype=torch.float64, device='cuda', generator=g_)*A_.master[:, None]; r_ = torch.randn(W.n, 3, dtype=torch.float64, device='cuda', generator=g_)
            rel_ = lambda a__, b__: float((a__-b__).norm()/b__.norm())
            xr_ = B_.force_to_accel(r_)
            print('BUILD old %s ms, gpu %s ms (first includes the graphs\' capture); gpu vs old: P %.1e, P^T %.1e, M_r^-1 %.1e; gpu M_r M_r^-1 r vs P^T r %.1e' % (
                ['%.1f' % v__ for v__ in ts_['old']], ['%.1f' % v__ for v__ in ts_['gpu']], rel_(B_.P(y_), A_.P(y_)), rel_(B_.Pt(r_), A_.Pt(r_)), rel_(xr_, A_.force_to_accel(r_)),
                rel_(B_.M_ref(m_, xr_), B_.Pt_ref(r_))), flush=True)
            continue
        if mode == 'rebuild':                                                       # (the V0 follow map's rebuild on this state's cut, timed by part)
            from free_surface.k1_slave import column_omega
            from free_surface.k1_follow import FollowMap
            mt_ = W._w[5]; t_ = W.tul; wi_, wj_, wa_ = t_['wi'][:mt_].long(), t_['wj'][:mt_].long(), t_['wa'][:mt_]; X_ = t_['xn'].clone()
            for rep_ in range(3):
                sy_ = torch.cuda.synchronize; pc_ = time.perf_counter
                sy_(); t0_ = pc_(); om_ = column_omega(W); sy_(); t1_ = pc_()
                FM_ = FollowMap(W.n); FM_.add(X_, wi_, wj_, wa_, W.V0 < W._ul_agg*W.V0.median()); FM_.add(X_, wi_, wj_, wa_, om_ > 9600.); sy_(); t2_ = pc_()
                FM_.build(W.V0); sy_(); t3_ = pc_()
                print('REBUILD %d: column estimate %.1f ms, rows %.1f ms, build %.1f ms; slaved %d' % (rep_, 1e3*(t1_-t0_), 1e3*(t2_-t1_), 1e3*(t3_-t2_), int(FM_.slave.sum())), flush=True)
            continue
        if mode == 'radii':                                                         # (why the radii solve stalls, k1_radii_diag.py)
            from free_surface.k1_radii_diag import diagnose
            diagnose(W, ST.tx.clone(), lambda m_: print(m_, flush=True)); continue
        if mode == 'split':                                                         # (11-06 expert: the written perturbation and the split product on C's group; k1_opcheck.patch_split)
            from free_surface.k1_opcheck import patch_split
            patch_split(W, ST, CG, tol, wref, lambda m_: print(m_, flush=True), at=args.op_at); continue
        if mode == 'patch':                                                         # (11-06 expert: A cheap blocks / B exact diagonal on the residual's cells / C their joint block, one frozen point; k1_opcheck.patch)
            from free_surface.k1_opcheck import patch as oppatch
            oppatch(W, ST, CG, tol, wref, lambda m_: print(m_, flush=True), at=args.op_at); continue
        if mode == 'opcheck':                                                       # (11-06 expert: the product's reliability, three blocks, the preconditioned spectrum at a stalled point; k1_opcheck.py)
            from free_surface.k1_opcheck import run as opcheck
            opcheck(W, ST, CG, tol, wref, lambda m_: print(m_, flush=True), at=args.op_at); continue
        if mode == 'tangentcheck':                                                  # (11-07 review of k1_tangent: the analytic H v against central differences of the production gradient)
            ST.newton_max = args.op_at; ST.c_two_pass = False
            e_, its, g2 = ST.step(tol, lambda g_: .5*tol)
            from free_surface.k1_tangent import Tangent
            T_ = Tangent(CG); T_.setbase()
            z_ = CG.t['z'].clone(); xw_, gw_ = wp_t(W.xw), wp_t(W.xw.grad); cm_ = CG.t['cm']
            def gE_(x):
                torch.cuda.synchronize(); xw_.copy_(x); W._run(); wp.synchronize(); return gw_.clone()
            def Hfd_(v, h):
                sc_ = float(v.abs().max()); return cm_[:, None]*v+(gE_(z_+(h/sc_)*v)-gE_(z_-(h/sc_)*v))*(sc_/(2*h))
            def Han_(v):
                CG.t['d'].copy_(v); wp.capture_launch(T_.graph); wp.synchronize(); return CG.t['Hd'].clone()
            gE_(z_)
            yl_ = W.tbe < 1-1e-12; bd_ = wp_t(W.B['bdelta']); body_ = wp_t(W.ball['body']).long(); tc_ = torch.zeros(W.n, dtype=torch.bool, device=dev); tc_[body_[bd_ > 0]] = True
            gen_ = torch.Generator(device=dev).manual_seed(0)
            dirs_ = {'random': torch.randn(W.n, 3, dtype=f64, device=dev, generator=gen_)}
            m_ = torch.zeros(W.n, 1, dtype=f64, device=dev); m_[yl_] = 1.; dirs_['random on yielding cells'] = dirs_['random']*m_
            m_ = torch.zeros(W.n, 1, dtype=f64, device=dev); m_[tc_] = 1.; dirs_['random on touching cells'] = dirs_['random']*m_
            m_ = torch.zeros(W.n, 1, dtype=f64, device=dev); m_[(W.V0 < .1*W.V0.median())] = 1.; dirs_['random on tiny cells'] = dirs_['random']*m_
            dirs_['gradient'] = wp_t(ST.a['gz']).clone()
            print('TANGENT step %d after %d Newton iterations: yielding %d, touching %d, tiny %d' % (s, its, int(yl_.sum()), int(tc_.sum()), int((W.V0 < .1*W.V0.median()).sum())), flush=True)
            for nm_, v_ in dirs_.items():
                if float(v_.abs().max()) == 0.:
                    continue
                v_ = v_/v_.norm(); a_ = Han_(v_); rows_ = []
                for h_ in (1e-7, 1e-8, 1e-9):
                    f_ = Hfd_(v_, h_); rows_.append('h %.0e: |H_an - H_fd| / |H_fd| %.2e' % (h_, float((a_-f_).norm()/f_.norm())))
                f8_ = Hfd_(v_, 1e-8); er_ = (a_-f8_).norm(dim=1); top_ = torch.argsort(er_, descending=True)[:4].tolist()
                print('TANGENT %s: %s; v.H v analytic %.6e, central 1e-8 %.6e; worst rows %s' % (nm_, '; '.join(rows_), float((v_*a_).sum()), float((v_*f8_).sum()),
                      ', '.join('%d (%s%s%s, |err| %.2e of |H_fd| row %.2e)' % (c_, 'Y' if bool(yl_[c_]) else '-', 'T' if bool(tc_[c_]) else '-', 't' if bool(W.V0[c_] < .1*W.V0.median()) else '-', float(er_[c_]), float(f8_[c_].norm())) for c_ in top_)), flush=True)
            u_, w_ = dirs_['random']/dirs_['random'].norm(), dirs_['gradient']/dirs_['gradient'].norm()
            print('TANGENT symmetry: u.Hw %.6e, w.Hu %.6e' % (float((u_*Han_(w_)).sum()), float((w_*Han_(u_)).sum())), flush=True)
            continue
        if mode == 'wallcheck':                                                     # (11-07 expert: at the wall, is the gradient normal to it (a constrained stationary point) or is there descent along it?)
            e_, its, g2 = ST.step(tol, lambda g_: .5*tol)
            st_ = ST.stats; gl_ = st_.get('guard_log', [])
            print('WALL step %d: %s after %d Newton iterations, |g| / weight %.4g; guard: shortened %d, at the wall %d, breaches %d / %d' % (
                s, 'converged' if math.sqrt(g2) <= tol else 'NOT converged', its, math.sqrt(g2)/wref, st_.get('guard', 0), st_.get('guard_wall', 0), st_.get('guard_breach', 0), st_.get('guard_breach_z', 0)), flush=True)
            if not gl_:
                continue
            c_ = gl_[-1][5]; th_ = ST.th; m_ = ST.flip_guard
            ST._eval(0.); gz_ = wp_t(ST.a['gz']).clone(); z_ = ST.tz.clone(); G_ = W.tG.clone()
            detc = lambda G__: float(torch.det(W.I3+G__[c_]/th_))
            d0_ = detc(G_)
            # the stencil: the cell's wall partners and theirs (the cut's walls)
            mt_ = W._w[5]; wi_, wj_ = W.tul['wi'][:mt_].long(), W.tul['wj'][:mt_].long()
            nb1 = torch.unique(torch.cat([wj_[wi_ == c_], wi_[wj_ == c_], torch.tensor([c_], device=dev)]))
            sel_ = torch.zeros(W.n, dtype=torch.bool, device=dev); sel_[nb1] = True
            nb2 = torch.unique(torch.cat([wj_[sel_[wi_]], wi_[sel_[wj_]], nb1]))
            xw_, h_ = wp_t(W.xw), 1e-7
            nvec = torch.zeros(W.n, 3, dtype=f64, device=dev)
            for k_ in nb2.tolist():
                for a_ in range(3):
                    xp_ = z_.clone(); xp_[k_, a_] += h_; xw_.copy_(xp_); W._run(); dp_ = detc(W.tG)
                    xm_ = z_.clone(); xm_[k_, a_] -= h_; xw_.copy_(xm_); W._run(); dm_ = detc(W.tG)
                    nvec[k_, a_] = (dp_-dm_)/(2*h_)
            xw_.copy_(z_); W._run()
            nn_ = float((nvec*nvec).sum()); gn_ = float((gz_*nvec).sum())/nn_
            g_norm = gn_*nvec; g_tan = gz_-g_norm
            print('WALL step %d cell %d (V0/med %.3f): det(I + dG) at z %.4f (margin %g); grad det over %d stencil sites, |n| %.3g; gradient split: |g| %.4g, along the wall normal %.4g (sign %+d: %s), along the wall %.4g (in weight units)' % (
                s, c_, float(W.V0[c_]/W.V0.median()), d0_, m_, len(nb2), math.sqrt(nn_), float(gz_.norm())/wref, float(g_norm.norm())/wref, 1 if gn_ > 0 else -1,
                'the energy pushes det DOWN through the wall' if gn_ > 0 else 'the energy pulls away from the wall', float(g_tan.norm())/wref), flush=True)
            # descent along the wall: the preconditioned tangent direction, line search with the guard, the energy by the step's own evaluation
            Pinv_ = CG.t['Pinv']; y_ = -torch.einsum('kab,kb->ka', Pinv_, g_tan); y_ = y_-(float((y_*nvec).sum())/nn_)*nvec
            from free_surface.closure_warp import PS_PZ, PS_PQ, PS_GQ2
            wp_t(ST.a['dz']).copy_(y_); e0_ = ST._eval(0.)[PS_PZ]
            I3_ = W.I3; A0_ = I3_+G_/th_
            rows_ = []
            for t_ in (1e-4, 1e-3, 1e-2, 1e-1, 1.):
                xw_.copy_(z_+t_*y_); W._run(); Dt_ = (W.tG-G_)/th_
                dmin_ = float(torch.linalg.det(A0_+Dt_).min())
                s_ = ST._eval(t_)
                rows_.append('t %.0e: energy change %+.4e (slope %.3e), det min %.4f, |g| %.4g' % (t_, s_[PS_PQ]-e0_, (s_[PS_PQ]-e0_)/t_, dmin_, math.sqrt(max(s_[PS_GQ2], 0.))/wref))
            print('WALL step %d descent along the wall (the preconditioned tangent gradient, |y| %.3g): %s' % (s, float(y_.norm()), ' | '.join(rows_)), flush=True)
            ST._eval(0.)
            continue
        if mode == 'retcheck':                                                      # (11-07: the step's end on the watched cells: the trial F_n, its polar factor and the return, warp and torch side by side)
            e_, its, g2 = ST.step(tol, lambda g_: .5*tol)
            print('RETCHECK step %d: %s after %d Newton iterations' % (s, 'converged' if math.sqrt(g2) <= tol else 'NOT converged', its), flush=True)
            from free_surface.closure_warp import k_F8
            th_ = ST.th; Fn_ = (W.tFc-(1-th_)*W.tsp['F0'])/th_; G_ = W.tG/th_
            wp.launch(k_F8, W.n, inputs=[wp.from_torch(Fn_.contiguous(), dtype=wp.mat33d)], outputs=[W.B['Fp'], W.B['Rp'], W.B['qp']]); Rk_ = W.tRp.clone()
            U_, S_, Vh_ = torch.linalg.svd(Fn_); Rs_ = U_@Vh_
            for c_ in (args.vol_account or []):
                A_ = Rk_[c_].T@Fn_[c_]; S2_ = .5*(A_+A_.T); w_ = torch.linalg.eigvalsh(S2_)
                A3_ = Rs_[c_].T@Fn_[c_]; w3_ = torch.linalg.eigvalsh(.5*(A3_+A3_.T))
                print('RETCHECK cell %d: V0/med %.3f, yielding %d (beta %.3f); dG sv %s det(I + dG) %.4f; F_n sv %s det %+.4e; rot_proper R: |R^T R - I| %.1e det %+.4f, eig sym(R^T F_n) %s; SVD polar: eig %s; |R_k - R_svd| %.2e' % (
                    c_, float(W.V0[c_]/W.V0.median()), int(W.tbe[c_] < 1), float(W.tbe[c_]), np.round(torch.linalg.svdvals(W.I3+G_[c_]).cpu().numpy(), 4).tolist(), float(torch.det(W.I3+G_[c_])), np.round(S_[c_].cpu().numpy(), 4).tolist(), float(torch.det(Fn_[c_])),
                    float((Rk_[c_].T@Rk_[c_]-W.I3).abs().max()), float(torch.det(Rk_[c_])), np.round(w_.cpu().numpy(), 5).tolist(), np.round(w3_.cpu().numpy(), 5).tolist(), float((Rk_[c_]-Rs_[c_]).abs().max())), flush=True)
            Fe0_ = W._Fe.clone(); L0_ = W.tul['Ln'].clone()
            W.end_mode = 'torch'; W._end_torch(th_); Fe_t = W._Fe.clone(); ep_t = W.ep.clone()
            W._Fe = Fe0_.clone(); W.tul['Ln'].copy_(L0_)
            for c_ in (args.vol_account or []):
                print('RETCHECK cell %d torch end: F_e sv %s, log det F_e - log det F_n %+.4f' % (c_, np.round(torch.linalg.svdvals(Fe_t[c_]).cpu().numpy(), 4).tolist(), float(torch.linalg.slogdet(Fe_t[c_])[1]-torch.linalg.slogdet(Fn_[c_])[1])), flush=True)
            continue
        if mode == 'blk_time':                                                      # (the cost of one rebuild of the preconditioner's blocks, W.ul_blocks, at this state)
            ST._eval(0.); W.reread(beta=True); ST._eval(0.)
            for _ in range(2):
                W.ul_blocks()
            torch.cuda.synchronize(); t0_ = time.perf_counter()
            for _ in range(5):
                ST.tK.copy_(W.ul_blocks())
            torch.cuda.synchronize(); print('PRE step %d: one rebuild of the blocks %.1f ms' % (s, 1e3*(time.perf_counter()-t0_)/5), flush=True); continue
        if mode.startswith('pre1') or mode.startswith('pre2'):                      # (11-06 expert: the preconditioner's blocks, cached as saved / rebuilt at this state; pass 1, then pass 2 from one z1)
            from free_surface.closure_warp import PS_GQ2 as PSG
            ST.c_two_pass = False; ST.newton_log = True; ST.pre_force = mode.endswith('fresh')
            t_r = time.time(); e_, its, g2 = ST.step(tol, lambda g_: .5*tol); ST.pre_force = False
            pre_report(s, 'pass 1, blocks %s' % ('rebuilt at this state' if mode.endswith('fresh') else 'as saved'), e_, its, g2, time.time()-t_r)
            if mode.startswith('pre1'):
                continue
            if not math.sqrt(g2) <= tol:
                print('PRE step %d: the first pass did not converge: no pass-2 comparison from this arm' % s, flush=True); continue
            z1, K1, Q1, age1 = ST.tz.clone(), ST.tK.clone(), W.tul['Q'].clone(), ST._pre_age; d1 = ST.delta_start if ST.delta_start else -1.
            c_harm(s, 'one pass (C1 read at the predictor, at z1)', Q1)
            for arm_ in ('keep', 'rebuild'):
                ST.tz.copy_(z1); ST.tK.copy_(K1); W.tul['Q'].copy_(Q1); ST._pre_age = age1
                g2_ = ST._eval(0.)[PSG]; ST._accept()                                # (z1's evaluation with C1: G(z1) in W.tG, as at the end of pass 1)
                st_ = dict(rejected=0, edge=0, negative=0, trace=[], nlog=[], g1=math.sqrt(max(g2_, 0.)))
                t_r = time.time(); ev_, it_, g2b, _ = ST.pass2(tol, lambda g_: .5*tol, st_, 0, 0, d1, arm_ == 'rebuild'); ST.stats = st_
                tp_ = st_['two_pass']
                print('PRE step %d pass 2 (%s blocks): from |g| / weight %.3g; max |C2 - C1| %.3g, after: max |C(z2) - C2| %.3g' % (s, arm_, tp_['g_start2']/wref, tp_['dC12'], tp_['dC23']), flush=True)
                pre_report(s, 'pass 2, blocks %s' % ('rebuilt with C2' if arm_ == 'rebuild' else 'kept from pass 1'), ev_, it_, g2b, time.time()-t_r)
                c_harm(s, 'two passes (%s blocks; C2 read at z1, at z2)' % arm_, W.tul['Q'].clone())
            continue
        if mode == 'c_only':
            def reread_c(*a__, **k__):
                out_ = reread0(*a__, **k__); W.tul['Q'].copy_(W.I3.expand_as(W.tul['Q'])); return out_
            W.reread = reread_c
        diag_ = []
        if mode == 'diag':                                                          # (the quadratic model against the potential along each Newton step)
            from free_surface.closure_warp import PS_PZ, PS_PQ, PS_GQ2
            cgs0, ev0 = CG.solve, ST._eval
            last_r = {}

            def cg_d(*a__, **k__):
                r__ = cgs0(*a__, **k__); last_r.clear(); last_r.update(r__); return r__

            def ev_d(t__):
                if t__ == 1. and last_r:
                    gp, php = last_r['GP'], last_r['PHP']; rowv = []
                    for tt in (.01, .1, .25, .5, .75):
                        q__ = ev0(tt); rowv.append((tt, q__[PS_PQ]-q__[PS_PZ], tt*gp+.5*tt*tt*php))
                    q__ = ev0(1.); rowv.append((1., q__[PS_PQ]-q__[PS_PZ], gp+.5*php))
                    diag_.append((last_r['K'], last_r['EDGE'], last_r['NEG'], gp, php, rowv, math.sqrt(max(q__[PS_GQ2], 0.))/wref))
                    last_r.clear()
                    return q__
                return ev0(t__)
            CG.solve, ST._eval = cg_d, ev_d
        ng0 = W._recut.stats['geometry']; W._recut.trace = []
        t_r = time.time(); e_, its, g2 = ST.step(tol, lambda g_: .5*tol); W.reread = reread0
        if mode == 'asis' and args.replay_steps > 1:
            ST.end()
            for _ in range(args.replay_steps-1):
                e_, its, g2 = ST.step(tol, lambda g_: .5*tol); ST.end()
            torch.save(dict(x=ST.tx.cpu(), F=W._Fe.cpu(), L=W.tul['Ln'].cpu(), t=(s+args.replay_steps-1)*args.dt), args.save/('xend_%s.pt' % args.tag))
            print('REPLAY %d steps from step %d: saved xend_%s.pt' % (args.replay_steps, s, args.tag), flush=True); continue
        print('REPLAY step %d mode %s: the cut: %d geometry calls, radii iterations %s, V / target at the start min %.3g max %.3g' % (s, mode, W._recut.stats['geometry']-ng0, W.ul_stats.get('radii_it'), W._recut.stats['start_ratio'][0], W._recut.stats['start_ratio'][1]), flush=True)
        print('REPLAY step %d radii Newton: %s; repairs %s' % (s, ' | '.join('%s, halvings %d' % (t_, h_) for t_, h_ in W._recut.trace), W._recut.stats.get('repairs', '-')), flush=True); W._recut.trace = None
        if mode == 'diag':
            CG.solve, ST._eval = cgs0, ev0
            for i_, (K_, ed_, ng_, gp, php, rowv, gq) in enumerate(diag_):
                if i_ < 40 or i_ % 20 == 0 or i_ == len(diag_)-1:
                    print('  DIAG step %d Newton %d: CG %d (edge %d, negative %d), model slope %.3g, curvature %.3g; actual / model change at t = %s; |g| at t = 1: %.3g' % (
                        s, i_+1, K_, ed_, ng_, gp, php, ', '.join('%g: %.3g / %.3g' % (tt, ac, mo) for tt, ac, mo in rowv), gq), flush=True)
        tr_ = ST.stats['trace']
        print('REPLAY step %d mode %s: %s after %d Newton iterations, %d products, %.1f s; first |g| %.3g, last %.3g (tol %.3g; in fg-norm units); CG %s; |g| %s; ratio %s' % (
            s, mode, 'converged' if math.sqrt(g2) <= tol else 'NOT converged', its, e_, time.time()-t_r, tr_[0][4]/wref if tr_ else float('nan'), math.sqrt(g2)/wref, tol/wref,
            [a__[0] for a__ in tr_][:40], ['%.3g' % (a__[4]/wref) for a__ in tr_][:40], ['%.2g' % a__[5] for a__ in tr_][:40]), flush=True)
        if mode in ('asis', 'c_only') and ST.c_two_pass:
            st_ = ST.stats; tp_ = st_.get('two_pass')
            print('REPLAY step %d two passes: pass 1 Newton %d, |g| %.3g%s; patch fallback triggered %d times: solve evaluations %d (%.2f s), checks %d (%.2f s), independent residuals at its exits %s; CG products in the step %d' % (
                s, st_['its1'], st_['g1']/wref, ' (FAILED, no pass 2)' if st_.get('pass1_failed') else '; pass 2 Newton %d from |g| %.3g, products %d' % (tp_['its2'], tp_['g_start2']/wref, tp_['evals2']) if tp_ else '',
                st_.get('patch', 0), st_.get('patch_evals', 0), st_.get('patch_time', 0.), st_.get('patch_checks', 0), st_.get('patch_time_checks', 0.), ['%.3g / %.3g' % r__ for r__ in st_.get('patch_res', [])], sum(a__[0] for a__ in tr_)), flush=True)
            if ST.flip_guard > 0 and st_.get('guard_log'):
                print('REPLAY step %d guard per iteration (it: det min at z, first crossing t_c, t* taken, det min at the trial, cell): %s' % (s, ' '.join('%d: %.3g %.3g %.3g %.3g (%d)' % g__ for g__ in st_['guard_log'][:30])), flush=True)
            if st_.get('slide_log'):
                print('REPLAY step %d sliding steps (active, normals, slope, t_c, t, dE, taken, |g|/w | crossing cell: its det at z | first active cell: det at z, first-order d det along y, actual d det at t 1e-3): %s' % (s, ' | '.join('%d %d %.3g %.3g %.3g %.3g %s %.4g | %d: %.3g | %d: %.3g %+.2e %+.2e' % (q_[:7]+(q_[7]/wref, q_[8], q_[9], q_[10], q_[13], q_[11], q_[12])) for q_ in st_['slide_log'][:16])), flush=True)
            if ST.flip_guard > 0:
                print('REPLAY step %d flip guard (det(I + dG) >= %g): predictor shortened to t %.3g; %d Newton steps shortened (smallest t* %.3g; stalled at the wall %d); CHECK BREACHES: trial infeasible after shortening %d, z infeasible %d; smallest det over the full trials %.3g' % (
                    s, ST.flip_guard, st_.get('guard_pred', 1.), st_.get('guard', 0), st_.get('guard_t', 1.), st_.get('guard_wall', 0), st_.get('guard_breach', 0), st_.get('guard_breach_z', 0), st_.get('guard_dmin', float('nan'))), flush=True)
            print('REPLAY step %d gradient evaluations: standard CG products %d (the capped ones included), fallback columns %d + fallback CG products %d, Newton\'s own Pi evaluations (predictor, trials, rejections, pass-2 start) %d; total %d (+ checks %d apart)' % (
                s, st_.get('cg_std', 0), st_.get('patch_cols', 0), st_.get('patch_K', 0), e_-st_.get('cg_std', 0)-st_.get('patch_evals', 0), e_, st_.get('patch_checks', 0)), flush=True)
            if not math.sqrt(g2) <= tol and ST.stats.get('nlog'):
                print('REPLAY step %d the stall, the last 12 Newton iterations (pass.it: delta | pred, ared, rho | CG K exit, |r|/stop | true linear residual / |g| | |p|_2, max site |p| (cell) | |g| before -> trial | taken, next delta):' % s, flush=True)
                for q_ in ST.stats['nlog'][-12:]:
                    print('    %d.%d: %.3g | %.3g %.3g %.3g | %d%s %.2g | %.3g | %.3g %.3g (%d) | %.4g -> %.4g | %s %.3g' % (q_['pass_'], q_['it'], q_['delta'], q_['pred'], q_['ared'], q_['rho'], q_['K'], q_['exit'], q_['rr']/max(q_['stop'], 1e-300), q_['res_true']/q_['g'], q_['p2'], q_['pmax'], q_['pcell'], q_['g']/wref, q_['g_trial']/wref, 'y' if q_['taken'] else 'n', q_['delta_next']), flush=True)
            if math.sqrt(g2) <= tol:
                sg0_, l0_ = torch.linalg.slogdet(W._Fe); dr0_ = l0_-W.tul['Ln']
                e0_ = energies(); ST.end(); e1_ = energies()
                sg_, ld_ = torch.linalg.slogdet(W._Fe); dr_ = ld_-W.tul['Ln']
                print('REPLAY step %d ended: energies before -> after the end: shape %.4e -> %.4e, volume %.4e -> %.4e, plastic work cells %.4e -> %.4e, springs %.4e -> %.4e; J %.3f..%.3f, det F <= 0: %d; drift log det F - L min %+.4f max %+.4f mean |.| %.2e; |x - x_n| max %.3e, |v| max %.3e' % (
                    s, e0_['s'], e1_['s'], e0_['v'], e1_['v'], e0_['wc'], e1_['wc'], e0_['ws'], e1_['ws'], float(torch.exp(W.tul['Ln']).min()), float(torch.exp(W.tul['Ln']).max()), int((sg_ <= 0).sum()),
                    float(dr_.min()), float(dr_.max()), float(dr_.abs().mean()), float((ST.tx-W.tul['xn']).norm(dim=1).max()), float(ST.tv.norm(dim=1).max())), flush=True)
                dG_ = W.tG/ST.th; sv_ = torch.linalg.svdvals(W._Fe); dx_ = (ST.tx-W.tul['xn']).norm(dim=1); Vm_ = W.V0.median()
                for lab_, c_ in (('most shrunk', int(dr_.argmin())), ('most grown', int(dr_.argmax())), ('largest move', int(dx_.argmax()))):
                    sH_, lH_ = torch.linalg.slogdet(W.I3+dG_[c_])
                    print('REPLAY step %d %s cell %d: V0/med %.3f, drift now %+.3f (before the step %+.3f); F sv %s; this step: log det(I + dG) %+.3f (sign %d), tr dG %+.3f, C : dG %+.3f, |dG| %.3f; |x - x_n| %.2e, |v| %.2f; yielded %d, touching %d' % (
                        s, lab_, c_, float(W.V0[c_]/Vm_), float(dr_[c_]), float(dr0_[c_]), np.round(sv_[c_].cpu().numpy(), 3).tolist(), float(lH_), int(sH_), float(dG_[c_].trace()), float((W.tul['Q'][c_]*dG_[c_]).sum()), float(dG_[c_].norm()),
                        float(dx_[c_]), float(ST.tv[c_].norm()), int(W._yld[c_]), int(bool(((wp_t(W.B['bdelta']) > 0) & (wp_t(ST.W.ball['body']).long() == c_)).any()))), flush=True)
        if mode == 'asis' and args.replay_compare is not None:
            ST.end()
            nxt = torch.load(args.replay_compare, weights_only=False)['values']
            cmp_ = []
            for name_, path_, cur_ in (('x', (('r', 'ST'), ('a', 'tx')), ST.tx), ('v', (('r', 'ST'), ('a', 'tv')), ST.tv), ('F', (('r', 'W'), ('a', '_Fe')), W._Fe),
                                       ('L', (('r', 'W'), ('a', 'tul'), ('k', 'Ln')), W.tul['Ln']), ('ep', (('r', 'W'), ('a', 'ep')), W.ep)):
                ref_ = nxt[path_][1].to(cur_.device)
                cmp_.append('%s %.1e' % (name_, float((cur_-ref_).abs().max()/ref_.abs().max().clamp(min=1e-300))))
            print('REPLAY step %d against the run\'s next state: max |difference| / max |value|: %s' % (s, ', '.join(cmp_)), flush=True)
    sys.exit(0)
s_start = 1
if args.resume is not None:                                                         # (11-07: the main loop from a saved state)
    snapR = torch.load(args.resume, weights_only=False); snap_restore(ROOTS, snapR); W._ulG = None; s_start = snapR['step']
    print('RESUME from the saved state of step %d (t %.4f): pass-1 C %s' % (s_start, s_start*args.dt, args.c1), flush=True)
FTEST = None
if args.f_volume_test != 'off':
    from free_surface.k1_f_volume_test import FVolumeTest
    FTEST = FVolumeTest(W, args.f_volume_test, args.save/('fvolume_%s.json' % args.tag))
    FTEST.state(s_start-1, 'initial')
drift_saved = []
for s in range(s_start, nsteps+1):
    t_s = time.time()
    stall0_ = W._recut.stats.get('stalled', 0)
    if args.snap_fail or args.snap_hard or args.snap_cap or args.snap_costliest or args.snap_drift or args.snap_stall or s in args.snap_at or s == args.snap_test:
        SNAP = snap_take(ROOTS)
        if s in args.snap_at:
            snap_save(SNAP, s, 'at')
    R_prev = W.tbR.clone()
    yb_ = W._yld.clone()
    ST.probe = omega_probe if s in args.omega_at else (probe if s >= args.probe_from else None)
    bud_e0 = energies() if s >= args.budget_from else None
    if args.radii_trace and s >= args.radii_trace:
        W._recut.trace = []; ng0_ = W._recut.stats['geometry']
    e_, its, g2 = ST.step(tol, lambda g_: .5*tol)
    if W._recut.stats.get('stalled', 0) > stall0_:
        print('  STALL step %d t %.4f: the radii solve stopped at error %.3g after %s iterations (worst V / target at the start %.3g..%.3g)' % (
            s, s*args.dt, W._recut.stats['stall_err'], W.ul_stats.get('radii_it'), W._recut.stats['start_ratio'][0], W._recut.stats['start_ratio'][1]), flush=True)
        if args.snap_stall and not getattr(args, 'stall_saved', False) and SNAP is not None and W._recut.stats['stall_err'] >= args.snap_stall_err:
            args.stall_saved = True; snap_save(SNAP, s, 'stall')
    if args.radii_trace and s >= args.radii_trace:
        print('  RADII step %d: %d geometry calls, start V / target %.3g..%.3g; %s' % (s, W._recut.stats['geometry']-ng0_, W._recut.stats['start_ratio'][0], W._recut.stats['start_ratio'][1],
              ' | '.join('%s, halvings %d' % (t_, h_) for t_, h_ in W._recut.trace)), flush=True); W._recut.trace = None
    if args.snap_hard and its >= args.snap_hard and SNAP is not None and getattr(args, 'n_hard', 0) < 3:
        args.n_hard = getattr(args, 'n_hard', 0)+1; print('  hard step %d: %d Newton iterations, %d products' % (s, its, e_), flush=True); snap_save(SNAP, s, 'hard')
    if s >= args.log_from:
        tr_ = ST.stats['trace']; bd = torch.as_tensor(W.B['bdelta'].numpy(), device=dev); R_ = W.tbR; tc = bd > 0
        both = (R_ > 0) & (R_prev > 0); gz = wp_t(ST.a['gz']).norm(dim=1); top = torch.argsort(gz, descending=True)[:4]
        print('  step %d t %.4f: Newton %d, products %d, rejected %d; |g| %s; CG %s; touching %d, overlap / R max %.3f, overlap / h max %.3f, ground/weight %.2f; R / h of the touching max %.2f; this cut R / last max %.2f min %.2f, new caps %d, overlap of the new caps / h max %.3f; radii %d, walls %d; |eps_p| max %.3f; end |g| top cells %s' % (
            s, s*args.dt, its, e_, ST.stats['rejected'], ['%.3g' % (a__[4]/wref) for a__ in tr_][:4], [a__[0] for a__ in tr_][:6], int(tc.sum()), float((bd[tc]/R_[tc]).max()) if bool(tc.any()) else 0., float(bd.max())/h, float(W.B['bfn'].numpy().sum())/wref,
            float(R_[tc].max())/h if bool(tc.any()) else 0., float((R_[both]/R_prev[both]).max()), float((R_[both]/R_prev[both]).min()), int(((R_ > 0) & ~(R_prev > 0)).sum()), float(bd[(R_ > 0) & ~(R_prev > 0)].max())/h if bool(((R_ > 0) & ~(R_prev > 0)).any()) else 0., W.ul_stats['radii_it'], W.ul_stats['walls'], float(W.ep.flatten(1).norm(dim=1).max()),
            ' '.join('%d: %.2f (delta/h %.3f, V0/med %.2f)' % (c_, float(gz[c_])/wref, float(bd[c_])/h, float(W.V0[c_]/W.V0.median())) for c_ in top.tolist())), flush=True)
    if not math.isfinite(g2) or math.sqrt(g2) > tol:
        tr_ = ST.stats['trace']
        g_ = wp_t(ST.a['gz']).norm(dim=1); top = torch.argsort(g_, descending=True)[:8]; I3_ = W.I3
        Gz = W.tG; sv = torch.linalg.svdvals(I3_+Gz); sf = torch.linalg.svdvals(W.tFc); dt_ = torch.linalg.det(I3_+Gz)
        print('  at the failure: I + G(z) smallest singular value 1%% %.3f, min %.3f, cells below 0.2: %d, det <= 0: %d; F(z) smallest singular value min %.3f, largest max %.3f; beta min %.3f; e_i / e_y max %.1f' % (float(torch.quantile(sv[:, 2], .01)), float(sv[:, 2].min()), int((sv[:, 2] < .2).sum()), int((dt_ <= 0).sum()), float(sf[:, 2].min()), float(sf[:, 0].max()), float(W.tbe.min()), float(W.tei.max())/W.ey), flush=True)
        print('  largest |g| cells (index: |g| / weight; smallest and largest singular value of I + G, of F; beta; yielded; V0 / median; height / h):', flush=True)
        for c_ in top.tolist():
            print('    %d: %.2f; %.3f %.3f; %.3f %.3f; beta %.3f; %d; V %.2f; y %.1f' % (c_, float(g_[c_])/wref, float(sv[c_, 2]), float(sv[c_, 0]), float(sf[c_, 2]), float(sf[c_, 0]), float(W.tbe[c_]), int(W._yld[c_]), float(W.V0[c_]/W.V0.median()), float(ST.tz[c_, 1]+.05)/h), flush=True)
        np.savez(args.save/('fail_%s.npz' % args.tag), x=ST.tx.cpu().numpy(), v=ST.tv.cpu().numpy(), a=ST.ta.cpu().numpy(), z=ST.tz.cpu().numpy(), step=s, g=wp_t(ST.a['gz']).cpu().numpy())
        print('step %d, t %.4f: the solve did not converge (|g| / weight %s after %d Newton iterations); CG %s; |g| %s; ratio %s' % (s, s*args.dt, math.sqrt(g2)/wref if math.isfinite(g2) else g2, its, [a__[0] for a__ in tr_][:30], ['%.3g' % (a__[4]/wref) for a__ in tr_][:30], ['%.2g' % a__[5] for a__ in tr_][:30]), flush=True)
        if ST.stats.get('pass1_failed'):                                            # (the per-step lines after this block do not run at a failure)
            print('  the first pass did not converge (Newton %d, |g| / weight %.3g, blocks age %s): no second pass' % (ST.stats['its1'], ST.stats['g1']/wref, ST.stats.get('blk_age')), flush=True)
        if VACC is not None:
            VACC(tag='at the failure, step %d' % s)
        print('  the Newton iterations (CG products, edge E / negative curvature N / cap C, the CG\'s last |r| / its stop, rho): %s' % ' '.join(
            '%d%s%s%s:%.2g/%.2g' % (a__[0], 'E' if a__[1] else '', 'N' if a__[2] else '', 'C' if a__[0] >= ST.cg_cap else '', (a__[7]/a__[3]) if len(a__) > 7 and a__[3] > 0 else float('nan'), a__[5]) for a__ in tr_[:40]), flush=True)
        if ST.newton_log:
            print('  NLOG step %d, every Newton iteration (pass.it: region delta | pred, ared, rho | CG products, exit E edge N negative C cap T tolerance, its last |r| / stop | |g + Hp| accumulated, by a new product | |p|_M, |p|_2, max site |p| (cell) | |g| / weight before, at the trial | taken, next delta):' % s, flush=True)
            for q_ in ST.stats['nlog']:
                print('    %d.%d: %.3g | %.3g %.3g %.3g | %d%s %.2g / %.2g | %.3g %.3g | %.3g %.3g %.3g (%d) | %.4g %.4g | %s %.3g' % (
                    q_['pass_'], q_['it'], q_['delta'], q_['pred'], q_['ared'], q_['rho'], q_['K'], q_['exit'], q_['rr'], q_['stop'], q_['res_acc']/wref, q_['res_true']/wref,
                    q_['pM'], q_['p2'], q_['pmax'], q_['pcell'], q_['g']/wref, q_['g_trial']/wref, 'y' if q_['taken'] else 'n', q_['delta_next']), flush=True)
            NLOG[s] = ST.stats['nlog']
            import json
            json.dump({str(k_): v_ for k_, v_ in NLOG.items()}, open(args.save/('nlog_%s.json' % args.tag), 'w'), default=float)
        if args.snap_fail and SNAP is not None:
            snap_save(SNAP, s, 'fail')
        failed = True
        break
    if s >= args.log_mem:
        ov_ = len(W._recut.A.ov[0]) if hasattr(W._recut.A, 'ov') else 0
        print('  mem step %d t %.4f: torch allocated %.2f GB, reserved %.2f GB, peak allocated %.2f GB; overlap pairs %d; walls %d; yielded %d; r / r0 max %.2f; J min %.3f; Newton %d, products %d' % (s, s*args.dt, torch.cuda.memory_allocated()/2**30, torch.cuda.memory_reserved()/2**30, torch.cuda.max_memory_allocated()/2**30, ov_, W.ul_stats['walls'], int(W._yld.sum()), float((W._ulr/W._rr0).max()), float(torch.exp(W.tul['Ln']).min()), its, e_), flush=True)
    bud_m = budget_mid() if s >= args.budget_from else None
    if s >= args.hold_from and s % args.hold_every == 0:
        hold()
    if FTEST is not None and not FTEST.trial(ST, s, e_, its, math.sqrt(g2)/wref):
        print('FVTEST STOP step %d: converged endpoint is inverted; no end update committed' % s, flush=True)
        if args.snap_fail and SNAP is not None:
            snap_save(SNAP, s, 'inversion')
        failed = True
        break
    ST.end()
    if FTEST is not None:
        FTEST.state(s, 'end')
    if args.snap_drift and SNAP is not None and len(drift_saved) < 3:
        sgd_, ldd_ = torch.linalg.slogdet(W._Fe); drd_ = ldd_-W.tul['Ln']; cmin_ = int(drd_.argmin())
        if float(drd_[cmin_]) < -args.snap_drift and cmin_ not in drift_saved:
            drift_saved.append(cmin_)
            print('  DRIFT step %d: cell %d (V0/med %.3f) log det F - L %+.3f after this step (sign %d); F sv %s; saved the state at the step\'s start' % (
                s, cmin_, float(W.V0[cmin_]/W.V0.median()), float(drd_[cmin_]), int(sgd_[cmin_]), np.round(torch.linalg.svdvals(W._Fe[cmin_]).cpu().numpy(), 4).tolist()), flush=True)
            snap_save(SNAP, s, 'drift%d' % cmin_)
    if s == args.snap_test:                                                         # (the replay check in this process)
        rec_ = lambda: [ST.tx.clone(), ST.tv.clone(), ST.ta.clone(), W._Fe.clone(), W.tul['Ln'].clone(), W._ulr.clone(), W.ep.clone()]
        r1_, n1_ = rec_(), (its, e_)
        snap_restore(ROOTS, SNAP)
        e_, its, g2 = ST.step(tol, lambda g_: .5*tol); ST.end()
        r2_ = rec_()
        print('  SNAPTEST step %d: %s; first run Newton %d products %d, again %d / %d; max |difference| / max |value|: %s' % (
            s, snap_inv(SNAP), n1_[0], n1_[1], its, e_, ', '.join('%s %.1e' % (k__, float((a__-b__).abs().max()/b__.abs().max().clamp(min=1e-300))) for k__, a__, b__ in zip(('x', 'v', 'a', 'F', 'L', 'radii', 'ep'), r1_, r2_))), flush=True)
    if s >= args.budget_from:
        budget_end(bud_e0, bud_m)
    nit.append(its); npr.append(e_)
    if args.snap_costliest and SNAP is not None and e_ > getattr(args, 'best_e', -1):
        args.best_e, args.best_s, BEST = e_, s, SNAP                                # (the snapshot object itself: SNAP is a new one each step)
        args.best_info = (its, ST.stats.get('its1'), ST.stats.get('two_pass', {}).get('evals1'), ST.stats.get('blk_age'))
    if args.snap_cap and SNAP is not None and getattr(args, 'n_cap', 0) < 2:
        tr1_ = ST.stats['trace'][:ST.stats.get('its1', len(ST.stats['trace']))]
        ncap_ = sum(1 for a__ in tr1_ if a__[0] >= ST.cg_cap and not a__[1] and not a__[2])
        if ncap_ >= args.snap_cap:
            args.n_cap = getattr(args, 'n_cap', 0)+1
            print('  CAP step %d (run %s): the first pass %d Newton iterations, %d CG exits at the cap, |g| / weight at its end %.3g; blocks rebuilt %s steps before' % (
                s, args.tag, ST.stats['its1'], ncap_, ST.stats['g1']/wref, ST.stats.get('blk_age')), flush=True)
            snap_save(SNAP, s, 'cap')
    if ST.flip_guard > 0 and ST.stats.get('guard', 0) + ST.stats.get('guard_breach', 0) + ST.stats.get('guard_breach_z', 0) + (1 if ST.stats.get('guard_pred') else 0) > 0:
        GUARD.append((s, ST.stats.get('guard_pred', 1.), ST.stats.get('guard', 0), ST.stats.get('guard_t', 1.), ST.stats.get('guard_wall', 0), ST.stats.get('guard_breach', 0), ST.stats.get('guard_breach_z', 0), ST.stats.get('guard_dmin', float('nan'))))
        if len(GUARD) <= 40 or s % 10 == 0:
            print('  GUARD step %d: predictor t %.3g; %d Newton steps shortened (smallest t* %.3g, at the wall %d); breaches: trial %d, z %d; smallest det over the full trials %.3g' % GUARD[-1], flush=True)
    if ST.c_two_pass and ST.stats.get('pass1_failed'):
        print('  C2 step %d: the first pass did not converge (Newton %d, |g| / weight %.3g): no second pass' % (s, ST.stats['its1'], ST.stats['g1']/wref), flush=True)
    if ST.c_two_pass and 'two_pass' in ST.stats:
        tp_ = ST.stats['two_pass']; TP.append(dict(tp_, step=s))
        if s % 25 == 0 or tp_['its2'] > 10 or not (tp_['Rh_same'] and tp_['sb_same']):
            print('  C2 step %d: pass 1 Newton %d, products %d, |g| %.3g; C read again at its root: max |C2 - C1| %.3g (cell %d); pass 2 from |g| %.3g: Newton %d, products %d; at the root max |C(z2) - C2| %.3g (cell %d, 99 %% %.2g); R, beta unchanged %s %s' % (
                s, tp_['its1'], tp_['evals1'], tp_['g1']/wref, tp_['dC12'], tp_['dC12_cell'], tp_['g_start2']/wref, tp_['its2'], tp_['evals2'], tp_['dC23'], tp_['dC23_cell'], tp_['dC23_q99'], tp_['Rh_same'], tp_['sb_same']), flush=True)
    if ST.newton_log and its > 50:
        NLOG[s] = ST.stats['nlog']
        import json
        json.dump({str(k_): v_ for k_, v_ in NLOG.items()}, open(args.save/('nlog_%s.json' % args.tag), 'w'), default=float)
    if s % args.every == 0:
        xs.append(ST.tx.cpu().numpy()); rs.append(W._ulr.cpu().numpy()); Ls.append(W.tul['Ln'].cpu().numpy()); ys_.append(W._yld.cpu().numpy()); eps.append(W.ep.cpu().numpy()); ts.append(s*args.dt); Fs_.append(W._Fe.cpu().numpy())
    if s % 20 == 0:
        i_, j_, Lw, _, _, m = W._w
        ek = .5*float((M3*ST.tv**2).sum()); ed, ev = W.stored(); eg = float((-fg*ST.tx).sum())
        bd = W.B['bdelta'].numpy()
        Es_ = float(((M3*(ST.tx-ST.tx.mean(0)))[:, 1]*(ST.tx[:, 1]-(M3*ST.tx).sum(0)[1]/M)).sum())
        print('  integrated mean strain yy %.4f xx %.4f zz %.4f; site height std / rest %.4f, its log %.4f; mean |ep| %.3f; springs %d, new %d, closed %d this step, without a wall %d; smallest one-cell dissipation / limit store %.2e' % (float(W.Hsum[1, 1]), float(W.Hsum[0, 0]), float(W.Hsum[2, 2]), (Es_/Y2)**.5, .5*math.log(Es_/Y2), float((mass*W.ep.flatten(1).norm(dim=1)).sum())/M, W.ul_stats['rows'], W.ul_stats['born'], W.ul_stats['closed'], W.ul_stats['ghost'], W.ul_stats.get('D_min', 0.)), flush=True)
        print('t %.4f (%.0f s): KE %.3e dev %.3e vol %.3e var %.3e springs %.3e W_p %.3e (springs %.3f), left %.3f of the start\'s energy; com y %.4f; yielding %d cells, yielded so far %d; walls %d; J %.3f..%.3f, |eps_p| median %.3f max %.3f; touching %d, ground/weight %.2f; min wall L / h %.3f, max r / h %.2f, radii %d (fails %d), flips %d; products %d, Newton %d, rejected %d, negative %d; %.2f s' % (
            s*args.dt, time.time()-t0, ek, ed, ev, float(W.tsum[2]), W.spring_store, W.Wp, W.Wp_spring/max(W.Wp, 1e-300), 1-(ek+ed+ev+float(W.tsum[2])+W.spring_store+W.Wp+eg)/H0, float((M3*ST.tx)[:, 1].sum())/M, W.ul_stats['yielding'], W.ul_stats['yielded'], m, float(torch.exp(W.tul['Ln']).min()), float(torch.exp(W.tul['Ln']).max()),
            float(W.ep.flatten(1).norm(dim=1).median()), float(W.ep.flatten(1).norm(dim=1).max()), int((bd > 0).sum()), float(W.B['bfn'].numpy().sum())/wref, float(Lw.min())/h, float(W._ulr.max())/h, W.ul_stats['radii_it'], W.ul_stats['radii_fail'], W.ul_stats['flips'], e_, its, ST.stats['rejected'], ST.stats['negative'], time.time()-t_s), flush=True)
    if s % nf == 0:
        rows.append(row(s//nf))
        save(); t_saved = time.time()
    elif time.time()-t_saved > 60:
        save(); t_saved = time.time()
if FTEST is not None:
    FTEST.finish(s, failed)
save()
if args.snap_costliest and getattr(args, 'best_e', -1) >= 0:
    print('  COSTLIEST step %d (run %s): %d products, Newton %d (first pass %s, its products %s), blocks age %s' % ((args.best_s, args.tag, args.best_e)+args.best_info), flush=True)
    snap_save(BEST, args.best_s, 'costliest')
ni, np_ = np.array(nit), np.array(npr)
if ni.size:
    print('Newton per step: median %d, 99 %% %d, max %d, steps above 50: %d of %d; products per step: median %d, max %d, total %d; plastic work %.4f' % (np.median(ni), np.percentile(ni, 99), ni.max(), int((ni > 50).sum()), len(ni), np.median(np_), np_.max(), np_.sum(), W.Wp), flush=True)
else:
    print('Newton per step: no accepted steps in this attempt', flush=True)
if GUARD:
    print('flip guard: %d steps touched (predictor shortened in %d, smallest t %.3g; %d Newton shortenings, smallest t* %.3g, at the wall %d); CHECK BREACHES: trial %d, z %d' % (
        len(GUARD), sum(1 for g_ in GUARD if g_[1] < 1.), min(g_[1] for g_ in GUARD), sum(g_[2] for g_ in GUARD), min(g_[3] for g_ in GUARD), sum(g_[4] for g_ in GUARD), sum(g_[5] for g_ in GUARD), sum(g_[6] for g_ in GUARD)), flush=True)
if TP:
    print('two-pass C: pass 1 Newton %d, products %d; pass 2 Newton %d, products %d; max |C(z2) - C2| over the run %.3g, max |C2 - C1| %.3g; R / beta changed in %d steps' % (
        sum(t_['its1'] for t_ in TP), sum(t_['evals1'] for t_ in TP), sum(t_['its2'] for t_ in TP), sum(t_['evals2'] for t_ in TP), max(t_['dC23'] for t_ in TP), max(t_['dC12'] for t_ in TP),
        sum(not (t_['Rh_same'] and t_['sb_same']) for t_ in TP)), flush=True)
if VACC is not None and not failed:
    VACC(tag='at the end, step %d' % s)
if args.profile:
    prof_report(PROF, time.time()-t0)
print('done: %.0f s' % (time.time()-t0), flush=True)
if failed:
    sys.exit(1)
