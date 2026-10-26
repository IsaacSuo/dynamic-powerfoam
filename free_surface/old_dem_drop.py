"""The old DEM on the jack (2026-10-24): the simulator of the old identification pipeline (experiments/springgaus_train.py,
its latest configuration as output/pacnerf_eval/clean_cm_scan/elastic0.log prints it: true-volume bonded model
(nodalvol, kinematics exact), surface completion, rotation constraint with explicit damping, clean volume model (walls,
no tetrahedra), consistent mass by its precomputed inverse (--mass consistent; default lumped: its dense 3n x 3n
float64 matrices, Cholesky factor and inverse take > 12 GB at the jack's 7478 bodies), PFC ground contact, float32, the
stable-step substep rule substeps_now), dropped as the new model's drop (skin_drop_check.py): at rest, its lowest contact-sphere bottom --height
above the ground plane through the origin, gravity -9.8 along --up, E 23008.38 Pa, nu 0.3008 (per unit density), every
parameter multiplier 1 (theta 0: bond dashpots 0.3 of critical, the old default, the friction coefficient friction0
0.5). Records every 1/--fps s: centre of mass along up and its velocity (central differences of the frame positions),
the contact force / weight from the centre of mass's acceleration (1 + a_up / g), the lowest body bottom; the wall
time of the rollout (after the build and a one-frame warm-up that captures the frame graph), the substep size and
count. The DEM folder needs geometry.npz / cells.npz of the foam run --run (the surface completion recomputes the
cells from it): our jack in that layout by old_dem_layout.py.
  python free_surface/old_dem_drop.py --up y"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import warp as wp

from experiments.foam_dem_warp import WarpFoam, explicit_damping_rate, rollout
from experiments.foam_surface_completion import SurfaceCompletion
from experiments.pacnerf_dem_forward import build
from prototypes.kb_derivation.clean_model import CleanTerms, consistent_mass_data, frequency_ratio as clean_frequency_ratio, \
    np_data as clean_np_data, slaved_rot_invmass



def fix_singular_gradients(ct):
    """CleanTerms' cell gradient fits F_B from the walls in front of the site only (h_B > 0) plus the completion
    springs; a body whose site lies outside its cell can be left with a rank-deficient moment (our jack keeps its tiny
    cells, which the old DEMs merged: body 2856, 0.003 of the median volume, 2 of 4 walls in front, interior, no
    springs: F at rest is not I and the polar fails). For the bodies whose M_B + M_air is singular to float64 (smallest
    eigenvalue <= eps x the largest) the fit also takes the walls behind the site, weighted |h_B| (any positive weights
    keep S_B = F M_B exact for affine maps, objective); the volume (front-wall pyramids, Vrest) is unchanged. Patches ct
    in place (M, MN_inv, Mt_inv, F; gradient weights ci, cj for cl_ci / cl_cj); returns the bodies."""
    e = torch.linalg.eigvalsh(ct.M+ct.Mair)
    bad = e[:, 0] <= torch.finfo(e.dtype).eps*e[:, 2]
    ci, cj = ct.wi*ct.A*ct.hi/ct.L, ct.wj*ct.A*ct.hj/ct.L
    if bool(bad.any()):
        ci = torch.where(bad[ct.i], ct.A*ct.hi.abs()/ct.L, ci); cj = torch.where(bad[ct.j], ct.A*ct.hj.abs()/ct.L, cj)
        nn = ct.nrm[:, :, None]*ct.nrm[:, None, :]
        M = torch.zeros_like(ct.M).index_add(0, ct.i, (ci*ct.L)[:, None, None]*nn).index_add(0, ct.j, (cj*ct.L)[:, None, None]*nn)
        ct.M, ct.MN_inv, ct.Mt_inv = M, torch.linalg.inv(ct.nbr_sum(M)), torch.linalg.inv(M+ct.Mair)

        def F(x):
            d = x[ct.j]-x[ct.i]
            on = lambda c: c[:, None, None]*d[:, :, None]*ct.nrm[:, None, :]
            S = torch.zeros(ct.n, 3, 3, dtype=x.dtype, device=x.device).index_add(0, ct.i, on(ci)).index_add(0, ct.j, on(cj))
            return (S+ct.nbr_sum(S)@ct.MN_inv@ct.Mair)@ct.Mt_inv
        ct.F = F
    ct.ci, ct.cj = ci, cj
    return torch.nonzero(bad).flatten().tolist()


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--dem', type=Path, default=Path('output/round01_pruned_exact/old_dem_layout/dem'))
    a.add_argument('--run', type=Path, default=Path('output/round01_pruned_exact/old_dem_layout/run'))
    a.add_argument('--mass', choices=('lumped', 'consistent'), default='lumped')
    a.add_argument('--up', choices=('y', 'z'), default='y')
    a.add_argument('--E', type=float, default=23008.38)
    a.add_argument('--nu', type=float, default=.3008)
    a.add_argument('--height', type=float, default=.02)
    a.add_argument('--duration', type=float, default=.5)
    a.add_argument('--fps', type=float, default=240.)
    a.add_argument('--frames', type=int, default=0, help='frames to simulate (0: --duration x --fps)')
    a.add_argument('--tag', default='')
    a.add_argument('--out', type=Path, default=Path('output/round01_pruned_exact'))
    args = a.parse_args()
    wp.init()
    dev, f32, f64 = 'cuda', torch.float32, torch.float64
    t_build = time.time()
    geometry = dict(np.load(args.dem/'geometry.npz'))
    kw = dict(nodalvol=True, kinematics='exact')                                   # --truevol 1 --kinematics exact
    model = build(geometry, args.E, args.nu, dev, 'noslip', .25, .3, f32, **kw)
    up = np.array([0., 1., 0.] if args.up == 'y' else [0., 0., 1.])
    model.set_environment(gravity=list(-9.8*up), ground_point=[0., 0., 0.], ground_normal=list(up),
                          contact_radius=model.contact_radius.cpu().numpy(), ground_mode='noslip')
    print('model: %d bodies, %d bonds, mass %.6f (per unit density), E %.6g nu %.4f, float32' % (
        model.n, len(model.i), float(model.mass.sum()), args.E, args.nu), flush=True)
    # surface completion, rotation constraint (cell-average weights), clean volume model, consistent mass
    sc = SurfaceCompletion(args.run, args.dem, model)
    sc.rot_coupling = 0.
    cache = args.out/'old_dem_cache'; cache.mkdir(parents=True, exist_ok=True)
    sc.cell_weights(cache=cache/('cell_weights_%s.npz' % str(args.dem).strip('/').replace('/', '_')))
    completion = sc.np_data
    m64 = build(geometry, args.E, args.nu, dev, 'noslip', .25, .3, f64, **kw)
    for a_ in ('kinematics', 'nonlinear_bonds', 'ogden_alpha', 'mu_ref'):
        if hasattr(model, a_):
            setattr(m64, a_, getattr(model, a_))
    sc64 = SurfaceCompletion(args.run, args.dem, m64, verbose=False)
    ct = CleanTerms(m64, sc64, geometry)
    fixed = fix_singular_gradients(ct)
    F0 = ct.F(m64.X)
    print('cell gradient: bodies with a singular wall moment, fitted with their walls behind the site too: %s; F at rest |F - I| max %.1e' % (
        fixed, float((F0-torch.eye(3, dtype=f64, device=dev)).abs().max())), flush=True)
    completion = dict(completion); completion.update(clean_np_data(ct))
    completion['cl_ci'], completion['cl_cj'] = ct.ci.detach().cpu().numpy(), ct.cj.detach().cpu().numpy()
    ratio, comp_w = clean_frequency_ratio(m64, ct)
    if args.mass == 'lumped':
        print('clean volume model (walls, no tetrahedra), lumped mass; highest frequency x%.4f (%.4g -> %.4g rad/s), substeps scaled by it' % (ratio, *comp_w), flush=True)
    sp_ = .004 if args.mass == 'consistent' else None                              # consistent_mass_data's default; halved until
    while sp_ is not None:                                                         # every body holds sample points (its assert)
        try:
            cmd, cm_M = consistent_mass_data(ct, spacing=sp_)
            break
        except AssertionError:
            sp_ /= 2
            print('consistent mass: a body without sample points, spacing -> %.4g m' % sp_, flush=True)
    if args.mass == 'consistent':
        print('consistent mass: sample spacing %.4g m' % sp_, flush=True)
        completion.update(cmd)
        cm_L = torch.linalg.cholesky(cm_M)
        x0c = m64.initial_state()[0]; thc = torch.zeros(m64.n, 9, dtype=f64, device=dev)
        gen = torch.Generator(device='cpu').manual_seed(0)
        yc = torch.randn(3*m64.n, generator=gen, dtype=f64).to(dev); wc = 0.
        for _ in range(60):                                                            # highest frequency with the consistent mass
            yc = yc/torch.sqrt(yc@(cm_M@yc))
            zc = torch.zeros(3*m64.n, dtype=f64, device=dev, requires_grad=True)
            Ec = ct.energy(thc, x0c+zc.view(-1, 3), smooth_polar=True)[0]
            gz, = torch.autograd.grad(Ec, zc, create_graph=True)
            Ky, = torch.autograd.grad(gz@yc, zc)
            wc = float(yc@Ky)
            yc = torch.cholesky_solve(Ky[:, None], cm_L).squeeze(1).detach()
        w_cm = np.sqrt(max(wc, 0.))
        print('consistent mass: diag(M)^-1 M spectrum %.3f..%.3f; highest omega %.0f rad/s (lumped %.0f), substeps x %.3f' % (
            cmd['cm_lo'], cmd['cm_hi'], w_cm, comp_w[1], w_cm/comp_w[1]), flush=True)
        ratio *= w_cm/comp_w[1]
        completion['cm_Minv'] = torch.cholesky_inverse(cm_L).float().cpu().numpy()
        print('clean volume model (walls, no tetrahedra); highest frequency x%.4f (%.4g -> %.4g rad/s), substeps scaled by it' % (ratio, *comp_w), flush=True)
    ncol = 8                                                                       # pfc: columns 0-7 (no contact field)
    wf = WarpFoam(model, contact='pfc', layer_dx=.002, friction0=.5, restitution0=.3, contact_scale=1., visco=False,
                  visco_cols=(ncol-2, ncol-1), ogden_col=-1, contact_offset_col=-1, implicit_damping=False, graphs=True,
                  faces=None, completion=completion)
    wf.pfc_stick, wf.spin_follow, wf.mc_scale, wf.rot_coupling = 0, 0., 0., 0.
    wf.rot_constraint, wf.surface_contact, wf.sc_iterations, wf.kv_damping = True, False, 16, False
    if args.mass == 'consistent':
        wf.enable_consistent_mass(0)
    th = torch.zeros(model.n, ncol, dtype=f32, device=dev)
    x0, q0, _, _ = model.initial_state()
    damp_unit = explicit_damping_rate(wf, x0, q0, th.clone())
    print('rotation constraint: explicit damping rate at multiplier 1 %.4g /s' % damp_unit, flush=True)
    # substeps_now (the rotation-constraint branch with the pfc contact term)
    R_c = model.contact_radius.double()
    inv_rot = slaved_rot_invmass(ct, ct.model.mass.double()).to(R_c.device)
    pfc_kA = np.pi*R_c*(1/model.mass.double()+R_c**2*inv_rot)
    torch.cuda.empty_cache()
    th7 = th[:, :7].to(model.dtype)
    dt_m = model.stable_dt(th7, safety=.5, include_damping=False)/ratio
    dt_d = .5*2/max(damp_unit*float(torch.exp(th[:, 6]).max()), 1e-30)
    dt_ = min(dt_m, dt_d)
    w_c = float(torch.sqrt((model.k_v*torch.exp(th7[:, 4].double())+2*model.mu_ref*torch.exp(th7[:, 5].double()))*pfc_kA).max())
    w_m = 2*.5/dt_
    dt_ = 2*.5/np.sqrt(w_m**2+w_c**2)
    sub = int(np.ceil(1/args.fps/dt_))
    dt = 1/args.fps/sub
    frames = args.frames or int(round(args.duration*args.fps))
    print('step: elastic limit %.4g s, damping limit %.4g s, with the contact (omega %.4g, model %.4g rad/s) %.4g s; %d substeps per 1/%g s '
          '(dt %.4g s), %d frames = %d substeps' % (dt_m, dt_d, w_c, w_m, dt_, sub, args.fps, dt, frames, frames*sub), flush=True)
    # at rest, the lowest contact-sphere bottom --height above the ground (springgaus_train energy --drop-height)
    upt = torch.tensor(up, dtype=f32, device=dev)
    low = float(((x0@upt)-model.contact_radius.to(f32)).min())
    xs = (x0-(low-args.height)*upt).contiguous()
    state = (xs, q0.contiguous(), torch.zeros_like(xs), torch.zeros_like(xs))
    print('build %.0f s' % (time.time()-t_build), flush=True)
    with torch.no_grad():
        rollout(wf, th, state, dt, sub, 1)                                         # warm-up: kernels, the frame graph
        torch.cuda.synchronize(); t0 = time.time()
        pos, _ = rollout(wf, th, state, dt, sub, frames)
        torch.cuda.synchronize(); wall = time.time()-t0
    pos = pos.double()
    m = model.mass.double()
    up64 = torch.tensor(up, dtype=f64, device=dev)
    h = ((pos@up64)*m).sum(1)/m.sum()                                              # centre of mass along up per frame
    bottom = ((pos@up64)-model.contact_radius.double()).min(1).values
    t = torch.arange(frames+1, dtype=f64, device=dev)/args.fps
    v = torch.zeros_like(h); v[1:-1] = (h[2:]-h[:-2])*args.fps/2; v[-1] = (h[-1]-h[-2])*args.fps
    acc = torch.zeros_like(h); acc[1:-1] = (h[2:]-2*h[1:-1]+h[:-2])*args.fps**2
    F = 1+acc/9.8
    lift = torch.nonzero((t > t[torch.argmax(F)]) & (bottom > 0)).flatten()
    res = dict(bodies=model.n, substeps_per_frame=sub, dt=dt, frames=frames, fps=args.fps, wall_s=wall, ms_per_substep=1e3*wall/(frames*sub),
               max_rise_speed=float(v.max()), t_max_rise=float(t[torch.argmax(v)]), deepest_com=float(h.min()), t_deepest=float(t[torch.argmin(h)]),
               com_start=float(h[0]), contact_peak_weights=float(F.max()), t_peak=float(t[torch.argmax(F)]),
               lift_off=float(t[lift[0]]) if len(lift) else None, torch_peak_GB=torch.cuda.max_memory_allocated()/1e9)
    print(json.dumps(res), flush=True)
    np.savez(args.out/('old_dem_drop%s.npz' % args.tag), t=t.cpu().numpy(), com=h.cpu().numpy(), v=v.cpu().numpy(), F=F.cpu().numpy(),
             bottom=bottom.cpu().numpy(), pos=pos.float().cpu().numpy(), **{k_: np.asarray(v_ if v_ is not None else np.nan) for k_, v_ in res.items()})


if __name__ == '__main__':
    main()
