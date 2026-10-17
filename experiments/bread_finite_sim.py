"""Bread with finite-motion parallel-bond histories; other pilot choices fixed."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from experiments.bonded_dem_finite_gpu import FiniteBondsGPU
from experiments.bread_linear_sim import prepare, prune_check
from experiments.powersim_bread_demo import load_scene, ATTRIBUTES, render_image, camera_from_bundle


def render(qs, samples, geometry, period, out):
    from PIL import Image, ImageDraw
    from experiments.bonded_dem_powerfoam import state_from_bodies
    model, cameras, _, state = load_scene()
    keep = geometry['keep']
    prune_check(model, cameras, state, keep, out)
    base = {name: state[name][keep].numpy() for name in ATTRIBUTES}
    p0 = geometry['points'][keep]
    images = []
    for frame, (q, row) in enumerate(zip(qs, samples)):
        output = state_from_bodies(base, p0+q[:, :3], Rotation.from_rotvec(q[:, 3:]).as_matrix())
        with torch.no_grad():
            for name in ('points', 'quaternions', 'texel_sv_axis'):
                getattr(model, name).copy_(torch.as_tensor(output[name], device='cuda', dtype=torch.float32))
            model.rebuild_adjacency()
        combined = Image.new('RGB', (1024, 512), 'black')
        for column, index in enumerate((0, 99)):
            image, _ = render_image(model, camera_from_bundle(cameras, index),
                                    f'Finite-motion bonded DEM | t={row[0]:.5f}')
            ImageDraw.Draw(image).text((10, 28), 'loading' if row[0] < period-1e-10 else 'free release', fill='white')
            combined.paste(image, (512*column, 0))
        images.append(combined)
        if frame % 8 == 0:
            combined.save(out/f'motion_{frame:03d}.png')
            print(f'Finite-motion rendering {frame+1}/{len(qs)}', flush=True)
    images[0].save(out/'bread_dem.gif', save_all=True, append_images=images[1:], duration=85, loop=0)
    indices = [0, int(np.argmin(samples[:, 1])), int(np.argmax(samples[:, 1]))]
    sheet = Image.new('RGB', (1024, 1536), 'black')
    for row, index in enumerate(indices): sheet.paste(images[index], (0, 512*row))
    sheet.save(out/'bread_dem_frames.png')
    (out/'render.json').write_text(json.dumps(dict(frames=len(images), selected_frames=indices,
        renderer='Original PowerfoamScene.forward, current graph rebuilt each frame',
        appearance='Same rigid local appearance transport and fixed radii as linear bread',
        rotations='Accumulated multiplicative world rotation matrices; rotvec is serialization only',
        display_displacement_scale=1), indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refinement', type=int, default=1)
    parser.add_argument('--out', type=Path, default=Path('output/powersim_bread/finite_dem/coarse'))
    parser.add_argument('--limit-steps', type=int, default=0, help='Short integration diagnostic; no full-run claims or rendering')
    parser.add_argument('--render-only', action='store_true')
    parser.add_argument('--no-render', action='store_true')
    parser.add_argument('--resume', type=Path, help='Continue a saved prefix with identical dt and force')
    args = parser.parse_args()
    if args.refinement < 1: parser.error('refinement must be positive')
    args.out.mkdir(parents=True, exist_ok=True)
    with np.load('output/powersim_bread/material_geometry.npz') as a: geometry = dict(a)
    if args.render_only:
        with np.load(args.out/'trajectory.npz') as a:
            render(a['q'], a['samples'], geometry, float(a['period']), args.out)
        return
    baseline = json.loads(Path('output/powersim_bread/linear_dem_10x/mechanics.json').read_text())
    linear, force, period, setup = prepare(geometry)
    keep = geometry['keep']
    p, r, m = geometry['points'][keep], geometry['radii'][keep], geometry['volumes'][keep]
    pairs, fixed = linear.edges.cpu().numpy(), linear.fixed.cpu().numpy()
    solver = FiniteBondsGPU(p, r, pairs, geometry['areas'], m, fixed, contacts=geometry['contacts'])
    # Confirm this upgrade preserves reference stiffness, masses and inertia.
    assert torch.equal(solver.mass, linear.mass)
    for key in ('kn', 'ks', 'kb', 'kt', 'n', 'ri', 'rj'):
        assert torch.allclose(getattr(solver, key), getattr(linear, key), rtol=1e-12, atol=1e-14), key
    from experiments.bonded_dem_tangent_warp import WarpTangent
    solver.stiffness = WarpTangent(solver)
    del linear; torch.cuda.empty_cache()
    applied = solver.tensor(force*baseline['total_peak_force'])
    from experiments.bonded_dem_adaptive import AdaptiveIntervals
    adaptive = AdaptiveIntervals(solver)
    def load(t):
        return applied*(np.sin(np.pi*t/period)**2 if t < period else 0.)
    steps = 96*args.refinement
    dt = 3*period/steps
    count = min(steps, args.limit_steps) if args.limit_steps else steps
    qs, samples, diagnostics = [], [], []
    work, worst_balance, peak_energy = 0., 0., 0.
    first_step, previous_elapsed = 0, 0.
    if args.resume:
        prior = json.loads((args.resume/'mechanics.json').read_text())
        assert np.isclose(prior['dt'], dt) and prior['total_peak_force'] == baseline['total_peak_force']
        digest = hashlib.sha256(Path(__file__).with_name('bonded_dem_finite_gpu.py').read_bytes()).hexdigest()
        assert prior['source_sha256']['bonded_dem_finite_gpu.py'] == digest, 'Resume needs identical material/solver code'
        with np.load(args.resume/'final_state.npz') as a:
            for key in ('x', 'rotation', 'fn', 'fs', 'mt', 'mb', 'normal', 'offset'):
                setattr(solver, key, solver.tensor(a[key]))
            solver.v = solver.tensor(a['velocity'])
        with np.load(args.resume/'trajectory.npz') as a:
            qs = list(a['q'][:-1]); samples = list(a['samples'][:-1])
            work = float(a['samples'][-1, 3])
            peak_energy = float(a['samples'][:, 2].max())
        first_step = prior['steps']; solver.time = first_step*dt
        worst_balance = prior['work_energy_error_over_peak']*peak_energy
        previous_elapsed = prior['elapsed_seconds']
        # Prefix aggregate retained without inventing individual substep data.
        diagnostics = [dict(subdivision_depth=prior['max_subdivision_depth'],
            nonlinear_iterations=prior['max_nonlinear_iterations'],
            nonlinear_relative_residual=prior['max_nonlinear_relative_residual'],
            local_pose_defect=prior['max_local_pose_defect'])]
        print(f'Resuming after {first_step} steps from {args.resume}', flush=True)
    start = time.monotonic()
    for step in range(first_step, count+1):
        energy = solver.energy()
        worst_balance = max(worst_balance, abs(energy-work)); peak_energy = max(peak_energy, energy)
        if step % (2*args.refinement) == 0 or step == count:
            rotation = solver.rotation.cpu().numpy()
            q = np.c_[solver.x.cpu().numpy()-p, Rotation.from_matrix(rotation).as_rotvec()]
            q[fixed] = 0.  # Only eliminate serialization roundoff at stationary clamps.
            assert torch.equal(solver.x[solver.fixed], solver.p[solver.fixed])
            qs.append(q)
            tip = float((force[:, 2]*q[:, 2]).sum()/-np.abs(force[:, 2]).sum())
            samples.append([solver.time, tip, energy, work])
        if step == count: break
        increment, info = adaptive.advance(dt, load)
        work += increment; diagnostics.extend(info)
        monitor = dict(step=step+1, time=solver.time, energy=solver.energy(), work=work,
                       max_velocity=float(torch.linalg.vector_norm(solver.v[:, :3], dim=1).max()),
                       max_omega=float(torch.linalg.vector_norm(solver.v[:, 3:], dim=1).max()),
                       max_displacement=float(torch.linalg.vector_norm(solver.x-solver.p, dim=1).max()),
                       substeps=len(info))
        with (args.out/'progress.jsonl').open('a') as stream:
            stream.write(json.dumps(monitor)+'\n')
        if step % 4 == 0:
            print(f'Finite DEM r={args.refinement}: {step+1}/{count}; '
                  f'nonlinear={max(x["nonlinear_iterations"] for x in info)}, '
                  f'CG={sum(x["cg_iterations"] for x in info)}, substeps={len(info)}, '
                  f'elapsed={time.monotonic()-start:.1f}s', flush=True)
    q, samples = np.asarray(qs), np.asarray(samples)
    assert np.isfinite(q).all()
    rotations = solver.rotation
    orthogonality = float(abs(rotations.transpose(1, 2)@rotations-torch.eye(3, device='cuda')).max())
    report = dict(setup=setup, complete=(count == steps), steps=count, dt=dt,
        total_peak_force=baseline['total_peak_force'], actual_substeps=len(diagnostics)+(prior['actual_substeps']-1 if args.resume else 0),
        max_subdivision_depth=max(x['subdivision_depth'] for x in diagnostics),
        max_nonlinear_iterations=max(x['nonlinear_iterations'] for x in diagnostics),
        max_nonlinear_relative_residual=max(x['nonlinear_relative_residual'] for x in diagnostics),
        max_local_pose_defect=max(x['local_pose_defect'] for x in diagnostics),
        work_energy_error_over_peak=worst_balance/max(peak_energy, 1e-30),
        max_translation=float(np.linalg.norm(q[:, :, :3], axis=2).max()),
        max_rotation=float(np.linalg.norm(q[:, :, 3:], axis=2).max()),
        loaded_z_range=[float(samples[:, 1].min()), float(samples[:, 1].max())],
        orientation_orthogonality_max_error=orthogonality,
        elapsed_seconds=time.monotonic()-start+previous_elapsed,
        resumed_from=str(args.resume) if args.resume else None,
        scope='Finite-motion history-dependent elastic parallel bonds; unchanged support mass, inertia, moduli, fixed disconnected cells and appearance transport. No damage/contact.',
        contact_rule='Earlier classical ball contact plus transported reference face-centroid offset; identical reference levers to previous bread.',
        source_sha256={name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in
            ('bread_finite_sim.py', 'bonded_dem_finite_gpu.py', 'bonded_dem_linear_gpu.py',
             'bonded_dem_tangent_warp.py', 'bonded_dem_adaptive.py')})
    (args.out/'mechanics.json').write_text(json.dumps(report, indent=2)+'\n')
    np.savez_compressed(args.out/'trajectory.npz', q=q, samples=samples, period=period, original_ids=np.flatnonzero(keep))
    np.savez_compressed(args.out/'final_state.npz', x=solver.x.cpu().numpy(), velocity=solver.v.cpu().numpy(),
        rotation=solver.rotation.cpu().numpy(), fn=solver.fn.cpu().numpy(), fs=solver.fs.cpu().numpy(),
        mt=solver.mt.cpu().numpy(), mb=solver.mb.cpu().numpy(), normal=solver.normal.cpu().numpy(), offset=solver.offset.cpu().numpy())
    print(json.dumps(report, indent=2), flush=True)
    if report['work_energy_error_over_peak'] > .002 or orthogonality > 1e-10:
        raise RuntimeError('Energy or orientation check failed; inspect saved diagnostics')
    del solver; torch.cuda.empty_cache()
    if count == steps and not args.no_render:
        render(q, samples, geometry, period, args.out)


if __name__ == '__main__':
    main()
