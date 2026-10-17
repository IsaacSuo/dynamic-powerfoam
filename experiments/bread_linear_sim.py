"""Official bread: native per-cell small-strain bonded dynamics and rendering.

Every retained cell has six independent mechanical DOFs. No coarse particles,
interpolated driver, retraining, fracture, or changing radii. This pilot treats
bounded geometric support as solid, with generator-lumped masses and isotropic
proxy inertia. It is not a calibrated bread material or finite-strain solver.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial.transform import Rotation

from experiments.bonded_dem_linear_gpu import LinearBondsGPU, checks
from experiments.powersim_bread_demo import ATTRIBUTES, load_scene, camera_from_bundle, render_image


def prune_check(model, cameras, state, keep, out):
    from experiments.full_model_rigid_probe import compare, render
    views = [0, 56, 99]
    reference = [render(model, camera_from_bundle(cameras, i)) for i in views]
    for name in ATTRIBUTES:
        value = state[name][keep].cuda().contiguous()
        setattr(model, name, torch.nn.Parameter(value, requires_grad=False))
    model.rebuild_adjacency()
    results = []
    for index, baseline in zip(views, reference):
        image, result = render_image(model, camera_from_bundle(cameras, index), f'Reference after zero-volume pruning | view {index}')
        image.save(out/f'pruned_reference_{index:03d}.png')
        metric = compare(baseline, result)
        if metric['rgb_mae_foreground'] > .002 or metric['alpha_mae'] > .001:
            raise RuntimeError(f'Pruning changed reference appearance too much: {metric}')
        results.append(dict(view=index, **metric))
    (out/'prune_check.json').write_text(json.dumps(results, indent=2)+'\n')
    return results


def prepare(geometry):
    keep = geometry['keep']
    mapping = np.full(len(keep), -1, dtype=int); mapping[keep] = np.arange(keep.sum())
    p, r, mass = geometry['points'][keep], geometry['radii'][keep], geometry['volumes'][keep]
    pairs = mapping[geometry['pairs']]
    graph = sparse.coo_matrix((np.ones(2*len(pairs)),
                              (np.r_[pairs[:, 0], pairs[:, 1]], np.r_[pairs[:, 1], pairs[:, 0]])),
                             shape=(len(p), len(p))).tocsr()
    count, labels = connected_components(graph)
    component_mass = np.bincount(labels, weights=mass)
    main = labels == component_mass.argmax()
    clamp_x = np.quantile(p[main, 0], .1)
    fixed = (p[:, 0] <= clamp_x) | ~main
    loaded = (p[:, 0] >= np.quantile(p[main, 0], .85)) & main
    force = np.zeros((len(p), 6)); force[loaded, 2] = -mass[loaded]/mass[loaded].sum()
    solver = LinearBondsGPU(p, r, pairs, geometry['areas'], geometry['contacts'], mass, fixed)
    # Rayleigh quotient of a smooth clamp-compatible bending trial shape; a
    # time-scale estimate only, not a prescribed displacement or eigenmode.
    length = p[main, 0].max()-clamp_x
    s = np.maximum(p[:, 0]-clamp_x, 0.)/length
    shape = np.zeros((len(p), 6))
    shape[:, 2] = -s*s; shape[:, 4] = 2*s/length
    shape[:, 0] = shape[:, 4]*(p[:, 2]-np.average(p[:, 2], weights=mass))
    shape[fixed] = 0
    q = solver.tensor(shape)
    frequency = float(torch.sqrt((q*solver.stiffness(q)).sum()/(solver.mass*q*q).sum()))
    period = 2*np.pi/frequency
    return solver, force, period, dict(particles=len(p), bonds=len(pairs), components=int(count),
           main_component_mass_fraction=float(component_mass.max()/mass.sum()), fixed_particles=int(fixed.sum()),
           loaded_particles=int(loaded.sum()), period_estimate=period, clamp_x=float(clamp_x),
           material_density=1., mass_domain='bounded geometric support; not optical occupancy')


def simulate(solver, force, period, refinement=1):
    solver.q.zero_(); solver.v.zero_(); solver.kq.zero_()
    steps = 96*refinement; dt = 3*period/steps
    external = solver.tensor(force)
    qs, sample, iterations = [], [], []
    work, worst_balance, max_energy = 0., 0., 0.
    start = time.monotonic()
    for step in range(steps+1):
        t = step*dt; energy = solver.energy()
        worst_balance = max(worst_balance, abs(energy-work)); max_energy = max(max_energy, energy)
        if step % (2*refinement) == 0:
            qs.append(solver.q.cpu().numpy().copy())
            tip = -float((external*solver.q).sum())/float(torch.linalg.vector_norm(external[:, :3], dim=1).sum())
            sample.append([t, tip, energy, work])
        if step == steps:
            break
        tm = (step+.5)*dt
        scale = np.sin(np.pi*tm/period)**2 if tm < period else 0.
        delta_work, info = solver.step(dt, scale*external)
        work += delta_work; iterations.append(info)
        if step % 8 == 0:
            print(f'Bread dynamics refine={refinement}: {step+1}/{steps}, CG={info["iterations"]}, '
                  f'residual={info["relative_residual"]:.2g}, elapsed={time.monotonic()-start:.1f}s', flush=True)
    relative_energy_error = worst_balance/max_energy
    if relative_energy_error > .002:
        raise RuntimeError(f'Work-energy mismatch: {relative_energy_error}')
    return np.array(qs), np.array(sample), dict(steps=steps, dt=dt,
        work_energy_error_over_peak=relative_energy_error,
        max_cg_iterations=max(a['iterations'] for a in iterations),
        max_cg_relative_residual=max(a['relative_residual'] for a in iterations),
        elapsed_seconds=time.monotonic()-start)


def render_motion(model, cameras, state, keep, geometry, qs, samples, period, out):
    from PIL import Image, ImageDraw
    from experiments.bonded_dem_powerfoam import state_from_bodies
    base = {name: state[name][keep].numpy() for name in ATTRIBUTES}
    p0 = geometry['points'][keep]
    images, stats = [], []
    for frame, (q, row) in enumerate(zip(qs, samples)):
        rotations = Rotation.from_rotvec(q[:, 3:]).as_matrix()
        output = state_from_bodies(base, p0+q[:, :3], rotations)
        with torch.no_grad():
            for name in ('points', 'quaternions', 'texel_sv_axis'):
                getattr(model, name).copy_(torch.as_tensor(output[name], device='cuda', dtype=torch.float32))
            model.rebuild_adjacency()
        combined = Image.new('RGB', (1024, 512), 'black')
        for view, index in enumerate((0, 99)):
            label = f'Official bread | linear bonded DEM | t={row[0]:.5f}'
            image, result = render_image(model, camera_from_bundle(cameras, index), label)
            ImageDraw.Draw(image).text((10, 28), 'loading' if row[0] < period else 'free release', fill='white')
            combined.paste(image, (512*view, 0))
            stats.append(dict(frame=frame, view=index, alpha_area=int((result['alpha']>.5).sum())))
        images.append(combined)
        if frame % 8 == 0:
            combined.save(out/f'motion_{frame:03d}.png')
            print(f'Rendering {frame+1}/{len(qs)}', flush=True)
    images[0].save(out/'bread_dem.gif', save_all=True, append_images=images[1:], duration=85, loop=0)
    selected = [0, int(np.argmin(samples[:, 1])), int(np.argmax(samples[:, 1]))]
    sheet = Image.new('RGB', (1024, 512*3), 'black')
    for row, index in enumerate(selected): sheet.paste(images[index], (0, 512*row))
    sheet.save(out/'bread_dem_frames.png')
    (out/'motion_render.json').write_text(json.dumps(dict(selected_frames=selected, frames=len(images),
        source='Original PowerfoamScene.forward; fixed learned radii, local sites, heights, RGB',
        node_rotation='Small-strain DEM theta mapped through exp(theta) into appearance frame/SV axes',
        display='No displacement exaggeration; GIF slowed relative to uncalibrated mechanical time',
        per_frame_views=stats), indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=Path('output/powersim_bread/linear_dem'))
    parser.add_argument('--render-only', action='store_true')
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    with np.load('output/powersim_bread/material_geometry.npz') as data: geometry = dict(data)
    keep = geometry['keep']
    model, cameras, config, state = load_scene()
    pruning = prune_check(model, cameras, state, keep, args.out)
    print('Reference appearance after pruning:', pruning, flush=True)
    if args.render_only:
        with np.load(args.out/'trajectory.npz') as archive:
            qs, samples, period = archive['q'], archive['samples'], float(archive['period'])
    else:
        validation = checks()
        solver, force, period, details = prepare(geometry)
        print('Simulation setup:', details, flush=True)
        coarse, samples_coarse, first = simulate(solver, force, period)
        max_translation = np.linalg.norm(coarse[:, :, :3], axis=2).max()
        max_rotation = np.linalg.norm(coarse[:, :, 3:], axis=2).max()
        # Choose a SMALL applied force using the linear pilot response, and
        # rerun the fine simulation with that actual force. No display scaling.
        force_scale = min(.008/max_translation, .08/max_rotation)
        print('Applied force selected from pilot:', force_scale, 'pilot maxima:', max_translation, max_rotation, flush=True)
        coarse *= force_scale; samples_coarse[:, 1] *= force_scale; samples_coarse[:, 2:] *= force_scale**2
        qs, samples, fine = simulate(solver, force*force_scale, period, 2)
        scale = np.linalg.norm(qs[:, :, :3], axis=2).max()
        difference = float(np.max(np.linalg.norm(qs[:, :, :3]-coarse[:, :, :3], axis=2))/scale)
        if difference > .03:
            raise RuntimeError(f'Temporal refinement disagreement too large: {difference}')
        report = dict(setup=details, gpu_checks=validation, coarse=first, fine=fine,
                      total_peak_force=force_scale, max_translation=float(scale),
                      max_rotation=float(np.linalg.norm(qs[:, :, 3:], axis=2).max()),
                      dt_refinement_max_position_difference_over_amplitude=difference,
                      pruning=pruning, scope='Native-cell small-strain linearized bonded DEM; geometric support solid, no calibrated material, no collision/damage.',
                      source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                        (Path(__file__), Path(__file__).with_name('bonded_dem_linear_gpu.py'), Path(__file__).with_name('bread_material_geometry.py'))})
        (args.out/'mechanics.json').write_text(json.dumps(report, indent=2)+'\n')
        np.savez_compressed(args.out/'trajectory.npz', q=qs, samples=samples, period=period, original_ids=np.flatnonzero(keep))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
        ax[0].plot(samples[:, 0], samples[:, 1], label='fine dt')
        ax[0].plot(samples_coarse[:, 0], samples_coarse[:, 1], '.', label='coarse dt')
        ax[0].axvline(period, color='gray', ls='--'); ax[0].set(title='Bread end load and release', xlabel='time', ylabel='loaded-region z displacement'); ax[0].legend()
        ax[1].plot(samples[:, 0], samples[:, 2], label='energy'); ax[1].plot(samples[:, 0], samples[:, 3], '--', label='external work'); ax[1].legend()
        fig.savefig(args.out/'motion.png', dpi=160); plt.close(fig)
        del solver
        torch.cuda.empty_cache()
        print(json.dumps(report, indent=2), flush=True)
    render_motion(model, cameras, state, keep, geometry, qs, samples, period, args.out)


if __name__ == '__main__':
    main()
