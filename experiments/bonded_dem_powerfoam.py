"""Stage 3: load/release bonded bodies and render through PowerfoamScene.forward.

Hand-authored volumetric scene, fixed geometric radii, no image optimization.
Mechanical bonds keep their initial identities; rendering rebuilds the Cech
graph. Assigned masses are synthetic box quadrature, not occupied-cell masses.
No fracture/collision, material-volume guarantee, or learned reconstruction.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from scipy.ndimage import binary_fill_holes
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from experiments.bonded_dem_core import ParallelBonds
from experiments.bonded_dem_dynamics import BondedDynamics, mass_vector, reference_dt_bound, reference_stiffness
from experiments.bonded_dem_material import specimen, solve_prescribed


def setup(resolution=3):
    s = specimen(resolution, length=3.)
    x, h = s['x'], s['h']
    # .9h covers regular-grid interstices (unlike the .775h stage-2 proxies).
    # Choose this once in the initial scene, and use it for BOTH bond candidates
    # and rendering. It never changes during motion. Stiffness rules unchanged.
    radii = np.full(len(x), .9*h)
    pairs = cKDTree(x).query_pairs(1.8*h, output_type='ndarray')
    lengths = np.linalg.norm(x[pairs[:, 1]]-x[pairs[:, 0]], axis=1)
    bonds = ParallelBonds(x, pairs, radii, .25*h, 1200/lengths, 480/lengths)
    inertia = .4*s['mass']*radii**2
    fixed, tip = np.isclose(x[:, 0], 0), np.isclose(x[:, 0], 3.)
    weights = s['mass'][tip]/s['mass'][tip].sum()
    load = np.zeros((len(x), 6)); load[tip, 1] = -weights
    k = reference_stiffness(bonds)
    q, _ = solve_prescribed(k, np.repeat(fixed[:, None], 6, axis=1), np.zeros_like(load), load)
    compliance = -weights@q.reshape(-1, 6)[tip, 1]
    free = np.flatnonzero(~np.repeat(fixed, 6))
    mass = mass_vector(s['mass'], inertia)
    eigenvalue = eigh(k[free][:, free].toarray(), np.diag(mass[free]),
                     subset_by_index=[0, 0], eigvals_only=True)[0]
    period = 2*np.pi/np.sqrt(eigenvalue)
    return dict(s, bonds=bonds, inertia=inertia, radii=radii, fixed=fixed, tip=tip,
                tip_weights=weights, period=period, force_magnitude=.18/compliance,
                dt_bound=reference_dt_bound(k, s['mass'], inertia, fixed))


def simulate(s, refinement=1, frames=81):
    bonds = copy.deepcopy(s['bonds'])
    body = BondedDynamics(bonds, s['mass'], s['inertia'], s['fixed'])
    duration, release = 3*s['period'], s['period']
    base_steps = int(np.ceil(duration/(.22*s['dt_bound']*(frames-1))))*(frames-1)
    steps = base_steps*refinement; dt = duration/steps
    force = np.zeros_like(bonds.x)
    force[s['tip'], 1] = -s['force_magnitude']*s['tip_weights']
    positions, rotations, times, samples = [], [], [], []
    work, max_balance, max_energy = 0., 0., 0.
    release_energies = []
    for step in range(steps+1):
        t = step*dt
        energy = body.energy()
        max_energy = max(max_energy, energy)
        max_balance = max(max_balance, abs(energy-work))
        if t >= release:
            release_energies.append(energy)
        if step % (steps//(frames-1)) == 0:
            positions.append(bonds.x.copy()); rotations.append(body.rotation.copy()); times.append(t)
            tip = s['tip_weights']@(bonds.x[s['tip'], 1]-s['x'][s['tip'], 1])
            samples.append([t, tip, energy, work])
        if step == steps:
            break
        tm = (step+.5)*dt
        fraction = np.sin(np.pi*tm/release)**2 if tm < release else 0.
        external = fraction*force
        old = bonds.x.copy()
        body.step(dt, force_external=external)
        work += float(np.sum(external*(bonds.x-old)))
        if step % 1500 == 0:
            print(f'Dynamics refinement={refinement}: {step}/{steps}', flush=True)
    assert np.all(bonds.intact)
    assert np.array_equal(bonds.x[s['fixed']], s['x'][s['fixed']])
    relative_balance = max_balance/max_energy
    energy_span = (max(release_energies)-min(release_energies))/np.mean(release_energies)
    if relative_balance > .01 or energy_span > .01:
        raise AssertionError(f'Work/energy verification failed: {relative_balance}, {energy_span}')
    trajectory = dict(reference=s['x'], radii=s['radii'], fixed=s['fixed'], pairs=bonds.pairs,
                      times=np.array(times), positions=np.array(positions), rotations=np.array(rotations),
                      samples=np.array(samples), release_time=np.array(release))
    metrics = dict(steps=steps, dt=dt, particles=len(bonds.x), bonds=len(bonds.pairs),
                   force_magnitude=s['force_magnitude'], load_duration=release,
                   tip_displacement_range=[float(np.min(np.array(samples)[:, 1])), float(np.max(np.array(samples)[:, 1]))],
                   max_work_energy_error_over_peak_energy=relative_balance,
                   post_release_relative_energy_span=float(energy_span), all_bonds_intact=True)
    return trajectory, metrics


def appearance(reference, radii, fixed):
    n = len(reference)
    frame = Rotation.from_euler('y', 90, degrees=True).as_matrix()
    uv = np.column_stack([np.tile([-.6, -.2, .2, .6], 2), np.repeat([-.4, .4], 4)])
    axes = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0],
                     [0, 0, 1], [0, 0, -1], [1, 1, 1], [-1, 1, 1]], dtype=float)
    axes /= np.linalg.norm(axes, axis=1)[:, None]
    color = np.where((np.floor(reference[:, 0]*3+1e-6).astype(int)%2)[:, None],
                     np.array([.87, .48, .13]), np.array([.12, .55, .79]))
    color[fixed] = [.48, .48, .5]
    rgb = np.broadcast_to((color-.5)[:, None, None, :], (n, 8, 8, 3)).copy()
    rgb += .04*np.sin(5*uv[:, 0])[None, :, None, None]
    rgb += .025*axes[None, None, :, 1, None]
    return dict(points=reference.copy(), radii=radii+np.log(-np.expm1(-100*radii))/100,
                quaternions=np.broadcast_to(np.roll(Rotation.from_matrix(frame).as_quat(), 1), (n, 4)).copy(),
                density=np.full(n, 60.), texel_sites=np.tile(uv, (n, 1, 1)),
                texel_height=np.tile(.6+.025*np.sin(4*uv[:, 0]), (n, 1)),
                texel_sv_axis=np.broadcast_to(3*axes, (n, 8, 8, 3)).reshape(n, 8, 24).copy(),
                texel_sv_rgb=rgb.reshape(n, 8, 24))


def state_from_bodies(base, positions, rotations):
    frame0 = Rotation.from_quat(np.roll(base['quaternions'], -1, axis=1)).as_matrix()
    frame = frame0@rotations.transpose(0, 2, 1)
    axes = base['texel_sv_axis'].reshape(-1, 8, 8, 3)
    rotated_axes = np.einsum('nab,nsdb->nsda', rotations, axes)
    return dict(base, points=positions, quaternions=np.roll(Rotation.from_matrix(frame).as_quat(), 1, axis=1),
                texel_sv_axis=rotated_axes.reshape(-1, 8, 24))


def render_trajectory(data, out):
    import torch
    import warp as wp
    from PIL import Image, ImageDraw
    from powerfoam.camera import TorchCamera
    from experiments.single_cell_shear import make_model
    from experiments.full_model_rigid_probe import tensor, load_state, render, compare, transformed
    if not torch.cuda.is_available():
        raise RuntimeError('The original Power Foam renderer requires CUDA')
    wp.init(); model = make_model(pinhole=True)
    base = appearance(data['reference'], data['radii'], data['fixed'])
    center = np.array([1.5, .4, .5])

    def camera(offset):
        eye = center+np.asarray(offset)
        forward = -np.asarray(offset)/np.linalg.norm(offset)
        right = np.cross(forward, [0., 1., 0.]); right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        cam = TorchCamera(tensor(eye), tensor(.285*right), tensor(.178125*up), 512, 320)
        cam.ray_maps = cam._build_pinhole_ray_maps()
        return cam

    cameras = [camera([0., 0., 8.]), camera([2., 1.4, 8.])]
    load_state(model, base); rest = render(model, cameras[0])
    # Adapter coordinate test: transform the entire scene AND the camera.
    rotation = Rotation.from_euler('xyz', [17., -13., 21.], degrees=True).as_matrix()
    shift = np.array([.3, -.1, .2])
    rotated = state_from_bodies(base, base['points']@rotation.T+shift,
                               np.broadcast_to(rotation, (len(base['points']), 3, 3)))
    expected = transformed(base, rotation, shift)
    axis_error = float(np.max(np.abs(rotated['texel_sv_axis']-expected['texel_sv_axis'])))
    assert axis_error < 1e-12
    old_cam = cameras[0]
    rotated_cam = TorchCamera(tensor(rotation@old_cam.eye.cpu().numpy()+shift),
                             tensor(rotation@old_cam.right.cpu().numpy()),
                             tensor(rotation@old_cam.up.cpu().numpy()), 512, 320)
    rotated_cam.ray_maps = rotated_cam._build_pinhole_ray_maps()
    load_state(model, rotated); rigid = compare(rest, render(model, rotated_cam))
    assert rigid['rgb_mae_foreground'] < 1e-3 and rigid['alpha_mae'] < 1e-3, rigid
    images, metrics, previous = [], [], [None, None]
    for index, (p, r, time) in enumerate(zip(data['positions'], data['rotations'], data['times'])):
        load_state(model, state_from_bodies(base, p, r))  # includes actual adjacency rebuild
        assert np.max(np.abs(model.get_radii().detach().cpu().numpy()-data['radii'])) < 1e-6
        combined = Image.new('RGB', (1024, 320), 'white')
        frame_metrics = []
        for view, cam in enumerate(cameras):
            result = render(model, cam)
            assert all(np.isfinite(v).all() for v in result.values())
            assert result['alpha'].max() > .95
            mask = result['alpha'] > .5
            holes = binary_fill_holes(mask) & ~mask
            temporal = None if previous[view] is None else float(np.abs(result['rgb']-previous[view]).mean())
            previous[view] = result['rgb'].copy()
            frame_metrics.append(dict(alpha_area=int(mask.sum()), enclosed_background_pixels=int(holes.sum()),
                                      adjacent_frame_rgb_mae=temporal))
            rgb = result['rgb']+(1-result['alpha'][..., None])*np.array([.96, .97, .98])
            image = Image.fromarray(np.uint8(np.clip(rgb, 0, 1)*255+.5))
            draw = ImageDraw.Draw(image)
            phase = 'load pulse' if time < float(data['release_time']) else 'free rebound'
            draw.text((12, 10), f'Bonded DEM -> original Power Foam | t={time:.3f}', fill=(20, 20, 20))
            draw.text((12, 28), f'{phase} | fixed radii | '+('front' if view == 0 else 'oblique'), fill=(20, 20, 20))
            combined.paste(image, (512*view, 0))
        images.append(combined); metrics.append(frame_metrics)
        if index % 10 == 0:
            combined.save(out/f'frame_{index:03d}.png')
            print(f'Original renderer: {index+1}/{len(data["times"])} frames', flush=True)
    images[0].save(out/'powerfoam.gif', save_all=True, append_images=images[1:], duration=65, loop=0)
    tip = data['samples'][:, 1]
    selected = [0, int(np.argmin(tip)), int(np.argmax(tip))]
    sheet = Image.new('RGB', (1024, 320*3), 'white')
    for row, idx in enumerate(selected):
        sheet.paste(images[idx], (0, row*320))
    sheet.save(out/'powerfoam_frames.png')
    report = dict(renderer='Original PowerfoamScene.forward: detail sites/heights + SV + rasterizer',
                  frames=len(images), views=2, frame_duration_ms=65, selected_frame_indices=selected,
                  rigid_coordinate_check=rigid, directional_axis_adapter_error=axis_error,
                  optical_density=60, initial_detail_height='0.6 + 0.025 sin(4u), multiplied by radius',
                  identity='Synthetic fixed body IDs; current rendering graph rebuilt every frame',
                  radius_update='None', per_frame_views=metrics,
                  note='Pixel holes are only enclosed alpha<0.5 regions; temporal RGB changes also include ordinary motion. Neither measures material transport truth.')
    (out/'render.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


def motion_plot(data, coarse, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    t, tip, energy, work = data['samples'].T
    axes[0].plot(t, tip, label='fine dt'); axes[0].plot(coarse['times'], coarse['samples'][:, 1], '.', label='coarse dt')
    axes[0].axvline(float(data['release_time']), color='gray', ls='--', label='load removed')
    axes[0].set(xlabel='time', ylabel='tip y displacement', title='Actual load and free rebound'); axes[0].legend()
    axes[1].plot(t, energy, label='kinetic + bond energy'); axes[1].plot(t, work, '--', label='external work')
    axes[1].set(xlabel='time', ylabel='energy / work', title='No damping, no broken bonds'); axes[1].legend()
    fig.savefig(out/'motion.png', dpi=160); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=Path('output/bonded_dem_powerfoam'))
    parser.add_argument('--render-only', action='store_true')
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    if args.render_only:
        with np.load(args.out/'trajectory.npz') as archive:
            data = dict(archive)
    else:
        s = setup()
        coarse, first = simulate(s, 1)
        data, fine = simulate(s, 2)
        displacement_scale = np.max(np.abs(data['samples'][:, 1]))
        difference = float(np.max(np.linalg.norm(data['positions']-coarse['positions'], axis=-1))/displacement_scale)
        assert difference < .005, difference
        np.savez_compressed(args.out/'trajectory.npz', **data)
        motion_plot(data, coarse, args.out)
        sources = [Path(__file__), Path(__file__).with_name('bonded_dem_core.py'),
                   Path(__file__).with_name('bonded_dem_dynamics.py'), Path(__file__).with_name('bonded_dem_material.py')]
        report = dict(coarse=first, fine=fine, max_position_dt_difference_over_tip_amplitude=difference,
                      graph='Initial Cech overlap pairs; radius=.9h. Permanent material bonds, no rebonding.',
                      load='sin(pi*t/T)^2 downward end force for 0<t<T; zero afterwards; no damping',
                      limits='Fixed radii; assigned box masses and isotropic inertia; no collision, damage, fitted material or material-region guarantee.',
                      source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
        (args.out/'mechanics.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print(json.dumps(report, indent=2), flush=True)
    render_trajectory(data, args.out)
    print(f'Complete: {args.out}', flush=True)


if __name__ == '__main__':
    main()
