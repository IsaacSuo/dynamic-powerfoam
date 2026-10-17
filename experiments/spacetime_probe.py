"""Native spacetime competition, original rasterizer slices and handoff probe.

Run: conda activate om; python -m experiments.spacetime_probe
No optimization, video reconstruction, physical simulation or time blending.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from experiments.affine_probe import GPURenderer, membership, save_image, texture
from experiments.shared_affine_render import camera_rays, oracle
from powerfoam.spacetime import SpacetimeFoam
import matplotlib.pyplot as plt


def appearance(foam, sliced, colors, offsets=None):
    # Fixed local texture offsets and normals; no claim of material deformation.
    if offsets is None:
        tx, ty = np.meshgrid(np.linspace(-.55, .55, 4), [-.35, .35])
        offsets = foam.radii[:, None, None]*np.column_stack((tx.ravel(), ty.ravel(), np.zeros(8)))
    sites = sliced.centers[:, None] + offsets[sliced.ids]
    normals = np.tile([0., 0., 1.], (len(sliced.ids), 1))
    return sites, colors[sliced.ids], normals


def render_slice(gpu, foam, colors, time, resolution):
    sliced = foam.slice(time)
    if not len(sliced.ids):
        return np.zeros((resolution, resolution, 3)), np.zeros((resolution, resolution))
    sites, color, normals = appearance(foam, sliced, colors)
    return gpu.render(sliced.centers, sliced.radii, sites, color, np.eye(3), explicit_normals=normals)


def make_scene():
    rng = np.random.default_rng(2026)
    xx, yy = np.meshgrid(np.linspace(-.85, .85, 7), np.linspace(-.85, .85, 7))
    p = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size)))
    p[:, :2] += rng.uniform(-.012, .012, (len(p), 2))
    # Spatially varying straight trajectories, not rigid rotation over time.
    v = np.column_stack((.45 + .35*p[:, 1], -.35*p[:, 0], np.zeros(len(p))))
    tau = .35*p[:, 0]
    radius = rng.uniform(.30, .34, len(p))
    lifetime = rng.uniform(1.15, 1.65, len(p))
    # p is the common t=0 configuration; store each moving center at its own tau.
    foam = SpacetimeFoam(p + v*tau[:, None], v, tau, radius, (radius/lifetime)**2)
    static = SpacetimeFoam(p, np.zeros_like(v), tau, radius, (radius/lifetime)**2)
    palette = np.array([[.96, .72, .16], [.08, .28, .78]])
    parity = (np.indices(xx.shape).sum(axis=0).ravel() % 2)
    colors = np.repeat(palette[parity, None], 8, axis=1)
    return foam, static, colors


def sliced_ownership(foam, points, time):
    s = foam.slice(time)
    if not len(s.ids):
        return np.full(len(points), -1, dtype=int)
    local = membership(points, s.centers, s.radii)
    return np.where(local >= 0, s.ids[np.maximum(local, 0)], -1)


def validate_geometry(foam):
    rng = np.random.default_rng(82)
    points = rng.uniform([-2., -2., -.5], [2., 2., .5], (4096, 3))
    times = np.linspace(-2., 2., 17)
    max_power_error = 0.
    for time in times:
        events = np.column_stack((points, np.full(len(points), time)))
        direct = foam.ownership(events)
        sliced = sliced_ownership(foam, points, time)
        np.testing.assert_array_equal(direct, sliced)
        s = foam.slice(time)
        if len(s.ids):
            q3 = ((points[:, None]-s.centers)**2).sum(axis=2)-s.radii**2
            max_power_error = max(max_power_error, float(np.max(np.abs(q3-foam.power4d(events)[:, s.ids]))))
    assert np.linalg.eigvalsh(foam.metrics4d()).min() > 0
    assert not len(foam.slice(10.).ids)
    # Exact support birth/death and just-inside/outside times for one primitive.
    single = SpacetimeFoam([[0., 0., 0.]], [[1., 0., 0.]], 0., 1., 1.)
    for time in [-1., 1., -1.00001, 1.00001]:
        assert not len(single.slice(time).ids)
    assert len(single.slice(.99999).ids) == 1
    # A pair with distinct velocities rotates its generated spatial bisector.
    p = np.array([[-.5, 0., 0.], [.5, 0., 0.]])
    v = np.array([[0., -.4, 0.], [0., .4, 0.]])
    pair = SpacetimeFoam(p, v, 0., 2., .1)
    angles = []
    for time in [-.5, .5]:
        s = pair.slice(time)
        delta = s.centers[1]-s.centers[0]
        normal = delta/np.linalg.norm(delta)
        angles.append(float(np.rad2deg(np.arctan2(normal[1], normal[0]))))
        midpoint = s.centers.mean(axis=0)
        tangent = np.array([-normal[1], normal[0], 0.])
        queries = midpoint + np.linspace(-.5, .5, 31)[:, None]*tangent
        values = pair.power4d(np.column_stack((queries, np.full(len(queries), time))))
        np.testing.assert_allclose(values[:, 0], values[:, 1], atol=1e-12)
    return dict(event_checks=len(points)*len(times), ownership_mismatches=0,
                max_power_error=max_power_error, bisector_angles_degrees=angles,
                bisector_rotation_degrees=angles[1]-angles[0], support_endpoints_passed=True)


def handoff_probe(out):
    # Overlapping lifetimes, continuous support, but unrelated cell appearances.
    # Small center offsets avoid coincident sites / whole-volume power ties.
    tau = np.array([-.6, .6])
    v = np.tile([.35, 0., 0.], (2, 1))
    p = v*tau[:, None] + np.array([[-.02, 0., 0.], [.02, 0., 0.]])
    foam = SpacetimeFoam(p, v, tau, .8, .4)
    colors = np.repeat(np.array([[.95, .12, .08], [.08, .30, .95]])[:, None], 8, axis=1)
    tied = np.repeat(np.array([[.25, .70, .35], [.25, .70, .35]])[:, None], 8, axis=1)
    times = np.linspace(-.25, .25, 501)
    sample_rgb, tied_rgb, owners, alpha = [], [], [], []
    for time in times:
        s = foam.slice(time)
        o = np.array([[.35*time, 0., 8.]])
        d = np.array([[0., 0., -1.]])
        sites, c, normals = appearance(foam, s, colors)
        # Exact float64 volume ray, as opposed to just reading cell color.
        expected = oracle(o, d, np.eye(3), np.zeros(3), s.centers, s.radii, normals, sites, c, 1000.)
        sample_rgb.append(expected['rgb'][0]); alpha.append(float(expected['alpha'][0]))
        owners.append(int(s.ids[expected['owner'][0]]))
        q = np.array([[.35*time, 0., -.001]])
        owner = membership(q, s.centers, s.radii)
        tied_rgb.append(texture(q, owner, sites, tied[s.ids], s.radii)[0])
    sample_rgb, tied_rgb = np.array(sample_rgb), np.array(tied_rgb)
    limits = []
    for eps in [1e-2, 1e-3, 1e-4, 1e-5]:
        pair_rgb = []
        for time in [-eps, eps]:
            s = foam.slice(time); sites, c, normals = appearance(foam, s, colors)
            ray = oracle(np.array([[.35*time, 0., 8.]]), np.array([[0., 0., -1.]]),
                         np.eye(3), np.zeros(3), s.centers, s.radii, normals, sites, c, 1000.)
            pair_rgb.append(ray['rgb'][0])
        limits.append(dict(epsilon=eps, rgb_jump_mae=float(np.abs(pair_rgb[1]-pair_rgb[0]).mean())))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for c, name in enumerate(['red', 'green', 'blue']):
        axes[0].plot(times, sample_rgb[:, c], color=name, label=name)
        axes[1].plot(times, tied_rgb[:, c], color=name, label=name)
    for ax, title in zip(axes, ['Independent cell colors: ray RGB jumps', 'Identical colors: appearance sanity control']):
        ax.set(xlabel='Time along moving probe', ylabel='RGB', title=title, ylim=(-.03, 1.03))
        ax.axvline(0., color='gray', linestyle=':'); ax.legend()
    fig.tight_layout(); fig.savefig(out/'handoff.png', dpi=150); plt.close(fig)
    assert min(alpha) > .999
    assert limits[-1]['rgb_jump_mae'] > .5
    return dict(min_ray_alpha=min(alpha), one_sided_limits=limits,
                owner_switches=int(np.count_nonzero(np.diff(owners))),
                tied_color_max_step=float(np.abs(np.diff(tied_rgb, axis=0)).max()),
                interpretation='A moving ray has a finite appearance jump despite continuous support; identical-color control is not a learned solution')


def spacetime_plot(out):
    p = np.column_stack(([-1., -.35, .4, 1.], np.zeros((4, 2))))
    v = np.column_stack(([.9, .4, -.3, -.65], np.zeros((4, 2))))
    tau = np.array([-.5, .1, -.1, .4])
    r = np.array([.65, .55, .6, .6])
    life = np.array([1.4, 1.1, 1.35, 1.2])
    foam = SpacetimeFoam(p, v, tau, r, (r/life)**2)
    x, t = np.meshgrid(np.linspace(-2., 2., 600), np.linspace(-1.8, 1.8, 500))
    events = np.column_stack((x.ravel(), np.zeros(x.size), np.zeros(x.size), t.ravel()))
    palette = np.array([[.20, .55, .9], [.95, .60, .15], [.25, .75, .45], [.85, .30, .55]])
    owner = foam.ownership(events); rgb = np.full((len(owner), 3), .97)
    inside = owner >= 0; rgb[inside] = palette[owner[inside]]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.imshow(rgb.reshape(*x.shape, 3), origin='lower', extent=(-2., 2., -1.8, 1.8), aspect='auto')
    for i in range(len(p)):
        times = np.linspace(tau[i]-life[i], tau[i]+life[i], 100)
        ax.plot(p[i, 0]+v[i, 0]*(times-tau[i]), times, '--', color='black', alpha=.6, linewidth=1)
    ax.axhline(0., color='black', linewidth=1)
    ax.set(xlabel='Space x (y=z=0)', ylabel='Time', title='4D Foam restricted to an x-time plane\nColors = winning cell; dashed = support center paths')
    fig.tight_layout(); fig.savefig(out/'spacetime_partition.png', dpi=150); plt.close(fig)


def precision_diagnostic(foam, colors, resolution):
    """Compare the earlier float32 clipper to the same float64 slice oracle.

    The original rasterizer quantizes neighbor offsets/radii to float16. This
    separate implementation helps distinguish slice algebra from rendering
    precision; it is not a single-variable ablation of the original renderer.
    """
    from powerfoam.affine_raytrace import SharedAffineRayTracer
    s = foam.slice(-.7)
    sites, c, n = appearance(foam, s, colors)
    o, d = camera_rays(resolution)
    ref = oracle(o, d, np.eye(3), np.zeros(3), s.centers, s.radii, n, sites, c, 1000.)
    tracer = SharedAffineRayTracer(s.centers, s.radii, n, sites, c)
    actual = tracer.render(o, d, np.eye(3))
    error = np.abs(actual['rgb']-ref['rgb'])
    assert error.mean() < 1e-5
    return dict(time=-.7, rgb_mae=float(error.mean()), rgb_max_error=float(error.max()),
                pixels_error_above_01=int((error.max(axis=1) > .1).sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('output/spacetime'))
    parser.add_argument('--resolution', type=int, default=192)
    parser.add_argument('--frames', type=int, default=25)
    args = parser.parse_args()
    if args.resolution < 32 or args.frames < 3:
        parser.error('Need resolution >=32 and frames >=3')
    args.out.mkdir(parents=True, exist_ok=True)
    foam, static, colors = make_scene()
    checks = validate_geometry(foam); print('Geometry:', checks, flush=True)
    gpu = GPURenderer(args.resolution)
    o, d = camera_rays(args.resolution)
    records = []
    times = [-1.4, -.7, 0., .7, 1.4]
    fig, axes = plt.subplots(2, len(times), figsize=(13, 5.5))
    for column, time in enumerate(times):
        sliced = foam.slice(time)
        sites, c, normals = appearance(foam, sliced, colors)
        expected = oracle(o, d, np.eye(3), np.zeros(3), sliced.centers, sliced.radii, normals, sites, c, 1000.)
        actual, alpha = render_slice(gpu, foam, colors, time, args.resolution)
        error = np.abs(actual.reshape(-1, 3)-expected['rgb'])
        alpha_error = np.abs(alpha.ravel()-expected['alpha'])
        foreground = (alpha.ravel() > .001) | (expected['alpha'] > .001)
        stats = dict(time=time, active_supports=len(sliced.ids), rgb_mae=float(error.mean()),
                     rgb_max_error=float(error.max()), alpha_mae=float(alpha_error.mean()),
                     foreground_rgb_mae=float(error[foreground].mean()),
                     pixels_error_above_001=int((error.max(axis=1) > .01).sum()),
                     pixels_error_above_01=int((error.max(axis=1) > .1).sum()))
        assert stats['rgb_mae'] < 1e-4, stats
        assert stats['alpha_mae'] < 1e-4, stats
        records.append(stats); print('Rasterizer:', stats, flush=True)
        axes[0, column].imshow(expected['rgb'].reshape(args.resolution, args.resolution, 3))
        axes[1, column].imshow(actual)
        axes[0, column].set_title(f't={time:+.1f}\n{len(sliced.ids)} active supports')
        save_image(args.out/f'slice_{column}.png', actual)
    for ax in axes.ravel(): ax.set_xticks([]); ax.set_yticks([])
    axes[0, 0].set_ylabel('Float64 all-face oracle'); axes[1, 0].set_ylabel('Original GPU rasterizer')
    fig.tight_layout(); fig.savefig(args.out/'slices.png', dpi=150); plt.close(fig)
    frames, animation = [], []
    for index, time in enumerate(np.linspace(-1.6, 1.6, args.frames)):
        old, _ = render_slice(gpu, static, colors, time, args.resolution)
        current, alpha = render_slice(gpu, foam, colors, time, args.resolution)
        canvas = Image.new('RGB', (2*args.resolution, args.resolution+48), 'white')
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 5), 'Time gate only: velocity = 0', fill='black')
        draw.text((args.resolution+8, 5), 'Tilted spacetime Foam', fill='black')
        draw.text((8, 23), f't={time:+.2f}   active supports={len(foam.slice(time).ids)}/49', fill='black')
        for column, im in enumerate((old, current)):
            canvas.paste(Image.fromarray(np.uint8(np.clip(im, 0, 1)*255+.5)), (column*args.resolution, 48))
        frames.append(canvas)
        animation.append(dict(time=float(time), active_supports=len(foam.slice(time).ids), opaque_pixels=int((alpha>.99).sum())))
        if index % 5 == 0: print(f'Animation {index+1}/{args.frames}', flush=True)
    # Ping-pong playback, explicitly for display; the model's trajectories are linear.
    playback = frames + frames[-2:0:-1]
    playback[0].save(args.out/'motion.gif', save_all=True, append_images=playback[1:], duration=110, loop=0)
    empty_rgb, empty_alpha = render_slice(gpu, foam, colors, 10., args.resolution)
    assert not empty_rgb.any() and not empty_alpha.any()
    handoff = handoff_probe(args.out); print('Handoff:', handoff, flush=True)
    spacetime_plot(args.out)
    precision = precision_diagnostic(foam, colors, args.resolution)
    print('Float32 clipper diagnostic:', precision, flush=True)
    result = dict(geometry=checks, rasterizer=records, handoff=handoff, animation=animation,
                  float32_clip_diagnostic=precision,
                  settings=dict(cells=len(foam.centers), resolution=args.resolution, frames=args.frames,
                                device='original Warp GPU rasterizer, om environment',
                                appearance='fixed +z dipoles; fixed local texel offsets; constant color per cell; slice radius sets texture scale',
                                limitations='hand-authored trajectories, no video training, no MPM, no speed claim; playback reverses time; support continuity does not imply appearance continuity'))
    (args.out/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Saved', args.out.resolve(), flush=True)


if __name__ == '__main__':
    main()
