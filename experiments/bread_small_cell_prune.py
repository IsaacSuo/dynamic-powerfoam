"""Render-only ablation: delete tiny positive-volume cells and rebuild adjacency.

Surviving parameters and poses are identical. This measures immediate rendering
effects, not the motion of a newly simulated pruned mechanical network.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

from experiments.powersim_bread_demo import ATTRIBUTES, load_scene, camera_from_bundle
from experiments.full_model_rigid_probe import render, compare
from experiments.bonded_dem_powerfoam import state_from_bodies


def rgb_image(value):
    return Image.fromarray(np.uint8(np.clip(value, 0, 1)*255+.5))


def install(model, state, selection):
    for name in ATTRIBUTES:
        value = torch.as_tensor(state[name][selection], dtype=torch.float32, device='cuda')
        setattr(model, name, torch.nn.Parameter(value.contiguous(), requires_grad=False))
    model.rebuild_adjacency()


def panel(images, labels, width):
    height = images[0].height
    sheet = Image.new('RGB', (width*len(images), height+30), '#202020')
    draw = ImageDraw.Draw(sheet)
    for col, (img, label) in enumerate(zip(images, labels)):
        sheet.paste(img, (col*width, 30))
        draw.text((col*width+8, 8), label, fill='white')
    return sheet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--ratio', type=float, default=1e-4)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--trajectory', type=Path, default=Path(
        'output/powersim_bread/macro_amplitude/target_10pct/trial_01_pilot/trajectory.npz'))
    args = parser.parse_args()
    if not np.isfinite(args.ratio) or args.ratio <= 0:
        raise ValueError('ratio must be positive and finite')
    args.out.mkdir(parents=True, exist_ok=False)
    geometry_path = Path('output/powersim_bread/material_geometry.npz')
    geometry = np.load(geometry_path)
    original_ids = np.flatnonzero(geometry['keep'])
    volumes = geometry['volumes'][original_ids]
    threshold = float(np.median(volumes)*args.ratio)
    retained = volumes >= threshold
    assert retained.any() and not retained.all()
    model, cameras, _, checkpoint = load_scene()
    base = {name: checkpoint[name][original_ids].numpy() for name in ATTRIBUTES}
    trajectory = np.load(args.trajectory)
    assert np.array_equal(trajectory['original_ids'], original_ids)
    frame = int(np.argmin(trajectory['samples'][:, 1]))
    q = trajectory['q'][frame]
    peak = state_from_bodies(base, geometry['points'][original_ids]+q[:, :3],
                            Rotation.from_rotvec(q[:, 3:]).as_matrix())
    views = sorted(set(np.linspace(0, len(cameras['eyes'])-1, 8).astype(int).tolist()+[56]))
    mask = np.zeros(len(geometry['keep']), dtype=bool)
    mask[original_ids[retained]] = True
    np.savez_compressed(args.out/'selection.npz', keep=mask,
                        removed_original_ids=original_ids[~retained], threshold=threshold)
    report = dict(ratio_to_median=args.ratio, volume_threshold=threshold,
                  before_cells=len(volumes), after_cells=int(retained.sum()),
                  removed_cells=int((~retained).sum()),
                  removed_mass_fraction=float(volumes[~retained].sum()/volumes.sum()),
                  width=args.width, views=views, difference_display_gain=20,
                  geometry_sha256=hashlib.sha256(geometry_path.read_bytes()).hexdigest(),
                  trajectory=str(args.trajectory),
                  trajectory_sha256=hashlib.sha256(args.trajectory.read_bytes()).hexdigest(),
                  scope='Delete tiny cells from the existing 91978-cell baseline; original renderer, '
                        'rebuilt adjacency, unchanged surviving appearance and poses. '
                        'Deformed state is a saved pre-failure pilot frame, not a pruned simulation.',
                  states={})
    summary_panels = []
    for name, state in [('rest', base), ('near_peak', peak)]:
        folder = args.out/name
        folder.mkdir()
        install(model, state, np.ones(len(volumes), dtype=bool))
        reference = {}
        for view in views:
            reference[view] = render(model, camera_from_bundle(cameras, view, args.width))
        # A repeat control detects nondeterministic render noise in tiny differences.
        repeat = render(model, camera_from_bundle(cameras, views[0], args.width))
        control = compare(reference[views[0]], repeat)
        install(model, state, retained)
        results, rows, crops = [], [], {}
        for view in views:
            before = reference[view]
            after = render(model, camera_from_bundle(cameras, view, args.width))
            assert all(np.isfinite(v).all() for v in after.values())
            metrics = compare(before, after)
            difference = np.abs(before['rgb']-after['rgb'])
            foreground = ((before['alpha'] > .05) | (after['alpha'] > .05)).squeeze()
            changed = difference.max(axis=-1) > .01
            metrics['foreground_changed_fraction_above_0_01'] = float(changed[foreground].mean())
            metrics['rgb_mae_foreground_255'] = metrics['rgb_mae_foreground']*255
            metrics['view'] = view
            results.append(metrics)
            images = [rgb_image(before['rgb']), rgb_image(after['rgb']), rgb_image(20*difference)]
            for image, suffix in zip(images, ['before', 'after', 'difference_x20']):
                image.save(folder/f'view_{view:03d}_{suffix}.png')
            row = panel(images, [f'{name} / view {view} / before: {len(volumes)}',
                                f'after deleting {(~retained).sum()} cells',
                                'absolute RGB difference x20'], args.width)
            row.save(folder/f'view_{view:03d}_comparison.png')
            rows.append(row)
            yy, xx = np.unravel_index(difference.max(axis=-1).argmax(), difference.shape[:2])
            side = min(128, args.width, images[0].height)
            x0 = max(0, min(int(xx)-side//2, args.width-side))
            y0 = max(0, min(int(yy)-side//2, images[0].height-side))
            box = (x0, y0, x0+side, y0+side)
            enlarged = [img.crop(box).resize((args.width, args.width), Image.Resampling.NEAREST)
                        for img in images]
            crops[view] = panel(enlarged, ['before / crop enlarged 4x', 'after / same crop',
                                          'difference x20 / same crop'], args.width)
            print(json.dumps(dict(state=name, **metrics)), flush=True)
        sheet = Image.new('RGB', (rows[0].width, sum(row.height for row in rows)), '#202020')
        y = 0
        for row in rows:
            sheet.paste(row, (0, y)); y += row.height
        sheet.save(folder/'all_views.png')
        worst = max(results, key=lambda d: d['rgb_mae_foreground'])
        crops[worst['view']].save(folder/'worst_view_crop.png')
        summary_panels.extend([rows[views.index(worst['view'])], crops[worst['view']]])
        report['states'][name] = dict(metrics=results, repeat_control=control,
            worst_view_by_foreground_mae=worst['view'],
            worst_rgb_mae_foreground_255=max(d['rgb_mae_foreground_255'] for d in results),
            worst_foreground_changed_fraction=max(d['foreground_changed_fraction_above_0_01'] for d in results))
        if name == 'near_peak':
            report['states'][name].update(frame=frame, time=float(trajectory['samples'][frame, 0]),
                downward_fraction_width=float(-trajectory['samples'][frame, 1]/np.ptp(geometry['points'][original_ids, 0])))
    overview = Image.new('RGB', (summary_panels[0].width, sum(p.height for p in summary_panels)), '#202020')
    y = 0
    for row in summary_panels:
        overview.paste(row, (0, y)); y += row.height
    overview.save(args.out/'comparison.png')
    (args.out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(f'Wrote comparison and report to {args.out}', flush=True)


if __name__ == '__main__':
    main()
