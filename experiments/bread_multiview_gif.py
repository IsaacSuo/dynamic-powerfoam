"""Render a completed bread trajectory from six fixed, distinct official cameras."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

from experiments.powersim_bread_demo import ATTRIBUTES, load_scene, camera_from_bundle, render_image
from experiments.bonded_dem_powerfoam import state_from_bodies


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('output/powersim_bread/prune_speed_1e-4/pruned_192'))
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    out = args.out or args.run/'multiview'
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    mechanics = json.loads((args.run/'mechanics.json').read_text())
    assert mechanics['complete'] and mechanics['checks_pass'] and mechanics['failure'] is None
    geometry_path = Path(mechanics['geometry'])
    assert hashlib.sha256(geometry_path.read_bytes()).hexdigest() == mechanics['geometry_sha256']
    with np.load(geometry_path) as a: geo = dict(a)
    with np.load(args.run/'trajectory.npz') as a: trajectory = dict(a)
    keep = geo['keep']
    assert np.array_equal(trajectory['original_ids'], np.flatnonzero(keep))
    model, cameras, _, checkpoint = load_scene()
    base = {name: checkpoint[name][keep].numpy() for name in ATTRIBUTES}
    for name in ATTRIBUTES:
        value = torch.as_tensor(base[name], dtype=torch.float32, device='cuda').contiguous()
        setattr(model, name, torch.nn.Parameter(value, requires_grad=False))
    p0 = geo['points'][keep]
    center = np.average(p0, axis=0, weights=geo['volumes'][keep])
    directions = cameras['eyes'].numpy()-center
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    targets = [('side_a', 'Side A', [0, 1, .25]),
               ('side_b', 'Opposite side', [0, -1, .25]),
               ('top', 'Top', [0, 0, 1]),
               ('end_a', 'End A', [-1, 0, .25]),
               ('end_b', 'Opposite end', [1, 0, .25]),
               ('oblique', 'Oblique', [1, 1, .8])]
    selected, used = [], {0, 99}
    for key, label, target in targets:
        target = np.asarray(target, dtype=float); target /= np.linalg.norm(target)
        candidates = np.argsort(-(directions@target))
        index = next(int(i) for i in candidates if int(i) not in used)
        used.add(index)
        selected.append(dict(key=key, label=label, index=index,
                             eye=cameras['eyes'][index].tolist()))
    print('Selected cameras:', json.dumps(selected), flush=True)
    views = [camera_from_bundle(cameras, item['index'], 512) for item in selected]
    individual = [[] for _ in views]
    combined = []
    samples = trajectory['samples']
    period = float(trajectory['period'])
    phase_indices = [0, int(np.argmin(samples[:, 1])), int(np.argmax(samples[:, 1]))]
    for frame, (q, row) in enumerate(zip(trajectory['q'], samples)):
        state = state_from_bodies(base, p0+q[:, :3], Rotation.from_rotvec(q[:, 3:]).as_matrix())
        with torch.no_grad():
            for name in ('points', 'quaternions', 'texel_sv_axis'):
                getattr(model, name).copy_(torch.as_tensor(state[name], dtype=torch.float32, device='cuda'))
            model.rebuild_adjacency()
        canvas = Image.new('RGB', (1536, 1024), 'black')
        for column, (item, camera) in enumerate(zip(selected, views)):
            label = f"{item['label']} | t={row[0]:.3f}"
            image, _ = render_image(model, camera, label)
            ImageDraw.Draw(image).text((10, 28), 'loading' if row[0] < period-1e-10 else 'free release', fill='white')
            individual[column].append(image)
            canvas.paste(image, ((column%3)*512, (column//3)*512))
        combined.append(canvas)
        if frame in phase_indices:
            canvas.save(out/f'frame_{frame:03d}.png')
        if frame%8 == 0:
            print(f'Multiview frames {frame+1}/{len(samples)}', flush=True)
    def save_gif(frames, path):
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=80, loop=0)
        with Image.open(path) as im:
            assert im.n_frames == len(frames)
            for index in range(im.n_frames):
                im.seek(index); im.load()
    save_gif(combined, out/'six_views.gif')
    for item, frames in zip(selected, individual):
        save_gif(frames, out/f"{item['key']}.gif")
    sheet = Image.new('RGB', (1536, 3*1024), 'black')
    for i, frame in enumerate(phase_indices):
        sheet.paste(combined[frame], (0, i*1024))
    sheet.save(out/'keyframes.png')
    report = dict(trajectory=str(args.run/'trajectory.npz'),
                  trajectory_sha256=hashlib.sha256((args.run/'trajectory.npz').read_bytes()).hexdigest(),
                  geometry_sha256=mechanics['geometry_sha256'], cameras=selected,
                  grid_order='row 1: side_a, side_b, top; row 2: end_a, end_b, oblique',
                  frames=len(samples), frame_duration_ms=80, display_displacement_scale=1,
                  keyframe_indices=phase_indices, same_trajectory=True,
                  renderer='Original PowerfoamScene.forward; rebuilt adjacency every frame',
                  playback='Slowed playback, same as previous two-view GIF',
                  files={p.name:p.stat().st_size for p in out.glob('*.gif')})
    (out/'render.json').write_text(json.dumps(report, indent=2)+'\n')
    print('Done:', out/'six_views.gif', flush=True)


if __name__ == '__main__':
    main()
