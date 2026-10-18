"""Vid2Sim GSO object -> the dynamic-data layout of springgaus_train.py (same trainer, same method).

Writes data/springgaus_powerfoam/vid2sim_<obj>/conversion.json and data/springgaus_powerfoam/vid2sim_<obj>_dynamic/:
- cameras.json: the 12 views (transforms_train.json, NeRF/OpenGL c2w, horizontal field of view) as OpenCV K, R, t in the
  Vid2Sim world (metres), principal point at the image centre ((W - 1) / 2, Blender pixel centres); fps 20 (one frame per
  0.05 s simulation step), 24 frames, gravity -z, floor z = -0.7 (normal +z), camera_names V00..V11, hit_frame (first
  frame at which the DEM's lowest contact sphere reaches the floor in free fall);
- masks/<cam>/<ff>.png: alpha of a_<view>_<frame>.png; images/<cam>/<ff>.png: its RGB over black;
- register_calib/initial_state_upright_nospin.json: the foam -> world map of the static scene (x_world = x_scene - offset,
  vid2sim_to_powerfoam) in the springgaus registration format (identity rotation, scale 1), initial velocity g dt / 2
  (their backward Euler with one step per frame falls g dt^2 n (n + 1) / 2 by frame n; this initial velocity makes an
  accurate free fall hit those positions, as in vid2sim_train.py), no spin, no camera time offsets.
The static foam (output/vid2sim_<obj>_spacing8k) and the DEM (output/vid2sim_dem/<obj>) are used as they are.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

SRC = Path('data/vid2sim/ext')
OUT = Path('data/springgaus_powerfoam')
DT, G, FLOOR = .05, 9.8, -.7


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--obj', default='sofa')
    p.add_argument('--dem', type=Path, default=None, help='default output/vid2sim_dem/<obj>')
    a = p.parse_args()
    src = SRC/a.obj
    scene = Path(f'data/vid2sim_powerfoam/{a.obj}_f0')
    dem = a.dem or Path(f'output/vid2sim_dem/{a.obj}')
    offset = np.array(json.loads((scene/'conversion.json').read_text())['offset'], float)
    tr = json.loads((src/'transforms_train.json').read_text())
    views = [int(Path(f['file_path']).name.split('_')[1]) for f in tr['frames']]
    W, H = Image.open(src/'data'/f'a_{views[0]}_0.png').size
    focal = .5*W/math.tan(.5*tr['camera_angle_x'])
    n_frames = 24
    names = [f'V{v:02d}' for v in views]
    cams = {}
    for f_, nm in zip(tr['frames'], names):
        c2w = np.array(f_['transform_matrix'], float)
        Rcv = c2w[:3, :3]@np.diag([1., -1., -1.])       # OpenGL (x right, y up, -z forward) -> OpenCV (y down, z forward)
        Rw2c = Rcv.T
        t = -Rw2c@c2w[:3, 3]
        K = [[focal, 0., (W-1)/2], [0., focal, (H-1)/2], [0., 0., 1.]]
        cams[nm] = dict(K=K, R=Rw2c.tolist(), t=t.tolist())
    # first contact of the DEM's lowest contact sphere in free fall (v0 = -g dt / 2 along z)
    geo = np.load(dem/'geometry.npz')
    P, m = geo['points'], geo['masses']
    R = (3*m/(4*np.pi))**(1/3)
    gap = float((P[:, 2]-R).min()-FLOOR)                 # height of the lowest sphere bottom above the floor
    v0 = G*DT/2
    t_hit = (-v0+math.sqrt(v0*v0+2*G*gap))/G
    hit = int(math.ceil(t_hit/DT))
    com = ((m[:, None]*P).sum(0)/m.sum()).tolist()
    dyn = OUT/f'vid2sim_{a.obj}_dynamic'
    for c in names:
        (dyn/'masks'/c).mkdir(parents=True, exist_ok=True); (dyn/'images'/c).mkdir(parents=True, exist_ok=True)
    for v, c in zip(views, names):
        for f in range(n_frames):
            rgba = np.asarray(Image.open(src/'data'/f'a_{v}_{f}.png').convert('RGBA'), np.float32)
            al = rgba[..., 3]
            Image.fromarray(al.astype(np.uint8)).save(dyn/'masks'/c/f'{f:02d}.png')
            Image.fromarray((rgba[..., :3]*al[..., None]/255).astype(np.uint8)).save(dyn/'images'/c/f'{f:02d}.png')
    meta = dict(obj=f'vid2sim_{a.obj}', source=str(src), fps=1/DT, hit_frame=hit, frames=n_frames, width=W, height=H,
                gravity=[0., 0., -G], ground_z=FLOOR, ground_normal=[0., 0., 1.], camera_names=names, cameras=cams,
                gt=yaml.safe_load(open(src/'gt_phys_params.yaml')))
    (dyn/'cameras.json').write_text(json.dumps(meta, indent=1))
    (dyn/'rest').mkdir(parents=True, exist_ok=True)          # object-free frames for the render stage: the images are on black
    for c in names:
        Image.new('RGB', (W, H), (0, 0, 0)).save(dyn/'rest'/f'background_{c}.png')
    (dyn/'register_calib').mkdir(parents=True, exist_ok=True)
    reg = dict(obj=f'vid2sim_{a.obj}', R=np.eye(3).tolist(), s=1., colmap_mean=[0., 0., 0.], t=(-offset).tolist(),
               v0=[0., 0., -v0], w0=[0., 0., 0.], rotation_centre_world=com,
               time_offsets_frames={c: 0. for c in names}, note='vid2sim_to_springgaus: x_world = x_scene - offset')
    (dyn/'register_calib'/'initial_state_upright_nospin.json').write_text(json.dumps(reg, indent=1))
    (OUT/f'vid2sim_{a.obj}').mkdir(parents=True, exist_ok=True)
    (OUT/f'vid2sim_{a.obj}'/'conversion.json').write_text(json.dumps(dict(scale=1., centre=[.5, .5, .5], offset=offset.tolist()), indent=1))
    print('vid2sim_%s: %d views %dx%d, focal %.1f px, lowest sphere %.3f m above the floor -> first contact at frame %d (t %.3f s)' % (
        a.obj, len(names), W, H, focal, gap, hit, t_hit), flush=True)


if __name__ == '__main__':
    main()
