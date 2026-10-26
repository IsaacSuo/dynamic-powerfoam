"""PowerSim's own pipeline (github.com/nttung1110/PowerSim, cloned at /home/fangsuo/py/PowerSim: PhysGaussian's Warp MPM
on the foam's primitives as particles; per frame positions from the particles, q0 (x) conj(R), r0 det(F)^(1/3), the
radiance axes by R; Power Foam's renderer) on PAC-NeRF elastic scene 0 with its truth (2026-10-25): E 23008, nu 0.3008
(gt/elastic.json), rho 1000 (configs/elastic/0.py), v0 from gt/elastic.json held for the first substep, gravity -9.8 y,
PAC-NeRF's sticky ground at world y = 0 (scene y = -0.05, conversion.json), frame_dt 1/24, 200 substeps per frame
(PAC-NeRF's dt). PowerSim's MPM frame puts the primitives' bbox centre at (1, 1, 1); its scale is set to the bbox's
longest side so one MPM unit is one metre. n_grid: --n-grid (64: about 8 primitives per grid cell, MPM's usual count).
The checkpoint is a copy of the round01_8k_pruned run's config without the thin-surface keys PowerSim's Power Foam lacks
(model.pt linked). Runs simulate.py per view (train split index; 0 = PAC-NeRF camera 5, 3 = camera 4) and writes a grid
against PAC-NeRF's frames (PowerSim's frame k is the state after k + 1 frames, PAC-NeRF's r_<cam>_<k+1>).
  python free_surface/powersim_pacnerf.py --out output/powersim_pacnerf"""
import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

POWERSIM = Path('/home/fangsuo/py/PowerSim')
RUN = Path('/home/fangsuo/py/dynamic-powerfoam/.claude/worktrees/thin-surface/output/pacnerf_elastic0_f0_round01_8k_pruned')
PAC = Path('/home/fangsuo/py/dynamic-powerfoam/data/pacnerf')
SCENE = Path('/home/fangsuo/py/dynamic-powerfoam/data/pacnerf_powerfoam/elastic_0_f0')
THIN_ONLY = ('ball_inside_surface', 'ball_inside_weight', 'ball_inside_weight_per_site', 'cvt_every', 'cvt_samples', 'cvt_weight',
             'hull_clip', 'hull_keep', 'hull_prune_only', 'hull_spacing', 'hull_transparent', 'init_from', 'spacing_k',
             'spacing_kappa', 'spacing_weight', 'thickness_from', 'thickness_weight')


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--out', type=Path, required=True)
    a.add_argument('--run', type=Path, default=RUN, help='the foam run (config.yaml + model.pt), e.g. a denser foam of the same scene')
    a.add_argument('--n-grid', type=int, default=64)
    a.add_argument('--views', type=int, nargs='+', default=[0, 3])
    a.add_argument('--frames', type=int, default=13)
    args = a.parse_args()
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    ck = out/'ckpt'; ck.mkdir(exist_ok=True)
    lines = [l for l in (args.run/'config.yaml').read_text().splitlines(True) if l.split(':')[0].strip() not in THIN_ONLY]
    (ck/'config.yaml').write_text(''.join(lines))
    if (ck/'model.pt').is_symlink():
        (ck/'model.pt').unlink()                                                    # a link left by another --run
    os.symlink(args.run.resolve()/'model.pt', ck/'model.pt')

    p = torch.load(args.run/'model.pt', map_location='cpu')['points'].double().numpy()
    lo, hi = p.min(0), p.max(0)
    centre, side = (lo+hi)/2, float((hi-lo).max())
    conv = json.loads((SCENE/'conversion.json').read_text())
    ground_scene = float(conv['ground_y_in_scene'])
    ground_sim = ground_scene-centre[1]+1.
    gt = json.loads((PAC/'gt'/'elastic.json').read_text())['0']
    dt = 1/24/200
    cfg = dict(material='jelly', n_grid=args.n_grid, grid_lim=2.0, E=gt['E'], nu=gt['nu'], density=1000.0, g=[0.0, -9.8, 0.0],
               substep_dt=dt, frame_dt=200*dt*(1+1e-9), frame_num=args.frames, scale=side, rotation_degree=[], rotation_axis=[],
               opacity_threshold=0.0, default_camera_index=0,
               boundary_conditions=[
                   dict(type='bounding_box'),
                   dict(type='surface_collider', point=[1.0, ground_sim, 1.0], normal=[0.0, 1.0, 0.0], surface='sticky', friction=0.0,
                        start_time=0.0, end_time=1e3),
                   dict(type='enforce_particle_translation', point=[1.0, 1.0, 1.0], size=[1.0, 1.0, 1.0], velocity=gt['v'],
                        start_time=0.0, end_time=.5*dt)])
    (out/'sim.json').write_text(json.dumps(cfg, indent=1))
    print('primitives %d; bbox side %.4f m (MPM scale 1: one unit = one metre); ground at MPM y %.4f (the lowest primitive %.4f above it); '
          'dx %.4f m, primitives per grid cell over the object about %.1f; v0 %s' % (
              len(p), side, ground_sim, lo[1]-ground_scene, 2./args.n_grid, len(p)*(2./args.n_grid)**3/0.035699, np.round(gt['v'], 3).tolist()), flush=True)

    env = dict(os.environ, PYTHONPATH=str(POWERSIM), PYTHONNOUSERSITE='1')
    py = '/home/fangsuo/anaconda3/envs/om/bin/python'
    for v in args.views:
        cmd = [py, '-m', 'powersim.foam_evolve.simulate', '--checkpoint-config', str(ck/'config.yaml'), '--sim-config', str(out/'sim.json'),
               '--split', 'train', '--camera-index', str(v), '--background', '1', '1', '1', '--output-dir', str(out/('view%d' % v)),
               '--save-positions']
        print(' '.join(cmd), flush=True)
        r = subprocess.run(cmd, cwd=POWERSIM, env=env)
        if r.returncode:
            raise SystemExit('simulate failed for view %d' % v)

    names = [f['file_path'].split('/')[-1] for f in json.loads((SCENE/'transforms_train.json').read_text())['frames']]
    cams = [int(names[v].split('_')[1]) for v in args.views]
    cols = list(range(1, args.frames+1))
    tiles = []
    for v, c in zip(args.views, cams):
        gt_row = []
        for k in cols:
            im = Image.open(PAC/'data'/'elastic'/'0'/'data'/('r_%d_%d.png' % (c, k)))
            if im.mode == 'RGBA':
                bg = Image.new('RGBA', im.size, (255, 255, 255, 255)); bg.alpha_composite(im); im = bg
            gt_row.append(im.convert('RGB'))
        ps_row = [Image.open(out/('view%d' % v)/('frame_%04d.png' % (k-1))).convert('RGB') for k in cols]
        tiles += [('PAC-NeRF r_%d' % c, gt_row), ('PowerSim', ps_row)]
    W, H = tiles[0][1][0].size
    grid = Image.new('RGB', (len(cols)*W, len(tiles)*H+20), 'white')
    d = ImageDraw.Draw(grid)
    for j, k in enumerate(cols):
        d.text((j*W+4, 4), 'frame %d (t %.3f s)' % (k, k/24), fill=(0, 0, 0))
    for i, (lab, row) in enumerate(tiles):
        for j, im in enumerate(row):
            grid.paste(im.resize((W, H)), (j*W, 20+i*H))
        d.text((4, 20+i*H+4), lab, fill=(0, 0, 0))
    grid.save(out/'compare_pacnerf.png')
    print('wrote %s' % (out/'compare_pacnerf.png'), flush=True)


if __name__ == '__main__':
    main()
