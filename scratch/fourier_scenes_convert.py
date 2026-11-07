"""the 12 elastic Fourier Power Foam scenes (11-07, user: thin-surface/fourier3d/output/fourier_pacnerf_<s>_8k_split_keep_clean)
-> inverse_data/fourier_<s>/ for lion_inverse.py's forward: conversion.json, transforms_all.json, masks (frame 0) from the
scene's training data, and foam/dem_exact_single from plasticine-k1's convert_foam.py (10-28, unchanged: visual hull ->
material-only cut, largest component), then the fields the DEM's build() reads that convert_foam does not write:
geometry radii = (3 V / 4 pi)^(1/3) (EnergyFoam's body radius, as the old scene 0's geometry), fixed = False,
body_of_cell = arange (one body per cell); cells density = 1.
  python scratch/fourier_scenes_convert.py <scene> [<scene> ...]       scene: elastic0..elastic9, torus, bird"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path('/home/fangsuo/py/dynamic-powerfoam')
WT = REPO/'.claude'/'worktrees'
ROOT = Path(__file__).resolve().parents[1]
PY = '/home/fangsuo/anaconda3/envs/om/bin/python'
CONVERT = WT/'plasticine-k1'/'free_surface'/'convert_foam.py'


def scene_dir(s):
    if s.startswith('elastic'):
        return REPO/'data'/'pacnerf_powerfoam'/('elastic_%s_f0' % s[len('elastic'):])
    return WT/'thin-surface'/'data'/'pacnerf_powerfoam'/('%s_f0' % s)


def run_dir(name):
    """<scene>[_<res>] -> the fourier3d run fourier_pacnerf_<scene>_<res>_split_keep_clean (res default 8k), the scene"""
    s, _, res = name.partition('_')
    return WT/'thin-surface'/'fourier3d'/'output'/('fourier_pacnerf_%s_%s_split_keep_clean' % (s, res or '8k')), s


def main():
    for name in sys.argv[1:]:
        t0 = time.time()
        run, s = run_dir(name); model = run/'model.pt'
        src = scene_dir(s); out = ROOT/'inverse_data'/('fourier_%s' % name)
        assert model.exists() and src.exists(), (model, src)
        (out/'masks').mkdir(parents=True, exist_ok=True)
        for f in ('conversion.json', 'transforms_all.json'):
            shutil.copy(src/f, out/f)
        for f in sorted((src/'masks').iterdir()):
            shutil.copy(f, out/'masks'/f.name)
        env = dict(os.environ, PYTHONPATH='%s:%s' % (REPO, WT/'updated-lagrange'))
        r = subprocess.run([PY, '-u', str(CONVERT), str(model), str(src), str(out/'foam')], cwd=str(WT/'updated-lagrange'), env=env,
                           capture_output=True, text=True)
        print(r.stdout.strip()); print(r.stderr.strip()[-2000:]) if r.returncode else None
        if r.returncode:
            raise SystemExit('%s: convert_foam failed (exit %d)' % (s, r.returncode))
        d = out/'foam'/'dem_exact_single'
        g = dict(np.load(d/'geometry.npz')); c = dict(np.load(d/'cells.npz'))
        n = len(g['masses'])
        g['radii'] = (3*g['masses']/(4*np.pi))**(1/3); g['fixed'] = np.zeros(n, bool); g['body_of_cell'] = np.arange(n)
        c['density'] = np.ones(n)
        np.savez(d/'geometry.npz', **g); np.savez(d/'cells.npz', **c)
        V = g['masses']
        print('%s: %d cells, volume %.4e m^3, h %.2f mm, cells with V < 1e-3 median %d; %.0f s -> %s' % (
            s, n, V.sum(), 1e3*(V.sum()/n)**(1/3), int((V < 1e-3*np.median(V)).sum()), time.time()-t0, out), flush=True)


if __name__ == '__main__':
    main()
