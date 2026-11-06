"""one forward of a Fourier PAC-NeRF scene (lion_inverse.SCENES fourier_*) at its truth E, nu and beta = the scene's
beta_min, frames 0..K (default all), saving per frame the sites x, the radii r0 det(F_B)^(1/3) and F_B (for the render)
to out_inverse/fourier_forward/<scene>/sim.npz. No timeout.
  python -u scratch/fourier_forward.py <scene> [K]"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import free_surface.lion_inverse as li

scene = sys.argv[1]
li.set_scene(scene)
K = int(sys.argv[2]) if len(sys.argv) > 2 else li.NFRAMES-1
th = np.array([math.log(li.GT['E']), li.GT['nu'], math.log(li.BETA_MIN)])
out = ROOT/'out_inverse'/'fourier_forward'/scene; out.mkdir(parents=True, exist_ok=True)
print('%s: E %.4g, nu %.4f, beta %.3g s (damping ratio as pacnerf_e0 at 0.003), v0 %s; frames 0..%d' % (
    scene, li.GT['E'], li.GT['nu'], li.BETA_MIN, li.V0, K), flush=True)
t0 = time.time()
try:
    o = li.forward(th, K, 1e-3, save_frames=True, verbose=True)
except RuntimeError as e:
    print('FAILED: %s' % e, flush=True)
    np.savez(out/('sim_K%d.npz' % K), failed=str(e), theta=th)
    sys.exit(1)
np.savez(out/('sim_K%d.npz' % K), theta=th, x=o['x'], r=o['r'], F=o['F'], f=o['f'], stats=json.dumps(o['stats']))
print('done: build %.0f s, run %.0f s, products %d -> %s' % (o['stats']['build'], o['stats']['seconds'], o['stats']['products'], out/('sim_K%d.npz' % K)), flush=True)
