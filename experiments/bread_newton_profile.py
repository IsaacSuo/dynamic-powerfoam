"""Measure one matched full bread Newton step, optionally with fused JVP.

Stage timers synchronize CUDA; timings include instrumentation and may be
affected by the independent background run. They are not production timings.
"""
import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.bonded_dem_newton import NewtonBonds
from experiments.bonded_dem_tangent_warp import WarpTangent
from experiments.bread_linear_sim import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fused', action='store_true')
    parser.add_argument('--fused-blocks', action='store_true')
    parser.add_argument('--fused-cg', action='store_true')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    with np.load('output/powersim_bread/material_geometry.npz') as a: geo = dict(a)
    root = Path('output/powersim_bread/finite_dem/diagnostic')
    prior = json.loads((root/'mechanics.json').read_text())
    linear, force, period, _ = prepare(geo)
    edges, fixed = linear.edges.cpu().numpy(), linear.fixed.cpu().numpy()
    del linear; gc.collect(); torch.cuda.empty_cache()
    keep = geo['keep']
    s = NewtonBonds(geo['points'][keep], geo['radii'][keep], edges, geo['areas'],
                    geo['volumes'][keep], fixed, contacts=geo['contacts'])
    with np.load(root/'final_state.npz') as a:
        for key in ('x', 'rotation', 'fn', 'fs', 'mt', 'mb', 'normal', 'offset'):
            setattr(s, key, s.tensor(a[key]))
        s.v = s.tensor(a['velocity'])
    s.stiffness = WarpTangent(s)
    dt = prior['dt']*.5
    tm = prior['steps']*prior['dt']+.5*dt
    external = s.tensor(force*prior['total_peak_force']*np.sin(np.pi*tm/period)**2)
    zero = torch.zeros_like(s.v)
    if args.fused:
        from experiments.bonded_dem_jvp_warp import WarpJacobian
        s.jacobian_vector = WarpJacobian(s)
    if args.fused_blocks or args.fused_cg:
        from experiments.bonded_dem_precondition_warp import WarpDiagonal, WarpElastic
        if args.fused_blocks: s._diagonal_blocks = WarpDiagonal(s)
        if args.fused_cg: s.elastic_precondition = WarpElastic(s)
        s.prepare_scaling(s.trial(zero), dt)
        s.inner_iterations = 0
        s.elastic_precondition(torch.ones_like(zero), dt)
    # Exclude initial JIT / forward-AD initialization from the measured solve.
    s.stiffness(zero); s.jacobian_vector(zero, zero, dt, external)
    timers = {}
    def instrument(name):
        method = getattr(s, name)
        timers[name] = dict(calls=0, seconds=0.)
        def wrapped(*a, **kw):
            torch.cuda.synchronize(); start = time.perf_counter()
            try: return method(*a, **kw)
            finally:
                torch.cuda.synchronize()
                timers[name]['calls'] += 1
                timers[name]['seconds'] += time.perf_counter()-start
        setattr(s, name, wrapped)
    for name in ('jacobian_vector', 'elastic_precondition', 'trial', 'prepare_scaling'):
        instrument(name)
    energy = s.energy()
    torch.cuda.synchronize(); start = time.perf_counter()
    work, info = s.step(dt, external)
    torch.cuda.synchronize(); elapsed = time.perf_counter()-start
    report = dict(fused=args.fused, fused_blocks=args.fused_blocks, fused_cg=args.fused_cg,
                  elapsed_seconds=elapsed, timers=timers, **info,
                  energy_error=abs(s.energy()-energy-work)/energy,
                  timing_scope='Instrumented shared GPU; stage fractions and paired runs are indicative')
    assert report['energy_error'] < .002
    names = ['bonded_dem_newton.py', 'bread_newton_profile.py']
    if args.fused: names.append('bonded_dem_jvp_warp.py')
    if args.fused_blocks or args.fused_cg: names.append('bonded_dem_precondition_warp.py')
    report['source_sha256'] = {n: hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in names}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'profile.json').write_text(json.dumps(report, indent=2)+'\n')
    np.savez_compressed(args.out/'state.npz', x=s.x.cpu().numpy(), rotation=s.rotation.cpu().numpy())
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
