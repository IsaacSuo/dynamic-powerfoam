"""One full-size bread step from a saved state; no change to the running job.

Compare the same equations and tolerances, not complete trajectories. GPU
timings are indicative only when other processes share the device.
"""
import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.bonded_dem_finite_gpu import FiniteBondsGPU
from experiments.bonded_dem_newton import NewtonBonds
from experiments.bonded_dem_tangent_warp import WarpTangent
from experiments.bread_linear_sim import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dt-factor', type=float, default=1.)
    args = parser.parse_args()
    if args.dt_factor <= 0: parser.error('dt-factor must be positive')
    torch.set_num_threads(1)
    root = Path('output/powersim_bread/finite_dem/diagnostic')
    prior = json.loads((root/'mechanics.json').read_text())
    with np.load('output/powersim_bread/material_geometry.npz') as a: geo = dict(a)
    with np.load(root/'final_state.npz') as a: state = dict(a)
    linear, force, period, setup = prepare(geo)
    pairs, fixed = linear.edges.cpu().numpy(), linear.fixed.cpu().numpy()
    del linear; gc.collect(); torch.cuda.empty_cache()
    dt = prior['dt']*args.dt_factor
    start_time = prior['steps']*prior['dt']
    external = force*prior['total_peak_force']*np.sin(np.pi*(start_time+.5*dt)/period)**2
    keep = geo['keep']
    report = dict(setup=setup, dt=dt, start_time=start_time,
                  source_state=str(root/'final_state.npz'), runs=[],
                  scope='Identical full-size single step, not trajectory validation; shared-GPU timings')
    print(json.dumps({k: v for k, v in report.items() if k != 'runs'}), flush=True)
    states = {}
    for name, cls in (('previous', FiniteBondsGPU), ('newton_elastic', NewtonBonds)):
        s = cls(geo['points'][keep], geo['radii'][keep], pairs, geo['areas'],
                geo['volumes'][keep], fixed, contacts=geo['contacts'])
        for key in ('x', 'rotation', 'fn', 'fs', 'mt', 'mb', 'normal', 'offset'):
            setattr(s, key, s.tensor(state[key]))
        s.v = s.tensor(state['velocity']); s.time = start_time
        assert not bool(s.v[s.fixed].any())
        s.stiffness = WarpTangent(s)
        s.stiffness(torch.zeros_like(s.mass))  # Exclude one-time kernel compilation.
        applied = s.tensor(external)
        energy = s.energy()
        torch.cuda.synchronize(); start = time.perf_counter()
        try:
            work, stats = s.step(dt, applied)
            torch.cuda.synchronize()
            row = dict(name=name, success=True, elapsed_seconds=time.perf_counter()-start,
                       energy_error=abs(s.energy()-energy-work)/max(energy, abs(work), 1e-30), **stats)
            assert row['energy_error'] < .002
            assert torch.equal(s.x[s.fixed], s.p[s.fixed])
            states[name] = (s.x.cpu().numpy(), s.rotation.cpu().numpy())
        except RuntimeError as error:
            if 'Subdivide:' not in str(error): raise
            torch.cuda.synchronize()
            row = dict(name=name, success=False, elapsed_seconds=time.perf_counter()-start,
                       reason=str(error))
            for key in ('x', 'rotation', 'fn', 'fs', 'mt', 'mb', 'normal', 'offset'):
                assert torch.equal(getattr(s, key), s.tensor(state[key]))
            assert torch.equal(s.v, s.tensor(state['velocity'])) and s.time == start_time
        report['runs'].append(row)
        print(json.dumps(row), flush=True)
        # WarpTangent holds a reference back to its solver.
        del s.stiffness; del s; gc.collect(); torch.cuda.empty_cache()
    if len(states) == 2:
        report['solver_difference'] = {
            key: float(np.max(np.abs(states['previous'][i]-states['newton_elastic'][i])))
            for i, key in enumerate(('max_position', 'max_rotation_matrix'))}
    report['source_sha256'] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ('bread_newton_step.py', 'bonded_dem_newton.py', 'bonded_dem_finite_gpu.py',
                     'bonded_dem_tangent_warp.py', 'bread_linear_sim.py')}
    report['state_sha256'] = hashlib.sha256((root/'final_state.npz').read_bytes()).hexdigest()
    out = Path('output/bonded_dem_newton'); out.mkdir(parents=True, exist_ok=True)
    target = out/f'full_step_{args.dt_factor:g}.json'
    target.write_text(json.dumps(report, indent=2)+'\n')
    print(target, flush=True)


if __name__ == '__main__':
    main()
