"""Matched one-step comparisons on saved bread neighborhoods, CPU only.

The outer neighbor ring is clamped in this TEST FIXTURE. It is not a bread
simulation and does not alter the ongoing full model's boundary conditions.
All solvers receive identical saved histories, masses, inertias and stiffness.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy import sparse

from experiments.bonded_dem_finite_gpu import FiniteBondsGPU
from experiments.bonded_dem_newton import NewtonBonds


def extract(seed):
    with np.load('output/powersim_bread/material_geometry.npz') as a: geo = dict(a)
    with np.load('output/powersim_bread/finite_dem/diagnostic/final_state.npz') as a: saved = dict(a)
    keep = geo['keep']; mapping = np.full(len(keep), -1, int); mapping[keep] = np.arange(keep.sum())
    pairs = mapping[geo['pairs']]
    graph = sparse.coo_matrix((np.ones(len(pairs)*2), (np.r_[pairs[:, 0], pairs[:, 1]],
                            np.r_[pairs[:, 1], pairs[:, 0]])), shape=(keep.sum(), keep.sum())).tocsr()
    inner = np.unique(np.r_[seed, graph[seed].indices])
    nodes = np.unique(np.r_[inner, graph[inner].indices])
    selected = np.zeros(keep.sum(), bool); selected[nodes] = True
    edges = np.flatnonzero(selected[pairs].all(1))
    mapping = np.full(len(keep), -1, int); mapping[nodes] = np.arange(len(nodes))
    kwargs = dict(points=geo['points'][keep][nodes], radii=geo['radii'][keep][nodes],
        pairs=mapping[pairs[edges]], areas=geo['areas'][edges], masses=geo['volumes'][keep][nodes],
        fixed=~np.isin(nodes, inner), contacts=geo['contacts'][edges], device='cpu')
    def make(cls):
        s = cls(**kwargs)
        for name in ('x', 'rotation'):
            setattr(s, name, s.tensor(saved[name][nodes]))
        for name in ('fn', 'fs', 'mt', 'mb', 'normal', 'offset'):
            setattr(s, name, s.tensor(saved[name][edges]))
        s.v = s.tensor(saved['velocity'][nodes]); s.v[s.fixed] = 0
        return s
    info = dict(seed_compact_id=seed, nodes=len(nodes), free_cells=len(inner), bonds=len(edges),
                mass_min=float(kwargs['masses'].min()), mass_max=float(kwargs['masses'].max()),
                source='finite_dem/diagnostic/final_state.npz',
                boundary='outer neighbor ring clamped, no external load, identical one-step problem')
    return make, info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=15427)
    parser.add_argument('--dt', type=float, default=.003361534273718165)
    args = parser.parse_args()
    torch.set_num_threads(1)
    make, info = extract(args.seed)
    print(info, flush=True)
    report = dict(fixture=info, dt=args.dt, runs=[])
    states = {}
    for name in ('previous', 'newton_elastic', 'newton_dense'):
        solver = make(FiniteBondsGPU if name == 'previous' else NewtonBonds)
        before = {k: getattr(solver, k).clone() for k in ('x', 'v', 'rotation', 'fn', 'fs', 'mt', 'mb', 'offset', 'normal')}
        energy = solver.energy(); start = time.perf_counter()
        try:
            external = torch.zeros_like(solver.mass)
            if name == 'previous':
                work, stats = solver.step(args.dt, external)
            else:
                work, stats = solver.step(args.dt, external, backend='dense' if name.endswith('dense') else 'fgmres')
            row = dict(name=name, success=True, elapsed_seconds=time.perf_counter()-start,
                       energy_error=abs(solver.energy()-energy-work)/max(energy, 1e-30), **stats)
            assert row['energy_error'] < .002
            states[name] = {k: getattr(solver, k).clone() for k in ('x', 'rotation')}
        except (RuntimeError, ValueError) as error:
            assert all(torch.equal(getattr(solver, k), v) for k, v in before.items())
            row = dict(name=name, success=False, elapsed_seconds=time.perf_counter()-start, reason=str(error))
        report['runs'].append(row); print(json.dumps(row), flush=True)
    assert 'newton_dense' in states and 'newton_elastic' in states, report['runs']
    if 'newton_dense' in states:
        for row in report['runs']:
            if not row['success']: continue
            state = states[row['name']]
            row['vs_dense_position_error'] = float((state['x']-states['newton_dense']['x']).abs().max())
            row['vs_dense_rotation_matrix_error'] = float((state['rotation']-states['newton_dense']['rotation']).abs().max())
            assert row['vs_dense_position_error'] < 1e-7, row
            assert row['vs_dense_rotation_matrix_error'] < 2e-6, row
    out = Path('output/bonded_dem_newton'); out.mkdir(parents=True, exist_ok=True)
    report['source_sha256'] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                             for name in ('bread_newton_patch.py', 'bonded_dem_newton.py', 'bonded_dem_finite_gpu.py')}
    (out/f'patch_{args.seed}.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
