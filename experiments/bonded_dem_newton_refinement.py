"""Time accuracy of the unchanged rate law with a consistent Newton solver.

Use a separate moderate-mass fixture to resolve its fastest modes. This does
not alter bread masses or claim that a stable bread step resolves all modes.
"""
import hashlib
import json
from pathlib import Path

import torch

from experiments.bonded_dem_newton_checks import fixture


def main():
    torch.set_num_threads(1)
    results, states = [], []
    for steps in (8, 16, 32):
        s = fixture(minimum_mass=.01)
        initial = s.energy()
        dt = .04/steps
        iterations = 0
        for _ in range(steps):
            _, row = s.step(dt, torch.zeros_like(s.mass), tolerance=1e-11,
                            pose_tolerance=1e-11, linear_tolerance=1e-9)
            iterations += row['nonlinear_iterations']
        states.append((s.x.clone(), s.rotation.clone(), s.v.clone()))
        row = dict(steps=steps, dt=dt, nonlinear_iterations=iterations,
                   energy_error=abs(s.energy()-initial)/initial)
        assert row['energy_error'] < 1e-6
        results.append(row)
        print(row, flush=True)
    differences = {}
    for i, name in enumerate(('position', 'rotation_matrix', 'velocity')):
        coarse = float((states[0][i]-states[1][i]).norm())
        fine = float((states[1][i]-states[2][i]).norm())
        differences[name] = dict(coarse_to_medium=coarse, medium_to_fine=fine,
                                 reduction=coarse/fine)
        assert 3.5 < coarse/fine < 4.5, differences[name]
    report = dict(scope='Moderate-mass synthetic time-accuracy fixture; bread masses unchanged',
                  runs=results, differences=differences,
                  source_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                    for name in ('bonded_dem_newton.py', 'bonded_dem_newton_checks.py',
                                 'bonded_dem_newton_refinement.py', 'bonded_dem_finite_gpu.py')})
    out = Path('output/bonded_dem_newton'); out.mkdir(parents=True, exist_ok=True)
    (out/'refinement.json').write_text(json.dumps(report, indent=2)+'\n')
    print(differences, flush=True)


if __name__ == '__main__':
    main()
