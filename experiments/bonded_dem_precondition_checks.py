"""CPU/CUDA equivalence checks for fused blocks and inner elastic CG."""
import hashlib
import json
from pathlib import Path

import torch

from experiments.bonded_dem_jvp_warp import WarpJacobian
from experiments.bonded_dem_newton import NewtonBonds
from experiments.bonded_dem_newton_checks import fixture
from experiments.bonded_dem_precondition_warp import WarpDiagonal, WarpElastic
from experiments.bonded_dem_tangent_warp import WarpTangent
from experiments.bread_newton_patch import extract


def main():
    torch.set_num_threads(1); torch.manual_seed(894)
    cases = [('unstressed', lambda: fixture(prestress=False)), ('prestressed', fixture)]
    for seed in (15427, 25204, 85720):
        make, _ = extract(seed)
        cases.append((f'patch_{seed}', lambda make=make: make(NewtonBonds)))
    rows = []
    for name, make in cases:
        for device in ('cpu', 'cuda'):
            s = make()
            if device == 'cuda':
                for key, value in list(vars(s).items()):
                    if isinstance(value, torch.Tensor): setattr(s, key, value.to(device))
                s.device = device
                s.stiffness = WarpTangent(s)
            dt = .02 if not name.startswith('patch') else .003361534273718165
            y = torch.randn_like(s.v)*.001
            y[:, :3] *= s.radii[:, None]; y[s.fixed] = 0
            s.prepare_scaling(s.trial(y), dt)
            fast = WarpDiagonal(s)()
            block_error = float(((fast-s.diagonal).flatten(1).norm(dim=1)/s.diagonal.flatten(1).norm(dim=1)).max())
            assert block_error < 1e-12, block_error
            errors = []
            inner_counts = []
            fused = WarpElastic(s)
            for _ in range(3):
                vector = torch.randn_like(s.v); vector[s.fixed] = 0
                s.inner_iterations = 0
                original = s.elastic_precondition(vector, dt)
                old_count = s.inner_iterations; s.inner_iterations = 0
                value = fused(vector, dt)
                error = float((value-original).norm()/original.norm())
                assert error < 1e-8, (name, device, error)
                assert torch.equal(value[s.fixed], torch.zeros_like(value[s.fixed]))
                errors.append(error); inner_counts.append([old_count, s.inner_iterations])
            assert torch.equal(fused(torch.zeros_like(s.v), dt), torch.zeros_like(s.v))
            # Compare a complete step, including the unchanged convergence test.
            reference = make(); external = torch.zeros_like(reference.v)
            if not name.startswith('patch'):
                external[:, 2] = -.03*reference.mass[:, 0]; external[reference.fixed] = 0
            _, old = reference.step(dt, external)
            s._diagonal_blocks = WarpDiagonal(s)
            s.elastic_precondition = fused
            s.jacobian_vector = WarpJacobian(s)
            energy = s.energy()
            work, new = s.step(dt, external.to(device))
            dx = float((s.x.cpu()-reference.x).abs().max())
            dr = float((s.rotation.cpu()-reference.rotation).abs().max())
            assert dx < 1e-8 and dr < 1e-7, (name, device, dx, dr)
            energy_error = abs(s.energy()-energy-work)/max(energy, abs(work), 1e-30)
            assert energy_error < 1e-5
            row = dict(case=name, device=device, block_relative_error_max=block_error,
                       inner_cg_relative_error_max=max(errors), inner_counts=inner_counts,
                       position_difference=dx, rotation_matrix_difference=dr,
                       energy_error=energy_error, old_iterations=old['nonlinear_iterations'],
                       new_iterations=new['nonlinear_iterations'])
            rows.append(row); print(row, flush=True)
    report = dict(checks=rows, source_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ('bonded_dem_precondition_warp.py', 'bonded_dem_precondition_checks.py', 'bonded_dem_jvp_warp.py', 'bonded_dem_newton.py')})
    out = Path('output/bonded_dem_newton'); out.mkdir(parents=True, exist_ok=True)
    (out/'precondition_checks.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
