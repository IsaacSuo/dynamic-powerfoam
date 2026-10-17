"""Validate fused full derivatives against forward AD and original residuals."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.bonded_dem_jvp_warp import WarpJacobian
from experiments.bonded_dem_newton import NewtonBonds
from experiments.bonded_dem_newton_checks import fixture, physical_state
from experiments.bread_newton_patch import extract


def main():
    torch.set_num_threads(1); torch.manual_seed(971)
    factories = [('unstressed', lambda: fixture(prestress=False)), ('prestressed', fixture)]
    for seed in (15427, 25204, 85720):
        make, _ = extract(seed)
        factories.append((f'patch_{seed}', lambda make=make: make(NewtonBonds)))
    rows = []
    for name, make in factories:
        for device in ('cpu', 'cuda'):
            s = make()
            if device == 'cuda':
                for key, value in list(vars(s).items()):
                    if isinstance(value, torch.Tensor): setattr(s, key, value.to(device))
                s.device = device
            fused = WarpJacobian(s)
            dt = .02 if name in ('unstressed', 'prestressed') else .003361534273718165
            force = torch.zeros_like(s.v)
            before = physical_state(s)
            errors, differences = [], []
            for trial in range(4):
                y = torch.randn_like(s.v)*(.003 if trial else 0.)
                y[:, :3] *= s.radii[:, None]; y[s.fixed] = 0
                direction = torch.randn_like(y)
                direction[:, :3] *= s.radii[:, None]; direction[s.fixed] = 0
                s.prepare_scaling(s.trial(y), dt)
                reference = s.jacobian_vector(y, direction, dt, force)
                value = fused(y, direction, dt, force)
                denominator = float(s.scale_residual(reference).norm())
                error = float(s.scale_residual(value-reference).norm())/denominator
                assert error < 2e-11, (name, device, error)
                errors.append(error)
                # Tiny-cell residual scaling amplifies subtraction roundoff;
                # scan perturbation sizes instead of relying on one FD step.
                fd_errors = []
                for h in (1e-4, 1e-5, 1e-6, 1e-7):
                    fd = (s.residual(y+h*direction, dt, force)-s.residual(y-h*direction, dt, force))/(2*h)
                    fd_errors.append(float(s.scale_residual(value-fd).norm())/denominator)
                difference = min(fd_errors)
                assert difference < 2e-6, (name, device, difference)
                differences.append(difference)
            assert all(torch.equal(getattr(s, key), value) for key, value in before.items())
            # Free directions on fixed DOFs must still produce identity rows.
            q = torch.randn_like(s.v)
            assert torch.equal(fused(torch.zeros_like(q), q, dt, force)[s.fixed], q[s.fixed])
            reference = make()
            external = torch.zeros_like(reference.v)
            if name in ('unstressed', 'prestressed'):
                external[:, 2] = -.03*reference.mass[:, 0]; external[reference.fixed] = 0
            _, original_info = reference.step(dt, external)
            s.jacobian_vector = fused
            _, fast_info = s.step(dt, external.to(device))
            dx = float((s.x.cpu()-reference.x).abs().max())
            dr = float((s.rotation.cpu()-reference.rotation).abs().max())
            assert dx < 1e-8 and dr < 1e-7, (name, device, dx, dr)
            row = dict(case=name, device=device, max_scaled_ad_relative_error=max(errors),
                       max_scaled_fd_relative_error=max(differences), position_difference=dx,
                       rotation_matrix_difference=dr, original_iterations=original_info['nonlinear_iterations'],
                       fused_iterations=fast_info['nonlinear_iterations'])
            rows.append(row); print(row, flush=True)
    out = Path('output/bonded_dem_newton'); out.mkdir(parents=True, exist_ok=True)
    report = dict(checks=rows, source_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                    for name in ('bonded_dem_jvp_warp.py', 'bonded_dem_jvp_checks.py', 'bonded_dem_newton.py')})
    (out/'fused_checks.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
