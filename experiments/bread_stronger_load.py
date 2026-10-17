"""Re-solve bread under a larger FORCE, and render without display scaling.

This is a load stress test of the existing LINEAR mechanical system. Some
local rotations leave its small-angle scope; convergence is not a validation
of a finite-strain constitutive law. The original small-load run is preserved.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.bread_linear_sim import prepare, simulate, prune_check, render_motion
from experiments.powersim_bread_demo import load_scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--multiplier', type=float, default=10.)
    parser.add_argument('--baseline', type=Path, default=Path('output/powersim_bread/linear_dem'))
    parser.add_argument('--out', type=Path, default=Path('output/powersim_bread/linear_dem_10x'))
    args = parser.parse_args()
    if not np.isfinite(args.multiplier) or args.multiplier <= 0:
        parser.error('multiplier must be positive and finite')
    if args.out.resolve() == args.baseline.resolve():
        parser.error('Use a separate output directory to preserve the baseline')
    args.out.mkdir(parents=True, exist_ok=True)
    baseline = json.loads((args.baseline/'mechanics.json').read_text())
    # Reuse the reference geometry, loads and solver only if the baseline
    # implementation matches. Do not compare different physical systems.
    for name in ('bread_material_geometry.py', 'bonded_dem_linear_gpu.py', 'bread_linear_sim.py'):
        digest = hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
        if digest != baseline['source_sha256'][name]:
            raise ValueError(f'Baseline source mismatch: {name}')
    with np.load('output/powersim_bread/material_geometry.npz') as archive:
        geometry = dict(archive)
    with np.load(args.baseline/'trajectory.npz') as archive:
        baseline_q, baseline_samples = archive['q'], archive['samples']
    solver, unit_force, period, setup = prepare(geometry)
    force = baseline['total_peak_force']*args.multiplier
    print(f'Re-solving at {args.multiplier:g} times the applied force: {force}', flush=True)
    q, samples, convergence = simulate(solver, unit_force*force, period, refinement=2)
    predicted = args.multiplier*baseline_q
    amplitude = np.linalg.norm(q[:, :, :3], axis=2).max()
    discrepancy = float(np.linalg.norm(q[:, :, :3]-predicted[:, :, :3], axis=2).max()/amplitude)
    if discrepancy > 1e-4:
        raise RuntimeError(f'Linear force-scaling check failed: {discrepancy}')
    fixed = solver.fixed.cpu().numpy()
    assert np.isfinite(q).all() and np.max(abs(q[:, fixed])) == 0
    mass = geometry['volumes'][geometry['keep']]
    rotation = np.linalg.norm(q[:, :, 3:], axis=2).max(0)
    ordered = np.argsort(rotation)
    cumulative = np.cumsum(mass[ordered])/mass.sum()
    levels = [.5, .9, .99, .999, 1.]
    weighted_rot = rotation[ordered[np.minimum(np.searchsorted(cumulative, levels), len(mass)-1)]]
    report = dict(setup=setup, force_multiplier=args.multiplier, total_peak_force=force,
        convergence=convergence, force_scaling_relative_position_error=discrepancy,
        max_translation=float(amplitude),
        max_translation_over_width=float(amplitude/np.ptp(geometry['points'][geometry['keep'], 0])),
        loaded_z_range=[float(samples[:, 1].min()), float(samples[:, 1].max())],
        max_rotation=float(rotation.max()), rotation_quantile_levels=levels,
        rotation_quantiles=np.quantile(rotation, levels).tolist(),
        mass_weighted_rotation_quantiles=weighted_rot.tolist(),
        mass_fraction_peak_rotation_above_0p2_rad=float(mass[rotation > .2].sum()/mass.sum()),
        cells_peak_rotation_above_0p2_rad=int((rotation > .2).sum()),
        fixed_max_abs=float(abs(q[:, fixed]).max()),
        scope='Higher applied load, actual new linear solve; no display exaggeration. Local large rotations are outside small-strain validity. No damage or collisions.',
        baseline=str(args.baseline),
        source_sha256={name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                       for name in ('bread_stronger_load.py', 'bread_linear_sim.py', 'bonded_dem_linear_gpu.py', 'bread_material_geometry.py')})
    (args.out/'mechanics.json').write_text(json.dumps(report, indent=2)+'\n')
    np.savez_compressed(args.out/'trajectory.npz', q=q, samples=samples, period=period,
                        original_ids=np.flatnonzero(geometry['keep']))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    ax[0].plot(samples[:, 0], samples[:, 1], label=f'{args.multiplier:g}x force: new solve')
    ax[0].plot(baseline_samples[:, 0], baseline_samples[:, 1], label='previous load')
    ax[0].axvline(period, color='gray', ls='--')
    ax[0].set(xlabel='time (uncalibrated)', ylabel='loaded region z displacement')
    ax[0].legend()
    ax[1].plot(samples[:, 0], samples[:, 2], label='energy')
    ax[1].plot(samples[:, 0], samples[:, 3], '--', label='external work')
    ax[1].legend(); fig.savefig(args.out/'motion.png', dpi=160); plt.close(fig)
    print(json.dumps(report, indent=2), flush=True)
    del solver, baseline_q, predicted
    torch.cuda.empty_cache()
    model, cameras, _, state = load_scene()
    prune_check(model, cameras, state, geometry['keep'], args.out)
    render_motion(model, cameras, state, geometry['keep'], geometry, q, samples, period, args.out)


if __name__ == '__main__':
    main()
