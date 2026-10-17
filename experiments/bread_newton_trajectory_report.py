"""Compare complete finite-model trajectories from two different solvers.

Their adaptive substeps may differ. Agreement is solver/trajectory evidence,
not an independent material validation or a time-refinement proof.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--new', type=Path, required=True)
    parser.add_argument('--previous', type=Path, default=Path('output/powersim_bread/finite_dem/adaptive_coarse'))
    args = parser.parse_args()
    old_report = json.loads((args.previous/'mechanics.json').read_text())
    new_report = json.loads((args.new/'mechanics.json').read_text())
    assert old_report['complete'] and new_report['complete'] and new_report['checks_pass']
    assert old_report['total_peak_force'] == new_report['total_peak_force']
    for key in ('bonded_dem_finite_gpu.py', 'bonded_dem_linear_gpu.py', 'bread_linear_sim.py'):
        if key in old_report['source_sha256']:
            assert old_report['source_sha256'][key] == new_report['source_sha256'][key]
    def read(path):
        with np.load(path/'trajectory.npz') as a: return a['q'], a['samples'], a['original_ids']
    qo, so, io = read(args.previous); qn, sn, ids = read(args.new)
    assert np.array_equal(io, ids) and np.allclose(so[:, 0], sn[:, 0], rtol=0, atol=1e-12)
    with np.load('output/powersim_bread/material_geometry.npz') as a: mass = a['volumes'][a['keep']]
    position_difference = np.linalg.norm(qn[:, :, :3]-qo[:, :, :3], axis=2)
    rms = np.sqrt((position_difference**2*mass).sum(axis=1)/mass.sum())
    angular_max, angular_rms = [], []
    for old, new in zip(qo, qn):
        angle = (Rotation.from_rotvec(old[:, 3:]).inv()*Rotation.from_rotvec(new[:, 3:])).magnitude()
        angular_max.append(float(angle.max()))
        angular_rms.append(float(np.sqrt(np.dot(mass, angle**2)/mass.sum())))
    result = dict(previous_seconds=old_report['elapsed_seconds'], new_seconds=new_report['elapsed_seconds'],
        previous_substeps=old_report['actual_substeps'], new_substeps=new_report['actual_substeps'],
        max_center_difference=float(position_difference.max()),
        mass_weighted_rms_center_difference_max=float(rms.max()),
        max_orientation_difference_rad=max(angular_max),
        mass_weighted_rms_orientation_difference_rad_max=max(angular_rms),
        loaded_region_mean_z_max_difference=float(abs(sn[:, 1]-so[:, 1]).max()),
        scope='Same finite law, different solvers/adaptive steps. Full-run timings from separate runs, not a controlled repeated performance benchmark.')
    (args.new/'previous_comparison.json').write_text(json.dumps(result, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.7), layout='constrained')
    labels = {'newton': 'Newton: original kernels', 'fast': 'Newton: fused kernels',
              'previous': 'Previous finite solver'}
    ax[0].plot(so[:, 0], so[:, 1], label=labels.get(old_report.get('solver'), 'Previous finite solver'))
    ax[0].plot(sn[:, 0], sn[:, 1], '--', label=labels.get(new_report.get('solver'), 'Newton-FGMRES'))
    ax[0].set(ylabel='Loaded region mean z displacement'); ax[0].legend()
    ax[1].plot(sn[:, 0], sn[:, 2], label='Total energy')
    ax[1].plot(sn[:, 0], sn[:, 3], '--', label='External work'); ax[1].legend()
    ax[2].plot(sn[:, 0], position_difference.max(axis=1), label='Max center difference')
    ax[2].plot(sn[:, 0], rms, label='Mass-weighted RMS difference'); ax[2].legend()
    for axis in ax: axis.set_xlabel('Time (model units)')
    fig.savefig(args.new/'previous_comparison.png', dpi=160); plt.close(fig)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
