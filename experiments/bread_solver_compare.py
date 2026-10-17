"""Compare completed matched intervals; differences are not an exact error bound."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, action='append', required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    def read(path):
        report = json.loads((path/'mechanics.json').read_text())
        assert report['complete'] and report['checks_pass'], path
        with np.load(path/'final_state.npz') as a: state = dict(a)
        return report, state
    rr, ref = read(args.reference)
    with np.load('output/powersim_bread/material_geometry.npz') as a:
        mass = a['volumes'][a['keep']]
        radii = a['radii'][a['keep']]
    rows = []
    for path in args.run:
        report, state = read(path)
        assert report['source_state_sha256'] == rr['source_state_sha256']
        assert abs(report['actual_time']-rr['actual_time']) < 1e-12
        assert report['total_peak_force'] == rr['total_peak_force']
        dx = np.linalg.norm(state['x']-ref['x'], axis=1)
        angle = Rotation.from_matrix(state['rotation']@ref['rotation'].transpose(0, 2, 1)).magnitude()
        dv = np.linalg.norm(state['velocity'][:, :3]-ref['velocity'][:, :3], axis=1)
        rows.append(dict(path=str(path), seconds=report['elapsed_seconds'],
            accepted_steps=report['actual_substeps'], failed_attempts=report['failed_attempts'],
            max_center_difference=float(dx.max()),
            mass_weighted_rms_center_difference=float(np.sqrt(np.dot(mass, dx*dx)/mass.sum())),
            max_orientation_difference_rad=float(angle.max()),
            mass_weighted_rms_orientation_difference_rad=float(np.sqrt(np.dot(mass, angle*angle)/mass.sum())),
            mass_weighted_rms_velocity_difference=float(np.sqrt(np.dot(mass, dv*dv)/mass.sum())),
            mass_fraction_orientation_difference_above_0p01_rad=float(mass[angle > .01].sum()/mass.sum()),
            max_center_difference_over_radius=float((dx/radii).max())))
    result = dict(reference=str(args.reference), comparisons=rows,
                  scope='Matched physical interval, identical initial state/load; adaptive step sizes may differ. Reference is numerical, not ground truth.')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
