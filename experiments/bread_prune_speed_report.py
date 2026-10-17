"""Compare matched physical intervals of original and rebuilt pruned bread."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    report = json.loads((path/'mechanics.json').read_text())
    rows = [json.loads(line) for line in (path/'progress.jsonl').read_text().splitlines()]
    rows = [row for row in rows if row['step'] <= report['completed_intervals']]
    return report, rows


def summary(rows, count):
    rows = rows[:count]
    assert len(rows) == count and rows[-1]['step'] == count
    return dict(intervals=count, seconds=rows[-1]['elapsed_seconds'],
                substeps=sum(r['substeps'] for r in rows),
                failed_attempts=sum(r['failed_attempts'] for r in rows),
                max_depth=max(r['max_subdivision_depth'] for r in rows),
                nonlinear_iterations=sum(r['nonlinear_iterations'] for r in rows),
                krylov_iterations=sum(r['krylov_iterations'] for r in rows),
                cg_iterations=sum(r['cg_iterations'] for r in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('output/powersim_bread/prune_speed_1e-4'))
    args = parser.parse_args()
    old_path = Path('output/powersim_bread/macro_amplitude/target_10pct/trial_01_full')
    old, previous = read(old_path)
    fresh, original = read(args.root/'baseline_16')
    new, pruned = read(args.root/'pruned_192')
    for report in (old, fresh):
        for key in ('dt', 'total_peak_force', 'load_multiplier'):
            assert np.isclose(report[key], new[key], rtol=1e-14, atol=0), key
        assert abs(report['setup']['period_estimate']-new['setup']['period_estimate']) < 1e-14
        for name, sha in report['source_sha256'].items():
            if name != 'bread_macro_run.py':
                assert new['source_sha256'][name] == sha, name
    groups = [('fresh_prefix', original, min(len(original), len(pruned))),
              ('historical_pilot', previous, min(96, len(pruned))),
              ('historical_common', previous, min(len(previous), len(pruned)))]
    comparisons = {}
    for label, reference, count in groups:
        a, b = summary(reference, count), summary(pruned, count)
        assert np.allclose([r['time'] for r in reference[:count]],
                           [r['time'] for r in pruned[:count]], rtol=0, atol=1e-11)
        comparisons[label] = dict(original=a, pruned=b, speedup=a['seconds']/b['seconds'],
            substep_reduction_factor=a['substeps']/b['substeps'],
            max_loaded_mean_difference_width_fraction=max(abs(x['downward_fraction_width']-y['downward_fraction_width'])
                for x, y in zip(reference[:count], pruned[:count])))
    comparisons['pruned_full'] = summary(pruned, len(pruned))
    comparisons['pruned_full'].update(complete=new['complete'], failure=new['failure'],
        checks_pass=new['checks_pass'], peak_downward_fraction=new['peak_downward_fraction'],
        work_energy_error_over_peak=new['work_energy_error_over_peak'])
    geometry = json.loads((args.root/'geometry/material_geometry.json').read_text())
    result = dict(geometry=geometry, comparisons=comparisons, historical_baseline=str(old_path),
        historical_failure=old['failure'],
        methodology='Serial GPU runs; fresh 16-interval baseline plus historical long trajectory. '
                    'Original coefficients, retained fixed/load IDs, total force, pulse duration, dt and '
                    'solver tolerances preserved. Force weights renormalized on surviving original load IDs. '
                    'New faces/volumes computed after deletion. Timing excludes setup, compilation and I/O. '
                    'Historical timings are not a contemporaneous repeated benchmark; no full baseline exists '
                    'past interval 146. Matching macro motion does not establish local or temporal accuracy.')
    (args.root/'comparison.json').write_text(json.dumps(result, indent=2)+'\n')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for rows, label in [(previous, 'original (earlier run)'), (original, 'original (fresh 16)'),
                        (pruned, 'delete 583 + rebuild')]:
        t = np.array([r['time'] for r in rows])/new['setup']['period_estimate']
        axes[0, 0].plot(t, [r['elapsed_seconds']/60 for r in rows], label=label)
        axes[0, 1].plot(t, [r['substeps'] for r in rows], label=label)
        axes[1, 0].plot(t, [100*r['downward_fraction_width'] for r in rows], label=label)
    count = min(len(previous), len(pruned))
    axes[1, 1].plot([r['time']/new['setup']['period_estimate'] for r in pruned[:count]],
                   [100*(p['downward_fraction_width']-o['downward_fraction_width'])
                    for p, o in zip(pruned[:count], previous[:count])])
    for ax, title, ylabel in zip(axes.flat,
        ['Cumulative solve time', 'Substeps per output interval', 'Loaded-region motion', 'Macro-motion difference'],
        ['minutes', 'substeps', 'downward / width (%)', 'pruned - original (percentage points)']):
        ax.set(title=title, xlabel='physical time / pulse duration', ylabel=ylabel)
        ax.grid(alpha=.3)
    axes[0, 1].set_yscale('log', base=2)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(args.root/'comparison.png', dpi=160); plt.close(fig)
    print(json.dumps(comparisons, indent=2))


if __name__ == '__main__':
    main()
