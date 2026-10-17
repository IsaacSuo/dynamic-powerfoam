"""Visualize the small inverse-dynamics pilot without implying real-data validation."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.run/'report.json').read_text())
    history = [json.loads(row) for row in (args.run/'training.jsonl').read_text().splitlines()]
    patch = np.load(args.run/'patch.npz')
    fig = plt.figure(figsize=(15, 9))
    ax = fig.add_subplot(2, 3, 1, projection='3d')
    p = patch['points']; p = (p-p.mean(0))*1000
    for i, j in patch['pairs']:
        ax.plot(*p[[i, j]].T, color='gray', alpha=.4, linewidth=.6)
    fixed = patch['fixed']
    ax.scatter(*p[~fixed].T, color='tab:blue', s=35, label='free')
    ax.scatter(*p[fixed].T, color='tab:red', s=45, label='fixed')
    ax.set(title='16 actual bread cells / 49 bonds', xlabel='x (scaled units)', ylabel='y', zlabel='z')
    ax.set_box_aspect(np.ptp(p, axis=0)); ax.legend(fontsize=8)
    ax = fig.add_subplot(2, 3, 2)
    ax.semilogy([r['evaluation'] for r in history], [100*r['normalized_rmse'] for r in history])
    ax.set(title='Fit two loading clips', xlabel='Objective evaluations (LBFGS)', ylabel='Normalized motion RMSE (%)')
    ax.grid(alpha=.3)
    cases = ['train_down', 'train_pull', 'test_side']
    names = ['Downward (train)', 'Pull (train)', 'Sideways (held out)']
    ax = fig.add_subplot(2, 3, 3)
    x = np.arange(3)
    before = [100*report['initial_normalized_rmse'][name] for name in cases]
    after = [100*report['final_normalized_rmse'][name] for name in cases]
    a = ax.bar(x-.18, before, .36, label='before fitting', color='tab:orange')
    b = ax.bar(x+.18, after, .36, label='after fitting', color='tab:green')
    ax.bar_label(a, fmt='%.2f', fontsize=8); ax.bar_label(b, fmt='%.2f', fontsize=8)
    ax.set_xticks(x, ['Down', 'Pull', 'Held-out side'])
    ax.set(title='All free cell positions', ylabel='Normalized motion RMSE (%)')
    ax.legend(fontsize=8); ax.set_ylim(0, max(before+after)*1.25)
    motion = {}
    width = report['config']['timing']['reference_width']
    for index, (name, title) in enumerate(zip(cases, names)):
        target = np.load(args.run/f'{name}_target.npz')
        data = np.load(args.run/f'{name}_comparison.npz')
        axis = report['targets'][name]['axis']
        weights = np.abs(target['loads'][:, :, axis]).sum(axis=0); weights /= weights.sum()
        times = np.arange(len(data['target']))*float(data['dt'])*1000
        ax = fig.add_subplot(2, 3, index+4)
        for key, label, color, style in [('target','target (synthetic)','black','-'),
                                        ('before','before fitting','tab:orange','--'),
                                        ('after','after fitting','tab:green',':')]:
            displacement = data[key]-data[key][0]
            curve = 100*(displacement[:, :, axis]@weights)/width
            ax.plot(times, curve, label=label, color=color, linestyle=style, linewidth=2)
        ax.set(title=title, xlabel='time (ms)', ylabel='Mean loaded displacement / width (%)')
        ax.grid(alpha=.3)
        if index == 0:
            ax.legend(fontsize=8)
        mask = ~patch['fixed']
        displacement = data['target'][1:, mask]-data['target'][0, mask]
        error = data['after'][1:, mask]-data['target'][1:, mask]
        motion[name] = dict(target_rms_component_over_width=float(np.sqrt(np.mean(displacement**2))/width),
                           learned_rms_component_error_over_width=float(np.sqrt(np.mean(error**2))/width),
                           learned_max_center_error_over_width=float(np.linalg.norm(error, axis=-1).max()/width))
    fig.suptitle('Learn cell stiffness from motion | synthetic within-model test, no real-material accuracy claim', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(args.run/'learning_summary.png', dpi=160); plt.close(fig)
    (args.run/'motion_metrics.json').write_text(json.dumps(motion, indent=2)+'\n')
    print(json.dumps(motion, indent=2))


if __name__ == '__main__':
    main()
