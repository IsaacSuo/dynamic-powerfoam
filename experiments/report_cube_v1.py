"""Tables for the closed-cube comparison (output/foam_fem/cube_compare_v1/).

Prints Markdown tables (and writes tables.json) grouping runs by specimen and model:
training / old held-out / fresh held-out errors (range over inits), parameters,
degrees of freedom and wall time.
"""
import json
from pathlib import Path

import numpy as np

D = Path('output/foam_fem/cube_compare_v1')
MULTI = Path('output/foam_fem/cube_multistart_v1')
CLASSES = Path('output/foam_fem/cube_classes_v1')
TRAIN = ('train_down', 'train_pull', 'train_twist')
OLD = ('test_side', 'test_local', 'test_strong_down')
FRESH = ('fresh_oblique', 'fresh_local_side', 'fresh_end_bend_y', 'fresh_end_bend_z', 'fresh_mid_top_down', 'fresh_mid_side')
LABELS = {
    'percell_tet': ('DEM · 每块积木各学一组 · 四面体体积', lambda n: 5*n),
    'percell_power': ('DEM · 每块积木各学一组 · 精确 power 体积', lambda n: 5*n),
    'uniform_tet': ('DEM · 所有积木共用一组 · 四面体体积', lambda n: 5),
    'uniform_power': ('DEM · 所有积木共用一组 · 精确 power 体积', lambda n: 5),
    'four': ('DEM · 只有 4 种弹簧（无体积项）', lambda n: 4*n),
    'multi_uniform_tet': ('DEM · 共用一组 · 四面体体积 · 20 起点', lambda n: 5),
    'multi_uniform_power': ('DEM · 共用一组 · 精确 power 体积 · 20 起点', lambda n: 5),
    'class_classes_tet': ('DEM · 按位置分类（内部/面/棱/角）· 四面体体积 · 20 起点', lambda n: 10 if n == 16 else 20),
    'class_classes_power': ('DEM · 按位置分类（内部/面/棱/角）· 精确 power 体积 · 20 起点', lambda n: 10 if n == 16 else 20),
    'femcentre': ('同分辨率 FEM（节点 = 积木中心）', lambda n: 2),
    'femfine': ('细网格 FEM（独立网格）', lambda n: 2),
}


def errors(run):
    report = json.loads((run/'report.json').read_text())
    values = dict(report['normalized_rmse'])
    for f in run.glob('eval_*.json'):
        values.update(json.loads(f.read_text())['normalized_rmse'])
    if 'test_strong_down' not in values and 'train_down' in values:
        values['test_strong_down'] = values['train_down']  # linear dynamics: same normalized error
    rms = lambda keys: 100*float(np.sqrt(np.mean([values[k]**2 for k in keys])))
    return dict(train=rms(TRAIN), old=rms(OLD), fresh=rms(FRESH), per_case={k: 100*v for k, v in values.items()},
                seconds=report.get('seconds'), dofs=report.get('free_dofs'), converged=report.get('stop_message', '')[:4])


def main():
    out, lines = {}, []
    for spec, cells in (('bar', 16), ('block', 64)):
        lines.append(f'\n### {"细杆 4×2×2" if spec == "bar" else "厚试件 4×4×4"}\n')
        lines.append('| 模型 | 参数 | 训练 | 旧考题 | 新考题 | 按训练误差选出的起点：新考题 | 单次训练耗时 | 完成/起点 |')
        lines.append('|---|---:|---|---|---|---|---|---|')
        for key, (label, params) in LABELS.items():
            if key.startswith('multi_'):
                base, pattern = MULTI, f'{spec}_{key[6:]}_s*'
            elif key.startswith('class_'):
                base, pattern = CLASSES, f'{spec}_{key[6:]}_s*'
            else:
                base, pattern = D, f'{spec}_{key}_*'
            allruns = [r for r in sorted(base.glob(pattern)) if r.is_dir() and not r.name.endswith('_stopped')]
            runs = [r for r in allruns if (r/'report.json').exists() and list(r.glob('eval_*.json')) or
                    (r/'report.json').exists() and key == 'femfine']
            started = len(allruns)
            if not runs:
                lines.append(f'| {label} | {params(cells)} | — | — | — | — | — | 0/{started} |')
                continue
            rows = [errors(r) for r in runs]
            out[f'{spec}_{key}'] = {r.name: e for r, e in zip(runs, rows)}
            fmt = lambda k: (f'{min(e[k] for e in rows):.1f}%' if len(rows) == 1 or
                             abs(max(e[k] for e in rows)-min(e[k] for e in rows)) < .05 else
                             f'{min(e[k] for e in rows):.1f}–{max(e[k] for e in rows):.1f}%')
            secs = [e['seconds'] for e in rows if e['seconds']]
            show = lambda t: f'{t:.0f} 秒' if t < 90 else f'{t/60:.0f} 分钟'
            time_s = (show(min(secs)) if len(secs) == 1 else f'{show(min(secs))}–{show(max(secs))}') if secs else '—'
            chosen = min(rows, key=lambda e: e['train'])
            lines.append(f'| {label} | {params(cells)} | {fmt("train")} | {fmt("old")} | {fmt("fresh")} | '
                         f'{chosen["fresh"]:.1f}%（训练 {chosen["train"]:.1f}%） | {time_s} | {len(rows)}/{started} |')
    text = '\n'.join(lines)
    print(text)
    (D/'tables.json').write_text(json.dumps(out, indent=1, ensure_ascii=False))
    (D/'tables.md').write_text(text+'\n')


if __name__ == '__main__':
    main()
