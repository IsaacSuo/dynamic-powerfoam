"""Figures and animations for the 2026-10-09 group-meeting report.

Writes PNG charts and GIF animations to output/report_v1/. Reads only finished
results (FEM targets, frozen parameters, report.json files); no fitting.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from experiments.foam_linear_inverse import LinearizedFoam, load_case

OUT = Path('output/report_v1')
FEM = Path('output/foam_fem')
DIAG = FEM/'linear_diagnose_v1'
BLOCK = FEM/'block_diagnose_v1'

font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False, 'font.size': 15})
SURFACE, INK, MUTED, GRID = '#fbfbf8', '#14213d', '#52514e', '#e3e2dc'
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.grid(axis='x', color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def rms(values):
    return 100*float(np.sqrt(np.mean(np.square(values))))


def report(path):
    return json.loads((path/'report.json').read_text())['normalized_rmse']


def fresh(path, name):
    return json.loads((path/name).read_text())['normalized_rmse']


# ---------------------------------------------------------------- specimen figure
def boundary_triangles(geometry_npz):
    with np.load(geometry_npz) as g:
        vertices, tets = g['vertices'], g['tetrahedra'][:, :4]
    faces = np.sort(np.concatenate([tets[:, [0, 1, 2]], tets[:, [0, 1, 3]], tets[:, [0, 2, 3]], tets[:, [1, 2, 3]]]), axis=1)
    unique, counts = np.unique(faces, axis=0, return_counts=True)
    return vertices, unique[counts == 1]


def draw_specimen(ax, geometry_npz, patch, title):
    vertices, tris = boundary_triangles(geometry_npz)
    poly = Poly3DCollection(vertices[tris], facecolor='#cfd8e6', edgecolor='none', alpha=.35)
    ax.add_collection3d(poly)
    p, pairs, fixed = patch['points'], patch['pairs'], patch['fixed']
    for i, j in pairs:
        ax.plot(*p[[i, j]].T, color=MUTED, linewidth=1.2)
    ax.scatter(*p[~fixed].T, s=60, color=BLUE, depthshade=False, label='自由积木')
    ax.scatter(*p[fixed].T, s=60, color=INK, marker='s', depthshade=False, label='固定积木（x=0 层）')
    lim = vertices.min(0), vertices.max(0)
    ax.set_xlim(lim[0][0], lim[1][0]); ax.set_ylim(lim[0][1], lim[1][1]); ax.set_zlim(lim[0][2], lim[1][2])
    ax.set_box_aspect(lim[1]-lim[0])
    ax.view_init(elev=22, azim=-58)
    ax.set_axis_off()
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=18, pad=0)


def specimen_figure():
    fig = plt.figure(figsize=(16, 7), facecolor=SURFACE)
    bar = dict(np.load(FEM/'reference_edge007/patch.npz'))
    block = dict(np.load(FEM/'block_reference_edge007/patch.npz'))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    draw_specimen(ax, FEM/'mesh_edge007/geometry.npz', bar, '细杆 4×2×2：16 块积木，28 根黏结')
    ax.legend(loc='lower left', frameon=False, fontsize=14)
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    draw_specimen(ax, FEM/'block_mesh_edge010/geometry.npz', block, '厚试件 4×4×4：64 块积木，144 根黏结')
    fig.subplots_adjust(left=0, right=1, top=.95, bottom=0, wspace=0)
    fig.savefig(OUT/'specimens.png', dpi=110, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- animations
def dem_positions(patch, params_npz, columns, data, substeps):
    model = LinearizedFoam(patch, volume=True)
    theta = np.load(params_npz)['x'].reshape(len(patch['points']), columns)
    _, positions, _ = model.simulate(theta, data, substeps)
    return positions


def animate(panels, path, scale, title):
    """panels: list of (label, patch, fem_positions, dem_positions or None)."""
    fig = plt.figure(figsize=(6*len(panels), 5.2), facecolor=SURFACE)
    axes, artists = [], []
    for k, (label, patch, target, dem) in enumerate(panels):
        ax = fig.add_subplot(1, len(panels), k+1, projection='3d')
        p0 = patch['points']
        moved = p0[None]+scale*(target-p0[None])
        lo, hi = np.minimum(p0.min(0), moved.min((0, 1)))-.3, np.maximum(p0.max(0), moved.max((0, 1)))+.3
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(hi-lo, zoom=1.25); ax.view_init(elev=20, azim=-60); ax.set_axis_off(); ax.set_facecolor(SURFACE)
        ax.set_title(label, color=INK, fontsize=16)
        lines = {'fem': [ax.plot([], [], [], color=BLUE, linewidth=1.6)[0] for _ in patch['pairs']]}
        dots = {'fem': ax.plot([], [], [], 'o', color=BLUE, markersize=6)[0]}
        if dem is not None:
            lines['dem'] = [ax.plot([], [], [], color=ORANGE, linewidth=1.6, linestyle='--')[0] for _ in patch['pairs']]
            dots['dem'] = ax.plot([], [], [], 'o', color=ORANGE, markersize=5, markerfacecolor='none')[0]
        artists.append((patch, target, dem, lines, dots))
    handles = [plt.Line2D([], [], color=BLUE, marker='o', label='FEM 标准答案')]
    if any(dem is not None for *_, dem in panels):
        handles.append(plt.Line2D([], [], color=ORANGE, marker='o', markerfacecolor='none', linestyle='--', label='DEM 预测'))
    fig.legend(handles=handles, loc='lower center', ncol=2, frameon=False, fontsize=15)
    fig.suptitle(title, color=INK, fontsize=18)
    count = len(panels[0][2])

    def update(frame):
        out = []
        for patch, target, dem, lines, dots in artists:
            p0 = patch['points']
            for key, series in (('fem', target), ('dem', dem)):
                if series is None:
                    continue
                q = p0+scale*(series[frame]-p0)
                for line, (i, j) in zip(lines[key], patch['pairs']):
                    line.set_data_3d(*q[[i, j]].T)
                dots[key].set_data_3d(*q.T)
                out += lines[key]+[dots[key]]
        return out
    FuncAnimation(fig, update, frames=count, blit=False).save(path, writer=PillowWriter(fps=6), dpi=80)
    plt.close(fig)


def animations():
    bar = dict(np.load(FEM/'reference_edge007/patch.npz'))
    block = dict(np.load(FEM/'block_reference_edge007/patch.npz'))
    target = lambda ref, name: load_case(ref, name)['target']
    # 1. FEM ground truth: four representative loads on the bar.
    cases = [('下压（训练）', FEM/'reference_edge007', 'train_down'), ('拧（训练）', FEM/'reference_edge007', 'train_twist'),
             ('中间上表面下压（新考题）', FEM/'reference_edge007_fresh_v1', 'fresh_mid_top_down'),
             ('侧弯力偶（新考题）', FEM/'reference_edge007_fresh_v1', 'fresh_end_bend_z')]
    animate([(label, bar, target(ref, name), None) for label, ref, name in cases], OUT/'fem_targets_bar.gif', 100.,
            'FEM 标准答案（位移放大 100 倍）')
    # 2. FEM vs trained DEM (five-mode, per cell) on a training and a fresh load, bar and block.
    panels = []
    for label, ref, name, steps in (('细杆 · 下压（训练）', FEM/'reference_edge007', 'train_down', 16),
                                    ('细杆 · 侧弯力偶（新考题）', FEM/'reference_edge007_fresh_v1', 'fresh_end_bend_z', 32)):
        data = load_case(ref, name)
        panels.append((label, bar, data['target'], dem_positions(bar, DIAG/'cell5_wide_s1/parameters.npz', 5, data, steps)))
    animate(panels, OUT/'fem_vs_dem_bar.gif', 100., '细杆：FEM 与 DEM（位移放大 100 倍）')
    tip_chart(panels, 'tip_bar.png')
    panels = []
    for label, name, steps in (('厚试件 · 下压（训练）', 'train_down', 16), ('厚试件 · 局部侧推（新考题）', 'fresh_local_side', 32)):
        data = load_case(FEM/'block_reference_edge007', name)
        panels.append((label, block, data['target'], dem_positions(block, BLOCK/'cell5_zero/parameters.npz', 5, data, steps)))
    animate(panels, OUT/'fem_vs_dem_block.gif', 100., '厚试件：FEM 与 DEM（位移放大 100 倍）')
    tip_chart(panels, 'tip_block.png')


def tip_chart(panels, name):
    """Displacement of the cell with the largest DEM error, along its main axis, FEM vs DEM."""
    fig, axes = plt.subplots(1, len(panels), figsize=(8*len(panels), 4.6), facecolor=SURFACE)
    for ax, (label, patch, target, dem) in zip(np.atleast_1d(axes), panels):
        style(ax)
        ax.grid(axis='y', color=GRID)
        worst = int(np.argmax(np.sum((dem-target)**2, axis=(0, 2))))
        fem_d = target[:, worst]-patch['points'][worst]
        axis = int(np.argmax(np.abs(fem_d).max(0)))
        dem_d = dem[:, worst]-patch['points'][worst]
        frames = np.arange(len(fem_d))
        ax.plot(frames, 1e3*fem_d[:, axis], color=BLUE, linewidth=2.5, label='FEM 标准答案')
        ax.plot(frames, 1e3*dem_d[:, axis], color=ORANGE, linewidth=2.5, linestyle='--', label='DEM 预测')
        cells = ~patch['fixed']
        err = 100*np.sqrt(np.mean((dem[1:, cells]-target[1:, cells])**2)/np.mean((target[1:, cells]-patch['points'][cells])**2))
        ax.set_title(f'{label}：全部积木误差 {err:.1f}%', color=INK, fontsize=17, loc='left')
        ax.set_xlabel('帧（共 24 帧，约 1.5 个振动周期）', color=MUTED)
        ax.set_ylabel(f'误差最大的积木：{"xyz"[axis]} 向位移（×0.001）', color=MUTED)
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=15, loc='best')
    fig.tight_layout()
    fig.savefig(OUT/name, dpi=110, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- charts
def speed_chart():
    fig, axes = plt.subplots(1, 2, figsize=(16, 4.8), facecolor=SURFACE)
    rows = [(axes[0], '训练：一次目标计算（3 段运动 + 梯度）', ['原实现（稠密雅可比）', '线性化训练器'], [62., .05], 's', True),
            (axes[1], 'FEM 数据生成：一次 16 小步模拟（edge007）', ['原求解器（SuperLU）', 'CHOLMOD'], [723., 176.], 's', False)]
    for ax, title, labels, values, unit, log in rows:
        style(ax)
        ax.barh(labels[::-1], values[::-1], color=[BLUE, MUTED], height=.5)
        if log:
            ax.set_xscale('log')
            ax.set_xticks([.01, .1, 1, 10, 100], ['0.01 s', '0.1 s', '1 s', '10 s', '100 s'])
        for k, v in enumerate(values[::-1]):
            ax.text(v*(1.15 if log else 1.02), k, f'{v:g} {unit}', va='center', color=INK, fontsize=16)
        ax.set_title(title, color=INK, fontsize=17, loc='left')
        ax.set_xlim(right=max(values)*(8 if log else 1.3))
        ax.tick_params(axis='y', labelsize=16, colors=INK)
    fig.tight_layout()
    fig.savefig(OUT/'speed.png', dpi=110, facecolor=SURFACE)
    plt.close(fig)


def capacity_chart():
    runs = [('原弹簧 5 种', DIAG/'cap12_cell5', 80), ('+ 几何方向着力点', DIAG/'cap12_levergeomcell5', 96),
            ('+ 形状项', DIAG/'shape6_everything', 96), ('守规矩的万能弹簧', DIAG/'cap12_pairobj', 588),
            ('任意刚度 · 积木表示上限', DIAG/'cap12_general', 2628)]
    values = [rms(list(report(path).values())) for _, path, _ in runs]
    fig, ax = plt.subplots(figsize=(16, 6), facecolor=SURFACE)
    style(ax)
    labels = [f'{name}（{n} 参数）' for name, _, n in runs]
    ax.barh(labels[::-1], values[::-1], color=BLUE, height=.55)
    ax.axvspan(0, 1.0, color='#e8eef8', zorder=0)
    ax.set_ylim(-1.1, len(runs)-.5)
    ax.text(.5, -.85, 'FEM 自身网格误差 ≈1%', ha='center', va='center', color=MUTED, fontsize=14)
    for k, v in enumerate(values[::-1]):
        ax.text(v+.06, k, f'{v:.1f}%', va='center', color=INK, fontsize=16)
    ax.set_xlabel('12 种运动全部训练后的拟合误差（位置 RMSE / 运动幅度）', color=MUTED)
    ax.tick_params(axis='y', labelsize=16, colors=INK)
    ax.set_xlim(0, max(values)*1.18)
    fig.tight_layout()
    fig.savefig(OUT/'capacity.png', dpi=110, facecolor=SURFACE)
    plt.close(fig)
    return dict(zip([r[0] for r in runs], values))


def generalization_chart():
    fresh_bar = 'eval_reference_edge007_fresh_v1.json'
    fresh_block = 'eval_block_reference_edge007.json'
    def bar_rows(names):
        train, test = [], []
        for name in names:
            r = report(DIAG/name)
            train.append(rms(list(r.values())[:3]))
            test.append(rms(list(fresh(DIAG/name, fresh_bar).values())))
        return train, test
    def block_rows(names):
        train, test = [], []
        for name in names:
            e = fresh(BLOCK/name, fresh_block)
            train.append(rms([e[k] for k in ('train_down', 'train_pull', 'train_twist')]))
            test.append(rms([e[k] for k in e if k.startswith('fresh')]))
        return train, test
    rows = [('细杆 · 原弹簧 5 种（80 参数）', *bar_rows(['cell5_wide', 'cell5_wide_s1', 'cell5_wide_s2', 'cell5_wide_s3', 'cell5_wide_s4', 'cell5_wide_s5'])),
            ('细杆 · 几何方向着力点', *bar_rows(['levergeom5_from5', 'levergeomcell5_from5', 'levergeom5_seed1', 'levergeom5_seed2', 'levergeomcell5_seed1', 'levergeomcell5_seed2'])),
            ('细杆 · 守规矩的万能弹簧（588）', *bar_rows(['pairobj_from4'])),
            ('细杆 · 任意刚度（2628）', *bar_rows(['general_from5', 'inertia_from5'])),
            ('厚试件 · 原弹簧 5 种（320 参数）', *block_rows(['cell5_zero', 'cell5_s1']))]
    fig, ax = plt.subplots(figsize=(16, 6.5), facecolor=SURFACE)
    style(ax)
    ax.set_xscale('log')
    for k, (label, train, test) in enumerate(rows[::-1]):
        for values, color, offset in ((train, BLUE, .12), (test, ORANGE, -.12)):
            ax.plot([min(values), max(values)], [k+offset]*2, color=color, linewidth=3, solid_capstyle='round')
            ax.plot(values, [k+offset]*len(values), 'o', color=color, markersize=9, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.text(max(test)*1.12, k-.12, (f'{min(test):.1f}–{max(test):.1f}%' if max(test) < 20 else f'{min(test):.0f}–{max(test):.0f}%') if len(test) > 1 else f'{test[0]:.0f}%', va='center', color=INK, fontsize=14)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows[::-1]], fontsize=16, color=INK)
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100], ['1%', '2%', '5%', '10%', '20%', '50%', '100%'])
    ax.set_xlim(.4, 300)
    ax.plot([], [], 'o-', color=BLUE, label='训练误差（3 种运动）')
    ax.plot([], [], 'o-', color=ORANGE, label='新考题误差（6 种，训练前设计）')
    ax.legend(loc='lower right', frameon=False, fontsize=15)
    ax.set_xlabel('位置 RMSE / 运动幅度（对数轴；每个点是一个起点）', color=MUTED)
    fig.tight_layout()
    fig.savefig(OUT/'generalization.png', dpi=110, facecolor=SURFACE)
    plt.close(fig)
    return {label: (min(test), max(test)) for label, _, test in rows}


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    summary = dict(capacity=capacity_chart(), generalization=generalization_chart())
    speed_chart()
    specimen_figure()
    animations()
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, ensure_ascii=False, indent=1))
