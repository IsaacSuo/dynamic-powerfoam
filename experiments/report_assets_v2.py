"""Explanatory figures, per-load animations and result tables for the report text.

Writes to output/report_v1/: model explainers, 12 FEM load animations (bar), and
tables.json with every number quoted in the report text. Reads finished results only.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyBboxPatch, Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import Voronoi, Delaunay

from experiments.foam_linear_inverse import load_case

OUT = Path('output/report_v1')
FEM = Path('output/foam_fem')
DIAG = FEM/'linear_diagnose_v1'
BLOCK = FEM/'block_diagnose_v1'
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False, 'font.size': 15})
SURFACE, INK, MUTED, GRID = '#fbfbf8', '#14213d', '#52514e', '#e3e2dc'
BLUE, ORANGE, AQUA, LIGHT = '#2a78d6', '#eb6834', '#1baf7a', '#cfd8e6'

LOADS = [  # (file stem, label, reference dir, case)
    ('01_train_down', '训练 1：右端下压', 'reference_edge007', 'train_down'),
    ('02_train_pull', '训练 2：右端外拉', 'reference_edge007', 'train_pull'),
    ('03_train_twist', '训练 3：右端绕长轴拧', 'reference_edge007', 'train_twist'),
    ('04_test_side', '考试 1：右端侧推', 'reference_edge007', 'test_side'),
    ('05_test_local', '考试 2：右端一角下压', 'reference_edge007', 'test_local'),
    ('06_test_strong_down', '考试 3：两倍力下压', 'reference_edge007', 'test_strong_down'),
    ('07_fresh_oblique', '考试 4：右端斜推', 'reference_edge007_fresh_v1', 'fresh_oblique'),
    ('08_fresh_local_side', '考试 5：右端一角侧推', 'reference_edge007_fresh_v1', 'fresh_local_side'),
    ('09_fresh_end_bend_y', '考试 6：右端下弯力偶（只弯不推）', 'reference_edge007_fresh_v1', 'fresh_end_bend_y'),
    ('10_fresh_end_bend_z', '考试 7：右端侧弯力偶（只弯不推）', 'reference_edge007_fresh_v1', 'fresh_end_bend_z'),
    ('11_fresh_mid_top_down', '考试 8：中间一层上表面下压', 'reference_edge007_fresh_v1', 'fresh_mid_top_down'),
    ('12_fresh_mid_side', '考试 9：中间一层侧面侧推', 'reference_edge007_fresh_v1', 'fresh_mid_side'),
]


# ------------------------------------------------------------ 1. Power Foam -> blocks
def powerfoam_to_blocks():
    rng = np.random.default_rng(3)
    pts = rng.uniform(0, 1, (26, 2))
    mirror = np.concatenate([pts, pts*[-1, 1], pts*[1, -1], [2, 0]+pts*[-1, 1], [0, 2]+pts*[1, -1]])
    vor = Voronoi(mirror)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.4), facecolor=SURFACE)
    colors = plt.cm.YlOrBr(np.linspace(.15, .45, len(pts)))
    for ax in axes:
        ax.set_facecolor(SURFACE); ax.set_aspect('equal'); ax.axis('off'); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        for k in range(len(pts)):
            region = vor.regions[vor.point_region[k]]
            ax.add_patch(Polygon(vor.vertices[region], closed=True, facecolor=colors[k] if ax is axes[0] else '#eef2f8',
                                 edgecolor='#8a6d3b' if ax is axes[0] else '#b8c4d6', linewidth=1.5))
    neighbors = {tuple(sorted(p)) for p in vor.ridge_points if p[0] < len(pts) and p[1] < len(pts)}
    for i, j in neighbors:
        axes[1].plot(*pts[[i, j]].T, color=MUTED, linewidth=2.2, zorder=3)
    axes[1].scatter(*pts.T, s=80, color=BLUE, zorder=4)
    axes[0].set_title('Power Foam：空间被分成许多格子', color=INK, fontsize=18)
    axes[1].set_title('我们的模型：每个格子 = 一块刚体积木\n相邻格子之间连一根黏结', color=INK, fontsize=18)
    fig.tight_layout()
    fig.savefig(OUT/'model_1_powerfoam_to_blocks.png', dpi=110, facecolor=SURFACE)
    plt.close(fig)


# ------------------------------------------------------------ 2. bond deformation modes
def cube_faces(center, rotation, half=.4):
    corners = np.array([[x, y, z] for x in (-half, half) for y in (-half, half) for z in (-half, half)])
    world = center+corners@rotation.T
    idx = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]
    return [world[f] for f in idx]


def rotation(axis, angle):
    axis = np.asarray(axis, float)/np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3)+np.sin(angle)*K+(1-np.cos(angle))*K@K


def bond_modes():
    modes = [('拉开 / 挤近 → 法向力', lambda s: (np.array([1+.25*s, 0, 0]), np.eye(3))),
             ('左右错开 → 切向力', lambda s: (np.array([1, .25*s, 0]), np.eye(3))),
             ('相对掰弯 → 弯曲力矩', lambda s: (np.array([1, 0, -.12*s]), rotation([0, 1, 0], .45*s))),
             ('绕连线相对拧 → 扭转力矩', lambda s: (np.array([1, 0, 0]), rotation([1, 0, 0], .6*s)))]
    fig = plt.figure(figsize=(12, 9), facecolor=SURFACE)
    axes = []
    for k, (title, _) in enumerate(modes):
        ax = fig.add_subplot(2, 2, k+1, projection='3d')
        ax.set_facecolor(SURFACE); ax.set_axis_off()
        ax.set_xlim(-.6, 1.8); ax.set_ylim(-.8, .8); ax.set_zlim(-.8, .8)
        ax.set_box_aspect((2.4, 1.6, 1.6), zoom=1.2); ax.view_init(elev=18, azim=-62)
        ax.set_title(title, color=INK, fontsize=18)
        axes.append(ax)
    fig.suptitle('黏结抵抗两块积木之间的 4 种相对变形（左块固定）', color=INK, fontsize=19)
    fig.text(.5, .04, '深蓝线 = 黏结：从积木中心经过接触面连到另一块积木的中心', ha='center', color=MUTED, fontsize=15)
    drawn = []

    def update(frame):
        s = np.sin(2*np.pi*frame/30)
        for artist in drawn:
            artist.remove()
        drawn.clear()
        for ax, (_, motion) in zip(axes, modes):
            center, R = motion(s)
            for faces, color in ((cube_faces(np.zeros(3), np.eye(3)), '#c9d3e3'), (cube_faces(center, R), '#f6c9b0')):
                drawn.append(ax.add_collection3d(Poly3DCollection(faces, facecolor=color, edgecolor=MUTED, alpha=.28, linewidth=1)))
            contact_left, contact_right = np.array([.4, 0, 0]), center+R@np.array([-.4, 0, 0])
            path = np.stack([np.zeros(3), contact_left, contact_right, center])
            drawn.extend(ax.plot(*path.T, color='#1f4fa3', linewidth=6, solid_capstyle='round', zorder=10))
            drawn.extend(ax.plot(*path[[0, 3]].T, 'o', color='#1f4fa3', markersize=9, zorder=11))
        return drawn
    FuncAnimation(fig, update, frames=30).save(OUT/'model_2_bond_modes.gif', writer=PillowWriter(fps=12), dpi=75)
    plt.close(fig)


# ------------------------------------------------------------ 3. volume term
def volume_term():
    grid = np.array([[x, y] for x in range(3) for y in range(3)], float)
    tri = Delaunay(grid)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=SURFACE)
    for ax, (title, scale) in zip(axes, (('变形前：中间积木周围一圈的“局部体积”', 1.), ('压扁后：局部体积变小 → 体积项把积木往外推', .7))):
        ax.set_facecolor(SURFACE); ax.set_aspect('equal'); ax.axis('off')
        p = grid.copy(); p[:, 1] = 1+(p[:, 1]-1)*scale
        for simplex in tri.simplices:
            shade = '#cfe0f5' if 4 in simplex else '#f1f0ea'
            ax.add_patch(Polygon(p[simplex], closed=True, facecolor=shade, edgecolor='#b8c4d6', linewidth=1.5))
        ax.scatter(*p.T, s=160, color=BLUE, zorder=3)
        ax.scatter(*p[4], s=260, color=ORANGE, zorder=4)
        if scale < 1:
            # Grid index k = 3*x + y: 3/5 are the top/bottom neighbours, 1/7 the left/right ones.
            for k in (3, 5):
                d = np.sign(p[k, 1]-1)
                ax.annotate('', xy=(p[k, 0], p[k, 1]+.4*d), xytext=(p[k, 0], p[k, 1]+.08*d),
                            arrowprops=dict(arrowstyle='->', color=ORANGE, lw=3.5))
            for k in (1, 7):
                d = np.sign(p[k, 0]-1)
                ax.annotate('', xy=(p[k, 0]+.4*d, p[k, 1]), xytext=(p[k, 0]+.08*d, p[k, 1]),
                            arrowprops=dict(arrowstyle='->', color=ORANGE, lw=3.5))
            ax.text(1, -.45, '上下：抵抗压扁    左右：侧面被挤出去', ha='center', color=ORANGE, fontsize=15)
        ax.set_xlim(-.5, 2.5); ax.set_ylim(-.6, 2.6)
        ax.set_title(title, color=INK, fontsize=17)
    fig.text(.5, .03, '实际是三维：用相邻积木中心连成的小四面体来量（这里用平面三角形示意）', ha='center', color=MUTED, fontsize=15)
    fig.tight_layout(rect=(0, .06, 1, 1))
    fig.savefig(OUT/'model_3_volume_term.png', dpi=110, facecolor=SURFACE)
    plt.close(fig)


# ------------------------------------------------------------ 4. training loop
def training_loop():
    fig, ax = plt.subplots(figsize=(14, 6.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE); ax.axis('off'); ax.set_xlim(0, 14); ax.set_ylim(0, 6.5)
    boxes = [(1.2, 4.6, '每块积木的\n5 个硬度倍率'), (5.8, 4.6, 'DEM 仿真\n（给定外力）'), (10.4, 4.6, '积木中心的\n运动轨迹'),
             (10.4, 1.2, '和 FEM 标准答案\n比较 → 误差'), (5.8, 1.2, '算出每个倍率\n该往哪调（梯度）'), (1.2, 1.2, '调整倍率')]
    for x, y, text in boxes:
        ax.add_patch(FancyBboxPatch((x, y), 2.8, 1.2, boxstyle='round,pad=0.15', facecolor='#ffffff', edgecolor=BLUE, linewidth=2))
        ax.text(x+1.4, y+.6, text, ha='center', va='center', color=INK, fontsize=17)
    arrows = [((4.15, 5.2), (5.65, 5.2)), ((8.75, 5.2), (10.25, 5.2)), ((11.8, 4.45), (11.8, 2.55)),
              ((10.25, 1.8), (8.75, 1.8)), ((5.65, 1.8), (4.15, 1.8)), ((2.6, 2.55), (2.6, 4.45))]
    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start, arrowprops=dict(arrowstyle='->', color=MUTED, lw=2.5))
    ax.text(7, 3.35, '重复，直到误差不再下降', ha='center', va='center', color=ORANGE, fontsize=18)
    ax.text(7, .3, '训练只看 3 种运动；材料常数、FEM 内部应力都不给模型看', ha='center', color=MUTED, fontsize=15)
    fig.tight_layout()
    fig.savefig(OUT/'model_4_training_loop.png', dpi=110, facecolor=SURFACE)
    plt.close(fig)


# ------------------------------------------------------------ 5. error definition
def error_definition():
    from experiments.foam_linear_inverse import LinearizedFoam
    patch = dict(np.load(FEM/'reference_edge007/patch.npz'))
    data = load_case(FEM/'reference_edge007_fresh_v1', 'fresh_end_bend_z')
    model = LinearizedFoam(patch, volume=True)
    theta = np.load(DIAG/'cell5_wide_s1/parameters.npz')['x'].reshape(16, 5)
    _, dem, _ = model.simulate(theta, data, 32)
    target, p0 = data['target'], patch['points']
    worst = int(np.argmax(np.sum((dem-target)**2, axis=(0, 2))))
    fem_y, dem_y = 1e3*(target[:, worst, 1]-p0[worst, 1]), 1e3*(dem[:, worst, 1]-p0[worst, 1])
    t = np.arange(len(fem_y))
    fig, ax = plt.subplots(figsize=(14, 5.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.fill_between(t, fem_y, dem_y, color=ORANGE, alpha=.18, label='位置差（误差的来源）')
    ax.plot(t, fem_y, color=BLUE, linewidth=2.5, label='FEM 标准答案')
    ax.plot(t, dem_y, color=ORANGE, linewidth=2.5, linestyle='--', label='DEM 预测')
    ax.set_xlabel('帧', color=MUTED); ax.set_ylabel('某块积木的位移（×0.001）', color=MUTED)
    ax.legend(frameon=False, fontsize=15, loc='upper left')
    ax.set_title('误差 = 所有积木、所有帧的位置差（均方根）÷ 标准答案运动幅度（均方根）', color=INK, fontsize=17, loc='left')
    fig.tight_layout()
    fig.savefig(OUT/'model_5_error_definition.png', dpi=110, facecolor=SURFACE)
    plt.close(fig)


# ------------------------------------------------------------ 6. per-load FEM animations
def load_animation(stem, label, ref, case, scale=100.):
    patch = dict(np.load(FEM/'reference_edge007/patch.npz'))
    data = load_case(FEM/ref, case)
    target, p0 = data['target'], patch['points']
    pattern = np.asarray(data['pattern'])[:, :3]
    frames = len(target)
    times = np.arange(frames)*data['dt']
    pulse = np.where(times < data['pulse'], np.sin(np.pi*times/data['pulse'])**2, 0.)
    arrows = pattern/max(np.abs(pattern).max(), 1e-30)*.9
    moved = p0[None]+scale*(target-p0[None])
    lo = np.minimum(moved.min((0, 1)), (p0+arrows).min(0))-.4
    hi = np.maximum(moved.max((0, 1)), (p0+arrows).max(0))+.4
    fig = plt.figure(figsize=(8, 5.2), facecolor=SURFACE)
    ax = fig.add_subplot(projection='3d')
    ax.set_facecolor(SURFACE); ax.set_axis_off()
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi-lo, zoom=1.3); ax.view_init(elev=20, azim=-60)
    fig.suptitle(label, color=INK, fontsize=18, y=.95)
    fig.text(.5, .04, '蓝：积木中心与黏结（位移放大 100 倍）   橙箭头：外力（加载阶段）   方块：固定的一层',
             ha='center', color=MUTED, fontsize=12)
    lines = [ax.plot([], [], [], color=BLUE, linewidth=1.8)[0] for _ in patch['pairs']]
    free = ~patch['fixed']
    dots = ax.plot([], [], [], 'o', color=BLUE, markersize=6)[0]
    fixed = ax.plot(*p0[patch['fixed']].T, 's', color=INK, markersize=7)[0]
    quiver = [None]
    loaded = np.abs(pattern).sum(1) > 0

    def update(frame):
        q = p0+scale*(target[frame]-p0)
        for line, (i, j) in zip(lines, patch['pairs']):
            line.set_data_3d(*q[[i, j]].T)
        dots.set_data_3d(*q[free].T)
        if quiver[0] is not None:
            quiver[0].remove(); quiver[0] = None
        if pulse[frame] > .02:
            a = arrows[loaded]*pulse[frame]
            quiver[0] = ax.quiver(*q[loaded].T, *a.T, color=ORANGE, linewidth=2.5, arrow_length_ratio=.35)
        return lines+[dots, fixed]
    FuncAnimation(fig, update, frames=frames).save(OUT/'loads'/f'{stem}.gif', writer=PillowWriter(fps=6), dpi=80)
    plt.close(fig)


# ------------------------------------------------------------ 7. FEM vs DEM comparison animations
def outer_surface(points):
    """Outer faces of the cell lattice (unit cubes around centres) as corner index quads.

    Corners sit at half-integer lattice positions; a corner moves with the mean
    displacement of the existing cells around it, so the surface stays closed.
    """
    axes = [np.unique(points[:, k]) for k in range(3)]
    index = np.stack([np.searchsorted(axes[k], points[:, k]) for k in range(3)], axis=1)
    occupied = {tuple(i): n for n, i in enumerate(index)}
    corner_ids, corner_cells, quads = {}, [], []
    def corner(c):
        if c not in corner_ids:
            corner_ids[c] = len(corner_cells)
            cells = [occupied[t] for t in ((c[0]-a, c[1]-b, c[2]-d) for a in (0, 1) for b in (0, 1) for d in (0, 1)) if t in occupied]
            corner_cells.append(cells)
        return corner_ids[c]
    for i in index:
        for axis in range(3):
            for side in (-1, 1):
                nb = list(i); nb[axis] += side
                if tuple(nb) in occupied:
                    continue
                o = [k for k in range(3) if k != axis]
                base = list(i); base[axis] += (side+1)//2
                quad = []
                for da, db in ((0, 0), (1, 0), (1, 1), (0, 1)):
                    c = list(base); c[o[0]] += da; c[o[1]] += db
                    quad.append(corner(tuple(c)))
                quads.append(quad)
    spacing = np.array([axes[k][1]-axes[k][0] for k in range(3)])
    origin = np.array([axes[k][0] for k in range(3)])-spacing/2
    positions = origin+spacing*np.array(sorted(corner_ids, key=corner_ids.get))
    return positions, corner_cells, np.array(quads)


def compare_animation(stem, label, patch_dir, ref, case, run_dir, substeps, scale=100., error_gain=4.):
    """Left: FEM and DEM bodies as translucent solids (overlap blends);
    right: the worst cell's displacement over time with a moving cursor."""
    from experiments.foam_linear_inverse import LinearizedFoam
    patch = dict(np.load(FEM/patch_dir/'patch.npz'))
    data = load_case(FEM/ref, case)
    model = LinearizedFoam(patch, volume=True)
    theta = np.load(run_dir/'parameters.npz')['x'].reshape(len(patch['points']), 5)
    _, dem, _ = model.simulate(theta, data, substeps)
    target, p0 = data['target'], patch['points']
    cells = ~patch['fixed']
    err = 100*np.sqrt(np.mean((dem[1:, cells]-target[1:, cells])**2)/np.mean((target[1:, cells]-p0[cells])**2))
    worst = int(np.argmax(np.sum((dem-target)**2, axis=(0, 2))))
    axis = int(np.argmax(np.abs(target[:, worst]-p0[worst]).max(0)))
    corners, corner_cells, quads = outer_surface(p0)
    def corner_disp(disp):
        return np.array([disp[c].mean(0) for c in corner_cells])
    fem_c = np.array([corners+scale*corner_disp(target[f]-p0) for f in range(len(target))])
    dem_c = np.array([fem_c[f]+scale*error_gain*corner_disp(dem[f]-target[f]) for f in range(len(target))])
    lo = np.minimum(fem_c.min((0, 1)), dem_c.min((0, 1)))-.2
    hi = np.maximum(fem_c.max((0, 1)), dem_c.max((0, 1)))+.2
    fig = plt.figure(figsize=(15, 6), facecolor=SURFACE)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.35, 1])
    ax = fig.add_subplot(grid[0], projection='3d')
    ax.set_facecolor(SURFACE); ax.set_axis_off()
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi-lo, zoom=1.4); ax.view_init(elev=20, azim=-60)
    ax.set_title('蓝：标准答案　橙：预测（半透明，重叠处混色）\n位移放大 100 倍；两者之差再放大 4 倍', color=INK, fontsize=14)
    ax2 = fig.add_subplot(grid[1])
    ax2.set_facecolor(SURFACE)
    for side in ('top', 'right'):
        ax2.spines[side].set_visible(False)
    t = np.arange(len(target))
    ax2.plot(t, 1e3*(target[:, worst, axis]-p0[worst, axis]), color=BLUE, linewidth=2.5, label='标准答案（FEM）')
    ax2.plot(t, 1e3*(dem[:, worst, axis]-p0[worst, axis]), color=ORANGE, linewidth=2.5, linestyle='--', label='预测（DEM）')
    cursor = ax2.axvline(0, color=MUTED, linewidth=1.5)
    ax2.set_xlabel('帧', color=MUTED); ax2.set_ylabel(f'{"xyz"[axis]} 向位移（×0.001）', color=MUTED)
    ax2.legend(frameon=False, fontsize=13, loc='upper left')
    ax2.set_title('误差最大的那块积木', color=INK, fontsize=14, loc='left')
    fig.suptitle(f'{label}　全部积木误差 {err:.1f}%', color=INK, fontsize=18)
    drawn = []

    def update(frame):
        for artist in drawn:
            artist.remove()
        drawn.clear()
        for corner_pos, color in ((fem_c[frame], BLUE), (dem_c[frame], ORANGE)):
            solid = Poly3DCollection(corner_pos[quads], facecolor=color, edgecolor=color, linewidth=.3, alpha=.3)
            drawn.append(ax.add_collection3d(solid))
        cursor.set_xdata([frame, frame])
        return drawn+[cursor]
    fig.tight_layout(rect=(0, 0, 1, .93))
    FuncAnimation(fig, update, frames=len(target)).save(OUT/f'{stem}.gif', writer=PillowWriter(fps=5), dpi=80)
    plt.close(fig)
    return err


# ------------------------------------------------------------ tables
def tables():
    def rel(values):
        return [100*float(v) for v in values]
    cases = ['train_down', 'train_pull', 'train_twist', 'test_side', 'test_local', 'test_strong_down',
             'fresh_oblique', 'fresh_local_side', 'fresh_end_bend_y', 'fresh_end_bend_z', 'fresh_mid_top_down', 'fresh_mid_side']
    def bar_errors(run):
        r = json.loads((DIAG/run/'report.json').read_text())['normalized_rmse']
        e = json.loads((DIAG/run/'eval_reference_edge007_fresh_v1.json').read_text())['normalized_rmse']
        return {c: 100*(r[c] if c in r else e[c]) for c in cases}
    def block_errors(run):
        e = json.loads((BLOCK/run/'eval_block_reference_edge007.json').read_text())['normalized_rmse']
        e['test_strong_down'] = e['train_down']  # linear dynamics: identical normalized error
        return {c: 100*e[c] for c in cases}
    bar_runs = ['cell5_wide', 'cell5_wide_s1', 'cell5_wide_s2', 'cell5_wide_s3', 'cell5_wide_s4', 'cell5_wide_s5']
    block_runs = ['cell5_zero', 'cell5_s1']
    per_case = {}
    for name, runs, fn in (('bar', bar_runs, bar_errors), ('block', block_runs, block_errors)):
        values = np.array([[fn(r)[c] for c in cases] for r in runs])
        per_case[name] = {c: dict(min=float(values[:, k].min()), median=float(np.median(values[:, k])), max=float(values[:, k].max()))
                          for k, c in enumerate(cases)}
    rms = lambda v: float(np.sqrt(np.mean(np.square(v))))
    def group(errors, subset):
        return rms([errors[c] for c in subset])
    train, old, fresh = cases[:3], cases[3:6], cases[6:]
    summary = {}
    for name, runs, fn in (('bar_percell', bar_runs, bar_errors), ('block_percell', block_runs, block_errors)):
        rows = [fn(r) for r in runs]
        summary[name] = {k: [min(group(r, s) for r in rows), max(group(r, s) for r in rows)] for k, s in (('train', train), ('old', old), ('fresh', fresh))}
    uni = json.loads((FEM/'block_fused_v1/uniform_s1/eval_block_reference_edge007.json').read_text())['normalized_rmse']
    uni['test_strong_down'] = uni['train_down']
    summary['block_uniform'] = {k: [100*group(uni, s)]*2 for k, s in (('train', train), ('old', old), ('fresh', fresh))}
    capacity = {}
    for label, run in (('ours', 'cap12_cell5'), ('shift', 'cap12_levergeomcell5'), ('shape', 'shape6_everything'),
                       ('pair_any', 'cap12_pairobj'), ('any', 'cap12_general')):
        capacity[label] = 100*rms(list(json.loads((DIAG/run/'report.json').read_text())['normalized_rmse'].values()))
    return dict(per_case=per_case, summary=summary, capacity=capacity)


if __name__ == '__main__':
    (OUT/'loads').mkdir(parents=True, exist_ok=True)
    powerfoam_to_blocks()
    volume_term()
    training_loop()
    error_definition()
    bond_modes()
    for stem, label, ref, case in LOADS:
        load_animation(stem, label, ref, case)
    compare = [('compare_bar_train_down', '细杆 · 右端下压（训练）', 'reference_edge007', 'reference_edge007', 'train_down', DIAG/'cell5_wide_s1', 16),
               ('compare_bar_bend_z', '细杆 · 右端侧弯力偶（考试）', 'reference_edge007', 'reference_edge007_fresh_v1', 'fresh_end_bend_z', DIAG/'cell5_wide_s1', 32),
               ('compare_block_train_down', '厚试件 · 右端下压（训练）', 'block_reference_edge007', 'block_reference_edge007', 'train_down', BLOCK/'cell5_zero', 16),
               ('compare_block_local_side', '厚试件 · 右端一角侧推（考试）', 'block_reference_edge007', 'block_reference_edge007', 'fresh_local_side', BLOCK/'cell5_zero', 32)]
    for args in compare:
        print(args[0], round(compare_animation(*args), 2))
    result = tables()
    (OUT/'tables.json').write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(json.dumps(result['summary'], indent=1), json.dumps(result['capacity'], indent=1))
