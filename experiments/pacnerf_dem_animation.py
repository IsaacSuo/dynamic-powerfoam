"""GIF: PAC-NeRF elastic/0 ground truth vs DEM before / after training (same 3D view).

Top: GT MPM particles (grey, subsampled), then DEM body centres (marker area ~ body size) over
the GT, before training (GT E, nu converted) and after training. Bottom: centroid height of
the material with a moving time marker; frames 11-15 were held out of training.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from PIL import Image

from experiments.pacnerf_dem_forward import read_ply

font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False, 'font.size': 13})
RUN = Path('output/pacnerf_dem/elastic_0_f0')
OUT = Path('output/report_v2/pacnerf_elastic0_对比.gif')
GT, BEFORE, AFTER, INK, MUTED = '#8a8f98', '#5b6abf', '#e8743b', '#1f2430', '#6b7280'


def main():
    report = json.loads((RUN/'train_v2/report.json').read_text())
    before = np.load(RUN/'train_v2/positions_initial.npz')
    after = np.load(RUN/'train_v2/positions_best.npz')
    has = after['has']
    size = np.load(RUN/'geometry.npz')['masses']
    marker = 6+40*(size/np.median(size))**(2/3)
    rng = np.random.default_rng(0)
    first = read_ply('data/pacnerf/simulation_data/elastic/0/0.ply')
    pick = rng.choice(len(first), 12000, replace=False)
    frames = len(after['positions'])
    gt = [read_ply(f'data/pacnerf/simulation_data/elastic/0/{k}.ply')[pick] for k in range(frames)]
    hb, ha, hg = (np.array(report[k]['centroid_y']) for k in ('initial', 'best', 'best'))
    hg = np.array(report['best']['gt_centroid_y'])
    t = np.arange(frames)/24
    images = []
    for k in range(frames):
        fig = plt.figure(figsize=(15, 8.4), facecolor='white')
        panels = [('标准答案（MPM 材料点）', None, None),
                  ('训练前 DEM（真实 E、ν 直接换算）\n灰 = 标准答案', before['positions'][k], BEFORE),
                  ('训练后 DEM（学 7 个参数）\n灰 = 标准答案', after['positions'][k], AFTER)]
        for col, (title, pos, colour) in enumerate(panels):
            ax = fig.add_axes([.005+col*.33, .33, .33, .56], projection='3d')
            g = gt[k]
            ax.scatter(g[:, 0], g[:, 2], g[:, 1], s=1.5, c=GT, alpha=.35 if pos is not None else .6, linewidths=0)
            if pos is not None:
                p = pos[has]
                ax.scatter(p[:, 0], p[:, 2], p[:, 1], s=marker[has]*.25, c=colour, alpha=.55, linewidths=0)
            xx, zz = np.meshgrid([-.5, .35], [-.35, .35])
            ax.plot_surface(xx, zz, np.zeros_like(xx), color='#d9c6a5', alpha=.35, linewidth=0)
            ax.set_xlim(-.5, .35); ax.set_ylim(-.35, .35); ax.set_zlim(0, .75)
            ax.set_box_aspect((.85, .7, .75), zoom=1.3); ax.view_init(elev=12, azim=-62)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            ax.set_axis_off()
            fig.text(.17+col*.33, .87, title, ha='center', va='top', color=INK, fontsize=14)
        ax = fig.add_axes([.08, .07, .86, .24])
        ax.axvspan(10.5/24, t[-1]+.01, color='#f3f4f6')
        ax.text(12.9/24, .66, '训练时没见过（考题）', ha='center', color=MUTED, fontsize=12)
        ax.plot(t, hg, color=INK, lw=2.4, label='标准答案')
        ax.plot(t, hb, color=BEFORE, lw=2, label='训练前 DEM')
        ax.plot(t, ha, color=AFTER, lw=2, label='训练后 DEM')
        ax.axvline(t[k], color=MUTED, lw=1, ls='--')
        ax.set_xlim(0, t[-1]); ax.set_ylim(0, .75)
        ax.set_xlabel('时间（秒）', color=MUTED); ax.set_ylabel('质心高度（米）', color=MUTED)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        ax.legend(loc='upper left', frameon=False, ncol=3, fontsize=12)
        fig.text(.5, .955, f'PAC-NeRF elastic/0：软十字落地反弹    第 {k} 帧 / t = {t[k]:.3f} s', ha='center',
                 color=INK, fontsize=16)
        fig.canvas.draw()
        images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
        plt.close(fig)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    durations = [350]*(frames-1)+[1500]
    images[0].save(OUT, save_all=True, append_images=images[1:], duration=durations, loop=0, optimize=True)
    images[12].save(OUT.with_name('pacnerf_elastic0_第12帧.png'))
    print(OUT, len(images), 'frames')


if __name__ == '__main__':
    main()
