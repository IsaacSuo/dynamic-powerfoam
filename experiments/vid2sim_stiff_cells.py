"""Which DEM bodies set the explicit time step? Per-body local frequency sqrt(diag(M^-1/2 K M^-1/2))
(Hutchinson estimate of the diagonal, K by forward-mode differentiation of the closed-form forces, GT
material), then a figure: the whole object with the stiffest 1% / 0.1% highlighted, and close-ups of
the stiffest bodies with their bonded neighbours (Power cells voxelised from the static foam), plus a
table comparing them with typical bodies (mass, volume, thinness, bonds, sum of A/L).
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy.spatial import cKDTree

from experiments.foam_dem_energy import quat_mul, quat_small
from experiments.pacnerf_dem_forward import build


def local_frequency(g, E, nu, probes=200):
    m = build(g, E, nu, 'cuda', 'noslip', .25, .3, torch.float64, kinematics='exact', residual=True)
    th = torch.zeros(m.n, 7, dtype=torch.float64, device='cuda')
    x0, q0, _, _ = m.initial_state()
    M = torch.cat((m.mass[:, None].expand(-1, 3), m.inertia[:, None].expand(-1, 3)), 1)

    def force(z):
        f, tau = m.forces(th, x0+z[:, :3], quat_mul(quat_small(z[:, 3:]), q0))
        return torch.cat((f, tau), 1)

    def A(y):
        _, Ky = torch.func.jvp(force, (torch.zeros_like(y),), (y/M.sqrt(),))
        return -Ky/M.sqrt()
    gen = torch.Generator(device='cuda').manual_seed(0)
    d = torch.zeros(m.n, 6, dtype=torch.float64, device='cuda')
    for _ in range(probes):
        v = torch.randint(0, 2, (m.n, 6), device='cuda', generator=gen).double()*2-1
        d += v*A(v)
    return (d/probes).clamp(min=0).sqrt().max(1).values.cpu().numpy()


def cell_samples(p, r, owner_of_site, region_lo, region_hi, h):
    """Voxel centres in a box, labelled by the Power cell (site index) that owns them; -1 = empty."""
    axes = [np.arange(region_lo[k]+h/2, region_hi[k], h) for k in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
    _, nn = cKDTree(p).query(grid, k=8)
    power = ((grid[:, None]-p[nn])**2).sum(-1)-r[nn]**2
    b = power.argmin(1)
    site = nn[np.arange(len(grid)), b]
    site[power[np.arange(len(grid)), b] > 0] = -1
    return grid, site


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--obj', default='sofa')
    a.add_argument('--close-ups', type=int, default=6)
    a.add_argument('--out', type=Path, default=Path('output/report_v2'))
    args = a.parse_args()
    dem = Path(f'output/vid2sim_dem/{args.obj}')
    g = dict(np.load(dem/'geometry.npz')); cells = dict(np.load(dem/'cells.npz'))
    gt = yaml.safe_load(open(f'data/vid2sim/ext/{args.obj}/gt_phys_params.yaml'))
    offset = np.array(json.loads(Path(f'data/vid2sim_powerfoam/{args.obj}_f0/conversion.json').read_text())['offset'])
    st = torch.load(f'output/vid2sim_{args.obj}_spacing8k/model.pt', map_location='cpu', weights_only=True)
    p = st['points'].double().numpy()-offset                                   # world frame
    r = torch.nn.functional.softplus(st['radii'].double(), beta=100).numpy()

    w = local_frequency(g, gt['yms'], gt['prs'])
    n = len(w)
    order = np.argsort(-w)
    top1, top01 = order[:max(1, n//100)], order[:max(1, n//1000)]
    sub = lambda q: int(np.ceil(.05*np.sort(w)[::-1][int(q*n)]))

    # per-body geometry statistics
    body_of_site = -np.ones(len(p), int); body_of_site[cells['source_index']] = g['body_of_cell']
    P, A_ = g['pairs'], g['areas']
    L = np.linalg.norm(g['points'][P[:, 0]]-g['points'][P[:, 1]], axis=1)
    AoL = np.bincount(P.ravel(), np.repeat(A_/L, 2), n)
    deg = np.bincount(P.ravel(), minlength=n)
    mass = g['masses']; vol = mass/1000.
    ncell = np.bincount(g['body_of_cell'], minlength=n)
    # thinness from a voxelisation of each studied body: smallest / largest principal std
    lo, hi = g['points'].min(0)-.03, g['points'].max(0)+.03
    grid, site = cell_samples(p, r, None, lo, hi, .004)
    body = np.where(site >= 0, body_of_site[np.maximum(site, 0)], -1)
    keep = body >= 0
    gb, gx = body[keep], grid[keep]
    thin = np.full(n, np.nan)
    srt = np.argsort(gb); gb, gx = gb[srt], gx[srt]
    starts = np.searchsorted(gb, np.arange(n)); ends = np.searchsorted(gb, np.arange(n), 'right')
    for i in range(n):
        if ends[i]-starts[i] >= 4:
            ev = np.linalg.eigvalsh(np.cov(gx[starts[i]:ends[i]].T))
            thin[i] = np.sqrt(max(ev[0], 0)/ev[-1])
    surface = np.zeros(n, bool)                       # body with a site neighbour that is not material
    rest = order[n//10:]
    rows = [('local frequency (rad/s)', w), ('mass / median', mass/np.median(mass)), ('sum A/L over bonds / median', AoL/np.median(AoL)),
            ('(sum A/L)/mass / median', (AoL/mass)/np.median(AoL/mass)), ('bonds', deg.astype(float)), ('cells merged', ncell.astype(float)),
            ('thinness (min/max extent, 1 = round)', thin)]
    table = []
    for name, v in rows:
        table.append((name, np.nanmedian(v[top01]), np.nanmedian(v[top1]), np.nanmedian(v[rest])))

    # ---------- figure
    fig = plt.figure(figsize=(16, 15))
    ax = fig.add_subplot(3, 3, (1, 2), projection='3d')
    X = g['points']
    ax.scatter(*X[rest].T, s=1, c='0.75', alpha=.3, depthshade=False)
    ax.scatter(*X[top1].T, s=14, c='tab:orange', label=f'stiffest 1% ({len(top1)} bodies)', depthshade=False)
    ax.scatter(*X[top01].T, s=50, c='red', edgecolors='k', label=f'stiffest 0.1% ({len(top01)} bodies)', depthshade=False)
    for k, i in enumerate(order[:args.close_ups]):
        ax.text(*X[i], f' {k+1}', color='k', fontsize=11, weight='bold')
    ax.set_box_aspect(np.ptp(X, 0)); ax.legend(loc='upper left'); ax.set_axis_off()
    ax.view_init(25, -60)
    ax.set_title(f'{args.obj}: {n} bodies. Substeps/frame set by the stiffest body: {sub(0)}; '
                 f'without the top 0.1%: {sub(.001)}, without the top 1%: {sub(.01)}', fontsize=11)
    axh = fig.add_subplot(3, 3, 3)
    axh.hist(w, bins=np.logspace(np.log10(w.min()+1), np.log10(w.max()), 60), color='0.5')
    axh.axvline(w[top1[-1]], color='tab:orange', ls='--', label='top 1%'); axh.axvline(w[top01[-1]], color='red', ls='--', label='top 0.1%')
    axh.set_xscale('log'); axh.set_yscale('log'); axh.set_xlabel('local frequency (rad/s)'); axh.set_ylabel('bodies'); axh.legend()
    axh.set_title('substeps/frame = 0.05 s x frequency')
    for k, i in enumerate(order[:args.close_ups]):
        axc = fig.add_subplot(3, 3, 4+k, projection='3d')
        nb = np.unique(np.concatenate((P[P[:, 0] == i, 1], P[P[:, 1] == i, 0])))
        c0 = X[i]; R = .045
        gg, ss = cell_samples(p, r, None, c0-R, c0+R, .0025)
        bb = np.where(ss >= 0, body_of_site[np.maximum(ss, 0)], -1)
        mine = bb == i
        near = np.isin(bb, nb)
        cmap = plt.get_cmap('tab20')
        for j_, j in enumerate(nb):
            s_ = bb == j
            if s_.any():
                ax_pts = gg[s_][::3]
                axc.scatter(*ax_pts.T, s=2, color=cmap(j_ % 20), alpha=.08, depthshade=False)
        axc.scatter(*gg[mine].T, s=6, c='red', alpha=.9, depthshade=False)
        axc.set_box_aspect((1, 1, 1)); axc.set_xlim(c0[0]-R, c0[0]+R); axc.set_ylim(c0[1]-R, c0[1]+R); axc.set_zlim(c0[2]-R, c0[2]+R)
        axc.set_xticks([]); axc.set_yticks([]); axc.set_zticks([])
        axc.set_title(f'#{k+1}: {w[i]:.0f} rad/s (x{w[i]/np.median(w):.0f} median)\nmass x{mass[i]/np.median(mass):.2f}, '
                      f'sum A/L x{AoL[i]/np.median(AoL):.1f}, thinness {thin[i]:.2f}, {deg[i]} bonds', fontsize=9)
    fig.text(.5, .005, 'close-ups: red = the stiff body (voxelised Power cell), coloured haze = its bonded neighbours; box 9 cm', ha='center')
    plt.tight_layout()
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out/f'vid2sim_{args.obj}_卡时间步的硬块.png'
    fig.savefig(out, dpi=110)
    print(out)
    print('%-38s %10s %10s %10s' % ('median of', 'top 0.1%', 'top 1%', 'other 90%'))
    for name, x1, x2, x3 in table:
        print('%-38s %10.3g %10.3g %10.3g' % (name, x1, x2, x3))
    print('substeps/frame: all %d, without top 0.1%% %d, without top 1%% %d, without top 5%% %d' % (sub(0), sub(.001), sub(.01), sub(.05)))


if __name__ == '__main__':
    main()
