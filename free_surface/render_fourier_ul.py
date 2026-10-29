"""Render a UL drop (ul_drop.py's <tag>_frames: sites x, F and the model's own radii r per record) with the fourier3d
Power Foam it was cut from (10-29; the placement of the DEM session's gt_compare.py with the model's radii instead of
r0 det F^(1/3)): sites at x, radii r, the Fourier wave vectors carried by F as the material's planes are,
dk = (r / r0) F^-T dk0 (dk in units of the radius), the adjacency rebuilt. The cells are the material cells of the cut
(cells.npz source_index). Writes a grid (rows: --views; columns: the rest foam, then --times) on white, and with --gif
every frame per view as a GIF; first the rest check (frame 0 placed against the untouched foam). --gt: also against the
Vid2Sim ground truth, a picture per GT frame (20 fps, frames 0-23; camera c = view V<c> of lion_f0) over the real
background: rows --gt-views, columns the GT photo, this run (its nearest frame), and with --dem a DEM run's frames npz
(t, x, F; radii r0 det F^(1/3), as the DEM session's gt_compare.py), their GIF and a strip of the first view at --times.
  python free_surface/render_fourier_ul.py --run <thin-surface>/fourier3d/output/fourier_vid2sim_lion_8k_round_clean \
      --cells <updated-lagrange>/output/fourier_lion_8k_round_clean/dem_exact_single/cells.npz \
      --frames output/fourier_lion_8k_round_clean/ul_lion_b0.05_js_frames --gif 1"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

FOUR = '/home/fangsuo/py/dynamic-powerfoam/.claude/worktrees/thin-surface/fourier3d'
sys.path.insert(0, FOUR)
import configargparse                                                              # noqa: E402
import warp as wp                                                                  # noqa: E402
from PIL import Image, ImageDraw                                                   # noqa: E402

from configs import Params, add_group                                              # noqa: E402
from data_loader import DataHandler                                                # noqa: E402
from powerfoam import fourier                                                      # noqa: E402
from powerfoam.scene import PowerfoamScene                                         # noqa: E402

NAMES = ('points', 'density', 'radii', 'sh_dc', 'sh_rest', 'fourier', 'fourier_phase_param', 'fourier_dk')


def inv_softplus(y, beta=100.):
    return y+torch.log(-torch.expm1(-beta*y))/beta


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--run', type=Path, required=True)
    a.add_argument('--cells', type=Path, required=True)
    a.add_argument('--frames', type=Path, required=True)
    a.add_argument('--out', type=Path, default=None)
    a.add_argument('--views', type=int, nargs='+', default=[0, 3])
    a.add_argument('--times', type=float, nargs='+', default=[0., .28, .3, .33, .35, .37, .4, .45, .5, .7, 1., 1.2])
    a.add_argument('--gif', type=int, default=0)
    a.add_argument('--gt', action='store_true', help='only the comparison with the Vid2Sim ground truth (see above)')
    a.add_argument('--gt-views', type=int, nargs='+', default=[0, 3, 6])
    a.add_argument('--dem', type=Path, default=None, help='a DEM run\'s frames npz (t, x, F) as a third column')
    args = a.parse_args()
    out = args.out or args.frames.parent/('render_'+args.frames.name)
    out.mkdir(parents=True, exist_ok=True)
    parser = configargparse.ArgParser()
    get_params = add_group(parser, Params)
    parser.add_argument('-c', '--config', is_config_file=True)
    pa = get_params(parser.parse_args(['-c', str(args.run/'config.yaml')]))
    wp.init()
    data = DataHandler(pa)
    data.reload('all', downsample=pa.downsample[-1])
    model = PowerfoamScene(pa)
    model.initialize_from_dataset(data, device='cuda')
    model.load_pt(str(args.run/'model.pt'))
    src = torch.as_tensor(np.load(args.cells)['source_index'], device=model.device).long()
    with torch.no_grad():
        for name in NAMES:
            if getattr(model, name, None) is not None:
                setattr(model, name, torch.nn.Parameter(getattr(model, name).data[src].clone()))
        model.rebuild_adjacency()
    p0, r0, n0 = model.points.data.clone(), model.get_radii().detach().clone(), model.fourier_dk.data.clone()
    M = fourier.FOURIER_MODE_COUNT
    cams = [data.cameras[v] for v in args.views]

    def render(cam):
        with torch.no_grad():
            q = .5*torch.ones(cam.height, cam.width, 1, device=model.device)
            o = model.forward(cam, depth_quantiles=q)
        rgb = o[0].clamp(0, 1).reshape(cam.height, cam.width, 3); al = o[1].clamp(0, 1).reshape(cam.height, cam.width, 1)
        return (rgb+(1-al)).clamp(0, 1).cpu().numpy()

    def place(X, F, r):
        with torch.no_grad():
            Fk = F.to(p0.dtype); r = r.to(r0.dtype)
            FiT = torch.linalg.inv(Fk).transpose(1, 2)
            n = (r/r0)[:, None, None]*(FiT[:, None]@n0.view(-1, M, 3)[..., None])[..., 0]
            model.points = torch.nn.Parameter(X.to(p0.dtype).contiguous())
            model.radii = torch.nn.Parameter(inv_softplus(r).contiguous())
            model.fourier_dk = torch.nn.Parameter(n.reshape(-1, M*3).contiguous())
            model.rebuild_adjacency()

    fz = [np.load(p_) for p_ in sorted(args.frames.glob('f*.npz'))]
    t = np.array([float(z_['t']) for z_ in fz])
    X = torch.as_tensor(np.stack([z_['x'] for z_ in fz]), dtype=torch.float64, device='cuda')
    F = torch.as_tensor(np.stack([z_['F'] for z_ in fz]), dtype=torch.float64, device='cuda')
    Rm = torch.as_tensor(np.stack([z_['r'] for z_ in fz]), dtype=torch.float64, device='cuda')
    print('%d cells, %d frames, t %.4f..%.4f s; frame 0 against the foam: max |x - p0| %.1e, max |r / r0 - 1| %.1e' % (
        len(src), len(t), t[0], t[-1], float((X[0]-p0.double()).abs().max()), float((Rm[0]/r0.double()-1).abs().max())), flush=True)
    if args.gt:
        compare_gt(args, out, data, model, place, t, X, F, Rm, r0)
        return
    rest = [render(c) for c in cams]
    place(X[0], F[0], Rm[0])
    again = [render(c) for c in cams]
    print('rest check (frame 0 placed against the untouched foam): max |rgb| %.1e' % max(float(np.abs(a_-b_).max()) for a_, b_ in zip(rest, again)), flush=True)
    H, W = cams[0].height, cams[0].width
    ks = [int(np.argmin(np.abs(t-tt))) for tt in args.times if tt <= t[-1]+1e-9]
    cols = []
    for k in ks:
        place(X[k], F[k], Rm[k]); cols.append((t[k], [render(c) for c in cams]))
    grid = np.ones((len(cams)*H+24, (len(cols)+1)*W, 3))
    for v in range(len(cams)):
        grid[24+v*H:24+(v+1)*H, :W] = rest[v]
        for j, (_, imgs) in enumerate(cols):
            grid[24+v*H:24+(v+1)*H, (j+1)*W:(j+2)*W] = imgs[v]
    im = Image.fromarray((grid*255).astype(np.uint8)); d = ImageDraw.Draw(im)
    d.text((4, 4), 'rest foam', fill=(0, 0, 0))
    for j, (tk, _) in enumerate(cols):
        d.text(((j+1)*W+4, 4), 't %.3f s (model radii)' % tk, fill=(0, 0, 0))
    im.save(out/'grid.png')
    print('wrote %s (%d views x %d times)' % (out/'grid.png', len(cams), len(cols)), flush=True)
    if args.gif:
        seq = [[] for _ in cams]
        for k in range(len(t)):
            place(X[k], F[k], Rm[k])
            for v, c in enumerate(cams):
                im = Image.fromarray((render(c)*255).astype(np.uint8))
                ImageDraw.Draw(im).text((4, 4), 't %.4f s' % t[k], fill=(0, 0, 0))
                seq[v].append(im)
        for v, vi in enumerate(args.views):
            g_ = out/('view%d.gif' % vi)
            seq[v][0].save(g_, save_all=True, append_images=seq[v][1:], duration=1000//24, loop=0)
            print('wrote %s (%d frames)' % (g_, len(seq[v])), flush=True)


def compare_gt(args, out, data, model, place, t, X, F, Rm, r0):
    """GT photo | this run | (DEM) per GT frame over the real background, no metric (10-29: the user judges by eye)"""
    D = Path('/home/fangsuo/py/dynamic-powerfoam/data/springgaus_powerfoam/vid2sim_lion_dynamic')
    cams = {v: data.cameras[v] for v in args.gt_views}
    load = lambda p: torch.as_tensor(np.array(Image.open(p).convert('RGB')), device='cuda').float()/255
    bgs = {v: load(D/'rest'/('background_V%02d.png' % v)) for v in args.gt_views}
    runs = [('UL', t, lambda k: place(X[k], F[k], Rm[k]))]
    if args.dem is not None:
        z = np.load(args.dem)
        Xd, Fd = torch.as_tensor(z['x'], device='cuda'), torch.as_tensor(z['F'], device='cuda')
        runs.append(('DEM', z['t'], lambda k: place(Xd[k], Fd[k], r0*torch.linalg.det(Fd[k].to(r0.dtype)).clamp(min=1e-3)**(1/3))))
    sub = out/'gt_compare'; sub.mkdir(exist_ok=True)
    W, H = cams[args.gt_views[0]].width, cams[args.gt_views[0]].height
    pics, full = [], []
    for f in range(24):
        tf = f/20
        rows = {v: [load(D/'images'/('V%02d' % v)/('%02d.png' % f))] for v in args.gt_views}
        lab = []
        for name, tr, put in runs:
            k = int(np.argmin(np.abs(tr-tf)))
            put(k); lab.append('%s t %.3f s' % (name, tr[k]))
            for v, c in cams.items():
                with torch.no_grad():
                    q = .5*torch.ones(c.height, c.width, 1, device=model.device)
                    o = model.forward(c, depth_quantiles=q)
                rgb = o[0].clamp(0, 1).reshape(c.height, c.width, 3); al = o[1].clamp(0, 1).reshape(c.height, c.width, 1)
                rows[v].append((rgb+(1-al)*bgs[v]).clamp(0, 1))
        im = Image.fromarray((torch.cat([torch.cat(rows[v], 1) for v in args.gt_views], 0).cpu().numpy()*255).astype(np.uint8))
        d = ImageDraw.Draw(im)
        d.text((6, 4), 'GT frame %d (t %.2f s)' % (f, tf), fill=(255, 255, 255))
        for j, l_ in enumerate(lab):
            d.text((W*(j+1)+6, 4), l_, fill=(255, 255, 255))
        im.save(sub/('%02d.png' % f)); full.append(im); pics.append(im.resize((im.width//2, im.height//2)))
    pics[0].save(sub/'compare.gif', save_all=True, append_images=pics[1:], duration=150, loop=0)
    ks = sorted({min(int(round(tt*20)), 23) for tt in args.times})
    Image.fromarray(np.concatenate([np.array(full[k])[:H] for k in ks], 0)).save(sub/'compare_grid.png')
    print('wrote %s (24 GT frames: GT | %s), compare.gif, compare_grid.png (first view at GT frames %s)' % (
        sub, ' | '.join(n_ for n_, _, _ in runs), ks), flush=True)


if __name__ == '__main__':
    main()
