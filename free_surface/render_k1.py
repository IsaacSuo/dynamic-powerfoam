"""Render a k1 landing with its Power Foam (11-06, the user: "我们每步解的半径就能更新"): the foam is the fourier3d run the k1 foam was cut
from (fourier_pacnerf_elastic0_8k_free_clean: cells = balls x power cells, per-cell SH colour + Fourier texture in the cell's own
coordinates u = (x - c) / r); the kept material cells are cells.npz's source_index. Per frame of the run (frames_<tag>.npz):
the cell's centre goes to its site x_k, its radius is the radius the run solved for that step (W._ulr, the cut's: every cell's
volume V0 e^L), the texture's wave vectors turn with the polar rotation R of F_k (the SH colour is left unturned), the adjacency
is rebuilt by the renderer's own Cech complex. Writes the frames as PNGs per view and a time grid.
  python free_surface/render_k1.py --frames output/.../frames_patchrun3.npz --views 0 3"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

F3 = '/home/fangsuo/py/dynamic-powerfoam/.claude/worktrees/thin-surface/fourier3d'
RUN = F3+'/output/fourier_pacnerf_elastic0_8k_free_clean'
sys.path.insert(0, F3)
import configargparse                                                              # noqa: E402
import warp as wp                                                                  # noqa: E402
from PIL import Image, ImageDraw                                                   # noqa: E402

from configs import Params, add_group                                              # noqa: E402
from data_loader import DataHandler                                                # noqa: E402
from powerfoam.scene import PowerfoamScene                                         # noqa: E402

PER_CELL = ('points', 'density', 'radii', 'sh_dc', 'sh_rest', 'fourier', 'fourier_phase_param', 'fourier_dk')


def load(run):
    parser = configargparse.ArgParser()
    get_params = add_group(parser, Params)
    parser.add_argument('-c', '--config', is_config_file=True)
    args = get_params(parser.parse_args(['-c', str(Path(run)/'config.yaml')]))
    wp.init()
    data = DataHandler(args)
    data.reload('all', downsample=args.downsample[-1])
    model = PowerfoamScene(args)
    model.initialize_from_dataset(data, device='cuda')
    model.load_pt(str(Path(run)/'model.pt'))
    return model, data


def inv_softplus(y, beta=100.):
    return y+torch.log(-torch.expm1(-beta*y))/beta


def polar(F):
    U, S, Vh = torch.linalg.svd(F)
    R = U@Vh
    neg = torch.det(R) < 0
    if neg.any():
        U2 = U.clone(); U2[neg, :, 2] *= -1; R = U2@Vh
    return R


def render(model, cam):
    with torch.no_grad():
        q = .5*torch.ones(cam.height, cam.width, 1, device=model.device)
        rgb, alpha = model.forward(cam, depth_quantiles=q)[:2]
    return rgb.clamp(0, 1), alpha.clamp(0, 1)


def on_white(rgb, alpha):
    alpha = alpha.reshape(*rgb.shape[:2], 1)
    return (rgb+(1-alpha)).clamp(0, 1).cpu().numpy()


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--frames', type=Path, required=True)
    a.add_argument('--run', default=RUN, help='the fourier3d run (config.yaml, model.pt) the foam was cut from')
    a.add_argument('--cells', type=Path, default=Path('foams/fourier8k_clean/dem_exact_single/cells.npz'))
    a.add_argument('--views', type=int, nargs='+', default=[0, 3])
    a.add_argument('--times', type=float, nargs='+', default=[0., .1, .2, .225, .25, .275, .3, .35, .4, .5, .625])
    a.add_argument('--every', type=int, default=1, help='every k-th frame as a PNG (0: none)')
    a.add_argument('--out', type=Path, default=None)
    a.add_argument('--bg', type=Path, default=None, help='the PAC-NeRF scene\'s image folder (data/pacnerf/data/<scene>/data): each view composited over its empty frame r_<view>_-1.png')
    a.add_argument('--rotate', type=int, default=1, help='turn the texture\'s wave vectors with polar(F) (0: not)')
    args = a.parse_args()
    out = args.out or args.frames.parent/('render_'+args.frames.stem)
    out.mkdir(parents=True, exist_ok=True)
    model, data = load(args.run)
    print('cameras %d, image %dx%d; model cells %d' % (len(data.cameras), data.cameras[0].width, data.cameras[0].height, len(model.points)), flush=True)
    c = np.load(args.cells)
    src = torch.as_tensor(c['source_index'], device=model.device)
    with torch.no_grad():
        for name in PER_CELL:
            p_ = getattr(model, name, None)
            if p_ is not None:
                setattr(model, name, torch.nn.Parameter(p_.data[src].clone()))
        model.rebuild_adjacency()
    p0 = model.points.data.clone(); r0 = model.get_radii().detach().clone(); dk0 = model.fourier_dk.data.clone() if model.fourier is not None else None
    cp = torch.as_tensor(c['points'], device=model.device, dtype=p0.dtype); cr = torch.as_tensor(c['radii'], device=model.device, dtype=p0.dtype)
    print('kept cells %d; the foam\'s points vs the model\'s max |diff| %.1e; the foam\'s radii vs the model\'s max |diff| %.1e' % (len(src), float((cp-p0).abs().max()), float((cr-r0).abs().max())), flush=True)
    z = np.load(args.frames)
    t = z['t']; X = torch.as_tensor(z['x'], device='cuda', dtype=p0.dtype); Rr = torch.as_tensor(z['r'], device='cuda', dtype=p0.dtype)
    Fz = torch.as_tensor(z['F'], device='cuda', dtype=torch.float32) if 'F' in z.files else None
    shift = (X[0]-p0).mean(0)
    print('frames %d (t %.3f..%.3f s); the run\'s start vs the foam: shift %s, spread %.1e; radii at the start vs the foam\'s max |diff| %.1e; F %s' % (
        len(t), t[0], t[-1], shift.cpu().numpy().round(5).tolist(), float((X[0]-p0-shift).abs().max()), float((Rr[0]-r0).abs().max()), 'saved' if Fz is not None else 'not saved: no texture rotation'), flush=True)
    cams = [data.cameras[v] for v in args.views]
    import json as _json
    meta = _json.loads((Path(model.args.data_path)/model.args.scene/'transforms_all.json').read_text()) if hasattr(model, 'args') else None
    names = [Path(meta['frames'][v]['file_path']).name.rsplit('_', 1)[0] for v in args.views] if meta else ['view%d' % v for v in args.views]
    (out/'views.json').write_text(_json.dumps(dict(views=args.views, names=names)))
    bgs = None
    if args.bg is not None:
        bgs = []
        for nm, cm in zip(names, cams):
            im = Image.open(args.bg/('%s_-1.png' % nm)).convert('RGB').resize((cm.width, cm.height), Image.LANCZOS)
            bgs.append(torch.as_tensor(np.asarray(im, dtype=np.float32)/255., device=model.device))
        print('backgrounds: %s from %s' % (names, args.bg), flush=True)
    comp = (lambda rgb, alpha, j: on_white(rgb, alpha) if bgs is None else (rgb+(1-alpha.reshape(*rgb.shape[:2], 1))*bgs[j]).clamp(0, 1).cpu().numpy())
    rest = [render(model, cm) for cm in cams]

    def apply(k):
        with torch.no_grad():
            model.points = torch.nn.Parameter((X[k]-shift).contiguous())
            model.radii = torch.nn.Parameter(inv_softplus(Rr[k]).contiguous())
            if dk0 is not None and Fz is not None and args.rotate:
                R = polar(Fz[k]).to(dk0.dtype)
                m = dk0.shape[1]//3
                dk = (R[:, None]@dk0.view(len(R), m, 3)[..., None])[..., 0].reshape(dk0.shape)
                model.fourier_dk = torch.nn.Parameter(dk.contiguous())
            model.rebuild_adjacency()
    apply(0)
    again = [render(model, cm) for cm in cams]
    print('rest check (frame 0 through the update against the untouched foam): max |rgb| %.1e, max |alpha| %.1e' % (
        max(float((r1[0]-r2[0]).abs().max()) for r1, r2 in zip(rest, again)), max(float((r1[1]-r2[1]).abs().max()) for r1, r2 in zip(rest, again))), flush=True)
    H, Wd = cams[0].height, cams[0].width
    ks = [int(np.argmin(np.abs(t-tt))) for tt in args.times]
    cols = {}
    for k in range(len(t)):
        if k not in ks and not (args.every and k % args.every == 0):
            continue
        apply(k)
        imgs = [comp(*render(model, cm), j) for j, cm in enumerate(cams)]
        if k in ks:
            cols[k] = imgs
        if args.every and k % args.every == 0:
            for v, im in zip(args.views, imgs):
                Image.fromarray((im*255).astype(np.uint8)).save(out/('view%d_%03d.jpg' % (v, k)), quality=88)
    grid = np.ones((len(cams)*H+24, (len(cols)+1)*Wd, 3))
    for v in range(len(cams)):
        grid[24+v*H:24+(v+1)*H, :Wd] = comp(*rest[v], v)
        for j, k in enumerate(sorted(cols)):
            grid[24+v*H:24+(v+1)*H, (j+1)*Wd:(j+2)*Wd] = cols[k][v]
    im = Image.fromarray((grid*255).astype(np.uint8)); d = ImageDraw.Draw(im)
    d.text((4, 4), 'rest foam', fill=(0, 0, 0))
    for j, k in enumerate(sorted(cols)):
        d.text(((j+1)*Wd+4, 4), 't %.3f s' % t[k], fill=(0, 0, 0))
    im.save(out/'grid.png')
    print('wrote %s (%d views x %d times, %dx%d) and the frames view*_*.jpg' % (out/'grid.png', len(cams), len(cols), Wd, H), flush=True)


if __name__ == '__main__':
    main()
