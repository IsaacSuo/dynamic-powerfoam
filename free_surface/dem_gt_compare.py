"""the new lion's DEM runs against the Vid2Sim ground truth: each saved frame rendered with the fourier3d Power Foam (as
render_fourier_drop.py: sites at x, radii r0 det F^(1/3), the Fourier wave vectors carried by F) in the training cameras
(lion_f0: the foam's frame, world = foam - offset; camera c = view V<c>, --check), over the real background, beside the
GT photo of the nearest GT frame (20 fps, frames 0-23). Columns: GT | each --frames run; rows: --views
usage: gt_compare.py --run <fourier3d run> --cells <cells.npz> --frames a.npz b.npz --labels A B --out <dir> [--views 0 3 6] [--check]"""
import argparse, sys
from pathlib import Path
import numpy as np, torch
FOUR = '/home/fangsuo/py/dynamic-powerfoam/.claude/worktrees/thin-surface/fourier3d'
sys.path.insert(0, FOUR)
import configargparse                                                              # noqa: E402
import warp as wp                                                                  # noqa: E402
from PIL import Image, ImageDraw                                                   # noqa: E402
from configs import Params, add_group                                              # noqa: E402
from data_loader import DataHandler                                                # noqa: E402
from powerfoam.scene import PowerfoamScene                                         # noqa: E402
from powerfoam import fourier                                                      # noqa: E402

NAMES = ('points', 'density', 'radii', 'sh_dc', 'sh_rest', 'fourier', 'fourier_phase_param', 'fourier_dk')
D = Path('/home/fangsuo/py/dynamic-powerfoam/data/springgaus_powerfoam/vid2sim_lion_dynamic')


def inv_softplus(y, beta=100.):
    return y+torch.log(-torch.expm1(-beta*y))/beta


a = argparse.ArgumentParser()
a.add_argument('--run', type=Path, required=True)
a.add_argument('--cells', type=Path, required=True)
a.add_argument('--frames', type=Path, nargs='+', required=True)
a.add_argument('--labels', nargs='+', required=True)
a.add_argument('--out', type=Path, required=True)
a.add_argument('--views', type=int, nargs='+', default=[0, 3, 6])
a.add_argument('--grid-times', type=float, nargs='+', default=[0., .3, .35, .4, .5, .7, 1.0, 1.15])
a.add_argument('--check', action='store_true')
args = a.parse_args()
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
cams = data.cameras
runs = []
for fpath in args.frames:
    z = np.load(fpath)
    runs.append(dict(t=z['t'], X=torch.as_tensor(z['x'], device='cuda'), F=torch.as_tensor(z['F'], device='cuda')))
    assert float((runs[-1]['X'][0].to(p0.dtype)-p0).abs().max()) < 1e-6, '%s: frame 0 is not the foam' % fpath


def render(cam):
    with torch.no_grad():
        q = .5*torch.ones(cam.height, cam.width, 1, device=model.device)
        out = model.forward(cam, depth_quantiles=q)
    return out[0].clamp(0, 1).reshape(cam.height, cam.width, 3), out[1].clamp(0, 1).reshape(cam.height, cam.width, 1)


def place(X, F):
    with torch.no_grad():
        Fk = F.to(p0.dtype)
        r = r0*torch.linalg.det(Fk).abs().clamp(min=1e-3)**(1/3)
        FiT = torch.linalg.inv(Fk).transpose(1, 2)
        n = (r/r0)[:, None, None]*(FiT[:, None]@n0.view(-1, M, 3)[..., None])[..., 0]
        model.points = torch.nn.Parameter(X.to(p0.dtype).contiguous())
        model.radii = torch.nn.Parameter(inv_softplus(r).contiguous())
        model.fourier_dk = torch.nn.Parameter(n.reshape(-1, M*3).contiguous())
        model.rebuild_adjacency()


load = lambda p: torch.as_tensor(np.array(Image.open(p).convert('RGB')), device='cuda').float()/255
if args.check:
    place(runs[0]['X'][0], runs[0]['F'][0])
    masks = [torch.as_tensor(np.array(Image.open(D/'masks'/('V%02d' % v)/'00.png').convert('L')) > 127, device='cuda') for v in range(12)]
    for c in range(len(cams)):
        al = render(cams[c])[1][..., 0] > .5
        iou = [float((al & m_).sum()/(al | m_).sum()) for m_ in masks]
        print('camera %d (%dx%d): best mask V%02d IoU %.4f; own index %.4f' % (c, cams[c].width, cams[c].height, int(np.argmax(iou)), max(iou), iou[c] if c < 12 else -1), flush=True)
    sys.exit()
args.out.mkdir(parents=True, exist_ok=True)
bgs = {v: load(D/'rest'/('background_V%02d.png' % v)) for v in args.views}
T = runs[0]['t']
for r_ in runs[1:]:
    assert np.allclose(r_['t'], T), 'the runs\' frame times differ'
pics, full = [], {}
for k, t in enumerate(T):
    f = min(int(round(t*20)), 23)
    rows = []
    for v in args.views:
        row = [load(D/'images'/('V%02d' % v)/('%02d.png' % f))]
        for r_ in runs:
            place(r_['X'][k], r_['F'][k])
            rgb, al = render(cams[v])
            row.append((rgb+(1-al)*bgs[v]).clamp(0, 1))
        rows.append(torch.cat(row, 1))
    img = (torch.cat(rows, 0).cpu().numpy()*255).astype(np.uint8)
    im = Image.fromarray(img); d = ImageDraw.Draw(im)
    W = cams[args.views[0]].width
    d.text((6, 4), 'GT frame %d (t %.2f s)' % (f, f/20), fill=(255, 255, 255))
    for j, lab in enumerate(args.labels):
        d.text((W*(j+1)+6, 4), '%s, t %.3f s' % (lab, t), fill=(255, 255, 255))
    im.save(args.out/('%03d.png' % k)); pics.append(im.resize((im.width//2, im.height//2)))
    full[k] = im
pics[0].save(args.out/'compare.gif', save_all=True, append_images=pics[1:], duration=50, loop=0)
ks = [int(np.argmin(np.abs(T-tt))) for tt in args.grid_times]                      # (a still: the first view's row at a few times)
H = cams[args.views[0]].height
strip = [np.array(full[k])[:H] for k in ks]
Image.fromarray(np.concatenate(strip, 0)).save(args.out/'compare_grid.png')
print('wrote %d frames, %s and %s' % (len(pics), args.out/'compare.gif', args.out/'compare_grid.png'), flush=True)
