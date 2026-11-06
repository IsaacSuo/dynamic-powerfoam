"""render a Fourier PAC-NeRF forward (scratch/fourier_forward.py's sim_K<K>.npz) with its own Power Foam, against MPM.

The foam is the training run's (thin-surface fourier3d/output/fourier_pacnerf_<s>_8k_split_keep_clean: config.yaml,
model.pt), restricted to the simulated cells (cells.npz source_index). Per frame every cell takes our simulation's state:
  site      x_k (the foam's frame = the simulation's frame),
  radius    r0 det(F_B)^(1/3) (PowerSim's rule),
  texture   carried by the material: c(x') = c(mu0 + F^-1 (x' - mu')), so k' = F^-T k with k = pi n / R, i.e.
            n' = (R' / R0) F^-T n, phases unchanged (exact for the affine map of the cell),
  SH        the view-dependent base turned with the cell's polar rotation R: SH(R^T d),
  density   unchanged; the adjacency rebuilt (the renderer's Cech complex).
Rest check: frame 0 (x = X, F = I) against the untouched restricted foam.
MPM: PAC-NeRF's own frames (data/<scene>/data/r_<view>_<k>.png, renders of their MPM) shown beside ours; the MPM
particles (simulation_data/elastic/<k>/<frame>.ply, world + offset) drawn as a silhouette (spheres of radius 0.6 x the
particle spacing) whose outline goes on our render (red); per frame and view IoU of our alpha > 0.5 against it.
Background for ours: PAC-NeRF's frame -1 (the empty scene).
  python -u scratch/fourier_render.py <scene> [--views 0 3 6 10] [--K 14]
writes out_inverse/fourier_forward/<scene>/render/: frame_<k>.png (rows: views; columns: PAC-NeRF / MPM, ours, ours +
MPM outline), sheet_view<v>.png (rows: frames), compare.gif, iou.json"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
FOURIER = Path('/home/fangsuo/py/dynamic-powerfoam/.claude/worktrees/thin-surface/fourier3d')
sys.path.insert(0, str(ROOT))
import free_surface.lion_inverse as li                                             # noqa: E402

a = argparse.ArgumentParser()
a.add_argument('scene')
a.add_argument('--views', type=int, nargs='+', default=[0, 3, 6, 10])
a.add_argument('--K', type=int, default=None)
a.add_argument('--sim', type=Path, default=None)
args = a.parse_args()
li.set_scene(args.scene)
s = args.scene[len('fourier_'):]
RUN = FOURIER/'output'/('fourier_pacnerf_%s_8k_split_keep_clean' % s)
K = args.K if args.K is not None else li.NFRAMES-1
SIM = args.sim or ROOT/'out_inverse'/'fourier_forward'/args.scene/('sim_K%d.npz' % K)
OUT = SIM.parent/'render'; OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(FOURIER))                                                    # the run's configs / data_loader / powerfoam
import configargparse                                                              # noqa: E402
import warp as wp                                                                  # noqa: E402
from PIL import Image, ImageDraw                                                   # noqa: E402
from configs import Params, add_group                                              # noqa: E402
from configs.fourier_config import flatten_fourier_config_argv                     # noqa: E402
from data_loader import DataHandler                                                # noqa: E402
from powerfoam.scene import PowerfoamScene                                         # noqa: E402
from powerfoam.sh_color import eval_sh                                             # noqa: E402
from powerfoam import fourier as fourier_mod                                       # noqa: E402


def load_model():
    argv = ['render', '-c', str(RUN/'config.yaml')]
    orig = flatten_fourier_config_argv(argv)
    parser = configargparse.ArgParser()
    get_params = add_group(parser, Params)
    parser.add_argument('-c', '--config', is_config_file=True)
    pa = get_params(parser.parse_args(argv[1:]))
    wp.init()
    data = DataHandler(pa)
    data.reload('all', downsample=pa.downsample[-1])
    model = PowerfoamScene(pa)
    model.initialize_from_dataset(data, device='cuda')
    model.load_pt(str(RUN/'model.pt'))
    return model, data


model, data = load_model()
cells = np.load(li.FOAM/'dem_exact_single'/'cells.npz')
src = torch.as_tensor(cells['source_index'], device='cuda').long()
with torch.no_grad():
    for name in ('points', 'density', 'radii', 'sh_dc', 'sh_rest', 'fourier', 'fourier_phase_param', 'fourier_dk'):
        setattr(model, name, torch.nn.Parameter(getattr(model, name).data[src].clone()))
    model.rebuild_adjacency()
P0 = model.points.data.clone(); R0 = model.get_radii().detach().clone(); N0 = model.fourier_dk.data.clone()
M = fourier_mod.FOURIER_MODE_COUNT
assert np.allclose(cells['points'], P0.double().cpu().numpy(), atol=1e-6), 'cells.npz points differ from the run\'s'
assert np.allclose(cells['radii'], R0.double().cpu().numpy(), rtol=1e-5), 'cells.npz radii differ from the run\'s'
# fourier3d's cameras are in transforms_all.json's file order (r_5_0, r_10_0, ...); li's views are sorted by camera number
_tr = json.loads((li.DATA/'transforms_all.json').read_text())['frames']
cam_of = {Path(f_['file_path']).name.split('_')[1]: i for i, f_ in enumerate(_tr)}
assert len(cam_of) == len(data.cameras), (len(cam_of), len(data.cameras))
meta, _, _ = li.cameras(); vnames = meta['camera_names']


def camera(v):
    """the fourier3d camera of li's view v (by name r_<cam>_0)"""
    return data.cameras[cam_of[vnames[v]]]


def render(cam, x=None, F=None):
    """our render (premultiplied rgb, alpha) of the restricted foam moved to x with F (None: the foam as loaded)"""
    with torch.no_grad():
        if x is None:
            pts, rad, dk, base = model.points, model.get_radii(), model.fourier_dk, model.get_base_rgb(cam)
        else:
            Fd = F.double()
            U, S, Vh = torch.linalg.svd(Fd); Rp = U@Vh
            J = torch.linalg.det(Fd)
            rad = (R0.double()*J.clamp(min=1e-3)**(1/3)).to(R0.dtype)
            Fit = torch.linalg.inv(Fd).transpose(1, 2)
            n0 = N0.double().view(-1, M, 3)
            dk = ((rad.double()/R0.double())[:, None, None]*(Fit[:, None]@n0[..., None])[..., 0]).reshape(N0.shape).to(N0.dtype)
            pts = x.to(P0.dtype).contiguous()
            d = pts.detach()-cam.eye.to(pts.device)[None, :]
            d = d/d.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            d = (Rp.transpose(1, 2)@d.double()[..., None])[..., 0]                   # (SH(R^T d))
            base = (eval_sh(int(model.active_sh_degree), model.sh_dc.float(), model.sh_rest.float(), d.float())+.5).to(model.tscalar)
            model.points = torch.nn.Parameter(pts); model.radii = torch.nn.Parameter((rad+torch.log(-torch.expm1(-100*rad))/100).contiguous())
            model.fourier_dk = torch.nn.Parameter(dk.contiguous())
            model.rebuild_adjacency()
        q = .5*torch.ones(cam.height, cam.width, 1, device='cuda')
        fa = model.get_fourier_arg() if x is None else torch.cat([model.get_fourier_arg()[:, :M*6], dk], 1)
        out = model.rasterizer.forward(cam, q, pts, rad if x is not None else model.get_radii(), model.get_density(), base, fa,
                                       model.adjacency, model.adjacency_offsets, None, False)
        rgb, alpha = out[0], out[1]
    return rgb.clamp(0, 1).float(), alpha.reshape(*rgb.shape[:2]).clamp(0, 1).float()


def restore():
    model.points = torch.nn.Parameter(P0.clone()); model.radii = torch.nn.Parameter((R0+torch.log(-torch.expm1(-100*R0))/100).contiguous())
    model.fourier_dk = torch.nn.Parameter(N0.clone()); model.rebuild_adjacency()


z = np.load(SIM)
if 'failed' in z.files:
    raise SystemExit('the run failed: %s' % z['failed'])
X = torch.as_tensor(z['x'], device='cuda'); Fs = torch.as_tensor(z['F'], device='cuda')
K = min(K, len(X)-1)
assert np.allclose(z['x'][0], cells['points'], atol=1e-6)
cams = [camera(v) for v in args.views]
H, Wd = cams[0].height, cams[0].width
# rest check
ref = [render(c) for c in cams]
eye = torch.eye(3, device='cuda').expand(len(src), 3, 3)
again = [render(c, X[0], eye) for c in cams]
print('rest check (frame 0 through the update against the untouched foam): max |rgb| %.1e, max |alpha| %.1e' % (
    max(float((p[0]-q_[0]).abs().max()) for p, q_ in zip(ref, again)), max(float((p[1]-q_[1]).abs().max()) for p, q_ in zip(ref, again))), flush=True)
restore()

# PAC-NeRF frames, the background, the MPM silhouettes
photo = lambda v, k: np.asarray(Image.open(Path(li.SCENE['photos'])/('r_%s_%d.png' % (vnames[v], k))).convert('RGB')).astype(np.float32)/255.
mask0 = {v: np.asarray(Image.open(li.DATA/'masks'/('r_%s_0.png' % vnames[v])), np.float32)/255. for v in args.views}
bg = {v: photo(v, -1) for v in args.views}                                     # (PAC-NeRF's frame -1: the empty scene)
mpm_dir = Path(li.SCENE['mpm']) if 'mpm' in li.SCENE else None
sil = li.Silhouette()
vidx = torch.as_tensor(args.views, device='cuda')


def mpm_mask(k):
    from plyfile import PlyData
    v = PlyData.read(str(mpm_dir/('%d.ply' % k)))['vertex']
    p = np.stack([v['x'], v['y'], v['z']], 1).astype(np.float64)+np.array([.5, -.05, .5])
    if not hasattr(mpm_mask, 'rad'):
        from scipy.spatial import cKDTree
        mpm_mask.rad = .6*float(np.median(cKDTree(p).query(p, k=2)[0][:, 1]))
    x = torch.as_tensor(p, device='cuda')
    return (sil.render(x, torch.full((len(p),), mpm_mask.rad, dtype=torch.float64, device='cuda'))[vidx] > .5).cpu().numpy()


def outline(m):
    inner = m.copy(); inner[1:-1, 1:-1] = m[1:-1, 1:-1] & m[:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, :-2] & m[1:-1, 2:]
    e = m & ~inner
    return e | np.roll(e, 1, 0) | np.roll(e, 1, 1)


iou = []; frames = []
for k in range(K+1):
    mm = mpm_mask(k) if mpm_dir is not None else None
    row_v = []; iou_k = []
    for j, (v, c) in enumerate(zip(args.views, cams)):
        rgb, al = render(c, X[k], Fs[k])
        ours = (rgb.cpu().numpy()+(1-al.cpu().numpy())[..., None]*bg[v]).clip(0, 1)
        ours_m = ours.copy()
        if mm is not None:
            a_ = al.cpu().numpy() > .5; b_ = mm[j]
            iou_k.append(float((a_ & b_).sum()/max((a_ | b_).sum(), 1)))
            ours_m[outline(b_)] = (1., 0., 0.)
        row_v.append(np.concatenate([photo(v, k), ours, ours_m], 1))
    iou.append(iou_k)
    fr = np.concatenate(row_v, 0)
    im = Image.fromarray((fr*255).astype(np.uint8)); d = ImageDraw.Draw(im)
    d.text((4, 4), 'PAC-NeRF (MPM) frame %d' % k, fill=(0, 0, 0)); d.text((Wd+4, 4), 'ours (DEM)', fill=(0, 0, 0))
    d.text((2*Wd+4, 4), 'ours + MPM particles\' outline (red)%s' % (('  IoU %s' % ' '.join('%.2f' % u for u in iou_k)) if iou_k else ''), fill=(0, 0, 0))
    im.save(OUT/('frame_%02d.png' % k)); frames.append(im)
    restore()
    print('frame %d: IoU against the MPM particles per view %s' % (k, ' '.join('%.3f' % u for u in iou_k)), flush=True)
for j, v in enumerate(args.views):
    rows = [np.asarray(f_)[j*H:(j+1)*H] for f_ in frames]
    Image.fromarray(np.concatenate(rows, 0)).resize((3*Wd//2, len(rows)*H//2)).save(OUT/('sheet_view%s.png' % vnames[v]))
frames[0].save(OUT/'compare.gif', save_all=True, append_images=frames[1:], duration=1000//li.FPS*3, loop=0)
json.dump(dict(views=[vnames[v] for v in args.views], iou=iou), open(OUT/'iou.json', 'w'))
if iou and iou[0]:
    I = np.array(iou)
    print('IoU against the MPM particles: frame 0 %.3f; frames 1-5 %.3f; 6-%d %.3f (mean over views)' % (I[0].mean(), I[1:6].mean(), K, I[6:].mean()), flush=True)
print('wrote %s: frame_00..%02d.png, sheet_view*.png, compare.gif, iou.json' % (OUT, K), flush=True)
