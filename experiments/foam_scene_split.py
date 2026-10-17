"""Split a whole-scene Power Foam into an object foam and a background foam.

render_hooked: PowerfoamScene.forward with a hook on the per-texel colours (offset / per-cell
scale) - used for render-weighted voting and for per-frame background darkening.
vote_object (PowerSim sec. 4.3): s+_i = sum over views/pixels of w_i(u) M(u), s-_i likewise with
(1 - M); the rendered colour is linear in a per-cell colour offset, so both are gradients of
mask-weighted colour sums. Object cells: s+ > beta s-, AND the site inside the 3D visual hull of
the (dilated) masks (removes camera floaters), AND the largest connected set (no strays).
Object foam = object cells + a 2-ring shell of their non-object neighbours, made transparent
(they only shape the object surface and move with it); background foam = all non-object cells
unchanged (static geometry).
check_split: at rest, object foam composited over background foam vs the whole-scene render and
the captured images (PSNR, mask IoU).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from experiments.foam_dynamic_render import air_shell, keep_cells, load


def render_hooked(model, camera, texel_hook=None):
    normals = model.get_normals()
    tangents, bitangent = model.get_tangents()
    radii = model.get_radii()
    offsets = model.texel_sites*radii[:, None, None]
    offsets = offsets[..., 0:1]*tangents[:, None, :]+offsets[..., 1:2]*bitangent[:, None, :]
    texel_sites = model.points[:, None, :]+offsets
    att_sites, att_values, att_temps = model.get_att_sv()
    texel_rgb = model.sv.forward(texel_sites.view(-1, 3).detach(), camera, att_sites, att_values, att_temps)
    texel_rgb = texel_rgb.view(model.points.shape[0], model.args.num_texel_sites, 3)
    if texel_hook is not None:
        texel_rgb = texel_hook(texel_rgb)
    q = .5*torch.ones(camera.height, camera.width, 1, device=model.device)
    out = model.rasterizer.forward(camera, q, model.points, radii, model.get_density(), normals, texel_sites,
                                   texel_rgb, model.texel_height*radii[:, None], model.adjacency,
                                   model.adjacency_offsets, None, False)
    rgb, alpha = out[0], out[1]
    return rgb, alpha.reshape(*alpha.shape[:2])


def load_masks(scene_dir, data_order):
    return [torch.tensor(np.asarray(Image.open(Path(scene_dir)/'masks'/f'{name}.png')) > 127, device='cuda')
            for name in data_order]


def frame_names(scene_dir):
    return [Path(f['file_path']).stem for f in json.loads((Path(scene_dir)/'transforms_all.json').read_text())['frames']]


def vote_object(model, data, masks, beta=.5):
    n = len(model.points)
    plus = torch.zeros(n, device='cuda'); minus = torch.zeros(n, device='cuda')
    for cam, m in zip(data.cameras, masks):
        for target, acc in ((m.float(), plus), (1-m.float(), minus)):
            offset = torch.zeros(n, 1, 3, device='cuda', requires_grad=True)
            rgb, _ = render_hooked(model, cam, lambda t: t+offset)
            (rgb[..., 0]*target).sum().backward()
            acc += offset.grad[:, 0, 0]
    return (plus > beta*minus) & (plus > 1e-6), plus, minus


def visual_hull_dilated(points, scene_dir, names, dilation=4):
    """Site projects inside the (dilated) object mask in every view (OpenGL cameras)."""
    from scipy import ndimage
    meta = json.loads((Path(scene_dir)/'transforms_all.json').read_text())
    W, H = meta['w'], meta['h']
    focal = .5*W/np.tan(.5*meta['camera_angle_x'])
    inside = np.ones(len(points), bool)
    for frame in meta['frames']:
        mask = np.asarray(Image.open(Path(scene_dir)/'masks'/f"{Path(frame['file_path']).stem}.png")) > 127
        mask = ndimage.binary_dilation(mask, iterations=dilation)
        cam = (np.c_[points, np.ones(len(points))]@np.linalg.inv(np.array(frame['transform_matrix'])).T)[:, :3]
        z = -cam[:, 2]
        u = focal*cam[:, 0]/z+W/2; v = -focal*cam[:, 1]/z+H/2
        ok = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        hit = np.zeros(len(points), bool)
        hit[ok] = mask[v[ok].astype(int), u[ok].astype(int)]
        inside &= hit
    return inside


def largest_component(model, selected):
    """Largest connected set of selected sites over regular-triangulation edges."""
    from scipy import sparse
    from scipy.sparse.csgraph import connected_components
    from experiments.powersim_bread_demo import regular_candidates
    p = model.points.detach().double().cpu().numpy(); r = model.get_radii().detach().double().cpu().numpy()
    pairs, possible, _ = regular_candidates(p, r)
    pairs = pairs[possible]
    pairs = pairs[selected[pairs].all(1)]
    g = sparse.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(p),)*2)
    _, labels = connected_components(g, directed=False)
    sizes = np.bincount(labels[selected], minlength=labels.max()+1)
    return selected & (labels == sizes.argmax())


def transparent_shell(foam, n_object):
    """Shell cells (after the first n_object kept cells) only shape the object surface in the
    object foam: zero density there (softplus(-1e3) = 0); the background foam keeps them as they are."""
    with torch.no_grad():
        foam.density.data[n_object:] = -1e3


def check_split(run, scene_dir, out, beta=.5):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    full, data = load(run)
    names = frame_names(scene_dir)
    masks = load_masks(scene_dir, names)
    for p_ in ('points', 'radii', 'density', 'quaternions', 'texel_sites', 'texel_sv_axis', 'texel_sv_rgb', 'texel_height'):
        getattr(full, p_).requires_grad_(False)
    obj, plus, minus = vote_object(full, data, masks, beta)
    voted = obj.cpu().numpy()
    hull = visual_hull_dilated(full.points.detach().cpu().numpy(), scene_dir, names, dilation=4)
    obj_np = largest_component(full, voted & hull)
    print('voted', int(voted.sum()), 'inside dilated hull', int((voted & hull).sum()), 'largest component', int(obj_np.sum()), flush=True)
    obj_idx = np.flatnonzero(obj_np)
    print('object cells', len(obj_idx), 'of', len(full.points), flush=True)
    with torch.no_grad():
        ref = [render_hooked(full, cam) for cam in data.cameras]
    # Object foam: object cells + 2-ring shell (shell attached to object, owner index unused here).
    fg, _ = load(run)
    keep, _ = air_shell(fg, obj_idx, np.zeros(len(obj_idx), int), rings=2)
    keep_cells(fg, keep)
    transparent_shell(fg, len(obj_idx))
    bg, _ = load(run)
    background = np.setdiff1d(np.arange(len(full.points)), obj_idx)
    keep_cells(bg, background)
    rows = []
    with torch.no_grad():
        for k, cam in enumerate(data.cameras):
            rgb_f, a_f = render_hooked(fg, cam)
            rgb_b, _ = render_hooked(bg, cam)
            comp = rgb_f.clamp(0, 1)+(1-a_f.clamp(0, 1))[..., None]*rgb_b.clamp(0, 1)
            gt = data.rgbs[k].cuda()
            psnr = lambda x: float(-10*torch.log10(((x.clamp(0, 1)-gt)**2).mean()))
            m, g = a_f > .5, masks[k]
            rows.append(dict(view=names[k], psnr_full=psnr(ref[k][0]), psnr_split=psnr(comp),
                             iou_object_alpha_vs_mask=float((m & g).sum()/(m | g).sum())))
            if k == 0:
                im = torch.cat((gt, ref[k][0].clamp(0, 1), comp, rgb_b.clamp(0, 1)), 1).cpu().numpy()
                Image.fromarray((im*255).astype(np.uint8)).save(out/'split_view0.png')
    summary = {key: float(np.mean([r[key] for r in rows])) for key in rows[0] if key != 'view'}
    np.savez_compressed(out/'split.npz', object_cells=obj_idx, object_keep=keep, background_cells=background,
                        s_plus=plus.cpu().numpy(), s_minus=minus.cpu().numpy())
    (out/'split.json').write_text(json.dumps(dict(beta=beta, object_cells=int(len(obj_idx)), object_with_shell=int(len(keep)),
                                                  background_cells=int(len(background)), summary=summary, views=rows), indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--run', default='output/pacnerf_elastic0_f0_full_v1')
    a.add_argument('--scene-dir', default='data/pacnerf_powerfoam/elastic_0_f0_full')
    a.add_argument('--out', default='output/pacnerf_dem/elastic_0_f0/scene_split')
    a.add_argument('--beta', type=float, default=.5)
    args = a.parse_args()
    check_split(args.run, args.scene_dir, args.out, args.beta)
