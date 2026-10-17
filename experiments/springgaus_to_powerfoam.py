"""Convert one Spring-Gaus real-capture static object (COLMAP + masks) into a Power Foam COLMAP scene.

Images become RGBA PNGs (alpha = object mask, straight alpha, black outside), downsampled by
--downsample into images_<d>/; the COLMAP model is renamed to the PNGs and moved by a similarity
x' = s (x - c) + 0.5 so the object's visual hull bounding box fits in Power Foam's random_bounded
init cube [0, 1]^3 (largest side -> --fill). The hull is carved on a grid around the point nearest
to all camera optical axes. Some COLMAP poses are wrong (burger: DSC09936/09990/09991 carve away
most of the object although their masks are fine): a loose hull allowing --misses disagreeing
views is carved first, views covering < --min-coverage of it are dropped (deregistered, PNG
removed), and the strict hull (inside the mask of every kept view) gives the box and
hull_colmap.npy (voxel centres).
conversion.json keeps (s, c) so foam coordinates can be mapped back to COLMAP's.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pycolmap
from PIL import Image

SRC = Path('data/springgaus/real_capture/static')
OUT = Path('data/springgaus_powerfoam')


def axes_meeting_point(recon):
    A = np.zeros((3, 3)); b = np.zeros(3)
    for im in recon.images.values():
        c = im.projection_center(); d = im.viewing_direction()
        P = np.eye(3)-np.outer(d, d)
        A += P; b += P@c
    return np.linalg.solve(A, b)


def carve(recon, masks, centre, half, n=96):
    """Grid around centre; hits[view, voxel] = voxel projects inside that view's mask."""
    ax = np.linspace(-half, half, n)
    grid = np.stack(np.meshgrid(ax, ax, ax, indexing='ij'), -1).reshape(-1, 3)+centre
    names, hits = [], []
    for im in recon.images.values():
        cam = recon.cameras[im.camera_id]
        fx, fy, cx, cy = cam.params
        Rt = im.cam_from_world().matrix()
        x = grid@Rt[:, :3].T+Rt[:, 3]
        z = x[:, 2]
        u = fx*x[:, 0]/z+cx; v = fy*x[:, 1]/z+cy
        ok = (z > 0) & (u >= 0) & (u < cam.width) & (v >= 0) & (v < cam.height)
        m = masks[im.name]
        hit = np.zeros(len(grid), bool)
        hit[ok] = m[(v[ok]*m.shape[0]/cam.height).astype(int), (u[ok]*m.shape[1]/cam.width).astype(int)]
        names.append(im.name); hits.append(hit)
    return grid, names, np.array(hits)


def robust_hull(recon, masks, centre, half, n, misses, min_coverage):
    """Hull allowing `misses` disagreeing views; views covering < min_coverage of it are flagged
    (wrong pose); strict hull (inside every kept view) returned with the flagged names."""
    grid, names, hits = carve(recon, masks, centre, half, n)
    loose = (~hits).sum(0) <= misses
    coverage = hits[:, loose].mean(1)
    bad = coverage < min_coverage
    strict = hits[~bad].all(0)
    return grid[strict], [nm for nm, b in zip(names, bad) if b], dict(zip(names, coverage.round(3).tolist()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--obj', default='burger')
    p.add_argument('--downsample', type=int, default=4)
    p.add_argument('--fill', type=float, default=.7)
    p.add_argument('--misses', type=int, default=3, help='views allowed to disagree in the loose hull')
    p.add_argument('--min-coverage', type=float, default=.9, help='drop views (bad poses) covering less of the loose hull')
    args = p.parse_args()
    recon = pycolmap.Reconstruction(str(SRC/'colmap'/args.obj))
    out = OUT/args.obj
    img_dir = out/f'images_{args.downsample}'
    img_dir.mkdir(parents=True, exist_ok=True)
    masks_small = {}
    for im in recon.images.values():
        stem = Path(im.name).stem
        rgb = Image.open(SRC/'images'/args.obj/im.name).convert('RGB')
        mask = Image.open(SRC/'masks'/args.obj/f'{stem}.png').convert('L')
        size = (rgb.width//args.downsample, rgb.height//args.downsample)
        rgb = rgb.resize(size, Image.LANCZOS); mask = mask.resize(size, Image.BILINEAR)
        a = np.asarray(mask, np.float32)/255.
        rgba = np.concatenate((np.asarray(rgb, np.float32)*a[..., None], a[..., None]*255.), -1)
        Image.fromarray(rgba.round().clip(0, 255).astype(np.uint8), 'RGBA').save(img_dir/f'{stem}.png')
        masks_small[im.name] = a > .5
    centre = axes_meeting_point(recon)
    cam_dist = np.median([np.linalg.norm(im.projection_center()-centre) for im in recon.images.values()])
    hull, _, _ = robust_hull(recon, masks_small, centre, .6*cam_dist, 96, args.misses, args.min_coverage)
    # refine on a finer grid around the first hull
    c2 = (hull.min(0)+hull.max(0))/2
    hull, bad, coverage = robust_hull(recon, masks_small, c2, .6*(hull.max(0)-hull.min(0)).max(), 128, args.misses, args.min_coverage)
    for name in bad:
        recon.deregister_image(recon.find_image_with_name(name).image_id)
        (img_dir/f'{Path(name).stem}.png').unlink()
    lo, hi = hull.min(0), hull.max(0)
    np.save(out/'hull_colmap.npy', hull.astype(np.float32))
    c = (lo+hi)/2
    s = args.fill/(hi-lo).max()
    recon.transform(pycolmap.Sim3d(s, pycolmap.Rotation3d(), -s*c+.5))
    for im in recon.images.values():
        im.name = Path(im.name).stem+'.png'
    sparse = out/'sparse'/'0'
    if sparse.exists():
        shutil.rmtree(sparse)
    sparse.mkdir(parents=True)
    recon.write_binary(str(sparse))
    info = dict(obj=args.obj, scale=float(s), centre=c.tolist(), hull_min=lo.tolist(), hull_max=hi.tolist(),
                hull_voxels=int(len(hull)), images=recon.num_reg_images(), dropped_views=bad,
                coverage_of_loose_hull=coverage, downsample=args.downsample,
                note='foam = scale*(colmap - centre) + 0.5')
    (out/'conversion.json').write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == '__main__':
    main()
