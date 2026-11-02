"""A fourier3d Power Foam run to the simulation's foam (11-02; the steps of fourier_lion_8k_round_clean's select.log and foam_exact.log, whose selection was not kept as a script):
  1. the sites inside every mask of frame 0: each site projected through every training camera (the cameras' ray model, powerfoam/camera.py: ray (x, y) = x right + y up + forward,
     x, y in [-1, 1] over the image), kept if its pixel lies in that view's mask (views V00..V11 of the dataset)
  2. the largest connected component of the power cut's walls among them, again until one is left
  3. dem_exact_single/ cells.npz, geometry.npz as foam_exact.py writes them (walls with area > 0, centroids, volumes as masses); no boundary skin, no FEM modes
  python free_surface/fourier_foam_build.py --run <fourier3d output dir> --out <dir>"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

FOUR = '/home/fangsuo/py/dynamic-powerfoam/.claude/worktrees/thin-surface/fourier3d'
D = Path('/home/fangsuo/py/dynamic-powerfoam/data/springgaus_powerfoam/vid2sim_lion_dynamic')
F_ = torch.nn.functional


def load_cameras(run):
    """the run's training cameras, as dem_gt_compare.py loads them (camera c is the dataset's view V<c>, its --check)"""
    sys.path.insert(0, FOUR)
    import configargparse
    import warp as wp
    from configs import Params, add_group
    from data_loader import DataHandler
    parser = configargparse.ArgParser()
    get_params = add_group(parser, Params)
    parser.add_argument('-c', '--config', is_config_file=True)
    pa = get_params(parser.parse_args(['-c', str(run/'config.yaml')]))
    wp.init()
    data = DataHandler(pa)
    data.reload('all', downsample=pa.downsample[-1])
    return data.cameras


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--run', type=Path, required=True)
    a.add_argument('--out', type=Path, required=True)
    a.add_argument('--views', type=int, default=12)
    args = a.parse_args()
    t0 = time.time()
    from free_surface.gpu_geometry import Geometry, structure_triangulation
    st = torch.load(args.run/'model.pt', map_location='cpu', weights_only=True)
    P_all = st['points'].double(); R_all = F_.softplus(st['radii'].double(), beta=100)
    n0 = len(P_all)
    # 1. the masks, through the run's own cameras (as dem_gt_compare.py loads them)
    cams = load_cameras(args.run)
    inside = torch.ones(n0, dtype=torch.bool)
    for v in range(args.views):
        cam = cams[v]
        eye, right, up = cam.eye.double().cpu(), cam.right.double().cpu(), cam.up.double().cpu()
        fwd = torch.linalg.cross(up, right); fwd = fwd/fwd.norm()
        d = P_all-eye
        s = d@fwd
        x = (d@right)/(right.norm()**2)/s; y = (d@up)/(up.norm()**2)/s
        m = np.array(Image.open(D/'masks'/('V%02d' % v)/'00.png').convert('L')) > 127
        H, W = m.shape
        j = torch.round((x+1)*(W-1)/2).long(); i = torch.round((1-y)*(H-1)/2).long()
        ok = (s > 0) & (i >= 0) & (i < H) & (j >= 0) & (j < W)
        hit = torch.zeros(n0, dtype=torch.bool)
        hit[ok] = torch.as_tensor(m[i[ok].numpy(), j[ok].numpy()])
        inside &= hit
    idx = torch.nonzero(inside)[:, 0].numpy()
    print('%d sites, %d inside every mask' % (n0, len(idx)), flush=True)
    # 2. the largest component of the cut's walls, until one
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    for k in range(20):
        p, r = P_all[idx].cuda(), R_all[idx].cuda()
        S = structure_triangulation(p.cpu().numpy(), r.cpu().numpy()); G = Geometry(p, r, S)
        keep = (G.wall_A > 0).cpu().numpy(); pr = G.P.cpu().numpy()[keep]
        nc, lab = connected_components(coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(len(idx), len(idx))), directed=False)
        big = np.argmax(np.bincount(lab))
        Vk = G.V.detach().cpu().numpy()
        sel = (lab == big) & (Vk > 1e-4*np.median(Vk))                              # (and no empty cell: the old foam's smallest was 5e-3 of the median, the empty ones 4e-10)
        print('pass %d: %d sites, %d kept (components %d, empty cells %d)' % (k, len(idx), int(sel.sum()), nc, int((Vk <= 1e-4*np.median(Vk)).sum())), flush=True)
        if bool(sel.all()):
            break
        idx = idx[sel]
    # 3. foam_exact.py's dem_exact_single
    src = idx.astype(np.int64)
    p = P_all[src].cuda(); r = R_all[src].cuda(); pn, rn = p.cpu().numpy(), r.cpu().numpy(); n = len(pn)
    S = structure_triangulation(pn, rn); G = Geometry(p, r, S)
    keep = (G.wall_A > 0).cpu().numpy()
    pairs = G.P.cpu().numpy()[keep]; area = G.wall_A.cpu().numpy()[keep]; cen = G.wall_c.cpu().numpy()[keep]; V = G.V.cpu().numpy()
    nw = np.bincount(pairs.ravel(), minlength=n)
    print('material sites %d (hidden %d); walls %d; volume %.6f m^3; median cell %.1f mm; smallest cell %.2e of the median; cells with < 3 walls %d; exposed area %.4f m^2 on %d cells; lowest z %.4f (%.0f s)' % (
        n, int(S['hidden'].sum()), len(pairs), V.sum(), 1e3*np.median(V)**(1/3), V.min()/np.median(V), int((nw < 3).sum()), float(G.patch_A.sum()), int((G.patch_A > 0).sum()), pn[:, 2].min(), time.time()-t0), flush=True)
    out = args.out/'dem_exact_single'; out.mkdir(parents=True, exist_ok=True)
    cells = dict(points=pn, radii=rn, pairs=pairs, areas=area, contacts=cen, masses=V, fixed=np.zeros(n, bool), density=np.ones(n), source_index=src)
    single = dict(points=pn, radii=(3*V/(4*math.pi))**(1/3), pairs=pairs, areas=area, contacts=cen, masses=V, fixed=np.zeros(n, bool), body_of_cell=np.arange(n))
    np.savez_compressed(out/'cells.npz', **cells)
    np.savez_compressed(out/'geometry.npz', **single)
    print('wrote %s' % out, flush=True)


if __name__ == '__main__':
    main()
