"""Build and check the boundary-only skin (boundary_skin.py) on the part and the new foam (2026-10-22), interpolation
Pi + transfinite Shepard (exact arc traces): affine reproduction and partition of unity at the quadrature points,
sums of the area and normal integrals against the exact patch geometry, the moment sum_q c_q (x) m_q against
int n n^T dA, mass-matrix conditioning, the approach to the arc traces (eta 1e-7 inside, both patches of every arc),
and convergence in the three quadrature parameters (area points nq, pieces per arc k_sub, boundary points nbd: none of
them changes the interpolant); writes <part>/boundary_skin.npz (k_sub 4, nq 20, nbd 16)."""
import argparse
from pathlib import Path

import numpy as np
import torch

from free_surface.boundary_skin import build, corners_and_mids
from free_surface.gpu_geometry import Geometry, structure_triangulation
from free_surface.gpu_geometry_check import load


def rel(a, b):
    return float(np.abs(a-b).max()/np.abs(b).max())


def moment(sk, G, p, r):
    pn, rn = p.cpu().numpy(), r.cpu().numpy(); n = len(pn)
    site = sk['ent_site']; m = (sk['Y'][sk['ent_pid']]-pn[site])/rn[site, None]
    Mc = np.zeros((n, 3, 3)); np.add.at(Mc, site, sk['ent_c'][:, :, None]*m[:, None, :])
    N = G.patch_nn.cpu().numpy(); keep = G.patch_A.cpu().numpy() > 0
    d = np.linalg.norm((Mc-N)[keep].reshape(-1, 9), axis=1)/np.linalg.norm(N[keep].reshape(-1, 9), axis=1)
    return float(np.median(d)), float(d.max())


def run(tag, path, save=None):
    p, r = load(path)
    S = structure_triangulation(p.cpu().numpy(), r.cpu().numpy()); G = Geometry(p, r, S)
    gen = torch.Generator(device='cuda').manual_seed(0)
    data = torch.randn(len(corners_and_mids(G, p)[0]), 3, generator=gen, dtype=torch.float64, device='cuda')
    res = {}
    for k_sub, nq, nbd in ((4, 12, 16), (4, 20, 16), (8, 20, 16), (4, 20, 24)):
        sk, d = build(p, r, k_sub=k_sub, nq=nq, nbd=nbd, G=G, S=S, data=data)
        pp = d.pop('per_patch')
        res[(k_sub, nq, nbd)] = sk
        keys = ('regions', 'sites_with_islands', 'affine_reproduction_max', 'partition_of_unity_max', 'tmvi_denominator_min', 'tmvi_denominator_nonpositive',
                'sumA_minus_A_of_sphere_max', 'sumC_minus_intn_of_sphere_max', 'mass_matrix_min_eig_ratio', 'seam_mismatch_max', 'off_trace_max', 'seconds')
        print('[%s] k_sub %d, nq %d, nbd %d: %s; moment sum c (x) m vs N median/max %.1e / %.1e; regions not star from m0 %d' % (
            tag, k_sub, nq, nbd, ', '.join('%s %.3g' % (k_, d[k_]) for k_ in keys), *moment(sk, G, p, r), int((pp['neg_share'] > 0).sum())), flush=True)
    b = res[(4, 20, 16)]
    for nm, x in (('area points nq 12 -> 20', res[(4, 12, 16)]), ('pieces per arc 4 -> 8', res[(8, 20, 16)]), ('boundary points 16 -> 24', res[(4, 20, 24)])):
        print('[%s] %s: a %.1e, c %.1e, m %.1e, T %.1e, U %.1e (max diff / max)' % (
            tag, nm, rel(x['ent_a'], b['ent_a']), rel(x['ent_c'], b['ent_c']), rel(x['part_m'], b['part_m']), rel(x['part_T'], b['part_T']), rel(x['part_U'], b['part_U'])), flush=True)
    if save is not None:
        np.savez_compressed(save, **b)
        print('[%s] wrote %s: %d skin points (%d corners), %d regions' % (tag, save, len(b['Y']), b['n_corners'], len(b['part_site'])), flush=True)


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--part', type=Path, default=Path('output/test_part'))
    a.add_argument('--foam', type=Path, default=Path('/home/fangsuo/py/dynamic-powerfoam/.claude/worktrees/thin-surface/output/pacnerf_elastic0_f0_round01_8k/model.pt'))
    args = a.parse_args()
    run('part', args.part/'run'/'model.pt', save=args.part/'boundary_skin.npz')
    run('foam', args.foam)


if __name__ == '__main__':
    main()
