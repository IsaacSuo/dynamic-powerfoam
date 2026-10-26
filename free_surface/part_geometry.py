"""Exact geometry of the synthetic part (test_part.py) and its sphere surface (sphere_skin.py), checked, written as DEMs.
Checks: corners on their three spheres and outside all other balls; three arcs end at each corner; loops close; patch
area of every loop by Gauss-Bonnet (boundary only) against the fan quadrature; Euler characteristic V - E + loops (every
patch region a disk) against the part's one through hole (0); every cell closed: sum of its walls' area vectors plus
its patch's int n dA = 0; cell volumes sum_walls A h / 3 + r A_patch / 3 (exact) against the DEM's (64-gon faces,
sampled sphere) and the union volume by grid sampling; patch areas against sampling. Writes <part>/dem_exact_single
(one body per cell), <part>/dem_exact (bodies by cluster_cells, min_fraction 0.1) and <part>/sphere_skin.npz."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import qmc

from experiments.pacnerf_foam_geometry import cluster_cells
from free_surface.sphere_skin import faces_exact, fan_integrals, gauss_bonnet_area, oriented_loops, skeleton


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--part', type=Path, default=Path('output/test_part'))
    args = a.parse_args()
    st = torch.load(args.part/'run'/'model.pt', map_location='cpu', weights_only=True)
    p = st['points'].double().numpy(); r = torch.nn.functional.softplus(st['radii'].double(), beta=100).numpy()
    n = len(p)
    sk = skeleton(p, r)
    print('skeleton: %s' % sk['check'], flush=True)
    loops, broken = oriented_loops(p, r, sk)
    print('loops %d on %d patches, broken chains %d' % (len(loops), len({s for s, _ in loops}), broken), flush=True)
    A = np.zeros(n); a1 = np.zeros((n, 3)); a2 = np.zeros((n, 3, 3))
    gb_err, dots, lA = [], [], []
    for site, lp in loops:
        Ag = gauss_bonnet_area(p, r, sk, site, lp)
        Af, v1, v2, md = fan_integrals(p, r, sk, site, lp)
        gb_err.append(abs(Ag-Af)/(4*np.pi*r[site]**2)); dots.append(md); lA.append(Af)
        A[site] += Af; a1[site] += v1; a2[site] += v2
    gb_err = np.array(gb_err)
    E = sum(1 for x in sk['arcs'] if x[2] >= 0)
    chi = sk['check']['corners']-E+len(loops)
    print('patch area per loop: Gauss-Bonnet vs fan |diff| / sphere area max %.1e; pole . boundary min %.2f; loop areas min %.2e max %.2e m^2 | '
          'Euler V - E + loops = %d (one through hole: 0)' % (gb_err.max(), min(dots), min(lA), max(lA), chi), flush=True)
    F = faces_exact(p, r, sk)
    keys = sorted(F)
    pairs = np.array(keys); area = np.array([F[k][0] for k in keys]); cen = np.array([F[k][1] for k in keys])
    nrm = np.array([F[k][2]['n'] for k in keys])
    clos = a1.copy()
    np.add.at(clos, pairs[:, 0], area[:, None]*nrm); np.add.at(clos, pairs[:, 1], -area[:, None]*nrm)
    tot_area = A+np.bincount(pairs[:, 0], area, n)+np.bincount(pairs[:, 1], area, n)
    hi = ((cen-p[pairs[:, 0]])*nrm).sum(1); hj = -((cen-p[pairs[:, 1]])*nrm).sum(1)
    V = (np.bincount(pairs[:, 0], area*hi, n)+np.bincount(pairs[:, 1], area*hj, n)+r*A)/3
    print('walls %d; cell closure |sum of area vectors| / cell surface max %.1e; volume %.8e m^3, cells min %.2e' % (
        len(pairs), (np.linalg.norm(clos, axis=1)/tot_area).max(), V.sum(), V.min()), flush=True)
    dem = np.load(args.part/'dem_single'/'cells.npz')
    dk = {tuple(x): k for k, x in enumerate(np.sort(dem['pairs'], axis=1))}
    common = [k for k in keys if k in dk]
    ad = np.array([dem['areas'][dk[k]] for k in common]); ae = np.array([F[k][0] for k in common])
    print('against the DEM (64-gon faces, sampled sphere): walls exact %d, DEM %d, common %d, area |rel diff| median %.1e max %.1e; '
          'cell volume |rel diff| median %.1e max %.1e, total %.2e' % (len(keys), len(dem['pairs']), len(common), np.median(np.abs(ad/ae-1)), np.abs(ad/ae-1).max(),
                                                                      np.median(np.abs(dem['masses']/V-1)), np.abs(dem['masses']/V-1).max(), dem['masses'].sum()/V.sum()-1), flush=True)
    # sampling: union volume on a 0.5 mm grid, patch areas by rays
    g = .0005
    lo, hi_ = (p-r[:, None]).min(0), (p+r[:, None]).max(0)
    pt, rt = torch.as_tensor(p, device='cuda'), torch.as_tensor(r, device='cuda')
    inside = 0
    axes = [np.arange(lo[k]+g/2, hi_[k], g) for k in range(3)]
    for x0 in axes[0]:
        G = np.stack(np.meshgrid([x0], axes[1], axes[2], indexing='ij'), -1).reshape(-1, 3)
        y = torch.as_tensor(G, device='cuda')
        inside += int(((torch.cdist(y, pt)**2-rt[None]**2).min(1).values <= 0).sum())
    smp = qmc.Sobol(2, scramble=True, seed=11).random_base2(14)
    z, ph = 1-2*smp[:, 0], 2*np.pi*smp[:, 1]
    U = torch.as_tensor(np.column_stack((np.sqrt(1-z*z)*np.cos(ph), np.sqrt(1-z*z)*np.sin(ph), z)), device='cuda')
    As = np.zeros(n)
    for i in np.flatnonzero(A > 0):
        X = pt[i]+rt[i]*U
        pw = torch.cdist(X, pt)**2-rt[None]**2
        As[i] = 4*np.pi*r[i]**2*float((pw.argmin(1) == i).double().mean())
    print('grid 0.5 mm union volume %.6e (exact / grid - 1 = %.1e); patch area total exact %.6e, rays %.6e (rel %.1e)' % (
        inside*g**3, V.sum()/(inside*g**3)-1, A.sum(), As.sum(), As.sum()/A.sum()-1), flush=True)
    # DEMs from the exact geometry
    cells = dict(points=p, radii=r, pairs=pairs, areas=area, contacts=cen, masses=V, fixed=np.zeros(n, bool), density=np.ones(n), source_index=np.arange(n))
    single = dict(points=p, radii=(3*V/(4*np.pi))**(1/3), pairs=pairs, areas=area, contacts=cen, masses=V, fixed=np.zeros(n, bool), body_of_cell=np.arange(n))
    merged = cluster_cells(cells, min_fraction=.1)
    for tag, bodies in (('dem_exact_single', single), ('dem_exact', merged)):
        (args.part/tag).mkdir(exist_ok=True)
        np.savez_compressed(args.part/tag/'cells.npz', **cells)
        np.savez_compressed(args.part/tag/'geometry.npz', **bodies)
    print('bodies: single %d, merged %d (cells per body max %d)' % (n, len(merged['masses']), np.bincount(merged['body_of_cell']).max()), flush=True)
    arcs = sk['arcs']
    np.savez_compressed(args.part/'sphere_skin.npz', corners=sk['corners'], corner_tri=sk['corner_tri'],
                        arc_ij=np.array([[x[0], x[1], x[2], x[3]] for x in arcs]), arc_theta=np.array([[x[4], x[5]] for x in arcs]),
                        arc_circle=np.array([np.concatenate((x[6]['c'], [x[6]['rho']], x[6]['u'], x[6]['v'], x[6]['n'], [x[6]['hi'], x[6]['hj']])) for x in arcs]),
                        loop_site=np.array([s for s, _ in loops]), loop_off=np.concatenate(([0], np.cumsum([len(l_) for _, l_ in loops]))),
                        loop_arc=np.array([ai for _, l_ in loops for ai, _ in l_]), loop_fwd=np.array([f for _, l_ in loops for _, f in l_]),
                        patch_area=A, patch_n=a1, patch_nn=a2)
    (args.part/'part_geometry.json').write_text(json.dumps(dict(skeleton=sk['check'], loops=len(loops), broken=broken, euler=int(chi),
                                                                gauss_bonnet_vs_fan_max=float(gb_err.max()), volume=float(V.sum()), walls=len(pairs),
                                                                patch_area=float(A.sum())), indent=1))


if __name__ == '__main__':
    main()
