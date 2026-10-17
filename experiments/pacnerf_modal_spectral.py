"""Error budget, layer 1-2: mode-shape-paired DEM vs FEM on the SAME (reconstructed) shape, and
(E, nu) identification of a DEM by spectral matching.

Shape: the material region of the reconstructed foam (voxel centre owned, by power distance and
inside the owner's support sphere, by a material cell), voxelised at h, holes filled, largest
face-connected component (no corner/edge hinges), trilinear hexahedra.
K = lam K_lam + mu K_mu, so one eigen-solve per nu (mu = 1) gives all E: omega^2 scales with mu.
DEM modes: EnergyFoam linearised at rest (pacnerf_modal_check.dem_stiffness), mode shapes taken
at the body centres (translations). FEM shapes are interpolated trilinearly at the same points.
Pairing: Hungarian assignment on the mass-weighted MAC between the first k modes of each.
Spectral fit for each nu: log mu = 2 mean(log f_DEM - log f_FEM,unit) over paired modes; the
fitted (E, nu) minimises the RMS log-frequency residual. Units: stiffness per unit density.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import torch
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from experiments.foam_fem_reference import cholesky
from experiments.pacnerf_dem_forward import build
from experiments.pacnerf_modal_check import dem_stiffness

RHO = 1000.


def material_voxels(model_pt, material_index, offset, h):
    st = torch.load(model_pt, map_location='cpu', weights_only=True)
    p = st['points'].double().numpy()
    r = torch.nn.functional.softplus(st['radii'].double(), beta=100).numpy()
    is_mat = np.zeros(len(p), bool); is_mat[material_index] = True
    q = p[is_mat]
    lo = q.min(0)-.05; hi = q.max(0)+.05
    axes = [np.arange(lo[k]+h/2, hi[k], h) for k in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
    _, nn = cKDTree(p).query(grid, k=8)
    power = ((grid[:, None]-p[nn])**2).sum(-1)-r[nn]**2
    best = power.argmin(1)
    owner = nn[np.arange(len(grid)), best]
    occ = is_mat[owner] & (power[np.arange(len(grid)), best] <= 0)
    shape = tuple(len(a) for a in axes)
    occ = ndimage.binary_fill_holes(occ.reshape(shape))
    labels, count = ndimage.label(occ)          # face (6-) connectivity: no corner/edge hinges
    sizes = ndimage.sum(occ, labels, range(1, count+1))
    occ = labels == 1+int(np.argmax(sizes))
    vox = np.argwhere(occ)
    return vox, lo-np.asarray(offset), h    # world coordinates of the grid origin


def hex_fem(vox, lo, h):
    from skfem import Basis, BilinearForm, ElementHex1, ElementVector, MeshHex, asm
    from skfem.helpers import dot
    from skfem.models.elasticity import linear_elasticity
    order = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
    nodes_all = (vox[:, None, :]+order[None]).reshape(-1, 3)
    uniq, inv = np.unique(nodes_all, axis=0, return_inverse=True)
    mesh = MeshHex(np.ascontiguousarray((uniq*h+lo).T), np.ascontiguousarray(inv.reshape(-1, 8).T))
    basis = Basis(mesh, ElementVector(ElementHex1()))
    K_lam = asm(linear_elasticity(1., 0.), basis).tocsc()
    K_mu = asm(linear_elasticity(0., 1.), basis).tocsc()

    @BilinearForm
    def mass(u, v, w):
        return dot(u, v)
    M = asm(mass, basis).tocsc()
    node_id = {tuple(n): k for k, n in enumerate(uniq)}
    return basis, K_lam, K_mu, M, node_id, set(map(tuple, vox))


def hex_probe(basis, node_id, vox_set, lo, h, points):
    """Trilinear interpolation matrix (3 n_points x N), component-major rows."""
    rows, cols, vals = [], [], []
    occupied = np.array(sorted(vox_set))
    tree = cKDTree((occupied+.5)*h+lo)
    for i, x in enumerate(points):
        v = np.floor((x-lo)/h).astype(int)
        if tuple(v) not in vox_set:
            v = occupied[tree.query(x)[1]]
        t = np.clip((x-lo)/h-v, 0, 1)
        for c in range(8):
            o = np.array([(c >> 0) & 1, (c >> 1) & 1, (c >> 2) & 1])
            w = np.prod(np.where(o == 1, t, 1-t))
            nid = node_id[tuple(v+o)]
            for comp in range(3):
                rows.append(comp*len(points)+i); cols.append(basis.nodal_dofs[comp, nid]); vals.append(w)
    return sp.csr_matrix((vals, (rows, cols)), shape=(3*len(points), basis.N))


def fem_modes(K_lam, K_mu, M, nu, k):
    lam = 2*nu/(1-2*nu)            # mu = 1
    K = (lam*K_lam+K_mu).tocsc()
    sigma = -1e-3
    factor = cholesky((K-sigma*M).tocsc())
    op = sla.LinearOperator(K.shape, matvec=factor.solve, dtype=float)
    vals, vecs = sla.eigsh(K, k=k+6, M=M, sigma=sigma, which='LM', OPinv=op)
    order = np.argsort(vals)
    return vals[order][6:], vecs[:, order][:, 6:]


def dem_modes_with_shapes(model, theta, k):
    K = dem_stiffness(model, theta)
    m = np.concatenate((np.repeat(model.mass.numpy()[:, None], 3, 1), np.repeat(model.inertia.numpy()[:, None], 3, 1)), 1).ravel()
    Mi = sp.diags(1/np.sqrt(m))
    vals, vecs = sla.eigsh((Mi@K@Mi).tocsc(), k=k+6, sigma=-1e-3, which='LM')
    order = np.argsort(vals)
    vals, vecs = vals[order][6:], (Mi@vecs[:, order])[:, 6:]
    shapes = vecs.reshape(model.n, 6, -1)[:, :3, :]           # body translations
    return vals, shapes


def mac_matrix(dem_shapes, fem_shapes, mass):
    a = dem_shapes*np.sqrt(mass)[:, None, None]
    b = fem_shapes*np.sqrt(mass)[:, None, None]
    A = a.reshape(-1, a.shape[-1]); B = b.reshape(-1, b.shape[-1])
    num = (A.T@B)**2
    return num/np.outer((A*A).sum(0), (B*B).sum(0))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dem', type=Path, default=Path('output/pacnerf_dem/elastic_0_f0'))
    p.add_argument('--foam', default='output/pacnerf_elastic0_f0_v1/model.pt')
    p.add_argument('--h', type=float, default=.01)
    p.add_argument('--modes', type=int, default=16)
    p.add_argument('--nus', type=float, nargs='+', default=[.15, .2, .25, .3, .35, .4, .45])
    args = p.parse_args()
    gt = json.loads(Path('data/pacnerf/gt/elastic.json').read_text())['0']
    g = dict(np.load(args.dem/'geometry.npz')); cells = dict(np.load(args.dem/'cells.npz'))
    offset = json.loads(Path('data/pacnerf_powerfoam/elastic_0_f0/conversion.json').read_text())['offset']
    vox, lo, h = material_voxels(args.foam, cells['source_index'], offset, args.h)
    basis, K_lam, K_mu, M, node_id, vox_set = hex_fem(vox, lo, h)
    P = hex_probe(basis, node_id, vox_set, lo, h, g['points'])
    out = dict(voxel_volume=float(len(vox)*h**3), dem_volume=float(g['masses'].sum()), fem_dofs=int(basis.N))
    print('shape: voxel volume %.4f, DEM volume %.4f, FEM dofs %d' % (out['voxel_volume'], out['dem_volume'], basis.N), flush=True)
    fem = {}
    for nu in args.nus:
        vals, vecs = fem_modes(K_lam, K_mu, M, nu, args.modes)
        shapes = (P@vecs).reshape(3, len(g['points']), -1).transpose(1, 0, 2)
        fem[nu] = (vals, shapes)
        print('FEM nu %.2f unit-mu frequencies (Hz at mu=1):' % nu, np.round(np.sqrt(vals)/(2*np.pi), 3), flush=True)
    model = build(g, gt['E'], gt['nu'], 'cpu', 'noslip', .25, .3, torch.float64)
    runs = dict(theta0=torch.zeros(model.n, 7, dtype=torch.float64),
                trained_shared=torch.tensor(json.loads((args.dem/'silhouette_v1/report.json').read_text())['best']['parameters'], dtype=torch.float64).expand(model.n, 7).clone(),
                trained_perbody=torch.tensor(np.load(args.dem/'perbody_smooth1e-2/theta_best.npz')['theta'], dtype=torch.float64))
    mass = g['masses']
    for name, theta in runs.items():
        dvals, dshapes = dem_modes_with_shapes(model, theta, args.modes)
        fits = []
        for nu, (fvals, fshapes) in fem.items():
            mac = mac_matrix(dshapes, fshapes, mass)
            rows, cols = linear_sum_assignment(-mac)
            good = mac[rows, cols] > .5
            r = .5*np.log(dvals[rows][good])-.5*np.log(fvals[cols][good])   # log f_DEM - log f_FEM,unit
            log_mu = 2*r.mean()
            resid = float(np.sqrt(np.mean((r-r.mean())**2)))
            mu = np.exp(log_mu)*RHO
            fits.append(dict(nu=nu, E=float(2*(1+nu)*mu), mu=float(mu), residual_log=resid, paired=int(good.sum()),
                             mac_mean=float(mac[rows, cols][good].mean())))
        best = min(fits, key=lambda f: f['residual_log'])
        # frequency ratios against FEM at the TRUE material (paired at true nu)
        nu_gt = min(fem, key=lambda v: abs(v-gt['nu']))
        fvals, fshapes = fem[nu_gt]
        mac = mac_matrix(dshapes, fshapes, mass)
        rows, cols = linear_sum_assignment(-mac)
        mu_gt = gt['E']/(2*(1+gt['nu']))/RHO
        ratio = np.sqrt(dvals[rows]/(fvals[cols]*mu_gt))
        out[name] = dict(fits=fits, best=best, paired_ratio_at_true=ratio.tolist(), paired_mac_at_true=mac[rows, cols].tolist())
        print('%-16s best fit E %.0f (x%.2f of GT) nu %.2f, log residual %.3f, paired %d (MAC mean %.2f) | freq ratio vs FEM(GT): mean %.3f [%.3f..%.3f]' % (
            name, best['E'], best['E']/gt['E'], best['nu'], best['residual_log'], best['paired'], best['mac_mean'],
            ratio.mean(), ratio.min(), ratio.max()), flush=True)
    dst = args.dem/'error_budget'; dst.mkdir(exist_ok=True)
    (dst/'modal_spectral.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
