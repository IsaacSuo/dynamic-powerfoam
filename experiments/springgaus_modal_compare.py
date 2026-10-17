"""Root cause, last step: DEM vs continuum FEM eigenmodes of the SAME burger shape (free-free).

Shape: material power cells of the static foam (owner by power distance, inside the support sphere),
voxelised at --h in the dynamic world (similarity foam -> world of the DEM registration), largest
face-connected component, Hex1 FEM (K = lam K_lam + mu K_mu, mass at rho). DEM: EnergyFoam
linearised at rest (pacnerf_modal_check.dem_stiffness), mode shapes at the body centres.
Pairing by mass-weighted MAC (Hungarian). Reported per paired mode: f_DEM / f_FEM (< 1: DEM softer)
and the share of the DEM mode's kinetic energy carried by surface bodies (fewer material neighbours)
compared with their mass share -> is the softness localised in the surface layer?
"""
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

import scipy.sparse.linalg as sla

from experiments.pacnerf_dem_forward import build, RHO
from experiments.pacnerf_foam_geometry import foam_to_dynamic
from experiments.pacnerf_modal_check import dem_stiffness

CORNERS = np.array([(i, j, k) for k in (0, 1) for j in (0, 1) for i in (0, 1)])


def hex_element(h):
    """Trilinear cube element of side h, 2x2x2 Gauss: K_lam (lam = 1) and K_mu (mu = 1), 24 x 24, dof (node, comp)."""
    gp = (np.array([-1, 1])/np.sqrt(3)+1)/2
    Kl = np.zeros((8, 3, 8, 3)); Km = np.zeros((8, 3, 8, 3))
    for a in gp:
        for b in gp:
            for c in gp:
                xi = np.array([a, b, c])
                grad = np.zeros((8, 3))
                for n, o in enumerate(CORNERS):
                    f = np.where(o == 1, xi, 1-xi)
                    df = np.where(o == 1, 1., -1.)
                    grad[n] = [df[0]*f[1]*f[2], f[0]*df[1]*f[2], f[0]*f[1]*df[2]]
                grad /= h
                w = h**3/8
                Kl += w*np.einsum('ai,bj->aibj', grad, grad)
                Km += w*(np.einsum('ak,bk,ij->aibj', grad, grad, np.eye(3))+np.einsum('aj,bi->aibj', grad, grad))
    return Kl.reshape(24, 24), Km.reshape(24, 24)


def hex_fem(vox, lo, h):
    nodes_all = (vox[:, None, :]+CORNERS[None]).reshape(-1, 3)
    uniq, inv = np.unique(nodes_all, axis=0, return_inverse=True)
    conn = inv.reshape(-1, 8)
    Kl_e, Km_e = hex_element(h)
    dof = (conn[:, :, None]*3+np.arange(3)[None, None]).reshape(len(conn), 24)
    rows = np.repeat(dof, 24, 1).ravel(); cols = np.tile(dof, (1, 24)).ravel()
    N = 3*len(uniq)
    K_lam = sp.csr_matrix((np.tile(Kl_e.ravel(), len(conn)), (rows, cols)), shape=(N, N))
    K_mu = sp.csr_matrix((np.tile(Km_e.ravel(), len(conn)), (rows, cols)), shape=(N, N))
    lump = np.zeros(len(uniq)); np.add.at(lump, conn.ravel(), h**3/8)
    M = sp.diags(np.repeat(lump, 3))
    return K_lam, K_mu, M, {tuple(n): k for k, n in enumerate(uniq)}, set(map(tuple, vox)), N


def hex_probe(node_id, vox_set, lo, h, points, N):
    rows, cols, vals = [], [], []
    occupied = np.array(sorted(vox_set)); tree = cKDTree((occupied+.5)*h+lo)
    for i, x in enumerate(points):
        v = np.floor((x-lo)/h).astype(int)
        if tuple(v) not in vox_set:
            v = occupied[tree.query(x)[1]]
        t = np.clip((x-lo)/h-v, 0, 1)
        for o in CORNERS:
            w = np.prod(np.where(o == 1, t, 1-t))
            nid = node_id[tuple(v+o)]
            for comp in range(3):
                rows.append(comp*len(points)+i); cols.append(3*nid+comp); vals.append(w)
    return sp.csr_matrix((vals, (rows, cols)), shape=(3*len(points), N))


def fem_modes(K_lam, K_mu, M, nu, k):
    lam = 2*nu/(1-2*nu)                                        # mu = 1
    K = (lam*K_lam+K_mu).tocsc()
    vals, vecs = sla.eigsh(K, k=k+6, M=M.tocsc(), sigma=-1e-3, which='LM')
    o = np.argsort(vals)
    return vals[o][6:], vecs[:, o][:, 6:]


def dem_modes_with_shapes(model, theta, k):
    K = dem_stiffness(model, theta)
    m = np.concatenate((np.repeat(model.mass.numpy()[:, None], 3, 1), np.repeat(model.inertia.numpy()[:, None], 3, 1)), 1).ravel()
    Mi = sp.diags(1/np.sqrt(m))
    vals, vecs = sla.eigsh((Mi@K@Mi).tocsc(), k=k+6, sigma=-1e-3, which='LM')
    o = np.argsort(vals)
    vals, vecs = vals[o][6:], (Mi@vecs[:, o])[:, 6:]
    return vals, vecs.reshape(model.n, 6, -1)[:, :3, :]


def mac_matrix(a_, b_, mass):
    a = (a_*np.sqrt(mass)[:, None, None]).reshape(-1, a_.shape[-1]); b = (b_*np.sqrt(mass)[:, None, None]).reshape(-1, b_.shape[-1])
    return (a.T@b)**2/np.outer((a*a).sum(0), (b*b).sum(0))


def world_voxels(run, dem, scene_dir, registration, h):
    """registration None: Blender scene, world = foam - offset (conversion.json)."""
    st = torch.load(Path(run)/'model.pt', map_location='cpu', weights_only=True)
    p = st['points'].double().numpy()
    r = torch.nn.functional.softplus(st['radii'].double(), beta=100).numpy()
    cells = dict(np.load(Path(dem)/'cells.npz'))
    is_mat = np.zeros(len(p), bool); is_mat[cells['source_index']] = True
    if registration is None:
        A, b, k = np.eye(3), -np.array(json.loads((Path(scene_dir)/'conversion.json').read_text())['offset']), 1.
    else:
        A, b, k = foam_to_dynamic(scene_dir, registration)
    pw, rw = p@A.T+b, r*k                                      # similarity: power distances scale by k^2
    q = pw[is_mat]
    lo = q.min(0)-.01; hi = q.max(0)+.01
    axes = [np.arange(lo[d]+h/2, hi[d], h) for d in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
    _, nn = cKDTree(pw).query(grid, k=8)
    power = ((grid[:, None]-pw[nn])**2).sum(-1)-rw[nn]**2
    best = power.argmin(1)
    owner = nn[np.arange(len(grid)), best]
    occ = (is_mat[owner] & (power[np.arange(len(grid)), best] <= 0)).reshape(tuple(len(x) for x in axes))
    occ = ndimage.binary_fill_holes(occ)
    lab, cnt = ndimage.label(occ)
    occ = lab == 1+int(np.argmax(ndimage.sum(occ, lab, range(1, cnt+1))))
    return np.argwhere(occ), lo, h


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='output/springgaus_burger_spacing8k')
    ap.add_argument('--dem', default='output/springgaus_dem/burger_spacing8k_mf0.1')
    ap.add_argument('--scene', default='data/springgaus_powerfoam/burger')
    ap.add_argument('--registration', default='data/springgaus_powerfoam/burger_dynamic/register_calib/initial_state_upright_nospin.json')
    ap.add_argument('--E', type=float, default=2e4)
    ap.add_argument('--nu', type=float, default=.3)
    ap.add_argument('--h', type=float, default=.0025)
    ap.add_argument('--modes', type=int, default=16)
    args = ap.parse_args()
    run, demdir, scene = args.run, Path(args.dem), Path(args.scene)
    reg = None if args.registration in ('', 'none') else Path(args.registration)
    E, nu, h, kmodes = args.E, args.nu, args.h, args.modes
    g = dict(np.load(demdir/'geometry.npz'))
    vox, lo, h = world_voxels(run, demdir, scene, reg, h)
    K_lam, K_mu, M, node_id, vox_set, N = hex_fem(vox, lo, h)
    print('FEM: %d hexes, volume %.1f cm^3, %d dofs | DEM volume %.1f cm^3, %d bodies' % (
        len(vox), len(vox)*h**3*1e6, N, g['masses'].sum()*1e6, len(g['masses'])), flush=True)
    fvals, fvecs = fem_modes(K_lam, K_mu, M, nu, kmodes+4)       # unit mu
    P = hex_probe(node_id, vox_set, lo, h, g['points'], N)
    fshapes = (P@fvecs).reshape(3, len(g['points']), -1).transpose(1, 0, 2)
    # FEM is in absolute units with mass h^3/8 per node (rho = 1): eigenvalues are omega^2 / (mu / rho) at mu = 1
    mu_true = E/(2*(1+nu))/RHO
    f_fem = np.sqrt(fvals*mu_true)/(2*np.pi)
    deg = np.bincount(g['pairs'].ravel(), minlength=len(g['masses']))
    surface = deg < np.median(deg)-2
    mass = g['masses']
    out = dict(surface_mass_share=float(mass[surface].sum()/mass.sum()), surface_bodies=int(surface.sum()))
    print('surface bodies %d (%.0f%% of mass)' % (surface.sum(), 100*out['surface_mass_share']), flush=True)
    for kin, res_on in (('small', False), ('small', True)):
        model = build(g, E, nu, 'cpu', 'noslip', .25, 0., torch.float64, kinematics=kin, residual=res_on)
        kin = kin+('+residual' if res_on else '')
        theta = torch.zeros(model.n, 7, dtype=torch.float64)
        dvals, dshapes = dem_modes_with_shapes(model, theta, kmodes)
        f_dem = np.sqrt(np.clip(dvals, 0, None))/(2*np.pi)
        mac = mac_matrix(dshapes, fshapes, mass)
        rows_, cols_ = linear_sum_assignment(-mac)
        res = []
        for r_, c_ in sorted(zip(rows_, cols_)):
            ke = mass[:, None]*(dshapes[:, :, r_]**2)
            surf_share = float(ke[surface].sum()/ke.sum())
            res.append(dict(dem_mode=int(r_), fem_mode=int(c_), f_dem=float(f_dem[r_]), f_fem=float(f_fem[c_]),
                            ratio=float(f_dem[r_]/f_fem[c_]), mac=float(mac[r_, c_]), surface_ke_share=surf_share))
        out[kin] = res
        good = [r for r in res if r['mac'] > .8]
        print('[%s] modes paired with MAC > 0.8: %d of %d; their frequency ratio DEM/FEM: %s; unpaired (spurious) among the lowest 8 DEM modes: %d' % (
            kin, len(good), len(res), np.round([r['ratio'] for r in good], 2).tolist(), sum(1 for r in res if r['dem_mode'] < 8 and r['mac'] < .3)))
        print('[%s] lowest DEM frequencies (Hz): %s' % (kin, np.round(f_dem[:8], 1).tolist()))
        print('[%s] lowest FEM frequencies (Hz): %s' % (kin, np.round(f_fem[:8], 1).tolist()))
        for r in res[:12]:
            print('  DEM mode %2d <-> FEM mode %2d  f %6.1f / %6.1f Hz  ratio %.2f  MAC %.2f  surface KE share %.2f' % (
                r['dem_mode'], r['fem_mode'], r['f_dem'], r['f_fem'], r['ratio'], r['mac'], r['surface_ke_share']), flush=True)
    (demdir/'modal_compare.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
