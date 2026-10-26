"""Free vibration of the synthetic part (exact geometry, one body per cell) against the voxel FEM of the same shape (the
union of the balls; dem_surface_completion_modal.fem_frequencies, per unit density as build_geometry.py), 2026-10-22.
Three models, linear bonds, lumped body masses, massless skin:
  A  current: bonds, walls, volume; the outer layer by completion springs from the exposed spheres (eigen-directions of
     r N per patch, as SurfaceCompletion's patch springs; strain form k / 2 |(F_B - R_B) l|^2, the gradient filled from
     the neighbours as MidpointTerms); also exact for affine maps on this part
  B  sphere skin (sphere_model.SphereTerms) with the skin points on the FEM displacement (Rayleigh only)
  C  sphere skin, skin free: the skin condensed out exactly, K = K_xx - K_xy K_yy^-1 K_yx (dense)
Per FEM mode (displacements interpolated to the bodies and skin points): stiffness phi^T K phi / omega_FEM^2, lumped mass
sum m |phi|^2, Rayleigh frequency ratio; and each model's own modes (M^-1/2 K M^-1/2), paired with the FEM's by
mass-weighted MAC (Hungarian): frequency ratios. For C, the energy share of each term along the condensed FEM modes."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

import prototypes.kb_derivation.clean_model as _cm
from experiments.dem_fem_mode_match import interpolate
from experiments.dem_surface_completion_modal import fem_frequencies
from experiments.pacnerf_dem_forward import build
from free_surface.grad_weights import MidpointTerms
from free_surface.sphere_model import SphereTerms
from free_surface.surface_model import cof3, det3

_cm.cof = cof3
torch.linalg.det = det3                      # only 3x3 determinants here; torch's has a NaN second derivative at F = I
f64, dev = torch.float64, 'cuda'


def dense_hessian(f, n, chunk=64):
    z0 = torch.zeros(n, dtype=f64, device=dev)
    g = torch.func.grad(f)
    hvp = lambda v: torch.func.jvp(g, (z0,), (v,))[1]
    H = torch.zeros(n, n, dtype=f64, device=dev); eye = torch.eye(n, dtype=f64, device=dev)
    for s in range(0, n, chunk):
        H[s:s+chunk] = torch.func.vmap(hvp)(eye[s:s+chunk])
    return .5*(H+H.T)


class PatchSprings:
    """stand-in for SurfaceCompletion: no air faces, the exposed-sphere springs of every patch"""
    def __init__(self, g, r, skin, mu, V):
        boc = g['body_of_cell']; b, lev, k = [], [], []
        for c in np.flatnonzero(skin['patch_area'] > 0):
            ev, evec = np.linalg.eigh(r[c]*skin['patch_nn'][c])
            for s_k, e_k in zip(ev, evec.T):
                if s_k > 1e-12*ev[-1]:
                    b.append(boc[c]); lev.append(r[c]*e_k); k.append(2*mu*s_k/r[c]**2)
        tt = lambda a: torch.as_tensor(np.asarray(a), dtype=f64, device=dev)
        self.vs_body = torch.as_tensor(np.array(b), device=dev); self.vs_lever = tt(lev); self.vs_k = tt(k)
        self.air_faces = dict(body=np.zeros(0, int), area=np.zeros(0)); self.V0 = V


def mac_pairs(Z, Uq, W):
    """mass-weighted MAC between model modes Z (bodies, 3, d) and FEM modes Uq (bodies, 3, f); Hungarian pairing"""
    num = np.einsum('n,nkd,nkf->df', W, Z, Uq)**2
    den = np.outer(np.einsum('n,nkd,nkd->d', W, Z, Z), np.einsum('n,nkf,nkf->f', W, Uq, Uq))
    M = num/den
    ri, ci = linear_sum_assignment(-M)
    return {int(f): (int(d), float(M[d, f])) for d, f in zip(ri, ci)}


def own_modes(K, mass, k):
    Mi = (1/torch.sqrt(mass)).repeat_interleave(3)
    w, V = torch.linalg.eigh(Mi[:, None]*K*Mi[None, :])
    w, V = w.cpu().numpy(), (Mi[:, None]*V).cpu().numpy()
    return w[:6], np.sqrt(np.clip(w[6:6+k], 0, None))/(2*np.pi), V[:, 6:6+k].reshape(-1, 3, k)


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--part', type=Path, default=Path('output/test_part'))
    a.add_argument('--fem-h', type=float, nargs='+', default=[.002, .0015])
    a.add_argument('--modes', type=int, default=20)
    args = a.parse_args()
    t0 = time.time()
    stt = torch.load(args.part/'run'/'model.pt', map_location='cpu', weights_only=True)
    p = stt['points'].double().numpy(); r = torch.nn.functional.softplus(stt['radii'].double(), beta=100).numpy()
    E, nu = 23008.38, .3008
    fem = {}
    for h in args.fem_h:
        cache = args.part/('fem_modes_h%g.npz' % h)
        if cache.exists():
            c_ = np.load(cache)
            f, V, F_ = c_['f'], c_['V'], dict(nodes=c_['nodes'], node_grid=c_['node_grid'], lo=c_['lo'], h=float(c_['h']), masses=c_['masses'])
        else:
            f, V, F_ = fem_frequencies(p, r, np.ones(len(p), bool), np.zeros(3), E/1000., nu, h, args.modes+6, return_modes=True)
            np.savez(cache, f=f, V=V, **{k_: np.asarray(v_) for k_, v_ in F_.items()})
        fem[h] = (f[:args.modes], V[:, :args.modes], F_)
        print('FEM h %.1f mm: %d nodes, lowest %s Hz (%.0f s)' % (1e3*h, len(F_['nodes']), np.round(f[:6], 2).tolist(), time.time()-t0), flush=True)
    hs = sorted(fem)
    if len(hs) > 1:
        print('FEM convergence: f(h %.1f mm) / f(h %.1f mm) - 1 per mode: %s' % (1e3*hs[0], 1e3*hs[-1], np.round(fem[hs[0]][0]/fem[hs[-1]][0]-1, 4).tolist()), flush=True)
    f_fem, V_fem, F_ = fem[hs[0]]
    g = dict(np.load(args.part/'dem_exact_single'/'geometry.npz'))
    skin = dict(np.load(args.part/'sphere_skin.npz')); pts = dict(np.load(args.part/'sphere_points.npz'))
    m = build(g, E, nu, dev, 'noslip', .25, .3, f64, kinematics='exact', nodalvol=True, nonlinear=False, ogden_alpha=2.)
    st = SphereTerms(m, g, p, r, skin, pts)
    n, npt = st.n, st.npt; n3, v3 = 3*n, 3*npt
    th = torch.zeros(n, 9, dtype=f64, device=dev)
    X, Y = st.X, st.Y
    sc = PatchSprings(g, r, skin, float(m.mu_ref), st.V)
    mid = MidpointTerms(m, sc, g)
    KA = dense_hessian(lambda z: mid.energy(th, X+z.view(-1, 3), smooth_polar=True)[0], n3)
    H = dense_hessian(lambda z: st.energy_xy(th, X+z[:n3].view(-1, 3), Y+z[n3:].view(-1, 3)), n3+v3)
    Kxx, Kxy, Kyy = H[:n3, :n3], H[:n3, n3:], H[n3:, n3:]
    G = torch.linalg.solve(Kyy, Kxy.T)                                  # skin response to body motion: y = -G x
    KC = Kxx-Kxy@G; KC = .5*(KC+KC.T)
    print('stiffness matrices: A %d, sphere %d (%.0f s)' % (n3, n3+v3, time.time()-t0), flush=True)
    Ux, nx = interpolate(F_, V_fem, X.cpu().numpy()); Uy, ny = interpolate(F_, V_fem, Y.cpu().numpy())
    print('FEM displacements interpolated: bodies outside the voxel mesh %d of %d, skin points %d of %d (nearest node)' % (nx, n, ny, npt), flush=True)
    mass = m.mass.to(f64)
    W = mass.cpu().numpy()
    rows = []
    print('mode | f_FEM Hz | mass ratio | stiffness / FEM: A, B (skin on FEM), C (skin free) | Rayleigh frequency / FEM: A, B, C | C skin-deviation share')
    for k in range(len(f_fem)):
        om2 = (2*np.pi*f_fem[k])**2
        px = torch.as_tensor(Ux[:, :, k].ravel(), dtype=f64, device=dev); py = torch.as_tensor(Uy[:, :, k].ravel(), dtype=f64, device=dev)
        mr = float((mass[:, None]*px.view(-1, 3)**2).sum())
        sA = float(px@KA@px)/om2
        z = torch.cat((px, py)); sB = float(z@H@z)/om2
        sC = float(px@KC@px)/om2
        yC = -(G@px)
        eps = 1e-6/float(px.abs().max())
        T = {kk: float(v)/(.5*eps*eps) for kk, v in st.energy_terms(th, X+eps*px.view(-1, 3), Y+eps*yC.view(-1, 3)).items()}
        share = T['outer_skin']/sum(T.values())
        rows.append(dict(mode=k, f_fem=float(f_fem[k]), mass=mr, stiff=dict(A=sA, B=sB, C=sC), freq=dict(A=np.sqrt(sA/mr), B=np.sqrt(sB/mr), C=np.sqrt(sC/mr)),
                         terms_C={kk: v/om2 for kk, v in T.items()}))
        print('%2d | %7.2f | %.3f | %.3f %.3f %.3f | %.3f %.3f %.3f | %.2e' % (k, f_fem[k], mr, sA, sB, sC, rows[-1]['freq']['A'], rows[-1]['freq']['B'], rows[-1]['freq']['C'], share), flush=True)
    med = lambda key, s: float(np.median([r_[key][s] for r_ in rows]))
    print('median Rayleigh frequency / FEM: A %.4f, B %.4f, C %.4f; stiffness A %.4f, B %.4f, C %.4f; mass %.4f' % (
        med('freq', 'A'), med('freq', 'B'), med('freq', 'C'), med('stiff', 'A'), med('stiff', 'B'), med('stiff', 'C'), np.median([r_['mass'] for r_ in rows])), flush=True)
    own = {}
    for tag, K in (('A', KA), ('C', KC)):
        rig, fz, Z = own_modes(K, mass, len(f_fem)+6)
        pair = mac_pairs(Z, Ux, W)
        own[tag] = [(k, fz[pair[k][0]]/f_fem[k], pair[k][1]) for k in range(len(f_fem))]
        good = [x for x in own[tag] if x[2] >= .8]
        print('own modes %s: rigid eigenvalues / largest elastic %s | paired (MAC >= 0.8) %d of %d, frequency / FEM median %.4f (%.4f..%.4f)' % (
            tag, np.array2string(np.abs(rig)/fz.max()**2/(4*np.pi**2), precision=1), len(good), len(f_fem), np.median([x[1] for x in good]),
            min(x[1] for x in good), max(x[1] for x in good)), flush=True)
    print('per mode (own modes, MAC): ' + '; '.join('%d: A %.3f (%.2f) C %.3f (%.2f)' % (k, own['A'][k][1], own['A'][k][2], own['C'][k][1], own['C'][k][2]) for k in range(len(f_fem))), flush=True)
    (args.part/'part_modes.json').write_text(json.dumps(dict(rows=rows, own={k: [list(map(float, x)) for x in v] for k, v in own.items()}), indent=1))


if __name__ == '__main__':
    main()
