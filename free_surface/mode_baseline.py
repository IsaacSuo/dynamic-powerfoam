"""Baseline for the continuous-surface work (2026-10-22): the current clean model (walls, completion springs) on the FEM's
own mode shapes, per geometry. Each low FEM mode (voxel FEM, M-normalised: phi^T K phi = (2 pi f)^2) interpolated to the
body centres is the body displacement; the model's quadratic form per term (bonds, volume incl. variance, completion;
rotations slaved by Newton polar), from energies at X + eps phi, over the FEM's; lumped mass ratio sum m |phi|^2; Rayleigh
frequency ratio sqrt(stiffness / mass). Also the geometry of the surface cells: site-to-face distances h of the air
faces (h <= 0: the site outside its own face) and the completion springs left out for it."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.dem_fem_mode_match import interpolate
from experiments.foam_surface_completion import SurfaceCompletion, matrix_quat, polar_newton
from experiments.pacnerf_dem_forward import build
from prototypes.kb_derivation.clean_model import CleanTerms

f64, dev = torch.float64, 'cuda'


def run(name, dem_z, dem_e, run_dir, fem_path, modes, cls=CleanTerms):
    t0 = time.time()
    E, nu = 23008.38, .3008
    g = dict(np.load(dem_z/'geometry.npz'))
    m = build(g, E, nu, dev, 'noslip', .25, .3, f64, kinematics='exact', nodalvol=True, nonlinear=True, ogden_alpha=2.)
    sc = SurfaceCompletion(run_dir, dem_z, m, verbose=False)
    ct = cls(m, sc, g)
    n = m.n; X = m.X
    et = torch.linalg.eigvalsh(ct.M+ct.Mair)/ct.V[:, None]
    print('[%s] gradient moment (M + M_air) / V: smallest eigenvalue median %.3f, worst five %s' % (name, float(et[:, 0].median()), np.round(torch.sort(et[:, 0])[0][:5].cpu().numpy(), 4).tolist()), flush=True)
    nair = len(sc.air_faces['body'])
    L_ = sc.vs_lever[:nair].norm(dim=1)
    h_ = sc.vs_k[:nair]*L_**2/(2*float(m.mu_ref)*torch.as_tensor(sc.air_faces['area'], dtype=f64, device=dev))
    surf = torch.unique(sc.vs_body)
    print('[%s] bodies %d, surface bodies %d, air faces %d (completion springs with h = 0, i.e. site outside the face: %d), exposed patches %d; setup %.0f s' % (
        name, n, len(surf), nair, int((h_ <= 0).sum()), len(sc.vs_k)-nair, time.time()-t0), flush=True)
    th0 = torch.zeros(n, 9, dtype=f64, device=dev)
    def terms(x):
        F = ct.F(x); R = polar_newton(F); q = matrix_quat(R)
        nodal = m._nodalvol(th0, x)[0]
        return dict(bonds=m.energy(th0, x, q)-nodal, volume=ct.volume_energy(x, F, R, m.k_v), completion=ct.completion_energy(F, R, th0))
    Aff = torch.eye(3, dtype=f64, device=dev)+.05*torch.randn(3, 3, generator=torch.Generator().manual_seed(0), dtype=f64).to(dev)
    print('[%s] checks: max |F(X) - I| %.1e, affine map max |F - A| %.1e, rest energy %.1e' % (name, float((ct.F(X)-torch.eye(3, dtype=f64, device=dev)).abs().max()),
          float((ct.F(X@Aff.T)-Aff).abs().max()), float(sum(terms(X).values()))), flush=True)

    cache = np.load(fem_path)
    f_fem, V = cache['f'], cache['V']
    fem = dict(nodes=cache['nodes'], node_grid=cache['node_grid'], lo=cache['lo'], h=float(cache['h']), masses=cache['masses'])
    Pe = np.load(dem_e/'geometry.npz')['points']; Pz = X.cpu().numpy()
    U_, _, Vt_ = np.linalg.svd((Pe-Pe.mean(0)).T@(Pz-Pz.mean(0))); Rz = Vt_.T@U_.T
    Uq, _ = interpolate(fem, V, Pe)
    rows = []
    print('[%s] mode | f_FEM Hz | stiffness / FEM (bonds + volume + completion) | lumped mass ratio | frequency / FEM' % name, flush=True)
    for k in range(min(modes, V.shape[1])):
        phi = torch.as_tensor(Uq[:, :, k]@Rz.T, dtype=f64, device=dev)
        eps = 1e-6/float(phi.abs().max())
        T = {k_: float(v_)/(.5*eps*eps) for k_, v_ in terms(X+eps*phi).items()}
        kf = (2*np.pi*f_fem[k])**2
        st = sum(T.values())/kf; mass = float((m.mass[:, None]*phi**2).sum())
        r = dict(mode=k, f_fem=float(f_fem[k]), stiffness=st, parts={k_: v_/kf for k_, v_ in T.items()}, mass_ratio=mass, freq=float(np.sqrt(st/mass)))
        rows.append(r)
        print('[%s] %2d | %.3f | %.3f (%.3f + %.3f + %.3f) | %.3f | %.3f' % (name, k, r['f_fem'], st, r['parts']['bonds'], r['parts']['volume'],
                                                                        r['parts']['completion'], mass, r['freq']), flush=True)
    print('[%s] median: stiffness %.3f, mass %.3f, frequency %.3f  (%.0f s)' % (name, np.median([r['stiffness'] for r in rows]),
          np.median([r['mass_ratio'] for r in rows]), np.median([r['freq'] for r in rows]), time.time()-t0), flush=True)
    return dict(bodies=n, surface_bodies=int(len(surf)), air_faces=nair, faces_site_outside=int((h_ <= 0).sum()), rows=rows)


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--modes', type=int, default=12)
    a.add_argument('--out', type=Path, required=True)
    args = a.parse_args()
    M = Path('/home/fangsuo/py/dynamic-powerfoam/output'); W = Path('output')
    from free_surface.grad_weights import MidpointTerms
    old = (M/'pacnerf_dem8k_hull/elastic_0_zup', M/'pacnerf_dem8k_hull/elastic_0', M/'pacnerf_elastic0_spacing8k', M/'pacnerf_eval/fem_modes_hull0_h0.008.npz')
    new = (W/'pacnerf_dem_cutft/elastic_0_zup', W/'pacnerf_dem_cutft/elastic_0', W/'pacnerf_elastic0_f0_transparent_cutft', W/'pacnerf_dem_cutft/elastic_0/fem_modes_h0.008.npz')
    res = {}
    for key, geo, cls in (('old_h', old, CleanTerms), ('old_mid', old, MidpointTerms), ('new_mid', new, MidpointTerms)):
        res[key] = run(key, *geo, args.modes, cls=cls)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
