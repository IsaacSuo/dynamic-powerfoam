"""Free-surface condition for the outer half-layer (2026-10-22), tested on the FEM's own mode shapes.

Clean model (prototypes/kb_derivation/clean_model.py): F_B = [S_B + Fill_B M_air,B] (M_B + M_air,B)^-1; the air directions
of a surface cell (M_air: its completion springs l l^T k/2mu) are filled with the neighbours' gradient, Fill = F_N.
Free-surface variant: Fill_B = C_B, one 3 x 3 per cell with air directions, at the minimum of the energy (sigma n = 0
on the air side: the outer layer's columns along the air directions take the energy minimum, the directions with walls
stay with the walls). C only acts through C M_air: a cell with one air normal frees one column, a corner cell more.

Test (as mode_energy.py): each low FEM mode (voxel FEM, M-normalised, phi^T K phi = (2 pi f)^2) interpolated to the body
centres is the body displacement; the model's quadratic form phi^T K phi (rest, rotations slaved by Newton polar) over the
FEM's, (a) with the neighbour fill, (b) with C condensed out: q = phi^T K_zz phi - b^T K_cc^-1 b, b = K_cz phi (CG on
K_cc by Hessian-vector products, autograd). Lumped mass ratio sum m |phi|^2 (FEM: 1). Frequency ratio sqrt(stiffness /
mass). Checks: rest energy, a rigid rotation (both forms 0)."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import prototypes.kb_derivation.face_modes as FM                      # polynomial det / cof patch for second derivatives
from experiments.dem_fem_mode_match import interpolate
from experiments.foam_surface_completion import SurfaceCompletion
from experiments.pacnerf_dem_forward import build
from prototypes.kb_derivation.clean_model import CleanTerms

f64, dev = torch.float64, 'cuda'
assert FM is not None


class FreeSurfaceTerms(CleanTerms):
    fill = None                                    # None: the model's neighbour fill F_N; else (n, 3, 3)

    def F(self, x):
        d = x[self.j]-x[self.i]
        on = lambda w, h: (w*self.A*h/self.L)[:, None, None]*d[:, :, None]*self.nrm[:, None, :]
        S = torch.zeros(self.n, 3, 3, dtype=x.dtype, device=x.device).index_add(0, self.i, on(self.wi, self.hi)).index_add(0, self.j, on(self.wj, self.hj))
        fill = self.nbr_sum(S)@self.MN_inv if self.fill is None else self.fill
        return (S+fill@self.Mair)@self.Mt_inv


def cg(Aop, b, it=2000, tol=1e-9):
    x = torch.zeros_like(b); r = b.clone(); p = r.clone(); rr = float(r@r); b2 = rr
    for k in range(it):
        Ap = Aop(p); a = rr/float(p@Ap)
        x = x+a*p; r = r-a*Ap; rr1 = float(r@r)
        if rr1 < tol**2*b2:
            return x, k+1, (rr1/b2)**.5
        p = r+(rr1/rr)*p; rr = rr1
    return x, it, (rr/b2)**.5


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--dem-z', type=Path, default=Path('/home/fangsuo/py/dynamic-powerfoam/output/pacnerf_dem8k_hull/elastic_0_zup'))
    a.add_argument('--dem-e', type=Path, default=Path('/home/fangsuo/py/dynamic-powerfoam/output/pacnerf_dem8k_hull/elastic_0'))
    a.add_argument('--run', type=Path, default=Path('/home/fangsuo/py/dynamic-powerfoam/output/pacnerf_elastic0_spacing8k'))
    a.add_argument('--fem', type=Path, default=Path('/home/fangsuo/py/dynamic-powerfoam/output/pacnerf_eval/fem_modes_hull0_h0.008.npz'))
    a.add_argument('--modes', type=int, default=12)
    a.add_argument('--out', type=Path, required=True)
    args = a.parse_args()
    E, nu = 23008.38, .3008
    g = dict(np.load(args.dem_z/'geometry.npz'))
    m = build(g, E, nu, dev, 'noslip', .25, .3, f64, kinematics='exact', nodalvol=True, nonlinear=True, ogden_alpha=2.)
    sc = SurfaceCompletion(args.run, args.dem_z, m, verbose=False)
    ct = FreeSurfaceTerms(m, sc, g)
    n = ct.n; X = ct.X
    air = torch.nonzero(ct.Mair.abs().sum((1, 2)) > 0).squeeze(1)
    na = len(air)
    rank = torch.linalg.matrix_rank(ct.Mair[air], rtol=1e-6)
    print('bodies %d; with air directions %d (rank of M_air: 1 -> %d, 2 -> %d, 3 -> %d)' % (
        n, na, int((rank == 1).sum()), int((rank == 2).sum()), int((rank == 3).sum())), flush=True)
    th0 = torch.zeros(n, 9, dtype=f64, device=dev)
    eye = torch.eye(3, dtype=f64, device=dev)

    def E_base(z):
        ct.fill = None
        return ct.energy(th0, X+z.view(n, 3), smooth_polar=True)[0]

    def E_fs(z, c):
        fill = eye.expand(n, 3, 3).clone()
        fill = fill.index_add(0, air, c.view(na, 3, 3))
        ct.fill = fill
        e = ct.energy(th0, X+z.view(n, 3), smooth_polar=True)[0]
        ct.fill = None
        return e
    z0 = torch.zeros(3*n, dtype=f64, device=dev); c0 = torch.zeros(9*na, dtype=f64, device=dev)
    gb = torch.func.grad(E_base)
    gz = torch.func.grad(E_fs, argnums=0); gc = torch.func.grad(E_fs, argnums=1)
    Kb = lambda v: torch.func.jvp(gb, (z0,), (v,))[1]
    Kzz = lambda v: torch.func.jvp(lambda z: gz(z, c0), (z0,), (v,))[1]
    Kcz = lambda v: torch.func.jvp(lambda z: gc(z, c0), (z0,), (v,))[1]
    Kcc = lambda v: torch.func.jvp(lambda c: gc(z0, c), (c0,), (v,))[1]
    print('rest: energy model %.2e, free-surface %.2e; |grad_c| %.2e' % (float(E_base(z0)), float(E_fs(z0, c0)), float(gc(z0, c0).norm())), flush=True)
    # rigid rotation check (small angle about z): both forms ~ 0 relative to a stretch
    rot = torch.cross(torch.tensor([0, 0, 1.], dtype=f64, device=dev).expand(n, 3), X-X.mean(0), dim=1).reshape(-1)
    st = ((X-X.mean(0))*torch.tensor([1., 0, 0], dtype=f64, device=dev)).reshape(-1)
    b = Kcz(rot); crot, itr, res = cg(Kcc, -b)
    print('rigid rotation: model %.2e, free-surface %.2e (a uniform 1-axis stretch: %.3e)  [CG %d its, res %.1e]' % (
        float(rot@Kb(rot)), float(rot@Kzz(rot)+b@crot), float(st@Kb(st)), itr, res), flush=True)
    # FEM modes at the body centres (elastic frame -> z-up)
    cache = np.load(args.fem)
    f_fem, V = cache['f'], cache['V']
    fem = dict(nodes=cache['nodes'], node_grid=cache['node_grid'], lo=cache['lo'], h=float(cache['h']), masses=cache['masses'])
    Pe = np.load(args.dem_e/'geometry.npz')['points']; Pz = X.cpu().numpy()
    U_, _, Vt_ = np.linalg.svd((Pe-Pe.mean(0)).T@(Pz-Pz.mean(0))); Rz = Vt_.T@U_.T
    Uq, _ = interpolate(fem, V, Pe)
    rows = []
    print('mode | f_FEM Hz | stiffness / FEM: model, free surface (drop) | lumped mass ratio | frequency / FEM: model, free surface | CG its')
    for k in range(min(args.modes, V.shape[1])):
        t0 = time.time()
        phi = torch.as_tensor(Uq[:, :, k]@Rz.T, dtype=f64, device=dev).reshape(-1)
        kf = (2*np.pi*f_fem[k])**2
        qb = float(phi@Kb(phi))
        b = Kcz(phi); cstar, itr, res = cg(Kcc, -b)
        qf = float(phi@Kzz(phi)+b@cstar)
        mass = float((m.mass[:, None]*phi.view(n, 3)**2).sum())
        r = dict(mode=k, f_fem=float(f_fem[k]), stiff_model=qb/kf, stiff_fs=qf/kf, mass_ratio=mass, freq_model=float(np.sqrt(qb/kf/mass)),
                 freq_fs=float(np.sqrt(max(qf, 0)/kf/mass)), cg_its=itr, cg_res=res, seconds=time.time()-t0)
        rows.append(r)
        print('%2d | %.3f | %.3f, %.3f (%+.1f%%) | %.3f | %.3f, %.3f | %d (res %.0e, %.0f s)' % (
            k, r['f_fem'], r['stiff_model'], r['stiff_fs'], 100*(qf/qb-1), mass, r['freq_model'], r['freq_fs'], itr, res, r['seconds']), flush=True)
    fm = np.median([r['freq_model'] for r in rows]); ff = np.median([r['freq_fs'] for r in rows])
    drop = np.array([r['stiff_fs']/r['stiff_model']-1 for r in rows])
    print('median frequency ratio: model %.3f, free surface %.3f; stiffness change %.1f%% .. %.1f%%' % (fm, ff, 100*drop.min(), 100*drop.max()), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(bodies=n, air_bodies=na, rows=rows), indent=1))


if __name__ == '__main__':
    main()
