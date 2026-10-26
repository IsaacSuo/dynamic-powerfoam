"""Largest vibration frequency of the bodies (skin held fixed: an upper bound of the skin-condensed one) and the stable
time step of symplectic Euler, 2 / omega_max, for a foam built by foam_exact.py (2026-10-23). Power iteration on M^-1 K_xx
(Hessian-vector products of the skin model, lumped masses = cell volumes, energy per unit density). Prints the bodies the
fastest mode sits on and the smallest cell volumes.
  python free_surface/omega_max.py --out output/round01_exact [--skin boundary_skin.npz]"""
import argparse
from pathlib import Path

import numpy as np
import torch

from experiments.pacnerf_dem_forward import build as dem_build
from free_surface.boundary_skin import BoundaryTerms

f64, dev = torch.float64, 'cuda'


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--out', type=Path, default=Path('output/round01_exact'))
    a.add_argument('--skin', default='boundary_skin.npz')
    a.add_argument('--iterations', type=int, default=60)
    a.add_argument('--dt', type=float, default=1/4800, help='the time step to compare with')
    args = a.parse_args()
    cells = np.load(args.out/'dem_exact_single'/'cells.npz'); p, r = cells['points'], cells['radii']
    g = dict(np.load(args.out/'dem_exact_single'/'geometry.npz')); bsk = dict(np.load(args.out/args.skin))
    m = dem_build(g, 23008.38, .3008, dev, 'noslip', .25, .3, f64, kinematics='exact', nodalvol=True, nonlinear=False, ogden_alpha=2.)
    st = BoundaryTerms(m, g, p, r, bsk, lever='c')
    X, Y, n = st.X, st.Y, st.n
    th = torch.zeros(n, 9, dtype=f64, device=dev)
    gx = torch.func.grad(lambda xx: st.energy_xy(th, xx.view(-1, 3), Y))
    H = lambda v: torch.func.jvp(gx, (X.reshape(-1),), (v,))[1]
    minv = (1/m.mass.to(f64)).repeat_interleave(3)
    v = torch.randn(3*n, dtype=f64, device=dev, generator=torch.Generator(device=dev).manual_seed(0))
    for _ in range(args.iterations):
        w = minv*H(v); lam = float(w.norm()/v.norm()); v = w/w.norm()
    om = lam**.5
    top = torch.topk(v.view(n, 3).norm(dim=1)**2, 5)
    V = st.V; med = float(V.median())
    src = cells['source_index']
    print('omega_max %.3e rad/s (skin held); stable dt < 2 / omega_max = %.3e s = 1 / %.0f s; compared with %.3e s: %s' % (
        om, 2/om, om/2, args.dt, 'fine' if 2/om >= args.dt else 'too small by %.1fx' % (args.dt*om/2)))
    print('fastest mode on bodies %s (share %s), model.pt index %s, volume / median %s' % (
        top.indices.tolist(), np.round(top.values.cpu().numpy(), 3).tolist(), src[top.indices.cpu().numpy()].tolist(),
        np.round((V[top.indices]/med).cpu().numpy(), 6).tolist()))
    print('cell volumes / median: smallest %s; counts below 1e-4 / 1e-3 / 1e-2 / 0.1: %s' % (
        np.round(torch.sort(V/med).values[:5].cpu().numpy(), 6).tolist(), [int((V < f*med).sum()) for f in (1e-4, 1e-3, 1e-2, .1)]))


if __name__ == '__main__':
    main()
