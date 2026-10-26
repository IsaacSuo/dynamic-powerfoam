"""Checks of the sphere-surface model (sphere_model.SphereTerms) on the synthetic part (exact geometry), offline:
  1. rest: F = I, energy and gradient zero; rigid motion (40 deg + shift): energy zero;
  2. affine maps (random, amplitude 1e-3 and 0.2; linear bonds): F_B exact, energy against the continuum
     mu V |U - I|^2 + lam / 2 V (ln J)^2 (exact expected with one body per cell: the bonds' A L n n and the patches'
     r N add up to V I); the terms' shares;
  3. patch test: skin points held on an affine map, bodies free: residual forces on the bodies (interior / surface);
  4. Hessian at rest (dense, bodies + skin): zero eigenvalues of the whole (6 rigid expected) and of the skin block.
Run on dem_exact_single (one body per cell) and dem_exact (merged bodies)."""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from experiments.pacnerf_dem_forward import build
from free_surface.boundary_skin import BoundaryTerms
from free_surface.sphere_model import SphereTerms

f64, dev = torch.float64, 'cuda'


def rot(deg, axis):
    k = np.asarray(axis, float); k /= np.linalg.norm(k); t = np.radians(deg)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return torch.as_tensor(np.eye(3)+np.sin(t)*K+(1-np.cos(t))*K@K, dtype=f64, device=dev)


def grads(st, th, x, y):
    x = x.clone().requires_grad_(True); y = y.clone().requires_grad_(True)
    E = st.energy_xy(th, x, y)
    gx, gy = torch.autograd.grad(E, (x, y))
    return float(E.detach()), gx, gy


def continuum(m, V, Fa):
    I = torch.eye(3, dtype=f64, device=dev)
    w, Q = torch.linalg.eigh(Fa.T@Fa); U = Q@torch.diag(w.sqrt())@Q.T
    return float(m.mu_ref)*V*float(((U-I)**2).sum())+.5*float(m.k_v)*V*float(torch.log(torch.linalg.det(Fa)))**2


def dense_hessian(f, n, chunk=64):
    z0 = torch.zeros(n, dtype=f64, device=dev)
    g = torch.func.grad(f)
    hvp = lambda v: torch.func.jvp(g, (z0,), (v,))[1]
    H = torch.zeros(n, n, dtype=f64, device=dev); eye = torch.eye(n, dtype=f64, device=dev)
    for s in range(0, n, chunk):
        H[s:s+chunk] = torch.func.vmap(hvp)(eye[s:s+chunk])
    return .5*(H+H.T)


def run(tag, part, dem, hessian, skin_kind='fan', skin_file='boundary_skin.npz', lever='point'):
    t0 = time.time()
    stt = torch.load(part/'run'/'model.pt', map_location='cpu', weights_only=True)
    p = stt['points'].double().numpy(); r = torch.nn.functional.softplus(stt['radii'].double(), beta=100).numpy()
    g = dict(np.load(part/dem/'geometry.npz'))
    skin = dict(np.load(part/'sphere_skin.npz')); pts = dict(np.load(part/'sphere_points.npz'))
    m = build(g, 23008.38, .3008, dev, 'noslip', .25, .3, f64, kinematics='exact', nodalvol=True, nonlinear=False, ogden_alpha=2.)
    st = SphereTerms(m, g, p, r, skin, pts) if skin_kind == 'fan' else BoundaryTerms(m, g, p, r, dict(np.load(part/skin_file)), lever=lever)
    n, npt = st.n, st.npt
    th = torch.zeros(n, 9, dtype=f64, device=dev)
    X, Y = st.X, st.Y; c = (m.mass[:, None]*X).sum(0)/m.mass.sum(); Vt = float(st.V.sum())
    size = float((X.max(0).values-X.min(0).values).max())
    I = torch.eye(3, dtype=f64, device=dev)
    print('\n[%s] bodies %d, skin points %d, parts %d; volume %.6e; setup %.0f s' % (tag, n, npt, len(st.p_r), Vt, time.time()-t0), flush=True)
    gen = torch.Generator(device='cpu').manual_seed(0)
    Fs = {amp: I+amp*torch.randn(3, 3, generator=gen, dtype=f64).to(dev) for amp in (1e-3, .2)}
    E0, gx0, gy0 = grads(st, th, X, Y)
    _, gxa, gya = grads(st, th, *st.affine(Fs[1e-3], c))
    ga = float(torch.cat((gxa, gya)).norm(dim=1).max()); Es = continuum(m, Vt, Fs[1e-3])
    print('[%s] rest: |F - I| max %.1e, energy %.1e of a 1e-3 strain, |grad| max bodies %.1e skin %.1e (of the largest force of a 1e-3 strain)' % (
        tag, float((st.F(X, Y)-I).abs().max()), E0/Es, float(gx0.norm(dim=1).max())/ga, float(gy0.norm(dim=1).max())/ga), flush=True)
    Q = rot(40, [1, 2, 3]); tr = torch.tensor([.1, .2, -.05], dtype=f64, device=dev)*size
    print('[%s] rigid (40 deg + shift): energy %.1e of a 1e-3 strain' % (tag, float(st.energy_xy(th, *st.affine(Q, c, tr)))/Es), flush=True)
    for amp, Fa in Fs.items():
        x, y = st.affine(Fa, c)
        T = {k: float(v) for k, v in st.energy_terms(th, x, y).items()}
        E = sum(T.values()); Ec = continuum(m, Vt, Fa)
        print('[%s] affine %g: |F_B - F| max %.1e | energy / continuum - 1 = %.1e | shares %s' % (
            tag, amp, float((st.F(x, y)-Fa).abs().max()), E/Ec-1, {k: '%.3e' % (v/E) for k, v in T.items()}), flush=True)
    # patch test
    a = torch.tensor([[.01, .004, 0], [.004, -.006, .003], [0, .003, .008]], dtype=f64, device=dev)
    x, y = st.affine(I+a, c)
    _, gx, _ = grads(st, th, x, y)
    surf = torch.zeros(n, dtype=torch.bool, device=dev); surf[st.c_body] = True
    fs = 23008.38*float(a.norm())*float(st.A.median())
    print('[%s] patch test (skin held affine, strain ~1.6e-2): body residual / (|sigma| x median wall area): interior max %.1e, surface bodies max %.1e (interior bodies %d)' % (
        tag, float(gx[~surf].norm(dim=1).max())/fs if (~surf).any() else 0., float(gx[surf].norm(dim=1).max())/fs, int((~surf).sum())), flush=True)
    if hessian:
        t1 = time.time()
        n3, v3 = 3*n, 3*npt
        H = dense_hessian(lambda z: st.energy_xy(th, X+z[:n3].view(-1, 3), Y+z[n3:].view(-1, 3)), n3+v3)
        ev = torch.linalg.eigvalsh(H); top = float(ev[-1])
        evy = torch.linalg.eigvalsh(H[n3:, n3:]); topy = float(evy[-1])
        print('[%s] Hessian at rest (%d x %d, %.0f s): whole: eigenvalues < 1e-10 of the largest %d (6 rigid expected), 7th %.1e, min %.1e | '
              'skin block: smallest %.1e of its largest, below 1e-10: %d' % (tag, n3+v3, n3+v3, time.time()-t1, int((ev < 1e-10*top).sum()), float(ev[6])/top,
                                                                           float(ev[0])/top, float(evy[0])/topy, int((evy < 1e-10*topy).sum())), flush=True)


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--part', type=Path, default=Path('output/test_part'))
    a.add_argument('--no-hessian', action='store_true')
    a.add_argument('--skin', choices=('fan', 'boundary'), default='fan', help='fan: sphere_points.npz; boundary: --skin-file')
    a.add_argument('--skin-file', default='boundary_skin.npz')
    a.add_argument('--lever', choices=('point', 'c'), default='point')
    args = a.parse_args()
    run('single', args.part, 'dem_exact_single', not args.no_hessian, args.skin, args.skin_file, args.lever)
    run('merged', args.part, 'dem_exact', False, args.skin, args.skin_file, args.lever)


if __name__ == '__main__':
    main()
