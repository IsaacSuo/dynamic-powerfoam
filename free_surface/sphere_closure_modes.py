"""Checks 2-3 of the skinless sphere model (sphere_closure.py) on the part's FEM modes (2026-10-26), against the skin
model's numbers in output/test_part/part_diag_gamma_sym_skin.log (not rerun):
  stiffness at rest, exactly: forces and stresses vanish there (dE/dx = dE/dF = 0), so K = [I; G]^T H [I; G], H the
  Hessian of the energy in (x, F_B) with F_B independent, G = dF_B / dx the closure's linearisation (per body
  J^-1 dS_B / dx); checked against central differences of the true gradient
  1. stiffness along the 20 FEM modes, u^T K u / omega^2 (skin model: median 1.069)
  2. own modes with consistent mass (part_diag: the cell's affine motion u_B + G_B (X - X_B), grid-sampled moments),
     paired by MAC >= 0.8 (skin model: 16 of 20, median 1.026), and with the masses at the bodies; the highest
     frequency with the masses at the bodies
  PYTHONPATH=<main repo>:<worktree> python free_surface/sphere_closure_modes.py"""
import argparse
from pathlib import Path

import numpy as np
import torch

from experiments.dem_fem_mode_match import interpolate
from experiments.pacnerf_dem_forward import build
from free_surface.part_diag import cell_moments
from free_surface.part_modes import dense_hessian, mac_pairs
from free_surface.sphere_closure import ClosureTerms

f64, dev = torch.float64, 'cuda'


def closure_stiffness(new, th):
    """the skinless model's stiffness at rest, exactly: K = [I; G]^T H [I; G] (see above); returns K and G"""
    n = new.n; n3, n9 = 3*n, 9*n
    X = new.X; I3 = torch.eye(3, dtype=f64, device=dev)
    S0 = new.walls(X)
    F0, _ = new.closure(S0)
    J = new.jacobian(F0.detach(), S0, new.kap0)
    W = torch.func.jacfwd(new.walls)(X).reshape(n, 9, n3)
    G = torch.linalg.solve(J, W).reshape(n9, n3)
    print('rest: |F_B - I| %.1e; closure Jacobians: condition number median %.1f, max %.1f' % (
        float((F0-I3).abs().max()), float(torch.linalg.cond(J).median()), float(torch.linalg.cond(J).max())), flush=True)
    H = dense_hessian(lambda z: sum(new.energy_F(th, X+z[:n3].view(-1, 3), I3+z[n3:].view(-1, 3, 3)).values()), n3+n9)
    Hxx, Hxf, Hff = H[:n3, :n3], H[:n3, n3:], H[n3:, n3:]
    K = Hxx+Hxf@G+G.T@Hxf.T+G.T@Hff@G
    return .5*(K+K.T), G


def consistent_mass(grad_u, mass, dT, JT):
    """part_diag's consistent mass: each cell moving as u_B + G_B (X - X_B), G_B = grad_u(u) (n, 3, 3), linear in u"""
    def T(v):
        u_ = v.view(-1, 3); Gr = grad_u(v)
        return .5*((mass*(u_*u_).sum(1)).sum()+2*(u_*torch.einsum('bij,bj->bi', Gr, dT)).sum()+torch.einsum('bij,bjk,bik->', Gr, JT, Gr))
    M = dense_hessian(T, 3*len(mass))
    return .5*(M+M.T)


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument('--part', type=Path, default=Path('output/test_part'))
    a_.add_argument('--fem-h', type=float, default=.0015)
    args = a_.parse_args()
    torch.manual_seed(0)
    stt = torch.load(args.part/'run'/'model.pt', map_location='cpu', weights_only=True)
    p = stt['points'].double().numpy(); r = torch.nn.functional.softplus(stt['radii'].double(), beta=100).numpy()
    c_ = np.load(args.part/('fem_modes_h%g.npz' % args.fem_h))
    f_fem, V_fem = c_['f'][:20], c_['V'][:, :20]
    F_ = dict(nodes=c_['nodes'], node_grid=c_['node_grid'], lo=c_['lo'], h=float(c_['h']), masses=c_['masses'])
    E, nu = 23008.38, .3008
    g = dict(np.load(args.part/'dem_exact_single'/'geometry.npz'))
    m = build(g, E, nu, dev, 'noslip', .25, .3, f64, kinematics='exact', nodalvol=True, nonlinear=False, ogden_alpha=2.)
    new = ClosureTerms(m, g, p, r, dict(np.load(args.part/'boundary_skin.npz')))
    n = new.n; n3, n9 = 3*n, 9*n
    th = torch.zeros(n, 9, dtype=f64, device=dev)
    X = new.X; I3 = torch.eye(3, dtype=f64, device=dev)

    K, G = closure_stiffness(new, th)

    def grad(x):
        x = x.detach().clone().requires_grad_(True)
        return torch.autograd.grad(new.energy_x(th, x), x)[0]
    u = torch.randn(n, 3, dtype=f64, device=dev)
    for h in (1e-6, 1e-7):
        gd = (grad(X+h*u)-grad(X-h*u))/(2*h)
        Ku = (K@u.reshape(-1)).view(n, 3)
        print('K u against central differences of the gradient (step %.0e m): relative difference %.1e' % (h, float((gd-Ku).norm()/Ku.norm())), flush=True)
    ev = torch.linalg.eigvalsh(K)
    print('K eigenvalues / largest: six smallest %s, seventh %.1e' % (np.array2string((ev[:6]/ev[-1]).cpu().numpy(), precision=1), float(ev[6]/ev[-1])), flush=True)

    # 1. stiffness along the FEM modes
    Ux, _ = interpolate(F_, V_fem, X.cpu().numpy())
    mass = m.mass.to(f64); W_ = mass.cpu().numpy()
    s = []
    for k in range(len(f_fem)):
        px = torch.as_tensor(Ux[:, :, k].ravel(), dtype=f64, device=dev)
        s.append(float(px@K@px)/(2*np.pi*f_fem[k])**2)
    s = np.array(s)
    print('1. stiffness along the FEM modes / FEM: median %.4f (range %.3f..%.3f; skin model 1.0689); per mode %s' % (
        np.median(s), s.min(), s.max(), ' '.join('%d:%.3f' % (k, v) for k, v in enumerate(s))), flush=True)

    # 2. own modes
    d, Jc, _ = cell_moments(p, r, X.cpu().numpy(), new.V.cpu().numpy())
    Mc = consistent_mass(lambda v: (G@v).view(n, 3, 3), mass, torch.as_tensor(d, device=dev), torch.as_tensor(Jc, device=dev))
    Ml = torch.diag(mass.repeat_interleave(3))
    for tag, M_ in (('consistent mass', Mc), ('masses at the bodies', Ml)):
        mr = np.median([float(torch.as_tensor(Ux[:, :, k].ravel(), dtype=f64, device=dev)@M_@torch.as_tensor(Ux[:, :, k].ravel(), dtype=f64, device=dev))
                        for k in range(len(f_fem))])
        Li = torch.linalg.inv(torch.linalg.cholesky(M_))
        w, Vv = torch.linalg.eigh(Li@K@Li.T)
        Z = (Li.T@Vv).cpu().numpy(); w = w.cpu().numpy()
        fz = np.sqrt(np.clip(w[6:6+26], 0, None))/(2*np.pi); Z = Z[:, 6:6+26].reshape(-1, 3, 26)
        pair = mac_pairs(Z, Ux, W_)
        good = [(k, fz[pair[k][0]]/f_fem[k]) for k in range(len(f_fem)) if pair[k][1] >= .8]
        wmax = float(np.sqrt(w[-1]))
        print('2. own modes, %s: mass along the FEM modes / FEM median %.4f; rigid %s; paired (MAC >= 0.8) %d of %d, frequency / FEM median %.4f (%.4f..%.4f); '
              'highest omega %.3e rad/s (explicit step limit 2 / omega %.2e s); per mode %s' % (
                  tag, mr, np.array2string(np.abs(w[:6])/w[-1], precision=1), len(good), len(f_fem), np.median([x[1] for x in good]),
                  min(x[1] for x in good), max(x[1] for x in good), wmax, 2/wmax,
                  ' '.join('%d:%.3f(%.2f)' % (k, fz[pair[k][0]]/f_fem[k], pair[k][1]) for k in range(len(f_fem)))), flush=True)


if __name__ == '__main__':
    main()
