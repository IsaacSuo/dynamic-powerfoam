"""Contact on Warp (2026-10-23, skin_warp.k_contact; ground y >= c0, every surface cell's exposed sphere carried by its
affine field):
  (1) lowest points: at rest and under a rough deformation (bodies and skin moved 5 mm at random), each cell's gap
      against dense sampling of its exposed sphere (a cap around the patch, kept where outside every neighbour's sphere):
      no sample lower than the computed point, and how close the best sample gets;
  (2) the contact term's gradient against central differences of its energy (object lowered into the ground, random
      multipliers): its share of the gradient is the gradient there minus the gradient with the object lifted clear;
  (3) the object at rest, lowered so that its lowest point is --push below the ground, the skin solved under the
      contact conditions (bodies held): passes, products, the gaps left, the contact forces, and the force balance
      sum over the bodies of -dE/dx = sum of the contact forces along the normal.
  python free_surface/skin_contact_warp_check.py --out output/round01_pruned_exact"""
import math

import numpy as np
import torch

from free_surface.skin_contact_check import f64, dev
from free_surface.skin_dynamics_check import build_model, parser


def lowest_sampled(W, F, x, cells, samples, gen, st_y=None):
    """the lowest sampled exposed point of each cell in cells (indices into the contact cells): gap and the margin of
    the sampled cap"""
    ct, st = W.cg, W.st
    out = []
    ng = torch.tensor([0., 1., 0.], dtype=f64, device=dev)
    for c in cells.tolist():
        a0, a1 = int(ct['aoff'][c]), int(ct['aoff'][c+1])
        p, r, B = ct['p'][c], ct['r'][c], int(ct['body'][c])
        th = torch.stack((ct['t0'][a0:a1], ct['t1'][a0:a1], .5*(ct['t0'][a0:a1]+ct['t1'][a0:a1])), 1)          # ends, middles
        P = ct['cc'][a0:a1, None]+ct['rho'][a0:a1, None, None]*(torch.cos(th)[..., None]*ct['u'][a0:a1, None]+torch.sin(th)[..., None]*ct['v'][a0:a1, None])
        dirs = ((P-p)/r).reshape(-1, 3)
        ax = torch.nn.functional.normalize(dirs.mean(0), dim=0)
        cmax = min(math.pi, float(torch.acos((dirs@ax).clamp(-1, 1)).max())+.1)
        # uniform in the cap of half-angle cmax around ax
        z = 1-torch.rand(samples, generator=gen, dtype=f64, device=dev)*(1-math.cos(cmax))
        ph = 2*math.pi*torch.rand(samples, generator=gen, dtype=f64, device=dev)
        e1 = torch.nn.functional.normalize(torch.linalg.cross(ax, torch.tensor([1., 0, 0], dtype=f64, device=dev) if abs(float(ax[0])) < .9 else torch.tensor([0, 1., 0], dtype=f64, device=dev)), dim=0)
        e2 = torch.linalg.cross(ax, e1)
        sz = (1-z*z).clamp(min=0).sqrt()
        m = z[:, None]*ax+sz[:, None]*(torch.cos(ph)[:, None]*e1+torch.sin(ph)[:, None]*e2)
        Xs = p+r*m
        n0, n1 = int(ct['noff'][c]), int(ct['noff'][c+1])
        pw = ((Xs[:, None]-ct['nbp'][n0:n1])**2).sum(-1)-ct['nbr'][n0:n1]**2
        Xs = Xs[(pw > 0).all(1)]
        h = (x[B]+(Xs-st.X[B])@F[B].T)@ng-float(W.tc0[0])
        if W.c_w is not None:                                                       # the skin's mean deviation
            e = torch.nonzero(st.e_body == B)[:, 0]
            de = st_y[st.e_pid[e]]-x[B]-(st.Y[st.e_pid[e]]-st.X[B])@F[B].T
            h = h+float((W.c_w[e][:, None]*de).sum(0)@ng)
        out.append(float(h.min()) if len(h) else float('inf'))
    return torch.tensor(out, dtype=f64, device=dev)


def main():
    a = parser()
    a.add_argument('--push', type=float, default=5e-4)
    a.add_argument('--samples', type=int, default=20000)
    a.add_argument('--ncells', type=int, default=300)
    a.add_argument('--shift', action='store_true')
    args = a.parse_args()
    b = build_model(args, contact=dict(normal=(0., 1., 0.), c0=0., shift=args.shift))
    st, W, S = b['st'], b['W'], b['S']
    X, Y = st.X, st.Y
    gen = torch.Generator(device=dev).manual_seed(0)
    k = W.ct['k']
    print('surface cells %d; arcs per cell median %d, max %d; neighbours per cell max %d; rho median %.3e' % (
        k, int(torch.diff(W.cg['aoff']).median()), int(torch.diff(W.cg['aoff']).max()), int(torch.diff(W.cg['noff']).max()), float(W.trho.median())), flush=True)
    # (1)
    sel = torch.randperm(k, generator=torch.Generator().manual_seed(0))[:args.ncells].to(dev)
    for nm, (x, y) in (('rest', (X, Y)), ('rough', (X+5e-3*torch.randn(X.shape, generator=gen, dtype=f64, device=dev),
                                                   Y+5e-3*torch.randn(Y.shape, generator=gen, dtype=f64, device=dev)))):
        W.grad(x, y)
        gk = W.tgap.clone()
        F = st.F(x, y)
        gs = lowest_sampled(W, F, x, sel, args.samples, gen, y)
        d = gs-gk[sel]
        Fd = float((F-torch.eye(3, dtype=f64, device=dev)).norm(dim=(1, 2)).median())
        print('(1) %-5s (|F - I| median %.2f): %d cells, sampled lowest - computed: min %.2e m (negative: a sample lower), median %.2e m, max %.2e m' % (
            nm, Fd, len(sel), float(d.min()), float(d.median()), float(d.max())), flush=True)
    # (2)
    W.grad(X, Y)
    lift = float(W.tgap.min())
    sh = torch.tensor([0., 1., 0.], dtype=f64, device=dev)
    x1 = X+5e-3*torch.randn(X.shape, generator=gen, dtype=f64, device=dev)*.2
    y1 = Y+5e-3*torch.randn(Y.shape, generator=gen, dtype=f64, device=dev)*.2
    W.grad(x1, y1); low = float(W.tgap.min())
    down = -(low+.002)*sh                                                           # lowest point 2 mm into the ground
    xd, yd = x1+down, y1+down
    W.tlam.copy_(torch.rand(k, generator=gen, dtype=f64, device=dev)*W.trho*.002)
    _, gxc, gyc = W.energy_grad(xd, yd)
    up = (1.-low)*sh
    lam_keep = W.tlam.clone(); W.tlam.zero_()
    _, gx0, gy0 = W.energy_grad(x1+up, y1+up)                                       # clear of the ground: no contact force
    W.tlam.copy_(lam_keep)
    Gx, Gy = gxc-gx0, gyc-gy0
    tsum = W.energy(xd, yd)
    for j in range(3):
        dx = torch.randn(X.shape, generator=gen, dtype=f64, device=dev); dy = torch.randn(Y.shape, generator=gen, dtype=f64, device=dev)
        h = 1e-7
        ep = float(W.energy(xd+h*dx, yd+h*dy)[6]); em = float(W.energy(xd-h*dx, yd-h*dy)[6])
        fd = (ep-em)/(2*h); an = float((Gx*dx).sum()+(Gy*dy).sum())
        print('(2) direction %d: contact energy %.3e, d/ds by differences %.6e, by the adjoint %.6e, relative difference %.1e' % (
            j, float(tsum[6]), fd, an, abs(fd-an)/abs(an)), flush=True)
    # (3)
    W.tlam.zero_()
    d3 = -(lift+args.push)*sh
    x3, y3 = X+d3, Y+d3
    W.grad(x3, y3)
    print('(3) start: gaps below zero %d, lowest %.2e m' % (int((W.tgap < 0).sum()), float(W.tgap.min())), flush=True)
    ref = float(S.grad(x3, y3).norm())
    y3s, s3 = S.solve_contact(x3, y3, b['prec'], ref, tol=1e-6, maxit=600, al_max=200)
    gx = s3['aux'][0]; gp = s3['aux'][2]; lam = s3['lam']
    act = lam > 0
    print('(3) solved: passes %d, products %d, last pass residual %.1e, multiplier change %.1e (tol %.1e); gaps: lowest %.2e m, active %d, '
          'gaps of the active: max |g| %.2e m; contact force sum %.6e, sum over bodies of -dE/dx (y) %.6e, (x, z) %.1e %.1e; skin moved max %.2e m' % (
              s3['al'], s3['products'], s3['residual'], s3['dlam'], 1e-6*ref, float(gp.min()), int(act.sum()), float(gp[act].abs().max()) if int(act.sum()) else 0.,
              float(lam.sum()), float(-gx[:, 1].sum()), float(-gx[:, 0].sum()), float(-gx[:, 2].sum()), float((y3s-y3).norm(dim=1).max())), flush=True)


if __name__ == '__main__':
    main()
