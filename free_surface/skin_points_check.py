"""Contact at held points of the skin's own surface (2026-10-23, skin_warp.k_cpts, WarpSkinSolver.refresh_points; ground
y >= 0, each region's lowest point on y(m) = x_B + F_B (p + r m - X_B) + sum_s W_s(m) delta_s):
  (1) SkinSurface on a subset of regions against the full one (same lowest points);
  (2) the lowest points under a rough state (bodies and skin moved 2 mm at random, object lowered onto the ground)
      against dense sampling of the deformed surface (a cap around each region, its inside points);
  (3) the held-point gaps on Warp against the torch surface's;
  (4) the contact term's gradient against central differences;
  (5) the object at rest lowered --push below the ground, the skin solved under the conditions (bodies held): passes,
      gaps, contact forces, force balance; then the fastest body mode with the skin free there (omega_state.omega_condensed):
      bounded by the material, as at rest?
  python free_surface/skin_points_check.py --out output/round01_pruned_exact"""
import math

import numpy as np
import torch

from free_surface.skin_contact import SkinSurface
from free_surface.skin_contact_check import f64, dev
from free_surface.skin_dynamics_check import build_model, parser
from free_surface.omega_state import omega_condensed


def sampled_lowest(S, st, x, y, n, c0, regions, samples, gen):
    """the lowest sampled point of the deformed surface of each region (global ids; sites with one region): a
    single-region SkinSurface, directions uniform in a cap around the region, kept where exposed"""
    out = []
    for qg in regions.tolist():
        s1 = SkinSurface(st, S.surf_all.R, S.p_all, S.r_all, S.boc_all, S.interp, sl=torch.tensor([qg], device=dev))
        vb = s1.vb[0]
        a0, a1 = s1.a0[0][vb], s1.a1[0][vb]
        th = torch.stack((a0, a1, .5*(a0+a1)), 1)
        Pt = s1.cc[0][vb][:, None]+s1.rho[0][vb][:, None, None]*(torch.cos(th)[..., None]*s1.u[0][vb][:, None]+torch.sin(th)[..., None]*s1.v[0][vb][:, None])
        dirs = ((Pt-s1.p[0])/s1.r[0]).reshape(-1, 3)
        ax = torch.nn.functional.normalize(dirs.mean(0), dim=0)
        cmax = min(math.pi, float(torch.acos((dirs@ax).clamp(-1, 1)).max())+.05)
        z = 1-torch.rand(samples, generator=gen, dtype=f64, device=dev)*(1-math.cos(cmax))
        ph = 2*math.pi*torch.rand(samples, generator=gen, dtype=f64, device=dev)
        ref = torch.tensor([1., 0, 0], dtype=f64, device=dev) if abs(float(ax[0])) < .9 else torch.tensor([0, 1., 0], dtype=f64, device=dev)
        e1 = torch.nn.functional.normalize(torch.linalg.cross(ax, ref), dim=0); e2 = torch.linalg.cross(ax, e1)
        sz = (1-z*z).clamp(min=0).sqrt()
        m = z[:, None]*ax+sz[:, None]*(torch.cos(ph)[:, None]*e1+torch.sin(ph)[:, None]*e2)
        m = m[s1.exposed(m)]
        FB, base, dl = s1.fields(x, y)
        best = float('inf')
        for k0 in range(0, len(m), 256):
            mm = m[k0:k0+256]
            Wm = s1.weights(mm[None])[0]
            h = base[0]+s1.r[0]*mm@FB[0].T+Wm@dl[0]
            best = min(best, float((h@n).min())-c0)
        out.append(best)
    return torch.tensor(out, dtype=f64, device=dev)


def main():
    a = parser()
    a.add_argument('--push', type=float, default=5e-4)
    a.add_argument('--samples', type=int, default=3000)
    a.add_argument('--nreg', type=int, default=40)
    args = a.parse_args()
    b = build_model(args, contact=dict(normal=(0., 1., 0.), c0=0., points=True))
    st, W, S, prec = b['st'], b['W'], b['S'], b['prec']
    X, Y = st.X, st.Y
    n = torch.tensor([0., 1., 0.], dtype=f64, device=dev)
    gen = torch.Generator(device=dev).manual_seed(0)
    surf = S.surf_all
    # a rough state on the ground: bodies and skin moved 2 mm at random, lowered so that the lowest affine patch touches
    W.grad(X, Y); low0 = float(W.tgap_cell.min())
    x1 = X+2e-3*torch.randn(X.shape, generator=gen, dtype=f64, device=dev)-low0*n
    y1 = Y+2e-3*torch.randn(Y.shape, generator=gen, dtype=f64, device=dev)-low0*n
    nh, nn, ng_ = S.refresh_points(x1, y1, prec)
    su = S.surf_u
    print('held regions %d (new %d, their lowest gap %.2e m)' % (nh, nn, ng_), flush=True)
    # (1) subset vs full
    lowS = su.lowest(x1, y1, n, 0.)
    lowF = surf.lowest(x1, y1, n, 0.)
    qi = su.sl
    print('(1) subset vs all regions: |gap diff| max %.2e m, |m diff| max %.2e' % (
        float((lowS['gap']-lowF['gap'][qi]).abs().max()), float((lowS['m']-lowF['m'][qi]).norm(dim=1).max())), flush=True)
    # (2) against sampling, on the held regions nearest the ground
    single = torch.bincount(su.gsite, minlength=int(su.gsite.max())+1)[su.gsite] == 1
    order = torch.argsort(torch.where(single, lowS['gap'], torch.full_like(lowS['gap'], float('inf'))))[:args.nreg]
    gs = sampled_lowest(S, st, x1, y1, n, 0., su.sl[order], args.samples, gen)
    d = gs-lowS['gap'][order]
    print('(2) %d regions nearest the ground: sampled lowest - computed: min %.2e m (negative: a sample lower), median %.2e, max %.2e; arcs %d of them' % (
        len(order), float(d.min()), float(d.median()), float(d.max()), int(lowS['kind'][order].sum())), flush=True)
    # (3) Warp's held-point gaps
    W.grad(x1, y1)
    print('(3) Warp vs torch held-point gaps: |diff| max %.2e m' % float((W.tgap[qi]-lowS['gap']).abs().max()), flush=True)
    # (4) gradient of the contact term: lowered 2 mm more, random multipliers on the held regions
    xd, yd = x1-.002*n, y1-.002*n
    S.refresh_points(xd, yd, prec)
    k = W.cp['k']
    W.tlam.copy_(torch.rand(k, generator=gen, dtype=f64, device=dev)*W.trho*.002*(W.tc['cand'] > 0))
    _, gxc, gyc = W.energy_grad(xd, yd)
    lam_keep = W.tlam.clone(); W.tlam.zero_()
    _, gx0, gy0 = W.energy_grad(xd+n, yd+n)
    W.tlam.copy_(lam_keep)
    Gx, Gy = gxc-gx0, gyc-gy0
    for j in range(3):
        dx = torch.randn(X.shape, generator=gen, dtype=f64, device=dev); dy = torch.randn(Y.shape, generator=gen, dtype=f64, device=dev)
        h = 1e-7
        fd = (float(W.energy(xd+h*dx, yd+h*dy)[6])-float(W.energy(xd-h*dx, yd-h*dy)[6]))/(2*h); an = float((Gx*dx).sum()+(Gy*dy).sum())
        print('(4) direction %d: contact energy %.3e; d/ds differences %.6e, adjoint %.6e, relative %.1e' % (j, float(W.energy(xd, yd)[6]), fd, an, abs(fd-an)/abs(an)), flush=True)
    # (5) static push from rest
    W.tlam.zero_()
    x3, y3 = X-(low0+args.push)*n, Y-(low0+args.push)*n
    S.refresh_points(x3, y3, prec)
    ref = float((9.8*b['mass']).norm())
    for rep in range(3):                                                            # held points refreshed between solves
        y3, s3 = S.solve_contact(x3, y3, prec, ref, tol=1e-6, maxit=600, al_max=100)
        nh, nn, ng_ = S.refresh_points(x3, y3, prec)
    gx, gp, lam = s3['aux'][0], s3['aux'][2], s3['lam']
    act = lam > 0
    print('(5) solved: passes %d, products %d, residual %.1e; held %d, active %d, lowest gap %.2e m, active max |g| %.2e m; '
          'contact force sum %.6e, sum of -dE/dx (y) %.6e, (x, z) %.1e %.1e' % (
              s3['al'], s3['products'], s3['residual'], nh, int(act.sum()), float(gp[W.tc['cand'] > 0].min()), float(gp[act].abs().max()) if bool(act.any()) else 0.,
              float(lam.sum()), float(-gx[:, 1].sum()), float(-gx[:, 0].sum()), float(-gx[:, 2].sum())), flush=True)
    om, B, share, cg = omega_condensed(W, S, prec, b['mass'], x3, y3)
    print('(5) fastest body mode, skin free, in contact: %.3e rad/s (omega dt at 1/4800: %.2f); %.2f on body %d (y %.4f m); CG per iteration %d' % (
        om, om/4800, share, B, float(x3[B, 1]), cg), flush=True)


if __name__ == '__main__':
    main()
