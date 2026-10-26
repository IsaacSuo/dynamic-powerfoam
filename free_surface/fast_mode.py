"""What makes the fastest body mode fast (2026-10-23, the pruned foam): the top eigenvector of M^-1 K_xx (skin held,
power iteration), each energy term's curvature along it (second directional derivative, y fixed) as a share of
omega^2 = v^T K v / v^T M v, and the geometry of the body it sits on.
  python free_surface/fast_mode.py --out output/round01_pruned_exact"""
import argparse
from pathlib import Path

import numpy as np
import torch

from experiments.pacnerf_dem_forward import build as dem_build
from free_surface.boundary_skin import BoundaryTerms

f64, dev = torch.float64, 'cuda'


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--out', type=Path, default=Path('output/round01_pruned_exact'))
    a.add_argument('--gamma', default='bonds')
    args = a.parse_args()
    out = args.out
    cells = np.load(out/'dem_exact_single'/'cells.npz'); g = dict(np.load(out/'dem_exact_single'/'geometry.npz'))
    bsk = dict(np.load(out/'boundary_skin.npz'))
    p, r = cells['points'], cells['radii']
    m = dem_build(g, 23008.38, .3008, dev, 'noslip', .25, .3, f64, kinematics='exact', nodalvol=True, nonlinear=False, ogden_alpha=2.)
    st = BoundaryTerms(m, g, p, r, bsk, lever='c', gamma=args.gamma)
    X, Y, n = st.X, st.Y, st.n
    th = torch.zeros(n, 9, dtype=f64, device=dev)
    M = m.mass.to(f64)
    gx = torch.func.grad(lambda xx: st.energy_xy(th, xx.view(-1, 3), Y))
    H = lambda v: torch.func.jvp(gx, (X.reshape(-1),), (v,))[1]
    minv = (1/M).repeat_interleave(3)
    v = torch.randn(3*n, dtype=f64, device=dev, generator=torch.Generator(device=dev).manual_seed(0))
    for _ in range(80):
        w = minv*H(v); v = w/w.norm()
    vM = float((M.repeat_interleave(3)*v*v).sum())
    om2 = float((v*H(v)).sum())/vM
    one = torch.ones((), dtype=f64, device=dev); zero = torch.zeros((), dtype=f64, device=dev)
    parts = {}
    for nm in st.energy_terms(th, X, Y):
        f = lambda s: st.energy_terms(th, X+s*v.view(-1, 3), Y)[nm]
        d1 = lambda s: torch.func.jvp(f, (s,), (one,))[1]
        parts[nm] = float(torch.func.jvp(d1, (zero,), (one,))[1])/vM/om2
    b = int(v.view(n, 3).norm(dim=1).argmax())
    share = float(v.view(n, 3)[b].norm()**2)
    print('omega_max %.3e rad/s (skin held); %.2f of the mode on body %d (model.pt index %d); share of omega^2 by term: %s' % (
        om2**.5, share, b, int(cells['source_index'][b]), ' '.join('%s %.3f' % kv for kv in parts.items())), flush=True)
    # the body's geometry
    pr = g['pairs']; A = g['areas']
    sel = np.flatnonzero((pr[:, 0] == b) | (pr[:, 1] == b))
    oth = np.where(pr[sel, 0] == b, pr[sel, 1], pr[sel, 0])
    L = np.linalg.norm(p[oth]-p[b], axis=1)
    V = st.V.cpu().numpy(); med = np.median(V)
    ne = int((bsk['ent_site'] == b).sum())
    pa = float(np.load(out/'sphere_skin.npz')['patch_area'][b])
    k = np.argsort(np.abs(np.asarray(((p[oth]-p[b])**2).sum(1))-r[oth]**2+r[b]**2))
    outside = bool(((((p[b]-p[oth])**2).sum(1)-r[oth]**2) < -r[b]**2).any())
    Mb = np.zeros((3, 3)); d = (p[oth]-p[b])/L[:, None]
    for a_, L_, d_ in zip(A[sel], L, d):
        Mb += a_*L_/2*np.outer(d_, d_)
    ev = np.linalg.eigvalsh(Mb)
    print('body %d: volume / median %.3f; bonds %d; wall area sum %.2e m^2 (A_sum / V^(2/3) %.1f); sum A / L %.2e m, typical cell (median V^(1/3)) %.2e m; '
          'shortest bond %.2e m (median spacing %.2e); radius %.3e m; skin entries %d, exposed sphere area %.2e m^2; site outside its own cell: %s; '
          'bond moment eigenvalues / largest %s; neighbours\' volumes / median %s' % (
              b, V[b]/med, len(sel), A[sel].sum(), A[sel].sum()/V[b]**(2/3), (A[sel]/L).sum(), med**(1/3), L.min(),
              np.median(np.linalg.norm(p[pr[:, 0]]-p[pr[:, 1]], axis=1)), r[b], ne, pa, outside, np.round(ev/ev.max(), 4).tolist(),
              np.round(np.sort(V[oth]/med), 3).tolist()), flush=True)


def condensed(out=Path('output/round01_pruned_exact'), iters=50, h=1e-7):
    """the fastest mode with the skin free: power iteration on M^-1 K_c, K_c v = K_xx v + K_xy z, K_yy z = -K_yx v (PCG,
    preconditioner: probed K_yy blocks per body, cached in skin_Kyy_rest_csr_<gamma>.npz); products by central differences of the
    Warp gradient; then each term's share of omega^2 along (v, z)"""
    import scipy.sparse as sp
    from free_surface.skin_solve import SkinSolver
    from free_surface.skin_warp import SkinWarp
    cells = np.load(out/'dem_exact_single'/'cells.npz'); g = dict(np.load(out/'dem_exact_single'/'geometry.npz'))
    bsk = dict(np.load(out/'boundary_skin.npz'))
    m = dem_build(g, 23008.38, .3008, dev, 'noslip', .25, .3, f64, kinematics='exact', nodalvol=True, nonlinear=False, ogden_alpha=2.)
    st = BoundaryTerms(m, g, cells['points'], cells['radii'], bsk, lever='c')
    X, Y, n, npt = st.X, st.Y, st.n, st.npt
    S = SkinSolver(st, bsk, g['body_of_cell'])
    kf = out/('skin_Kyy_rest_csr_%s.npz' % st.gamma)
    K = sp.load_npz(kf) if kf.exists() else S.probe_K()
    if not kf.exists():
        sp.save_npz(kf, K)
    prec = S.preconditioner(S.blocks_probed(K, S.bodies), S.bodies)(torch.eye(3, dtype=f64, device=dev).expand(n, 3, 3))
    W = SkinWarp(st); W.prepare()
    def gxy(dx, dy):
        _, ax, ay = W.grad(X+h*dx, Y+h*dy); _, bx, by = W.grad(X-h*dx, Y-h*dy)
        return (ax-bx)/(2*h), (ay-by)/(2*h)
    def Kyy(w):
        return gxy(torch.zeros_like(X), w)[1]
    M = m.mass.to(f64)
    v = torch.randn(n, 3, dtype=f64, device=dev, generator=torch.Generator(device=dev).manual_seed(0))
    zs, cg = [], []
    for it in range(iters):
        v = v/v.abs().max()
        b = -gxy(v, torch.zeros_like(Y))[1]
        z = torch.zeros_like(Y); r_ = b.clone(); q = prec(r_.reshape(1, -1)).view(-1, 3); d = q.clone(); rq = float((r_*q).sum()); bn = float(b.norm()); k = 0
        while float(r_.norm()) > 1e-7*bn and k < 500:
            Ad = Kyy(d/d.abs().max())*d.abs().max(); k += 1
            al = rq/float((d*Ad).sum()); z = z+al*d; r_ = r_-al*Ad
            q = prec(r_.reshape(1, -1)).view(-1, 3); rq1 = float((r_*q).sum()); d = q+(rq1/rq)*d; rq = rq1
        cg.append(k)
        s = max(float(v.abs().max()), float(z.abs().max()))
        Kc = gxy(v/s, z/s)[0]*s
        om2 = float((v*Kc).sum())/float((M[:, None]*v*v).sum())
        v = Kc/M[:, None]
    th = torch.zeros(n, 9, dtype=f64, device=dev)
    v = v/v.abs().max()
    b = -gxy(v, torch.zeros_like(Y))[1]
    one = torch.ones((), dtype=f64, device=dev); zero = torch.zeros((), dtype=f64, device=dev)
    parts = {}
    vM = float((M[:, None]*v*v).sum())
    for nm in st.energy_terms(th, X, Y):
        f = lambda s_: st.energy_terms(th, X+s_*v, Y+s_*z)[nm]
        d1 = lambda s_: torch.func.jvp(f, (s_,), (one,))[1]
        parts[nm] = float(torch.func.jvp(d1, (zero,), (one,))[1])/vM
    tot = sum(parts.values())
    bb = int(v.norm(dim=1).argmax()); share = float(v[bb].norm()**2/(v*v).sum())
    print('skin free: omega_max %.3e rad/s (Rayleigh along the final (v, z): %.3e); stable dt < %.3e s = 1 / %.0f s; CG per iteration median %d; '
          '%.2f of the mode on body %d (model.pt index %d, volume / median %.3f); share of omega^2 by term: %s' % (
              om2**.5, tot**.5, 2/om2**.5, om2**.5/2, int(np.median(cg)), share, bb, int(cells['source_index'][bb]), float(st.V[bb]/st.V.median()),
              ' '.join('%s %.3f' % (k_, v_/tot) for k_, v_ in parts.items())), flush=True)
    return st, cells, v, z, om2


def consistent_rayleigh(out=Path('output/round01_pruned_exact'), ncells=300, samples=200000):
    """the skin-free fastest mode (v, z) with the consistent mass: kinetic energy sum over cells of
    rho int |v_B + dF_B (X - X_B)|^2 dV = V |v|^2 + 2 v . dF m1 + tr(dF M2 dF^T) (dF = F(v, z), F linear; m1, M2 the cell's
    first and second moments about its body point, sampled for the ncells cells with the largest V (|v|^2 + |dF|^2 r^2), the
    others lumped); prints omega for the lumped and the consistent mass (a lower bound of the consistent omega_max)"""
    from scipy.spatial import cKDTree
    st, cells, v, z, om2 = condensed(out)
    p, r = cells['points'], cells['radii']; n = st.n
    dF = st.F(v, z)
    V = st.V; rt = torch.as_tensor(r, dtype=f64, device=dev); pt = torch.as_tensor(p, dtype=f64, device=dev)
    imp = V*((v*v).sum(1)+(dF*dF).sum((1, 2))*rt**2)
    sel = torch.argsort(imp, descending=True)[:ncells].cpu().numpy()
    tree = cKDTree(p); g = torch.Generator(device=dev).manual_seed(0)
    lumped = float((V[:, None]*v*v).sum()); den = lumped; verr = []
    for c in sel:
        cand = torch.as_tensor(tree.query_ball_point(p[c], r[c]+r.max()), device=dev)
        u = torch.randn(samples, 3, generator=g, dtype=f64, device=dev); u = u/u.norm(dim=1, keepdim=True)
        X = pt[c]+u*(rt[c]*torch.rand(samples, 1, generator=g, dtype=f64, device=dev)**(1/3))
        pw = ((X[:, None]-pt[cand][None])**2).sum(-1)-rt[cand][None]**2
        own = pw.argmin(1)
        keep = cand[own] == int(c)
        ball = 4/3*np.pi*r[c]**3
        Rk = X[keep]-st.X[c]
        w = ball/samples
        m1 = Rk.sum(0)*w; M2 = (Rk[:, :, None]*Rk[:, None, :]).sum(0)*w
        verr.append(abs(float(keep.sum())*w/float(V[c])-1))
        den = den+2*float(v[c]@(dF[c]@m1))+float(torch.trace(dF[c]@M2@dF[c].T))
    num = om2*lumped
    print('fastest skin-free mode: omega lumped %.3e rad/s, consistent %.3e rad/s (mass ratio %.2f); moments sampled for %d cells (volume error median %.3f, max %.3f), '
          'their share of the lumped-mass kinetic energy %.3f' % (om2**.5, (num/den)**.5, den/lumped, ncells, np.median(verr), max(verr),
                                                                  float((V[sel][:, None]*v[sel]*v[sel]).sum())/lumped), flush=True)


if __name__ == '__main__':
    main()
