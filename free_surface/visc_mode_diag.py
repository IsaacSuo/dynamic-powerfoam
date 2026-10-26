"""Why the Kelvin-Voigt damping of a mode falls short of beta omega / 2 (2026-10-24, static). Each FEM mode u
(visc_mode_decay.scaled_shape) displaced by --amp, the skin solved (viscous term off); then
  E_el   the model's elastic energy (bonds, volume, skin terms);
  R      the dissipation 1/2 w . D w of the velocity field w = (u, skin displacement) at rest (SkinWarp.visc_hvp; until
         10-24 the Kelvin-Voigt law on the cells' fitted strain: then R = beta sum_B V_B psi(eps_B));
  fitted the cells' fitted strain energy sum_B V_B psi(eps_B), eps_B = sym(F_B - I) (F_B is affine in x, y and I at
         rest), psi = mu |dev eps|^2 + (lam + 2 mu / 3) (tr eps)^2 / 2;
  R / (beta E_el)  the damping ratio over beta omega / 2 this mode gets (its Rayleigh quotients);
  E_FEM  the FEM's energy of the mode, (amp / s)^2 (2 pi f)^2 / 2 (M-normalised, phi^T K phi = (2 pi f)^2), and the
         cells' sampled FEM strain energy sum_B V_B psi(eps_FEM(X_B)) (trilinear gradient; cells whose voxel lacks
         nodes left out).
Per cell the fitted against the FEM strain energy, split into surface cells (with skin entries) and interior ones;
--plot modes: the bodies coloured by |u| and by the missing strain energy density, three views.
  python free_surface/visc_mode_diag.py --out output/round01_pruned_exact --fem output/round01_exact/fem_modes_h0.008.npz --beta 0.005 --amp 0.001"""
import numpy as np
import torch

from free_surface.skin_contact_check import f64, dev
from free_surface.skin_dynamics_check import build_model, parser
from free_surface.visc_mode_decay import scaled_shape


def fem_grad(cache, k, Xq):
    """the FEM mode's displacement gradient du_i/dX_j (n, 3, 3) at the points by the trilinear voxel shape functions,
    and the points whose voxel has all 8 nodes"""
    lo, h, grid = cache['lo'], float(cache['h']), cache['node_grid']
    key = lambda g: (g[:, 0]*100003+g[:, 1])*100003+g[:, 2]
    order = np.argsort(key(grid)); keys = key(grid)[order]
    def node_of(g):
        k_ = key(g); i_ = np.clip(np.searchsorted(keys, k_), 0, len(keys)-1)
        return np.where(keys[i_] == k_, order[i_], -1)
    t = (Xq-lo)/h; b = np.floor(t).astype(int); f = t-b
    U = cache['V'][:, k].reshape(len(grid), 3)
    G = np.zeros((len(Xq), 3, 3)); ok = np.ones(len(Xq), bool)
    for c in np.array(np.meshgrid([0, 1], [0, 1], [0, 1], indexing='ij')).reshape(3, -1).T:
        n_ = node_of(b+c); ok &= n_ >= 0
        w1 = np.where(c[None] == 1, f, 1-f)                                          # per axis
        dw = np.stack([np.where(c[d] == 1, 1., -1.)/h*np.prod(np.delete(w1, d, axis=1), axis=1) for d in range(3)], 1)
        G += U[np.maximum(n_, 0)][:, :, None]*dw[:, None, :]
    return G, ok


def psi(eps, mu, lam):
    tr = eps.diagonal(dim1=-2, dim2=-1).sum(-1)
    dv = eps-tr[..., None, None]/3*torch.eye(3, dtype=f64, device=dev)
    return mu*(dv*dv).sum((-2, -1))+.5*(lam+2*mu/3)*tr**2


def main():
    a = parser()
    a.add_argument('--modes', type=int, nargs='+', default=None, help='default: all')
    a.add_argument('--plot', type=int, nargs='*', default=[0, 20])
    args = a.parse_args()
    assert args.beta > 0 and args.eta is None and args.zeta is None
    b = build_model(args)
    st, W, S, prec, mass = b['st'], b['W'], b['S'], b['prec'], b['mass']
    X, Y, V = st.X, st.Y, st.V
    mu, lam, beta, amp = W.mu, W.lam, args.beta, args.amp
    cache = np.load(args.fem)
    modes = args.modes if args.modes is not None else list(range(len(cache['f'])))
    W.visc_ref(X, Y)
    print('rest: |F_B - I| max %.1e' % float((st.F(X, Y)-torch.eye(3, dtype=f64, device=dev)).abs().max()), flush=True)
    surf = torch.zeros(len(X), dtype=torch.bool, device=dev); surf[st.e_body] = True
    print('cells %d, surface (with skin entries) %d, their volume share %.3f' % (len(X), int(surf.sum()), float(V[surf].sum()/V.sum())), flush=True)
    print('mode | f FEM | E_el / E_FEM | fitted / E_FEM | sampled FEM / E_FEM | R / (beta E_el) | missing (sampled - fitted) / sampled: all, surface, interior | cells without FEM gradient', flush=True)
    saved = {}
    for k in modes:
        u, s = scaled_shape(cache, k, X, mass)
        x = X+amp*u
        W.elastic_mode()                                                             # the skin's viscous term off
        y = Y.clone()
        for _ in range(5):
            y = S.affine_skin(x, y)
        y, s0 = S.solve(x, y, prec, float(S.grad(x, y).norm()), tol=1e-8, maxit=1000)
        W.grad(x, y); E_el = float(W.tsum[:6].sum())
        W.total_mode()
        Dx, Dy = W.visc_hvp(X, Y, x-X, y-Y)
        R = .5*float(((x-X)*Dx).sum()+((y-Y)*Dy).sum())
        F = st.F(x, y); eps = .5*(F+F.transpose(1, 2))-torch.eye(3, dtype=f64, device=dev)
        pf = V*psi(eps, mu, lam)
        G, ok = fem_grad(cache, k, X.cpu().numpy())
        G = torch.as_tensor(G, dtype=f64, device=dev)*(amp/s); okt = torch.as_tensor(ok, device=dev)
        pF = V*psi(.5*(G+G.transpose(1, 2)), mu, lam)
        E_fem = .5*(amp/s)**2*(2*np.pi*cache['f'][k])**2
        miss = lambda m: float((pF-pf)[m & okt].sum()/pF[m & okt].sum())
        print('%2d | %.2f | %.4f | %.4f | %.4f | %.4f | %.3f %.3f %.3f | %d   (skin %d products)' % (
            k, cache['f'][k], E_el/E_fem, float(pf.sum())/E_fem, float(pF[okt].sum())/E_fem, R/(beta*E_el), miss(torch.ones_like(surf)), miss(surf), miss(~surf),
            int((~ok).sum()), s0['products']), flush=True)
        if k in args.plot:
            saved[k] = dict(u=u.cpu().numpy(), pf=pf.cpu().numpy(), pF=pF.cpu().numpy(), ok=ok)
    if saved:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        P = X.cpu().numpy(); Vn = V.cpu().numpy()
        views = ((0, 1, 'x', 'y'), (0, 2, 'x', 'z'), (2, 1, 'z', 'y'))
        fig, ax = plt.subplots(2*len(saved), 3, figsize=(15, 8.5*len(saved)))
        ax = np.atleast_2d(ax)
        for r, (k, d) in enumerate(saved.items()):
            un = np.linalg.norm(d['u'], axis=1)
            dens = np.where(d['ok'], (d['pF']-d['pf'])/Vn, np.nan)
            lim = np.nanpercentile(np.abs(dens), 99)
            for c, (i, j, li, lj) in enumerate(views):
                o = np.argsort(P[:, 3-i-j])                                          # the far side first
                a1 = ax[2*r, c]; s1 = a1.scatter(P[o, i], P[o, j], c=un[o], s=4, cmap='viridis'); a1.set_aspect('equal')
                a1.set_title('mode %d (FEM %.2f Hz): |u| (largest 1), view %s-%s' % (k, cache['f'][k], li, lj)); fig.colorbar(s1, ax=a1, shrink=.7)
                a2 = ax[2*r+1, c]; s2 = a2.scatter(P[o, i], P[o, j], c=dens[o], s=4, cmap='coolwarm', vmin=-lim, vmax=lim); a2.set_aspect('equal')
                a2.set_title('mode %d: FEM - fitted strain energy density (red: fit misses)' % k); fig.colorbar(s2, ax=a2, shrink=.7)
        fig.tight_layout()
        fig.savefig(b['out']/'visc_mode_diag.png', dpi=80)
        print('figure: %s' % (b['out']/'visc_mode_diag.png'), flush=True)


if __name__ == '__main__':
    main()
