"""The in-solve return's one approximation (2026-10-26): each cell's plastic increment chosen from its own F_B strain
as if the cell were one block of continuum (V mu |dev eps|^2), while the model keeps the cell's energy in the bonds it
shares and its outer layer. The force error that leaves, in a yielding cell, is the gap between two stresses:
  sigma_FB   2 mu dev(log U), U = R^T F of the cell's F_B
  sigma_vir  -(1/V) dE/deps_p of the model's own energy (the bonds' d - dp, o1, o2 on the cell's plastic references):
             -(1/V) sym(-sum_bonds g (x) r_i (+ g (x) r_j at the j end) - 2 mu r R^T D N (o1) - 2 mu R^T daff (x) c (o2))
             with g = k_n dn n + k_s d_t the bond's force, at eps_p = 0
(deviatoric parts). On the part: a uniform pure shear (exact: 0 expected), each of the first 6 FEM modes and their
random mix (5 mm), the skin solved; dE/deps_p checked by central differences on a few cells."""
import math

import numpy as np
import torch

from experiments.dem_fem_mode_match import interpolate
from free_surface.plastic import Plasticity, sym_fun, dev_part
from free_surface.skin_contact_check import f64, dev
from free_surface.skin_dynamics_check import build, parser


def dE_deps(W, st, x, y):
    """dE/deps_p per cell (n, 3, 3) at eps_p = 0 (W's buffers from grad(x, y))"""
    n = W.n
    G = torch.zeros(n, 3, 3, dtype=f64, device=dev)
    d, nn, k = W.tbd, st.model.n0.to(f64), W._kb_t
    dn = (d*nn).sum(1, keepdim=True)
    g = k[:, :1]*dn*nn+k[:, 1:2]*(d-dn*nn)
    bi, bj = st.i.long(), st.j.long()
    ri, rj = st.model.ri.to(f64), st.model.rj.to(f64)
    G.index_add_(0, bi, -torch.einsum('ea,eb->eab', g, ri))
    G.index_add_(0, bj, torch.einsum('ea,eb->eab', g, rj))
    F, R = W.tF, W.tR
    cb = st.c_body.long()
    D = F[cb]-R[cb]
    G.index_add_(0, cb, -2*W.mu*st.c_r[:, None, None]*R[cb].transpose(1, 2)@D@st.c_N)
    eb, ep = st.e_body.long(), st.e_pid.long()
    daff = y[ep]-x[eb]-torch.einsum('eab,eb->ea', F[eb], st.e_d)
    G.index_add_(0, eb, -2*W.mu*torch.einsum('ea,eb->eab', torch.einsum('eba,eb->ea', R[eb], daff), st.e_c))
    return G


def main():
    args = parser().parse_args(['--case', 'part', '--integrator', 'midpoint'])
    b = build(args)
    st, W, S, prec = b['st'], b['W'], b['S'], b['prec']
    X, Y, V = st.X, st.Y, st.V.to(f64)
    pl = Plasticity(W, st, math.inf)
    surf = torch.zeros(W.n, dtype=torch.bool, device=dev); surf[st.c_body.long()] = True
    cache = np.load(b['out']/'fem_modes_h0.0015.npz')
    fem = dict(nodes=cache['nodes'], node_grid=cache['node_grid'], lo=cache['lo'], h=float(cache['h']), masses=cache['masses'])
    Ux, _ = interpolate(fem, cache['V'][:, :6], X.cpu().numpy())
    U6 = torch.as_tensor(Ux.transpose(2, 0, 1), dtype=f64, device=dev)
    xc = X.mean(0)
    A = torch.tensor([[.01, .004, 0.], [.004, -.006, .002], [0., .002, -.004]], dtype=f64, device=dev)
    states = [('uniform pure shear 1%', X+(X-xc)@A.T)]
    for m in range(6):
        u = U6[m]; states.append(('FEM mode %d, 5 mm' % (m+1), X+u*.005/float(u.norm(dim=1).max())))
    states.append(('the 6 modes mixed (start state)', b['x0']))
    rng = np.random.default_rng(0)
    for name, x in states:
        y = Y.clone()
        for _ in range(5):
            y = S.affine_skin(x, y)
        y, s_ = S.solve(x, y, prec, float(S.grad(x, y).norm()), tol=1e-10, maxit=3000)
        pl.eps.zero_(); pl.apply()
        pl.elastic(x, y)
        G = dE_deps(W, st, x, y)
        svir = -dev_part(.5*(G+G.transpose(1, 2)))/V[:, None, None]
        sfb = 2*W.mu*dev_part(sym_fun(W.tR.transpose(1, 2)@W.tF, torch.log))
        nf = sfb.flatten(1).norm(dim=1); dd = (svir-sfb).flatten(1).norm(dim=1)
        ref = float(nf.median())
        rel = dd/nf.clamp(min=1e-30)
        big = nf >= ref                                                              # (the cells at or above the median stress)
        q = lambda t: '%.3f / %.3f / %.3f' % tuple(float(v) for v in torch.quantile(t, torch.tensor([.5, .9, 1.], dtype=f64, device=dev)))
        print('%s (skin %d products): |dev sigma_vir - dev sigma_FB| / |dev sigma_FB|, median / 90 %% / max: interior %s, surface %s; '
              'cells at or above the median stress %s; all cells, |difference| / median |sigma_FB|: %s; mass-weighted |sum V diff| / |sum V sigma_FB| %.3f' % (
                  name, s_['products'], q(rel[~surf]), q(rel[surf]), q(rel[big]), q(dd/ref),
                  float(((V[:, None, None]*(svir-sfb)).flatten(1).norm(dim=1)).sum()/((V[:, None, None]*sfb).flatten(1).norm(dim=1)).sum())), flush=True)
        if name.startswith('FEM mode 1'):                                            # dE/deps_p by central differences
            for c in rng.choice(W.n, 4, replace=False).tolist():
                Sd = torch.as_tensor(rng.standard_normal((3, 3)), dtype=f64, device=dev); Sd = dev_part(.5*(Sd+Sd.T))
                h = 1e-6
                e = []
                for sg in (1, -1):
                    pl.eps.zero_(); pl.eps[c] = sg*h*Sd; pl.apply()
                    e.append(pl.elastic(x, y))
                fd = (e[0]-e[1])/(2*h); an = float((G[c]*Sd).sum())
                print('   cell %d (%s): dE/deps_p . S by differences %.6e, analytic %.6e (rel %.1e)' % (
                    c, 'surface' if bool(surf[c]) else 'interior', fd, an, abs(fd-an)/max(abs(fd), 1e-300)), flush=True)
            pl.eps.zero_(); pl.apply()


if __name__ == '__main__':
    main()
