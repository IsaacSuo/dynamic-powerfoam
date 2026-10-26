"""Which skin point of the new foam's boundary skin has a diagonal stiffness block that is not positive definite, and
why (2026-10-22, diagnostic): blocks probed colour by colour as in foam_modes.py; for the bad points, the exact block
by single-point probes, split per energy term, and the point's geometry (corner or arc midpoint, its regions' cells,
area shares a_q, |c_q|, the angle between c_q and the lever d_q, the arcs it sits on, the cells' radii and volumes)."""
import argparse
from pathlib import Path

import numpy as np
import torch

from experiments.pacnerf_dem_forward import build
from free_surface.boundary_skin import BoundaryTerms

f64, dev = torch.float64, 'cuda'


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--out', type=Path, default=Path('output/round01_exact'))
    a.add_argument('--skin', default='boundary_skin.npz')
    a.add_argument('--batch', type=int, default=4)
    args = a.parse_args()
    cells = np.load(args.out/'dem_exact_single'/'cells.npz'); p, r = cells['points'], cells['radii']
    g = dict(np.load(args.out/'dem_exact_single'/'geometry.npz')); bsk = dict(np.load(args.out/args.skin))
    m = build(g, 23008.38, .3008, dev, 'noslip', .25, .3, f64, kinematics='exact', nodalvol=True, nonlinear=False, ogden_alpha=2.)
    st = BoundaryTerms(m, g, p, r, bsk, lever='c')
    n, npt = st.n, st.npt; n3, v3 = 3*n, 3*npt
    th = torch.zeros(n, 9, dtype=f64, device=dev)
    def hvp_for(f):
        z0 = torch.zeros(v3, dtype=f64, device=dev); gf = torch.func.grad(f)
        return torch.func.vmap(lambda v: torch.func.jvp(gf, (z0,), (v,))[1])
    total = hvp_for(lambda zy: st.energy_xy(th, st.X, st.Y+zy.view(-1, 3)))
    # colours as in foam_modes (cells within two bonds)
    nbr = [set([c_]) for c_ in range(n)]
    for i_, j_ in g['pairs']:
        nbr[i_].add(j_); nbr[j_].add(i_)
    nbr2 = [set().union(*(nbr[k_] for k_ in nbr[c_])) for c_ in range(n)]
    cells_of = [set() for _ in range(npt)]
    for q_, c_ in zip(bsk['ent_pid'], bsk['ent_site']):
        cells_of[q_].add(int(c_))
    pts_of_cell = [[] for _ in range(n)]
    for q_ in range(npt):
        for c_ in cells_of[q_]:
            pts_of_cell[c_].append(q_)
    colour = -np.ones(npt, int)
    for q_ in range(npt):
        reach = set().union(*(nbr2[c_] for c_ in cells_of[q_]))
        used = {colour[p_] for c_ in reach for p_ in pts_of_cell[c_] if colour[p_] >= 0}
        k_ = 0
        while k_ in used:
            k_ += 1
        colour[q_] = k_
    col_t = torch.as_tensor(colour, device=dev)
    blocks = torch.zeros(npt, 3, 3, dtype=f64, device=dev)
    probes = [(k_, d_) for k_ in range(int(colour.max())+1) for d_ in range(3)]
    for s_ in range(0, len(probes), args.batch):
        ch = probes[s_:s_+args.batch]
        Vp = torch.zeros(len(ch), npt, 3, dtype=f64, device=dev)
        for b_, (k_, d_) in enumerate(ch):
            Vp[b_, col_t == k_, d_] = 1.
        Kv = total(Vp.reshape(len(ch), -1)).view(len(ch), npt, 3)
        for b_, (k_, d_) in enumerate(ch):
            sel = col_t == k_
            blocks[sel, :, d_] = Kv[b_, sel]
    ev = torch.linalg.eigvalsh(.5*(blocks+blocks.transpose(1, 2)))
    bad = torch.nonzero(ev[:, 0] <= 0)[:, 0].tolist()
    print('skin points %d; blocks not positive definite: %d %s; smallest-eigenvalue quantiles over points (of the largest block) 0/0.1/1/50%%: %s' % (
        npt, len(bad), bad, np.round(np.quantile(ev[:, 0].cpu().numpy()/float(ev[:, 2].max()), [0, .001, .01, .5]), 5).tolist()), flush=True)
    terms = ('bonds', 'volume', 'outer_stretch', 'outer_cross', 'outer_skin', 'outer_lateral')
    pid, site, aq, cq = bsk['ent_pid'], bsk['ent_site'], bsk['ent_a'], bsk['ent_c']
    Y = bsk['Y']; nc = int(bsk['n_corners'])
    for q in bad[:5]:
        V = torch.zeros(3, v3, dtype=f64, device=dev); V[[0, 1, 2], [3*q, 3*q+1, 3*q+2]] = 1.
        Kq = total(V)[:, 3*q:3*q+3]; Kq = .5*(Kq+Kq.T)
        w, U = torch.linalg.eigh(Kq)
        print('\npoint %d (%s): exact block eigenvalues %s; softest direction %s' % (q, 'corner' if q < nc else 'arc midpoint', np.round(w.cpu().numpy(), 4).tolist(),
              np.round(U[:, 0].cpu().numpy(), 3).tolist()), flush=True)
        for t in terms:
            Kt = hvp_for(lambda zy, t=t: st.energy_terms(th, st.X, st.Y+zy.view(-1, 3))[t])(V)[:, 3*q:3*q+3]
            Kt = .5*(Kt+Kt.T)
            print('   %-14s along the softest direction %+.4f; eigenvalues %s' % (t, float(U[:, 0]@Kt@U[:, 0]), np.round(torch.linalg.eigvalsh(Kt).cpu().numpy(), 4).tolist()), flush=True)
        e_ = np.flatnonzero(pid == q)
        for e in e_:
            s_ = site[e]; d = Y[q]-p[s_]
            print('   region of cell %d: r %.4f, cell volume %.3e (median %.3e), area share a %.3e (region area %.3e), |c| %.3e, cos(c, d) %.3f' % (
                s_, r[s_], g['masses'][s_], np.median(g['masses']), aq[e], aq[np.flatnonzero(site == s_)].sum(), np.linalg.norm(cq[e]), (cq[e]@d)/np.linalg.norm(cq[e])/np.linalg.norm(d)), flush=True)


if __name__ == '__main__':
    main()
