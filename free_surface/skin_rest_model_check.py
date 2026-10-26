"""Is SkinSolver.rest_model the skin Hessian at rest? (2026-10-23, new foam, single bodies, skin C: shepard0, lever c)
Hessian-vector products of the model's Q against the energy's own (torch autodiff) for random skin directions, total and
by term (the energy's terms with SkinTerms.energy_terms, Q's with its dict); then the model's groups' blocks (by bodies)
against the probed K_rest blocks (blocks_check: 50 random bodies more than four bonds apart, probed together on Q)."""
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from experiments.pacnerf_dem_forward import build as dem_build
from free_surface.boundary_skin import BoundaryTerms
from free_surface.skin_solve import SkinSolver

f64, dev = torch.float64, 'cuda'


def main():
    t0 = time.time()
    out = Path('output/round01_exact')
    cells = np.load(out/'dem_exact_single'/'cells.npz'); p, r = cells['points'], cells['radii']
    g = dict(np.load(out/'dem_exact_single'/'geometry.npz')); bsk = dict(np.load(out/'boundary_skin.npz'))
    m = dem_build(g, 23008.38, .3008, dev, 'noslip', .25, .3, f64, kinematics='exact', nodalvol=True, nonlinear=False, ogden_alpha=2.)
    st = BoundaryTerms(m, g, p, r, bsk, lever='c')
    S = SkinSolver(st, bsk, g['body_of_cell'])
    X, Y, npt = st.X, st.Y, st.npt
    t1 = time.time()
    pieces = S.rest_pieces()
    print('rest pieces (bond rotation Hessian, wall volume derivative): %d colours (two bonds apart), %.1f s' % (pieces['colours'], time.time()-t1), flush=True)
    Q = S.rest_model(pieces)
    names = list(Q(torch.zeros_like(Y)).keys())
    gen = torch.Generator(device=dev).manual_seed(11)
    for k in range(3):
        u = torch.randn(npt, 3, generator=gen, dtype=f64, device=dev)
        # the energy's terms: second directional derivative along u, and the Hessian-vector product of the total
        ref = {}
        for nm in st.energy_terms(S.th, X, Y):
            f = lambda s: st.energy_terms(S.th, X, Y+s*u)[nm]
            d1 = lambda s: torch.func.jvp(f, (s,), (torch.ones((), dtype=f64, device=dev),))[1]
            ref[nm] = float(torch.func.jvp(d1, (torch.zeros((), dtype=f64, device=dev),), (torch.ones((), dtype=f64, device=dev),))[1])
        mod = {}
        for nm in names:
            fq = lambda s: Q(s*u)[nm]
            d1 = lambda s: torch.func.jvp(fq, (s,), (torch.ones((), dtype=f64, device=dev),))[1]
            mod[nm] = float(torch.func.jvp(d1, (torch.zeros((), dtype=f64, device=dev),), (torch.ones((), dtype=f64, device=dev),))[1])
        gE = torch.func.grad(lambda yy: S.energy(X, yy.view(-1, 3)))
        Hu = torch.func.jvp(gE, (Y.reshape(-1),), (u.reshape(-1),))[1]
        gQ = torch.func.grad(lambda U: sum(Q(U.view(-1, 3)).values()))
        Qu = torch.func.jvp(gQ, (torch.zeros(3*npt, dtype=f64, device=dev),), (u.reshape(-1),))[1]
        pair = dict(bonds=('bonds',), volume=('volume',), outer_stretch=('o1',), outer_cross=('o2',), outer_skin=('o3',), outer_lateral=('lateral',))
        print('direction %d: |H_model u - H u| / |H u| %.2e; u^T H u by term, energy / model: %s' % (
            k, float((Qu-Hu).norm()/Hu.norm()), '; '.join('%s %.6e / %.6e' % (nm, ref[nm], sum(mod[q] for q in pair[nm])) for nm in ref)), flush=True)
    print('total %.0f s' % (time.time()-t0), flush=True)


def blocks_check(n_groups=50):
    """the model's own blocks for random bodies (probed together on Q) against the probed K_rest blocks"""
    out = Path('output/round01_exact')
    cells = np.load(out/'dem_exact_single'/'cells.npz'); p, r = cells['points'], cells['radii']
    g = dict(np.load(out/'dem_exact_single'/'geometry.npz')); bsk = dict(np.load(out/'boundary_skin.npz'))
    m = dem_build(g, 23008.38, .3008, dev, 'noslip', .25, .3, f64, kinematics='exact', nodalvol=True, nonlinear=False, ogden_alpha=2.)
    st = BoundaryTerms(m, g, p, r, bsk, lever='c')
    S = SkinSolver(st, bsk, g['body_of_cell'])
    npt = st.npt; v3 = 3*npt
    Q = S.rest_model()
    gQ = torch.func.grad(lambda U: sum(Q(U.view(-1, 3)).values()))
    z = np.load(out/'skin_Kyy_rest.npz'); P_, Q_, vals = z['P'], z['Q'], z['vals']
    rows = (3*P_[:, None, None]+np.arange(3)[None, :, None]).repeat(3, 2); cols = (3*Q_[:, None, None]+np.arange(3)[None, None, :]).repeat(3, 1)
    K = sp.coo_matrix((vals.ravel(), (rows.ravel(), cols.ravel())), shape=(v3, v3)).tocsr(); K = (.5*(K+K.T)).tocsr()
    # bodies more than four bonds apart (no energy term holds skin points of two of them), so one probe per local index
    # gives all their blocks at once
    n = st.n; pr = g['pairs']
    A1 = sp.coo_matrix((np.ones(2*len(pr)), (np.r_[pr[:, 0], pr[:, 1]], np.r_[pr[:, 1], pr[:, 0]])), shape=(n, n)).tocsr()+sp.eye(n, format='csr')
    A2 = (A1@A1).astype(bool).astype(float); A4 = (A2@A2).astype(bool).tocsr()
    rng = np.random.default_rng(2)
    owner = S.bodies['owner'].cpu().numpy()
    blocked = np.zeros(n, bool); chosen = []
    for k_ in rng.permutation(S.bodies['G']):
        b_ = owner[k_]
        if not blocked[b_]:
            chosen.append(k_); blocked[A4.indices[A4.indptr[b_]:A4.indptr[b_+1]]] = True
        if len(chosen) == n_groups:
            break
    S3 = 3*S.bodies['S']; z3 = torch.zeros(v3, dtype=f64, device=dev)
    I3 = [torch.as_tensor((3*np.asarray(S.bodies['lists'][k_])[:, None]+np.arange(3)).ravel(), device=dev) for k_ in chosen]
    Bq = [np.zeros((len(i3), len(i3))) for i3 in I3]
    for c in range(S3):
        e = torch.zeros(v3, dtype=f64, device=dev)
        for i3 in I3:
            if c < len(i3):
                e[i3[c]] = 1.
        col = torch.func.jvp(gQ, (z3,), (e,))[1]
        for k_, i3 in enumerate(I3):
            if c < len(i3):
                Bq[k_][:, c] = col[i3].cpu().numpy()
    errs = []
    for k_, i3 in enumerate(I3):
        ii = i3.cpu().numpy(); Bk = K[ii][:, ii].toarray()
        errs.append(np.linalg.norm(Bq[k_]-Bk)/np.linalg.norm(Bk))
    errs = np.array(errs); n_groups = len(chosen)
    print('model blocks of %d random bodies against probed K_rest: relative difference median %.1e, max %.1e' % (n_groups, np.median(errs), errs.max()), flush=True)


if __name__ == '__main__':
    main()
    blocks_check()
