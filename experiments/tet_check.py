"""Checks for a tet sub-mesh file (tet_cells.npz layout): per-cell spectrum (soft modes) and global omega_max.

Per sampled cell: linear-elastic K over the cell's tets (P1, points expanded to vertices), lumped
vertex masses, scaled frequencies omega h / c_s (h = cell volume^(1/3)); global omega_max of
the whole mesh (no binding, no mass scaling) -> explicit steps per frame.
"""
import argparse
import json
import numpy as np
import torch

E_, NU, RHO = 23008.4, .30079, 1000.
LAM, MU = E_*NU/((1+NU)*(1-2*NU)), E_/(2*(1+NU))
CS = np.sqrt(MU/RHO)


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('path')
    a.add_argument('--samples', type=int, default=60)
    args = a.parse_args()
    d = np.load(args.path)
    nv, nf = len(d['X']), int(d['n_face'])
    pts = d['sp_pt'].reshape(-1, 4); g = d['sp_g'].reshape(-1, 4, 3); V = d['tet_volume']; tc = d['tet_cell']
    rows = {}
    def expand(pidx):
        if pidx < nv:
            return [(pidx, 1.)]
        if pidx < nv+nf:
            m = d['fc_face'] == pidx-nv
            return list(zip(d['fc_vert'][m], d['fc_w'][m]))
        m = d['ap_cell'] == pidx-nv-nf
        return list(zip(d['ap_vert'][m], d['ap_w'][m]))
    mass = RHO*d['vertex_volume']
    rng = np.random.default_rng(1)
    cells = rng.choice(np.unique(tc), args.samples, replace=False)
    if 'cells_with_drops' in d.files and len(d['cells_with_drops']):
        dc = d['cells_with_drops']
        cells = np.unique(np.r_[cells, rng.choice(dc, min(60, len(dc)), replace=False)])   # always include cells that lost tets
    D = np.zeros((6, 6)); D[:3, :3] = LAM; D[range(3), range(3)] += 2*MU; D[range(3, 6), range(3, 6)] = MU
    res = []
    for c in cells:
        ts = np.flatnonzero(tc == c)
        cache = {}
        verts = set()
        for t in ts:
            for p in pts[t]:
                if p not in cache:
                    cache[p] = expand(int(p))
                verts.update(v for v, _ in cache[p])
        loc = {v: i for i, v in enumerate(sorted(verts))}; n = len(loc)
        K = np.zeros((3*n, 3*n))
        for t in ts:
            gw = {}
            for p, gr in zip(pts[t], g[t]):
                for v, w in cache[p]:
                    gw[loc[v]] = gw.get(loc[v], 0)+w*gr
            idx = np.array(list(gw)); G = np.array([gw[i] for i in idx]); q = len(idx)
            B = np.zeros((6, 3*q))
            for j in range(q):
                gx, gy, gz = G[j]
                B[:, 3*j:3*j+3] = [[gx, 0, 0], [0, gy, 0], [0, 0, gz], [gy, gx, 0], [0, gz, gy], [gz, 0, gx]]
            dofs = (3*idx[:, None]+np.arange(3)).ravel()
            K[np.ix_(dofs, dofs)] += V[t]*B.T@D@B
        m = mass[sorted(verts)]
        s = 1/np.sqrt(np.repeat(m, 3))
        w2 = np.linalg.eigvalsh(.5*(K+K.T)*s[:, None]*s[None, :])
        h = np.cbrt(V[ts].sum())
        w = np.sqrt(np.clip(w2, 0, None))*h/CS
        nr = np.sort(w)[6:]
        if len(nr) == 0:
            continue
        res.append(dict(zero=int((w2 < 1e-10*w2.max()).sum())-6, min=float(nr.min()), below1=int((nr < 1).sum()), max=float(w.max())))
    summ = dict(cells_with_zero_modes=int(sum(r['zero'] > 0 for r in res)), min_scaled_median=float(np.median([r['min'] for r in res])),
                min_scaled_min=float(min(r['min'] for r in res)), modes_below_1_mean=float(np.mean([r['below1'] for r in res])),
                max_scaled_median=float(np.median([r['max'] for r in res])), max_scaled_max=float(max(r['max'] for r in res)))
    from experiments.tet_dynamics import TetSolid
    gt = json.load(open('data/pacnerf/gt/elastic.json'))['0']
    m = TetSolid(args.path, 'cuda', torch.float64, cluster_tol=0)
    w = m.max_frequency(tuple(torch.tensor(t, dtype=torch.float64, device='cuda') for t in (gt['E'], gt['nu'], 1.)), iterations=300)
    summ.update(omega_max=w, steps_per_frame=int(np.ceil(w/24)))
    print(json.dumps(summ, indent=1), flush=True)


if __name__ == '__main__':
    main()
