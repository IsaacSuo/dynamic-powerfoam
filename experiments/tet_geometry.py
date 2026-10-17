"""Linear tetrahedral sub-mesh of the Power Foam material cells (no stabilisation needed).

Every material power cell is split into tetrahedra (cell vertex mean, face vertex mean, face edge
a-b). The two centres are averages of vertices ("points": vertex v, face centre nv + f, cell apex
nv + n_face + c, as in vem_geometry.pyramid_subcells), so the unknowns are only the cell vertices.
Each tet has the exact P1 gradient F_t = sum_p x_p (x) g_tp (g from the rest positions); energy
sum_t |t| psi(F_t). The element spectrum check (tet_spectrum.py) found no mode softer than the
material at the cell scale (omega h / c_s >= 1), unlike the pyramid VEM / SFVEM variants.
Tiny cells are not merged: vertex binding at run time (VEMSolid.set_clusters) makes them rigid.
Lumped mass: |t| / 4 per tet point, spread to the vertices through the point averages.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--geometry', type=Path, default=Path('output/pacnerf_vem/elastic_0_f0/vem_cap1_bodies_pyr.npz'))
    p.add_argument('--out', type=Path, default=Path('output/pacnerf_vem/elastic_0_f0/tet_cells.npz'))
    args = p.parse_args()
    d = np.load(args.geometry)
    X, fv, fs, fp, src = d['X'], d['face_vertices'], d['face_start'], d['face_pair'], d['source_index']
    nv, nf, nc = len(X), len(fs), len(src)
    k = np.diff(np.r_[fs, len(fv)])
    cell_of = -np.ones(max(fp.max(), src.max())+1, int); cell_of[src] = np.arange(nc)
    fid = np.repeat(np.arange(nf), k)
    pos = np.arange(len(fv))-fs[fid]
    a_all, b_all = fv, fv[fs[fid]+(pos+1) % k[fid]]
    fcen = np.zeros((nf, 3)); np.add.at(fcen, fid, X[fv]); fcen /= k[:, None]
    # cell vertex sets -> apex (vertex mean)
    inc_f, inc_c, inc_s = [], [], []
    for side, sign in ((0, 1.), (1, -1.)):
        c = cell_of[fp[:, side]]; ok = c >= 0
        inc_f.append(np.flatnonzero(ok)); inc_c.append(c[ok]); inc_s.append(np.full(ok.sum(), sign))
    inc_f, inc_c, inc_s = map(np.concatenate, (inc_f, inc_c, inc_s))
    rep = k[inc_f]
    corner_inc = np.repeat(np.arange(len(inc_f)), rep)
    off = np.repeat(fs[inc_f], rep)+(np.arange(rep.sum())-np.repeat(np.cumsum(rep)-rep, rep))
    cv = np.unique(inc_c[corner_inc].astype(np.int64)*nv+fv[off])
    ap_cell, ap_vert = cv//nv, cv % nv
    n_cv = np.bincount(ap_cell, minlength=nc)
    apex = np.zeros((nc, 3)); np.add.at(apex, ap_cell, X[ap_vert]); apex /= n_cv[:, None]
    # tets: one per (cell, face, edge)
    tc, tf, ts = inc_c[corner_inc], inc_f[corner_inc], inc_s[corner_inc]
    ta, tb = a_all[off], b_all[off]
    P0, P1, P2, P3 = apex[tc], fcen[tf], X[ta], X[tb]
    Dm = np.stack((P1-P0, P2-P0, P3-P0), 2)                  # columns
    det = np.linalg.det(Dm)
    V = ts*det/6
    keep = np.abs(det) > 1e-18
    Gi = np.linalg.inv(Dm[keep])                              # rows = gradients of barycentrics 1..3
    grads = np.concatenate((-Gi.sum(1, keepdims=True), Gi), 1)   # (t, 4, 3)
    tc, tf, ta, tb, V = tc[keep], tf[keep], ta[keep], tb[keep], V[keep]
    nt = len(V)
    pts = np.stack((nv+nf+tc, nv+tf, ta, tb), 1)              # point ids of the 4 tet points
    sp_id = np.repeat(np.arange(nt), 4); sp_pt = pts.ravel(); sp_g = grads.reshape(-1, 3)
    # lumped mass |t|/4 per point -> vertices
    pm = np.zeros(nv+nf+nc); np.add.at(pm, sp_pt, np.repeat(np.abs(V)/4, 4))
    vm = pm[:nv].copy()
    np.add.at(vm, fv, pm[nv+fid]/k[fid])
    np.add.at(vm, ap_vert, pm[nv+nf+ap_cell]/n_cv[ap_cell])
    # checks
    A = np.array([[1.1, .05, 0], [0, .95, .08], [.03, 0, 1.02]])
    Pt = np.concatenate((X, fcen, apex))@A.T
    F = np.zeros((nt, 3, 3)); np.add.at(F, sp_id, Pt[sp_pt][:, :, None]*sp_g[:, None, :])
    cellvol = np.bincount(tc, V, minlength=nc)
    used = np.unique(np.concatenate((fv, ap_vert)))
    h = np.cbrt(np.abs(V))
    summary = dict(cells=nc, tets=nt, dropped_degenerate=int((~keep).sum()), negative_tets=int((V <= 0).sum()),
                   volume_total=float(V.sum()), cell_volume_vs_pyramids=float(np.abs(cellvol.sum()/d['volume'].sum()-1)),
                   patch_max=float(np.abs(F-A).max()), tet_size_mm_quantiles=(np.quantile(h, [0, .01, .5])*1e3).tolist(),
                   point_rows=int(len(sp_id)), vertices=nv, vertices_used=int(len(used)), mass_zero_vertices=int((vm[used] <= 0).sum()))
    print(json.dumps(summary, indent=1))
    np.savez_compressed(args.out, X=X, sp_id=sp_id, sp_pt=sp_pt, sp_g=sp_g, tet_volume=np.abs(V), tet_cell=tc,
                        fc_face=fid, fc_vert=fv, fc_w=1/k[fid], ap_cell=ap_cell, ap_vert=ap_vert, ap_w=1/n_cv[ap_cell],
                        n_face=nf, n_cell=nc, vertex_volume=vm, cell_volume=cellvol, source_index=src)
    args.out.with_suffix('.json').write_text(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
