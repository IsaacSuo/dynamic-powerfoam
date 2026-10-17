"""Are there soft (near-hourglass) modes in the SFVEM elements as simulated?

For sampled elements: linear-elastic stiffness with (a) the runtime quadrature (one point per
boundary-triangle cone) and (b) exact polynomial integration (the projection's own mass matrix),
lumped element mass (runtime), generalized eigenvalues omega^2. Scaled frequency
omega * h / c_s (h = |E|^(1/3), c_s = sqrt(mu / rho)): continuum deformation modes at the element
scale are O(1) or above; << 1 means a mode far softer than the material.
"""
import json
import numpy as np
from experiments.sfvem_geometry import degree_for, element_data
from experiments.vem_geometry import merge_small_cells

E_, NU, RHO = 23008.4, .30079, 1000.
LAM, MU = E_*NU/((1+NU)*(1-2*NU)), E_/(2*(1+NU))
CS = np.sqrt(MU/RHO)


def stiffness(Rk, Phi_w_M, quad):
    """Rk (nb, 3, n). quad=True: Phi_w_M = (Phi (q, nb), w (q)); else the exact M (nb, nb)."""
    nb, _, n = Rk.shape
    B = np.zeros((nb, 3, 3, n, 3))                      # d C_k[i, j] / d u_{v, i'}
    for i in range(3):
        B[:, i, :, :, i] = Rk
    Bs = .5*(B+B.transpose(0, 2, 1, 3, 4)).reshape(nb, 9, n*3)
    tr = (Bs[:, 0]+Bs[:, 4]+Bs[:, 8])                   # (nb, n*3)
    if quad:
        Phi, w = Phi_w_M
        G = (Phi*w[:, None]).T@Phi                      # quadrature Gram matrix
    else:
        G = Phi_w_M
    K = LAM*tr.T@G@tr
    for a in range(9):
        K += 2*MU*Bs[:, a].T@G@Bs[:, a]
    return .5*(K+K.T)


def spectrum(K, m):
    s = 1/np.sqrt(np.repeat(m, 3))
    w2 = np.linalg.eigvalsh(K*s[:, None]*s[None, :])
    return np.sqrt(np.clip(w2, 0, None)), w2


def main():
    d = np.load('output/pacnerf_vem/elastic_0_f0/vem_cap1_bodies_pyr.npz'); cells = dict(np.load('output/pacnerf_dem/elastic_0_f0/cells.npz'))
    X, fv, fs, fp, src = d['X'], d['face_vertices'], d['face_start'], d['face_pair'], d['source_index']
    k = np.diff(np.r_[fs, len(fv)])
    group, _ = merge_small_cells(cells['masses'], cells['pairs'], cells['areas'], np.zeros(len(src), int), .1)
    gid = {int(s): int(group[i]) for i, s in enumerate(src)}
    faces = [[] for _ in range(group.max()+1)]
    for f_, (i, j) in enumerate(fp):
        gi, gj = gid.get(int(i), -1), gid.get(int(j), -1)
        if gi == gj:
            continue
        if gi >= 0: faces[gi].append((f_, 1.))
        if gj >= 0: faces[gj].append((f_, -1.))
    rng = np.random.default_rng(1)
    rows = []
    for e in rng.choice(len(faces), 60, replace=False):
        tris = []
        for f_, s_ in faces[e]:
            vs = fv[fs[f_]:fs[f_]+k[f_]]
            for t in range(len(vs)):
                tris.append((int(vs[t]), int(vs[(t+1) % len(vs)]), vs, s_))
        n = len({v for a, b, fvs, _ in tris for v in (a, b, *fvs)})
        l = degree_for(n)
        r = element_data(X, tris, l)
        h = r['volume']**(1/3)
        m = RHO*r['mass']
        out = dict(element=int(e), degree=l, n=n, h_mm=h*1e3, negative_weights=int((r['qw'] < 0).sum()))
        for name, K in (('quad', stiffness(r['R'], (r['phi_q'], np.maximum(r['qw'], 0)), True)), ('exact', stiffness(r['R'], r['M'], False))):
            w, w2 = spectrum(K, m)
            scaled = np.sort(w*h/CS)
            nonrigid = scaled[6:]
            out[name] = dict(zero_modes_beyond_rigid=int((w2 < 1e-10*w2.max()).sum())-6,
                             min=float(nonrigid.min()), p05=float(np.quantile(nonrigid, .05)), median=float(np.median(nonrigid)),
                             max=float(nonrigid.max()), below_0p3=int((nonrigid < .3).sum()), below_1=int((nonrigid < 1).sum()))
            if name == 'quad':
                # localisation of the softest non-rigid mode: inverse participation ratio over vertices
                vals, vecs = np.linalg.eigh(K/np.sqrt(np.outer(np.repeat(m, 3), np.repeat(m, 3))))
                v = vecs[:, 6].reshape(-1, 3); a2 = (v**2).sum(1)
                out['softest_mode_ipr'] = float(a2.sum()**2/(len(a2)*(a2**2).sum()))
        rows.append(out)
        print(json.dumps(out), flush=True)
    summ = {}
    for name in ('quad', 'exact'):
        summ[name] = dict(elements_with_zero_modes=int(sum(r_[name]['zero_modes_beyond_rigid'] > 0 for r_ in rows)),
                          min_scaled_freq_median=float(np.median([r_[name]['min'] for r_ in rows])),
                          min_scaled_freq_min=float(min(r_[name]['min'] for r_ in rows)),
                          elements_with_mode_below_0p3=int(sum(r_[name]['below_0p3'] > 0 for r_ in rows)),
                          modes_below_1_mean=float(np.mean([r_[name]['below_1'] for r_ in rows])))
    print('SUMMARY', json.dumps(summ, indent=1), flush=True)
    json.dump(dict(rows=rows, summary=summ), open('output/pacnerf_vem/elastic_0_f0/sfvem_spectrum_v1.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
