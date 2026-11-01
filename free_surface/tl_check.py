"""Static checks of the elastic end read from the rest shape (2026-11-01), a prototype beside merged_ul.py, nothing from it changed. F is a function of x, not a running sum:
  V F_i = sum_walls A x_w (x) n_out + x_i (x) a_i + Fe_i Mc_i (the rest cut; x_w = (1 - t) x_i + t x_j + 1/2 (Fe_i + Fe_j) tau; Fe the sites' fit, tiny cells their neighbours' mean, ul_plastic.py's gg on u = x - X)
  r_iw = x_w - x_i - F_i (c_w - X_i), one reading per cell per wall. Divergence theorem: sum_w A r_iw (x) n_out + (Fe_i - F_i) Mc_i = 0 at any x, so a pyramid gradient F_i + (beta / h) r (x) n has no cross term with F_i:
  E(x) = sum V psi(F_i) + beta^2 / 3 sum_iw (A / h) [mu / 2 |r|^2 + 1/2 (kap + mu / 3) (r . R_i n)^2] + cap term mu / 2 tr((Fe - F) Mc (Fe - F)^T)
psi = mu |dev U|^2 + kap / 2 ln^2 J. beta^2 = 3 is the value at which an orthogonal mesh gives dual linear FE's energy; h from the cell's centroid (the site can lie beyond a wall), weights only.
Checks: affine fields, the orthogonality, the patch test by depth, modes against the voxel FEM, clamped compression's negative modes (with the material's own ellipticity), large deformations released to see whether the rest shape comes back.
  python free_surface/tl_check.py <part foam dir with dem_exact_single and fem_modes_h0.0015.npz> <deep foam dir for the patch test>"""
import resource
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment, minimize
from scipy.sparse.linalg import LinearOperator, eigsh, splu

sys.path.insert(0, '/home/fangsuo/py/dynamic-powerfoam')
E_, NU = 23.00838, .3008
mu = E_/(2*(1+NU)); lam = E_*NU/((1+NU)*(1-2*NU)); kap = lam+2*mu/3


def coo(rows, cols, vals, shape):
    return sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=shape).tocsr()


def build(out, poor=0., tc=None):
    """poor: a cell whose ring sees a direction less than this (M / 2V's smallest eigenvalue, 1 = a full ring) lends its neighbours' fit in Fe, as a tiny cell does.
    tc: the wall's value read on the segment between the sites, t in [tc, 1 - tc], what is left over going through the fits with tau: a site beyond its own wall extrapolates otherwise; still exact on an affine field, the two sides still the same value"""
    g = np.load(out/'dem_exact_single'/'geometry.npz'); cl = np.load(out/'dem_exact_single'/'cells.npz')
    X, V, A, c, r0 = g['points'], g['masses'], g['areas'], g['contacts'], cl['radii']; n = len(X)
    i, j = g['pairs'][:, 0].astype(np.int64), g['pairs'][:, 1].astype(np.int64); m = len(i)
    d = X[j]-X[i]; L = np.linalg.norm(d, axis=1); nh = d/L[:, None]
    hi = ((c-X[i])*nh).sum(1); t = hi/L
    if tc is not None:
        t = np.clip(t, tc, 1-tc)
    tau = c-(X[i]+(t*L)[:, None]*nh)
    w = A/L
    q = w[:, None, None]*d[:, :, None]*d[:, None, :]
    M = np.zeros((n, 3, 3)); np.add.at(M, i, q); np.add.at(M, j, q)
    Mi = np.linalg.pinv(M)
    Ri = np.einsum('fa,fab->fb', w[:, None]*d, Mi[i]); Rj = np.einsum('fa,fab->fb', -w[:, None]*d, Mi[j])
    S_, O_, R_ = np.concatenate([i, j]), np.concatenate([j, i]), np.concatenate([Ri, Rj])
    rw, cl_, vl = [], [], []
    for a in range(3):
        for b in range(3):
            ga = 9*S_+3*a+b
            rw += [ga, ga]; cl_ += [3*O_+a, 3*S_+a]; vl += [R_[:, b], -R_[:, b]]
    B0 = coo(rw, cl_, vl, (9*n, 3*n))
    tiny = V < .1*np.median(V)
    if poor > 0:
        tiny = tiny | (np.linalg.eigvalsh(M)[:, 0]/(2*V) < poor)
    m1, m2 = tiny[i] & ~tiny[j], tiny[j] & ~tiny[i]
    as_, ao_ = np.concatenate([i[m1], j[m2]]), np.concatenate([j[m1], i[m2]])
    va = V[ao_]; den = np.zeros(n); np.add.at(den, as_, va)
    used = tiny & (den > 0); keep = np.where(~used)[0]
    rw = [9*np.repeat(keep, 9)+np.tile(np.arange(9), len(keep)), 9*np.repeat(as_, 9)+np.tile(np.arange(9), len(as_))]
    cl_ = [9*np.repeat(keep, 9)+np.tile(np.arange(9), len(keep)), 9*np.repeat(ao_, 9)+np.tile(np.arange(9), len(ao_))]
    vl = [np.ones(9*len(keep)), np.repeat(va/den[as_], 9)]
    Age = coo(rw, cl_, vl, (9*n, 9*n))
    a_ = np.zeros((n, 3)); np.add.at(a_, i, -A[:, None]*nh); np.add.at(a_, j, A[:, None]*nh)
    Mw = np.zeros((n, 3, 3))
    np.add.at(Mw, i, A[:, None, None]*(c-X[i])[:, :, None]*nh[:, None, :]); np.add.at(Mw, j, -A[:, None, None]*(c-X[j])[:, :, None]*nh[:, None, :])
    Mc = V[:, None, None]*np.eye(3)-Mw
    closed = np.linalg.norm(a_, axis=1) < 1e-9*np.median(A)
    Mc[closed] = 0.; a_[closed] = 0.
    # (the rest cut's walls count all; a cap cell's moment is r times the integral of N N, symmetric)
    rw, cl_, vl = [], [], []
    for a in range(3):
        for b in range(3):
            for s_, sg in ((i, 1.), (j, -1.)):
                gs = 9*s_+3*a+b
                rw += [gs, gs]; cl_ += [3*i+a, 3*j+a]; vl += [sg*A*(1-t)*nh[:, b], sg*A*t*nh[:, b]]
            gs = 9*np.arange(n)+3*a+b
            rw.append(gs); cl_.append(3*np.arange(n)+a); vl.append(a_[:, b])
    Bx = coo(rw, cl_, vl, (9*n, 3*n))
    rw, cl_, vl = [], [], []
    for a in range(3):
        for b in range(3):
            for k_ in range(3):
                for s_, sg in ((i, 1.), (j, -1.)):
                    gs = 9*s_+3*a+b
                    rw += [gs, gs]; cl_ += [9*i+3*a+k_, 9*j+3*a+k_]; vl += [.5*sg*A*tau[:, k_]*nh[:, b], .5*sg*A*tau[:, k_]*nh[:, b]]
                gs = 9*np.arange(n)+3*a+b
                rw.append(gs); cl_.append(9*np.arange(n)+3*a+k_); vl.append(Mc[:, k_, b])
    Bg_ = coo(rw, cl_, vl, (9*n, 9*n))
    Bg = (sp.diags(np.repeat(1/V, 9))@(Bx+Bg_@Age@B0)).tocsr()
    Ge = (Age@B0).tocsr()
    # the wall's value, one row of 3 per wall; each side's reading
    e_ = np.arange(m)
    rw, cl_, vl = [], [], []
    for a in range(3):
        rw += [3*e_+a, 3*e_+a]; cl_ += [3*i+a, 3*j+a]; vl += [1-t, t]
    W1 = coo(rw, cl_, vl, (3*m, 3*n))
    rw, cl_, vl = [], [], []
    for a in range(3):
        for k_ in range(3):
            rw += [3*e_+a, 3*e_+a]; cl_ += [9*i+3*a+k_, 9*j+3*a+k_]; vl += [.5*tau[:, k_], .5*tau[:, k_]]
    T9 = coo(rw, cl_, vl, (3*m, 9*n))
    Uw = (W1+T9@Ge).tocsr()

    def side(s_, cs):
        Is = sp.coo_matrix((np.ones(3*m), (np.arange(3*m), (3*s_[:, None]+np.arange(3)).ravel())), shape=(3*m, 3*n)).tocsr()
        rw, cl_, vl = [], [], []
        for a in range(3):
            for b in range(3):
                rw.append(3*e_+a); cl_.append(9*s_+3*a+b); vl.append(cs[:, b])
        Ds = coo(rw, cl_, vl, (3*m, 9*n))
        return (Uw-Is-Ds@Bg).tocsr()

    Ci, Cj = side(i, c-X[i]), side(j, c-X[j])
    # centroids: pyramids on the walls from the site, a spherical sector on the cap; heights from the centroid
    Cen = np.zeros((n, 3))
    np.add.at(Cen, i, (A*hi/4)[:, None]*(c-X[i])); np.add.at(Cen, j, (A*(L-hi)/4)[:, None]*(c-X[j]))
    a_full = np.zeros((n, 3)); np.add.at(a_full, i, -A[:, None]*nh); np.add.at(a_full, j, A[:, None]*nh)
    Cen = X+(Cen+(r0*r0/4)[:, None]*a_full)/V[:, None]
    hc_i = ((c-Cen[i])*nh).sum(1); hc_j = ((Cen[j]-c)*nh).sum(1)
    return dict(X=X, V=V, A=A, c=c, r0=r0, n=n, m=m, i=i, j=j, nh=nh, L=L, t=t, tau=tau, Bg=Bg, Ge=Ge, Ci=Ci, Cj=Cj, hc_i=hc_i, hc_j=hc_j, Mc=Mc, a=a_, closed=closed, B0=B0)


def block33(blocks, shape_n):
    """block diagonal of (k, 3, 3)"""
    k = len(blocks)
    ro = np.repeat(np.arange(k), 9)*3+np.tile(np.repeat(np.arange(3), 3), k)
    co = np.repeat(np.arange(k), 9)*3+np.tile(np.tile(np.arange(3), 3), k)
    return sp.coo_matrix((blocks.ravel(), (ro, co)), shape=(3*k, 3*k)).tocsr()


def stiff_parts(D):
    """K_r at beta^2 = 3 and K_cap, quadratic at a state with r = 0 and R = I"""
    nn = D['nh'][:, :, None]*D['nh'][:, None, :]
    I3 = np.eye(3)
    hi_, hj_ = np.maximum(D['hc_i'], 1e-12), np.maximum(D['hc_j'], 1e-12)
    Si = block33((D['A']/hi_)[:, None, None]*(mu*I3+(kap+mu/3)*nn), D['m']); Sj = block33((D['A']/hj_)[:, None, None]*(mu*I3+(kap+mu/3)*nn), D['m'])
    Kr = (D['Ci'].T@Si@D['Ci']+D['Cj'].T@Sj@D['Cj']).tocsr()
    n = D['n']
    Mc = .5*(D['Mc']+D['Mc'].transpose(0, 2, 1))
    blocks = np.zeros((n, 9, 9))
    for a in range(3):
        blocks[:, 3*a:3*a+3, 3*a:3*a+3] = mu*Mc
    ro = np.repeat(np.arange(n), 81)*9+np.tile(np.repeat(np.arange(9), 9), n)
    co = np.repeat(np.arange(n), 81)*9+np.tile(np.tile(np.arange(9), 9), n)
    Dcap = sp.coo_matrix((blocks.ravel(), (ro, co)), shape=(9*n, 9*n)).tocsr()
    Bc = (D['Ge']-D['Bg']).tocsr()
    Kcap = (Bc.T@Dcap@Bc).tocsr()
    return Kr, Kcap


def psi_P(F):
    """psi, P = dpsi/dF, with the proper rotation; J at or under 0.05 continues ln J linearly in J"""
    U, s, Vt = np.linalg.svd(F)
    det = np.linalg.det(U@Vt)
    s = s.copy(); s[:, 2] *= det; U = U.copy(); U[:, :, 2] *= det[:, None]
    R = U@Vt
    tr = s.sum(1)/3
    dev = s-tr[:, None]
    J = np.prod(s, axis=1)
    J0 = .05
    Ls = np.where(J > J0, np.log(np.maximum(J, 1e-300)), np.log(J0)+(J-J0)/J0)
    dL = np.where(J > J0, 1/np.where(J > J0, J, 1.), 1/J0)
    psi = mu*(dev*dev).sum(1)+.5*kap*Ls*Ls
    # cofactor from the singular values: d J / dF
    cof = np.stack([s[:, 1]*s[:, 2], s[:, 0]*s[:, 2], s[:, 0]*s[:, 1]], 1)
    Pd = 2*mu*dev+(kap*Ls*dL)[:, None]*cof
    P = (U*Pd[:, None, :])@Vt
    return psi, P, R, J


def main():
    resource.setrlimit(resource.RLIMIT_AS, (10*2**30, 10*2**30))                     # (CPU only here; importers on CUDA must not get it)
    part, deep = Path(sys.argv[1]), Path(sys.argv[2])
    D = build(part)
    X, V, A, n, m = D['X'], D['V'], D['A'], D['n'], D['m']
    h = float(np.median(V))**(1/3)
    print('%s: %d cells, %d walls, h %.2f mm; cap cells %d, inside %d' % (part.name, n, m, 1e3*h, int((~D['closed']).sum()), int(D['closed'].sum())), flush=True)
    t = D['t']; out_ = (t < 0) | (t > 1)
    print('walls outside the site segment %.1f %%; height from the centroid: both sides positive on %.1f %% of the walls; min %.2e h; A / h over A / (L / 2): median %.2f, 99%% %.2f, max %.1f' % (100*out_.mean(), 100*((D['hc_i'] > 0) & (D['hc_j'] > 0)).mean(), min(D['hc_i'].min(), D['hc_j'].min())/h, np.median(.5*D['L']/D['hc_i']), np.quantile(.5*D['L']/D['hc_i'], .99), (.5*D['L']/D['hc_i']).max()), flush=True)
    rng = np.random.default_rng(0)
    H_ = np.array([[.3, .1, -.2], [.05, -.4, .7], [.2, .6, .1]])
    u = (X@H_.T).ravel()
    print('affine field: F error %.1e, r_i %.1e, r_j %.1e, Fe - F %.1e' % (np.abs(D['Bg']@u-np.tile(H_.ravel(), n)).max(), np.abs(D['Ci']@u).max(), np.abs(D['Cj']@u).max(), np.abs((D['Ge']-D['Bg'])@u).max()), flush=True)
    # the orthogonality with a random field, cell by cell
    u = rng.normal(size=3*n)
    ri, rj = (D['Ci']@u).reshape(m, 3), (D['Cj']@u).reshape(m, 3)
    Sm = np.zeros((n, 3, 3))
    np.add.at(Sm, D['i'], A[:, None, None]*ri[:, :, None]*D['nh'][:, None, :]); np.add.at(Sm, D['j'], -A[:, None, None]*rj[:, :, None]*D['nh'][:, None, :])
    Gr = (D['Bg']@u).reshape(n, 3, 3); Gf = (D['Ge']@u).reshape(n, 3, 3)
    Sm = Sm+(Gf-Gr)@D['Mc']
    print('the orthogonality sum A r (x) n + (Fe - F) Mc, random field, relative to V |F|: largest %.1e' % (np.abs(Sm).max(axis=(1, 2))/(V*np.abs(Gr).max(axis=(1, 2))+1e-300)).max(), flush=True)
    # the cross term, on inside cells
    # modes
    Kr, Kcap = stiff_parts(D)
    Psym = np.zeros((9, 9)); I9 = np.eye(3).ravel()
    for a in range(3):
        for b in range(3):
            Psym[3*a+b, 3*a+b] += .5; Psym[3*a+b, 3*b+a] += .5
    Q = 2*mu*(Psym-np.outer(I9, I9)/3)+kap*np.outer(I9, I9)
    Dc = sp.block_diag([V[s]*Q for s in range(n)], format='csr')
    Kc = (D['Bg'].T@Dc@D['Bg']).tocsr()
    u_ = X@H_.T
    from experiments.dem_fem_mode_match import interpolate
    c_ = np.load(part/'fem_modes_h0.0015.npz'); nm = min(26, c_['V'].shape[1]); f_fem, V_fem = c_['f'][:nm], c_['V'][:, :nm]
    F_ = dict(nodes=c_['nodes'], node_grid=c_['node_grid'], lo=c_['lo'], h=float(c_['h']), masses=c_['masses'])
    Ux, _ = interpolate(F_, V_fem, X); UF = Ux.reshape(-1, nm)
    print('FEM first modes (Hz): %s' % np.array2string(f_fem[:6], precision=2), flush=True)
    Md = sp.diags(np.repeat(V, 3))
    w1 = (2*np.pi*f_fem[0])**2

    def report(name, K):
        t0 = time.time()
        sigma = -1e-3*w1
        lu = splu((K-sigma*Md).tocsc())
        OP = LinearOperator(K.shape, matvec=lu.solve, dtype=np.float64)
        lam_, vec = eigsh(K, k=70, M=Md, sigma=sigma, OPinv=OP, which='LM', tol=1e-8)
        o = np.argsort(lam_); lam_, vec = lam_[o], vec[:, o]
        nz = int((np.abs(lam_) < 1e-6*w1).sum())
        fr = np.sqrt(np.clip(lam_, 0, None))/(2*np.pi)
        Z = vec[:, nz:].reshape(n, 3, -1)
        num = np.einsum('n,nkd,nkf->df', V, Z, Ux)**2
        den = np.outer(np.einsum('n,nkd,nkd->d', V, Z, Z), np.einsum('n,nkf,nkf->f', V, Ux, Ux))
        Mac = num/den
        ri_, ci_ = linear_sum_assignment(-Mac)
        good = [fr[nz+a__]/f_fem[b__] for a__, b__ in zip(ri_, ci_) if Mac[a__, b__] >= .8]
        sf = np.array([UF[:, k]@(K@UF[:, k])/(UF[:, k]@(Md@UF[:, k]))/(2*np.pi*f_fem[k])**2 for k in range(nm)])
        print('  %-32s zeros %d, negative %d; first modes / FEM %s; along the FEM modes, frequency^2 / FEM median %.3f (%.3f..%.3f); paired %d of %d, frequency / FEM median %s; below half the FEM first %d (%.0f s)' % (
            name, nz, int((lam_ < -1e-6*w1).sum()), np.array2string(fr[nz:nz+6]/f_fem[:6], precision=3), np.median(sf), sf.min(), sf.max(), len(good), nm, '%.3f' % np.median(good) if good else '-', int(((lam_ > 1e-6*w1) & (fr < .5*f_fem[0])).sum()), time.time()-t0), flush=True)

    print('modes at rest:', flush=True)
    report('cells only', Kc)
    report('beta^2 = 1', (Kc+(Kr+Kcap)/3).tocsr())
    report('beta^2 = 2', (Kc+(Kr+Kcap)*2/3).tocsr())
    report('beta^2 = 3 (FE on orthogonal)', (Kc+Kr+Kcap).tocsr())
    report('beta^2 = 3, no cap term', (Kc+Kr).tocsr())
    # clamped compression: cap cells held, the inside free, a uniform state; constant, strongly elliptic moduli give no negative mode on a conforming mesh
    print('clamped compression (inside %d cells free): dense eigenvalues in the FEM first mode^2' % int(D['closed'].sum()), flush=True)
    free = np.where(np.repeat(D['closed'], 3))[0]
    Mf = np.repeat(V, 3)[free]

    def tangent(F0):
        Fv = F0.ravel(); T = np.zeros((9, 9)); e = 1e-6
        for k in range(9):
            dF = np.zeros(9); dF[k] = e
            T[:, k] = (psi_P((Fv+dF).reshape(1, 3, 3))[1].ravel()-psi_P((Fv-dF).reshape(1, 3, 3))[1].ravel())/(2*e)
        return .5*(T+T.T)

    dirs = rng.normal(size=(2000, 3)); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    for name, F0 in (('uniaxial 0.8', np.diag([1., 1., .8])), ('uniaxial 0.5', np.diag([1., 1., .5])), ('uniaxial 0.3', np.diag([1., 1., .3])), ('J 0.5', np.eye(3)*.5**(1/3)), ('tension 1.2', np.diag([1., 1., 1.2]))):
        T = tangent(F0)
        ac = 1e30
        for nv in dirs:
            N3 = np.zeros((3, 9))
            for a in range(3):
                N3[a, 3*a:3*a+3] = nv
            ac = min(ac, np.linalg.eigvalsh(N3@T@N3.T)[0])
        DT = sp.block_diag([V[s]*T for s in range(n)], format='csr')
        KT = (D['Bg'].T@DT@D['Bg']).tocsr()
        out = []
        for nm_, K in (('cells', KT), ('beta^2 1', (KT+(Kr+Kcap)/3).tocsr()), ('beta^2 3', (KT+Kr+Kcap).tocsr())):
            Kf = K[free][:, free].toarray()
            Kf = .5*(Kf+Kf.T)/np.sqrt(Mf[:, None]*Mf[None, :])
            ev = np.linalg.eigvalsh(Kf)
            out.append('%s: negative %d, lowest %.1f' % (nm_, int((ev < -1e-6*w1).sum()), ev[0]/w1))
        print('  %-14s material: lowest eigenvalue of the 9 x 9 tangent / mu %.2f, acoustic tensor min / mu %.2f; %s' % (name, np.linalg.eigvalsh(T)[0]/mu, ac/mu, '; '.join(out)), flush=True)
    # the patch test on a foam with an inside: a uniform stress, forces by depth
    Dd = build(deep)
    nd, id_, jd = Dd['n'], Dd['i'], Dd['j']
    depth = np.full(nd, -1); depth[~Dd['closed']] = 0
    Ad = sp.coo_matrix((np.ones(2*len(id_), dtype=bool), (np.r_[id_, jd], np.r_[jd, id_])), shape=(nd, nd)).tocsr()
    front = depth == 0; k_ = 0
    while (depth < 0).any() and front.any():
        k_ += 1
        nxt = (Ad@front.astype(np.int64)) > 0
        nxt &= depth < 0
        depth[nxt] = k_; front = nxt
    Fu = np.array([[1.2, .1, 0.], [0., .9, .05], [.02, 0., .95]])
    P = psi_P(Fu[None])[1][0]
    fp = (Dd['Bg'].T@((Dd['V'][:, None]*P.ravel()[None, :]).ravel())).reshape(nd, 3)
    Am = float(np.median(Dd['A'])); nP = float(np.linalg.norm(P))
    print('patch test on %s, %d cells: |force| / (|P| median wall), max by depth from the cap cells: %s' % (deep.name, nd, ', '.join('%d: %.1e (%d)' % (k__, np.linalg.norm(fp[depth == k__], axis=1).max()/(nP*Am), int((depth == k__).sum())) for k__ in range(min(k_+1, 7)))), flush=True)
    # released from large deformations
    print('released from a large deformation, no gravity, nothing holding it:', flush=True)
    Xc = X-(V[:, None]*X).sum(0)/V.sum()
    ax = np.argmax(Xc.max(0)-Xc.min(0)); S = float(np.abs(Xc[:, ax]).max())
    Bg, Ci, Cj, Ge = D['Bg'], D['Ci'], D['Cj'], D['Ge']
    Bc = (Ge-Bg).tocsr(); Mcs = .5*(D['Mc']+D['Mc'].transpose(0, 2, 1))
    hi_, hj_ = np.maximum(D['hc_i'], 1e-12), np.maximum(D['hc_j'], 1e-12)
    nh_, i_, j_ = D['nh'], D['i'], D['j']
    Xf = X.ravel()

    def energy(xv):
        uv = xv-Xf
        G = (Bg@uv).reshape(n, 3, 3)
        F = G+np.eye(3)
        psi, P, R, J = psi_P(F)
        E = float((V*psi).sum())
        g_ = Bg.T@((V[:, None, None]*P).ravel())
        for Cs, s_, hs in ((Ci, i_, hi_), (Cj, j_, hj_)):
            rv = Cs@uv; r = rv.reshape(m, 3)
            nw = np.einsum('fab,fb->fa', R[s_], nh_)                                  # (the cell's rotated normal; the gradient leaves out its change, so L-BFGS can stop early: site noise did, with R frozen per round it returns too)
            rn = (r*nw).sum(1)
            w_ = A/hs
            E += float((w_*(.5*mu*(r*r).sum(1)+.5*(kap+mu/3)*rn*rn)).sum())
            g_ = g_+Cs.T@((w_[:, None]*(mu*r+(kap+mu/3)*rn[:, None]*nw)).ravel())
        Bv = (Bc@uv).reshape(n, 3, 3)
        E += float((.5*mu*(Bv@Mcs*Bv).sum((1, 2))).sum())
        g_ = g_+Bc.T@((mu*Bv@Mcs).ravel())
        return E, g_, J

    def rigid_rms(xv):
        x = xv.reshape(n, 3); a = x-(V[:, None]*x).sum(0)/V.sum(); b = Xc
        U_, _, Vt_ = np.linalg.svd((V[:, None]*b).T@a)
        Rr = U_@np.diag([1, 1, np.linalg.det(U_@Vt_)])@Vt_
        dd = a-b@Rr
        return float(np.sqrt((V*(dd*dd).sum(1)).sum()/V.sum())), float(np.linalg.norm(dd, axis=1).max())

    def twist(p, ang):
        s = p[:, ax]/S; th = ang*s
        o1, o2 = [k__ for k__ in range(3) if k__ != ax]
        q_ = p.copy()
        q_[:, o1] = np.cos(th)*p[:, o1]-np.sin(th)*p[:, o2]; q_[:, o2] = np.sin(th)*p[:, o1]+np.cos(th)*p[:, o2]
        return q_

    def bend(p, ang):
        o1 = [k__ for k__ in range(3) if k__ != ax][0]
        kappa = ang/S
        q_ = p.copy()
        th = kappa*p[:, ax]; rr = 1/kappa-p[:, o1]
        q_[:, ax] = rr*np.sin(th); q_[:, o1] = 1/kappa-rr*np.cos(th)
        return q_

    cases = [('twist 90 deg', twist(Xc, np.pi/2)), ('bend 120 deg', bend(Xc, 2*np.pi/3)), ('squash to 0.5 along the short axis', Xc*np.where(np.arange(3) == np.argmin(Xc.max(0)-Xc.min(0)), .5, 1.2)),
             ('twist and bend', bend(twist(Xc, np.pi/2), 2*np.pi/3)), ('twist and bend + site noise 0.2 h', bend(twist(Xc, np.pi/2), 2*np.pi/3)+.2*h*rng.normal(size=Xc.shape)), ('site noise 0.5 h alone', Xc+.5*h*rng.normal(size=Xc.shape))]
    for name, x0 in cases:
        xv = (x0+(V[:, None]*X).sum(0)/V.sum()).ravel()
        E0, g0, J0 = energy(xv); rm0, mx0 = rigid_rms(xv)
        t0 = time.time()
        res = minimize(lambda z: energy(z)[:2], xv, jac=True, method='L-BFGS-B', options=dict(maxiter=20000, maxfun=40000, ftol=1e-16, gtol=1e-9*np.linalg.norm(g0, np.inf), maxcor=40))
        Ee, ge, Je = energy(res.x); rme, mxe = rigid_rms(res.x)
        print('  %-36s start: E %.3e, off rest rms %.2f h, max %.2f h, inverted %d, J min %.2f; end: E / start %.2e, off rest rms %.1e h, max %.1e h, inverted %d, J %.3f..%.3f; %d evaluations, %.0f s' % (name, E0, rm0/h, mx0/h, int((J0 <= 0).sum()), J0.min(), Ee/E0, rme/h, mxe/h, int((Je <= 0).sum()), Je.min(), Je.max(), res.nfev, time.time()-t0), flush=True)
    print('done', flush=True)


if __name__ == '__main__':
    main()
