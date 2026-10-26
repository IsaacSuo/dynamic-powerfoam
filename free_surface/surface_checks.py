"""Checks of the continuous-surface model (surface_model.SurfaceTerms), step 2 of free_surface/PLAN.md, offline.

Cube lattice 8^3 (face_model.lattice; its surface: one square per outer cell face, vertices at the cube corners):
  1. rest: energy and gradient zero; rigid motion (40 deg + shift): energy zero;
  2. affine maps (random, amplitude 1e-3 and 0.2): F_B exact, energy = the current model's (MidpointTerms, exact too);
  3. patch test: vertices held on an affine map, bodies free: residual forces on the bodies (current model: its surface
     bodies' residual at the same state);
  4. free surface, uniaxial stress -1 % / -20 % (sigma . n = 0 on the four sides): (a) at the uniform state
     diag(a, a, 1 + e) (a from the minimum over affine maps) the forces on everything but the top and bottom vertices,
     against the current model's on its side bodies; (b) top / bottom vertices held in z only, all else free, energy
     minimised from a = 1: uniform again? (affine fit residual, energy against the uniform state);
  5. Hessian at rest (dense): zero eigenvalues of the whole (6 rigid expected) and of the vertex block, with and
     without the surface term.
Real shapes (new: pacnerf_dem_cutft, old: pacnerf_dem8k_hull scene 0):
  1. mesh vertices in the DEM frame; mesh faces against the DEM's air faces (site pairs, areas, h);
  2. h, centroid weights, in-plane offsets t, gradient moment (M + M_face) / V, volume residue V_res / V;
  3. rest / rigid / affine as above; patch test against the current model;
  4. vertex block of the surface term (sparse, exact: G^T W G per coordinate): smallest eigenvalues. At rest every
     other term's vertex block is positive semi-definite, so this bounds the whole vertex block from below."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
from scipy.optimize import minimize_scalar

from experiments.foam_surface_completion import SurfaceCompletion
from experiments.pacnerf_dem_forward import build
from free_surface.grad_weights import MidpointTerms
from free_surface.surface_model import SurfaceTerms

f64, dev = torch.float64, 'cuda'


def rot(deg, axis):
    k = np.asarray(axis, float); k /= np.linalg.norm(k); t = np.radians(deg)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return torch.as_tensor(np.eye(3)+np.sin(t)*K+(1-np.cos(t))*K@K, dtype=f64, device=dev)


def grads(st, th, x, y):
    x = x.clone().requires_grad_(True); y = y.clone().requires_grad_(True)
    E = st.energy_xy(th, x, y)
    gx, gy = torch.autograd.grad(E, (x, y))
    return float(E), gx, gy


def common(name, st, mid, th, c, size):
    """rest, rigid, affine; returns the energy scale (affine 1e-3)"""
    X, Y = st.X, st.Y
    gen = torch.Generator(device='cpu').manual_seed(0)
    aff = {}
    for amp in (1e-3, .2):
        Fa = torch.eye(3, dtype=f64, device=dev)+amp*torch.randn(3, 3, generator=gen, dtype=f64).to(dev)
        x, y = st.affine(Fa, c)
        T = {k: float(v) for k, v in st.energy_terms(th, x, y).items()}
        E_m = float(mid.energy(th, x, smooth_polar=True)[0])
        aff[amp] = (float((st.F(x, y)-Fa).abs().max()), sum(T.values()), E_m, T)
    Es = aff[1e-3][1]
    E0, gx0, gy0 = grads(st, th, X, Y)
    _, gxa, gya = grads(st, th, *st.affine(torch.eye(3, dtype=f64, device=dev)+1e-3*torch.randn(3, 3, generator=gen, dtype=f64).to(dev), c))
    ga = float(torch.cat((gxa, gya)).norm(dim=1).max())                   # largest force of a 1e-3 strain (vertex reactions)
    print('[%s] rest: |F(X, Y) - I| max %.1e, energy %.1e of a 1e-3 strain, |grad| max bodies %.1e vertices %.1e (of the largest force of a 1e-3 strain)' % (
        name, float((st.F(X, Y)-torch.eye(3, dtype=f64, device=dev)).abs().max()), E0/Es, float(gx0.norm(dim=1).max())/ga,
        float(gy0.norm(dim=1).max())/ga), flush=True)
    Q = rot(40, [1, 2, 3]); t = torch.tensor([.1, .2, -.05], dtype=f64, device=dev)*size
    print('[%s] rigid (40 deg + shift): energy %.1e of a 1e-3 strain' % (name, float(st.energy_xy(th, *st.affine(Q, c, t)))/Es), flush=True)
    for amp, (dF, E, E_m, T) in aff.items():
        print('[%s] affine %g: |F_B - F| max %.1e | energy / current model - 1 = %.1e | parts %s' % (
            name, amp, dF, E/E_m-1, {k: '%.3e' % (v/E) for k, v in T.items()}), flush=True)
    return Es


def patch(name, st, mid, th, c, surf_bodies, force_scale):
    """vertices held on an affine map, bodies free: residual forces on the bodies"""
    a = torch.tensor([[.01, .004, 0], [.004, -.006, .003], [0, .003, .008]], dtype=f64, device=dev)
    Fa = torch.eye(3, dtype=f64, device=dev)+a
    x, y = st.affine(Fa, c)
    _, gx, _ = grads(st, th, x, y)
    xm = x.clone().requires_grad_(True)
    gm, = torch.autograd.grad(mid.energy(th, xm, smooth_polar=True)[0], xm)
    s = surf_bodies
    print('[%s] patch test (vertices held affine, strain ~1e-2): body residual max / (|sigma| x face area): interior %.1e (current model %.1e), '
          'surface bodies %.1e (current model %.1e)' % (name, float(gx[~s].norm(dim=1).max())/force_scale, float(gm[~s].norm(dim=1).max())/force_scale,
                                                         float(gx[s].norm(dim=1).max())/force_scale, float(gm[s].norm(dim=1).max())/force_scale), flush=True)


def surface_block(st):
    """vertex block of the surface term at fixed bodies: 2 G^T W G per coordinate (delta = G y + const, scalar weights)"""
    c = lambda t: t.detach().cpu().numpy()
    fb, fw, r, Mi = c(st.fb), c(st.fw), c(st.r), c(st.Mt_inv)
    ev_, eb, d, w = c(st.ev), c(st.s_body), c(st.s_d), c(st.s_w)
    E, Fn, V = len(ev_), st.nf, st.nv
    order = np.argsort(fb, kind='stable'); cnt = np.bincount(fb, minlength=st.n); bfo = np.concatenate(([0], np.cumsum(cnt)))
    ce = cnt[eb]; e_rep = np.repeat(np.arange(E), ce)
    within = np.arange(ce.sum())-np.repeat(np.cumsum(ce)-ce, ce)
    f_rep = order[np.repeat(bfo[eb], ce)+within]
    coef = fw[f_rep]*np.einsum('ri,rij,rj->r', r[f_rep], Mi[eb[e_rep]], d[e_rep])
    C = sp.csr_matrix((coef, (e_rep, f_rep)), shape=(E, Fn))
    Bf = sp.csr_matrix((c(st.ew), (c(st.ef), c(st.ev))), shape=(Fn, V))
    G = sp.csr_matrix((np.ones(E), (np.arange(E), ev_)), shape=(E, V))-C@Bf
    K = (G.T@sp.diags(2*w)@G).tocsc()
    dg = K.diagonal(); md = float(np.median(dg))
    lo = spla.eigsh(K, k=6, sigma=-1e-9*md, which='LM', return_eigenvectors=False)
    hi = spla.eigsh(K, k=1, which='LA', return_eigenvectors=False)
    return np.sort(lo)/md, float(hi[0])/md, float(dg.min())/md


# ---------------------------------------------------------------- lattice
def lattice_mesh(idx, hl, N):
    keys, Yl, off, fvl, fbl = {}, [], [0], [], []
    for B, ijk in enumerate(idx):
        for a in range(3):
            for s in (-1, 1):
                if (s < 0 and ijk[a] > 0) or (s > 0 and ijk[a] < N-1):
                    continue
                b, cc = [e for e in range(3) if e != a]
                ring = []
                for sb, scc in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                    k = 2*np.array(ijk); k[a] += s; k[b] += sb; k[cc] += scc
                    t = tuple(int(v) for v in k)
                    if t not in keys:
                        keys[t] = len(Yl); Yl.append(np.array(t)*hl/2)
                    ring.append(keys[t])
                P = np.array([Yl[r] for r in ring])
                if np.cross(P[1]-P[0], P[2]-P[0])[a]*s < 0:
                    ring = ring[::-1]
                fvl += ring; off.append(len(fvl)); fbl.append(B)
    fv = np.array(fvl); off = np.array(off)
    nxt = np.concatenate([np.roll(fv[a:b], -1) for a, b in zip(off[:-1], off[1:])])
    dk = set(zip(fv.tolist(), nxt.tolist()))
    closed = all((b, a) in dk for a, b in dk) and len(dk) == len(fv)
    return dict(Y=np.array(Yl), face_off=off, face_vert=fv, face_body=np.array(fbl)), closed


def dense_hessian(f, n, chunk=64):
    z0 = torch.zeros(n, dtype=f64, device=dev)
    g = torch.func.grad(f)
    hvp = lambda v: torch.func.jvp(g, (z0,), (v,))[1]
    H = torch.zeros(n, n, dtype=f64, device=dev); eye = torch.eye(n, dtype=f64, device=dev)
    for s in range(0, n, chunk):
        H[s:s+chunk] = torch.func.vmap(hvp)(eye[s:s+chunk])
    return .5*(H+H.T)


def relax(st, th, x0, y0, fix_y, scale, iters):
    zx = torch.zeros_like(x0, requires_grad=True); zy = torch.zeros_like(y0, requires_grad=True)
    my = (~fix_y).to(f64)
    opt = torch.optim.LBFGS([zx, zy], lr=1, max_iter=iters, tolerance_grad=1e-30, tolerance_change=1e-30, history_size=100, line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad()
        E = st.energy_xy(th, x0+zx, y0+zy*my)/scale
        E.backward()
        return E
    opt.step(closure)
    x, y = (x0+zx).detach(), (y0+zy*my).detach()
    _, gx, gy = grads(st, th, x, y)
    return x, y, max(float(gx.norm(dim=1).max()), float((gy*my).norm(dim=1).max()))


def lattice_checks(iters):
    from prototypes.kb_derivation.face_model import lattice
    m, sc, g, idx = lattice()
    N, hl, mu, lam = 8, .02, float(m.mu_ref), float(m.k_v)
    mesh, closed = lattice_mesh(idx, hl, N)
    st = SurfaceTerms(m, sc, g, mesh); mid = MidpointTerms(m, sc, g)
    th = torch.zeros(m.n, 9, dtype=f64, device=dev)
    X, Y = st.X, st.Y; c = X.mean(0); size = float(N*hl)
    print('\n[lattice] bodies %d, faces %d, vertices %d, closed %s; h %.4f..%.4f, in-plane offset |t| max %.1e' % (
        m.n, st.nf, st.nv, closed, float(st.fh.min()), float(st.fh.max()), float(st.rt.norm(dim=1).max())), flush=True)
    common('lattice', st, mid, th, c, size)
    bnd = torch.as_tensor(((idx == 0) | (idx == N-1)).any(1), device=dev)
    young = mu*(3*lam+2*mu)/(lam+mu)
    patch('lattice', st, mid, th, c, bnd, .016*young*hl*hl)
    # 4. uniaxial stress, free sides
    tb_body = torch.as_tensor((idx[:, 2] == 0) | (idx[:, 2] == N-1), device=dev)
    zlo, zhi = float(Y[:, 2].min()), float(Y[:, 2].max())
    tb_vert = (Y[:, 2] - zlo).abs() < 1e-12
    tb_vert = tb_vert | ((Y[:, 2]-zhi).abs() < 1e-12)
    side_body = bnd & ~tb_body
    for e in (-.01, -.2):
        Fd = lambda a_: torch.diag(torch.tensor([a_, a_, 1+e], dtype=f64, device=dev))
        res = minimize_scalar(lambda a_: float(st.energy_xy(th, *st.affine(Fd(a_), c))), bounds=(1., 1.+abs(e)), method='bounded', options=dict(xatol=1e-13))
        a = float(res.x)
        x, y = st.affine(Fd(a), c)
        E_u, gx, gy = grads(st, th, x, y)
        xm = x.clone().requires_grad_(True)
        gm, = torch.autograd.grad(mid.energy(th, xm, smooth_polar=True)[0], xm)
        fs = abs(young*e)*hl*hl
        print('[lattice] uniaxial %+.0f%% (a = %.6f, nu_eff %.4f): at the uniform state, force / (|E e| x face area): new model side bodies %.1e, interior %.1e, '
              'top/bottom bodies %.1e, free vertices %.1e | current model side bodies %.1e, interior %.1e' % (
                  100*e, a, -(a-1)/e, float(gx[side_body].norm(dim=1).max())/fs, float(gx[~bnd].norm(dim=1).max())/fs, float(gx[tb_body].norm(dim=1).max())/fs,
                  float(gy[~tb_vert].norm(dim=1).max())/fs, float(gm[side_body].norm(dim=1).max())/fs, float(gm[~bnd].norm(dim=1).max())/fs), flush=True)
        fix = torch.zeros_like(Y, dtype=torch.bool); fix[tb_vert, 2] = True
        t0 = time.time()
        x0, y0 = st.affine(Fd(1.), c)
        xr, yr, gres = relax(st, th, x0, y0, fix, E_u, iters)
        P0 = torch.cat((X, Y)); P1 = torch.cat((xr, yr))
        Aug = torch.cat((P0-c, torch.ones(len(P0), 1, dtype=f64, device=dev)), 1)
        sol = torch.linalg.lstsq(Aug, P1-c).solution
        fit = Aug@sol
        Ff = sol[:3].T
        half = size/2
        E_r = float(st.energy_xy(th, xr, yr))
        print('[lattice] uniaxial %+.0f%%: free relaxation from a = 1 (%.0f s, residual grad %.1e of the uniform state\'s top force): energy / uniform - 1 = %.1e, '
              'affine fit residual max %.1e of the lateral displacement, fitted F - diag(a, a, 1 + e) max %.1e' % (
                  100*e, time.time()-t0, gres/float(gy[tb_vert].norm(dim=1).max()), E_r/E_u-1, float((P1-c-fit).norm(dim=1).max())/((a-1)*half),
                  float((Ff-Fd(a)).abs().max())), flush=True)
    # 5. Hessian at rest
    n3, v3 = 3*m.n, 3*st.nv
    st0 = SurfaceTerms(m, sc, g, mesh, surface_term=False)
    for tag, s_ in (('with surface term', st), ('without', st0)):
        H = dense_hessian(lambda z: s_.energy_xy(th, X+z[:n3].view(-1, 3), Y+z[n3:].view(-1, 3)), n3+v3)
        ev = torch.linalg.eigvalsh(H); top = float(ev[-1])
        evy = torch.linalg.eigvalsh(H[n3:, n3:]); topy = float(evy[-1])
        print('[lattice] Hessian at rest %s: whole: eigenvalues < 1e-10 of the largest: %d (6 rigid expected), next %.1e | vertex block: '
              'smallest %.1e of its largest, count < 1e-10: %d' % (tag, int((ev < 1e-10*top).sum()), float(ev[6])/top, float(evy[0])/topy,
                                                                   int((evy < 1e-10*topy).sum())), flush=True)
    lo, hi, dmin = surface_block(st)
    print('[lattice] surface-term vertex block (sparse): smallest eigenvalues / median diagonal %s, largest %.2f, smallest diagonal %.2f' % (
        np.array2string(lo[:3], precision=3), hi, dmin), flush=True)


# ---------------------------------------------------------------- real shapes
def real_checks(name, dem_z, run_dir, mesh_dir):
    t0 = time.time()
    g = dict(np.load(dem_z/'geometry.npz'))
    m = build(g, 23008.38, .3008, dev, 'noslip', .25, .3, f64, kinematics='exact', nodalvol=True, nonlinear=True, ogden_alpha=2.)
    sc = SurfaceCompletion(run_dir, dem_z, m, verbose=False)
    S = np.load(mesh_dir/'surface_mesh.npz')
    fr = sc.frame
    Yd = fr['scale']*S['vert_foam']@fr['rotation'].T+fr['shift']
    msg = ''
    if 'vert_z' in S.files:
        msg = ', vs the stored z-up copy %.1e m' % np.abs(Yd-S['vert_z']).max()
    mesh = dict(Y=Yd, face_off=S['face_off'], face_vert=S['face_vert'], face_body=S['face_body'])
    st = SurfaceTerms(m, sc, g, mesh); mid = MidpointTerms(m, sc, g)
    th = torch.zeros(m.n, 9, dtype=f64, device=dev)
    print('\n[%s] bodies %d, faces %d, vertices %d; setup %.0f s; mesh in the DEM frame (scale %.6f)%s' % (name, m.n, st.nf, st.nv, time.time()-t0, fr['scale'], msg), flush=True)
    # 1. against the DEM's air faces
    cells = np.load(dem_z/'cells.npz'); src = cells['source_index']
    stt = torch.load(run_dir/'model.pt', map_location='cpu', weights_only=True)
    n_real = int(stt['num_sites']) if 'num_sites' in stt else len(stt['points'])
    N = len(stt['points'])+(len(cells['ghost_points']) if 'ghost_points' in cells.files else 0)
    pm = src[S['face_cell']].astype(np.int64)*N+S['face_air']
    pr = sc._surface_args['pairs']; pd = pr[:, 0].astype(np.int64)*N+pr[:, 1]
    A_d = np.asarray(sc.air_faces['area']); A_m = st.fA.cpu().numpy()
    both, im, idd = np.intersect1d(pm, pd, return_indices=True)
    only_d = np.setdiff1d(np.arange(len(pd)), idd); only_m = np.setdiff1d(np.arange(len(pm)), im)
    nair = len(A_d)
    L_ = sc.vs_lever[:nair].norm(dim=1).cpu().numpy()
    h_d = sc.vs_k[:nair].cpu().numpy()*L_**2/(2*float(m.mu_ref)*A_d)
    ncell = np.bincount(g['body_of_cell'], minlength=m.n)
    single = (ncell[S['face_body'][im]] == 1) & (h_d[idd] > 0)
    h_m = st.fh.cpu().numpy()
    print('[%s] faces vs the DEM air faces: matched %d; DEM only %d (area %.2e m^2 = %.1e of the surface; air site a mirror point %d, a real site %d); mesh only %d | '
          'matched area |rel diff| median %.1e max %.1e | h on single-cell bodies |diff| max %.1e m' % (
              name, len(both), len(only_d), A_d[only_d].sum(), A_d[only_d].sum()/A_d.sum(), int((pr[only_d, 1] >= n_real).sum()), int((pr[only_d, 1] < n_real).sum()),
              len(only_m), np.median(np.abs(A_m[im]/A_d[idd]-1)), np.abs(A_m[im]/A_d[idd]-1).max(), np.abs(h_m[im][single]-h_d[idd][single]).max()), flush=True)
    # 2. geometry statistics
    hh = h_m*1e3
    tr = (st.rt.norm(dim=1)/st.r.norm(dim=1)).cpu().numpy()
    et = torch.linalg.eigvalsh(st.M+st.Mface)/st.V[:, None]
    surf = torch.zeros(m.n, dtype=torch.bool, device=dev); surf[st.fb] = True
    vr = (st.Vres/st.V).abs().cpu().numpy()
    print('[%s] h (mm): min %.3f, <= 0: %d, < 0.1 mm: %d, median %.2f | centroid weights min %.3f max %.3f | |t| / |r| median %.2f, 90%% %.2f | '
          '(M + M_face) / V smallest eigenvalue: median %.3f, worst five %s (surface bodies median %.3f) | |V_res| / V: median %.1e, 99%% %.1e, max %.1e' % (
              name, hh.min(), int((hh <= 0).sum()), int((hh < .1).sum()), np.median(hh), float(st.ew.min()), float(st.ew.max()), np.median(tr), np.quantile(tr, .9),
              float(et[:, 0].median()), np.round(torch.sort(et[:, 0])[0][:5].cpu().numpy(), 3).tolist(), float(et[surf, 0].median()), np.median(vr),
              np.quantile(vr, .99), vr.max()), flush=True)
    # 3. rest, rigid, affine, patch test
    c = (m.mass[:, None]*st.X).sum(0)/m.mass.sum()
    size = float((st.X.max(0).values-st.X.min(0).values).max())
    common(name, st, mid, th, c, size)
    young = 23008.38
    patch(name, st, mid, th, c, surf, .016*young*float(st.fA.median()))
    # 4. surface-term vertex block
    t1 = time.time()
    lo, hi, dmin = surface_block(st)
    print('[%s] surface-term vertex block (sparse, %d x %d): smallest eigenvalues / median diagonal %s, largest %.2f, smallest diagonal %.2e (%.0f s)' % (
        name, st.nv, st.nv, np.array2string(lo, precision=3), hi, dmin, time.time()-t1), flush=True)


def main():
    a = argparse.ArgumentParser()
    a.add_argument('parts', nargs='*', default=['lattice', 'new', 'old'])
    a.add_argument('--iters', type=int, default=4000)
    args = a.parse_args()
    M = Path('/home/fangsuo/py/dynamic-powerfoam/output'); W = Path('output')
    if 'lattice' in args.parts:
        lattice_checks(args.iters)
    if 'new' in args.parts:
        real_checks('new', W/'pacnerf_dem_cutft/elastic_0_zup', W/'pacnerf_elastic0_f0_transparent_cutft', W/'pacnerf_dem_cutft/elastic_0')
    if 'old' in args.parts:
        real_checks('old', M/'pacnerf_dem8k_hull/elastic_0_zup', M/'pacnerf_elastic0_spacing8k', W/'talk_old_hull0')


if __name__ == '__main__':
    main()
