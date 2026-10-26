"""Open anomalies of step 2 (surface_checks.py logs, 2026-10-22), each traced to its cause, data only (no model change):
  faces     matched faces whose mesh area differs from the DEM's (6.2e3 x at most), mesh-only and DEM-only faces;
  vres      bodies with a large volume residue V_res (det F_B part): walls behind the site, merged-face residue, rest;
  soft      vertices the surface term holds weakly (smallest eigenvalues of its vertex block): where and why;
  offset    energy of affine maps against the current model (+5e-4 new, -1e-3..-2.5e-3 old): term by term, face by face;
  patch     interior residual of the patch test (1-3 %): against the strain (finite strain or mesh?), per term, linear bonds;
  hneg      air faces with h <= 0: merged bodies or a site outside its own power cell;
  lattice   the 2e-7 left on the lattice's free vertices at the uniform uniaxial state: with a from Newton on dE/da."""
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from experiments.foam_surface_completion import SurfaceCompletion, polar_newton
from experiments.pacnerf_dem_forward import build
from free_surface.grad_weights import MidpointTerms
from free_surface.surface_model import SurfaceTerms

f64, dev = torch.float64, 'cuda'
M_ = Path('/home/fangsuo/py/dynamic-powerfoam/output'); W_ = Path('output')
SHAPES = {'new': (W_/'pacnerf_dem_cutft/elastic_0_zup', W_/'pacnerf_elastic0_f0_transparent_cutft', W_/'pacnerf_dem_cutft/elastic_0'),
          'old': (M_/'pacnerf_dem8k_hull/elastic_0_zup', M_/'pacnerf_elastic0_spacing8k', W_/'talk_old_hull0')}
c_ = lambda t: t.detach().cpu().numpy()


def load(name, nonlinear=True):
    dem_z, run, mesh_dir = SHAPES[name]
    g = dict(np.load(dem_z/'geometry.npz'))
    m = build(g, 23008.38, .3008, dev, 'noslip', .25, .3, f64, kinematics='exact', nodalvol=True, nonlinear=nonlinear, ogden_alpha=2.)
    sc = SurfaceCompletion(run, dem_z, m, verbose=False)
    S = np.load(mesh_dir/'surface_mesh.npz')
    fr = sc.frame
    Y = fr['scale']*S['vert_foam']@fr['rotation'].T+fr['shift']
    st = SurfaceTerms(m, sc, g, dict(Y=Y, face_off=S['face_off'], face_vert=S['face_vert'], face_body=S['face_body']))
    mid = MidpointTerms(m, sc, g)
    cells = np.load(dem_z/'cells.npz')
    stt = torch.load(run/'model.pt', map_location='cpu', weights_only=True)
    n_real = int(stt['num_sites']) if 'num_sites' in stt else len(stt['points'])
    N = len(stt['points'])+(len(cells['ghost_points']) if 'ghost_points' in cells.files else 0)
    pm = cells['source_index'][S['face_cell']].astype(np.int64)*N+S['face_air']
    pr = sc._surface_args['pairs']; pd = pr[:, 0].astype(np.int64)*N+pr[:, 1]
    _, im, idd = np.intersect1d(pm, pd, return_indices=True)
    ncell = np.bincount(g['body_of_cell'], minlength=m.n)
    nair = len(sc.air_faces['area'])
    L_ = c_(sc.vs_lever[:nair].norm(dim=1)); A_d = np.asarray(sc.air_faces['area'])
    h_d = c_(sc.vs_k[:nair])*L_**2/(2*float(m.mu_ref)*A_d)
    return dict(name=name, m=m, sc=sc, g=g, S=S, st=st, mid=mid, n_real=n_real, N=N, pm=pm, pd=pd, pr=pr, im=im, idd=idd, ncell=ncell, nair=nair,
                A_m=c_(st.fA), A_d=A_d, h_d=h_d, h_m=c_(st.fh), fb=c_(st.fb))


def faces(D):
    nm = D['name']; S = D['S']; im, idd = D['im'], D['idd']; A_m, A_d = D['A_m'], D['A_d']
    sc = D['sc']; scale = sc.frame['scale']; r = sc._cell_args['r']*scale
    ratio = A_m[im]/A_d[idd]
    print('\n[%s faces] matched %d: mesh area / DEM area > 1 + 1e-9: %d, > 1.01: %d; excess area over the matched faces %.2e of the surface' % (
        nm, len(im), int((ratio > 1+1e-9).sum()), int((ratio > 1.01).sum()), (A_m[im]-A_d[idd]).sum()/A_d.sum()), flush=True)
    print('[%s faces] largest ratios: (mesh area, DEM area, npz clipped area, site ball radius, site-to-face distance h, cells of the body, air site mirror?)' % nm)
    for k in np.argsort(-ratio)[:5]:
        f, s = im[k], idd[k]
        site, air = D['pr'][s]
        print('   ratio %.3g: %.3e %.3e %.3e | r %.2f mm, h %.2f mm | cells %d | mirror %s' % (ratio[k], A_m[f], A_d[s], S['area_clipped'][f]*scale**2,
              1e3*r[site], 1e3*D['h_d'][s], D['ncell'][D['fb'][f]], air >= D['n_real']), flush=True)
    om = np.setdiff1d(np.arange(len(D['pm'])), im); od = np.setdiff1d(np.arange(len(D['pd'])), idd)
    print('[%s faces] mesh only %d: mesh area sum %.2e (max %.2e), their clipped area (npz) max %.1e, air site mirror %d / real %d' % (
        nm, len(om), A_m[om].sum(), A_m[om].max() if len(om) else 0, (S['area_clipped'][om]*scale**2).max() if len(om) else 0,
        int((S['face_air'][om] >= D['n_real']).sum()), int((S['face_air'][om] < D['n_real']).sum())), flush=True)
    print('[%s faces] DEM only %d: area sum %.2e (max %.2e), air site mirror %d / real %d' % (
        nm, len(od), A_d[od].sum(), A_d[od].max() if len(od) else 0, int((D['pr'][od, 1] >= D['n_real']).sum()), int((D['pr'][od, 1] < D['n_real']).sum())), flush=True)


def vres(D):
    nm = D['name']; st, mid = D['st'], D['mid']
    V = c_(st.V); A = c_(mid.A); hi, hj = c_(mid.hi), c_(mid.hj); wi, wj = c_(mid.wi), c_(mid.wj); i, j = c_(mid.i), c_(mid.j)
    behind = np.bincount(i, (1-wi)*A*hi/3, len(V))+np.bincount(j, (1-wj)*A*hj/3, len(V))
    front = V-c_(mid.Vrest)
    air = np.bincount(D['fb'], D['A_m']*D['h_m']/3, len(V))
    vr = c_(st.Vres); rest = V-front-behind-air
    single = D['ncell'] == 1
    top = np.argsort(-np.abs(vr)/V)[:max(1, len(V)//100)]
    print('\n[%s vres] V_res = V - front walls - air pyramids = (walls behind the site, signed) + rest (merged faces, ball clipping, DEM-only faces)' % nm)
    print('[%s vres] all bodies: sum |behind| %.2e, sum |rest| %.2e of the volume; single-cell bodies: |rest| / V max %.1e; merged bodies %d: |rest| / V median %.1e max %.1e' % (
        nm, np.abs(behind).sum()/V.sum(), np.abs(rest).sum()/V.sum(), (np.abs(rest)/V)[single].max(), int((~single).sum()),
        np.median((np.abs(rest)/V)[~single]) if (~single).any() else 0, (np.abs(rest)/V)[~single].max() if (~single).any() else 0), flush=True)
    print('[%s vres] top 1 %% (%d bodies, |V_res| / V >= %.2f): single-cell %d; behind / V_res median %.2f; surface bodies %d; |V_res| / V of the top five %s, '
          'their behind / V %s, rest / V %s' % (nm, len(top), float(np.abs(vr[top[-1]])/V[top[-1]]), int(single[top].sum()), np.median(behind[top]/vr[top]),
                                                 int(np.isin(top, D['fb']).sum()), np.round(vr[top[:5]]/V[top[:5]], 2).tolist(),
                                                 np.round(behind[top[:5]]/V[top[:5]], 2).tolist(), np.round(rest[top[:5]]/V[top[:5]], 3).tolist()), flush=True)


def soft(D):
    nm = D['name']; st = D['st']
    fb, fw, r, Mi = c_(st.fb), c_(st.fw), c_(st.r), c_(st.Mt_inv)
    ev_, eb, d, w, ef = c_(st.ev), c_(st.s_body), c_(st.s_d), c_(st.s_w), c_(st.ef)
    E, Fn, V = len(ev_), st.nf, st.nv
    order = np.argsort(fb, kind='stable'); cnt = np.bincount(fb, minlength=st.n); bfo = np.concatenate(([0], np.cumsum(cnt)))
    ce = cnt[eb]; e_rep = np.repeat(np.arange(E), ce)
    within = np.arange(ce.sum())-np.repeat(np.cumsum(ce)-ce, ce)
    f_rep = order[np.repeat(bfo[eb], ce)+within]
    coef = fw[f_rep]*np.einsum('ri,rij,rj->r', r[f_rep], Mi[eb[e_rep]], d[e_rep])
    C = sp.csr_matrix((coef, (e_rep, f_rep)), shape=(E, Fn))
    Bf = sp.csr_matrix((c_(st.ew), (ef, ev_)), shape=(Fn, V))
    G = sp.csr_matrix((np.ones(E), (np.arange(E), ev_)), shape=(E, V))-C@Bf
    K = (G.T@sp.diags(2*w)@G).tocsc()
    dg = K.diagonal(); md = np.median(dg)
    lo, vec = spla.eigsh(K, k=6, sigma=-1e-9*md, which='LM')
    Ah = np.abs(D['A_m']*D['h_m'])/3                                     # rest pyramid volume per face
    vmax = np.zeros(V); np.maximum.at(vmax, ev_, Ah[ef])                 # largest pyramid among a vertex's faces
    hmax = np.zeros(V); np.maximum.at(hmax, ev_, np.abs(D['h_m'])[ef])
    print('\n[%s soft] vertex diagonal / median: 1 %% quantile %.3f, min %.1e; diagonal vs largest pyramid of its faces: correlation of logs %.2f' % (
        nm, np.quantile(dg/md, .01), dg.min()/md, np.corrcoef(np.log(dg), np.log(vmax+1e-300))[0, 1]), flush=True)
    for k in range(len(lo)):
        v = vec[:, k]**2; o = np.argsort(-v); n90 = int(np.searchsorted(np.cumsum(v[o]), .9))+1
        t = o[0]
        fs_ = ef[ev_ == t]
        print('[%s soft] eigenvalue %.1e of the median diagonal: 90 %% on %d vertices; top vertex: %d faces, their |h| mm %s, area mm^2 %s, bodies %s (cells %s)' % (
            nm, lo[k]/md, n90, len(fs_), np.round(1e3*np.abs(D['h_m'][fs_]), 3).tolist(), np.round(1e6*D['A_m'][fs_], 2).tolist(),
            D['fb'][fs_].tolist(), D['ncell'][D['fb'][fs_]].tolist()), flush=True)
    print('[%s soft] faces: |h| median %.2f mm, area median %.1f mm^2' % (nm, 1e3*np.median(np.abs(D['h_m'])), 1e6*np.median(D['A_m'])), flush=True)


def offset(D):
    nm = D['name']; st, mid, sc, m = D['st'], D['mid'], D['sc'], D['m']
    gen = torch.Generator(device='cpu').manual_seed(1)
    Fa = torch.eye(3, dtype=f64, device=dev)+.05*torch.randn(3, 3, generator=gen, dtype=f64).to(dev)
    R = polar_newton(Fa[None])[0]; D_ = Fa-R
    th = torch.zeros(m.n, 9, dtype=f64, device=dev); cc = st.X.mean(0)
    x, y = st.affine(Fa, cc)
    E_n = float(st.energy_xy(th, x, y)); E_c = float(mid.energy(th, x, smooth_polar=True)[0])
    comp = c_(.5*sc.vs_k*((sc.vs_lever@D_.T)**2).sum(1)); nair = D['nair']
    half = c_(.5*st.kh*((st.rn@D_.T)**2).sum(1))
    im, idd = D['im'], D['idd']
    single = D['ncell'][D['fb'][im]] == 1
    ratio = D['A_m'][im]/D['A_d'][idd]
    clip = np.abs(ratio-1) > 1e-9
    om = np.setdiff1d(np.arange(len(D['pm'])), im); od = np.setdiff1d(np.arange(len(D['pd'])), idd)
    groups = {'single-cell, unclipped': single & ~clip, 'single-cell, clipped': single & clip, 'merged bodies': ~single}
    print('\n[%s offset] affine (amplitude 0.05): E_new / E_current - 1 = %.2e; half bonds - completion, by group, / E_current:' % (nm, E_n/E_c-1), flush=True)
    tot = 0.
    for k, mk in groups.items():
        dlt = half[im[mk]].sum()-comp[idd[mk]].sum(); tot += dlt
        print('   %-24s faces %5d: %+.2e' % (k, int(mk.sum()), dlt/E_c), flush=True)
    rows = (('mesh-only faces', half[om].sum()), ('DEM-only faces', -comp[od].sum()), ('sphere patches (dropped)', -comp[nair:].sum()))
    for k, dlt in rows:
        tot += dlt
        print('   %-24s       : %+.2e' % (k, dlt/E_c), flush=True)
    print('   sum %+.2e (against the total difference %+.2e: the rest is the other terms)' % (tot/E_c, E_n/E_c-1), flush=True)
    mk = ~single
    b = D['fb'][im[mk]]
    print('[%s offset] merged bodies: half bond / completion per face: median %.3f; h from the body point / h from the cell site: median %.3f, 10-90 %% %s' % (
        nm, np.median(half[im[mk]]/np.maximum(comp[idd[mk]], 1e-300)), np.median(D['h_m'][im[mk]]/D['h_d'][idd[mk]]),
        np.round(np.quantile(D['h_m'][im[mk]]/D['h_d'][idd[mk]], [.1, .9]), 2).tolist()), flush=True)


def patch(D, Dlin=None):
    nm = D['name']; st, mid, m = D['st'], D['mid'], D['m']
    th = torch.zeros(m.n, 9, dtype=f64, device=dev)
    cc = (m.mass[:, None]*st.X).sum(0)/m.mass.sum()
    surf = torch.zeros(m.n, dtype=torch.bool, device=dev); surf[st.fb] = True
    a0 = torch.tensor([[.01, .004, 0], [.004, -.006, .003], [0, .003, .008]], dtype=f64, device=dev)
    A_med = float(st.fA.median())
    print('\n[%s patch] vertices held affine, bodies free: body residual / (|sigma| x median face area) - interior | surface bodies; new model, current model' % nm)
    for s in (.01, .1, 1.):
        a = s*a0; Fa = torch.eye(3, dtype=f64, device=dev)+a
        fs = 23008.38*float(a.norm())*A_med
        x, y = st.affine(Fa, cc)
        x = x.clone().requires_grad_(True)
        T = st.energy_terms(th, x, y)
        gx = {k: torch.autograd.grad(v, x, retain_graph=True)[0] for k, v in T.items()}
        g = sum(gx.values())
        xm = x.detach().clone().requires_grad_(True)
        gm, = torch.autograd.grad(mid.energy(th, xm, smooth_polar=True)[0], xm)
        mx = lambda t_, mk: float(t_[mk].norm(dim=1).max())/fs
        print('   strain %.1e: new %.2e | %.2e ; current %.2e | %.2e ; new per term (interior max): %s' % (
            float(a.norm()), mx(g, ~surf), mx(g, surf), mx(gm, ~surf), mx(gm, surf), {k: '%.1e' % mx(v, ~surf) for k, v in gx.items()}), flush=True)
    if Dlin is not None:
        a = a0; Fa = torch.eye(3, dtype=f64, device=dev)+a; fs = 23008.38*float(a.norm())*A_med
        x, y = Dlin['st'].affine(Fa, cc)
        x = x.clone().requires_grad_(True)
        g, = torch.autograd.grad(Dlin['st'].energy_xy(th, x, y), x)
        print('   linear bonds (nonlinear=False), strain %.1e: new %.2e | %.2e' % (float(a.norm()), float(g[~surf].norm(dim=1).max())/fs, float(g[surf].norm(dim=1).max())/fs), flush=True)


def hneg(D):
    nm = D['name']; h = D['h_m']; fb = D['fb']; V = c_(D['st'].V)
    neg = h <= 0
    size = V[fb]**(1/3)
    single = D['ncell'][fb] == 1
    print('\n[%s hneg] air faces with h <= 0: %d on %d bodies; on single-cell bodies %d (their site lies outside its own power cell), merged %d' % (
        nm, int(neg.sum()), len(np.unique(fb[neg])), int((neg & single).sum()), int((neg & ~single).sum())), flush=True)
    for k, mk in (('single-cell', neg & single), ('merged', neg & ~single)):
        if mk.any():
            q = np.quantile(-h[mk]/size[mk], [.5, .9, 1.])
            print('   %-11s: -h / body size median %.3f, 90 %% %.3f, max %.3f; face area / body size^2 median %.3f' % (
                k, q[0], q[1], q[2], np.median(D['A_m'][mk]/size[mk]**2)), flush=True)


def lattice_a():
    from free_surface.surface_checks import lattice_mesh
    from prototypes.kb_derivation.face_model import lattice
    m, sc, g, idx = lattice()
    N, hl = 8, .02
    mesh, _ = lattice_mesh(idx, hl, N)
    st = SurfaceTerms(m, sc, g, mesh)
    th = torch.zeros(m.n, 9, dtype=f64, device=dev); cc = st.X.mean(0)
    mu, lam = float(m.mu_ref), float(m.k_v); young = mu*(3*lam+2*mu)/(lam+mu)
    Y = st.Y; zlo, zhi = float(Y[:, 2].min()), float(Y[:, 2].max())
    tb = ((Y[:, 2]-zlo).abs() < 1e-12) | ((Y[:, 2]-zhi).abs() < 1e-12)
    print('\n[lattice a] uniform uniaxial state with a from Newton on dE/da = 0 (autograd), free vertices\' residual / (|E e| x face area)')
    for e in (-.01, -.2):
        a = torch.tensor(1.+.3*abs(e), dtype=f64, device=dev)
        for _ in range(8):
            a = a.detach().requires_grad_(True)
            Fd = torch.diag(torch.stack((a, a, torch.ones_like(a)+e)))
            E = st.energy_xy(th, *st.affine(Fd, cc))
            g1, = torch.autograd.grad(E, a, create_graph=True)
            g2, = torch.autograd.grad(g1, a)
            a = a-g1/g2
        a = a.detach()
        Fd = torch.diag(torch.stack((a, a, torch.ones_like(a)+e)))
        x, y = st.affine(Fd, cc)
        y = y.clone().requires_grad_(True)
        gy, = torch.autograd.grad(st.energy_xy(th, x, y), y)
        print('   e %+.2f: a = %.15f, |dE/da| after Newton %.1e, free vertices max %.1e' % (e, float(a), float(g1.abs()), float(gy[~tb].norm(dim=1).max())/(abs(young*e)*hl*hl)), flush=True)


def moment(D):
    """Is the mu part exact for affine maps? A bond's energy mu A L |(F - R) n|^2 is its two cells' shares A h_i + A h_j;
    per cell sum_faces A (X_c - X_B) (x) n = V I (divergence theorem), so the bonds' in-plane offsets t cancel between the
    two cells of a wall but an air face has no partner: the air side owes M_air,B = sum_air A sym((h n + t) (x) n), while the
    completion / half bond gives sum_air A h+ n n. Checked against the continuum mu V |U - I|^2 + lam/2 V (ln J)^2."""
    nm = D['name']; st, mid, m = D['st'], D['mid'], D['m']
    nn = lambda a, b: a[:, :, None]*b[:, None, :]
    sym = lambda T: .5*(T+T.transpose(1, 2))
    A, h = st.fA, st.fh
    Mh = (A*h.clamp(min=0))[:, None, None]*nn(st.fn, st.fn)
    Mt = A[:, None, None]*sym(nn(st.rt, st.fn))
    Mneg = (A*h.clamp(max=0))[:, None, None]*nn(st.fn, st.fn)
    Mx = (A*h)[:, None, None]*nn(st.fn, st.fn)+Mt                           # exact owed moment per face (site based)
    z = lambda T: torch.zeros(m.n, 3, 3, dtype=f64, device=dev).index_add(0, st.fb, T)
    MB = z(Mx); Vt = float(st.V.sum())
    Mb = ((mid.A*mid.L)[:, None, None]*nn(mid.nrm, mid.nrm)).sum(0)
    I = torch.eye(3, dtype=f64, device=dev)
    print('\n[%s moment] global (divide by V): bonds + air owed - I max %.1e | bonds + air h+ (completion / half bond) - I max %.1e | owed parts: '
          't-part %.2e, h<0 part %.2e, h+ part %.2e (Frobenius / V)' % (
              nm, float((Mb+Mx.sum(0)-Vt*I).abs().max())/Vt, float((Mb+Mh.sum(0)-Vt*I).abs().max())/Vt, float(Mt.sum(0).norm())/Vt,
              float(Mneg.sum(0).norm())/Vt, float(Mh.sum(0).norm())/Vt), flush=True)
    surf = torch.unique(st.fb)
    ev = torch.linalg.eigvalsh(MB[surf])/st.V[surf, None]
    tr = (torch.stack([MB[surf][:, k, k] for k in range(3)], 1).sum(1))
    print('[%s moment] per surface cell M_air,B / V: smallest eigenvalue < 0 on %d of %d cells (min %.3f; median of the smallest %.3f); '
          '|t-part| / |h-part| per cell median %.2f' % (nm, int((ev[:, 0] < 0).sum()), len(surf), float(ev[:, 0].min()), float(ev[:, 0].median()),
                                                         float((z(Mt)[surf].norm(dim=(1, 2))/z(Mh+Mneg)[surf].norm(dim=(1, 2)).clamp(min=1e-300)).median())), flush=True)
    th = torch.zeros(m.n, 9, dtype=f64, device=dev); cc = st.X.mean(0)
    gen = torch.Generator(device='cpu').manual_seed(2)
    mu, lam = float(m.mu_ref), float(m.k_v)
    for amp in (1e-4,):
        for k in range(3):
            Fa = I+amp*torch.randn(3, 3, generator=gen, dtype=f64).to(dev)
            C = Fa.T@Fa; w_, Q_ = torch.linalg.eigh(C); Us = Q_@torch.diag(w_.sqrt())@Q_.T
            Ec = mu*Vt*float(((Us-I)**2).sum())+.5*lam*Vt*float(torch.log(torch.linalg.det(Fa)))**2
            x, y = st.affine(Fa, cc)
            En = float(st.energy_xy(th, x, y)); Em = float(mid.energy(th, x, smooth_polar=True)[0])
            print('   affine %g #%d: E / continuum - 1: new %+.2e, current %+.2e' % (amp, k, En/Ec-1, Em/Ec-1), flush=True)


def rest(D):
    """single-cell bodies with the largest volume residue not from walls behind the site: what is in it"""
    nm = D['name']; st, mid, sc = D['st'], D['mid'], D['sc']
    V = c_(st.V); A = c_(mid.A); hi, hj = c_(mid.hi), c_(mid.hj); wi, wj = c_(mid.wi), c_(mid.wj); i, j = c_(mid.i), c_(mid.j)
    behind = np.bincount(i, (1-wi)*A*hi/3, len(V))+np.bincount(j, (1-wj)*A*hj/3, len(V))
    rs = c_(st.Vres)-behind
    single = D['ncell'] == 1
    od = np.setdiff1d(np.arange(len(D['pd'])), D['idd'])
    nair = D['nair']; patch_b = c_(sc.vs_body[nair:])
    dem_only = np.bincount(np.asarray(sc.air_faces['body'])[od], D['A_d'][od], len(V))
    exc = np.bincount(D['fb'][D['im']], D['A_m'][D['im']]-D['A_d'][D['idd']], len(V))
    o = np.argsort(-np.abs(rs)*single/V)[:5]
    print('\n[%s rest] single-cell bodies with the largest |rest| / V (rest = V_res - walls behind the site):' % nm)
    for b in o:
        print('   body %d: rest / V %+.3f, surface %s, sphere patches %d, DEM-only face area %.2e, mesh - DEM area of its matched faces %.2e, V %.2e m^3' % (
            b, rs[b]/V[b], b in set(D['fb'].tolist()), int((patch_b == b).sum()), dem_only[b], exc[b], V[b]), flush=True)


def patch_split(D):
    """interior residual of the patch test: bodies in an all-single-cell neighbourhood vs bodies touching a merged body"""
    nm = D['name']; st, m = D['st'], D['m']
    th = torch.zeros(m.n, 9, dtype=f64, device=dev)
    cc = (m.mass[:, None]*st.X).sum(0)/m.mass.sum()
    a = torch.tensor([[.01, .004, 0], [.004, -.006, .003], [0, .003, .008]], dtype=f64, device=dev)
    fs = 23008.38*float(a.norm())*float(st.fA.median())
    x, y = st.affine(torch.eye(3, dtype=f64, device=dev)+a, cc)
    x = x.clone().requires_grad_(True)
    T = st.energy_terms(th, x, y)
    g = {k: torch.autograd.grad(v, x, retain_graph=True)[0].norm(dim=1).cpu().numpy()/fs for k, v in T.items()}
    tot = torch.autograd.grad(st.energy_xy(th, x, y), x)[0].norm(dim=1).cpu().numpy()/fs
    i, j = c_(st.i), c_(st.j)
    merged = D['ncell'] > 1
    touch = merged.copy(); touch[i[merged[j]]] = True; touch[j[merged[i]]] = True
    surf = np.zeros(m.n, bool); surf[D['fb']] = True
    # walls behind the site (h <= 0) on the body or a neighbour
    hb = np.zeros(m.n, bool); hb[i[c_(st.wi) == 0]] = True; hb[j[c_(st.wj) == 0]] = True
    hbn = hb.copy(); hbn[i[hb[j]]] = True; hbn[j[hb[i]]] = True
    print('\n[%s patch split] interior bodies, residual / (|sigma| x face area), max (median):' % nm)
    for k, mk in (('no merged body, no wall behind a site, nearby', ~surf & ~touch & ~hbn), ('touching a merged body', ~surf & touch),
                  ('a wall behind a site, on it or a neighbour', ~surf & hbn)):
        print('   %-48s %5d bodies: total %.1e (%.1e) | bonds %.1e volume %.1e' % (k, int(mk.sum()), tot[mk].max(), np.median(tot[mk]), g['bonds'][mk].max(), g['volume'][mk].max()), flush=True)


def centroid(D, spacing=None):
    """The outer layer measured from the cell centroid X_bar_B (sampled as consistent_mass_data; predicted position
    x_B + F_B (X_bar_B - X_B), exact for affine maps, no inverse needed) instead of the body point: h_bar of the air faces,
    the owed outer moment per cell sum_air A sym((X_c - X_bar) (x) n), and the global moment (bonds are tied to the body
    points: a bond's A L n n equals its wall's moment measured from the two body points only)."""
    from prototypes.kb_derivation.local_energy_map import material_points
    nm = D['name']; st, mid, sc, m = D['st'], D['mid'], D['sc'], D['m']
    V = c_(st.V); X = c_(st.X)
    spacing = spacing or .15*float(np.median(V))**(1/3)
    P, b = material_points(sc, spacing)
    cnt = np.bincount(b, minlength=m.n)
    Xbar = np.stack([np.bincount(b, P[:, k], m.n) for k in range(3)], 1)/np.maximum(cnt, 1)[:, None]
    Xbar[cnt == 0] = X[cnt == 0]
    fb = D['fb']; n = c_(st.fn); A = D['A_m']; Xc = c_(st.r)+X[fb]
    hbar = ((Xc-Xbar[fb])*n).sum(1)
    single = D['ncell'][fb] == 1
    surf = np.unique(fb)
    d = X-Xbar; size = V**(1/3)
    aair = np.zeros((m.n, 3)); np.add.at(aair, fb, A[:, None]*n)
    out = (d[surf]*aair[surf]).sum(1)/np.linalg.norm(aair[surf], axis=1).clip(1e-300)
    print('\n[%s centroid] sampling %.2f mm, points per body min %d median %d | surface bodies: |body point - centroid| / size median %.2f, 90 %% %.2f; '
          'outward along the air area %.2f mm median' % (nm, 1e3*spacing, cnt.min(), int(np.median(cnt)), np.median(np.linalg.norm(d[surf], axis=1)/size[surf]),
                                                        np.quantile(np.linalg.norm(d[surf], axis=1)/size[surf], .9), 1e3*np.median(out)), flush=True)
    print('[%s centroid] air faces with h <= 0: from the body point %d, from the centroid %d (single-cell bodies %d, merged %d)' % (
        nm, int((D['h_m'] <= 0).sum()), int((hbar <= 0).sum()), int(((hbar <= 0) & single).sum()), int(((hbar <= 0) & ~single).sum())), flush=True)
    rb = Xc-Xbar[fb]
    T = A[:, None, None]*.5*(rb[:, :, None]*n[:, None, :]+n[:, :, None]*rb[:, None, :])
    MB = np.zeros((m.n, 3, 3)); np.add.at(MB, fb, T)
    ev = np.linalg.eigvalsh(MB[surf])/V[surf, None]
    print('[%s centroid] owed outer moment per surface cell, from the centroid: smallest eigenvalue < 0 on %d of %d cells (min %.3f, median of the smallest %.3f)' % (
        nm, int((ev[:, 0] < 0).sum()), len(surf), ev[:, 0].min(), np.median(ev[:, 0])), flush=True)
    Mb = c_(((mid.A*mid.L)[:, None, None]*mid.nrm[:, :, None]*mid.nrm[:, None, :]).sum(0)); Vt = V.sum(); I = np.eye(3)
    nn = n[:, :, None]*n[:, None, :]
    Mh_site = ((A*np.clip(D['h_m'], 0, None))[:, None, None]*nn).sum(0)
    Mh_bar = ((A*np.clip(hbar, 0, None))[:, None, None]*nn).sum(0)
    print('[%s centroid] global moment (bonds + outer - V I) / V, max entry: outer A h+ n n from the body point %.1e, from the centroid %.1e; '
          'owed from the centroid in full (t-part incl.) %.1e' % (nm, np.abs(Mb+Mh_site-Vt*I).max()/Vt, np.abs(Mb+Mh_bar-Vt*I).max()/Vt,
                                                                 np.abs(Mb+T.sum(0)-Vt*I).max()/Vt), flush=True)


def main():
    parts = sys.argv[1:] or ['lattice', 'new', 'old']
    only = [p for p in parts if p.startswith('only=')]
    sections = only[0][5:].split(',') if only else ['faces', 'vres', 'hneg', 'soft', 'offset', 'patch']
    if 'lattice' in parts:
        lattice_a()
    for nm in ('new', 'old'):
        if nm not in parts:
            continue
        D = load(nm)
        for s in sections:
            if s == 'patch':
                patch(D, load(nm, nonlinear=False) if nm == 'new' else None)
            else:
                globals()[s](D)


if __name__ == '__main__':
    main()
