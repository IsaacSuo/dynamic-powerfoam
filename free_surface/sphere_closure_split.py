"""Where the skinless sphere model (sphere_closure.py) misses energy along the part's FEM modes (2026-10-26): modes 1 and
7 came out 12 % soft (sphere_closure_modes.log) where the skin model was within 1.5 %. Per mode (1, 7 and the controls 0,
2), along the mass-normalised FEM mode (energy 1/2 omega^2):
  FEM      the continuum energy per cell: the FEM's trilinear voxels (all 8 nodes in the mesh, centre in the union of the
           balls), 2 x 2 x 2 Gauss points, each point to the cell whose power distance is lowest
  new      the skinless model, per body: bonds (normal, shear, bending kb, torsion kt; half to each end), volume (and the
           variance term, half to each end), o1
  skin     the skin model (BoundaryTerms, lever c, Gamma sym_skin) with its skin relaxed (y = -K_yy^-1 K_yx u, K_yy dense at
           rest), per body: the same plus cross (o2), deviation (o3), lateral
Quadratic values from (E(+a u) + E(-a u)) / 2 a^2. Regions: the upright plate (sites above z = 0.025: y < 0.025, about two
cells thick) and the base; bodies with an exposed sphere or not.
  PYTHONPATH=<main repo>:<worktree> python free_surface/sphere_closure_split.py"""
import argparse
from pathlib import Path

import numpy as np
import torch

from experiments.dem_fem_mode_match import interpolate
from experiments.foam_surface_completion import polar_newton
from experiments.pacnerf_dem_forward import build
from free_surface.boundary_skin import BoundaryTerms
from free_surface.part_modes import dense_hessian
from free_surface.sphere_closure import ClosureTerms
from free_surface.sphere_model import matrix_quat
from free_surface.surface_model import cof3, det3

f64, dev = torch.float64, 'cuda'


def fem_cell_energy(c_, mode, pt, rt, mu, lam):
    """continuum energy per cell of FEM mode `mode` (trilinear voxels, 2 x 2 x 2 Gauss points)"""
    X, eps, vol, cell = fem_gauss(c_, mode, pt, rt)
    w = (mu*(eps*eps).sum((1, 2))+.5*lam*eps.diagonal(dim1=1, dim2=2).sum(1)**2)*vol
    return torch.zeros(len(pt), dtype=f64, device=dev).index_add(0, cell, w), float(w.sum())


def fem_gauss(c_, mode, pt, rt):
    """FEM mode `mode` at the Gauss points (2 x 2 x 2 per trilinear voxel with all 8 nodes in the mesh and its centre in
    the union of the balls): positions, strains, volumes, the cell of lowest power distance"""
    G, lo, h = c_['node_grid'], c_['lo'], float(c_['h'])
    U = c_['V'][:, mode].reshape(-1, 3)
    shape = G.max(0)+2
    idx = -np.ones(shape, int); idx[G[:, 0], G[:, 1], G[:, 2]] = np.arange(len(G))
    order = np.array([[a, b, c] for a in (0, 1) for b in (0, 1) for c in (0, 1)])
    cand = G[(G < shape-1).all(1)]
    corner = idx[cand[:, None, 0]+order[None, :, 0], cand[:, None, 1]+order[None, :, 1], cand[:, None, 2]+order[None, :, 2]]   # (v, 8)
    vox, corner = cand[(corner >= 0).all(1)], corner[(corner >= 0).all(1)]
    cen = torch.as_tensor(lo+(vox+.5)*h, dtype=f64, device=dev)
    inside = torch.cat([(torch.cdist(cen[s:s+4096], pt)**2-rt**2).min(1).values <= 0 for s in range(0, len(cen), 4096)]).cpu().numpy()
    vox, corner = vox[inside], corner[inside]
    Uv = torch.as_tensor(U[corner], dtype=f64, device=dev)                                   # (v, 8, 3)
    od = torch.as_tensor(order, dtype=f64, device=dev)
    out = []
    for g in order:
        xi = torch.as_tensor(.5+(g-.5)/np.sqrt(3), dtype=f64, device=dev)                    # Gauss point in [0, 1]^3
        f = od*xi+(1-od)*(1-xi)                                                               # (8, 3) factors per corner
        dN = torch.stack([(2*od[:, d]-1)*f[:, (d+1) % 3]*f[:, (d+2) % 3] for d in range(3)], 1)/h   # (8, 3)
        gu = torch.einsum('vci,cj->vij', Uv, dN)
        X = torch.as_tensor(lo+(vox+xi.cpu().numpy())*h, dtype=f64, device=dev)
        cell = torch.cat([(torch.cdist(X[s:s+4096], pt)**2-rt**2).argmin(1) for s in range(0, len(X), 4096)])
        out.append((X, .5*(gu+gu.transpose(1, 2)), torch.full((len(X),), h**3/8, dtype=f64, device=dev), cell))
    return tuple(torch.cat(z) for z in zip(*out))


def bond_terms(m, th, x, q, n):
    *_, k, dn, dt, pn, pt = m._bond_exact(th, x, q)
    out = {}
    for name, e in (('bond normal', .5*k[:, 0]*dn**2), ('bond shear', .5*k[:, 1]*(dt*dt).sum(-1)),
                    ('bond bend', .5*k[:, 2]*(pt*pt).sum(-1)), ('bond twist', .5*k[:, 3]*pn**2)):
        out[name] = torch.zeros(n, dtype=f64, device=dev).index_add(0, m.i, .5*e).index_add(0, m.j, .5*e)
    return out


def volume_terms(st, lam, Vd, n):
    lj, _ = st.model._log_J(Vd/st.V)
    v = st.V*.5*lam*lj**2
    var = .5*lam*st.A*st.L/12*(lj[st.i]-lj[st.j])**2
    return v.index_add(0, st.i, .5*var).index_add(0, st.j, .5*var)


def per_body_new(st, th, x):
    m, n, mu, lam = st.model, st.n, st.mu, float(st.model.k_v)
    F = st.F(x); R = polar_newton(F); q = matrix_quat(R)
    out = bond_terms(m, th, x, q, n)
    D = F-R; Dc = D[st.c_body]
    out['o1'] = torch.zeros(n, dtype=f64, device=dev).index_add(0, st.c_body, mu*st.c_r*torch.einsum('cki,ckj,cij->c', Dc, Dc, st.c_N))
    Vd = st.wall_volume(x, F, R)+det3(F)*(st.Vres+torch.zeros_like(st.V).index_add(0, st.c_body, st.c_r*st.c_A/3))
    out['volume'] = volume_terms(st, lam, Vd, n)
    return out, float(sum(st.energy_terms(th, x).values()))


def per_body_skin(st, th, x, y):
    """SkinTerms.energy_terms (quad_ref, gamma sym_skin, lateral) per body"""
    m, n, mu, lam = st.model, st.n, st.mu, float(st.model.k_v)
    z = lambda: torch.zeros(n, dtype=f64, device=dev)
    F = st.F(x, y); R = polar_newton(F); q = matrix_quat(R)
    out = bond_terms(m, th, x, q, n)
    D = F-R; Fe = F[st.e_body]
    delta_aff = y[st.e_pid]-x[st.e_body]-torch.einsum('eij,ej->ei', Fe, st.e_d)
    rb = st.wb[:, None, None]*torch.einsum('bkm,bik->bim', st.g_A, F[st.j]-F[st.i])
    rhs = torch.zeros(n, 3, 6, dtype=f64, device=dev).index_add(0, st.i, rb).index_add(0, st.j, rb)
    sg = torch.einsum('bmn,bin->bim', st.g_Ninv, rhs)
    delta = delta_aff-.5*torch.einsum('eim,em->ei', sg[st.e_body], st.g_q)
    Dc = D[st.c_body]
    out['o1'] = z().index_add(0, st.c_body, mu*st.c_r*torch.einsum('cki,ckj,cij->c', Dc, Dc, st.c_N))
    out['o2'] = z().index_add(0, st.e_body, 2*mu*(delta_aff*torch.einsum('eij,ej->ei', D[st.e_body], st.e_c)).sum(1))
    delta = torch.einsum('eji,ej->ei', R[st.e_body], delta)
    da, db = delta[st.pr_a], delta[st.pr_b]
    pb = st.p_body[st.pr_part]; r_p = st.p_r[st.pr_part]
    out['o3'] = z().index_add(0, pb, mu/r_p*st.pr_m*(da*db).sum(1))
    kap = lam*mu/(lam+2*mu)
    tU = torch.einsum('ni,nij,nj->n', da, st.pr_U, db); tTs = torch.einsum('ni,nij,nj->n', db, st.pr_T, da)
    tT = torch.einsum('ni,nij,nj->n', da, st.pr_T, db)
    out['lateral'] = z().index_add(0, pb, r_p/3*(.5*mu*(tU+tTs)+kap*tT))
    Vd = st.wall_volume(x, F, R)+det3(F)*(st.Vres+torch.zeros_like(st.V).index_add(0, st.c_body, st.c_r*st.c_A/3))
    Vd = Vd.index_add(0, st.e_body, (delta_aff*torch.einsum('eij,ej->ei', cof3(Fe), st.e_c)).sum(1)/3)
    out['volume'] = volume_terms(st, lam, Vd, n)
    return out, float(st.energy_xy(th, x, y))


def quad(fn, a):
    """(T(+a) + T(-a)) / 2 a^2 per term and in total; fn(s) -> (dict of per-body tensors, total)"""
    (p_, tp), (m_, tm) = fn(a), fn(-a)
    return {k: (p_[k]+m_[k])/(2*a*a) for k in p_}, (tp+tm)/(2*a*a)


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument('--part', type=Path, default=Path('output/test_part'))
    a_.add_argument('--fem-h', type=float, default=.0015)
    a_.add_argument('--modes', type=int, nargs='*', default=[1, 7, 0, 2])
    a_.add_argument('--zcut', type=float, default=.025)
    args = a_.parse_args()
    stt = torch.load(args.part/'run'/'model.pt', map_location='cpu', weights_only=True)
    p = stt['points'].double().numpy(); r = torch.nn.functional.softplus(stt['radii'].double(), beta=100).numpy()
    c_ = dict(np.load(args.part/('fem_modes_h%g.npz' % args.fem_h)))
    F_ = dict(nodes=c_['nodes'], node_grid=c_['node_grid'], lo=c_['lo'], h=float(c_['h']), masses=c_['masses'])
    E, nu = 23008.38, .3008
    g = dict(np.load(args.part/'dem_exact_single'/'geometry.npz'))
    m = build(g, E, nu, dev, 'noslip', .25, .3, f64, kinematics='exact', nodalvol=True, nonlinear=False, ogden_alpha=2.)
    bsk = dict(np.load(args.part/'boundary_skin.npz'))
    new = ClosureTerms(m, g, p, r, bsk)
    old = BoundaryTerms(m, g, p, r, bsk, lever='c', gamma='sym_skin')
    n = new.n; th = torch.zeros(n, 9, dtype=f64, device=dev)
    X, Y = new.X, old.Y
    pt, rt = torch.as_tensor(p, dtype=f64, device=dev), torch.as_tensor(r, dtype=f64, device=dev)
    mu, lam = new.mu, float(m.k_v)
    mass = m.mass.to(f64)
    plate = pt[:, 2] > args.zcut
    cap = new.cap
    regions = (('plate, exposed', plate & cap), ('plate, inner', plate & ~cap), ('base, exposed', ~plate & cap), ('base, inner', ~plate & ~cap))
    print('regions (bodies): %s' % ', '.join('%s %d' % (nm, int(msk.sum())) for nm, msk in regions), flush=True)
    Kyy = dense_hessian(lambda z: old.energy_xy(th, X, Y+z.view(-1, 3)), 3*old.npt)
    LU = torch.linalg.lu_factor(Kyy)

    def gy(x):
        y = Y.clone().requires_grad_(True)
        return torch.autograd.grad(old.energy_xy(th, x, y), y)[0]
    for k in args.modes:
        om2 = (2*np.pi*float(c_['f'][k]))**2
        Ef, tot = fem_cell_energy(c_, k, pt, rt, mu, lam)
        u, _ = interpolate(F_, c_['V'][:, k:k+1], X.cpu().numpy())
        u = torch.as_tensor(u[:, :, 0], dtype=f64, device=dev)
        a = 1e-6/float(u.norm(dim=1).max())
        Kyx_u = (gy(X+a*u)-gy(X-a*u))/(2*a)
        w = -torch.linalg.lu_solve(*LU, Kyx_u.reshape(-1, 1)).view(-1, 3)
        Pn, tn = quad(lambda s: per_body_new(new, th, X+s*u), a)
        Po, to = quad(lambda s: per_body_skin(old, th, X+s*u, Y+s*w), a)
        ref = .5*om2
        kin = mass*(u*u).sum(1)
        ydir = float((mass*u[:, 1]**2)[plate].sum()/kin[plate].sum())
        print('mode %d (FEM %.2f Hz): FEM energy by Gauss points / (omega^2 / 2) %.4f; the plate holds %.2f of the FEM energy and %.2f of the motion '
              '(its motion %.2f along y, through the plate); totals / FEM: new %.4f (terms add to %.4f), skin model %.4f (%.4f)' % (
                  k, float(c_['f'][k]), tot/ref, float(Ef[plate].sum())/tot, float(kin[plate].sum()/kin.sum()), ydir,
                  tn/ref, float(sum(v.sum() for v in Pn.values()))/ref, to/ref, float(sum(v.sum() for v in Po.values()))/ref), flush=True)
        print('   terms / FEM total:  new %s | skin model %s' % (
            ', '.join('%s %.3f' % (nm, float(v.sum())/ref) for nm, v in Pn.items()),
            ', '.join('%s %.3f' % (nm, float(v.sum())/ref) for nm, v in Po.items())), flush=True)
        for nm, msk in regions:
            ef = float(Ef[msk].sum())
            en, eo = float(sum(v[msk].sum() for v in Pn.values())), float(sum(v[msk].sum() for v in Po.values()))
            print('   %-15s FEM %.3f of the total; new / FEM %.3f, skin model / FEM %.3f; new terms %s; skin model terms %s' % (
                nm, ef/tot, en/max(ef, 1e-300), eo/max(ef, 1e-300),
                ' '.join('%s %.3f' % (t.split()[-1], float(v[msk].sum())/max(ef, 1e-300)) for t, v in Pn.items()),
                ' '.join('%s %.3f' % (t.split()[-1], float(v[msk].sum())/max(ef, 1e-300)) for t, v in Po.items())), flush=True)


if __name__ == '__main__':
    main()
