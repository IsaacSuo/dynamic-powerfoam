"""Which term of the skin model is inconsistent? (2026-10-22) A cube-lattice beam (nx x ny x nz bodies, spacing a) with
the skin model of sphere_model.SkinTerms on flat faces: per outer face 9 points (4 corners and 4 edge middles, shared,
and the face centre), linear hats on the 8 triangles (centre, boundary point, next), outer-layer thickness h = a / 2.
Exact solution: Saint-Venant pure bending about y (beam along x), traction-free sides:
  u_x = -k x z,  u_y = nu k y z,  u_z = k / 2 (x^2 + nu (z^2 - y^2)),   sigma = -E k z e_x e_x
(x, y, z from the beam's centre). With the bodies and the skin on it: the force of every term on the interior bodies,
the side bodies, the side skin points (all should vanish: div sigma = 0, sigma . n = 0) and the end skin points (they
carry the moment); the energy against 1/2 int E k^2 z^2 dV; and the energy with the side skin free (bodies on the
solution, ends held): a consistent model stays put. Also uniform uniaxial stress (sides free) as before."""
import argparse
import itertools

import numpy as np
import torch

from experiments.foam_dem_energy import EnergyFoam
from free_surface.part_modes import dense_hessian
from free_surface.sphere_model import SkinTerms

f64, dev = torch.float64, 'cuda'


def beam(nx, ny, nz, a, mu, lam):
    idx = np.array(list(itertools.product(range(nx), range(ny), range(nz))))
    X = idx*a; X = X-X.mean(0)
    nid = {tuple(t): k for k, t in enumerate(idx)}
    pairs = []
    for k, t in enumerate(idx):
        for ax in range(3):
            u = list(t); u[ax] += 1
            if tuple(u) in nid:
                pairs.append((k, nid[tuple(u)]))
    pairs = np.array(pairs)
    geom = dict(points=X, radii=np.full(len(X), (3*a**3/(4*np.pi))**(1/3)), pairs=pairs, areas=np.full(len(pairs), a*a),
                contacts=.5*(X[pairs[:, 0]]+X[pairs[:, 1]]), masses=np.full(len(X), a**3), fixed=np.zeros(len(X), bool),
                body_of_cell=np.arange(len(X)))
    m = EnergyFoam(geom, k_v=lam, volume=False, device=dev, young=2*mu, shear=2*mu, body_law='volume', k_mu=0., damping=0., dtype=f64, kinematics='exact')
    m.mu_ref, m.k_v = mu, lam
    return m, geom, idx, nid


def flat_skin(X, idx, nid, a):
    keys, Y = {}, []
    def pt(v):
        k_ = tuple(np.round(2*v/a).astype(int))
        if k_ not in keys:
            keys[k_] = len(Y); Y.append(v.copy())
        return keys[k_]
    ent = dict(pid=[], body=[], a=[], c=[], n=[]); parts = dict(off=[0], body=[], h=[], m=[], k=[], T=[], U=[])
    patches = dict(body=[], h=[], A=[], N=[]); face_n = []
    for B, t in enumerate(idx):
        for ax, s in itertools.product(range(3), (-1, 1)):
            u = list(t); u[ax] += s
            if tuple(u) in nid:
                continue
            n = np.zeros(3); n[ax] = s
            b_, c_ = [e for e in range(3) if e != ax]
            e1, e2 = np.eye(3)[b_], np.eye(3)[c_]
            if np.cross(e1, e2)@n < 0:
                e1, e2 = e2, e1
            C = X[B]+.5*a*n
            corners = [C+.5*a*(sb*e1+sc*e2) for sb, sc in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            ring = []
            for k in range(4):
                ring += [corners[k], .5*(corners[k]+corners[(k+1) % 4])]
            P = [C]+ring
            ids = [pt(v) for v in P]
            nl = len(P); M = np.zeros((nl, nl)); a_ = np.zeros(nl); Kg = np.zeros((nl, nl)); T = np.zeros((nl, nl, 3, 3)); U = np.zeros((nl, nl, 3, 3))
            Pn = np.eye(3)-np.outer(n, n)
            for k in range(8):
                tri = [0, 1+k, 1+(k+1) % 8]
                V = [P[q] for q in tri]
                At = .5*np.cross(V[1]-V[0], V[2]-V[0])@n
                g = [np.cross(n, V[(q+2) % 3]-V[(q+1) % 3])/(2*At) for q in range(3)]
                for qa in range(3):
                    a_[tri[qa]] += At/3
                    for qb in range(3):
                        M[tri[qa], tri[qb]] += At/12*(2 if qa == qb else 1)
                        Kg[tri[qa], tri[qb]] += At*g[qa]@g[qb]
                        T[tri[qa], tri[qb]] += At*np.outer(g[qa], g[qb])
                        U[tri[qa], tri[qb]] += At*(g[qa]@g[qb])*Pn
            ent['pid'] += ids; ent['body'] += [B]*nl; ent['a'] += list(a_); ent['c'] += [ai*n for ai in a_]; ent['n'] += [n]*nl
            parts['off'].append(parts['off'][-1]+nl); parts['body'].append(B); parts['h'].append(.5*a)
            parts['m'].append(M.ravel()); parts['k'].append(Kg.ravel()); parts['T'].append(T.ravel()); parts['U'].append(U.ravel())
            patches['body'].append(B); patches['h'].append(.5*a); patches['A'].append(a*a); patches['N'].append(a*a*np.outer(n, n))
            face_n.append(n)
    ent = {k_: np.array(v) for k_, v in ent.items()}
    parts = dict(off=np.array(parts['off']), body=np.array(parts['body']), h=np.array(parts['h']),
                 m=np.concatenate(parts['m']), k=np.concatenate(parts['k']), T=np.concatenate(parts['T']), U=np.concatenate(parts['U']))
    patches = {k_: np.array(v) for k_, v in patches.items()}
    return np.array(Y), ent, parts, patches


def saint_venant(P, k, nu):
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    return np.column_stack((-k*x*z, nu*k*y*z, .5*k*(x*x+nu*(z*z-y*y))))


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument('--n', type=int, nargs=3, default=[12, 4, 4])
    a_.add_argument('--no-lateral', action='store_true')
    args = a_.parse_args()
    a, mu, lam = .02, 1., 1.5
    nu = lam/(2*(lam+mu)); E = mu*(3*lam+2*mu)/(lam+mu)
    m, geom, idx, nid = beam(*args.n, a, mu, lam)
    X = geom['points']
    Y, ent, parts, patches = flat_skin(X, idx, nid, a)
    if args.no_lateral:
        parts = {k_: v for k_, v in parts.items() if k_ not in ('T', 'U')}
    st = SkinTerms(m, geom, Y, ent, parts, patches)
    n, npt = st.n, st.npt
    th = torch.zeros(n, 9, dtype=f64, device=dev)
    half = .5*a*np.array(args.n)
    onx = np.abs(np.abs(Y[:, 0])-half[0]) < 1e-12
    side_pt = ~onx
    side_body = (idx[:, 1] == 0) | (idx[:, 1] == args.n[1]-1) | (idx[:, 2] == 0) | (idx[:, 2] == args.n[2]-1)
    inner = ~side_body & (idx[:, 0] > 0) & (idx[:, 0] < args.n[0]-1)
    print('beam %s, a %.3f: bodies %d, skin points %d, faces %d; nu %.3f, E %.3f; lateral term %s' % (args.n, a, n, npt, len(patches['h']), nu, E, not args.no_lateral), flush=True)
    kap = 1e-3/a                                                   # strains ~1e-3 a z / a
    ux = torch.as_tensor(saint_venant(X, kap, nu), device=dev); uy = torch.as_tensor(saint_venant(Y, kap, nu), device=dev)
    x = (st.X+ux).requires_grad_(True); y = (st.Y+uy).requires_grad_(True)
    T = st.energy_terms(th, x, y)
    fs = E*kap*half[2]*a*a                                         # the end traction at the outer fibre x a face
    zmax_body = float(np.abs(X[:, 2]).max())
    print('forces on the exact bending solution / (E k z_max a^2): max |f| over interior bodies | side bodies | side skin points | end skin points', flush=True)
    tot_x = torch.zeros_like(x); tot_y = torch.zeros_like(y)
    for k_, v in T.items():
        gx, gy = torch.autograd.grad(v, (x, y), retain_graph=True)
        tot_x += gx; tot_y += gy
        print('   %-14s %.2e | %.2e | %.2e | %.2e' % (k_, float(gx[torch.as_tensor(inner, device=dev)].norm(dim=1).max())/fs if inner.any() else 0.,
                                                   float(gx[torch.as_tensor(side_body, device=dev)].norm(dim=1).max())/fs, float(gy[torch.as_tensor(side_pt, device=dev)].norm(dim=1).max())/fs,
                                                   float(gy[torch.as_tensor(onx, device=dev)].norm(dim=1).max())/fs), flush=True)
    print('   %-14s %.2e | %.2e | %.2e | %.2e' % ('TOTAL', float(tot_x[torch.as_tensor(inner, device=dev)].norm(dim=1).max())/fs if inner.any() else 0.,
                                               float(tot_x[torch.as_tensor(side_body, device=dev)].norm(dim=1).max())/fs, float(tot_y[torch.as_tensor(side_pt, device=dev)].norm(dim=1).max())/fs,
                                               float(tot_y[torch.as_tensor(onx, device=dev)].norm(dim=1).max())/fs), flush=True)
    # where on the sides: by point kind (corner / edge middle / face centre) and by direction
    kind = np.full(npt, 'corner', object)
    for q in range(npt):
        off_ = np.round(2*(Y[q]-X[np.argmin(np.linalg.norm(X-Y[q], axis=1))])/a).astype(int)
        kind[q] = {1: 'centre', 2: 'edge', 3: 'corner'}[int((off_ != 0).sum())]
    gy = tot_y.detach().cpu().numpy()
    for kk in ('centre', 'edge', 'corner'):
        mk = side_pt & (kind == kk)
        print('   side %-7s points %3d: total force max %.2e, its mean (x, y, z) %s' % (kk, int(mk.sum()), np.linalg.norm(gy[mk], axis=1).max()/fs,
                                                                              np.array2string(gy[mk].mean(0)/fs, precision=2)), flush=True)
    vol = (a**3*(X[:, 2]**2+a*a/12)).sum()
    Ee = .5*E*kap*kap*vol
    Em = sum(float(v) for v in T.values())
    print('energy / exact %.4f (terms %s)' % (Em/Ee, {k_: '%.4f' % (float(v)/Ee) for k_, v in T.items()}), flush=True)
    # side skin free, bodies and end skin on the solution
    n3 = 3*n
    xs, ys = st.X+ux, st.Y+uy
    free = torch.as_tensor(np.repeat(side_pt, 3), device=dev)
    Hyy = dense_hessian(lambda z: st.energy_xy(th, xs, ys+z.view(-1, 3)), 3*npt)
    yv = ys.clone().requires_grad_(True); gy0, = torch.autograd.grad(st.energy_xy(th, xs, yv), yv)
    Hf = Hyy[free][:, free]; dz = torch.linalg.solve(Hf, -gy0.reshape(-1)[free])
    yr = ys.clone().reshape(-1); yr[free] = yr[free]+dz; yr = yr.view(-1, 3)
    Er = float(st.energy_xy(th, xs, yr))
    disp = dz.view(-1, 3).norm(dim=1)
    print('side skin free (quadratic step): energy / exact %.4f (drop %.2e of the exact); skin moved by max %.2e, median %.2e of the largest bending displacement' % (
        Er/Ee, (Em-Er)/Ee, float(disp.max())/float(uy.norm(dim=1).max()), float(disp.median())/float(uy.norm(dim=1).max())), flush=True)


if __name__ == '__main__':
    main()
