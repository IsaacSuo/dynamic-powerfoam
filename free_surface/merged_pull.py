"""the merged form's spring at the neck: bond_pull.py with its remainder k (x_j - x_i - 1/2 (F_i + F_j) dX) beside the DEM bond. (10-30) On sc16's saved states, at each time: the bonds across the neck of each lost leg, R = polar of the frame's F (UL's, which reports about 1 while the pairs are at 2+), the bonds' direct force on the leg (F held fixed)
along the leg's outward direction, in body weights; beside it the model's measured elastic pull and a bar of the real material at that stretch (pull_curve.py); also with R from the part's own rigid fit, to see what the misread F costs. A state check on sc16's path, not a run.
usage: bond_pull.py <foam dir> <sc16 frames> <neck_trace npz> t..."""
import sys
from pathlib import Path
import numpy as np, torch
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.optimize import brentq
from torch.func import vmap, grad
from experiments.pacnerf_dem_forward import build
f64, dev = torch.float64, 'cuda'
foam, f16, ntr = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]; times = [float(a) for a in sys.argv[4:]]
E, NU = 10648.614517785627, .3912951704721983
g = dict(np.load(foam/'dem_exact_single'/'geometry.npz')); cl = np.load(foam/'dem_exact_single'/'cells.npz')
X, r, V = cl['points'], cl['radii'], g['masses']; n = len(X); ref = 9.8*V.sum()
md = build(g, E, NU, dev, 'noslip', .25, .3, f64, kinematics='exact', nodalvol=True, nonlinear=False, ogden_alpha=2.)
mu, lam = float(md.mu_ref), float(md.k_v)
bi, bj = md.i.long(), md.j.long(); ke_, ri_, rj_, n0_ = md.base.to(f64), md.ri.to(f64), md.rj.to(f64), md.n0.to(f64)
i0, j0, A0 = bi.cpu().numpy(), bj.cpu().numpy(), np.asarray(g['areas'])


def polar(F):
    Y = F
    for _ in range(14):
        Yi = torch.linalg.inv(Y); c = ((Yi*Yi).sum()/(Y*Y).sum())**.25
        Y = .5*(c*Y+Yi.T/c)
    return Y


def quat_R(w, v):
    x_, y_, z_ = v[0], v[1], v[2]
    return torch.stack([torch.stack([1-2*(y_*y_+z_*z_), 2*(x_*y_-z_*w), 2*(x_*z_+y_*w)]), torch.stack([2*(x_*y_+z_*w), 1-2*(x_*x_+z_*z_), 2*(y_*z_-x_*w)]), torch.stack([2*(x_*z_-y_*w), 2*(y_*z_+x_*w), 1-2*(x_*x_+y_*y_)])])


def e_bond(xs, Ri, Rj, p):
    xi, xj = xs[:3], xs[3:6]; ri, rj, n0, ke = p[:3], p[3:6], p[6:9], p[9:13]
    D = xj+Rj@rj-xi-Ri@ri
    Rr = Ri.T@Rj; tr = Rr[0, 0]+Rr[1, 1]+Rr[2, 2]
    w = .5*torch.sqrt(torch.clamp(1+tr, min=1e-6)); v = torch.stack([Rr[2, 1]-Rr[1, 2], Rr[0, 2]-Rr[2, 0], Rr[1, 0]-Rr[0, 1]])/(4*w)
    wh, vh = w+1, v; nh = torch.sqrt(wh*wh+(vh*vh).sum()); Rm = Ri@quat_R(wh/nh, vh/nh)
    d = Rm.T@D; dn = (d*n0).sum(); dt = d-dn*n0
    return .5*(ke[0]*dn*dn+ke[1]*(dt*dt).sum())


fb = vmap(grad(e_bond)); pol = vmap(polar)
st0 = torch.as_tensor(X, dtype=f64, device=dev)
par = torch.cat([ri_, rj_, n0_, ke_], 1)


def nominal(s):
    lt = brentq(lambda q: mu*(q*q-1)+lam*np.log(s*q*q), .2, 3.)
    return (mu*(s*s-1)+lam*np.log(s*lt*lt))/s


fs = sorted(f16.glob('f*.npz')); ft = np.array([float(np.load(f_)['t']) for f_ in fs]); key = lambda a, b: np.minimum(a, b)*n+np.maximum(a, b)
d = np.load(ntr); rk = key(i0, j0)
z = np.load(fs[int(np.argmin(np.abs(ft-.7)))])
from free_surface.recut import Recut
R_ = Recut(n, dev); tt = lambda q: torch.as_tensor(q, dtype=f64, device=dev)
G, _ = R_.geometry(tt(z['x']), tt(z['r'])); kp = G.wall_A > 0; i1, j1 = np.array(G.i[kp].cpu()), np.array(G.j[kp].cpu()); kq = np.isin(key(i1, j1), rk)
nc, lab = connected_components(sp.coo_matrix((np.ones(int(kq.sum())), (i1[kq], j1[kq])), shape=(n, n)), directed=False)
cnt = np.bincount(lab); parts = {'part%d' % cnt[q]: lab == q for q in np.argsort(-cnt)[1:] if cnt[q] >= 100}
for nm, fr in parts.items():
    rv = d[nm+'_vp']-d[nm+'_vb']; u = rv[-1]/np.linalg.norm(rv[-1]); ut = tt(u)
    rb = fr[i0] ^ fr[j0]; ids = np.where(rb)[0]; Ar = A0[rb].sum(); idt = torch.as_tensor(ids, device=dev); d0 = np.linalg.norm(X[i0[rb]]-X[j0[rb]], axis=1)
    print('%s: %d bonds across its neck; rest area %.2e; columns: t, stretch median, |F difference| across, bonds direct inward pull with UL\'s F, with R = I for all (no rotation read at all), a real bar at that stretch (body weights)' % (nm, len(ids), Ar), flush=True)
    frt = torch.as_tensor(fr, device=dev)
    for t in times:
        k_ = int(np.argmin(np.abs(ft-t))); z = np.load(fs[k_]); x, F = tt(z['x']), tt(z['F'].astype(np.float64))
        ai, aj = bi[idt], bj[idt]
        st_ = np.median((x[aj]-x[ai]).norm(dim=1).cpu().numpy()/d0)
        dF = float(np.median((F[ai]-F[aj]).flatten(1).norm(dim=1).cpu().numpy()))
        res = []
        for Rs in (pol(F[ai]), None):
            Ri = Rs if Rs is not None else torch.eye(3, dtype=f64, device=dev).expand(len(ai), 3, 3)
            Rj = pol(F[aj]) if Rs is not None else Ri
            gx = fb(torch.cat([x[ai], x[aj]], 1), Ri, Rj, par[idt])
            f_on = torch.where(frt[ai][:, None], -gx[:, :3], -gx[:, 3:6])                  # force on the end inside the part
            res.append(float(-(f_on.sum(0)*ut).sum())/ref)
        dX = (st0[aj]-st0[ai]); Fb_ = .5*(F[ai]+F[aj]); rem = x[aj]-x[ai]-torch.einsum('fab,fb->fa', Fb_, dX)
        ka = ke_[idt, 0][:, None]
        f_rem = torch.where(frt[ai][:, None], ka*rem, -ka*rem)
        res.append(float(-(f_rem.sum(0)*ut).sum())/ref)
        Rb_ = .5*(pol(F[ai])+pol(F[aj])); sh = torch.einsum('fab,fb->fa', Fb_-Rb_, dX)
        res.append(float(-((torch.where(frt[ai][:, None], ka*sh, -ka*sh)).sum(0)*ut).sum())/ref)
        print('  %.3f  stretch %.2f  |dF| %.2f  DEM bond %+.3f  R = I %+.3f  remainder %+.3f  shape part on the same springs %+.3f  bar %+.3f' % (ft[k_], st_, dF, res[0], res[1], res[2], res[3], nominal(st_)*Ar/ref if st_ > 1 else 0.), flush=True)
