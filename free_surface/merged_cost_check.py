"""does a spring's dependence on the gradient explain the merged landing's CG cost? On saved frames only, no model change, no run. r = du - 1/2 (G_i + G_j) d, with V G = sum A u_w (x) n_out + u (x) a, so a spring's row in x has 1 at its two ends and
about (A' / V) (L / 2) on the rings of its two cells, A' the rings' walls: the spring's stiffness on those sites is k (A' L / 2 V)^2, while the preconditioner has only k on the two ends. Per frame: A L / V over the walls (each wall from each of its cells);
phi = lambda_max(sum k J^T J at a site's own block) / (sum k at the site), direct flux only (the tau correction and the caps' fit left out), beta left out, it scales both; the same over the blocks with the mass term cm = m / (th dt)^2; beside the log's
products per step. At the last state: where the largest |g| lie among the ratios. The same geometry numbers on d10's frames, whose k is 1/6.
  python free_surface/merged_cost_check.py <foam dir> <frames npz> <log> [--fail fail npz] [--other frames npz, log]"""
import argparse
import re
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from free_surface.recut import Recut

f64, dev = torch.float64, 'cuda'
a_ = argparse.ArgumentParser()
a_.add_argument('foam', type=Path)
a_.add_argument('frames', type=Path)
a_.add_argument('log', type=Path)
a_.add_argument('--fail', type=Path, default=None)
a_.add_argument('--other', nargs=2, default=None)
a_.add_argument('--from-t', type=float, default=.185)
a_.add_argument('--every', type=int, default=1)
args = a_.parse_args()
g = np.load(args.foam/'dem_exact_single'/'geometry.npz'); V0 = g['masses']; n = len(V0)
E, nu = 174.2839, .3008; mu = E/(2*(1+nu)); dt = 1/1200; th = .5
cm = V0/(th*dt)**2
R_ = Recut(n, dev)
if hasattr(R_.A, 'f32'):
    R_.A.f32 = True


def cut(x, r):
    G, _ = R_.geometry(torch.as_tensor(x, dtype=f64, device=dev), torch.as_tensor(r, dtype=f64, device=dev))
    kp = G.wall_A.detach() > 0
    return G.i[kp].long().cpu().numpy(), G.j[kp].long().cpu().numpy(), G.wall_A.detach()[kp].cpu().numpy(), G.wall_c.detach()[kp].cpu().numpy(), G.V.detach().cpu().numpy(), G.patch_A.detach().cpu().numpy()


def measures(x, r):
    i, j, A, c, V, pa = cut(x, r)
    m = len(i)
    d = x[j]-x[i]; L = np.linalg.norm(d, axis=1); nh = d/L[:, None]; t = ((c-x[i])*nh).sum(1)/L
    Vc = np.maximum(V, 1e-12*np.median(V))
    ratio = np.concatenate([A*L/Vc[i], A*L/Vc[j]]); owner = np.concatenate([i, j])
    per = np.zeros(n); np.maximum.at(per, owner, ratio)
    k = 2*mu*A/L
    # the rows of r in x: direct flux only
    a0 = np.zeros((n, 3)); np.add.at(a0, i, -A[:, None]*nh); np.add.at(a0, j, A[:, None]*nh)
    open_ = pa > 0
    rows, cols, vals = [], [], []
    for a in range(3):
        for b in range(3):
            for s_, sg in ((i, 1.), (j, -1.)):
                gs = 9*s_+3*a+b
                rows += [gs, gs]; cols += [3*i+a, 3*j+a]; vals += [sg*A*(1-t)*nh[:, b]/Vc[s_], sg*A*t*nh[:, b]/Vc[s_]]
            sel = np.where(open_)[0]
            rows.append(9*sel+3*a+b); cols.append(3*sel+a); vals.append(a0[sel, b]/Vc[sel])
    Bx = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(9*n, 3*n)).tocsr()
    e_ = np.arange(m); rows, cols, vals = [], [], []
    for a in range(3):
        rows += [3*e_+a, 3*e_+a]; cols += [3*j+a, 3*i+a]; vals += [np.ones(m), -np.ones(m)]
    D1 = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(3*m, 3*n)).tocsr()
    rows, cols, vals = [], [], []
    for a in range(3):
        for b in range(3):
            rows += [3*e_+a, 3*e_+a]; cols += [9*i+3*a+b, 9*j+3*a+b]; vals += [.5*d[:, b], .5*d[:, b]]
    D2 = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(3*m, 9*n)).tocsr()
    C = (D1-D2@Bx).tocsc()
    w = np.repeat(k, 3)
    Cs = [C[:, a::3] for a in range(3)]
    K = np.zeros((n, 3, 3))
    for a in range(3):
        for b in range(a, 3):
            q = np.asarray(Cs[a].multiply(Cs[b]).T.dot(w)).ravel()
            K[:, a, b] = q; K[:, b, a] = q
    lam = np.linalg.eigvalsh(K)[:, 2]
    own = np.zeros(n); np.add.at(own, i, k); np.add.at(own, j, k)
    phi = lam/np.maximum(own, 1e-300)
    psi = (lam+cm)/(own+cm)
    return dict(ratio=ratio, per=per, phi=phi, psi=psi, Vr=V/V0, m=m)


def products(log):
    out = {}
    for line in open(log, errors='ignore'):
        m_ = re.search(r'mem step (\d+) t ([0-9.]+):.*Newton (\d+), products (\d+)', line)
        if m_:
            out[int(m_.group(1))] = (float(m_.group(2)), int(m_.group(3)), int(m_.group(4)))
    return out


def table(frames, log, every=1, title=''):
    f = np.load(frames); ts, xs, rs = f['t'], f['x'], f['r']
    pr = products(log)
    print(title)
    print('t | products per step around it, Newton | A L / V: median, 99 %, 99.9 %, max; cells with a wall above 10, 30 | V / V0 min | phi: median, 99 %, max | (chain + mass) / (own + mass): 99 %, max', flush=True)
    last = None
    for k in range(0, len(ts), every):
        if ts[k] < args.from_t:
            continue
        M = measures(xs[k], rs[k])
        s = int(round(ts[k]*1200))
        near = [pr[q] for q in range(s-4, s+6) if q in pr]
        pn = '%6.0f %5.1f' % (np.mean([q[2] for q in near]), np.mean([q[1] for q in near])) if near else '     -     -'
        q_ = np.quantile(M['ratio'], [.5, .99, .999])
        print('%.4f | %s | %6.1f %7.1f %8.1f %9.1f; %5d %5d | %.2e | %6.2f %8.1f %10.1f | %8.2f %10.1f' % (ts[k], pn, q_[0], q_[1], q_[2], M['ratio'].max(), (M['per'] > 10).sum(), (M['per'] > 30).sum(), M['Vr'].min(),
              np.median(M['phi']), np.quantile(M['phi'], .99), M['phi'].max(), np.quantile(M['psi'], .99), M['psi'].max()), flush=True)
        last = (k, M)
    return f, last


f, last = table(args.frames, args.log, args.every, 'the merged landing, k = the DEM\'s')
if args.fail is not None:
    z = np.load(args.fail); gz = np.linalg.norm(z['g'], axis=1)
    M = measures(z['x'], f['r'][-1])
    top = np.argsort(-gz)[:12]
    rk = M['per'].argsort().argsort()/n
    print('at the failure (the state before the stalled step, the last saved radii): the 12 largest |g| of the solve: their A L / V max %s; their rank among the cells by that, 1 the largest: %s; phi %s; V / V0 %s' % (
        np.round(M['per'][top], 1).tolist(), np.round(rk[top], 3).tolist(), np.round(M['phi'][top], 1).tolist(), np.round(M['Vr'][top], 3).tolist()), flush=True)
    print('share of |g|^2 on the 1 %% of the cells with the largest A L / V: %.3f, on the 1 %% with the largest phi: %.3f' % ((gz[M['per'] >= np.quantile(M['per'], .99)]**2).sum()/(gz**2).sum(), (gz[M['phi'] >= np.quantile(M['phi'], .99)]**2).sum()/(gz**2).sum()), flush=True)
if args.other is not None:
    table(Path(args.other[0]), Path(args.other[1]), args.every, 'the same geometry numbers on the other run\'s frames (its k is k / 6)')
print('done', flush=True)
