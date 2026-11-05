"""does the consistent mass slow the midpoint step's CG? The step's linear system at rest, H = 4 M / dt^2 + K (K by
central differences of our gradient, dt = lion_inverse.DT, the ground far away), preconditioned as MidpointStep does
(3 x 3 blocks: the rest blocks K_bb + the mass's own block), lumped M = diag(m) against the consistent M = Mh (x) I3
(consistent_mass_check.py's Mh): plain PCG from 0 on random right-hand sides, iterations to relative residuals 1e-2, 1e-4,
1e-6; and the extreme eigenvalues of P^-1 H by the PCG's Lanczos coefficients"""
import sys
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import free_surface.lion_inverse as li
from experiments.pacnerf_dem_forward import build
from free_surface.closure_drop import colouring, probe_blocks
from free_surface.closure_warp import ClosureWarp
from free_surface.sphere_closure import ClosureTerms
f64, dev = torch.float64, 'cuda'
li.set_scene('pacnerf_e0')
cells, g, site_np = li.load_foam()
m = build(g, li.GT['E'], li.GT['nu'], dev, 'noslip', .25, .3, f64, kinematics='exact', nodalvol=True, nonlinear=False, ogden_alpha=2.)
st = ClosureTerms(m, g, cells['points'], cells['radii'])
W = ClosureWarp(st)
W.set_viscosity(1e-4*W.mu, 1e-4*(W.lam+2*W.mu/3), li.DT); W.visc_weight = 2.; W.total_mode()
site = torch.as_tensor(site_np, device=dev).long()
W.set_ball(site, torch.as_tensor(cells['radii'], dtype=f64, device=dev)[site], (0., 1., 0.), -10., 0.)
W.set_cap_positions(); W.set_cap_floor(0.)
W.prepare()
X = W.X_ext; n = W.n
W.visc_ref(X); W.elastic_mode()
Kbb = probe_blocks(W, X, colouring(st.i.cpu().numpy(), st.j.cpu().numpy(), n))
mass = m.mass.to(f64)
Mh = sp.load_npz(ROOT/'out_inverse'/'consistent_mass'/'Mh.npz').tocoo()
Mt = torch.sparse_coo_tensor(np.vstack([Mh.row, Mh.col]), Mh.data, (n, n), dtype=f64, device=dev).coalesce()
c4 = 4/li.DT**2
g0 = W.grad(X)[1].clone()


def Kv(v):
    s = 1e-6/float(v.abs().max())
    return (W.grad(X+s*v)[1]-W.grad(X-s*v)[1])/(2*s)


I3 = torch.eye(3, dtype=f64, device=dev)
Mhd = torch.as_tensor(Mh.toarray(), device=dev)
kbar = Kbb.diagonal(dim1=1, dim2=2).sum(1)/3
Ad = torch.linalg.inv(c4*Mhd+torch.diag(kbar)); Am = torch.linalg.inv(c4*Mhd); del Mhd
blk = lambda P_: (lambda Pi_: (lambda r: torch.einsum('nij,nj->ni', Pi_, r)))(torch.linalg.inv(P_))
Mcv = lambda v: c4*torch.sparse.mm(Mt, v)
cases = {'lumped, blocks': (lambda v: c4*mass[:, None]*v, blk(Kbb+c4*mass[:, None, None]*I3)),
         'consistent, blocks': (Mcv, blk(Kbb+c4*torch.as_tensor(Mh.tocsr().diagonal(), device=dev)[:, None, None]*I3)),
         'consistent, dense (4M/dt^2 + kbar)^-1': (Mcv, lambda r: Ad@r),
         'consistent, dense (4M/dt^2)^-1': (Mcv, lambda r: Am@r)}
print('dt %.2e: 4 m / dt^2 median %.3g against the rest blocks K_bb trace / 3 median %.3g' % (li.DT, float((c4*mass).median()), float((Kbb.diagonal(dim1=1, dim2=2).sum(1)/3).median())), flush=True)
gen = torch.Generator(device=dev).manual_seed(0)
for name, (Mv, Pap) in cases.items():
    for trial in range(1):
        b = torch.randn(n, 3, dtype=f64, device=dev, generator=gen)
        x = torch.zeros_like(b); r = b.clone(); y = Pap(r); d = y.clone(); rz = (r*y).sum()
        b0 = float(b.norm()); hit = {}; al, be = [], []
        for it in range(1, 3001):
            Hd = Mv(d)+Kv(d)
            a = rz/(d*Hd).sum(); x += a*d; r -= a*Hd
            rel = float(r.norm())/b0
            for t in (1e-2, 1e-4, 1e-6):
                if rel < t and t not in hit:
                    hit[t] = it
            if rel < 1e-6:
                break
            y = Pap(r); rz1 = (r*y).sum(); bt = rz1/rz; rz = rz1; d = y+bt*d
            al.append(float(a)); be.append(float(bt))
        k = len(al); Tm = np.zeros((k, k))
        for i in range(k):
            Tm[i, i] = 1/al[i]+(be[i-1]/al[i-1] if i else 0.)
            if i+1 < k:
                Tm[i, i+1] = Tm[i+1, i] = np.sqrt(be[i])/al[i]
        ev = np.linalg.eigvalsh(Tm)
        print('%-38s rhs %d: iterations to 1e-2 / 1e-4 / 1e-6: %s / %s / %s | P^-1 H eigenvalues (Lanczos) %.3g .. %.3g, condition %.0f' % (
            name, trial, hit.get(1e-2), hit.get(1e-4), hit.get(1e-6), ev[0], ev[-1], ev[-1]/ev[0]), flush=True)
