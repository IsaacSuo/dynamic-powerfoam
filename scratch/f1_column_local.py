"""F1's columns one at a time (the reviewer's cell 3432): at rest, each cap cell's column alone (everything else held), its
3 x 3 stiffness K_cc (probe_blocks with the columns coloured apart) against its own inertia M_cc (the consistent mass's
diagonal entry): the isolated column's lowest frequency f = sqrt(lam_min(K_cc) / M_cc) / 2 pi, per cell. With each cell's
geometry: V, its sphere's r, cap area A, Vdet / V (the det part of the volume), w = Mt^-1 Mc m (dF_B = dc w^T) and the
material's second moment along w (w^T J w / |w|^2 / V: its radius of gyration squared about the site along w), and the
stiffness and inertia per |w|^2. For the softest cells: the per-term curvature (bonds, volume, o1) along their softest
direction, dF_B's symmetric share, and the rows of the neighbours' gradient. args: number of soft cells to list (10)"""
import math
import sys
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import free_surface.lion_inverse as li
from experiments.pacnerf_dem_forward import build
from free_surface.closure_drop import colouring, probe_blocks
from free_surface.closure_warp import ClosureWarp
from free_surface.sphere_closure import ClosureTerms
K_ = int(sys.argv[1]) if len(sys.argv) > 1 else 10
f64, dev = torch.float64, 'cuda'
li.set_scene('pacnerf_e0')
cells, g, site_np = li.load_foam()
m = build(g, li.GT['E'], li.GT['nu'], dev, 'noslip', .25, .3, f64, kinematics='exact', nodalvol=True, nonlinear=False, ogden_alpha=2.)
st = ClosureTerms(m, g, cells['points'], cells['radii'])
W = ClosureWarp(st)
W.set_viscosity(0., 0., li.DT); W.visc_weight = 2.; W.total_mode()
site = torch.as_tensor(site_np, device=dev).long()
W.set_ball(site, torch.as_tensor(cells['radii'], dtype=f64, device=dev)[site], (0., 1., 0.), -10., 0.)
W.set_cap_positions(); W.set_cap_floor(0.)
W.set_cap_columns()
W.cm_cache = str(li.FOAM/'dem_exact_single'/'cell_moments.npz')
W.prepare()
n, N = W.n, W.ndof; X = W.X_ext.clone(); M = W.cmass['M']
col = colouring(st.i.cpu().numpy(), st.j.cpu().numpy(), n)
col = np.concatenate([col, col[W.cap_cells.cpu().numpy()]+col.max()+1])
Kbb = probe_blocks(W, X, col)
W.elastic_mode()
cc = W.cap_cells.cpu().numpy(); nc = len(cc)
Kc = Kbb[n:].cpu().numpy(); Mc_ = np.asarray(M.diagonal())[n:]
ev, evec = np.linalg.eigh(Kc)
f = np.sqrt(np.maximum(ev[:, 0], 0)/Mc_)/(2*math.pi)
V = st.V.cpu().numpy()[cc]; r = st.r.cpu().numpy()[cc]
A = np.bincount(st.c_body.cpu().numpy(), weights=st.c_A.cpu().numpy(), minlength=n)[cc]
Vdet = W.Vdet.numpy()[cc]
w = np.einsum('nij,nj->ni', st.Mt_inv.cpu().numpy()[cc], np.einsum('nij,nj->ni', st.Mcap.cpu().numpy()[cc], st.mA.cpu().numpy()[cc]))
wn = np.linalg.norm(w, axis=1); wh = w/wn[:, None]
Jm = W.cmass['J'][cc]
rg2 = np.einsum('ni,nij,nj->n', wh, Jm, wh)/V
nwall = np.bincount(np.concatenate([st.i.cpu().numpy(), st.j.cpu().numpy()]), minlength=n)[cc]
q = lambda a: '%.3g [%.3g, %.3g]' % (np.median(a), np.percentile(a, 5), np.percentile(a, 95))
print('isolated columns: %d cap cells; lowest frequency (Hz) median [5 %%, 95 %%] %s; under 10 Hz %d, under 30 Hz %d' % (nc, q(f), (f < 10).sum(), (f < 30).sum()), flush=True)
print('all caps: V %s mm^3, r %s mm, cap area %s mm^2, Vdet/V %s, |w| %s, radius of gyration along w %s mm' % (
    q(1e9*V), q(1e3*r), q(1e6*A), q(Vdet/V), q(wn), q(1e3*np.sqrt(rg2))), flush=True)
order = np.argsort(f)
print('\nsoftest %d: cell | f Hz | V mm^3 | r mm | cap A mm^2 | A r / V | Vdet/V | |w| | rg along w mm | walls | K/|w|^2 , M/|w|^2' % K_, flush=True)
for k in order[:K_]:
    print('  %5d | %6.2f | %7.0f | %5.1f | %6.1f | %6.3f | %5.3f | %5.3f | %5.1f | %3d | %.3g, %.3g' % (
        cc[k], f[k], 1e9*V[k], 1e3*r[k], 1e6*A[k], A[k]*r[k]/V[k], Vdet[k]/V[k], wn[k], 1e3*math.sqrt(rg2[k]), nwall[k], ev[k, 0]/wn[k]**2, Mc_[k]/wn[k]**2), flush=True)
print('\nthe same quantities, median over all caps: | %6.2f | %7.0f | %5.1f | %6.1f | %6.3f | %5.3f | %5.3f | %5.1f | %3d | %.3g, %.3g' % (
    np.median(f), 1e9*np.median(V), 1e3*np.median(r), 1e6*np.median(A), np.median(A*r/V), np.median(Vdet/V), np.median(wn), 1e3*np.median(np.sqrt(rg2)), np.median(nwall),
    np.median(ev[:, 0]/wn**2), np.median(Mc_/wn**2)), flush=True)
lf, lg = np.log(f), np.log
for nm, a in (('A r / V', A*r/V), ('Vdet / V', Vdet/V), ('|w|', wn), ('rg along w / r', np.sqrt(rg2)/r), ('walls', nwall.astype(float))):
    print('  corr(log f, log %s) %+.2f' % (nm, np.corrcoef(lf, np.log(np.maximum(a, 1e-12)))[0, 1]), flush=True)
print('\nper-term curvature along each softest cell\'s own softest direction (energy rows bonds, volume, o1; % of the total):', flush=True)
E0 = W.grad(X)[0]; T0 = W.tsum.cpu().numpy().ravel()[:3].copy()
for k in order[:K_]:
    u = torch.as_tensor(evec[k][:, 0], device=dev)
    rows = []; tot = []
    for h in (1e-4,):
        v = torch.zeros_like(X); v[n+k] = u
        W.grad(X+h*v); Tp = W.tsum.cpu().numpy().ravel()[:3].copy(); Fp = W.tF[cc[k]].clone()
        W.grad(X-h*v); Tm = W.tsum.cpu().numpy().ravel()[:3].copy()
        crv = (Tp+Tm-2*T0)/h**2
    dF = (Fp-torch.eye(3, dtype=f64, device=dev))/h
    sym = float(((dF+dF.T)/2).norm()/dF.norm())
    print('  %5d: %.4g total; bonds %.1f %%, volume %.1f %%, o1 %.1f %%; dF_B symmetric share %.3f, its trace / |dF| %+.3f' % (
        cc[k], crv.sum(), 100*crv[0]/crv.sum(), 100*crv[1]/crv.sum(), 100*crv[2]/crv.sum(), sym, float(torch.trace(dF)/dF.norm())), flush=True)
np.savez('out_inverse/f1_column_local.npz', cells=cc, f=f, V=V, r=r, A=A, Vdet=Vdet, w=w, rg2=rg2, nwall=nwall, Kc=Kc, Mc=Mc_)
print('saved out_inverse/f1_column_local.npz', flush=True)
