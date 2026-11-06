"""the full tangent stiffness's low modes at rest, K v = lam M v (K by central-difference Hessian-vector products of the
gradient, M the consistent mass), PAC-NeRF elastic scene 0's foam at GT E, nu: the old model (plane-stress completion) or
F1 (argv[1] 'cols': the caps' columns as unknowns). The 6 rigid modes (sites translate / rotate, columns rotate) checked
in K's null space (|K r| against |K v| of random v), then held out as LOBPCG's constraints: the lowest elastic modes,
their frequencies against the FEM's (out_inverse/freevib/prep.npz), and each mode's share of its kinetic norm in the
columns. args: old|cols, number of modes (default 16), out npz"""
import math
import sys
from pathlib import Path
import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator, lobpcg
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import free_surface.lion_inverse as li
from experiments.pacnerf_dem_forward import build
from free_surface.closure_drop import colouring, probe_blocks
from free_surface.closure_warp import ClosureWarp
from free_surface.sphere_closure import ClosureTerms
mode, K_ = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 16
out = sys.argv[3] if len(sys.argv) > 3 else 'out_inverse/f1_modes_%s.npz' % mode
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
if mode == 'cols':
    W.set_cap_columns()
W.cm_cache = str(li.FOAM/'dem_exact_single'/'cell_moments.npz')
W.prepare()
n, N = W.n, W.ndof; X = W.X_ext.clone(); M = W.cmass['M']
W.elastic_mode()


def Kv(v):                                                                          # (v: (N, 3))
    s_ = float(v.abs().max())
    if s_ == 0:
        return torch.zeros_like(v)
    h = 1e-6/s_
    return (W.grad(X+h*v)[1]-W.grad(X-h*v)[1])/(2*h)


Mop = lambda a: np.stack([M@a.reshape(N, 3, -1)[:, k] for k in range(3)], 1).reshape(3*N, -1)
Xn = X[:n].cpu().numpy(); xc = Xn-Xn.mean(0)
Rb = np.zeros((N, 3, 6))
for k in range(3):
    Rb[:n, k, k] = 1.; Rb[:n, :, 3+k] = np.cross(np.eye(3)[k], xc)
    if mode == 'cols':
        Rb[n:, :, 3+k] = np.cross(np.eye(3)[k], W.c_rest.cpu().numpy())
Rf = Rb.reshape(3*N, 6)
gen = torch.Generator(device='cpu').manual_seed(0)
rk = [float(Kv(torch.as_tensor(Rb[:, :, j], device=dev)).norm()/math.sqrt((Rf[:, j]*Mop(Rf[:, j:j+1])[:, 0]).sum())) for j in range(6)]
vr = torch.randn(N, 3, generator=gen, dtype=f64).to(dev); vrf = vr.cpu().numpy().reshape(-1)
krand = float(Kv(vr).norm())/math.sqrt(float((vrf*Mop(vrf[:, None])[:, 0]).sum()))
print('%s: ndof %d; rigid modes |K r| / |r|_M %s against a random v %.3e' % (mode, N, np.array2string(np.array(rk), precision=2), krand), flush=True)
col = colouring(st.i.cpu().numpy(), st.j.cpu().numpy(), n)
if mode == 'cols':
    col = np.concatenate([col, col[W.cap_cells.cpu().numpy()]+col.max()+1])          # (a column's colour: its cell's, apart from the sites')
Kbb = probe_blocks(W, X, col)
W.elastic_mode()
sig = (2*math.pi*3.)**2                                                              # (the shift: about the low modes' lam)
Md = np.asarray(M.diagonal())
Pb = torch.linalg.inv(Kbb+sig*torch.as_tensor(Md, device=dev)[:, None, None]*torch.eye(3, dtype=f64, device=dev)).cpu().numpy()
A = LinearOperator((3*N, 3*N), matvec=lambda a: Kv(torch.as_tensor(a.reshape(N, 3), device=dev)).cpu().numpy().reshape(-1),
                   matmat=lambda a: np.stack([Kv(torch.as_tensor(a[:, j].reshape(N, 3), device=dev)).cpu().numpy().reshape(-1) for j in range(a.shape[1])], 1), dtype=np.float64)
B = LinearOperator((3*N, 3*N), matvec=lambda a: Mop(a[:, None])[:, 0], matmat=Mop, dtype=np.float64)
T = LinearOperator((3*N, 3*N), matvec=lambda a: np.einsum('nij,nj->ni', Pb, a.reshape(N, 3)).reshape(-1),
                   matmat=lambda a: np.einsum('nij,njk->nik', Pb, a.reshape(N, 3, -1)).reshape(3*N, -1), dtype=np.float64)
X0 = np.random.default_rng(0).standard_normal((3*N, K_+8))
lam, V = lobpcg(A, X0, B=B, M=T, Y=Rf, largest=False, tol=1e-5, maxiter=600, verbosityLevel=0)
o = np.argsort(lam); lam, V = lam[o][:K_], V[:, o][:, :K_]
f = np.sqrt(np.maximum(lam, 0))/(2*math.pi)
ffem = np.load(ROOT/'out_inverse'/'freevib'/'prep.npz')['f']
share = []
for j in range(K_):
    v = V[:, j]; Mv = Mop(v[:, None])[:, 0]
    tot = float(v@Mv); vc = v.copy(); vc[:3*n] = 0.
    share.append(float(vc@Mop(vc[:, None])[:, 0])/tot)
print('lowest elastic modes (Hz), / FEM, columns\' share of the kinetic norm:', flush=True)
for j in range(K_):
    print('  %2d: %.4f Hz  (FEM %.4f: %.3f)  columns %.3f' % (j, f[j], ffem[j] if j < len(ffem) else float('nan'), f[j]/ffem[j] if j < len(ffem) else float('nan'), share[j]), flush=True)
np.savez(out, lam=lam, f=f, share=np.array(share), rigid=np.array(rk), krand=krand)
print('saved', out, flush=True)
