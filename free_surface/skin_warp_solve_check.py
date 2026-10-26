"""W2 check (2026-10-23): SkinWarp with persistent buffers and a CUDA graph (prepare / grad), central-difference
Hessian-vector products against torch's, and the trust-region skin solve on Warp (warp_solver; preconditioner: probed
K_rest blocks per body) on the test of skin_tr_check.py: cold start at the 18.3 mm + 1 rad state, then 5 time steps of
1 / 4800 s; products and wall time. tolerance(): on steps 2-5, from the extrapolated skin, solves to 1e-2 / 1e-3 / 1e-4 /
1e-6 and the error of the bodies' elastic force -dE/dx against the converged skin (1e-10)."""
import math
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from experiments.dem_fem_mode_match import interpolate
from free_surface.skin_contact_check import setup, f64, dev
from free_surface.skin_warp import SkinWarp, warp_solver


def rot(axis, ang):
    k = axis/axis.norm()
    K = torch.tensor([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], dtype=f64, device=dev)
    return torch.eye(3, dtype=f64, device=dev)+math.sin(ang)*K+(1-math.cos(ang))*K@K


def main():
    out = Path('output/round01_exact')
    st, _ = setup('foam', 'boundary_skin.npz', 'shepard0')
    X, Y, npt = st.X, st.Y, st.npt; v3 = 3*npt; n = st.n
    th = torch.zeros(n, 9, dtype=f64, device=dev)
    W = SkinWarp(st)
    W.prepare()
    gen = torch.Generator(device=dev).manual_seed(0)
    size = float((X.max(0).values-X.min(0).values).max())
    kv = torch.randn(3, 3, generator=gen, dtype=f64, device=dev)*2*math.pi/size
    amp = torch.randn(3, 3, generator=gen, dtype=f64, device=dev)*.03*size
    ph = torch.rand(3, generator=gen, dtype=f64, device=dev)*2*math.pi
    field = lambda Z: torch.sin(Z@kv.T+ph)@amp.T
    x1, y1 = X+field(X), Y+field(Y)
    e_a, gx_a, gy_a = W.energy_grad(x1, y1)
    e_b, gx_b, gy_b = W.grad(x1, y1)
    torch.cuda.synchronize(); t = time.time()
    for _ in range(20):
        W.grad(x1, y1)
    torch.cuda.synchronize(); tg = (time.time()-t)/20
    print('graph gradient: energy %.1e, grad_x %.1e, grad_y %.1e relative to the W1 call; %.4f s per call' % (
        abs(float(e_b)-float(e_a.sum()))/float(e_a.sum()), float((gx_b-gx_a).norm()/gx_a.norm()), float((gy_b-gy_a).norm()/gy_a.norm()), tg), flush=True)
    gE = torch.func.grad(lambda yy: st.energy_xy(th, x1, yy.view(-1, 3)))
    V = torch.randn(3, v3, generator=gen, dtype=f64, device=dev); V = V/V.abs().amax(1, keepdim=True)
    Hv = torch.stack([torch.func.jvp(gE, (y1.reshape(-1),), (v,))[1] for v in V])
    for h in (1e-4, 1e-5, 1e-6, 1e-7, 1e-8):
        Hw = W.hvp_fd(x1, y1, V, h)
        print('central difference step %.0e m (largest skin move): |Hv_warp - Hv_torch| / |Hv_torch| %s' % (
            h, ' '.join('%.1e' % float((Hw[k]-Hv[k]).norm()/Hv[k].norm()) for k in range(3))), flush=True)
    # the skin solve on Warp (central differences at 1e-7 m), preconditioner: probed K_rest blocks per body
    bsk = dict(np.load(out/'boundary_skin.npz')); g = dict(np.load(out/'dem_exact_single'/'geometry.npz'))
    S = warp_solver(st, bsk, g['body_of_cell'], W, 1e-7)
    z = np.load(out/'skin_Kyy_rest.npz'); P_, Q_, vals = z['P'], z['Q'], z['vals']
    rows = (3*P_[:, None, None]+np.arange(3)[None, :, None]).repeat(3, 2); cols = (3*Q_[:, None, None]+np.arange(3)[None, None, :]).repeat(3, 1)
    K = sp.coo_matrix((vals.ravel(), (rows.ravel(), cols.ravel())), shape=(v3, v3)).tocsr(); K = (.5*(K+K.T)).tocsr()
    prec = S.preconditioner(S.blocks_probed(K, S.bodies), S.bodies)
    cache = np.load(out/'fem_modes_h0.008.npz')
    fem = dict(nodes=cache['nodes'], node_grid=cache['node_grid'], lo=cache['lo'], h=float(cache['h']), masses=cache['masses'])
    Ux, _ = interpolate(fem, cache['V'][:, :6], X.cpu().numpy())
    U = torch.as_tensor(Ux.transpose(2, 0, 1), dtype=f64, device=dev)
    rng = np.random.default_rng(1)
    am = torch.as_tensor(rng.standard_normal(6), dtype=f64, device=dev); phs = torch.as_tensor(rng.uniform(0, 2*np.pi, 6), dtype=f64, device=dev)
    om = torch.as_tensor(2*np.pi*cache['f'][:6], dtype=f64, device=dev)
    disp = lambda t: torch.einsum('k,kni->ni', am*torch.sin(om*t+phs), U)
    am = am*.03/float(max(disp(t_).norm(dim=1).max() for t_ in np.linspace(0, .5, 51)))
    cen = X.mean(0); axis = torch.tensor([.3, 1., -.2], dtype=f64, device=dev); spin = torch.tensor([1., 0, .4], dtype=f64, device=dev)
    Qt = lambda t: rot(spin, 3.*t)@rot(axis, 1.)
    xt = lambda t: (X+disp(t)-cen)@Qt(t).T+cen
    x0 = xt(0.); y0 = (Y-cen)@Qt(0.).T+cen
    for _ in range(5):
        y0 = S.affine_skin(x0, y0)
    ref0 = float(S.grad(x0, y0).norm())
    torch.cuda.synchronize(); t1 = time.time()
    yc, s_ = S.solve(x0, y0, prec, ref0, tol=1e-6, maxit=600)
    torch.cuda.synchronize()
    print('warp cold start: products to 1e-2 / 1e-4 / 1e-6 %s / %s / %s (torch: 85 / 167 / 171); %d outer, %d rejected; %.2f s' % (
        s_['hits'][1e-2], s_['hits'][1e-4], s_['hits'][1e-6], s_['outer'], s_['rejected'], time.time()-t1), flush=True)
    yc, _ = S.solve(x0, yc, prec, ref0, tol=1e-9, maxit=600)
    yp = yc
    for s in range(1, 6):
        x = xt(s/4800)
        yi = 2*yc-yp
        rf = float(S.grad(x, S.affine_skin(x, yi)).norm())
        torch.cuda.synchronize(); t1 = time.time()
        y4, s4 = S.solve(x, yi, prec, rf, tol=1e-4, maxit=300)
        torch.cuda.synchronize(); t4 = time.time()-t1
        y6, s6 = S.solve(x, y4, prec, rf, tol=1e-6, maxit=300)
        torch.cuda.synchronize(); t6 = time.time()-t1
        print('step %d: to 1e-4 %d products (%.3f s), to 1e-6 %d more (%.3f s in all)' % (s, s4['products'], t4, s6['products'], t6), flush=True)
        yn, _ = S.solve(x, y6, prec, rf, tol=1e-9, maxit=300)
        yp, yc = yc, yn
        if s >= 2:
            f_ref = W.grad(x, yn)[1]
            line = 'step %d, start |dE/dy| %.1e of the affine skin; body force error / products:' % (s, float(S.grad(x, yi).norm())/rf)
            for tol in (1e-2, 1e-3, 1e-4, 1e-6):
                yt, st_ = S.solve(x, yi, prec, rf, tol=tol, maxit=300)
                line += ' %.0e: %.1e / %d;' % (tol, float((W.grad(x, yt)[1]-f_ref).norm()/f_ref.norm()), st_['products'])
            line += ' extrapolated start alone: %.1e' % float((W.grad(x, yi)[1]-f_ref).norm()/f_ref.norm())
            print(line, flush=True)


if __name__ == '__main__':
    main()
