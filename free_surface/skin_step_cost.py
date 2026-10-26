"""What one time step's skin solve costs, and with which preconditioner (2026-10-23, new foam, single bodies, skin C:
shepard0, lever c). The skin is massless and solved every step: given the bodies x, find y with dE/dy (x, y) = 0.
  K_rest     K_yy at rest by coloured probing (two skin points couple when their cells are within two bonds, through
             Gamma_B; a colour holds no two points coupled to a common point), saved to skin_Kyy_rest.npz; coupling strength
             by where it comes from (the two points share a cell / their cells are bond neighbours / two bonds apart):
             block norm |K_pq| / sqrt(|K_pp| |K_qq|)
  precond    on the GPU, turned by the bodies' rotations (polar of F_B): (a) the 3 x 3 block of each skin point; (b) the
             dense block of each region's skin points, added over the regions (additive Schwarz); for the cold start also
             the PARDISO factorisation of K_rest (CPU, seconds a solve: a reference only)
  motion     the first 6 FEM modes (random amplitudes and phases, largest displacement --amp), turned by 1 rad and spinning
             at 3 rad/s, 1 / 4800 s a step
  measured   cold start (from the cells' affine skin) to 1e-6 (capped: point blocks 500, PARDISO 100 products); per step
             from the extrapolated skin (region blocks), products to 1e-2 / 1e-4 / 1e-6 of |dE/dy| at the cells' affine skin; with a fixed 1, 2 or 3 products per step (the skin carried
             from step to step), the error of the body forces against the converged skin
  timing     a Hessian-vector product (torch, batch 1) and each preconditioner"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import pypardiso
import scipy.sparse as sp
import torch

from experiments.dem_fem_mode_match import interpolate
from experiments.foam_surface_completion import polar_newton
from experiments.pacnerf_dem_forward import build as dem_build
from free_surface.boundary_skin import BoundaryTerms

f64, dev = torch.float64, 'cuda'


def rot(axis, ang):
    k = axis/axis.norm()
    K = torch.tensor([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], dtype=f64, device=dev)
    return torch.eye(3, dtype=f64, device=dev)+math.sin(ang)*K+(1-math.cos(ang))*K@K


def sync_time(f, reps=5):
    f(); torch.cuda.synchronize(); t = time.time()
    for _ in range(reps):
        f()
    torch.cuda.synchronize()
    return (time.time()-t)/reps


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--out', type=Path, default=Path('output/round01_exact'))
    a.add_argument('--skin', default='boundary_skin.npz')
    a.add_argument('--steps', type=int, default=30)
    a.add_argument('--batch', type=int, default=8)
    a.add_argument('--amp', type=float, default=.03)
    args = a.parse_args()
    t0 = time.time()
    E, nu = 23008.38, .3008
    cells = np.load(args.out/'dem_exact_single'/'cells.npz'); p, r = cells['points'], cells['radii']
    g = dict(np.load(args.out/'dem_exact_single'/'geometry.npz')); bsk = dict(np.load(args.out/args.skin))
    m = dem_build(g, E, nu, dev, 'noslip', .25, .3, f64, kinematics='exact', nodalvol=True, nonlinear=False, ogden_alpha=2.)
    st = BoundaryTerms(m, g, p, r, bsk, lever='c')
    n, npt = st.n, st.npt; v3 = 3*npt
    th = torch.zeros(n, 9, dtype=f64, device=dev)
    X, Y = st.X, st.Y
    gy = lambda x, y: torch.func.grad(lambda yy: st.energy_xy(th, x, yy.view(-1, 3)))(y.reshape(-1))
    fx = lambda x, y: torch.func.grad(lambda xx: st.energy_xy(th, xx.view(-1, 3), y))(x.reshape(-1))
    def hvp(x, y):
        f = lambda yy: torch.func.grad(lambda z: st.energy_xy(th, x, z.view(-1, 3)))(yy)
        y0 = y.reshape(-1)
        return torch.func.vmap(lambda v: torch.func.jvp(f, (y0,), (v,))[1])
    print('bodies %d, skin points %d; setup %.0f s' % (n, npt, time.time()-t0), flush=True)
    # coupling: cells within two bonds (Gamma_B); colours from the square of the point coupling
    pairs = g['pairs']
    A1 = sp.coo_matrix((np.ones(2*len(pairs)), (np.r_[pairs[:, 0], pairs[:, 1]], np.r_[pairs[:, 1], pairs[:, 0]])), shape=(n, n)).tocsr()+sp.eye(n, format='csr')
    A1.data[:] = 1
    A2 = (A1@A1).astype(bool).astype(float)
    Pc = sp.coo_matrix((np.ones(len(bsk['ent_pid'])), (bsk['ent_pid'], bsk['ent_site'])), shape=(npt, n)).tocsr(); Pc.data[:] = 1
    C2 = (Pc@A2@Pc.T).tocsr(); C2.data[:] = 1
    C2c = C2.tocoo(); P_, Q_ = C2c.row, C2c.col
    kfile = args.out/'skin_Kyy_rest.npz'
    if kfile.exists():
        z = np.load(kfile); assert np.array_equal(z['P'], P_) and np.array_equal(z['Q'], Q_); vals = z['vals']
        print('K_rest from %s' % kfile, flush=True)
    else:
        t1 = time.time()
        C4 = (C2@C2).tocsr()                                                         # points that share a coupled point
        colour = -np.ones(npt, int)
        for q in range(npt):
            used = colour[C4.indices[C4.indptr[q]:C4.indptr[q+1]]]
            used = np.unique(used[used >= 0])
            k = 0
            for u in used:
                if u != k:
                    break
                k += 1
            colour[q] = k
        ncol = int(colour.max())+1
        H0 = hvp(X, Y)
        col_t = torch.as_tensor(colour, device=dev)
        Pt, Qt = torch.as_tensor(P_, device=dev), torch.as_tensor(Q_, device=dev)
        ccol = col_t[Qt]
        order = torch.argsort(ccol); bounds = torch.searchsorted(ccol[order], torch.arange(ncol+1, device=dev))
        vals_t = torch.zeros(len(P_), 3, 3, dtype=f64, device=dev)
        probes = [(k, d) for k in range(ncol) for d in range(3)]
        for s0 in range(0, len(probes), args.batch):
            ch = probes[s0:s0+args.batch]
            V = torch.zeros(len(ch), npt, 3, dtype=f64, device=dev)
            for b, (k, d) in enumerate(ch):
                V[b, col_t == k, d] = 1.
            Kv = H0(V.reshape(len(ch), -1)).view(len(ch), npt, 3)
            for b, (k, d) in enumerate(ch):
                sel = order[bounds[k]:bounds[k+1]]
                vals_t[sel, :, d] = Kv[b, Pt[sel]]
        vals = vals_t.cpu().numpy()
        np.savez(kfile, P=P_, Q=Q_, vals=vals)
        print('K_rest assembled: %d coupled point pairs (%.0f per point), %d colours, %d probes (%.0f s); saved %s' % (
            len(P_), len(P_)/npt, ncol, len(probes), time.time()-t1, kfile), flush=True)
    rows = (3*P_[:, None, None]+np.arange(3)[None, :, None]).repeat(3, 2)
    cols = (3*Q_[:, None, None]+np.arange(3)[None, None, :]).repeat(3, 1)
    K = sp.coo_matrix((vals.ravel(), (rows.ravel(), cols.ravel())), shape=(v3, v3)).tocsr()
    asym = float(abs(K-K.T).max()/abs(K).max())
    K = (.5*(K+K.T)).tocsr()
    H0 = hvp(X, Y)
    vt = torch.randn(4, v3, generator=torch.Generator(device=dev).manual_seed(3), dtype=f64, device=dev)
    ref = H0(vt).cpu().numpy(); got = (K@vt.cpu().numpy().T).T
    res_pt = np.linalg.norm((got-ref).reshape(4, npt, 3), axis=2).max(0)
    print('K_rest: %d nonzeros (%.0f per row), asymmetry %.1e, |K v - Hessian-vector product| / |K v| %.1e (largest at %d points above 1e-3 of the max)' % (
        K.nnz, K.nnz/v3, asym, float(np.abs(got-ref).max()/np.abs(ref).max()), int((res_pt > 1e-3*res_pt.max()).sum())), flush=True)
    # coupling strength by source
    S0 = (Pc@Pc.T).tocsr(); S1 = (Pc@A1@Pc.T).tocsr()
    cat = np.where(np.asarray(S0[P_, Q_]).ravel() > 0, 0, np.where(np.asarray(S1[P_, Q_]).ravel() > 0, 1, 2))
    nb = np.linalg.norm(vals.reshape(-1, 9), axis=1)
    dg = np.zeros(npt); dg[P_[P_ == Q_]] = nb[P_ == Q_]
    rel = nb/np.sqrt(dg[P_]*dg[Q_])
    off = P_ != Q_; tot = (nb[off]**2).sum()
    for c_, nm_ in enumerate(('share a cell', 'cells bond neighbours', 'cells two bonds apart')):
        s_ = off & (cat == c_)
        print('  coupling, points whose %s: %d pairs (%.0f per point); |K_pq| / sqrt(|K_pp| |K_qq|) median %.1e, 99%% %.1e, max %.1e; share of the off-diagonal |K|^2 %.3f' % (
            nm_, s_.sum(), s_.sum()/npt, np.median(rel[s_]), np.quantile(rel[s_], .99), rel[s_].max(), (nb[s_]**2).sum()/tot), flush=True)
    # preconditioners
    t1 = time.time()
    Dg = torch.as_tensor(vals[P_ == Q_][np.argsort(P_[P_ == Q_])], dtype=f64, device=dev)
    Dinv = torch.linalg.inv(.5*(Dg+Dg.transpose(1, 2)))
    _, first = np.unique(bsk['ent_pid'], return_index=True)
    boc = np.asarray(g['body_of_cell'])
    pt_body = torch.as_tensor(boc[bsk['ent_site'][first]], device=dev)
    off_ = bsk['part_off']; P = len(off_)-1
    loc = [bsk['ent_pid'][off_[k]:off_[k+1]] for k in range(P)]
    Smax = max(len(l) for l in loc)
    L = -np.ones((P, Smax), int)
    for k, l in enumerate(loc):
        L[k, :len(l)] = l
    blocks = np.zeros((P, 3*Smax, 3*Smax))
    for k, l in enumerate(loc):
        i3 = (3*l[:, None]+np.arange(3)).ravel()
        blocks[k, :len(i3), :len(i3)] = K[i3][:, i3].toarray()
        blocks[k, len(i3):, len(i3):] = np.eye(3*Smax-len(i3))
    Bt = torch.as_tensor(blocks, dtype=f64, device=dev)
    Binv = torch.linalg.inv(.5*(Bt+Bt.transpose(1, 2)))
    Lt = torch.as_tensor(L, device=dev); Lc = Lt.clamp(min=0); Lm = (Lt >= 0).to(f64)
    reg_body = torch.as_tensor(boc[bsk['part_site']], device=dev)
    print('preconditioners: point blocks, %d region blocks (up to %d points); built in %.0f s' % (P, Smax, time.time()-t1), flush=True)
    def prec_point(Rb):
        Rp = Rb[pt_body]
        Bi = torch.einsum('qij,qjk,qlk->qil', Rp, Dinv, Rp)
        return lambda rv: torch.einsum('qij,bqj->bqi', Bi, rv.view(rv.shape[0], npt, 3)).reshape(rv.shape[0], -1)
    def prec_region(Rb):
        Rr = Rb[reg_body]
        def f(rv):
            B = rv.shape[0]
            rl = rv.view(B, npt, 3)[:, Lc]*Lm[None, :, :, None]                        # (B, P, Smax, 3)
            rl = torch.einsum('pji,bpsj->bpsi', Rr, rl).reshape(B, P, -1)
            zl = torch.einsum('pij,bpj->bpi', Binv, rl).view(B, P, Smax, 3)
            zl = torch.einsum('pij,bpsj->bpsi', Rr, zl)*Lm[None, :, :, None]
            return torch.zeros(B, npt, 3, dtype=f64, device=dev).index_add(1, Lc.reshape(-1), zl.reshape(B, -1, 3)).reshape(B, -1)
        return f
    solver = pypardiso.PyPardisoSolver(); solver.set_matrix_type(11); solver.factorize(K)
    def prec_factor(Q):
        def f(rv):
            rr = (rv.view(rv.shape[0], npt, 3)@Q).reshape(rv.shape[0], -1)
            zz = torch.as_tensor(solver.solve(K, rr.T.cpu().numpy()).T.reshape(rv.shape[0], npt, 3), dtype=f64, device=dev)
            return (zz@Q.T).reshape(rv.shape[0], -1)
        return f
    # motion
    cache = np.load(args.out/'fem_modes_h0.008.npz')
    fem = dict(nodes=cache['nodes'], node_grid=cache['node_grid'], lo=cache['lo'], h=float(cache['h']), masses=cache['masses'])
    nm = 6
    Ux, _ = interpolate(fem, cache['V'][:, :nm], X.cpu().numpy())
    U = torch.as_tensor(Ux.transpose(2, 0, 1), dtype=f64, device=dev)
    rng = np.random.default_rng(1)
    amp = torch.as_tensor(rng.standard_normal(nm), dtype=f64, device=dev); phase = torch.as_tensor(rng.uniform(0, 2*np.pi, nm), dtype=f64, device=dev)
    om = torch.as_tensor(2*np.pi*cache['f'][:nm], dtype=f64, device=dev)
    disp = lambda t: torch.einsum('k,kni->ni', amp*torch.sin(om*t+phase), U)
    amp = amp*args.amp/float(max(disp(t_).norm(dim=1).max() for t_ in np.linspace(0, .5, 51)))
    cen = X.mean(0); axis = torch.tensor([.3, 1., -.2], dtype=f64, device=dev); spin_axis = torch.tensor([1., 0, .4], dtype=f64, device=dev)
    Qt_ = lambda t: rot(spin_axis, 3.*t)@rot(axis, 1.)
    xt = lambda t: (X+disp(t)-cen)@Qt_(t).T+cen
    vmax = float(((xt(1e-5)-xt(0.))/1e-5).norm(dim=1).max())
    dt = 1/4800
    cell_of_pt = torch.as_tensor(bsk['ent_site'][first], device=dev)
    def affine_skin(x, y):
        F = st.F(x, y)
        return x[cell_of_pt]+torch.einsum('qij,qj->qi', F[cell_of_pt], Y-X[cell_of_pt])
    def solve(x, y, mk, ref, tol, maxit=3000, marks=(1e-2, 1e-4, 1e-6), label=None):
        """Newton + PCG (preconditioner mk(R_B) turned at the current state); products used, and when |dE/dy| / ref first
        fell below each mark (counted at the ends of Newton steps)"""
        hits = {mk_: None for mk_ in marks}; nh = 0; t_lab = time.time()
        gr = gy(x, y); res = float(gr.norm())/ref
        for mk_ in marks:
            if res <= mk_:
                hits[mk_] = 0
        while res > tol and nh < maxit:
            H = hvp(x, y); prec = mk(polar_newton(st.F(x, y)))
            b = -gr[None]; zv = torch.zeros_like(b); rr = b.clone(); zz = prec(rr); pv = zz.clone(); rz = (rr*zz).sum()
            while nh < maxit:
                Ap = H(pv); nh += 1
                al = rz/(pv*Ap).sum(); zv = zv+al*pv; rr = rr-al*Ap
                if label is not None and nh % 100 == 0:
                    print('   %s: %d products, linear residual %.1e (%.0f s)' % (label, nh, float(rr.norm())/ref, time.time()-t_lab), flush=True)
                if float(rr.norm())/ref <= max(tol, .01*res):                         # inexact Newton
                    break
                zz = prec(rr); rz1 = (rr*zz).sum(); pv = zz+(rz1/rz)*pv; rz = rz1
            y = y+zv.view(-1, 3)
            gr = gy(x, y); res = float(gr.norm())/ref
            for mk_ in marks:
                if hits[mk_] is None and res <= mk_:
                    hits[mk_] = nh
        return y, nh, hits, res
    def fixed(x, y, mk, k):
        """one Newton step with exactly k preconditioned CG products"""
        gr = gy(x, y); H = hvp(x, y); prec = mk(polar_newton(st.F(x, y)))
        b = -gr[None]; zv = torch.zeros_like(b); rr = b.clone(); zz = prec(rr); pv = zz.clone(); rz = (rr*zz).sum()
        for i in range(k):
            Ap = H(pv); al = rz/(pv*Ap).sum(); zv = zv+al*pv; rr = rr-al*Ap
            if i < k-1:
                zz = prec(rr); rz1 = (rr*zz).sum(); pv = zz+(rz1/rz)*pv; rz = rz1
        return y+zv.view(-1, 3)
    precs = (('point blocks', prec_point), ('region blocks', prec_region))
    # cold start at t = 0
    x0 = xt(0.)
    ya = affine_skin(x0, Y)
    for _ in range(3):
        ya = affine_skin(x0, ya)
    ref0 = float(gy(x0, ya).norm())
    print('motion: largest displacement %.3f m, largest body speed %.2f m/s; |dE/dy| at the affine skin %.2e' % (args.amp, vmax, ref0), flush=True)
    xv = torch.randn(1, v3, dtype=f64, device=dev)
    print('one Hessian-vector product: %.3f s (torch, batch 1)' % sync_time(lambda: hvp(x0, ya)(xv)), flush=True)
    for (nm_, mk), cap in zip(precs+(('PARDISO factorisation (CPU)', lambda Rb: prec_factor(Qt_(0.))),), (500, 3000, 100)):
        t1 = time.time()
        _, nhc, hc, resc = solve(x0, ya, mk, ref0, 1e-6, maxit=cap, label='cold start, '+nm_)
        print('cold start, %-28s products to 1e-2 / 1e-4 / 1e-6: %s / %s / %s (residual %.1e after %d products, cap %d; %.0f s)' % (
            nm_, hc[1e-2], hc[1e-4], hc[1e-6], resc, nhc, cap, time.time()-t1), flush=True)
    # converged trajectory (reference)
    t1 = time.time()
    yref, refs, xs = [], [], []
    y_, _, _, _ = solve(x0, ya, prec_region, ref0, 1e-10)
    yp, yc = y_, y_
    for s in range(1, args.steps+1):
        x = xt(s*dt); xs.append(x)
        yi = 2*yc-yp
        rf = float(gy(x, affine_skin(x, yi)).norm()); refs.append(rf)
        ys, nh, _, _ = solve(x, yi, prec_region, rf, 1e-10)
        yref.append(ys); yp, yc = yc, ys
        print('   reference step %d: %d products (%.0f s)' % (s, nh, time.time()-t1), flush=True)
    fref = [fx(x_, y_) for x_, y_ in zip(xs, yref)]
    print('reference: converged skins (1e-10) over %d steps (%.0f s)' % (args.steps, time.time()-t1), flush=True)
    for nm_, mk in precs:
        t1 = time.time()
        line = '%-14s' % nm_
        if nm_ == 'region blocks':                                                   # point blocks: the cold start says enough
            cnt = []; yp, yc = y_, y_
            for s in range(1, args.steps+1):
                x = xs[s-1]
                yi = 2*yc-yp
                ys, nh, hits, res = solve(x, yi, mk, refs[s-1], 1e-6)
                cnt.append([hits[1e-2], hits[1e-4], hits[1e-6]])
                yp, yc = yc, ys
                print('   %s step %d: products to 1e-2 / 1e-4 / 1e-6 %s (%.0f s)' % (nm_, s, cnt[-1], time.time()-t1), flush=True)
            cnt = np.array(cnt, float)
            line += ' per step to 1e-2 / 1e-4 / 1e-6: median %s, max %s' % (np.median(cnt, 0).tolist(), cnt.max(0).tolist())
        for k in (1, 2, 3):
            yp, yc = y_, y_; e = []
            for s in range(1, args.steps+1):
                x = xs[s-1]
                yi = 2*yc-yp
                ys = fixed(x, yi, mk, k)
                if s in (1, 10, 20, args.steps):
                    e.append(float((fx(x, ys)-fref[s-1]).norm()/fref[s-1].norm()))
                yp, yc = yc, ys
            line += '; %d per step: body force error at steps 1/10/20/%d %s' % (k, args.steps, '/'.join('%.0e' % e_ for e_ in e))
            print('   %s, %d per step done (%.0f s)' % (nm_, k, time.time()-t1), flush=True)
        print(line+' (%.0f s)' % (time.time()-t1), flush=True)
    # extrapolation alone
    e = []; yp, yc = y_, y_
    for s in range(1, args.steps+1):
        yi = 2*yc-yp
        if s in (1, 10, 20, args.steps):
            e.append(float((fx(xs[s-1], yi)-fref[s-1]).norm()/fref[s-1].norm()))
        yp, yc = yc, yref[s-1]
    print('no solve (extrapolated from the converged skins): body force error at steps 1/10/20/%d %s' % (args.steps, '/'.join('%.0e' % e_ for e_ in e)), flush=True)
    H = hvp(x0, y_); Rb = polar_newton(st.F(x0, y_))
    pp, pr_ = prec_point(Rb), prec_region(Rb)
    print('timing: Hessian-vector product %.3f s (torch, batch 1); point blocks %.4f s; region blocks %.4f s; total %.0f s' % (
        sync_time(lambda: H(xv)), sync_time(lambda: pp(xv)), sync_time(lambda: pr_(xv)), time.time()-t0), flush=True)


if __name__ == '__main__':
    main()
