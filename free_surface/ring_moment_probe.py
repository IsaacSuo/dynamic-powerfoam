"""Where a step's cost comes from, at a loaded state: no full run.
  1 the conditioning: Lanczos on P^-1 H at the frozen problem of the last step, H as the CG's products; the extreme Ritz values, where their vectors live, which energy terms carry them; H's symmetry
  2 what a new cut does to the force at the same x
  3 the radii's Newton iterations from the last radii against an extrapolated start"""
import math

import numpy as np
import torch
import warp as wp

f64 = torch.float64


def probe(W, ST, mass, tol, wref, V0, nl=150):
    dev = W.dev; n = W.n
    h_ = float(np.median(V0)**(1/3))
    recs = []
    uo = W.update_ul

    def rec(x):
        uo(x)
        recs.append((x.detach().clone(), W._ulr.clone(), W._recut.stats.get('geom', 0)))
    W.update_ul = rec
    for k in range(3):
        ST.step(tol, lambda g_: .5*tol)
        print('step: outer %d, products %s, first |g| / tol %.0f' % (ST.stats['outer'], sum(t_[0] for t_ in ST.stats['trace']), ST.stats['trace'][0][4]/tol if ST.stats['trace'] else 0.), flush=True)
        if k < 2:
            ST.end()
    W.update_ul = uo
    z = ST.tz.clone()
    _, g0 = W.grad(z); g0 = g0.clone()
    cm = (4*mass/ST.dt**2)
    Pi = wp.to_torch(ST.a['Pinv'], requires_grad=False).clone()
    P = torch.linalg.inv(Pi.cpu()).to(dev)

    def H(d):
        s_ = float(d.abs().max()); s_ = s_ if s_ > 0 else 1.
        _, g1 = W.grad(z+1e-9*d/s_)
        return cm[:, None]*d+(g1-g0)*s_/1e-9

    pre = lambda r: torch.einsum('fab,fb->fa', Pi, r)
    torch.manual_seed(1)
    u_, v_ = torch.randn(n, 3, dtype=f64, device=dev), torch.randn(n, 3, dtype=f64, device=dev)
    a_, b_ = float((u_*H(v_)).sum()), float((v_*H(u_)).sum())
    print('symmetry of H: |u H v - v H u| / |u H v| = %.2e' % (abs(a_-b_)/abs(a_)), flush=True)
    r = torch.randn(n, 3, dtype=f64, device=dev)
    zr = pre(r); be = math.sqrt(float((r*zr).sum()))
    V, U, al, bt = [], [], [], []
    for j in range(nl):
        v = zr/be; u = r/be
        V.append(v); U.append(u)
        Hv = H(v)
        a = float((v*Hv).sum()); al.append(a)
        r = Hv-a*u-(be*U[-2] if j > 0 else 0.)
        for k in range(len(U)):                                                           # (full reorthogonalisation)
            r = r-float((V[k]*r).sum())*U[k]
        zr = pre(r); be = math.sqrt(max(float((r*zr).sum()), 1e-300)); bt.append(be)
    T = torch.diag(torch.tensor(al, dtype=f64))+torch.diag(torch.tensor(bt[:-1], dtype=f64), 1)+torch.diag(torch.tensor(bt[:-1], dtype=f64), -1)
    th, S = torch.linalg.eigh(T)
    print('P^-1 H: Ritz values after %d: smallest %s, largest %s; condition %.3e; sqrt %.0f' % (nl, np.round(th[:4].numpy(), 4).tolist(), np.round(th[-6:].numpy(), 2).tolist(), float(th[-1]/th[0]), math.sqrt(float(th[-1]/th[0]))), flush=True)
    Vm = torch.stack(V, 0)
    lo, hi = W.tlo, W.thi; ms = W._m_s; T_ = W.tr; Y = W.ty
    wi, wj = T_['wi'][:ms], T_['wj'][:ms]
    d = z[hi]-z[lo]; L = d.norm(dim=1)
    wmax = torch.zeros(n, dtype=f64, device=dev); wmax.scatter_reduce_(0, lo, wi+wj, 'amax'); wmax.scatter_reduce_(0, hi, wi+wj, 'amax')
    Lmin = torch.full((n,), 1e9, dtype=f64, device=dev); Lmin.scatter_reduce_(0, lo, L, 'amin'); Lmin.scatter_reduce_(0, hi, L, 'amin')
    cap = W.tul['Mc'].abs().sum((1, 2)) > 0
    tn = W.tbd > 0
    Vmed = float(np.median(V0))
    for name, idx in (('largest', -1), ('2nd largest', -2), ('smallest', 0)):
        y = (S[:, idx].to(dev)[:, None, None]*Vm).sum(0)
        y = y/math.sqrt(float((y*torch.einsum('fab,fb->fa', P, y)).sum()))
        sh = (y*torch.einsum('fab,fb->fa', P, y)).sum(1)
        pr = 1/float((sh*sh).sum())
        top = torch.argsort(sh, descending=True)[:6].tolist()
        print('%s, %.3g: lives on %.1f cells; of it touching the ground %.3f, surface %.3f, V under .1 median %.3f' % (name, float(th[idx]), pr, float(sh[tn].sum()), float(sh[cap].sum()), float(sh[torch.as_tensor(V0, device=dev) < .1*Vmed].sum())), flush=True)
        for c in top:
            print('    cell %d: share %.3f; V %.3f; nearest spring L / h %.4f; largest w / median %.1f; beta %.2f; ground %d stuck %d; surface %d; group %d' % (c, float(sh[c]), float(V0[c])/Vmed, float(Lmin[c])/h_, float(wmax[c]/(wi+wj).median()), float(Y['be'][W._gid[c]]), int(tn[c]), int(W._stuck[c]) if hasattr(W, '_stuck') else 0, int(cap[c]), int(W._gid[c] != c)), flush=True)
        # which terms carry y^T H y
        eps = 1e-4*h_/float(y.abs().max())
        W.grad(z+eps*y); ep = W.tsum.clone().sum(1) if W.tsum.dim() > 1 else W.tsum.clone()
        W.grad(z-eps*y); em = W.tsum.clone().sum(1) if W.tsum.dim() > 1 else W.tsum.clone()
        W.grad(z); e0 = W.tsum.clone().sum(1) if W.tsum.dim() > 1 else W.tsum.clone()
        q = (ep+em-2*e0)/eps**2
        print('    y H y by term (0 tensor, 4 remainder, 5 outer, 1 / 3 volume, 6 ground, 8 hold): %s; inertia %.3e' % ({k_: float('%.3e' % float(q[k_])) for k_ in range(len(q)) if abs(float(q[k_])) > 0}, float((cm[:, None]*y*y).sum())), flush=True)
    # the same problem on the true diagonal blocks
    from free_surface.ring_moment import exact_blocks
    K, npr = exact_blocks(W, z, g0, cm)
    W.grad(z)
    ev = torch.linalg.eigvalsh(K.cpu())
    print('the true blocks by %d products; smallest eigenvalue over cm: min %.3f; how P differed: largest eigenvalue of P^-1 K over the cells %.1f, smallest %.4f' % (npr, float((ev[:, 0].to(dev)/cm).min()), float(torch.linalg.eigvals(Pi.cpu()@K.cpu()).real.max()), float(torch.linalg.eigvals(Pi.cpu()@K.cpu()).real.min())), flush=True)
    Pix = torch.linalg.inv(K.cpu()).to(dev)
    pre2 = lambda r: torch.einsum('fab,fb->fa', Pix, r)
    r = torch.randn(n, 3, dtype=f64, device=dev)
    zr = pre2(r); be = math.sqrt(float((r*zr).sum()))
    V2, U2, al2, bt2 = [], [], [], []
    for j in range(100):
        v = zr/be; u = r/be
        V2.append(v); U2.append(u)
        Hv = H(v)
        a = float((v*Hv).sum()); al2.append(a)
        r = Hv-a*u-(be*U2[-2] if j > 0 else 0.)
        for k in range(len(U2)):
            r = r-float((V2[k]*r).sum())*U2[k]
        zr = pre2(r); be = math.sqrt(max(float((r*zr).sum()), 1e-300)); bt2.append(be)
    T2 = torch.diag(torch.tensor(al2, dtype=f64))+torch.diag(torch.tensor(bt2[:-1], dtype=f64), 1)+torch.diag(torch.tensor(bt2[:-1], dtype=f64), -1)
    th2, S2 = torch.linalg.eigh(T2)
    print('on the true blocks: Ritz smallest %s, largest %s; condition %.3e, sqrt %.0f' % (np.round(th2[:4].numpy(), 4).tolist(), np.round(th2[-4:].numpy(), 2).tolist(), float(th2[-1]/th2[0]), math.sqrt(float(th2[-1]/th2[0]))), flush=True)
    y = (S2[:, 0].to(dev)[:, None, None]*torch.stack(V2, 0)).sum(0)
    sh = (y*torch.einsum('fab,fb->fa', K, y)).sum(1); sh = sh/sh.sum()
    print('  its smallest lives on %.1f cells; ground %.3f, surface %.3f; top cells %s' % (1/float((sh*sh).sum()), float(sh[tn].sum()), float(sh[cap].sum()), [(int(c), round(float(sh[c]), 3), round(float(Lmin[c])/h_, 3), round(float(V0[c])/Vmed, 3)) for c in torch.argsort(sh, descending=True)[:5]]), flush=True)
    # a new cut at the same x
    ST.end()
    x1 = ST.tx.clone(); W.update_ul(x1)
    W.grad(x1); W.reread(); _, f0 = W.grad(x1); f0 = f0.clone()
    dx = x1-recs[-2][0]
    x2 = x1+dx
    _, fa = W.grad(x2); fa = fa.clone()
    W._ret.zero_()
    W.update_ul(x2)
    _, fb = W.grad(x2); fb = fb.clone()
    de = (fb-fa); N_ = float(wref)
    gr = W.tbd > 0
    print('a new cut at x + the last step: |f(new cut) - f(old cut)| %.3e = %.1f tol = %.3f weights; the motion itself |f(old cut, x + dx) - f(x)| %.3e; ground cells\' part %.3e, surface %.3e, deep %.3e' % (float(de.norm()), float(de.norm())/tol, float(de.norm())/N_, float((fa-f0).norm()), float(de[gr].norm()), float(de[cap & ~gr].norm()), float(de[~cap & ~gr].norm())), flush=True)
    # the radii's start
    geom = W._recut.geometry; cnt = [0]

    def g_(*a, **kw):
        cnt[0] += 1
        return geom(*a, **kw)
    W._recut.geometry = g_
    xs = [r_[0] for r_ in recs]; rs = [r_[1] for r_ in recs]
    Vt = W.V0*torch.exp(W.tul['Ln']).clamp(min=.02)
    for name, st_ in (('last radii', rs[1]), ('extrapolated', torch.sqrt((2*rs[1]**2-rs[0]**2).clamp(min=(.25*rs[1])**2)))):
        cnt[0] = 0
        _, _, _, it, nrep = W._recut.solve_radii(xs[2], st_, Vt, tol=W._ul_tol, r_base=W._rr0)
        print('radii to %.0e from %s: Newton %d, geometries %d, repairs %d' % (W._ul_tol, name, it, cnt[0], nrep), flush=True)
