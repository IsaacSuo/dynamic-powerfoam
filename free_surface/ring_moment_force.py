"""At a real saved state of a run (nothing rebuilt): is the force at x what the frames' motion asks, and on a leg, per term, along its own motion and along straightening in the ground's plane."""
import math

import numpy as np
import torch
import warp as wp
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from free_surface.closure_warp import k_mid_ball

f64, dev = torch.float64, 'cuda'
TERMS = {0: 'tensor', 1: 'volume', 2: 'var', 4: 'remainder', 5: 'outer', 6: 'ground'}


def probe(W, ST, mass, fg, V0, X0, P, dt):
    n = W.n; A, bl = ST.a, W.ball; m3 = mass[:, None]
    wp.launch(k_mid_ball, bl['k'], inputs=[A['x'], bl['body'], bl['R'], bl['ng'], ST.c0, ST.g8, ST.cb], outputs=[bl['xn'], bl['kt'], bl['bound']])
    x, v, a = ST.tx.clone(), ST.tv.clone(), ST.ta.clone()
    W.update_ul(x)
    W.grad(x); W.reread(); _, g0 = W.grad(x); g0 = g0.clone()

    def terms(y):
        W.grad(y)
        t = W.tsum.clone(); t = t.sum(1) if t.dim() > 1 else t
        return {TERMS[k]: float(t[k]) for k in TERMS}
    res = m3*a-fg+g0                                                                       # (what the elastic force and gravity leave of m a: the viscous force and the step's form)
    wc = fg.norm(dim=1)
    print('t %.4f: per cell over its own weight: |elastic force| median %.1f, 90 %% %.1f; |m a - gravity| median %.1f; what is left median %.2f, 90 %% %.1f, 99 %% %.1f; speed median %.3f' % (float(W.tul['xn'].shape[0])*0+0., float((g0.norm(dim=1)/wc).median()), float(torch.quantile(g0.norm(dim=1)/wc, .9)), float(((m3*a-fg).norm(dim=1)/wc).median()), float((res.norm(dim=1)/wc).median()), float(torch.quantile(res.norm(dim=1)/wc, .9)), float(torch.quantile(res.norm(dim=1)/wc, .99)), float(v.norm(dim=1).median())), flush=True)
    zr = X0[:, 2]-X0[:, 2].min(); low = zr < .18
    idx = -np.ones(n, int); idx[low] = np.arange(low.sum())
    m = low[P[:, 0]] & low[P[:, 1]]; p = idx[P[m]]
    _, lab = connected_components(coo_matrix((np.ones(len(p)), (p[:, 0], p[:, 1])), shape=(low.sum(), low.sum())), directed=False)
    ids = np.nonzero(low)[0]
    legs = sorted([ids[lab == k] for k in np.argsort(-np.bincount(lab))[:4]], key=lambda a_: X0[a_, 0].mean())
    e0 = terms(x)
    up = torch.tensor([0., 0., 1.], dtype=f64, device=dev)
    ed = np.linspace(0., .17, 8)
    for li, a_ in enumerate(legs[:2]):
        at = torch.as_tensor(a_, device=dev)
        bins = [a_[(zr[a_] >= ed[i]) & (zr[a_] < ed[i+1])] for i in range(7)]
        lo_ = torch.as_tensor(np.concatenate(bins[:4]), device=dev)
        mm = mass[lo_].sum()
        print('  leg %d: lower part mass %.2e, mean velocity %s; net force on it: elastic %s, m a %s, gravity z %.2e; left %s' % (li, float(mm), np.round((m3[lo_]*v[lo_]).sum(0).cpu().numpy()/float(mm), 3).tolist(), np.round((-g0[lo_]).sum(0).cpu().numpy(), 5).tolist(), np.round((m3[lo_]*a[lo_]).sum(0).cpu().numpy(), 5).tolist(), float(fg[lo_].sum(0)[2]), np.round(res[lo_].sum(0).cpu().numpy(), 5).tolist()), flush=True)
        # along its own motion
        d = torch.zeros_like(x); d[at] = v[at]
        nr = float((d*d).sum().sqrt()); d = d/max(nr, 1e-12)
        eps = 1e-5
        ep, em = terms(x+eps*d), terms(x-eps*d)
        print('    along its own motion, -dE/ds per term: %s' % {k: float('%+.1e' % (-(ep[k]-em[k])/(2*eps))) for k in ep}, flush=True)
        for j in (3, 5):
            joint = x[torch.as_tensor(bins[j], device=dev)].mean(0)
            below = torch.as_tensor(np.concatenate(bins[:j]), device=dev)
            rot = torch.linalg.cross(up.expand(len(below), 3), x[below]-joint)
            pc, tc = x[torch.as_tensor(bins[0], device=dev)].mean(0), x[torch.as_tensor(bins[-1], device=dev)].mean(0)
            if float((rot[:len(bins[0])].mean(0)*(pc-tc)).sum()) < 0:
                rot = -rot
            e_ = 1e-3
            xp, xm = x.clone(), x.clone(); xp[below] += e_*rot; xm[below] -= e_*rot
            ep, em = terms(xp), terms(xm)
            print('    joint at slice %d in the ground plane, torque, + straightens: %s; viscous torque the left force gives %+.1e; angular speed there %+.2f' % (j, {k: float('%+.1e' % (-(ep[k]-em[k])/(2*e_))) for k in ep}, float((res[below]*rot).sum()), float((m3[below]*v[below]*rot).sum()/(m3[below]*rot*rot).sum())), flush=True)


def slices(W, ST, mass, V0, X0, P):
    """slice by slice along a leg at a real state: the volume state, the fit, whether each cell's R follows the slice's own turn, the energy in the tensor and in the remainder"""
    n = W.n
    x = ST.tx.clone(); W.update_ul(x); W.grad(x); W.reread(); W.grad(x)
    Y, T = W.ty, W.tr
    zr = X0[:, 2]-X0[:, 2].min(); low = zr < .18
    idx = -np.ones(n, int); idx[low] = np.arange(low.sum())
    m = low[P[:, 0]] & low[P[:, 1]]; p = idx[P[m]]
    _, lab = connected_components(coo_matrix((np.ones(len(p)), (p[:, 0], p[:, 1])), shape=(low.sum(), low.sum())), directed=False)
    ids = np.nonzero(low)[0]
    legs = sorted([ids[lab == k] for k in np.argsort(-np.bincount(lab))[:4]], key=lambda a_: X0[a_, 0].mean())
    ed = np.linspace(0., .17, 8)
    Xm = Y['Ff']-Y['vm'][:, None, None]*Y['Rz']
    lead = W._gid == torch.arange(n, device=dev)
    e = .5*((Xm@Y['Mtt'])*Xm).sum((1, 2))*lead
    ms = W._m_s; lo, hi = W.tlo, W.thi; g = W._gid
    d = x[hi]-x[lo]
    wi, wj, qi, qj = T['wi'][:ms], T['wj'][:ms], T['qi'][:ms], T['qj'][:ms]
    ri = d-torch.einsum('fab,fb->fa', Y['Ff'][g[lo]], qi); rj = d-torch.einsum('fab,fb->fa', Y['Ff'][g[hi]], qj)
    er = torch.zeros(n, dtype=f64, device=dev).index_add(0, lo, .5*wi*(ri*ri).sum(1)).index_add(0, hi, .5*wj*(rj*rj).sum(1))
    rel = torch.zeros(n, dtype=f64, device=dev).index_add(0, lo, (ri*ri).sum(1)).index_add(0, hi, (rj*rj).sum(1))
    den = torch.zeros(n, dtype=f64, device=dev).index_add(0, lo, (qi*qi).sum(1)).index_add(0, hi, (qj*qj).sum(1))
    Xr = torch.as_tensor(X0, device=dev, dtype=f64)
    print('whole body: tensor %.3e, remainder %.3e; e^L %.3f..%.3f; lowest z %.3f, KE %.3e' % (float(e.sum()), float(er.sum()), float(torch.exp(W.tul['Ln']).min()), float(torch.exp(W.tul['Ln']).max()), float(x[:, 2].min()), float(.5*(mass[:, None]*ST.tv**2).sum())), flush=True)
    sv = torch.linalg.svdvals(Y['Ff'].cpu()).to(dev); dt = torch.linalg.det(Y['Ff'])
    for li, a_ in enumerate(legs[:3]):
        bins = [a_[(zr[a_] >= ed[i]) & (zr[a_] < ed[i+1])] for i in range(7)]
        Rgs = {}
        print('leg %d: end to end %.4f' % (li, float((x[torch.as_tensor(bins[-1], device=dev)].mean(0)-x[torch.as_tensor(bins[0], device=dev)].mean(0)).norm())), flush=True)
        for i, b in enumerate(bins):
            bt = torch.as_tensor(b, device=dev)
            c0, c1 = Xr[bt].mean(0), x[bt].mean(0)
            U, _, Vt = torch.linalg.svd(((x[bt]-c1).T@(Xr[bt]-c0)).cpu())
            S = torch.diag(torch.tensor([1., 1., float(torch.linalg.det(U@Vt))], dtype=f64))
            Rg = (U@S@Vt).to(dev)
            Rc = Y['Rz'][g[bt]]
            cs = (((Rc.transpose(1, 2)@Rg).diagonal(dim1=1, dim2=2).sum(1)-1)/2).clamp(-1., 1.)
            ang = torch.rad2deg(torch.arccos(cs))
            seg = float((x[torch.as_tensor(bins[i+1], device=dev)].mean(0)-c1).norm()/(Xr[torch.as_tensor(bins[i+1], device=dev)].mean(0)-c0).norm()) if i < 6 else 0.
            Rgs[i] = Rg
            print('   slice %d: length ratio %.2f; slice turned %.0f deg from rest; cell R off the slice\'s turn: median %.0f, 90 %% %.0f deg; e^L median %.2f (%.2f..%.2f); det F median %.2f, min %.2f; sv median %.2f %.2f; tensor %.2e, remainder %.2e; |r| / |q| rms %.2f' % (i, seg, float(torch.rad2deg(torch.arccos(((Rg.diagonal().sum()-1)/2).clamp(-1., 1.)))), float(ang.median()), float(torch.quantile(ang, .9)), float(torch.exp(W.tul['Ln'][bt]).median()), float(torch.exp(W.tul['Ln'][bt]).min()), float(torch.exp(W.tul['Ln'][bt]).max()), float(dt[bt].median()), float(dt[bt].min()), float(sv[bt, 0].median()), float(sv[bt, 2].median()), float(e[bt].sum()), float(er[bt].sum()), float((rel[bt].sum()/den[bt].sum()).sqrt())), flush=True)
        # the bend: everything below it turned back toward the part above, a smooth blend across it: per term the torque against the turn (+ unbends), real state
        Rrel = Rgs[1]@Rgs[5].T
        Ru = Rrel.T
        ang = float(torch.arccos(((Ru.diagonal().sum()-1)/2).clamp(-1., 1.)))
        axis = torch.stack([Ru[2, 1]-Ru[1, 2], Ru[0, 2]-Ru[2, 0], Ru[1, 0]-Ru[0, 1]]); axis = axis/axis.norm().clamp(min=1e-12)
        joint = x[torch.as_tensor(bins[3], device=dev)].mean(0)
        at = torch.as_tensor(a_, device=dev)
        zc = torch.as_tensor(zr[a_], device=dev, dtype=f64)
        z1, z2 = float(zr[bins[2]].mean()), float(zr[bins[4]].mean())
        wgt = ((z2-zc)/(z2-z1)).clamp(0., 1.)
        rot = wgt[:, None]*torch.linalg.cross(axis.expand(len(a_), 3), x[at]-joint)
        tv = lambda y_: (W.grad(y_), (W.tsum.clone().sum(1) if W.tsum.dim() > 1 else W.tsum.clone()))[1]
        e_ = 1e-3
        e00 = tv(x); xp, xm = x.clone(), x.clone(); xp[at] += e_*rot; xm[at] -= e_*rot
        ep, em = tv(xp), tv(xm); W.grad(x)
        names = {0: 'tensor', 1: 'volume', 2: 'var', 4: 'remainder', 5: 'outer'}
        Q = {names[k]: -(float(ep[k])-float(em[k]))/(2*e_) for k in names}; St = {names[k]: (float(ep[k])+float(em[k])-2*float(e00[k]))/e_**2 for k in names}
        I_ = float((mass[at][:, None]*rot*rot).sum()); om = float((mass[at][:, None]*ST.tv[at]*rot).sum())/I_
        print('   the bend %.0f deg about %s: torque, + unbends: total %+.2e, %s; stiffness total %.2e, %s; inertia %.2e; so angular acceleration %+.1f rad/s2, omega^2 %.0f; angular speed now %+.2f rad/s, viscous time beta omega^2 = %.2f /s' % (np.degrees(ang), np.round(axis.cpu().numpy(), 2).tolist(), sum(Q.values()), {k: float('%+.1e' % v_) for k, v_ in Q.items()}, sum(St.values()), {k: float('%.1e' % v_) for k, v_ in St.items()}, I_, sum(Q.values())/I_, sum(St.values())/I_, om, .05*sum(St.values())/I_), flush=True)


def equil(W, ST, mass, V0, X0, P):
    """at a real state, a leg with the rest of the body held: its force, its true tangent (R re-read at each product), the Newton step and whether the energy falls along it, the lowest modes. Is the leg at an equilibrium of the model?"""
    n = W.n; h_ = float(np.median(V0)**(1/3))
    x = ST.tx.clone(); W.update_ul(x); W.grad(x); W.reread(); W.grad(x)
    zr = X0[:, 2]-X0[:, 2].min(); low = zr < .18
    idx = -np.ones(n, int); idx[low] = np.arange(low.sum())
    m = low[P[:, 0]] & low[P[:, 1]]; p = idx[P[m]]
    _, lab = connected_components(coo_matrix((np.ones(len(p)), (p[:, 0], p[:, 1])), shape=(low.sum(), low.sum())), directed=False)
    ids = np.nonzero(low)[0]
    legs = sorted([ids[lab == k] for k in np.argsort(-np.bincount(lab))[:4]], key=lambda a_: X0[a_, 0].mean())
    ed = np.linspace(0., .17, 8)

    def force(y, at):
        W.grad(y); W.reread_light(); _, g_ = W.grad(y)
        return g_[at].flatten().clone()

    def energy(y):
        W.grad(y); W.reread_light(); W.grad(y)
        t = W.tsum.clone(); t = t.sum(1) if t.dim() > 1 else t
        return float(t[0]), float(t[1]), float(t[2]), float(t[4]), float(t[5]), float(t[6])

    def e2e(y, bins):
        return float((y[torch.as_tensor(bins[-1], device=dev)].mean(0)-y[torch.as_tensor(bins[0], device=dev)].mean(0)).norm())
    for li, a0 in enumerate(legs[:3]):
        a_ = a0[zr[a0] < .15]
        bins = [a0[(zr[a0] >= ed[i]) & (zr[a0] < ed[i+1])] for i in range(7)]
        at = torch.as_tensor(a_, device=dev); N = len(a_)
        Mv = np.repeat(V0[a_], 3)
        f0 = force(x, at)
        K = torch.zeros(3*N, 3*N, dtype=f64)
        for j in range(3*N):
            y = x.clone(); y[at[j//3], j % 3] += 1e-7
            K[:, j] = ((force(y, at)-f0)/1e-7).cpu()
        K = K.numpy(); g_ = f0.cpu().numpy()
        w = np.linalg.eigvals(K/Mv[:, None]); wr = np.sort(w.real)
        acc = (g_/Mv).reshape(N, 3); weight = 9.8
        print('leg %d: %d cells; asymmetry of K %.2f; lowest real omega^2 %s; negative %d; elastic acceleration on its cells: median %.1f, 90 %% %.0f, mean vector %s; speed of the leg against its root %.3f' % (li, N, float(np.linalg.norm(K-K.T)/np.linalg.norm(K)), np.round(wr[:6], 0).tolist(), int((wr < -1.).sum()), float(np.median(np.linalg.norm(acc, axis=1))), float(np.quantile(np.linalg.norm(acc, axis=1), .9)), np.round((-(g_.reshape(N, 3))).sum(0)/V0[a_].sum(), 2).tolist(), float((ST.tv[at]-ST.tv[torch.as_tensor(bins[-1], device=dev)].mean(0)).norm(dim=1).median())), flush=True)
        e0 = energy(x)
        for c in (0., 30., 300., 3000.):
            A = K+np.diag(c*Mv)
            try:
                dz = -np.linalg.solve(A, g_)
            except np.linalg.LinAlgError:
                dz = -np.linalg.lstsq(A, g_, rcond=1e-8)[0]
            d = torch.as_tensor(dz.reshape(N, 3), device=dev, dtype=f64)
            nr = d.norm(dim=1)
            line = []
            for s in (.1, .25, .5, 1.):
                y = x.clone(); y[at] = x[at]+s*d
                e = energy(y)
                line.append('%.2f: dE %+.1e, end to end %.4f' % (s, sum(e)-sum(e0), e2e(y, bins)))
            print('    (K + %5.0f M) step: |dx| / h median %.2f, max %.1f; along it %s' % (c, float(nr.median())/h_, float(nr.max())/h_, '; '.join(line)), flush=True)
        print('    end to end now %.4f; energies tensor, volume, var, remainder, outer, ground %s' % (e2e(x, bins), ['%.2e' % v for v in e0]), flush=True)


def lock(W, ST, mass, V0, X0, P, tag):
    """the leg's lowest modes at a real state, viscosity out: which energy term carries each, where it lives; the same with the weightless springs' walls as free surfaces"""
    n = W.n
    zr = X0[:, 2]-X0[:, 2].min(); low = zr < .18
    idx = -np.ones(n, int); idx[low] = np.arange(low.sum())
    m = low[P[:, 0]] & low[P[:, 1]]; p = idx[P[m]]
    _, lab = connected_components(coo_matrix((np.ones(len(p)), (p[:, 0], p[:, 1])), shape=(low.sum(), low.sum())), directed=False)
    ids = np.nonzero(low)[0]
    legs = sorted([ids[lab == k] for k in np.argsort(-np.bincount(lab))[:4]], key=lambda a_: X0[a_, 0].mean())
    a0 = legs[0]; a_ = a0[zr[a0] < .15]
    at = torch.as_tensor(a_, device=dev); N = len(a_)
    Mv = np.repeat(V0[a_], 3)
    x = ST.tx.clone()
    names = {0: 'tensor', 1: 'volume', 2: 'var', 4: 'remainder', 5: 'outer', 6: 'ground'}

    def force(y):
        W.grad(y); W.reread_light(); _, g_ = W.grad(y)
        return g_[at].flatten().clone()

    def energy(y):
        W.grad(y); W.reread_light(); W.grad(y)
        t = W.tsum.clone(); t = t.sum(1) if t.dim() > 1 else t
        return np.array([float(t[k]) for k in names])

    for trim in (False, True):
        W.update_ul(x)
        if trim:
            W.elastic_births = False
            keep = (W._wi+W._wj > 0)
            W.tlo, W.thi, W.tkey, W._wi, W._wj, W._qkey, W._qi, W._qj, W.tk0, W.tdl = W.tlo[keep], W.thi[keep], W.tkey[keep], W._wi[keep], W._wj[keep], W._qkey[keep], W._qi[keep], W._qj[keep], W.tk0[keep], W.tdl[keep]
            W.update_ul(x)
        W.grad(x); W.reread(); W.grad(x)
        lo_, hi_ = W.tlo, W.thi
        inleg = torch.zeros(n, dtype=torch.bool, device=dev); inleg[at] = True
        wt = W._wi+W._wj
        rest = torch.as_tensor(X0, device=dev, dtype=f64)
        i_w, j_w = W._w[0], W._w[1]
        touch = inleg[i_w] | inleg[j_w]
        rd = (rest[i_w[touch]]-rest[j_w[touch]]).norm(dim=1)/float(np.median(V0)**(1/3))
        f0 = force(x)
        K = torch.zeros(3*N, 3*N, dtype=f64)
        for j in range(3*N):
            y = x.clone(); y[at[j//3], j % 3] += 1e-7
            K[:, j] = ((force(y)-f0)/1e-7).cpu()
        K = K.numpy()
        w, Q = np.linalg.eig(K/Mv[:, None]); o = np.argsort(w.real); w, Q = w.real[o], Q.real[:, o]
        print('%s %s: lowest real omega^2 %s; connected walls touching the leg %d, of them more than 3 h apart at rest %d; springs on the leg with weight %d, weightless %d' % (tag, 'no weightless' if trim else 'as run       ', np.round(w[:6], 0).tolist(), int(touch.sum()), int((rd > 3).sum()), int((wt[inleg[lo_] | inleg[hi_]] > 0).sum()), int((wt[inleg[lo_] | inleg[hi_]] == 0).sum())), flush=True)
        if trim:
            continue
        e0 = energy(x)
        for i in range(3):
            u = torch.as_tensor(Q[:, i].reshape(N, 3), device=dev, dtype=f64)
            u = u/float(u.abs().max())
            eps = 1e-4*float(np.median(V0)**(1/3))
            xp, xm = x.clone(), x.clone(); xp[at] += eps*u; xm[at] -= eps*u
            q_ = (energy(xp)+energy(xm)-2*e0)/eps**2
            Mu = float((torch.as_tensor(V0[a_], device=dev)[:, None]*u*u).sum())
            sh = (torch.as_tensor(V0[a_], device=dev)[:, None]*u*u).sum(1); sh = sh/sh.sum()
            s_ = zr[a_]
            print('    mode %d, omega^2 %.0f: by term %s; lives on %.0f cells, mean rest height %.3f (leg is 0 .. .15)' % (i, w[i], {names[k]: round(float(q_[j_])/Mu, 0) for j_, k in enumerate(names)}, 1/float((sh*sh).sum()), float((sh.cpu().numpy()*s_).sum())), flush=True)


def unstable(W, ST, mass, V0, X0, P):
    """at a real state, each leg with the rest of the body held: the lowest modes on the true tangent, viscosity out; per term, where they live, which way the leg's own lower part goes against the body's mid plane and up"""
    n = W.n
    zr = X0[:, 2]-X0[:, 2].min(); low = zr < .18
    idx = -np.ones(n, int); idx[low] = np.arange(low.sum())
    m = low[P[:, 0]] & low[P[:, 1]]; p = idx[P[m]]
    _, lab = connected_components(coo_matrix((np.ones(len(p)), (p[:, 0], p[:, 1])), shape=(low.sum(), low.sum())), directed=False)
    ids = np.nonzero(low)[0]
    legs = sorted([ids[lab == k] for k in np.argsort(-np.bincount(lab))[:4]], key=lambda a_: X0[a_, 0].mean())
    x = ST.tx.clone(); W.update_ul(x); W.grad(x); W.reread(); W.grad(x)
    names = {0: 'tensor', 1: 'volume', 2: 'var', 4: 'remainder', 5: 'outer', 6: 'ground'}
    com = x.mean(0)
    print('state: body centre %s; e^L %.2f..%.2f; speed median %.2f' % (np.round(com.cpu().numpy(), 3).tolist(), float(torch.exp(W.tul['Ln']).min()), float(torch.exp(W.tul['Ln']).max()), float(ST.tv.norm(dim=1).median())), flush=True)
    for li, a0 in enumerate(legs):
        a_ = a0[zr[a0] < .15]; at = torch.as_tensor(a_, device=dev); N = len(a_)
        Mv = np.repeat(V0[a_], 3)

        def force(y):
            W.grad(y); W.reread_light(); _, g_ = W.grad(y)
            return g_[at].flatten().clone()

        def energy(y):
            W.grad(y); W.reread_light(); W.grad(y)
            t = W.tsum.clone(); t = t.sum(1) if t.dim() > 1 else t
            return np.array([float(t[k]) for k in names])
        f0 = force(x)
        K = torch.zeros(3*N, 3*N, dtype=f64)
        for j in range(3*N):
            y = x.clone(); y[at[j//3], j % 3] += 1e-7
            K[:, j] = ((force(y)-f0)/1e-7).cpu()
        K = K.numpy()
        w, Q = np.linalg.eig(K/Mv[:, None]); o = np.argsort(w.real); w, Q = w.real[o], Q.real[:, o]
        side = np.sign(float(x[at, 1].mean()-com[1]))
        print('leg %d: %d cells, centre %s, on the %s side; lowest real omega^2 %s; negative %d; force on it, net %s' % (li, N, np.round(x[at].mean(0).cpu().numpy(), 3).tolist(), '+y' if side > 0 else '-y', np.round(w[:6], 0).tolist(), int((w < -1.).sum()), np.round((-f0.reshape(N, 3)).sum(0).cpu().numpy(), 5).tolist()), flush=True)
        e0 = energy(x)
        for i in range(min(3, int((w < -1.).sum())+1)):
            u = torch.as_tensor(Q[:, i].reshape(N, 3), device=dev, dtype=f64); u = u/float(u.abs().max())
            if float((u*(-f0.reshape(N, 3))).sum()) < 0:
                u = -u                                                                     # (the way the present force takes it)
            eps = 1e-4*float(np.median(V0)**(1/3))
            xp, xm = x.clone(), x.clone(); xp[at] += eps*u; xm[at] -= eps*u
            q_ = (energy(xp)+energy(xm)-2*e0)/eps**2
            Mu = float((torch.as_tensor(V0[a_], device=dev)[:, None]*u*u).sum())
            sh = (torch.as_tensor(V0[a_], device=dev)[:, None]*u*u).sum(1); sh = sh/sh.sum()
            k_ = torch.argsort(sh, descending=True)[:max(N//8, 3)]
            d_ = (sh[k_, None]*u[k_]).sum(0)/sh[k_].sum()
            print('    mode %d, omega^2 %.0f: by term %s; lives on %.0f cells; its heaviest cells at rest height %.3f; they go (x, y out of the body, z up) %s; work of the present force %+.2e' % (i, w[i], {names[k]: round(float(q_[j_])/Mu, 0) for j_, k in enumerate(names)}, 1/float((sh*sh).sum()), float(zr[a_][k_.cpu().numpy()].mean()), np.round([float(d_[0]), side*float(d_[1]), float(d_[2])], 2).tolist(), float((u*(-f0.reshape(N, 3))).sum())), flush=True)


def check30(W, ST, mass, V0, X0, P, dense, step, dt):
    """the leg's first bow, at a real state: can the paw slide; which term pushes the leg, slice by slice; the cells' couples (R = polar(F), M not round) against the ground's moment; the lowest modes and how much of the motion that follows goes along them"""
    n = W.n; h_ = float(np.median(V0)**(1/3))
    A, bl = ST.a, W.ball
    wp.launch(k_mid_ball, bl['k'], inputs=[A['x'], bl['body'], bl['R'], bl['ng'], ST.c0, ST.g8, ST.cb], outputs=[bl['xn'], bl['kt'], bl['bound']])
    x, v = ST.tx.clone(), ST.tv.clone()
    W.update_ul(x); W.grad(x); W.reread(); _, g0 = W.grad(x); g0 = g0.clone()
    zr = X0[:, 2]-X0[:, 2].min(); low = zr < .18
    idx = -np.ones(n, int); idx[low] = np.arange(low.sum())
    m = low[P[:, 0]] & low[P[:, 1]]; p = idx[P[m]]
    _, lab = connected_components(coo_matrix((np.ones(len(p)), (p[:, 0], p[:, 1])), shape=(low.sum(), low.sum())), directed=False)
    ids = np.nonzero(low)[0]
    legs = sorted([ids[lab == k] for k in np.argsort(-np.bincount(lab))[:4]], key=lambda a_: X0[a_, 0].mean())
    ed = np.linspace(0., .17, 8)
    Y = W.ty; g = W._gid; lead = g == torch.arange(n, device=dev)
    Xm = Y['Ff']-Y['vm'][:, None, None]*Y['Rz']
    Ar = Y['Ff']@Y['Mtt']@Xm.transpose(1, 2)
    tau = -torch.stack([Ar[:, 1, 2]-Ar[:, 2, 1], Ar[:, 2, 0]-Ar[:, 0, 2], Ar[:, 0, 1]-Ar[:, 1, 0]], 1)*lead[:, None]
    fn = torch.as_tensor(W.B['bfn'].numpy(), device=dev, dtype=f64)
    ff = torch.as_tensor(W.B['bff'].numpy(), device=dev, dtype=f64)
    if ff.dim() == 1:
        ff = torch.zeros(n, 3, dtype=f64, device=dev)
    ups = torch.tensor([0., 0., 1.], dtype=f64, device=dev)
    fg_ = fn[:, None]*ups+ff
    com = x.mean(0)
    Mg_all = torch.linalg.cross(x-com, fg_).sum(0)
    print('t %.4f: speed median %.2f; couples of all the cells, sum |%.2e|, the sum of their sizes %.2e; the ground\'s moment on the body |%.2e|; ground force %.2e, tangential %.2e, contact cells %d; contact moment of the couples per cell: median |tau| / (2 mu V e) %.3f' % (step*dt, float(v.norm(dim=1).median()), float(tau.sum(0).norm()), float(tau.norm(dim=1).sum()), float(Mg_all.norm()), float(fn.sum()), float(ff.norm(dim=1).sum()), int((fn > 0).sum()), float((tau.norm(dim=1)[lead]/(((Xm@Y['Mtt'])*Xm).sum((1, 2))[lead].sqrt()*Y['Mtt'].diagonal(dim1=1, dim2=2).sum(1)[lead].sqrt()/3**.5).clamp(min=1e-30)).median())), flush=True)
    names = {0: 'tensor', 1: 'volume', 2: 'var', 4: 'remainder', 5: 'outer', 6: 'ground'}

    def term_e(y):
        W.grad(y)
        t = W.tsum.clone(); t = t.sum(1) if t.dim() > 1 else t
        return np.array([float(t[k]) for k in names])
    k_next = int(np.argmin(np.abs(dense['s']-(step+12))))
    has_next = abs(int(dense['s'][k_next])-(step+12)) == 0
    for li, a0 in enumerate(legs):
        side = float(np.sign(X0[a0, 1].mean()-X0[:, 1].mean()))
        bins = [a0[(zr[a0] >= ed[i]) & (zr[a0] < ed[i+1])] for i in range(7)]
        bt = [torch.as_tensor(b, device=dev) for b in bins]
        paw = bt[0]; knee = bt[3]
        cz = lambda u_: np.round([float(u_[0]), side*float(u_[1]), float(u_[2])], 3).tolist()
        print('  leg %d: paw at %s, knee at %s (x, y out of the mid plane, z); paw velocity %s, knee velocity %s; ground on the paw: normal %.2e, tangential %.2e, contact cells %d; on the whole leg: normal %.2e, %d cells' % (li, cz(x[paw].mean(0)-torch.tensor([0., float(com[1]), 0.], dtype=f64, device=dev)), cz(x[knee].mean(0)-torch.tensor([0., float(com[1]), 0.], dtype=f64, device=dev)), cz(v[paw].mean(0)), cz(v[knee].mean(0)), float(fn[paw].sum()), float(ff[paw].norm(dim=1).sum()), int((fn[paw] > 0).sum()), float(fn[torch.as_tensor(a0, device=dev)].sum()), int((fn[torch.as_tensor(a0, device=dev)] > 0).sum())), flush=True)
        # the paw slides? its horizontal force from the ground's own energy
        e_ = 1e-6
        for ax, nm in ((0, 'x'), (1, 'y')):
            xp, xm_ = x.clone(), x.clone(); xp[paw, ax] += e_; xm_[paw, ax] -= e_
            print('    the ground\'s energy against the paw moved in %s: %+.2e' % (nm, -(term_e(xp)[5]-term_e(xm_)[5])/(2*e_)), flush=True)
        # slice by slice, the force on it per term
        for i, b in enumerate(bt):
            row = {}
            for ax in range(3):
                xp, xm_ = x.clone(), x.clone(); xp[b, ax] += e_; xm_[b, ax] -= e_
                row[ax] = -(term_e(xp)-term_e(xm_))/(2*e_)
            print('    slice %d: force (x, y out, z) %s; %s' % (i, cz(-g0[b].sum(0)*1e3), ' '.join('%s %s' % (nm_, np.round([1e3*row[0][j_], 1e3*side*row[1][j_], 1e3*row[2][j_]], 2).tolist()) for j_, nm_ in enumerate(names.values()))), flush=True)
        # moments about each joint on the part below it
        for j in (2, 3, 4, 5):
            below = torch.as_tensor(np.concatenate(bins[:j]), device=dev)
            jx = x[bt[j]].mean(0)
            Mt = tau[below].sum(0)
            Mgr = torch.linalg.cross(x[below]-jx, fg_[below]).sum(0)
            Mn = torch.linalg.cross(x[below]-jx, -g0[below]).sum(0)
            print('    moment about the joint at slice %d on what is below it: the cells\' couples %s |%.2e|; the ground %s |%.2e|; all the forces %s |%.2e|' % (j, np.round(Mt.cpu().numpy()*1e4, 2).tolist(), float(Mt.norm()), np.round(Mgr.cpu().numpy()*1e4, 2).tolist(), float(Mgr.norm()), np.round(Mn.cpu().numpy()*1e4, 2).tolist(), float(Mn.norm())), flush=True)
        # the modes, with the ground, and the motion to come
        a_ = a0[zr[a0] < .15]; at = torch.as_tensor(a_, device=dev); N = len(a_)
        Mv = np.repeat(V0[a_], 3)

        def force(y):
            W.grad(y); W.reread_light(); _, gg = W.grad(y)
            return gg[at].flatten().clone()
        f0 = force(x)
        K = torch.zeros(3*N, 3*N, dtype=f64)
        for jj in range(3*N):
            y = x.clone(); y[at[jj//3], jj % 3] += 1e-7
            K[:, jj] = ((force(y)-f0)/1e-7).cpu()
        K = K.numpy()
        w, Q = np.linalg.eig(K/Mv[:, None]); o = np.argsort(w.real); w, Q = w.real[o], Q.real[:, o]
        W.grad(x); W.reread(); W.grad(x)
        if has_next:
            xn_ = dense['x'][k_next]
            u = xn_[a_]-x[at].cpu().numpy()
            top = np.isin(a_, np.concatenate(bins[-1:]))
            u = u-(u[top].mean(0) if top.any() else 0.)
            un = u.flatten(); nu = np.sqrt((Mv*un*un).sum())
        print('    modes: lowest real omega^2 %s; negative %d; the motion of the next .01 s, |u| rms %.4f' % (np.round(w[:6], 0).tolist(), int((w < -1.).sum()), (nu/np.sqrt(Mv.sum())) if has_next else -1.), flush=True)
        for i in range(4):
            q_ = Q[:, i]; nq = np.sqrt((Mv*q_*q_).sum()); q_ = q_/nq
            qr = q_.reshape(N, 3)
            pk = np.isin(a_, bins[0])
            sh = (V0[a_][:, None]*qr*qr).sum(1)
            print('      mode %d, omega^2 %.0f: paw %s, knee %s (x, y out, z), the force along it %+.2e; cos with the motion %s' % (i, w[i], np.round([qr[pk, 0].mean(), side*qr[pk, 1].mean(), qr[pk, 2].mean()], 2).tolist(), np.round([qr[np.isin(a_, bins[3]), 0].mean(), side*qr[np.isin(a_, bins[3]), 1].mean(), qr[np.isin(a_, bins[3]), 2].mean()], 2).tolist(), float((-f0.cpu().numpy()*q_).sum()), ('%+.2f' % ((Mv*un*q_).sum()/nu)) if has_next else '-'), flush=True)


def lock_colour(W, ST, mass, V0, X0, P, r0, sigma=1.19):
    """at a real state with weightless springs, leg 0: the lowest modes as run, without the weightless springs between unlike colours (the contacts), without all of them; the colours of those springs against their rest distance"""
    from free_surface.ring_moment import color_code, color_sim
    n = W.n; h_ = float(np.median(V0)**(1/3))
    Qc = color_code(W, r0, sigma)
    zr = X0[:, 2]-X0[:, 2].min(); low = zr < .18
    idx = -np.ones(n, int); idx[low] = np.arange(low.sum())
    m = low[P[:, 0]] & low[P[:, 1]]; p = idx[P[m]]
    _, lab = connected_components(coo_matrix((np.ones(len(p)), (p[:, 0], p[:, 1])), shape=(low.sum(), low.sum())), directed=False)
    ids = np.nonzero(low)[0]
    legs = sorted([ids[lab == k] for k in np.argsort(-np.bincount(lab))[:4]], key=lambda a_: X0[a_, 0].mean())
    a0 = legs[0]; a_ = a0[zr[a0] < .15]; at = torch.as_tensor(a_, device=dev); N = len(a_)
    Mv = np.repeat(V0[a_], 3)
    x = ST.tx.clone()
    save = (W.tlo.clone(), W.thi.clone(), W.tkey.clone(), W._wi.clone(), W._wj.clone(), W._qkey.clone(), W._qi.clone(), W._qj.clone(), W.tk0.clone(), W.tdl.clone())
    W.update_ul(x)
    wt = W._wi+W._wj; zero = wt == 0
    sim = color_sim(Qc, W.tlo, W.thi)
    rest = torch.as_tensor(X0, device=dev, dtype=f64)
    rd = (rest[W.tlo]-rest[W.thi]).norm(dim=1)/h_
    inleg = torch.zeros(n, dtype=torch.bool, device=dev); inleg[at] = True
    zl = zero & (inleg[W.tlo] | inleg[W.thi])
    hb = lambda v_: np.histogram(v_.cpu().numpy(), bins=[0., .1, .3, .5, .7, .9, 1.01])[0].tolist()
    print('weightless springs %d (on the leg %d); colour similarity, counts in 0 .1 .3 .5 .7 .9 1: all %s, on the leg %s; rest distance / h median by similarity under .5: %.1f, over .5: %.1f; springs with weight, similarity median %.2f, 1 %% %.2f' % (int(zero.sum()), int(zl.sum()), hb(sim[zero]), hb(sim[zl]), float(rd[zero & (sim < .5)].median()) if bool((zero & (sim < .5)).any()) else -1., float(rd[zero & (sim >= .5)].median()) if bool((zero & (sim >= .5)).any()) else -1., float(sim[~zero].median()), float(torch.quantile(sim[~zero], .01))), flush=True)

    def force(y):
        W.grad(y); W.reread_light(); _, g_ = W.grad(y)
        return g_[at].flatten().clone()
    for name, drop in (('as run', None), ('unlike colours (< .5) out', zero & (sim < .5)), ('all weightless out', zero)):
        W.tlo, W.thi, W.tkey, W._wi, W._wj, W._qkey, W._qi, W._qj, W.tk0, W.tdl = [t_.clone() for t_ in save]
        W.update_ul(x)
        if drop is not None:
            keep = ~drop
            W.elastic_births = False
            W.tlo, W.thi, W.tkey, W._wi, W._wj, W._qkey, W._qi, W._qj, W.tk0, W.tdl = W.tlo[keep], W.thi[keep], W.tkey[keep], W._wi[keep], W._wj[keep], W._qkey[keep], W._qi[keep], W._qj[keep], W.tk0[keep], W.tdl[keep]
            W._spring_rule_keep = keep
            W.update_ul(x)
        W.elastic_births = True
        W.grad(x); W.reread(); W.grad(x)
        f0 = force(x)
        K = torch.zeros(3*N, 3*N, dtype=f64)
        for j in range(3*N):
            y = x.clone(); y[at[j//3], j % 3] += 1e-7
            K[:, j] = ((force(y)-f0)/1e-7).cpu()
        w = np.sort(np.linalg.eigvals(K.numpy()/Mv[:, None]).real)
        print('  %-26s lowest real omega^2 %s; springs in the list %d' % (name, np.round(w[:6], 0).tolist(), int(W._m_s)), flush=True)


def contact_check(W, ST, V0):
    """the colour contact on the state's gap. At rest (F = I): rho0, and no unlike pair may overlap. At the real state: one step's start (the cut, births by colour, the contact list), the pairs, the overlapping ones;
    then every pair put at lambda .95 at z = x_n: the contact's gradient (with it less without it) against central differences of its energy, term 7, along a random direction"""
    h_ = float(np.median(V0)**(1/3))
    q = lambda v_: np.round(np.quantile(v_.cpu().numpy(), [0., .1, .5, .9, 1.]), 3).tolist()
    I3 = torch.eye(3, dtype=f64, device=dev).expand(W.n, 3, 3)
    Fe_ = W._Fe.clone()
    if W.jump_share:
        W._Fe.copy_(I3)                                                                         # (at rest: F = I for the jump test too)
    a, b, ra, rb, pw, L = W.contact_pairs(W.X, I3, h_)
    if W.jump_share and len(a):
        sb_ = W._blend(W.X)
        print('  debug: F_e is I %s; the site blends as made %s; a sample pair s %s' % (bool((W._Fe[W._gid]-I3).abs().max() < 1e-12), all(bool((u_ == v_).all()) for u_, v_ in zip(sb_[:2], W._site_blend[:2])),
              np.round(W.jump_sim(a[:5], b[:5], W.X[b[:5]]-W.X[a[:5]], torch.full((5,), .5, dtype=f64, device=dev)).cpu().numpy(), 4).tolist()), flush=True)
    W._Fe.copy_(Fe_)
    dl = (ra+rb-L)/(ra+rb)
    if len(a) == 0:
        print('rho0 / h quantiles 0 .1 .5 .9 1: %s; at rest no pair pushes (none within reach with 1 - s over 1e-3)' % q(W._rho0/h_), flush=True)
    else:
        print('rho0 / h quantiles 0 .1 .5 .9 1: %s; at rest, unlike pairs within reach + h: %d, overlapping %d, the deepest %.3f of rho_i + rho_k, the nearest gap %.3f h' % (q(W._rho0/h_), len(a), int((dl > 0).sum()), float(dl.max()), float((L-ra-rb).min())/h_), flush=True)
    if bool((dl > 0).any()):
        ov = dl > 0
        Er = float((.5*W.Mp*.5*(W.V0[a[ov]]+W.V0[b[ov]])*pw[ov]*torch.log(L[ov]/(ra+rb)[ov])**2).sum())
        Fr = float((2*.5*W.Mp*.5*(W.V0[a[ov]]+W.V0[b[ov]])*pw[ov]*(-torch.log(L[ov]/(ra+rb)[ov]))/L[ov]).sum())/(9.8*float(np.median(V0)))
        print('  at rest the overlapping pairs\' push weight 1 - s quantiles 0 .1 .5 .9 1: %s; their energy %.3e, push %.2f median cells\' weights in all' % (q(pw[ov]), Er, Fr), flush=True)
    x = ST.tx.clone()
    W.update_ul(x); W.update_self(x)
    st, us = W.self_stats, W.ul_stats
    P = W._pairs; m = len(P)
    t = W.tsc
    Rs = t['Rs'][:m].clone()
    L = (x[P[:, 1]]-x[P[:, 0]]).norm(dim=1)
    print('one start: springs born %d; contact pairs %d, overlapping %d, the deepest %.3f of rho_i + rho_k; rho_i + rho_k / h %s; 1 - similarity %s' % (us['born'], m, st['active'], st['d_max'], q(Rs/h_), q(W._pw)), flush=True)
    if m == 0:
        return
    W.grad(x); W.reread()
    Kc = t['Kc'][:m].clone()
    t['Kc'][:m] = 0.
    _, g_off = W.grad(x); g_off = g_off.clone()
    t['Kc'][:m] = Kc; t['Rs'][:m] = L/.95                                                   # (every pair at lambda .95)
    e0, g_on = W.grad(x); g_on = g_on.clone(); E7 = float(W.tsum[7])
    E7c = float((float(W.tball[1])*Kc*math.log(.95)**2).sum())
    gc = g_on-g_off
    torch.manual_seed(0)
    d = torch.randn_like(x); d = d/d.norm(dim=1).max()
    act = t['fn'][:m] > 0
    print('  every pair at lambda .95: %d pushing, energy %.4e (closed form %.4e); push per pair median %.3e = %.2f of a median cell\'s weight' % (int(act.sum()), E7, E7c, float(t['fn'][:m][act].median()), float(t['fn'][:m][act].median())/(9.8*float(np.median(V0)))), flush=True)
    for e in (1e-4*h_, 1e-5*h_, 1e-6*h_):
        W.grad(x+e*d); Ep = float(W.tsum[7])
        W.grad(x-e*d); Em = float(W.tsum[7])
        fd = (Ep-Em)/(2*e); an = float((gc*d).sum())
        print('  step %.0e h: central difference %.6e, gradient %.6e, rel. error %.2e' % (e/h_, fd, an, abs(fd-an)/abs(an)), flush=True)
    t['Rs'][:m] = Rs


def jump_check(W, ST, V0, tag, Q, rest_keys, skin=None):
    """the jump test (11-02) at a real state: a wall (i, j) of the cut at x lies in one piece of material iff the inverse motion is continuous across it.
    J = X_j - X_i - int_{x_i}^{x_j} F^-1 dx, the integral F_i^-1 t d + F_j^-1 (1 - t) d (t the wall's place on the line, d = x_j - x_i): 0 in one piece, the rest jump
    across an interface. Its size by the static colour: s = BC of the colour of the rest cell holding the predicted point X_i + int and j's own (and from j), averaged.
    Per wall class: rest walls, new walls of like colour (>= .5), of unlike colour (< .5); F the state's F_e, and the fit at x"""
    from free_surface.ring_moment import color_sim
    n = W.n; h_ = float(np.median(V0))**(1/3)
    X = W.X; x = ST.tx.clone(); r0 = W._rr0
    Fe = W._Fe[W._gid].clone()
    W.update_ul(x)
    Ffit = W.ty['Fn'][W._gid].clone()
    G = W._ulG; kp = G.wall_A.detach() > 0
    i, j = G.i[kp].long(), G.j[kp].long(); c = G.wall_c.detach()[kp]
    d = x[j]-x[i]
    t = ((c-x[i])*d).sum(1)/(d*d).sum(1)
    t_out = int(((t < 0) | (t > 1)).sum()); t = t.clamp(0., 1.)
    a_, b_ = torch.minimum(i, j), torch.maximum(i, j)
    rest = torch.isin(a_*n+b_, rest_keys)
    sc = color_sim(Q, i, j)
    cls = {'rest walls': rest, 'new, like colour': ~rest & (sc >= .5), 'new, unlike colour': ~rest & (sc < .5)}
    w2 = r0*r0

    def locate(p):
        out = torch.empty(len(p), dtype=torch.long, device=p.device)
        for c0 in range(0, len(p), 4096):
            pw = torch.cdist(p[c0:c0+4096].float(), X.float())**2-w2.float()[None]
            out[c0:c0+4096] = pw.argmin(1)
        return out
    q = lambda v_: np.round(np.quantile(v_.cpu().numpy(), [.05, .25, .5, .75, .95]), 3).tolist() if len(v_) else '-'
    print('== %s: walls %d (t outside [0, 1]: %d); rest %d, new like %d, new unlike %d' % (tag, len(i), t_out, int(rest.sum()), int(cls['new, like colour'].sum()), int(cls['new, unlike colour'].sum())), flush=True)
    for name, F in (('F_e of the state', Fe), ('the fit at x', Ffit)):
        Fi = W._inv(F)
        I_ = torch.einsum('fab,fb->fa', Fi[i], t[:, None]*d)+torch.einsum('fab,fb->fa', Fi[j], (1-t)[:, None]*d)
        J = X[j]-X[i]-I_
        cj, ci = locate(X[i]+I_), locate(X[j]-I_)
        s = .5*(color_sim(Q, cj, j)+color_sim(Q, ci, i))
        hit = .5*((cj == j).double()+(ci == i).double())
        print('  %s: |F^-1| largest singular value quantiles .5 .99 max %s' % (name, np.round(np.quantile(torch.linalg.matrix_norm(Fi, ord=2).cpu().numpy(), [.5, .99, 1.]), 2).tolist()), flush=True)
        for k_, m_ in cls.items():
            if not bool(m_.any()):
                continue
            print('    %-19s |J| / h q.05 .25 .5 .75 .95 %s; s %s; s > .5: %.3f; predicted cell = the partner: %.3f' % (k_, q(J[m_].norm(dim=1)/h_), q(s[m_]), float((s[m_] > .5).double().mean()), float(hit[m_].mean())), flush=True)
        nw = ~rest
        if bool(nw.any()):
            H = np.histogram2d(sc[nw].cpu().numpy(), s[nw].cpu().numpy(), bins=[[0., .1, .3, .5, .7, .9, 1.01], [0., .1, .3, .5, .7, .9, 1.01]])[0].astype(int)
            print('    new walls, rows colour similarity 0 .1 .3 .5 .7 .9 1, columns jump similarity, the same bins:\n'+'\n'.join('      '+' '.join('%6d' % v_ for v_ in row) for row in H), flush=True)
        if skin is None or name != 'the fit at x':
            continue
        # the walls where the two disagree, against those where both read contact or both material: rest and current distance, the rest segment inside the body, skin, legs
        Xr = X; zr = X[:, 2]-X[:, 2].min(); leg = zr < .18
        sg = torch.linspace(.1, .9, 9, device=x.device, dtype=x.dtype)

        def inside(a, b):
            pts = (Xr[a][:, None, :]+sg[None, :, None]*(Xr[b]-Xr[a])[:, None, :]).reshape(-1, 3)
            k = locate(pts)
            ok = (pts-Xr[k]).norm(dim=1) <= r0[k]
            return ok.reshape(len(a), -1).all(1)
        sets = {'colour < .5, jump > .5 (disagree)': nw & (sc < .5) & (s > .5), 'colour < .5, jump < .5 (both contact)': nw & (sc < .5) & (s <= .5),
                'colour >= .5, jump < .5 (disagree)': nw & (sc >= .5) & (s <= .5), 'colour >= .5, jump > .5 (both material)': nw & (sc >= .5) & (s > .5)}
        for k_, m_ in sets.items():
            if not bool(m_.any()):
                print('    %-40s none' % k_, flush=True)
                continue
            a, b = i[m_], j[m_]
            rd = (Xr[b]-Xr[a]).norm(dim=1)/h_; cd = d[m_].norm(dim=1)/h_
            print('    %-40s %5d: rest distance / h %s; now %s; now / rest %s; rest segment inside the body %.2f; both skin %.2f, one %.2f; both on legs %.2f, one %.2f' % (
                k_, int(m_.sum()), q(rd), q(cd), q(cd/rd), float(inside(a, b).double().mean()), float((skin[a] & skin[b]).double().mean()), float((skin[a] ^ skin[b]).double().mean()),
                float((leg[a] & leg[b]).double().mean()), float((leg[a] ^ leg[b]).double().mean())), flush=True)


def ground_check(W, ST, V0):
    """the ground in the cut (merged_ul.set_ground_cut) at a real state: one step's start (mirrors, faces, which count, the radii), the volumes against V0 e^L, none
    of a cell under the plane by construction; the whole energy's gradient against central differences along a random direction, the ground faces in it"""
    x = ST.tx.clone()
    h_ = float(np.median(V0)**(1/3))
    c0 = W._recut.c0; ng = W._recut.ng
    ht = x@ng-c0
    print('sites under the plane %d (lowest %.3f h above it); balls reaching it %d' % (int((ht < 0).sum()), float(ht.min())/h_, int((ht < W._ulr).sum())), flush=True)
    W.update_ul(x)
    gs, us = W.ground_stats, W.ul_stats
    G = W._ulG
    Vt = W.V0*torch.exp(W.tul['Ln'])
    err = ((G.V.detach()-Vt)/Vt).abs()
    print('one start: mirrors %d; ground faces %d, area %.4f m^2; walls to another cell\'s mirror, largest area %.2e; radii iterations %d; volume error max %.2e, total V %.6f against V0 e^L %.6f; cells against their images %d, pressing %d, lambda min %.3f' % (
        G.n_mirror, gs['faces'], gs['area'], gs['other_A_max'], us.get('radii_it', -1), float(err.max()), float(G.V.detach().sum()), float(Vt.sum()), gs['near'], gs['pushing'], gs['lam_min']), flush=True)
    W.grad(x); W.reread()
    e0, g0 = W.grad(x); g0 = g0.clone()
    t_ = W.tgc; on_ = t_['Kc'] > 0
    i_ = t_['i'][on_].long(); hh_ = x[i_]@ng-c0; lam_ = hh_/t_['Rs'][on_]
    E6_ = float(W.tball[1])*float((t_['Kc'][on_]*torch.log(lam_.clamp(max=1.))**2).sum())      # (the term's weight wt, as the kernel; x is the step's start, so the end point is x)
    print('  the ground term at x: %.6e, by hand from the list %.6e; push / weight %.4f; rho / h %.3f..%.3f' % (
        float(W.tsum[6]), E6_, float(t_['fn'].sum())/float(9.8*W.V0.sum()), float(t_['Rs'][on_].min())/h_ if bool(on_.any()) else 0., float(t_['Rs'][on_].max())/h_ if bool(on_.any()) else 0.), flush=True)
    torch.manual_seed(1)
    d = torch.randn_like(x); d = d/d.norm(dim=1).max()
    near = ht < 2*h_
    for name, m_ in (('every site', torch.ones_like(near)), ('the sites near the ground', near)):
        dd = d*m_[:, None]
        for e in (1e-5*h_, 1e-6*h_):
            Ep, _ = W.grad(x+e*dd); Em, _ = W.grad(x-e*dd)
            fd = float(Ep-Em)/(2*e); an = float((g0*dd).sum())
            print('  %s, step %.0e h: central difference %.8e, gradient %.8e, rel. error %.2e' % (name, e/h_, fd, an, abs(fd-an)/max(abs(an), 1e-300)), flush=True)
