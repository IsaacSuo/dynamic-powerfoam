"""At a real saved state of a run (nothing rebuilt): is the force at x what the frames' motion asks, and on a leg, per term, along its own motion and along straightening in the ground's plane."""
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
