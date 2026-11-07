"""Fixed-time-step feasibility directions for k1's frozen linear gradient.

No material/energy changes. Analytic determinant normals use the existing exact
G adjoint, in private buffers. Optional nonlinear normal corrections account for
the curvature of the determinant constraints before an energy trial is accepted.
"""
import math
import time

import torch
import warp as wp

from free_surface.closure_warp import PS_GQ2, PS_PQ, PS_PZ, k_Aul, k_Gul, k_Gul_adj
from free_surface.ul_plastic import (k_Aul_adj2, k_Ge, k_Ge_adj, k_gg_cell,
                                    k_gg_cell_adj, k_gg_wall, k_gg_wall_adj)


def cofactor(a):
    """Polynomial derivative of det, including at singular matrices."""
    return torch.stack((torch.linalg.cross(a[..., :, 1], a[..., :, 2], dim=-1),
                        torch.linalg.cross(a[..., :, 2], a[..., :, 0], dim=-1),
                        torch.linalg.cross(a[..., :, 0], a[..., :, 1], dim=-1)), -1)


class GradientRows:
    """G_c(u) = sum_k outer(u_k, row[c,k]); cache valid for ONE frozen cut."""
    def __init__(self, W):
        self.W = W
        self.rows = {}
        self.builds = 0
        self.actions = 0
        self.seconds = 0.
        self.b = {k: wp.zeros(W.n, dtype=wp.mat33d, device=W.dev)
                  for k in ('G0', 'Ga', 'Ge', 'Sg', 'G', 'seed', 'Gs', 'Gbe', 'Gb0')}
        self.x = wp.zeros(W.n, dtype=wp.vec3d, device=W.dev)
        self.zero = wp.zeros_like(self.x)
        self.g = wp.zeros_like(self.x)
        self.tx = wp.to_torch(self.x, requires_grad=False)
        self.tg = wp.to_torch(self.g, requires_grad=False)
        self.tseed = wp.to_torch(self.b['seed'], requires_grad=False)
        self.tG = wp.to_torch(self.b['G'], requires_grad=False)

    def action(self, u):
        """Linear G action, no energy/gradient evaluation or solver-buffer writes."""
        W, U, b = self.W, self.W.ul, self.b
        n, kw = W.n, U['Kw']
        self.tx.copy_(u)
        for k in ('G0', 'Ga', 'Sg'):
            b[k].zero_()
        wp.launch(k_Gul, 2*kw, inputs=[self.x, self.zero, U['S'], U['O'], U['R']], outputs=[b['G0']])
        wp.launch(k_Aul, kw, inputs=[b['G0'], U['As'], U['Ao'], U['Aw']], outputs=[b['Ga']])
        wp.launch(k_Ge, n, inputs=[b['G0'], b['Ga'], U['used']], outputs=[b['Ge']])
        wall = [U[k] for k in ('wi', 'wj', 'wa', 'wn', 'wt', 'wtau')]
        wp.launch(k_gg_wall, kw, inputs=[self.x, self.zero]+wall+[b['Ge']], outputs=[b['Sg']])
        wp.launch(k_gg_cell, n, inputs=[self.x, self.zero, b['Sg'], b['Ge'], U['a'], U['Mc'], U['Vi']], outputs=[b['G']])
        self.actions += 1
        return self.tG.clone()

    def row(self, c):
        if c in self.rows:
            return self.rows[c]
        start = time.perf_counter()
        W, U, b = self.W, self.W.ul, self.b
        n, kw = W.n, U['Kw']
        wall = [U[k] for k in ('wi', 'wj', 'wa', 'wn', 'wt', 'wtau')]
        r = torch.empty((n, 3), dtype=torch.float64, device=W.dev)
        # All spatial components have the same scalar stencil. Three transpose
        # actions recover it, including aggregation's extended dependencies.
        for j in range(3):
            for k in ('seed', 'Gbe', 'Gb0'):
                b[k].zero_()
            self.g.zero_()
            self.tseed[c, 0, j] = 1.
            wp.launch(k_gg_cell_adj, n, inputs=[b['seed'], U['a'], U['Mc'], U['Vi']], outputs=[self.g, b['Gs'], b['Gbe']])
            wp.launch(k_gg_wall_adj, kw, inputs=wall+[b['Gs']], outputs=[self.g, b['Gbe']])
            wp.launch(k_Ge_adj, n, inputs=[b['Gbe'], U['used']], outputs=[b['Gb0']])
            wp.launch(k_Aul_adj2, kw, inputs=[b['Gbe'], U['As'], U['Ao'], U['Aw']], outputs=[b['Gb0']])
            wp.launch(k_Gul_adj, 2*kw, inputs=[b['Gb0'], U['S'], U['O'], U['R']], outputs=[self.g])
            r[:, j].copy_(self.tg[:, 0])
        ids = torch.nonzero(r.abs().amax(1) > 0).flatten()
        self.rows[c] = (ids, r[ids].clone())
        self.builds += 1
        self.seconds += time.perf_counter()-start
        return self.rows[c]

    def normal(self, c, A, theta):
        ids, r = self.row(c)
        n = torch.zeros((self.W.n, 3), dtype=torch.float64, device=self.W.dev)
        n[ids] = (r @ cofactor(A).T)/theta
        return n

    def local_action(self, cells, u, theta):
        return torch.stack([u[self.row(c)[0]].T @ self.row(c)[1] / theta for c in cells])


def metric_system(normals, pinv):
    """Scaled Schur system for simultaneous, minimum-P-norm corrections."""
    N = torch.stack(normals)
    scale = N.flatten(1).norm(dim=1).clamp_min(1e-300)
    N = N/scale[:, None, None]
    Q = torch.einsum('kab,ikb->ika', pinv, N)
    S = torch.einsum('ika,jka->ij', N, Q)
    # Active normals can be dependent; solve their small system by a symmetric
    # pseudoinverse. This is not an inverse of the full energy Hessian.
    Sinv = torch.linalg.pinv(.5*(S+S.T), hermitian=True, rtol=1e-12)
    return N, Q, Sinv, scale


def correction(system, rhs):
    N, Q, Sinv, scale = system
    lam = Sinv @ (rhs/scale)
    return torch.einsum('i,ika->ka', lam, Q)


def check(ST, log=print):
    """Independent checks on the replay's actual cut, including shared buffers."""
    W = ST.W
    rows = GradientRows(W)
    z = ST.tz.clone()
    saved = [v.clone() for v in (W.tx, W.tG, W.tgx)]
    G = rows.action(z-W.tul['xn'])
    rel = float((G-W.tG).norm()/W.tG.norm().clamp_min(1e-30))
    assert rel < 1e-11, ('forward G', rel)
    gen = torch.Generator(device=W.dev).manual_seed(1729)
    v = torch.randn(z.shape, dtype=z.dtype, device=z.device, generator=gen)
    v /= v.abs().max()
    cells = [6791, 7272, 501]
    forward = rows.action(v)
    local = rows.local_action(cells, v, 1.)
    rerr = float((local-forward[cells]).norm()/forward[cells].norm())
    assert rerr < 1e-11, ('adjoint rows', rerr)
    for a, b in zip(saved, (W.tx, W.tG, W.tgx)):
        assert torch.equal(a, b), 'analytic helper changed production buffers'
    log('FEASIBLECHECK G relative %.3e; row/action %.3e; solver buffers unchanged' % (rel, rerr))
    A = W.I3+G/ST.th
    normals = [rows.normal(c, A[c], ST.th) for c in cells]
    predicted = torch.stack([(nv*v).sum() for nv in normals])
    for h in (1e-7, 1e-8):
        W.tx.copy_(z+h*v); W._run()
        plus = torch.linalg.det(W.I3+W.tG[cells]/ST.th)
        W.tx.copy_(z-h*v); W._run()
        minus = torch.linalg.det(W.I3+W.tG[cells]/ST.th)
        actual = (plus-minus)/(2*h)
        err = float((actual-predicted).norm()/predicted.norm())
        assert err < 2e-5, ('det derivative', h, err)
        log('FEASIBLECHECK determinant directional h %.1e relative %.3e' % (h, err))
    pinv = ST.CG.t['Pinv']
    rhs = torch.tensor([.01, .02, .015], dtype=z.dtype, device=z.device)
    p = correction(metric_system(normals, pinv), rhs)
    actual = torch.stack([(nv*p).sum() for nv in normals])
    err = float((actual-rhs).norm()/rhs.norm())
    assert err < 1e-9, ('coupled lift', err)
    log('FEASIBLECHECK coupled lift relative %.3e; cached rows %d (3 adjoints each, no energy)' % (err, rows.builds))
    ST._eval(0.)
    ST._accept()


def stationarity(ST):
    """Euclidean tangent residual near active boundaries, not a success test."""
    rows = getattr(ST, '_feasible_rows', None) or GradientRows(ST.W)
    A = ST.W.I3+ST.W.tG/ST.th
    det = torch.linalg.det(A)
    cells = torch.nonzero(det < ST.flip_guard*1.01).flatten().tolist()
    g = wp.to_torch(ST.a['gz'], requires_grad=False).flatten().clone()
    tang = g.clone()
    rank = 0
    if cells:
        N = torch.stack([rows.normal(c, A[c], ST.th).flatten() for c in cells])
        N /= N.norm(dim=1)[:, None].clamp_min(1e-300)
        U, s, Vh = torch.linalg.svd(N, full_matrices=False)
        rank = int((s > s[0]*1e-10).sum())
        Q = Vh[:rank]
        tang -= Q.T @ (Q @ g)
    return dict(cells=cells, rank=rank, min_det=float(det.min()),
                gradient=float(g.norm()), tangent=float(tang.norm()), normal=float((g-tang).norm()))


def slide(ST, st, margin, blocking, pnorm, evals):
    """Projected direction, optional curved retraction, actual-step Armijo.

    SOC mode corrects determinant defects of a tangent trial with simultaneous
    normal Newton steps. It uses the full nonlinear determinant (hence includes
    its quadratic/cubic terms), not a full constrained energy Hessian/SQP solve.
    Only feasible endpoints with actual energy decrease are accepted.
    """
    W = ST.W
    rows = ST._feasible_rows
    start = time.perf_counter()
    z = ST.tz.clone()
    gz = wp.to_torch(ST.a['gz'], requires_grad=False).clone()
    A0 = W.I3+W.tG.clone()/ST.th
    d0 = torch.linalg.det(A0)
    near = torch.nonzero(d0 < 1.5*margin).flatten().tolist()
    near.sort(key=lambda c: float(d0[c]))
    # Never let sorting/truncation drop the cell that actually blocked the step.
    active = ([blocking] if blocking >= 0 else []) + [c for c in near if c != blocking]
    active = active[:ST.slide_max]
    if not active:
        return dict(taken=False, evals=evals)
    pinv = ST.CG.t['Pinv']
    P = wp.to_torch(ST.a['P'], requires_grad=False)
    normP = lambda u: torch.sqrt((u*torch.einsum('kab,kb->ka', P, u)).sum().clamp_min(0.))
    normals = [rows.normal(c, A0[c], ST.th) for c in active]
    system = metric_system(normals, pinv)
    N, Q, Sinv, scale = system
    y0 = -torch.einsum('kab,kb->ka', pinv, gz)
    # P-metric projection; all active constraints are handled together.
    tang = y0-torch.einsum('i,ika->ka', Sinv @ torch.einsum('ika,ka->i', N, y0), Q)
    radius = pnorm if ST.slide_delta <= 0 else min(pnorm, ST.slide_delta)
    tang *= radius/max(float(normP(tang)), 1e-300)
    target = torch.clamp(1.2*margin-d0[active], min=0.)
    lift = correction(system, target)
    # If necessary shorten only the tangent, not the already specified lift.
    # Triangle bound keeps the combined step in a controlled P-norm ball.
    lift_norm = float(normP(lift))
    tang *= max(0., radius-lift_norm)/max(float(normP(tang)), 1e-300)
    y = tang+lift
    slope = float((gz*y).sum())
    if slope >= 0 or not math.isfinite(slope):
        y = tang
        slope = float((gz*y).sum())
    if slope >= 0 or not math.isfinite(slope):
        st['slide_fail'] = st.get('slide_fail', 0)+1
        return dict(taken=False, evals=evals)
    D = rows.action(y)/ST.th
    tc, cblock2 = ST._first_crossing_cell(A0, D, margin)
    use_soc = ST.slide_method == 'soc'
    t = 1. if use_soc else min(1., .9*tc)
    stepbuf = wp.to_torch(ST.CG.a['p'], requires_grad=False)
    e0 = ST._eval(0.)[PS_PZ]
    evals += 1
    taken = False
    soc_iterations = 0
    max_local = 0
    error = math.inf
    min_det = float(d0.min())
    actual_slope = slope
    delta_e = math.nan
    accepted_norm = 0.
    for trial in range(18):
        p = t*y
        Atrial = A0+t*D
        if use_soc:
            # Retract a short tangent trial onto its nearby feasible surface.
            # Reject large corrections rather than jumping to another branch.
            bound = max(float(normP(p)), 1e-300)
            total_corr = torch.zeros_like(p)
            for _ in range(8):
                det = torch.linalg.det(Atrial)
                violated = torch.nonzero(det < margin*(1+1e-5)).flatten().tolist()
                if not violated:
                    break
                violated.sort(key=lambda c: float(det[c]))
                if len(violated) > 4*ST.slide_max:
                    break
                max_local = max(max_local, len(violated))
                ns = [rows.normal(c, Atrial[c], ST.th) for c in violated]
                sys = metric_system(ns, pinv)
                need = margin*(1+2e-5)-det[violated]
                dp = correction(sys, need)
                if float(normP(total_corr+dp)) > .5*bound:
                    break
                total_corr += dp
                p += dp
                Atrial = A0+rows.action(p)/ST.th
                soc_iterations += 1
        min_det = float(torch.linalg.det(Atrial).min())
        actual_slope = float((gz*p).sum())
        if min_det < margin or actual_slope >= 0 or not math.isfinite(actual_slope):
            t *= .5
            continue
        stepbuf.copy_(p)
        out = ST._eval(1.)
        evals += 1
        # Independent check on the production buffers, no relaxation to det(z).
        min_det = float(torch.linalg.det(W.I3+W.tG/ST.th).min())
        delta_e = out[PS_PQ]-e0
        if (math.isfinite(out[PS_PQ]) and math.isfinite(out[PS_GQ2]) and
                min_det >= margin*(1-1e-9) and delta_e <= 1e-4*actual_slope):
            ST._accept()
            taken = True
            error = math.sqrt(max(out[PS_GQ2], 0.))
            accepted_norm = float(normP(p))
            break
        t *= .5
    st['slide'] = st.get('slide', 0)+1
    st['slide_soc_iters'] = st.get('slide_soc_iters', 0)+soc_iterations
    st['slide_seconds'] = st.get('slide_seconds', 0.)+time.perf_counter()-start
    st['slide_row_builds'] = rows.builds
    st['slide_G_actions'] = rows.actions
    st.setdefault('slide_detail', []).append(dict(method=ST.slide_method, active=active, t=t,
        tc=tc, taken=taken, dE=delta_e, slope=actual_slope, det=min_det,
        g=error, pnorm=accepted_norm, soc=soc_iterations, soc_active=max_local))
    if taken:
        return dict(taken=True, evals=evals, g2=out[PS_GQ2])
    st['slide_fail'] = st.get('slide_fail', 0)+1
    ST._eval(0.)
    evals += 1
    ST._accept()
    return dict(taken=False, evals=evals)
