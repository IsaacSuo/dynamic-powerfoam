"""MergedUL.end_ul in Warp without host reads (11-06, stage 4 of the explicit step's restructuring; the same update,
to rounding):
  k_end_cell    per cell: Q = polar(I + G / th) by Newton's R <- (R + R^-T) / 2 to 1e-14 (polar_batch's rule: I where
                det <= 0.1; each cell to its own convergence, polar_batch to the largest change), H = sym(Q^T (I + G/th))
                - I (V0-weighted into the step's mean), the flips (det <= 1e-3); the yielding cell's dissipation and the
                least of them in units of the limit store; F_end = (F_c - (1 - th) F_0) / th and its rotation (rot_proper,
                k_F8's); ep <- Q ep Q^T; the yielding cell's return (the eigenvalues of sym(R^T F_end) by wp.eig3 in
                place of eigh on the CPU): U_e = V exp(lm + sc dv) V^T, F_e = R U_e, ep += R V ((1 - sc) dv) V^T R^T;
                the yielded flags; L += C : G / th
  k_end_spring  per spring: its dissipation, the remainder kept (dl = beta r), the springs' store
The plastic work, the stores and the step's counts stay on the device; flush(W) reads them into W.Wp, W.Wp_cell,
W.Wp_spring, W.spring_store and ul_stats (the runner calls it where it logs)."""
import torch
import warp as wp

from free_surface.closure_warp import rot_proper

f64, i32 = torch.float64, torch.int32


@wp.func
def sym(A: wp.mat33d):
    return (A+wp.transpose(A))*wp.float64(0.5)


@wp.func
def ddot(A: wp.mat33d, B: wp.mat33d):
    s = wp.float64(0.0)
    for r in range(3):
        for c in range(3):
            s = s+A[r, c]*B[r, c]
    return s


@wp.kernel(enable_backward=False)
def k_end_cell(G: wp.array(dtype=wp.mat33d), th: wp.float64, be: wp.array(dtype=wp.float64), ei: wp.array(dtype=wp.float64), er: wp.array(dtype=wp.mat33d),
               en: wp.array(dtype=wp.mat33d), Fc: wp.array(dtype=wp.mat33d), F0: wp.array(dtype=wp.mat33d), V0: wp.array(dtype=wp.float64),
               Qc: wp.array(dtype=wp.mat33d), yld: wp.array(dtype=wp.int32), sy: wp.float64, ey: wp.float64, mu: wp.float64, Fri: wp.array(dtype=wp.mat33d),
               ep: wp.array(dtype=wp.mat33d), Ln: wp.array(dtype=wp.float64), Fend: wp.array(dtype=wp.mat33d), Fe: wp.array(dtype=wp.mat33d),
               ynew: wp.array(dtype=wp.int32), Hc: wp.array(dtype=wp.mat33d), acc: wp.array(dtype=wp.float64), cnt: wp.array(dtype=wp.int32)):
    """acc: 0 the plastic work so far, 1 the cells' share, 3 the step's least dissipation / limit store; Hc the cells'
    V0 H (summed by torch); cnt: 0 flips, 1 yielding, 2 yielded"""
    k = wp.tid()
    one = wp.float64(1.0)
    I3 = wp.identity(n=3, dtype=wp.float64)
    Gs = G[k]/th
    dg = wp.determinant(I3+Gs)
    Fg = (I3+Gs)@Fri[k]                                                          # (the step's increment: Fri = F_rel(x_n)^-1, I at a cut)
    Q = I3
    if wp.determinant(Fg) > wp.float64(0.1):
        R = Fg
        for it in range(40):
            R1 = (R+wp.transpose(wp.inverse(R)))*wp.float64(0.5)
            ch = wp.float64(0.0)
            for r in range(3):
                for c in range(3):
                    ch = wp.max(ch, wp.abs(R1[r, c]-R[r, c]))
            R = R1
            if ch < wp.float64(1e-14):
                break
        Q = R
    H = sym(wp.transpose(Q)@Fg)-I3
    v0 = V0[k]
    Hc[k] = v0*H
    if dg <= wp.float64(1e-3):
        wp.atomic_add(cnt, 0, 1)
    b = be[k]
    yn = b < one
    if yn:
        wc = wp.float64(0.0)
        if th == one:
            wc = v0*sy*(ei[k]-ey)
        else:
            wc = v0*mu*(one-b)*(ddot(en[k], er[k])+b*ddot(er[k], er[k]))
        wp.atomic_add(acc, 0, wc)
        wp.atomic_add(acc, 1, wc)
        wp.atomic_min(acc, 3, wc/(v0*mu*ey*ey))
        wp.atomic_add(cnt, 1, 1)
    Fn = (Fc[k]-(one-th)*F0[k])/th
    Fend[k] = Fn
    Rr = rot_proper(Fn)
    epk = Q@ep[k]@wp.transpose(Q)
    Fek = Fn
    if yn:
        S = sym(wp.transpose(Rr)@Fn)
        Vv, w = wp.eig3(S)
        lw = wp.vec3d(wp.log(wp.max(w[0], wp.float64(1e-6))), wp.log(wp.max(w[1], wp.float64(1e-6))), wp.log(wp.max(w[2], wp.float64(1e-6))))
        lm = (lw[0]+lw[1]+lw[2])/wp.float64(3.0)
        dv = lw-wp.vec3d(lm, lm, lm)
        nr = wp.max(wp.length(dv), wp.float64(1e-300))
        a = wp.exp(lm)
        sc = wp.min(one, (ey/a)/nr)
        Ue = wp.mat33d()
        Dp = wp.mat33d()
        for e in range(3):
            ve = wp.vec3d(Vv[0, e], Vv[1, e], Vv[2, e])
            Ue = Ue+wp.exp(lm+sc*dv[e])*wp.outer(ve, ve)
            Dp = Dp+((one-sc)*dv[e])*wp.outer(ve, ve)
        Fek = Rr@Ue
        epk = epk+Rr@Dp@wp.transpose(Rr)
    ep[k] = epk
    Fe[k] = Fek
    y = int(0)
    if yld[k] != 0 or yn:
        y = 1
        wp.atomic_add(cnt, 2, 1)
    ynew[k] = y
    Ln[k] = Ln[k]+ddot(Qc[k], G[k])/th


@wp.kernel(enable_backward=False)
def k_end_spring(sb: wp.array(dtype=wp.float64), sk: wp.array(dtype=wp.float64), sr: wp.array(dtype=wp.vec3d), sdn: wp.array(dtype=wp.vec3d), th: wp.float64,
                 tdl: wp.array(dtype=wp.vec3d), acc: wp.array(dtype=wp.float64)):
    """acc: 0 the plastic work so far, 2 the springs' share so far, 4 the springs' store now"""
    t = wp.tid()
    one = wp.float64(1.0)
    b = sb[t]
    k = sk[t]
    r = sr[t]
    rr = wp.dot(r, r)
    w = wp.float64(0.0)
    if th == one:
        w = wp.float64(0.5)*k*b*(one-b)*rr
    else:
        w = wp.float64(0.5)*k*(one-b)*(wp.dot(r, sdn[t])+b*rr)
    wp.atomic_add(acc, 0, w)
    wp.atomic_add(acc, 2, w)
    d = b*r
    tdl[t] = d
    wp.atomic_add(acc, 4, wp.float64(0.5)*k*wp.dot(d, d))


class EndWarp:
    def __init__(self, W):
        n, dev, K = W.n, W.dev, W.sp['K']
        self.n, self.dev = n, dev
        self.acc = torch.zeros(5, dtype=f64, device=dev); self.cnt = torch.zeros(3, dtype=i32, device=dev)
        self.Hc = torch.zeros(n, 3, 3, dtype=f64, device=dev)
        self.base = None                                                              # (W's plastic work when the device's sums start: at the first end, after any restore)
        self.Fe = torch.zeros(n, 3, 3, dtype=f64, device=dev); self.ynew = torch.zeros(n, dtype=i32, device=dev)
        self.tdl = torch.zeros(K, 3, dtype=f64, device=dev)
        self.sumV0 = W.V0.sum()
        self.ident = torch.eye(3, dtype=f64, device=dev).expand(n, 3, 3).contiguous()

    def run(self, W, th, Fri=None):
        assert th == W.th, 'the model and the step on one theta'
        n, dev, t, s, m = self.n, self.dev, W.tul, W.tsp, W._m_s
        if self.base is None:
            self.base = (float(W.Wp), float(getattr(W, 'Wp_cell', 0.)), float(getattr(W, 'Wp_spring', 0.))); self.acc[:3].zero_()
        Wa = lambda t_, d_: wp.from_torch(t_, dtype=d_)
        F_, I_, V3, M3 = (lambda t_: Wa(t_, wp.float64)), (lambda t_: Wa(t_, wp.int32)), (lambda t_: Wa(t_, wp.vec3d)), (lambda t_: Wa(t_, wp.mat33d))
        self.acc[3] = float('inf'); self.acc[4] = 0.; self.cnt.zero_()                # (the step's least, store and counts afresh)
        if not W.ep.is_contiguous():
            W.ep = W.ep.contiguous()
        wp.launch(k_end_cell, n, inputs=[M3(W.tG), float(th), F_(W.tbe), F_(W.tei), M3(W.ter), M3(s['en']), M3(W.tFc), M3(s['F0']), F_(W.V0), M3(t['Q']),
                                         I_(W._yld.to(i32)), float(W.sy), float(W.ey), float(W.mu), M3(self.ident if Fri is None else Fri.contiguous())],
                  outputs=[M3(W.ep), F_(t['Ln']), M3(s['Fend']), M3(self.Fe), I_(self.ynew), M3(self.Hc), F_(self.acc), I_(self.cnt)], device=dev)
        if m:
            wp.launch(k_end_spring, m, inputs=[F_(s['sb']), F_(s['sk']), V3(s['sr']), V3(s['sdn']), float(th)], outputs=[V3(self.tdl), F_(self.acc)], device=dev)
        W.tdl = self.tdl[:m]
        W._Fe = self.Fe
        W._yld = self.ynew != 0
        W.Hsum += self.Hc.sum(0)/self.sumV0
        if getattr(W, 'end_flush_each', False):                                    # (the implicit driver reads the work and the counts each step)
            self.flush(W)

    def check(self, W, th):
        """both on the same state: the torch end_ul's outputs kept, the state put back, this one run, the largest
        differences printed (this one's result stays)"""
        names = ('ep', 'Hsum', '_Fe', '_yld', 'tdl', 'Wp', 'Wp_cell', 'Wp_spring')
        cp = lambda v_: v_.clone() if torch.is_tensor(v_) else v_
        keep = {k_: cp(getattr(W, k_, 0.)) for k_ in names}; Ln0 = W.tul['Ln'].clone()
        st0 = dict(W.ul_stats)
        W._end_torch(th)
        ref = {k_: cp(getattr(W, k_)) for k_ in names}; Ln1 = W.tul['Ln'].clone(); st1 = dict(W.ul_stats); store1 = W.spring_store
        for k_, v_ in keep.items():
            setattr(W, k_, v_)
        W.tul['Ln'].copy_(Ln0); W.ul_stats.clear(); W.ul_stats.update(st0)
        b0 = self.base; self.base = (float(keep['Wp']), float(keep['Wp_cell']), float(keep['Wp_spring'])); self.acc[:3].zero_()
        self.run(W, th); self.flush(W)
        rel = lambda a_, b_: float((a_.double()-b_.double()).abs().max()/b_.double().abs().max().clamp(min=1e-300)) if a_.numel() else 0.
        out = ['Ln %.1e' % rel(W.tul['Ln'], Ln1)]
        for k_ in ('ep', 'Hsum', '_Fe', '_yld', 'tdl'):
            a_, b_ = getattr(W, k_), ref[k_]
            out.append('%s %s' % (k_, ('%.1e' % rel(a_, b_)) if a_.shape == b_.shape else 'shape %s / %s' % (tuple(a_.shape), tuple(b_.shape))))
        for k_ in ('Wp', 'Wp_cell', 'Wp_spring'):
            out.append('%s %.1e' % (k_, abs(getattr(W, k_)-ref[k_])/max(abs(ref[k_]), 1e-300)))
        out.append('store %.1e' % (abs(W.spring_store-store1)/max(abs(store1), 1e-300)))
        out.append('counts %s / %s' % ([W.ul_stats.get(k_) for k_ in ('flips', 'yielding', 'yielded')], [st1.get(k_) for k_ in ('flips', 'yielding', 'yielded')]))
        print('  ENDCHECK %s' % ', '.join(out), flush=True)
        # (the device's totals now carry this step: the base stays the one before it)
        self.base = (float(keep['Wp']), float(keep['Wp_cell']), float(keep['Wp_spring']))

    def flush(self, W):
        """the device's plastic work, stores and the last step's counts into W's attributes (one read)"""
        a = self.acc[:5].tolist(); c = self.cnt.tolist()
        W.Wp, W.Wp_cell, W.Wp_spring = self.base[0]+a[0], self.base[1]+a[1], self.base[2]+a[2]
        W.spring_store = a[4]
        st = W.ul_stats
        if a[3] != float('inf'):
            st['D_min'] = a[3]
        st['flips'], st['yielding'], st['yielded'] = c
