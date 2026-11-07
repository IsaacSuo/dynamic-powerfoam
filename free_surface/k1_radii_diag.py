"""Why the radii solve stalls (11-06). merged_drop.py --replay <stall snapshot> --replay-mode radii: logging only.
At the step's start state of a saved stall, the radii solve of update_ul again (same p, start radii, targets V0 exp(L),
tol), traced; at the point where it stops, for the cells with the largest |V - target| / target:
  the cell       its V, target, cap area, walls (count; the smallest site distance L / h: twins), hidden neighbours
  the solve      the Newton direction dw by Recut.cg at rtol 1e-12, its residual |J dw - res| / |res|
  the Jacobian   J's column of the cell against central and one-sided differences of the geometry's volumes (the
                 one-sided slopes differing = a kink: a cap, a wall or a cell appearing or vanishing right there)
  scans          |res| along w + t dw (t from 1e-8 to 1), and res_k along the cell's own weight alone"""
import math

import torch

f64 = torch.float64


def diagnose(W, x, log, ncell=3):
    R = W._recut; n = W.n
    r0 = W._ulr.clone(); J = torch.exp(W.tul['Ln']); Vt = W.V0*J.clamp(min=.02)
    R.trace = []
    r, G, S, it, nrep = R.solve_radii(x, r0, Vt, tol=W._ul_tol, r_base=W._rr0)
    tr = R.trace; R.trace = None
    res = Vt-G.V.detach(); err = float((res/Vt).abs().max())
    log('RADII replay: %d iterations, stopped at error %.3g (stalls so far %s); %s' % (
        it, err, R.stats.get('stalled', 0), ' | '.join('%s, halvings %d' % (t_, h_) for t_, h_ in tr[-4:])))
    h = float(W.V0.median())**(1/3)
    Jt = R.jacobian(G, r); dw = R.cg(Jt, res, Vt, rtol=1e-12)
    dw = dw[0] if isinstance(dw, tuple) else dw
    i_, j_, c_, d_ = Jt
    Hdw = d_*dw-torch.zeros_like(dw).index_add(0, i_, c_*dw[j_]).index_add(0, j_, c_*dw[i_])
    log('RADII the Newton direction at rtol 1e-12: |J dw - res| / |res| %.1e; |dw / w| max %.2e' % (
        float((Hdw-res).norm()/res.norm()), float((dw/(r*r)).abs().max())))
    w = r*r

    def V_at(wv):
        try:
            Gx, Sx = R.geometry(x, torch.sqrt(wv), reuse=True)
        except RuntimeError as e_:
            return None, str(e_)[:60]
        return Gx.V.detach(), Sx
    # the single precision clipping against the double one, at w and at tiny trial steps
    import time as _t
    for f32 in (True, False):
        R.A.f32 = f32; rows = []
        Gb, Sb = R.geometry(x, r, reuse=True)
        for t in (1e-8, 1e-6, 1e-4):
            Gx, Sx = R.geometry(x, torch.sqrt(w+t*dw), reuse=True)
            dV = (Gx.V.detach()-Gb.V.detach()).abs()/Vt; cm_ = int(dV.argmax())
            rows.append('t %g: pairs %d vs %d, largest |dV| / target %.3g at %d' % (t, len(Sx['pairs']), len(Sb['pairs']), float(dV[cm_]), cm_))
        torch.cuda.synchronize(); t0 = _t.perf_counter()
        for _ in range(10):
            R.A(x, r, reuse=True)
        torch.cuda.synchronize(); ta = 1e3*(_t.perf_counter()-t0)/10
        log('RADII alpha complex in %s precision: %s; the alpha call %.2f ms' % ('single' if f32 else 'double', ' | '.join(rows), ta))
    R.A.f32 = True
    G64, S64 = None, None
    R.A.f32 = False; G64, S64 = R.geometry(x, r, reuse=True); R.A.f32 = True; G32, S32 = R.geometry(x, r, reuse=True)
    k1 = set(map(tuple, S32['pairs'].tolist())); k2 = set(map(tuple, S64['pairs'].tolist()))
    dV = (G32.V.detach()-G64.V.detach()).abs()/Vt; cm_ = int(dV.argmax())
    log('RADII single vs double at w: pairs only in single %s, only in double %s; largest |V32 - V64| / target %.3g at %d (V0 / med %.3g); cells off by more than 1e-6: %d' % (
        sorted(k1-k2)[:6], sorted(k2-k1)[:6], float(dV[cm_]), cm_, float(W.V0[cm_]/W.V0.median()), int((dV > 1e-6).sum())))
    # the same inputs again: is the cut deterministic here
    for reuse in (True, False):
        outs = []
        for _ in range(4):
            Gx, Sx = R.geometry(x, r, reuse=reuse); outs.append((len(Sx['pairs']), Gx.V.detach().clone(), Sx['pairs'].clone()))
        dv = max(float(((o[1]-outs[0][1]).abs()/Vt).max()) for o in outs[1:])
        log('RADII repeat at the same p, r (reuse overlaps %s): pairs %s; max |V - V_first| / target %.3g' % (reuse, [o[0] for o in outs], dv))
        if dv > 1e-9:
            o1, o2 = outs[0][2], next(o[2] for o in outs[1:] if not torch.equal(o[2], outs[0][2])) if any(not torch.equal(o[2], outs[0][2]) for o in outs[1:]) else (None, None)
            if o1 is not None:
                k1 = set(map(tuple, o1.tolist())); k2 = set(map(tuple, o2.tolist()))
                log('RADII   pairs only in the first: %s; only in another: %s' % (sorted(k1-k2)[:6], sorted(k2-k1)[:6]))
    # what changes at a tiny trial step: the structure, the hidden cells, the single precision clipping's fallback
    for t in (1e-8, 1e-6, 1e-4):
        fb0 = R.A.f32_falls
        Gx, Sx = R.geometry(x, torch.sqrt(w+t*dw), reuse=True); fb1 = R.A.f32_falls
        Gb, Sb = R.geometry(x, r, reuse=True); fb2 = R.A.f32_falls
        dV = (Gx.V.detach()-Gb.V.detach()).abs()/Vt; top = torch.argsort(dV, descending=True)[:3].tolist()
        log('RADII trial t %g: pairs %d vs %d at w (structure pairs %d vs %d), hidden %d vs %d, single precision fell back: trial %s, base %s; largest |dV| / target %s' % (
            t, len(Gx.i), len(Gb.i), len(Sx['pairs']), len(Sb['pairs']), int(Sx['hidden'].sum()), int(Sb['hidden'].sum()), fb1 > fb0, fb2 > fb1,
            ', '.join('%d: %.3g (V / target %.3g -> %.3g, V0 / med %.3g)' % (c_, float(dV[c_]), float(Gb.V[c_]/Vt[c_]), float(Gx.V[c_]/Vt[c_]), float(W.V0[c_]/W.V0.median())) for c_ in top)))
    worst = torch.argsort((res/Vt).abs(), descending=True)[:ncell].tolist()
    kw = int((dw/w).abs().argmax())
    if kw not in worst:
        worst = [kw]+worst[:ncell-1]
    log('RADII the largest |dw / w| %.3g at %d: its J diagonal %.3g (walls %.3g, cap %.3g), res %.3g (V / target %.3g); the median J diagonal %.3g' % (
        float((dw/w)[kw]), kw, float(d_[kw]), float(d_[kw]-G.patch_A.detach()[kw]/(2*r[kw])), float(G.patch_A.detach()[kw]/(2*r[kw])), float(res[kw]), float(G.V[kw]/Vt[kw]), float(d_.median())))
    for k in worst:
        kp = G.wall_A.detach() > 0
        mine = kp & ((G.i == k) | (G.j == k))
        Ls = (G.p[G.j[mine]]-G.p[G.i[mine]]).norm(dim=1)
        nb = torch.cat([G.j[mine & (G.i == k)], G.i[mine & (G.j == k)]])
        hid = S['hidden'][nb].sum() if len(nb) else 0
        log('RADII cell %d: V %.4g, target %.4g (V / target %.4g), cap area %.3g, walls %d (smallest L / h %.4f), hidden neighbours %d, r / h %.3f' % (
            k, float(G.V[k]), float(Vt[k]), float(G.V[k]/Vt[k]), float(G.patch_A[k]), int(mine.sum()), float(Ls.min())/h if len(Ls) else float('nan'), int(hid), float(r[k])/h))
        # the Jacobian's column against differences
        col = torch.zeros(n, dtype=f64, device=x.device); col[k] = d_[k]
        col.index_add_(0, j_[i_ == k], -c_[i_ == k]); col.index_add_(0, i_[j_ == k], -c_[j_ == k])
        for eps in (1e-6, 1e-4, 1e-2):
            dk = eps*w[k]
            wp_, wm_ = w.clone(), w.clone(); wp_[k] += dk; wm_[k] -= dk
            Vp, _ = V_at(wp_); Vm, _ = V_at(wm_); V0_ = G.V.detach()
            if Vp is None or Vm is None:
                log('RADII cell %d eps %g: the geometry failed' % (k, eps)); continue
            up, dn, ce = (Vp-V0_)/dk, (V0_-Vm)/dk, (Vp-Vm)/(2*dk)
            sel = (col.abs() > 0) | (ce.abs() > 1e-12*ce.abs().max())
            log('RADII cell %d dV/dw_k, eps %g w_k: own entry J %.4g, central %.4g, up %.4g, down %.4g; the column over %d cells: |J - central| / |J| %.2e, |up - down| / |central| %.2e' % (
                k, eps, float(col[k]), float(ce[k]), float(up[k]), float(dn[k]), int(sel.sum()), float((col-ce)[sel].norm()/col[sel].norm()), float((up-dn)[sel].norm()/ce[sel].norm().clamp(min=1e-300))))
        # scans
        row = []
        for t in (1e-8, 1e-6, 1e-4, 1e-2, .1, .5, 1.):
            Vx, _ = V_at(w+t*dw)
            row.append('%g: %s' % (t, 'fail' if Vx is None else '%.6g' % (float((Vt-Vx).norm()/res.norm()))))
        row2 = []
        for s_ in (-.3, -.1, -1e-2, -1e-4, 1e-4, 1e-2, .1):
            wv = w.clone(); wv[k] *= (1+s_); Vx, _ = V_at(wv)
            row2.append('%g: %s' % (s_, 'fail' if Vx is None else '%.4g' % float((Vx[k]-Vt[k])/Vt[k])))
        log('RADII cell %d scans: |res(w + t dw)| / |res| at t = %s; (V_k - target) / target with w_k x (1 + s) at s = %s' % (k, ', '.join(row), ', '.join(row2)))
