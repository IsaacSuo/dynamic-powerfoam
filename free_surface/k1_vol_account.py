"""The volume account of F against L, step by step (11-06, the expert's check of the collapse: logging only, nothing
changes). merged_drop.py --vol-account [cells]. Around each W.end_ul, from its own inputs (G at the accepted z in W.tG,
the frozen C in tul['Q'], F0 in tsp['F0'], L before, F_c at z) and its outputs (W._Fe, L after), per cell:
  a      log det(I + G / theta)              (the step's end map H = I + G / theta: F_end = H F0)
  b      C : G / theta                       (L's increment, checked against L after - L before)
  trial  log det F_end - log det F0          (checked against a)
  ret    log det F_e - log det F_end         (the return's change; 0 when it keeps the volume)
so  d(log det F - L) = (a - b) + ret  exactly; flips (det H <= 0, det F <= 0) counted apart. Summed over the run per
cell on the device, with 1/2 tr(dG^2) (dG = G / theta) and the steps where tr dG changed sign. Each step a line for the
watched cells; every 25 steps from step 200 the cells with the largest drift and its parts."""
import numpy as np
import torch

f64 = torch.float64


def drift_stats(d):
    """a drift field's distribution (log det F - L per cell): both tails, the mean |.|, quantiles of the signed and of |.|, the volume ratio at the extremes"""
    q = lambda t_, p_: float(torch.quantile(t_, p_))
    a = d.abs()
    return ('min %+.4f (det F / e^L %.3f), max %+.4f (%.3f); mean |.| %.2e, quantiles 1 / 5 / 50 / 95 / 99 %% %+.2e %+.2e %+.2e %+.2e %+.2e; |.| 50 / 90 / 99 / 99.9 %% %.2e %.2e %.2e %.2e; |.| > 0.01: %d, > 0.05: %d, > 0.1: %d of %d' % (
        float(d.min()), float(torch.exp(d.min())), float(d.max()), float(torch.exp(d.max())), float(a.mean()), q(d, .01), q(d, .05), q(d, .5), q(d, .95), q(d, .99),
        q(a, .5), q(a, .9), q(a, .99), q(a, .999), int((a > .01).sum()), int((a > .05).sum()), int((a > .1).sum()), len(d)))


def install(W, ST, step_of, watch=(6696,), log=print, every=25, from_step=200):
    end0 = W.end_ul
    n, dev = W.n, W.dev
    Z = lambda: torch.zeros(n, dtype=f64, device=dev)
    S = dict(ab=Z(), ret=Z(), half=Z(), alt=Z(), steps=0, flips=Z(), fin=Z(), frz=Z(), atr=Z(), adL=Z())   # (11-06: a - b = fin + frz, fin = a - tr dG, frz = (I - C) : dG; sums of |tr dG| and |dL|)
    prev = dict(tr=None)
    I3 = torch.eye(3, dtype=f64, device=dev)

    def logdet(M):
        sg, la = torch.linalg.slogdet(M)
        return sg, la

    def end_ul(th=.5, **k):
        G = W.tG.clone(); Q = W.tul['Q'].clone(); F0 = W.tsp['F0'].clone(); L0 = W.tul['Ln'].clone(); Fc = W.tFc.clone()
        out = end0(th, **k)
        Fend = (Fc-(1-th)*F0)/th
        dG = G/th; H = I3+dG
        sH, a = logdet(H); s0, l0 = logdet(F0); se, lend = logdet(Fend); sf, lfe = logdet(W._Fe)
        b = (Q*G).sum((1, 2))/th; dL = W.tul['Ln']-L0
        trial = lend-l0; ret = lfe-lend
        ok = (sH > 0) & (s0 > 0) & (se > 0) & (sf > 0)
        S['ab'] += torch.where(ok, a-b, torch.zeros_like(a)); S['ret'] += torch.where(ok, ret, torch.zeros_like(a))
        S['flips'] += (~ok).to(f64)
        trd = dG.diagonal(dim1=1, dim2=2).sum(1)
        S['fin'] += torch.where(ok, a-trd, torch.zeros_like(a)); S['frz'] += torch.where(ok, trd-b, torch.zeros_like(a))
        S['atr'] += trd.abs(); S['adL'] += dL.abs()
        S['half'] += .5*(dG*dG.transpose(1, 2)).sum((1, 2))                         # (1/2 tr(dG^2))
        tr = dG.diagonal(dim1=1, dim2=2).sum(1)
        if prev['tr'] is not None:
            S['alt'] += ((tr*prev['tr']) < 0).to(f64)
        prev['tr'] = tr; S['steps'] += 1
        s = step_of()
        for c in watch:
            svH = torch.linalg.svdvals(H[c]); svF = torch.linalg.svdvals(W._Fe[c])
            log('VOL step %d cell %d: sv(I + dG) %s, sv(F) %s; tr dG %+.3e, a %+.3e, b %+.3e (dL %+.3e), trial %+.3e, ret %+.3e; 1/2 tr dG^2 %.3e; now log det F %+.4f, L %+.4f; sums: a-b %+.4f (fin %+.4f, frz %+.4f), ret %+.4f, |tr dG| %.3f, |dL| %.3f, 1/2 tr dG^2 %.4f, sign changes %d / %d%s' % (
                s, c, np.round(svH.cpu().numpy(), 3).tolist(), np.round(svF.cpu().numpy(), 3).tolist(), float(tr[c]), float(a[c]), float(b[c]), float(dL[c]), float(trial[c]), float(ret[c]), float(.5*(dG[c]@dG[c]).trace()), float(lfe[c]), float(W.tul['Ln'][c]),
                float(S['ab'][c]), float(S['fin'][c]), float(S['frz'][c]), float(S['ret'][c]), float(S['atr'][c]), float(S['adL'][c]), float(S['half'][c]), int(S['alt'][c]), S['steps']-1, '' if bool(ok[c]) else ' (a flip this step)'))
        if s >= from_step and s % every == 0:
            report(tag='step %d' % s)
        return out

    def report(tag='', top=6):
        drift = S['ab']+S['ret']
        q = lambda t_, p_: float(torch.quantile(t_, p_))
        log('VOL %s: drift distribution: %s' % (tag, drift_stats(drift)))
        log('VOL %s: all %d cells: drift (log det F - L) min %+.3f, 1 %% %+.3f, median %+.4f; below -0.3: %d, -1: %d, -3: %d; fin summed over cells %+.2f (min %+.3f), frz %+.2f (min %+.3f); |tr dG| per cell median %.3f, 99 %% %.3f, max %.3f; |dL| median %.3f, 99 %% %.3f, max %.3f' % (
            tag, n, float(drift.min()), q(drift, .01), q(drift, .5), int((drift < -.3).sum()), int((drift < -1).sum()), int((drift < -3).sum()), float(S['fin'].sum()), float(S['fin'].min()), float(S['frz'].sum()), float(S['frz'].min()),
            q(S['atr'], .5), q(S['atr'], .99), float(S['atr'].max()), q(S['adL'], .5), q(S['adL'], .99), float(S['adL'].max())))
        o = torch.argsort(drift)[:top]
        V0m = W.V0.median()
        log('VOL %s: the most shrunk F (log det F - L summed): %s' % (tag, '; '.join(
            '%d: %+.3f = (a-b) %+.3f [fin %+.3f, frz %+.3f] + ret %+.3f; sum |tr dG| %.3f, |dL| %.3f (L now %+.3f); sum 1/2 tr dG^2 %.3f; sign changes %d / %d; flips %d; V0/med %.3f' % (
                int(c), float(drift[c]), float(S['ab'][c]), float(S['fin'][c]), float(S['frz'][c]), float(S['ret'][c]), float(S['atr'][c]), float(S['adL'][c]), float(W.tul['Ln'][c]), float(S['half'][c]), int(S['alt'][c]), S['steps']-1, int(S['flips'][c]), float(W.V0[c]/V0m)) for c in o)))
    W.end_ul = end_ul
    return report
