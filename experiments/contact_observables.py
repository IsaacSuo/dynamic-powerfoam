"""Search for silhouette observables that single out the contact-field coefficients (c0 height, c1 / c2 tilt).

Input: a featbatch directory run with --save-masks 1 (masks_<run>.npz: frames x cams x H x W uint8, masks_real.npz)
and its batch.json (central differences per parameter). Per frame and camera a set of silhouette features is computed
(moments, orientation, skewness, edge quantiles, bottom-contour slope / lean, ...). Scalar observables are linear
functionals of a feature series over automatic windows (pre-impact, impact, post-impact; impact = most squashed real
frame), e.g. pre-impact slope, slope change across the impact, post-impact mean. Noise per frame and feature is
estimated from the real masks (residual of a quadratic over the free-flight frames), and propagated to each
observable. For every observable: sensitivity to each parameter in noise units (z) and the 'purity' for each contact
coefficient (|z_target| / norm of z over the other parameters). Then Fisher analyses: (a) information per frame
(which feature families add information on c beyond feat6), (b) the current training loss Hessian (feat6 with its
scales) plus a candidate observable set: correlations of c with the rest and the conditioning.
"""
import argparse
import json
from pathlib import Path

import numpy as np

UNITS = dict(E=.1, nu=.01, damping=.1, friction=.1, alpha=.1, vx=.01, vy=.01, vz=.01, wx=.1, wy=.1, wz=.1, c0=1., c1=1., c2=1.)
CONTACT = ['c0', 'c1', 'c2']


def features(a):
    """a: H x W float in [0, 1]. Returns dict of per-frame features (px / rad / dimensionless)."""
    H, W = a.shape
    ys = np.arange(H, dtype=np.float64)[:, None]; xs = np.arange(W, dtype=np.float64)[None, :]
    m0 = max(a.sum(), 1e-6)
    cx = (a*xs).sum()/m0; cy = (a*ys).sum()/m0
    mu20 = (a*(xs-cx)**2).sum()/m0; mu02 = (a*(ys-cy)**2).sum()/m0; mu11 = (a*(xs-cx)*(ys-cy)).sum()/m0
    sx, sy = np.sqrt(mu20), np.sqrt(mu02)
    f = dict(cx=cx, cy=cy, sx=sx, sy=sy, sqrtA=np.sqrt(m0), aspect=sy/sx,
             theta=.5*np.arctan2(2*mu11, mu20-mu02),
             skx=(a*(xs-cx)**3).sum()/m0/sx**3, sky=(a*(ys-cy)**3).sum()/m0/sy**3,
             kxy=(a*(xs-cx)*(ys-cy)**2).sum()/m0/(sx*sy*sy), kyx=(a*(xs-cx)**2*(ys-cy)).sum()/m0/(sx*sx*sy))
    row = a.sum(1); col = a.sum(0)
    q = lambda m, p: float(np.interp(p*m.sum(), np.cumsum(m), np.arange(len(m))))
    f.update(top=q(row, .01), bottom=q(row, .99), left=q(col, .01), right=q(col, .99))
    hgt = f['bottom']-f['top']
    # bottom strip (lowest 15 % of the height): its x centroid relative to cx (lean of the contact zone) and its width
    w_strip = np.clip((ys[:, 0]-(f['bottom']-.15*hgt))/2+.5, 0, 1)[:, None]*a
    ms = max(w_strip.sum(), 1e-6)
    xb = (w_strip*xs).sum()/ms
    f.update(lean_bottom=xb-cx, width_bottom=2*np.sqrt((w_strip*(xs-xb)**2).sum()/ms))
    w_top = np.clip(((f['top']+.15*hgt)-ys[:, 0])/2+.5, 0, 1)[:, None]*a
    mt = max(w_top.sum(), 1e-6)
    f.update(lean_top=(w_top*xs).sum()/mt-cx)
    # bottom / top contour per column (columns carrying >= 30 % of the max column mass): lowest / highest point, slope
    good = col >= .3*col.max()
    cs = np.cumsum(a, 0)
    tot = cs[-1]
    xg = np.nonzero(good)[0]
    yb = np.array([np.interp(.98*tot[j], cs[:, j], np.arange(H)) for j in xg])
    yt = np.array([np.interp(.02*tot[j], cs[:, j], np.arange(H)) for j in xg])
    A = np.stack((xg-cx, np.ones(len(xg))), 1)
    f['slope_bottom'] = np.linalg.lstsq(A, yb, rcond=None)[0][0]
    f['slope_top'] = np.linalg.lstsq(A, yt, rcond=None)[0][0]
    half = xg < cx
    f['lr_bottom'] = yb[~half].mean()-yb[half].mean() if half.any() and (~half).any() else 0.
    return f


def series(m, cams):
    """m: frames x cams x H x W uint8 -> {feature: array frames x len(cams)}."""
    out = {}
    for f in range(m.shape[0]):
        for k, c in enumerate(cams):
            for n, v in features(m[f, c].astype(np.float64)/255).items():
                out.setdefault(n, np.zeros((m.shape[0], len(cams))))[f, k] = v
    return out


def linfit_w(fr):
    t = np.asarray(fr, float); t = t-t.mean()
    return t/(t**2).sum()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dir', type=Path, default=Path('output/springgaus_dem/burger_filled_mf0.1/synth2/ident'))
    p.add_argument('--cams', type=int, nargs='+', default=[1, 2])
    p.add_argument('--last', type=int, default=19, help='last training frame')
    p.add_argument('--pos-scale', type=float, default=2.)
    p.add_argument('--top', type=int, default=25)
    args = p.parse_args()
    batch = {r['name']: r for r in json.loads((args.dir/'batch.json').read_text())}
    load = lambda n: np.load(args.dir/('masks_%s.npz' % n))['m']
    real = series(load('real'), args.cams)
    S = {n: series(load(n), args.cams) for n in batch}
    params = [n[:-1] for n in batch if n.endswith('+')]
    fnames = list(real)
    L = args.last
    k_imp = [int(np.argmin(real['aspect'][1:L+1, k]))+1 for k in range(len(args.cams))]
    print('impact frame per camera', k_imp)
    # per-frame noise per feature and camera: residual std of a quadratic over the free-flight frames of the real video
    sig = {}
    for n in fnames:
        sig[n] = np.zeros(len(args.cams))
        for k in range(len(args.cams)):
            fr = np.arange(1, k_imp[k]-1)
            y = real[n][fr, k]
            r = y-np.polyval(np.polyfit(fr, y, 2), fr)
            sig[n][k] = max(r.std()*np.sqrt(len(fr)/(len(fr)-3)), 1e-3)
    print('per-frame noise (cam %s):' % args.cams, '  '.join('%s %s' % (n, np.round(sig[n], 4).tolist()) for n in fnames))
    # parameter derivative of every feature series, natural units
    D = {}
    s5 = 1/(1+np.exp(-(np.log(1.5)+batch['base']['parameters'][5])))      # nu = 0.5 s(p5): dnu/dp5 = 0.5 s (1 - s)
    for pn in params:
        h = batch[pn+'+']['step']
        u = UNITS[pn]/(.5*s5*(1-s5)) if pn == 'nu' else UNITS[pn]
        D[pn] = {n: (S[pn+'+'][n]-S[pn+'-'][n])/(2*h)*u for n in fnames}
    # observables: (feature, kind) -> weights over frames, per camera
    obs = []
    for k in range(len(args.cams)):
        ki = k_imp[k]
        pre, post, imp = list(range(1, ki-1)), list(range(ki+2, L+1)), [ki-1, ki, ki+1]
        wins = dict(pre_slope=(pre, linfit_w(pre)), post_slope=(post, linfit_w(post)),
                    pre_mean=(pre, np.full(len(pre), 1/len(pre))), post_mean=(post, np.full(len(post), 1/len(post))),
                    impact_mean=(imp, np.full(3, 1/3)), early_post_mean=(post[:4], np.full(4, .25)))
        for n in fnames:
            for wn, (fr, w) in wins.items():
                obs.append((n, wn, k, [(fr, w)]))
            obs.append((n, 'd_slope', k, [(post, linfit_w(post)), (pre, -linfit_w(pre))]))
            obs.append((n, 'impact-pre', k, [(imp, np.full(3, 1/3)), (pre[-3:], np.full(3, -1/3))]))
            obs.append((n, 'post-pre_mean', k, [(post, np.full(len(post), 1/len(post))), (pre, np.full(len(pre), -1/len(pre)))]))

    def apply(terms, s, k):
        return sum((w*s[fr, k]).sum() for fr, w in terms)
    rows = []
    for n, wn, k, terms in obs:
        noise = sig[n][k]*np.sqrt(sum((w**2).sum() for _, w in terms))
        z = np.array([apply(terms, D[pn][n], k)/noise for pn in params])
        rows.append((n, wn, k, z))
    print('\nbest observables per contact coefficient (z: change per natural unit in noise sigmas; purity = |z_target| / |z_others|):')
    for tgt in CONTACT:
        ti = params.index(tgt)
        sc = []
        for n, wn, k, z in rows:
            oth = np.delete(z, ti)
            sc.append((abs(z[ti])/max(np.linalg.norm(oth), 1e-9), abs(z[ti]), n, wn, k, z))
        sc.sort(key=lambda r: -min(r[0], 50)*np.log1p(r[1]))
        print('== %s' % tgt)
        for pur, zt, n, wn, k, z in [s_ for s_ in sc if s_[1] > 2][:args.top]:
            oth = [(params[j], z[j]) for j in np.argsort(-np.abs(z)) if params[j] != tgt][:3]
            print('  purity %6.2f  z %7.1f  %-13s %-14s cam%d | others: %s' % (pur, zt, n, wn, args.cams[k],
                  ', '.join('%s %+.1f' % o for o in oth)))
    # (a) per-frame Fisher information: feat6 only vs feat6 + each extra feature family
    feat6 = ['cx', 'cy', 'sx', 'sy', 'sqrtA', 'aspect']
    fr_all = np.arange(1, L+1)

    def fisher(feats, scale=None):
        F = np.zeros((len(params), len(params)))
        for n in feats:
            for k in range(len(args.cams)):
                s_ = scale(n, k) if scale else sig[n][k]
                J = np.stack([D[pn][n][fr_all, k] for pn in params], 1)/s_
                F += J.T@J
        return F

    def report(F, label):
        C = np.linalg.pinv(F); sd = np.sqrt(np.diag(C)); R = C/np.outer(sd, sd)
        out = []
        for tgt in CONTACT:
            i = params.index(tgt); r = R[i].copy(); r[i] = 0; j = np.argmax(np.abs(r))
            out.append('%s std %.3g (max corr %s %+.2f)' % (tgt, sd[i], params[j], r[j]))
        ev = np.linalg.eigvalsh(F)
        print('  %-34s %s | cond %.3g' % (label, ' | '.join(out), np.sqrt(ev[-1]/max(ev[0], 1e-30))))
    print('\n(a) per-frame information (noise from the real masks):')
    report(fisher(feat6), 'feat6')
    for extra in [n for n in fnames if n not in feat6]:
        report(fisher(feat6+[extra]), 'feat6 + %s' % extra)
    report(fisher(fnames), 'all features')
    # (b) current loss Hessian (Gauss-Newton): feat6 with the training scales (centroid pos-scale, others real std)
    def tscale(n, k):
        if n in ('cx', 'cy'):
            return args.pos_scale
        floor = .01 if n == 'aspect' else .5
        return max(real[n][1:L+1, k].std(), floor)
    H0 = fisher(feat6, tscale)/(L*len(args.cams))
    print('\n(b) training loss (feat6 terms, mean over frames x cams):')
    report(H0, 'current loss')
    rowd = {(n, wn, k): z for n, wn, k, z in rows}
    for cand in [['cx:d_slope'], ['cx:d_slope', 'cy:post-pre_mean'], ['bottom:post_mean'], ['slope_bottom:post_mean'],
                 ['lean_bottom:post_mean'], ['theta:d_slope'], ['cx:d_slope', 'bottom:post_mean'],
                 ['cx:d_slope', 'slope_bottom:post_mean', 'bottom:post_mean']]:
        H = H0.copy()
        for c_ in cand:
            n, wn = c_.split(':')
            for k in range(len(args.cams)):
                z = rowd[(n, wn, k)]
                H += np.outer(z, z)/len(args.cams)
        report(H, '+ ' + ', '.join(cand))


if __name__ == '__main__':
    main()
