"""Sensitivity of silhouette moment features to the material / contact parameters (springgaus_train.py features).

Feature groups, all cameras, from the per-frame moments (centroid cx, cy; sqrt second moments sx, sy; principal angle):
  bounce   - vertical centroid cy, frames 9..19 (px)
  drift    - horizontal centroid cx, frames 9..19 (px)
  impact   - sx, sy at frames 8, 9 (px)
  aspect   - 100 * sy / sx, frames 10..19 (percent)
  angle    - principal-axis angle x relative anisotropy |sy-sx|/sx, frames 10..19 (deg; near-round frames count little)
and derived scalars per camera: period and decay rate of a damped cosine fitted to the aspect series (frames 10..19),
the bounce apex (min cy after frame 9, parabola through the three lowest-cy frames: frame and height).
Prints, for each perturbation, the rms change of every group vs the base run, and the base-vs-real residual.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

GROUPS = ('bounce', 'drift', 'impact', 'aspect', 'angle')


def series(feat, cam, key):
    return np.array([fr[key] for fr in feat[cam]])


def groups(feat, cams):
    out = {g: [] for g in GROUPS}
    for c in cams:
        cx, cy, sx, sy, an = (series(feat, c, k) for k in ('cx', 'cy', 'sx', 'sy', 'angle'))
        out['bounce'].append(cy[9:20]); out['drift'].append(cx[9:20])
        out['impact'].append(np.concatenate((sx[8:10], sy[8:10]))); out['aspect'].append(100*sy[10:20]/sx[10:20])
        # principal angle weighted by the anisotropy (ill-defined for a near-round silhouette): (sy-sx) * angle
        out['angle'].append(np.degrees(an[10:20])*np.abs(sy[10:20]-sx[10:20])/sx[10:20])
    return {g: np.concatenate(v) for g, v in out.items()}


def derived(feat, cam):
    sx, sy, cy = (series(feat, cam, k) for k in ('sx', 'sy', 'cy'))
    t = np.arange(10, 20, dtype=float); a = sy[10:20]/sx[10:20]
    f = lambda t_, A, k, T, ph, c: c+A*np.exp(-k*(t_-10))*np.cos(2*np.pi*(t_-10)/T+ph)
    best = None
    for T0 in (3., 4., 5., 6., 7., 8.):
        for ph0 in (0., 1.5, 3., 4.5):
            try:
                p, _ = curve_fit(f, t, a, p0=(np.ptp(a)/2, .1, T0, ph0, a.mean()), maxfev=20000,
                                 bounds=((0, -.5, 2., -10, 0), (5, 2., 20., 10, 5)))
            except RuntimeError:
                continue
            r = ((f(t, *p)-a)**2).sum()
            if best is None or r < best[0]:
                best = (r, p)
    period, decay = (best[1][2], best[1][1]) if best else (np.nan, np.nan)
    k = 9+int(np.argmin(cy[9:20])); k = min(max(k, 10), 18)
    y0, y1, y2 = cy[k-1], cy[k], cy[k+1]
    den = y0-2*y1+y2
    off = float(np.clip(.5*(y0-y2)/den, -1, 1)) if abs(den) > 1e-9 else 0.
    return dict(period=period, decay=decay, apex_frame=k+off, apex_cy=y1-.25*(y0-y2)*off)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dir', type=Path, default=Path('output/springgaus_dem/burger_filled_mf0.1/sens'))
    p.add_argument('--runs', nargs='+', default=['E+10%', 'nu0.47', 'damp+20%', 'rest+0.3logit', 'fric+0.3logit'])
    args = p.parse_args()
    base = json.loads((args.dir/'base'/'features.json').read_text())
    cams = base['cams']
    gb, gr = groups(base['sim'], cams), groups(base['real'], cams)
    print('%-16s' % 'rms change'+''.join('%12s' % g for g in GROUPS)+'   | derived (mean over cams): period / decay / apex frame / apex cy')
    units = dict(bounce='px', drift='px', impact='px', aspect='%', angle='deg')
    db = {c: derived(base['sim'], c) for c in cams}; dr = {c: derived(base['real'], c) for c in cams}
    mean = lambda d, k: np.nanmean([d[c][k] for c in cams])
    rows = {}
    for run in args.runs:
        f = json.loads((args.dir/run/'features.json').read_text())
        g = groups(f['sim'], cams)
        dd = {c: derived(f['sim'], c) for c in cams}
        rows[run] = {k: float(np.sqrt(((g[k]-gb[k])**2).mean())) for k in GROUPS}
        print('%-16s' % run+''.join('%12.2f' % rows[run][k] for k in GROUPS)+
              '   | dT %+.2f fr, dk %+.3f /fr, d apex %+.2f fr, %+.1f px' % (mean(dd, 'period')-mean(db, 'period'), mean(dd, 'decay')-mean(db, 'decay'),
                                                                       mean(dd, 'apex_frame')-mean(db, 'apex_frame'), mean(dd, 'apex_cy')-mean(db, 'apex_cy')))
    res = {k: float(np.sqrt(((gb[k]-gr[k])**2).mean())) for k in GROUPS}
    print('%-16s' % 'base vs real'+''.join('%12.2f' % res[k] for k in GROUPS))
    print('%-16s' % 'units'+''.join('%12s' % units[k] for k in GROUPS))
    print('derived  base: period %.2f fr, decay %.3f /fr, apex frame %.2f, apex cy %.1f | real: period %.2f, decay %.3f, apex frame %.2f, apex cy %.1f' % (
        mean(db, 'period'), mean(db, 'decay'), mean(db, 'apex_frame'), mean(db, 'apex_cy'),
        mean(dr, 'period'), mean(dr, 'decay'), mean(dr, 'apex_frame'), mean(dr, 'apex_cy')))
    # each row normalised by its largest entry: how "diagonal" is the map parameter -> feature group
    print('\nrow-normalised (1 = the group this parameter moves most):')
    for run in args.runs:
        m = max(rows[run].values())
        print('%-16s' % run+''.join('%12.2f' % (rows[run][k]/m) for k in GROUPS))
    (args.dir/'sensitivity.json').write_text(json.dumps(dict(rows=rows, base_vs_real=res, units=units), indent=1))


if __name__ == '__main__':
    main()
