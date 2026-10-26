"""Our drop (skin_drop_check.py --shift, saved as skin_drop_shift.npz) against PAC-NeRF's MPM on the same shape
(mpm_particles.py + experiments/mpm_drop_energy.py --ground slip), 2026-10-23: both at rest with the surface 2 cm above a
frictionless ground, ground-truth E and nu. Per kg; the centres of mass differ in definition (MPM particles, ours the
cells' masses at their sites), so their change from the start is compared. Ground force / weight: ours as recorded and
from the centre of mass's acceleration (1 + a / g), the MPM's from its centre of mass. Key events and a figure.
  python free_surface/mpm_compare.py --out output/round01_pruned_exact"""
import argparse
import json
from pathlib import Path

import numpy as np


def events(t, f, dy, vy, thr=.01):
    """touchdown (force first above thr), peak force, deepest centre of mass, lift-off (force back below thr after the
    peak), fastest rise after touchdown"""
    on = np.flatnonzero(f > thr)
    t0 = t[on[0]] if len(on) else np.nan
    ip = int(np.nanargmax(f)); idp = int(np.nanargmin(dy))
    off = np.flatnonzero((f < thr) & (t > t[ip]))
    t1 = t[off[0]] if len(off) else np.nan
    after = t > t0
    iv = int(np.flatnonzero(after)[np.argmax(vy[after])]) if after.any() else 0
    return dict(touch=t0, peak_force=f[ip], t_peak=t[ip], deepest=dy[idp], t_deep=t[idp], liftoff=t1, rise=vy[iv], t_rise=t[iv])


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--out', type=Path, default=Path('output/round01_pruned_exact'))
    a.add_argument('--ours', default='skin_drop_shift.npz')
    a.add_argument('--mpm', nargs='+', default=['mpm_drop_slip.json'])
    args = a.parse_args()
    z = np.load(args.out/args.ours)
    cols = [str(c) for c in z['cols']]; R = z['rec']; c = {k: R[:, i] for i, k in enumerate(cols)}
    M = float(np.load(args.out/'dem_exact_single'/'cells.npz')['masses'].sum())
    g = 9.8
    to = c['t']
    o = dict(t=to, dy=c['com_y']-c['com_y'][0], vy=c['com_vy'], K=c['K']/M, el=c['elastic']/M, f=c['contact/weight'])
    o['f_com'] = np.gradient(o['vy'], to)/g+1
    runs = []
    for name in args.mpm:
        m = json.loads((args.out/name).read_text())
        rows = m['rows']
        tm = np.array([r_['t'] for r_ in rows])
        p = dict(t=tm, dy=np.array([r_['com_y'] for r_ in rows])-rows[0]['com_y'], vy=np.array([r_['vy_com'] for r_ in rows]),
                 K=np.array([r_['ke_total'] for r_ in rows]), el=np.array([r_['elastic'] for r_ in rows]))
        p['f'] = np.gradient(p['vy'], tm)/g+1
        Hm = np.array([r_['ke_total']+r_['elastic']+r_['gravity'] for r_ in rows]); p['dH'] = Hm-Hm[0]
        p['label'] = 'MPM dx %g mm' % (1e3*m['dx'])
        runs.append(p)
        print('%s: E %.1f, nu %.4f, dt %.3e, %d particles, %.0f s' % (p['label'], m['E'], m['nu'], m['dt'], m.get('particles', -1), m['seconds']))
    o['dH'] = (c['K']+c['elastic'])/M+g*(c['com_y']-c['com_y'][0])
    ev = [events(to, o['f'], o['dy'], o['vy'])]+[events(p['t'], p['f'], p['dy'], p['vy']) for p in runs]
    print('%-38s %10s' % ('', 'ours')+''.join(' %14s' % p['label'] for p in runs))
    for k, lab, sc in (('touch', 'touchdown (force > 1% weight), s', 1), ('t_peak', 'peak force at, s', 1), ('peak_force', 'peak force / weight', 1),
                       ('t_deep', 'deepest centre of mass at, s', 1), ('deepest', 'deepest centre of mass drop, mm', 1e3), ('liftoff', 'lift-off, s', 1),
                       ('rise', 'fastest rise after impact, m/s', 1), ('t_rise', '  at, s', 1)):
        print('%-38s %10.4f' % (lab, ev[0][k]*sc)+''.join(' %14.4f' % (e_[k]*sc) for e_ in ev[1:]))
    print('\n   t    d com y (mm)        vy (m/s)          force / weight     total energy change (J/kg)')
    print('       ours'+''.join('  %7s' % ('mpm%d' % (i+1)) for i in range(len(runs)))+'   ours'+''.join('  %6s' % ('mpm%d' % (i+1)) for i in range(len(runs)))
          +'   ours'+''.join(' %5s' % ('mpm%d' % (i+1)) for i in range(len(runs)))+'     ours'+''.join('   %7s' % ('mpm%d' % (i+1)) for i in range(len(runs))))
    for tt in np.arange(0, min([to[-1]]+[p['t'][-1] for p in runs])+1e-9, .025):
        i = int(np.argmin(abs(to-tt))); js = [int(np.argmin(abs(p['t']-tt))) for p in runs]
        print('%5.3f %7.1f' % (tt, 1e3*o['dy'][i])+''.join(' %8.1f' % (1e3*p['dy'][j]) for p, j in zip(runs, js))
              +' %7.3f' % o['vy'][i]+''.join(' %7.3f' % p['vy'][j] for p, j in zip(runs, js))
              +' %5.2f' % o['f'][i]+''.join(' %5.2f' % p['f'][j] for p, j in zip(runs, js))
              +' %9.4f' % o['dH'][i]+''.join(' %9.4f' % p['dH'][j] for p, j in zip(runs, js)))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
        for (k, lab), axx in zip((('dy', 'centre of mass drop from start (m)'), ('vy', 'centre of mass vy (m/s)'), ('f', 'ground force / weight'),
                                  ('K', 'kinetic and elastic energy (J/kg)')), ax.ravel()):
            axx.plot(o['t'], o[k], color='#1f6fb4', label='ours' if k != 'K' else 'ours kinetic')
            for p, col in zip(runs, ('#d1495b', '#e8a33d', '#6a994e')):
                axx.plot(p['t'], p[k], color=col, label=p['label'] if k != 'K' else p['label']+' kinetic')
            if k == 'K':
                axx.plot(o['t'], o['el'], color='#1f6fb4', ls='--', label='ours elastic')
                for p, col in zip(runs, ('#d1495b', '#e8a33d', '#6a994e')):
                    axx.plot(p['t'], p['el'], color=col, ls='--', label=p['label']+' elastic')
            axx.set_xlabel('t (s)'); axx.set_title(lab, fontsize=10); axx.grid(alpha=.3); axx.legend(fontsize=8)
        fig.savefig(args.out/'mpm_compare.png', dpi=110)
        print('figure %s' % (args.out/'mpm_compare.png'))
    except Exception as e_:
        print('no figure:', e_)


if __name__ == '__main__':
    main()
