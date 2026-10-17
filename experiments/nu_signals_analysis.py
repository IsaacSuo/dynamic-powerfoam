"""Analysis of nu_signals.py fisher features: for a change of nu by 0.1, the part of each observation that the other
parameters (log E*, log damping, friction, restitution) cannot reproduce after a linearised least-squares refit, in px,
and where in time it sits; plus the sensitivities of the free-flight state after the bounce (centroid velocity, spin)."""
import argparse
from pathlib import Path

import numpy as np

from experiments.nu_signals import PARAMS, STEP, lattice_shape


def rigid_state(p, X):
    c = p.mean(0); a = X-X.mean(0); b = p-c
    U, S, Vt = np.linalg.svd(b.T@a); D = np.diag([1, 1, np.sign(np.linalg.det(U@Vt))])
    return c, U@D@Vt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('files', nargs='+', type=Path)
    ap.add_argument('--shape', default='ball')
    ap.add_argument('--tilt', type=float, nargs=2, default=[30., 0.])
    ap.add_argument('--axes', type=float, nargs=3, default=[1., .8, .6])
    ap.add_argument('--R', type=float, default=.035)
    ap.add_argument('--cell', type=float, default=.0035)
    ap.add_argument('--fps', type=float, default=1000.)
    args = ap.parse_args()
    g = lattice_shape(args.shape, args.R, args.cell, args.axes, args.tilt)
    X = g['points']; n = len(X); px = 1e-3*(args.R/.035)
    for f in args.files:
        d = np.load(f)
        print(f.name)
        for obs in ('silhouette', 'texture', 'd3'):
            J = {k: (d['%s_%s_+1' % (obs, k)]-d['%s_%s_-1' % (obs, k)])/(2*STEP[k]) for k in PARAMS}
            for rate, dec in (('1000fps', 1), ('125fps', 8)):
                sl = slice(dec-1, None, dec)
                A = np.stack([J[k][sl].reshape(-1) for k in PARAMS if k != 'nu'], 1)
                jn = J['nu'][sl].reshape(-1)
                res = jn-A@np.linalg.lstsq(A, jn, rcond=None)[0]
                m = J['nu'][sl].shape[1]
                pf = np.sqrt((res.reshape(-1, m)**2).mean(1))*.1
                raw, un = np.sqrt((jn**2).mean())*.1, np.sqrt((res**2).mean())*.1
                print('   %-10s %-7s dnu 0.1: raw %.3f px, unexplained %.3f px (%.0f%%), peak %.3f px at %d ms' % (
                    obs, rate, raw, un, 100*un/max(raw, 1e-12), pf.max(), (int(np.argmax(pf))+1)*dec))
        # free-flight state over the last 40 ms
        F = len(d['d3_base'])

        def flight(key):
            d3 = d[key]
            (c1, R1), (c0, R0) = rigid_state(d3[F-1].reshape(n, 3)*px, X), rigid_state(d3[F-41].reshape(n, 3)*px, X)
            dR = R1@R0.T
            w = np.array([dR[2, 1]-dR[1, 2], dR[0, 2]-dR[2, 0], dR[1, 0]-dR[0, 1]])/2/(40/args.fps)
            zmax = max(rigid_state(d3[k].reshape(n, 3)*px, X)[0][2] for k in range(100, F, 10))
            return np.concatenate(((c1-c0)/(40/args.fps), w, [zmax]))
        b = flight('d3_base')
        print('   flight (last 40 ms): v = (%.3f, %.3f, %.3f) m/s, spin = (%.2f, %.2f, %.2f) rad/s, peak centroid height %.1f mm' % (*b[:6], 1e3*b[6]))
        for k in PARAMS:
            dp = (flight('d3_%s_+1' % k)-flight('d3_%s_-1' % k))/(2*STEP[k])*(.1 if k == 'nu' else STEP[k])
            print('   per %-18s dv = (%+.4f, %+.4f, %+.4f) m/s, dspin = (%+.3f, %+.3f, %+.3f) rad/s, dheight %+.2f mm' % (
                ('nu +0.1' if k == 'nu' else '%s +%g' % (k, STEP[k])), *dp[:6], 1e3*dp[6]))


if __name__ == '__main__':
    main()
