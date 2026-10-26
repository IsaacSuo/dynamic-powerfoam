"""Is the energy continuous along the anomaly's step (2026-10-23)? Warp energy terms on straight lines: the skin from the
previous to the step's at the new positions, and the bodies from the old to the new positions with the previous skin;
the largest jump between neighbouring samples against the smooth variation (second differences), per term.
  python free_surface/skin_energy_scan.py --out output/round01_pruned_exact --dt 2.0833333e-4"""
import numpy as np
import torch

from free_surface.skin_contact_check import f64, dev
from free_surface.skin_dynamics_check import build_model, parser

NAMES = ('bonds', 'volume', 'outer_stretch', 'outer_cross', 'outer_skin', 'outer_lateral', 'contact')


def scan(W, xs, ys, n):
    E = []
    for k in range(n+1):
        s = k/n
        E.append(W.energy(xs(s), ys(s)).detach().cpu().numpy())
    return np.array(E)                                                              # (n + 1, 7)


def main():
    a = parser(); a.add_argument('--n', type=int, default=2000)
    args = a.parse_args()
    b = build_model(args, contact=dict(normal=(0., 1., 0.), c0=0.))
    W = b['W']
    z = np.load(b['out']/'skin_drop_anomaly.npz')
    T = lambda a_: torch.as_tensor(a_, dtype=f64, device=dev)
    x1, v1, y1, y0 = T(z['x']), T(z['v']), T(z['y']), T(z['yp'])
    x0 = x1-args.dt*v1
    W.tlam.copy_(T(z['lam'])); W.trho.copy_(T(z['rho']))
    for nm, xs, ys in (('skin y799 -> y800 at x800', lambda s: x1, lambda s: y0+s*(y1-y0)),
                       ('bodies x799 -> x800 with y799', lambda s: x0+s*(x1-x0), lambda s: y0)):
        E = scan(W, xs, ys, args.n)
        d1 = np.diff(E, axis=0); d2 = np.diff(E, 2, axis=0)
        print('%s: total %.9e -> %.9e' % (nm, E[0].sum(), E[-1].sum()), flush=True)
        for t, k in enumerate(NAMES):
            j = int(np.abs(d1[:, t]).argmax())
            print('  %-14s %.6e -> %.6e; largest step between samples %.3e at s = %.4f (median step %.1e); largest |second difference| %.3e (median %.1e)' % (
                k, E[0, t], E[-1, t], d1[j, t], (j+.5)/args.n, np.median(np.abs(d1[:, t])), np.abs(d2[:, t]).max(), np.median(np.abs(d2[:, t]))), flush=True)


if __name__ == '__main__':
    main()
