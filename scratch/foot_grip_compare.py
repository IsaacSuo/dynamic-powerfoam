"""mpm_foot_grip.py against dem_foot_grip.py, per foot: the braking force B = -T (per unit mass) at half frames, the area
the ground reaches (MPM: floor nodes with material; ours: glued cap points, rest area) and the area pressed, and over each
half-frame interval the change of B split into: held (the points / nodes held in both windows, their own change), new
(the ones taken on, their braking), lost (the braking the released ones carried away, subtracted). args: mpm.npz dem.npz"""
import sys
import numpy as np
FPS = 24
G = {'MPM': np.load(sys.argv[1])['grip'], 'ours': np.load(sys.argv[2])['grip']}
for foot, k_ in (('front', 0), ('back', 1)):
    print('==== %s foot' % foot)
    for nm, g in G.items():
        o = 1+k_*((g.shape[1]-1)//2)
        t = g[:, 0]*FPS
        print('  %s: frame | braking B, normal N | area reached/glued, pressed (cm^2)%s | interval to the next half frame: change of B = held + new - lost' % (
            nm, ', pressed caps' if nm == 'ours' else ''))
        tot = np.zeros(3)
        for f in np.arange(5., 9.01, .5):
            i = int(np.argmin(np.abs(t-f))); j = int(np.argmin(np.abs(t-(f+.5))))
            sl = slice(i+1, j+1)
            held, new, lost = -g[sl, o+4].sum(), -g[sl, o+5].sum(), g[sl, o+6].sum()
            if 6. <= f < 8.:
                tot += (held, new, lost)
            caps = ', %3d' % g[i, o+7] if nm == 'ours' else ''
            print('     %4.1f | %6.2f %6.1f | %6.1f %6.1f%s | %+6.2f = %+6.2f %+6.2f %+6.2f' % (
                f, -g[i, o], g[i, o+1], 1e4*g[i, o+2], 1e4*g[i, o+3], caps, -g[j, o]+g[i, o], held, new, -lost))
        print('     frames 6 -> 8: held %+.2f, new %+.2f, lost %+.2f (sum %+.2f)' % (tot[0], tot[1], -tot[2], tot[0]+tot[1]-tot[2]))
