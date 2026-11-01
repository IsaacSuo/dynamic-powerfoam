"""The elastic claim on a motion that is not uniform, surface cells included: nothing yields, the walls change. RingMoment: q, weights, outer layer and groups do not move, a new cut does not move the stored energy, the way back ends at zero energy in the springs.
  python free_surface/ring_moment_check.py --foam <dir with dem_exact_single> [--walls] [--amp 1.] [--steps 40]"""
import argparse
import math
from pathlib import Path

import numpy as np
import torch

from free_surface.ring_moment import RingMoment
from free_surface.ring_yield import RingYield

f64, dev = torch.float64, 'cuda'
a_ = argparse.ArgumentParser()
a_.add_argument('--foam', type=Path, required=True)
a_.add_argument('--walls', action='store_true', help='RingYield, the springs with the walls')
a_.add_argument('--amp', type=float, default=1.)
a_.add_argument('--steps', type=int, default=40)
a_.add_argument('--E', type=float, default=23.00838)
a_.add_argument('--nu', type=float, default=.3008)
args = a_.parse_args()
g = np.load(args.foam/'dem_exact_single'/'geometry.npz'); cl = np.load(args.foam/'dem_exact_single'/'cells.npz')
W = (RingYield if args.walls else RingMoment)(g['points'], cl['radii'], g['masses'], g['pairs'], g['areas'], g['contacts'], args.E, args.nu, 1e6, theta=.5, kappa=5., eps_cut=-1., ledger=True)
W.set_ball((0., 1., 0.), -100., 0.)
W.start(); W.prepare()
X = W.X.clone(); Xc = (W.V0[:, None]*X).sum(0)/W.V0.sum(); Rb = (X-Xc).norm(dim=1).max()


def motion(s):
    """a twist about y growing with height, a stretch and a bend: not uniform"""
    xi = (X-Xc)/Rb
    th_ = args.amp*s*.8*xi[:, 1]
    c_, s_ = torch.cos(th_), torch.sin(th_)
    y = xi[:, 1]*(1+.3*args.amp*s)
    a, b = xi[:, 0]*(1+.3*args.amp*s)**-.5, xi[:, 2]*(1+.3*args.amp*s)**-.5
    x_ = torch.stack([c_*a-s_*b+.3*args.amp*s*xi[:, 1]**2, y, s_*a+c_*b], 1)
    return Xc+Rb*x_


def snap():
    T, ms = W.tr, W._m_s
    return dict(key=W.tkey.clone(), qi=W.tr['qi'][:ms].clone(), qj=W.tr['qj'][:ms].clone(), wi=T['wi'][:ms].clone(), wj=T['wj'][:ms].clone(), Mo=T['Mo'].clone(), g=W._gid.clone())


path = list(np.linspace(0., 1., args.steps+1))+list(np.linspace(1., 0., args.steps+1)[1:])
x = motion(0.)
W.update_ul(x)
S0 = snap()
print('%d cells, %d springs; surface cells %d' % (W.n, W._m_s, int((W.tul['Mc'].abs().sum((1, 2)) > 0).sum())), flush=True)
born = closed = 0
for k in range(1, len(path)):
    xn = x
    x = motion(float(path[k]))
    E0 = sum(W.stored(xn)) if k > 1 else 0.
    W.update_ul(xn)
    E1 = sum(W.stored(xn)) if k > 1 else 0.
    z = .5*(xn+x)
    W.grad(z); W.reread(); W.grad(z); W.end_ul(.5)
    st = W.ul_stats; born += st.get('born', 0); closed += st.get('closed', 0)
    if k % (args.steps//4) == 0 or k == len(path)-1:
        S = snap()
        pos = torch.searchsorted(S['key'], S0['key']).clamp(max=len(S['key'])-1); hit = S['key'][pos] == S0['key']
        qs = S0['qi'].norm(dim=1)[hit]
        wb = float(S0['wi'].median())
        dq = float(((S['qi'][pos[hit]]-S0['qi'][hit]).norm(dim=1)/qs).max())
        dw = float((S['wi'][pos[hit]]-S0['wi'][hit]).abs().max())/wb
        dM = float((S['Mo']-S0['Mo']).abs().max())/max(float(S0['Mo'].abs().max()), 1e-300)
        print('s %.3f: rest springs still there %d of %d; max |dq| / |q| %.2e; max |dw| / w %.2e; max |dMo| %.2e; group changes %d; born so far %d, gone %d; new cut moved E by %.3e of E %.3e; yielding %d' % (path[k], int(hit.sum()), len(hit), dq, dw, dM, int((S['g'] != S0['g']).sum()), born, closed, E1-E0, E1, st.get('yielding', 0)), flush=True)
E_s, E_v = W.stored(x)
print('back at rest: springs and outer layer %.3e, volume %.3e; the largest on the way was of order %.3e' % (E_s, E_v, E1), flush=True)
