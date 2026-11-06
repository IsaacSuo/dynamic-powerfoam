"""the cap contact with the band (--cap-band delta = argv[2], K ratio argv[3]): its gradient against central differences of the energy, inside the impact (frame 6 + 10 steps) of
lion_inverse.forward (PAC-NeRF elastic scene 0, beta 0.003, --cap-contact alpha = argv[1]); the step's cap data fixed"""
import sys
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import free_surface.lion_inverse as li
li.set_scene('pacnerf_e0'); li.CAPC = float(sys.argv[1]); li.CAPB = float(sys.argv[2]); li.CAPBK = float(sys.argv[3])
S0 = 6*li.SPF+10


class Stop(Exception):
    pass


def probe(s, W, ST, cbody, Rb):
    if s < S0:
        return
    n = W.n; x0 = ST.tz.clone(); c = W.tcq
    on = c['fnq'] > 0; B = torch.unique(c['cb'][on].long())
    print('step %d: cap points touching %d on %d cells, normal force %.4g, |tangential| %.4g, deepest %.2f mm, glued points %d' % (
        s, int(on.sum()), len(B), float(c['fnq'].sum()), float(c['ffq'].norm(dim=1).sum()), 1e3*float(c['dq'].max()), int((c['bound'] > 0).sum())), flush=True)
    E = lambda x: float(W.grad(x)[0])
    g = W.grad(x0)[1].clone()
    gen = torch.Generator(device='cuda').manual_seed(0)
    for nm, mask in (('all sites', None), ('touching cells', B)):
        for k in range(3):
            v = torch.zeros_like(x0)
            if mask is None:
                v[:n] = torch.randn(n, 3, dtype=x0.dtype, device='cuda', generator=gen)
            else:
                v[mask] = torch.randn(len(mask), 3, dtype=x0.dtype, device='cuda', generator=gen)
            v /= v.abs().max()
            h = 1e-7
            fd = (E(x0+h*v)-E(x0-h*v))/(2*h); an = float((g*v).sum())
            print('  %-15s dir %d: difference %+.8e, gradient %+.8e, relative error %.2e' % (nm, k, fd, an, abs(fd-an)/max(abs(an), 1e-300)), flush=True)
    raise Stop


try:
    li.forward(np.array([10.04361, .30079413, -5.809143]), 7, verbose=False, on_step=probe)
except Stop:
    pass
