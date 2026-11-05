"""the material-point glue's gradient against central differences of the energy: lion_inverse.forward (PAC-NeRF elastic
scene 0, beta 0.003, glue at 'site' or 'material' by argv[1]) stopped inside the impact (frame 6 + 10 steps), the energy
W.grad(x)[0] around the step's accepted midpoint with that step's ball data: directional derivatives along random
directions on all sites and on the touching cells only, against g . v; also the tangential force's size there"""
import sys
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import free_surface.lion_inverse as li
li.set_scene('pacnerf_e0'); li.GLUE = sys.argv[1]
if len(sys.argv) > 2:
    li.SHAPE = sys.argv[2]
S0 = 6*li.SPF+10


class Stop(Exception):
    pass


def probe(s, W, ST, cbody, Rb):
    if s < S0:
        return
    n = W.n; x0 = ST.tz.clone()
    on = W.tbfn > 0; B = cbody[on]
    print('glue %s, ball %s, step %d: touching balls %d, |F_n| sum %.4g, |f_t| sum %.4g' % (W.glue_point, W.ball_shape, s, int(on.sum()), float(W.tbfn.sum()), float(W.tbff.norm(dim=1).sum())), flush=True)
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
            for h in (1e-6, 1e-7):
                fd = (E(x0+h*v)-E(x0-h*v))/(2*h); an = float((g*v).sum())
                print('  %-15s dir %d h %.0e: difference %+.8e, gradient %+.8e, relative error %.2e' % (nm, k, h, fd, an, abs(fd-an)/max(abs(an), 1e-300)), flush=True)
    raise Stop


try:
    li.forward(np.array([10.04361, .30079413, -5.809143]), 7, verbose=False, on_step=probe)
except Stop:
    pass
