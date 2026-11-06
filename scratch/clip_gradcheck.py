"""the touching caps' clip (ClosureWarp.set_cap_clip): its gradient against central differences of the energy, inside the
impact of lion_inverse.forward (PAC-NeRF elastic scene 0, beta 0.003, --cap-clip w = argv[1], --cap-clip-pen argv[2]);
the step's glue data fixed. Directions over all sites, and over the clipped cells only"""
import sys
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import free_surface.lion_inverse as li
li.set_scene('pacnerf_e0'); li.CLIP = float(sys.argv[1]); li.CLIPP = float(sys.argv[2]) if len(sys.argv) > 2 else 0.
S0 = int(sys.argv[3]) if len(sys.argv) > 3 else 6*li.SPF+10


class Stop(Exception):
    pass


def probe(s, W, ST, cbody, Rb):
    if s < S0:
        return
    n = W.n; x0 = ST.tz.clone()
    E = lambda x: float(W.grad(x)[0])
    g = W.grad(x0)[1].clone()
    on = (W.tsps-W.tsclip > 1e-3) & (W.tsps > 0)
    B = torch.nonzero(on)[:, 0]
    rat = W.tsclip[on]/W.tsps[on]
    print('step %d: clipped caps %d (s / s_ps median %.3f, min %.3f), glued points %d, clip normal force %.4g' % (
        s, len(B), float(rat.median()), float(rat.min()), int((W.tcq['bound'] > 0).sum()), float(W.tcq['cnq'].clamp(min=0).sum())), flush=True)
    gen = torch.Generator(device='cuda').manual_seed(0)
    for nm, mask in (('all sites', None), ('clipped cells', B)):
        for k in range(3):
            v = torch.zeros_like(x0)
            if mask is None:
                v[:n] = torch.randn(n, 3, dtype=x0.dtype, device='cuda', generator=gen)
            else:
                v[mask] = torch.randn(len(mask), 3, dtype=x0.dtype, device='cuda', generator=gen)
            v /= v.abs().max()
            for h in (1e-6, 1e-7):
                fd = (E(x0+h*v)-E(x0-h*v))/(2*h); an = float((g*v).sum())
                print('  %-14s dir %d, h %.0e: difference %+.10e, gradient %+.10e, relative error %.2e' % (nm, k, h, fd, an, abs(fd-an)/max(abs(an), 1e-300)), flush=True)
    raise Stop


try:
    li.forward(np.array([10.04361, .30079413, -5.809143]), 8, verbose=True, on_step=probe)
except Stop:
    pass
