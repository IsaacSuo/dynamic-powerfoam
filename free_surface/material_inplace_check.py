"""SkinWarp.set_material (2026-10-25): the material changed in place (no rebuild, the graphs kept). --mode baseline: the
gradient (loss, dE/dx, dE/dy, terms) at a saved state written before the code change; --mode check: (1) the same
gradient with the new code at the default material against the baseline (the kernels now read the scalars from a device
array: equal to the rounding of the atomic sums), (2) a model built at --E2 / --nu2 / --beta (rebuilt) against the
default model set in place to them, at the same state and references.
  python free_surface/material_inplace_check.py --out output/round01_pruned_exact --dt 2.0833333e-4 --beta 0.01 --integrator midpoint --mode baseline"""
from pathlib import Path

import numpy as np
import torch

from free_surface.skin_contact_check import f64, dev
from free_surface.skin_dynamics_check import build_model, parser
from free_surface.tangent import NAMES


def state(W, out):
    z = np.load(Path(out)/'skin_drop_state_step900.npz')
    for k in NAMES:
        W.tref[k].copy_(torch.as_tensor(z['ref_'+k], device=dev))
    W.tlam.copy_(torch.as_tensor(z['lam'], device=dev)); W.tact.copy_(torch.as_tensor(z['act'], device=dev).to(W.tact.dtype))
    W.tgoff.copy_(torch.as_tensor(z['goff'], device=dev)); W.trho.copy_(torch.as_tensor(z['rho'], device=dev))
    W.tcpar[0], W.tcpar[1], W.tcpar[2] = 2., .5, 1.
    W.visc_weight = 2.; W.total_mode()
    return torch.as_tensor(z['xm1'], dtype=f64, device=dev), torch.as_tensor(z['ym1'], dtype=f64, device=dev)


def grad_all(W, x, y):
    e, gx, gy = W.grad(x, y)
    return dict(e=e.cpu().numpy(), gx=gx.cpu().numpy(), gy=gy.cpu().numpy(), terms=W.tsum.cpu().numpy())


def main():
    a = parser()
    a.add_argument('--mode', choices=('baseline', 'check'), required=True)
    a.add_argument('--E2', type=float, default=23008.38*1.7)
    a.add_argument('--nu2', type=float, default=.42)
    a.add_argument('--beta2', type=float, default=.003)
    args = a.parse_args()
    f = Path(args.out)/'material_inplace_baseline.npz'
    b = build_model(args, contact=dict(normal=(0., 1., 0.), c0=0., points=False, shift=True))
    W = b['W']
    x, y = state(W, args.out)
    g = grad_all(W, x, y)
    if args.mode == 'baseline':
        np.savez(f, **g)
        print('baseline written: loss %.12e, terms %s' % (float(g['e']), np.array2string(g['terms'], precision=6)), flush=True)
        return
    z = np.load(f)
    rel = lambda k: float(np.abs(g[k]-z[k]).max()/max(np.abs(z[k]).max(), 1e-300))
    print('(1) new code, default material, against the baseline: loss %.1e, dE/dx %.1e, dE/dy %.1e, terms %.1e (max relative)' % (
        rel('e'), rel('gx'), rel('gy'), rel('terms')), flush=True)
    W.set_material(E=args.E2, nu=args.nu2, beta=args.beta2)
    g_in = grad_all(W, x, y)
    W.set_material(E=args.E, nu=args.nu, beta=args.beta)
    g_back = grad_all(W, x, y)
    print('    set to (E2, nu2, beta2) and back: loss %.1e, dE/dx %.1e, dE/dy %.1e' % (
        *(float(np.abs(g_back[k]-g[k]).max()/max(np.abs(g[k]).max(), 1e-300)) for k in ('e', 'gx', 'gy')),), flush=True)
    del b, W
    torch.cuda.empty_cache()
    args.E, args.nu, args.beta = args.E2, args.nu2, args.beta2
    b2 = build_model(args, contact=dict(normal=(0., 1., 0.), c0=0., points=False, shift=True))
    W2 = b2['W']
    x, y = state(W2, args.out)
    g2 = grad_all(W2, x, y)
    r2 = lambda k: float(np.abs(g_in[k]-g2[k]).max()/max(np.abs(g2[k]).max(), 1e-300))
    print('(2) in place at E %.6g, nu %.4g, beta %.4g against a model built there: loss %.1e, dE/dx %.1e, dE/dy %.1e, terms %s' % (
        args.E2, args.nu2, args.beta2, r2('e'), r2('gx'), r2('gy'),
        np.array2string(np.abs(g_in['terms']-g2['terms'])/np.maximum(np.abs(g2['terms']), 1e-300), precision=1)), flush=True)


if __name__ == '__main__':
    main()
