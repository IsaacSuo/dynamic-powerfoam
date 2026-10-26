"""Hessian-vector products in y by differences of the Warp gradient (one-sided, central; step h, direction scaled to
largest entry 1) against torch's exact ones (jvp of the gradient), 2026-10-23. States: the dynamics start (solved skin)
and its cells' affine skin (off equilibrium). Directions: 3 random, and the first preconditioned residual there.
  python free_surface/skin_hvp_check.py --out output/round01_pruned_exact --fem output/round01_exact/fem_modes_h0.008.npz"""
import numpy as np
import torch

from free_surface.skin_dynamics_check import build, parser
from free_surface.skin_solve import SkinSolver
from free_surface.skin_contact_check import f64, dev


def main():
    a = parser(); a.add_argument('--h', type=float, nargs='+', default=[1e-7, 1e-6])
    args = a.parse_args()
    b = build(args)
    st, W, S, x0, y0 = b['st'], b['W'], b['S'], b['x0'], b['y0']
    g = dict(np.load(b['out']/'dem_exact_single'/'geometry.npz'))
    T = SkinSolver(st, dict(np.load(b['out']/'boundary_skin.npz')), g['body_of_cell'])          # torch, exact products
    gen = torch.Generator(device=dev).manual_seed(0)
    pt = T.preconditioner(b['blocks'], T.bodies)                                                # torch, against the Warp one
    Rb = torch.linalg.qr(torch.randn(st.n, 3, 3, generator=gen, dtype=f64, device=dev))[0]
    rv = torch.randn(2, 3*st.npt, generator=gen, dtype=f64, device=dev)
    zt, zw = pt(Rb)(rv), b['prec'](Rb)(rv)
    print('preconditioner, Warp vs torch (random rotations): |diff| / |z| %.1e' % float((zw-zt).norm()/zt.norm()), flush=True)
    for nm, y in (('start (solved skin)', y0), ('affine skin', S.affine_skin(x0, y0))):
        e, gy, aux = S.eg(x0, y)
        prec = b['prec'](aux[1])
        V = torch.cat((torch.randn(3, 3*st.npt, generator=gen, dtype=f64, device=dev), prec(-gy[None])))
        Hx = T.hvp(x0, y)(V)
        for h in args.h:
            for mode in ('forward', 'central'):
                Hf = W.hvp_fd(x0, y, V, h, gy if mode == 'forward' else None)
                err = ((Hf-Hx).norm(dim=1)/Hx.norm(dim=1)).tolist()
                print('%-20s h %.0e %-8s |H_fd v - H v| / |H v|: random %s, preconditioned residual %.1e' % (
                    nm, h, mode, ' '.join('%.1e' % e_ for e_ in err[:3]), err[3]), flush=True)


if __name__ == '__main__':
    main()
