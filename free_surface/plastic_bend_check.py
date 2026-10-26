"""The bonds' plastic rotation (plastic.Plasticity.thp, 10-26) on the part: a compatible, isochoric plastic bending
field u = kappa (x y, -y^2 / 4 - x^2 / 2 + z^2 / 4, -y z / 2) (X about the centroid), eps_p = sym grad u = kappa y
diag(1, -1/2, -1/2), omega_xy = kappa x: each cell's eps_p set to it, the body placed at R0 (X + u(X)) + t (random R0,
the skin from Y + u(Y), solved). The energy on the plastic references with thp, without thp, and on the elastic ones;
at two kappa: with thp it should be second order in u's own error (~ kappa^4), without ~ kappa^2"""
import math

import numpy as np
import torch

from free_surface.plastic import Plasticity
from free_surface.plastic_part_check import rotation
from free_surface.skin_contact_check import f64, dev
from free_surface.skin_dynamics_check import build, parser


def main():
    args = parser().parse_args(['--case', 'part', '--integrator', 'midpoint'])
    b = build(args)
    st, W, S, prec = b['st'], b['W'], b['S'], b['prec']
    X, Y = st.X, st.Y
    xc = X.mean(0)
    pl = Plasticity(W, st, math.inf)
    rng = np.random.default_rng(5)
    R0, t0 = rotation(rng), torch.tensor([.01, -.02, .015], dtype=f64, device=dev)
    ymax = float((X[:, 1]-xc[1]).abs().max())
    D = torch.diag(torch.tensor([1., -.5, -.5], dtype=f64, device=dev))

    def field(P, kap):
        x_, y_, z_ = (P-xc).unbind(1)
        return torch.stack((kap*x_*y_, -kap*y_**2/4-kap*x_**2/2+kap*z_**2/4, -kap*y_*z_/2), 1)
    print('the part: %d cells; |y| max %.1f mm' % (W.n, 1e3*ymax), flush=True)
    out = []
    for strain in (.02, .01):
        kap = strain/ymax
        eps = kap*(X[:, 1]-xc[1])[:, None, None]*D
        x = (X+field(X, kap))@R0.T+t0
        y = (Y+field(Y, kap))@R0.T+t0
        for _ in range(3):
            y = S.affine_skin(x, y)
        e = {}
        for name, rot, grd, eps_ in (('plastic, thp + branch', True, True, eps), ('plastic, thp', True, False, eps), ('plastic, first order', False, False, eps),
                                     ('elastic', True, True, torch.zeros_like(eps))):
            pl.rotation, pl.gradient = rot, grd
            pl.apply(eps_)
            ys, s_ = S.solve(x, y, prec, float(S.grad(x, y).norm()), tol=1e-10, maxit=3000)
            e[name] = pl.elastic(x, ys)
            e[name+' terms'] = W.energy_grad(x, ys)[0][:6].detach().cpu().numpy()
        pl.rotation, pl.gradient = True, True; pl.apply(eps)
        thn = W.tplas['thp'].norm(dim=1)
        out.append((strain, kap, e))
        print('largest plastic strain %.3f (kappa %.3f 1/m): energy / elastic: thp + branch %.2e, thp %.2e, first order %.2e (elastic %.4e); '
              '|thp| median %.2e, max %.2e rad (kappa x bond ~ %.2e)' % (
                  strain, kap, e['plastic, thp + branch']/e['elastic'], e['plastic, thp']/e['elastic'], e['plastic, first order']/e['elastic'], e['elastic'],
                  float(thn.median()), float(thn.max()), kap*float(st.dX.norm(dim=1).median())), flush=True)
        for name in ('plastic, thp + branch', 'plastic, thp', 'plastic, first order'):
            print('    %s: terms / the elastic total (bonds, volume, o1, o2, o3, lateral) %s' % (name, np.array2string(e[name+' terms']/e['elastic'], precision=2)), flush=True)
    for name in ('plastic, thp + branch', 'plastic, thp', 'plastic, first order', 'elastic'):
        print('  %s: energy ratio at strain 0.02 / 0.01 %.2f (kappa^2: 4, kappa^4: 16)' % (name, out[0][2][name]/out[1][2][name]), flush=True)


if __name__ == '__main__':
    main()
