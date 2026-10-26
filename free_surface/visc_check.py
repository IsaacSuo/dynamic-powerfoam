"""Checks of the viscosity on the elasticity's own measures (2026-10-24, SkinWarp.set_viscosity / visc_ref / visc_hvp):
(0) with the reference at rest (all measures 0 there), Et = beta E at any state when eta = beta mu, lam_v = beta lam
    (every kernel's viscous counterpart at once);
(1) at a deformed state (bodies and skin moved at random by 0.3 mm, ~2 % strain; 3 mm turns cells over, where the
    differences below lose their meaning) as the reference: rigid translation and rotation of all
    velocities dissipate nothing, R = 1/2 w . D w;
(2) Et's gradient (the loss in visc_mode, hand adjoints included) against central differences of Et, with the
    reference at another state;
(3) D symmetric, R >= 0 for random velocities;
(4) its size: a uniform state x = A X, y = A Y moving at w = L x has the continuum's R = det A sum V (eta d : d +
    lam_v (tr d)^2 / 2), d = sym(L), at rest (and A near I: the measures' rates there differ from d at the order of the
    strain); the elastic energy of a small uniform strain against the continuum's alongside.
  python free_surface/visc_check.py --out output/round01_pruned_exact --beta 0.005"""
import torch

from free_surface.skin_contact_check import f64, dev
from free_surface.skin_dynamics_check import build_model, parser


def main():
    args = parser().parse_args()
    b = build_model(args)
    st, W = b['st'], b['W']
    X, Y = st.X, st.Y
    gen = torch.Generator(device=dev).manual_seed(0)
    rn = lambda sh: torch.randn(sh, generator=gen, dtype=f64, device=dev)
    E = lambda x_, y_: W.energy(x_, y_)
    def R(x_, y_, wx, wy):
        Dx, Dy = W.visc_hvp(x_, y_, wx, wy)
        return .5*float((wx*Dx).sum()+(wy*Dy).sum())
    eta, lamv = W.eta, W.lamv
    print('viscosity: eta %.5g, lam_v %.5g (eta / mu %.5g, lam_v / lam %.5g)' % (eta, lamv, W.smu, W.slam), flush=True)
    x, y = X+3e-4*rn(X.shape), Y+3e-4*rn(Y.shape)                                # ~2 % strain (3 mm turns cells over:
    print('random state: det F_B min %.3f, max %.3f' % tuple(float(f) for f in torch.linalg.det(st.F(x, y)).aminmax()), flush=True)   # shortest bond 2.8 mm)
    # (0)
    W.visc_ref(X, Y)
    t = E(x, y)
    print('(0) reference at rest, a random state: Et %.10e, beta E %.10e, relative %.1e (single ratio: %s); terms 0-5 %s' % (
        float(t[7]), W.smu*float(t[:6].sum()), abs(float(t[7])-W.smu*float(t[:6].sum()))/float(t[7]), abs(W.smu-W.slam) < 1e-12*W.smu,
        ' '.join('%.3e' % float(a) for a in t[:6])), flush=True)
    # (1)
    W.visc_ref(x, y)
    v, yd = rn(X.shape), rn(Y.shape)
    Rv = R(x, y, v, yd)
    u = torch.tensor([.2, .1, -.4], dtype=f64, device=dev); w = torch.tensor([.3, -.7, .5], dtype=f64, device=dev); c = x.mean(0)
    Rt = R(x, y, u.expand_as(x).clone(), u.expand_as(y).clone())
    Rr = R(x, y, torch.linalg.cross(w.expand_as(x), x-c), torch.linalg.cross(w.expand_as(y), y-c))
    print('(1) R of a random velocity %.4e; of a rigid translation %.1e, of a rigid rotation %.1e (relative %.1e, %.1e)' % (Rv, Rt, Rr, Rt/Rv, Rr/Rv), flush=True)
    # (2) Et's gradient at (x2, y2) with the reference at (x, y)
    x2, y2 = x+1e-4*rn(X.shape), y+1e-4*rn(Y.shape)
    W.visc_mode(); _, gx, gy = W.grad(x2, y2); W.total_mode()
    for j in range(3):
        dx, dy = (rn(X.shape), torch.zeros_like(Y)) if j == 0 else (torch.zeros_like(X), rn(Y.shape)) if j == 1 else (rn(X.shape), rn(Y.shape))
        h = 1e-6
        fd = (float(E(x2+h*dx, y2+h*dy)[7])-float(E(x2-h*dx, y2-h*dy)[7]))/(2*h)
        an = float((gx*dx).sum()+(gy*dy).sum())
        print('(2) direction %s: Et differences %.8e, gradient %.8e, relative %.1e' % (('x', 'y', 'x and y')[j], fd, an, abs(fd-an)/abs(an)), flush=True)
    # (3)
    u1, u2 = (rn(X.shape), rn(Y.shape)), (rn(X.shape), rn(Y.shape))
    D1, D2 = W.visc_hvp(x, y, *u1), W.visc_hvp(x, y, *u2)
    a12 = float((u2[0]*D1[0]).sum()+(u2[1]*D1[1]).sum()); a21 = float((u1[0]*D2[0]).sum()+(u1[1]*D2[1]).sum())
    Rs = [R(x, y, rn(X.shape), rn(Y.shape)) for _ in range(20)]
    print('(3) u2 . D u1 %.8e, u1 . D u2 %.8e (relative %.1e); R of 20 random velocities: min %.3e (>= 0: %s)' % (
        a12, a21, abs(a12-a21)/abs(a12), min(Rs), min(Rs) >= 0), flush=True)
    # (4)
    I3 = torch.eye(3, dtype=f64, device=dev)
    V = float(st.V.sum())
    L = rn((3, 3)); d = .5*(L+L.T)
    for name, A in (('rest', I3), ('A = I + 0.01 randn', I3+.01*rn((3, 3)))):
        xa, ya = X@A.T, Y@A.T
        W.visc_ref(xa, ya)
        Ru = R(xa, ya, xa@L.T, ya@L.T)
        Rc = float(torch.det(A))*V*(eta*float((d*d).sum())+.5*lamv*float(torch.trace(d))**2)
        print('(4) uniform strain rate, %s: R %.8e, continuum %.8e, relative %+.2e' % (name, Ru, Rc, (Ru-Rc)/Rc), flush=True)
    h = 1e-5
    Ee = float(E(X+h*X@L.T, Y+h*Y@L.T)[:6].sum())
    Ec = V*(W.mu*float((d*d).sum())+.5*W.lam*float(torch.trace(d))**2)*h*h
    print('(4) elastic, uniform strain h sym(L), h %.0e: E %.8e, continuum %.8e, relative %+.2e' % (h, Ee, Ec, (Ee-Ec)/Ec), flush=True)


if __name__ == '__main__':
    main()
