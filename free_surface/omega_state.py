"""The fastest body mode with the skin held at a given state (2026-10-23): power iteration on M^-1 K_xx, products by
central differences of the Warp gradient in x (skin and contact multipliers held); omega dt against the symplectic
Euler bound 2. At rest and at the drop test's anomaly state (skin_drop_anomaly.npz).
  python free_surface/omega_state.py --out output/round01_pruned_exact --dt 2.0833333e-4"""
import numpy as np
import torch

from free_surface.skin_contact_check import f64, dev
from free_surface.skin_dynamics_check import build_model, parser


def omega_max(W, mass, x, y, iters=200, h=1e-7):
    v = torch.randn(x.shape, generator=torch.Generator(device=dev).manual_seed(0), dtype=f64, device=dev)
    om2 = 0.
    for _ in range(iters):
        v = v/v.abs().max()
        Kv = (W.grad(x+h*v, y)[1]-W.grad(x-h*v, y)[1])/(2*h)
        om2 = float((v*Kv).sum())/float((mass[:, None]*v*v).sum())
        v = Kv/mass[:, None]
    b = int(v.norm(dim=1).argmax())
    return om2**.5, b, float(v[b].norm()**2/(v*v).sum())


def omega_condensed(W, S, prec_make, mass, x, y, iters=40, h=1e-7, cg_tol=1e-6):
    """the fastest body mode with the skin free (and the contact multipliers held): power iteration on M^-1 K_c,
    K_c v = K_xx v + K_xy z, K_yy z = -K_yx v by PCG (preconditioner at (x, y) with the active contacts' penalties)"""
    e, gy, aux = S.eg(x, y)
    prec = S.prec_at(prec_make, x, y, aux)
    def gxy(dx, dy):
        s = max(float(dx.abs().max()), float(dy.abs().max()), 1e-300)
        _, ax, ay = W.grad(x+h*dx/s, y+h*dy/s); _, bx, by = W.grad(x-h*dx/s, y-h*dy/s)
        return (ax-bx)/(2*h)*s, (ay-by)/(2*h)*s
    v = torch.randn(x.shape, generator=torch.Generator(device=dev).manual_seed(1), dtype=f64, device=dev)
    om2, cgs = 0., []
    for _ in range(iters):
        v = v/v.abs().max()
        b = -gxy(v, torch.zeros_like(y))[1]
        z = torch.zeros_like(y); r = b.clone(); q = prec(r.reshape(1, -1)).view(-1, 3); d = q.clone(); rq = float((r*q).sum()); bn = float(b.norm()); k = 0
        while float(r.norm()) > cg_tol*bn and k < 1000:
            Ad = gxy(torch.zeros_like(x), d)[1]; k += 1
            al = rq/float((d*Ad).sum()); z = z+al*d; r = r-al*Ad
            q = prec(r.reshape(1, -1)).view(-1, 3); rq1 = float((r*q).sum()); d = q+(rq1/rq)*d; rq = rq1
        cgs.append(k)
        Kc = gxy(v, z)[0]
        om2 = float((v*Kc).sum())/float((mass[:, None]*v*v).sum())
        v = Kc/mass[:, None]
    B = int(v.norm(dim=1).argmax())
    return om2**.5, B, float(v[B].norm()**2/(v*v).sum()), int(np.median(cgs))


def main():
    a_ = parser(); a_.add_argument('--rho-scale', type=float, nargs='+', default=[1.]); a_.add_argument('--only-before', action='store_true')
    args = a_.parse_args()
    b = build_model(args, contact=dict(normal=(0., 1., 0.), c0=0.))
    W, mass, st = b['W'], b['mass'], b['st']
    z = np.load(b['out']/'skin_drop_anomaly.npz')
    T = lambda a_: torch.as_tensor(a_, dtype=f64, device=dev)
    states = [('rest', st.X, st.Y, None, None), ('anomaly step %d' % int(z['step']), T(z['x']), T(z['y']), T(z['lam']), T(z['rho']))]
    if 'prev_x' in z.files:
        states.append(('step before', T(z['prev_x']), T(z['prev_y']), T(z['prev_lam_used']), T(z['prev_rho'])))
    if args.only_before:
        states = [(('step before, rho x %g' % sc), *states[-1][1:4], states[-1][4]*sc) for sc in args.rho_scale]
    for nm, x, y, lam, rho in states:
        if lam is not None:
            W.tlam.copy_(lam); W.trho.copy_(rho)
        om, B, share = omega_max(W, mass, x, y)
        W.grad(x, y)
        print('%-18s: omega_max (skin held) %.3e rad/s, omega dt %.2f (stable < 2); mode %.2f on body %d (y %.4f m, log J %.3f, volume / median %.3f)' % (
            nm, om, om*args.dt, share, B, float(x[B, 1]), float(W.tlj[B]), float(st.V[B]/st.V.median())), flush=True)
        if not nm.startswith('anomaly'):
            om, B, share, cg = omega_condensed(W, b['S'], b['prec'], mass, x, y)
            print('%-18s: omega_max (skin free)  %.3e rad/s, omega dt %.2f (stable < 2); mode %.2f on body %d (y %.4f m); CG per iteration %d' % (
                nm, om, om*args.dt, share, B, float(x[B, 1]), cg), flush=True)


if __name__ == '__main__':
    main()
