"""What jumped at the drop test's anomaly (skin_drop_anomaly.npz, 2026-10-23): the state before (x - dt v, the previous
skin) and after (x, the solved skin): energy terms, det F_B and the rotations' change per body, the bodies where they
change most (position, contact cell or not, gap).
  python free_surface/skin_drop_anomaly.py --out output/round01_pruned_exact --dt 2.0833333e-4"""
import numpy as np
import torch

from free_surface.skin_contact_check import f64, dev
from free_surface.skin_dynamics_check import build_model, parser
from experiments.foam_surface_completion import polar_newton


def main():
    args = parser().parse_args()
    b = build_model(args, contact=dict(normal=(0., 1., 0.), c0=0.))
    st, W = b['st'], b['W']
    z = np.load(b['out']/'skin_drop_anomaly.npz')
    T = lambda a: torch.as_tensor(a, dtype=f64, device=dev)
    x1, v1, y1, y0 = T(z['x']), T(z['v']), T(z['y']), T(z['yp'])
    x0 = x1-args.dt*v1
    names = ('bonds', 'volume', 'outer_stretch', 'outer_cross', 'outer_skin', 'outer_lateral')
    th = torch.zeros(st.n, 9, dtype=f64, device=dev)
    out = {}
    for nm, (x, y) in (('before', (x0, y0)), ('after', (x1, y1))):
        Tt = {k: float(v_) for k, v_ in st.energy_terms(th, x, y).items()}
        F = st.F(x, y); R = polar_newton(F); d = torch.linalg.det(F)
        W.grad(x, y)
        out[nm] = (F, R, d)
        print('%-6s: %s; Warp sum %.6e; det F_B min %.3e (body %d), bodies with det F <= 0: %d; |R^T R - I| max %.1e, det R min %.3f' % (
            nm, ' '.join('%s %.6e' % (k, Tt[k]) for k in names), float(W.tsum[:6].sum()), float(d.min()), int(d.argmin()), int((d <= 0).sum()),
            float((R.transpose(1, 2)@R-torch.eye(3, dtype=f64, device=dev)).abs().amax(dim=(1, 2)).max()), float(torch.linalg.det(R).min())), flush=True)
    F0, R0, d0 = out['before']; F1, R1, d1 = out['after']
    dR = (R1-R0).norm(dim=(1, 2)); dF = (F1-F0).norm(dim=(1, 2))
    cb = W.cg['body']; isc = torch.zeros(st.n, dtype=torch.bool, device=dev); isc[cb] = True
    gap = torch.full((st.n,), float('nan'), dtype=f64, device=dev); gap[cb] = T(z['gap'])
    lam = torch.zeros(st.n, dtype=f64, device=dev); lam[cb] = T(z['lam'])
    print('|R_after - R_before| per body: median %.2e, 99%% %.2e, max %.2e; |F_after - F_before| median %.2e, max %.2e' % (
        float(dR.median()), float(dR.quantile(.99)), float(dR.max()), float(dF.median()), float(dF.max())), flush=True)
    for B in torch.argsort(dR, descending=True)[:8].tolist():
        print('  body %d: |dR| %.2e, |dF| %.2e, det F %.3f -> %.3f, y %.4f m (lowest body y %.4f), surface %s, gap %.2e m, contact force %.2e; |F - I| %.3f' % (
            B, float(dR[B]), float(dF[B]), float(d0[B]), float(d1[B]), float(x1[B, 1]), float(x1[:, 1].min()), bool(isc[B]), float(gap[B]), float(lam[B]),
            float((F1[B]-torch.eye(3, dtype=f64, device=dev)).norm())), flush=True)
    # the skin at the new positions from the previous skin and from the step's own, both solved to 1e-8 (multipliers held)
    S, prec = b['S'], b['prec']
    W.tlam.copy_(T(z['lam'])); W.trho.copy_(T(z['rho']))
    ref = float((9.8*b['mass']).norm())
    B0 = int(torch.argmax(dR))
    sel = torch.zeros(st.npt, dtype=torch.bool, device=dev); sel[st.e_pid[st.e_body == B0]] = True
    res = {}
    for nm, ys in (('from the previous skin', y0), ('from the step\'s skin', y1)):
        yy, s_ = S.solve(x1, ys.clone(), prec, ref, tol=1e-8, maxit=3000)
        res[nm] = yy
        print('%-24s: energy (with the AL term) %.9e, elastic %.9e, residual %.1e, products %d, rejected %d, negative %d; F_B of body %d moved %.3e from the step\'s' % (
            nm, s_['energy'], float(s_['aux'][3][:6].sum()), s_['residual'], s_['products'], s_['rejected'], s_['negative'], B0,
            float((st.F(x1, yy)[B0]-F1[B0]).norm())), flush=True)
    a_, b_ = res['from the previous skin'], res['from the step\'s skin']
    print('the two solutions: |dy| max %.3e m (at body %d\'s skin points %.3e m), elsewhere %.3e m' % (
        float((a_-b_).norm(dim=1).max()), B0, float((a_-b_)[sel].norm(dim=1).max()), float((a_-b_)[~sel].norm(dim=1).max())), flush=True)


if __name__ == '__main__':
    main()
