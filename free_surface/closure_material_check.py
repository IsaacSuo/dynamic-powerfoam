"""ClosureWarp.set_material (and MidpointStep.update_material) against a rebuild at the new material (2026-10-26).
  static   W1 built at (E1, nu1, beta1), set_material(E2, nu2, beta2); W2 built at (E2, nu2, beta2): energy, its terms and
           the gradient at four states (near rest; stretched 5 %; stretched and rough; pressed into the ground with the
           glue mu 0.3 and a held force), the same viscous references, ball data and held forces in both
  dynamic  hertz_drop.py's drop (2 cm, from rest) for --steps steps with each, the same rest blocks (W1's, scaled by
           update_material): final positions and gradient evaluations
  PYTHONPATH=<main repo>:<worktree> python free_surface/closure_material_check.py"""
import argparse
import math
from pathlib import Path

import numpy as np
import torch

from experiments.pacnerf_dem_forward import build
from free_surface.closure_drop import colouring, probe_blocks
from free_surface.closure_warp import ClosureCG, ClosureWarp, MidpointStep
from free_surface.sphere_closure import ClosureTerms

f64, dev = torch.float64, 'cuda'


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument('--out', type=Path, default=Path('output/round01_pruned_exact'))
    a_.add_argument('--m1', type=float, nargs=3, default=[23008.38, .3008, .001], help='E, nu, beta built')
    a_.add_argument('--m2', type=float, nargs=3, default=[11504.19, .40, .003], help='E, nu, beta set in place / rebuilt')
    a_.add_argument('--mu', type=float, default=.3)
    a_.add_argument('--dt', type=float, default=1/1200)
    a_.add_argument('--steps', type=int, default=160)
    args = a_.parse_args()
    cells = np.load(args.out/'dem_exact_single'/'cells.npz')
    g = dict(np.load(args.out/'dem_exact_single'/'geometry.npz'))
    bsk = dict(np.load(args.out/'boundary_skin.npz'))
    p, r = cells['points'], cells['radii']
    dt = args.dt
    boc = torch.as_tensor(np.asarray(g['body_of_cell']), device=dev).long()
    site = torch.as_tensor(bsk['patch_site'], device=dev).long()
    cbody = boc[site]; Rb = torch.as_tensor(r, dtype=f64, device=dev)[site]

    def make(E, nu, beta):
        m = build(g, E, nu, dev, 'noslip', .25, .3, f64, kinematics='exact', nodalvol=True, nonlinear=False, ogden_alpha=2.)
        st = ClosureTerms(m, g, p, r, bsk)
        W = ClosureWarp(st)
        W.set_viscosity(beta*W.mu, beta*(W.lam+2*W.mu/3), dt)
        W.visc_weight = 2.; W.total_mode()
        W.set_ball(cbody, Rb, (0., 1., 0.), 0., args.mu)
        W.prepare()
        return m, st, W
    m, st, W1 = make(*args.m1)
    W1.set_material(E=args.m2[0], nu=args.m2[1], beta=args.m2[2])
    _, _, W2 = make(*args.m2)
    X = st.X; n = W1.n
    up = torch.tensor([0., 1., 0.], dtype=f64, device=dev)
    print('in place: mu %.10e lam %.10e E* %.10e G* %.10e beta %.4g; rebuilt: mu %.10e lam %.10e E* %.10e G* %.10e beta %.4g' % (
        W1.mu, W1.lam, W1.Es, W1.Gs, W1.smu, W2.mu, W2.lam, W2.Es, W2.Gs, W2.smu), flush=True)

    # static
    torch.manual_seed(0)
    c = X.mean(0)
    T = torch.eye(3, dtype=f64, device=dev)+.05*torch.randn(3, 3, dtype=f64, device=dev)
    low = lambda x_: float(((x_[cbody]*up).sum(1)-Rb).min())
    xs = c+(X-c)@T.T
    states = (('near rest', X+1e-5*torch.randn_like(X)), ('stretched', xs), ('stretched, rough', xs+3e-4*torch.randn_like(X)),
              ('pressed, glued', xs+(-.004-low(xs))*up))
    xref = X+2e-5*torch.randn_like(X)
    for W in (W1, W2):
        W.visc_ref(xref)
        W.tball[0], W.tball[1] = 2., .5
    lamt = 1e-4*torch.randn(len(cbody), 3, dtype=f64, device=dev)
    for name, x in states:
        xn = x+.001*up
        out = []
        for W in (W1, W2):
            xc = xn[cbody]; dn = (Rb-(xc*up).sum(1)).clamp(min=0.)
            W.tbxn.copy_(xc); W.tbkt.copy_(8*W.Gs*torch.sqrt(Rb*dn)); W.tbbound.copy_(args.mu*4/3*W.Es*torch.sqrt(Rb*dn)*dn); W.tblamt.copy_(lamt)
            W.cold()
            for _ in range(4):
                l_, gx = W.grad(x)
            out.append((float(l_), gx, W.tsum.clone(), int((W.tbdelta > 0).sum())))
        (l1, g1, t1, k1), (l2, g2, t2, k2) = out
        tr = ((t1-t2).abs()/t2.abs().clamp(min=1e-300)).cpu().numpy()
        print('%-17s energy %.10e / %.10e (rel %.1e); |dE/dx| rel diff %.1e; terms rel diff max %.1e (term %d); balls touching %d / %d' % (
            name, l1, l2, abs(l1-l2)/abs(l2), float((g1-g2).norm()/g2.norm()), tr[t2.abs().cpu().numpy() > 0].max(),
            int(np.argmax(np.where(t2.abs().cpu().numpy() > 0, tr, 0))), k1, k2), flush=True)

    # dynamic
    mass = m.mass.to(f64); M3 = mass[:, None]
    fg = -9.8*M3*up; ref = float(fg.norm())
    color = colouring(st.i.cpu().numpy(), st.j.cpu().numpy(), n)
    Kbb = probe_blocks(W1, X, color)                                             # (at the in-place material: both use these)
    x0 = X+(.02-low(X))*up
    res = []
    for W in (W1, W2):
        W.cold(); W.tblamt.zero_()
        W.visc_ref(x0)
        CG = ClosureCG(W, mass, dt, per=2, product_iters=1)
        ST = MidpointStep(W, CG, mass, Kbb, fg, dt, args.mu)
        ST.set_state(x0, torch.zeros_like(x0), fg/M3)
        prods, nw = [], []
        for s in range(1, args.steps+1):
            pr, its, g2 = ST.step(1e-3*ref, lambda g2_: .5e-3*ref)
            if not (math.isfinite(g2) and math.sqrt(g2) <= 1e-3*ref) or its > 3:
                print('  %s: step %d: Newton %d, |grad| / weight %.2e, gradient evaluations %d, balls touching %d' % (
                    'in place' if W is W1 else 'rebuilt', s, its, math.sqrt(g2)/ref if math.isfinite(g2) else g2, pr, int((W.tbdelta > 0).sum())), flush=True)
            if not (math.isfinite(g2) and math.sqrt(g2) <= 1e-3*ref):
                break
            ST.end(); prods.append(pr); nw.append(its)
        res.append((ST.tx.clone(), np.array(prods), int((W.tbdelta > 0).sum())))
        print('  %s: %d steps, Newton per step max %d' % ('in place' if W is W1 else 'rebuilt', len(prods), max(nw)), flush=True)
    (xa, pa, ka), (xb, pb, kb) = res
    if len(pa) != len(pb):
        print('dynamic: the runs stopped at different steps (%d, %d)' % (len(pa), len(pb)))
        return
    print('dynamic, %d steps (landing near 77): final x max diff %.2e m; gradient evaluations %d / %d (per step differing: %d); balls touching %d / %d' % (
        args.steps, float((xa-xb).abs().max()), pa.sum(), pb.sum(), int((pa != pb).sum()), ka, kb), flush=True)


if __name__ == '__main__':
    main()
