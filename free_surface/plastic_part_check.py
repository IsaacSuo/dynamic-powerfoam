"""plastic.py on the part (2026-10-26). 1. The elastic model unchanged: the 10-23 start state's energy (skin_dynamics_check
--case part: 2.624810e-07). 2. The references exact for a uniform plastic flow: one eps_p everywhere (a = 0.1 along a
random axis), the body placed at x = R0 P_p X + t (random R0) and the skin solved: the energy and force there against
the same state on the elastic references. 3. The step-after return's error: the free part (no gravity, no contact)
started with an isochoric velocity gradient v = a diag(-1, 1/2, 1/2) (X - Xc) along x (a for an elastic peak strain of
--peak), yield strain --ey (sigma_y = E ey); first elastic (sigma_y infinite) at the coarsest dt, then elastoplastic at
each --dts: the mean plastic strain, the energy balance KE + E + W_p, the cells yielding"""
import argparse
import math
import time

import numpy as np
import torch

from free_surface.joint_step import MidpointIntegrator
from free_surface.plastic import Plasticity, sym_fun
from free_surface.skin_contact_check import f64, dev
from free_surface.skin_dynamics_check import build, parser


def rotation(rng):
    q = rng.standard_normal(4); q /= np.linalg.norm(q)
    w, x, y, z = q
    return torch.tensor([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)], [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                         [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]], dtype=f64, device=dev)


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument('--dts', type=float, nargs='+', default=[1/1200, 1/2400, 1/4800])
    a_.add_argument('--T', type=float, default=.08)
    a_.add_argument('--ey', type=float, default=.01, help='the yield strain sigma_y / E')
    a_.add_argument('--peak', type=float, default=.05, help='the elastic peak strain the start velocity is set for (energy estimate)')
    a_.add_argument('--tol', type=float, default=1e-6)
    a_.add_argument('--ret', choices=('after', 'mid'), default='mid', help='the return after the step, or within it (Plasticity.step)')
    a_.add_argument('--skip-static', action='store_true', help='checks 2 (the references) skipped')
    a_.add_argument('--criterion', choices=('virial', 'FB'), default='FB', help='the yield on the F_B stress or on -(1/V) dE/deps_p')
    a_.add_argument('--nested', action='store_true', help='the return\'s passes outside the step (step_fb), not inside its solve')
    a_.add_argument('--linear', choices=('off', 'rotation', 'full'), default='off', help='the linear reconstruction: off (piecewise constant, the default), thp only, thp and the branch gradient')
    a_.add_argument('--log-steps', action='store_true', help='each plastic step: passes (products, contact passes, force change / tol, yielding), inner iterations, |delta|, |eps|')
    pa = a_.parse_args()
    args = parser().parse_args(['--case', 'part', '--integrator', 'midpoint', '--dt', str(pa.dts[0])])
    b = build(args)                                                                 # (prints the start energy: check 1)
    st, W, S, prec, mass = b['st'], b['W'], b['S'], b['prec'], b['mass']
    X, Y = st.X, st.Y
    nu = W.lam/(2*(W.lam+W.mu)); E = 2*W.mu*(1+nu)
    print('check 1: the start energy above against 2.624810e-07 (10-23); E %.6g, nu %.4f per unit density' % (E, nu), flush=True)
    rng = np.random.default_rng(3)
    if not pa.skip_static:                                                        # 2. uniform plastic flow
        pl = Plasticity(W, st, math.inf)
        Q = rotation(rng)
        pl.eps = (Q@torch.diag(torch.tensor([-.1, .05, .05], dtype=f64, device=dev))@Q.T).expand(W.n, 3, 3).clone()
        pl.apply()
        P = sym_fun(pl.eps[:1], torch.exp)[0]
        R0, t0 = rotation(rng), torch.tensor([.01, -.02, .03], dtype=f64, device=dev)
        A = R0@P
        x, y = X@A.T+t0, Y@A.T+t0
        for _ in range(5):
            y = S.affine_skin(x, y)
        y, s_ = S.solve(x, y, prec, float(S.grad(x, y).norm()), tol=1e-10, maxit=2000)
        e_p = pl.elastic(x, y); _, gx, gy = W.grad(x, y)
        dmax = float((W.tbd-W.tplas['dp']).norm(dim=1).max())
        keep = pl.eps.clone(); pl.eps.zero_(); pl.apply()
        e_e = pl.elastic(x, y); _, gx0, _ = W.grad(x, y)
        pl.eps = keep; pl.apply()
        print('check 2 (uniform eps_p, a 0.1, x = R0 P X + t, skin solved %d products): energy %.3e on the plastic references, %.3e on '
              'the elastic (ratio %.1e); |dE/dx| max %.3e vs %.3e; |dE/dy| max %.3e; bonds |d - dp| max %.3e m' % (
                  s_['products'], e_p, e_e, e_p/e_e, float(gx.norm(dim=1).max()), float(gx0.norm(dim=1).max()), float(gy.norm(dim=1).max()), dmax), flush=True)
    # 3. dynamics
    xc = (mass[:, None]*X).sum(0)/mass.sum()
    Dg = torch.diag(torch.tensor([-1., .5, .5], dtype=f64, device=dev))
    k0 = .5*float((mass[:, None]*((X-xc)@Dg.T)**2).sum())                          # KE per a^2
    a = math.sqrt(W.mu*1.5*pa.peak**2*float(mass.sum())/k0)                         # KE = mu |eps|^2 V at eps = peak diag(-1, 1/2, 1/2)
    v0 = a*(X-xc)@Dg.T
    sig = E*pa.ey
    lo, hi = X.min(0).values, X.max(0).values
    print('part: %d cells, extent %s mm; start velocity gradient a %.3f 1/s (|v| max %.3f m/s), KE %.4e; sigma_y %.4g (E ey, ey %.3g); '
          'per step strain a dt at dt %s: %s' % (W.n, np.round(1e3*(hi-lo).cpu().numpy(), 1).tolist(), a, float(v0.norm(dim=1).max()), a*a*k0, sig, pa.ey,
                                                 ['%.4g' % d for d in pa.dts], ['%.4f' % (a*d) for d in pa.dts]), flush=True)
    zero = torch.zeros_like(X)
    runs = [(pa.dts[0], math.inf)]+[(d, sig) for d in pa.dts]
    res = {}
    for dt, sy in runs:
        t1 = time.time()
        pl = Plasticity(W, st, sy, pa.criterion)
        pl.merged = not pa.nested
        pl.rotation, pl.gradient = pa.linear != 'off', pa.linear == 'full'
        I = MidpointIntegrator(b, dt, X.clone(), v0.clone(), Y.clone(), zero, tol=pa.tol)
        KE = lambda: .5*float((mass[:, None]*I.v*I.v).sum())
        H0 = KE()+I.E_int
        steps = int(round(pa.T/dt))
        hist, ever = [], torch.zeros(W.n, dtype=torch.bool, device=dev)
        prods, passes, checks, dws = 0, [], [], []
        for s in range(1, steps+1):
            if pa.ret == 'mid' and sy != math.inf:
                s_, u = pl.step(I)
                prods += u['products']; passes.append(u['passes']); dws.append(u['dW'])
                if u.get('check'):
                    checks.append(u['check'])
                if pa.log_steps:
                    print('  step %d: passes %s, inner %s; |eps_p| max %.3e, step |d eps| max %.3e; z finite %s; check %s' % (
                        s, ' '.join('(%d, %d, %.1e, %d)' % (a, c, ch/max(u['lim'], 1e-300), n_) for a, c, ch, n_ in u['log']), u.get('inner'),
                        float(pl.eps.flatten(1).norm(dim=1).max()), float((pl.eps-pl.eps_prev).flatten(1).norm(dim=1).max()),
                        bool(torch.isfinite(I.x).all()), {k_: round(v_, 4) for k_, v_ in (u.get('check') or {}).items()}), flush=True)
            else:
                s_ = I.step()
                prods += int(s_['products'])
                u = pl.update(I.x, I.y_int)
                I.E_int = u['e_after']
            U = W.tR.transpose(1, 2)@W.tF
            lu = (mass[:, None, None]*sym_fun(U, torch.log)).sum(0)/mass.sum()      # the mean total log strain
            em, emax = pl.strain(mass)
            ever |= (pl.eps.flatten(1).norm(dim=1) > 0)
            hist.append((s*dt, KE(), I.E_int, pl.dissipated, u['yielding'], float(lu[0, 0]), float(em[0, 0]), emax))
        h = np.array(hist)
        bal = (h[:, 1]+h[:, 2]+h[:, 3])/H0-1
        em, emax = pl.strain(mass)
        w_ = torch.linalg.eigvalsh(em).cpu().numpy()
        res[(dt, sy)] = dict(em=em.cpu().numpy(), h=h)
        if passes:
            pz = np.array(passes)
            print('   passes per step (return within it): mean %.2f, max %d; steps with > 1: %d of %d' % (pz.mean(), pz.max(), int((pz > 1).sum()), len(pz)), flush=True)
        if dws:
            dw_ = np.array(dws); surf = torch.zeros(W.n, dtype=torch.bool, device=dev); surf[st.c_body.long()] = True
            en = pl.eps.flatten(1).norm(dim=1); top = torch.argsort(en, descending=True)[:20]
            print('   per-step dissipation: negative in %d of %d steps, most negative %.2e (largest %.2e); the 20 largest |eps_p|: %d on the surface (surface cells %d of %d), |eps_p| median %.3f' % (
                int((dw_ < 0).sum()), len(dw_), dw_.min(), dw_.max(), int(surf[top].sum()), int(surf.sum()), W.n, float(en.median())), flush=True)
        if checks:
            print('   yield on the model\'s stress (each step\'s end of passes): yielding cells | |dev tau| / sigma^ V - 1 | max %.1e, cos(tau, delta) min %.6f; elastic cells |dev tau| / sigma^ V max %.4f' % (
                max(c['surf'] for c in checks), min(c['cos'] for c in checks), max(c['elastic_max'] for c in checks)), flush=True)
        print('%s dt 1/%d: %.0f s, %d steps, %d products; mean total strain xx: min %.4f (t %.4f), last %.4f; plastic: W_p / KE0 %.3f, '
              'cells ever yielding %d, yielding last step %d, mean eps_p eigenvalues %s, xx %.5f, max |eps_p| %.4f; balance (KE + E + W_p) / H0 - 1: '
              'max %.1e, min %.1e, last %.1e' % ('elastic' if sy == math.inf else 'plastic', round(1/dt), time.time()-t1, steps, prods, h[:, 5].min(),
                                                  h[h[:, 5].argmin(), 0], h[-1, 5], pl.dissipated/H0, int(ever.sum()), int(h[-1, 4]), np.round(w_, 5).tolist(),
                                                  float(em[0, 0]), emax, bal.max(), bal.min(), bal[-1]), flush=True)
        ks = [k for k in range(len(h)) if (k+1) % max(1, round(.005/dt)) == 0]
        print('   t, mean total xx, mean plastic xx, yielding: %s' % '; '.join('%.3f %.4f %.4f %d' % (h[k, 0], h[k, 5], h[k, 6], h[k, 4]) for k in ks), flush=True)
    pls = [(d, sig) for d in pa.dts]
    ref = res[pls[-1]]['em']
    for key in pls[:-1]:
        print('mean eps_p at dt 1/%d against 1/%d: |difference| / |finest| %.3f (xx %.5f vs %.5f)' % (
            round(1/key[0]), round(1/pls[-1][0]), np.linalg.norm(res[key]['em']-ref)/np.linalg.norm(ref), res[key]['em'][0, 0], ref[0, 0]), flush=True)
    np.savez('output/test_part/plastic_part_check_%s.npz' % pa.ret, **{'h_%s_%d' % ('el' if sy == math.inf else 'pl', round(1/d)): r['h'] for (d, sy), r in res.items()})


if __name__ == '__main__':
    main()
