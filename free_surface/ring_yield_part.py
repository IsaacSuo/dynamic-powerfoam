"""ring_yield.py on the free part (2026-11-01), as merged_part.py (2026-10-31): plastic_part_check.py's dynamic case, no gravity, no contact, start velocity v = a diag(-1, 1/2, 1/2)(X - Xc), a for an
elastic peak strain of --peak, yield strain --ey. The reference: the 10-26 skin model with the return within the step, whose mean plastic strain xx converged in dt
(--ref, plastic_part_check_mid.npz: -0.1193 / -0.1190 / -0.1190 at 1/1200, 1/2400, 1/4800). Per dt: the mean plastic and total strain, the energy's account (the integrator's own
dissipation is what is left), the solve's work.
  python free_surface/merged_part.py --foam <dir with dem_exact_single> --ref <npz> --save <dir>"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

from free_surface.closure_warp import TrustCG
from free_surface.ring_yield import RingYield, YieldStep

f64, dev = torch.float64, 'cuda'
a_ = argparse.ArgumentParser()
a_.add_argument('--foam', type=Path, required=True)
a_.add_argument('--ref', type=Path, default=None)
a_.add_argument('--save', type=Path, default=Path('output/plastic'))
a_.add_argument('--dts', type=float, nargs='+', default=[1/1200, 1/2400, 1/4800])
a_.add_argument('--T', type=float, default=.08)
a_.add_argument('--ey', type=float, default=.01)
a_.add_argument('--peak', type=float, default=.05)
a_.add_argument('--E', type=float, default=23.00838)
a_.add_argument('--nu', type=float, default=.3008)
a_.add_argument('--tol', type=float, default=1e-3)
a_.add_argument('--grad', default='gg')
a_.add_argument('--theta', type=float, default=.5)
a_.add_argument('--kappa', type=float, default=5.)
a_.add_argument('--eps-cut', type=float, default=0.)
a_.add_argument('--aa', type=int, default=6)
a_.add_argument('--tag', default='yield_part')
a_.add_argument('--field', action='store_true', help='the last dt\'s end state into <tag>_field.npz')
args = a_.parse_args()
g = np.load(args.foam/'dem_exact_single'/'geometry.npz'); cl = np.load(args.foam/'dem_exact_single'/'cells.npz')
X0, r0, V0 = g['points'], cl['radii'], g['masses']; n = len(X0)
args.save.mkdir(parents=True, exist_ok=True)
ref = dict(np.load(args.ref)) if args.ref else {}
mass = torch.as_tensor(V0, dtype=f64, device=dev); M = float(mass.sum()); M3 = mass[:, None]
X = torch.as_tensor(X0, dtype=f64, device=dev)
xc = (M3*X).sum(0)/M
Dg = torch.diag(torch.tensor([-1., .5, .5], dtype=f64, device=dev))
k0 = .5*float((M3*((X-xc)@Dg.T)**2).sum())
mu = args.E/(2*(1+args.nu))
a = math.sqrt(mu*1.5*args.peak**2*M/k0)
print('part: %d cells, E %g, nu %g, yield strain %g, a %.3f 1/s; grad %s, tol %g of the weight' % (n, args.E, args.nu, args.ey, a, args.grad, args.tol), flush=True)
Xd = (X-xc)
Mm = torch.einsum('s,si,sj->ij', mass, Xd, Xd)


def total_strain(x):
    """the mean total log strain: the least-squares affine map of the sites, its log stretch"""
    xd = x-(M3*x).sum(0)/M
    F = torch.einsum('s,si,sj->ij', mass, xd, Xd)@torch.linalg.inv(Mm)
    w, Q = torch.linalg.eigh(F.T@F)
    return Q@torch.diag(.5*torch.log(w))@Q.T


out = {}
for dt in args.dts:
    t0 = time.time()
    W = RingYield(X0, r0, V0, g['pairs'], g['areas'], g['contacts'], args.E, args.nu, args.E*args.ey, theta=args.theta, kappa=args.kappa, grad=args.grad, eps_cut=args.eps_cut, ledger=True)
    W.set_ball((0., 1., 0.), -100., 0.)
    W.start()
    W.prepare()
    fg = torch.zeros(n, 3, dtype=f64, device=dev)
    wref = 9.8*float(M3.norm()*3**0)
    CG = TrustCG(W, mass, dt, per=2, product_iters=1)
    W.update_ul(W.X); W.grad(W.X); W.reread(); W.grad(W.X); ST = YieldStep(W, CG, mass, W.ul_blocks(), fg, dt, 0.); ST.aa_m = args.aa
    ST.pre_ul = True; ST.pre_every = 4; ST.smooth_pred = ST.carry_delta = True
    ST.set_theta(args.theta, mass)
    v = a*Xd@Dg.T
    ST.set_state(X.clone(), v, torch.zeros_like(X))
    tol = args.tol*wref
    KE = lambda: .5*float((M3*ST.tv**2).sum())
    H0 = KE()
    steps = int(round(args.T/dt)); rec = []; prods = 0; born = closed = 0
    every = max(1, round(.005/dt))
    for s in range(1, steps+1):
        e_, its, g2 = ST.step(tol, lambda g_: .5*tol)
        if not math.isfinite(g2) or math.sqrt(g2) > tol:
            print('step %d: the solve did not converge (|g| / weight %s after %d Newton iterations); stopping' % (s, math.sqrt(g2)/wref if math.isfinite(g2) else g2, its), flush=True)
            break
        prods += e_
        ST.end()
        born += W.ul_stats['born']; closed += W.ul_stats['closed']
        ed, ev = W.stored(); ej = 0.
        ep = (mass[:, None, None]*W.ep).sum(0)/M
        rec.append((s*dt, KE(), ed, ev, ej, W.Wp, float(ep[0, 0]), float(total_strain(ST.tx)[0, 0]), W.ul_stats['yielding'], W.ul_stats['yielded'], float(W.ep.flatten(1).norm(dim=1).max()), float(torch.exp(W.tul['Ln']).min()),
                    float(torch.exp(W.tul['Ln']).max()), e_, its, W.ul_stats['radii_it'], W.ul_stats['radii_fail'], W.ul_stats['flips'], W.ul_stats['walls']))
        if s % every == 0 or s == steps:
            q = rec[-1]
            print('  integrated mean strain xx %.5f yy %.5f zz %.5f; springs %d, new %d, closed %d this step, without a wall %d; smallest one-cell dissipation / limit store %.2e' % (0., 0., 0., W.ul_stats['rows'], W.ul_stats['born'], W.ul_stats['closed'], W.ul_stats['ghost'], W.ul_stats.get('D_min', 0.)), flush=True)
            print('dt 1/%d t %.4f (%.0f s): KE %.3e dev %.3e vol %.3e var+springs %.3e W_p %.3e, left %.3f of H0; mean plastic xx %.5f, total xx %.5f; yielding cells %d, yielded so far %d, walls %d; max |eps_p| %.3f, J %.3f..%.3f; products %d, Newton %d, radii %d, fails %d, flips %d' % (
                round(1/dt), q[0], time.time()-t0, q[1], q[2], q[3], q[4], q[5], 1-(q[1]+q[2]+q[3]+q[4]+q[5])/H0, q[6], q[7], q[8], q[9], q[18], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17]), flush=True)
    r_ = np.array(rec)
    out['dt%d' % round(1/dt)] = r_
    key = 'h_pl_%d' % round(1/dt)
    line = ''
    if key in ref and len(r_):
        h = ref[key]
        line = '; the 10-26 reference at this dt: mean plastic xx %.5f, W_p / KE0 %.3f, total xx last %.5f' % (h[-1, 6], h[-1, 3]/(h[0, 1]+h[0, 2]+h[0, 3]), h[-1, 5])
    if len(r_):
        print('dt 1/%d: %.0f s, %d steps, %d products; mean plastic xx %.5f, W_p / KE0 %.3f, total xx last %.5f%s' % (round(1/dt), time.time()-t0, len(r_), prods, r_[-1, 6], r_[-1, 5]/H0, r_[-1, 7], line), flush=True)
    print('dt 1/%d: new springs %d, closed %d over the run, of %d at rest; springs\' share of W_p %.3f' % (round(1/dt), born, closed, len(g['pairs']), W.Wp_half/max(W.Wp, 1e-300)), flush=True)
    if args.field:
        np.savez(args.save/(args.tag+'_field.npz'), x=ST.tx.cpu().numpy(), X=X.cpu().numpy(), V=mass.cpu().numpy(), ep=W.ep.cpu().numpy(), ei=W.tei.cpu().numpy(), be=W.tbe.cpu().numpy(), yld=W._yld.cpu().numpy(), Ln=W.tul['Ln'].cpu().numpy())
    del ST, CG, W
np.savez(args.save/(args.tag+'.npz'), **out)
print('done', flush=True)
