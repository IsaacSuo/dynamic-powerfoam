"""Check 1 of the tangent sensitivities (tangent.py, 2026-10-25): one midpoint step from a start state, the tangents
d(x_n+1, v_n+1, end gaps)/d theta against central differences of the step re-solved at theta +- eps.
States: 'start' (the drop's start, lowest point --height above the ground, velocity --v0: free flight), and saved
integrator states (--states: skin_drop_state_step<N>.npz of skin_drop_check --save-state; their run's --dt and --beta).
  --mode tangent: per state the base step, a repeat of it (the solves' own scatter), the tangents of --params, and the
                  in-place differences of the start velocity (v0x, v0y, v0z)
  --mode fd --param lnE|lnbeta|nu --sign 1|-1: the step with the model rebuilt at the perturbed parameter
  --mode compare: the table
Writes <out>/tangent_check/<state>_<run>.npz.
  python free_surface/tangent_check.py --out output/round01_pruned_exact --dt 2.0833333e-4 --beta 0.01 --integrator midpoint --mode tangent"""
import json
import math
from pathlib import Path

import numpy as np
import torch

from free_surface.joint_step import MidpointIntegrator
from free_surface.skin_contact_check import f64, dev
from free_surface.skin_dynamics_check import build_model, parser
from free_surface.tangent import NAMES, Tangent, make_W_nu

LOADED = ('x', 'v', 'a', 'xm1', 'ym1', 'xm2', 'ym2')


def main():
    a = parser()
    a.add_argument('--mode', choices=('tangent', 'fd', 'compare'), required=True)
    a.add_argument('--param', default=None)
    a.add_argument('--sign', type=int, default=1)
    a.add_argument('--eps', type=float, default=1e-4, help='lnE, lnbeta, nu, v0 (m/s)')
    a.add_argument('--states', nargs='+', default=['start', 'step349', 'step900'])
    a.add_argument('--params', nargs='+', default=['lnE', 'lnbeta', 'nu', 'v0x', 'v0y', 'v0z'])
    a.add_argument('--height', type=float, default=.02)
    a.add_argument('--v0', type=float, nargs=3, default=[-.408, -.5, -.066])
    a.add_argument('--tol', type=float, default=1e-9)
    a.add_argument('--nu-model', type=int, default=0, help='dG/dnu from a second model built at nu + dnu (else SkinWarp.set_material in place)')
    a.add_argument('--dnu', type=float, default=1e-6, help='tangent.py\'s one-sided step in nu')
    args = a.parse_args()
    d = Path(args.out)/'tangent_check'; d.mkdir(parents=True, exist_ok=True)
    if args.mode == 'compare':
        return compare(args, d)
    base = dict(E=args.E, beta=args.beta, nu=args.nu)
    run = 'base'
    if args.mode == 'fd':
        e = args.sign*args.eps
        if args.param == 'lnE':
            args.E = args.E*math.exp(e)
        elif args.param == 'lnbeta':
            args.beta = args.beta*math.exp(e)
        elif args.param == 'nu':
            args.nu = args.nu+e
        run = '%s%+d' % (args.param, args.sign)
    print('model E %.6g, nu %.6g, beta %.6g; dt %.4e; tol %.0e' % (args.E, args.nu, args.beta, args.dt, args.tol), flush=True)
    b = build_model(args, contact=dict(normal=(0., 1., 0.), c0=0., points=False, shift=True))
    st, W, mass = b['st'], b['W'], b['mass']
    up = torch.tensor([0., 1., 0.], dtype=f64, device=dev)
    fg = -9.8*mass[:, None]*up
    ref = float(fg.norm())
    W.grad(st.X, st.Y)
    lift = (args.height-float(W.tgap.min()))*up
    x0, y0 = st.X+lift, st.Y+lift
    v0 = torch.tensor(args.v0, dtype=f64, device=dev).expand_as(x0).clone()
    I = MidpointIntegrator(b, args.dt, x0, v0, y0, fg, tol=args.tol, ref=ref, contact=True)
    start = snapshot(I, W)
    W_nu = make_W_nu(args, args.dnu, b['out']) if args.mode == 'tangent' and 'nu' in args.params and args.nu_model else None   # (default: in place)
    for name in args.states:
        if name == 'start':
            restore(I, W, start)
        else:
            z = np.load(b['out']/('skin_drop_state_%s.npz' % name))
            snap = {k: torch.as_tensor(z[k], device=dev) for k in LOADED}
            snap.update(lam=torch.as_tensor(z['lam'], device=dev), act=torch.as_tensor(z['act'], device=dev).to(W.tact.dtype),
                        goff=torch.as_tensor(z['goff'], device=dev), rho=torch.as_tensor(z['rho'], device=dev),
                        ref={k: torch.as_tensor(z['ref_'+k], device=dev) for k in NAMES})
            restore(I, W, snap)
        s0 = snapshot(I, W)
        res = {}
        if args.mode == 'tangent':
            Tg = Tangent(I, args.params, W_nu=W_nu, dnu=args.dnu, tol=args.tol, al_tol=args.tol)
            s_, info = Tg.step()
            res.update(outcome(I, W, s_, 'base'))
            for p in args.params:
                T = Tg.T[p]
                res['T_%s_x' % p], res['T_%s_v' % p], res['T_%s_g' % p] = T['x'].cpu().numpy(), T['v'].cpu().numpy(), T['g'].cpu().numpy()
            print('%s: step residual %.1e of the weight, active %d, settled %s; %s' % (name, info['step_residual'], info['active'], info['settled'],
                  json.dumps(info['params'])), flush=True)
            restore(I, W, s0)
            res.update(outcome(I, W, I.step(), 'repeat'))
            for p in [p for p in args.params if p.startswith('v0')]:
                for sg in (1, -1):
                    restore(I, W, s0)
                    I.v[:, 'xyz'.index(p[2])] += sg*args.eps
                    res.update(outcome(I, W, I.step(), '%s%+d' % (p, sg)))
        else:
            res.update(outcome(I, W, I.step(), run))
        restore(I, W, s0)
        np.savez(d/('%s_%s.npz' % (name, run)), base=json.dumps(base), eps=args.eps, **res)
        print('%s %s: written' % (name, run), flush=True)


def outcome(I, W, s_, tag):
    return {'%s_x' % tag: I.x.cpu().numpy(), '%s_v' % tag: I.v.cpu().numpy(), '%s_g' % tag: W.tgoff.cpu().numpy(),
            '%s_act' % tag: W.tact.cpu().numpy(), '%s_res' % tag: np.array([s_.get('residual', np.nan), s_['al']])}


def snapshot(I, W):
    s = {k: getattr(I, k).clone() for k in LOADED}
    s.update(lam=W.tlam.clone(), act=W.tact.clone(), goff=W.tgoff.clone(), rho=W.trho.clone(), ref={k: r.clone() for k, r in W.tref.items()})
    return s


def restore(I, W, s):
    for k in LOADED:
        setattr(I, k, s[k].clone())
    W.tlam.copy_(s['lam']); W.tact.copy_(s['act']); W.tgoff.copy_(s['goff']); W.trho.copy_(s['rho'])
    for k, r in s['ref'].items():
        W.tref[k].copy_(r)


def compare(args, d):
    rel = lambda a_, b_: float(np.linalg.norm(a_-b_)/max(np.linalg.norm(b_), 1e-300))
    for name in args.states:
        f = d/('%s_base.npz' % name)
        if not f.exists():
            continue
        z = np.load(f)
        eps = float(z['eps'])
        print('%s: repeat of the base step |dx| %.1e, |dv| %.1e (the solves\' scatter); active %d' % (
            name, np.abs(z['repeat_x']-z['base_x']).max(), np.abs(z['repeat_v']-z['base_v']).max(), int((z['base_act'] > 0).sum())), flush=True)
        for p in args.params:
            if p.startswith('v0'):
                zp, zm = z, z
                kp, km = '%s+1' % p, '%s-1' % p
            else:
                fp, fm = d/('%s_%s+1.npz' % (name, p)), d/('%s_%s-1.npz' % (name, p))
                if not (fp.exists() and fm.exists()):
                    print('   %s: no differences yet' % p); continue
                zp, zm = np.load(fp), np.load(fm)
                kp, km = '%s+1' % p, '%s-1' % p
            fdx = (zp[kp+'_x']-zm[km+'_x'])/(2*eps); fdv = (zp[kp+'_v']-zm[km+'_v'])/(2*eps); fdg = (zp[kp+'_g']-zm[km+'_g'])/(2*eps)
            Tx, Tv, Tg_ = z['T_%s_x' % p], z['T_%s_v' % p], z['T_%s_g' % p]
            same = bool((zp[kp+'_act'] == z['base_act']).all() and (zm[km+'_act'] == z['base_act']).all())
            noise = np.abs(z['repeat_x']-z['base_x']).max()/(2*eps)
            print('   %-6s x_n+1: |FD| %.3e, tangent vs FD %.2e | v_n+1: |FD| %.3e, %.2e | end gaps: |FD| %.3e, %.2e | active sets equal %s; '
                  'the scatter / 2 eps %.1e' % (p, np.linalg.norm(fdx), rel(Tx, fdx), np.linalg.norm(fdv), rel(Tv, fdv), np.linalg.norm(fdg),
                                                 rel(Tg_, fdg), same, noise), flush=True)


if __name__ == '__main__':
    main()
