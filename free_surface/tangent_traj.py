"""Check 2 of the tangent sensitivities (tangent.py, 2026-10-25): the tangents carried along a whole drop against central
differences of whole runs. The drop: tangent_check's start (lowest point --height above the ground, velocity --v0, PAC-
NeRF scene 0's), the joint midpoint rule, records every 1/240 s.
  --mode tangent: the run with the tangents of --params (per step: products, the tangents' CG iterations and passes)
  --mode fd --param P --sign 1|-1: the run at the perturbed parameter (lnE, lnbeta, nu rebuilt; v0x / v0y / v0z: the
                                   start's velocity; lnmu: the friction's mu), eps --eps (relative for lnE, lnbeta, lnmu;
                                   absolute for nu, v0 in m/s)
  --mu: the ground's friction (SkinWarp's glue, 10-25; 0 off), --fkappa its glue's stiffness factor (default SkinWarp's)
  --mode compare: per parameter, |tangent - FD| / |FD| of the bodies' positions per record, and of the centre of mass
Writes <out>/tangent_traj/<run>.npz.
  python free_surface/tangent_traj.py --out output/round01_pruned_exact --beta 0.01 --integrator midpoint --mode tangent"""
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from free_surface.joint_step import MidpointIntegrator
from free_surface.skin_contact_check import f64, dev
from free_surface.skin_dynamics_check import build_model, parser
from free_surface.tangent import Tangent, make_W_nu


def main():
    a = parser()
    a.add_argument('--mode', choices=('tangent', 'fd', 'compare'), required=True)
    a.add_argument('--param', default=None)
    a.add_argument('--sign', type=int, default=1)
    a.add_argument('--eps', type=float, default=1e-2)
    a.add_argument('--params', nargs='+', default=['lnE', 'lnbeta', 'nu', 'v0x', 'v0y'])
    a.add_argument('--height', type=float, default=.02)
    a.add_argument('--v0', type=float, nargs=3, default=[-.408, -.5, -.066])
    a.add_argument('--T', type=float, default=.25)
    a.add_argument('--tag', default='', help='suffix of the tangent run\'s file (tangent<tag>.npz), read by compare too')
    a.add_argument('--tol', type=float, default=1e-6, help='the step\'s solve')
    a.add_argument('--tan-tol', type=float, default=1e-6, help='the tangents\' CG and constraint')
    a.add_argument('--nu-model', type=int, default=0, help='dG/dnu from a second model built at nu + dnu (else SkinWarp.set_material in place)')
    a.add_argument('--dnu', type=float, default=1e-6)
    a.add_argument('--start', choices=('predict', 'inertia', 'last'), default='predict', help='the tangents\' CG start (tangent.Tangent)')
    a.add_argument('--copies', type=int, default=0, help='product SkinWarps: the parameters\' CGs at once on 1 + copies streams')
    a.add_argument('--floor', type=int, default=1, help='the tangents\' CG stops at tol max(|b|, the weight)')
    a.add_argument('--seed', type=int, default=0, help='the first parameter\'s CG directions seed the others\' starts')
    a.add_argument('--log-starts', type=int, default=0, help='log the start residuals / |b| of each start')
    a.add_argument('--mu', type=float, default=0., help='the ground\'s friction coefficient (0: frictionless)')
    a.add_argument('--fkappa', type=float, default=None, help='the friction glue\'s stiffness factor (SkinWarp.fkappa)')
    a.set_defaults(dt=1/1200)
    args = a.parse_args()
    d = Path(args.out)/'tangent_traj'; d.mkdir(parents=True, exist_ok=True)
    if args.mode == 'compare':
        return compare(args, d)
    run = 'tangent'+args.tag
    v0 = list(args.v0)
    if args.mode == 'fd':
        e = args.sign*args.eps
        if args.param == 'lnE':
            args.E *= math.exp(e)
        elif args.param == 'lnbeta':
            args.beta *= math.exp(e)
        elif args.param == 'nu':
            args.nu += e
        elif args.param == 'lnmu':
            args.mu *= math.exp(e)
        else:
            v0['xyz'.index(args.param[2])] += e
        run = '%s%+d' % (args.param, args.sign)
    print('run %s: E %.6g, nu %.6g, beta %.6g, v0 %s, mu %.6g; dt %.4e, %.3f s; tol %.0e, tangents %.0e' % (
        run, args.E, args.nu, args.beta, v0, args.mu, args.dt, args.T, args.tol, args.tan_tol), flush=True)
    b = build_model(args, contact=dict(normal=(0., 1., 0.), c0=0., points=False, shift=True))
    st, W, mass = b['st'], b['W'], b['mass']
    W.tfpar[0] = args.mu
    if args.fkappa is not None:
        W.fkappa = args.fkappa
    W.glue()
    up = torch.tensor([0., 1., 0.], dtype=f64, device=dev)
    fg = -9.8*mass[:, None]*up
    W.grad(st.X, st.Y)
    lift = (args.height-float(W.tgap.min()))*up
    x0, y0 = st.X+lift, st.Y+lift
    I = MidpointIntegrator(b, args.dt, x0, torch.tensor(v0, dtype=f64, device=dev).expand_as(x0).clone(), y0, fg, tol=args.tol,
                           ref=float(fg.norm()), contact=True)
    Tg = None
    if args.mode == 'tangent':
        W_nu = make_W_nu(args, args.dnu, b['out']) if 'nu' in args.params and args.nu_model else None   # (default: in place)
        cps = [b['make_copy']() for _ in range(args.copies)]
        if cps:
            free, total = torch.cuda.mem_get_info()
            print('%d product copies; GPU memory in use %.2f of %.2f GB' % (len(cps), (total-free)/1e9, total/1e9), flush=True)
        Tg = Tangent(I, args.params, W_nu=W_nu, dnu=args.dnu, tol=args.tan_tol, al_tol=args.tan_tol,
                     start=args.start, log_starts=bool(args.log_starts), copies=cps, floor=bool(args.floor),
                     seed=bool(args.seed))
    steps = int(round(args.T/args.dt)); every = max(1, round(1/(240*args.dt)))
    t_rec, X, logs = [0.], [I.x.cpu().numpy()], []
    TX = {p: [np.zeros_like(X[0])] for p in (Tg.params if Tg else [])}
    t0 = time.time()
    for s in range(1, steps+1):
        t1 = time.time()
        if Tg is None:
            s_ = I.step(); info = None
        else:
            s_, info = Tg.step()
        logs.append(dict(step=s, products=int(s_['products']), passes=int(s_['al']), seconds=time.time()-t1,
                         active=int((W.tact > 0).sum()), tangent=info['params'] if info else None))
        if s % every == 0:
            t_rec.append(s*args.dt); X.append(I.x.cpu().numpy())
            for p in TX:
                TX[p].append(Tg.T[p]['x'].cpu().numpy())
            lg = logs[-every:]
            msg = 't %.4f s: active %d; forward products %d over the last %d steps' % (s*args.dt, lg[-1]['active'], sum(l_['products'] for l_ in lg), every)
            if Tg is not None:
                cgs = {p: sum(l_['tangent'][p]['cg_iterations'] for l_ in lg) for p in Tg.params}
                als = {p: max(l_['tangent'][p]['al_iterations'] for l_ in lg) for p in Tg.params}
                msg += '; tangents CG %s, passes max %s' % (cgs, als)
                if args.log_starts:
                    st_ = {nm: np.median([l_['tangent'][p]['starts'].get(nm, np.nan) for l_ in lg for p in Tg.params]) for nm in ('last', 'predict', 'inertia', 'used')}
                    msg += '; start |r| / |b| (median) %s' % ', '.join('%s %.1e' % kv for kv in st_.items())
            print(msg+'; %.0f s' % (time.time()-t0), flush=True)
    np.savez(d/('%s.npz' % run), t=np.array(t_rec), X=np.stack(X), mass=mass.cpu().numpy(), eps=args.eps, log=json.dumps(logs),
             **{'T_'+p: np.stack(v) for p, v in TX.items()})
    print('%s written, %.0f s' % (run, time.time()-t0), flush=True)


def compare(args, d):
    z = np.load(d/('tangent%s.npz' % args.tag))
    print('tangent run: %s' % ('tangent%s' % args.tag), flush=True)
    t, X, m = z['t'], z['X'], z['mass']
    eps = float(z['eps'])
    logs = json.loads(str(z['log']))
    fwd = sum(l_['products'] for l_ in logs)
    print('forward products %d over %d steps; tangent CG iterations per parameter: %s' % (
        fwd, len(logs), {p: sum(l_['tangent'][p]['cg_iterations'] for l_ in logs) for p in args.params if 'T_'+p in z}), flush=True)
    com = lambda A: (m[None, :, None]*A).sum(1)/m.sum()
    for p in args.params:
        fp, fm = d/('%s+1.npz' % p), d/('%s-1.npz' % p)
        if 'T_'+p not in z or not (fp.exists() and fm.exists()):
            print('%s: missing' % p); continue
        Xp, Xm = np.load(fp)['X'], np.load(fm)['X']
        fd = (Xp-Xm)/(2*eps)
        T = z['T_'+p]
        nfd = np.linalg.norm(fd.reshape(len(t), -1), axis=1)
        err = np.linalg.norm((T-fd).reshape(len(t), -1), axis=1)/np.maximum(nfd, 1e-300)
        cerr = np.linalg.norm(com(T)-com(fd), axis=1)/np.maximum(np.linalg.norm(com(fd), axis=1), 1e-300)
        idx = [k for k in range(len(t)) if k % 6 == 0 or k == len(t)-1]
        print('%s: positions |T - FD| / |FD| at t = %s: %s; max over t > 0 %.2e (t %.4f); centre of mass: max %.2e' % (
            p, ', '.join('%.3f' % t[k] for k in idx), ', '.join('%.1e' % err[k] for k in idx), err[1:].max(), t[1:][err[1:].argmax()], cerr[1:].max()),
            flush=True)


if __name__ == '__main__':
    main()
