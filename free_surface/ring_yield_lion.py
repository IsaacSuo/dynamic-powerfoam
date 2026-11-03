"""ring_yield.py on the lion, as ring_lion.py: the foam and conditions of dem_contact_lion.py (round_clean, GT E nu, mu 0, dt 1/1200, 1.2 s), no self contact, the limit out of reach, every cell on the cut from the first step, beta on the model's own viscosity (ring_yield.py's header; --beta 0 and --eps-cut 1e9: the controls, both blow up on landing, as ring_lion.py at beta 0). Frames (x, F, t, L) every --every steps from t = 0, for dem_gt_compare.py; the log line-buffered; no timeout.
  python free_surface/ring_yield_lion.py --foam <dir with dem_exact_single> --save <dir> --tag <tag>"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import warp as wp

from free_surface.closure_warp import TrustCG
from free_surface.ring_yield import RingYield, YieldStep
from free_surface.ring_moment import RingMoment

f64, dev = torch.float64, 'cuda'
wp_t = lambda a_: wp.to_torch(a_, requires_grad=False)
a_ = argparse.ArgumentParser()
a_.add_argument('--foam', type=Path, required=True)
a_.add_argument('--save', type=Path, default=Path('output/lion'))
a_.add_argument('--tag', default='ry')
a_.add_argument('--T', type=float, default=1.2)
a_.add_argument('--every', type=int, default=20)
a_.add_argument('--dt', type=float, default=1/1200)
a_.add_argument('--tol', type=float, default=1e-3)
a_.add_argument('--pre-every', type=int, default=4)
a_.add_argument('--aa', type=int, default=6)
a_.add_argument('--hold-after', type=int, default=3)
a_.add_argument('--eps-cut', type=float, default=-1.)
a_.add_argument('--beta', type=float, default=.05)
a_.add_argument('--no-elastic-births', action='store_true', help='a wall no return has reached: a free surface')
a_.add_argument('--start-frames', type=Path, default=None, help='restart from a saved frame of an elastic run')
a_.add_argument('--start-k', type=int, default=0)
a_.add_argument('--save-state', type=int, nargs='*', default=[])
a_.add_argument('--load-state', type=Path, default=None)
a_.add_argument('--ground-k', type=float, default=1., help='the ground\'s Hertz modulus times this (a test)')
a_.add_argument('--rest-ground', action='store_true', help='the ground on the rest radii, not the cut\'s')
a_.add_argument('--dense-from', type=int, default=0, help='every step from this one: the sites into dense_<tag>.npz')
a_.add_argument('--probe-30', type=Path, nargs='*', default=[], help='these saved states: ring_moment_force.check30, then stop')
a_.add_argument('--probe-jump', type=Path, nargs='*', default=[], help='these saved states: ring_moment_force.jump_check, then stop')
a_.add_argument('--dense', type=Path, default=None)
a_.add_argument('--probe-colour', action='store_true', help='at the loaded state: the lock against the colours, then stop')
a_.add_argument('--probe-contact', action='store_true', help='at the loaded state: the colour contact\'s births and gradient, then stop')
a_.add_argument('--probe-ground', action='store_true', help='at the loaded state: the ground in the cut, its faces and the gradient, then stop')
a_.add_argument('--probe-unstable', action='store_true', help='at the loaded state: each leg\'s lowest modes, then stop')
a_.add_argument('--probe-lock', action='store_true', help='at the loaded state: what stiffens the leg, then stop')
a_.add_argument('--probe-equil', action='store_true', help='at the loaded state: is a leg at an equilibrium, then stop')
a_.add_argument('--probe-leg', action='store_true', help='at the loaded state: ring_moment_force.py, then stop')
a_.add_argument('--g', type=float, default=9.8, help='gravity')
a_.add_argument('--moment', action='store_true', help='the springs not tied to the walls: ring_moment.py')
a_.add_argument('--true-blocks', action='store_true', help='the preconditioner on the true diagonal blocks')
a_.add_argument('--colour', type=float, default=0., help='ring_moment.py\'s colour contact at this sigma (graph steps; the lion\'s 1.19), with --moment')
a_.add_argument('--jump', action='store_true', help='with --colour: a wall without a spring counts by the jump test (RingMoment.set_jump_share), no weightless springs')
a_.add_argument('--ground-cut', action='store_true', help='the ground as a plane of the cut (MergedUL.set_ground_cut), the Hertz ground off')
a_.add_argument('--watch-ground', action='store_true', help='with --ground-cut: after every step, the centre points under the plane and what their cells had at the step\'s start')
a_.add_argument('--probe-kernels', action='store_true', help='at the loaded state: one evaluation kernel by kernel (ring_moment_force.kernel_times), then stop')
a_.add_argument('--trace-solves', action='store_true', help='after every step, its solve outer iteration by outer: the inner solves, products, edges, |g| / tol')
a_.add_argument('--cuda-profile', action='store_true', help='cudaProfilerStart / Stop around the steps (nsys --capture-range=cudaProfilerApi)')
args = a_.parse_args()
E, nu, C0, v0 = 10648.614517785627/1000, .3912951704721983, -.2, (0., 0., -.245)
g = np.load(args.foam/'dem_exact_single'/'geometry.npz'); cl = np.load(args.foam/'dem_exact_single'/'cells.npz')
X0, r0, V0 = g['points'], cl['radii'], g['masses']; n = len(X0)
args.save.mkdir(parents=True, exist_ok=True)
mass = torch.as_tensor(V0, dtype=f64, device=dev); M = float(mass.sum()); M3 = mass[:, None]
up = torch.tensor([0., 0., 1.], dtype=f64, device=dev)
W = (RingMoment if args.moment else RingYield)(X0, r0, V0, g['pairs'], g['areas'], g['contacts'], E, nu, 1e9, theta=.5, kappa=5., eps_cut=args.eps_cut)
W.set_ball((0., 0., 1.), C0, 0.)
if args.beta > 0:
    W.set_viscosity(args.beta, args.dt)
W.start()
if args.colour > 0:
    W.set_colour_contact(args.colour)
    if args.jump:
        W.set_jump_share()
    print('colour contact: sigma %g, M %.3f, %s' % (args.colour, W.Mp, 'the jump share, no weightless springs' if args.jump else 'births at similarity >= %g' % W.sim_birth), flush=True)
if args.ground_cut:
    W.set_ground_cut(C0)
    print('the ground in the cut at z = %g (mirror sites), one-sided faces; no Hertz ground' % C0, flush=True)
W.prepare()
fg = -args.g*M3*up; wref = float((9.8*M3*up).norm()); tol = args.tol*wref
wtot = 9.8*M                                                                         # (pushes over the whole weight; wref, the sites' weights' norm, about it / sqrt(n), scales the gradient's tolerance)
CG = TrustCG(W, mass, args.dt, per=2, product_iters=1)
W.update_ul(W.X)
W.grad(W.X); W.reread(); W.grad(W.X)
ST = YieldStep(W, CG, mass, W.ul_blocks(), fg, args.dt, 0.)
ST.pre_ul = True; ST.pre_every = args.pre_every; ST.smooth_pred = ST.carry_delta = True
ST.set_theta(.5, mass)
ST.aa_m = args.aa
ST.hold_after = args.hold_after
W.true_blocks = args.true_blocks
ST._pre_age = 0
W.rest_ground = args.rest_ground
Es0_ = float(W.Es)


def ground_k():
    W.Es = Es0_*args.ground_k; W.tball[3] = W.Es
ground_k()
ST.set_state(W.X.clone(), torch.tensor(v0, dtype=f64, device=dev).expand(n, 3).clone(), fg/M3)
if args.no_elastic_births:
    W.elastic_births = False
s_first = 1
if args.start_frames is not None:
    Z_ = np.load(args.start_frames); k_ = args.start_k
    W._Ft = torch.as_tensor(Z_['F'][k_], device=dev, dtype=f64); W._Fe = W._Ft.clone()
    W.tul['Ln'].copy_(torch.as_tensor(Z_['L'][k_], device=dev, dtype=f64))
    x_ = torch.as_tensor(Z_['x'][k_], device=dev, dtype=f64)
    v_ = (torch.as_tensor(Z_['x'][k_+1], device=dev, dtype=f64)-torch.as_tensor(Z_['x'][k_-1], device=dev, dtype=f64))/(float(Z_['t'][k_+1])-float(Z_['t'][k_-1]))
    ST.set_state(x_, v_, fg/M3)
    s_first = int(round(float(Z_['t'][k_])/args.dt))+1
    print('restart at step %d' % (s_first-1), flush=True)
H0 = .5*float((M3*ST.tv**2).sum())+float((-fg*ST.tx).sum())
h = float(np.median(V0))**(1/3)
print('lion, ring springs on the cut: %d cells, h %.2f mm; E / rho %.3f, nu %.4f, beta %g, dt 1/%d, tol %g of the weight; ground z = %g, frictionless; the start\'s energy %.5f' % (n, 1e3*h, E, nu, args.beta, round(1/args.dt), args.tol, C0, H0), flush=True)
xs, Fs, Ls, ts = [W.X.cpu().numpy()], [np.tile(np.eye(3), (n, 1, 1))], [np.zeros(n)], [0.]


def save():
    np.savez(args.save/('frames_%s.npz' % args.tag), x=np.array(xs), F=np.array(Fs), L=np.array(Ls), t=np.array(ts))


from free_surface.run_state import pack, unpack
r0_t = torch.as_tensor(r0, dtype=f64, device=dev)


def rest_radii():
    """after a state's load: the rest radii from the foam (a checkpoint of a restarted run before 11-02 holds its cut radii there: _rr0 and _ulr were one tensor), the jump share's site colours"""
    W._rr0.copy_(r0_t)
    if getattr(W, 'jump_share', False):
        W._site_blend = W._blend(W.X)
    if getattr(W, 'selfc', None) is not None:                                              # (the contact list's capacity is its arrays': a checkpoint's K of another capacity, 262144
        W.selfc['K'] = W.selfc['i'].shape[0]                                               #  before 11-03, read over 65536 arrays was an illegal access)


if args.probe_jump:
    from free_surface.ring_moment import color_code
    from free_surface.ring_moment_force import jump_check
    Qc_ = color_code(W, W._rr0, 1.19)
    G0_, _ = W._recut.geometry(W.X, W._rr0); k0_ = G0_.wall_A.detach() > 0
    rk_ = torch.unique(torch.minimum(G0_.i[k0_], G0_.j[k0_]).long()*n+torch.maximum(G0_.i[k0_], G0_.j[k0_]).long())
    for f_ in args.probe_jump:
        ck_ = torch.load(f_, weights_only=False)
        unpack(W, ck_['W']); unpack(ST, ck_['ST']); rest_radii()
        jump_check(W, ST, V0, '%s (step %d)' % (f_.name, ck_['s']), Qc_, rk_, G0_.patch_A.detach() > 0)
    raise SystemExit
if args.probe_30:
    from free_surface.ring_moment_force import check30
    dn_ = np.load(args.dense)
    W.elastic_births = not args.no_elastic_births
    for f_ in args.probe_30:
        ck_ = torch.load(f_, weights_only=False)
        unpack(W, ck_['W']); unpack(ST, ck_['ST']); rest_radii()
        check30(W, ST, mass, V0, X0, g['pairs'], dn_, ck_['s'], args.dt)
    raise SystemExit
dense_x, dense_s = [], []
if args.load_state is not None:
    ck_ = torch.load(args.load_state, weights_only=False)
    unpack(W, ck_['W']); unpack(ST, ck_['ST']); rest_radii()
    W.true_blocks = args.true_blocks; ST._pre_age = 0; ST.aa_m = args.aa; ST.pre_every = args.pre_every
    W.elastic_births = not args.no_elastic_births and not args.jump                       # (the command line's, not the checkpoint's)
    ground_k()
    fg.copy_(-args.g*M3*up); wp.to_torch(ST.a['fg'], requires_grad=False).copy_(-args.g*M3*up)      # (the command line's gravity over the checkpoint's: fg shares the array the state was just written into)
    print('gravity on the sites now: %.4e, asked %.4e' % (float(wp.to_torch(ST.a['fg'], requires_grad=False).sum(0)[2]), -args.g*float(M3.sum())), flush=True)
    if args.no_elastic_births:                                                              # (the checkpoint's weightless springs out: their walls are free surfaces)
        W.elastic_births = False
        keep_ = W._wi+W._wj > 0
        W.tlo, W.thi, W.tkey, W._wi, W._wj, W._qkey, W._qi, W._qj, W.tk0, W.tdl = W.tlo[keep_], W.thi[keep_], W.tkey[keep_], W._wi[keep_], W._wj[keep_], W._qkey[keep_], W._qi[keep_], W._qj[keep_], W.tk0[keep_], W.tdl[keep_]
    s_first = ck_['s']+1
    print('from step %d' % ck_['s'], flush=True)
if args.probe_colour:
    from free_surface.ring_moment_force import lock_colour
    lock_colour(W, ST, mass, V0, X0, g['pairs'], torch.as_tensor(r0, device=dev, dtype=f64))
    raise SystemExit
if args.probe_ground:
    from free_surface.ring_moment_force import ground_check
    ground_check(W, ST, V0)
    raise SystemExit
if args.probe_kernels:
    from free_surface.ring_moment_force import kernel_times
    kernel_times(W, ST, V0)
    raise SystemExit
if args.probe_contact:
    from free_surface.ring_moment_force import contact_check
    contact_check(W, ST, V0)
    raise SystemExit
if args.probe_unstable:
    from free_surface.ring_moment_force import unstable
    unstable(W, ST, mass, V0, X0, g['pairs'])
    raise SystemExit
if args.probe_lock:
    from free_surface.ring_moment_force import lock
    lock(W, ST, mass, V0, X0, g['pairs'], 'step %d' % ck_['s'] if args.load_state is not None else 'rest')
    raise SystemExit
if args.probe_equil:
    from free_surface.ring_moment_force import equil
    equil(W, ST, mass, V0, X0, g['pairs'])
    raise SystemExit
if args.probe_leg:
    from free_surface.ring_moment_force import probe, slices
    slices(W, ST, mass, V0, X0, g['pairs'])
    raise SystemExit
print('the ground: Hertz modulus %.3f (x %g), in the kernel %.3f' % (W.Es, args.ground_k, float(W.tball[3])), flush=True)
t0 = t_saved = time.time()
if args.cuda_profile:
    torch.cuda.synchronize(); torch.cuda.cudart().cudaProfilerStart()
nit, npr, nout = [], [], []
tbeg, tsol, rcg = [], [], [W._recut.stats['cg']]
nsteps = int(round(args.T/args.dt))
failed = False
for s in range(s_first, nsteps+1):
    tb_ = time.time(); ST.begin(); ts_ = time.time()                                       # (the step's start, the cut and the lists, apart from the solve)
    e_, its, g2 = ST.solve(tol, lambda g_: .5*tol)
    tbeg.append(ts_-tb_); tsol.append(time.time()-ts_); rcg.append(W._recut.stats['cg'])
    if args.trace_solves:
        tr_ = ST.stats['trace']; parts_ = []
        for o_ in sorted(set(a__[7] for a__ in tr_)):
            q_ = [a__ for a__ in tr_ if a__[7] == o_]
            parts_.append('o%d: |g| %.1f -> %.1f tol, %d solves, products %s, edge %d, neg %d, refused %d' % (
                o_, q_[0][4]/tol, q_[-1][4]/tol, len(q_), '+'.join(str(a__[0]) for a__ in q_), sum(a__[1] for a__ in q_), sum(a__[2] for a__ in q_), sum(not a__[6] for a__ in q_)))
        print('trace step %d (%.1f s, %d products): ' % (s, tsol[-1], e_)+'; '.join(parts_), flush=True)
    if not math.isfinite(g2) or math.sqrt(g2) > tol:
        tr_ = ST.stats['trace']
        gz = wp_t(ST.a['gz']).norm(dim=1); top = torch.argsort(gz, descending=True)[:6]
        gl = W._gid == torch.arange(n, device=dev)
        Y_ = W.ty
        bad = {k_: int((~torch.isfinite(Y_[k_][gl]).all(dim=(1, 2))).sum()) for k_ in ('Fn', 'Ff', 'Mt', 'Mti', 'Mtt', 'Kt', 'Fni', 'Ms', 'Vz', 'Rz', 'Rn', 'Hn', 'Cz')}
        badx = {k_: int((~torch.isfinite(v_).all(dim=1)).sum()) for k_, v_ in (('x', ST.tx), ('z', ST.tz), ('g', wp_t(ST.a['gz'])))}
        bo = torch.nonzero(gl & ~torch.isfinite(Y_['Ff']).all(dim=(1, 2)))[:, 0]
        sn = torch.linalg.svdvals(torch.nan_to_num(Y_['Fn'][gl], nan=0., posinf=0., neginf=0.).cpu())
        print('step %d, t %.4f: the solve did not converge (|g| / weight %s, %d outer, %d inner); |g| %s; F_n smallest min %.3f, largest max %.3f; non-finite, groups: %s, sites: %s; groups with a bad fit %s, V0 / median %s; largest |g| cells %s, V0 / median %s' % (s, s*args.dt, math.sqrt(g2)/wref if math.isfinite(g2) else g2, ST.stats['outer'], its, ['%.3g' % (a__[4]/wref) for a__ in tr_][-20:], float(sn[:, 2].min()), float(sn[:, 0].max()), bad, badx, bo[:12].tolist(), np.array2string(V0[bo[:12].cpu().numpy()]/np.median(V0), precision=3), top.tolist(), np.array2string(V0[top.cpu().numpy()]/np.median(V0), precision=3)), flush=True)
        print('  energy terms at the failure: %s; |g| / weight over the first evaluations %s' % (np.array2string(W.tsum.detach().cpu().numpy(), precision=3), ['%.3g' % (a__[4]/wref) for a__ in tr_][:8]), flush=True)
        if W.ground_cut:
            Gv_ = W._ulG; Vt_ = W.V0*torch.exp(W.tul['Ln']); V_ = Gv_.V.detach(); gi_ = Gv_.g_i.long()
            hasg_ = torch.zeros(n, dtype=torch.bool, device=dev); hasg_[gi_] = True
            Ag_ = torch.zeros(n, dtype=f64, device=dev).index_add(0, gi_, Gv_.g_A.detach())
            ht_ = (W.tul['xn']@W._recut.ng-W._recut.c0)/h
            rat_ = V_/Vt_; k_ = int(torch.argmin(rat_))
            Gn_ = wp.to_torch(W.B['G'], requires_grad=False).norm(dim=(1, 2))
            print('  ground at the failure: %d faces on %d cells, areas %.2e..%.2e; V / V0 e^L min %.3e (cell %d, ground face %s, height %.3f h), max %.3e; cells under .5 %d; |G| max %.3e (cell %d)' % (
                len(gi_), int(hasg_.sum()), float(Gv_.g_A.min()) if len(gi_) else 0., float(Gv_.g_A.max()) if len(gi_) else 0., float(rat_[k_]), k_, bool(hasg_[k_]), float(ht_[k_]), float(rat_.max()),
                int((rat_ < .5).sum()), float(Gn_.max()), int(torch.argmax(Gn_))), flush=True)
            for c_ in top.tolist():
                print('    cell %d: |g| / weight %.3e, ground face %s (area / V^(2/3) %.3f), height %.3f h, radius %.3f h, V / V0 e^L %.3f, |G| %.3e' % (
                    c_, float(gz[c_])/wref, bool(hasg_[c_]), float(Ag_[c_]/V_[c_]**(2/3)), float(ht_[c_]), float(W._ulr[c_])/h, float(rat_[c_]), float(Gn_[c_])), flush=True)
        np.savez(args.save/('fail_%s.npz' % args.tag), x=ST.tx.cpu().numpy(), v=ST.tv.cpu().numpy(), a=ST.ta.cpu().numpy(), z=ST.tz.cpu().numpy(), step=s)
        save()
        failed = True
        break
    if ST.stats.get('aa_nan', 0):
        print('AA step %d: the extrapolation read NaN %d times, back to the plain point' % (s, ST.stats['aa_nan']), flush=True)
    ST.end()
    if args.watch_ground and W.ground_cut:                                                  # (after end: the step's end x against its start's list)
        ng_, c0_ = W._recut.ng, W._recut.c0
        h1_ = (ST.tx@ng_-c0_)/h
        gl_ = W.tgc['i'][W.tgc['Kc'] > 0].long()
        lam_ = (ST.tx[gl_]@ng_-c0_)/W.tgc['Rs'][W.tgc['Kc'] > 0]
        k3_ = torch.argsort(h1_)[:3]
        print('watch step %d: under the plane %d, under .05 h %d; lowest %s h (cells %s, a face in the cut %s); lambda at the end: min %.3f, under .5 %d, under .2 %d; push / weight %.2f' % (
            s, int((h1_ < 0).sum()), int((h1_ < .05).sum()), np.array2string(h1_[k3_].cpu().numpy(), precision=4), k3_.tolist(), W._g_face[k3_].tolist(),
            float(lam_.min()) if len(lam_) else 1., int((lam_ < .5).sum()), int((lam_ < .2).sum()), float(W.tgc['fn'].sum())/wtot), flush=True)
    if args.dense_from and s >= args.dense_from:
        dense_x.append(ST.tx.cpu().numpy()); dense_s.append(s)
        np.savez(args.save/('dense_%s.npz' % args.tag), x=np.array(dense_x), s=np.array(dense_s))
    if s in args.save_state:
        torch.save(dict(W=pack(W), ST=pack(ST), s=s), args.save/('state_%s_%d.pt' % (args.tag, s)))
    nit.append(its); npr.append(e_); nout.append(ST.stats['outer'])
    if s % args.every == 0:
        xs.append(ST.tx.cpu().numpy()); Fs.append(W._Ft.cpu().numpy()); Ls.append(W.tul['Ln'].cpu().numpy()); ts.append(s*args.dt)
        ek = .5*float((M3*ST.tv**2).sum()); ed, ev = W.stored(); eg = float((-fg*ST.tx).sum())
        wt_ = float(W.tball[1])                                                             # (the step's objective holds wt E of a term read at the step's end, wt = theta: the stored energy is E;
        e6_ = float(W.tgc['E'].sum()) if W.ground_cut else float(W.tsum[6])/wt_            # before 11-03 the balance took wt E, so a landing read half its ground's and contacts' energy)
        e7_ = float(W.tsc['E'].sum()) if W.selfc is not None else 0.
        Lm = W.tul['Ln']
        sf = torch.linalg.svdvals(W._Ft.cpu())
        bd = W.B['bdelta'].numpy()
        if W.selfc is not None:
            ss_ = W.self_stats
            print('  contact: energy %.3e, pairs within reach %d, overlapping at the step\'s start %d, the deepest %.3f of rho_i + rho_k, push / weight %.3f' % (e7_, ss_['pairs'], ss_['active'], ss_['d_max'], float(W.tsc['fn'].sum())/wtot), flush=True)
            if W.ground_cut:
                gs_ = W.ground_stats
                print('  ground: faces in the cut %d, area %.4f m^2 (walls to another cell\'s mirror, largest area %.2e); cells against their images %d, pressing at the step\'s start %d, lambda min %.3f; push / weight %.3f' % (
                    gs_['faces'], gs_['area'], gs_.get('other_A_max', 0.), gs_['near'], gs_['pushing'], gs_['lam_min'], float(W.tgc['fn'].sum())/wtot), flush=True)
            if W.jump_share:
                us_ = W.ul_stats
                print('  share: walls without a spring %d, over .5 %d, between .1 and .9 %d, mean %.3f' % (us_.get('share_n', 0), us_.get('share_hi', 0), us_.get('share_mid', 0), us_.get('share_mean', 0.)), flush=True)
        print('t %.4f (%.0f s): KE %.3e springs+outer %.3e vol %.3e ground %.3e W_v %.3e, balance / start %.4f; com up %.5f, lowest %.4f; e^L %.3f..%.3f; F_e singular %.3f..%.3f; touching %d, ground/weight %.2f; walls %d, new %d, closed %d; radii %d (fails %d), flips %d; last %d steps: products %d, outer median %d max %d, rejected %d; the starts %.1f s, the solves %.1f s, radii CG iterations %d' % (
            s*args.dt, time.time()-t0, ek, ed, ev, e6_, W.Wv, (ek+ed+ev+e6_+e7_+W.Wv+eg)/H0, float((M3*ST.tx)[:, 2].sum())/M, float(ST.tx[:, 2].min()), float(torch.exp(Lm).min()), float(torch.exp(Lm).max()), float(sf[:, 2].min()), float(sf[:, 0].max()),
            int((bd > 0).sum()), float(W.B['bfn'].numpy().sum())/wtot, W.ul_stats.get('walls', 0), W.ul_stats.get('born', 0), W.ul_stats.get('closed', 0), W.ul_stats.get('radii_it', 0), W.ul_stats.get('radii_fail', 0), W.ul_stats.get('flips', 0), args.every, sum(npr[-args.every:]), int(np.median(nout[-args.every:])), max(nout[-args.every:]), ST.stats['rejected'], sum(tbeg[-args.every:]), sum(tsol[-args.every:]), rcg[-1]-rcg[max(len(rcg)-1-args.every, 0)]), flush=True)
        if time.time()-t_saved > 60:
            save(); t_saved = time.time()
if args.cuda_profile:
    torch.cuda.synchronize(); torch.cuda.cudart().cudaProfilerStop()
save()
ni, no, np_ = np.array(nit), np.array(nout), np.array(npr)
print('steps %d of %d; outer per step: median %d, 99 %% %d, max %d; products per step: median %d, max %d, total %d' % (len(ni), nsteps, np.median(no), np.percentile(no, 99), no.max(), np.median(np_), np_.max(), np_.sum()), flush=True)
print('done: %.0f s' % (time.time()-t0), flush=True)
if failed:                                                                           # (a run that stopped on a solve that did not converge exits 1, for a remote job's status)
    raise SystemExit(1)
