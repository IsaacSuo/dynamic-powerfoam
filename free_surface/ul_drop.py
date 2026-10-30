"""The jack's drop with the updated-Lagrangian Power cells (ul_model.ULModel, 2026-10-27), set up as hertz_drop.py: the 8k
foam at rest with its lowest ball point --height above y = 0, gravity -y, frictionless Hertz ground on the balls of the
cells with an exposed cap (radius this step's), backward Euler dt 1/1200, tol 1e-3 of |m g|, 0.5 s, no viscosity.
Records every 1/240 s in mpm_compare.py's columns (t, com_y, com_vy, K, elastic, contact/weight) and more: the total
energy's change (kinetic + elastic + ground + gravity), walls born / died and the elastic energy's jump at the recuts
since the last record, the solve's and the volume fit's work. --up / --ground / --v0 (10-28, the lion as hertz_drop.py's
lion runs): the up axis, the foam left in place above the plane at --ground on it, a starting velocity.
10-28 (the lion runs lost to a timeout and to missing frames): every record's state (sites, F, radii, free columns) goes
to <tag>_frames/fNNNN.npz and the records to <tag>.npz as they come; the energy account per record (ULModel.step
account): within the steps backward Euler's kinetic dissipation, the Bregman remainder and the residual's work; at the
recuts the elastic energy's jump split into the cells (0 for the rest fits), the stabilisation, the jump term of the
walls kept / born / gone, and the ground energy's into the balls kept (their radii refitted) / joining / leaving the
contact set; the account's closure against H. A solve that stops above --tol stops the run (the states saved).
  PYTHONPATH=<main repo>:<worktree> python free_surface/ul_drop.py --foam <dir with dem_exact_single> --out <dir>"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from free_surface.ul_model import GraphedProduct, ULModel, cell_moments_grid

f64, dev = torch.float64, 'cuda'


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument('--foam', type=Path, default=Path('output/round01_pruned_exact'))
    a_.add_argument('--out', type=Path, default=Path('output/round01_pruned_exact'))
    a_.add_argument('--height', type=float, default=.02)
    a_.add_argument('--dt', type=float, default=1/1200)
    a_.add_argument('--T', type=float, default=.5)
    a_.add_argument('--start', type=Path, default=None, help='a <tag>_frames directory: start from its frame nearest --start-t (x, F, r; v from the '
                    'frame before it), for solver tests at a given state (10-29)')
    a_.add_argument('--start-t', type=float, default=0.)
    a_.add_argument('--steps', type=int, default=0, help='> 0: this many steps instead of T / dt')
    a_.add_argument('--tol', type=float, default=1e-3)
    a_.add_argument('--maxit', type=int, default=5000, help="the step solve's product budget (a solve above --tol at it stops the run)")
    a_.add_argument('--E', type=float, default=23008.38)
    a_.add_argument('--nu', type=float, default=.3008)
    a_.add_argument('--grid', type=float, default=.002, help="the cells' centroids and second moments: sample spacing (m)")
    a_.add_argument('--contact-radius', choices=('current', 'rest'), default='current',
                    help="current: the volume fit's radii; rest: the rest radii (10-27 test of the contact / volume-fit feedback)")
    a_.add_argument('--squeeze', type=float, default=0., help='> 0: no ground, no gravity; the foam starts at rest in place with the '
                    'velocity v_y = -a (y - y_c) (a smooth compression; 10-28 test against the drop\'s contact)')
    a_.add_argument('--f-mode', default='total', help="ul_model f_mode: 'increment' or 'total' (UPDATED_LAGRANGE.md section 20)")
    a_.add_argument('--new-ref', default='rest', help="f_mode 'total': a newly met pair's reference, 'contact' or 'rest'")
    a_.add_argument('--fit-w', default='rest', help="f_mode 'total': the fits' pairs and weights, 'wall' (current cut), 'rest' (rest neighbours and weights) or 'lens'")
    a_.add_argument('--up', type=int, default=1, help='the up axis (0, 1, 2); gravity along -up')
    a_.add_argument('--ground', type=float, default=None, help='the plane at this up coordinate, the foam left in place (else the '
                    'plane at 0 and the foam moved to its lowest ball point --height above it)')
    a_.add_argument('--v0', type=float, nargs=3, default=(0., 0., 0.), help='the starting velocity (m/s)')
    a_.add_argument('--tet', type=float, default=0., help='> 0: the site tetrahedra\'s volumetric Bregman term with this weight (ul_model tet_w, 10-29)')
    a_.add_argument('--mass', default='diag', help="ul_model mass: 'diag' (the sites' own cells) or 'consistent' (the kinetic energy of the model's field, 10-29)")
    a_.add_argument('--shift', default='displacement', help="ul_model shift: 'displacement' (two-step fit, carried to the centroid) or 'none' (site-level fit)")
    a_.add_argument('--agg', type=float, default=0., help='> 0: cells below this x the median volume take their neighbours\' fit (ul_model agg, 10-29)')
    a_.add_argument('--beta', type=float, default=0., help='Kelvin-Voigt time (s), the DEM\'s: eta = beta mu, zeta = beta (lam + 2 mu / 3) (10-29)')
    a_.add_argument('--self-contact', default=None, choices=['air', 'compat', 'cap', 'skin', 'bond', 'rest', 'arc', 'color'], help='contact walls (air: grown in air or out of a contact interface; '
                    'compat: the pair\'s offset not explained by either side\'s deformation from rest) out of the '
                    'material terms, Hertz across them (ul_model self_contact, 10-29)')
    a_.add_argument('--sc-birth', type=int, default=0, help='1: the self-contact Hertz overlap counted from each contact wall\'s birth')
    a_.add_argument('--sc-eps', type=float, default=.25, help='self-contact bond: the visibility tolerance, x the median cell size')
    a_.add_argument('--cut-bonds', type=int, default=0, help='1: the recut over the bond family only (rest walls + pairs within --bond-delta h at rest, '
                    'visible through the rest body); the other overlapping balls are the contact (ul_model cut_bonds, 10-29)')
    a_.add_argument('--bond-delta', type=float, default=4.)
    a_.add_argument('--bond-floor', type=float, default=0., help='> 0: the rest pairs keep this x their rest jump stiffness at least, also once their wall is gone (ul_model bond_floor, 10-29)')
    a_.add_argument('--bond-eps', type=float, default=.25)
    a_.add_argument('--sc-memory', type=int, default=0, help='1: a pair ever material (rest walls included) is material again when its wall comes back (ul_model sc_memory, 10-30)')
    a_.add_argument('--sc-sigma', type=float, default=0., help='self-contact color: > 0, skin colours change gradually over this many steps on the rest skin (ul_model _color_code, 10-30)')
    a_.add_argument('--sc-thr', type=float, default=2.5, help='self-contact compat threshold, x the median cell size')
    a_.add_argument('--jump-n', default='current', choices=['current', 'start'], help='the jump term\'s normal: along the pair now, or at the step\'s cut (ul_model jump_n, 10-29)')
    a_.add_argument('--precond', default='scalar', choices=['scalar', 'visc', 'block'], help='the CG\'s preconditioner (ul_model, 10-29)')
    a_.add_argument('--grad', default='fit', help="ul_model grad: 'fit' or 'corrected' (the fit with the integration constraint, 10-29)")
    a_.add_argument('--fit-cap', type=float, default=None, help="ul_model fit_cap (the current-cut fits' conditioning cap, 10-29)")
    a_.add_argument('--carry-cap', type=float, default=None, help='ul_model carry_cap (the carried displacement\'s sensitivity cap, 10-29)')
    a_.add_argument('--f-substeps', type=int, default=1, help="ul_model f_substeps (f_mode 'increment': F updated in this many substeps, 10-29)")
    a_.add_argument('--tag', default='ul_drop')
    args = a_.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    g = np.load(args.foam/'dem_exact_single'/'geometry.npz'); cl = np.load(args.foam/'dem_exact_single'/'cells.npz')
    X, r, V = g['points'], cl['radii'], g['masses']; n = len(X)
    cf = args.out/('ul_moments_%g.npz' % args.grid)
    if cf.exists():
        c = np.load(cf); dlt, J0 = c['dlt'], c['J0']
    else:
        dlt, J0, cnt = cell_moments_grid(X, r, V, args.grid); np.savez(cf, dlt=dlt, J0=J0, cnt=cnt)
    E, nu = args.E/1000, args.nu
    mdl = ULModel(X, r, V, dlt, J0, E, nu, f_mode=args.f_mode, new_ref=args.new_ref, fit_w=args.fit_w, tet_w=args.tet, mass=args.mass, grad=args.grad,
                  fit_cap=args.fit_cap, carry_cap=args.carry_cap, f_substeps=args.f_substeps, shift=args.shift, agg=args.agg, beta=args.beta, precond=args.precond, jump_n=args.jump_n, self_contact=args.self_contact, sc_thr=args.sc_thr, sc_birth=bool(args.sc_birth), sc_eps=args.sc_eps, cut_bonds=bool(args.cut_bonds), bond_delta=args.bond_delta, bond_eps=args.bond_eps, bond_floor=args.bond_floor, sc_memory=bool(args.sc_memory), sc_sigma=args.sc_sigma)
    if args.tet > 0:
        print('site tetrahedra: %d, their volume / the body %.3f, weight %g' % (len(mdl.tets), float(mdl.tet_V.sum()/mdl.V0.sum()), args.tet), flush=True)
    k_ = args.up; e_up = tuple(float(a == k_) for a in range(3))
    up = torch.tensor(e_up, dtype=f64, device=dev)
    I = torch.eye(3, dtype=f64, device=dev).expand(n, 3, 3).contiguous()
    low = float((mdl.X0[:, k_]-mdl.r0).min())
    gc = 0. if args.ground is None else args.ground
    x0 = mdl.X0+(args.height-low)*up if args.ground is None else mdl.X0.clone()
    st = dict(x=x0, v=torch.tensor(args.v0, dtype=f64, device=dev).expand(n, 3).clone(), F=I.clone(), r=mdl.r0.clone())
    ground = (e_up, gc, E/(1-nu**2))
    M = float(mdl.m.sum()); gacc = 9.8; ref = gacc*M
    if args.grad == 'corrected':
        print('corrected operator: %s' % {k: round(v, 6) if isinstance(v, float) else v for k, v in mdl.cor_stats.items()}, flush=True)
    if args.mass == 'consistent':
        print('consistent mass: total %.6e (sum V %.6e), lumped masses min / median V %.3f, Mc eigenvalue min / median V %.2e' % (
            M, float(mdl.V0.sum()), float(mdl.m.min()/mdl.V0.median()), float(torch.linalg.eigvalsh(mdl.Mc)[0]/mdl.V0.median())), flush=True)
    if args.squeeze > 0:
        yc = float((mdl.m*st['x'][:, k_]).sum())/M
        st['v'] = torch.zeros_like(st['v']); st['v'][:, k_] = -args.squeeze*(st['x'][:, k_]-yc); ground = None; gacc = 0.
    T0 = 0.
    if args.start is not None:                                                     # (10-29: a saved frame as the start)
        fs = sorted(args.start.glob('f*.npz')); ts_ = np.array([float(np.load(f_)['t']) for f_ in fs]); kf = int(np.argmin(np.abs(ts_-args.start_t)))
        z1, z0 = np.load(fs[kf]), np.load(fs[kf-1]); T0 = float(z1['t'])
        tt = lambda a: torch.as_tensor(a, dtype=f64, device=dev)
        st = dict(x=tt(z1['x']), v=(tt(z1['x'])-tt(z0['x']))/(T0-float(z0['t'])), F=tt(z1['F']), r=tt(z1['r']))
        print('start: %s at t %.4f s (v from the frame at %.4f s)' % (fs[kf], T0, float(z0['t'])), flush=True)
    hgt = lambda x: x[:, k_]-float((x[:, k_]-mdl.r0).min())                          # a site's height above the lowest rest-radius ball point
    dt = args.dt; nst = args.steps if args.steps > 0 else int(round(args.T/dt)); every = max(1, round(1/(240*dt)))
    print('%s: %d cells, lowest ball point %.4f m above the plane, v0 %s m/s, up axis %d; E %.1f, nu %.4f, E* %.3f (per unit density); dt 1/%.0f, '
          'tol %.0e of the weight, %d steps' % (args.foam.name, n, float((st['x'][:, k_]-mdl.r0).min())-gc, list(args.v0), k_, args.E, nu, E/(1-nu**2),
                                                1/dt, args.tol, nst), flush=True)
    cols = ['t', 'com_y', 'com_vy', 'K', 'elastic', 'contact/weight', 'touching', 'overlap/R', 'min J', 'H change', 'born', 'died', 'recut jump',
            'products', 'Newton', 'radii it', 'step s', 'V/target lo', 'V/target hi', 'vol err', 'V/target p1', 'V/target p99', 'off share',
            'off height']
    cols += ['diss K', 'Bregman', 'res work', 'jump cells', 'jump stab', 'jump kept', 'jump born', 'jump gone', 'ground kept', 'ground in',
             'ground out', 'self-contact', 'viscous', 'closure']
    rec = []
    fdir = args.out/(args.tag+'_frames'); fdir.mkdir(exist_ok=True)
    def frame(k, t, st_):
        sck = mdl.sc_prev['keys'][mdl.sc_prev['contact']].cpu().numpy() if mdl.sc and mdl.sc_prev is not None else (
            mdl.sc_keys_last.cpu().numpy() if getattr(mdl, 'sc_keys_last', None) is not None else np.zeros(0, np.int64))
        np.savez(fdir/('f%04d.npz' % k), sck=sck, t=t, x=st_['x'].cpu().numpy(), r=st_['r'].cpu().numpy(), F=st_['F'].cpu().numpy().astype(np.float32),
                 cols=st_['cols'].cpu().numpy() if 'cols' in st_ else np.zeros((0, 3)))
    frame(0, T0, dict(st, cols=mdl.deg_q))
    def split(k_new, e_new, k_old, e_old):
        """a per-item energy's change between two lists keyed by k: the kept items' change, the new ones', minus the gone ones'"""
        o = torch.argsort(k_old); ko = k_old[o]
        pos = torch.searchsorted(ko, k_new).clamp(max=max(len(ko)-1, 0))
        found = (ko[pos] == k_new) if len(ko) else torch.zeros_like(k_new, dtype=torch.bool)
        return (float((e_new[found]-e_old[o][pos[found]]).sum()), float(e_new[~found].sum()), -float(e_old[~torch.isin(k_old, k_new)].sum()))
    ekeys = ('dK', 'B', 'res', 'cells', 'stab', 'kept', 'born', 'gone', 'gkept', 'gin', 'gout', 'selfc', 'visc', 'closure')
    eacc = dict.fromkeys(ekeys, 0.); prev = None
    com = lambda x: float((mdl.m*x[:, k_]).sum())/M
    H0 = gacc*float((mdl.m*st['x'][:, k_]).sum())+mdl.kinetic(st['v']); v_init = st['v'].clone(); U_prev = None; keys_prev = None; acc = dict(born=0, died=0, jump=0.)
    t0 = time.time()
    sacc = dict(born=0, overlap=0., last=None)
    tkeys = ('t_geometry', 't_cg', 'geometry', 'cg')
    tacc = dict(ops=0., solve=0., radii=0., products=0, hvp=0., eg=0., newton=0, rej=0, inf=0, neg=0, edge=0, caps=GraphedProduct.captures, **{k__: mdl.recut.stats[k__] for k__ in tkeys})
    hist = []
    def fail(s, msg):
        print('step %d (t %.4f s): %s; stopping' % (s, s*dt, msg), flush=True)
        dump = dict(steps=np.array([s-len(hist)+1+k_ for k_ in range(len(hist))]))
        for k_, h_ in enumerate(hist):
            for key in ('x', 'v', 'F', 'r', 'r_prev', 'cols'):
                if key in h_:
                    dump['s%d_%s' % (k_, key)] = h_[key].detach().cpu().numpy()
        np.savez(args.out/(args.tag+'_fail.npz'), **dump)
        print('the states entering steps %s saved to %s' % (dump['steps'].tolist(), args.out/(args.tag+'_fail.npz')), flush=True)
    H_prev = H0
    for s in range(1, nst+1):
        ts = time.time()
        hist = (hist+[st])[-2:]                                                        # the states entering the last two steps
        try:
            st, rep = mdl.step(st, dt, grav=tuple(-gacc*a for a in e_up), ref=ref, tol=args.tol, maxit=args.maxit, ground=ground,
                               ground_r=mdl.r0 if args.contact_radius == 'rest' else None, account=True)
            if ground is None:
                rep['ground'] = dict(force=0., touching=0, overlap=0., energy=0.)
        except RuntimeError as e_:
            fail(s, str(e_))
            break
        if rep['solve']['residual'] > args.tol:
            fail(s, 'the solve stopped at %.1e of the weight (%s)' % (rep['solve']['residual'], rep['solve']))
            break
        tacc['ops'] += rep['time']['operators']; tacc['solve'] += rep['time']['solve']; tacc['radii'] += rep['time']['radii']
        if 'sc' in rep:
            sacc['born'] += rep['sc']['born_contact']; sacc['overlap'] = max(sacc['overlap'], rep['sc']['overlap']); sacc['last'] = rep['sc']
        tacc['products'] += rep['solve']['products']; tacc['hvp'] += rep['solve']['t_hvp']; tacc['eg'] += rep['solve']['t_eg']; tacc['newton'] += rep['solve']['outer']
        tacc['rej'] += rep['solve']['rejected']; tacc['inf'] += rep['solve']['inf']; tacc['neg'] += rep['solve']['negative']; tacc['edge'] += rep['solve']['edge']
        q_ = rep['radii_start']; acc['qlo'] = min(acc.get('qlo', 1.), q_[0]); acc['qhi'] = max(acc.get('qhi', 1.), q_[1])
        acc['p1'] = min(acc.get('p1', 1.), q_[2]); acc['p99'] = max(acc.get('p99', 1.), q_[3]); acc['off'] = max(acc.get('off', 0.), q_[4])
        if len(q_[5]):
            acc['offh'] = float(hgt(st['x'])[q_[5]].median())/float(np.median(V))**(1/3)
        acc['verr'] = max(acc.get('verr', 0.), rep['vol_err'])
        i, j, _ = rep['walls']; kk = torch.minimum(i, j)*n+torch.maximum(i, j)
        if keys_prev is not None:
            acc['born'] += int((~torch.isin(kk, keys_prev)).sum()); acc['died'] += int((~torch.isin(keys_prev, kk)).sum())
        keys_prev = kk
        U0, U1 = sum(rep['e_start'].values()), sum(rep['e_end'].values())
        if U_prev is not None:
            acc['jump'] += U0-U_prev
        U_prev = U1
        K = mdl.kinetic(st['v'])
        H = K+U1+rep['ground']['energy']+gacc*float((mdl.m*st['x'][:, k_]).sum())
        if s == 1 and args.squeeze > 0:
            H0 = mdl.kinetic(v_init); H_prev = H0
        a_ = rep['acct']; eacc['dK'] -= a_['DK']; eacc['B'] -= a_['B']; eacc['res'] += a_['res']; eacc['visc'] -= a_.get('DV', 0.)
        across = a_['H0']-a_['K0']-a_['P0']                                            # U0 + G0 at the step's start
        if prev is not None:
            across -= prev['H1']-prev['K1']-prev['P1']
            eacc['selfc'] += a_['U0'].get('selfc', 0.)-prev['U1'].get('selfc', 0.)
            eacc['cells'] += a_['U0']['cells']-prev['U1']['cells']; eacc['stab'] += a_['U0'].get('stab', 0.)-prev['U1'].get('stab', 0.)
            kp, bn, gn_ = split(*a_['wall_start'], *prev['wall_end']); eacc['kept'] += kp; eacc['born'] += bn; eacc['gone'] += gn_
            if 'ball_start' in a_:
                kp, bn, gn_ = split(*a_['ball_start'], *prev['ball_end']); eacc['gkept'] += kp; eacc['gin'] += bn; eacc['gout'] += gn_
        else:
            eacc['selfc'] += a_['U0'].get('selfc', 0.)
            eacc['cells'] += a_['U0']['cells']; eacc['stab'] += a_['U0'].get('stab', 0.); eacc['kept'] += a_['U0'].get('jump', 0.)
            if 'ball_start' in a_:
                eacc['gin'] += float(a_['ball_start'][1].sum())
        eacc['closure'] += (H-H_prev)-(a_['H1']-a_['H0']+across)
        H_prev = H; prev = {k__: v__ for k__, v__ in a_.items() if k__ in ('H1', 'K1', 'P1', 'U1', 'wall_end', 'ball_end')}
        if s % every == 0 or s == nst:
            rec.append((T0+s*dt, com(st['x']), float((mdl.m*st['v'][:, k_]).sum())/M, K, U1, rep['ground']['force']/ref, rep['ground']['touching'],
                        rep['ground']['overlap'], rep['minJ'], (H-H0)/M, acc['born'], acc['died'], acc['jump']/M, rep['solve']['products'],
                        rep['solve']['outer'], rep['radii_it'], time.time()-ts, acc['qlo'], acc['qhi'], acc['verr'], acc['p1'], acc['p99'], acc['off'],
                        acc.get('offh', np.nan))+tuple(eacc[k__]/M for k__ in ekeys))
            acc = dict(born=0, died=0, jump=0.)
            q = rec[-1]
            print('t %.4f s (%.0f s): com up %.5f m, v up %+.4f m/s; ground / weight %.3f on %d balls, overlap max %.3f R; min J %.3f; '
                  'H change %+.4f J/kg; walls +%d -%d, recut jump %+.1e J/kg; products %d, Newton %d, radii %d (V / target before %.2f..%.2f, '
                  '1-99 %% %.2f..%.2f, share outside [0.5, 2] %.1e at median height %.1f h; after %.0e), %.2f s / step' % (
                      q[0], time.time()-t0, q[1], q[2], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[17], q[18], q[20],
                      q[21], q[22], q[23], q[19], q[16])+('; free columns |b| %.3f..%.3f' % (float(rep['cols'].min()), float(rep['cols'].max()))
                                                           if 'cols' in rep else '')+('; tetrahedra min J %.3f' % rep['tet_minJ'] if 'tet_minJ' in rep else '')+'; radii floor exits %d, overflowed trials %d (so far)' % (
                      mdl.recut.stats.get('floor', 0), mdl.recut.stats.get('overflow', 0)), flush=True)
            print('    energy (J/kg, since the last record): within the steps: kinetic dissipation %+.2e, Bregman %+.2e, residual work %+.1e; '
                  'at the recuts: elastic cells %+.1e, stabilisation %+.2e, jump term kept walls %+.2e, born %+.2e, gone %+.2e; ground kept balls '
                  '%+.2e, joining %+.2e, leaving %+.2e; self-contact at the recuts %+.2e; viscous dissipation %+.2e; account closure %+.1e' % tuple(eacc[k__]/M for k__ in ekeys), flush=True)
            eacc = dict.fromkeys(ekeys, 0.)
            rs_ = {k__: mdl.recut.stats[k__]-tacc[k__] for k__ in tkeys}
            if sacc['last'] is not None:
                print('    self-contact: %d contact walls of %d (new this record %d), largest overlap %.3f of the smaller radius, energy %.2e J/kg' % (
                    sacc['last']['contact'], sacc['last']['walls'], sacc['born'], sacc['overlap'], sacc['last']['energy']/M), flush=True)
                sacc = dict(born=0, overlap=0., last=None)
            print('    time (s, since the last record): operators %.2f, solve %.2f (%d products %.2f, %d Newton iterates\' energy and gradient %.2f, '
                  '%d graph captures; trials rejected %d (J <= 0 %d), negative curvature %d, region edge %d), radii %.2f (%d cuts %.2f, %d radii CG iterations %.2f; all cuts this record incl. the step start\'s)' % (
                      tacc['ops'], tacc['solve'], tacc['products'], tacc['hvp'], tacc['newton'], tacc['eg'], GraphedProduct.captures-tacc['caps'], tacc['rej'], tacc['inf'], tacc['neg'], tacc['edge'],
                      tacc['radii'], rs_['geometry'], rs_['t_geometry'], rs_['cg'], rs_['t_cg']), flush=True)
            tacc = dict(ops=0., solve=0., radii=0., products=0, hvp=0., eg=0., newton=0, rej=0, inf=0, neg=0, edge=0, caps=GraphedProduct.captures,
                        **{k__: mdl.recut.stats[k__] for k__ in tkeys})
            frame(len(rec), T0+s*dt, st)
            np.savez(args.out/(args.tag+'.npz'), cols=np.array(cols), rec=np.array(rec))
    R = np.array(rec)
    np.savez(args.out/(args.tag+'.npz'), cols=np.array(cols), rec=R)
    if len(R) == 0:
        print('no record', flush=True)
        return
    c = {k: R[:, i] for i, k in enumerate(cols)}
    on = np.flatnonzero(c['contact/weight'] > .01); t_on = c['t'][on[0]] if len(on) else np.nan
    imin = int(np.argmin(c['com_y'])); after = c['t'] > t_on
    print('total %.0f s for %d steps (recorded to t %.4f s); centre of mass lowest %.4f m at %.3f s; largest upward velocity %.3f m/s; at the end %.4f m, %+.3f m/s; '
          'peak ground force / weight %.2f at %.3f s; min J %.3f; products per record median %d, max %d' % (
              time.time()-t0, nst, c['t'][-1], c['com_y'][imin], c['t'][imin], c['com_vy'][after].max() if after.any() else np.nan, c['com_y'][-1], c['com_vy'][-1],
              c['contact/weight'].max(), c['t'][int(np.argmax(c['contact/weight']))], c['min J'].min(), np.median(c['products']), c['products'].max()), flush=True)


if __name__ == '__main__':
    main()
