"""Capacity diagnostics for the linearized FEM-target fit (one variant per run).

Variants, all on the same linearized implicit-midpoint dynamics:
  cell     per-cell log multipliers (as foam_fem_volume_inverse), chosen bound/init
  bond     independent log multipliers per bond and mode (+ per-cell k_v if --volume)
  general  any symmetric PSD stiffness on the free DOFs, K = L L^T (rigid-cell ceiling)
  inertia  general K plus free per-DOF log rotational inertia
  pair     any 12x12 PSD stiffness per bond (local, not necessarily objective)
  beam     Timoshenko beam bond: per-cell log multipliers of E and G (+ k_v), giving
           kn=EA/L, kt=GJ/L, kb=EI/L, ks=12EI/((1+Phi)L^3), Phi=12EI/(kappa G A L^2);
           this is the parallel bond with the beam-theory shear/bending relation
  lever    per-cell log multipliers (as cell) plus a learnable shift of each bond's
           contact point along the bond axis (--lever bond: one per bond; global: one
           shared). Equivalent to the only section-symmetric, non-chiral shear-bending
           coupling; keeps parallel-bond section symmetry.
  fem      standard linear-tetrahedron (P1) FEM on the tetrahedra joining cell centres
           (nodes = centres, rotations carry no stiffness); learns log(lambda), log(mu)
           relative to 400, globally (2 parameters). Same normalization c as the volume term.
  pairobj  any PSD stiffness per bond on its 6 non-rigid relative modes, i.e. the
           most general linear objective pair interaction (four-mode law is a case)

These are diagnostics. Held-out cases were already inspected; --cases all trains
on every case and is only a capacity probe, never a generalization result.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize

from experiments.foam_dem_moduli import write
from experiments.foam_fem_inverse import TRAIN, TEST
from experiments.foam_linear_inverse import LinearizedFoam, load_case, projected_gradient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, default=Path('output/foam_fem/reference_edge007'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--variant', choices=('cell', 'bond', 'beam', 'lever', 'fem', 'general', 'inertia', 'pair', 'pairobj'), required=True)
    parser.add_argument('--volume', action='store_true', help='Include the nodal tetra volume mode.')
    parser.add_argument('--volume-mode', choices=('tet', 'power'), default='tet',
                        help='tet: nodal average of centre tetrahedra; power: exact power-cell volume with air ghosts.')
    parser.add_argument('--init-x', type=Path, help='parameters.npz to start from (cell variant; a single shared vector is tiled).')
    parser.add_argument('--clamp-face', action='store_true',
                        help='Power terms: zero displacement on faces shared with fixed cells (Dirichlet face).')
    parser.add_argument('--shape-mode', choices=('tet', 'power'), default='tet',
                        help='Shape term strain: nodal tetrahedral average or exact power-cell gradient.')
    parser.add_argument('--deviatoric', action='store_true',
                        help='Include the nodal deviatoric (shape) multi-body mode as the last column (cell variant).')
    parser.add_argument('--bound', type=float, default=20., help='Multiplier bound (cell/bond).')
    parser.add_argument('--seed', type=int, default=-1, help='-1: zero init; else Uniform(-1,1) log init.')
    parser.add_argument('--cases', choices=('train', 'all', 'everything'), default='train',
                        help='everything: old six cases plus all --extra-reference cases (capacity probe only).')
    parser.add_argument('--max-iterations', type=int, default=10000)
    parser.add_argument('--gradient-tolerance', type=float, default=1e-7)
    parser.add_argument('--check-gradient', action='store_true')
    parser.add_argument('--cell-param', choices=('percell', 'uniform', 'fused', 'classes'), default='percell',
                        help='cell variant: per-cell multipliers; one shared set; or shared set plus per-cell '
                             'deviations under a fused-lasso (graph total variation + L1) penalty.')
    parser.add_argument('--fused-weight', type=float, default=1e-5)
    parser.add_argument('--init-scan', action='store_true',
                        help='Before fitting, shift every log stiffness by one common offset chosen on a grid '
                             'to minimize the training loss (frequency matching); training data only.')
    parser.add_argument('--fused-eps', type=float, default=1e-3, help='Smoothing of |z| as sqrt(z^2+eps^2)-eps.')
    parser.add_argument('--fit-cases', help='Comma-separated subset of training cases (e.g. leave-one-load-out).')
    parser.add_argument('--lever', choices=('bond', 'global', 'cell', 'geom', 'geomcell'), default='bond',
                        help="cell: per-cell size sigma, shift s=(sigma_i-sigma_j)/2 toward the smaller "
                             "cell, like a power-diagram weight; one parameter per cell for all directions. "
                             "geom: s = alpha * (V_i-V_j)/(V_i+V_j) * L, the direction fixed by cell-volume "
                             "asymmetry and one learned strength; geomcell: alpha_b=(alpha_i+alpha_j)/2.")
    parser.add_argument('--lever-mode', choices=('full', 'bend'), default='full',
                        help='bend: keep only the s^2 bending increase, drop the shear-bending coupling.')
    parser.add_argument('--lever-reg', type=float, default=0.,
                        help='Penalty lever_reg * mean(bond shift^2) / L^2: no evidence -> zero shift.')
    parser.add_argument('--shift-init', type=float, default=0.,
                        help='Initial contact shift / bond length (bend mode needs nonzero: its gradient vanishes at 0).')
    parser.add_argument('--section', choices=('circle', 'actual'), default='circle',
                        help='actual: bending/twist from the real power-face second moments (no new parameters).')
    parser.add_argument('--lever-bound', type=float, default=.4, help='Max contact shift / bond length.')
    parser.add_argument('--init-cell', type=Path, help='general/inertia: start from K of a finished cell run.')
    parser.add_argument('--evaluate-run', type=Path,
                        help='Evaluate a finished run (its config/parameters) on --extra-reference cases; no fitting.')
    parser.add_argument('--extra-reference', type=Path, help='Directory of extra target cases (e.g. fresh held-out).')
    args = parser.parse_args()
    if args.evaluate_run is not None:
        stored = json.loads((args.evaluate_run/'config.json').read_text())
        for key, value in stored.items():
            if key in ('evaluate_run', 'extra_reference', 'out', 'check_gradient', 'fit_cases', 'init_x') or not hasattr(args, key):
                continue
            setattr(args, key, Path(value) if key in ('reference', 'init_cell') and value is not None else value)
    with np.load(args.reference/'patch.npz') as a:
        geometry = dict(a)
    steps = dict(train_down=16, train_pull=32, train_twist=16)
    for name in TEST:
        steps[name] = max(steps.values())
    model = LinearizedFoam(geometry, volume=args.volume, deviatoric=args.deviatoric, volume_mode=args.volume_mode,
                           shape_mode=args.shape_mode, clamp_face=args.clamp_face)
    if args.section == 'actual':
        from experiments.foam_face_sections import face_sections
        sections = face_sections(geometry)
        equivalent = np.asarray(geometry['areas'])**2/(4*np.pi)  # what base kb, kt assume
        E = np.concatenate((np.zeros((3, 3)), -np.eye(3), np.zeros((3, 3)), np.eye(3)), axis=1)
        for b, section in enumerate(sections):
            normal = model.points[model.j[b]]-model.points[model.i[b]]
            normal /= np.linalg.norm(normal)
            model.unit[b, 2] = E.T@(section['bending']/equivalent[b])@E
            model.unit[b, 3] = E.T@np.outer(normal, normal)@E*section['polar']/(2*equivalent[b])
    n, nf, free = model.n, len(model.free), model.free
    fit_cases = list(TRAIN) if args.cases == 'train' else list(TRAIN+TEST)
    if args.fit_cases:
        fit_cases = args.fit_cases.split(',')
        assert set(fit_cases) <= set(TRAIN), 'fit-cases must be training cases'
    datasets = {name: load_case(args.reference, name) for name in TRAIN+TEST}
    report_cases = list(TRAIN+TEST)
    if args.cases == 'everything':
        if args.extra_reference is None or args.evaluate_run is not None:
            raise SystemExit('--cases everything needs --extra-reference (and no --evaluate-run).')
        extra_cases = json.loads((args.extra_reference/'config.json').read_text())['load_cases']
        for name in extra_cases:
            datasets[name] = load_case(args.extra_reference, name)
            steps[name] = 32
        fit_cases += extra_cases
        report_cases += extra_cases
    regularization = 1e-6
    bound = float(np.log(args.bound))
    rotation = np.flatnonzero(np.isin(free % 6, (3, 4, 5)))
    K0 = model.stiffness(np.zeros((n, model.parameter_count)))[free][:, free].toarray()
    if args.init_cell is not None:
        source = json.loads((args.init_cell/'report.json').read_text())
        assert source['variant'] == 'cell'
        donor = LinearizedFoam(geometry, volume=source['volume'])
        theta = np.load(args.init_cell/'parameters.npz')['x'].reshape(n, donor.parameter_count)
        K0 = donor.stiffness(theta)[free][:, free].toarray()
        donor_bonds = np.einsum('bm,bmij->bij', donor.bond_scale(theta), donor.unit)
    tril = np.tril_indices(nf)

    # Each variant: x0, bounds, and evaluate(x) -> (Kf, M, contract(adjoint) -> gradient).
    if args.variant == 'cell':
        P = model.parameter_count
        if args.cell_param == 'percell':
            shape = (n, P)
            cell_theta = lambda x: x.reshape(shape)
            cell_grad = lambda g: g.ravel()
        elif args.cell_param == 'uniform':
            shape = (P,)
            cell_theta = lambda x: np.tile(x, (n, 1))
            cell_grad = lambda g: g.sum(axis=0)
        elif args.cell_param == 'classes':
            # One parameter set per geometric class: number of lattice faces toward air
            # (0 interior, 1 face, 2 edge, 3 corner); classes absent from the specimen are dropped.
            pts = model.points
            axes = [np.unique(pts[:, k]) for k in range(3)]
            idx = np.stack([np.searchsorted(axes[k], pts[:, k]) for k in range(3)], axis=1)
            exposed = sum(((idx[:, k] == 0).astype(int)+(idx[:, k] == len(axes[k])-1).astype(int)) for k in range(3))
            present = np.unique(exposed)
            cls = np.searchsorted(present, exposed)
            C = len(present)
            shape = (C*P,)
            cell_theta = lambda x: x.reshape(C, P)[cls]
            def cell_grad(g):
                out = np.zeros((C, P))
                np.add.at(out, cls, g)
                return out.ravel()
        else:
            shape = (P+n*P,)
            cell_theta = lambda x: x[:P]+x[P:].reshape(n, P)
            cell_grad = lambda g: np.concatenate((g.sum(axis=0), g.ravel()))
        def fused_penalty(x):
            """lambda * (sum_bonds |d_i-d_j| + sum_i |d_i|), smoothed; d = per-cell deviations."""
            if args.cell_param != 'fused':
                return 0., np.zeros_like(x)
            d = x[P:].reshape(n, P)
            eps = args.fused_eps
            diff = d[model.i]-d[model.j]
            value = np.sum(np.sqrt(diff**2+eps**2)-eps)+np.sum(np.sqrt(d**2+eps**2)-eps)
            gd = d/np.sqrt(d**2+eps**2)
            gdiff = diff/np.sqrt(diff**2+eps**2)
            np.add.at(gd, model.i, gdiff)
            np.add.at(gd, model.j, -gdiff)
            grad = np.zeros_like(x)
            grad[P:] = gd.ravel()
            return args.fused_weight*value, args.fused_weight*grad
        bounds = [(-bound, bound)]*int(np.prod(shape))
    elif args.variant == 'bond':
        nb = len(model.i)
        shape = (nb*4+(n if args.volume else 0),)
        bounds = [(-bound, bound)]*shape[0]
    elif args.variant == 'fem':
        from experiments.foam_tetra_volume import NodalTetraVolume, grid_tetrahedra
        term = NodalTetraVolume(geometry['points'], grid_tetrahedra(geometry['points'], geometry['radii']),
                                float(np.sum(geometry['masses'])))
        vol, grads = (t.detach().numpy() for t in term.volumes_and_gradients(term.reference))
        tets = term.tetrahedra.numpy()
        c = float(term.normalization)  # material volume / hull volume, as in the volume term
        # Per tetrahedron: grad N_a = g_a / V; trace row b = sum_a g_a.u_a / V; strain in an
        # orthonormal symmetric basis E_m: e_m = sum_a (E_m g_a).u_a / V.
        basis = np.zeros((6, 3, 3))
        for m in range(3):
            basis[m, m, m] = 1
        for m, (a, b) in enumerate(((1, 2), (0, 2), (0, 1)), start=3):
            basis[m, a, b] = basis[m, b, a] = 2**-.5
        cols = (6*tets[:, :, None]+np.arange(3)).reshape(len(tets), 12)
        Bstrain = np.einsum('mpq,taq->tmap', basis, grads).reshape(len(tets), 6, 12)/vol[:, None, None]
        Btrace = grads.reshape(len(tets), 12)/vol[:, None]
        def assemble(blocks):
            rows = np.repeat(cols[:, :, None], 12, axis=2)
            return sp.coo_matrix((blocks.ravel(), (rows.ravel(), rows.transpose(0, 2, 1).ravel())), shape=(6*n,)*2).tocsr()
        # Energy = c sum_T V_T (lambda/2 tr(eps)^2 + mu |eps|^2): K = lambda K_lam + mu K_mu.
        K_lam = assemble(c*vol[:, None, None]*Btrace[:, :, None]*Btrace[:, None, :])[free][:, free].tocsc()
        K_mu = assemble(2*c*vol[:, None, None]*np.einsum('tmi,tmj->tij', Bstrain, Bstrain))[free][:, free].tocsc()
        fem_base = 400.
        shape = (2,)
        bounds = [(-bound, bound)]*2
    elif args.variant == 'lever':
        P = model.parameter_count
        nb = len(model.i)
        ns = dict(bond=nb, global_=1, cell=n, geom=1, geomcell=n)[args.lever.replace('global', 'global_')]
        volume = np.asarray(geometry['masses'], float)  # unit density: cell material volumes
        asymmetry = (volume[model.i]-volume[model.j])/(volume[model.i]+volume[model.j])
        shape = (n*P+ns,)
        normal = model.points[model.j]-model.points[model.i]
        length = np.linalg.norm(normal, axis=1)
        normal /= length[:, None]
        contact = np.asarray(geometry['contacts'], float)
        from experiments.foam_linear_inverse import skew
        eye = np.broadcast_to(np.eye(3), (nb, 3, 3))
        D0 = np.concatenate((-eye, skew(contact-model.points[model.i]), eye, -skew(contact-model.points[model.j])), axis=2)
        D1 = np.concatenate((0*eye, skew(normal), 0*eye, -skew(normal)), axis=2)  # d D / d shift
        tt = eye-normal[:, :, None]*normal[:, None, :]
        assert np.allclose(D0.transpose(0, 2, 1)@tt@D0, model.unit[:, 1]), 'shear unit mismatch'
        if args.lever in ('geom', 'geomcell'):
            if np.abs(asymmetry).max() == 0:
                raise SystemExit('No cell-volume asymmetry: geometric shift is identically zero.')
            limit = args.lever_bound/np.abs(asymmetry).max()  # keeps |s| <= lever_bound * L
        else:
            limit = args.lever_bound*length.min()
        bounds = [(-bound, bound)]*(n*P)+[(-limit, limit)]*ns
        def bond_shift(x):
            if args.lever == 'cell':
                sigma = x[n*P:]
                return .5*(sigma[model.i]-sigma[model.j])
            if args.lever == 'geom':
                return x[n*P]*asymmetry*length
            if args.lever == 'geomcell':
                alpha = x[n*P:]
                return .5*(alpha[model.i]+alpha[model.j])*asymmetry*length
            return x[n*P:]*np.ones(nb)
        def shift_to_parameters(gs):
            # Map d/d(bond shift) to d/d(lever parameters).
            if args.lever == 'bond':
                return gs
            if args.lever == 'global':
                return gs.sum(keepdims=True)
            if args.lever == 'geom':
                return np.sum(gs*asymmetry*length, keepdims=True)
            if args.lever == 'geomcell':
                gc = np.zeros(n)
                np.add.at(gc, model.i, .5*gs*asymmetry*length)
                np.add.at(gc, model.j, .5*gs*asymmetry*length)
                return gc
            gc = np.zeros(n)
            np.add.at(gc, model.i, .5*gs)
            np.add.at(gc, model.j, -.5*gs)
            return gc
        def lever_units(x):
            shift = bond_shift(x)
            D = D0+shift[:, None, None]*D1
            unit = model.unit.copy()
            if args.lever_mode == 'full':
                unit[:, 1] = D.transpose(0, 2, 1)@tt@D
                dshear = D1.transpose(0, 2, 1)@tt@D
                return unit, dshear+dshear.transpose(0, 2, 1)
            # Same s^2 bending increase, without the term linear in s (the coupling).
            bend = D1.transpose(0, 2, 1)@tt@D1
            unit[:, 1] = model.unit[:, 1]+shift[:, None, None]**2*bend
            return unit, 2*shift[:, None, None]*bend
    elif args.variant == 'beam':
        shape = (n, 3 if args.volume else 2)
        bounds = [(-bound, bound)]*int(np.prod(shape))
        area = np.asarray(geometry['areas'], float)
        length = np.linalg.norm(model.points[model.j]-model.points[model.i], axis=1)
        second = area**2/(4*np.pi)  # same equivalent circular section as LinearBondsGPU
        kappa = 6*(1+.25)/(7+6*.25)  # circular-section shear factor at nu=0.25
        def beam_stiffness(x):
            theta = x.reshape(shape)
            E = 1200.*np.exp(.5*(theta[model.i, 0]+theta[model.j, 0]))
            G = 480.*np.exp(.5*(theta[model.i, 1]+theta[model.j, 1]))
            phi = 12*E*second/(kappa*G*area*length**2)
            ks = 12*E*second/((1+phi)*length**3)
            k = np.stack((E*area/length, ks, E*second/length, G*2*second/length), axis=1)
            # d k / d (log E_b, log G_b) per bond, mode order kn, ks, kb, kt.
            dE = np.stack((k[:, 0], ks/(1+phi), k[:, 2], 0*ks), axis=1)
            dG = np.stack((0*ks, ks*phi/(1+phi), 0*ks, k[:, 3]), axis=1)
            return k, dE, dG
    elif args.variant in ('pair', 'pairobj'):
        # Rigid motions of a pair in local (u_i, w_i, u_j, w_j): translation and rotation.
        nb = len(model.i)
        if args.variant == 'pair':
            Q = np.broadcast_to(np.eye(12), (nb, 12, 12))
        else:
            Q = []
            for b in range(nb):
                rigid = np.zeros((12, 6))
                for k, point in enumerate((model.points[model.i[b]], model.points[model.j[b]])):
                    rigid[6*k:6*k+3, :3] = np.eye(3)
                    rigid[6*k:6*k+3, 3:] = -np.array([[0, -point[2], point[1]], [point[2], 0, -point[0]],
                                                     [-point[1], point[0], 0]])
                    rigid[6*k+3:6*k+6, 3:] = np.eye(3)
                basis = np.linalg.svd(rigid, full_matrices=True)[0]
                Q.append(basis[:, 6:])
            Q = np.array(Q)
        dim = Q.shape[2]
        ptril = np.tril_indices(dim)
        if args.init_cell is None:
            raise SystemExit('pair variants need --init-cell')
        local = np.einsum('bki,bkl,blj->bij', Q, donor_bonds, Q)
        if args.variant == 'pairobj':
            leak = np.abs(np.einsum('bij,bjk->bik', donor_bonds, rigid_check := np.eye(12)[None]-Q@Q.transpose(0, 2, 1))).max()
            assert leak < 1e-8*np.abs(donor_bonds).max(), 'donor bonds must be objective'
        L0 = np.linalg.cholesky(local+1e-9*np.abs(local).max()*np.eye(dim))
        shape = (nb*len(ptril[0]),)
        bounds = None
    else:
        L0 = np.linalg.cholesky(K0+1e-9*np.abs(K0).max()*np.eye(nf))
        shape = (len(tril[0])+(len(rotation) if args.variant == 'inertia' else 0),)
        bounds = None
    rng = np.random.default_rng(max(args.seed, 0))
    if args.variant == 'fem':
        x0 = np.zeros(2) if args.seed < 0 else rng.uniform(-1, 1, 2)
    elif args.variant == 'lever':
        x0 = np.zeros(shape[0])
        x0[n*P:] = args.shift_init*(length.min() if args.lever not in ('geom', 'geomcell') else 1.)
        if args.lever_mode == 'bend' and args.shift_init == 0:
            raise SystemExit('--lever-mode bend needs --shift-init != 0 (zero is a stationary point).')
        if args.init_cell is not None:
            assert donor.parameter_count == model.parameter_count
            x0[:n*P] = np.load(args.init_cell/'parameters.npz')['x']
        elif args.seed >= 0:
            x0[:n*P] = rng.uniform(-1, 1, n*P)
    elif args.variant in ('pair', 'pairobj'):
        x0 = L0[:, ptril[0], ptril[1]].ravel()
    elif args.variant in ('general', 'inertia'):
        x0 = np.concatenate((L0[tril], np.zeros(shape[0]-len(tril[0]))))
    else:
        x0 = np.zeros(int(np.prod(shape))) if args.seed < 0 else rng.uniform(-1, 1, int(np.prod(shape)))
    if args.init_x is not None:
        assert args.variant == 'cell', '--init-x is for the cell variant'
        start = np.load(args.init_x)['x']
        if start.size == model.parameter_count and x0.size != start.size:
            start = np.tile(start, x0.size//start.size)  # one shared vector -> every cell / class
        assert start.shape == x0.shape, (start.shape, x0.shape)
        x0 = start.copy()

    def bond_K(x):
        nb = len(model.i)
        scale = model.base*np.exp(x[:nb*4].reshape(nb, 4))
        blocks = np.einsum('bm,bmij->bij', scale, model.unit)
        K = sp.coo_matrix((blocks.ravel(), (model._rows, model._cols)), shape=(6*n,)*2).tocsr()
        if args.volume:
            K = K+model.volume_B.T@sp.diags(model.volume_base*np.exp(x[nb*4:]))@model.volume_B
        return K[free][:, free].tocsc(), scale

    def evaluate(x, cases, gradient=True):
        losses, grad, results = [], np.zeros_like(x), {}
        mass = model.mass[free].copy()
        if args.variant == 'inertia':
            mass[rotation] *= np.exp(x[len(tril[0]):])
        if args.variant == 'cell':
            for name in cases:
                loss, pos, g = model.simulate(cell_theta(x), datasets[name], steps[name], gradient)
                losses.append(loss); results[name] = pos
                if gradient:
                    grad += cell_grad(g)/len(cases)
        else:
            if args.variant == 'bond':
                Kf, scale = bond_K(x)
            elif args.variant == 'fem':
                lam, mu_ = fem_base*np.exp(x)
                Kf = (lam*K_lam+mu_*K_mu).tocsc()
            elif args.variant == 'lever':
                theta = x[:n*P].reshape(n, P)
                unit, dunit = lever_units(x)
                scale = model.bond_scale(theta)
                blocks = np.einsum('bm,bmij->bij', scale, unit)
                K = sp.coo_matrix((blocks.ravel(), (model._rows, model._cols)), shape=(6*n,)*2).tocsr()
                if args.volume:
                    K = K+model.volume_B.T@sp.diags(model.volume_base*np.exp(theta[:, 4]))@model.volume_B
                Kf = K[free][:, free].tocsc()
            elif args.variant == 'beam':
                k, dE, dG = beam_stiffness(x)
                blocks = np.einsum('bm,bmij->bij', k, model.unit)
                K = sp.coo_matrix((blocks.ravel(), (model._rows, model._cols)), shape=(6*n,)*2).tocsr()
                if args.volume:
                    K = K+model.volume_B.T@sp.diags(model.volume_base*np.exp(x.reshape(shape)[:, 2]))@model.volume_B
                Kf = K[free][:, free].tocsc()
            elif args.variant in ('pair', 'pairobj'):
                Lb = np.zeros((len(model.i), dim, dim)); Lb[:, ptril[0], ptril[1]] = x.reshape(len(model.i), -1)
                S = Q@Lb@Lb.transpose(0, 2, 1)@Q.transpose(0, 2, 1)
                Kf = sp.coo_matrix((S.ravel(), (model._rows, model._cols)), shape=(6*n,)*2).tocsr()[free][:, free].tocsc()
            else:
                L = np.zeros((nf, nf)); L[tril] = x[:len(tril[0])]
                Kf = sp.csc_matrix(L@L.T)
            for name in cases:
                loss, pos, adj = model.dynamics(Kf, mass, datasets[name], steps[name], gradient)
                losses.append(loss); results[name] = pos
                if not gradient:
                    continue
                h = adj['h']
                if args.variant == 'fem':
                    M_, W_ = adj['mu'], adj['w']
                    grad[0] += -h*lam*np.sum(M_*(K_lam@W_.T).T)/len(cases)
                    grad[1] += -h*mu_*np.sum(M_*(K_mu@W_.T).T)/len(cases)
                elif args.variant == 'lever':
                    mu_full = np.zeros((len(adj['mu']), 6*n)); mu_full[:, free] = adj['mu']
                    w_full = np.zeros_like(mu_full); w_full[:, free] = adj['w']
                    C = np.einsum('tbi,tbj->bij', mu_full[:, model.dofs], w_full[:, model.dofs])
                    dk = -h*np.einsum('bmij,bij->bm', unit, C)
                    g = np.zeros((n, P))
                    np.add.at(g[:, :4], model.i, .5*dk*scale)
                    np.add.at(g[:, :4], model.j, .5*dk*scale)
                    if args.volume:
                        weight = model.volume_base*np.exp(theta[:, 4])
                        g[:, 4] = -h*weight*np.sum((model.volume_B@mu_full.T)*(model.volume_B@w_full.T), 1)
                    gs = -h*scale[:, 1]*np.einsum('bij,bij->b', dunit, C)
                    grad[:n*P] += g.ravel()/len(cases)
                    grad[n*P:] += shift_to_parameters(gs)/len(cases)
                elif args.variant == 'beam':
                    mu_full = np.zeros((len(adj['mu']), 6*n)); mu_full[:, free] = adj['mu']
                    w_full = np.zeros_like(mu_full); w_full[:, free] = adj['w']
                    C = np.einsum('tbi,tbj->bij', mu_full[:, model.dofs], w_full[:, model.dofs])
                    dk = -h*np.einsum('bmij,bij->bm', model.unit, C)  # dL / d k_bm
                    g = np.zeros(shape)
                    for column, dk_dlog in ((0, dE), (1, dG)):
                        per_bond = .5*np.sum(dk*dk_dlog, axis=1)
                        np.add.at(g[:, column], model.i, per_bond)
                        np.add.at(g[:, column], model.j, per_bond)
                    if args.volume:
                        weight = model.volume_base*np.exp(x.reshape(shape)[:, 2])
                        g[:, 2] = -h*weight*np.sum((model.volume_B@mu_full.T)*(model.volume_B@w_full.T), 1)
                    grad += g.ravel()/len(cases)
                elif args.variant == 'bond':
                    nb = len(model.i)
                    mu_full = np.zeros((len(adj['mu']), 6*n)); mu_full[:, free] = adj['mu']
                    w_full = np.zeros_like(mu_full); w_full[:, free] = adj['w']
                    C = np.einsum('tbi,tbj->bij', mu_full[:, model.dofs], w_full[:, model.dofs])
                    grad[:nb*4] += (-h*np.einsum('bmij,bij->bm', model.unit, C)*scale).ravel()/len(cases)
                    if args.volume:
                        weight = model.volume_base*np.exp(x[nb*4:])
                        grad[nb*4:] += -h*weight*np.sum((model.volume_B@mu_full.T)*(model.volume_B@w_full.T), 1)/len(cases)
                elif args.variant in ('pair', 'pairobj'):
                    GK = np.zeros((6*n, 6*n)); GK[np.ix_(free, free)] = -h*adj['mu'].T@adj['w']
                    Gb = GK[model.dofs[:, :, None], model.dofs[:, None, :]]
                    GL = Q.transpose(0, 2, 1)@(Gb+Gb.transpose(0, 2, 1))@Q@Lb
                    grad += GL[:, ptril[0], ptril[1]].ravel()/len(cases)
                else:
                    GK = -h*adj['mu'].T@adj['w']
                    grad[:len(tril[0])] += ((GK+GK.T)@L)[tril]/len(cases)
                    if args.variant == 'inertia':
                        grad[len(tril[0]):] += adj['mass'][rotation]*mass[rotation]/len(cases)
        value = float(np.mean(losses))
        if args.variant in ('cell', 'bond', 'beam', 'lever'):
            r = x[:n*P] if args.variant == 'lever' else x
            value += regularization*float(np.sum(r**2))/(4*n)
            grad[:len(r)] += 2*regularization*r/(4*n)
            if args.variant == 'cell':
                penalty, penalty_grad = fused_penalty(x)
                value += penalty
                grad += penalty_grad
            if args.variant == 'lever' and args.lever_reg > 0:
                shift = bond_shift(x)/length
                value += args.lever_reg*float(np.mean(shift**2))
                grad[n*P:] += shift_to_parameters(2*args.lever_reg*shift/(length*len(shift)))
        return value, grad, losses, results

    if args.evaluate_run is not None:
        extra = json.loads((args.extra_reference/'config.json').read_text())['load_cases']
        with np.load(args.extra_reference/'patch.npz') as a:
            assert all(np.array_equal(a[k], geometry[k]) for k in ('points', 'pairs', 'fixed')), 'geometry differs'
        for name in extra:
            datasets[name] = load_case(args.extra_reference, name)
            steps[name] = 32  # same substeps as the original held-out cases
        x = np.load(args.evaluate_run/'parameters.npz')['x']
        _, _, losses, _ = evaluate(x, extra, False)
        metrics = dict(zip(extra, np.sqrt(losses).tolist()))
        write(args.evaluate_run/f'eval_{args.extra_reference.name}.json',
              dict(extra_reference=str(args.extra_reference), normalized_rmse=metrics))
        print(json.dumps(metrics), flush=True)
        return
    if args.check_gradient:
        check = np.random.default_rng(7)
        x = x0+1e-2*check.normal(size=x0.shape)*np.maximum(np.abs(x0), 1e-3*np.abs(x0).max()+1e-12)
        direction = check.normal(size=x.shape)*np.maximum(np.abs(x), 1e-3*np.abs(x).max()+1e-12)
        _, grad, _, _ = evaluate(x, ['train_down'])
        e = 1e-5
        fd = (evaluate(x+e*direction, ['train_down'], False)[0]-evaluate(x-e*direction, ['train_down'], False)[0])/(2*e)
        print(args.variant, args.volume, 'finite difference', fd, 'adjoint', grad@direction,
              'relative', abs(fd-grad@direction)/abs(fd))
        return
    args.out.mkdir(parents=True, exist_ok=False)
    write(args.out/'config.json', vars(args) | dict(reference=str(args.reference), out=str(args.out),
        init_cell=None if args.init_cell is None else str(args.init_cell),
        init_x=None if args.init_x is None else str(args.init_x),
        extra_reference=None if args.extra_reference is None else str(args.extra_reference), evaluate_run=None,
        fit_cases=fit_cases, parameters=int(np.prod(shape))))
    scan = None
    if args.init_scan:
        if args.variant not in ('cell', 'fem'):
            raise SystemExit('--init-scan supports the cell and fem variants.')
        offsets = np.linspace(-3, 3, 25)
        losses = []
        for offset in offsets:
            trial = x0.copy()
            if args.variant == 'cell' and args.cell_param == 'fused':
                trial[:model.parameter_count] += offset  # shared part only
            else:
                trial += offset
            losses.append(evaluate(np.clip(trial, -bound, bound), fit_cases, False)[0])
        best = int(np.argmin(losses))
        shift = offsets[best]
        if args.variant == 'cell' and args.cell_param == 'fused':
            x0[:model.parameter_count] += shift
        else:
            x0 = x0+shift
        x0 = np.clip(x0, -bound, bound)
        scan = dict(offsets=offsets.tolist(), losses=[float(v) for v in losses], chosen=float(shift))
        print(json.dumps(dict(init_scan_offset=float(shift), loss=float(losses[best]))), flush=True)
    history = []
    start = time.perf_counter()
    def value_gradient(x):
        value, grad, losses, _ = evaluate(x, fit_cases)
        history.append(dict(evaluation=len(history), objective=value, rmse=float(np.sqrt(np.mean(losses)))))
        if len(history) % 200 == 0:
            print(json.dumps(history[-1]), flush=True)
        return value, grad
    fitted = minimize(value_gradient, x0, jac=True, method='L-BFGS-B', bounds=bounds,
                      options=dict(maxiter=args.max_iterations, maxfun=2*args.max_iterations,
                                   gtol=args.gradient_tolerance, ftol=0., maxls=40, maxcor=20))
    _, grad, _, results = evaluate(fitted.x, report_cases)
    metrics = {}
    for name in report_cases:
        target, cells = datasets[name]['target'], ~model.fixed
        scale2 = np.mean((target[1:, cells]-model.points[cells])**2)
        metrics[name] = float(np.sqrt(np.mean((results[name][1:, cells]-target[1:, cells])**2)/scale2))
    _, fit_grad, _, _ = evaluate(fitted.x, fit_cases)
    if bounds:
        lower, upper = np.array(bounds).T
        pg = fitted.x-np.clip(fitted.x-fit_grad, lower, upper)
    else:
        pg = fit_grad
    report = dict(variant=args.variant, volume=args.volume, bound=args.bound, seed=args.seed, cases=args.cases,
        parameters=int(np.prod(shape)), stop_message=str(fitted.message), iterations=int(fitted.nit),
        evaluations=len(history), projected_gradient_max=float(np.max(np.abs(pg))),
        converged=bool(np.max(np.abs(pg)) <= args.gradient_tolerance),
        fraction_near_bound=(float(np.mean(np.abs(fitted.x[:n*model.parameter_count] if args.variant == 'lever'
                                                   else fitted.x) > .99*bound)) if bounds else None),
        contact_shifts=(bond_shift(fitted.x).tolist() if args.variant == 'lever' else None),
        lever_parameters=(fitted.x[n*model.parameter_count:].tolist() if args.variant == 'lever' else None),
        seconds=time.perf_counter()-start, init_scan=scan, normalized_rmse=metrics)
    np.savez_compressed(args.out/'parameters.npz', x=fitted.x)
    write(args.out/'report.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
