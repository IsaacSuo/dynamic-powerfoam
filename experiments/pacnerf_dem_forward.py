"""EnergyFoam on a reconstructed PAC-NeRF object vs the ground-truth MPM particle trajectories.

Targets: every GT particle (simulation_data/<scene>/<k>.ply, frame k at t = k/24 s, same order
in every frame) is assigned at frame 0 to the material cell of smallest power distance among
its 16 nearest centres; a cell's target is the mean position of its particles. Cells without
particles are left out of the error.

Parameters from the GT (E, nu): bonds carry a share alpha of mu (kn = ks = 2 alpha mu A/d, the
cube-lattice equivalent of an isotropic mu alpha, lam 0; bend/twist at beam values) and the
bodies PAC-NeRF's Neo-Hookean law with mu (1 - alpha) and lam, per unit density (rho = 1000),
theta = 0. On irregular cells this split is only an initial guess. Error: normalized RMSE over frames 1.., as for the cube runs.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from experiments.foam_dem_energy import EnergyFoam

RHO = 1000.


def read_ply(path):
    raw = Path(path).read_bytes()
    head = raw.index(b'end_header\n')+len(b'end_header\n')
    n = int(raw[:head].split(b'element vertex ')[1].split(b'\n')[0])
    return np.frombuffer(raw[head:head+24*n], dtype='<f8').reshape(n, 3)


def cell_targets(cells, body_of_cell, bodies, sim_dir, frames):
    """Particles -> power cell (support radii) -> rigid body; body target = particle mean."""
    p, r = cells['points'], cells['radii']
    x0 = read_ply(sim_dir/'0.ply')
    _, near = cKDTree(p).query(x0, k=16)
    power = ((x0[:, None]-p[near])**2).sum(-1)-r[near]**2
    owner = body_of_cell[near[np.arange(len(x0)), power.argmin(1)]]
    p = bodies
    count = np.bincount(owner, minlength=len(p))
    targets = []
    for k in range(frames+1):
        x = read_ply(sim_dir/f'{k}.ply')
        s = np.zeros_like(p)
        np.add.at(s, owner, x)
        targets.append(s/np.maximum(count, 1)[:, None])
    return np.array(targets), count > 0, count


def build(geometry, E, nu, device, ground_mode, alpha=.25, damping=0., dtype=torch.float64, inertia_scale=1.,
          contact_scale=1., layer_dx=.01, kinematics='small', residual=False, shell=False, thin_ratio=.1, shell_eas=None, cosserat=False, bond_level=None, nonlinear=False, ogden_alpha=2.,
          clean=False, viscosity=.01, dilatation=False, nodalvol=False):
    """Bonds kn = ks = 2 alpha mu A/d (lattice-equivalent mu alpha, lam 0) + Neo-Hookean bodies
    (mu (1 - alpha), lam), all per unit density; theta = 0 is this split of the GT material.
    inertia_scale: rotational inertia multiplier, default 1 (true inertia) since 2026-10-13 (was 4, a DEM time-step trick:
    on the burger the fastest mode is translational, omega_max 1.34e5 rad/s for x1 and x4 alike, so x4 only slowed the
    body rotations). springgaus_free_oscillation.py keeps an explicit 4 for its old results.
    shell=True (BOND_BENDING.md): residual along the bond direction, enhanced strain on free-surface bodies,
    shell closure for thin bodies (moment-matrix eigenvalue ratio < thin_ratio) with full plate bending at their bonds.
    shell_eas=None: enhanced strain on for the bonded model, off for cosserat=True. The enhanced strain leaves a
    free-surface body no stiffness along its normal (the condensed thickness stretch absorbs any normal approach of
    its inner neighbours); without bond springs that is a zero-energy direction, which under compression becomes a
    negative-stiffness collapse mode (burger after impact: bodies crushed to J ~ 0.05-0.15, tangent growth ~1e7 per
    frame, exploding adjoints). Pass shell_eas=True to compare.
    bond_level=None: on for cosserat=True (COSSERAT_FOAM.md section 7, Breitzman & Dayal 2018): the body law is
    evaluated per neighbour pair at the bond-level deformation gradient; this replaces the residual penalty and the
    enhanced strain, so both are switched off.
    clean=True (CLEAN_FOAM.md): classical continuum - bond-level Neo-Hookean on translations only (no micropolar terms,
    rotations not integrated), thin closure kept; damping by central dashpots along the current pair directions (objective:
    rigid rotations are not damped), c = 2 viscosity A / L per unit density (viscosity = kinematic, m^2/s; scaled by
    exp(theta[:, 6])), applied at the body centres (zero levers, no torques), integrated by the exact per-pair implicit
    solver (WarpFoam implicit_damping); the damping argument is ignored.
    dilatation=True (SPRING_DILATATION.md): bonded DEM with the springs carrying all of mu (alpha = 1) and, instead of the
    averaged-F Neo-Hookean, a volumetric multi-body term per body from its bonds' log stretches (lam); no inverse of M."""
    if dilatation or nodalvol:
        alpha = 1.
    if clean:
        cosserat, bond_level, damping, kinematics = True, True, 1., 'exact'
    if bond_level is None:
        bond_level = cosserat
    if bond_level:
        residual, shell_eas = False, False
    if shell_eas is None:
        shell_eas = not cosserat
    lam, mu = E*nu/((1+nu)*(1-2*nu)), E/(2*(1+nu))
    extra = dict(residual='normal' if residual else False, enhanced=shell_eas, thin_ratio=thin_ratio,
                 thin_bend_factor=1./(alpha*(1-nu))) if shell else dict(residual=residual)
    if cosserat:      # COSSERAT_FOAM.md: no bond springs (alpha irrelevant), full mu in the body law, true micro-inertia
        extra.update(cosserat=True, thin_bend_factor=1., bond_level=bool(bond_level))
        if clean:
            extra.update(micropolar=False, viscosity=0.)
        alpha, inertia_scale = .5, 1.
    model = EnergyFoam(geometry, k_v=lam/RHO, device=device, young=2*alpha*mu/RHO, shear=2*alpha*mu/RHO,
                       body_law='dilatation' if dilatation else 'nodalvol' if nodalvol else 'neohookean', k_mu=(1-alpha)*mu/RHO if not cosserat else mu/RHO, damping=damping, dtype=dtype,
                       inertia_scale=inertia_scale, kinematics='exact' if cosserat else kinematics, **extra)
    model.nonlinear_bonds = bool(nonlinear)   # Ogden normal bond springs (small kinematics; alpha 2 = neo-Hookean)
    model.ogden_alpha = float(ogden_alpha)
    model.mu_ref = mu/RHO          # material shear modulus per density (theta[:, 5] = 0); the springs and k_mu carry parts of it
    if nodalvol:     # tetrahedra built eagerly (not inside a later functorch transform)
        model._setup_tets()
    if clean:        # central dashpots at the body centres
        A = torch.as_tensor(np.asarray(geometry['areas']), dtype=model.c_n.dtype, device=model.c_n.device)
        Lb = torch.linalg.vector_norm(model.X[model.j]-model.X[model.i], dim=1).to(model.c_n.dtype)
        model.c_n = 2*viscosity*A/Lb
        model.c_t, model.c_r = torch.zeros_like(model.c_n), torch.zeros_like(model.c_n)
        model.ri, model.rj = torch.zeros_like(model.ri), torch.zeros_like(model.rj)
        model.clean = True
    if bond_level:   # two-end gradient variant: the chord is compared with (F_a + F_b)/2 dX (no kappa L / 2 shear in bending)
        model.bond_level_mid = True
    contact = contact_scale*(3*geometry['masses']/(4*np.pi))**(1/3)  # equivalent-sphere radius
    model.set_environment(gravity=[0, -9.8, 0], ground_point=[0, 0, 0], ground_normal=[0, 1, 0],
                          contact_radius=contact, ground_mode=ground_mode, layer_dx=layer_dx)
    return model


def simulate(model, theta, v0, frames, substeps, dt, compile=False):
    x0, q0, _, w0 = model.initial_state()
    v = torch.as_tensor(np.asarray(v0), dtype=model.dtype, device=model.device).expand(model.n, 3).clone()
    positions, _ = model.rollout(theta, (x0, q0, v, w0), dt, substeps, frames, compile=compile)
    return positions


def normalized_rmse(pred, target, has):
    pred, target = pred[:, has], target[:, has]
    return float(np.sqrt(np.mean((pred[1:]-target[1:])**2)/np.mean((target[1:]-target[:1])**2)))


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--scene', default='elastic/0')
    a.add_argument('--geometry', type=Path, default=Path('output/pacnerf_dem/elastic_0_f0/geometry.npz'))
    a.add_argument('--frames', type=int, default=15)
    a.add_argument('--substeps', type=int, default=0, help='0: from the largest frequency (dt = 1/omega_max)')
    a.add_argument('--ground-mode', default='noslip')
    a.add_argument('--alpha', type=float, default=.25, help='share of mu carried by the bonds')
    a.add_argument('--damping', type=float, default=0., help='bond dashpots, fraction of critical')
    a.add_argument('--dtype', default='float32', choices=('float32', 'float64'))
    a.add_argument('--contact-scale', type=float, default=1., help='ground contact radius / equivalent-sphere radius')
    a.add_argument('--layer-dx', type=float, default=.01, help="'layer' ground: MPM grid spacing (GT ~ 1 cm)")
    a.add_argument('--out', type=Path, default=Path('output/pacnerf_dem/elastic_0_f0'))
    args = a.parse_args()
    gt = json.loads(Path('data/pacnerf/gt/elastic.json').read_text())[args.scene.split('/')[1]]
    geometry = dict(np.load(args.geometry))
    cells = dict(np.load(args.geometry.parent/'cells.npz'))
    targets, has, count = cell_targets(cells, geometry['body_of_cell'], geometry['points'],
                                       Path('data/pacnerf/simulation_data')/args.scene, args.frames)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build(geometry, gt['E'], gt['nu'], device, args.ground_mode, args.alpha, args.damping,
                  dict(float32=torch.float32, float64=torch.float64)[args.dtype], contact_scale=args.contact_scale,
                  layer_dx=args.layer_dx)
    theta = torch.zeros(model.n, 6, dtype=model.dtype, device=device)
    omega = model.max_frequency(theta)
    if args.substeps <= 0:
        args.substeps = int(np.ceil(1/24/model.stable_dt(theta)))  # half the stability limit
    print('omega_max %.4g rad/s, substeps per frame %d' % (omega, args.substeps), flush=True)
    start = time.perf_counter()
    with torch.no_grad():
        pos = simulate(model, theta, gt['v'], args.frames, args.substeps, 1/24/args.substeps, compile=True).cpu().numpy()
    seconds = time.perf_counter()-start
    # Offset between cell centres and their particle means at frame 0 (assignment quality).
    start_gap = float(np.sqrt(np.mean((pos[0, has]-targets[0, has])**2)))
    shifted = pos-pos[:1]+targets[:1]  # compare motion, not the frame-0 centre/particle-mean gap
    per_frame = np.sqrt(((shifted-targets)[:, has]**2).sum(-1).mean(-1))
    report = dict(scene=args.scene, E=gt['E'], nu=gt['nu'], v0=gt['v'], ground_mode=args.ground_mode,
                  omega_max=omega, substeps=args.substeps,
                  cells=int(model.n), cells_with_particles=int(has.sum()), particles_per_cell_median=float(np.median(count[has])),
                  frame0_centre_vs_particle_mean_rms=start_gap, seconds=seconds,
                  normalized_rmse=normalized_rmse(shifted, targets, has),
                  per_frame_rms_m=per_frame.tolist(),
                  centroid_pred=shifted[:, has].mean(1).tolist(), centroid_gt=targets[:, has].mean(1).tolist())
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out/f'forward_{args.ground_mode}_a{args.alpha:g}_z{args.damping:g}_c{args.contact_scale:g}_l{args.layer_dx:g}_{args.dtype}.npz', positions=pos, targets=targets, has=has)
    (args.out/f'forward_{args.ground_mode}_a{args.alpha:g}_z{args.damping:g}_c{args.contact_scale:g}_l{args.layer_dx:g}_{args.dtype}.json').write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k not in ('centroid_pred', 'centroid_gt')}, indent=1))
    for k in range(0, args.frames+1, 3):
        print(k, 'centroid y pred %.3f gt %.3f' % (report['centroid_pred'][k][1], report['centroid_gt'][k][1]),
              'min y pred %.3f gt %.3f' % (shifted[k, has, 1].min(), targets[k, has, 1].min()))


if __name__ == '__main__':
    main()
