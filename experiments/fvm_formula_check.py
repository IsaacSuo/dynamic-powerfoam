"""DEM with parameters derived from (E, nu) instead of training (finite-volume reading).

On the unit-cube lattice (A = d = 1) under a homogeneous strain with zero cell rotation,
the pair springs and the power volume term give (per unit volume, complete lattice)
    kn -> C11 += kn,   ks -> C44 += ks/2,   k_v -> C11 += k_v, C12 += k_v,
so isotropic elasticity (C11 = lam+2mu, C12 = lam, C44 = mu) needs
    kn = 2 mu A/d,   ks = 2 mu A/d,   k_v = lam,   kb, kt free (micro-rotation terms).
This script writes run directories with those parameters (kb, kt either off or at their
beam values), evaluates them on every load case with the same protocol as trained runs,
and reports the affine moduli of the finite specimen (missing outer bonds included).
"""
import json
from pathlib import Path

import numpy as np

from experiments.foam_linear_inverse import LinearizedFoam
from experiments.foam_affine_moduli_check import strain_fields

F = Path('output/foam_fem')
OUT = F/'fvm_formula_v1'
OFF = -np.log(1e4)  # lower parameter bound: term effectively removed
K_V = 400.  # LinearizedFoam reference k_v; theta[4] scales it


def formula_theta(model, E, nu, rotation):
    lam, mu = E*nu/((1+nu)*(1-2*nu)), E/(2*(1+nu))
    kn, ks, kb, kt = model.base[0]
    theta = [np.log(2*mu/kn), np.log(2*mu/ks)]
    theta += [0., 0.] if rotation == 'beam' else [OFF, OFF]
    theta.append(np.log(lam/K_V))
    return np.array(theta), lam, mu


def specimen_moduli(model, theta):
    K = model.stiffness(np.tile(theta, (model.n, 1)))
    E = strain_fields(model.points)
    return (E.T@(K@E))/model.n  # unit cells: material volume = n


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for spec in ('block', 'bar'):
        ref = F/f'cube_{spec}_reference_q018'
        material = json.loads((ref/'config.json').read_text())['material']
        geometry = dict(np.load(ref/'patch.npz'))
        model = LinearizedFoam(geometry, volume=True, volume_mode='power', k_v=K_V)
        template = F/f'cube_multistart_v1/{spec}_uniform_power_s1/config.json'
        for rotation in ('off', 'beam'):
            theta, lam, mu = formula_theta(model, material['E'], material['nu'], rotation)
            run = OUT/f'{spec}_formula_rot{rotation}'
            run.mkdir(exist_ok=True)
            config = json.loads(template.read_text())
            config.update(out=str(run), seed=-1, max_iterations=0)
            (run/'config.json').write_text(json.dumps(config, indent=1))
            np.savez_compressed(run/'parameters.npz', x=theta)
            C = specimen_moduli(model, theta)
            summary[run.name] = dict(theta=theta.tolist(), lam=lam, mu=mu,
                                     specimen_C11=C[0, 0], specimen_C12=C[0, 1], specimen_C44=C[3, 3],
                                     target_C11=lam+2*mu, target_C12=lam, target_C44=mu)
    (OUT/'formula.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
