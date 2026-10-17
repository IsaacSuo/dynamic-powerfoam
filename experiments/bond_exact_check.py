"""Checks of the geometrically exact bond kinematics (BOND_EXACT.md) in the torch EnergyFoam.

On a real DEM geometry (float64, CPU): (1) closed-form forces / torques vs autograd of the energy at
random large-rotation states; (2) objectivity under a superposed rigid rotation; (3) rest state;
(4) small-deformation agreement with the old ('small') kinematics; (5) total force and angular
momentum balance of the bond forces.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from experiments.foam_dem_energy import quat_mul
from experiments.pacnerf_dem_forward import build


def random_quat(n, max_angle, gen):
    axis = torch.randn(n, 3, generator=gen, dtype=torch.float64)
    axis = axis/axis.norm(dim=-1, keepdim=True)
    ang = torch.rand(n, generator=gen, dtype=torch.float64)*max_angle
    return torch.cat((torch.cos(ang/2)[:, None], torch.sin(ang/2)[:, None]*axis), -1)


def rel(a, b):
    return float((a-b).norm()/b.norm().clamp(min=1e-300))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dem', type=Path, default=Path('output/springgaus_dem/burger_spacing8k_mf0.1'))
    a = p.parse_args()
    g = dict(np.load(a.dem/'geometry.npz'))
    gen = torch.Generator().manual_seed(0)
    ok = True
    for kin in ('exact',):
        m = build(g, 2e4, .3, 'cpu', 'noslip', .25, 0., torch.float64, kinematics=kin)
        m_small = build(g, 2e4, .3, 'cpu', 'noslip', .25, 0., torch.float64, kinematics='small')
        theta = .2*torch.randn(m.n, 7, generator=gen, dtype=torch.float64)
        X = m.X.clone()
        h = float(np.median(np.cbrt(g['masses'])))
        print('bodies %d, bonds %d, typical size %.2f mm' % (m.n, len(m.i), 1000*h))
        # (3) rest
        q0 = torch.zeros(m.n, 4, dtype=torch.float64); q0[:, 0] = 1
        f, tau = m.forces(theta, X, q0)
        E0 = float(m.energy(theta, X, q0))
        print('[rest] energy %.2e, |f| %.2e, |tau| %.2e' % (E0, float(f.abs().max()), float(tau.abs().max())))
        # (1) closed form vs autograd at large rotations
        for max_deg in (5., 60., 120., 170.):
            q = random_quat(m.n, np.radians(max_deg), gen)
            x = X+.3*h*torch.randn(X.shape, generator=gen, dtype=torch.float64)
            f, tau = m.forces(theta, x, q)
            fa, ta = m.forces_autograd(theta, x, q)
            ef, et = rel(f, fa), rel(tau, ta)
            ok &= ef < 1e-8 and et < 1e-8
            print('[closed form vs autograd] body rotations up to %3.0f deg: force rel err %.1e, torque rel err %.1e' % (max_deg, ef, et))
        # (2) objectivity
        Q = random_quat(1, np.radians(150.), gen)
        from experiments.foam_dem_energy import quat_matrix
        RQ = quat_matrix(Q)[0]
        xQ = x@RQ.T+torch.tensor([.3, -.2, .1], dtype=torch.float64)
        qQ = quat_mul(Q.expand_as(q), q)
        E1, E2 = float(m.energy(theta, x, q)), float(m.energy(theta, xQ, qQ))
        f2, t2 = m.forces(theta, xQ, qQ)
        print('[objectivity, 150 deg + shift] energy rel diff %.1e, force rel err %.1e, torque rel err %.1e' % (
            abs(E1-E2)/abs(E1), rel(f2, f@RQ.T), rel(t2, tau@RQ.T)))
        # (5) balance (bond part only: volume forces are internal too)
        Ftot = f.sum(0); L = (torch.linalg.cross(x, f)+tau).sum(0)
        print('[balance] |sum f| %.1e, |sum x x f + tau| %.1e (scale %.1e)' % (float(Ftot.norm()), float(L.norm()), float(tau.abs().sum())))
        # (4) small deformation vs old kinematics
        for eps in (1e-2, 1e-3):
            q = random_quat(m.n, eps, gen); x = X+eps*h*torch.randn(X.shape, generator=gen, dtype=torch.float64)
            fe, te = m.forces(theta, x, q); fs, ts = m_small.forces(theta, x, q)
            print('[small deformation eps %.0e] exact vs small: energy rel %.1e, force rel %.1e, torque rel %.1e' % (
                eps, abs(float(m.energy(theta, x, q)-m_small.energy(theta, x, q)))/float(m_small.energy(theta, x, q)), rel(fe, fs), rel(te, ts)))
        # single-well check: one bond, twist body j by angle a about a random axis, energy vs a
        angs = np.radians([0, 45, 90, 135, 170, 180, 190])
        line = []
        for kk, mm in (('exact', m), ('small', m_small)):
            Es = []
            for ang in angs:
                qq = q0.clone(); axis = torch.tensor([1., 2., 3.], dtype=torch.float64); axis /= axis.norm()
                b = int(mm.j[0])
                qq[b] = torch.cat((torch.tensor([np.cos(ang/2)], dtype=torch.float64), np.sin(ang/2)*axis))
                Es.append(float(mm.energy(torch.zeros_like(theta), X, qq)))
            line.append('%s: %s' % (kk, ' '.join('%.3g' % (e/Es[2]) for e in Es)))
        print('[single body rotated 0/45/90/135/170/180/190 deg, energy / energy(90)]', ' | '.join(line))
    print('ALL CLOSED-FORM CHECKS PASSED' if ok else 'CLOSED-FORM CHECK FAILED')


if __name__ == '__main__':
    main()
