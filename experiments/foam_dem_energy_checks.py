"""Checks of EnergyFoam (experiments/foam_dem_energy.py) on the cube specimens.

1. Hessian at the reference equals LinearizedFoam(volume_mode='power') stiffness (random theta).
2. Rigid motion (large rotation + translation): energy and forces vanish.
   Closed-form forces equal autograd of the energy at a large deformation; momentum balance.
3. Objectivity: a deformed state and its rigidly rotated copy have equal energy.
4. Free tumbling + vibration (nothing fixed): symplectic Euler energy error vs step size.
5. Drop with gravity onto the sticky ground: ballistic phase, no energy gain, no penetration.
"""
import json
import sys

import numpy as np
import torch

from experiments.foam_dem_energy import EnergyFoam, quat_mul, quat_small, quat_matrix
from experiments.foam_linear_inverse import LinearizedFoam


def axis_angle(axis, angle):
    axis = np.asarray(axis, float)/np.linalg.norm(axis)
    return torch.tensor([np.cos(angle/2), *(np.sin(angle/2)*axis)], dtype=torch.float64)


def run(label, geometry):
    out = {}
    model = EnergyFoam(geometry)
    rng = np.random.default_rng(0)
    theta = torch.tensor(rng.uniform(-1, 1, (model.n, 5)))
    x0, q0, _, _ = model.initial_state()

    def energy_of(z):
        z = z.reshape(model.n, 6)
        return model.energy(theta, x0+z[:, :3], quat_mul(quat_small(z[:, 3:]), q0))
    H = torch.autograd.functional.hessian(energy_of, torch.zeros(6*model.n, dtype=torch.float64))
    linear = LinearizedFoam(geometry, volume=True, volume_mode='power').stiffness(theta.numpy()).toarray()
    out['hessian_vs_linear_relative'] = float(np.abs(H.numpy()-linear).max()/np.abs(linear).max())
    Q = axis_angle([.3, -1., .4], 1.3)
    R = quat_matrix(Q)
    t = torch.tensor([.7, -2., .4], dtype=torch.float64)
    x = x0@R.T+t
    q = quat_mul(Q.expand(model.n, 4), q0)
    f, tau = model.forces(theta, x, q)
    out['rigid_energy'] = float(model.energy(theta, x, q))
    out['rigid_force_max'] = float(torch.cat((f, tau)).abs().max())
    # Deformed state (strain ~ 1e-2 plus cell spins), then rotate rigidly by Q.
    u = 1e-2*torch.tensor(rng.normal(size=(model.n, 3)))
    spin = quat_small(2e-2*torch.tensor(rng.normal(size=(model.n, 3))))
    E1 = model.energy(theta, x0+u, quat_mul(spin, q0))
    E2 = model.energy(theta, (x0+u)@R.T+t, quat_mul(Q.expand(model.n, 4), quat_mul(spin, q0)))
    out['objectivity_relative'] = float(abs(E1-E2)/E1)
    # Closed-form forces vs autograd at a large deformation (strain ~0.2) under a large rotation,
    # and balance of linear / angular momentum of the internal forces.
    xs = ((x0+.2*torch.tensor(rng.normal(size=(model.n, 3))))@R.T+t)
    qs = quat_mul(Q.expand(model.n, 4), quat_small(.5*torch.tensor(rng.normal(size=(model.n, 3)))))
    fa, ta = model.forces(theta, xs, qs)
    fb, tb = model.forces_autograd(theta, xs, qs)
    scale = float(torch.cat((fb, tb)).abs().max())
    out['closed_form_vs_autograd_relative'] = float(torch.cat((fa-fb, ta-tb)).abs().max())/scale
    out['net_force_relative'] = float(fa.sum(0).abs().max())/scale
    out['net_torque_relative'] = float((ta+torch.linalg.cross(xs, fa)).sum(0).abs().max())/scale
    # Free tumbling + vibration (nothing fixed): rigid spin about the centre plus random
    # perturbation; symplectic Euler energy error at two step sizes (should scale ~ dt).
    model.fixed[:] = False
    omega = torch.tensor([.5, 3., -1.], dtype=torch.float64)
    centre = x0.mean(0)
    for dt in (4e-4, 2e-4):
        x = x0+u
        v = torch.linalg.cross(omega.expand(model.n, 3), x-centre)+torch.tensor([.3, 0, 0])
        v = v+.05*torch.tensor(np.random.default_rng(1).normal(size=(model.n, 3)))
        state = (x, quat_mul(spin, q0), v, omega.expand(model.n, 3).clone())
        energies = []
        steps = int(round(1./dt))
        for k in range(steps):
            if k % 50 == 0:
                energies.append(float(model.kinetic(state)+model.energy(theta, state[0], state[1]).detach()))
            state = tuple(s.detach() for s in model.step(theta, state, dt))
        energies = np.array(energies)
        out[f'tumble_energy_relative_error_dt{dt:g}'] = float(np.abs(energies-energies[0]).max()/energies[0])
    out['tumble_rotation_angle_rad'] = float(2*torch.acos(state[1][:, 0].abs().clamp(max=1)).mean())
    out['tumble_strain_energy_end'] = float(model.energy(theta, state[0], state[1]).detach())
    out['tumble_strain_energy_start'] = float(E1.detach())
    print(label, json.dumps(out))
    return out



def drop(label, geometry, dt=1e-3, T=2.5):
    """Gravity + sticky ground: ballistic phase vs analytic, then landing (y-up, ground y = -3)."""
    model = EnergyFoam(geometry)
    model.fixed[:] = False
    g = np.array([0., -9.8, 0.])
    model.set_environment(gravity=g, ground_point=[0., -3., 0.], ground_normal=[0., 1., 0.],
                          contact_radius=np.full(model.n, .5))
    theta = torch.zeros(model.n, 5, dtype=torch.float64)
    x0, q0, _, _ = model.initial_state()
    v0 = torch.tensor([.5, -1., 0.], dtype=torch.float64)
    w0 = torch.tensor([0., 0., .6], dtype=torch.float64)
    c0 = x0.mean(0)
    state = (x0, q0, v0+torch.linalg.cross(w0.expand(model.n, 3), x0-c0), w0.expand(model.n, 3).clone())
    out, energy, lowest, touch_time = {}, [], [], None
    for k in range(int(round(T/dt))):
        t = (k+1)*dt
        state = tuple(s.detach() for s in model.step(theta, state, dt))
        x = state[0]
        if touch_time is None and bool(((x[:, 1]+3.) <= .5).any()):
            touch_time = t
            centre = x.mean(0).numpy()
            expected = (c0+v0).numpy()*0+c0.numpy()+v0.numpy()*t+.5*g*t*t
            out['ballistic_centre_error_at_touch'] = float(np.abs(centre-expected).max())
            out['touch_time'] = t
        if k % 50 == 0:
            U = float(model.energy(theta, x, state[1]).detach())
            energy.append(float(model.kinetic(state))+U-float((model.mass*(x@torch.tensor(g))).sum()))
            lowest.append(float(x[:, 1].min()+3.))
    energy = np.array(energy)
    out['max_energy_increase_relative'] = float(np.max(energy-np.maximum.accumulate(energy[::-1])[::-1].min()*0-energy[0])/abs(energy[0]))
    out['energy_start_end'] = [float(energy[0]), float(energy[-1])]
    out['lowest_centre_above_ground_min'] = float(min(lowest))
    out['final_kinetic'] = float(model.kinetic(state))
    out['final_max_strain_proxy'] = float(torch.linalg.vector_norm(state[0][model.j]-state[0][model.i], dim=1).sub(1).abs().max())
    print(label, 'drop', json.dumps(out))
    return out


if __name__ == '__main__':
    for path in sys.argv[1:]:
        run(path, dict(np.load(path)))
        drop(path, dict(np.load(path)))
