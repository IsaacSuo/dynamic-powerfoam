"""Independent CPU-law comparison, finite rotation and nonlinear dynamics checks."""
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from experiments.bonded_dem_core import ParallelBonds
from experiments.bonded_dem_dynamics import BondedDynamics
from experiments.bonded_dem_finite_gpu import FiniteBondsGPU
from experiments.bonded_dem_probe import prescribed, expected, rigid_motion, X


class Adapter:
    """Expose the GPU material law to the existing prescribed-path fixtures."""
    def __init__(self, gpu):
        self.gpu = gpu

    def __getattr__(self, key):
        return getattr(self.gpu, key).cpu().numpy()

    def advance(self, end, velocity, omega, dt, x_mid=None):
        g = self.gpu
        y = g.tensor(np.c_[velocity, omega]*dt)
        trial = g.trial(y, end_positions=g.tensor(end),
                        mid_positions=None if x_mid is None else g.tensor(x_mid))
        g.commit(trial)

    def loads(self):
        value = self.gpu.loads().cpu().numpy()
        return value[:, :3], value[:, 3:]

    def elastic_energy_per_bond(self):
        g = self.gpu
        return (.5*(g.fn*g.fn/g.kn+(g.fs*g.fs).sum(1)/g.ks
                    +g.mt*g.mt/g.kt+(g.mb*g.mb).sum(1)/g.kb)).cpu().numpy()


def pair(contacts=None):
    return FiniteBondsGPU(X, np.array([.8, .8]), np.array([[0, 1]]),
                         np.array([np.pi*.12**2]), np.ones(2), np.zeros(2, bool), contacts=contacts)


def run_checks():
    result = {}
    errors = {}
    for mode, amplitude in [('normal', .3), ('shear', .4), ('bend', .8), ('twist', .8)]:
        a = Adapter(pair())
        prescribed(a, mode, 0., amplitude, steps=256)
        value = np.r_[a.fn, a.fs.ravel(), a.mt, a.mb.ravel()]
        error = np.linalg.norm(value-expected(mode, amplitude))/np.linalg.norm(expected(mode, amplitude))
        assert error < 2e-6, (mode, error)
        prescribed(a, mode, amplitude, 0., steps=256)
        assert a.gpu.energy() < 1e-20
        errors[mode] = float(error)
    result['analytic_modes_relative_errors'] = errors
    # Full network with unequal radii and arbitrary coupled translations/spins.
    rng = np.random.default_rng(953)
    p = rng.normal(size=(8, 3)); r = rng.uniform(.6, 1.1, len(p))
    pairs = np.array([(i, j) for i in range(len(p)) for j in range(i+1, len(p))])
    length = np.linalg.norm(p[pairs[:, 1]]-p[pairs[:, 0]], axis=1)
    cpu = ParallelBonds(p, pairs, r, .12, 1200/length, 480/length)
    gpu = FiniteBondsGPU(p, r, pairs, cpu.area, np.ones(len(p)), np.zeros(len(p), bool))
    a = Adapter(gpu)
    for _ in range(80):
        v, w = rng.normal(0, .2, (len(p), 3)), rng.normal(0, 1., (len(p), 3))
        end = cpu.x+.01*v
        cpu.advance(end, v, w, .01); a.advance(end, v, w, .01)
    error = max(np.linalg.norm(getattr(a, name)-getattr(cpu, name))/max(np.linalg.norm(getattr(cpu, name)), 1e-20)
                for name in ('fn', 'fs', 'mt', 'mb'))
    f, t = a.loads(); fc, tc = cpu.loads()
    assert error < 1e-12 and np.linalg.norm(f-fc)+np.linalg.norm(t-tc) < 1e-10
    assert np.linalg.norm(f.sum(0)) < 1e-10
    assert np.linalg.norm((np.cross(a.x, f)+t).sum(0)) < 1e-10
    result['gpu_vs_original_cpu_history_relative_error'] = float(error)
    # Unloaded rigid motion must not create stress, including face offset.
    virgin = []
    for contacts in (None, np.array([[.08, .15, -.06]])):
        a = Adapter(pair(contacts))
        rigid_motion(a, 2.3, 80)
        residual = np.linalg.norm(np.r_[a.fn, a.fs.ravel(), a.mt, a.mb.ravel()])
        assert residual < 1e-10
        virgin.append(float(residual))
    result['unloaded_rigid_motion_classical_and_offset'] = virgin
    # Loaded rigid motion: rotating history, axial spin, frame convergence.
    rigid_errors = []
    for steps in (20, 40, 80):
        a = Adapter(pair()); prescribed(a, 'shear', 0., .08)
        f0, t0 = a.loads(); e0 = a.gpu.energy()
        rot = rigid_motion(a, 2.3, steps)
        f, t = a.loads()
        err = np.linalg.norm(np.r_[f-f0@rot.T, t-t0@rot.T])/np.linalg.norm(np.r_[f0, t0])
        assert abs(a.gpu.energy()-e0) < 1e-10
        rigid_errors.append(float(err))
    assert rigid_errors[-1] < 1e-4 and rigid_errors[0]/rigid_errors[1] > 3.5
    result['loaded_rigid_rotation_20_40_80_errors'] = rigid_errors
    a = Adapter(pair()); prescribed(a, 'shear', 0., .08)
    before = a.fs.copy(); rot = rigid_motion(a, 1.7, 80, a.normal[0])
    axial = np.linalg.norm(a.fs-before@rot.T)/np.linalg.norm(before)
    assert axial < 1e-11
    result['axial_history_rotation_error'] = float(axial)
    # Trial rejection cannot mutate physical state.
    g = pair(); snapshot = {k: getattr(g, k).clone() for k in ('x', 'v', 'rotation', 'fn', 'fs', 'mt', 'mb', 'offset')}
    g.trial(g.tensor([[0, 0, 0, 0, 0, 0], [.1, .03, 0, .01, 0, 0]]))
    try:
        g.trial(torch.ones_like(g.mass))
    except RuntimeError:
        pass
    else:
        raise AssertionError('Oversized rotation was not rejected')
    assert all(torch.equal(getattr(g, k), v) for k, v in snapshot.items())
    result['trial_is_transactional'] = True
    # Regression: a cell with 1e-18 mass must still satisfy its local equation.
    # One axial spring has an independent scalar implicit-midpoint solution.
    g = FiniteBondsGPU(X, np.ones(2), np.array([[0, 1]]), np.array([np.pi*.12**2]),
                      np.array([1., 1e-18]), np.array([True, False]))
    g.step(.01, torch.zeros_like(g.mass))
    assert g.energy() == 0 and torch.equal(g.x, g.p)
    load = torch.zeros_like(g.mass); load[1, 0] = 1e-5
    u, velocity, dt = 0., 0., .01
    for _ in range(8):
        dq = (dt*1e-18*velocity+.5*dt*dt*(1e-5-float(g.kn[0])*u))/(1e-18+dt*dt*float(g.kn[0])/4)
        u += dq; velocity = 2*dq/dt-velocity
        g.step(dt, load)
    tiny_error = abs(float(g.x[1, 0]-g.p[1, 0])-u)
    assert tiny_error < 1e-12
    result['zero_load_rest_and_tiny_mass_scalar_solution_error'] = tiny_error
    # Compare nonlinear implicit motion with the earlier explicit CPU solver
    # on a free network, using small steps for the independent reference.
    p = np.array([[0., 0., 0.], [1., .2, 0.], [.1, .8, .3]])
    r = np.array([.8, .7, .9]); pairs = np.array([[0, 1], [0, 2], [1, 2]])
    length = np.linalg.norm(p[pairs[:, 1]]-p[pairs[:, 0]], axis=1)
    cpu = ParallelBonds(p, pairs, r, .12, 1200/length, 480/length)
    dyn = BondedDynamics(cpu, np.ones(3), .4*r*r)
    v = rng.normal(0, .2, (3, 3)); w = rng.normal(0, .5, (3, 3))
    dyn.velocity[:], dyn.omega[:] = v, w
    for _ in range(1600): dyn.step(.1/1600)
    errors = []
    for steps in (20, 40, 80):
        g = FiniteBondsGPU(p, r, pairs, cpu.area, np.ones(3), np.zeros(3, bool))
        g.v = g.tensor(np.c_[v, w]); initial = g.energy()
        momentum0 = (g.mass[:, :3]*g.v[:, :3]).sum(0)
        angular0 = (torch.linalg.cross(g.x, g.mass[:, :3]*g.v[:, :3])+g.mass[:, 3:]*g.v[:, 3:]).sum(0)
        for _ in range(steps): g.step(.1/steps, torch.zeros_like(g.mass), tolerance=1e-10)
        error = np.linalg.norm(g.x.cpu().numpy()-cpu.x)
        errors.append(float(error))
        assert abs(g.energy()/initial-1) < 1e-7
        assert torch.linalg.vector_norm((g.mass[:, :3]*g.v[:, :3]).sum(0)-momentum0) < 1e-8
        angular = (torch.linalg.cross(g.x, g.mass[:, :3]*g.v[:, :3])+g.mass[:, 3:]*g.v[:, 3:]).sum(0)
        assert torch.linalg.vector_norm(angular-angular0) < 1e-8
    assert errors[-1] < 1e-5 and errors[0]/errors[1] > 3.
    result['implicit_vs_original_explicit_20_40_80_position_errors'] = errors
    result['free_dynamics_energy_linear_and_angular_momentum'] = 'passed'
    return result


if __name__ == '__main__':
    result = run_checks()
    out = Path('output/powersim_bread/finite_dem'); out.mkdir(parents=True, exist_ok=True)
    import hashlib
    result['source_sha256'] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                              for name in ('bonded_dem_finite_checks.py', 'bonded_dem_finite_gpu.py')}
    (out/'checks.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)
