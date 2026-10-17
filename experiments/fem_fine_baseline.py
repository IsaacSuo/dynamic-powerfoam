"""Baseline: differentiable P2 FEM on an independent (coarser) mesh drives the cells.

Learns global log(lambda), log(mu) (relative to 400) from the three training loads,
observing only the material points at the cell centres, exactly like the DEM runs:
same loads, timing, substeps, loss and held-out cases. Implicit midpoint with the
consistent mass matrix; exact discrete adjoint (K = lambda K_lam + mu K_mu).

Caveat: the targets are linear-elastic FEM, so this baseline has the correct model
class and only a coarser mesh; that is an advantage the DEM does not have.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from skfem import asm
from skfem.models.elasticity import linear_elasticity

from experiments.foam_fem_reference import Reference, CASES, FRESH, cholesky
from experiments.foam_linear_inverse import load_case

TRAIN = ('train_down', 'train_pull', 'train_twist')
SUBSTEPS = dict(train_down=16, train_pull=32, train_twist=16)


class FineFEM:
    def __init__(self, geometry, amplitudes):
        self.ref = Reference(geometry, CASES+FRESH)
        r = self.ref
        k_lam = asm(linear_elasticity(1., 0.), r.basis).tocsc()
        k_mu = asm(linear_elasticity(0., 1.), r.basis).tocsc()
        self.K_lam = k_lam[r.free][:, r.free].tocsc()
        self.K_mu = k_mu[r.free][:, r.free].tocsc()
        self.M = r.M
        self.loads = {name: r.loads[:, k]*amplitudes[name] for k, name in enumerate(r.cases)}
        self.loads['test_strong_down'] = 2*self.loads['train_down']
        # Centre positions of the free cells from the free DOFs.
        self.probe = r.probe[:, r.free]
        self.points = r.points
        self.cells = ~r.g['fixed']

    def centres(self, u):
        return self.points+(self.probe@u).reshape(3, -1).T

    def simulate(self, lam, mu, data, name, substeps, gradient=False):
        K = lam*self.K_lam+mu*self.K_mu
        dt = data['dt']/substeps
        count = len(data['target'])-1
        steps = count*substeps
        times = (np.arange(steps)+.5)*dt
        pulse = np.where(times < data['pulse'], np.sin(np.pi*times/data['pulse'])**2, 0.)
        A = (self.M+.25*dt*dt*K).tocsc()
        factor = cholesky(A)
        h = .5*dt*dt
        f = self.loads[name]
        n = A.shape[0]
        u, v = np.zeros(n), np.zeros(n)
        U = np.zeros((steps+1, n)) if gradient else None
        Y = np.zeros((steps, n)) if gradient else None
        frames = [self.centres(u)]
        for k in range(steps):
            y = factor.solve(dt*(self.M@v)-h*(K@u)+h*pulse[k]*f)
            if gradient:
                Y[k] = y
            u = u+y
            v = 2*y/dt-v
            if gradient:
                U[k+1] = u
            if (k+1) % substeps == 0:
                frames.append(self.centres(u))
        positions = np.array(frames)
        target, cells = data['target'], self.cells
        scale2 = np.mean((target[1:, cells]-self.points[cells])**2)
        residual = positions[1:, cells]-target[1:, cells]
        loss = float(np.mean(residual**2)/scale2)
        if not gradient:
            return loss, positions, None
        dpos = np.zeros(positions.shape)
        dpos[1:, cells] = 2*residual/(residual.size*scale2)
        a = (dpos.transpose(0, 2, 1).reshape(len(positions), -1))@self.probe  # (frames, n)
        ubar, vbar = a[-1].copy(), np.zeros(n)
        g_lam = g_mu = 0.
        for k in range(steps-1, -1, -1):
            mu_k = factor.solve(ubar+(2/dt)*vbar)
            w = U[k]+.5*Y[k]
            g_lam += -h*mu_k@(self.K_lam@w)
            g_mu += -h*mu_k@(self.K_mu@w)
            ubar = ubar-h*(K@mu_k)
            vbar = -vbar+dt*(self.M@mu_k)
            if k % substeps == 0 and k:
                ubar = ubar+a[k//substeps]
        return loss, positions, np.array([lam*g_lam, mu*g_mu])  # d loss / d log(lambda, mu)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--geometry', type=Path, required=True)
    p.add_argument('--targets', type=Path, nargs='+', required=True, help='Target dirs holding all 12 cases.')
    p.add_argument('--configs', type=Path, nargs='+', required=True, help='config.json files with case amplitudes.')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seed', type=int, default=-1)
    p.add_argument('--max-iterations', type=int, default=200)
    p.add_argument('--check-gradient', action='store_true')
    args = p.parse_args()
    amplitudes = {}
    for c in args.configs:
        cfg = json.loads(c.read_text())
        amplitudes.update(dict(zip(cfg['load_cases'], cfg['amplitudes'])))
    model = FineFEM(args.geometry, amplitudes)
    data = {}
    for d in args.targets:
        for f in d.glob('*_target.npz'):
            data.setdefault(f.name[:-len('_target.npz')], load_case(d, f.name[:-len('_target.npz')]))
    x0 = np.zeros(2) if args.seed < 0 else np.random.default_rng(args.seed).uniform(-1, 1, 2)

    def objective(x, cases=TRAIN):
        lam, mu = 400*np.exp(x)
        total, grad = 0., np.zeros(2)
        for name in cases:
            loss, _, g = model.simulate(lam, mu, data[name], name, SUBSTEPS[name], gradient=True)
            total += loss/len(cases); grad += g/len(cases)
        return total, grad

    if args.check_gradient:
        x = x0+.1
        d = np.array([.7, -.4])
        _, g = objective(x, ('train_down',))
        e = 1e-5
        fd = (objective(x+e*d, ('train_down',))[0]-objective(x-e*d, ('train_down',))[0])/(2*e)
        print('finite difference', fd, 'adjoint', g@d, 'relative', abs(fd-g@d)/abs(fd))
        return
    args.out.mkdir(parents=True, exist_ok=False)
    history = []
    start = time.perf_counter()
    def wrapped(x):
        v, g = objective(x)
        history.append(dict(x=x.tolist(), objective=v, seconds=time.perf_counter()-start))
        print(json.dumps(history[-1]), flush=True)
        return v, g
    fit = minimize(wrapped, x0, jac=True, method='L-BFGS-B', bounds=[(-np.log(1e4), np.log(1e4))]*2,
                   options=dict(maxiter=args.max_iterations, gtol=1e-8, ftol=0.))
    lam, mu = 400*np.exp(fit.x)
    metrics = {}
    for name in sorted(data):
        loss, _, _ = model.simulate(lam, mu, data[name], name, SUBSTEPS.get(name, 32))
        metrics[name] = float(np.sqrt(loss))
    report = dict(stop_message=str(fit.message), iterations=int(fit.nit), evaluations=len(history),
                  lam=float(lam), mu=float(mu), E=float(mu*(3*lam+2*mu)/(lam+mu)), nu=float(lam/(2*(lam+mu))),
                  free_dofs=int(model.K_lam.shape[0]), seconds=time.perf_counter()-start, normalized_rmse=metrics)
    (args.out/'report.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
