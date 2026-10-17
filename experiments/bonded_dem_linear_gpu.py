"""Matrix-free, small-strain 6-DOF bonded DEM on CUDA, float64.

This is the stress-free linearization of the validated parallel-bond model.
It retains independent translations/spins for every body, with fixed reference
face geometry. It is not the finite-rotation, history-dependent nonlinear DEM.
Implicit midpoint plus block-preconditioned CG avoids changing masses to cope
with tiny cells. No modal reduction, interpolated driving field, or radius law.
"""

import numpy as np
import torch


class LinearBondsGPU:
    def __init__(self, points, radii, pairs, areas, contacts, masses, fixed,
                 young=1200., shear=480., device='cuda'):
        self.device = device
        self.tensor = lambda x: torch.as_tensor(x, dtype=torch.float64, device=device)
        self.p = self.tensor(points)
        self.edges = torch.as_tensor(pairs, dtype=torch.long, device=device)
        self.i, self.j = self.edges.T
        r = self.tensor(radii); area = self.tensor(areas); c = self.tensor(contacts)
        distance = self.p[self.j]-self.p[self.i]
        length = torch.linalg.vector_norm(distance, dim=1)
        self.n = distance/length[:, None]
        self.ri, self.rj = c-self.p[self.i], c-self.p[self.j]
        self.kn, self.ks = young*area/length, shear*area/length
        second = area**2/(4*np.pi)  # equivalent circular cross section
        self.kb, self.kt = young*second/length, shear*(2*second)/length
        mass = self.tensor(masses)
        inertia = .4*mass*r*r  # explicit isotropic proxy about generator
        self.mass = torch.cat((mass[:, None].expand(-1, 3), inertia[:, None].expand(-1, 3)), dim=1)
        if bool(torch.any(self.mass <= 0)):
            raise ValueError('Only positive-mass bodies can enter this solver')
        self.fixed = torch.as_tensor(fixed, dtype=torch.bool, device=device)
        self.q, self.v = torch.zeros_like(self.mass), torch.zeros_like(self.mass)
        self.kq = torch.zeros_like(self.mass)
        self.diagonal = self._diagonal_blocks()
        self.preconditioner = None
        self.preconditioner_dt = None

    @torch.no_grad()
    def stiffness(self, q):
        a, b = q[self.i], q[self.j]
        du = b[:, :3]-a[:, :3]+torch.linalg.cross(b[:, 3:], self.rj)-torch.linalg.cross(a[:, 3:], self.ri)
        dw = b[:, 3:]-a[:, 3:]
        f = self.ks[:, None]*du+(self.kn-self.ks)[:, None]*(du*self.n).sum(1)[:, None]*self.n
        m = self.kb[:, None]*dw+(self.kt-self.kb)[:, None]*(dw*self.n).sum(1)[:, None]*self.n
        out = torch.zeros_like(q)
        out.index_add_(0, self.i, torch.cat((-f, -torch.linalg.cross(self.ri, f)-m), dim=1))
        out.index_add_(0, self.j, torch.cat((f, torch.linalg.cross(self.rj, f)+m), dim=1))
        return out

    @torch.no_grad()
    def _diagonal_blocks(self):
        eye = torch.eye(3, device=self.p.device, dtype=self.p.dtype)
        nn = self.n[:, :, None]*self.n[:, None, :]
        d = self.ks[:, None, None]*eye+(self.kn-self.ks)[:, None, None]*nn
        h = self.kb[:, None, None]*eye+(self.kt-self.kb)[:, None, None]*nn
        diagonal = torch.zeros((len(self.p), 6, 6), device=self.p.device, dtype=self.p.dtype)
        for indices, r in ((self.i, self.ri), (self.j, self.rj)):
            s = torch.zeros_like(d)
            s[:, 0, 1], s[:, 0, 2] = -r[:, 2], r[:, 1]
            s[:, 1, 0], s[:, 1, 2] = r[:, 2], -r[:, 0]
            s[:, 2, 0], s[:, 2, 1] = -r[:, 1], r[:, 0]
            block = torch.zeros((len(indices), 6, 6), device=self.p.device, dtype=self.p.dtype)
            block[:, :3, :3] = d
            block[:, :3, 3:] = -d@s
            block[:, 3:, :3] = (block[:, :3, 3:]).transpose(1, 2)
            block[:, 3:, 3:] = s.transpose(1, 2)@d@s+h
            diagonal.index_add_(0, indices, block)
        return diagonal

    def _operator(self, q, alpha):
        out = self.mass*q+alpha*self.stiffness(q)
        out[self.fixed] = q[self.fixed]
        return out

    @torch.no_grad()
    def solve_increment(self, rhs, dt, guess=None, tolerance=2e-8, max_iterations=1000):
        alpha = dt*dt/4
        if self.preconditioner_dt != dt:
            diagonal = torch.diag_embed(self.mass)+alpha*self.diagonal
            diagonal[self.fixed] = torch.eye(6, device=self.p.device, dtype=self.p.dtype)
            self.preconditioner = torch.linalg.inv(diagonal)
            self.preconditioner_dt = dt
        rhs = rhs.clone(); rhs[self.fixed] = 0
        norm = float(torch.linalg.vector_norm(rhs))
        if norm < 1e-30:
            return torch.zeros_like(rhs), dict(iterations=0, relative_residual=0.)
        x = torch.zeros_like(rhs) if guess is None else guess.clone()
        x[self.fixed] = 0
        residual = rhs-self._operator(x, alpha)
        z = torch.einsum('nij,nj->ni', self.preconditioner, residual)
        direction = z.clone(); rz = (residual*z).sum()
        for iteration in range(max_iterations):
            ad = self._operator(direction, alpha)
            step = rz/(direction*ad).sum()
            x += step*direction; residual -= step*ad
            if iteration % 10 == 0:
                relative = float(torch.linalg.vector_norm(residual))/norm
                if relative < tolerance:
                    break
                if not np.isfinite(relative):
                    raise RuntimeError('Nonfinite conjugate-gradient residual')
            z = torch.einsum('nij,nj->ni', self.preconditioner, residual)
            next_rz = (residual*z).sum()
            direction = z+(next_rz/rz)*direction; rz = next_rz
        actual = float(torch.linalg.vector_norm(rhs-self._operator(x, alpha)))/norm
        if actual > 5*tolerance or not np.isfinite(actual):
            raise RuntimeError(f'Linear solve failed after {iteration+1} iterations: {actual}')
        return x, dict(iterations=iteration+1, relative_residual=actual)

    @torch.no_grad()
    def step(self, dt, external):
        # Exact implicit-midpoint equations for M q'' + K q = f_mid.
        rhs = dt*self.mass*self.v+.5*dt*dt*(external-self.kq)
        dq, metrics = self.solve_increment(rhs, dt, guess=dt*self.v)
        self.v = 2*dq/dt-self.v
        self.q += dq; self.kq = self.stiffness(self.q)
        return float((external*dq).sum()), metrics

    def energy(self):
        return float(.5*((self.mass*self.v*self.v).sum()+(self.q*self.kq).sum()))


def checks():
    from experiments.bonded_dem_core import ParallelBonds
    from experiments.bonded_dem_dynamics import reference_stiffness
    rng = np.random.default_rng(217)
    p = rng.normal(size=(12, 3))
    pairs = np.array([(i, j) for i in range(len(p)) for j in range(i+1, len(p))])
    r = rng.uniform(.8, 1.2, len(p))
    lengths = np.linalg.norm(p[pairs[:, 1]]-p[pairs[:, 0]], axis=1)
    cpu = ParallelBonds(p, pairs, r, .12, 1200/lengths, 480/lengths)
    k = reference_stiffness(cpu).toarray()
    gpu = LinearBondsGPU(p, r, pairs, cpu.area, cpu.geometry(p)[1], np.ones(len(p)), np.zeros(len(p), bool))
    q = rng.normal(size=(len(p), 6))
    actual = gpu.stiffness(gpu.tensor(q)).cpu().numpy().ravel()
    error = np.linalg.norm(actual-k@q.ravel())/np.linalg.norm(k@q.ravel())
    assert error < 1e-12
    diagonal = gpu.diagonal.cpu().numpy()
    assert max(np.max(abs(diagonal[i]-k[6*i:6*i+6, 6*i:6*i+6])) for i in range(len(p))) < 1e-10
    # Compare midpoint step to independent dense linear algebra.
    gpu.fixed[0] = True
    mass = gpu.mass.cpu().numpy().ravel(); free = np.arange(6, len(mass))
    force = rng.normal(size=(len(p), 6)); force[0] = 0
    dt = .03
    expected = np.zeros(len(mass))
    expected[free] = np.linalg.solve(np.diag(mass[free])+dt*dt*k[np.ix_(free, free)]/4,
                                     .5*dt*dt*force.ravel()[free])
    work, metrics = gpu.step(dt, gpu.tensor(force))
    step_error = np.linalg.norm(gpu.q.cpu().numpy().ravel()-expected)/np.linalg.norm(expected)
    assert step_error < 1e-6
    assert abs(gpu.energy()-work)/work < 1e-6
    return dict(gpu_vs_cpu_tangent_relative_error=float(error),
                midpoint_vs_dense_relative_error=float(step_error), linear_solver=metrics,
                work_energy_relative_error=float(abs(gpu.energy()-work)/work))


if __name__ == '__main__':
    print(checks())
