"""Consistent, scaled Newton-FGMRES for the unchanged finite bond rate law.

Differentiate the complete current-step residual with forward AD. Committed
history is constant during a solve; no differentiation through earlier steps.
The old elastic tangent is only a right preconditioner. No model/mass changes.
"""
import numpy as np
import torch

from experiments.bonded_dem_finite_gpu import FiniteBondsGPU, exp_rotation, transport, dot


def fgmres(operator, rhs, precondition, tolerance=1e-6, restart=32, max_iterations=256):
    """Right-preconditioned flexible GMRES, twice-orthogonalized Arnoldi.

    Store preconditioned basis vectors: an inexact inner CG is not a fixed
    linear map. Always verify the TRUE linear residual before returning.
    """
    shape = rhs.shape
    b = rhs.flatten(); norm = float(b.norm())
    x = torch.zeros_like(b)
    if norm == 0: return x.reshape(shape), dict(iterations=0, relative_residual=0.)
    residual = b.clone(); iterations = 0
    while iterations < max_iterations:
        width = min(restart, max_iterations-iterations)
        basis = torch.empty((width+1, len(b)), dtype=b.dtype, device=b.device)
        transformed = torch.empty((width, len(b)), dtype=b.dtype, device=b.device)
        h = np.zeros((width+1, width)); cs, sn = np.zeros(width), np.zeros(width)
        g = np.zeros(width+1); g[0] = float(residual.norm())
        basis[0] = residual/g[0]
        for j in range(width):
            z = precondition(basis[j].reshape(shape)).flatten()
            transformed[j] = z
            w = operator(z.reshape(shape)).flatten()
            projections = torch.zeros(j+1, device=b.device, dtype=b.dtype)
            for _ in range(2):
                values = basis[:j+1]@w
                projections += values
                w = w-basis[:j+1].T@values
            h[:j+1, j] = projections.cpu().numpy()
            h[j+1, j] = float(w.norm())
            breakdown = h[j+1, j] < 1e-14
            if not breakdown: basis[j+1] = w/h[j+1, j]
            for k in range(j):
                h[k, j], h[k+1, j] = cs[k]*h[k, j]+sn[k]*h[k+1, j], -sn[k]*h[k, j]+cs[k]*h[k+1, j]
            denominator = np.hypot(h[j, j], h[j+1, j])
            if denominator == 0: raise RuntimeError('Subdivide: singular Newton linear system')
            cs[j], sn[j] = h[j, j]/denominator, h[j+1, j]/denominator
            h[j, j], h[j+1, j] = denominator, 0.
            g[j+1], g[j] = -sn[j]*g[j], cs[j]*g[j]
            iterations += 1
            if abs(g[j+1]) <= tolerance*norm or breakdown or j+1 == width:
                coefficients = np.linalg.solve(h[:j+1, :j+1], g[:j+1])
                x += transformed[:j+1].T@torch.as_tensor(coefficients, dtype=b.dtype, device=b.device)
                residual = b-operator(x.reshape(shape)).flatten()
                relative = float(residual.norm())/norm
                if relative <= 1.05*tolerance:
                    return x.reshape(shape), dict(iterations=iterations, relative_residual=relative)
                break
    raise RuntimeError(f'Subdivide: FGMRES did not converge ({relative:g})')


class NewtonBonds(FiniteBondsGPU):
    def midpoint_force(self, y):
        """Same midpoint force; omit end-history work unused by its derivative.

        Primal trial/commit and their guards remain the original implementation.
        Equality with trial()['force'] is verified on loaded offset networks.
        """
        mid = self.x+.5*y[:, :3]
        n, contact = self.geometry(mid)
        half_spin = .25*(y[self.i, 3:]+y[self.j, 3:])
        contact = contact+transport(self.offset, self.normal, n, half_spin)
        ri, rj = contact-mid[self.i], contact-mid[self.j]
        fs0 = transport(self.fs, self.normal, n, half_spin)
        mb0 = transport(self.mb, self.normal, n, half_spin)
        du = y[self.j, :3]-y[self.i, :3]+torch.linalg.cross(y[self.j, 3:], rj)-torch.linalg.cross(y[self.i, 3:], ri)
        dw = y[self.j, 3:]-y[self.i, 3:]
        dn, twist = dot(du, n), dot(dw, n)
        return self.assemble(self.fn+.5*self.kn*dn, fs0+.5*self.ks[:, None]*(du-dn[:, None]*n),
            self.mt+.5*self.kt*twist, mb0+.5*self.kb[:, None]*(dw-twist[:, None]*n), n, ri, rj)

    def residual(self, y, dt, external):
        trial = self.trial(y)
        r = self.mass*(y-dt*self.v)-.5*dt*dt*(trial['force']+external)
        return torch.where(self.fixed[:, None], y, r)

    def jacobian_vector(self, y, direction, dt, external):
        def residual_midpoint(z):
            r = self.mass*(z-dt*self.v)-.5*dt*dt*(self.midpoint_force(z)+external)
            return torch.where(self.fixed[:, None], z, r)
        return torch.func.jvp(residual_midpoint, (y,), (direction,))[1]

    def prepare_scaling(self, trial, dt):
        self.n, self.ri, self.rj = trial['n'], trial['ri'], trial['rj']
        self.diagonal = self._diagonal_blocks()
        self.block = torch.diag_embed(self.mass)+.25*dt*dt*self.diagonal
        self.block[self.fixed] = torch.eye(6, dtype=self.p.dtype, device=self.device)
        self.preconditioner = torch.linalg.inv(self.block)
        self.units = torch.ones_like(self.mass)
        self.units[:, :3] = self.radii[:, None]
        self.units[self.fixed] = 1.

    def scale_residual(self, r):
        return torch.einsum('nij,nj->ni', self.preconditioner, r)/self.units

    def elastic_precondition(self, vector, dt, tolerance=.03, max_iterations=80):
        """Approximately invert the coupled elastic graph, without changing J.

        Scaling: solve A w = B S vector and return S^-1 w. Inner CG may
        terminate early; FGMRES tolerates this variable preconditioner.
        """
        rhs = torch.einsum('nij,nj->ni', self.block, self.units*vector)
        rhs[self.fixed] = 0
        x = torch.zeros_like(rhs); r = rhs.clone()
        z = torch.einsum('nij,nj->ni', self.preconditioner, r)
        direction = z.clone(); rz = (r*z).sum(); initial = float(rz)
        if initial == 0: return x
        for iteration in range(max_iterations):
            ad = self._operator(direction, .25*dt*dt)
            denominator = (direction*ad).sum()
            alpha = rz/denominator
            x += alpha*direction; r -= alpha*ad
            z = torch.einsum('nij,nj->ni', self.preconditioner, r)
            following = (r*z).sum()
            if iteration % 5 == 0 and float(following) <= tolerance*tolerance*initial:
                break
            direction = z+(following/rz)*direction; rz = following
        self.inner_iterations += iteration+1
        return x/self.units

    @torch.no_grad()
    def step(self, dt, external, tolerance=2e-7, max_iterations=12, *,
             pose_tolerance=2e-6, backend='fgmres', linear_tolerance=1e-6,
             preconditioning='elastic'):
        if not np.isfinite(dt) or dt <= 0: raise ValueError('Positive finite dt required')
        y = torch.zeros_like(self.v)
        rhs_scale = max(float((dt*self.mass*self.v+.5*dt*dt*external).norm()),
                        .5*dt*dt*float(self.loads()[~self.fixed].norm()), 1e-30)
        krylov, self.inner_iterations = 0, 0
        residual_trace = []
        for iteration in range(max_iterations+1):
            trial = self.trial(y)
            self.prepare_scaling(trial, dt)
            r = self.residual(y, dt, external)
            scaled = self.scale_residual(r)
            relative, local = float(r.norm())/rhs_scale, float(scaled.abs().max())
            residual_trace.append([relative, local])
            if relative < tolerance and local < pose_tolerance:
                velocity = 2*y/dt-self.v
                rotation = exp_rotation(y[:, 3:])@self.rotation
                self.commit(trial)
                self.v, self.rotation = velocity, rotation
                self.time += dt
                return float((external*y).sum()), dict(nonlinear_iterations=iteration,
                    nonlinear_relative_residual=relative, local_pose_defect=local,
                    krylov_iterations=krylov, cg_iterations=self.inner_iterations,
                    residual_trace=residual_trace,
                    max_step_rotation=float(torch.linalg.vector_norm(y[:, 3:], dim=1).max()))
            if iteration == max_iterations: break
            def operator(v):
                return self.scale_residual(self.jacobian_vector(y, self.units*v, dt, external))
            if backend == 'dense':
                free = torch.nonzero(~self.fixed.repeat_interleave(6)).flatten()
                if len(free) > 400: raise ValueError('Dense reference limited to 400 free DOFs')
                eye = torch.eye(len(free), dtype=y.dtype, device=y.device)
                columns = []
                for col in eye:
                    direction = torch.zeros_like(y).flatten(); direction[free] = col
                    columns.append(operator(direction.reshape_as(y)).flatten()[free])
                jac = torch.stack(columns, dim=1)
                delta = torch.zeros_like(y).flatten()
                delta[free] = torch.linalg.solve(jac, -scaled.flatten()[free])
                delta = delta.reshape_as(y)
            elif backend == 'fgmres':
                if preconditioning == 'elastic':
                    precondition = lambda v: self.elastic_precondition(v, dt)
                elif preconditioning == 'block':
                    precondition = lambda v: v
                else:
                    raise ValueError(preconditioning)
                delta, metrics = fgmres(operator, -scaled, precondition, tolerance=linear_tolerance)
                krylov += metrics['iterations']
            else:
                raise ValueError(backend)
            correction = self.units*delta
            correction[self.fixed] = 0
            merit = float(scaled.norm())
            accepted = False
            for power in range(10):
                factor = 2.**(-power)
                candidate = y+factor*correction
                try:
                    value = self.scale_residual(self.residual(candidate, dt, external))
                except RuntimeError as error:
                    if 'Subdivide:' not in str(error): raise
                    continue
                if float(value.norm()) < (1-1e-4*factor)*merit:
                    y = candidate; accepted = True; break
            if not accepted: raise RuntimeError('Subdivide: Newton line search failed')
        raise RuntimeError(f'Subdivide: Newton did not converge ({relative:g}, {local:g})')
