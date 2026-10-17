"""Small-system implicit adjoint for the existing finite parallel-bond law.

Uses FiniteBondsGPU.trial itself (with its no-grad decorator unwrapped), not a
new spring law. All committed histories remain in the time-backpropagation
graph. Dense Jacobians are deliberately limited to small CPU experiments.
"""
import copy

import torch

from experiments.bonded_dem_finite_gpu import FiniteBondsGPU, exp_rotation


STATE_NAMES = ('x', 'v', 'rotation', 'normal', 'offset', 'fn', 'fs', 'mt', 'mb')
STIFFNESS_NAMES = ('kn', 'ks', 'kb', 'kt')


class _ImplicitIncrement(torch.autograd.Function):
    @staticmethod
    def forward(ctx, model, dt, external, parameters, *state):
        root = model.solve(parameters, state, external, dt)
        ctx.model, ctx.dt = model, dt
        ctx.save_for_backward(root, external, parameters, *state)
        return root

    @staticmethod
    def backward(ctx, root_gradient):
        root, external, parameters, *state = ctx.saved_tensors
        with torch.enable_grad():
            root = root.detach().requires_grad_(True)
            variables = [v.detach().requires_grad_(True) for v in (external, parameters, *state)]
            load, theta, *history = variables
            def residual(z):
                return ctx.model.residual(z, theta, history, load, ctx.dt)
            jacobian = torch.autograd.functional.jacobian(residual, root, vectorize=True)
            adjoint = torch.linalg.solve(jacobian.T, root_gradient)
            gradients = torch.autograd.grad(residual(root), variables, -adjoint, allow_unused=True)
        return (None, None, *gradients)


class DifferentiableBonds:
    def __init__(self, volume_term=None, **kwargs):
        self.reference = FiniteBondsGPU(**kwargs)
        s = self.reference
        self.free = torch.nonzero(~s.fixed.repeat_interleave(6)).flatten()
        if len(self.free) > 240:
            raise ValueError('Dense differentiable pilot is limited to 240 free DOFs')
        self.units = torch.ones_like(s.mass)
        self.units[:, :3] = s.radii[:, None]
        self.diagonal = s._diagonal_blocks().diagonal(dim1=-2, dim2=-1).clone()
        self.base_stiffness = tuple(getattr(s, key).clone() for key in STIFFNESS_NAMES)
        self.volume_term = volume_term
        self.parameter_count = 4 if volume_term is None else 5
        if volume_term is not None:
            self.volume_term = volume_term.to(s.p)
            if volume_term.reference.shape != s.p.shape or not torch.allclose(volume_term.reference, s.p):
                raise ValueError('Volume and pair models must have identical reference centers.')
            # Fixed reference scaling only; the accepted force and Newton
            # Jacobian use the full nonlinear midpoint volume law below.
            zero = torch.zeros(len(s.p), dtype=s.p.dtype, device=s.p.device)
            self.diagonal[:, :3] += volume_term.reference_tangent(zero).diagonal().reshape(-1, 3)
        self.stats = []

    def initial_state(self):
        return tuple(getattr(self.reference, key).clone() for key in STATE_NAMES)

    def bind(self, parameters, state):
        s = copy.copy(self.reference)
        if parameters.shape != (len(s.p), self.parameter_count):
            raise ValueError(f'Expected {self.parameter_count} log multipliers per cell.')
        for key, value in zip(STATE_NAMES, state):
            setattr(s, key, value)
        # Symmetric, positive geometric mean of the two cell multipliers.
        # This is an explicit parameterization choice, not a material identity.
        multipliers = torch.exp(.5*(parameters[s.i]+parameters[s.j]))
        for column, (key, base) in enumerate(zip(STIFFNESS_NAMES, self.base_stiffness)):
            setattr(s, key, base*multipliers[:, column])
        return s

    def increment(self, z):
        flat = torch.zeros(self.reference.mass.numel(), dtype=z.dtype, device=z.device)
        return flat.index_copy(0, self.free, z).reshape_as(self.units)*self.units

    def trial(self, parameters, state, y):
        result = FiniteBondsGPU.trial.__wrapped__(self.bind(parameters, state), y)
        if self.volume_term is not None:
            mid = state[0]+.5*y[:, :3]
            # Reject invalid trial/endpoint geometry during Newton line search.
            # A signed-volume energy is not itself an inversion barrier.
            try:
                self.volume_term.validate_configuration(mid)
                self.volume_term.validate_configuration(result['x'])
            except ValueError as error:
                raise RuntimeError('Subdivide: nonpositive tetrahedron volume') from error
            _, force, _ = self.volume_term.evaluate(mid, parameters[:, 4])
            result['force'] = result['force']+torch.cat((force, torch.zeros_like(force)), dim=1)
        return result

    def residual(self, z, parameters, state, external, dt):
        y = self.increment(z)
        force = self.trial(parameters, state, y)['force']
        mass = self.reference.mass
        value = mass*(y-dt*state[1])-.5*dt*dt*(force+external)
        scale = (mass+.25*dt*dt*self.diagonal)*self.units
        return (value/scale).flatten()[self.free]

    def solve(self, parameters, state, external, dt):
        parameters = parameters.detach()
        state = tuple(v.detach() for v in state)
        external = external.detach()
        z = torch.zeros(len(self.free), dtype=parameters.dtype, device=parameters.device)
        with torch.enable_grad():
            def residual(value):
                return self.residual(value, parameters, state, external, dt)
            for iteration in range(16):
                value = residual(z)
                defect = float(value.detach().abs().max())
                if defect < 1e-11:
                    self.stats.append(dict(iterations=iteration, scaled_residual=defect))
                    return z.detach()
                jacobian = torch.autograd.functional.jacobian(residual, z, vectorize=True)
                correction = torch.linalg.solve(jacobian, -value.detach())
                merit = float(value.detach().norm())
                for backtrack in range(12):
                    candidate = (z+2.**(-backtrack)*correction).detach()
                    try:
                        candidate_merit = float(residual(candidate).detach().norm())
                    except RuntimeError as error:
                        if 'Subdivide:' not in str(error):
                            raise
                        continue
                    if candidate_merit < (1-1e-4*2.**(-backtrack))*merit:
                        z = candidate
                        break
                else:
                    raise RuntimeError('Differentiable pilot Newton line search failed')
        raise RuntimeError(f'Differentiable pilot did not converge: {defect:g}')

    def step(self, state, parameters, external, dt):
        z = _ImplicitIncrement.apply(self, dt, external, parameters, *state)
        y = self.increment(z)
        trial = self.trial(parameters, state, y)
        updated = dict(trial, v=2*y/dt-state[1], rotation=exp_rotation(y[:, 3:])@state[2])
        return tuple(updated[key] for key in STATE_NAMES)

    def rollout(self, parameters, loads, dt):
        state = self.initial_state()
        positions = [state[0]]
        for external in loads:
            state = self.step(state, parameters, external, dt)
            positions.append(state[0])
        return torch.stack(positions), state
