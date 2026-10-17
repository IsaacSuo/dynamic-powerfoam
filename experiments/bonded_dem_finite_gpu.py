"""Finite-motion elastic parallel bonds, ported from bonded_dem_core to CUDA.

Normal/shear/bend/twist histories use the SAME midpoint rate law and split
frame transport as ParallelBonds. Damage/contact are deliberately disabled.
Implicit midpoint solves actual nonlinear midpoint loads; its frozen-geometry
tangent is only an iterative solver approximation, never the accepted force.

Optional reference contact offsets preserve bread's reference face-centroid
levers. They rotate with the bond frame; they are not recomputed power faces.
With contacts=None this is exactly the earlier classical ball contact rule.
"""
import numpy as np
import torch

from experiments.bonded_dem_linear_gpu import LinearBondsGPU


def dot(a, b):
    return (a*b).sum(-1)


def transport(v, n0, n1, spin):
    cross = torch.linalg.cross(n0, n1)
    cosine = dot(n0, n1).clamp(-1., 1.)
    if bool(torch.any(cosine < np.cos(.25))):
        raise RuntimeError('Subdivide: normal turns more than .25 rad per half-step')
    out = v+torch.linalg.cross(cross, v)
    out += torch.linalg.cross(cross, torch.linalg.cross(cross, v))/(1+cosine)[:, None]
    bisector = torch.nn.functional.normalize(n0+n1, dim=1)
    angle = dot(spin, bisector)[:, None]
    c, s = torch.cos(angle), torch.sin(angle)
    return c*out+s*torch.linalg.cross(n1, out)+(1-c)*dot(n1, out)[:, None]*n1


def exp_rotation(v):
    angle = torch.linalg.vector_norm(v, dim=1)
    s = torch.zeros((len(v), 3, 3), device=v.device, dtype=v.dtype)
    s[:, 0, 1], s[:, 0, 2] = -v[:, 2], v[:, 1]
    s[:, 1, 0], s[:, 1, 2] = v[:, 2], -v[:, 0]
    s[:, 2, 0], s[:, 2, 1] = -v[:, 1], v[:, 0]
    return (torch.eye(3, device=v.device, dtype=v.dtype)
            +torch.sinc(angle/np.pi)[:, None, None]*s
            +(.5*torch.sinc(angle/(2*np.pi))**2)[:, None, None]*(s@s))


class FiniteBondsGPU(LinearBondsGPU):
    def __init__(self, points, radii, pairs, areas, masses, fixed, contacts=None,
                 young=1200., shear=480., device='cuda'):
        i, j = np.asarray(pairs).T
        n = np.asarray(points)[j]-np.asarray(points)[i]
        n /= np.linalg.norm(n, axis=1)[:, None]
        classical = .5*(np.asarray(points)[i]+np.asarray(points)[j])
        classical += .5*(np.asarray(radii)[i]-np.asarray(radii)[j])[:, None]*n
        if contacts is None:
            contacts = classical
        super().__init__(points, radii, pairs, areas, contacts, masses, fixed, young, shear, device)
        self.x = self.p.clone()
        self.radii = self.tensor(radii)
        self.normal = self.n.clone()
        self.offset = self.tensor(contacts-classical)
        self.fn, self.mt = torch.zeros_like(self.kn), torch.zeros_like(self.kn)
        self.fs, self.mb = torch.zeros_like(self.n), torch.zeros_like(self.n)
        self.rotation = torch.eye(3, dtype=self.p.dtype, device=device).repeat(len(points), 1, 1)
        self.time = 0.

    def geometry(self, x):
        d = x[self.j]-x[self.i]
        length = torch.linalg.vector_norm(d, dim=1)
        if bool(torch.any(length <= 1e-12)):
            raise RuntimeError('Subdivide: coincident centers')
        n = d/length[:, None]
        c = .5*(x[self.i]+x[self.j])+.5*(self.radii[self.i]-self.radii[self.j])[:, None]*n
        return n, c

    def assemble(self, fn, fs, mt, mb, n, ri, rj):
        f, couple = fn[:, None]*n+fs, mt[:, None]*n+mb
        out = torch.zeros_like(self.mass)
        out.index_add_(0, self.i, torch.cat((f, torch.linalg.cross(ri, f)+couple), 1))
        out.index_add_(0, self.j, torch.cat((-f, -torch.linalg.cross(rj, f)-couple), 1))
        return out

    def loads(self):
        n, c = self.geometry(self.x)
        c = c+self.offset
        return self.assemble(self.fn, self.fs, self.mt, self.mb, n, c-self.x[self.i], c-self.x[self.j])

    @torch.no_grad()
    def trial(self, increment, *, mid_positions=None, end_positions=None):
        """Pure constitutive trial from committed history; no history mutation."""
        if float(torch.linalg.vector_norm(increment[:, 3:], dim=1).max().detach()) > .25:
            raise RuntimeError('Subdivide: body rotation exceeds .25 rad in a step')
        mid = self.x+.5*increment[:, :3] if mid_positions is None else mid_positions
        end = self.x+increment[:, :3] if end_positions is None else end_positions
        nm, contact = self.geometry(mid)
        ne, _ = self.geometry(end)
        spin = .5*(increment[self.i, 3:]+increment[self.j, 3:])
        offset_mid = transport(self.offset, self.normal, nm, .5*spin)
        contact = contact+offset_mid
        ri, rj = contact-mid[self.i], contact-mid[self.j]
        fs0 = transport(self.fs, self.normal, nm, .5*spin)
        mb0 = transport(self.mb, self.normal, nm, .5*spin)
        du = (increment[self.j, :3]-increment[self.i, :3]
              +torch.linalg.cross(increment[self.j, 3:], rj)
              -torch.linalg.cross(increment[self.i, 3:], ri))
        dw = increment[self.j, 3:]-increment[self.i, 3:]
        dn, twist = dot(du, nm), dot(dw, nm)
        fn = self.fn+self.kn*dn
        fs = fs0+self.ks[:, None]*(du-dn[:, None]*nm)
        mt = self.mt+self.kt*twist
        mb = mb0+self.kb[:, None]*(dw-twist[:, None]*nm)
        force = self.assemble(.5*(self.fn+fn), .5*(fs0+fs),
                              .5*(self.mt+mt), .5*(mb0+mb), nm, ri, rj)
        return dict(force=force, n=nm, ri=ri, rj=rj, x=end, normal=ne, fn=fn, mt=mt,
                    fs=transport(fs, nm, ne, .5*spin),
                    mb=transport(mb, nm, ne, .5*spin),
                    offset=transport(offset_mid, nm, ne, .5*spin))

    def commit(self, trial):
        for name in ('x', 'normal', 'fn', 'fs', 'mt', 'mb', 'offset'):
            setattr(self, name, trial[name])

    def energy(self):
        elastic = .5*(self.fn**2/self.kn+dot(self.fs, self.fs)/self.ks
                      +self.mt**2/self.kt+dot(self.mb, self.mb)/self.kb).sum()
        return float(elastic+.5*(self.mass*self.v*self.v).sum())

    def local_defect(self, residual):
        correction = torch.einsum('nij,nj->ni', self.preconditioner, residual)
        scale = torch.ones_like(correction)
        scale[:, :3] = self.radii[:, None]
        return float((correction/scale).abs().max())

    @torch.no_grad()
    def solve_increment(self, rhs, dt, tolerance=2e-7, max_iterations=1600):
        """PCG with local displacement-defect AND global residual checks.

        A raw global force residual alone misses cells with masses ~1e-18.
        The block-scaled test controls their local pose error without changing
        any masses or removing bodies. The true residual is checked at exit.
        """
        diagonal = torch.diag_embed(self.mass)+dt*dt*self.diagonal/4
        diagonal[self.fixed] = torch.eye(6, device=self.device, dtype=self.p.dtype)
        self.preconditioner = torch.linalg.inv(diagonal)
        rhs = rhs.clone(); rhs[self.fixed] = 0
        norm = float(torch.linalg.vector_norm(rhs))
        if norm < 1e-40:
            return torch.zeros_like(rhs), dict(iterations=0)
        x, residual = torch.zeros_like(rhs), rhs.clone()
        z = torch.einsum('nij,nj->ni', self.preconditioner, residual)
        direction = z.clone(); rz = dot(residual.flatten(), z.flatten())
        for iteration in range(max_iterations):
            ad = self._operator(direction, dt*dt/4)
            step = rz/(direction*ad).sum()
            x += step*direction; residual -= step*ad
            if iteration % 10 == 0:
                relative = float(torch.linalg.vector_norm(residual))/norm
                if relative < tolerance and self.local_defect(residual) < 2e-8:
                    actual = rhs-self._operator(x, dt*dt/4)
                    if (float(torch.linalg.vector_norm(actual))/norm < 5*tolerance
                            and self.local_defect(actual) < 1e-7):
                        return x, dict(iterations=iteration+1)
                if not np.isfinite(relative):
                    raise RuntimeError('Nonfinite PCG residual')
            z = torch.einsum('nij,nj->ni', self.preconditioner, residual)
            next_rz = (residual*z).sum()
            direction = z+(next_rz/rz)*direction; rz = next_rz
        raise RuntimeError('Subdivide: balanced PCG did not converge')

    @torch.no_grad()
    def step(self, dt, external, tolerance=2e-7, max_iterations=18):
        """Nonlinear implicit midpoint. Commit once, only after convergence.

        M (dq-dt*v) = dt²/2 (f_mid(dq)+f_ext_mid).
        A frozen-geometry positive tangent preconditions the nonlinear solve.
        Both direction and history transport are reevaluated on every trial.
        """
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('Positive finite dt required')
        # An explicit dt*v predictor can rotate tiny-inertia cells arbitrarily
        # far even when the implicit midpoint increment is small. Start from
        # the committed configuration; never subdivide on that predictor.
        y = torch.zeros_like(self.v)
        rhs_scale = float(torch.linalg.vector_norm(dt*self.mass*self.v+.5*dt*dt*external))
        rhs_scale = max(rhs_scale, .5*dt*dt*float(torch.linalg.vector_norm(self.loads()[~self.fixed])), 1e-30)
        cg_total = 0
        iterates, corrections = [], []
        # Anderson acceleration of the nonlinear preconditioned iteration.
        # Scale spins to length units for the small least-squares problem;
        # acceptance still uses the actual mechanical residual below.
        units = torch.ones_like(self.mass)
        units[:, 3:] = self.radii[:, None]
        for iteration in range(max_iterations):
            trial = self.trial(y)
            residual = self.mass*(y-dt*self.v)-.5*dt*dt*(trial['force']+external)
            residual[self.fixed] = 0
            relative = float(torch.linalg.vector_norm(residual))/rhs_scale
            local = self.local_defect(residual) if iteration else (0. if relative == 0 else float('inf'))
            if relative < tolerance and local < 2e-6:
                velocity = 2*y/dt-self.v
                rotation = exp_rotation(y[:, 3:])@self.rotation
                # All physical states remain untouched before this point.
                self.commit(trial)
                self.v, self.rotation = velocity, rotation
                self.time += dt
                return float((external*y).sum()), dict(nonlinear_iterations=iteration,
                    nonlinear_relative_residual=relative, cg_iterations=cg_total,
                    local_pose_defect=local,
                    max_step_rotation=float(torch.linalg.vector_norm(y[:, 3:], dim=1).max()))
            self.n, self.ri, self.rj = trial['n'], trial['ri'], trial['rj']
            self.diagonal = self._diagonal_blocks()
            self.preconditioner_dt = None
            correction, cg = self.solve_increment(-residual, dt)
            cg_total += cg['iterations']
            merit = max(relative/tolerance, self.local_defect(residual)/2e-6)
            iterates.append(y.clone()); corrections.append(correction.clone())
            iterates, corrections = iterates[-6:], corrections[-6:]
            directions = []
            if len(corrections) > 1:
                dc = torch.stack([b-a for a, b in zip(corrections[:-1], corrections[1:])])
                dg = torch.stack([b+db-a-da for a, b, da, db in
                                  zip(iterates[:-1], iterates[1:], corrections[:-1], corrections[1:])])
                weighted = (dc*units).flatten(1)
                gram = weighted@weighted.T
                ridge = 1e-10*torch.trace(gram).clamp_min(1e-40)
                coefficients = torch.linalg.solve(gram+ridge*torch.eye(len(gram), device=self.device),
                                                    weighted@(correction*units).flatten())
                accelerated = correction-torch.einsum('h,hij->ij', coefficients, dg)
                if bool(torch.isfinite(accelerated).all()):
                    directions.append(accelerated)
            directions.append(correction)
            # Backtracking uses the actual nonlinear residual, and never
            # commits trial histories. Failure requests time subdivision.
            accepted = False
            for direction in directions:
                for factor in (1., .5, .25, .125, .0625):
                    candidate = y+factor*direction
                    try:
                        state = self.trial(candidate)
                    except RuntimeError:
                        continue
                    r = self.mass*(candidate-dt*self.v)-.5*dt*dt*(state['force']+external)
                    r[self.fixed] = 0
                    candidate_merit = max(float(torch.linalg.vector_norm(r))/rhs_scale/tolerance,
                                          self.local_defect(r)/2e-6)
                    if candidate_merit < merit:
                        y = candidate; accepted = True; break
                if accepted: break
            if not accepted:
                raise RuntimeError('Subdivide: nonlinear line search failed')
        raise RuntimeError(f'Subdivide: nonlinear residual {relative} did not converge')

    def advance_interval(self, dt, force_at_time, depth=0):
        try:
            work, info = self.step(dt, force_at_time(self.time+.5*dt))
            return work, [dict(dt=dt, subdivision_depth=depth, **info)]
        except RuntimeError as error:
            if depth >= 8 or 'Subdivide:' not in str(error):
                raise
            a, ia = self.advance_interval(.5*dt, force_at_time, depth+1)
            b, ib = self.advance_interval(.5*dt, force_at_time, depth+1)
            return a+b, ia+ib
