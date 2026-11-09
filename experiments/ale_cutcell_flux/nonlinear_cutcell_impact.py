"""Exact two-shock interaction as a moving-cut finite-volume test.

X is reference mass per unit area, tau = dx/dX.  All quantities use rho0=1.
This is a space-time benchmark with known topology, not a general ALE solver.
"""
import json
from pathlib import Path

import numpy as np

DT = 0.025


def fields(state):
    tau, velocity = state
    pressure = 1 / tau - 1
    energy = velocity**2 / 2 + tau - 1 - np.log(tau)
    return np.array([tau, velocity, energy]), np.array(
        [-velocity, pressure, pressure * velocity]
    )


class Impact:
    def __init__(self, width, compression):
        a = compression
        velocity = (1 - a) / np.sqrt(a)
        self.left = (a, velocity)
        self.right = (a, -velocity)
        self.middle = (1., 0.)
        self.star = (a * a, 0.)
        speed_in = 1 / np.sqrt(a)
        speed_out = a**-1.5
        self.collision = width / (2 * speed_in)
        self.phases = [
            (0., self.collision, width / 2, -speed_in, self.middle),
            (self.collision, DT, 0., speed_out, self.star),
        ]
        for _, _, _, speed, center in self.phases:
            for s, l, r in [(-speed, self.left, center), (speed, center, self.right)]:
                q_l, f_l = fields(l)
                q_r, f_r = fields(r)
                np.testing.assert_allclose(
                    s * (q_r - q_l)[:2], (f_r - f_l)[:2], atol=1e-14
                )
                assert self.entropy_production(s, l, r) <= 1e-14

    @staticmethod
    def entropy_production(speed, left, right):
        q_l, f_l = fields(left)
        q_r, f_r = fields(right)
        return f_r[2] - f_l[2] - speed * (q_r[2] - q_l[2])

    def state(self, x, half, center):
        if x < -half:
            return self.left
        if x > half:
            return self.right
        return center

    def physical_position(self, x, time):
        t0, _, start, speed, center = self.phases[int(time >= self.collision)]
        half = start + speed * (time - t0)
        distance = np.abs(x)
        return np.sign(x) * (center[0] * np.minimum(distance, half)
                             + self.left[0] * np.maximum(distance - half, 0.))

    def cell_integrals(self, edges, time):
        t0, _, start, speed, center = self.phases[int(time >= self.collision)]
        half = start + speed * (time - t0)
        left, right = edges[:-1], edges[1:]
        lengths = np.array([
            np.maximum(0., np.minimum(right, -half) - left),
            np.maximum(0., np.minimum(right, half) - np.maximum(left, -half)),
            np.maximum(0., right - np.maximum(left, half)),
        ])
        states = np.array([fields(q)[0] for q in (self.left, center, self.right)])
        return lengths.T @ states

    def face_integral(self, start_face, w, begin_time=0., end_time=DT):
        integral = np.zeros(3)
        moments = np.zeros(3)
        eulerian = np.zeros(3)
        for t0, t1, start, speed, center in self.phases:
            lo, hi = max(t0, begin_time), min(t1, end_time)
            if hi <= lo:
                continue
            cuts = [lo, hi]
            for sign in (-1, 1):
                relative_speed = w - sign * speed
                if relative_speed != 0:
                    t = (sign * (start - speed * t0) - start_face) / relative_speed
                    if lo < t < hi:
                        cuts.append(t)
            cuts.sort()
            for begin, end in zip(cuts[:-1], cuts[1:]):
                mid = (begin + end) / 2
                state = self.state(start_face + w * mid, start + speed * (mid - t0), center)
                q, f = fields(state)
                dt = end - begin
                integral += dt * (f - w * q)
                moments += dt * np.array([-f[0], f[1], f[2]])
                # Recompute with the physical sweep, without using the
                # reference-coordinate mesh speed in the flux formula.
                sweep = (self.physical_position(start_face + w * end, end)
                         - self.physical_position(start_face + w * begin, begin))
                dm = (dt * q[1] - sweep) / q[0]
                eulerian += (dm, dm * q[1] + dt * f[1], dm * q[2] + dt * f[2])
        return integral, moments, eulerian

    def shock_loss(self, edges, speeds):
        loss = np.zeros(len(speeds) - 1)
        for t0, t1, start, speed, center in self.phases:
            for sign, left, right in [(-1, self.left, center), (1, center, self.right)]:
                s = sign * speed
                intercept = sign * (start - speed * t0)
                cuts = [t0, t1]
                for x, w in zip(edges, speeds):
                    if w != s:
                        t = (intercept - x) / (w - s)
                        if t0 < t < t1:
                            cuts.append(t)
                cuts.sort()
                rate = -self.entropy_production(s, left, right)
                for begin, end in zip(cuts[:-1], cuts[1:]):
                    mid = (begin + end) / 2
                    i = np.searchsorted(edges + speeds * mid, intercept + s * mid) - 1
                    if 0 <= i < len(loss):
                        loss[i] += rate * (end - begin)
        return loss


def run(width, motion, compression):
    problem = Impact(width, compression)
    regular = np.linspace(-.5, .5, 21)
    edges = np.sort(np.r_[regular[regular != 0], [-width/2, width/2]])
    cut = np.flatnonzero(edges == -width / 2)[0]
    speeds = np.zeros_like(edges)
    speeds[cut:cut+2] = {"fixed": (0, 0), "drift": (.85, .85),
                          "squeeze": (10*width, -10*width)}[motion]
    end = edges + DT * speeds
    sizes = np.diff(end)
    assert min(sizes) > 0
    old = problem.cell_integrals(edges, 0.)
    truth = problem.cell_integrals(end, DT)
    face_data = [problem.face_integral(x, w) for x, w in zip(edges, speeds)]
    flux, moments, eulerian = map(np.array, zip(*face_data))
    loss = problem.shock_loss(edges, speeds)
    update = old - np.diff(flux, axis=0)
    update[:, 2] -= loss
    errors = np.max(abs(update - truth), axis=0)
    # Geometry computed from the physical position of each material face.
    # x(0,t) = 0 by symmetry. No reference density assumption in this check.
    sweep = problem.physical_position(end, DT) - problem.physical_position(edges, 0.)
    sweep_error = np.max(abs(sweep + flux[:, 0]))
    expected_flux = np.column_stack((-DT * speeds, flux[:, 1:]))
    eulerian_error = np.max(abs(eulerian - expected_flux), axis=0)

    # Independent initial face problems, extended through the whole step.
    _, _, start, s, center = problem.phases[0]
    isolated_flux = []
    for i, (x, w) in enumerate(zip(edges, speeds)):
        if i == cut:
            state = problem.left if w < -s else center
        elif i == cut + 1:
            state = center if w < s else problem.right
        else:
            state = problem.state(x, start, center)
        q, f = fields(state)
        isolated_flux.append(DT * (f - w * q))
    isolated = old - np.diff(isolated_flux, axis=0)
    # Mechanical energy loss from averaging at the end is distinct from shock loss.
    final_means = truth[:, :2] / sizes[:, None]
    resolved_energy = sizes * np.array([fields(state)[0][2] for state in final_means])
    projection = truth[:, 2] - resolved_energy
    work_mismatch = np.max(abs(moments[:, 2] - moments[:, 0] * moments[:, 1] / DT))
    assert np.min(projection) > -1e-14
    assert max(errors) < 2e-14 and sweep_error < 2e-14
    assert max(eulerian_error) < 2e-14
    assert min(final_means[:, 0]) > 0.
    assert isolated[cut, 0] < 0.
    return dict(width=width, compression=compression, motion=motion,
                collision_time=problem.collision, min_stretch=float(min(final_means[:, 0])),
                thin_final_stretch=float(final_means[cut, 0]),
                thin_isolated_stretch=float(isolated[cut, 0] / sizes[cut]),
                length_balance=float(errors[0]), momentum_balance=float(errors[1]),
                mechanical_energy_balance=float(errors[2]), physical_sweep_error=float(sweep_error),
                eulerian_flux_error=eulerian_error.tolist(),
                thin_face_pressure_impulses=moments[cut:cut+2, 1].tolist(),
                thin_face_pressure_work=moments[cut:cut+2, 2].tolist(),
                shock_loss=float(loss.sum()), projection_loss=float(projection.sum()),
                max_pressure_work_covariance=float(work_mismatch))


def constant_state_check():
    edges = np.linspace(-.5, .5, 8)
    w = np.array([0., .1, -.3, .7, .4, -.8, .3, 0.])
    q, f = fields((.75, .27))
    old = np.diff(edges)[:, None] * q
    new = np.diff(edges + DT*w)[:, None] * q
    flux = DT * (f - w[:, None]*q)
    return float(np.max(abs(new - (old - np.diff(flux, axis=0)))))


if __name__ == "__main__":
    rows = [run(width, motion, compression)
            for width in (5e-3, 1e-5, 1e-8)
            for motion in ("fixed", "drift", "squeeze")
            for compression in (.65, .8)]
    uniform_error = constant_state_check()
    assert uniform_error < 1e-14
    result = {"dt": DT, "uniform_error": uniform_error, "cases": rows}
    Path("nonlinear_cutcell_impact.json").write_text(json.dumps(result, indent=2)+"\n")
