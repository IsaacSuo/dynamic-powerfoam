"""Predict two moving ALE face fluxes through at most one wave interaction."""
from dataclasses import dataclass
import math

import numpy as np

from nonlinear_cutcell_continuous import LOG_NODES, LOG_WEIGHTS, RiemannRays
from nonlinear_cutcell_predictor import fluxes, intervals


def face_values(state, speed):
    return np.r_[fluxes(state, speed), -state[1]]


@dataclass(frozen=True)
class Wave:
    origin: float
    birth: float
    low: float
    high: float
    right: tuple
    family: int = 0
    outer: tuple = ()

    def state(self, ray):
        if not self.family:
            return self.right
        tau = 1/abs(ray)
        return tau, self.outer[1]-self.family*math.log(tau/self.outer[0])


def waves(left, right, origin, birth):
    if left == right:
        return []
    rays = iter(RiemannRays(left, right).rays)
    result = []
    for low, family, state in rays:
        if family:
            high, _, terminal = next(rays)
            result.append(Wave(origin, birth, low, high, terminal, family, state))
        else:
            result.append(Wave(origin, birth, low, low, state))
    return result


def next_interaction(packets, begin, end):
    first = end
    index = None
    for i, (left, right) in enumerate(zip(packets[:-1], packets[1:])):
        gap = right.origin + right.low*(begin-right.birth) - left.origin - left.high*(begin-left.birth)
        if gap < -1e-13:
            return begin, i
        approach = left.high-right.low
        if approach > 0.:
            hit = begin+max(0., gap)/approach
            if hit < first:
                first, index = hit, i
    return first, index


def sample(left, packets, x, t):
    state = left
    for wave in packets:
        ray = (x-wave.origin)/(t-wave.birth)
        if ray < wave.low:
            break
        if ray < wave.high:
            return wave.state(ray), wave
        state = wave.right
    return state, None


def integrate(left, packets, starts, speeds, begin, end):
    result = np.zeros((2, 6))
    for i, (x, w) in enumerate(zip(starts, speeds)):
        crossings = ((wave.origin-x-s*wave.birth)/(w-s)
                     for wave in packets for s in (wave.low, wave.high) if w != s)
        for a, b in intervals(begin, end, sorted(set(crossings))):
            mid = .5*(a+b)
            state, fan = sample(left, packets, x+w*mid, mid)
            if fan is None or abs(x+w*fan.birth-fan.origin) < 1e-15:
                result[i] += (b-a)*face_values(state, w)
                continue
            # Smooth inside the fan. A logarithmic time variable resolves the
            # transition even when the age at the first crossing is tiny.
            lo, hi = math.log(a-fan.birth), math.log(b-fan.birth)
            ages = np.exp(.5*((hi-lo)*LOG_NODES+hi+lo))
            ray = w+(x+w*fan.birth-fan.origin)/ages
            values = np.array([face_values(fan.state(r), w) for r in ray])
            result[i] += .5*(hi-lo)*((LOG_WEIGHTS*ages)@values)
    return result


def predict_faces(states, width, speeds, dt):
    """Return time-averaged physical and total fluxes, or an unresolved status."""
    left, middle, right = map(tuple, states)
    starts = np.array([-width/2, width/2])
    packets = waves(left, middle, starts[0], 0.) + waves(middle, right, starts[1], 0.)
    hit, index = next_interaction(packets, 0., dt)
    if index is None:
        integral = integrate(left, packets, starts, speeds, 0., dt)
        status = "independent"
    else:
        incoming_l, incoming_r = packets[index:index+2]
        if incoming_l.family or incoming_r.family:
            return "interacting_fans", None, None
        before = left if index == 0 else packets[index-1].right
        origin = incoming_l.origin+incoming_l.high*hit
        outgoing = waves(before, incoming_r.right, origin, hit)
        after = packets[:index]+outgoing+packets[index+2:]
        if next_interaction(after, hit, dt)[1] is not None:
            return "second_interaction", None, None
        integral = integrate(left, packets, starts, speeds, 0., hit)
        integral += integrate(left, after, starts, speeds, hit, dt)
        status = "one_shock_collision"
    total = integral[:, :3]/dt
    physical = integral[:, [5, 3, 4]]/dt
    return status, physical, total
