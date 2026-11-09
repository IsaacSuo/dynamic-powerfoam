"""Continuous fan correction for the three-cell nonlinear ALE predictor.

All three terms in the correction are space-time fluxes. The matched state
uses the same pair of conservative fluxes, evaluated as a spatial integral
difference to avoid cancellation in the thinnest cell. This is a single-step
experiment; the nonlinear mechanical energy inequality is checked separately.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad_vec
from scipy.optimize import brentq

from nonlinear_cutcell_impact import DT, Impact, fields
from nonlinear_cutcell_predictor import case_geometry, fluxes, intervals, predict, riemann, samples


NODES, WEIGHTS = np.polynomial.legendre.leggauss(12)
LOG_NODES, LOG_WEIGHTS = np.polynomial.legendre.leggauss(24)


def exact_star(left, right):
    la, lb = math.log(left[0]), math.log(right[0])
    ul, ur = left[1], right[1]

    def phi(d):
        return d if d >= 0 else 2 * math.sinh(d / 2)

    extent = abs(ul - ur) + 1
    ls = brentq(lambda x: phi(x - la) + phi(x - lb) + ul - ur,
                min(la, lb) - extent, max(la, lb) + extent, xtol=5e-15)
    return math.exp(ls), ul + phi(ls - la)


class RiemannRays:
    def __init__(self, left, right, fan_step=None):
        self.left = left
        self.rays = []
        self.jumps = []
        if fan_step is not None:
            for speed, a, b in riemann(left, right, fan_step):
                self.add_jump(speed, a, b)
            return

        star = exact_star(left, right)
        for sign, outer, a, b in [(-1, left, left, star), (1, right, star, right)]:
            if abs(math.log(outer[0] / star[0])) < 2e-13:
                continue
            if star[0] < outer[0]:
                self.add_jump(sign / math.sqrt(a[0] * b[0]), a, b)
                continue
            lo, hi = sorted((sign / outer[0], sign / star[0]))
            self.rays.append((lo, sign, outer))
            self.rays.append((hi, 0, b))

    def add_jump(self, speed, a, b):
        self.rays.append((speed, 0, b))
        self.jumps.append((speed, Impact.entropy_production(speed, a, b)))

    def state(self, ray):
        q = self.left
        for speed, sign, outer in self.rays:
            if ray < speed:
                break
            if sign:
                tau = 1 / abs(ray)
                q = (tau, outer[1] - sign * math.log(tau / outer[0]))
            else:
                q = outer
        return q

    def mean(self, start, end, time):
        value = np.zeros(3)
        for a, b in intervals(start, end, (t * time for t, _, _ in self.rays)):
            x = (a + b) / 2 + (b - a) / 2 * NODES
            q = np.array([fields(self.state(p / time))[0] for p in x])
            value += (b - a) / 2 * (WEIGHTS @ q)
        return value / (end - start)

    def face_flux(self, start, speed, time, adaptive=False):
        integral = np.zeros(5)
        count = 0
        hits = (start / (s - speed) for s, _, _ in self.rays if s != speed)
        for a, b in intervals(0., time, hits):
            ray = speed + 2 * start / (a + b)
            state = self.state(ray)
            inside_fan = False
            for edge, sign, _ in self.rays:
                if edge > ray:
                    break
                inside_fan = sign != 0
            if inside_fan and start != 0 and adaptive:
                val, err, info = quad_vec(
                    lambda t: fluxes(self.state(speed + start / t), speed),
                    a, b, epsabs=2e-13, epsrel=1e-11, full_output=True,
                )
                assert info.success and err < 2e-11
                integral += val
                count += info.neval
            elif inside_fan and start != 0:
                # The near-origin scale shrinks with cell width. Log-time
                # quadrature resolves it without subdividing into small steps.
                scale = .5*math.log(b/a)
                times = b*np.exp(scale*(LOG_NODES-1))
                terminal = fluxes(self.state(speed+start/b),speed)
                correction = np.array([fluxes(self.state(speed+start/t),speed)-terminal for t in times])
                integral += (b-a)*terminal+scale*((LOG_WEIGHTS*times)@correction)
                count += len(times)+1
            else:
                integral += (b - a) * fluxes(state, speed)
                count += 1
        return integral, count

    def thin_entropy(self, starts, speeds, time):
        prod = 0.
        for speed, rate in self.jumps:
            hits = (a / (speed - w) for a, w in zip(starts, speeds) if speed != w)
            for a, b in intervals(0., time, hits):
                mid = (a + b) / 2
                left, right = starts + mid * speeds
                if left < speed * mid < right:
                    prod += (b - a) * rate
        return prod


def matched(states, starts, speeds, time=DT):
    base = predict(states, starts, speeds, time, fan_step=.2)
    if base["status"] != "ok":
        return {"status": "base_budget"}
    exact = RiemannRays(states[0], states[2])
    discrete = RiemannRays(states[0], states[2], fan_step=.2)
    end = starts + speeds * time
    old_mass, new_mass = np.diff(starts)[0], np.diff(end)[0]
    initial = old_mass * fields(states[1])[0]
    flux0 = np.array(base["face_fluxes"])
    flux_e, count_e = zip(*(exact.face_flux(x,w,time) for x,w in zip(starts,speeds)))
    flux_check, count_check = zip(*(exact.face_flux(x,w,time,adaptive=True) for x,w in zip(starts,speeds)))
    quadrature_error = float(np.max(abs(np.array(flux_e)-flux_check)))
    flux_d, count_d = zip(*(discrete.face_flux(x,w,time) for x,w in zip(starts,speeds)))
    flux = flux0 + (np.array(flux_e) - flux_d)
    base_mean = np.array([base["stretch"], base["velocity"]])
    outer_mean = exact.mean(*end, time)
    discrete_mean = discrete.mean(*end, time)
    fixed_mean = base_mean + (outer_mean - discrete_mean)[:2]
    entropy0 = base["expansion_entropy_gain"] - base["shock_entropy_loss"]
    entropy_e = exact.thin_entropy(starts, speeds, time)
    entropy_d = discrete.thin_entropy(starts, speeds, time)
    entropy = entropy0 + entropy_e - entropy_d
    integrated = initial - np.diff(flux, axis=0)[0,:3]
    integrated[2] += entropy
    expected_e = ((initial[2] - np.diff(flux0, axis=0)[0,2] + entropy0) / new_mass
                  + outer_mean[2] - discrete_mean[2])
    balance = integrated / new_mass - np.r_[fixed_mean, expected_e]
    energy_margin = expected_e - fields(fixed_mean)[0][2] if fixed_mean[0] > 0 else None
    assert max(abs(balance)) < 5e-8
    return {"status":"ok", "base_events":base["events"],
            "continuous_evaluations":sum(count_e), "adaptive_evaluations":sum(count_check),
            "quadrature_error":quadrature_error, "discrete_evaluations":sum(count_d),
            "stretch":float(fixed_mean[0]), "velocity":float(fixed_mean[1]),
            "face_fluxes":flux.tolist(), "base_stretch":base["stretch"],
            "base_velocity":base["velocity"], "base_fluxes":base["face_fluxes"],
            "outer_stretch":float(outer_mean[0]), "outer_fluxes":np.array(flux_e).tolist(),
            "outer_mean_difference":(outer_mean-discrete_mean).tolist(),
            "thin_entropy":float(entropy), "energy_margin":energy_margin,
            "balance_error":float(max(abs(balance)))}


def error_against(row, flux, stretch):
    f = np.array(flux)
    scales = np.maximum(np.max(abs(f),axis=0), 1e-10)
    def error(value, mean):
        err = np.max(abs(np.array(value)-f),axis=0)
        return {"flux":err.tolist(), "relative_flux":(err/scales).tolist(),
                "stretch":abs(mean-stretch)}
    return {"coarse":error(row["base_fluxes"],row["base_stretch"]),
            "matched":error(row["face_fluxes"],row["stretch"]),
            "outer":error(row["outer_fluxes"],row["outer_stretch"])}


def main():
    data = json.loads(Path("nonlinear_cutcell_predictor.json").read_text())
    oracle_checks = []
    for ref in data["analytic"]:
        width, compression = ref["width"], ref["compression"]
        oracle = Impact(width,compression)
        states = (oracle.left,oracle.middle,oracle.right)
        starts,speeds = case_geometry(width,ref["motion"])
        row = matched(states,starts,speeds)
        err = error_against(row,ref["face_fluxes"],ref["stretch"])["matched"]
        assert err["stretch"] < 1e-12 and max(err["flux"]) < 1e-13
        oracle_checks.append(err)

    rows = []
    ref_by_case = {(r["kind"],r["index"]):r for r in data["reference"] if r["fan_step"] == .025}
    for kind, i, states in samples():
        for width in (.005,1e-5,1e-8):
            motion = ("fixed","drift","squeeze")[i%3]
            starts,speeds = case_geometry(width,motion)
            row = matched(states,starts,speeds)
            assert row["status"] == "ok"
            row.update(kind=kind,index=i,width=width,motion=motion)
            if width == .005 and (kind,i) in ref_by_case:
                ref = ref_by_case[kind,i]
                row["fine_fv_error"] = error_against(row,ref["reference_flux"],ref["reference_stretch"])
                row["fine_fv_resolution_difference"] = ref["flux_refinement_difference"]
            if width == 1e-8:
                ref = predict(states,starts,speeds,DT,fan_step=.025,event_budget=256)
                row["fine_wave_status"] = ref["status"]
                if ref["status"] == "ok":
                    row["fine_wave_difference"] = error_against(row,ref["face_fluxes"],ref["stretch"])
            rows.append(row)

    Path("nonlinear_cutcell_continuous.json").write_text(
        json.dumps({"oracle":oracle_checks,"cases":rows},indent=2)+"\n")
    print("analytic",len(oracle_checks),"matched",len(rows))


if __name__ == "__main__":
    main()
