"""Compare repeated half-state reconstruction with the mean-only DoD update."""
from collections import Counter
from functools import partial
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from check_dod import check_exact, grid, run
from check_dod_bounded_fans import bounded_faces
from check_dod_one_interaction import summarize
from dod_half_state import ThinHistory
from nonlinear_cutcell_impact import DT, Impact
from nonlinear_cutcell_predictor import fine_reference, samples


def rollout(width, motion, states, use_history, horizon=DT, max_dt=np.inf):
    history = ThinHistory() if use_history else None
    predictor = None if use_history else partial(bounded_faces, fan_limit=1, event_budget=4)
    result = run(width, motion, states, True, horizon=horizon, variant="godunov",
                 predictor=predictor, blend_prediction=True, history=history, max_dt=max_dt)
    if use_history:
        result["history"] = history.steps
    return result


def metrics(result, reference, work, stretch):
    values = summarize(result, reference, work, stretch)
    values["prediction_weights"] = result["prediction_weights"]
    if "history" in result:
        values["history"] = result["history"]
        values["stored_energy_ledger_error"] = result["stored_energy_ledger_error"]
    return values


def impact_cases(max_dt=np.inf):
    rows = []
    for width in (.005, 1e-8):
        for compression in (.65, .8):
            p = Impact(width, compression)
            states = (p.left, p.middle, p.right)
            for motion in ("fixed", "drift", "squeeze"):
                edges, cut, speeds = grid(width, motion)
                exact, moments, _ = map(np.array, zip(
                    *(p.face_integral(x, w) for x, w in zip(edges[cut:cut+2], speeds[cut:cut+2]))))
                row = dict(width=width, compression=compression, motion=motion)
                for name, option in (("mean_only", False), ("half_state", True)):
                    result = rollout(width, motion, states, option, max_dt=max_dt)
                    ref = check_exact(result, p)
                    row[name] = metrics(result, exact, moments[:, 2], ref["exact_thin_tau"])
                    row[name]["max_position_error"] = ref["max_position_error"]
                rows.append(row)
    return rows


def sample_cases(references, horizon, max_dt):
    totals = Counter()
    rows = []
    for kind, index, states in samples():
        motion = ("fixed", "drift", "squeeze")[index%3]
        base = rollout(.005, motion, states, False, horizon, max_dt)
        candidate = rollout(.005, motion, states, True, horizon, max_dt)
        totals["cases"] += 1
        totals["same_schedule"] += base["steps"] == candidate["steps"]
        totals["steps"] += len(candidate["steps"])
        totals["stored_contrast_steps"] += sum(step["used_halves"] for step in candidate["history"])
        totals["nonzero_profile_limited_steps"] += sum(
            bool(step["extra_energy"] > 0. and step["shape_scale"] < 1.) for step in candidate["history"])
        totals.update(candidate["predictions"])
        if (kind, index) in references:
            ref, stretch = references[kind, index]
            rows.append(dict(kind=kind, index=index, motion=motion,
                             mean_only=metrics(base, ref[:, :3], ref[:, 4], stretch),
                             half_state=metrics(candidate, ref[:, :3], ref[:, 4], stretch)))
    return dict(horizon=horizon, max_dt=None if np.isinf(max_dt) else max_dt,
                coverage=dict(totals), reference_cases=rows)


def uniform_checks():
    q = np.array([.75, .27])
    rows = []
    for width in (.005, 1e-8):
        for motion in ("fixed", "drift", "squeeze"):
            result = rollout(width, motion, (q, q, q), True)
            error = float(np.max(abs(np.array(result["state"])-q)))
            energy = float(max(step["extra_energy"] for step in result["history"]))
            assert error < 1e-9 and energy < 1e-20
            rows.append(dict(width=width, motion=motion, uniform_error=error, extra_energy=energy))
    return rows


def timed_batch():
    cases = list(samples())
    seconds = [[], []]
    for iteration in range(11):
        for i in ((0, 1) if iteration%2 == 0 else (1, 0)):
            tick = perf_counter()
            for _, index, states in cases:
                rollout(.005, ("fixed", "drift", "squeeze")[index%3], states, bool(i))
            seconds[i].append(perf_counter()-tick)
    return [round(float(np.median(values))*1000, 2) for values in seconds]


def main():
    source = Path(__file__).with_name("dod_godunov_results.json")
    reference = {(r["kind"], r["index"]): (np.array(r["reference_flux"]), r["reference_thin_state"][0])
                 for r in json.loads(source.read_text())["fine_fv"]}
    key = ("expansion", 1)
    states = next(q for kind, index, q in samples() if (kind, index) == key)
    edges, cut, w = grid(.005, "drift")
    faces = slice(cut, cut+2)
    fine, thin = fine_reference(states, edges[faces], w[faces], 10240, horizon=.04)
    coarser, _ = fine_reference(states, edges[faces], w[faces], 5120, horizon=.04)
    adaptive = sample_cases(reference, DT, np.inf)
    matched = sample_cases(reference, DT, .005)
    extended = sample_cases({key: (fine, thin[0])}, .04, .003)
    assert matched["coverage"]["same_schedule"] == extended["coverage"]["same_schedule"] == 48
    matched_impact = impact_cases(.005)
    assert all(r["mean_only"]["steps"] == r["half_state"]["steps"] for r in matched_impact)
    results = dict(uniform=uniform_checks(), impact=impact_cases(), matched_impact=matched_impact,
                   adaptive=adaptive, matched=matched,
                   extended=extended,
                   extended_reference_refinement=np.max(abs(fine-coarser), axis=0).tolist())
    Path("dod_half_state_results.json").write_text(json.dumps(results, indent=2)+"\n")
    for trial in (adaptive, matched, extended):
        print("horizon", trial["horizon"], "dt cap", trial["max_dt"], trial["coverage"], flush=True)
        for row in trial["reference_cases"]:
            a, b = row["mean_only"], row["half_state"]
            print(row["kind"], row["motion"],
                  "momentum/energy %", np.round(100*np.array([a["ale_flux_error"][1:], b["ale_flux_error"][1:]]), 2),
                  "stretch", a["thin_stretch_error"], b["thin_stretch_error"], flush=True)
    print("wall time per 48 adaptive rollouts (ms), mean / halves:", timed_batch())


if __name__ == "__main__":
    main()
