"""Compare bounded fan tracking against the single-interaction DoD baseline."""
from collections import Counter
from functools import partial
import json
from pathlib import Path
import time

import numpy as np

from check_dod import check_exact, run
from check_dod_one_interaction import summarize
from dod_one_interaction import predict_faces
from nonlinear_cutcell_impact import Impact
from nonlinear_cutcell_predictor import predict, samples


def bounded_faces(states, width, speeds, dt, fan_limit, event_budget):
    status, physical, total = predict_faces(states, width, speeds, dt)
    if total is not None:
        return status, physical, total
    starts = np.array([-width/2, width/2])
    tracked = predict(list(map(tuple, states)), starts, speeds, dt,
                      fan_limit=fan_limit, event_budget=event_budget, faces_only=True)
    if tracked["status"] != "ok":
        return "fan_budget", None, None
    flux = tracked["face_fluxes"]/dt
    return "bounded_fans", flux[:, [5, 3, 4]], flux[:, :3]


def evaluate(predictor, references, blend=False):
    counts = Counter()
    rows = []
    partial_weights = []
    for kind, index, states in samples():
        motion = ("fixed", "drift", "squeeze")[index%3]
        result = run(.005, motion, states, True, variant="godunov", predictor=predictor, blend_prediction=blend)
        weights = result.get("prediction_weights", [])
        partial_weights.extend(w for w in weights if 0 < w < 1)
        counts.update((kind, status) for status in result["predictions"])
        ref = references.get((kind, index))
        if ref:
            f = np.array(ref["reference_flux"])
            metrics = summarize(result, f[:, :3], f[:, 4], ref["reference_thin_state"][0])
            if blend:
                metrics["prediction_weights"] = weights
            rows.append(dict(kind=kind, index=index, motion=motion, **metrics))
    coverage = [{"kind": k, "status": s, "count": n} for (k, s), n in sorted(counts.items())]
    return dict(coverage=coverage, fine_reference=rows, partial_weights=partial_weights)


def impact_comparison(predictor):
    rows = []
    for width in (.005, 1e-8):
        for compression in (.65, .8):
            problem = Impact(width, compression)
            q = (problem.left, problem.middle, problem.right)
            for motion in ("fixed", "drift", "squeeze"):
                case = dict(width=width, compression=compression, motion=motion)
                for name, faces, blend in (("one_interaction", predict_faces, False),
                                           ("bounded_fans", predictor, True)):
                    result = run(width, motion, q, True, variant="godunov",
                                 predictor=faces, blend_prediction=blend)
                    exact = check_exact(result, problem)
                    case[name] = summarize(result, np.array(exact["exact_face_flux"]),
                                           np.array(exact["exact_physical_flux"])[:, 2], exact["exact_thin_tau"])
                    if blend:
                        case[name]["prediction_weights"] = result["prediction_weights"]
                rows.append(case)
    return rows


def timing(predictor):
    states = list(samples())
    elapsed = [[], []]
    for trial in range(11):
        for i in ((0, 1) if trial%2 == 0 else (1, 0)):
            faces = (predict_faces, predictor)[i]
            tick = time.perf_counter()
            for _, index, q in states:
                motion = ("fixed", "drift", "squeeze")[index%3]
                run(.005, motion, q, True, variant="godunov", predictor=faces, blend_prediction=bool(i))
            elapsed[i].append(time.perf_counter()-tick)
    return [round(float(np.median(t)*1000), 2) for t in elapsed]


def main():
    source = Path(__file__).with_name("dod_godunov_results.json")
    references = {(r["kind"], r["index"]): r for r in json.loads(source.read_text())["fine_fv"]}
    configs = [(1, 4), (2, 4), (2, 8), (2, 16), (3, 12)]
    runs = []
    for cap, blend in [(None, False), *((cap, False) for cap in configs), ((1, 4), True), ((2, 8), True)]:
        predictor = predict_faces if cap is None else partial(bounded_faces, fan_limit=cap[0], event_budget=cap[1])
        tick = time.perf_counter()
        result = evaluate(predictor, references, blend)
        elapsed = time.perf_counter()-tick
        print(cap, "blend", blend, "seconds", round(elapsed, 3), result["coverage"], flush=True)
        for row in result["fine_reference"]:
            print(row["kind"], row["motion"], row["predictions"], np.round(100*np.array(row["ale_flux_error"][1:]), 2),
                  "stretch", round(row["thin_stretch_error"], 5), flush=True)
        runs.append(dict(cap=cap, blend=blend, **result))
    bounded = partial(bounded_faces, fan_limit=1, event_budget=4)
    data = dict(sample_comparisons=runs, impact=impact_comparison(bounded))
    Path("dod_bounded_fan_results.json").write_text(json.dumps(data, indent=2) + "\n")
    print("wall time per 48 short rollouts (ms), baseline / limited fans:", timing(bounded))


if __name__ == "__main__":
    main()
