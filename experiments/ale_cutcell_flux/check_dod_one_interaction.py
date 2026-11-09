"""Compare a bounded wave predictor against the guarded one-dimensional ALE baseline."""
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np

from check_dod import check_exact, relative, run
from dod_one_interaction import predict_faces
from nonlinear_cutcell_impact import DT, Impact
from nonlinear_cutcell_predictor import case_geometry, samples


def run_pair(width, motion, states):
    baseline = run(width, motion, states, True, variant="godunov")
    candidate = run(width, motion, states, True, variant="godunov", predictor=predict_faces)
    return baseline, candidate


def summarize(result, reference, reference_work, thin_stretch):
    return dict(steps=result["steps"], predictions=result.get("predictions"),
                thin_stretch=result["thin_tau"],
                thin_stretch_error=abs(result["thin_tau"]-thin_stretch),
                ale_flux_error=relative(result["face_flux"], reference),
                pressure_work_error=relative(np.array(result["physical_flux"])[:, 2], reference_work),
                max_balance=result["balance"], max_entropy_defect=result["entropy_inequality_defect"])


def analytic_cases():
    for width in (.005, 1e-8):
        for compression in (.65, .8):
            problem = Impact(width, compression)
            states = (problem.left, problem.middle, problem.right)
            for motion in ("fixed", "drift", "squeeze"):
                starts, speeds = case_geometry(width, motion)
                status, physical, flux = predict_faces(states, width, speeds, DT)
                assert status == "one_shock_collision"
                oracle, moments, _ = map(np.array, zip(
                    *(problem.face_integral(x, w) for x, w in zip(starts, speeds))))
                absolute = max(np.max(abs(DT*flux-oracle)),
                               np.max(abs(DT*physical-moments*[-1, 1, 1])))
                assert absolute < 2e-14
                baseline, candidate = run_pair(width, motion, states)
                results = []
                for result in (baseline, candidate):
                    exact = check_exact(result, problem)
                    results.append(summarize(result, oracle, moments[:, 2], exact["exact_thin_tau"]))
                    results[-1]["max_position_error"] = exact["max_position_error"]
                yield dict(width=width, motion=motion, compression=compression,
                           isolated_flux_absolute_error=float(absolute),
                           baseline=results[0], candidate=results[1])


def sample_cases():
    base = Path(__file__).with_name("dod_godunov_results.json")
    references = {(row["kind"], row["index"]): row for row in json.loads(base.read_text())["fine_fv"]}
    rows = []
    coverage = Counter()
    attempted = Counter()
    for kind, index, states in samples():
        motion = ("fixed", "drift", "squeeze")[index%3]
        for width in (.005, 1e-8):
            _, speeds = case_geometry(width, motion)
            status, _, _ = predict_faces(states, width, speeds, DT)
            coverage[width, kind, status] += 1
        ref = references.get((kind, index))
        if ref is None:
            candidate = run(.005, motion, states, True, variant="godunov", predictor=predict_faces)
        else:
            baseline, candidate = run_pair(.005, motion, states)
            reference = np.array(ref["reference_flux"])
            rows.append(dict(kind=kind, motion=motion, index=index,
                             baseline=summarize(baseline, reference[:, :3], reference[:, 4],
                                                ref["reference_thin_state"][0]),
                             candidate=summarize(candidate, reference[:, :3], reference[:, 4],
                                                 ref["reference_thin_state"][0])))
        attempted.update((kind, status) for status in candidate["predictions"])
    return rows, [{"width": w, "kind": k, "status": s, "count": n} for (w, k, s), n in sorted(coverage.items())], [
        {"kind": k, "status": s, "count": n} for (k, s), n in sorted(attempted.items())]


def timing():
    cases = [(.005, ("fixed", "drift", "squeeze")[i%3], q)
             for _, i, q in samples() if i in (0, 1)]
    for compression in (.65, .8):
        p = Impact(.005, compression)
        cases.extend((.005, motion, (p.left, p.middle, p.right)) for motion in ("fixed", "drift", "squeeze"))
    elapsed = [[], []]
    for trial in range(11):
        for i in (range(2) if trial%2 == 0 else (1, 0)):
            start = time.perf_counter()
            for width, motion, states in cases:
                run(width, motion, states, True, variant="godunov", predictor=(None, predict_faces)[i])
            elapsed[i].append(time.perf_counter()-start)
    return [round(float(np.median(e)*1000), 2) for e in elapsed]


def main():
    analytic = list(analytic_cases())
    mixed, coverage, rollout = sample_cases()
    data = dict(analytic=analytic, fine_reference=mixed, direct_coverage=coverage, rollout_coverage=rollout)
    Path("dod_one_interaction_results.json").write_text(json.dumps(data, indent=2) + "\n")
    print("max analytic local error", max(r["isolated_flux_absolute_error"] for r in analytic))
    for row in analytic+mixed:
        b, c = row["baseline"], row["candidate"]
        print(row.get("width", .005), row.get("compression", row.get("kind")), row["motion"],
              len(b["steps"]), len(c["steps"]), c["predictions"],
              "errors", np.round(100*np.array([b["ale_flux_error"][1:], c["ale_flux_error"][1:]]), 2).tolist(),
              "stretch", round(b["thin_stretch_error"], 4), round(c["thin_stretch_error"], 4))
    print("rollout coverage", rollout)
    print("wall time per 12 short rollouts, baseline / candidate (ms)", timing())


if __name__ == "__main__":
    main()
