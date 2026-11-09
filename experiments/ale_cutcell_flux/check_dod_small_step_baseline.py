"""Compare the repeated-rezone DoD rollout with ordinary CFL subcycling."""
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from check_dod import channels, entropy, godunov_flux, grid, initial_state, update
from check_dod_repeated_rezone import geometry, run as run_dod
from dod_remap import remap
from nonlinear_cutcell_impact import Impact
from nonlinear_cutcell_predictor import samples


def ordinary_cap(data, width, w):
    # At beta=1 the bypass vanishes. The sum of physical wave speeds and
    # outward geometric transport is the original per-cell CFL condition.
    speed = data[4][:-1]
    outward = np.maximum(w[:-1], 0.)+np.maximum(-w[1:], 0.)
    return float(np.min(width/(speed[:-1]+speed[1:]+outward)))


def run_small(stages, states):
    _, edges, cut, *_ = stages[0]
    mean = initial_state(np.diff(edges), cut, states)
    flux = np.zeros((2, 4))
    steps_per_rezone = []
    min_dt, max_dt = np.inf, 0.
    max_balance, max_energy_gain = 0., 0.
    for duration, edges, cut, w, advanced, new_edges, new_cut, _ in stages:
        h = np.diff(edges)
        before_q, before_energy = h@mean, float(h@entropy(mean))
        boundary = np.zeros(3)
        time, steps = 0., 0
        while time < duration-1e-15:
            data = channels(mean, (states[0], states[-1]), w, cut, directional=True)
            dt = min(duration-time, .99*ordinary_cap(data, h, w))
            mean, h, mechanical, total, _, _ = update(mean, h, w, data, cut, dt, 1., godunov_flux)
            faces = slice(cut, cut+2)
            flux += np.column_stack((total[faces, 1], mechanical[faces, 2],
                                     total[faces, 2]-mechanical[faces, 2], total[faces, 2]))
            boundary += total[-1]-total[0]
            time += dt
            steps += 1
            min_dt, max_dt = min(min_dt, dt), max(max_dt, dt)
            assert steps < 10000
        steps_per_rezone.append(steps)
        np.testing.assert_allclose(h, np.diff(advanced), atol=2e-15, rtol=0)
        new_width = np.diff(new_edges)
        mean, _ = remap(advanced, mean, np.zeros_like(mean), new_edges, np.zeros(len(new_width), dtype=bool))
        max_balance = max(max_balance, float(max(abs(new_width@mean-before_q+boundary[:2]))))
        max_energy_gain = max(max_energy_gain, float(new_width@entropy(mean)-before_energy+boundary[2]))
        assert max_balance < 2e-14 and max_energy_gain < 2e-13
    return dict(exchange=flux.tolist(), thin_state=mean[new_cut].tolist(), steps=sum(steps_per_rezone),
                steps_per_rezone=steps_per_rezone, min_dt=float(min_dt), max_dt=max_dt,
                max_conserved_residual=max_balance, max_energy_gain=max_energy_gain)


def attach_errors(result, reference):
    error = np.asarray(result["exchange"])-np.asarray(reference["reference_exchange"])
    return dict(result, max_abs_exchange_error=np.max(abs(error), axis=0).tolist(),
                thin_stretch_error=abs(result["thin_state"][0]-reference["reference_thin_state"][0]))


def benchmark(samples_to_run):
    durations = [[], []]
    for trial in range(9):
        for i in ((0, 1) if trial%2 == 0 else (1, 0)):
            tick = perf_counter()
            for _, index, states in samples_to_run:
                stages = geometry(.005, ("fixed", "drift", "squeeze")[index%3])
                if i:
                    run_dod(stages, states, True)
                else:
                    run_small(stages, states)
            durations[i].append(perf_counter()-tick)
    return [round(float(np.median(times))*1000, 2) for times in durations]


def first_step_caps():
    rows = []
    for compression in (.65, .8):
        problem = Impact(1e-8, compression)
        states = (problem.left, problem.middle, problem.right)
        for motion in ("fixed", "drift", "squeeze"):
            edges, cut, w = grid(1e-8, motion)
            h = np.diff(edges)
            q = initial_state(h, cut, states)
            data = channels(q, (states[0], states[-1]), w, cut, directional=True)
            rows.append(dict(compression=compression, motion=motion, cap=ordinary_cap(data, h, w)))
    return rows


def main():
    tests = list(samples())
    saved = Path(__file__).with_name("dod_repeated_rezone_results.json")
    reference = json.loads(saved.read_text())
    sampled = []
    for kind, index, q in tests:
        motion = ("fixed", "drift", "squeeze")[index%3]
        result = run_small(geometry(.005, motion), q)
        sampled.append(dict(kind=kind, index=index, motion=motion, **result))
    expansion = next(row for row in sampled if row["kind"] == "expansion" and row["index"] == 1)
    expansion = {key: attach_errors(val, reference["expansion"]) for key, val in (
        ("small_step", expansion), ("dod", reference["expansion"]["reuse"]))}
    shocks = []
    for ref in reference["shocks"]:
        if ref["width"] != .005:
            continue
        p = Impact(.005, ref["compression"])
        result = run_small(geometry(.005, ref["motion"]), (p.left, p.middle, p.right))
        shocks.append(dict(compression=ref["compression"], motion=ref["motion"],
                           small_step=attach_errors(result, ref), dod=attach_errors(ref["reuse"], ref)))
    data = dict(columns=reference["columns"], sample=sampled, expansion=expansion, shocks=shocks,
                extreme_thin_first_step_caps=first_step_caps())
    Path("dod_small_step_baseline_results.json").write_text(json.dumps(data, indent=2)+"\n")
    print("ordinary steps total / range", sum(r["steps"] for r in sampled),
          min(r["steps"] for r in sampled), max(r["steps"] for r in sampled), flush=True)
    for row in [expansion, *shocks]:
        print(row.get("compression", "expansion"), row.get("motion", "drift"), row["small_step"]["steps"],
              "max abs work / net E errors",
              [list(np.asarray(row[k]["max_abs_exchange_error"])[[1, 3]]) for k in ("small_step", "dod")],
              "stretch errors", [row[k]["thin_stretch_error"] for k in ("small_step", "dod")], flush=True)
    print("wall time per 48 rollouts in ms (small step, DoD)", benchmark(tests))


if __name__ == "__main__":
    main()
