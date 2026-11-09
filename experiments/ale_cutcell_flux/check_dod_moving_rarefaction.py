"""Audit moving rarefaction face fluxes against a refined independent solve."""
from functools import partial
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from check_dod import channels, cfl_lines, godunov_flux, grid, initial_state, relative, run, update
from check_dod_bounded_fans import bounded_faces
from dod_one_interaction import predict_faces
from nonlinear_cutcell_predictor import DT, fine_reference, predict, samples


def ledger(physical, total):
    return np.column_stack((total[:, 1], physical[:, 2], total[:, 2]-physical[:, 2], total[:, 2]))


def reference_at(states, faces, speeds, t, resolution, initial_breaks=None):
    flux, mean = fine_reference(states, faces, speeds, resolution, horizon=t, initial_breaks=initial_breaks)
    return np.column_stack((flux[:, 1], flux[:, 4], flux[:, 2]-flux[:, 4], flux[:, 2])), mean


def step_history(states, schedule, betas, predictor, limited, motion):
    edges, cut, w = grid(.005, motion)
    h = np.diff(edges)
    q = initial_state(h, cut, states)
    time, rows, status = 0., [], []
    weights = [] if limited else None
    for dt, beta in zip(schedule, betas):
        data = channels(q, (states[0], states[2]), w, cut, directional=True)
        a, b = cfl_lines(data, h)
        cfl = max(dt*(a+beta*b))
        if cfl > 1.+2e-14:
            return rows, float(cfl)
        q, h, physical, total, *_ = update(q, h, w, data, cut, dt, beta, godunov_flux,
                                          predictor, status, weights)
        time += dt
        faces = slice(cut, cut+2)
        rows.append(dict(time=time, dt=dt, beta=beta, cfl=float(cfl),
                         prediction=status[-1], weight=None if weights is None else weights[-1],
                         exchange=ledger(physical[faces], total[faces]).tolist(),
                         patch_state=q[cut-1:cut+2].tolist()))
    return rows, None


def tracked_flux(q, width, speeds, dt, segments):
    raw = predict(list(map(tuple, q)), np.array([-width/2, width/2]), speeds, dt,
                  fan_limit=segments, event_budget=256, faces_only=True)
    if raw["status"] != "ok":
        return raw["events"], None, None
    flux = raw["face_fluxes"]/dt
    return raw["events"], flux[:, [5, 3, 4]], flux[:, :3]


def projection_test(states, edges, w, cut, first, second, ref):
    t0 = first["time"]
    faces = slice(cut, cut+2)
    patch = slice(cut-1, cut+3)
    moved = edges+t0*w
    patch_edges = moved[patch]
    subdivisions = 16
    refined_edges = np.r_[np.concatenate([np.linspace(l, r, subdivisions+1)[:-1]
                                         for l, r in zip(patch_edges[:-1], patch_edges[1:])]),
                          patch_edges[-1]]
    _, _, refined = fine_reference(states, edges[faces], w[faces], 10240, horizon=t0,
                                   average_edges=refined_edges)
    exact_average = refined.reshape(3, subdivisions, 2).mean(axis=1)
    rows = dict(reference=ref.tolist(), exact_averages=exact_average.tolist(),
                coarse_averages=first["patch_state"], comparisons=[], restart_resolution=[])

    def restart(partitions):
        breaks = []
        averages = []
        for cell, bins in enumerate(partitions):
            offset = cell*subdivisions
            stride = subdivisions//bins
            for start in range(offset, offset+subdivisions, stride):
                breaks.append(refined_edges[start])
                averages.append(np.mean(refined[start:start+stride], axis=0))
        restarted, _ = reference_at(averages, moved[faces], w[faces], second["dt"], 10240,
                                    initial_breaks=breaks[1:])
        return dict(bins=list(partitions), exchange=restarted.tolist(), signed_error=(restarted-ref).tolist())

    rows["restart_resolution"] = [restart((bins,)*3) for bins in (1, 2, 4, 8, 16)]
    rows["selective_restart"] = [restart(partitions) for partitions in ((2, 1, 1), (1, 2, 1), (1, 1, 2))]
    # Can the cheap first-step wave prediction supply these two averages?
    # Replay at fine resolution to isolate the stored profile from the
    # choice of second-step flux approximation.
    left, thin, right = np.asarray(first["patch_state"])
    width = edges[cut+1]-edges[cut]
    local_starts = np.array([-width/2, width/2])
    halves = predict(states, local_starts, w[faces], first["dt"], fan_limit=1, event_budget=4,
                     faces_only=True, sample_edges=np.linspace(*(local_starts+first["dt"]*w[faces]), 3))
    assert halves["status"] == "ok"
    half_states = halves["means"]
    np.testing.assert_allclose(half_states.mean(axis=0), thin, atol=2e-13, rtol=0)
    breaks = np.linspace(*moved[faces], 3)
    restarted, _ = reference_at([left, *half_states, right], moved[faces], w[faces], second["dt"],
                                10240, initial_breaks=breaks)
    rows["predicted_thin_halves"] = dict(half_states=half_states.tolist(), exchange=restarted.tolist(),
                                        signed_error=(restarted-ref).tolist())

    events = []

    def replay_predictor(q, width, speeds, dt):
        if not events:
            events.append(None)
            return bounded_faces(q, width, speeds, dt, fan_limit=1, event_budget=4)
        starts = np.array([-width/2, width/2])
        raw = predict([tuple(q[0]), *map(tuple, half_states), tuple(q[2])], starts, speeds, dt,
                      fan_limit=1, event_budget=4, faces_only=True,
                      initial_breaks=np.linspace(*starts, 3))
        events.append(raw["events"])
        if raw["status"] != "ok":
            return "fan_budget", None, None
        flux = raw["face_fluxes"]/dt
        return "predicted_thin_halves", flux[:, [5, 3, 4]], flux[:, :3]

    steps, failure = step_history(states, [first["dt"], second["dt"]], [first["beta"], second["beta"]],
                                  replay_predictor, True, "drift")
    assert failure is None
    np.testing.assert_array_equal(steps[0]["patch_state"], first["patch_state"])
    replay = steps[1]
    replay["events"] = events[1]
    replay["signed_error"] = (np.asarray(replay["exchange"])-ref).tolist()
    rows["two_half_bounded_replay"] = replay
    width = moved[cut+1]-moved[cut]
    for name, initial in (("coarse_averages", first["patch_state"]), ("exact_averages", exact_average)):
        for segments in (1, 2, 4, 8, 16):
            events, physical, total = tracked_flux(initial, width, w[faces], second["dt"], segments)
            exchange = None if total is None else ledger(physical, total)*second["dt"]
            rows["comparisons"].append(dict(initial=name, segments=segments, events=events,
                                            exchange=None if exchange is None else exchange.tolist()))
    return rows


def main():
    states = next(q for kind, i, q in samples() if kind == "expansion" and i == 1)
    motion = "drift"
    edges, cut, w = grid(.005, motion)
    faces, speeds = edges[cut:cut+2], w[cut:cut+2]
    coarse = partial(bounded_faces, fan_limit=1, event_budget=4)
    histories = {}
    for name, predictor, limited in (("one_interaction", predict_faces, False), ("one_front", coarse, True)):
        rollout = run(.005, motion, states, True, variant="godunov", predictor=predictor, blend_prediction=limited)
        steps, failure = step_history(states, rollout["steps"], rollout["beta"], predictor, limited, motion)
        assert failure is None
        total = np.sum([row["exchange"] for row in steps], axis=0)
        np.testing.assert_allclose(total, ledger(np.array(rollout["physical_flux"]), np.array(rollout["face_flux"])),
                                   atol=2e-14, rtol=0)
        histories[name] = steps
    times = sorted({row["time"] for steps in histories.values() for row in steps} | {DT})
    reference = {}
    refinement = []
    for t in times:
        low, _ = reference_at(states, faces, speeds, t, 5120)
        high, _ = reference_at(states, faces, speeds, t, 10240)
        reference[t] = high
        refinement.append(dict(time=t, max_difference=np.max(abs(high-low), axis=0).tolist()))
        print("ref", t, "ledger", high, "delta", np.max(abs(high-low), axis=0), flush=True)
    for name, steps in histories.items():
        prev = np.zeros((2, 4))
        for step in steps:
            ref = reference[step["time"]]-prev
            prev = reference[step["time"]]
            step["reference_exchange"] = ref.tolist()
            step["signed_error"] = (np.asarray(step["exchange"])-ref).tolist()
            print(name, step["time"], "error", np.round(1e6*np.array(step["signed_error"]), 2), flush=True)

    base = histories["one_front"][0]
    first = []
    for segments in (1, 2, 3, 4, 6, 8):
        calls = []

        def predictor(q, width, speeds, dt):
            events, physical, total = tracked_flux(q, width, speeds, dt, segments)
            calls.append(events)
            return ("fan_budget" if total is None else "bounded_fans"), physical, total

        tick = perf_counter()
        steps, failure = step_history(states, [base["dt"]], [base["beta"]], predictor, True, motion)
        assert failure is None
        row = steps[0]
        row.update(segments=segments, interactions=calls[0],
                   relative_error=relative(row["exchange"], reference[row["time"]]))
        first.append(row)
        print(segments, calls, row["prediction"], row["weight"], np.round(100*np.array(row["relative_error"]), 2),
              "ms", (perf_counter()-tick)*1000, flush=True)

    first_step, second_step = histories["one_front"][:2]
    projected = projection_test(states, edges, w, cut, first_step, second_step,
                                np.asarray(second_step["reference_exchange"]))
    output = dict(columns=["momentum", "pressure_work", "transport_energy", "net_energy"],
                  histories=histories, reference_refinement=refinement, common_first_step=first,
                  second_step_projection=projected)
    Path("dod_moving_rarefaction_ledger.json").write_text(json.dumps(output, indent=2)+"\n")


if __name__ == "__main__":
    main()
