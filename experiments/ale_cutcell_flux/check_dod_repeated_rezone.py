"""Run the one-dimensional ALE solver with a rezone after every physical step."""
from collections import Counter
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from check_dod import beta_interval, cfl_lines, channels, entropy, godunov_flux, grid, initial_state, step_cap, update
from check_dod_rezone_reuse import available_from_old, limit_reuse, replay, stored_energy
from dod_half_state import ThinHistory
from dod_one_interaction import half_means, integrate
from dod_remap import donor_bins, remap
from nonlinear_cutcell_impact import DT, Impact
from nonlinear_cutcell_predictor import fine_reference, piecewise_cell_averages, samples


def geometry(width, motion):
    edges, cut, w = grid(width, motion)
    original_face = edges[2]
    steps = 10
    dt = DT/steps
    stages = []
    for i in range(steps):
        advanced = edges+dt*w
        new_edges = advanced.copy()
        shifted = advanced[cut:cut+2]+(-1)**i*.25*width
        new_edges[cut:cut+2] = shifted
        remove, insert = (original_face, .475) if i%2 == 0 else (.475, original_face)
        new_edges = np.sort(np.r_[new_edges[new_edges != remove], insert])
        assert len(new_edges) == len(edges)
        new_cut = int(np.searchsorted(new_edges, shifted[0]))
        new_w = np.zeros_like(new_edges)
        new_w[new_cut:new_cut+2] = w[cut:cut+2]
        stages.append((dt, edges, cut, w, advanced, new_edges, new_cut, new_w))
        edges, cut, w = new_edges, new_cut, new_w
    return stages


def project_cached(stage, initial, initial_d, old_flux, loss, old_mean, old_d, low, history):
    dt, edges, cut, _, advanced, new_edges, new_cut, new_w = stage
    cells = slice(new_cut-1, new_cut+2)
    old_cells = slice(cut-1, cut+2)
    old_faces = slice(cut-1, cut+3)
    local = new_edges[new_cut-1:new_cut+3]
    velocities = new_w[new_cut-1:new_cut+3]
    origins = local-dt*velocities
    new_faces = local[1:3]
    center = .5*sum(edges[cut:cut+2])
    halves = np.linspace(*new_faces, 3)-center
    face_origins = origins[1:3]-center
    if history.wave is not None:
        mean_halves, interior = replay(history.wave, face_origins, velocities[1:3], halves)
        path = "coarse"
    else:
        left, trace, packets = history.exact_wave
        interior = sum(integrate(left, waves, face_origins, velocities[1:3], a, b)
                       for a, b, waves in trace)
        mean_halves = half_means(left, packets, halves, dt)
        path = "exact"

    # Keep the accepted fluxes on the outer faces of the three-cell patch.
    assert np.array_equal(local[[0, -1]], advanced[[cut-1, cut+2]])
    flux = np.vstack((old_flux[cut-1], interior[:, :3], old_flux[cut+2]))
    donor_edges, donor = donor_bins(edges[old_faces], initial[old_cells], initial_d[old_cells])
    starts_q = np.diff(origins)[:, None]*piecewise_cell_averages(origins, donor_edges[1:-1], donor)
    starts_energy = np.diff(origins)*piecewise_cell_averages(
        origins, donor_edges[1:-1], entropy(donor)[:, None])[:, 0]
    widths = np.diff(local)
    high_q = (starts_q-np.diff(flux[:, :2], axis=0))/widths[:, None]
    high_d = np.zeros_like(high_q)
    high_d[1] = .5*(mean_halves[1]-mean_halves[0])
    assert np.allclose(high_q[1], mean_halves.mean(axis=0), atol=2e-8, rtol=2e-14)
    available_high = starts_energy-np.diff(flux[:, 2])
    available_low = available_from_old(advanced[old_faces], old_mean[old_cells], old_d[old_cells],
                                       loss[old_cells], local)
    q0, d0 = low[0][cells], low[1][cells]
    assert np.max(abs(widths@(high_q-q0))) < 2e-14
    assert abs(sum(available_high)-sum(available_low)) < 2e-14
    q, d, info = limit_reuse((q0, d0), (high_q, high_d), widths, available_low, available_high)
    low[0][cells], low[1][cells] = q, d
    return path, info["weight"]


def run(stages, states, reuse):
    _, edges, cut, *_ = stages[0]
    mean = initial_state(np.diff(edges), cut, states)
    contrast = np.zeros_like(mean)
    history = ThinHistory(keep_wave=reuse, keep_exact_wave=reuse)
    flux = np.zeros((2, 4))
    counts, weights = Counter(), []
    max_balance, max_energy_gain = 0., 0.
    for stage in stages:
        dt, edges, cut, w, advanced, new_edges, new_cut, _ = stage
        h = np.diff(edges)
        data = channels(mean, (states[0], states[-1]), w, cut, directional=True)
        interval = beta_interval(cfl_lines(data, h), dt)
        assert interval is not None
        history.contrast = contrast[cut].copy()
        status, limits = [], []
        advanced_mean, _, fm, ft, loss, _ = update(
            mean, h, w, data, cut, dt, interval[1], godunov_flux,
            outcomes=status, prediction_weights=limits, history=history)
        advanced_d = np.zeros_like(advanced_mean)
        advanced_d[cut] = history.contrast
        low = remap(advanced, advanced_mean, advanced_d, new_edges, np.arange(len(mean)) == new_cut)
        if reuse:
            if history.wave is not None or history.exact_wave is not None:
                path, weight = project_cached(stage, mean, contrast, ft, loss, advanced_mean, advanced_d, low, history)
                counts[path] += 1
                weights.append(weight)
            else:
                counts[status[0]] += 1
        widths = np.diff(new_edges)
        residual = widths@low[0]-h@mean+ft[-1, :2]-ft[0, :2]
        max_balance = max(max_balance, float(max(abs(residual))))
        energy_before = sum(stored_energy(h, mean, contrast))
        energy_after = sum(stored_energy(widths, *low))
        gain = energy_after-energy_before+ft[-1, 2]-ft[0, 2]
        max_energy_gain = max(max_energy_gain, float(gain))
        assert max_balance < 2e-14 and max_energy_gain < 2e-13
        assert np.min(low[0][:, 0]-abs(low[1][:, 0])) > 0.
        faces = slice(cut, cut+2)
        flux += np.column_stack((ft[faces, 1], fm[faces, 2], ft[faces, 2]-fm[faces, 2], ft[faces, 2]))
        mean, contrast = low
    uniform = np.array_equal(states[0], states[1]) and np.array_equal(states[1], states[2])
    return dict(exchange=flux.tolist(), thin_state=mean[new_cut].tolist(), counts=dict(counts),
                weights=weights, max_conserved_residual=max_balance, max_energy_gain=max_energy_gain,
                **({"uniform_error": float(np.max(abs(mean-states[0])))} if uniform else {}))


def path_segments(stages):
    time = 0.
    for dt, edges, cut, w, *_ in stages:
        faces = slice(cut, cut+2)
        yield time+dt, edges[faces]-time*w[faces], w[faces]
        time += dt


def final_faces(stages):
    *_, edges, cut, _ = stages[-1]
    return edges[cut:cut+2]


def impact_reference(problem, stages):
    flux = np.zeros((2, 4))
    begin = 0.
    for end, origins, speeds in path_segments(stages):
        total, physical, _ = map(np.array, zip(
            *(problem.face_integral(x, w, begin, end) for x, w in zip(origins, speeds))))
        flux += np.column_stack((total[:, 1], physical[:, 2], total[:, 2]-physical[:, 2], total[:, 2]))
        begin = end
    faces = final_faces(stages)
    mean = problem.cell_integrals(faces, DT)[0]/np.ptp(faces)
    return flux, mean[:2]


def rarefaction_reference(states, stages, resolution):
    _, edges, cut, w, *_ = stages[0]
    faces = slice(cut, cut+2)
    flux, _, average = fine_reference(states, edges[faces], w[faces], resolution, horizon=DT,
                                     face_schedule=list(path_segments(stages)), average_edges=final_faces(stages),
                                     conservative_initialization=True)
    return np.column_stack((flux[:, 1], flux[:, 4], flux[:, 2]-flux[:, 4], flux[:, 2])), average[0]


def compare(stages, states, reference, reference_thin):
    outputs = {}
    for name, reuse in (("overlap", False), ("reuse", True)):
        row = run(stages, states, reuse)
        row["signed_error"] = (np.array(row["exchange"])-reference).tolist()
        row["thin_stretch_error"] = abs(row["thin_state"][0]-reference_thin[0])
        outputs[name] = row
    return dict(reference_exchange=reference.tolist(), reference_thin_state=reference_thin.tolist(), **outputs)


def original_first_step_coverage(tests):
    counts = Counter()
    for _, index, states in tests:
        edges, cut, w = grid(.005, ("fixed", "drift", "squeeze")[index%3])
        h = np.diff(edges)
        q = initial_state(h, cut, states)
        data = channels(q, (states[0], states[-1]), w, cut, directional=True)
        lines = cfl_lines(data, h)
        dt = .99*step_cap(lines, h, w, True, 1.)
        history = ThinHistory(keep_wave=True, keep_exact_wave=True)
        outcomes = []
        update(q, h, w, data, cut, dt, beta_interval(lines, dt)[1], godunov_flux,
               outcomes=outcomes, prediction_weights=[], history=history)
        if history.wave is not None:
            counts["coarse"] += 1
        elif history.exact_wave is not None:
            counts["exact"] += 1
        else:
            counts[outcomes[0]] += 1
    return dict(counts)


def uniform_checks():
    q = np.array([.75, .27])
    rows = []
    for width in (.005, 1e-8):
        for motion in ("fixed", "drift", "squeeze"):
            result = run(geometry(width, motion), (q, q, q), True)
            assert result["uniform_error"] < 1e-8
            rows.append(dict(width=width, motion=motion, error=result["uniform_error"]))
    return rows


def main():
    tests = list(samples())
    q = next(q for kind, i, q in tests if kind == "expansion" and i == 1)
    stages = geometry(.005, "drift")
    fine, mean = rarefaction_reference(q, stages, 10240)
    coarse, _ = rarefaction_reference(q, stages, 5120)
    expansion = compare(stages, q, fine, mean)
    expansion["reference_refinement"] = np.max(abs(fine-coarse), axis=0).tolist()
    shocks = []
    for width in (.005, 1e-8):
        for compression in (.65, .8):
            p = Impact(width, compression)
            for motion in ("fixed", "drift", "squeeze"):
                stages = geometry(width, motion)
                ref, mean = impact_reference(p, stages)
                shocks.append(dict(width=width, compression=compression, motion=motion,
                                   **compare(stages, (p.left, p.middle, p.right), ref, mean)))
    sampled = []
    for kind, i, states in tests:
        motion = ("fixed", "drift", "squeeze")[i%3]
        stages = geometry(.005, motion)
        baseline = run(stages, states, False)
        result = run(stages, states, True)
        sampled.append(dict(kind=kind, index=i, motion=motion, overlap=baseline, reuse=result))

    print("expansion", expansion, flush=True)
    for row in shocks:
        print("shock", row["width"], row["compression"], row["motion"],
              [max(abs(np.array(row[k]["signed_error"])[:, 3])) for k in ("overlap", "reuse")],
              row["reuse"]["counts"], flush=True)
    times = [[], []]
    for trial in range(7):
        for i in ((0, 1) if trial%2 == 0 else (1, 0)):
            tick = perf_counter()
            for _, index, states in tests:
                run(geometry(.005, ("fixed", "drift", "squeeze")[index%3]), states, bool(i))
            times[i].append(perf_counter()-tick)
    print("batch wall time median (ms)", [float(np.median(t))*1000 for t in times])
    output = dict(columns=["momentum", "pressure_work", "transport_energy", "net_energy"],
                  expansion=expansion, shocks=shocks, sampled=sampled,
                  original_first_step_coverage=original_first_step_coverage(tests), uniform=uniform_checks())
    Path("dod_repeated_rezone_results.json").write_text(json.dumps(output, indent=2)+"\n")


if __name__ == "__main__":
    main()
