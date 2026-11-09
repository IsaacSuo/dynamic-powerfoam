"""Reuse the accepted ALE step when limiting a deferred rezone projection."""
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from check_dod import beta_interval, cfl_lines, channels, entropy, godunov_flux, grid, step_cap, update
from check_dod_rezone import advance, rezone_geometry
from check_dod_rezone_error_sources import limit_on_new_paths, unpack_rollout
from dod_half_state import ThinHistory, extra_energy
from dod_remap import bin_partition, donor_bins, overlap_integrals, remap
from nonlinear_cutcell_predictor import DT, face_interval, mean_in_cell, piecewise_cell_averages, samples


def accepted_step(states, motion="drift"):
    edges, cut, w = grid(.005, motion)
    width = np.diff(edges)
    q = np.tile(states[0], (len(width), 1))
    q[cut] = states[1]
    q[cut+1:] = states[2]
    data = channels(q, (states[0], states[-1]), w, cut, directional=True)
    cfl = cfl_lines(data, width)
    dt = .99*step_cap(cfl, width, w, True, 1.)
    beta = beta_interval(cfl, dt)[1]
    history = ThinHistory(keep_wave=True)
    statuses, weights = [], []
    mean, _, _, flux, loss, _ = update(q, width, w, data, cut, dt, beta, godunov_flux,
                                       outcomes=statuses, prediction_weights=weights, history=history)
    contrast = np.zeros_like(mean)
    contrast[cut] = history.contrast
    return edges, cut, w, dt, mean, contrast, flux, loss, history.wave, weights[0]


def replay(wave, origins, velocities, bins):
    left, trajectory, final = wave
    flux = np.zeros((len(origins), 6))
    for begin, end, fronts in trajectory:
        flux += face_interval(left, fronts, origins, velocities, begin, end, faces_only=True)[0]
    means = np.array([mean_in_cell(left, final, a, b)[:2] for a, b in zip(bins[:-1], bins[1:])])
    return means, flux


def stored_energy(width, mean, contrast):
    return width*entropy(mean)+np.array([extra_energy(q, d, h) for q, d, h in zip(mean, contrast, width)])


def available_from_old(edges, mean, contrast, loss, targets):
    source_edges, source = donor_bins(edges, mean, contrast)
    cell = np.searchsorted(edges[1:], source_edges[:-1], side="right")
    density = entropy(source)+loss[cell]/np.diff(edges)[cell]
    return overlap_integrals(source_edges, density[:, None], targets)[:, 0]


def integrated_flux(initial, available, left_boundary):
    delta = initial-available
    return np.r_[left_boundary, left_boundary+np.cumsum(delta)]


def limit_reuse(low, high, width, available_low, available_high):
    delta_q, delta_d = (b-a for a, b in zip(low, high))
    delta_energy = available_high-available_low

    def evaluate(weight):
        q = low[0]+weight*delta_q
        d = low[1]+weight*delta_d
        if np.any(q[:, 0] <= abs(d[:, 0])):
            return q, d, np.inf
        excess = stored_energy(width, q, d)-(available_low+weight*delta_energy)
        return q, d, float(max(excess))

    left, right = 0., 1.
    assert evaluate(left)[2] <= 1e-13
    if evaluate(right)[2] <= 1e-13:
        left = right
    else:
        for _ in range(24):
            middle = .5*(left+right)
            if evaluate(middle)[2] <= 1e-13:
                left = middle
            else:
                right = middle
    q, d, excess = evaluate(left)
    return q, d, dict(weight=left, max_local_energy_excess=excess)


def case(states, first, shift, reference=None):
    initial_edges, old_cut, old_w, dt, old_mean, old_d, old_flux, loss, wave, _ = first
    old_edges = initial_edges+dt*old_w
    old_faces = initial_edges[old_cut:old_cut+2]
    new_edges, cut = rezone_geometry(old_edges, old_cut, shift)
    width = np.diff(new_edges)
    flags = np.arange(len(width)) == cut
    bins, _ = bin_partition(new_edges, flags)
    w = np.zeros_like(new_edges)
    w[cut:cut+2] = old_w[old_cut:old_cut+2]
    origins = new_edges-dt*w

    wave_means, wave_flux = replay(wave, origins, w, bins)
    high = remap(bins, wave_means, np.zeros_like(wave_means), new_edges, flags)
    low = remap(old_edges, old_mean, old_d, new_edges, flags)
    initial_q = np.diff(origins)[:, None]*piecewise_cell_averages(origins, old_faces, states)
    initial_energy = np.diff(origins)*piecewise_cell_averages(origins, old_faces, entropy(np.array(states))[:, None])[:, 0]
    high_available = initial_energy-np.diff(wave_flux[:, 2])
    low_available = available_from_old(old_edges, old_mean, old_d, loss, new_edges)

    # The old-step-plus-remap budget is equivalent to a face flux on the new
    # paths, with the same two boundary exchanges as the wave candidate.
    low_energy_flux = integrated_flux(initial_energy, low_available, old_flux[0, 2])
    np.testing.assert_allclose(low_energy_flux[[0, -1]], wave_flux[[0, -1], 2], atol=2e-14, rtol=0)
    low_q_flux = np.column_stack([
        integrated_flux(initial_q[:, k], width*low[0][:, k], old_flux[0, k]) for k in range(2)])
    np.testing.assert_allclose(low_q_flux[[0, -1]], wave_flux[[0, -1], :2], atol=2e-14, rtol=0)
    np.testing.assert_allclose(initial_q-np.diff(wave_flux[:, :2], axis=0), width[:, None]*high[0],
                               atol=2e-14, rtol=0)

    def reused():
        available = available_from_old(old_edges, old_mean, old_d, loss, new_edges)
        return limit_reuse(low, high, width, available, high_available)

    q, d, info = reused()
    weight = info["weight"]
    mixed_q = low_q_flux+weight*(wave_flux[:, :2]-low_q_flux)
    mixed_energy = low_energy_flux+weight*(wave_flux[:, 2]-low_energy_flux)
    np.testing.assert_allclose(initial_q-np.diff(mixed_q, axis=0), width[:, None]*q, atol=2e-14, rtol=0)
    mixed_available = initial_energy-np.diff(mixed_energy)
    assert max(stored_energy(width, q, d)-mixed_available) <= 1e-13
    assert np.min(q[:, 0]-abs(d[:, 0])) > 0.
    if reference is None:
        return info

    def extra_substeps():
        return limit_on_new_paths(states, high, initial_q/np.diff(origins)[:, None], origins,
                                  new_edges, w, cut, dt, initial_energy, wave_flux)

    timings = [[], []]
    for trial in range(21):
        for i in ((0, 1) if trial%2 == 0 else (1, 0)):
            tick = perf_counter()
            (extra_substeps, reused)[i]()
            timings[i].append(perf_counter()-tick)

    rows = {}
    for name, solver in (("new_path_substeps", extra_substeps), ("reuse_old_step", reused)):
        q, d, info = solver()
        assert np.min(q[:, 0]-abs(d[:, 0])) > 0.
        result = advance(q, new_edges, cut, w, (states[0], states[-1]), d, [.004, .004, DT-dt-.008])
        rows[name] = dict(limit=info, post_rezone_flux=unpack_rollout(result).tolist(),
                          thin_state=result["thin_state"], statuses=result["statuses"])
    return dict(shift=shift, reference=reference["reference"], results=rows,
                old_budget_sum=float(sum(low_available)), wave_budget_sum=float(sum(high_available)),
                recorded_intervals=len(wave[1]),
                timing_median_ms=[round(float(np.median(samples))*1000, 4) for samples in timings])


def stress_test():
    cached = []
    unavailable = 0
    for _, index, states in samples():
        motion = ("fixed", "drift", "squeeze")[index%3]
        first = accepted_step(states, motion)
        if first[-2] is None:
            unavailable += 1
            continue
        for shift in (-.00125, .00125):
            result = case(states, first, shift)
            cached.append(dict(motion=motion, shift=shift, accepted_old_weight=first[-1], **result))
    return dict(initial_states=48, unavailable=unavailable, cached_checks=cached)


def main():
    states = next(q for kind, i, q in samples() if kind == "expansion" and i == 1)
    saved = Path(__file__).with_name("dod_rezone_error_sources.json")
    references = [r for r in json.loads(saved.read_text())["results"] if r["resolution"] == 10240]
    first = accepted_step(states)
    assert first[-2] is not None and first[-1] == 1.
    rows = [case(states, first, ref['shift'], ref) for ref in references]
    # Keep the reproducible numerical comparison separate from wall-clock timing.
    for row in rows:
        print(row["shift"], "limit ms", row.pop("timing_median_ms"), flush=True)
        print("reference right energy", row["reference"][1][3], flush=True)
        for name, value in row["results"].items():
            print(name, value["limit"], "right energy", value["post_rezone_flux"][1][3], flush=True)
    stress = stress_test()
    print("cached checks", len(stress["cached_checks"]), "unavailable", stress["unavailable"], flush=True)
    output = dict(continuations=rows, stress=stress)
    Path("dod_rezone_reuse_results.json").write_text(json.dumps(output, indent=2)+"\n")


if __name__ == "__main__":
    main()
