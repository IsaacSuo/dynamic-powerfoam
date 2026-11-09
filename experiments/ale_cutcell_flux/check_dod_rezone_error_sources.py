"""Separate pre-rezone, projection, and post-rezone errors in the moving test."""
import json
from pathlib import Path

import numpy as np

from check_dod import beta_interval, cfl_lines, channels, entropy, godunov_flux, grid, step_cap, update
from check_dod_half_state import rollout
from check_dod_rezone import advance, energy, rezone_geometry
from dod_half_state import extra_energy
from dod_remap import bin_partition, donor_bins, remap
from nonlinear_cutcell_predictor import DT, fine_reference, piecewise_cell_averages, predict, samples


def unpack_reference(flux):
    return np.column_stack((flux[:, 1], flux[:, 4], flux[:, 2]-flux[:, 4], flux[:, 2]))


def unpack_rollout(result):
    total, physical = map(np.asarray, (result["total_flux"], result["mechanical_flux"]))
    return np.column_stack((total[:, 1], physical[:, 2], total[:, 2]-physical[:, 2], total[:, 2]))


def restriction(union, averages, edges):
    prefix = np.vstack((np.zeros(averages.shape[1]), np.cumsum(np.diff(union)[:, None]*averages, axis=0)))
    index = np.searchsorted(union, edges)
    return np.diff(prefix[index], axis=0)/np.diff(edges)[:, None]


def limit_on_new_paths(states, wave_profile, initial_means, origins, edges, w, cut, dt,
                       initial_energy, wave_flux):
    # Both solutions use the same initial conservative data and virtual
    # face paths. The averaged Godunov update supplies an admissible endpoint.
    width = np.diff(edges)
    widths = np.diff(origins)
    baseline = initial_means
    base_flux = np.zeros_like(wave_flux[:, :3])
    time = 0.
    steps = []
    while time < dt-1e-15:
        data = channels(baseline, (states[0], states[-1]), w, cut, directional=True)
        lines = cfl_lines(data, widths)
        substep = min(dt-time, .99*step_cap(lines, widths, w, True, dt))
        beta = beta_interval(lines, substep)[1]
        baseline, widths, _, flux, _, _ = update(baseline, widths, w, data, cut, substep, beta, godunov_flux)
        base_flux += flux
        time += substep
        steps.append(substep)
    wave_mean, wave_contrast = wave_profile
    available0 = initial_energy-np.diff(base_flux[:, 2])
    available_delta = -np.diff(wave_flux[:, 2]-base_flux[:, 2])

    def check(weight):
        q = baseline+weight*(wave_mean-baseline)
        d = weight*wave_contrast
        stored = width*entropy(q)+np.array([extra_energy(qi, di, h) for qi, di, h in zip(q, d, width)])
        excess = stored-(available0+weight*available_delta)
        return q, d, float(max(excess))

    low, high = 0., 1.
    assert check(low)[2] <= 1e-13
    if check(high)[2] <= 1e-13:
        low = high
    else:
        for _ in range(24):
            middle = .5*(low+high)
            if check(middle)[2] <= 1e-13:
                low = middle
            else:
                high = middle
    q, d, excess = check(low)
    return q, d, dict(weight=low, max_local_energy_excess=excess, baseline_steps=steps)


def decompose(states, first, shift, resolution):
    old_edges = np.asarray(first["edges"])
    old_mean = np.asarray(first["state"])
    initial, old_cut, speeds = grid(.005, "drift")
    old_flags = np.arange(len(old_mean)) == old_cut
    old_d = np.zeros_like(old_mean)
    old_d[old_cut] = first["history"][-1]["contrast"]
    new_edges, cut = rezone_geometry(old_edges, old_cut, shift)
    new_flags = np.arange(len(new_edges)-1) == cut
    old_bins, _ = bin_partition(old_edges, old_flags)
    new_bins, _ = bin_partition(new_edges, new_flags)

    t0 = sum(first["steps"])
    horizon = DT-t0
    schedule = [.004, .004, horizon-.008]
    w = np.zeros_like(new_edges)
    w[cut:cut+2] = speeds[old_cut:old_cut+2]
    faces = new_edges[cut:cut+2]
    face_speeds = w[cut:cut+2]
    old_faces = initial[old_cut:old_cut+2]
    origins = faces-t0*face_speeds
    union = np.union1d(old_bins, new_bins)
    before, _, values = fine_reference(
        states, origins, face_speeds, resolution, horizon=t0, initial_breaks=old_faces,
        average_edges=union, conservative_initialization=True)
    after, final = fine_reference(
        states, origins, face_speeds, resolution, horizon=DT, initial_breaks=old_faces,
        conservative_initialization=True)
    reference = unpack_reference(after-before)

    exact_old = restriction(union, values, old_bins)
    exact_new = restriction(union, values, new_bins)
    old_zeros = np.zeros_like(exact_old)
    new_zeros = np.zeros_like(exact_new)
    profiles = {
        "computed_old_remap": remap(old_edges, old_mean, old_d, new_edges, new_flags),
        "exact_old_remap": remap(old_bins, exact_old, old_zeros, new_edges, new_flags),
        "direct_exact_new": remap(new_bins, exact_new, new_zeros, new_edges, new_flags),
    }
    baseline = profiles["computed_old_remap"]
    wave = predict(states, old_faces, face_speeds, t0, fan_limit=1, event_budget=4,
                   faces_only=True, sample_edges=new_bins)
    assert wave["status"] == "ok" and first["prediction_weights"] == [1.]
    wave_profile = remap(new_bins, wave["means"], new_zeros, new_edges, new_flags)
    difference = np.diff(new_edges)@(wave_profile[0]-baseline[0])
    np.testing.assert_allclose(difference, 0., atol=2e-14, rtol=0)
    committed_energy = energy(old_edges, old_mean, old_d)
    step_energy_cap = committed_energy+first["entropy_loss"]
    wave_energy = energy(new_edges, *wave_profile)
    new_origins = new_edges-t0*w
    all_faces = predict(states, new_origins, w, t0, fan_limit=1, event_budget=4,
                        faces_only=True, initial_breaks=old_faces)
    assert all_faces["status"] == "ok"
    widths = np.diff(new_edges)
    initial_means = piecewise_cell_averages(new_origins, old_faces, states)
    conserved = np.diff(new_origins)[:, None]*initial_means-np.diff(all_faces["face_fluxes"][:, :2], axis=0)
    np.testing.assert_allclose(conserved, widths[:, None]*wave_profile[0], atol=2e-14, rtol=0)
    initial_energy = piecewise_cell_averages(new_origins, old_faces, entropy(np.array(states))[:, None])[:, 0]
    available = np.diff(new_origins)*initial_energy-np.diff(all_faces["face_fluxes"][:, 2])
    cell_energy = widths*entropy(wave_profile[0])
    subcell_energy = np.array([extra_energy(q, d, h) for q, d, h in zip(*wave_profile, widths)])
    local_check = dict(mean_excess=(cell_energy-available).tolist(),
                       total_excess=(cell_energy+subcell_energy-available).tolist(), thin_cell=cut)
    energy_limits = {
        "deferred_before_old_projection": step_energy_cap,
        "preserve_committed_energy": committed_energy,
    }
    projection = {}
    for name, cap in energy_limits.items():
        low, high = 0., 1.
        assert energy(new_edges, *baseline) <= cap+1e-13
        if wave_energy <= cap+1e-13:
            low = 1.
        else:
            for _ in range(24):
                mid = .5*(low+high)
                candidate = tuple(a+mid*(b-a) for a, b in zip(baseline, wave_profile))
                if energy(new_edges, *candidate) <= cap:
                    low = mid
                else:
                    high = mid
        profiles[name] = tuple(a+low*(b-a) for a, b in zip(baseline, wave_profile))
        projection[name] = dict(weight=low, energy=energy(new_edges, *profiles[name]), cap=cap)
    q, d, info = limit_on_new_paths(states, wave_profile, initial_means, new_origins, new_edges, w, cut, t0,
                                    np.diff(new_origins)*initial_energy, all_faces["face_fluxes"])
    profiles["locally_limited_on_new_paths"] = (q, d)
    projection["locally_limited_on_new_paths"] = info
    fine_flux, coarse_flux, details = {}, {}, {}
    for name, (mean, contrast) in profiles.items():
        bins, values = donor_bins(new_edges, mean, contrast)
        flux, _ = fine_reference(values, faces, face_speeds, resolution, horizon=horizon,
                                 initial_breaks=bins[1:-1], conservative_initialization=True)
        fine_flux[name] = unpack_reference(flux)
        result = advance(mean, new_edges, cut, w, (states[0], states[-1]), contrast, schedule)
        coarse_flux[name] = unpack_rollout(result)
        details[name] = dict(fine=fine_flux[name].tolist(), coarse=coarse_flux[name].tolist(),
                             thin_mean=mean[cut].tolist(), thin_contrast=contrast[cut].tolist(),
                             final_thin_state=result["thin_state"], statuses=result["statuses"],
                             weights=result["weights"])

    a, b, c = "computed_old_remap", "exact_old_remap", "direct_exact_new"
    contributions = {
        "coarse_advance": coarse_flux[a]-fine_flux[a],
        "old_state_error": fine_flux[a]-fine_flux[b],
        "old_partition_projection": fine_flux[b]-fine_flux[c],
        "new_partition_projection": fine_flux[c]-reference,
    }
    np.testing.assert_allclose(sum(contributions.values()), coarse_flux[a]-reference, atol=2e-17, rtol=0)

    mean, contrast = profiles[a]
    target_mean, target_contrast = profiles[c]
    isolated = {}
    for name in ("replace_mean", "replace_contrast", "replace_both"):
        q, d = mean.copy(), contrast.copy()
        if name in ("replace_mean", "replace_both"):
            q[cut] = target_mean[cut]
        if name in ("replace_contrast", "replace_both"):
            d[cut] = target_contrast[cut]
        result = advance(q, new_edges, cut, w, (states[0], states[-1]), d, schedule)
        isolated[name] = unpack_rollout(result).tolist()
    return dict(shift=shift, resolution=resolution, reference=reference.tolist(), reference_stretch=float(final[0]),
                cases=details, error_components={k: v.tolist() for k, v in contributions.items()},
                replace_thin_only=isolated, wave_projection=projection,
                wave_conservation_error=difference.tolist(), wave_energy=wave_energy, local_wave_energy=local_check)


def main():
    states = next(q for kind, i, q in samples() if kind == "expansion" and i == 1)
    t0 = rollout(.005, "drift", states, True)["steps"][0]
    first = rollout(.005, "drift", states, True, horizon=t0)
    data = []
    for shift in (.00125, -.00125):
        for n in (5120, 10240):
            row = decompose(states, first, shift, n)
            data.append(row)
            print(shift, n, "right face, reference", np.round(1e6*np.array(row["reference"])[1], 3), flush=True)
            for k, r in row["cases"].items():
                print(k, np.round(1e6*np.array(r["fine"])[1], 3),
                      np.round(1e6*np.array(r["coarse"])[1], 3), flush=True)
            local = row["local_wave_energy"]
            print("local max energy excess", max(local["mean_excess"]), max(local["total_excess"]), flush=True)
            print("projection limits", row["wave_projection"], flush=True)
            print("components", {k: round(1e6*r[1][3], 3) for k, r in row["error_components"].items()},
                  flush=True)
    report = dict(columns=["momentum", "pressure_work", "transport_energy", "net_energy"], results=data)
    Path("dod_rezone_error_sources.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
