"""Validate one-dimensional half-state remapping and a post-rezone rollout."""
import json
from pathlib import Path

import numpy as np

from check_dod import beta_interval, cfl_lines, channels, entropy, godunov_flux, grid, update
from check_dod_half_state import rollout
from dod_half_state import ThinHistory, extra_energy
from dod_remap import donor_bins, remap
from nonlinear_cutcell_predictor import DT, fine_reference, predict, samples


def energy(edges, mean, contrast):
    width = np.diff(edges)
    return float(width@entropy(mean) + sum(extra_energy(q, d, h) for q, d, h in zip(mean, contrast, width)))


def audit(old_edges, old_mean, old_contrast, new_edges, new_mean, new_contrast):
    residual = np.diff(new_edges)@new_mean-np.diff(old_edges)@old_mean
    loss = energy(old_edges, old_mean, old_contrast)-energy(new_edges, new_mean, new_contrast)
    assert max(abs(residual)) < 2e-14
    assert np.min(new_mean[:, 0]-abs(new_contrast[:, 0])) > 0.
    assert loss >= -2e-14
    return dict(conserved_residual=residual.tolist(), energy_loss=loss)


def geometric_checks():
    old = np.array([0., .2, .7, 1.])
    mean = np.array([[.9, .2], [1.1, -.1], [.8, .1]])
    contrast = np.array([[0., 0.], [.25, -.2], [0., 0.]])
    flags = [False, True, False]
    m, d = remap(old, mean, contrast, old, flags)
    np.testing.assert_allclose(m, mean, rtol=0, atol=4e-15)
    np.testing.assert_allclose(d, contrast, rtol=0, atol=4e-15)
    identical = audit(old, mean, contrast, old, m, d)
    aligned = np.insert(old, 2, .45)
    m, d = remap(old, mean, contrast, aligned, [False]*4)
    aligned_split = audit(old, mean, contrast, aligned, m, d)
    r, s = remap(aligned, m, d, old, flags)
    np.testing.assert_allclose(r, mean, rtol=0, atol=4e-15)
    np.testing.assert_allclose(s, contrast, rtol=0, atol=4e-15)
    aligned_merge = audit(aligned, m, d, old, r, s)

    sliver = np.array([0., .45-5e-9, .45+5e-9, 1.])
    q = np.broadcast_to([.75, .27], mean.shape)
    flat, flat_d = remap(old, q, np.zeros_like(q), sliver, flags)
    uniform_error = float(np.max(abs(flat-q)))
    assert uniform_error < 4e-15
    m, d = remap(old, mean, contrast, sliver, flags)
    sliver_record = audit(old, mean, contrast, sliver, m, d)

    shifted = np.array([0., .225, .725, 1.])
    q, d, edges = mean, contrast, old
    accumulated_loss = 0.
    for i in range(50):
        target = shifted if i%2 == 0 else old
        next_q, next_d = remap(edges, q, d, target, flags)
        accumulated_loss += audit(edges, q, d, target, next_q, next_d)["energy_loss"]
        q, d, edges = next_q, next_d, target
    round_trip = audit(old, mean, contrast, edges, q, d)
    np.testing.assert_allclose(round_trip["energy_loss"], accumulated_loss, atol=2e-14, rtol=0)
    return dict(identity=identical, aligned_split=aligned_split, aligned_merge=aligned_merge,
                sliver=sliver_record, uniform_error=uniform_error, round_trip=round_trip,
                round_trip_initial_contrast=contrast[1].tolist(), round_trip_final_contrast=d[1].tolist())


def random_checks():
    rng = np.random.default_rng(392)
    worst_balance, least_loss = 0., np.inf
    for i in range(200):
        old = np.r_[0., np.sort(rng.uniform(.01, .99, size=3+i%11)), 1.]
        new = np.r_[0., np.sort(rng.uniform(.01, .99, size=2+i%13)), 1.]
        shape = (len(old)-1, 2)
        mean = rng.uniform([.5, -.8], [1.5, .8], size=shape)
        contrast = rng.uniform(-.4, .4, size=shape)
        contrast[rng.random(len(mean)) < .5] = 0.
        flags = rng.random(len(new)-1) < .5
        q, d = remap(old, mean, contrast, new, flags)
        info = audit(old, mean, contrast, new, q, d)
        worst_balance = max(worst_balance, max(abs(np.array(info["conserved_residual"]))))
        least_loss = min(least_loss, info["energy_loss"])
    return dict(cases=200, max_conserved_residual=worst_balance, smallest_energy_loss=least_loss)


class FirstRezoneProfile(ThinHistory):
    """Keep the overlap intervals through the first flux evaluation only."""

    def __init__(self, breaks, states):
        super().__init__()
        self.profile = (breaks, states)
        self.event_count = None

    def predict(self, states, width, speeds, dt):
        if self.profile is None:
            return super().predict(states, width, speeds, dt)
        breaks, interior = self.profile
        self.profile = None
        self.predicted_halves = None
        q = [tuple(states[0]), *map(tuple, interior), tuple(states[-1])]
        starts = breaks[[0, -1]]
        result = predict(q, starts, speeds, dt, fan_limit=1, event_budget=4, faces_only=True,
                         initial_breaks=breaks, sample_edges=np.linspace(*(starts+speeds*dt), 3))
        self.event_count = result["events"]
        if result["status"] != "ok":
            return super().predict(states, width, speeds, dt)
        self.predicted_halves = result["means"]
        flux = result["face_fluxes"]/dt
        return "overlap_profile", flux[:, [5, 3, 4]], flux[:, :3]


def thin_overlap_profile(source_edges, source_mean, source_d, new_faces):
    edges, values = donor_bins(source_edges, source_mean, source_d)
    breaks = np.r_[new_faces[0], edges[(edges > new_faces[0]) & (edges < new_faces[1])], new_faces[1]]
    state = values[np.searchsorted(edges, breaks[:-1], side="right")-1]
    return breaks-.5*sum(new_faces), state


def advance(mean, edges, cut, speeds, boundary, contrast, schedule, stored=None):
    h = np.diff(edges)
    before = energy(edges, mean, contrast)
    stored = ThinHistory() if stored is None else stored
    stored.contrast = contrast[cut].copy()
    total, mechanical = np.zeros((2, 3)), np.zeros((2, 3))
    statuses, weights = [], []
    outflow, dissipation = 0., 0.
    for dt in schedule:
        data = channels(mean, boundary, speeds, cut, directional=True)
        lines = cfl_lines(data, h)
        interval = beta_interval(lines, dt)
        assert interval is not None
        beta = interval[1]
        assert max(dt*(lines[0]+beta*lines[1])) <= 1+2e-14
        mean, h, fm, ft, entropy_budget, _ = update(mean, h, speeds, data, cut, dt, beta,
                                                   godunov_flux, outcomes=statuses,
                                                   prediction_weights=weights, history=stored)
        faces = slice(cut, cut+2)
        mechanical += fm[faces]
        total += ft[faces]
        outflow += ft[-1, 2]-ft[0, 2]
        dissipation += sum(entropy_budget)
    final = float(h@entropy(mean))+stored.energy(mean[cut], h[cut])
    residual = before-outflow-final-dissipation
    assert abs(residual) < 1e-12
    return dict(total_flux=total.tolist(), mechanical_flux=mechanical.tolist(),
                thin_state=mean[cut].tolist(), statuses=statuses, weights=weights,
                energy_residual=residual, history=stored.steps,
                overlap_events=getattr(stored, "event_count", None))


def reference(states, old_faces, new_faces, speeds, t0, stop, count):
    origin = new_faces-t0*speeds
    before, _ = fine_reference(states, origin, speeds, count, horizon=t0, initial_breaks=old_faces)
    after, mean = fine_reference(states, origin, speeds, count, horizon=stop, initial_breaks=old_faces)
    return after-before, mean


def continuation_case(states, first, shift):
    source_edges = np.array(first["edges"])
    source_mean = np.array(first["state"])
    _, source_cut, initial_speed = grid(.005, "drift")
    source_d = np.zeros_like(source_mean)
    source_d[source_cut] = first["history"][-1]["contrast"]
    target_edges = source_edges.copy()
    target_edges[source_cut:source_cut+2] += shift
    # Delete a distant face and split another ordinary cell. The interacting
    # three-cell patch still has a single thin control volume after rezoning.
    target_edges = np.delete(target_edges, 2)
    target_edges = np.insert(target_edges, -1, .5*(target_edges[-2]+target_edges[-1]))
    cut = source_cut-1
    flags = np.zeros(len(target_edges)-1, dtype=bool)
    flags[cut] = True
    keep_mean, keep_d = remap(source_edges, source_mean, source_d, target_edges, flags)
    reset_mean, reset_d = remap(source_edges, source_mean, source_d, target_edges, np.zeros_like(flags))
    np.testing.assert_allclose(keep_mean, reset_mean, atol=2e-14, rtol=0)
    w = np.zeros_like(target_edges)
    w[cut:cut+2] = initial_speed[source_cut:source_cut+2]
    t0 = sum(first["steps"])
    schedule = [.004, .004, DT-t0-.008]
    assert schedule[-1] > 0.
    profile_edges, profile_states = thin_overlap_profile(
        source_edges, source_mean, source_d, target_edges[cut:cut+2])
    np.testing.assert_allclose(np.diff(profile_edges)@profile_states/np.ptp(profile_edges),
                               keep_mean[cut], atol=2e-14, rtol=0)
    profile = FirstRezoneProfile(profile_edges, profile_states)
    rows = {}
    variants = (("retain", keep_mean, keep_d, None), ("reset", reset_mean, reset_d, None),
                ("retain_profile_once", keep_mean, keep_d, profile))
    for name, q, d, history in variants:
        projection = audit(source_edges, source_mean, source_d, target_edges, q, d)
        rows[name] = dict(remap=projection, starting_contrast=d[cut].tolist(),
                          **advance(q, target_edges, cut, w, (states[0], states[2]), d, schedule, history))
    old_faces = np.asarray(first["initial_edges"])[source_cut:source_cut+2]
    new_faces = target_edges[cut:cut+2]
    fine, thin = reference(states, old_faces, new_faces, w[cut:cut+2], t0, DT, 10240)
    coarse, _ = reference(states, old_faces, new_faces, w[cut:cut+2], t0, DT, 5120)
    for row in rows.values():
        computed = np.asarray(row["total_flux"])
        row["max_momentum_error"] = float(max(abs(computed[:, 1]-fine[:, 1])))
        row["max_energy_error"] = float(max(abs(computed[:, 2]-fine[:, 2])))
        row["thin_stretch_error"] = abs(row["thin_state"][0]-float(thin[0]))
    return dict(shift=shift, rezone_time=t0, dt=schedule, reference_flux=fine.tolist(),
                reference_thin_state=thin.tolist(), reference_refinement=np.max(abs(fine-coarse), axis=0).tolist(),
                thin_profile_breaks=profile_edges.tolist(), thin_profile=profile_states.tolist(),
                **rows)


def main():
    states = next(q for kind, i, q in samples() if kind == "expansion" and i == 1)
    cap = rollout(.005, "drift", states, True)["steps"][0]
    first = rollout(.005, "drift", states, True, horizon=cap)
    assert len(first["steps"]) == 1
    checks = geometric_checks()
    cases = [continuation_case(states, first, shift) for shift in (-.00125, .00125)]
    output = dict(geometry=checks, randomized_geometry=random_checks(), continuations=cases)
    Path("dod_rezone_results.json").write_text(json.dumps(output, indent=2)+"\n")
    print(checks, flush=True)
    for case in cases:
        print(case["shift"], "energy reference resolution change", case["reference_refinement"][2], flush=True)
        for k in ("reset", "retain", "retain_profile_once"):
            row = case[k]
            print(k, "remap loss", row["remap"]["energy_loss"], "net energy error", row["max_energy_error"],
                  "momentum error", row["max_momentum_error"], "stretch", row["thin_stretch_error"],
                  row["statuses"], flush=True)


if __name__ == "__main__":
    main()
