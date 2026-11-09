"""Benchmark P0 domain-of-dependence stabilization on moving 1D cut cells."""
import json
from pathlib import Path
from itertools import combinations

import numpy as np

from nonlinear_cutcell_impact import DT, Impact, fields
from nonlinear_cutcell_predictor import fine_reference, samples


def grid(width, motion):
    regular = np.linspace(-.5, .5, 21)
    edges = np.sort(np.r_[regular[regular != 0], [-width / 2, width / 2]])
    cut = int(np.flatnonzero(edges == -width / 2)[0])
    w = np.zeros_like(edges)
    w[cut:cut+2] = {"fixed": (0., 0.), "drift": (.85, .85),
                    "squeeze": (10*width, -10*width)}[motion]
    return edges, cut, w


def channels(q, boundary, w, cut):
    padded = np.vstack((boundary[0], q, boundary[1]))
    i = np.r_[np.arange(len(q)+1), cut]
    j = np.r_[np.arange(1, len(q)+2), cut+2]
    a, b = padded[i], padded[j]
    speed = 1 / np.minimum(np.minimum(a[:, 0], b[:, 0]),
                           np.sqrt(a[:, 0]*b[:, 0]) * np.exp(-abs(a[:, 1]-b[:, 1])/2))
    weight0, weight1 = np.ones_like(speed), np.zeros_like(speed)
    weight0[[cut, cut+1]] = 0
    weight1[[cut, cut+1, -1]] = (1., 1., -1.)
    mean, diff = (w[cut]+w[cut+1])/2, w[cut+1]-w[cut]
    mu0, mu1 = np.r_[w, mean], np.zeros_like(speed)
    mu0[[cut, cut+1]] = (-diff/2, diff/2)
    mu1[[cut, cut+1, -1]] = (mean, mean, -mean)
    return i, j, a, b, speed, weight0, weight1, mu0, mu1


def cfl_lines(data, h):
    i, j, _, _, s, weight0, weight1, mu0, mu1 = data
    # A sum of positive parts is the maximum over all subsets of summands.
    # Each subset gives one affine inequality in beta.
    limits = []
    for k, size in enumerate(h, 1):
        es = np.flatnonzero((i == k) | (j == k))
        sign = np.where(i[es] == k, -1, 1)
        d0, d1 = weight0[es] @ s[es], weight1[es] @ s[es]
        o0, o1 = sign*mu0[es], sign*mu1[es]
        for count in range(len(es)+1):
            for subset in combinations(range(len(es)), count):
                indices = list(subset)
                limits.append(((d0+sum(o0[indices]))/size, (d1+sum(o1[indices]))/size))
    return np.array(limits).T


def beta_interval(lines, dt):
    a, b = lines
    if np.any(a[b == 0]*dt > 1):
        return None
    positive, negative = b > 0, b < 0
    hi = min(1., np.min((1/dt-a[positive])/b[positive], initial=np.inf))
    lo = max(0., np.max((1/dt-a[negative])/b[negative], initial=-np.inf))
    return (lo, hi) if hi >= lo and hi > 0 else None


def step_cap(lines, h, w, use_dod, horizon):
    shrink = -np.diff(w)
    geo = float(np.min(h[shrink > 0]/shrink[shrink > 0], initial=np.inf))
    if not use_dod:
        return min(horizon, geo, 1/max(lines.sum(axis=0)))
    low, high = 0., min(horizon, geo)
    for _ in range(40):
        mid = (low+high)/2
        if beta_interval(lines, mid) is None:
            high = mid
        else:
            low = mid
    return low


def entropy(q):
    return .5*q[:, 1]**2 + q[:, 0] - 1 - np.log(q[:, 0])


def update(q, h, w, data, cut, dt, beta):
    i, j, a, b, speed, weight0, weight1, mu0, mu1 = data
    weight, mu = weight0+beta*weight1, mu0+beta*mu1
    fa = np.column_stack((-a[:, 1], 1/a[:, 0]-1))
    fb = np.column_stack((-b[:, 1], 1/b[:, 0]-1))
    ea, eb = entropy(a), entropy(b)
    h_flux = .5*(fa+fb-speed[:, None]*(b-a))
    g_flux = .5*(fa[:, 1]*a[:, 1]+fb[:, 1]*b[:, 1]-speed*(eb-ea))
    mechanical = weight[:, None]*np.column_stack((h_flux, g_flux))
    transported = np.where(mu[:, None] < 0, np.column_stack((a, ea)), np.column_stack((b, eb)))
    total = mechanical - mu[:, None]*transported
    face_mech, face_total = mechanical[:-1].copy(), total[:-1].copy()
    face_mech[cut:cut+2] += mechanical[-1]
    face_total[cut:cut+2] += total[-1]

    h_next = h + dt*np.diff(w)
    Q_next = q*h[:, None] - dt*np.diff(face_total[:, :2], axis=0)
    q_next = Q_next / h_next[:, None]
    assert np.min(h_next) > 0 and np.min(q_next[:, 0]) > 0
    entropy_budget = h*entropy(q) - dt*np.diff(face_total[:, 2]) - h_next*entropy(q_next)
    balance = Q_next.sum(axis=0)-(q*h[:, None]).sum(axis=0)+dt*(face_total[-1, :2]-face_total[0, :2])
    assert max(abs(balance)) < 1e-13
    assert min(entropy_budget) >= -1e-13
    return q_next, h_next, dt*face_mech, dt*face_total, entropy_budget, max(abs(balance))


def run(width, motion, states, use_dod, horizon=DT, single=False):
    x0, cut, w = grid(width, motion)
    h = np.diff(x0)
    state = np.tile(states[0], (len(h), 1))
    state[cut] = states[1]
    state[cut+1:] = states[2]
    time, betas, steps = 0., [], []
    flux = np.zeros((2, 3))
    mech = np.zeros_like(flux)
    balance, energy_defect, energy_loss = 0., 0., 0.
    min_tau = float(state[:, 0].min())
    while time < horizon-1e-15:
        data = channels(state, (states[0], states[2]), w, cut)
        lines = cfl_lines(data, h)
        cap = step_cap(lines, h, w, use_dod, 1.)
        if not steps:
            first_cap = cap
        remaining = horizon-time
        dt = min(remaining, .99*cap)
        if single and dt < remaining:
            return dict(status="CFL refused", cap=first_cap)
        beta = beta_interval(lines, dt)[1] if use_dod else 1.
        assert max((lines[0]+beta*lines[1])*dt) <= 1+2e-14
        state, h, f_mech, f_total, entropy_budget, b = update(state, h, w, data, cut, dt, beta)
        min_tau = min(min_tau, state[:, 0].min())
        flux += f_total[cut:cut+2]
        mech += f_mech[cut:cut+2]
        balance = max(balance, b)
        energy_defect = max(energy_defect, -min(entropy_budget))
        energy_loss += sum(entropy_budget)
        time += dt
        betas.append(beta)
        steps.append(dt)
        assert len(steps) < 1000
    return dict(status="ok", cap=first_cap, steps=steps, beta=betas, balance=balance,
                entropy_inequality_defect=energy_defect, entropy_loss=energy_loss,
                min_tau=min_tau, thin_tau=float(state[cut, 0]), thin_state=state[cut].tolist(),
                face_flux=flux.tolist(), physical_flux=mech.tolist(),
                edges=(x0+time*w).tolist(), initial_edges=x0.tolist(),
                state=state.tolist(), final_h=h.tolist())


def relative(actual, reference):
    actual, reference = np.array(actual), np.array(reference)
    return (np.max(abs(actual-reference), axis=0)
            / np.maximum(np.max(abs(reference), axis=0), 1e-14)).tolist()


def check_exact(run_data, problem, t=DT):
    original = np.array(run_data["initial_edges"])
    final = np.array(run_data["edges"])
    cut = int(np.flatnonzero(original == -problem.phases[0][2])[0])
    face = [problem.face_integral(x, (y-x)/t) for x, y in zip(original[cut:cut+2], final[cut:cut+2])]
    ref_flux, ref_mech, _ = map(np.array, zip(*face))
    exact = problem.cell_integrals(final, t)
    h = np.array(run_data["final_h"])
    q = np.array(run_data["state"])
    physical = problem.physical_position(original[0], 0.) + problem.left[1]*t + np.r_[0., np.cumsum(h*q[:, 0])]
    ref_physical = problem.physical_position(final, t)
    resolved_ref_energy = h @ entropy(exact[:, :2]/h[:, None])
    resolved_ref_loss = problem.cell_integrals(original, 0.)[:, 2].sum()-resolved_ref_energy
    left_boundary = problem.face_integral(original[0], 0.)[0][2]
    right_boundary = problem.face_integral(original[-1], 0.)[0][2]
    resolved_ref_loss += left_boundary - right_boundary
    return dict(exact_thin_tau=float(exact[cut, 0]/h[cut]),
                exact_face_flux=ref_flux.tolist(), exact_physical_flux=ref_mech.tolist(),
                ale_relative_error=relative(run_data["face_flux"], ref_flux),
                mechanical_relative_error=relative(np.array(run_data["physical_flux"])[:, 1:], ref_mech[:, 1:]),
                max_position_error=float(max(abs(physical-ref_physical))),
                normalized_position_error=float(max(abs(physical-ref_physical))/(problem.left[1]*t)),
                reference_resolved_energy_loss=float(resolved_ref_loss))


def main():
    uniform = []
    for width in (.005, 1e-8):
        for motion in ("fixed", "drift", "squeeze"):
            q = np.array([.75, .27])
            r = run(width, motion, (q, q, q), True)
            error = float(np.max(abs(np.array(r["state"])-q)))
            assert error < 1e-9
            uniform.append(dict(width=width, motion=motion, steps=len(r["steps"]),
                                uniform_error=error))
    cases = []
    for width in (.005, 1e-8):
        for compression in (.65, .8):
            for motion in ("fixed", "drift", "squeeze"):
                problem = Impact(width, compression)
                state = (problem.left, problem.middle, problem.right)
                result = run(width, motion, state, True)
                exact = check_exact(result, problem)
                ordinary_cap = run(width, motion, state, False, single=True)["cap"]
                base = (run(width, motion, state, False) if width == .005 else None)
                if base is not None:
                    base["error"] = check_exact(base, problem)
                one_step = run(width, motion, state, True, single=True)
                cases.append(dict(width=width, compression=compression, motion=motion,
                                  ordinary_first_cap=ordinary_cap,
                                  dod_first_cap=result["cap"], one_step=one_step["status"],
                                  dod=result, error=exact, ordinary=base))
                print("impact", width, compression, motion, f"{ordinary_cap:.5g}->{result['cap']:.5g}",
                      len(result["steps"]), "tau", f"{result['thin_tau']:.4g}/{exact['exact_thin_tau']:.4g}",
                      "imp/work", np.round(100*np.array(exact["mechanical_relative_error"]), 2),
                      "pos", f"{exact['max_position_error']:.5g}", flush=True)

    general = []
    refinements = []
    for kind, index, states in samples():
        if kind not in ("compression", "mixed", "expansion") or index not in (0, 1):
            continue
        motion = ("fixed", "drift", "squeeze")[index%3]
        result = run(.005, motion, states, True)
        base = run(.005, motion, states, False)
        edges, cut, w = grid(.005, motion)
        flux, mean = fine_reference(states, edges[cut:cut+2], w[cut:cut+2], 5120)
        if (kind, index) in (("mixed", 1), ("expansion", 0)):
            fine_flux, fine_mean = fine_reference(states, edges[cut:cut+2], w[cut:cut+2], 10240)
            refinements.append(dict(kind=kind, index=index,
                                    flux_relative_change=relative(flux, fine_flux),
                                    thin_stretch_change=float(abs(mean[0]-fine_mean[0]))))
        for value in (result, base):
            value["ale_relative_error"] = relative(np.array(value["face_flux"])[:, 1], flux[:, 1])
            value["mechanical_relative_error"] = relative(np.array(value["physical_flux"])[:, 1:],
                                                           flux[:, [3, 4]])
            value["thin_tau_error"] = abs(value["thin_tau"]-mean[0])
        general.append(dict(kind=kind, index=index, motion=motion, reference_thin_state=mean.tolist(),
                            reference_flux=flux.tolist(), dod=result, ordinary=base))
        print("general", kind, motion, f"{base['cap']:.5g}->{result['cap']:.5g}",
              len(base["steps"]), len(result["steps"]),
              "imp/work", np.round(100*np.array(result["mechanical_relative_error"]), 2), flush=True)
    report = dict(uniform=uniform, exact_impact=cases, fine_fv=general,
                  fine_reference_refinement=refinements, time=DT)
    Path("dod_results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
