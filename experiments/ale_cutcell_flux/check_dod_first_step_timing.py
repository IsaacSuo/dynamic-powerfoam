"""Isolate first-step pressure and transport errors in the exact two-shock test."""
import json
from pathlib import Path

import numpy as np

from check_dod import cfl_lines, channels, entropy, godunov_flux, grid, initial_state, run
from nonlinear_cutcell_impact import Impact, fields


def first_step_faces(problem, x, cut, w, dt, beta):
    h = np.diff(x)
    states = (problem.left, problem.middle, problem.right)
    q = initial_state(h, cut, states)
    data = channels(q, (states[0], states[2]), w, cut, directional=True)
    _, _, left, right, speed, weight0, weight1, mu0, mu1 = data
    weight = weight0 + beta*weight1
    mu = mu0 + beta*mu1
    physical, total = godunov_flux(left, right, speed, weight, mu)
    physical, transport = dt*physical, dt*(total-physical)
    for flux in (physical, transport):
        flux[cut:cut+2] += flux[-1]
    return q, physical[:-1], transport[:-1], cfl_lines(data, h)


def scheduled_faces(problem, x, cut, w, dt):
    collision = problem.collision
    reflected_speed = problem.phases[1][3]
    faces = slice(cut, cut+2)
    arrival = (abs(x[faces]) + reflected_speed*collision) / (reflected_speed + w[faces]*[1, -1])
    pre = np.minimum(arrival, dt)[:, None]
    outer_state, outer_flux = map(np.array, zip(fields(problem.left), fields(problem.right)))
    star_state, star_flux = fields(problem.star)
    physical = pre*outer_flux + (dt-pre)*star_flux
    transport = -w[faces, None]*(pre*outer_state + (dt-pre)*star_state)
    return arrival, physical, transport


def measure(q, h, h_next, physical, transport, cut, reference, reference_physical, reference_transport):
    flux = physical+transport
    conserved = q*h[:, None] - np.diff(flux[:, :2], axis=0)
    mean = conserved/h_next[:, None]
    assert min(mean[:, 0]) > 0.
    net_input = -np.diff(flux[:, 2])
    energy = h_next*entropy(mean)
    residual = energy - h*entropy(q) - net_input
    balance = conserved.sum(axis=0) - (q*h[:, None]).sum(axis=0) + flux[-1, :2]-flux[0, :2]
    assert max(abs(balance)) < 1e-13
    faces = slice(cut, cut+2)
    return dict(thin_state=mean[cut].tolist(),
                max_thin_state_error=float(max(abs(mean[cut]-reference))),
                face_pressure_work=physical[faces, 2].tolist(),
                max_pressure_work_error=float(max(abs(physical[faces, 2]-reference_physical[:, 2]))),
                face_transport_energy=transport[faces, 2].tolist(),
                max_transport_energy_error=float(max(abs(transport[faces, 2]-reference_transport[:, 2]))),
                face_net_energy=flux[faces, 2].tolist(),
                thin_energy=float(energy[cut]),
                thin_net_input=float(net_input[cut]),
                thin_entropy_loss=float(-residual[cut]),
                max_positive_entropy_residual=float(max(0., max(residual))))


def check(compression, motion):
    width = .005
    problem = Impact(width, compression)
    x, cut, w = grid(width, motion)
    h = np.diff(x)
    saved = run(width, motion, (problem.left, problem.middle, problem.right), True, variant="godunov")
    dt, beta = saved["steps"][0], saved["beta"][0]
    q, physical, transport, cfl = first_step_faces(problem, x, cut, w, dt, beta)
    arrival, exact_physical, exact_transport = scheduled_faces(problem, x, cut, w, dt)
    assert max(arrival) < dt
    x_next = x+dt*w
    h_next = h+dt*np.diff(w)
    reference = problem.cell_integrals(x_next, dt)[cut]/h_next[cut]

    # Independent space-time integration over wave crossings, restricted to the first step.
    problem.phases = [(t0, min(t1, dt), start, speed, center)
                      for t0, t1, start, speed, center in problem.phases if t0 < dt]
    integrals = [problem.face_integral(x[i], w[i]) for i in (cut, cut+1)]
    oracle_flux, oracle_moments, _ = map(np.array, zip(*integrals))
    np.testing.assert_allclose(exact_physical, oracle_moments*[-1, 1, 1], atol=2e-14, rtol=0)
    np.testing.assert_allclose(exact_physical+exact_transport, oracle_flux, atol=2e-14, rtol=0)
    shock_loss = problem.shock_loss(x, w)[cut]

    corrected_physical, corrected_transport = physical.copy(), transport.copy()
    corrected_physical[cut:cut+2] = exact_physical
    corrected_transport[cut:cut+2] = exact_transport
    variants = {
        "godunov": (physical, transport),
        "timed_physical_only": (corrected_physical, transport),
        "timed_transport_only": (physical, corrected_transport),
        "timed_both": (corrected_physical, corrected_transport),
    }
    results = {name: measure(q, h, h_next, p, g, cut, reference[:2], exact_physical, exact_transport)
               for name, (p, g) in variants.items()}
    np.testing.assert_allclose(results["timed_both"]["thin_state"], reference[:2], atol=2e-14, rtol=0)
    np.testing.assert_allclose(results["timed_both"]["thin_entropy_loss"], shock_loss, atol=2e-14, rtol=0)
    assert results["timed_both"]["max_positive_entropy_residual"] < 1e-13
    max_cfl_for_face_fraction = [float(max(dt*(cfl[0]+a/dt*cfl[1]))) for a in arrival]
    return dict(width=width, compression=compression, motion=motion, dt=dt,
                collision=problem.collision, arrival=arrival.tolist(), beta=beta,
                max_cfl_for_face_fraction=max_cfl_for_face_fraction,
                reference_thin_state=reference[:2].tolist(),
                reference_shock_loss=float(shock_loss), reference_thin_energy=float(h_next[cut]*reference[2]),
                variants=results)


def main():
    cases = [check(compression, motion) for compression in (.65, .8)
             for motion in ("fixed", "drift", "squeeze")]
    Path("dod_first_step_timing.json").write_text(json.dumps(cases, indent=2) + "\n")
    for case in cases:
        print(case["compression"], case["motion"], "stretch",
              [round(v["thin_state"][0], 6) for v in case["variants"].values()],
              "entropy defect", [f"{v['max_positive_entropy_residual']:.2g}"
                                 for v in case["variants"].values()])


if __name__ == "__main__":
    main()
