"""Interior quadrature of moving-face work along an RK2 predicted path."""
import math
import torch

from free_surface.fv_ale_planar_boundary import geometric_rate, values
from free_surface.fv_ale_solid import State


def correct(solver, start, dt, first, second, p2, r2, q2, planes2):
    """Replace the moving-face integrals using paired force, strain and volume rates.

    The RK2 endpoint predicts the sample states. The corrected plane controls,
    material update and work account use the same quadrature. The final geometry
    is cut from the corrected controls and checked by the usual step audit.
    """
    planes0 = start.mesh.planes
    nodes = (0.5,) if solver.moving_face_midpoint else (0.5 - math.sqrt(3.) / 6., 0.5 + math.sqrt(3.) / 6.)
    samples = []
    for t in nodes:
        p = start.p.lerp(p2, t)
        r = start.r.lerp(r2, t)
        q = start.q.lerp(q2, t)
        planes = planes0.clone()
        planes.a = planes0.a.lerp(planes2.a, t)
        planes.b = planes0.b.lerp(planes2.b, t)
        mesh = solver.cut(p, r, planes)
        sample = State(p, r, solver.plastic_return(q, mesh), mesh)
        samples.append(solver.device_ops().moving_face_sample(sample))

    def integrate(values):
        return (dt / len(nodes)) * sum(values[1:], values[0])

    q_boundary, boundary_volume, boundary_work, a, b = (integrate(values) for values in zip(*samples))
    start_rate, end_rate = first.boundary, second.boundary
    factor = 0.5 * dt
    corrected_q = q2 + (q_boundary - factor * (start_rate[0] + end_rate[0]))
    planes2.a = planes0.a + a
    planes2.b = planes0.b + b

    integrated_volume = start.mesh.volume + factor * (
        start_rate[3] + end_rate[3] - start_rate[1] - end_rate[1]) + boundary_volume
    work = factor * (first[3]['mechanical_rate_chain_W'] + second[3]['mechanical_rate_chain_W'])
    work += float(boundary_work - factor * (start_rate[2] + end_rate[2]))
    diagnostics = {
        'moving_face_quadrature_nodes': len(nodes),
        'moving_face_swept_volume_m3': float(boundary_volume.sum()),
        'moving_face_work_joule': float(boundary_work),
        'moving_face_energy_exchange_joule': float(q_boundary[:, 4].sum()),
    }
    return corrected_q, planes2, (work, integrated_volume, diagnostics)


def correct_endpoints(solver, start, stage, end, dt, first, second):
    """Correct the normal moving-face terms with the RK2 endpoint secant.

    The predicted geometry, tangential contribution and momentum update stay
    fixed. The swept correction changes both the log-shape update and the
    stress work. Total energy stays conserved and the correction is limited
    per plane by the original integrated acoustic dissipation.
    """
    initial_planes, final_planes = start.mesh.planes, end.mesh.planes
    dp, dr = (end.p - start.p) / dt, (end.r - start.r) / dt
    da = (final_planes.a - initial_planes.a) / dt
    db = (final_planes.b - initial_planes.b) / dt
    factor = 0.5 * dt
    old_speed0, heat0, vdot0 = first.boundary
    old_speed1, heat1, vdot1 = second.boundary
    groups = solver.groups
    ops = solver.device_ops()

    def path_speed(state, forward):
        mesh = state.mesh
        b = {'p': state.p, 'r': state.r, **values(mesh)}
        speed = geometric_rate(b, dp, dr, da, db)
        planes = mesh.planes
        on_floor = (planes.a[:, 0] == 0) & (planes.a[:, 1] == -1) & (
            planes.a[:, 2] == 0) & (planes.b == -solver.ground)
        on_floor = on_floor[mesh.plane_j] & (mesh.plane_i == planes.owner[mesh.plane_j])
        # At coincident floor faces use the derivative from inside the step.
        blocked = (speed > 0) if forward else (speed < 0)
        return torch.where(on_floor & blocked, 0., speed)

    samples = (
        (start, path_speed(start, True) - old_speed0),
        (stage, -old_speed1),
        (end, path_speed(end, False)),
    )
    plane_work = torch.zeros_like(heat0)
    face_data = []
    for state, speed_delta in samples:
        mesh = state.mesh
        total, volume, _, _, _, _, sigma, _, valid = ops.primitive(state.q, mesh.volume, groups)
        if not bool(valid):
            raise RuntimeError('moving-face endpoint correction encountered an invalid material state')
        i, j, normal = mesh.plane_i, mesh.plane_j, mesh.plane_normal
        dv = factor * mesh.plane_area * speed_delta
        normal_stress = (normal * (sigma[i] @ normal[:, :, None])[:, :, 0]).sum(1)
        plane_work.index_add_(0, j, normal_stress * dv)
        face_data.append((state, total, volume, dv))

    heat = factor * (heat0 + heat1)
    keep = torch.where(plane_work > heat, heat / plane_work.clamp_min(1e-300), 1.)
    delta_volume = torch.zeros_like(start.r)
    delta_shape = torch.zeros_like(start.q[:, 5:14])
    for state, total, volume, dv in face_data:
        mesh = state.mesh
        i, j, normal = mesh.plane_i, mesh.plane_j, mesh.plane_normal
        dv = dv * keep[j]
        delta_volume.index_add_(0, i, dv)
        tensors = dv[:, None, None] * normal[:, :, None] * normal[:, None, :]
        gradient = state.p.new_zeros((len(state.p), 3, 3)).index_add_(0, i, tensors)
        gradient = ops.group(gradient, groups)[groups] / volume[:, None, None]
        mass = total[:, 0] * mesh.volume / volume
        delta_shape += mass[:, None] * ops.material_rate(gradient, None, total, mass).reshape(-1, 9)

    corrected_q = end.q.clone()
    corrected_q[:, 5:14] += delta_shape
    work_delta = (plane_work * keep).sum()
    work = factor * (first[3]['mechanical_rate_chain_W'] + second[3]['mechanical_rate_chain_W'])
    integrated_volume = start.mesh.volume + factor * (vdot0 + vdot1) + delta_volume
    diagnostics = {
        'moving_face_endpoint_correction': True,
        'moving_face_work_correction_joule': float(work_delta),
        'moving_face_limited_planes': int((keep < 1).sum()),
        'moving_face_unlimited_work_correction_joule': float(plane_work.sum()),
        'moving_face_volume_correction_m3': float(delta_volume.abs().max()),
    }
    return corrected_q, (work + float(work_delta), integrated_volume, diagnostics)
