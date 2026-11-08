"""Low-node GCL projection on complete production paths; diagnostic only.

Correct interior shared-face swept volumes with a weighted graph solve. Keep
the free cap and floor fixed. Each connected component's compatibility defect
is reported, NOT concealed: interior corrections cannot change its net volume.
An initial-volume-weighted allocation of that uncorrectable defect is not committed.
Positive/negative material transport is reintegrated at the SAME coarse nodes
after distributing each face correction in proportion to instantaneous area.
No mass, energy, state, surface motion or production solver is modified.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import coo_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve

from free_surface.fv_ale_profile import load
from free_surface.fv_ale_solid import State
from free_surface.fv_ale_swept_check import PathAudit, endpoint, cpu
from free_surface.fv_ale_swept_local import LocalPath, errors, aligned


def project(audit, coarse):
    """Minimum sum(delta_W**2 / conductance), with cap flux held fixed.

    B delta_W = e is solvable iff sum(e)=0 on every component. Project only
    the compatible part: subtract component mean weighted by initial volume.
    The remaining original GCL residual must be reported and tested separately.
    This subtraction defines the diagnostic projection, not a boundary fix.
    """
    tick = time.perf_counter()
    a = audit
    keys = coarse['keys']; i, j = keys // a.n, keys % a.n
    gi, gj = a.g[i], a.g[j]
    external = (gi != gj) & (coarse['area'] > 0)
    ii, jj = gi[external], gj[external]
    columns = np.arange(len(ii))
    B = coo_matrix((np.r_[np.ones(len(ii)), -np.ones(len(ii))],
                    (np.r_[ii, jj], np.r_[columns, columns])),
                   shape=(a.ng, len(ii))).tocsr()
    midpoint = cpu((a.start.p + a.end.p) * .5)
    distance = np.linalg.norm(midpoint[j[external]] - midpoint[i[external]], axis=1)
    if np.any(distance <= 0):
        raise RuntimeError('coincident site centers in face weighting')
    weights = coarse['area'][external] / distance
    L = (B @ diags(weights) @ B.T).tocsr()
    count, labels = connected_components(L, directed=False)
    target = a.gv1 - a.gv0
    old_rate = B @ coarse['flux'][external] + a.group(coarse['cap'])
    defect = target - old_rate
    component_defect = np.bincount(labels, weights=defect, minlength=count)
    component_volume = np.bincount(labels, weights=a.gv0, minlength=count)
    incompatible = component_defect[labels] * a.gv0 / component_volume[labels]
    compatible = defect - incompatible
    roots = np.unique(labels, return_index=True)[1]
    keep = np.ones(a.ng, dtype=bool); keep[roots] = False
    potential = np.zeros(a.ng)
    if keep.any():
        potential[keep] = spsolve(L[keep][:, keep], compatible[keep])
    delta = np.zeros(len(keys))
    delta[external] = weights * (B.T @ potential)
    projected = {name: value.copy() for name, value in coarse.items()}
    projected['flux'] += delta
    solve_error = B @ delta[external] - compatible
    final_defect = target - (B @ projected['flux'][external] + a.group(coarse['cap']))
    residual = np.abs(final_defect) / a.gv0
    # A velocity-sized quantity for interpreting the missing boundary motion,
    # not a proposal to spread the error over the free surface.
    cap_area = np.bincount(labels, weights=a.group(coarse['cap_area']), minlength=count)
    speed = np.divide(component_defect, a.dt * cap_area,
                      out=np.full(count, np.nan), where=cap_area > 0)
    report = {
        'projection_seconds': time.perf_counter() - tick,
        'components': int(count),
        'component_volume_defect_m3': component_defect.tolist(),
        'component_relative_volume_defect': (component_defect / component_volume).tolist(),
        'equivalent_uniform_cap_speed_defect_m_per_s': speed.tolist(),
        'compatible_solve_max_relative_error': float(np.max(np.abs(solve_error) / a.gv0)),
        'uncorrectable_max_group_relative_defect': float(residual.max()),
        'cap_and_floor_unchanged': bool(np.array_equal(projected['cap'], coarse['cap'])
                                      and np.array_equal(projected['floor'], coarse['floor'])),
        'correction_max_face_scaled': float(np.max(np.abs(delta[external]) /
                                             np.minimum(a.gv0[ii], a.gv0[jj]), initial=0)),
        'correction_l1_m3': float(np.abs(delta).sum()),
        'strict_gcl_at_1e_10': bool(residual.max() < 1e-10),
    }
    assert report['compatible_solve_max_relative_error'] < 1e-9, report
    assert report['cap_and_floor_unchanged']
    assert np.allclose(final_defect, incompatible, rtol=0, atol=1e-9 * a.gv0.min())
    return projected, delta, report


def transport(audit, local, coarse, projected, delta, panels):
    """Re-split donor transport after correction; never correct net-only flux."""
    weights = np.ones(panels + 1)
    weights[1:-1:2] = 4; weights[2:-1:2] = 2; weights /= 3 * panels
    result = {key: val.copy() for key, val in projected.items()}
    result['positive'].fill(0); result['negative'].fill(0)
    for k, weight in enumerate(weights):
        sample = local.sample(k / panels)
        ix = np.searchsorted(coarse['keys'], sample['keys'])
        shift = delta[ix] * sample['area'] / coarse['area'][ix]
        relative = sample['positive'] + sample['negative'] - shift
        result['positive'][ix] += weight * np.maximum(relative, 0)
        result['negative'][ix] += weight * np.minimum(relative, 0)
    net_difference = (result['positive'] + result['negative'] -
                      coarse['positive'] - coarse['negative'] + delta)
    assert np.max(np.abs(net_difference), initial=0) < 1e-15
    return result


def transport_error(a, result, reference):
    """Frozen-end-state donor transfer error, NOT a coupled material update."""
    union = np.union1d(result['keys'], reference['keys'])
    i, j = union // a.n, union % a.n
    ext = a.g[i] != a.g[j]
    rho = cpu(a.rho)
    plus = aligned(result, reference, 'positive')[1]
    minus = aligned(result, reference, 'negative')[1]
    face_mass = plus * rho[i] + minus * rho[j]
    cell_mass = np.bincount(i[ext], weights=-face_mass[ext], minlength=a.n)
    cell_mass += np.bincount(j[ext], weights=face_mass[ext], minlength=a.n)
    group_mass = a.group(cpu(a.end.q[:, 0]))
    report = {
        'frozen_donor_max_group_mass_error_relative': float(np.max(np.abs(a.group(cell_mass)) / group_mass)),
        'frozen_donor_face_mass_error_l1_kg': float(np.abs(face_mass[ext]).sum()),
        'internal_mass_cancellation_kg': float(cell_mass.sum()),
    }
    return report


def analytic_controls(sol, start, reference_volume):
    records = []
    for kind in ['horizontal_translation', 'homothety_about_floor']:
        if kind == 'horizontal_translation':
            dp = torch.zeros_like(start.p); dp[:, 0] = .001
            dr = torch.zeros_like(start.r); scale = 1.
        else:
            center = torch.tensor([0., sol.ground, 0.], device=start.p.device, dtype=start.p.dtype)
            scale = 1.002; dp = (scale - 1) * (start.p - center); dr = (scale - 1) * start.r
        mesh = sol.cut(start.p + dp, start.r + dr)
        end = State(start.p + dp, start.r + dr, start.q.clone(), mesh)
        a = PathAudit(sol, start, end, 800e-6, reference_volume)
        coarse = a.integrate(2)
        corrected, delta, info = project(a, coarse)
        exact = (scale**3 - 1) * a.gv0
        cut_error = float(np.max(np.abs(a.gv1 - a.gv0 - exact) / a.gv0))
        assert cut_error < 1e-9, cut_error
        assert info['correction_max_face_scaled'] < 1e-9, info
        assert info['strict_gcl_at_1e_10'], info
        records.append({'motion': kind, 'actual_volume_error_relative': cut_error, **info})
    return records


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tags', nargs='+', default=['dt8us', 'cfl20_long'])
    ap.add_argument('--nodes', type=int, nargs='+', default=[3, 5, 9])
    ap.add_argument('--repeat', type=int, default=2)
    ap.add_argument('--controls', action='store_true')
    ap.add_argument('--output', default='swept_gcl_trial.json')
    args = ap.parse_args()
    out = Path('output/ale'); records = []
    for tag in args.tags:
        sol, reset, data = load(out / f'fv_state_{tag}.pt')
        start = reset(); end, explicit_cost = endpoint(sol, start, 800e-6)
        if args.controls:
            control = {'snapshot': tag, 'controls': analytic_controls(sol, start, data['reference_volume'])}
            records.append(control); print(json.dumps(control), flush=True)
        # Fine reference is used only for reporting after the correction.
        for rep in range(args.repeat):
            for nodes in args.nodes if rep % 2 == 0 else list(reversed(args.nodes)):
                panels = nodes - 1
                if panels < 2 or panels % 2: raise ValueError('Simpson requires even panels')
                a = PathAudit(sol, start, end, 800e-6, data['reference_volume'])
                a.sample(0); a.sample(1)
                torch.cuda.synchronize(); tick = time.perf_counter()
                local = LocalPath(a, np.arange(a.n), batch=4)
                coarse = local.integrate(panels)
                corrected, delta, info = project(a, coarse)
                corrected = transport(a, local, coarse, corrected, delta, panels)
                torch.cuda.synchronize(); elapsed = time.perf_counter() - tick
                reference = dict(np.load(out / f'swept_{tag}_800us.npz'))
                # Oracle boundary ablation ONLY. Excluded from production timing;
                # it cannot be used as a claimed cheap algorithm or topology fix.
                boundary_oracle = {key: val.copy() for key, val in coarse.items()}
                boundary_oracle['cap'] = reference['cap'].copy()
                oracle, oracle_delta, oracle_info = project(a, boundary_oracle)
                oracle = transport(a, local, boundary_oracle, oracle, oracle_delta, panels)
                row = {'snapshot': tag, 'repeat': rep, 'nodes': nodes,
                       'seconds_excluding_cached_endpoints': elapsed,
                       'geometry_seconds': local.cut_seconds, 'explicit_reference': explicit_cost,
                       **info,
                       'before': {**errors(a, coarse, reference), **transport_error(a, coarse, reference)},
                       'after': {**errors(a, corrected, reference), **transport_error(a, corrected, reference)},
                       'fine_cap_oracle_only': {'not_a_production_method': True, **oracle_info,
                            'errors': {**errors(a, oracle, reference), **transport_error(a, oracle, reference)}},
                       'full_implicit_step': False, 'boundary_motion_corrected': False,
                       'topology_certified': False}
                records.append(row); print(json.dumps(row), flush=True)
                (out / args.output).write_text(json.dumps(records, indent=2) + '\n')


if __name__ == '__main__':
    main()
