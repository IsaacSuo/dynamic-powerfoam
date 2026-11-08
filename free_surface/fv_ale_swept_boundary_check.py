"""Full-foam path/free-cap block solve, not a complete implicit material step.

The time-integrated production acoustic cap equations determine translations
and radius rates. Geometry depends on that same motion. Test prescribed end
primitives first, then EOS pressure feedback at fixed candidate group Q.
The latter is a nonlinear block of a coupled solve, NOT material conservation
over a step: momentum, energy, stress and contact are not advanced here.
No state is committed. All candidate cuts and nonlinear retries are timed.
"""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.sparse import coo_matrix, eye, diags
from scipy.sparse.linalg import spsolve

from free_surface.fv_ale_profile import load
from free_surface.fv_ale_solid import State
from free_surface.fv_ale_boundary import solve_boundary
from free_surface.fv_ale_swept_check import PathAudit, endpoint, cpu
from free_surface.fv_ale_swept_local import LocalPath, errors
from free_surface.fv_ale_swept_gcl_check import project


class BoundaryPath:
    def __init__(self, sol, start, candidate, dt, nodes, rv, mode, flat_mode='production'):
        self.sol, self.start, self.candidate = sol, start, candidate
        self.dt, self.nodes, self.rv, self.mode = dt, nodes, rv, mode
        self.reference_primitive = sol.primitive(candidate)
        self.scale = cpu(start.r)[:, None]
        self.evaluations = 0
        self.flat_mode = flat_mode
        initial_n = start.mesh.cap_normal_integral / start.mesh.cap_area.clamp(min=1e-300)[:, None]
        self.initial_radial = start.mesh.surface & (1 - initial_n.square().sum(1) > 1e-3)

    def cap_mesh(self, integral):
        tensor = lambda name: torch.as_tensor(integral[name], device=self.start.p.device,
                                              dtype=self.start.p.dtype)
        area, N, Q = tensor('cap_area'), tensor('cap_n'), tensor('cap_Q')
        area_scale = max(float(area.max()), float(np.max(integral['area'], initial=0)))
        return SimpleNamespace(cap_area=area, cap_normal_integral=N, cap_Q=Q,
                               surface=area > 1e-12 * area_scale, contact_quad=None)

    def solve_cap(self, mesh, primitive):
        _, v, rho, pressure, stress, sigma, sound = primitive
        if self.flat_mode == 'covariance':
            Z = rho * sound; A = mesh.cap_area.clamp(min=1e-300)
            n = mesh.cap_normal_integral / A[:, None]
            covariance = mesh.cap_Q / A[:, None, None] - n[:, :, None] * n[:, None, :]
            variance = covariance.diagonal(dim1=1, dim2=2).sum(1)
            # Algebraically the same 4-DOF acoustic solve away from degeneracy:
            # dr=(P-s:Cov/trace(Cov))/Z. Isotropic pressure does not need a
            # cancellation of two almost equal numbers or an on/off switch.
            # Below 1e-8 only the deviatoric ratio is regularized; report it.
            K = covariance / variance.clamp(min=1e-8)[:, None, None]
            dr = torch.where(mesh.surface, (pressure - (stress * K).sum((1, 2))) / Z, 0.)
            f = -(sigma @ n[:, :, None])[:, :, 0] / Z[:, None]
            w = torch.where(mesh.surface[:, None], v + f - dr[:, None] * n, v)
            return w, dr
        if self.flat_mode == 'frozen':
            Z = rho * sound; A = mesh.cap_area.clamp(min=1e-300)
            n = mesh.cap_normal_integral / A[:, None]
            f = -(sigma @ n[:, :, None])[:, :, 0] / Z[:, None]
            b = -(sigma * mesh.cap_Q).sum((1, 2)) / (Z * A)
            variance = 1 - n.square().sum(1)
            active = mesh.surface & self.initial_radial
            if bool((active & (variance <= 1e-8)).any()):
                raise RuntimeError('frozen radial branch became geometrically singular')
            dr = torch.where(active, (b - (f * n).sum(1)) / variance.clamp(min=1e-300), 0.)
            w = torch.where(mesh.surface[:, None], v + f - dr[:, None] * n, v)
            return w, dr
        return solve_boundary(mesh, v, sigma, rho * sound, self.sol.contact_k,
                              self.sol.friction, self.sol.contact_damping)

    def radial_mask(self, mesh, variance):
        if self.flat_mode == 'covariance': return mesh.surface
        return mesh.surface & (self.initial_radial if self.flat_mode == 'frozen' else variance > 1e-3)

    def weak_residual(self, x, integral, primitive):
        mesh = self.cap_mesh(integral)
        _, v, rho, _, _, sigma, sound = primitive
        Z = rho * sound
        motion = torch.as_tensor(x * self.scale, device=v.device, dtype=v.dtype)
        dp, dr = motion[:, :3], motion[:, 3]
        A, N, Q = mesh.cap_area, mesh.cap_normal_integral, mesh.cap_Q
        dv = dp - self.dt * v
        vector = A[:, None] * dv + N * dr[:, None] + self.dt * (sigma @ N[:, :, None])[:, :, 0] / Z[:, None]
        radial = (dv * N).sum(1) + A * dr + self.dt * (sigma * Q).sum((1, 2)) / Z
        variance = 1 - (N / A.clamp(min=1e-300)[:, None]).square().sum(1)
        active = self.radial_mask(mesh, variance)
        scale = self.start.r * A.clamp(min=1e-300)
        rows = torch.cat((vector / scale[:, None], (radial / scale)[:, None]), 1)
        rows[~mesh.surface] = 0
        rows[~active, 3] = 0  # Production flat-cap control has translation only.
        return {'weak_cap_residual_max_scaled': float(rows.abs().max()),
                'translation_only_caps': int((mesh.surface & ~active).sum()),
                'cap_cells': int(mesh.surface.sum())}

    def evaluate(self, x):
        self.evaluations += 1
        motion = torch.as_tensor(x * self.scale, device=self.start.p.device, dtype=self.start.p.dtype)
        p, r = self.start.p + motion[:, :3], self.start.r + motion[:, 3]
        mesh = self.sol.cut(p, r)
        # Keep group conserved candidate totals fixed. Repartition only within a
        # group; no mass or energy is regenerated from the newly cut volumes.
        q = self.sol.distribute_groups(self.candidate.q, mesh.volume)
        end = State(p, r, q, mesh)
        actual = self.sol.primitive(end)
        primitive = self.reference_primitive if self.mode == 'fixed_primitives' else actual
        a = PathAudit(self.sol, self.start, end, self.dt, self.rv)
        _, a.v, a.rho, _, _, a.sigma, sound = primitive
        a.Z = a.rho * sound
        a.sample(0); a.sample(1)
        local = LocalPath(a, np.arange(a.n), batch=4)
        integral = local.integrate(self.nodes - 1)
        w, dr, *_ = self.solve_cap(self.cap_mesh(integral), primitive)
        target = self.dt * np.c_[cpu(w), cpu(dr)] / self.scale
        residual = target - x
        info = {'residual_max_scaled_by_radius': float(np.abs(residual).max()),
                'residual_l2': float(np.linalg.norm(residual)),
                'pressure_min_Pa': float(primitive[3].min()),
                'pressure_max_Pa': float(primitive[3].max()),
                **self.weak_residual(x, integral, primitive)}
        return target, info, (a, local, integral, primitive, end)

    def eos_response(self, integral, primitive):
        """Exact cap-motion derivative for dV/V, holding cap moments and Q fixed."""
        _, v, rho, pressure, stress, sigma, sound = primitive
        mesh = self.cap_mesh(integral)
        A, N, Q = mesh.cap_area, mesh.cap_normal_integral, mesh.cap_Q
        Z = rho * sound
        mean_n = N / A.clamp(min=1e-300)[:, None]
        variance = 1 - mean_n.square().sum(1)
        active = self.radial_mask(mesh, variance)
        dP = -(pressure + self.sol.gamma * self.sol.pinf)
        dlogZ = .5 * (-1 + self.sol.gamma * dP /
                     (self.sol.gamma * (pressure + self.sol.pinf) + 4 * self.sol.mu / 3))
        I = torch.eye(3, device=v.device, dtype=v.dtype)
        dS = (-dP[:, None, None] * I - sigma * dlogZ[:, None, None]) / Z[:, None, None]
        df = -(dS @ mean_n[:, :, None])[:, :, 0]
        db = -(dS * Q).sum((1, 2)) / A.clamp(min=1e-300)
        ddr = torch.where(active, (db - (df * mean_n).sum(1)) / variance.clamp(min=1e-300), 0.)
        if self.flat_mode == 'covariance':
            covariance = Q / A.clamp(min=1e-300)[:, None, None] - mean_n[:, :, None] * mean_n[:, None, :]
            trace = covariance.diagonal(dim1=1, dim2=2).sum(1)
            K = covariance / trace.clamp(min=1e-8)[:, None, None]
            radial = (pressure - (stress * K).sum((1, 2))) / Z
            ddr = torch.where(active, dP / Z - radial * dlogZ, 0.)
        dw = torch.where(mesh.surface[:, None], df - ddr[:, None] * mean_n, 0.)
        return self.dt * np.c_[cpu(dw), cpu(ddr)]

    def eos_step(self, x, target, pack):
        """Schur solve for pressure/volume feedback; cap-geometry derivatives lagged.

        dx = b + T_V dz, dz = V_x dx / V. Thus (I - V_x T_V/V) dz=V_x b/V.
        V_x is the exact instantaneous derivative of the actual endpoint cut.
        No new cuts or finite-difference products are needed for this matrix.
        """
        tick = time.perf_counter()
        a, local, integral, primitive, end = pack
        response = self.eos_response(integral, primitive)
        m = end.mesh; i, j = cpu(m.i), cpu(m.j); g = a.g
        p, r, c = cpu(end.p), cpu(end.r), cpu(m.centroid)
        area = cpu(m.area); N = cpu(m.cap_normal_integral); cap = cpu(m.cap_area)
        length = np.linalg.norm(p[j] - p[i], axis=1)
        ai = area / length * (np.sum((c - p[i]) * response[i, :3], axis=1) + r[i] * response[i, 3])
        aj = -area / length * (np.sum((c - p[j]) * response[j, :3], axis=1) + r[j] * response[j, 3])
        ac = np.sum(N * response[:, :3], axis=1) + cap * response[:, 3]
        C = coo_matrix((np.r_[ai, aj, -ai, -aj, ac],
                         (np.r_[g[i], g[i], g[j], g[j], g],
                          np.r_[g[i], g[j], g[i], g[j], g])), shape=(a.ng, a.ng)).tocsr()
        C = diags(1 / a.gv1) @ C
        b = (target - x) * self.scale
        wall = area / length * (np.sum((c - p[i]) * b[i, :3] - (c - p[j]) * b[j, :3], axis=1)
                                + r[i] * b[i, 3] - r[j] * b[j, 3])
        cap_b = np.sum(N * b[:, :3], axis=1) + cap * b[:, 3]
        rhs = (np.bincount(g[i], weights=wall, minlength=a.ng) -
               np.bincount(g[j], weights=wall, minlength=a.ng) + a.group(cap_b)) / a.gv1
        matrix = eye(a.ng, format='csr') - C
        dz = spsolve(matrix, rhs)
        step = (b + response * dz[g, None]) / self.scale
        linear = np.linalg.norm(matrix @ dz - rhs) / max(np.linalg.norm(rhs), 1e-300)
        if not np.isfinite(step).all() or linear > 1e-8:
            raise RuntimeError(f'volume-pressure Schur solve residual {linear}')
        return step, {'pressure_block_seconds': time.perf_counter() - tick,
                      'pressure_block_linear_relative_residual': float(linear)}

    def derivative_check(self, pack):
        a, local, integral, primitive, end = pack
        response = self.eos_response(integral, primitive)
        direction = np.sin(np.arange(a.ng) * 1.234)[a.g]
        ds = torch.as_tensor(direction, device=end.p.device, dtype=end.p.dtype)
        _, v, rho, P, stress, sigma, sound = primitive
        outputs = []
        eps = 1e-6
        for sign in [-1, 1]:
            new_rho = rho / (1 + sign * eps * ds)
            new_p = (P + self.sol.gamma * self.sol.pinf) / (1 + sign * eps * ds) - self.sol.gamma * self.sol.pinf
            I = torch.eye(3, device=v.device, dtype=v.dtype)
            new_sigma = stress - new_p[:, None, None] * I
            new_sound = ((self.sol.gamma * (new_p + self.sol.pinf) + 4 * self.sol.mu / 3) / new_rho).sqrt()
            w, dr, *_ = self.solve_cap(self.cap_mesh(integral), (primitive[0], v, new_rho, new_p, stress, new_sigma, new_sound))
            outputs.append(self.dt * np.c_[cpu(w), cpu(dr)])
        measured = (outputs[1] - outputs[0]) / (2 * eps)
        expected = response * direction[:, None]
        error = float(np.linalg.norm(measured - expected) / np.linalg.norm(expected))
        assert error < 1e-6, error
        return error

    def formula_checks(self):
        """Algebra checks on the real full-foam cap moments, outside timings."""
        mesh = self.start.mesh; primitive = self.reference_primitive
        mode = self.flat_mode
        self.flat_mode = 'production'; wp, rp, *_ = self.solve_cap(mesh, primitive)
        self.flat_mode = 'covariance'; wc, rc = self.solve_cap(mesh, primitive)
        n = mesh.cap_normal_integral / mesh.cap_area.clamp(min=1e-300)[:, None]
        active = mesh.surface & (1 - n.square().sum(1) > 1e-3)
        discrepancy = max(float((wc[active] - wp[active]).abs().max()), float((rc[active] - rp[active]).abs().max()))
        assert discrepancy < 1e-7, discrepancy
        m, v, rho, P, stress, sigma, sound = primitive
        test_p = torch.full_like(P, 1234.)
        eye3 = torch.eye(3, dtype=v.dtype, device=v.device)
        test = (m, v, rho, test_p, torch.zeros_like(stress), -test_p[:, None, None] * eye3, sound)
        w, dr = self.solve_cap(mesh, test)
        pressure_error = float((dr[mesh.surface] - (test_p / (rho * sound))[mesh.surface]).abs().max())
        assert pressure_error < 1e-12, pressure_error
        test = (m, v, rho, torch.zeros_like(P), torch.zeros_like(stress), torch.zeros_like(sigma), sound)
        w, dr = self.solve_cap(mesh, test)
        translation_error = max(float((w - v).abs().max()), float(dr.abs().max()))
        assert translation_error < 1e-12, translation_error
        self.flat_mode = mode
        return {'production_agreement_on_nonflat_caps_max_m_per_s': discrepancy,
                'isotropic_pressure_radial_speed_max_error_m_per_s': pressure_error,
                'zero_stress_translation_max_error_m_per_s': translation_error}

    def run(self, max_iter=16, tol=1e-6):
        torch.cuda.synchronize(); tick = time.perf_counter()
        cuts = self.sol.geometry_calls
        # A production cap solve on the old geometry is the initializer; the
        # explicit end positions are not supplied as a convergence shortcut.
        _, v0, rho0, _, _, sigma0, sound0 = self.reference_primitive
        w, dr, *_ = solve_boundary(self.start.mesh, v0, sigma0, rho0 * sound0,
                                  self.sol.contact_k, self.sol.friction, self.sol.contact_damping)
        x = self.dt * np.c_[cpu(w), cpu(dr)] / self.scale
        initial_x_max = float(np.abs(x).max())
        initial_worst = int(np.argmax(np.max(np.abs(x), axis=1)))
        initial_mesh = self.start.mesh
        initial_A = float(initial_mesh.cap_area[initial_worst])
        initial_N = cpu(initial_mesh.cap_normal_integral[initial_worst]) / max(initial_A, 1e-300)
        initial_Q = cpu(initial_mesh.cap_Q[initial_worst]) / max(initial_A, 1e-300)
        hx, hf, history = [], [], []
        last = None; last_pack = None; last_x = None
        pending_step = None; retry_alpha = 1.
        converged = False
        for it in range(max_iter):
            try:
                target, info, pack = self.evaluate(x)
            except RuntimeError as exc:
                history.append({'iteration': it, 'invalid_candidate': str(exc)})
                if last is None:
                    x *= .5  # Initial guess only; dt is unchanged.
                else:
                    retry_alpha *= .5
                    x = last_x + retry_alpha * pending_step
                hx, hf = [], []
                continue
            info['iteration'] = it
            if last is not None and info['residual_l2'] > (1 - 1e-4 * retry_alpha) * last[1]['residual_l2']:
                info['rejected_accelerated_candidate'] = True; history.append(info)
                retry_alpha *= .5
                x = last_x + retry_alpha * pending_step
                hx, hf = [], []
                continue
            history.append(info)
            last, last_pack, last_x = (target, info), pack, x.copy()
            if info['residual_max_scaled_by_radius'] < tol:
                converged = True; break
            if self.mode == 'eos_feedback':
                pending_step, linear_info = self.eos_step(x, target, pack)
                info.update(linear_info)
                x = x + pending_step; retry_alpha = 1.
                continue
            f = target - x
            hx.append(x.copy()); hf.append(f.copy()); hx, hf = hx[-5:], hf[-5:]
            if len(hf) == 1:
                x = x + .5 * f
            else:
                # Type-II Anderson mixing; only vectors are extrapolated. Every
                # candidate is recut and its actual nonlinear residual checked.
                DF = np.stack([(g - hf[-1]).ravel() for g in hf[:-1]], axis=1)
                coeff = np.linalg.lstsq(DF, -hf[-1].ravel(), rcond=1e-10)[0]
                candidate = target.copy()
                for c, old_x, old_f in zip(coeff, hx[:-1], hf[:-1]):
                    candidate += c * (old_x + old_f - target)
                change = candidate - x
                limit = 3 * np.linalg.norm(f)
                if np.linalg.norm(change) > limit:
                    candidate = x + change * limit / np.linalg.norm(change)
                x = candidate
            pending_step = x - last_x; retry_alpha = 1.
        torch.cuda.synchronize()
        record = {'mode': self.mode, 'flat_mode': self.flat_mode, 'nodes': self.nodes, 'dt': self.dt,
                  'converged': converged, 'candidate_evaluations': self.evaluations,
                  'solve_seconds': time.perf_counter() - tick,
                  'endpoint_recuts': self.sol.geometry_calls - cuts,
                  'history': history, 'tolerance_scaled_by_radius': tol,
                  'initial_motion_max_scaled': initial_x_max,
                  'initial_worst_cell': initial_worst,
                  'initial_worst_cap_area': initial_A,
                  'initial_worst_covariance': (initial_Q - np.outer(initial_N, initial_N)).tolist(),
                  'full_material_implicit_solve': False, 'committed': False}
        if last_pack is None: return record
        a, local, integral, primitive, end = last_pack
        record['last_accepted_residual'] = last[1]
        residual = last[0] - last_x
        A, N = integral['cap_area'], integral['cap_n']
        variance = 1 - np.sum((N / np.maximum(A, 1e-300)[:, None])**2, axis=1)
        cap_mesh = self.cap_mesh(integral)
        cov = cap_mesh.cap_Q / cap_mesh.cap_area.clamp(min=1e-300)[:, None, None]
        mean_n = cap_mesh.cap_normal_integral / cap_mesh.cap_area.clamp(min=1e-300)[:, None]
        cov = cov - mean_n[:, :, None] * mean_n[:, None, :]
        trace_cov = cov.diagonal(dim1=1, dim2=2).sum(1)
        record['covariance_regularized_caps'] = int((cap_mesh.surface & (trace_cov < 1e-8)).sum())
        production_radial = cpu(cap_mesh.surface) & (variance > 1e-3)
        record['frozen_branch_disagrees_with_final_production_rule'] = int(
            np.count_nonzero(cpu(cap_mesh.surface) & (production_radial != cpu(self.initial_radial))))
        cells = np.argsort(np.max(np.abs(residual), axis=1))[-6:][::-1]
        record['largest_residual_cells'] = [
            {'cell': int(k), 'residual_scaled': residual[k].tolist(),
             'cap_area': float(A[k]), 'cap_variance': float(variance[k]),
             'group_volume_m3': float(a.gv1[a.g[k]]),
             'pressure_Pa': float(primitive[3][k])} for k in cells]
        if self.mode == 'eos_feedback':
            record['pressure_derivative_check_relative_error'] = self.derivative_check(last_pack)
        record['max_displacement_m'] = float(a.dp.norm(dim=1).max())
        record['max_radius_change_m'] = float(a.dr.abs().max())
        record['min_radius_m'] = float(end.r.min())
        corrected, delta, proj = project(a, integral)
        record['gcl_projection'] = proj
        record['solve_plus_projection_seconds'] = record['solve_seconds'] + proj['projection_seconds']
        # Independent geometry accuracy check, NOT part of the proposed cheap
        # solve or reported cost. The candidate material state stays the same.
        fine = local.integrate(16)
        record['independent_17_node_boundary'] = self.weak_residual(last_x, fine, primitive)
        record['coarse_vs_17_nodes'] = errors(a, integral, fine)
        record['projected_vs_17_nodes'] = errors(a, corrected, fine)
        record['candidate_group_mass_relative_change'] = float(
            (self.sol.group_sum(end.q[:, 0]) - self.sol.group_sum(self.candidate.q[:, 0])).abs().max()
            / self.sol.group_sum(self.candidate.q[:, 0]).max())
        return record


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tags', nargs='+', default=['dt8us', 'cfl20_long'])
    ap.add_argument('--nodes', type=int, nargs='+', default=[3, 5])
    ap.add_argument('--modes', nargs='+', default=['fixed_primitives', 'eos_feedback'])
    ap.add_argument('--dt', type=float, default=800e-6)
    ap.add_argument('--max-iter', type=int, default=16)
    ap.add_argument('--flat-mode', choices=['production', 'frozen', 'covariance'], default='production')
    ap.add_argument('--output', default='swept_boundary_trial.json')
    args = ap.parse_args(); out = Path('output/ale'); records = []
    for tag in args.tags:
        sol, reset, data = load(out / f'fv_state_{tag}.pt')
        start = reset(); candidate, ref = endpoint(sol, start, args.dt)
        check_block = BoundaryPath(sol, start, candidate, args.dt, args.nodes[0], data['reference_volume'], args.modes[0])
        checks = check_block.formula_checks()
        for nodes in args.nodes:
            for mode in args.modes:
                block = BoundaryPath(sol, start, candidate, args.dt, nodes, data['reference_volume'], mode, args.flat_mode)
                row = {'snapshot': tag, 'explicit_reference': ref, 'formula_checks': checks,
                       **block.run(max_iter=args.max_iter)}
                records.append(row); print(json.dumps(row), flush=True)
                (out / args.output).write_text(json.dumps(records, indent=2) + '\n')


if __name__ == '__main__': main()
