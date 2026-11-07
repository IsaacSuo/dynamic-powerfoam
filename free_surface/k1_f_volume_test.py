"""Opt-in diagnostic in the complete k1 driver, NOT a consistent constitutive law.

Observe: unchanged evolution, stop before the first non-positive endpoint map.
Match-l: additionally rescale positive returned elastic F to exp(L), preserving
its isochoric part. This post-step intervention does NOT modify the force or
plastic-work formula. Its extra stored-energy change is explicitly recorded.
Never normalizes an inverted trial or passes it to the logarithmic return.
"""
import json
import torch


class FVolumeTest:
    def __init__(self, W, mode, path):
        self.W, self.mode, self.path = W, mode, path
        self.rows = []
        self.energy_added = 0.
        self.energy_absolute = 0.

    def emit(self, row):
        self.rows.append(row)
        with open(self.path, 'w') as f:
            json.dump(dict(mode=self.mode, energy_added=self.energy_added,
                           energy_absolute=self.energy_absolute, rows=self.rows), f, indent=2)
        if row.get('phase') != 'end' or row['step'] % 20 == 0:
            print('FVTEST ' + json.dumps(row), flush=True)

    def state(self, step, phase):
        W = self.W
        F, L = W._Fe, W.tul['Ln']
        sg, ld = torch.linalg.slogdet(F)
        if not bool(((sg > 0) & torch.isfinite(ld) & torch.isfinite(L)).all()):
            raise RuntimeError('FVTEST: invalid input F/L; refusing to repair an inversion')
        drift = ld-L
        row = dict(step=step, phase=phase, before_min=float(drift.min()),
                   before_max=float(drift.max()), before_mean_abs=float(drift.abs().mean()))
        if self.mode == 'match-l':
            factor = torch.exp((L-ld)/3)
            out = F*factor[:, None, None]
            # For det F > 0, the eigenvalues of the proper polar stretch are
            # the singular values; isotropic scaling multiplies the dev energy by factor^2.
            sv = torch.linalg.svdvals(F)
            e_cell = W.V0*W.mu*((sv-sv.mean(1, keepdim=True))**2).sum(1)
            delta = e_cell*(factor**2-1)
            W._Fe = out.contiguous()
            sg1, ld1 = torch.linalg.slogdet(out)
            iso0 = F*torch.exp(-ld/3)[:, None, None]
            iso1 = out*torch.exp(-ld1/3)[:, None, None]
            iso_err = float((iso0-iso1).abs().max()/iso0.abs().max())
            err = float((ld1-L).abs().max())
            if not bool((sg1 > 0).all()) or err > 1e-9 or iso_err > 1e-10:
                raise RuntimeError('FVTEST: volume/shape invariant failed')
            self.energy_added += float(delta.sum())
            self.energy_absolute += float(delta.abs().sum())
            row.update(scale_min=float(factor.min()), scale_max=float(factor.max()),
                       after_max_abs=err, isochoric_relative_error=iso_err,
                       extra_shape_energy=float(delta.sum()), extra_shape_energy_abs=float(delta.abs().sum()))
        else:
            row['after_max_abs'] = float(drift.abs().max())
        row['F_sv_min'] = float(torch.linalg.svdvals(W._Fe)[:, -1].min())
        self.emit(row)

    def trial(self, ST, step, products, iterations, g_relative):
        W = self.W
        F = (W.tFc-(1-ST.th)*W.tsp['F0'])/ST.th
        H = W.I3+W.tG/ST.th
        detH, detF = torch.linalg.det(H), torch.linalg.det(F)
        bad = (~torch.isfinite(F).all(dim=(1, 2))) | (~torch.isfinite(detH)) | (detH <= 0) | (detF <= 0)
        row = dict(step=step, phase='trial', products=products, iterations=iterations,
                   g_relative=g_relative, min_det_increment=float(detH.min()),
                   min_det_F=float(detF.min()), flipped=int(bad.sum()))
        if bool(bad.any()):
            ids = bad.nonzero().flatten()
            row['cells'] = ids.tolist()
            row['V0_over_median'] = (W.V0[ids]/W.V0.median()).tolist()
            row['trial_singular_values'] = torch.linalg.svdvals(F[ids]).tolist()
            row['L_next'] = (W.tul['Ln']+(W.tul['Q']*W.tG).sum((1, 2))/ST.th)[ids].tolist()
            torch.save(dict(step=step, cells=ids.cpu(), Ftrial=F.cpu(), H=H.cpu(),
                            F0=W.tsp['F0'].cpu(), L0=W.tul['Ln'].cpu(),
                            C=W.tul['Q'].cpu(), G=W.tG.cpu()), str(self.path)+'.trial.pt')
        self.emit(row)
        return not bool(bad.any())

    def finish(self, step, failed):
        self.emit(dict(step=step, phase='finished', failed=bool(failed)))
