"""Carry the thin-cell half-state difference between conservative 1D updates."""
import math

import numpy as np

from dod_one_interaction import predict_patch
from nonlinear_cutcell_predictor import predict


def extra_energy(mean, contrast, width):
    # Mean half energy minus energy at the cell average, evaluated without
    # subtracting nearly equal energies.
    ratio = contrast[0]/mean[0]
    return .5*width*(contrast[1]**2-math.log1p(-ratio**2))


class ThinHistory:
    def __init__(self, keep_wave=False):
        self.contrast = np.zeros(2)
        self.predicted_halves = None
        self.steps = []
        self.keep_wave = keep_wave
        self.wave = None

    def energy(self, mean, width):
        return extra_energy(mean, self.contrast, width)

    def predict(self, states, width, speeds, dt):
        self.predicted_halves = None
        self.wave = None
        starts = np.array([-width/2, width/2])
        if not self.contrast.any():
            status, physical, total, halves = predict_patch(states, width, speeds, dt, keep_halves=True)
            if total is not None:
                self.predicted_halves = halves
                return status, physical, total
            breaks = starts
        else:
            left, mean, right = states
            states = (left, mean-self.contrast, mean+self.contrast, right)
            breaks = np.linspace(*starts, 3)
        result = predict(list(map(tuple, states)), starts, speeds, dt, fan_limit=1,
                         event_budget=4, faces_only=True, initial_breaks=breaks,
                         sample_edges=np.linspace(*(starts+dt*speeds), 3), keep_wave=self.keep_wave)
        if result["status"] != "ok":
            return "fan_budget", None, None
        self.predicted_halves = result["means"]
        self.wave = result.get("wave")
        flux = result["face_fluxes"]/dt
        status = "bounded_halves" if self.contrast.any() else "bounded_fans"
        return status, flux[:, [5, 3, 4]], flux[:, :3]

    def commit(self, mean, width, budget, weight):
        old = self.contrast
        self.contrast = np.zeros(2)
        scale = 0.
        if weight > 0. and self.predicted_halves is not None:
            left, right = self.predicted_halves
            proposed = .5*weight*(right-left)
            scale = min(1., (1.-1e-12)*mean[0]/abs(proposed[0])) if proposed[0] != 0 else 1.
            available = max(0., budget)
            if extra_energy(mean, scale*proposed, width) > available:
                low, high = 0., scale
                for _ in range(16):
                    mid = .5*(low+high)
                    if extra_energy(mean, mid*proposed, width) <= available:
                        low = mid
                    else:
                        high = mid
                scale = low
            self.contrast = scale*proposed
            if (abs(self.contrast[0]) <= 1e-12*mean[0]
                    and abs(self.contrast[1]) <= 1e-12*max(1., abs(mean[1]))):
                self.contrast.fill(0.)
        halves = np.array([mean-self.contrast, mean+self.contrast])
        assert np.min(halves[:, 0]) > 0
        assert np.all(abs(halves.mean(axis=0)-mean) <= 2e-14*(1.+abs(mean)))
        retained = self.energy(mean, width)
        self.steps.append(dict(used_halves=bool(old.any()), contrast=self.contrast.tolist(),
                               shape_scale=scale, extra_energy=retained))
        return retained
