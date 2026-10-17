"""Restricted 4D quadratic Foam whose time slices are 3D power diagrams.

q_i(x,t) = |x - p_i - v_i (t-tau_i)|^2 + lambda_i (t-tau_i)^2 - r_i^2.
Different velocities imply different 4D metrics, but their spatial block is I.
This is not an ordinary Euclidean 4D power diagram or a material simulator.
Times must use the same units as velocities and temporal precision.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PowerSlice:
    ids: np.ndarray
    centers: np.ndarray
    radii: np.ndarray


class SpacetimeFoam:
    def __init__(self, centers, velocities, times, radii, temporal_precision):
        self.centers = np.asarray(centers, dtype=np.float64).copy()
        self.velocities = np.asarray(velocities, dtype=np.float64).copy()
        if self.centers.ndim != 2 or self.centers.shape[1] != 3:
            raise ValueError('Centers must have shape [N,3]')
        n = len(self.centers)
        if not n or self.velocities.shape != self.centers.shape:
            raise ValueError('Need nonempty matching centers and velocities')
        self.times, self.radii, self.temporal_precision = [
            np.broadcast_to(np.asarray(x, dtype=np.float64), (n,)).copy()
            for x in (times, radii, temporal_precision)]
        if not all(np.isfinite(x).all() for x in (
                self.centers, self.velocities, self.times, self.radii, self.temporal_precision)):
            raise ValueError('All parameters must be finite')
        if np.any(self.radii <= 0) or np.any(self.temporal_precision <= 0):
            raise ValueError('Positive radii and temporal precision required')

    @property
    def half_lifetimes(self):
        """Half width of the finite support's time interval, not opacity decay."""
        return self.radii / np.sqrt(self.temporal_precision)

    def slice(self, time):
        if not np.isfinite(time):
            raise ValueError('Time must be finite')
        dt = float(time) - self.times
        radius_squared = self.radii**2 - self.temporal_precision * dt**2
        # Zero-radius tangencies have zero ray volume and are omitted explicitly.
        ids = np.flatnonzero(radius_squared > 0)
        return PowerSlice(ids, self.centers[ids] + self.velocities[ids]*dt[ids, None],
                          np.sqrt(radius_squared[ids]))

    def metrics4d(self):
        """SPD matrices in z=(x-p_i,t-tau_i); q=z^T H_i z-r_i^2."""
        h = np.zeros((len(self.centers), 4, 4))
        h[:, :3, :3] = np.eye(3)
        h[:, :3, 3] = h[:, 3, :3] = -self.velocities
        h[:, 3, 3] = np.sum(self.velocities**2, axis=1) + self.temporal_precision
        return h

    def power4d(self, events):
        """Direct lifted 4D quadratic evaluation, independent of slice algebra."""
        events = np.asarray(events, dtype=np.float64)
        if events.ndim != 2 or events.shape[1] != 4 or not np.isfinite(events).all():
            raise ValueError('Events must have shape [N,4] and be finite')
        z = events[:, None] - np.column_stack((self.centers, self.times))[None]
        return np.einsum('bni,nij,bnj->bn', z, self.metrics4d(), z) - self.radii**2

    def ownership(self, events, batch_size=2048):
        events = np.asarray(events, dtype=np.float64)
        labels = []
        for start in range(0, len(events), batch_size):
            q = self.power4d(events[start:start+batch_size])
            ids = q.argmin(axis=1)
            ids[q[np.arange(len(q)), ids] > 0] = -1
            labels.append(ids)
        return np.concatenate(labels) if labels else np.empty(0, dtype=np.int64)
