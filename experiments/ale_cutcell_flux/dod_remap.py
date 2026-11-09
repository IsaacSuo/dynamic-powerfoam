"""Conservative overlap remap for cell averages and half-state contrasts in 1D."""
import numpy as np


def bin_partition(edges, split):
    """Split marked intervals at their midpoints and return bin-to-cell indices."""
    widths = np.diff(edges)
    if np.any(widths <= 0.):
        raise ValueError("remap requires strictly increasing edges")
    counts = 1+np.asarray(split, dtype=int)
    ends = np.empty(int(sum(counts))+1)
    ends[0] = edges[0]
    cell_ends = np.cumsum(counts)
    ends[cell_ends] = edges[1:]
    ends[cell_ends[split]-1] = edges[:-1][split]+.5*widths[split]
    if np.any(np.diff(ends) <= 0.):
        raise ValueError("a requested half-cell midpoint is not representable")
    return ends, cell_ends


def donor_bins(edges, mean, contrast):
    split = np.any(contrast != 0., axis=1)
    source_edges, source_ends = bin_partition(edges, split)
    source = np.empty((len(source_edges)-1, 2))
    source[source_ends-1] = mean
    left_index = source_ends[split]-2
    widths = np.diff(source_edges)
    # Use represented lengths to retain the donor mean when the
    # floating-point midpoint is slightly off center.
    left_fraction = widths[left_index]/np.diff(edges)[split]
    jump = 2*contrast[split]
    source[left_index] = mean[split]-(1-left_fraction)[:, None]*jump
    source[left_index+1] = mean[split]+left_fraction[:, None]*jump
    if np.any(source[:, 0] <= 0.):
        raise ValueError("represented donor halves must have positive stretch")
    return source_edges, source


def overlap_integrals(source_edges, source, target_edges):
    """Integrate arbitrary piecewise-constant values over the target partition."""
    integral = np.zeros((len(target_edges)-1, source.shape[1]))
    i = j = 0
    while i < len(source) and j < len(integral):
        lo = max(source_edges[i], target_edges[j])
        hi = min(source_edges[i+1], target_edges[j+1])
        if hi > lo:
            integral[j] += (hi-lo)*source[i]
        source_end, target_end = source_edges[i+1], target_edges[j+1]
        i += source_end <= target_end
        j += target_end <= source_end
    assert i == len(source) and j == len(integral)

    return integral


def remap(old_edges, old_mean, old_contrast, new_edges, keep_halves):
    """Average the old piecewise-constant state over every new cell or half."""
    old_edges = np.asarray(old_edges, dtype=float)
    new_edges = np.asarray(new_edges, dtype=float)
    old_mean = np.asarray(old_mean, dtype=float)
    old_contrast = np.asarray(old_contrast, dtype=float)
    keep_halves = np.asarray(keep_halves, dtype=bool)
    if any(edges.ndim != 1 or len(edges) < 2 for edges in (old_edges, new_edges)):
        raise ValueError("remap requires one-dimensional cell partitions")
    if old_edges[0] != new_edges[0] or old_edges[-1] != new_edges[-1]:
        raise ValueError("remap requires the same reference domain")
    if old_mean.shape != (len(old_edges)-1, 2) or old_contrast.shape != old_mean.shape:
        raise ValueError("expected a stretch and velocity per old cell")
    if keep_halves.shape != (len(new_edges)-1,):
        raise ValueError("expected one half-state flag per new cell")
    if not all(np.isfinite(a).all() for a in (old_edges, new_edges, old_mean, old_contrast)):
        raise ValueError("remap requires finite edges and states")
    if np.any(old_mean[:, 0] <= abs(old_contrast[:, 0])):
        raise ValueError("remap requires positive donor stretch")

    source_edges, source = donor_bins(old_edges, old_mean, old_contrast)
    target_edges, target_ends = bin_partition(new_edges, keep_halves)
    integral = overlap_integrals(source_edges, source, target_edges)

    cell_integral = integral[target_ends-1].copy()
    cell_integral[keep_halves] += integral[target_ends[keep_halves]-2]
    mean = cell_integral/np.diff(new_edges)[:, None]
    contrast = np.zeros_like(mean)
    left_index = target_ends[keep_halves]-2
    left = integral[left_index]/(target_edges[left_index+1]-target_edges[left_index])[:, None]
    right = integral[left_index+1]/(target_edges[left_index+2]-target_edges[left_index+1])[:, None]
    contrast[keep_halves] = .5*(right-left)
    at_limit = abs(contrast[:, 0]) >= mean[:, 0]
    if np.any(at_limit):
        bound = np.nextafter(mean[at_limit, 0], 0.)
        contrast[at_limit, 1] *= bound/abs(contrast[at_limit, 0])
        contrast[at_limit, 0] = np.copysign(bound, contrast[at_limit, 0])
    return mean, contrast
