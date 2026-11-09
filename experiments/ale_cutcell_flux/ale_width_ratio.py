"""Measure the initial ALE acoustic length V/A from the saved power-ball foam."""
import argparse
from pathlib import Path

import numpy as np


DEFAULT_FOAM = Path(__file__).resolve().parents[2] / "foams/fourier8k_clean/dem_exact_single"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("foam", nargs="?", type=Path, default=DEFAULT_FOAM)
    args = parser.parse_args()

    with np.load(args.foam / "geometry.npz") as geometry, np.load(args.foam / "cells.npz") as cells:
        points = geometry["points"]
        radii = cells["radii"].ravel()
        # The archived unit-density geometry stores volumes under "masses".
        volumes = geometry["masses"].ravel()
        i, j = geometry["pairs"].T
        area = geometry["areas"].ravel()
        if not np.array_equal(points, cells["points"]):
            raise ValueError("geometry and radii must use the same cell ordering")

    distance = np.linalg.norm(points[j] - points[i], axis=1)
    distance_i = (distance**2 + radii[i]**2 - radii[j]**2) / (2 * distance)
    wall_area = np.bincount(i, area, minlength=len(points)) + np.bincount(j, area, minlength=len(points))
    wall_moment = np.bincount(i, area * distance_i, minlength=len(points))
    wall_moment += np.bincount(j, area * (distance - distance_i), minlength=len(points))
    # The divergence theorem gives 3V = sum(wall area * distance) + radius * cap area.
    cap_area = (3 * volumes - wall_moment) / radii
    if np.any(cap_area < -1e-10):
        raise ValueError("geometry contains a negative cap area beyond roundoff")

    occupied = volumes > 0
    length_mm = 1000 * volumes[occupied] / (wall_area + np.maximum(cap_area, 0))[occupied]
    quantiles = np.array([0, 0.001, 0.01, 0.05, 0.5])
    median = np.median(length_mm)
    print(f"{len(length_mm)} unmerged cells")
    print("percentile    V/A (mm)    percent of median")
    for quantile, length in zip(quantiles, np.quantile(length_mm, quantiles)):
        print(f"{100 * quantile:10g}    {length:8.4f}    {100 * length / median:17.2f}")
    print(f"cells below 10% of the median: {np.count_nonzero(length_mm < 0.1 * median)}")


if __name__ == "__main__":
    main()
