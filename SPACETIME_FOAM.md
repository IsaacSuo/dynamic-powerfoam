# Spacetime Power Foam: first prototype

> **Scope correction (2026-09-25):** This is our hand-authored geometry probe,
> not a reproduction or trained dynamic extension of the complete Power Foam
> model. The red/blue handoff is a constructed appearance counterexample, not
> evidence that learned 4D Power Foam must flicker. See the
> [paper/code reading and corrections](POWERFOAM_READING_NOTES.md).

2026-09-24. This is a separate dynamic-reconstruction probe alongside
[the local-F deformation experiment](ANISOTROPIC_FOAM.md). MPM integration is
deferred while exploring the spacetime representation. No video fitting or
physical simulation is implemented here.

## Definition and implementation

Each primitive stores its center p at time tau, constant velocity v, peak support
radius r, and positive temporal precision lambda:

```
q_i(x,t) = |x - p_i - v_i (t-tau_i)|^2 + lambda_i (t-tau_i)^2 - r_i^2
C_i      = { (x,t): q_i <= 0 and q_i <= q_j for all j }
```

This is a restricted generalized quadratic partition in four dimensions, **not**
an ordinary Euclidean 4D power diagram. Its SPD matrix has blocks
`[[I, -v], [-v^T, |v|^2+lambda]]`. Different velocities change the 4D metric but
leave the spatial block equal to I for every cell. Therefore each time slice is
exactly a standard 3D bounded power diagram:

```
center_i(t)    = p_i + v_i (t-tau_i)
radius_i(t)^2  = r_i^2 - lambda_i (t-tau_i)^2
```

Negative squared radii imply empty support; the zero-radius endpoint is omitted
from volume rendering as a measure-zero tangency. An inactive cell has positive
power everywhere and cannot defeat any negative-power occupied cell, so removing
it is safe. Cell IDs persist in the slice adapter even when active counts change.
Pairwise spatial bisectors remain planes, but their normals change when velocities
differ. Support lifetime is `tau +/- r/sqrt(lambda)`; it is distinct from actual
cell occupancy, which may be removed by competition with other cells.

[powerfoam/spacetime.py](powerfoam/spacetime.py) implements parameters, slices and
independent lifted 4D power evaluation. The experiment rebuilds the GPU sphere
overlap graph at every time and feeds the slice into the original rasterizer.
No 4D triangulation, curved-face renderer or modification to that rasterizer is
needed for this restricted geometry.

The prototype uses constant +z dipole normals, eight fixed world-space texture offsets
relative to each center (scaled by the peak/reference radius),
constant color within each cell, and zero displacement. Thus the surface plane
is `n_i dot (x-p_i-v_i(t-tau_i)) = 0`. The existing texture denominator is the
current slice radius squared. Constant colors deliberately isolate geometry and
cell handoff; this does not validate transport of detailed texture. In particular,
the offsets do not rescale with the current slice radius as they would in
`PowerfoamScene.forward()`. The probe also bypasses Spherical Voronoi appearance
and joint optimization of the original local parameters.

## Reproduce and inspect

```bash
conda activate om
python -m experiments.spacetime_probe
```

Defaults: seed 2026, 49 initially coplanar cells, spatially varying straight
trajectories, staggered support lifetimes, 192x192 pixels, 25 sampled times.
The zero-velocity control has identical geometry at t=0 and identical temporal
support parameters. Both lose support near their temporal endpoints; this is
intentional for the lifetime test, not a claim of preserving an intact object.
Display playback runs forward and backward, producing a 48-frame GIF. The model
itself has constant velocities rather than periodic trajectories.

Artifacts in Git-ignored `output/spacetime/`:

- `motion.gif`: temporal gates with zero velocity versus moving spacetime Foam.
- `spacetime_partition.png`: ownership on an x-time plane through four supports.
- `slices.png`: independent all-face float64 oracle versus original GPU renderer.
- `handoff.png`: appearance jump across a cell ownership change.
- `results.json`: checks, image errors, active supports, handoff and precision probe.

## Observations

1. **Slice equivalence passes.** Across 69,632 events at 17 times, direct lifted
   4D quadratic competition and slice-based 3D power competition produce identical
   owners. Maximum power-value discrepancy is `1.07e-14`. SPD, empty frames, and
   support lifetime endpoints also pass checks.
2. **Generated faces rotate.** A two-cell fixture with different velocities
   rotates the bisector normal from -21.80 to +21.80 degrees. Direct 4D evaluation
   confirms points on both sliced planes satisfy equal power. This is an analytic
   fixture, not evidence of arbitrary rotational deformation of a material cell.
3. **Existing rendering works, with a numerical limitation.** Five time slices
   have 22, 49, 49, 49 and 24 active supports. Full-image mean RGB errors against
   the independent oracle range from `6.68e-7` to `3.88e-5`; maximum foreground
   mean error is `2.81e-4`. However, each 192x192 image has 0–4 pixels with maximum
   channel error above 0.1, and the worst channel error is 0.857. These errors are
   not hidden by reporting only the background-dominated mean.
4. **Deliberately mismatched colors create a discontinuity in this fixture.** Two overlapping
   lifetimes remain opaque along a moving test ray, but its owner switches from
   a red cell to a blue cell. Comparing times +/-epsilon produces RGB MAE 0.64
   for epsilon from `1e-2` down to `1e-5`. Identical cell colors remove this jump
   in a control. Neither control tests appearance learning, and this result
   cannot establish an inherent failure of trained dynamic Power Foam.

Source inspection found float16 neighbor offsets/radii in the original
`rasterize.py` adjacency buffer. In a separate float32 clipping implementation
(`SharedAffineRayTracer` with identity F), the t=-0.7 slice has mean RGB error
`1.90e-8`, maximum error `4.87e-4`, and no pixels above 0.1. This supports treating
the original renderer's outliers as a rendering-implementation issue rather than
a failure of the exact slice reduction. It is not a controlled proof that all
outliers have only one cause: the two GPU implementations differ in other ways.

## What this establishes and what remains

The geometry can move, appear and disappear as slices of a single explicit
spacetime competition model, while preserving the 3D power-diagram structure
needed by the existing rendering algorithms. It does **not** establish video
reconstruction quality, compression, speed, arbitrary material deformation,
physical conservation or temporal appearance consistency.

The earlier recommendation to prioritize handoff coupling was not justified by
this fixture and is withdrawn. Before choosing a new experiment, specify the
complete local-site, displacement, directional-appearance and training pipeline
against the original implementation. Position/velocity fitting, temporal
splitting and temporal regularization have not been evaluated. Image quality,
held-out times, temporal changes and primitive budget would all need evaluation
before claiming an advantage over time-conditioned 3D Foam or 4D Gaussians.

Conceptual references: [Power Foam](https://arxiv.org/abs/2604.24994) for bounded
power cells and [native 4D Gaussian splatting](https://arxiv.org/abs/2310.10642)
for representing a sequence in spacetime and extracting time-conditioned views.
The construction above is our derived prototype; no novelty claim is established.
