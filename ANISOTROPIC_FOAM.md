# Dynamic anisotropic Power Foam

Prototype direction explored on 2026-09-23. The objective was to make the Foam
representation itself deform. The tetrahedral-carrier proposal in
DESIGN_RETHINK.md is retained as an alternative.

> **Scope correction (2026-09-25):** These are custom frozen-appearance geometry
> experiments, not a reproduction of Power Foam's full detail-site, displacement,
> Spherical Voronoi and optimization pipeline. Advected cell-label consistency
> is an additional material-model objective, not the paper's reconstruction
> criterion. No dynamic direction is validated by these probes; see the
> [paper/code analysis](POWERFOAM_READING_NOTES.md).

## Representation

Replace spherical power distance with

    q_i(x,t) = (x-p_i(t))^T A_i(t) (x-p_i(t)) - (r_i^0)^2
    A_i(t) = F_i(t)^{-T} F_i(t)^{-1}.

The bounded cell is defined by q_i <= 0 and q_i <= q_j for all j. This remains
a partition within the union of supports (up to ties), not independent overlapping
ellipsoids. Positions, deformation tensors and reference radii generate all
interfaces; no interface is independently animated. Positive-definite A_i is
required, and MPM must provide the appropriate total material deformation rather
than silently substituting a plasticity-return-mapped elastic tensor.

The reference radius is not multiplied by J^(1/3) in this formulation: F already
changes the support volume through A. Doing both would double-count deformation.
The support ellipsoid volume is J times its reference sphere volume, which still
does not imply that the clipped cell volume equals the particle material volume.

## Implemented checkpoint: shared F

For p_i(t)=F*p_i^0+t and A=F^{-T}F^{-1} shared across all cells,

    q_i(F*X+t,t) = ||X-p_i^0||^2 - (r_i^0)^2.

Consequently the entire bounded partition is transported exactly, apart from
floating-point error. Pairwise interfaces remain planes because the common
quadratic term cancels. Their world normal is proportional to

    A*(p_j-p_i) = F^{-T}*(p_j^0-p_i^0).

Support intersections and cell adjacency are preserved by the invertible global
map. The canonical sphere-overlap graph therefore remains a valid conservative
set of clipping constraints in this checkpoint. This is not a claim about
different F_i or their time-varying adjacency.

Implementation: [powerfoam/affine_raytrace.py](powerfoam/affine_raytrace.py).

- Computes ellipsoid intersections and metric-generated power planes directly
  in world space, on the GPU with Warp.
- Transforms dipole normals by inverse transpose.
- Queries the frozen soft texture in canonical coordinates.
- Composites occupied ray intervals with world-space path lengths.
- Supports perspective rays, arbitrary origins and rays starting inside cells.
- Forward-only, zero displacement, view-independent color. Scans cells in
  camera-power order; this is a correctness prototype, not an accelerated
  neighbor-walking implementation or a speed benchmark.

The independent float64 oracle instead transforms rays to reference space,
clips against ALL competitors (not just the GPU's sphere-overlap graph), and
sorts intersections separately per ray (not by the GPU's global painter order).
It preserves the world ray parameter by retaining unnormalized reference ray
directions. Both paths evaluate the same finite-thickness volume model, allowing
meaningful pixelwise comparisons including silhouettes and transparency.

Density is an explicit fixed extinction coefficient per unit world length.
It is not interpreted as physical mass density or a promise that deformation
preserves opacity. Current tests freeze density at 1000 for the checker panels
and 8 for transparent diagnostics.

## Run and inspect

```bash
conda activate om
python -m experiments.shared_affine_render
```

Default scene: 121 cells, 8 texels/cell, 192x192 images. Includes identity,
rotation/translation, uniform scaling, isochoric stretch, shear and a tilted
anisotropic stretch. Additional checks cover an oblique transparent scene,
arbitrary ray origins/parallel rays, and noncoplanar cells with varied normals.
Singular and orientation-reversing maps are explicitly rejected.

Artifacts under `output/shared_affine/` (Git-ignored):

- `motion.gif`: old volume-radius update beside shared-metric Foam, 32-frame loop.
- `comparison.png`: float64 reference, old spherical renderer, new GPU renderer,
  and amplified GPU/reference difference.
- `results.json`: geometry labels, RGB, alpha, depth, normals and hit comparisons.
- Individual reference, new/old RGB and alpha PNGs.

The validation is deterministic and measures implementation correctness for this
model. It does not demonstrate a new optimal representation or solve spatially
varying material deformation by itself.

## Implemented checkpoint: different F_i

With different A_i, q_i-q_j contains a quadratic term. Boundaries are generally
quadrics; a cell need not remain convex or connected, and a ray may cross it in
multiple intervals. The existing plane clipper and global painter order cannot
be assumed valid. Simply passing per-cell tensors to the current kernel would
be incorrect.

Implemented in [powerfoam/quadratic_raytrace.py](powerfoam/quadratic_raytrace.py):
a forward-only float64 PyTorch GPU reference that collects support, pairwise
quadratic and dipole intersections, classifies intervals by power competition,
merges redundant subdivisions, and composites in per-ray depth order. It permits
multiple separated intervals of the same cell. It does not use old adjacency or
planar clipping assumptions, and is not an accelerated production renderer.

Run `python -m experiments.varying_metric_render` in `om`. Analytic injective bend
and twist maps supply consistent positions and Jacobians. Material labels are
compared with exact nonlinear transport; image references use sampled integration
of that nonlinear field, with a separate step-refinement check. The local-F
representation is only a first-order approximation to nonlinear transport.

The tests pass a constructed three-interval shell case, reduction to the shared-F
oracle, and 98,304 direct point classifications. In the 121-cell probe, volume
label error decreases from 13.19% to 1.41% for the stronger bend, and from 13.91%
to 0.36% for twist, compared with spherical cells. Surface label error decreases
as cells are refined from 49 to 289 in both probes. These are single-seed results,
not a general convergence guarantee. Full results and limitations are recorded in
[experiments/VARYING_METRIC_RESULTS.md](experiments/VARYING_METRIC_RESULTS.md).

## Previously proposed follow-up: elastic MPM (not implemented)

This follow-up remains a proposal; the geometry probes alone do not justify
selecting it over other dynamic representations.

Connect a small elastic MPM sequence to the existing `(p_i, F_i)` interface and
retain the quadratic-event renderer as a correctness baseline. Check total
material deformation, support coverage and persistent material labels before
designing conservative broad-phase adjacency and accelerated curved-interface
rendering. Sample counts currently identify unobserved cells; they do not certify
geometrically empty cells or physical volume conservation.

Separate the contribution of ellipsoid support, generated interfaces, and
canonical texture querying with ablations. Test topology events and temporal
continuity. Keep the spherical and position/radius-fit experiments as baselines.

The novelty would have to lie in the physically driven dynamic Foam and its
rendering/topology algorithms; anisotropic distance itself is established.
