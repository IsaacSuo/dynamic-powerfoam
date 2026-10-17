# MPM coupling redesign: transport a canonical Power Foam through a shared material map

> **Archived alternative:** This carrier proposal was not implemented.
> Later custom geometry probes are in [ANISOTROPIC_FOAM.md](ANISOTROPIC_FOAM.md).
> Their simplified appearance does not validate a complete dynamic Power Foam
> design. See the [paper/code reading corrections](POWERFOAM_READING_NOTES.md)
> before drawing conclusions from the proposals or results below.

Status: proposed design, 2026-09-23. The user removed the one-particle/one-cell
and unchanged-renderer constraints. This document revises the direction of the
original [ROADMAP.md](ROADMAP.md); it does not claim that the new renderer exists.

## Decision

Use MPM to evolve a volumetric material body. Store Power Foam geometry and
appearance in the reference material configuration. Couple them through a
shared, continuous deformation map Phi_t(X), rather than independently updating
each foam cell's position and scalar radius in world space.

The three resolutions are independent:

1. MPM material particles and background grid: physical accuracy and stability.
2. A reference tetrahedral carrier with shared vertices: a continuous, piecewise
   affine approximation of the material map used for rendering.
3. Canonical Power Foam cells and texture sites: image reconstruction capacity.

The carrier is a rendering data structure. It does not replace the MPM solver
with tetrahedral finite-element simulation. One foam cell may cross several
carrier tetrahedra; one carrier tetrahedron may contain several foam cells.

## What the previous experiments established

[Affine probe](experiments/AFFINE_RESULTS.md) and
[position/radius fit](experiments/AFFINE_FIT_RESULTS.md) showed that the tested
spherical, physically centered bounded-power reconstruction did not transport
material regions faithfully under shear/stretch. Small optimized center offsets
gave limited gains under the chosen objective and constraints. These are pilot
results, not a global impossibility result for power partitions or optimization.

An important mathematical qualification: an unbounded power partition under a
global affine map CAN be represented by another Euclidean power diagram when
its generators and weights are unrestricted. Those new generators generally
are not the physical particle positions; the transformed sphere bounds are
ellipsoids, and the new weights cannot simultaneously serve as their spherical
bound radii. Thus the obstacle is the coupled physical-center / power-generator /
spherical-bound parameterization, not the existence of planar interfaces alone.

## Geometry and material correspondence

Let C_i^0 be canonical bounded cells and Phi_t a continuous, globally injective
material map over the reference body. Define the current material region by

    C_i(t) = Phi_t(C_i^0 intersect reference_body).

Then distinct canonical cell interiors remain disjoint; an existing shared face
is transformed once by the same map from either side. The transformation does
not create additional gaps inside the transported domain. This does not repair
holes already present in the canonical foam. Global injectivity is essential:
positive Jacobians alone do not rule out distant parts overlapping.

Material ownership and appearance are queried through X = Phi_t^{-1}(x).
Canonical texture interpolation therefore uses its original sites, radii, and
metric. World-space re-partitioning never selects an unrelated material chart.
Cell IDs may persist as appearance IDs; they do not need to equal MPM IDs.

## Exact global-affine checkpoint

For x = F X + t with invertible F, P_i = F p_i^0 + t, define

    M = F^{-T} F^{-1}
    phi_i(x) = (x-P_i)^T M (x-P_i) - (r_i^0)^2.

This equals the canonical power distance exactly. The owner is argmin_i phi_i,
with background where min_i phi_i > 0. A shared M cancels from the quadratic
part of phi_i-phi_j, so pairwise interfaces remain planes; support becomes an
ellipsoid. Both the interfaces and bounds follow F without fitting.

A float64 spot check with 8,192 independent 3D points and the existing 121-cell
scene returned zero ownership mismatches for identity, rigid motion, uniform
scale, isochoric stretch and shear; maximum power-distance discrepancy was
7.11e-15. This was a mathematical membership check, not a GPU rendering test.

This checkpoint is for validating an implementation, not a novel research claim.
It also specifies exactly what a general coupling should reproduce when F is
constant throughout the object.

## Connecting the shared map to MPM

Start with a modest-deformation elastic solid and a known filled reference body.
The physical particles sample its volume, including the interior. Canonical foam
surface primitives are fitted separately to reference images/synthetic geometry.

Create a conforming tetrahedral carrier in that reference body. During each MPM
substep, advect its shared vertices with the same updated grid velocity field
used for material transport. Record reference and current carrier vertices.
Use one consistent interpolation/integration convention and document it.

Inside reference tetrahedron k:

    Phi_t(X) = A_k(t) X + b_k(t).

Shared vertex positions imply identical maps on shared triangular faces, so
neighboring tetrahedra cannot independently disagree about the same face.
This is the key difference from independently assigning a different affine map
to every foam cell.

The carrier only approximates the MPM motion. Measure Phi_t(p_i^0)-p_i(t) and
compare its Jacobian with the exported total material F_i; refine the carrier
when the approximation is inadequate. Do not silently substitute an elastic
return-mapped F_i^e for the total material map. If trajectory playback lacks grid
velocities, fitting shared carrier vertices to particle positions/F is a separate
fallback requiring regularization and validity checks, not assumed equivalent.

Reject/flag inverted tetrahedra and check nonlocal self-intersections. Begin
without fracture, merging, long fluid flow, or large self-contact. Those may
require changing the carrier topology and its reference charts.

## A renderer that can actually evaluate this model

First implement a correctness-oriented piecewise-affine ray renderer:

1. Intersect a world ray with current carrier tetrahedra and obtain ordered
   world-ray parameter intervals. Use a BVH initially.
2. Within interval k, pull the ray back using A_k^{-1}. It is a straight ray in
   canonical coordinates for that interval.
3. Intersect the canonical Power Foam restricted to that interval, including
   sphere bounds, power faces, and the dipole. Evaluate canonical texture.
4. Composite using world-space path length and an explicit optical-density
   convention. Transform output normals by A_k^{-T}.

For a unit world ray x(s)=o+s*d, use

    o0=A_k^{-1}(o-b_k), d0=A_k^{-1}d

without renormalizing d0, so s remains a world-distance parameter. The repository's
current ray_sphere_intersect forward helper assumes a unit direction; it must not
be reused blindly. Use the general quadratic coefficient dot(d0,d0), or explicitly
track the parameter scale if normalizing the pulled-back direction.

Initially use zero displacement and view-independent color. Separate physical
mass density from optical extinction; fixed material-attached extinction is a
declared baseline, not a mass-conservation law. Add displacement and directional
appearance only after ray intervals, ownership and texture transport are correct.

A canonical cell crossing tetrahedra is processed as several temporary geometric
fragments with the same appearance owner. Boundary ties must not double count
intervals. A full regular adjacency or another complete canonical query structure
is needed to traverse empty space; a Cech graph alone does not solve this.

The original world-space rasterizer cannot directly render these deformed cells.
Fast rasterization, fragment construction, sorting, and acceleration updates are
later engineering/research tasks. Do not claim the original FPS or painter order
survives. Record total frame costs, including the carrier and acceleration updates.

## Why not immediately use one metric tensor per particle?

Independent functions (x-P_i)^T M_i (x-P_i)-w_i define an anisotropic partition,
but for M_i != M_j their difference has a quadratic term. Interfaces can be
curved; convex-cell and single-ray-interval properties are not guaranteed. More
importantly, those competing local metrics do not automatically recover the same
shared material deformation or guarantee material correspondence.

This is a legitimate alternative research direction, but it does not supply the
clean reference solution that the shared-map formulation gives us. Arbitrarily
independent affine images of whole cells have the analogous overlap/gap problem.

## Revised milestones and decision gates

| Stage | Deliverable | Required evidence |
| --- | --- | --- |
| A: global affine | Canonical-ray renderer and shared-metric oracle | Float64 labels exactly agree off ties; float32 images agree within measured numeric error under rotation, scale, stretch and shear; correct ray-parameter conversion |
| B: varying deformation | Shared-vertex carrier driven by an analytic bend/twist | No additional cracks, overlaps, or cell-identity changes; inverse-map and image convergence as carrier resolution increases |
| C: real MPM | Small elastic block driven by recorded MPM grid velocities | Carrier/particle position and F residuals, valid tetrahedra, held-out views/times, frozen-appearance consistency |
| D: speed | GPU intervals, BVH/refit, fragment or tile acceleration | End-to-end frame time, memory, visual agreement against A/B/C reference |
| E: richer behavior | Displacement, view dependence, contact/plasticity | Separate correctness tests; total-vs-elastic deformation semantics; explicit chart/topology event handling |

B is the decisive test; a single shared affine transform alone is insufficient
evidence that arbitrary MPM motion has been solved. Measure visual error relative
to the carrier's own deformation separately from carrier-vs-MPM approximation.

If the fragmented foam renderer is too expensive, use the same carrier to drive
an extracted canonical surface mesh as a practical animation baseline. That
trades away the original volumetric representation and must be named as such.

## Research scope and references

Canonical deformation and cage-based material maps are established ideas. The
potential contribution here would be an efficient, accurate MPM-driven cellular
renderer with material-consistent appearance, not merely attaching MPM to a
reference field. Compare against a deformed textured mesh and MPM-driven Gaussian
rendering before making novelty or performance claims.

- [Power Foam, main paper and supplement](https://arxiv.org/pdf/2604.24994)
- [Constructing Affine and Curved Voronoi Diagrams, Boissonnat](https://www.astro.rug.nl/~weygaert/tim1publication/jigsaw/boissonnat_jigsaw.pdf)
- [Deforming Radiance Fields with Cages, project and paper](https://xth430.github.io/deforming-nerf/)
- [PhysGaussian](https://arxiv.org/abs/2311.12198)
