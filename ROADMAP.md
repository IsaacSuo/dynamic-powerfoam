# Historical roadmap: MPM-Coupled Dynamic Power Foam

> **Current research route:** [RESEARCH_ROUTE.md](RESEARCH_ROUTE.md) is the
> authoritative plan as of 2026-09-25. It defines the research hypotheses,
> competing representations, full experiment sequence, decision branches and
> eventual MPM extension. The one-particle/one-cell proposal below is historical.

> **Reading correction (2026-09-25):** This roadmap is an earlier proposal, not
> a validated dynamic Power Foam design. Our affine, anisotropic and spacetime
> probes test custom geometry with simplified appearance; they do not reproduce
> the complete local detail-site, displacement, Spherical Voronoi and training
> pipeline. Persistent cell/material identity is our extra hypothesis.
> Read [the paper/code analysis](POWERFOAM_READING_NOTES.md) before using these
> results to select a direction. Existing prototypes are retained in
> [ANISOTROPIC_FOAM.md](ANISOTROPIC_FOAM.md) and
> [SPACETIME_FOAM.md](SPACETIME_FOAM.md); MPM coupling remains unimplemented.

## 1. Research objective

The original proposal investigates a dynamic extension of Power Foam in which one bounded
power cell corresponds to one persistent Material Point Method (MPM) particle:

\[
\text{power cell ID} = \text{material point ID}.
\]

MPM evolves the material state

\[
(p_i, v_i, F_i),
\]

while Power Foam reconstructs a bounded power partition from the current particle
positions and support radii, then renders the same state through rasterization or
ray tracing.

The intended contribution is not a per-primitive neural deformation field. The
foam is treated as a time-varying cellular complex with persistent material
identity, physically driven geometry, and an explicitly maintained rendering
topology.

The central research question is:

\[
\boxed{F_i(t) \longrightarrow r_i(t)}
\]

where the Power Foam radius is simultaneously a power weight and a bounding
sphere radius.

## 2. Core semantics and invariants

Each canonical material point/cell stores:

- reference position \(p_i^0\);
- reference material volume \(V_i^0\);
- reference Power Foam radius \(r_i^0\);
- reference dipole normal and tangent frame;
- cell-local texture sites, displacement values, density, and appearance;
- persistent material neighbors \(E_M\).

At time \(t\), MPM provides:

- current position \(p_i(t)\);
- velocity \(v_i(t)\);
- total deformation gradient \(F_i(t)\);
- optionally elastic/plastic factors \(F_i^e,F_i^p\), mass, density, material
  label, damage, and contact state.

The following invariants should hold in the first implementation:

1. Cell IDs persist over time.
2. Texture and appearance remain attached to material IDs.
3. Rigid motion does not change the radius.
4. Uniform scaling transforms the bounded power diagram equivariantly.
5. Material topology and rendering topology are represented separately.
6. Radius corrections remain small unless the geometry demonstrates that the
   physical prior is insufficient.

## 3. Mapping MPM state to Power Foam geometry

### 3.1 Position

The current MPM particle position directly becomes the current power site:

\[
p_i^{\mathrm{foam}}(t)=p_i^{\mathrm{MPM}}(t).
\]

### 3.2 Surface frame

The deformation gradient must affect more than the radius. For a canonical
dipole normal \(n_i^0\), use the inverse-transpose rule:

\[
n_i(t)=\frac{F_i(t)^{-T}n_i^0}{\|F_i(t)^{-T}n_i^0\|}.
\]

Canonical tangent and texture offsets are pushed forward by \(F_i\):

\[
u_i(t)=F_i(t)u_i^0.
\]

This allows shear and anisotropic stretch to appear in the oriented surface and
texture chart even though the bounded power support still uses a scalar radius.
The current Power Foam implementation derives texture offsets from an orthogonal
frame multiplied by one scalar radius; the dynamic implementation must instead
support transformed three-dimensional canonical offsets.

### 3.3 Density and appearance

Rendering density is not automatically material mass density. It should remain a
separate material-attached appearance parameter in the baseline. A later ablation
may test optical-density conservation, for example \(\sigma_i(t)=\sigma_i^0/J_i\),
only when that interpretation matches the scene model.

## 4. Radius model

Let

\[
J_i(t)=\det F_i(t).
\]

The current material volume is

\[
V_i(t)=J_i(t)V_i^0.
\]

### R0: fixed-radius control

\[
r_i(t)=r_i^0.
\]

This is the required control experiment. It isolates the effect of particle
motion from the effect of radius evolution.

### R1: volume-equivalent physics prior

Use

\[
\boxed{r_i^{\mathrm{vol}}(t)=r_i^0J_i(t)^{1/3}}.
\]

Equivalently,

\[
\log r_i^{\mathrm{vol}}(t)=\log r_i^0+\frac{1}{3}\log J_i(t).
\]

This formula is exact for the volume-equivalent sphere and is equivariant under
uniform scaling. It is not a claim that the bounded power-cell volume equals
\(4\pi r_i^3/3\): the actual cell is clipped by radical planes and neighboring
cells.

Numerical guards are mandatory:

- require orientation-preserving states or explicitly handle damage/inversion;
- clamp singular values and \(J\) away from zero for rendering;
- distinguish total, elastic, and plastic deformation gradients by material
  model;
- for fluids whose implementation does not retain a meaningful total \(F\), use
  the current particle volume or \(m_i/\rho_i\) instead.

### R2: anisotropic isotropic-closure model

Let the singular values of \(F_i\) be

\[
\sigma_{i,1},\sigma_{i,2},\sigma_{i,3}>0.
\]

The smallest sphere enclosing the local affine image of the reference sphere has
radius

\[
r_i^{\mathrm{cover}}(t)=r_i^0\sigma_{i,\max}(t).
\]

Interpolate in log-radius space:

\[
\boxed{
r_i(t)=
r_i^{\mathrm{vol}}(t)
\exp\left(
\alpha_i(t)\log\frac{r_i^{\mathrm{cover}}(t)}
{r_i^{\mathrm{vol}}(t)}
\right),
\quad \alpha_i(t)\in[0,1].
}
\]

Interpretation:

- \(\alpha=0\): volume-equivalent support;
- \(\alpha=1\): conservative local affine coverage;
- intermediate values: an isotropic closure for anisotropic deformation.

Start with one global fixed \(\alpha\). Then test per-material and per-particle
values. The coverage formula is only first-order when \(F\) varies appreciably
across a particle's support.

A robust anisotropy measure is

\[
a_i=\|\operatorname{dev}(\log U_i)\|_F,
\qquad F_i=R_iU_i,
\]

rather than relying only on the largest singular value. This measure can control
the permitted correction range or the strength of the physics prior.

### R3: physics prior with local geometry correction

Use the primary parameterization

\[
\boxed{
\log r_i(t)=
\log r_i^0+rac13\log J_i(t)+\delta_i(t)
}
\]

with \(\delta_i\) optimized directly or solved locally, not initially predicted
by an MLP.

Regularize it with

\[
L_{\mathrm{prior}}=\sum_i\delta_i^2,
\]

\[
L_{\mathrm{time}}=
\sum_i\|\delta_i(t)-\delta_i(t-\Delta t)\|^2,
\]

and

\[
L_{\mathrm{material\ smooth}}=
\sum_{(i,j)\in E_M}(\delta_i-\delta_j)^2.
\]

For persistent material neighbors, define

\[
o_{ij}(t)=r_i(t)+r_j(t)-\|p_i(t)-p_j(t)\|.
\]

Penalize gaps and excessive overlap with separate hinge losses. Contact pairs
that are not material neighbors must not automatically be treated as topology
errors: material adjacency, collision/contact, and render adjacency have
different semantics.

### R4: target-volume constrained bounded power cells

Define the actual bounded cell

\[
C_i(t)=B(p_i(t),r_i(t))\cap P_i(t),
\]

where \(P_i\) is the power cell. The long-term objective is

\[
\boxed{
L_{\mathrm{volume}}=
\sum_i
\left[
\log\operatorname{Vol}(C_i(t))-
\log(J_i(t)V_i^0)
\right]^2.
}
\]

Prescribed-volume Laguerre diagrams are related to semi-discrete optimal
transport, but standard solvers do not directly solve this bounded problem. In a
standard power diagram, adding a constant to every weight does not change the
partition. In Power Foam, the same values also define bounding sphere radii, so
that gauge transformation changes the bounds. Efficient bounded-cell volume
evaluation and differentiation are therefore research tasks rather than assumed
building blocks.

## 5. Dual topology

### 5.1 Material topology

The material graph \(E_M\) is persistent and is used for:

- material correspondence;
- spatial regularization of radius corrections;
- deformation and strain diagnostics;
- texture ownership;
- damage, fracture, and material-neighbor semantics.

It should be initialized from the reference configuration and only changed by an
explicit material event such as fracture, merging, insertion, or deletion.

### 5.2 Render topology

The render graph \(E_R(t)\) is generated from current sites and radii and is used
for the bounded power diagram, rasterization, and ray traversal. It may include
temporary contact pairs and may change without changing material identity.

The baseline rebuilds the GPU Cech graph at every rendered time. Later stages
should investigate:

- conservative temporal-union adjacency from swept bounds;
- incremental graph updates;
- event-based kinetic regular triangulation;
- separate handling of material fracture and geometric contact.

Missing a required render edge is incorrect; extra conservative edges should
preserve geometry but increase work.

## 6. Coupling modes

### Offline one-way coupling

1. Run MPM and save \(p_i,v_i,F_i\) per frame.
2. Load trajectories into Dynamic Power Foam.
3. Optimize appearance and radius corrections while keeping physics fixed.

This is the recommended first implementation because it isolates rendering and
geometry problems from differentiable simulation problems.

### Joint differentiable coupling

After the offline pipeline is validated, optionally differentiate image losses
through the renderer and MPM to infer physical parameters or initial conditions.
This requires a differentiable MPM implementation, stable checkpointing, and
careful treatment of contact and topology events.

## 7. Training objective

The initial dynamic objective is

\[
L=
L_{\mathrm{rgb}}+
\lambda_{\mathrm{ssim}}L_{\mathrm{ssim}}+
\lambda_nL_{\mathrm{normal}}+
\lambda_cL_{\mathrm{connect}}+
\lambda_pL_{\mathrm{prior}}+
\lambda_tL_{\mathrm{time}}+
\lambda_sL_{\mathrm{material\ smooth}}+
\lambda_gL_{\mathrm{gap}}.
\]

The bounded-cell volume term is introduced only after its estimator has been
validated against exact small diagrams.

MPM state is authoritative in the one-way baseline: rendering losses may update
appearance and radius correction, but not particle trajectories.

## 8. Canonical refinement policy

Dynamic densification must preserve cell/material identity. Do not independently
clone particles at different frames.

When a canonical cell is refined:

1. split the corresponding material sample in the reference configuration;
2. conserve mass and reference volume;
3. initialize child material state consistently;
4. inherit/split texture and appearance parameters;
5. update material adjacency once;
6. apply the refinement to all frames.

The first MPM-coupled prototype should avoid refinement entirely and use a fixed
particle set. Refinement is a separate research milestone.

## 9. Proposed code architecture

```text
powerfoam/
  dynamics/
    mpm_state.py          # trajectory/state schema and interpolation
    material_graph.py     # persistent material topology
    radius_model.py       # R0-R4 radius closures
    kinematics.py         # F-based normals and texture offsets
    render_topology.py    # per-time Cech/adjacency management
  scene.py                # accept time-dependent explicit attributes
data_loader/
  dynamic_sequence.py     # timestamped cameras and MPM state
configs/
  dynamic/
train_dynamic.py
test_dynamic.py
benchmark_dynamic.py
```

The static path must remain operational throughout development. Dynamic state
should be passed as explicit temporary tensors rather than mutating canonical
model parameters in place.

## 10. Milestones

### M0: reproducible static baseline

- Train and evaluate the upstream Power Foam implementation.
- Record image metrics, point count, graph degree, graph build time, raster FPS,
  and ray-tracing FPS.

Exit criterion: static results and kernels are reproducible on the target GPU.

### M1: prescribed particle trajectories

- Add persistent timestamps and MPM state loading.
- Drive sites directly from recorded particle positions.
- Transform normals and texture offsets using \(F\).
- Implement fixed radius and \(J^{1/3}\) radius modes.
- Rebuild render adjacency for every frame.

Exit criterion: rigid transforms and uniform scaling pass analytic tests; the
same material ID retains its appearance across time.

### M2: anisotropic closure

- Compute polar decomposition/SVD robustly.
- Add global \(\alpha\) interpolation from volume to coverage radius.
- Evaluate pure shear and isochoric anisotropic stretch.

Exit criterion: support misses decrease without unacceptable graph densification.

### M3: local radius correction

- Add direct per-particle/per-keyframe \(\delta\).
- Add spatial and temporal regularization.
- Add material-neighbor gap and render-overlap objectives.

Exit criterion: correction improves image/support metrics while remaining small
and smooth.

### M4: bounded cell-volume matching

- Implement exact volume checks for small diagrams.
- Implement and validate an approximate differentiable estimator.
- Add target volume \(J_iV_i^0\).
- Compare direct optimization, local Newton updates, and OT-inspired solvers.

Exit criterion: bounded-cell volume error improves materially without collapsing
cells or making the render graph prohibitively dense.

### M5: dynamic topology and two-way physics

- Add temporal-union or incremental adjacency.
- Add explicit contact/fracture semantics.
- Evaluate differentiable MPM coupling only after the one-way system is stable.

## 11. Required analytic and synthetic tests

Every radius model must be tested on:

1. rigid translation;
2. rigid rotation;
3. uniform expansion/compression;
4. volume-preserving stretch \(\operatorname{diag}(2,1,0.5)\);
5. pure shear;
6. near-incompressible large deformation;
7. two previously distant regions entering contact;
8. material separation or fracture;
9. spatially varying \(F\);
10. particle disorder and changing render adjacency.

## 12. Ablation matrix

Radius variants:

- fixed \(r_0\);
- \(r_0J^{1/3}\);
- \(r_0\sigma_{\max}\);
- global \(\alpha\) closure;
- per-material \(\alpha\);
- local \(\delta\) correction;
- target bounded-cell volume.

Geometry variants:

- update position only;
- position plus rotated normal;
- full inverse-transpose normal and \(F\)-transformed texture offsets.

Topology variants:

- rebuild every frame;
- fixed canonical graph where valid;
- temporal-union Cech graph;
- incremental/event-based updates.

## 13. Evaluation metrics

Rendering:

- PSNR, SSIM, LPIPS;
- temporal LPIPS or warping error;
- raster and ray-traced FPS;
- ray/raster numerical agreement.

Physics and geometry:

- relative bounded-cell volume error;
- support miss rate against deformed reference samples;
- radius correction magnitude;
- material-neighbor gap statistics;
- normal and texture distortion;
- conservation error.

Topology and performance:

- mean/max render graph degree;
- number of graph edge insertions/deletions;
- topology build/update time;
- primitive and adjacency memory;
- number of empty or collapsed cells.

## 14. Immediate implementation checklist

1. Select or generate a small MPM sequence with particle IDs, \(p,v,F,V^0\).
2. Define a versioned trajectory file format.
3. Add analytic kinematics unit tests before modifying the renderer.
4. Refactor `PowerfoamScene.forward` to accept explicit current geometry.
5. Implement `fixed` and `volume` radius models.
6. Rebuild render adjacency from current geometry.
7. Render rigid, uniform-scale, and isochoric-stretch sequences.
8. Record the complete R0/R1 baseline before adding learned correction.

## 15. Research claims to avoid prematurely

- Do not equate sphere volume with bounded power-cell volume.
- Do not call \(r_0\sigma_{\max}\) a material-volume radius.
- Do not treat MPM interpolation support as a literal material boundary.
- Do not equate temporary render contact with persistent material adjacency.
- Do not claim prescribed-volume optimal transport applies unchanged to bounded
  Power Foam cells.
- Do not introduce a learned correction before measuring the failure modes of
  the physics-only baseline.

## 16. References

- Govindarajan et al., *Power Foam: Unifying Real-Time Differentiable Ray
  Tracing and Rasterization*, 2026: <https://powerfoam.github.io/>
- Jiang et al., *The Material Point Method for Simulating Continuum Materials*:
  <https://www.math.ucla.edu/~cffjiang/research/mpmcourse/mpmcourse.pdf>
- Sumner et al., *Embedded Deformation for Shape Manipulation*, 2007:
  <https://people.inf.ethz.ch/~sumnerb/research/embdef/Sumner2007EDF.pdf>
- Sorkine and Alexa, *As-Rigid-As-Possible Surface Modeling*, 2007:
  <https://doi.org/10.2312/SGP/SGP07/109-116>
- Gallouet and Merigot, *A Lagrangian Scheme à la Brenier for the Incompressible
  Euler Equations*, and subsequent prescribed-volume Laguerre work.
- Lévy, *Partial Optimal Transport for a Constant-Volume Lagrangian Mesh with
  Free Boundaries*, 2021: <https://arxiv.org/abs/2106.03936>
