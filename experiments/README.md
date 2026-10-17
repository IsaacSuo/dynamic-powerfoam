# Affine material transport probe

## 2026-10-08: fixed tetrahedron volume candidate (current mechanics diagnostic)

`foam_tetra_volume.py` implements a separate differentiable volume energy for
the existing rigid-cell centers. The four pair modes, masses and inertias are
unchanged. Each cell averages the current signed volumes of its incident
reference tetrahedra, then contributes
`c/2 * k_v * exp(theta_i) * W_i * (J_i-1)^2`.
Here `W_i=sum_T V_T^0/4` and `c=actual_material_volume/center_hull_volume`.
Connectivity is fixed, independent of render-time power-diagram reconstruction.
The current builder is a Cartesian equal-radius dual subdivision, not a general
weighted Delaunay/alpha-complex implementation. It covers the center convex hull,
not the whole clipped-cell material. The global factor corrects total energy,
not the missing exterior spatial distribution.

Run from the repository root in `om` (no FEM dependency installation required):

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m experiments.foam_tetra_volume_probe \
  --out output/foam_fem/tetra_volume_v1
```

Use a fresh output directory for each run. The probe reuses completed
`moduli_v1` pair stiffness matrices and actual material volumes; it does not
recompute FEM trajectories or launch training. It checks A/B reflected
subdivisions independently and scans their individual energies and energy mean.
Boundary vertices are detected from face incidence before checking volume
gradients. Face counts/orientations alone do not prove arbitrary meshes are
non-overlapping; the complete Cartesian construction provides that guarantee.

Completed results in `output/foam_fem/tetra_volume_v1/report.json`:

- 64 cells: 162 tetrahedra, 56 boundary vertices, 8 internal vertices, hull
  volume 27, material volume 93.7574186, global normalization 3.472497.
  For 0.1% uniform expansion and unit modulus, the internal force norm is
  `5.48e-15`, compared with `0.0117813` for the previous least-squares candidate.
- General finite affine motion reproduces `J=det(F)`. Rigid motion,
  volume-preserving shear and stretch have zero energy/force to roundoff.
  Nonaffine heterogeneous tests check analytic force against energy AD and
  finite differences, the stress-free Hessian against force AD, multiplier
  derivatives, rigid objectivity and zero net force/moment. Force-parameter
  finite-difference relative errors are at most `1.92e-9`. Inverted signed
  volumes are detected; the energy itself contains no inversion barrier.
- On the trained 16-cell pair model, adding a uniform volume modulus of 400
  changes the relaxed cubic projections `(C11,C12,C44)` from
  `(1143.33,46.56,262.20)` to `(1543.33,446.56,262.20)`; the FEM target is
  `(1440,480,480)`. This is a projected summary, not an isotropy assertion.
  All 16 centers are boundary vertices with prescribed affine translations in
  this static measurement, so only rotations relax. The new energy cannot
  change that rotation minimization; unchanged shear here is not a general law.
- In the 64-cell uniform pair specimen at modulus 400, shear projection changes
  slightly from 133.4858 to 133.5078. Maximum full-stiffness rotation defect over
  all 24 cubic rotations is `4.08e-6` for A or B and `2.34e-6` for their energy
  mean. Averaging these two does not restore exact cubic symmetry. Uniform
  dilation relaxation remains at roundoff. Full matrices, directional moduli
  and derivatives with respect to the uniform multiplier are saved.

This completes the isolated force/tangent implementation and static scans.
The follow-up dynamics/adjoint integration is described below. No held-out
improvement or volume-term adoption is claimed from these static checks.

### Implicit dynamics and matched four/five-mode training

`DifferentiableBonds` accepts an optional `volume_term`. The analytic center force
is evaluated at the same midpoint as the existing pair law. Invalid midpoint or
endpoint signed volumes reject the Newton trial. The fifth log multiplier and
history positions participate in the existing implicit adjoint; the original
four-mode path remains available. The reference volume tangent contributes only
to fixed solver scaling; the accepted force/Jacobian are nonlinear.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m experiments.foam_tetra_dynamics_probe \
  --out output/foam_fem/tetra_dynamics_v1
```

The completed eight-step synthetic check took about 9.2 seconds. It uses the
actual specimen and FEM load patterns, with synthetic heterogeneous target
parameters. A separate SciPy root solve using numerical Jacobians and AD of
volume energy reproduces positions within `2.19e-12`; comparison to the old
four-mode solver is within `1.42e-12`. The vanishing-volume limit matches the
four-mode trajectory. Directional full-trajectory derivatives for all five
families pass centered finite differences at two perturbation scales; the
volume direction error is `2.49e-7` at step 1e-3, and the worst error across all
checks/scales is `2.30e-6`. Refinement from 8/16 to 16/32 steps reduces trajectory
change from 1.206% to 0.319%. These are implementation checks, not material fits.

The serial independent-FEM comparison entry point is:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m experiments.foam_fem_volume_inverse \
  --out output/foam_fem/volume_inverse_v1
```

Both models use per-cell parameters (64 vs 80), common pair initial multipliers
of 1, one initialization each, multiplier bounds `[0.05,20]`, and the same
regularization coefficient per parameter (`1e-6/64`). The fifth multiplier
starts at 1 with k_v=400. Direct bounded log parameters replace tanh. SciPy
L-BFGS-B checks the projected gradient against 1e-6; loss stagnation, boundary
saturation and actual stopping messages are reported separately. Both runs have
upper budgets of 120 iterations/240 objective evaluations, using the optimizer's
line-search budget convention. Budget exhaustion is not convergence. Parameters
are saved after every accepted iteration; this entry point starts fresh runs,
not a full optimizer-state resume.

Initial time checks start from the already validated 16/32/16 substep counts,
check both initial models at common counts, and refine both together if needed.
The threshold is a 0.5% adjacent-step trajectory change. No FEM regeneration or
repeat coarse-to-fine reference ladder is performed. Both models are trained
and frozen before held-out observations are opened; final trajectories also
receive a time-step check. Those held-out cases become validation data if used
to choose the model; an independent final generalization claim needs new cases.

A temporary end-to-end smoke run using the first two existing FEM observation
intervals completed both one-iteration fits, checkpoints and all six frozen
evaluations; both capped fits correctly reported `converged=false`. The formal
serial run is now in `output/foam_fem/volume_inverse_v1/`, with sibling
`volume_inverse_v1.log` and live `status.json`. This launch is not a completed
fit or evidence of held-out improvement. It was stopped by the user at four-mode
iteration 67 (train RMSE 9.87%); the five-mode fit never started. The linearized
trainer below replaces it for this small-displacement target.

### Linearized trainer (2026-10-08)

`foam_linear_inverse.py` linearizes the same four-bond (+ optional volume) model
about the stress-free reference: `M q'' + K(theta) q = f` with the implicit-midpoint
scheme of `LinearBondsGPU.step`, the same substeps (16/32/16), loss, bounds,
regularization and L-BFGS-B budgets as `foam_fem_volume_inverse.py`.
`K(theta)` is assembled from fixed 12x12 unit bond matrices per mode and the
volume rows `B_i = grad J_i`. The gradient uses a discrete adjoint
(`dL/dK = -dt^2/2 sum_k mu_k w_k^T`, contracted per bond); one sparse
factorization per clip, no dense Jacobians. Valid only near the reference.

Checks (scratch scripts, results reproduced here):

- Tangent vs autograd Jacobian of the finite `DifferentiableBonds.trial` at the
  reference, random theta: relative error `1.6e-16` (four) / `2.4e-16` (five).
- Adjoint gradient vs centered finite differences: relative error `<4.3e-7`;
  `foam_linear_diagnose.py --check-gradient` passes for all variants (`<2.2e-8`).
- Linear vs finite trajectories (theta=0 and the running job's iteration-31
  checkpoint): per-case RMSE agrees to 4 digits; trajectory difference 0.006%-0.93%
  of target motion, largest for twist (finite rotation).
- Against the stopped finite run: the same L-BFGS-B path for 62 evaluations
  (initial objective `4.00663` vs `4.00670`; drift reaches ~5% by evaluation 60).
- At converged linear parameters, linear vs finite dense-adjoint gradients:
  cosine `>=0.996`, magnitude difference 3%-9%.
- Cost: ~0.01-0.02 s per clip with gradient vs 13-28 s for the dense finite path.

Runs (`--finite-eval` re-evaluates frozen parameters with the finite model;
the tabulated values are finite-model normalized RMSE):

| Run | Stop | down | pull | twist | side | local | 2x down |
|---|---|---:|---:|---:|---:|---:|---:|
| `linear_inverse_v1` four, 120 it | budget | 6.85% | 9.63% | 10.62% | 6.86% | 7.07% | 6.86% |
| `linear_inverse_v1` five, 120 it | budget | 7.10% | 3.97% | 10.33% | 7.15% | 6.93% | 7.11% |
| `linear_inverse_long_v1` four | converged | 6.34% | 9.18% | 3.96% | 6.39% | 6.72% | 6.35% |
| `linear_inverse_long_v1` five | converged | 6.11% | 3.29% | 8.13% | 7.11% | 5.95% | 6.12% |

In the converged runs 55% (four) / 36% (five) of log multipliers sit at the
`[0.05,20]` bounds; median fitted `k_v` multiplier is 0.16.

### Capacity diagnostics on the linearized model (2026-10-08)

`foam_linear_diagnose.py` runs one variant per call (`cell`, `bond`, `general`,
`inertia`), outputs in `output/foam_fem/linear_diagnose_v1/<run>/report.json`.
Held-out cases had already been inspected, so these are diagnostics, not
generalization results. `--cases all` trains on all six cases (capacity only).
Linear normalized RMSE, train cases unless noted:

| Run | Params | Converged | down | pull | twist | side | local | 2x down |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| cell4, bound 20 | 64 | yes | 6.34% | 9.19% | 3.92% | 6.39% | 5.86% | 6.34% |
| cell4, bound 1e4, zero / s1 / s2 | 64 | no (10k it) | 5.0-5.4% | 8.5-8.8% | 2.6-3.1% | 5.0-9.1% | 4.9-5.4% | 5.0-5.4% |
| cell5, bound 1e4, zero | 80 | no (10k it) | 5.31% | 3.25% | 4.04% | 6.07% | 4.81% | 5.32% |
| cell5, bound 1e4, s1..s5 | 80 | yes | 4.3-5.3% | 3.2-3.4% | 2.8-6.5% | 4.8-8.3% | 4.3-4.8% | 4.3-5.3% |
| bond4, bound 1e4 | 112 | no | 5.25% | 6.20% | 2.10% | 5.41% | 5.30% | 5.26% |
| bond5, bound 1e4 | 128 | no | 3.53% | 3.22% | 2.34% | 9.58% | 3.69% | 3.54% |
| cell5, all six cases | 80 | no | 4.12% | 3.80% | 6.21% | 4.14% | 3.51% | 4.13% |
| general K from cell5 s1 | 2628 | no | 0.42% | 0.86% | 0.38% | 114.4% | 7.57% | 0.42% |
| general K + rot. inertia from s1 | 2664 | no | 0.70% | 0.88% | 0.45% | 56.4% | 11.0% | 0.70% |
| general K from s1, all six cases | 2628 | no | 0.41% | 0.94% | 0.44% | 0.54% | 0.71% | 0.41% |

`general` is any symmetric PSD stiffness on the 72 free DOFs (`K=LL^T`), with
the original masses and `.4 m r^2` inertia. Started from the default `K`, it
stalled at a stationary point with twist at 92.9% (`general_badinit`); restarted
from the cell5 s1 stiffness it fits well. Observations:

- A linear rigid-cell model with the existing masses/inertia can reproduce all
  six FEM motions within 1% (general K, all cases). The rigid-cell representation,
  center observation and inertia proxy are not the limiting factor here.
- Every bond parameterization stays at roughly 2-6% per case even when trained on all six
  cases, so the remaining error is a structural limit of the pair/volume law,
  not optimization budget, bounds or initialization. Bending (down) is the most
  persistent residual.
- Unconstrained K trained on three motions fails on the unseen side load
  (56%-114%): the training motions do not constrain y-direction response.
- Five-mode seeds reach similar training loss but different parameters (median
  `kb` multiplier 0.06-4.25 across seeds); held-out side varies 4.8%-8.3%.
  Fitted multipliers are not identifiable material constants. Median `k_v`
  stays low (0.11-0.44) for every seed.
- `strong_down` equals `down` within rounding, as expected for linear dynamics.

Follow-up variants (same directory; all 10k-iteration runs were still decreasing):

| Run | Params | Cases | down | pull | twist | side | local | 2x down |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| pair (12x12 PSD per bond) from cell4 s2 | 2184 | train | 0.73% | 1.86% | 0.50% | 99.9% | 13.9% | 0.73% |
| pairobj (objective 6x6 per bond) from cell4 s2 | 588 | train | 0.91% | 2.76% | 0.67% | 92.1% | 1.45% | 0.91% |
| pair, all six cases | 2184 | all | 0.73% | 2.22% | 1.20% | 0.76% | 1.04% | 0.72% |
| pairobj, all six cases | 588 | all | 0.84% | 2.85% | 1.30% | 1.22% | 1.25% | 0.84% |
| beam (E,G per cell), zero / s2 (s1 stuck: pull 88.6%) | 32 | train | 9.4-10.0% | 13.7-15.5% | 8.3-8.9% | 9.5-10.1% | 9.4-10.0% | 9.4-10.0% |
| beam + volume, three inits | 48 | train | 8.9-9.8% | 10.9-14.3% | 9.1-9.5% | 10.1-25.9% | 8.8-9.6% | 8.9-9.8% |

- A pair-local, objective interaction (pairobj) fits all six cases to 0.8%-2.9%,
  close to the general-K ceiling, so multi-body terms are not required for fitting
  this target; the restrictive parallel-bond form is.
- Trained on three cases, the flexible variants fail on `side`: pair predicts 0.9%
  of the true motion (y-direction stiffened), pairobj and general K move with the
  right scale but correlation 0.43-0.46 with the target. The 5-mode spring predicts
  1.03x the motion with correlation 1.00. Section symmetry of the parallel bond is
  what transfers z-bending to y-bending.
- The objectivity constraint helps `local` (1.45% vs 13.9%).
- Timoshenko beam bond == parallel bond with contact at the midpoint and
  `ks = 12EI/((1+Phi)L^3)`, `kb = EI/L` (beam theory ties shear to bending).
  Fitted parallel bonds violate `ks <= 12 kb/L^2` on 13-22 of 28 bonds, and the
  beam variant doubles the error. Not a candidate.
- Decomposing pairobj (all cases) bonds into axial/shear1/shear2/twist/bend1/bend2:
  coupling part is ~47% of the matrix norm (parallel bond: 0), section anisotropy
  ranges 0.01-56x, and twist/bending stiffness is ~40-70x the fitted parallel bond.
  With 588 parameters and six motions these values are not identifiable; they only
  indicate candidate missing mechanisms (shear-bending coupling, anisotropic sections,
  stiffer rotational coupling) to test as few-parameter structured terms.

Learnable contact shift (`--variant lever`): the parallel bond with its contact
point moved by `s` along the bond axis (|s| <= 0.4 L). With section symmetry and
no chirality this is the only allowed shear-bending coupling; coupling strength is
`ks*s` and bending gains `ks*s^2`. Multipliers bound 1e4; 10k iterations unless noted.

| Run | Shifts | Cases | down | pull | twist | side | local | 2x down | shifts at bound |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| four, from cell4 s2 | per bond | train | 3.92% | 8.23% | 2.69% | 4.95% | 4.10% | 3.92% | 8/28 |
| four, zero init | per bond | train | 4.47% | 8.45% | 2.79% | 34.8% | 4.79% | 4.48% | 7/28 |
| four, from cell4 s2 (converged) | global (0.04) | train | 4.94% | 8.71% | 3.11% | 4.95% | 4.83% | 4.95% | 0/1 |
| five, from cell5 s1 | per bond | train | 2.49% | 3.30% | 2.33% | 4.09% | 2.58% | 2.49% | 8/28 |
| five, zero init | per bond | train | 3.64% | 3.36% | 2.13% | 4.46% | 3.86% | 3.64% | 10/28 |
| five, seed 2 | per bond | train | 4.14% | 3.22% | 2.04% | 4.79% | 4.92% | 4.15% | 13/28 |
| five, from cell5 s1 | global (-0.08) | train | 4.34% | 3.29% | 2.76% | 5.10% | 4.11% | 4.35% | 0/1 |
| five, from cell5 s1 | per bond | all | 2.13% | 3.62% | 2.34% | 2.18% | 2.17% | 2.13% | 6/28 |

- One shared shift is negligible. Per-bond shifts close roughly 40%-65% of the gap
  between the five-mode spring and pairobj (train-only: combined 3.55% -> 2.74%
  vs 1.72%; all cases: ~4.4% -> ~2.5% vs ~1.5%), keeping side at 4.1%-4.8% for
  the five-mode runs. The four-mode zero-init run broke side (34.8%), so shifts
  alone do not guarantee transfer.
- Pull stays at 3.2%-3.6% for every five-mode variant and 8.2%-8.7% for four-mode:
  per-bond shifts do not address it.
- Many shifts saturate at 0.4 L, i.e. near a cell center; the bound, not the data,
  sets those values, and the physical reading as a contact location is doubtful.

Decomposition and stability (five-mode, bound 1e4):

- `--lever-mode bend` keeps the `ks*s^2` bending increase and drops the coupling
  (zero shift is a stationary point there, so it starts at s=0.1L). Zero init:
  combined train RMSE none 4.29% -> bend-only 3.81% -> full 3.12%; all six cases
  ~4.4% -> ~3.7% -> ~2.5%. The coupling supplies ~60% of the gain. Bend-only from
  the cell5 s1 start fell into a twist trap (96%).
- `--section actual` (`foam_face_sections.py`: real power-face second moments,
  area/centroid within 0.16%/4e-4 of CAD; principal bending 0.85-1.26x the
  equivalent circle) leaves every matched run unchanged within ~0.1 point.
- Seven full-lever runs (various inits, s0=0 or 0.1): combined train 2.66%-3.29%;
  held-out side 2.6%-8.8%, local 2.5%-4.8%, 2x down 2.4%-4.2%; max held-out /
  train 1.0-2.7. The best run (`lever5_bond_from5_s01`, 2.66% / 2.6, 2.5, 2.4%)
  is not representative. Training fit improves reliably; side transfer does not.
- Pull error (best lever run) is 84%-91% axial, not lateral contraction (lateral
  motion ratio 0.99). It concentrates in the loaded end layer (3.6% vs 1.1%-1.7%)
  and in the first five frames of loading, where the model end moves early/too much.
- Phase split (load pulse = frames 1-8 of 24, same whole-clip normalization):
  pull loading/free 4.4%/2.5% (spring5), 4.5%/2.4% (best lever), 1.3%/0.7% (general
  K); twist 3.9%/2.0%, 3.1%/1.4%; down roughly equal in both phases. Loading frames
  carry ~60% of the pull squared error. Even the general-K ceiling shows the 2x
  loading excess, so this part is attributed to rigid cells receiving the surface
  load at their centers, not to the bond law.

Per-cell size parameterization (`--lever cell`): one `sigma_i` per cell, bond shift
`s=(sigma_i-sigma_j)/2` toward the smaller cell (power-weight analogue), 16 instead
of 28 parameters. Same six inits as the per-bond runs (20k iterations):

| Init | per-bond: train / side | per-cell: train / side / local / 2x down |
|---|---|---|
| from cell5 s1 | 2.74% / 4.1% | 2.74% / 2.7% / 2.7% / 2.6% |
| zero | 3.11% / 4.5% | 2.90% / 3.1% / 3.0% / 3.0% |
| seed 1 | 2.81% / 6.7% | 2.88% / 3.0% / 2.8% / 2.7% |
| seed 2 | 3.24% / 4.5% | 2.93% / 13.9% / 3.1% / 3.3% |
| seed 3 | 3.29% / 8.8% | 2.55% / 7.1% / 2.6% / 2.4% |
| seed 4 | 2.84% / 6.1% | 2.66% / 17.7% / 2.6% / 2.5% |
| all six cases (from cell5 s1) | ~2.5% mean | 1.9 / 3.9 / 2.2 / 1.9 / 2.2 / 1.9% (~2.4% mean) |

Training and local/2x-down improve slightly with fewer parameters, but side is not
stabilized: y-bond shifts depend on y-neighbour sigma differences, z-bond shifts on
z-neighbour differences, so the y direction remains unconstrained. Across all
variants, side transfer is reliable only when structure forces y/z equality; the
three training loads never excite y motion.

### FEM reference solver: CHOLMOD (2026-10-08)

`foam_fem_reference.py` now factors the SPD systems (`K` for calibration, including
the eigsh shift-invert, and `M + dt^2/4 K` for time stepping) with CHOLMOD sparse
Cholesky (scikit-sparse 0.5.0, installed in `output/fem_dependencies`, linked to the
system libcholmod). SuperLU is no longer used. The system libcholmod links the
reference BLAS; preload MKL from the base conda install for fast factorization:

```bash
LD_PRELOAD=$HOME/anaconda3/lib/libmkl_rt.so.2 MKL_THREADING_LAYER=GNU MKL_NUM_THREADS=4 \
OMP_NUM_THREADS=4 PYTHONPATH=output/fem_dependencies:. \
python -m experiments.foam_fem_reference --geometry output/foam_fem/mesh_edge007/geometry.npz \
  --config output/foam_fem/reference_q050_v3/config.json --min-substeps 16 --out <dir>
```

edge007 system (101,280 free DOFs, 5 RHS): factor SuperLU 275 s -> CHOLMOD 20 s
(reference BLAS) -> 6.8 s (MKL, 1 thread) -> 4.8 s (MKL, 4 threads); per step
1.37 s -> 0.20-0.23 s (memory bound; threads barely help). Verification run
`output/foam_fem/reference_edge007_cholmod/` with the shared q050_v3 config: all
trajectories match the original edge007 targets to `8.9e-16` max absolute
difference (normalized `2.9e-14`); loads and dt identical; rollouts 723 -> 176 s
(16 substeps) and 1144 -> 285 s (32 substeps). Additional load cases share the
factorization, so adding columns costs little.

All mesh levels reuse the q050_v3 calibration (period `1.0199`, dt, pulse, force
amplitudes) so trajectories are comparable across meshes. The edge007 lowest
period computed on its own mesh is `1.0347` (`reference_edge007_cholmod_recalibrated`,
an accidental recalibration run, stopped after the 16-substep rollout). The fitted
DEM models' lowest period (~1.033-1.035) therefore matches the edge007 continuum;
the earlier 1.5% "period mismatch" came from comparing with the coarse-mesh value.
New target cases must reuse the same config (dt, pulse) to stay comparable.

### Fresh held-out loads `fresh_v1` (2026-10-08)

`foam_fem_reference.py --case-set fresh_v1` (regression: the original case set
reproduces `reference_q050_v3` to 8.9e-16). Loads, fixed before evaluation: end
oblique (y,-z)/sqrt2, end quarter patch +y, end zero-force couples about y and
about z (`t_x = z`, `t_x = y`), x=2-layer top surface -z, x=2-layer -y side +y.
Timing from q050_v3; amplitudes by the same static rule (0.5% length x stored
reduction). edge007 targets in `output/foam_fem/reference_edge007_fresh_v1/`:
edge010-vs-edge007 1.05%-1.13%, 16-vs-32 substeps <=0.074%, peak strain 1.5%.

`foam_linear_diagnose.py --evaluate-run RUN --extra-reference DIR` evaluates a
finished run's frozen parameters (writes `eval_<dir>.json`). Train-only runs,
range over inits (fresh RMSE over the six cases; z = end_bend_y, mid_top_down;
y = the four cases with y bending):

| Model | n | train | old side | fresh z | fresh y | fresh all |
|---|---:|---|---|---|---|---|
| cell4 | 4 | 5.9-6.8% | 5.0-9.1% | 9.5-10.3% | 7.6-10.8% | 8.2-10.6% |
| cell5 | 6 | 3.6-4.9% | 4.8-8.3% | 8.3-9.0% | 6.4-9.6% | 7.1-9.3% |
| cell5 actual section | 3 | 3.6-4.9% | 4.9-5.8% | 8.3-8.7% | 6.7-7.4% | 7.3-7.9% |
| bond4/5 | 2 | 3.1-4.8% | 5.4-9.6% | 7.4-9.6% | 7.8-10.7% | 8.4-9.7% |
| beam / beam+volume | 6 | 9.7-51.8% | 9.4-25.9% | 21-35% | 17-31% | 18-30% |
| lever per bond | 8 | 2.7-3.3% | 2.6-8.8% | 6.3-11.6% | 5.1-10.4% | 5.7-10.1% |
| lever per-cell size | 6 | 2.5-2.9% | 2.7-17.7% | 6.0-7.9% | 5.3-18.0% | 5.9-15.2% |
| lever + reg 1e-3 | 2 | 2.9-3.4% | 6.8-6.9% | 7.8-8.7% | 7.9-10.5% | 7.9-10.0% |
| lever geometric (global) | 3 | 3.0-3.8% | 5.4-9.5% | 6.1-8.9% | 7.4-9.7% | 7.9-8.6% |
| lever geometric (per cell) | 2 | 2.8-2.9% | 3.4-8.1% | 6.1-8.5% | 6.3-8.1% | 7.1-7.5% |
| pairobj / pair (train only) | 2 | 1.2-1.7% | 92-100% | 5.1-8.0% | 87-93% | 71-76% |
| general K (+inertia) | 2 | 0.6-0.7% | 56-114% | 26-36% | 67-123% | 58-102% |

Runs trained on all six old cases (capacity references): lever per bond 4.9%,
levercell 5.9%, cell5 7.2%, pair 8.4%, pairobj 10.4%, general K 11.3% on fresh.

### Thick block 4x4x4 (2026-10-09)

Geometry/targets: see RESEARCH_ROUTE item 22. Training as for the bar (three
training loads, 20k-iteration cap, zero init unless noted); frozen parameters
evaluated on test_side/test_local and fresh_v1 (`eval_block_reference_edge007.json`).

| Run (`block_diagnose_v1/`) | train | side | local | fresh (6) |
|---|---:|---:|---:|---:|
| cell4 zero / s1 | 80.9% / 80.4% | 99.7% / 99.3% | 98.7% / 97.6% | 98.9% / 99.1% |
| cell5 zero / s1 | 3.45% / 3.56% | 5.1% / 6.6% | 14.1% / 14.6% | 12.6% / 12.9% |
| lever per bond | 3.06% | 26.8% | 14.3% | 23.1% |
| lever per-cell size | 3.13% | 11.7% | 14.1% | 16.2% |
| lever geometric global / per cell | 3.59% / 3.29% | 7.9% / 7.2% | 14.6% / 14.4% | 12.9% / 15.0% |
| uniform five-mode (`block_fused_v1/`, 3k cap) zero / s1 | 98.2% / 13.1% | 98.9% / 9.3% | 98.9% / 16.4% | 99.1% / 21.5% |

The four-mode model fits pull but leaves bending almost rigid (bending multiplier
x57, predicted bending motion 13% of target): with axis bonds on a cubic lattice
only the volume term supplies C12. Uniform multipliers cannot fit the training
loads of the homogeneous block; the per-cell spread mostly corrects truncated
boundary cells. Runs stopped mid-way are kept as `*_stopped`. Fused-lasso runs
(`block_fused_v1/`): leave-one-training-load-out validation is 76%-108% for every
weight (each training load is the only one exciting its deformation mode), so it
cannot select a weight; all-three-load fits at 1e-5/1e-4/1e-3 start at shared zero
and stall on the stiff no-motion plateau (train 81%-99%, held-out ~99%). Inconclusive.

## New: complete scene-forward coordinate check

The small [full-model rigid diagnostic](FULL_MODEL_RIGID_RESULTS.md) uses
`PowerfoamScene.forward()` with nonzero local displacement, varying local color
and Spherical Voronoi appearance. It checks simultaneous scene/camera coordinate
changes and omission controls for 1 and 9 cells. Run
`python -m experiments.full_model_rigid_probe` in `om`.
Its parameters are hand-authored; it is not a trained reconstruction experiment.

The follow-up [single-cell shear diagnostic](SINGLE_CELL_SHEAR_RESULTS.md) uses
the original forward for an affine-pullback reference and fits local sites/heights.
It reports interior color/surface-entry errors separately from fixed-sphere
coverage, including a held-out view. Run `python -m experiments.single_cell_shear`.

## Earlier geometry probes

> **Scope correction (2026-09-25):** The earlier experiments below were written during our
> exploration; they are not the paper's experiments or a full-method reproduction.
> Calling the original rasterizer does not reproduce `PowerfoamScene.forward()`:
> the probes simplify or bypass local-coordinate generation, displacement,
> directional appearance and joint training. Cell-label transport is our extra
> material hypothesis. See [the reading notes](../POWERFOAM_READING_NOTES.md).

The latest spacetime probe is documented in [SPACETIME_FOAM.md](../SPACETIME_FOAM.md).
Run `python -m experiments.spacetime_probe` in `om` to check native 4D competition,
render its 3D slices using the original rasterizer, and inspect cell handoffs.

The current anisotropic renderer experiments are documented in
[ANISOTROPIC_FOAM.md](../ANISOTROPIC_FOAM.md). Run it with
`python -m experiments.shared_affine_render` (shared F) or
`python -m experiments.varying_metric_render` (different F per cell) in the `om`
environment. See [per-cell results](VARYING_METRIC_RESULTS.md). The
experiments below remain the original spherical-representation baselines.

Run from the repository root using the existing local `om` environment:

```bash
conda activate om
python -m experiments.affine_probe
python -m experiments.affine_probe --uniform-radii --no-gpu --out output/affine_probe_uniform
```

On this machine its interpreter is `/home/fangsuo/anaconda3/envs/om/bin/python`.
Verified imports: Python 3.10.19, PyTorch 2.9.1 / CUDA 12.8, Warp 1.10.0.
The GPU is an RTX 5070 Ti Laptop GPU. No dependency installation is needed.
`--no-gpu` runs only the NumPy/Matplotlib diagnostics, without importing Warp.

## Scope

This is a deterministic, frozen-appearance experiment, not an MPM simulation,
trained reconstruction, or performance benchmark. It uses 121 jittered power
sites on a planar patch in 3D, eight texture sites per cell, zero displacement,
and view-independent texel colors sampled from a checkerboard. All transforms
keep the surface in the XY plane. Nothing is optimized between frames.

The reference appearance is the actual soft texture represented by these cells,
not the sharp checkerboard used to initialize texel colors. This prevents initial
representation error from being confused with motion-induced error.

For every reference sample X we compute its original bounded-power owner, move
it to x = F X + t, and recompute ownership from the current centers and radii.
Membership uses **all** power-distance constraints and the sphere bounds in
float64. Reference samples outside all spheres are excluded; their fraction is
recorded. Each transform uses the exact same reference samples.

Radius rules:

- `fixed`: R0, r = r0.
- `volume`: R1, r = r0 det(F)^(1/3).
- `cover`: the alpha=1 endpoint of roadmap R2, r = r0 sigma_max(F).

Cases: identity; 32-degree rotation plus translation; uniform scale 1.6;
isochoric stretch diag(2, 0.5, 1); XY shear with F[0,1] = 0.9.
The additional uniform-radius run removes radius heterogeneity as a confounder.

## Metrics and checks

| Metric | Definition |
| --- | --- |
| `support_miss_pct` | Advected samples outside **every** current bounded cell / valid reference samples |
| `owner_switch_pct_covered` | Changed owner / currently covered reference samples |
| `identity_loss_pct` | Missing or changed owner / valid reference samples |
| `texture_metric_mae` | Original owner retained artificially; evaluate moved texel sites with current world-space distance and radius; RGB MAE against reference |
| `ownership_only_mae_covered` | New owner, but texture queried with inverse deformation in canonical coordinates; RGB MAE on covered samples |
| `combined_mae_covered` | New owner and current world-space texture metric; RGB MAE on covered samples |
| `combined_mae_including_gaps` | Combined error on all valid reference samples, with black for missing support |
| `pullback_fixed_owner_mae` | Reference-owner texture queried after inverse deformation; must be numerically zero |
| `mean_degree`, `max_degree` | Brute-force sphere-overlap graph degree, over all 121 cells |

Colors are in [0,1]. The isolated MAEs are diagnostic interventions, not additive
components of total error. Artificially retaining an owner does not assert that
the sample is still inside its current cell. A local inverse map does not repair
ownership changes; the fixed-owner pullback is an oracle, not an implemented
general dynamic renderer.

Assertions verify a two-cell analytic ownership counterexample, inverse-map
consistency, rigid equivariance for all radius rules, uniform-scale equivariance
for R1/R2, and zero support misses for the enclosing-radius rule. GPU execution
also checks its Cech graph against the exact sphere-overlap matrix, nonempty and
finite renders, and identity-frame agreement with the analytic surface reference
on opaque pixels (MAE < 0.015).

## Artifacts

Default output directory: `output/affine_probe/` (Git-ignored).

- `material_comparison.png`: fixed reference-coordinate views; exposes texture
  changes without the visual distraction of the overall deformation.
- `ownership_comparison.png`: orange = different cell; blue = missing support;
  gray = retained owner; black = invalid reference sample.
- `gpu_comparison.png`: analytic advected surface reference beside actual
  `Rasterizer.visualize` images with freshly rebuilt GPU Cech adjacency.
- Individual GPU RGB/alpha PNGs, `metrics.csv`, `results.json`, and the reference
  scene arrays in `reference_state.npz`.

GPU images use the original renderer, density 1000, a fixed perspective camera,
and explicit transformed tensors. They include finite optical thickness and
cell-side effects; the analytic reference is an opaque transported surface, so
it is not an exact volume-rendering ground truth at silhouettes. Quantitative
material metrics are computed analytically, not extracted from the GPU images.
This experiment does not validate ray tracing, arbitrary 3D deformation,
spatially varying F, contact, or long-sequence temporal stability.

## Fitting positions and radii

```bash
conda activate om
python -m experiments.fit_affine_cells
```

This second experiment retains the same bounded-power parameterization and
optimizes render-site positions and radii against known affine material regions.
It compares the volume-radius baseline, radius-only optimization, and joint
optimization with center offsets capped at 5%, 10%, and 20% of reference spacing.
Spacing is each site's mean distance to its four nearest reference neighbors.
Offsets are constrained in XY; z remains zero for this planar configuration.
Radii are constrained to [0.5, 2] times the volume-radius prior for every fit.

Training uses 16,384 fixed 3D samples, half uniform in a padded reference box and
half from points near reference cell/background or cell/cell boundaries. A
separate random set of 65,536 **uniform** 3D samples is used only for reporting.
A further 256x256 surface grid matches the first experiment's evaluation region.
Both sampling domains are advected by the known affine transform. The two test
deformations are isochoric stretch and shear. These are independently fitted
frames, not a learned mapping or a temporally regularized sequence.

Let phi_i = (||x-p_i||^2-r_i^2)/median(h)^2. A labeled material sample i pays
relu(phi_i - min(0, min_{j!=i} phi_j))^2. An exterior sample pays
relu(-min_j phi_j)^2. The loss sums the separate interior/exterior means plus
0.01 * mean(||offset/h||^2) and 0.01 * mean(log(r/r_prior)^2). All cells are
considered, without an approximate neighbor graph in the fitting loss.

The optimizer uses 1,000 full-batch Adam steps with a decaying learning rate,
hard projection of parameter bounds, and selection by the best **training**
objective. Test data never selects a checkpoint. This squared violation loss
measures the magnitude of geometric constraint violations; it is not the same
as minimizing the percentage of mislabeled samples. Report both geometry errors
and exterior false occupancy rather than assuming lower loss means improvement
on every metric. Training/test percentages also sample different distributions.

Gradient correctness is checked against finite differences. Additional checks
cover affine oracle labels, exact uniform scaling, hard bounds, and the GPU
adjacency graph when rendering. The GPU renderer remains unchanged.

All appearance parameters are frozen. For the standard GPU panels, texel sites
move with F and then with their fitted cell offset; heights stay zero. Thus
radius/position changes can still alter the world-space texture interpolation.
A separate canonical-texture metric isolates ownership changes. No image loss
is optimized, and good geometric fitting does not promise a lower texture MAE.

Artifacts are in `output/affine_fit/`: `results.json`, `metrics.csv`, fitted
parameters per case, individual GPU images, ownership/GPU comparison panels,
and convergence curves. `--no-render` skips GPU images but still fits on CUDA.

## TLSPH mechanics and Power Foam rendering

```bash
conda activate om
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m experiments.tlsph_probe
python -m experiments.tlsph_irregular
python -m experiments.tlsph_render --out output/tlsph
python -m experiments.tlsph_render --out output/tlsph_irregular
```

This is an independent implementation of corrected total-Lagrangian SPH with
compressible neo-Hookean elasticity and the radial hourglass force from
[Ganzenmuller (2015), equation 32](https://arxiv.org/abs/1410.7221).
It uses reference volumes and a 3D cubic spline kernel, and independent 3D test
fixtures; it is not an exact reproduction of the paper's benchmarks. The Wu
non-hourglass formulation was read for comparison but is not implemented.

`tlsph_core.py` fixes material neighbors and gradient corrections in the initial
configuration. Current center positions determine deformation gradients, stress,
and pair forces. Material masses remain fixed. No grid transfer or independent
cell-vertex degrees of freedom are used. SPH smoothing length is separate from
the Power Foam radius. The renderer rebuilds its own adjacency every frame.

`tlsph_probe.py` checks affine deformation-gradient reproduction, rigid-motion
invariance, force/torque balance, and the elastic force against finite differences
of energy. The affine check is not a complete boundary-traction patch test.
For the rectangular reference beam, the small-load deflection differs from the
Timoshenko estimate by 16.6% at 272 particles and 7.4% at 900 particles with
hourglass strength 10. On the coarse beam, strength 0 gives 53.5% error and
strength 50 gives 33.6% error. These are comparisons with an approximate beam
model, not proof of convergence to the exact 3D solution or a universal choice
of stabilization strength.

The nonlinear loaded state is solved with a finite-difference Newton tangent
before releasing the load. Velocity Verlet simulates 1.6 time units without
damping. Halving the time step changes the sampled tip displacement by at most
0.080% of its initial amplitude (samples are at the same nominal frame times,
rounded to integration steps). The hourglass force is not assumed conservative:
its measured work is tracked explicitly. Elastic plus kinetic energy changes
by about 20% at the end of this run; the balance after subtracting hourglass
work has maximum relative error 1.3e-6. This is not an energy-conservation claim.

`tlsph_irregular.py` uses 272 jittered centers and varying radii. Each entire
initial bounded power cell is explicitly declared solid; its numerically
integrated volume sets its fixed mass and integration weight. This does not
infer physical occupancy from a learned dipole field. The test reaches loaded
equilibrium and completes release with positive det(F). It is a small bending
test, not validation of large deformation, fracture, contact, or long-term stability.

`tlsph_render.py` passes these trajectories through the original
`PowerfoamScene.forward`, including dipole/detail/SV appearance and rasterization.
The synthetic blue/orange material is hand authored, without image fitting.
Polar rotation updates local frames; radius uses the PowerSim-style baseline
`r = r0 * det(F)^(1/3)`. Radius is not a mechanical degree of freedom in this
baseline, and current cell boundaries do not feed back into the TLSPH forces.
Consequently this implements an initial-volume-aware TLSPH baseline, not a
fully coupled center/radius material model.

`cell_volumes.json` compares rebuilt bounded-cell volumes against each initial
cell volume times det(F), using actual power halfspace membership. Sobol
integration is checked against isolated-sphere and two-sphere analytic volumes;
the loaded frame is refined to 262,144 points per cell. Volume agreement alone
does not establish that a rebuilt cell contains the same material points.
Use `--volumes-only` to recompute these diagnostics without rendering.

At the loaded frame, refined summed absolute cell-volume mismatch divided by
total target volume is 0.239% for the regular fixture and 0.108% for the
irregular fixture with radius scaling. Keeping radii fixed gives 0.128% and
0.064%, respectively. The corresponding rest-volume change between 65,536 and
262,144 samples is 0.031% and 0.068%. Thus the irregular comparison is still
close to integration uncertainty, and this small-deformation test provides no
evidence that cube-root determinant scaling improves cell-volume agreement.
For the regular fixture, rendered support volume is 0.8625 whereas the
rectangular mechanical reference volume is 0.5; use it as a mechanics control,
not as a geometry-matched physical foam. The irregular fixture uses its actual
initial support volume, 0.8583, for mechanics.

Outputs in `output/tlsph/` and `output/tlsph_irregular/` include `results.json`,
`trajectory.npz`, `cell_volumes.json`, `render.json`, `powerfoam.gif`, and PNG
frames. `output/tlsph/mechanics.png` contains displacement and work-balance plots.

## Joint radius solve on the frozen TLSPH trajectory

```bash
conda activate om
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.tlsph_radius_fit
```

This compares fixed radii, `r0 * det(F)^(1/3)`, and jointly solved radii on all
65 saved frames of the irregular beam. Centers, F, mass, and motion are identical
between methods. The target for cell i is exactly its saved mechanical reference
volume times det(F_i); targets are not silently recalibrated to the fitting
quadrature. The original reference volumes have integration error, reported
separately in `saved_reference_volume_vs_independent_geometry`.
`reference_sensitivity.json` additionally repeats the loaded-frame comparison
with refined geometric reference-volume targets while keeping the same center/F
trajectory. That is a target sensitivity diagnostic, not a second mechanics run.

`power_cell_volume.py` integrates radial boundary distances from each generator:
the first intersection with its bounding sphere or any power halfspace determines
the radius in that direction. It integrates rho^3/3 over solid angle and
differentiates with respect to **all** squared radii. Off-diagonal derivatives
capture volume exchange between adjacent cells. Newton updates solve this
coupled Jacobian with a line search; there is no soft occupancy or image loss.

The implementation requires each generator to lie strictly inside its own cell.
It explicitly rejects other configurations, rather than treating them as empty
cells. Radius bounds are [0.5, 2] times the original radii. Positive interior
clearance implies a nonempty cell; successful runs report bounds and independent
volume checks. This restriction is appropriate to the current fixture and does
not establish a solver for arbitrary learned Power Foam configurations.

Fitting uses 16,384 angular Sobol directions, seed 19. Independent validation uses
131,072 directions, seed 901, on seven frames; the loaded frame is checked again
with 262,144 directions, seed 1901. Checks include analytic two-sphere volumes
and Jacobians, finite differences, uniform scaling, and recovery of known radii.
Frame-to-frame continuity uses only the previous solution as initialization;
there is no temporal smoothing penalty. Radius increments and second differences
are reported instead of assuming that convergence implies smooth motion.

Outputs are under `output/tlsph_radius_fit/`: `radii.npz`, `results.json`,
`comparison.png` (volume errors and radius changes), and `comparison.gif`
(fixed/scaled/joint from left to right, through the original renderer).
`--resume` reuses completed fits with a matching source hash; `--no-render`
skips rendering. Fitted radii do not feed back into stress or motion, and volume
agreement does not establish preservation of the same material points.

Observed on the saved 272-cell, 65-frame small-bending trajectory (seven
independently evaluated frames):

| Radius method | Mean volume error | Worst frame volume error |
| --- | ---: | ---: |
| Fixed | 0.09072% | 0.10158% |
| Cube-root determinant | 0.10984% | 0.13630% |
| Joint volume solve | 0.00355% | 0.00369% |

Here volume error is `sum(abs(V_cell - V_target)) / sum(V_target)`, not the
signed error of total volume. The worst individual-cell relative error across
validation frames is 0.02812% after fitting. All 65 solves take two Newton
updates. Maximum radius change from rest is 0.18658%, and maximum change between
adjacent stored frames is 0.02502% of rest radius, with no smoothing penalty or
bound hits. Every fitted generator remains inside its cell, so no cell vanishes.

The saved reference-volume integration discrepancy is 0.07684%. To check that
the result is not solely correction of this discrepancy, the refined-reference
loaded-frame diagnostic gives fixed/scaled/joint errors of 0.05489%, 0.10018%,
and 0.00365%, respectively. This supports solving volume constraints for this
fixture; it does not establish feasibility or temporal smoothness for arbitrary
large deformation or for generators lying outside their cells.

## Material identity after radius fitting

```bash
conda activate om
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.tlsph_material_probe
```

This freezes the existing trajectories and radius fits. It places passive labeled
integration points inside each initial bounded cell and asks whether a moved
point is still owned by that cell, owned by another cell, or outside every cell.
These are not additional mechanical particles and do not transfer mass to a grid.
Hard power distances determine ownership; no appearance or RGB loss is used.

Interior motion is an explicit modeling assumption, because the current TLSPH
implementation supplies only center positions and center deformation gradients:

- `local_affine`: `x_i + F_i * (X - X_i)`. Matches the center and its estimated
  F, but separate cell maps need not agree across a shared initial boundary.
- `continuous_mls`: affine-reproducing moving least squares of center
  displacements in reference coordinates, with the same cubic kernel, h=0.15,
  and reference-volume weights. It is continuous where its moment matrix is
  nonsingular, but does not interpolate every center exactly. Its maximum center
  discrepancy is 0.000282, or 0.188% of h, and its gradient differs from the
  TLSPH estimator. Neither extension is declared continuum ground truth.

Reference markers use angular/radial Sobol quadrature, weighted by the cubed
radial boundary distance and normalized to each cell's saved reference mass.
Plain counts would bias this measurement. Runs use 2,048 and 8,192 points per
cell (557,056 and 2,228,224 points total) at seven saved times. An independent
seed repeats the loaded-frame measurement. Rest/translation/rotation/uniform
scale controls have zero ownership loss. MLS reproduces affine motion to about
3e-15, and spatially culled ownership matches exhaustive classification.

Mean fraction of initial material mass outside its original cell across the
seven evaluated times (other-cell ownership plus outside-all-cells):

| Radius method | Local affine motion | Continuous MLS motion |
| --- | ---: | ---: |
| Fixed | 0.05493% | 0.08182% |
| Cube-root determinant | 0.06801% | 0.08476% |
| Joint solve, saved reference volumes | 0.08743% | 0.10846% |
| Joint solve, refined geometric reference targets | 0.06073% | 0.08438% |

The last row isolates the influence of inaccurate initial-volume measurements:
reference geometry is integrated using 262,144 angular directions, radii are
refitted on the seven frames, and motion/F/marker masses remain unchanged.
This is a target sensitivity check, not a new mechanical simulation. Correcting
reference targets explains some of the increased label changes in the original
joint solve. For the refined targets, switched-to-other-cell fractions are
0.05270% / 0.06391%, and outside-all-cells fractions are 0.00803% / 0.02047%.

The largest coarse/fine difference in the three original radius methods is
0.00824 percentage points, and the largest loaded-frame independent-seed
difference is 0.00637 percentage points. Small differences between methods
should not be overinterpreted. Both motion extensions still show nonzero label
changes; matching cell volumes does not by itself demonstrate exact material
region preservation. These are reference-to-frame measurements, not evidence
of accumulating drift over repeated cycles or of long-term instability.

Outputs under `output/tlsph_material/` include `results.json`, convergence runs,
per-cell retained/switched/outside fractions, `identity_loss.png`, and
`loaded_identity.png`. The last image is a thin-slice point diagnostic, not an
image from the Power Foam renderer. `--resume` reuses completed measurements
after checking stored source hashes.

## Larger deformation from actual applied loads

```bash
conda activate om
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.tlsph_load_sweep
```

This increases the total downward tip force on the same irregular beam through
0.16, 0.4, 0.8, 1.6, and 3.2. All displayed states solve the nonlinear TLSPH
force equations with the original material, masses, fixed material neighbors,
clamp, and stabilization. They are not amplified versions of an existing
displacement field. Newton iteration uses a finite-difference tangent and line
search. Failed large load increments are bisected, with attempts retained in
the log. The largest-load equilibrium is independently revisited using smaller
increments from the 1.6 state. Force residuals, support-reaction balance, det(F),
principal stretches, rotations, and position differences are recorded.

These are **static force equilibria**, not a time integration or a demonstration
of dynamical stability. `trajectory.npz` contains `loads`; its compatibility
field `times` also contains loads for shared helper functions. The renderer
explicitly labels frames `load`, uses one camera encompassing all five states,
and depicts a sequence of loading equilibria. Source trajectories are unchanged.

Each equilibrium is evaluated with fixed radii, cube-root determinant radii,
jointly solved radii against the saved mechanical reference volumes, and the
refined-geometric-reference target sensitivity control. Volume fitting and
validation retain independent quadratures (2^14 seed 19 / 2^17 seed 901), with
reference geometry evaluated at 2^18 seed 2901. The extra target control changes
only radius targets, not the masses or mechanical solution.

Passive material-point tests repeat at 2,048 and 8,192 points per cell, with an
independent seed at the largest load. Local-affine and continuous-MLS extensions
are still explicit assumptions about interior motion, not exact physical truth.
The MLS center-position discrepancy and sampled gradient determinants are
audited at the larger deformations. Radius fitting still does not feed back into
forces. No fracture or contact model is introduced.

Outputs under `output/tlsph_load_sweep/` include `mechanics.json`, `geometry.json`,
`results.json`, `load_comparison.png`, `load_frames.png`, and
`render/powerfoam.gif`. `--resume` reuses completed stages; `--no-render` omits
the original-renderer images. Partial mechanics/geometry logs preserve progress
and unsuccessful attempts if a stage fails.

Observed equilibria and material identity loss using the refined-reference
joint-radius control (loss is initial material mass leaving its original cell):

| Force / original force | Tip displacement | Local-affine identity loss | MLS identity loss |
| --- | ---: | ---: | ---: |
| 1 | 0.03582 | 0.09304% | 0.13037% |
| 2.5 | 0.08939 | 0.22916% | 0.32537% |
| 5 | 0.17763 | 0.45775% | 0.63795% |
| 10 | 0.34651 | 0.89268% | 1.24381% |
| 20 | 0.63587 | 1.66802% | 2.27897% |

These are loaded-equilibrium snapshots, not the averages over release frames
reported in the preceding experiment. Nominal beam length is 2. At the largest
load, maximum polar rotation is 30.63 degrees, principal stretches range from
0.8955 to 1.0912, and det(F) ranges from 0.9325 to 1.0557. The free-force residual
divided by total load is below 8e-10 at every reported equilibrium. Smaller load
increments reproduce the final center positions to 3.2e-12. One attempted large
increment reached the Newton iteration limit and succeeded after bisection;
this is recorded and is not treated as a physical failure.

At the largest load, independent volume errors against the refined geometric
target are 1.0016% (fixed), 1.8129% (cube-root), and 0.00359% (joint refined).
The saved-mechanical-target joint solve also achieves 0.00359% against its own
target. Refining validation from 2^17 to 2^18 directions with a different seed
gives 0.00358% for either joint solve. Fitted radius ratios remain within roughly
[0.969, 1.034], with no empty cells or active bounds in these tested states.

Material loss grows despite accurate volumes. At maximum load, the refined
joint-radius control sends 1.4487% / 1.7991% of mass into another cell and
0.2193% / 0.4799% outside all cells (local / MLS). The worst cell loses 8.17% /
11.43% of its own material under these extensions. Fixed radii give total losses
of 1.5570% / 2.1642%, so volume fitting does not improve this indicator. The
largest-load refined-joint coarse/fine differences are 0.00572 / 0.00209
percentage points; independent-seed differences are 0.00979 / 0.00348 points.

The interior-motion assumption becomes more consequential at this amplitude.
MLS misses its input center positions by up to 0.00499 (3.33% of h), and its
gradient differs from the TLSPH F estimator. At 4,352 interior sample points,
MLS det(gradient) ranges from 0.4115 to 1.3488, although no sampled determinant
is nonpositive. Boundary extrapolation therefore introduces substantial local
compression/stretch beyond the center estimates. Neither the MLS error figure
nor the discontinuous local-affine extension is a unique physical ground truth.
The evidence establishes volume-fit feasibility and growing label discrepancies
under both stated extensions, not an impossibility theorem for all coupled
Power Foam mechanics or a claim of dynamical failure. Additional quadrature and
interior-gradient checks are saved in `additional_checks.json`.

### Integrated-volume consistency audit on the same load states

Run `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m
experiments.tlsph_integrated_volume` (as one shell line). This reuses
`output/tlsph_load_sweep/{trajectory,radii}.npz`; no forces, positions, deformation
gradients, masses, or radii are changed. Outputs are in
`output/tlsph_integrated_volume/`. `--resume` reuses quadrature arrays after
checking input hashes and sampling parameters.

The test asks whether the earlier material-motion diagnostic and radius target
describe the same volume change. It compares:

1. `V0_refined * det(F_TLS_i)` (the refined-reference radius target);
2. `integral_cell det(grad(phi_MLS)) dV` for the exact continuous MLS extension
   used in the earlier material-label test;
3. independently integrated rebuilt cell volumes using the existing
   `joint_refined` radii.

The saved mechanical-volume target is also reported separately. Using the same
refined rest geometry in (1) and (2) removes the old reference-volume quadrature
noise from their comparison, without changing the MLS weights or mechanics.
The integral uses a radial, importance-weighted Sobol cell average multiplied
by refined rest volume: 2,048 and 8,192 points per cell, plus an independent
8,192-point seed at the largest load. Each fine pass samples 2,228,224 reference
points. Analytic differentiation includes the dependence of the MLS fit on the
query position; it is checked against centered differences and an affine patch.

| Load | Integral vs TLSPH target, weighted L1 | Worst cell relative difference | Rebuilt volume vs TLSPH target |
| ---: | ---: | ---: | ---: |
| 0.16 | 0.04137% | 0.58477% | 0.00367% |
| 0.4 | 0.10328% | 1.44871% | 0.00367% |
| 0.8 | 0.20548% | 2.84667% | 0.00368% |
| 1.6 | 0.40268% | 5.44973% | 0.00390% |
| 3.2 | 0.75234% | 9.75863% | 0.00359% |

Weighted L1 is `sum_i abs(V_i - target_i) / sum_i target_i`, not the difference
between total volumes. At maximum load, coarse/fine quadrature disagreement is
0.00881% by this metric; changing the fine seed gives 0.00230%. These differences
are well below the 0.75234% mismatch. The determinant-of-cell-mean-gradient
versus cell-mean-of-determinant discrepancy is only 0.02490%; nonlinear averaging
of determinants alone does not explain the mismatch. The underlying gradient
estimates differ as well. General gradient finite-difference error is below
5.6e-9 and affine reproduction error below 4.1e-13.

**Correction to the interpretation of earlier MLS material-label results:**
the dense audit detects negative MLS Jacobians at loads 0.8, 1.6, and 3.2.
Earlier checking of 4,352 interior samples missed these regions. At load 3.2,
the sampled Jacobian range is [-4.97484, 2.58987]; points with nonpositive
Jacobian account for an estimated 0.02699% of reference volume. This small
fraction does not explain all label changes, but it invalidates treating this
MLS extension as a globally admissible, orientation-preserving material flow.
The table therefore reports **signed Jacobian integrals**, not physical image
volumes. No absolute value or removal of inverted samples is used. Even where
sampled Jacobians are positive, global injectivity has not been proved.

Three extreme points were confirmed inside their original cells, and their
negative determinants were independently verified by finite differences. For
the most negative point, det(gradient) is -4.97484259 analytically and
-4.97484211 by the finer difference step; this is not a derivative-code sign
error. The earlier roughly 11% worst-cell MLS label loss is a result of this
chosen diagnostic extension, not validated physical material transport. The
roughly 8% local-affine result belongs to a different, discontinuous extension;
its per-cell volume equals `V0 * det(F)` by construction, but that does not make
the maps a coherent continuous flow across cells.

The experiment establishes a mismatch between the assigned volume target and
the assumed interior motion, alongside accurate fitting of that assigned
target. It does not show that the continuum Jacobian volume formula is wrong,
or establish an unavoidable defect of Power Foam or TLSPH. A physical claim
requires a consistent interior motion, deformation estimate, and volume update.
No radii are fitted to the flawed MLS signed-volume targets in this audit.

### 黏结 DEM 文献与 Power Foam 接口设计（2026-10-07）

本节是文献核对和待实现设计，不是已经完成的 DEM 实验。资料保存在
`references/bonded_dem/`；`manifest.json` 记录版本、文件哈希、阅读范围和访问失败。
本次没有修改现有力学求解器，也没有把文献公式拼接成未经验证的动态演示。

**模型选择。** 第一基准采用三维线性平行黏结（parallel bond）的法向、切向、
弯曲和扭转响应，以及不可逆的拉伸/剪切断裂。先保持粒子半径固定。
渐进拉伸软化使用已有 soft-bond 规则作为独立扩展，不能把它误称为完整的
拉伸和剪切共同软化模型。Power Foam 的几何如何定义参考连接与尺度，是我们
需要验证的改动；标准 DEM 的力学规律和我们的几何改动分别记录。

#### 阅读结果与来源边界

- Potyondy & Cundall (2004), *A bonded-particle model for rock*,
  DOI `10.1016/j.ijrmms.2004.09.011`：经典基础。读到出版方摘要、引言和章节预览，
  没有获得完整期刊 PDF。公式级核对来自 Itasca 9.6 官方文档，以及作者 2021 年讲座
  中模型对比部分；不能声称已精读 2004 全文，也不能假定当前软件与原论文逐式相同。
- Kuzkin & Krivtsov (2017), *Enhanced vector-based model for elastic bonds in
  solids*, DOI `10.22226/2410-3535-2017-4-455-458`，已下载并读完正文 §§1–5。
  实际版本是 `arXiv:1507.06957v2`（2017），不是较早的 2015 v1。
  它以相对位置和随粒子转动的方向向量定义能量，推导力与力矩，给出单键拉压、
  剪切、弯曲、扭转的解析验证，以及长梁/短连接的参数关系。它只提供弹性模型，
  不提供损伤演化；用于能量一致性与转动处理的对照，不直接混入 PFC 更新公式。
  论文中关于辛积分“守恒能量”的措辞不能照搬成任意有限步长下能量精确守恒。
- Neveu et al. (2016), *Fracture of granular materials composed of arbitrary
  grain shapes: a new cohesive interaction model*, DOI `10.1016/j.jmps.2016.06.008`，
  已下载，读了黏结点、力和刚度、不可逆断裂，以及初始化和参数敏感性部分。
  该文将接触与黏结分开；通过偏离中心的黏结点产生抗转能力，并讨论三维拓展。
  但文中数值验证是二维，不能声称复现过三维断裂；单连接是阈值断裂，
  宏观逐渐损伤来自不同连接先后破坏。邻接数、连接尺度影响宏观刚度。
- André et al. (2012), DOI `10.1016/j.cma.2011.12.002`：出版方预览说明了
  cohesive-beam 的微观到宏观标定。全文访问未成功，不将它列为全文精读来源。
- Itasca 官方实现资料：
  [parallel bond](https://docs.itascasoftware.com/itasca960/common/contactmodel/linearpbond/doc/manual/cmlinearpbond.html)、
  [contact resolution](https://docs.itascasoftware.com/itasca960/pfc/docproject/source/manual/pfc_model_components/contacts/contact_resolution.html)、
  [soft bond](https://docs.itascasoftware.com/itasca960/common/contactmodel/softbond/doc/manual/cmsoftbond.html)。
  平行黏结承受力和力矩；断裂后黏结消失，但接触机制仍可存在。Soft bond
  增加拉伸软化和卸载分支，剪切超限仍可直接断裂。Flat-joint 的分区损伤是另一候选，
  本轮没有审计其完整公式。

#### 不再混淆的三个“半径”和两个图

Power Foam 的 `get_radii()` 是生成 power cell 的几何半径；黏结截面半径是
计算连接刚度和强度的尺度；碎片碰撞的形状/接触半径决定物体是否互相穿透。
这三者不能未经验证地当成同一个参数。标准 soft-sphere DEM 中“soft”指接触
允许小的重叠，并不意味着每个球都有可伸缩的半径自由度。

参考黏结图 `bonds` 保存永久粒子 ID、参考构型和断裂历史；当前渲染邻接图
`scene.adjacency` 仍由 `rebuild_adjacency()` 更新。重建 diagram 不重置黏结应变，
不让断键复活，也不把刚进入 Čech 邻域的粒子自动焊接。碰撞候选图是第三种用途，
可以共用宽阶段搜索，但不能共用黏结状态。

#### 参考连接和截面：建议的 Power Foam 改动

1. 使用初始 Čech 边作为候选。第一种有几何依据的选择是保留参考 bounded
   power cells 之间有正面积共同分界的边；小面积边的阈值与连通性要报告。
   原始 Čech 全图作为对照。多余 Čech 边在渲染裁剪中无效，在力学里不能自动无效。
2. 原论文常用由球半径比例指定的圆形黏结截面。我们的候选改动是使用初始
   实际分界面积 `A0` 定义等效圆截面：`Rb=sqrt(A0/pi)`，
   `I0=A0^2/(4*pi)`，`J0=2*I0`。这只是连接尺度的选择，不把分界面视为真实膜。
   等效圆会舍弃截面形状信息；需要与截面的两个面积惯性主值作敏感性对照。
3. 不能把两球交圆的整个圆盘直接称为实际共同分界：还需要被其他球的 power
   半空间裁剪；如果用于已训练场景，还要判断物体占据区是否真的连通。
4. 参考长度先定义为 `L0=|p_j0-p_i0|>0`，而不是表面间距 `L0-r_i-r_j`。
   后者在重叠输入上为负，不能放进 `EA/L`。这是等效连接长度的建模选择。
5. `A0,L0,I0,J0` 在弹性基线里固定，不能每帧更新截面却遗漏相应功和历史变换。
   先用几何确定尺度，再标定连接材料参数；微观模量不能直接宣称等于物体宏观模量。

#### 力学状态与每步更新

每个力学载体保存 `body_id, position, velocity, rotation, angular_velocity,
mass, inertia`。每条连接保存 `i,j,L0,A0,I0,J0,frame,normal_force,
shear_force,bending_moment,twisting_moment,bond_state,strengths`。
Soft-bond 还需要保存峰值、软化阶段和卸载历史，不能只加一个任意衰减系数。

在共同的瞬时作用点 `c` 处，速度必须包含自转：

```
u_rel = (v_j + omega_j × (c - center_j))
      - (v_i + omega_i × (c - center_i))
omega_rel = omega_j - omega_i
```

将这两个量分别拆到连接的法向与切平面，得到法向/切向位移增量与扭转/弯曲
角度增量。切向历史必须随连接坐标系旋转，包括绕法向的共同自转；仅把旧切向力
投影到新平面不够。累加到粒子时，除黏结力矩之外还必须加入力臂叉乘力。

平行黏结采用的刚度量纲为：`kbar_n,kbar_s` 是应力/长度；乘截面积以后才是
力/长度；乘面积惯性矩以后才是力矩/转角。对我们选择的正参考长度，可以取
`kbar_n=E_b/L0`、`kbar_s=G_b/L0` 作为待标定起点；这不是保证宏观材料为
`E_b,G_b` 的公式。基线必须显式保留四种响应，不能漏掉剪切力造成的力矩。

每个物理步：更新连接坐标系和运动增量 → 更新力/力矩与损伤 → 汇总力和力矩
并更新平移与转动 → 在需要渲染的帧写回 Power Foam 参数并重建 diagram。
损伤前先做无阻尼验证；阻尼和破坏耗能单独记账。时间步依据累积平移/转动刚度
和质量/惯量估计，并用减半时间步检查；不能照搬某篇颗粒规模不同的时间步。

#### 初始重叠、质量与接触：不能跳过的接口

- 初始黏结力和力矩归零。Power Foam 球的几何重叠本来就存在，不能直接将
  `r_i+r_j-d_ij` 代入无参考偏移的压缩接触力，否则初始场景会被撑开。
  平行黏结归零不等于与它并联的接触分支也归零。
- 参考黏结对的小变形基线可定义 `g_ref=d_ij-d_ij0` 作为无应力间隙偏移，
  但这只是局部等效接触规则，并没有解决任意新碰撞或大滑移。不得把整批
  渲染包围球作为真实硬碰撞体。完整碎片再接触需要材料表面或明确验证过的
  接触代理；由每帧重新分区而永不重叠的 cell 直接检查穿透，也是不够的。
- 质量来自材料密度乘参考实际占据体积，不能累加相交球体积，也不能用光学
  `density`。合成全实心 cell 可以使用整个 bounded cell；真实 Power Foam
  的 dipole/细节表面还决定占据区，不能默认每个 cell 都是实心。
- 生成中心通常不等于参考材料质心。若使用参考体积积分得到质量、质心和转动
  惯量，力学更新以质心 `c_i` 为基准，保存偏移 `a_i=p_i0-c_i0`，再用
  `p_i=c_i+R_i*a_i` 写回生成点。它不引入独立网格顶点。若简化为质量集中在
  生成点、采用球形惯量，必须明确记录这是另一项近似。

#### 损伤模型的实现顺序

先实现平行黏结阈值断裂：最大拉应力要包含弯曲贡献，最大剪应力要包含扭转
贡献；超过强度后不可逆失效。保存破坏原因及破坏前的储能，并与动力能量变化
一起核对。不能既删除储能、又把同一份能量额外释放一次。

如果需要单条黏结逐渐变弱，再完整实现官方 soft-bond 的拉伸软化/卸载/断裂
状态机；不把剪切软化说成该模型已有功能。若要一般混合模式软化，应另外选定
并验证黏聚损伤定律，给出断裂能与连接面积、粒子分辨率的关系。

#### Power Foam 代码接口

- 现有 `powerfoam/scene.py` 的 `points`、`get_radii()`、`quaternions` 和
  `rebuild_adjacency()` 是输出接口；模拟过程中冻结训练优化器、densify/prune
  和重排，或者用永久 ID 显式维护重排映射。
- 第一基线保持 `r_i=r_i0`，直接模拟中心和转动。固定半径不代表重建 cell
  体积固定。可变半径需要额外的可变形粒子定律；不能重新套 `det(F)^(1/3)`。
- 力学初始转动设为单位旋转，渲染的原始朝向另存。对初始法向/切向应用
  力学旋转，再转换成 scene 的四元数；scene 使用的行向量约定见现有
  `experiments/tlsph_render.py`，不可直接覆盖成力学四元数。
- `texel_sites` 与 `texel_height` 是由半径缩放的局部细节，基线中随各自载体
  刚性移动。`get_att_sv()` 没有自动按 body rotation 旋转 SV 方向；
  `forward()` 将它们直接传给 `sv.forward`。若采用随材料旋转的方向外观假设，
  需要旋转 `texel_sv_axis` 的方向且保持其长度（温度），并单独验证视角相关颜色。
  这不是对真实光照/反射变化的完整物理解。
- 建议新建 `experiments/bonded_dem_core.py`（纯力学）、
  `bonded_dem_probe.py`（解析/材料验证）、`bonded_dem_powerfoam.py`
  （参考几何、质量和渲染转接），通过验证后再移入正式模块。设计时尚未创建这些文件；
  第一阶段的后续实现及范围见下节。

#### 预先确定的验收路线

| 阶段 | 验收内容 | 不能由该阶段推出的结论 |
| --- | --- | --- |
| 双粒子 | 单键拉压、剪切、弯曲、扭转；力与力矩平衡；整体刚体转动；无损伤加载卸载及时间步收敛 | 单键正确不保证整块材料刚度正确 |
| 小试件 | 同一组参数下标定并验证宏观拉伸、侧向收缩、剪切、弯曲；检查不规则邻域、分辨率和低能模式 | 不能分别对每个试验随意重调参数后声称统一材料 |
| 断裂/再接触 | 阈值、不可逆性、软化曲线及耗能；断后张开、闭合和摩擦；避免初始重叠预应力 | 断键数量不等于已形成真实可见裂缝 |
| Foam 耦合 | 同一轨迹使用原始 renderer；独立检查体积、纹理、裂缝开口和材料归属 | 动画好看不保证每个 cell 保持同一份材料 |

损伤导致力学断开，不会自动让球并集里出现空气裂缝；保留相交球并重建图时，
邻接 cell 仍可能填满开口。显式裂缝占据、可见内部表面和材质需要额外表示，
不能仅靠把 bond_state 设成 broken。直接冻结并搬动旧 cell 网格又违反用户要求，
不能将其偷偷作为最终几何方案。

**保留的研究要求：** 标准黏结 DEM 跟踪的是力学载体与黏结历史，不能证明当前
power cell 始终是初始同一份材料的像。固定半径的基线用于验证力学和接口，
并不满足完整的材料对应要求。可变半径、裂缝占据和重建后的材料对应必须作为
后续耦合问题显式解决；此前 MLS 插值的材料诊断也不能未经修正再次充当真值。

### 黏结 DEM 第一阶段：连接力学实现与验证（2026-10-07）

**目的：** 先验证一条三维黏结在拉压、剪切、弯曲、扭转和断裂时的响应，
以及多个连接汇总后是否满足力和力矩平衡。本阶段没有渲染、RGB 优化、
Power Foam 半径更新或整体材料参数标定。测试构型是我们编写的解析验证例，
不是论文实验复现，也不代表已完成 Power Foam 仿真。

新增文件：

- `experiments/bonded_dem_core.py`：NumPy float64 的平行黏结分支，固定参考连接，
  保存法向/切向力、弯曲/扭转力矩与不可逆断裂状态。初始重叠不产生预应力。
  按共同作用点的速度计算增量，计入自转与力臂；切向历史同时跟随法向变化和
  绕法向的共同自转。拉伸破坏计入弯曲，剪切破坏计入扭转及平均法向应力。
- `experiments/bonded_dem_probe.py`：解析加载/卸载、刚体运动、四粒子网络的
  坐标变换与平衡、断裂阈值/不愈合、无阻尼平移和扭转振动，以及时间步收敛。

公式来源是已归档 Itasca 9.6 的 parallel-bond (5)–(15) 和 contact-resolution
(2)、(5)、(7)、(12)。这是独立实现。我们对速率定律使用中点积分，切向历史
使用最短弧旋转和轴向自转的分步更新，未逐行复制文档的一阶坐标系更新式。
一般有限转动有时间离散误差，验证其收敛，不声称任意步长下精确客观。

当前只实现 **bond 分支**，还没有完整 parallel-bond 模型中并联的无拉力摩擦
接触分支。断键之后本模块不产生碰撞力，不能用于碎片再接触。当前损伤是
瞬时不可逆断裂，未实现 soft-bond 渐进软化。断键时移除的储能单独记录，
没有额外注入动能；该数值不等同于已经标定的材料断裂能。

运行：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/fangsuo/anaconda3/envs/om/bin/python -m experiments.bonded_dem_probe
```

产物：`output/bonded_dem_stage1/results.json`（含源码 SHA256、环境版本）、
`curves.npz` 和 `mechanics.png`。运行于已有 `om` 环境，无新增依赖。

本次全部断言通过，主要结果：

| 检查 | 结果 |
| --- | --- |
| 法向、弯曲、扭转与解析式对比 | 相对误差均小于 `3.2e-15` |
| 横向中心位移与解析式对比 | 相对误差 `1.68e-8` |
| 四种加载后沿原路径卸载 | 力/力矩残差小于 `1e-14` |
| 不等半径粒子整体平移并转动 2.3 rad | 无载连接的力/力矩范数 `4.43e-16` |
| 已加载切向历史绕连接法向转动 | 相对误差 `4.79e-15`；漏掉轴向历史旋转的对照误差约 150% |
| 已加载连接作一般刚体转动，20/40/80 步 | 载荷旋转误差 `8.25e-5 / 2.06e-5 / 5.16e-6`，每次约降到四分之一 |
| 四粒子不规则网络 | 换坐标后载荷误差 `7.86e-16`；合力/总力矩残差小于 `2e-16` |
| 拉伸、剪切、弯曲、扭转分别触发破坏 | 全部符合判据；闭合和重新加载不恢复连接；无重复移除储能 |
| 压缩与压力相关剪切强度 | 压缩不误触发拉断；压缩增强、拉伸降低剪切强度的检查通过 |
| 无阻尼平移/扭转振动，4 周期、1600 步 | 最大轨迹误差约振幅的 **0.0243%**，最大总能量偏差 **0.00617%** |
| 振动 400/800/1600 步 | 轨迹和能量误差在步长减半后均约降到四分之一 |

横向中心位移不是有限形变下的纯剪切：中心距也会增长。对参考长度为 1、
横向位移为 `q` 的轨迹，解析法向伸长为 `sqrt(1+q*q)-1`，随动切向伸长为
`asinh(q)`。验证同时检查这两项，没有把必然出现的法向响应当成错误消掉。

自由振动使用两粒子的平移或扭转简谐模式，以独立解析频率比较真实惯性运动；
步进采用 kick-drift-kick。转动测试使用标量转动惯量；当前还不是支持任意惯量
张量、外载荷和边界条件的通用刚体积分器。单独的本构功—储能等式仅作为记账检查，
没有把它当作独立能量守恒证明。小测试使用自洽的单位体系，不对应已标定的材料。

**下一阶段：** 完成多粒子的平移/转动积分及约束，构造小块三维试件；检查刚体
零模态和额外松动模式，用同一组参数验证拉伸、横向收缩、剪切和弯曲响应，再
决定是否接入真实 Foam 参考邻接/截面及渲染。图重建、材料归属、可变半径和裂缝
占据仍按上一节保留为独立耦合任务。

### 黏结 DEM 第二阶段：小块材料与多粒子运动（2026-10-07）

**本阶段目标：** 检查连接组装后是否存在异常松动，用同一组微观参数测量
拉伸、侧向收缩、剪切、弯曲，并验证真实多粒子运动的积分。模型仍是合成
试件上的 bond 分支，没有 Foam 参考几何、碰撞、渐进软化或图像优化。

新增 `experiments/bonded_dem_dynamics.py`：

- 从初始接触点相对位移和相对转角构造稀疏参考刚度，包含切向力的力臂力矩。
  该矩阵仅适用于无预应力的小变形线性化；动态受力仍由实际黏结历史计算。
- 多粒子 kick-drift-kick 积分、世界朝向矩阵、静止夹持、外力和外力矩。
  当前支持每粒子的标量各向同性转动惯量，不支持任意惯量张量或移动夹具。
  质量/惯量显式传入，未从光学密度或相交球体积推导。
- 参考质量加权刚度的行和给出保守线性稳定步长上界，实际使用较小步长并
  做收敛检查。这不是大变形/断裂情况下的自适应稳定性保证。

新增 `experiments/bonded_dem_material.py`，运行方式：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/fangsuo/anaconda3/envs/om/bin/python -m experiments.bonded_dem_material
```

输出为 `output/bonded_dem_stage2/results.json`、`curves.npz`、`material.png`，
其中 JSON 保存本轮源码哈希。使用已有 `om` 环境，无新依赖。

#### 试件与一致使用的参数

单位立方体使用 27、64、125 个粒子；长宽高 3:1:1 的梁使用 63、160、325
个粒子。分别检查规则排列和坐标扰动幅度为 `0.08h` 的排列，边界平面保留。
初始中心距离小于 `1.55h` 就连接，因此这是球重叠图，不是已经筛选的实际
power 邻接。常规规则内部粒子最多连接 18 个邻居。

所有分辨率、排列和加载共用 `E_b=1200`、`G_b=480`，并使用
`kbar_n=E_b/L0`、`kbar_s=G_b/L0`、`bond_radius=0.25h`。
粒子代理半径为 `0.775h`。没有按加载种类或分辨率重新拟合参数。
质量使用单位密度下的梯形积分权重，总质量等于盒体体积；这是合成试件的
赋值，不是当前 Power Foam cell 的体积。惯量采用明确的球形代理 `0.4*m*r^2`。

静态解来自参考线性刚度。额外用真实 `ParallelBonds.advance()` 沿位移路径
加载，检查实际力与线性预测是否吻合；没有把线性解冒充完整非线性平衡。
剪切和三维表观弹性张量使用边界仿射位移、内部位移和全部粒子转角自由松弛。
这是有限试件的位移边界估计，不是无限体或周期均匀化材料常数。拉伸另做
横向自由的端面加载，只用最少的附加约束去除刚体运动。梁根部固定六个自由度，
末端施加面积加权的总力 0.001。

#### 已通过的数值验证

- 两组 27 粒子试件都恰好有 **6 个刚体零模态**，无额外零模态或负刚度模态。
  这里的 6 个是整块平移/旋转，不是材料内部异常松动。
- 参考刚度与实际黏结力的中央差分方向导数一致，相对误差小于 `2.3e-14`。
- 拉伸、剪切、弯曲的真实加载历史力与线性预测误差分别为
  `2.46e-5`、`8.04e-6`、`2.09e-4`；加载幅度减半后误差约减半。
  这说明当前小形变线性计算与已实现的黏结律一致，不证明任意大形变正确。
- 所有静态解的自由自由度相对残差小于 `2.3e-13`。
- 自由多粒子系统的线动量、角动量漂移均小于 `7e-19`（本测试单位下）。
  均匀外加加速度与解析轨迹误差小于 `2e-16`；夹持后外力/力矩不使夹具运动。
- 45 粒子扰动梁以最大初始位移 `2e-6` 释放，模拟两个周期。
  使用完整黏结历史更新，不以固定刚度矩阵计算动态力。线性本征频率
  `8.889697` rad/时间单位作为独立小振幅预测，粒子朝向由实际积分更新。

| 动态步数 | 最大归一化振幅误差 | 最大相对能量误差 |
| --- | --- | --- |
| 523 | `2.656e-4` | `1.443e-4` |
| 1046 | `6.640e-5` | `3.608e-5` |
| 2092 | `1.660e-5` | `9.021e-6` |

步长减半，两类误差均约降为四分之一。最细步长的振幅误差约 **0.00166%**，
能量误差约 **0.000902%**。这些是小振幅、未断裂的结果。

#### 材料层面发现的问题

| 规则立方体粒子数 | 横向自由拉伸得到的 E | 横向收缩比 nu | 仿射边界剪切得到的 G_xy |
| --- | --- | --- | --- |
| 27 | 1165.62 | 0.12237 | 433.32 |
| 64 | 997.26 | 0.12021 | 381.28 |
| 125 | 915.24 | 0.11978 | 354.93 |

同样大小的试件细化后，E 在本次范围内下降 **21.48%**。扰动排列得到类似
趋势（1164.15 → 994.94 → 908.20），因此轻微打乱排列没有消除这个现象。
不能据此声称已得到分辨率无关的材料，也不能仅凭这三组数据断言永不收敛。
当前统一圆截面、邻接计数、表面/边界和随 h 改变的微观长度尺度都有可能参与。

对**同一个仿射边界弹性张量**取 E、nu、G，剪切模量也比各向同性关系
`E/[2*(1+nu)]` 给出的值低约 13%–17%。这里没有混用上一表横向自由试验的 E
来计算该差异。当前规则网络是立方对称，不能未经验证地称为各向同性材料。

同一组参数的梁可以承受弯曲。规则梁的末端位移为
`5.9447e-5 / 8.1104e-5 / 9.6049e-5`，比使用同分辨率立方体表观 E/G 的
Timoshenko 估计分别低约 41% / 31% / 25%。这是短梁、离散截面与六自由度夹持，
该连续梁公式仅作为对照，不能把差值直接称为求解器误差，也不能声称弯曲标定完成。

**结论和下一步：** 多粒子受力、约束和运动积分已通过本轮检查；仍需解决的是
连接几何与宏观材料参数的对应。接 Foam 前应比较原始球重叠连接和实际 power
邻接，并检验由初始几何生成连接截面后，整体刚度和分辨率依赖如何变化。
这一步不应通过每个分辨率/每种加载单独调参来掩盖差异。任何 power 分界都只作为
中心和半径生成的参考几何尺度，不引入独立运动顶点或物理膜。

### 黏结 DEM 第三阶段：载荷—回弹—原始 Power Foam 渲染（2026-10-07）

按用户要求，暂缓第二阶段的材料/分辨率标定，先验证最小完整链路。
新增 `experiments/bonded_dem_powerfoam.py`：手工构造 160 个 foam 的 3:1:1
悬臂场景，从初始中心和半径生成 1320 条球重叠黏结，固定 x=0 一端；在另一端
施加向下的平滑力脉冲，再完全撤力。位移和转动由真实 DEM 力学积分产生。

使用已有刚度规则 `kbar_n=1200/L0`、`kbar_s=480/L0`、黏结截面半径 `0.25h`，
没有针对这次动画重新拟合材料模量。初始几何半径选为 **0.9h**，并在全程固定。
它与第二阶段的 `0.775h` 不同：0.9h 覆盖规则格点间隙，使初始演示没有规则
格点球并集的内部空缺；初始黏结图也由这个半径重新生成，与渲染初始球一致。
不沿用第二阶段较小半径的旧图。质量仍是合成盒体的积分权重，惯量仍是球形
标量代理；尚未按真实 dipole 占据区积分，也未解决生成点与材料质心的偏移。

#### 运行与输出

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/fangsuo/anaconda3/envs/om/bin/python -m experiments.bonded_dem_powerfoam

# 复用已保存轨迹，仅重新执行渲染及朝向转换检查：
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/fangsuo/anaconda3/envs/om/bin/python -m experiments.bonded_dem_powerfoam --render-only
```

产物位于 `output/bonded_dem_powerfoam/`：

- `powerfoam.gif`：81 帧，正面/斜侧面并排；每帧 65 ms，画面标注实际模拟时间。
- `powerfoam_frames.png`：初始、最大向下弯曲、最大向上回弹的两视角对照。
- `motion.png`：真实末端位移、粗细时间步对比、总能量与外力做功。
- `trajectory.npz`：中心、机械朝向、固定半径、永久黏结、采样时间及能量记录。
- `mechanics.json`、`render.json`：参数、误差、逐帧渲染检查及源码哈希。

#### 力学结果

载荷为末端面积加权的总力 `2.68409*sin(pi*t/T)^2`，`0<t<T`；之后为零。
`T=1.46206`，来自参考线性化第一振动周期；总模拟时间 `3T=4.38617`。
力的幅度根据参考线性柔度选定，使恒定峰值力的线性静挠度为 0.18。
实际过程是动态脉冲响应，不是把线性静位移直接当动画，也没有缩放渲染位移。

| 项目 | 测量结果 |
| --- | --- |
| 最大向下末端位移 | 0.29869（参考梁长 3） |
| 最大向上末端位移 | 0.27119 |
| 粗/细时间步 | `0.00107504 / 0.000537521`，4080 / 8160 步 |
| 粗细轨迹最大粒子位置差 / 末端运动幅度 | `1.096e-5`，约 **0.00110%** |
| 细步长外力做功与总能量最大差 / 峰值能量 | `1.130e-6` |
| 撤力后总能量相对变化范围 | `1.339e-6` |
| 黏结状态、固定端 | 全部完整，固定端位置严格不变 |

没有阻尼，没有启动断裂判据；强度保持无穷大。当前模块也没有碎片碰撞。
时间步减半后能量误差约降到四分之一。本次是中等可见弯曲的接口演示，
不意味着第二阶段发现的宏观材料参数问题已经解决。

#### 写回与原始渲染检查

每帧执行以下步骤：中心直接写入 `scene.points`；机械旋转与原始外观坐标系
组合成 scene 四元数；同步旋转 `texel_sv_axis`，保持轴向量长度；保持局部
`texel_sites`、`texel_height`、RGB 参数和半径不变；调用 `rebuild_adjacency()`，
然后执行 **`PowerfoamScene.forward()`**。渲染图每帧重建，黏结图及其应力历史
不重置，也不因新的几何邻居而增添黏结。

使用 8 个局部细节控制点和每点 8 个 SV 方向，光学密度为 60，局部细节高度为
`(0.6+0.025*sin(4u))*radius`，带有非均匀局部颜色和小幅方向颜色变化。
灰色表示固定端，蓝/橙条纹用于观察运动。场景未经图像拟合，不是训练重建结果。
采用原有 dipole、局部细节、SV 和 rasterizer，没有将粒子球另画成动画冒充原始渲染。

同时对整场景及相机施加刚体变换，检查这次新转接代码：前景 RGB MAE
`7.61e-5`，alpha MAE `9.05e-6`，SV 方向转换与已有刚体参考一致。少数像素
误差较大（328 个像素 RGB 最大通道误差 >0.01，最大值约 0.148），因此只报告
上述实际统计，不宣称渲染逐像素严格不变。所有实际轨迹帧输出有限，半径未改变。

#### 观察及局限

初始、向下弯曲、向上回弹的渲染均保持可见整体连接，条纹方向随粒子运动变化。
表面存在手工球/局部平面构成的分块和起伏，初始帧也有这些特征，不能都归因于运动。
在当前分辨率，`alpha>0.5` 掩码内部的封闭背景区域：正面每帧 0–1 像素，
斜侧面每帧 0 像素。它只排查当前视角的封闭投影孔洞，不证明内部无空隙或材料守恒。
相邻帧 RGB MAE 最大约 0.016，其中包含正常运动，不能直接当作纹理错误指标。

本次已跑通 **黏结 DEM → 中心/旋转 → 重建 diagram → 原始渲染**。
局部控制点随各自载体刚性移动只是当前外观传递规则，尚未证明跨 cell 纹理连续、
每个重建 cell 保持原材料归属、体积符合材料定律，或大变形/断裂后仍可用。

### 官方 PowerSim 面包卷：下载、原始渲染与邻接规模核查（2026-10-07）

已下载 [官方发布资源](https://huggingface.co/datasets/Tung11/PowerSim/tree/main/powerfoam_ckpt/bread_roll%40ec2ced65)
中的 `model.pt`（368,546,661 bytes）、配置和训练/测试相机，保存在
`data/powersim/powerfoam_ckpt/bread_roll@ec2ced65/`。无需自己训练，也不需要原始
训练图像即可使用提供的相机渲染。这是 Blender 图像训练所得的官方 checkpoint，
不是此前手工构造的 foam 场景，也不是实物拍摄重建。

`data/powersim/bread_roll_manifest.json` 记录下载地址、字节数、SHA256。
模型 SHA256 为 `b2d24bc3f0805658c53d19ea8659dd235e2305d09181512a1497bcb4379313b4`，
与官方 Hugging Face LFS 元数据一致；相机文件也校验了官方 SHA256。
没有运行远程仓库脚本；checkpoint 通过 `torch.load(..., weights_only=True)` 加载。

新增 `experiments/powersim_bread_demo.py`，运行方式：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/fangsuo/anaconda3/envs/om/bin/python -m experiments.powersim_bread_demo

# 已完成静态渲染后，仅重新执行邻接核查：
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/fangsuo/anaconda3/envs/om/bin/python -m experiments.powersim_bread_demo --graph-only
```

#### 已确认能直接使用

- 10 万个 primitives，8 个局部细节控制点、每点 8 个 SV 方向。
- 所有原始中心、半径、四元数、密度、局部高度和颜色参数均保留，未简化成球或
  用统一颜色代替纹理；未训练或优化。使用原始 `PowerfoamScene.forward()`。
- 读取 `powersim.cameras.v1` 相机包，保留 pinhole 的 eye/right/up 和视场，
  从原生 1000×1000 降到 512×512；保持原来的黑色背景。
- 100 个测试相机中选取 8 个视角，全部得到非空、有限输出。
  `output/powersim_bread/static_views.png` 是八视角预览，`static_*.png` 是单帧。
  没有提供对应真值图，因此未测 PSNR，也不声称复现了论文的画质数值。

#### 在接入真实 DEM 前发现的规模与几何问题

当前重新构建的 Čech 图有 **50,333,258 条有向邻接记录**，即约
**25,166,629 条无向球重叠候选边**。中位邻接度 505，最大 1219。
此前手工梁只有 1320 条黏结，不能未经检查就将真实模型的所有候选边都转为同样
的 CPU 黏结状态。这些边是裁剪候选，不能仅凭它们存在就认定有材料连接。

几何半径的中位数约 0.0156，范围约 0.00588–0.02929。光学密度有零值，
不能把它直接当质量密度。训练 checkpoint 保存的邻接数为 50,335,810；
本轮实际渲染使用本地代码重建的图，记录两者差异，不假定缓存图逐项相同。

已做一个**参考几何候选初筛**：将中心和半径加权提升到四维，对下凸包构造
regular triangulation 的边，再检查两球分界平面是否穿过两个球。
结果：

| 项目 | 数量 |
| --- | ---: |
| 具有无界 power cell 的生成点（下凸包顶点） | 91,994 |
| 没有无界 power cell 的被遮蔽生成点 | 8,006 |
| regular triangulation 边 | 677,143 |
| 进一步通过两球分界平面相交初筛的边 | 667,750 |

候选边数量比原始球重叠图少约 **37.7 倍**。实现采用 float64 Qhull，做了等权
情况下与普通 Delaunay 的边集合一致性检查，以及大权重邻居遮蔽小权重中心的
检查，均通过。输出保存在 `regular_candidates.npz`、`graph_audit.json`；
模型尺寸、张量形状及静态渲染统计见 `checkpoint_audit.json`。

**该初筛不是最终材料黏结图。** 退化 triangulation 可能添加零面积边；两球
平面相交只是必要条件，还需被其他 power 半空间裁剪。无界 cell 存在也不保证
球内或 dipole 实际占据区有正体积。被遮蔽生成点目前没有删掉；运动后它们是否
重新出现及如何随材料移动，需要明确规则，不能偷偷固定或随意插值后忽略。

本轮完成的是官方模型可用性和几何规模核查，**尚未运行真实面包卷的 DEM
形变或生成其动态视频**。后续应在这些候选上核实正面积分界和材料连接，确定
零体积/被遮蔽生成点与质量的处理，再接现有力学；不能把此前合成梁的成功直接
当成真实模型已经验证通过。

### 官方面包卷：逐 cell 小形变黏结力学（2026-10-07，后续实验）

本节接续上面的静态核查。新增三份代码：

- `bread_material_geometry.py`：裁剪参考分界、估计有界 power 支持域体积。
- `bonded_dem_linear_gpu.py`：float64 CUDA、每个有效 cell 六自由度的线性化黏结力学。
- `bread_linear_sim.py`：官方 checkpoint 的固定端受力/撤力实验及原始渲染。

#### 材料和图的定义

当前明确把 **球内的 power 支持域当作实心材料域**，不把光学密度当作质量密度，
也没有根据 dipole 或局部位移重建实际可见表面的内部。因此这是待验证的力学
建模假设，不能将支持域体积直接称为面包真实体积。

对每个 regular 邻居候选，以两球分界圆盘开始，用所有 regular 邻居半空间
继续裁剪，取得正面积及其形心。圆盘用内接 64 边形近似；没有直接拿整个
两球交圆当作截面。分界只提供黏结尺度和力臂，不增加独立的面或顶点自由度。
体积通过平面分界和暴露球面组成的边界积分计算，球面使用 4096 个 Sobol 方向。

几何结果：91,978 个数值正体积 cell、655,257 条保留黏结，总支持域体积
0.0447565（模型单位）。8,022 个零/极小体积生成点仅在本实验的内存副本中移除，
原始 checkpoint 不变；其中 8,006 个原本就没有无界 power cell。
三相机静态前景 RGB MAE 最大 `5.41e-8`，没有像素的最大通道误差超过 0.01。
这是参考时刻的外观检查，不证明移除这些参数对任意后续变形都没有影响。

几何验证包括两球解析面积/体积、第三球裁掉半个分界圆盘，以及细化比较。
抽查 64→128 边形，面积相对 L1 差异 `0.0168%`；球面 1024→4096 方向，
体积相对 L1 差异 `0.446%`。后者是细化差异，不是对真值的误差上界，微小 cell
的体积与连通性仍可能对数值精度敏感。

#### 力学和边界条件

每个保留 cell 单独求解三维平移和三维小转角，不合并粗粒子、不插值驱动位移。
采用既有 parallel-bond 力学在无应力参考构型的线性化：
`kn=E*A/L`、`ks=G*A/L`、`kb=E*I/L`、`kt=G*J/L`，其中 `E=1200`、`G=480`，
`I=A²/(4π)`、`J=2I` 为等面积圆截面近似，`L` 是参考中心距离。
因此各边刚度不同。黏结力臂取裁剪分界形心到两侧生成点的向量。

质量取密度 1 乘支持域体积，集中在生成点；转动惯量使用 `0.4*m*r²` 的各向同性
近似。生成点不一定是材料质心，该质量/惯量模型尚未校准。没有碰撞、损伤、
阻尼、有限转动应力历史或半径演化。采用隐式中点法与块预条件共轭梯度，
没有为避开小 cell 的时间步限制而增大质量。

参考图有 50 个连通分量；最大分量占总质量约 94.53%。本次固定其余分量，
同时固定主分量最左侧 10% 的 cell；主分量最右侧 15% 的 cell 按质量分配向下力。
这项处理会影响物理解释，不能宣称完整面包的所有部分均自由参与变形。
加载采用平滑 `sin²` 脉冲，随后撤力；一段平滑弯曲试探场只用来估计加载时间尺度，
并未用作指定运动或模态降阶。先做单位力线性试算，按全体 cell 的位移/转角峰值
选择小载荷，再以更细时间步重新求解实际载荷。

渲染逐帧写回中心和局部外观旋转，同步旋转 SV 方向，保持训练得到的半径、局部
控制点、局部高度、密度、颜色不变；调用原始 `rebuild_adjacency()` 和
`PowerfoamScene.forward()`。力学使用参考黏结图，渲染使用当前重建图。
没有 RGB 优化或位移视觉放大；GIF 播放速度与未校准的力学时间不对应。
这项实验不证明重建后每个 cell 仍占据同一份材料。

#### 复现与输出

几何裁剪新增依赖 `numba==0.61.2`（本机 `om` 环境已安装，连带安装
`llvmlite==0.44.0`；保留原有 NumPy 2.2.6）。先完成上一节的模型下载和候选图核查，
然后在仓库根目录执行：

```bash
NUMBA_NUM_THREADS=6 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/fangsuo/anaconda3/envs/om/bin/python -m experiments.bread_material_geometry
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/fangsuo/anaconda3/envs/om/bin/python -m experiments.bread_linear_sim
```

几何保存在 `output/powersim_bread/material_geometry.{npz,json}`；动态结果目录为
`output/powersim_bread/linear_dem/`。其中 `mechanics.json` 记录参数、数值检查和
源码哈希，`trajectory.npz` 保存逐 cell 轨迹，`motion.png` 对比时间步细化和能量，
`bread_dem.gif` 是双视角原始渲染动画，`bread_dem_frames.png` 是三个阶段的预览。
已有轨迹可用 `python -m experiments.bread_linear_sim --render-only` 重新渲染。

本机实际运行的粗/细时间步分别为 0.00672307 / 0.00336153，总共 96 / 192 步；
CUDA 力学部分约耗时 190 / 206 秒，不包括几何预处理和渲染。49 个共同输出时刻，
细化前后最大逐粒子位置差异占最大位移的 **0.541%**。细步能量减外力功的最大
误差占峰值能量 `2.87e-8`，共轭梯度真残差最大 `1.99e-8`；与独立 CPU 刚度矩阵
和小规模稠密中点法求解的比较也通过。固定端六自由度及初始位移均严格为零。

实际峰值总力为 0.00236569（模型单位），最大中心位移 0.00256649，约为模型
x 向跨度的 **0.431%**；最大局部转角 0.08027 rad（4.60°）。受力区域按质量平均
的 z 位移从约 -0.001312 回弹到 +0.001297；撤力后无阻尼振动、能量保持稳定。
这是小幅运动，动画未放大位移。时间步检查证明当前离散线性系统求解稳定，
不等价于真实材料参数、空间分辨率收敛或大变形正确性得到验证。

双视角 49 帧均完成原始渲染且输出有限。检查初始、最大下弯和回弹阶段的图像，
在这个小幅度下没有看到明显的大块裂口；细节较小，静态拼图中的运动不明显，
应结合 GIF 和位移曲线查看。没有用这项目视检查推断跨 cell 纹理严格连续，
也未给出大变形画质结论。

#### 提高实际载荷：10 倍力的可见形变测试

用户指出上述运动过于微弱。排查发现，全局载荷被最大转角 cell 限制：原轨迹
99.9% 的 cell 峰值转角不超过 0.003673 rad（约 0.21°），最大值却为 0.08027 rad。
这不是整体只能发生微小形变的证据，而是使用全局最坏值选载荷造成的保守结果。

`bread_stronger_load.py` 读取原参考状态，将实际峰值总力提高到 10 倍，以同样
192 个时间步重新求解；输出与原实验分开保存。它没有把位移乘系数直接送给
渲染器。作为验证，将新求解轨迹与原线性系统的十倍响应比较；这个比较只验证
线性求解的一致性，不验证大转角下的真实力学。原有最大转角限幅不再用于这项
加载压力测试，因此少数 cell 会离开小转角适用范围；没有偷偷锁死或删去它们。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/fangsuo/anaconda3/envs/om/bin/python -m experiments.bread_stronger_load
/home/fangsuo/anaconda3/envs/om/bin/python -m experiments.bread_load_comparison
```

输出在 `output/powersim_bread/linear_dem_10x/`，`bread_dem.gif` 为新双视角动画，
`force_comparison.gif` 将原载荷与新载荷的相同相机视角并排播放，未缩放运动幅度。
`mechanics.json` 包含新载荷、能量检查、固定端检查、转角分布和基线比较。

实际新求解最大位移为 0.0256649，即宽度的 **4.31%**；受力端平均 z 位移范围
为 -0.0131178 至 +0.0129688。与原线性响应十倍的最大位置差异相对幅度为
`1.91e-13`，能量收支相对误差 `2.87e-8`，固定端严格不动。新求解耗时约 200 秒。
99.9% cell 的峰值转角不超过 2.10°，但 13 个 cell 超过 0.2 rad（11.46°），
它们占参考质量约 0.269%；最大转角约 46.0°。这些局部大转动不符合当前线性化
模型的适用条件，不能因数值稳定就称其物理准确；这是本次载荷测试暴露的限制。

### 恢复有限运动黏结历史模型（仅升级力学，第 2～7 项保持现状）

用户要求接回早先的黏结模型。本轮新增 `bonded_dem_finite_gpu.py`，移植
`bonded_dem_core.ParallelBonds` 的弹性分支：法向力、切向力、扭转力矩、弯曲力矩
四种历史，每步按中点相对接触速度/角速度更新；切向力与弯曲力矩按同样的
最短弧旋转和绕法向自旋规则传递。当前方向、力臂和力矩装配均随运动更新。
累计姿态用 `R_new=exp(dt*omega_mid)*R_old`，不把累计小转角当成有限姿态。
这仍是早先的增量率形式弹性黏结模型，不另换成超弹性本构。

按用户指定，仍保留支持域实心假设、原质量和球形惯量近似、E/G 和等效圆截面、
固定非主连通体、固定半径和原外观传递方式。损伤、断裂、碰撞、阻尼均未接入。
原黏结图保持不变，原渲染图继续逐帧重建。

接触点有一个需要明示的适配：早先 CPU 核心默认使用经典两球接触位置；面包
线性版本用初始 power 分界形心。新版以“经典两球接触位置 + 随连接坐标系旋转
的参考偏移”保留面包原有的初始力臂。因此参考状态的质量、惯量、四种刚度及
力臂与之前面包完全相同，不因更换求解器而重新选参数。这个接触点不是每步
重新裁剪得到的实际 power 分界形心，也没有把新边界当成新的材料划分来重置历史。
偏移为零时，GPU 本构逐步对应早先 CPU 实现。

为处理极小质量 cell，时间推进采用**非线性隐式中点法**。迭代中重新计算运动
后的方向、历史传递和真实中点力；冻结几何的切线仅用于求解迭代修正，不是
最终力学模型。每步必须通过非线性残差检查才提交物理状态，线搜索或失败重试
不会重复累加历史。转角过大或不收敛时可二分时间步，不改质量。没有隐式阻尼项。

验证脚本 `bonded_dem_finite_checks.py` 包括：四种模式解析解与卸载；不等半径
网络与原 CPU 代码逐步对比；无应力刚体运动（含偏心接触）；带应力任意方向
转动的时间步收敛；绕法向旋转已有切向历史；试算不修改物理状态；新隐式动力学
与原显式 CPU 动力学的收敛比较，以及自由网络的能量、线动量、角动量检查。
CPU/GPU 历史相对差约 `6.61e-16`；带应力刚体旋转的误差在步数 20/40/80 时
为 `8.25e-5 / 2.06e-5 / 5.16e-6`；隐式对照原显式轨迹的误差随时间步减半
约缩小四倍。有限时间步的历史传递有截断误差，不宣称任意步长下严格客观。

复现顺序（仓库根目录、`om` 环境）：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.bonded_dem_finite_checks
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.bread_finite_sim \
  --refinement 2 --out output/powersim_bread/finite_dem/adaptive_coarse --no-render
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.bread_finite_sim \
  --refinement 4 --out output/powersim_bread/finite_dem/fine
python -m experiments.bread_finite_report
```

沿用之前 10 倍载荷（峰值总力 0.02365694347）和加载时长，粗/细输出区间分别
192/384 步，区间内还会按非线性收敛情况自动细分。
结果存入 `output/powersim_bread/finite_dem/`：`checks.json` 为基础验证；
`adaptive_coarse/` 和 `fine/` 分别保存轨迹、最终物理状态及运行指标；`comparison.json`
记录时间步细化差异，`comparison.png` 对比位移/能量；`linear_vs_finite.gif`
并排显示相同载荷下的旧线性版本和新有限运动版本。细步的原始双视角动画为
`fine/bread_dem.gif`。这些检查验证力学实现与数值精度，不解决尚未修改的第 2～7 项。

大规模求解额外使用逐 cell 的块缩放残差，限制平移误差相对半径、转角误差的
局部估计（阈值 `2e-6`），避免仅以全局力残差掩盖极小质量 cell。内层线性迭代
也检查局部误差。`bonded_dem_adaptive.py` 记住成功的细分级别，连续多个区间
容易收敛后才尝试增大步长。切线乘法用 `bonded_dem_tangent_warp.py` 的融合核
加速，与原 PyTorch 实现的随机非轴对齐力臂对照误差为 `2.58e-16`。这些都是
求解器处理，没有加质量、删除小 cell 或改变力学刚度。

已完成的 `adaptive_coarse` 全程结果：192 个输出区间、3,785 个实际子步，
总耗时 6,293.74 秒（约 105 分钟），最大细分级别 5，即每个输出区间最多
32 个小步。保存的源码哈希与当前对应文件一致；轨迹、最终状态数值均有限。
全程最大能量收支误差相对峰值能量为 `4.75e-11`，局部缩放残差不超过
`2e-6`。保存帧中的最大中心位移为模型宽度的 `4.47%`，最大累计转角约
`43.97°`。这些说明旧迭代法通过细分步长完成了有限运动模型的完整轨迹；
尚无独立细步完整轨迹，不能把收敛残差及能量检查当成时间精度验证。

与原十倍载荷线性轨迹在相同保存时刻比较：受力区域平均 z 位移曲线最大差
为 `9.48e-5`，约为有限运动该曲线幅度的 `0.72%`；单个 cell 的最大中心位置差
为 `0.00428`，约为有限运动最大位移的 `16.08%`。整体运动相近不意味着
每个 cell 的局部运动都相同。比较数据存于 `adaptive_coarse/linear_comparison.json`。
可用 `bread_finite_sim --out output/powersim_bread/finite_dem/adaptive_coarse --render-only`
直接从已保存轨迹调用原 Power Foam 渲染器，不重跑力学。

### 完整导数 Newton–FGMRES：独立验证求解器升级

`bonded_dem_newton.py` 继承原有限运动模型，求解相同的隐式中点方程。
每个时间步的历史状态保持固定，使用前向自动微分计算残差对当前运动增量的
完整方向导数，包含连接方向、接触力臂和历史力转动的变化。求导只计算中点
受力需要的部分，避免计算无关的步末历史；与原 `trial()['force']` 的相对差为
`1.30e-16`。最终状态提交、运动幅度检查及本构均沿用原实现。

未知量按半径和转角单位缩放，方程按质量和弹性刚度组成的逐 cell 六维块缩放。
这只是改变数值单位，不改质量、惯量或刚度。完整导数不强制对称，因此外层
线性修正使用 FGMRES；旧弹性图算子通过近似 CG 逆仅作为预条件器。
FGMRES 检查真实线性残差，失败的非线性试算不提交物理状态。
局部块缩放残差仍是误差估计，不是严格的位置误差界。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.bonded_dem_newton_checks
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.bonded_dem_newton_refinement
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.bread_newton_patch --seed 15427
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.bread_newton_patch --seed 25204
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.bread_newton_patch --seed 85720
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.bread_newton_step
```

结果保存到 `output/bonded_dem_newton/`，包含代码哈希和原始残差记录。
带预应力、偏心接触和大幅累计转动的小系统中，完整导数与中心差分误差随
差分步长呈二阶下降，最低相对误差 `6.24e-12`；迭代解与显式组装完整导数的
稠密求解一致。独立中等质量测试的时间步减半后，位置、姿态和速度差分别
缩小约 `3.82 / 3.82 / 4.04` 倍。这是时间精度检查，不意味着面包中所有极快
局部振动都被当前步长解析。

实际面包的三个局部测试从早先保存的 `finite_dem/diagnostic/final_state.npz`
恢复原质量、刚度和受力历史，固定邻域外圈，仅比较相同初值的一个时间步。
它们不是完整面包的新边界条件。最小质量分别为 `3.44e-17 / 2.56e-18 / 5.70e-17`。
第一处旧方法线搜索失败，新方法四次 Newton 修正收敛；另外两处旧方法需要
六次、九次修正，新方法均为四次。三个新解均与稠密参考解一致。
这支持改进局部收敛性的判断，但自动微分和内层预条件求解有开销，CPU 小测试
中总耗时仍高于旧方法，不能据迭代数减少声称已加速。

`bread_newton_step.py` 使用同一保存状态、原全模型约束及载荷，对 91,978 个
cell、655,257 条黏结进行独立单步比较；它不覆盖后台任务或替换其求解器。
单步结果也不等价于完整轨迹验证，GPU 共享运行时的耗时仅作为参考。

完整规模单步实测：从保存时刻 `t=0.06723068547` 出发，原步长
`dt=0.00336153427` 下两种求解器均失败，新方法十二次修正后局部缩放残差
仍为 `0.00336`，未达到 `2e-6`。整体残差很小不能替代局部检查。
以 `--dt-factor 0.5` 减半步长后，旧方法仍在线搜索处失败；新方法九次修正
收敛，局部缩放残差 `4.50e-10`，单步能量收支误差 `5.15e-14`，固定端不动。
该次新方法耗时约 21.4 秒。这验证了一个完整规模困难状态下的收敛改善，
但未比较旧方法进一步细分后推进同一时间区间的总耗时，不能宣称整体加速，
也尚未验证新方法的完整面包轨迹。后台原任务保持独立。

### 新旧求解器推进同一时间区间的对照

`bread_solver_run.py` 将旧、新有限运动求解器接到相同的载荷和自适应控制器上。
`early` 从早先保存的加载初期状态出发；`release` 从旧完整轨迹的最终状态继续
一小段无外力运动。两者初始物理状态完全相同，控制器均从未细分的步长开始。
计时包含失败试算及实际子步，不包含模型初始化及保存文件。测试按顺序独占
GPU 执行，单次耗时不是重复统计后的性能结论；完整轨迹中控制器会记住已成功
的步长，因此不能将这种重新开始的局部测试耗时直接外推到全程。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.bread_solver_run \
  --solver previous --case early --intervals 1 --dt-factor 0.5 \
  --out output/bonded_dem_newton/interval_early_previous
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.bread_solver_run \
  --solver newton --case early --intervals 1 --dt-factor 0.5 \
  --out output/bonded_dem_newton/interval_early_newton
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.bread_solver_run \
  --solver newton --case early --intervals 2 --dt-factor 0.25 \
  --out output/bonded_dem_newton/interval_early_newton_fine
```

初期同一段 `0.00168076714` 时间，旧方法接受 4 个子步、失败重试 3 次，耗时
11.06 秒；新方法接受 1 步、无失败重试，耗时 9.73 秒。新方法减半步长走两步
耗时 10.98 秒。新方法一步与两步结果的最大中心差 `1.03e-5`、最大姿态差
`0.00833 rad`（约 0.48°），质量加权位置 RMS 差 `3.05e-9`。这说明整体差异
很小，但个别轻质量 cell 的姿态仍对时间步敏感；两步结果不是精确真值。
数据与对照保存在 `output/bonded_dem_newton/interval_early_*`。

从旧完整轨迹末态继续相同长度的无外力区间：旧方法接受 8 步、失败 7 次，
总耗时 49.17 秒（成功试算 18.08 秒）；新方法接受 4 步、失败 3 次，总耗时
71.39 秒（成功试算 27.87 秒）。两者均通过能量及固定端检查，但新方法在
这个状态更慢。最终位置质量加权 RMS 差为 `1.50e-8`，最大单 cell 位置差
`4.78e-4`，最大姿态差 `0.14758 rad`（约 8.46°）。姿态差超过 `0.01 rad`
的 cell 总质量只占约 `2.40e-10`；这并不豁免它们的局部时间精度要求。
两种自适应离散的差异不等价于已知精确误差，仍需后期的独立细步对照。
原始数据见 `interval_release_*`。该结果不支持当前新方法已经整体加速的说法。

完整测试入口 `bread_newton_job.py --out <新目录>` 会先运行 192 个输出区间，
完成后与旧有限运动轨迹比较，再调用原 Power Foam 渲染器生成动画。
`job.json` 记录运行阶段，`progress.jsonl` 记录每个区间，`attempts.jsonl` 记录
成功及失败试算。每 8 个区间保存物理状态及轨迹，`mechanics.json` 的 `complete`
标记区分中途保存与完整结果。新目录不能覆盖已有实验；旧模型及第 2～7 项不改。

本次完整 Newton 测试输出目录为 `output/powersim_bread/finite_dem/newton_full/`；
查看其中的 `job.json` 和 `mechanics.json` 判断当前状态。启动测试不代表已完成。

### Newton 导数计算的融合加速（独立于正在运行的基线）

`bread_newton_profile.py` 对同一保存的完整面包状态、相同时间步进行分项计时。
一次未优化测量中，117 次完整方向导数耗时 8.08 秒，内层弹性预条件求解
6.99 秒，块缩放准备 2.58 秒，整步共 19.63 秒。计时包括同步测量开销，且
GPU 同时有独立后台轨迹任务，不能当作独占设备的精确性能统计。

新增 `bonded_dem_jvp_warp.WarpJacobian`，将每条黏结的几何、历史旋转、四种
受力及力矩装配的方向导数合并为一个 float64 Warp 核。它计算完整导数，
没有重新冻结方向或力臂，没有改变原受力、时间步、收敛容差或历史提交。
用法为 `solver.jacobian_vector = WarpJacobian(solver)`。调用点须已由原 `trial`
验证当前迭代状态的几何条件；原 Newton 流程满足这一要求。

`bonded_dem_jvp_checks.py` 对无预应力、有预应力及三个实际面包邻域分别执行
CPU/GPU 导数与整步核对。块缩放后的解析导数与原自动微分相对差最高约
`2.49e-12`，并通过中心差分核对；实际 CPU/GPU 小系统求解结果一致。
完整规模同一步，融合版本保持 9 次 Newton 修正、108 次 FGMRES 迭代、
1,913 次内层 CG，中心位置最大差 `1.11e-16`，姿态矩阵最大差 `3.71e-14`。
首组测量中导数部分降至 0.83 秒，整步降至 12.33 秒，约为原来的 63%。
这只说明该步有加速，尚未证明整个运动全程有相同比例的提升。

按原版/融合/融合/原版顺序复测，整步耗时为 `19.63 / 12.33 / 12.74 / 19.41`
秒，平均约加速 `1.56` 倍（耗时减少约 36%），迭代次数均相同。明细和位置、
姿态差见 `output/bonded_dem_newton/fused_profile_comparison.json`。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.bonded_dem_jvp_checks
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.bread_newton_profile \
  --out output/bonded_dem_newton/profile_ad
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.bread_newton_profile \
  --fused --out output/bonded_dem_newton/profile_fused
```

融合后主要剩余开销是内层弹性预条件迭代和六自由度块的构建，可继续针对
这些实际热点优化。后台 `newton_full` 仍使用未融合版本作为完整基线，未修改
其运行中的源码或求解器。

### 继续融合弹性预条件计算

`bonded_dem_precondition_warp.py` 提供两项独立优化：`WarpDiagonal` 直接累加
每条黏结对 cell 六自由度对角块的贡献，避免创建逐黏结大批临时矩阵；
`WarpElastic` 合并内层 CG 的质量项、向量更新和六维块乘法。标量归约仍用
PyTorch，CG 的容差、80 次上限及每五次检查的规则与原实现相同。
这些只替换计算实现，不改变预条件器公式或完整 Newton 方程。

`bonded_dem_precondition_checks.py` 在五个测试网络上分别检查 CPU/CUDA。
对角块相对差最高约 `2.59e-16`，内层 CG 结果相对差最高约 `2.75e-12`，
测试向量的停止次数一致；完整单步的结果、能量和固定端检查也通过。
`bonded_dem_newton_fast.FastNewtonBonds` 将完整导数、对角块和内层 CG 三项
融合封装为可选类，可按原 `NewtonBonds` 的构造及求解接口调用。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.bonded_dem_precondition_checks
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.bread_newton_profile \
  --fused --fused-blocks --fused-cg --out output/bonded_dem_newton/profile_all_fused
```

首组完整规模单步测试中，内层求解从约 7 秒降至 2.52 秒，块准备从约
2.6 秒降至 0.20 秒，整步从只融合导数时的约 12.5 秒进一步降至 5.40 秒。
仍为 9 次 Newton、108 次 FGMRES、1,913 次内层 CG；与只融合导数版的
中心位置最大差 `1.39e-16`，姿态矩阵最大差 `4.52e-14`。
GPU 仍与后台基线共享，不能将单步比例当成完整轨迹的已验证加速比。

分项对照中，只在融合导数上增加对角块优化为 10.40 秒，只增加内层 CG
优化为 8.18 秒，三项全部启用的复测为 5.83 秒。两次全部优化平均 5.61 秒，
相对最初未优化 Newton 的单步均值约加速 3.48 倍，相对仅融合导数约加速
2.23 倍。所有测试迭代次数一致，详见
`output/bonded_dem_newton/precondition_profile_comparison.json`。

### 加速版的连续运行与完整对照安排

`bread_fast_solver_run.py` 是独立的可选加速入口。它沿用原运行器的载荷、
自适应控制、记录和检查逻辑，增加 `FastNewtonBonds` 及对应源码哈希。
暂时保留独立入口，是为了不修改正在运行并验证源码哈希的基线文件。

加载初期从相同保存状态连续推进四个区间的短程测试已完成：实际接受七个
子步，发生一次失败后自动二分，总计约 34.19 秒；能量收支相对误差
`7.80e-13`，固定端与姿态检查通过。输出为
`output/bonded_dem_newton/fast_multistep_early/`。这只是短程验证。

`bread_fast_job.py --out output/powersim_bread/finite_dem/newton_fast_full`
会等待 `newton_full` 的完整基线任务结束，随后独立运行加速版的 192 个区间，
与未加速 Newton 的完整轨迹比较，最后从保存轨迹调用原渲染器。
状态记录在 `newton_fast_full/job.json`；`queued` 表示尚未开始仿真。
当前已安排该后台队列，不能把排队或短程通过视作全程验证完成。

用户随后停止了任务：原 `newton_full` 在第 152/192 个区间保存后停止，原
`newton_fast_full` 队列未执行。再次继续时，使用新的 `--resume` 支持从保存
的中心、速度、姿态、黏结历史及时间恢复，同时重建自适应控制器记忆和累计
统计。力学文件哈希必须一致；旧目录保留，续跑写入新目录。
独立的“连续两步”与“保存第一步后续跑第二步”检查中，中心差为零，速度
最大差 `1.32e-15`，姿态矩阵差 `2.20e-18`；记录见 `resume_check.json`。

当前续跑目录为 `newton_resumed_full`，从第 152 个区间继续剩余 40 个；
加速版队列目录为 `newton_fast_after_resume`，等待续跑、比较与渲染全部完成后
开始，保持串行运行。两者当前状态分别由各自的 `job.json` 记录。

这两项完整任务现均完成（192/192），检查通过，各自的 49 帧双视角动画已生成。
未加速 Newton 累计求解 8,219.96 秒（约 137 分钟）；加速版 2,325.40 秒
（约 38.8 分钟），本次运行记录比值约 3.53。未加速版早期曾与局部测试共享
GPU，因此这不是严格重复、独占设备条件下的性能比较。

完整轨迹的受力区平均 z 位移最大差 `6.16e-8`，宏观曲线几乎重合；但局部
轨迹并非仅有舍入量级差异。两版分别接受 2,578/3,002 个自适应子步，最大
中心位置差 `0.0011863`、最大姿态差 `24.65°`。49 个保存时刻的并集中，
8 个 cell 姿态差超过 10°，合计质量占比 `3.41e-12`；117 个超过 1°。
质量加权位置 RMS 差最多 `4.07e-8`。这些差异不能被整体能量通过或质量
占比小所豁免，也不能直接认定其中一版更准确，需要独立细步对照。
详细数据见加速版目录的 `previous_comparison.json` 与 `local_difference_audit.json`。

### 按宏观形变幅度设计载荷实验

新实验以**原受力区域的参考质量加权平均下移量 / 初始模型 x 向宽度**为指标。
宽度为 `0.5954539478`，原载荷的实测峰值约为 `2.2038%`。目标依次为 10% 和
20%，分别允许相对目标 ±5% 的偏差，即 9.5%～10.5% 和 19%～21%。目标只用于
试验之间选择外力幅值，不直接指定 cell 位移，不缩放渲染运动。

保持参考质量、球形惯量、E/G、黏结图、左侧约束、右侧按参考质量分配的受力
区域、固定半径和外观处理不变。仍在 T 内施加原 sin² 力脉冲，随后自由运动
2T，只有峰值总力可变。10% 档的首次试验按比例估计使用原载荷的 4.5376 倍，
峰值总力 `0.1073449249`；它是初始猜测，不代表已经达到 10%。20% 档从已校准
的 10% 档响应推算初始载荷。

`bread_macro_run.py` 使用融合 Newton 求解器，每个输出区间记录平均位移、
归一化幅度、能量、收敛和子步数，并支持保留历史及控制器状态的续跑。
`bread_macro_job.py` 串行执行校准：先跑到 1.5T；若峰值接近区间末端则延长。
初步达标后续跑到 3T，按完整时间窗口的实测峰值再次验收。未达标则修正外力，
每档最多四次载荷试验；求解失败或仍不达标会明确停止并保存结果，不改误差
容差或偷偷更改质量。接受 10% 档并渲染后才开始 20% 档。

主输出目录为 `output/powersim_bread/macro_amplitude/`。`job.json` 记录当前档位、
载荷、校准状态及已接受结果；每次试验有独立的 `progress.jsonl`、状态及轨迹。
`bread_macro_report.py` 输出实际外力/宏观位移/能量曲线和原载荷、10%、20% 的
同相机动画对照。每帧使用原 Power Foam 渲染器，显示位移比例为 1。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.bread_macro_job \
  --out output/powersim_bread/macro_amplitude
```

启动前已核对实际力幅值、宏观指标以及保存后续跑的一致性；短程续跑与连续
求解中心差为零、速度差约 `4.94e-15`。这只是实现检查；10%/20% 档是否达到
目标及能否完成，必须以后台实测结果为准。暂不改变第 2～7 项，未加入损伤、
断裂或碰撞；强形变出现自接触时，当前模型不能提供可信的接触响应。
本轮也不把宏观幅度达标视为已经完成独立的时间精度或材料参数验证。

### 极小 cell 删除外观对比

`bread_small_cell_prune.py` 从当前 91,978 个正体积 cell 中删除参考体积低于
中位数 `1e-4` 的 583 个，保留 91,395 个。阈值为 `1.690389328e-11`，
删除部分占原总质量 `5.565352393e-8`。原 checkpoint 与几何文件未修改；
新筛选掩码及原始 ID 保存在独立输出的 `selection.npz`。

比较使用原 Power Foam renderer，每次重新构建 adjacency，剩余 cell 的参数
和位置完全相同；512 像素宽，9 个相机，分别检查静止和已保存的近峰值形变
（下移 9.95798%）。这是外观消融，未重建力学网络或重跑删除后的运动。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.bread_small_cell_prune \
  --out output/powersim_bread/small_cell_prune_1e-4
```

最差视角前景平均 RGB 误差（0–255）在静止时为 `0.0000264`，变形时为
`0.0003238`；任一通道误差超过 0.01（0–1）的像素数每张分别最多为 0 和 3。
每个状态的首视角重复渲染误差为零。`comparison.png` 展示最差平均误差视角、
同位置局部放大和差异乘 20；各状态目录包含全部视角及未加标签的前后图。
这些结果仅说明本阈值、分辨率、相机与两个状态下直接删除的外观影响很小。

宏观实验本次状态：首轮 pilot 完成 96 步，峰值下移 9.96956%；续跑完成
146/192 个完整区间后，在第 147 个区间达到细分上限并发生 Newton line-search
failure。后台已失败退出，没有完整接受 10% 档，20% 档未启动。上述变形对比
使用已完成 pilot 的第 18 个保存帧，不使用失败后的部分区间状态。

### 删除极小 cell 后的求解速度

`bread_pruned_geometry.py` 使用上述删除掩码，重新计算 regular triangulation、
裁剪面、面积、面中心和体积；保留原 checkpoint 的全局 ID。离散设置与原几何
一致（64 边形面、4096 条球面采样），不是简单删除黏结边。结果保存于
`output/powersim_bread/prune_speed_1e-4/geometry/`：91,395 个 cell，652,764 条
黏结，相比原图新增 26 条、删除 2,519 条。总体积相对变化 `-7.46e-9`，几何
重建约 10.8 秒，未计入求解时间。

`bread_macro_run.py --geometry PATH` 支持这一独立几何文件。仍使用原模型的
固定/受力 cell ID（取保留部分），保留受力权重比例并归一化到相同总外力；
脉冲时长和输出步长来自原模型，材料系数和全部求解器容差未改。参考加载权重
仅损失 `4.10e-8`，不重新按新 cell 数量选择固定/受力分位数。新网络从静止
开始求解，不把旧黏结历史移植到新图上。

串行计时包括此次重跑的原模型前 16 步，和删除后从初始状态跑到 192 步的任务：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=8 python -m experiments.bread_pruned_geometry \
  --out output/powersim_bread/prune_speed_1e-4/geometry
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.bread_macro_run \
  --out output/powersim_bread/prune_speed_1e-4/baseline_16 --intervals 16 \
  --load-multiplier 4.537565263395305
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.bread_macro_run \
  --out output/powersim_bread/prune_speed_1e-4/pruned_192 --intervals 192 \
  --load-multiplier 4.537565263395305 \
  --geometry output/powersim_bread/prune_speed_1e-4/geometry/material_geometry.npz
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.bread_prune_speed_report
```

已完成的共同区间计时：

| 比较区间 | 原模型 | 删除后 | 提速 |
|---|---:|---:|---:|
| 前 16 步（本次串行重跑） | 67.92 s | 9.98 s | 6.80× |
| 前 96 步（旧记录作对照） | 1568.91 s | 296.89 s | 5.28× |
| 前 146 步（旧记录作对照） | 3270.76 s | 609.68 s | 5.36× |

前 146 步子步总数 5,364 → 356，Newton 迭代 20,530 → 1,219，FGMRES 迭代
143,364 → 13,771，CG 迭代 817,109 → 401,271。平均加载区位移的最大差异为
原宽度 `3.60e-5`（`0.00360%`）；峰值下移 9.96956% → 9.96856%。这些仅是
宏观响应对照，不是局部轨迹等价或时间精度证明。后段删除后仍出现细分，未消除
所有困难自由度。长段基线来自此前同设备记录，不能当作同时期多轮控制计时。
原模型第 147 步失败；删除后的任务完整完成 192 步，求解耗时 `1185.88 s`
（19.76 分钟），共 1,227 个接受子步，11 次失败尝试经细分恢复，检查通过。
末尾第 191、192 步各需 128 个子步（最大深度 7），说明此次删除没有根除
所有后期困难自由度。最大功/能量相对误差 `2.07e-7`，固定端保持精确不动。
结果见 `pruned_192/mechanics.json`、`comparison.json` 和 `comparison.png`。
所有原始结果保留，未覆盖默认几何或 checkpoint。

### 小规模 cell 刚度反演：可微有限运动 DEM

`bonded_dem_differentiable.py` 为当前有限运动黏结模型添加小规模 CPU 隐式反向
求导：直接调用 `FiniteBondsGPU.trial.__wrapped__`，复用原法向/切向/弯曲/扭转
历史及接触偏移运输规则。正向求解相同隐式中点方程，反向解转置 Jacobian，
再对参数和上一步状态计算梯度；没有对 Newton 迭代展开求导。位置、速度、
转动、法向、偏移及四种黏结历史均保留在跨时间步计算图中。稠密 Jacobian
目前限制在最多 240 个自由 DOF，未接入全尺寸 GPU 训练。

`bread_inverse_pilot.py` 从删除小 cell 后的真实面包参考几何中截取一个连通
子图：16 个 cell、49 条黏结，固定最左 4 个 cell，其余 12 个自由。保留原
体积、半径、面积和面中心，去掉连向外部的黏结。因此它是新的局部试件，
不代表完整面包中的原位边界条件。选点仅按几何、体积和连通性，不按拟合结果。

每个 cell 学习四个 log 刚度倍率，共 64 个参数；黏结倍率取两端倍率的几何
平均，保持对称、正值。训练用 `theta = 1.5*tanh(raw)`，初始 raw 为零；
质量、外力、边界、几何和时间步固定，没有神经网络、耗散、塑性或损伤。

目标数据由已有 `NewtonBonds` 正向求解器使用另一组空间变化的刚度生成，
不是实测视频或 FEM 数据。每段 24 步，共约 0.04217 个模型时间单位，观测为
自由 cell 的中心位置，不使用转动或 RGB 标签。下压和拉伸两段用于训练；
侧向加载在方向、施力区域和幅值上不同，完全不进入优化目标或参数选择。
已知目标参数仅用于生成数据，不作为参数拟合损失。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.bread_inverse_pilot \
  --out output/bonded_dem_inverse/pilot_16cells
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.bread_inverse_report \
  --run output/bonded_dem_inverse/pilot_16cells
```

检查结果：可微正向与原求解器在完整训练片段上的最大位置差为试件宽度的
`2.17e-11`；最终全部状态的相对差最大 `1.68e-9`。整段运动损失对参数的
方向导数与中心有限差分的最小相对差 `6.36e-10`。梯度包含之前步骤的黏结
历史影响，不是仅对单步、固定历史求导。

40 次 LBFGS 迭代、44 次目标函数/梯度计算，CPU 单线程训练 136.64 秒。
运动误差定义为所有自由 cell、全部非初始帧的坐标 RMSE，除以该片段目标
位移的坐标 RMS；它不是图像误差或相对于物体尺寸的误差。

| 加载 | 训练前 | 训练后 |
|---|---:|---:|
| 下压（训练） | 54.13% | 0.878% |
| 拉伸（训练） | 53.80% | 0.990% |
| 侧向（留出测试） | 83.49% | 1.463% |

结果文件包括 `checks.json`、`training.jsonl`、`report.json`、学得的参数、
每个加载的完整前后轨迹及 `learning_summary.png`。本轮仅证明可微反演链路
在同模型合成目标上有效，并展示该试件的留出加载预测。未证明参数唯一可辨识、
真实材料精度、独立时间精度或不同力学模型间的拟合能力；后者需要独立目标数据。

### 统一刚度与逐 cell 刚度对照

`bread_inverse_capacity.py` 复用上述 16-cell 试件、训练轨迹和冻结的逐 cell
拟合结果，另外训练所有 cell 共用四个刚度倍率的基线。两者从倍率 1 开始，
使用同一参数范围、正则化和两段训练运动，以及最多 40 次 LBFGS 迭代。
统一倍率仍乘在原有按面积、长度计算的每条黏结刚度上，不是让所有黏结的
绝对刚度相等。所有实验串行执行；旧逐 cell 模型不重新训练或选择。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.bread_inverse_capacity \
  --out output/bonded_dem_inverse/capacity_16cells
```

测试包括原有侧向加载，以及三个预先固定的新加载：最右四个 cell 上绕 x
轴的零合力力偶、最右四个 cell 中最高的单个 cell 向下受力、两倍训练下压力。
前两个加载仅用未训练的参考线性刚度设置幅值，使静态最大位移为宽度的 3%；
不按目标参数或拟合结果设置幅值。时间步、脉冲时长和总时长与训练相同。
各测试独立从静止开始，均不参与训练或模型选择。

输出 `config.json` 在优化前记录测试设计；`parameters.npz` 保存两组模型参数；
`report.json`、`comparison.png` 和六段 `*_comparison.npz` 记录误差与完整轨迹。
脚本核验原求解器哈希、复现旧逐 cell 误差，并检查轨迹有限且固定端精确不动。
目标仍来自同一 DEM 的空间变化刚度，不能将这个对照视为真实材料或独立 FEM
验证，也不代表已经找到统一参数模型的全局最优解。

对照已完成。统一模型在第 27 次接受迭代收敛，共 30 次目标计算；逐 cell
模型沿用原来的 40 次迭代、44 次目标计算。两者上限相同，不强制让已收敛
的基线继续迭代。统一模型四个倍率依次为 1.98746、0.75160、1.38146、1.04250。
本次训练约 300 秒，旧逐 cell 训练约 137 秒；两次运行时间不同，不能据此
得出参数更多的模型训练更快。

| 加载 | 统一 4 参数 | 逐 cell 64 参数 |
|---|---:|---:|
| 下压（训练） | 2.056% | 0.878% |
| 拉伸（训练） | 2.246% | 0.990% |
| 侧向（测试） | 2.471% | 1.463% |
| 扭转（测试） | 1.848% | 0.998% |
| 单 cell 下压（测试） | 11.747% | 13.540% |
| 两倍下压力（测试） | 2.331% | 1.024% |

误差仍为位置 RMSE / 目标位移 RMS。所有检查通过，旧逐 cell 三段结果精确
复现。逐 cell 在三个留出加载上改善，但局部加载反而略差，两者局部误差都
明显上升。说明拟合两段整体运动后，当前模型尚不能可靠预测任意局部受力；
不能把较低训练误差直接等同于已学准所有局部材料响应。这个单例尚不能区分
训练信息不足、参数歧义或有限优化预算的影响。建议下一个受控实验加入局部
加载训练，并在不同位置的局部加载上测试，再接独立 FEM 目标。

### 区分优化不足与训练约束不足

`bread_inverse_diagnose.py` 保持原有下压、拉伸两段训练数据和正则化不变，
串行执行三组逐 cell 参数拟合：从原 40 次迭代的结果继续，以及从种子 1729、
3141 生成的 `raw ~ Uniform(-0.2, 0.2)` 分别开始。每组累计迭代上限为 160，
每 40 次新增迭代检查全部六段运动。留出测试不用于梯度、停止或模型选择。
原任务没有保存 L-BFGS 状态，因此旧参数继续训练时重置优化器历史；本次各
40 次迭代块之间保留历史，并保存 `optimizer.pt` 以支持后续完整续跑。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.bread_inverse_diagnose \
  --out output/bonded_dem_inverse/diagnose_16cells
```

首先用已知生成参数检查局部加载的正向复现，但不将这些参数用于初始化或
训练监督。每组记录训练误差、局部预测、其他留出误差、最终梯度和停止原因。
另计算模型之间的训练轨迹差异与局部轨迹差异；参数不同本身不作为物理预测
不确定的证据。达到迭代上限不表示已收敛；有限次多初值实验也不能证明参数
不可辨识。两种原因可能同时存在，需按结果分别判断。

三组均已完成预定预算，全部因累计达到 160 次迭代停止，未达到优化器收敛
条件。已知生成参数对局部加载的归一化误差为 `4.61e-10`；所有预测有限，
固定端检查通过。结果见 `output/bonded_dem_inverse/diagnose_16cells/report.json`
及 `comparison.png`。

| 模型 | 累计迭代 | 两段训练综合误差 | 单 cell 加载误差 |
|---|---:|---:|---:|
| 原模型 | 40 | 0.936% | 13.540% |
| 原参数继续训练 | 160 | 0.129% | 1.838% |
| 随机初值 1729 | 160 | 0.312% | 4.272% |
| 随机初值 3141 | 160 | 0.306% | 4.300% |

原参数继续训练的局部误差在累计 80、120、160 次迭代时依次为 4.275%、
2.297%、1.838%，没有增加局部加载监督。这直接说明原 40 次优化不足是此前
局部预测差的重要原因。两个随机初值最终训练误差接近，但它们的局部预测
彼此相差目标位移 RMS 的 2.422%；训练下压和拉伸轨迹彼此相差 0.161%、
0.244%。这个差异尚不能证明训练信息不足，因为三组仍未收敛，最终 raw
梯度最大绝对值分别为 `2.79e-5`、`1.37e-4`、`3.16e-4`，均高于 `1e-8`。
因此当前优先结论是优化不足已得到验证；数据是否足以唯一约束局部响应仍
未确定，不能把不同初值下的有限预算结果当作不可辨识的证明。

### 独立 FEM 目标：共用 Power Foam 试件

**四类黏结的静态模量与内部残余诊断：** `foam_dem_moduli.py` 等待本轮训练
完成，然后依次检查初始、统一倍率、逐 cell 倍率的参考态切线。切线调用原
`LinearBondsGPU.stiffness`，并用 `FiniteBondsGPU.trial` 的小增量响应校验；
不另写一套力学来代替当前模型。保存完整 6×6 工程应变刚度矩阵、方向性
Young/剪切/Poisson 响应、均匀应变及均匀应力下的体积模量。两者不相同时
不把它们混为单一各向同性常数。测量边界是表面 cell 中心的仿射平移，全部
转角与内部中心允许松弛；这不同于训练中左端夹持的动力学边界。

对每 cell 的四个 log 倍率给出静态模量灵敏度，通过重新求静态平衡的方向
有限差分核对。FEM 使用已有 q050 曲面 P2 网格，直接计算六种仿射场的能量
和内部残余，不细化网格或重算轨迹。FEM 的边界位移场和 DEM 的中心边界
不是逐点相同的约束，结果用于有限试件的响应比较，不能当作精确同边界的
误差收敛试验。

16-cell 试件没有真正的内部 cell，内部残余字段为 null；另建同样等半径
规则的 4×4×4 CAD 试件，包含 8 个内部 cell，只取体积/分界而不生成网格。
在均匀胀缩下比较仿射与松弛状态。候选体积能量只按单位 `k_v*m_v=1` 单独
计算仿射内力，报告几何一致性残余，不进入现有四类黏结训练或动力学。

```bash
PYTHONPATH=output/fem_dependencies OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m experiments.foam_dem_moduli --out output/foam_fem/moduli_v1 \
  --wait-for-training
```

状态 `output/foam_fem/moduli_v1/status.json`，日志 `output/foam_fem/moduli_v1.log`。
只有最终 `report.json` 存在且 complete=true 才表示诊断实际通过；脚本写完
或排队并不表示已经得到结果。该试验不计算刚体 cell 像与渲染分区的对称差，
也不默认启动第五倍率拟合。体积项是否采用仍需单独的四类/五类模型验证加载
对照；用于做这种选择的加载不再是最终独立测试集。

**2026-10-08 首轮结果：** 两组训练和串行静态诊断均已完成。两组各达到
40 次迭代上限，未达到优化器停止条件；不能据此宣判材料表达的最终上限。
以下轨迹误差均为相对目标自由中心位移 RMS 的归一化位置 RMSE：

| 参数形式 | 三段训练综合 | 留出侧向 | 留出局部 | 留出两倍下压 |
| --- | ---: | ---: | ---: | ---: |
| 统一四倍率 | 16.416% | 9.052% | 9.865% | 9.048% |
| 逐 cell 四倍率 | 11.180% | 6.516% | 7.100% | 6.523% |

最终参数下时间步减半的轨迹差最大约 0.086%，两组均通过本轮检查。静态
测得的逐 cell 模型表观体积模量为 412.04（均匀应力定义），方向剪切模量
251.91～282.77，方向 Poisson 比 0.0263～0.0641；FEM 数值仿射能量测量
对应 K=800、G=480、nu=0.25。这是有限试件和上述边界定义下的表观比较，
不能解释为任意 DEM 的普遍上限。参考态切线与实际有限运动力的小增量响应
一致；六个刚体零模式、静态平衡及倍率灵敏度检查通过。

64-cell 规则试件的原四类 pair 模型在 0.01%、0.1% 均匀胀缩下内部残余力
与非仿射松弛在舍入误差量级。候选体积项的相同检查不通过：F 虽准确恢复
均匀胀缩，但 8 个内部 cell 出现非零力，归一化后不随应变减小而消失。
单位 `k_v*m_v`、应变 0.001 时，内部力整体范数 0.0117813；除以相应
一阶 Piola 压力后的几何残余为 3.91535（面积量纲）。该项仅单独求力，
尚未做加入后的轨迹训练，不能判断留出误差是否改善。原模型通过规则试件
不代表任意不规则分区也通过；新项的残余不能仅靠加密时间步消除。

**2026-10-08 执行修正：** 用户已停止参考加密检查。首轮使用已经完整导出的
`reference_edge007`，撤销 0.5% 网格变化的前置门槛，保留原报告不变：相邻
网格轨迹差 0.90%～1.58%，16→32 子步最大差 0.298%。这不是 FEM 对真实
材料的误差上界。以下旧加密过程保留为历史，不再作为当前执行安排。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.foam_fem_pipeline \
  --reference output/foam_fem/reference_edge007 --exploratory-reference \
  --out output/foam_fem/inverse_exploratory_v1 --max-iterations 40
```

该入口只读取已有数据，不生成或细化 FEM。统一 4 参数、逐 cell 64 参数各
最多 40 次迭代，依次运行，每 5 次保存参数和优化器。仍保留较小 DEM 系统的
时间步检查，防止将积分误差当成模型能力。初始检查记录每次耗时，训练记录
每段运动的正反向耗时和误差；参考检查原样复制到 `reference_checks.json`。
训练日志为 `output/foam_fem/inverse_exploratory_v1.log`，总状态仍在
`pipeline_status.json`。到达 40 次不等于收敛，也不能据此判定模型精度上限。

设计和判断顺序见 `RESEARCH_ROUTE.md` 的 2026-10-08 当前路线。新入口：

- `foam_fem_geometry.py`：16 个等半径、4×2×2 排列的 bounded power cell。
  用 Gmsh 将球与其精确 power 区域相交，生成共形曲面二次四面体。
  cell 体积、28 条分界的面积及面中心来自同一 CAD 几何；x=3 额外分割仅
  为准确划定施力表面，不新增 cell 或黏结。
- `foam_fem_reference.py`：scikit-fem P2 连续体线弹性和一致质量，独立中点
  积分。夹持 x<=0.5 的材料；表面力映射为 cell 合力及关于中心的合力矩。
  材料点使用所在曲面单元的形函数插值，不是最近节点替代。库默认批量点
  定位在远离目标点的候选曲面单元上可能 Newton 失败，因此逐候选反求局部
  坐标并检查包含关系；刚体/仿射场插值、应变能和质量恒等式均作检查。
- `foam_fem_inverse.py`：默认使用原参考门槛，`--exploratory-reference` 显式
  接受完整参考中的已知离散敏感性并记录；同一有限运动 DEM 串行拟合统一
  4 参数、逐 cell 64 参数，迭代预算可设置。内部子步在
  初始参数上按各训练运动检查；最终参数再检查时间步减半。测试不用于优化。
  每个训练片段分别反向后释放计算图，以控制长时间序列的内存占用。

依赖局部安装在 `output/fem_dependencies`：gmsh 4.15.0、scikit-fem 12.0.2、
meshio 5.3.5，运行使用 `om` Python。Gmsh wheel 的 `libgmsh.so.4.15` 被
pip --target 遗漏时，从同一个已下载 wheel 恢复到该依赖目录；不修改全局
环境或库源码。命令例子：

```bash
PYTHONPATH=output/fem_dependencies OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m experiments.foam_fem_geometry --out output/foam_fem/mesh_q025 --mesh-size .25
PYTHONPATH=output/fem_dependencies OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m experiments.foam_fem_reference --geometry output/foam_fem/mesh_q025/geometry.npz \
  --out output/foam_fem/reference_q025 --config output/foam_fem/reference_q050_v3/config.json \
  --compare output/foam_fem/reference_q035 --min-substeps 16
```

每次输出到新目录，不覆盖旧结果。早期 `mesh_h050` 是平面边界几何检查，
体积差 3.885%，没有用于 FEM 目标；`mesh_q*` 使用曲面二次几何。首轮曲面
粗网格的时间步检查需要每观测间隔 32 子步，16→32 子步的最大轨迹差为
0.316%。q050→q035 网格加密的轨迹差仍为 2.14%～2.97%，不满足 0.5%
参考门槛，因此 q035 报告明确不允许训练；正在用更细网格验证。当前还没有
独立 FEM 目标上的训练结果，不能声称模型已经拟合了连续体材料。

q025 也已完成：最大时间步变化 0.306%，q035→q025 空间网格变化仍为
1.50%～2.39%，因此继续禁止训练。后续改为对实际球面交界、夹持边缘的
46 条外部交线局部加密；不改变材料、几何域、载荷或 0.5% 门槛。
`mesh_edge010` 为主体尺度 0.35、交线尺度 0.10，16,265 个单元；对应参考
计算正在进行。线性求解采用 MMD_AT_PLUS_A 重排序以控制因子填充，仍解
同一方程，并检查首步线性残差与整段功—能量关系。

`foam_fem_pipeline.py --wait-reference` 已作为后台串行任务启动。它等待
当前 edge010 完成；必要时依次生成 edge007（0.30/0.07）、edge005
（0.25/0.05）参考，再启动 `foam_fem_inverse.py`。每一步等待上一进程退出，
不会并行运行模拟。若这两档后仍未通过参考检查，或网格超过 60,000 个 P2
四面体的本机内存预检上限，则明确停止并写状态，不绕过参考门槛。

当前状态入口为 `output/foam_fem/pipeline_status.json`，总日志为该目录下
`pipeline.log`。若进入训练，输出为 `inverse_v1/`，日志 `inverse_v1.log`。
训练器支持 `--resume`，读取已保存的时间步检查和 L-BFGS 状态；参考路径
必须相同。初始时间精度、最终梯度、参数贴近界限的比例、最终时间步检查均
保留。源码语法检查通过；训练入口尚待合格参考数据上的实际运行验证。
