# Per-cell deformation experiment

> **Scope correction (2026-09-25):** This custom geometry probe uses frozen
> appearance and zero displacement. Its material-label objective is an added
> hypothesis, not the original paper's image-reconstruction criterion. It does
> not validate the complete dynamic Power Foam model. See the
> [paper/code reading notes](../POWERFOAM_READING_NOTES.md).

Completed locally on 2026-09-24 with the existing `om` environment and RTX 5070 Ti
Laptop GPU. Reproduce from the repository root:

```bash
conda activate om
python -m experiments.varying_metric_render
```

## What is moving

Each cell has its own center and deformation gradient. It competes with all
other cells using `q_i = |F_i^-1 (x-p_i)|^2 - r_i^2`; interfaces are consequently
generated quadrics. Dipole normals and canonical texture coordinates follow the
same local deformation. No position, radius, interface or appearance parameter
is fitted per frame. Reference radii stay fixed; both tested maps have determinant
one, so the spherical baseline's volume-radius update also leaves radii fixed.

The scene contains 121 initially coplanar cells, eight frozen texels per cell,
zero displacement and view-independent color. Bend means the nonlinear vertical
shear `(X,Y,Z) -> (X,Y,Z+a sin(pi X/2))`, not an isometric beam bend. Twist rotates
the YZ plane by `a X`. Both maps are injective with known inverse and Jacobian.
They provide controlled deformations, not simulated mechanics.

## Material transport

Identity error is the fraction of originally covered material samples whose
advected position changes cell owner or leaves every current support. The volume
probe starts with 32,768 uniformly drawn points in the reference support bounding
box and evaluates this fraction on the covered subset. Surface probes use a
192 by 192 grid within the interior reference patch.

| Deformation | Spherical volume error | Local-F volume error | Local-F surface error | Local-F volume support miss |
| --- | ---: | ---: | ---: | ---: |
| Identity | 0% | 0% | 0% | 0% |
| Bend, a=0.2 | 8.0115% | 0.7432% | 0.0109% | 0.3374% |
| Bend, a=0.4 | 13.1868% | 1.4090% | 0.0488% | 0.6612% |
| Twist, a=0.6 | 13.9073% | 0.3648% | 0.3228% | 0.0684% |

All 121 cells receive volume probe samples in each case. This does not establish
exact cell volumes, exclude unsampled holes, or certify cells nonempty for every
possible motion. The surface grid omits outer cells even in the identity frame;
its unobserved-cell count must not be interpreted as cell disappearance.

| Number of cells | Bend surface identity error | Twist surface identity error |
| ---: | ---: | ---: |
| 49 | 0.1248% | 0.5588% |
| 121 | 0.0488% | 0.3228% |
| 289 | 0.0217% | 0.2224% |

These are independently generated jittered grids using one seed, not a nested
discretization or proof of convergence. Nevertheless, both probes improve as
the cell size decreases, consistent with the local-Jacobian approximation.

## Rendering validation

- A two-cell shell fixture yields owners `[0,1,0]` along one ray. Endpoints agree
  with closed-form roots and transparency with the analytic total path length.
- With a shared F, maximum RGB difference from the independent float64 oracle
  is `1.61e-14` in a transparent, oblique-camera test.
- Variable-F interval ownership agrees with direct power/dipole evaluation at
  all 98,304 points along 48 arbitrary rays.
- Analytic Jacobians agree with finite differences to `1.40e-10`; inverses and
  unit determinants also pass assertions.

Image panels are 128 by 128, extinction 1000 per world length, black background.
The nonlinear reference uses the exact inverse material map but **sampled ray
integration** (1024 steps), unlike the quadratic-event renderer of the local-F
model. Full-image RGB MAEs, with RGB in [0,1]:

| Case | Spherical vs sampled reference | Local-F vs sampled reference |
| --- | ---: | ---: |
| Identity | 0.00000937 | 0.00000921 |
| Mild bend | 0.00059444 | 0.00023947 |
| Strong bend | 0.00168099 | 0.00028445 |
| Twist | 0.00264442 | 0.00051467 |

Background pixels lower these means. Foreground Local-F MAEs are respectively
0.0000504, 0.001302, 0.001542 and 0.003040. At 64 by 64, refining the twist
reference from 1024 to 2048 steps changes mean RGB by 0.0001134, with maximum
channel error 0.1436 near sharp transitions. Thus image metrics are approximate;
the exact-map material-label measurements give stronger evidence of geometric
improvement. Texture querying also differs between spherical and local-F models,
so RGB improvement cannot be attributed solely to interface geometry.

The analytic scenes use up to four occupied segments per ray; they happen not
to reenter the same cell in the tested views. The shell fixture separately checks
that case. A 16-frame twist loop records material-coordinate color changes and
label errors; it is not sufficient to claim absence of temporal flicker or robust
topology transitions.

## Artifacts and remaining work

Generated files are Git-ignored in `output/varying_metric/`: `motion.gif`,
`comparison.png`, individual reference/baseline/local-F images and `results.json`.
The JSON contains validation checks, per-case metrics, refinement and temporal
samples. Rendering is a forward-only correctness prototype without acceleration
or a performance claim. No MPM, learned reconstruction, mass conservation,
contact, plasticity or long-sequence stability has been demonstrated yet.

The result supports testing actual elastic MPM positions and total deformation
gradients with this representation. It does not establish exact preservation of
one material region per cell under general nonlinear motion.
