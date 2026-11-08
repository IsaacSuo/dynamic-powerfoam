# 超弹性接触回弹：验证状态与修复记录

## 当前有效结果（2026-11-08）

完整幂图与球相交模型已完成一次 **140 ms 的撞击、压缩、回弹和离地**。
原始记录：`output/ale/hyperelastic_releasing_floorfix140ms.json`。
这是无重力、固定无摩擦地面的隔离试验，初始向下速度 1 m/s；不是长期重力落地试验。

- 12681 个实际时间步，平均约 11.04 us；累计诊断墙钟 3668.5 s。
  时间包含几何检查、分段续跑与部分并行验证的干扰，不是纯推进基准。
- 末端向上质心速度 0.720186 m/s，接触面积为零，最后至少 5 ms 持续离地。
- 独立几何检查把地面只在复算中抬高 5 mm，材料分组体积变化不超过 8e-15，
  仍无地面接触；抬高 10 mm 则切到物体。因此末帧具有 **至少 5 mm 的实际间隙**。
  没有用未裁切球的最低点替代真实材料边界，也没有用抬高后的地面继续推进。
- 逐步做功误差最大 1.846e-5 J，低于原 1.860e-5 J 阈值；最大局部相对能量亏空
  1.253e-11。记录帧的总能量误差最大 3.56e-14 J，质量保持到浮点精度。
- 机械能损失仍为 34.27%，转入独立记录的非弹性能量余项。
  **通过回弹不等于低耗散或空间收敛**，没有把这一损失当作真实材料阻尼。

证据：同名前缀的 `_contact_summary.json`、`_contact_history.png`、
`_gap_check.json`、`.gif` 与 `_render.png`。逐段接受状态及源码存档保留。
地面重合约束、高精度极薄格几何、保存续跑的验证见下方记录；
`hyperelastic_releasing_precise_entry/summary.json` 在原阈值下通过。
最终主入口从原始 foam 初始化的 50 ms 自由释放、保存续跑以及旧默认实际时间步
均已通过：`hyperelastic_final_entry/summary.json`。总验收记录为
`output/ale/hyperelastic_acceptance.json`；通过的范围和仍有的耗散、成本限制明确保留。

## 修复记录（历史状态，不替代上方当前结论）

**Newest boundary correction:** the hidden-cell-corrected trajectory passed
65.158 ms and began upward COM motion by 70 ms. At 73.133 ms, adding owner 99's
mathematically redundant floor plane changed group 890's volume by −1.3013%.
That insertion was rejected before committing any state. Both complete-hull
and conservative-topology cuts reproduced the same coincident-constraint
failure, locating it in the common geometry integration rather than missing
neighbors. At positive lifts of 1, 10 and 100 nm the relative volume changes
were continuous (3.77e-10, 3.77e-9, 3.77e-8).

The exact floor-face reuse below is therefore now enabled for correctness,
not claimed as a speed optimization. At exact coincidence it leaves volumes
and energy unchanged; as soon as the plane moves it uses the same original
cut. A GPU continuation of the saved failure state passed 1 ms / 105 steps in
29.9 s, with maximum work error 6.02e-7 J and no negative local energy remainder.
The existing accepted state recuts to its recorded energy within 2e-14 J, so
continuation from that saved state is valid; the rejected 1.3% volume jump was
never accepted. Main-entry/snapshot verification and long rebound continuation
are in progress under the `hyperelastic_releasing_floorfix*` tags.

The 80 ms continuation completed. A main-entry replay at 74.133 ms initially
used 14+15 versus 24 adaptive steps: tiny negative cut volumes triggered
different retries. Raw boundary-parameter differences exceeded the original
1e-7 test threshold; that failed check was retained, not reclassified as a
pass. The saved material/boundary arrays themselves were exactly preserved.

An actual rejected cut identified cell 3257: computed V=−3.94e-18 m³, versus
the existing roundoff bound 3.14e-18. It has only two real faces and no exposed
floor/vacuum face. The exceptional 60-digit positive-integral calculation now
also proves ALL vacuum generators and the floor inactive over the enclosing
cap. It gives V=4.684339299e-20 m³, changes its group's volume by only 9.07e-13,
and does not change q. The old real-only proof is never applied to a planar
domain. Disjoint balls are conservatively excluded before MP arithmetic;
uncertain constraints remain in the proof (101 checked here, 7608 excluded).

With this correction, the same production main-entry experiment passes its
UNCHANGED 1e-7 threshold: 10+10 versus 20 steps, zero rejected steps. Relative
differences p=5.75e-11, r=2.01e-10, q=2.03e-10, plane direction=2.25e-15 and
offset=5.44e-15. Records: `hyperelastic_releasing_precise_entry/summary.json`,
`hyperelastic_planar_negative_probe.json`. The already running 80→100 ms job
loaded the earlier implementation; subsequent chunks will load the new
verified geometric reevaluation. Its accepted states remain audited; rejected
cuts were never committed.

**Latest correction: earlier 60 ms boundary pass is superseded as correctness
evidence.** At 65.158 ms the extension requested release for owner 99 despite
an existing floor seed. The saved state exposed a real implementation error:
base-hidden real cell 4969 had no real neighbors, but augmentation generated a
spurious vacuum disk for it (area 8.15e-6 m²). Its centroid violates owner 99's
power inequality by 1.86e-5 m²; it lies inside another material cell. That false
face influenced the shared boundary activation. Adding half-spaces cannot
unhide an empty real power cell, so augmentation now excludes all base-hidden
material cells from new face construction. The saved-state regression now
agrees with an independent complete augmented hull: volume difference 5.3e-18,
and boundary area/normal/centroid moments also agree. The original failed
state advanced another 1 ms / 104 steps on GPU in 50.3 s; maximum work error
2.0e-6 J is below the unchanged 1.86e-5 J tolerance, with no negative local
energy remainder. The corrected trajectory is now restarting before any
vacuum boundaries were inserted. Free-release material/MPM results do not use
this boundary extension and are unaffected. Previous run artifacts remain
unchanged; this correction is not described as a successful rebound yet.

The floor-reuse implementation was initially kept experimental for its limited
speed benefit, then enabled for the coincident-constraint correction above.
A vacuum plane exactly coincident with the ground
is redundant everywhere above ground and can relabel its owner's existing
floor face. Full-state CPU geometry and RHS agree with explicit generators to
approximately 1e-12. In a mixed 14-plane state the CPU cut cost only changed
0.780→0.770 s; that is not a meaningful measured speedup. Files:
`fv_ale_planar_floor_reuse.py`, `hyperelastic_planar_floor_reuse_check.json`.

**Earlier pre-correction status, retained for provenance:** the moving power-plane extension passes
full-state CPU and GPU short release checks and completes the 60 ms impact
continuation. The centre of mass is still descending at 0.185 m/s, so complete
rebound is not yet established; the same run is being continued. Earlier fixed-plane and
prototype limitations below are retained as development history.

At 60 ms: 5100 total physical steps, 761.4 s cumulative diagnostic time;
the 47.915→60 ms boundary continuation itself took 449.1 s. Mass and total
energy error are at roundoff, maximum per-step work error 4.18e-7 J, maximum
per-group normalized negative remainder 1.25e-11. J ranges 0.638–1.089. The
geometric boundary bookkeeping is expensive: 1116 insertions and 802 checked
prunings. These timings include that cost; no boundary speed claim is made.

The actual main scene entry also passed direct versus saved/split continuation
with moving planes. The seed was an explicitly recorded format conversion of
the audited GPU contact replay, without changing p, r, q, groups or planes.
Relative differences after 20 us were p=2.8e-11, r=6.8e-10, q=3.6e-9,
plane direction=2.0e-14 and offset=1.9e-13. This validates serialization and
the main entry path, not a new fresh landing. Records:
`output/ale/hyperelastic_releasing_entry/summary.json`.

The neo-Hookean constitutive response is not what prevents lift-off. The
current plane path represents the body as the union of its power-ball cells
intersected with one **fixed** ground half-space. Wherever a ball/cell still
crosses that plane, the geometry has a planar material boundary with zero
normal mesh velocity. The old wall Riemann state also enforces zero normal
material velocity. Compression/slide is supported; a tensile prediction is
explicitly rejected.

Clamping wall traction to zero and continuing would not implement separation:
the planar face would remain at the floor while its putative free-boundary
velocity points upward. Mass, geometric volume, deformation and boundary work
would again refer to different motions. Removing that check is not a fix.
Changing F/H or normalizing its determinant cannot provide the missing boundary
position either.

The standard cell-centered Lagrangian contact approach can release moving
boundary nodes and consistently reconstruct boundary geometry. The paper is
Morgan et al., *An approach for treating contact surfaces in Lagrangian
cell-centered hydrodynamics*, JCP 250 (2013), 527–554,
https://doi.org/10.1016/j.jcp.2013.05.015 . Its boundary-node degrees of freedom
are not present in this power-ball/fixed-plane representation, so it is not a
drop-in wall-flux replacement.

Two geometric extensions were initially considered (the second has since been
implemented through shared vacuum generators, as detailed later):

* A moving common clipping plane can represent an initially flat bottom moving
  away from the floor. It imposes a common normal velocity over that patch;
  this behaves like a massless rigid platen and can transmit local tension.
  It is not a general unilateral rubber-ground boundary. It must not be used
  to claim that arbitrary-cell lift-off is solved.
* Per-cell moving material planes allow local separation, but adjacent cells
  then have partly shared and partly free faces. Geometry must distinguish
  their overlap from the newly exposed strips; moment integrals, mass flux,
  shape source and boundary work must all use those same surfaces. This is a
  substantial interface-tracking extension, not a one-line change to pressure.

An opt-in `wall_mode=compatible_slip`, requiring `friction=0`, now supplies
the compatible-affine force with a fixed-plane boundary term. The tangential
wall velocity is reconstructed by the same affine map, its normal velocity is
zero, and the force is the negative transpose of that exact gradient map.
There is no second acoustic reaction added on top, and stationary-wall work
is zero. The tensile guard is retained. This is a compressive-wall experiment,
not a completed release implementation; it is not the default.

`hyperelastic_compatible_wall_check.json` checks all 7,690 original geometry
sites with an actual ground cut. For acoustic blends 0, 0.1 and 1, independent
autodifferentiation of the scalar power agrees with the force to 4e-15,
net force agrees with the integrated normal reaction to 1.3e-13, and total
energy rate is approximately 1e-12 W. The old Wilkins clipped-wall RHS agrees
with its archived pre-hyperelastic version for friction 0 and 10. These are
frozen-state checks only. The CPU and GPU versions now both pass, including
graph replay. A complete 5 ms first-contact run also passes its unchanged
energy checks, but is still almost entirely translating downward at that
early time; it is not a rebound. A 50 ms impact run is being tested with the
new local energy guard. Separation remains unvalidated.

### Acoustic correction and the actual onset of release

The first `compatible_slip` run stopped at 17.534 ms when a new face had
tensile constitutive stress even though it was approaching the wall. Looking
only at `-sigma_yy` omits the incident-velocity part of the wall Riemann
reaction. This early stop was not sufficient evidence of physical separation.
The old experimental mode is preserved for reproduction; a separate
`compatible_slip_acoustic` mode adds only the missing impedance correction.

Let v_f,y be the affine-reconstructed normal velocity at the floor face. The
normal reaction is `p*=-sigma_yy-Z*v_f,y`. Its additional force is the negative
reconstruction transpose of the derivative of `A*Z*v_f,y^2/2`; its mechanical
loss `A*Z*v_f,y^2` is retained as nonelastic energy. The elastic reaction is
already present and is not added a second time. Stationary-wall total work is
zero. The tensile stop now tests p*, and is not removed or relaxed.

Independent full-foam CPU/GPU scalar-power differentiation agrees with this
force to 4.1e-15, the integrated reaction to 1.4e-14, and the energy-rate
ledger to 1.2e-13 relative. GPU graph replay agrees to 1.8e-16. Stress-free
incoming and outgoing tests respectively produce compression and request
release. The actual impact run reaches 47.91458 ms, then stops at cell 2919
as p* crosses zero. That cell's reconstructed normal velocity is -0.480 mm/s
and sigma_yy is +9.83 Pa; the traction-free boundary velocity is approaching
zero from below. The force changes sign on the next trial. The ball's own
lowest-point velocity is still -2.04 mm/s, so sphere motion alone will not
open the required planar gap. This is the missing boundary representation,
not a constitutive flip or a local energy failure (maximum relative energy
deficit 1.3e-11 in this run).

The existing power-ball geometry can insert a single weighted vacuum
generator at that floor face without changing the domain beyond rounding:
full-foam volume relative difference 4.5e-16, maximum group-volume relative
difference 4.9e-14. A positive radius rounded upward gives a certified plane
height at or above the floor (4.2e-17 m offset). Coincident floor constraints
are explicitly deduplicated. Cancellation-sized free-plane disks are
reevaluated with 70-digit area, first moments and arcs; they are not assigned
an area floor. See `hyperelastic_acoustic_impact50ms_release_geometry_probe.json`.
This is a geometric prerequisite only: moving-boundary dynamics is not yet
implemented and no rebound has been claimed.

Additional literature check: the hybrid Lagrangian Voronoi--SPH work
(https://doi.org/10.1007/s40571-017-0173-4, sections 3.4.3--3.4.4) uses virtual
free-surface generators and reflected wall generators. Their transition gives
an example of a Voronoi boundary that can separate. It is a fluid method and
its free-surface placement approximates zero pressure; this is not a verified
hyperelastic traction/energy update for the present solver. Adding independent
vacuum boundary generators could preserve a power-diagram representation, but
would add geometric state, boundary motion equations and planar free faces.
It is therefore an unimplemented geometric extension, not evidence that the
existing fixed clipped plane already supports lift-off.

## A possible power-diagram boundary representation (not implemented)

Ordinary reflected generators assume that a real generator stays above the
floor. The current program does not require a generator to lie inside its
clipped cell, so literal reflection would be invalid for generators below the
plane. There is a weighted construction that avoids that particular problem.
For a common positive offset d and a plane y=h, define a geometric vacuum
generator paired with each material generator:

    p_g = p_i - d e_y
    w_g = r_i^2 + d^2 - 2 d (p_i,y - h).

For the power functions psi=|x-p|^2-w, this gives exactly
`psi_g(x) = psi_i(x) + 2 d (y-h)`. With all pairs and the same d and h,
`min psi_g = min psi_i + 2 d (y-h)`. Selecting material ownership therefore
recovers the original ground half-space exactly, even when generators lie
below the plane. The real material cells still intersect their original balls.
This is an algebraic observation, not a validated contact method.

Allowing separate h_i would provide geometric state for a bottom free boundary
after release. All real/ghost shared faces would then be explicit planar free
faces, automatically including the overlap/strip changes missing from separate
per-cell cuts. It would also introduce several unresolved requirements:

* The global ghost lower envelope matters. A nearest-neighbor ghost list is
  not automatically complete when h_i differ.
* A ghost power weight can be negative, whereas existing geometry kernels tie
  weight to a real squared ball radius. Silent clipping of that weight would
  change the boundary. Removing or reparameterizing a ghost also needs an
  exact unchanged-domain criterion.
* Geometry needs an independent planar-free-face velocity and its same
  transpose in mechanical work. Appending empty cells to the existing
  positive-mass material arrays is invalid.
* Unilateral contact needs a compressive reaction and nonpenetrating boundary
  motion, followed by release when appropriate. Merely clamping h_dot cannot
  establish momentum/energy consistency.

Thus this could retain the required power-ball partition, but it is still a
boundary-state extension with force, flux and contact work to derive. No
current free-release result is being labeled a ground-rebound result on the
basis of this construction.

### Pending compatible moving-plane derivation (not a working solver)

Store each geometric vacuum boundary as owner o, positive fixed d, covector a
and offset b. Reconstruct its generator q=p_o+d*a and weight
w=r_o^2+d^2*(a.a)-2*d*(b-a.p_o). Thus psi_g=psi_o+2*d*(a.x-b).
The floor corresponds to a=-ey, b=-ground. Advect free a by -G_o^T*a;
do not normalize a without also transforming d. This is objective under rigid
motion and keeps an active fixed floor exactly fixed through RK stages.

For a real/vacuum face with real i, center c, normal n, area A,
ell=|q-p_i| and k=d/ell, its geometric normal velocity is s=s0+k*bdot,
where s0=((c-p_i).w_i+r_i*rdot_i-(c-p_o).w_o-r_o*rdot_o
-d*c.adot)/ell. Use the material boundary velocity
u=(I-nn^T)*U_i(c)+n*s in the SAME gradient and geometric volume rate.
With impedance Z, define D=sum Z*A*k^2, F=sum A*sigma_nn*k.
The proposed free bdot is sum Z*A*k*(n.U-s0)/D-F/D. Its stress correction
dissipates F^2/D. No mass flux crosses this material boundary.

The negative transpose must include the geometric controls. Define
L_e=(F/D)*Z*A*k, B_e=A*sigma_nn-L_e,
T_e=A*(I-nn^T)*sigma*n+L_e*n. The kinematic boundary power is
sum T_e.U_i+B_e*s0. Its direct force is -T_e, affine conjugate
T_e outer (c-center_i), and site/radius conjugates are respectively
+B/ell*(c-p_i), +B/ell*r_i at i and their owner-subtracted terms at o.
The a-rate conjugate is sum -d*B/ell*c; transposing adot=-G_o^T*a adds
-a outer that conjugate at o. Transpose the existing balanced sphere-motion
map too. Independent autodiff of this scalar power and actual recut volume
differences are required before using these formulas for advancement.

On a horizontal floor, hold adot=0 and constrain bdot<=0 (opening upward).
The own-face free solution is bdot=p_star/Z, so positive bdot activates the
fixed contact boundary and negative bdot releases. Active own faces use the
existing compatible acoustic wall operator, with no duplicate floor area.
Other cross-pair faces need the clamped-control transpose, not deletion.
The global ground half-space remains in the cut. Partial recontact,
multiple boundary patches, all RK stages, snapshots and rendering must be
handled explicitly. None of this has yet been certified by a full rebound.

Geometry insertion probe on the full saved 47.9 ms impact state now passes:
relative volume change 4.4e-16; maximum grouped relative change 4.9e-14;
a 100 micrometer boundary lift has volume-rate finite-difference error
1.7e-8. The formerly inaccurate tiny cross-face is reevaluated with exact
input doubles at 70-digit precision, including centroid and arcs. This
establishes geometry only, not force or time-integration correctness.

### Integrated moving-plane implementation and checks (2026-11-08, ongoing)

The derived operator is now in `fv_ale_planar_mechanics.py`, with actual
power-ball geometry in `fv_ale_planar_boundary.py` / `fv_ale_planar_cutter.py`
and full SSPRK2 integration in `fv_ale_releasing.py`. It is an explicit opt-in:
the old fixed-wall and Wilkins defaults are retained. No vacuum generator has
mass or material state. Each saved state/frame retains its plane covectors,
offsets, owners and distances. The moving-plane renderer clips rays against
these power half-spaces; rendering only the real ball union would refill gaps.

Validation so far (not yet a full rebound):

* Actual full-foam topology with three nonuniform/tilted boundary patches
  agrees with a complete augmented power hull. The independently differenced
  volume rate has relative error 5.3e-8. This includes cross-owner faces.
* The full-state negative transpose agrees with autodiff of the same scalar
  power to 4e-15. Independent mechanical/heat accounting closes to 2e-14.
  Recut geometry finite differences agree with mechanical power to 9.4e-10.
* Rotating and translating the coordinate system transforms boundary rates
  and forces correctly (7.5e-15 error), including the offset-rate transform.
* From the original stopped 47.91458 ms impact, the production RK2 code on
  CPU advanced another 0.1 ms / ten steps. Owner 2919's redundant boundary
  was inserted without changing q; maximum grouped volume change 4.9e-14.
  Maximum work error was about 1.4e-9 J, no local negative energy remainder,
  and serializing/rebuilding the boundary reproduced volumes exactly.
  The matching GPU replay now passes ten steps / 0.1 ms in 2.85 s, maximum
  work error 1.4e-9 J and serialized recut error 1.1e-16. Long rebound remains
  pending; this short check is not evidence of complete separation.

Diagnostic opt-in: `--releasing-planes`. Converting a saved old fixed-wall
run additionally requires `--upgrade-release-boundary`, which is recorded
as a model change, not described as identical-config replay. Main scene
entry uses `configs/hyperelastic_releasing_drop.json`, `--mu 0`, plane ground,
and the existing hyperelastic entry. Use the validated experimental CFL 0.5,
not the older plastic-material default 2. Geometric derefinement is blocked
for this mode because boundary owner indices must be retained; P0 material
agglomeration remains available.

The correctness-first boundary cut uses batched tensor topology assembly and
an exceptional high-precision tiny-face path. Its cost is reported; no speedup
claim is made. More than the supported half-plane capacity, invalid geometry,
or nonpositive weights explicitly stop/retry rather than silently deleting
constraints or changing material state.

A longer 1 ms full CPU replay exposed a representation issue: partial
recontact/release can create successive power planes for one owner. The first
replay completed with 39 planes, but only six had any exposed face. An
independent removal/re-cut probe showed that removing the 33 unexposed planes
changed every real volume and ground area by exactly zero. Keeping only the
newest plane per owner did NOT preserve geometry (maximum group volume change
3.2e-5), so that shortcut is not used.

The integrated cleanup only considers zero exposed area and re-cuts/checks
the domain before accepting a removal. Fully hidden real cells are therefore
not accidentally recovered by deleting their occluding plane. A newly seeded
horizontal floor plane is retained through its trial even if currently
unexposed; removing that seed caused an activation/retry loop in the first
cleanup test. That test was stopped, not reported as a completed run. The
corrected 1 ms replay passed 105 steps in 280.0 s on CPU, ended with six planes,
and performed 27 checked cleanup operations. Mass error and serialized recut
volume error were zero, maximum work error 1.4e-9 J, and local negative energy
remainder was zero. Geometry-only insertion/removal changes
in mechanical energy are explicitly recorded as roundoff, never reset.

Historical failed prototype probe (before the high-precision disk corrections):
a read-only full-foam geometric probe is saved as
`hyperelastic_ghost_plane_geometry_check.json`. Paired generators reproduce
the fixed-plane integrated volumes/areas to approximately 1e-13; a 2 mm
nonuniform boundary-height perturbation has a volume-rate difference-check
error of 3.3e-9. This is not a passed contact implementation. The existing
disk-moment formulas give unreliable centroids on some almost-zero ghost
faces (one face of area 5.0e-16 m² reports a centroid 7.1e-5 m below the
floor). The probe therefore reports `passed=false`, despite the accurate
integrated quantities. No such ghost face is being advanced or used to
compute contact forces through that old probe. The later precise-disk path and
compatible boundary-coordinate law above address those specific deficiencies;
the failed probe record is retained rather than relabelled as passing.
