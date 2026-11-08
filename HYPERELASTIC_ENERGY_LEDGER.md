# 超弹性空间耗散诊断（2026-11-08，进行中）

本记录是完整显式程序的诊断与实验，不代表超弹性分支已验收。旧 Wilkins 和原 k1 默认保持不变。所有耗散系数实验仍为非默认。

## 基线能量账

同一 7690 格子、50% 预拉伸、无重力无接触释放 10 ms，初始弹性能 726.8378 J。独立按机械能对守恒状态和切分体积求导，得到瞬时机械能变化，再与实际 RHS 的中心差分核对。恒等式相对误差约 1e-15，独立差分约 1e-9。

完整 RK2 轨迹积分结果（`output/ale/hyperelastic_release_ledger.json`）：

| 来源 | 机械能变化 / J |
|---|---:|
| 内部面的声学通量 | −268.1775 |
| 自由球面的声学运动 | −138.0728 |
| 一阶迎风状态输运混合 | −16.2156 |
| 合并时的直接状态混合 | −0.1560 |
| 时间推进与实际几何更新余差 | +0.0221 |

合计损失约 422.60 J，即 58.14%。这是可恢复机械能进入不反馈压力的能量余项，不是材料塑性。合并的**直接**损失很小，不排除空间粗化间接改变通量损失。

内部面以 `ti=sigma_i n`、`tj=sigma_j n`、`Zi=rho_i c_i` 记，原声学解的机械能损失率是

`-A (|tj-ti|² + Zi Zj |vj-vi|²)/(Zi+Zj)`。

自由球面损失率是 `-Z ∫cap |(w-v)+dr n|² dA`。因此不能靠减小时间步恢复这里丢掉的机械能。

## 受控的低耗散实验

内部面只给声学速度跳跃和牵引跳跃项乘 alpha；自由球面的应力驱动速度乘 beta。材料 E、nu、物理波速和稳定步长估算不改。两个系数默认都是 1。

alpha 小于 1 后，不能照搬原来的能量通量 `t*·u*`：它会使一个面的某一侧出现负耗散。实验分支改用

`Q* = [(vi+vj)·t* + (ti+tj)·u* - ti·vi - tj·vj]/2`。

两侧各获得非负面耗散的一半；局部随机测试同时验证守恒、机械功恒等式和伽利略变换，误差约 1e-15。alpha=1 仍走原始通量，保护基线。系数并非拟合材料参数，尚未选定生产值。

| alpha=beta | 10 ms 机械能损失 | 内部声学 / 输运混合 / 自由面损失 J |
|---|---:|---|
| 1 | 58.14% | 268.18 / 16.22 / 138.07 |
| 0.1 | 28.68% | 121.55 / 65.66 / 20.36 |
| 0.01 | 21.33% | 20.48 / 130.15 / 2.16 |

低耗散后运动更强，输运混合成了主要损失。因此不能把继续减系数当作完整修法。0.01 组最终 J 范围约 0.491–1.609；需要继续检查空间精度、振荡及长时间稳定性。两组末端没有负的能量余项。

## 空间一致性检查

`fv_ale_hyperelastic_flux_checks.py` 对真实格子的体积质心赋线性速度场，直接检查完整 RHS。当前分片常数通量不是线性精确的：角速度约 1 rad/s 的纯旋转输入，Hdot 的体积加权 RMS 误差约 0.81/s；单位简单剪切的误差约 0.59/s。这不是材料公式的客观性失败（材料检查约 1e-15），而是完整空间离散的误差。单个不规则网格上的瞬时误差尚不能代替空间收敛试验。

成熟的 cell-centered 固体有限体积方法通常用分片线性重构改善一阶通量的过度耗散和精度。参考：

- Burton 等，2013，固体 CCH 二阶重构及兼容耗散：https://doi.org/10.1016/j.compfluid.2012.09.008
- Kluth 等，2020，三维更新拉格朗日超弹性 EUCCLHYD 二阶扩展：https://doi.org/10.1016/j.compfluid.2020.104523

这些文献并不直接提供本程序“幂图与球相交”的自由边界实现，不能把当前代码称为它们的完整复现。

## 后续完整程序实验

### 线性输运及其能量通量

`--transport-reconstruction linear` 只重构跨格输运的密度、速度和 H；**不是完整二阶固体算法**，应力通量和边界阶数尚未提升。用真实体积质心、面积/距离平方加权最小二乘和邻居极值限幅，缺秩时退回常量。无约束线性场梯度误差 4.6e-14，面值极值检查通过。

直接重构机械状态再套用原能量通量，会让少数格子的能量余项为负。因此实验采用独立推导的兼容输运通量。令机械能对守恒量的导数为 chi，满足 `chi.U-eta=p`：

- 若高阶面值使 `relative_flux * [(chi_i-chi_j).U_face - (p_i-p_j)]` 为负，把它与原迎风面值凸混合到非负边界。
- 能量面值取 `[(chi_i+chi_j).U_face - p_i-p_j]/2 + rho_face*thermal_upwind`。

这样把非负的机械混合耗散平分给两侧，并另行迎风搬运已有能量余项。没有步末重设能量。局部随机检查验证两侧产热、守恒及伽利略变换到约 1e-15。这是本程序的适配推导，不能冒称文献原公式。

10 ms、alpha=beta=0.01 的完整结果：损失 140.15 J（19.28%），耗时 120.5 s；对照常量输运损失 155.02 J（21.33%）、83.3 s。历史记录最大负能量缺额仅 3.35e-9 J。收益有限、成本增加，**没有设为默认**。

### 内部网格运动拟合

`--mesh-fit-iterations` 默认 0。启用时只调整没有自由球面和地面面的内部生成点速度；可选同时调整内部半径速度。当前自由表面/地面运动、材料面速度、牵引和形变源项保持原值。面积加权最小二乘降低面法向网格速度与材料速度之差，并加正的对角正则项；GPU 固定次数 PCG。若目标函数不下降，保留原运动。

ALE 允许独立选择网格速度，以减小相对输运。参考 [Springel 2010, §3.2](https://doi.org/10.1111/j.1365-2966.2009.15715.x)。这里的具体拟合是实验适配，不是 AREPO 的完整网格策略。

冻结 10 ms 状态，24 次迭代并允许内部半径调整：面法向滑移 RMS 从 0.749 降至 0.645 m/s，输运损失率从 10.88 kW 降至 7.29 kW；RHS 耗时约 4.5→9.0 ms。边界速度差只有独立重算的约 1e-15 舍入误差，独立机械能恒等式仍通过。完整轨迹的收益小于这一冻结点的比例，且时间/几何余差增大，仍是非默认试验。

### 50 ms 同材料 MPM 对照

`hyperelastic_release_blend01_50ms.json` 与 `mpm_release_dx005_dt2p5us_50ms.json`；后者实际步长是 2.5e-6 s，即 2.5 微秒。

- FV alpha=beta=0.1，4926 步、255.7 s，完整到 50 ms；机械能损失 75.74%。
- MPM dx=5 mm、2.38 百万粒子，20000 步、143.9 s，机械能损失 41.98%。两者空间分辨率与精度不同，耗时不是公平性能排名。
- 整体收缩/展开的时间曲线接近；末端质量 RMS 尺寸相对 MPM 的差为 X −6.80%、Y +0.76%、Z +0.87%。整个区间最大 RMS 尺寸绝对差为 4.88、1.75、2.00 mm。
- FV 质心漂移约 0.14 mm，总动量仍守恒到舍入量级；几何质量矩与有限体积动量并非完全相同的离散量。
- 图：`output/ale/hyperelastic_50ms_energy_motion.png`、`hyperelastic_50ms_shapes.png`。后者为同尺度正交投影，FV 画真实球并集，MPM 画抽样材料点。

**长程耗散仍偏大，不能因形状接近就宣称已解决。**

### 近不可压缩材料

同一实际 foam，nu=0.49，50% 预拉伸，alpha=beta=0.1，释放 5 ms：

- FV 2208 步、106.9 s，损失 15.35%，J 为 0.933–1.051；完整完成。
- 同材料 MPM dx=5 mm、dt=1 us，5000 步、30.7 s，损失 10.82%，J 为 0.879–1.148。
- RMS 尺寸 FV 为 0.096190/0.103250/0.103538 m；MPM 为 0.096169/0.103820/0.104050 m。

MPM 使用 FP32，nu 接近 1/2 会放大体积模量的参数舍入。实际 K 与请求 K 差约 9.8e-7；应力检查现分别核对偏应力和压力对应的行列式，未修改 MPM 推进或上游源码。该结果不证明无体积锁死或空间收敛。

额外 CPU 材料检查覆盖 J=0.0397–9.43、最大主伸长 12.34，nu=0/0.3/0.45/0.49/0.499；功导数、对数源项、声学正性及上界通过。有限刚体转动的 RK2 误差随步数二阶下降。这仍然只是本构验证，不能代替上述完整程序结果。
# Compatible affine reconstruction experiments (2026-11-08)

## Continuous stencil transition and balanced cap motion (in progress)

Implementation follow-up: the smooth map is now stored as a compact directed
edge list with stable zero-padded capacity for CUDA graphs, rather than an
N-by-maximum-neighborhood dense table. The dense table remains an independent
reference. On all three saved stages of the twice-prestretched case, CPU
gradient and transpose differences are around 2e-16; on the actual GPU force,
the difference is below 2e-16 and the independent power derivative remains
within 6e-15. This changes storage/reduction order, not the mathematical map;
it does not promise bitwise-identical long nonlinear trajectories.

The latest diagnostic runner also offers `--work-retry-fraction`, default off.
With 1e-6 it rejects and halves only the offending step if either its work
defect or summed local negative energy remainder exceeds one millionth of the
initial total energy. It retries from the unchanged accepted state; it neither
resets energy nor modifies deformation. Every actual accepted dt, rejection
reason, minimum/maximum dt and total cost are reported. This is an explicit
energy-consistency guard, not a complete solution-error estimator or a proof
of spatial accuracy. It is being tested rather than silently enabled in the
production driver.

The first extended-stencil rule fixed near-coplanar fits, but introduced a
discontinuous activation at normal-matrix ratio 1e-3. The twice-prestretched
full run captured a work-error event at 5.768 ms. Group 4161 crossed from
ratio 0.000897 to 0.001289 inside RK2; its stencil shrank from 25 supports to
5 and its gradient norm jumped from 99 to 1449 /s. The saved step's 0.018838 J
work error is mostly the mismatch between endpoint geometric volume and the
two-stage volume-rate integral (0.017656 J). This is not evidence of an
incorrect analytic cutting formula or of material failure.

`--affine-stencil smooth` instead grows a fixed three-ring topological stencil
for candidate roots and multiplies the added weights by a smoothstep of the
original ratio, from zero at 1e-2 to one at 1e-3. The primal and transpose use
the same map. It retains the monitored zero-slope fallback if a stencil still
cannot be made usable. The transition constants are an implementation choice,
not a proven CFL bound. On the identical captured step, unchanged dt gives
-4.37e-5 J error; halving dt gives -5.43e-6, -6.79e-7, -8.47e-8 J. No negative
energy remainder occurs in these replays. Independent force/power checks on
this actually activated geometry pass. Full runs remain necessary.

The cap-volume-only rule restores the affine patch test but can produce more
thin cells by assigning the entire correction to radius motion. An additional
mesh-motion choice, `--affine-cap-motion balanced`, distributes the scalar
flux defect over site and radius velocities with the minimum Euclidean change:
with `m=N/A` and defect per area `d`, add `d*m/(1+|m|^2)` to site velocity and
`d/(1+|m|^2)` to radius velocity. These controls have the same velocity units;
their combined cap flux is exactly `A*d`. Rigid rotation and isotropic dilation
are unchanged. This is a bounded geometric-control choice, not a different
material or a claim of optimal mesh quality. The same volume-compatible force
transpose applies. Full-gradient patch, power derivative and coordinate
covariance checks pass; long-run cost/dissipation is being measured.

A separate low-acoustic cap-volume run stopped at 2.376 ms with 0.01494 J work
error. Its stencils were well conditioned, and independent recuts agree to
about 1e-17 m3; 0.01435 J comes from endpoint volume integration. Balanced cap
motion reduces this same-state error to 0.00883 J, but does not eliminate it.
The full and half steps give 0.00883 and 0.00150 J respectively, so this must
not be presented as another complete cure. The recorded state is retained.

## Affine cap-volume consistency (new opt-in check)

The original affine cap motion used `r_dot = r tr(G)/3`. That is correct for
isotropic dilation or the average over a whole sphere, but not for an exposed
clipped spherical cap. With `N=int n dA`, `Q=int n tensor n dA` and group-center
offset `o=p-center`, an affine field has cap flux

`(v+G o).N + r G:Q`.

Keeping site velocity `v+G o`, its integrated volume is represented exactly by
`r_dot = r (G:Q)/cap_area`. It still does not represent every pointwise affine
normal velocity with a sphere; the claim is the integrated flux and the full
cell affine-gradient patch test. Interior spheres retain the old isotropic
radius rule. The pressure contribution in the adjoint force changes with this
map: its radius term uses `r Q`, instead of `r cap_area I/3`. Changing only the
radius rate would break the force/power identity.

On the actual initial foam, a unit trace-free stretch gave an old divergence
error of 0.3264 /s (volume-weighted RMS), maximum 5.087 /s; pure shear gave
0.2202 /s, maximum 5.604 /s. Both should have zero divergence. The integrated
cap-volume map reduces these to about 1e-13 /s RMS and below 1e-11 /s maximum.
Production full-gradient patch checks, independent scalar-power derivatives,
global force/torque checks and rotated-coordinate checks pass. This is exposed
as `--affine-cap-motion volume`, only with the compatible force. It is not yet
the default; full trajectories are needed before claiming lower dissipation
or improved stability. See `hyperelastic_affine_volume_check.json`.

## Ill-conditioned reconstruction: captured full-step failure and correction

The one-ring entropy-transport release completed 50 ms, but must not be called
validated: near 32 ms it generated a 5.7 J time-discretization/work discrepancy
and a local negative nonelastic remainder. A deterministic diagnostic capture
stopped before submitting the first step with work error above 0.01 J:
`hyperelastic_entropy_work_capture_work_error_step3065.pt` contains both RK
stages, exact cut geometry, conserved states, RHS values and configuration.

At 31.675349 ms, with dt=10.01016 microseconds, its work error is 0.161076 J.
Only 0.000162 J comes from endpoint geometric-volume integration; the RK2
material/kinetic chain contributes 0.160914 J. Independent recuts, a different
sphere-area formula and quadrature of the exact volume derivative agree.
The singular behavior is not a sphere-cutting error.

Group 1421 has only three almost coplanar neighbors: its weighted least-squares
normal matrix has min/max eigenvalue ratio 3.78e-7. The old acceptance cutoff
1e-10 admits it. Its reconstructed force gives acceleration components around
2.25e6, 5.28e6 and -5.39e6 m/s^2. A volume/area acoustic CFL does not cover
this extra ill-conditioned reconstruction stiffness.

The opt-in `--affine-stencil extended` grows only ill-conditioned stencils
through actual shared material faces, up to three rings, until the ratio is
at least 1e-3. Extra supports use the maximum bottleneck face area along their
paths. The velocity reconstruction and negative-adjoint force use exactly the
same directed map. Unresolved lower-dimensional stencils use zero slope and
are explicitly counted; they are not silently inverted. This is a new
implementation choice, not a theorem that this ratio guarantees stability.
The broader mesh-quality limitation of least-squares reconstruction is
documented by Mavriplis, NASA/CR-2003-212683:
https://ntrs.nasa.gov/citations/20040070704 . That report does not validate our
particular three-dimensional threshold, weighting or force discretization.

On the identical captured state only one root requires extension (22 extra
directed supports, no zero-slope fallback). At unchanged dt the work error
falls to -1.38e-7 J and the negative energy remainder disappears. Halving dt
then produces -1.91e-8, -2.47e-9, -3.09e-10 J, consistent with the expected
local RK2 order. Independent scalar-power differentiation still reproduces
the analytic force; net internal force, torque and energy chain checks pass.
See `fv_ale_hyperelastic_work_replay.py --stencil extended` and the saved
`*_extended_analysis.json`. The new full 50 ms run remains a separate check;
the same-state result alone is not long-run validation.

## Mechanical-entropy transport experiment

`--transport-reconstruction entropy` is an additional opt-in experiment. Its
face state satisfies `[chi] dot U_face = [p]`, with chi the derivative of
mechanical energy density with respect to density, momentum density and
mass-weighted log shape. This is an application of the algebraic framework
in Tadmor's original entropy-conservative construction, not a claim that an
Euler-gas flux can be reused for this solid:
https://www.ams.org/mcom/1987-49-179/S0025-5718-1987-0890255-3/S0025-5718-1987-0890255-3.pdf

For the specific volumetric law used here, the density mean is

`rho_hat = 2 rho0 (1 + 1/(Ji Jj)) / (Ji + Jj + 2/logmean(Ji,Jj))`.

It is positive and lies between the two input densities. Velocity uses the
arithmetic mean. With `T = d e_shape/dH`, the shape mean corrects the arithmetic
mean along `[T]` to satisfy
`[T]:H_hat = [T:H - e_shape]`. Near equal shapes the cubic correction is below
roundoff, and the arithmetic mean is used rather than dividing cancellation
noise by a tiny stress difference. The reported identity tolerance includes
this branch. A configurable fraction of the original upwind state supplies
mechanical dissipation; this fraction does not change material stiffness.

Random noncommuting shape states, equal/nearly equal states, density bounds,
rotations, boosts and state exchange are tested independently. The actual-foam
power ledger is checked separately. Neither those tests nor the semidiscrete
identity establish fully discrete stability; full release runs are required.

Convexity of the shape energy additionally bounds the shape correction:
`|H_hat - (Hi+Hj)/2| <= |Hi-Hj|/2` in Frobenius norm. It follows from the
two tangent inequalities for a convex function; it does not imply a
componentwise maximum principle. This bound and preservation of zero trace
are checked on the noncommuting test pairs as well.


These are opt-in production-FV experiments, not a replacement mesh or a change
to the neo-Hookean material parameters. The old Wilkins default stays on its
original mechanical path.

## Affine reconstruction alone

Reconstructing face velocity and spherical-cap gradient moments eliminated a
large rigid-spin inconsistency. On the actual 7,690-cell foam, the instantaneous
rotation/shear log-shape-rate error fell from approximately 0.8/0.6 per second
to 8e-14. In a 5 ms free-spin run, false elastic energy fell from 0.00558 J to
1.53e-6 J (same MPM comparison: 2.58e-6 J); mechanical loss fell from 2.3% to
0.002%. This is a free body initially in rotation, not externally enforced
rigid rotation.

This alone did not cure release dissipation. The affine release with zero
explicit cap acoustic coefficient completed 10 ms but lost approximately
212.8 J of 726.8 J: 105.7 J internal, 93.2 J cap work limiting, 13.7 J transport,
0.20 J group mixing. The cap loss remained because the local work limiter
suppressed reconstructed kinematic work. Force and the reconstructed gradient
were not an adjoint pair. Local nonpositive-work limiting cannot replace that
compatibility.

The vector/tensor reconstruction limiter now uses a common Euclidean/Frobenius
norm factor, rather than coordinate-dependent component limits. A rotated,
translated and boosted actual state gives face-traction discrepancies around
1e-14. Scalar density still uses neighbor-extrema limiting.

## Negative-adjoint force experiment

`--mechanical-reconstruction compatible` uses the same full program and an
affine-exact group least-squares velocity map. The elastic force is the negative
transpose of the resulting integrated elastic-power operator. Consequently the
force has a wider stencil than a two-cell shared-face traction. A quadratic
penalty on reconstructed velocity jumps supplies dissipative force through
the same transpose. Original stress-jump acoustic diffusion remains optional.
The cap affine motion/gradient is no longer clipped by a local work limiter.

The total-energy RHS uses the differential mechanical-energy chain rule plus
the explicitly derived nonnegative acoustic and transport heating. It does
not overwrite energy or deformation after a step. Global energy conservation,
zero net internal force, and zero internal torque are independent checks.
This version currently rejects ground-contact states: no unsupported wall
formula is silently inherited. It is not yet the default or a completed
contact/rebound method.

On a nonuniform prestrained full-foam state, an independently differentiated
scalar power matches the analytic force; the energy-chain finite differences
and the split work ledger also agree. See `hyperelastic_compatible_checks.json`
for actual tolerances and results. Long-run accuracy/stability still has to be
measured; these identities alone do not establish convergence.

## Geometric roundoff guard update

One affine run stopped on an already grouped child volume -4.86e-18 m3;
an independent complete-arc area formula gave +4.89e-17 m3 on the identical
cut. The hyperelastic-only guard changed from 64 to 256 machine epsilons times
the ball volume, with a new additional limit: the total correction in any
group must not exceed 1e-10 of its positive volume. This only zeros tiny
negative grouped child volumes. Singleton/larger negatives are rejected;
mass and energy are unchanged. This is a numerical guard, not a rigorous
error bound for the cutting algorithm.

## Latest guarded 50 ms release

`hyperelastic_smooth_balanced01_guarded50ms.json` completed 5,145 steps in
757.6 s including full per-step audits. Mechanical-energy loss is 52.2459%,
versus 41.9842% for the matched PowerSim MPM run. End mass-RMS extents differ
by +0.84%, -1.27%, -0.70%. The discrete work defect sums to +0.07797 J of
726.84 J initially. Group mixing, acoustic flux, and material transport lose
27.15, 280.58, and 72.09 J respectively. This makes the remaining long-time
loss attributable; it does not establish that this loss is acceptable.
Every accepted step checks the summed local negative mechanical-energy
remainder, with a one-part-per-million initial-energy tolerance. No energy
or shape is reset to satisfy the check.

## Positive high-precision reevaluation of an exceptional thin wedge

The two-times-prestretch run stopped at 2.23435 ms on grouped cell 929.
On the saved rejected geometry, the original double-precision spherical
formula gives -3.0278e-18 m3. An independent nearest-point/KKT construction
proves a positive two-plane spherical wedge. Its positive cross-section
integral at 60 decimal digits gives 1.1191004048e-19 m3. One million targeted
samples agree within 0.80 standard errors. Thus this failure is cancellation
in geometry, not material inversion. The separate direct-area formula was
also inaccurate in this case and is not adopted globally.

`--precision-geometry` optionally reevaluates only certified, already grouped,
non-ground two-plane wedges. All other planes must be inactive on an enclosing
cap, the correction must be at most 4096 eps times the ball volume and at
most 1e-10 of the group volume. Shared faces are retained, and cap area/normal
are reconstructed consistently from divergence/closure. q, mass, energy,
positions and radii are untouched. Unsupported cases still reject. The full
7,690-cell CPU and GPU geometry checks pass. The two-times-prestretch run
then completed 10 ms, 1,340 steps, 272.6 s including diagnostics. There were
48 exceptional reevaluations (including trial cuts), no merge-target skips,
and no energy/state reset. Mechanical loss is 32.78%; a matched strong-strain
MPM comparison is still pending. This is not a general cure for geometric
degeneracy.

## Optional tighter wave estimate and cached audits

`--acoustic-bound tight` maximizes the same neo-Hookean acoustic tensor over
all directions analytically using its extreme metric eigenvalues. Tests cover
4,096 states, including near incompressibility and large stretch. The bound
agrees with attaining directions to 2.92e-14 and exceeds all sampled directions.
It remains optional: a continuum wave bound does not prove the stability limit
of the wide-stencil spatial operator, and changing impedance also changes
numerical dissipation. Dynamic and half-step comparisons are pending.

The compatible GPU RHS already evaluates the mechanical-energy chain rule.
`--fast-energy-audit` reuses these rates and a GPU energy guard, with an
independent full rate/energy check every 500 evaluations. The original audit
remains the default. A same-initial-state full-program comparison is required
before using the cached path for longer tests; no physical formula is changed.

That comparison now passes: 237 independent rate checks have maximum relative
error 2.31e-14; 113 independent energy-guard checks differ by at most 2.27e-13 J.
An initially strict trajectory comparison failed at 1.62e-7 relative state
change, so a same-code repeat was added rather than assuming a cause. The
repeat spread was 2.45e-7, versus 7.26e-8 between audit methods; all three
runs took 109 accepted steps, 19 retries and identical final grouping. These
are floating-point repeatability bounds, not bitwise reproducibility.

## Interior-only mesh-motion investigation

Persistent grouping leaves 2,394 control volumes after the 0.01-coefficient
50 ms run. An acoustic-only screen finds 298 groups whose 693 members would
individually exceed twice the merge target, but relative material transport
has not been included in that screen. No group splitting is implemented.

The already available interior mesh-motion fit was audited with the compatible
force on that full saved state. It changes neither exposed boundary motion,
material face velocities nor the elastic force/gradient map. The power ledger
and finite-difference energy derivative remain consistent. Twelve velocity-only
iterations reduce instantaneous transport loss from 2258 to 1659 W; allowing
interior radius motion with 24 iterations gives 1428 W. These are frozen-state
rates, not full-run savings. GPU graph checks also passed, but both complete
trajectory attempts stopped near 0.4 ms because the local negative-energy
remainder reached the unchanged one-part-per-million threshold. Velocity-only
12-iteration motion stopped at 0.41196 ms; 24 iterations including radius
motion stopped at 0.39894 ms. Halving the attempted step did not allow the
remaining retry budget to pass. This combination is blocked again in the run
constructor pending diagnosis. Frozen power identities alone were insufficient.

## Local energy deficit hidden by the global tolerance

The nu=0.49 50 ms run completed, but its per-step diagnostic reached a local
negative remainder of 50.23 J/kg, or 16.54% of `(elastic + mu/rho0)`, while
the sum over all negative cells remained below the global 1 ppm tolerance.
This fails a local accuracy assessment despite the close MPM/global match.

A complete first-defect RK2 snapshot was captured in
`hyperelastic_nu049_local_capture_local_energy_step231.pt` at 0.572951 ms,
dt=2.030649 us. Group 4041 is a single cell of mass 0.137 g. Its remainder
was -2.965e-6 J before the step. The semidiscrete heat/transport account
predicts +6.061e-5 J after the step; the actual remainder is -7.875e-5 J.
The exact difference separates into:

* Actual re-cut endpoint volume versus RK-integrated volume: +1.05516e-4 J
  of mechanical energy, for a volume difference of 6.64405e-10 m³.
* RK2 nonlinear energy chain-rule defect: +3.38458e-5 J.
* Post-update material-state projection: zero.

`fv_ale_local_energy_replay.py` independently closes this per-group identity.
The repeat cut differs by 1.36e-20 m³; the alternative sphere formula differs
by 6.78e-18 m³. Integrating the exact geometric volume derivative along the
endpoint segment closes all group volumes to 5.31e-17 m³ (10-point rule).
Thus this captured event is a finite time/geometry update error, not a
disagreement between the two sphere integration formulas.

At the identical saved state, a half-size step gives zero negative remainder
and a global work defect of -9.26e-7 J, versus +1.92509e-4 J for the original
step. Quarter and eighth steps also give zero deficit. Very small fractions
retain part of the already-negative starting remainder, so their absolute
deficit must not be mistaken for an error generated entirely by that step.

An optional `local_energy_tolerance` now rejects candidates using the maximum
over groups of `max(-R,0)/(elastic + mass*mu/rho0)`. It complements, rather
than replaces, the absolute total-work and summed-deficit checks. The proposed
trial setting is 1e-6. It only retries from the unchanged accepted state; no
mass, energy or material state is repaired. This accuracy criterion is not a
proof of an entropy-positive fully discrete scheme. New full-run validation
is in progress; old results retain their original acceptance policy.

