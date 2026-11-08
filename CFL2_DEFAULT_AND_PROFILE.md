# Provisional default multiplier 2 and current performance

## Current decision (2026-11-08)

The user chose to continue with explicit FV and stop pursuing the implicit
experiments for now. Verified that the FV driver still calls the production
two-stage explicit `SolidALE.advance`, with default CFL multiplier 2 and the
original sphere-area formula. The swept-geometry, boundary-block and implicit
trial scripts remain diagnostic records, not production updates. No rollback
was needed. Preserve the explicit configuration described below; actual dt
remains adaptive, not a guaranteed fixed 80 microseconds.

User selected 2 as the provisional FV default. Updated `--fv-cfl` in
`merged_drop.py` and the default `SolidALE.advance(..., cfl=2.)` consistently.
Explicit values in older diagnostic scripts are unchanged for reproducibility.
Original k1 working files remain unchanged (49 baseline hashes checked).

To retain the tested local grouping scale, use `--fv-agglomerate
--fv-agglomerate-dt 80e-6` with this default. The merge-target argument's
default remains 0 (volume-only criterion); a caller explicitly passing the
old 8e-6 target will not reproduce the tested 2 / 80e-6 configuration.

Driver smoke run without `--fv-cfl`: `default2_smoke`, resumed from
`fv_state_dt8us.pt`, accepted one 80.335 us step and then stopped at the
requested one-step limit. No rejection; status budget_stopped_not_complete
is intentional, not a physics failure. Compile and diff checks passed.

Fresh warm full-step replays at multiplier 2, with production-equivalent
dynamic merge selection and a reused initial RHS. Six ordinary timings and
four synchronized nested timings per real saved state; reset geometry is
outside timing, runtime merges inside. Detailed file:
`output/ale/cfl2_current_profile.json`.

| Snapshot time | Full-step median | Two geometry calls combined | RHS total | Merges |
|---|---:|---:|---:|---:|
| 0.2104425 s | 50.38 ms | 37.94 ms | 4.97 ms | 1 |
| 0.2204425 s | 63.62 ms | 39.51 ms | 8.29 ms | 3 |

Synchronized full-step medians 50.56 and 62.94 ms respectively. Geometry
accounts for roughly 62–75% of wall time at these points, including the
ground geometry path. These are sample-state profiles, not a full-run average.

Next optimization candidate: exact geometry work (topology/structure versus
geometric value kernels) and sequential grouping overhead. Existing overlap
candidate caching, incremental candidate sorting, local ground updates and
GPU numerical replay already exist; do not propose them as new optimizations.
Reusing unchanged connectivity with validation, while updating current areas
and volumes, remains an unimplemented candidate requiring equivalence tests.
No timestep lagging, additional coarsening or material softening introduced.
