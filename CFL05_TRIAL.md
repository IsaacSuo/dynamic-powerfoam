# CFL 0.5 production trials

Changed only runtime CFL via `--fv-cfl 0.5`; default remains 0.2 and no
material law, flux, timestep checks or group size limit was altered.
Both pairs use full production geometry and stepping and stop after 2 ms.

## Existing coarse groups, fixed

Start `output/ale/fv_state_dt800us.pt`, end `--fv-until 0.21044252114006612`.
No `--fv-agglomerate`: use its existing 854 control volumes without changes.
This input already coarsened about 94% of mass during the prior 800 us target
experiment; it is NOT a replacement for the fine accepted trajectory.

| CFL / tag | Initial allowed dt | Median dt | Steps | Wall time |
|---|---:|---:|---:|---:|
| 0.2 / cfl02_coarse | 57.84 us | 55.84 us | 36 | 2.743 s |
| 0.5 / cfl05_coarse | 144.59 us | 138.82 us | 15 | 1.802 s |

Both completed without rejection. Equal-time maximum center distance is
0.103 mm; mass-weighted RMS is 0.00535 mm. Velocity RMS difference is
0.000153 m/s. Contact area differs by 0.00228%. These compare time integration
on the SAME coarse groups, not spatial fidelity versus unmerged dynamics.

## Local grouping trajectory

Start `output/ale/fv_state_dt8us.pt`, end `--fv-until 0.21244252114006612`.
An initial fixed-group diagnostic (`cfl05_local`) reached 20 us initially
but its dt collapsed during geometry evolution; it was interrupted at
t=0.2119938163, dt=2.21e-9 s, 724 accepted steps. It did not complete.
Fixed grouping is insufficient as cells change size under recutting.

Retained existing dynamic merging for the successful pair. To preserve the
same wave-timescale trigger, use target/CFL = 40 us in both runs:

| CFL / tag | Merge target | Median dt | Steps | Wall time | Merges |
|---|---:|---:|---:|---:|---:|
| 0.2 / cfl02_local_dynamic | 8 us | 8.026 us | 250 | 12.351 s | 24 |
| 0.5 / cfl05_local_dynamic | 20 us | 20.066 us | 100 | 5.917 s | 23 |

Both completed without rejection. Maximum center distance 0.0205 mm,
mass-weighted RMS 0.000502 mm; velocity RMS difference 0.00241 m/s,
maximum 0.318 m/s; contact area differs by 0.000643%.
Group histories can differ slightly because dt changes trajectory and trigger
times. Mass and source-accounted total-energy residuals are roundoff scale.

All commands otherwise use `--foam foams/fourier8k_clean --scene 0
--save output/ale --frames 6 --fv-ale --fv-ground-cut
--fv-surface-motion coupled --fv-steps 10000 --fv-budget 90`.
Dynamic pair also uses `--fv-agglomerate --fv-agglomerate-dt TARGET`.
Full numbers: `output/ale/cfl05_comparison.json`. No full-landing or rigorous
stability-limit claim follows from these short tests. Timing is one sample
including startup, graph capture and (where enabled) merging.
