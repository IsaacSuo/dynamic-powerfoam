# CFL multiplier 5 production trials

User requested testing 5 and stopping further diagnosis of multiplier 10.
No further multiplier-10 analysis was performed. The already completed
multiplier-2 long run is reused as reference.

Start for both durations: `output/ale/fv_state_dt8us.pt` at
0.21044252114006612 s. Runtime `--fv-cfl 5 --fv-agglomerate-dt 200e-6`
retains target/CFL = 40 us, equivalent to the reference 2 / 80 us.
All other settings match `CFL20_TRIAL.md`; physics, checks and defaults
remain unchanged.

| Duration | Tag | Steps | Rejected attempts | Median accepted dt | Wall time |
|---|---|---:|---:|---:|---:|
| 2 ms | cfl20_local_dynamic | 25 | 0 | 80.21 us | 2.198 s |
| 2 ms | cfl50_local_dynamic | 12 | 3 | 200.61 us | 1.804 s |
| 10 ms | cfl20_long | 125 | 0 | 80.21 us | 8.494 s |
| 10 ms | cfl50_long | 67 | 33 | 100.75 us | 6.673 s |

Both multiplier-5 runs completed through retries. In 2 ms all 3 rejections
were unsupported tensile/release predictions at the planar ground. In 10 ms,
32 rejections were that condition, and one was loss of whole-group volume or
mass. The implementation halves dt on rejection; completion does not mean
that sustained 200 us stepping succeeded. These failures do not by themselves
identify a linear acoustic instability: contact release is also unimplemented.

Against multiplier 2 at 10 ms: maximum particle-center distance 4.859 mm,
initial-mass-weighted RMS 0.05123 mm; velocity RMS difference 0.01112 m/s,
maximum 0.6873 m/s; ground area relative difference -0.001213%.
At 2 ms maximum center distance was 0.1673 mm. Local differences thus matter
despite small global contact-area differences. Dynamic merge histories differ
slightly (200 vs 201 merges in 10 ms), so differences are not pure time error.

Mass and source-accounted energy residuals remain at roundoff scale. This
does not prove accuracy or stability. No full landing run. Single timing
sample per configuration, including startup and merges.

Run tags above identify logs, histories, frames and final snapshots. End times
are 0.21244252114006612 and 0.22044252114006612, set by `--fv-until`.
Other command flags: `--foam foams/fourier8k_clean --scene 0 --save output/ale
--frames 6 --fv-ale --fv-ground-cut --fv-surface-motion coupled
--fv-agglomerate --fv-steps 10000 --fv-budget 90`.
Detailed results: `output/ale/cfl50_comparison.json`.
