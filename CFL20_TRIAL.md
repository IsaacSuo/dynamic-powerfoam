# CFL multiplier 2 short production trial

Same start (`output/ale/fv_state_dt8us.pt`) and end time
0.21244252114006612 s as the CFL 0.2/0.5/0.8 local-grouping tests.
Runtime `--fv-cfl 2 --fv-agglomerate-dt 80e-6` preserves the wave-timescale
merge trigger target/CFL = 40 us. All other settings match `CFL08_TRIAL.md`;
tag is `cfl20_local_dynamic`. No source defaults, physics, rejection checks
or group-size limits were changed.

Completed 2 ms in 25 accepted steps, 2.198 s wall time; no rejected attempts.
Median dt 80.207 us, maximum 80.335 us; final dt shortened to 74.645 us.
23 dynamic merges; 7538 final control volumes, same count as CFL 0.5/0.8.
Mass residual 0; source-accounted energy residual 7.95e-16.
Stage density range 963.210 to 1086.271 kg/m3.

Compared to the CFL 0.2 end state: maximum center distance 0.07613 mm,
initial-mass-weighted RMS 0.001806 mm; velocity RMS 0.002288 m/s,
maximum 0.30064 m/s; ground area relative difference -0.003418%.
Compared to CFL 0.8: velocity RMS 0.0002419 m/s, maximum 0.02246 m/s.

Single measured wall-time ratio: 1.889x faster than CFL 0.8, 5.618x faster
than CFL 0.2, including initialization and merges. No repeated timing study.
Detailed output: `output/ale/cfl20_comparison.json` and per-step/run files
with the `cfl20_local_dynamic` tag.

No full-landing stability claim. A short unperturbed trajectory cannot
establish the stable multiplier or rule out slowly growing instabilities.
Dynamic grouping histories may differ; this is not pure time error isolation.
