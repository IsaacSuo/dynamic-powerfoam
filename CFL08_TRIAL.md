# CFL 0.8 short production trial

Same initial state and 2 ms interval as the local pair in `CFL05_TRIAL.md`.
Runtime CFL 0.8, dynamic merge target 32 us; target/CFL remains 40 us,
preserving the wave-timescale merge criterion. No flux, material, group-size
limit, timestep rejection check or source default was changed.

```sh
PYTHONPATH=. /home/fangsuo/anaconda3/envs/om/bin/python -m free_surface.merged_drop \
  --foam foams/fourier8k_clean --scene 0 --save output/ale \
  --tag cfl08_local_dynamic --frames 6 --fv-ale --fv-ground-cut \
  --fv-surface-motion coupled --fv-cfl 0.8 --fv-agglomerate \
  --fv-agglomerate-dt 32e-6 --fv-resume output/ale/fv_state_dt8us.pt \
  --fv-until 0.21244252114006612 --fv-steps 10000 --fv-budget 90
```

Completed 63 steps, no rejection, 4.152 s wall time. Median dt 32.107 us;
maximum 32.134 us; final step shortened to hit the shared end time.
23 merges, final 7538 control volumes (same counts as the CFL 0.5 run).
Mass/source-energy residuals are roundoff scale. Stage densities span
963.128 to 1089.934 kg/m3.

Prior equal-duration runs: CFL 0.2 took 250 steps / 12.351 s; CFL 0.5
took 100 steps / 5.917 s. This single timing sample is 1.425x faster than
0.5 and 2.974x faster than 0.2, including startup and grouping.

Against 0.2 at the same end time: maximum center distance 0.02898 mm,
initial-mass-weighted RMS 0.000872 mm; velocity RMS difference 0.002416 m/s,
maximum 0.3183 m/s; ground area difference -0.001313%.
Against 0.5: velocity RMS difference 0.00005999 m/s, maximum 0.001241 m/s.

This is a short trajectory check, not a stability-limit or full-landing proof.
Dynamic grouping may differ in timing with the trajectory; no claim of pure
temporal error isolation. Detailed report: `output/ale/cfl08_comparison.json`.
