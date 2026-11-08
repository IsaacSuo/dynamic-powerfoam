# 8 microsecond agglomeration target: equal-duration production comparison

Both runs use the full FV production driver with the optimized GPU numerical
segments, real ground-clipped power-ball geometry, fresh recuts and CFL 0.2.
Only `--fv-agglomerate-dt` differs (1e-6 versus 8e-6). No default material,
integrator, CFL, or original k1 source was changed for this test.

Start: `output/ale/fv_state_device80.pt`, physical time 0.20844252114006612 s.
End: 0.21044252114006612 s (2 ms of physical motion, NOT full landing).
Added `--fv-until` to stop at an exact common physical time, independently
of output frames; the final physical step is shortened to hit this time.

Command (run sequentially, replacing TAG and TARGET):

```sh
PYTHONPATH=. /home/fangsuo/anaconda3/envs/om/bin/python -m free_surface.merged_drop \
  --foam foams/fourier8k_clean --scene 0 --save output/ale --tag TAG --frames 6 \
  --fv-ale --fv-ground-cut --fv-surface-motion coupled --fv-agglomerate \
  --fv-agglomerate-dt TARGET --fv-resume output/ale/fv_state_device80.pt \
  --fv-until 0.21044252114006612 --fv-steps 10000 --fv-budget 180
```

| Run | Target | Steps | Median actual dt | Wall time | Rejected attempts |
|---|---:|---:|---:|---:|---:|
| dt1us_control | 1 us | 1241 | 1.330 us | 57.272 s | 0 |
| dt8us | 8 us | 250 | 8.034 us | 12.856 s | 0 |

Measured speedup: 4.45x, one timing sample per configuration, including
startup, graph capture and merges. The 8 us run performed 93 merges,
touching 189 geometric cells including anchors (1.92% of initial mass).
The control performed 13 merges touching 26 cells (0.37% of initial mass).
Both retain all geometric generators.

Equal-time differences:

- Maximum particle-center distance: 1.328 mm; initial-mass-weighted RMS: 0.00754 mm.
- Maximum radius difference: 0.166 mm.
- Mass-weighted velocity difference: 0.00657 m/s, relative L2 0.265%; maximum 0.647 m/s.
- Ground contact area difference: 0.00234%.
- Mass and source-accounted total-energy residuals remain at roundoff scale.

Local changes are appreciable despite small global metrics. Grouping and dt
both change, so these differences do not isolate temporal truncation error.
This short test supports a speed gain and successful advancement at this
state, not full-landing robustness, constitutive correctness or convergence.

Full numbers: `output/ale/dt8us_comparison.json`; per-step histories and
snapshots have corresponding `dt8us` / `dt1us_control` tags.
