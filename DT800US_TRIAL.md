# 800 microsecond target with existing agglomeration

Same command and start snapshot as `DT8US_TRIAL.md`, with tag `dt800us`
and `--fv-agglomerate-dt 8e-4`. CFL remains 0.2, group-size limit 32,
maximum physical dt 1/1200 s. No production model or safety check changed.

The run stopped during initial grouping, BEFORE accepting any physical step.
It performed 6800 merges in about 40 seconds, reducing 7654 current control
volumes to 854; all 7690 geometric generators remain. Grouping touched 7586
geometric cells, representing 93.63% of the initial mass. Largest group has
32 members; largest recorded merged-site diameter bound is 0.105 m.

At the stop, the estimated allowed physical step is 57.84 microseconds,
still 13.8 times below the requested 800 microseconds. The limiting group
(cell 219) has 15 members; adjacent groups have 20, 28, 30, 20, 29, 29.
Every adjacent merge would exceed the existing 32-member limit.

This demonstrates failure of the CURRENT grouping policy to reach the
target, not instability of an actually attempted 800 us step, and not a
mathematical upper bound over all possible partitions. Removing the group
cap or forcibly bypassing CFL was not tested. The failed trial's heavily
coarsened state was saved separately and is not the accepted 8 us trajectory.

Reports: `output/ale/fv_solid_dt800us.json`, `dt800us_diagnosis.json`;
log: `output/ale/dt800us.log`. Original small-step and 8 us snapshots retained.
