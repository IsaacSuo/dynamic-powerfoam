# Local cut-hole boundary experiment (default OFF)

Implemented `alpha_boundary.local_boundary`: seed one removed dual triangle,
maintain the hole's successor ring, attach compatible removed triangles by
local edge cancellation, and emit boundary edges in the original order.
Unsupported attachments fall back to the exhaustive reference search.
The separate experimental kernel is `alpha_local_cut.k_alpha`; the original
`alpha_complex.py` is unchanged. Enable only by setting
`GeoPipe.local_boundary=True`; default is False. No production material,
precision, timestep, geometric predicate or clipping order changed.

Two full real states, at 0.2104425 and 0.2204425 seconds, were compared.
Twelve alternating pairs of full geometry calls per state:

| State | Reference median | Experimental median | Faster pairs |
|---|---:|---:|---:|
| dt8us | 20.212 ms | 18.708 ms | 11/12 |
| cfl20_long | 20.423 ms | 18.765 ms | 12/12 |

Geometry connectivity matched; relative L2 field differences were below
2.7e-16 (floating accumulation order). Full production-equivalent one-step
replay, including dynamic grouping, used eight alternating pairs per state:

| State | Reference median | Experimental median | Faster pairs |
|---|---:|---:|---:|
| dt8us | 57.385 ms | 54.127 ms | 7/8 |
| cfl20_long | 66.852 ms | 69.271 ms | 2/8 |

Complete-step speed gain is not established across both states. Do not
promote based on the isolated geometry gain. Active wall lists/counts match
exactly in these replays; maximum position difference <6e-14 m, velocity
<1.4e-12 m/s, pressure <4e-8 Pa. No full landing or broader degeneracy claim.

Reproduce full-step comparison with
`PYTHONPATH=. python -m free_surface.fv_ale_boundary_check`.
Reports: `output/ale/recut_local_boundary.json`,
`output/ale/recut_local_boundary_step.json`. Original k1 remains unchanged
according to the 49-file baseline hash check.

The direct spherical-area formula has not been implemented in this trial.
