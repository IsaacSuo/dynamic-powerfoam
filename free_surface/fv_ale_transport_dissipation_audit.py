"""Frozen full-state transport-rate audit; does not establish stability."""
import json
from pathlib import Path
import torch
import warp as wp
from free_surface.fv_ale_compatible_mesh_audit import restored
from free_surface.fv_ale_hyperelastic import HyperelasticDeviceOps
from free_surface.fv_ale_energy_ledger import rate


def main():
    torch.set_num_threads(1);wp.init();root=Path('output/ale')
    data=torch.load(root/'hyperelastic_smooth_balanced001_guarded50ms.pt',map_location='cpu',weights_only=False)
    sol,state,_,b=restored(data);rows=[]
    for alpha in [.1,.03,.01,0.]:
        sol.transport_dissipation=alpha;ops=HyperelasticDeviceOps(sol);sol._device_ops=ops
        with torch.no_grad():
            w,dr,dq,packed=ops.rhs_gpu(b);diag=dict(zip(ops.rhs_keys,packed.tolist()))
            ledger=rate(sol,state,(w,dr,dq,diag),fd=True)
        rows.append({'transport_dissipation':alpha,'dt_scale':diag['dt_scale'],'ledger':ledger})
    result={'rows':rows,'scope':'Same 50-ms full state; instantaneous rates only, not trajectory stability or accuracy.'}
    (root/'hyperelastic_transport_dissipation_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
