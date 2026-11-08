"""Serial long near-incompressible and original-entry-point verification."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--wait-pid',type=int,default=0)
    args=parser.parse_args()
    while args.wait_pid:
        try:os.kill(args.wait_pid,0)
        except ProcessLookupError:break
        time.sleep(5)
    root=Path('output/ale')
    audit=json.loads((root/'hyperelastic_fast_audit_check.json').read_text())
    precision=json.loads((root/'hyperelastic_precise_geometry_cuda_check.json').read_text())
    if not audit['passed']:raise RuntimeError('cached audit was not independently verified')
    if not precision['success']:raise RuntimeError('GPU thin-cell geometry check did not pass')
    # The precision checker raises before writing if any assertion fails.
    common=['--case','release','--mechanical-reconstruction','compatible',
        '--affine-stencil','smooth','--affine-cap-motion','balanced',
        '--transport-reconstruction','entropy','--transport-dissipation','.1',
        '--interface-dissipation','.1','--cap-dissipation','0',
        '--energy-ledger','--fast-energy-audit','--work-retry-fraction','1e-6',
        '--precision-geometry','--steps','60000','--frames','--frame-interval','.002',
        '--record-interval','.001']
    jobs=[
        ('hyperelastic_main_entry_recheck','fv_ale_hyperelastic_main_check',[]),
        ('hyperelastic_compatible_mesh_gpu_audit','fv_ale_compatible_mesh_audit',[
            'hyperelastic_smooth_balanced001_guarded50ms','--device','cuda']),
        ('hyperelastic_interior_mesh12_guarded10ms','fv_ale_hyperelastic_check',common+[
            '--until','.01','--merge-dt','1e-5','--interface-dissipation','.01',
            '--mesh-fit-iterations','12','--output','hyperelastic_interior_mesh12_guarded10ms.json']),
        ('hyperelastic_interior_mesh24r_guarded10ms','fv_ale_hyperelastic_check',common+[
            '--until','.01','--merge-dt','1e-5','--interface-dissipation','.01',
            '--mesh-fit-iterations','24','--mesh-fit-radius','--output','hyperelastic_interior_mesh24r_guarded10ms.json']),
        ('mpm_release_nu049_50ms','fv_ale_mpm_compare',[
            '--dx','.005','--dt','1e-6','--until','.05','--poisson','.49','--tag','mpm_release_nu049_50ms']),
        ('hyperelastic_nu049_guarded50ms','fv_ale_hyperelastic_check',common+[
            '--poisson','.49','--until','.05','--merge-dt','2e-6','--output','hyperelastic_nu049_guarded50ms.json']),
        ('mpm_release_stretch2_10ms','fv_ale_mpm_compare',[
            '--dx','.005','--dt','1e-6','--until','.01','--stretch','2','--tag','mpm_release_stretch2_10ms']),
        ('hyperelastic_main_full50ms','merged_drop',[
            '--foam','foams/fourier8k_clean','--scene','0','--save','output/ale/hyperelastic_main_full50ms',
            '--tag','release','--fv-ale','--fv-ground-cut','--fv-constitutive','neo_hookean',
            '--fv-hyperelastic-config','configs/hyperelastic_free_release.json',
            '--fv-agglomerate','--fv-agglomerate-dt','1e-5','--fv-cfl','.5',
            '--frames','2','--fv-until','.05','--fv-steps','30000','--fv-budget','3600']),
    ]
    summary=[]
    for tag,module,extra in jobs:
        command=[sys.executable,'-m','free_surface.'+module,*extra]
        print(json.dumps({'start':tag,'command':command}),flush=True)
        tick=time.perf_counter()
        with (root/(tag+'_batch.log')).open('w') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        row={'tag':tag,'returncode':result.returncode,'seconds':time.perf_counter()-tick}
        path=root/(tag+'.json')
        if path.exists():
            data=json.loads(path.read_text())
            row.update({k:data.get(k) for k in ['completed','status','time_reached','steps','diagnostic_stop','final']})
        summary.append(row)
        (root/'hyperelastic_long_material_batch.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps(row),flush=True)
        if module in ['fv_ale_hyperelastic_main_check','fv_ale_compatible_mesh_audit'] and result.returncode:
            raise RuntimeError('main entry or resume regression failed; inspect before long runs')


if __name__=='__main__':main()
