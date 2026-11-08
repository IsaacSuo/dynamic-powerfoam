"""Serial full-foam checks; no concurrent GPU timing or model changes.

This is an experiment runner, not a solver. Each job records its own exact
source archive and arguments through the existing full-program check driver.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--wait-pid',type=int,default=0)
    parser.add_argument('--suite',choices=['extended','cap-volume','guarded','long','wave'],default='extended')
    args=parser.parse_args()
    if args.wait_pid:
        while True:
            try:os.kill(args.wait_pid,0)
            except ProcessLookupError:break
            time.sleep(5)
    root=Path('output/ale')
    common=['--mechanical-reconstruction','compatible','--affine-stencil','extended',
            '--transport-reconstruction','entropy','--transport-dissipation','.1',
            '--interface-dissipation','.1','--cap-dissipation','0','--energy-ledger',
            '--work-snapshot-threshold','.01','--record-interval','.001',
            '--steps','7000','--merge-dt','1e-5']
    jobs=[
        ('hyperelastic_extended_rotation_5ms',['--case','rotation','--until','.005']),
        ('hyperelastic_extended_nu049_5ms',['--case','release','--poisson','.49','--until','.005','--merge-dt','2e-6']),
        ('hyperelastic_extended_stretch2_10ms',['--case','release','--stretch','2','--until','.01']),
        # Half the physical step and merge target together to retain the same
        # acoustic geometric grouping threshold, rather than coarsening it.
        ('hyperelastic_extended_halfdt_10ms',['--case','release','--until','.01','--cfl','.25','--merge-dt','5e-6']),
        ('hyperelastic_extended_alpha001_10ms',['--case','release','--until','.01','--interface-dissipation','.01']),
    ]
    if args.suite=='cap-volume':
        common+=['--affine-cap-motion','volume']
        jobs=[
            ('hyperelastic_capvolume01_10ms',['--case','release','--until','.01']),
            ('hyperelastic_capvolume001_10ms',['--case','release','--until','.01','--interface-dissipation','.01']),
            ('hyperelastic_capvolume_nu049_5ms',['--case','release','--poisson','.49','--until','.005','--merge-dt','2e-6']),
        ]
    elif args.suite=='guarded':
        common+=['--affine-stencil','smooth','--affine-cap-motion','balanced','--work-retry-fraction','1e-6']
        jobs=[
            ('hyperelastic_smooth_balanced001_guarded10ms',['--case','release','--until','.01','--interface-dissipation','.01']),
            ('hyperelastic_smooth_balanced_stretch2_guarded10ms',['--case','release','--stretch','2','--until','.01']),
            ('hyperelastic_smooth_balanced_split_keep_guarded5ms',['--case','release','--until','.005','--foam','foams/plasticine0_split_keep_clean']),
            ('hyperelastic_smooth_balanced_nu049_guarded5ms',['--case','release','--poisson','.49','--until','.005','--merge-dt','2e-6']),
        ]
    elif args.suite=='long':
        common+=['--affine-stencil','smooth','--affine-cap-motion','balanced','--work-retry-fraction','1e-6',
                 '--steps','30000','--frames','--frame-interval','.002']
        jobs=[
            ('hyperelastic_smooth_stretch2_cfl_fallback10ms',['--case','release','--stretch','2','--until','.01']),
            ('hyperelastic_smooth_balanced01_guarded50ms',['--case','release','--until','.05']),
            ('hyperelastic_smooth_balanced001_guarded50ms',['--case','release','--until','.05','--interface-dissipation','.01']),
        ]
    elif args.suite=='wave':
        common+=['--affine-stencil','smooth','--affine-cap-motion','balanced','--work-retry-fraction','1e-6',
                 '--acoustic-bound','tight','--steps','20000','--fast-energy-audit']
        jobs=[
            ('hyperelastic_tight_wave_gpu_check',['--case','gpu']),
            ('hyperelastic_old_default_final_regression',['--case','regression']),
            ('hyperelastic_smooth_stretch2_precise10ms',['--case','release','--stretch','2','--until','.01',
                '--acoustic-bound','conservative','--precision-geometry']),
            ('hyperelastic_tight_wave01_guarded10ms',['--case','release','--until','.01']),
            ('hyperelastic_tight_wave01_halfdt10ms',['--case','release','--until','.01','--cfl','.25','--merge-dt','5e-6']),
        ]
    summary=[]
    if args.suite=='wave':
        with (root/'hyperelastic_fast_audit_check.log').open('w') as log:
            result=subprocess.run([sys.executable,'-m','free_surface.fv_ale_fast_audit_check'],
                                  stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError('independent/cached energy audit comparison failed; see its log')
        with (root/'hyperelastic_precise_geometry_cuda_check.log').open('w') as log:
            result=subprocess.run([sys.executable,'-m','free_surface.fv_ale_precise_geometry_check','--device','cuda'],
                                  stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError('full saved-geometry GPU precision check failed; see its log')
    for tag,extra in jobs:
        command=[sys.executable,'-m','free_surface.fv_ale_hyperelastic_check',*common,*extra,'--output',tag+'.json']
        print(json.dumps({'start':tag,'command':command}),flush=True)
        tick=time.perf_counter()
        with (root/(tag+'.log')).open('w') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        row={'tag':tag,'returncode':result.returncode,'wall_seconds':time.perf_counter()-tick}
        if (root/(tag+'.json')).exists():
            data=json.loads((root/(tag+'.json')).read_text())
            row.update({k:data.get(k) for k in ['completed','time_reached','steps','final','step_audit','diagnostic_stop','stencil_statistics']})
            row.update(min_accepted_dt=data.get('min_accepted_dt'),max_accepted_dt=data.get('max_accepted_dt'),
                       retries=len(data.get('rejected_attempts',[])),work_retry_tolerance_joule=data.get('work_retry_tolerance_joule'))
        summary.append(row)
        (root/('hyperelastic_'+args.suite.replace('-','_')+'_validation_batch.json')).write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps(row),flush=True)


if __name__=='__main__':main()
