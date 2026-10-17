"""Serial, resumable execution of the fixed independent-FEM experiment.

Continue reference refinement, then train only after the reference gate passes.
No simultaneous simulation jobs; each child must finish before the next starts.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-reference', action='store_true')
    parser.add_argument('--reference', type=Path, help='Train from an existing completed export; do not generate or refine FEM data.')
    parser.add_argument('--exploratory-reference', action='store_true')
    parser.add_argument('--out', type=Path, default=Path('output/foam_fem/inverse_v1'))
    parser.add_argument('--max-iterations', type=int, default=400)
    args = parser.parse_args()
    root = Path('output/foam_fem')
    status_file = root/'pipeline_status.json'
    environment = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    environment['PYTHONPATH'] = str(Path('output/fem_dependencies').resolve())
    started = time.time()

    def status(stage, **kwargs):
        data = dict(stage=stage, started_unix=started, updated_unix=time.time(), **kwargs)
        temporary = status_file.with_suffix('.tmp')
        temporary.write_text(json.dumps(data, indent=2)+'\n')
        temporary.replace(status_file)
        print(json.dumps(data), flush=True)

    def run(stage, module, arguments, logfile):
        status(stage, command=[sys.executable, '-m', module, *arguments], log=str(logfile))
        with logfile.open('a') as stream:
            subprocess.run([sys.executable, '-m', module, *arguments], env=environment,
                           stdout=stream, stderr=subprocess.STDOUT, check=True)

    try:
        if args.reference is not None:
            arguments = ['--reference', str(args.reference), '--out', str(args.out),
                         '--max-iterations', str(args.max_iterations)]
            if args.exploratory_reference:
                arguments.append('--exploratory-reference')
            if args.out.exists():
                arguments.append('--resume')
            run('inverse_initialization_and_training', 'experiments.foam_fem_inverse',
                arguments, args.out.with_suffix('.log'))
            status('complete', reference=str(args.reference), inverse_report=str(args.out/'report.json'))
            return
        previous = root/'reference_edge010'
        if args.wait_reference and not (previous/'report.json').exists():
            status('waiting_existing_reference', reference=str(previous),
                   log='/tmp/powerfoam_fem_reference_edge010.log')
            deadline = time.time()+7200
            while not (previous/'report.json').exists():
                if time.time() > deadline:
                    raise RuntimeError('Timed out waiting for the already running reference job.')
                logfile = Path('/tmp/powerfoam_fem_reference_edge010.log')
                if logfile.exists() and 'Traceback (most recent call last)' in logfile.read_text():
                    raise RuntimeError('The existing reference job failed; see its log.')
                time.sleep(5)
        if not (previous/'report.json').exists():
            raise RuntimeError('Finish reference_edge010 before starting this serial continuation.')
        checked = json.loads((previous/'report.json').read_text())
        for name, bulk, edge in [('edge007', .30, .07), ('edge005', .25, .05)]:
            if checked['ready_for_training']:
                break
            geometry = root/f'mesh_{name}'
            if not (geometry/'geometry.json').exists():
                run('reference_meshing', 'experiments.foam_fem_geometry',
                    ['--out', str(geometry), '--mesh-size', str(bulk), '--edge-size', str(edge)],
                    root/f'{name}_geometry.log')
            geometry_info = json.loads((geometry/'geometry.json').read_text())
            if geometry_info['tetrahedra'] > 60000:
                status('stopped_mesh_size_limit', geometry=str(geometry),
                    reason='More than 60000 P2 tetrahedra; review memory before assembling.', training_started=False)
                return
            current = root/f'reference_{name}'
            if not (current/'report.json').exists():
                run('reference_motion', 'experiments.foam_fem_reference',
                    ['--geometry', str(geometry/'geometry.npz'), '--out', str(current),
                     '--config', str(root/'reference_q050_v3/config.json'),
                     '--compare', str(previous), '--min-substeps', '16'], root/f'{name}_reference.log')
            checked = json.loads((current/'report.json').read_text())
            previous = current
        if not checked['ready_for_training']:
            status('stopped_reference_not_converged', reference=str(previous),
                errors=checked['mesh_refinement_errors'], training_started=False)
            return
        output = root/'inverse_v1'
        arguments = ['--reference', str(previous), '--out', str(output)]
        if output.exists(): arguments.append('--resume')
        run('parameter_learning', 'experiments.foam_fem_inverse', arguments, root/'inverse_v1.log')
        status('complete', reference=str(previous), inverse_report=str(output/'report.json'))
    except Exception as error:
        status('failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
