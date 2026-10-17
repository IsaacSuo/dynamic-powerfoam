"""Cell-centred finite-volume baseline: OpenFOAM solidDisplacementFoam on the cube specimens.

Same specimen, clamp, material, loads, pulse and frames as the FEM targets
(output/foam_fem/cube_{spec}_reference_q018). The clamped first cell layer is left out
(fixedValue D = 0 on x = 0.5); the free part x in [0.5, 3.5] is meshed with hexahedra of
size 1/r (r odd, so a cell centre sits on every DEM cell centre). r = 1 is exactly the
DEM / Power Foam cell layout.

The conda OpenFOAM build only has constant tractions, so each load runs once as a unit
step load and the pulse response follows by discrete superposition (the scheme is linear
and time-invariant): D_n = sum_k (u_k - u_{k-1}) S_{n-k+1}. Tractions per boundary face
are the face average of the FEM traction field restricted to the load region (partial
faces included), checked against the FEM total force and torque.
"""
import argparse
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from experiments.foam_fem_reference import REGION, traction_field

F = Path('output/foam_fem')
OUT = F/'openfoam_v1'
ENV = Path.home()/'anaconda3/envs/foam'
CASES = ('train_down', 'train_pull', 'train_twist', 'test_side', 'test_local', 'fresh_oblique',
         'fresh_local_side', 'fresh_end_bend_y', 'fresh_end_bend_z', 'fresh_mid_top_down', 'fresh_mid_side')
TRAIN = CASES[:3]
OLD = ('test_side', 'test_local', 'test_strong_down')
FRESH = CASES[5:]
HEADER = 'FoamFile\n{{\n    version 2.0;\n    format ascii;\n    class {cls};\n    object {obj};\n}}\n'


def foam(path, cls, obj, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(HEADER.format(cls=cls, obj=obj)+body)


def run(command, case):
    result = subprocess.run(['bash', '-c', f'source {Path.home()}/anaconda3/bin/activate foam && {command}'],
                            cwd=case, capture_output=True, text=True)
    (case/f'log.{command.split()[0]}').write_text(result.stdout+result.stderr)
    if result.returncode:
        raise RuntimeError(f'{command} failed in {case}:\n{result.stdout[-2000:]}{result.stderr[-2000:]}')


def numbers(text):
    return np.array([float(v) for v in re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', text)])


def foam_list(path):
    """Body of an ASCII OpenFOAM list file (after the header)."""
    text = path.read_text()
    text = text[text.index('}', text.index('FoamFile'))+1:]
    return text[text.index('(')+1:text.rindex(')')]


def read_mesh(case):
    mesh = case/'constant/polyMesh'
    points = numbers(foam_list(mesh/'points')).reshape(-1, 3)
    faces = [list(map(int, f.split())) for f in re.findall(r'\d+\(([\d ]+)\)', foam_list(mesh/'faces'))]
    boundary = (mesh/'boundary').read_text()
    patches = {}
    for name, body in re.findall(r'(\w+)\s*\{([^}]*)\}', boundary[boundary.index('}')+1:]):
        start = int(re.search(r'startFace\s+(\d+)', body).group(1))
        count = int(re.search(r'nFaces\s+(\d+)', body).group(1))
        patches[name] = [faces[i] for i in range(start, start+count)]
    return points, patches


def build_mesh(case, m, r):
    h = m/2
    foam(case/'system/blockMeshDict', 'dictionary', 'blockMeshDict', f'''
convertToMeters 1;
vertices ((0.5 {-h} {-h}) (3.5 {-h} {-h}) (3.5 {h} {-h}) (0.5 {h} {-h})
          (0.5 {-h} {h}) (3.5 {-h} {h}) (3.5 {h} {h}) (0.5 {h} {h}));
blocks (hex (0 1 2 3 4 5 6 7) ({3*r} {m*r} {m*r}) simpleGrading (1 1 1));
boundary
(
    clamp {{ type patch; faces ((0 4 7 3)); }}
    surface {{ type patch; faces ((1 2 6 5) (0 1 5 4) (3 7 6 2) (0 3 2 1) (4 5 6 7)); }}
);
''')
    controls(case, 1., 1., [])
    run('blockMesh', case)


def face_tractions(name, faces, points, m, samples=12):
    """Average traction of each quad face over its part inside the load region."""
    top, side = m/2-.5, -(m/2-.5)  # outermost centre rows, as in the FEM selection
    s = (np.arange(samples)+.5)/samples
    a, b = np.meshgrid(s, s, indexing='ij')
    a, b = a.ravel(), b.ravel()
    out, force, torque = np.zeros((len(faces), 3)), np.zeros(3), np.zeros(3)
    for k, f in enumerate(faces):
        p = points[f]
        xyz = ((1-a)*(1-b))[:, None]*p[0]+(a*(1-b))[:, None]*p[1]+(a*b)[:, None]*p[2]+((1-a)*b)[:, None]*p[3]
        region = REGION[name]
        if region == 'end':
            inside = xyz[:, 0] > 3.
        else:
            middle = (xyz[:, 0] > 1.5) & (xyz[:, 0] < 2.5)
            inside = middle & (xyz[:, 2] > top+1e-9 if region == 'mid_top' else xyz[:, 1] < side-1e-9)
        if inside.any():
            try:
                t = traction_field(name, xyz.T).T*inside[:, None]
            except ValueError:  # local loads: zero on this face (name checked by the caller)
                t = np.zeros_like(xyz)
        else:
            t = np.zeros_like(xyz)
        out[k] = t.mean(axis=0)
        area = np.linalg.norm(np.cross(p[2]-p[0], p[3]-p[1]))/2
        force += out[k]*area
        torque += np.cross(xyz, t).mean(axis=0)*area
    return out, force, torque


def controls(case, step, end, probes):
    locations = ' '.join(f'({x:.12g} {y:.12g} {z:.12g})' for x, y, z in probes)
    probe = f'''functions
{{
    probes {{ type probes; libs (sampling); writeControl timeStep; writeInterval 1;
              interpolationScheme cell; fields (D); probeLocations ({locations}); }}
}}''' if len(probes) else ''
    foam(case/'system/controlDict', 'dictionary', 'controlDict', f'''
application solidDisplacementFoam;
startFrom startTime; startTime 0; stopAt endTime; endTime {end:.17g};
deltaT {step:.17g}; writeControl timeStep; writeInterval 1000000000;
purgeWrite 0; writeFormat ascii; writePrecision 12; writeCompression off;
timeFormat general; timePrecision 12; runTimeModifiable false;
{probe}
''')


def write_case(case, mesh, name, amplitude, m, step, steps, probes, tolerance):
    shutil.copytree(mesh/'constant/polyMesh', case/'constant/polyMesh')
    points, patches = read_mesh(case)
    traction, force, torque = face_tractions(name, patches['surface'], points, m)
    traction *= amplitude
    values = '\n'.join(f'({x:.17g} {y:.17g} {z:.17g})' for x, y, z in traction)
    foam(case/'0/D', 'volVectorField', 'D', f'''
dimensions [0 1 0 0 0 0 0];
internalField uniform (0 0 0);
boundaryField
{{
    clamp {{ type fixedValue; value uniform (0 0 0); }}
    surface
    {{
        type tractionDisplacement;
        traction nonuniform List<vector> {len(traction)}
(
{values}
);
        pressure uniform 0;
        value uniform (0 0 0);
    }}
}}
''')
    foam(case/'constant/mechanicalProperties', 'dictionary', 'mechanicalProperties', '''
rho { type uniform; value 1; }
nu { type uniform; value 0.25; }
E { type uniform; value 1200; }
planeStress no;
''')
    foam(case/'constant/thermalProperties', 'dictionary', 'thermalProperties', 'thermalStress no;\n')
    foam(case/'system/fvSchemes', 'dictionary', 'fvSchemes', '''
d2dt2Schemes { default Euler; }
ddtSchemes { default Euler; }
gradSchemes { default leastSquares; }
divSchemes { default none; div(sigmaD) Gauss linear; }
laplacianSchemes { default none; laplacian(DD,D) Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default none; }
''')
    foam(case/'system/fvSolution', 'dictionary', 'fvSolution', f'''
solvers {{ "(D|T)" {{ solver PCG; preconditioner DIC; tolerance 1e-12; relTol 0; }} }}
stressAnalysis {{ compactNormalStress yes; nCorrectors 200; D {tolerance:.17g}; }}
''')
    controls(case, step, step*steps, probes)
    return force, torque


def read_probes(case, count):
    rows = []
    for line in (case/'postProcessing/probes/0/D').read_text().splitlines():
        if line.startswith('#') or not line.strip():
            continue
        rows.append(numbers(line))
    data = np.array(rows)
    return data[:, 0], data[:, 1:].reshape(len(data), count, 3)


def pulse_response(step_response, step, sub, frames, pulse_duration):
    """Superpose the unit-step response for the sin^2 pulse; values at frame times."""
    n = len(step_response)
    t = step*np.arange(n+1)
    u = np.where(t < pulse_duration, np.sin(np.pi*t/pulse_duration)**2, 0.)
    du = np.diff(u)  # du[k-1] = u_k - u_{k-1}, k = 1..n
    out = [np.zeros_like(step_response[0])]
    for f in range(1, frames+1):
        steps = f*sub
        out.append(np.tensordot(du[:steps], step_response[:steps][::-1], axes=1))
    return np.array(out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec', choices=('block', 'bar'), required=True)
    p.add_argument('--refine', type=int, default=1)
    p.add_argument('--sub', type=int, default=512, help='Time steps per output frame.')
    p.add_argument('--tolerance', type=float, default=1e-10, help='Corrector convergence of D.')
    p.add_argument('--cases', nargs='+', default=list(CASES))
    args = p.parse_args()
    assert args.refine % 2 == 1, 'odd refinement keeps a cell centre on every DEM centre'
    m = 4 if args.spec == 'block' else 2
    ref = F/f'cube_{args.spec}_reference_q018'
    config = json.loads((ref/'config.json').read_text())
    amplitudes = dict(zip(config['load_cases'], config['amplitudes']))
    checks = json.loads((ref/'report.json').read_text())['checks']['load_transfer']
    geometry = np.load(ref/'patch.npz')
    points, fixed = geometry['points'], geometry['fixed']
    free = np.flatnonzero(~fixed)
    root = OUT/f'{args.spec}_r{args.refine}_sub{args.sub}'
    mesh = root/'mesh'
    if not (mesh/'constant/polyMesh/faces').exists():
        build_mesh(mesh, m, args.refine)
    for name in args.cases:
        case = root/name
        if case.exists():
            shutil.rmtree(case)
        step = config['dt']/args.sub
        steps = config['frames']*args.sub
        force, torque = write_case(case, mesh, name, amplitudes[name], m, step, steps, points[free], args.tolerance)
        ref_force, ref_torque = np.array(checks[name]['force']), np.array(checks[name]['torque'])
        load_error = max(np.abs(force-ref_force).max(), np.abs(torque-ref_torque).max())/max(np.abs(ref_force).max(), np.abs(ref_torque).max())
        start = time.perf_counter()
        run('solidDisplacementFoam', case)
        seconds = time.perf_counter()-start
        times, S = read_probes(case, len(free))
        assert len(times) == steps and np.allclose(times, step*np.arange(1, steps+1), rtol=1e-6), (len(times), steps)
        D = pulse_response(S, step, args.sub, config['frames'], config['pulse_duration'])
        target = np.load(ref/f'{name}_target.npz')['positions'][:, free]
        prediction = points[free]+D
        scale2 = np.mean((target[1:]-points[free])**2)
        error = float(np.sqrt(np.mean((prediction[1:]-target[1:])**2)/scale2))
        result = dict(normalized_rmse=error, seconds=seconds, load_relative_mismatch=float(load_error))
        (case/'result.json').write_text(json.dumps(result, indent=1))
        np.savez_compressed(case/'positions.npz', positions=prediction)
        shutil.rmtree(case/'postProcessing')
        print(name, json.dumps(result), flush=True)
    # Aggregate every finished case of this mesh (runs may be split over processes).
    done = {c: json.loads((root/c/'result.json').read_text()) for c in CASES if (root/c/'result.json').exists()}
    rms = lambda keys: 100*float(np.sqrt(np.mean([done[k]['normalized_rmse']**2 for k in keys])))
    summary = dict(cells=3*m*m*args.refine**3, sub=args.sub, finished=len(done), per_case={k: 100*v['normalized_rmse'] for k, v in done.items()})
    if len(done) == len(CASES):
        summary.update(train=rms(TRAIN), old=rms(OLD[:2]+('train_down',)), fresh=rms(FRESH))
    (root/'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary), flush=True)

if __name__ == '__main__':
    main()
