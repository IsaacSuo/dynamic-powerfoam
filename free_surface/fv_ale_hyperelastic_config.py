"""Explicit, saved configuration for the opt-in full-scene hyperelastic path.

The configuration is also stored in snapshots. Resuming with different
mechanics or initialization is rejected instead of silently changing a run.
It does not change the old FV or k1 defaults.
"""
import json
import math
from pathlib import Path


def archive_run(args,config):
    """Capture source files, explicit configuration and input fingerprints."""
    import datetime
    import hashlib
    import zipfile
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    path=args.save/('fv_sources_'+Path(args.tag).name+'_'+stamp+'.zip')
    hashes={}
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for source in sorted(Path(__file__).parent.glob('*.py')):
            data=source.read_bytes();name='free_surface/'+source.name
            hashes[name]=hashlib.sha256(data).hexdigest();archive.writestr(name,data)
        archive.writestr('hyperelastic_config.json',json.dumps(config,indent=2))
    inputs={}
    for relative in ['dem_exact_single/geometry.npz','dem_exact_single/cells.npz','plasticine_scenes.json']:
        source=args.foam/relative
        inputs[str(source.resolve())]=hashlib.sha256(source.read_bytes()).hexdigest()
    return {'source_archive':str(path.resolve()),'source_sha256':hashes,'initial_input_sha256':inputs}


SOLVER_KEYS = {
    'interface_dissipation', 'cap_dissipation', 'transport_reconstruction',
    'transport_dissipation', 'mechanical_reconstruction', 'affine_stencil',
    'affine_cap_motion', 'mesh_fit_iterations', 'mesh_fit_regularization',
    'mesh_fit_radius', 'ground', 'gravity',
    'acoustic_bound',
    'precision_geometry',
    'wall_mode',
    'boundary_mode',
    'cap_motion_regularization',
}


def load_config(path):
    if path is None:
        return None
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict) or data.get('schema') != 'fv_hyperelastic_run_v1':
        raise ValueError('unrecognized hyperelastic run configuration')
    unknown = set(data) - {'schema', 'solver', 'initial_stretch', 'initial_velocity', 'work_retry_fraction', 'local_energy_tolerance', 'material','step_control'}
    options = data.get('solver', {})
    if not isinstance(options, dict):
        raise ValueError('hyperelastic solver configuration must be an object')
    if unknown or set(options) - SOLVER_KEYS:
        raise ValueError('unknown hyperelastic configuration fields: ' + str(unknown or set(options) - SOLVER_KEYS))
    stretch = float(data.get('initial_stretch', 1.))
    fraction = float(data.get('work_retry_fraction', 0.))
    local_tolerance = float(data.get('local_energy_tolerance', 0.))
    if not math.isfinite(local_tolerance) or local_tolerance < 0 or (local_tolerance > 0 and fraction <= 0):
        raise ValueError('local energy tolerance must be nonnegative and requires work retry')
    if not math.isfinite(stretch) or stretch <= 0 or not math.isfinite(fraction) or fraction < 0:
        raise ValueError('initial stretch must be positive and work retry fraction nonnegative')
    for key, vector in [('gravity', options.get('gravity')), ('initial_velocity', data.get('initial_velocity'))]:
        if vector is not None and (not isinstance(vector, list) or len(vector) != 3 or not all(math.isfinite(float(v)) for v in vector)):
            raise ValueError(key + ' must contain three finite numbers')
    if 'ground' in options and not math.isfinite(float(options['ground'])):
        raise ValueError('ground height must be finite')
    if options.get('boundary_mode','fixed') not in ['fixed','releasing_planes']:
        raise ValueError('unknown hyperelastic boundary mode')
    # Normalize optional top-level fields for deterministic resume comparison.
    normalized = {'schema': data['schema'], 'solver': options, 'initial_stretch': stretch,
            'initial_velocity': data.get('initial_velocity'), 'work_retry_fraction': fraction}
    # Preserve old configuration dictionaries exactly when the new guard is
    # absent, so existing snapshots still resume under their recorded policy.
    if local_tolerance > 0:
        normalized['local_energy_tolerance'] = local_tolerance
    material=data.get('material',{})
    if not isinstance(material,dict) or set(material)-{'E','nu'}:
        raise ValueError('hyperelastic material overrides support only E and nu')
    material={key:float(value) for key,value in material.items()}
    if any(not math.isfinite(value) for value in material.values()) or material.get('E',1.)<=0 or not -1<material.get('nu',0.)<.5:
        raise ValueError('hyperelastic material requires E > 0 and -1 < nu < 0.5')
    if material:normalized['material']=material
    if 'step_control' in data:
        if data['step_control'] not in ['original','remember']:
            raise ValueError('step_control must be original or remember')
        normalized['step_control']=data['step_control']
    return normalized


def initial_work_reference(solver,state):
    """A saved physical energy scale, including release from rest in gravity.

    Stored mechanical/total energy can be zero before a drop. The available
    vertical gravitational potential above the fixed floor is then a physical
    scale for the error tolerance. It is used ONLY for that tolerance; neither
    energy nor material state is changed. Free-release tests retain their
    existing initial-energy scale exactly.
    """
    total=float(state.q[:,4].sum());potential=0.
    if float(solver.g[1])<0:
        from free_surface.fv_ale_reconstruction import group_centroids
        mesh=state.mesh
        b={'p':state.p,'r':state.r,'V':mesh.volume,'groups':solver.groups,
           'i':mesh.i,'j':mesh.j,'A':mesh.area,'n':mesh.normal,'c':mesh.centroid,
           'N':mesh.cap_normal_integral,'floor':mesh.ground_area,
           'floor_c':mesh.ground_centroid,'floor_n':mesh.ground_normal}
        if hasattr(mesh,'planes'):
            from free_surface.fv_ale_planar_boundary import values as plane_values
            b.update(plane_values(mesh))
        height=(group_centroids(b)[:,1]-solver.ground).clamp(min=0)
        potential=float((-solver.g[1]*state.q[:,0]*height).sum())
    return {'scale_joule':max(total,potential),'initial_total_energy_joule':total,
            'initial_available_vertical_gravity_joule':potential,
            'definition':'max(initial total energy, initial downward-gravity potential above the fixed floor)'}


class StepAudit:
    """Same work-defect criterion as the full-foam diagnostic driver.

    Only a trial is rejected. Accepted q, geometry, total energy and material
    state are never repaired. The tolerance is saved in physical joules so
    a resumed run cannot accidentally change its error scale.
    """
    def __init__(self, solver, tolerance, local_tolerance=0.):
        from free_surface.fv_ale_energy_ledger import rate
        self.solver = solver
        self.tolerance = float(tolerance)
        self.local_tolerance = float(local_tolerance)
        self.active = False
        self.stage = None
        self.maximum_work_defect = 0.
        self.maximum_negative_remainder = 0.
        self.maximum_negative_specific = 0.
        self.maximum_negative_relative = 0.
        self.cumulative_work_defect = 0.
        self.checked_rate = False
        self.checked_guard = False
        self.crosscheck = {}
        self.rate = rate
        original = solver.rhs

        def measured(state, result):
            if all(key in result[3] for key in ['mechanical_rate_chain_W','mechanical_energy_joule']):
                return result[3]
            return rate(solver,state,result)
        self.measured=measured

        def wrapped(state):
            result = original(state)
            if self.active:
                self.stage = measured(state,result)
            return result

        solver.rhs = wrapped

    def begin(self, state, first):
        self.first = self.measured(state,first)
        if not self.checked_rate:
            reference=self.rate(self.solver,state,first)
            for key in ['mechanical_rate_chain_W','mechanical_energy_joule']:
                error=abs(self.first[key]-reference[key])/max(abs(reference[key]),1.)
                if error>1e-10:
                    raise AssertionError('GPU work diagnostic disagrees with independent account: '+key)
                self.crosscheck[key+'_relative_error']=error
            self.checked_rate=True
        self.active = True

    def check(self, candidate, dt):
        expected = getattr(candidate, 'integrated_mechanical_work', None)
        if expected is None:
            expected = .5 * dt * (self.first['mechanical_rate_chain_W'] + self.stage['mechanical_rate_chain_W'])
        mechanical,negative,specific,relative=self.solver.device_ops().energy_guard_full(candidate)
        if not self.checked_guard:
            reference=self.solver.energy_audit(candidate)
            reference_energy=reference['elastic_energy_joule']+reference['kinetic_energy_joule']
            error=abs(mechanical-reference_energy)/max(abs(reference_energy),1.)
            negative_error=abs(negative-reference['negative_remainder_deficit_joule'])
            if error>1e-12 or negative_error>1e-12*max(abs(mechanical),1.):
                raise AssertionError('GPU energy guard disagrees with the full independent audit')
            self.crosscheck.update(guard_energy_relative_error=error,guard_negative_absolute_error_joule=negative_error)
            self.checked_guard=True
        actual = mechanical - self.first['mechanical_energy_joule']
        defect = actual - expected
        if abs(defect) > self.tolerance:
            raise RuntimeError('mechanical work defect exceeds saved tolerance: %.9g J / %.9g J' % (defect, self.tolerance))
        if negative > self.tolerance:
            raise RuntimeError('local mechanical energy deficit exceeds saved tolerance: %.9g J / %.9g J' % (negative, self.tolerance))
        if self.local_tolerance > 0 and relative > self.local_tolerance:
            raise RuntimeError('per-group energy deficit / (elastic + mass*mu/rho0) exceeds saved tolerance: %.9g / %.9g' % (relative, self.local_tolerance))
        self.maximum_work_defect = max(self.maximum_work_defect, abs(defect))
        self.maximum_negative_remainder = max(self.maximum_negative_remainder, negative)
        self.maximum_negative_specific = max(self.maximum_negative_specific, specific)
        self.maximum_negative_relative = max(self.maximum_negative_relative, relative)
        self.cumulative_work_defect += defect
        self.active = False
        return {'mechanical_work_defect_joule': defect, 'local_negative_remainder_joule':negative}

    def report(self):
        return {'tolerance_joule': self.tolerance,
                'local_energy_tolerance': self.local_tolerance,
                'scope': 'every accepted physical step in this invocation',
                'max_abs_work_error_joule': self.maximum_work_defect,
                'max_local_negative_remainder_joule': self.maximum_negative_remainder,
                'max_negative_specific_remainder_joule_per_kg':self.maximum_negative_specific,
                'max_negative_remainder_over_local_elastic_plus_mu_over_rho0':self.maximum_negative_relative,
                'cumulative_work_error_joule': self.cumulative_work_defect,
                'initial_independent_crosscheck':self.crosscheck}
