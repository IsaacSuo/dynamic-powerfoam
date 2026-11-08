"""Collect completed full-program evidence; never convert partial runs to passes."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--contact-tag', required=True)
    args = ap.parse_args()
    root = Path('output/ale')
    sources = {}

    def read(name):
        path = root / name
        raw = path.read_bytes()
        sources[name] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    materials = []
    for tag, end in [('hyperelastic_alpha001_localguard50ms', .05),
                     ('hyperelastic_nu049_localguard50ms', .05),
                     ('hyperelastic_stretch2_localguard10ms', .01)]:
        run = read(tag + '.json')
        comparison = read(tag + '_comparison.json')
        audit = run['step_audit']
        assert run['completed'] and abs(run['time_reached'] - end) < 1e-12
        assert not run['diagnostic_stop']
        assert run['local_energy_tolerance'] == 1e-6
        assert audit['max_negative_remainder_over_local_elastic_plus_mu_over_rho0'] <= 1e-6
        assert audit['max_abs_work_error_joule'] <= run['work_retry_tolerance_joule']
        assert comparison['mpm_status'] == 'completed'
        materials.append({'tag': tag, 'duration_s': end, 'steps': run['steps'],
                          'material': run['material'], 'comparison': comparison,
                          'audit': audit})

    contact = read(args.contact_tag + '.json')
    contact_summary = read(args.contact_tag + '_contact_summary.json')
    gap = read(args.contact_tag + '_gap_check.json')
    assert contact['completed'] and not contact['diagnostic_stop']
    assert contact_summary['five_ms_sustained_separation'] and gap['passed']
    assert gap['resolved_gap_lower_bound_m'] >= 1e-5
    assert contact['step_audit']['max_abs_work_error_joule'] <= contact['work_retry_tolerance_joule']
    assert contact['step_audit']['max_negative_remainder_over_local_elastic_plus_mu_over_rho0'] <= 1e-6
    assert abs(contact['mass_relative_error']) < 1e-12
    assert abs(contact['total_energy_error_joule']) < 1e-10 * contact['initial_total_energy_joule']

    entry = read('hyperelastic_final_entry/summary.json')
    restart = read('hyperelastic_releasing_precise_entry/summary.json')
    old_source = read('source_unchanged_check.json')
    assert entry['passed'] and restart['passed']
    assert not old_source['source_working_file_hash_changes']
    checks = {name: read(name + '.json') for name in [
        'hyperelastic_material_check', 'hyperelastic_material_extended',
        'hyperelastic_gpu_check', 'hyperelastic_objectivity_check',
        'hyperelastic_regression_check', 'hyperelastic_planar_mechanics_check']}
    result = {
        'passed': True,
        'scope': 'Completed full-foam hyperelastic release, frictionless impact/separation, main entry and restart; not spatial convergence or all-material certification.',
        'material_full_runs': materials,
        'contact': contact_summary, 'independent_contact_gap': gap,
        'fresh_main_entry_and_default_regression': entry,
        'moving_boundary_restart': restart,
        'material_operator_and_old_model_checks': checks,
        'protected_source_check': old_source,
        'evidence_sha256': sources,
        'limits': [
            'MPM is the recorded PowerSim/PhysGaussian implementation with the same elastic energy, not a converged truth.',
            'Numerical mechanical-energy loss remains substantial in both methods.',
            'Near-incompressibility still requires small explicit steps; recorded timings include diagnostics.',
            'The moving contact/release boundary currently supports a fixed frictionless plane only.',
            'Hyperelastic and boundary changes are opt-in; old Wilkins defaults are retained.',
        ],
    }
    path = root / 'hyperelastic_acceptance.json'
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'passed': True, 'report': str(path), 'contact_time_s': contact['time_reached'],
                      'gap_lower_bound_m': gap['resolved_gap_lower_bound_m'],
                      'material_runs': len(materials)}, indent=2))


if __name__ == '__main__':
    main()
