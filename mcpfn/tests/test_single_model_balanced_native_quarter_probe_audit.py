import test_single_model_balanced_native_coverage_probe_audit as blueprint


SOURCE = blueprint.SOURCE
for old, new in {
    'import audit_single_model_balanced_native_coverage_probe as audit':
        'import audit_single_model_balanced_native_quarter_probe as audit',
    'balanced_native_coverage_preflight_v1_mean_audit': 'balanced_native_quarter_preflight_v1_mean_audit',
    'test_single_model_balanced_native_coverage_probe_audit.py':
        'test_single_model_balanced_native_quarter_probe_audit.py',
}.items():
    if old not in SOURCE:
        raise ValueError(f'Missing explicit quarter audit-test adaptation: {old}')
    SOURCE = SOURCE.replace(old, new)
exec(compile(SOURCE, __file__, 'exec'), globals())


def test_mean_audit_binds_new_probe_but_preserves_half_retention_weights():
    assert audit.probe.paired.FRACTION == .25
    assert audit.probe.reference_identities() == audit.probe.probe_blueprint.reference_identities()
