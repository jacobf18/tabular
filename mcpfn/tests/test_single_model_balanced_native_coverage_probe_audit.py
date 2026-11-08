from pathlib import Path

import test_single_model_standardized_teacher_huber_probe_audit as blueprint


SOURCE = Path(blueprint.__file__).read_text()
for old, new in {
    'import audit_single_model_standardized_teacher_huber_probe as audit':
        'import audit_single_model_balanced_native_coverage_probe as audit',
    'standardized_teacher_huber_preflight_v1_mean_audit': 'balanced_native_coverage_preflight_v1_mean_audit',
    'test_single_model_standardized_teacher_huber_probe_audit.py':
        'test_single_model_balanced_native_coverage_probe_audit.py',
}.items():
    if old not in SOURCE:
        raise ValueError(f'Missing explicit audit-test adaptation: {old}')
    SOURCE = SOURCE.replace(old, new)
exec(compile(SOURCE, __file__, 'exec'), globals())
