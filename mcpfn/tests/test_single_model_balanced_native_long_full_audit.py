import types

import test_single_model_balanced_native_quarter_full_audit as parent_tests
from recover_single_model_stronger_mean_cuda_entry import unresolved_globals


TEST_SOURCE = parent_tests.TEST_SOURCE
for old, new in {
    'audit_single_model_balanced_native_quarter': 'audit_single_model_balanced_native_long',
    'balanced_native_quarter_v55': 'balanced_native_long_v56',
}.items():
    if old not in TEST_SOURCE:
        raise ValueError(f'Missing long-budget audit test adaptation: {old}')
    TEST_SOURCE = TEST_SOURCE.replace(old, new)
exec(compile(TEST_SOURCE, __file__, 'exec'), globals())
del test_actual_full_audit_installs_component_replay_without_changing_fixed_half_or_retention


def test_full_audit_uses_every_extended_attempt_and_explicit_quarter_protection():
    assert audit.training.STEPS == 256
    assert audit.retention_training is audit.training.retention_training
    assert audit.blend_replay is audit.training.blend_replay
    assert audit.retention_training.expected_component_counts(256) == {"v8": 6144, "v29": 6656}
    assert audit.checks is audit.training.paired is audit.fixed_trial.protection
    assert audit.checks.FRACTION == .25
    assert 'audit(output, training.STEPS)' in audit.RUN_AUDIT_SOURCE
    assert '"fixed_quarter_training": True' in audit.ADAPTED_SOURCE
    assert 'validate_saved_components' in audit.retention_training.RetentionReplay.check.__code__.co_names
    for value in vars(audit).values():
        if isinstance(value, types.FunctionType) and value.__globals__ is vars(audit):
            assert not unresolved_globals(value), value.__name__
