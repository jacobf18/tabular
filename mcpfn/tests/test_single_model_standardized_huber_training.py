import test_single_model_all_imputation_column_mean_training as parent_tests
from single_model_registered_adaptation import replace_once


TEST_SOURCE = parent_tests.TEST_SOURCE
for old, new in {
    'single_model_all_imputation_column_mean_training as integration': 'single_model_standardized_huber_training as integration',
    'preflight_single_model_all_imputation_column_mean as probe': 'preflight_single_model_standardized_teacher_huber as probe',
}.items():
    if old not in TEST_SOURCE:
        raise ValueError(f'Missing standardized-Huber replay test adaptation: {old}')
    TEST_SOURCE = TEST_SOURCE.replace(old, new)
TEST_SOURCE = replace_once(TEST_SOURCE,
    'native_teacher_mean=integration.retention.MEAN_SPECIFICATION,',
    'native_teacher_mean=integration.retention.MEAN_SPECIFICATION,\n'
    '        retention_backward_sha256=integration.retention.BACKWARD_SOURCE_SHA256,')
TEST_SOURCE = replace_once(TEST_SOURCE,
    'manifest.update(policy=integration.retention.POLICY, specification=integration.retention.SPECIFICATION,',
    'manifest.update(policy=integration.retention.POLICY, specification=integration.retention.SPECIFICATION,\n'
    '        backward_source_sha256=integration.retention.BACKWARD_SOURCE_SHA256,')
exec(compile(TEST_SOURCE, __file__, 'exec'), globals())


def test_standardized_geometry_and_original_reference_counts_are_bound():
    assert integration.retention is probe.retention
    assert integration.expected_component_counts(0) == {}
    assert integration.expected_component_counts(1) == {'v8': 24, 'v29': 26}
    assert integration.expected_component_counts(64) == {'v8': 1536, 'v29': 1664}
    assert 'native_mean_scales[lookup]' in integration.retention.BACKWARD_SOURCE
    assert integration.INITIALIZER == parent_tests.integration.INITIALIZER
    assert integration.EVALUATION_BASELINE == parent_tests.integration.EVALUATION_BASELINE


def test_replay_rejects_pre_standardization_backward_identity(tmp_path, monkeypatch):
    training, config, counts, references, backend, calls = fixture(tmp_path, monkeypatch)
    config['retention_backward_sha256'] = integration.predecessor.retention.BACKWARD_SOURCE_SHA256
    with pytest.raises(ValueError, match='checkpoint identities'):
        integration.RetentionReplay(training, config, counts, references, None, backend)
    assert not calls
