import hashlib
import inspect
import types

import test_single_model_balanced_native_quarter_pipeline as parent_tests
import test_single_model_full_network_pipeline as admission_tests
import test_single_model_legacy_blend_teacher_pipeline as qualification_tests
from recover_single_model_stronger_mean_cuda_entry import unresolved_globals


TEST_SOURCE = parent_tests.TEST_SOURCE
for old, new in {
    'run_single_model_balanced_native_quarter_pipeline': 'run_single_model_balanced_native_long_pipeline',
    'audit_single_model_balanced_native_quarter': 'audit_single_model_balanced_native_long',
    'train_single_matrixformer_balanced_native_quarter': 'train_single_matrixformer_balanced_native_long',
    'balanced_native_quarter_v55': 'balanced_native_long_v56',
}.items():
    if old not in TEST_SOURCE:
        raise ValueError(f'Missing long-budget pipeline test adaptation: {old}')
    TEST_SOURCE = TEST_SOURCE.replace(old, new)
exec(compile(TEST_SOURCE, __file__, 'exec'), globals())

for name in ("test_actual_prerequisite_requires_all_original_and_added_evidence",
        "test_frozen_confirmation_campaign_and_ordinary_model_scope_remain_unchanged"):
    adapted = inspect.getsource(getattr(qualification_tests, name))
    adapted = adapted.replace('"through_step": 64', '"through_step": 256')
    adapted = adapted.replace('"attempted_batches": 64', '"attempted_batches": 256')
    adapted = adapted.replace('1792', '7168').replace('1368', '1824')
    adapted = adapted.replace('(0, 10, 25, 50, 64)', '(0, 10, 25, 50, 64, 128, 192, 256)')
    exec(compile(adapted, __file__, 'exec'), globals())
ADMISSION_TEST_SOURCE = inspect.getsource(admission_tests.test_independent_admission_counts_and_full_optimizer_scope_are_mandatory)
for old, new in {
    '"scheduled_attempts": 64': '"scheduled_attempts": 256',
    '"accepted_updates": 60': '"accepted_updates": 252',
    '1736': '7112', '124': '508', '1536': '6144', '1792': '7168', '128': '512', '24 * 62': '24 * 254',
    '"top_block_training": True': '"top_block_training": True, "top_two_block_training": True',
}.items():
    if old not in ADMISSION_TEST_SOURCE:
        raise ValueError(f'Missing extended admission test adaptation: {old}')
    ADMISSION_TEST_SOURCE = ADMISSION_TEST_SOURCE.replace(old, new)
exec(compile(ADMISSION_TEST_SOURCE, __file__, 'exec'), globals())


def test_pipeline_requires_all_extended_attempts_and_prediction_grids():
    assert 'audit["through_step"] != training.STEPS' in pipeline.ADAPTED_SOURCE
    assert 'audit["attempted_batches"] != training.STEPS' in pipeline.ADAPTED_SOURCE
    assert 'audit["prediction_grids"] != (4 + len(training.BOUNDARIES)) * 152' in pipeline.ADAPTED_SOURCE
    assert "256 attempts and1824" in pipeline.POLICY
    assert 'audit["prediction_grids"] != 1368' not in pipeline.ADAPTED_SOURCE
    assert pipeline.ADAPTED_SOURCE_SHA256 == hashlib.sha256(pipeline.ADAPTED_SOURCE.encode()).hexdigest()
    assert pipeline.retention_training is pipeline.training_audit.retention_training is pipeline.training.retention_training
    assert pipeline.blend_replay is pipeline.training.blend_replay
    assert pipeline.training.diagnostics is pipeline.cohort_trial.component_storage
    assert pipeline.training_audit.training is pipeline.training
    assert pipeline.fixed_trial.protection is pipeline.training.paired
    for value in vars(pipeline).values():
        if isinstance(value, types.FunctionType) and value.__globals__ is vars(pipeline):
            assert not unresolved_globals(value), value.__name__
