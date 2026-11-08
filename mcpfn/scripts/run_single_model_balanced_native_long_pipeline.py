"""Qualify V56 under all frozen synthetic guards and the complete real benchmark suite."""

import run_single_model_balanced_native_quarter_pipeline as pipeline_blueprint
import single_model_balanced_native_long_blend_replay as blend_replay
import single_model_fixed_quarter_trial as fixed_trial
import single_model_balanced_native_long_trial as cohort_trial
import recover_single_model_legacy_blend_blas_stable as arithmetic_recovery
from single_model_registered_adaptation import replace_once


ARITHMETIC_PATCH = pipeline_blueprint.ARITHMETIC_PATCH
PIPELINE_ADAPTATION_SOURCE = pipeline_blueprint.PIPELINE_ADAPTATION_SOURCE
for old, new in {
    'train_single_matrixformer_balanced_native_quarter': 'train_single_matrixformer_balanced_native_long',
    'audit_single_model_balanced_native_quarter': 'audit_single_model_balanced_native_long',
    'single_model_standardized_huber_training': 'single_model_balanced_native_long_retention',
    'single_model_balanced_native_quarter_storage': 'single_model_balanced_native_long_storage',
    'balanced_native_quarter_v55': 'balanced_native_long_v56',
    'test_single_model_balanced_native_quarter_trial.py': 'test_single_model_balanced_native_long_trial.py',
    'test_single_model_balanced_native_quarter_pipeline.py': 'test_single_model_balanced_native_long_pipeline.py',
    'V55': 'V56',
}.items():
    if old not in PIPELINE_ADAPTATION_SOURCE:
        raise ValueError(f'Missing long-budget pipeline adaptation: {old}')
    PIPELINE_ADAPTATION_SOURCE = PIPELINE_ADAPTATION_SOURCE.replace(old, new)
PIPELINE_ADAPTATION_SOURCE = replace_once(PIPELINE_ADAPTATION_SOURCE,
    'SOURCE = cohort_trial.wrap_cli(SOURCE)',
    'SOURCE = cohort_trial.wrap_cli(SOURCE)\nSOURCE = cohort_trial.adapt_pipeline_recipe(SOURCE)')
exec(compile(PIPELINE_ADAPTATION_SOURCE, __file__, 'exec'), globals())
