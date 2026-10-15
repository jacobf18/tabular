"""Qualify V53 with all original synthetic guards and frozen real suites."""

import run_single_model_all_imputation_column_mean_pipeline as pipeline_blueprint
import single_model_all_imputation_blend_full_replay as blend_replay
import single_model_fixed_half_trial as fixed_trial
import single_model_balanced_structured_trial as cohort_trial
import recover_single_model_legacy_blend_blas_stable as arithmetic_recovery
from single_model_registered_adaptation import replace_once


ARITHMETIC_PATCH = pipeline_blueprint.ARITHMETIC_PATCH
PIPELINE_ADAPTATION_SOURCE = pipeline_blueprint.PIPELINE_ADAPTATION_SOURCE
for old, new in {
    'train_single_matrixformer_all_imputation_column_mean': 'train_single_matrixformer_balanced_structured',
    'audit_single_model_all_imputation_column_mean': 'audit_single_model_balanced_structured',
    'single_model_all_imputation_column_mean_storage': 'single_model_balanced_structured_storage',
    'all_imputation_column_mean_v52': 'balanced_structured_components_v53',
    'test_single_model_all_imputation_column_mean_trial.py': 'test_single_model_balanced_structured_trial.py',
    'test_single_model_all_imputation_column_mean_pipeline.py': 'test_single_model_balanced_structured_pipeline.py',
    'V52': 'V53',
}.items():
    if old not in PIPELINE_ADAPTATION_SOURCE:
        raise ValueError(f'Missing explicit balanced-prior pipeline adaptation: {old}')
    PIPELINE_ADAPTATION_SOURCE = PIPELINE_ADAPTATION_SOURCE.replace(old, new)
exec(compile(PIPELINE_ADAPTATION_SOURCE, __file__, 'exec'), globals())
