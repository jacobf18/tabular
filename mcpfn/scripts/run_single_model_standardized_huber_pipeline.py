"""Qualify V54 with every original synthetic guard and frozen real suite."""

import run_single_model_balanced_structured_pipeline as pipeline_blueprint
import single_model_all_imputation_blend_full_replay as blend_replay
import single_model_fixed_half_trial as fixed_trial
import single_model_standardized_huber_trial as cohort_trial
import recover_single_model_legacy_blend_blas_stable as arithmetic_recovery
from single_model_registered_adaptation import replace_once


ARITHMETIC_PATCH = pipeline_blueprint.ARITHMETIC_PATCH
PIPELINE_ADAPTATION_SOURCE = pipeline_blueprint.PIPELINE_ADAPTATION_SOURCE
for old, new in {
    'train_single_matrixformer_balanced_structured': 'train_single_matrixformer_standardized_huber',
    'audit_single_model_balanced_structured': 'audit_single_model_standardized_huber',
    'single_model_all_imputation_column_mean_training': 'single_model_standardized_huber_training',
    'single_model_balanced_structured_storage': 'single_model_standardized_huber_storage',
    'balanced_structured_components_v53': 'standardized_teacher_huber_v54',
    'test_single_model_balanced_structured_trial.py': 'test_single_model_standardized_huber_trial.py',
    'test_single_model_balanced_structured_pipeline.py': 'test_single_model_standardized_huber_pipeline.py',
    'V53': 'V54',
}.items():
    if old not in PIPELINE_ADAPTATION_SOURCE:
        raise ValueError(f'Missing standardized-Huber coordinator adaptation: {old}')
    PIPELINE_ADAPTATION_SOURCE = PIPELINE_ADAPTATION_SOURCE.replace(old, new)
exec(compile(PIPELINE_ADAPTATION_SOURCE, __file__, 'exec'), globals())
