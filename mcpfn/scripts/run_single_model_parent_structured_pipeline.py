"""Qualify V49 with all original synthetic guards and frozen real suites."""

import run_single_model_native_column_mean_pipeline as pipeline_blueprint
import single_model_all_imputation_blend_full_replay as blend_replay
import single_model_fixed_half_trial as fixed_trial
import single_model_parent_structured_trial as cohort_trial
import recover_single_model_legacy_blend_blas_stable as arithmetic_recovery
from single_model_registered_adaptation import replace_once


ARITHMETIC_PATCH = pipeline_blueprint.ARITHMETIC_PATCH
PIPELINE_ADAPTATION_SOURCE = pipeline_blueprint.PIPELINE_ADAPTATION_SOURCE
for old, new in {
    "train_single_matrixformer_native_column_mean": "train_single_matrixformer_parent_structured",
    "audit_single_model_native_column_mean": "audit_single_model_parent_structured",
    "single_model_native_column_mean_training": "single_model_parent_structured_training",
    "native_column_mean_v48": "parent_structured_v49",
    "test_single_model_native_column_trial.py": "test_single_model_parent_structured_trial.py",
    "test_single_model_native_column_pipeline.py": "test_single_model_parent_structured_pipeline.py",
    "V48": "V49",
}.items():
    if old not in PIPELINE_ADAPTATION_SOURCE:
        raise ValueError(f"Missing explicit parent-structured pipeline adaptation: {old}")
    PIPELINE_ADAPTATION_SOURCE = PIPELINE_ADAPTATION_SOURCE.replace(old, new)
exec(compile(PIPELINE_ADAPTATION_SOURCE, __file__, "exec"), globals())
